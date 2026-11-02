import llm_profiler as L


def test_diagnose_names_a_missing_package_or_key(monkeypatch):
    monkeypatch.setattr(L, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(L, "GROQ_AVAILABLE", False)
    assert "pip install groq" in L.diagnose()
    monkeypatch.setattr(L, "GROQ_AVAILABLE", True)
    monkeypatch.setattr(L, "GROQ_API_KEY", None)
    assert "GROQ_API_KEY" in L.diagnose()
    monkeypatch.setattr(L, "GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr(L, "LAST_ERROR", "")
    assert L.diagnose() == ""


def test_a_failed_call_is_remembered_without_the_key(monkeypatch):
    class Boom:
        def __init__(self, **kw):
            pass

        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    raise ValueError("bad request for key gsk_SECRET123abc")

    monkeypatch.setattr(L, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(L, "RETRY_WAITS", [])
    monkeypatch.setattr(L, "GROQ_AVAILABLE", True)
    monkeypatch.setattr(L, "GROQ_API_KEY", "gsk_SECRET123abc")
    monkeypatch.setattr(L, "GroqClient", Boom, raising=False)
    try:
        L.chat_completion([{"role": "user", "content": "x"}])
        assert False
    except RuntimeError:
        pass
    assert "ValueError" in L.LAST_ERROR and "SECRET123abc" not in L.LAST_ERROR and "gsk_***" in L.LAST_ERROR
    assert L.diagnose() == L.LAST_ERROR


def test_extract_profile_raises_a_clear_error_without_a_key(monkeypatch):
    monkeypatch.setattr(L, "GROQ_API_KEY", None)
    monkeypatch.setattr(L, "GEMINI_API_KEY", None)
    try:
        L.extract_profile("safe growth")
        assert False
    except RuntimeError as e:
        assert "unavailable" in str(e)


def test_eval_script_stops_with_the_reason_when_setup_is_wrong(monkeypatch):
    import pytest
    import eval_profiler as EP
    monkeypatch.setattr(L, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(L, "GROQ_AVAILABLE", False)
    with pytest.raises(SystemExit) as e:
        EP.main()
    assert "pip install groq" in str(e.value)


class FakeClient:
    """Stands in for the Groq client: records the calls, fails with 404 for names in `missing`."""
    calls = []
    missing = set()

    def __init__(self, **kw):
        pass

    class _Msg:
        def __init__(self, content):
            self.message = type("M", (), {"content": content})()

    @property
    def chat(self):
        outer = self

        class completions:
            @staticmethod
            def create(**kw):
                FakeClient.calls.append(kw)
                if kw["model"] in FakeClient.missing:
                    err = Exception("Error code: 404 - model_not_found")
                    err.status_code = 404
                    raise err
                return type("R", (), {"choices": [FakeClient._Msg(FakeClient.reply)]})()

        return type("C", (), {"completions": completions})()

    reply = "hello"


def use_fake(monkeypatch, models, missing=(), reply="hello"):
    FakeClient.calls, FakeClient.missing, FakeClient.reply = [], set(missing), reply
    monkeypatch.setattr(L, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(L, "RETRY_WAITS", [])
    monkeypatch.setattr(L, "GROQ_AVAILABLE", True)
    monkeypatch.setattr(L, "GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr(L, "GroqClient", FakeClient, raising=False)
    monkeypatch.setattr(L, "_models", list(models))


def test_a_model_groq_says_is_missing_is_skipped(monkeypatch):
    use_fake(monkeypatch, ["old-model", "openai/gpt-oss-20b"], missing=["old-model"])
    assert L.chat_completion([{"role": "user", "content": "x"}]) == "hello"
    assert [c["model"] for c in FakeClient.calls] == ["old-model", "openai/gpt-oss-20b"]
    assert L.active_model() == "openai/gpt-oss-20b"
    L.chat_completion([{"role": "user", "content": "x"}])
    assert FakeClient.calls[-1]["model"] == "openai/gpt-oss-20b" and len(FakeClient.calls) == 3


def test_the_only_model_missing_is_an_error_with_the_reason(monkeypatch):
    use_fake(monkeypatch, ["only-model"], missing=["only-model"])
    try:
        L.chat_completion([{"role": "user", "content": "x"}])
        assert False
    except RuntimeError:
        pass
    assert "model_not_found" in L.LAST_ERROR


def test_reasoning_models_get_low_effort_and_extra_token_room_but_others_do_not(monkeypatch):
    use_fake(monkeypatch, ["openai/gpt-oss-20b"])
    L.chat_completion([{"role": "user", "content": "x"}], max_tokens=5)
    kw = FakeClient.calls[0]
    assert kw["extra_body"] == {"reasoning_effort": "low"} and kw["max_tokens"] == 5 + L.REASONING_BUDGET
    use_fake(monkeypatch, ["some-llama"])
    L.chat_completion([{"role": "user", "content": "x"}], max_tokens=5)
    assert "extra_body" not in FakeClient.calls[0] and FakeClient.calls[0]["max_tokens"] == 5


def test_empty_model_reply_does_not_crash(monkeypatch):
    use_fake(monkeypatch, ["openai/gpt-oss-20b"], reply=None)
    assert L.chat_completion([{"role": "user", "content": "x"}]) == ""


def test_profile_extraction_parses_json_from_the_model(monkeypatch):
    use_fake(monkeypatch, ["openai/gpt-oss-20b"],
             reply='```json\n{"risk_level": "low", "horizon": "long", "is_intent": true, "preferred_sectors": ["Energy"]}\n```')
    p = L.extract_profile("safe long term energy")
    assert p.risk_level == "low" and p.horizon == "long" and p.is_intent and p.preferred_sectors == ["Energy"]
    assert FakeClient.calls[0]["temperature"] == 0.05


# Gemini provider and retries
class FakeReply:
    def __init__(self, status, text="", body=None):
        self.status_code, self.text, self._body = status, text, body

    def json(self):
        return {"candidates": [{"content": {"parts": [{"text": self._body}]}}]}


def use_gemini(monkeypatch, replies, waits=(2, 4)):
    sent, slept = [], []
    monkeypatch.setattr(L, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(L, "GEMINI_API_KEY", "AIza-test-key")
    monkeypatch.setattr(L, "RETRY_WAITS", list(waits))
    monkeypatch.setattr(L.time, "sleep", slept.append)
    monkeypatch.setattr(L, "requests", type("R", (), {"post": staticmethod(lambda url, **kw: (sent.append((url, kw)), replies.pop(0))[1])}))
    return sent, slept


def test_provider_choice(monkeypatch):
    monkeypatch.setattr(L, "LLM_PROVIDER", "")
    monkeypatch.setattr(L, "GROQ_API_KEY", "gsk_x")
    monkeypatch.setattr(L, "GEMINI_API_KEY", "AIza")
    assert L.provider() == "groq"                       # an existing Groq setup keeps working
    monkeypatch.setattr(L, "GROQ_API_KEY", None)
    assert L.provider() == "gemini"
    monkeypatch.setattr(L, "LLM_PROVIDER", "groq")
    assert L.provider() == "groq"


def test_gemini_request_shape_and_reply(monkeypatch):
    sent, _ = use_gemini(monkeypatch, [FakeReply(200, body="  YES ")])
    out = L.chat_completion([{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"},
                             {"role": "assistant", "content": "hello"}, {"role": "user", "content": "q"}], max_tokens=5, temperature=0)
    url, kw = sent[0]
    assert out == "YES" and L.GEMINI_MODEL in url and kw["headers"]["x-goog-api-key"] == "AIza-test-key"
    assert "AIza" not in url and kw["json"]["systemInstruction"]["parts"][0]["text"] == "be brief"
    assert [c["role"] for c in kw["json"]["contents"]] == ["user", "model", "user"]
    assert kw["json"]["generationConfig"] == {"temperature": 0, "maxOutputTokens": 5 + L.REASONING_BUDGET}


def test_rate_limit_is_retried_after_a_wait_then_succeeds(monkeypatch):
    sent, slept = use_gemini(monkeypatch, [FakeReply(429, "quota"), FakeReply(503, "busy"), FakeReply(200, body="ok")])
    assert L.chat_completion([{"role": "user", "content": "x"}]) == "ok" and slept == [2, 4] and len(sent) == 3


def test_rate_limit_that_never_clears_is_an_error_with_the_reason(monkeypatch):
    sent, slept = use_gemini(monkeypatch, [FakeReply(429, "RESOURCE_EXHAUSTED quota")] * 5)
    try:
        L.chat_completion([{"role": "user", "content": "x"}])
        assert False
    except RuntimeError:
        pass
    assert len(sent) == 3 and "429" in L.LAST_ERROR and "RESOURCE_EXHAUSTED" in L.LAST_ERROR and "AIza" not in L.LAST_ERROR


def test_bad_key_is_not_retried(monkeypatch):
    sent, slept = use_gemini(monkeypatch, [FakeReply(403, "API key not valid")] * 3)
    try:
        L.chat_completion([{"role": "user", "content": "x"}])
        assert False
    except RuntimeError:
        pass
    assert len(sent) == 1 and slept == []


def test_gemini_profile_extraction_and_diagnosis(monkeypatch):
    use_gemini(monkeypatch, [FakeReply(200, body='{"risk_level": "high", "horizon": "short", "is_intent": true}')])
    p = L.extract_profile("aggressive short term")
    assert p.risk_level == "high" and p.horizon == "short" and p.is_intent
    monkeypatch.setattr(L, "GEMINI_API_KEY", None)
    assert "GEMINI_API_KEY" in L.diagnose()
