"""
llm_profiler.py

Uses the Groq API with the Groq-hosted model to parse a user's natural-language investment goal into a structured
UserProfile. Whatever the model returns is checked by sanitize_profile before anything else uses it, so a bad
value (an unknown risk level, a negative fee limit, an invented sector) falls back to a safe default.

The key comes from the GROQ_API_KEY environment variable (or a local .env file, which git ignores).
There is no rule-based fallback: without the key or the service, extract_profile raises RuntimeError.
"""

import json
import re
from dataclasses import dataclass, asdict
import os
import time

from dotenv import load_dotenv
load_dotenv()

LAST_ERROR = ""        # why the most recent Groq call failed (key removed), for diagnosis
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
LLM_PROVIDER = (os.getenv("LLM_PROVIDER") or "").strip().lower()    # "groq" or "gemini"; empty = whichever key is set
RETRY_WAITS  = [2]       # seconds to wait before each retry after a rate limit or a busy server (empty = no retries)
RETRY_STATUS = {429, 500, 502, 503, 504}
# llama-3.1-8b-instant is no longer available to free keys (404 model_not_found). These are tried in order, and a
# model Groq reports as missing is skipped. Set GROQ_MODEL to use one specific model instead.
DEFAULT_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
_models      = [os.getenv("GROQ_MODEL")] if os.getenv("GROQ_MODEL") else list(DEFAULT_MODELS)
REASONING_BUDGET = 700   # gpt-oss models think before answering, and that thinking counts toward max_tokens
GROQ_TIMEOUT = 20      # seconds
MAX_GOAL_CHARS = 500

try:
    from groq import Groq as GroqClient
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    import requests
except ImportError:
    requests = None


class LLMHTTPError(Exception):
    """A failed Gemini request: the status code plus a short piece of Google's message (never the key)."""

    def __init__(self, status_code, text):
        super().__init__(f"HTTP {status_code}: {text[:200]}")
        self.status_code = status_code


def provider() -> str:
    """'gemini' or 'groq'. LLM_PROVIDER wins; otherwise Groq if its key is set (so existing setups keep working), else Gemini."""
    if LLM_PROVIDER in ("groq", "gemini"):
        return LLM_PROVIDER
    if GROQ_API_KEY:
        return "groq"
    return "gemini" if GEMINI_API_KEY else "groq"


def _ready() -> bool:
    if provider() == "gemini":
        return bool(GEMINI_API_KEY) and requests is not None
    return bool(GROQ_AVAILABLE and GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here")


def active_model() -> str:
    return GEMINI_MODEL if provider() == "gemini" else _models[0]


def _is_missing_model(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) == 404 or "model_not_found" in str(exc)


def _groq_create(messages: list, max_tokens: int, temperature: float) -> str:
    """One Groq chat call. A model Groq says does not exist is dropped and the next one is tried."""
    client = GroqClient(api_key=GROQ_API_KEY, timeout=GROQ_TIMEOUT, max_retries=1)
    while True:
        model = _models[0]
        kwargs = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        if "gpt-oss" in model:
            kwargs["max_tokens"] = max_tokens + REASONING_BUDGET
            kwargs["extra_body"] = {"reasoning_effort": "low"}
        try:
            resp = client.chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            if _is_missing_model(e) and len(_models) > 1:
                _models.pop(0)
                continue
            raise


def _gemini_create(messages: list, max_tokens: int, temperature: float) -> str:
    """One Gemini call over REST. System messages become the system instruction; 'assistant' turns become 'model'."""
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                for m in messages if m["role"] != "system"]
    body = {"contents": contents,
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens + REASONING_BUDGET}}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
                      headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
                      json=body, timeout=GROQ_TIMEOUT)
    if r.status_code >= 400:
        raise LLMHTTPError(r.status_code, r.text)
    parts = r.json()["candidates"][0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts).strip()


def _create(messages: list, max_tokens: int, temperature: float) -> str:
    """One model call with the chosen provider, retried after RETRY_WAITS when it is rate-limited or busy."""
    call = _gemini_create if provider() == "gemini" else _groq_create
    waits = list(RETRY_WAITS)
    while True:
        try:
            return call(messages, max_tokens, temperature)
        except Exception as e:
            status = getattr(e, "status_code", None)
            if status in RETRY_STATUS and waits:
                time.sleep(waits.pop(0))
                continue
            raise


def _remember(exc: Exception):
    """Keep a short, key-free description of a failed call."""
    global LAST_ERROR
    LAST_ERROR = re.sub(r"gsk_[A-Za-z0-9_\-]+", "gsk_***", f"{type(exc).__name__}: {exc}")[:300]


def diagnose() -> str:
    """Empty string if the model looks usable, else the reason it is not (missing package, missing key, last error)."""
    if provider() == "gemini":
        if requests is None:
            return "the 'requests' package is not installed in this Python (pip install requests)"
        if not GEMINI_API_KEY:
            return "GEMINI_API_KEY is not set in this shell or in a .env file"
        return LAST_ERROR
    if not GROQ_AVAILABLE:
        return "the 'groq' package is not installed in this Python (pip install groq)"
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        return "GROQ_API_KEY (or GEMINI_API_KEY) is not set in this shell or in a .env file"
    return LAST_ERROR


# ── Allowed values ───────────────────────────────────────────────────────────
RISK_LEVELS   = ("low", "medium", "high")
HORIZONS      = ("short", "medium", "long")
DISTRIBUTIONS = ("accumulating", "distributing")
CURRENCIES    = ("EUR", "USD", "any")
SECTORS = ["Technology", "Consumer Staples", "Industrials", "Consumer Discretionary", "Financials",
           "Basic Materials", "Real Estate", "Utilities", "Energy", "Health Care", "Telecommunication", "Other"]
REGIONS = ["United States", "United Kingdom", "Germany", "Japan", "France", "China", "Canada", "Switzerland",
           "Australia", "Other"]
MAX_TER_RANGE = (0.0005, 0.02)
MAX_FUND_SIZE = 10000.0


# ── UserProfile dataclass ────────────────────────────────────────────────────
@dataclass
class UserProfile:
    risk_level:         str    # "low" | "medium" | "high"
    horizon:            str    # "short" | "medium" | "long"
    distribution:       str    # "accumulating" | "distributing"
    max_ter:            float  # annual expense ratio, e.g. 0.002 = 0.20%
    preferred_sectors:  list   # e.g. ["Technology", "Health Care"]
    preferred_regions:  list   # e.g. ["United States", "Germany"]
    min_fund_size:      float  # in millions, e.g. 500.0
    currency:           str    # "EUR" | "USD" | "any"
    raw_goal:           str    # original user text
    is_intent:          bool   # True when the input expresses an investment intent


def _number(value, default, low, high):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    if x != x:                      # NaN
        return default
    return min(max(x, low), high)


def _choice(value, allowed, default):
    v = str(value).strip().lower() if value is not None else ""
    for a in allowed:
        if v == a.lower():
            return a
    return default


def _pick(values, allowed, limit=3):
    out = []
    for v in values if isinstance(values, list) else []:
        a = _choice(v, allowed, None)
        if a and a not in out:
            out.append(a)
    return out[:limit]


def sanitize_profile(d: dict, goal: str = "") -> UserProfile:
    """Turn whatever the model (or a client) sent into a valid UserProfile. Unknown or out-of-range values get defaults."""
    d = d if isinstance(d, dict) else {}
    intent = d.get("is_intent", False)
    if isinstance(intent, str):
        intent = intent.strip().lower() == "true"
    return UserProfile(
        risk_level        = _choice(d.get("risk_level"), RISK_LEVELS, "medium"),
        horizon           = _choice(d.get("horizon"), HORIZONS, "medium"),
        distribution      = _choice(d.get("distribution"), DISTRIBUTIONS, "accumulating"),
        max_ter           = _number(d.get("max_ter"), 0.005, *MAX_TER_RANGE),
        preferred_sectors = _pick(d.get("preferred_sectors"), SECTORS),
        preferred_regions = _pick(d.get("preferred_regions"), REGIONS),
        min_fund_size     = _number(d.get("min_fund_size"), 100.0, 0.0, MAX_FUND_SIZE),
        currency          = _choice(d.get("currency"), CURRENCIES, "any"),
        raw_goal          = str(goal)[:MAX_GOAL_CHARS],
        is_intent         = intent is True,
    )


# ── System prompt sent to the model ───────────────────────────────────────────
_SYSTEM_PROMPT = """You are a financial profile extraction assistant.
Given a user's investment goal in natural language, return ONLY a valid JSON object
with exactly these keys — no markdown, no explanation, no extra text:

{
  "risk_level": "low" | "medium" | "high",
  "horizon": "short" | "medium" | "long",
  "distribution": "accumulating" | "distributing",
  "max_ter": <float: annual TER limit, e.g. 0.002 means 0.20%>,
  "preferred_sectors": [list from: Technology, Consumer Staples, Industrials,
    Consumer Discretionary, Financials, Basic Materials, Real Estate, Utilities,
    Energy, Health Care, Telecommunication, Other],
  "preferred_regions": [list from: United States, United Kingdom, Germany, Japan,
    France, China, Canada, Switzerland, Australia, Other],
  "min_fund_size": <float in millions, 0 if no preference>,
  "currency": "EUR" | "USD" | "any"
}

Inference rules:
- "safe", "conservative", "low risk", "stable", "protect capital" → risk_level: "low"
- "balanced", "moderate", "medium risk" → risk_level: "medium"
- "aggressive", "growth", "high return", "maximize" → risk_level: "high"
- "short term", "1-2 year", "near future" → horizon: "short"
- "medium term", "3-7 year" → horizon: "medium"
- "long term", "retirement", "decade", "10+ year" → horizon: "long"
- "no dividends", "reinvest", "tax efficient", "accumulate" → distribution: "accumulating"
- "income", "dividend", "cash flow", "payout", "yield" → distribution: "distributing"
- "low cost", "cheap", "low fees", "minimal cost" → max_ter: 0.002
- "very cheap", "under 0.1%" → max_ter: 0.001
- default max_ter: 0.005
- "large fund", "liquid", "established" → min_fund_size: 1000
- default min_fund_size: 100
- No sector mentioned → preferred_sectors: []
- No region mentioned → preferred_regions: []

If the input is unrelated to investing or is casual text (e.g. "Hello, how are you?"),
    return { "is_intent": false } along with the keys above set to sensible defaults.
    Otherwise set "is_intent": true.

Examples (required format):

1) Investment intent example
Input: "I want safe long-term growth with low fees and no dividends; prefer large European funds"
Output:
{
    "risk_level": "low",
    "horizon": "long",
    "distribution": "accumulating",
    "max_ter": 0.002,
    "preferred_sectors": [],
    "preferred_regions": ["Germany"],
    "min_fund_size": 1000.0,
    "currency": "EUR",
    "is_intent": true
}

2) Non-investment example (casual greeting)
Input: "Hello, how are you?"
Output:
{
    "risk_level": "medium",
    "horizon": "medium",
    "distribution": "accumulating",
    "max_ter": 0.005,
    "preferred_sectors": [],
    "preferred_regions": [],
    "min_fund_size": 100.0,
    "currency": "any",
    "is_intent": false
}
"""


# ── Groq API call ─────────────────────────────────────────────────────────────
def _call_groq(goal: str) -> dict | None:
    """Call the model to extract a profile. Returns the parsed dict or None on failure."""
    if not _ready():
        return None

    try:
        raw = _create(
            [{"role": "system", "content": _SYSTEM_PROMPT},
             {"role": "user",   "content": f"Investment goal: {goal[:MAX_GOAL_CHARS]}"}],
            max_tokens=512, temperature=0.05,
        )
        # Strip accidental markdown fences
        raw = re.sub(r"```json|```", "", raw).strip()
        # Extract first JSON object found
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        _remember(e)
        print(f"[llm_profiler] model call failed: {type(e).__name__}")
    return None


# ── Public API ────────────────────────────────────────────────────────────────
def chat_completion(messages: list, max_tokens: int = 220, temperature: float = 0.35) -> str:
    """One chat call with the chosen provider. Raises RuntimeError if no key is set or the service fails."""
    if not _ready():
        raise RuntimeError("Language model unavailable or API key not set")
    try:
        return _create(messages, max_tokens, temperature)
    except Exception as e:
        _remember(e)
        raise RuntimeError(f"Model call failed: {type(e).__name__}") from e


def extract_profile(goal: str) -> UserProfile:
    """
    Parse a natural-language investment goal into a validated UserProfile with Groq or Gemini.
    Raises RuntimeError if the key is missing or the call fails; there is no rule-based fallback.
    """
    if not _ready():
        raise RuntimeError("Language model unavailable or API key not set (server down)")

    profile_dict = _call_groq(goal)
    if profile_dict is None:
        raise RuntimeError("Language model call failed (server down)")

    # A reply without an explicit is_intent field counts as "not an investment request".
    return sanitize_profile(profile_dict, goal)


def profile_to_dict(profile: UserProfile) -> dict:
    return asdict(profile)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_goals = [
        "Hello, How are you?",
        "I want safe long-term growth with low fees and no dividends",
        "I need income from dividends, moderate risk, prefer tech and healthcare",
        "Aggressive growth in US tech stocks, don't care about fees",
        "Conservative European equity fund, EUR currency, accumulating",
        "What is the weather like today?",
        "My name is John Doe and I like pizza.",
    ]
    for goal in test_goals:
        print(f"\nGoal  : {goal}")
        try:
            p = extract_profile(goal)
        except RuntimeError:
            print("Server is down — the LLM profile service is unavailable. Please try again later.")
            continue

        if not getattr(p, "is_intent", False):
            print(
                "I am an AI finance advisor and can help with finance-related questions."
                " I didn't detect an investment request in your input — please ask a finance-related question or rephrase your goal."
            )
        else:
            print(f"Risk  : {p.risk_level}  |  Horizon: {p.horizon}  |  "
                  f"Dist: {p.distribution}  |  TER: {p.max_ter:.3f}")
            print(f"Sectors: {p.preferred_sectors}")
            print(f"Regions: {p.preferred_regions}  |  Currency: {p.currency}")
