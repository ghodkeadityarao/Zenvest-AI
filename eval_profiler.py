"""
eval_profiler.py
How well does the language model turn a sentence into a profile? Needs GROQ_API_KEY or GEMINI_API_KEY (42 calls).

    python eval_profiler.py            # prints accuracy per field, writes profiler_results.json

Reports the share of explicitly stated fields the model got right, how many goals were fully right, and whether
it separated investment requests from other text. Calls that fail (rate limit, network) are counted separately
and are not treated as wrong answers.
"""
import json
import os
import time

import llm_profiler
from profiler_goals import GOALS

ROOT = os.path.dirname(os.path.abspath(__file__))
FIELDS = ["risk_level", "horizon", "distribution", "currency", "sectors", "regions", "fee", "min_size", "is_intent"]


def check(profile, key, want) -> bool:
    if key == "sectors":
        return set(profile.preferred_sectors) == want
    if key == "regions":
        return set(profile.preferred_regions) == want
    if key == "fee":
        return profile.max_ter <= want[1]
    if key == "min_size":
        return profile.min_fund_size >= want[1]
    return getattr(profile, key) == want


def score(goals, extract):
    """Per-goal results. `extract` maps a goal to a UserProfile or raises RuntimeError."""
    rows = []
    for goal, expected in goals:
        try:
            profile = extract(goal)
        except RuntimeError:
            rows.append({"goal": goal, "error": True})
            continue
        ok = {k: check(profile, k, v) for k, v in expected.items()}
        rows.append({"goal": goal, "error": False, "fields": ok, "all_correct": all(ok.values()),
                     "got": {"risk": profile.risk_level, "horizon": profile.horizon, "dist": profile.distribution,
                             "max_ter": profile.max_ter, "sectors": profile.preferred_sectors,
                             "regions": profile.preferred_regions, "currency": profile.currency,
                             "is_intent": profile.is_intent}})
    return rows


def summarize(rows):
    done = [r for r in rows if not r["error"]]
    out = {"goals": len(rows), "call_errors": len(rows) - len(done),
           "goals_fully_correct": f"{sum(r['all_correct'] for r in done)}/{len(done)}"}
    per = {}
    for f in FIELDS:
        vals = [r["fields"][f] for r in done if f in r["fields"]]
        if vals:
            per[f] = f"{sum(vals)}/{len(vals)}"
    out["per_field_correct"] = per
    return out


def main():
    problem = llm_profiler.diagnose()
    if problem:
        raise SystemExit(f"Cannot run: {problem}")
    try:
        llm_profiler.extract_profile(GOALS[0][0])            # one call first, so a setup problem shows up at once
    except RuntimeError:
        raise SystemExit(f"The first call failed: {llm_profiler.diagnose() or 'unknown error'}")
    llm_profiler.RETRY_WAITS[:] = [5, 15, 45, 60]            # wait out a free-tier rate limit instead of failing
    pause = 4.5 if llm_profiler.provider() == "gemini" else 0.5
    rows = score(GOALS, lambda g: (time.sleep(pause), llm_profiler.extract_profile(g))[1])
    report = {"provider": llm_profiler.provider(), "model": llm_profiler.active_model(), "summary": summarize(rows), "details": rows}
    json.dump(report, open(os.path.join(ROOT, "profiler_results.json"), "w"), indent=1, default=list)
    print(json.dumps(report["summary"], indent=1))
    for r in rows:
        if not r["error"] and not r["all_correct"]:
            print("MISSED:", r["goal"], [k for k, v in r["fields"].items() if not v], r["got"])


if __name__ == "__main__":
    main()
