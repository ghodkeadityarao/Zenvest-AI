"""
evaluate.py
Measures whether the recommender does what the user asked, on profiles the ranker was NOT trained on.
No API key and no user data are needed: profiles are built directly, without the language model.

    python evaluate.py            # prints tables, writes evaluation_results.json

For each of 150 held-out profiles the top 5 ETFs of each method are checked against the profile's hard limits
(fee, distribution, volatility for the stated risk, minimum fund size, currency). It also checks the sector and
region tilt, and that asking for more risk gives a more volatile top 5.

There are no human relevance judgements, so this says nothing about whether the funds are good investments.
"""
import json
import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from engine import TRAIN_PROFILES, TRAIN_SEED, load_system
from llm_profiler import UserProfile
from recommender import (HybridRanker, LIMITS, build_ranker_features, compute_content_scores, limit_checks, map_cf,
                         rank_scores, sample_profiles)

ROOT = os.path.dirname(os.path.abspath(__file__))
VAL_SEED = 7                # used only to choose the collaborative boost (--ablation)
EVAL_SEED = 31337           # the final report; the ranker was trained on seed 42
N_PROFILES = 150
TOP_N = 5
CONSTRAINTS = LIMITS
METHODS = ["hybrid", "original_design", "hybrid_unfiltered", "content_only", "cf_only", "cheapest", "largest_fund", "best_return", "random"]


def method_scores(df, cf_scores, ranker, p, rng, original=None) -> dict:
    content = compute_content_scores(df, p)
    ret_col = "threeYearReturnCUR" if p.horizon == "medium" else "yearReturnCUR"
    ranked = ranker.predict(build_ranker_features(df, content, cf_scores, p))
    out = {
        "hybrid":            rank_scores(df, p, ranked),
        "original_design":   (original.predict(build_ranker_features(df, content, cf_scores, p))
                              if original is not None else np.zeros(len(df))),
        "hybrid_unfiltered": ranked,
        "content_only": content.values,
        "cf_only":      map_cf(df, cf_scores).values,
        "cheapest":     -df["ter"].values,
        "largest_fund": df["fundSizeMillions"].values,
        "best_return":  df[ret_col].values,
        "random":       rng.random(len(df)),
    }
    return out


def top_n(scores, n=TOP_N):
    return np.argsort(-np.asarray(scores, dtype=float), kind="stable")[:n]


def exposure(df, p):
    """(sector exposure, region exposure) of every ETF for the profile's preferences, each clipped to 0..1."""
    sec = np.clip(sum((df[f"exposureSector_{s}"].fillna(0).values for s in p.preferred_sectors), np.zeros(len(df))), 0, 1)
    reg = np.clip(sum((df[f"exposureCountry_{r}"].fillna(0).values for r in p.preferred_regions), np.zeros(len(df))), 0, 1)
    return sec, reg


def bootstrap_ci(values, seed=0, n=2000):
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, (n, len(v)), replace=True).mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 3), round(float(np.percentile(means, 97.5)), 3)]


def constraint_eval(df, cf_scores, ranker, profiles, original=None):
    rng = np.random.default_rng(0)
    rows = []
    for i, p in enumerate(profiles):
        checks = limit_checks(df, p)
        feasible = int(np.logical_and.reduce(list(checks.values())).sum())
        sec, reg = exposure(df, p)
        for m, s in method_scores(df, cf_scores, ranker, p, rng, original).items():
            idx = top_n(s)
            row = {"profile": i, "method": m, "feasible_etfs": feasible,
                   "all_pass": float(np.logical_and.reduce([c[idx] for c in checks.values()]).mean())}
            row.update({c: float(checks[c][idx].mean()) for c in CONSTRAINTS})
            row["sector_exposure"] = float(sec[idx].mean()) if p.preferred_sectors else np.nan
            row["region_exposure"] = float(reg[idx].mean()) if p.preferred_regions else np.nan
            row["sector_base"] = float(sec.mean()) if p.preferred_sectors else np.nan
            row["region_base"] = float(reg.mean()) if p.preferred_regions else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def risk_ordering_eval(df, cf_scores, ranker, original=None):
    """Same goal at low, medium and high risk: does the top-5 median volatility rise with the risk level?"""
    rng = np.random.default_rng(1)
    groups = [(h, d, t, sec) for h in ("short", "medium", "long") for d in ("accumulating", "distributing")
              for t in (0.002, 0.005) for sec in ([], ["Technology"])]
    ok = {m: 0 for m in METHODS}
    vols = {m: {r: [] for r in ("low", "medium", "high")} for m in METHODS}
    for h, d, t, sec in groups:
        med = {m: {} for m in METHODS}
        for risk in ("low", "medium", "high"):
            p = UserProfile(risk, h, d, t, list(sec), [], 100.0, "any", "", True)
            for m, s in method_scores(df, cf_scores, ranker, p, rng, original).items():
                v = float(np.median(df["yearVolatilityCUR"].values[top_n(s)]))
                med[m][risk] = v
                vols[m][risk].append(v)
        for m in METHODS:
            ok[m] += med[m]["low"] < med[m]["medium"] < med[m]["high"]
    return {"groups": len(groups),
            "strictly_increasing": {m: ok[m] for m in METHODS},
            "mean_top5_median_volatility_pct": {m: {r: round(100 * float(np.mean(v)), 1) for r, v in vols[m].items()}
                                                for m in ("hybrid", "original_design", "hybrid_unfiltered", "content_only")}}


def summarize(res: pd.DataFrame, n_profiles: int) -> dict:
    out = {"profiles": n_profiles}
    feas = res[res["feasible_etfs"] >= TOP_N]
    out["profiles_with_at_least_5_feasible_etfs"] = int(feas["profile"].nunique())
    table = {}
    for m in METHODS:
        sub = feas[feas["method"] == m]
        entry = {"all_limits_met": round(float(sub["all_pass"].mean()), 3),
                 "all_limits_met_ci95": bootstrap_ci(sub["all_pass"]),
                 **{c: round(float(sub[c].mean()), 3) for c in CONSTRAINTS}}
        s = res[(res["method"] == m) & res["sector_exposure"].notna()]
        r = res[(res["method"] == m) & res["region_exposure"].notna()]
        entry["sector_exposure_top5"] = round(float(s["sector_exposure"].mean()), 3)
        entry["region_exposure_top5"] = round(float(r["region_exposure"].mean()), 3)
        table[m] = entry
    base = res[(res["method"] == "random") & res["sector_base"].notna()]
    out["sector_exposure_all_etfs"] = round(float(base["sector_base"].mean()), 3)
    out["profiles_with_sector_preference"] = int(len(base))
    base = res[(res["method"] == "random") & res["region_base"].notna()]
    out["region_exposure_all_etfs"] = round(float(base["region_base"].mean()), 3)
    out["profiles_with_region_preference"] = int(len(base))
    out["methods"] = table
    return out


def original_ranker(df, cf_scores):
    """The first version of the ranker: trained on ONE profile, target = 0.6 * content + 0.4 * collaborative, limits soft."""
    p = UserProfile("medium", "long", "accumulating", 0.002, [], [], 100.0, "any", "", True)
    content = compute_content_scores(df, p)
    labels = pd.Series(MinMaxScaler().fit_transform((0.6 * content + 0.4 * map_cf(df, cf_scores)).values.reshape(-1, 1)).ravel(),
                       index=df.index)
    r = HybridRanker()
    r.train(build_ranker_features(df, content, cf_scores, p), labels)
    return r


def ablation(df, cf_scores):
    """Validation profiles only: how the collaborative boost changes personalisation. Chooses CF_BOOST."""
    val = sample_profiles(N_PROFILES, VAL_SEED)
    rows = []
    for boost in (0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0):
        r = HybridRanker()
        r.train_on_profiles(df, cf_scores, sample_profiles(TRAIN_PROFILES, TRAIN_SEED), cf_boost=boost)
        res = constraint_eval(df, cf_scores, r, val)
        h = res[res["method"] == "hybrid"]
        rows.append({"cf_boost": boost,
                     "all_limits_met": round(float(h[h["feasible_etfs"] >= TOP_N]["all_pass"].mean()), 3),
                     "sector_exposure_top5": round(float(h["sector_exposure"].mean()), 3),
                     "region_exposure_top5": round(float(h["region_exposure"].mean()), 3),
                     "risk_groups_ordered_of_24": int(risk_ordering_eval(df, cf_scores, r)["strictly_increasing"]["hybrid"])})
    return {"validation_seed": VAL_SEED, "rows": rows,
            "rule": "largest boost that keeps at least 90% of the zero-boost sector exposure and risk ordering"}


def main():
    import sys
    s = load_system()
    df, cf, ranker = s["df"], s["cf_scores"], s["ranker"]
    if "--ablation" in sys.argv:
        report = ablation(df, cf)
        json.dump(report, open(os.path.join(ROOT, "ablation_cf_prior.json"), "w"), indent=1)
        print(json.dumps(report, indent=1))
        return
    profiles = sample_profiles(N_PROFILES, EVAL_SEED)
    original = original_ranker(df, cf)
    res = constraint_eval(df, cf, ranker, profiles, original)
    report = {"evaluation_seed": EVAL_SEED, "constraints": summarize(res, len(profiles)),
              "risk_ordering": risk_ordering_eval(df, cf, ranker, original)}
    json.dump(report, open(os.path.join(ROOT, "evaluation_results.json"), "w"), indent=1)
    res.to_csv(os.path.join(ROOT, "evaluation_per_profile.csv"), index=False)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
