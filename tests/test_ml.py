import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("shap")

import engine  # noqa: E402
import eval_profiler as EP  # noqa: E402
import evaluate as E  # noqa: E402
import profiler_goals as PG  # noqa: E402
import recommender as R  # noqa: E402
from collaborative_filter import train_cf_model  # noqa: E402
from data_pipeline import load_and_clean  # noqa: E402
from llm_profiler import REGIONS, SECTORS, UserProfile, sanitize_profile  # noqa: E402


def profile(**kw):
    base = dict(risk_level="medium", horizon="long", distribution="accumulating", max_ter=0.005, preferred_sectors=[],
                preferred_regions=[], min_fund_size=100.0, currency="any", raw_goal="", is_intent=True)
    base.update(kw)
    return UserProfile(**base)


@pytest.fixture(scope="module")
def df():
    return load_and_clean(engine.DATA_PATH)


@pytest.fixture(scope="module")
def cf(df):
    return train_cf_model(df)[1]


@pytest.fixture(scope="module")
def ranker(df, cf):
    r = R.HybridRanker()
    r.train_on_profiles(df, cf, R.sample_profiles(8, 1))
    return r


# profile validation
def test_sanitize_replaces_bad_values_with_defaults():
    p = sanitize_profile({"risk_level": "very high", "horizon": "LONG", "distribution": 5, "max_ter": -3,
                          "preferred_sectors": ["technology", "Nope", "Health Care", "technology"],
                          "preferred_regions": "Germany", "min_fund_size": "abc", "currency": "eur", "is_intent": "true"}, "g")
    assert (p.risk_level, p.horizon, p.distribution) == ("medium", "long", "accumulating")
    assert p.max_ter == 0.0005 and p.min_fund_size == 100.0 and p.currency == "EUR" and p.is_intent
    assert p.preferred_sectors == ["Technology", "Health Care"] and p.preferred_regions == [] and p.raw_goal == "g"


def test_sanitize_handles_non_dicts_and_clamps():
    assert not sanitize_profile(None).is_intent and not sanitize_profile("x").is_intent
    p = sanitize_profile({"max_ter": 5, "min_fund_size": 1e12, "is_intent": 1, "preferred_sectors": SECTORS})
    assert p.max_ter == 0.02 and p.min_fund_size == 10000.0 and p.is_intent is False and len(p.preferred_sectors) == 3


# data and scoring
def test_data_loads_clean(df):
    assert len(df) == 2264 and df["fundSizeMillions"].notna().all() and df["yearVolatilityCUR"].notna().all()


def test_currency_match(df):
    assert R.currency_match(df, profile(currency="any")).all()
    eur = R.currency_match(df, profile(currency="EUR"))
    assert eur[df["fundCurrency"] == "EUR Hedged"].all() and not eur[df["fundCurrency"] == "USD"].any()


def test_currency_changes_the_content_score(df):
    usd = R.compute_content_scores(df, profile(currency="USD"))
    base = R.compute_content_scores(df, profile())
    is_usd = df["fundCurrency"].str.startswith("USD")
    assert usd[is_usd].mean() > usd[~is_usd].mean() and (usd[is_usd].mean() - usd[~is_usd].mean()) > (base[is_usd].mean() - base[~is_usd].mean())


def test_limit_checks_and_meets_limits(df):
    p = profile(risk_level="low", max_ter=0.002, min_fund_size=500.0)
    checks = R.limit_checks(df, p)
    assert set(checks) == set(R.LIMITS)
    ok = df[R.meets_limits(df, p)]
    assert len(ok) > 20 and (ok["ter"] <= 0.002).all() and (ok["distributionPolicy"] == "Accumulating").all()
    assert (ok["yearVolatilityCUR"] <= 0.10).all() and (ok["fundSizeMillions"] >= 500).all()


def test_pseudo_labels_follow_the_boost(df, cf):
    c = R.compute_content_scores(df, profile())
    m = R.map_cf(df, cf)
    off = R.generate_pseudo_labels(c, m, 0.0)
    assert np.allclose(off, c, atol=1e-9) and not np.allclose(R.generate_pseudo_labels(c, m, 1.0), c)
    assert R.CF_BOOST == 0.0


def test_sample_profiles_are_deterministic_valid_and_vary():
    a, b, c = R.sample_profiles(30, 5), R.sample_profiles(30, 5), R.sample_profiles(30, 6)
    assert a == b and a != c
    for p in a:
        assert sanitize_profile(p.__dict__, "") .__dict__ == {**p.__dict__, "raw_goal": ""}
        assert set(p.preferred_sectors) <= set(SECTORS) and set(p.preferred_regions) <= set(REGIONS)


# ranker
def test_ranker_save_and_load_round_trip(df, cf, ranker, tmp_path):
    ranker.save(str(tmp_path))
    loaded = R.HybridRanker.load(str(tmp_path))
    f = R.build_ranker_features(df, R.compute_content_scores(df, profile()), cf, profile())
    assert np.allclose(ranker.predict(f), loaded.predict(f), atol=1e-6)
    assert not any(p.suffix == ".pkl" for p in tmp_path.iterdir())


def test_recommend_returns_five_distinct_ranked_funds_within_limits(df, cf, ranker):
    p = profile(risk_level="low", max_ter=0.003, currency="EUR")
    res = R.recommend(df, p, cf, ranker, top_n=5)
    assert len(res) == 5 and len({r["isin"] for r in res}) == 5
    assert [r["rank"] for r in res] == [1, 2, 3, 4, 5] and all(r["meets_limits"] for r in res)
    assert all(r["currency"].startswith("EUR") and r["ter"] <= 0.003 and r["volatility"] <= 0.10 for r in res)
    assert all(r["explanation"].startswith("**Rank #") and 0 < len(r["shap_values"]) <= 6 for r in res)
    assert [r["final_score"] for r in res] == sorted((r["final_score"] for r in res), reverse=True)


def test_more_risk_gives_a_more_volatile_top_five(df, cf, ranker):
    vol = {k: np.median([r["volatility"] for r in R.recommend(df, profile(risk_level=k), cf, ranker)]) for k in ("low", "high")}
    assert vol["low"] < vol["high"]


def test_limits_relaxed_when_nothing_meets_them(df, cf, ranker):
    res = R.recommend(df, profile(max_ter=0.0005, min_fund_size=10000.0, risk_level="low", currency="USD"), cf, ranker)
    assert len(res) == 5 and not all(r["meets_limits"] for r in res)


def test_engine_builds_and_reloads_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "_cache", None)
    monkeypatch.setattr(engine, "TRAIN_PROFILES", 4)
    folder = str(tmp_path / "art")
    built = engine.load_system(folder=folder)
    assert (tmp_path / "art" / "ranker.json").exists() and (tmp_path / "art" / "cf_scores.csv").exists()
    monkeypatch.setattr(engine, "_cache", None)
    again = engine.load_system(folder=folder)
    pd.testing.assert_series_equal(built["cf_scores"], again["cf_scores"], check_names=False, check_dtype=False, atol=1e-6)


# evaluation code
def test_top_n_is_deterministic_and_stable_on_ties():
    assert list(E.top_n([1, 3, 3, 2, 0], 3)) == [1, 2, 3]


def test_bootstrap_ci_is_inside_the_data_range():
    lo, hi = E.bootstrap_ci([0.2, 0.4, 0.6, 0.8])
    assert 0.2 <= lo <= hi <= 0.8


def test_constraint_eval_gives_a_perfect_system_and_weaker_baselines(df, cf, ranker):
    res = E.constraint_eval(df, cf, ranker, R.sample_profiles(12, 99))
    feas = res[res["feasible_etfs"] >= 5]
    assert set(res["method"]) >= {"hybrid", "random", "cheapest"}
    assert (feas[feas["method"] == "hybrid"]["all_pass"] == 1.0).all()
    assert feas[feas["method"] == "random"]["all_pass"].mean() < 0.5


# profile-extraction scorer
def test_goal_labels_are_well_formed():
    assert len(PG.GOALS) == 42 and sum(e["is_intent"] for _, e in PG.GOALS) == 32
    for goal, e in PG.GOALS:
        assert set(e) <= set(EP.FIELDS) and "is_intent" in e
        assert set(e.get("sectors", ())) <= set(SECTORS) and set(e.get("regions", ())) <= set(REGIONS)


def test_profiler_scorer_counts_right_wrong_and_failed_calls():
    goals = [("a", {"risk_level": "low", "sectors": {"Technology"}, "fee": ("le", 0.002), "is_intent": True}),
             ("b", {"risk_level": "high", "is_intent": True}), ("c", {"is_intent": False})]

    def extract(g):
        if g == "c":
            raise RuntimeError("down")
        return profile(risk_level="low" if g == "a" else "medium", preferred_sectors=["Technology"], max_ter=0.001)

    rows = EP.score(goals, extract)
    s = EP.summarize(rows)
    assert rows[0]["all_correct"] and not rows[1]["all_correct"] and rows[2]["error"]
    assert s["call_errors"] == 1 and s["goals_fully_correct"] == "1/2"
    assert s["per_field_correct"]["risk_level"] == "1/2" and s["per_field_correct"]["fee"] == "1/1"


# missing data and the data date
def test_missing_numbers_are_flagged_before_they_are_filled(df):
    raw = pd.read_csv(engine.DATA_PATH)
    assert int(df["vol_missing"].sum()) == int(raw["yearVolatilityCUR"].isna().sum()) == 226
    assert int(df["return_missing"].sum()) == int(raw["yearReturnCUR"].isna().sum()) == 228
    assert df["incomplete"].sum() == 232 and df["yearVolatilityCUR"].notna().all()


def test_funds_with_filled_in_data_are_not_recommended(df, cf, ranker):
    incomplete = df.set_index("isin")["incomplete"]
    for p in R.sample_profiles(25, 3):
        assert not incomplete.loc[[r["isin"] for r in R.recommend(df, p, cf, ranker)]].any()


def test_rank_scores_order_limits_first_and_incomplete_last(df):
    p = profile(risk_level="low", max_ter=0.003)
    scores = R.rank_scores(df, p, np.zeros(len(df)))
    meets, inc = R.meets_limits(df, p), df["incomplete"].values
    assert scores[meets & ~inc].min() > scores[~meets & ~inc].max() > scores[inc].max()


def test_data_date_is_march_2023():
    from data_pipeline import data_as_of
    assert data_as_of(load_and_clean(engine.DATA_PATH)) == "2023-03-21"
