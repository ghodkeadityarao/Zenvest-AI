"""
recommender.py
Content-based scoring, XGBoost hybrid ranker, and SHAP-based explanations.
Falls back gracefully to a linear ranker if XGBoost is not installed.
"""

import json
import os
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    import shap
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

print(f"[recommender] XGBoost available: {XGB_AVAILABLE}")

from llm_profiler import UserProfile, SECTORS, REGIONS
from data_pipeline import SECTOR_COLS, COUNTRY_COLS

# ── Risk → target volatility map ─────────────────────────────────────────────
RISK_VOL = {"low": 0.10, "medium": 0.20, "high": 0.50}

# ── Human-readable feature labels for SHAP display ───────────────────────────
FEATURE_LABELS = {
    "content_score":                        "Overall profile match",
    "cf_score":                             "Similar investor preference",
    "ter":                                  "Expense ratio (TER)",
    "ter_excess":                           "TER above your limit",
    "vol_excess":                           "Volatility above risk limit",
    "volatility":                           "Fund volatility",
    "year_return":                          "1-year return",
    "three_yr_return":                      "3-year return",
    "log_size":                             "Fund size (liquidity)",
    "return_per_risk":                      "Return-to-risk ratio",
    "is_accumulating":                      "Accumulating policy",
    "dist_match":                           "Distribution policy match",
    "size_ok":                              "Fund size meets minimum",
    "currency_match":                       "Fund currency match",
    "exposureSector_Technology":            "Technology sector",
    "exposureSector_Health_Care":           "Health Care sector",
    "exposureSector_Financials":            "Financials sector",
    "exposureSector_Energy":                "Energy sector",
    "exposureSector_Consumer_Discretionary":"Consumer Discretionary",
    "exposureSector_Industrials":           "Industrials sector",
    "exposureSector_Real_Estate":           "Real Estate sector",
    "exposureSector_Utilities":             "Utilities sector",
    "exposureSector_Basic_Materials":       "Basic Materials sector",
    "exposureSector_Telecommunication":     "Telecommunication sector",
    "exposureSector_Consumer_Staples":      "Consumer Staples sector",
    "exposureSector_Other":                 "Other sectors",
}


def currency_match(df: pd.DataFrame, profile: UserProfile) -> pd.Series:
    """1 for funds quoted in the requested currency ('EUR' also counts 'EUR Hedged'), else 0; all 1 when no preference."""
    if profile.currency == "any":
        return pd.Series(1, index=df.index)
    return df["fundCurrency"].astype(str).str.startswith(profile.currency).astype(int)


LIMITS = ["fee", "distribution", "volatility", "size", "currency"]
LIMIT_PRIORITY = 10.0     # added to the score of every ETF that meets all stated limits, so those rank first
INCOMPLETE_PENALTY = 100.0  # taken off funds with a missing volatility, 1-year return or size: their numbers were filled in


def limit_checks(df: pd.DataFrame, profile: UserProfile) -> dict:
    """Boolean array per limit the user stated: fee, distribution, volatility for the risk level, size, currency."""
    return {
        "fee":          (df["ter"] <= profile.max_ter).values,
        "distribution": (df["distributionPolicy"].str.lower() == profile.distribution).values,
        "volatility":   (df["yearVolatilityCUR"] <= RISK_VOL[profile.risk_level]).values,
        "size":         (df["fundSizeMillions"] >= profile.min_fund_size).values,
        "currency":     (currency_match(df, profile) == 1).values,
    }


def meets_limits(df: pd.DataFrame, profile: UserProfile) -> np.ndarray:
    return np.logical_and.reduce(list(limit_checks(df, profile).values()))


def rank_scores(df: pd.DataFrame, profile: UserProfile, model_scores) -> np.ndarray:
    """Final ordering: funds meeting every limit first, funds with filled-in (missing) data last, model score within each."""
    return (np.asarray(model_scores, dtype=float) + LIMIT_PRIORITY * meets_limits(df, profile)
            - INCOMPLETE_PENALTY * df["incomplete"].values)


# ── Content-based scorer ──────────────────────────────────────────────────────
def compute_content_scores(df: pd.DataFrame, profile: UserProfile) -> pd.Series:
    """
    Score each ETF against the user profile using weighted feature matching.
    Returns normalised Series [0, 1] indexed by df.index.
    """
    scores     = np.zeros(len(df))
    target_vol = RISK_VOL[profile.risk_level]

    # 1. Volatility fit (25%)
    vol       = df["yearVolatilityCUR"].fillna(0.5).values
    scores   += np.exp(-2 * np.maximum(vol - target_vol, 0)) * 0.25

    # 2. Return for horizon (20%)
    ret_col   = {"short": "yearReturnCUR", "medium": "threeYearReturnCUR",
                 "long": "yearReturnCUR"}[profile.horizon]
    returns   = df[ret_col].fillna(df[ret_col].median()).values
    scores   += MinMaxScaler().fit_transform(returns.reshape(-1, 1)).flatten() * 0.20

    # 3. TER fit (20%)
    ter       = df["ter"].fillna(0.01).values
    scores   += np.exp(-300 * np.maximum(ter - profile.max_ter, 0)) * 0.20

    # 4. Distribution policy match (15%)
    dist_match = (
        df["distributionPolicy"].str.lower() == profile.distribution
    ).astype(float).values
    scores   += dist_match * 0.15

    # 5. Preferred sectors (10%)
    if profile.preferred_sectors:
        sec_score = np.zeros(len(df))
        for sector in profile.preferred_sectors:
            col = f"exposureSector_{sector}"
            if col in df.columns:
                sec_score += df[col].fillna(0).values
        scores += np.clip(sec_score, 0, 1) * 0.10
    else:
        scores += 0.05

    # 6. Preferred regions (5%)
    if profile.preferred_regions:
        reg_score = np.zeros(len(df))
        for region in profile.preferred_regions:
            col = f"exposureCountry_{region}"
            if col in df.columns:
                reg_score += df[col].fillna(0).values
        scores += np.clip(reg_score, 0, 1) * 0.05

    # 7. Fund size (5%)
    size      = df["fundSizeMillions"].fillna(100).values
    scores   += (size >= profile.min_fund_size).astype(float) * 0.05

    # 8. Currency (5%, only when the user named one)
    if profile.currency != "any":
        scores += currency_match(df, profile).values * 0.05

    return pd.Series(
        MinMaxScaler().fit_transform(scores.reshape(-1, 1)).flatten(),
        index=df.index,
        name="content_score",
    )


# ── Feature matrix for XGBoost ────────────────────────────────────────────────
def build_ranker_features(
    df: pd.DataFrame,
    content_scores: pd.Series,
    cf_scores: pd.Series,
    profile: UserProfile,
) -> pd.DataFrame:
    target_vol = RISK_VOL[profile.risk_level]
    feat       = pd.DataFrame(index=df.index)

    feat["content_score"]   = content_scores
    feat["cf_score"]        = df["isin"].map(cf_scores).fillna(0.0)
    feat["ter"]             = df["ter"].fillna(df["ter"].median())
    feat["year_return"]     = df["yearReturnCUR"].fillna(0)
    feat["three_yr_return"] = df["threeYearReturnCUR"].fillna(0)
    feat["volatility"]      = df["yearVolatilityCUR"].fillna(0.5)
    feat["log_size"]        = np.log1p(df["fundSizeMillions"].fillna(100))
    feat["return_per_risk"] = df["yearReturnPerRiskCUR"].fillna(0)
    feat["is_accumulating"] = (df["distributionPolicy"] == "Accumulating").astype(int)
    feat["is_physical"]     = df["replicationMethod"].apply(
        lambda x: 1 if "replication" in str(x).lower() else 0
    )
    feat["vol_excess"]      = np.maximum(feat["volatility"] - target_vol, 0)
    feat["ter_excess"]      = np.maximum(feat["ter"] - profile.max_ter, 0)
    feat["dist_match"]      = (
        df["distributionPolicy"].str.lower() == profile.distribution
    ).astype(int)
    feat["size_ok"]         = (
        df["fundSizeMillions"].fillna(0) >= profile.min_fund_size
    ).astype(int)
    feat["currency_match"]  = currency_match(df, profile)

    for col in SECTOR_COLS:
        if col in df.columns:
            feat[col.replace(" ", "_")] = df[col].fillna(0)

    return feat.fillna(0)


CF_BOOST = 0.0      # how much the collaborative prior can lift a score. 0 = off: on validation profiles any positive
                    # value lowered personalisation and nothing improved (see evaluate.py --ablation)


def generate_pseudo_labels(
    content_scores: pd.Series, cf_scores_mapped: pd.Series, cf_boost: float = CF_BOOST
) -> pd.Series:
    """Training target: the profile-specific content score, lifted by up to cf_boost (a fraction) by a profile-independent
    prior. It is multiplicative so the prior only reorders close matches and cannot override the user's profile."""
    combined = content_scores * (1 + cf_boost * cf_scores_mapped)
    return pd.Series(
        MinMaxScaler().fit_transform(combined.values.reshape(-1, 1)).flatten(),
        index=content_scores.index,
    )


# ── Training profiles ─────────────────────────────────────────────────────────
def sample_profiles(n: int = 80, seed: int = 42) -> list[UserProfile]:
    """n varied, valid profiles. The ranker is trained on one set (seed 42) and evaluated on another seed."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.append(UserProfile(
            risk_level        = rng.choice(["low", "medium", "high"]),
            horizon           = rng.choice(["short", "medium", "long"]),
            distribution      = rng.choice(["accumulating", "distributing"]),
            max_ter           = rng.choice([0.001, 0.002, 0.003, 0.005, 0.0075]),
            preferred_sectors = rng.sample(SECTORS, rng.choice([0, 0, 1, 2])),
            preferred_regions = rng.sample(REGIONS, rng.choice([0, 0, 1])),
            min_fund_size     = rng.choice([0.0, 100.0, 500.0, 1000.0]),
            currency          = rng.choice(["any", "any", "EUR", "USD"]),
            raw_goal          = "", is_intent = True,
        ))
    return out


def map_cf(df: pd.DataFrame, cf_scores: pd.Series) -> pd.Series:
    return pd.Series(df["isin"].map(cf_scores).fillna(0).values, index=df.index)


# ── Hybrid XGBoost ranker ─────────────────────────────────────────────────────
class HybridRanker:
    """XGBoost regressor trained to reproduce the blended content + collaborative score for many profiles.
    It is a smooth, explainable stand-in for that formula (there are no human relevance labels)."""

    def __init__(self):
        self.model         = None
        self.feature_names = None
        self.explainer     = None

    def train(self, features: pd.DataFrame, labels: pd.Series):
        if not XGB_AVAILABLE:
            raise RuntimeError("XGBoost is required but not installed. Run: pip install xgboost")
        self.feature_names = list(features.columns)
        self.model = xgb.XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            random_state=42, verbosity=0,
        )
        self.model.fit(features, labels)
        self.explainer = shap.TreeExplainer(self.model)
        print(f"[recommender] XGBoost ranker trained on {len(features)} rows.")

    def train_on_profiles(self, df: pd.DataFrame, cf_scores: pd.Series, profiles: list[UserProfile], cf_boost: float = CF_BOOST):
        cf_mapped = map_cf(df, cf_scores)
        feats, labels = [], []
        for p in profiles:
            content = compute_content_scores(df, p)
            feats.append(build_ranker_features(df, content, cf_scores, p))
            labels.append(generate_pseudo_labels(content, cf_mapped, cf_boost))
        self.train(pd.concat(feats, ignore_index=True), pd.concat(labels, ignore_index=True))

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("HybridRanker has not been trained yet.")
        return self.model.predict(features[self.feature_names])

    def explain_rows(self, features: pd.DataFrame, positions: list[int]) -> list[dict]:
        if self.explainer is None:
            raise RuntimeError("HybridRanker has not been trained yet.")
        rows = features[self.feature_names].iloc[positions]
        return [dict(zip(self.feature_names, vals)) for vals in self.explainer.shap_values(rows)]

    def explain(self, features: pd.DataFrame, idx: int) -> dict:
        return self.explain_rows(features, [idx])[0]

    def save(self, folder: str):
        """Native XGBoost JSON plus the feature names: no pickle."""
        os.makedirs(folder, exist_ok=True)
        self.model.save_model(os.path.join(folder, "ranker.json"))
        with open(os.path.join(folder, "features.json"), "w") as f:
            json.dump(self.feature_names, f)

    @classmethod
    def load(cls, folder: str) -> "HybridRanker":
        if not XGB_AVAILABLE:
            raise RuntimeError("XGBoost is required but not installed. Run: pip install xgboost")
        r = cls()
        r.model = xgb.XGBRegressor()
        r.model.load_model(os.path.join(folder, "ranker.json"))
        with open(os.path.join(folder, "features.json")) as f:
            r.feature_names = json.load(f)
        r.explainer = shap.TreeExplainer(r.model)
        return r


# ── Plain-English explanation ─────────────────────────────────────────────────
def generate_explanation(etf_row, shap_dict: dict, rank: int) -> str:
    name    = etf_row.get("name", "This ETF")
    ter_pct = etf_row.get("ter", 0) * 100
    vol_pct = etf_row.get("yearVolatilityCUR", 0) * 100
    ret_pct = etf_row.get("yearReturnCUR", 0) * 100
    dist    = etf_row.get("distributionPolicy", "N/A")
    size    = etf_row.get("fundSizeMillions", 0)

    sorted_shap  = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    top_pos  = [(k, v) for k, v in sorted_shap if v > 0][:3]
    top_neg  = [(k, v) for k, v in sorted_shap if v < 0][:2]

    reasons  = [
        f"{FEATURE_LABELS.get(k, k.replace('_',' ').title())} (+{abs(v):.3f})"
        for k, v in top_pos
    ]
    concerns = [
        f"{FEATURE_LABELS.get(k, k.replace('_',' ').title())} (−{abs(v):.3f})"
        for k, v in top_neg
    ]

    lines = [
        f"**Rank #{rank} — {name}**\n",
        f"Recommended because: {'; '.join(reasons) if reasons else 'strong overall profile match'}.\n",
        f"**TER** {ter_pct:.2f}% · **1-yr return** {ret_pct:+.1f}% · "
        f"**Volatility** {vol_pct:.1f}% · {dist} · "
        f"**Size** €{size:,.0f}M",
    ]
    if concerns:
        lines.append(f"\nMinor trade-offs: {'; '.join(concerns)}.")
    return "\n".join(lines)


# ── Full recommendation pipeline ──────────────────────────────────────────────
def recommend(
    df: pd.DataFrame,
    profile: UserProfile,
    cf_scores: pd.Series,
    ranker: HybridRanker,
    top_n: int = 5,
) -> list[dict]:
    content_scores = compute_content_scores(df, profile)
    features       = build_ranker_features(df, content_scores, cf_scores, profile)
    meets          = meets_limits(df, profile)
    model_scores   = ranker.predict(features)

    work_df           = df.copy()
    work_df["_model"] = model_scores
    work_df["_score"] = rank_scores(df, profile, model_scores)
    top_df            = work_df.nlargest(top_n, "_score").reset_index(drop=False)
    positions         = [features.index.get_loc(i) for i in top_df["index"]]
    shap_rows         = ranker.explain_rows(features, positions)

    results = []
    for rank, ((_, row), shap_dict) in enumerate(zip(top_df.iterrows(), shap_rows), start=1):
        orig_idx = row["index"]
        shap_display = {
            FEATURE_LABELS.get(k, k): round(float(v), 4)
            for k, v in sorted(shap_dict.items(),
                                key=lambda x: abs(x[1]), reverse=True)[:6]
        }

        results.append({
            "rank":          rank,
            "isin":          row.get("isin", ""),
            "name":          row.get("name", ""),
            "provider":      row.get("fundProvider", ""),
            "ter":           row.get("ter", 0),
            "year_return":   row.get("yearReturnCUR", 0),
            "volatility":    row.get("yearVolatilityCUR", 0),
            "distribution":  row.get("distributionPolicy", ""),
            "fund_size":     row.get("fundSizeMillions", 0),
            "currency":      row.get("fundCurrency", ""),
            "labels":        row.get("labels", ""),
            "final_score":   float(row["_model"]),
            "meets_limits":  bool(meets[features.index.get_loc(orig_idx)]),
            "content_score": float(content_scores.loc[orig_idx]),
            "cf_score":      float(cf_scores.get(row.get("isin", ""), 0.0)),
            "explanation":   generate_explanation(row, shap_dict, rank),
            "shap_values":   shap_display,
        })

    return results


if __name__ == "__main__":
    from engine import load_system
    sysm = load_system()
    profile = UserProfile("low", "long", "accumulating", 0.002, [], [], 500.0, "any", "demo", True)
    for r in recommend(sysm["df"], profile, sysm["cf_scores"], sysm["ranker"], top_n=5):
        print(f"Rank {r['rank']}: {r['name']}  score {r['final_score']:.3f}  TER {r['ter']*100:.2f}%  vol {r['volatility']*100:.1f}%")
