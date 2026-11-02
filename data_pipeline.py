"""
data_pipeline.py
Loads and preprocesses the justetf_2.csv dataset for the recommender system.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import ast
import warnings
warnings.filterwarnings("ignore")

# ── Column groups ────────────────────────────────────────────────────────────
SECTOR_COLS = [
    "exposureSector_Technology", "exposureSector_Consumer Staples",
    "exposureSector_Industrials", "exposureSector_Consumer Discretionary",
    "exposureSector_Other", "exposureSector_Financials",
    "exposureSector_Basic Materials", "exposureSector_Real Estate",
    "exposureSector_Utilities", "exposureSector_Energy",
    "exposureSector_Health Care", "exposureSector_Telecommunication",
]
COUNTRY_COLS = [
    "exposureCountry_United States", "exposureCountry_United Kingdom",
    "exposureCountry_Germany", "exposureCountry_Japan",
    "exposureCountry_France", "exposureCountry_China",
    "exposureCountry_Canada", "exposureCountry_Switzerland",
    "exposureCountry_Australia", "exposureCountry_Other",
]
NUMERIC_FEATURES = [
    "ter", "yearReturnCUR", "threeYearReturnCUR",
    "yearVolatilityCUR", "fundSizeMillions",
    "yearReturnPerRiskCUR",
]


def _parse_numeric_string(series: pd.Series) -> pd.Series:
    """
    Safely convert a column that may be stored as string (possibly with
    thousands-separator commas like '1,535') to float.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series
    return (
        series.astype(str)
              .str.replace(",", "", regex=False)   # remove thousand separators
              .str.strip()
              .pipe(pd.to_numeric, errors="coerce")
    )


def load_and_clean(path: str = "justetf_2.csv") -> pd.DataFrame:
    """Load raw CSV and apply cleaning steps."""
    df = pd.read_csv(path)

    # ── Fix fundSizeMillions (stored as string '1,535' in dataset) ───────────
    df["fundSizeMillions"] = _parse_numeric_string(df["fundSizeMillions"])

    # ── Remember which key numbers were missing BEFORE they are filled in below ─
    df["vol_missing"]    = df["yearVolatilityCUR"].isna()
    df["return_missing"] = df["yearReturnCUR"].isna()
    df["size_missing"]   = df["fundSizeMillions"].isna()
    df["incomplete"]     = df["vol_missing"] | df["return_missing"] | df["size_missing"]

    # ── Parse labels column (stored as string repr of list) ─────────────────
    def safe_parse_labels(x):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []

    df["labels_list"] = df["labels"].apply(safe_parse_labels)
    df["asset_class"] = df["labels_list"].apply(
        lambda lst: lst[1] if len(lst) > 1 else "Unknown"
    )
    df["region"] = df["labels_list"].apply(
        lambda lst: lst[2] if len(lst) > 2 else "Unknown"
    )

    # ── Binary encode distributionPolicy ────────────────────────────────────
    df["is_accumulating"] = (df["distributionPolicy"] == "Accumulating").astype(int)

    # ── Encode replication method ────────────────────────────────────────────
    df["is_physical"] = df["replicationMethod"].apply(
        lambda x: 1 if "replication" in str(x).lower() or "sampling" in str(x).lower()
        else 0
    )

    # ── Encode fund currency ─────────────────────────────────────────────────
    df["is_eur"] = (df["fundCurrency"].str.startswith("EUR")).astype(int)

    # ── Fill missing numerics with median ────────────────────────────────────
    for col in NUMERIC_FEATURES:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())

    # ── Fill missing sector/country exposures with 0 ────────────────────────
    for col in SECTOR_COLS + COUNTRY_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # ── Fill remaining fund size NaNs ────────────────────────────────────────
    df["fundSizeMillions"] = df["fundSizeMillions"].fillna(
        df["fundSizeMillions"].median()
    )

    # ── Cap TER outliers at 99th percentile ─────────────────────────────────
    ter_cap = df["ter"].quantile(0.99)
    df["ter"] = df["ter"].clip(upper=ter_cap)

    # ── Log-transform fund size for scale normalization ──────────────────────
    df["log_fund_size"] = np.log1p(df["fundSizeMillions"])

    # ── Invert volatility so higher score = lower volatility ────────────────
    df["inv_volatility"] = 1 / (df["yearVolatilityCUR"] + 0.001)

    # ── Invert TER so higher score = lower cost ──────────────────────────────
    df["inv_ter"] = 1 / (df["ter"] + 0.0001)

    print(f"[data_pipeline] Loaded {len(df)} ETFs with {len(df.columns)} columns.")
    return df


def data_as_of(df: pd.DataFrame) -> str:
    """The most common quote date in the data (ISO), e.g. '2023-03-21'."""
    dates = pd.to_datetime(df["quoteDate"], errors="coerce").dropna()
    return dates.mode().iloc[0].date().isoformat() if len(dates) else "unknown"


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Build and scale the feature matrix used by the content-based model.
    Returns (scaled_df, feature_columns).
    """
    feature_cols = (
        ["inv_ter", "yearReturnCUR", "threeYearReturnCUR",
         "inv_volatility", "log_fund_size", "yearReturnPerRiskCUR",
         "is_accumulating", "is_physical", "is_eur"]
        + [c for c in SECTOR_COLS if c in df.columns]
        + [c for c in COUNTRY_COLS if c in df.columns]
    )

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[feature_cols].fillna(0))
    scaled_df = pd.DataFrame(scaled, columns=feature_cols, index=df.index)

    return scaled_df, feature_cols


if __name__ == "__main__":
    df = load_and_clean("justetf_2.csv")
    feat_df, feat_cols = build_feature_matrix(df)
    print(f"Feature matrix shape: {feat_df.shape}")
    print(f"Sample features: {feat_cols[:6]}")
    print(f"Fund size sample: {df['fundSizeMillions'].describe()}")
