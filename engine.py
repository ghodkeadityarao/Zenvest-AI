"""
engine.py
Builds and loads everything the app needs: the cleaned data, the collaborative prior and the XGBoost ranker.
Artifacts are stored as native XGBoost JSON and a CSV in artifacts/ (no pickle), so they are safe to load and do
not depend on the Python or library version that wrote them.

    python engine.py          # rebuild artifacts/ from justetf_2.csv
"""
import os

import pandas as pd

from collaborative_filter import train_cf_model
from data_pipeline import load_and_clean
from recommender import HybridRanker, sample_profiles

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "justetf_2.csv")
ARTIFACT_DIR = os.path.join(ROOT, "artifacts")
TRAIN_PROFILES = 80          # varied profiles the ranker is trained on
TRAIN_SEED = 42              # evaluate.py uses a different seed, so its profiles are held out

_cache = None


def build_artifacts(data_path: str = DATA_PATH, folder: str = ARTIFACT_DIR) -> dict:
    df = load_and_clean(data_path)
    _, cf_scores = train_cf_model(df)
    ranker = HybridRanker()
    ranker.train_on_profiles(df, cf_scores, sample_profiles(TRAIN_PROFILES, TRAIN_SEED))
    ranker.save(folder)
    cf_scores.rename_axis("isin").to_csv(os.path.join(folder, "cf_scores.csv"))
    return {"df": df, "cf_scores": cf_scores, "ranker": ranker}


def load_system(data_path: str = DATA_PATH, folder: str = ARTIFACT_DIR) -> dict:
    """The cleaned data, the collaborative prior and the ranker. Builds the artifacts first if they are missing."""
    global _cache
    if _cache is not None:
        return _cache
    needed = [os.path.join(folder, n) for n in ("ranker.json", "features.json", "cf_scores.csv")]
    if all(os.path.exists(p) for p in needed):
        cf = pd.read_csv(needed[2], index_col="isin")["cf_score"]
        _cache = {"df": load_and_clean(data_path), "cf_scores": cf, "ranker": HybridRanker.load(folder)}
    else:
        _cache = build_artifacts(data_path, folder)
    return _cache


if __name__ == "__main__":
    build_artifacts()
    print(f"Artifacts written to {ARTIFACT_DIR}")
