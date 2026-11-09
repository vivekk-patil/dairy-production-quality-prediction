"""Builds the leakage-safe feature table (all rows) from the cleaned master."""
import pandas as pd
from common import PROC
from features import build_features

F = build_features(pd.read_pickle(PROC / "master_clean.pkl"))
F.to_pickle(PROC / "features_all.pkl")
print("features:", F.shape)
