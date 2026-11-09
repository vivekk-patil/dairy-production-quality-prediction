"""Shared paths and I/O helpers."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"
REP = ROOT / "reports"
DOCS = ROOT / "docs"
MODELS = ROOT / "models"
for p in (RAW, PROC, FIG, REP, DOCS, MODELS):
    p.mkdir(parents=True, exist_ok=True)

SEED = 42


def load_raw():
    prod = pd.read_csv(RAW / "production.csv", parse_dates=["production_timestamp"])
    raw = pd.read_csv(RAW / "raw_material.csv")
    mach = pd.read_csv(RAW / "machine.csv")
    qual = pd.read_csv(RAW / "quality.csv", parse_dates=["quality_test_timestamp"])
    return prod, raw, mach, qual
