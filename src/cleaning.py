"""Cleaning + merge logic shared by all scripts. Nothing is dropped silently: invalid values are
set to NaN (so they get imputed + flagged), outliers are kept, orphan events are kept."""
import numpy as np, pandas as pd
from common import load_raw

# (file, column): (lower_exclusive_or_inclusive, upper) physically plausible bounds
VALID = {
    "production_volume_l": (0, 20000), "production_duration_min": (0, 600), "machine_speed": (0, 300),
    "ambient_temperature": (0, 55),
    "raw_material_volume": (0, 25000), "raw_material_age": (0, 168), "fat_percentage": (0, 12),
    "protein_percentage": (0, 10), "acidity": (0.05, 0.30), "raw_material_quality_score": (0, 100),
    "machine_temperature": (40, 110), "pressure": (0, 10), "downtime": (-1e-9, 600),
    "maintenance_age_days": (-1e-9, 400), "vibration": (0, 50),
    "moisture_percentage": (50, 100),
}


def clean_and_merge(verbose=True):
    prod, raw, mach, qual = load_raw()
    log = {}
    # 1. exact duplicates -> then enforce unique event_id per file
    for name, df in [("production", prod), ("raw_material", raw), ("machine", mach), ("quality", qual)]:
        n0 = len(df); exact = df.duplicated().sum()
        key_dup = df.drop_duplicates().event_id.duplicated().sum()
        log[f"{name}_rows"] = n0; log[f"{name}_exact_dups"] = int(exact); log[f"{name}_conflicting_key_dups"] = int(key_dup)
    prod = prod.drop_duplicates(); raw = raw.drop_duplicates(); mach = mach.drop_duplicates(); qual = qual.drop_duplicates()
    for df in (prod, raw, mach, qual):
        assert df.event_id.is_unique, "conflicting duplicate event_id - needs manual rule"

    # 2. batch_id reuse (informational)
    log["batch_id_unique_in_production"] = bool(prod.batch_id.is_unique)
    log["batch_id_reused_values"] = int(prod.batch_id.duplicated(keep=False).sum())

    # 3. cross-file consistency (event_id level)
    ev = set(prod.event_id)
    log["events_missing_in_raw_material"] = len(ev - set(raw.event_id))
    log["events_missing_in_machine"] = len(ev - set(mach.event_id))
    log["events_missing_in_quality"] = len(ev - set(qual.event_id))
    log["orphans_in_other_files"] = int(len(set(raw.event_id) - ev) + len(set(mach.event_id) - ev) + len(set(qual.event_id) - ev))
    m = prod.merge(raw.drop(columns="batch_id"), on="event_id", how="left", validate="1:1") \
            .merge(mach.drop(columns="batch_id"), on="event_id", how="left", validate="1:1") \
            .merge(qual.drop(columns="batch_id"), on="event_id", how="left", validate="1:1")
    # batch_id agreement across files where present
    chk = prod[["event_id", "batch_id"]].merge(raw[["event_id", "batch_id"]], on="event_id", suffixes=("", "_r"))
    log["batch_id_mismatch_prod_vs_raw"] = int((chk.batch_id != chk.batch_id_r).sum())
    log["rows_if_joined_on_batch_id_only"] = int(len(prod.merge(raw[["batch_id"]], on="batch_id")))  # shows the blow-up

    # 4. invalid values -> NaN (kept, counted)
    inv = {}
    for c, (lo, hi) in VALID.items():
        bad = m[c].notna() & ((m[c] < lo) | (m[c] > hi) | ((lo == 0) & (m[c] == 0)))
        inv[c] = int(bad.sum()); m.loc[bad, c] = np.nan
    log["invalid_values_set_to_nan"] = inv

    # 5. timestamp validation + label availability time
    lag = (m.quality_test_timestamp - m.production_timestamp).dt.total_seconds() / 3600
    bad_ts = m.quality_test_timestamp.notna() & (lag <= 0)
    log["quality_ts_not_after_production"] = int(bad_ts.sum())
    p95 = lag[~bad_ts].quantile(.95)
    log["quality_lag_hours_median"] = float(lag[~bad_ts].median()); log["quality_lag_hours_p95"] = float(p95)
    m["label_avail_ts"] = m.quality_test_timestamp
    fallback = (m.production_timestamp + pd.to_timedelta(round(p95 * 3600), unit="s")).astype(m.label_avail_ts.dtype)
    needs_fb = bad_ts | m.quality_test_timestamp.isna()
    m["label_avail_ts"] = m.label_avail_ts.where(~needs_fb, fallback)      # conservative fallback
    m["label_ts_imputed"] = (bad_ts | m.quality_test_timestamp.isna()).astype(int)
    log["timestamps_monotonic_in_production_file"] = bool(prod.production_timestamp.is_monotonic_increasing)

    # 6. label validity
    m["has_label"] = m.rejected.notna()
    conflict = m.has_label & (m.rejected.map({1: "REJECTED", 0: "ACCEPTED"}) != m.quality_status)
    log["label_conflicts_status_vs_flag"] = int(conflict.sum())
    log["events_without_label"] = int((~m.has_label).sum())
    m["label_ok"] = m.has_label & ~conflict          # only these rows are used as training/eval targets / label history

    m["supplier_id"] = m.supplier_id.fillna("UNKNOWN")
    m = m.sort_values(["production_timestamp", "event_id"]).reset_index(drop=True)
    if verbose:
        for k, v in log.items(): print(f"{k}: {v}")
    return m, log
