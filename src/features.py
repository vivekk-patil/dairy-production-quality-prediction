"""Leakage-safe feature engineering. Prediction time t = production_timestamp (batch completion).

Label-history rule: the label of event j is usable for event i only if label_avail_ts[j] <= t_i
(i.e. the lab result has actually been reported, ~24h after production).
"""
import numpy as np, pandas as pd

SETPOINT_C = 72.0
H = 3600 * 1e9  # ns per hour


def available_label_counts(df, group_col=None, window_days=None, last_n=None):
    """For every row i return (n, rejections) among labels in the same group that are AVAILABLE at t_i.
    window_days -> only labels that became available within the last N days; last_n -> only the last N available labels."""
    t = df["production_timestamp"].values.astype("datetime64[ns]").astype("int64")
    A = df["label_avail_ts"].values.astype("datetime64[ns]").astype("int64")
    y = df["rejected"].fillna(0).values
    ok = df["label_ok"].values
    n_out = np.zeros(len(df)); r_out = np.zeros(len(df))
    groups = df[group_col].values if group_col else np.zeros(len(df), dtype=int)
    for g in np.unique(groups):
        gi = np.where(groups == g)[0]
        pool = gi[ok[gi]]
        order = np.argsort(A[pool], kind="stable")
        As = A[pool][order]
        cum = np.concatenate([[0], np.cumsum(y[pool][order])])
        hi = np.searchsorted(As, t[gi], side="right")          # labels reported at or before t
        if last_n:
            lo = np.maximum(hi - last_n, 0)
        elif window_days:
            lo = np.searchsorted(As, t[gi] - int(window_days * 24 * H), side="right")
        else:
            lo = np.zeros_like(hi)
        n_out[gi] = hi - lo; r_out[gi] = cum[hi] - cum[lo]
    return n_out, r_out


def smoothed_rate(df, group_col, window_days=None, last_n=None, k=20):
    """Bayesian-smoothed historical rejection rate shrunk to the overall available rate at time t.
    No history at all (first day / unseen category) -> falls back to overall rate; if even that is empty -> NaN."""
    n, r = available_label_counts(df, group_col, window_days, last_n)
    n_all, r_all = available_label_counts(df, None, window_days=None)
    prior = np.where(n_all > 0, r_all / np.maximum(n_all, 1), np.nan)
    return (r + k * prior) / (n + k)


def build_features(m: pd.DataFrame) -> pd.DataFrame:
    m = m.sort_values(["production_timestamp", "event_id"]).reset_index(drop=True).copy()
    f = pd.DataFrame(index=m.index)
    g = m.groupby("production_line")

    # --- process / machine ---
    f["machine_temp_deviation"] = m.machine_temperature - SETPOINT_C
    f["machine_temp_abs_deviation"] = f.machine_temp_deviation.abs()
    f["temp_variability_10"] = g["machine_temperature"].transform(lambda s: s.rolling(10, min_periods=3).std())
    f["production_duration_min"] = m.production_duration_min
    f["downtime_min"] = m.downtime
    f["downtime_ratio"] = (m.downtime / m.production_duration_min).clip(0, 1)
    base_v = g["vibration"].transform(lambda s: s.shift(1).rolling(50, min_periods=10).mean())
    f["vibration_deviation"] = m.vibration - base_v
    f["vibration"] = m.vibration
    f["maintenance_age_days"] = m.maintenance_age_days
    f["pressure"] = m.pressure
    base_vol = g["production_volume_l"].transform(lambda s: s.shift(1).rolling(100, min_periods=10).median())
    f["volume_deviation_pct"] = (m.production_volume_l - base_vol) / base_vol
    f["machine_speed"] = m.machine_speed
    run_min = (m.production_duration_min - m.downtime).clip(lower=1)
    f["machine_efficiency"] = m.production_volume_l / (m.machine_speed * run_min)

    # --- raw material ---
    f["raw_material_ratio"] = m.raw_material_volume / m.production_volume_l
    f["raw_material_age_h"] = m.raw_material_age
    f["raw_material_freshness"] = 1 / (1 + m.raw_material_age / 12)
    f["fat_percentage"] = m.fat_percentage
    f["protein_percentage"] = m.protein_percentage
    f["fat_protein_ratio"] = m.fat_percentage / m.protein_percentage
    f["acidity"] = m.acidity
    f["raw_material_quality_score"] = m.raw_material_quality_score

    # --- context / time (cyclical so the model cannot extrapolate a raw timeline index) ---
    h = m.production_timestamp.dt.hour + m.production_timestamp.dt.minute / 60
    f["hour_sin"] = np.sin(2 * np.pi * h / 24); f["hour_cos"] = np.cos(2 * np.pi * h / 24)
    f["day_of_week"] = m.production_timestamp.dt.dayofweek
    f["ambient_temperature"] = m.ambient_temperature

    # --- label-history features (delay-aware) ---
    f["supplier_hist_rej_rate"] = smoothed_rate(m, "supplier_id")
    f["supplier_rej_rate_30d"] = smoothed_rate(m, "supplier_id", window_days=30)
    f["shift_hist_rej_rate"] = smoothed_rate(m, "shift", window_days=90)
    f["line_rej_rate_last200"] = smoothed_rate(m, "production_line", last_n=200)
    n1, r1 = available_label_counts(m, "production_line", last_n=1)
    f["prev_batch_rejected"] = np.where(n1 == 1, r1, np.nan)     # most recent REPORTED batch on this line
    n_sup, _ = available_label_counts(m, "supplier_id")
    f["supplier_label_history_n"] = n_sup                       # 0 for unseen suppliers -> model can tell the rate is a prior

    # --- categoricals (raw) ---
    f["production_line"] = m.production_line
    f["shift"] = m["shift"]
    f["supplier_id"] = m.supplier_id

    meta = m[["event_id", "production_timestamp", "label_avail_ts", "rejected", "label_ok"]]
    return pd.concat([meta, f], axis=1)


FEATURE_DOC = [
 ("machine_temp_deviation", "machine_temperature - 72 degC setpoint (signed)", "Under/over-pasteurisation risk"),
 ("machine_temp_abs_deviation", "|machine_temp_deviation|", "Any drift from setpoint hurts quality"),
 ("temp_variability_10", "Rolling std of machine temperature, last 10 batches on the line (incl. current)", "Unstable heat treatment / control loop"),
 ("production_duration_min", "Batch run time", "Long runs suggest slowdowns/stoppages"),
 ("downtime_min", "Unplanned stoppage minutes", "Stop-start operation stresses product"),
 ("downtime_ratio", "downtime / production_duration", "Share of the run lost to stoppages"),
 ("vibration_deviation", "vibration - mean of previous 50 batches on the line", "Mechanical wear relative to the line's own baseline"),
 ("vibration", "Raw vibration level", "Absolute mechanical condition"),
 ("maintenance_age_days", "Days since last maintenance", "Maintenance overdue risk"),
 ("pressure", "Line pressure", "Process stability"),
 ("volume_deviation_pct", "(volume - median of previous 100 batches on line)/median", "Unusual batch size vs line norm"),
 ("machine_speed", "Line speed setting", "Throughput pressure"),
 ("machine_efficiency", "volume / (speed x (duration - downtime))", "Litres per unit of effective running capacity"),
 ("raw_material_ratio", "raw_material_volume / production_volume_l", "Yield / loss; abnormal ratio signals process loss or mis-measurement"),
 ("raw_material_age_h", "Hours from collection to processing", "Staler milk -> higher acidity/spoilage"),
 ("raw_material_freshness", "1/(1+age/12)", "Bounded, non-linear freshness score"),
 ("fat_percentage / protein_percentage / fat_protein_ratio / acidity", "Incoming composition", "Composition drives product spec"),
 ("raw_material_quality_score", "Composite incoming quality 0-100", "Supplier/lot quality"),
 ("hour_sin / hour_cos / day_of_week", "Cyclical time-of-day + weekday", "Fatigue / shift patterns without a raw time index"),
 ("ambient_temperature", "Plant ambient temperature", "Cooling load / hot-day effects"),
 ("supplier_hist_rej_rate", "Smoothed rejection rate of the supplier using ONLY labels reported before t", "Supplier track record"),
 ("supplier_rej_rate_30d", "Same, restricted to labels reported in the last 30 days", "Captures supplier drift (e.g. degrading supplier)"),
 ("shift_hist_rej_rate", "Smoothed shift rejection rate, labels reported in last 90 days", "Shift-level process discipline"),
 ("line_rej_rate_last200", "Smoothed rejection rate of the last 200 REPORTED batches on the line", "Current line health (rolling rejection rate)"),
 ("prev_batch_rejected", "Outcome of the most recent already-reported batch on the line", "Short-term persistence"),
 ("supplier_label_history_n", "# reported labels for the supplier", "Tells model when the supplier rate is only a prior (new supplier)"),
 ("production_line / shift / supplier_id", "Categoricals", "Group effects (OHE / frequency encoding)"),
]
