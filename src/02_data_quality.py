"""Section 4 - data quality audit. Writes reports/data_quality_report.md"""
import pandas as pd, numpy as np
from common import REP, PROC
from cleaning import clean_and_merge, VALID

m, log = clean_and_merge(verbose=False)
m.to_pickle(PROC / "master_clean.pkl")

num_cols = list(VALID.keys())
nulls = m[num_cols + ["supplier_id"]].isna().mean().mul(100).round(2).rename("null_%")

# outliers (IQR 1.5 / 3 rule) - flagged, NOT removed
rows = []
for c in num_cols:
    s = m[c].dropna(); q1, q3 = s.quantile([.25, .75]); i = q3 - q1
    mild = ((s < q1 - 1.5*i) | (s > q3 + 1.5*i)).mean()*100
    ext = ((s < q1 - 3*i) | (s > q3 + 3*i)).mean()*100
    rej_out = m.loc[s.index[(s < q1 - 3*i) | (s > q3 + 3*i)], "rejected"].mean()
    rows.append((c, round(mild, 2), round(ext, 3), round(rej_out*100, 2) if rej_out == rej_out else np.nan))
out = pd.DataFrame(rows, columns=["column", "IQR1.5_outlier_%", "IQR3_extreme_%", "rejection_rate_among_extremes_%"])

# missingness of moisture by line x shift
lab = m[m.label_ok]
miss_ls = lab.assign(m_miss=lab.moisture_percentage.isna()).pivot_table(index="production_line", columns="shift", values="m_miss", aggfunc="mean").mul(100).round(1)
miss_rej = lab.groupby(lab.moisture_percentage.isna()).rejected.mean().mul(100).round(2)
vib_line = m.groupby("production_line").vibration.apply(lambda s: s.isna().mean()*100).round(2)

with open(REP / "data_quality_report.md", "w") as f:
    f.write("# Data Quality Report (synthetic data)\n\n")
    f.write("## 1. Duplicates, keys and cross-file consistency\n\n| Check | Result |\n|---|---|\n")
    for k, v in log.items():
        if k != "invalid_values_set_to_nan": f.write(f"| {k} | {v} |\n")
    f.write("\n**Handling.** Exact duplicates were dropped per file (no conflicting duplicate event_ids exist). "
            "`batch_id` is reused after system resets, so a join on `batch_id` would multiply rows "
            f"(see `rows_if_joined_on_batch_id_only`); every join uses the globally unique `event_id` with `validate='1:1'`. "
            "Events absent from raw_material / machine files are **kept** (features become NaN -> median-imputed + missing indicators). "
            "Events without a quality result or with a label conflict are kept in the table but excluded as targets (`label_ok=False`).\n\n")
    f.write("## 2. Invalid values (set to NaN, not deleted)\n\n| Column | Invalid count |\n|---|---|\n")
    for k, v in log["invalid_values_set_to_nan"].items():
        if v: f.write(f"| {k} | {v} |\n")
    f.write("\nImpossible readings (e.g. -999 sentinel, 0 deg C pasteurisation temperature, protein x10 typos) "
            "are treated as sensor/entry faults -> NaN, so the pipeline imputes them and a missing-indicator records that they happened.\n\n")
    f.write("## 3. Null rates after cleaning (%)\n\n" + nulls.to_frame().to_markdown() + "\n\n")
    f.write("## 4. Outliers (flagged, retained)\n\n" + out.to_markdown(index=False) + "\n\n"
            "Extreme values (long downtimes, vibration spikes, off-setpoint temperatures) are physically plausible process "
            "events and are often precisely the rejection drivers, so they are retained. Only physically impossible values were nulled.\n\n")
    f.write("## 5. Timestamp validation\n\n"
            f"- Quality timestamp not after production timestamp: **{log['quality_ts_not_after_production']}** rows. "
            f"These (and missing ones) get a conservative label-availability time = production + p95 lag ({log['quality_lag_hours_p95']:.1f} h).\n"
            f"- Median reporting lag: **{log['quality_lag_hours_median']:.1f} h**.\n\n")
    f.write("## 6. Missingness investigation: moisture\n\nMoisture missing % by line x shift (rows with valid labels):\n\n"
            + miss_ls.to_markdown() + "\n\n"
            f"Rejection rate when moisture is present vs missing (%):\n\n" + miss_rej.to_frame("rejection_%").to_markdown() + "\n\n"
            "Missingness is strongly shift-dependent (night worst) and higher on L3, i.e. **MAR given shift/line**, not random. "
            "Because moisture is a next-day lab value it is excluded from the model anyway; if it were used, a missing-indicator and "
            "shift-aware imputation would be required.\n\n"
            "Vibration sensor missing % by line:\n\n" + vib_line.to_frame("missing_%").to_markdown() + "\n")
print(open(REP / "data_quality_report.md").read()[:3500])
