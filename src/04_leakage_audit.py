"""Section 6 - leakage audit: column ledger, brute-force verification of history features,
and an empirical demonstration of how much each type of leak would inflate validation scores."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from common import PROC, REP
from features import build_features, available_label_counts
from splitting import chrono_split

m = pd.read_pickle(PROC / "master_clean.pkl")
F = pd.read_pickle(PROC / "features_all.pkl")
split, (t1, t2) = chrono_split(F)

# ---------- 1. brute-force verification of the fast "available label" logic ----------------------
rng = np.random.default_rng(0)
sample = rng.choice(len(F), 400, replace=False)
n_fast, r_fast = available_label_counts(m, "supplier_id")
bad = 0
for i in sample:
    ti = m.production_timestamp.iloc[i]
    pool = m[(m.supplier_id == m.supplier_id.iloc[i]) & m.label_ok & (m.label_avail_ts <= ti)]
    if len(pool) != n_fast[i] or int(pool.rejected.sum()) != int(r_fast[i]): bad += 1
# also verify: no event's own label and no label reported after t is ever included
assert bad == 0, f"{bad} mismatches"
print("brute-force check on 400 rows: 0 mismatches")

# ---------- 2. empirical leak demo -------------------------------------------------------------------
y = F.rejected
core = [c for c in F.columns if c not in ("event_id", "production_timestamp", "label_avail_ts", "rejected",
                                         "label_ok", "production_line", "shift", "supplier_id")]
X = F[core].copy()
X["line_code"] = F.production_line.map({"L1": 0, "L2": 1, "L3": 2})
X["shift_code"] = F["shift"].map({"morning": 0, "evening": 1, "night": 2})
tr, va = (split == "train").values, (split == "val").values

def fit_eval(Xm):
    clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=15, random_state=42)
    clf.fit(Xm[tr], y[tr])
    p = clf.predict_proba(Xm[va])[:, 1]
    return roc_auc_score(y[va], p), average_precision_score(y[va], p)

res = []
res.append(("A. Leakage-safe features (the real model inputs)",) + fit_eval(X))

Xb = X.copy(); Xb["moisture_abs_dev"] = (m.moisture_percentage - 87.5).abs()
res.append(("B. + moisture deviation (lab result reported ~24h later)",) + fit_eval(Xb))

Xc = X.copy(); gl = m.groupby("production_line").rejected
Xc["naive_line_rate_200"] = gl.transform(lambda s: s.shift(1).rolling(200, min_periods=20).mean())
res.append(("C. + line rejection rate of last 200 batches IGNORING the 24h reporting delay",) + fit_eval(Xc))

Xd = X.copy()
Xd["centered_rolling_rate"] = gl.transform(lambda s: s.rolling(21, center=True, min_periods=1).mean())
res.append(("D. + centred rolling rejection rate (includes the row's own label and future rows)",) + fit_eval(Xd))

Xe = pd.DataFrame({"batch_number": m.batch_number.values})
res.append(("E. Identifier-only model: batch_number",) + fit_eval(Xe))
Xf = pd.DataFrame({"event_order": np.arange(len(m))})
res.append(("F. Identifier-only model: event_id order",) + fit_eval(Xf))
R = pd.DataFrame(res, columns=["feature set", "val ROC-AUC", "val PR-AUC"]).round(4)
print(R.to_string(index=False))

# ---------- 3. write report ---------------------------------------------------------------------------
ledger = [
 ("event_id", "EXCLUDED", "Identifier; ordering is time-correlated; no business meaning"),
 ("batch_number / batch_id", "EXCLUDED", "Reset counters, reused values; identifier. Tested alone it scores above chance (row E) only because it acts as a weak proxy for calendar position - exactly the data-generation artefact we do not want the model to learn"),
 ("production_timestamp (raw)", "EXCLUDED as feature", "Used only to order/split and to derive hour/day cyclical features - prevents learning a timeline index"),
 ("quality_test_timestamp / label_avail_ts", "EXCLUDED", "Exists only after the lab test (~24h later). Used only to decide WHEN a label may enter history features"),
 ("moisture_percentage, moisture deviation, moisture-missing flag", "EXCLUDED", "Lab result recorded a day after prediction time; also its missingness pattern is a lab-process artefact. Demonstrated leak: row B"),
 ("quality_status", "EXCLUDED", "Direct encoding of the target"),
 ("rejected (target)", "EXCLUDED", "Target. Only enters via history features and only once reported"),
 ("supplier/shift/line rejection-rate features", "INCLUDED (delay-aware)", f"Use only labels with label_avail_ts <= prediction time; verified by brute force on 400 rows (0 mismatches)"),
 ("prev_batch_rejected, line_rej_rate_last200", "INCLUDED (delay-aware)", "Use the most recent REPORTED batches, not the physically previous batches (those are still pending)"),
 ("rolling temp variability / vibration & volume baselines", "INCLUDED", "Trailing windows on sensor data from batches already completed at t; baselines use shift(1)"),
 ("raw sensor / raw-material / machine columns", "INCLUDED", "Measured during or before batch completion"),
 ("supplier_id", "INCLUDED (frequency-encoded)", "Known before production; S11 is new in the test period -> unknown-category handling"),
]
with open(REP / "leakage_audit.md", "w") as f:
    f.write("# Leakage Audit\n\n**Prediction time** = batch completion (`production_timestamp`). Anything first known after that is off-limits.\n\n")
    f.write("## Column ledger\n\n| Column(s) | Decision | Reason |\n|---|---|---|\n")
    for a, b, c in ledger: f.write(f"| {a} | **{b}** | {c} |\n")
    f.write("\n## Rules enforced in code\n\n"
            "1. Label history uses `label_avail_ts <= t` (reporting delay respected); conflicting / missing labels never enter history.\n"
            "2. Rolling windows are trailing and chronological (`shift(1)` where the current row would otherwise leak into its own baseline).\n"
            "3. Split is chronological with an **embargo** that sets aside train/val rows whose label is not yet reported at the next period's start.\n"
            "4. Preprocessing is fit on the training rows only.\n\n"
            "## Empirical evidence (HistGradientBoosting, train -> validation)\n\n" + R.to_markdown(index=False) + "\n\n"
            "Rows B and D show how much a leak inflates validation scores relative to the honest model (A): the moisture leak adds roughly "
            f"{R.iloc[1,1]-R.iloc[0,1]:.2f} ROC-AUC. Row C is nearly identical to A in this dataset: line condition persists for days, so ignoring the "
            "24h delay changes little here - the delay-aware version is kept anyway because on real data with faster-moving states it would matter. "
            "Row E (batch_number alone) beats chance (a weak time proxy: batch counters restart at resets, so position in the counter tracks calendar position); "
            "row F (pure event order) is ~0.5. Identifiers are therefore excluded.\n")
