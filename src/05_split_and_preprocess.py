"""Sections 7-9: chronological split, time-series CV check, feature docs, leakage-safe preprocessing."""
import joblib, numpy as np, pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from common import PROC, REP, DOCS, MODELS
from features import FEATURE_DOC
from splitting import chrono_split
from preprocessing import make_preprocessor, NON_FEATURES, LOW_CARD, HIGH_CARD

F = pd.read_pickle(PROC / "features_all.pkl")
split, (t1, t2) = chrono_split(F)
F["split"] = split
numeric_cols = [c for c in F.columns if c not in NON_FEATURES + LOW_CARD + HIGH_CARD]
D = {k: F[F.split == k] for k in ["train", "val", "test"]}

# ---------------- integrity assertions (fail loudly if anything leaks) ----------------
tr, va, te = D["train"], D["val"], D["test"]
assert tr.production_timestamp.max() < va.production_timestamp.min() and va.production_timestamp.max() < te.production_timestamp.min()
assert tr.label_avail_ts.max() <= va.production_timestamp.min(), "train labels not yet reported at validation start"
assert va.label_avail_ts.max() <= te.production_timestamp.min(), "val labels not yet reported at test start"
assert not set(NON_FEATURES) & set(numeric_cols + LOW_CARD + HIGH_CARD)
print("chronology + label-availability assertions passed")

# ---------------- fit preprocessing on TRAIN only ----------------
pre = make_preprocessor(numeric_cols)
Xtr = pre.fit_transform(tr[numeric_cols + LOW_CARD + HIGH_CARD])
Xva = pre.transform(va[numeric_cols + LOW_CARD + HIGH_CARD]); Xte = pre.transform(te[numeric_cols + LOW_CARD + HIGH_CARD])
names = pre.get_feature_names_out()
assert not np.isnan(Xtr).any() and not np.isnan(Xva).any() and not np.isnan(Xte).any()
print("transformed shapes:", Xtr.shape, Xva.shape, Xte.shape)

imp = pre.named_transformers_["num"].named_steps["impute"]
ind_cols = [numeric_cols[i] for i in imp.indicator_.features_]
unseen_val = (~va.supplier_id.isin(tr.supplier_id.unique())).sum(); unseen_te = (~te.supplier_id.isin(tr.supplier_id.unique())).sum()
freq_map = pre.named_transformers_["freq"].maps_["supplier_id"]

# train-only medians vs all-data medians (shows the imputer did not see the future)
cmp = pd.DataFrame({"train_median (used)": tr[["raw_material_age_h", "vibration", "maintenance_age_days", "line_rej_rate_last200"]].median(),
                    "test_median (NOT used)": te[["raw_material_age_h", "vibration", "maintenance_age_days", "line_rej_rate_last200"]].median()}).round(3)

# ---------------- TimeSeriesSplit inside the training period ----------------
trs = tr.sort_values("production_timestamp").reset_index(drop=True)
GAP = 1000   # rows (~2.4 days at ~410 rows/day) > max 48h reporting lag
tscv = TimeSeriesSplit(n_splits=5, gap=GAP)
cv_rows = []
for k, (a, b) in enumerate(tscv.split(trs), 1):
    assert trs.label_avail_ts.iloc[a].max() <= trs.production_timestamp.iloc[b].min()
    cv_rows.append((k, len(a), trs.production_timestamp.iloc[a].min().date(), trs.production_timestamp.iloc[a].max().date(),
                    len(b), trs.production_timestamp.iloc[b].min().date(), trs.production_timestamp.iloc[b].max().date(),
                    round(trs.rejected.iloc[b].mean() * 100, 2)))
CV = pd.DataFrame(cv_rows, columns=["fold", "n_train", "train_from", "train_to", "n_valid", "valid_from", "valid_to", "valid_reject_%"])

# ---------------- artefacts ----------------
joblib.dump(pre, MODELS / "preprocessor.joblib")
F.to_pickle(PROC / "modeling_dataset.pkl")
np.savez_compressed(PROC / "transformed_matrices.npz", Xtr=Xtr, Xva=Xva, Xte=Xte,
                    ytr=tr.rejected.values.astype(int), yva=va.rejected.values.astype(int), yte=te.rejected.values.astype(int), names=names)

S = pd.DataFrame({k: dict(rows=len(v), start=v.production_timestamp.min().date(), end=v.production_timestamp.max().date(),
                          reject_pct=round(v.rejected.mean() * 100, 2)) for k, v in D.items()}).T
cnt = F.split.value_counts()
with open(REP / "split_and_preprocessing.md", "w") as f:
    f.write("# Time-Aware Split and Preprocessing\n\n## Chronological split\n\n"
            "Rows are ordered by batch completion time; earliest 70% = train, next 15% = validation, final 15% = test. No shuffling.\n\n"
            + S.to_markdown() + "\n\n"
            f"**Embargo.** {cnt.get('embargo', 0)} rows from the last ~day of train/validation are set aside because their lab results are reported only after the next period begins; "
            f"{cnt.get('unlabeled', 0)} rows without a valid label (missing result or conflicting status) are excluded from targets. Assertions in code prove: "
            "(1) all train timestamps precede validation which precede test, (2) every training label was reported before the first validation prediction, (3) same for validation vs test.\n\n"
            "**Why chronological?** Production is autocorrelated (line condition, supplier drift, maintenance cycles). A random split would let the model see batches adjacent to test batches "
            "and overstate performance; the real deployment always predicts the future from the past. Validation is for tuning/threshold choice; the test set is untouched until the final report.\n\n"
            f"## Time-series cross-validation inside the training period\n\n`TimeSeriesSplit(n_splits=5, gap={GAP} rows)` - expanding window, validation folds always later than training folds, "
            f"gap larger than the maximum reporting delay (asserted per fold that all training labels were reported before the fold's first prediction).\n\n" + CV.to_markdown(index=False) + "\n\n"
            "## Preprocessing (ColumnTransformer + Pipeline, fit on training rows only)\n\n"
            "| Block | Columns | Treatment |\n|---|---|---|\n"
            f"| Numeric | {len(numeric_cols)} features | StandardScaler -> SimpleImputer(median, add_indicator=True) (scaling optional; disable for tree models) |\n"
            f"| Low-cardinality categorical | {', '.join(LOW_CARD)} | OneHotEncoder(handle_unknown='ignore') |\n"
            f"| High-cardinality categorical | {', '.join(HIGH_CARD)} | FrequencyEncoder (train frequencies, rare/unseen -> 0.0) |\n\n"
            f"Output matrix: {Xtr.shape[1]} columns ({len(ind_cols)} missing-indicator columns for: {', '.join(ind_cols)}).\n\n"
            f"**Unknown categories.** Supplier S11 is onboarded late: {unseen_val} validation rows and {unseen_te} test rows carry a supplier never seen in training. "
            "They receive frequency 0, historical supplier rate = overall rate (shrinkage prior) and `supplier_label_history_n` = labels reported so far, "
            "so the pipeline runs without error and the model can tell the supplier is new.\n\n"
            "**No future information in preprocessing.** Medians, scaling statistics, category frequencies and the set of indicator columns come from training rows only:\n\n"
            + cmp.to_markdown() + "\n\n"
            "Missingness itself is informative here (e.g. vibration sensor gaps on L3, sensor faults nulled during cleaning), hence indicators for every numeric column that has NaNs in training. "
            "The moisture column is not in the feature set (see leakage audit), so no moisture imputation is needed.\n")
with open(DOCS / "feature_dictionary.md", "w") as f:
    f.write("# Engineered Feature Dictionary\n\nAll features are computable at batch completion time. History features use only labels already reported at that moment.\n\n"
            "| Feature | Definition | Business meaning |\n|---|---|---|\n")
    for a, b, c in FEATURE_DOC: f.write(f"| `{a}` | {b} | {c} |\n")
    f.write("\n**Not built:** moisture deviation - moisture is a lab result reported ~24h after prediction time (see `reports/leakage_audit.md`).\n")
print(S.to_string()); print(CV.to_string(index=False)); print("indicator cols:", ind_cols); print("unseen supplier rows val/test:", unseen_val, unseen_te); print(cmp)
