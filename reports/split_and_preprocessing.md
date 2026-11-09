# Time-Aware Split and Preprocessing

## Chronological split

Rows are ordered by batch completion time; earliest 70% = train, next 15% = validation, final 15% = test. No shuffling.

|       |   rows | start      | end        |   reject_pct |
|:------|-------:|:-----------|:-----------|-------------:|
| train | 209535 | 2023-01-01 | 2024-05-26 |         6.77 |
| val   |  44563 | 2024-05-27 | 2024-09-12 |         7.43 |
| test  |  44974 | 2024-09-13 | 2024-12-31 |         7.52 |

**Embargo.** 798 rows from the last ~day of train/validation are set aside because their lab results are reported only after the next period begins; 130 rows without a valid label (missing result or conflicting status) are excluded from targets. Assertions in code prove: (1) all train timestamps precede validation which precede test, (2) every training label was reported before the first validation prediction, (3) same for validation vs test.

**Why chronological?** Production is autocorrelated (line condition, supplier drift, maintenance cycles). A random split would let the model see batches adjacent to test batches and overstate performance; the real deployment always predicts the future from the past. Validation is for tuning/threshold choice; the test set is untouched until the final report.

## Time-series cross-validation inside the training period

`TimeSeriesSplit(n_splits=5, gap=1000 rows)` - expanding window, validation folds always later than training folds, gap larger than the maximum reporting delay (asserted per fold that all training labels were reported before the fold's first prediction).

|   fold |   n_train | train_from   | train_to   |   n_valid | valid_from   | valid_to   |   valid_reject_% |
|-------:|----------:|:-------------|:-----------|----------:|:-------------|:-----------|-----------------:|
|      1 |     33925 | 2023-01-01   | 2023-03-25 |     34922 | 2023-03-27   | 2023-06-20 |             6.85 |
|      2 |     68847 | 2023-01-01   | 2023-06-17 |     34922 | 2023-06-20   | 2023-09-13 |             7.13 |
|      3 |    103769 | 2023-01-01   | 2023-09-11 |     34922 | 2023-09-13   | 2023-12-07 |             7.68 |
|      4 |    138691 | 2023-01-01   | 2023-12-05 |     34922 | 2023-12-07   | 2024-03-01 |             6.25 |
|      5 |    173613 | 2023-01-01   | 2024-02-28 |     34922 | 2024-03-01   | 2024-05-26 |             7.14 |

## Preprocessing (ColumnTransformer + Pipeline, fit on training rows only)

| Block | Columns | Treatment |
|---|---|---|
| Numeric | 31 features | StandardScaler -> SimpleImputer(median, add_indicator=True) (scaling optional; disable for tree models) |
| Low-cardinality categorical | production_line, shift | OneHotEncoder(handle_unknown='ignore') |
| High-cardinality categorical | supplier_id | FrequencyEncoder (train frequencies, rare/unseen -> 0.0) |

Output matrix: 64 columns (26 missing-indicator columns for: machine_temp_deviation, machine_temp_abs_deviation, temp_variability_10, production_duration_min, downtime_min, downtime_ratio, vibration_deviation, vibration, maintenance_age_days, pressure, volume_deviation_pct, machine_efficiency, raw_material_ratio, raw_material_age_h, raw_material_freshness, fat_percentage, protein_percentage, fat_protein_ratio, acidity, raw_material_quality_score, ambient_temperature, supplier_hist_rej_rate, supplier_rej_rate_30d, shift_hist_rej_rate, line_rej_rate_last200, prev_batch_rejected).

**Unknown categories.** Supplier S11 is onboarded late: 0 validation rows and 4212 test rows carry a supplier never seen in training. They receive frequency 0, historical supplier rate = overall rate (shrinkage prior) and `supplier_label_history_n` = labels reported so far, so the pipeline runs without error and the model can tell the supplier is new.

**No future information in preprocessing.** Medians, scaling statistics, category frequencies and the set of indicator columns come from training rows only:

|                       |   train_median (used) |   test_median (NOT used) |
|:----------------------|----------------------:|-------------------------:|
| raw_material_age_h    |                13.9   |                   13.9   |
| vibration             |                 2.57  |                    2.575 |
| maintenance_age_days  |                36     |                   38     |
| line_rej_rate_last200 |                 0.055 |                    0.065 |

Missingness itself is informative here (e.g. vibration sensor gaps on L3, sensor faults nulled during cleaning), hence indicators for every numeric column that has NaNs in training. The moisture column is not in the feature set (see leakage audit), so no moisture imputation is needed.
