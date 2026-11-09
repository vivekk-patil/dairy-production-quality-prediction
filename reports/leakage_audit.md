# Leakage Audit

**Prediction time** = batch completion (`production_timestamp`). Anything first known after that is off-limits.

## Column ledger

| Column(s) | Decision | Reason |
|---|---|---|
| event_id | **EXCLUDED** | Identifier; ordering is time-correlated; no business meaning |
| batch_number / batch_id | **EXCLUDED** | Reset counters, reused values; identifier. Tested alone it scores above chance (row E) only because it acts as a weak proxy for calendar position - exactly the data-generation artefact we do not want the model to learn |
| production_timestamp (raw) | **EXCLUDED as feature** | Used only to order/split and to derive hour/day cyclical features - prevents learning a timeline index |
| quality_test_timestamp / label_avail_ts | **EXCLUDED** | Exists only after the lab test (~24h later). Used only to decide WHEN a label may enter history features |
| moisture_percentage, moisture deviation, moisture-missing flag | **EXCLUDED** | Lab result recorded a day after prediction time; also its missingness pattern is a lab-process artefact. Demonstrated leak: row B |
| quality_status | **EXCLUDED** | Direct encoding of the target |
| rejected (target) | **EXCLUDED** | Target. Only enters via history features and only once reported |
| supplier/shift/line rejection-rate features | **INCLUDED (delay-aware)** | Use only labels with label_avail_ts <= prediction time; verified by brute force on 400 rows (0 mismatches) |
| prev_batch_rejected, line_rej_rate_last200 | **INCLUDED (delay-aware)** | Use the most recent REPORTED batches, not the physically previous batches (those are still pending) |
| rolling temp variability / vibration & volume baselines | **INCLUDED** | Trailing windows on sensor data from batches already completed at t; baselines use shift(1) |
| raw sensor / raw-material / machine columns | **INCLUDED** | Measured during or before batch completion |
| supplier_id | **INCLUDED (frequency-encoded)** | Known before production; S11 is new in the test period -> unknown-category handling |

## Rules enforced in code

1. Label history uses `label_avail_ts <= t` (reporting delay respected); conflicting / missing labels never enter history.
2. Rolling windows are trailing and chronological (`shift(1)` where the current row would otherwise leak into its own baseline).
3. Split is chronological with an **embargo** that sets aside train/val rows whose label is not yet reported at the next period's start.
4. Preprocessing is fit on the training rows only.

## Empirical evidence (HistGradientBoosting, train -> validation)

| feature set                                                                        |   val ROC-AUC |   val PR-AUC |
|:-----------------------------------------------------------------------------------|--------------:|-------------:|
| A. Leakage-safe features (the real model inputs)                                   |        0.805  |       0.3404 |
| B. + moisture deviation (lab result reported ~24h later)                           |        0.9352 |       0.7521 |
| C. + line rejection rate of last 200 batches IGNORING the 24h reporting delay      |        0.8071 |       0.3408 |
| D. + centred rolling rejection rate (includes the row's own label and future rows) |        0.8725 |       0.4309 |
| E. Identifier-only model: batch_number                                             |        0.5701 |       0.0903 |
| F. Identifier-only model: event_id order                                           |        0.5    |       0.0743 |

Rows B and D show how much a leak inflates validation scores relative to the honest model (A): the moisture leak adds roughly 0.13 ROC-AUC. Row C is nearly identical to A in this dataset: line condition persists for days, so ignoring the 24h delay changes little here - the delay-aware version is kept anyway because on real data with faster-moving states it would matter. Row E (batch_number alone) beats chance (a weak time proxy: batch counters restart at resets, so position in the counter tracks calendar position); row F (pure event order) is ~0.5. Identifiers are therefore excluded.
