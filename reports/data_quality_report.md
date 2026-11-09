# Data Quality Report (synthetic data)

## 1. Duplicates, keys and cross-file consistency

| Check | Result |
|---|---|
| production_rows | 300250 |
| production_exact_dups | 250 |
| production_conflicting_key_dups | 0 |
| raw_material_rows | 299850 |
| raw_material_exact_dups | 0 |
| raw_material_conflicting_key_dups | 0 |
| machine_rows | 299780 |
| machine_exact_dups | 80 |
| machine_conflicting_key_dups | 0 |
| quality_rows | 300020 |
| quality_exact_dups | 120 |
| quality_conflicting_key_dups | 0 |
| batch_id_unique_in_production | False |
| batch_id_reused_values | 299247 |
| events_missing_in_raw_material | 150 |
| events_missing_in_machine | 300 |
| events_missing_in_quality | 100 |
| orphans_in_other_files | 0 |
| batch_id_mismatch_prod_vs_raw | 0 |
| rows_if_joined_on_batch_id_only | 1228941 |
| quality_ts_not_after_production | 60 |
| quality_lag_hours_median | 23.99277777777778 |
| quality_lag_hours_p95 | 28.92861111111111 |
| timestamps_monotonic_in_production_file | True |
| label_conflicts_status_vs_flag | 30 |
| events_without_label | 100 |

**Handling.** Exact duplicates were dropped per file (no conflicting duplicate event_ids exist). `batch_id` is reused after system resets, so a join on `batch_id` would multiply rows (see `rows_if_joined_on_batch_id_only`); every join uses the globally unique `event_id` with `validate='1:1'`. Events absent from raw_material / machine files are **kept** (features become NaN -> median-imputed + missing indicators). Events without a quality result or with a label conflict are kept in the table but excluded as targets (`label_ok=False`).

## 2. Invalid values (set to NaN, not deleted)

| Column | Invalid count |
|---|---|
| production_volume_l | 40 |
| production_duration_min | 21 |
| ambient_temperature | 25 |
| fat_percentage | 15 |
| protein_percentage | 12 |
| machine_temperature | 149 |
| pressure | 30 |

Impossible readings (e.g. -999 sentinel, 0 deg C pasteurisation temperature, protein x10 typos) are treated as sensor/entry faults -> NaN, so the pipeline imputes them and a missing-indicator records that they happened.

## 3. Null rates after cleaning (%)

|                            |   null_% |
|:---------------------------|---------:|
| production_volume_l        |     0.01 |
| production_duration_min    |     0.01 |
| machine_speed              |     0    |
| ambient_temperature        |     0.01 |
| raw_material_volume        |     0.05 |
| raw_material_age           |     0.05 |
| fat_percentage             |     1.22 |
| protein_percentage         |     1.28 |
| acidity                    |     0.86 |
| raw_material_quality_score |     0.05 |
| machine_temperature        |     0.15 |
| pressure                   |     1.1  |
| downtime                   |     0.1  |
| maintenance_age_days       |     0.1  |
| vibration                  |     2.67 |
| moisture_percentage        |    15.9  |
| supplier_id                |     0    |

## 4. Outliers (flagged, retained)

| column                     |   IQR1.5_outlier_% |   IQR3_extreme_% |   rejection_rate_among_extremes_% |
|:---------------------------|-------------------:|-----------------:|----------------------------------:|
| production_volume_l        |               0.32 |            0.076 |                              7.49 |
| production_duration_min    |               1.71 |            0.466 |                             11.59 |
| machine_speed              |               0.1  |            0     |                            nan    |
| ambient_temperature        |               0    |            0     |                            nan    |
| raw_material_volume        |               0.34 |            0.076 |                              7.49 |
| raw_material_age           |               3.87 |            0.613 |                             51.55 |
| fat_percentage             |               0.56 |            0     |                            nan    |
| protein_percentage         |               0.86 |            0     |                            nan    |
| acidity                    |               3.87 |            0.788 |                             46.82 |
| raw_material_quality_score |               6.06 |            1.777 |                             39.4  |
| machine_temperature        |               1.8  |            0.155 |                             51.61 |
| pressure                   |               0.7  |            0     |                              0    |
| downtime                   |              10.63 |            4.575 |                             11.28 |
| maintenance_age_days       |               0    |            0     |                            nan    |
| vibration                  |               1.91 |            0.353 |                             31.23 |
| moisture_percentage        |               3.71 |            0.68  |                            100    |

Extreme values (long downtimes, vibration spikes, off-setpoint temperatures) are physically plausible process events and are often precisely the rejection drivers, so they are retained. Only physically impossible values were nulled.

## 5. Timestamp validation

- Quality timestamp not after production timestamp: **60** rows. These (and missing ones) get a conservative label-availability time = production + p95 lag (28.9 h).
- Median reporting lag: **24.0 h**.

## 6. Missingness investigation: moisture

Moisture missing % by line x shift (rows with valid labels):

| production_line   |   evening |   morning |   night |
|:------------------|----------:|----------:|--------:|
| L1                |       6.2 |       4   |    29.6 |
| L2                |       7.3 |       4.8 |    35.6 |
| L3                |       8.8 |       5.9 |    45.4 |

Rejection rate when moisture is present vs missing (%):

| moisture_percentage   |   rejection_% |
|:----------------------|--------------:|
| False                 |          6.72 |
| True                  |          8.36 |

Missingness is strongly shift-dependent (night worst) and higher on L3, i.e. **MAR given shift/line**, not random. Because moisture is a next-day lab value it is excluded from the model anyway; if it were used, a missing-indicator and shift-aware imputation would be required.

Vibration sensor missing % by line:

| production_line   |   missing_% |
|:------------------|------------:|
| L1                |        1.56 |
| L2                |        1.53 |
| L3                |        6.06 |
