# Engineered Feature Dictionary

All features are computable at batch completion time. History features use only labels already reported at that moment.

| Feature | Definition | Business meaning |
|---|---|---|
| `machine_temp_deviation` | machine_temperature - 72 degC setpoint (signed) | Under/over-pasteurisation risk |
| `machine_temp_abs_deviation` | |machine_temp_deviation| | Any drift from setpoint hurts quality |
| `temp_variability_10` | Rolling std of machine temperature, last 10 batches on the line (incl. current) | Unstable heat treatment / control loop |
| `production_duration_min` | Batch run time | Long runs suggest slowdowns/stoppages |
| `downtime_min` | Unplanned stoppage minutes | Stop-start operation stresses product |
| `downtime_ratio` | downtime / production_duration | Share of the run lost to stoppages |
| `vibration_deviation` | vibration - mean of previous 50 batches on the line | Mechanical wear relative to the line's own baseline |
| `vibration` | Raw vibration level | Absolute mechanical condition |
| `maintenance_age_days` | Days since last maintenance | Maintenance overdue risk |
| `pressure` | Line pressure | Process stability |
| `volume_deviation_pct` | (volume - median of previous 100 batches on line)/median | Unusual batch size vs line norm |
| `machine_speed` | Line speed setting | Throughput pressure |
| `machine_efficiency` | volume / (speed x (duration - downtime)) | Litres per unit of effective running capacity |
| `raw_material_ratio` | raw_material_volume / production_volume_l | Yield / loss; abnormal ratio signals process loss or mis-measurement |
| `raw_material_age_h` | Hours from collection to processing | Staler milk -> higher acidity/spoilage |
| `raw_material_freshness` | 1/(1+age/12) | Bounded, non-linear freshness score |
| `fat_percentage / protein_percentage / fat_protein_ratio / acidity` | Incoming composition | Composition drives product spec |
| `raw_material_quality_score` | Composite incoming quality 0-100 | Supplier/lot quality |
| `hour_sin / hour_cos / day_of_week` | Cyclical time-of-day + weekday | Fatigue / shift patterns without a raw time index |
| `ambient_temperature` | Plant ambient temperature | Cooling load / hot-day effects |
| `supplier_hist_rej_rate` | Smoothed rejection rate of the supplier using ONLY labels reported before t | Supplier track record |
| `supplier_rej_rate_30d` | Same, restricted to labels reported in the last 30 days | Captures supplier drift (e.g. degrading supplier) |
| `shift_hist_rej_rate` | Smoothed shift rejection rate, labels reported in last 90 days | Shift-level process discipline |
| `line_rej_rate_last200` | Smoothed rejection rate of the last 200 REPORTED batches on the line | Current line health (rolling rejection rate) |
| `prev_batch_rejected` | Outcome of the most recent already-reported batch on the line | Short-term persistence |
| `supplier_label_history_n` | # reported labels for the supplier | Tells model when the supplier rate is only a prior (new supplier) |
| `production_line / shift / supplier_id` | Categoricals | Group effects (OHE / frequency encoding) |

**Not built:** moisture deviation - moisture is a lab result reported ~24h after prediction time (see `reports/leakage_audit.md`).
