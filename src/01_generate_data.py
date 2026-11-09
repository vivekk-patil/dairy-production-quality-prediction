"""
Synthetic data generator - Dairy Production Batch Quality Rejection.

!! ALL DATA PRODUCED HERE IS SYNTHETIC. It is not factory data and must never be
   presented as such. Relationships are invented for demonstration purposes. !!

Design notes
------------
* One row per production event (batch run). `event_id` is the globally unique key.
* `batch_number` is a counter that is RESET several times, so `batch_id` is NOT unique.
  Joining on `batch_id` would silently create many-to-many joins -> always join on event_id.
* `production_timestamp` = batch COMPLETION time; this is also the prediction time.
* Quality results (moisture, status) are lab results recorded ~24h later.
* Moisture is a lab measurement tied (unrealistically cleanly) to rejection ->
  a deliberate leakage trap. It is NOT available at prediction time.
* Deliberately dirty: duplicates, sentinel/invalid values, orphan/missing events across
  files, reversed timestamps, label conflicts, night-shift-heavy missing moisture.
"""
import numpy as np, pandas as pd
from common import RAW, DOCS, SEED

N = 300_000
rng = np.random.default_rng(SEED)

START = pd.Timestamp("2023-01-01")
END = pd.Timestamp("2024-12-31 23:59:59")
span_s = (END - START).total_seconds()
ts = START + pd.to_timedelta(np.sort(rng.uniform(0, span_s, N)), unit="s")
ts = ts.round("s")
idx = np.arange(N)
frac = idx / N

event_id = np.array([f"E{i+1:07d}" for i in idx])

# ---- batch numbers reused after system resets ----------------------------------------------
reset_pts = np.sort(rng.choice(np.arange(int(.15*N), int(.9*N)), 4, replace=False))
epoch = np.searchsorted(reset_pts, idx, side="right")
epoch_start = np.concatenate([[0], reset_pts])[epoch]
batch_number = idx - epoch_start + 1
batch_id = np.array([f"B{b:06d}" for b in batch_number])

# ---- categorical context -------------------------------------------------------------------
line = rng.choice(["L1", "L2", "L3"], N, p=[.40, .35, .25])
hour = ts.hour.values
shift = np.where((hour >= 6) & (hour < 14), "morning",
         np.where((hour >= 14) & (hour < 22), "evening", "night"))

sups = [f"S{i:02d}" for i in range(1, 11)]
sup_p = np.array([.16, .14, .12, .12, .10, .09, .09, .07, .06, .05]); sup_p /= sup_p.sum()
supplier = rng.choice(sups, N, p=sup_p)
new_sup = (frac > .88) & (rng.random(N) < .12)      # S11 onboarded late -> unseen in training
supplier = np.where(new_sup, "S11", supplier)

sup_eff = dict(S01=0, S02=-.1, S03=.6, S04=.1, S05=-.2, S06=0, S07=.5, S08=-.3, S09=.2, S10=.05, S11=.3)
sup_age = dict(S01=14, S02=12, S03=26, S04=18, S05=10, S06=16, S07=24, S08=11, S09=20, S10=17, S11=15)
sup_fat = dict(S01=3.8, S02=3.9, S03=3.6, S04=3.8, S05=4.0, S06=3.8, S07=3.7, S08=4.1, S09=3.7, S10=3.8, S11=3.8)
sup_e = np.array([sup_eff[s] for s in supplier])
sup_e += np.where((supplier == "S07") & (frac > .5), .45, 0)     # S07 degrades in the 2nd half (drift)

# ---- production ---------------------------------------------------------------------------
vol_mean = dict(L1=5000, L2=4200, L3=6000); spd_mean = dict(L1=120, L2=100, L3=140)
vol = np.array([vol_mean[l] for l in line]) * rng.lognormal(0, .08, N)
o = rng.random(N) < .0015; vol[o] *= rng.choice([.3, 2.5], o.sum())
spd = np.array([spd_mean[l] for l in line]) + rng.normal(0, 6, N)
o = rng.random(N) < .001; spd[o] *= .3

night = shift == "night"
downtime = np.where(rng.random(N) < .55, 0, rng.exponential(8, N)) * np.where(night, 1.3, 1.0)
o = rng.random(N) < .002; downtime[o] = rng.uniform(60, 240, o.sum())
duration = vol / (spd * .35) * rng.lognormal(0, .07, N) + downtime

doy = ts.dayofyear.values
amb = 29 + 5*np.sin(2*np.pi*(doy-120)/365) + 3*np.sin(2*np.pi*(hour-9)/24) + rng.normal(0, 1.5, N)

# ---- raw material -------------------------------------------------------------------------
raw_vol = vol * rng.uniform(1.01, 1.06, N)
age_mu = np.array([sup_age[s] for s in supplier])
age = np.clip(rng.gamma(3, age_mu/3), 1, 72)
fat = np.clip(rng.normal([sup_fat[s] for s in supplier], .25), 2.5, 5.5)
protein = np.clip(rng.normal(3.3, .15, N), 2.7, 4.0)
acid = rng.normal(.145, .01, N) + .0015*np.clip(age-20, 0, None)
rm_score = np.clip(95 - .55*age - 400*np.abs(acid-.145) + 6*(fat-3.8) + rng.normal(0, 3, N), 0, 100)

# ---- machine ------------------------------------------------------------------------------
day = (ts - START).days.values
period = dict(L1=75, L2=60, L3=90); off = dict(L1=10, L2=25, L3=40)
maint = np.array([(d + off[l]) % period[l] for d, l in zip(day, line)], float)
maint = np.clip(np.round(maint + rng.normal(0, 2, N)), 0, None)
t_bias = dict(L1=0, L2=.2, L3=-.15)
mtemp = 72 + np.array([t_bias[l] for l in line]) + .04*(amb-30) + rng.normal(0, 1, N) * (.6 + .01*maint)
o = rng.random(N) < .002; mtemp[o] += rng.choice([-5, 5], o.sum())
pressure = rng.normal(2.5, .15, N) + .002*maint
vib = rng.lognormal(np.log(2.2), .2, N) * (1 + .004*maint) * np.where(line == "L3", 1.1, 1.0)
o = rng.random(N) < .003; vib[o] *= 3

# ---- rejection (latent logit) ---------------------------------------------------------------
def z(x): return (x - x.mean()) / x.std()
tdev = np.abs(mtemp - 72)
line_eff = np.array([dict(L1=0, L2=-.15, L3=.35)[l] for l in line])
shift_eff = np.array([dict(morning=-.1, evening=0, night=.3)[s] for s in shift])
# slowly varying line condition (6h blocks, AR(1)) -> recent rejections carry signal
nb = int(span_s // (6*3600)) + 2
state = {l: np.zeros(nb) for l in ["L1", "L2", "L3"]}
for l in state:
    for b in range(1, nb): state[l][b] = .95*state[l][b-1] + rng.normal(0, .15)
blk = ((ts - START).total_seconds().values // (6*3600)).astype(int)
line_state = np.array([state[l][b] for l, b in zip(line, blk)])

logit = (.45*z(tdev) + .30*z(np.log1p(downtime)) + .30*z(vib) + .35*z(age) - .30*z(rm_score)
         + .20*z(maint) + .12*z(amb) + .20*z(maint)*(line == "L3") + .25*z(tdev)*night
         + sup_e + line_eff + shift_eff + line_state + rng.normal(0, .8, N))
lo, hi = -10, 5
for _ in range(60):
    mid = (lo+hi)/2
    if (1/(1+np.exp(-(logit+mid)))).mean() > .07: hi = mid
    else: lo = mid
p_rej = 1/(1+np.exp(-(logit+mid)))
rejected = (rng.random(N) < p_rej).astype(int)

# ---- quality (lab) --------------------------------------------------------------------------
moist = 87.5 + rng.normal(0, .3, N) + rejected * rng.choice([-1, 1], N) * np.abs(rng.normal(.9, .4, N))
miss_p = np.array([dict(morning=.04, evening=.06, night=.30)[s] for s in shift]) \
         * np.array([dict(L1=1, L2=1.2, L3=1.5)[l] for l in line])
moist_obs = np.where(rng.random(N) < np.clip(miss_p, 0, .8), np.nan, moist)
lag_h = np.clip(rng.normal(24, 3, N), 12, 48)
q_ts = (ts + pd.to_timedelta(lag_h, unit="h")).round("s")
status = np.where(rejected == 1, "REJECTED", "ACCEPTED")

# ---- assemble tables ------------------------------------------------------------------------
production = pd.DataFrame(dict(event_id=event_id, batch_number=batch_number, batch_id=batch_id,
    production_timestamp=ts, production_line=line, shift=shift, production_volume_l=vol.round(1),
    production_duration_min=duration.round(1), machine_speed=spd.round(1), ambient_temperature=amb.round(2)))
raw_material = pd.DataFrame(dict(event_id=event_id, batch_id=batch_id, supplier_id=supplier,
    raw_material_volume=raw_vol.round(1), raw_material_age=age.round(1), fat_percentage=fat.round(2),
    protein_percentage=protein.round(2), acidity=acid.round(4), raw_material_quality_score=rm_score.round(1)))
machine = pd.DataFrame(dict(event_id=event_id, batch_id=batch_id, machine_temperature=mtemp.round(2),
    pressure=pressure.round(3), downtime=downtime.round(1), maintenance_age_days=maint,
    vibration=vib.round(3)))
quality = pd.DataFrame(dict(event_id=event_id, batch_id=batch_id, quality_test_timestamp=q_ts,
    moisture_percentage=np.round(moist_obs, 2), quality_status=status, rejected=rejected))

# ---- inject realistic data-quality problems --------------------------------------------------
def pick(n, size=N): return rng.choice(size, n, replace=False)

for df, col, pr in [(raw_material, "fat_percentage", .012), (raw_material, "protein_percentage", .012),
                    (raw_material, "acidity", .008), (machine, "pressure", .01)]:
    df.loc[rng.random(N) < pr, col] = np.nan
machine.loc[rng.random(N) < np.where(line == "L3", .06, .015), "vibration"] = np.nan

i = pick(40);  production.loc[i, "production_volume_l"] = -production.loc[i, "production_volume_l"].abs()
production.loc[pick(25), "ambient_temperature"] = -999
production.loc[pick(20), "production_duration_min"] = 0
raw_material.loc[pick(15), "fat_percentage"] = -1
i = pick(12); raw_material.loc[i, "protein_percentage"] = raw_material.loc[i, "protein_percentage"] * 10
machine.loc[pick(150), "machine_temperature"] = 0
machine.loc[pick(30), "pressure"] = -1

bad = pick(60)
quality.loc[bad, "quality_test_timestamp"] = production.loc[bad, "production_timestamp"] - pd.Timedelta(hours=3)
conf = pick(30)
quality.loc[conf, "quality_status"] = np.where(quality.loc[conf, "rejected"] == 1, "ACCEPTED", "REJECTED")

raw_material = raw_material.drop(pick(150)).reset_index(drop=True)
machine = machine.drop(pick(300)).reset_index(drop=True)
quality = quality.drop(pick(100)).reset_index(drop=True)

production = pd.concat([production, production.sample(250, random_state=1)], ignore_index=True)
machine = pd.concat([machine, machine.sample(80, random_state=2)], ignore_index=True)
quality = pd.concat([quality, quality.sample(120, random_state=3)], ignore_index=True)

raw_material = raw_material.sample(frac=1, random_state=4).reset_index(drop=True)
machine = machine.sample(frac=1, random_state=5).reset_index(drop=True)
quality = quality.sample(frac=1, random_state=6).reset_index(drop=True)

production.to_csv(RAW / "production.csv", index=False)
raw_material.to_csv(RAW / "raw_material.csv", index=False)
machine.to_csv(RAW / "machine.csv", index=False)
quality.to_csv(RAW / "quality.csv", index=False)

# ---- data dictionary ------------------------------------------------------------------------
D = [
("production.csv","event_id","string","Globally unique event key (join key across all files)"),
("production.csv","batch_number","int","Plant batch counter; RESET several times -> NOT unique. Identifier only"),
("production.csv","batch_id","string","'B'+batch_number; reused after resets -> NOT unique. Never join on it alone"),
("production.csv","production_timestamp","datetime","Batch completion time = prediction time"),
("production.csv","production_line","category","L1 / L2 / L3"),
("production.csv","shift","category","morning 06-14, evening 14-22, night 22-06 (by completion hour)"),
("production.csv","production_volume_l","float","Litres of finished product"),
("production.csv","production_duration_min","float","Run time incl. downtime (minutes)"),
("production.csv","machine_speed","float","Line speed setting (rpm)"),
("production.csv","ambient_temperature","float","Plant ambient temperature (deg C)"),
("raw_material.csv","event_id","string","Join key"),
("raw_material.csv","batch_id","string","Non-unique batch label (redundant with production)"),
("raw_material.csv","supplier_id","category","S01-S10 (+S11 onboarded late in the timeline)"),
("raw_material.csv","raw_material_volume","float","Litres of raw milk consumed"),
("raw_material.csv","raw_material_age","float","Hours between collection and processing"),
("raw_material.csv","fat_percentage","float","Fat content (%)"),
("raw_material.csv","protein_percentage","float","Protein content (%)"),
("raw_material.csv","acidity","float","Titratable acidity (% lactic acid)"),
("raw_material.csv","raw_material_quality_score","float","Composite incoming-quality score 0-100"),
("machine.csv","event_id","string","Join key"),
("machine.csv","batch_id","string","Non-unique batch label"),
("machine.csv","machine_temperature","float","Pasteurisation temperature (deg C); setpoint 72"),
("machine.csv","pressure","float","Line pressure (bar)"),
("machine.csv","downtime","float","Unplanned stoppage minutes during the batch"),
("machine.csv","maintenance_age_days","float","Days since last maintenance of the line"),
("machine.csv","vibration","float","Vibration (mm/s); sensor missing more often on L3"),
("quality.csv","event_id","string","Join key"),
("quality.csv","batch_id","string","Non-unique batch label"),
("quality.csv","quality_test_timestamp","datetime","Lab result time (~24h after production) - determines WHEN the label becomes known"),
("quality.csv","moisture_percentage","float","Lab moisture (%); LATER than prediction time -> not a valid predictor; missing mostly at night"),
("quality.csv","quality_status","category","ACCEPTED / REJECTED (lab verdict)"),
("quality.csv","rejected","int","TARGET: 1 = rejected, 0 = accepted"),
]
dd = pd.DataFrame(D, columns=["file", "column", "dtype", "description"])
dd.to_csv(DOCS / "data_dictionary.csv", index=False)
with open(DOCS / "data_dictionary.md", "w") as f:
    f.write("# Data Dictionary (SYNTHETIC DATA)\n\n> All data was generated by `src/01_generate_data.py` (seed=42). "
            "It is synthetic and must not be presented as real factory data.\n\n")
    for fn, g in dd.groupby("file", sort=False):
        f.write(f"## {fn}\n\n| Column | Type | Description |\n|---|---|---|\n")
        for _, r in g.iterrows(): f.write(f"| `{r.column}` | {r.dtype} | {r.description} |\n")
        f.write("\n")
    f.write("## Known injected data-quality issues (for the cleaning step to discover)\n\n"
            "- Batch numbers reset 4 times (batch_id duplicates); event_id is the only safe key.\n"
            "- Exact duplicate rows in production / machine / quality.\n"
            "- Sentinel / impossible values (-999 ambient, negative volumes, 0 deg C temperature, protein x10, ...).\n"
            "- Events missing from raw_material / machine / quality files.\n"
            "- Quality timestamps earlier than production; label conflicts between `quality_status` and `rejected`.\n"
            "- Moisture missing ~4-6% in morning/evening vs ~30%+ at night; vibration missing more on L3.\n"
            "- Supplier S11 appears only near the end of the timeline; S07 degrades in the second half (drift).\n")
print("rows:", len(production), len(raw_material), len(machine), len(quality), "| reject rate:", rejected.mean().round(4))
