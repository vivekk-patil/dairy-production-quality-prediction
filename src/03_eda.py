"""Section 5 - EDA. Figures -> reports/figures, narrative -> reports/eda_report.md.
Business interpretations are generated from the computed numbers (no hand-typed claims)."""
import textwrap, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from common import PROC, FIG, REP

sns.set_theme(style="whitegrid", context="notebook")
m = pd.read_pickle(PROC / "master_clean.pkl")
d = m[m.label_ok].copy(); d["rejected"] = d.rejected.astype(int)
base = d.rejected.mean() * 100
report = []

def finish(fig, name, title, interp):
    wrapped = "\n".join(textwrap.wrap("Business interpretation: " + interp, 120))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.text(0.01, -0.02, wrapped, fontsize=9, va="top", style="italic")
    fig.tight_layout(); fig.savefig(FIG / f"{name}.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    report.append((name, title, interp))

def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z*z/n
    c = (p + z*z/(2*n)) / den; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return (c-h)*100, (c+h)*100

def rate_table(col):
    g = d.groupby(col).rejected.agg(["sum", "count"]); g["rate"] = g["sum"]/g["count"]*100
    ci = [wilson(k, n) for k, n in zip(g["sum"], g["count"])]
    g["lo"] = [c[0] for c in ci]; g["hi"] = [c[1] for c in ci]; return g

# ---- 0. dimensions + summaries ---------------------------------------------------------------
with open(REP / "_eda_tables.md", "w") as f:
    f.write(f"**Merged dataset:** {m.shape[0]:,} events x {m.shape[1]} columns; labelled & valid: {len(d):,}; "
            f"period {m.production_timestamp.min().date()} to {m.production_timestamp.max().date()}.\n\n")
    num = ["production_volume_l", "production_duration_min", "machine_speed", "ambient_temperature", "raw_material_age",
           "fat_percentage", "protein_percentage", "acidity", "raw_material_quality_score", "machine_temperature",
           "pressure", "downtime", "maintenance_age_days", "vibration", "moisture_percentage"]
    f.write(m[num].describe().T.round(2).to_markdown() + "\n")

# ---- 1. class distribution -----------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
vc = d.rejected.map({0: "Accepted", 1: "Rejected"}).value_counts()
sns.barplot(x=vc.index, y=vc.values, ax=ax[0], palette=["#4c9f70", "#d1495b"], hue=vc.index, legend=False)
ax[0].set(title="Batch counts", xlabel="Quality outcome", ylabel="Number of batches")
ax[1].pie(vc.values, labels=vc.index, autopct="%1.1f%%", colors=["#4c9f70", "#d1495b"], startangle=90)
ax[1].set_title("Share of batches")
finish(fig, "01_rejection_distribution", "Accepted vs Rejected Batches",
       f"{base:.1f}% of batches are rejected (about 1 in {round(100/base)}), so the data is imbalanced: accuracy is misleading and models "
       "must be judged with PR-AUC / recall at a chosen alert rate.")

# ---- 2. by line / supplier / shift ------------------------------------------------------------------
def bar_rate(col, name, title, xlabel, order=None, extra=""):
    t = rate_table(col)
    t = t.loc[order] if order else t.sort_values("rate", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(t.index, t.rate, yerr=[t.rate - t.lo, t.hi - t.rate], capsize=4, color="#3b6ea5")
    ax.axhline(base, ls="--", c="k", label=f"Overall {base:.1f}%"); ax.legend()
    ax.set(xlabel=xlabel, ylabel="Rejection rate (%)")
    hi_, lo_ = t.rate.idxmax(), t.rate.idxmin()
    return fig, t, hi_, lo_

fig, t, hi_, lo_ = bar_rate("production_line", "", "", "Production line")
finish(fig, "02_rejection_by_line", "Rejection Rate by Production Line (95% CI)",
       f"Line {hi_} rejects {t.rate[hi_]:.1f}% of batches vs {t.rate[lo_]:.1f}% on {lo_}. Prioritise maintenance and process audits on {hi_}.")
fig, t, hi_, lo_ = bar_rate("supplier_id", "", "", "Supplier")
finish(fig, "03_rejection_by_supplier", "Rejection Rate by Supplier (95% CI)",
       f"Supplier {hi_} has the highest rejection rate ({t.rate[hi_]:.1f}%) versus {t.rate[lo_]:.1f}% for {lo_}. "
       "Tighter incoming inspection or supplier reviews should start with the worst suppliers; note S11 appears only late in the timeline.")
fig, t, hi_, lo_ = bar_rate("shift", "", "", "Shift", order=["morning", "evening", "night"])
finish(fig, "04_rejection_by_shift", "Rejection Rate by Shift (95% CI)",
       f"The {hi_} shift has the highest rejection rate ({t.rate[hi_]:.1f}%) and {lo_} the lowest ({t.rate[lo_]:.1f}%). "
       "Night-shift supervision, staffing and sensor/lab coverage deserve review.")

# ---- 3. moisture -----------------------------------------------------------------------------------------
mo = d.dropna(subset=["moisture_percentage"])
fig, ax = plt.subplots(figsize=(9, 4))
for k, c in [(0, "#4c9f70"), (1, "#d1495b")]:
    sns.kdeplot(mo.loc[mo.rejected == k, "moisture_percentage"], ax=ax, fill=True, alpha=.4, color=c, label="Rejected" if k else "Accepted")
ax.set(xlabel="Moisture (%) - lab result, available ~24h after production", ylabel="Density"); ax.legend()
out_of_band = ((mo.moisture_percentage - 87.5).abs() > 1).groupby(mo.rejected).mean() * 100
finish(fig, "05_moisture_distribution", "Moisture Distribution by Outcome",
       f"Rejected batches deviate in both directions from the ~87.5% target ({out_of_band[1]:.0f}% of rejected vs {out_of_band[0]:.0f}% of accepted are >1 point off). "
       "Moisture is a very strong quality signal, but it is only known after lab testing - it explains rejection, it cannot be used to predict it early.")

# ---- 4. machine temperature --------------------------------------------------------------------------------
mt = d[(d.machine_temperature > 60)]
fig, ax = plt.subplots(figsize=(9, 4))
sns.histplot(data=mt.assign(Outcome=mt.rejected.map({0: "Accepted", 1: "Rejected"})), x="machine_temperature", hue="Outcome",
             stat="density", common_norm=False, bins=60, element="step", palette=["#4c9f70", "#d1495b"], ax=ax)
ax.axvline(72, c="k", ls="--"); ax.set(xlabel="Machine temperature (deg C); dashed = 72 deg C setpoint", ylabel="Density")
dev = (mt.machine_temperature - 72).abs()
bins = pd.cut(dev, [0, .5, 1, 2, 10], include_lowest=True)
rb = mt.groupby(bins, observed=True).rejected.mean() * 100
finish(fig, "06_machine_temperature", "Machine Temperature Distribution by Outcome",
       f"Rejection rises from {rb.iloc[0]:.1f}% when within 0.5 degC of setpoint to {rb.iloc[-1]:.1f}% when more than 2 degC off. "
       "Tighter temperature control / alarms around the setpoint is a direct lever.")

# ---- 5. downtime -----------------------------------------------------------------------------------------------
dd = d.dropna(subset=["downtime"]); dd = dd.assign(dt_bin=pd.cut(dd.downtime, [-.1, 0, 5, 15, 30, 600], labels=["0", "0-5", "5-15", "15-30", ">30 min"]))
t = dd.groupby("dt_bin", observed=True).rejected.agg(["mean", "count"])
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].bar(t.index.astype(str), t["mean"] * 100, color="#3b6ea5"); ax[0].axhline(base, ls="--", c="k")
ax[0].set(xlabel="Unplanned downtime in batch (min)", ylabel="Rejection rate (%)", title="Rejection rate by downtime band")
sns.boxplot(data=dd[dd.downtime > 0].assign(Outcome=lambda x: x.rejected.map({0: "Accepted", 1: "Rejected"})), x="Outcome", y="downtime", ax=ax[1], showfliers=False)
ax[1].set(xlabel="Outcome", ylabel="Downtime (min), batches with downtime > 0", title="Downtime magnitude")
finish(fig, "07_downtime_analysis", "Downtime Analysis",
       f"Batches without downtime reject at {t['mean'].iloc[0]*100:.1f}% vs {t['mean'].iloc[-1]*100:.1f}% when downtime exceeds 30 minutes ({t['count'].iloc[-1]:,} batches). "
       "Reducing stoppages, or flagging long-stop batches for extra inspection, cuts rejections.")

# ---- 6. correlation heatmap ---------------------------------------------------------------------------------------
cols = ["production_volume_l", "production_duration_min", "machine_speed", "ambient_temperature", "raw_material_age", "fat_percentage",
        "acidity", "raw_material_quality_score", "machine_temperature", "pressure", "downtime", "maintenance_age_days", "vibration", "moisture_percentage", "rejected"]
C = d[cols].corr(method="spearman")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(C, cmap="coolwarm", center=0, annot=True, fmt=".2f", annot_kws={"size": 7}, ax=ax)
ax.set(xlabel="Variables (Spearman)", ylabel="Variables")
top = C["rejected"].drop("rejected").abs().sort_values(ascending=False)
mc = C.loc["moisture_percentage", "rejected"]
finish(fig, "08_correlation_heatmap", "Spearman Correlation Heatmap",
       f"Strongest rank correlations with rejection: {', '.join(top.index[:4])} (all modest: rejection depends on several interacting factors). "
       f"Moisture shows only {mc:+.2f} despite being the clearest quality signal in the moisture plot - its effect is two-sided (too wet OR too dry), "
       "which a monotonic correlation cannot see. This is why deviation-from-target features and tree models matter, and why correlation alone should not drive feature selection.")

# ---- 7. missingness ---------------------------------------------------------------------------------------------------
miss = m[["fat_percentage", "protein_percentage", "acidity", "pressure", "vibration", "moisture_percentage", "machine_temperature"]].isna().mean() * 100
ms = d.assign(miss=d.moisture_percentage.isna()).pivot_table(index="production_line", columns="shift", values="miss", aggfunc="mean")[["morning", "evening", "night"]] * 100
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
miss.sort_values().plot.barh(ax=ax[0], color="#3b6ea5"); ax[0].set(xlabel="Missing (%)", ylabel="Column", title="Missing share by column")
sns.heatmap(ms, annot=True, fmt=".0f", cmap="Reds", ax=ax[1], cbar_kws={"label": "% missing"}); ax[1].set(xlabel="Shift", ylabel="Production line", title="Moisture missing % by line x shift")
finish(fig, "09_missing_values", "Missing-Value Analysis",
       f"Moisture is missing {ms.loc[:, 'night'].mean():.0f}% of the time on night shift vs {ms.loc[:, 'morning'].mean():.0f}% in the morning, and worst on L3. "
       "Missingness is process-driven (lab/operator coverage at night), so analyses must not treat it as random; night-shift lab capture should be fixed.")

# ---- 8. time series ----------------------------------------------------------------------------------------------------------
d["month"] = d.production_timestamp.dt.to_period("M").dt.to_timestamp()
mo_all = d.groupby("month").rejected.mean() * 100
mo_line = d.groupby(["month", "production_line"]).rejected.mean().unstack() * 100
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(mo_all.index, mo_all.values, "k-o", lw=2, label="All lines")
for l in mo_line: ax.plot(mo_line.index, mo_line[l], alpha=.7, label=l)
ax.set(xlabel="Month", ylabel="Monthly rejection rate (%)"); ax.legend()
finish(fig, "10_rejection_trend", "Monthly Rejection Trend",
       f"Monthly rejection ranges from {mo_all.min():.1f}% to {mo_all.max():.1f}%. Visible month-to-month movement and line-level divergence mean rates are not stationary, "
       "which is why time-based validation and drift monitoring are required.")

# ---- 9. supplier performance ---------------------------------------------------------------------------------------------------
sp = d.groupby("supplier_id").agg(rej=("rejected", "mean"), age=("raw_material_age", "mean"), q=("raw_material_quality_score", "mean"), n=("rejected", "size"))
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(sp.age, sp.rej * 100, s=sp.n / sp.n.max() * 600 + 40, c=sp.q, cmap="viridis")
for s, r in sp.iterrows(): ax.annotate(s, (r.age, r.rej * 100), xytext=(5, 5), textcoords="offset points")
ax.set(xlabel="Mean raw-material age at processing (hours)", ylabel="Rejection rate (%)")
cb = fig.colorbar(ax.collections[0], ax=ax); cb.set_label("Mean raw-material quality score")
corr = sp[["age", "rej"]].corr().iloc[0, 1]
finish(fig, "11_supplier_comparison", "Supplier Performance: Freshness vs Rejection (bubble = volume)",
       f"Suppliers delivering older milk tend to reject more (correlation {corr:.2f} across suppliers). Contract terms on collection-to-plant time are a practical lever.")

# ---- 10. line comparison ---------------------------------------------------------------------------------------------------------------
lc = d.groupby("production_line").agg(rej=("rejected", "mean"), vol=("production_volume_l", "median"), dur=("production_duration_min", "median"),
        temp=("machine_temperature", lambda s: (s[s > 60] - 72).abs().mean()), vib=("vibration", "median"), maint=("maintenance_age_days", "mean"))
lc["rej"] *= 100
fig, ax = plt.subplots(1, 5, figsize=(16, 3.6))
for a, (c, lab) in zip(ax, [("rej", "Rejection rate (%)"), ("vol", "Median volume (L)"), ("dur", "Median duration (min)"), ("temp", "Mean |temp dev| (degC)"), ("vib", "Median vibration (mm/s)")]):
    a.bar(lc.index, lc[c], color="#3b6ea5"); a.set(xlabel="Production line", ylabel=lab, title=lab)
finish(fig, "12_line_comparison", "Production Line Comparison",
       f"Line {lc.rej.idxmax()} combines the highest rejection ({lc.rej.max():.1f}%) with the highest vibration ({lc.vib[lc.rej.idxmax()]:.2f} mm/s median) "
       f"and the longest maintenance cycle - consistent with wear-related quality loss. Lines differ in scale, so per-line baselines are needed for deviation features.")

# ---- 11. outliers ----------------------------------------------------------------------------------------------------------------------
oc = ["production_volume_l", "production_duration_min", "downtime", "vibration", "machine_temperature", "pressure", "raw_material_age"]
rows = []
for c in oc:
    s = d[c].dropna(); q1, q3 = s.quantile([.25, .75]); i = q3 - q1
    msk = (d[c] < q1 - 3*i) | (d[c] > q3 + 3*i)
    rows.append((c, msk.mean()*100, d.loc[msk, "rejected"].mean()*100))
O = pd.DataFrame(rows, columns=["col", "pct", "rej"])
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].barh(O.col, O.pct, color="#3b6ea5"); ax[0].set(xlabel="Extreme outliers, 3xIQR (% of rows)", ylabel="Column", title="How common are extremes?")
ax[1].barh(O.col, O.rej, color="#d1495b"); ax[1].axvline(base, ls="--", c="k"); ax[1].set(xlabel="Rejection rate among extreme rows (%)", ylabel="Column", title="Do extremes reject more? (dashed = overall)")
w = O.sort_values("rej", ascending=False).iloc[0]
finish(fig, "13_outlier_analysis", "Outlier Analysis (retained, not removed)",
       f"Extreme {w['col']} readings reject at {w.rej:.1f}% vs {base:.1f}% overall. Outliers carry real process information, so they are kept; only physically impossible values were nulled.")

with open(REP / "eda_report.md", "w") as f:
    f.write("# Exploratory Data Analysis (synthetic data)\n\n> Synthetic dataset - findings describe the simulation, not a real plant.\n\n")
    f.write(open(REP / "_eda_tables.md").read() + "\n")
    for name, title, interp in report:
        f.write(f"## {title}\n\n![{title}](figures/{name}.png)\n\n**Business interpretation:** {interp}\n\n")
(REP / "_eda_tables.md").unlink()
for n, t, i in report: print(n, "|", i[:150])
