import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# IEEE-like style
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})

QOS_CSV = "/home/avi/NSSON_V02/logs/qos_aggregated/qos_all_runs_quality_checked.csv"
OUT_DIR = "/home/avi/NSSON_V02/figures_ieee"
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(QOS_CSV):
    raise FileNotFoundError(f"QoS CSV not found at: {QOS_CSV}")

df = pd.read_csv(QOS_CSV)
print("Columns found:", list(df.columns))
print("Modes found:", df["mode"].unique())
print("Loads found:", df["load"].unique() if "load" in df.columns else None)

# Define NSSON vs baseline by mode string
baseline_mask = df["mode"].str.contains("baseline", case=False, na=False)
nsson_mask = ~baseline_mask

df_nsson = df[nsson_mask].copy()
df_base = df[baseline_mask].copy()

if df_nsson.empty or df_base.empty:
    raise ValueError("Both NSSON and baseline rows must be present.")

# Ensure load ordering if present
load_order = ["light", "medium", "heavy"]
if "load" in df.columns:
    df_nsson["load"] = pd.Categorical(df_nsson["load"], categories=load_order, ordered=True)
    df_base["load"] = pd.Categorical(df_base["load"], categories=load_order, ordered=True)

# -------------------------
# 1) 2x2 QoS panel: RTT, jitter, loss, throughput vs. load, per mode
# -------------------------

metrics = [
    ("rtt_avg_ms", "Average RTT (ms)"),
    ("rtt_jitter_mdev_ms", "RTT jitter (ms)"),
    ("ping_loss_pct", "Ping loss (%)"),
    ("control_tcp_throughput_mbps", "TCP throughput (Mbps)"),
]

fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.0))
axes = axes.ravel()

for ax, (col, ylabel) in zip(axes, metrics):
    # Group by mode and load
    grp = df.groupby(["mode", "load"])[col].mean().unstack("load")
    # Reorder columns if load exists
    if all(l in grp.columns for l in load_order):
        grp = grp[load_order]

    # Plot as grouped bars
    x = np.arange(len(grp.index))
    width = 0.8 / len(grp.columns)
    for i, load_name in enumerate(grp.columns):
        vals = grp[load_name].values
        ax.bar(x + i*width - 0.4, vals, width, label=str(load_name))

    ax.set_xticks(x)
    ax.set_xticklabels([str(m).replace("baseline_sdn_iot", "Baseline").replace("nsson", "NSSON")
                        for m in grp.index], rotation=15)
    ax.set_ylabel(ylabel)
    ax.set_title(col.replace("_", " ").upper())
    ax.grid(True, linestyle="--", alpha=0.25)
    if ax == axes[0]:
        ax.legend(title="Load", fontsize=7, title_fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ieee_qos_panel_2x2.png"), dpi=300)
plt.close()

# -------------------------
# 2) SLA fulfillment ratio (measured compliance)
# -------------------------

# Define SLA thresholds (tune if needed)
SLA_LATENCY_MAX = 30.0      # ms (rtt_avg_ms)
SLA_LOSS_MAX = 5.0          # % (ping_loss_pct)
SLA_THROUGHPUT_MIN = 5.0    # Mbps (control_tcp_throughput_mbps)

n = df_nsson.copy()
n["sla_met"] = (
    (n["rtt_avg_ms"] <= SLA_LATENCY_MAX) &
    (n["ping_loss_pct"] <= SLA_LOSS_MAX) &
    (n["control_tcp_throughput_mbps"] >= SLA_THROUGHPUT_MIN)
).astype(int)

if "load" in n.columns:
    sfr = n.groupby("load")["sla_met"].mean().reindex(load_order)
else:
    sfr = pd.Series({"overall": n["sla_met"].mean()})

fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(sfr.index, sfr.values, color="#2ca02c")
ax.set_ylabel("SLA Fulfillment Ratio")
ax.set_title("Offload SLA Compliance (Measured)")
ax.set_ylim(0, 1.05)
for i, v in enumerate(sfr.values):
    ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ieee_sla_fulfillment.png"), dpi=300)
plt.close()

# -------------------------
# 3) Optional radar chart (summary of measured metrics)
# -------------------------

# Build aggregated metrics for NSSON vs baseline
def agg_qos(df_part):
    return {
        "latency": df_part["rtt_avg_ms"].mean(),
        "jitter": df_part["rtt_jitter_mdev_ms"].mean(),
        "loss": df_part["ping_loss_pct"].mean(),
        "throughput": df_part["control_tcp_throughput_mbps"].mean(),
        "pdr": (1.0 - df_part["ping_loss_pct"]/100.0).mean(),
    }

agg_nsson = agg_qos(df_nsson)
agg_base = agg_qos(df_base)

all_vals = pd.DataFrame({
    "NSSON": agg_nsson,
    "Baseline": agg_base,
}).T

# Normalize to [0,1]
def norm_bad(s):
    mx = s.max()
    return s / mx if mx != 0 else s * 0.0

def norm_good(s):
    mx = s.max()
    return s / mx if mx != 0 else s * 0.0

norm = pd.DataFrame()
norm["latency"] = norm_bad(all_vals["latency"])
norm["jitter"] = norm_bad(all_vals["jitter"])
norm["loss"] = norm_bad(all_vals["loss"])
norm["throughput"] = norm_good(all_vals["throughput"])
norm["pdr"] = norm_good(all_vals["pdr"])

categories = list(norm.columns)
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))
for label, color in zip(norm.index, ["#1f77b4", "#d62728"]):
    values = norm.loc[label].values.tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=label, color=color)
    ax.fill(angles, values, alpha=0.15, color=color)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]))
ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=8)
ax.set_ylim(0, 1)
ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ieee_qos_radar.png"), dpi=300)
plt.close()

print("IEEE-style figures saved to:", OUT_DIR)
