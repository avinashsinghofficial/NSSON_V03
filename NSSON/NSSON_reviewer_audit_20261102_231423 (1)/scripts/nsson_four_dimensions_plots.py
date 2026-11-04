# nsson_four_dimensions_plots.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# CONFIGURATION
# -------------------------

# Your existing aggregated QoS CSV
QOS_CSV = "/home/avi/NSSON_V02/logs/qos_aggregated/qos_all_runs_quality_checked.csv"

# Output directory for figures
OUT_DIR = "/home/avi/NSSON_V02/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# SLA thresholds for offload traffic (tune if needed)
SLA_LATENCY_MAX = 30.0      # ms (rtt_avg_ms)
SLA_LOSS_MAX = 5.0          # % (ping_loss_pct)
SLA_THROUGHPUT_MIN = 5.0    # Mbps (control_tcp_throughput_mbps)

# Hypothetical slice bandwidth SLAs (Mbps)
SLICE_BW_SLAS = {
    "autonomous": (8.0, 15.0),
    "healthcare": (6.0, 10.0),
    "industrial": (3.0, 6.0),
}

# -------------------------
# LOAD DATA
# -------------------------

if not os.path.exists(QOS_CSV):
    raise FileNotFoundError(f"QoS CSV not found at: {QOS_CSV}")

df = pd.read_csv(QOS_CSV)

# Expected columns (from your file):
# mode, load, rtt_avg_ms, rtt_jitter_mdev_ms, ping_loss_pct,
# control_tcp_throughput_mbps, background_udp_throughput_mbps,
# background_udp_jitter_ms, background_udp_loss_pct, ...

# Define NSSON vs baseline by mode string
# Adjust these patterns if your modes differ slightly.
baseline_mask = df["mode"].str.contains("baseline", case=False, na=False)
nsson_mask = ~baseline_mask

offload = df[nsson_mask].copy()      # treat NSSON rows as "offload"
background = df[baseline_mask].copy()  # baseline as "background"

if offload.empty or background.empty:
    raise ValueError(
        "Both NSSON and baseline rows must be present.\n"
        f"Modes found: {df['mode'].unique()}"
    )

# -------------------------
# 1. MULTI-DIMENSIONAL QoS RADAR CHART
# -------------------------

def normalize_series_bad(s):
    mx = s.max()
    if mx == 0:
        return s * 0.0
    return s / mx

def normalize_series_good(s):
    mx = s.max()
    if mx == 0:
        return s * 0.0
    return s / mx

# Use metrics that exist in your CSV:
# - latency: rtt_avg_ms
# - jitter: rtt_jitter_mdev_ms (or background_udp_jitter_ms)
# - loss: ping_loss_pct (convert to fraction)
# - throughput: control_tcp_throughput_mbps
# - PDR: 1 - ping_loss_pct/100

offload["loss_frac"] = offload["ping_loss_pct"] / 100.0
background["loss_frac"] = background["ping_loss_pct"] / 100.0
offload["pdr"] = 1.0 - offload["loss_frac"]
background["pdr"] = 1.0 - background["loss_frac"]

agg_off = {
    "latency": offload["rtt_avg_ms"].mean(),
    "jitter": offload["rtt_jitter_mdev_ms"].mean(),
    "loss": offload["loss_frac"].mean(),
    "throughput": offload["control_tcp_throughput_mbps"].mean(),
    "pdr": offload["pdr"].mean(),
}

agg_bg = {
    "latency": background["rtt_avg_ms"].mean(),
    "jitter": background["rtt_jitter_mdev_ms"].mean(),
    "loss": background["loss_frac"].mean(),
    "throughput": background["control_tcp_throughput_mbps"].mean(),
    "pdr": background["pdr"].mean(),
}

all_vals = pd.DataFrame({
    "offload": agg_off,
    "background": agg_bg,
}).T

norm = pd.DataFrame()
norm["latency"] = normalize_series_bad(all_vals["latency"])
norm["jitter"] = normalize_series_bad(all_vals["jitter"])
norm["loss"] = normalize_series_bad(all_vals["loss"])
norm["throughput"] = normalize_series_good(all_vals["throughput"])
norm["pdr"] = normalize_series_good(all_vals["pdr"])

categories = list(norm.columns)
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))

for label, color in zip(norm.index, ["red", "blue"]):
    values = norm.loc[label].values.tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=label, color=color)
    ax.fill(angles, values, alpha=0.15, color=color)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]))
ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=9)
ax.set_ylim(0, 1)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "nsson_qos_radar.pdf"))
plt.close()

# -------------------------
# 2. SLA FULFILLMENT RATIO
# -------------------------

# SLA: latency <= SLA_LATENCY_MAX, loss <= SLA_LOSS_MAX (%), throughput >= SLA_THROUGHPUT_MIN
offload["sla_met"] = (
    (offload["rtt_avg_ms"] <= SLA_LATENCY_MAX) &
    (offload["ping_loss_pct"] <= SLA_LOSS_MAX) &
    (offload["control_tcp_throughput_mbps"] >= SLA_THROUGHPUT_MIN)
).astype(int)

# Group by load if available
if "load" in offload.columns:
    load_order = ["light", "medium", "heavy"]
    sfr_by_load = offload.groupby("load")["sla_met"].mean().reindex(load_order)
else:
    sfr_by_load = pd.Series({"overall": offload["sla_met"].mean()})

fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(sfr_by_load.index, sfr_by_load.values, color="green")
ax.set_ylabel("SLA Fulfillment Ratio")
ax.set_title("NSSON Offload SLA Fulfillment")
ax.set_ylim(0, 1.05)
for i, v in enumerate(sfr_by_load.values):
    ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "nsson_sla_fulfillment.pdf"))
plt.close()

# -------------------------
# 3. NETWORK SLICING: THROUGHPUT + BW-SLA BANDS (CONCEPTUAL)
# -------------------------

# Map current classes to hypothetical slices:
# - NSSON (offload) → autonomous
# - Baseline (background) → split between healthcare and industrial
throughput_autonomous = offload["control_tcp_throughput_mbps"].mean()
throughput_bg = background["control_tcp_throughput_mbps"].mean()

throughput_healthcare = throughput_bg * 0.6
throughput_industrial = throughput_bg * 0.4

slice_throughput = {
    "autonomous": throughput_autonomous,
    "healthcare": throughput_healthcare,
    "industrial": throughput_industrial,
}

slices = list(slice_throughput.keys())
values = [slice_throughput[s] for s in slices]
mins = [SLICE_BW_SLAS[s][0] for s in slices]
maxs = [SLICE_BW_SLAS[s][1] for s in slices]

fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(len(slices))
ax.bar(x, values, yerr=[np.array(values)-np.array(mins), np.array(maxs)-np.array(values)],
       capsize=5, alpha=0.7, color="orange", label="Measured throughput")

for i, s in enumerate(slices):
    ax.axhline(mins[i], color="green", linestyle="--", linewidth=1)
    ax.axhline(maxs[i], color="red", linestyle="--", linewidth=1)

ax.set_xticks(x)
ax.set_xticklabels(slices)
ax.set_ylabel("Throughput (Mbps)")
ax.set_title("Per-Slice Throughput with Bandwidth-SLA Bands (Conceptual)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "nsson_slice_throughput.pdf"))
plt.close()

# -------------------------
# 4. ADAPTIVE ROUTING: RTT CDF WITH CONCEPTUAL ADAPTIVE CURVE
# -------------------------

rtt_samples = offload["rtt_avg_ms"].dropna().values

if len(rtt_samples) == 0:
    raise ValueError("No RTT samples found for offload traffic.")

sorted_rtt = np.sort(rtt_samples)
cdf_fixed = np.arange(1, len(sorted_rtt)+1) / len(sorted_rtt)

# Conceptual adaptive gain: assume 3 ms lower RTT on average
regret = 3.0
rtt_adaptive = np.maximum(sorted_rtt - regret, 0.1)
cdf_adaptive = cdf_fixed

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(sorted_rtt, cdf_fixed, label="Fixed path (NSSON)", color="blue")
ax.plot(rtt_adaptive, cdf_adaptive, label="Ideal adaptive path (conceptual)", color="red", linestyle="--")

ax.set_xlabel("RTT (ms)")
ax.set_ylabel("CDF")
ax.set_title("RTT CDF: Fixed vs. Ideal Adaptive Routing")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "nsson_adaptive_routing_cdf.pdf"))
plt.close()

print("All four figures generated in:", OUT_DIR)
