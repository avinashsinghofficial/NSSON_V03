import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.figsize": (6.0, 4.0),
    "figure.dpi": 300,
})

QOS_CSV = "/home/avi/NSSON_V02/logs/qos_aggregated/qos_all_runs_quality_checked.csv"
OUT_DIR = "/home/avi/NSSON_V02/figures_ieee"
os.makedirs(OUT_DIR, exist_ok=True)

SLA_LATENCY_MAX = 30.0      # ms
SLA_LOSS_MAX = 5.0          # %
SLA_THROUGHPUT_MIN = 5.0    # Mbps

SLICE_BW_SLAS = {
    "autonomous": (8.0, 15.0),
    "healthcare": (6.0, 10.0),
    "industrial": (3.0, 6.0),
}

df = pd.read_csv(QOS_CSV)
baseline_mask = df["mode"].str.contains("baseline", case=False, na=False)
nsson_mask = ~baseline_mask
offload = df[nsson_mask].copy()
background = df[baseline_mask].copy()

if offload.empty or background.empty:
    raise ValueError("Both NSSON and baseline rows must be present.")

offload["loss_frac"] = offload["ping_loss_pct"] / 100.0
background["loss_frac"] = background["ping_loss_pct"] / 100.0
offload["pdr"] = 1.0 - offload["loss_frac"]
background["pdr"] = 1.0 - background["loss_frac"]

# 1) Multi-dimensional QoS profile (grouped bar, not radar)
metrics = ["rtt_avg_ms", "rtt_jitter_mdev_ms", "loss_frac", "control_tcp_throughput_mbps", "pdr"]
labels = ["RTT (ms)", "Jitter (ms)", "Loss", "Throughput (Mbps)", "PDR"]
vals_off = [offload[m].mean() for m in metrics]
vals_bg = [background[m].mean() for m in metrics]

x = np.arange(len(metrics))
width = 0.35
fig, ax = plt.subplots()
ax.bar(x - width/2, vals_off, width, label="Offload (NSSON)", color="#1f77b4")
ax.bar(x + width/2, vals_bg, width, label="Background (Baseline)", color="#d62728")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15)
ax.set_ylabel("Normalized value (per metric scale)")
ax.set_title("Multi-dimensional QoS Profile (Measured)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ieee_qos_profile.png"), dpi=300)
plt.close()

# 2) SLA fulfillment ratio by load
offload["sla_met"] = (
    (offload["rtt_avg_ms"] <= SLA_LATENCY_MAX) &
    (offload["ping_loss_pct"] <= SLA_LOSS_MAX) &
    (offload["control_tcp_throughput_mbps"] >= SLA_THROUGHPUT_MIN)
).astype(int)

load_order = ["light", "medium", "heavy"]
sfr = offload.groupby("load")["sla_met"].mean().reindex(load_order)

fig, ax = plt.subplots()
ax.bar(sfr.index, sfr.values, color="#2ca02c")
ax.set_ylabel("SLA Fulfillment Ratio")
ax.set_title("Offload SLA Compliance (Measured)")
ax.set_ylim(0, 1.05)
for i, v in enumerate(sfr.values):
    ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ieee_sla_fulfillment.png"), dpi=300)
plt.close()

# 3) Network slicing (conceptual bands + measured autonomous)
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

fig, ax = plt.subplots()
x = np.arange(len(slices))
ax.bar(x, values, alpha=0.7, color="#ff7f0e", label="Measured/derived throughput")
for i, s in enumerate(slices):
    ax.axhline(mins[i], color="green", linestyle="--", linewidth=1)
    ax.axhline(maxs[i], color="red", linestyle="--", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(slices)
ax.set_ylabel("Throughput (Mbps)")
ax.set_title("Per-Slice Throughput with BW-SLA Bands (Conceptual)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ieee_slice_throughput.png"), dpi=300)
plt.close()

# 4) Adaptive routing CDF (measured fixed + conceptual shift)
rtt_samples = offload["rtt_avg_ms"].dropna().values
if len(rtt_samples) == 0:
    raise ValueError("No RTT samples found for offload traffic.")
sorted_rtt = np.sort(rtt_samples)
cdf_fixed = np.arange(1, len(sorted_rtt)+1) / len(sorted_rtt)
regret = 3.0
rtt_adaptive = np.maximum(sorted_rtt - regret, 0.1)
cdf_adaptive = cdf_fixed

fig, ax = plt.subplots()
ax.plot(sorted_rtt, cdf_fixed, label="Fixed path (NSSON)", color="#1f77b4")
ax.plot(rtt_adaptive, cdf_adaptive, label="Ideal adaptive (conceptual)", color="#d62728", linestyle="--")
ax.set_xlabel("RTT (ms)")
ax.set_ylabel("CDF")
ax.set_title("RTT CDF: Fixed vs. Ideal Adaptive (Conceptual)")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ieee_adaptive_routing_cdf.png"), dpi=300)
plt.close()

print("IEEE-style figures saved to:", OUT_DIR)
