#!/usr/bin/env python3
"""Generate publication-quality QoS comparison figures."""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

STATS_JSON = Path("/home/avi/NSSON_V02/logs/qos_aggregated/qos_stats_by_condition.json")
OUT_DIR = Path("/home/avi/NSSON_V02/logs/qos_aggregated/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(STATS_JSON) as f:
    stats = json.load(f)
df = pd.DataFrame(stats)

load_order = ["light", "moderate", "heavy"]
df["load"] = pd.Categorical(df["load"], categories=load_order, ordered=True)
df = df.sort_values(["controller", "load"])

baseline = df[df["controller"] == "baseline"].set_index("load")
nsson = df[df["controller"] == "nsson"].set_index("load")

baseline_color = "#1f77b4"
nsson_color = "#d62728"

fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(len(load_order))
width = 0.35

ax.bar(x - width/2, baseline.loc[load_order, "rtt_avg_ms_mean"].values,
       width, label="Baseline SDN-IoT", color=baseline_color,
       yerr=baseline.loc[load_order, "rtt_avg_ms_ci95"].values,
       capsize=4, error_kw={'ecolor': 'gray', 'elinewidth': 1})
ax.bar(x + width/2, nsson.loc[load_order, "rtt_avg_ms_mean"].values,
       width, label="NSSON (QoS-aware)", color=nsson_color,
       yerr=nsson.loc[load_order, "rtt_avg_ms_ci95"].values,
       capsize=4, error_kw={'ecolor': 'gray', 'elinewidth': 1})

ax.set_xlabel("Background load")
ax.set_ylabel("Average RTT (ms)")
ax.set_title("Inference-path RTT: Baseline vs NSSON")
ax.set_xticks(x)
ax.set_xticklabels(["Light (2 Mb/s)", "Moderate (12 Mb/s)", "Heavy (19 Mb/s)"])
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "qos_rtt_comparison.png", dpi=300)
print("Saved: qos_rtt_comparison.png")

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(x - width/2, baseline.loc[load_order, "control_tcp_throughput_mbps_mean"].values,
       width, label="Baseline SDN-IoT", color=baseline_color,
       yerr=baseline.loc[load_order, "control_tcp_throughput_mbps_ci95"].values,
       capsize=4, error_kw={'ecolor': 'gray', 'elinewidth': 1})
ax.bar(x + width/2, nsson.loc[load_order, "control_tcp_throughput_mbps_mean"].values,
       width, label="NSSON (QoS-aware)", color=nsson_color,
       yerr=nsson.loc[load_order, "control_tcp_throughput_mbps_ci95"].values,
       capsize=4, error_kw={'ecolor': 'gray', 'elinewidth': 1})

ax.set_xlabel("Background load")
ax.set_ylabel("TCP/8090 throughput (Mbit/s)")
ax.set_title("Inference-service throughput: Baseline vs NSSON")
ax.set_xticks(x)
ax.set_xticklabels(["Light (2 Mb/s)", "Moderate (12 Mb/s)", "Heavy (19 Mb/s)"])
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "qos_tcp_throughput_comparison.png", dpi=300)
print("Saved: qos_tcp_throughput_comparison.png")

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(x - width/2, baseline.loc[load_order, "ping_loss_pct_mean"].values,
       width, label="Baseline SDN-IoT", color=baseline_color)
ax.bar(x + width/2, nsson.loc[load_order, "ping_loss_pct_mean"].values,
       width, label="NSSON (QoS-aware)", color=nsson_color)

ax.set_xlabel("Background load")
ax.set_ylabel("ICMP ping loss (%)")
ax.set_title("Inference-path packet loss: Baseline vs NSSON")
ax.set_xticks(x)
ax.set_xticklabels(["Light (2 Mb/s)", "Moderate (12 Mb/s)", "Heavy (19 Mb/s)"])
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "qos_ping_loss_comparison.png", dpi=300)
print("Saved: qos_ping_loss_comparison.png")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7), sharex=True)
ax1.bar(x - width/2, baseline.loc[load_order, "rtt_avg_ms_mean"].values,
        width, label="Baseline", color=baseline_color,
        yerr=baseline.loc[load_order, "rtt_avg_ms_ci95"].values,
        capsize=3, error_kw={'ecolor': 'gray', 'elinewidth': 1})
ax1.bar(x + width/2, nsson.loc[load_order, "rtt_avg_ms_mean"].values,
        width, label="NSSON", color=nsson_color,
        yerr=nsson.loc[load_order, "rtt_avg_ms_ci95"].values,
        capsize=3, error_kw={'ecolor': 'gray', 'elinewidth': 1})
ax1.set_ylabel("RTT (ms)")
ax1.legend(loc='upper left')
ax1.grid(axis='y', alpha=0.3)

ax2.bar(x - width/2, baseline.loc[load_order, "control_tcp_throughput_mbps_mean"].values,
        width, color=baseline_color,
        yerr=baseline.loc[load_order, "control_tcp_throughput_mbps_ci95"].values,
        capsize=3, error_kw={'ecolor': 'gray', 'elinewidth': 1})
ax2.bar(x + width/2, nsson.loc[load_order, "control_tcp_throughput_mbps_mean"].values,
        width, color=nsson_color,
        yerr=nsson.loc[load_order, "control_tcp_throughput_mbps_ci95"].values,
        capsize=3, error_kw={'ecolor': 'gray', 'elinewidth': 1})
ax2.set_ylabel("TCP/8090 (Mb/s)")
ax2.set_xlabel("Background load")
ax2.set_xticks(x)
ax2.set_xticklabels(["Light", "Moderate", "Heavy"])
ax2.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "qos_combined_panel.png", dpi=300)
print("Saved: qos_combined_panel.png")

print("\nAll figures saved to:", OUT_DIR)
