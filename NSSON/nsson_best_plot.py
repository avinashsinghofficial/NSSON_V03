# /home/avi/NSSON/nsson_best_plot.py
#
# Best composite figure for NSSON (QoS, SLA, SMCC trade-offs).
# No camnsson dependency.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

NSSON_ROOT = "/home/avi/NSSON"

QOS_CSV = os.path.join(NSSON_ROOT, "qos_metrics.csv")
ROUTING_CSV = os.path.join(NSSON_ROOT, "routing_paths.csv")
OFFLOAD_CSV = os.path.join(NSSON_ROOT, "offload_feedback.csv")
OUTPUT_DIR = os.path.join(NSSON_ROOT, "plots_qos")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8")
sns.set_theme(context="talk", style="whitegrid")

def savefig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved", path)

def main():
    qos = pd.read_csv(QOS_CSV)
    offload = pd.read_csv(OFFLOAD_CSV)

    for col in ["mode", "slice", "app_class"]:
        if col in qos.columns:
            qos[col] = qos[col].astype("category")
    if "mode" in offload.columns:
        offload["mode"] = offload["mode"].astype("category")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Latency vs cycle
    ax = axes[0, 0]
    sns.lineplot(data=qos, x="cycle", y="latency_ms", hue="mode", ax=ax, linewidth=2.0)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("(a) End-to-End Latency vs Cycle")
    ax.legend(title="Mode", loc="upper right")

    # (b) Throughput vs cycle
    ax = axes[0, 1]
    sns.lineplot(data=qos, x="cycle", y="throughput_kbps", hue="mode", ax=ax, linewidth=2.0)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Throughput (kbps)")
    ax.set_title("(b) Throughput vs Cycle")
    ax.legend(title="Mode", loc="upper right")

    # (c) SLA compliance per slice
    ax = axes[1, 0]
    if {"sla_met_latency", "sla_met_pdr", "slice"} <= set(qos.columns):
        sla_lat = qos.groupby("slice")["sla_met_latency"].mean().reset_index()
        sla_pdr = qos.groupby("slice")["sla_met_pdr"].mean().reset_index()
        sla_lat["metric"] = "Latency SLA"
        sla_pdr["metric"] = "PDR SLA"
        sla_lat = sla_lat.rename(columns={"sla_met_latency": "compliance"})
        sla_pdr = sla_pdr.rename(columns={"sla_met_pdr": "compliance"})
        sla_all = pd.concat([sla_lat, sla_pdr], ignore_index=True)

        sns.barplot(data=sla_all, x="slice", y="compliance", hue="metric", ax=ax)
        ax.set_xlabel("Slice")
        ax.set_ylabel("SLA Compliance Rate")
        ax.set_title("(c) SLA Compliance per Slice")
        ax.legend(title="Metric", loc="lower right")
    else:
        ax.text(0.5, 0.5, "SLA columns missing in qos_metrics.csv",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(c) SLA Compliance per Slice (N/A)")

    # (d) Offload ratio + threshold
    ax = axes[1, 1]
    if {"offload_ratio", "threshold", "cycle"} <= set(offload.columns):
        ax2 = ax.twinx()
        sns.lineplot(data=offload, x="cycle", y="offload_ratio",
                     color="tab:blue", label="Offload ratio", ax=ax)
        sns.lineplot(data=offload, x="cycle", y="threshold",
                     color="tab:red", label="Threshold θ(t)", ax=ax2)
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Offload Ratio", color="tab:blue")
        ax2.set_ylabel("Threshold θ(t)", color="tab:red")
        ax.set_title("(d) Offload Ratio and Adaptive Threshold")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    else:
        ax.text(0.5, 0.5, "Offload/feedback columns missing",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(d) Offload Ratio and Adaptive Threshold (N/A)")

    fig.suptitle("NSSON Composite Figure: QoS, SLA, and SMCC Trade-offs", fontsize=14, y=1.02)
    savefig("nsson_composite_figure.png")

if __name__ == "__main__":
    main()
