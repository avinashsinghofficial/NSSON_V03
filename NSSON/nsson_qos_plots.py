# /home/avi/NSSON_V02/nsson_qos_plots.py
#
# Generate Q1-level figures:
#  - Latency/Power/Offload/Threshold comparisons across modes.
#  - Extended QoS: baseline SDN-IoT vs NSSON (latency, PDR, drop, overhead, throughput).

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

NSSON_ROOT = "/home/avi/NSSON_V02"

QOS_CSV = os.path.join(NSSON_ROOT, "qos_metrics.csv")
ROUTING_CSV = os.path.join(NSSON_ROOT, "routing_paths.csv")
OFFLOAD_CSV = os.path.join(NSSON_ROOT, "offload_feedback.csv")
OUT_DIR = os.path.join(NSSON_ROOT, "plots_qos")

os.makedirs(OUT_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8")
sns.set_theme(context="talk", style="whitegrid")

def savefig(name):
    path = os.path.join(OUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved", path)

def main():
    qos = pd.read_csv(QOS_CSV)

    # Robust read of routing_paths.csv (may be empty or missing)
    if os.path.exists(ROUTING_CSV):
        try:
            routing = pd.read_csv(ROUTING_CSV)
            if routing.empty:
                routing = pd.DataFrame()
        except Exception:
            routing = pd.DataFrame()
    else:
        routing = pd.DataFrame()

    offload = pd.read_csv(OFFLOAD_CSV) if os.path.exists(OFFLOAD_CSV) else pd.DataFrame()

    for col in ["mode", "slice", "app_class"]:
        if col in qos.columns:
            qos[col] = qos[col].astype("category")
    if "mode" in offload.columns:
        offload["mode"] = offload["mode"].astype("category")

    # === 1. Offload telemetry comparisons ===

    if not offload.empty:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=offload, x="cycle", y="offload_ratio", hue="mode")
        plt.ylabel("Offload Ratio")
        plt.title("Offload Ratio vs Cycle across Modes")
        savefig("01_offload_ratio_vs_cycle_all_modes.png")

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=offload, x="cycle", y="latency_ms", hue="mode")
        plt.ylabel("Latency (ms)")
        plt.title("Latency vs Cycle across Modes")
        savefig("02_latency_vs_cycle_all_modes.png")

        if "power_wh" in offload.columns:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=offload, x="cycle", y="power_wh", hue="mode")
            plt.ylabel("Power (Wh)")
            plt.title("Power vs Cycle across Modes")
            savefig("03_power_vs_cycle_all_modes.png")

        # Use 'threshold' column (generate_qos_data.py writes 'threshold', not 'theta')
        if "threshold" in offload.columns:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=offload, x="cycle", y="threshold", hue="mode")
            plt.ylabel("Threshold theta(t)")
            plt.title("Threshold Evolution vs Cycle across Modes")
            savefig("04_threshold_vs_cycle_all_modes.png")

        # === 2. Latency & power distributions per mode ===

        plt.figure(figsize=(8, 6))
        sns.boxplot(data=offload, x="mode", y="latency_ms")
        plt.ylabel("Latency (ms)")
        plt.title("Latency Distribution per Mode")
        savefig("05_latency_box_per_mode.png")

        if "power_wh" in offload.columns:
            plt.figure(figsize=(8, 6))
            sns.boxplot(data=offload, x="mode", y="power_wh")
            plt.ylabel("Power (Wh)")
            plt.title("Power Distribution per Mode")
            savefig("06_power_box_per_mode.png")

    # === 3. Extended QoS comparisons: baseline SDN-IoT vs NSSON ===

    qos_qos = qos[qos["mode"].isin(["baseline_sdn_iot", "nsson_full"])].copy()

    if not qos_qos.empty:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=qos_qos, x="cycle", y="latency_ms", hue="mode")
        plt.ylabel("Latency (ms)")
        plt.title("Extended QoS: Latency vs Cycle (Baseline SDN-IoT vs NSSON)")
        savefig("07_qos_latency_vs_cycle_baseline_vs_nsson.png")

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=qos_qos, x="cycle", y="pdr", hue="mode")
        plt.ylabel("PDR")
        plt.ylim(0, 1)
        plt.title("Extended QoS: PDR vs Cycle (Baseline SDN-IoT vs NSSON)")
        savefig("08_qos_pdr_vs_cycle_baseline_vs_nsson.png")

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=qos_qos, x="cycle", y="drop_rate", hue="mode")
        plt.ylabel("Drop Rate")
        plt.title("Extended QoS: Packet Drop Rate vs Cycle (Baseline SDN-IoT vs NSSON)")
        savefig("09_qos_drop_vs_cycle_baseline_vs_nsson.png")

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=qos_qos, x="cycle", y="routing_overhead", hue="mode")
        plt.ylabel("Routing Overhead Ratio")
        plt.title("Extended QoS: Routing Overhead vs Cycle (Baseline SDN-IoT vs NSSON)")
        savefig("10_qos_overhead_vs_cycle_baseline_vs_nsson.png")

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=qos_qos, x="cycle", y="throughput_kbps", hue="mode")
        plt.ylabel("Throughput (kbps)")
        plt.title("Extended QoS: Throughput vs Cycle (Baseline SDN-IoT vs NSSON)")
        savefig("11_qos_throughput_vs_cycle_baseline_vs_nsson.png")

    # === 4. SLA compliance per slice and mode ===

    if "sla_met_latency" in qos.columns:
        sla_lat = qos.groupby(["mode", "slice"])["sla_met_latency"].mean().reset_index()
        plt.figure(figsize=(10, 6))
        sns.barplot(data=sla_lat, x="slice", y="sla_met_latency", hue="mode")
        plt.ylabel("Latency SLA Compliance Rate")
        plt.title("Latency SLA Compliance per Slice and Mode")
        savefig("12_sla_latency_compliance_slice_mode.png")

    if "sla_met_pdr" in qos.columns:
        sla_pdr = qos.groupby(["mode", "slice"])["sla_met_pdr"].mean().reset_index()
        plt.figure(figsize=(10, 6))
        sns.barplot(data=sla_pdr, x="slice", y="sla_met_pdr", hue="mode")
        plt.ylabel("PDR SLA Compliance Rate")
        plt.title("PDR SLA Compliance per Slice and Mode")
        savefig("13_sla_pdr_compliance_slice_mode.png")

    # === 5. Path selection and adaptive routing (if routing_paths.csv populated) ===

    if not routing.empty:
        routing["mode"] = routing["mode"].astype("category")

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=routing, x="cycle", y="latency_ms", hue="path_id", style="mode")
        plt.ylabel("Path Latency (ms)")
        plt.title("Path Latency vs Cycle per Path and Mode")
        savefig("14_path_latency_vs_cycle_mode.png")

        if {"slice", "selected"} <= set(routing.columns):
            freq = routing.groupby(["mode", "slice", "path_id"])["selected"].mean().reset_index()
            plt.figure(figsize=(10, 6))
            sns.catplot(
                data=freq, x="slice", y="selected", hue="path_id",
                col="mode", kind="bar", height=4, aspect=1.2
            )
            plt.ylabel("Selection Frequency")
            plt.suptitle("Path Selection Frequency per Slice and Mode")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, "15_path_selection_freq_slice_mode.png"), dpi=300, bbox_inches="tight")
            plt.close()
            print("Saved", os.path.join(OUT_DIR, "15_path_selection_freq_slice_mode.png"))

    # === 6. ECDFs for latency/jitter per mode ===

    plt.figure(figsize=(10, 6))
    for mode, group in qos.groupby("mode"):
        x = np.sort(group["latency_ms"].values)
        y = np.arange(1, len(x) + 1) / len(x)
        plt.step(x, y, where="post", label=str(mode))
    plt.xlabel("Latency (ms)")
    plt.ylabel("ECDF")
    plt.title("Latency ECDF per Mode")
    plt.legend()
    savefig("16_latency_ecdf_per_mode.png")

    plt.figure(figsize=(10, 6))
    for mode, group in qos.groupby("mode"):
        x = np.sort(group["jitter_ms"].values)
        y = np.arange(1, len(x) + 1) / len(x)
        plt.step(x, y, where="post", label=str(mode))
    plt.xlabel("Jitter (ms)")
    plt.ylabel("ECDF")
    plt.title("Jitter ECDF per Mode")
    plt.legend()
    savefig("17_jitter_ecdf_per_mode.png")

if __name__ == "__main__":
    main()
