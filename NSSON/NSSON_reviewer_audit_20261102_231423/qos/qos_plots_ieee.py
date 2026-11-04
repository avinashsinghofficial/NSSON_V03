# /home/avi/NSSON_V02/qos/qos_plots_ieee.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

NSSON_ROOT = "/home/avi/NSSON_V02"
OUTPUT_DIR = os.path.join(NSSON_ROOT, "plots_qos")
os.makedirs(OUTPUT_DIR, exist_ok=True)

QOS_CSV = os.path.join(NSSON_ROOT, "qos_metrics.csv")
ROUTING_CSV = os.path.join(NSSON_ROOT, "routing_paths.csv")
OFFLOAD_CSV = os.path.join(NSSON_ROOT, "offload_feedback.csv")

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
    routing = pd.read_csv(ROUTING_CSV)
    offload = pd.read_csv(OFFLOAD_CSV)

    for col in ["mode", "slice", "app_class"]:
        if col in qos.columns:
            qos[col] = qos[col].astype("category")

    # Ensure offload has 'mode' column
    if "mode" not in offload.columns:
        offload["mode"] = "nsson"
    if "mode" in offload.columns:
        offload["mode"] = offload["mode"].astype("category")

    if not routing.empty and "path_id" in routing.columns:
        for col in ["path_id", "slice"]:
            if col in routing.columns:
                routing[col] = routing[col].astype("category")

    # 1–6: QoS vs cycle
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=qos, x="cycle", y="latency_ms", hue="mode")
    plt.ylabel("Latency (ms)")
    plt.title("Latency vs Cycle per Mode")
    savefig("01_latency_vs_cycle_mode.png")

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=qos, x="cycle", y="jitter_ms", hue="mode")
    plt.ylabel("Jitter (ms)")
    plt.title("Jitter vs Cycle per Mode")
    savefig("02_jitter_vs_cycle_mode.png")

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=qos, x="cycle", y="throughput_kbps", hue="mode")
    plt.ylabel("Throughput (kbps)")
    plt.title("Throughput vs Cycle per Mode")
    savefig("03_throughput_vs_cycle_mode.png")

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=qos, x="cycle", y="pdr", hue="mode")
    plt.ylabel("PDR")
    plt.title("Packet Delivery Ratio vs Cycle per Mode")
    savefig("04_pdr_vs_cycle_mode.png")

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=qos, x="cycle", y="drop_rate", hue="mode")
    plt.ylabel("Drop Rate")
    plt.title("Packet Drop Rate vs Cycle per Mode")
    savefig("05_drop_rate_vs_cycle_mode.png")

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=qos, x="cycle", y="routing_overhead", hue="mode")
    plt.ylabel("Routing Overhead")
    plt.title("Routing Overhead vs Cycle per Mode")
    savefig("06_overhead_vs_cycle_mode.png")

    # 7–12: Distributions per slice/app
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=qos, x="slice", y="latency_ms", hue="mode")
    plt.ylabel("Latency (ms)")
    plt.title("Latency Distribution per Slice and Mode")
    savefig("07_latency_box_slice_mode.png")

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=qos, x="slice", y="jitter_ms", hue="mode")
    plt.ylabel("Jitter (ms)")
    plt.title("Jitter Distribution per Slice and Mode")
    savefig("08_jitter_box_slice_mode.png")

    plt.figure(figsize=(8, 6))
    sns.violinplot(data=qos, x="slice", y="throughput_kbps", hue="mode", split=True)
    plt.ylabel("Throughput (kbps)")
    plt.title("Throughput Distribution per Slice and Mode")
    savefig("09_throughput_violin_slice_mode.png")

    plt.figure(figsize=(8, 6))
    sns.histplot(data=qos, x="latency_ms", hue="app_class", kde=True, bins=30, stat="density")
    plt.ylabel("Density")
    plt.title("Latency Distribution per Application Class")
    savefig("10_latency_hist_app_class.png")

    plt.figure(figsize=(8, 6))
    sns.kdeplot(data=qos, x="jitter_ms", hue="slice", common_norm=False)
    plt.ylabel("Density")
    plt.title("Jitter KDE per Slice")
    savefig("11_jitter_kde_slice.png")

    plt.figure(figsize=(8, 6))
    sns.kdeplot(data=qos, x="latency_ms", hue="mode", common_norm=False)
    plt.ylabel("Density")
    plt.title("Latency KDE per Mode")
    savefig("12_latency_kde_mode.png")

    # 13–16: Correlations
    metrics = ["latency_ms", "jitter_ms", "pdr", "drop_rate", "routing_overhead", "throughput_kbps"]
    corr = qos[metrics].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of QoS Metrics")
    savefig("13_qos_corr_heatmap.png")

    sns.pairplot(qos[metrics + ["mode"]], hue="mode", corner=True)
    savefig("14_qos_pairplot_mode.png")

    sns.jointplot(data=qos, x="latency_ms", y="throughput_kbps", hue="mode", kind="scatter")
    savefig("15_latency_vs_throughput_joint.png")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=qos, x="jitter_ms", y="drop_rate", hue="slice", style="mode")
    plt.xlabel("Jitter (ms)")
    plt.ylabel("Drop Rate")
    plt.title("Jitter vs Drop Rate per Slice")
    savefig("16_jitter_vs_drop_slice.png")

    # 17–20: SLA views
    if "sla_met_latency" in qos.columns:
        sla_lat_slice = qos.groupby("slice", observed=False)["sla_met_latency"].mean().reset_index()
        plt.figure(figsize=(8, 6))
        sns.barplot(data=sla_lat_slice, x="slice", y="sla_met_latency")
        plt.ylabel("Latency SLA Compliance Rate")
        plt.title("Latency SLA Compliance per Slice")
        savefig("17_sla_latency_compliance_slice.png")

    if "sla_met_pdr" in qos.columns:
        sla_pdr_slice = qos.groupby("slice", observed=False)["sla_met_pdr"].mean().reset_index()
        plt.figure(figsize=(8, 6))
        sns.barplot(data=sla_pdr_slice, x="slice", y="sla_met_pdr")
        plt.ylabel("PDR SLA Compliance Rate")
        plt.title("PDR SLA Compliance per Slice")
        savefig("18_sla_pdr_compliance_slice.png")

    if {"sla_met_latency", "sla_met_pdr"} <= set(qos.columns):
        qos["viol_latency"] = 1 - qos["sla_met_latency"]
        qos["viol_pdr"] = 1 - qos["sla_met_pdr"]
        viol = qos.groupby("mode", observed=False)[["viol_latency", "viol_pdr"]].sum().reset_index()
        viol_melt = viol.melt(id_vars="mode", var_name="metric", value_name="violations")
        plt.figure(figsize=(8, 6))
        sns.barplot(data=viol_melt, x="mode", y="violations", hue="metric")
        plt.ylabel("Violation Count")
        plt.title("SLA Violations per Mode and Metric")
        savefig("19_sla_violations_mode_metric.png")

    if "sla_met_latency" in qos.columns:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=qos, x="cycle", y="sla_met_latency", hue="mode")
        plt.ylabel("Latency SLA Met")
        plt.title("Latency SLA Compliance over Cycles per Mode")
        savefig("20_sla_latency_over_cycles_mode.png")

    # 21–24: Adaptive routing (skip if no data)
    if not routing.empty and "path_id" in routing.columns and routing["path_id"].notna().any():
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=routing, x="cycle", y="latency_ms", hue="path_id")
        plt.ylabel("Path Latency (ms)")
        plt.title("Path Latency vs Cycle per Path")
        savefig("21_path_latency_vs_cycle.png")

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=routing, x="cycle", y="loss_rate", hue="path_id")
        plt.ylabel("Loss Rate")
        plt.title("Path Loss Rate vs Cycle per Path")
        savefig("22_path_loss_vs_cycle.png")

        agg = routing.groupby(["path_id", "selected"], observed=False)["latency_ms"].mean().reset_index()
        plt.figure(figsize=(8, 6))
        sns.barplot(data=agg, x="path_id", y="latency_ms", hue="selected")
        plt.ylabel("Average Latency (ms)")
        plt.title("Average Path Latency (Selected vs Not Selected)")
        savefig("23_path_avg_latency_selected.png")

        if "slice" in routing.columns:
            freq = routing.groupby(["slice", "path_id"], observed=False)["selected"].mean().reset_index()
            freq_pivot = freq.pivot(index="slice", columns="path_id", values="selected")
            plt.figure(figsize=(8, 6))
            sns.heatmap(freq_pivot, annot=True, cmap="Blues", fmt=".2f")
            plt.title("Path Selection Frequency per Slice")
            savefig("24_path_selection_freq_slice.png")
    else:
        # Create placeholder text files to keep numbering consistent
        for i in range(21, 25):
            path = os.path.join(OUTPUT_DIR, f"{i:02d}_routing_placeholder.txt")
            with open(path, "w") as f:
                f.write("No routing data available for this plot.\n")
            print("Placeholder", path)

    # 25–31: Offload + feedback
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=offload, x="cycle", y="offload_ratio", hue="mode")
    plt.ylabel("Offload Ratio")
    plt.title("Offload Ratio vs Cycle per Mode")
    savefig("25_offload_ratio_vs_cycle_mode.png")

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=offload, x="cycle", y="threshold")
    plt.ylabel("Threshold θ(t)")
    plt.title("Adaptive Threshold Evolution vs Cycle")
    savefig("26_threshold_vs_cycle.png")

    if "slice" in qos.columns:
        merged = pd.merge(offload, qos[["cycle", "slice"]], on="cycle", how="left")
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=merged, x="slice", y="offload_ratio", hue="mode")
        plt.ylabel("Offload Ratio")
        plt.title("Offload Ratio per Slice and Mode")
        savefig("27_offload_box_slice_mode.png")

    if "slice" in qos.columns:
        lat_slice_mode = qos.groupby(["slice", "mode"], observed=False)["latency_ms"].mean().reset_index()
        plt.figure(figsize=(8, 6))
        sns.barplot(data=lat_slice_mode, x="slice", y="latency_ms", hue="mode")
        plt.ylabel("Average Latency (ms)")
        plt.title("Average Latency per Slice and Mode")
        savefig("28_avg_latency_slice_mode.png")

    if "power_wh" in offload.columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=offload, x="latency_ms", y="power_wh", hue="mode")
        plt.xlabel("Latency (ms)")
        plt.ylabel("Modeled Power (Wh)")
        plt.title("Latency–Power Trade-off per Mode")
        savefig("29_latency_vs_power_mode.png")

    plt.figure(figsize=(8, 6))
    for mode, group in qos.groupby("mode", observed=False):
        x = np.sort(group["latency_ms"].values)
        y = np.arange(1, len(x) + 1) / len(x)
        plt.step(x, y, where="post", label=str(mode))
    plt.xlabel("Latency (ms)")
    plt.ylabel("ECDF")
    plt.title("Latency ECDF per Mode")
    plt.legend()
    savefig("30_latency_ecdf_mode.png")

    plt.figure(figsize=(8, 6))
    for slice_name, group in qos.groupby("slice", observed=False):
        x = np.sort(group["jitter_ms"].values)
        y = np.arange(1, len(x) + 1) / len(x)
        plt.step(x, y, where="post", label=str(slice_name))
    plt.xlabel("Jitter (ms)")
    plt.ylabel("ECDF")
    plt.title("Jitter ECDF per Slice")
    plt.legend()
    savefig("31_jitter_ecdf_slice.png")

if __name__ == "__main__":
    main()
