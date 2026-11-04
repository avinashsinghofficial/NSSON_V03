#!/usr/bin/env python3
import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# TEMPLATE / STYLE ONLY
# =========================================================
FIG_DIR = "logs/figures_templated"
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 600,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
    "font.family": "sans-serif",
    "grid.alpha": 0.28,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Architecture colors: without NSSON vs with NSSON
ARCH_COLORS = {
    "without NSSON": "#d62728",  # red
    "with NSSON": "#1f77b4",     # blue
}

# Model colors (when needed)
MODEL_COLORS = {
    "cnn": "#1f77b4",
    "snn": "#ff7f0e",
}

DECISION_COLORS = {
    "local": "#4c72b0",
    "offload": "#dd8452",
}

THRESHOLD_COLOR = "#2ca02c"
POWER_COLOR = "#d62728"

NODE_ORDER = [2, 10, 20, 30, 40, 50, 100, 300]

def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name), bbox_inches="tight")
    plt.close(fig)

def clean_frame(df):
    df = df.copy()
    for col in ["model", "architecture", "decision"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .replace({"nan": pd.NA, "none": pd.NA, "": pd.NA})
            )
    if "nodes" in df.columns:
        df["nodes_num"] = pd.to_numeric(df["nodes"], errors="coerce")
    elif "nodes_num" not in df.columns:
        df["nodes_num"] = pd.NA
    return df

# =========================================================
# LOAD TODAY'S REAL DATA
# =========================================================
telemetry_files = sorted(glob.glob("logs/telemetry/*.csv"))
meta_files = sorted(glob.glob("logs/meta/*.json"))

if not telemetry_files:
    raise SystemExit("No files found in logs/telemetry/")
if not meta_files:
    raise SystemExit("No files found in logs/meta/")

telemetry = pd.concat([pd.read_csv(f) for f in telemetry_files], ignore_index=True)
summary = pd.DataFrame([json.load(open(f)) for f in meta_files])

telemetry = clean_frame(telemetry)
summary = clean_frame(summary)

telemetry = telemetry.dropna(subset=["nodes_num"]).copy()
summary = summary.dropna(subset=["nodes_num"]).copy()

telemetry["nodes_num"] = telemetry["nodes_num"].astype(int)
summary["nodes_num"] = summary["nodes_num"].astype(int)

telemetry = telemetry[telemetry["nodes_num"].isin(NODE_ORDER)].copy()
summary = summary[summary["nodes_num"].isin(NODE_ORDER)].copy()

if "model" in telemetry.columns:
    telemetry = telemetry[telemetry["model"].isin(["cnn", "snn"])].copy()
if "model" in summary.columns:
    summary = summary[summary["model"].isin(["cnn", "snn"])].copy()

if "architecture" in telemetry.columns:
    telemetry = telemetry[telemetry["architecture"].isin(["baseline", "nsson"])].copy()
if "architecture" in summary.columns:
    summary = summary[summary["architecture"].isin(["baseline", "nsson"])].copy()

# Map architecture to clear labels for plotting
ARCH_LABELS = {"baseline": "without NSSON", "nsson": "with NSSON"}
summary["arch_label"] = summary["architecture"].map(ARCH_LABELS)
telemetry["arch_label"] = telemetry.get("architecture", pd.NA)
telemetry["arch_label"] = telemetry["arch_label"].map(ARCH_LABELS)

# =========================================================
# FIGURE 1: Overview with and without NSSON
# =========================================================
overview = summary.groupby("arch_label", as_index=False)[
    ["accuracy", "avg_latency_ms", "avg_power_mw", "avg_throughput_fps"]
].mean()

fig, axs = plt.subplots(2, 2, figsize=(10, 7))
metrics = [
    ("accuracy", "Accuracy"),
    ("avg_latency_ms", "Latency (ms)"),
    ("avg_power_mw", "Power (mW)"),
    ("avg_throughput_fps", "Throughput (fps)")
]

for ax, (metric, title) in zip(axs.flat, metrics):
    sns.barplot(
        data=overview,
        x="arch_label",
        y=metric,
        hue="arch_label",
        palette=ARCH_COLORS,
        legend=False,
        ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")

fig.suptitle("Overview: With and Without NSSON", fontsize=16, fontweight="bold", y=1.02)
fig.subplots_adjust(wspace=0.25, hspace=0.35)
savefig(fig, "baseline_vs_nsson_overview_templated.png")

# =========================================================
# FIGURE 2: Latency vs nodes (color-only, with/without NSSON)
# =========================================================
fig, ax = plt.subplots(figsize=(8.8, 5.2))
lat_df = summary.sort_values(["arch_label", "nodes_num"])

sns.lineplot(
    data=lat_df,
    x="nodes_num",
    y="avg_latency_ms",
    hue="arch_label",
    palette=ARCH_COLORS,
    linewidth=2.2,
    errorbar=None,
    ax=ax
)
ax.set_title("Latency vs Node Scale (with/without NSSON)")
ax.set_xlabel("Nodes")
ax.set_ylabel("Latency (ms)")
ax.legend(title="Architecture")
savefig(fig, "latency_vs_nodes_architecture_templated.png")

# =========================================================
# FIGURE 3: Throughput vs nodes (color-only, with/without NSSON)
# =========================================================
fig, ax = plt.subplots(figsize=(8.8, 5.2))
thr_df = summary.sort_values(["arch_label", "nodes_num"])

sns.lineplot(
    data=thr_df,
    x="nodes_num",
    y="avg_throughput_fps",
    hue="arch_label",
    palette=ARCH_COLORS,
    linewidth=2.2,
    errorbar=None,
    ax=ax
)
ax.set_title("Throughput vs Node Scale (with/without NSSON)")
ax.set_xlabel("Nodes")
ax.set_ylabel("Throughput (fps)")
ax.legend(title="Architecture")
savefig(fig, "throughput_vs_nodes_architecture_templated.png")

# =========================================================
# FIGURE 4: Power vs nodes (color-only, with/without NSSON)
# =========================================================
fig, ax = plt.subplots(figsize=(8.8, 5.2))
pwr_df = summary.sort_values(["arch_label", "nodes_num"])

sns.lineplot(
    data=pwr_df,
    x="nodes_num",
    y="avg_power_mw",
    hue="arch_label",
    palette=ARCH_COLORS,
    linewidth=2.2,
    errorbar=None,
    ax=ax
)
ax.set_title("Power vs Node Scale (with/without NSSON)")
ax.set_xlabel("Nodes")
ax.set_ylabel("Power (mW)")
ax.legend(title="Architecture")
savefig(fig, "power_vs_nodes_templated.png")

# =========================================================
# FIGURE 5: Accuracy vs nodes (color-only, with/without NSSON)
# =========================================================
fig, ax = plt.subplots(figsize=(8.8, 5.2))
acc_df = summary.sort_values(["arch_label", "nodes_num"])

sns.lineplot(
    data=acc_df,
    x="nodes_num",
    y="accuracy",
    hue="arch_label",
    palette=ARCH_COLORS,
    linewidth=2.2,
    errorbar=None,
    ax=ax
)
ax.set_title("Accuracy Across Node Scales (with/without NSSON)")
ax.set_xlabel("Nodes")
ax.set_ylabel("Accuracy (%)")
ax.legend(title="Architecture")
savefig(fig, "accuracy_vs_nodes_templated.png")

# =========================================================
# FIGURE 6: Latency distribution
# =========================================================
sampled = telemetry.sample(n=min(25000, len(telemetry)), random_state=42)

fig, ax = plt.subplots(figsize=(8.8, 5.2))
sns.histplot(
    data=sampled,
    x="total_latency_ms",
    bins=24,
    color="#4c72b0",
    alpha=0.75,
    ax=ax
)
ax.set_title("Latency Distribution Across All Nodes")
ax.set_xlabel("Latency (ms)")
ax.set_ylabel("Count")
savefig(fig, "latency_distribution_all_templated.png")

# =========================================================
# FIGURE 7: Local vs offload latency box
# =========================================================
if "decision" in telemetry.columns:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    sns.boxplot(
        data=sampled[sampled["decision"].isin(["local", "offload"])],
        x="decision",
        y="total_latency_ms",
        hue="decision",
        palette=DECISION_COLORS,
        legend=False,
        ax=ax
    )
    ax.set_title("Local vs Offload Latency")
    ax.set_xlabel("Decision")
    ax.set_ylabel("Latency (ms)")
    savefig(fig, "local_offload_latency_box_templated.png")

# =========================================================
# FIGURE 8: Local vs offload latency across nodes (color-only)
# =========================================================
if {"decision", "nodes_num", "total_latency_ms"}.issubset(telemetry.columns):
    line_df = (
        telemetry[telemetry["decision"].isin(["local", "offload"])]
        .groupby(["nodes_num", "decision"], as_index=False)["total_latency_ms"]
        .mean()
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    sns.lineplot(
        data=line_df.sort_values(["decision", "nodes_num"]),
        x="nodes_num",
        y="total_latency_ms",
        hue="decision",
        palette=DECISION_COLORS,
        linewidth=2.2,
        ax=ax
    )
    ax.set_title("Local vs Offload Latency Across Nodes")
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Average Latency (ms)")
    ax.legend(title="Decision")
    savefig(fig, "local_vs_offload_latency_templated.png")

# =========================================================
# FIGURE 9: Decision ratio by node count (stacked bars)
# =========================================================
if {"decision", "nodes_num"}.issubset(telemetry.columns):
    count_df = (
        telemetry[telemetry["decision"].isin(["local", "offload"])]
        .groupby(["nodes_num", "decision"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )

    pivot = count_df.pivot_table(
        index="nodes_num", columns="decision", values="count", aggfunc="sum", fill_value=0
    ).reset_index()

    for col in ["local", "offload"]:
        if col not in pivot.columns:
            pivot[col] = 0

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.bar(pivot["nodes_num"], pivot["local"], label="local", color=DECISION_COLORS["local"])
    ax.bar(
        pivot["nodes_num"],
        pivot["offload"],
        bottom=pivot["local"],
        label="offload",
        color=DECISION_COLORS["offload"]
    )
    ax.set_title("Decision Ratio by Node Count")
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Count")
    ax.legend(title="Decision")
    savefig(fig, "decision_ratio_nodes_templated.png")

# =========================================================
# FIGURE 10: Adaptive threshold across nodes
# =========================================================
if {"nodes_num", "threshold", "architecture"}.issubset(telemetry.columns):
    thr_df = (
        telemetry[telemetry["architecture"] == "nsson"]
        .groupby("nodes_num", as_index=False)
        .agg(
            mean=("threshold", "mean"),
            std=("threshold", "std")
        )
        .sort_values("nodes_num")
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(
        thr_df["nodes_num"],
        thr_df["mean"],
        color=THRESHOLD_COLOR,
        linewidth=2.2
    )
    ax.fill_between(
        thr_df["nodes_num"].values,
        (thr_df["mean"] - thr_df["std"].fillna(0)).values,
        (thr_df["mean"] + thr_df["std"].fillna(0)).values,
        color=THRESHOLD_COLOR,
        alpha=0.18
    )
    ax.set_title("Adaptive Threshold Evolution Across Nodes (NSSON)")
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Threshold")
    savefig(fig, "adaptive_threshold_all_templated.png")
    
# =========================================================
# FIGURE 11: Power scatter across all nodes
# =========================================================
if {"nodes_num", "effective_power_mw"}.issubset(telemetry.columns):
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.scatter(
        telemetry["nodes_num"],
        telemetry["effective_power_mw"],
        alpha=0.45,
        s=18,
        color=POWER_COLOR
    )
    ax.set_title("Power Scatter Across All Nodes")
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Power (mW)")
    savefig(fig, "power_scatter_all_templated.png")

# =========================================================
# FIGURE 12: Offload telemetry panel (color-only)
# =========================================================
if {"decision", "nodes_num", "total_latency_ms", "threshold", "effective_power_mw"}.issubset(telemetry.columns):
    fig, axs = plt.subplots(2, 2, figsize=(13.5, 9))
    fig.suptitle("Offload Telemetry — Multi-Node Results", fontsize=16, fontweight="bold", y=0.98)

    ratio_df = (
        telemetry[telemetry["decision"].isin(["local", "offload"])]
        .groupby(["nodes_num", "decision"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    ratio_df["ratio"] = ratio_df.groupby("nodes_num")["count"].transform(lambda s: s / s.sum())

    sns.lineplot(
        data=ratio_df,
        x="nodes_num",
        y="ratio",
        hue="decision",
        palette=DECISION_COLORS,
        linewidth=2.0,
        ax=axs[0, 0]
    )
    axs[0, 0].set_title("Decision Ratio vs Nodes")
    axs[0, 0].set_xlabel("Nodes")
    axs[0, 0].set_ylabel("Ratio")
    axs[0, 0].legend(title="Decision")

    latency_box = telemetry[telemetry["decision"].isin(["local", "offload"])].copy()
    sns.boxplot(
        data=latency_box,
        x="nodes_num",
        y="total_latency_ms",
        hue="decision",
        palette=DECISION_COLORS,
        ax=axs[0, 1]
    )
    axs[0, 1].set_title("Latency Distribution by Decision")
    axs[0, 1].set_xlabel("Nodes")
    axs[0, 1].set_ylabel("Latency (ms)")

    axs[1, 0].scatter(
        telemetry["nodes_num"],
        telemetry["effective_power_mw"],
        alpha=0.45,
        s=14,
        color=POWER_COLOR
    )
    axs[1, 0].set_title("Power Scatter Across Nodes")
    axs[1, 0].set_xlabel("Nodes")
    axs[1, 0].set_ylabel("Power (mW)")

    thr_panel = (
        telemetry[telemetry["architecture"] == "nsson"]
        .groupby("nodes_num", as_index=False)["threshold"]
        .mean()
    )
    axs[1, 1].plot(
        thr_panel["nodes_num"],
        thr_panel["threshold"],
        color=THRESHOLD_COLOR,
        linewidth=2.1
    )
    axs[1, 1].set_title("Adaptive Threshold vs Nodes (NSSON)")
    axs[1, 1].set_xlabel("Nodes")
    axs[1, 1].set_ylabel("Threshold")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIG_DIR, "nsson_offload_telemetry_templated.png"), bbox_inches="tight")
    plt.close(fig)

# =========================================================
# FIGURE 13: Latency improvement heatmap
# =========================================================
base_lat = summary[summary["architecture"] == "baseline"].pivot_table(
    index="model", columns="nodes_num", values="avg_latency_ms", aggfunc="mean"
)
nsson_lat = summary[summary["architecture"] == "nsson"].pivot_table(
    index="model", columns="nodes_num", values="avg_latency_ms", aggfunc="mean"
)
lat_improve = ((base_lat - nsson_lat) / base_lat) * 100.0

fig, ax = plt.subplots(figsize=(9.5, 3.8))
sns.heatmap(lat_improve, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax)
ax.set_title("Latency Improvement of NSSON over Baseline (%)")
ax.set_xlabel("Nodes")
ax.set_ylabel("Model")
savefig(fig, "latency_improvement_heatmap_templated.png")

print(f"Templated figures saved in: {FIG_DIR}")
for f in sorted(os.listdir(FIG_DIR)):
    print(" -", f)
