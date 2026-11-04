#!/usr/bin/env python3
import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.25)

FIG_DIR = "logs/figures"
os.makedirs(FIG_DIR, exist_ok=True)

telemetry_files = sorted(glob.glob("logs/telemetry/*.csv"))
meta_files = sorted(glob.glob("logs/meta/*.json"))

if not telemetry_files or not meta_files:
    raise SystemExit("No telemetry/meta files found. Run the simulator first.")

telemetry = pd.concat([pd.read_csv(f) for f in telemetry_files], ignore_index=True)
summary = pd.DataFrame([json.load(open(f)) for f in meta_files])

# Clean key categorical columns
for df in (telemetry, summary):
    df["model"] = df["model"].astype(str).str.strip().str.lower()
    df["architecture"] = df["architecture"].astype(str).str.strip().str.lower()
    df["model"] = df["model"].replace({"nan": pd.NA, "none": pd.NA, "": pd.NA})
    df["architecture"] = df["architecture"].replace({"nan": pd.NA, "none": pd.NA, "": pd.NA})
    if "profile" in df.columns:
        df["profile"] = df["profile"].astype(str).str.strip().str.lower()

telemetry = telemetry.dropna(subset=["model", "architecture"]).copy()
summary = summary.dropna(subset=["model", "architecture"]).copy()

telemetry = telemetry[telemetry["model"].isin(["cnn", "snn"])].copy()
summary = summary[summary["model"].isin(["cnn", "snn"])].copy()
telemetry = telemetry[telemetry["architecture"].isin(["baseline", "nsson"])].copy()
summary = summary[summary["architecture"].isin(["baseline", "nsson"])].copy()

# Robust numeric conversion for nodes
telemetry["nodes_num"] = pd.to_numeric(telemetry["nodes"], errors="coerce")
summary["nodes_num"] = pd.to_numeric(summary["nodes"], errors="coerce")

telemetry = telemetry.dropna(subset=["nodes_num"]).copy()
summary = summary.dropna(subset=["nodes_num"]).copy()

telemetry["nodes_num"] = telemetry["nodes_num"].astype(int)
summary["nodes_num"] = summary["nodes_num"].astype(int)

node_order = [2, 10, 20, 30, 40, 50, 100, 300]
telemetry = telemetry[telemetry["nodes_num"].isin(node_order)].copy()
summary = summary[summary["nodes_num"].isin(node_order)].copy()

telemetry["nodes"] = pd.Categorical(telemetry["nodes_num"], categories=node_order, ordered=True)
summary["nodes"] = pd.Categorical(summary["nodes_num"], categories=node_order, ordered=True)

arch_palette = {"baseline": "#c44e52", "nsson": "#4c72b0"}
model_palette = {"cnn": "#dd8452", "snn": "#55a868"}

# 1 Latency vs nodes
fig, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(
    data=summary.sort_values(["architecture", "model", "nodes_num"]),
    x="nodes_num", y="avg_latency_ms",
    hue="architecture", style="model",
    markers=True, dashes=False, linewidth=2.2, errorbar=None,
    palette=arch_palette, ax=ax
)
ax.set_title("Latency vs Node Scale")
ax.set_xlabel("Number of Nodes")
ax.set_ylabel("Latency (ms)")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/latency_vs_nodes_architecture.png", dpi=600, bbox_inches="tight")
plt.close(fig)

# 2 Throughput vs nodes
fig, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(
    data=summary.sort_values(["architecture", "model", "nodes_num"]),
    x="nodes_num", y="avg_throughput_fps",
    hue="architecture", style="model",
    markers=True, dashes=False, linewidth=2.2, errorbar=None,
    palette=arch_palette, ax=ax
)
ax.set_title("Throughput vs Node Scale")
ax.set_xlabel("Number of Nodes")
ax.set_ylabel("Throughput (fps)")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/throughput_vs_nodes_architecture.png", dpi=600, bbox_inches="tight")
plt.close(fig)

# 3 Accuracy vs nodes
fig, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(
    data=summary.sort_values(["model", "architecture", "nodes_num"]),
    x="nodes_num", y="accuracy",
    hue="model", style="architecture",
    markers=True, dashes=False, linewidth=2.2, errorbar=None,
    palette=model_palette, ax=ax
)
ax.set_title("Accuracy Across Node Scales")
ax.set_xlabel("Number of Nodes")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(summary["accuracy"].min() - 0.4, summary["accuracy"].max() + 0.4)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/accuracy_vs_nodes.png", dpi=600, bbox_inches="tight")
plt.close(fig)

# 4 Power vs nodes
fig, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(
    data=summary.sort_values(["model", "architecture", "nodes_num"]),
    x="nodes_num", y="avg_power_mw",
    hue="model", style="architecture",
    markers=True, dashes=False, linewidth=2.2, errorbar=None,
    palette=model_palette, ax=ax
)
ax.set_title("Power vs Node Scale")
ax.set_xlabel("Number of Nodes")
ax.set_ylabel("Power (mW)")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/power_vs_nodes.png", dpi=600, bbox_inches="tight")
plt.close(fig)

# 5 Overview
overview = summary.groupby("architecture", as_index=False)[
    ["accuracy", "avg_latency_ms", "avg_power_mw", "avg_throughput_fps"]
].mean()

fig, axs = plt.subplots(2, 2, figsize=(10, 7))
sns.barplot(data=overview, x="architecture", y="accuracy", hue="architecture", palette=arch_palette, legend=False, ax=axs[0, 0])
sns.barplot(data=overview, x="architecture", y="avg_latency_ms", hue="architecture", palette=arch_palette, legend=False, ax=axs[0, 1])
sns.barplot(data=overview, x="architecture", y="avg_power_mw", hue="architecture", palette=arch_palette, legend=False, ax=axs[1, 0])
sns.barplot(data=overview, x="architecture", y="avg_throughput_fps", hue="architecture", palette=arch_palette, legend=False, ax=axs[1, 1])

axs[0, 0].set_title("Accuracy")
axs[0, 1].set_title("Latency")
axs[1, 0].set_title("Power")
axs[1, 1].set_title("Throughput")
for ax in axs.flat:
    ax.set_xlabel("")
fig.subplots_adjust(wspace=0.28, hspace=0.35, top=0.90)
fig.savefig(f"{FIG_DIR}/baseline_vs_nsson_overview.png", dpi=600, bbox_inches="tight")
plt.close(fig)

# 6 Latency distribution
sampled = telemetry.sample(n=min(25000, len(telemetry)), random_state=42)
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=sampled, x="architecture", y="total_latency_ms", hue="model", ax=ax)
ax.set_title("Latency Distribution by Architecture")
ax.set_xlabel("Architecture")
ax.set_ylabel("Latency (ms)")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/latency_distribution_architecture.png", dpi=600, bbox_inches="tight")
plt.close(fig)

# 7 Tradeoff
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(
    data=summary,
    x="avg_power_mw", y="avg_throughput_fps",
    hue="architecture", style="model", s=120, palette=arch_palette, ax=ax
)
for _, row in summary.iterrows():
    ax.text(float(row["avg_power_mw"]) + 1.2, float(row["avg_throughput_fps"]), str(int(row["nodes_num"])), fontsize=8)
ax.set_title("Throughput-Power Tradeoff")
ax.set_xlabel("Power (mW)")
ax.set_ylabel("Throughput (fps)")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/throughput_power_tradeoff.png", dpi=600, bbox_inches="tight")
plt.close(fig)

# 8 Threshold evolution ablation
thr = telemetry.loc[
    telemetry["architecture"] == "nsson",
    ["model", "nodes_num", "cycle_id", "threshold"]
].copy()

thr["model"] = thr["model"].astype(str).str.strip().str.lower()
thr = thr[thr["model"].isin(["cnn", "snn"])].copy()
thr["nodes_num"] = pd.to_numeric(thr["nodes_num"], errors="coerce")
thr = thr.dropna(subset=["nodes_num"]).copy()
thr["nodes_num"] = thr["nodes_num"].astype(int)
thr = thr[thr["nodes_num"].isin(node_order)].copy()

thr = (
    thr.groupby(["model", "nodes_num", "cycle_id"], as_index=False)["threshold"]
    .mean()
    .sort_values(["model", "nodes_num", "cycle_id"])
)

# Stable downsampling without groupby.apply
thr_parts = []
for (model, nodes_num), grp in thr.groupby(["model", "nodes_num"], sort=True):
    grp = grp.sort_values("cycle_id").iloc[::20].copy()
    grp["model"] = model
    grp["nodes_num"] = nodes_num
    thr_parts.append(grp)

thr = pd.concat(thr_parts, ignore_index=True) if thr_parts else pd.DataFrame(
    columns=["model", "nodes_num", "cycle_id", "threshold"]
)

fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharey=True)
axes = axes.ravel()
node_list = sorted(thr["nodes_num"].unique()) if not thr.empty else []

for i, nodes in enumerate(node_list):
    ax = axes[i]
    sub = thr[thr["nodes_num"] == nodes]
    sns.lineplot(
        data=sub,
        x="cycle_id",
        y="threshold",
        hue="model",
        errorbar=None,
        palette=model_palette,
        ax=ax
    )
    ax.set_title(f"Nodes: {nodes}")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Threshold")
    if ax.legend_:
        ax.legend_.remove()

for j in range(len(node_list), len(axes)):
    axes[j].axis("off")

if len(node_list) > 0:
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

fig.suptitle("Threshold Evolution Ablation (NSSON)", y=0.98)
fig.subplots_adjust(top=0.88, wspace=0.22, hspace=0.35)
fig.savefig(f"{FIG_DIR}/threshold_evolution_ablation.png", dpi=600, bbox_inches="tight")
plt.close(fig)

# 9 Decision ablation
decision_counts = telemetry.groupby(["architecture", "model", "decision"], as_index=False).size()
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=decision_counts, x="decision", y="size", hue="architecture", palette=arch_palette, ax=ax)
ax.set_title("Decision Ablation")
ax.set_xlabel("Decision")
ax.set_ylabel("Count")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/decision_ablation.png", dpi=600, bbox_inches="tight")
plt.close(fig)

# 10 Improvement heatmap
base_lat = summary[summary["architecture"] == "baseline"].pivot_table(
    index="model",
    columns="nodes_num",
    values="avg_latency_ms",
    aggfunc="mean"
)
nsson_lat = summary[summary["architecture"] == "nsson"].pivot_table(
    index="model",
    columns="nodes_num",
    values="avg_latency_ms",
    aggfunc="mean"
)
lat_improve = ((base_lat - nsson_lat) / base_lat) * 100.0

fig, ax = plt.subplots(figsize=(9, 3.5))
sns.heatmap(lat_improve, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax)
ax.set_title("Latency Improvement of NSSON over Baseline (%)")
ax.set_xlabel("Nodes")
ax.set_ylabel("Model")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/latency_improvement_heatmap.png", dpi=600, bbox_inches="tight")
plt.close(fig)

print(f"Saved plots to {FIG_DIR}")
