"""
NSSON results plotting script using matplotlib + seaborn only.

Reads:
- shared/net_runs/*.csv     -> aggregated per-node NSSON runs
- logs/logs/telemetry/*.csv -> detailed telemetry (local vs offload, cnn vs snn, latency)
- logs/logs/telemetry_new/*.csv -> Jetson hardware metrics per node (cnn vs snn)

Produces:
- accuracy_vs_nodes_nsson.png
- local_vs_offload_vs_nodes.png
- cnn_vs_snn_usage_vs_nodes.png
- latency_vs_nodes_nsson.png
- latency_vs_nodes_hw.png
- energy_vs_nodes_hw.png
- power_vs_nodes_hw.png
"""

import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# Project root: ~/NSSON
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Directories
NET_RUNS_DIR = PROJECT_ROOT / "shared" / "shared" / "net_runs"
TELEMETRY_DIR = PROJECT_ROOT / "logs" / "logs" / "telemetry"
TELEMETRY_NEW_DIR = PROJECT_ROOT / "logs" / "logs" / "telemetry_new"
FIG_DIR = PROJECT_ROOT / "logs" / "logs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_net_runs():
    """
    Load aggregated NSSON runs from shared/net_runs.

    Expected: CSVs like nsson_cnn_100nodes_*.csv, nsson_snn_100nodes_*.csv, etc.
    Assumes each CSV has at least:
    - nodes
    - accuracy
    - maybe fields like 'mode' or 'arch' indicating cnn/snn/nsson
    """
    csv_files = sorted(NET_RUNS_DIR.glob("nsson_*_nodes_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No net_runs CSVs found in {NET_RUNS_DIR}")

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        df["source_file"] = f.name
        dfs.append(df)

    net_df = pd.concat(dfs, ignore_index=True)
    return net_df


def load_telemetry():
    """
    Load detailed telemetry from logs/logs/telemetry.

    This is where local vs offload and cnn vs snn decisions should live,
    depending on how nsson_simulate_with_telemetry.py logs them.

    You will need to adapt the column names below to your actual telemetry.
    """
    csv_files = sorted(TELEMETRY_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No telemetry CSVs found in {TELEMETRY_DIR}")

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        df["source_file"] = f.name
        dfs.append(df)

    tel_df = pd.concat(dfs, ignore_index=True)
    return tel_df


def load_telemetry_new():
    """
    Load Jetson hardware metrics from logs/logs/telemetry_new.

    These are your new nsson_*_nodes.csv files with:
    - nodes, aps, switches
    - cnn_acc, snn_acc
    - snn_batch_latency_ms
    - cnn_hw_latency_ms, snn_hw_latency_ms
    - cnn_power_mw, snn_power_mw
    - cnn_energy_mj, snn_energy_mj
    """
    csv_files = sorted(TELEMETRY_NEW_DIR.glob("nsson_*_nodes.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No telemetry_new CSVs found in {TELEMETRY_NEW_DIR}")

    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    return df


# ---------- Aggregations ----------

def aggregate_accuracy_vs_nodes(net_df):
    """
    Build a dataframe for accuracy vs nodes.

    You need to adapt the filters here to match your net_runs naming scheme
    and columns (e.g. 'arch' or 'mode').
    """
    # Example assumptions:
    # - CNN baseline files contain 'nsson_cnn_*'
    # - SNN baseline files contain 'nsson_snn_*'
    # - NSSON hybrid files are in telemetry or have a separate label

    cnn_df = net_df[net_df["source_file"].str.contains("nsson_cnn_")]
    snn_df = net_df[net_df["source_file"].str.contains("nsson_snn_")]

    # Assume each df has columns: 'nodes' and 'accuracy'
    cnn_acc = cnn_df.groupby("nodes")["accuracy"].mean().reset_index()
    cnn_acc["model"] = "CNN baseline"

    snn_acc = snn_df.groupby("nodes")["accuracy"].mean().reset_index()
    snn_acc["model"] = "SNN baseline"

    acc_df = pd.concat([cnn_acc, snn_acc], ignore_index=True)
    return acc_df


def aggregate_decisions_vs_nodes(tel_df):
    """
    Aggregate local vs offload, cnn vs snn usage per node.

    You must adapt column names here to your telemetry schema.

    For example, if telemetry has:
    - 'nodes' (number of nodes)
    - 'decision_type' in {'local', 'offload'}
    - 'model_type' in {'cnn', 'snn'}

    this function will compute percentages per node.
    """
    # Example required columns; adjust to your actual telemetry
    required_cols = ["nodes", "decision_type", "model_type"]
    for c in required_cols:
        if c not in tel_df.columns:
            raise ValueError(f"Telemetry missing expected column '{c}'. "
                             f"Update aggregate_decisions_vs_nodes to use your actual columns.")

    # Local vs offload counts
    local_offload = (
        tel_df.groupby(["nodes", "decision_type"])
        .size()
        .reset_index(name="count")
    )
    total_per_node = (
        local_offload.groupby("nodes")["count"]
        .sum()
        .reset_index(name="total")
    )
    local_offload = local_offload.merge(total_per_node, on="nodes")
    local_offload["percent"] = 100.0 * local_offload["count"] / local_offload["total"]

    # CNN vs SNN counts
    cnn_snn = (
        tel_df.groupby(["nodes", "model_type"])
        .size()
        .reset_index(name="count")
    )
    total_per_node2 = (
        cnn_snn.groupby("nodes")["count"]
        .sum()
        .reset_index(name="total")
    )
    cnn_snn = cnn_snn.merge(total_per_node2, on="nodes")
    cnn_snn["percent"] = 100.0 * cnn_snn["count"] / cnn_snn["total"]

    return local_offload, cnn_snn


def aggregate_latency_vs_nodes(tel_df):
    """
    Aggregate latency stats per node.

    Assumes telemetry has:
    - 'nodes'
    - 'latency_ms' (end-to-end latency per request)

    Returns a dataframe with median and 95th percentile per node.
    """
    if "latency_ms" not in tel_df.columns or "nodes" not in tel_df.columns:
        raise ValueError("Telemetry missing 'latency_ms' or 'nodes'. "
                         "Update aggregate_latency_vs_nodes to match your schema.")

    def p95(x):
        return x.quantile(0.95)

    grouped = tel_df.groupby("nodes")["latency_ms"].agg(
        median_latency_ms="median",
        p95_latency_ms=p95,
    ).reset_index()

    return grouped


# ---------- Plotting functions ----------

def plot_accuracy_vs_nodes(acc_df):
    plt.figure(figsize=(6, 4))
    sns.lineplot(
        data=acc_df,
        x="nodes",
        y="accuracy",
        hue="model",
        marker="o",
    )
    plt.xlabel("Number of nodes")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs nodes (CNN vs SNN baselines)")
    plt.legend()
    out_path = FIG_DIR / "accuracy_vs_nodes_nsson.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_local_vs_offload_vs_nodes(local_offload):
    plt.figure(figsize=(6, 4))
    sns.barplot(
        data=local_offload,
        x="nodes",
        y="percent",
        hue="decision_type",
    )
    plt.xlabel("Number of nodes")
    plt.ylabel("Percent of requests")
    plt.title("Local vs offload decisions vs nodes")
    plt.legend(title="Decision type")
    out_path = FIG_DIR / "local_vs_offload_vs_nodes.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_cnn_vs_snn_usage_vs_nodes(cnn_snn):
    plt.figure(figsize=(6, 4))
    sns.barplot(
        data=cnn_snn,
        x="nodes",
        y="percent",
        hue="model_type",
    )
    plt.xlabel("Number of nodes")
    plt.ylabel("Percent of requests")
    plt.title("CNN vs SNN usage vs nodes")
    plt.legend(title="Model type")
    out_path = FIG_DIR / "cnn_vs_snn_usage_vs_nodes.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_latency_vs_nodes(lat_df):
    plt.figure(figsize=(6, 4))
    sns.lineplot(
        data=lat_df,
        x="nodes",
        y="median_latency_ms",
        marker="o",
        label="Median latency",
    )
    sns.lineplot(
        data=lat_df,
        x="nodes",
        y="p95_latency_ms",
        marker="o",
        label="95th percentile latency",
    )
    plt.xlabel("Number of nodes")
    plt.ylabel("Latency (ms)")
    plt.title("Latency vs nodes (NSSON end-to-end)")
    plt.legend()
    out_path = FIG_DIR / "latency_vs_nodes_nsson.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_hw_latency_power_energy(hw_df):
    # Latency
    plt.figure(figsize=(6, 4))
    sns.lineplot(
        data=hw_df,
        x="nodes",
        y="cnn_hw_latency_ms",
        marker="o",
        label="CNN hw latency",
    )
    sns.lineplot(
        data=hw_df,
        x="nodes",
        y="snn_hw_latency_ms",
        marker="o",
        label="SNN hw latency",
    )
    plt.xlabel("Number of nodes")
    plt.ylabel("Latency (ms per batch)")
    plt.title("Hardware latency vs nodes (CNN vs SNN)")
    plt.legend()
    out_path = FIG_DIR / "latency_vs_nodes_hw.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    # Power
    plt.figure(figsize=(6, 4))
    sns.lineplot(
        data=hw_df,
        x="nodes",
        y="cnn_power_mw",
        marker="o",
        label="CNN power",
    )
    sns.lineplot(
        data=hw_df,
        x="nodes",
        y="snn_power_mw",
        marker="o",
        label="SNN power",
    )
    plt.xlabel("Number of nodes")
    plt.ylabel("Power (mW)")
    plt.title("Hardware power vs nodes (CNN vs SNN)")
    plt.legend()
    out_path = FIG_DIR / "power_vs_nodes_hw.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    # Energy
    plt.figure(figsize=(6, 4))
    sns.lineplot(
        data=hw_df,
        x="nodes",
        y="cnn_energy_mj",
        marker="o",
        label="CNN energy",
    )
    sns.lineplot(
        data=hw_df,
        x="nodes",
        y="snn_energy_mj",
        marker="o",
        label="SNN energy",
    )
    plt.xlabel("Number of nodes")
    plt.ylabel("Energy per batch (mJ)")
    plt.title("Hardware energy vs nodes (CNN vs SNN)")
    plt.legend()
    out_path = FIG_DIR / "energy_vs_nodes_hw.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    # 1. Load data
    net_df = load_net_runs()
    tel_df = load_telemetry()
    hw_df = load_telemetry_new()

    # 2. Aggregate
    acc_df = aggregate_accuracy_vs_nodes(net_df)
    local_offload, cnn_snn = aggregate_decisions_vs_nodes(tel_df)
    lat_df = aggregate_latency_vs_nodes(tel_df)

    # 3. Plot NSSON behavior
    plot_accuracy_vs_nodes(acc_df)
    plot_local_vs_offload_vs_nodes(local_offload)
    plot_cnn_vs_snn_usage_vs_nodes(cnn_snn)
    plot_latency_vs_nodes(lat_df)

    # 4. Plot hardware metrics (Jetson emulator)
    plot_hw_latency_power_energy(hw_df)

    print("Plots saved in:", FIG_DIR)


if __name__ == "__main__":
    main()
