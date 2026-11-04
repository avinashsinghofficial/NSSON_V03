import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TELEMETRY_DIR = PROJECT_ROOT / "logs" / "logs" / "telemetry_new"
FIG_DIR = PROJECT_ROOT / "logs" / "logs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_telemetry():
    csv_files = sorted(TELEMETRY_DIR.glob("nsson_*_nodes.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSVs found in {TELEMETRY_DIR}")
    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    return df


def plot_latency_vs_nodes(df):
    plt.figure(figsize=(6, 4))
    sns.lineplot(
        data=df,
        x="nodes",
        y="cnn_hw_latency_ms",
        marker="o",
        label="CNN hw latency",
    )
    sns.lineplot(
        data=df,
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


def plot_energy_vs_nodes(df):
    plt.figure(figsize=(6, 4))
    sns.lineplot(
        data=df,
        x="nodes",
        y="cnn_energy_mj",
        marker="o",
        label="CNN energy",
    )
    sns.lineplot(
        data=df,
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


def plot_power_vs_nodes(df):
    plt.figure(figsize=(6, 4))
    sns.lineplot(
        data=df,
        x="nodes",
        y="cnn_power_mw",
        marker="o",
        label="CNN power",
    )
    sns.lineplot(
        data=df,
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


def main():
    df = load_telemetry()
    print(df[["nodes", "cnn_acc", "snn_acc"]])
    plot_latency_vs_nodes(df)
    plot_energy_vs_nodes(df)
    plot_power_vs_nodes(df)


if __name__ == "__main__":
    main()
