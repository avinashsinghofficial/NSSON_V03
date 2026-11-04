#!/usr/bin/env python3
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_DIR = PROJECT_ROOT / "latest_experiment"
LOG_DIR = BASE_DIR / "logs" / "telemetry_nsson_experimental"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_all_csv():
    rows = []
    for csv_path in sorted(LOG_DIR.glob("nsson_experimental_nodes_*.csv")):
        df = pd.read_csv(csv_path)
        rows.append(df)
    if not rows:
        raise RuntimeError(f"No CSV files found in {LOG_DIR}")
    return pd.concat(rows, ignore_index=True)


def plot_latency_vs_nodes(df):
    fig, ax = plt.subplots()
    for model in ["cnn", "snn"]:
        sub = df[df["model_used"] == model]
        ax.plot(
            sub["nodes"],
            sub["latency_ms"],
            marker="o",
            label=model.upper(),
        )
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency vs nodes (latest NSSON experimental)")
    ax.legend()
    fig.tight_layout()
    fig_path = FIG_DIR / "latency_vs_nodes_nsson_experimental_latest.png"
    fig.savefig(fig_path, dpi=300)
    print("Saved", fig_path)


def plot_power_vs_nodes(df):
    fig, ax = plt.subplots()
    for model in ["cnn", "snn"]:
        sub = df[df["model_used"] == model]
        ax.plot(
            sub["nodes"],
            sub["power_mw"],
            marker="s",
            label=model.upper(),
        )
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Power (mW)")
    ax.set_title("Power vs nodes (latest NSSON experimental)")
    ax.legend()
    fig.tight_layout()
    fig_path = FIG_DIR / "power_vs_nodes_nsson_experimental_latest.png"
    fig.savefig(fig_path, dpi=300)
    print("Saved", fig_path)


def plot_energy_vs_nodes(df):
    fig, ax = plt.subplots()
    for model in ["cnn", "snn"]:
        sub = df[df["model_used"] == model]
        ax.plot(
            sub["nodes"],
            sub["energy_mj"],
            marker="^",
            label=model.upper(),
        )
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Energy (mJ)")
    ax.set_title("Energy vs nodes (latest NSSON experimental)")
    ax.legend()
    fig.tight_layout()
    fig_path = FIG_DIR / "energy_vs_nodes_nsson_experimental_latest.png"
    fig.savefig(fig_path, dpi=300)
    print("Saved", fig_path)


def plot_accuracy_vs_nodes(df):
    fig, ax = plt.subplots()
    for model in ["cnn", "snn"]:
        sub = df[df["model_used"] == model]
        ax.plot(
            sub["nodes"],
            sub["accuracy"],
            marker="d",
            label=model.upper(),
        )
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs nodes (latest NSSON experimental)")
    ax.legend()
    fig.tight_layout()
    fig_path = FIG_DIR / "accuracy_vs_nodes_nsson_experimental_latest.png"
    fig.savefig(fig_path, dpi=300)
    print("Saved", fig_path)


def main():
    df = load_all_csv()
    plot_latency_vs_nodes(df)
    plot_power_vs_nodes(df)
    plot_energy_vs_nodes(df)
    plot_accuracy_vs_nodes(df)


if __name__ == "__main__":
    main()
