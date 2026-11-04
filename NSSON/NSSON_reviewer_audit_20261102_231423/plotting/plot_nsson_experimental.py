import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEL_DIR = PROJECT_ROOT / "logs" / "logs" / "telemetry_nsson_experimental"
FIG_DIR = PROJECT_ROOT / "logs" / "logs" / "figures_nsson_experimental"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_nsson_experimental():
    csv_files = sorted(TEL_DIR.glob("nsson_experimental_nodes_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No NSSON experimental telemetry in {TEL_DIR}")

    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    print("Loaded NSSON experimental files:", [f.name for f in csv_files])
    print("Columns:", df.columns.tolist())
    return df


def plot_accuracy_vs_nodes(df):
    acc_df = (
        df.groupby(["nodes", "model_used"])["accuracy"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(6, 4))
    sns.lineplot(
        data=acc_df,
        x="nodes",
        y="accuracy",
        hue="model_used",
        marker="o",
    )
    plt.xlabel("Number of nodes")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs nodes (NSSON experimental CNN vs SNN)")
    plt.legend(title="Model")
    out = FIG_DIR / "accuracy_vs_nodes_nsson_experimental.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print("Saved:", out)


def plot_latency_vs_nodes(df):
    lat_df = (
        df.groupby(["nodes", "model_used"])["latency_ms"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(6, 4))
    sns.lineplot(
        data=lat_df,
        x="nodes",
        y="latency_ms",
        hue="model_used",
        marker="o",
    )
    plt.xlabel("Number of nodes")
    plt.ylabel("Latency (ms per batch)")
    plt.title("Hardware latency vs nodes (NSSON experimental CNN vs SNN)")
    plt.legend(title="Model")
    out = FIG_DIR / "latency_vs_nodes_nsson_experimental.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print("Saved:", out)


def plot_power_vs_nodes(df):
    pwr_df = (
        df.groupby(["nodes", "model_used"])["power_mw"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(6, 4))
    sns.lineplot(
        data=pwr_df,
        x="nodes",
        y="power_mw",
        hue="model_used",
        marker="o",
    )
    plt.xlabel("Number of nodes")
    plt.ylabel("Power (mW)")
    plt.title("Hardware power vs nodes (NSSON experimental CNN vs SNN)")
    plt.legend(title="Model")
    out = FIG_DIR / "power_vs_nodes_nsson_experimental.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print("Saved:", out)


def plot_energy_vs_nodes(df):
    eng_df = (
        df.groupby(["nodes", "model_used"])["energy_mj"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(6, 4))
    sns.lineplot(
        data=eng_df,
        x="nodes",
        y="energy_mj",
        hue="model_used",
        marker="o",
    )
    plt.xlabel("Number of nodes")
    plt.ylabel("Energy per batch (mJ)")
    plt.title("Hardware energy vs nodes (NSSON experimental CNN vs SNN)")
    plt.legend(title="Model")
    out = FIG_DIR / "energy_vs_nodes_nsson_experimental.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print("Saved:", out)


def main():
    df = load_nsson_experimental()
    plot_accuracy_vs_nodes(df)
    plot_latency_vs_nodes(df)
    plot_power_vs_nodes(df)
    plot_energy_vs_nodes(df)
    print("NSSON experimental plots in:", FIG_DIR)


if __name__ == "__main__":
    main()
