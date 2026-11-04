#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


NSSON_ROOT = Path("/home/avi/NSSON_V02")

INPUT_CSV = NSSON_ROOT / "results/qos_slicing/qos_slice_metrics.csv"
OUTPUT_DIR = NSSON_ROOT / "plots_qos_slicing"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODES = ["baseline", "priority", "slicing", "adaptive"]
LOADS = ["light", "moderate", "heavy"]
SLICES = ["autonomous", "healthcare", "industrial"]

MODE_NAMES = {
    "baseline": "Baseline SDN-IoT",
    "priority": "NSSON: Binary Priority",
    "slicing": "NSSON: Slice + SLA",
    "adaptive": "NSSON: Slice + SLA + Routing",
}

SLICE_NAMES = {
    "autonomous": "Autonomous",
    "healthcare": "Healthcare",
    "industrial": "Industrial",
}

SLICE_COLORS = {
    "autonomous": "#D62728",
    "healthcare": "#1F77B4",
    "industrial": "#2CA02C",
}

METRICS = [
    ("rtt_avg_ms", "Average RTT (ms)", False),
    ("loss_percent", "ICMP packet loss (%)", False),
    ("throughput_mbps", "TCP throughput (Mbit/s)", True),
    ("sla_violation", "SLA violation rate", False),
]


def aggregate(dataframe, metric):
    return (
        dataframe
        .groupby(["mode", "slice", "load"], observed=True)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input telemetry does not exist: {INPUT_CSV}\n"
            "Run the QoS slicing experiment first."
        )

    df = pd.read_csv(INPUT_CSV)

    required_columns = {
        "mode",
        "slice",
        "load",
        "rtt_avg_ms",
        "loss_percent",
        "throughput_mbps",
        "sla_violation",
    }

    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV misses required columns: {sorted(missing)}"
        )

    df = df[
        df["mode"].isin(MODES)
        & df["slice"].isin(SLICES)
        & df["load"].isin(LOADS)
    ].copy()

    df["mode"] = pd.Categorical(df["mode"], categories=MODES, ordered=True)
    df["slice"] = pd.Categorical(df["slice"], categories=SLICES, ordered=True)
    df["load"] = pd.Categorical(df["load"], categories=LOADS, ordered=True)

    summaries = {
        metric: aggregate(df, metric)
        for metric, _, _ in METRICS
    }

    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
    })

    figure, axes = plt.subplots(
        nrows=4,
        ncols=4,
        figsize=(18, 16),
        sharex="col",
    )

    x_values = [0, 1, 2]
    x_labels = ["Light\n(2 Mb/s)", "Moderate\n(12 Mb/s)", "Heavy\n(19 Mb/s)"]

    for column_index, mode in enumerate(MODES):
        axes[0, column_index].set_title(
            MODE_NAMES[mode],
            fontweight="bold",
        )

        for row_index, (metric, y_label, higher_is_better) in enumerate(METRICS):
            axis = axes[row_index, column_index]
            metric_summary = summaries[metric]
            mode_summary = metric_summary[metric_summary["mode"] == mode]

            for slice_name in SLICES:
                sub = mode_summary[
                    mode_summary["slice"] == slice_name
                ].copy()

                sub["load_order"] = sub["load"].map(
                    {"light": 0, "moderate": 1, "heavy": 2}
                )
                sub = sub.sort_values("load_order")

                if sub.empty:
                    continue

                axis.errorbar(
                    x_values,
                    sub["mean"].to_numpy(),
                    yerr=sub["std"].fillna(0.0).to_numpy(),
                    marker="o",
                    markersize=5,
                    linewidth=2.0,
                    capsize=3,
                    color=SLICE_COLORS[slice_name],
                    label=SLICE_NAMES[slice_name],
                )

            axis.set_xticks(x_values)
            axis.set_xticklabels(x_labels)
            axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

            if column_index == 0:
                axis.set_ylabel(y_label)
            else:
                axis.set_ylabel("")

            if row_index < 3:
                axis.set_xlabel("")
            else:
                axis.set_xlabel("Background load")

            axis.set_ylim(bottom=0)

            if metric == "sla_violation":
                axis.set_ylim(-0.02, 1.02)

    handles, labels = axes[0, 0].get_legend_handles_labels()

    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.985),
        frameon=True,
    )

    figure.suptitle(
        "Multi-Slice QoS, SLA Compliance, and Adaptive-Routing Evaluation",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    figure.tight_layout(rect=[0.01, 0.01, 1.0, 0.95])

    png_file = OUTPUT_DIR / "qos_slicing_adaptive_4x4_panel.png"
    pdf_file = OUTPUT_DIR / "qos_slicing_adaptive_4x4_panel.pdf"

    figure.savefig(
        png_file,
        dpi=450,
        bbox_inches="tight",
    )

    figure.savefig(
        pdf_file,
        bbox_inches="tight",
    )

    plt.close(figure)

    statistics = []

    for metric, _, _ in METRICS:
        summary = summaries[metric].copy()
        summary["metric"] = metric
        statistics.append(summary)

    statistics_df = pd.concat(statistics, ignore_index=True)
    statistics_file = OUTPUT_DIR / "qos_slicing_summary_statistics.csv"
    statistics_df.to_csv(statistics_file, index=False)

    print(f"Saved: {png_file}")
    print(f"Saved: {pdf_file}")
    print(f"Saved: {statistics_file}")


if __name__ == "__main__":
    main()
