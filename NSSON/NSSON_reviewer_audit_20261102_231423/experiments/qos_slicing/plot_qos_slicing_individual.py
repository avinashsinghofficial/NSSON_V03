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
    "priority": "Binary Priority",
    "slicing": "Per-Slice SLA",
    "adaptive": "SLA + Adaptive Routing",
}

METRICS = {
    "rtt_avg_ms": {
        "title": "Inference-Slice RTT Under Background Load",
        "ylabel": "Average RTT (ms)",
        "file": "slice_rtt_comparison.png",
    },
    "loss_percent": {
        "title": "Inference-Slice Packet Loss Under Background Load",
        "ylabel": "ICMP packet loss (%)",
        "file": "slice_packet_loss_comparison.png",
    },
    "throughput_mbps": {
        "title": "Inference-Slice TCP Throughput Under Background Load",
        "ylabel": "TCP throughput (Mbit/s)",
        "file": "slice_throughput_comparison.png",
    },
    "sla_violation": {
        "title": "Per-Slice SLA Violation Rate Under Background Load",
        "ylabel": "SLA violation rate",
        "file": "slice_sla_violation_comparison.png",
    },
}


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input telemetry does not exist: {INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)

    df = df[
        df["mode"].isin(MODES)
        & df["slice"].isin(SLICES)
        & df["load"].isin(LOADS)
    ].copy()

    df["mode"] = pd.Categorical(df["mode"], categories=MODES, ordered=True)
    df["slice"] = pd.Categorical(df["slice"], categories=SLICES, ordered=True)
    df["load"] = pd.Categorical(df["load"], categories=LOADS, ordered=True)

    sns.set_theme(style="whitegrid", context="talk")

    for metric, specification in METRICS.items():
        figure, axes = plt.subplots(
            1,
            4,
            figsize=(23, 5.5),
            sharey=False,
        )

        for axis, mode in zip(axes, MODES):
            subset = df[df["mode"] == mode]

            sns.barplot(
                data=subset,
                x="load",
                y=metric,
                hue="slice",
                order=LOADS,
                hue_order=SLICES,
                errorbar="sd",
                ax=axis,
            )

            axis.set_title(MODE_NAMES[mode])
            axis.set_xlabel("Background load")
            axis.set_ylabel(
                specification["ylabel"] if axis == axes[0] else ""
            )
            axis.grid(True, axis="y", alpha=0.4)

            legend = axis.get_legend()
            if legend is not None:
                legend.remove()

            if metric == "sla_violation":
                axis.set_ylim(-0.02, 1.02)

        handles, labels = axes[0].get_legend_handles_labels()

        figure.legend(
            handles,
            labels,
            loc="upper center",
            ncol=3,
            bbox_to_anchor=(0.5, 1.10),
            frameon=True,
        )

        figure.suptitle(
            specification["title"],
            fontsize=18,
            fontweight="bold",
            y=1.18,
        )

        figure.tight_layout()

        output_file = OUTPUT_DIR / specification["file"]

        figure.savefig(
            output_file,
            dpi=450,
            bbox_inches="tight",
        )

        plt.close(figure)

        print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
