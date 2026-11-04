#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path("/home/avi/NSSON_V02")
RESULTS = ROOT / "results" / "qos_multislice_final"
PLOTS = RESULTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

MODE_ORDER = [
    "baseline_sdn_iot",
    "priority_only",
    "slicing_sla",
]

MODE_LABELS = {
    "baseline_sdn_iot": "Baseline SDN-IoT",
    "priority_only": "NSSON Binary Priority",
    "slicing_sla": "NSSON Slice + SLA",
}

LOAD_ORDER = ["light", "moderate", "heavy"]

LOAD_LABELS = {
    "light": "Light (2 Mbps)",
    "moderate": "Moderate (12 Mbps)",
    "heavy": "Heavy (19 Mbps)",
}

SLICE_ORDER = ["autonomous", "healthcare", "industrial"]

SLICE_LABELS = {
    "autonomous": "Autonomous",
    "healthcare": "Healthcare",
    "industrial": "Industrial",
}

MODE_COLORS = {
    "baseline_sdn_iot": "#6C757D",
    "priority_only": "#E69F00",
    "slicing_sla": "#0072B2",
}

SLICE_COLORS = {
    "autonomous": "#D55E00",
    "healthcare": "#009E73",
    "industrial": "#CC79A7",
}

THROUGHPUT_SLA = {
    "autonomous": 8.0,
    "healthcare": 6.0,
    "industrial": 3.0,
}


def load_data():
    files = sorted(RESULTS.glob("*_VALID.csv"))

    if not files:
        raise FileNotFoundError(
            f"No valid CSV files found in {RESULTS}"
        )

    frames = [pd.read_csv(path) for path in files]
    df = pd.concat(frames, ignore_index=True)

    df["mode"] = pd.Categorical(
        df["mode"],
        categories=MODE_ORDER,
        ordered=True,
    )
    df["load"] = pd.Categorical(
        df["load"],
        categories=LOAD_ORDER,
        ordered=True,
    )
    df["slice"] = pd.Categorical(
        df["slice"],
        categories=SLICE_ORDER,
        ordered=True,
    )

    df["load_label"] = df["load"].map(LOAD_LABELS)
    df["mode_label"] = df["mode"].map(MODE_LABELS)
    df["slice_label"] = df["slice"].map(SLICE_LABELS)

    return df


def save(fig, name):
    png = PLOTS / f"{name}.png"
    pdf = PLOTS / f"{name}.pdf"
    fig.savefig(png, dpi=400, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def plot_throughput_by_slice(df):
    fig, axes = plt.subplots(
        1, 3, figsize=(18, 5.4), sharey=True
    )

    for ax, slice_name in zip(axes, SLICE_ORDER):
        subset = df[df["slice"] == slice_name].copy()

        sns.barplot(
            data=subset,
            x="load",
            y="tcp_throughput_mbps",
            hue="mode",
            order=LOAD_ORDER,
            hue_order=MODE_ORDER,
            palette=MODE_COLORS,
            ax=ax,
            edgecolor="black",
            linewidth=0.6,
        )

        threshold = THROUGHPUT_SLA[slice_name]
        ax.axhline(
            threshold,
            color=SLICE_COLORS[slice_name],
            linestyle="--",
            linewidth=2.0,
            label=f"SLA threshold = {threshold:g} Mbps",
        )

        ax.set_title(
            f"{SLICE_LABELS[slice_name]} slice",
            fontweight="bold",
        )
        ax.set_xlabel("Background load")
        ax.set_ylabel("TCP steady-state throughput (Mbps)")
        ax.set_xticklabels(
            [LOAD_LABELS[x] for x in LOAD_ORDER],
            rotation=20,
            ha="right",
        )
        ax.grid(axis="y", alpha=0.28)

        handles, labels = ax.get_legend_handles_labels()
        dedup = dict(zip(labels, handles))
        ax.legend(
            dedup.values(),
            dedup.keys(),
            fontsize=8,
            loc="upper right",
            frameon=True,
        )

    fig.suptitle(
        "Per-slice throughput under increasing background load",
        fontsize=15,
        fontweight="bold",
        y=1.03,
    )

    fig.tight_layout()
    save(fig, "fig1_per_slice_throughput")


def plot_heavy_load(df):
    heavy = df[df["load"] == "heavy"].copy()

    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    sns.barplot(
        data=heavy,
        x="slice",
        y="tcp_throughput_mbps",
        hue="mode",
        order=SLICE_ORDER,
        hue_order=MODE_ORDER,
        palette=MODE_COLORS,
        ax=ax,
        edgecolor="black",
        linewidth=0.7,
    )

    for idx, slice_name in enumerate(SLICE_ORDER):
        ax.scatter(
            idx,
            THROUGHPUT_SLA[slice_name],
            marker="_",
            s=700,
            color=SLICE_COLORS[slice_name],
            linewidths=3,
            zorder=5,
        )

    ax.set_title(
        "Heavy-load throughput: baseline, binary priority, and SLA slicing",
        fontweight="bold",
        fontsize=14,
    )
    ax.set_xlabel("Logical inference slice")
    ax.set_ylabel("TCP steady-state throughput (Mbps)")
    ax.set_xticklabels([SLICE_LABELS[s] for s in SLICE_ORDER])
    ax.grid(axis="y", alpha=0.28)

    legend = ax.legend(
        title="Controller policy",
        frameon=True,
    )

    fig.tight_layout()
    save(fig, "fig2_heavy_load_throughput")


def plot_sla_heatmap(df):
    sla = (
        df.groupby(["mode", "load"], observed=False)["sla_all_met"]
        .mean()
        .mul(100.0)
        .reset_index()
    )

    matrix = sla.pivot(
        index="mode",
        columns="load",
        values="sla_all_met",
    ).reindex(
        index=MODE_ORDER,
        columns=LOAD_ORDER,
    )

    fig, ax = plt.subplots(figsize=(9.2, 5.4))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".1f",
        cmap=sns.color_palette("YlGnBu", as_cmap=True),
        vmin=0,
        vmax=100,
        linewidths=1.2,
        linecolor="white",
        cbar_kws={"label": "Aggregate SLA fulfillment (%)"},
        ax=ax,
    )

    ax.set_title(
        "Aggregate SLA fulfillment across three inference slices",
        fontweight="bold",
        fontsize=14,
    )
    ax.set_xlabel("Background load")
    ax.set_ylabel("Controller policy")
    ax.set_xticklabels([LOAD_LABELS[x] for x in LOAD_ORDER])
    ax.set_yticklabels(
        [MODE_LABELS[x] for x in MODE_ORDER],
        rotation=0,
    )

    fig.tight_layout()
    save(fig, "fig3_sla_fulfillment_heatmap")


def plot_throughput_multiplier(df):
    baseline = (
        df[df["mode"] == "baseline_sdn_iot"]
        [["load", "slice", "tcp_throughput_mbps"]]
        .rename(
            columns={
                "tcp_throughput_mbps": "baseline_mbps"
            }
        )
    )

    comparison = df.merge(
        baseline,
        on=["load", "slice"],
        how="left",
    )

    comparison["multiplier"] = (
        comparison["tcp_throughput_mbps"]
        / comparison["baseline_mbps"]
    )

    comparison = comparison[
        comparison["mode"].isin(["priority_only", "slicing_sla"])
    ].copy()

    fig, axes = plt.subplots(
        1, 3, figsize=(18, 5.4), sharey=True
    )

    for ax, slice_name in zip(axes, SLICE_ORDER):
        subset = comparison[
            comparison["slice"] == slice_name
        ].copy()

        sns.barplot(
            data=subset,
            x="load",
            y="multiplier",
            hue="mode",
            order=LOAD_ORDER,
            hue_order=["priority_only", "slicing_sla"],
            palette={
                "priority_only": MODE_COLORS["priority_only"],
                "slicing_sla": MODE_COLORS["slicing_sla"],
            },
            ax=ax,
            edgecolor="black",
            linewidth=0.6,
        )

        ax.axhline(
            1.0,
            color="black",
            linestyle="--",
            linewidth=1.3,
        )

        ax.set_title(
            f"{SLICE_LABELS[slice_name]} slice",
            fontweight="bold",
        )
        ax.set_xlabel("Background load")
        ax.set_ylabel("Throughput multiplier vs baseline")
        ax.set_xticklabels(
            [LOAD_LABELS[x] for x in LOAD_ORDER],
            rotation=20,
            ha="right",
        )
        ax.grid(axis="y", alpha=0.28)

    fig.suptitle(
        "Throughput multiplier relative to baseline SDN-IoT",
        fontsize=15,
        fontweight="bold",
        y=1.03,
    )

    fig.tight_layout()
    save(fig, "fig4_throughput_multiplier")


def plot_policy_table(df):
    policy = (
        df[
            [
                "slice",
                "slice_priority",
                "slice_queue_id",
                "slice_min_rate_mbps",
                "slice_max_rate_mbps",
                "sla_jitter_ms_max",
                "sla_loss_pct_max",
                "sla_pdr_min",
                "sla_throughput_mbps_min",
            ]
        ]
        .drop_duplicates()
        .sort_values("slice")
    )

    policy["slice"] = policy["slice"].map(SLICE_LABELS)

    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.axis("off")

    table = ax.table(
        cellText=policy.values,
        colLabels=[
            "Slice",
            "Priority",
            "Queue ID",
            "Min-rate\n(Mbps)",
            "Configured target\n(Mbps)",
            "Jitter limit\n(ms)",
            "Loss limit\n(%)",
            "PDR minimum",
            "Throughput SLA\n(Mbps)",
        ],
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.0)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1F4E78")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#F4F7FA")

    ax.set_title(
        "NSSON logical-slice QoS policy and SLA configuration",
        fontweight="bold",
        fontsize=14,
        pad=18,
    )

    fig.tight_layout()
    save(fig, "fig5_slice_policy_table")


def plot_preworkload_qos(df):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    sns.lineplot(
        data=df,
        x="load",
        y="udp_jitter_ms",
        hue="mode",
        style="slice",
        hue_order=MODE_ORDER,
        style_order=SLICE_ORDER,
        palette=MODE_COLORS,
        markers=True,
        dashes=False,
        ax=axes[0],
    )

    axes[0].set_title(
        "Pre-workload UDP probe jitter",
        fontweight="bold",
    )
    axes[0].set_xlabel("Background-load configuration")
    axes[0].set_ylabel("UDP jitter (ms)")
    axes[0].set_xticklabels(
        [LOAD_LABELS[x] for x in LOAD_ORDER],
        rotation=20,
        ha="right",
    )
    axes[0].grid(alpha=0.28)

    sns.lineplot(
        data=df,
        x="load",
        y="udp_pdr",
        hue="mode",
        style="slice",
        hue_order=MODE_ORDER,
        style_order=SLICE_ORDER,
        palette=MODE_COLORS,
        markers=True,
        dashes=False,
        ax=axes[1],
    )

    axes[1].set_title(
        "Pre-workload UDP probe PDR",
        fontweight="bold",
    )
    axes[1].set_xlabel("Background-load configuration")
    axes[1].set_ylabel("Packet delivery ratio")
    axes[1].set_ylim(0.95, 1.005)
    axes[1].set_xticklabels(
        [LOAD_LABELS[x] for x in LOAD_ORDER],
        rotation=20,
        ha="right",
    )
    axes[1].grid(alpha=0.28)

    fig.suptitle(
        "Controlled pre-workload UDP probe metrics",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    fig.tight_layout()
    save(fig, "fig6_preworkload_udp_qos")


def create_latex_table(df):
    table = (
        df.pivot_table(
            index=["load", "slice"],
            columns="mode",
            values="tcp_throughput_mbps",
            aggfunc="mean",
        )
        .reindex(
            index=pd.MultiIndex.from_product(
                [LOAD_ORDER, SLICE_ORDER],
                names=["load", "slice"],
            )
        )
        .reindex(columns=MODE_ORDER)
        .reset_index()
    )

    table["load"] = table["load"].map(LOAD_LABELS)
    table["slice"] = table["slice"].map(SLICE_LABELS)

    table = table.rename(
        columns={
            "load": "Load",
            "slice": "Slice",
            "baseline_sdn_iot": "Baseline",
            "priority_only": "Priority",
            "slicing_sla": "SliceSLA",
        }
    )

    for col in ["Baseline", "Priority", "SliceSLA"]:
        table[col] = table[col].map(lambda x: f"{x:.3f}")

    latex = table.to_latex(
        index=False,
        escape=False,
        column_format="llccc",
    )

    out = PLOTS / "table_throughput_results.tex"
    out.write_text(latex, encoding="utf-8")
    print(f"Saved: {out}")


def main():
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.2,
    )

    df = load_data()

    plot_throughput_by_slice(df)
    plot_heavy_load(df)
    plot_sla_heatmap(df)
    plot_throughput_multiplier(df)
    plot_policy_table(df)
    plot_preworkload_qos(df)
    create_latex_table(df)

    print("\nAll plots generated from measured final CSV files.")
    print(f"Output directory: {PLOTS}")


if __name__ == "__main__":
    main()
