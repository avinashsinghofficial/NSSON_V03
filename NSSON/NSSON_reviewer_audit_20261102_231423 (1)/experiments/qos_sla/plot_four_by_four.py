#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


MODE_ORDER = ["baseline", "nsson_priority", "nsson_slice_sla", "nsson_isar"]
MODE_LABELS = {
    "baseline": "B0: Baseline",
    "nsson_priority": "B1: Priority",
    "nsson_slice_sla": "B2: Slice + SLA",
    "nsson_isar": "B3: ISAR",
}
SLICE_ORDER = ["control", "healthcare", "monitoring"]
COLORS = {
    "control": "#d62728",
    "healthcare": "#2ca02c",
    "monitoring": "#1f77b4",
}


def load_mode(run_dir: Path, mode: str):
    p = run_dir / mode / f"qos_slice_metrics_{mode}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    df = pd.read_csv(p)
    df["mode"] = mode
    return df


def ci95(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0.0
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def plot_grouped(ax, summary, ycol, ylabel, title, higher_is_better=False):
    x = np.arange(len(SLICE_ORDER))
    width = 0.22
    offsets = np.linspace(-width, width, len(MODE_ORDER))

    for j, mode in enumerate(MODE_ORDER):
        m = summary[summary["mode"] == mode].set_index("slice").reindex(SLICE_ORDER)
        values = m[ycol].fillna(0.0).values
        errors = m[f"{ycol}_ci"].fillna(0.0).values
        bars = ax.bar(
            x + offsets[j], values, width,
            yerr=errors, capsize=3,
            label=MODE_LABELS[mode] if ax is None else None,
            alpha=0.90
        )
        for bar, slice_name in zip(bars, SLICE_ORDER):
            bar.set_edgecolor(COLORS[slice_name])
            bar.set_linewidth(1.8)

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in SLICE_ORDER], rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_df = pd.concat([load_mode(run_dir, m) for m in MODE_ORDER], ignore_index=True)
    all_df = all_df[all_df["slice"].isin(SLICE_ORDER)].copy()
    all_df["sla_violation_pct"] = 100.0 * (1.0 - all_df["sla_all_ok"].astype(float))

    metrics = ["rtt_ms", "loss_pct", "throughput_mbps", "sla_violation_pct"]
    grouped = all_df.groupby(["mode", "slice"])

    rows = []
    for (mode, slice_name), g in grouped:
        r = {"mode": mode, "slice": slice_name}
        for metric in metrics:
            r[metric] = g[metric].mean()
            r[f"{metric}_ci"] = ci95(g[metric].values)
        rows.append(r)
    summary = pd.DataFrame(rows)

    fig, axes = plt.subplots(4, 4, figsize=(18, 16), sharex="col")
    fig.suptitle(
        "NSSON Ablation: QoS, SLA Slicing, and Inference-Aware Adaptive Routing",
        fontsize=18, fontweight="bold", y=0.995
    )

    row_specs = [
        ("rtt_ms", "RTT (ms)", "Inference-path RTT"),
        ("loss_pct", "Packet loss (%)", "Packet loss"),
        ("throughput_mbps", "Goodput (Mbps)", "Inference goodput"),
        ("sla_violation_pct", "SLA violations (%)", "SLA violation rate"),
    ]

    # Each column emphasizes its corresponding system mode.
    # All modes are still visible in every plot to preserve the ablation comparison.
    for col, focus_mode in enumerate(MODE_ORDER):
        for row, (metric, ylabel, title) in enumerate(row_specs):
            ax = axes[row, col]
            x = np.arange(len(SLICE_ORDER))
            width = 0.18
            offsets = np.linspace(-0.27, 0.27, len(MODE_ORDER))

            for j, mode in enumerate(MODE_ORDER):
                m = summary[summary["mode"] == mode].set_index("slice").reindex(SLICE_ORDER)
                vals = m[metric].fillna(0.0).values
                errs = m[f"{metric}_ci"].fillna(0.0).values
                alpha = 1.0 if mode == focus_mode else 0.24
                bars = ax.bar(
                    x + offsets[j], vals, width,
                    yerr=errs, capsize=2,
                    color=[COLORS[s] for s in SLICE_ORDER],
                    alpha=alpha,
                    edgecolor="black",
                    linewidth=0.55,
                )
                if row == 0:
                    for bar, mode_label in zip(bars, [mode] * len(bars)):
                        bar.set_label(mode_label)

            ax.set_title(f"{MODE_LABELS[focus_mode]}\n{title}", fontsize=10, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels([s.capitalize() for s in SLICE_ORDER], fontsize=8)
            if col == 0:
                ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.22)

    legend_handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    dedup = {}
    for h, l in zip(legend_handles, legend_labels):
        dedup[l] = h
    fig.legend(
        dedup.values(),
        [MODE_LABELS[k] for k in dedup.keys()],
        loc="lower center",
        ncol=4,
        frameon=True,
        bbox_to_anchor=(0.5, 0.005),
    )

    fig.tight_layout(rect=[0, 0.06, 1, 0.965])
    out = out_dir / "nsson_qos_sla_adaptive_4x4.png"
    fig.savefig(out, dpi=350, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {out}")
    print(f"Saved: {out.with_suffix('.pdf')}")

    summary.to_csv(out_dir / "nsson_4x4_summary.csv", index=False)

    reroute_rows = []
    for mode in MODE_ORDER:
        p = run_dir / mode / f"reroute_events_{mode}.csv"
        if p.exists() and p.stat().st_size > 100:
            df = pd.read_csv(p)
            if len(df):
                df["mode"] = mode
                reroute_rows.append(df)

    if reroute_rows:
        events = pd.concat(reroute_rows, ignore_index=True)
        fig2, ax = plt.subplots(figsize=(10, 5))
        order = [m for m in MODE_ORDER if m in events["mode"].unique()]
        sns.countplot(data=events, x="slice", hue="mode", order=SLICE_ORDER, hue_order=order, ax=ax)
        ax.set_title("SLA-triggered rerouting events by slice", fontweight="bold")
        ax.set_xlabel("Logical inference slice")
        ax.set_ylabel("Reroute events")
        ax.grid(axis="y", alpha=0.25)
        fig2.tight_layout()
        fig2.savefig(out_dir / "nsson_reroute_event_counts.png", dpi=350)
        fig2.savefig(out_dir / "nsson_reroute_event_counts.pdf")
        print("Saved reroute-event plots")


if __name__ == "__main__":
    main()
