from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_records(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
    if not records:
        raise ValueError("No records found in the decision log.")
    return pd.DataFrame(records)


def pick_column(df: pd.DataFrame, candidates: list[str], meaning: str) -> str:
    lookup = {str(col).strip().lower(): col for col in df.columns}
    for name in candidates:
        if name.lower() in lookup:
            return lookup[name.lower()]
    raise ValueError(
        f"Cannot find a {meaning} column. Available columns: {list(df.columns)}"
    )


def normalize_decision(value) -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
        return int(float(value) > 0)
    token = str(value).strip().lower()
    if token in {"offload", "remote", "edge", "1", "true", "yes"}:
        return 1
    if token in {"local", "0", "false", "no"}:
        return 0
    raise ValueError(f"Unrecognized decision value: {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Real NSSON decision log (.jsonl/.json/.csv)")
    parser.add_argument("--output", required=True, help="Output PDF or PNG path")
    parser.add_argument("--bins", type=int, default=8, help="Number of observed-threshold bins (default: 8)")
    parser.add_argument("--title", default="Measured Relationship Between Threshold and Offload Ratio")
    args = parser.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = read_records(in_path)
    theta_col = pick_column(df, ["threshold", "theta", "theta_t", "adaptive_threshold"], "threshold")
    decision_col = pick_column(df, ["decision", "offload_decision", "placement", "action"], "decision")

    work = pd.DataFrame({
        "theta": pd.to_numeric(df[theta_col], errors="coerce"),
        "offload": df[decision_col].map(normalize_decision),
    }).dropna()

    if len(work) < 10:
        raise ValueError("At least 10 valid decisions are required for a meaningful plot.")
    if work["theta"].nunique() < 2:
        raise ValueError(
            "The log contains only one threshold value. Run a threshold sweep or log an adaptive run "
            "with changing theta(t) before producing this figure."
        )

    # Use quantile-like equally spaced bins over observed threshold range. Each point
    # reports the empirical offload fraction from actual decisions in that threshold band.
    n_bins = min(args.bins, work["theta"].nunique())
    edges = np.linspace(work["theta"].min(), work["theta"].max(), n_bins + 1)
    edges[0] -= 1e-12
    work["theta_bin"] = pd.cut(work["theta"], bins=edges, include_lowest=True)

    grouped = work.groupby("theta_bin", observed=True).agg(
        theta_mean=("theta", "mean"),
        offload_ratio=("offload", "mean"),
        samples=("offload", "size"),
    ).reset_index().sort_values("theta_mean")

    # Wilson 95% confidence intervals for a binomial offload proportion.
    z = 1.96
    p = grouped["offload_ratio"].to_numpy()
    n = grouped["samples"].to_numpy(dtype=float)
    center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2))) / (1 + z**2 / n)
    lower = np.clip(center - half, 0, 1)
    upper = np.clip(center + half, 0, 1)

    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9})
    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    ax.plot(grouped["theta_mean"], grouped["offload_ratio"], marker="o", linewidth=1.6,
            color="#1f77b4", label="Measured offload ratio")
    ax.fill_between(grouped["theta_mean"], lower, upper, color="#1f77b4", alpha=0.18,
                    label="95% Wilson interval")
    for x, y, count in zip(grouped["theta_mean"], grouped["offload_ratio"], grouped["samples"]):
        ax.annotate(f"n={count}", (x, y), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=7)

    ax.set_xlabel(r"Adaptive confidence threshold $\theta(t)$")
    ax.set_ylabel("Empirical offload ratio")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(args.title)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.8)
    ax.legend(loc="best", fontsize=7, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    print("Saved:", out_path)
    print(grouped[["theta_mean", "offload_ratio", "samples"]].to_string(index=False))


if __name__ == "__main__":
    main()
