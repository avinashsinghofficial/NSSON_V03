#!/usr/bin/env python3
"""Aggregate QoS experiment CSVs and compute per-condition statistics."""

from pathlib import Path
import pandas as pd
import numpy as np
import json

ROOT = Path("/home/avi/NSSON_V02")
RAW_DIR = ROOT / "logs" / "qos_raw"
OUT_DIR = ROOT / "logs" / "qos_aggregated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_all_qos_csvs():
    rows = []
    for path in RAW_DIR.glob("qos_*.csv"):
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            rows.append(row.to_dict())
    if not rows:
        raise SystemExit("No qos_*.csv files found in logs/qos_raw")
    return pd.DataFrame(rows)

def add_controller_label(df):
    def label(mode):
        if mode.startswith("nsson"):
            return "nsson"
        if mode.startswith("baseline"):
            return "baseline"
        return "unknown"
    df = df.copy()
    df["controller"] = df["mode"].apply(label)
    return df

def compute_stats(df):
    metrics = [
        "rtt_avg_ms",
        "rtt_jitter_mdev_ms",
        "ping_loss_pct",
        "control_tcp_throughput_mbps",
        "background_udp_throughput_mbps",
        "background_udp_jitter_ms",
        "background_udp_loss_pct",
    ]
    out = []
    for (controller, load), g in df.groupby(["controller", "load"]):
        n = len(g)
        row = {
            "controller": controller,
            "load": load,
            "n_reps": n,
        }
        for m in metrics:
            vals = g[m].dropna().to_numpy()
            if len(vals) == 0:
                row[f"{m}_mean"] = np.nan
                row[f"{m}_std"] = np.nan
                row[f"{m}_ci95"] = np.nan
                continue
            mean = float(vals.mean())
            std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            # 95% CI for mean: mean ± t * s/√n
            if n > 1:
                t = 2.776  # approx for n=4; will refine below
                if n >= 5:
                    t = 2.571
                if n >= 10:
                    t = 2.262
                ci = t * std / np.sqrt(n)
            else:
                ci = 0.0
            row[f"{m}_mean"] = mean
            row[f"{m}_std"] = std
            row[f"{m}_ci95"] = ci
        out.append(row)
    return pd.DataFrame(out)

def main():
    df = load_all_qos_csvs()
    df = add_controller_label(df)

    # Save merged long-form CSV
    merged_path = OUT_DIR / "qos_all_runs.csv"
    df.to_csv(merged_path, index=False)

    stats = compute_stats(df)
    stats_path = OUT_DIR / "qos_stats_by_condition.csv"
    stats.to_csv(stats_path, index=False)

    # Also save JSON for easy plotting
    json_path = OUT_DIR / "qos_stats_by_condition.json"
    with open(json_path, "w") as f:
        json.dump(stats.to_dict(orient="records"), f, indent=2)

    print("Aggregated CSV:", merged_path)
    print("Stats CSV:", stats_path)
    print("Stats JSON:", json_path)
    print("\nPer-condition summary (means):")
    display_cols = [
        "controller", "load", "n_reps",
        "rtt_avg_ms_mean", "control_tcp_throughput_mbps_mean",
        "ping_loss_pct_mean", "background_udp_loss_pct_mean",
    ]
    print(stats[display_cols].to_string(index=False))

if __name__ == "__main__":
    main()
