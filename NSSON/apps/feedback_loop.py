# /home/avi/NSSON/apps/feedback_loop.py
#
# Accuracy-aware NSSON feedback using SpikingJelly SNN accuracy

import numpy as np
import pandas as pd
import os

NSSON_ROOT = "/home/avi/NSSON"

LATENCY_LOG = os.path.join(NSSON_ROOT, "logs", "latencylog.csv")
POWER_LOG = os.path.join(NSSON_ROOT, "logs", "powerlog.csv")
ACCURACY_LOG = os.path.join(NSSON_ROOT, "logs", "snn_accuracylog.csv")  # NEW: per-cycle SNN accuracy/confidence

BASE_THRESHOLD = 0.6
ALPHA_L = 0.1
ALPHA_P = 0.05
ALPHA_A = 0.05   # tune this based on your experiments
WINDOW_N = 10

def load_latency_power():
    lat_df = pd.read_csv(LATENCY_LOG)
    pow_df = pd.read_csv(POWER_LOG)
    lat_df["cycle"] = np.arange(1, len(lat_df) + 1)
    pow_df["cycle"] = np.arange(1, len(pow_df) + 1)
    df = pd.merge(lat_df, pow_df, on="cycle", how="inner")
    df.rename(columns={"totalsec": "latency_sec", "totalpowerwh": "power_wh"}, inplace=True)
    return df

def load_accuracy():
    if not os.path.exists(ACCURACY_LOG):
        # fallback: use constant or confidence-based proxy
        return None
    acc_df = pd.read_csv(ACCURACY_LOG)  # columns: cycle, snn_acc
    return acc_df

def compute_threshold_series():
    lat_pow_df = load_latency_power()
    acc_df = load_accuracy()

    thresholds = []
    cycles = lat_pow_df["cycle"].values

    for i, cycle in enumerate(cycles):
        # sliding window [max(0, i-WINDOW_N+1) : i]
        start_idx = max(0, i - WINDOW_N + 1)
        window_lat = lat_pow_df["latency_sec"].values[start_idx:i+1]
        window_pow = lat_pow_df["power_wh"].values[start_idx:i+1]

        L_bar = np.mean(window_lat)
        P_bar = np.mean(window_pow)

        if acc_df is not None:
            window_acc = acc_df[acc_df["cycle"].between(cycles[start_idx], cycle)]["snn_acc"]
            A_bar = window_acc.mean() if len(window_acc) > 0 else 0.0
        else:
            A_bar = 0.0  # or some default

        theta = BASE_THRESHOLD - ALPHA_L * L_bar - ALPHA_P * P_bar - ALPHA_A * A_bar
        theta = max(0.30, min(0.90, theta))
        thresholds.append(theta)

    out_df = pd.DataFrame({"cycle": cycles, "threshold": thresholds})
    out_path = os.path.join(NSSON_ROOT, "logs", "threshold_series_acc_power_latency.csv")
    out_df.to_csv(out_path, index=False)
    print("Written accuracy-aware threshold series to", out_path)

if __name__ == "__main__":
    compute_threshold_series()
