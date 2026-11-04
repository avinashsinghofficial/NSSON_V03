import os
import pandas as pd
import numpy as np

NSSON_ROOT = "/home/avi/NSSON_V02"
LATENCY_LOG = os.path.join(NSSON_ROOT, "logs", "offloadlatencylog.csv")
POWER_LOG = os.path.join(NSSON_ROOT, "logs", "offloadpowerlog.csv")
DECISION_LOG = os.path.join(NSSON_ROOT, "logs", "offloaddecisionlog.json")
THRESHOLD_LOG = os.path.join(NSSON_ROOT, "logs", "thresholdlog.csv")

BASE_THETA = 0.6
ALPHA_L = 0.1
ALPHA_P = 0.05
ALPHA_A = 0.05
WINDOW_N = 10

def load_latency_power(mode):
    lat_df = pd.read_csv(LATENCY_LOG)
    pow_df = pd.read_csv(POWER_LOG)
    lat_df = lat_df[lat_df["mode"] == mode]
    pow_df = pow_df[pow_df["mode"] == mode]
    df = pd.merge(lat_df, pow_df, on="cycle", how="inner")
    return df

def compute_accuracy_window(decisions_df, cycles_window):
    sub = decisions_df[decisions_df["cycle"].isin(cycles_window)]
    if sub.empty:
        return 0.0
    return np.mean(sub["conf_cnn"] > sub["conf_snn"])

def update_threshold(mode="adaptive_no_sdn"):
    df = load_latency_power(mode)
    decisions = pd.read_json(DECISION_LOG, lines=True)
    decisions = decisions[decisions["mode"] == mode]

    cycles = df["cycle"].values
    thresholds = []
    theta = BASE_THETA

    for i, cycle in enumerate(cycles):
        start = max(0, i - WINDOW_N + 1)
        window_cycles = cycles[start:i+1]

        L_bar = df[df["cycle"].isin(window_cycles)]["total_sec"].mean()
        P_bar = df[df["cycle"].isin(window_cycles)]["total_power_wh"].mean()
        A_bar = compute_accuracy_window(decisions, window_cycles)

        theta = BASE_THETA - ALPHA_L * L_bar - ALPHA_P * P_bar - ALPHA_A * A_bar
        theta = max(0.30, min(0.90, theta))
        thresholds.append({"cycle": int(cycle), "theta": float(theta),
                           "L_bar": float(L_bar), "P_bar": float(P_bar), "A_bar": float(A_bar),
                           "mode": mode})

    pd.DataFrame(thresholds).to_csv(THRESHOLD_LOG, index=False)
    print("Threshold log written to", THRESHOLD_LOG)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True,
                   choices=["local_only", "offload_only", "adaptive_no_sdn", "nsson_full"])
    args = p.parse_args()
    update_threshold(mode=args.mode)
