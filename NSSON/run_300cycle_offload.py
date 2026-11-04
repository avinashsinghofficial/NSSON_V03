from pathlib import Path
import json
import time
import numpy as np
import pandas as pd

ROOT = Path("/home/avi/NSSON")
OUT_ROOT = ROOT / "revision_runs_300cycle"
OUT_ROOT.mkdir(exist_ok=True)

CYCLES = 300
SEED = 42
CNN_ACC = 0.9484
SNN_ACC = 0.7766
BASE_THETA = 0.60

W_C = 5.0    # W, local inference power
W_I = 1.5    # W, sensing/idle power
W_T = 0.5    # J/MB, uplink transfer energy
DATA_MB = 0.1

TRANSFER_NO_SDN = 0.01071  # s, from measured 10.71 ms RTT
TRANSFER_SDN = 0.00736     # s, from measured 7.36 ms RTT

MODES = {
    "local_only": {"policy": "local", "sdn": False},
    "offload_only": {"policy": "offload", "sdn": False},
    "adaptive_no_sdn": {"policy": "adaptive", "sdn": False},
    "nsson_full": {"policy": "adaptive", "sdn": True},
}

def generate_confidences(rng):
    snn_correct = rng.random() < SNN_ACC
    conf_snn = 0.6 + 0.4 * rng.random() if snn_correct else 0.2 + 0.4 * rng.random()
    cnn_correct = rng.random() < CNN_ACC
    conf_cnn = 0.7 + 0.3 * rng.random() if cnn_correct else 0.3 + 0.4 * rng.random()
    return float(conf_snn), float(conf_cnn)

summary = []

for mode, cfg in MODES.items():
    mode_dir = OUT_ROOT / mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    decisions, latency_rows, power_rows = [], [], []

    for cycle in range(CYCLES):
        label = int(rng.integers(0, 10))
        conf_snn, conf_cnn = generate_confidences(rng)

        if cfg["policy"] == "local":
            decision = "local"
            theta = 0.0
        elif cfg["policy"] == "offload":
            decision = "offload"
            theta = 1.0
        else:
            theta = BASE_THETA
            decision = "offload" if conf_snn < theta else "local"

        sense_sec = 0.010
        idle_sec = 0.005

        if decision == "local":
            compute_sec = 0.020
            transfer_sec = 0.0
        else:
            compute_sec = 0.0
            transfer_sec = TRANSFER_SDN if cfg["sdn"] else TRANSFER_NO_SDN

        total_sec = sense_sec + compute_sec + transfer_sec + idle_sec

        energy_compute_J = W_C * compute_sec
        energy_sensing_idle_J = W_I * (sense_sec + idle_sec)
        energy_transfer_J = W_T * DATA_MB if decision == "offload" else 0.0
        energy_device_J = energy_compute_J + energy_sensing_idle_J + energy_transfer_J

        decisions.append({
            "sampleid": cycle,
            "label": label,
            "conf_snn": conf_snn,
            "conf_cnn": conf_cnn,
            "decision": decision,
            "theta": theta,
            "timestamp": time.time(),
            "cycle": cycle,
            "mode": mode,
        })

        latency_rows.append({
            "cycle": cycle,
            "sense_sec": sense_sec,
            "compute_sec": compute_sec,
            "transfer_sec": transfer_sec,
            "idle_sec": idle_sec,
            "total_sec": total_sec,
            "mode": mode,
        })

        power_rows.append({
            "cycle": cycle,
            "data_mb": DATA_MB,
            "compute_sec": compute_sec,
            "idle_sec": idle_sec,
            "energy_compute_J": energy_compute_J,
            "energy_sensing_idle_J": energy_sensing_idle_J,
            "energy_transfer_J": energy_transfer_J,
            "energy_device_J": energy_device_J,
            "energy_device_Wh": energy_device_J / 3600.0,
            "average_device_power_W": energy_device_J / total_sec,
            "decision": decision,
            "mode": mode,
        })

    pd.DataFrame(decisions).to_json(
        mode_dir / "decision_log.json", orient="records", lines=True
    )
    pd.DataFrame(latency_rows).to_csv(mode_dir / "latency_log.csv", index=False)
    pd.DataFrame(power_rows).to_csv(mode_dir / "power_log.csv", index=False)

    pdf = pd.DataFrame(power_rows)
    ldf = pd.DataFrame(latency_rows)
    offload_ratio = float((pdf["decision"] == "offload").mean())

    summary.append({
        "mode": mode,
        "cycles": CYCLES,
        "offload_ratio": offload_ratio,
        "mean_latency_ms": 1000 * ldf["total_sec"].mean(),
        "mean_energy_device_J": pdf["energy_device_J"].mean(),
        "mean_energy_device_Wh": pdf["energy_device_Wh"].mean(),
        "total_energy_device_J": pdf["energy_device_J"].sum(),
        "total_energy_device_Wh": pdf["energy_device_Wh"].sum(),
        "mean_average_device_power_W": pdf["average_device_power_W"].mean(),
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv(OUT_ROOT / "summary_300cycle.csv", index=False)

local_energy = summary_df.loc[
    summary_df["mode"] == "local_only", "mean_energy_device_J"
].iloc[0]
local_latency = summary_df.loc[
    summary_df["mode"] == "local_only", "mean_latency_ms"
].iloc[0]

for mode in summary_df["mode"]:
    row = summary_df[summary_df["mode"] == mode].iloc[0]
    energy_saving = 100 * (1 - row["mean_energy_device_J"] / local_energy)
    latency_change = 100 * (row["mean_latency_ms"] / local_latency - 1)
    print(
        f"{mode:16s} | cycles={row['cycles']} | "
        f"offload={row['offload_ratio']:.4f} | "
        f"latency={row['mean_latency_ms']:.4f} ms ({latency_change:+.2f}%) | "
        f"energy={row['mean_energy_device_J']:.6f} J/cycle | "
        f"Wh={row['mean_energy_device_Wh']:.10f} | "
        f"power={row['mean_average_device_power_W']:.4f} W | "
        f"energy_saving={energy_saving:.2f}%"
    )

print("\nSummary CSV:")
print(OUT_ROOT / "summary_300cycle.csv")
