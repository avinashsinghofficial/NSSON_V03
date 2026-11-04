from pathlib import Path
import json
import pandas as pd

BASE = Path("/home/avi/NSSON/offload")
OUT_DIR = Path("/home/avi/NSSON/revision_audit")
OUT_DIR.mkdir(exist_ok=True)

MODES = ["local_only", "offload_only", "adaptive_no_sdn", "nsson_full"]

W_C = 5.0   # W, local inference power
W_I = 1.5   # W, device sensing/idle power
W_T = 0.5   # J/MB, uplink transfer energy

summary = []

for mode in MODES:
    latency_file = BASE / mode / "latency_log.csv"
    decision_file = BASE / mode / "decision_log.json"

    if not latency_file.exists():
        raise SystemExit(f"Missing: {latency_file}")
    if not decision_file.exists():
        raise SystemExit(f"Missing: {decision_file}")

    lat = pd.read_csv(latency_file)

    decisions = []
    with open(decision_file) as f:
        for line in f:
            line = line.strip()
            if line:
                decisions.append(json.loads(line))

    dec = pd.DataFrame(decisions)

    required_lat = {"cycle", "sense_sec", "compute_sec", "transfer_sec", "idle_sec", "total_sec"}
    required_dec = {"cycle", "decision"}
    if not required_lat.issubset(lat.columns):
        raise SystemExit(f"{latency_file} missing: {required_lat - set(lat.columns)}")
    if not required_dec.issubset(dec.columns):
        raise SystemExit(f"{decision_file} missing: {required_dec - set(dec.columns)}")

    df = lat.merge(dec[["cycle", "decision"]], on="cycle", how="inner", validate="one_to_one")

    is_local = df["decision"].eq("local")
    is_offload = df["decision"].eq("offload")

    df["energy_local_compute_J"] = W_C * df["compute_sec"] * is_local.astype(int)
    df["energy_sensing_idle_J"] = W_I * (df["sense_sec"] + df["idle_sec"])
    df["energy_transfer_J"] = W_T * 0.1 * is_offload.astype(int)

    df["energy_device_J"] = (
        df["energy_local_compute_J"]
        + df["energy_sensing_idle_J"]
        + df["energy_transfer_J"]
    )
    df["energy_device_Wh"] = df["energy_device_J"] / 3600.0
    df["average_device_power_W"] = df["energy_device_J"] / df["total_sec"]

    out_csv = OUT_DIR / f"energy_decision_aware_{mode}.csv"
    df.to_csv(out_csv, index=False)

    offload_count = int(is_offload.sum())
    local_count = int(is_local.sum())

    summary.append({
        "mode": mode,
        "cycles": len(df),
        "local_cycles": local_count,
        "offload_cycles": offload_count,
        "offload_ratio": offload_count / len(df),
        "mean_latency_ms": 1000 * df["total_sec"].mean(),
        "mean_energy_device_J": df["energy_device_J"].mean(),
        "mean_energy_device_Wh": df["energy_device_Wh"].mean(),
        "total_energy_device_J": df["energy_device_J"].sum(),
        "total_energy_device_Wh": df["energy_device_Wh"].sum(),
        "mean_average_device_power_W": df["average_device_power_W"].mean(),
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv(OUT_DIR / "energy_decision_aware_summary.csv", index=False)

local_energy = summary_df.loc[
    summary_df["mode"] == "local_only", "mean_energy_device_J"
].iloc[0]

for mode in MODES:
    row = summary_df[summary_df["mode"] == mode].iloc[0]
    saving = 100 * (1 - row["mean_energy_device_J"] / local_energy)
    print(
        f"{mode:16s} | cycles={row['cycles']:3d} | "
        f"offload={row['offload_ratio']:.2f} | "
        f"energy={row['mean_energy_device_J']:.6f} J/cycle | "
        f"Wh={row['mean_energy_device_Wh']:.10f} | "
        f"power={row['mean_average_device_power_W']:.4f} W | "
        f"saving={saving:.2f}%"
    )

print("\nSaved:")
print(OUT_DIR / "energy_decision_aware_summary.csv")
for mode in MODES:
    print(OUT_DIR / f"energy_decision_aware_{mode}.csv")
