from pathlib import Path
import pandas as pd
import json

BASE = Path("/home/avi/NSSON/offload")
OUT_DIR = Path("/home/avi/NSSON/revision_audit")
OUT_DIR.mkdir(exist_ok=True)

MODES = ["local_only", "offload_only", "adaptive_no_sdn", "nsson_full"]

# Correct coefficient-based model
W_C = 5.0      # W, compute power coefficient
W_I = 1.5      # W, idle power coefficient
W_T = 0.5      # J/MB, transfer energy coefficient

summary = []

for mode in MODES:
    power_file = BASE / mode / "power_log.csv"
    latency_file = BASE / mode / "latency_log.csv"
    decision_file = BASE / mode / "decision_log.json"

    if not power_file.exists():
        raise SystemExit(f"Missing file: {power_file}")
    if not latency_file.exists():
        raise SystemExit(f"Missing file: {latency_file}")

    p = pd.read_csv(power_file)
    l = pd.read_csv(latency_file)

    df = p.merge(
        l[["cycle", "total_sec"]],
        on="cycle",
        how="inner",
        validate="one_to_one"
    )

    required = [
        "cycle", "data_mb", "compute_sec", "idle_sec",
        "total_sec", "mode"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"After merge, {power_file} missing columns: {missing}")

    # Correct dimensional model
    df["energy_compute_J"] = W_C * df["compute_sec"]
    df["energy_idle_J"] = W_I * df["idle_sec"]
    df["energy_transfer_J"] = W_T * df["data_mb"]
    df["energy_cycle_J"] = (
        df["energy_compute_J"]
        + df["energy_idle_J"]
        + df["energy_transfer_J"]
    )
    df["energy_cycle_Wh"] = df["energy_cycle_J"] / 3600.0
    df["average_power_W"] = df["energy_cycle_J"] / df["total_sec"]

    # Preserve original logged values for audit comparison
    df["original_logged_energy_J"] = df["energy_j"]
    df["original_logged_energy_Wh"] = df["total_power_wh"]

    out_csv = OUT_DIR / f"energy_recomputed_{mode}.csv"
    df.to_csv(out_csv, index=False)

    decisions = []
    if decision_file.exists():
        with open(decision_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    decisions.append(json.loads(line))

    offload_count = sum(1 for x in decisions if x.get("decision") == "offload")
    local_count = sum(1 for x in decisions if x.get("decision") == "local")

    threshold_file = decision_file.parent / "threshold_log.csv"
    mean_theta = None
    if threshold_file.exists():
        mean_theta = pd.read_csv(threshold_file)["theta"].mean()

    summary.append({
        "mode": mode,
        "cycles": len(df),
        "offload_count": offload_count,
        "local_count": local_count,
        "offload_ratio": offload_count / len(df) if len(df) else None,
        "mean_latency_s": df["total_sec"].mean(),
        "mean_energy_cycle_J": df["energy_cycle_J"].mean(),
        "mean_energy_cycle_Wh": df["energy_cycle_Wh"].mean(),
        "total_energy_J": df["energy_cycle_J"].sum(),
        "total_energy_Wh": df["energy_cycle_Wh"].sum(),
        "mean_average_power_W": df["average_power_W"].mean(),
        "mean_theta": mean_theta,
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv(OUT_DIR / "energy_summary_correct.csv", index=False)

local_energy = summary_df.loc[
    summary_df["mode"] == "local_only", "mean_energy_cycle_J"
].iloc[0]

for mode in MODES:
    row = summary_df[summary_df["mode"] == mode].iloc[0]
    saving = 100 * (1 - row["mean_energy_cycle_J"] / local_energy)
    print(
        f"{mode:16s} | cycles={row['cycles']:3d} | "
        f"energy={row['mean_energy_cycle_J']:.6f} J | "
        f"Wh={row['mean_energy_cycle_Wh']:.10f} | "
        f"avg_power={row['mean_average_power_W']:.4f} W | "
        f"offload={row['offload_ratio']:.2f} | "
        f"saving_vs_local={saving:.2f}%"
    )

print("\nSaved per-cycle results:")
for mode in MODES:
    print(OUT_DIR / f"energy_recomputed_{mode}.csv")

print("\nSaved summary:")
print(OUT_DIR / "energy_summary_correct.csv")
