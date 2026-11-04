import os
import pandas as pd

NSSON_ROOT = "/home/avi/NSSON_V02"
LATENCY_LOG = os.path.join(NSSON_ROOT, "logs", "offloadlatencylog.csv")
POWER_LOG = os.path.join(NSSON_ROOT, "logs", "offloadpowerlog.csv")

Wc = 5.0
Wt = 0.5
Wi = 1.5

def compute_power(mb_per_cycle=0.1, mode="adaptive_no_sdn"):
    lat_df = pd.read_csv(LATENCY_LOG)
    lat_df = lat_df[lat_df["mode"] == mode].copy()
    rows = []
    for _, row in lat_df.iterrows():
        cycle = row["cycle"]
        Lcompute = row["compute_sec"]
        Ltransfer = row["transfer_sec"]
        Lidle = row["idle_sec"]
        Le2e = row["total_sec"]

        Pcompute = Wc * (Lcompute / Le2e) if Le2e > 0 else 0.0
        Ptransfer = Wt * mb_per_cycle
        Pidle = Wi * (Lidle / Le2e) if Le2e > 0 else 0.0
        Ptotal = Pcompute + Ptransfer + Pidle
        E = Ptotal * Le2e

        rows.append({
            "cycle": cycle,
            "data_mb": mb_per_cycle,
            "compute_sec": Lcompute,
            "idle_sec": Lidle,
            "compute_power_w": Pcompute,
            "transfer_power_w": Ptransfer,
            "idle_power_w": Pidle,
            "total_power_wh": Ptotal * (Le2e / 3600.0),
            "energy_j": E,
            "mode": mode,
        })

    pd.DataFrame(rows).to_csv(POWER_LOG, index=False)
    print("Power log written to", POWER_LOG)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True,
                   choices=["local_only", "offload_only", "adaptive_no_sdn", "nsson_full"])
    args = p.parse_args()
    compute_power(mb_per_cycle=0.1, mode=args.mode)
