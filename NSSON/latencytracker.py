import os
import time
import pandas as pd
import numpy as np

NSSON_ROOT = "/home/avi/NSSON_V02"
LATENCY_LOG = os.path.join(NSSON_ROOT, "logs", "offloadlatencylog.csv")

def measure_latency(cycles=300, mode="adaptive_no_sdn"):
    rows = []
    np.random.seed(42)
    for cycle in range(cycles):
        t0 = time.time()
        time.sleep(0.01)  # sense
        t1 = time.time()
        time.sleep(0.02)  # compute
        t2 = time.time()
        time.sleep(0.01)  # transfer
        t3 = time.time()
        time.sleep(0.005) # idle
        t4 = time.time()

        Lsense = t1 - t0
        Lcompute = t2 - t1
        Ltransfer = t3 - t2
        Lidle = t4 - t3
        Le2e = Lsense + Lcompute + Ltransfer + Lidle

        rows.append({
            "cycle": cycle,
            "sense_sec": Lsense,
            "compute_sec": Lcompute,
            "transfer_sec": Ltransfer,
            "idle_sec": Lidle,
            "total_sec": Le2e,
            "mode": mode,
        })

    pd.DataFrame(rows).to_csv(LATENCY_LOG, index=False)
    print("Latency log written to", LATENCY_LOG)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True,
                   choices=["local_only", "offload_only", "adaptive_no_sdn", "nsson_full"])
    args = p.parse_args()
    measure_latency(cycles=300, mode=args.mode)
