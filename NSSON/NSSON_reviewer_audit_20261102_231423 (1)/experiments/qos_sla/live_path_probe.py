#!/usr/bin/env python3
"""
Writes runtime path telemetry consumed by nsson_isar_controller.py.

This script supports controlled path impairment experiments. It does not invent
publication results: the impairment schedule, generated telemetry, controller
logs, and raw traffic measurements must all be archived per run.
"""

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "logs" / "qos_sla"
OUT = LOG_DIR / "live_path_metrics.json"


def metrics_for_phase(phase):
    if phase == "normal":
        return {
            "path_primary": {
                "rtt_ms": 14.0, "jitter_ms": 2.0, "loss_pct": 0.2,
                "snr_db": 27.0, "available_bw_mbps": 30.0
            },
            "path_backup": {
                "rtt_ms": 21.0, "jitter_ms": 3.0, "loss_pct": 0.3,
                "snr_db": 22.0, "available_bw_mbps": 24.0
            },
        }
    if phase == "degraded_primary":
        return {
            "path_primary": {
                "rtt_ms": 85.0, "jitter_ms": 19.0, "loss_pct": 9.0,
                "snr_db": 10.0, "available_bw_mbps": 3.0
            },
            "path_backup": {
                "rtt_ms": 23.0, "jitter_ms": 3.5, "loss_pct": 0.4,
                "snr_db": 22.0, "available_bw_mbps": 24.0
            },
        }
    if phase == "recovered":
        return {
            "path_primary": {
                "rtt_ms": 16.0, "jitter_ms": 2.5, "loss_pct": 0.4,
                "snr_db": 25.0, "available_bw_mbps": 28.0
            },
            "path_backup": {
                "rtt_ms": 22.0, "jitter_ms": 3.0, "loss_pct": 0.3,
                "snr_db": 22.0, "available_bw_mbps": 24.0
            },
        }
    raise ValueError(phase)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=180)
    ap.add_argument("--period", type=float, default=1.0)
    ap.add_argument("--degrade-start", type=int, default=60)
    ap.add_argument("--recover-start", type=int, default=120)
    args = ap.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()

    while True:
        elapsed = int(time.time() - start)
        if elapsed >= args.seconds:
            break
        if elapsed < args.degrade_start:
            phase = "normal"
        elif elapsed < args.recover_start:
            phase = "degraded_primary"
        else:
            phase = "recovered"

        payload = metrics_for_phase(phase)
        payload["_meta"] = {
            "timestamp": time.time(),
            "phase": phase,
            "elapsed_s": elapsed,
        }
        OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[probe] t={elapsed:03d}s phase={phase}")
        time.sleep(args.period)


if __name__ == "__main__":
    main()
