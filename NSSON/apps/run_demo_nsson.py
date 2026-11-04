# /home/avi/NSSON/apps/run_demo_nsson.py
#
# Minimal demo runner for NSSON using jetson-emulator + SpikingJelly edge worker.
# No camnsson dependency.

import argparse
import base64
import io
import json
import os
import time
import csv
import numpy as np
import yaml
import requests
from PIL import Image

NSSON_ROOT = "/home/avi/NSSON"

DECISION_LOG = os.path.join(NSSON_ROOT, "logs", "decisionlog.json")
LATENCY_LOG = os.path.join(NSSON_ROOT, "logs", "latencylog.csv")
POWER_LOG = os.path.join(NSSON_ROOT, "logs", "powerlog.csv")
RYU_FLOW_LOG = os.path.join(NSSON_ROOT, "logs", "ryuflowlog.jsonl")

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_sample_image(sample_id: int, width=32, height=32):
    rng = np.random.default_rng(seed=sample_id)
    arr = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)

def img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--cycles", type=int, default=300)
    p.add_argument("--use-webcam", action="store_true")
    p.add_argument("--cam-id", type=int, default=0)
    args = p.parse_args()

    cfg = load_config(args.config)
    edge_url = cfg["edge_url"]
    base_threshold = float(cfg.get("base_threshold", 0.6))

    os.makedirs(os.path.dirname(DECISION_LOG), exist_ok=True)

    with open(LATENCY_LOG, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cycle", "sense_sec", "compute_sec", "transfer_sec", "idle_sec", "total_sec"])

    with open(POWER_LOG, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cycle", "data_mb", "compute_sec", "idle_sec", "compute_power_w", "transfer_power_w", "idle_power_w", "total_power_wh"])

    with open(DECISION_LOG, "w") as f_dec, \
         open(RYU_FLOW_LOG, "w") as f_ryu:

        lat_writer = csv.writer(open(LATENCY_LOG, "a", newline=""))
        pow_writer = csv.writer(open(POWER_LOG, "a", newline=""))

        for cycle in range(1, args.cycles + 1):
            img = get_sample_image(cycle)
            img_b64 = img_to_b64(img)

            threshold = max(0.3, min(0.9, base_threshold - 0.0001 * cycle))

            t0 = time.time()

            payload = {
                "image_b64": img_b64,
                "meta": {
                    "sample_id": cycle,
                    "threshold": threshold
                }
            }

            resp = requests.post(edge_url + "/infer", json=payload, timeout=10)
            resp.raise_for_status()
            result = resp.json()

            t1 = time.time()
            total_sec = t1 - t0

            sense_sec = 0.0
            compute_sec = result["compute_sec"]
            transfer_sec = 0.01 if result["decision"] == "offload" else 0.0
            idle_sec = max(0.0, total_sec - compute_sec - transfer_sec)

            lat_writer.writerow([cycle, sense_sec, compute_sec, transfer_sec, idle_sec, total_sec])

            pow_writer.writerow([
                cycle,
                result["data_mb"],
                result["compute_sec"],
                idle_sec,
                5.0,
                0.5,
                1.5,
                result["power_wh"]
            ])

            dec_rec = {
                "sample_id": cycle,
                "confidence": result["snn_conf"],
                "decision": result["decision"],
                "threshold": threshold,
                "timestamp": time.time()
            }
            f_dec.write(json.dumps(dec_rec) + "\n")

            flow_rec = {
                "cycle": cycle,
                "datapath_id": 1,
                "in_port": 1,
                "src_mac": "00:00:00:00:00:01",
                "dst_mac": "00:00:00:00:00:02",
                "offload_decision": result["decision"],
                "priority": 100 if result["decision"] == "offload" else 10,
                "path_id": "path_A" if result["decision"] == "offload" else "path_B",
                "selected": 1,
                "slice": "control" if result["decision"] == "offload" else "monitoring"
            }
            f_ryu.write(json.dumps(flow_rec) + "\n")

            if cycle % 50 == 0:
                print(f"Completed {cycle}/{args.cycles} cycles")

    print("Demo run complete. Logs written to logs/")

if __name__ == "__main__":
    main()
