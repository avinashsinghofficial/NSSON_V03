# /home/avi/NSSON/apps/jetson_edge_worker.py
#
# Minimal edge worker using jetson-emulator + SpikingJelly (no camnsson).
# Exposes /infer endpoint used by run_demo_nsson.py

import argparse
import base64
import io
import time
import json
import os
import numpy as np
import yaml
import torch
from flask import Flask, request, jsonify
from PIL import Image

# SpikingJelly import (adjust if your env uses a different name)
try:
    import spikingjelly as sj
except ImportError:
    sj = None

# Optional: jetson-emulator for power modeling
try:
    import jetson_emulator
except ImportError:
    jetson_emulator = None

app = Flask(__name__)

# Global model handles
cnn_model = None
snn_model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# Power model coefficients (Jetson Nano-like, calibrated or from emulator)
Wc = 5.0   # W
Wt = 0.5   # W/MB
Wi = 1.5   # W

def load_cnn(path: str):
    model = torch.load(path, map_location=device)
    model.eval()
    return model

def load_snn(path: str):
    if sj is None:
        raise RuntimeError("SpikingJelly not available but SNN checkpoint specified")
    model = torch.load(path, map_location=device)
    model.eval()
    return model

def preprocess_image(img: Image.Image, size=32):
    img = img.convert("RGB").resize((size, size))
    arr = np.array(img).astype("float32") / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return torch.from_numpy(arr).unsqueeze(0).to(device)

def infer_cnn(img_tensor):
    with torch.no_grad():
        logits = cnn_model(img_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        conf, pred = torch.max(probs, 0)
    return float(conf.item()), int(pred.item())

def infer_snn(img_tensor):
    if sj is None:
        raise RuntimeError("SpikingJelly not available")
    with torch.no_grad():
        out = snn_model(img_tensor)
        if isinstance(out, (tuple, list)):
            out = out[0]
        probs = torch.softmax(out, dim=1).squeeze(0)
        conf, pred = torch.max(probs, 0)
    return float(conf.item()), int(pred.item())

def model_power_estimate(compute_sec: float, data_mb: float, idle_sec: float) -> float:
    p_compute = Wc * compute_sec
    p_transfer = Wt * data_mb
    p_idle = Wi * idle_sec
    e = (p_compute + p_transfer + p_idle) * compute_sec
    return e

@app.route("/infer", methods=["POST"])
def infer():
    data = request.get_json(force=True)
    img_bytes = base64.b64decode(data["image_b64"])
    img = Image.open(io.BytesIO(img_bytes))

    meta = data.get("meta", {})
    sample_id = meta.get("sample_id", -1)

    t0 = time.time()

    img_tensor = preprocess_image(img)
    cnn_conf, cnn_pred = infer_cnn(img_tensor)
    snn_conf, snn_pred = infer_snn(img_tensor)

    t1 = time.time()
    compute_sec = t1 - t0

    threshold = float(meta.get("threshold", 0.6))
    decision = "local" if snn_conf >= threshold else "offload"

    data_mb = 0.1 if decision == "offload" else 0.0
    idle_sec = 0.0

    power_wh = model_power_estimate(compute_sec, data_mb, idle_sec)

    result = {
        "sample_id": sample_id,
        "cnn_conf": cnn_conf,
        "cnn_pred": cnn_pred,
        "snn_conf": snn_conf,
        "snn_pred": snn_pred,
        "decision": decision,
        "threshold": threshold,
        "compute_sec": compute_sec,
        "power_wh": power_wh,
        "data_mb": data_mb,
    }

    return jsonify(result)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    cnn_path = cfg["models"]["cnn_model"]
    snn_path = cfg["models"].get("snn_model")

    cnn_model = load_cnn(cnn_path)
    if snn_path:
        snn_model = load_snn(snn_path)

    host = cfg.get("host", "0.0.0.0")
    port = cfg.get("port", 8090)

    print(f"Starting edge worker on {host}:{port}")
    app.run(host=host, port=port)
