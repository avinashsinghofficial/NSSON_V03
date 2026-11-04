#!/usr/bin/env python3
import sys
from pathlib import Path
import csv

# ---------------------------------------------------------
# Project root and imports
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.cifar10_cnn import (
    load_cnn_checkpoint,
    get_cifar10_test_loader,
    evaluate_cnn,
)
from snn.snn_snasnet.eval_snasnet_cifar10 import (
    load_snn_checkpoint,
    get_snn_test_loader,
    evaluate_snn,
)

# Import the runner we just defined
from jetson_emulation.jetson_emulation.jetson_runner import JetsonEmulatedRunner

# ---------------------------------------------------------
# Node configs and output directory (latest_experiment)
# ---------------------------------------------------------
NODE_CONFIGS = [
    {"nodes": 2, "aps": 1, "switches": 1},
    {"nodes": 10, "aps": 1, "switches": 1},
    {"nodes": 20, "aps": 2, "switches": 1},
    {"nodes": 30, "aps": 3, "switches": 2},
    {"nodes": 40, "aps": 4, "switches": 2},
    {"nodes": 50, "aps": 5, "switches": 3},
    {"nodes": 100, "aps": 10, "switches": 5},
    {"nodes": 300, "aps": 20, "switches": 10},
]

BASE_DIR = PROJECT_ROOT / "latest_experiment"
LOG_DIR = BASE_DIR / "logs"
TEL_DIR = LOG_DIR / "telemetry_nsson_experimental"

TEL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def reset_snn_state(model):
    for m in model.modules():
        if hasattr(m, "reset"):
            m.reset()

# ---------------------------------------------------------
# Single config run (Jetson + NSSON experimental)
# ---------------------------------------------------------
def run_single_config(cfg):
    nodes = cfg["nodes"]
    aps = cfg["aps"]
    switches = cfg["switches"]

    print(f"[NSSON experimental] nodes={nodes}, aps={aps}, switches={switches}")

    # CNN
    cnn_model, cnn_device = load_cnn_checkpoint()
    cnn_loader = get_cifar10_test_loader()
    cnn_acc = evaluate_cnn(cnn_model, cnn_loader, cnn_device)

    # SNN
    snn_model, snn_args, snn_device = load_snn_checkpoint()
    snn_loader = get_snn_test_loader(snn_args)
    snn_metrics = evaluate_snn(snn_model, snn_loader, snn_device)
    snn_acc = snn_metrics["accuracy"]

    # Jetson emulator runner
    runner = JetsonEmulatedRunner(power_mode="5W")

    # Use one batch from SNN loader for both models
    batch, _ = next(iter(snn_loader))
    batch_cnn = batch.to(cnn_device)
    batch_snn = batch.to(snn_device)

    reset_snn_state(snn_model)

    cnn_hw = runner.run_inference(cnn_model, batch_cnn, mode="cnn")
    snn_hw = runner.run_inference(snn_model, batch_snn, mode="snn")

    out_path = TEL_DIR / f"nsson_experimental_nodes_{nodes}.csv"
    with out_path.open("w", newline="") as f:
        fieldnames = [
            "nodes",
            "aps",
            "switches",
            "model_used",
            "execution_site",
            "latency_ms",
            "power_mw",
            "energy_mj",
            "accuracy",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # CNN row (local exec)
        writer.writerow(
            {
                "nodes": nodes,
                "aps": aps,
                "switches": switches,
                "model_used": "cnn",
                "execution_site": "local",
                "latency_ms": cnn_hw["latency_ms"],
                "power_mw": cnn_hw["power_mw"],
                "energy_mj": cnn_hw["energy_mj"],
                "accuracy": cnn_acc,
            }
        )

        # SNN row (local exec)
        writer.writerow(
            {
                "nodes": nodes,
                "aps": aps,
                "switches": switches,
                "model_used": "snn",
                "execution_site": "local",
                "latency_ms": snn_hw["latency_ms"],
                "power_mw": snn_hw["power_mw"],
                "energy_mj": snn_hw["energy_mj"],
                "accuracy": snn_acc,
            }
        )

    print(f"[NSSON experimental] written {out_path}")

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    for cfg in NODE_CONFIGS:
        run_single_config(cfg)

if __name__ == "__main__":
    main()
