#!/usr/bin/env python3
import time
import json
from pathlib import Path
from typing import Dict, Any

import torch
from torch import nn

OUT_PATH = Path("jetson_emulation") / "jetson_bases.json"


class JetsonEmulatedRunner:
    """
    Simple synthetic Jetson emulator.

    - Measures wall-clock latency of a single forward pass.
    - Estimates power and energy using fixed coefficients.
    - Supports two modes: 'cnn' and 'snn'.
    """

    def __init__(self, power_mode: str = "5W") -> None:
        self.power_mode = power_mode
        # Simple base power in mW depending on mode string
        if power_mode == "5W":
            self.base_power_mw = 5000.0
        elif power_mode == "10W":
            self.base_power_mw = 10000.0
        else:
            self.base_power_mw = 7000.0

    @torch.no_grad()
    def run_inference(self, model: nn.Module, batch: torch.Tensor, mode: str) -> Dict[str, Any]:
        """
        Run a single forward pass and return synthetic HW metrics.
        Returns dict with keys: latency_ms, power_mw, energy_mj, throughput_fps.
        """
        model.eval()
        device = next(model.parameters()).device
        batch = batch.to(device)

        # Measure latency (ms)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        _ = model(batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()

        latency_ms = (t1 - t0) * 1000.0

        # Very simple synthetic power model:
        #   - base power depends on power_mode
        #   - SNN assumed slightly lower power than CNN
        if mode.lower() == "snn":
            power_mw = self.base_power_mw * 0.8
        else:
            power_mw = self.base_power_mw

        # energy (mJ) = power (mW) * time (ms) / 1000
        energy_mj = power_mw * latency_ms / 1000.0

        # throughput in frames per second (assuming batch size is len(batch))
        batch_size = batch.shape[0]
        latency_s = latency_ms / 1000.0
        throughput_fps = batch_size / latency_s if latency_s > 0 else 0.0

        return {
            "latency_ms": latency_ms,
            "power_mw": power_mw,
            "energy_mj": energy_mj,
            "throughput_fps": throughput_fps,
        }

    def profile_mode(self, mode: str, base_accuracy: float = 0.0) -> Dict[str, Any]:
        """
        Synthetic profile for a given mode without a real model.
        Used only for generating base values, not for real runs.
        """
        # Choose a synthetic latency depending on mode
        if mode.lower() == "cnn":
            latency_ms = 10.0
        else:
            latency_ms = 15.0

        if mode.lower() == "snn":
            power_mw = self.base_power_mw * 0.8
        else:
            power_mw = self.base_power_mw

        energy_mj = power_mw * latency_ms / 1000.0
        throughput_fps = 1.0 / (latency_ms / 1000.0)

        return {
            "mode": mode,
            "base_accuracy": base_accuracy,
            "latency_ms": latency_ms,
            "power_mw": power_mw,
            "energy_mj": energy_mj,
            "throughput_fps": throughput_fps,
        }


def measure_model(mode: str, base_accuracy: float) -> Dict[str, Any]:
    """
    Wrapper around JetsonEmulatedRunner.profile_mode to build base JSON.
    """
    runner = JetsonEmulatedRunner(power_mode="5W")
    hw = runner.profile_mode(mode, base_accuracy=base_accuracy)

    return {
        "base_accuracy": hw["base_accuracy"],
        "base_local_latency": hw["latency_ms"],
        "base_offload_latency": hw["latency_ms"],
        "base_power_local": hw["power_mw"],
        "base_power_offload": hw["power_mw"],
        "base_throughput": hw["throughput_fps"],
    }


def main() -> None:
    cnn_vals = measure_model("cnn", base_accuracy=93.0)
    snn_vals = measure_model("snn", base_accuracy=90.2)

    bases = {"cnn": cnn_vals, "snn": snn_vals}
    OUT_PATH.write_text(json.dumps(bases, indent=2))
    print("Saved Jetson base values to", OUT_PATH)


if __name__ == "__main__":
    main()
