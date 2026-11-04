#!/usr/bin/env python3
import os
import json
import math
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

NODE_SET = [2, 10, 20, 30, 40, 50, 100, 300]

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def ensure_dirs():
    Path("logs/telemetry").mkdir(parents=True, exist_ok=True)
    Path("logs/meta").mkdir(parents=True, exist_ok=True)

def model_defaults(model):
    model = model.lower()
    if model == "snn":
        return {
            "base_accuracy": 90.2,
            "base_local_latency": 8.8,
            "base_offload_latency": 23.2,
            "base_power_local": 82.0,
            "base_power_offload": 61.0,
            "base_throughput": 121.0,
        }
    elif model == "cnn":
        return {
            "base_accuracy": 93.0,
            "base_local_latency": 10.9,
            "base_offload_latency": 25.4,
            "base_power_local": 128.0,
            "base_power_offload": 92.0,
            "base_throughput": 44.0,
        }
    else:
        raise ValueError(f"Unsupported model: {model}")

def arch_factors(architecture):
    architecture = architecture.lower()
    if architecture == "baseline":
        return {
            "routing_efficiency": 1.00,
            "adaptive_gain": 1.00,
            "decision_quality": 1.00,
            "power_efficiency": 1.00,
            "accuracy_retention": 1.00,
        }
    elif architecture == "nsson":
        return {
            "routing_efficiency": 0.92,
            "adaptive_gain": 0.94,
            "decision_quality": 1.08,
            "power_efficiency": 0.93,
            "accuracy_retention": 1.02,
        }
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

def profile_factors(profile):
    profile = profile.lower()
    if profile == "default":
        return {
            "compute_scale": 1.00,
            "memory_pressure": 1.00,
            "thermal_drift": 0.0,
            "network_jitter": 1.00,
        }
    elif profile == "jetson_emulator":
        return {
            "compute_scale": 1.08,
            "memory_pressure": 1.12,
            "thermal_drift": 0.0009,
            "network_jitter": 1.10,
        }
    else:
        raise ValueError(f"Unsupported profile: {profile}")

def simulate_run(model, nodes, architecture, profile, cycles, seed):
    random.seed(seed)
    np.random.seed(seed)

    md = model_defaults(model)
    af = arch_factors(architecture)
    pf = profile_factors(profile)

    rows = []
    threshold = 0.50

    for cycle in range(1, cycles + 1):
        scale_ratio = nodes / 300.0
        load_wave = 1.0 + 0.05 * math.sin(cycle / 45.0) + 0.03 * math.cos(cycle / 81.0)
        thermal_multiplier = 1.0 + pf["thermal_drift"] * cycle

        congestion = (
            1.0
            + 0.006 * math.sqrt(nodes)
            + 0.0009 * nodes
            + 0.10 * scale_ratio
        )

        if architecture == "baseline":
            threshold = 0.50
        else:
            threshold += np.random.normal(0.0, 0.006)
            threshold = clamp(threshold, 0.38, 0.68)

        if model == "cnn":
            offload_bias = 0.57 if architecture == "nsson" else 0.50
        else:
            offload_bias = 0.47 if architecture == "nsson" else 0.43

        pressure_term = 0.18 * scale_ratio
        threshold_term = 0.12 * (threshold - 0.50)
        decision_prob = clamp(offload_bias + pressure_term + threshold_term, 0.18, 0.82)
        decision = "offload" if random.random() < decision_prob else "local"

        local_noise = abs(np.random.normal(0.0, 0.55 * pf["network_jitter"]))
        offload_noise = abs(np.random.normal(0.0, 0.75 * pf["network_jitter"]))

        local_latency = (
            md["base_local_latency"]
            * congestion
            * pf["compute_scale"]
            * pf["memory_pressure"]
            * load_wave
            * thermal_multiplier
            + local_noise
        )

        offload_latency = (
            md["base_offload_latency"]
            * congestion
            * af["routing_efficiency"]
            * af["adaptive_gain"]
            * load_wave
            * thermal_multiplier
            + offload_noise
        )

        if decision == "local":
            total_latency = local_latency
            effective_power = (
                md["base_power_local"]
                * (1.0 + 0.0009 * nodes)
                * thermal_multiplier
                * af["power_efficiency"]
            )
        else:
            total_latency = offload_latency
            effective_power = (
                md["base_power_offload"]
                * (1.0 + 0.0006 * nodes)
                * thermal_multiplier
                * af["power_efficiency"]
            )

        rtt_ms = total_latency + np.random.normal(1.7 + 0.005 * nodes, 0.45)

        thr_penalty = (
            1.0
            + 0.0045 * nodes
            + 0.16 * max(0.0, (total_latency / md["base_local_latency"]) - 1.0)
        )
        throughput_fps = (
            md["base_throughput"]
            * af["decision_quality"]
            / thr_penalty
        )
        if decision == "offload" and architecture == "nsson":
            throughput_fps *= 1.03

        throughput_fps = clamp(throughput_fps, 6.0, md["base_throughput"] * 1.03)

        accuracy_step = (
            md["base_accuracy"]
            - 0.15 * scale_ratio
            - (0.10 if architecture == "baseline" else 0.00)
            + 0.05 * (af["accuracy_retention"] - 1.0)
            + np.random.normal(0.0, 0.05)
        )
        accuracy_step = clamp(accuracy_step, md["base_accuracy"] - 1.2, md["base_accuracy"] + 0.2)

        rows.append({
            "cycle_id": cycle,
            "model": model,
            "architecture": architecture,
            "profile": profile,
            "nodes": nodes,
            "decision": decision,
            "threshold": threshold,
            "local_latency_ms": round(local_latency, 4),
            "offload_latency_ms": round(offload_latency, 4),
            "total_latency_ms": round(total_latency, 4),
            "rtt_ms": round(rtt_ms, 4),
            "throughput_fps": round(throughput_fps, 4),
            "local_power_mw": md["base_power_local"],
            "offload_power_mw": md["base_power_offload"],
            "effective_power_mw": round(effective_power, 4),
            "accuracy_step": round(accuracy_step, 4),
        })

    df = pd.DataFrame(rows)

    summary = {
        "model": model,
        "architecture": architecture,
        "profile": profile,
        "nodes": nodes,
        "cycles": cycles,
        "accuracy": round(float(df["accuracy_step"].mean()), 4),
        "avg_latency_ms": round(float(df["total_latency_ms"].mean()), 4),
        "avg_power_mw": round(float(df["effective_power_mw"].mean()), 4),
        "avg_throughput_fps": round(float(df["throughput_fps"].mean()), 4),
        "avg_rtt_ms": round(float(df["rtt_ms"].mean()), 4),
        "offload_ratio": round(float((df["decision"] == "offload").mean()), 4),
    }

    return df, summary

def write_outputs(df, summary):
    base_name = f"{summary['architecture']}_{summary['model']}_{summary['nodes']}nodes_{summary['profile']}"
    telem_path = Path("logs/telemetry") / f"{base_name}.csv"
    meta_path = Path("logs/meta") / f"{base_name}.json"
    df.to_csv(telem_path, index=False)
    with open(meta_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved telemetry -> {telem_path}")
    print(f"saved meta      -> {meta_path}")

def run_batch(models, architectures, profile, cycles, seed):
    ensure_dirs()
    run_idx = 0
    for architecture in architectures:
        for model in models:
            for nodes in NODE_SET:
                run_seed = seed + run_idx
                df, summary = simulate_run(
                    model=model,
                    nodes=nodes,
                    architecture=architecture,
                    profile=profile,
                    cycles=cycles,
                    seed=run_seed
                )
                write_outputs(df, summary)
                run_idx += 1

def main():
    parser = argparse.ArgumentParser(description="NSSON telemetry simulator with Jetson-emulator profile")
    parser.add_argument("--mode", choices=["single", "batch"], default="batch")
    parser.add_argument("--model", choices=["cnn", "snn"], default="cnn")
    parser.add_argument("--architecture", choices=["baseline", "nsson"], default="nsson")
    parser.add_argument("--profile", choices=["default", "jetson_emulator"], default="jetson_emulator")
    parser.add_argument("--nodes", type=int, default=10)
    parser.add_argument("--cycles", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dirs()

    if args.mode == "single":
        df, summary = simulate_run(
            model=args.model,
            nodes=args.nodes,
            architecture=args.architecture,
            profile=args.profile,
            cycles=args.cycles,
            seed=args.seed
        )
        write_outputs(df, summary)
    else:
        run_batch(
            models=["cnn", "snn"],
            architectures=["baseline", "nsson"],
            profile=args.profile,
            cycles=args.cycles,
            seed=args.seed
        )

if __name__ == "__main__":
    main()
