#!/usr/bin/env python3
import json
from pathlib import Path

# We use the meta summaries from Jetson-emulator runs as the source of base values.
META_DIR = Path("logs/logs/meta")
OUT_PATH = Path("jetson_emulation") / "jetson_bases.json"

def collect_meta():
    """Load all meta JSON files under logs/logs/meta that use the jetson_emulator profile."""
    metas = []
    for meta_path in META_DIR.glob("*jetson_emulator.json"):
        with meta_path.open("r") as f:
            m = json.load(f)
        metas.append(m)
    if not metas:
        raise RuntimeError(f"No jetson_emulator meta files found in {META_DIR}")
    return metas

def compute_bases(metas):
    """Derive base values per model ('cnn', 'snn') from jetson_emulator meta summaries."""
    per_model = {}
    for m in metas:
        model = m["model"].lower()
        if model not in per_model:
            per_model[model] = {
                "latencies": [],
                "powers": [],
                "throughputs": [],
                "accuracies": [],
            }
        per_model[model]["latencies"].append(m["avg_latency_ms"])
        per_model[model]["powers"].append(m["avg_power_mw"])
        per_model[model]["throughputs"].append(m["avg_throughput_fps"])
        per_model[model]["accuracies"].append(m["accuracy"])

    bases = {}
    for model, d in per_model.items():
        bases[model] = {
            "base_accuracy": sum(d["accuracies"]) / len(d["accuracies"]),
            "base_local_latency": sum(d["latencies"]) / len(d["latencies"]),
            "base_offload_latency": sum(d["latencies"]) / len(d["latencies"]),
            "base_power_local": sum(d["powers"]) / len(d["powers"]),
            "base_power_offload": sum(d["powers"]) / len(d["powers"]),
            "base_throughput": sum(d["throughputs"]) / len(d["throughputs"]),
        }
    return bases

def main():
    metas = collect_meta()
    bases = compute_bases(metas)
    OUT_PATH.write_text(json.dumps(bases, indent=2))
    print("Saved Jetson-based model defaults to", OUT_PATH)

if __name__ == "__main__":
    main()
