#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import yaml


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def quality_inverse(value, minimum, maximum):
    if maximum <= minimum:
        return 0.0
    clipped = clamp(value, minimum, maximum)
    return 1.0 - (clipped - minimum) / (maximum - minimum)


def quality_direct(value, minimum, maximum):
    if maximum <= minimum:
        return 0.0
    clipped = clamp(value, minimum, maximum)
    return (clipped - minimum) / (maximum - minimum)


def meets_core_sla(metrics, sla):
    return (
        metrics["rtt_avg_ms"] <= float(sla["latency_ms_max"])
        and metrics["loss_percent"] <= float(sla["loss_percent_max"])
        and metrics["throughput_mbps"] >= float(sla["throughput_mbps_min"])
    )


def route_score(metrics, weights):
    latency_quality = quality_inverse(metrics["rtt_avg_ms"], 0.0, 100.0)
    loss_quality = quality_inverse(metrics["loss_percent"], 0.0, 100.0)
    snr_quality = quality_direct(metrics["snr_db"], 0.0, 40.0)
    bandwidth_quality = quality_direct(
        metrics["available_bandwidth_mbps"],
        0.0,
        25.0,
    )

    return (
        float(weights["latency"]) * latency_quality
        + float(weights["loss"]) * loss_quality
        + float(weights["snr"]) * snr_quality
        + float(weights["bandwidth"]) * bandwidth_quality
    )


def read_latest_path_metrics(csv_path, slice_name):
    latest = {}

    with Path(csv_path).open() as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            if row["slice"] != slice_name:
                continue

            path_id = row["path_id"]

            latest[path_id] = {
                "path_id": path_id,
                "rtt_avg_ms": float(row["rtt_avg_ms"]),
                "loss_percent": float(row["loss_percent"]),
                "snr_db": float(row["snr_db"]),
                "throughput_mbps": float(row["throughput_mbps"]),
                "available_bandwidth_mbps": float(
                    row["available_bandwidth_mbps"]
                ),
            }

    return latest


def main():
    parser = argparse.ArgumentParser(
        description="Select an adaptive path from real telemetry."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--slice", dest="slice_name", required=True)
    parser.add_argument("--path-metrics-csv", required=True)
    parser.add_argument("--current-path", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    slice_sla = config["slices"][args.slice_name]["sla"]
    weights = config["routing"]["weights"]

    candidates = read_latest_path_metrics(
        args.path_metrics_csv,
        args.slice_name,
    )

    if not candidates:
        raise RuntimeError(
            "No path metrics found for the selected slice. "
            "Collect path telemetry before selecting a route."
        )

    compliant = {
        path_id: metrics
        for path_id, metrics in candidates.items()
        if meets_core_sla(metrics, slice_sla)
    }

    candidate_set = compliant if compliant else candidates

    ranked = sorted(
        candidate_set.values(),
        key=lambda metrics: route_score(metrics, weights),
        reverse=True,
    )

    selected = ranked[0]
    current = candidates.get(args.current_path)

    if current is None:
        reason = "current_path_telemetry_unavailable"
    elif not meets_core_sla(current, slice_sla):
        reason = "current_path_sla_violation"
    else:
        reason = "best_weighted_qos_score"

    decision = {
        "slice": args.slice_name,
        "current_path": args.current_path,
        "selected_path": selected["path_id"],
        "reroute": int(selected["path_id"] != args.current_path),
        "reason": reason,
        "selected_score": route_score(selected, weights),
        "selected_metrics": selected,
        "candidate_paths": candidates,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(decision, indent=2))

    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
