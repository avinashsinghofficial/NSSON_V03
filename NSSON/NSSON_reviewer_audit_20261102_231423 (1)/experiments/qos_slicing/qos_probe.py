#!/usr/bin/env python3

import argparse
import csv
import datetime as dt
import json
import re
import subprocess
from pathlib import Path

import yaml


def run_command(command, timeout_seconds):
    try:
        result = subprocess.run(
            command,
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
        return result.returncode, result.stdout
    except subprocess.TimeoutExpired:
        return 124, "COMMAND_TIMEOUT"


def parse_ping_output(output):
    loss_match = re.search(r"(\d+(?:\.\d+)?)%\s*packet loss", output)
    loss_percent = float(loss_match.group(1)) if loss_match else 100.0

    rtt_match = re.search(
        r"=\s*([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)\s*ms",
        output,
    )

    if rtt_match:
        rtt_min_ms = float(rtt_match.group(1))
        rtt_avg_ms = float(rtt_match.group(2))
        rtt_max_ms = float(rtt_match.group(3))
        jitter_ms = float(rtt_match.group(4))
    else:
        rtt_min_ms = float("nan")
        rtt_avg_ms = float("nan")
        rtt_max_ms = float("nan")
        jitter_ms = float("nan")

    return {
        "rtt_min_ms": rtt_min_ms,
        "rtt_avg_ms": rtt_avg_ms,
        "rtt_max_ms": rtt_max_ms,
        "jitter_ms": jitter_ms,
        "loss_percent": loss_percent,
        "pdr": max(0.0, 1.0 - loss_percent / 100.0),
    }


def parse_iperf_json(output):
    try:
        data = json.loads(output)
        end = data.get("end", {})
        summary = end.get("sum_received", {})

        throughput_bps = float(summary.get("bits_per_second", 0.0))
        retransmits = int(summary.get("retransmits", 0))

        return {
            "throughput_mbps": throughput_bps / 1e6,
            "tcp_retransmits": retransmits,
        }
    except Exception:
        return {
            "throughput_mbps": float("nan"),
            "tcp_retransmits": -1,
        }


def compute_available_bandwidth(link_capacity_mbps, throughput_mbps):
    if throughput_mbps != throughput_mbps:
        return float("nan")
    return max(0.0, link_capacity_mbps - throughput_mbps)


def evaluate_sla(metrics, sla):
    flags = {
        "sla_latency_met": int(
            metrics["rtt_avg_ms"] <= float(sla["latency_ms_max"])
        ),
        "sla_jitter_met": int(
            metrics["jitter_ms"] <= float(sla["jitter_ms_max"])
        ),
        "sla_loss_met": int(
            metrics["loss_percent"] <= float(sla["loss_percent_max"])
        ),
        "sla_pdr_met": int(
            metrics["pdr"] >= float(sla["pdr_min"])
        ),
        "sla_throughput_met": int(
            metrics["throughput_mbps"] >= float(sla["throughput_mbps_min"])
        ),
    }

    flags["sla_all_met"] = int(all(flags.values()))
    flags["sla_violation"] = int(not bool(flags["sla_all_met"]))
    return flags


def main():
    parser = argparse.ArgumentParser(
        description="Collect real per-slice QoS and SLA telemetry."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--load", required=True)
    parser.add_argument("--slice", dest="slice_name", required=True)
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--src-host", required=True)
    parser.add_argument("--dst-ip", required=True)
    parser.add_argument("--dst-port", type=int, required=True)
    parser.add_argument("--path-id", required=True)
    parser.add_argument("--snr-db", type=float, default=float("nan"))
    parser.add_argument("--link-capacity-mbps", type=float, default=25.0)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--ping-count", type=int, default=20)
    parser.add_argument("--iperf-seconds", type=int, default=5)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    slice_config = config["slices"][args.slice_name]
    sla = slice_config["sla"]

    ping_command = (
        f"{args.src_host} ping -n -q -c {args.ping_count} "
        f"-W 1 {args.dst_ip}"
    )
    ping_returncode, ping_output = run_command(
        ping_command,
        args.ping_count + 15,
    )
    ping_metrics = parse_ping_output(ping_output)

    iperf_command = (
        f"{args.src_host} iperf3 -J -c {args.dst_ip} "
        f"-p {args.dst_port} -t {args.iperf_seconds}"
    )
    iperf_returncode, iperf_output = run_command(
        iperf_command,
        args.iperf_seconds + 20,
    )
    iperf_metrics = parse_iperf_json(iperf_output)

    available_bandwidth_mbps = compute_available_bandwidth(
        args.link_capacity_mbps,
        iperf_metrics["throughput_mbps"],
    )

    metrics = {
        **ping_metrics,
        **iperf_metrics,
        "available_bandwidth_mbps": available_bandwidth_mbps,
    }
    sla_flags = evaluate_sla(metrics, sla)

    record = {
        "timestamp_utc": dt.datetime.utcnow().isoformat(),
        "cycle": args.cycle,
        "mode": args.mode,
        "load": args.load,
        "slice": args.slice_name,
        "app_class": slice_config["app_class"],
        "path_id": args.path_id,
        "src_host": args.src_host,
        "dst_ip": args.dst_ip,
        "dst_port": args.dst_port,
        "snr_db": args.snr_db,
        "link_capacity_mbps": args.link_capacity_mbps,
        **metrics,
        "sla_latency_ms": sla["latency_ms_max"],
        "sla_jitter_ms": sla["jitter_ms_max"],
        "sla_loss_percent": sla["loss_percent_max"],
        "sla_pdr_min": sla["pdr_min"],
        "sla_throughput_mbps": sla["throughput_mbps_min"],
        **sla_flags,
        "ping_returncode": ping_returncode,
        "iperf_returncode": iperf_returncode,
    }

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()

    with csv_path.open("a", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=record.keys(),
        )
        if write_header:
            writer.writeheader()
        writer.writerow(record)

    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
