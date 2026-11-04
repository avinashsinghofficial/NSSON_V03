#!/usr/bin/env python3
"""
NSSON_V02 experiment runner.

Run exactly one scenario per invocation:
  local_only
  offload_only
  adaptive_no_sdn
  baseline_sdn_iot
  nsson_full

The runner writes all raw outputs into --output-dir. The shell experiment
orchestrator snapshots these logs after each individual run.
"""

import argparse
import inspect
import os
import shutil
import sys

ROOT = "/home/avi/NSSON_V02"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

VALID_MODES = (
    "local_only",
    "offload_only",
    "adaptive_no_sdn",
    "baseline_sdn_iot",
    "nsson_full",
)


def call_supported(function, **kwargs):
    """Call a project function using only arguments accepted by its signature."""
    signature = inspect.signature(function)
    accepted = {
        name: value
        for name, value in kwargs.items()
        if name in signature.parameters
    }
    return function(**accepted)


def clean_logs(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for name in (
        "metrics.csv",
        "offloaddecisionlog.json",
        "decisionlog.json",
        "offloadlatencylog.csv",
        "latencylog.csv",
        "offloadpowerlog.csv",
        "powerlog.csv",
        "thresholdlog.csv",
        "ryuflowlog.jsonl",
        "linkstats.csv",
    ):
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            os.remove(path)


def copy_expected_log(source_dir, destination_dir, candidate_names):
    for name in candidate_names:
        source = os.path.join(source_dir, name)
        if os.path.isfile(source):
            target = os.path.join(destination_dir, name)
            if os.path.abspath(source) != os.path.abspath(target):
                shutil.copy2(source, target)
            return target
    return None


def normalize_output_logs(output_dir):
    """
    Keep output file names consistent with the aggregation script.
    Existing project modules may write either original or new names.
    """
    latency = copy_expected_log(
        ROOT,
        output_dir,
        ("offloadlatencylog.csv", "latencylog.csv")
    )
    power = copy_expected_log(
        ROOT,
        output_dir,
        ("offloadpowerlog.csv", "powerlog.csv")
    )
    decision = copy_expected_log(
        ROOT,
        output_dir,
        ("offloaddecisionlog.json", "decisionlog.json")
    )

    print("[LOGS] latency:", latency or "not found")
    print("[LOGS] power:", power or "not found")
    print("[LOGS] decision:", decision or "not found")


def run_mode(mode, cycles, output_dir):
    clean_logs(output_dir)

    try:
        from offloadengine import run_offload
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import run_offload from /home/avi/NSSON_V02/offloadengine.py"
        ) from exc

    try:
        from latencytracker import measure_latency
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import measure_latency from /home/avi/NSSON_V02/latencytracker.py"
        ) from exc

    try:
        from powermodel import compute_power
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import compute_power from /home/avi/NSSON_V02/powermodel.py"
        ) from exc

    try:
        from feedbackloop import update_threshold
    except ImportError:
        update_threshold = None

    if mode == "local_only":
        print("[MODE] local_only: all inference decisions must be local")
        call_supported(
            run_offload,
            cycles=cycles,
            base_theta=0.0,
            mode=mode,
            force_decision="local",
            output_dir=output_dir,
        )

    elif mode == "offload_only":
        print("[MODE] offload_only: all inference decisions must be offloaded")
        call_supported(
            run_offload,
            cycles=cycles,
            base_theta=1.0,
            mode=mode,
            force_decision="offload",
            output_dir=output_dir,
        )

    elif mode == "adaptive_no_sdn":
        print("[MODE] adaptive_no_sdn: confidence feedback without SDN")
        call_supported(
            run_offload,
            cycles=cycles,
            base_theta=0.60,
            mode=mode,
            sdn_enabled=False,
            output_dir=output_dir,
        )

    elif mode == "baseline_sdn_iot":
        print("[MODE] baseline_sdn_iot: static/best-effort SDN, no NSSON QoS control")
        call_supported(
            run_offload,
            cycles=cycles,
            base_theta=0.60,
            mode=mode,
            sdn_enabled=True,
            qos_enabled=False,
            output_dir=output_dir,
        )

    elif mode == "nsson_full":
        print("[MODE] nsson_full: adaptive SMCC plus QoS-aware NSSON SDN")
        call_supported(
            run_offload,
            cycles=cycles,
            base_theta=0.60,
            mode=mode,
            sdn_enabled=True,
            qos_enabled=True,
            output_dir=output_dir,
        )

    call_supported(
        measure_latency,
        cycles=cycles,
        mode=mode,
        output_dir=output_dir,
    )

    call_supported(
        compute_power,
        mode=mode,
        output_dir=output_dir,
    )

    if mode in ("adaptive_no_sdn", "baseline_sdn_iot", "nsson_full"):
        if update_threshold is None:
            print("[WARN] feedbackloop.py unavailable; threshold update not run")
        else:
            call_supported(
                update_threshold,
                mode=mode,
                output_dir=output_dir,
            )

    normalize_output_logs(output_dir)
    print(f"[DONE] {mode} completed. Raw logs are in: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one NSSON_V02 experimental mode."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=VALID_MODES,
        help="Experimental scenario to execute."
    )
    parser.add_argument(
        "--cycles",
        default=300,
        type=int,
        help="Inference cycles for this run."
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(ROOT, "logs"),
        help="Directory where runtime logs are written."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.cycles <= 0:
        raise ValueError("--cycles must be a positive integer")

    output_dir = os.path.abspath(args.output_dir)
    run_mode(args.mode, args.cycles, output_dir)


if __name__ == "__main__":
    main()
