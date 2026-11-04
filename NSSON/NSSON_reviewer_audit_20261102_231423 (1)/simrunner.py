#!/usr/bin/env python3
"""
NSSON_V02 single-mode experiment runner.

Runs exactly one scenario per invocation and leaves all generated raw telemetry
directly in the requested --output-dir. No unnecessary copy step is performed.

Modes:
  local_only       : every sample is processed locally
  offload_only     : every sample is offloaded
  adaptive_no_sdn  : adaptive confidence policy; no SDN controller
  baseline_sdn_iot : conventional/static SDN; no NSSON QoS policy
  nsson_full       : adaptive NSSON policy plus QoS-aware SDN
"""

import argparse
import inspect
import os
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
    """
    Call an existing NSSON function using only keyword arguments accepted by
    that function. This lets the runner work during incremental refactoring.
    """
    parameters = inspect.signature(function).parameters
    accepted = {k: v for k, v in kwargs.items() if k in parameters}
    return function(**accepted)


def ensure_clean_log_dir(output_dir):
    """
    Remove only runtime telemetry files. Do not delete model checkpoints,
    training histories, or unrelated log artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)

    runtime_files = (
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
    )

    for name in runtime_files:
        path = os.path.join(output_dir, name)
        if os.path.isfile(path):
            os.remove(path)


def report_output_logs(output_dir):
    """
    Report only the files actually written by offloadengine, latencytracker,
    powermodel, feedbackloop, or the controller.
    """
    candidates = (
        "offloaddecisionlog.json",
        "decisionlog.json",
        "offloadlatencylog.csv",
        "latencylog.csv",
        "offloadpowerlog.csv",
        "powerlog.csv",
        "thresholdlog.csv",
        "ryuflowlog.jsonl",
        "linkstats.csv",
        "metrics.csv",
    )

    print("\n[OUTPUT] Runtime files:")
    found = 0

    for name in candidates:
        path = os.path.join(output_dir, name)
        if os.path.isfile(path):
            size = os.path.getsize(path)
            print(f"  [OK] {path} ({size} bytes)")
            found += 1

    if found == 0:
        raise RuntimeError(
            f"No telemetry logs were found in {output_dir}. "
            "Check the hard-coded output paths in offloadengine.py, "
            "latencytracker.py, and powermodel.py."
        )


def run_one_mode(mode, cycles, output_dir):
    """
    Execute exactly one mode. Existing modules may accept a subset of the
    provided parameters; call_supported() retains only supported arguments.
    """
    ensure_clean_log_dir(output_dir)

    try:
        from offloadengine import run_offload
        from latencytracker import measure_latency
        from powermodel import compute_power
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import NSSON modules. Confirm these files exist:\n"
            "  /home/avi/NSSON_V02/offloadengine.py\n"
            "  /home/avi/NSSON_V02/latencytracker.py\n"
            "  /home/avi/NSSON_V02/powermodel.py"
        ) from exc

    try:
        from feedbackloop import update_threshold
    except ImportError:
        update_threshold = None

    common = {
        "cycles": cycles,
        "mode": mode,
        "output_dir": output_dir,
    }

    if mode == "local_only":
        print("[MODE] local_only: all decisions must be local; theta = 0.0")
        call_supported(
            run_offload,
            **common,
            base_theta=0.0,
            force_decision="local",
            sdn_enabled=False,
            qos_enabled=False,
        )

    elif mode == "offload_only":
        print("[MODE] offload_only: all decisions must be offload; theta = 1.0")
        call_supported(
            run_offload,
            **common,
            base_theta=1.0,
            force_decision="offload",
            sdn_enabled=False,
            qos_enabled=False,
        )

    elif mode == "adaptive_no_sdn":
        print("[MODE] adaptive_no_sdn: confidence feedback; no SDN/QoS controller")
        call_supported(
            run_offload,
            **common,
            base_theta=0.60,
            sdn_enabled=False,
            qos_enabled=False,
        )

    elif mode == "baseline_sdn_iot":
        print("[MODE] baseline_sdn_iot: conventional static/best-effort SDN")
        call_supported(
            run_offload,
            **common,
            base_theta=0.60,
            sdn_enabled=True,
            qos_enabled=False,
        )

    elif mode == "nsson_full":
        print("[MODE] nsson_full: adaptive confidence policy plus QoS-aware SDN")
        call_supported(
            run_offload,
            **common,
            base_theta=0.60,
            sdn_enabled=True,
            qos_enabled=True,
        )

    call_supported(measure_latency, **common)
    call_supported(compute_power, **common)

    if mode in ("adaptive_no_sdn", "baseline_sdn_iot", "nsson_full"):
        if update_threshold is None:
            print("[WARN] feedbackloop.update_threshold is not importable.")
        else:
            call_supported(update_threshold, **common)

    report_output_logs(output_dir)
    print(f"\n[DONE] Mode '{mode}' completed successfully.")
    print(f"[DONE] Raw logs are stored in: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run exactly one NSSON_V02 experimental mode."
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=VALID_MODES,
        help="Experimental scenario."
    )

    parser.add_argument(
        "--cycles",
        type=int,
        default=300,
        help="Number of inference cycles."
    )

    parser.add_argument(
        "--output-dir",
        default=os.path.join(ROOT, "logs"),
        help="Directory used for raw telemetry files."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.cycles <= 0:
        raise ValueError("--cycles must be greater than zero.")

    run_one_mode(
        mode=args.mode,
        cycles=args.cycles,
        output_dir=os.path.abspath(args.output_dir),
    )


if __name__ == "__main__":
    main()
