#!/usr/bin/env python3
"""
Measured multi-slice QoS and SLA experiment for NSSON_V02.

Topology:
    qc + qb1 -- s1 -- s2 -- qs + qb2

Traffic:
    qc -> qs TCP/8090 : autonomous slice
    qc -> qs TCP/8091 : healthcare slice
    qc -> qs TCP/8092 : industrial slice
    qb1 -> qb2 UDP/5001 : best-effort background traffic

Modes:
    baseline_sdn_iot : no per-slice queue action; all traffic uses queue 0.
    priority_only    : OpenFlow priorities are active; no OVS queue reservation.
    slicing_sla      : OpenFlow priorities plus OVS per-slice queues.

This script measures actual Mininet/OVS values and stores one row per
slice, load, mode, and repetition. It does not generate synthetic QoS data.

Run:
    sudo /usr/bin/python3 \
      /home/avi/NSSON_V02/qos_multislice/qos_multislice_sla_experiment.py \
      --mode slicing_sla --load heavy --rep 1 --controller-port 6633
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from datetime import datetime

import yaml

from mininet.link import TCLink
from mininet.log import info, setLogLevel
from mininet.net import Mininet
from mininet.node import OVSKernelSwitch, RemoteController


ROOT = "/home/avi/NSSON_V02"
CONFIG_FILE = os.path.join(ROOT, "configs", "qos_multislice_sla.yaml")
RAW_DIR = os.path.join(ROOT, "logs", "qos_multislice_raw")

with open(CONFIG_FILE, "r", encoding="utf-8") as config_file:
    CONFIG = yaml.safe_load(config_file)

LOADS = CONFIG["loads"]
SLICES = CONFIG["slices"]

DURATION_SEC = 14
PING_COUNT = 30
TCP_SECONDS = 5


def shell(command):
    return os.system(command)


def safe_cleanup():
    """Remove only experiment bridges and temporary processes."""
    for bridge in ("s1", "s2"):
        shell(f"ovs-vsctl --if-exists del-br {bridge} >/dev/null 2>&1")

    shell("pkill -f 'mininet:qosms_' >/dev/null 2>&1 || true")
    shell("pkill -f 'nsson_qosms_' >/dev/null 2>&1 || true")
    shell("rm -f /tmp/nsson_qosms_* >/dev/null 2>&1 || true")
    time.sleep(1)


def parse_ping(output):
    tx = 0
    rx = 0
    loss_pct = 100.0
    rtt_min = float("nan")
    rtt_avg = float("nan")
    rtt_max = float("nan")
    rtt_mdev = float("nan")

    match = re.search(
        r"(\d+) packets transmitted, (\d+) received, ([0-9.]+)% packet loss",
        output,
    )
    if match:
        tx = int(match.group(1))
        rx = int(match.group(2))
        loss_pct = float(match.group(3))

    match = re.search(
        r"rtt min/avg/max/mdev = ([0-9.]+)/([0-9.]+)/([0-9.]+)/([0-9.]+) ms",
        output,
    )
    if match:
        rtt_min = float(match.group(1))
        rtt_avg = float(match.group(2))
        rtt_max = float(match.group(3))
        rtt_mdev = float(match.group(4))

    return {
        "ping_tx": tx,
        "ping_rx": rx,
        "ping_loss_pct": loss_pct,
        "pdr": max(0.0, 1.0 - loss_pct / 100.0),
        "rtt_min_ms": rtt_min,
        "rtt_avg_ms": rtt_avg,
        "rtt_max_ms": rtt_max,
        "rtt_jitter_mdev_ms": rtt_mdev,
    }


def parse_iperf_bandwidth(output):
    """
    Parse the final Mbit/s result from classic iperf output.
    """
    lines = [line for line in output.splitlines() if "Mbits/sec" in line]

    if not lines:
        return float("nan")

    match = re.search(r"([0-9.]+)\s+Mbits/sec", lines[-1])

    if match:
        return float(match.group(1))

    return float("nan")


def append_row(csv_path, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    new_file = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))

        if new_file:
            writer.writeheader()

        writer.writerow(row)



def configure_ovs_queues(interface_name, mode):
    """
    Configure four Linux-HTB queues on a bottleneck-facing OVS port.

    The OVS transaction syntax requires a separate ``--`` before every
    create/set operation. Queue identifiers are:
      queue 0: UDP background traffic,
      queue 1: autonomous inference,
      queue 2: healthcare inference,
      queue 3: industrial inference.
    """
    if mode != "slicing_sla":
        return

    background = SLICES["background"]
    autonomous = SLICES["autonomous"]
    healthcare = SLICES["healthcare"]
    industrial = SLICES["industrial"]

    capacity_bps = int(
        float(CONFIG["project"]["bottleneck_capacity_mbps"]) * 1_000_000
    )

    queue_specs = [
        (0, background),
        (1, autonomous),
        (2, healthcare),
        (3, industrial),
    ]

    command_parts = ["ovs-vsctl"]

    queue_aliases = []

    for queue_id, slice_cfg in queue_specs:
        min_rate = int(float(slice_cfg["min_rate_mbps"]) * 1_000_000)
        max_rate = int(float(slice_cfg["max_rate_mbps"]) * 1_000_000)
        alias = f"@q{queue_id}"

        command_parts.extend(
            [
                "--",
                f"--id={alias}",
                "create",
                "Queue",
                f"other-config:min-rate={min_rate}",
                f"other-config:max-rate={max_rate}",
            ]
        )

        queue_aliases.append((queue_id, alias))

    command_parts.extend(
        [
            "--",
            "--id=@qos",
            "create",
            "QoS",
            "type=linux-htb",
            f"other-config:max-rate={capacity_bps}",
        ]
    )

    for queue_id, alias in queue_aliases:
        command_parts.append(f"queues:{queue_id}={alias}")

    command_parts.extend(
        [
            "--",
            "set",
            "Port",
            interface_name,
            "qos=@qos",
        ]
    )

    command = " ".join(command_parts)

    info(f"*** Configuring OVS queues on {interface_name}\n")
    info(f"*** OVS command: {command}\n")

    result = shell(command)

    if result != 0:
        raise RuntimeError(
            f"Failed to configure OVS queues on {interface_name}. "
            f"Command was: {command}"
        )

def get_bottleneck_interface(switch_name):
    """
    Return the deterministic inter-switch bottleneck interface.

    The topology creates links in this fixed order:
      s1-eth1 <-> qc
      s1-eth2 <-> qb1
      s1-eth3 <-> s2

      s2-eth1 <-> qs
      s2-eth2 <-> qb2
      s2-eth3 <-> s1

    Therefore, the s1--s2 bottleneck always uses port/interface 3.
    """
    interface_name = f"{switch_name}-eth3"

    check = subprocess.run(
        ["ip", "link", "show", interface_name],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    if check.returncode != 0:
        raise RuntimeError(
            f"Bottleneck interface {interface_name} was not found. "
            f"Output:\n{check.stdout}"
        )

    return interface_name

def compute_sla(slice_cfg, metrics):
    sla = slice_cfg["sla"]

    latency_ok = metrics["rtt_avg_ms"] <= float(sla["rtt_ms_max"])
    jitter_ok = metrics["rtt_jitter_mdev_ms"] <= float(sla["jitter_ms_max"])
    loss_ok = metrics["ping_loss_pct"] <= float(sla["loss_pct_max"])
    pdr_ok = metrics["pdr"] >= float(sla["pdr_min"])
    throughput_ok = (
        metrics["tcp_throughput_mbps"]
        >= float(sla["throughput_mbps_min"])
    )

    all_ok = latency_ok and jitter_ok and loss_ok and pdr_ok and throughput_ok

    return {
        "sla_rtt_met": int(latency_ok),
        "sla_jitter_met": int(jitter_ok),
        "sla_loss_met": int(loss_ok),
        "sla_pdr_met": int(pdr_ok),
        "sla_throughput_met": int(throughput_ok),
        "sla_all_met": int(all_ok),
        "sla_violation": int(not all_ok),
    }


def run_experiment(mode, load_name, rep, controller_port):
    if mode not in ("baseline_sdn_iot", "priority_only", "slicing_sla"):
        raise ValueError(f"Unsupported mode: {mode}")

    if load_name not in LOADS:
        raise ValueError(f"Unsupported load: {load_name}")

    load_cfg = LOADS[load_name]
    background_rate = float(load_cfg["background_mbps"])
    bottleneck_capacity = float(CONFIG["project"]["bottleneck_capacity_mbps"])
    bottleneck_delay = f"{CONFIG['project']['bottleneck_delay_ms']}ms"
    max_queue_packets = int(CONFIG["project"]["bottleneck_queue_packets"])

    safe_cleanup()
    os.makedirs(RAW_DIR, exist_ok=True)

    info(
        f"\n*** NSSON multi-slice SLA experiment: "
        f"mode={mode}, load={load_name}, rep={rep}\n"
    )

    net = Mininet(
        controller=RemoteController,
        switch=OVSKernelSwitch,
        link=TCLink,
        autoSetMacs=True,
        build=False,
    )

    controller = net.addController(
        "c0",
        controller=RemoteController,
        ip="127.0.0.1",
        port=controller_port,
    )

    client = net.addHost("qc", ip="10.20.0.1/24")
    server = net.addHost("qs", ip="10.20.0.2/24")
    bg_source = net.addHost("qb1", ip="10.20.0.11/24")
    bg_sink = net.addHost("qb2", ip="10.20.0.12/24")

    s1 = net.addSwitch("s1", protocols="OpenFlow13")
    s2 = net.addSwitch("s2", protocols="OpenFlow13")

    net.addLink(client, s1, bw=100, delay="1ms")
    net.addLink(bg_source, s1, bw=100, delay="1ms")
    net.addLink(server, s2, bw=100, delay="1ms")
    net.addLink(bg_sink, s2, bw=100, delay="1ms")

    net.addLink(
        s1,
        s2,
        bw=bottleneck_capacity,
        delay=bottleneck_delay,
        max_queue_size=max_queue_packets,
        use_htb=True,
    )

    net.build()
    controller.start()
    s1.start([controller])
    s2.start([controller])

    info("*** Waiting for OpenFlow handshakes\n")
    time.sleep(4)

    s1_to_s2_interface = get_bottleneck_interface("s1")
    s2_to_s1_interface = get_bottleneck_interface("s2")

    if mode == "slicing_sla":
        configure_ovs_queues(s1_to_s2_interface, mode)
        configure_ovs_queues(s2_to_s1_interface, mode)

    server.cmd("pkill -f 'iperf -s' >/dev/null 2>&1 || true")
    bg_sink.cmd("pkill -f 'iperf -s' >/dev/null 2>&1 || true")

    for slice_name in ("autonomous", "healthcare", "industrial"):
        port = int(SLICES[slice_name]["port"])
        server.cmd(
            f"iperf -s -p {port} "
            f">/tmp/nsson_qosms_{slice_name}_server.log 2>&1 &"
        )

    bg_port = int(SLICES["background"]["port"])
    bg_sink.cmd(
        f"iperf -s -u -p {bg_port} "
        f">/tmp/nsson_qosms_background_server.log 2>&1 &"
    )

    time.sleep(1)

    for slice_name in ("autonomous", "healthcare", "industrial"):
        port = int(SLICES[slice_name]["port"])
        probe = client.cmd(f"nc -zvw 2 {server.IP()} {port} 2>&1")
        info(f"*** {slice_name} TCP/{port} reachability: {probe}")

    info(
        f"*** Starting UDP background load: "
        f"{background_rate:.0f} Mbit/s\n"
    )

    bg_source.cmd(
        f"iperf -c {bg_sink.IP()} -u -p {bg_port} "
        f"-b {background_rate:.0f}M -t {DURATION_SEC} -i 1 "
        f">/tmp/nsson_qosms_background.log 2>&1 &"
    )

    time.sleep(2)

    csv_path = os.path.join(
        RAW_DIR,
        f"multislice_{mode}_{load_name}_rep{rep}.csv",
    )

    text_path = os.path.join(
        RAW_DIR,
        f"multislice_{mode}_{load_name}_rep{rep}.txt",
    )

    with open(text_path, "w", encoding="utf-8") as text_file:
        text_file.write(
            f"MODE={mode}, LOAD={load_name}, REP={rep}\n"
        )
        text_file.write(
            f"BACKGROUND_OFFERED_MBPS={background_rate}\n\n"
        )

        for slice_name in ("autonomous", "healthcare", "industrial"):
            slice_cfg = SLICES[slice_name]
            port = int(slice_cfg["port"])

            info(f"*** Measuring {slice_name} slice on TCP/{port}\n")

            ping_output = client.cmd(
                f"ping -c {PING_COUNT} -i 0.2 -W 1 {server.IP()}"
            )

            tcp_output = client.cmd(
                f"iperf -c {server.IP()} -p {port} "
                f"-t {TCP_SECONDS} -i 1"
            )

            ping_metrics = parse_ping(ping_output)
            tcp_throughput = parse_iperf_bandwidth(tcp_output)

            metrics = {
                **ping_metrics,
                "tcp_throughput_mbps": tcp_throughput,
            }

            sla_result = compute_sla(slice_cfg, metrics)

            row = {
                "timestamp_utc": datetime.utcnow().isoformat(
                    timespec="seconds"
                ) + "Z",
                "mode": mode,
                "load": load_name,
                "rep": rep,
                "slice": slice_name,
                "app_class": slice_cfg["description"],
                "controller_port": controller_port,
                "bottleneck_capacity_mbps": bottleneck_capacity,
                "bottleneck_delay_ms": float(
                    CONFIG["project"]["bottleneck_delay_ms"]
                ),
                "bottleneck_queue_packets": max_queue_packets,
                "offered_background_mbps": background_rate,
                "slice_port": port,
                "slice_priority": int(slice_cfg["priority"]),
                "slice_queue_id": int(slice_cfg["queue_id"]),
                "slice_min_rate_mbps": float(slice_cfg["min_rate_mbps"]),
                "slice_max_rate_mbps": float(slice_cfg["max_rate_mbps"]),
                **metrics,
                "sla_rtt_ms_max": float(slice_cfg["sla"]["rtt_ms_max"]),
                "sla_jitter_ms_max": float(
                    slice_cfg["sla"]["jitter_ms_max"]
                ),
                "sla_loss_pct_max": float(
                    slice_cfg["sla"]["loss_pct_max"]
                ),
                "sla_pdr_min": float(slice_cfg["sla"]["pdr_min"]),
                "sla_throughput_mbps_min": float(
                    slice_cfg["sla"]["throughput_mbps_min"]
                ),
                **sla_result,
            }

            append_row(csv_path, row)

            text_file.write(
                f"\n===== {slice_name.upper()} TCP/{port} =====\n"
            )
            text_file.write("\n--- ICMP ping ---\n")
            text_file.write(ping_output)
            text_file.write("\n--- TCP iperf ---\n")
            text_file.write(tcp_output)

            info(
                f"*** {slice_name}: "
                f"RTT={metrics['rtt_avg_ms']:.3f} ms; "
                f"jitter={metrics['rtt_jitter_mdev_ms']:.3f} ms; "
                f"loss={metrics['ping_loss_pct']:.2f}%; "
                f"throughput={metrics['tcp_throughput_mbps']:.3f} Mbit/s; "
                f"SLA_met={sla_result['sla_all_met']}\n"
            )

    info(f"*** Saved: {csv_path}\n")
    info(f"*** Saved: {text_path}\n")

    net.stop()
    safe_cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="Measured multi-slice SLA experiment for NSSON."
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=["baseline_sdn_iot", "priority_only", "slicing_sla"],
    )

    parser.add_argument(
        "--load",
        required=True,
        choices=sorted(LOADS.keys()),
    )

    parser.add_argument(
        "--rep",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--controller-port",
        type=int,
        default=int(CONFIG["project"]["controller_port"]),
    )

    args = parser.parse_args()

    setLogLevel("info")

    run_experiment(
        mode=args.mode,
        load_name=args.load,
        rep=args.rep,
        controller_port=args.controller_port,
    )


if __name__ == "__main__":
    main()
