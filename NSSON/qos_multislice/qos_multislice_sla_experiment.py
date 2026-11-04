#!/usr/bin/env python3
"""
NSSON multi-slice SLA experiment using classic iperf.

This version is intentionally based on classic iperf because it is proven to
work reliably inside the current Mininet namespace environment, whereas iperf3
clients returned 'Bad file descriptor' during direct validation.

Topology:
    qa + qh + qi + qb1 -- s1 -- s2 -- qs + qb2

Inference flows:
    qa -> qs TCP/8090 = autonomous
    qh -> qs TCP/8091 = healthcare
    qi -> qs TCP/8092 = industrial

UDP QoS probes:
    qa -> qs UDP/9090 = autonomous
    qh -> qs UDP/9091 = healthcare
    qi -> qs UDP/9092 = industrial

Background:
    qb1 -> qb2 UDP/5001

All metrics are parsed from measured classic iperf output. No synthetic values.
"""

import argparse
import csv
import os
import re
import subprocess
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

TCP_FLOW_SECONDS = 30
BACKGROUND_SECONDS = 36
UDP_PROBE_SECONDS = 8
POST_WARMUP_SECONDS = 2


def shell(command):
    return os.system(command)


def safe_cleanup():
    for bridge in ("s1", "s2"):
        shell(f"ovs-vsctl --if-exists del-br {bridge} >/dev/null 2>&1")

    shell("pkill -f 'mininet:qosms_' >/dev/null 2>&1 || true")
    shell("pkill -f 'nsson_qosms_' >/dev/null 2>&1 || true")
    shell("rm -f /tmp/nsson_qosms_* >/dev/null 2>&1 || true")
    time.sleep(1)


def bandwidth_to_mbps(value, unit):
    unit = unit.upper()

    factors = {
        "K": 1.0 / 1000.0,
        "M": 1.0,
        "G": 1000.0,
    }

    return float(value) * factors[unit]


def parse_bandwidth_mbps(output):
    """
    Return the steady-state median TCP throughput from classic iperf output.

    The first second may include TCP startup; the final two seconds may contain
    shutdown/teardown artifacts. We therefore retain only nonzero intervals
    starting at >=1 s and ending at <= TCP_FLOW_SECONDS-2 s. The median of
    those intervals is reported for every mode and slice.
    """
    active_rates = []

    interval_pattern = re.compile(
        r"\[\s*\d+\]\s+"
        r"([0-9.]+)-([0-9.]+)\s+sec\s+"
        r".*?\s+([0-9.]+)\s+([KMG])bits/sec"
    )

    steady_start_s = 1.0
    steady_end_s = float(TCP_FLOW_SECONDS) - 2.0

    for line in output.splitlines():
        match = interval_pattern.search(line)

        if not match:
            continue

        start_s = float(match.group(1))
        end_s = float(match.group(2))
        duration_s = end_s - start_s

        if duration_s < 0.5:
            continue

        if start_s < steady_start_s:
            continue

        if end_s > steady_end_s:
            continue

        rate_mbps = bandwidth_to_mbps(
            match.group(3),
            match.group(4),
        )

        if rate_mbps > 0.001:
            active_rates.append(rate_mbps)

    if not active_rates:
        return float("nan")

    active_rates.sort()
    middle = len(active_rates) // 2

    if len(active_rates) % 2 == 1:
        return active_rates[middle]

    return 0.5 * (
        active_rates[middle - 1] + active_rates[middle]
    )

def parse_udp_iperf(output):
    """
    Parse classic iperf UDP output:
    bandwidth, jitter, packet loss, PDR, packets/lost packets.

    Expected summary form:
      [  3] 0.0-8.0 sec ... 300 Kbits/sec 0.012 ms 0/208 (0%)
    """
    bandwidth_mbps = parse_bandwidth_mbps(output)
    jitter_ms = float("nan")
    loss_pct = 100.0
    packets = 0
    lost_packets = 0
    valid = 0

    for line in output.splitlines():
        match = re.search(
            r"([0-9.]+)\s+([KMG])bits/sec\s+"
            r"([0-9.]+)\s+ms\s+"
            r"(\d+)\s*/\s*(\d+)\s+\(([0-9.]+)%\)",
            line,
        )

        if match:
            bandwidth_mbps = bandwidth_to_mbps(
                match.group(1),
                match.group(2),
            )
            jitter_ms = float(match.group(3))
            lost_packets = int(match.group(4))
            packets = int(match.group(5))
            loss_pct = float(match.group(6))
            valid = 1

    return {
        "udp_probe_throughput_mbps": bandwidth_mbps,
        "udp_jitter_ms": jitter_ms,
        "udp_loss_pct": loss_pct,
        "udp_pdr": max(0.0, 1.0 - loss_pct / 100.0),
        "udp_packets": packets,
        "udp_lost_packets": lost_packets,
        "udp_valid_measurement": valid,
        "udp_error": "" if valid else "udp_summary_not_found",
    }


def append_row(csv_path, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    new_file = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))

        if new_file:
            writer.writeheader()

        writer.writerow(row)


def get_bottleneck_interface(switch_name):
    interface_name = "s1-eth5" if switch_name == "s1" else "s2-eth3"

    result = subprocess.run(
        ["ip", "link", "show", interface_name],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Bottleneck interface {interface_name} was not found.\n"
            f"{result.stdout}"
        )

    return interface_name


def configure_ovs_queues(interface_name, mode):
    if mode != "slicing_sla":
        return

    queue_specs = [
        (0, SLICES["background"]),
        (1, SLICES["autonomous"]),
        (2, SLICES["healthcare"]),
        (3, SLICES["industrial"]),
    ]

    capacity_bps = int(
        float(CONFIG["project"]["bottleneck_capacity_mbps"]) * 1_000_000
    )

    command_parts = ["ovs-vsctl"]
    aliases = []

    for queue_id, slice_cfg in queue_specs:
        min_rate_bps = int(float(slice_cfg["min_rate_mbps"]) * 1_000_000)
        max_rate_bps = int(float(slice_cfg["max_rate_mbps"]) * 1_000_000)
        alias = f"@q{queue_id}"

        command_parts.extend(
            [
                "--",
                f"--id={alias}",
                "create",
                "Queue",
                f"other-config:min-rate={min_rate_bps}",
                f"other-config:max-rate={max_rate_bps}",
            ]
        )

        aliases.append((queue_id, alias))

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

    for queue_id, alias in aliases:
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

    if shell(" ".join(command_parts)) != 0:
        raise RuntimeError(
            f"OVS queue setup failed on interface {interface_name}."
        )


def compute_sla(slice_cfg, metrics):
    sla = slice_cfg["sla"]

    jitter_ok = metrics["udp_jitter_ms"] <= float(sla["jitter_ms_max"])
    loss_ok = metrics["udp_loss_pct"] <= float(sla["loss_pct_max"])
    pdr_ok = metrics["udp_pdr"] >= float(sla["pdr_min"])
    throughput_ok = (
        metrics["tcp_throughput_mbps"]
        >= float(sla["throughput_mbps_min"])
    )

    all_ok = jitter_ok and loss_ok and pdr_ok and throughput_ok

    return {
        "sla_jitter_met": int(jitter_ok),
        "sla_loss_met": int(loss_ok),
        "sla_pdr_met": int(pdr_ok),
        "sla_throughput_met": int(throughput_ok),
        "sla_all_met": int(all_ok),
        "sla_violation": int(not all_ok),
    }


def kill_iperf_port(host, port):
    host.cmd(f"fuser -k {port}/tcp >/dev/null 2>&1 || true")
    host.cmd(f"fuser -k {port}/udp >/dev/null 2>&1 || true")



# NSSON_UDP_HEALTHCHECK_V2
def udp_server_log_path(slice_name):
    return f"/tmp/nsson_qosms_udp_server_{slice_name}.log"


def udp_port_is_listening(server, udp_port):
    """
    Check whether a classic iperf UDP server has bound the requested port
    inside the Mininet server namespace.
    """
    output = server.cmd(
        f"ss -lun 2>/dev/null | grep -E '[:.]({udp_port})[[:space:]]' || true"
    )
    return str(udp_port) in output


def start_udp_server(server, slice_name):
    """
    Ensure the dedicated UDP receiver is fresh and listening. A dedicated
    per-slice receiver avoids lifecycle conflicts among sequential probes.
    """
    slice_cfg = SLICES[slice_name]
    udp_port = int(slice_cfg["udp_probe_port"])
    log_path = udp_server_log_path(slice_name)

    kill_iperf_port(server, udp_port)
    server.cmd(f"rm -f {log_path}")

    server.cmd(
        f"iperf -s -u -p {udp_port} >{log_path} 2>&1 &"
    )

    time.sleep(0.75)

    if not udp_port_is_listening(server, udp_port):
        diagnostic = server.cmd(f"cat {log_path} 2>/dev/null || true")
        raise RuntimeError(
            f"UDP iperf receiver did not bind: slice={slice_name}, "
            f"port={udp_port}, log={log_path}, output={diagnostic}"
        )

    info(
        f"*** UDP receiver ready: slice={slice_name}, "
        f"port={udp_port}, log={log_path}\n"
    )


def ensure_udp_server(server, slice_name):
    """
    Restart a receiver only if it is not currently listening. This is invoked
    directly before the associated UDP probe.
    """
    udp_port = int(SLICES[slice_name]["udp_probe_port"])

    if not udp_port_is_listening(server, udp_port):
        info(
            f"*** UDP receiver missing for {slice_name}; restarting "
            f"port {udp_port}\n"
        )
        start_udp_server(server, slice_name)


def start_all_servers(server, background_sink):
    """
    Start classic iperf servers for TCP inference, UDP probes, and background.
    Distinct ports prevent conflicts.
    """
    for slice_name in ("autonomous", "healthcare", "industrial"):
        slice_cfg = SLICES[slice_name]

        tcp_port = int(slice_cfg["port"])
        udp_port = int(slice_cfg["udp_probe_port"])

        kill_iperf_port(server, tcp_port)
        kill_iperf_port(server, udp_port)

        server.cmd(
            f"iperf -s -p {tcp_port} "
            f">/tmp/nsson_qosms_tcp_server_{slice_name}.log 2>&1 &"
        )

        start_udp_server(server, slice_name)

    background_port = int(SLICES["background"]["port"])
    kill_iperf_port(background_sink, background_port)

    background_sink.cmd(
        f"iperf -s -u -p {background_port} "
        f">/tmp/nsson_qosms_background_server.log 2>&1 &"
    )

    time.sleep(2)


def warm_up_tcp(slice_sources, server):
    info("*** Warming TCP inference services\n")

    for slice_name, source in slice_sources.items():
        tcp_port = int(SLICES[slice_name]["port"])

        source.cmd(
            f"iperf -c {server.IP()} -p {tcp_port} -t 1 -i 1 "
            f">/tmp/nsson_qosms_warmup_tcp_{slice_name}.log 2>&1"
        )

    time.sleep(POST_WARMUP_SECONDS)


def start_concurrent_tcp_and_background(
    slice_sources,
    server,
    background_source,
    background_sink,
    load_name,
):
    tcp_paths = {}

    for slice_name, source in slice_sources.items():
        tcp_port = int(SLICES[slice_name]["port"])
        output_path = f"/tmp/nsson_qosms_tcp_{slice_name}.log"

        tcp_paths[slice_name] = output_path

        source.cmd(
            f"iperf -c {server.IP()} -p {tcp_port} "
            f"-t {TCP_FLOW_SECONDS} -i 1 "
            f">{output_path} 2>&1 &"
        )

    background_rate_mbps = float(LOADS[load_name]["background_mbps"])
    background_port = int(SLICES["background"]["port"])

    background_source.cmd(
        f"iperf -c {background_sink.IP()} -u "
        f"-p {background_port} "
        f"-b {background_rate_mbps:.0f}M "
        f"-t {BACKGROUND_SECONDS} -i 1 "
        f">/tmp/nsson_qosms_background.log 2>&1 &"
    )

    return tcp_paths


def run_udp_probe(source, server, slice_name):
    """
    Run one classic iperf UDP probe on its persistent dedicated port.
    """
    slice_cfg = SLICES[slice_name]

    udp_port = int(slice_cfg["udp_probe_port"])
    udp_rate_mbps = float(slice_cfg["udp_probe_rate_mbps"])

    ensure_udp_server(server, slice_name)

    client_output = source.cmd(
        f"iperf -c {server.IP()} -u "
        f"-p {udp_port} "
        f"-b {udp_rate_mbps:.2f}M "
        f"-t {UDP_PROBE_SECONDS} -i 1"
    )

    receiver_output = server.cmd(
        f"cat {udp_server_log_path(slice_name)} 2>/dev/null || true"
    )

    # Preserve both sender and receiver output. The parser can use sender
    # summary normally; receiver output makes invalid runs diagnosable.
    return (
        "\n===== UDP CLIENT OUTPUT =====\n"
        + client_output
        + "\n===== UDP RECEIVER OUTPUT =====\n"
        + receiver_output
    )


def read_host_file(host, path):
    return host.cmd(f"cat {path} 2>/dev/null")


def run_experiment(mode, load_name, rep, controller_port):
    valid_modes = (
        "baseline_sdn_iot",
        "priority_only",
        "slicing_sla",
    )

    if mode not in valid_modes:
        raise ValueError(f"Unsupported mode: {mode}")

    if load_name not in LOADS:
        raise ValueError(f"Unsupported load: {load_name}")

    background_rate_mbps = float(LOADS[load_name]["background_mbps"])
    bottleneck_capacity_mbps = float(
        CONFIG["project"]["bottleneck_capacity_mbps"]
    )
    bottleneck_delay_ms = float(CONFIG["project"]["bottleneck_delay_ms"])
    bottleneck_queue_packets = int(
        CONFIG["project"]["bottleneck_queue_packets"]
    )

    safe_cleanup()
    os.makedirs(RAW_DIR, exist_ok=True)

    info(
        f"\n*** NSSON classic-iperf multi-slice experiment: "
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

    qa = net.addHost("qa", ip="10.20.0.1/24")
    qh = net.addHost("qh", ip="10.20.0.2/24")
    qi = net.addHost("qi", ip="10.20.0.3/24")
    qs = net.addHost("qs", ip="10.20.0.10/24")
    qb1 = net.addHost("qb1", ip="10.20.0.11/24")
    qb2 = net.addHost("qb2", ip="10.20.0.12/24")

    slice_sources = {
        "autonomous": qa,
        "healthcare": qh,
        "industrial": qi,
    }

    s1 = net.addSwitch("s1", protocols="OpenFlow13")
    s2 = net.addSwitch("s2", protocols="OpenFlow13")

    net.addLink(qa, s1, bw=100, delay="1ms")
    net.addLink(qh, s1, bw=100, delay="1ms")
    net.addLink(qi, s1, bw=100, delay="1ms")
    net.addLink(qb1, s1, bw=100, delay="1ms")

    net.addLink(qs, s2, bw=100, delay="1ms")
    net.addLink(qb2, s2, bw=100, delay="1ms")

    net.addLink(
        s1,
        s2,
        bw=bottleneck_capacity_mbps,
        delay=f"{bottleneck_delay_ms}ms",
        max_queue_size=bottleneck_queue_packets,
        use_htb=True,
    )

    net.build()
    controller.start()
    s1.start([controller])
    s2.start([controller])

    info("*** Waiting for OpenFlow handshakes\n")
    time.sleep(4)

    if mode == "slicing_sla":
        configure_ovs_queues(get_bottleneck_interface("s1"), mode)
        configure_ovs_queues(get_bottleneck_interface("s2"), mode)

    start_all_servers(qs, qb2)
    warm_up_tcp(slice_sources, qs)

    # NSSON_UDP_BEFORE_TCP_V1
    # Run UDP QoS probes before the long concurrent TCP/background workload.
    # Classic iperf sends the final jitter/loss report over the reverse UDP
    # direction; separating probes from sustained TCP traffic prevents that
    # report exchange from being confounded by long-flow queue state.
    udp_outputs = {}

    info("*** Running controlled UDP QoS probes before TCP workload\n")

    for slice_name, source in slice_sources.items():
        info(f"*** Running UDP probe for {slice_name}\n")
        udp_outputs[slice_name] = run_udp_probe(
            source,
            qs,
            slice_name,
        )
        time.sleep(1)

    info(
        f"*** Starting concurrent TCP slices with "
        f"{background_rate_mbps:.0f} Mbit/s UDP background\n"
    )

    tcp_paths = start_concurrent_tcp_and_background(
        slice_sources,
        qs,
        qb1,
        qb2,
        load_name,
    )

    time.sleep(TCP_FLOW_SECONDS + 2)

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
            f"MODE={mode}\n"
            f"LOAD={load_name}\n"
            f"REP={rep}\n"
            f"TOPOLOGY=qa,qh,qi,qb1--s1--s2--qs,qb2\n"
            f"TCP_TOOL=classic_iperf\n"
            f"UDP_PROBE_TOOL=classic_iperf\n"
            f"BACKGROUND_OFFERED_MBPS={background_rate_mbps}\n"
        )

        for slice_name, source in slice_sources.items():
            slice_cfg = SLICES[slice_name]

            tcp_output = read_host_file(source, tcp_paths[slice_name])
            udp_output = udp_outputs[slice_name]

            tcp_throughput_mbps = parse_bandwidth_mbps(tcp_output)
            udp_metrics = parse_udp_iperf(udp_output)

            metrics = {
                "tcp_throughput_mbps": tcp_throughput_mbps,
                **udp_metrics,
            }

            sla_result = compute_sla(slice_cfg, metrics)

            row = {
                "timestamp_utc": (
                    datetime.utcnow().isoformat(timespec="seconds") + "Z"
                ),
                "mode": mode,
                "load": load_name,
                "rep": rep,
                "slice": slice_name,
                "app_class": slice_cfg["description"],
                "topology_type": "concurrent_multisource_classic_iperf",
                "controller_port": controller_port,
                "bottleneck_capacity_mbps": bottleneck_capacity_mbps,
                "bottleneck_delay_ms": bottleneck_delay_ms,
                "bottleneck_queue_packets": bottleneck_queue_packets,
                "offered_background_mbps": background_rate_mbps,
                "tcp_inference_port": int(slice_cfg["port"]),
                "udp_probe_port": int(slice_cfg["udp_probe_port"]),
                "slice_priority": int(slice_cfg["priority"]),
                "slice_queue_id": int(slice_cfg["queue_id"]),
                "slice_min_rate_mbps": float(slice_cfg["min_rate_mbps"]),
                "slice_max_rate_mbps": float(slice_cfg["max_rate_mbps"]),
                "udp_probe_rate_mbps": float(
                    slice_cfg["udp_probe_rate_mbps"]
                ),
                "tcp_flow_seconds": TCP_FLOW_SECONDS,
                "udp_probe_seconds": UDP_PROBE_SECONDS,
                "tcp_valid_measurement": int(
                    tcp_throughput_mbps == tcp_throughput_mbps
                ),
                **metrics,
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

            text_file.write(f"\n===== {slice_name.upper()} =====\n")
            text_file.write("\n--- TCP IPERF OUTPUT ---\n")
            text_file.write(tcp_output)
            text_file.write("\n--- UDP IPERF OUTPUT ---\n")
            text_file.write(udp_output)

            info(
                f"*** {slice_name}: "
                f"TCP={metrics['tcp_throughput_mbps']:.3f} Mbit/s; "
                f"UDP={metrics['udp_probe_throughput_mbps']:.3f} Mbit/s; "
                f"jitter={metrics['udp_jitter_ms']:.3f} ms; "
                f"loss={metrics['udp_loss_pct']:.2f}%; "
                f"PDR={metrics['udp_pdr']:.4f}; "
                f"tcp_valid={row['tcp_valid_measurement']}; "
                f"udp_valid={metrics['udp_valid_measurement']}; "
                f"SLA_met={sla_result['sla_all_met']}\n"
            )

    info(f"*** Saved: {csv_path}\n")
    info(f"*** Saved: {text_path}\n")

    net.stop()
    safe_cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="NSSON classic-iperf multi-slice QoS/SLA experiment."
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "baseline_sdn_iot",
            "priority_only",
            "slicing_sla",
        ],
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
