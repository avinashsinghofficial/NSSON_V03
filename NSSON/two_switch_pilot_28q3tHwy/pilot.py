from pathlib import Path
import json
import subprocess
import time
import uuid

from mininet.net import Mininet
from mininet.node import OVSSwitch
from mininet.log import setLogLevel

ROOT = Path(__file__).resolve().parent
PROFILES = [
    ("autonomous", 1, 8, 15, 5211),
    ("healthcare", 2, 6, 10, 5212),
    ("industrial", 3, 3, 6, 5213),
    ("background", 0, 1, 3, 5214),
]

def command(args):
    p = subprocess.run(
        args, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=15
    )
    if p.returncode:
        raise RuntimeError(f"{' '.join(args)}\n{p.stdout}")
    return p.stdout.strip()

def interval_result(path):
    data = json.loads(path.read_text())
    if "error" in data:
        raise RuntimeError(f"{path.name}: {data['error']}")
    records = [
        item["sum"] for item in data.get("intervals", [])
        if "sum" in item
        and not item["sum"].get("omitted", False)
    ]
    if not records:
        raise RuntimeError(f"No interval records: {path}")
    records.sort(key=lambda r: r["start"])
    gaps = [
        records[i]["start"] - records[i-1]["end"]
        for i in range(1, len(records))
        if abs(records[i]["start"] - records[i-1]["end"]) > 0.002
    ]
    seconds = records[-1]["end"] - records[0]["start"]
    count = sum(int(r["bytes"]) for r in records)
    return {
        "bytes": count,
        "seconds": seconds,
        "mbps": 8 * count / seconds / 1e6,
        "zero_byte_intervals": sum(r["bytes"] == 0 for r in records),
        "interval_boundary_discrepancies_s": gaps,
        "final_report": data.get("end", {}),
    }

def run_case(mode):
    out = ROOT / mode
    out.mkdir()
    tag = uuid.uuid4().hex[:5]
    net = None
    processes = []
    handles = []
    queues = []
    qos = None

    def launch(host, args, filename):
        handle = (out / filename).open("w")
        handles.append(handle)
        proc = host.popen(
            args, stdout=handle, stderr=subprocess.STDOUT
        )
        processes.append(proc)
        return proc

    def capture(label, ingress, egress, switches):
        commands = {
            "ingress_link": ["ip", "-s", "link", "show", ingress],
            "egress_link": ["ip", "-s", "link", "show", egress],
            "classes": ["tc", "-s", "class", "show", "dev", egress],
            "qdisc": ["tc", "-s", "qdisc", "show", "dev", egress],
        }
        for sw in switches:
            commands[f"{sw.name}_flows"] = [
                "ovs-ofctl", "-O", "OpenFlow13",
                "dump-flows", sw.name
            ]
        for name, args in commands.items():
            try:
                text = command(args)
            except Exception as exc:
                text = str(exc)
            (out / f"{label}_{name}.txt").write_text(text + "\n")

    try:
        net = Mininet(
            controller=None, switch=OVSSwitch,
            autoSetMacs=True, build=False
        )
        s1 = net.addSwitch(
            "pa" + tag, protocols="OpenFlow13", failMode="secure"
        )
        s2 = net.addSwitch(
            "pb" + tag, protocols="OpenFlow13", failMode="secure"
        )
        sources = {}
        source_links = {}
        for i, (name, *_rest) in enumerate(PROFILES, 1):
            host = net.addHost(
                f"h{i}{tag}", ip=f"10.233.0.{i}/24"
            )
            source_links[name] = net.addLink(host, s1)
            sources[name] = host

        receiver = net.addHost("hr" + tag, ip="10.233.0.10/24")
        net.addLink(receiver, s2)
        bottleneck = net.addLink(s1, s2)
        net.build()
        net.start()

        egress = bottleneck.intf1.name
        bg_ingress = source_links["background"].intf2.name
        output_port = int(command([
            "ovs-vsctl", "get", "Interface", egress, "ofport"
        ]))
        if output_port <= 0:
            raise RuntimeError("Invalid bottleneck OpenFlow port")

        allocations = (
            [("shared", 0, 20, 20, 0)]
            if mode == "baseline" else PROFILES
        )
        mapping = {}
        for _, number, minimum, maximum, _ in allocations:
            record = command([
                "ovs-vsctl", "create", "Queue",
                f"other-config:min-rate={minimum * 1000000}",
                f"other-config:max-rate={maximum * 1000000}",
            ])
            queues.append(record)
            mapping[number] = record

        qos = command([
            "ovs-vsctl", "create", "QoS",
            "type=linux-htb", "other-config:max-rate=20000000",
            *[f"queues:{k}={v}" for k, v in mapping.items()]
        ])
        command(["ovs-vsctl", "set", "Port", egress, f"qos={qos}"])

        for sw in (s1, s2):
            command([
                "ovs-ofctl", "-O", "OpenFlow13", "add-flow",
                sw.name, "priority=0,actions=NORMAL"
            ])

        for name, number, _, _, port in PROFILES:
            protocol = "udp" if name == "background" else "tcp"
            selected = 0 if mode == "baseline" else number
            command([
                "ovs-ofctl", "-O", "OpenFlow13", "add-flow", s1.name,
                f"priority=100,{protocol},nw_dst=10.233.0.10,"
                f"tp_dst={port},actions=set_queue:{selected},"
                f"output:{output_port}"
            ])

        time.sleep(2)
        for name, host in sources.items():
            text = host.cmd("ping -c 1 -W 1 10.233.0.10")
            (out / f"connectivity_{name}.txt").write_text(text)
            if "1 received" not in text:
                raise RuntimeError(f"Connectivity failed: {name}")

        servers = {}
        for name, _, _, _, port in PROFILES:
            servers[name] = launch(
                receiver,
                ["iperf3", "-s", "-1", "-J", "-p", str(port)],
                f"receiver_{name}.json"
            )
        time.sleep(1)
        if any(p.poll() is not None for p in servers.values()):
            raise RuntimeError("An iperf server exited before testing")

        capture("before", bg_ingress, egress, (s1, s2))
        timestamps = {}
        clients = {}

        timestamps["background_launch"] = time.time()
        clients["background"] = launch(
            sources["background"],
            ["iperf3", "-c", "10.233.0.10", "-p", "5214",
             "-u", "-b", "19M", "-l", "1200",
             "-t", "26", "-i", "1", "-J"],
            "client_background.json"
        )

        time.sleep(3)
        for name, _, _, _, port in PROFILES[:3]:
            timestamps[f"{name}_launch"] = time.time()
            clients[name] = launch(
                sources[name],
                ["iperf3", "-c", "10.233.0.10", "-p", str(port),
                 "-t", "18", "-i", "1", "-J"],
                f"client_{name}.json"
            )

        time.sleep(5)
        timestamps["snapshot"] = time.time()
        capture("during", bg_ingress, egress, (s1, s2))

        for name, proc in clients.items():
            proc.wait(timeout=45)
            if proc.returncode:
                raise RuntimeError(f"Client failed: {name}")
        for name, proc in servers.items():
            proc.wait(timeout=20)
            if proc.returncode:
                raise RuntimeError(f"Server failed: {name}")

        capture("after", bg_ingress, egress, (s1, s2))
        for handle in handles:
            handle.flush()

        results = {
            name: interval_result(out / f"receiver_{name}.json")
            for name, *_ in PROFILES
        }
        sender = interval_result(out / "client_background.json")
        summary = {
            "mode": mode,
            "scope": "Static-network pilot, not inference-aware NSSON",
            "parent_limit_mbps": 20,
            "requested_background_mbps": 19,
            "actual_background_sender_mbps": sender["mbps"],
            "timestamps": timestamps,
            "receiver_results": results,
            "note": "Per-flow measurement windows differ; "
                    "do not treat their sum as synchronized aggregate goodput."
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\n=== {mode} ===")
        print("Background sender Mbps:", round(sender["mbps"], 3))
        for name, result in results.items():
            print(
                name,
                "receiver Mbps:", round(result["mbps"], 3),
                "zero intervals:", result["zero_byte_intervals"]
            )

    except Exception as exc:
        (out / "ERROR.txt").write_text(str(exc) + "\n")
        raise
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for proc in processes:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        for handle in handles:
            handle.close()
        if net is not None:
            net.stop()
        for table, record in (
            [("QoS", qos)] + [("Queue", q) for q in queues]
        ):
            if record:
                subprocess.run(
                    ["ovs-vsctl", "--timeout=5", "--if-exists",
                     "destroy", table, record],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    timeout=10
                )

setLogLevel("warning")
for mode in ("baseline", "static_queues"):
    run_case(mode)
print("\nPilot results:", ROOT)
