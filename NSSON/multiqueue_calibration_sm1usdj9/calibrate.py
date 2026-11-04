from pathlib import Path
import json
import subprocess
import time
import uuid

from mininet.net import Mininet
from mininet.node import OVSSwitch
from mininet.log import setLogLevel

out = Path(__file__).resolve().parent
tag = uuid.uuid4().hex[:5]
bridge = "mq" + tag
net = None
processes = []
handles = []
queue_ids = []
qos_id = None

profiles = [
    ("autonomous", 1, 8, 15, 5211),
    ("healthcare", 2, 6, 10, 5212),
    ("industrial", 3, 3, 6, 5213),
    ("background", 0, 1, 3, 5214),
]

def run(args):
    result = subprocess.run(
        args, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=15
    )
    if result.returncode:
        raise RuntimeError(
            f"Command failed: {' '.join(args)}\n{result.stdout}"
        )
    return result.stdout.strip()

def launch(host, args, filename):
    handle = (out / filename).open("w")
    handles.append(handle)
    process = host.popen(
        args, stdout=handle, stderr=subprocess.STDOUT
    )
    processes.append(process)
    return process

def snapshot(label, interface):
    commands = {
        "classes": ["tc", "-s", "class", "show", "dev", interface],
        "qdisc": ["tc", "-s", "qdisc", "show", "dev", interface],
        "flows": [
            "ovs-ofctl", "-O", "OpenFlow13",
            "dump-flows", bridge
        ],
        "queues": [
            "ovs-ofctl", "-O", "OpenFlow13",
            "queue-stats", bridge
        ],
    }
    for name, command in commands.items():
        try:
            text = run(command)
        except Exception as exc:
            text = str(exc)
        (out / f"{label}_{name}.txt").write_text(text + "\n")

setLogLevel("warning")
try:
    net = Mininet(
        controller=None, switch=OVSSwitch,
        autoSetMacs=True, build=False
    )
    switch = net.addSwitch(
        bridge, protocols="OpenFlow13", failMode="secure"
    )
    sources = {}
    for index, (name, _, _, _, _) in enumerate(profiles, 1):
        host = net.addHost(
            f"h{index}{tag}", ip=f"10.232.0.{index}/24"
        )
        net.addLink(host, switch)
        sources[name] = host

    receiver = net.addHost("hr" + tag, ip="10.232.0.10/24")
    link = net.addLink(receiver, switch)
    net.build()
    net.start()

    egress = link.intf2.name
    port = int(run([
        "ovs-vsctl", "get", "Interface", egress, "ofport"
    ]))
    if port <= 0:
        raise RuntimeError(f"Invalid output port: {port}")

    queue_map = {}
    for name, number, minimum, maximum, service in profiles:
        record = run([
            "ovs-vsctl", "create", "Queue",
            f"other-config:min-rate={minimum * 1000000}",
            f"other-config:max-rate={maximum * 1000000}",
        ])
        queue_ids.append(record)
        queue_map[number] = record

    qos_id = run([
        "ovs-vsctl", "create", "QoS",
        "type=linux-htb", "other-config:max-rate=20000000",
        *[f"queues:{number}={record}"
          for number, record in queue_map.items()],
    ])
    run(["ovs-vsctl", "set", "Port", egress, f"qos={qos_id}"])

    run([
        "ovs-ofctl", "-O", "OpenFlow13", "add-flow", bridge,
        "priority=0,actions=NORMAL",
    ])
    for name, number, minimum, maximum, service in profiles:
        protocol = "udp" if name == "background" else "tcp"
        run([
            "ovs-ofctl", "-O", "OpenFlow13", "add-flow", bridge,
            f"priority=100,{protocol},nw_dst=10.232.0.10,"
            f"tp_dst={service},actions=set_queue:{number},output:{port}",
        ])

    time.sleep(2)
    for name, host in sources.items():
        ping = host.cmd("ping -c 1 -W 1 10.232.0.10")
        (out / f"ping_{name}.txt").write_text(ping)
        if "1 received" not in ping:
            raise RuntimeError(f"Connectivity failed for {name}")

    servers = {}
    for name, _, _, _, service in profiles:
        servers[name] = launch(
            receiver,
            ["iperf3", "-s", "-1", "-J", "-p", str(service)],
            f"receiver_{name}.json"
        )

    time.sleep(1)
    if any(p.poll() is not None for p in servers.values()):
        raise RuntimeError("An iperf3 server exited before testing.")

    snapshot("before", egress)
    launches = {}
    clients = {}

    background = profiles[-1]
    name, _, _, _, service = background
    launches[name] = time.time()
    clients[name] = launch(
        sources[name],
        ["iperf3", "-c", "10.232.0.10", "-p", str(service),
         "-u", "-b", "19M", "-t", "24", "-i", "1", "-J"],
        f"client_{name}.json"
    )

    time.sleep(2)
    for name, _, _, _, service in profiles[:3]:
        launches[name] = time.time()
        clients[name] = launch(
            sources[name],
            ["iperf3", "-c", "10.232.0.10", "-p", str(service),
             "-t", "16", "-i", "1", "-J"],
            f"client_{name}.json"
        )

    time.sleep(6)
    snapshot("during", egress)

    for name, process in clients.items():
        process.wait(timeout=45)
        if process.returncode:
            raise RuntimeError(
                f"{name} client exited with code {process.returncode}"
            )

    for name, process in servers.items():
        process.wait(timeout=20)
        if process.returncode:
            raise RuntimeError(
                f"{name} server exited with code {process.returncode}"
            )

    snapshot("after", egress)
    for handle in handles:
        handle.flush()

    results = []
    for name, number, minimum, maximum, service in profiles[:3]:
        receiver_report = json.loads(
            (out / f"receiver_{name}.json").read_text()
        )
        if "error" in receiver_report:
            raise RuntimeError(receiver_report["error"])

        received = receiver_report["end"]["sum_received"]
        results.append({
            "slice": name,
            "queue_id": number,
            "minimum_mbps": minimum,
            "maximum_mbps": maximum,
            "receiver_mbps": received["bits_per_second"] / 1e6,
            "receiver_bytes": received["bytes"],
            "receiver_seconds": received["seconds"],
        })

    summary = {
        "purpose": "Multi-queue calibration, not NSSON evaluation",
        "parent_limit_mbps": 20,
        "background_offered_mbps": 19,
        "launch_epoch": launches,
        "tcp_results": results,
        "sum_separate_tcp_receiver_averages_mbps":
            sum(item["receiver_mbps"] for item in results),
        "scope_note":
            "Per-flow averages have separate timing boundaries; "
            "their sum is diagnostic, not synchronized aggregate goodput.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

except Exception as exc:
    (out / "ERROR.txt").write_text(str(exc) + "\n")
    raise

finally:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    for handle in handles:
        handle.close()

    if net is not None:
        try:
            net.stop()
        except Exception as exc:
            print("Topology cleanup warning:", exc)

    for table, record in (
        [("QoS", qos_id)] +
        [("Queue", record) for record in queue_ids]
    ):
        if record:
            subprocess.run(
                ["ovs-vsctl", "--timeout=5", "--if-exists",
                 "destroy", table, record],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=10
            )
    print("Calibration results:", out)
