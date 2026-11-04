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
bridge = "cal" + tag
net = None
server = None
qos_id = None
queue_id = None

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

def snapshot(label, interface):
    commands = {
        "qdisc": ["tc", "-s", "qdisc", "show", "dev", interface],
        "classes": ["tc", "-s", "class", "show", "dev", interface],
        "flows": ["ovs-ofctl", "-O", "OpenFlow13", "dump-flows", bridge],
        "queues": ["ovs-ofctl", "-O", "OpenFlow13",
                   "queue-stats", bridge],
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
    client = net.addHost("ca" + tag, ip="10.231.0.1/24")
    receiver = net.addHost("cb" + tag, ip="10.231.0.2/24")
    switch = net.addSwitch(
        bridge, protocols="OpenFlow13", failMode="secure"
    )
    net.addLink(client, switch)
    receiver_link = net.addLink(receiver, switch)
    net.build()
    net.start()

    egress = receiver_link.intf2.name
    port = int(run([
        "ovs-vsctl", "get", "Interface", egress, "ofport"
    ]))
    if port <= 0:
        raise RuntimeError(f"Invalid output port: {port}")

    queue_id = run([
        "ovs-vsctl", "create", "Queue",
        "other-config:min-rate=20000000",
        "other-config:max-rate=20000000",
    ])
    qos_id = run([
        "ovs-vsctl", "create", "QoS",
        "type=linux-htb", "other-config:max-rate=20000000",
        f"queues:0={queue_id}",
    ])
    run(["ovs-vsctl", "set", "Port", egress, f"qos={qos_id}"])

    run([
        "ovs-ofctl", "-O", "OpenFlow13", "add-flow", bridge,
        f"priority=100,ip,nw_dst=10.231.0.2,"
        f"actions=set_queue:0,output:{port}",
    ])
    run([
        "ovs-ofctl", "-O", "OpenFlow13", "add-flow", bridge,
        "priority=0,actions=NORMAL",
    ])
    time.sleep(2)

    ping = client.cmd("ping -c 2 -W 1 10.231.0.2")
    (out / "connectivity.txt").write_text(ping)
    if "0% packet loss" not in ping:
        raise RuntimeError("Connectivity check failed; inspect connectivity.txt")

    server_log = (out / "receiver.json").open("w")
    client_log = (out / "client.json").open("w")
    try:
        server = receiver.popen(
            ["iperf3", "-s", "-1", "-J", "-p", "5209"],
            stdout=server_log, stderr=subprocess.STDOUT
        )
        time.sleep(1)
        snapshot("before", egress)

        started = time.time()
        process = client.popen(
            ["iperf3", "-c", "10.231.0.2", "-p", "5209",
             "-t", "12", "-i", "1", "-J"],
            stdout=client_log, stderr=subprocess.STDOUT
        )
        try:
            time.sleep(5)
            snapshot("during", egress)
            process.wait(timeout=30)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)

        server.wait(timeout=15)
        snapshot("after", egress)
        ended = time.time()
    finally:
        client_log.close()
        server_log.close()

    report = json.loads((out / "client.json").read_text())
    if "error" in report:
        raise RuntimeError(report["error"])

    received = report["end"]["sum_received"]
    summary = {
        "purpose": "Single-flow OVS shaping calibration, not NSSON evaluation",
        "configured_limit_mbps": 20,
        "receiver_mbps": received["bits_per_second"] / 1e6,
        "receiver_bytes": received["bytes"],
        "receiver_seconds": received["seconds"],
        "wall_start_epoch": started,
        "wall_end_epoch": ended,
        "bridge": bridge,
        "egress_interface": egress,
        "client_exit_code": process.returncode,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

except Exception as exc:
    (out / "ERROR.txt").write_text(str(exc) + "\n")
    raise

finally:
    if server is not None and server.poll() is None:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()

    if net is not None:
        try:
            net.stop()
        except Exception as exc:
            print("Topology cleanup warning:", exc)

    for table, record in (("QoS", qos_id), ("Queue", queue_id)):
        if record:
            subprocess.run(
                ["ovs-vsctl", "--timeout=5", "--if-exists",
                 "destroy", table, record],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=10
            )
    print("Calibration results:", out)
