import argparse
import csv
import hashlib
import json
import os
import pwd
import shutil
import signal
import socket
import statistics
import struct
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

ROOT = Path("/home/avi/NSSON")
PYTHON = ROOT / ".venv_nsson_gpu/bin/python3"
SCRIPT = Path(__file__).resolve()

CLIENT_IP = "10.231.77.1"
SERVER_IP = "10.231.77.2"
SERVER_PORT = 8090
REQUESTS_PER_POLICY = 20
THETA = 0.60

POLICIES = (
    "local_cnn",
    "local_snn",
    "remote_cnn",
    "adaptive_fixed",
)


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, default=str) + "\n"
    )


def read_exact(sock, length):
    parts = []
    remaining = length
    while remaining:
        data = sock.recv(remaining)
        if not data:
            raise EOFError("Connection closed before frame completed")
        parts.append(data)
        remaining -= len(data)
    return b"".join(parts)


def send_json(sock, value):
    payload = json.dumps(value).encode("utf-8")
    if len(payload) > 65536:
        raise ValueError("JSON frame too large")
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def receive_json(sock):
    length = struct.unpack("!I", read_exact(sock, 4))[0]
    if length < 2 or length > 65536:
        raise ValueError(f"Invalid JSON frame length: {length}")
    value = json.loads(read_exact(sock, length))
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    return value


def tensor_environment():
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch

    torch.manual_seed(20261102)
    np.random.seed(20261102)
    torch.backends.cudnn.benchmark = False

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    return np, torch, device


def synchronize(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cnn_components(device):
    from models.cifar10_cnn import (
        load_cnn_checkpoint,
        get_cifar10_test_loader,
    )

    model, _ = load_cnn_checkpoint(
        ckpt_path=str(
            ROOT / "models/cnn_baseline/checkpoints/"
            "resnet18_cifar10_best.pth"
        ),
        device=device,
    )
    transform = get_cifar10_test_loader(
        data_root=str(ROOT / "data/cifar10"),
        batch_size=1,
    ).dataset.transform
    return model.eval(), transform


def snn_components(device):
    saved_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]
    try:
        import snn.snn_snasnet.eval_snasnet_cifar10 as api
        model, config, _ = api.load_snn_checkpoint(device=device)
    finally:
        sys.argv = saved_argv

    from spikingjelly.clock_driven import functional

    _, transform = api.data_transforms(config)

    def reset():
        functional.reset_net(model)
        model.neuron_init()

    return model.eval(), transform, reset


def infer(torch, device, model, transform, image, reset=None):
    if reset is not None:
        reset()

    batch = transform(image).unsqueeze(0).to(device)
    scores = model(batch)
    confidence, prediction = torch.softmax(
        scores.float(), dim=1
    ).max(dim=1)

    prediction = int(prediction.item())
    confidence = float(confidence.item())
    synchronize(torch, device)

    if tuple(scores.shape) != (1, 10):
        raise ValueError(f"Unexpected scores: {tuple(scores.shape)}")
    if not torch.isfinite(scores).all().item():
        raise ValueError("Non-finite model output")

    return prediction, confidence


def run_server(args):
    _, torch, device = tensor_environment()
    from PIL import Image

    model, transform = cnn_components(device)

    with torch.no_grad():
        warm_image = Image.new("RGB", (32, 32))
        for _ in range(10):
            infer(torch, device, model, transform, warm_image)

    out = Path(args.out)
    fields = [
        "run_id", "session_id", "request_id", "sample_id",
        "client_ip", "client_port", "server_ip", "server_port",
        "image_sha256", "prediction",
        "confidence_uncalibrated", "server_inference_sec",
    ]

    seen = set()

    with (out / "server_requests.csv").open("w", newline="") as log:
        writer = csv.DictWriter(log, fieldnames=fields)
        writer.writeheader()
        log.flush()

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((SERVER_IP, SERVER_PORT))
            listener.listen(8)

            write_json(out / "server_ready.json", {
                "server_ip": SERVER_IP,
                "server_port": SERVER_PORT,
                "device": str(device),
                "scope": "Workstation CNN inside Mininet server namespace",
            })

            print("SERVER READY", flush=True)

            while True:
                connection, peer = listener.accept()

                with connection:
                    connection.settimeout(60)
                    while True:
                        try:
                            header = receive_json(connection)
                        except EOFError:
                            break

                        for key in (
                            "run_id", "session_id",
                            "request_id", "sample_id",
                            "payload_bytes", "image_sha256",
                        ):
                            if key not in header:
                                raise ValueError(f"Missing field: {key}")

                        if header["run_id"] != out.name:
                            raise ValueError("Wrong run identity")
                        if header["payload_bytes"] != 3072:
                            raise ValueError("Expected one raw RGB 32x32 image")

                        payload = read_exact(connection, 3072)
                        digest = hashlib.sha256(payload).hexdigest()
                        if digest != header["image_sha256"]:
                            raise ValueError("Payload checksum mismatch")

                        identity = (
                            header["run_id"],
                            header["session_id"],
                            header["request_id"],
                        )
                        if identity in seen:
                            raise ValueError("Duplicate request identity")
                        seen.add(identity)

                        image = Image.frombytes("RGB", (32, 32), payload)

                        synchronize(torch, device)
                        start = time.perf_counter_ns()
                        with torch.no_grad():
                            prediction, confidence = infer(
                                torch, device, model, transform, image
                            )
                        end = time.perf_counter_ns()

                        response = {
                            "run_id": header["run_id"],
                            "session_id": header["session_id"],
                            "request_id": header["request_id"],
                            "sample_id": int(header["sample_id"]),
                            "image_sha256": digest,
                            "prediction": prediction,
                            "confidence_uncalibrated": confidence,
                            "server_inference_sec": (end - start) / 1e9,
                            "client_ip": peer[0],
                            "client_port": peer[1],
                            "server_ip": SERVER_IP,
                            "server_port": SERVER_PORT,
                        }

                        writer.writerow(response)
                        log.flush()
                        send_json(connection, response)


def run_client(args):
    np, torch, device = tensor_environment()
    import torchvision

    out = Path(args.out)
    cnn, cnn_transform = cnn_components(device)
    snn, snn_transform, reset_snn = snn_components(device)

    dataset = torchvision.datasets.CIFAR10(
        root=str(ROOT / "data/cifar10"),
        train=False,
        download=False,
        transform=None,
    )

    ids = np.random.default_rng(20261102).permutation(
        len(dataset)
    )[:REQUESTS_PER_POLICY].tolist()

    samples = {}
    for sample_id in ids:
        image, label = dataset[sample_id]
        payload = image.tobytes()
        if len(payload) != 3072:
            raise ValueError("Unexpected raw image size")
        samples[sample_id] = (
            image, int(label), payload,
            hashlib.sha256(payload).hexdigest(),
        )

    # Warm-up both local model paths outside recorded requests.
    with torch.no_grad():
        for sample_id in ids[:10]:
            image = samples[sample_id][0]
            infer(torch, device, cnn, cnn_transform, image)
            infer(
                torch, device, snn, snn_transform, image,
                reset=reset_snn,
            )

    fields = [
        "run_id", "policy", "sample_id", "label",
        "decision", "threshold", "snn_confidence_uncalibrated",
        "final_prediction", "correct",
        "local_phase_sec", "remote_rpc_sec",
        "request_sec", "residual_overhead_sec",
        "session_id", "request_id",
        "client_ip", "client_port", "server_ip", "server_port",
        "image_sha256", "server_inference_sec",
    ]

    write_json(out / "client_configuration.json", {
        "policies": POLICIES,
        "requests_per_policy": REQUESTS_PER_POLICY,
        "sample_ids": ids,
        "threshold": THETA,
        "device": str(device),
        "scope": (
            "Two network namespaces on the same workstation. "
            "Shared physical CPU/GPU resources. No Jetson measurements."
        ),
        "timing_scope": (
            "Original image retrieval and raw-payload preparation excluded. "
            "Local phase includes preprocessing, reset where applicable, "
            "inference, and prediction extraction. Remote RPC includes "
            "connection establishment, transport, server execution, and response."
        ),
        "limitations": [
            "Smoke test, not a final repeated benchmark.",
            "Static OpenFlow forwarding; no Ryu association or QoS queues.",
            "Fixed threshold; no adaptive feedback.",
            "No energy model applied.",
        ],
    })

    with (out / "client_requests.csv").open("w", newline="") as log:
        writer = csv.DictWriter(log, fieldnames=fields)
        writer.writeheader()

        for policy in POLICIES:
            session_id = str(uuid.uuid4())
            connection = None

            print(f"CLIENT POLICY: {policy}", flush=True)

            try:
                with torch.no_grad():
                    for sequence, sample_id in enumerate(ids):
                        image, label, payload, digest = samples[sample_id]

                        local_sec = 0.0
                        rpc_sec = 0.0
                        snn_confidence = ""
                        response = None

                        synchronize(torch, device)
                        start = time.perf_counter_ns()

                        if policy == "local_cnn":
                            phase_start = time.perf_counter_ns()
                            prediction, _ = infer(
                                torch, device,
                                cnn, cnn_transform, image,
                            )
                            local_sec = (
                                time.perf_counter_ns() - phase_start
                            ) / 1e9
                            offload = False

                        elif policy in ("local_snn", "adaptive_fixed"):
                            phase_start = time.perf_counter_ns()
                            prediction, snn_confidence = infer(
                                torch, device,
                                snn, snn_transform, image,
                                reset=reset_snn,
                            )
                            local_sec = (
                                time.perf_counter_ns() - phase_start
                            ) / 1e9
                            offload = (
                                policy == "adaptive_fixed"
                                and snn_confidence < THETA
                            )

                        else:
                            # Explicit offload-only: not theta=1.
                            offload = True

                        request_id = f"{policy}:{sequence}:{sample_id}"

                        if offload:
                            rpc_start = time.perf_counter_ns()

                            if connection is None:
                                connection = socket.create_connection(
                                    (SERVER_IP, SERVER_PORT),
                                    timeout=60,
                                )
                                connection.settimeout(60)

                            header = {
                                "run_id": out.name,
                                "session_id": session_id,
                                "request_id": request_id,
                                "sample_id": sample_id,
                                "payload_bytes": len(payload),
                                "image_sha256": digest,
                            }

                            send_json(connection, header)
                            connection.sendall(payload)
                            response = receive_json(connection)

                            for key in (
                                "run_id", "session_id", "request_id",
                                "sample_id", "image_sha256",
                            ):
                                if response.get(key) != header[key]:
                                    raise RuntimeError(
                                        f"Response association mismatch: {key}"
                                    )

                            local_address = connection.getsockname()
                            if (
                                response["client_ip"] != local_address[0]
                                or response["client_port"] != local_address[1]
                            ):
                                raise RuntimeError(
                                    "Client/server transport tuple mismatch"
                                )

                            prediction = int(response["prediction"])
                            rpc_sec = (
                                time.perf_counter_ns() - rpc_start
                            ) / 1e9

                        total_sec = (
                            time.perf_counter_ns() - start
                        ) / 1e9

                        row = {
                            "run_id": out.name,
                            "policy": policy,
                            "sample_id": sample_id,
                            "label": label,
                            "decision": "offload" if offload else "local",
                            "threshold": (
                                THETA if policy == "adaptive_fixed" else ""
                            ),
                            "snn_confidence_uncalibrated": snn_confidence,
                            "final_prediction": prediction,
                            "correct": int(prediction == label),
                            "local_phase_sec": local_sec,
                            "remote_rpc_sec": rpc_sec,
                            "request_sec": total_sec,
                            "residual_overhead_sec": (
                                total_sec - local_sec - rpc_sec
                            ),
                            "session_id": (
                                session_id if offload else ""
                            ),
                            "request_id": (
                                request_id if offload else ""
                            ),
                            "client_ip": (
                                response["client_ip"] if offload else ""
                            ),
                            "client_port": (
                                response["client_port"] if offload else ""
                            ),
                            "server_ip": (
                                response["server_ip"] if offload else ""
                            ),
                            "server_port": (
                                response["server_port"] if offload else ""
                            ),
                            "image_sha256": digest,
                            "server_inference_sec": (
                                response["server_inference_sec"]
                                if offload else ""
                            ),
                        }

                        writer.writerow(row)
                        log.flush()

            finally:
                if connection is not None:
                    connection.close()

    print("CLIENT COMPLETED", flush=True)


def run_topology(args):
    if os.geteuid() != 0:
        raise RuntimeError("Topology creation requires sudo")

    from mininet.net import Mininet
    from mininet.node import OVSSwitch
    from mininet.link import TCLink
    from mininet.log import setLogLevel

    setLogLevel("info")

    out = Path(args.out).resolve()
    owner = pwd.getpwnam(args.user)
    runuser = shutil.which("runuser")

    if runuser is None:
        raise RuntimeError("runuser is unavailable")

    suffix = uuid.uuid4().hex[:6]
    bridge = "nr" + suffix + "s"
    client_name = "nr" + suffix + "c"
    server_name = "nr" + suffix + "r"

    def command(argv):
        result = subprocess.run(
            argv, text=True, capture_output=True, timeout=20
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Command failed: {argv}\n"
                f"{result.stdout}\n{result.stderr}"
            )
        return result.stdout

    if subprocess.run(
        ["ovs-vsctl", "br-exists", bridge],
        capture_output=True,
    ).returncode == 0:
        raise RuntimeError(f"Refusing to reuse bridge: {bridge}")

    def user_command(role):
        return [
            runuser, "-u", args.user, "--",
            "env",
            f"PYTHONPATH={ROOT}",
            "PYTHONUNBUFFERED=1",
            "MPLBACKEND=Agg",
            str(PYTHON), "-u", str(SCRIPT),
            "--role", role,
            "--out", str(out),
        ]

    def save_owned(name, text):
        path = out / name
        path.write_text(text)
        os.chown(path, owner.pw_uid, owner.pw_gid)

    network = None
    processes = []
    log_handles = []

    try:
        network = Mininet(
            controller=None,
            switch=OVSSwitch,
            link=TCLink,
            build=False,
            autoSetMacs=True,
        )

        client = network.addHost(
            client_name, ip=CLIENT_IP + "/24"
        )
        server = network.addHost(
            server_name, ip=SERVER_IP + "/24"
        )
        switch = network.addSwitch(
            bridge,
            protocols="OpenFlow13",
            failMode="secure",
        )

        network.addLink(
            client, switch,
            port1=0, port2=1,
            bw=20, delay="2ms",
        )
        network.addLink(
            server, switch,
            port1=0, port2=2,
            bw=20, delay="2ms",
        )

        network.build()
        network.start()

        controllers = command([
            "ovs-vsctl", "get-controller", bridge
        ]).strip()
        if controllers:
            raise RuntimeError(
                f"Unexpected controller on isolated bridge: {controllers}"
            )

        # Explicit, static bidirectional baseline.
        # These are NOT QoS queue-selection rules.
        for rule in (
            "priority=10,in_port=1,actions=output:2",
            "priority=10,in_port=2,actions=output:1",
        ):
            command([
                "ovs-ofctl", "-O", "OpenFlow13",
                "add-flow", bridge, rule,
            ])

        save_owned("flows_before.txt", command([
            "ovs-ofctl", "-O", "OpenFlow13",
            "dump-flows", bridge,
        ]))

        ping = client.cmd(
            f"ping -c 2 -W 3 {SERVER_IP}"
        )
        print(ping, flush=True)
        save_owned("connectivity.txt", ping)

        if " 0% packet loss" not in ping:
            raise RuntimeError("Isolated network connectivity failed")

        save_owned(
            "network_configuration.json",
            json.dumps({
                "bridge": bridge,
                "controller": None,
                "forwarding": "Static OpenFlow 1.3 baseline",
                "client": CLIENT_IP,
                "server": SERVER_IP,
                "each_link_configured_mbps": 20,
                "each_link_configured_delay": "2ms",
                "warning": (
                    "Configured emulation settings are not measured "
                    "latency or bandwidth guarantees."
                ),
            }, indent=2) + "\n",
        )

        server_log = (out / "server_console.log").open("w")
        log_handles.append(server_log)
        os.chown(
            out / "server_console.log",
            owner.pw_uid, owner.pw_gid,
        )

        server_process = server.popen(
            user_command("server"),
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes.append(server_process)

        deadline = time.monotonic() + 180
        while not (out / "server_ready.json").is_file():
            if server_process.poll() is not None:
                raise RuntimeError(
                    "Server exited. Inspect server_console.log"
                )
            if time.monotonic() > deadline:
                raise TimeoutError(
                    "Server startup timeout. Inspect server_console.log"
                )
            time.sleep(0.5)

        client_log = (out / "client_console.log").open("w")
        log_handles.append(client_log)
        os.chown(
            out / "client_console.log",
            owner.pw_uid, owner.pw_gid,
        )

        client_process = client.popen(
            user_command("client"),
            stdout=client_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes.append(client_process)

        try:
            returncode = client_process.wait(timeout=900)
        except subprocess.TimeoutExpired:
            raise TimeoutError(
                "Client timeout. Inspect client_console.log"
            )

        if returncode != 0:
            raise RuntimeError(
                "Client failed. Inspect client_console.log"
            )

        save_owned("flows_after.txt", command([
            "ovs-ofctl", "-O", "OpenFlow13",
            "dump-flows", bridge,
        ]))
        save_owned("ports_after.txt", command([
            "ovs-ofctl", "-O", "OpenFlow13",
            "dump-ports", bridge,
        ]))

        with (out / "client_requests.csv").open() as f:
            rows = list(csv.DictReader(f))

        with (out / "server_requests.csv").open() as f:
            server_rows = list(csv.DictReader(f))

        if len(rows) != REQUESTS_PER_POLICY * len(POLICIES):
            raise RuntimeError("Incomplete client request log")

        expected = [
            row for row in rows
            if row["decision"] == "offload"
        ]

        def identity(row):
            return (
                row["run_id"],
                row["session_id"],
                row["request_id"],
            )

        actual = {identity(row): row for row in server_rows}

        if len(actual) != len(server_rows):
            raise RuntimeError("Duplicate server identities")
        if len(actual) != len(expected):
            raise RuntimeError("Client/server request-count mismatch")

        for row in expected:
            observed = actual.get(identity(row))
            if observed is None:
                raise RuntimeError("Missing server request")
            for key in (
                "sample_id", "image_sha256",
                "client_ip", "client_port",
                "server_ip", "server_port",
            ):
                if row[key] != observed[key]:
                    raise RuntimeError(
                        f"Cross-log association mismatch: {key}"
                    )
            if row["final_prediction"] != observed["prediction"]:
                raise RuntimeError("Prediction association mismatch")

        summary = {
            "status": "completed",
            "results_directory": str(out),
            "total_client_requests": len(rows),
            "matched_remote_requests": len(expected),
            "observed_identity_mismatches": 0,
            "policies": {},
            "limitations": [
                "Smoke test only; 20 requests per policy.",
                "Same workstation and shared physical GPU resources.",
                "Static OpenFlow forwarding, not Ryu inference registration.",
                "No queue differentiation, SLA enforcement, or adaptive routing.",
                "Fixed threshold; no feedback update.",
                "No Jetson energy or hardware measurements.",
                "Timing excludes original image acquisition.",
            ],
        }

        for policy in POLICIES:
            selected = [r for r in rows if r["policy"] == policy]
            summary["policies"][policy] = {
                "requests": len(selected),
                "offloaded": sum(
                    r["decision"] == "offload" for r in selected
                ),
                "correct": sum(int(r["correct"]) for r in selected),
                "mean_request_ms": 1000 * statistics.mean(
                    float(r["request_sec"]) for r in selected
                ),
            }

        save_owned(
            "summary.json",
            json.dumps(summary, indent=2) + "\n",
        )
        print(json.dumps(summary, indent=2), flush=True)

    finally:
        # Stop only worker process groups created by this script.
        for process in reversed(processes):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5)
                except ProcessLookupError:
                    pass
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=5)

        for handle in log_handles:
            handle.close()

        if network is not None:
            network.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--role", choices=("topology", "server", "client"),
        required=True,
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--user")
    args = parser.parse_args()

    if args.role == "topology":
        if not args.user:
            raise ValueError("--user is required for topology")
        run_topology(args)
    elif args.role == "server":
        run_server(args)
    else:
        run_client(args)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
