from pathlib import Path
from datetime import datetime
import csv
import hashlib
import importlib.util
import json
import subprocess
import time

ROOT = Path("/home/avi/NSSON")
SOURCE = ROOT / "revision_causal_20261102_110047_TfyZ9O/network_causal.py"
OUT = ROOT / (
    "revision_gpu_measured_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
)
OUT.mkdir(exist_ok=False)

GPU = "GPU-703505b2-17ec-6732-79e3-5124680b5d19"

active = subprocess.check_output([
    "nvidia-smi", f"--id={GPU}",
    "--query-compute-apps=pid", "--format=csv,noheader"
], text=True).strip()
if active:
    raise RuntimeError(f"GPU compute processes already active:\n{active}")

spec = importlib.util.spec_from_file_location("verified_runner", SOURCE)
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)

np, torch, device = api.tensor_environment()
if device.type != "cuda":
    raise RuntimeError("CUDA unavailable; stopped instead of switching hardware.")

import torchvision

cnn, cnn_transform = api.cnn_components(device)
snn, snn_transform, reset = api.snn_components(device)

dataset = torchvision.datasets.CIFAR10(
    root=str(ROOT / "data/cifar10"),
    train=False, download=False, transform=None
)
ids = np.random.default_rng(20261102).permutation(len(dataset))[:300].tolist()
samples = [(i, *dataset[i]) for i in ids]

models = {
    "cnn": (cnn, cnn_transform, None),
    "snn": (snn, snn_transform, reset),
}

with torch.no_grad():
    for model, transform, reset_fn in models.values():
        for _, image, _ in samples[:20]:
            api.infer(torch, device, model, transform, image, reset=reset_fn)

checkpoints = [
    ROOT / "models/cnn_baseline/checkpoints/resnet18_cifar10_best.pth",
    ROOT / "snasnet/savemodel/save_cifar10_bw.pth.tar",
]
metadata = {
    "scope": "Workstation GPU telemetry during real local inference; not Jetson, network-offload, or whole-device energy.",
    "source": str(SOURCE),
    "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "checkpoint_sha256": {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in checkpoints
    },
    "gpu_uuid": GPU,
    "torch_version": torch.__version__,
    "gpu_name": torch.cuda.get_device_name(),
    "sample_ids": ids,
    "batch_size": 1,
    "repetitions": 3,
    "minimum_workload_seconds": 30,
    "preblock_idle_seconds": 10,
    "requested_poll_interval_ms": 200,
    "timing_scope": "Reset, transform, transfer to GPU, forward pass, and prediction/confidence extraction; excludes dataset loading and CSV writing.",
    "energy_boundary": "Whole GPU telemetry over each workload block, including between-request overhead; not isolated per-request energy.",
}
(OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))

subprocess.run(
    ["nvidia-smi", "-q"],
    stdout=(OUT / "gpu_configuration.txt").open("w"),
    check=True
)

power_file = (OUT / "gpu_power.csv").open("w")
error_file = (OUT / "monitor_errors.txt").open("w")
monitor = subprocess.Popen([
    "nvidia-smi", f"--id={GPU}",
    "--query-gpu=timestamp,uuid,power.draw,temperature.gpu,utilization.gpu",
    "--format=csv", "--loop-ms=200"
], stdout=power_file, stderr=error_file)

try:
    with (OUT / "requests.csv").open("w", newline="") as request_file, \
         (OUT / "blocks.csv").open("w", newline="") as block_file:
        requests = csv.writer(request_file)
        blocks = csv.writer(block_file)

        requests.writerow([
            "repetition", "model", "sequence", "sample_id", "label",
            "prediction", "confidence_uncalibrated", "correct",
            "start_epoch_s", "end_epoch_s", "elapsed_s"
        ])
        blocks.writerow([
            "repetition", "model", "phase", "start_epoch_s",
            "end_epoch_s", "elapsed_s", "requests"
        ])

        time.sleep(2)
        if monitor.poll() is not None:
            raise RuntimeError("Power monitor exited; inspect monitor_errors.txt.")

        with torch.no_grad():
            for rep in (1, 2, 3):
                order = ("cnn", "snn") if rep % 2 else ("snn", "cnn")
                for name in order:
                    model, transform, reset_fn = models[name]

                    api.synchronize(torch, device)
                    idle_start = time.time()
                    idle_clock = time.perf_counter()
                    time.sleep(10)
                    blocks.writerow([
                        rep, name, "preblock_idle", idle_start, time.time(),
                        time.perf_counter() - idle_clock, 0
                    ])
                    block_file.flush()

                    observations = []
                    api.synchronize(torch, device)
                    block_start = time.time()
                    block_clock = time.perf_counter()
                    sequence = 0

                    while time.perf_counter() - block_clock < 30:
                        sample_id, image, label = samples[sequence % len(samples)]
                        api.synchronize(torch, device)
                        start_epoch = time.time()
                        start_clock = time.perf_counter()

                        prediction, confidence = api.infer(
                            torch, device, model, transform, image,
                            reset=reset_fn
                        )

                        elapsed = time.perf_counter() - start_clock
                        end_epoch = time.time()
                        observations.append([
                            rep, name, sequence, sample_id, label,
                            prediction, confidence, int(prediction == label),
                            start_epoch, end_epoch, elapsed
                        ])
                        sequence += 1

                    api.synchronize(torch, device)
                    block_end = time.time()
                    duration = time.perf_counter() - block_clock

                    blocks.writerow([
                        rep, name, "workload", block_start,
                        block_end, duration, sequence
                    ])
                    requests.writerows(observations)
                    request_file.flush()
                    block_file.flush()

                    print(
                        f"repetition={rep} model={name} "
                        f"requests={sequence} duration={duration:.3f}s",
                        flush=True
                    )

        time.sleep(2)
finally:
    if monitor.poll() is None:
        monitor.terminate()
    monitor.wait(timeout=10)
    power_file.close()
    error_file.close()

print(f"\nSaved real observations to: {OUT}")
print("Files: gpu_power.csv, requests.csv, blocks.csv, metadata.json")
errors = (OUT / "monitor_errors.txt").read_text().strip()
if errors:
    print("Monitor messages:", errors)
