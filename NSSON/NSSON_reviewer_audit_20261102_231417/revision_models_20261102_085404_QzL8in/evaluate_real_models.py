import csv
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(os.environ["NSSON_ROOT"]).resolve()
OUT = Path(os.environ["NSSON_RESULTS"]).resolve()
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from models.cifar10_cnn import (
    load_cnn_checkpoint,
    get_cifar10_test_loader,
)

# The legacy SNASNet configuration uses argparse.parse_args().
# Prevent it from consuming unrelated command-line arguments.
saved_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
try:
    import snn.snn_snasnet.eval_snasnet_cifar10 as snn_api
finally:
    sys.argv = saved_argv

from spikingjelly.clock_driven import functional

SEED = 20261102
BATCH_SIZE = 32
TIMING_REQUESTS = 300
REPETITIONS = 3
WARMUP_REQUESTS = 10

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = False


def save_json(name, value):
    (OUT / name).write_text(
        json.dumps(value, indent=2, default=str) + "\n"
    )


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_model(model, name):
    if name == "snn":
        functional.reset_net(model)
        # Also clear SNASNet-specific cell recurrence.
        model.neuron_init()


def validate_scores(scores, batch_size):
    if not isinstance(scores, torch.Tensor):
        raise TypeError(
            f"Expected Tensor, received {type(scores)}"
        )
    if tuple(scores.shape) != (batch_size, 10):
        raise ValueError(
            f"Expected ({batch_size}, 10), got {tuple(scores.shape)}"
        )
    if not torch.isfinite(scores).all().item():
        raise ValueError("Model returned non-finite scores")


def predict(model, batch, name):
    reset_model(model, name)
    scores = model(batch)
    validate_scores(scores, batch.size(0))
    confidence, prediction = torch.softmax(
        scores.float(), dim=1
    ).max(dim=1)
    return (
        prediction.detach().cpu().tolist(),
        confidence.detach().cpu().tolist(),
    )


class PairedDataset(Dataset):
    def __init__(self, raw_dataset, cnn_transform, snn_transform):
        self.raw = raw_dataset
        self.cnn_transform = cnn_transform
        self.snn_transform = snn_transform

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, index):
        image, label = self.raw[index]
        return (
            index,
            self.cnn_transform(image.copy()),
            self.snn_transform(image.copy()),
            int(label),
        )


def main():
    start_wall = time.time()

    expected_snasnet = (ROOT / "snasnet").resolve()
    for name in ("config", "model_snn", "utils"):
        module = sys.modules.get(name)
        location = getattr(module, "__file__", None)
        if location is None:
            raise RuntimeError(f"Cannot verify module: {name}")
        resolved = Path(location).resolve()
        if not resolved.is_relative_to(expected_snasnet):
            raise RuntimeError(
                f"Module shadowing: {name} loaded from {resolved}"
            )

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    print(f"Execution device: {device}", flush=True)

    cnn_checkpoint = (
        ROOT / "models/cnn_baseline/checkpoints/"
        "resnet18_cifar10_best.pth"
    )
    snn_checkpoint = (
        ROOT / "snasnet/savemodel/save_cifar10_bw.pth.tar"
    )

    print("Loading CNN checkpoint...", flush=True)
    cnn, _ = load_cnn_checkpoint(
        ckpt_path=str(cnn_checkpoint),
        device=device,
    )

    print("Loading SNN checkpoint...", flush=True)
    saved = sys.argv[:]
    sys.argv = [sys.argv[0]]
    try:
        snn, snn_args, _ = snn_api.load_snn_checkpoint(
            device=device
        )
    finally:
        sys.argv = saved

    cnn.eval()
    snn.eval()

    # Reuse the actual transforms specified by each model loader.
    cnn_transform = get_cifar10_test_loader(
        data_root=str(ROOT / "data/cifar10"),
        batch_size=1,
    ).dataset.transform

    _, snn_transform = snn_api.data_transforms(snn_args)

    # Both transforms receive the same original image.
    raw = torchvision.datasets.CIFAR10(
        root=str(ROOT / "data/cifar10"),
        train=False,
        download=False,
        transform=None,
    )

    if len(raw) != 10000:
        raise RuntimeError(
            f"Expected 10000 test images, found {len(raw)}"
        )

    paired = PairedDataset(
        raw, cnn_transform, snn_transform
    )

    source_files = [
        ROOT / "models/cifar10_cnn.py",
        ROOT / "snn/snn_snasnet/eval_snasnet_cifar10.py",
        ROOT / "snasnet/config.py",
        ROOT / "snasnet/model_snn.py",
        ROOT / "snasnet/utils.py",
        Path(__file__).resolve(),
    ]

    metadata = {
        "seed": SEED,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torch_cuda_build": torch.version.cuda,
        "device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda" else None
        ),
        "accuracy_batch_size": BATCH_SIZE,
        "accuracy_drop_last": False,
        "timing_batch_size": 1,
        "timing_requests_per_repetition": TIMING_REQUESTS,
        "timing_repetitions": REPETITIONS,
        "timing_warmups_per_model_per_repetition": WARMUP_REQUESTS,
        "cnn_transform": repr(cnn_transform),
        "snn_transform": repr(snn_transform),
        "snn_args": vars(snn_args),
        "snn_reset": (
            "functional.reset_net(model) and model.neuron_init() "
            "before each independent forward"
        ),
        "checkpoint_hashes": {
            str(cnn_checkpoint): checksum(cnn_checkpoint),
            str(snn_checkpoint): checksum(snn_checkpoint),
        },
        "source_hashes": {
            str(path): checksum(path) for path in source_files
        },
        "raw_image_array_sha256": hashlib.sha256(
            raw.data.tobytes()
        ).hexdigest(),
        "raw_targets_sha256": hashlib.sha256(
            np.asarray(raw.targets, dtype=np.int64).tobytes()
        ).hexdigest(),
        "scope": (
            "Real model evaluation on the current host. "
            "Not Jetson hardware measurement, network offloading, "
            "SDN evaluation, or an energy experiment."
        ),
        "confidence": (
            "Maximum softmax of model class scores; uncalibrated. "
            "SNN scores are temporally averaged outputs returned by SNASNet."
        ),
    }
    save_json("metadata.json", metadata)

    # Test A -> B -> A with a full reset before each independent input.
    print("\nChecking SNN reset consistency...", flush=True)
    _, _, image_a, _ = paired[0]
    _, _, image_b, _ = paired[1]

    with torch.no_grad():
        reset_model(snn, "snn")
        output_a1 = snn(
            image_a.unsqueeze(0).to(device)
        ).detach().cpu().clone()

        reset_model(snn, "snn")
        _ = snn(image_b.unsqueeze(0).to(device))

        reset_model(snn, "snn")
        output_a2 = snn(
            image_a.unsqueeze(0).to(device)
        ).detach().cpu().clone()

    validate_scores(output_a1, 1)
    validate_scores(output_a2, 1)

    max_difference = float(
        (output_a1 - output_a2).abs().max().item()
    )
    reset_passed = bool(torch.allclose(
        output_a1, output_a2, rtol=1e-4, atol=1e-5
    ))

    reset_result = {
        "sequence": "A, B, A; reset before every forward",
        "rtol": 1e-4,
        "atol": 1e-5,
        "maximum_absolute_score_difference": max_difference,
        "passed": reset_passed,
        "limitation": (
            "A finite consistency check, not proof of all possible "
            "state-isolation cases."
        ),
    }
    save_json("snn_reset_test.json", reset_result)
    print(json.dumps(reset_result, indent=2), flush=True)

    if not reset_passed:
        raise RuntimeError(
            "SNN reset consistency check failed. "
            "Stop and inspect state handling; do not continue benchmarking."
        )

    # Full paired accuracy evaluation.
    print("\nEvaluating both models on all 10000 images...", flush=True)
    loader = DataLoader(
        paired,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    prediction_fields = [
        "sample_id", "label",
        "cnn_prediction", "cnn_correct",
        "cnn_confidence_uncalibrated",
        "snn_prediction", "snn_correct",
        "snn_confidence_uncalibrated",
    ]

    all_predictions = []

    with (OUT / "paired_predictions.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=prediction_fields)
        writer.writeheader()

        with torch.no_grad():
            for batch_number, (
                ids, cnn_images, snn_images, labels
            ) in enumerate(loader, 1):

                cnn_pred, cnn_conf = predict(
                    cnn, cnn_images.to(device), "cnn"
                )
                snn_pred, snn_conf = predict(
                    snn, snn_images.to(device), "snn"
                )

                for j in range(len(labels)):
                    label = int(labels[j])
                    row = {
                        "sample_id": int(ids[j]),
                        "label": label,
                        "cnn_prediction": int(cnn_pred[j]),
                        "cnn_correct": int(cnn_pred[j] == label),
                        "cnn_confidence_uncalibrated": float(cnn_conf[j]),
                        "snn_prediction": int(snn_pred[j]),
                        "snn_correct": int(snn_pred[j] == label),
                        "snn_confidence_uncalibrated": float(snn_conf[j]),
                    }
                    writer.writerow(row)
                    all_predictions.append(row)

                f.flush()
                if batch_number % 20 == 0:
                    print(
                        f"  Accuracy evaluation: "
                        f"{len(all_predictions)}/10000",
                        flush=True,
                    )

    if len(all_predictions) != 10000:
        raise RuntimeError("Incomplete paired evaluation")

    # Model-only timing, separated from batched accuracy.
    rng = np.random.default_rng(SEED)
    timing_ids = rng.permutation(len(paired))[
        :TIMING_REQUESTS
    ].tolist()

    save_json("timing_sample_ids.json", timing_ids)

    # Preprocessing is deliberately outside the measured timing interval.
    cached = {}
    for sample_id in timing_ids:
        _, cnn_image, snn_image, label = paired[sample_id]
        cached[sample_id] = {
            "cnn": cnn_image.unsqueeze(0),
            "snn": snn_image.unsqueeze(0),
            "label": label,
        }

    models = {"cnn": cnn, "snn": snn}

    def timed_request(name, sample_id):
        model = models[name]
        cpu_batch = cached[sample_id][name]

        synchronize(device)
        t0 = time.perf_counter_ns()

        reset_model(model, name)
        synchronize(device)
        t1 = time.perf_counter_ns()

        batch = cpu_batch.to(device)
        synchronize(device)
        t2 = time.perf_counter_ns()

        scores = model(batch)
        synchronize(device)
        t3 = time.perf_counter_ns()

        confidence, prediction = torch.softmax(
            scores.float(), dim=1
        ).max(dim=1)

        confidence = float(confidence.item())
        prediction = int(prediction.item())
        synchronize(device)
        t4 = time.perf_counter_ns()

        # Validation is outside the measured interval.
        validate_scores(scores, 1)

        return {
            "sample_id": sample_id,
            "label": int(cached[sample_id]["label"]),
            "model": name,
            "prediction": prediction,
            "correct": int(
                prediction == cached[sample_id]["label"]
            ),
            "confidence_uncalibrated": confidence,
            "reset_sec": (t1 - t0) / 1e9,
            "input_transfer_sec": (t2 - t1) / 1e9,
            "forward_sec": (t3 - t2) / 1e9,
            "score_to_host_sec": (t4 - t3) / 1e9,
            "tensor_to_result_sec": (t4 - t0) / 1e9,
            "batch_size": 1,
            "device": str(device),
        }

    timing_rows = []
    timing_fields = [
        "repetition", "execution_order",
        "sample_id", "label", "model",
        "prediction", "correct", "confidence_uncalibrated",
        "reset_sec", "input_transfer_sec", "forward_sec",
        "score_to_host_sec", "tensor_to_result_sec",
        "batch_size", "device",
    ]

    print("\nStarting repeated batch-size-one timing...", flush=True)

    with (OUT / "request_timings.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=timing_fields)
        writer.writeheader()

        with torch.no_grad():
            for repetition in range(1, REPETITIONS + 1):
                warm_order = (
                    ("cnn", "snn") if repetition % 2
                    else ("snn", "cnn")
                )

                for name in warm_order:
                    for sample_id in timing_ids[:WARMUP_REQUESTS]:
                        timed_request(name, sample_id)

                order = timing_ids.copy()
                random.Random(SEED + repetition).shuffle(order)

                for position, sample_id in enumerate(order):
                    model_order = (
                        ("cnn", "snn")
                        if (position + repetition) % 2
                        else ("snn", "cnn")
                    )

                    for execution_order, name in enumerate(model_order):
                        row = timed_request(name, sample_id)
                        row.update({
                            "repetition": repetition,
                            "execution_order": execution_order,
                        })
                        writer.writerow(row)
                        timing_rows.append(row)

                    if (position + 1) % 50 == 0:
                        f.flush()
                        print(
                            f"  Timing repetition {repetition}/{REPETITIONS}: "
                            f"{position + 1}/{TIMING_REQUESTS} paired requests",
                            flush=True,
                        )
                f.flush()

    timing_summary = []
    for repetition in range(1, REPETITIONS + 1):
        for name in ("cnn", "snn"):
            rows = [
                r for r in timing_rows
                if r["repetition"] == repetition
                and r["model"] == name
            ]
            request_ms = np.asarray([
                r["tensor_to_result_sec"] * 1000 for r in rows
            ])
            forward_ms = np.asarray([
                r["forward_sec"] * 1000 for r in rows
            ])

            timing_summary.append({
                "repetition": repetition,
                "model": name,
                "requests": len(rows),
                "mean_forward_ms": float(forward_ms.mean()),
                "mean_tensor_to_result_ms": float(request_ms.mean()),
                "median_tensor_to_result_ms": float(
                    np.median(request_ms)
                ),
                "p95_tensor_to_result_ms": float(
                    np.percentile(request_ms, 95)
                ),
                "p99_tensor_to_result_ms": float(
                    np.percentile(request_ms, 99)
                ),
            })

    with (OUT / "timing_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(timing_summary[0])
        )
        writer.writeheader()
        writer.writerows(timing_summary)

    # Offline sweep only: no claimed live offload latency or energy.
    sweep_rows = []
    previous_count = -1

    for integer_threshold in range(101):
        theta = integer_threshold / 100.0
        offloaded = 0
        final_correct = 0

        for row in all_predictions:
            choose_cnn = (
                row["snn_confidence_uncalibrated"] < theta
            )
            offloaded += int(choose_cnn)
            final_correct += (
                row["cnn_correct"] if choose_cnn
                else row["snn_correct"]
            )

        if offloaded < previous_count:
            raise AssertionError(
                "Offload count decreased as threshold increased"
            )
        previous_count = offloaded

        sweep_rows.append({
            "theta": theta,
            "samples": len(all_predictions),
            "offload_count": offloaded,
            "offload_ratio": offloaded / len(all_predictions),
            "offline_selected_prediction_accuracy": (
                final_correct / len(all_predictions)
            ),
            "analysis_type": "offline_fixed_prediction_sweep",
        })

    with (OUT / "offline_threshold_sweep.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(sweep_rows[0])
        )
        writer.writeheader()
        writer.writerows(sweep_rows)

    cnn_accuracy = statistics.mean(
        r["cnn_correct"] for r in all_predictions
    )
    snn_accuracy = statistics.mean(
        r["snn_correct"] for r in all_predictions
    )

    summary = {
        "status": "completed",
        "results_directory": str(OUT),
        "test_samples": len(all_predictions),
        "cnn_accuracy_percent": 100.0 * cnn_accuracy,
        "snn_accuracy_percent": 100.0 * snn_accuracy,
        "snn_timesteps": int(snn_args.timestep),
        "snn_reset_consistency_passed": reset_passed,
        "timing_summary": timing_summary,
        "offline_threshold_monotonicity_passed": True,
        "elapsed_wall_seconds": time.time() - start_wall,
        "limitations": [
            "Host RTX/CPU results, not Jetson Nano measurements.",
            "Timing excludes image loading and preprocessing.",
            "Timing repetitions share one process; independent-process runs remain necessary for final uncertainty estimates.",
            "Maximum-softmax confidence is not calibrated accuracy.",
            "Threshold sweep is offline, not a live feedback or network experiment.",
            "No energy, QoS, SLA, slicing, or SDN improvement is established.",
            "Checkpoint training/preprocessing provenance must still be documented.",
        ],
    }
    save_json("summary.json", summary)

    print("\n===== COMPLETED =====", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        details = traceback.format_exc()
        (OUT / "FAILURE.txt").write_text(details)
        print(details, file=sys.stderr, flush=True)
        sys.exit(1)
