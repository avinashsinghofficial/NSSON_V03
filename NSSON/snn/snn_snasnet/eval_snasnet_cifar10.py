import os
import sys
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Project root and SNASNet repo paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SNASNET_ROOT = os.path.expanduser(os.path.join(PROJECT_ROOT, "snasnet"))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SNASNET_ROOT not in sys.path:
    sys.path.insert(0, SNASNET_ROOT)

from sim.nsson_utils.run_utils import make_run_dir, csv_logger  # you already have this
import config
from model_snn import SNASNet
from utils import data_transforms


def build_snn_model():
    """Build SNASNet SNN model for CIFAR-10."""
    args = config.get_args()
    args.dataset = "cifar10"
    args.celltype = "backward"
    args.cnt_mat = [
        ["0", "3", "0", "3"],
        ["0", "0", "3", "0"],
        ["2", "0", "0", "2"],
        ["0", "2", "0", "0"],
    ]
    args.second_avgpooling = 4
    args.savemodel_pth = os.path.join(SNASNET_ROOT, "savemodel", "save_cifar10_bw.pth.tar")
    args.data_dir = os.path.join(SNASNET_ROOT, "dataset")

    int_list = [[int(e) for e in row] for row in args.cnt_mat]
    best_neuroncell = torch.tensor(int_list, dtype=torch.float32)

    model = SNASNet(args, best_neuroncell)
    return model, args


def load_snn_checkpoint(device=None):
    """Load pretrained SNASNet weights and return model ready for eval."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, args = build_snn_model()
    model = model.to(device)

    weight_path = args.savemodel_pth
    assert os.path.isfile(weight_path), f"Missing pretrained weights: {weight_path}"

    ckpt = torch.load(weight_path, map_location=device)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    return model, args, device


def get_snn_test_loader(args, batch_size=64):
    """Return CIFAR-10 test loader with SNASNet transforms."""
    train_transform, valid_transform = data_transforms(args)
    testset = torchvision.datasets.CIFAR10(
        root=os.path.join(args.data_dir, "cifar10"),
        train=False,
        download=True,
        transform=valid_transform,
    )
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    return test_loader


def evaluate_snn(model, loader, device, log_dir="~/NSSON/shared/snn_runs"):
    """Evaluate SNASNet: accuracy, loss, avg latency per batch."""
    criterion = nn.CrossEntropyLoss()

    run_dir = make_run_dir(log_dir, prefix="snn_snasnet_eval")
    metrics_log = csv_logger(
        os.path.join(run_dir, "metrics.csv"),
        fieldnames=["dataset", "acc", "loss", "avg_latency_ms", "total_samples"],
    )

    total_loss = 0.0
    correct = 0
    total = 0
    latencies = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            t0 = time.time()
            outputs = model(inputs)
            t1 = time.time()

            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

            latencies.append((t1 - t0) * 1000.0)  # ms per batch

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0

    metrics_log(
        {
            "dataset": "cifar10",
            "acc": f"{acc:.6f}",
            "loss": f"{avg_loss:.6f}",
            "avg_latency_ms": f"{avg_latency_ms:.6f}",
            "total_samples": total,
        }
    )

    return {
        "accuracy": acc,
        "loss": avg_loss,
        "avg_latency_ms": avg_latency_ms,
        "total_samples": total,
        "run_dir": run_dir,
    }
