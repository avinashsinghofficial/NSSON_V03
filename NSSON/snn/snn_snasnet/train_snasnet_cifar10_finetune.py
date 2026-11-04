import os
import sys
import time
import torch
import torch.nn as nn
import torchvision

# Add project root (~/NSSON) to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sim.nsson_utils.run_utils import make_run_dir, csv_logger
from spikingjelly.clock_driven.functional import reset_net

# Path to the SNASNet repo (where search_snn.py, config.py, model_snn.py live)
SNASNET_ROOT = os.path.expanduser("~/NSSON/snasnet")
if SNASNET_ROOT not in sys.path:
    # Insert at front so this utils.py is found before site-packages 'utils'
    sys.path.insert(0, SNASNET_ROOT)

import config
from model_snn import SNASNet
from utils import data_transforms  # this now comes from ~/NSSON/snasnet/utils.py


def get_loaders(args, batch_size):
    train_transform, valid_transform = data_transforms(args)

    trainset = torchvision.datasets.CIFAR10(
        root=os.path.join(args.data_dir, "cifar10"),
        train=True,
        download=True,
        transform=train_transform,
    )
    testset = torchvision.datasets.CIFAR10(
        root=os.path.join(args.data_dir, "cifar10"),
        train=False,
        download=True,
        transform=valid_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, test_loader


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

        # Reset spiking states between batches
        reset_net(model)

    return running_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

            reset_net(model)

    return running_loss / total, correct / total


def main():
    # Load default SNASNet args and override key settings
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
    args.data_dir = os.path.join(SNASNET_ROOT, "dataset")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Per-run directory under shared/snn_runs
    run_root = "~/NSSON/shared/snn_runs"
    run_dir = make_run_dir(run_root, prefix="snn_snasnet_finetune")
    print("SNN fine-tune run dir:", run_dir)

    train_log = csv_logger(
        os.path.join(run_dir, "train_log.csv"),
        fieldnames=["epoch", "train_loss", "train_acc"],
    )
    test_log = csv_logger(
        os.path.join(run_dir, "test_log.csv"),
        fieldnames=["epoch", "test_loss", "test_acc"],
    )

    # Build neuron cell tensor from cnt_mat
    int_list = []
    for row in args.cnt_mat:
        int_list.append([int(e) for e in row])
    best_neuroncell = torch.tensor(int_list, dtype=torch.float32)
    print("best_neuroncell:")
    print(best_neuroncell)

    # Dataloaders
    train_loader, test_loader = get_loaders(args, batch_size=16)

    # Build model
    model = SNASNet(args, best_neuroncell).to(device)

    # Optionally load pretrained CIFAR-10 weights as starting point
    weight_path = os.path.join(SNASNET_ROOT, "savemodel", "save_cifar10_bw.pth.tar")
    if os.path.isfile(weight_path):
        print(f"Loading pretrained weights from {weight_path}")
        ckpt = torch.load(weight_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
    else:
        print(f"WARNING: pretrained weights not found at {weight_path}, training from scratch")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,          # smaller LR for fine-tuning
        momentum=0.9,
        weight_decay=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[10, 15],
        gamma=0.1,
    )

    num_epochs = 1  # adjust if you want longer fine-tuning

    start_time = time.time()
    best_acc = 0.0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = eval_epoch(
            model, test_loader, criterion, device
        )
        scheduler.step()

        print(
            f"Epoch {epoch+1:03d}/{num_epochs} "
            f"Train loss {train_loss:.4f} acc {train_acc*100:.2f}% | "
            f"Test loss {test_loss:.4f} acc {test_acc*100:.2f}%"
        )

        train_log(
            {
                "epoch": epoch + 1,
                "train_loss": f"{train_loss:.6f}",
                "train_acc": f"{train_acc:.6f}",
            }
        )
        test_log(
            {
                "epoch": epoch + 1,
                "test_loss": f"{test_loss:.6f}",
                "test_acc": f"{test_acc:.6f}",
            }
        )

        if test_acc > best_acc:
            best_acc = test_acc
            ckpt_dir = os.path.join(SNASNET_ROOT, "savemodel")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, "save_cifar10_bw_finetuned.pth.tar")
            torch.save(
                {"state_dict": model.state_dict(), "acc": best_acc},
                ckpt_path,
            )

    elapsed = time.time() - start_time
    print(f"Best fine-tuned test accuracy: {best_acc*100:.2f}%")
    print(f"Total fine-tuning time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
