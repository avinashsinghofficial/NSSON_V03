import os
import torch
import torchvision
import config
from model_snn import SNASNet, find_best_neuroncell
from utils import data_transforms

def build_model_and_load(device, args):
    # dataset transforms (for consistency with training)
    train_transform, valid_transform = data_transforms(args)

    if args.dataset == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(
            root=os.path.join(args.data_dir, 'cifar10'),
            train=True,
            download=True,
            transform=train_transform
        )
    elif args.dataset == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(
            root=os.path.join(args.data_dir, 'cifar100'),
            train=True,
            download=True,
            transform=train_transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # neuron cell
    if args.cnt_mat is None:
        best_neuroncell = find_best_neuroncell(args, dataset)
    else:
        int_list = []
        for line in args.cnt_mat:
            row_list = []
            for element in line:
                row_list.append(int(element))
            int_list.append(row_list)
        best_neuroncell = torch.tensor(int_list, dtype=torch.float32)

    model = SNASNet(args, best_neuroncell).to(device)

    if args.savemodel_pth is not None and os.path.isfile(args.savemodel_pth):
        print(f"[INFO] Loading pretrained weights from {args.savemodel_pth}")
        checkpoint = torch.load(args.savemodel_pth, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise FileNotFoundError(f"Pretrained checkpoint not found at {args.savemodel_pth}")

    model.eval()
    return model

def main():
    args = config.get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    if args.dataset != 'cifar10':
        raise ValueError("Set --dataset 'cifar10' for export.")

    model = build_model_and_load(device, args)

    print("[INFO] Scripting TorchScript model...")
    with torch.no_grad():
        scripted = torch.jit.script(model)

    out_path = os.path.join(os.path.dirname(__file__), "snasnet_cifar10_bw.ts")
    scripted.save(out_path)
    print(f"[INFO] Saved TorchScript model to {out_path}")

if __name__ == "__main__":
    main()
