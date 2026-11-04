import torch
import torchvision
import torchvision.transforms as transforms

from snn.snn_snasnet.train_snasnet_cifar10_finetune import build_model


def build_snasnet_model(memory_budget=None):
    model = build_model(memory_budget=memory_budget)
    return model


def load_snasnet_checkpoint(path="snn/snn_snasnet/checkpoints/snasnet_cifar10_best.pth"):
    model = build_snasnet_model()
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def get_cifar10_test_loader(data_root="data/cifar10", batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    test_set = torchvision.datasets.CIFAR10(root=data_root,
                                            train=False,
                                            download=False,
                                            transform=transform)
    loader = torch.utils.data.DataLoader(test_set,
                                         batch_size=batch_size,
                                         shuffle=False)
    return loader


def evaluate_snasnet_over_T(model, loader, T=300, device="cuda"):
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images, T=T)  # assuming your snasnet supports time steps T
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total
