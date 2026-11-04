import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

NSSON_ROOT = "/home/avi/NSSON_V02"
OUT_DIR = os.path.join(NSSON_ROOT, "logs", "cnnruns")
os.makedirs(OUT_DIR, exist_ok=True)
HIST_PATH = os.path.join(OUT_DIR, "run15_history.csv")

EPOCHS = 30
BATCH_SIZE = 128
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SimpleResNetCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = torchvision.models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def get_dataloaders():
    transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root=os.path.join(NSSON_ROOT, "datasets"),
                                            train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=os.path.join(NSSON_ROOT, "datasets"),
                                           train=False, download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    return train_loader, test_loader

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            _, preds = logits.max(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total

def main():
    train_loader, test_loader = get_dataloaders()
    model = SimpleResNetCNN(num_classes=10).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    history_rows = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            _, preds = logits.max(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        train_loss = running_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate(model, test_loader, criterion)

        print(f"[CNN] Epoch {epoch}/{EPOCHS} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        history_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

    torch.save(model.state_dict(), os.path.join(NSSON_ROOT, "models", "cnn_run15.pth"))
    import pandas as pd
    pd.DataFrame(history_rows).to_csv(HIST_PATH, index=False)
    print("CNN history written to", HIST_PATH)

if __name__ == "__main__":
    main()
