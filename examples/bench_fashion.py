import struct
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def load_idx_images(path: Path) -> torch.Tensor:
    data = path.read_bytes()
    _magic, n, rows, cols = struct.unpack(">IIII", data[:16])
    pixels = torch.frombuffer(bytearray(data[16:]), dtype=torch.uint8)
    return pixels.view(n, rows * cols).float().div_(255.0)


def load_idx_labels(path: Path) -> torch.Tensor:
    data = path.read_bytes()
    _magic, n = struct.unpack(">II", data[:8])
    return torch.frombuffer(bytearray(data[8:]), dtype=torch.uint8).long()[:n]


def metrics(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    samples = 0
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(features)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            samples += labels.size(0)
    return total_loss / samples, 100.0 * correct / samples


def main() -> int:
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    root = Path(sys.argv[2] if len(sys.argv) > 2 else "data/fashion-mnist")
    if not torch.backends.mps.is_available():
        raise SystemExit("bench needs Apple MPS")
    device = torch.device("mps")
    train_x = load_idx_images(root / "train-images-idx3-ubyte")
    train_y = load_idx_labels(root / "train-labels-idx1-ubyte")
    test_x = load_idx_images(root / "t10k-images-idx3-ubyte")
    test_y = load_idx_labels(root / "t10k-labels-idx1-ubyte")
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=512,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(test_x, test_y),
        batch_size=512,
        shuffle=False,
    )
    model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10)).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.08, momentum=0.0)
    loss_fn = nn.CrossEntropyLoss()
    print("framework: pytorch")
    print(f"GPU: {torch.mps.device_count()} MPS")
    print(f"samples: {len(train_x)}")
    print(f"epochs: {epochs}")
    print("batch_size: 512")
    train_seconds = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        started = time.perf_counter()
        total_loss = 0.0
        correct = 0
        samples = 0
        for features, labels in train_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(features)
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            samples += labels.size(0)
        torch.mps.synchronize()
        seconds = time.perf_counter() - started
        train_seconds += seconds
        loss = total_loss / samples
        accuracy = 100.0 * correct / samples
        print(
            f"epoch {epoch}/{epochs}: loss={loss:.4f}, accuracy={accuracy:.2f}% ({seconds:.3f}s)"
        )
    test_loss, test_accuracy = metrics(model, test_loader, loss_fn, device)
    print(f"test: loss={test_loss:.4f}, accuracy={test_accuracy:.2f}%")
    print(f"train_seconds: {train_seconds:.4f}")
    print(f"samples_per_second: {len(train_x) * epochs / train_seconds:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
