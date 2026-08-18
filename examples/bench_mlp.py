import struct
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parents[1]

DATASETS = {
    "fashion": "fashion",
    "fashion-wide": "fashion",
    "fashion-deep": "fashion",
    "fashion-xl": "fashion",
    "mnist": "mnist",
    "mnist-deep": "mnist",
    "mnist-xl": "mnist",
    "kmnist": "kmnist",
    "cifar10": "cifar10",
    "cifar-wide": "cifar10",
    "cifar-deep": "cifar10",
    "cifar-bottleneck": "cifar10",
    "cifar100": "cifar100",
    "cifar100-wide": "cifar100",
    "cifar100-deep": "cifar100",
}


def ensure_data(name: str) -> None:
    dataset = DATASETS.get(name)
    if dataset is None:
        raise SystemExit(f"unknown workload {name}")
    subprocess.run(
        [sys.executable, str(REPO / "examples" / "prepare_bench_data.py"), dataset],
        check=True,
    )


def load_idx_images(path: Path) -> torch.Tensor:
    data = path.read_bytes()
    _magic, n, rows, cols = struct.unpack(">IIII", data[:16])
    pixels = torch.frombuffer(bytearray(data[16:]), dtype=torch.uint8)
    return pixels.view(n, rows * cols).float().div_(255.0)


def load_idx_labels(path: Path) -> torch.Tensor:
    data = path.read_bytes()
    _magic, n = struct.unpack(">II", data[:8])
    return torch.frombuffer(bytearray(data[8:]), dtype=torch.uint8).long()[:n]


def load_packed_images(path: Path, width: int) -> torch.Tensor:
    pixels = torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.uint8)
    return pixels.view(-1, width).float().div_(255.0)


def load_packed_labels(path: Path) -> torch.Tensor:
    return torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.uint8).long()


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


def load_idx_pair(root: Path):
    return (
        load_idx_images(root / "train-images-idx3-ubyte"),
        load_idx_labels(root / "train-labels-idx1-ubyte"),
        load_idx_images(root / "t10k-images-idx3-ubyte"),
        load_idx_labels(root / "t10k-labels-idx1-ubyte"),
    )


def load_workload(name: str):
    if name in ("fashion", "fashion-wide", "fashion-deep", "fashion-xl"):
        if name == "fashion-wide":
            hidden = [2048]
        elif name == "fashion-deep":
            hidden = [256, 256, 256]
        elif name == "fashion-xl":
            hidden = [4096]
        else:
            hidden = [256]
        return (*load_idx_pair(REPO / "data" / "fashion-mnist"), hidden, 10)
    if name in ("mnist", "mnist-deep", "mnist-xl"):
        if name == "mnist-deep":
            hidden = [256, 256, 256]
        elif name == "mnist-xl":
            hidden = [8192]
        else:
            hidden = [256]
        return (*load_idx_pair(REPO / "data" / "mnist"), hidden, 10)
    if name == "kmnist":
        return (*load_idx_pair(REPO / "data" / "kmnist"), [256], 10)
    if name in ("cifar10", "cifar-wide", "cifar-deep", "cifar-bottleneck"):
        root = REPO / "data" / "cifar-10"
        if name == "cifar-wide":
            hidden = [1024]
        elif name == "cifar-deep":
            hidden = [512, 512, 512]
        elif name == "cifar-bottleneck":
            hidden = [1024, 256, 1024]
        else:
            hidden = [512]
        return (
            load_packed_images(root / "train-images.bin", 3072),
            load_packed_labels(root / "train-labels.bin"),
            load_packed_images(root / "test-images.bin", 3072),
            load_packed_labels(root / "test-labels.bin"),
            hidden,
            10,
        )
    if name in ("cifar100", "cifar100-wide", "cifar100-deep"):
        root = REPO / "data" / "cifar-100"
        if name == "cifar100-wide":
            hidden = [2048]
        elif name == "cifar100-deep":
            hidden = [512, 512, 512]
        else:
            hidden = [512]
        return (
            load_packed_images(root / "train-images.bin", 3072),
            load_packed_labels(root / "train-labels.bin"),
            load_packed_images(root / "test-images.bin", 3072),
            load_packed_labels(root / "test-labels.bin"),
            hidden,
            100,
        )
    raise SystemExit(f"unknown workload {name}")


def make_model(features: int, hidden: list[int], classes: int) -> nn.Module:
    layers: list[nn.Module] = []
    width = features
    for units in hidden:
        layers.append(nn.Linear(width, units))
        layers.append(nn.ReLU())
        width = units
    layers.append(nn.Linear(width, classes))
    return nn.Sequential(*layers)


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "fashion"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    if not torch.backends.mps.is_available():
        raise SystemExit("bench needs Apple MPS")
    ensure_data(name)
    device = torch.device("mps")
    train_x, train_y, test_x, test_y, hidden, classes = load_workload(name)
    features = train_x.shape[1]
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
    model = make_model(features, hidden, classes).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.08, momentum=0.0)
    loss_fn = nn.CrossEntropyLoss()
    print("framework: pytorch")
    print(f"workload: {name}")
    print(f"GPU: {torch.mps.device_count()} MPS")
    print(f"samples: {len(train_x)}")
    print(f"features: {features}")
    print(f"epochs: {epochs}")
    print("batch_size: 512")
    print(f"hidden: {hidden}")
    print(f"classes: {classes}")
    train_seconds = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        started = time.perf_counter()
        total_loss = 0.0
        correct = 0
        samples = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch_y.size(0)
            correct += (logits.argmax(dim=1) == batch_y).sum().item()
            samples += batch_y.size(0)
        torch.mps.synchronize()
        seconds = time.perf_counter() - started
        train_seconds += seconds
        print(
            f"epoch {epoch}/{epochs}: loss={total_loss / samples:.4f}, "
            f"accuracy={100.0 * correct / samples:.2f}% ({seconds:.3f}s)"
        )
    test_loss, test_accuracy = metrics(model, test_loader, loss_fn, device)
    print(f"test: loss={test_loss:.4f}, accuracy={test_accuracy:.2f}%")
    print(f"train_seconds: {train_seconds:.4f}")
    print(f"samples_per_second: {len(train_x) * epochs / train_seconds:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
