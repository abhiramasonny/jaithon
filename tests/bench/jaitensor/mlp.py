import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parents[3]

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
        [sys.executable, str(Path(__file__).with_name("prepare_data.py")), dataset],
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


def batch_size_of(hidden: list[int], features: int) -> int:
    widest = max(hidden) if hidden else 0
    if widest >= 8192:
        return 512
    # Match Jaithon's safe batch below the fused MPSGraph workspace cliff.
    if features >= 3072 and widest >= 2048:
        return 512
    if widest >= 1024 or len(hidden) >= 3 or features >= 3072:
        return 2048
    return 512


def use_amp(hidden: list[int]) -> bool:
    widest = max(hidden) if hidden else 0
    return widest >= 1024


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
    name = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("JAITENSOR_WORKLOAD", "fashion")
    verbose = os.environ.get("JAITENSOR_VERBOSE", "0") != "0"
    if len(sys.argv) > 2 and sys.argv[2].isdigit():
        epochs = int(sys.argv[2])
    else:
        level = os.environ.get("BENCH_LEVEL", "hard")
        epochs = 1 if level == "easy" else 2 if level == "medium" else 8
    if not torch.backends.mps.is_available():
        raise SystemExit("bench needs Apple MPS")
    ensure_data(name)
    device = torch.device("mps")
    train_x, train_y, test_x, test_y, hidden, classes = load_workload(name)
    batch_size = batch_size_of(hidden, train_x.shape[1])
    del test_x, test_y
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )
    torch.manual_seed(7)
    model = make_model(train_x.shape[1], hidden, classes).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.08, momentum=0.0)
    loss_fn = nn.CrossEntropyLoss()
    def train_epoch() -> torch.Tensor:
        model.train()
        epoch_loss = torch.zeros((), device=device)
        for batch_x, batch_y in train_loader:
            opt.zero_grad(set_to_none=True)
            if use_amp(hidden):
                with torch.autocast(device_type="mps", dtype=torch.float16):
                    loss = loss_fn(model(batch_x), batch_y)
            else:
                loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            opt.step()
            epoch_loss = epoch_loss + loss.detach()
        return epoch_loss

    train_epoch()
    torch.mps.synchronize()
    started = time.perf_counter()
    epoch_losses = []
    for _epoch in range(epochs):
        epoch_losses.append(train_epoch())
    torch.mps.synchronize()
    total = time.perf_counter() - started
    losses = [value.item() / len(train_loader) for value in epoch_losses]
    if verbose:
        print(f"framework: pytorch")
        print(f"workload: {name}")
        print(f"GPU: {torch.backends.mps.is_available() and 'MPS' or 'none'}")
        print(f"samples: {len(train_x)}")
        print(f"features: {train_x.shape[1]}")
        print(f"epochs: {epochs}")
        print(f"batch_size: {batch_size}")
        print(f"hidden: {hidden}")
        print(f"classes: {classes}")
        print(f"train_seconds: {total:.4f}")
        print(f"samples_per_second: {len(train_x) * epochs / total:.1f}")
        return 0
    processed_per_epoch = len(train_loader) * batch_size
    processed = processed_per_epoch * epochs
    widths = [train_x.shape[1], *hidden, classes]
    macs = sum(left * right for left, right in zip(widths, widths[1:]))
    flops = 6 * processed * macs
    valid = len(losses) == epochs and total > 0.0 and all(
        value > 0.0 and value < float("inf") for value in losses
    )
    if len(losses) > 1 and losses[-1] > losses[0] * 1.05:
        valid = False
    check = "ok" if valid else "invalid"
    final_loss = losses[-1] if losses else 0.0
    print(f"jtb\t{name}\t{total:.9f}\t{flops}\t{processed}\t{check}\t{final_loss:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
