"""PyTorch peer for conv.jai: the same two-convolution CIFAR-10 network.

Kept in step with conv.jai deliberately -- the same widths, the same kernel and
pool sizes, the same batch size, the same optimiser and learning rate, and the
same flop count. A benchmark whose two sides train different networks measures
nothing, and convolution was the one part of jaitensor with no peer at all: it
had a Jaithon benchmark, nothing to compare it against, and an OverflowError in
the plan lookup that no gate noticed for as long as it had been there.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch import nn

REPO = Path(__file__).resolve().parents[3]


#: Warm to a wall-clock floor, not a count: one epoch of fashion is a few tens
#: of milliseconds, which does not bring the GPU up to its working clock. Both
#: sides use the same rule (see the .jai twin).
WARM_SECONDS = 0.5
WARM_EPOCH_CAP = 40


def warm(train_epoch) -> None:
    opening = time.perf_counter()
    for _ in range(WARM_EPOCH_CAP):
        train_epoch()
        torch.mps.synchronize()
        if time.perf_counter() - opening >= WARM_SECONDS:
            break


class SliceLoader:
    """Contiguous batches by slicing, in order, dropping the ragged tail.

    Yields exactly what DataLoader(TensorDataset(x, y), batch_size,
    shuffle=False, drop_last=True) yields. DataLoader does not slice: torch
    2.13's TensorDataset has no __getitems__, so every batch was built from
    one indexing op per sample plus a collate -- about 2.6 ms of a 2.6 ms step
    on fashion, which made this peer measure its own loader and every training
    ratio read 1.4-3.6x in jaithon's favour. Jaithon binds a window into the
    resident dataset, so a slice is the like-for-like comparison.
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor, batch_size: int) -> None:
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.count = len(x) // batch_size

    def __len__(self) -> int:
        return self.count

    def __iter__(self):
        b = self.batch_size
        for i in range(self.count):
            yield self.x[i * b:(i + 1) * b], self.y[i * b:(i + 1) * b]


SIDE = 32
CHANNELS = 3
CLASSES = 10
PIXELS = SIDE * SIDE * CHANNELS


def ensure_data() -> None:
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("prepare_data.py")), "cifar10"],
        check=True,
    )


def load_cifar(samples: int):
    root = REPO / "data" / "cifar-10"
    images = torch.frombuffer(bytearray((root / "train-images.bin").read_bytes()), dtype=torch.uint8)
    labels = torch.frombuffer(bytearray((root / "train-labels.bin").read_bytes()), dtype=torch.uint8)
    if images.numel() // PIXELS < samples:
        raise SystemExit("cifar-10 train split is too small")
    # The file is planar, which is what NCHW wants; conv.jai permutes to NHWC
    # because that is the layout its kernels take, and torch stays in NCHW.
    features = images[: samples * PIXELS].view(samples, CHANNELS, SIDE, SIDE).float().div_(255.0)
    return features, labels[:samples].long()


def make_model(width: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(CHANNELS, width, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(width, width * 2, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(width * 2 * (SIDE // 4) * (SIDE // 4), 128),
        nn.ReLU(),
        nn.Linear(128, CLASSES),
    )


def macs_per_sample(width: int) -> int:
    total = 9 * CHANNELS * width * SIDE * SIDE
    half = SIDE // 2
    total += 9 * width * (width * 2) * half * half
    quarter = SIDE // 4
    total += width * 2 * quarter * quarter * 128
    return total + 128 * CLASSES


def run(name: str, width: int, samples: int, batch_size: int, epochs: int, verbose: bool) -> None:
    device = torch.device("mps")
    features, labels = load_cifar(samples)
    loader = SliceLoader(features.to(device), labels.to(device), batch_size)
    torch.manual_seed(7)
    model = make_model(width).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.0)
    loss_fn = nn.CrossEntropyLoss()

    def train_epoch() -> torch.Tensor:
        model.train()
        epoch_loss = torch.zeros((), device=device)
        for batch_x, batch_y in loader:
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="mps", dtype=torch.float16):
                loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            opt.step()
            epoch_loss = epoch_loss + loss.detach()
        return epoch_loss

    warm(train_epoch)
    started = time.perf_counter()
    epoch_losses = [train_epoch() for _ in range(epochs)]
    torch.mps.synchronize()
    total = time.perf_counter() - started
    losses = [value.item() / len(loader) for value in epoch_losses]

    if verbose:
        print(f"framework: pytorch")
        print(f"workload: {name}")
        print(f"samples: {samples}")
        print(f"epochs: {epochs}")
        print(f"batch_size: {batch_size}")
        print(f"train_seconds: {total:.4f}")
        return
    processed = len(loader) * batch_size * epochs
    flops = 6 * processed * macs_per_sample(width)
    valid = len(losses) == epochs and total > 0.0 and all(
        value > 0.0 and value < float("inf") for value in losses
    )
    if len(losses) > 1 and losses[-1] > losses[0] * 1.05:
        valid = False
    check = "ok" if valid else "invalid"
    final_loss = losses[-1] if losses else 0.0
    print(f"jtb\t{name}\t{total:.9f}\t{flops}\t{processed}\t{check}\t{final_loss:.6f}")


def main() -> int:
    if not torch.backends.mps.is_available():
        raise SystemExit("bench needs Apple MPS")
    verbose = os.environ.get("JAITENSOR_VERBOSE", "0") != "0"
    level = os.environ.get("BENCH_LEVEL", "hard")
    ensure_data()
    samples = 4096 if level == "easy" else 8192 if level == "medium" else 16384
    epochs = 1 if level == "easy" else 2 if level == "medium" else 3
    run("conv-small", 32, samples, 128, epochs, verbose)
    run("conv-wide", 64, samples, 256, epochs, verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
