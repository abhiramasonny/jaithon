"""The PyTorch peer of train.jai: the same ResNet, data, augmentation and recipe
on MPS in float32 NCHW.

    ~/.venvs/scratch/bin/python examples/cifar_resnet/cifar_resnet.py [--epochs 30] [--seed 0]
        [--depth 20] [--width 16] [--batch 128] [--lr 0.2] [--pct 0.25] [--wd 5e-4]

Kept identical to the Jaithon side on purpose:
- data: uint8 planar CIFAR-10 / 255, then standardised per channel with the
  training set's mean and std (the same six constants), --normalize 0 to skip;
- model: 3x3 stem conv (with bias) + BN + ReLU, (depth-2)/6 basic blocks per
  stage, bias-free block convs, a 1x1 stride-2 projection + BN where the shape
  changes, global average pool, linear head; every weight drawn uniform with
  variance 1/fan_in and every bias zero, as jaitensor draws them;
- augmentation: the same hash-based per-image shift in [-pad, pad] and flip,
  computed for each image index and seed exactly as jaitensor's
  random_crop_flip does, applied on the GPU with two gathers;
- optimiser: SGD, Nesterov momentum 0.9, weight decay 5e-4, OneCycleLR
  (cosine, cycle_momentum off) stepped every batch;
- batches: a fresh permutation each epoch, batch 128, the short last batch kept;
- evaluation: BatchNorm in batch-statistics mode at batch 1000 (what jaitensor
  does, as it keeps no running statistics), and running statistics alongside.

Never channels_last: on MPS it is 2.3-43x slower for these convolutions.
"""

import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

dev = torch.device("mps")


CHANNEL_MEAN = [0.49139968, 0.48215841, 0.44653091]
CHANNEL_STD = [0.24703223, 0.24348513, 0.26158784]


def load(split, normalize):
    x = np.fromfile(f"data/cifar-10/{split}-images.bin", dtype=np.uint8)
    y = np.fromfile(f"data/cifar-10/{split}-labels.bin", dtype=np.uint8)
    n = y.shape[0]
    x = torch.from_numpy(x[: n * 3072].reshape(n, 3, 32, 32).astype(np.float32) / 255.0).to(dev)
    if normalize:
        mean = torch.tensor(CHANNEL_MEAN, device=dev)[None, :, None, None]
        std = torch.tensor(CHANNEL_STD, device=dev)[None, :, None, None]
        x = ((x - mean) * (1.0 / std)).contiguous()
    return x, torch.from_numpy(y.astype(np.int64)).to(dev)


def init_like_jaitensor(module):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            fan = m.weight[0].numel()
            limit = math.sqrt(3.0 / fan)
            nn.init.uniform_(m.weight, -limit, limit)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.proj = None
        if stride != 1 or cin != cout:
            self.proj = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(h))
        s = x if self.proj is None else self.proj(x)
        return F.relu(y + s)


class ResNet(nn.Module):
    def __init__(self, depth, width):
        super().__init__()
        blocks = (depth - 2) // 6
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, 1, 1), nn.BatchNorm2d(width), nn.ReLU())
        layers = []
        cin = width
        for stage in range(3):
            cout = width << stage
            for b in range(blocks):
                layers.append(Block(cin, cout, 2 if stage > 0 and b == 0 else 1))
                cin = cout
        self.blocks = nn.Sequential(*layers)
        self.fc = nn.Linear(cin, 10)

    def forward(self, x):
        x = self.blocks(self.stem(x))
        return self.fc(F.avg_pool2d(x, 8).flatten(1))


def mix32(h):
    h = h ^ (h >> np.uint32(16))
    h = h * np.uint32(0x7FEB352D)
    h = h ^ (h >> np.uint32(15))
    h = h * np.uint32(0x846CA68B)
    h = h ^ (h >> np.uint32(16))
    return h


def crop_params(n, seed, pad):
    """Per-image (dx, dy, flip), exactly jaitensor's random_crop_flip hash."""
    with np.errstate(over="ignore"):
        idx = np.arange(n, dtype=np.uint32)
        s = mix32(np.array([seed & 0xFFFFFFFF], dtype=np.uint32))[0]
        h = mix32((idx * np.uint32(2654435761)) ^ s)
    span = np.uint32(2 * pad + 1)
    dx = (h % span).astype(np.int64) - pad
    dy = ((h // span) % span).astype(np.int64) - pad
    fl = ((h // (span * span)) & np.uint32(1)).astype(bool)
    return dx, dy, fl


def augment(x, seed, pad):
    """Shift each image by (dx, dy) with zeros brought in, then maybe mirror it:
    a crop of the zero-padded image. Two gathers on the device."""
    n = x.shape[0]
    dx, dy, fl = crop_params(n, seed, pad)
    dx = torch.from_numpy(dx).to(dev)
    dy = torch.from_numpy(dy).to(dev)
    fl = torch.from_numpy(fl).to(dev)
    p = F.pad(x, (pad, pad, pad, pad))
    ar = torch.arange(32, device=dev)
    cols = torch.where(fl[:, None], 31 - ar[None, :], ar[None, :]) + dx[:, None] + pad  # n,32
    rows = ar[None, :] + dy[:, None] + pad  # n,32
    p = torch.gather(p, 2, rows[:, None, :, None].expand(n, 3, 32, p.shape[3]))
    return torch.gather(p, 3, cols[:, None, None, :].expand(n, 3, 32, 32)).contiguous()


def evaluate(net, x, y, batch_stats):
    net.train(batch_stats)
    correct = 0
    with torch.no_grad():
        for i in range(0, x.shape[0], 1000):
            correct += int((net(x[i:i + 1000]).argmax(1) == y[i:i + 1000]).sum())
    return 100.0 * correct / x.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depth", type=int, default=20)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--pct", type=float, default=0.25)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--pad", type=int, default=4)
    ap.add_argument("--normalize", type=int, default=1)
    a = ap.parse_args()
    print(f"torch {torch.__version__} MPS fp32 NCHW: ResNet-{a.depth} width {a.width}, {a.epochs} epochs, "
          f"batch {a.batch}, SGD nesterov 0.9 wd {a.wd}, one-cycle max lr {a.lr} pct_start {a.pct}, "
          f"{'mean/std' if a.normalize else 'raw [0, 1]'} inputs, seed {a.seed}")
    torch.manual_seed(a.seed)
    xtr, ytr = load("train", a.normalize)
    xte, yte = load("test", a.normalize)
    net = ResNet(a.depth, a.width).to(dev)
    init_like_jaitensor(net)
    params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.SGD(net.parameters(), lr=a.lr, momentum=0.9, nesterov=True, weight_decay=a.wd)
    n = xtr.shape[0]
    steps = (n + a.batch - 1) // a.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps * a.epochs,
                                                pct_start=a.pct, cycle_momentum=False)
    epoch_times = []
    run_start = time.perf_counter()
    for e in range(a.epochs):
        t0 = time.perf_counter()
        xa = augment(xtr, a.seed * 1000003 + e, a.pad)
        torch.mps.synchronize()
        aug_ms = (time.perf_counter() - t0) * 1000
        net.train()
        perm = torch.randperm(n, device=dev)
        tot = torch.zeros((), device=dev)
        hits = torch.zeros((), device=dev, dtype=torch.int64)
        for i in range(steps):
            idx = perm[i * a.batch:(i + 1) * a.batch]
            xb = xa.index_select(0, idx)
            yb = ytr.index_select(0, idx)
            out = net(xb)
            loss = F.cross_entropy(out, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step() if e * steps + i + 1 < steps * a.epochs else None
            tot += loss.detach() * yb.shape[0]
            hits += (out.detach().argmax(1) == yb).sum()
        torch.mps.synchronize()
        dt = time.perf_counter() - t0
        epoch_times.append(dt)
        print(f"epoch {e + 1}/{a.epochs}  {dt:.3f}s  aug {aug_ms:.1f}ms  loss {float(tot) / n:.4f}  "
              f"train {100.0 * float(hits) / n:.2f}%  lr {opt.param_groups[0]['lr']:.5f}", flush=True)
        del xa
    train_s = time.perf_counter() - run_start
    steady = sorted(epoch_times[1:])[len(epoch_times[1:]) // 2] if len(epoch_times) > 1 else epoch_times[0]
    print(f"parameters {params}")
    print(f"train {train_s:.2f}s  steady epoch {steady:.3f}s  first epoch {epoch_times[0]:.3f}s")
    print(f"test accuracy {evaluate(net, xte, yte, True):.2f}%  (batch statistics, batch 1000)")
    print(f"test accuracy {evaluate(net, xte, yte, False):.2f}%  (running statistics)")


main()
