"""The PyTorch side of the tensor-operation rows. See tensorops.jai."""

from __future__ import annotations

import os
import sys
import time

import torch

TARGET_SECONDS = 0.08
MIN_REPEATS = 30
MAX_REPEATS = 2048
WARMUP = 40


def repeats_for(work) -> int:
    for _ in range(8):
        work()
    torch.mps.synchronize()
    started = time.perf_counter()
    for _ in range(8):
        work()
    torch.mps.synchronize()
    each = (time.perf_counter() - started) / 8
    if each <= 0.0:
        return MAX_REPEATS
    return max(MIN_REPEATS, min(MAX_REPEATS, int(TARGET_SECONDS / each)))


def run(name: str, bytes_moved: int, work) -> None:
    product = work()
    flat = product.flatten()
    probes = flat[[0, flat.numel() // 2, flat.numel() - 1]]
    sane = bool(torch.isfinite(probes).all().item()) and product.numel() > 0
    counted = repeats_for(work)
    for _ in range(WARMUP):
        work()
    torch.mps.synchronize()
    started = time.perf_counter()
    for _ in range(counted):
        work()
    torch.mps.synchronize()
    seconds = time.perf_counter() - started
    scaled = seconds / counted * MIN_REPEATS
    check = "ok" if sane else "invalid"
    print(f"jtb\t{name}\t{scaled:.9f}\t{bytes_moved * MIN_REPEATS}\t0\t{check}\tbytes/elementwise")


def main() -> int:
    if not torch.backends.mps.is_available():
        raise SystemExit("bench needs a GPU")
    level = os.environ.get("BENCH_LEVEL", "hard")
    scale = 4 if level == "easy" else 2 if level == "medium" else 1

    rows = 4096 // scale
    batch = 64 // scale
    wide = torch.randn(rows, 4096, device="mps")
    # The jaithon side holds pictures channels-last and swaps to planar for
    # the other direction; both sides therefore time the same two swaps.
    images = torch.randn(batch, 56, 56, 64, device="mps")
    planar = images.permute(0, 3, 1, 2).contiguous()
    table = torch.randn(50000, 512, device="mps")
    ids = torch.randint(0, 50000, (rows,), device="mps")
    left = torch.randn(rows, 2048, device="mps")
    right = torch.randn(rows, 2048, device="mps")
    small = torch.randn(batch, 32, 32, 64, device="mps")
    small_planar = small.permute(0, 3, 1, 2).contiguous()
    max_pool = torch.nn.MaxPool2d(2)
    avg_pool = torch.nn.AvgPool2d(2)

    word = 4
    flat = wide.numel() * word
    picture = images.numel() * word

    run("op-softmax", flat * 2, lambda: torch.softmax(wide, dim=1))
    run("op-transpose", flat * 2, lambda: wide.t().contiguous())
    run("op-row-sum", flat + rows * word, lambda: wide.sum(dim=1))
    run("op-gelu", flat * 2, lambda: torch.nn.functional.gelu(wide))
    run("op-to-nchw", picture * 2, lambda: images.permute(0, 3, 1, 2).contiguous())
    run("op-to-nhwc", picture * 2, lambda: planar.permute(0, 2, 3, 1).contiguous())
    run(
        "op-pad",
        picture + batch * 58 * 58 * 64 * word,
        lambda: torch.nn.functional.pad(planar, (1, 1, 1, 1)),
    )
    run("op-max-pool", picture + picture // 4, lambda: max_pool(planar))
    run("op-avg-pool", picture + picture // 4, lambda: avg_pool(planar))
    run(
        "op-upsample",
        small.numel() * word * 5,
        lambda: torch.nn.functional.interpolate(small_planar, scale_factor=2, mode="nearest"),
    )
    run("op-gather", rows * 512 * word * 2, lambda: table[ids])
    run("op-cat", left.numel() * word * 4, lambda: torch.cat([left, right], dim=1))
    run("op-repeat", left.numel() * word * 5, lambda: left.repeat(1, 4))
    return 0


if __name__ == "__main__":
    sys.exit(main())
