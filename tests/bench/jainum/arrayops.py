"""The numpy side of the array-operation rows. See arrayops.jai."""

from __future__ import annotations

import os
import sys
import time

import numpy as np

TARGET_SECONDS = 0.08
MIN_REPEATS = 30
MAX_REPEATS = 2048
WARMUP = 40
WIDTH = 4


# One row a process, the two sides turn about; see the note in arrayops.jai.
_only = None
_listing = False


def wanted(name: str) -> bool:
    if _listing:
        print(name)
        return False
    return _only is None or _only == name


def repeats_for(work) -> int:
    for _ in range(8):
        work()
    started = time.perf_counter()
    for _ in range(8):
        work()
    each = (time.perf_counter() - started) / 8
    if each <= 0.0:
        return MAX_REPEATS
    return max(MIN_REPEATS, min(MAX_REPEATS, int(TARGET_SECONDS / each)))


def run(name: str, bytes_moved: int, work) -> None:
    if not wanted(name):
        return
    product = np.asarray(work())
    flat = product.reshape(-1)
    probes = flat[[0, flat.size // 2, flat.size - 1]]
    sane = bool(np.isfinite(probes.astype(np.float64)).all()) and product.size > 0
    counted = repeats_for(work)
    for _ in range(WARMUP):
        work()
    started = time.perf_counter()
    for _ in range(counted):
        work()
    seconds = time.perf_counter() - started
    scaled = seconds / counted * MIN_REPEATS
    check = "ok" if sane else "invalid"
    print(f"jtb\t{name}\t{scaled:.9f}\t{bytes_moved * MIN_REPEATS}\t0\t{check}\tbytes/elementwise")


def main() -> int:
    global _only, _listing
    if len(sys.argv) > 1:
        if sys.argv[1] == "list":
            _listing = True
        else:
            _only = sys.argv[1]
    level = os.environ.get("BENCH_LEVEL", "hard")
    scale = 4 if level == "easy" else 2 if level == "medium" else 1

    side = 4096 // scale
    square = side * side
    gemm_side = 2048 // scale
    planes = 32 // scale
    plane_side = 256

    rng = np.random.default_rng(20260824)
    wide = rng.standard_normal((side, side), dtype=np.float32)
    other = rng.standard_normal((side, side), dtype=np.float32)
    row = rng.standard_normal(side, dtype=np.float32)
    left = rng.standard_normal((gemm_side, gemm_side), dtype=np.float32)
    right = rng.standard_normal((gemm_side, gemm_side), dtype=np.float32)
    stack_a = rng.standard_normal((planes, plane_side, plane_side), dtype=np.float32)
    stack_b = rng.standard_normal((planes, plane_side, plane_side), dtype=np.float32)
    table = rng.standard_normal((50000, 512), dtype=np.float32)
    picks = rng.integers(0, 50000, side)
    tall = rng.standard_normal((1024 // scale, side), dtype=np.float32)

    gemm_bytes = gemm_side * gemm_side * 3 * WIDTH
    batch_bytes = planes * plane_side * plane_side * 3 * WIDTH
    pair_bytes = square * 3 * WIDTH
    unary_bytes = square * 2 * WIDTH
    reduce_bytes = square * WIDTH
    sort_bytes = (1024 // scale) * side * 2 * WIDTH

    run("matmul", gemm_bytes, lambda: left @ right)
    run("matmul-batched", batch_bytes, lambda: stack_a @ stack_b)
    run("add", pair_bytes, lambda: wide + other)
    run("add-broadcast", pair_bytes, lambda: wide + row)
    run("multiply", pair_bytes, lambda: wide * other)
    run("exp", unary_bytes, lambda: np.exp(wide))
    # A row against every row of a matrix; see the note in arrayops.jai.
    run("maximum-broadcast", pair_bytes, lambda: np.maximum(wide, row))
    run("sum-axis0", reduce_bytes, lambda: wide.sum(axis=0))
    run("sum-axis1", reduce_bytes, lambda: wide.sum(axis=1))
    run("argmax-axis1", reduce_bytes, lambda: wide.argmax(axis=1))
    run("transpose-copy", unary_bytes, lambda: np.ascontiguousarray(wide.T))
    run("cumsum-axis1", unary_bytes, lambda: np.cumsum(wide, axis=1))
    run("sort-axis1", sort_bytes, lambda: np.sort(tall, axis=1))
    run("take-rows", side * 512 * 2 * WIDTH, lambda: np.take(table, picks, axis=0))
    # Timed apart; see the note in arrayops.jai.
    mask = wide > other
    run("greater", pair_bytes, lambda: wide > other)
    run("where", pair_bytes + reduce_bytes, lambda: np.where(mask, wide, other))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
