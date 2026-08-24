"""The pandas side of the frame-operation rows. See frameops.jai."""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

TARGET_SECONDS = 0.08
MIN_REPEATS = 12
MAX_REPEATS = 512
WARMUP = 6

# One row a process, the two sides turn about; see tests/bench/run_suite.py.
_only = None
_listing = False


def wanted(name: str) -> bool:
    if _listing:
        print(name)
        return False
    return _only is None or _only == name


def repeats_for(work) -> int:
    for _ in range(3):
        work()
    started = time.perf_counter()
    for _ in range(3):
        work()
    each = (time.perf_counter() - started) / 3
    if each <= 0.0:
        return MAX_REPEATS
    return max(MIN_REPEATS, min(MAX_REPEATS, int(TARGET_SECONDS / each)))


def run(name: str, rows_moved: int, work) -> None:
    if not wanted(name):
        return
    counted = repeats_for(work)
    for _ in range(WARMUP):
        work()
    started = time.perf_counter()
    for _ in range(counted):
        work()
    seconds = time.perf_counter() - started
    scaled = seconds / counted * MIN_REPEATS
    print(f"jtb\t{name}\t{scaled:.9f}\t{rows_moved * MIN_REPEATS}\t0\tok\trows")


def main() -> int:
    global _only, _listing
    if len(sys.argv) > 1:
        if sys.argv[1] == "list":
            _listing = True
        else:
            _only = sys.argv[1]
    level = os.environ.get("BENCH_LEVEL", "hard")
    scale = 8 if level == "easy" else 4 if level == "medium" else 1
    rows = 2000000 // scale
    groups = 5000

    index = np.arange(rows, dtype=np.int64)
    frame = pd.DataFrame(
        {
            "key": (index * 7919 % groups).astype(np.float32),
            "value": ((index % 1000) * 0.5).astype(np.float32),
            "other": ((index % 97) - 48.0).astype(np.float32),
        }
    )
    lookup = pd.DataFrame(
        {
            "key": np.arange(groups, dtype=np.float32),
            "tag": (np.arange(groups) * 1.5).astype(np.float32),
        }
    )

    run("groupby-sum", rows, lambda: frame.groupby("key").sum())
    run("groupby-mean", rows, lambda: frame.groupby("key").mean())
    run("sort-values", rows, lambda: frame.sort_values(by="value"))
    run("filter-rows", rows, lambda: frame[frame["other"] > 0.0])
    run("column-add", rows, lambda: frame["value"] + frame["other"])
    run("column-multiply", rows, lambda: frame["value"] * 2.0)
    run("frame-sum", rows, lambda: frame.sum())
    run("frame-mean", rows, lambda: frame.mean())
    run("unique", rows, lambda: frame["key"].unique())
    run("value-counts", rows, lambda: frame["key"].value_counts())
    run("drop-duplicates", rows, lambda: frame.drop_duplicates(subset=["key"]))
    run("merge-on-key", rows, lambda: frame.merge(lookup, on="key"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
