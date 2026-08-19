#!/usr/bin/env python3
"""Record what OpenCV produces, so the Jaithon side can be compared against it.

Each case is written as three lines: a header naming the operation and its
arguments, the input, and OpenCV's output. Values are printed with enough
digits to round-trip a float32.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent / "cases.txt"

DEPTHS = {
    np.dtype(np.uint8): 0,
    np.dtype(np.int8): 1,
    np.dtype(np.uint16): 2,
    np.dtype(np.int16): 3,
    np.dtype(np.int32): 4,
    np.dtype(np.float32): 5,
    np.dtype(np.float64): 6,
}


def type_code(array: np.ndarray) -> int:
    channels = 1 if array.ndim == 2 else array.shape[2]
    return DEPTHS[array.dtype] + ((channels - 1) << 3)


def emit(handle, tag: str, array: np.ndarray) -> None:
    flat = np.asarray(array).reshape(-1).astype(np.float64)
    rows = array.shape[0]
    cols = array.shape[1]
    values = " ".join(f"{v:.9g}" for v in flat)
    handle.write(f"{tag} {rows} {cols} {type_code(array)} {values}\n")


def case(handle, name: str, args: str, source: np.ndarray, result: np.ndarray) -> None:
    handle.write(f"case {name} {args}\n")
    emit(handle, "in", source)
    emit(handle, "out", result)


def ramp(rows: int, cols: int, channels: int, seed: int, dtype=np.uint8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if dtype == np.uint8:
        data = rng.integers(0, 256, size=(rows, cols, channels), dtype=np.uint8)
    else:
        data = rng.random((rows, cols, channels)).astype(np.float32)
    return data[:, :, 0] if channels == 1 else data


COLOR_CODES = {
    "BGR2GRAY": cv2.COLOR_BGR2GRAY,
    "RGB2GRAY": cv2.COLOR_RGB2GRAY,
    "BGR2RGB": cv2.COLOR_BGR2RGB,
    "BGR2HSV": cv2.COLOR_BGR2HSV,
    "HSV2BGR": cv2.COLOR_HSV2BGR,
    "BGR2HLS": cv2.COLOR_BGR2HLS,
    "HLS2BGR": cv2.COLOR_HLS2BGR,
    "BGR2Lab": cv2.COLOR_BGR2Lab,
    "Lab2BGR": cv2.COLOR_Lab2BGR,
    "BGR2YCrCb": cv2.COLOR_BGR2YCrCb,
    "YCrCb2BGR": cv2.COLOR_YCrCb2BGR,
    "BGR2XYZ": cv2.COLOR_BGR2XYZ,
    "XYZ2BGR": cv2.COLOR_XYZ2BGR,
    "BGR2YUV": cv2.COLOR_BGR2YUV,
    "YUV2BGR": cv2.COLOR_YUV2BGR,
    "GRAY2BGR": cv2.COLOR_GRAY2BGR,
    "BGR2BGRA": cv2.COLOR_BGR2BGRA,
    "BGRA2BGR": cv2.COLOR_BGRA2BGR,
}


def write_colour(handle) -> None:
    for name, code in COLOR_CODES.items():
        channels = 1 if name.startswith("GRAY") else (4 if name.startswith("BGRA") else 3)
        source = ramp(6, 7, channels, seed=hash(name) % 9999)
        case(handle, "cvtColor", name, source, cv2.cvtColor(source, code))


def main() -> int:
    with OUT.open("w") as handle:
        write_colour(handle)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
