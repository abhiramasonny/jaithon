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


def write_geometric(handle) -> None:
    source = ramp(8, 10, 3, seed=11)
    gray = ramp(8, 10, 1, seed=12)
    for name, flag in (("NEAREST", cv2.INTER_NEAREST), ("LINEAR", cv2.INTER_LINEAR),
                       ("CUBIC", cv2.INTER_CUBIC), ("AREA", cv2.INTER_AREA),
                       ("LANCZOS4", cv2.INTER_LANCZOS4)):
        case(handle, "resize", f"{name} 5 4",
             source, cv2.resize(source, (5, 4), interpolation=flag))
        case(handle, "resize", f"{name} 20 16",
             source, cv2.resize(source, (20, 16), interpolation=flag))
    case(handle, "pyrDown", "-", source, cv2.pyrDown(source))
    case(handle, "pyrUp", "-", source, cv2.pyrUp(source))
    for name, mode in (("CONSTANT", cv2.BORDER_CONSTANT),
                       ("REPLICATE", cv2.BORDER_REPLICATE),
                       ("REFLECT", cv2.BORDER_REFLECT),
                       ("REFLECT_101", cv2.BORDER_REFLECT_101),
                       ("WRAP", cv2.BORDER_WRAP)):
        case(handle, "copyMakeBorder", f"{name} 2 3 1 4", source,
             cv2.copyMakeBorder(source, 2, 3, 1, 4, mode, value=(0, 0, 0)))

    matrix = cv2.getRotationMatrix2D((4.5, 3.5), 27.0, 1.3).astype(np.float32)
    case(handle, "warpAffine", "rot27", source,
         cv2.warpAffine(source, matrix, (10, 8), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)))
    src_pts = np.float32([[0, 0], [9, 0], [9, 7], [0, 7]])
    dst_pts = np.float32([[1, 1], [8, 0.5], [9, 6], [0.5, 7]])
    persp = cv2.getPerspectiveTransform(src_pts, dst_pts).astype(np.float32)
    case(handle, "warpPerspective", "quad", source,
         cv2.warpPerspective(source, persp, (10, 8), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)))
    case(handle, "getRotationMatrix2D", "4.5 3.5 27 1.3", gray, matrix)
    case(handle, "getPerspectiveTransform", "quad", gray, persp)
    affine = cv2.getAffineTransform(src_pts[:3], dst_pts[:3]).astype(np.float32)
    case(handle, "getAffineTransform", "tri", gray, affine)

    map_x = np.tile(np.arange(10, dtype=np.float32)[::-1], (8, 1))
    map_y = np.tile(np.arange(8, dtype=np.float32).reshape(-1, 1), (1, 10))
    remapped = cv2.remap(source, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    case(handle, "remap", "flipx", source, remapped)


def main() -> int:
    with OUT.open("w") as handle:
        write_colour(handle)
        write_geometric(handle)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
