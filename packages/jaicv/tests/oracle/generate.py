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


def write_filter(handle) -> None:
    source = ramp(9, 11, 3, seed=21)
    gray = ramp(9, 11, 1, seed=22)
    for k in (3, 5, 7):
        case(handle, "blur", f"{k} {k}", source, cv2.blur(source, (k, k)))
        case(handle, "GaussianBlur", f"{k} {k} 0", source,
             cv2.GaussianBlur(source, (k, k), 0))
        case(handle, "GaussianBlur", f"{k} {k} 1.7", source,
             cv2.GaussianBlur(source, (k, k), 1.7))
        case(handle, "medianBlur", f"{k}", source, cv2.medianBlur(source, k))
    case(handle, "boxFilter", "3 3 nonorm", gray,
         cv2.boxFilter(gray, -1, (3, 3), normalize=False))

    weights = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    case(handle, "filter2D", "sharpen", source, cv2.filter2D(source, -1, weights))
    weights5 = (np.arange(25, dtype=np.float32).reshape(5, 5) - 12.0) / 40.0
    case(handle, "filter2D", "ramp5", gray, cv2.filter2D(gray, -1, weights5))

    for k in (1, 3, 5, 7):
        for dx, dy in ((1, 0), (0, 1), (2, 0), (1, 1)):
            if k == 1 and (dx > 1 or dy > 1):
                continue
            out = cv2.Sobel(gray, cv2.CV_32F, dx, dy, ksize=k)
            case(handle, "Sobel", f"{dx} {dy} {k}", gray, out)
    case(handle, "Scharr", "1 0", gray, cv2.Scharr(gray, cv2.CV_32F, 1, 0))
    case(handle, "Scharr", "0 1", gray, cv2.Scharr(gray, cv2.CV_32F, 0, 1))
    case(handle, "Laplacian", "1", gray, cv2.Laplacian(gray, cv2.CV_32F, ksize=1))
    case(handle, "getGaussianKernel", "5 1.2",
         gray, cv2.getGaussianKernel(5, 1.2).astype(np.float32))
    case(handle, "getGaussianKernel", "7 0",
         gray, cv2.getGaussianKernel(7, 0).astype(np.float32))


def write_threshold(handle) -> None:
    gray = ramp(9, 11, 1, seed=31)
    bimodal = np.where(ramp(12, 12, 1, seed=32) > 128, 210, 40).astype(np.uint8)
    for name, mode in (("BINARY", cv2.THRESH_BINARY), ("BINARY_INV", cv2.THRESH_BINARY_INV),
                       ("TRUNC", cv2.THRESH_TRUNC), ("TOZERO", cv2.THRESH_TOZERO),
                       ("TOZERO_INV", cv2.THRESH_TOZERO_INV)):
        _, out = cv2.threshold(gray, 120, 255, mode)
        case(handle, "threshold", f"{name} 120 255", gray, out)
    level, out = cv2.threshold(bimodal, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    case(handle, "thresholdOtsu", f"{level}", bimodal, out)
    for method, mname in ((cv2.ADAPTIVE_THRESH_MEAN_C, "MEAN"),
                          (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, "GAUSSIAN")):
        out = cv2.adaptiveThreshold(gray, 255, method, cv2.THRESH_BINARY, 5, 3)
        case(handle, "adaptiveThreshold", f"{mname} 5 3", gray, out)


def write_morph(handle) -> None:
    gray = ramp(10, 12, 1, seed=41)
    colour = ramp(10, 12, 3, seed=42)
    for shape, sname in ((cv2.MORPH_RECT, "RECT"), (cv2.MORPH_CROSS, "CROSS"),
                         (cv2.MORPH_ELLIPSE, "ELLIPSE")):
        for k in (3, 5):
            element = cv2.getStructuringElement(shape, (k, k))
            case(handle, "getStructuringElement", f"{sname} {k}", gray, element)
            case(handle, "erode", f"{sname} {k}", gray,
                 cv2.erode(gray, element, borderType=cv2.BORDER_REPLICATE))
            case(handle, "dilate", f"{sname} {k}", gray,
                 cv2.dilate(gray, element, borderType=cv2.BORDER_REPLICATE))
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for op, oname in ((cv2.MORPH_OPEN, "OPEN"), (cv2.MORPH_CLOSE, "CLOSE"),
                      (cv2.MORPH_GRADIENT, "GRADIENT"), (cv2.MORPH_TOPHAT, "TOPHAT"),
                      (cv2.MORPH_BLACKHAT, "BLACKHAT")):
        case(handle, "morphologyEx", f"{oname}", colour,
             cv2.morphologyEx(colour, op, element, borderType=cv2.BORDER_REPLICATE))


def write_edges(handle) -> None:
    rng = np.random.default_rng(51)
    gray = np.zeros((24, 28), np.uint8)
    gray[6:18, 8:20] = 200
    gray[10:14, 12:16] = 60
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    noisy = np.clip(gray.astype(np.int16) + rng.integers(-8, 9, gray.shape), 0, 255).astype(np.uint8)
    for lo, hi in ((50, 150), (80, 200), (30, 90)):
        case(handle, "Canny", f"{lo} {hi} 3 0", noisy, cv2.Canny(noisy, lo, hi, apertureSize=3))
        case(handle, "Canny", f"{lo} {hi} 3 1", noisy,
             cv2.Canny(noisy, lo, hi, apertureSize=3, L2gradient=True))


def main() -> int:
    with OUT.open("w") as handle:
        write_colour(handle)
        write_geometric(handle)
        write_filter(handle)
        write_threshold(handle)
        write_morph(handle)
        write_edges(handle)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
