#!/usr/bin/env python3
"""Record what OpenCV produces, so the Jaithon side can be compared against it.

Each case is written as three lines: a header naming the operation and its
arguments, the input, and OpenCV's output. Values are printed with enough
digits to round-trip a float32.
"""
from __future__ import annotations

import sys
import zlib
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
        # A stable seed: Python's hash() is salted per process, so using it
        # here made the recorded inputs change on every regeneration.
        source = ramp(6, 7, channels, seed=zlib.crc32(name.encode()) % 9999)
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


COLOUR = (10, 200, 90)


def canvas(channels: int = 3, rows: int = 26, cols: int = 32) -> np.ndarray:
    if channels == 1:
        return np.zeros((rows, cols), np.uint8)
    return np.zeros((rows, cols, channels), np.uint8)


def paint(handle, name: str, args: str, source: np.ndarray, draw) -> None:
    out = source.copy()
    draw(out)
    case(handle, name, args, source, out)


def write_drawing(handle) -> None:
    base = canvas()
    grey = canvas(1)
    lines = [(2, 3, 28, 20), (28, 3, 2, 20), (5, 22, 25, 2), (0, 0, 31, 25),
             (4, 12, 27, 12), (16, 1, 16, 24), (-8, 5, 40, 18), (10, -6, 20, 30)]
    for x1, y1, x2, y2 in lines:
        for thickness in (1, 2, 3, 5):
            for lt, ltn in ((cv2.LINE_8, 8), (cv2.LINE_4, 4)):
                if thickness > 1 and ltn == 4:
                    continue
                paint(handle, "drawLine", f"{x1} {y1} {x2} {y2} {thickness} {ltn}", base,
                      lambda m, a=(x1, y1), b=(x2, y2), t=thickness, l=lt:
                          cv2.line(m, a, b, COLOUR, t, l))
    for thickness in (1, 3):
        paint(handle, "drawLine", f"3 4 20 15 {thickness} 8", grey,
              lambda m, t=thickness: cv2.line(m, (3, 4), (20, 15), (200, 0, 0), t, cv2.LINE_8))

    rects = [(3, 3, 26, 20), (0, 0, 31, 25), (10, 8, 12, 10), (-5, -4, 20, 14)]
    for x1, y1, x2, y2 in rects:
        for thickness in (1, 2, 4, -1):
            paint(handle, "drawRect", f"{x1} {y1} {x2} {y2} {thickness} 8", base,
                  lambda m, a=(x1, y1), b=(x2, y2), t=thickness:
                      cv2.rectangle(m, a, b, COLOUR, t, cv2.LINE_8))

    for cx, cy in ((16, 13), (3, 3), (30, 24)):
        for radius in (0, 1, 2, 5, 9, 14):
            for thickness in (1, -1, 3):
                paint(handle, "drawCircle", f"{cx} {cy} {radius} {thickness} 8", base,
                      lambda m, c=(cx, cy), r=radius, t=thickness:
                          cv2.circle(m, c, r, COLOUR, t, cv2.LINE_8))

    ellipses = [(16, 13, 12, 7, 0, 0, 360), (16, 13, 12, 7, 30, 0, 360),
                (16, 13, 10, 10, 0, 0, 180), (16, 13, 9, 5, 45, 30, 300),
                (16, 13, 2, 2, 0, 0, 360), (8, 8, 14, 4, 120, 0, 360)]
    for cx, cy, ax, ay, angle, start, end in ellipses:
        for thickness in (1, 2, -1):
            paint(handle, "drawEllipse",
                  f"{cx} {cy} {ax} {ay} {angle} {start} {end} {thickness} 8", base,
                  lambda m, c=(cx, cy), a=(ax, ay), an=angle, s=start, e=end, t=thickness:
                      cv2.ellipse(m, c, a, an, s, e, COLOUR, t, cv2.LINE_8))

    polys = {
        "tri": [(4, 3), (28, 8), (12, 22)],
        "quad": [(3, 4), (27, 2), (29, 21), (6, 23)],
        "concave": [(2, 2), (29, 4), (16, 12), (29, 23), (3, 21)],
        "star": [(16, 1), (20, 10), (30, 11), (22, 17), (25, 24),
                 (16, 19), (7, 24), (10, 17), (2, 11), (12, 10)],
    }
    for tag, pts in polys.items():
        flat = " ".join(f"{x} {y}" for x, y in pts)
        array = np.array([pts], np.int32)
        for closed in (0, 1):
            for thickness in (1, 2):
                paint(handle, "drawPolylines",
                      f"{closed} {thickness} 8 {len(pts)} {flat}", base,
                      lambda m, a=array, c=bool(closed), t=thickness:
                          cv2.polylines(m, a, c, COLOUR, t, cv2.LINE_8))
        paint(handle, "drawFillPoly", f"{len(pts)} {flat}", base,
              lambda m, a=array: cv2.fillPoly(m, a, COLOUR, cv2.LINE_8))
        if tag != "concave" and tag != "star":
            paint(handle, "drawFillConvexPoly", f"{len(pts)} {flat}", base,
                  lambda m, p=np.array(pts, np.int32):
                      cv2.fillConvexPoly(m, p, COLOUR, cv2.LINE_8))

    arrows = [(4, 4, 26, 18, 1), (26, 20, 5, 6, 2)]
    for x1, y1, x2, y2, thickness in arrows:
        paint(handle, "drawArrow", f"{x1} {y1} {x2} {y2} {thickness} 8 10", base,
              lambda m, a=(x1, y1), b=(x2, y2), t=thickness:
                  cv2.arrowedLine(m, a, b, COLOUR, t, cv2.LINE_8, 0, 0.1))

    markers = [(cv2.MARKER_CROSS, 0), (cv2.MARKER_TILTED_CROSS, 1), (cv2.MARKER_STAR, 2),
               (cv2.MARKER_DIAMOND, 3), (cv2.MARKER_SQUARE, 4),
               (cv2.MARKER_TRIANGLE_UP, 5), (cv2.MARKER_TRIANGLE_DOWN, 6)]
    for marker, code in markers:
        for size in (7, 12):
            paint(handle, "drawMarker", f"16 13 {code} {size} 1 8", base,
                  lambda m, k=marker, s=size:
                      cv2.drawMarker(m, (16, 13), COLOUR, k, s, 1, cv2.LINE_8))


def contour_shapes():
    shapes = []
    a = np.zeros((22, 26), np.uint8)
    a[4:16, 5:20] = 255
    a[7:12, 9:16] = 0
    a[9:10, 11:13] = 255
    shapes.append(("nested", a))

    b = np.zeros((20, 24), np.uint8)
    cv2.circle(b, (7, 8), 5, 255, -1)
    cv2.circle(b, (17, 13), 4, 255, -1)
    cv2.circle(b, (7, 8), 2, 0, -1)
    b[0, 0] = 255
    b[19, 23] = 255
    shapes.append(("blobs", b))

    rng = np.random.default_rng(77)
    shapes.append(("speckle", (rng.random((18, 21)) < 0.42).astype(np.uint8) * 255))

    d = np.zeros((16, 18), np.uint8)
    d[2:14, 2:4] = 255
    d[2:4, 2:16] = 255
    d[12:14, 2:16] = 255
    shapes.append(("bars", d))
    return shapes


CONTOUR_MODES = {"EXTERNAL": cv2.RETR_EXTERNAL, "LIST": cv2.RETR_LIST,
                 "CCOMP": cv2.RETR_CCOMP, "TREE": cv2.RETR_TREE}
CONTOUR_METHODS = {"NONE": cv2.CHAIN_APPROX_NONE, "SIMPLE": cv2.CHAIN_APPROX_SIMPLE}


def write_contours(handle) -> None:
    for tag, image in contour_shapes():
        for mname, mode in CONTOUR_MODES.items():
            for cname, method in CONTOUR_METHODS.items():
                found, hier = cv2.findContours(image, mode, method)
                rows = []
                for ci, contour in enumerate(found):
                    for pi, point in enumerate(contour):
                        rows.append([ci, pi, int(point[0][0]), int(point[0][1])])
                points = np.array(rows, np.int32).reshape(-1, 4)
                case(handle, "findContours", f"{tag} {mname} {cname}", image, points)
                tree = np.array(hier[0] if hier is not None else [], np.int32).reshape(-1, 4)
                case(handle, "contourHierarchy", f"{tag} {mname} {cname}", image, tree)


def shape_contours():
    out = []
    for tag, image in contour_shapes():
        found, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for index, contour in enumerate(found):
            if len(contour) < 6:
                continue
            out.append((tag, image, index, contour))
    return out


def write_shape(handle) -> None:
    for tag, image, index, contour in shape_contours():
        head = f"{tag} {index}"
        m = cv2.moments(contour)
        keys = ["m00", "m10", "m01", "m20", "m11", "m02", "m30", "m21", "m12", "m03",
                "mu20", "mu11", "mu02", "mu30", "mu21", "mu12", "mu03",
                "nu20", "nu11", "nu02", "nu30", "nu21", "nu12", "nu03"]
        case(handle, "moments", head, image,
             np.array([[m[k] for k in keys]], np.float32))
        case(handle, "huMoments", head, image,
             cv2.HuMoments(m).reshape(1, 7).astype(np.float32))
        case(handle, "contourArea", head, image,
             np.array([[cv2.contourArea(contour)]], np.float32))
        case(handle, "arcLengthClosed", head, image,
             np.array([[cv2.arcLength(contour, True)]], np.float32))
        case(handle, "arcLengthOpen", head, image,
             np.array([[cv2.arcLength(contour, False)]], np.float32))
        x, y, w, h = cv2.boundingRect(contour)
        case(handle, "boundingRect", head, image, np.array([[x, y, w, h]], np.int32))
        case(handle, "isContourConvex", head, image,
             np.array([[1 if cv2.isContourConvex(contour) else 0]], np.int32))
        hull = cv2.convexHull(contour)
        case(handle, "convexHull", head, image,
             np.array([[int(p[0][0]), int(p[0][1])] for p in hull], np.int32).reshape(-1, 2))
        for eps in (1.0, 3.0):
            approx = cv2.approxPolyDP(contour, eps, True)
            case(handle, "approxPolyDP", f"{head} {eps}", image,
                 np.array([[int(p[0][0]), int(p[0][1])] for p in approx], np.int32).reshape(-1, 2))
        (cxr, cyr), (wr, hr), ar = cv2.minAreaRect(contour)
        case(handle, "minAreaRect", head, image,
             np.array([[cxr, cyr, wr, hr, ar]], np.float32))
        (ccx, ccy), rad = cv2.minEnclosingCircle(contour)
        case(handle, "minEnclosingCircle", head, image,
             np.array([[ccx, ccy, rad]], np.float32))
        rows, cols = image.shape
        grid = np.zeros((rows, cols), np.float32)
        for yy in range(rows):
            for xx in range(cols):
                grid[yy, xx] = cv2.pointPolygonTest(contour, (float(xx), float(yy)), True)
        case(handle, "pointPolygonTest", head, image, grid)
        # A round blob has no long axis, so neither fitEllipse's angle nor
        # fitLine's direction is determined for it and comparing them would be
        # comparing noise. Only shapes with a clear axis are recorded.
        spread = np.cov(contour.reshape(-1, 2).astype(np.float64).T)
        eigen = np.sort(np.linalg.eigvalsh(spread))
        elongated = eigen[0] > 1e-9 and eigen[1] / eigen[0] > 1.3
        if len(contour) >= 5 and elongated:
            (ecx, ecy), (ew, eh), ea = cv2.fitEllipse(contour)
            case(handle, "fitEllipse", head, image,
                 np.array([[ecx, ecy, ew, eh, ea]], np.float32))
            line = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01).reshape(1, 4)
            case(handle, "fitLine", head, image, line.astype(np.float32))


def write_segmentation(handle) -> None:
    gradient = np.zeros((20, 24), np.uint8)
    for y in range(20):
        for x in range(24):
            gradient[y, x] = (x * 6 + y * 3) % 256
    blocks = np.zeros((20, 24), np.uint8)
    blocks[2:9, 3:11] = 200
    blocks[12:18, 6:20] = 120
    blocks[4:7, 15:22] = 200

    for seed, lo, up, flags in (((5, 5), 0, 0, 4), ((5, 5), 20, 20, 4),
                                ((5, 5), 20, 20, 8), ((12, 3), 30, 30, 4)):
        for tag, image in (("gradient", gradient), ("blocks", blocks)):
            filled = image.copy()
            count, _, _, rect = cv2.floodFill(filled, None, seed, 255, lo, up, flags)
            case(handle, "floodFill", f"{tag} {seed[0]} {seed[1]} {lo} {up} {flags}", image, filled)
            case(handle, "floodFillStats", f"{tag} {seed[0]} {seed[1]} {lo} {up} {flags}", image,
                 np.array([[count, rect[0], rect[1], rect[2], rect[3]]], np.int32))

    rng = np.random.default_rng(91)
    speckle = (rng.random((18, 22)) < 0.45).astype(np.uint8) * 255
    for tag, image in (("blocks", blocks), ("speckle", speckle)):
        for conn in (4, 8):
            count, labels, stats, centroids = cv2.connectedComponentsWithStats(
                image, connectivity=conn)
            case(handle, "connectedComponents", f"{tag} {conn}", image, labels.astype(np.int32))
            case(handle, "connectedComponentsStats", f"{tag} {conn}", image, stats.astype(np.int32))
            case(handle, "connectedComponentsCentroids", f"{tag} {conn}", image,
                 centroids.astype(np.float32))

    for tag, image in (("blocks", blocks), ("speckle", speckle)):
        for dname, dist in (("L1", cv2.DIST_L1), ("L2", cv2.DIST_L2), ("C", cv2.DIST_C)):
            for msize in (3, 5):
                out = cv2.distanceTransform(image, dist, msize)
                case(handle, "distanceTransform", f"{tag} {dname} {msize}", image, out)
        case(handle, "integral", tag, image, cv2.integral(image).astype(np.float32))


def write_hist(handle) -> None:
    rng = np.random.default_rng(93)
    gray = rng.integers(20, 210, size=(16, 20), dtype=np.uint8)
    colour = rng.integers(0, 256, size=(16, 20, 3), dtype=np.uint8)
    mask = np.zeros((16, 20), np.uint8)
    mask[4:12, 5:16] = 255

    case(handle, "calcHist1D", "gray 256", gray,
         cv2.calcHist([gray], [0], None, [256], [0, 256]).reshape(256, 1))
    case(handle, "calcHist1D", "gray 32", gray,
         cv2.calcHist([gray], [0], None, [32], [0, 256]).reshape(32, 1))
    case(handle, "calcHist1DMasked", "gray 16", gray,
         cv2.calcHist([gray], [0], mask, [16], [0, 256]).reshape(16, 1))
    case(handle, "calcHist2D", "colour 8 8", colour,
         cv2.calcHist([colour], [0, 1], None, [8, 8], [0, 256, 0, 256]))
    case(handle, "equalizeHist", "gray", gray, cv2.equalizeHist(gray))

    flat = np.zeros((16, 20), np.uint8)
    flat[:] = 90
    case(handle, "equalizeHist", "flat", flat, cv2.equalizeHist(flat))
    dark = gray.copy()
    dark[dark > 120] = 120
    case(handle, "equalizeHist", "dark", dark, cv2.equalizeHist(dark))

    for clip in (2.0, 4.0, 40.0):
        for tiles in (2, 4):
            engine = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
            case(handle, "clahe", f"gray {clip} {tiles}", gray, engine.apply(gray))

    hist_a = cv2.calcHist([gray], [0], None, [32], [0, 256])
    hist_b = cv2.calcHist([np.roll(gray, 3, axis=1)], [0], None, [32], [0, 256])
    methods = {"CORREL": cv2.HISTCMP_CORREL, "CHISQR": cv2.HISTCMP_CHISQR,
               "INTERSECT": cv2.HISTCMP_INTERSECT,
               "BHATTACHARYYA": cv2.HISTCMP_BHATTACHARYYA,
               "CHISQR_ALT": cv2.HISTCMP_CHISQR_ALT, "KL_DIV": cv2.HISTCMP_KL_DIV}
    for name, method in methods.items():
        case(handle, "compareHist", name, gray,
             np.array([[cv2.compareHist(hist_a, hist_b, method)]], np.float32))

    hue = cv2.cvtColor(colour, cv2.COLOR_BGR2HSV)[:, :, 0]
    hue_hist = cv2.calcHist([hue], [0], None, [16], [0, 180])
    case(handle, "calcBackProject", "hue 16", hue,
         cv2.calcBackProject([hue], [0], hue_hist, [0, 180], 1).astype(np.uint8))


def write_features(handle) -> None:
    rng = np.random.default_rng(101)
    scene = rng.integers(0, 256, size=(24, 30), dtype=np.uint8)
    scene[8:14, 10:17] = np.array([[10, 200, 30, 40, 250, 60, 70],
                                   [80, 90, 100, 110, 120, 130, 140],
                                   [150, 160, 170, 180, 190, 200, 210],
                                   [20, 30, 40, 50, 60, 70, 80],
                                   [220, 230, 240, 250, 5, 15, 25],
                                   [35, 45, 55, 65, 75, 85, 95]], np.uint8)
    patch = scene[8:14, 10:17].copy()
    colour = rng.integers(0, 256, size=(24, 30, 3), dtype=np.uint8)
    colour_patch = colour[5:11, 7:15].copy()

    methods = {"SQDIFF": cv2.TM_SQDIFF, "SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED,
               "CCORR": cv2.TM_CCORR, "CCORR_NORMED": cv2.TM_CCORR_NORMED,
               "CCOEFF": cv2.TM_CCOEFF, "CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED}
    for name, method in methods.items():
        case(handle, "matchTemplate", f"gray {name}", scene,
             cv2.matchTemplate(scene, patch, method))
        case(handle, "matchTemplateColour", f"colour {name}", colour,
             cv2.matchTemplate(colour, colour_patch, method))

    shapes = np.zeros((26, 32), np.uint8)
    cv2.rectangle(shapes, (4, 4), (14, 16), 255, -1)
    cv2.rectangle(shapes, (18, 8), (28, 20), 255, 2)
    for block in (2, 3, 5):
        for aperture in (3, 5):
            case(handle, "cornerHarris", f"{block} {aperture}", shapes,
                 cv2.cornerHarris(shapes, block, aperture, 0.04))
            # cornerMinEigenVal's third positional argument is `dst`, not the
            # aperture, so the aperture has to be named or it silently stays 3.
            case(handle, "cornerMinEigenVal", f"{block} {aperture}", shapes,
                 cv2.cornerMinEigenVal(shapes, block, ksize=aperture))

    lines_image = np.zeros((40, 48), np.uint8)
    cv2.line(lines_image, (2, 5), (45, 5), 255, 1)
    cv2.line(lines_image, (6, 2), (6, 37), 255, 1)
    cv2.line(lines_image, (3, 36), (44, 9), 255, 1)
    for threshold in (18, 25):
        found = cv2.HoughLines(lines_image, 1, np.pi / 180, threshold)
        rows = [] if found is None else [[float(l[0][0]), float(l[0][1])] for l in found]
        case(handle, "houghLines", f"1 180 {threshold}", lines_image,
             np.array(rows, np.float32).reshape(-1, 2))


def write_features2d(handle) -> None:
    rng = np.random.default_rng(131)
    noisy = rng.integers(0, 256, size=(30, 36), dtype=np.uint8)
    shapes = np.zeros((30, 36), np.uint8)
    cv2.rectangle(shapes, (5, 5), (16, 18), 255, -1)
    cv2.rectangle(shapes, (21, 9), (32, 24), 200, -1)
    cv2.circle(shapes, (12, 24), 4, 120, -1)
    blurred = cv2.GaussianBlur(noisy, (5, 5), 0)

    for tag, image in (("noisy", noisy), ("shapes", shapes), ("blurred", blurred)):
        for threshold in (10, 20, 40):
            for nonmax in (True, False):
                detector = cv2.FastFeatureDetector_create(threshold, nonmax,
                                                          cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
                found = detector.detect(image, None)
                rows = [[k.pt[0], k.pt[1], k.response] for k in found]
                case(handle, "fast", f"{tag} {threshold} {1 if nonmax else 0}", image,
                     np.array(rows, np.float32).reshape(-1, 3))


# OpenCV 5 dropped CascadeClassifier, so there is no reference in cv2 to
# compare the cascade detector against. What follows is an independent
# implementation of the same published algorithm, written against numpy rather
# than against jaicv's kernels, and the cases it records check that jaicv's XML
# reading, integral arithmetic, scan order, and tree evaluation all agree with
# it. It is a weaker guarantee than the rest of this file and is labelled as
# such where it is read.
def cascade_reference_load(path):
    import xml.etree.ElementTree as ET
    root = ET.parse(path).getroot()
    node = root.find("cascade")
    width = int(node.findtext("width"))
    height = int(node.findtext("height"))
    features = []
    for entry in node.find("features"):
        rects = []
        for rect in entry.find("rects"):
            v = [float(t) for t in rect.text.split()]
            rects.append((int(v[0]), int(v[1]), int(v[2]), int(v[3]), v[4]))
        features.append(rects)
    stages = []
    for entry in node.find("stages"):
        weaks = []
        for weak in entry.find("weakClassifiers"):
            nodes = [float(t) for t in weak.findtext("internalNodes").split()]
            leaves = [float(t) for t in weak.findtext("leafValues").split()]
            weaks.append((nodes, leaves))
        stages.append((float(entry.findtext("stageThreshold")), weaks))
    return width, height, features, stages


def cascade_reference_detect(img, width, height, features, stages, scale_factor):
    import math
    found = []
    factor = 1.0
    while True:
        w = int(img.shape[1] / factor + 0.5)
        h = int(img.shape[0] / factor + 0.5)
        if w < width or h < height:
            break
        level = img if factor == 1.0 else cv2.resize(img, (w, h))
        ii = cv2.integral(level.astype(np.float64))
        ii2 = cv2.integral(level.astype(np.float64) ** 2)

        def rs(t, x, y, rw, rh):
            return t[y + rh, x + rw] - t[y, x + rw] - t[y + rh, x] + t[y, x]

        step = 1 if factor > 2.0 else 2
        sw = int(width * factor + 0.5)
        sh = int(height * factor + 0.5)
        for y in range(0, h - height + 1, step):
            for x in range(0, w - width + 1, step):
                area = (width - 2) * (height - 2)
                s = rs(ii, x + 1, y + 1, width - 2, height - 2)
                q = rs(ii2, x + 1, y + 1, width - 2, height - 2)
                nf = area * q - s * s
                if nf <= 0:
                    continue
                inv = 1.0 / math.sqrt(nf)
                ok = True
                for threshold, weaks in stages:
                    vote = 0.0
                    for nodes, leaves in weaks:
                        left, right, fi, thr = int(nodes[0]), int(nodes[1]), int(nodes[2]), nodes[3]
                        val = sum(wt * rs(ii, x + rx, y + ry, rw, rh)
                                  for (rx, ry, rw, rh, wt) in features[fi]) * inv
                        vote += leaves[-(left if val < thr else right)]
                    if vote < threshold:
                        ok = False
                        break
                if ok:
                    found.append([int(x * factor + 0.5), int(y * factor + 0.5), sw, sh])
        factor *= scale_factor
    return found


def write_cascade(handle) -> None:
    rng = np.random.default_rng(5)
    scene = np.full((60, 80), 40, np.uint8)
    scene = np.clip(scene.astype(np.int16) + rng.integers(-8, 9, scene.shape), 0, 255).astype(np.uint8)
    for (cx, cy, r) in ((20, 20, 4), (55, 38, 4), (66, 12, 4)):
        cv2.rectangle(scene, (cx - r, cy - r), (cx + r, cy + r), 220, -1)

    path = Path(__file__).resolve().parent / "synthetic_cascade.xml"
    width, height, features, stages = cascade_reference_load(str(path))
    for factor in (1.1, 1.3):
        rows = cascade_reference_detect(scene, width, height, features, stages, factor)
        case(handle, "cascadeDetect", f"{factor}", scene,
             np.array(rows, np.int32).reshape(-1, 4))


def write_calib(handle) -> None:
    rng = np.random.default_rng(151)
    blank = np.zeros((4, 4), np.uint8)

    for rv in ([0.1, -0.2, 0.35], [0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.3, 0.4, 0.5]):
        vector = np.array(rv, np.float64).reshape(3, 1)
        matrix, _ = cv2.Rodrigues(vector)
        case(handle, "rodriguesForward", " ".join(repr(v) for v in rv), blank,
             matrix.astype(np.float32))
        back, _ = cv2.Rodrigues(matrix)
        case(handle, "rodriguesBack", " ".join(repr(v) for v in rv), blank,
             back.reshape(1, 3).astype(np.float32))

    src = np.array([[10.0, 12.0], [90.0, 8.0], [95.0, 70.0], [5.0, 75.0],
                    [40.0, 30.0], [60.0, 55.0], [20.0, 60.0]], np.float32)
    h_true = np.array([[1.05, 0.08, -6.0], [-0.04, 0.96, 4.0], [0.0002, -0.0003, 1.0]])
    ones = np.hstack([src, np.ones((len(src), 1), np.float32)])
    mapped = (h_true @ ones.T).T
    dst = (mapped[:, :2] / mapped[:, 2:3]).astype(np.float32)
    h, _ = cv2.findHomography(src, dst, 0)
    case(handle, "findHomography", "direct", blank, h.astype(np.float32))

    affine_true = np.array([[1.1, -0.15, 7.0], [0.2, 0.95, -4.0]])
    affine_dst = (affine_true @ ones.T).T.astype(np.float32)
    a, _ = cv2.estimateAffine2D(src, affine_dst, method=cv2.LMEDS)
    case(handle, "estimateAffine2D", "direct", blank, a.astype(np.float32))
    similarity = np.array([[0.9, -0.3, 5.0], [0.3, 0.9, -2.0]])
    sim_dst = (similarity @ ones.T).T.astype(np.float32)
    p2, _ = cv2.estimateAffinePartial2D(src, sim_dst, method=cv2.LMEDS)
    case(handle, "estimateAffinePartial2D", "direct", blank, p2.astype(np.float32))

    camera = np.array([[520.0, 0.0, 320.0], [0.0, 515.0, 240.0], [0.0, 0.0, 1.0]])
    for coeffs in ([0.0, 0.0, 0.0, 0.0, 0.0], [-0.28, 0.09, 0.001, -0.002, 0.0]):
        dist = np.array(coeffs, np.float64)
        objects = rng.uniform(-1.0, 1.0, size=(12, 3))
        objects[:, 2] += 6.0
        rvec = np.array([0.05, -0.12, 0.2], np.float64).reshape(3, 1)
        tvec = np.array([0.3, -0.2, 4.0], np.float64).reshape(3, 1)
        projected, _ = cv2.projectPoints(objects, rvec, tvec, camera, dist)
        tag = "clean" if coeffs[0] == 0.0 else "distorted"
        case(handle, "projectPoints", tag,
             objects.reshape(-1, 3).astype(np.float32),
             projected.reshape(-1, 2).astype(np.float32))

        grid = np.array([[x * 40.0 + 20.0, y * 40.0 + 20.0]
                         for y in range(6) for x in range(8)], np.float64)
        undone = cv2.undistortPoints(grid.reshape(-1, 1, 2), camera, dist)
        case(handle, "undistortPoints", tag, grid.reshape(-1, 2).astype(np.float32),
             undone.reshape(-1, 2).astype(np.float32))

    proj1 = camera @ np.hstack([np.eye(3), np.zeros((3, 1))])
    r2, _ = cv2.Rodrigues(np.array([0.02, 0.3, -0.05]))
    proj2 = camera @ np.hstack([r2, np.array([[-1.5], [0.1], [0.2]])])
    world = rng.uniform(-1.0, 1.0, size=(10, 3))
    world[:, 2] += 7.0
    p1p = (proj1 @ np.hstack([world, np.ones((10, 1))]).T).T
    p2p = (proj2 @ np.hstack([world, np.ones((10, 1))]).T).T
    x1 = (p1p[:, :2] / p1p[:, 2:3]).astype(np.float32)
    x2 = (p2p[:, :2] / p2p[:, 2:3]).astype(np.float32)
    tri = cv2.triangulatePoints(proj1, proj2, x1.T, x2.T).T
    tri = tri / tri[:, 3:4]
    case(handle, "triangulatePoints", "pair", world.reshape(-1, 3).astype(np.float32),
         tri[:, :3].astype(np.float32))


def chessboard_image(squares_x=9, squares_y=7, side=30, margin=20):
    inner = np.zeros((squares_y * side, squares_x * side), np.uint8)
    for r in range(squares_y):
        for c in range(squares_x):
            if (r + c) % 2 == 0:
                inner[r * side:(r + 1) * side, c * side:(c + 1) * side] = 255
    board = np.full((squares_y * side + 2 * margin, squares_x * side + 2 * margin), 255, np.uint8)
    board[margin:margin + squares_y * side, margin:margin + squares_x * side] = inner
    return board


def write_calibrate(handle) -> None:
    board = chessboard_image()
    found, corners = cv2.findChessboardCorners(board, (8, 6))
    assert found
    case(handle, "findChessboardCorners", "synthetic", board,
         corners.reshape(-1, 2).astype(np.float32))

    camera = np.array([[520.0, 0.0, 320.0], [0.0, 515.0, 240.0], [0.0, 0.0, 1.0]])
    grid = np.array([[float(x), float(y), 0.0] for y in range(6) for x in range(8)])
    poses = [([0.02, -0.05, 0.01], [-3.5, -2.5, 22.0]),
             ([0.25, 0.12, -0.08], [-4.0, -2.0, 25.0]),
             ([-0.18, 0.3, 0.05], [-3.0, -3.0, 20.0]),
             ([0.1, -0.28, 0.12], [-3.6, -2.4, 24.0]),
             ([-0.3, -0.1, -0.15], [-3.2, -2.8, 21.0])]
    for coeffs, tag in (([0.0] * 5, "clean"), ([-0.2, 0.05, 0.0005, -0.001, 0.0], "distorted")):
        dist = np.array(coeffs, np.float64)
        object_points = []
        image_points = []
        for rv, tv in poses:
            projected, _ = cv2.projectPoints(grid, np.array(rv), np.array(tv), camera, dist)
            object_points.append(grid.astype(np.float32))
            image_points.append(projected.reshape(-1, 2).astype(np.float32))

        ok, rvec, tvec = cv2.solvePnP(grid, image_points[1], camera, dist)
        assert ok
        case(handle, "solvePnP", tag, np.zeros((4, 4), np.uint8),
             np.concatenate([rvec.reshape(1, 3), tvec.reshape(1, 3)]).astype(np.float32))

        err, k, d, _rv, _tv = cv2.calibrateCamera(
            object_points, image_points, (640, 480), None, None)
        case(handle, "calibrateCamera", tag, np.zeros((4, 4), np.uint8),
             np.concatenate([k.reshape(1, 9), d.reshape(1, -1)[:, :5]], axis=1).astype(np.float32))

    scene = chessboard_image(6, 5, 40, 30)
    scene = cv2.resize(scene, (200, 160))
    small_camera = np.array([[150.0, 0.0, 100.0], [0.0, 150.0, 80.0], [0.0, 0.0, 1.0]])
    small_dist = np.array([-0.25, 0.08, 0.001, -0.0015, 0.0])
    case(handle, "undistort", "board", scene,
         cv2.undistort(scene, small_camera, small_dist))


def write_transform(handle) -> None:
    rng = np.random.default_rng(171)
    for rows, cols in ((1, 8), (1, 12), (4, 6), (5, 5), (6, 10), (1, 7)):
        signal = rng.standard_normal((rows, cols)).astype(np.float32)
        spectrum = cv2.dft(signal, flags=cv2.DFT_COMPLEX_OUTPUT)
        case(handle, "dft", f"{rows} {cols}", signal, spectrum)
        back = cv2.idft(spectrum, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        case(handle, "idft", f"{rows} {cols}", signal, back.reshape(rows, cols).astype(np.float32))

    for rows, cols in ((4, 8), (6, 6)):
        signal = rng.standard_normal((rows, cols)).astype(np.float32)
        case(handle, "dct", f"{rows} {cols}", signal, cv2.dct(signal))

    blank = np.zeros((4, 4), np.uint8)
    sizes = np.array([[float(cv2.getOptimalDFTSize(n)) for n in range(1, 41)]], np.float32)
    case(handle, "getOptimalDFTSize", "1to40", blank, sizes)

    samples = np.vstack([
        rng.normal(0.0, 0.35, size=(30, 2)) + np.array([0.0, 0.0]),
        rng.normal(0.0, 0.35, size=(30, 2)) + np.array([5.0, 5.0]),
        rng.normal(0.0, 0.35, size=(30, 2)) + np.array([5.0, 0.0]),
    ]).astype(np.float32)
    mean, vectors = cv2.PCACompute(samples, mean=None)
    case(handle, "pcaMean", "clusters", samples, mean.astype(np.float32))
    # An eigenvector is only defined up to its sign, so the recorded value is
    # the outer product, which is not.
    outer = vectors.T @ vectors
    case(handle, "pcaBasis", "clusters", samples, outer.astype(np.float32))


def write_dnn(handle) -> None:
    rng = np.random.default_rng(191)
    image = rng.integers(0, 256, size=(20, 24, 3), dtype=np.uint8)
    for scale, size, mean, swap in ((1.0, (0, 0), (0, 0, 0), False),
                                    (1.0 / 255.0, (12, 10), (0, 0, 0), False),
                                    (1.0, (16, 16), (104.0, 117.0, 123.0), True),
                                    (0.5, (8, 8), (10.0, 20.0, 30.0), False)):
        blob = cv2.dnn.blobFromImage(image, scale, size, mean, swap, False)
        rows = blob.shape[1]
        cols = blob.shape[2] * blob.shape[3]
        tag = f"{scale} {size[0]} {size[1]} {mean[0]} {mean[1]} {mean[2]} {1 if swap else 0}"
        case(handle, "blobFromImage", tag, image, blob.reshape(rows, cols).astype(np.float32))

    boxes = [[10, 10, 30, 30], [12, 12, 30, 30], [60, 60, 20, 20],
             [61, 59, 22, 21], [100, 10, 15, 40], [11, 9, 31, 32]]
    scores = [0.9, 0.75, 0.8, 0.6, 0.55, 0.5]
    for score_threshold, nms_threshold in ((0.0, 0.3), (0.6, 0.5), (0.0, 0.1)):
        kept = cv2.dnn.NMSBoxes(boxes, scores, score_threshold, nms_threshold)
        rows = np.array(list(kept), np.int32).reshape(-1, 1)
        case(handle, "nmsBoxes", f"{score_threshold} {nms_threshold}",
             np.zeros((4, 4), np.uint8), rows)


def main() -> int:
    with OUT.open("w") as handle:
        write_colour(handle)
        write_geometric(handle)
        write_filter(handle)
        write_threshold(handle)
        write_morph(handle)
        write_edges(handle)
        write_drawing(handle)
        write_contours(handle)
        write_shape(handle)
        write_segmentation(handle)
        write_hist(handle)
        write_features(handle)
        write_features2d(handle)
        write_cascade(handle)
        write_calib(handle)
        write_calibrate(handle)
        write_transform(handle)
        write_dnn(handle)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
