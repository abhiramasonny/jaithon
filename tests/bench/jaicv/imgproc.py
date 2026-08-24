"""OpenCV peer for imgproc.jai: the same operations on the same picture.

Kept in step with imgproc.jai deliberately -- same size, same kernel sizes,
same structuring elements, same float32 input, same repetition count. OpenCV
runs these on the CPU across all cores, which is the comparison worth making:
it is what a pipeline uses today.
"""
import os
import sys
import time

import cv2
import numpy as np

WIDTH = 1920
HEIGHT = 1080
WARMUP = 40
WARM_SECONDS = 0.25


def picture(channels: int) -> np.ndarray:
    count = WIDTH * HEIGHT * channels
    values = (np.arange(count, dtype=np.int64) * 37 % 251).astype(np.float32) / 251.0
    if channels == 1:
        return values.reshape(HEIGHT, WIDTH)
    return values.reshape(HEIGHT, WIDTH, channels)


# Which row this process was asked for, or None for all of them. See the note
# on `_only` in imgproc.jai: one row a process, the two sides turn about, is
# the only way the ratio means anything on a laptop.
_only = None
_listing = False


def wanted(name: str) -> bool:
    if _listing:
        print(name)
        return False
    return _only is None or _only == name


def run(name: str, repeats: int, work) -> None:
    if not wanted(name):
        return
    # Enough repetitions to settle, and no more. See the note on `warm` in
    # imgproc.jai: a row a process pays the warm-up each time, and forty calls
    # of a row that takes half a second is nearly twenty seconds of warming for
    # a tenth of a second of measurement.
    opening = time.perf_counter()
    for round_index in range(WARMUP):
        work()
        if round_index >= 7 and time.perf_counter() - opening >= WARM_SECONDS:
            break
    started = time.perf_counter()
    last = None
    for _ in range(repeats):
        last = work()
    total = time.perf_counter() - started
    flat = np.asarray(last, dtype=object).ravel() if last is not None else []
    first = flat[0] if len(flat) else None
    check = "ok" if first is not None and np.isfinite(np.asarray(first, dtype=np.float64)) else "invalid"
    pixels = repeats * WIDTH * HEIGHT
    print(f"jtb\t{name}\t{total:.9f}\t0\t{pixels}\t{check}\t—")


def byte_picture(rows: int, cols: int):
    """The same frame as bytes. See the note in imgproc.jai."""
    flat = (np.arange(rows * cols, dtype=np.int64) * 37 % 251).astype(np.uint8)
    return flat.reshape(rows, cols)


def blobby(rows: int, cols: int):
    """Regions rather than a gradient, so labelling has something to find."""
    y, x = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    cell = (x // 23 + y // 19) % 3
    return np.where(cell == 0, 255, 0).astype(np.uint8)


def picture_at(rows: int, cols: int):
    """Bytes: the peer's pyramid builder accepts nothing else."""
    y, x = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    return (((x * 7 + y * 13) % 211) + ((x * y) % 37)).astype(np.uint8)


def shifted_by(source, dx: int, dy: int):
    rows, cols = source.shape
    ys = np.clip(np.arange(rows) - dy, 0, rows - 1)
    xs = np.clip(np.arange(cols) - dx, 0, cols - 1)
    return source[ys][:, xs].copy()


def main() -> int:
    global _only, _listing
    if len(sys.argv) > 1:
        if sys.argv[1] == "list":
            _listing = True
        else:
            _only = sys.argv[1]
    level = os.environ.get("BENCH_LEVEL", "hard")
    repeats = 8 if level == "easy" else 16 if level == "medium" else 30
    grey = picture(1)
    colour = picture(3)
    small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    wide = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    square = ((np.arange(1024 * 1024, dtype=np.int64) * 37 % 251).astype(np.float32) / 251.0).reshape(1024, 1024)

    run("cvt-gray", repeats, lambda: cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY))
    run("blur-3", repeats, lambda: cv2.GaussianBlur(grey, (3, 3), 0))
    run("blur-5", repeats, lambda: cv2.GaussianBlur(grey, (5, 5), 0))
    run("blur-15", repeats, lambda: cv2.GaussianBlur(grey, (15, 15), 0))
    run("box-9", repeats, lambda: cv2.blur(grey, (9, 9)))
    run("sobel", repeats, lambda: cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3))
    run("erode-3", repeats, lambda: cv2.erode(grey, small))
    run("erode-15", repeats, lambda: cv2.erode(grey, wide))
    run("resize-half", repeats, lambda: cv2.resize(colour, (WIDTH // 2, HEIGHT // 2)))
    run("pyr-down", repeats, lambda: cv2.pyrDown(grey))
    run("dft", repeats, lambda: cv2.dft(square, flags=cv2.DFT_COMPLEX_OUTPUT))
    run("min-max", repeats, lambda: cv2.minMaxLoc(grey))
    run("norm", repeats, lambda: cv2.norm(grey))
    run("sum", repeats, lambda: cv2.sumElems(grey))

    # The rest of a real pipeline; see the note in imgproc.jai.
    bytes_frame = byte_picture(HEIGHT, WIDTH)
    regions = blobby(HEIGHT, WIDTH)
    run("canny", repeats, lambda: cv2.Canny(bytes_frame, 60, 140))
    run("connected-components", repeats, lambda: cv2.connectedComponents(regions))
    run("distance-transform", repeats, lambda: cv2.distanceTransform(regions, cv2.DIST_L2, 3))
    run("integral", repeats, lambda: cv2.integral(grey))
    run(
        "good-features",
        repeats,
        lambda: cv2.goodFeaturesToTrack(grey, 200, 0.01, 6),
    )

    # The operations a pipeline reaches for around the ones above. See the
    # note in imgproc.jai.
    turn = cv2.getRotationMatrix2D((WIDTH * 0.5, HEIGHT * 0.5), 30.0, 1.0)
    grid_y, grid_x = np.mgrid[0:HEIGHT, 0:WIDTH]
    map_x = ((grid_x + 7) % WIDTH).astype(np.float32)
    map_y = ((grid_y + 5) % HEIGHT).astype(np.float32)
    cross = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    run("threshold", repeats, lambda: cv2.threshold(grey, 0.5, 1.0, cv2.THRESH_BINARY)[1])
    run(
        "adaptive-threshold",
        repeats,
        lambda: cv2.adaptiveThreshold(
            bytes_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
        ),
    )
    run("equalize-hist", repeats, lambda: cv2.equalizeHist(bytes_frame))
    run("morph-open-5", repeats, lambda: cv2.morphologyEx(grey, cv2.MORPH_OPEN, cross))
    run("morph-gradient-5", repeats, lambda: cv2.morphologyEx(grey, cv2.MORPH_GRADIENT, cross))
    run("warp-affine", repeats, lambda: cv2.warpAffine(grey, turn, (WIDTH, HEIGHT)))
    run("remap", repeats, lambda: cv2.remap(grey, map_x, map_y, cv2.INTER_LINEAR))

    # Three colour channels, and eight-bit because the peer's filter takes
    # nothing else. See the note in imgproc.jai.
    photo = (picture(3) * 255.0).astype(np.uint8)
    run("edge-preserving", max(2, repeats // 10), lambda: cv2.edgePreservingFilter(photo))
    run("detail-enhance", max(2, repeats // 10), lambda: cv2.detailEnhance(photo))

    # Contours and the shapes fitted to them. See the note in imgproc.jai.
    shapes, _tree = cv2.findContours(
        bytes_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    single = shapes[0]
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    run(
        "find-contours",
        max(2, repeats // 10),
        lambda: float(len(cv2.findContours(bytes_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])),
    )
    run(
        "find-contours-tree",
        max(2, repeats // 10),
        lambda: float(len(cv2.findContours(bytes_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0])),
    )
    run("convex-hull", repeats, lambda: float(sum(len(cv2.convexHull(s)) for s in shapes)))
    run("min-area-rect", repeats, lambda: float(sum(cv2.minAreaRect(s)[2] for s in shapes)))
    run(
        "approx-poly-dp",
        repeats,
        lambda: float(sum(len(cv2.approxPolyDP(s, cv2.arcLength(s, True) * 0.02, True)) for s in shapes)),
    )
    run("min-enclosing-circle", repeats, lambda: float(cv2.minEnclosingCircle(single)[1]))
    run(
        "point-polygon-test",
        repeats,
        lambda: float(
            sum(
                cv2.pointPolygonTest(single, (float(k % 640), float(k % 480)), True)
                for k in range(1000)
            )
        ),
    )
    run(
        "draw-contours",
        repeats,
        lambda: cv2.drawContours(canvas, shapes, -1, (255, 255, 255), 2),
    )

    small_rows, small_cols = 480, 640
    damaged = byte_picture(small_rows, small_cols)
    holes = blobby(small_rows, small_cols)
    run(
        "inpaint",
        max(2, repeats // 10),
        lambda: cv2.inpaint(damaged, holes, 3, cv2.INPAINT_TELEA),
    )

    first = picture_at(small_rows, small_cols)
    second = shifted_by(first, 2, 1)
    tracked = np.array(
        [[[40 + index * 13 % 540, 40 + index * 7 % 380]] for index in range(200)],
        dtype=np.float32,
    )
    run(
        "optical-flow",
        max(2, repeats // 10),
        # Just the first landed x, so the checker has one number to look at
        # rather than a tuple of arrays -- which is what the jai side returns
        # from this row too.
        lambda: float(cv2.calcOpticalFlowPyrLK(first, second, tracked, None)[0][0][0][0]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
