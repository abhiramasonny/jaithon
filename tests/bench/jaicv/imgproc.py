"""OpenCV peer for imgproc.jai: the same operations on the same picture.

Kept in step with imgproc.jai deliberately -- same size, same kernel sizes,
same structuring elements, same float32 input, same repetition count. OpenCV
runs these on the CPU across all cores, which is the comparison worth making:
it is what a pipeline uses today.
"""
import os
import time

import cv2
import numpy as np

WIDTH = 1920
HEIGHT = 1080
WARMUP = 40


def picture(channels: int) -> np.ndarray:
    count = WIDTH * HEIGHT * channels
    values = (np.arange(count, dtype=np.int64) * 37 % 251).astype(np.float32) / 251.0
    if channels == 1:
        return values.reshape(HEIGHT, WIDTH)
    return values.reshape(HEIGHT, WIDTH, channels)


def run(name: str, repeats: int, work) -> None:
    for _ in range(WARMUP):
        work()
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
    run(
        "good-features",
        repeats,
        lambda: cv2.goodFeaturesToTrack(grey, 200, 0.01, 6),
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
