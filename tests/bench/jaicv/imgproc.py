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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
