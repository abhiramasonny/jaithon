#: The image operations a pipeline actually spends its time in, at 1080p.
#:
#: Against OpenCV rather than against a remembered number, because "fast" for a
#: convolution only means anything next to the library everyone reaches for
#: first. The sizes are the ones people use: a 3x3 and a 5x5 blur because that
#: is what smoothing means in practice, a 15x15 because separability stops
#: being a detail there, and a rectangle erode because that is the default
#: structuring element.
import jaicv as cv
import std.gpu as gpu
import std.math
from jaicv import Mat, Size
from jaicv.imgproc.morph import erode, get_structuring_element, MORPH_RECT
from std.os import env_or
from std.time import Instant

let WIDTH = 1920
let HEIGHT = 1080

#: Enough repetitions that the clock is at its working speed before the timed
#: run starts; see tests/bench/jaitensor/gemm.jai for why that matters.
let WARMUP = 12

fn picture(channels: int) -> Mat {
    var values: list[float] = []
    for index in 0..WIDTH * HEIGHT * channels {
        values.push(float(index * 37 % 251) / 251.0)
    }
    let kind = if channels == 1 { cv.CV_32FC1 } else { cv.CV_32FC3 }
    return Mat.from_list(HEIGHT, WIDTH, kind, values)
}

fn run(name: str, repeats: int, work: fn() -> Mat) -> void {
    for _warm in 0..WARMUP {
        let m = work()
        m.release()
    }
    gpu.synchronize()
    let started = Instant.now()
    var live: list[Mat] = []
    for _r in 0..repeats { live.push(work()) }
    gpu.synchronize()
    let seconds = started.elapsed().to_seconds()
    var check = "ok"
    let sample = live[live.len() - 1].to_list()
    if sample.len() == 0 or not math.is_finite(sample[0]) { check = "invalid" }
    for m in live { m.release() }
    let pixels = repeats * WIDTH * HEIGHT
    print(f"jtb\t{name}\t{seconds:.9f}\t0\t{pixels}\t{check}\t—")
}

fn main() -> int {
    if not gpu.is_available() { throw RuntimeError("bench needs a GPU") }
    let level = env_or("BENCH_LEVEL", "hard")
    let repeats = if level == "easy" { 8 } elif level == "medium" { 16 } else { 30 }
    let grey = picture(1)
    let colour = picture(3)
    let small = get_structuring_element(MORPH_RECT, Size(3, 3))
    let wide = get_structuring_element(MORPH_RECT, Size(15, 15))

    run("cvt-gray", repeats, || cv.cvt_color(colour, cv.COLOR_BGR2GRAY))
    run("blur-3", repeats, || cv.gaussian_blur(grey, Size(3, 3), 0.0))
    run("blur-5", repeats, || cv.gaussian_blur(grey, Size(5, 5), 0.0))
    run("blur-15", repeats, || cv.gaussian_blur(grey, Size(15, 15), 0.0))
    run("box-9", repeats, || cv.blur(grey, Size(9, 9)))
    run("sobel", repeats, || cv.sobel(grey, cv.CV_32F, 1, 0, ksize: 3))
    run("erode-3", repeats, || erode(grey, small))
    run("erode-15", repeats, || erode(grey, wide))
    run("resize-half", repeats, || cv.resize(colour, Size(WIDTH // 2, HEIGHT // 2)))
    run("pyr-down", repeats, || cv.pyr_down(grey))
    return 0
}
