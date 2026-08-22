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
from jaicv.core.stats import min_max_loc, norm, sum_mat
from jaicv.imgproc.morph import erode, get_structuring_element, MORPH_RECT
from std.os import env_or
from std.time import Instant

let WIDTH = 1920
let HEIGHT = 1080

#: Enough repetitions that the clock is at its working speed before the timed
#: run starts; see tests/bench/jaitensor/gemm.jai for why that matters.
#:
#: Forty, not a dozen. The device takes about that many dispatches to reach its
#: working clock, and at twelve the first operation measured reads three times
#: slow -- which is a property of the harness, not of the operation. The peer
#: warms the same number of times; a CPU has no ramp to climb, so it costs
#: nothing there and keeps the two sides honest.
let WARMUP = 40

fn picture(channels: int) -> Mat {
    var values: list[float] = []
    for index in 0..WIDTH * HEIGHT * channels {
        values.push(float(index * 37 % 251) / 251.0)
    }
    let kind = if channels == 1 { cv.CV_32FC1 } else { cv.CV_32FC3 }
    return Mat.from_list(HEIGHT, WIDTH, kind, values)
}

#: The same timing for the reductions, which hand back a number rather than a
#: picture and so have nothing to keep alive.
fn run_scalar(name: str, repeats: int, work: fn() -> float) -> void {
    for _warm in 0..WARMUP {
        let _ = work()
    }
    gpu.synchronize()
    let started = Instant.now()
    var last = 0.0
    for _r in 0..repeats { last = work() }
    gpu.synchronize()
    let seconds = started.elapsed().to_seconds()
    let check = if math.is_finite(last) { "ok" } else { "invalid" }
    print(f"jtb\t{name}\t{seconds:.9f}\t0\t{repeats * WIDTH * HEIGHT}\t{check}\t—")
}

fn run(name: str, repeats: int, work: fn() -> Mat) -> void {
    for _warm in 0..WARMUP {
        let m = work()
        m.release()
    }
    gpu.synchronize()
    let started = Instant.now()
    #: The latest result stays alive so the work cannot be elided, and every
    #: earlier one goes back immediately. Holding all thirty means a quarter of
    #: a gigabyte outstanding and an allocator measured instead of an operation
    #: -- and the peer holds one, so keeping them all made the two sides answer
    #: different questions.
    var last = work()
    for _r in 1..repeats {
        let made = work()
        last.release()
        last = made
    }
    gpu.synchronize()
    let seconds = started.elapsed().to_seconds()
    var check = "ok"
    let sample = last.to_list()
    if sample.len() == 0 or not math.is_finite(sample[0]) { check = "invalid" }
    last.release()
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
    #: A power of two a side, which is what a transform is given in practice.
    var square_values: list[float] = []
    for index in 0..1024 * 1024 { square_values.push(float(index * 37 % 251) / 251.0) }
    let square = Mat.from_list(1024, 1024, cv.CV_32FC1, square_values)
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
    #: A transform of the whole frame. Here because it used to run on the host
    #: and cost seconds, which no benchmark would have noticed.
    run("dft", repeats, || cv.dft(square))
    #: Reductions to a single number. They are here because the fold used to
    #: happen on the host, and because a benchmark of only elementwise work
    #: says nothing about the shape that gets wrong.
    run_scalar("min-max", repeats, || min_max_loc(grey)[0])
    run_scalar("norm", repeats, || norm(grey))
    run_scalar("sum", repeats, || sum_mat(grey)[0])
    return 0
}
