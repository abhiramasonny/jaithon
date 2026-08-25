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
from jaicv.core.types import Point2f, Scalar
from jaicv.imgproc.contours import CHAIN_APPROX_SIMPLE, RETR_EXTERNAL, RETR_TREE, draw_contours, find_contours
from jaicv.imgproc.geometric import get_rotation_matrix_2d, remap, warp_affine
from jaicv.imgproc.hist import equalize_hist
from jaicv.imgproc.shape import approx_poly_dp, arc_length, convex_hull, min_area_rect, min_enclosing_circle, point_polygon_test
from jaicv.imgproc.morph import erode, get_structuring_element, morphology_ex, MORPH_RECT
from jaicv.imgproc.segmentation import connected_components, distance_transform, integral
from jaicv.imgproc.threshold import adaptive_threshold, threshold
from jaicv.photo.denoise import detail_enhance, edge_preserving_filter, inpaint
from jaicv.video.optflow import calc_optical_flow_pyr_lk
from std.os import argv, env_or
from std.time import Instant

let WIDTH = 1920
let HEIGHT = 1080

#: How long the device must have been working before a row is timed.
#:
#: A FLOOR, not a cap. This was forty rounds bounded above by a quarter second,
#: on the reading that the clock climbs with dispatch count -- and forty
#: dispatches of a fast row is seven milliseconds of work, which does not climb
#: it at all. Every row here runs in its own process and therefore starts cold,
#: so every fast row was timed on a ramping clock: `blur-3` read 0.4476 ms
#: under the old warm-up and 0.1854 ms under the same protocol once the device
#: was hot, and the suite reported the first as a loss to OpenCV. Warming to a
#: wall-clock floor instead fixes it for the fast rows and costs the slow ones
#: nothing, since they pass the floor in a handful of calls.
#:
#: The peer warms the same way. A CPU has no ramp to climb, so it costs a
#: little time there and keeps the two sides answering the same question.
let WARM_SECONDS = 0.25

#: A stop for the warm-up, so a row that is somehow instant cannot spin here.
let WARM_MAX_ROUNDS = 20000

#: The same frame as bytes, which is what an edge detector and an inpainter
#: are given in practice and what OpenCV is fastest on.
fn byte_picture(rows: int, cols: int) -> Mat {
    var values: list[float] = []
    for index in 0..rows * cols { values.push(float(index * 37 % 251)) }
    return Mat.from_list(rows, cols, cv.CV_8UC1, values)
}

#: A frame with regions in it rather than a gradient, so that labelling and a
#: distance transform have something to find.
fn blobby(rows: int, cols: int) -> Mat {
    var values: list[float] = []
    for index in 0..rows * cols {
        let y = index // cols
        let x = index - y * cols
        let cell = (x // 23 + y // 19) % 3
        values.push(if cell == 0 { 255.0 } else { 0.0 })
    }
    return Mat.from_list(rows, cols, cv.CV_8UC1, values)
}

#: A textured single-channel frame of any size, for the pair the tracker
#: follows points through. Bytes, because the peer's pyramid builder accepts
#: nothing else; the values fit either way.
fn picture_at(rows: int, cols: int) -> Mat {
    var values: list[float] = []
    for index in 0..rows * cols {
        let y = index // cols
        let x = index - y * cols
        values.push(float((x * 7 + y * 13) % 211) + float(x * y % 37))
    }
    return Mat.from_list(rows, cols, cv.CV_32FC1, values)
}

#: The same frame moved, which is what the tracker has to recover.
fn shifted_by(source: Mat, rows: int, cols: int, dx: int, dy: int) -> Mat {
    let held = source.values()
    var values: list[float] = []
    for index in 0..rows * cols {
        let y = index // cols
        let x = index - y * cols
        values.push(held[max(0, y - dy) * cols + max(0, x - dx)])
    }
    return Mat.from_list(rows, cols, cv.CV_8UC1, values)
}

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
#: Which row this process was asked for, or empty for all of them.
#:
#: The suite used to run every row of our side three times and then every row
#: of the peer's three times. That is not a fair comparison on a laptop: by the
#: time the later rows came up, our side had held the GPU flat out for a minute
#: and the peer -- which runs on the CPU and leaves the device cool -- reached
#: the same rows on a cold one. good-features read 1.32x when it was the last
#: row of twenty-one and 0.75x when nine more rows were added in front of it,
#: with nothing about the code changed in between. One row a process, the two
#: sides turn about, is the only way the ratio means anything.
var _only = ""
var _listing = false

fn wanted(name: str) -> bool {
    if _listing {
        print(name)
        return false
    }
    return _only == "" or _only == name
}

fn run_scalar(name: str, repeats: int, work: fn() -> float) -> void {
    if not wanted(name) { return }
    let opening = Instant.now()
    for round in 0..WARM_MAX_ROUNDS {
        let _ = work()
        if round % 4 == 3 {
            gpu.synchronize()
            if opening.elapsed().to_seconds() >= WARM_SECONDS { break }
        }
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

#: Enough dispatches to bring the clock up, and no more.
#:
#: Forty was right when every row ran in one process and the warm-up was paid
#: once. A row a process pays it each time, and forty calls of a row that takes
#: forty milliseconds is a second and a half of warming for a tenth of a second
#: of measurement. A slow row does a dozen dispatches a call, so the clock is up
#: long before the count is.
fn warm(work: fn() -> Mat) -> void {
    let opening = Instant.now()
    for round in 0..WARM_MAX_ROUNDS {
        let m = work()
        m.release()
        if round % 4 == 3 {
            gpu.synchronize()
            if opening.elapsed().to_seconds() >= WARM_SECONDS { break }
        }
    }
    gpu.synchronize()
}

fn run(name: str, repeats: int, work: fn() -> Mat) -> void {
    if not wanted(name) { return }
    warm(work)
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
    let given = argv()
    if given.len() > 1 {
        if given[1] == "list" { _listing = true } else { _only = given[1] }
    }
    if not _listing and not gpu.is_available() { throw RuntimeError("bench needs a GPU") }
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

    #: The rest of a real pipeline. These were all host loops until recently
    #: -- inpainting a 640 by 480 frame took a second -- and none of them was
    #: measured against anything, so nothing would have said if they went
    #: back. An operation nobody times is an operation nobody keeps fast.
    let bytes_frame = byte_picture(HEIGHT, WIDTH)
    let regions = blobby(HEIGHT, WIDTH)
    run("canny", repeats, || cv.canny(bytes_frame, 60.0, 140.0))
    run("connected-components", repeats, || connected_components(regions)[1])
    run("distance-transform", repeats, || distance_transform(regions, cv.DIST_L2, cv.DIST_MASK_3))
    run("integral", repeats, || integral(grey))
    run_scalar("good-features", repeats, fn() -> float {
        return float(cv.good_features_to_track(grey, 200, 0.01, 6.0).len())
    })

    #: The operations a pipeline reaches for around the ones above, none of
    #: which was measured against anything. Every one of these was at or above
    #: the peer when it was first timed except the smoothing filter, which ran
    #: entirely on the host and took 3.7 seconds a frame.
    let turn = get_rotation_matrix_2d(Point2f(float(WIDTH) * 0.5, float(HEIGHT) * 0.5), 30.0, 1.0)
    var across: list[float] = []
    var down: list[float] = []
    for index in 0..HEIGHT * WIDTH {
        let y = index // WIDTH
        let x = index - y * WIDTH
        across.push(float((x + 7) % WIDTH))
        down.push(float((y + 5) % HEIGHT))
    }
    let map_x = Mat.from_list(HEIGHT, WIDTH, cv.CV_32FC1, across)
    let map_y = Mat.from_list(HEIGHT, WIDTH, cv.CV_32FC1, down)
    let cross = get_structuring_element(MORPH_RECT, Size(5, 5))
    run_scalar("threshold", repeats, fn() -> float {
        let (level, made) = threshold(grey, 0.5, 1.0, cv.THRESH_BINARY)
        made.release()
        return level
    })
    run(
        "adaptive-threshold",
        repeats,
        || adaptive_threshold(bytes_frame, 255.0, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2.0)
    )
    run("equalize-hist", repeats, || equalize_hist(bytes_frame))
    run("morph-open-5", repeats, || morphology_ex(grey, cv.MORPH_OPEN, cross))
    run("morph-gradient-5", repeats, || morphology_ex(grey, cv.MORPH_GRADIENT, cross))
    run("warp-affine", repeats, || warp_affine(grey, turn, Size(WIDTH, HEIGHT)))
    run("remap", repeats, || remap(grey, map_x, map_y))

    #: Three colour channels, because the peer's edge-preserving filter takes
    #: nothing else, and at a tenth of the repetitions because both sides take
    #: tens of milliseconds a frame.
    let photo = picture(3)
    run("edge-preserving", max(2, repeats // 10), || edge_preserving_filter(photo))
    run("detail-enhance", max(2, repeats // 10), || detail_enhance(photo))

    #: Contours and the shapes fitted to them. Every one of these was between
    #: twenty and forty thousand times slower than the peer when it was first
    #: timed, and none of it was measured by anything.
    let (shapes, _tree) = find_contours(bytes_frame, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
    let single = shapes[0]
    let canvas = Mat.zeros(HEIGHT, WIDTH, cv.CV_8UC3)
    run_scalar("find-contours", max(2, repeats // 10), fn() -> float {
        let (got, _h) = find_contours(bytes_frame, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
        return float(got.len())
    })
    run_scalar("find-contours-tree", max(2, repeats // 10), fn() -> float {
        let (got, _h) = find_contours(bytes_frame, RETR_TREE, CHAIN_APPROX_SIMPLE)
        return float(got.len())
    })
    run_scalar("convex-hull", repeats, fn() -> float {
        var total = 0
        for shape in shapes {
            total += convex_hull(shape).len()
        }
        return float(total)
    })
    run_scalar("min-area-rect", repeats, fn() -> float {
        var total = 0.0
        for shape in shapes {
            total += min_area_rect(shape).angle
        }
        return total
    })
    run_scalar("approx-poly-dp", repeats, fn() -> float {
        var total = 0
        for shape in shapes {
            total += approx_poly_dp(shape, arc_length(shape, true) * 0.02, true).len()
        }
        return float(total)
    })
    run_scalar("min-enclosing-circle", repeats, fn() -> float {
        let (_centre, radius) = min_enclosing_circle(single)
        return radius
    })
    run_scalar("point-polygon-test", repeats, fn() -> float {
        var total = 0.0
        for k in 0..1000 {
            total += point_polygon_test(single, Point2f(float(k % 640), float(k % 480)), true)
        }
        return total
    })
    #: Scalar rather than the usual `run`, because drawing is in place and
    #: hands back the picture it was given: the harness releases the previous
    #: result each iteration, and that would free the canvas out from under the
    #: next call.
    run_scalar("draw-contours", repeats, fn() -> float {
        let painted = draw_contours(canvas, shapes, -1, Scalar(255.0, 255.0, 255.0, 0.0), 2)
        return float(painted.rows)
    })

    #: Inpainting and optical flow at 640 by 480. The peer's inpainter is a
    #: sequential march inward and takes the best part of a second on a 1080p
    #: frame, which would make the suite about that one row.
    let small_rows = 480
    let small_cols = 640
    let damaged = byte_picture(small_rows, small_cols)
    let holes = blobby(small_rows, small_cols)
    run("inpaint", max(2, repeats // 10), || inpaint(damaged, holes, 3.0, cv.INPAINT_TELEA))

    let first = picture_at(small_rows, small_cols)
    let second = shifted_by(first, small_rows, small_cols, 2, 1)
    var tracked: list[Point2f] = []
    for index in 0..200 {
        tracked.push(Point2f(float(40 + index * 13 % 540), float(40 + index * 7 % 380)))
    }
    #: Each call is long enough on its own; a full thirty would make the
    #: suite mostly this one row.
    run_scalar("optical-flow", max(2, repeats // 10), fn() -> float {
        let (moved, _found, _error) = calc_optical_flow_pyr_lk(first, second, tracked)
        return moved[0].x
    })
    return 0
}
