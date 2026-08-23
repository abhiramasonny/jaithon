#: The same picture in, the same answer out, for everything iterative.
#:
#: An operation that spreads a value across an image, or that settles into a
#: fixed point, is written as a kernel every pixel runs at once -- and if it
#: reads and writes one buffer while doing so, what it lands on depends on the
#: order the threads happened to run in. That is invisible until somebody
#: compares two runs, and it stayed invisible here: Canny's hysteresis gave a
#: different set of edges every time and every test still passed, because no
#: test asked it twice.
#:
#: The bar is equality. These are not operations whose answer is allowed to
#: drift; each has one fixed point and running to it is the whole job.
import std.gpu as gpu
import std.math
from std.test import assert_eq, assert_true
from jaicv.core.mat import Mat
from jaicv.core.types import CV_8UC1, CV_32FC1, CV_32FC3, Point2f, Rect
from jaicv.core.cluster import kmeans, KMEANS_PP_CENTERS
from jaicv.imgproc.contours import find_contours, CHAIN_APPROX_SIMPLE, RETR_LIST
from jaicv.imgproc.edges import canny
from jaicv.imgproc.features import good_features_to_track, hough_lines
from jaicv.imgproc.grabcut import grab_cut, GC_INIT_WITH_RECT
from jaicv.imgproc.hist import equalize_hist
from jaicv.imgproc.misc import pyr_mean_shift_filtering
from jaicv.imgproc.segmentation import connected_components, distance_transform, watershed
from jaicv.imgproc.segmentation import DIST_L2, DIST_MASK_3
from jaicv.imgproc.threshold import threshold, THRESH_BINARY, THRESH_OTSU
from jaicv.photo.denoise import inpaint, INPAINT_TELEA
from jaicv.video.optflow import calc_optical_flow_farneback, calc_optical_flow_pyr_lk

let ROWS = 96
let COLS = 128
let ROUNDS = 3

fn have_gpu() -> bool { return gpu.is_available() }

fn digest(m: Mat) -> float {
    var total = 0.0
    for (index, value) in m.to_list().enumerate() { total += value * float(index % 29 + 1) }
    return total
}

#: Run it `ROUNDS` times and insist every answer is the first one.
fn repeats(name: str, work: fn() -> float) -> void {
    var first = 0.0
    for round in 0..ROUNDS {
        let value = work()
        if round == 0 {
            first = value
            continue
        }
        assert_eq(value, first)
    }
    #: An answer of zero every time would pass the comparison and mean the
    #: operation did nothing, so say so rather than letting it through.
    if first == 0.0 { throw AssertionError(f"{name} answered zero, so it settled nothing") }
}

fn noise() -> list[float] {
    var seed = 4242
    var out: list[float] = []
    for _index in 0..ROWS * COLS {
        seed = (seed * 1103515245 + 12345) % 2147483648
        out.push(float(seed % 256))
    }
    return out
}

fn test_the_iterative_operations_repeat() -> void {
    if not have_gpu() { return }
    let flat = noise()
    var wide: list[float] = []
    for value in flat {
        wide.push(value)
        wide.push(math.fmod(value * 3.0, 256.0))
        wide.push(math.fmod(value * 7.0, 256.0))
    }
    let grey = Mat.from_list(ROWS, COLS, CV_32FC1, flat)
    let colour = Mat.from_list(ROWS, COLS, CV_32FC3, wide)
    let raw = Mat.from_list(ROWS, COLS, CV_8UC1, flat)
    let edges = canny(raw, 60.0, 140.0)
    #: The flow pair has to actually move, or both answer zero everywhere and
    #: agreeing on that says nothing.
    var moved_frame: list[float] = []
    for index in 0..ROWS * COLS {
        let y = index // COLS
        let x = index - y * COLS
        let sx = max(0, x - 2)
        let sy = max(0, y - 1)
        moved_frame.push(flat[sy * COLS + sx])
    }
    let later = Mat.from_list(ROWS, COLS, CV_32FC1, moved_frame)

    repeats("canny", || digest(canny(raw, 60.0, 140.0)))
    repeats("connected-components", fn() -> float {
        let (count, labels) = connected_components(edges)
        return float(count) * 7.0 + digest(labels)
    })
    repeats("distance-transform", || digest(distance_transform(edges, DIST_L2, DIST_MASK_3)))
    repeats("inpaint", || digest(inpaint(grey, edges, 3.0, INPAINT_TELEA)))
    repeats("watershed", fn() -> float {
        var marks: list[float] = []
        for index in 0..ROWS * COLS {
            let y = index // COLS
            let x = index - y * COLS
            marks.push(if y < 4 and x < 4 {
                1.0
            } elif y > ROWS - 5 and x > COLS - 5 {
                2.0
            } else {
                0.0
            })
        }
        return digest(watershed(colour, Mat.from_list(ROWS, COLS, CV_32FC1, marks)))
    })
    repeats("good-features", fn() -> float {
        var total = 0.0
        for (at, point) in good_features_to_track(grey, 60, 0.01, 5.0).enumerate() {
            total += (point.x * 31.0 + point.y * 17.0) * float(at % 7 + 1)
        }
        return total
    })
    repeats("hough-lines", fn() -> float {
        var total = 0.0
        for (at, line) in hough_lines(edges, 1.0, math.PI / 180.0, 60).enumerate() {
            total += (line.rho * 3.0 + line.theta * 11.0) * float(at % 5 + 1)
        }
        return total
    })
    repeats("find-contours", fn() -> float {
        let (contours, _tree) = find_contours(edges, RETR_LIST, CHAIN_APPROX_SIMPLE)
        var total = 0.0
        for (at, part) in contours.enumerate() {
            total += float(part.len() * (at % 5 + 1))
        }
        return total
    })
    repeats("kmeans", fn() -> float {
        var samples: list[float] = []
        for index in 0..400 * 3 {
            samples.push(float(index * 37 % 251))
        }
        let data = Mat.from_list(400, 3, CV_32FC1, samples)
        let (compactness, labels, centres) = kmeans(data, 5, 3, 20, 1.0e-4, KMEANS_PP_CENTERS)
        return compactness + digest(labels) + digest(centres)
    })
    repeats("pyr-mean-shift", || digest(pyr_mean_shift_filtering(colour, 10.0, 20.0)))
    repeats("threshold-otsu", fn() -> float {
        let (level, mask) = threshold(raw, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU)
        return level + digest(mask)
    })
    repeats("equalize-hist", || digest(equalize_hist(raw)))
    repeats("grabcut", fn() -> float {
        let hint = Mat.zeros(ROWS, COLS, CV_8UC1)
        let back = Mat.zeros(1, 65, CV_32FC1)
        let front = Mat.zeros(1, 65, CV_32FC1)
        let (mask, _b, _f) = grab_cut(colour, hint, Rect(15, 15, 90, 70), back, front, 2, GC_INIT_WITH_RECT)
        return digest(mask)
    })
    repeats("optical-flow-lk", fn() -> float {
        var points: list[Point2f] = []
        for index in 0..16 {
            points.push(Point2f(float(30 + index * 7 % 90), float(25 + index * 5 % 70)))
        }
        let (moved, _found, _err) = calc_optical_flow_pyr_lk(grey, later, points)
        var total = 0.0
        for (at, point) in moved.enumerate() {
            total += (point.x * 13.0 + point.y * 29.0) * float(at % 3 + 1)
        }
        return total
    })
    repeats("farneback", || digest(calc_optical_flow_farneback(grey, later)))
}
