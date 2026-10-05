#: Every operation gives back the device memory it borrowed.
#:
#: Device memory is not garbage collected, so an intermediate an operation
#: forgets to release outlives the call for good. That is not only a leak: the
#: allocator comes under pressure and everything slows. good-features leaked
#: about 74 MB a call at 1080p and ran at half speed for it; canny, the
#: adaptive threshold, the compound morphology, the edge-preserving filter and
#: optical flow all leaked frames, and every filter leaked its taps.
#:
#: Each case calls an operation a few times, releasing what it returns, and
#: asserts the device holds exactly the buffers it held before. The first call
#: is outside the count, so anything an operation caches once is allowed.
import std.gpu as gpu
from std.test import assert_eq
import jaicv as cv
from jaicv import Mat, Size
from jaicv.core.arith import convert_scale_abs, magnitude, phase
from jaicv.core.stats import norm
from jaicv.core.types import Point2f
from jaicv.imgproc.geometric import get_rotation_matrix_2d, warp_affine
from jaicv.imgproc.hist import equalize_hist
from jaicv.imgproc.misc import phase_correlate
from jaicv.calib3d.calibrate import undistort
from jaicv.imgproc.morph import get_structuring_element, morphology_ex, MORPH_RECT
from jaicv.imgproc.threshold import adaptive_threshold
from jaicv.photo.denoise import detail_enhance, edge_preserving_filter
from jaicv.video.optflow import calc_optical_flow_farneback, calc_optical_flow_pyr_lk
from jaicv.calib3d.stereo import StereoBM, StereoSGBM
from jaicv.objdetect.cascade import CascadeClassifier
from jaicv.objdetect.qr import qr_encode, QRCodeDetector
from jaicv.features2d.detect import BRIEF, ORB, SimpleBlobDetector, SimpleBlobDetectorParams

let ROWS = 48
let COLS = 64

fn picture(kind: int) -> Mat {
    let channels = cv.CV_MAT_CN(kind)
    var values: list[float] = []
    for index in 0..ROWS * COLS * channels { values.push(float(index * 37 % 251)) }
    return Mat.from_list(ROWS, COLS, kind, values)
}

#: Run `work` once to settle any cache, then three more times, and report how
#: many device buffers those three left behind.
fn growth(work: fn() -> Mat) -> int {
    work().release()
    gpu.synchronize()
    let before = gpu.live_buffers()
    for _round in 0..3 { work().release() }
    gpu.synchronize()
    return gpu.live_buffers() - before
}

fn growth_of_points(work: fn() -> int) -> int {
    let _ = work()
    gpu.synchronize()
    let before = gpu.live_buffers()
    for _round in 0..3 { let _ = work() }
    gpu.synchronize()
    return gpu.live_buffers() - before
}

fn test_filters_release_their_taps() -> void {
    if not gpu.is_available() { return }
    let grey = picture(cv.CV_32FC1)
    assert_eq(growth(|| cv.gaussian_blur(grey, Size(5, 5), 0.0)), 0)
    assert_eq(growth(|| cv.gaussian_blur(grey, Size(7, 7), 1.5)), 0)
    assert_eq(growth(|| cv.blur(grey, Size(9, 9))), 0)
    assert_eq(growth(|| cv.sobel(grey, cv.CV_32F, 1, 0, ksize: 3)), 0)
    assert_eq(growth(|| cv.sobel(grey, cv.CV_32F, 0, 1, ksize: 5, scale: 0.5)), 0)
    assert_eq(growth(|| cv.laplacian(grey, cv.CV_32F)), 0)
    assert_eq(growth(|| cv.laplacian(grey, cv.CV_32F, ksize: 3)), 0)
    grey.release()
}

fn test_detectors_release_their_intermediates() -> void {
    if not gpu.is_available() { return }
    let grey = picture(cv.CV_32FC1)
    let bytes = picture(cv.CV_8UC1)
    assert_eq(growth(|| cv.canny(bytes, 60.0, 140.0)), 0)
    assert_eq(growth(|| cv.corner_harris(grey)), 0)
    assert_eq(growth(|| cv.corner_min_eigen_val(grey)), 0)
    assert_eq(growth_of_points(|| cv.good_features_to_track(grey, 20, 0.01, 3.0).len()), 0)
    assert_eq(
        growth(|| adaptive_threshold(bytes, 255.0, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2.0)),
        0
    )
    assert_eq(
        growth(
            || adaptive_threshold(bytes, 255.0, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 5, 2.0)
        ),
        0
    )
    assert_eq(growth(|| equalize_hist(bytes)), 0)
    grey.release()
    bytes.release()
}

fn test_compound_morphology_releases_its_intermediates() -> void {
    if not gpu.is_available() { return }
    let grey = picture(cv.CV_32FC1)
    let cross = get_structuring_element(MORPH_RECT, Size(5, 5))
    for op in [cv.MORPH_OPEN, cv.MORPH_CLOSE, cv.MORPH_GRADIENT, cv.MORPH_TOPHAT, cv.MORPH_BLACKHAT] {
        assert_eq(growth(|| morphology_ex(grey, op, cross)), 0)
    }
    #: Repeated passes, and the default element.
    assert_eq(growth(|| morphology_ex(grey, cv.MORPH_OPEN, cross, iterations: 3)), 0)
    assert_eq(growth(|| morphology_ex(grey, cv.MORPH_ERODE)), 0)
    grey.release()
}

fn test_photo_warp_and_flow_release_their_intermediates() -> void {
    if not gpu.is_available() { return }
    let colour = picture(cv.CV_8UC3)
    let grey = picture(cv.CV_32FC1)
    assert_eq(growth(|| edge_preserving_filter(colour)), 0)
    assert_eq(growth(|| detail_enhance(colour)), 0)
    let turn = get_rotation_matrix_2d(Point2f(32.0, 24.0), 30.0, 1.0)
    assert_eq(growth(|| warp_affine(grey, turn, Size(COLS, ROWS))), 0)
    let tracked = [Point2f(20.0, 20.0), Point2f(30.0, 25.0)]
    assert_eq(growth_of_points(fn() -> int {
        let (moved, _found, _error) = calc_optical_flow_pyr_lk(grey, grey, tracked)
        return moved.len()
    }), 0)
    colour.release()
    grey.release()
}

fn test_arithmetic_statistics_and_matching_release_their_intermediates() -> void {
    if not gpu.is_available() { return }
    let grey = picture(cv.CV_32FC1)
    let other = picture(cv.CV_32FC1)
    let colour = picture(cv.CV_8UC3)
    let bytes = picture(cv.CV_8UC1)
    assert_eq(growth(|| convert_scale_abs(grey, 2.0, 1.0)), 0)
    assert_eq(growth(|| magnitude(grey, other)), 0)
    assert_eq(growth(|| phase(grey, other)), 0)
    let templ = Mat.from_list(5, 5, cv.CV_32FC1, [float(index) for index in 0..25])
    assert_eq(growth(|| cv.match_template(grey, templ)), 0)
    assert_eq(growth_of_points(|| int(norm(grey, other: other))), 0)
    assert_eq(growth_of_points(|| int(norm(colour, norm_type: cv.NORM_INF))), 0)
    assert_eq(growth_of_points(|| cv.hough_circles(bytes, 1.0, 10.0, 100.0, 30.0, 5, 20).len()), 0)
    assert_eq(growth_of_points(|| int(phase_correlate(grey, other)[1] * 1000.0)), 0)
    let camera = Mat.from_list(3, 3, cv.CV_32FC1, [50.0, 0.0, 32.0, 0.0, 50.0, 24.0, 0.0, 0.0, 1.0])
    assert_eq(growth(|| undistort(bytes, camera, [-0.2, 0.05, 0.001, -0.001, 0.0])), 0)
    grey.release()
    other.release()
    colour.release()
    bytes.release()
    templ.release()
}

fn test_detectors_flow_and_stereo_release_their_intermediates() -> void {
    if not gpu.is_available() { return }
    let bytes = picture(cv.CV_8UC1)
    let other = picture(cv.CV_8UC1)
    let orb = ORB(40)
    assert_eq(growth_of_points(fn() -> int {
        let (found, described) = orb.detect_and_compute(bytes)
        described.release()
        return found.len()
    }), 0)
    let keys = orb.detect(bytes)
    let brief = BRIEF()
    assert_eq(growth(|| brief.compute(bytes, keys)[1]), 0)
    assert_eq(growth_of_points(|| SimpleBlobDetector(SimpleBlobDetectorParams()).detect(bytes).len()), 0)
    assert_eq(growth(|| calc_optical_flow_farneback(bytes, other)), 0)
    let bm = StereoBM(16, 9)
    assert_eq(growth(|| bm.compute(bytes, other)), 0)
    let sgbm = StereoSGBM()
    assert_eq(growth(|| sgbm.compute(bytes, other)), 0)
    let cascade = CascadeClassifier("packages/jaicv/tests/oracle/synthetic_cascade.xml")
    assert_eq(growth_of_points(|| cascade.detect_multi_scale(bytes, 1.1, 0).len()), 0)
    let symbol = qr_encode("jaicv", 0, 2, 4)
    let reader = QRCodeDetector()
    assert_eq(growth_of_points(|| reader.detect_and_decode(symbol)[0].len()), 0)
    bytes.release()
    other.release()
}

#: Releasing an intermediate must never reach the caller's own matrices.
fn test_inputs_survive_the_releases() -> void {
    if not gpu.is_available() { return }
    let grey = picture(cv.CV_32FC1)
    let expected = grey.to_list()
    let turn = get_rotation_matrix_2d(Point2f(32.0, 24.0), 30.0, 1.0)
    let cross = get_structuring_element(MORPH_RECT, Size(3, 3))
    for _round in 0..2 {
        cv.gaussian_blur(grey, Size(3, 3), 0.0).release()
        morphology_ex(grey, cv.MORPH_GRADIENT, cross).release()
        warp_affine(grey, turn, Size(COLS, ROWS), flags: cv.WARP_INVERSE_MAP).release()
        let (_moved, _found, _error) = calc_optical_flow_pyr_lk(grey, grey, [Point2f(20.0, 20.0)])
    }
    gpu.synchronize()
    assert_eq(grey.to_list(), expected)
    assert_eq(cross.to_list().len(), 9)
    assert_eq(turn.to_list().len(), 6)
}
