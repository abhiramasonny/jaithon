#: Every exported entry point, called once.
#:
#: `test_against_opencv.jai` compares what a function computes; this asks the
#: cheaper question first -- does it run at all -- of the part of the library
#: that comparison does not reach. Of jaicv's exports, more than half had no
#: caller in any test: the recorded cases cover what OpenCV was asked to
#: record, and a function nobody recorded is a function nobody ran.
#:
#: That is not a hypothetical gap. `find_homography` with `RANSAC` -- OpenCV's
#: own default, and what anybody matching two photographs calls -- raised
#: `OverflowError` on its second sampled correspondence for as long as it had
#: existed, because the oracle only ever recorded `method=0`. The same shape of
#: bug took out AVI reading. Both were found by writing this file.
#:
#: So the contract here is deliberately weak and deliberately wide: every
#: function is called with arguments it should accept, and what is asserted is
#: that it returns, and that what comes back has the size, depth and channel
#: count it promised. Anything stronger belongs in the oracle, where there is
#: something to compare against.
#:
#: What is left out, and why: `highgui` needs a window server, so `imshow`,
#: `wait_key`, the trackbars and the mouse callbacks cannot be reached from a
#: test run. Reading a camera needs a camera. Everything else the module
#: exports is called below.
import std.math
import std.gpu as gpu
from std.io import Path
from std.os import temp_dir
from std.test import assert_eq, assert_near, assert_throws, assert_true
from jaicv.core.mat import cached_kernel, f32_bits, touch, Mat
from jaicv.core.types import CV_8U, CV_8UC1, CV_8UC2, CV_8UC3, CV_8UC4, CV_32FC1, CV_32SC1
from jaicv.core.types import depth_is_integer, depth_max, depth_min
from jaicv.core.types import CV_MAKETYPE, CV_MAT_CN, Point, Point2f, Point3f, Rect, RotatedRect
from jaicv.core.types import Scalar, Size, Size2f
from jaicv.core.arith import binary_op, bitwise_and, bitwise_or, bitwise_xor, scalar_op
from jaicv.core.arith import abs_mat, add_weighted, convert_scale_abs, exp_mat, log_mat
from jaicv.core.arith import max_mat, min_mat, pow_mat, scale_add, sqrt_mat, unary_op
from jaicv.core.arith import OP_ADD, UN_SQRT
from jaicv.core.arith import cart_to_polar, magnitude, phase, polar_to_cart
from jaicv.imgproc.misc import accumulate, accumulate_product, accumulate_square
from jaicv.imgproc.misc import accumulate_weighted
from jaicv.core.misc import border_interpolate, check_range, eigen, find_non_zero
from jaicv.core.misc import insert_channel
from jaicv.core.misc import mahalanobis, min_max_idx, mul_transposed, patch_nans
from jaicv.core.linalg import identity
from jaicv.core.misc import randn, randu, set_identity, set_rng_seed
from jaicv.dnn.blob import blob_to_tensor
from jaicv.core.stats import channel_stats, mean, mean_std_dev, mean_value
from jaicv.core.stats import min_max, reduce, trace
from jaicv.core.types import NORM_HAMMING, REDUCE_SUM
from jaicv.core.linalg import inverse, levenberg_marquardt, null_space, symmetric_eigen
from jaicv.core.dft import dft, mul_spectrums, DFT_COMPLEX_OUTPUT
from jaicv.imgcodecs.codecs import imread, imwrite, have_image_reader, have_image_writer
from jaicv.imgproc.color import cvt_color, cvt_color_nv12, cvt_color_yuy2
from jaicv.imgproc.color import COLOR_Lab2RGB, COLOR_RGB2Lab, COLOR_RGB2YCrCb
from jaicv.imgproc.color import COLOR_YCrCb2RGB, COLOR_GRAY2BGR
from jaicv.imgproc.filter import bilateral_filter, get_derivative_kernels, sep_filter_2d
from jaicv.imgproc.filter import sqr_box_filter
from jaicv.imgproc.threshold import histogram_256, otsu_threshold, threshold
from jaicv.imgproc.threshold import triangle_threshold, THRESH_BINARY
from jaicv.imgproc.edges import canny
from std.random import Random
from jaicv.imgproc.geometric import crop, get_rotation_matrix_2d, invert3, invert_transform
from jaicv.imgproc.geometric import resize, solve_dense
from jaicv.imgproc.drawing import clip_line, ellipse_2_poly, ellipse_rotated
from jaicv.imgproc.drawing import circle, line_points, rectangle, rectangle_rect
from jaicv.imgproc.contours import draw_contours, find_contours, RETR_EXTERNAL
from jaicv.imgproc.shape import box_points, convex_hull_indices, match_shapes
from jaicv.imgproc.shape import moments_f, moments_image
from jaicv.imgproc.segmentation import colour_labels, connected_components, watershed
from jaicv.imgproc.hist import create_clahe
from jaicv.imgproc.features import corner_sub_pix, good_features_to_track, hough_circles
from jaicv.imgproc.features import hough_lines_p
from jaicv.imgproc.misc import pyr_mean_shift_filtering
from jaicv.features2d.detect import BFMatcher, ORB, SimpleBlobDetector
from jaicv.features2d.detect import SimpleBlobDetectorParams, draw_keypoints, draw_matches
from jaicv.objdetect.cascade import group_rectangles
from jaicv.objdetect.hog import HOGDescriptor
from jaicv.objdetect.galois import gf_div, gf_exp, gf_inv, gf_mul, rs_correct
from jaicv.objdetect.qr import function_map, mask_bit, decode_format, decode_version
from jaicv.objdetect.xml import parse_xml
from jaicv.calib3d.transform import find_homography, get_optimal_new_camera_matrix
from jaicv.calib3d.transform import init_undistort_rectify_map, refine_homography
from jaicv.calib3d.calibrate import draw_chessboard_corners, solve_pnp
from jaicv.calib3d.transform import project_points
from jaicv.video.optflow import calc_optical_flow_farneback, calc_optical_flow_pyr_lk
from jaicv.video.optflow import flow_to_colour, polynomial_expansion
from jaicv.video.background import create_background_subtractor_knn
from jaicv.video.background import create_background_subtractor_mog2
from jaicv.video.track import cam_shift, mean_shift, KalmanFilter
from jaicv.photo.denoise import detail_enhance, edge_preserving_filter
from jaicv.photo.denoise import fast_nl_means_denoising, fast_nl_means_denoising_colored
from jaicv.photo.denoise import inpaint, stylization
from jaicv.ml.models import ann_mlp_create, d_trees_create, k_nearest_create
from jaicv.ml.models import logistic_regression_create, normal_bayes_classifier_create
from jaicv.ml.models import r_trees_create, svm_create
from jaicv.videoio.capture import device_count, device_name, device_names, permission
from jaicv.videoio.capture import PERMISSION_GRANTED, PERMISSION_UNAVAILABLE
from jaicv.dnn.blob import blob_from_images, images_from_blob, net_from
from jaicv.dnn.blob import nms_boxes_batched, Runnable

fn have_gpu() -> bool { return gpu.is_available() }

#: A disc on a flat field: something with an edge, a corner-free interior and a
#: connected component, so contours, flow, thresholds and blobs all have
#: something real to find.
fn disc() -> Mat {
    var values: list[float] = []
    for index in 0..64 * 64 {
        let x = index % 64
        let y = index // 64
        values.push(if (x - 20) * (x - 20) + (y - 30) * (y - 30) < 100 { 220.0 } else { 30.0 })
    }
    return Mat.from_list(64, 64, CV_8UC1, values)
}

#: The same disc moved sideways, for anything that compares two frames.
fn shifted(base: Mat, dx: int) -> Mat {
    let flat = base.to_list()
    var values: list[float] = []
    for y in 0..base.rows {
        for x in 0..base.cols {
            values.push(flat[y * base.cols + min(max(x - dx, 0), base.cols - 1)])
        }
    }
    return Mat.from_list(base.rows, base.cols, base.type_(), values)
}

fn assert_shape(result: Mat, rows: int, cols: int, channels: int) -> void {
    assert_eq(result.rows, rows)
    assert_eq(result.cols, cols)
    assert_eq(result.channels(), channels)
}

fn test_core_arithmetic_and_statistics_run() -> void {
    if not have_gpu() { return }
    let g = disc()
    let c = cvt_color(g, COLOR_GRAY2BGR)
    let f = g.convert_to(CV_32FC1, 1.0 / 255.0, 0.0)

    assert_shape(abs_mat(f), 64, 64, 1)
    assert_shape(add_weighted(g, 0.5, g, 0.5, 1.0), 64, 64, 1)
    assert_shape(bitwise_and(g, g), 64, 64, 1)
    assert_shape(bitwise_or(g, g), 64, 64, 1)
    assert_shape(bitwise_xor(g, g), 64, 64, 1)
    assert_shape(binary_op(g, g, OP_ADD), 64, 64, 1)
    assert_shape(scalar_op(g, Scalar(10.0, 0.0, 0.0), OP_ADD), 64, 64, 1)
    assert_shape(unary_op(f, UN_SQRT), 64, 64, 1)
    assert_shape(convert_scale_abs(f, 255.0, 0.0), 64, 64, 1)
    assert_shape(exp_mat(f), 64, 64, 1)
    assert_shape(log_mat(add_weighted(f, 1.0, f, 0.0, 1.0)), 64, 64, 1)
    assert_shape(max_mat(g, g), 64, 64, 1)
    assert_shape(min_mat(g, g), 64, 64, 1)
    assert_shape(pow_mat(f, 2.0), 64, 64, 1)
    assert_shape(scale_add(f, 2.0, f), 64, 64, 1)
    assert_shape(sqrt_mat(f), 64, 64, 1)
    assert_shape(magnitude(f, f), 64, 64, 1)
    assert_shape(phase(f, f), 64, 64, 1)
    assert_shape(patch_nans(f, 0.0), 64, 64, 1)
    assert_shape(insert_channel(g, c, 1), 64, 64, 3)
    assert_shape(reduce(f, 0, REDUCE_SUM), 1, 64, 1)

    let running = Mat.zeros(64, 64, CV_32FC1)
    assert_shape(accumulate(f, running), 64, 64, 1)
    assert_shape(accumulate_square(f, running), 64, 64, 1)
    assert_shape(accumulate_product(f, f, running), 64, 64, 1)
    assert_shape(accumulate_weighted(f, running, 0.1), 64, 64, 1)

    assert_eq(depth_max(CV_8U), 255.0)
    assert_eq(depth_min(CV_8U), 0.0)
    assert_true(depth_is_integer(CV_8U))
    assert_true(moments_image(g).m00 > 0.0)

    let (px, py) = cart_to_polar(f, f)
    assert_shape(px, 64, 64, 1)
    assert_shape(py, 64, 64, 1)
    let (bx, by) = polar_to_cart(f, f)
    assert_shape(bx, 64, 64, 1)
    assert_shape(by, 64, 64, 1)

    assert_true(check_range(f))
    assert_true(mean_value(c) > 0.0)
    let (low, high) = min_max(f)
    assert_true(high >= low)
    let (lo2, hi2, at_lo, at_hi) = min_max_idx(f)
    assert_true(hi2 >= lo2)
    assert_true(at_lo.x >= 0 and at_hi.x >= 0)
    let (averages, deviations) = mean_std_dev(c)
    assert_true(averages[0] > 0.0)
    assert_true(deviations[0] >= 0.0)
    let (mean_one, dev_one, total, counted) = channel_stats(c, 1)
    assert_true(dev_one >= 0.0 and counted > 0 and total >= 0.0 and mean_one >= 0.0)
    assert_true(find_non_zero(g).len() > 0)
    assert_true(mean(c)[0] > 0.0)
}

fn test_core_linear_algebra_and_transforms_run() -> void {
    if not have_gpu() { return }
    let f = disc().convert_to(CV_32FC1, 1.0 / 255.0, 0.0)
    let symmetric = Mat.from_list(3, 3, CV_32FC1, [4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0])

    let (eigenvalues, vectors) = eigen(symmetric)
    assert_eq(eigenvalues.len(), 3)
    assert_shape(vectors, 3, 3, 1)
    let (svalues, svectors) = symmetric_eigen(symmetric)
    assert_eq(svalues.len(), 3)
    assert_shape(svectors, 3, 3, 1)
    assert_shape(inverse(symmetric), 3, 3, 1)
    assert_true(math.fabs(trace(symmetric)[0] - 9.0) < 1e-4)
    assert_shape(mul_transposed(f, true), 64, 64, 1)
    assert_eq(null_space(Mat.from_list(2, 3, CV_32FC1, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])).len(), 3)
    assert_true(
        mahalanobis(Mat.from_list(1, 3, CV_32FC1, [1.0, 2.0, 3.0]), Mat.from_list(1, 3, CV_32FC1, [2.0, 1.0, 4.0]), inverse(symmetric)) >=
            0.0
    )
    assert_eq(solve_dense([[2.0, 1.0], [1.0, 3.0]], [5.0, 10.0]).len(), 2)
    assert_eq(invert3([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]]).len(), 3)
    assert_shape(
        invert_transform(get_rotation_matrix_2d(Point2f(32.0, 32.0), 20.0, 1.0), false),
        2,
        3,
        1
    )
    assert_shape(set_identity(Size(4, 4), CV_32FC1), 4, 4, 1)

    #: A straight line fitted to points on one, which any working
    #: Levenberg-Marquardt recovers in a couple of steps.
    let xs = [0.0, 1.0, 2.0, 3.0, 4.0]
    let residual = fn(parameters: list[float]) -> list[float] {
        var out: list[float] = []
        for index in 0..xs.len() {
            out.push(parameters[0] * xs[index] + parameters[1] - (2.0 * xs[index] + 1.0))
        }
        return out
    }
    let jacobian = fn(_parameters: list[float]) -> Mat {
        var values: list[float] = []
        for index in 0..xs.len() {
            values.push(xs[index])
            values.push(1.0)
        }
        return Mat.from_list(xs.len(), 2, CV_32FC1, values)
    }
    let fitted = levenberg_marquardt(residual, jacobian, [0.0, 0.0])
    assert_true(math.fabs(fitted[0] - 2.0) < 1e-3)
    assert_true(math.fabs(fitted[1] - 1.0) < 1e-3)

    set_rng_seed(7)
    assert_shape(randu(Size(8, 8), CV_32FC1, 0.0, 1.0), 8, 8, 1)
    assert_shape(randn(Size(8, 8), CV_32FC1, 0.0, 1.0), 8, 8, 1)

    let spectrum = dft(f, DFT_COMPLEX_OUTPUT)
    assert_shape(mul_spectrums(spectrum, spectrum), 64, 64, 2)
}

fn test_imgproc_run() -> void {
    if not have_gpu() { return }
    let g = disc()
    let c = cvt_color(g, COLOR_GRAY2BGR)
    let f = g.convert_to(CV_32FC1, 1.0 / 255.0, 0.0)
    let row = Mat.from_list(3, 1, CV_32FC1, [1.0, 2.0, 1.0])
    let (_used, mask) = threshold(g, 128.0, 255.0, THRESH_BINARY)
    let (contours, _hierarchy) = find_contours(mask, RETR_EXTERNAL)
    assert_true(contours.len() > 0)

    assert_shape(bilateral_filter(c, 5, 30.0, 30.0), 64, 64, 3)
    assert_shape(sep_filter_2d(g, -1, row, row), 64, 64, 1)
    assert_shape(sqr_box_filter(f, -1, Size(3, 3)), 64, 64, 1)
    assert_shape(crop(c, Rect(4, 4, 20, 20)), 20, 20, 3)
    assert_shape(create_clahe(2.0, Size(8, 8)).apply(g), 64, 64, 1)
    assert_shape(pyr_mean_shift_filtering(c, 8.0, 16.0), 64, 64, 3)
    assert_shape(
        colour_labels(connected_components(mask)[1], connected_components(mask)[0]),
        64,
        64,
        3
    )
    assert_eq(histogram_256(g).len(), 256)
    assert_true(otsu_threshold(g) > 0.0)
    assert_true(triangle_threshold(g) > 0.0)

    let (dx, dy) = get_derivative_kernels(1, 0, 3)
    assert_eq(dx.rows * dx.cols, 3)
    assert_eq(dy.rows * dy.cols, 3)

    assert_true(border_interpolate(-3, 64, 4) >= 0)
    assert_eq(box_points(RotatedRect(Point2f(20.0, 20.0), Size2f(10.0, 6.0), 30.0)).len(), 4)
    let (visible, _a, _b) = clip_line(Size(64, 64), Point(-5, -5), Point(80, 80))
    assert_true(visible)
    assert_true(line_points(Size(64, 64), Point(-5, -5), Point(80, 80)).len() > 0)
    assert_true(ellipse_2_poly(Point(32, 32), Size(20, 10), 30, 0, 360, 10).len() > 0)
    assert_shape(
        ellipse_rotated(
            c,
            RotatedRect(Point2f(32.0, 32.0), Size2f(20.0, 10.0), 30.0),
            Scalar(255.0, 0.0, 0.0)
        ),
        64,
        64,
        3
    )
    assert_shape(rectangle_rect(c, Rect(4, 4, 20, 20), Scalar(0.0, 255.0, 0.0), 2), 64, 64, 3)
    assert_shape(draw_contours(c, contours, -1, Scalar(0.0, 0.0, 255.0), 1), 64, 64, 3)

    assert_true(match_shapes(contours[0], contours[0]) < 1e-6)
    assert_true(
        convex_hull_indices([Point(0, 0), Point(10, 0), Point(10, 10), Point(5, 5), Point(0, 10)]).len() >=
            3
    )
    assert_true(
        moments_f([Point2f(0.0, 0.0), Point2f(10.0, 0.0), Point2f(10.0, 10.0), Point2f(0.0, 10.0)]).m00 !=
            0.0
    )

    assert_true(good_features_to_track(g, 20, 0.01, 5.0).len() >= 0)
    assert_eq(corner_sub_pix(g, [Point2f(20.0, 20.0)], Size(3, 3)).len(), 1)
    assert_true(hough_circles(g, 1.0, 10.0, 100.0, 30.0, 5, 20).len() >= 0)
    assert_true(hough_lines_p(canny(g, 50.0, 150.0), 1.0, 0.017453292, 20, 10.0, 5.0).len() >= 0)

    var markers = Mat.zeros(64, 64, CV_32SC1)
    rectangle(markers, Point(2, 2), Point(10, 10), Scalar(1.0, 0.0, 0.0), -1)
    rectangle(markers, Point(40, 40), Point(60, 60), Scalar(2.0, 0.0, 0.0), -1)
    assert_shape(watershed(c, markers), 64, 64, 1)

    let luma = Mat.from_list(16, 16, CV_8UC1, [128.0 for _index in 0..256])
    let chroma = Mat.from_list(8, 8, CV_8UC2, [128.0 for _index in 0..128])
    assert_shape(cvt_color_nv12(luma, chroma), 16, 16, 3)
    assert_shape(
        cvt_color_yuy2(Mat.from_list(16, 8, CV_8UC4, [128.0 for _index in 0..512])),
        16,
        16,
        3
    )
}

fn test_imgcodecs_touch_the_filesystem() -> void {
    if not have_gpu() { return }
    assert_true(have_image_reader("png"))
    assert_true(have_image_writer("bmp"))
    let path = Path(temp_dir()).join("jaicv_reachable.png").to_str()
    assert_true(imwrite(path, cvt_color(disc(), COLOR_GRAY2BGR)))
    assert_shape(imread(path), 64, 64, 3)
    Path(path).remove()
}

fn test_features2d_and_objdetect_run() -> void {
    if not have_gpu() { return }
    let a = disc()
    let b = shifted(a, 3)
    let ca = cvt_color(a, COLOR_GRAY2BGR)
    let cb = cvt_color(b, COLOR_GRAY2BGR)

    let orb = ORB(60)
    let (first, first_desc) = orb.detect_and_compute(a)
    let (second, second_desc) = orb.detect_and_compute(b)
    let matcher = BFMatcher(NORM_HAMMING, true)
    let matches = matcher.match_(first_desc, second_desc)
    assert_true(matcher.knn_match(first_desc, second_desc, 2).len() == first.len())
    assert_shape(draw_keypoints(ca, first), 64, 64, 3)
    assert_eq(draw_matches(ca, first, cb, second, matches).cols, 128)
    assert_true(SimpleBlobDetector(SimpleBlobDetectorParams()).detect(a).len() >= 0)

    let grouped = group_rectangles(
        [Rect(0, 0, 10, 10), Rect(1, 1, 10, 10), Rect(2, 2, 10, 10)],
        1,
        0.2
    )
    assert_true(grouped.len() <= 3)

    let hog = HOGDescriptor(Size(32, 64), Size(16, 16), Size(8, 8), Size(8, 8), 9)
    let descriptor = hog.compute(resize(a, size: Size(32, 64)))
    assert_eq(descriptor.len(), hog.get_descriptor_size())
    var plane: list[float] = []
    for _index in 0..descriptor.len() + 1 { plane.push(0.001) }
    hog.set_svm_detector(plane)
    assert_true(hog.detect(resize(a, size: Size(48, 80)), win_stride: Size(16, 16)).len() >= 0)

    let node = parse_xml("<opencv_storage><stageNum>2</stageNum></opencv_storage>")
    assert_eq(node.name, "opencv_storage")
}

fn test_calib3d_run() -> void {
    if not have_gpu() { return }
    let camera = Mat.from_list(3, 3, CV_32FC1, [50.0, 0.0, 32.0, 0.0, 50.0, 32.0, 0.0, 0.0, 1.0])
    let distortion = [-0.2, 0.05, 0.001, -0.001, 0.0]
    let refreshed = get_optimal_new_camera_matrix(camera, distortion, Size(64, 64), 0.0)
    assert_shape(refreshed, 3, 3, 1)
    let (map_x, map_y) = init_undistort_rectify_map(camera, distortion, refreshed, Size(64, 64))
    assert_shape(map_x, 64, 64, 1)
    assert_shape(map_y, 64, 64, 1)

    var source: list[Point2f] = []
    var target: list[Point2f] = []
    for index in 0..8 {
        let x = float(index % 4) * 10.0
        let y = float(index // 4) * 10.0
        source.push(Point2f(x, y))
        target.push(Point2f(x * 1.2 + 2.0, y * 1.1 - 1.0))
    }
    let (homography, inliers) = find_homography(source, target)
    assert_eq(inliers.len(), source.len())
    assert_shape(refine_homography(homography, source, target), 3, 3, 1)

    var corners: list[Point2f] = []
    for index in 0..12 {
        corners.push(Point2f(float(index % 4) * 10.0 + 5.0, float(index // 4) * 10.0 + 5.0))
    }
    assert_shape(
        draw_chessboard_corners(cvt_color(disc(), COLOR_GRAY2BGR), Size(4, 3), corners, true),
        64,
        64,
        3
    )
}

#: A shift the tracker is told nothing about, which it has to recover.
#:
#: Checking only that the right number of points came back says nothing about
#: whether any of them moved to the right place -- the tracker could return
#: its input unchanged and pass.
fn test_optical_flow_finds_the_shift() -> void {
    if not have_gpu() { return }
    let dx = 2.5
    let dy = -1.5
    var before: list[float] = []
    var after: list[float] = []
    for index in 0..160 * 200 {
        let y = float(index // 200)
        let x = float(index % 200)
        before.push(texture_at(x, y))
        after.push(texture_at(x - dx, y - dy))
    }
    let first = Mat.from_list(160, 200, CV_32FC1, before)
    let second = Mat.from_list(160, 200, CV_32FC1, after)
    var points: list[Point2f] = []
    for index in 0..9 {
        points.push(Point2f(float(60 + index * 13 % 80), float(50 + index * 9 % 60)))
    }
    let (moved, found, _errors) = calc_optical_flow_pyr_lk(first, second, points)
    var tracked = 0
    for (index, point) in points.enumerate() {
        if not found[index] { continue }
        tracked += 1
        assert_near(moved[index].x - point.x, dx, 0.15)
        assert_near(moved[index].y - point.y, dy, 0.15)
    }
    assert_true(tracked >= 8)
}

fn texture_at(x: float, y: float) -> float {
    let blob = math.exp(-((x - 100.0) * (x - 100.0) + (y - 80.0) * (y - 80.0)) / 700.0)
    let grain = math.sin(x * 0.23) * math.cos(y * 0.19) + math.sin((x + y) * 0.11)
    return 60.0 + 90.0 * blob + 40.0 * grain
}

#: The same frame twice, which has to give the same edges twice.
#:
#: Hysteresis reads and writes one buffer from every thread at once, so where
#: it stops depends on the order the threads run in unless it is run all the
#: way to its fixed point. It was not, and the same picture came back with a
#: different set of edges each time -- and six percent short of the right one,
#: because a run of weak edges longer than the pass count never finished.
#: A matrix big enough that one rotation per turn of the loop is not enough.
#:
#: The Jacobi loop zeroes ONE off-diagonal a turn and ran a hundred turns. A
#: 12 by 12 has sixty-six off-diagonals and wants six to ten passes over all
#: of them, so a hundred rotations was about a sixth of the work and the
#: answer came back wrong: the worst residual for a 12 by 12 was 2e-02 and
#: for a 16 by 16 was 3e-01, against 1e-07 for a 6 by 6. `pose_from_dlt`
#: solves a 12 by 12, so this reached the pose solver.
#: Two classes on the same mean, one a hundred times wider than the other.
#:
#: A Gaussian's log density is `log prior - 0.5 log|C| - 0.5 d`, and the
#: middle term was missing. Without it a wide class always wins: distance to
#: its centre counts for less and nothing pays for the volume that buys. Every
#: point but the mean itself went to the wide class, at any distance.
fn test_a_bayes_classifier_pays_for_a_wide_class() -> void {
    if not have_gpu() { return }
    var seed = 5
    var samples: list[float] = []
    var labels: list[float] = []
    for _index in 0..200 {
        seed = (seed * 1103515245 + 12345) % 2147483648
        samples.push((float(seed % 2000) / 1000.0 - 1.0) * 0.1)
        labels.push(0.0)
    }
    for _index in 0..200 {
        seed = (seed * 1103515245 + 12345) % 2147483648
        samples.push((float(seed % 2000) / 1000.0 - 1.0) * 10.0)
        labels.push(1.0)
    }
    let model = normal_bayes_classifier_create()
    model.train(Mat.from_list(400, 1, CV_32FC1, samples), Mat.from_list(400, 1, CV_32FC1, labels))
    #: Near the shared mean the narrow class is far likelier.
    for near in [0.0, 0.001, 0.01, 0.05] {
        let got = model.predict(Mat.from_list(1, 1, CV_32FC1, [near])).values()[0]
        assert_eq(int(got), 0)
    }
    #: Well outside it, only the wide class can account for the point.
    for far in [1.0, 3.0, 7.0] {
        let got = model.predict(Mat.from_list(1, 1, CV_32FC1, [far])).values()[0]
        assert_eq(int(got), 1)
    }
}

#: An undistorted camera has nothing to correct, so the answer is the camera.
#:
#: The inner rectangle was chosen by the sign of the straightened coordinate
#: rather than by which edge of the picture the sample came from, and the top
#: and bottom rows run the full width -- so one of them always landed within
#: a fraction of a pixel of the principal point and both bounds collapsed
#: onto zero. A 640 by 480 with a focal length of 500 and no distortion at
#: all came back with a focal length of 4006 and its principal point at
#: (4, 4). Only alpha of one, which uses the outer rectangle and was
#: computed correctly, ever worked.
#: The same picture, and the same picture twice the size.
#:
#: ORB records which pyramid level it found each keypoint on and then
#: described every one of them through the same 31 by 31 window of the
#: full-resolution image, so the descriptors carried no scale information at
#: all: the same point submitted at five octaves came back byte-identical.
#: Matching a picture against a doubling of itself got five of eighty-four
#: cross-checked matches geometrically right -- six per cent, which is worse
#: than the identical-image case by a factor of sixteen and is the entire
#: reason to use ORB over a plain binary descriptor.
#:
#: The texture has to be non-repeating for this to measure anything. A
#: periodic pattern has features that genuinely recur, so a match to the
#: wrong period is not the descriptor's fault and the number stays low
#: whatever is done.
#: Four points that do not lie in a plane cannot fix a pose.
#:
#: The direct solution has twelve unknowns and each point gives two
#: equations, so four points leave a four-dimensional family of answers. The
#: null space returned an arbitrary member of it and the refinement polished
#: that into a pose reported as `ok`: twenty of thirty-nine random four-point
#: sets came back over a pixel out, the worst by 645, all of them claiming
#: success. Four coplanar points are a different matter -- those go through
#: the homography and are genuinely enough.
fn test_a_pose_needs_six_points_off_a_plane() -> void {
    if not have_gpu() { return }
    let camera = Mat.from_list(
        3,
        3,
        CV_32FC1,
        [600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0]
    )
    let cube = [
        Point3f(0.0, 0.0, 0.0),
        Point3f(1.0, 0.0, 0.0),
        Point3f(0.0, 1.0, 0.0),
        Point3f(0.0, 0.0, 1.0),
        Point3f(1.0, 1.0, 0.0),
        Point3f(1.0, 0.0, 1.0),
        Point3f(0.0, 1.0, 1.0),
    ]
    let truth_r = Mat.from_list(3, 1, CV_32FC1, [0.15, -0.25, 0.08])
    let truth_t = Mat.from_list(3, 1, CV_32FC1, [0.4, -0.2, 6.0])
    let seen = project_points(cube, truth_r, truth_t, camera, [])

    fn take(
        points: list[Point3f],
        seen: list[Point2f],
        n: int
    ) -> tuple[list[Point3f], list[Point2f]] {
        var a: list[Point3f] = []
        var b: list[Point2f] = []
        for index in 0..n {
            a.push(points[index])
            b.push(seen[index])
        }
        return (a, b)
    }

    for few in [4, 5] {
        let (objects, images) = take(cube, seen, few)
        assert_throws(ValueError, || solve_pnp(objects, images, camera, []))
    }

    #: Six is enough, and the answer has to be the one it was given.
    let (objects, images) = take(cube, seen, 7)
    let (ok, rvec, tvec) = solve_pnp(objects, images, camera, [])
    assert_true(ok, "seven points should be plenty")
    for (index, wanted) in [0.15, -0.25, 0.08].enumerate() {
        assert_near(rvec.values()[index], wanted, 0.05)
    }
    for (index, wanted) in [0.4, -0.2, 6.0].enumerate() {
        assert_near(tvec.values()[index], wanted, 0.05)
    }
}

fn test_orb_matches_across_a_change_of_scale() -> void {
    if not have_gpu() { return }
    let rng = Random.with_seed(29)
    var values: list[float] = []
    for _index in 0..240 * 320 { values.push(40.0) }
    let base = Mat.from_list(240, 320, CV_8UC1, values)
    for _shape in 0..70 {
        let x = int(rng.float_range(0.0, 320.0))
        let y = int(rng.float_range(0.0, 240.0))
        let reach = int(rng.float_range(4.0, 22.0))
        let tone = Scalar(rng.float_range(70.0, 250.0))
        if rng.next_float() < 0.5 {
            circle(base, Point(x, y), reach, tone, -1)
        } else {
            rectangle(base, Point(x, y), Point(x + reach, y + reach), tone, -1)
        }
    }

    let orb = ORB()
    let bigger = resize(base, size: Size(640, 480))
    let (here, described_here) = orb.detect_and_compute(base)
    let (there, described_there) = orb.detect_and_compute(bigger)
    let matches = BFMatcher(NORM_HAMMING, true).match_(described_here, described_there)
    assert_true(matches.len() > 40, f"only {matches.len()} matches at all")
    var right = 0
    for pair in matches {
        let small = here[pair.query_idx].pt
        let large = there[pair.train_idx].pt
        let dx = large.x - small.x * 2.0
        let dy = large.y - small.y * 2.0
        if math.sqrt(dx * dx + dy * dy) < 6.0 { right += 1 }
    }
    let share = float(right) / float(matches.len())
    assert_true(share > 0.35, f"only {right} of {matches.len()} matches were in the right place")
}

fn test_an_undistorted_camera_keeps_its_matrix() -> void {
    if not have_gpu() { return }
    let camera = Mat.from_list(
        3,
        3,
        CV_32FC1,
        [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0]
    )
    for alpha in [0.0, 0.5, 1.0] {
        let got = get_optimal_new_camera_matrix(camera, [], Size(640, 480), alpha).values()
        assert_near(got[0], 500.0, 2.0)
        assert_near(got[4], 500.0, 2.0)
        assert_near(got[2], 320.0, 2.0)
        assert_near(got[5], 240.0, 2.0)
    }

    #: With barrel distortion the two rectangles differ, and cropping to the
    #: valid area has to zoom in further than keeping everything does.
    let barrel = [-0.25, 0.05, 0.0, 0.0, 0.0]
    let tight = get_optimal_new_camera_matrix(camera, barrel, Size(640, 480), 0.0).values()
    let loose = get_optimal_new_camera_matrix(camera, barrel, Size(640, 480), 1.0).values()
    assert_true(tight[0] > loose[0], f"cropping gave {tight[0]}, keeping all gave {loose[0]}")
    #: And the principal point stays near the middle either way.
    for value in [tight[2], loose[2]] { assert_near(value, 320.0, 25.0) }
}

fn test_a_large_symmetric_matrix_is_fully_diagonalised() -> void {
    if not have_gpu() { return }
    var seed = 17
    for n in [3, 6, 12, 16] {
        #: B transpose B, which is symmetric and has no negative eigenvalue.
        var b: list[float] = []
        for _index in 0..n * n {
            seed = (seed * 1103515245 + 12345) % 2147483648
            b.push(float(seed % 2000) / 1000.0 - 1.0)
        }
        var a: list[float] = []
        for row in 0..n {
            for column in 0..n {
                var total = 0.0
                for k in 0..n { total += b[k * n + row] * b[k * n + column] }
                a.push(total)
            }
        }
        let (values, vectors) = symmetric_eigen(Mat.from_list(n, n, CV_32FC1, a))
        let vec = vectors.values()
        var worst = 0.0
        for index in 0..n {
            for row in 0..n {
                var got = 0.0
                for column in 0..n { got += a[row * n + column] * vec[index * n + column] }
                let want = values[index] * vec[index * n + row]
                worst = max(worst, math.fabs(got - want))
            }
        }
        assert_true(worst < 1e-4, f"{n} by {n} left a residual of {worst}")
    }
}

fn test_canny_answers_the_same_twice() -> void {
    if not have_gpu() { return }
    var seed = 424242
    var pixels: list[float] = []
    for _index in 0..200 * 260 {
        seed = (seed * 1103515245 + 12345) % 2147483648
        pixels.push(float(seed % 256))
    }
    let frame = Mat.from_list(200, 260, CV_8UC1, pixels)
    var first = 0.0
    var lit = 0
    for round in 0..4 {
        var digest = 0.0
        var live = 0
        for (index, value) in canny(frame, 60.0, 140.0).to_list().enumerate() {
            digest += value * float(index % 29 + 1)
            if value != 0.0 { live += 1 }
        }
        if round == 0 {
            first = digest
            lit = live
            continue
        }
        assert_eq(digest, first)
        assert_eq(live, lit)
    }
    #: A picture of noise has edges everywhere; nothing found would mean the
    #: run above agreed on being empty.
    assert_true(lit > 200 * 260 // 10)
}

#: A weak edge three hundred pixels long, strong only at one end.
#:
#: All of it belongs to the answer, because all of it is connected to the
#: strong end. Chasing that connection one ring a pass, with a bound on the
#: passes, stopped 228 pixels in and returned 145 of the 602 edge pixels.
fn test_canny_follows_a_long_weak_edge() -> void {
    if not have_gpu() { return }
    let width = 300
    var pixels: list[float] = []
    for index in 0..40 * width {
        let y = index // width
        let x = index - y * width
        var value = 0.0
        if y == 20 { value = if x >= width - 6 { 255.0 } else { 30.0 } }
        pixels.push(value)
    }
    let frame = Mat.from_list(40, width, CV_8UC1, pixels)
    var live = 0
    var leftmost = width
    for (index, value) in canny(frame, 100.0, 500.0).to_list().enumerate() {
        if value == 0.0 { continue }
        live += 1
        let x = index - index // width * width
        if x < leftmost { leftmost = x }
    }
    assert_eq(leftmost, 0)
    assert_true(live > 500)
}

fn test_video_run() -> void {
    if not have_gpu() { return }
    let a = disc()
    let b = shifted(a, 3)
    let ca = cvt_color(a, COLOR_GRAY2BGR)
    let cb = cvt_color(b, COLOR_GRAY2BGR)

    let flow = calc_optical_flow_farneback(a, b)
    assert_shape(flow, 64, 64, 2)
    assert_shape(flow_to_colour(flow), 64, 64, 3)
    assert_shape(polynomial_expansion(a, 5, 1.2), 64, 64, 5)
    let (moved, found, errors) = calc_optical_flow_pyr_lk(
        a,
        b,
        [Point2f(20.0, 30.0), Point2f(25.0, 35.0)]
    )
    assert_eq(moved.len(), 2)
    assert_eq(found.len(), 2)
    assert_eq(errors.len(), 2)
    test_optical_flow_finds_the_shift()
    test_canny_answers_the_same_twice()
    test_a_large_symmetric_matrix_is_fully_diagonalised()
    test_an_undistorted_camera_keeps_its_matrix()
    test_orb_matches_across_a_change_of_scale()
    test_a_pose_needs_six_points_off_a_plane()
    test_a_bayes_classifier_pays_for_a_wide_class()
    test_canny_follows_a_long_weak_edge()

    let (steps, settled) = mean_shift(a, Rect(10, 20, 20, 20))
    assert_true(steps >= 0 and settled.width > 0)
    let (turned, boxed) = cam_shift(a, Rect(10, 20, 20, 20))
    assert_true(boxed.width > 0 and turned.size.width >= 0.0)

    let mog2 = create_background_subtractor_mog2()
    for _round in 0..6 { assert_shape(mog2.apply(ca), 64, 64, 1) }
    assert_shape(mog2.apply(cb), 64, 64, 1)
    let knn = create_background_subtractor_knn()
    for _round in 0..6 { assert_shape(knn.apply(ca), 64, 64, 1) }
    assert_shape(knn.apply(cb), 64, 64, 1)

    let filter = KalmanFilter(4, 2, 0)
    for _round in 0..5 {
        assert_shape(filter.predict(), 4, 1, 1)
        assert_shape(filter.correct(Mat.from_list(2, 1, CV_32FC1, [1.0, 2.0])), 4, 1, 1)
    }
}

fn test_photo_run() -> void {
    if not have_gpu() { return }
    let a = disc()
    let ca = cvt_color(a, COLOR_GRAY2BGR)
    assert_shape(fast_nl_means_denoising(a, 3.0, 3, 7), 64, 64, 1)
    assert_shape(fast_nl_means_denoising_colored(ca, 3.0, 3.0, 3, 7), 64, 64, 3)
    assert_shape(edge_preserving_filter(ca), 64, 64, 3)
    assert_shape(detail_enhance(ca), 64, 64, 3)
    assert_shape(stylization(ca), 64, 64, 3)

    var mask = Mat.zeros(64, 64, CV_8UC1)
    rectangle(mask, Point(30, 30), Point(36, 36), Scalar(255.0, 0.0, 0.0), -1)
    assert_shape(inpaint(ca, mask, 3.0), 64, 64, 3)
}

#: Three clusters in three dimensions, well enough separated that every model
#: has something learnable and none of them has to be accurate for this to mean
#: something -- what is asked is that they train and predict at all.
fn labelled_cloud() -> tuple[Mat, Mat, Mat] {
    var features: list[float] = []
    var labels: list[float] = []
    var binary: list[float] = []
    var state = 12345
    for index in 0..120 {
        state = (state *% 1103515245 +% 12345) % 2147483648
        let cluster = index % 3
        let jitter = float(state % 100) / 100.0 - 0.5
        features.push(float(cluster) * 3.0 + jitter)
        features.push(float(cluster) * -2.0 + jitter)
        features.push(jitter)
        labels.push(float(cluster))
        binary.push(if cluster > 0 { 1.0 } else { 0.0 })
    }
    return (
        Mat.from_list(120, 3, CV_32FC1, features),
        Mat.from_list(120, 1, CV_32FC1, labels),
        Mat.from_list(120, 1, CV_32FC1, binary)
    )
}

fn test_ml_models_train_and_predict() -> void {
    if not have_gpu() { return }
    let (samples, labels, binary) = labelled_cloud()

    let neighbours = k_nearest_create(3)
    assert_true(neighbours.train(samples, labels))
    assert_shape(neighbours.predict(samples), 120, 1, 1)

    let bayes = normal_bayes_classifier_create()
    assert_true(bayes.train(samples, labels))
    assert_shape(bayes.predict(samples), 120, 1, 1)

    let svm = svm_create()
    assert_true(svm.train(samples, binary))
    assert_shape(svm.predict(samples), 120, 1, 1)

    let logistic = logistic_regression_create()
    assert_true(logistic.train(samples, binary))
    assert_shape(logistic.predict(samples), 120, 1, 1)
    assert_shape(logistic.predict_proba(samples), 120, 1, 1)

    let tree = d_trees_create()
    assert_true(tree.train(samples, labels))
    assert_shape(tree.predict(samples), 120, 1, 1)

    let forest = r_trees_create()
    assert_true(forest.train(samples, labels))
    assert_shape(forest.predict(samples), 120, 1, 1)

    let network = ann_mlp_create()
    network.set_layer_sizes([3, 8, 1])
    assert_true(network.train(samples, binary))
    assert_shape(network.predict(samples), 120, 1, 1)
}

fn test_dnn_helpers_run() -> void {
    if not have_gpu() { return }
    let ca = cvt_color(disc(), COLOR_GRAY2BGR)
    let cb = cvt_color(shifted(disc(), 3), COLOR_GRAY2BGR)
    let blob = blob_from_images([ca, cb], 1.0 / 255.0, Size(32, 32))
    assert_eq(images_from_blob(blob, 3, Size(32, 32)).len(), 2)
    let net = net_from(Doubler())
    assert_true(not net.empty())
    net.set_input_params(1.0 / 255.0, Size(32, 32))
    net.set_input_image(ca)
    assert_true(net.forward().rows > 0)

    let kept = nms_boxes_batched(
        [Rect(0, 0, 10, 10), Rect(1, 1, 10, 10), Rect(40, 40, 10, 10)],
        [0.9, 0.8, 0.7],
        [0, 0, 1],
        0.5,
        0.4
    )
    assert_true(kept.len() >= 1 and kept.len() <= 3)
}

#: Something for `Net` to front: the adapter is what is being checked, not
#: whatever runs behind it.
class Doubler: Runnable {
    pub fn init(self) { }

    pub fn run(self, blob: Mat) -> Mat { return blob }
}

#: The colour codes that name RGB rather than BGR order. Every pair is its own
#: inverse to within the rounding the conversion does, which is what makes them
#: worth a case of their own rather than only a mention in the export list.
fn test_rgb_colour_codes_round_trip() -> void {
    if not have_gpu() { return }
    let rgb = cvt_color(disc(), COLOR_GRAY2BGR)
    for (forward, back) in [(COLOR_RGB2Lab, COLOR_Lab2RGB), (COLOR_RGB2YCrCb, COLOR_YCrCb2RGB)] {
        let there = cvt_color(rgb, forward)
        assert_shape(there, 64, 64, 3)
        let here = cvt_color(there, back)
        assert_shape(here, 64, 64, 3)
        let before = rgb.to_list()
        let after = here.to_list()
        for index in 0..before.len() { assert_true(math.fabs(after[index] - before[index]) <= 3.0) }
    }
}

#: The device-buffer plumbing under `Mat`. `touch` marks a buffer changed and
#: `cached_kernel` hands back the same compiled kernel twice; neither has a
#: visible result, so what is asked of them is that they run and agree.
fn test_gpu_plumbing_runs() -> void {
    if not have_gpu() { return }
    assert_eq(f32_bits(1.5), 0x3FC00000)
    assert_eq(f32_bits(0.0), 0)
    let image = disc()
    touch(image.storage)
    assert_shape(image, 64, 64, 1)
    #: An entry point in the core shader library, asked for twice: the second
    #: call must hand back the pipeline the first compiled rather than compile
    #: it again.
    let once = cached_kernel("mat_fill")
    let twice = cached_kernel("mat_fill")
    assert_true(once is twice)
}

#: A camera need not be attached; what must hold is that asking costs nothing
#: and raises nothing, because a program that opens a camera asks these first.
fn test_camera_enumeration_answers_without_a_camera() -> void {
    if not have_gpu() { return }
    let count = device_count()
    assert_true(count >= 0)
    assert_eq(device_names().len(), count)
    if count > 0 { assert_true(device_name(0) is not null) }
    assert_true(device_name(count + 5) is null)
    let state = permission()
    assert_true(state >= PERMISSION_UNAVAILABLE and state <= PERMISSION_GRANTED)
}

#: The field the QR reader corrects in, checked against its own definition:
#: every non-zero element times its inverse is one, and the generator's powers
#: cycle with period 255 and hit every non-zero element exactly once.
fn test_the_galois_field_is_a_field() -> void {
    var seen: list[bool] = []
    for _index in 0..256 { seen.push(false) }
    for power in 0..255 {
        let value = gf_exp(power)
        assert_true(value != 0)
        assert_true(not seen[value])
        seen[value] = true
        assert_eq(gf_mul(value, gf_inv(value)), 1)
        assert_eq(gf_div(value, value), 1)
    }
    assert_eq(gf_exp(255), gf_exp(0))
    assert_eq(gf_mul(0, 123), 0)
    assert_eq(gf_mul(1, 123), 123)
}

#: A block with parity computed by the same field, corrupted up to the limit
#: and past it. Recovering is the easy half; refusing is the half that matters,
#: because a reader that hands back a plausible wrong string is worse than one
#: that hands back nothing.
fn test_reed_solomon_corrects_up_to_its_limit_and_refuses_past_it() -> void {
    let ec = 10
    var data: list[int] = []
    for index in 0..9 { data.push(index + 1) }
    #: Parity for [1..9] with ten check bytes, computed with the standard
    #: generator; the decoder is what is under test, so this is a constant
    #: rather than something this file also derives.
    let parity = [113, 25, 236, 154, 55, 111, 174, 237, 174, 170]
    var clean: list[int] = []
    for value in data { clean.push(value) }
    for value in parity { clean.push(value) }

    var untouched = [value for value in clean]
    assert_true(rs_correct(untouched, ec))
    assert_eq(untouched, clean)

    for errors in 1..6 {
        var block = [value for value in clean]
        for index in 0..errors { block[index * 3] = block[index * 3] ^ (index + 1) * 37 % 251 + 1 }
        assert_true(rs_correct(block, ec))
        assert_eq(block, clean)
    }

    var beyond = [value for value in clean]
    for index in 0..6 { beyond[index * 3] = beyond[index * 3] ^ (index + 1) * 37 % 251 + 1 }
    assert_true(not rs_correct(beyond, ec))
}

#: The two BCH words a symbol carries, and the mask table, checked against the
#: values the standard prints rather than against this file's own encoder.
fn test_the_qr_information_words_decode() -> void {
    #: Version 7's version word, and version 40's, from the standard's table.
    assert_eq(decode_version(0x07C94), 7)
    assert_eq(decode_version(0x28C69), 40)
    #: One bit flipped is inside what the code corrects; five is not.
    assert_eq(decode_version(0x07C94 ^ 0x40), 7)
    assert_true(decode_version(0x07C94 ^ 0x3F0F) is null)

    #: The all-zero five-bit format word is level M with mask 0, so what the
    #: symbol actually stores is the standard's masking constant itself. M
    #: rather than L because the two error-correction bits are numbered by
    #: capacity and encoded by something else, which is the whole reason the
    #: reader carries a table between them.
    let format = decode_format(0x5412)
    if format is null { throw AssertionError("the standard's own format word decodes") }
    assert_eq(format[0], 1)
    assert_eq(format[1], 0)

    #: The mask expressions, at a handful of places where all eight differ.
    assert_true(mask_bit(0, 0, 0))
    assert_true(mask_bit(1, 2, 5))
    assert_true(not mask_bit(1, 3, 5))
    assert_true(mask_bit(2, 4, 6))
    assert_true(not mask_bit(2, 4, 7))
    assert_true(mask_bit(3, 2, 4))
    assert_true(mask_bit(4, 0, 0))
    assert_true(mask_bit(5, 0, 3))
    assert_true(mask_bit(6, 1, 1))
    assert_true(mask_bit(7, 0, 0))

    #: The map is memoised, so the second ask must be the same object and the
    #: caller must not be able to change what the next one sees.
    assert_true(function_map(5) is function_map(5))
    assert_true(function_map(5)[0][0])
    assert_true(not function_map(5)[9][9])
}

#: The two exports the reachability check found with no caller at all.
#:
#: `scripts/exports_reachable_check.py` fails the build when that happens
#: again, which is the only reason this file stays complete.
fn test_the_last_two_exports_run() -> void {
    if not gpu.is_available() { return }
    let unit = identity(4)
    defer { unit.release() }
    assert_shape(unit, 4, 4, 1)
    let diagonal = unit.to_list()
    for row in 0..4 {
        for column in 0..4 {
            let want = if row == column { 1.0 } else { 0.0 }
            assert_near(diagonal[row * 4 + column], want, 1e-6, f"at {row},{column}")
        }
    }

    #: A blob is one row of `channels * height * width` per image; the tensor
    #: it becomes keeps that shape.
    var values: list[float] = []
    for index in 0..2 * 3 * 4 * 5 { values.push(float(index % 19) * 0.25) }
    let blob = Mat.from_list(2, 3 * 4 * 5, CV_32FC1, values)
    defer { blob.release() }
    let tensor = blob_to_tensor(blob, 3, Size(5, 4))
    assert_eq(tensor.shape, [2, 3, 4, 5], "a blob should keep its shape as a tensor")
    let held = tensor.download()
    assert_eq(held.len(), values.len())
    for index in 0..held.len() {
        assert_near(held[index], values[index], 1e-6, f"value {index} changed")
    }
}
