#: Unit tests for the parts of jaicv that do not need OpenCV to check them:
#: `Mat` semantics, arithmetic and statistics, and codec round trips.
#:
#: The differential comparison against OpenCV itself lives in
#: `test_against_opencv.jai`.
import std.gpu as gpu
import std.math
from std.io import Path
from std.test import assert_eq, assert_near, assert_throws, assert_true
from jaicv.core.mat import Mat, flip, merge
from jaicv.core.types import CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC3, CV_32FC1, CV_32FC3
from jaicv.core.types import CV_MAKETYPE, CV_MAT_CN, CV_MAT_DEPTH, CV_8U, CV_32F
from jaicv.core.types import CMP_GT, Point, Range, Rect, Scalar, Size, type_to_string
from jaicv.core.arith import add, absdiff, bitwise_not, compare_scalar, divide, in_range
from jaicv.core.arith import lut, multiply, subtract
from jaicv.core.stats import count_non_zero, mean, min_max_loc, norm, sum_mat
from jaicv.core.cluster import kmeans, PCA
from jaicv.core.dft import dft, idft, DFT_COMPLEX_OUTPUT, DFT_REAL_OUTPUT, DFT_SCALE
from jaicv.imgcodecs.codecs import imdecode, imencode, sniff
from jaicv.imgcodecs.codecs import IMREAD_COLOR, IMREAD_GRAYSCALE, IMREAD_UNCHANGED
from jaicv.imgproc.geometric import get_perspective_transform, perspective_transform
from jaicv.core.types import Point2f

fn have_gpu() -> bool { return gpu.is_available() }

fn ramp(rows: int, cols: int, type_code: int, seed: int) -> Mat {
    let channels = CV_MAT_CN(type_code)
    var values: list[float] = []
    for index in 0..rows * cols * channels {
        values.push(float((index * 37 + seed * 11) % 251))
    }
    return Mat.from_list(rows, cols, type_code, values)
}

fn test_type_codes_pack_depth_and_channels() -> void {
    assert_eq(CV_MAT_DEPTH(CV_8UC3), CV_8U)
    assert_eq(CV_MAT_CN(CV_8UC3), 3)
    assert_eq(type_to_string(CV_32FC3), "CV_32FC3")
    assert_eq(CV_MAKETYPE(CV_32F, 4), CV_MAKETYPE(CV_32F, 4))
}

fn test_a_region_of_interest_writes_through_to_its_parent() -> void {
    if not have_gpu() { return }
    let image = Mat.zeros(6, 8, CV_8UC3)
    let window = image.roi(Rect(2, 1, 3, 2))
    window.set_to(Scalar(7.0, 8.0, 9.0))
    #: The window and its parent share one allocation, so a write through
    #: either has to be visible in the other — including through the host
    #: mirror, which is keyed on the allocation rather than the view.
    assert_near(image.at(1, 2, 0), 7.0, 1e-6)
    assert_near(image.at(2, 4, 2), 9.0, 1e-6)
    assert_near(image.at(0, 0, 0), 0.0, 1e-6)
    assert_near(image.at(3, 2, 0), 0.0, 1e-6)
    assert_eq(window.rows, 2)
    assert_eq(window.cols, 3)
}

fn test_a_region_of_interest_outside_the_image_is_refused() -> void {
    if not have_gpu() { return }
    let image = Mat.zeros(4, 4, CV_8UC1)
    assert_throws(ValueError, || image.roi(Rect(2, 2, 4, 4)))
}

fn test_split_and_merge_are_inverses() -> void {
    if not have_gpu() { return }
    let image = ramp(5, 7, CV_8UC3, 3)
    let planes = image.split()
    assert_eq(planes.len(), 3)
    let joined = merge(planes)
    let before = image.to_list()
    let after = joined.to_list()
    for index in 0..before.len() { assert_near(after[index], before[index], 1e-6) }
}

fn test_convert_to_rounds_and_saturates() -> void {
    if not have_gpu() { return }
    let source = Mat.from_list(1, 4, CV_32FC1, [-5.0, 0.4, 200.6, 400.0])
    let out = source.convert_to(CV_8UC1)
    assert_eq(out.to_list(), [0.0, 0.0, 201.0, 255.0])
}

fn test_arithmetic_saturates_in_the_destination_depth() -> void {
    if not have_gpu() { return }
    let a = Mat.from_list(1, 3, CV_8UC1, [10.0, 200.0, 30.0])
    let b = Mat.from_list(1, 3, CV_8UC1, [5.0, 100.0, 60.0])
    assert_eq(add(a, b).to_list(), [15.0, 255.0, 90.0])
    assert_eq(subtract(a, b).to_list(), [5.0, 100.0, 0.0])
    assert_eq(absdiff(a, b).to_list(), [5.0, 100.0, 30.0])
    assert_eq(multiply(a, b, 0.01).to_list(), [0.0, 200.0, 18.0])
    assert_eq(divide(a, b).to_list(), [2.0, 2.0, 0.0])
    assert_eq(bitwise_not(a).to_list(), [245.0, 55.0, 225.0])
}

fn test_dividing_by_zero_yields_zero() -> void {
    if not have_gpu() { return }
    let a = Mat.from_list(1, 2, CV_32FC1, [4.0, 9.0])
    let b = Mat.from_list(1, 2, CV_32FC1, [2.0, 0.0])
    assert_eq(divide(a, b).to_list(), [2.0, 0.0])
}

fn test_compare_and_in_range_produce_masks() -> void {
    if not have_gpu() { return }
    let values = Mat.from_list(1, 4, CV_8UC1, [10.0, 200.0, 30.0, 40.0])
    assert_eq(compare_scalar(values, 50.0, CMP_GT).to_list(), [0.0, 255.0, 0.0, 0.0])
    let colour = Mat.from_list(1, 2, CV_8UC3, [10.0, 120.0, 200.0, 250.0, 5.0, 5.0])
    let mask = in_range(colour, Scalar(0.0, 100.0, 100.0), Scalar(50.0, 255.0, 255.0))
    assert_eq(mask.channels(), 1)
    assert_eq(mask.to_list(), [255.0, 0.0])
}

fn test_a_lookup_table_maps_every_element() -> void {
    if not have_gpu() { return }
    var table_values: list[float] = []
    for index in 0..256 { table_values.push(float(255 - index)) }
    let table = Mat.from_list(1, 256, CV_8UC1, table_values)
    let values = Mat.from_list(1, 3, CV_8UC1, [0.0, 128.0, 255.0])
    assert_eq(lut(values, table).to_list(), [255.0, 127.0, 0.0])
}

fn test_statistics_agree_with_the_definitions() -> void {
    if not have_gpu() { return }
    let m = Mat.from_list(2, 3, CV_32FC1, [1.0, 5.0, 3.0, -2.0, 9.0, 0.0])
    let (low, high, low_at, high_at) = min_max_loc(m)
    assert_near(low, -2.0, 1e-6)
    assert_near(high, 9.0, 1e-6)
    assert_eq(low_at, Point(0, 1))
    assert_eq(high_at, Point(1, 1))
    assert_near(sum_mat(m)[0], 16.0, 1e-4)
    assert_near(mean(m)[0], 16.0 / 6.0, 1e-6)
    assert_eq(count_non_zero(m), 5)
    assert_near(norm(m), math.sqrt(120.0), 1e-4)
}

fn test_flip_mirrors_each_axis() -> void {
    if not have_gpu() { return }
    let m = Mat.from_list(2, 3, CV_8UC1, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert_eq(flip(m, 1).to_list(), [3.0, 2.0, 1.0, 6.0, 5.0, 4.0])
    assert_eq(flip(m, 0).to_list(), [4.0, 5.0, 6.0, 1.0, 2.0, 3.0])
    assert_eq(flip(m, -1).to_list(), [6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
}

fn test_transpose_swaps_the_axes() -> void {
    if not have_gpu() { return }
    let m = Mat.from_list(2, 3, CV_8UC1, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    let t = m.transpose()
    assert_eq(t.rows, 3)
    assert_eq(t.cols, 2)
    assert_eq(t.to_list(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
}

fn round_trip(extension: str, image: Mat) -> Mat {
    let encoded = imencode(extension, image)
    return imdecode(encoded, IMREAD_UNCHANGED)
}

fn test_the_lossless_codecs_round_trip_exactly() -> void {
    if not have_gpu() { return }
    for type_code in [CV_8UC1, CV_8UC3] {
        let image = ramp(9, 11, type_code, 5)
        for extension in ["png", "bmp", "ppm"] {
            if extension == "ppm" and CV_MAT_CN(type_code) == 4 { continue }
            let back = round_trip(extension, image)
            assert_eq(back.rows, image.rows)
            assert_eq(back.cols, image.cols)
            assert_eq(back.channels(), image.channels())
            let before = image.to_list()
            let after = back.to_list()
            for index in 0..before.len() { assert_near(after[index], before[index], 1e-6) }
        }
    }
}

fn test_png_carries_four_channels_and_sixteen_bits() -> void {
    if not have_gpu() { return }
    let alpha = ramp(4, 5, CV_8UC4, 6)
    let back = round_trip("png", alpha)
    assert_eq(back.channels(), 4)
    assert_eq(back.to_list(), alpha.to_list())

    let deep = ramp(3, 4, CV_16UC3, 7).convert_to(CV_16UC3, 257.0, 0.0)
    let deep_back = round_trip("png", deep)
    assert_eq(deep_back.depth(), deep.depth())
    assert_eq(deep_back.to_list(), deep.to_list())
}

fn test_jpeg_round_trips_within_its_quantisation() -> void {
    if not have_gpu() { return }
    #: A smooth image, so the comparison measures the codec rather than the
    #: high frequencies JPEG is designed to throw away.
    var values: list[float] = []
    for y in 0..24 {
        for x in 0..32 {
            values.push(float(120 + int(60.0 * math.sin(float(x) / 6.0))))
            values.push(float(120 + int(50.0 * math.cos(float(y) / 5.0))))
            values.push(float(100 + int(40.0 * math.sin(float(x + y) / 7.0))))
        }
    }
    let image = Mat.from_list(24, 32, CV_8UC3, values)
    let back = imdecode(imencode("jpg", image, [1, 100]), IMREAD_UNCHANGED)
    assert_eq(back.rows, 24)
    assert_eq(back.channels(), 3)
    let before = image.to_list()
    let after = back.to_list()
    var worst = 0.0
    for index in 0..before.len() {
        let drift = math.fabs(after[index] - before[index])
        if drift > worst { worst = drift }
    }
    assert_true(worst <= 2.0)
}

fn test_the_reader_recognises_each_format_from_its_bytes() -> void {
    if not have_gpu() { return }
    let image = ramp(4, 6, CV_8UC3, 8)
    assert_eq(sniff(imencode("png", image)), "png")
    assert_eq(sniff(imencode("bmp", image)), "bmp")
    assert_eq(sniff(imencode("ppm", image)), "pnm")
    assert_eq(sniff(imencode("jpg", image)), "jpeg")
    assert_eq(sniff(bytes([1, 2, 3, 4, 5, 6, 7, 8, 9])), "")
}

fn test_read_flags_force_the_channel_count() -> void {
    if not have_gpu() { return }
    let colour = imencode("png", ramp(5, 5, CV_8UC3, 9))
    assert_eq(imdecode(colour, IMREAD_GRAYSCALE).channels(), 1)
    assert_eq(imdecode(colour, IMREAD_COLOR).channels(), 3)
    let grey = imencode("png", ramp(5, 5, CV_8UC1, 10))
    assert_eq(imdecode(grey, IMREAD_COLOR).channels(), 3)
    assert_eq(imdecode(grey, IMREAD_UNCHANGED).channels(), 1)
}

fn test_a_perspective_transform_maps_its_own_corners() -> void {
    if not have_gpu() { return }
    let source = [Point2f(0.0, 0.0), Point2f(9.0, 0.0), Point2f(9.0, 7.0), Point2f(0.0, 7.0)]
    let target = [Point2f(1.0, 1.0), Point2f(8.0, 0.5), Point2f(9.0, 6.0), Point2f(0.5, 7.0)]
    let matrix = get_perspective_transform(source, target)
    let mapped = perspective_transform(source, matrix)
    for index in 0..4 {
        assert_near(mapped[index].x, target[index].x, 1e-3)
        assert_near(mapped[index].y, target[index].y, 1e-3)
    }
}

fn test_rectangles_intersect_and_combine() -> void {
    let a = Rect(0, 0, 10, 10)
    let b = Rect(5, 5, 10, 10)
    assert_eq(a.intersect(b), Rect(5, 5, 5, 5))
    assert_eq(a.union_with(b), Rect(0, 0, 15, 15))
    assert_true(a.contains(Point(9, 9)))
    assert_true(not a.contains(Point(10, 9)))
    assert_true(a.intersect(Rect(20, 20, 3, 3)).empty())
}

fn test_a_range_covers_a_half_open_span() -> void {
    let span = Range(2, 7)
    assert_eq(span.size(), 5)
    assert_true(not span.empty())
    assert_true(Range.all().is_all())
    assert_true(Range(4, 4).empty())
}

fn test_a_size_reports_its_area() -> void {
    assert_eq(Size(4, 5).area(), 20)
    assert_true(Size(0, 5).empty())
}

fn test_kmeans_finds_clusters_that_are_there() -> void {
    if not have_gpu() { return }
    var values: list[float] = []
    let centres = [[0.0, 0.0], [8.0, 1.0], [3.0, 9.0]]
    for group in 0..3 {
        for index in 0..12 {
            let jitter = float(index * 37 % 11) / 40.0 - 0.125
            values.push(centres[group][0] + jitter)
            values.push(centres[group][1] - jitter)
        }
    }
    let samples = Mat.from_list(36, 2, CV_32FC1, values)
    let (compactness, labels, found) = kmeans(samples, 3, 5)
    assert_true(compactness < 2.0, f"the groups should be tight, got {compactness}")
    assert_eq(found.rows, 3)

    #: Every sample of a group has to land in the same group as the rest of it,
    #: whichever number that group ended up with.
    let assigned = labels.to_list()
    for group in 0..3 {
        let first = assigned[group * 12]
        for index in 0..12 { assert_eq(assigned[group * 12 + index], first) }
    }
    assert_true(assigned[0] != assigned[12])
    assert_true(assigned[12] != assigned[24])
}

fn test_a_transform_and_its_inverse_return_the_signal() -> void {
    if not have_gpu() { return }
    var values: list[float] = []
    for index in 0..24 { values.push(float(index * 53 % 29) - 14.0) }
    let signal = Mat.from_list(4, 6, CV_32FC1, values)
    let spectrum = dft(signal, DFT_COMPLEX_OUTPUT)
    assert_eq(spectrum.channels(), 2)
    let back = idft(spectrum, DFT_SCALE | DFT_REAL_OUTPUT).to_list()
    for index in 0..values.len() { assert_near(back[index], values[index], 1e-3) }
}

fn test_principal_components_line_up_with_the_spread() -> void {
    if not have_gpu() { return }
    var values: list[float] = []
    for index in 0..40 {
        let t = float(index) * 0.25 - 5.0
        #: A cloud along the line y = 2x, so the first component has to point
        #: that way whatever sign it comes back with.
        values.push(t)
        values.push(2.0 * t)
    }
    let analysis = PCA(Mat.from_list(40, 2, CV_32FC1, values), 1)
    let direction = analysis.eigenvectors.to_list()
    let ratio = direction[1] / direction[0]
    assert_near(ratio, 2.0, 1e-6)
}
