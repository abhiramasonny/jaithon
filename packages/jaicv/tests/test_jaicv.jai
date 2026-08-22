#: Unit tests for the parts of jaicv that do not need OpenCV to check them:
#: `Mat` semantics, arithmetic and statistics, and codec round trips.
#:
#: The differential comparison against OpenCV itself lives in
#: `test_against_opencv.jai`.
import std.gpu as gpu
import std.math
from std.io import Path, write_bytes
from std.os import temp_dir
from std.test import assert_eq, assert_near, assert_throws, assert_true
from jaicv.imgproc.filter import filter_2d, sep_filter_2d
from jaicv.imgproc.morph import dilate, erode, get_structuring_element, MORPH_RECT
from jaicv.core.types import BORDER_CONSTANT, BORDER_REFLECT, BORDER_REFLECT_101
from jaicv.core.types import BORDER_REPLICATE
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
from jaicv.imgproc.drawing import circle, fill_poly, line, rectangle
from jaicv.videoio.capture import VideoCapture, CAP_PROP_FRAME_COUNT, CAP_PROP_FPS
from jaicv.videoio.capture import CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH
from jaicv.videoio.capture import CAP_PROP_POS_FRAMES
from jaicv.videoio.writer import fourcc, VideoWriter
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

#: Five frames with a bar that moves and brightens, so a reader that dropped
#: one, repeated one or reordered them shows it in the pixels rather than only
#: in the count.
fn clip_frames(size: Size) -> list[Mat] {
    var frames: list[Mat] = []
    for n in 0..5 {
        let frame = Mat.zeros(size.height, size.width, CV_8UC3)
        rectangle(
            frame,
            Point(2 + n * 4, 2),
            Point(20 + n * 4, 24),
            Scalar(20.0 * float(n), 200.0, 40.0),
            -1
        )
        frames.push(frame)
    }
    return frames
}

fn write_clip(path: str, codec: int, size: Size, frames: list[Mat]) -> void {
    let writer = VideoWriter(path, codec, 12.0, size)
    for frame in frames { writer.write(frame) }
    writer.release()
}

#: The uncompressed codec stores the pixels as they were, so the round trip is
#: exact and any disagreement is a container bug rather than a codec one.
fn test_an_uncompressed_video_round_trips_exactly() -> void {
    if not have_gpu() { return }
    let size = Size(48, 32)
    let frames = clip_frames(size)
    let path = Path(temp_dir()).join("jaicv_video_raw.avi").to_str()
    write_clip(path, 0, size, frames)

    let clip = VideoCapture(path)
    assert_true(clip.is_opened())
    assert_eq(int(clip.get(CAP_PROP_FRAME_COUNT)), 5)
    assert_eq(int(clip.get(CAP_PROP_FRAME_WIDTH)), size.width)
    assert_eq(int(clip.get(CAP_PROP_FRAME_HEIGHT)), size.height)
    assert_near(clip.get(CAP_PROP_FPS), 12.0, 1e-6)

    var seen = 0
    loop {
        let (ok, frame) = clip.read()
        if not ok { break }
        assert_eq(frame.rows, size.height)
        assert_eq(frame.cols, size.width)
        assert_eq(frame.channels(), 3)
        assert_eq(frame.to_list(), frames[seen].to_list())
        seen += 1
    }
    assert_eq(seen, 5)
    clip.release()
    Path(path).remove()
}

#: Seeking is what an indexed reader buys over a streaming one, so it is
#: checked against the frame that was written rather than only for a `true`.
fn test_a_video_can_be_seeked_by_frame() -> void {
    if not have_gpu() { return }
    let size = Size(32, 24)
    let frames = clip_frames(size)
    let path = Path(temp_dir()).join("jaicv_video_seek.avi").to_str()
    write_clip(path, 0, size, frames)

    let clip = VideoCapture(path)
    for wanted in [3, 0, 4, 1] {
        assert_true(clip.set(CAP_PROP_POS_FRAMES, float(wanted)))
        let (ok, frame) = clip.read()
        assert_true(ok)
        assert_eq(frame.to_list(), frames[wanted].to_list())
        assert_eq(int(clip.get(CAP_PROP_POS_FRAMES)), wanted + 1)
    }
    assert_true(not clip.set(CAP_PROP_POS_FRAMES, 99.0))
    clip.release()
    Path(path).remove()
}

#: Motion JPEG is lossy, so this asks only that every frame arrives, is the
#: right shape, and is near what went in.
fn test_a_motion_jpeg_video_round_trips_within_its_quantisation() -> void {
    if not have_gpu() { return }
    let size = Size(48, 32)
    let frames = clip_frames(size)
    let path = Path(temp_dir()).join("jaicv_video_mjpg.avi").to_str()
    write_clip(path, fourcc("M", "J", "P", "G"), size, frames)

    let clip = VideoCapture(path)
    assert_eq(int(clip.get(CAP_PROP_FRAME_COUNT)), 5)
    var seen = 0
    loop {
        let (ok, frame) = clip.read()
        if not ok { break }
        assert_eq(frame.rows, size.height)
        assert_eq(frame.cols, size.width)
        let want = mean(frames[seen])
        let got = mean(frame)
        for channel in 0..3 { assert_near(got[channel], want[channel], 6.0) }
        seen += 1
    }
    assert_eq(seen, 5)
    clip.release()
    Path(path).remove()
}

fn test_opening_something_that_is_not_a_video_reports_rather_than_raises() -> void {
    if not have_gpu() { return }
    let path = Path(temp_dir()).join("jaicv_video_not.avi").to_str()
    write_bytes(path, bytes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]))
    let clip = VideoCapture(path)
    assert_true(not clip.is_opened())
    assert_true(clip.why().len() > 0)
    let (ok, _frame) = clip.read()
    assert_true(not ok)
    Path(path).remove()
}

#: Drawing puts the same pixels down whether the picture is on the host or
#: still on the device.
#:
#: The two are different code paths: one writes boxed numbers into a host
#: mirror, the other writes floats straight into the device buffer's own
#: memory, which is shared. They have to agree exactly, because which one runs
#: depends only on whether something happened to pull the picture over
#: earlier -- a difference between them would show up as a frame that looks
#: subtly wrong only sometimes.
fn drawn_on(force_host: bool) -> list[float] {
    var frame = Mat.zeros(90, 120, CV_8UC3)
    #: Reading the pixels is what settles the picture on the host, and the
    #: painter draws wherever it finds them.
    if force_host { let _pull = frame.values() }
    rectangle(frame, Point(4, 4), Point(40, 30), Scalar(10.0, 200.0, 30.0), -1)
    rectangle(frame, Point(50, 8), Point(110, 50), Scalar(200.0, 10.0, 30.0), 3)
    circle(frame, Point(70, 62), 20, Scalar(0.0, 0.0, 255.0), -1)
    circle(frame, Point(25, 62), 14, Scalar(255.0, 255.0, 0.0), 2)
    line(frame, Point(0, 0), Point(119, 89), Scalar(255.0, 255.0, 255.0), 1)
    #: Off the edge at both ends, so the clipping is exercised too.
    line(frame, Point(-30, 45), Point(150, 20), Scalar(90.0, 90.0, 250.0), 2)
    fill_poly(frame, [[Point(90, 55), Point(118, 62), Point(100, 88)]], Scalar(40.0, 180.0, 180.0))
    return frame.values()
}

fn test_drawing_agrees_wherever_the_pixels_live() -> void {
    if not gpu.is_available() { return }
    let on_device = drawn_on(false)
    let on_host = drawn_on(true)
    assert_eq(on_device.len(), on_host.len())
    var differing = 0
    for index in 0..on_device.len() {
        if on_device[index] != on_host[index] { differing += 1 }
    }
    assert_eq(differing, 0)
}

#: Shapes that fall off the picture, or have no area, or are absurdly far
#: away, put down the same pixels on either surface and none outside it.
#:
#: The raster primitives write into a host mirror or into a device buffer's own
#: memory, and the second has no list bounds behind it -- a coordinate that got
#: past the clipping would be a stray write into whatever the allocator handed
#: out next. These are the cases where the clipping has to hold.
fn awkwardly_drawn(force_host: bool, channels: int) -> list[float] {
    var frame = Mat.zeros(17, 23, CV_MAKETYPE(CV_8U, channels))
    if force_host { let _pull = frame.values() }
    let ink = Scalar(200.0, 30.0, 90.0, 255.0)
    #: Wholly outside, on each side in turn.
    rectangle(frame, Point(-90, -90), Point(-40, -40), ink, -1)
    rectangle(frame, Point(400, 400), Point(500, 500), ink, -1)
    line(frame, Point(-50, -50), Point(-10, -60), ink, 1)
    line(frame, Point(90, 3), Point(120, 9), ink, 2)
    #: Straddling every edge.
    rectangle(frame, Point(-8, -8), Point(9, 9), ink, -1)
    rectangle(frame, Point(15, 10), Point(60, 60), ink, 2)
    line(frame, Point(-40, 8), Point(70, 12), ink, 1)
    line(frame, Point(11, -30), Point(13, 50), ink, 3)
    #: No area at all.
    rectangle(frame, Point(5, 5), Point(5, 5), ink, -1)
    line(frame, Point(7, 7), Point(7, 7), ink, 1)
    circle(frame, Point(8, 8), 0, ink, -1)
    #: Collinear corners, which give the scanline fill nothing to walk.
    fill_poly(frame, [[Point(2, 2), Point(9, 2), Point(16, 2)]], ink)
    #: Far enough out that the clipping arithmetic has room to go wrong.
    line(frame, Point(-100000, -100000), Point(100000, 100000), ink, 1)
    circle(frame, Point(4, 4), 40000, ink, 1)
    return frame.values()
}

fn test_drawing_off_the_edge_agrees_and_stays_inside() -> void {
    if not gpu.is_available() { return }
    for channels in [1, 3, 4] {
        let on_device = awkwardly_drawn(false, channels)
        let on_host = awkwardly_drawn(true, channels)
        assert_eq(on_device.len(), 17 * 23 * channels, f"{channels} channels, wrong size")
        assert_eq(on_device.len(), on_host.len())
        var differing = 0
        for index in 0..on_device.len() {
            if on_device[index] != on_host[index] { differing += 1 }
        }
        assert_eq(differing, 0, f"{channels} channels differ between host and device")
    }
}

#: A one-pixel picture is the smallest thing every clip has to survive.
fn test_drawing_on_a_single_pixel() -> void {
    var frame = Mat.zeros(1, 1, CV_8UC3)
    rectangle(frame, Point(-5, -5), Point(5, 5), Scalar(10.0, 20.0, 30.0), -1)
    assert_eq(frame.values(), [10.0, 20.0, 30.0])
    line(frame, Point(-9, 0), Point(9, 0), Scalar(40.0, 50.0, 60.0), 1)
    assert_eq(frame.values(), [40.0, 50.0, 60.0])
}

#: A separable filter is two one-dimensional passes, and it has to land where
#: the two-dimensional filter built from the same outer product lands.
#:
#: It once did the whole square window per pixel, which is correct and is the
#: work being separable exists to avoid. Splitting it means each pass extends
#: only its own axis, so a border handled on the wrong axis, or the two kernels
#: swapped, now has somewhere to hide -- an asymmetric kernel and a reflecting
#: border are what bring both out.
fn test_a_separable_filter_matches_the_square_it_stands_for() -> void {
    if not gpu.is_available() { return }
    let side = 37
    var pixels: list[float] = []
    for index in 0..side * side { pixels.push(float(index * 41 % 251) * 0.01 - 1.25) }
    let source = Mat.from_list(side, side, CV_32FC1, pixels)

    let across = [0.11, -0.27, 0.63, 0.19, -0.08]
    let down = [-0.31, 0.44, 0.22, 0.07, 0.51, -0.13, 0.29]
    let row = Mat.from_list(1, across.len(), CV_32FC1, across)
    let column = Mat.from_list(down.len(), 1, CV_32FC1, down)
    var square: list[float] = []
    for vertical in down {
        for horizontal in across { square.push(vertical * horizontal) }
    }
    let whole = Mat.from_list(down.len(), across.len(), CV_32FC1, square)

    for border in [BORDER_REPLICATE, BORDER_REFLECT, BORDER_REFLECT_101, BORDER_CONSTANT] {
        let apart = sep_filter_2d(source, CV_32F, row, column, border_mode: border)
        let together = filter_2d(source, CV_32F, whole, border_mode: border)
        let got = apart.to_list()
        let want = together.to_list()
        assert_eq(got.len(), want.len())
        for index in 0..got.len() {
            assert_near(got[index], want[index], 1e-4, f"border {border}, pixel {index}")
        }
        apart.release()
        together.release()
    }
    source.release()
    row.release()
    column.release()
    whole.release()
}

#: Erosion and dilation through a solid rectangle, against the window walked
#: here.
#:
#: A solid element is separable -- the least value in a box is the least of the
#: row minima -- and that is the path a rectangle now takes. An ellipse is not
#: separable and keeps the square sweep, so both are checked.
fn test_a_rectangular_morphology_matches_the_window_it_stands_for() -> void {
    if not gpu.is_available() { return }
    let rows = 23
    let cols = 29
    var pixels: list[float] = []
    for index in 0..rows * cols { pixels.push(float(index * 61 % 197) * 0.01) }
    let source = Mat.from_list(rows, cols, CV_32FC1, pixels)
    let width = 5
    let height = 3
    let element = get_structuring_element(MORPH_RECT, Size(width, height))
    let anchor_x = width // 2
    let anchor_y = height // 2

    for growing in [false, true] {
        let got = if growing { dilate(source, element) } else { erode(source, element) }
        let values = got.to_list()
        for y in 0..rows {
            for x in 0..cols {
                var best = pixels[y * cols + x]
                for ky in 0..height {
                    for kx in 0..width {
                        #: The default border replicates, so a step outside the
                        #: picture takes the nearest edge.
                        let sy = min(max(y + ky - anchor_y, 0), rows - 1)
                        let sx = min(max(x + kx - anchor_x, 0), cols - 1)
                        let sample = pixels[sy * cols + sx]
                        if growing {
                            if sample > best { best = sample }
                        } elif sample < best {
                            best = sample
                        }
                    }
                }
                assert_near(values[y * cols + x], best, 1e-5, f"grow {growing} at {y},{x}")
            }
        }
        got.release()
    }
    source.release()
    element.release()
}
