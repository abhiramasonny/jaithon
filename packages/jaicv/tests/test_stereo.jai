#: Depth from two cameras.
#:
#: `test_against_opencv.jai` compares the rectifying rotations against OpenCV's,
#: which is the part where there is one right answer. The rest is checked by
#: what it has to DO: a correspondence must land on the same row after
#: rectification, a matcher must recover a disparity that was put there, and a
#: disparity must reproject to the depth the geometry says.
#:
#: The principal point in `P1`, `P2` and `Q` is deliberately not compared with
#: OpenCV's. OpenCV recomputes it by projecting the image border and fitting
#: the largest usable rectangle inside it -- the `alpha` machinery -- and jaicv
#: keeps the cameras' own, so the two drift by a few pixels on a pair with any
#: real rotation. What matters is that the pair is consistent with itself,
#: which is what `test_a_rectified_pair_puts_matches_on_one_row` asks.
import std.math
import std.gpu as gpu
from std.test import assert_eq, assert_near, assert_throws, assert_true
from jaicv.calib3d.stereo import reproject_image_to_3d, stereo_rectify
from jaicv.calib3d.stereo import StereoBM, StereoSGBM
from jaicv.calib3d.transform import rodrigues
from jaicv.core.linalg import matmul
from jaicv.core.mat import Mat
from jaicv.core.types import CV_8UC1, CV_8UC3, CV_32FC1, Size

fn have_gpu() -> bool { return gpu.is_available() }

#: A textured field and the same field shifted, so every pixel's true
#: disparity is known. The texture has no repeating period as short as the
#: disparity range, or a matcher would rightly find more than one answer.
fn shifted_pair(rows: int, cols: int, shift: int) -> tuple[Mat, Mat] {
    var left: list[float] = []
    var right: list[float] = []
    for y in 0..rows {
        for x in 0..cols { left.push(float((x * 37 + y * 61 + x * y * 13) % 251)) }
    }
    for y in 0..rows {
        for x in 0..cols {
            let sx = x + shift
            right.push(float((sx * 37 + y * 61 + sx * y * 13) % 251))
        }
    }
    return (Mat.from_list(rows, cols, CV_8UC1, left), Mat.from_list(rows, cols, CV_8UC1, right))
}

#: How many pixels away from the border came back within a pixel of the truth.
fn agreement(disparity: Mat, want: int, margin: int) -> float {
    let values = disparity.to_list()
    var hit = 0
    var seen = 0
    for y in margin..disparity.rows - margin {
        for x in margin + want..disparity.cols - margin {
            seen += 1
            if math.fabs(values[y * disparity.cols + x] / 16.0 - float(want)) <= 1.0 { hit += 1 }
        }
    }
    if seen == 0 { return 0.0 }
    return float(hit) / float(seen)
}

fn test_block_matching_finds_a_known_shift() -> void {
    if not have_gpu() { return }
    for shift in [4, 12, 20] {
        let (left, right) = shifted_pair(48, 96, shift)
        let matcher = StereoBM(num_disparities: 32, block_size: 11)
        let disparity = matcher.compute(left, right)
        assert_eq(disparity.rows, 48)
        assert_eq(disparity.cols, 96)
        assert_true(agreement(disparity, shift, 8) > 0.98)
    }
}

fn test_semi_global_matching_finds_a_known_shift() -> void {
    if not have_gpu() { return }
    for shift in [4, 12, 20] {
        let (left, right) = shifted_pair(48, 96, shift)
        let matcher = StereoSGBM(num_disparities: 32, block_size: 5)
        assert_true(agreement(matcher.compute(left, right), shift, 8) > 0.98)
    }
}

#: Aggregating along four directions instead of eight is cheaper and should
#: still find a shift this clear.
fn test_four_direction_aggregation_also_works() -> void {
    if not have_gpu() { return }
    let (left, right) = shifted_pair(48, 96, 12)
    let matcher = StereoSGBM(num_disparities: 32, block_size: 5)
    matcher.directions = 4
    assert_true(agreement(matcher.compute(left, right), 12, 8) > 0.95)
}

#: A pixel whose true disparity is past the search range has no right answer,
#: and the matcher must say so rather than pick the closest thing it looked at.
fn test_a_disparity_past_the_range_is_reported_as_missing() -> void {
    if not have_gpu() { return }
    let (left, right) = shifted_pair(48, 96, 40)
    let matcher = StereoBM(num_disparities: 16, block_size: 11)
    matcher.uniqueness_ratio = 0.5
    let values = matcher.compute(left, right).to_list()
    var missing = 0
    for value in values {
        if value < 0.0 { missing += 1 }
    }
    #: Not every pixel: a periodic texture can match by accident. Most of them.
    assert_true(float(missing) / float(values.len()) > 0.5)
}

fn test_the_matchers_check_their_arguments() -> void {
    if not have_gpu() { return }
    assert_throws(ValueError, || StereoBM(num_disparities: 30))
    assert_throws(ValueError, || StereoBM(num_disparities: 32, block_size: 10))
    assert_throws(ValueError, || StereoBM(num_disparities: 512))
    assert_throws(ValueError, || StereoSGBM(num_disparities: 17))

    let (left, _right) = shifted_pair(48, 96, 8)
    let taller = Mat.zeros(64, 96, CV_8UC1)
    let colour = Mat.zeros(48, 96, CV_8UC3)
    let matcher = StereoBM(num_disparities: 16, block_size: 9)
    assert_throws(ValueError, || matcher.compute(left, taller))
    assert_throws(ValueError, || matcher.compute(colour, colour))
}

fn camera() -> Mat {
    return Mat.from_list(3, 3, CV_32FC1, [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0])
}

#: The property rectification exists for: after it, a point and its partner
#: share a row. Checked over a cloud of points and three relative rotations,
#: because the case that is easy to get right is the one where the cameras are
#: already aligned.
fn test_a_rectified_pair_puts_matches_on_one_row() -> void {
    if not have_gpu() { return }
    let translation = Mat.from_list(3, 1, CV_32FC1, [-0.12, 0.01, -0.004])
    for vector in [[0.0, 0.0, 0.0], [0.01, 0.02, -0.015], [0.05, -0.09, 0.03]] {
        let rotation = rodrigues(Mat.from_list(3, 1, CV_32FC1, vector))
        let (r1, r2, p1, p2, _q, _a, _b) = stereo_rectify(
            camera(),
            [],
            camera(),
            [],
            Size(640, 480),
            rotation,
            translation
        )
        let m1 = matmul(left_block(p1), r1).to_list()
        let m2 = matmul(left_block(p2), r2).to_list()
        let r = rotation.to_list()
        let t = translation.to_list()
        var state = 12345
        for _index in 0..200 {
            state = (state *% 1103515245 +% 12345) % 2147483648
            let a = float((state >> 8) % 2000) / 1000.0 - 1.0
            state = (state *% 1103515245 +% 12345) % 2147483648
            let b = float((state >> 8) % 2000) / 1000.0 - 1.0
            state = (state *% 1103515245 +% 12345) % 2147483648
            let c = float((state >> 8) % 4000) / 1000.0 + 3.0
            let x2 = r[0] * a + r[1] * b + r[2] * c + t[0]
            let y2 = r[3] * a + r[4] * b + r[5] * c + t[1]
            let z2 = r[6] * a + r[7] * b + r[8] * c + t[2]
            let row1 = (m1[3] * a + m1[4] * b + m1[5] * c) / (m1[6] * a + m1[7] * b + m1[8] * c)
            let row2 = (m2[3] * x2 + m2[4] * y2 + m2[5] * z2) /
                (m2[6] * x2 + m2[7] * y2 + m2[8] * z2)
            assert_near(row1, row2, 1e-2)
        }
    }
}

fn left_block(projection: Mat) -> Mat {
    let p = projection.to_list()
    return Mat.from_list(3, 3, CV_32FC1, [p[0], p[1], p[2], p[4], p[5], p[6], p[8], p[9], p[10]])
}

#: Rectifying an already-aligned pair should change nothing, and the
#: reprojection matrix should be the textbook one.
fn test_an_aligned_pair_rectifies_to_itself() -> void {
    if not have_gpu() { return }
    let identity = Mat.from_list(3, 3, CV_32FC1, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    let translation = Mat.from_list(3, 1, CV_32FC1, [-0.12, 0.0, 0.0])
    let (r1, r2, p1, p2, q, roi1, roi2) = stereo_rectify(
        camera(),
        [],
        camera(),
        [],
        Size(640, 480),
        identity,
        translation
    )
    let want = identity.to_list()
    for index in 0..9 {
        assert_near(r1.to_list()[index], want[index], 1e-5)
        assert_near(r2.to_list()[index], want[index], 1e-5)
    }
    #: `P2`'s last column is minus the focal length times the baseline, which
    #: is the whole of what makes a disparity a depth.
    assert_near(p2.to_list()[3], -500.0 * 0.12, 1e-3)
    assert_near(p1.to_list()[3], 0.0, 1e-6)
    #: `Q[3][2]` is one over the baseline, positive. Negative reprojects the
    #: whole scene behind the camera.
    assert_near(q.to_list()[14], 1.0 / 0.12, 1e-3)
    assert_eq(roi1.width, 640)
    assert_eq(roi2.height, 480)
}

#: A disparity, through `Q`, must come back as the depth the geometry says:
#: focal length times baseline over disparity.
fn test_a_disparity_reprojects_to_its_depth() -> void {
    if not have_gpu() { return }
    let identity = Mat.from_list(3, 3, CV_32FC1, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    let translation = Mat.from_list(3, 1, CV_32FC1, [-0.12, 0.0, 0.0])
    let (_r1, _r2, _p1, _p2, q, _a, _b) = stereo_rectify(
        camera(),
        [],
        camera(),
        [],
        Size(64, 48),
        identity,
        translation
    )
    for disparity in [8.0, 12.0, 20.0] {
        var values: list[float] = []
        for _index in 0..64 * 48 { values.push(disparity * 16.0) }
        let map = Mat.from_list(48, 64, CV_32FC1, values)
        let cloud = reproject_image_to_3d(map, q)
        assert_eq(cloud.len(), 64 * 48)
        let wanted = 500.0 * 0.12 / disparity
        assert_near(cloud[24 * 64 + 32].z, wanted, 1e-3)
        #: The principal point projects to zero sideways, since that is where
        #: the optical axis is.
        assert_near(cloud[240 * 0 + 24 * 64 + 32].x, (32.0 - 320.0) * wanted / 500.0, 1e-3)
    }
}

#: A pixel with no disparity must not be handed back as a point in front of the
#: camera; `handle_missing_values` is what says so.
fn test_missing_disparities_are_pushed_away() -> void {
    if not have_gpu() { return }
    let q = Mat.from_list(
        4,
        4,
        CV_32FC1,
        [
            1.0,
            0.0,
            0.0,
            -32.0,
            0.0,
            1.0,
            0.0,
            -24.0,
            0.0,
            0.0,
            0.0,
            500.0,
            0.0,
            0.0,
            8.3333,
            0.0,
        ]
    )
    var values: list[float] = []
    for index in 0..64 * 48 { values.push(if index % 2 == 0 { 160.0 } else { -16.0 }) }
    let map = Mat.from_list(48, 64, CV_32FC1, values)
    let cloud = reproject_image_to_3d(map, q, true)
    assert_true(cloud[0].z < 100.0)
    assert_true(cloud[1].z > 1000.0)
}

fn test_reprojection_checks_its_matrix() -> void {
    if not have_gpu() { return }
    let map = Mat.zeros(4, 4, CV_32FC1)
    assert_throws(ValueError, || reproject_image_to_3d(map, Mat.zeros(3, 3, CV_32FC1)))
}
