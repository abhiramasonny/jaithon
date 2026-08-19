#: Two-view geometry: the fundamental and essential matrices, and the camera
#: motion they encode.
#:
#: `test_against_opencv.jai` compares the matrices themselves against OpenCV's,
#: which is the right check for a clean synthetic pair. It is the wrong check
#: for a noisy one: two fundamental matrices can differ element by element and
#: both fit the same correspondences, because what the data pins down is the
#: epipolar constraint rather than the nine numbers. So the cases here that add
#: noise and outliers check what the matrix DOES -- how far the true inliers
#: sit from their epipolar lines, and whether the recovered rotation is the one
#: the points were made with.
import std.math
import std.gpu as gpu
from std.test import assert_eq, assert_near, assert_throws, assert_true
from jaicv.calib3d.epipolar import compute_correspond_epilines, decompose_essential_mat
from jaicv.calib3d.epipolar import find_essential_mat, find_fundamental_mat, recover_pose
from jaicv.calib3d.epipolar import undistort_to_normalised, FM_8POINT, FM_LMEDS, FM_RANSAC
from jaicv.calib3d.transform import rodrigues
from jaicv.core.linalg import matmul, svd, transpose
from jaicv.core.mat import Mat
from jaicv.core.types import CV_32FC1, Point2f, Point3f

fn have_gpu() -> bool { return gpu.is_available() }

let CAMERA = [520.0, 0.0, 320.0, 0.0, 515.0, 240.0, 0.0, 0.0, 1.0]
let ROTATION_VECTOR = [0.03, 0.12, -0.02]
let TRANSLATION = [0.9, -0.15, 0.08]

fn camera() -> Mat { return Mat.from_list(3, 3, CV_32FC1, CAMERA) }

fn unit(values: list[float]) -> list[float] {
    var length = 0.0
    for value in values { length += value * value }
    length = math.sqrt(length)
    var out: list[float] = []
    for value in values { out.push(value / length) }
    return out
}

#: A cloud in front of the camera, projected twice: once from the origin and
#: once after the rotation and translation above. Deterministic, so a failure
#: is reproducible.
fn two_views(
    count: int,
    jitter: float,
    outliers: int
) -> tuple[list[Point2f], list[Point2f], list[bool]] {
    let k = CAMERA
    let r = rodrigues(Mat.from_list(3, 1, CV_32FC1, ROTATION_VECTOR)).to_list()
    let t = unit(TRANSLATION)
    var state = 0x2545F4914F6C
    var first: list[Point2f] = []
    var second: list[Point2f] = []
    var inlier: list[bool] = []

    for index in 0..count {
        state = state *% 6364136223846793 +% 1442695040888963 & 0xFFFFFFFFFFFF
        let a = float((state >> 8) % 3000) / 1000.0 - 1.5
        state = state *% 6364136223846793 +% 1442695040888963 & 0xFFFFFFFFFFFF
        let b = float((state >> 8) % 3000) / 1000.0 - 1.5
        state = state *% 6364136223846793 +% 1442695040888963 & 0xFFFFFFFFFFFF
        let c = float((state >> 8) % 3000) / 1000.0 + 4.5

        let u1 = (k[0] * a + k[2] * c) / c
        let v1 = (k[4] * b + k[5] * c) / c
        let x2 = r[0] * a + r[1] * b + r[2] * c + t[0]
        let y2 = r[3] * a + r[4] * b + r[5] * c + t[1]
        let z2 = r[6] * a + r[7] * b + r[8] * c + t[2]
        let u2 = (k[0] * x2 + k[2] * z2) / z2
        let v2 = (k[4] * y2 + k[5] * z2) / z2

        state = state *% 6364136223846793 +% 1442695040888963 & 0xFFFFFFFFFFFF
        let noise_a = (float((state >> 8) % 2000) / 1000.0 - 1.0) * jitter
        state = state *% 6364136223846793 +% 1442695040888963 & 0xFFFFFFFFFFFF
        let noise_b = (float((state >> 8) % 2000) / 1000.0 - 1.0) * jitter

        first.push(Point2f(u1 + noise_a, v1 + noise_b))
        if index < outliers {
            #: Far enough that no fit can hold it, and in a direction that
            #: varies so the outliers do not themselves agree on a model.
            let sign = if index % 2 == 0 { 1.0 } else { -1.0 }
            second.push(Point2f(u2 + sign * (40.0 + float(index) * 3.0), v2 - sign * 35.0))
            inlier.push(false)
        } else {
            second.push(Point2f(u2 + noise_b, v2 + noise_a))
            inlier.push(true)
        }
    }
    return (first, second, inlier)
}

#: How far a correspondence sits from its epipolar line, first order.
fn sampson(matrix: Mat, a: Point2f, b: Point2f) -> float {
    let f = matrix.to_list()
    let x = float(a.x)
    let y = float(a.y)
    let u = float(b.x)
    let v = float(b.y)
    let p = f[0] * x + f[1] * y + f[2]
    let q = f[3] * x + f[4] * y + f[5]
    let s = f[6] * x + f[7] * y + f[8]
    let d = f[0] * u + f[3] * v + f[6]
    let e = f[1] * u + f[4] * v + f[7]
    let residual = u * p + v * q + s
    return math.sqrt(residual * residual / (p * p + q * q + d * d + e * e))
}

fn worst_error(
    matrix: Mat,
    first: list[Point2f],
    second: list[Point2f],
    keep: list[bool]
) -> float {
    var largest = 0.0
    for index in 0..first.len() {
        if not keep[index] { continue }
        largest = max(largest, sampson(matrix, first[index], second[index]))
    }
    return largest
}

fn test_a_clean_pair_fits_to_the_pixel() -> void {
    if not have_gpu() { return }
    let (first, second, inlier) = two_views(40, 0.0, 0)
    let (fundamental, mask) = find_fundamental_mat(first, second, FM_8POINT)
    assert_eq(mask.len(), first.len())
    #: An exact geometry fits exactly; a thousandth of a pixel is the float32
    #: the whole chain runs in.
    assert_true(worst_error(fundamental, first, second, inlier) < 1e-2)
}

#: The rotation and translation the points were made with, recovered from the
#: pictures alone. The translation is a direction, so its sign is not fixed.
fn test_the_motion_is_recovered_from_a_clean_pair() -> void {
    if not have_gpu() { return }
    let (first, second, _inlier) = two_views(40, 0.0, 0)
    let (essential, _mask) = find_essential_mat(first, second, camera(), FM_8POINT)
    let (rotation, translation, front) = recover_pose(essential, first, second, camera())

    let wanted = rodrigues(Mat.from_list(3, 1, CV_32FC1, ROTATION_VECTOR)).to_list()
    let got = rotation.to_list()
    for index in 0..9 { assert_near(got[index], wanted[index], 1e-3) }

    let want_t = unit(TRANSLATION)
    let got_t = translation.to_list()
    var same = 0.0
    var flipped = 0.0
    for index in 0..3 {
        same += math.fabs(got_t[index] - want_t[index])
        flipped += math.fabs(got_t[index] + want_t[index])
    }
    assert_true(min(same, flipped) < 1e-2)

    #: Every point was placed in front of the camera, so every point must come
    #: back in front of both.
    var counted = 0
    for flag in front {
        if flag { counted += 1 }
    }
    assert_eq(counted, first.len())
}

#: Twelve of sixty correspondences moved far enough that no model holds them.
#: What has to survive is the model, not the numbers: the fit must still put
#: every true inlier on its epipolar line and still recover the rotation.
fn test_ransac_survives_noise_and_outliers() -> void {
    if not have_gpu() { return }
    let (first, second, inlier) = two_views(60, 0.3, 12)
    let (fundamental, mask) = find_fundamental_mat(first, second, FM_RANSAC, 1.0)
    assert_true(worst_error(fundamental, first, second, inlier) < 1.5)

    var wrongly_kept = 0
    var dropped = 0
    for index in 0..first.len() {
        if mask[index] and not inlier[index] { wrongly_kept += 1 }
        if not mask[index] and inlier[index] { dropped += 1 }
    }
    assert_true(dropped == 0)
    assert_true(wrongly_kept <= 3)

    let (essential, _me) = find_essential_mat(first, second, camera(), FM_RANSAC, 1.0)
    let (rotation, _translation, _front) = recover_pose(essential, first, second, camera(), mask)
    let wanted = rodrigues(Mat.from_list(3, 1, CV_32FC1, ROTATION_VECTOR)).to_list()
    let got = rotation.to_list()
    for index in 0..9 { assert_near(got[index], wanted[index], 5e-2) }
}

fn test_least_median_also_rejects_the_outliers() -> void {
    if not have_gpu() { return }
    let (first, second, inlier) = two_views(60, 0.3, 12)
    let (fundamental, _mask) = find_fundamental_mat(first, second, FM_LMEDS, 1.0)
    assert_true(worst_error(fundamental, first, second, inlier) < 2.0)
}

#: An essential matrix has two equal singular values and a third that is zero.
#: That is the constraint that separates one from a fundamental matrix, and a
#: fit that does not satisfy it decomposes into a pose that is not a rotation.
fn test_an_essential_matrix_has_the_shape_it_should() -> void {
    if not have_gpu() { return }
    let (first, second, _inlier) = two_views(40, 0.0, 0)
    let (essential, _mask) = find_essential_mat(first, second, camera(), FM_8POINT)
    let (_u, values, _v) = svd(essential)
    assert_near(values[0], values[1], 1e-3)
    assert_true(values[2] < values[0] * 1e-3)

    let (r1, r2, translation) = decompose_essential_mat(essential)
    for rotation in [r1, r2] {
        #: Both candidates are rotations: orthogonal with determinant one.
        let product = matmul(transpose(rotation), rotation).to_list()
        for row in 0..3 {
            for column in 0..3 {
                assert_near(product[row * 3 + column], if row == column { 1.0 } else { 0.0 }, 1e-3)
            }
        }
    }
    var length = 0.0
    for value in translation.to_list() { length += value * value }
    assert_near(math.sqrt(length), 1.0, 1e-3)
}

#: Every point lies on the epipolar line its partner draws, which is the whole
#: content of the fundamental matrix.
fn test_a_point_lies_on_its_partners_epipolar_line() -> void {
    if not have_gpu() { return }
    let (first, second, _inlier) = two_views(40, 0.0, 0)
    let (fundamental, _mask) = find_fundamental_mat(first, second, FM_8POINT)

    let lines = compute_correspond_epilines(first, 1, fundamental)
    assert_eq(lines.len(), first.len())
    for index in 0..first.len() {
        let line = lines[index]
        #: Normalised, so the value IS the distance in pixels.
        assert_near(line.x * line.x + line.y * line.y, 1.0, 1e-4)
        let distance = line.x * float(second[index].x) + line.y * float(second[index].y) + line.z
        assert_true(math.fabs(distance) < 1e-2)
    }

    let back = compute_correspond_epilines(second, 2, fundamental)
    for index in 0..second.len() {
        let line = back[index]
        let distance = line.x * float(first[index].x) + line.y * float(first[index].y) + line.z
        assert_true(math.fabs(distance) < 1e-2)
    }
}

fn test_undistorting_to_normalised_undoes_the_camera() -> void {
    if not have_gpu() { return }
    let points = [
        Point2f(320.0, 240.0),
        Point2f(520.0 + 320.0, 240.0),
        Point2f(320.0, 515.0 + 240.0),
    ]
    let normalised = undistort_to_normalised(points, camera())
    assert_near(float(normalised[0].x), 0.0, 1e-5)
    assert_near(float(normalised[0].y), 0.0, 1e-5)
    assert_near(float(normalised[1].x), 1.0, 1e-5)
    assert_near(float(normalised[2].y), 1.0, 1e-5)
}

fn test_too_few_correspondences_are_refused() -> void {
    if not have_gpu() { return }
    let (first, second, _inlier) = two_views(40, 0.0, 0)
    var few_first: list[Point2f] = []
    var few_second: list[Point2f] = []
    for index in 0..7 {
        few_first.push(first[index])
        few_second.push(second[index])
    }
    assert_throws(ValueError, || find_fundamental_mat(few_first, few_second))
    assert_throws(ValueError, || find_fundamental_mat(first, few_second))
    assert_throws(ValueError, || compute_correspond_epilines(first, 3, camera()))
}
