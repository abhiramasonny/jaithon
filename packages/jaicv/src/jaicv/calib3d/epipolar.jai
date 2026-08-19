#: Two views of one scene: the fundamental and essential matrices, the pose
#: they encode, and the epipolar lines they draw.
#:
#: These are the functions a program reaches for when it has two pictures and
#: wants to know how the camera moved between them — visual odometry, structure
#: from motion, stereo rectification. Names and argument order follow OpenCV.
#:
#: ```
#: let (fundamental, mask) = find_fundamental_mat(left, right)
#: let (essential, _mask) = find_essential_mat(left, right, camera)
#: let (rotation, translation, inliers) = recover_pose(essential, left, right, camera)
#: ```
#:
#: The fits are the normalised eight-point algorithm inside RANSAC, which is
#: what OpenCV's `FM_RANSAC` is. What is not here: the seven-point and five-point
#: minimal solvers, which need polynomial root-finding this does not carry, so
#: `FM_7POINT` and OpenCV's `RANSAC` five-point essential estimator have no
#: equivalent. Eight correspondences are needed rather than five or seven.
import std.math
from ..core.linalg import inverse, matmul, null_space, svd, transpose
from ..core.mat import Mat
from ..core.types import CV_32F, CV_MAKETYPE, Point2f, Point3f

#: Fit to every point, with no outlier rejection.
pub const FM_8POINT = 2
#: Fit inside RANSAC, keeping the model with the most inliers.
pub const FM_RANSAC = 8
#: Fit by least median of squares.
pub const FM_LMEDS = 4

fn mat3(values: list[float]) -> Mat {
    return Mat.from_list(3, 3, CV_MAKETYPE(CV_32F, 1), values)
}

#: Move the points so their centroid is the origin and their mean distance from
#: it is the square root of two, and say what transform did it.
#:
#: Hartley's conditioning. Without it the eight-point algorithm's design matrix
#: has columns whose magnitudes differ by the square of the image size, and the
#: smallest singular vector of such a matrix is decided by rounding rather than
#: by the data.
fn normalise(points: list[Point2f]) -> tuple[list[Point2f], Mat] {
    let count = points.len()
    if count == 0 { throw ValueError("no points to normalise") }
    var mean_x = 0.0
    var mean_y = 0.0
    for point in points {
        mean_x += float(point.x)
        mean_y += float(point.y)
    }
    mean_x /= float(count)
    mean_y /= float(count)

    var spread = 0.0
    for point in points {
        let dx = float(point.x) - mean_x
        let dy = float(point.y) - mean_y
        spread += math.sqrt(dx * dx + dy * dy)
    }
    spread /= float(count)
    let scale = if spread > 1e-12 { math.sqrt(2.0) / spread } else { 1.0 }

    var moved: list[Point2f] = []
    for point in points {
        moved.push(Point2f((float(point.x) - mean_x) * scale, (float(point.y) - mean_y) * scale))
    }
    let transform = mat3([scale, 0.0, -scale * mean_x, 0.0, scale, -scale * mean_y, 0.0, 0.0, 1.0])
    return (moved, transform)
}

#: The eight-point algorithm on already-normalised points, with the rank-two
#: constraint enforced by dropping the smallest singular value.
fn eight_point(first: list[Point2f], second: list[Point2f]) -> Mat {
    let count = first.len()
    if count < 8 { throw ValueError("the eight-point algorithm needs eight correspondences") }
    var design: list[float] = []
    for index in 0..count {
        let x = float(first[index].x)
        let y = float(first[index].y)
        let u = float(second[index].x)
        let v = float(second[index].y)
        design.push(u * x)
        design.push(u * y)
        design.push(u)
        design.push(v * x)
        design.push(v * y)
        design.push(v)
        design.push(x)
        design.push(y)
        design.push(1.0)
    }
    #: The null space of the design, not its SVD: with exactly eight
    #: correspondences the design is eight rows by nine, so a reduced SVD has
    #: only eight right singular vectors and the ninth -- the answer -- is not
    #: among them. `null_space` goes through `AᵀA`, which is nine by nine
    #: whatever the row count, and hands back the direction sent closest to
    #: zero.
    let a = Mat.from_list(count, 9, CV_MAKETYPE(CV_32F, 1), design)
    let raw = null_space(a)

    #: A fundamental matrix has rank two: the two views share an epipole, and a
    #: fit that does not is one whose epipolar lines do not meet.
    let (u2, s2, v2) = svd(mat3(raw))
    let clipped = [s2[0], 0.0, 0.0, 0.0, s2[1], 0.0, 0.0, 0.0, 0.0]
    return matmul(matmul(u2, mat3(clipped)), v2)
}

fn epipolar_error(matrix: Mat, first: Point2f, second: Point2f) -> float {
    let f = matrix.to_list()
    let x = float(first.x)
    let y = float(first.y)
    let u = float(second.x)
    let v = float(second.y)
    #: The line in the second image, and the line in the first.
    let a = f[0] * x + f[1] * y + f[2]
    let b = f[3] * x + f[4] * y + f[5]
    let c = f[6] * x + f[7] * y + f[8]
    let d = f[0] * u + f[3] * v + f[6]
    let e = f[1] * u + f[4] * v + f[7]
    let residual = u * a + v * b + c
    let scale = a * a + b * b + d * d + e * e
    if scale < 1e-20 { return 1e30 }
    #: Sampson distance: the first-order approximation to the true geometric
    #: distance, which is what OpenCV's RANSAC threshold is measured in.
    return residual * residual / scale
}

fn fit_denormalised(first: list[Point2f], second: list[Point2f], picks: list[int]) -> Mat {
    var left: list[Point2f] = []
    var right: list[Point2f] = []
    for index in picks {
        left.push(first[index])
        right.push(second[index])
    }
    let (moved_left, t_left) = normalise(left)
    let (moved_right, t_right) = normalise(right)
    let raw = eight_point(moved_left, moved_right)
    return matmul(matmul(transpose(t_right), raw), t_left)
}

fn scale_to_unit(matrix: Mat) -> Mat {
    let values = matrix.to_list()
    var largest = 0.0
    for value in values { largest = max(largest, math.fabs(value)) }
    if largest < 1e-20 { return matrix }
    var scaled: list[float] = []
    for value in values { scaled.push(value / largest) }
    return mat3(scaled)
}

#: The fundamental matrix relating two views, and which correspondences it fits.
#:
#: `threshold` is in pixels of Sampson distance and only means anything for
#: `FM_RANSAC` and `FM_LMEDS`.
pub fn find_fundamental_mat(
    first: list[Point2f],
    second: list[Point2f],
    method: int = FM_RANSAC,
    threshold: float = 3.0,
    confidence: float = 0.99
) -> tuple[Mat, list[bool]] {
    let count = first.len()
    if count != second.len() {
        throw ValueError("both views need the same number of points")
    }
    if count < 8 {
        throw ValueError(f"a fundamental matrix needs eight correspondences, got {count}")
    }
    let _unused = confidence

    var all: list[int] = []
    for index in 0..count { all.push(index) }
    if method == FM_8POINT or count == 8 {
        var inliers: list[bool] = []
        for _index in 0..count { inliers.push(true) }
        return (scale_to_unit(fit_denormalised(first, second, all)), inliers)
    }

    #: A deterministic sampler, so the same input gives the same answer every
    #: time; OpenCV draws from its own generator and does not.
    var state = 0x9E3779B97F4A
    let squared = threshold * threshold
    var best = fit_denormalised(first, second, all)
    var best_score = -1.0
    var best_inliers: list[bool] = []
    for _index in 0..count { best_inliers.push(true) }

    for _round in 0..500 {
        var picks: list[int] = []
        while picks.len() < 8 {
            state = state *% 6364136223846793 +% 1442695040888963 & 0xFFFFFFFFFFFF
            let pick = (state >> 8) % count
            if picks.contains(pick) { continue }
            picks.push(pick)
        }
        var candidate: Mat = best
        try {
            candidate = fit_denormalised(first, second, picks)
        } catch _error: ValueError {
            continue
        }

        var inliers: list[bool] = []
        var score = 0.0
        var kept = 0
        for index in 0..count {
            let error = epipolar_error(candidate, first[index], second[index])
            let ok = error <= squared
            inliers.push(ok)
            if ok {
                kept += 1
                score += squared - error
            }
        }
        #: Least median keeps the model whose median error is smallest;
        #: RANSAC keeps the one with the most inliers, tie-broken by how well
        #: they fit.
        let quality = if method == FM_LMEDS {
            var errors: list[float] = []
            for index in 0..count {
                errors.push(epipolar_error(candidate, first[index], second[index]))
            }
            errors.sort()
            -errors[errors.len() // 2]
        } else {
            float(kept) * 1e6 + score
        }
        if quality > best_score {
            best_score = quality
            best = candidate
            best_inliers = inliers
        }
    }

    #: Refit on everything the winning model accepted, which is what makes the
    #: answer as good as its inliers rather than as good as eight of them.
    var kept_indices: list[int] = []
    for index in 0..count {
        if best_inliers[index] { kept_indices.push(index) }
    }
    if kept_indices.len() >= 8 {
        try {
            best = fit_denormalised(first, second, kept_indices)
        } catch _error: ValueError {
            #: The inliers turned out to be degenerate; the sampled fit stands.
        }
    }
    return (scale_to_unit(best), best_inliers)
}

#: The essential matrix relating two calibrated views.
#:
#: Found as the fundamental matrix of the same points and then pulled through
#: the camera: `E = Kᵀ F K`. The two singular values are then equalised, which
#: is the constraint an essential matrix satisfies and a fundamental one does
#: not.
pub fn find_essential_mat(
    first: list[Point2f],
    second: list[Point2f],
    camera_matrix: Mat,
    method: int = FM_RANSAC,
    threshold: float = 1.0,
    confidence: float = 0.99
) -> tuple[Mat, list[bool]] {
    if camera_matrix.rows != 3 or camera_matrix.cols != 3 {
        throw ValueError("a camera matrix is three by three")
    }
    let (fundamental, inliers) = find_fundamental_mat(first, second, method, threshold, confidence)
    let raw = matmul(matmul(transpose(camera_matrix), fundamental), camera_matrix)
    let (u, s, vt) = svd(raw)
    let average = (s[0] + s[1]) / 2.0
    let equal = [average, 0.0, 0.0, 0.0, average, 0.0, 0.0, 0.0, 0.0]
    return (scale_to_unit(matmul(matmul(u, mat3(equal)), vt)), inliers)
}

#: The four camera poses an essential matrix admits, as `(R1, R2, t)`.
#:
#: The pose is one of `(R1, t)`, `(R1, -t)`, `(R2, t)`, `(R2, -t)`; which one
#: is decided by where the points end up, which is what `recover_pose` does.
pub fn decompose_essential_mat(essential: Mat) -> tuple[Mat, Mat, Mat] {
    let (u0, _s, vt0) = svd(essential)
    var u = u0
    var vt = vt0
    #: Both factors must be rotations. An SVD is free to hand back a reflection
    #: instead, and a reflection here produces a "rotation" that turns the scene
    #: inside out and puts every point behind the camera.
    if determinant3(u) < 0.0 { u = negate(u) }
    if determinant3(vt) < 0.0 { vt = negate(vt) }

    let w = mat3([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    let first = matmul(matmul(u, w), vt)
    let second = matmul(matmul(u, transpose(w)), vt)
    let columns = u.to_list()
    let translation = Mat.from_list(
        3,
        1,
        CV_MAKETYPE(CV_32F, 1),
        [columns[2], columns[5], columns[8]]
    )
    return (first, second, translation)
}

fn determinant3(m: Mat) -> float {
    let v = m.to_list()
    return v[0] * (v[4] * v[8] - v[5] * v[7]) - v[1] * (v[3] * v[8] - v[5] * v[6]) +
        v[2] * (v[3] * v[7] - v[4] * v[6])
}

fn negate(m: Mat) -> Mat {
    var out: list[float] = []
    for value in m.to_list() { out.push(-value) }
    return mat3(out)
}

#: Where a correspondence lands in space, given the two projection matrices.
fn triangulate_one(left: list[float], right: list[float], a: Point2f, b: Point2f) -> Point3f {
    var rows: list[float] = []
    let x = float(a.x)
    let y = float(a.y)
    let u = float(b.x)
    let v = float(b.y)
    for column in 0..4 { rows.push(x * left[8 + column] - left[column]) }
    for column in 0..4 { rows.push(y * left[8 + column] - left[4 + column]) }
    for column in 0..4 { rows.push(u * right[8 + column] - right[column]) }
    for column in 0..4 { rows.push(v * right[8 + column] - right[4 + column]) }
    let (_u, _s, vt) = svd(Mat.from_list(4, 4, CV_MAKETYPE(CV_32F, 1), rows))
    let flat = vt.to_list()
    let w = flat[3 * 4 + 3]
    if math.fabs(w) < 1e-20 { return Point3f(0.0, 0.0, 0.0) }
    return Point3f(flat[12] / w, flat[13] / w, flat[14] / w)
}

fn projection_of(rotation: Mat, translation: Mat, camera: Mat) -> list[float] {
    let r = rotation.to_list()
    let t = translation.to_list()
    var rt: list[float] = []
    for row in 0..3 {
        for column in 0..3 { rt.push(r[row * 3 + column]) }
        rt.push(t[row])
    }
    let k = camera.to_list()
    var out: list[float] = []
    for row in 0..3 {
        for column in 0..4 {
            var total = 0.0
            for inner in 0..3 { total += k[row * 3 + inner] * rt[inner * 4 + column] }
            out.push(total)
        }
    }
    return out
}

#: The camera motion between two views, as a rotation, a unit translation, and
#: which correspondences are in front of both cameras.
#:
#: An essential matrix names four poses and only one of them puts the scene in
#: front of both cameras. That test — cheirality — is the whole of what this
#: adds over `decompose_essential_mat`, and it is why the answer needs the
#: points as well as the matrix. The translation is a direction: an essential
#: matrix cannot say how far apart the cameras are, only which way.
pub fn recover_pose(
    essential: Mat,
    first: list[Point2f],
    second: list[Point2f],
    camera_matrix: Mat,
    mask: list[bool]? = null
) -> tuple[Mat, Mat, list[bool]] {
    let count = first.len()
    if count != second.len() {
        throw ValueError("both views need the same number of points")
    }
    let (rotation_a, rotation_b, translation) = decompose_essential_mat(essential)
    let identity = mat3([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    let origin = Mat.from_list(3, 1, CV_MAKETYPE(CV_32F, 1), [0.0, 0.0, 0.0])
    let left = projection_of(identity, origin, camera_matrix)

    var best_rotation = rotation_a
    var best_translation = translation
    var best_front: list[bool] = []
    for _index in 0..count { best_front.push(false) }
    var best_count = -1

    for choice in 0..4 {
        let rotation = if choice < 2 { rotation_a } else { rotation_b }
        let sign = if choice % 2 == 0 { 1.0 } else { -1.0 }
        var moved: list[float] = []
        for value in translation.to_list() { moved.push(value * sign) }
        let shifted = Mat.from_list(3, 1, CV_MAKETYPE(CV_32F, 1), moved)
        let right = projection_of(rotation, shifted, camera_matrix)
        let r = rotation.to_list()

        var front: list[bool] = []
        var kept = 0
        for index in 0..count {
            if mask is not null and not mask[index] {
                front.push(false)
                continue
            }
            let point = triangulate_one(left, right, first[index], second[index])
            #: In front of the first camera is a positive depth; in front of
            #: the second is a positive depth after the motion is applied.
            let depth_second = r[6] * point.x + r[7] * point.y + r[8] * point.z + moved[2]
            let ok = point.z > 0.0 and depth_second > 0.0
            front.push(ok)
            if ok { kept += 1 }
        }
        if kept > best_count {
            best_count = kept
            best_rotation = rotation
            best_translation = shifted
            best_front = front
        }
    }
    return (best_rotation, best_translation, best_front)
}

#: The epipolar lines the given points draw in the other image, each as the
#: `(a, b, c)` of `a x + b y + c = 0` with `a² + b²` one.
#:
#: `which_image` is 1 when the points come from the first image and 2 when they
#: come from the second, exactly as OpenCV numbers them.
pub fn compute_correspond_epilines(
    points: list[Point2f],
    which_image: int,
    fundamental: Mat
) -> list[Point3f] {
    if which_image != 1 and which_image != 2 {
        throw ValueError("which_image is 1 or 2")
    }
    let f = if which_image == 1 { fundamental } else { transpose(fundamental) }
    let m = f.to_list()
    var out: list[Point3f] = []
    for point in points {
        let x = float(point.x)
        let y = float(point.y)
        let a = m[0] * x + m[1] * y + m[2]
        let b = m[3] * x + m[4] * y + m[5]
        let c = m[6] * x + m[7] * y + m[8]
        let length = math.sqrt(a * a + b * b)
        if length < 1e-20 {
            out.push(Point3f(a, b, c))
            continue
        }
        out.push(Point3f(a / length, b / length, c / length))
    }
    return out
}

#: The camera matrix's inverse, applied to image points: what a pinhole camera
#: saw, in the units the essential matrix works in.
pub fn undistort_to_normalised(points: list[Point2f], camera_matrix: Mat) -> list[Point2f] {
    let inv = inverse(camera_matrix).to_list()
    var out: list[Point2f] = []
    for point in points {
        let x = float(point.x)
        let y = float(point.y)
        let w = inv[6] * x + inv[7] * y + inv[8]
        if math.fabs(w) < 1e-20 {
            out.push(Point2f(0.0, 0.0))
            continue
        }
        out.push(
            Point2f((inv[0] * x + inv[1] * y + inv[2]) / w, (inv[3] * x + inv[4] * y + inv[5]) / w)
        )
    }
    return out
}
