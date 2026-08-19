#: The geometry a camera imposes: rotations, projection, homographies, and
#: undoing lens distortion.
#:
#: A camera here is OpenCV's camera: a three by three matrix of intrinsics and
#: a vector of distortion coefficients `[k1, k2, p1, p2, k3, ...]`, with a pose
#: given as a Rodrigues rotation vector and a translation. Everything in this
#: module takes and returns those, so a calibration computed by OpenCV drops
#: straight in.
#:
#: ```
#: let h = find_homography(source, target)
#: let warped = warp_perspective(image, h, size)
#: let straight = undistort(frame, camera_matrix, dist_coeffs)
#: ```
import std.math
from ..core.linalg import identity, inverse, matmul, null_space, subtract, transpose
from ..core.linalg import levenberg_marquardt, scale, solve_least_squares
from ..core.mat import Mat
from ..core.types import CV_32F, CV_MAKETYPE, Point2f, Point3f, Size

#: Fit using every point, with no attempt to reject outliers.
pub const LMEDS_NONE = 0
#: Fit by random sampling and keeping whichever fit the most points agree with.
pub const RANSAC = 8
#: Fit by minimising the median squared error.
pub const LMEDS = 4

fn matrix_of(values: list[float], rows: int, cols: int) -> Mat {
    return Mat.from_list(rows, cols, CV_MAKETYPE(CV_32F, 1), values)
}

#: Turn a rotation vector into a rotation matrix, or the other way round.
#:
#: A three-entry input is read as an axis whose length is the angle and a three
#: by three matrix comes back; a three by three input goes the other way. This
#: is OpenCV's `Rodrigues`, and the round trip is exact to the precision of the
#: trigonometry.
pub fn rodrigues(source: Mat) -> Mat {
    let values = source.to_list()
    if source.rows * source.cols == 3 {
        let x = values[0]
        let y = values[1]
        let z = values[2]
        let angle = math.sqrt(x * x + y * y + z * z)
        if angle < 1e-12 { return identity(3) }
        let ux = x / angle
        let uy = y / angle
        let uz = z / angle
        let c = math.cos(angle)
        let s = math.sin(angle)
        let t = 1.0 - c
        return matrix_of(
            [
                c + ux * ux * t,
                ux * uy * t - uz * s,
                ux * uz * t + uy * s,
                uy * ux * t + uz * s,
                c + uy * uy * t,
                uy * uz * t - ux * s,
                uz * ux * t - uy * s,
                uz * uy * t + ux * s,
                c + uz * uz * t,
            ],
            3,
            3
        )
    }
    if source.rows != 3 or source.cols != 3 {
        throw ValueError("rodrigues takes a three-vector or a three by three matrix")
    }
    #: Going back: the trace gives the angle and the antisymmetric part gives
    #: the axis, with the half-turn taken apart separately because its
    #: antisymmetric part vanishes.
    let trace = values[0] + values[4] + values[8]
    var cosine = (trace - 1.0) * 0.5
    if cosine > 1.0 { cosine = 1.0 }
    if cosine < -1.0 { cosine = -1.0 }
    let angle = math.acos(cosine)
    if angle < 1e-12 { return matrix_of([0.0, 0.0, 0.0], 3, 1) }
    if math.PI - angle < 1e-7 {
        var axis = [
            math.sqrt(max((values[0] + 1.0) * 0.5, 0.0)),
            math.sqrt(max((values[4] + 1.0) * 0.5, 0.0)),
            math.sqrt(max((values[8] + 1.0) * 0.5, 0.0)),
        ]
        if values[5] < 0.0 { axis[1] = -axis[1] }
        if values[2] < 0.0 { axis[2] = -axis[2] }
        return matrix_of([axis[0] * angle, axis[1] * angle, axis[2] * angle], 3, 1)
    }
    let factor = angle / (2.0 * math.sin(angle))
    return matrix_of(
        [
            (values[7] - values[5]) * factor,
            (values[2] - values[6]) * factor,
            (values[3] - values[1]) * factor,
        ],
        3,
        1
    )
}

fn normalising(points: list[Point2f]) -> tuple[Mat, list[Point2f]] {
    #: Hartley's normalisation: shift the centroid to the origin and scale so
    #: the mean distance from it is root two. Without it the design matrix is
    #: badly conditioned and the fit drifts with the size of the image.
    let count = points.len()
    var cx = 0.0
    var cy = 0.0
    for point in points {
        cx += point.x
        cy += point.y
    }
    cx /= float(count)
    cy /= float(count)
    var spread = 0.0
    for point in points {
        let dx = point.x - cx
        let dy = point.y - cy
        spread += math.sqrt(dx * dx + dy * dy)
    }
    spread /= float(count)
    let factor = if spread > 1e-12 { math.sqrt(2.0) / spread } else { 1.0 }
    var moved: list[Point2f] = []
    for point in points { moved.push(Point2f((point.x - cx) * factor, (point.y - cy) * factor)) }
    let transform = matrix_of(
        [factor, 0.0, -factor * cx, 0.0, factor, -factor * cy, 0.0, 0.0, 1.0],
        3,
        3
    )
    return (transform, moved)
}

fn homography_from(source: list[Point2f], target: list[Point2f]) -> Mat {
    let count = source.len()
    if count < 4 { throw ValueError("a homography needs at least four point pairs") }
    let (ts, ns) = normalising(source)
    let (tt, nt) = normalising(target)

    var rows: list[float] = []
    for index in 0..count {
        let x = ns[index].x
        let y = ns[index].y
        let u = nt[index].x
        let v = nt[index].y
        for value in [-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u] { rows.push(value) }
        for value in [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v] { rows.push(value) }
    }
    let h = null_space(matrix_of(rows, count * 2, 9))
    let normalised = matrix_of(h, 3, 3)
    let denormalised = matmul(matmul(inverse(tt), normalised), ts)
    let flat = denormalised.to_list()
    if math.fabs(flat[8]) < 1e-12 { return denormalised }
    return scale(denormalised, 1.0 / flat[8])
}

fn apply_homography(h: list[float], point: Point2f) -> Point2f {
    let w = h[6] * point.x + h[7] * point.y + h[8]
    if math.fabs(w) < 1e-12 { return Point2f(0.0, 0.0) }
    return Point2f(
        (h[0] * point.x + h[1] * point.y + h[2]) / w,
        (h[3] * point.x + h[4] * point.y + h[5]) / w
    )
}

#: The perspective transform that carries one set of points onto another.
#:
#: With `method` left at zero every pair is used and the fit is the direct
#: linear one. `RANSAC` instead samples four pairs at a time, keeps whichever
#: fit the most pairs agree with to within `ransac_threshold` pixels, and then
#: refits on those, which is what makes it survive wrong matches.
#:
#: Returns the homography and, alongside it, which pairs it counted as inliers.
pub fn find_homography(
    source: list[Point2f],
    target: list[Point2f],
    method: int = 0,
    ransac_threshold: float = 3.0,
    max_iterations: int = 2000
) -> tuple[Mat, list[bool]] {
    let count = source.len()
    if count != target.len() { throw ValueError("both point sets have to be the same length") }
    if count < 4 { throw ValueError("a homography needs at least four point pairs") }

    var inliers: list[bool] = []
    for _index in 0..count { inliers.push(true) }
    if method == 0 or count == 4 { return (homography_from(source, target), inliers) }

    var best_h = homography_from(source, target)
    var best_inliers = inliers
    var best_count = 0
    #: A deterministic sampler, so the same input gives the same answer every
    #: time; OpenCV draws from its own generator and does not.
    var state = 0x9E3779B97F4A
    let squared = ransac_threshold * ransac_threshold
    let rounds = min(max_iterations, 2000)

    for _round in 0..rounds {
        var picks: list[int] = []
        while picks.len() < 4 {
            state = state *% 6364136223846793 +% 1442695040888963 & 0xFFFFFFFFFFFF
            let pick = (state >> 8) % count
            if picks.contains(pick) { continue }
            picks.push(pick)
        }
        var sample_source: list[Point2f] = []
        var sample_target: list[Point2f] = []
        for pick in picks {
            sample_source.push(source[pick])
            sample_target.push(target[pick])
        }
        var candidate: Mat = best_h
        try {
            candidate = homography_from(sample_source, sample_target)
        } catch ValueError {
            continue
        }
        let flat = candidate.to_list()
        var agreed: list[bool] = []
        var agreement = 0
        for index in 0..count {
            let mapped = apply_homography(flat, source[index])
            let dx = mapped.x - target[index].x
            let dy = mapped.y - target[index].y
            let close = dx * dx + dy * dy <= squared
            agreed.push(close)
            if close { agreement += 1 }
        }
        if agreement > best_count {
            best_count = agreement
            best_h = candidate
            best_inliers = agreed
        }
    }

    if best_count >= 4 {
        var kept_source: list[Point2f] = []
        var kept_target: list[Point2f] = []
        for index in 0..count {
            if not best_inliers[index] { continue }
            kept_source.push(source[index])
            kept_target.push(target[index])
        }
        best_h = homography_from(kept_source, kept_target)
    }
    return (best_h, best_inliers)
}

#: The affine transform that best carries one set of points onto another.
#:
#: Returns a two by three matrix. Six unknowns, so three pairs determine it and
#: more are fitted by least squares.
pub fn estimate_affine_2d(source: list[Point2f], target: list[Point2f]) -> Mat {
    let count = source.len()
    if count != target.len() { throw ValueError("both point sets have to be the same length") }
    if count < 3 { throw ValueError("an affine transform needs at least three point pairs") }
    var rows: list[float] = []
    var right: list[float] = []
    for index in 0..count {
        for value in [source[index].x, source[index].y, 1.0, 0.0, 0.0, 0.0] { rows.push(value) }
        right.push(target[index].x)
        for value in [0.0, 0.0, 0.0, source[index].x, source[index].y, 1.0] { rows.push(value) }
        right.push(target[index].y)
    }
    let solution = solve_least_squares(
        matrix_of(rows, count * 2, 6),
        matrix_of(right, count * 2, 1)
    )
    return matrix_of(solution.to_list(), 2, 3)
}

#: The best rotation, scale, and shift — no shear — carrying one set of points
#: onto another.
pub fn estimate_affine_partial_2d(source: list[Point2f], target: list[Point2f]) -> Mat {
    let count = source.len()
    if count != target.len() { throw ValueError("both point sets have to be the same length") }
    if count < 2 { throw ValueError("a similarity needs at least two point pairs") }
    #: Four unknowns rather than six: `[a -b tx; b a ty]`.
    var rows: list[float] = []
    var right: list[float] = []
    for index in 0..count {
        for value in [source[index].x, -source[index].y, 1.0, 0.0] { rows.push(value) }
        right.push(target[index].x)
        for value in [source[index].y, source[index].x, 0.0, 1.0] { rows.push(value) }
        right.push(target[index].y)
    }
    let solution = solve_least_squares(
        matrix_of(rows, count * 2, 4),
        matrix_of(right, count * 2, 1)
    )
    let v = solution.to_list()
    return matrix_of([v[0], -v[1], v[2], v[1], v[0], v[3]], 2, 3)
}

fn distort(x: float, y: float, k: list[float]) -> tuple[float, float] {
    let r2 = x * x + y * y
    let r4 = r2 * r2
    let r6 = r4 * r2
    let k1 = if k.len() > 0 { k[0] } else { 0.0 }
    let k2 = if k.len() > 1 { k[1] } else { 0.0 }
    let p1 = if k.len() > 2 { k[2] } else { 0.0 }
    let p2 = if k.len() > 3 { k[3] } else { 0.0 }
    let k3 = if k.len() > 4 { k[4] } else { 0.0 }
    let radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    let tx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    let ty = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return (x * radial + tx, y * radial + ty)
}

#: Where a set of points in space lands on the image.
#:
#: `rvec` and `tvec` place the camera, `camera_matrix` holds the intrinsics,
#: and `dist_coeffs` bends the result the way the lens does.
pub fn project_points(
    object_points: list[Point3f],
    rvec: Mat,
    tvec: Mat,
    camera_matrix: Mat,
    dist_coeffs: list[float] = []
) -> list[Point2f] {
    let r = rodrigues(rvec).to_list()
    let t = tvec.to_list()
    let k = camera_matrix.to_list()
    var out: list[Point2f] = []
    for point in object_points {
        let cx = r[0] * point.x + r[1] * point.y + r[2] * point.z + t[0]
        let cy = r[3] * point.x + r[4] * point.y + r[5] * point.z + t[1]
        let cz = r[6] * point.x + r[7] * point.y + r[8] * point.z + t[2]
        if math.fabs(cz) < 1e-12 {
            out.push(Point2f(0.0, 0.0))
            continue
        }
        let (dx, dy) = distort(cx / cz, cy / cz, dist_coeffs)
        out.push(Point2f(k[0] * dx + k[1] * dy + k[2], k[4] * dy + k[5]))
    }
    return out
}

#: The maps that `remap` needs to straighten a lens.
#:
#: The maps run the other way round from the correction: for each pixel of the
#: straightened image they say where to read from in the original, which is
#: what makes the resampling a gather rather than a scatter.
pub fn init_undistort_rectify_map(
    camera_matrix: Mat,
    dist_coeffs: list[float],
    new_camera_matrix: Mat,
    size: Size
) -> tuple[Mat, Mat] {
    let k = camera_matrix.to_list()
    let n = new_camera_matrix.to_list()
    let inverse_new = inverse(new_camera_matrix).to_list()
    var map_x: list[float] = []
    var map_y: list[float] = []
    let _unused = n
    for y in 0..size.height {
        for x in 0..size.width {
            let fx = float(x)
            let fy = float(y)
            let ux = inverse_new[0] * fx + inverse_new[1] * fy + inverse_new[2]
            let uy = inverse_new[3] * fx + inverse_new[4] * fy + inverse_new[5]
            let uw = inverse_new[6] * fx + inverse_new[7] * fy + inverse_new[8]
            let nx = if math.fabs(uw) > 1e-12 { ux / uw } else { ux }
            let ny = if math.fabs(uw) > 1e-12 { uy / uw } else { uy }
            let (dx, dy) = distort(nx, ny, dist_coeffs)
            map_x.push(k[0] * dx + k[1] * dy + k[2])
            map_y.push(k[4] * dy + k[5])
        }
    }
    return (matrix_of(map_x, size.height, size.width), matrix_of(map_y, size.height, size.width))
}

#: Take the lens distortion out of a set of points.
#:
#: The distortion has no closed-form inverse, so this iterates: guess the
#: undistorted point, distort it, and correct by the difference. Five rounds
#: is enough for any lens a camera is likely to have.
pub fn undistort_points(
    points: list[Point2f],
    camera_matrix: Mat,
    dist_coeffs: list[float],
    new_camera_matrix: Mat? = null
) -> list[Point2f] {
    let k = camera_matrix.to_list()
    var out: list[Point2f] = []
    for point in points {
        let x0 = (point.x - k[2]) / k[0]
        let y0 = (point.y - k[5]) / k[4]
        var x = x0
        var y = y0
        for _round in 0..20 {
            let (dx, dy) = distort(x, y, dist_coeffs)
            x = x0 - (dx - x)
            y = y0 - (dy - y)
        }
        let target = new_camera_matrix
        if target is null {
            out.push(Point2f(x, y))
            continue
        }
        let n = target.to_list()
        out.push(Point2f(n[0] * x + n[1] * y + n[2], n[4] * y + n[5]))
    }
    return out
}

#: A camera matrix that keeps as much of the straightened image as `alpha` asks
#: for: zero crops to only valid pixels, one keeps every original pixel.
pub fn get_optimal_new_camera_matrix(
    camera_matrix: Mat,
    dist_coeffs: list[float],
    size: Size,
    alpha: float = 0.0
) -> Mat {
    let k = camera_matrix.to_list()
    #: The straightened corners of the image bound what is worth keeping; the
    #: inner rectangle is what stays valid and the outer one is everything.
    var corners: list[Point2f] = []
    for step in 0..9 {
        let t = float(step) / 8.0
        corners.push(Point2f(t * float(size.width - 1), 0.0))
        corners.push(Point2f(t * float(size.width - 1), float(size.height - 1)))
        corners.push(Point2f(0.0, t * float(size.height - 1)))
        corners.push(Point2f(float(size.width - 1), t * float(size.height - 1)))
    }
    let straight = undistort_points(corners, camera_matrix, dist_coeffs)

    var left = -1e30
    var right = 1e30
    var top = -1e30
    var bottom = 1e30
    var outer_left = 1e30
    var outer_right = -1e30
    var outer_top = 1e30
    var outer_bottom = -1e30
    for point in straight {
        if point.x < 0.0 and point.x > left { left = point.x }
        if point.x > 0.0 and point.x < right { right = point.x }
        if point.y < 0.0 and point.y > top { top = point.y }
        if point.y > 0.0 and point.y < bottom { bottom = point.y }
        if point.x < outer_left { outer_left = point.x }
        if point.x > outer_right { outer_right = point.x }
        if point.y < outer_top { outer_top = point.y }
        if point.y > outer_bottom { outer_bottom = point.y }
    }
    let inner_fx = float(size.width) / (right - left)
    let inner_fy = float(size.height) / (bottom - top)
    let outer_fx = float(size.width) / (outer_right - outer_left)
    let outer_fy = float(size.height) / (outer_bottom - outer_top)
    let fx = inner_fx * (1.0 - alpha) + outer_fx * alpha
    let fy = inner_fy * (1.0 - alpha) + outer_fy * alpha
    let cx = -(left * (1.0 - alpha) + outer_left * alpha) * fx
    let cy = -(top * (1.0 - alpha) + outer_top * alpha) * fy
    let _unused = k
    return matrix_of([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0], 3, 3)
}

#: Where two rays from two cameras meet.
#:
#: `first` and `second` are three by four projection matrices; the answer comes
#: back in homogeneous coordinates, four numbers per point.
pub fn triangulate_points(
    first: Mat,
    second: Mat,
    first_points: list[Point2f],
    second_points: list[Point2f]
) -> list[list[float]] {
    let p1 = first.to_list()
    let p2 = second.to_list()
    var out: list[list[float]] = []
    for index in 0..first_points.len() {
        let a = first_points[index]
        let b = second_points[index]
        var rows: list[float] = []
        for column in 0..4 { rows.push(a.x * p1[8 + column] - p1[0 + column]) }
        for column in 0..4 { rows.push(a.y * p1[8 + column] - p1[4 + column]) }
        for column in 0..4 { rows.push(b.x * p2[8 + column] - p2[0 + column]) }
        for column in 0..4 { rows.push(b.y * p2[8 + column] - p2[4 + column]) }
        out.push(null_space(matrix_of(rows, 4, 4)))
    }
    return out
}

#: Refine a homography so it fits the points as well as it can.
#:
#: The direct linear fit minimises an algebraic error, which is not quite the
#: distance in the image; this minimises that distance instead.
pub fn refine_homography(
    h: Mat,
    source: list[Point2f],
    target: list[Point2f],
    iterations: int = 30
) -> Mat {
    let flat = h.to_list()
    var start: list[float] = []
    for index in 0..8 { start.push(flat[index] / flat[8]) }

    let residual = fn(parameters: list[float]) -> list[float] {
        var full = [value for value in parameters]
        full.push(1.0)
        var out: list[float] = []
        for index in 0..source.len() {
            let mapped = apply_homography(full, source[index])
            out.push(mapped.x - target[index].x)
            out.push(mapped.y - target[index].y)
        }
        return out
    }
    let jacobian = fn(parameters: list[float]) -> Mat {
        #: A numeric Jacobian: eight parameters and a cheap residual, so the
        #: difference quotients cost less than deriving and maintaining the
        #: analytic form would.
        let base = residual(parameters)
        var columns: list[list[float]] = []
        for index in 0..parameters.len() {
            let step = max(math.fabs(parameters[index]) * 1e-6, 1e-9)
            var moved = [value for value in parameters]
            moved[index] += step
            let shifted = residual(moved)
            var column: list[float] = []
            for row in 0..base.len() { column.push((shifted[row] - base[row]) / step) }
            columns.push(column)
        }
        var rows: list[float] = []
        for row in 0..base.len() {
            for index in 0..parameters.len() { rows.push(columns[index][row]) }
        }
        return matrix_of(rows, base.len(), parameters.len())
    }

    let fitted = levenberg_marquardt(residual, jacobian, start, iterations)
    var full = [value for value in fitted]
    full.push(1.0)
    return matrix_of(full, 3, 3)
}
