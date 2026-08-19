#: Working out where a camera is and what its lens does.
#:
#: `solve_pnp` answers the first from a known object, `calibrate_camera` the
#: second from a chessboard seen several times, and `find_chessboard_corners`
#: finds the board. The parameters they produce are OpenCV's, in OpenCV's
#: layout, so a calibration from either can be used by the other.
#:
#: ```
#: let (found, corners) = find_chessboard_corners(gray, Size(9, 6))
#: let (error, camera, dist, rvecs, tvecs) =
#:     calibrate_camera(object_points, image_points, gray.size())
#: let straight = undistort(frame, camera, dist)
#: ```
import std.math
from ..core.linalg import inverse, levenberg_marquardt, matmul, null_space
from ..core.arith import bitwise_not
from ..core.mat import Mat
from ..core.types import CV_32F, CV_8U, CV_MAKETYPE, INTER_LINEAR, Point, Point2f, Point3f
from ..core.types import Rect, Scalar, Size
from ..imgproc.contours import find_contours, CHAIN_APPROX_SIMPLE, RETR_LIST
from ..imgproc.drawing import circle, line
from ..imgproc.geometric import remap
from ..imgproc.shape import approx_poly_dp, arc_length, contour_area, is_contour_convex
from ..imgproc.shape import min_area_rect
from ..imgproc.morph import erode, get_structuring_element, MORPH_RECT
from ..imgproc.threshold import adaptive_threshold, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY
from ..imgproc.features import corner_sub_pix
from .transform import find_homography, init_undistort_rectify_map, project_points, rodrigues

#: Refine the pose by minimising the reprojection error, starting from a linear
#: guess. This is the default and the only one jaicv implements.
pub const SOLVEPNP_ITERATIVE = 0

fn matrix_of(values: list[float], rows: int, cols: int) -> Mat {
    return Mat.from_list(rows, cols, CV_MAKETYPE(CV_32F, 1), values)
}

#: Straighten a lens.
pub fn undistort(
    image: Mat,
    camera_matrix: Mat,
    dist_coeffs: list[float],
    new_camera_matrix: Mat? = null
) -> Mat {
    let target = new_camera_matrix
    let used = if target is not null { target } else { camera_matrix }
    let (map_x, map_y) = init_undistort_rectify_map(
        camera_matrix,
        dist_coeffs,
        used,
        Size(image.cols, image.rows)
    )
    return remap(image, map_x, map_y, interpolation: INTER_LINEAR)
}

fn pose_residual(
    object_points: list[Point3f],
    image_points: list[Point2f],
    camera_matrix: Mat,
    dist_coeffs: list[float]
) -> fn(list[float]) -> list[float] {
    return fn(parameters: list[float]) -> list[float] {
        let rvec = matrix_of([parameters[0], parameters[1], parameters[2]], 3, 1)
        let tvec = matrix_of([parameters[3], parameters[4], parameters[5]], 3, 1)
        let projected = project_points(object_points, rvec, tvec, camera_matrix, dist_coeffs)
        var out: list[float] = []
        for index in 0..projected.len() {
            out.push(projected[index].x - image_points[index].x)
            out.push(projected[index].y - image_points[index].y)
        }
        return out
    }
}

fn numeric_jacobian(residual: fn(list[float]) -> list[float]) -> fn(list[float]) -> Mat {
    return fn(parameters: list[float]) -> Mat {
        let base = residual(parameters)
        var columns: list[list[float]] = []
        for index in 0..parameters.len() {
            let step = max(math.fabs(parameters[index]) * 1e-6, 1e-8)
            var forward = [value for value in parameters]
            var backward = [value for value in parameters]
            forward[index] += step
            backward[index] -= step
            let up = residual(forward)
            let down = residual(backward)
            var column: list[float] = []
            #: A central difference: twice the work of a one-sided one and two
            #: orders of magnitude closer, which matters because these
            #: parameters differ in scale by thousands.
            for row in 0..base.len() { column.push((up[row] - down[row]) / (2.0 * step)) }
            columns.push(column)
        }
        var rows: list[float] = []
        for row in 0..base.len() {
            for index in 0..parameters.len() { rows.push(columns[index][row]) }
        }
        return matrix_of(rows, base.len(), parameters.len())
    }
}

fn planar(object_points: list[Point3f]) -> bool {
    for point in object_points {
        if math.fabs(point.z) > 1e-9 { return false }
    }
    return true
}

fn pose_from_homography(h: Mat, camera_matrix: Mat) -> tuple[list[float], list[float]] {
    #: A planar object seen by a calibrated camera gives its pose directly: the
    #: homography's first two columns are the first two rotation columns scaled
    #: by one common factor, and the third follows from their cross product.
    let m = matmul(inverse(camera_matrix), h).to_list()
    let c1 = [m[0], m[3], m[6]]
    let c2 = [m[1], m[4], m[7]]
    let t = [m[2], m[5], m[8]]
    let n1 = math.sqrt(c1[0] * c1[0] + c1[1] * c1[1] + c1[2] * c1[2])
    let n2 = math.sqrt(c2[0] * c2[0] + c2[1] * c2[1] + c2[2] * c2[2])
    let factor = 2.0 / max(n1 + n2, 1e-12)
    var r1 = [c1[0] * factor, c1[1] * factor, c1[2] * factor]
    var r2 = [c2[0] * factor, c2[1] * factor, c2[2] * factor]
    var translation = [t[0] * factor, t[1] * factor, t[2] * factor]
    if translation[2] < 0.0 {
        r1 = [-r1[0], -r1[1], -r1[2]]
        r2 = [-r2[0], -r2[1], -r2[2]]
        translation = [-translation[0], -translation[1], -translation[2]]
    }
    let r3 = [
        r1[1] * r2[2] - r1[2] * r2[1],
        r1[2] * r2[0] - r1[0] * r2[2],
        r1[0] * r2[1] - r1[1] * r2[0],
    ]
    let rotation = orthonormal([r1[0], r2[0], r3[0], r1[1], r2[1], r3[1], r1[2], r2[2], r3[2]])
    return (rodrigues(matrix_of(rotation, 3, 3)).to_list(), translation)
}

fn orthonormal(values: list[float]) -> list[float] {
    #: Gram-Schmidt: the three columns come out of a fit and are only nearly
    #: perpendicular, and a rotation that is only nearly a rotation makes the
    #: refinement below wander.
    var c1 = [values[0], values[3], values[6]]
    var c2 = [values[1], values[4], values[7]]
    let n1 = math.sqrt(c1[0] * c1[0] + c1[1] * c1[1] + c1[2] * c1[2])
    if n1 > 1e-12 {
        c1 = [c1[0] / n1, c1[1] / n1, c1[2] / n1]
    }
    let dot = c2[0] * c1[0] + c2[1] * c1[1] + c2[2] * c1[2]
    c2 = [c2[0] - dot * c1[0], c2[1] - dot * c1[1], c2[2] - dot * c1[2]]
    let n2 = math.sqrt(c2[0] * c2[0] + c2[1] * c2[1] + c2[2] * c2[2])
    if n2 > 1e-12 {
        c2 = [c2[0] / n2, c2[1] / n2, c2[2] / n2]
    }
    let c3 = [
        c1[1] * c2[2] - c1[2] * c2[1],
        c1[2] * c2[0] - c1[0] * c2[2],
        c1[0] * c2[1] - c1[1] * c2[0],
    ]
    return [c1[0], c2[0], c3[0], c1[1], c2[1], c3[1], c1[2], c2[2], c3[2]]
}

fn pose_from_dlt(
    object_points: list[Point3f],
    image_points: list[Point2f],
    camera_matrix: Mat
) -> tuple[list[float], list[float]] {
    let k = inverse(camera_matrix).values()
    var rows: list[float] = []
    for index in 0..object_points.len() {
        let point = object_points[index]
        let image = image_points[index]
        let x = k[0] * image.x + k[1] * image.y + k[2]
        let y = k[3] * image.x + k[4] * image.y + k[5]
        for value in [
            point.x,
            point.y,
            point.z,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -x * point.x,
            -x * point.y,
            -x * point.z,
            -x,
        ] {
            rows.push(value)
        }
        for value in [
            0.0,
            0.0,
            0.0,
            0.0,
            point.x,
            point.y,
            point.z,
            1.0,
            -y * point.x,
            -y * point.y,
            -y * point.z,
            -y,
        ] {
            rows.push(value)
        }
    }
    let p = null_space(matrix_of(rows, object_points.len() * 2, 12))
    var rotation = [p[0], p[1], p[2], p[4], p[5], p[6], p[8], p[9], p[10]]
    var translation = [p[3], p[7], p[11]]
    let size = math.sqrt(
        rotation[0] * rotation[0] + rotation[3] * rotation[3] + rotation[6] * rotation[6]
    )
    if size > 1e-12 {
        for index in 0..9 { rotation[index] /= size }
        for index in 0..3 { translation[index] /= size }
    }
    if translation[2] < 0.0 {
        for index in 0..9 { rotation[index] = -rotation[index] }
        for index in 0..3 { translation[index] = -translation[index] }
    }
    return (rodrigues(matrix_of(orthonormal(rotation), 3, 3)).to_list(), translation)
}

#: Where the camera was when it took this picture of a known object.
#:
#: Returns whether it worked and the rotation and translation that carry the
#: object into the camera's frame.
pub fn solve_pnp(
    object_points: list[Point3f],
    image_points: list[Point2f],
    camera_matrix: Mat,
    dist_coeffs: list[float] = [],
    iterations: int = 60
) -> tuple[bool, Mat, Mat] {
    let count = object_points.len()
    if count != image_points.len() {
        throw ValueError("there has to be one image point per object point")
    }
    if count < 4 { throw ValueError("a pose needs at least four points") }

    var start: list[float] = []
    if planar(object_points) {
        var flat: list[Point2f] = []
        for point in object_points { flat.push(Point2f(point.x, point.y)) }
        let (h, _inliers) = find_homography(flat, image_points)
        let (rvec, tvec) = pose_from_homography(h, camera_matrix)
        start = [rvec[0], rvec[1], rvec[2], tvec[0], tvec[1], tvec[2]]
    } else {
        let (rvec, tvec) = pose_from_dlt(object_points, image_points, camera_matrix)
        start = [rvec[0], rvec[1], rvec[2], tvec[0], tvec[1], tvec[2]]
    }

    let residual = pose_residual(object_points, image_points, camera_matrix, dist_coeffs)
    let fitted = levenberg_marquardt(residual, numeric_jacobian(residual), start, iterations)
    return (
        true,
        matrix_of([fitted[0], fitted[1], fitted[2]], 3, 1),
        matrix_of([fitted[3], fitted[4], fitted[5]], 3, 1)
    )
}

fn intrinsics_from(homographies: list[Mat]) -> Mat {
    #: Zhang's constraint: every homography says two things about
    #: `B = K⁻ᵀK⁻¹`, and six views' worth of those pin B down. K comes back out
    #: of B in closed form because B is what a Cholesky factor squares to.
    var rows: list[float] = []
    for h in homographies {
        let m = h.values()
        let h1 = [m[0], m[3], m[6]]
        let h2 = [m[1], m[4], m[7]]
        for value in constraint(h1, h2) { rows.push(value) }
        var difference: list[float] = []
        let first = constraint(h1, h1)
        let second = constraint(h2, h2)
        for index in 0..6 { difference.push(first[index] - second[index]) }
        for value in difference { rows.push(value) }
    }
    let b = null_space(matrix_of(rows, homographies.len() * 2, 6))

    let b11 = b[0]
    let b12 = b[1]
    let b22 = b[2]
    let b13 = b[3]
    let b23 = b[4]
    let b33 = b[5]
    let denominator = b11 * b22 - b12 * b12
    if math.fabs(denominator) < 1e-20 { throw ValueError("these views do not determine a camera") }
    let cy = (b12 * b13 - b11 * b23) / denominator
    let lambda = b33 - (b13 * b13 + cy * (b12 * b13 - b11 * b23)) / b11
    if lambda / b11 <= 0.0 { throw ValueError("these views do not determine a camera") }
    let fx = math.sqrt(lambda / b11)
    let fy = math.sqrt(lambda * b11 / denominator)
    let skew = -b12 * fx * fx * fy / lambda
    let cx = skew * cy / fy - b13 * fx * fx / lambda
    return matrix_of([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0], 3, 3)
}

fn constraint(a: list[float], b: list[float]) -> list[float] {
    return [
        a[0] * b[0],
        a[0] * b[1] + a[1] * b[0],
        a[1] * b[1],
        a[2] * b[0] + a[0] * b[2],
        a[2] * b[1] + a[1] * b[2],
        a[2] * b[2],
    ]
}

#: Work out a camera's intrinsics and distortion from several views of a known
#: planar pattern.
#:
#: Zhang's method for the first guess, then everything — the intrinsics, the
#: distortion, and every view's pose — refined together against the
#: reprojection error. Returns that error along with the parameters.
pub fn calibrate_camera(
    object_points: list[list[Point3f]],
    image_points: list[list[Point2f]],
    image_size: Size,
    iterations: int = 200
) -> tuple[float, Mat, list[float], list[Mat], list[Mat]] {
    let views = object_points.len()
    if views != image_points.len() {
        throw ValueError("there has to be one view of points per view of image")
    }
    if views < 3 { throw ValueError("a calibration needs at least three views") }

    var homographies: list[Mat] = []
    for index in 0..views {
        var flat: list[Point2f] = []
        for point in object_points[index] { flat.push(Point2f(point.x, point.y)) }
        let (h, _inliers) = find_homography(flat, image_points[index])
        homographies.push(h)
    }
    var camera = intrinsics_from(homographies)
    let _unused = image_size

    var parameters: list[float] = []
    let k = camera.values()
    parameters.push(k[0])
    parameters.push(k[4])
    parameters.push(k[2])
    parameters.push(k[5])
    for _index in 0..5 { parameters.push(0.0) }
    for index in 0..views {
        let (rvec, tvec) = pose_from_homography(homographies[index], camera)
        for value in rvec { parameters.push(value) }
        for value in tvec { parameters.push(value) }
    }

    let residual = fn(values: list[float]) -> list[float] {
        let matrix = matrix_of([values[0], 0.0, values[2], 0.0, values[1], values[3], 0.0, 0.0, 1.0], 3, 3)
        let dist = [values[4], values[5], values[6], values[7], values[8]]
        var out: list[float] = []
        for index in 0..views {
            let base = 9 + index * 6
            let rvec = matrix_of([values[base], values[base + 1], values[base + 2]], 3, 1)
            let tvec = matrix_of([values[base + 3], values[base + 4], values[base + 5]], 3, 1)
            let projected = project_points(object_points[index], rvec, tvec, matrix, dist)
            for at in 0..projected.len() {
                out.push(projected[at].x - image_points[index][at].x)
                out.push(projected[at].y - image_points[index][at].y)
            }
        }
        return out
    }

    let fitted = levenberg_marquardt(residual, numeric_jacobian(residual), parameters, iterations)
    camera = matrix_of([fitted[0], 0.0, fitted[2], 0.0, fitted[1], fitted[3], 0.0, 0.0, 1.0], 3, 3)
    let dist = [fitted[4], fitted[5], fitted[6], fitted[7], fitted[8]]

    var rvecs: list[Mat] = []
    var tvecs: list[Mat] = []
    for index in 0..views {
        let base = 9 + index * 6
        rvecs.push(matrix_of([fitted[base], fitted[base + 1], fitted[base + 2]], 3, 1))
        tvecs.push(matrix_of([fitted[base + 3], fitted[base + 4], fitted[base + 5]], 3, 1))
    }

    let final_residual = residual(fitted)
    var total = 0.0
    for value in final_residual { total += value * value }
    let observations = final_residual.len() / 2
    let error = if observations > 0 { math.sqrt(total / float(observations)) } else { 0.0 }
    return (error, camera, dist, rvecs, tvecs)
}

fn quad_centre(corners: list[Point]) -> Point2f {
    var cx = 0.0
    var cy = 0.0
    for corner in corners {
        cx += float(corner.x)
        cy += float(corner.y)
    }
    let count = float(corners.len())
    return Point2f(cx / count, cy / count)
}

#: Find the inner corners of a chessboard.
#:
#: `pattern_size` counts inner corners, not squares, the way OpenCV does: a
#: board of ten by seven squares has nine by six of them. Returns whether the
#: whole board was found and the corners in row-major order.
#:
#: The board is found the way OpenCV finds it. The picture is thresholded, and
#: the squares are taken as quadrilaterals — but a chessboard's squares touch
#: at their corners, so as one region they are a single blob and no
#: quadrilateral comes out of them. Eroding pulls them apart; how much it takes
#: depends on how large the squares are in the picture, so this tries
#: increasing amounts. Both polarities are eroded, because a square touching
#: the edge of the picture merges with whatever is outside the board and only
#: the other colour survives there. Every erosion shrinks a square about its
#: own centre, which is undone before the corners are used.
#:
#: An inner corner is where four squares meet, so it is a corner that at least
#: two of the recovered quadrilaterals agree on; a corner on the board's
#: outside edge belongs to only one and drops out, which is exactly the ones
#: that are not wanted.
pub fn find_chessboard_corners(image: Mat, pattern_size: Size) -> tuple[bool, list[Point2f]] {
    if image.channels() != 1 { throw ValueError("chessboard detection needs a single-channel Mat") }
    let wanted = pattern_size.width * pattern_size.height
    if wanted <= 0 { throw ValueError("a chessboard needs a positive pattern size") }
    image.flush()

    let binary = adaptive_threshold(image, 255.0, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 25, 7.0)
    let inverted = bitwise_not(binary)
    let element = get_structuring_element(MORPH_RECT, Size(3, 3))
    var light = binary
    var dark = inverted
    for rounds in 1..7 {
        light = erode(light, element: element)
        dark = erode(dark, element: element)
        var quads = quads_in(light, float(rounds))
        for quad in quads_in(dark, float(rounds)) { quads.push(quad) }
        #: Four squares meet at an inner corner and two at a corner on the
        #: board's outside edge, so counting to four keeps exactly the ones
        #: wanted. Where a square at the edge of the picture has merged into
        #: what surrounds it and left no quadrilateral behind, three is enough,
        #: which is why both are tried.
        for agreement in [4, 3] {
            let corners = shared_corners(quads, pattern_size, image, agreement)
            if corners.len() == wanted { return (true, corners) }
        }
    }
    return (false, [])
}

class Quad {
    pub var corners: list[Point2f]
    pub var size: float

    pub fn init(self, corners: list[Point2f], size: float) {
        self.corners = corners
        self.size = size
    }
}

fn quads_in(binary: Mat, erosions: float) -> list[Quad] {
    let (contours, _hierarchy) = find_contours(binary, RETR_LIST, CHAIN_APPROX_SIMPLE)
    var out: list[Quad] = []
    for contour in contours {
        let area = contour_area(contour)
        if area < 16.0 { continue }
        let approximated = approx_poly_dp(contour, arc_length(contour, true) * 0.06, true)
        if approximated.len() != 4 { continue }
        if not is_contour_convex(approximated) { continue }
        let box = min_area_rect(approximated)
        let long_side = max(box.size.width, box.size.height)
        let short_side = min(box.size.width, box.size.height)
        if short_side < 3.0 { continue }
        if long_side > short_side * 2.5 { continue }

        let centre = quad_centre(approximated)
        #: Each erosion took a pixel off every side, so each corner moved
        #: towards the centre by that much along both axes; pushing it back out
        #: along the diagonal restores where the square really was.
        var restored: list[Point2f] = []
        for corner in approximated {
            let dx = float(corner.x) - centre.x
            let dy = float(corner.y) - centre.y
            let reach = math.sqrt(dx * dx + dy * dy)
            if reach < 1e-6 {
                restored.push(centre)
                continue
            }
            let factor = (reach + erosions * math.sqrt(2.0)) / reach
            restored.push(Point2f(centre.x + dx * factor, centre.y + dy * factor))
        }
        out.push(Quad(restored, long_side + 2.0 * erosions))
    }
    return out
}

fn shared_corners(
    quads: list[Quad],
    pattern_size: Size,
    image: Mat,
    agreement: int
) -> list[Point2f] {
    if quads.len() == 0 { return [] }
    var spacing = 0.0
    for quad in quads { spacing += quad.size }
    spacing /= float(quads.len())
    if spacing <= 0.0 { return [] }
    let tolerance = spacing * 0.3

    var merged: list[Point2f] = []
    var counts: list[int] = []
    for quad in quads {
        for corner in quad.corners {
            var joined = false
            for index in 0..merged.len() {
                let dx = merged[index].x - corner.x
                let dy = merged[index].y - corner.y
                if math.sqrt(dx * dx + dy * dy) > tolerance { continue }
                let weight = float(counts[index])
                merged[index] = Point2f(
                    (merged[index].x * weight + corner.x) / (weight + 1.0),
                    (merged[index].y * weight + corner.y) / (weight + 1.0)
                )
                counts[index] += 1
                joined = true
                break
            }
            if not joined {
                merged.push(corner)
                counts.push(1)
            }
        }
    }

    var strong: list[Point2f] = []
    for index in 0..merged.len() {
        if counts[index] >= agreement { strong.push(merged[index]) }
    }
    if strong.len() != pattern_size.width * pattern_size.height { return [] }
    let ordered = order_grid(strong, pattern_size)
    if ordered.len() != strong.len() { return [] }
    return corner_sub_pix(image, ordered, Size(5, 5), 30, 0.01)
}

fn order_grid(points: list[Point2f], pattern_size: Size) -> list[Point2f] {
    #: The board's own axes are the two directions its points spread along, so
    #: sorting by the first and then the second puts them in reading order
    #: whichever way round the board is.
    let count = points.len()
    var cx = 0.0
    var cy = 0.0
    for point in points {
        cx += point.x
        cy += point.y
    }
    cx /= float(count)
    cy /= float(count)

    var sxx = 0.0
    var sxy = 0.0
    var syy = 0.0
    for point in points {
        let dx = point.x - cx
        let dy = point.y - cy
        sxx += dx * dx
        sxy += dx * dy
        syy += dy * dy
    }
    let angle = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    let ux = math.cos(angle)
    let uy = math.sin(angle)

    var keyed: list[list[float]] = []
    for index in 0..count {
        let dx = points[index].x - cx
        let dy = points[index].y - cy
        keyed.push([dx * ux + dy * uy, -dx * uy + dy * ux, float(index)])
    }
    var by_row: list[list[float]] = []
    for entry in keyed {
        var at = by_row.len()
        while at > 0 and entry[1] < by_row[at - 1][1] { at -= 1 }
        by_row.insert(at, entry)
    }

    var out: list[Point2f] = []
    var at = 0
    while at < count {
        let width = pattern_size.width
        if at + width > count { return [] }
        var row: list[list[float]] = []
        for index in at..at + width { row.push(by_row[index]) }
        var sorted_row: list[list[float]] = []
        for entry in row {
            var slot = sorted_row.len()
            while slot > 0 and entry[0] < sorted_row[slot - 1][0] { slot -= 1 }
            sorted_row.insert(slot, entry)
        }
        for entry in sorted_row { out.push(points[int(entry[2])]) }
        at += width
    }
    return out
}

fn quad_centre_f(corners: list[Point2f]) -> Point2f {
    var cx = 0.0
    var cy = 0.0
    for corner in corners {
        cx += corner.x
        cy += corner.y
    }
    let count = float(corners.len())
    return Point2f(cx / count, cy / count)
}

#: Draw the corners a chessboard search found, joined in the order they came.
pub fn draw_chessboard_corners(
    image: Mat,
    pattern_size: Size,
    corners: list[Point2f],
    found: bool
) -> Mat {
    if not found or corners.len() == 0 {
        for point in corners {
            circle(
                image,
                Point(int(point.x + 0.5), int(point.y + 0.5)),
                4,
                Scalar(0.0, 0.0, 255.0),
                1
            )
        }
        return image
    }
    let palette = [
        Scalar(0.0, 0.0, 255.0),
        Scalar(0.0, 128.0, 255.0),
        Scalar(0.0, 200.0, 200.0),
        Scalar(0.0, 255.0, 0.0),
        Scalar(200.0, 200.0, 0.0),
        Scalar(255.0, 0.0, 0.0),
        Scalar(255.0, 0.0, 255.0),
    ]
    for row in 0..pattern_size.height {
        let colour = palette[row % palette.len()]
        for column in 0..pattern_size.width {
            let index = row * pattern_size.width + column
            if index >= corners.len() { break }
            let here = Point(int(corners[index].x + 0.5), int(corners[index].y + 0.5))
            circle(image, here, 4, colour, 1)
            if column + 1 < pattern_size.width and index + 1 < corners.len() {
                let next = corners[index + 1]
                line(image, here, Point(int(next.x + 0.5), int(next.y + 0.5)), colour, 1)
            }
        }
    }
    return image
}
