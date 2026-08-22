#: What a contour is: its moments, its area and perimeter, the shapes that
#: bound it, and the tests that describe it.
#:
#: These take a contour as a list of points — what `find_contours` returns —
#: and they follow OpenCV's formulas rather than the textbook ones wherever the
#: two differ, because the differences are visible. A contour's moments come
#: from Green's theorem over the polygon, not from summing pixels; the area is
#: the shoelace sum and so is smaller than the pixel count by roughly half the
#: perimeter; and `bounding_rect` is inclusive of the far edge.
#:
#: ```
#: let (contours, _) = find_contours(mask, RETR_EXTERNAL)
#: for contour in contours {
#:     if contour_area(contour) < 50.0 { continue }
#:     let box = bounding_rect(contour)
#:     rectangle_rect(frame, box, Scalar(0.0, 255.0, 0.0), 1)
#: }
#: ```
import std.math
from std.algo.sort import natural_order, sort_by
from std.core import Printable
from ..core.mat import Mat
from ..core.types import Point, Point2f, Rect, RotatedRect, Scalar, Size2f
from .geometric import solve_dense

#: Compare two shapes by the reciprocals of their log Hu moments.
pub const CONTOURS_MATCH_I1 = 1
#: Compare two shapes by the differences of their log Hu moments.
pub const CONTOURS_MATCH_I2 = 2
#: Compare two shapes by the largest relative difference among them.
pub const CONTOURS_MATCH_I3 = 3

#: Fit a line by least squares.
pub const DIST_L2 = 2
#: Fit a line by the L1 norm, which ignores outliers more readily.
pub const DIST_L1 = 1
#: Fit a line by the Fair loss.
pub const DIST_FAIR = 5
#: Fit a line by the Welsch loss.
pub const DIST_WELSCH = 6
#: Fit a line by the Huber loss.
pub const DIST_HUBER = 7

let EPSILON = 1e-12

#: The raw, central, and normalised moments of a shape.
#:
#: The field names are OpenCV's: `m` for raw, `mu` for central, `nu` for
#: normalised.
pub class Moments: Printable {
    pub let m00: float
    pub let m10: float
    pub let m01: float
    pub let m20: float
    pub let m11: float
    pub let m02: float
    pub let m30: float
    pub let m21: float
    pub let m12: float
    pub let m03: float
    pub let mu20: float
    pub let mu11: float
    pub let mu02: float
    pub let mu30: float
    pub let mu21: float
    pub let mu12: float
    pub let mu03: float
    pub let nu20: float
    pub let nu11: float
    pub let nu02: float
    pub let nu30: float
    pub let nu21: float
    pub let nu12: float
    pub let nu03: float

    pub fn init(self, raw: list[float]) {
        self.m00 = raw[0]
        self.m10 = raw[1]
        self.m01 = raw[2]
        self.m20 = raw[3]
        self.m11 = raw[4]
        self.m02 = raw[5]
        self.m30 = raw[6]
        self.m21 = raw[7]
        self.m12 = raw[8]
        self.m03 = raw[9]

        var inverse = 0.0
        var cx = 0.0
        var cy = 0.0
        if math.fabs(self.m00) > EPSILON {
            inverse = 1.0 / self.m00
            cx = self.m10 * inverse
            cy = self.m01 * inverse
        }
        let mu20 = self.m20 - self.m10 * cx
        var mu11 = self.m11 - self.m10 * cy
        let mu02 = self.m02 - self.m01 * cy
        self.mu20 = mu20
        self.mu11 = mu11
        self.mu02 = mu02
        self.mu30 = self.m30 - cx * (3.0 * mu20 + cx * self.m10)
        #: Doubled here rather than in each line below, exactly as OpenCV does,
        #: because the two third-order terms both want twice it.
        mu11 = mu11 + mu11
        self.mu21 = self.m21 - cx * (mu11 + cx * self.m01) - cy * mu20
        self.mu12 = self.m12 - cy * (mu11 + cy * self.m10) - cx * mu02
        self.mu03 = self.m03 - cy * (3.0 * mu02 + cy * self.m01)

        let root = math.sqrt(math.fabs(inverse))
        let s2 = inverse * inverse
        let s3 = s2 * root
        self.nu20 = self.mu20 * s2
        self.nu11 = self.mu11 * s2
        self.nu02 = self.mu02 * s2
        self.nu30 = self.mu30 * s3
        self.nu21 = self.mu21 * s3
        self.nu12 = self.mu12 * s3
        self.nu03 = self.mu03 * s3
    }

    #: The centre of mass, or `null` when the shape has no area.
    pub fn centroid(self) -> Point2f? {
        if math.fabs(self.m00) <= EPSILON { return null }
        return Point2f(self.m10 / self.m00, self.m01 / self.m00)
    }

    #: Every field, in OpenCV's order.
    pub fn to_list(self) -> list[float] {
        return [
            self.m00,
            self.m10,
            self.m01,
            self.m20,
            self.m11,
            self.m02,
            self.m30,
            self.m21,
            self.m12,
            self.m03,
            self.mu20,
            self.mu11,
            self.mu02,
            self.mu30,
            self.mu21,
            self.mu12,
            self.mu03,
            self.nu20,
            self.nu11,
            self.nu02,
            self.nu30,
            self.nu21,
            self.nu12,
            self.nu03,
        ]
    }

    pub fn to_str(self) -> str { return f"Moments(m00={self.m00})" }

    fn __str__(self) -> str { return self.to_str() }
}

#: The moments of a polygon.
pub fn moments(contour: list[Point]) -> Moments {
    return moments_f([Point2f(float(point.x), float(point.y)) for point in contour])
}

#: The moments of a polygon whose corners are not on whole pixels.
pub fn moments_f(contour: list[Point2f]) -> Moments {
    let count = contour.len()
    if count == 0 { return Moments([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) }

    var a00 = 0.0
    var a10 = 0.0
    var a01 = 0.0
    var a20 = 0.0
    var a11 = 0.0
    var a02 = 0.0
    var a30 = 0.0
    var a21 = 0.0
    var a12 = 0.0
    var a03 = 0.0

    var px = contour[count - 1].x
    var py = contour[count - 1].y
    var px2 = px * px
    var py2 = py * py
    for index in 0..count {
        let x = contour[index].x
        let y = contour[index].y
        let x2 = x * x
        let y2 = y * y
        let cross = px * y - x * py
        let sx = px + x
        let sy = py + y

        a00 += cross
        a10 += cross * sx
        a01 += cross * sy
        a20 += cross * (px * sx + x2)
        a11 += cross * (px * (sy + py) + x * (sy + y))
        a02 += cross * (py * sy + y2)
        a30 += cross * sx * (px2 + x2)
        a03 += cross * sy * (py2 + y2)
        a21 += cross * (px2 * (3.0 * py + y) + 2.0 * px * x * sy + x2 * (py + 3.0 * y))
        a12 += cross * (py2 * (3.0 * px + x) + 2.0 * py * y * sx + y2 * (px + 3.0 * x))

        px = x
        py = y
        px2 = x2
        py2 = y2
    }

    if math.fabs(a00) <= EPSILON {
        return Moments([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    }
    #: The sign of the enclosed area carries through every weight, so a contour
    #: wound the other way still reports a positive `m00`.
    let sign = if a00 > 0.0 { 1.0 } else { -1.0 }
    return Moments(
        [
            a00 * sign / 2.0,
            a10 * sign / 6.0,
            a01 * sign / 6.0,
            a20 * sign / 12.0,
            a11 * sign / 24.0,
            a02 * sign / 12.0,
            a30 * sign / 20.0,
            a21 * sign / 60.0,
            a12 * sign / 60.0,
            a03 * sign / 20.0,
        ]
    )
}

#: The moments of an image, summed over its pixels.
pub fn moments_image(image: Mat, binary: bool = false) -> Moments {
    if image.channels() != 1 { throw ValueError("image moments need a single-channel Mat") }
    image.flush()
    let values = image.values()
    var m00 = 0.0
    var m10 = 0.0
    var m01 = 0.0
    var m20 = 0.0
    var m11 = 0.0
    var m02 = 0.0
    var m30 = 0.0
    var m21 = 0.0
    var m12 = 0.0
    var m03 = 0.0
    for y in 0..image.rows {
        let fy = float(y)
        for x in 0..image.cols {
            var weight = values[y * image.cols + x]
            if weight == 0.0 { continue }
            if binary { weight = 1.0 }
            let fx = float(x)
            m00 += weight
            m10 += weight * fx
            m01 += weight * fy
            m20 += weight * fx * fx
            m11 += weight * fx * fy
            m02 += weight * fy * fy
            m30 += weight * fx * fx * fx
            m21 += weight * fx * fx * fy
            m12 += weight * fx * fy * fy
            m03 += weight * fy * fy * fy
        }
    }
    return Moments([m00, m10, m01, m20, m11, m02, m30, m21, m12, m03])
}

#: The seven moment invariants, which do not change under translation, scale,
#: or rotation.
pub fn hu_moments(source: Moments) -> list[float] {
    let n20 = source.nu20
    let n11 = source.nu11
    let n02 = source.nu02
    let n30 = source.nu30
    let n21 = source.nu21
    let n12 = source.nu12
    let n03 = source.nu03

    let t0 = n30 + n12
    let t1 = n21 + n03
    let q0 = t0 * t0
    let q1 = t1 * t1
    let n4 = 4.0 * n11
    let s = n20 + n02
    let d = n20 - n02

    var out: list[float] = []
    out.push(s)
    out.push(d * d + n4 * n11)
    out.push((n30 - 3.0 * n12) * (n30 - 3.0 * n12) + (3.0 * n21 - n03) * (3.0 * n21 - n03))
    out.push(q0 + q1)

    let sum0 = q0 - 3.0 * q1
    let sum1 = 3.0 * q0 - q1
    out.push((n30 - 3.0 * n12) * t0 * sum0 + (3.0 * n21 - n03) * t1 * sum1)
    out.push(d * (q0 - q1) + n4 * t0 * t1)
    out.push((3.0 * n21 - n03) * t0 * sum0 - (n30 - 3.0 * n12) * t1 * sum1)
    return out
}

#: The area a contour encloses.
#:
#: This is the shoelace sum over the corners, so a shape drawn on pixels comes
#: out smaller than the number of pixels in it — the corners sit at pixel
#: centres and half the boundary falls outside. `oriented` keeps the sign,
#: which says which way the contour is wound.
pub fn contour_area(contour: list[Point], oriented: bool = false) -> float {
    let count = contour.len()
    if count < 2 { return 0.0 }
    var total = 0.0
    var previous = contour[count - 1]
    for index in 0..count {
        let point = contour[index]
        total += float(previous.x) * float(point.y) - float(point.x) * float(previous.y)
        previous = point
    }
    total *= 0.5
    if not oriented { return math.fabs(total) }
    return total
}

#: The length of a curve, closed or open.
pub fn arc_length(curve: list[Point], closed: bool) -> float {
    let count = curve.len()
    if count <= 1 { return 0.0 }
    var total = 0.0
    var previous = if closed { curve[count - 1] } else { curve[0] }
    let first = if closed { 0 } else { 1 }
    for index in first..count {
        let point = curve[index]
        let dx = float(point.x - previous.x)
        let dy = float(point.y - previous.y)
        total += math.sqrt(dx * dx + dy * dy)
        previous = point
    }
    return total
}

#: The smallest upright rectangle containing every point.
#:
#: The far edge is inclusive, as OpenCV's is: a single point gives a one by one
#: rectangle rather than an empty one.
pub fn bounding_rect(points: list[Point]) -> Rect {
    if points.len() == 0 { return Rect(0, 0, 0, 0) }
    var left = points[0].x
    var right = points[0].x
    var top = points[0].y
    var bottom = points[0].y
    for point in points {
        if point.x < left { left = point.x }
        if point.x > right { right = point.x }
        if point.y < top { top = point.y }
        if point.y > bottom { bottom = point.y }
    }
    return Rect(left, top, right - left + 1, bottom - top + 1)
}

fn normalised_angle(degrees: float) -> float {
    var angle = degrees
    while angle >= 0.0 { angle -= 90.0 }
    while angle < -90.0 { angle += 90.0 }
    return angle
}

fn cross_sign(o: Point, a: Point, b: Point) -> int {
    let value = (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
    if value > 0 { return 1 }
    if value < 0 { return -1 }
    return 0
}

fn sorted_unique(points: list[Point]) -> list[Point] {
    #: By x, and by y where the x agree. A hull is often asked for over a whole
    #: contour, which is thousands of points, and inserting each into a list
    #: already in order costs the square of that.
    let out = sort_by(points, |a, b| if a.x != b.x {
        natural_order(a.x, b.x)
    } else {
        natural_order(a.y, b.y)
    })
    var unique: list[Point] = []
    for point in out {
        let count = unique.len()
        if count > 0 and unique[count - 1].x == point.x and unique[count - 1].y == point.y {
            continue
        }
        unique.push(point)
    }
    return unique
}

#: The convex hull of a set of points.
#:
#: `clockwise` is read in OpenCV's frame, where `y` points up, so the default
#: comes out clockwise on screen. The ring is the same one OpenCV returns and
#: turns the same way; which of its vertices comes first is not: OpenCV's falls
#: out of the two chains its Sklansky scan happens to build, and this starts at
#: the leftmost point instead.
pub fn convex_hull(points: list[Point], clockwise: bool = false) -> list[Point] {
    let sorted = sorted_unique(points)
    let count = sorted.len()
    if count < 3 { return sorted }

    var lower: list[Point] = []
    for index in 0..count {
        let point = sorted[index]
        while lower.len() >= 2 and
            cross_sign(lower[lower.len() - 2], lower[lower.len() - 1], point) <= 0 {
            lower.pop()
        }
        lower.push(point)
    }
    var upper: list[Point] = []
    for index in 0..count {
        let point = sorted[count - 1 - index]
        while upper.len() >= 2 and
            cross_sign(upper[upper.len() - 2], upper[upper.len() - 1], point) <= 0 {
            upper.pop()
        }
        upper.push(point)
    }

    #: `lower` walks left to right below the hull and `upper` right to left
    #: above it; dropping each one's last point joins them into a single ring.
    var ring: list[Point] = []
    for index in 0..upper.len() - 1 { ring.push(upper[index]) }
    for index in 0..lower.len() - 1 { ring.push(lower[index]) }
    if not clockwise { return ring }
    var flipped = [ring[0]]
    for index in 0..ring.len() - 1 { flipped.push(ring[ring.len() - 1 - index]) }
    return flipped
}

#: The indices, into `points`, of the hull points.
pub fn convex_hull_indices(points: list[Point], clockwise: bool = false) -> list[int] {
    let hull = convex_hull(points, clockwise)
    var out: list[int] = []
    for corner in hull {
        for index in 0..points.len() {
            if points[index].x == corner.x and points[index].y == corner.y {
                out.push(index)
                break
            }
        }
    }
    return out
}

#: Whether a contour turns the same way at every corner.
pub fn is_contour_convex(contour: list[Point]) -> bool {
    let count = contour.len()
    if count < 3 { return false }
    var sign = 0
    for index in 0..count {
        let a = contour[index]
        let b = contour[(index + 1) % count]
        let c = contour[(index + 2) % count]
        let turn = cross_sign(a, b, c)
        if turn == 0 { continue }
        if sign == 0 {
            sign = turn
        } elif turn != sign {
            return false
        }
    }
    return true
}

fn distance_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float {
    let dx = bx - ax
    let dy = by - ay
    let squared = dx * dx + dy * dy
    if squared <= EPSILON { return math.sqrt((px - ax) * (px - ax) + (py - ay) * (py - ay)) }
    var t = ((px - ax) * dx + (py - ay) * dy) / squared
    if t < 0.0 { t = 0.0 }
    if t > 1.0 { t = 1.0 }
    let cx = ax + t * dx
    let cy = ay + t * dy
    return math.sqrt((px - cx) * (px - cx) + (py - cy) * (py - cy))
}

#: Where a point falls relative to a contour.
#:
#: With `measure_dist` off the answer is `1.0` inside, `-1.0` outside, and
#: `0.0` exactly on the edge. With it on the answer is the distance to the
#: nearest edge, positive inside and negative outside.
pub fn point_polygon_test(
    contour: list[Point],
    point: Point2f,
    measure_dist: bool = false
) -> float {
    let count = contour.len()
    if count == 0 { return -1.0 }
    var inside = false
    var nearest = 1e300
    var on_edge = false

    var previous = contour[count - 1]
    for index in 0..count {
        let current = contour[index]
        let ax = float(previous.x)
        let ay = float(previous.y)
        let bx = float(current.x)
        let by = float(current.y)
        let gap = distance_to_segment(point.x, point.y, ax, ay, bx, by)
        if gap < nearest { nearest = gap }
        if gap <= 1e-9 { on_edge = true }
        if (ay > point.y) != (by > point.y) {
            let cut = ax + (point.y - ay) / (by - ay) * (bx - ax)
            if point.x < cut { inside = not inside }
        }
        previous = current
    }
    if on_edge { return 0.0 }
    if not measure_dist { return if inside { 1.0 } else { -1.0 } }
    return if inside { nearest } else { -nearest }
}

fn douglas_peucker(
    points: list[Point],
    keep: list[bool],
    first: int,
    last: int,
    epsilon: float
) -> void {
    if last <= first + 1 { return }
    let ax = float(points[first].x)
    let ay = float(points[first].y)
    let bx = float(points[last].x)
    let by = float(points[last].y)
    var worst = 0.0
    var worst_at = first
    for index in first + 1..last {
        let gap = distance_to_segment(
            float(points[index].x),
            float(points[index].y),
            ax,
            ay,
            bx,
            by
        )
        if gap > worst {
            worst = gap
            worst_at = index
        }
    }
    if worst <= epsilon { return }
    keep[worst_at] = true
    douglas_peucker(points, keep, first, worst_at, epsilon)
    douglas_peucker(points, keep, worst_at, last, epsilon)
}

#: Simplify a curve, dropping points that sit within `epsilon` of the line
#: through their neighbours.
pub fn approx_poly_dp(curve: list[Point], epsilon: float, closed: bool) -> list[Point] {
    let count = curve.len()
    if count < 3 { return curve }
    if epsilon < 0.0 { throw ValueError("an approximation tolerance cannot be negative") }

    #: A closed curve has no natural ends, so two are made: the point furthest
    #: from the first, and then the point furthest from that one. The curve is
    #: cut there and each half simplified on its own. Taking the second of the
    #: two as the start is what puts the result in OpenCV's rotation.
    var start = 0
    if closed {
        var far = 0
        var furthest = -1.0
        for index in 0..count {
            let dx = float(curve[index].x - curve[0].x)
            let dy = float(curve[index].y - curve[0].y)
            let span = dx * dx + dy * dy
            if span > furthest {
                furthest = span
                far = index
            }
        }
        furthest = -1.0
        for index in 0..count {
            let dx = float(curve[index].x - curve[far].x)
            let dy = float(curve[index].y - curve[far].y)
            let span = dx * dx + dy * dy
            if span > furthest {
                furthest = span
                start = index
            }
        }
    }
    var rotated: list[Point] = []
    for index in 0..count { rotated.push(curve[(start + index) % count]) }
    if closed { rotated.push(rotated[0]) }

    var keep: list[bool] = []
    for _index in 0..rotated.len() { keep.push(false) }
    keep[0] = true
    keep[rotated.len() - 1] = true
    douglas_peucker(rotated, keep, 0, rotated.len() - 1, epsilon)

    var out: list[Point] = []
    let limit = if closed { rotated.len() - 1 } else { rotated.len() }
    for index in 0..limit {
        if keep[index] { out.push(rotated[index]) }
    }
    return out
}

fn hull_f(points: list[Point]) -> list[Point2f] {
    var out: list[Point2f] = []
    for point in convex_hull(points) { out.push(Point2f(float(point.x), float(point.y))) }
    return out
}

#: The smallest rotated rectangle containing every point.
#:
#: Rotating calipers over the convex hull: every minimal box has a side flush
#: with a hull edge, so trying each edge in turn is enough.
pub fn min_area_rect(points: list[Point]) -> RotatedRect {
    let hull = hull_f(points)
    let count = hull.len()
    if count == 0 { return RotatedRect(Point2f(0.0, 0.0), Size2f(0.0, 0.0), 0.0) }
    if count == 1 { return RotatedRect(hull[0], Size2f(0.0, 0.0), 0.0) }

    var best_area = 1e300
    var best_score = -1e300
    var best_angle = 0.0
    var best_min_u = 0.0
    var best_max_u = 0.0
    var best_min_v = 0.0
    var best_max_v = 0.0

    for index in 0..count {
        let a = hull[index]
        let b = hull[(index + 1) % count]
        let dx = b.x - a.x
        let dy = b.y - a.y
        let length = math.sqrt(dx * dx + dy * dy)
        if length <= EPSILON { continue }
        let ux = dx / length
        let uy = dy / length

        var min_u = 1e300
        var max_u = -1e300
        var min_v = 1e300
        var max_v = -1e300
        for corner in hull {
            let u = corner.x * ux + corner.y * uy
            let v = -corner.x * uy + corner.y * ux
            if u < min_u { min_u = u }
            if u > max_u { max_u = u }
            if v < min_v { min_v = v }
            if v > max_v { max_v = v }
        }
        let area = (max_u - min_u) * (max_v - min_v)
        #: A square has two minimal boxes and a nearly square one has two that
        #: tie numerically, so the tie is broken on the angle: the one lying
        #: closest to the horizontal wins, which is the one OpenCV reports.
        let candidate = math.atan2(dy, dx)
        let score = -math.fabs(normalised_angle(candidate * 180.0 / math.PI))
        if area < best_area - 1e-9 or area < best_area + 1e-9 and score > best_score {
            if area < best_area { best_area = area }
            best_score = score
            best_angle = candidate
            best_min_u = min_u
            best_max_u = max_u
            best_min_v = min_v
            best_max_v = max_v
        }
    }

    let ux = math.cos(best_angle)
    let uy = math.sin(best_angle)
    let cu = (best_min_u + best_max_u) * 0.5
    let cv = (best_min_v + best_max_v) * 0.5
    let centre = Point2f(cu * ux - cv * uy, cu * uy + cv * ux)
    var width = best_max_u - best_min_u
    var height = best_max_v - best_min_v
    #: OpenCV measures the angle with `y` pointing up and reports it in
    #: `[-90, 0)`, rotating by a quarter turn and swapping the sides until it
    #: lands there, so one box always comes back the one way.
    var angle = best_angle * 180.0 / math.PI
    while angle >= 0.0 {
        angle -= 90.0
        let swap = width
        width = height
        height = swap
    }
    while angle < -90.0 {
        angle += 90.0
        let swap = width
        width = height
        height = swap
    }
    return RotatedRect(centre, Size2f(width, height), angle)
}

#: The four corners of a rotated rectangle.
pub fn box_points(box: RotatedRect) -> list[Point2f] {
    let radians = box.angle * math.PI / 180.0
    let ux = math.cos(radians) * 0.5
    let uy = math.sin(radians) * 0.5
    let w = box.size.width
    let h = box.size.height
    let px = ux * w
    let py = uy * w
    let qx = -uy * h
    let qy = ux * h
    return [
        Point2f(box.center.x - px - qx, box.center.y - py - qy),
        Point2f(box.center.x + px - qx, box.center.y + py - qy),
        Point2f(box.center.x + px + qx, box.center.y + py + qy),
        Point2f(box.center.x - px + qx, box.center.y - py + qy),
    ]
}

fn circle_through(a: Point2f, b: Point2f, c: Point2f) -> tuple[Point2f, float] {
    let d = 2.0 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y))
    if math.fabs(d) <= EPSILON { return (Point2f(0.0, 0.0), -1.0) }
    let a2 = a.x * a.x + a.y * a.y
    let b2 = b.x * b.x + b.y * b.y
    let c2 = c.x * c.x + c.y * c.y
    let cx = (a2 * (b.y - c.y) + b2 * (c.y - a.y) + c2 * (a.y - b.y)) / d
    let cy = (a2 * (c.x - b.x) + b2 * (a.x - c.x) + c2 * (b.x - a.x)) / d
    let centre = Point2f(cx, cy)
    return (centre, math.sqrt((a.x - cx) * (a.x - cx) + (a.y - cy) * (a.y - cy)))
}

fn covers(centre: Point2f, radius: float, points: list[Point2f]) -> bool {
    for point in points {
        let dx = point.x - centre.x
        let dy = point.y - centre.y
        if math.sqrt(dx * dx + dy * dy) > radius + 1e-6 { return false }
    }
    return true
}

#: The smallest circle containing every point.
pub fn min_enclosing_circle(points: list[Point]) -> tuple[Point2f, float] {
    var all: list[Point2f] = []
    for point in points { all.push(Point2f(float(point.x), float(point.y))) }
    let count = all.len()
    if count == 0 { return (Point2f(0.0, 0.0), 0.0) }
    if count == 1 { return (all[0], 0.0) }

    #: The smallest circle is fixed by two or three of the hull's points, so
    #: only those combinations need trying.
    let hull = hull_f(points)
    let n = hull.len()
    var best_centre = Point2f(0.0, 0.0)
    var best_radius = 1e300
    for i in 0..n {
        for j in i + 1..n {
            let centre = Point2f((hull[i].x + hull[j].x) * 0.5, (hull[i].y + hull[j].y) * 0.5)
            let dx = hull[i].x - hull[j].x
            let dy = hull[i].y - hull[j].y
            let radius = math.sqrt(dx * dx + dy * dy) * 0.5
            if radius < best_radius and covers(centre, radius, all) {
                best_centre = centre
                best_radius = radius
            }
        }
    }
    for i in 0..n {
        for j in i + 1..n {
            for k in j + 1..n {
                let (centre, radius) = circle_through(hull[i], hull[j], hull[k])
                if radius < 0.0 { continue }
                if radius < best_radius and covers(centre, radius, all) {
                    best_centre = centre
                    best_radius = radius
                }
            }
        }
    }
    return (best_centre, best_radius)
}

#: Compare two shapes by their Hu moments.
pub fn match_shapes(a: list[Point], b: list[Point], method: int = CONTOURS_MATCH_I1) -> float {
    let ha = hu_moments(moments(a))
    let hb = hu_moments(moments(b))
    var total = 0.0
    var worst = 0.0
    for index in 0..7 {
        let ma = math.fabs(ha[index])
        let mb = math.fabs(hb[index])
        if ma <= 1e-30 or mb <= 1e-30 { continue }
        let sa = if ha[index] < 0.0 { -1.0 } else { 1.0 }
        let sb = if hb[index] < 0.0 { -1.0 } else { 1.0 }
        let la = sa * math.log10(ma)
        let lb = sb * math.log10(mb)
        if method == CONTOURS_MATCH_I1 {
            total += math.fabs(1.0 / la - 1.0 / lb)
        } elif method == CONTOURS_MATCH_I2 {
            total += math.fabs(la - lb)
        } else {
            let relative = math.fabs((la - lb) / la)
            if relative > worst { worst = relative }
        }
    }
    if method == CONTOURS_MATCH_I3 { return worst }
    return total
}

#: Fit a line to a set of points.
#:
#: Returns `[vx, vy, x0, y0]`: a unit vector along the line and a point on it,
#: which is the layout OpenCV's `fitLine` returns. `dist_type` other than
#: `DIST_L2` runs the same fit again under an iteratively reweighted loss,
#: which is what makes it tolerate outliers.
pub fn fit_line(
    points: list[Point],
    dist_type: int = DIST_L2,
    reps: float = 0.01,
    aeps: float = 0.01
) -> list[float] {
    let count = points.len()
    if count < 2 { throw ValueError("a line needs at least two points") }
    let _unused = reps + aeps
    var weights: list[float] = []
    for _index in 0..count { weights.push(1.0) }

    var result = weighted_fit(points, weights)
    if dist_type == DIST_L2 { return result }
    for _pass in 0..10 {
        for index in 0..count {
            let dx = float(points[index].x) - result[2]
            let dy = float(points[index].y) - result[3]
            let along = dx * result[0] + dy * result[1]
            let away = math.fabs(dx * -result[1] + dy * result[0])
            let _keep = along
            weights[index] = loss_weight(dist_type, away)
        }
        result = weighted_fit(points, weights)
    }
    return result
}

fn loss_weight(dist_type: int, distance: float) -> float {
    let r = distance
    if dist_type == DIST_L1 { return 1.0 / max(r, 1e-6) }
    if dist_type == DIST_FAIR { return 1.0 / (1.0 + r / 1.3998) }
    if dist_type == DIST_WELSCH {
        let t = r / 2.9846
        return math.exp(-t * t)
    }
    if dist_type == DIST_HUBER {
        if r < 1.345 { return 1.0 }
        return 1.345 / r
    }
    return 1.0
}

fn weighted_fit(points: list[Point], weights: list[float]) -> list[float] {
    var total = 0.0
    var sx = 0.0
    var sy = 0.0
    for index in 0..points.len() {
        let w = weights[index]
        total += w
        sx += w * float(points[index].x)
        sy += w * float(points[index].y)
    }
    if total <= EPSILON { throw ValueError("every point was weighted to nothing") }
    let cx = sx / total
    let cy = sy / total
    var sxx = 0.0
    var syy = 0.0
    var sxy = 0.0
    for index in 0..points.len() {
        let w = weights[index]
        let dx = float(points[index].x) - cx
        let dy = float(points[index].y) - cy
        sxx += w * dx * dx
        syy += w * dy * dy
        sxy += w * dx * dy
    }
    #: The direction is the principal eigenvector of the scatter matrix, which
    #: for two by two is a closed form rather than an iteration.
    let difference = sxx - syy
    let angle = 0.5 * math.atan2(2.0 * sxy, difference)
    return [math.cos(angle), math.sin(angle), cx, cy]
}

#: Fit an ellipse to a set of points.
#:
#: Three least-squares fits, which is how OpenCV does it and why the answer
#: agrees with it rather than merely being close: a conic through the points
#: about their centroid, a two by two solve for where its centre really is, and
#: a second conic about that centre. Fitting once about the centroid alone
#: leaves the centre off by enough to move the axes.
#:
#: Returns the ellipse as a `RotatedRect`, with `width` the shorter axis.
pub fn fit_ellipse(points: list[Point]) -> RotatedRect {
    let count = points.len()
    if count < 5 { throw ValueError("an ellipse needs at least five points") }

    var cx = 0.0
    var cy = 0.0
    for point in points {
        cx += float(point.x)
        cy += float(point.y)
    }
    cx /= float(count)
    cy /= float(count)

    var terms: list[list[float]] = []
    for point in points {
        let u = float(point.x) - cx
        let v = float(point.y) - cy
        terms.push([-u * u, -v * v, -u * v, u, v])
    }
    let conic = least_squares(terms, 5)

    #: Where the conic's gradient vanishes is its centre; two equations in two
    #: unknowns, taken from differentiating the general form.
    let offset = solve_dense(
        [[2.0 * conic[0], conic[2]], [conic[2], 2.0 * conic[1]]],
        [conic[3], conic[4]]
    )
    if offset.len() < 2 { throw ValueError("these points do not determine an ellipse") }

    var centred: list[list[float]] = []
    for point in points {
        let u = float(point.x) - cx - offset[0]
        let v = float(point.y) - cy - offset[1]
        centred.push([u * u, v * v, u * v])
    }
    let shape = least_squares(centred, 3)
    let a = shape[0]
    let b = shape[1]
    let c = shape[2]

    let root = math.sqrt((a - b) * (a - b) + c * c)
    let larger = (a + b + root) * 0.5
    let smaller = (a + b - root) * 0.5
    if math.fabs(larger) <= EPSILON or math.fabs(smaller) <= EPSILON {
        throw ValueError("these points do not determine an ellipse")
    }
    #: The larger eigenvalue belongs to the shorter axis, since an axis is one
    #: over the root of its eigenvalue.
    let short_axis = 2.0 / math.sqrt(math.fabs(larger))
    let long_axis = 2.0 / math.sqrt(math.fabs(smaller))
    var angle = 0.5 * math.atan2(c, a - b) * 180.0 / math.PI + 180.0
    while angle >= 180.0 { angle -= 180.0 }
    while angle < 0.0 { angle += 180.0 }
    return RotatedRect(
        Point2f(cx + offset[0], cy + offset[1]),
        Size2f(short_axis, long_axis),
        angle
    )
}

#: Least squares through the normal equations, for a fit whose right-hand side
#: is all ones.
fn least_squares(rows: list[list[float]], width: int) -> list[float] {
    var normal: list[list[float]] = []
    var right: list[float] = []
    for _row in 0..width {
        var row: list[float] = []
        for _column in 0..width { row.push(0.0) }
        normal.push(row)
        right.push(0.0)
    }
    for row in rows {
        for i in 0..width {
            for j in 0..width { normal[i][j] += row[i] * row[j] }
            right[i] += row[i]
        }
    }
    let solution = solve_dense(normal, right)
    if solution.len() < width { throw ValueError("these points do not determine a fit") }
    return solution
}
