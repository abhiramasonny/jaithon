#: A shape's moments, and the measurements taken straight off a contour.
import std.math
from std.core import Printable
from ...core.mat import Mat
from ...core.types import Point, Point2f, Rect

pub let EPSILON = 1e-12

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

#: The edges from `opening` onward summed, which is the whole of what
#: `arc_length` does.
#:
#: Split out so the arm64 tier will take it: `curve.len()` is an OP_INVOKE and
#: one of those anywhere in the body made the tier decline the WHOLE function,
#: which then walked the curve interpreted at 160 ns a point. The count and
#: the starting neighbour come in already worked out; nothing in here calls
#: out.
#:
#: `** 0.5` rather than `math.sqrt` for the same reason — reading `sqrt` off
#: the module is an OP_GET_FIELD, and calling the imported one is a call the
#: tier will not put in a loop; either spelling costs 74 ns a point where this
#: one costs 1.6. Compiled it is a bare `fsqrt`, which is the correctly
#: rounded root `math.sqrt` returns. Until the body is compiled the
#: interpreter routes `**` through `pow`, and this platform's `pow(x, 0.5)`
#: differs from its `sqrt` by one unit in the last place for about one input
#: in eight hundred; a perimeter is a sum of roots and carries more error than
#: that from the sum alone.
fn walk_length(curve: list[Point], count: int, first: int, opening: int) -> float {
    let start = curve[opening]
    var px = start.x
    var py = start.y
    var total = 0.0
    for index in first..count {
        let point = curve[index]
        let dx = float(point.x - px)
        let dy = float(point.y - py)
        total += (dx * dx + dy * dy) ** 0.5
        px = point.x
        py = point.y
    }
    return total
}

#: The length of a curve, closed or open.
pub fn arc_length(curve: list[Point], closed: bool) -> float {
    let count = curve.len()
    if count <= 1 { return 0.0 }
    let opening = if closed { count - 1 } else { 0 }
    let first = if closed { 0 } else { 1 }
    return walk_length(curve, count, first, opening)
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
