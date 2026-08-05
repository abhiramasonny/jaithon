#: Computational geometry: the point and vector types, the primitive shapes,
#: and the classical planar algorithms over them.
#:
#: ## Floating-point robustness
#:
#: Almost every decision made here reduces to the sign of one cross product:
#: which side of a line a point falls on, whether a hull turns left, whether
#: three vertices are collinear, whether a triangle is an ear. In exact
#: arithmetic that sign is never in doubt. In binary64 it is. For three nearly
#: collinear points the determinant `(b - a) x (c - a)` is a difference of two
#: products that agree in almost every bit, so what survives the subtraction is
#: rounding error and its sign is arbitrary. Worse, it is *inconsistently*
#: arbitrary: two algebraically identical expressions can disagree, and then a
#: hull walk pops a vertex it should have kept, or a point tests inside a
#: polygon from one edge and outside from the next, and the loop never
#: terminates.
#:
#: There are two honest responses. The first is exact arithmetic — Shewchuk's
#: adaptive predicates evaluate the determinant to as many bits as the sign
#: needs and no more. The second is to decide in advance how close is too close
#: to call, and answer "collinear" inside that band. This module takes the
#: second. `orientation` compares the determinant against
#: `EPSILON * (1 + |x1*y2| + |y1*x2|)` — an absolute floor of `EPSILON` plus a
#: term that grows with the magnitude of the operands — and every other
#: predicate is expressed through `orientation` so that they cannot contradict
#: one another.
#:
#: What that buys, and what it costs:
#:
#:   - Answers are consistent. The same three points always get the same
#:     verdict, which is what keeps the loops here terminating.
#:   - Answers are not exact. A polygon whose vertices differ only in the
#:     fifteenth digit is reported as degenerate. That is usually the truth
#:     about the data rather than a defect in the test.
#:   - Accuracy is best for coordinates between about 1 and 1e6. Near 1e15 a
#:     single ulp already exceeds `EPSILON` and no tolerance can rescue the
#:     predicate; translate to a local origin before calling.
#:   - Every routine that can subtract a local origin before multiplying does
#:     so — `orientation`, the shoelace sum, the circumcentre, the closest-pair
#:     strip — because that removes the leading cancellation instead of
#:     tolerating it.
#:   - Never compare coordinates with `==`. `Point2.approx_eq` is the test.
import std.math
from std.core import Printable
from std.algo.sort import sort_by

#: The tolerance that separates "zero" from "small" in every predicate here.
#:
#: Used as an absolute floor and scaled by the magnitude of the operands where
#: the quantity being tested is a product. See the module doc comment.
pub const EPSILON: float = 1e-9

#: A point in the plane, doubling as a two-dimensional vector.
#:
#: Immutable: every operation returns a new point, so a `Point2` can be shared
#: without copying and stored in a dict without the key changing underfoot.
#: The vector operations (`dot`, `cross`, `length`, `normalize`) read the point
#: as the vector from the origin to it, which is what makes `b - a` a
#: displacement and `(b - a).length()` a distance.
pub class Point2: Printable {
    #: The horizontal coordinate.
    pub let x: float

    #: The vertical coordinate.
    pub let y: float

    #: Create the point `(x, y)`.
    pub fn init(self, x: float, y: float) {
        self.x = x
        self.y = y
    }

    #: Create the point `radius` away from the origin in direction `angle`.
    #:
    #: ```
    #: Point2.from_polar(2.0, 0.0).x    # 2.0
    #: ```
    pub static fn from_polar(radius: float, angle: float) -> Point2 {
        return Point2(radius * math.cos(angle), radius * math.sin(angle))
    }

    #: Return the scalar product of `self` and `other`.
    #:
    #: Zero exactly when the two are perpendicular, positive when they point
    #: within a quarter turn of each other.
    #:
    #: ```
    #: Point2(1.0, 0.0).dot(Point2(0.0, 1.0))    # 0.0
    #: ```
    pub fn dot(self, other: Point2) -> float {
        return self.x * other.x + self.y * other.y
    }

    #: Return the z component of the three-dimensional cross product.
    #:
    #: In the plane the cross product is a single number: the signed area of
    #: the parallelogram the two vectors span. It is positive when `other` lies
    #: counter-clockwise of `self`, negative when clockwise, zero when the two
    #: are parallel. Every orientation test in this module is this number.
    #:
    #: ```
    #: Point2(1.0, 0.0).cross(Point2(0.0, 1.0))    # 1.0
    #: ```
    pub fn cross(self, other: Point2) -> float {
        return self.x * other.y - self.y * other.x
    }

    #: Return the distance from the origin.
    #:
    #: Computed with `hypot`, which scales before squaring and so survives
    #: coordinates whose squares would overflow or underflow.
    #:
    #: ```
    #: Point2(3.0, 4.0).length()    # 5.0
    #: ```
    pub fn length(self) -> float { return math.hypot(self.x, self.y) }

    #: Return the squared distance from the origin.
    #:
    #: Prefer this wherever only a comparison of lengths is needed: it saves a
    #: square root and, being a sum of products rather than a root of one, it
    #: is the more accurate comparison.
    pub fn length_squared(self) -> float { return self.x * self.x + self.y * self.y }

    #: Return a vector of length one pointing the same way as `self`.
    #:
    #: Raises `ValueError` for the zero vector, which has no direction.
    #:
    #: ```
    #: Point2(3.0, 4.0).normalize().x    # 0.6
    #: ```
    pub fn normalize(self) -> Point2 {
        let magnitude = self.length()
        if magnitude == 0.0 { throw ValueError("cannot normalize the zero vector") }
        return Point2(self.x / magnitude, self.y / magnitude)
    }

    #: Return `self` rotated `angle` radians counter-clockwise about the origin.
    #:
    #: ```
    #: let turned = Point2(1.0, 0.0).rotate(math.PI / 2.0)
    #: math.approx_eq(turned.y, 1.0)    # true
    #: ```
    pub fn rotate(self, angle: float) -> Point2 {
        let cosine = math.cos(angle)
        let sine = math.sin(angle)
        return Point2(self.x * cosine - self.y * sine, self.x * sine + self.y * cosine)
    }

    #: Return `self` rotated `angle` radians counter-clockwise about `center`.
    pub fn rotate_about(self, center: Point2, angle: float) -> Point2 {
        return center + (self - center).rotate(angle)
    }

    #: Return the direction of `self` as an angle in `(-PI, PI]`.
    #:
    #: Measured counter-clockwise from the positive x axis. The origin reports
    #: 0.0, which is a convention rather than an answer.
    #:
    #: ```
    #: Point2(0.0, 1.0).angle() == math.PI / 2.0    # true
    #: ```
    pub fn angle(self) -> float { return math.atan2(self.y, self.x) }

    #: Return the signed angle from `self` to `other`, in `(-PI, PI]`.
    #:
    #: Positive when `other` is counter-clockwise of `self`. Taken as
    #: `atan2(cross, dot)` rather than `acos(dot / (|a| * |b|))`: the arccosine
    #: form loses all its precision for nearly parallel vectors, where the
    #: cosine is flat, and cannot tell the two turn directions apart.
    #:
    #: Raises `ValueError` when either vector is zero.
    #:
    #: ```
    #: Point2(1.0, 0.0).angle_to(Point2(0.0, 1.0)) == math.PI / 2.0    # true
    #: ```
    pub fn angle_to(self, other: Point2) -> float {
        if self.length_squared() == 0.0 or other.length_squared() == 0.0 {
            throw ValueError("angle_to is undefined when either vector is zero")
        }
        return math.atan2(self.cross(other), self.dot(other))
    }

    #: Return the distance between `self` and `other`.
    #:
    #: ```
    #: Point2(0.0, 0.0).distance(Point2(3.0, 4.0))    # 5.0
    #: ```
    pub fn distance(self, other: Point2) -> float {
        return math.hypot(self.x - other.x, self.y - other.y)
    }

    #: Return the squared distance between `self` and `other`.
    pub fn distance_squared(self, other: Point2) -> float {
        let dx = self.x - other.x
        let dy = self.y - other.y
        return dx * dx + dy * dy
    }

    #: Return the point `t` of the way from `self` to `other`.
    #:
    #: Exact at both ends, so `lerp(other, 1.0)` is `other` and not a rounding
    #: short of it. Values outside `[0, 1]` extrapolate.
    #:
    #: ```
    #: Point2(0.0, 0.0).lerp(Point2(4.0, 8.0), 0.25).y    # 2.0
    #: ```
    pub fn lerp(self, other: Point2, t: float) -> Point2 {
        return Point2(math.lerp(self.x, other.x, t), math.lerp(self.y, other.y, t))
    }

    #: Return `self` rotated a quarter turn counter-clockwise: `(-y, x)`.
    #:
    #: This is the left normal of the vector, the one that points into the
    #: half-plane a counter-clockwise polygon encloses.
    pub fn perpendicular(self) -> Point2 { return Point2(-self.y, self.x) }

    #: Report whether `self` and `other` agree to within `tolerance`.
    #:
    #: The comparison is absolute on each coordinate, so choose a `tolerance`
    #: on the scale of the data. This, never `==`, is how two computed points
    #: are compared.
    #:
    #: ```
    #: Point2(1.0, 1.0).approx_eq(Point2(1.0, 1.0 + 1e-15))    # true
    #: ```
    pub fn approx_eq(self, other: Point2, tolerance: float = EPSILON) -> bool {
        return math.fabs(self.x - other.x) <= tolerance and math.fabs(self.y - other.y) <= tolerance
    }

    #: Return the coordinates as a plain tuple, ready for destructuring.
    pub fn to_tuple(self) -> tuple[float, float] { return (self.x, self.y) }

    #: Return the point rendered as `Point2(x, y)`.
    pub fn to_str(self) -> str { return f"Point2({self.x}, {self.y})" }

    fn __add__(self, other: Point2) -> Point2 {
        return Point2(self.x + other.x, self.y + other.y)
    }

    fn __sub__(self, other: Point2) -> Point2 {
        return Point2(self.x - other.x, self.y - other.y)
    }

    fn __mul__(self, factor: float) -> Point2 {
        return Point2(self.x * factor, self.y * factor)
    }

    fn __div__(self, divisor: float) -> Point2 {
        if divisor == 0.0 { throw DivisionByZeroError("cannot divide a Point2 by zero") }
        return Point2(self.x / divisor, self.y / divisor)
    }

    fn __neg__(self) -> Point2 { return Point2(-self.x, -self.y) }

    fn __eq__(self, other: any) -> bool {
        if not isinstance(other, Point2) { return false }
        return self.x == other.x and self.y == other.y
    }

    fn __hash__(self) -> int {
        let x_hash = hash(self.x)
        let y_hash = hash(self.y)
        # Rotate before mixing so that (a, b) and (b, a) do not collide.
        return x_hash ^ (y_hash << 13 | y_hash >> 51 & 0x1FFF)
    }

    fn __str__(self) -> str { return self.to_str() }
}

#: A point in space, doubling as a three-dimensional vector.
#:
#: The same value semantics as `Point2`. The cross product is a vector here
#: rather than a scalar, and it is the whole reason the type exists: surface
#: normals, torques and the plane through three points all come from it.
pub class Vec3: Printable {
    #: The x coordinate.
    pub let x: float

    #: The y coordinate.
    pub let y: float

    #: The z coordinate.
    pub let z: float

    #: Create the vector `(x, y, z)`.
    pub fn init(self, x: float, y: float, z: float) {
        self.x = x
        self.y = y
        self.z = z
    }

    #: Return the scalar product of `self` and `other`.
    pub fn dot(self, other: Vec3) -> float {
        return self.x * other.x + self.y * other.y + self.z * other.z
    }

    #: Return the vector perpendicular to both `self` and `other`.
    #:
    #: Its length is the area of the parallelogram they span, and its direction
    #: follows the right-hand rule, so the result is zero exactly when the two
    #: are parallel.
    #:
    #: ```
    #: Vec3(1.0, 0.0, 0.0).cross(Vec3(0.0, 1.0, 0.0)).z    # 1.0
    #: ```
    pub fn cross(self, other: Vec3) -> Vec3 {
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    }

    #: Return the distance from the origin.
    #:
    #: ```
    #: math.approx_eq(Vec3(2.0, 3.0, 6.0).length(), 7.0)    # true
    #: ```
    pub fn length(self) -> float {
        return math.hypot(math.hypot(self.x, self.y), self.z)
    }

    #: Return the squared distance from the origin.
    pub fn length_squared(self) -> float {
        return self.x * self.x + self.y * self.y + self.z * self.z
    }

    #: Return a vector of length one pointing the same way as `self`.
    #:
    #: Raises `ValueError` for the zero vector.
    pub fn normalize(self) -> Vec3 {
        let magnitude = self.length()
        if magnitude == 0.0 { throw ValueError("cannot normalize the zero vector") }
        return Vec3(self.x / magnitude, self.y / magnitude, self.z / magnitude)
    }

    #: Return the distance between `self` and `other`.
    pub fn distance(self, other: Vec3) -> float { return (self - other).length() }

    #: Return the squared distance between `self` and `other`.
    pub fn distance_squared(self, other: Vec3) -> float {
        return (self - other).length_squared()
    }

    #: Return the point `t` of the way from `self` to `other`, exact at the ends.
    pub fn lerp(self, other: Vec3, t: float) -> Vec3 {
        return Vec3(
            math.lerp(self.x, other.x, t),
            math.lerp(self.y, other.y, t),
            math.lerp(self.z, other.z, t)
        )
    }

    #: Return the unsigned angle between `self` and `other`, in `[0, PI]`.
    #:
    #: Taken as `atan2(|cross|, dot)`. The textbook `acos(dot / (|a| * |b|))`
    #: is accurate only near a right angle: at small angles the cosine is flat,
    #: so the arccosine amplifies the rounding error without bound, and a dot
    #: product that rounds just past one makes it raise outright.
    #:
    #: Raises `ValueError` when either vector is zero.
    #:
    #: ```
    #: let right = Vec3(1.0, 0.0, 0.0).angle_to(Vec3(0.0, 1.0, 0.0))
    #: math.approx_eq(right, math.PI / 2.0)    # true
    #: ```
    pub fn angle_to(self, other: Vec3) -> float {
        if self.length_squared() == 0.0 or other.length_squared() == 0.0 {
            throw ValueError("angle_to is undefined when either vector is zero")
        }
        return math.atan2(self.cross(other).length(), self.dot(other))
    }

    #: Report whether `self` and `other` agree to within `tolerance`.
    pub fn approx_eq(self, other: Vec3, tolerance: float = EPSILON) -> bool {
        return math.fabs(self.x - other.x) <= tolerance and
            math.fabs(self.y - other.y) <= tolerance and
            math.fabs(self.z - other.z) <= tolerance
    }

    #: Return the coordinates as a plain tuple, ready for destructuring.
    pub fn to_tuple(self) -> tuple[float, float, float] { return (self.x, self.y, self.z) }

    #: Return the vector rendered as `Vec3(x, y, z)`.
    pub fn to_str(self) -> str { return f"Vec3({self.x}, {self.y}, {self.z})" }

    fn __add__(self, other: Vec3) -> Vec3 {
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    fn __sub__(self, other: Vec3) -> Vec3 {
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    }

    fn __mul__(self, factor: float) -> Vec3 {
        return Vec3(self.x * factor, self.y * factor, self.z * factor)
    }

    fn __div__(self, divisor: float) -> Vec3 {
        if divisor == 0.0 { throw DivisionByZeroError("cannot divide a Vec3 by zero") }
        return Vec3(self.x / divisor, self.y / divisor, self.z / divisor)
    }

    fn __neg__(self) -> Vec3 { return Vec3(-self.x, -self.y, -self.z) }

    fn __eq__(self, other: any) -> bool {
        if not isinstance(other, Vec3) { return false }
        return self.x == other.x and self.y == other.y and self.z == other.z
    }

    fn __hash__(self) -> int {
        let x_hash = hash(self.x)
        let y_hash = hash(self.y)
        let z_hash = hash(self.z)
        return x_hash ^ (y_hash << 13 | y_hash >> 51 & 0x1FFF) ^ z_hash << 27
    }

    fn __str__(self) -> str { return self.to_str() }
}

#: An infinite line, stored as a point on it and a direction along it.
#:
#: The direction is kept unnormalized so that `at(0.0)` and `at(1.0)` are the
#: two points the line was built from.
pub class Line: Printable {
    #: A point the line passes through.
    pub let point: Point2

    #: The direction along the line; never zero.
    pub let direction: Point2

    #: Create the line through `a` and `b`.
    #:
    #: Raises `ValueError` when the two points coincide, which fixes no line.
    pub fn init(self, a: Point2, b: Point2) {
        if a.approx_eq(b) {
            throw ValueError(f"a line needs two distinct points, both are {a}")
        }
        self.point = a
        self.direction = b - a
    }

    #: Create the line through `point` running along `direction`.
    #:
    #: Raises `ValueError` for a zero direction.
    pub static fn from_direction(point: Point2, direction: Point2) -> Line {
        if direction.length_squared() == 0.0 {
            throw ValueError("a line needs a non-zero direction")
        }
        return Line(point, point + direction)
    }

    #: Return the point at parameter `t`: `point + direction * t`.
    pub fn at(self, t: float) -> Point2 { return self.point + self.direction * t }

    #: Return 1 when `p` is left of the line, -1 when right, 0 when on it.
    #:
    #: "Left" is the side the direction's counter-clockwise normal points to.
    pub fn side(self, p: Point2) -> int {
        return orientation(self.point, self.point + self.direction, p)
    }

    #: Report whether `p` lies on the line, within the module tolerance.
    pub fn contains(self, p: Point2) -> bool { return self.side(p) == 0 }

    #: Return the point on the line closest to `p`.
    #:
    #: ```
    #: Line(Point2(0.0, 0.0), Point2(1.0, 0.0)).project(Point2(3.0, 5.0)).y    # 0.0
    #: ```
    pub fn project(self, p: Point2) -> Point2 {
        let t = (p - self.point).dot(self.direction) / self.direction.length_squared()
        return self.at(t)
    }

    #: Return the perpendicular distance from `p` to the line.
    #:
    #: ```
    #: Line(Point2(0.0, 0.0), Point2(1.0, 0.0)).distance_to(Point2(3.0, 5.0))    # 5.0
    #: ```
    pub fn distance_to(self, p: Point2) -> float {
        return math.fabs(self.direction.cross(p - self.point)) / self.direction.length()
    }

    #: Report whether the two lines have parallel directions.
    #:
    #: Coincident lines are parallel by this test: they never cross in exactly
    #: one point either.
    pub fn is_parallel(self, other: Line) -> bool {
        return _cross_sign(self.direction, other.direction) == 0
    }

    #: Return the single point the two lines share, or null.
    #:
    #: Null covers both parallel and coincident lines — neither has one
    #: intersection point.
    #:
    #: ```
    #: let horizontal = Line(Point2(0.0, 0.0), Point2(1.0, 0.0))
    #: let vertical = Line(Point2(2.0, -1.0), Point2(2.0, 1.0))
    #: horizontal.intersection(vertical).x    # 2.0
    #: ```
    pub fn intersection(self, other: Line) -> Point2? {
        if self.is_parallel(other) { return null }
        let denominator = self.direction.cross(other.direction)
        let t = (other.point - self.point).cross(other.direction) / denominator
        return self.at(t)
    }

    #: Return the line rendered as `Line(point, direction)`.
    pub fn to_str(self) -> str { return f"Line({self.point}, {self.direction})" }

    fn __str__(self) -> str { return self.to_str() }
}

#: A straight segment between two endpoints.
#:
#: Degenerate segments — both endpoints equal — are allowed and behave as the
#: single point they are.
pub class Segment: Printable {
    #: The first endpoint.
    pub let a: Point2

    #: The second endpoint.
    pub let b: Point2

    #: Create the segment from `a` to `b`.
    pub fn init(self, a: Point2, b: Point2) {
        self.a = a
        self.b = b
    }

    #: Return the length of the segment.
    pub fn length(self) -> float { return self.a.distance(self.b) }

    #: Return the point halfway along the segment.
    pub fn midpoint(self) -> Point2 { return self.a.lerp(self.b, 0.5) }

    #: Return the displacement from `a` to `b`.
    pub fn direction(self) -> Point2 { return self.b - self.a }

    #: Return the point on the segment closest to `p`.
    #:
    #: The projection onto the supporting line, clamped to the endpoints.
    #:
    #: ```
    #: Segment(Point2(0.0, 0.0), Point2(2.0, 0.0)).closest_point(Point2(5.0, 1.0)).x    # 2.0
    #: ```
    pub fn closest_point(self, p: Point2) -> Point2 {
        let direction = self.b - self.a
        let squared = direction.length_squared()
        if squared == 0.0 { return self.a }
        let t = math.clamp((p - self.a).dot(direction) / squared, 0.0, 1.0)
        return self.a + direction * t
    }

    #: Return the distance from `p` to the nearest point of the segment.
    pub fn distance_to(self, p: Point2) -> float { return p.distance(self.closest_point(p)) }

    #: Report whether `p` lies on the segment, within the module tolerance.
    pub fn contains(self, p: Point2) -> bool {
        if orientation(self.a, self.b, p) != 0 { return false }
        return _within_bounds(self.a, self.b, p)
    }

    #: Report whether the two segments share at least one point.
    #:
    #: The complete test, collinear overlaps and touching endpoints included.
    #: Four orientations decide the general case: the segments cross when each
    #: straddles the other's supporting line. When any orientation is zero the
    #: configuration is collinear or touching and is settled by a bounding-box
    #: containment test instead.
    #:
    #: ```
    #: let rising = Segment(Point2(0.0, 0.0), Point2(2.0, 2.0))
    #: let falling = Segment(Point2(0.0, 2.0), Point2(2.0, 0.0))
    #: rising.intersects(falling)    # true
    #: ```
    pub fn intersects(self, other: Segment) -> bool {
        let d1 = orientation(self.a, self.b, other.a)
        let d2 = orientation(self.a, self.b, other.b)
        let d3 = orientation(other.a, other.b, self.a)
        let d4 = orientation(other.a, other.b, self.b)
        if d1 != d2 and d3 != d4 { return true }
        if d1 == 0 and _within_bounds(self.a, self.b, other.a) { return true }
        if d2 == 0 and _within_bounds(self.a, self.b, other.b) { return true }
        if d3 == 0 and _within_bounds(other.a, other.b, self.a) { return true }
        return d4 == 0 and _within_bounds(other.a, other.b, self.b)
    }

    #: Return the single point the two segments share, or null.
    #:
    #: Null means either that they miss each other entirely or that they
    #: overlap along a stretch, where there is no single point to return. Use
    #: `intersects` for the yes-or-no question; the two disagree exactly on the
    #: collinear-overlap case, and that difference is the point of having both.
    #:
    #: ```
    #: let rising = Segment(Point2(0.0, 0.0), Point2(2.0, 2.0))
    #: let falling = Segment(Point2(0.0, 2.0), Point2(2.0, 0.0))
    #: rising.intersection(falling).x    # 1.0
    #: ```
    pub fn intersection(self, other: Segment) -> Point2? {
        if not self.intersects(other) { return null }
        let r = self.b - self.a
        let s = other.b - other.a
        if _cross_sign(r, s) == 0 { return _collinear_touch(self, other) }
        let t = (other.a - self.a).cross(s) / r.cross(s)
        return self.a + r * t
    }

    #: Return the segment rendered as `Segment(a, b)`.
    pub fn to_str(self) -> str { return f"Segment({self.a}, {self.b})" }

    fn __str__(self) -> str { return self.to_str() }
}

#: An axis-aligned rectangle, as its lower-left and upper-right corners.
pub class Rect: Printable {
    #: The corner with the smaller coordinates.
    pub let min_corner: Point2

    #: The corner with the larger coordinates.
    pub let max_corner: Point2

    #: Create the rectangle spanning `min_corner` to `max_corner`.
    #:
    #: Raises `ValueError` when either coordinate of `min_corner` exceeds the
    #: matching coordinate of `max_corner`; an inverted rectangle is a bug in
    #: the caller rather than an empty region.
    pub fn init(self, min_corner: Point2, max_corner: Point2) {
        if min_corner.x > max_corner.x or min_corner.y > max_corner.y {
            throw ValueError(f"Rect corners are inverted: {min_corner} to {max_corner}")
        }
        self.min_corner = min_corner
        self.max_corner = max_corner
    }

    #: Create the rectangle with its lower-left corner at `(x, y)`.
    #:
    #: Raises `ValueError` for a negative width or height.
    #:
    #: ```
    #: Rect.from_size(1.0, 1.0, 3.0, 2.0).area()    # 6.0
    #: ```
    pub static fn from_size(x: float, y: float, width: float, height: float) -> Rect {
        if width < 0.0 or height < 0.0 {
            throw ValueError(f"size must be non-negative, got {width} by {height}")
        }
        return Rect(Point2(x, y), Point2(x + width, y + height))
    }

    #: Return the horizontal extent.
    pub fn width(self) -> float { return self.max_corner.x - self.min_corner.x }

    #: Return the vertical extent.
    pub fn height(self) -> float { return self.max_corner.y - self.min_corner.y }

    #: Return the enclosed area.
    pub fn area(self) -> float { return self.width() * self.height() }

    #: Return the length of the boundary.
    pub fn perimeter(self) -> float { return 2.0 * (self.width() + self.height()) }

    #: Return the centre point.
    pub fn center(self) -> Point2 { return self.min_corner.lerp(self.max_corner, 0.5) }

    #: Report whether `p` lies inside the rectangle or on its boundary.
    pub fn contains(self, p: Point2) -> bool {
        return self.min_corner.x - EPSILON <= p.x <= self.max_corner.x + EPSILON and
            self.min_corner.y - EPSILON <= p.y <= self.max_corner.y + EPSILON
    }

    #: Report whether `other` lies wholly inside this rectangle.
    pub fn contains_rect(self, other: Rect) -> bool {
        return self.contains(other.min_corner) and self.contains(other.max_corner)
    }

    #: Report whether the two rectangles share any point, edges included.
    pub fn intersects(self, other: Rect) -> bool {
        return self.min_corner.x <= other.max_corner.x + EPSILON and
            other.min_corner.x <= self.max_corner.x + EPSILON and
            self.min_corner.y <= other.max_corner.y + EPSILON and
            other.min_corner.y <= self.max_corner.y + EPSILON
    }

    #: Return the overlapping region, or null when the two are disjoint.
    #:
    #: ```
    #: let left = Rect.from_size(0.0, 0.0, 2.0, 2.0)
    #: let right = Rect.from_size(1.0, 1.0, 2.0, 2.0)
    #: left.intersection(right).area()    # 1.0
    #: ```
    pub fn intersection(self, other: Rect) -> Rect? {
        let low = Point2(
            max(self.min_corner.x, other.min_corner.x),
            max(self.min_corner.y, other.min_corner.y)
        )
        let high = Point2(
            min(self.max_corner.x, other.max_corner.x),
            min(self.max_corner.y, other.max_corner.y)
        )
        if low.x > high.x or low.y > high.y { return null }
        return Rect(low, high)
    }

    #: Return the smallest rectangle containing both rectangles.
    pub fn union(self, other: Rect) -> Rect {
        return Rect(
            Point2(
                min(self.min_corner.x, other.min_corner.x),
                min(self.min_corner.y, other.min_corner.y)
            ),
            Point2(
                max(self.max_corner.x, other.max_corner.x),
                max(self.max_corner.y, other.max_corner.y)
            )
        )
    }

    #: Return the rectangle grown by `amount` on every side.
    #:
    #: A negative `amount` shrinks it. Raises `ValueError` when that would
    #: invert the rectangle.
    pub fn expanded(self, amount: float) -> Rect {
        if self.width() + 2.0 * amount < 0.0 or self.height() + 2.0 * amount < 0.0 {
            throw ValueError(f"expanding by {amount} would invert the rect")
        }
        return Rect(
            Point2(self.min_corner.x - amount, self.min_corner.y - amount),
            Point2(self.max_corner.x + amount, self.max_corner.y + amount)
        )
    }

    #: Return the four corners counter-clockwise from the lower left.
    pub fn corners(self) -> list[Point2] {
        return [
            self.min_corner,
            Point2(self.max_corner.x, self.min_corner.y),
            self.max_corner,
            Point2(self.min_corner.x, self.max_corner.y),
        ]
    }

    #: Return the rectangle rendered as `Rect(min_corner, max_corner)`.
    pub fn to_str(self) -> str { return f"Rect({self.min_corner}, {self.max_corner})" }

    fn __str__(self) -> str { return self.to_str() }
}

#: A circle, as a centre and a radius.
pub class Circle: Printable {
    #: The centre of the circle.
    pub let center: Point2

    #: The radius; zero for a degenerate point-circle.
    pub let radius: float

    #: Create the circle of radius `radius` around `center`.
    #:
    #: Raises `ValueError` for a negative radius.
    pub fn init(self, center: Point2, radius: float) {
        if radius < 0.0 { throw ValueError(f"radius must be non-negative, got {radius}") }
        self.center = center
        self.radius = radius
    }

    #: Return the enclosed area.
    pub fn area(self) -> float { return math.PI * self.radius * self.radius }

    #: Return the length of the boundary.
    pub fn circumference(self) -> float { return math.TAU * self.radius }

    #: Report whether `p` lies inside the circle or on its boundary.
    pub fn contains(self, p: Point2) -> bool {
        return self.center.distance(p) <= self.radius + EPSILON * (1.0 + self.radius)
    }

    #: Report whether `p` lies on the boundary, within the module tolerance.
    pub fn on_boundary(self, p: Point2) -> bool {
        let distance = self.center.distance(p)
        return math.fabs(distance - self.radius) <= EPSILON * (1.0 + self.radius)
    }

    #: Return the smallest axis-aligned rectangle containing the circle.
    pub fn bounding_box(self) -> Rect {
        return Rect(
            Point2(self.center.x - self.radius, self.center.y - self.radius),
            Point2(self.center.x + self.radius, self.center.y + self.radius)
        )
    }

    #: Return the points the two boundaries share: two, one, or none.
    #:
    #: One point means tangency. Two identical circles share their whole
    #: boundary and so return the empty list — there is no finite answer.
    #:
    #: ```
    #: let left = Circle(Point2(0.0, 0.0), 1.0)
    #: let right = Circle(Point2(2.0, 0.0), 1.0)
    #: left.intersection(right).len()    # 1
    #: ```
    pub fn intersection(self, other: Circle) -> list[Point2] {
        let separation = self.center.distance(other.center)
        let tolerance = EPSILON * (1.0 + self.radius + other.radius)
        if separation <= tolerance and math.fabs(self.radius - other.radius) <= tolerance {
            return []
        }
        if separation > self.radius + other.radius + tolerance { return [] }
        if separation < math.fabs(self.radius - other.radius) - tolerance { return [] }
        # Project onto the centre line: `along` is the distance from this
        # centre to the chord, `half_chord` the distance from the chord's
        # midpoint to each intersection.
        let along = (separation * separation + self.radius * self.radius - other.radius * other.radius) /
            (2.0 * separation)
        let squared = self.radius * self.radius - along * along
        let half_chord = math.sqrt(max(squared, 0.0))
        let unit = (other.center - self.center) / separation
        let base = self.center + unit * along
        if half_chord <= tolerance { return [base] }
        let offset = unit.perpendicular() * half_chord
        return [base + offset, base - offset]
    }

    #: Return the circle rendered as `Circle(center, radius)`.
    pub fn to_str(self) -> str { return f"Circle({self.center}, {self.radius})" }

    fn __str__(self) -> str { return self.to_str() }
}

#: A closed polygon, as a ring of vertices.
#:
#: The ring is implicitly closed: the last vertex joins back to the first, and
#: repeating the first vertex at the end would create a zero-length edge. The
#: class validates the ring once at construction so that the free functions
#: below, which take a bare `list[Point2]`, need not keep re-checking it.
pub class Polygon: Printable {
    let vertices: list[Point2]

    #: Create the polygon with the given vertex ring.
    #:
    #: Raises `ValueError` for fewer than three vertices.
    pub fn init(self, vertices: list[Point2]) {
        if vertices.len() < 3 {
            throw ValueError(f"a polygon needs at least 3 vertices, got {vertices.len()}")
        }
        self.vertices = vertices[0:vertices.len()]
    }

    #: Return the number of vertices.
    pub fn vertex_count(self) -> int { return self.vertices.len() }

    #: Return the vertex at `index`.
    #:
    #: Raises `IndexError` when `index` is out of range.
    pub fn vertex(self, index: int) -> Point2 {
        if index < 0 or index >= self.vertices.len() {
            throw IndexError(f"vertex index must be in 0..{self.vertices.len()}, got {index}")
        }
        return self.vertices[index]
    }

    #: Return a copy of the vertex ring, safe to hand to the free functions.
    pub fn to_list(self) -> list[Point2] { return self.vertices[0:self.vertices.len()] }

    #: Return the edges as segments, the closing edge last.
    pub fn edges(self) -> list[Segment] {
        let count = self.vertices.len()
        return [Segment(self.vertices[i], self.vertices[(i + 1) % count]) for i in 0..count]
    }

    #: Return the signed area: positive when the ring runs counter-clockwise.
    pub fn signed_area(self) -> float { return polygon_signed_area(self.vertices) }

    #: Return the enclosed area, regardless of winding direction.
    #:
    #: ```
    #: Polygon([Point2(0.0, 0.0), Point2(4.0, 0.0), Point2(4.0, 3.0)]).area()    # 6.0
    #: ```
    pub fn area(self) -> float { return polygon_area(self.vertices) }

    #: Return the total edge length.
    pub fn perimeter(self) -> float { return polygon_perimeter(self.vertices) }

    #: Return the area-weighted centroid.
    pub fn centroid(self) -> Point2 { return polygon_centroid(self.vertices) }

    #: Return the smallest axis-aligned rectangle containing the polygon.
    #:
    #: Calls the shared helper rather than the free `bounding_box`, which has
    #: the same name as this method.
    pub fn bounding_box(self) -> Rect { return _bounds_of(self.vertices) }

    #: Report whether `p` is inside the polygon or on its boundary.
    pub fn contains(self, p: Point2) -> bool { return point_in_polygon(self.vertices, p) }

    #: Report whether the ring turns the same way at every vertex.
    #:
    #: Collinear vertices do not break convexity. A self-intersecting ring is
    #: never convex by this test, since it must reverse somewhere.
    pub fn is_convex(self) -> bool { return _is_convex_ring(self.vertices) }

    #: Report whether the ring runs counter-clockwise.
    pub fn is_counter_clockwise(self) -> bool { return self.signed_area() > 0.0 }

    #: Report whether no two non-adjacent edges touch.
    #:
    #: O(n^2), comparing every pair of edges. Worth calling before
    #: `triangulate`, which assumes a simple polygon and can only report the
    #: failure late and vaguely.
    pub fn is_simple(self) -> bool {
        let sides = self.edges()
        let count = sides.len()
        for i in 0..count {
            for j in i + 1..count {
                # Adjacent edges are meant to share exactly one endpoint.
                if j == i + 1 or i == 0 and j == count - 1 { continue }
                if sides[i].intersects(sides[j]) { return false }
            }
        }
        return true
    }

    #: Return the polygon rendered as `Polygon(n vertices)`.
    pub fn to_str(self) -> str { return f"Polygon({self.vertices.len()} vertices)" }

    fn __len__(self) -> int { return self.vertices.len() }

    fn __str__(self) -> str { return self.to_str() }
}

#: Return 1 when `a`, `b`, `c` turn counter-clockwise, -1 clockwise, 0 collinear.
#:
#: The predicate the whole module rests on. It is the sign of the determinant
#: `(b - a) x (c - a)`, compared against a tolerance that grows with the
#: magnitude of the two products so that the answer is the same however far
#: from the origin the points sit. See the module doc comment for what that
#: tolerance can and cannot do.
#:
#: ```
#: orientation(Point2(0.0, 0.0), Point2(1.0, 0.0), Point2(1.0, 1.0))    # 1
#: orientation(Point2(0.0, 0.0), Point2(1.0, 0.0), Point2(2.0, 0.0))    # 0
#: ```
pub fn orientation(a: Point2, b: Point2, c: Point2) -> int {
    let x1 = b.x - a.x
    let y1 = b.y - a.y
    let x2 = c.x - a.x
    let y2 = c.y - a.y
    let first = x1 * y2
    let second = y1 * x2
    return _sign_within(first - second, math.fabs(first) + math.fabs(second))
}

#: Return how many times the polygon's boundary winds around `point`.
#:
#: Zero means outside. Any other value means inside, and its magnitude counts
#: the loops: a doubly wound ring reports 2, a ring wound the other way -1.
#: Points on the boundary are unspecified — ask `point_in_polygon`, which
#: settles them first.
#:
#: ```
#: let square = [Point2(0.0, 0.0), Point2(2.0, 0.0), Point2(2.0, 2.0), Point2(0.0, 2.0)]
#: winding_number(square, Point2(1.0, 1.0))     # 1
#: winding_number(square, Point2(3.0, 1.0))     # 0
#: ```
pub fn winding_number(ring: list[Point2], point: Point2) -> int {
    _require_ring(ring, "winding_number")
    let count = ring.len()
    var winding = 0
    for index in 0..count {
        let a = ring[index]
        let b = ring[(index + 1) % count]
        # Half-open in y so that a vertex exactly at the ray's height is
        # counted by one of its two edges and not by both.
        if a.y <= point.y {
            if b.y > point.y and orientation(a, b, point) > 0 { winding += 1 }
        } elif b.y <= point.y and orientation(a, b, point) < 0 {
            winding -= 1
        }
    }
    return winding
}

#: Report whether `point` is inside the polygon or on its boundary.
#:
#: The non-zero winding rule, and the one to reach for. It is the rule that
#: matches the intuition "the ring encloses this point": for a self-
#: intersecting ring, a region covered twice counts as inside, where the
#: even-odd rule of `point_in_polygon_crossing` punches it out again. It also
#: costs the same, and unlike the crossing test it answers boundary points
#: definitely — they are inside — instead of leaving them to the rounding.
#: On a simple polygon the two rules agree everywhere except the boundary.
#:
#: ```
#: let square = [Point2(0.0, 0.0), Point2(2.0, 0.0), Point2(2.0, 2.0), Point2(0.0, 2.0)]
#: point_in_polygon(square, Point2(1.0, 1.0))    # true
#: point_in_polygon(square, Point2(2.0, 1.0))    # true
#: point_in_polygon(square, Point2(2.5, 1.0))    # false
#: ```
pub fn point_in_polygon(ring: list[Point2], point: Point2) -> bool {
    _require_ring(ring, "point_in_polygon")
    let count = ring.len()
    for index in 0..count {
        let a = ring[index]
        let b = ring[(index + 1) % count]
        if orientation(a, b, point) == 0 and _within_bounds(a, b, point) { return true }
    }
    return winding_number(ring, point) != 0
}

#: Report whether `point` is inside the polygon by the even-odd rule.
#:
#: Ray casting: shoot a ray to the right and count boundary crossings, odd
#: meaning inside. Use it when even-odd is the semantics you want — it is what
#: SVG's `fill-rule: evenodd` and most CAD formats mean by a hole cut into a
#: ring drawn the same way round. Otherwise prefer `point_in_polygon`, whose
#: winding rule handles self-intersection more intuitively and does not leave
#: boundary points to chance; this test says nothing useful about a point that
#: lies exactly on an edge.
#:
#: ```
#: let square = [Point2(0.0, 0.0), Point2(2.0, 0.0), Point2(2.0, 2.0), Point2(0.0, 2.0)]
#: point_in_polygon_crossing(square, Point2(1.0, 1.0))    # true
#: point_in_polygon_crossing(square, Point2(3.0, 1.0))    # false
#: ```
pub fn point_in_polygon_crossing(ring: list[Point2], point: Point2) -> bool {
    _require_ring(ring, "point_in_polygon_crossing")
    let count = ring.len()
    var inside = false
    for index in 0..count {
        let a = ring[index]
        let b = ring[(index + 1) % count]
        # The half-open comparison is what makes a vertex on the ray count
        # once: exactly one of the two edges meeting there straddles it.
        if (a.y > point.y) != (b.y > point.y) {
            let crossing = a.x + (point.y - a.y) / (b.y - a.y) * (b.x - a.x)
            if point.x < crossing { inside = not inside }
        }
    }
    return inside
}

#: Return the vertices of the convex hull, counter-clockwise.
#:
#: Andrew's monotone chain: sort the points by x then y, sweep left to right
#: building the lower chain and right to left building the upper one, popping
#: any vertex that is not a strict left turn. O(n log n), all of it in the
#: sort. Duplicate points are removed and collinear points are dropped, so the
#: result is the minimal strictly convex ring. Fewer than three distinct
#: points, or all of them collinear, gives the one or two extreme points back
#: — a hull, just not a polygon.
#:
#: ```
#: let cloud = [Point2(0.0, 0.0), Point2(2.0, 0.0), Point2(1.0, 1.0), Point2(1.0, 0.2)]
#: let hull = convex_hull(cloud)
#: hull.len()      # 3
#: hull[0].x       # 0.0
#: ```
pub fn convex_hull(points: list[Point2]) -> list[Point2] {
    let ordered = _sorted_distinct(points)
    let count = ordered.len()
    if count <= 2 { return ordered }
    let lower: list[Point2] = []
    for point in ordered {
        _push_turning_left(lower, point)
    }
    let upper: list[Point2] = []
    var index = count - 1
    while index >= 0 {
        _push_turning_left(upper, ordered[index])
        index -= 1
    }
    # Each chain ends where the other begins; drop the duplicated endpoints.
    lower.pop()
    upper.pop()
    return lower + upper
}

#: Return the two closest points and the distance between them.
#:
#: Divide and conquer in O(n log n): split the points at the median x, solve
#: both halves, then check only the points within the better of the two
#: distances of the dividing line. That strip is scanned in y order, where a
#: geometric packing argument caps the work at a constant number of comparisons
#: per point — no pair further than seven positions apart in y can beat the
#: distance already found. The brute-force alternative is O(n^2).
#:
#: Both index lists are sorted once up front and partitioned on the way down,
#: which is what keeps the recursion linear per level rather than re-sorting.
#: Raises `ValueError` for fewer than two points. Duplicate points are a
#: legitimate answer at distance zero.
#:
#: ```
#: let cloud = [Point2(0.0, 0.0), Point2(5.0, 5.0), Point2(1.0, 1.0)]
#: let (first, second, gap) = closest_pair(cloud)
#: first.x                                # 0.0
#: second.x                               # 1.0
#: math.approx_eq(gap, math.SQRT_2)       # true
#: ```
pub fn closest_pair(points: list[Point2]) -> tuple[Point2, Point2, float] {
    let count = points.len()
    if count < 2 {
        throw ValueError(f"closest_pair needs at least 2 points, got {count}")
    }
    let indices = [index for index in 0..count]
    let by_x = sort_by(indices, |left, right| _compare_xy(points[left], points[right]))
    let by_y = sort_by(indices, |left, right| _compare_yx(points[left], points[right]))
    let in_left = [false for _ in 0..count]
    let (first, second, distance) = _closest_pair_of(points, by_x, by_y, in_left)
    return (points[first], points[second], distance)
}

#: Return the signed area of the ring: positive counter-clockwise.
#:
#: The shoelace formula, evaluated as a triangle fan from the first vertex so
#: that every coordinate is a local offset. Summing `x_i * y_(i+1)` on the raw
#: coordinates instead would, for a small polygon far from the origin, form
#: huge products and then subtract them — the classic cancellation, and the
#: reason a survey polygon in metres-from-the-equator can come out with the
#: wrong area entirely.
#:
#: ```
#: polygon_signed_area([Point2(0.0, 0.0), Point2(4.0, 0.0), Point2(4.0, 3.0)])    # 6.0
#: polygon_signed_area([Point2(0.0, 0.0), Point2(4.0, 3.0), Point2(4.0, 0.0)])    # -6.0
#: ```
pub fn polygon_signed_area(ring: list[Point2]) -> float {
    _require_ring(ring, "polygon_signed_area")
    let origin = ring[0]
    var total = 0.0
    for index in 1..ring.len() - 1 {
        total += (ring[index] - origin).cross(ring[index + 1] - origin)
    }
    return total * 0.5
}

#: Return the area enclosed by the ring, whichever way it winds.
#:
#: ```
#: polygon_area([Point2(0.0, 0.0), Point2(4.0, 0.0), Point2(4.0, 3.0)])    # 6.0
#: ```
pub fn polygon_area(ring: list[Point2]) -> float {
    return math.fabs(polygon_signed_area(ring))
}

#: Return the total length of the ring's edges, closing edge included.
#:
#: ```
#: polygon_perimeter([Point2(0.0, 0.0), Point2(3.0, 0.0), Point2(3.0, 4.0)])    # 12.0
#: ```
pub fn polygon_perimeter(ring: list[Point2]) -> float {
    _require_ring(ring, "polygon_perimeter")
    let count = ring.len()
    var total = 0.0
    for index in 0..count {
        total += ring[index].distance(ring[(index + 1) % count])
    }
    return total
}

#: Return the area-weighted centroid of the region the ring encloses.
#:
#: This is the centre of mass of the filled polygon, which is not the mean of
#: the vertices — see `centroid`. Subdividing an edge changes the vertex mean
#: and leaves this unmoved, which is the property that makes it the right
#: answer for a shape.
#:
#: Raises `ValueError` for a degenerate ring, whose zero area leaves the
#: weighted average undefined.
#:
#: ```
#: let square = [Point2(0.0, 0.0), Point2(2.0, 0.0), Point2(2.0, 2.0), Point2(0.0, 2.0)]
#: polygon_centroid(square).x    # 1.0
#: ```
pub fn polygon_centroid(ring: list[Point2]) -> Point2 {
    _require_ring(ring, "polygon_centroid")
    let origin = ring[0]
    var twice_area = 0.0
    var weighted_x = 0.0
    var weighted_y = 0.0
    for index in 1..ring.len() - 1 {
        let a = ring[index] - origin
        let b = ring[index + 1] - origin
        let cross = a.cross(b)
        twice_area += cross
        weighted_x += (a.x + b.x) * cross
        weighted_y += (a.y + b.y) * cross
    }
    if math.fabs(twice_area) <= EPSILON {
        throw ValueError("polygon_centroid is undefined for a polygon of zero area")
    }
    let scale = 3.0 * twice_area
    return Point2(origin.x + weighted_x / scale, origin.y + weighted_y / scale)
}

#: Return the arithmetic mean of the points.
#:
#: The centre of the *vertices*, not of the shape they outline: adding a point
#: pulls it, and subdividing an edge of a polygon moves it. For the centre of
#: mass of a filled polygon use `polygon_centroid`. This one is what a
#: clustering step or an angular sort wants.
#:
#: Raises `ValueError` for an empty list.
#:
#: ```
#: centroid([Point2(0.0, 0.0), Point2(2.0, 0.0), Point2(1.0, 3.0)]).y    # 1.0
#: ```
pub fn centroid(points: list[Point2]) -> Point2 {
    if points.len() == 0 { throw ValueError("centroid of an empty point set") }
    var total_x = 0.0
    var total_y = 0.0
    for point in points {
        total_x += point.x
        total_y += point.y
    }
    let count = float(points.len())
    return Point2(total_x / count, total_y / count)
}

#: Return the smallest axis-aligned rectangle containing every point.
#:
#: Raises `ValueError` for an empty list.
#:
#: ```
#: bounding_box([Point2(1.0, 5.0), Point2(-2.0, 0.0)]).width()    # 3.0
#: ```
pub fn bounding_box(points: list[Point2]) -> Rect { return _bounds_of(points) }

#: Return the circle through three points.
#:
#: The circumcircle: its centre is where the perpendicular bisectors of the
#: three chords meet. Computed after translating `a` to the origin, which
#: turns the textbook formula's differences of squared coordinates into
#: squares of differences and so removes the cancellation that dominates it
#: for points far from the origin.
#:
#: Raises `ValueError` when the three points are collinear — including when
#: two coincide — since no finite circle passes through them.
#:
#: ```
#: let unit = circle_from_points(Point2(1.0, 0.0), Point2(0.0, 1.0), Point2(-1.0, 0.0))
#: unit.center.x                            # 0.0
#: math.approx_eq(unit.radius, 1.0)         # true
#: ```
pub fn circle_from_points(a: Point2, b: Point2, c: Point2) -> Circle {
    if orientation(a, b, c) == 0 {
        throw ValueError(f"no circle passes through the collinear points {a}, {b}, {c}")
    }
    let u = b - a
    let v = c - a
    let denominator = 2.0 * u.cross(v)
    let u2 = u.length_squared()
    let v2 = v.length_squared()
    let offset = Point2((v.y * u2 - u.y * v2) / denominator, (u.x * v2 - v.x * u2) / denominator)
    return Circle(a + offset, offset.length())
}

#: Return the points where a line meets a circle: two, one, or none.
#:
#: Substituting the parametric line into the circle gives a quadratic in the
#: line parameter; its discriminant decides the count. One point means the line
#: is tangent, and because a discriminant of exactly zero is not something
#: floating point produces, near-zero counts as tangency. The results come back
#: in increasing order along the line's direction.
#:
#: ```
#: let axis = Line(Point2(-5.0, 0.0), Point2(5.0, 0.0))
#: let unit = Circle(Point2(0.0, 0.0), 1.0)
#: let hits = line_circle_intersection(axis, unit)
#: hits.len()      # 2
#: hits[0].x       # -1.0
#: ```
pub fn line_circle_intersection(line: Line, circle: Circle) -> list[Point2] {
    let offset = line.point - circle.center
    let a = line.direction.length_squared()
    let b = 2.0 * offset.dot(line.direction)
    let c = offset.length_squared() - circle.radius * circle.radius
    let discriminant = b * b - 4.0 * a * c
    let tolerance = EPSILON * (b * b + math.fabs(4.0 * a * c) + 1.0)
    if discriminant < -tolerance { return [] }
    if discriminant <= tolerance { return [line.at(-b / (2.0 * a))] }
    let root = math.sqrt(discriminant)
    return [line.at((-b - root) / (2.0 * a)), line.at((-b + root) / (2.0 * a))]
}

#: Return `subject` clipped to the convex region `window`.
#:
#: The Sutherland-Hodgman algorithm: run the whole subject ring through one
#: window edge at a time, keeping the vertices on the inside and adding a new
#: vertex wherever the ring crosses. n passes over the ring, so O(n * m).
#:
#: `window` must be convex — that is the algorithm's premise, and it is checked
#: — but `subject` need not be. A concave subject clipped by a window that cuts
#: it into two pieces comes back as one ring joined by a degenerate edge along
#: the window boundary; that is the known artefact of the method, and the price
#: of its simplicity. Greiner-Hormann is the algorithm to reach for when the
#: pieces must stay separate.
#:
#: ```
#: let square = [Point2(0.0, 0.0), Point2(4.0, 0.0), Point2(4.0, 4.0), Point2(0.0, 4.0)]
#: let window = [Point2(1.0, 1.0), Point2(3.0, 1.0), Point2(3.0, 3.0), Point2(1.0, 3.0)]
#: polygon_area(polygon_clip(square, window))    # 4.0
#: ```
pub fn polygon_clip(subject: list[Point2], window: list[Point2]) -> list[Point2] {
    _require_ring(subject, "polygon_clip")
    _require_ring(window, "polygon_clip")
    if not _is_convex_ring(window) {
        throw ValueError("polygon_clip requires a convex clipping window")
    }
    let bounds = _counter_clockwise(window)
    var output = subject[0:subject.len()]
    let edge_count = bounds.len()
    for index in 0..edge_count {
        if output.len() == 0 { break }
        let edge_start = bounds[index]
        let edge_end = bounds[(index + 1) % edge_count]
        let pending = output
        output = []
        var previous = pending[pending.len() - 1]
        for current in pending {
            let current_inside = orientation(edge_start, edge_end, current) >= 0
            let previous_inside = orientation(edge_start, edge_end, previous) >= 0
            if current_inside {
                if not previous_inside {
                    output.push(_line_crossing(edge_start, edge_end, previous, current))
                }
                output.push(current)
            } elif previous_inside {
                output.push(_line_crossing(edge_start, edge_end, previous, current))
            }
            previous = current
        }
    }
    return output
}

#: Return a triangulation of a simple polygon, as counter-clockwise triples.
#:
#: Ear clipping: repeatedly find a convex vertex whose triangle with its two
#: neighbours holds no other vertex, emit that triangle, and unlink the vertex.
#: Every simple polygon has at least two such ears at all times (the two-ears
#: theorem), so the search always finds one and the loop always makes progress.
#: O(n^2), which is the practical choice below a few thousand vertices;
#: Seidel's and Chazelle's algorithms do better asymptotically and are far
#: harder to make robust.
#:
#: The triangles tile the polygon exactly and none of them is degenerate.
#: Collinear vertices are dropped rather than turned into zero-area triangles,
#: both the ones present at the start and the ones that only become collinear
#: as their neighbours are clipped away, so the count is `m - 2` for the `m`
#: vertices that survive that pruning — at most `n - 2`, and fewer for a ring
#: with straight runs in it.
#:
#: Raises `ValueError` when the ring encloses no area, and when no ear can be
#: found, which means the ring intersects itself; `Polygon.is_simple`
#: diagnoses that ahead of time.
#:
#: ```
#: let square = [Point2(0.0, 0.0), Point2(2.0, 0.0), Point2(2.0, 2.0), Point2(0.0, 2.0)]
#: triangulate(square).len()    # 2
#: ```
pub fn triangulate(ring: list[Point2]) -> list[tuple[Point2, Point2, Point2]] {
    _require_ring(ring, "triangulate")
    let cleaned = _cleaned_ring(ring)
    if cleaned.len() < 3 {
        throw ValueError("triangulate expects a polygon with non-zero area")
    }
    let working = _counter_clockwise(cleaned)
    let count = working.len()
    let previous_of = [(index + count - 1) % count for index in 0..count]
    let next_of = [(index + 1) % count for index in 0..count]
    let triangles: list[tuple[Point2, Point2, Point2]] = []
    var remaining = count
    var current = 0
    var failures = 0
    while remaining > 3 {
        let previous = previous_of[current]
        let next = next_of[current]
        let turn = orientation(working[previous], working[current], working[next])
        if turn == 0 {
            # Clipping the neighbours has left this vertex in a straight line
            # between the two that remain. It bounds no area, so unlink it
            # without emitting anything: a zero-area triangle is not a
            # triangle, and leaving it in place deadlocks the search — a ring
            # of collinear leftovers has no ear at all.
            next_of[previous] = next
            previous_of[next] = previous
            remaining -= 1
            failures = 0
            current = next
        } elif turn > 0 and _is_ear(working, previous_of, next_of, previous, current, next) {
            triangles.push((working[previous], working[current], working[next]))
            next_of[previous] = next
            previous_of[next] = previous
            remaining -= 1
            failures = 0
            current = next
        } else {
            current = next
            failures += 1
            if failures > remaining {
                throw ValueError("triangulate found no ear; the polygon is not simple")
            }
        }
    }
    let last_previous = previous_of[current]
    let last_next = next_of[current]
    if orientation(working[last_previous], working[current], working[last_next]) != 0 {
        triangles.push((working[last_previous], working[current], working[last_next]))
    }
    return triangles
}

#: Return the two furthest-apart points and the distance between them.
#:
#: Rotating calipers over the convex hull. The diameter is always realised by
#: two hull vertices, and specifically by an antipodal pair — two vertices
#: with parallel supporting lines. Walking one edge of the hull forward while
#: advancing the opposite vertex only as far as the triangle area keeps growing
#: enumerates every antipodal pair in a single loop, so the whole search after
#: the hull is O(h) rather than the O(h^2) of comparing all pairs.
#:
#: Raises `ValueError` for an empty list. Identical points give that point
#: twice at distance zero.
#:
#: ```
#: let cloud = [Point2(0.0, 0.0), Point2(3.0, 0.0), Point2(0.0, 4.0), Point2(1.0, 1.0)]
#: let (first, second, span) = diameter(cloud)
#: span            # 5.0
#: first.x         # 3.0
#: second.y        # 4.0
#: ```
pub fn diameter(points: list[Point2]) -> tuple[Point2, Point2, float] {
    if points.len() == 0 { throw ValueError("diameter of an empty point set") }
    let hull = convex_hull(points)
    let count = hull.len()
    if count == 1 { return (hull[0], hull[0], 0.0) }
    if count == 2 { return (hull[0], hull[1], hull[0].distance(hull[1])) }
    var best_first = hull[0]
    var best_second = hull[1]
    var best = hull[0].distance(hull[1])
    var opposite = 1
    for index in 0..count {
        let next = (index + 1) % count
        # Advance the opposite vertex while it keeps moving away from this
        # edge. The bound only guards against a tie cycling forever.
        var steps = 0
        loop {
            if steps >= count { break }
            let ahead = _twice_triangle_area(hull[index], hull[next], hull[(opposite + 1) % count])
            let here = _twice_triangle_area(hull[index], hull[next], hull[opposite])
            if ahead <= here { break }
            opposite = (opposite + 1) % count
            steps += 1
        }
        let from_start = hull[index].distance(hull[opposite])
        if from_start > best {
            best = from_start
            best_first = hull[index]
            best_second = hull[opposite]
        }
        let from_end = hull[next].distance(hull[opposite])
        if from_end > best {
            best = from_end
            best_first = hull[next]
            best_second = hull[opposite]
        }
    }
    return (best_first, best_second, best)
}

#: Return the smallest circle containing every point.
#:
#: Welzl's algorithm in its incremental form. The key fact is that the answer
#: is fixed by at most three of the points, so the circle is grown one point at
#: a time and, whenever a point falls outside, rebuilt with that point pinned
#: to the boundary — recursively, to a depth of three. A point processed after
#: the circle already contains it costs one distance test, and a random order
#: makes the expensive rebuild rare enough that the whole thing is O(n)
#: expected. Processing the points in their given order is what would make it
#: quadratic, so they are permuted first; the permutation comes from a fixed
#: internal generator, so the same input always yields the same circle.
#:
#: Collinear inputs are handled: with no circumcircle to be had, the two
#: furthest-apart of the three candidate points define a diametral circle.
#:
#: Raises `ValueError` for an empty list.
#:
#: ```
#: let corners = [Point2(0.0, 0.0), Point2(2.0, 0.0), Point2(2.0, 2.0), Point2(0.0, 2.0)]
#: let circle = minimum_enclosing_circle(corners)
#: circle.center.x                              # 1.0
#: math.approx_eq(circle.radius, math.SQRT_2)   # true
#: ```
pub fn minimum_enclosing_circle(points: list[Point2]) -> Circle {
    if points.len() == 0 {
        throw ValueError("minimum_enclosing_circle of an empty point set")
    }
    let ordered = _pseudo_shuffled(points)
    var circle = Circle(ordered[0], 0.0)
    for index in 1..ordered.len() {
        if not _circle_holds(circle, ordered[index]) {
            circle = _enclosing_pinning_one(ordered, index, ordered[index])
        }
    }
    return circle
}

# --- Private helpers ---------------------------------------------------------

#: Return the sign of `value`, treating anything near enough to zero as zero.
#:
#: `scale` is the magnitude of the terms `value` was formed from, so the
#: tolerance is absolute near the origin and relative once the products grow.
fn _sign_within(value: float, scale: float) -> int {
    let tolerance = EPSILON * (1.0 + scale)
    if value > tolerance { return 1 }
    if value < -tolerance { return -1 }
    return 0
}

#: Return the tolerant sign of the cross product of two vectors.
fn _cross_sign(u: Point2, v: Point2) -> int {
    let first = u.x * v.y
    let second = u.y * v.x
    return _sign_within(first - second, math.fabs(first) + math.fabs(second))
}

fn _require_ring(ring: list[Point2], function_name: str) -> void {
    if ring.len() < 3 {
        throw ValueError(f"{function_name} needs at least 3 vertices, got {ring.len()}")
    }
}

#: Report whether `p` lies within the bounding box of `a` and `b`.
#:
#: Only meaningful once `p` is known to be collinear with them, which is how
#: every caller uses it.
fn _within_bounds(a: Point2, b: Point2, p: Point2) -> bool {
    return min(a.x, b.x) - EPSILON <= p.x <= max(a.x, b.x) + EPSILON and
        min(a.y, b.y) - EPSILON <= p.y <= max(a.y, b.y) + EPSILON
}

#: Return the single point two overlapping collinear segments share, or null.
fn _collinear_touch(first: Segment, second: Segment) -> Point2? {
    let candidates = [first.a, first.b, second.a, second.b]
    let shared: list[Point2] = []
    for point in candidates {
        if first.contains(point) and second.contains(point) {
            var seen = false
            for already in shared {
                if already.approx_eq(point) { seen = true }
            }
            if not seen { shared.push(point) }
        }
    }
    if shared.len() == 1 { return shared[0] }
    return null
}

fn _compare_xy(a: Point2, b: Point2) -> int {
    if a.x < b.x { return -1 }
    if a.x > b.x { return 1 }
    if a.y < b.y { return -1 }
    if a.y > b.y { return 1 }
    return 0
}

fn _compare_yx(a: Point2, b: Point2) -> int {
    if a.y < b.y { return -1 }
    if a.y > b.y { return 1 }
    if a.x < b.x { return -1 }
    if a.x > b.x { return 1 }
    return 0
}

#: Return the points sorted by x then y, with exact duplicates removed.
fn _sorted_distinct(points: list[Point2]) -> list[Point2] {
    let ordered = sort_by(points, _compare_xy)
    let distinct: list[Point2] = []
    for point in ordered {
        if distinct.len() == 0 or not distinct[distinct.len() - 1].approx_eq(point) {
            distinct.push(point)
        }
    }
    return distinct
}

#: Append `point` to a hull chain, popping vertices that do not turn left.
fn _push_turning_left(chain: list[Point2], point: Point2) -> void {
    loop {
        if chain.len() < 2 { break }
        let last = chain.len() - 1
        if orientation(chain[last - 1], chain[last], point) > 0 { break }
        chain.pop()
    }
    chain.push(point)
}

#: Return the closest pair among the indices in `by_x`, as (i, j, distance).
#:
#: `by_x` and `by_y` hold the same indices ordered by x and by y. `in_left` is
#: scratch space shared by the whole recursion so that the partition costs one
#: pass instead of a set lookup per element.
fn _closest_pair_of(
    points: list[Point2],
    by_x: list[int],
    by_y: list[int],
    in_left: list[bool]
) -> tuple[int, int, float] {
    let count = by_x.len()
    if count <= 3 { return _closest_pair_brute(points, by_x) }
    let middle = count // 2
    let split_x = points[by_x[middle]].x
    for position in 0..middle { in_left[by_x[position]] = true }
    let left_y: list[int] = []
    let right_y: list[int] = []
    for index in by_y {
        if in_left[index] { left_y.push(index) } else { right_y.push(index) }
    }
    for position in 0..middle { in_left[by_x[position]] = false }
    let (left_a, left_b, left_gap) = _closest_pair_of(points, by_x[0:middle], left_y, in_left)
    let (right_a, right_b, right_gap) = _closest_pair_of(
        points,
        by_x[middle:count],
        right_y,
        in_left
    )
    var best_a = left_a
    var best_b = left_b
    var best = left_gap
    if right_gap < best {
        best_a = right_a
        best_b = right_b
        best = right_gap
    }
    let strip: list[int] = []
    for index in by_y {
        if math.fabs(points[index].x - split_x) < best { strip.push(index) }
    }
    for i in 0..strip.len() {
        var j = i + 1
        # The y ordering bounds this inner loop at a constant number of steps:
        # past `best` in y no pair can improve on `best` in distance.
        while j < strip.len() and points[strip[j]].y - points[strip[i]].y < best {
            let gap = points[strip[i]].distance(points[strip[j]])
            if gap < best {
                best = gap
                best_a = strip[i]
                best_b = strip[j]
            }
            j += 1
        }
    }
    return (best_a, best_b, best)
}

#: Return the closest pair among two or three indices, by direct comparison.
#:
#: The base case of the recursion: below four points the divide step cannot
#: split them usefully, and three comparisons settle it.
fn _closest_pair_brute(points: list[Point2], indices: list[int]) -> tuple[int, int, float] {
    var best_a = indices[0]
    var best_b = indices[1]
    var best = points[best_a].distance(points[best_b])
    for i in 0..indices.len() {
        for j in i + 1..indices.len() {
            let gap = points[indices[i]].distance(points[indices[j]])
            if gap < best {
                best = gap
                best_a = indices[i]
                best_b = indices[j]
            }
        }
    }
    return (best_a, best_b, best)
}

#: Return the smallest axis-aligned rectangle containing every point.
fn _bounds_of(points: list[Point2]) -> Rect {
    if points.len() == 0 { throw ValueError("bounding_box of an empty point set") }
    var min_x = points[0].x
    var min_y = points[0].y
    var max_x = points[0].x
    var max_y = points[0].y
    for point in points {
        min_x = min(min_x, point.x)
        min_y = min(min_y, point.y)
        max_x = max(max_x, point.x)
        max_y = max(max_y, point.y)
    }
    return Rect(Point2(min_x, min_y), Point2(max_x, max_y))
}

fn _twice_triangle_area(a: Point2, b: Point2, c: Point2) -> float {
    return math.fabs((b - a).cross(c - a))
}

#: Report whether every turn of the ring goes the same way.
fn _is_convex_ring(ring: list[Point2]) -> bool {
    let count = ring.len()
    var sign = 0
    for index in 0..count {
        let turn = orientation(ring[index], ring[(index + 1) % count], ring[(index + 2) % count])
        if turn != 0 {
            if sign != 0 and turn != sign { return false }
            sign = turn
        }
    }
    return true
}

#: Return the ring wound counter-clockwise, reversing it if it is not.
fn _counter_clockwise(ring: list[Point2]) -> list[Point2] {
    if polygon_signed_area(ring) >= 0.0 { return ring[0:ring.len()] }
    let reversed_ring: list[Point2] = []
    var index = ring.len() - 1
    while index >= 0 {
        reversed_ring.push(ring[index])
        index -= 1
    }
    return reversed_ring
}

#: Return the ring without duplicate or collinear vertices.
fn _cleaned_ring(ring: list[Point2]) -> list[Point2] {
    let distinct: list[Point2] = []
    for point in ring {
        if distinct.len() == 0 or not distinct[distinct.len() - 1].approx_eq(point) {
            distinct.push(point)
        }
    }
    while distinct.len() > 1 and distinct[0].approx_eq(distinct[distinct.len() - 1]) {
        distinct.pop()
    }
    let count = distinct.len()
    if count < 3 { return distinct }
    let cleaned: list[Point2] = []
    for index in 0..count {
        let previous = distinct[(index + count - 1) % count]
        let next = distinct[(index + 1) % count]
        if orientation(previous, distinct[index], next) != 0 { cleaned.push(distinct[index]) }
    }
    return cleaned
}

#: Report whether the vertex `current` is an ear of the remaining ring.
fn _is_ear(
    ring: list[Point2],
    previous_of: list[int],
    next_of: list[int],
    previous: int,
    current: int,
    next: int
) -> bool {
    if orientation(ring[previous], ring[current], ring[next]) <= 0 { return false }
    var index = next_of[next]
    while index != previous {
        # Only a reflex vertex can be trapped inside another vertex's ear, and
        # skipping the convex ones also avoids rejecting an ear because a
        # neighbouring convex vertex grazes its boundary.
        let is_reflex = orientation(ring[previous_of[index]], ring[index], ring[next_of[index]]) < 0
        if is_reflex and _in_triangle(ring[index], ring[previous], ring[current], ring[next]) {
            return false
        }
        index = next_of[index]
    }
    return true
}

#: Report whether `p` is inside the counter-clockwise triangle, boundary included.
fn _in_triangle(p: Point2, a: Point2, b: Point2, c: Point2) -> bool {
    return orientation(a, b, p) >= 0 and orientation(b, c, p) >= 0 and orientation(c, a, p) >= 0
}

#: Return where the infinite lines `a`-`b` and `c`-`d` cross.
#:
#: The callers only reach this with a crossing that exists: one endpoint of
#: `c`-`d` is inside the half-plane of `a`-`b` and the other is not.
fn _line_crossing(a: Point2, b: Point2, c: Point2, d: Point2) -> Point2 {
    let along = d - c
    let edge = b - a
    let denominator = along.cross(edge)
    if denominator == 0.0 { return c }
    return c + along * ((a - c).cross(edge) / denominator)
}

#: Report whether `circle` contains `point`, tolerantly.
fn _circle_holds(circle: Circle, point: Point2) -> bool {
    return circle.center.distance(point) <= circle.radius + EPSILON * (1.0 + circle.radius)
}

#: Return the smallest circle with `first` and `second` on its boundary.
fn _circle_through_two(first: Point2, second: Point2) -> Circle {
    return Circle(first.lerp(second, 0.5), first.distance(second) / 2.0)
}

#: Return the smallest circle through three points, collinear ones included.
fn _circle_through_three(a: Point2, b: Point2, c: Point2) -> Circle {
    if orientation(a, b, c) != 0 { return circle_from_points(a, b, c) }
    # Collinear: the outermost two points span the whole set.
    var best = _circle_through_two(a, b)
    let with_c = _circle_through_two(a, c)
    if with_c.radius > best.radius { best = with_c }
    let other = _circle_through_two(b, c)
    if other.radius > best.radius { best = other }
    return best
}

#: Return the smallest circle over `points[0..count]` with `pinned` on its edge.
fn _enclosing_pinning_one(points: list[Point2], count: int, pinned: Point2) -> Circle {
    var circle = _circle_through_two(points[0], pinned)
    for index in 1..count {
        if not _circle_holds(circle, points[index]) {
            circle = _enclosing_pinning_two(points, index, pinned, points[index])
        }
    }
    return circle
}

#: Return the smallest circle over `points[0..count]` with two points on its edge.
fn _enclosing_pinning_two(
    points: list[Point2],
    count: int,
    first: Point2,
    second: Point2
) -> Circle {
    var circle = _circle_through_two(first, second)
    for index in 0..count {
        if not _circle_holds(circle, points[index]) {
            circle = _circle_through_three(first, second, points[index])
        }
    }
    return circle
}

#: Return the points in a fixed pseudo-random order.
#:
#: Fisher-Yates driven by a 31-bit linear congruential generator with the
#: constants from the C standard. The seed is fixed, so the permutation — and
#: therefore every circle `minimum_enclosing_circle` returns — is reproducible.
#: The generator's low bits are notoriously correlated, so the index comes from
#: the high half of the state.
fn _pseudo_shuffled(points: list[Point2]) -> list[Point2] {
    let out = points[0:points.len()]
    var state = 20250804
    var index = out.len() - 1
    while index > 0 {
        state = (state * 1103515245 + 12345) % 2147483648
        let target = state // 65536 % (index + 1)
        let held = out[index]
        out[index] = out[target]
        out[target] = held
        index -= 1
    }
    return out
}
