#: `Point2` and `Vec3`: the point and vector primitive types the rest of
#: `std.algo.geometry` is built from.
#:
#: Split out of `geometry.jai` because these two classes are the one piece of
#: that module with no dependency on anything else in it -- no orientation
#: test, no shape, no algorithm, just `std.math` and each other's arithmetic.
#: `EPSILON` moved out with them: `Point2.approx_eq` and `Vec3.approx_eq` use
#: it as their default tolerance, which makes this the natural dependency-free
#: base the rest of the split imports from. See `geometry.jai`'s module
#: comment for why the tolerance is shaped the way it is.
import std.math
from std.core import Printable

#: The tolerance that separates "zero" from "small" in every predicate in
#: `std.algo.geometry`.
#:
#: Used as an absolute floor and scaled by the magnitude of the operands where
#: the quantity being tested is a product. See `geometry.jai`'s module doc
#: comment.
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
