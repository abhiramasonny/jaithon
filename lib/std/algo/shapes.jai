#: The primitive shapes: `Line`, `Segment`, `Rect`, `Circle`, and the two free
#: functions that build or cut a circle (`circle_from_points`,
#: `line_circle_intersection`).
#:
#: Split out of `geometry.jai`. `Line` and `Segment` are the reason this group
#: did not split further, shape by shape: both reach directly into the
#: tolerant cross-product predicate (`_cross_sign`) and the collinear-overlap
#: helper (`_collinear_touch`), and those two stay `fn`, not `pub fn` -- they
#: are not useful outside the two methods that call them, so there is no
#: reason to widen them the way `sign_within` and `within_bounds` were widened
#: in `geometry.jai`. Keeping the classes that call them in the same file is
#: what lets them stay private. `Rect` and `Circle` do not need either helper;
#: they sit here because they are the same kind of thing, not because they
#: are coupled to `Line`/`Segment`.
import std.math
from std.core import Printable
from std.algo.vectors import Point2, EPSILON
from std.algo.geometry import orientation, sign_within, within_bounds

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
        return within_bounds(self.a, self.b, p)
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
        if d1 == 0 and within_bounds(self.a, self.b, other.a) { return true }
        if d2 == 0 and within_bounds(self.a, self.b, other.b) { return true }
        if d3 == 0 and within_bounds(other.a, other.b, self.a) { return true }
        return d4 == 0 and within_bounds(other.a, other.b, self.b)
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

# --- Private helpers ---------------------------------------------------------
#:
#: Used only by `Line` and `Segment` above, so unlike `sign_within` and
#: `within_bounds` in `geometry.jai` these never needed to become `pub`.

#: Return the tolerant sign of the cross product of two vectors.
fn _cross_sign(u: Point2, v: Point2) -> int {
    let first = u.x * v.y
    let second = u.y * v.x
    return sign_within(first - second, math.fabs(first) + math.fabs(second))
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
