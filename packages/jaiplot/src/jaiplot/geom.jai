#: Points, axis-aligned rectangles, and the data-to-device transform.
#:
#: These three types are the whole coordinate vocabulary of jaiplot. Series,
#: axes and backends all speak in `Point` and `Rect`, and `Transform` is the
#: single place where a data coordinate becomes a device coordinate.
#:
#: ```
#: let transform = Transform(Rect(0.0, 0.0, 10.0, 1.0), Rect(0.0, 0.0, 100.0, 50.0))
#: transform.apply(0.0, 0.0)      # Point(0.0, 50.0)
#: transform.apply(10.0, 1.0)     # Point(100.0, 0.0)
#: transform.invert(50.0, 25.0)   # (5.0, 0.5)
#: ```
import std.math

#: A monotone warp of one axis, carrying a data value into the linear space
#: that the device rectangle spans, or back out of it.
#:
#: This is the seam a non-linear scale is composed in through. `LinearScale`
#: contributes nothing and lets `Transform` default to the identity; a
#: `LogScale` contributes `log` one way and `exp` the other.
#:
#: A map warps values along an axis, it does not reverse one. `Transform` warps
#: the data rectangle's own edges with the same function, so `data.left()`
#: always lands on `device.left()` whether the map increases or decreases.
pub type AxisMap = fn(float) -> float

#: A location in the plane.
#:
#: Immutable: the arithmetic operators return fresh points, so a point handed
#: to a display list cannot be edited underneath the backend that holds it.
#:
#: ```
#: let point = Point(3.0, 4.0) + Point(1.0, 1.0)
#: point.x                       # 4.0
#: point - Point(1.0, 1.0)       # Point(3.0, 4.0)
#: ```
pub class Point {
    #: The horizontal coordinate.
    pub let x: float

    #: The vertical coordinate.
    pub let y: float

    #: Create a point at `(x, y)`.
    pub fn init(self, x: float, y: float) {
        self.x = x
        self.y = y
    }

    fn __str__(self) -> str {
        return f"Point({self.x}, {self.y})"
    }

    fn __eq__(self, other: any) -> bool {
        if not isinstance(other, Point) { return false }
        return self.x == other.x and self.y == other.y
    }

    fn __hash__(self) -> int {
        return hash((self.x, self.y))
    }

    fn __add__(self, other: Point) -> Point {
        return Point(self.x + other.x, self.y + other.y)
    }

    fn __sub__(self, other: Point) -> Point {
        return Point(self.x - other.x, self.y - other.y)
    }
}

#: An axis-aligned rectangle, stored as its two extreme corners.
#:
#: The corners are stored rather than an origin plus a width and a height
#: because `intersect`, `union` and `from_points` — the operations a plotting
#: library performs constantly, on clip stacks and on series extents — are pure
#: `min` and `max` over the stored fields in this form, and four signed
#: additions in the other, each of which is somewhere to get a sign wrong. The
#: constructor normalises, so `left() <= right()` and `bottom() <= top()` always
#: hold and nothing downstream has to defend against a reversed rectangle.
#:
#: `bottom()` is the numerically smallest y and `top()` the largest, which is
#: the data-space reading. A device rectangle, whose y grows downward, is
#: therefore stored with its on-screen top edge as `bottom()`. That is
#: deliberate: a `Rect` describes an interval product in whichever space it was
#: handed, and the y flip lives in exactly one expression, inside `Transform`.
#:
#: Zero extent on either axis is legal and every method handles it. A constant
#: data series has no y range at all, and that is ordinary input rather than an
#: error.
#:
#: ```
#: let rect = Rect(0.0, 10.0, 4.0, 0.0)
#: rect.bottom()                     # 0.0
#: rect.top()                        # 10.0
#: rect.width()                      # 4.0
#: rect.center()                     # Point(2.0, 5.0)
#: ```
pub class Rect {
    let x_min: float
    let y_min: float
    let x_max: float
    let y_max: float

    #: Create the rectangle spanned by the corners `(left, bottom)` and
    #: `(right, top)`.
    #:
    #: The corners may arrive in any order; each axis is normalised so the
    #: smaller coordinate becomes the minimum. Passing a device rectangle in the
    #: screen order `(left, top, right, bottom)` therefore still produces the
    #: right box.
    #:
    #: ```
    #: Rect(4.0, 3.0, 1.0, 9.0)      # Rect(1.0, 3.0, 4.0, 9.0)
    #: ```
    pub fn init(self, left: float, bottom: float, right: float, top: float) {
        self.x_min = min(left, right)
        self.x_max = max(left, right)
        self.y_min = min(bottom, top)
        self.y_max = max(bottom, top)
    }

    #: Return the bounding box of `points`, or null when the list is empty.
    #:
    #: ```
    #: Rect.from_points([Point(1.0, 5.0), Point(-2.0, 0.0)])   # Rect(-2.0, 0.0, 1.0, 5.0)
    #: ```
    pub static fn from_points(points: list[Point]) -> Rect? {
        if points.len() == 0 { return null }
        var x_min = points[0].x
        var x_max = points[0].x
        var y_min = points[0].y
        var y_max = points[0].y
        for point in points {
            x_min = min(x_min, point.x)
            x_max = max(x_max, point.x)
            y_min = min(y_min, point.y)
            y_max = max(y_max, point.y)
        }
        return Rect(x_min, y_min, x_max, y_max)
    }

    #: The smallest x coordinate.
    pub fn left(self) -> float {
        return self.x_min
    }

    #: The largest x coordinate.
    pub fn right(self) -> float {
        return self.x_max
    }

    #: The smallest y coordinate, which is the visual bottom in data space.
    pub fn bottom(self) -> float {
        return self.y_min
    }

    #: The largest y coordinate, which is the visual top in data space.
    pub fn top(self) -> float {
        return self.y_max
    }

    #: The extent along x, never negative.
    pub fn width(self) -> float {
        return self.x_max - self.x_min
    }

    #: The extent along y, never negative.
    pub fn height(self) -> float {
        return self.y_max - self.y_min
    }

    #: The midpoint of the rectangle.
    pub fn center(self) -> Point {
        return Point((self.x_min + self.x_max) * 0.5, (self.y_min + self.y_max) * 0.5)
    }

    #: Report whether `point` lies inside the rectangle or on its border.
    #:
    #: The border counts, so a sample sitting exactly on an axis limit is in
    #: view rather than one ulp outside it.
    #:
    #: ```
    #: Rect(0.0, 0.0, 2.0, 2.0).contains(Point(2.0, 0.0))    # true
    #: Rect(0.0, 0.0, 2.0, 2.0).contains(Point(3.0, 1.0))    # false
    #: ```
    pub fn contains(self, point: Point) -> bool {
        return self.x_min <= point.x <= self.x_max and self.y_min <= point.y <= self.y_max
    }

    #: Return the overlap with `other`, or null when they do not overlap.
    #:
    #: Rectangles that merely touch overlap in a rectangle of zero extent,
    #: which is returned rather than reported as no overlap: a clip stack that
    #: has narrowed to nothing still has a well-defined region.
    #:
    #: ```
    #: Rect(0.0, 0.0, 4.0, 4.0).intersect(Rect(2.0, 2.0, 6.0, 6.0))   # Rect(2.0, 2.0, 4.0, 4.0)
    #: Rect(0.0, 0.0, 1.0, 1.0).intersect(Rect(5.0, 5.0, 6.0, 6.0))   # null
    #: ```
    pub fn intersect(self, other: Rect) -> Rect? {
        let left = max(self.x_min, other.x_min)
        let right = min(self.x_max, other.x_max)
        if left > right { return null }
        let bottom = max(self.y_min, other.y_min)
        let top = min(self.y_max, other.y_max)
        if bottom > top { return null }
        return Rect(left, bottom, right, top)
    }

    #: Return the smallest rectangle containing both this one and `other`.
    #:
    #: ```
    #: Rect(0.0, 0.0, 1.0, 1.0).union(Rect(3.0, 3.0, 4.0, 4.0))   # Rect(0.0, 0.0, 4.0, 4.0)
    #: ```
    pub fn union(self, other: Rect) -> Rect {
        return Rect(
            min(self.x_min, other.x_min),
            min(self.y_min, other.y_min),
            max(self.x_max, other.x_max),
            max(self.y_max, other.y_max)
        )
    }

    #: Return this rectangle shrunk by `dx` on the left and right and by `dy` on
    #: the top and bottom.
    #:
    #: Negative amounts grow it, which is how a padded extent is built. An inset
    #: deeper than half the extent collapses that axis onto its midpoint instead
    #: of turning the rectangle inside out, so the result is always contained in
    #: the original.
    #:
    #: ```
    #: Rect(0.0, 0.0, 10.0, 10.0).inset(1.0, 2.0)   # Rect(1.0, 2.0, 9.0, 8.0)
    #: Rect(0.0, 0.0, 10.0, 10.0).inset(99.0, 0.0)  # Rect(5.0, 0.0, 5.0, 10.0)
    #: ```
    pub fn inset(self, dx: float, dy: float) -> Rect {
        let (left, right) = _inset_axis(self.x_min, self.x_max, dx)
        let (bottom, top) = _inset_axis(self.y_min, self.y_max, dy)
        return Rect(left, bottom, right, top)
    }

    #: Report whether the rectangle encloses no area.
    #:
    #: True when either extent is zero, which is the shape a constant series or
    #: a fully collapsed clip region takes.
    #:
    #: ```
    #: Rect(0.0, 3.0, 5.0, 3.0).is_empty()    # true
    #: Rect(0.0, 0.0, 5.0, 3.0).is_empty()    # false
    #: ```
    pub fn is_empty(self) -> bool {
        return self.width() == 0.0 or self.height() == 0.0
    }

    fn __str__(self) -> str {
        return f"Rect({self.x_min}, {self.y_min}, {self.x_max}, {self.y_max})"
    }

    fn __eq__(self, other: any) -> bool {
        if not isinstance(other, Rect) { return false }
        let same_x = self.x_min == other.x_min and self.x_max == other.x_max
        let same_y = self.y_min == other.y_min and self.y_max == other.y_max
        return same_x and same_y
    }

    fn __hash__(self) -> int {
        return hash((self.x_min, self.y_min, self.x_max, self.y_max))
    }
}

#: An affine map from a data rectangle onto a device rectangle, with the y axis
#: flipped.
#:
#: Data y increases upward and device y increases downward, so the bottom of the
#: data rectangle lands on the largest device y. This is the only place in
#: jaiplot where that flip happens.
#:
#: A non-linear axis is composed in through the optional per-axis mapping
#: functions. `x_forward` warps a data value into the linear space the device
#: rectangle spans, and `x_inverse` undoes it; both default to the identity, so
#: a plain `Transform(data, device)` is linear on both axes. The data rectangle
#: is always given in data units — the constructor warps its edges — so a
#: logarithmic x axis is `Transform(data, device, x_forward: |v| math.log10(v),
#: x_inverse: |v| 10.0 ** v)` with `data` still holding 1.0 and 1000.0.
#:
#: ```
#: import std.math
#: let device = Rect(0.0, 0.0, 100.0, 100.0)
#: let logarithmic = Transform(
#:     Rect(1.0, 0.0, 1000.0, 1.0),
#:     device,
#:     x_forward: |v| math.log10(v),
#:     x_inverse: |v| 10.0 ** v
#: )
#: logarithmic.apply(1000.0, 1.0)     # Point(100.0, 0.0)
#: logarithmic.apply(10.0, 0.0)       # Point(33.33333333333333, 100.0)
#: ```
pub class Transform {
    #: The source rectangle, in data units.
    pub let data: Rect

    #: The destination rectangle, in device pixels.
    pub let device: Rect

    let x_forward: AxisMap
    let x_inverse: AxisMap
    let y_forward: AxisMap
    let y_inverse: AxisMap
    let x_low: float
    let x_span: float
    let y_low: float
    let y_span: float
    let device_width: float
    let device_height: float

    #: Build the transform from `data` onto `device`.
    #:
    #: Raises `ValueError` when either rectangle has a zero extent, or when a
    #: mapping function turns a data extent into something non-finite. A zero
    #: divisor here would not fail loudly, it would fill the display list with
    #: infinities and take the whole figure down with it.
    pub fn init(
        self,
        data: Rect,
        device: Rect,
        x_forward: AxisMap? = null,
        x_inverse: AxisMap? = null,
        y_forward: AxisMap? = null,
        y_inverse: AxisMap? = null
    ) {
        self.data = data
        self.device = device
        self.x_forward = x_forward ?? (|value| value)
        self.x_inverse = x_inverse ?? (|value| value)
        self.y_forward = y_forward ?? (|value| value)
        self.y_inverse = y_inverse ?? (|value| value)
        self.x_low = self.x_forward(data.left())
        self.y_low = self.y_forward(data.bottom())
        self.x_span = _extent(self.x_low, self.x_forward(data.right()), "data x extent")
        self.y_span = _extent(self.y_low, self.y_forward(data.top()), "data y extent")
        self.device_width = _extent(device.left(), device.right(), "device width")
        self.device_height = _extent(device.bottom(), device.top(), "device height")
    }

    #: Map the data coordinate `(x, y)` to its device point.
    #:
    #: ```
    #: let transform = Transform(Rect(0.0, 0.0, 2.0, 2.0), Rect(0.0, 0.0, 8.0, 8.0))
    #: transform.apply(1.0, 1.0)    # Point(4.0, 4.0)
    #: transform.apply(0.0, 2.0)    # Point(0.0, 0.0)
    #: ```
    pub fn apply(self, x: float, y: float) -> Point {
        let unit_x = (self.x_forward(x) - self.x_low) / self.x_span
        let unit_y = (self.y_forward(y) - self.y_low) / self.y_span
        return Point(
            self.device.left() + unit_x * self.device_width,
            self.device.top() - unit_y * self.device_height
        )
    }

    #: Map the device coordinate `(px, py)` back to `(x, y)` in data units.
    #:
    #: This is what the window backend's cursor readout and its zoom about the
    #: pointer are built on. Points outside the device rectangle extrapolate
    #: rather than clamp.
    #:
    #: ```
    #: Transform(Rect(0.0, 0.0, 2.0, 2.0), Rect(0.0, 0.0, 8.0, 8.0)).invert(4.0, 4.0)   # (1.0, 1.0)
    #: ```
    pub fn invert(self, px: float, py: float) -> tuple[float, float] {
        let unit_x = (px - self.device.left()) / self.device_width
        let unit_y = (self.device.top() - py) / self.device_height
        return (
            self.x_inverse(self.x_low + unit_x * self.x_span),
            self.y_inverse(self.y_low + unit_y * self.y_span)
        )
    }

    fn __str__(self) -> str {
        return f"Transform({self.data} -> {self.device})"
    }
}

#: The outline of a pose marker: a wedge pointing along `angle`.
#:
#: A particle filter's whole point is that a particle carries a heading as well
#: as a position, so it cannot be drawn as a dot. The wedge is a narrow triangle
#: with a notched tail, which reads as a direction even at three pixels across.
#:
#: `angle` is in radians anticlockwise from the positive x axis in data space.
#: Device y grows downward, so the sine is negated on the way in and the marker
#: points the way the data says.
#:
#: ```
#: pose_outline(Point(0.0, 0.0), 2.0, 0.0).len()      # 4
#: ```
pub fn pose_outline(at: Point, radius: float, angle: float) -> list[Point] {
    let forward_x = math.cos(angle)
    let forward_y = -math.sin(angle)
    let side_x = -forward_y
    let side_y = forward_x
    let back = radius * 0.7
    let flank = radius * 0.55
    let notch = radius * 0.25
    return [
        Point(at.x + forward_x * radius, at.y + forward_y * radius),
        Point(
            at.x - forward_x * back + side_x * flank,
            at.y - forward_y * back + side_y * flank
        ),
        Point(at.x - forward_x * notch, at.y - forward_y * notch),
        Point(
            at.x - forward_x * back - side_x * flank,
            at.y - forward_y * back - side_y * flank
        ),
    ]
}

#: The two line segments of a plus or a diagonal cross marker.
#:
#: `diagonal` picks the two strokes: the axis-aligned horizontal and vertical
#: bars of a `Plus`, or the two corner-to-corner diagonals of a `Cross`. Shared
#: by the SVG and raster backends, which stroke the returned segments rather
#: than fill them.
#:
#: ```
#: cross_arms(Point(0.0, 0.0), 2.0, false).len()      # 2
#: ```
pub fn cross_arms(at: Point, radius: float, diagonal: bool) -> list[list[Point]] {
    if diagonal {
        let reach = radius * 0.7071
        return [
            [Point(at.x - reach, at.y - reach), Point(at.x + reach, at.y + reach)],
            [Point(at.x - reach, at.y + reach), Point(at.x + reach, at.y - reach)],
        ]
    }
    return [
        [Point(at.x - radius, at.y), Point(at.x + radius, at.y)],
        [Point(at.x, at.y - radius), Point(at.x, at.y + radius)],
    ]
}

#: Signed span from `low` to `high`, rejecting the values that would poison
#: every coordinate downstream.
#:
#: The span may be negative, because a decreasing `AxisMap` warps the low edge
#: above the high one. That needs no correction: the numerator of the fraction
#: it divides flips with it. Zero and non-finite are the two that cannot be
#: tolerated, because every point computed from them is infinity or NaN and the
#: whole display list inherits it.
fn _extent(low: float, high: float, description: str) -> float {
    let span = high - low
    if not math.is_finite(span) {
        throw ValueError(f"{description} must be finite, got {low} to {high}")
    }
    if span == 0.0 {
        throw ValueError(f"{description} must be non-zero, got {low} to {high}")
    }
    return span
}

#: One axis of `Rect.inset`, collapsed to its midpoint when the inset is deeper
#: than the extent it is eating into.
fn _inset_axis(low: float, high: float, amount: float) -> tuple[float, float] {
    let new_low = low + amount
    let new_high = high - amount
    if new_low > new_high {
        let middle = (low + high) * 0.5
        return (middle, middle)
    }
    return (new_low, new_high)
}
