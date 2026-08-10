#: Series for spatial data: pose clouds, ray bundles and grids.
#:
#: These exist for the case a chart library usually does badly — a robot's view
#: of the world, redrawn every frame. A particle filter carries thousands of
#: weighted poses, a laser scan a few hundred rays that share only their origin,
#: and a map an occupancy grid of tens of thousands of cells. Each of those is
#: one primitive here rather than one per element, which is the difference
#: between a frame budget that works and one that does not.
import std.math

from .base import Series
from ..frame import Frame
from ..geom import Point, Rect
from ..primitive import Primitive
from ..style import Color, MarkerShape, Paint, Stroke

#: Marker size for a particle when none is given, in device pixels.
pub const DEFAULT_PARTICLE_SIZE: float = 3.5

#: A cloud of positions, each optionally carrying a heading and a weight.
#:
#: With headings the particles are drawn as wedges, so the spread of *belief
#: about orientation* is visible rather than just the spread of position — which
#: is most of what a reader wants from a particle filter.
#:
#: With weights the opacity tracks the weight, so a converging filter visibly
#: concentrates. Weights are normalised against the largest, so their absolute
#: scale does not matter.
#:
#: ```
#: let cloud = PoseCloud([0.0, 1.0], [0.0, 1.0], label: "particles")
#: cloud.len()      # 2
#: ```
pub class PoseCloud: Series {
    var xs: list[float]
    var ys: list[float]
    var headings: list[float]
    var weights: list[float]
    var name: str?
    var size: float
    var shape: MarkerShape
    var base: Color
    var chosen: bool
    var opacity: float

    #: Create a cloud at `xs`, `ys`.
    #:
    #: `headings` are radians anticlockwise from the positive x axis; an empty
    #: list draws plain markers instead of wedges. `weights` may be empty, in
    #: which case every particle is drawn at full opacity.
    #:
    #: Raises `ValueError` when any supplied list disagrees in length.
    pub fn init(
        self,
        xs: list[float],
        ys: list[float],
        headings: list[float] = [],
        weights: list[float] = [],
        label: str? = null,
        color: Color? = null,
        size: float = DEFAULT_PARTICLE_SIZE,
        shape: MarkerShape = MarkerShape.Circle,
        opacity: float = 0.75
    ) {
        _check_lengths(xs, ys, headings, weights)
        self.xs = [value for value in xs]
        self.ys = [value for value in ys]
        self.headings = [value for value in headings]
        self.weights = [value for value in weights]
        self.name = label
        self.size = size
        self.shape = shape
        self.chosen = color != null
        self.base = color ?? Color.from_rgb(0, 0, 0)
        self.opacity = opacity
    }

    #: Replace the whole cloud, keeping the styling.
    #:
    #: This is the call a filter makes every frame, so it takes all four lists
    #: at once rather than leaving the cloud briefly inconsistent between
    #: separate updates.
    pub fn set_cloud(
        self,
        xs: list[float],
        ys: list[float],
        headings: list[float] = [],
        weights: list[float] = []
    ) -> void {
        _check_lengths(xs, ys, headings, weights)
        self.xs = [value for value in xs]
        self.ys = [value for value in ys]
        self.headings = [value for value in headings]
        self.weights = [value for value in weights]
    }

    #: How many particles the cloud holds.
    pub fn len(self) -> int {
        return self.xs.len()
    }

    #: Override the colour, after construction.
    pub fn color(self, value: Color) -> PoseCloud {
        self.base = value
        self.chosen = true
        return self
    }

    #: The box the particles occupy, or null when none is drawable.
    pub fn extent(self) -> Rect? {
        var points = []
        for (index, x) in self.xs.enumerate() {
            let y = self.ys[index]
            if math.is_finite(x) and math.is_finite(y) { points.push(Point(x, y)) }
        }
        return Rect.from_points(points)
    }

    #: One batched primitive, or a few when the weights split the cloud.
    #:
    #: Weighted particles are bucketed by opacity rather than drawn one at a
    #: time: a handful of batches keeps the per-primitive cost down while still
    #: showing the filter concentrating.
    pub fn draw(self, frame: Frame) -> list[Primitive] {
        if self.xs.len() == 0 { return [] }
        let oriented = self.headings.len() == self.xs.len()

        if self.weights.len() != self.xs.len() {
            let placed = self._place(frame, 0.0, 1.1)
            if placed.len() == 0 { return [] }
            let paint = Paint.solid(self.base.with_alpha(self.opacity))
            return [self._batch(placed, oriented, paint, frame)]
        }

        var heaviest = 0.0
        for weight in self.weights {
            if math.is_finite(weight) { heaviest = max(heaviest, weight) }
        }
        if heaviest <= 0.0 {
            let placed = self._place(frame, 0.0, 1.1)
            if placed.len() == 0 { return [] }
            let paint = Paint.solid(self.base.with_alpha(self.opacity))
            return [self._batch(placed, oriented, paint, frame)]
        }

        var out = []
        for bucket in 0..WEIGHT_BUCKETS {
            let low = bucket.to_float() / WEIGHT_BUCKETS.to_float()
            let high = (bucket + 1).to_float() / WEIGHT_BUCKETS.to_float()
            let placed = self._place(frame, low * heaviest, high * heaviest + 1e-12)
            if placed.len() == 0 { continue }
            # The faintest bucket still has to be visible, or a filter that has
            # not converged looks empty rather than uncertain.
            let strength = 0.25 + 0.75 * (bucket + 1).to_float() / WEIGHT_BUCKETS.to_float()
            let paint = Paint.solid(self.base.with_alpha(self.opacity * strength))
            out.push(self._batch(placed, oriented, paint, frame))
        }
        return out
    }

    #: The legend entry, or null when the cloud was not labelled.
    pub fn label(self) -> str? {
        return self.name
    }

    #: A few particles scattered across the swatch box.
    pub fn swatch(self, box: Rect) -> list[Primitive] {
        let paint = Paint.solid(self.base.with_alpha(self.opacity))
        let middle = box.center().y
        let dots = [
            Point(box.left() + box.width() * 0.25, middle),
            Point(box.center().x, middle - 2.0),
            Point(box.left() + box.width() * 0.75, middle + 1.0),
        ]
        return [Primitive.Points(dots, MarkerShape.Circle, self.size, paint)]
    }

    #: Take the cycle's colour unless one was chosen explicitly.
    pub fn adopt(self, color: Color) -> void {
        if not self.chosen { self.base = color }
    }

    #: The particles whose weight falls in `[low, high)`, in device space.
    fn _place(self, frame: Frame, low: float, high: float) -> list[tuple[Point, float]] {
        let weighted = self.weights.len() == self.xs.len()
        let oriented = self.headings.len() == self.xs.len()
        var out = []
        for (index, x) in self.xs.enumerate() {
            if weighted {
                let weight = self.weights[index]
                if not math.is_finite(weight) { continue }
                if weight < low or weight >= high { continue }
            }
            let placed = frame.place(x, self.ys[index])
            if placed == null { continue }
            out.push((placed, if oriented { self.headings[index] } else { 0.0 }))
        }
        return out
    }

    #: One primitive for a placed batch, oriented or not.
    fn _batch(
        self,
        placed: list[tuple[Point, float]],
        oriented: bool,
        paint: Paint,
        frame: Frame
    ) -> Primitive {
        var points = []
        var angles = []
        for (point, heading) in placed {
            points.push(point)
            angles.push(heading)
        }
        if oriented { return Primitive.Poses(points, angles, self.size * 2.0, paint) }
        return Primitive.Points(points, self.shape, self.size, paint)
    }

    fn __str__(self) -> str {
        return f"PoseCloud({self.xs.len()} particles, {self.name ?? "unlabelled"})"
    }
}

#: How many opacity buckets a weighted cloud is split into.
#:
#: Enough that a converging filter reads as a gradient, few enough that the
#: batching still pays.
pub const WEIGHT_BUCKETS = 4

#: A bundle of disconnected line segments, such as a laser scan.
#:
#: ```
#: let scan = Rays([0.0, 0.0], [0.0, 0.0], [1.0, -1.0], [1.0, 1.0])
#: scan.len()      # 2
#: ```
pub class Rays: Series {
    var origin_xs: list[float]
    var origin_ys: list[float]
    var end_xs: list[float]
    var end_ys: list[float]
    var name: str?
    var stroke: Stroke
    var chosen: bool
    var opacity: float

    #: Create a bundle from matching origin and endpoint lists.
    #:
    #: Raises `ValueError` when the four lists are not the same length.
    pub fn init(
        self,
        origin_xs: list[float],
        origin_ys: list[float],
        end_xs: list[float],
        end_ys: list[float],
        label: str? = null,
        color: Color? = null,
        width: float = 1.0,
        opacity: float = 0.35
    ) {
        let counts = [origin_xs.len(), origin_ys.len(), end_xs.len(), end_ys.len()]
        for count in counts {
            if count != counts[0] {
                throw ValueError(f"every ray list must be the same length, got {counts}")
            }
        }
        self.origin_xs = [value for value in origin_xs]
        self.origin_ys = [value for value in origin_ys]
        self.end_xs = [value for value in end_xs]
        self.end_ys = [value for value in end_ys]
        self.name = label
        self.chosen = color != null
        self.opacity = opacity
        self.stroke = Stroke((color ?? Color.from_rgb(0, 0, 0)).with_alpha(opacity), width: width)
    }

    #: Replace every ray, keeping the styling.
    pub fn set_rays(
        self,
        origin_xs: list[float],
        origin_ys: list[float],
        end_xs: list[float],
        end_ys: list[float]
    ) -> void {
        let counts = [origin_xs.len(), origin_ys.len(), end_xs.len(), end_ys.len()]
        for count in counts {
            if count != counts[0] {
                throw ValueError(f"every ray list must be the same length, got {counts}")
            }
        }
        self.origin_xs = [value for value in origin_xs]
        self.origin_ys = [value for value in origin_ys]
        self.end_xs = [value for value in end_xs]
        self.end_ys = [value for value in end_ys]
    }

    #: How many rays the bundle holds.
    pub fn len(self) -> int {
        return self.origin_xs.len()
    }

    #: The box every endpoint occupies, or null when none is finite.
    pub fn extent(self) -> Rect? {
        var points = []
        for (index, x) in self.origin_xs.enumerate() {
            let y = self.origin_ys[index]
            if math.is_finite(x) and math.is_finite(y) { points.push(Point(x, y)) }
            let end_x = self.end_xs[index]
            let end_y = self.end_ys[index]
            if math.is_finite(end_x) and math.is_finite(end_y) {
                points.push(Point(end_x, end_y))
            }
        }
        return Rect.from_points(points)
    }

    #: One `Segments` primitive holding every ray the frame can place.
    pub fn draw(self, frame: Frame) -> list[Primitive] {
        var points = []
        for (index, x) in self.origin_xs.enumerate() {
            let start = frame.place(x, self.origin_ys[index])
            let end = frame.place(self.end_xs[index], self.end_ys[index])
            if start == null or end == null { continue }
            points.push(start)
            points.push(end)
        }
        if points.len() < 2 { return [] }
        return [Primitive.Segments(points, self.stroke)]
    }

    #: The legend entry, or null when the bundle was not labelled.
    pub fn label(self) -> str? {
        return self.name
    }

    #: A fan of short rays across the swatch box.
    pub fn swatch(self, box: Rect) -> list[Primitive] {
        let origin = Point(box.left(), box.center().y)
        let points = [
            origin,
            Point(box.right(), box.bottom()),
            origin,
            Point(box.right(), box.center().y),
            origin,
            Point(box.right(), box.top()),
        ]
        return [Primitive.Segments(points, self.stroke)]
    }

    #: Take the cycle's colour unless one was chosen explicitly.
    pub fn adopt(self, color: Color) -> void {
        if not self.chosen { self.stroke.color = color.with_alpha(self.opacity) }
    }

    fn __str__(self) -> str {
        return f"Rays({self.origin_xs.len()} rays, {self.name ?? "unlabelled"})"
    }
}

#: A grid of values drawn as an image, such as an occupancy map.
#:
#: Cell `(row, column)` covers a rectangle of data space, with row zero at the
#: bottom so the picture matches the coordinates rather than the memory order.
#:
#: ```
#: let grid = Grid([0.0, 1.0, 1.0, 0.0], 2, 2, Rect(0.0, 0.0, 2.0, 2.0))
#: grid.extent().top()      # 2.0
#: ```
pub class Grid: Series {
    var cells: list[float]
    var columns: int
    var rows: int
    var box: Rect
    var name: str?
    var low_color: Color
    var high_color: Color
    var low: float
    var high: float

    #: Create a grid of `columns` by `rows` cells covering `box`.
    #:
    #: Values are mapped from `low` to `high` between the two colours; leaving
    #: them null takes the range from the data. A NaN cell is drawn fully
    #: transparent, which is how an unobserved cell shows as unknown rather than
    #: as free space.
    #:
    #: Raises `ValueError` when the cell count does not match the shape.
    pub fn init(
        self,
        cells: list[float],
        columns: int,
        rows: int,
        box: Rect,
        label: str? = null,
        low_color: Color? = null,
        high_color: Color? = null,
        low: float? = null,
        high: float? = null
    ) {
        if columns <= 0 or rows <= 0 {
            throw ValueError(f"a grid needs a positive shape, got {columns}x{rows}")
        }
        if cells.len() != columns * rows {
            throw ValueError(
                f"a {columns}x{rows} grid needs {columns * rows} cells, got {cells.len()}"
            )
        }
        self.cells = [value for value in cells]
        self.columns = columns
        self.rows = rows
        self.box = box
        self.name = label
        self.low_color = low_color ?? Color.from_rgb(255, 255, 255)
        self.high_color = high_color ?? Color.from_rgb(30, 30, 40)
        let (found_low, found_high) = _range_of(self.cells)
        self.low = low ?? found_low
        self.high = high ?? found_high
    }

    #: Replace the cell values, keeping the shape and the colouring.
    #:
    #: Raises `ValueError` when the count no longer matches the shape.
    pub fn set_cells(self, cells: list[float]) -> void {
        if cells.len() != self.columns * self.rows {
            throw ValueError(f"expected {self.columns * self.rows} cells, got {cells.len()}")
        }
        self.cells = [value for value in cells]
    }

    #: The rectangle the grid covers.
    pub fn extent(self) -> Rect? {
        return self.box
    }

    #: One `Raster` primitive covering the grid's device rectangle.
    pub fn draw(self, frame: Frame) -> list[Primitive] {
        let corner = frame.place(self.box.left(), self.box.top())
        let opposite = frame.place(self.box.right(), self.box.bottom())
        if corner == null or opposite == null { return [] }

        var pixels = []
        let span = self.high - self.low
        for row in 0..self.rows {
            # Row zero is the bottom of the data, and a pixel buffer starts at
            # the top, so the rows are emitted in reverse.
            let source = (self.rows - 1 - row) * self.columns
            for column in 0..self.columns {
                pixels.push(self._shade(self.cells[source + column], span))
            }
        }
        return [
            Primitive.Raster(
                Rect(corner.x, corner.y, opposite.x, opposite.y),
                pixels,
                self.columns,
                self.rows
            ),
        ]
    }

    #: The legend entry, or null when the grid was not labelled.
    pub fn label(self) -> str? {
        return self.name
    }

    #: The swatch box filled with the high end of the ramp.
    pub fn swatch(self, box: Rect) -> list[Primitive] {
        let corners = [
            Point(box.left(), box.bottom()),
            Point(box.right(), box.bottom()),
            Point(box.right(), box.top()),
            Point(box.left(), box.top()),
        ]
        return [Primitive.Polygon(corners, Paint.solid(self.high_color), null)]
    }

    #: A grid keeps its own colouring and ignores the cycle.
    pub fn adopt(self, color: Color) -> void {
    }

    #: One cell as a packed colour.
    fn _shade(self, value: float, span: float) -> int {
        if not math.is_finite(value) { return 0 }
        if span <= 0.0 { return self.high_color.packed() }
        let unit = max(0.0, min(1.0, (value - self.low) / span))
        return self.low_color.lerp(self.high_color, unit).packed()
    }

    fn __str__(self) -> str {
        return f"Grid({self.columns}x{self.rows}, {self.name ?? "unlabelled"})"
    }
}

#: The finite range of a cell list, falling back to the unit interval.
fn _range_of(cells: list[float]) -> tuple[float, float] {
    var low = 0.0
    var high = 0.0
    var seen = false
    for value in cells {
        if not math.is_finite(value) { continue }
        if not seen {
            low = value
            high = value
            seen = true
        } else {
            low = min(low, value)
            high = max(high, value)
        }
    }
    if not seen or high <= low { return (0.0, 1.0) }
    return (low, high)
}

#: Reject a cloud whose supplied lists disagree in length.
fn _check_lengths(
    xs: list[float],
    ys: list[float],
    headings: list[float],
    weights: list[float]
) -> void {
    if xs.len() != ys.len() {
        throw ValueError(f"x and y must be the same length, got {xs.len()} and {ys.len()}")
    }
    if headings.len() > 0 and headings.len() != xs.len() {
        throw ValueError(f"expected {xs.len()} headings, got {headings.len()}")
    }
    if weights.len() > 0 and weights.len() != xs.len() {
        throw ValueError(f"expected {xs.len()} weights, got {weights.len()}")
    }
}
