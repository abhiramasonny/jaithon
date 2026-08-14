#: Line, step and filled-band series.
#:
#: All three share one shape: a pair of equal-length coordinate lists, walked
#: into device space and emitted as polylines. They differ only in what they put
#: between two consecutive samples — a straight segment, a pair of right angles,
#: or a filled region down to a second series.
import std.math

from .base import Series
from ..frame import Frame, runs
from ..geom import Point, Rect
from ..primitive import Primitive
from ..style import Color, LineCap, LineJoin, MarkerShape, Paint, Stroke

#: Default thickness for a line series, in device pixels.
pub const DEFAULT_WIDTH: float = 1.75

#: Opacity a filled band takes when none is given.
pub const DEFAULT_FILL_OPACITY: float = 0.28

#: A series drawn as straight segments between consecutive samples.
#:
#: A sample either scale cannot place — a NaN, or a zero on a log axis — breaks
#: the line rather than being drawn through, so a gap in the data reads as a gap
#: in the picture.
#:
#: ```
#: let line = Line([0.0, 1.0, 2.0], [0.0, 1.0, 4.0], label: "squares")
#: line.label()                     # "squares"
#: line.extent().top()              # 4.0
#: ```
pub class Line: Series {
    var xs: list[float]
    var ys: list[float]
    var name: str?
    var stroke: Stroke
    var chosen: bool
    var marker: MarkerShape?
    var marker_size: float

    #: Plot `ys` against `xs`.
    #:
    #: Raises `ValueError` when the two lists differ in length, which is nearly
    #: always a mistake at the call site rather than something to draw.
    pub fn init(
        self,
        xs: list[float],
        ys: list[float],
        label: str? = null,
        color: Color? = null,
        width: float = DEFAULT_WIDTH,
        dash: list[float] = [],
        marker: MarkerShape? = null,
        marker_size: float = 5.0
    ) {
        if xs.len() != ys.len() {
            throw ValueError(f"x and y must be the same length, got {xs.len()} and {ys.len()}")
        }
        self.xs = [value for value in xs]
        self.ys = [value for value in ys]
        self.name = label
        self.chosen = color != null
        self.stroke = Stroke(
            color ?? Color.from_rgb(0, 0, 0),
            width: width,
            dash: dash,
            cap: LineCap.Round,
            join: LineJoin.Round
        )
        self.marker = marker
        self.marker_size = marker_size
    }

    #: Override the colour, after construction.
    pub fn color(self, value: Color) -> Line {
        self.stroke.color = value
        self.chosen = true
        return self
    }

    #: Override the thickness, after construction.
    pub fn width(self, value: float) -> Line {
        self.stroke.width = value
        return self
    }

    #: Override the dash pattern, after construction.
    pub fn dash(self, pattern: list[float]) -> Line {
        self.stroke.dash = [length for length in pattern]
        return self
    }

    #: Replace the data, keeping the styling.
    #:
    #: An animation updates a series thousands of times, and rebuilding the
    #: figure each frame would throw away its colour, its place in the legend
    #: and the cycle position it was given.
    #:
    #: Raises `ValueError` when the two lists differ in length.
    pub fn set_data(self, xs: list[float], ys: list[float]) -> void {
        if xs.len() != ys.len() {
            throw ValueError(f"x and y must be the same length, got {xs.len()} and {ys.len()}")
        }
        self.xs = [value for value in xs]
        self.ys = [value for value in ys]
    }

    #: The box the samples occupy, or null when none can be drawn.
    pub fn extent(self) -> Rect? {
        return finite_extent(self.xs, self.ys)
    }

    #: One polyline per unbroken run, plus a marker at every placed sample.
    pub fn draw(self, frame: Frame) -> list[Primitive] {
        var out = []
        for run in runs(frame, self.xs, self.ys) {
            if run.len() >= 2 { out.push(Primitive.Polyline(run, self.stroke)) }
            if self.marker != null {
                let shape = self.marker ?? MarkerShape.Circle
                let paint = Paint.solid(self.stroke.color)
                for point in run {
                    out.push(Primitive.Marker(point, shape, self.marker_size, paint, null))
                }
            }
        }
        return out
    }

    #: The legend entry, or null when the series was not labelled.
    pub fn label(self) -> str? {
        return self.name
    }

    #: A short rule across the swatch box, in this series' stroke.
    pub fn swatch(self, box: Rect) -> list[Primitive] {
        let middle = box.center().y
        let ends = [Point(box.left(), middle), Point(box.right(), middle)]
        return [Primitive.Polyline(ends, self.stroke)]
    }

    #: Take the cycle's colour unless one was chosen explicitly.
    pub fn adopt(self, color: Color) -> void {
        if not self.chosen { self.stroke.color = color }
    }

    fn __str__(self) -> str {
        return f"Line({self.xs.len()} points, {self.name ?? "unlabelled"})"
    }
}

#: A series held constant between samples, stepping at each one.
#:
#: Use it where the value genuinely does not change between measurements — a
#: counter, a piecewise rate, a histogram outline — so a straight interpolation
#: would claim something the data does not say.
#:
#: ```
#: let step = Step([0.0, 1.0, 2.0], [1.0, 3.0, 2.0])
#: step.extent().top()      # 3.0
#: ```
pub class Step: Series {
    var xs: list[float]
    var ys: list[float]
    var name: str?
    var stroke: Stroke
    var chosen: bool
    var after: bool

    #: Plot `ys` against `xs` as a staircase.
    #:
    #: `after` puts the riser at the right of each sample, which is the usual
    #: reading for a value that holds until the next measurement. Setting it
    #: false puts the riser at the left instead.
    pub fn init(
        self,
        xs: list[float],
        ys: list[float],
        label: str? = null,
        color: Color? = null,
        width: float = DEFAULT_WIDTH,
        after: bool = true
    ) {
        if xs.len() != ys.len() {
            throw ValueError(f"x and y must be the same length, got {xs.len()} and {ys.len()}")
        }
        self.xs = [value for value in xs]
        self.ys = [value for value in ys]
        self.name = label
        self.chosen = color != null
        self.stroke = Stroke(
            color ?? Color.from_rgb(0, 0, 0),
            width: width,
            cap: LineCap.Butt,
            join: LineJoin.Miter
        )
        self.after = after
    }

    #: Replace the data, keeping the styling.
    #:
    #: An animation updates a series thousands of times, and rebuilding the
    #: figure each frame would throw away its colour, its place in the legend
    #: and the cycle position it was given.
    #:
    #: Raises `ValueError` when the two lists differ in length.
    pub fn set_data(self, xs: list[float], ys: list[float]) -> void {
        if xs.len() != ys.len() {
            throw ValueError(f"x and y must be the same length, got {xs.len()} and {ys.len()}")
        }
        self.xs = [value for value in xs]
        self.ys = [value for value in ys]
    }

    #: The box the samples occupy, or null when none can be drawn.
    pub fn extent(self) -> Rect? {
        return finite_extent(self.xs, self.ys)
    }

    #: One polyline per run, with a riser inserted between each pair.
    pub fn draw(self, frame: Frame) -> list[Primitive] {
        var out = []
        for run in runs(frame, self.xs, self.ys) {
            if run.len() < 2 { continue }
            var walked = [run[0]]
            for index in 1..run.len() {
                let previous = run[index - 1]
                let current = run[index]
                let corner = if self.after {
                    Point(current.x, previous.y)
                } else {
                    Point(previous.x, current.y)
                }
                walked.push(corner)
                walked.push(current)
            }
            out.push(Primitive.Polyline(walked, self.stroke))
        }
        return out
    }

    #: The legend entry, or null when the series was not labelled.
    pub fn label(self) -> str? {
        return self.name
    }

    #: A short rule across the swatch box.
    pub fn swatch(self, box: Rect) -> list[Primitive] {
        let middle = box.center().y
        let ends = [Point(box.left(), middle), Point(box.right(), middle)]
        return [Primitive.Polyline(ends, self.stroke)]
    }

    #: Take the cycle's colour unless one was chosen explicitly.
    pub fn adopt(self, color: Color) -> void {
        if not self.chosen { self.stroke.color = color }
    }

    fn __str__(self) -> str {
        return f"Step({self.xs.len()} points, {self.name ?? "unlabelled"})"
    }
}

#: The region between two curves over the same x values.
#:
#: The band is drawn as one closed polygon per unbroken run: out along the upper
#: curve and back along the lower. A gap either curve cannot place ends the
#: current band and starts a new one, so a filled region never spans a hole in
#: the data.
#:
#: ```
#: let band = FillBetween([0.0, 1.0], [0.0, 0.0], [1.0, 2.0])
#: band.extent().top()      # 2.0
#: ```
pub class FillBetween: Series {
    var xs: list[float]
    var lower: list[float]
    var upper: list[float]
    var name: str?
    var fill: Paint
    var chosen: bool
    var opacity: float

    #: Fill between `lower` and `upper` over `xs`.
    #:
    #: Raises `ValueError` unless all three lists are the same length.
    pub fn init(
        self,
        xs: list[float],
        lower: list[float],
        upper: list[float],
        label: str? = null,
        color: Color? = null,
        opacity: float = DEFAULT_FILL_OPACITY
    ) {
        if xs.len() != lower.len() or xs.len() != upper.len() {
            throw ValueError(
                f"x, lower and upper must match, got {xs.len()}, {lower.len()}, {upper.len()}"
            )
        }
        self.xs = [value for value in xs]
        self.lower = [value for value in lower]
        self.upper = [value for value in upper]
        self.name = label
        self.chosen = color != null
        self.opacity = opacity
        self.fill = Paint.solid((color ?? Color.from_rgb(0, 0, 0)).with_alpha(opacity))
    }

    #: The box both curves occupy, or null when neither can be drawn.
    pub fn extent(self) -> Rect? {
        let low = finite_extent(self.xs, self.lower)
        let high = finite_extent(self.xs, self.upper)
        if low == null { return high }
        if high == null { return low }
        return low.union(high)
    }

    #: Replace the band, keeping the styling.
    #:
    #: Raises `ValueError` unless all three lists are the same length.
    pub fn set_band(self, xs: list[float], lower: list[float], upper: list[float]) -> void {
        if xs.len() != lower.len() or xs.len() != upper.len() {
            throw ValueError(
                f"x, lower and upper must match, got {xs.len()}, {lower.len()}, {upper.len()}"
            )
        }
        self.xs = [value for value in xs]
        self.lower = [value for value in lower]
        self.upper = [value for value in upper]
    }

    #: One closed polygon per run where both curves are placeable.
    pub fn draw(self, frame: Frame) -> list[Primitive] {
        var out = []
        var top = []
        var bottom = []
        for (index, x) in self.xs.enumerate() {
            let high = frame.place(x, self.upper[index])
            let low = frame.place(x, self.lower[index])
            if high == null or low == null {
                _flush_band(out, top, bottom, self.fill)
                top = []
                bottom = []
            } else {
                top.push(high)
                bottom.push(low)
            }
        }
        _flush_band(out, top, bottom, self.fill)
        return out
    }

    #: The legend entry, or null when the series was not labelled.
    pub fn label(self) -> str? {
        return self.name
    }

    #: The swatch box filled in the band's colour.
    pub fn swatch(self, box: Rect) -> list[Primitive] {
        let corners = [
            Point(box.left(), box.bottom()),
            Point(box.right(), box.bottom()),
            Point(box.right(), box.top()),
            Point(box.left(), box.top()),
        ]
        return [Primitive.Polygon(corners, self.fill, null)]
    }

    #: Take the cycle's colour unless one was chosen explicitly.
    pub fn adopt(self, color: Color) -> void {
        if not self.chosen { self.fill = Paint.solid(color.with_alpha(self.opacity)) }
    }

    fn __str__(self) -> str {
        return f"FillBetween({self.xs.len()} points, {self.name ?? "unlabelled"})"
    }
}

#: Close one band and append it, if it holds enough points to have area.
fn _flush_band(out: list[Primitive], top: list[Point], bottom: list[Point], fill: Paint) -> void {
    if top.len() < 2 { return }
    var ring = [point for point in top]
    var index = bottom.len() - 1
    while index >= 0 {
        ring.push(bottom[index])
        index -= 1
    }
    out.push(Primitive.Polygon(ring, fill, null))
}

#: The bounding box of the finite samples, or null when there are none.
#:
#: Non-finite values are skipped rather than widening the box to infinity, which
#: would take the whole figure's limits with it.
pub fn finite_extent(xs: list[float], ys: list[float]) -> Rect? {
    var points = []
    for (index, x) in xs.enumerate() {
        let y = ys[index]
        if math.is_finite(x) and math.is_finite(y) { points.push(Point(x, y)) }
    }
    return Rect.from_points(points)
}
