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

#: Opacity a filled band takes when none is given and the colour carries none.
pub const DEFAULT_FILL_OPACITY: float = 0.28

#: Width a marker outline takes when a colour was asked for but no width.
#:
#: One pixel is matplotlib's `lines.markeredgewidth`, and it is the width at
#: which an outline separates two overlapping markers without becoming the
#: thing you see first.
pub const DEFAULT_MARKER_EDGE_WIDTH: float = 1.0

#: The cap a stroke should take when the caller has not chosen one.
#:
#: A dashed stroke butts, because a round or square cap extends every dash by
#: half the line width at *each* end: at the default 1.75 width the pattern
#: [1, 2] grows to 2.75 on and 0.25 off and draws as a solid line, and at width
#: 6 the pattern [6, 3] closes its gaps entirely. That is why jaiplot could not
#: draw a dotted line at all. matplotlib butts its dashes for the same reason
#: (`lines.dash_capstyle`), and leaves the join round.
#:
#: A solid stroke stays round, which is what keeps a dense polyline from
#: growing spikes where it doubles back on itself.
#:
#: Note that the pattern is in device pixels either way: unlike matplotlib,
#: jaiplot does not scale it by the line width, because `Stroke.dash` is the
#: SVG `stroke-dasharray` and both backends hand it straight through.
fn _default_cap(dash: list[float]) -> LineCap {
    if dash.len() > 0 { return LineCap.Butt }
    return LineCap.Round
}

#: The opacity a series should draw at, given what the caller supplied.
#:
#: An explicit `opacity` wins. Otherwise a colour that already carries an alpha
#: keeps it — four series used to force their own figure over it, so a colour
#: the caller had made 25% transparent came out fully opaque and the
#: overplotting they were trying to see through stayed hidden. A colour with no
#: alpha of its own, and the opaque colours the cycle hands out, fall back to
#: the series default.
#:
#: It lives here beside `finite_extent` rather than in `style`, because it is
#: about how a *series* reads its arguments and not about what a colour is.
pub fn carried_opacity(color: Color?, opacity: float?, fallback: float) -> float {
    if opacity != null { return opacity }
    if color != null and color.alpha() < 255 { return color.opacity() }
    return fallback
}

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
    var capped: bool
    var marker: MarkerShape?
    var marker_size: float
    var marker_edge_color: Color?
    var marker_edge_width: float

    #: Plot `ys` against `xs`.
    #:
    #: `dash` is alternating on and off lengths in device pixels, and `cap`
    #: says how each of those ends. Leaving `cap` null butts a dashed stroke
    #: and rounds a solid one, which is the only pair of defaults under which
    #: the pattern you asked for is the pattern you get — see `_default_cap`.
    #:
    #: `marker_edge_color` and `marker_edge_width` outline the markers, which
    #: is how a marker stays legible over a fill of its own colour. Giving a
    #: colour alone takes `DEFAULT_MARKER_EDGE_WIDTH`; giving a width alone
    #: outlines in the series colour, as matplotlib's `markeredgecolor`
    #: default does.
    #:
    #: Raises `ValueError` when the two lists differ in length, which is nearly
    #: always a mistake at the call site rather than something to draw. Also on
    #: a dash pattern with a negative length or one summing to zero.
    pub fn init(
        self,
        xs: list[float],
        ys: list[float],
        label: str? = null,
        color: Color? = null,
        width: float = DEFAULT_WIDTH,
        dash: list[float] = [],
        cap: LineCap? = null,
        join: LineJoin? = null,
        marker: MarkerShape? = null,
        marker_size: float = 5.0,
        marker_edge_color: Color? = null,
        marker_edge_width: float = 0.0
    ) {
        if xs.len() != ys.len() {
            throw ValueError(f"x and y must be the same length, got {xs.len()} and {ys.len()}")
        }
        self.xs = [value for value in xs]
        self.ys = [value for value in ys]
        self.name = label
        self.chosen = color != null
        self.capped = cap != null
        self.stroke = Stroke(
            color ?? Color.from_rgb(0, 0, 0),
            width: width,
            dash: dash,
            cap: cap ?? _default_cap(dash),
            join: join ?? LineJoin.Round
        )
        self.marker = marker
        self.marker_size = marker_size
        self.marker_edge_color = marker_edge_color
        self.marker_edge_width = if marker_edge_width > 0.0 {
            marker_edge_width
        } elif marker_edge_color != null {
            DEFAULT_MARKER_EDGE_WIDTH
        } else {
            0.0
        }
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
    #:
    #: The pattern is rebuilt through `Stroke.init` rather than assigned to the
    #: field, because assigning skipped every check the constructor makes:
    #: `[5.0, -3.0]` reached the document as `stroke-dasharray="5,-3"`, which
    #: is invalid SVG that the raster backend ignored and a viewer honoured, so
    #: the two backends drew different pictures of the same figure.
    #:
    #: The cap follows the new pattern unless one was chosen explicitly, so
    #: dashing a line after the fact does not leave it with the round cap that
    #: would fill the gaps back in.
    #:
    #: Raises `ValueError` on a negative length, or on a pattern summing to
    #: zero, which would leave a dashing walker stepping nowhere for ever.
    pub fn dash(self, pattern: list[float]) -> Line {
        self.stroke = Stroke(self.stroke.color, width: self.stroke.width, dash: pattern, cap: if self.capped {
            self.stroke.cap
        } else {
            _default_cap(pattern)
        }, join: self.stroke.join)
        return self
    }

    #: Override how the stroke ends, after construction.
    #:
    #: Choosing one pins it: a later `dash` will not take it back.
    pub fn cap(self, value: LineCap) -> Line {
        self.stroke.cap = value
        self.capped = true
        return self
    }

    #: Override how two segments of the stroke meet, after construction.
    pub fn join(self, value: LineJoin) -> Line {
        self.stroke.join = value
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
        let edge = self._marker_edge()
        for run in runs(frame, self.xs, self.ys) {
            if run.len() >= 2 { out.push(Primitive.Polyline(run, self.stroke)) }
            if self.marker != null {
                let shape = self.marker ?? MarkerShape.Circle
                let paint = Paint.solid(self.stroke.color)
                for point in run {
                    out.push(Primitive.Marker(point, shape, self.marker_size, paint, edge))
                }
            }
        }
        return out
    }

    #: The marker outline, or null when none was asked for.
    #:
    #: Built here rather than held as a field so that an edge left to default
    #: follows the series colour through `adopt`: a stroke frozen at
    #: construction would outline every cycled series in the black the
    #: constructor started from.
    fn _marker_edge(self) -> Stroke? {
        if self.marker_edge_width <= 0.0 { return null }
        return Stroke(
            self.marker_edge_color ?? self.stroke.color,
            width: self.marker_edge_width,
            cap: LineCap.Butt,
            join: LineJoin.Miter
        )
    }

    #: The legend entry, or null when the series was not labelled.
    pub fn label(self) -> str? {
        return self.name
    }

    #: A short rule across the swatch box, carrying the series' marker.
    #:
    #: The marker is drawn as well as the rule because two series that differ
    #: only in their marker had identical legend entries otherwise, and because
    #: a series drawn at zero width is *only* its markers — the swatch was
    #: blank for it.
    pub fn swatch(self, box: Rect) -> list[Primitive] {
        let middle = box.center().y
        let ends = [Point(box.left(), middle), Point(box.right(), middle)]
        var out = []
        if self.stroke.is_visible() { out.push(Primitive.Polyline(ends, self.stroke)) }
        if self.marker != null {
            let shape = self.marker ?? MarkerShape.Circle
            let size = min(self.marker_size, box.height())
            let paint = Paint.solid(self.stroke.color)
            out.push(Primitive.Marker(box.center(), shape, size, paint, self._marker_edge()))
        }
        return out
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
    #: A null `opacity` means "whatever the colour says": a colour handed in
    #: with an alpha of its own keeps it, and anything else — an opaque colour,
    #: or the one the cycle will hand over later — takes
    #: `DEFAULT_FILL_OPACITY`. Passing a number overrides both.
    #:
    #: Raises `ValueError` unless all three lists are the same length.
    pub fn init(
        self,
        xs: list[float],
        lower: list[float],
        upper: list[float],
        label: str? = null,
        color: Color? = null,
        opacity: float? = null
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
        self.opacity = carried_opacity(color, opacity, DEFAULT_FILL_OPACITY)
        self.fill = Paint.solid((color ?? Color.from_rgb(0, 0, 0)).with_alpha(self.opacity))
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
