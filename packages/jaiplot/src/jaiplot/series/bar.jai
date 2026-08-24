#: A series drawn as rectangles rising from a baseline.
import std.math

from .base import Series
from ..frame import Frame
from ..geom import Point, Rect
from ..primitive import Primitive
from ..scale import Scale
from ..style import Color, Paint, Stroke

#: Fraction of the slot between neighbouring bars that a bar fills.
#:
#: The remainder is the gap. Eight tenths reads as separate bars while still
#: making the group obvious.
pub const DEFAULT_FILL_FRACTION: float = 0.8

#: The scale and the low limit of one of `frame`'s two axes.
#:
#: Bars and histogram bins both have a value axis and a category axis that trade
#: places when the chart is turned on its side, and both need the scale in order
#: to ask whether a coordinate can be shown at all. Picking the axis in one
#: place keeps the two files from disagreeing about which one is which.
pub fn axis_of(frame: Frame, vertical: bool) -> tuple[Scale, float] {
    if vertical { return (frame.y_scale, frame.data.bottom()) }
    return (frame.x_scale, frame.data.left())
}

#: `value` if the axis can show it, otherwise the low end of that axis.
#:
#: A bar stands on zero and a logarithmic axis has no position for zero, so
#: `frame.place` returned null for the near corner of every bar and `draw`
#: dropped the lot: `ax.bar([0, 1, 2, 3], [3, 50, 200, 9])` followed by
#: `ax.set_yscale("log")` produced a white panel with tick labels on it and no
#: complaint. matplotlib runs the rectangle off the bottom of the axes instead,
#: which is the only honest picture — the bottom of a log panel is not zero and
#: pretending otherwise is what the clipping says. The clip rectangle `Axes`
#: pushes around the series trims the overhang.
pub fn clamp_to_axis(scale: Scale, value: float, low: float) -> float {
    if scale.accepts(value) { return value }
    return low
}

#: Spread a scalar-or-list baseline over `count` marks.
#:
#: matplotlib's `bottom=` takes either, and the list form is the whole of how a
#: stack is expressed: the second layer sits on the first layer's tops. Taking
#: only a scalar meant a stack had to be faked by moving the baseline of the
#: whole series, which drew the second layer *downward* through the first.
#:
#: Raises `ValueError` on a list of the wrong length, since a bar whose baseline
#: is missing has nothing to stand on.
pub fn baselines_over(baseline: any, count: int, what: str) -> list[float] {
    if baseline is list {
        let given = [float(value) for value in baseline]
        if given.len() != count {
            throw ValueError(f"expected {count} baselines to match the {what}, got {given.len()}")
        }
        return given
    }
    let flat = float(baseline)
    if not math.is_finite(flat) {
        throw ValueError(f"the baseline must be finite, got {flat}")
    }
    return [flat for _index in 0..count]
}

#: Rectangles from a baseline to each value.
#:
#: Bars are positioned by their centres. Widths are taken from the closest pair
#: of neighbouring centres, so an irregularly spaced set of bars still leaves a
#: gap everywhere rather than overlapping at the tightest pair.
#:
#: A bar chart whose baseline is not zero misleads, so the baseline is zero
#: unless deliberately moved, and an axis with bars on it is told to include
#: the baseline in its limits.
#:
#: `values` are the tops of the bars, not their heights. That is the natural
#: spelling for the common chart and the one `Axes.bar` offers; `from_heights`
#: is the other spelling, and the one a stack wants.
#:
#: ```
#: let bars = Bar([0.0, 1.0, 2.0], [3.0, 1.0, 4.0], label: "counts")
#: bars.extent().top()          # 4.0
#: bars.extent().bottom()       # 0.0
#: ```
pub class Bar: Series {
    var centres: list[float]
    var values: list[float]
    var bottoms: list[float]
    var name: str?
    var fill: Paint
    var edge: Stroke?
    var chosen: bool
    var fraction: float
    var horizontal: bool
    var explicit_width: float?

    #: Draw a bar at each centre, reaching from the baseline to its value.
    #:
    #: `baseline` is one number for every bar or one per bar, matching
    #: matplotlib's `bottom=`.
    pub fn init(
        self,
        centres: list[float],
        values: list[float],
        label: str? = null,
        color: Color? = null,
        baseline: any = 0.0,
        width: float? = null,
        fraction: float = DEFAULT_FILL_FRACTION,
        horizontal: bool = false,
        edge: Stroke? = null
    ) {
        if centres.len() != values.len() {
            throw ValueError(
                f"centres and values must match, got {centres.len()} and {values.len()}"
            )
        }
        if fraction <= 0.0 or fraction > 1.0 {
            throw ValueError(f"the fill fraction must lie in (0, 1], got {fraction}")
        }
        self.centres = [value for value in centres]
        self.values = [value for value in values]
        self.bottoms = baselines_over(baseline, self.centres.len(), "bars")
        self.name = label
        self.chosen = color != null
        self.fill = Paint.solid(color ?? Color.from_rgb(0, 0, 0))
        self.edge = edge
        self.fraction = fraction
        self.horizontal = horizontal
        self.explicit_width = width
    }

    #: Bars given as heights above a baseline, the way matplotlib's `bar` takes
    #: them.
    #:
    #: Stacking is the reason this exists. A layer of a stack knows its own
    #: height and the layer beneath it, never the sum, so building the second
    #: layer means adding the two — and doing that at the call site is exactly
    #: the arithmetic that gets skipped, leaving a `Bar` whose baseline is above
    #: its values and which therefore draws downward through the layer below.
    #:
    #: ```
    #: let first = Bar([0.0, 1.0], [3.0, 1.0])
    #: let second = Bar.from_heights([0.0, 1.0], [2.0, 4.0], baseline: first.tops())
    #: second.tops()          # [5.0, 5.0]
    #: ```
    pub static fn from_heights(
        centres: list[float],
        heights: list[float],
        label: str? = null,
        color: Color? = null,
        baseline: any = 0.0,
        width: float? = null,
        fraction: float = DEFAULT_FILL_FRACTION,
        horizontal: bool = false,
        edge: Stroke? = null
    ) -> Bar {
        if centres.len() != heights.len() {
            throw ValueError(
                f"centres and heights must match, got {centres.len()} and {heights.len()}"
            )
        }
        let bottoms = baselines_over(baseline, centres.len(), "bars")
        let tops = [bottoms[index] + height for (index, height) in heights.enumerate()]
        return Bar(
            centres,
            tops,
            label: label,
            color: color,
            baseline: bottoms,
            width: width,
            fraction: fraction,
            horizontal: horizontal,
            edge: edge
        )
    }

    #: The top of each bar, in data units.
    #:
    #: The list a second layer of a stack takes as its baseline.
    pub fn tops(self) -> list[float] {
        return [value for value in self.values]
    }

    #: The baseline under each bar, in data units, one entry per bar.
    pub fn baselines(self) -> list[float] {
        return [value for value in self.bottoms]
    }

    #: Override the colour, after construction.
    pub fn color(self, value: Color) -> Bar {
        self.fill = Paint.solid(value)
        self.chosen = true
        return self
    }

    #: The box the bars occupy, including the baseline and the bar widths.
    #:
    #: The baseline is included because a bar chart that does not show it
    #: misrepresents the ratios it exists to display. Half a bar width is added
    #: at each end so the outermost bars are not clipped in half by the axis.
    pub fn extent(self) -> Rect? {
        var points = []
        let half = self._width() * 0.5
        for (index, centre) in self.centres.enumerate() {
            let value = self.values[index]
            let base = self.bottoms[index]
            if not math.is_finite(centre) or not math.is_finite(value) { continue }
            if not math.is_finite(base) { continue }
            let low = min(value, base)
            let high = max(value, base)
            if self.horizontal {
                points.push(Point(low, centre - half))
                points.push(Point(high, centre + half))
            } else {
                points.push(Point(centre - half, low))
                points.push(Point(centre + half, high))
            }
        }
        return Rect.from_points(points)
    }

    #: Replace the bar values, keeping the centres and the styling.
    #:
    #: Raises `ValueError` when the count does not match the centres, since a
    #: bar with no centre has nowhere to stand.
    pub fn set_values(self, values: list[float]) -> void {
        if values.len() != self.centres.len() {
            throw ValueError(
                f"expected {self.centres.len()} values to match the centres, got {values.len()}"
            )
        }
        self.values = [value for value in values]
    }

    #: Move the baseline, as one number or one per bar.
    #:
    #: Raises `ValueError` on a list of the wrong length.
    pub fn set_baseline(self, baseline: any) -> void {
        self.bottoms = baselines_over(baseline, self.centres.len(), "bars")
    }

    #: One filled rectangle per bar whose corners the frame can place.
    #:
    #: A bar whose *top* the value axis rejects is dropped: there is no height
    #: to show and matplotlib draws nothing for it either. A baseline the axis
    #: rejects is clamped instead — see `clamp_to_axis`, which is the difference
    #: between a log bar chart and a blank panel.
    pub fn draw(self, frame: Frame) -> list[Primitive] {
        var out = []
        let half = self._width() * 0.5
        let (value_scale, value_low) = axis_of(frame, not self.horizontal)
        let (category_scale, category_low) = axis_of(frame, self.horizontal)
        for (index, centre) in self.centres.enumerate() {
            let value = self.values[index]
            let base = self.bottoms[index]
            if not math.is_finite(centre) or not math.is_finite(base) { continue }
            if not value_scale.accepts(value) { continue }
            let floor = clamp_to_axis(value_scale, base, value_low)
            let left = clamp_to_axis(category_scale, centre - half, category_low)
            let right = clamp_to_axis(category_scale, centre + half, category_low)
            let near = self._corner(frame, left, floor)
            let far = self._corner(frame, right, value)
            if near == null or far == null { continue }
            let corners = [near, Point(far.x, near.y), far, Point(near.x, far.y)]
            out.push(Primitive.Polygon(corners, self.fill, self.edge))
        }
        return out
    }

    #: The legend entry, or null when the series was not labelled.
    pub fn label(self) -> str? {
        return self.name
    }

    #: The swatch box filled in the bar colour.
    pub fn swatch(self, box: Rect) -> list[Primitive] {
        let corners = [
            Point(box.left(), box.bottom()),
            Point(box.right(), box.bottom()),
            Point(box.right(), box.top()),
            Point(box.left(), box.top()),
        ]
        return [Primitive.Polygon(corners, self.fill, self.edge)]
    }

    #: Take the cycle's colour unless one was chosen explicitly.
    pub fn adopt(self, color: Color) -> void {
        if not self.chosen { self.fill = Paint.solid(color) }
    }

    #: Place one corner, swapping the axes for a horizontal chart.
    fn _corner(self, frame: Frame, along: float, across: float) -> Point? {
        if self.horizontal { return frame.place(across, along) }
        return frame.place(along, across)
    }

    #: The width of one bar in data units along the category axis.
    #:
    #: Taken from the tightest gap between neighbouring centres, so no pair
    #: overlaps. A single bar has no neighbour to measure against and gets one
    #: unit, which is the only defensible guess.
    fn _width(self) -> float {
        let given = self.explicit_width
        if given != null { return given }
        var sorted = [value for value in self.centres if math.is_finite(value)]
        if sorted.len() < 2 { return self.fraction }
        sorted.sort()
        var tightest = sorted[sorted.len() - 1] - sorted[0]
        for index in 1..sorted.len() {
            let gap = sorted[index] - sorted[index - 1]
            if gap > 0.0 and gap < tightest { tightest = gap }
        }
        if tightest <= 0.0 { return self.fraction }
        return tightest * self.fraction
    }

    fn __str__(self) -> str {
        return f"Bar({self.centres.len()} bars, {self.name ?? "unlabelled"})"
    }
}
