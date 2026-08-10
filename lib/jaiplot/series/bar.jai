#: A series drawn as rectangles rising from a baseline.
import std.math

from .base import Series
from ..frame import Frame
from ..geom import Point, Rect
from ..primitive import Primitive
from ..style import Color, Paint, Stroke

#: Fraction of the slot between neighbouring bars that a bar fills.
#:
#: The remainder is the gap. Eight tenths reads as separate bars while still
#: making the group obvious.
pub const DEFAULT_FILL_FRACTION: float = 0.8

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
#: ```
#: let bars = Bar([0.0, 1.0, 2.0], [3.0, 1.0, 4.0], label: "counts")
#: bars.extent().top()          # 4.0
#: bars.extent().bottom()       # 0.0
#: ```
pub class Bar: Series {
    var centres: list[float]
    var values: list[float]
    var name: str?
    var fill: Paint
    var edge: Stroke?
    var chosen: bool
    var baseline: float
    var fraction: float
    var horizontal: bool
    var explicit_width: float?

    #: Draw a bar at each centre, reaching from the baseline to its value.
    pub fn init(
        self,
        centres: list[float],
        values: list[float],
        label: str? = null,
        color: Color? = null,
        baseline: float = 0.0,
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
        if not math.is_finite(baseline) {
            throw ValueError(f"the baseline must be finite, got {baseline}")
        }
        self.centres = [value for value in centres]
        self.values = [value for value in values]
        self.name = label
        self.chosen = color != null
        self.fill = Paint.solid(color ?? Color.from_rgb(0, 0, 0))
        self.edge = edge
        self.baseline = baseline
        self.fraction = fraction
        self.horizontal = horizontal
        self.explicit_width = width
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
            if not math.is_finite(centre) or not math.is_finite(value) { continue }
            let low = min(value, self.baseline)
            let high = max(value, self.baseline)
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

    #: One filled rectangle per bar whose corners the frame can place.
    pub fn draw(self, frame: Frame) -> list[Primitive] {
        var out = []
        let half = self._width() * 0.5
        for (index, centre) in self.centres.enumerate() {
            let value = self.values[index]
            if not math.is_finite(centre) or not math.is_finite(value) { continue }
            let near = self._corner(frame, centre - half, self.baseline)
            let far = self._corner(frame, centre + half, value)
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
