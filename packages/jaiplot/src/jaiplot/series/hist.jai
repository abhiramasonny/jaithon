#: A distribution, binned and drawn as adjacent rectangles.
import std.math
from std.algo.stat import histogram

from .base import Series
from ..frame import Frame
from ..geom import Point, Rect
from ..primitive import Primitive
from ..style import Color, Paint, Stroke

#: The counts of a sample, drawn as touching bars over the bin edges.
#:
#: The bins are computed once, at construction, so `extent` and `draw` cannot
#: disagree about where the edges are. Bars touch rather than leaving gaps,
#: because the bins are contiguous and a gap would suggest they are not.
#:
#: ```
#: let counts = Hist([1.0, 1.5, 2.0, 2.5, 3.0], bins: 2, label: "sample")
#: counts.bins()                # 2
#: counts.extent().bottom()     # 0.0
#: ```
pub class Hist: Series {
    var edges: list[float]
    var counts: list[int]
    var name: str?
    var fill: Paint
    var edge: Stroke?
    var chosen: bool
    var density: bool

    #: Bin `values` and hold the result.
    #:
    #: `bins` defaults to the rule `std.algo.stat.histogram` chooses. With
    #: `density` set the bars are scaled so their total area is one, which is
    #: what makes two samples of different sizes comparable.
    #:
    #: Raises `ValueError` on an empty sample, since there is no distribution to
    #: show and a silently empty chart hides the mistake.
    pub fn init(
        self,
        values: list[float],
        bins: int? = null,
        label: str? = null,
        color: Color? = null,
        density: bool = false,
        edge: Stroke? = null
    ) {
        let usable = [value for value in values if math.is_finite(value)]
        if usable.len() == 0 {
            throw ValueError(f"a histogram needs at least one finite value, got {values.len()}")
        }
        let (edges, counts) = histogram(usable, bins)
        self.edges = edges
        self.counts = counts
        self.name = label
        self.chosen = color != null
        self.fill = Paint.solid(color ?? Color.from_rgb(0, 0, 0))
        self.edge = edge
        self.density = density
    }

    #: How many bins the sample was divided into.
    pub fn bins(self) -> int {
        return self.counts.len()
    }

    #: The bin edges, one more than there are bins.
    pub fn edges_of(self) -> list[float] {
        return [value for value in self.edges]
    }

    #: The height of each bar, counts or densities as configured.
    pub fn heights(self) -> list[float] {
        if not self.density { return [count.to_float() for count in self.counts] }
        var total = 0
        for count in self.counts { total += count }
        if total == 0 { return [0.0 for _count in self.counts] }
        var out = []
        for (index, count) in self.counts.enumerate() {
            let span = self.edges[index + 1] - self.edges[index]
            if span <= 0.0 {
                out.push(0.0)
            } else {
                out.push(count.to_float() / (total.to_float() * span))
            }
        }
        return out
    }

    #: Override the colour, after construction.
    pub fn color(self, value: Color) -> Hist {
        self.fill = Paint.solid(value)
        self.chosen = true
        return self
    }

    #: The full span of the bins, from zero up to the tallest bar.
    #:
    #: Zero is included because a histogram drawn from a raised baseline
    #: misrepresents the ratios between its bars.
    pub fn extent(self) -> Rect? {
        if self.counts.len() == 0 { return null }
        let heights = self.heights()
        var tallest = 0.0
        for height in heights { tallest = max(tallest, height) }
        return Rect(self.edges[0], 0.0, self.edges[self.edges.len() - 1], tallest)
    }

    #: One filled rectangle per bin, sharing edges with its neighbours.
    pub fn draw(self, frame: Frame) -> list[Primitive] {
        var out = []
        let heights = self.heights()
        for (index, height) in heights.enumerate() {
            let base = frame.place(self.edges[index], 0.0)
            let top = frame.place(self.edges[index + 1], height)
            if base == null or top == null { continue }
            let corners = [base, Point(top.x, base.y), top, Point(base.x, top.y)]
            out.push(Primitive.Polygon(corners, self.fill, self.edge))
        }
        return out
    }

    #: The legend entry, or null when the series was not labelled.
    pub fn label(self) -> str? {
        return self.name
    }

    #: The swatch box filled in the histogram's colour.
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

    fn __str__(self) -> str {
        return f"Hist({self.counts.len()} bins, {self.name ?? "unlabelled"})"
    }
}
