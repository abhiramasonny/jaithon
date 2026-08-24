#: A distribution, binned and drawn as adjacent rectangles.
import std.math
from std.algo.stat import freedman_diaconis_bins, min_max

from .bar import axis_of, baselines_over, clamp_to_axis
from .base import Series
from .line import DEFAULT_WIDTH
from ..frame import Frame
from ..geom import Point, Rect
from ..primitive import Primitive
from ..scale import Scale
from ..style import Color, Paint, Stroke

#: The counts of a sample, drawn as touching bars over the bin edges.
#:
#: The bins are computed once, at construction, so `extent` and `draw` cannot
#: disagree about where the edges are. Bars touch rather than leaving gaps,
#: because the bins are contiguous and a gap would suggest they are not.
#:
#: `bins` is a count or the edges themselves; `span` is matplotlib's `range=`,
#: spelled differently because `range` is a builtin here. Between them they are
#: what makes two histograms comparable: two samples binned over their own
#: extents share no edge and cannot be stacked or subtracted.
#:
#: ```
#: let counts = Hist([1.0, 1.5, 2.0, 2.5, 3.0], bins: 2, label: "sample")
#: counts.bins()                # 2
#: counts.extent().bottom()     # 0.0
#: ```
pub class Hist: Series {
    var edges: list[float]
    var tallies: list[float]
    var bottoms: list[float]
    var name: str?
    var fill: Paint
    var edge: Stroke?
    var chosen: bool
    var density: bool
    var cumulative: bool
    var step: bool
    var horizontal: bool

    #: Bin `values` and hold the result.
    #:
    #: `bins` defaults to the count `std.algo.stat.freedman_diaconis_bins`
    #: chooses, takes an integer count, or takes the edges themselves as a list
    #: of at least two increasing numbers. `span` confines the bins to a range
    #: and, unlike the default, discards the samples outside it — which is the
    #: point of asking for one.
    #:
    #: `weights` gives each sample a size other than one, so a histogram can
    #: total money or exposure rather than occurrences. `density` scales the
    #: bars so their total area is one, which is what makes two samples of
    #: different sizes comparable, and `cumulative` accumulates them left to
    #: right — with `density` as well, the last bar reaches exactly 1.0.
    #:
    #: `baseline` lifts the bars off zero, as one number or one per bin, and is
    #: how a stack is built: give the second histogram the first's `edges_of()`
    #: and its `tops()`. `step` draws the outline only, matplotlib's
    #: `histtype='step'`, which is how several distributions overlay without
    #: burying each other.
    #:
    #: Raises `ValueError` on an empty sample, since there is no distribution to
    #: show and a silently empty chart hides the mistake.
    pub fn init(
        self,
        values: list[float],
        bins: any? = null,
        label: str? = null,
        color: Color? = null,
        density: bool = false,
        edge: Stroke? = null,
        span: tuple[float, float]? = null,
        weights: list[float]? = null,
        cumulative: bool = false,
        step: bool = false,
        horizontal: bool = false,
        baseline: any = 0.0
    ) {
        if weights != null and weights.len() != values.len() {
            throw ValueError(
                f"expected {values.len()} weights to match the sample, got {weights.len()}"
            )
        }
        var usable = []
        var sizes = []
        for (index, value) in values.enumerate() {
            if not math.is_finite(value) { continue }
            let weight = if weights == null { 1.0 } else { weights[index] }
            if not math.is_finite(weight) {
                throw ValueError(f"weights must be finite, got {weight}")
            }
            usable.push(value)
            sizes.push(weight)
        }
        if usable.len() == 0 {
            throw ValueError(f"a histogram needs at least one finite value, got {values.len()}")
        }
        self.edges = _edges_for(usable, bins, span)
        self.tallies = _tally(self.edges, usable, sizes)
        self.bottoms = baselines_over(baseline, self.tallies.len(), "bins")
        self.name = label
        self.chosen = color != null
        self.fill = Paint.solid(color ?? Color.from_rgb(0, 0, 0))
        self.edge = edge
        self.density = density
        self.cumulative = cumulative
        self.step = step
        self.horizontal = horizontal
    }

    #: How many bins the sample was divided into.
    pub fn bins(self) -> int {
        return self.tallies.len()
    }

    #: The bin edges, one more than there are bins.
    pub fn edges_of(self) -> list[float] {
        return [value for value in self.edges]
    }

    #: What landed in each bin: the sample count, or the weights' total.
    #:
    #: Untouched by `density` and `cumulative`, both of which are presentation.
    pub fn counts(self) -> list[float] {
        return [value for value in self.tallies]
    }

    #: The height of each bar, counts or densities as configured.
    pub fn heights(self) -> list[float] {
        var total = 0.0
        for weight in self.tallies { total += weight }
        if self.cumulative {
            #: A cumulative density is the distribution function, so the bars
            #: accumulate the share of the sample in each bin and the last one
            #: reaches 1.0 — dividing by the bin width first, as the
            #: non-cumulative case does, would make the total depend on how
            #: wide the bins happen to be.
            var running = 0.0
            var out = []
            for weight in self.tallies {
                running += if self.density and total > 0.0 { weight / total } else { weight }
                out.push(running)
            }
            return out
        }
        if not self.density { return [weight for weight in self.tallies] }
        if total == 0.0 { return [0.0 for _weight in self.tallies] }
        var out = []
        for (index, weight) in self.tallies.enumerate() {
            let width = self.edges[index + 1] - self.edges[index]
            if width <= 0.0 {
                out.push(0.0)
            } else {
                out.push(weight / (total * width))
            }
        }
        return out
    }

    #: The top of each bar: its baseline plus its height.
    #:
    #: The list a second histogram of a stack takes as its baseline.
    pub fn tops(self) -> list[float] {
        let heights = self.heights()
        return [self.bottoms[index] + height for (index, height) in heights.enumerate()]
    }

    #: The baseline under each bar, one entry per bin.
    pub fn baselines(self) -> list[float] {
        return [value for value in self.bottoms]
    }

    #: Override the colour, after construction.
    pub fn color(self, value: Color) -> Hist {
        self.fill = Paint.solid(value)
        self.chosen = true
        return self
    }

    #: The full span of the bins, from the baseline up to the tallest bar.
    #:
    #: The baseline is included because a histogram drawn from a raised
    #: baseline misrepresents the ratios between its bars, and it is zero unless
    #: the caller stacked this histogram on another.
    pub fn extent(self) -> Rect? {
        if self.tallies.len() == 0 { return null }
        let tops = self.tops()
        var lowest = 0.0
        var highest = 0.0
        var seen = false
        for (index, top) in tops.enumerate() {
            let base = self.bottoms[index]
            if not math.is_finite(top) or not math.is_finite(base) { continue }
            let low = min(top, base)
            let high = max(top, base)
            lowest = if seen { min(lowest, low) } else { low }
            highest = if seen { max(highest, high) } else { high }
            seen = true
        }
        if not seen { return null }
        let last = self.edges.len() - 1
        if self.horizontal { return Rect(lowest, self.edges[0], highest, self.edges[last]) }
        return Rect(self.edges[0], lowest, self.edges[last], highest)
    }

    #: One filled rectangle per bin, sharing edges with its neighbours, or one
    #: staircase outline when `step` was asked for.
    #:
    #: A bar whose top the value axis rejects is dropped, because an empty bin
    #: on a logarithmic axis has no height to show. The outline clamps that same
    #: point to the foot of the axis instead: a staircase with a rung missing is
    #: no longer a staircase, and running down to the bottom of the panel is
    #: what matplotlib's step outline does there too.
    pub fn draw(self, frame: Frame) -> list[Primitive] {
        let (value_scale, value_low) = axis_of(frame, not self.horizontal)
        let (category_scale, category_low) = axis_of(frame, self.horizontal)
        let heights = self.heights()
        if self.step {
            return self._outline(
                frame,
                heights,
                value_scale,
                value_low,
                category_scale,
                category_low
            )
        }
        var out = []
        for (index, height) in heights.enumerate() {
            let base = self.bottoms[index]
            let top = base + height
            if not math.is_finite(base) or not math.is_finite(top) { continue }
            if not value_scale.accepts(top) { continue }
            let floor = clamp_to_axis(value_scale, base, value_low)
            let start = clamp_to_axis(category_scale, self.edges[index], category_low)
            let stop = clamp_to_axis(category_scale, self.edges[index + 1], category_low)
            let near = self._corner(frame, start, floor)
            let far = self._corner(frame, stop, top)
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

    #: The swatch box filled in the histogram's colour, or ruled for an outline.
    #:
    #: An outlined histogram drawn with a filled swatch reads as a filled one,
    #: which is the whole distinction the caller asked for by choosing `step`.
    pub fn swatch(self, box: Rect) -> list[Primitive] {
        if self.step {
            let middle = box.center().y
            let ends = [Point(box.left(), middle), Point(box.right(), middle)]
            return [Primitive.Polyline(ends, self._stroke())]
        }
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

    #: The staircase over the bin tops, dropping to the baseline at both ends.
    #:
    #: Open rather than closed: matplotlib leaves the segment along the bottom
    #: out, and drawing it would put a rule under every overlaid distribution
    #: where the reader expects the axis.
    fn _outline(
        self,
        frame: Frame,
        heights: list[float],
        value_scale: Scale,
        value_low: float,
        category_scale: Scale,
        category_low: float
    ) -> list[Primitive] {
        let count = heights.len()
        if count == 0 { return [] }
        var alongs = [self.edges[0]]
        var acrosses = [self.bottoms[0]]
        for (index, height) in heights.enumerate() {
            let top = self.bottoms[index] + height
            alongs.push(self.edges[index])
            acrosses.push(top)
            alongs.push(self.edges[index + 1])
            acrosses.push(top)
        }
        alongs.push(self.edges[count])
        acrosses.push(self.bottoms[count - 1])
        var points = []
        for (index, along) in alongs.enumerate() {
            let across = acrosses[index]
            if not math.is_finite(along) or not math.is_finite(across) { continue }
            let placed = self._corner(
                frame,
                clamp_to_axis(category_scale, along, category_low),
                clamp_to_axis(value_scale, across, value_low)
            )
            if placed == null { continue }
            points.push(placed)
        }
        if points.len() < 2 { return [] }
        return [Primitive.Polyline(points, self._stroke())]
    }

    #: The stroke an outlined histogram is drawn in.
    #:
    #: Built at draw time from the fill colour rather than held as a field, so
    #: that an outline left to default follows the series through `adopt`
    #: instead of staying the black the constructor started from.
    fn _stroke(self) -> Stroke {
        return self.edge ?? Stroke(self.fill.color(), width: DEFAULT_WIDTH)
    }

    #: Place one corner, swapping the axes for a horizontal histogram.
    fn _corner(self, frame: Frame, along: float, across: float) -> Point? {
        if self.horizontal { return frame.place(across, along) }
        return frame.place(along, across)
    }

    fn __str__(self) -> str {
        return f"Hist({self.tallies.len()} bins, {self.name ?? "unlabelled"})"
    }
}

#: The bin edges for a sample, from a count, a range, or the edges themselves.
#:
#: `std.algo.stat.histogram` is not called even though it bins: it takes neither
#: a range nor weights nor explicit edges, so the options would have arrived as
#: a second binner alongside it and two rules for where an edge falls is how
#: `extent` and `draw` come to disagree. The rule here is that one — the same
#: uniform edges over the sample's own extent, and the same unit-wide bin for a
#: sample of identical values — and a test pins the two together. The default
#: count still comes from `freedman_diaconis_bins`, which is the part worth
#: sharing.
fn _edges_for(values: list[float], bins: any?, span: tuple[float, float]?) -> list[float] {
    if bins is list {
        let given = [float(edge) for edge in bins]
        if given.len() < 2 {
            throw ValueError(f"explicit bin edges need at least two entries, got {given.len()}")
        }
        for (index, edge) in given.enumerate() {
            if not math.is_finite(edge) {
                throw ValueError(f"bin edges must be finite, got {edge}")
            }
            if index > 0 and edge <= given[index - 1] {
                throw ValueError(f"bin edges must increase, got {given[index - 1]} then {edge}")
            }
        }
        return given
    }
    let count = if bins == null { freedman_diaconis_bins(values) } else { int(bins) }
    if count < 1 { throw ValueError(f"bins must be at least 1, got {count}") }
    var (start, stop) = min_max(values)
    if span != null {
        let (low, high) = span
        if not math.is_finite(low) or not math.is_finite(high) or low >= high {
            throw ValueError(
                f"a histogram range must be finite and increasing, got {low} to {high}"
            )
        }
        start = low
        stop = high
    } elif start == stop {
        start -= 0.5
        stop += 0.5
    }
    let width = (stop - start) / float(count)
    return [start + width * float(index) for index in 0..=count]
}

#: Total the weight of every sample that falls inside the edges.
#:
#: A sample outside them is discarded rather than pulled into the nearest bin:
#: with the default edges nothing is outside, and when a caller has named a
#: range, folding the outliers into the end bins would put a spike there that
#: the data does not have.
fn _tally(edges: list[float], values: list[float], weights: list[float]) -> list[float] {
    var totals = [0.0 for _index in 0..edges.len() - 1]
    for (index, value) in values.enumerate() {
        let bin = _bin_of(edges, value)
        if bin == null { continue }
        totals[bin] += weights[index]
    }
    return totals
}

#: The bin `value` falls in, or null when it falls outside the edges.
#:
#: The last bin is closed at both ends, so the largest sample lands inside it
#: rather than in a bin of its own. Bisection rather than arithmetic, because
#: explicit edges need not be evenly spaced.
fn _bin_of(edges: list[float], value: float) -> int? {
    let last = edges.len() - 1
    if value < edges[0] or value > edges[last] { return null }
    if value >= edges[last] { return last - 1 }
    var low = 0
    var high = last
    while high - low > 1 {
        let middle = (low + high) // 2
        if value < edges[middle] { high = middle } else { low = middle }
    }
    return low
}
