#: One set of axes: the data, the limits, and everything drawn around them.
#:
#: `Axes` is where a figure's decisions are made. It holds the series, decides
#: the limits, asks the scales where the ticks go, measures the labels those
#: ticks need, and only then knows how much room is left for the data. The
#: result is a `DisplayList`; nothing here draws.
#:
#: ```
#: from jaiplot.geom import Rect
#: let axes = Axes()
#: axes.line([0.0, 1.0, 2.0], [0.0, 1.0, 4.0], label: "squares")
#: axes.set_title("a title")
#: axes.render(Rect(0.0, 0.0, 400.0, 300.0)).len() > 0     # true
#: ```
import std.math

from .frame import Frame
from .geom import Point, Rect
from .legend import LegendCorner, draw_legend
from .metrics import text_height, text_width
from .primitive import DisplayList, Primitive
from .scale import LinearScale, Scale, scale_named
from .series import Bar, FillBetween, Grid, Hist, Line, PoseCloud, Rays, Scatter, Series
from .style import Color, HorizontalAnchor, MarkerShape, Paint, Stroke
from .style import TextStyle, Theme, VerticalAnchor

#: How many ticks an axis aims for before rounding to nice numbers.
#:
#: A ceiling, not a promise. A panel too narrow to write this many labels side
#: by side is thinned down until they clear each other, so the number only ever
#: describes a panel with room to spare.
pub const TICK_HINT = 6

#: Fewest ticks thinning will leave on an axis.
#:
#: Two, because an axis showing one number says nothing about the direction it
#: runs in, and `linear_ticks` will not accept a hint below two anyway.
pub const MIN_TICK_HINT = 2

#: Clear space demanded between two neighbouring tick labels.
#:
#: Labels that merely touch read as one long number -- `0.00081.0002` is what a
#: 200px panel produced before this existed -- so the gap has to be wide enough
#: to be seen as a gap. Six pixels is one glyph cell.
pub const LABEL_CLEARANCE: float = 6.0

#: Length of a tick mark, in device pixels.
pub const TICK_LENGTH: float = 5.0

#: Length of an unlabelled minor tick.
#:
#: Matplotlib's ratio, 2 points against 3.5: short enough that the labelled
#: ticks still read as the ones that matter.
pub const MINOR_TICK_LENGTH: float = 3.0

#: Opacity a minor gridline is drawn at, against a major one at full strength.
#:
#: A log decade carries eight minor rules. At the grid colour's full weight they
#: merge into a grey band near the top of each decade, which is the thing a
#: gridline is supposed to help a reader see past.
pub const MINOR_GRID_ALPHA: float = 0.45

#: Gap between a tick mark and its label.
pub const LABEL_GAP: float = 4.0

#: The sets of gridlines `Axes.grid` will draw.
pub let GRID_WHICH: list[str] = ["major", "minor", "both"]

#: Padding at the edges of the figure that no content may enter.
pub const FIGURE_PADDING: float = 10.0

#: Fraction of the data range left as breathing room when ranging automatically.
#:
#: matplotlib's `axes.xmargin` and `axes.ymargin`, and the same 0.05. Without it
#: a scatter over 0..1 ranges to exactly 0..1 and the clip round the panel takes
#: three quarters of the marker standing in each corner.
pub const AUTO_PAD: float = 0.05

#: A set of axes and the series drawn on them.
pub class Axes {
    var layers: list[Series]
    var x_scale: Scale
    var y_scale: Scale
    var x_low: float?
    var x_high: float?
    var y_low: float?
    var y_high: float?
    var title_text: str?
    var x_text: str?
    var y_text: str?
    var grid_on: bool
    var grid_which: str
    var minor_wanted: bool?
    var x_fixed: list[float]?
    var y_fixed: list[float]?
    var x_names: list[str]?
    var y_names: list[str]?
    var x_hint: int
    var y_hint: int
    var spine_left: bool
    var spine_right: bool
    var spine_top: bool
    var spine_bottom: bool
    var legend_on: bool
    var legend_corner: LegendCorner
    var theme: Theme
    var taken: int
    var equal_aspect: bool
    var frozen: int

    #: Create empty axes in `theme`.
    pub fn init(self, theme: Theme? = null) {
        self.layers = []
        self.x_scale = LinearScale()
        self.y_scale = LinearScale()
        self.x_low = null
        self.x_high = null
        self.y_low = null
        self.y_high = null
        self.title_text = null
        self.x_text = null
        self.y_text = null
        self.grid_on = false
        self.grid_which = "major"
        self.minor_wanted = null
        self.x_fixed = null
        self.y_fixed = null
        self.x_names = null
        self.y_names = null
        self.x_hint = TICK_HINT
        self.y_hint = TICK_HINT
        self.spine_left = true
        self.spine_right = true
        self.spine_top = true
        self.spine_bottom = true
        self.legend_on = false
        self.legend_corner = LegendCorner.TopRight
        self.theme = theme ?? Theme.light()
        self.taken = 0
        self.equal_aspect = false
        self.frozen = 0
    }

    #: Add a series, giving it the next colour from the theme's cycle.
    pub fn add(self, series: Series) -> Series {
        series.adopt(self.theme.cycle.at(self.taken))
        self.taken += 1
        self.layers.push(series)
        return series
    }

    #: Plot `ys` against `xs` as a line, and return it for restyling.
    pub fn line(
        self,
        xs: list[float],
        ys: list[float],
        label: str? = null,
        color: Color? = null,
        width: float = 1.75,
        dash: list[float] = [],
        marker: MarkerShape? = null
    ) -> Line {
        let series = Line(
            xs,
            ys,
            label: label,
            color: color,
            width: width,
            dash: dash,
            marker: marker
        )
        let _added = self.add(series)
        return series
    }

    #: Plot `ys` against `xs` as markers, and return the series.
    pub fn scatter(
        self,
        xs: list[float],
        ys: list[float],
        label: str? = null,
        color: Color? = null,
        size: float = 6.0,
        shape: MarkerShape = MarkerShape.Circle
    ) -> Scatter {
        let series = Scatter(xs, ys, label: label, color: color, size: size, shape: shape)
        let _added = self.add(series)
        return series
    }

    #: Draw bars at `centres` reaching to `values`, and return the series.
    pub fn bar(
        self,
        centres: list[float],
        values: list[float],
        label: str? = null,
        color: Color? = null,
        baseline: float = 0.0,
        horizontal: bool = false
    ) -> Bar {
        let series = Bar(
            centres,
            values,
            label: label,
            color: color,
            baseline: baseline,
            horizontal: horizontal
        )
        let _added = self.add(series)
        return series
    }

    #: Bin `values` and draw the distribution, returning the series.
    pub fn hist(
        self,
        values: list[float],
        bins: int? = null,
        label: str? = null,
        color: Color? = null,
        density: bool = false
    ) -> Hist {
        let series = Hist(values, bins: bins, label: label, color: color, density: density)
        let _added = self.add(series)
        return series
    }

    #: Fill the band between `lower` and `upper`, returning the series.
    pub fn fill_between(
        self,
        xs: list[float],
        lower: list[float],
        upper: list[float],
        label: str? = null,
        color: Color? = null,
        opacity: float = 0.28
    ) -> FillBetween {
        let series = FillBetween(xs, lower, upper, label: label, color: color, opacity: opacity)
        let _added = self.add(series)
        return series
    }

    #: Draw a cloud of poses, and return it for restyling.
    pub fn poses(
        self,
        xs: list[float],
        ys: list[float],
        headings: list[float] = [],
        weights: list[float] = [],
        label: str? = null,
        color: Color? = null,
        size: float = 3.5
    ) -> PoseCloud {
        let series = PoseCloud(
            xs,
            ys,
            headings: headings,
            weights: weights,
            label: label,
            color: color,
            size: size
        )
        let _added = self.add(series)
        return series
    }

    #: Draw a bundle of rays, and return it for restyling.
    pub fn rays(
        self,
        origin_xs: list[float],
        origin_ys: list[float],
        end_xs: list[float],
        end_ys: list[float],
        label: str? = null,
        color: Color? = null,
        width: float = 1.0
    ) -> Rays {
        let series = Rays(
            origin_xs,
            origin_ys,
            end_xs,
            end_ys,
            label: label,
            color: color,
            width: width
        )
        let _added = self.add(series)
        return series
    }

    #: Draw a grid of values as an image, and return it for restyling.
    pub fn grid_of(
        self,
        cells: list[float],
        columns: int,
        rows: int,
        box: Rect,
        label: str? = null,
        low_color: Color? = null,
        high_color: Color? = null
    ) -> Grid {
        let series = Grid(
            cells,
            columns,
            rows,
            box,
            label: label,
            low_color: low_color,
            high_color: high_color
        )
        let _added = self.add(series)
        return series
    }

    #: Give one data unit the same length on both axes.
    #:
    #: A map drawn without this is stretched to the panel, so a square room
    #: comes out oblong and a circular scan elliptical. The limits are widened —
    #: never narrowed — on whichever axis has room, so nothing that was in view
    #: leaves it.
    pub fn set_aspect(self, equal: bool = true) -> Axes {
        self.equal_aspect = equal
        return self
    }

    #: Set the heading drawn above the axes.
    pub fn set_title(self, text: str) -> Axes {
        self.title_text = text
        return self
    }

    #: Set the label drawn under the horizontal axis.
    pub fn set_xlabel(self, text: str) -> Axes {
        self.x_text = text
        return self
    }

    #: Set the label drawn beside the vertical axis, turned to read upward.
    pub fn set_ylabel(self, text: str) -> Axes {
        self.y_text = text
        return self
    }

    #: Fix the horizontal limits, overriding automatic ranging.
    pub fn set_xlim(self, low: float, high: float) -> Axes {
        if low >= high {
            throw ValueError(f"x limits must be ordered, got {low} to {high}")
        }
        self.x_low = low
        self.x_high = high
        return self
    }

    #: Fix the vertical limits, overriding automatic ranging.
    pub fn set_ylim(self, low: float, high: float) -> Axes {
        if low >= high {
            throw ValueError(f"y limits must be ordered, got {low} to {high}")
        }
        self.y_low = low
        self.y_high = high
        return self
    }

    #: Set the horizontal scale, by name or by instance.
    pub fn set_xscale(self, scale: any) -> Axes {
        self.x_scale = _as_scale(scale)
        return self
    }

    #: Set the vertical scale, by name or by instance.
    pub fn set_yscale(self, scale: any) -> Axes {
        self.y_scale = _as_scale(scale)
        return self
    }

    #: Turn the gridlines on or off, for one set of ticks or both.
    #:
    #: `which` is `"major"`, `"minor"` or `"both"`. Asking for the minor
    #: gridlines turns the minor positions on by itself, which is a deliberate
    #: departure from matplotlib: there, `grid(which="both")` on a linear axis
    #: draws nothing at all until `minorticks_on` has been called separately,
    #: and the empty result looks like a bug rather than a missing step. The
    #: minor tick *marks* still stay off unless asked for.
    #:
    #: Raises `ValueError` naming the three sets, since a misspelling would
    #: otherwise silently leave the minor lines off.
    pub fn grid(self, on: bool = true, which: str = "major") -> Axes {
        if not GRID_WHICH.contains(which) {
            throw ValueError(f"a grid set must be one of {GRID_WHICH.join(", ")}, got {which}")
        }
        self.grid_on = on
        self.grid_which = which
        return self
    }

    #: Draw the unlabelled ticks between the labelled ones.
    #:
    #: A logarithmic axis has these already -- the multiples 2 to 9 of each
    #: power -- and draws them without being asked. This is for a linear axis,
    #: which has no natural intermediate positions and so gets each step split
    #: into four or five.
    pub fn minorticks_on(self) -> Axes {
        self.minor_wanted = true
        return self
    }

    #: Suppress the minor ticks, including the ones a log axis draws by default.
    pub fn minorticks_off(self) -> Axes {
        self.minor_wanted = false
        return self
    }

    #: Put ticks exactly at `values`, optionally naming each one.
    #:
    #: This is how a categorical axis is written: a `Bar` series is positioned
    #: by number like everything else, and the names of the categories arrive
    #: here. Values outside the limits are dropped when the axis is drawn rather
    #: than pushed against the panel edge.
    #:
    #: Fixed ticks are never thinned. The caller asked for these positions, and
    #: an axis that quietly drew four of six named categories would be worse
    #: than one whose names touch.
    #:
    #: Raises `ValueError` when `labels` is given and is a different length.
    #:
    #: ```
    #: from jaiplot.geom import Rect
    #: let axes = Axes()
    #: axes.bar([0.0, 1.0, 2.0], [3.0, 1.0, 4.0])
    #: axes.set_xticks([0.0, 1.0, 2.0], ["north", "south", "east"])
    #: axes.render(Rect(0.0, 0.0, 400.0, 300.0)).render().contains("north")     # true
    #: ```
    pub fn set_xticks(self, values: list[float], labels: list[str]? = null) -> Axes {
        self.x_fixed = _checked_ticks(values, labels, "x")
        self.x_names = labels
        return self
    }

    #: Put ticks exactly at `values` on the vertical axis, optionally named.
    pub fn set_yticks(self, values: list[float], labels: list[str]? = null) -> Axes {
        self.y_fixed = _checked_ticks(values, labels, "y")
        self.y_names = labels
        return self
    }

    #: Name the horizontal ticks that `set_xticks` fixed.
    #:
    #: Raises `ValueError` when no positions have been fixed. Naming ticks the
    #: axis chose for itself would attach the names to positions that move the
    #: moment the limits or the panel width change, so matplotlib refuses it
    #: too.
    pub fn set_xticklabels(self, labels: list[str]) -> Axes {
        self.x_names = _checked_names(self.x_fixed, labels, "x")
        return self
    }

    #: Name the vertical ticks that `set_yticks` fixed.
    pub fn set_yticklabels(self, labels: list[str]) -> Axes {
        self.y_names = _checked_names(self.y_fixed, labels, "y")
        return self
    }

    #: Cap how many ticks each axis aims for, overriding `TICK_HINT`.
    #:
    #: A cap on what is drawn, not on how the limits are chosen: an axis that
    #: ranges itself still rounds outward to whole `TICK_HINT` ticks, so raising
    #: the count here subdivides the same span rather than moving its ends.
    #:
    #: Raises `ValueError` below `MIN_TICK_HINT`, which is the fewest the tick
    #: rule will answer for.
    pub fn set_tick_count(self, x: int? = null, y: int? = null) -> Axes {
        self.x_hint = _checked_hint(x, self.x_hint, "x")
        self.y_hint = _checked_hint(y, self.y_hint, "y")
        return self
    }

    #: Show or hide each of the four axis lines round the panel.
    #:
    #: The sides are named as a reader sees them, not as device coordinates
    #: store them. Dropping the top and the right is the commonest single piece
    #: of styling there is; the ticks and their labels are unaffected, since a
    #: tick belongs to the axis rather than to the line drawn along it.
    #:
    #: A side left as `null` keeps whatever it had.
    pub fn set_spines(
        self,
        left: bool? = null,
        right: bool? = null,
        top: bool? = null,
        bottom: bool? = null
    ) -> Axes {
        self.spine_left = left ?? self.spine_left
        self.spine_right = right ?? self.spine_right
        self.spine_top = top ?? self.spine_top
        self.spine_bottom = bottom ?? self.spine_bottom
        return self
    }

    #: Show a legend for every labelled series.
    pub fn legend(self, corner: LegendCorner = LegendCorner.TopRight) -> Axes {
        self.legend_on = true
        self.legend_corner = corner
        return self
    }

    #: The series added so far.
    pub fn series(self) -> list[Series] {
        return [layer for layer in self.layers]
    }

    #: The limits in force, after automatic ranging.
    #:
    #: Returned as `(x_low, x_high, y_low, y_high)`. Axes holding no series at
    #: all fall back to the unit square, since something has to be drawn and an
    #: empty pair of axes with no numbers on it is not useful.
    pub fn limits(self) -> tuple[float, float, float, float] {
        let box = self._data_extent()
        let (x_low, x_high) = self._axis_limits(self.x_low, self.x_high, if box == null {
            null
        } else {
            (box.left(), box.right())
        }, self.x_scale)
        let (y_low, y_high) = self._axis_limits(self.y_low, self.y_high, if box == null {
            null
        } else {
            (box.bottom(), box.top())
        }, self.y_scale)
        return (x_low, x_high, y_low, y_high)
    }

    #: The scale governing the horizontal axis.
    pub fn x_scale_of(self) -> Scale {
        return self.x_scale
    }

    #: The scale governing the vertical axis.
    pub fn y_scale_of(self) -> Scale {
        return self.y_scale
    }

    #: The rectangle the data occupies when laid out inside `device`.
    #:
    #: The same rectangle `render` computes. The window backend needs it to turn
    #: a cursor position back into data coordinates, and asking for it here is
    #: what keeps that from being a second copy of the layout rule that could
    #: drift from this one.
    pub fn plot_area(self, device: Rect) -> Rect {
        let (x_low, x_high, y_low, y_high) = self.limits()
        return self._measure(device, x_low, x_high, y_low, y_high).plot
    }

    #: The data-to-device mapping in force when laid out inside `device`.
    pub fn frame(self, device: Rect) -> Frame {
        let (x_low, x_high, y_low, y_high) = self.limits()
        let plot = self.plot_area(device)
        let (view_x_low, view_x_high, view_y_low, view_y_high) = self._squared(
            x_low,
            x_high,
            y_low,
            y_high,
            plot
        )
        return Frame(
            Rect(view_x_low, view_y_low, view_x_high, view_y_high),
            plot,
            self.x_scale,
            self.y_scale
        )
    }

    #: Lay the axes out inside `device` and produce everything that draws them.
    #:
    #: The order is deliberate: background, grid, series under a clip, spines,
    #: ticks and their labels, then the axis labels, title and legend. Later
    #: primitives paint over earlier ones, so the frame always wins against a
    #: series that runs to the edge.
    #: How many series belong to the static layer.
    #:
    #: Everything before this index is drawn once into a cached background;
    #: everything from it onward is redrawn every frame.
    pub fn freeze_at(self, count: int) -> void {
        self.frozen = max(0, min(count, self.layers.len()))
    }

    #: The index the dynamic layer starts at.
    pub fn frozen_count(self) -> int {
        return self.frozen
    }

    #: Render the static and dynamic layers separately.
    #:
    #: The static list holds the axes' own furniture and every frozen series;
    #: the dynamic list holds the rest. An animation draws the first once and
    #: the second every frame, which for a map with a filter running over it is
    #: most of the frame saved.
    #: Render only the series that were not frozen.
    #:
    #: What an animation calls every frame once its background is cached. It
    #: skips the furniture and the frozen series entirely rather than building
    #: them and discarding them.
    pub fn render_dynamic(self, device: Rect) -> DisplayList {
        return self._render_range(device, self.frozen, self.layers.len(), false)
    }

    pub fn render_layers(self, device: Rect) -> tuple[DisplayList, DisplayList] {
        let background = self._render_range(device, 0, self.frozen, true)
        let foreground = self._render_range(device, self.frozen, self.layers.len(), false)
        return (background, foreground)
    }

    pub fn render(self, device: Rect) -> DisplayList {
        return self._render_range(device, 0, self.layers.len(), true)
    }

    #: Render the furniture and the series in `[from_index, to_index)`.
    fn _render_range(
        self,
        device: Rect,
        from_index: int,
        to_index: int,
        furniture: bool
    ) -> DisplayList {
        let list = DisplayList()
        if device.is_empty() { return list }

        let (x_low, x_high, y_low, y_high) = self.limits()
        let measured = self._measure(device, x_low, x_high, y_low, y_high)
        let plot = measured.plot
        if plot.is_empty() { return list }

        let frame = Frame(Rect(x_low, y_low, x_high, y_high), plot, self.x_scale, self.y_scale)

        if furniture {
            list.push(_fill(device, self.theme.background))
            list.push(_fill(plot, self.theme.background))
            if self.grid_on { self._draw_grid(list, frame, plot, measured) }
        }

        list.push(Primitive.PushClip(plot))
        for index in from_index..to_index {
            for primitive in self.layers[index].draw(frame) { list.push(primitive) }
        }
        list.push(Primitive.PopClip)

        if not furniture { return list }

        self._draw_spines(list, plot)
        if measured.x_marks {
            self._draw_ticks(list, frame, plot, measured.x_minor, [], true, MINOR_TICK_LENGTH)
        }
        if measured.y_marks {
            self._draw_ticks(list, frame, plot, measured.y_minor, [], false, MINOR_TICK_LENGTH)
        }
        self._draw_ticks(list, frame, plot, measured.x_ticks, measured.x_labels, true)
        self._draw_ticks(list, frame, plot, measured.y_ticks, measured.y_labels, false)
        self._draw_labels(list, device, plot)

        if self.legend_on {
            let labelled = [layer for layer in self.layers if layer.label() != null]
            for primitive in draw_legend(labelled, plot, self.theme, self.legend_corner) {
                list.push(primitive)
            }
        }
        return list
    }

    #: Widen one axis so a data unit is the same length on both.
    #:
    #: Only ever widens, so the limits the caller asked for stay visible. A
    #: logarithmic axis is left alone: a "unit" there is a ratio, and equating
    #: it with a linear distance would mean nothing.
    fn _squared(
        self,
        x_low: float,
        x_high: float,
        y_low: float,
        y_high: float,
        plot: Rect
    ) -> tuple[float, float, float, float] {
        if not self.equal_aspect { return (x_low, x_high, y_low, y_high) }
        if not self.x_scale is LinearScale or not self.y_scale is LinearScale {
            return (x_low, x_high, y_low, y_high)
        }
        let x_span = x_high - x_low
        let y_span = y_high - y_low
        if x_span <= 0.0 or y_span <= 0.0 or plot.width() <= 0.0 or plot.height() <= 0.0 {
            return (x_low, x_high, y_low, y_high)
        }

        let wanted = plot.width() / plot.height()
        let have = x_span / y_span
        if math.fabs(wanted - have) < 1e-12 { return (x_low, x_high, y_low, y_high) }

        if have < wanted {
            let widened = y_span * wanted
            let middle = (x_low + x_high) * 0.5
            return (middle - widened * 0.5, middle + widened * 0.5, y_low, y_high)
        }
        let taller = x_span / wanted
        let middle = (y_low + y_high) * 0.5
        return (x_low, x_high, middle - taller * 0.5, middle + taller * 0.5)
    }

    #: The union of every series' extent, or null when nothing is drawable.
    fn _data_extent(self) -> Rect? {
        var box = null
        for layer in self.layers {
            let extent = layer.extent()
            if extent == null { continue }
            box = if box == null { extent } else { box.union(extent) }
        }
        return box
    }

    #: Resolve one axis's limits from what was set and what the data needs.
    #:
    #: Two things happen to the data's own range on the way to becoming a pair
    #: of limits. It is masked to what the scale can place, since a point the
    #: axis drops when drawing must not drag the limits about, and it is given
    #: `AUTO_PAD` of room at each end so a sample sitting on the extreme is not
    #: drawn half outside the panel.
    #:
    #: The padding is unioned with the nice bounds rather than composed with
    #: them. Rounding outward to whole ticks already leaves room whenever the
    #: data stops short of one, and padding first would push the range past that
    #: tick and round it up to the next, so an axis over 0..1 would end at 1.2
    #: with a fifth of the panel empty. Taking whichever is wider keeps the
    #: round ends where they were already generous and pads the ends where they
    #: were flush.
    #:
    #: A limit the caller fixed is returned exactly, and never padded.
    fn _axis_limits(
        self,
        low: float?,
        high: float?,
        data: tuple[float, float]?,
        scale: Scale
    ) -> tuple[float, float] {
        if low != null and high != null { return (low, high) }
        if data == null { return _fallback(scale) }

        let (found_low, found_high) = data
        let shown = _drawable(found_low, found_high, scale)
        if shown == null { return _fallback(scale) }

        let (drawn_low, drawn_high) = shown
        var lower = low ?? drawn_low
        var upper = high ?? drawn_high
        # Only a limit the caller fixed can be off the scale by this point; the
        # data's own range has been masked to what the scale accepts already.
        if not scale.accepts(lower) or not scale.accepts(upper) {
            return _fallback(scale)
        }
        if lower == upper {
            let reach = if math.fabs(lower) > 0.0 { math.fabs(lower) * 0.5 } else { 1.0 }
            lower -= reach
            upper += reach
            if not scale.accepts(lower) { lower = upper * 0.1 }
        }
        let (nice_low, nice_high) = scale.nice(lower, upper, TICK_HINT)
        let (pad_low, pad_high) = _padded(lower, upper, scale)
        return (if low == null {
            min(nice_low, pad_low)
        } else {
            low
        }, if high == null {
            max(nice_high, pad_high)
        } else {
            high
        })
    }

    #: Choose the ticks, name them, and fit the panel round what they need.
    #:
    #: The order is forced by what depends on what. A y label's height is the
    #: same whatever it says, so the vertical thinning can be settled from the
    #: device rectangle alone; its width is not, so the panel's left edge is
    #: only known once the y labels are. Only then is there a width to thin the
    #: x labels against, and thinning them can shorten them, so the panel is
    #: measured once more afterwards.
    fn _measure(
        self,
        device: Rect,
        x_low: float,
        x_high: float,
        y_low: float,
        y_high: float
    ) -> _Furniture {
        let y_room = device.height() - self._vertical_reserve()
        let y_hint = _fitted(y_room, text_height("0", 1) + LABEL_CLEARANCE, self.y_hint)
        let (y_ticks, y_labels) = self._axis_ticks(
            self.y_scale,
            self.y_fixed,
            self.y_names,
            y_low,
            y_high,
            y_hint
        )

        var x_hint = self.x_hint
        var (x_ticks, x_labels) = self._axis_ticks(
            self.x_scale,
            self.x_fixed,
            self.x_names,
            x_low,
            x_high,
            x_hint
        )
        var plot = self._plot_area(
            device,
            x_labels,
            y_labels,
            _fractions(self.x_scale, x_ticks, x_low, x_high)
        )

        if self.x_fixed == null {
            let fitted = _fitted(plot.width(), _widest(x_labels) + LABEL_CLEARANCE, x_hint)
            if fitted < x_hint {
                x_hint = fitted
                (x_ticks, x_labels) = self._axis_ticks(
                    self.x_scale,
                    null,
                    null,
                    x_low,
                    x_high,
                    x_hint
                )
                plot = self._plot_area(
                    device,
                    x_labels,
                    y_labels,
                    _fractions(self.x_scale, x_ticks, x_low, x_high)
                )
            }
        }

        let (x_minor, x_marks) = self._minor_of(self.x_scale, x_ticks, x_low, x_high, x_hint)
        let (y_minor, y_marks) = self._minor_of(self.y_scale, y_ticks, y_low, y_high, y_hint)
        return _Furniture(
            x_ticks,
            x_labels,
            x_minor,
            x_marks,
            y_ticks,
            y_labels,
            y_minor,
            y_marks,
            plot
        )
    }

    #: The positions and names one axis draws, fixed or chosen.
    #:
    #: A fixed position outside the limits is dropped together with its name: it
    #: has nowhere to go, and drawing it against the panel edge would attach the
    #: wrong name to the wrong place.
    fn _axis_ticks(
        self,
        scale: Scale,
        fixed: list[float]?,
        names: list[str]?,
        low: float,
        high: float,
        hint: int
    ) -> tuple[list[float], list[str]] {
        if fixed == null {
            let values = scale.ticks(low, high, hint)
            return (values, scale.labels(values, low, high, hint))
        }

        var values = []
        var kept = []
        for (index, value) in fixed.enumerate() {
            if not scale.accepts(value) or value < low or value > high { continue }
            values.push(value)
            if names != null { kept.push(names[index]) }
        }
        if names == null { return (values, scale.labels(values, low, high, hint)) }
        return (values, kept)
    }

    #: The intermediate positions for one axis, and whether they get marks.
    #:
    #: Three sources, in order: what the scale offers -- a log axis knows its
    #: own multiples -- then, if minor ticks were asked for outright, an even
    #: split of the labelled step, and finally the same split without marks when
    #: only the minor gridlines were wanted.
    fn _minor_of(
        self,
        scale: Scale,
        ticks: list[float],
        low: float,
        high: float,
        hint: int
    ) -> tuple[list[float], bool] {
        if self.minor_wanted == false { return ([], false) }

        let supplied = scale.minor_ticks(low, high, hint)
        if supplied.len() > 0 { return (supplied, true) }
        if self.minor_wanted == true { return (_subdivide(ticks, low, high), true) }
        if self.grid_on and self.grid_which != "major" {
            return (_subdivide(ticks, low, high), false)
        }
        return ([], false)
    }

    #: Room the title, the axis labels and the x tick labels take vertically.
    fn _vertical_reserve(self) -> float {
        var reserve = FIGURE_PADDING * 2.0 + text_height("0", 1) + TICK_LENGTH + LABEL_GAP
        if self.x_text != null { reserve += text_height(self.x_text ?? "", 1) + LABEL_GAP }
        if self.title_text != null { reserve += text_height(self.title_text ?? "", 2) + LABEL_GAP }
        return reserve
    }

    #: The rectangle left for the data once the furniture has taken its room.
    #:
    #: `x_at` is where each x tick falls along the panel, from 0 at the left
    #: edge to 1 at the right. An x label is centred on its tick, so the ones
    #: near the ends hang past the panel by up to half their own width, and how
    #: far past depends on how wide the panel is -- which is what is being
    #: computed. So the two sides are solved rather than derived: each pass
    #: reserves what the current width says the labels need, both sides only
    #: ever grow, and the loop climbs to the answer. Without it `set_xlim(0,
    #: 12000)` on a 400px figure wrote its last label as `1200` with the final
    #: digit off the canvas.
    fn _plot_area(
        self,
        device: Rect,
        x_labels: list[str],
        y_labels: list[str],
        x_at: list[float]
    ) -> Rect {
        var top = FIGURE_PADDING
        var bottom = FIGURE_PADDING + text_height("0", 1) + TICK_LENGTH + LABEL_GAP
        var base_left = FIGURE_PADDING + _widest(y_labels) + TICK_LENGTH + LABEL_GAP

        if self.y_text != null { base_left += text_height(self.y_text ?? "", 1) + LABEL_GAP }
        if self.x_text != null { bottom += text_height(self.x_text ?? "", 1) + LABEL_GAP }
        if self.title_text != null { top += text_height(self.title_text ?? "", 2) + LABEL_GAP }

        var left = base_left
        var right = FIGURE_PADDING
        var pass = 0
        while pass < 6 {
            let span = device.width() - left - right
            if span <= 0.0 { break }
            var over_left = 0.0
            var over_right = 0.0
            for (index, place) in x_at.enumerate() {
                if index >= x_labels.len() { break }
                let half = text_width(x_labels[index], 1) * 0.5
                over_left = max(over_left, half - place * span)
                over_right = max(over_right, half - (1.0 - place) * span)
            }
            let next_left = max(base_left, FIGURE_PADDING + over_left)
            let next_right = FIGURE_PADDING + over_right
            let settled = next_left - left < 0.5 and next_right - right < 0.5
            left = next_left
            right = next_right
            if settled { break }
            pass += 1
        }

        return Rect(
            device.left() + left,
            device.bottom() + top,
            device.right() - right,
            device.top() - bottom
        )
    }

    fn _draw_grid(self, list: DisplayList, frame: Frame, plot: Rect, measured: _Furniture) -> void {
        # Minor first, so where a major line and a minor one land on the same
        # pixel the full-strength one is what shows.
        if self.grid_which != "major" {
            let faint = Stroke(self.theme.grid.with_alpha(MINOR_GRID_ALPHA), width: 1.0)
            self._rules(list, frame, plot, measured.x_minor, measured.y_minor, faint)
        }
        if self.grid_which != "minor" {
            let rule = Stroke(self.theme.grid, width: 1.0)
            self._rules(list, frame, plot, measured.x_ticks, measured.y_ticks, rule)
        }
    }

    #: Rules across the panel at every position given, in one stroke style.
    fn _rules(
        self,
        list: DisplayList,
        frame: Frame,
        plot: Rect,
        x_at: list[float],
        y_at: list[float],
        stroke: Stroke
    ) -> void {
        for value in x_at {
            let placed = frame.place(value, frame.data.bottom())
            if placed == null { continue }
            let ends = [Point(placed.x, plot.bottom()), Point(placed.x, plot.top())]
            list.push(Primitive.Polyline(ends, stroke))
        }
        for value in y_at {
            let placed = frame.place(frame.data.left(), value)
            if placed == null { continue }
            let ends = [Point(plot.left(), placed.y), Point(plot.right(), placed.y)]
            list.push(Primitive.Polyline(ends, stroke))
        }
    }

    #: Draw whichever of the four axis lines are showing.
    #:
    #: Device y grows downward, so the panel's visual top edge is `plot.bottom()`
    #: and its visual bottom is `plot.top()`. The names below are the reader's,
    #: which is also what `set_spines` takes.
    fn _draw_spines(self, list: DisplayList, plot: Rect) -> void {
        let spine = Stroke(self.theme.spine, width: 1.0)
        let top_left = Point(plot.left(), plot.bottom())
        let top_right = Point(plot.right(), plot.bottom())
        let low_left = Point(plot.left(), plot.top())
        let low_right = Point(plot.right(), plot.top())

        # Intact, the frame is one closed stroke: a single primitive that mitres
        # its own corners. Hiding a side breaks the loop, and the survivors then
        # have to go out separately.
        if self.spine_left and self.spine_right and self.spine_top and self.spine_bottom {
            list.push(
                Primitive.Polyline([top_left, top_right, low_right, low_left, top_left], spine)
            )
            return
        }
        if self.spine_bottom { list.push(Primitive.Polyline([low_left, low_right], spine)) }
        if self.spine_top { list.push(Primitive.Polyline([top_left, top_right], spine)) }
        if self.spine_left { list.push(Primitive.Polyline([top_left, low_left], spine)) }
        if self.spine_right { list.push(Primitive.Polyline([top_right, low_right], spine)) }
    }

    #: Draw one axis's ticks and their labels.
    #:
    #: `horizontal` picks the x axis (ticks along the top of `plot`, labels
    #: centred beneath) or the y axis (ticks along the left of `plot`, labels
    #: end-anchored beside them) -- the two used to be separate functions that
    #: differed only in these mechanical choices.
    #:
    #: An empty `labels` draws bare marks, which is what a minor tick is; the
    #: shorter `length` is the only other thing that distinguishes one.
    fn _draw_ticks(
        self,
        list: DisplayList,
        frame: Frame,
        plot: Rect,
        ticks: list[float],
        labels: list[str],
        horizontal: bool,
        length: float = TICK_LENGTH
    ) -> void {
        let mark = Stroke(self.theme.spine, width: 1.0)
        let style = if horizontal {
            TextStyle(self.theme.text, scale: 1, horizontal: HorizontalAnchor.Middle, vertical: VerticalAnchor.Top)
        } else {
            TextStyle(self.theme.text, scale: 1, horizontal: HorizontalAnchor.End, vertical: VerticalAnchor.Middle)
        }
        for (index, value) in ticks.enumerate() {
            let placed = if horizontal {
                frame.place(value, frame.data.bottom())
            } else {
                frame.place(frame.data.left(), value)
            }
            if placed == null { continue }

            let (ends, at) = if horizontal {
                let foot = plot.top()
                ([Point(placed.x, foot), Point(placed.x, foot + length)], Point(placed.x, foot + TICK_LENGTH + LABEL_GAP))
            } else {
                let edge = plot.left()
                ([Point(edge - length, placed.y), Point(edge, placed.y)], Point(edge - TICK_LENGTH - LABEL_GAP, placed.y))
            }
            list.push(Primitive.Polyline(ends, mark))
            if index < labels.len() {
                list.push(Primitive.Text(at, labels[index], style))
            }
        }
    }

    fn _draw_labels(self, list: DisplayList, device: Rect, plot: Rect) -> void {
        let title = self.title_text
        if title != null {
            let style = TextStyle(
                self.theme.foreground,
                scale: 2,
                horizontal: HorizontalAnchor.Middle,
                vertical: VerticalAnchor.Top
            )
            let at = Point(plot.center().x, device.bottom() + FIGURE_PADDING)
            list.push(Primitive.Text(at, title, style))
        }

        let x_label = self.x_text
        if x_label != null {
            let style = TextStyle(
                self.theme.text,
                scale: 1,
                horizontal: HorizontalAnchor.Middle,
                vertical: VerticalAnchor.Bottom
            )
            let at = Point(plot.center().x, device.top() - FIGURE_PADDING)
            list.push(Primitive.Text(at, x_label, style))
        }

        let y_label = self.y_text
        if y_label != null {
            let style = TextStyle(
                self.theme.text,
                scale: 1,
                horizontal: HorizontalAnchor.Middle,
                vertical: VerticalAnchor.Top,
                quarter_turns: 1
            )
            let at = Point(device.left() + FIGURE_PADDING, plot.center().y)
            list.push(Primitive.Text(at, y_label, style))
        }
    }

    fn __str__(self) -> str {
        return f"Axes({self.layers.len()} series)"
    }
}

#: Everything the furniture needs, settled once per render.
#:
#: `_measure` produces one of these and `_render_range` reads it. Passing the
#: ticks, their names and the panel together is what stops `plot_area` and
#: `render` from each running the layout and quietly disagreeing about where the
#: panel is -- the window backend turns a cursor position back into data through
#: the first and draws through the second.
class _Furniture {
    #: Labelled positions along the horizontal axis.
    pub let x_ticks: list[float]

    #: What is written beside each of them.
    pub let x_labels: list[str]

    #: Unlabelled positions along the horizontal axis.
    pub let x_minor: list[float]

    #: Whether the horizontal minor positions get tick marks as well as rules.
    pub let x_marks: bool

    #: Labelled positions along the vertical axis.
    pub let y_ticks: list[float]

    #: What is written beside each of them.
    pub let y_labels: list[str]

    #: Unlabelled positions along the vertical axis.
    pub let y_minor: list[float]

    #: Whether the vertical minor positions get tick marks as well as rules.
    pub let y_marks: bool

    #: The rectangle the data is drawn into.
    pub let plot: Rect

    pub fn init(
        self,
        x_ticks: list[float],
        x_labels: list[str],
        x_minor: list[float],
        x_marks: bool,
        y_ticks: list[float],
        y_labels: list[str],
        y_minor: list[float],
        y_marks: bool,
        plot: Rect
    ) {
        self.x_ticks = x_ticks
        self.x_labels = x_labels
        self.x_minor = x_minor
        self.x_marks = x_marks
        self.y_ticks = y_ticks
        self.y_labels = y_labels
        self.y_minor = y_minor
        self.y_marks = y_marks
        self.plot = plot
    }
}

#: The width of the longest label in `labels`, at scale one.
fn _widest(labels: list[str]) -> float {
    var widest = 0.0
    for label in labels { widest = max(widest, text_width(label, 1)) }
    return widest
}

#: How many labels of `per_label` pixels fit across `span`, capped at `hint`.
#:
#: Ticks bound a run of gaps rather than filling them, so `n` labels need only
#: `n - 1` gaps between them and one more fits than the division suggests.
#: Never returns fewer than `MIN_TICK_HINT`: a panel too small even for two
#: labels is going to look wrong whatever is done, and the tick rule refuses a
#: hint below two outright.
fn _fitted(span: float, per_label: float, hint: int) -> int {
    if per_label <= 0.0 or span <= 0.0 { return hint }
    let room = int(span / per_label) + 1
    return max(MIN_TICK_HINT, min(hint, room))
}

#: Where each of `values` falls along the axis, from 0 at `low` to 1 at `high`.
#:
#: In warped space, so a tick at 100 on a decade from 1 to 1000 comes out at two
#: thirds and the label overhang is measured where the label actually lands.
fn _fractions(scale: Scale, values: list[float], low: float, high: float) -> list[float] {
    let start = scale.forward(low)
    let span = scale.forward(high) - start
    if span == 0.0 { return [0.0 for _value in values] }
    return [(scale.forward(value) - start) / span for value in values]
}

#: Positions between `ticks`, for an axis whose scale supplies none.
#:
#: A linear axis has no intermediate positions of its own -- the labelled ones
#: are already evenly spaced -- so `minorticks_on` has to invent them. Refuses
#: unevenly spaced input, which is the honest answer for a log axis whose own
#: minor rule declined the range: subdividing 1, 10, 100 by arithmetic would put
#: marks at 5.5 and 55, positions that mean nothing on that axis.
fn _subdivide(ticks: list[float], low: float, high: float) -> list[float] {
    if ticks.len() < 2 { return [] }
    let step = ticks[1] - ticks[0]
    if not math.is_finite(step) or step <= 0.0 { return [] }
    for index in 1..ticks.len() {
        if math.fabs(ticks[index] - ticks[index - 1] - step) > step * 1e-6 { return [] }
    }

    let parts = _division_count(step)
    let fine = step / parts.to_float()
    # One whole step past each end, since the labelled ticks stop inside the
    # limits and the space beyond the last one is still axis.
    let last = (ticks.len() - 1) * parts + parts
    var values = []
    var index = -parts
    while index <= last {
        if index % parts != 0 {
            let value = ticks[0] + fine * index.to_float()
            if value >= low and value <= high { values.push(value) }
        }
        index += 1
    }
    return values
}

#: How many parts a labelled step is split into for its minor ticks.
#:
#: Matplotlib's rule, and it is the right one: a step of 1 splits five ways so
#: the minors land on tenths, and a step of 2 splits four ways so they land on
#: halves. Splitting a step of 2 five ways would put a mark at 0.4, a position
#: no reader can name.
fn _division_count(step: float) -> int {
    let mantissa = step / 10.0 ** math.floor(math.log10(step))
    if mantissa < 1.5 { return 5 }
    if mantissa < 3.5 { return 4 }
    return 5
}

#: Copy fixed tick positions, checking they can carry the names given.
fn _checked_ticks(values: list[float], labels: list[str]?, axis: str) -> list[float] {
    if labels != null and labels.len() != values.len() {
        throw ValueError(
            f"the {axis} axis was given {values.len()} ticks and {labels.len()} labels"
        )
    }
    return [value for value in values]
}

#: Copy tick names, checking that positions were fixed for them to sit on.
fn _checked_names(fixed: list[float]?, labels: list[str], axis: str) -> list[str] {
    if fixed == null {
        throw ValueError(
            f"set_{axis}ticks must fix the positions before set_{axis}ticklabels names them"
        )
    }
    if fixed.len() != labels.len() {
        throw ValueError(
            f"the {axis} axis has {fixed.len()} fixed ticks and was given {labels.len()} labels"
        )
    }
    return [label for label in labels]
}

#: Accept a tick-count cap, or keep the one in force.
fn _checked_hint(wanted: int?, current: int, axis: str) -> int {
    if wanted == null { return current }
    if wanted < MIN_TICK_HINT {
        throw ValueError(f"the {axis} axis needs at least {MIN_TICK_HINT} ticks, got {wanted}")
    }
    return wanted
}

#: A filled rectangle covering `box`.
fn _fill(box: Rect, color: Color) -> Primitive {
    let corners = [
        Point(box.left(), box.bottom()),
        Point(box.right(), box.bottom()),
        Point(box.right(), box.top()),
        Point(box.left(), box.top()),
    ]
    return Primitive.Polygon(corners, Paint.solid(color), null)
}

#: The part of `[low, high]` this scale can place, or null when none of it is.
#:
#: A log axis has no position for zero or for a negative number, and a series
#: drops those samples rather than inventing one for them. Ranging has to drop
#: them the same way. Throwing the whole range away instead is what put a y of
#: `[1, 10, 0, 10, 100]` on an axis running 1 to 10, with the hundred drawn
#: above the top of a picture that otherwise looked perfectly finished.
#:
#: The top comes through masking unchanged: the scale accepts it, so it is a
#: real sample. The bottom cannot be recovered from a bounding box at all -- a
#: box that reaches zero says nothing about the smallest positive sample inside
#: it -- so the origin of the space the axis is linear in stands in for it,
#: which on a log axis of any base is one. Erring wide is the safe direction,
#: since a view taller than the data only wastes space while a view shorter than
#: it hides a sample, so a top above one keeps one rather than guessing at a
#: nearer decade. Data that lives below one gets the decade under its top.
#:
#: The exact answer is the smallest positive sample, which every series knows
#: and none of them can be asked for: `Series.extent` hands back a box.
fn _drawable(low: float, high: float, scale: Scale) -> tuple[float, float]? {
    if scale.accepts(low) and scale.accepts(high) { return (low, high) }
    if not scale.accepts(high) { return null }
    let origin = scale.inverse(0.0)
    if scale.accepts(origin) and origin < high { return (origin, high) }
    return (scale.inverse(scale.forward(high) - 1.0), high)
}

#: `low` and `high` moved apart by `AUTO_PAD` of the span between them.
#:
#: The span is measured in the space the scale is linear in, so the margin means
#: the same thing on every axis: a twentieth of the panel either way, rather
#: than a twentieth of the difference between two numbers that may be three
#: decades apart. On a log axis running 1 to 100 that is a tenth of a decade at
#: each end, which is what matplotlib's `axes.ymargin` gives as well.
fn _padded(low: float, high: float, scale: Scale) -> tuple[float, float] {
    let near = scale.forward(low)
    let far = scale.forward(high)
    let room = (far - near) * AUTO_PAD
    return (scale.inverse(near - room), scale.inverse(far + room))
}

#: Limits for an axis with nothing plottable on it.
#:
#: A log axis cannot use the unit square, since it would have to place zero.
fn _fallback(scale: Scale) -> tuple[float, float] {
    if scale.accepts(0.0) { return (0.0, 1.0) }
    return (1.0, 10.0)
}

#: Accept a scale either as an instance or by name.
fn _as_scale(scale: any) -> Scale {
    if scale is str { return scale_named(scale) }
    if scale is Scale { return scale }
    throw ValueError(f"a scale must be a Scale or its name, got {type_of(scale)}")
}
