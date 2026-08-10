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
pub const TICK_HINT = 6

#: Length of a tick mark, in device pixels.
pub const TICK_LENGTH: float = 5.0

#: Gap between a tick mark and its label.
pub const LABEL_GAP: float = 4.0

#: Padding at the edges of the figure that no content may enter.
pub const FIGURE_PADDING: float = 10.0

#: Fraction of the data range left as breathing room when ranging automatically.
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

    #: Turn the gridlines on or off.
    pub fn grid(self, on: bool = true) -> Axes {
        self.grid_on = on
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
        let x_ticks = self.x_scale.ticks(x_low, x_high, TICK_HINT)
        let y_ticks = self.y_scale.ticks(y_low, y_high, TICK_HINT)
        return self._plot_area(
            device,
            self.x_scale.labels(x_ticks, x_low, x_high, TICK_HINT),
            self.y_scale.labels(y_ticks, y_low, y_high, TICK_HINT)
        )
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
        let x_ticks = self.x_scale.ticks(x_low, x_high, TICK_HINT)
        let y_ticks = self.y_scale.ticks(y_low, y_high, TICK_HINT)
        let x_labels = self.x_scale.labels(x_ticks, x_low, x_high, TICK_HINT)
        let y_labels = self.y_scale.labels(y_ticks, y_low, y_high, TICK_HINT)

        let plot = self._plot_area(device, x_labels, y_labels)
        if plot.is_empty() { return list }

        let frame = Frame(Rect(x_low, y_low, x_high, y_high), plot, self.x_scale, self.y_scale)

        if furniture {
            list.push(_fill(device, self.theme.background))
            list.push(_fill(plot, self.theme.background))
            if self.grid_on { self._draw_grid(list, frame, plot, x_ticks, y_ticks) }
        }

        list.push(Primitive.PushClip(plot))
        for index in from_index..to_index {
            for primitive in self.layers[index].draw(frame) { list.push(primitive) }
        }
        list.push(Primitive.PopClip)

        if not furniture { return list }

        self._draw_spines(list, plot)
        self._draw_x_ticks(list, frame, plot, x_ticks, x_labels)
        self._draw_y_ticks(list, frame, plot, y_ticks, y_labels)
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
        var lower = low ?? found_low
        var upper = high ?? found_high
        if not scale.accepts(lower) or not scale.accepts(upper) {
            return _fallback(scale)
        }
        if lower == upper {
            # A constant series has no range to divide by. Widening it around
            # the value keeps the line visible in the middle of the axes rather
            # than raising from a zero-extent transform.
            let reach = if math.fabs(lower) > 0.0 { math.fabs(lower) * 0.5 } else { 1.0 }
            lower -= reach
            upper += reach
            if not scale.accepts(lower) { lower = upper * 0.1 }
        }
        let (nice_low, nice_high) = scale.nice(lower, upper, TICK_HINT)
        return (if low == null {
            nice_low
        } else {
            low
        }, if high == null {
            nice_high
        } else {
            high
        })
    }

    #: The rectangle left for the data once the furniture has taken its room.
    fn _plot_area(self, device: Rect, x_labels: list[str], y_labels: list[str]) -> Rect {
        var widest = 0.0
        for label in y_labels { widest = max(widest, text_width(label, 1)) }

        var left = FIGURE_PADDING + widest + TICK_LENGTH + LABEL_GAP
        var bottom = FIGURE_PADDING + text_height("0", 1) + TICK_LENGTH + LABEL_GAP
        var top = FIGURE_PADDING
        let right = FIGURE_PADDING

        if self.y_text != null { left += text_height(self.y_text ?? "", 1) + LABEL_GAP }
        if self.x_text != null { bottom += text_height(self.x_text ?? "", 1) + LABEL_GAP }
        if self.title_text != null { top += text_height(self.title_text ?? "", 2) + LABEL_GAP }

        # Device y grows downward, so the visual top is the smaller coordinate.
        return Rect(
            device.left() + left,
            device.bottom() + top,
            device.right() - right,
            device.top() - bottom
        )
    }

    fn _draw_grid(
        self,
        list: DisplayList,
        frame: Frame,
        plot: Rect,
        x_ticks: list[float],
        y_ticks: list[float]
    ) -> void {
        let rule = Stroke(self.theme.grid, width: 1.0)
        for value in x_ticks {
            let placed = frame.place(value, frame.data.bottom())
            if placed == null { continue }
            let ends = [Point(placed.x, plot.bottom()), Point(placed.x, plot.top())]
            list.push(Primitive.Polyline(ends, rule))
        }
        for value in y_ticks {
            let placed = frame.place(frame.data.left(), value)
            if placed == null { continue }
            let ends = [Point(plot.left(), placed.y), Point(plot.right(), placed.y)]
            list.push(Primitive.Polyline(ends, rule))
        }
    }

    fn _draw_spines(self, list: DisplayList, plot: Rect) -> void {
        let spine = Stroke(self.theme.spine, width: 1.0)
        let frame_points = [
            Point(plot.left(), plot.bottom()),
            Point(plot.right(), plot.bottom()),
            Point(plot.right(), plot.top()),
            Point(plot.left(), plot.top()),
            Point(plot.left(), plot.bottom()),
        ]
        list.push(Primitive.Polyline(frame_points, spine))
    }

    fn _draw_x_ticks(
        self,
        list: DisplayList,
        frame: Frame,
        plot: Rect,
        ticks: list[float],
        labels: list[str]
    ) -> void {
        let mark = Stroke(self.theme.spine, width: 1.0)
        let style = TextStyle(
            self.theme.text,
            scale: 1,
            horizontal: HorizontalAnchor.Middle,
            vertical: VerticalAnchor.Top
        )
        for (index, value) in ticks.enumerate() {
            let placed = frame.place(value, frame.data.bottom())
            if placed == null { continue }
            let foot = plot.top()
            let ends = [Point(placed.x, foot), Point(placed.x, foot + TICK_LENGTH)]
            list.push(Primitive.Polyline(ends, mark))
            if index < labels.len() {
                let at = Point(placed.x, foot + TICK_LENGTH + LABEL_GAP)
                list.push(Primitive.Text(at, labels[index], style))
            }
        }
    }

    fn _draw_y_ticks(
        self,
        list: DisplayList,
        frame: Frame,
        plot: Rect,
        ticks: list[float],
        labels: list[str]
    ) -> void {
        let mark = Stroke(self.theme.spine, width: 1.0)
        let style = TextStyle(
            self.theme.text,
            scale: 1,
            horizontal: HorizontalAnchor.End,
            vertical: VerticalAnchor.Middle
        )
        for (index, value) in ticks.enumerate() {
            let placed = frame.place(frame.data.left(), value)
            if placed == null { continue }
            let edge = plot.left()
            let ends = [Point(edge - TICK_LENGTH, placed.y), Point(edge, placed.y)]
            list.push(Primitive.Polyline(ends, mark))
            if index < labels.len() {
                let at = Point(edge - TICK_LENGTH - LABEL_GAP, placed.y)
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
            # One quarter turn anticlockwise, so the label reads bottom to top
            # the way a y axis label conventionally does.
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
