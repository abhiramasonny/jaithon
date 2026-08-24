#: The page: a size in pixels, and the axes laid out on it.
#:
#: A figure owns the device rectangle and divides it among its axes. It renders
#: by asking each set of axes for its display list and concatenating them, in
#: the order the axes were created.
#:
#: Where a panel goes is decided at render time rather than when it is created.
#: A panel does not know how wide its tick labels are until it has data, and two
#: panels in one column have to agree on a left margin before either can be
#: placed; a `suptitle` added at the end has to take its band off a page that is
#: already full. Both need the whole page in hand, so `layout` recomputes it.
#:
#: ```
#: let figure = Figure(width: 400, height: 300)
#: let axes = figure.axes()
#: axes.line([0.0, 1.0], [0.0, 1.0])
#: figure.render().len() > 0        # true
#: ```
from .axes import Axes, FIGURE_PADDING
from .geom import Point, Rect
from .metrics import text_height
from .primitive import DisplayList, Primitive
from .style import Color, HorizontalAnchor, Paint, TextStyle, Theme, VerticalAnchor

#: Space left between neighbouring panels in a grid, in device pixels.
pub const PANEL_GAP: float = 8.0

#: Font multiplier for a figure-wide title, one step above an axes title.
pub const SUPTITLE_SCALE = 3

#: Space under a figure-wide title before the panels begin.
pub const SUPTITLE_GAP: float = 6.0

#: The mosaic entry that leaves a cell empty, as in matplotlib.
pub const MOSAIC_BLANK: str = "."

#: Two edges within this many pixels of each other count as the same edge.
#:
#: A gridspec column's left edge is a running sum of ratios and gaps, so two
#: panels in one column agree to the last bit only by luck. A tenth of a pixel
#: is far below anything a reader can see and far above the drift a dozen
#: additions produce.
pub const EDGE_TOLERANCE: float = 0.1

#: The square a panel's margins are measured against.
#:
#: `Axes` subtracts its margins from the rectangle it is handed and `Rect`
#: normalises, so measuring against a panel narrower than its own tick labels
#: returns a box turned inside out rather than a margin that overflows. The
#: margins do not depend on the size of the rectangle at all, only on the
#: labels, so measuring against one nothing can overflow gives them directly.
pub const MEASURE_SPAN: float = 1.0e6

#: A division of a rectangle into rows and columns, with ratios and spans.
#:
#: The geometry behind `subplots`, `mosaic` and every hand-built layout: it
#: computes rectangles and nothing else, so `figure.add_axes(spec.cell(...))`
#: is the whole of what matplotlib spells `fig.add_subplot(gs[0, :])`.
#:
#: Rows and columns may be weighted, which is what a wide time-series strip
#: over two square detail panels needs, and a cell may span several of either.
#:
#: ```
#: let spec = GridSpec(Rect(0.0, 0.0, 208.0, 100.0), 1, 2, x_gap: 8.0)
#: spec.cell(0, 1)                    # Rect(108.0, 0.0, 208.0, 100.0)
#: spec.cell(0, 0, col_span: 2)       # Rect(0.0, 0.0, 208.0, 100.0)
#: ```
pub class GridSpec {
    #: How many rows the rectangle was divided into.
    pub let rows: int

    #: How many columns the rectangle was divided into.
    pub let cols: int

    var lefts: list[float]
    var rights: list[float]
    var tops: list[float]
    var bottoms: list[float]

    #: Divide `region` into `rows` by `cols` tracks.
    #:
    #: A ratio list must be empty, meaning even tracks, or name every track.
    #:
    #: Raises `ValueError` on a non-positive count, a negative gap, a ratio list
    #: of the wrong length, a ratio that is not positive, and on a region the
    #: gaps alone would fill.
    pub fn init(
        self,
        region: Rect,
        rows: int,
        cols: int,
        width_ratios: list[float] = [],
        height_ratios: list[float] = [],
        x_gap: float = PANEL_GAP,
        y_gap: float = PANEL_GAP
    ) {
        if rows <= 0 or cols <= 0 {
            throw ValueError(f"a grid needs positive counts, got {rows}x{cols}")
        }
        if x_gap < 0.0 or y_gap < 0.0 {
            throw ValueError(f"a grid gap cannot be negative, got {x_gap} and {y_gap}")
        }
        self.rows = rows
        self.cols = cols
        let (lefts, rights) = _tracks(
            region.left(),
            region.width(),
            cols,
            width_ratios,
            x_gap,
            "column"
        )
        let (tops, bottoms) = _tracks(
            region.bottom(),
            region.height(),
            rows,
            height_ratios,
            y_gap,
            "row"
        )
        self.lefts = lefts
        self.rights = rights
        self.tops = tops
        self.bottoms = bottoms
    }

    #: The rectangle covering `row_span` rows and `col_span` columns from
    #: `(row, col)`, gaps between the spanned tracks included.
    #:
    #: Raises `ValueError` on a non-positive span and on a cell that reaches
    #: past the edge of the grid.
    pub fn cell(self, row: int, col: int, row_span: int = 1, col_span: int = 1) -> Rect {
        if row_span <= 0 or col_span <= 0 {
            throw ValueError(f"a cell must span at least one track, got {row_span}x{col_span}")
        }
        if row < 0 or col < 0 or row + row_span > self.rows or col + col_span > self.cols {
            throw ValueError(
                f"cell ({row}, {col}) spanning {row_span}x{col_span} leaves a " +
                    f"{self.rows}x{self.cols} grid"
            )
        }
        return Rect(
            self.lefts[col],
            self.tops[row],
            self.rights[col + col_span - 1],
            self.bottoms[row + row_span - 1]
        )
    }

    fn __str__(self) -> str {
        return f"GridSpec({self.rows}x{self.cols})"
    }
}

#: A figure and the axes drawn on it.
pub class Figure {
    #: Width of the figure in device pixels.
    pub let width: int

    #: Height of the figure in device pixels.
    pub let height: int

    var panels: list[Axes]
    var requests: list[Rect]
    var hosts: list[int?]
    var theme: Theme
    var title_text: str?
    var title_scale: int

    #: Create a figure of this size.
    #:
    #: Raises `ValueError` on a non-positive dimension, which would leave
    #: nothing to draw into and produce a transform with no extent.
    pub fn init(self, width: int = 640, height: int = 480, theme: Theme? = null) {
        if width <= 0 or height <= 0 {
            throw ValueError(f"a figure must have positive size, got {width}x{height}")
        }
        self.width = width
        self.height = height
        self.panels = []
        self.requests = []
        self.hosts = []
        self.theme = theme ?? Theme.light()
        self.title_text = null
        self.title_scale = SUPTITLE_SCALE
    }

    #: The whole figure as one set of axes.
    #:
    #: Calling this a second time does not build a twinned axis. Every set of
    #: axes opens by filling its whole region with the background colour, so the
    #: second one paints the first out -- grid, series, title and all; rendered
    #: on a 400x300 page the first plot leaves no pixels behind whatsoever. A
    #: twin needs the upper axes to skip that fill and to hang its ticks off the
    #: right-hand spine, and `Axes` cannot yet be asked for either.
    #:
    #: For a grid use `subplots` or `gridspec`, for a panel placed by hand
    #: `add_axes`, and for one drawn inside another `inset`.
    pub fn axes(self) -> Axes {
        return self.add_axes(self.content())
    }

    #: Register a set of axes covering `region`, in device pixels.
    #:
    #: The one door the other layouts go through: `subplots`, `gridspec`,
    #: `mosaic` and `inset` each compute a rectangle and hand it here, so a
    #: colorbar's narrow strip or a panel in a place none of them describes is
    #: the same call with a different rectangle.
    #:
    #: The region is not checked for room, since a strip too narrow for a tick
    #: label is a reasonable thing to ask for. Instead every panel is clipped to
    #: its own region as it draws, so one that runs out of room is cropped at
    #: its edge rather than painted across its neighbour.
    #:
    #: Raises `ValueError` on a region with no area and on one that leaves the
    #: page, either of which is a layout arithmetic mistake rather than a
    #: request worth honouring.
    pub fn add_axes(self, region: Rect) -> Axes {
        if region.is_empty() {
            throw ValueError(f"a panel needs a region with area, got {region}")
        }
        let page = self.device().inset(-EDGE_TOLERANCE, -EDGE_TOLERANCE)
        if page.intersect(region) != region {
            throw ValueError(f"{region} does not fit on a {self.width}x{self.height} page")
        }
        return self._panel(region, null)
    }

    #: A `rows` by `cols` grid of axes, row-major.
    #:
    #: Raises `ValueError` on a non-positive count, and when a cell is too small
    #: to hold a panel. The second is not the old test that a cell has positive
    #: area: a cell smaller than the furniture an empty set of axes reserves
    #: produces a data box that starts outside the cell, which used to be drawn
    #: over the neighbouring panel and is now merely clipped away.
    pub fn subplots(self, rows: int, cols: int) -> list[list[Axes]] {
        let spec = self.gridspec(rows, cols)
        let (need_x, need_y) = _bare_margins(self.theme)
        let cell = spec.cell(0, 0)
        if cell.width() <= need_x or cell.height() <= need_y {
            throw ValueError(
                f"a {rows}x{cols} grid of {self.width}x{self.height} pixels gives each panel " +
                    f"{cell.width()}x{cell.height()}, and a panel needs more than {need_x}x{need_y}"
            )
        }

        var grid = []
        for row in 0..rows {
            var line = []
            for column in 0..cols { line.push(self.add_axes(spec.cell(row, column))) }
            grid.push(line)
        }
        return grid
    }

    #: A division of the page into weighted rows and columns.
    #:
    #: Nothing is registered by this call. Ask the result for the cells you want
    #: and hand each to `add_axes`, which is how a cell that spans a whole row
    #: becomes one wide panel:
    #:
    #: ```
    #: let figure = Figure(width: 400, height: 300)
    #: let spec = figure.gridspec(2, 2, height_ratios: [1.0, 2.0])
    #: let strip = figure.add_axes(spec.cell(0, 0, col_span: 2))
    #: figure.all_axes().len()      # 1
    #: ```
    pub fn gridspec(
        self,
        rows: int,
        cols: int,
        width_ratios: list[float] = [],
        height_ratios: list[float] = [],
        x_gap: float = PANEL_GAP,
        y_gap: float = PANEL_GAP
    ) -> GridSpec {
        return GridSpec(
            self.content(),
            rows,
            cols,
            width_ratios: width_ratios,
            height_ratios: height_ratios,
            x_gap: x_gap,
            y_gap: y_gap
        )
    }

    #: Lay the page out from a picture of it, and return the panels by name.
    #:
    #: `layout` is either a multi-line string, one character a cell, or a list
    #: of rows of names. A name repeated over neighbouring cells becomes one
    #: panel spanning them, and `MOSAIC_BLANK` leaves a cell empty.
    #:
    #: ```
    #: let figure = Figure(width: 400, height: 300)
    #: let panels = figure.mosaic("AA\nBC")
    #: panels.keys().len()      # 3
    #: ```
    #:
    #: Raises `ValueError` on a ragged layout, an empty one, and on a name whose
    #: cells do not form a rectangle -- an L cannot be one set of axes, and
    #: silently taking its bounding box would cover a cell somebody else owns.
    pub fn mosaic(
        self,
        layout: any,
        width_ratios: list[float] = [],
        height_ratios: list[float] = [],
        x_gap: float = PANEL_GAP,
        y_gap: float = PANEL_GAP
    ) -> dict[str, Axes] {
        let rows = _mosaic_rows(layout)
        if rows.len() == 0 { throw ValueError("a mosaic needs at least one row") }
        let cols = rows[0].len()
        if cols == 0 { throw ValueError("a mosaic needs at least one column") }
        for line in rows {
            if line.len() != cols {
                throw ValueError(
                    f"every mosaic row must name {cols} cells, got one of {line.len()}"
                )
            }
        }

        let spec = self.gridspec(
            rows.len(),
            cols,
            width_ratios: width_ratios,
            height_ratios: height_ratios,
            x_gap: x_gap,
            y_gap: y_gap
        )
        var placed: dict[str, Axes] = {}
        for (row, line) in rows.enumerate() {
            for (column, name) in line.enumerate() {
                if name == MOSAIC_BLANK or name in placed { continue }
                let (first_row, first_col, last_row, last_col) = _mosaic_span(rows, name)
                placed[name] = self.add_axes(
                    spec.cell(
                        first_row,
                        first_col,
                        row_span: last_row - first_row + 1,
                        col_span: last_col - first_col + 1
                    )
                )
            }
        }
        return placed
    }

    #: A small set of axes drawn inside `host`, for a zoom or a second view.
    #:
    #: `x` and `y` place the inset's top-left corner as a fraction of the host's
    #: data box, and `width` and `height` size it as a fraction of the same.
    #: Device y grows downward here as it does everywhere else in the library,
    #: so `y: 0.05` is near the top of the host rather than near its foot.
    #:
    #: Measuring against the host's data box rather than its region is what
    #: makes an inset land where it looks like it should: the region includes
    #: the tick labels, and `0.5` of it is not the middle of the plot.
    #:
    #: The inset is positioned after the rest of the page has been laid out, so
    #: it follows its host if aligning the margins moved it.
    #:
    #: Raises `ValueError` on a non-positive size and on a host that belongs to
    #: another figure.
    pub fn inset(self, host: Axes, x: float, y: float, width: float, height: float) -> Axes {
        if width <= 0.0 or height <= 0.0 {
            throw ValueError(f"an inset needs positive size, got {width}x{height}")
        }
        let anchor = self._index_of(host)
        if anchor == null {
            throw ValueError("the host axes were not created on this figure")
        }
        return self._panel(Rect(x, y, x + width, y + height), anchor)
    }

    #: Put `text` across the top of the page, above every panel.
    #:
    #: The band the title needs comes off the top of the page and everything
    #: already laid out is scaled into what is left, so this can be called last,
    #: which is where it reads best in the code that builds a figure. A
    #: `GridSpec` taken before the call is the one thing that goes stale: it
    #: holds the pixels it was divided from.
    #:
    #: Raises `ValueError` on a scale below 1 and on a title that would leave no
    #: page under it.
    pub fn suptitle(self, text: str, scale: int = SUPTITLE_SCALE) -> Figure {
        if scale < 1 {
            throw ValueError(f"a title scale must be at least 1, got {scale}")
        }
        let before = self.content()
        let band = FIGURE_PADDING + text_height(text, scale) + SUPTITLE_GAP
        if band >= self.height.to_float() {
            throw ValueError(
                f"a title at scale {scale} needs {band} of {self.height} pixels, " +
                    "leaving no page under it"
            )
        }
        self.title_text = text
        self.title_scale = scale

        let after = self.content()
        for (index, region) in self.requests.enumerate() {
            if self.hosts[index] == null {
                self.requests[index] = _rescale(region, before, after)
            }
        }
        return self
    }

    #: The whole figure rectangle, in device pixels.
    pub fn device(self) -> Rect {
        return Rect(0.0, 0.0, self.width.to_float(), self.height.to_float())
    }

    #: The page minus the band a figure-wide title has taken.
    #:
    #: What every layout divides. With no `suptitle` it is the whole device.
    pub fn content(self) -> Rect {
        let title = self.title_text
        if title == null { return self.device() }
        let band = FIGURE_PADDING + text_height(title, self.title_scale) + SUPTITLE_GAP
        return Rect(0.0, band, self.width.to_float(), self.height.to_float())
    }

    #: The theme this figure and its axes were created in.
    pub fn theme_of(self) -> Theme {
        return self.theme
    }

    #: Every set of axes on the figure, in creation order.
    pub fn all_axes(self) -> list[Axes] {
        return [panel for panel in self.panels]
    }

    #: The region every panel is finally drawn into, in creation order.
    #:
    #: Two passes. First the panels that stand on their own: each one's margins
    #: are measured, panels sharing an edge take the widest margin found on that
    #: side, and the region is pulled in by the difference so that the data
    #: boxes line up. A 2x1 grid of small numbers over millions used to start at
    #: x=36 and x=54; both now start at 54, which is where matplotlib puts them.
    #:
    #: Then the insets, which sit inside a host's data box and so cannot be
    #: placed until it has stopped moving.
    #:
    #: A pull that would swallow the panel is dropped rather than applied: a
    #: cell too small for its own labels is already the worst it can be, and
    #: making it inside out to match a roomier neighbour helps nobody.
    pub fn layout(self) -> list[Rect] {
        var placed = [region for region in self.requests]
        let count = self.panels.len()
        var margins = [_margins(panel) for panel in self.panels]

        for index in 0..count {
            if self.hosts[index] != null { continue }
            let region = self.requests[index]
            let (own_left, own_top, own_right, own_bottom) = margins[index]
            var want_left = own_left
            var want_top = own_top
            var want_right = own_right
            var want_bottom = own_bottom
            for other in 0..count {
                if self.hosts[other] != null { continue }
                let beside = self.requests[other]
                let (its_left, its_top, its_right, its_bottom) = margins[other]
                if _same(beside.left(), region.left()) { want_left = max(want_left, its_left) }
                if _same(beside.right(), region.right()) { want_right = max(want_right, its_right) }
                if _same(beside.bottom(), region.bottom()) { want_top = max(want_top, its_top) }
                if _same(beside.top(), region.top()) { want_bottom = max(want_bottom, its_bottom) }
            }

            let left = region.left() + want_left - own_left
            let y_min = region.bottom() + want_top - own_top
            let right = region.right() - (want_right - own_right)
            let y_max = region.top() - (want_bottom - own_bottom)
            if left < right and y_min < y_max { placed[index] = Rect(left, y_min, right, y_max) }
        }

        for index in 0..count {
            let anchor = self.hosts[index]
            if anchor != null {
                let inside = self.panels[anchor].plot_area(placed[anchor])
                let box = self.requests[index]
                placed[index] = Rect(
                    inside.left() + box.left() * inside.width(),
                    inside.bottom() + box.bottom() * inside.height(),
                    inside.left() + box.right() * inside.width(),
                    inside.bottom() + box.top() * inside.height()
                )
            }
        }
        return placed
    }

    #: Where `axes` finally draws its data, or null when they are not on this
    #: figure.
    #:
    #: Not the region `axes` was registered with: aligning the margins moves it.
    #: This is the rectangle a cursor position has to be tested against to find
    #: which panel of a grid it fell in.
    pub fn plot_area_of(self, axes: Axes) -> Rect? {
        let placed = self.layout()
        for (index, panel) in self.panels.enumerate() {
            if panel == axes { return panel.plot_area(placed[index]) }
        }
        return null
    }

    #: Treat everything added so far as a static background.
    #:
    #: An animation redraws only what was added after this call, blitting the
    #: rest from a cached image. For a filter running over a fixed map that is
    #: the map, the grid and every label -- most of the frame.
    #:
    #: Call it once the scene is built and before the series that will move.
    pub fn freeze(self) -> Figure {
        for panel in self.panels { panel.freeze_at(panel.series().len()) }
        return self
    }

    #: Report whether anything has been frozen into the background.
    pub fn is_frozen(self) -> bool {
        for panel in self.panels {
            if panel.frozen_count() > 0 { return true }
        }
        return false
    }

    #: Render only the unfrozen series of every panel.
    pub fn render_dynamic(self) -> DisplayList {
        let list = DisplayList()
        let placed = self.layout()
        for (index, panel) in self.panels.enumerate() {
            self._stack(list, placed[index], panel.render_dynamic(placed[index]))
        }
        list.validate()
        return list
    }

    #: Render the static and dynamic layers of every panel, separately.
    pub fn render_layers(self) -> tuple[DisplayList, DisplayList] {
        let background = self._furniture()
        let foreground = DisplayList()
        let placed = self.layout()
        for (index, panel) in self.panels.enumerate() {
            let (static_list, dynamic_list) = panel.render_layers(placed[index])
            self._stack(background, placed[index], static_list)
            self._stack(foreground, placed[index], dynamic_list)
        }
        background.validate()
        foreground.validate()
        return (background, foreground)
    }

    #: Render every set of axes into one display list.
    #:
    #: The result is validated before it is returned, so a backend can walk it
    #: without defending against an unbalanced clip.
    pub fn render(self) -> DisplayList {
        let list = self._furniture()
        let placed = self.layout()
        for (index, panel) in self.panels.enumerate() {
            self._stack(list, placed[index], panel.render(placed[index]))
        }
        list.validate()
        return list
    }

    #: The page's own background and title, under everything a panel draws.
    #:
    #: The panels no longer tile the page: aligning margins pulls a panel in
    #: from its cell, and the gaps of a grid were never covered by anything. The
    #: raster backend clears to white whatever the theme says, so without this
    #: fill a dark figure showed white between its panels.
    fn _furniture(self) -> DisplayList {
        let list = DisplayList()
        list.push(_fill(self.device(), self.theme.background))
        let title = self.title_text
        if title != null {
            let style = TextStyle(
                self.theme.foreground,
                scale: self.title_scale,
                horizontal: HorizontalAnchor.Middle,
                vertical: VerticalAnchor.Top
            )
            let at = Point(self.device().center().x, FIGURE_PADDING)
            list.push(Primitive.Text(at, title, style))
        }
        return list
    }

    #: Append what one panel drew, fenced into the region it owns.
    #:
    #: The clip is what stops a panel that has run out of room from drawing over
    #: its neighbour. A lone panel owns the whole page and has nothing to run
    #: into, so it is added as it is and a single-axes figure produces the same
    #: list it always did.
    fn _stack(self, list: DisplayList, region: Rect, part: DisplayList) -> void {
        if part.is_empty() { return }
        let fenced = self.panels.len() > 1
        if fenced { list.push(Primitive.PushClip(region)) }
        list.extend(part)
        if fenced { list.push(Primitive.PopClip) }
    }

    #: The index of `axes` on this figure, or null.
    fn _index_of(self, axes: Axes) -> int? {
        for (index, panel) in self.panels.enumerate() {
            if panel == axes { return index }
        }
        return null
    }

    #: Add a set of axes covering `region` and remember where it goes.
    #:
    #: `host` is the index of the panel an inset hangs off, and null for a panel
    #: whose region is already in device pixels. An inset keeps fractions here
    #: instead, since where it lands depends on where its host ends up.
    fn _panel(self, region: Rect, host: int?) -> Axes {
        let panel = Axes(theme: self.theme)
        self.panels.push(panel)
        self.requests.push(region)
        self.hosts.push(host)
        return panel
    }

    fn __str__(self) -> str {
        return f"Figure({self.width}x{self.height}, {self.panels.len()} axes)"
    }
}

#: The edges of `count` tracks laid along one axis of a rectangle.
#:
#: Returns the low edges and the high edges, so a span from one track to another
#: is a pair of lookups rather than a sum.
fn _tracks(
    start: float,
    extent: float,
    count: int,
    ratios: list[float],
    gap: float,
    what: str
) -> tuple[list[float], list[float]] {
    let weights = if ratios.len() == 0 { [1.0 for _each in 0..count] } else { ratios }
    if weights.len() != count {
        throw ValueError(f"{count} {what} ratios expected, got {ratios.len()}")
    }
    var total = 0.0
    for weight in weights {
        if weight <= 0.0 {
            throw ValueError(f"a {what} ratio must be positive, got {weight}")
        }
        total += weight
    }

    let free = extent - gap * (count - 1).to_float()
    if free <= 0.0 {
        throw ValueError(f"{count} {what}s and their gaps do not fit in {extent} pixels")
    }

    var lows = []
    var highs = []
    var at = start
    for weight in weights {
        let size = free * weight / total
        lows.push(at)
        highs.push(at + size)
        at += size + gap
    }
    return (lows, highs)
}

#: Where a panel's data box sits inside its region, as the four margins
#: (left, top, right, bottom) in device pixels.
#:
#: `Rect.bottom` is the smallest y and device y grows downward, so the box's
#: `bottom` edge is its visual top and that is the margin it reports second.
fn _margins(panel: Axes) -> tuple[float, float, float, float] {
    let plot = panel.plot_area(Rect(0.0, 0.0, MEASURE_SPAN, MEASURE_SPAN))
    return (plot.left(), plot.bottom(), MEASURE_SPAN - plot.right(), MEASURE_SPAN - plot.top())
}

#: The width and height a panel spends on furniture before it has any labels.
#:
#: An empty set of axes still ranges 0 to 1 and still writes those ticks, so
#: this is a floor no panel can go under, which is what makes it a fair test of
#: whether a grid cell is big enough to be worth drawing.
fn _bare_margins(theme: Theme) -> tuple[float, float] {
    let (left, top, right, bottom) = _margins(Axes(theme: theme))
    return (left + right, top + bottom)
}

#: Report whether two edges are the same edge, to `EDGE_TOLERANCE`.
fn _same(one: float, other: float) -> bool {
    let apart = one - other
    return -EDGE_TOLERANCE <= apart <= EDGE_TOLERANCE
}

#: Map a rectangle from the page `before` onto the page `after`.
#:
#: Only the vertical is touched, since a title takes a band off the top and
#: leaves the width alone.
fn _rescale(box: Rect, before: Rect, after: Rect) -> Rect {
    if before.height() <= 0.0 { return box }
    let factor = after.height() / before.height()
    return Rect(
        box.left(),
        after.bottom() + (box.bottom() - before.bottom()) * factor,
        box.right(),
        after.bottom() + (box.top() - before.bottom()) * factor
    )
}

#: A mosaic layout as rows of names, whichever way it was written.
#:
#: A multi-line string is one character a cell, which is how a layout is read at
#: a glance; a list of lists carries names longer than a character.
fn _mosaic_rows(layout: any) -> list[list[str]] {
    var rows = []
    if layout is str {
        for line in layout.split("\n") {
            let trimmed = line.strip()
            if trimmed.len() > 0 { rows.push([character for character in trimmed]) }
        }
        return rows
    }
    for line in layout {
        if line is str {
            rows.push([character for character in line])
        } else {
            rows.push([str(name) for name in line])
        }
    }
    return rows
}

#: The bounding cells of `name`, as (first row, first column, last row, last
#: column), checked for being a solid rectangle.
fn _mosaic_span(rows: list[list[str]], name: str) -> tuple[int, int, int, int] {
    var first_row = rows.len()
    var first_col = rows[0].len()
    var last_row = -1
    var last_col = -1
    for (row, line) in rows.enumerate() {
        for (column, label) in line.enumerate() {
            if label != name { continue }
            first_row = min(first_row, row)
            last_row = max(last_row, row)
            first_col = min(first_col, column)
            last_col = max(last_col, column)
        }
    }
    for row in first_row..last_row + 1 {
        for column in first_col..last_col + 1 {
            if rows[row][column] != name {
                throw ValueError(
                    f"mosaic entry {name} is not a rectangle: cell ({row}, {column}) " +
                        f"holds {rows[row][column]}"
                )
            }
        }
    }
    return (first_row, first_col, last_row, last_col)
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
