#: The page: a size in pixels, and the axes laid out on it.
#:
#: A figure owns the device rectangle and divides it among its axes. It renders
#: by asking each set of axes for its display list and concatenating them, in
#: the order the axes were created.
#:
#: ```
#: let figure = Figure(width: 400, height: 300)
#: let axes = figure.axes()
#: axes.line([0.0, 1.0], [0.0, 1.0])
#: figure.render().len() > 0        # true
#: ```
from .axes import Axes
from .geom import Rect
from .primitive import DisplayList
from .style import Theme

#: Space left between neighbouring panels in a grid, in device pixels.
pub const PANEL_GAP: float = 8.0

#: A figure and the axes drawn on it.
pub class Figure {
    #: Width of the figure in device pixels.
    pub let width: int

    #: Height of the figure in device pixels.
    pub let height: int

    var panels: list[Axes]
    var regions: list[Rect]
    var theme: Theme

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
        self.regions = []
        self.theme = theme ?? Theme.light()
    }

    #: The whole figure as one set of axes.
    #:
    #: Calling this more than once stacks another set of axes over the same
    #: region, which is how a twinned axis is built. Use `subplots` for a grid.
    pub fn axes(self) -> Axes {
        return self._panel(self.device())
    }

    #: A `rows` by `cols` grid of axes, row-major.
    #:
    #: Raises `ValueError` on a non-positive count, and when the grid is finer
    #: than the figure has pixels for.
    pub fn subplots(self, rows: int, cols: int) -> list[list[Axes]] {
        if rows <= 0 or cols <= 0 {
            throw ValueError(f"a subplot grid needs positive counts, got {rows}x{cols}")
        }
        let area = self.device()
        let cell_width = (area.width() - PANEL_GAP * (cols - 1).to_float()) / cols.to_float()
        let cell_height = (area.height() - PANEL_GAP * (rows - 1).to_float()) / rows.to_float()
        if cell_width <= 0.0 or cell_height <= 0.0 {
            throw ValueError(
                f"a {rows}x{cols} grid does not fit in {self.width}x{self.height} pixels"
            )
        }

        var grid = []
        for row in 0..rows {
            var line = []
            for column in 0..cols {
                let left = area.left() + (cell_width + PANEL_GAP) * column.to_float()
                let top = area.bottom() + (cell_height + PANEL_GAP) * row.to_float()
                line.push(self._panel(Rect(left, top, left + cell_width, top + cell_height)))
            }
            grid.push(line)
        }
        return grid
    }

    #: The whole figure rectangle, in device pixels.
    pub fn device(self) -> Rect {
        return Rect(0.0, 0.0, self.width.to_float(), self.height.to_float())
    }

    #: The theme this figure and its axes were created in.
    pub fn theme_of(self) -> Theme {
        return self.theme
    }

    #: Every set of axes on the figure, in creation order.
    pub fn all_axes(self) -> list[Axes] {
        return [panel for panel in self.panels]
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
        for (index, panel) in self.panels.enumerate() {
            list.extend(panel.render_dynamic(self.regions[index]))
        }
        list.validate()
        return list
    }

    #: Render the static and dynamic layers of every panel, separately.
    pub fn render_layers(self) -> tuple[DisplayList, DisplayList] {
        let background = DisplayList()
        let foreground = DisplayList()
        for (index, panel) in self.panels.enumerate() {
            let (static_list, dynamic_list) = panel.render_layers(self.regions[index])
            background.extend(static_list)
            foreground.extend(dynamic_list)
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
        let list = DisplayList()
        for (index, panel) in self.panels.enumerate() {
            list.extend(panel.render(self.regions[index]))
        }
        list.validate()
        return list
    }

    #: Add a set of axes covering `region` and remember where it goes.
    fn _panel(self, region: Rect) -> Axes {
        let panel = Axes(theme: self.theme)
        self.panels.push(panel)
        self.regions.push(region)
        return panel
    }

    fn __str__(self) -> str {
        return f"Figure({self.width}x{self.height}, {self.panels.len()} axes)"
    }
}
