#: A series drawn as one marker per sample, with nothing joining them.
#:
#: Use it where the samples are independent observations rather than a path
#: through time — the thing a line would falsely imply.
import std.math

from .base import Series
from ..frame import Frame
from ..geom import Rect
from ..primitive import Primitive
from ..style import Color, MarkerShape, Paint, Stroke

from .line import carried_opacity, finite_extent

#: Marker diameter in device pixels when none is given.
pub const DEFAULT_SIZE: float = 6.0

#: Independent samples, one marker each.
#:
#: Sizes may vary per point: pass a `sizes` list as long as the data and each
#: marker takes its own. That is how a third variable is shown without a third
#: axis.
#:
#: ```
#: let points = Scatter([0.0, 1.0], [2.0, 3.0], label: "runs")
#: points.label()                   # "runs"
#: points.extent().right()          # 1.0
#: ```
pub class Scatter: Series {
    var xs: list[float]
    var ys: list[float]
    var sizes: list[float]
    var name: str?
    var shape: MarkerShape
    var fill: Paint
    var edge: Stroke?
    var chosen: bool
    var opacity: float

    #: Plot `ys` against `xs` as markers.
    #:
    #: A null `opacity` means "whatever the colour says". This used to default
    #: to 1.0 and be applied unconditionally, so a colour handed in at 25%
    #: alpha — the usual way of making an overplotted cloud readable — came
    #: back fully opaque and hid exactly what it was asked to reveal.
    #:
    #: Raises `ValueError` when the lists differ in length, or when `sizes` is
    #: given and does not match them.
    pub fn init(
        self,
        xs: list[float],
        ys: list[float],
        label: str? = null,
        color: Color? = null,
        size: float = DEFAULT_SIZE,
        sizes: list[float] = [],
        shape: MarkerShape = MarkerShape.Circle,
        opacity: float? = null,
        edge: Stroke? = null
    ) {
        if xs.len() != ys.len() {
            throw ValueError(f"x and y must be the same length, got {xs.len()} and {ys.len()}")
        }
        if sizes.len() > 0 and sizes.len() != xs.len() {
            throw ValueError(f"sizes must match the data, got {sizes.len()} for {xs.len()} points")
        }
        self.xs = [value for value in xs]
        self.ys = [value for value in ys]
        self.sizes = if sizes.len() > 0 {
            [value for value in sizes]
        } else {
            [size for _index in 0..xs.len()]
        }
        self.name = label
        self.shape = shape
        self.opacity = carried_opacity(color, opacity, 1.0)
        self.chosen = color != null
        self.fill = Paint.solid((color ?? Color.from_rgb(0, 0, 0)).with_alpha(self.opacity))
        self.edge = edge
    }

    #: Override the colour, after construction.
    #:
    #: A colour with an alpha of its own replaces the series opacity too;
    #: otherwise the opacity in force is kept, so restyling a translucent
    #: cloud with an opaque colour does not silently make it solid.
    pub fn color(self, value: Color) -> Scatter {
        self.opacity = carried_opacity(value, null, self.opacity)
        self.fill = Paint.solid(value.with_alpha(self.opacity))
        self.chosen = true
        return self
    }

    #: Replace the data, keeping the styling and the per-point sizes uniform.
    #:
    #: Raises `ValueError` when the two lists differ in length.
    pub fn set_data(self, xs: list[float], ys: list[float]) -> void {
        if xs.len() != ys.len() {
            throw ValueError(f"x and y must be the same length, got {xs.len()} and {ys.len()}")
        }
        let previous = if self.sizes.len() > 0 { self.sizes[0] } else { DEFAULT_SIZE }
        self.xs = [value for value in xs]
        self.ys = [value for value in ys]
        self.sizes = [previous for _index in 0..xs.len()]
    }

    #: The box the samples occupy, or null when none can be drawn.
    pub fn extent(self) -> Rect? {
        return finite_extent(self.xs, self.ys)
    }

    #: One marker per placeable sample.
    pub fn draw(self, frame: Frame) -> list[Primitive] {
        var out = []
        for (index, x) in self.xs.enumerate() {
            let placed = frame.place(x, self.ys[index])
            if placed == null { continue }
            let size = self.sizes[index]
            if not math.is_finite(size) or size <= 0.0 { continue }
            out.push(Primitive.Marker(placed, self.shape, size, self.fill, self.edge))
        }
        return out
    }

    #: The legend entry, or null when the series was not labelled.
    pub fn label(self) -> str? {
        return self.name
    }

    #: A single marker at the centre of the swatch box.
    pub fn swatch(self, box: Rect) -> list[Primitive] {
        let size = min(DEFAULT_SIZE, box.height())
        return [Primitive.Marker(box.center(), self.shape, size, self.fill, self.edge)]
    }

    #: Take the cycle's colour unless one was chosen explicitly.
    pub fn adopt(self, color: Color) -> void {
        if not self.chosen { self.fill = Paint.solid(color.with_alpha(self.opacity)) }
    }

    fn __str__(self) -> str {
        return f"Scatter({self.xs.len()} points, {self.name ?? "unlabelled"})"
    }
}
