#: Plotting for Jaithon.
#:
#: A figure holds axes, axes hold series, and rendering reduces the lot to a
#: display list of primitives that a backend turns into a file. Nothing in the
#: library draws until a backend is asked to.
#:
#: ```
#: let figure = Figure(width: 480, height: 320)
#: let axes = figure.axes()
#: axes.line([0.0, 1.0, 2.0], [0.0, 1.0, 4.0], label: "squares")
#: axes.set_title("a first plot")
#: axes.grid(true)
#: save(figure, "/tmp/first.svg")
#: ```
import std.io
from std.gui.color import blue_of, green_of, red_of
from std.gui.image import Image

from .axes import Axes
from .backend import encode, render_image, to_svg
from .data import arange, extent, linspace, logspace, minmax, percentile
from .figure import Figure
from .frame import Frame
from .geom import Point, Rect, Transform
from .legend import LegendCorner
from .metrics import text_height, text_width
from .primitive import DisplayList, Primitive, describe
from .scale import LinearScale, LogScale, Scale, scale_named
from .series import Bar, FillBetween, Grid, Hist, Line, PoseCloud, Rays, Scatter, Series, Step
from .style import Color, ColorCycle, HorizontalAnchor, LineCap, LineJoin
from .style import MarkerShape, Paint, Stroke, TextStyle, Theme, VerticalAnchor

export { Axes, Figure, Frame, DisplayList, Primitive, describe }
export { Point, Rect, Transform, LegendCorner }
export { arange, extent, linspace, logspace, minmax, percentile }
export { LinearScale, LogScale, Scale, scale_named }
export { Bar, FillBetween, Hist, Line, Scatter, Series, Step }
export { Grid, PoseCloud, Rays }
export { Color, ColorCycle, MarkerShape, Paint, Stroke, TextStyle, Theme }
export { HorizontalAnchor, VerticalAnchor, LineCap, LineJoin }
export { text_height, text_width, to_svg }

#: The file extensions `save` understands.
pub let FORMATS: list[str] = ["svg", "png", "ppm"]

#: Render `figure` and write it to `path`, choosing the format by extension.
#:
#: Raises `ValueError` naming the formats that exist when the extension is not
#: one of them, since guessing would write a file the caller cannot open.
#:
#: ```
#: import jaiplot
#: let figure = Figure(width: 200, height: 150)
#: figure.axes().line([0.0, 1.0], [0.0, 1.0])
#: jaiplot.render_svg(figure).starts_with("<svg")     # true
#: ```
pub fn save(figure: Figure, path: str) -> void {
    let dot = path.rfind(".")
    if dot < 0 {
        throw ValueError(f"cannot tell the format of {path}: it has no extension")
    }
    let suffix = path[dot + 1:].lower()
    if suffix == "svg" {
        io.write_file(path, render_svg(figure))
        return
    }
    if suffix == "png" {
        io.write_bytes(path, encode(render_raster(figure)))
        return
    }
    if suffix == "ppm" {
        io.write_file(path, _to_ppm(render_raster(figure)))
        return
    }
    throw ValueError(f"unsupported format {suffix}, expected one of {FORMATS.join(", ")}")
}

#: Render `figure` to an offscreen image.
#:
#: No window is involved, so this works on a machine with no display.
pub fn render_raster(figure: Figure) -> Image {
    return render_image(figure.render(), figure.width, figure.height)
}

#: `image` as a plain-text PPM.
#:
#: PPM earns its place by being the format every image tool reads and nothing
#: has to encode: three decimal numbers a pixel and no compression at all.
fn _to_ppm(image: Image) -> str {
    let width = image.width()
    let height = image.height()
    let pixels = image.pixels()
    var parts = [f"P3\n{width} {height}\n255\n"]
    for row in 0..height {
        var line = []
        for column in 0..width {
            let packed = pixels[row * width + column]
            line.push(f"{red_of(packed)} {green_of(packed)} {blue_of(packed)}")
        }
        parts.push(line.join(" ") + "\n")
    }
    return "".join(parts)
}

#: Render `figure` as an SVG document.
pub fn render_svg(figure: Figure) -> str {
    return to_svg(figure.render(), figure.width, figure.height)
}

export { FORMATS, save, render_svg, render_raster, encode }
