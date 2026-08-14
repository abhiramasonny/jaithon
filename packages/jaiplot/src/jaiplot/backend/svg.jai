#: The SVG backend: a display list in, an SVG document out.
#:
#: Nothing here rasterises. Every primitive has a direct SVG counterpart, so the
#: whole backend is one exhaustive `match` and a handful of attribute writers.
#: That is the point of the display list: adding a backend costs one file.
#:
#: Clips become nested `<g clip-path=...>` groups, which is exactly the stack
#: discipline `PushClip` and `PopClip` describe.
import std.math
from std.algo.codecs import base64_encode
from std.fmt import format
from std.gui.image import Image

from ..geom import Point, Rect, cross_arms, pose_outline
from ..metrics import text_height
from ..primitive import DisplayList, Primitive
from .png import encode
from ..style import Color, HorizontalAnchor, LineCap, LineJoin, MarkerShape
from ..style import Paint, Stroke, TextStyle, VerticalAnchor

#: Decimal places coordinates are written to in the document.
#:
#: Two is below the resolution of any display and keeps the file small; going
#: further would only record the noise in the last bits of a float.
pub const COORDINATE_PRECISION = 2

#: The font stack SVG text is asked for.
#:
#: Monospace, because layout measured the labels with the canvas's fixed-pitch
#: metric and the two must agree about how wide a label is.
pub const FONT_FAMILY = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

#: Render `list` as a complete SVG document `width` by `height` pixels.
#:
#: The list is validated first, so an unbalanced clip is reported here rather
#: than producing a document with a stray `</g>` in it.
pub fn to_svg(list: DisplayList, width: int, height: int) -> str {
    if width <= 0 or height <= 0 {
        throw ValueError(f"an SVG document must have positive size, got {width}x{height}")
    }
    list.validate()

    var parts = []
    parts.push(f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\"")
    parts.push(f" viewBox=\"0 0 {width} {height}\">\n")

    var clips = []
    var body = []
    for primitive in list.to_list() {
        match primitive {
            Primitive.PushClip(rect) => {
                let name = f"clip{clips.len()}"
                clips.push(_clip_path(name, rect))
                body.push(f"<g clip-path=\"url(#{name})\">\n")
            }
            Primitive.PopClip => { body.push("</g>\n") }
            _ => { body.push(_element(primitive) + "\n") }
        }
    }

    if clips.len() > 0 {
        parts.push("<defs>\n")
        for clip in clips { parts.push(clip) }
        parts.push("</defs>\n")
    }
    for line in body { parts.push(line) }
    parts.push("</svg>\n")
    return "".join(parts)
}

#: One primitive as one SVG element.
fn _element(primitive: Primitive) -> str {
    return match primitive {
        Primitive.Polyline(points, stroke) => _polyline(points, stroke),
        Primitive.Polygon(points, fill, edge) => _polygon(points, fill, edge),
        Primitive.Text(at, content, style) => _text(at, content, style),
        Primitive.Marker(at, shape, size, fill, edge) => _marker(at, shape, size, fill, edge),
        Primitive.Points(points, shape, size, fill) => _points_batch(points, shape, size, fill),
        Primitive.Poses(points, angles, size, fill) => _poses(points, angles, size, fill),
        Primitive.Segments(points, stroke) => _segments(points, stroke),
        Primitive.Raster(rect, pixels, width, height) => _raster(rect, pixels, width, height),
        Primitive.PushClip(_rect) => "",
        Primitive.PopClip => "",
    }
}

fn _polyline(points: list[Point], stroke: Stroke) -> str {
    if points.len() < 2 or not stroke.is_visible() { return "" }
    return f"<polyline points=\"{_points(points)}\" fill=\"none\" {_stroke_attrs(stroke)}/>"
}

fn _polygon(points: list[Point], fill: Paint, edge: Stroke?) -> str {
    if points.len() < 3 { return "" }
    let paint = _fill_attrs(fill)
    let outline = if edge == null { "stroke=\"none\"" } else { _stroke_attrs(edge) }
    return f"<polygon points=\"{_points(points)}\" {paint} {outline}/>"
}

fn _text(at: Point, content: str, style: TextStyle) -> str {
    if content.len() == 0 { return "" }
    let size = _font_size(style)
    let anchor = match style.horizontal {
        HorizontalAnchor.Start => "start",
        HorizontalAnchor.Middle => "middle",
        HorizontalAnchor.End => "end",
    }
    let cell = text_height(content, style.scale)
    let baseline = match style.vertical {
        VerticalAnchor.Top => cell,
        VerticalAnchor.Middle => cell * 0.5,
        VerticalAnchor.Baseline => 0.0,
        VerticalAnchor.Bottom => 0.0,
    }
    let fill = _color_attrs(style.color, "fill")
    let degrees = -90 * style.quarter_turns % 360
    var frame = f"translate({_number(at.x)} {_number(at.y)})"
    if degrees != 0 { frame += f" rotate({degrees})" }
    let position = f"x=\"0\" y=\"{_number(baseline)}\""
    let font = f"font-family=\"{FONT_FAMILY}\" font-size=\"{_number(size)}\""
    let attributes = f"{position} {font} text-anchor=\"{anchor}\" {fill}"
    return f"<text {attributes} transform=\"{frame}\">{_escape(content)}</text>"
}

fn _marker(at: Point, shape: MarkerShape, size: float, fill: Paint, edge: Stroke?) -> str {
    if size <= 0.0 or not math.is_finite(size) { return "" }
    let paint = _fill_attrs(fill)
    let outline = if edge == null { "stroke=\"none\"" } else { _stroke_attrs(edge) }
    let radius = size * 0.5
    if shape == MarkerShape.Circle {
        let centre = f"cx=\"{_number(at.x)}\" cy=\"{_number(at.y)}\" r=\"{_number(radius)}\""
        return f"<circle {centre} {paint} {outline}/>"
    }
    if shape == MarkerShape.Plus or shape == MarkerShape.Cross {
        let arms = cross_arms(at, radius, shape == MarkerShape.Cross)
        let line_stroke = Stroke(fill.color(), width: max(1.0, size * 0.22), cap: LineCap.Butt)
        var out = []
        for arm in arms {
            out.push(
                f"<polyline points=\"{_points(arm)}\" fill=\"none\" {_stroke_attrs(line_stroke)}/>"
            )
        }
        return "".join(out)
    }
    return f"<polygon points=\"{_points(_marker_points(at, radius, shape))}\" {paint} {outline}/>"
}

#: A whole batch of markers as one path.
#:
#: One element per particle would make a filter's output a document of many
#: thousands of nodes that no viewer opens happily. Every marker in a batch
#: shares its shape and paint, so they share one `<path>` too.
fn _points_batch(points: list[Point], shape: MarkerShape, size: float, fill: Paint) -> str {
    if points.len() == 0 or size <= 0.0 { return "" }
    let radius = size * 0.5
    var parts = []
    for point in points {
        parts.push(_closed_subpath(_marker_points(point, radius, shape)))
    }
    return f"<path d=\"{parts.join(" ")}\" {_fill_attrs(fill)} stroke=\"none\"/>"
}

#: A batch of oriented wedges, one subpath each.
fn _poses(points: list[Point], angles: list[float], size: float, fill: Paint) -> str {
    if points.len() == 0 or size <= 0.0 { return "" }
    var parts = []
    for (index, point) in points.enumerate() {
        let angle = if index < angles.len() { angles[index] } else { 0.0 }
        parts.push(_closed_subpath(pose_outline(point, size * 0.5, angle)))
    }
    return f"<path d=\"{parts.join(" ")}\" {_fill_attrs(fill)} stroke=\"none\"/>"
}

#: Disconnected segments as one path of move-line pairs.
fn _segments(points: list[Point], stroke: Stroke) -> str {
    if points.len() < 2 or not stroke.is_visible() { return "" }
    var parts = []
    var index = 0
    while index + 1 < points.len() {
        let start = points[index]
        let end = points[index + 1]
        parts.push(f"M{_number(start.x)} {_number(start.y)}L{_number(end.x)} {_number(end.y)}")
        index += 2
    }
    return f"<path d=\"{parts.join("")}\" fill=\"none\" {_stroke_attrs(stroke)}/>"
}

#: A pixel buffer as an embedded PNG.
#:
#: An occupancy grid is tens of thousands of cells; as rectangles it would dwarf
#: the rest of the document, so it goes in as an image the way a raster belongs
#: in a vector format.
fn _raster(rect: Rect, pixels: list[int], width: int, height: int) -> str {
    if width <= 0 or height <= 0 or pixels.len() < width * height { return "" }
    let image = Image(width, height)
    let buffer = image.pixels()
    for index in 0..width * height { buffer[index] = pixels[index] }
    let encoded = base64_encode(encode(image).to_list())
    let box = f"x=\"{_number(rect.left())}\" y=\"{_number(rect.bottom())}\""
    let size = f"width=\"{_number(rect.width())}\" height=\"{_number(rect.height())}\""
    let rendering = "image-rendering=\"pixelated\""
    return f"<image {box} {size} {rendering} href=\"data:image/png;base64,{encoded}\"/>"
}

#: One closed subpath of a `d` attribute.
fn _closed_subpath(points: list[Point]) -> str {
    if points.len() == 0 { return "" }
    var out = f"M{_number(points[0].x)} {_number(points[0].y)}"
    for index in 1..points.len() {
        out += f"L{_number(points[index].x)} {_number(points[index].y)}"
    }
    return out + "Z"
}

#: The outline of a filled marker shape.
fn _marker_points(at: Point, radius: float, shape: MarkerShape) -> list[Point] {
    if shape == MarkerShape.Square {
        return [
            Point(at.x - radius, at.y - radius),
            Point(at.x + radius, at.y - radius),
            Point(at.x + radius, at.y + radius),
            Point(at.x - radius, at.y + radius),
        ]
    }
    if shape == MarkerShape.Diamond {
        return [
            Point(at.x, at.y - radius),
            Point(at.x + radius, at.y),
            Point(at.x, at.y + radius),
            Point(at.x - radius, at.y),
        ]
    }
    if shape == MarkerShape.Star {
        var points = []
        # Ten alternating radii make the five points; the inner radius is the
        # golden ratio's reciprocal squared, which is the classic star.
        for index in 0..10 {
            let reach = if index % 2 == 0 { radius } else { radius * 0.382 }
            let angle = -math.PI * 0.5 + math.PI * index.to_float() / 5.0
            points.push(Point(at.x + reach * math.cos(angle), at.y + reach * math.sin(angle)))
        }
        return points
    }
    return [
        Point(at.x, at.y - radius),
        Point(at.x + radius, at.y + radius * 0.8),
        Point(at.x - radius, at.y + radius * 0.8),
    ]
}

fn _clip_path(name: str, rect: Rect) -> str {
    let box = f"x=\"{_number(rect.left())}\" y=\"{_number(rect.bottom())}\""
    let size = f"width=\"{_number(rect.width())}\" height=\"{_number(rect.height())}\""
    return f"<clipPath id=\"{name}\"><rect {box} {size}/></clipPath>\n"
}

fn _points(points: list[Point]) -> str {
    return [f"{_number(point.x)},{_number(point.y)}" for point in points].join(" ")
}

#: Stroke colour, width, dash and joins as SVG attributes.
fn _stroke_attrs(stroke: Stroke) -> str {
    let cap = match stroke.cap {
        LineCap.Butt => "butt",
        LineCap.Round => "round",
        LineCap.Square => "square",
    }
    let join = match stroke.join {
        LineJoin.Miter => "miter",
        LineJoin.Round => "round",
        LineJoin.Bevel => "bevel",
    }
    var out = f"{_color_attrs(stroke.color, "stroke")} stroke-width=\"{_number(stroke.width)}\""
    out += f" stroke-linecap=\"{cap}\" stroke-linejoin=\"{join}\""
    if stroke.dash.len() > 0 {
        let pattern = [_number(length) for length in stroke.dash].join(",")
        out += f" stroke-dasharray=\"{pattern}\""
    }
    return out
}

fn _fill_attrs(paint: Paint) -> str {
    return _color_attrs(paint.color(), "fill")
}

#: A colour as a `name` attribute, with opacity split out.
#:
#: SVG carries alpha in a separate attribute rather than in the hex, and
#: omitting it when the colour is opaque keeps the document readable.
fn _color_attrs(color: Color, name: str) -> str {
    let hex = color.to_hex()
    let opacity = color.opacity()
    if opacity >= 1.0 { return f"{name}=\"{hex}\"" }
    return f"{name}=\"{hex}\" {name}-opacity=\"{_number(opacity)}\""
}

#: The nominal font size that matches the canvas metric at this scale.
fn _font_size(style: TextStyle) -> float {
    return (style.scale * 10).to_float() * 0.7
}

fn _number(value: float) -> str {
    if not math.is_finite(value) { return "0" }
    let text = format("{:." + str(COORDINATE_PRECISION) + "f}", value)
    if text.contains(".") {
        var trimmed = text
        while trimmed.ends_with("0") { trimmed = trimmed[0:trimmed.len() - 1] }
        if trimmed.ends_with(".") { trimmed = trimmed[0:trimmed.len() - 1] }
        if trimmed == "-0" { return "0" }
        return trimmed
    }
    return text
}

#: Escape the five characters that cannot appear literally in XML text.
fn _escape(content: str) -> str {
    var out = content.replace("&", "&amp;")
    out = out.replace("<", "&lt;").replace(">", "&gt;")
    return out.replace("\"", "&quot;").replace("'", "&apos;")
}
