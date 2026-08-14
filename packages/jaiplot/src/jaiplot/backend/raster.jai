#: The raster backend: a display list onto a pixel buffer.
#:
#: The counterpart of `svg.jai`, walking the same list and reaching the same
#: geometry through `std.gui.canvas`. Everything is anti-aliased, so a chart
#: rendered here and the same chart rendered to SVG differ only in resolution.
#:
#: It needs no window. `render` takes any `PixelTarget`, and an offscreen
#: `Image` is one, which is what makes a raster figure testable on a machine
#: with no display attached.
#:
#: ```
#: from jaiplot import Figure
#: let figure = Figure(width: 120, height: 90)
#: figure.axes().line([0.0, 1.0], [0.0, 1.0])
#: let image = render_image(figure.render(), 120, 90)
#: image.width()      # 120
#: ```
import std.math
from std.gui.canvas import Canvas
from std.gui.color import rgb
from std.gui.image import Image
from std.gui.path import Cap, Join, Path, Winding

from ..geom import Point, Rect, pose_outline
from ..metrics import text_height, text_width
from ..primitive import DisplayList, Primitive
from ..style import HorizontalAnchor, LineCap, LineJoin, MarkerShape, Paint
from ..style import Stroke, TextStyle, VerticalAnchor

#: Marker size at or below which a batch is drawn as boxes rather than shapes.
#:
#: At four pixels a diamond and a square differ by less than one pixel of
#: coverage, and a filter draws thousands of them.
pub const SMALL_MARKER: float = 4.0

#: Render `list` into a fresh `Image` of this size.
#:
#: Raises `ValueError` on a non-positive size, by way of `Image`.
pub fn render_image(list: DisplayList, width: int, height: int) -> Image {
    let image = Image(width, height, rgb(255, 255, 255))
    draw(list, Canvas(image))
    return image
}

#: Draw `list` onto `canvas`.
#:
#: The list is validated first, so the clip stack cannot underflow part way
#: through and leave the canvas clipped to something arbitrary.
pub fn draw(list: DisplayList, canvas: Canvas) -> void {
    list.validate()
    for primitive in list.to_list() {
        match primitive {
            Primitive.Polyline(points, stroke) => _polyline(canvas, points, stroke),
            Primitive.Polygon(points, fill, edge) => _polygon(canvas, points, fill, edge),
            Primitive.Text(at, content, style) => _text(canvas, at, content, style),
            Primitive.Marker(at, shape, size, fill, edge) => {
                _marker(canvas, at, shape, size, fill, edge)
            }
            Primitive.Points(points, shape, size, fill) => {
                _points_batch(canvas, points, shape, size, fill)
            }
            Primitive.Poses(points, angles, size, fill) => {
                _poses(canvas, points, angles, size, fill)
            }
            Primitive.Segments(points, stroke) => _segments(canvas, points, stroke),
            Primitive.Raster(rect, pixels, width, height) => {
                _raster(canvas, rect, pixels, width, height)
            }
            Primitive.PushClip(rect) => {
                canvas.push_clip(
                    int(math.floor(rect.left())),
                    int(math.floor(rect.bottom())),
                    int(math.ceil(rect.width())),
                    int(math.ceil(rect.height()))
                )
            }
            Primitive.PopClip => { canvas.pop_clip() }
        }
    }
}

fn _polyline(canvas: Canvas, points: list[Point], stroke: Stroke) -> void {
    if points.len() < 2 or not stroke.is_visible() { return }
    let path = Path()
    let _started = path.move_to(points[0].x, points[0].y)
    for index in 1..points.len() {
        let _extended = path.line_to(points[index].x, points[index].y)
    }
    canvas.stroke_path(
        path,
        stroke.color.packed(),
        width: stroke.width,
        cap: _cap(stroke.cap),
        join: _join(stroke.join),
        dash: stroke.dash
    )
}

fn _polygon(canvas: Canvas, points: list[Point], fill: Paint, edge: Stroke?) -> void {
    if points.len() < 3 { return }
    let path = Path()
    let _shape = path.polygon([(point.x, point.y) for point in points])
    if fill.color().alpha() > 0 {
        canvas.fill_path(path, fill.color().packed(), Winding.NonZero)
    }
    if edge == null { return }
    if not edge.is_visible() { return }
    var ring = [point for point in points]
    ring.push(points[0])
    _polyline(canvas, ring, edge)
}

fn _text(canvas: Canvas, at: Point, content: str, style: TextStyle) -> void {
    if content.len() == 0 { return }
    let width = text_width(content, style.scale)
    let height = text_height(content, style.scale)
    let turns = style.quarter_turns % 4

    let along = match style.horizontal {
        HorizontalAnchor.Start => 0.0,
        HorizontalAnchor.Middle => -width * 0.5,
        HorizontalAnchor.End => -width,
    }
    let across = match style.vertical {
        VerticalAnchor.Top => 0.0,
        VerticalAnchor.Middle => -height * 0.5,
        VerticalAnchor.Baseline => -height,
        VerticalAnchor.Bottom => -height,
    }

    let (dx, dy) = match turns {
        1 => (across, -along),
        2 => (-along, -across),
        3 => (-across, along),
        _ => (along, across),
    }
    canvas.text_rotated(
        int(at.x + dx + 0.5),
        int(at.y + dy + 0.5),
        content,
        style.color.packed(),
        turns,
        style.scale
    )
}

fn _marker(
    canvas: Canvas,
    at: Point,
    shape: MarkerShape,
    size: float,
    fill: Paint,
    edge: Stroke?
) -> void {
    if size <= 0.0 or not math.is_finite(size) { return }
    let radius = size * 0.5
    if shape == MarkerShape.Plus or shape == MarkerShape.Cross {
        let stroke = Stroke(fill.color(), width: max(1.0, size * 0.22))
        for arm in _cross_arms(at, radius, shape == MarkerShape.Cross) {
            _polyline(canvas, arm, stroke)
        }
        return
    }
    _polygon(canvas, _marker_points(at, radius, shape), fill, edge)
}

#: The outline of a filled marker shape, matching the SVG backend's.
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
    if shape == MarkerShape.Triangle {
        return [
            Point(at.x, at.y - radius),
            Point(at.x + radius, at.y + radius * 0.8),
            Point(at.x - radius, at.y + radius * 0.8),
        ]
    }
    if shape == MarkerShape.Star {
        var points = []
        for index in 0..10 {
            let reach = if index % 2 == 0 { radius } else { radius * 0.382 }
            let angle = -math.PI * 0.5 + math.PI * index.to_float() / 5.0
            points.push(Point(at.x + reach * math.cos(angle), at.y + reach * math.sin(angle)))
        }
        return points
    }
    var points = []
    let steps = max(8, min(32, int(radius * 4.0)))
    for index in 0..steps {
        let angle = math.TAU * index.to_float() / steps.to_float()
        points.push(Point(at.x + radius * math.cos(angle), at.y + radius * math.sin(angle)))
    }
    return points
}

fn _cross_arms(at: Point, radius: float, diagonal: bool) -> list[list[Point]] {
    if diagonal {
        let reach = radius * 0.7071
        return [
            [Point(at.x - reach, at.y - reach), Point(at.x + reach, at.y + reach)],
            [Point(at.x - reach, at.y + reach), Point(at.x + reach, at.y - reach)],
        ]
    }
    return [
        [Point(at.x - radius, at.y), Point(at.x + radius, at.y)],
        [Point(at.x, at.y - radius), Point(at.x, at.y + radius)],
    ]
}

#: A batch of identical markers.
#:
#: Below a few pixels across a marker is drawn as a filled box with exact edge
#: coverage rather than through the path rasteriser. A particle cloud is
#: thousands of these a frame, and the per-path setup would dominate the frame
#: at that count while being invisible at that size.
fn _points_batch(
    canvas: Canvas,
    points: list[Point],
    shape: MarkerShape,
    size: float,
    fill: Paint
) -> void {
    if points.len() == 0 or size <= 0.0 { return }
    let packed = fill.color().packed()
    if size <= SMALL_MARKER {
        let radius = size * 0.5
        for point in points {
            canvas.fill_rect_aa(point.x - radius, point.y - radius, size, size, packed)
        }
        return
    }
    for point in points {
        canvas.fill_convex(
            [(corner.x, corner.y) for corner in _marker_points(point, size * 0.5, shape)],
            packed
        )
    }
}

#: A batch of oriented wedges.
fn _poses(
    canvas: Canvas,
    points: list[Point],
    angles: list[float],
    size: float,
    fill: Paint
) -> void {
    if points.len() == 0 or size <= 0.0 { return }
    let packed = fill.color().packed()
    let radius = size * 0.5
    for (index, point) in points.enumerate() {
        let angle = if index < angles.len() { angles[index] } else { 0.0 }
        canvas.fill_convex(
            [(corner.x, corner.y) for corner in pose_outline(point, radius, angle)],
            packed
        )
    }
}

#: Disconnected segments, taken two points at a time.
fn _segments(canvas: Canvas, points: list[Point], stroke: Stroke) -> void {
    if points.len() < 2 or not stroke.is_visible() { return }
    let packed = stroke.color.packed()
    let half = stroke.width * 0.5

    if stroke.dash.len() == 0 {
        var index = 0
        while index + 1 < points.len() {
            let start = points[index]
            let end = points[index + 1]
            let dx = end.x - start.x
            let dy = end.y - start.y
            let length = math.hypot(dx, dy)
            if length > 0.0 {
                let nx = -dy / length * half
                let ny = dx / length * half
                canvas.fill_convex(
                    [
                        (start.x + nx, start.y + ny),
                        (end.x + nx, end.y + ny),
                        (end.x - nx, end.y - ny),
                        (start.x - nx, start.y - ny),
                    ],
                    packed
                )
            }
            index += 2
        }
        return
    }

    var index = 0
    while index + 1 < points.len() {
        let path = Path()
        let _from = path.move_to(points[index].x, points[index].y)
        let _to = path.line_to(points[index + 1].x, points[index + 1].y)
        canvas.stroke_path(
            path,
            packed,
            width: stroke.width,
            cap: _cap(stroke.cap),
            join: _join(stroke.join),
            dash: stroke.dash
        )
        index += 2
    }
}

#: A pixel buffer scaled into a device rectangle.
fn _raster(canvas: Canvas, rect: Rect, pixels: list[int], width: int, height: int) -> void {
    if width <= 0 or height <= 0 or pixels.len() < width * height { return }
    canvas.blit_pixels(
        pixels,
        width,
        height,
        int(rect.left() + 0.5),
        int(rect.bottom() + 0.5),
        max(1, int(rect.width() + 0.5)),
        max(1, int(rect.height() + 0.5))
    )
}

fn _cap(cap: LineCap) -> Cap {
    return match cap {
        LineCap.Butt => Cap.Butt,
        LineCap.Round => Cap.Round,
        LineCap.Square => Cap.Square,
    }
}

fn _join(join: LineJoin) -> Join {
    return match join {
        LineJoin.Miter => Join.Miter,
        LineJoin.Round => Join.Round,
        LineJoin.Bevel => Join.Bevel,
    }
}
