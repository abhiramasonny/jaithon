#: The drawing primitives a figure reduces to, and the list that collects them.
#:
#: `Figure` and `Axes` do not draw. They produce a `DisplayList` of `Primitive`
#: values in device pixels, and a backend turns that list into an SVG document
#: or into pixels on a canvas. Every backend sees the same list, so two backends
#: cannot disagree about what a figure contains — only about how to draw it.
#:
#: The list is also where regression tests bite. `DisplayList.render()` prints
#: it deterministically, at fixed precision, so a snapshot of a figure is a
#: readable diff rather than an image comparison.
#:
#: ```
#: from jaiplot.geom import Point
#: from jaiplot.style import Color, Paint, Stroke
#: let ink = Stroke(Color.from_rgb(0, 0, 0))
#: let list = DisplayList()
#: list.push(Primitive.Polyline([Point(0.0, 0.0), Point(4.0, 2.0)], ink))
#: list.len()                        # 1
#: ```
from std.core import Iterable, Iterator
from std.fmt import format

from .geom import Point, Rect
from .style import MarkerShape, Paint, Stroke, TextStyle

#: How many decimal places a coordinate is printed to by `render`.
#:
#: Three is enough to distinguish any two device positions a layout can
#: produce and few enough that an irrelevant difference in the last bits of a
#: float does not show up as a failing snapshot.
pub const RENDER_PRECISION = 3

#: One drawable thing, in device pixels.
#:
#: Device coordinates grow rightward and downward, matching the canvas and SVG.
#: `Transform` has already flipped the y axis by the time a primitive exists, so
#: nothing downstream thinks about data orientation.
#:
#: `PushClip` and `PopClip` bracket a region: a backend intersects the pushed
#: rectangle with whatever is already in force, and `PopClip` restores the
#: previous one. They must balance, which `DisplayList.validate` checks.
pub enum Primitive {
    #: A connected run of line segments, stroked but never filled.
    Polyline(points: list[Point], stroke: Stroke),

    #: A closed shape, filled and optionally outlined.
    Polygon(points: list[Point], fill: Paint, edge: Stroke?),

    #: A run of text anchored at a point, as `style` describes.
    Text(at: Point, content: str, style: TextStyle),

    #: A data marker centred on a point, `size` pixels across.
    Marker(at: Point, shape: MarkerShape, size: float, fill: Paint, edge: Stroke?),

    #: Many identical markers in one primitive.
    #:
    #: A particle filter draws thousands of these a frame. One primitive per
    #: cloud rather than one per particle is what lets a backend rasterise them
    #: in a tight loop instead of paying per-shape setup thousands of times.
    Points(points: list[Point], shape: MarkerShape, size: float, fill: Paint),

    #: Many markers that each carry a heading, drawn as oriented wedges.
    #:
    #: `angles` are in radians, measured anticlockwise from the positive x axis
    #: in data space, and must be as long as `points`.
    Poses(points: list[Point], angles: list[float], size: float, fill: Paint),

    #: Disconnected line segments, taken from `points` two at a time.
    #:
    #: A laser scan is a few hundred rays that share nothing but their origin,
    #: and a polyline would join them into one continuous zigzag.
    Segments(points: list[Point], stroke: Stroke),

    #: A pixel buffer placed into a rectangle, for an occupancy grid or a
    #: heatmap.
    #:
    #: `pixels` is `width * height` packed `0xAARRGGBB` values in row-major
    #: order, the same layout `std.gui.canvas` uses.
    Raster(rect: Rect, pixels: list[int], width: int, height: int),

    #: Restrict everything until the matching `PopClip` to this rectangle.
    PushClip(rect: Rect),

    #: Restore the clip rectangle in force before the matching `PushClip`.
    PopClip,
}

#: An ordered sequence of primitives, in the order they are drawn.
#:
#: Later primitives paint over earlier ones. A figure builds one of these per
#: render; nothing in it is retained between renders, so a display list is a
#: description of one picture rather than a scene graph.
#:
#: ```
#: let list = DisplayList()
#: list.len()          # 0
#: list.is_empty()     # true
#: ```
pub class DisplayList: Iterable[Primitive] {
    var entries: list[Primitive]

    #: Create an empty display list.
    pub fn init(self) {
        self.entries = []
    }

    #: Build a display list from primitives that already exist.
    pub static fn from_list(primitives: list[Primitive]) -> DisplayList {
        let list = DisplayList()
        for primitive in primitives { list.push(primitive) }
        return list
    }

    #: Append one primitive.
    pub fn push(self, primitive: Primitive) -> void {
        self.entries.push(primitive)
    }

    #: Append every primitive of `other`, in order.
    pub fn extend(self, other: DisplayList) -> void {
        for primitive in other.entries { self.entries.push(primitive) }
    }

    #: Append every primitive of a plain list, in order.
    pub fn extend_list(self, primitives: list[Primitive]) -> void {
        for primitive in primitives { self.entries.push(primitive) }
    }

    #: Number of primitives held.
    pub fn len(self) -> int {
        return self.entries.len()
    }

    #: Report whether nothing has been pushed.
    pub fn is_empty(self) -> bool {
        return self.entries.len() == 0
    }

    #: The primitives, as a fresh list.
    pub fn to_list(self) -> list[Primitive] {
        return [primitive for primitive in self.entries]
    }

    #: Iterate the primitives in drawing order.
    pub fn iter(self) -> Iterator[Primitive] {
        return self.entries.iter()
    }

    #: Raise `ValueError` when the clip brackets do not balance.
    #:
    #: A backend maintains a clip stack while it walks the list and would either
    #: underflow on a stray `PopClip` or leak a clip past the end of the figure.
    #: Checking once here means neither backend has to defend against it.
    pub fn validate(self) -> void {
        var depth = 0
        for (index, primitive) in self.entries.enumerate() {
            match primitive {
                Primitive.PushClip(_) => { depth += 1 }
                Primitive.PopClip => {
                    depth -= 1
                    if depth < 0 {
                        throw ValueError(f"PopClip at index {index} has no matching PushClip")
                    }
                }
                _ => { }
            }
        }
        if depth != 0 {
            throw ValueError(f"{depth} PushClip left unclosed at the end of the display list")
        }
    }

    #: Render the list as deterministic text, one primitive per line.
    #:
    #: Coordinates print to `RENDER_PRECISION` decimal places and colours as hex,
    #: so the same figure always produces the same string on any machine. This is
    #: what golden tests compare.
    pub fn render(self) -> str {
        let lines = [describe(primitive) for primitive in self.entries]
        return lines.join("\n")
    }

    fn __len__(self) -> int {
        return self.entries.len()
    }

    fn __str__(self) -> str {
        return f"DisplayList({self.entries.len()} primitives)"
    }
}

#: Describe one primitive as a single deterministic line.
#:
#: ```
#: from jaiplot.geom import Rect
#: describe(Primitive.PushClip(Rect(0.0, 0.0, 4.0, 2.0)))
#: # "PushClip 0.000,0.000 4.000x2.000"
#: ```
pub fn describe(primitive: Primitive) -> str {
    return match primitive {
        Primitive.Polyline(points, stroke) => f"Polyline [{_points(points)}] {_stroke(stroke)}",
        Primitive.Polygon(points, fill, edge) => f"Polygon [{_points(points)}] fill={_paint(fill)} edge={_optional_stroke(edge)}",
        Primitive.Text(at, content, style) => f"Text {_point(at)} {_quote(content)} {_text_style(style)}",
        Primitive.Marker(at, shape, size, fill, edge) => f"Marker {_point(at)} {_variant(shape)} size={_number(size)} fill={_paint(fill)} edge={_optional_stroke(edge)}",
        Primitive.Points(points, shape, size, fill) => f"Points x{points.len()} {_variant(shape)} size={_number(size)} fill={_paint(fill)} {_bounds(points)}",
        Primitive.Poses(points, angles, size, fill) => f"Poses x{points.len()} angles={angles.len()} size={_number(size)} fill={_paint(fill)} {_bounds(points)}",
        Primitive.Segments(points, stroke) => f"Segments x{points.len() // 2} {_stroke(stroke)} {_bounds(points)}",
        Primitive.Raster(rect, _pixels, width, height) => f"Raster {width}x{height} at {_number(rect.left())},{_number(rect.bottom())} {_number(rect.width())}x{_number(rect.height())}",
        Primitive.PushClip(rect) => f"PushClip {_number(rect.left())},{_number(rect.bottom())} {_number(rect.width())}x{_number(rect.height())}",
        Primitive.PopClip => "PopClip",
    }
}

#: The bounding box of a batch, so a snapshot stays short but still changes when
#: the batch moves.
fn _bounds(points: list[Point]) -> str {
    let box = Rect.from_points(points)
    if box == null { return "[]" }
    let low = f"{_number(box.left())},{_number(box.bottom())}"
    let high = f"{_number(box.right())},{_number(box.top())}"
    return f"[{low} {high}]"
}

#: Format one float at `RENDER_PRECISION`, without a signed zero.
#:
#: `-0.0` and `0.0` are the same position but print differently, and which one a
#: layout produces depends on the sign of a subtraction that rounded to nothing.
#: Collapsing them keeps a snapshot from failing over a difference that cannot
#: be seen.
fn _number(value: float) -> str {
    let text = format("{:." + str(RENDER_PRECISION) + "f}", value)
    if text.starts_with("-") and text.to_float() == 0.0 { return text[1:] }
    return text
}

fn _point(point: Point) -> str {
    return f"{_number(point.x)},{_number(point.y)}"
}

fn _points(points: list[Point]) -> str {
    return [_point(point) for point in points].join(" ")
}

fn _paint(paint: Paint) -> str {
    return paint.color().to_hex()
}

fn _stroke(stroke: Stroke) -> str {
    let dash = [_number(length) for length in stroke.dash].join(",")
    let pattern = if dash.len() > 0 { f" dash={dash}" } else { "" }
    return f"{stroke.color.to_hex()} width={_number(stroke.width)}{pattern}"
}

fn _optional_stroke(stroke: Stroke?) -> str {
    if stroke == null { return "none" }
    return _stroke(stroke)
}

fn _text_style(style: TextStyle) -> str {
    let anchor = f"{_variant(style.horizontal)}/{_variant(style.vertical)}"
    return f"{style.color.to_hex()} scale={style.scale} {anchor} turns={style.quarter_turns}"
}

#: Print an enum value as its bare variant, dropping the type that qualifies it.
#:
#: A snapshot reads better as `Diamond` than as `MarkerShape.Diamond`, and the
#: type is already obvious from the field it sits in.
fn _variant(value: any) -> str {
    let text = str(value)
    let dot = text.find(".")
    if dot < 0 { return text }
    return text[dot + 1:]
}

#: Quote a string so a newline or a quote inside a label cannot break the
#: one-primitive-per-line shape a snapshot relies on.
fn _quote(content: str) -> str {
    let escaped = content.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n")
    return f"\"{escaped}\""
}
