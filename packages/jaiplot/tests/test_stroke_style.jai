#: Tests for what a caller asks a stroke to look like: dash caps, dash
#: validation, marker edges and opacity.
#:
#: All four used to be quietly discarded. `Line` hardcoded a round cap, which
#: extends every dash by half the line width at each end and closed the gaps of
#: any pattern narrower than the line — jaiplot could not draw a dotted line at
#: all. `Line.dash` wrote past `Stroke.init`, so a negative length reached the
#: document. And four series overwrote the alpha of the colour they were given.
#:
#: The dash tests count ink runs in the rendered raster rather than trusting the
#: attributes, because "the SVG says butt" and "the picture has gaps in it" are
#: different claims and it is the second one that was failing.
from std.test import assert_eq, assert_near, assert_true, assert_false, assert_throws
from std.gui.color import blue_of, green_of, red_of
from std.gui.image import Image

from jaiplot import Figure, MarkerShape
from jaiplot.backend.raster import render_image
from jaiplot.backend.svg import to_svg
from jaiplot.frame import Frame
from jaiplot.geom import Rect
from jaiplot.primitive import Primitive
from jaiplot.scale import LinearScale
from jaiplot.series import FillBetween, Line, PoseCloud, Rays, Scatter
from jaiplot.style import Color, ColorCycle, LineCap, LineJoin, Stroke

#: The colour every series in this file is drawn in.
#:
#: Pure red on a white ground, so a pixel either belongs to the series or does
#: not and no threshold has to guess.
const INK: str = "#ff0000"

#: Assert `body` raises `kind`, as a statement.
fn expect_error(kind: any, body: fn() -> any) -> void {
    let _caught = assert_throws(kind, body)
}

#: A frame that maps the unit square onto a 200x100 device rectangle.
fn unit_frame() -> Frame {
    return Frame(
        Rect(0.0, 0.0, 1.0, 1.0),
        Rect(0.0, 0.0, 200.0, 100.0),
        LinearScale(),
        LinearScale()
    )
}

#: The stroke of the first `Polyline` in `primitives`.
#:
#: Raises `ValueError` when there is none, so a test that stops drawing the
#: line it is asking about fails loudly rather than skipping its assertions.
fn first_stroke(primitives: list[Primitive]) -> Stroke {
    for primitive in primitives {
        match primitive {
            Primitive.Polyline(_points, stroke) => { return stroke }
            _ => { }
        }
    }
    throw ValueError("expected a polyline among the primitives, found none")
}

#: The fill colour of the first primitive that carries one.
fn first_fill(primitives: list[Primitive]) -> Color {
    for primitive in primitives {
        match primitive {
            Primitive.Polygon(_points, fill, _edge) => { return fill.color() }
            Primitive.Marker(_at, _shape, _size, fill, _edge) => { return fill.color() }
            Primitive.Points(_points, _shape, _size, fill) => { return fill.color() }
            Primitive.Poses(_points, _angles, _size, fill) => { return fill.color() }
            Primitive.Segments(_points, stroke) => { return stroke.color }
            _ => { }
        }
    }
    throw ValueError("expected a filled primitive among the primitives, found none")
}

#: The marker outline of the first `Marker` in `primitives`, if it has one.
fn first_marker_edge(primitives: list[Primitive]) -> Stroke? {
    for primitive in primitives {
        match primitive {
            Primitive.Marker(_at, _shape, _size, _fill, edge) => { return edge }
            _ => { }
        }
    }
    throw ValueError("expected a marker among the primitives, found none")
}

#: Whether `packed` holds ink from the series rather than paper or furniture.
#:
#: Red minus green, and not "is it dark", because an anti-aliased spine is a
#: mid grey and a mid grey passes any brightness test you write. Pure red laid
#: at coverage `c` over white paper comes out at a separation of 255c, so this
#: counts a pixel from a quarter covered upwards.
fn is_ink(packed: int) -> bool {
    return red_of(packed) - green_of(packed) > 60 and red_of(packed) - blue_of(packed) > 60
}

#: How many separate groups of ink lie along `row`.
#:
#: One for a solid line, one per dash for a dashed one.
fn ink_runs(image: Image, row: int) -> int {
    let pixels = image.pixels()
    var runs = 0
    var inside = false
    for column in 0..image.width() {
        let inked = is_ink(pixels[row * image.width() + column])
        if inked and not inside { runs += 1 }
        inside = inked
    }
    return runs
}

#: The row of `image` holding the most ink.
fn reddest_row(image: Image) -> int {
    let pixels = image.pixels()
    var best = 0
    var most = 0
    for row in 0..image.height() {
        var count = 0
        for column in 0..image.width() {
            if is_ink(pixels[row * image.width() + column]) { count += 1 }
        }
        if count > most {
            most = count
            best = row
        }
    }
    return best
}

#: A figure holding one horizontal line across the middle, in `INK`.
fn ruled_figure(width: float, dash: list[float]) -> Figure {
    let figure = Figure(width: 320, height: 120)
    let axes = figure.axes()
    let _limits = axes.set_ylim(-1.0, 1.0)
    let _series = axes.line(
        [0.0, 1.0],
        [0.0, 0.0],
        color: Color.from_hex(INK),
        width: width,
        dash: dash
    )
    return figure
}

pub fn test_a_solid_line_keeps_its_round_cap() -> void {
    let line = Line([0.0, 1.0], [0.0, 1.0])
    let stroke = first_stroke(line.draw(unit_frame()))
    assert_true(stroke.cap == LineCap.Round)
    assert_true(stroke.join == LineJoin.Round)
    assert_eq(stroke.dash.len(), 0)
}

pub fn test_a_dashed_line_butts_its_dashes() -> void {
    # A round cap adds half the line width at each end of every dash, which at
    # the default 1.75 width turns [1, 2] into 2.75 on and 0.25 off.
    let line = Line([0.0, 1.0], [0.0, 1.0], dash: [1.0, 2.0])
    let stroke = first_stroke(line.draw(unit_frame()))
    assert_true(stroke.cap == LineCap.Butt)
    assert_true(stroke.join == LineJoin.Round)
    assert_eq(stroke.dash.len(), 2)
    assert_near(stroke.dash[0], 1.0)
    assert_near(stroke.dash[1], 2.0)
}

pub fn test_a_chosen_cap_beats_the_default_and_survives_a_later_dash() -> void {
    let square = Line([0.0, 1.0], [0.0, 1.0], dash: [4.0, 2.0], cap: LineCap.Square)
    assert_true(first_stroke(square.draw(unit_frame())).cap == LineCap.Square)

    let pinned = Line([0.0, 1.0], [0.0, 1.0]).cap(LineCap.Square)
    let _dashed = pinned.dash([4.0, 2.0])
    assert_true(first_stroke(pinned.draw(unit_frame())).cap == LineCap.Square)

    let mitred = Line([0.0, 1.0], [0.0, 1.0]).join(LineJoin.Miter)
    assert_true(first_stroke(mitred.draw(unit_frame())).join == LineJoin.Miter)
}

pub fn test_dashing_after_construction_butts_the_cap_too() -> void {
    let line = Line([0.0, 1.0], [0.0, 1.0])
    assert_true(first_stroke(line.draw(unit_frame())).cap == LineCap.Round)
    let _dashed = line.dash([1.0, 2.0])
    assert_true(first_stroke(line.draw(unit_frame())).cap == LineCap.Butt)

    let _solid = line.dash([])
    assert_true(first_stroke(line.draw(unit_frame())).cap == LineCap.Round)
}

pub fn test_a_bad_dash_pattern_never_reaches_the_document() -> void {
    let line = Line([0.0, 1.0], [0.0, 1.0])
    expect_error(ValueError, || line.dash([5.0, -3.0]))
    expect_error(ValueError, || line.dash([0.0, 0.0]))
    expect_error(ValueError, || Line([0.0, 1.0], [0.0, 1.0], dash: [5.0, -3.0]))

    # The rejected pattern must not have been half-applied on its way out.
    assert_eq(first_stroke(line.draw(unit_frame())).dash.len(), 0)

    let figure = ruled_figure(2.0, [5.0, 3.0])
    let document = to_svg(figure.render(), 320, 120)
    assert_true(document.contains("stroke-dasharray=\"5,3\""))
    assert_false(document.contains("stroke-dasharray=\"5,-3\""))
}

pub fn test_a_dotted_line_draws_as_dots_and_not_as_a_rule() -> void {
    # The whole point of the butt cap: at 1.75 pixels wide, [1, 2] used to come
    # out as one unbroken run, and jaiplot could not draw a dotted line.
    let dotted = render_image(ruled_figure(1.75, [1.0, 2.0]).render(), 320, 120)
    let dotted_row = reddest_row(dotted)
    assert_true(ink_runs(dotted, dotted_row) > 20)

    let solid = render_image(ruled_figure(1.75, []).render(), 320, 120)
    assert_eq(ink_runs(solid, reddest_row(solid)), 1)
}

pub fn test_a_wide_dash_keeps_its_gaps() -> void {
    # At width 6 the round cap added 3 pixels to each end of every dash, which
    # is more than the 3-pixel gap: [6, 3] rendered solid.
    let image = render_image(ruled_figure(6.0, [6.0, 3.0]).render(), 320, 120)
    let row = reddest_row(image)
    assert_true(ink_runs(image, row) > 8)
}

pub fn test_a_marker_edge_reaches_the_marker() -> void {
    let plain = Line([0.0, 1.0], [0.0, 1.0], marker: MarkerShape.Circle)
    assert_true(first_marker_edge(plain.draw(unit_frame())) == null)

    let white = Color.from_hex("#ffffff")
    let outlined = Line(
        [0.0, 1.0],
        [0.0, 1.0],
        marker: MarkerShape.Circle,
        marker_edge_color: white,
        marker_edge_width: 2.0
    )
    let edge = first_marker_edge(outlined.draw(unit_frame())) ?? Stroke(Color.from_rgb(0, 0, 0))
    assert_eq(edge.color.to_hex(), "#ffffff")
    assert_near(edge.width, 2.0)

    # A colour on its own still draws, at the default edge width.
    let thin = Line([0.0, 1.0], [0.0, 1.0], marker: MarkerShape.Circle, marker_edge_color: white)
    let default_edge = first_marker_edge(thin.draw(unit_frame())) ?? Stroke(Color.from_rgb(0, 0, 0))
    assert_near(default_edge.width, 1.0)
}

pub fn test_an_edge_left_to_default_follows_the_cycle_colour() -> void {
    # Freezing the edge stroke at construction would outline every cycled
    # series in the black the constructor started from.
    let cycled = ColorCycle().at(0)

    let line = Line([0.0, 1.0], [0.0, 1.0], marker: MarkerShape.Circle, marker_edge_width: 1.5)
    line.adopt(cycled)
    let edge = first_marker_edge(line.draw(unit_frame())) ?? Stroke(Color.from_rgb(0, 0, 0))
    assert_eq(edge.color.to_hex(), cycled.to_hex())
}

pub fn test_a_line_swatch_carries_its_marker() -> void {
    let box = Rect(0.0, 0.0, 22.0, 7.0)

    let plain = Line([0.0, 1.0], [0.0, 1.0]).swatch(box)
    assert_eq(plain.len(), 1)

    let marked = Line([0.0, 1.0], [0.0, 1.0], marker: MarkerShape.Square).swatch(box)
    assert_eq(marked.len(), 2)
    var markers = 0
    for primitive in marked {
        match primitive {
            Primitive.Marker(at, shape, size, _fill, _edge) => {
                markers += 1
                assert_near(at.x, 11.0)
                assert_near(at.y, 3.5)
                assert_true(shape == MarkerShape.Square)
                assert_true(size <= box.height())
            }
            _ => { }
        }
    }
    assert_eq(markers, 1)

    # A series with no ink in its stroke is only its markers, and the swatch
    # has to show them or the entry is blank.
    let invisible = Line([0.0, 1.0], [0.0, 1.0], width: 0.0, marker: MarkerShape.Circle)
    assert_eq(invisible.swatch(box).len(), 1)
}

pub fn test_a_translucent_colour_survives_every_series() -> void {
    # 0.25 * 255 rounds to 64. Each of these forced its own figure over it.
    let quarter = Color.from_hex(INK).with_alpha(0.25)
    let frame = unit_frame()

    let points = Scatter([0.0, 1.0], [0.0, 1.0], color: quarter)
    assert_eq(first_fill(points.draw(frame)).alpha(), 64)

    let band = FillBetween([0.0, 1.0], [0.0, 0.0], [1.0, 1.0], color: quarter)
    assert_eq(first_fill(band.draw(frame)).alpha(), 64)

    let scan = Rays([0.0, 0.0], [0.0, 0.0], [1.0, 0.5], [1.0, 1.0], color: quarter)
    assert_eq(first_fill(scan.draw(frame)).alpha(), 64)

    let cloud = PoseCloud([0.0, 1.0], [0.0, 1.0], color: quarter)
    assert_eq(first_fill(cloud.draw(frame)).alpha(), 64)
}

pub fn test_an_explicit_opacity_beats_the_colour_it_was_given() -> void {
    let quarter = Color.from_hex(INK).with_alpha(0.25)
    let frame = unit_frame()

    let opaque = Scatter([0.0, 1.0], [0.0, 1.0], color: quarter, opacity: 1.0)
    assert_eq(first_fill(opaque.draw(frame)).alpha(), 255)

    let band = FillBetween([0.0, 1.0], [0.0, 0.0], [1.0, 1.0], color: quarter, opacity: 0.5)
    assert_eq(first_fill(band.draw(frame)).alpha(), 128)
}

pub fn test_an_opaque_colour_still_takes_the_series_default() -> void {
    # The bands and the clouds are translucent by default for a reason, and a
    # colour that says nothing about alpha must not turn that off.
    let frame = unit_frame()
    let red = Color.from_hex(INK)

    let band = FillBetween([0.0, 1.0], [0.0, 0.0], [1.0, 1.0], color: red)
    assert_eq(first_fill(band.draw(frame)).alpha(), 71)

    let cycled = FillBetween([0.0, 1.0], [0.0, 0.0], [1.0, 1.0])
    cycled.adopt(ColorCycle().at(0))
    assert_eq(first_fill(cycled.draw(frame)).alpha(), 71)

    let scan = Rays([0.0, 0.0], [0.0, 0.0], [1.0, 0.5], [1.0, 1.0], color: red)
    assert_eq(first_fill(scan.draw(frame)).alpha(), 89)

    let points = Scatter([0.0, 1.0], [0.0, 1.0], color: red)
    assert_eq(first_fill(points.draw(frame)).alpha(), 255)
}

pub fn test_restyling_a_series_does_not_flatten_its_alpha() -> void {
    let frame = unit_frame()
    let quarter = Color.from_hex(INK).with_alpha(0.25)

    let points = Scatter([0.0, 1.0], [0.0, 1.0]).color(quarter)
    assert_eq(first_fill(points.draw(frame)).alpha(), 64)

    let cloud = PoseCloud([0.0, 1.0], [0.0, 1.0]).color(quarter)
    assert_eq(first_fill(cloud.draw(frame)).alpha(), 64)

    # An opaque colour keeps whatever opacity the series is already drawing at.
    let faded = Scatter([0.0, 1.0], [0.0, 1.0], opacity: 0.5).color(Color.from_hex(INK))
    assert_eq(first_fill(faded.draw(frame)).alpha(), 128)
}

pub fn test_a_translucent_scatter_reaches_the_document() -> void {
    let figure = Figure(width: 200, height: 150)
    let axes = figure.axes()
    let _points = axes.scatter([0.0, 1.0], [0.0, 1.0]).color(Color.from_hex(INK).with_alpha(0.25))
    let document = to_svg(figure.render(), 200, 150)
    assert_true(document.contains("fill=\"#ff0000\" fill-opacity=\"0.25\""))
}
