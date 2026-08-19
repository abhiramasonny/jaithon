#: Vector text, drawn with the Hershey fonts.
#:
#: The glyph data lives in `hershey.jai`, generated from the public-domain
#: Hershey distribution; the acknowledgements its terms require are at the top
#: of that file.
#:
#: **This does not reproduce OpenCV's glyphs.** OpenCV renders from its own
#: re-derived tables, not the public-domain ones: measured against every font
#: in the distribution, no face matches on glyph widths — the closest fit is
#: out by about one and a half units a glyph, with no exact matches at all. So
#: `put_text` here draws the authentic Hershey letterforms at the same visual
#: size OpenCV draws its own, which is what layout code ported from OpenCV
#: actually depends on. Cap height at `font_scale` 1.0 matches OpenCV's to the
#: pixel for every face; the individual letters do not.
import std.math

from ..core.mat import Mat
from ..core.types import LINE_8, Point, Scalar, Size
from .drawing import line, polylines
from .hershey import FIRST_CHAR, GLYPH_COUNT, face_glyphs

pub let FONT_HERSHEY_SIMPLEX = 0
pub let FONT_HERSHEY_PLAIN = 1
pub let FONT_HERSHEY_DUPLEX = 2
pub let FONT_HERSHEY_COMPLEX = 3
pub let FONT_HERSHEY_TRIPLEX = 4
pub let FONT_HERSHEY_COMPLEX_SMALL = 5
pub let FONT_HERSHEY_SCRIPT_SIMPLEX = 6
pub let FONT_HERSHEY_SCRIPT_COMPLEX = 7

#: Or-ed into a face to slant it, exactly as OpenCV spells it.
pub let FONT_ITALIC = 16

#: Hershey puts the baseline nine units below the glyph origin and the cap
#: line twelve above it, so a capital stands twenty-one units tall in every
#: font of the distribution.
let BASELINE_UNIT = 9
let CAP_UNITS = 21.0

#: Cap height in pixels at `font_scale` 1.0, one a face, measured from what
#: OpenCV actually draws. Dividing by `CAP_UNITS` gives the factor a glyph
#: coordinate is multiplied by.
let FACE_CAP_PIXELS = [20.0, 11.0, 20.0, 20.0, 20.0, 16.0, 18.0, 18.0]

#: How far a slanted face leans, as a fraction of the distance above the
#: baseline. A quarter is the usual italic slope and is what OpenCV uses.
let ITALIC_SLOPE = 0.25

#: Marks the end of a stroke in a decoded glyph, where a real coordinate
#: cannot reach: the encoding gives every coordinate as one printable
#: character less `R`, so nothing is below -82.
let PEN_UP = -1000

#: A face decoded once and kept. Every glyph of a face lives in one flat
#: coordinate list so that drawing a character is a walk rather than a walk
#: plus an allocation a stroke.
class Font {
    #: `GLYPH_COUNT + 1` offsets into `coords`, in x-y pairs.
    pub var starts: list[int]
    pub var left: list[int]
    pub var right: list[int]
    pub var coords: list[int]

    pub fn init(self, starts: list[int], left: list[int], right: list[int], coords: list[int]) {
        self.starts = starts
        self.left = left
        self.right = right
        self.coords = coords
    }
}

var DECODED: dict[int, Font] = {}

fn decode_face(face: int) -> Font {
    let glyphs = face_glyphs(face)
    var starts: list[int] = []
    var left: list[int] = []
    var right: list[int] = []
    var coords: list[int] = []
    for index in 0..GLYPH_COUNT {
        let raw = glyphs[index].bytes()
        starts.push(coords.len() // 2)
        if raw.len() < 2 {
            left.push(0)
            right.push(0)
            continue
        }
        left.push(raw[0] - 82)
        right.push(raw[1] - 82)
        var at = 2
        while at + 1 < raw.len() {
            let x = raw[at]
            let y = raw[at + 1]
            #: A space followed by `R` lifts the pen rather than naming a
            #: point, which is how the format starts a new stroke.
            if x == 32 and y == 82 {
                coords.push(PEN_UP)
                coords.push(PEN_UP)
            } else {
                coords.push(x - 82)
                coords.push(y - 82)
            }
            at += 2
        }
    }
    starts.push(coords.len() // 2)
    return Font(starts, left, right, coords)
}

fn font_for(face: int) -> Font {
    let key = base_face(face)
    let cached = DECODED.get(key, null)
    if cached is not null { return cached }
    let built = decode_face(key)
    DECODED[key] = built
    return built
}

fn base_face(face: int) -> int {
    let plain = face & ~FONT_ITALIC
    if plain < 0 or plain >= FACE_CAP_PIXELS.len() { return FONT_HERSHEY_SIMPLEX }
    return plain
}

fn unit_scale(face: int, font_scale: float) -> float {
    return FACE_CAP_PIXELS[base_face(face)] / CAP_UNITS * font_scale
}

fn glyph_index(code: int) -> int {
    let index = code - FIRST_CHAR
    if index < 0 or index >= GLYPH_COUNT { return 0 }
    return index
}

#: The width one string occupies, in glyph units.
fn advance_units(font: Font, text: str) -> int {
    var total = 0
    for code in text.bytes() {
        let index = glyph_index(code)
        total += font.right[index] - font.left[index]
    }
    return total
}

#: How large `text` would be, and how far its lowest descender falls below the
#: baseline. The pair matches OpenCV's `getTextSize`: the size is the drawn
#: extent above the baseline plus that descender, and the second value is the
#: descender alone, so `org.y + baseline` is the bottom of the drawn text.
pub fn get_text_size(
    text: str,
    font_face: int,
    font_scale: float,
    thickness: int = 1
) -> tuple[Size, int] {
    if thickness < 0 { throw ValueError("text needs a thickness of at least zero") }
    let font = font_for(font_face)
    let unit = unit_scale(font_face, font_scale)
    var width = float(advance_units(font, text)) * unit
    if font_face & FONT_ITALIC != 0 {
        #: A slanted face leans the top of a capital to the right, which is
        #: wider than the upright advance accounts for.
        width += float(BASELINE_UNIT + 12) * ITALIC_SLOPE * unit
    }
    #: Cap line to baseline, plus however far this particular string descends.
    #: No allowance for `thickness`: a thicker stroke straddles the path
    #: rather than growing the glyph, and OpenCV reports the same size for
    #: every thickness too.
    let cap = float(BASELINE_UNIT + 12) * unit
    let descent = float(descender_units(font, text)) * unit
    let height = int(math.floor(cap + descent + 0.5))
    let baseline = int(math.floor(descent + 0.5))
    return (Size(int(math.floor(width + 0.5)), height), baseline)
}

#: How far the deepest glyph of `text` reaches below the baseline, in units.
fn descender_units(font: Font, text: str) -> int {
    var deepest = 0
    for code in text.bytes() {
        let index = glyph_index(code)
        var at = font.starts[index]
        let stop = font.starts[index + 1]
        while at < stop {
            let y = font.coords[at * 2 + 1]
            if y != PEN_UP and y - BASELINE_UNIT > deepest { deepest = y - BASELINE_UNIT }
            at += 1
        }
    }
    return deepest
}

#: Draw `text` with its baseline starting at `org`.
#:
#: `bottom_left_origin` flips the vertical axis, so the text reads upward from
#: `org` instead of downward -- the same switch OpenCV offers for images whose
#: origin is at the bottom left.
pub fn put_text(
    image: Mat,
    text: str,
    org: Point,
    font_face: int,
    font_scale: float,
    colour: Scalar,
    thickness: int = 1,
    line_type: int = LINE_8,
    bottom_left_origin: bool = false
) -> Mat {
    if thickness <= 0 { throw ValueError("text needs a thickness of at least one") }
    if font_scale <= 0.0 { throw ValueError("text needs a scale above zero") }
    let font = font_for(font_face)
    let unit = unit_scale(font_face, font_scale)
    let slope = if font_face & FONT_ITALIC != 0 { ITALIC_SLOPE } else { 0.0 }
    let flip = if bottom_left_origin { -1.0 } else { 1.0 }

    var pen = float(org.x)
    var strokes: list[list[Point]] = []
    for code in text.bytes() {
        let index = glyph_index(code)
        let bearing = float(font.left[index])
        var stroke: list[Point] = []
        var at = font.starts[index]
        let stop = font.starts[index + 1]
        while at < stop {
            let gx = font.coords[at * 2]
            let gy = font.coords[at * 2 + 1]
            if gx == PEN_UP {
                if stroke.len() > 0 { strokes.push(stroke) }
                stroke = []
            } else {
                let above = float(BASELINE_UNIT - gy)
                let x = pen + (float(gx) - bearing) * unit + above * slope * unit
                let y = float(org.y) - above * unit * flip
                stroke.push(Point(int(math.floor(x + 0.5)), int(math.floor(y + 0.5))))
            }
            at += 1
        }
        if stroke.len() > 0 { strokes.push(stroke) }
        pen += float(font.right[index] - font.left[index]) * unit
    }

    #: A one-point stroke is a dot the polyline walker would draw nothing for,
    #: so it goes out as a degenerate line instead.
    var runs: list[list[Point]] = []
    for stroke in strokes {
        if stroke.len() == 1 {
            line(image, stroke[0], stroke[0], colour, thickness, line_type)
        } else {
            runs.push(stroke)
        }
    }
    if runs.len() > 0 {
        polylines(image, runs, false, colour, thickness, line_type)
    }
    return image
}
