#: How much room a run of text takes, and where its anchor sits.
#:
#: Layout has to reserve space for a label before either backend has drawn
#: anything, and both backends must then agree with the reservation. So there is
#: one metric here and both use it: the canvas 5x7 bitmap font, six pixels of
#: advance per character, scaled by whole numbers.
#:
#: The SVG backend asks for a monospace family at a matching size. A real
#: monospace glyph is usually a little narrower than six pixels at the same
#: nominal size, so SVG text tends to sit slightly inside the box reserved for
#: it. That direction is safe — it shows as marginally generous padding, never
#: as a label overrunning its neighbour.
#:
#: ```
#: text_width("abc", 1)         # 17.0
#: text_width("abc", 2)         # 34.0
#: text_height("abc", 1)        # 7.0
#: ```
from .geom import Point, Rect
from .style import HorizontalAnchor, TextStyle, VerticalAnchor

#: Width of one glyph cell, matching `std.gui.font.GLYPH_WIDTH`.
pub const GLYPH_WIDTH = 5

#: Height of one glyph cell, matching `std.gui.font.GLYPH_HEIGHT`.
pub const GLYPH_HEIGHT = 7

#: Blank columns between two glyphs, so one character advances six pixels.
pub const GLYPH_SPACING = 1

#: Baseline-to-baseline distance for text with a newline in it.
pub const LINE_HEIGHT = 9

#: The width of `content` at `scale`, in device pixels.
#:
#: A trailing character carries no spacing, which is why this is one less than
#: six times the count. Newlines split the run and the widest line wins.
pub fn text_width(content: str, scale: int) -> float {
    var widest = 0
    for line in content.split("\n") {
        let count = line.len()
        let width = if count == 0 {
            0
        } else {
            count * (GLYPH_WIDTH + GLYPH_SPACING) - GLYPH_SPACING
        }
        widest = max(widest, width)
    }
    return (widest * scale).to_float()
}

#: The height of `content` at `scale`, in device pixels.
pub fn text_height(content: str, scale: int) -> float {
    let lines = content.split("\n").len()
    if lines <= 1 { return (GLYPH_HEIGHT * scale).to_float() }
    return ((lines - 1) * LINE_HEIGHT * scale + GLYPH_HEIGHT * scale).to_float()
}

#: The box `content` occupies when anchored at `at` in `style`.
#:
#: The canvas draws from a run's top-left corner, so every other anchor is this
#: box shifted. Backends ask for the box rather than recomputing the shift, so
#: a change of anchoring rule lands in one place.
#:
#: A quarter-turned run swaps its width and height, since the glyphs are
#: transposed rather than rotated.
pub fn text_box(at: Point, content: str, style: TextStyle) -> Rect {
    var width = text_width(content, style.scale)
    var height = text_height(content, style.scale)
    if style.quarter_turns % 2 == 1 {
        let swap = width
        width = height
        height = swap
    }

    let left = match style.horizontal {
        HorizontalAnchor.Start => at.x,
        HorizontalAnchor.Middle => at.x - width * 0.5,
        HorizontalAnchor.End => at.x - width,
    }
    # `Baseline` sits on the bottom of the glyph cell rather than below it,
    # because the bitmap font has no descenders to leave room for.
    let top = match style.vertical {
        VerticalAnchor.Top => at.y,
        VerticalAnchor.Middle => at.y - height * 0.5,
        VerticalAnchor.Baseline => at.y - height,
        VerticalAnchor.Bottom => at.y - height,
    }
    return Rect(left, top, left + width, top + height)
}
