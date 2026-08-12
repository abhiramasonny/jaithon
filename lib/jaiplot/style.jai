#: Colour and styling for jaiplot: `Color`, `ColorCycle`, `Paint`, `Stroke`,
#: `TextStyle`, `MarkerShape` and `Theme`.
#:
#: `std` packs three bytes into an int and stops there: no hex parsing, no HSL,
#: no luminance, no contrast, no palette. Worse, `std.gui.color.mix`
#: interpolates the sRGB bytes directly, and sRGB bytes are not light — mixing
#: black and white that way gives 128, a colour that carries a fifth of white's
#: light rather than half of it, so every ramp built on it sags and bands
#: through the midtones. `Color.lerp` decodes to linear light, interpolates
#: there, and re-encodes; the same midpoint comes out at 188.
#:
#: ```
#: let blue = Color.from_hex("#1f77b4")
#: blue.to_hex()                             # "#1f77b4"
#: blue.with_alpha(0.25).alpha()             # 64
#: Theme.dark().background.to_hex()          # "#14161a"
#: ```
import std.math
from std.gui.color import rgba, red_of, green_of, blue_of, alpha_of

# The sRGB transfer function, IEC 61966-2-1: a short linear toe below the
# cutoff, a 2.4 power curve above it.
const SRGB_LINEAR_CUTOFF = 0.04045
const SRGB_ENCODED_CUTOFF = 0.0031308
const SRGB_SLOPE = 12.92
const SRGB_OFFSET = 0.055
const SRGB_GAMMA = 2.4

const HEX_DIGITS = "0123456789abcdef"

#: How a stroke ends: flat at the endpoint, or overshooting it by half a width.
#:
#: Named for SVG's `stroke-linecap`, which both backends have to speak.
pub enum LineCap { Butt, Round, Square }

#: How two stroke segments meet at a corner.
#:
#: Named for SVG's `stroke-linejoin`, which both backends have to speak.
pub enum LineJoin { Miter, Round, Bevel }

#: Where a text anchor sits along the horizontal run of the text.
pub enum HorizontalAnchor { Start, Middle, End }

#: Where a text anchor sits vertically: `Baseline` is the bottom of an
#: unaccented capital, `Bottom` the bottom of the whole line box.
pub enum VerticalAnchor { Top, Middle, Baseline, Bottom }

#: The glyph a scatter point is drawn with.
#:
#: `Plus` and `Cross` are strokes rather than areas, so they stay legible at two
#: or three pixels where a filled `Circle` turns into a smudge.
pub enum MarkerShape { Circle, Square, Diamond, Triangle, Plus, Cross, Star }

#: A colour, held as a packed `0xAARRGGBB` integer.
#:
#: Immutable: every operation returns a new `Color`. The packed form is what
#: `std.gui.canvas` takes, so the raster backend hands `packed()` straight to a
#: canvas primitive, while the SVG backend splits it into `to_hex()` and
#: `opacity()`.
#:
#: ```
#: Color.from_hex("#f00").packed()    # 4294901760
#: Color.from_hsl(0.0, 1.0, 0.5).to_hex()    # "#ff0000"
#: ```
pub class Color {
    let value: int

    fn init(self, value: int) {
        self.value = value
    }

    #: Build an opaque colour from three 0-255 channels.
    #:
    #: Raises `ValueError` on a channel outside 0-255, which is almost always a
    #: 0-1 float that should have been scaled by 255 first.
    #:
    #: ```
    #: Color.from_rgb(31, 119, 180).to_hex()    # "#1f77b4"
    #: ```
    pub static fn from_rgb(red: int, green: int, blue: int) -> Color {
        return Color(rgba(red, green, blue, 255))
    }

    #: Build a colour from three 0-255 channels and a 0-255 alpha.
    #:
    #: Raises `ValueError` on any channel outside 0-255.
    #:
    #: ```
    #: Color.from_rgba(0, 0, 0, 128).opacity()    # 0.5019607843137255
    #: ```
    pub static fn from_rgba(red: int, green: int, blue: int, alpha: int) -> Color {
        return Color(rgba(red, green, blue, alpha))
    }

    #: Parse a CSS hex colour: `#rgb`, `#rrggbb` or `#rrggbbaa`.
    #:
    #: Case does not matter and the `#` is required. `#rgb` expands each digit
    #: by repeating it, so `#f80` is `#ff8800`. Anything else raises
    #: `ValueError` quoting the whole offending string.
    #:
    #: ```
    #: Color.from_hex("#f80").to_hex()         # "#ff8800"
    #: Color.from_hex("#1F77B4").to_hex()      # "#1f77b4"
    #: Color.from_hex("#1f77b480").alpha()     # 128
    #: ```
    pub static fn from_hex(text: str) -> Color {
        let digits = _hex_digits(text)
        if digits.len() == 3 {
            return Color.from_rgb(digits[0] * 17, digits[1] * 17, digits[2] * 17)
        }
        if digits.len() == 6 {
            return Color.from_rgb(
                digits[0] * 16 + digits[1],
                digits[2] * 16 + digits[3],
                digits[4] * 16 + digits[5]
            )
        }
        if digits.len() == 8 {
            return Color.from_rgba(
                digits[0] * 16 + digits[1],
                digits[2] * 16 + digits[3],
                digits[4] * 16 + digits[5],
                digits[6] * 16 + digits[7]
            )
        }
        throw ValueError(f"hex colour must be #rgb, #rrggbb or #rrggbbaa, got {text}")
    }

    #: Build an opaque colour from hue in degrees, saturation and lightness.
    #:
    #: The hue wraps, so -30 and 330 are the same red-magenta; saturation and
    #: lightness are clamped to 0-1. This is CSS `hsl()`, which is defined on
    #: the gamma-encoded sRGB channels, so the result is what a browser would
    #: show for the same three numbers — it is not a perceptual space, and
    #: sweeping the hue at fixed lightness still passes through a bright yellow
    #: and a dark blue.
    #:
    #: ```
    #: Color.from_hsl(120.0, 1.0, 0.5).to_hex()     # "#00ff00"
    #: Color.from_hsl(-30.0, 1.0, 0.5).to_hex()     # "#ff0080"
    #: Color.from_hsl(210.0, 0.0, 0.5).to_hex()     # "#808080"
    #: ```
    pub static fn from_hsl(hue: float, saturation: float, lightness: float) -> Color {
        let turned = math.fmod(hue, 360.0)
        let wrapped = if turned < 0.0 { turned + 360.0 } else { turned }
        let amount = math.clamp(saturation, 0.0, 1.0)
        let level = math.clamp(lightness, 0.0, 1.0)
        let chroma = (1.0 - math.fabs(2.0 * level - 1.0)) * amount
        let sector = wrapped / 60.0
        let second = chroma * (1.0 - math.fabs(math.fmod(sector, 2.0) - 1.0))
        let (red, green, blue) = match int(sector) {
            0 => (chroma, second, 0.0),
            1 => (second, chroma, 0.0),
            2 => (0.0, chroma, second),
            3 => (0.0, second, chroma),
            4 => (second, 0.0, chroma),
            _ => (chroma, 0.0, second),
        }
        let base = level - chroma / 2.0
        return Color.from_rgb(
            _quantize(red + base),
            _quantize(green + base),
            _quantize(blue + base)
        )
    }

    #: Wrap an already packed `0xAARRGGBB` value.
    #:
    #: Raises `ValueError` on anything outside 0-0xFFFFFFFF, because a negative
    #: or oversized value would silently corrupt a neighbouring channel.
    #:
    #: ```
    #: Color.from_packed(0xFF1F77B4).to_hex()    # "#1f77b4"
    #: ```
    pub static fn from_packed(value: int) -> Color {
        if value < 0 or value > 0xFFFFFFFF {
            throw ValueError(f"packed colour must be 0-0xFFFFFFFF, got {value}")
        }
        return Color(value)
    }

    #: The packed `0xAARRGGBB` value, ready for `std.gui.canvas`.
    pub fn packed(self) -> int {
        return self.value
    }

    #: The red channel, 0-255.
    pub fn red(self) -> int {
        return red_of(self.value)
    }

    #: The green channel, 0-255.
    pub fn green(self) -> int {
        return green_of(self.value)
    }

    #: The blue channel, 0-255.
    pub fn blue(self) -> int {
        return blue_of(self.value)
    }

    #: The alpha channel, 0 clear to 255 solid.
    pub fn alpha(self) -> int {
        return alpha_of(self.value)
    }

    #: The alpha channel as a 0-1 fraction, for SVG's `fill-opacity`.
    pub fn opacity(self) -> float {
        return float(self.alpha()) / 255.0
    }

    #: The `#rrggbb` form SVG's `fill` and `stroke` attributes take.
    #:
    #: Alpha is deliberately dropped: SVG 1.1 has no eight-digit hex, and a
    #: backend carries transparency in `fill-opacity` or `stroke-opacity`
    #: instead. `opacity()` is the other half of the pair.
    #:
    #: ```
    #: Color.from_rgba(31, 119, 180, 0).to_hex()    # "#1f77b4"
    #: ```
    pub fn to_hex(self) -> str {
        return f"#{self.red():02x}{self.green():02x}{self.blue():02x}"
    }

    #: This colour at a new opacity, `amount` 0.0 clear to 1.0 solid.
    #:
    #: The fraction rather than a 0-255 byte, because callers write it as a
    #: literal: a confidence band is `color.with_alpha(0.2)`.
    #:
    #: ```
    #: Color.from_hex("#1f77b4").with_alpha(0.2).alpha()    # 51
    #: ```
    pub fn with_alpha(self, amount: float) -> Color {
        let alpha = int(math.round(math.clamp(amount, 0.0, 1.0) * 255.0))
        return Color(self.value & 0x00FFFFFF | alpha << 24)
    }

    #: WCAG relative luminance, 0.0 for black and 1.0 for white.
    #:
    #: Gamma-correct: each channel is decoded to linear light before the
    #: Rec. 709 weights are applied, which is why mid grey lands near 0.21 and
    #: not at 0.5. Alpha is ignored — luminance is only defined for what
    #: actually reaches the eye, so composite first if the colour is
    #: translucent.
    #:
    #: ```
    #: Color.from_hex("#000000").luminance()    # 0.0
    #: Color.from_hex("#ffffff").luminance() > 0.999    # true
    #: ```
    pub fn luminance(self) -> float {
        let red = _linear_channel(self.red())
        let green = _linear_channel(self.green())
        let blue = _linear_channel(self.blue())
        return 0.2126 * red + 0.7152 * green + 0.0722 * blue
    }

    #: The WCAG contrast ratio between this colour and `other`, 1.0 to 21.0.
    #:
    #: Symmetric — the lighter of the two always goes on top. WCAG 2.1 asks for
    #: 4.5 for body text and 3.0 for large text, so a legend picks its label
    #: colour by taking whichever candidate scores higher against the swatch it
    #: sits on.
    #:
    #: ```
    #: Color.from_hex("#000000").contrast_ratio(Color.from_hex("#ffffff")) > 20.9    # true
    #: Color.from_hex("#777777").contrast_ratio(Color.from_hex("#777777"))           # 1.0
    #: ```
    pub fn contrast_ratio(self, other: Color) -> float {
        let mine = self.luminance()
        let theirs = other.luminance()
        return (max(mine, theirs) + 0.05) / (min(mine, theirs) + 0.05)
    }

    #: Interpolate towards `other` in linear light, `t` 0.0 here and 1.0 there.
    #:
    #: This is the gamma-correct mix `std.gui.color.mix` does not do. That one
    #: averages the sRGB bytes, so black-to-white at the midpoint gives 128 — a
    #: colour carrying 21% of white's light, visibly too dark, and the same
    #: error bends every ramp towards its darker end. Here both colours are
    #: decoded to linear light, mixed there, and re-encoded, which puts the same
    #: midpoint at 188.
    #:
    #: Alpha is interpolated directly, since it is already linear. `t` is
    #: clamped, so this is safe to drive from an animation clock that
    #: overshoots.
    #:
    #: ```
    #: Color.from_hex("#000000").lerp(Color.from_hex("#ffffff"), 0.5).red()    # 188
    #: Color.from_hex("#1f77b4").lerp(Color.from_hex("#ffffff"), 0.0).to_hex()    # "#1f77b4"
    #: ```
    pub fn lerp(self, other: Color, t: float) -> Color {
        let amount = math.clamp(t, 0.0, 1.0)
        let red = _mix_linear(self.red(), other.red(), amount)
        let green = _mix_linear(self.green(), other.green(), amount)
        let blue = _mix_linear(self.blue(), other.blue(), amount)
        let alpha = math.lerp(float(self.alpha()), float(other.alpha()), amount)
        return Color.from_rgba(red, green, blue, int(math.round(alpha)))
    }

    #: This colour moved `amount` of the way towards white, in linear light.
    #:
    #: Alpha is preserved, so lightening a translucent fill does not make it
    #: opaque.
    #:
    #: ```
    #: Color.from_hex("#000000").lighten(1.0).to_hex()    # "#ffffff"
    #: Color.from_hex("#808080").lighten(0.0).to_hex()    # "#808080"
    #: ```
    pub fn lighten(self, amount: float) -> Color {
        return self.lerp(Color.from_rgba(255, 255, 255, self.alpha()), amount)
    }

    #: This colour moved `amount` of the way towards black, in linear light.
    #:
    #: Alpha is preserved.
    #:
    #: ```
    #: Color.from_hex("#ffffff").darken(1.0).to_hex()    # "#000000"
    #: ```
    pub fn darken(self, amount: float) -> Color {
        return self.lerp(Color.from_rgba(0, 0, 0, self.alpha()), amount)
    }

    fn __eq__(self, other: Color) -> bool {
        return self.value == other.value
    }

    fn __str__(self) -> str {
        return f"Color({self.to_hex()})"
    }
}

#: The categorical palette a chart draws each new series from, cycling on
#: demand.
#:
#: The ten defaults are Paul Tol's *muted* qualitative scheme — rose, indigo,
#: sand, green, cyan, wine, teal, olive and purple, in that order, from the SRON
#: technical note *Colour Schemes* (EPS-TN-09-002), chosen so no two collapse
#: together under deuteranopia, protanopia or tritanopia — followed by Okabe and
#: Ito's orange `#e69f00` from the Color Universal Design set, because *muted*
#: has no orange and ten is the length a plotting default wants. Tol's pale grey
#: `#dddddd` is left out on purpose: he reserves it for data that should not
#: draw the eye, which is the opposite of what a series colour is for.
#:
#: ```
#: let cycle = ColorCycle()
#: cycle.len()             # 10
#: cycle.next().to_hex()   # "#cc6677"
#: cycle.next().to_hex()   # "#332288"
#: ```
pub class ColorCycle {
    let colors: list[Color]
    var position: int

    #: Create a cycle over the default ten-colour categorical palette.
    pub fn init(self) {
        self.colors = [
            Color.from_hex("#cc6677"),
            Color.from_hex("#332288"),
            Color.from_hex("#ddcc77"),
            Color.from_hex("#117733"),
            Color.from_hex("#88ccee"),
            Color.from_hex("#882255"),
            Color.from_hex("#44aa99"),
            Color.from_hex("#999933"),
            Color.from_hex("#aa4499"),
            Color.from_hex("#e69f00"),
        ]
        self.position = 0
    }

    #: Create a cycle over a palette of your own.
    #:
    #: Raises `ValueError` on an empty list, which would leave `next` with
    #: nothing to hand out.
    #:
    #: ```
    #: ColorCycle.from_colors([Color.from_hex("#000")]).next().to_hex()    # "#000000"
    #: ```
    pub static fn from_colors(colors: list[Color]) -> ColorCycle {
        if colors.len() == 0 {
            throw ValueError("a colour cycle needs at least one colour, got an empty list")
        }
        let cycle = ColorCycle()
        cycle.colors = [color for color in colors]
        return cycle
    }

    #: Number of colours before the cycle repeats.
    pub fn len(self) -> int {
        return self.colors.len()
    }

    #: The colour at `index`, wrapping in both directions.
    #:
    #: Wrapping rather than raising, because the caller is a plot with more
    #: series than the palette has colours and that is ordinary. `at(-1)` is the
    #: last colour.
    #:
    #: ```
    #: ColorCycle().at(10).to_hex()    # "#cc6677"
    #: ColorCycle().at(-1).to_hex()    # "#e69f00"
    #: ```
    pub fn at(self, index: int) -> Color {
        return self.colors[index % self.colors.len()]
    }

    #: The next colour, advancing the cycle.
    pub fn next(self) -> Color {
        let color = self.at(self.position)
        self.position += 1
        return color
    }

    #: Rewind so the next colour handed out is the first one again.
    pub fn reset(self) -> void {
        self.position = 0
    }

    #: The palette in order, as a fresh list.
    pub fn to_list(self) -> list[Color] {
        return [color for color in self.colors]
    }

    fn __len__(self) -> int { return self.colors.len() }

    fn __str__(self) -> str {
        return f"ColorCycle({self.colors.len()} colours, at {self.position})"
    }
}

#: What fills the inside of a shape.
#:
#: One flat colour today. The name is `Paint` and not `FillColor` so that a
#: linear or radial gradient can arrive as a second kind of `Paint` without
#: renaming the field on every primitive that holds one. For the same reason
#: the colour is reached through `color()` and is not a public field: a
#: gradient will not have one, and code that reached past the accessor today
#: would break on the day it arrives.
#:
#: ```
#: Paint.solid(Color.from_hex("#1f77b4")).color().to_hex()    # "#1f77b4"
#: ```
pub class Paint {
    let fill: Color

    #: Create a flat fill of `color`.
    pub fn init(self, color: Color) {
        self.fill = color
    }

    #: Create a flat fill of `color`.
    #:
    #: The named constructor exists so that call sites read as
    #: `Paint.solid(c)`, which will still be true when `Paint.gradient(...)`
    #: joins it.
    pub static fn solid(color: Color) -> Paint {
        return Paint(color)
    }

    #: The single colour this paint lays down.
    pub fn color(self) -> Color {
        return self.fill
    }

    #: The same paint at a new opacity, `amount` 0.0 clear to 1.0 solid.
    #:
    #: ```
    #: Paint.solid(Color.from_hex("#1f77b4")).with_alpha(0.5).color().alpha()    # 128
    #: ```
    pub fn with_alpha(self, amount: float) -> Paint {
        return Paint(self.fill.with_alpha(amount))
    }

    fn __str__(self) -> str {
        return f"Paint({self.fill})"
    }
}

#: What outlines a shape: colour, width in device pixels, and a dash pattern.
#:
#: `dash` is the SVG `stroke-dasharray`: alternating on and off lengths in
#: device pixels, empty for a solid line. The list is copied on the way in, so a
#: caller reusing one pattern across several strokes cannot mutate them all at
#: once.
#:
#: ```
#: let stroke = Stroke(Color.from_hex("#1f77b4"), width: 2.0, dash: [4.0, 2.0])
#: stroke.width           # 2.0
#: stroke.dash.len()      # 2
#: stroke.cap             # LineCap.Butt
#: ```
pub class Stroke {
    #: The colour of the outline.
    pub var color: Color

    #: The outline's thickness in device pixels, centred on the path.
    pub var width: float

    #: Alternating on and off lengths in device pixels; empty means solid.
    pub var dash: list[float]

    #: How the outline ends.
    pub var cap: LineCap

    #: How two segments of the outline meet.
    pub var join: LineJoin

    #: Create a stroke, solid and one pixel wide unless told otherwise.
    #:
    #: Raises `ValueError` on a negative width, on a negative dash length, or
    #: on a dash pattern that sums to zero — the last would leave a dashing
    #: walker stepping nowhere for ever.
    pub fn init(
        self,
        color: Color,
        width: float = 1.0,
        dash: list[float] = [],
        cap: LineCap = LineCap.Butt,
        join: LineJoin = LineJoin.Miter
    ) {
        if width < 0.0 {
            throw ValueError(f"stroke width must be non-negative, got {width}")
        }
        var total = 0.0
        for length in dash {
            if length < 0.0 {
                throw ValueError(f"dash lengths must be non-negative, got {length}")
            }
            total += length
        }
        if dash.len() > 0 and total == 0.0 {
            throw ValueError(f"dash pattern must have a positive total length, got {dash}")
        }
        self.color = color
        self.width = width
        self.dash = [length for length in dash]
        self.cap = cap
        self.join = join
    }

    #: Report whether this stroke would put down any ink.
    #:
    #: A backend skips the geometry entirely when it would not, which is how
    #: `stroke: none` is expressed without a separate optional.
    pub fn is_visible(self) -> bool {
        return self.width > 0.0 and self.color.alpha() > 0
    }

    fn __str__(self) -> str {
        let shape = f"{self.cap}, {self.join}"
        return f"Stroke({self.color}, width={self.width}, dash={self.dash}, {shape})"
    }
}

#: How a run of text is coloured, sized, anchored and turned.
#:
#: `scale` is an integer because the only font is a 5x7 bitmap and the canvas
#: scales it by whole pixels: 1 gives 7px tall text, 2 gives 14px, and there is
#: nothing in between. `quarter_turns` is likewise quantised, because the
#: renderer transposes glyph bitmaps rather than rotating them; it counts
#: anticlockwise turns and wraps, so a y-axis label reads naturally as either 1
#: or -3.
#:
#: ```
#: let style = TextStyle(Color.from_hex("#1a1a1a"), scale: 2, quarter_turns: -1)
#: style.scale             # 2
#: style.quarter_turns     # 3
#: style.horizontal        # HorizontalAnchor.Start
#: ```
pub class TextStyle {
    #: The colour of the glyphs.
    pub var color: Color

    #: Whole-pixel multiplier on the 5x7 font, so 1 is 7px tall and 2 is 14px.
    pub var scale: int

    #: Where the anchor point sits along the run of the text.
    pub var horizontal: HorizontalAnchor

    #: Where the anchor point sits across the height of the text.
    pub var vertical: VerticalAnchor

    #: Anticlockwise quarter turns about the anchor point, 0 to 3.
    pub var quarter_turns: int

    #: Create a text style anchored at its top-left corner, unturned.
    #:
    #: `Top` and `Start` are the defaults because that is where the canvas puts
    #: a glyph run with no help, so the common case costs no adjustment.
    #:
    #: Raises `ValueError` on a scale below 1, since the font cannot be drawn
    #: smaller than its own bitmap.
    pub fn init(
        self,
        color: Color,
        scale: int = 1,
        horizontal: HorizontalAnchor = HorizontalAnchor.Start,
        vertical: VerticalAnchor = VerticalAnchor.Top,
        quarter_turns: int = 0
    ) {
        if scale < 1 {
            throw ValueError(f"text scale must be at least 1, got {scale}")
        }
        self.color = color
        self.scale = scale
        self.horizontal = horizontal
        self.vertical = vertical
        self.quarter_turns = quarter_turns % 4
    }

    fn __str__(self) -> str {
        let anchor = f"{self.horizontal}, {self.vertical}"
        return f"TextStyle({self.color}, scale={self.scale}, {anchor}, turns={self.quarter_turns})"
    }
}

#: The colours a figure draws itself in, plus the palette its series come from.
#:
#: `foreground` is the ink for anything that is not a series and not a word —
#: tick marks, legend frames, the default marker. `text` is separate from it so
#: a theme can quieten its labels without dulling its rules.
#:
#: ```
#: Theme.light().background.to_hex()    # "#ffffff"
#: Theme.dark().background.to_hex()     # "#14161a"
#: Theme.light().cycle.len()            # 10
#: ```
pub class Theme {
    #: The colour the figure is cleared to.
    pub var background: Color

    #: Ink for everything that is neither a series nor a word.
    pub var foreground: Color

    #: The gridlines behind the data.
    pub var grid: Color

    #: The axis lines bounding the plotting area.
    pub var spine: Color

    #: Titles, axis labels and tick labels.
    pub var text: Color

    #: The palette each new series draws its colour from.
    pub var cycle: ColorCycle

    #: Create a theme from its five colours and a series palette.
    pub fn init(
        self,
        background: Color,
        foreground: Color,
        grid: Color,
        spine: Color,
        text: Color,
        cycle: ColorCycle
    ) {
        self.background = background
        self.foreground = foreground
        self.grid = grid
        self.spine = spine
        self.text = text
        self.cycle = cycle
    }

    #: The default theme: dark ink on white paper.
    #:
    #: The grid is pale enough to sit under the data rather than compete with
    #: it, and the near-black text clears WCAG AA against the background with
    #: room to spare.
    #:
    #: ```
    #: let theme = Theme.light()
    #: theme.text.contrast_ratio(theme.background) > 4.5    # true
    #: ```
    pub static fn light() -> Theme {
        return Theme(
            Color.from_hex("#ffffff"),
            Color.from_hex("#1a1a1a"),
            Color.from_hex("#d9d9d9"),
            Color.from_hex("#333333"),
            Color.from_hex("#1a1a1a"),
            ColorCycle()
        )
    }

    #: The inverted theme: light ink on a near-black ground.
    #:
    #: The background is `#14161a` rather than pure black because a plot on
    #: true black haloes badly on an OLED panel, and the grid is lifted just
    #: far enough off it to be seen at one pixel.
    #:
    #: ```
    #: let theme = Theme.dark()
    #: theme.text.contrast_ratio(theme.background) > 4.5    # true
    #: ```
    pub static fn dark() -> Theme {
        return Theme(
            Color.from_hex("#14161a"),
            Color.from_hex("#e6e6e6"),
            Color.from_hex("#2e3238"),
            Color.from_hex("#8a9099"),
            Color.from_hex("#e6e6e6"),
            ColorCycle()
        )
    }
}

# Every nibble after the `#`, in order. The message quotes the whole string
# rather than the one bad character, because "#12g456" is far easier to spot
# than "g".
fn _hex_digits(text: str) -> list[int] {
    if not text.starts_with("#") {
        throw ValueError(f"hex colour must be #rgb, #rrggbb or #rrggbbaa, got {text}")
    }
    let digits: list[int] = []
    for character in text[1:].chars() {
        let position = HEX_DIGITS.find(character.lower())
        if position < 0 {
            throw ValueError(f"hex colour must be #rgb, #rrggbb or #rrggbbaa, got {text}")
        }
        digits.push(position)
    }
    return digits
}

# Decode a 0-255 sRGB channel to linear light.
fn _linear_channel(channel: int) -> float {
    let encoded = float(channel) / 255.0
    if encoded <= SRGB_LINEAR_CUTOFF {
        return encoded / SRGB_SLOPE
    }
    return ((encoded + SRGB_OFFSET) / (1.0 + SRGB_OFFSET)) ** SRGB_GAMMA
}

# Encode linear light back to a 0-255 sRGB channel.
fn _encode_channel(linear: float) -> int {
    let level = math.clamp(linear, 0.0, 1.0)
    if level <= SRGB_ENCODED_CUTOFF {
        return _quantize(level * SRGB_SLOPE)
    }
    return _quantize((1.0 + SRGB_OFFSET) * level ** (1.0 / SRGB_GAMMA) - SRGB_OFFSET)
}

fn _mix_linear(from_channel: int, to_channel: int, amount: float) -> int {
    let mixed = math.lerp(_linear_channel(from_channel), _linear_channel(to_channel), amount)
    return _encode_channel(mixed)
}

fn _quantize(unit: float) -> int {
    return int(math.round(math.clamp(unit, 0.0, 1.0) * 255.0))
}
