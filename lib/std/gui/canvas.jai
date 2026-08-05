#: Software rasterisation: colours, shapes, curves, text and blitting, written
#: entirely in Jaithon over a flat pixel buffer.
#:
#: A colour is one `int` packed `0xAARRGGBB`, so it costs nothing to pass and
#: can be a `const`. Build one with `rgb` or `rgba`; take it apart with
#: `red_of` and friends. Every drawing routine composites the source colour
#: over the destination when its alpha is below 255, so translucency is
#: available everywhere and costs nothing when unused.
#:
#: A `Canvas` draws into any `PixelTarget`: a `Window` for the screen, an
#: `Image` for an offscreen sprite or a test. Coordinates are integer pixels
#: with the origin at the top left and y increasing downward. Everything is
#: clipped — to the canvas, and to the current `set_clip` rectangle — so
#: drawing off the edge is ordinary and silent rather than an error.
#:
#: ```
#: let canvas = Canvas(Image(8, 8))
#: canvas.fill_rect(1, 1, 3, 3, rgb(255, 0, 0))
#: canvas.get_pixel(2, 2) == rgb(255, 0, 0)    # true
#: canvas.get_pixel(6, 6) == rgb(255, 0, 0)    # false
#: ```
#:
#: One caveat about alpha: a *filled* shape writes each pixel exactly once, but
#: an *outline* — `polygon`, `triangle`, `thick_line` — can cross itself where
#: two segments meet, and a translucent stroke shows a darker pixel at those
#: joints. Stroke with an opaque colour, or fill.
from std.gui import PixelTarget
import std.math

#: Width of one glyph cell in the built-in font, in pixels.
pub const GLYPH_WIDTH = 5

#: Height of one glyph cell in the built-in font, in pixels.
pub const GLYPH_HEIGHT = 7

#: Blank columns between two glyphs, so one character advances 6 pixels.
pub const GLYPH_SPACING = 1

#: Baseline-to-baseline distance for `text` with a newline in it.
pub const LINE_HEIGHT = 9

# The 5x7 font, five column bytes per glyph, covering ASCII 32 to 126 in order.
# Bit 0 of a column is its top pixel and bit 6 its bottom, so column 0x5F —
# every row but the sixth — is the stem and dot of an exclamation mark. This is
# the classic 5x7 cell used by LCD character generators, chosen because every
# glyph is legible at one pixel per dot with no anti-aliasing to smear it.
let _GLYPHS: list[int] = [
    0x00, 0x00, 0x00, 0x00, 0x00,  0x00, 0x00, 0x5F, 0x00, 0x00,   #   !
    0x00, 0x07, 0x00, 0x07, 0x00,  0x14, 0x7F, 0x14, 0x7F, 0x14,   # " #
    0x24, 0x2A, 0x7F, 0x2A, 0x12,  0x23, 0x13, 0x08, 0x64, 0x62,   # $ %
    0x36, 0x49, 0x55, 0x22, 0x50,  0x00, 0x05, 0x03, 0x00, 0x00,   # & '
    0x00, 0x1C, 0x22, 0x41, 0x00,  0x00, 0x41, 0x22, 0x1C, 0x00,   # ( )
    0x14, 0x08, 0x3E, 0x08, 0x14,  0x08, 0x08, 0x3E, 0x08, 0x08,   # * +
    0x00, 0x50, 0x30, 0x00, 0x00,  0x08, 0x08, 0x08, 0x08, 0x08,   # , -
    0x00, 0x60, 0x60, 0x00, 0x00,  0x20, 0x10, 0x08, 0x04, 0x02,   # . /
    0x3E, 0x51, 0x49, 0x45, 0x3E,  0x00, 0x42, 0x7F, 0x40, 0x00,   # 0 1
    0x42, 0x61, 0x51, 0x49, 0x46,  0x21, 0x41, 0x45, 0x4B, 0x31,   # 2 3
    0x18, 0x14, 0x12, 0x7F, 0x10,  0x27, 0x45, 0x45, 0x45, 0x39,   # 4 5
    0x3C, 0x4A, 0x49, 0x49, 0x30,  0x01, 0x71, 0x09, 0x05, 0x03,   # 6 7
    0x36, 0x49, 0x49, 0x49, 0x36,  0x06, 0x49, 0x49, 0x29, 0x1E,   # 8 9
    0x00, 0x36, 0x36, 0x00, 0x00,  0x00, 0x56, 0x36, 0x00, 0x00,   # : ;
    0x08, 0x14, 0x22, 0x41, 0x00,  0x14, 0x14, 0x14, 0x14, 0x14,   # < =
    0x00, 0x41, 0x22, 0x14, 0x08,  0x02, 0x01, 0x51, 0x09, 0x06,   # > ?
    0x32, 0x49, 0x79, 0x41, 0x3E,  0x7E, 0x11, 0x11, 0x11, 0x7E,   # @ A
    0x7F, 0x49, 0x49, 0x49, 0x36,  0x3E, 0x41, 0x41, 0x41, 0x22,   # B C
    0x7F, 0x41, 0x41, 0x22, 0x1C,  0x7F, 0x49, 0x49, 0x49, 0x41,   # D E
    0x7F, 0x09, 0x09, 0x09, 0x01,  0x3E, 0x41, 0x49, 0x49, 0x7A,   # F G
    0x7F, 0x08, 0x08, 0x08, 0x7F,  0x00, 0x41, 0x7F, 0x41, 0x00,   # H I
    0x20, 0x40, 0x41, 0x3F, 0x01,  0x7F, 0x08, 0x14, 0x22, 0x41,   # J K
    0x7F, 0x40, 0x40, 0x40, 0x40,  0x7F, 0x02, 0x0C, 0x02, 0x7F,   # L M
    0x7F, 0x04, 0x08, 0x10, 0x7F,  0x3E, 0x41, 0x41, 0x41, 0x3E,   # N O
    0x7F, 0x09, 0x09, 0x09, 0x06,  0x3E, 0x41, 0x51, 0x21, 0x5E,   # P Q
    0x7F, 0x09, 0x19, 0x29, 0x46,  0x46, 0x49, 0x49, 0x49, 0x31,   # R S
    0x01, 0x01, 0x7F, 0x01, 0x01,  0x3F, 0x40, 0x40, 0x40, 0x3F,   # T U
    0x1F, 0x20, 0x40, 0x20, 0x1F,  0x3F, 0x40, 0x38, 0x40, 0x3F,   # V W
    0x63, 0x14, 0x08, 0x14, 0x63,  0x07, 0x08, 0x70, 0x08, 0x07,   # X Y
    0x61, 0x51, 0x49, 0x45, 0x43,  0x00, 0x7F, 0x41, 0x41, 0x00,   # Z [
    0x02, 0x04, 0x08, 0x10, 0x20,  0x00, 0x41, 0x41, 0x7F, 0x00,   # \ ]
    0x04, 0x02, 0x01, 0x02, 0x04,  0x40, 0x40, 0x40, 0x40, 0x40,   # ^ _
    0x00, 0x01, 0x02, 0x04, 0x00,  0x20, 0x54, 0x54, 0x54, 0x78,   # ` a
    0x7F, 0x48, 0x44, 0x44, 0x38,  0x38, 0x44, 0x44, 0x44, 0x20,   # b c
    0x38, 0x44, 0x44, 0x48, 0x7F,  0x38, 0x54, 0x54, 0x54, 0x18,   # d e
    0x08, 0x7E, 0x09, 0x01, 0x02,  0x0C, 0x52, 0x52, 0x52, 0x3E,   # f g
    0x7F, 0x08, 0x04, 0x04, 0x78,  0x00, 0x44, 0x7D, 0x40, 0x00,   # h i
    0x20, 0x40, 0x44, 0x3D, 0x00,  0x7F, 0x10, 0x28, 0x44, 0x00,   # j k
    0x00, 0x41, 0x7F, 0x40, 0x00,  0x7C, 0x04, 0x18, 0x04, 0x78,   # l m
    0x7C, 0x08, 0x04, 0x04, 0x78,  0x38, 0x44, 0x44, 0x44, 0x38,   # n o
    0x7C, 0x14, 0x14, 0x14, 0x08,  0x08, 0x14, 0x14, 0x18, 0x7C,   # p q
    0x7C, 0x08, 0x04, 0x04, 0x08,  0x48, 0x54, 0x54, 0x54, 0x20,   # r s
    0x04, 0x3F, 0x44, 0x40, 0x20,  0x3C, 0x40, 0x40, 0x20, 0x7C,   # t u
    0x1C, 0x20, 0x40, 0x20, 0x1C,  0x3C, 0x40, 0x30, 0x40, 0x3C,   # v w
    0x44, 0x28, 0x10, 0x28, 0x44,  0x0C, 0x50, 0x50, 0x50, 0x3C,   # x y
    0x44, 0x64, 0x54, 0x4C, 0x44,  0x00, 0x08, 0x36, 0x41, 0x00,   # z {
    0x00, 0x00, 0x7F, 0x00, 0x00,  0x00, 0x41, 0x36, 0x08, 0x00,   # | }
    0x08, 0x04, 0x08, 0x10, 0x08,                                  # ~
]

#: An offscreen pixel buffer: a sprite, a tile sheet, or a canvas for a test.
#:
#: ```
#: let sprite = Image(4, 4, rgb(0, 0, 0))
#: sprite.width()    # 4
#: ```
pub class Image: PixelTarget {
    let data: list[int]
    let columns: int
    let rows: int

    #: Create a `width` by `height` image filled with `fill`.
    pub fn init(self, width: int, height: int, fill: int = 0xFF000000) {
        if width <= 0 or height <= 0 {
            throw ValueError(f"image size must be positive, got {width}x{height}")
        }
        self.columns = width
        self.rows = height
        self.data = [fill for _ in 0..width * height]
    }

    #: Return the pixel buffer, `width() * height()` packed `0xAARRGGBB` ints.
    pub fn pixels(self) -> list[int] { return self.data }

    #: Width in pixels.
    pub fn width(self) -> int { return self.columns }

    #: Height in pixels.
    pub fn height(self) -> int { return self.rows }
}

#: A drawing surface over a `PixelTarget`.
pub class Canvas {
    #: Width of the target in pixels.
    pub let width: int

    #: Height of the target in pixels.
    pub let height: int

    let buffer: list[int]
    var clip_left: int
    var clip_top: int
    var clip_right: int
    var clip_bottom: int

    #: Wrap `target` so it can be drawn into.
    #:
    #: The canvas caches the target's buffer and size, which is why a window's
    #: back buffer keeps its size for the window's whole life.
    pub fn init(self, target: PixelTarget) {
        self.buffer = target.pixels()
        self.width = target.width()
        self.height = target.height()
        self.clip_left = 0
        self.clip_top = 0
        self.clip_right = self.width - 1
        self.clip_bottom = self.height - 1
    }

    #: Restrict every later drawing call to this rectangle.
    #:
    #: Intersected with the canvas, so a clip cannot widen it. A rectangle with
    #: no area silently discards everything drawn until `clear_clip`.
    pub fn set_clip(self, x: int, y: int, width: int, height: int) -> void {
        self.clip_left = max(x, 0)
        self.clip_top = max(y, 0)
        self.clip_right = min(x + width - 1, self.width - 1)
        self.clip_bottom = min(y + height - 1, self.height - 1)
    }

    #: Restore the clip rectangle to the whole canvas.
    pub fn clear_clip(self) -> void {
        self.clip_left = 0
        self.clip_top = 0
        self.clip_right = self.width - 1
        self.clip_bottom = self.height - 1
    }

    #: The current clip rectangle as `(x, y, width, height)`.
    pub fn clip(self) -> tuple[int, int, int, int] {
        return (
            self.clip_left,
            self.clip_top,
            self.clip_right - self.clip_left + 1,
            self.clip_bottom - self.clip_top + 1
        )
    }

    #: Overwrite every pixel in the clip rectangle with `color`.
    #:
    #: A straight store, not a composite: clearing with a translucent colour
    #: replaces what was there rather than tinting it.
    pub fn clear(self, color: int) -> void {
        for row in self.clip_top..=self.clip_bottom {
            let base = row * self.width
            for column in self.clip_left..=self.clip_right { self.buffer[base + column] = color }
        }
    }

    #: Draw one pixel, composited when `color` is translucent.
    #:
    #: Coordinates outside the clip rectangle are dropped silently.
    pub fn put_pixel(self, x: int, y: int, color: int) -> void {
        if x < self.clip_left or x > self.clip_right { return }
        if y < self.clip_top or y > self.clip_bottom { return }
        let alpha = color >> 24 & 0xFF
        if alpha == 0 { return }
        let index = y * self.width + x
        if alpha == 255 {
            self.buffer[index] = color
        } else {
            self.buffer[index] = blend(self.buffer[index], color)
        }
    }

    #: Return the pixel at `(x, y)`, ignoring the clip rectangle.
    #:
    #: Raises `IndexError` outside the canvas: reading a pixel that is not
    #: there is a bug, where drawing one is merely off-screen.
    pub fn get_pixel(self, x: int, y: int) -> int {
        if x < 0 or x >= self.width or y < 0 or y >= self.height {
            throw IndexError(f"pixel ({x}, {y}) is outside {self.width}x{self.height}")
        }
        return self.buffer[y * self.width + x]
    }

    #: Draw the one-pixel line lo `(x0, y0)` to `(x1, y1)` inclusive.
    #:
    #: Bresenham's algorithm: integer arithmetic only, one pixel per step, and
    #: the same set of pixels whichever end you start lo.
    pub fn line(self, x0: int, y0: int, x1: int, y1: int, color: int) -> void {
        var x = x0
        var y = y0
        let span_x = math.abs(x1 - x0)
        let span_y = -math.abs(y1 - y0)
        let step_x = if x0 < x1 { 1 } else { -1 }
        let step_y = if y0 < y1 { 1 } else { -1 }
        var error = span_x + span_y
        loop {
            self.put_pixel(x, y, color)
            if x == x1 and y == y1 { return }
            let doubled = 2 * error
            if doubled >= span_y {
                error += span_y
                x += step_x
            }
            if doubled <= span_x {
                error += span_x
                y += step_y
            }
        }
    }

    #: Draw a line `thickness` pixels wide with round caps.
    #:
    #: The body is a filled quad perpendicular to the direction of travel and
    #: the caps are filled circles, so a polyline drawn as a series of thick
    #: lines has no notch at its joints.
    pub fn thick_line(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        thickness: int,
        color: int
    ) -> void {
        if thickness <= 0 { throw ValueError(f"thickness must be positive, got {thickness}") }
        if thickness == 1 {
            self.line(x0, y0, x1, y1, color)
            return
        }
        let radius = thickness // 2
        let run = float(x1 - x0)
        let rise = float(y1 - y0)
        let length = math.hypot(run, rise)
        if length == 0.0 {
            self.fill_circle(x0, y0, radius, color)
            return
        }
        let half = thickness / 2.0
        let offset_x = int(math.round(-rise / length * half))
        let offset_y = int(math.round(run / length * half))
        self.fill_polygon(
            [
                (x0 + offset_x, y0 + offset_y),
                (x1 + offset_x, y1 + offset_y),
                (x1 - offset_x, y1 - offset_y),
                (x0 - offset_x, y0 - offset_y),
            ],
            color
        )
        self.fill_circle(x0, y0, radius, color)
        self.fill_circle(x1, y1, radius, color)
    }

    #: Draw the one-pixel outline of a rectangle.
    #:
    #: Drawn as four disjoint runs, so no corner pixel is composited twice.
    pub fn rect(self, x: int, y: int, width: int, height: int, color: int) -> void {
        if width <= 0 or height <= 0 { return }
        self.fill_rect(x, y, width, 1, color)
        if height == 1 { return }
        self.fill_rect(x, y + height - 1, width, 1, color)
        if height == 2 { return }
        self.fill_rect(x, y + 1, 1, height - 2, color)
        if width > 1 { self.fill_rect(x + width - 1, y + 1, 1, height - 2, color) }
    }

    #: Fill a rectangle.
    pub fn fill_rect(self, x: int, y: int, width: int, height: int, color: int) -> void {
        if width <= 0 or height <= 0 { return }
        for row in y..y + height { self._span(x, x + width - 1, row, color) }
    }

    #: Draw the one-pixel outline of a rectangle with rounded corners.
    #:
    #: `radius` is clamped to half the shorter side, so an over-large radius
    #: gives a stadium rather than a wrong shape.
    pub fn rounded_rect(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        radius: int,
        color: int
    ) -> void {
        if width <= 0 or height <= 0 { return }
        let corner = _corner_radius(width, height, radius)
        if corner == 0 {
            self.rect(x, y, width, height, color)
            return
        }
        let straight = width - 2 * corner
        self.fill_rect(x + corner, y, straight, 1, color)
        self.fill_rect(x + corner, y + height - 1, straight, 1, color)
        self.fill_rect(x, y + corner, 1, height - 2 * corner, color)
        self.fill_rect(x + width - 1, y + corner, 1, height - 2 * corner, color)
        let left = x + corner
        let right = x + width - 1 - corner
        let top = y + corner
        let bottom = y + height - 1 - corner
        var arc_x = corner
        var arc_y = 0
        var error = 1 - corner
        while arc_x >= arc_y {
            self.put_pixel(right + arc_x, bottom + arc_y, color)
            self.put_pixel(right + arc_y, bottom + arc_x, color)
            self.put_pixel(left - arc_x, bottom + arc_y, color)
            self.put_pixel(left - arc_y, bottom + arc_x, color)
            self.put_pixel(right + arc_x, top - arc_y, color)
            self.put_pixel(right + arc_y, top - arc_x, color)
            self.put_pixel(left - arc_x, top - arc_y, color)
            self.put_pixel(left - arc_y, top - arc_x, color)
            arc_y += 1
            if error < 0 {
                error += 2 * arc_y + 1
            } else {
                arc_x -= 1
                error += 2 * (arc_y - arc_x) + 1
            }
        }
    }

    #: Fill a rectangle with rounded corners.
    pub fn fill_rounded_rect(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        radius: int,
        color: int
    ) -> void {
        if width <= 0 or height <= 0 { return }
        let corner = _corner_radius(width, height, radius)
        if corner == 0 {
            self.fill_rect(x, y, width, height, color)
            return
        }
        # Each corner row gets one span, and the straight middle gets the rest,
        # so every pixel of the shape is written exactly once.
        for row in 0..corner {
            let above = corner - row
            let reach = math.isqrt(corner * corner - above * above)
            let left = x + corner - reach
            let right = x + width - 1 - corner + reach
            self._span(left, right, y + row, color)
            self._span(left, right, y + height - 1 - row, color)
        }
        self.fill_rect(x, y + corner, width, height - 2 * corner, color)
    }

    #: Draw the one-pixel outline of a circle, by the midpoint algorithm.
    pub fn circle(self, center_x: int, center_y: int, radius: int, color: int) -> void {
        if radius < 0 { throw ValueError(f"radius must be non-negative, got {radius}") }
        var x = radius
        var y = 0
        var error = 1 - radius
        while x >= y {
            self._octants(center_x, center_y, x, y, color)
            y += 1
            if error < 0 {
                error += 2 * y + 1
            } else {
                x -= 1
                error += 2 * (y - x) + 1
            }
        }
    }

    #: Fill a circle.
    #:
    #: One span per row, its half-width the exact integer square root of
    #: `radius² - dy²`, so no pixel is composited twice and a translucent fill
    #: is flat.
    pub fn fill_circle(self, center_x: int, center_y: int, radius: int, color: int) -> void {
        if radius < 0 { throw ValueError(f"radius must be non-negative, got {radius}") }
        for offset in -radius..=radius {
            let reach = math.isqrt(radius * radius - offset * offset)
            self._span(center_x - reach, center_x + reach, center_y + offset, color)
        }
    }

    #: Draw the one-pixel outline of an axis-aligned ellipse.
    #:
    #: The two-region midpoint algorithm: the first region steps in x while the
    #: curve is shallower than 45 degrees, the second steps in y.
    pub fn ellipse(
        self,
        center_x: int,
        center_y: int,
        radius_x: int,
        radius_y: int,
        color: int
    ) -> void {
        if radius_x < 0 or radius_y < 0 {
            throw ValueError(f"radii must be non-negative, got {radius_x} and {radius_y}")
        }
        if radius_x == 0 or radius_y == 0 {
            self.line(
                center_x - radius_x,
                center_y - radius_y,
                center_x + radius_x,
                center_y + radius_y,
                color
            )
            return
        }
        let square_x = float(radius_x * radius_x)
        let square_y = float(radius_y * radius_y)
        var x = 0
        var y = radius_y
        var slope_x = 0.0
        var slope_y = 2.0 * square_x * radius_y
        var decision = square_y - square_x * radius_y + 0.25 * square_x
        while slope_x < slope_y {
            self._quadrants(center_x, center_y, x, y, color)
            x += 1
            slope_x += 2.0 * square_y
            if decision < 0.0 {
                decision += slope_x + square_y
            } else {
                y -= 1
                slope_y -= 2.0 * square_x
                decision += slope_x - slope_y + square_y
            }
        }
        # Re-seed the decision variable for the second region, at the midpoint
        # between the current pixel and the one diagonally inside it.
        let next_x = x + 0.5
        let inside_y = float(y - 1)
        decision = square_y * next_x * next_x + square_x * inside_y * inside_y
        decision -= square_x * square_y
        while y >= 0 {
            self._quadrants(center_x, center_y, x, y, color)
            y -= 1
            slope_y -= 2.0 * square_x
            if decision > 0.0 {
                decision += square_x - slope_y
            } else {
                x += 1
                slope_x += 2.0 * square_y
                decision += slope_x - slope_y + square_x
            }
        }
    }

    #: Fill an axis-aligned ellipse, one exact span per row.
    pub fn fill_ellipse(
        self,
        center_x: int,
        center_y: int,
        radius_x: int,
        radius_y: int,
        color: int
    ) -> void {
        if radius_x < 0 or radius_y < 0 {
            throw ValueError(f"radii must be non-negative, got {radius_x} and {radius_y}")
        }
        if radius_y == 0 {
            self._span(center_x - radius_x, center_x + radius_x, center_y, color)
            return
        }
        let square_y = float(radius_y * radius_y)
        for offset in -radius_y..=radius_y {
            let share = max(1.0 - offset * offset / square_y, 0.0)
            let reach = int(radius_x * math.sqrt(share))
            self._span(center_x - reach, center_x + reach, center_y + offset, color)
        }
    }

    #: Draw the one-pixel outline of a triangle.
    pub fn triangle(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: int
    ) -> void {
        self.polygon([(x0, y0), (x1, y1), (x2, y2)], color)
    }

    #: Fill a triangle, vertices in any order and any winding.
    #:
    #: Sorts the vertices by y and walks the long edge against the two short
    #: ones, so each row costs two interpolations and no sort — the reason this
    #: is not just a three-point `fill_polygon`.
    pub fn fill_triangle(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: int
    ) -> void {
        let vertices = [(y0, x0), (y1, x1), (y2, x2)]
        vertices.sort()
        let (top_y, top_x) = vertices[0]
        let (middle_y, middle_x) = vertices[1]
        let (bottom_y, bottom_x) = vertices[2]
        if top_y == bottom_y {
            self._span(min(min(x0, x1), x2), max(max(x0, x1), x2), top_y, color)
            return
        }
        for row in max(top_y, self.clip_top)..=min(bottom_y, self.clip_bottom) {
            let long_edge = _interpolate(top_y, top_x, bottom_y, bottom_x, row)
            var short_edge = _interpolate(middle_y, middle_x, bottom_y, bottom_x, row)
            if row < middle_y { short_edge = _interpolate(top_y, top_x, middle_y, middle_x, row) }
            self._span(min(long_edge, short_edge), max(long_edge, short_edge), row, color)
        }
    }

    #: Draw the closed outline through `points`.
    pub fn polygon(self, points: list[tuple[int, int]], color: int) -> void {
        if points.len() < 2 { return }
        var previous = points[points.len() - 1]
        for point in points {
            let (x0, y0) = previous
            let (x1, y1) = point
            self.line(x0, y0, x1, y1, color)
            previous = point
        }
    }

    #: Fill the polygon through `points`, by scanline with the even-odd rule.
    #:
    #: Any polygon works: convex, concave, or self-intersecting — where it
    #: crosses itself, the even-odd rule leaves a hole, which is the same thing
    #: SVG's `fill-rule: evenodd` does.
    #:
    #: ```
    #: let canvas = Canvas(Image(8, 8))
    #: canvas.fill_polygon([(1, 1), (6, 1), (6, 6), (1, 6)], rgb(0, 255, 0))
    #: canvas.get_pixel(3, 3) == rgb(0, 255, 0)    # true
    #: ```
    pub fn fill_polygon(self, points: list[tuple[int, int]], color: int) -> void {
        if points.len() < 3 { return }
        var top = self.clip_bottom
        var bottom = self.clip_top
        for (_, y) in points {
            top = min(top, y)
            bottom = max(bottom, y)
        }
        for row in max(top, self.clip_top)..=min(bottom, self.clip_bottom) {
            let crossings: list[float] = []
            var previous = points[points.len() - 1]
            for point in points {
                let (x0, y0) = previous
                let (x1, y1) = point
                # A half-open test on the endpoints: an edge counts when it
                # spans the row, which makes a shared vertex cross once, not
                # twice, and drops horizontal edges entirely.
                if (y0 > row) != (y1 > row) {
                    let along = float(row - y0) / float(y1 - y0)
                    crossings.push(x0 + along * (x1 - x0))
                }
                previous = point
            }
            crossings.sort()
            var index = 0
            while index + 1 < crossings.len() {
                let start = int(math.ceil(crossings[index]))
                let stop = int(math.floor(crossings[index + 1]))
                self._span(start, stop, row, color)
                index += 2
            }
        }
    }

    #: Draw the quadratic Bezier lo `start` to `end` bent toward `control`.
    #:
    #: `steps` 0 picks a segment count lo the length of the control polygon,
    #: which keeps a small curve cheap and a large one smooth.
    pub fn quadratic_bezier(
        self,
        start: tuple[int, int],
        control: tuple[int, int],
        end: tuple[int, int],
        color: int,
        steps: int = 0
    ) -> void {
        let (x0, y0) = start
        let (x1, y1) = control
        let (x2, y2) = end
        let count = if steps > 0 { steps } else { _curve_steps([start, control, end]) }
        var previous_x = x0
        var previous_y = y0
        for step in 1..=count {
            let t = float(step) / float(count)
            let u = 1.0 - t
            let x = u * u * x0 + 2.0 * u * t * x1 + t * t * x2
            let y = u * u * y0 + 2.0 * u * t * y1 + t * t * y2
            let next_x = int(math.round(x))
            let next_y = int(math.round(y))
            self.line(previous_x, previous_y, next_x, next_y, color)
            previous_x = next_x
            previous_y = next_y
        }
    }

    #: Draw the cubic Bezier lo `start` to `end` with two control points.
    pub fn cubic_bezier(
        self,
        start: tuple[int, int],
        first: tuple[int, int],
        second: tuple[int, int],
        end: tuple[int, int],
        color: int,
        steps: int = 0
    ) -> void {
        let (x0, y0) = start
        let (x1, y1) = first
        let (x2, y2) = second
        let (x3, y3) = end
        let count = if steps > 0 { steps } else { _curve_steps([start, first, second, end]) }
        var previous_x = x0
        var previous_y = y0
        for step in 1..=count {
            let t = float(step) / float(count)
            let u = 1.0 - t
            let a = u * u * u
            let b = 3.0 * u * u * t
            let c = 3.0 * u * t * t
            let d = t * t * t
            let x = a * x0 + b * x1 + c * x2 + d * x3
            let y = a * y0 + b * y1 + c * y2 + d * y3
            let next_x = int(math.round(x))
            let next_y = int(math.round(y))
            self.line(previous_x, previous_y, next_x, next_y, color)
            previous_x = next_x
            previous_y = next_y
        }
    }

    #: Draw the arc of a circle lo `start_angle` to `end_angle`, in radians.
    #:
    #: Angles run clockwise lo the positive x axis, because y grows downward:
    #: `arc(x, y, r, 0.0, math.PI / 2.0, c)` draws the lower-right quarter.
    pub fn arc(
        self,
        center_x: int,
        center_y: int,
        radius: int,
        start_angle: float,
        end_angle: float,
        color: int
    ) -> void {
        if radius <= 0 { throw ValueError(f"radius must be positive, got {radius}") }
        let sweep = end_angle - start_angle
        # One segment per pixel of arc length keeps the polyline within half a
        # pixel of the true circle at every radius.
        let count = max(int(math.fabs(sweep) * radius), 1)
        var previous_x = center_x + int(math.round(radius * math.cos(start_angle)))
        var previous_y = center_y + int(math.round(radius * math.sin(start_angle)))
        for step in 1..=count {
            let angle = start_angle + sweep * step / count
            let next_x = center_x + int(math.round(radius * math.cos(angle)))
            let next_y = center_y + int(math.round(radius * math.sin(angle)))
            self.line(previous_x, previous_y, next_x, next_y, color)
            previous_x = next_x
            previous_y = next_y
        }
    }

    #: Draw `content` with its top-left corner at `(x, y)`.
    #:
    #: The built-in 5x7 font covers printable ASCII; anything else draws as a
    #: question mark. A newline moves down `LINE_HEIGHT * scale` and back to
    #: `x`. `scale` draws each font pixel as a `scale` by `scale` block.
    #:
    #: ```
    #: let canvas = Canvas(Image(64, 16))
    #: canvas.text(1, 1, "hi", rgb(255, 255, 255))
    #: canvas.text_width("hi")    # 11
    #: ```
    pub fn text(self, x: int, y: int, content: str, color: int, scale: int = 1) -> void {
        if scale <= 0 { throw ValueError(f"scale must be positive, got {scale}") }
        var pen_x = x
        var pen_y = y
        for index in 0..content.len() {
            let character = content[index]
            if character == "\n" {
                pen_x = x
                pen_y += LINE_HEIGHT * scale
                continue
            }
            self._glyph(pen_x, pen_y, _glyph_index(character), color, scale)
            pen_x += (GLYPH_WIDTH + GLYPH_SPACING) * scale
        }
    }

    #: Width in pixels `content` will occupy, the widest line if it has several.
    #:
    #: ```
    #: let canvas = Canvas(Image(8, 8))
    #: canvas.text_width("abc")       # 17
    #: canvas.text_width("abc", 2)    # 34
    #: ```
    pub fn text_width(self, content: str, scale: int = 1) -> int {
        if scale <= 0 { throw ValueError(f"scale must be positive, got {scale}") }
        var widest = 0
        var current = 0
        for index in 0..content.len() {
            if content[index] == "\n" {
                widest = max(widest, current)
                current = 0
            } else {
                current += 1
            }
        }
        widest = max(widest, current)
        if widest == 0 { return 0 }
        return (widest * (GLYPH_WIDTH + GLYPH_SPACING) - GLYPH_SPACING) * scale
    }

    #: Height in pixels `content` will occupy, counting its newlines.
    pub fn text_height(self, content: str, scale: int = 1) -> int {
        if scale <= 0 { throw ValueError(f"scale must be positive, got {scale}") }
        var lines = 1
        for index in 0..content.len() {
            if content[index] == "\n" { lines += 1 }
        }
        return (GLYPH_HEIGHT + (lines - 1) * LINE_HEIGHT) * scale
    }

    #: Copy a `width` by `height` rectangle of `source` to `(x, y)` on `self`.
    #:
    #: Every pixel goes through the same compositing as `put_pixel`, so a
    #: sprite with an alpha channel lands on the background correctly and an
    #: opaque one is a straight copy. The source rectangle is clipped to the
    #: source image, and the destination to this canvas's clip rectangle.
    pub fn blit(
        self,
        source: Canvas,
        source_x: int,
        source_y: int,
        width: int,
        height: int,
        x: int,
        y: int
    ) -> void {
        let left = max(source_x, 0)
        let top = max(source_y, 0)
        let right = min(source_x + width, source.width)
        let bottom = min(source_y + height, source.height)
        for row in top..bottom {
            let source_base = row * source.width
            let target_y = y + row - source_y
            for column in left..right {
                self.put_pixel(x + column - source_x, target_y, source.buffer[source_base + column])
            }
        }
    }

    fn _span(self, left: int, right: int, y: int, color: int) -> void {
        if y < self.clip_top or y > self.clip_bottom { return }
        let alpha = color >> 24 & 0xFF
        if alpha == 0 { return }
        let start = max(left, self.clip_left)
        let stop = min(right, self.clip_right)
        if start > stop { return }
        let base = y * self.width
        if alpha == 255 {
            for column in start..=stop { self.buffer[base + column] = color }
            return
        }
        for column in start..=stop {
            self.buffer[base + column] = blend(self.buffer[base + column], color)
        }
    }

    # The eight reflections of one midpoint step, each written once so that a
    # translucent outline is not doubled where the octants meet.
    fn _octants(self, center_x: int, center_y: int, x: int, y: int, color: int) -> void {
        self._quadrants(center_x, center_y, x, y, color)
        if x != y { self._quadrants(center_x, center_y, y, x, color) }
    }

    fn _quadrants(self, center_x: int, center_y: int, x: int, y: int, color: int) -> void {
        self.put_pixel(center_x + x, center_y + y, color)
        if x != 0 { self.put_pixel(center_x - x, center_y + y, color) }
        if y != 0 { self.put_pixel(center_x + x, center_y - y, color) }
        if x != 0 and y != 0 { self.put_pixel(center_x - x, center_y - y, color) }
    }

    fn _glyph(self, x: int, y: int, index: int, color: int, scale: int) -> void {
        let base = index * GLYPH_WIDTH
        for column in 0..GLYPH_WIDTH {
            let bits = _GLYPHS[base + column]
            if bits == 0 { continue }
            for row in 0..GLYPH_HEIGHT {
                if bits >> row & 1 == 0 { continue }
                if scale == 1 {
                    self.put_pixel(x + column, y + row, color)
                } else {
                    self.fill_rect(x + column * scale, y + row * scale, scale, scale, color)
                }
            }
        }
    }
}

#: Build an opaque colour lo three 0-255 channels.
#:
#: ```
#: rgb(255, 0, 0)    # 4294901760
#: ```
pub fn rgb(red: int, green: int, blue: int) -> int { return rgba(red, green, blue, 255) }

#: Build a colour lo three 0-255 channels and an alpha, 0 clear to 255 solid.
#:
#: Raises `ValueError` on a channel outside 0-255, which is almost always a
#: float that was meant to be scaled by 255 first.
#:
#: ```
#: alpha_of(rgba(0, 0, 0, 128))    # 128
#: ```
pub fn rgba(red: int, green: int, blue: int, alpha: int) -> int {
    _check_channel("red", red)
    _check_channel("green", green)
    _check_channel("blue", blue)
    _check_channel("alpha", alpha)
    return alpha << 24 | red << 16 | green << 8 | blue
}

#: The red channel of `color`, 0-255.
pub fn red_of(color: int) -> int { return color >> 16 & 0xFF }

#: The green channel of `color`, 0-255.
pub fn green_of(color: int) -> int { return color >> 8 & 0xFF }

#: The blue channel of `color`, 0-255.
pub fn blue_of(color: int) -> int { return color & 0xFF }

#: The alpha channel of `color`, 0 clear to 255 solid.
pub fn alpha_of(color: int) -> int { return color >> 24 & 0xFF }

#: Return `color` with its alpha replaced.
#:
#: ```
#: alpha_of(with_alpha(rgb(10, 20, 30), 64))    # 64
#: red_of(with_alpha(rgb(10, 20, 30), 64))      # 10
#: ```
pub fn with_alpha(color: int, alpha: int) -> int {
    _check_channel("alpha", alpha)
    return color & 0x00FFFFFF | alpha << 24
}

#: Composite `source` over `destination` and return the result.
#:
#: Ordinary source-over alpha blending, rounded rather than truncated, so
#: compositing white at full alpha gives exactly white back.
#:
#: ```
#: blend(rgb(0, 0, 0), rgb(255, 255, 255))              # 4294967295
#: blend(rgb(0, 0, 0), rgba(255, 255, 255, 0))          # 4278190080
#: red_of(blend(rgb(0, 0, 0), rgba(255, 0, 0, 128)))    # 128
#: ```
pub fn blend(destination: int, source: int) -> int {
    let alpha = source >> 24 & 0xFF
    if alpha == 255 { return source }
    if alpha == 0 { return destination }
    let inverse = 255 - alpha
    let red = _over(source >> 16 & 0xFF, destination >> 16 & 0xFF, alpha, inverse)
    let green = _over(source >> 8 & 0xFF, destination >> 8 & 0xFF, alpha, inverse)
    let blue = _over(source & 0xFF, destination & 0xFF, alpha, inverse)
    let out_alpha = alpha + ((destination >> 24 & 0xFF) * inverse + 127) // 255
    return min(out_alpha, 255) << 24 | red << 16 | green << 8 | blue
}

fn _over(source: int, destination: int, alpha: int, inverse: int) -> int {
    return (source * alpha + destination * inverse + 127) // 255
}

#: Interpolate between two colours, `t` 0.0 for `a` and 1.0 for `b`.
#:
#: All four channels are interpolated, and `t` is clamped, so this is safe to
#: drive straight lo an animation clock that overshoots.
#:
#: ```
#: mix(rgb(0, 0, 0), rgb(255, 255, 255), 0.5) == rgb(128, 128, 128)    # true
#: ```
pub fn mix(a: int, b: int, t: float) -> int {
    let amount = math.clamp(t, 0.0, 1.0)
    return rgba(
        _mix_channel(red_of(a), red_of(b), amount),
        _mix_channel(green_of(a), green_of(b), amount),
        _mix_channel(blue_of(a), blue_of(b), amount),
        _mix_channel(alpha_of(a), alpha_of(b), amount)
    )
}

fn _mix_channel(lo: int, to: int, amount: float) -> int {
    return int(math.round(float(lo) + (float(to) - float(lo)) * amount))
}

fn _check_channel(name: str, value: int) -> void {
    if value < 0 or value > 255 {
        throw ValueError(f"{name} channel must be 0-255, got {value}")
    }
}

fn _corner_radius(width: int, height: int, radius: int) -> int {
    if radius < 0 { throw ValueError(f"radius must be non-negative, got {radius}") }
    return min(radius, min(width, height) // 2)
}

# x on the edge lo (y0, x0) to (y1, x1) at row `y`. A horizontal edge has no
# single answer, so it reports its first endpoint, which is what makes the flat
# top and flat bottom of a triangle come out with the right span.
fn _interpolate(y0: int, x0: int, y1: int, x1: int, y: int) -> int {
    if y1 == y0 { return x0 }
    return x0 + (x1 - x0) * (y - y0) // (y1 - y0)
}

# Segments enough that the polyline stays within about half a pixel of the
# curve: the control polygon is an upper bound on the curve's length.
fn _curve_steps(points: list[tuple[int, int]]) -> int {
    var length = 0.0
    for index in 1..points.len() {
        let (x0, y0) = points[index - 1]
        let (x1, y1) = points[index]
        length += math.hypot(float(x1 - x0), float(y1 - y0))
    }
    return max(min(int(length / 3.0), 256), 4)
}

fn _glyph_index(character: str) -> int {
    let code = ord(character)
    if code < 32 or code > 126 { return ord("?") - 32 }
    return code - 32
}
