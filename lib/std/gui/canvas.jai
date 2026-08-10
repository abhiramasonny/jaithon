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
    var clip_stack: list[tuple[int, int, int, int]]

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
        self.clip_stack = []
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
        let count = self.clip_right - self.clip_left + 1
        if count <= 0 { return }
        for row in self.clip_top..=self.clip_bottom {
            __prim__.canvas_fill_span(self.buffer, row * self.width + self.clip_left, count, color)
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

    #: Narrow the clip to the intersection of the current one and this
    #: rectangle, remembering what it was.
    #:
    #: Unlike `set_clip`, which intersects with the whole canvas and can
    #: therefore widen a clip already in force, this can only ever restrict.
    #: Nesting is what a chart needs: an axis clips to its plot area, a series
    #: clips again inside that, and neither has to know about the other.
    #:
    #: ```
    #: let canvas = Canvas(Image(20, 20))
    #: canvas.push_clip(0, 0, 10, 10)
    #: canvas.push_clip(5, 5, 100, 100)
    #: canvas.clip()          # (5, 5, 5, 5)
    #: canvas.pop_clip()
    #: canvas.clip()          # (0, 0, 10, 10)
    #: ```
    pub fn push_clip(self, x: int, y: int, width: int, height: int) -> void {
        self.clip_stack.push((self.clip_left, self.clip_top, self.clip_right, self.clip_bottom))
        self.clip_left = max(self.clip_left, x)
        self.clip_top = max(self.clip_top, y)
        self.clip_right = min(self.clip_right, x + width - 1)
        self.clip_bottom = min(self.clip_bottom, y + height - 1)
    }

    #: Restore the clip rectangle in force before the matching `push_clip`.
    #:
    #: Raises `IndexError` when nothing was pushed, since an unbalanced pop is a
    #: bug in the caller rather than something to paper over.
    pub fn pop_clip(self) -> void {
        if self.clip_stack.len() == 0 {
            throw IndexError("pop_clip with no matching push_clip")
        }
        let (left, top, right, bottom) = self.clip_stack.pop()
        self.clip_left = left
        self.clip_top = top
        self.clip_right = right
        self.clip_bottom = bottom
    }

    #: Fill `path` with `color`, anti-aliased.
    #:
    #: Coverage is computed exactly across each row and sampled `SUBSAMPLES`
    #: times down it, then one native call composites the whole row. Because the
    #: whole path is resolved into a single coverage value per pixel before
    #: anything is written, a shape that overlaps itself still writes each pixel
    #: once — which is what lets a translucent stroke avoid the dark knots the
    #: integer outline routines leave at their joins.
    #:
    #: ```
    #: let image = Image(10, 10, 0xFF000000)
    #: let canvas = Canvas(image)
    #: let path = Path()
    #: path.polygon([(1.0, 1.0), (9.0, 1.0), (9.0, 9.0), (1.0, 9.0)])
    #: canvas.fill_path(path, rgb(255, 255, 255))
    #: red_of(canvas.get_pixel(5, 5))      # 255
    #: ```
    pub fn fill_path(self, path: Path, color: int, winding: Winding = Winding.NonZero) -> void {
        if color >> 24 & 0xFF == 0 { return }
        # Almost everything a chart draws is an axis-aligned rectangle: every
        # gridline, tick, spine, bar, background and legend panel. Their coverage
        # is a product of two independent overlaps rather than something to
        # sample, so they skip the scanline machinery entirely.
        let box = _axis_aligned_rect(path)
        if box != null {
            let (left, top, right, bottom) = box
            self._fill_axis_rect(left, top, right, bottom, color)
            return
        }
        let edges = _edges_of(path)
        if edges.len() == 0 { return }

        var top = edges[0].y0
        var bottom = edges[0].y1
        var left = edges[0].x0
        var right = edges[0].x0
        for edge in edges {
            top = min(top, edge.y0)
            bottom = max(bottom, edge.y1)
            left = min(left, min(edge.x0, edge.x1))
            right = max(right, max(edge.x0, edge.x1))
        }

        let first_row = max(int(math.floor(top)), self.clip_top)
        let last_row = min(int(math.ceil(bottom)), self.clip_bottom)
        let first_column = max(int(math.floor(left)), self.clip_left)
        let last_column = min(int(math.ceil(right)), self.clip_right)
        if first_row > last_row or first_column > last_column { return }

        let share = 255.0 / SUBSAMPLES.to_float()

        # An active edge list. Testing every edge at every sub-scanline is
        # quadratic in the shape's complexity, and a stroked series is thousands
        # of edges over hundreds of rows -- millions of tests for a picture where
        # only a handful of edges cross any given row. Sorting once by the top of
        # each edge and carrying a window down the shape makes the work
        # proportional to what actually crosses the scanline.
        let ordered = edges.sorted(|edge| edge.y0)
        var pending = 0
        var active = []

        for row in first_row..=last_row {
            let row_bottom = (row + 1).to_float()
            while pending < ordered.len() and ordered[pending].y0 < row_bottom {
                active.push(ordered[pending])
                pending += 1
            }
            if active.len() == 0 { continue }
            let row_top = row.to_float()
            active = [edge for edge in active if edge.y1 > row_top]
            if active.len() == 0 { continue }

            # Gather every sub-scanline first so the coverage buffer can be cut
            # to the width this row actually needs. Sizing it to the whole shape
            # instead would allocate a float per pixel of the bounding box on
            # every row, which for a curve spanning the plot is most of the cost
            # of the frame and nearly all of it wasted on empty space.
            var samples = []
            var row_left = 0.0
            var row_right = 0.0
            var found = false
            for sample in 0..SUBSAMPLES {
                let y = row_top + (sample.to_float() + 0.5) / SUBSAMPLES.to_float()
                let hits = _crossings(active, y)
                if hits.len() < 2 { continue }
                let (low, _low_direction) = hits[0]
                let (high, _high_direction) = hits[hits.len() - 1]
                if not found {
                    row_left = low
                    row_right = high
                    found = true
                } else {
                    row_left = min(row_left, low)
                    row_right = max(row_right, high)
                }
                samples.push(hits)
            }
            if not found { continue }

            let window_left = max(int(math.floor(row_left)), first_column)
            let window_right = min(int(math.ceil(row_right)), last_column)
            if window_left > window_right { continue }

            # A row whose sub-scanlines each cover one interval -- a rectangle,
            # a bar, a legend panel, a marker, the figure background -- has a
            # solid middle with a fringe of at most one pixel on each side. The
            # middle is written with a single native call, which is the
            # difference between a chart that animates and one that does not.
            var intervals = []
            var single = true
            for hits in samples {
                let spans = _spans(hits, winding)
                if spans.len() != 1 { single = false }
                intervals.push(spans)
            }

            if single and intervals.len() == SUBSAMPLES {
                if self._solid_row(row, intervals, window_left, window_right, color) { continue }
            }

            # A curve crosses one row in several places, so the row's extent is
            # wide while the ink on it is a few pixels at each crossing. Working
            # per cluster of overlapping spans keeps the per-pixel cost
            # proportional to the ink rather than to the distance between the
            # leftmost and rightmost crossing.
            var flat = []
            for spans in intervals {
                for interval in spans { flat.push(interval) }
            }
            if flat.len() == 0 { continue }
            let ordered_spans = flat.sorted(|interval| interval[0])

            var clusters = []
            let (opening, closing) = ordered_spans[0]
            var cluster_start = opening
            var cluster_end = closing
            for index in 1..ordered_spans.len() {
                let (start, end) = ordered_spans[index]
                # Within a pixel counts as touching: two spans that close on the
                # same pixel share its coverage and must be blended together.
                if start <= cluster_end + 1.0 {
                    cluster_end = max(cluster_end, end)
                } else {
                    clusters.push((cluster_start, cluster_end))
                    cluster_start = start
                    cluster_end = end
                }
            }
            clusters.push((cluster_start, cluster_end))

            for (start, end) in clusters {
                let cluster_left = max(int(math.floor(start)), window_left)
                let cluster_right = min(int(math.ceil(end)), window_right)
                if cluster_left > cluster_right { continue }
                let reach = cluster_right - cluster_left + 1
                var coverage = [0.0 for _index in 0..reach]
                var touched = false
                for spans in intervals {
                    for (low, high) in spans {
                        if high < start or low > end { continue }
                        let shifted_low = low - cluster_left.to_float()
                        let shifted_high = high - cluster_left.to_float()
                        if _add_span(coverage, shifted_low, shifted_high, reach, share) {
                            touched = true
                        }
                    }
                }
                if touched { self._blend_coverage(row, cluster_left, coverage, color) }
            }
        }
    }

    #: Fill an axis-aligned rectangle with exact coverage on all four edges.
    #:
    #: Vertical coverage is how much of the row the rectangle spans and
    #: horizontal coverage how much of the column, so a pixel's coverage is
    #: simply their product. A run of whole rows with a whole interior needs one
    #: native call each and no per-pixel work at all.
    fn _fill_axis_rect(
        self,
        left: float,
        top: float,
        right: float,
        bottom: float,
        color: int
    ) -> void {
        let first_row = max(int(math.floor(top)), self.clip_top)
        let last_row = min(int(math.ceil(bottom)) - 1, self.clip_bottom)
        let first_column = max(int(math.floor(left)), self.clip_left)
        let last_column = min(int(math.ceil(right)) - 1, self.clip_right)
        if first_row > last_row or first_column > last_column { return }

        # Horizontal coverage is the same on every row, so it is built once.
        var columns = []
        for column in first_column..=last_column {
            let low = max(left, column.to_float())
            let high = min(right, column.to_float() + 1.0)
            columns.push(max(0.0, high - low))
        }

        let opaque = color >> 24 & 0xFF == 255
        var solid = true
        for share in columns {
            if share < 0.999 { solid = false }
        }

        let base_offset = first_column
        let count = last_column - first_column + 1
        for row in first_row..=last_row {
            let low = max(top, row.to_float())
            let high = min(bottom, row.to_float() + 1.0)
            let vertical = max(0.0, high - low)
            if vertical <= 0.0 { continue }
            let offset = row * self.width + base_offset
            if solid and vertical >= 0.999 and opaque {
                __prim__.canvas_fill_span(self.buffer, offset, count, color)
                continue
            }
            var coverage = []
            for share in columns {
                coverage.push(min(255, max(0, int(share * vertical * 255.0 + 0.5))))
            }
            __prim__.canvas_blend_span(self.buffer, offset, coverage, color)
        }
    }

    #: Fill a float-coordinate rectangle, anti-aliased on every edge.
    #:
    #: The shape a chart draws most, and cheap enough to be worth reaching
    #: directly rather than through a `Path`. A particle cloud drawn as small
    #: boxes is thousands of these a frame.
    #:
    #: ```
    #: let canvas = Canvas(Image(8, 8))
    #: canvas.fill_rect_aa(1.5, 1.5, 3.0, 3.0, rgb(255, 255, 255))
    #: red_of(canvas.get_pixel(3, 3))      # 255
    #: ```
    pub fn fill_rect_aa(self, x: float, y: float, width: float, height: float, color: int) -> void {
        if width <= 0.0 or height <= 0.0 { return }
        if color >> 24 & 0xFF == 0 { return }
        self._fill_axis_rect(x, y, x + width, y + height, color)
    }

    #: Fill a small convex polygon, anti-aliased.
    #:
    #: A convex shape crosses any scanline exactly twice, so its span is a
    #: minimum and a maximum rather than something to sort and pair up. Skipping
    #: the general machinery matters because the shape this is for — a particle
    #: drawn as an oriented wedge — is drawn thousands of times a frame, and
    #: there the per-shape setup is the whole cost.
    #:
    #: The polygon must be convex; a concave one comes out filled to its hull.
    pub fn fill_convex(self, points: list[tuple[float, float]], color: int) -> void {
        if points.len() < 3 { return }
        if color >> 24 & 0xFF == 0 { return }
        var flat = []
        for (x, y) in points {
            flat.push(x)
            flat.push(y)
        }
        __prim__.canvas_fill_convex(
            self.buffer,
            self.width,
            self.height,
            flat,
            color,
            self.clip_left,
            self.clip_top,
            self.clip_right,
            self.clip_bottom
        )
    }

    #: Blit a pixel buffer into a rectangle, scaling by nearest neighbour.
    #:
    #: Nearest neighbour rather than a smooth filter because the usual source is
    #: data — an occupancy grid, a heatmap — and interpolating would invent
    #: values that were never measured.
    #:
    #: Raises `ValueError` when the buffer is smaller than the size it claims.
    pub fn blit_pixels(
        self,
        pixels: list[int],
        source_width: int,
        source_height: int,
        x: int,
        y: int,
        width: int,
        height: int
    ) -> void {
        if source_width <= 0 or source_height <= 0 { return }
        if pixels.len() < source_width * source_height {
            throw ValueError(
                f"a {source_width}x{source_height} buffer needs " +
                    f"{source_width * source_height} pixels, got {pixels.len()}"
            )
        }
        if width <= 0 or height <= 0 { return }

        # Straight to the native blitter: building a row of pixels as a Jaithon
        # list and handing that over costs a push per pixel, which for a full
        # window is most of a frame spent copying an image that has not changed.
        __prim__.canvas_blit_scaled(
            self.buffer,
            self.width,
            pixels,
            source_width,
            source_height,
            0,
            0,
            source_width,
            source_height,
            x,
            y,
            width,
            height
        )
    }

    #: Stroke `path` with `color`, anti-aliased.
    #:
    #: The outline is built as a single filled path — a quad per segment, a join
    #: at every interior vertex, a cap at each end — and filled once with
    #: non-zero winding. Overlaps between those pieces therefore cancel instead
    #: of compositing twice, so a translucent stroke is evenly translucent all
    #: the way along.
    pub fn stroke_path(
        self,
        path: Path,
        color: int,
        width: float = 1.0,
        cap: Cap = Cap.Butt,
        join: Join = Join.Miter,
        dash: list[float] = []
    ) -> void {
        if width <= 0.0 or color >> 24 & 0xFF == 0 { return }

        # An opaque stroke can be laid down one segment at a time, because
        # writing an opaque pixel twice is the same as writing it once. Each
        # segment is then a convex quad and goes straight to the native filler,
        # instead of the whole run becoming one large non-convex ring whose
        # every scanline meets most of the path.
        #
        # A translucent one cannot: the overlaps at the joins would composite
        # twice and show as dark knots, which is the bug the single-ring form
        # exists to avoid.
        if color >> 24 & 0xFF == 255 and dash.len() == 0 {
            self._stroke_segments(path, color, width, cap, join)
            return
        }

        let outline = Path()
        for run in path.subpaths() {
            for piece in _dashed(run, dash) {
                _outline_run(outline, piece, width, cap, join)
            }
        }
        self.fill_path(outline, color, Winding.NonZero)
    }

    #: Stroke each segment as its own quad, with a disc at every corner.
    fn _stroke_segments(self, path: Path, color: int, width: float, cap: Cap, join: Join) -> void {
        let half = width * 0.5
        for run in path.subpaths() {
            let count = run.len() // 2
            if count < 2 { continue }
            for segment in 0..count - 1 {
                let x0 = run[segment * 2]
                let y0 = run[segment * 2 + 1]
                let x1 = run[(segment + 1) * 2]
                let y1 = run[(segment + 1) * 2 + 1]
                let length = math.hypot(x1 - x0, y1 - y0)
                if length <= 0.0 { continue }
                let nx = -(y1 - y0) / length * half
                let ny = (x1 - x0) / length * half
                var ax = x0
                var ay = y0
                var bx = x1
                var by = y1
                if cap == Cap.Square {
                    let ex = (x1 - x0) / length * half
                    let ey = (y1 - y0) / length * half
                    if segment == 0 {
                        ax -= ex
                        ay -= ey
                    }
                    if segment == count - 2 {
                        bx += ex
                        by += ey
                    }
                }
                let quad = [
                    (ax + nx, ay + ny),
                    (bx + nx, by + ny),
                    (bx - nx, by - ny),
                    (ax - nx, ay - ny),
                ]
                self.fill_convex(quad, color)
            }
            if join != Join.Bevel and count > 2 {
                for vertex in 1..count - 1 {
                    if _turns_visibly(run, vertex, half) {
                        self._fill_disc(run[vertex * 2], run[vertex * 2 + 1], half, color)
                    }
                }
            }
            if cap == Cap.Round {
                self._fill_disc(run[0], run[1], half, color)
                self._fill_disc(run[(count - 1) * 2], run[(count - 1) * 2 + 1], half, color)
            }
        }
    }

    #: A filled disc, as a polygon fine enough for its size.
    fn _fill_disc(self, x: float, y: float, radius: float, color: int) -> void {
        if radius <= 0.0 { return }
        let steps = max(6, min(24, int(radius * 4.0)))
        var points = []
        for index in 0..steps {
            let angle = math.TAU * index.to_float() / steps.to_float()
            points.push((x + radius * math.cos(angle), y + radius * math.sin(angle)))
        }
        self.fill_convex(points, color)
    }

    #: Draw `content` turned by `quarter_turns` anticlockwise about `(x, y)`.
    #:
    #: The glyph bitmaps are transposed rather than rotated, so only whole
    #: quarter turns exist. One turn is what a conventional y-axis label wants.
    #:
    #: ```
    #: let canvas = Canvas(Image(40, 40))
    #: canvas.text_rotated(20, 30, "hi", rgb(255, 255, 255), 1)
    #: canvas.text_width("hi")      # 11
    #: ```
    pub fn text_rotated(
        self,
        x: int,
        y: int,
        content: str,
        color: int,
        quarter_turns: int,
        scale: int = 1
    ) -> void {
        let turns = (quarter_turns % 4 + 4) % 4
        if turns == 0 {
            self.text(x, y, content, color, scale)
            return
        }
        var pen = 0
        for character in content.chars() {
            self._glyph_turned(x, y, pen, _glyph_index(character), color, scale, turns)
            pen += (GLYPH_WIDTH + GLYPH_SPACING) * scale
        }
    }

    #: One glyph, transposed into the turned frame.
    fn _glyph_turned(
        self,
        origin_x: int,
        origin_y: int,
        advance: int,
        index: int,
        color: int,
        scale: int,
        turns: int
    ) -> void {
        let base = index * GLYPH_WIDTH
        for column in 0..GLYPH_WIDTH {
            let bits = _GLYPHS[base + column]
            for row in 0..GLYPH_HEIGHT {
                if bits >> row & 1 == 0 { continue }
                let local_x = advance + column * scale
                let local_y = row * scale
                # Anticlockwise on screen, where y grows downward: one turn
                # sends +x upward and +y rightward.
                let (dx, dy) = match turns {
                    1 => (local_y, -local_x),
                    2 => (-local_x, -local_y),
                    _ => (-local_y, local_x),
                }
                self._fill_cell(origin_x + dx, origin_y + dy, scale, color)
            }
        }
    }

    fn _fill_cell(self, x: int, y: int, scale: int, color: int) -> void {
        if scale <= 1 {
            self.put_pixel(x, y, color)
            return
        }
        for row in 0..scale { self._span(x, x + scale - 1, y + row, color) }
    }

    #: Write a row whose interior every sub-scanline covers, or report that it
    #: has none worth the special case.
    #:
    #: The fringe pixels either side still need real coverage, so they go
    #: through the per-pixel path; only the solid middle is handed over whole.
    fn _solid_row(
        self,
        row: int,
        intervals: list[list[tuple[float, float]]],
        window_left: int,
        window_right: int,
        color: int
    ) -> bool {
        let (first_start, first_end) = intervals[0][0]
        var widest_start = first_start
        var narrow_end = first_end
        var narrow_start = first_start
        var widest_end = first_end
        for spans in intervals {
            let (start, end) = spans[0]
            widest_start = max(widest_start, start)
            narrow_start = min(narrow_start, start)
            narrow_end = min(narrow_end, end)
            widest_end = max(widest_end, end)
        }

        let solid_from = max(int(math.ceil(widest_start)), window_left)
        let solid_to = min(int(math.floor(narrow_end)) - 1, window_right)
        # Below a handful of pixels the bookkeeping costs more than the loop it
        # replaces, so a narrow run is left to the general path.
        if solid_to - solid_from < 4 { return false }

        let alpha = color >> 24 & 0xFF
        if alpha == 255 {
            __prim__.canvas_fill_span(
                self.buffer,
                row * self.width + solid_from,
                solid_to - solid_from + 1,
                color
            )
        } else {
            let full = [255 for _index in solid_from..=solid_to]
            __prim__.canvas_blend_span(self.buffer, row * self.width + solid_from, full, color)
        }

        self._fringe(
            row,
            max(int(math.floor(narrow_start)), window_left),
            solid_from - 1,
            intervals,
            color
        )
        self._fringe(
            row,
            solid_to + 1,
            min(int(math.ceil(widest_end)), window_right),
            intervals,
            color
        )
        return true
    }

    #: Composite the partially covered pixels at one end of a solid run.
    fn _fringe(
        self,
        row: int,
        from_column: int,
        to_column: int,
        intervals: list[list[tuple[float, float]]],
        color: int
    ) -> void {
        if from_column > to_column { return }
        let share = 255.0 / SUBSAMPLES.to_float()
        var coverage = []
        for column in from_column..=to_column {
            var total = 0.0
            for spans in intervals {
                let (start, end) = spans[0]
                let low = max(start, column.to_float())
                let high = min(end, column.to_float() + 1.0)
                if high > low { total += share * (high - low) }
            }
            coverage.push(min(255, max(0, int(total + 0.5))))
        }
        __prim__.canvas_blend_span(self.buffer, row * self.width + from_column, coverage, color)
    }

    #: Composite one row of coverage, trimming the fully transparent ends.
    fn _blend_coverage(self, row: int, first: int, coverage: list[float], color: int) -> void {
        var start = 0
        while start < coverage.len() and coverage[start] < 0.5 { start += 1 }
        if start >= coverage.len() { return }
        var stop = coverage.len() - 1
        while stop > start and coverage[stop] < 0.5 { stop -= 1 }

        var bytes = []
        for index in start..=stop {
            bytes.push(min(255, max(0, int(coverage[index] + 0.5))))
        }
        __prim__.canvas_blend_span(self.buffer, row * self.width + first + start, bytes, color)
    }

    fn _span(self, left: int, right: int, y: int, color: int) -> void {
        if y < self.clip_top or y > self.clip_bottom { return }
        let alpha = color >> 24 & 0xFF
        if alpha == 0 { return }
        let start = max(left, self.clip_left)
        let stop = min(right, self.clip_right)
        if start > stop { return }
        let base = y * self.width
        # Every filled primitive funnels through here, so this run is the one
        # place worth handing to the native layer: a whole row of pixels per
        # call instead of one interpreted store each.
        if alpha == 255 {
            __prim__.canvas_fill_span(self.buffer, base + start, stop - start + 1, color)
            return
        }
        let coverage = [255 for _column in start..=stop]
        __prim__.canvas_blend_span(self.buffer, base + start, coverage, color)
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

#: How a filled path decides which side of an edge is inside.
#:
#: `NonZero` counts edge directions and fills wherever they do not cancel, so a
#: self-crossing outline is solid. `EvenOdd` fills where a ray crosses an odd
#: number of edges, which leaves a hole wherever the shape overlaps itself.
pub enum Winding { NonZero, EvenOdd }

#: How a stroke finishes at an open end.
pub enum Cap { Butt, Round, Square }

#: How two segments of a stroke meet.
pub enum Join { Miter, Round, Bevel }

#: Sub-scanlines sampled per output row when filling a path.
#:
#: Horizontal coverage is computed exactly, so only the vertical direction is
#: sampled. Four rows put the remaining error below what the eye resolves on a
#: chart while keeping the inner loop short.
pub const SUBSAMPLES = 4

#: Segments a curve is flattened into.
pub const CURVE_STEPS = 16

#: A sequence of straight and curved segments in float coordinates.
#:
#: Paths exist so a shape can be described once and then filled or stroked with
#: anti-aliasing, which the integer primitives above cannot do. A path may hold
#: several subpaths; `move_to` begins each one.
#:
#: ```
#: let path = Path()
#: path.move_to(0.0, 0.0)
#: path.line_to(10.0, 0.0)
#: path.line_to(10.0, 10.0)
#: path.close()
#: path.subpaths().len()      # 1
#: ```
pub class Path {
    var runs: list[list[float]]
    var current: list[float]

    #: Create an empty path.
    pub fn init(self) {
        self.runs = []
        self.current = []
    }

    #: Begin a new subpath at `(x, y)`.
    pub fn move_to(self, x: float, y: float) -> Path {
        self._flush()
        self.current = [x, y]
        return self
    }

    #: Extend the current subpath with a straight segment.
    #:
    #: Raises `ValueError` when no subpath has been started, since there is no
    #: point to draw from.
    pub fn line_to(self, x: float, y: float) -> Path {
        if self.current.len() == 0 {
            throw ValueError("line_to before move_to: the path has no current point")
        }
        self.current.push(x)
        self.current.push(y)
        return self
    }

    #: Extend the current subpath with a cubic Bezier, flattened as it is added.
    pub fn cubic_to(
        self,
        cx0: float,
        cy0: float,
        cx1: float,
        cy1: float,
        x: float,
        y: float
    ) -> Path {
        if self.current.len() == 0 {
            throw ValueError("cubic_to before move_to: the path has no current point")
        }
        let x0 = self.current[self.current.len() - 2]
        let y0 = self.current[self.current.len() - 1]
        for step in 1..=CURVE_STEPS {
            let t = step.to_float() / CURVE_STEPS.to_float()
            let u = 1.0 - t
            let a = u * u * u
            let b = 3.0 * u * u * t
            let c = 3.0 * u * t * t
            let d = t * t * t
            self.current.push(a * x0 + b * cx0 + c * cx1 + d * x)
            self.current.push(a * y0 + b * cy0 + c * cy1 + d * y)
        }
        return self
    }

    #: Close the current subpath, joining its end back to its start.
    pub fn close(self) -> Path {
        if self.current.len() >= 4 {
            self.current.push(self.current[0])
            self.current.push(self.current[1])
        }
        self._flush()
        return self
    }

    #: Add a whole subpath from a list of points, closing it.
    pub fn polygon(self, points: list[tuple[float, float]]) -> Path {
        if points.len() == 0 { return self }
        let (first_x, first_y) = points[0]
        let _started = self.move_to(first_x, first_y)
        for index in 1..points.len() {
            let (x, y) = points[index]
            let _extended = self.line_to(x, y)
        }
        return self.close()
    }

    #: The subpaths, each a flat list of alternating x and y coordinates.
    pub fn subpaths(self) -> list[list[float]] {
        var out = [run for run in self.runs]
        if self.current.len() >= 4 { out.push(self.current) }
        return out
    }

    #: Report whether the path holds nothing drawable.
    pub fn is_empty(self) -> bool {
        return self.subpaths().len() == 0
    }

    fn _flush(self) -> void {
        if self.current.len() >= 4 { self.runs.push(self.current) }
        self.current = []
    }

    fn __str__(self) -> str {
        return f"Path({self.subpaths().len()} subpaths)"
    }
}

#: One edge of a flattened path, kept in the order the path walked it.
class _Edge {
    pub let x0: float
    pub let y0: float
    pub let x1: float
    pub let y1: float
    pub let direction: int

    pub fn init(self, x0: float, y0: float, x1: float, y1: float) {
        # Edges are stored top to bottom so a scanline test is one comparison,
        # and the direction that was lost in the swap is kept for the winding
        # count.
        if y0 <= y1 {
            self.x0 = x0
            self.y0 = y0
            self.x1 = x1
            self.y1 = y1
            self.direction = 1
        } else {
            self.x0 = x1
            self.y0 = y1
            self.x1 = x0
            self.y1 = y0
            self.direction = -1
        }
    }

    #: Where this edge crosses the horizontal line at `y`.
    pub fn crossing(self, y: float) -> float {
        let span = self.y1 - self.y0
        if span == 0.0 { return self.x0 }
        return self.x0 + (y - self.y0) * (self.x1 - self.x0) / span
    }
}

#: The rectangle a path describes, when it is exactly one axis-aligned box.
#:
#: Returned as `(left, top, right, bottom)` in device order, or null when the
#: path is anything else.
fn _axis_aligned_rect(path: Path) -> tuple[float, float, float, float]? {
    let runs = path.subpaths()
    if runs.len() != 1 { return null }
    let run = runs[0]
    let count = run.len() // 2
    # Four corners, or five when `close` repeated the first.
    if count != 4 and count != 5 { return null }
    if count == 5 {
        if run[0] != run[8] or run[1] != run[9] { return null }
    }

    var left = run[0]
    var right = run[0]
    var top = run[1]
    var bottom = run[1]
    for index in 0..4 {
        left = min(left, run[index * 2])
        right = max(right, run[index * 2])
        top = min(top, run[index * 2 + 1])
        bottom = max(bottom, run[index * 2 + 1])
    }
    # Every corner must sit on the box, and every edge must be axis aligned.
    for index in 0..4 {
        let x = run[index * 2]
        let y = run[index * 2 + 1]
        if x != left and x != right { return null }
        if y != top and y != bottom { return null }
        let next = (index + 1) % 4
        let nx = run[next * 2]
        let ny = run[next * 2 + 1]
        if x != nx and y != ny { return null }
    }
    if right <= left or bottom <= top { return null }
    return (left, top, right, bottom)
}

#: Every edge of every subpath, implicitly closed.
#:
#: A fill treats an open subpath as closed, which is what makes a polygon
#: described by its corners alone come out solid.
fn _edges_of(path: Path) -> list[_Edge] {
    var edges = []
    for run in path.subpaths() {
        let count = run.len() // 2
        if count < 2 { continue }
        for index in 0..count {
            let next = (index + 1) % count
            let x0 = run[index * 2]
            let y0 = run[index * 2 + 1]
            let x1 = run[next * 2]
            let y1 = run[next * 2 + 1]
            # A horizontal edge contributes no crossing and would only divide by
            # zero, so it is dropped rather than special-cased downstream.
            if y0 != y1 { edges.push(_Edge(x0, y0, x1, y1)) }
        }
    }
    return edges
}

#: Where the edges cross the line at `y`, sorted left to right.
#:
#: Each entry is the x position paired with the edge's direction, which the
#: winding rule needs.
fn _crossings(edges: list[_Edge], y: float) -> list[tuple[float, int]] {
    var hits = []
    for edge in edges {
        # Half-open in y: a vertex shared by two edges is counted once, so
        # abutting shapes neither double-cover nor leave a seam.
        if y >= edge.y0 and y < edge.y1 { hits.push((edge.crossing(y), edge.direction)) }
    }
    hits.sort(|entry| entry[0])
    return hits
}

#: The intervals a sub-scanline covers, in device x.
#:
#: Separating this from the accumulation is what lets a row notice that its
#: interior is solid and hand a whole run to the native span writer instead of
#: incrementing a coverage value per pixel.
fn _spans(hits: list[tuple[float, int]], winding: Winding) -> list[tuple[float, float]] {
    var out = []
    var count = 0
    for index in 0..hits.len() - 1 {
        let (x, direction) = hits[index]
        count += match winding {
            Winding.NonZero => direction,
            Winding.EvenOdd => 1,
        }
        let inside = match winding {
            Winding.NonZero => count != 0,
            Winding.EvenOdd => count % 2 != 0,
        }
        if not inside { continue }
        let (next_x, _next_direction) = hits[index + 1]
        if next_x > x { out.push((x, next_x)) }
    }
    return out
}

#: Add one sub-scanline's spans into `coverage`, returning whether any landed.
fn _accumulate(
    coverage: list[float],
    hits: list[tuple[float, int]],
    winding: Winding,
    first_column: int,
    span: int,
    share: float
) -> bool {
    var touched = false
    var count = 0
    for index in 0..hits.len() - 1 {
        let (x, direction) = hits[index]
        count += match winding {
            Winding.NonZero => direction,
            Winding.EvenOdd => 1,
        }
        let inside = match winding {
            Winding.NonZero => count != 0,
            Winding.EvenOdd => count % 2 != 0,
        }
        if not inside { continue }
        let (next_x, _next_direction) = hits[index + 1]
        if _add_span(
            coverage,
            x - first_column.to_float(),
            next_x - first_column.to_float(),
            span,
            share
        ) {
            touched = true
        }
    }
    return touched
}

#: Add `share` of coverage over `[from, to)`, exactly at the fractional ends.
#:
#: The partial pixels at each end are what make the edge smooth: a span ending
#: three tenths of the way into a pixel contributes three tenths of the share to
#: it.
fn _add_span(coverage: list[float], start: float, end: float, span: int, share: float) -> bool {
    let left = max(start, 0.0)
    let right = min(end, span.to_float())
    if right <= left { return false }

    let first = int(math.floor(left))
    let last = int(math.ceil(right)) - 1
    if first > last { return false }

    for column in first..=last {
        if column < 0 or column >= span { continue }
        let low = max(left, column.to_float())
        let high = min(right, column.to_float() + 1.0)
        if high > low { coverage[column] += share * (high - low) }
    }
    return true
}

#: Split one subpath into the pieces a dash pattern leaves visible.
#:
#: An empty pattern returns the run unchanged, which is the solid case and the
#: common one.
fn _dashed(run: list[float], dash: list[float]) -> list[list[float]] {
    if dash.len() == 0 { return [run] }
    var total = 0.0
    for length in dash { total += length }
    if total <= 0.0 { return [run] }

    var pieces = []
    var current = []
    var index = 0
    var remaining = dash[0]
    var on = true

    let count = run.len() // 2
    for segment in 0..count - 1 {
        let x0 = run[segment * 2]
        let y0 = run[segment * 2 + 1]
        let x1 = run[(segment + 1) * 2]
        let y1 = run[(segment + 1) * 2 + 1]
        let length = math.hypot(x1 - x0, y1 - y0)
        if length <= 0.0 { continue }
        let ux = (x1 - x0) / length
        let uy = (y1 - y0) / length

        var travelled = 0.0
        while travelled < length {
            let step = min(length - travelled, remaining)
            if on {
                if current.len() == 0 {
                    current.push(x0 + ux * travelled)
                    current.push(y0 + uy * travelled)
                }
                current.push(x0 + ux * (travelled + step))
                current.push(y0 + uy * (travelled + step))
            }
            travelled += step
            remaining -= step
            # Floating point will not land exactly on zero, and a pattern that
            # never advanced would spin here for ever.
            if remaining <= 1e-9 {
                if on and current.len() >= 4 { pieces.push(current) }
                if on { current = [] }
                on = not on
                index = (index + 1) % dash.len()
                remaining = dash[index]
            }
        }
    }
    if current.len() >= 4 { pieces.push(current) }
    return pieces
}

#: Add the outline of one stroked run into `outline`.
#:
#: The run becomes a single closed ring: out along one offset side and back
#: along the other. That is half the edges of a quad per segment, and one
#: subpath instead of hundreds, which is most of the cost of stroking a sampled
#: curve. Sharp corners still get a disc, because a mitre there would either
#: shoot off to infinity or leave a notch.
fn _outline_run(outline: Path, run: list[float], width: float, cap: Cap, join: Join) -> void {
    let half = width * 0.5
    let count = run.len() // 2
    if count < 2 { return }

    var extended = [value for value in run]
    if cap == Cap.Square {
        _extend_ends(extended, half)
    }

    let left = _offset_side(extended, half)
    let right = _offset_side(_reversed_run(extended), half)
    if left.len() < 2 or right.len() < 2 { return }

    var ring = [point for point in left]
    for point in right { ring.push(point) }
    _add_ring(outline, ring)

    if join != Join.Bevel and count > 2 {
        for vertex in 1..count - 1 {
            # A join only has a wedge to fill when the path actually turns. A
            # sampled curve turns by a fraction of a degree between samples, so
            # this is rare and the ring covers the rest.
            if _turns_visibly(extended, vertex, half) {
                _add_disc(outline, extended[vertex * 2], extended[vertex * 2 + 1], half)
            }
        }
    }
    if cap == Cap.Round {
        _add_disc(outline, extended[0], extended[1], half)
        _add_disc(outline, extended[(count - 1) * 2], extended[(count - 1) * 2 + 1], half)
    }
}

#: One side of a stroke, offset by `half` along each vertex's bisector.
fn _offset_side(run: list[float], half: float) -> list[tuple[float, float]] {
    let count = run.len() // 2
    var out = []
    for index in 0..count {
        let (nx, ny) = _vertex_normal(run, index, count)
        if nx == 0.0 and ny == 0.0 { continue }
        out.push((run[index * 2] + nx * half, run[index * 2 + 1] + ny * half))
    }
    return out
}

#: The outward normal at a vertex: the segment's own at an end, the bisector of
#: its two segments in the middle.
#:
#: The bisector is scaled so the offset side stays `half` from both segments,
#: and clamped because that scale grows without bound as a corner tightens.
fn _vertex_normal(run: list[float], index: int, count: int) -> tuple[float, float] {
    var before_x = 0.0
    var before_y = 0.0
    var after_x = 0.0
    var after_y = 0.0
    if index > 0 {
        (before_x, before_y) = _unit_normal(run, index - 1, index)
    }
    if index < count - 1 {
        (after_x, after_y) = _unit_normal(run, index, index + 1)
    }
    if index == 0 { return (after_x, after_y) }
    if index == count - 1 { return (before_x, before_y) }

    var mx = before_x + after_x
    var my = before_y + after_y
    let length = math.hypot(mx, my)
    if length < 1e-9 { return (after_x, after_y) }
    mx = mx / length
    my = my / length
    let scale = min(4.0, 1.0 / max(1e-6, mx * after_x + my * after_y))
    return (mx * scale, my * scale)
}

#: The unit normal of the segment between two vertices.
fn _unit_normal(run: list[float], from_index: int, to_index: int) -> tuple[float, float] {
    let dx = run[to_index * 2] - run[from_index * 2]
    let dy = run[to_index * 2 + 1] - run[from_index * 2 + 1]
    let length = math.hypot(dx, dy)
    if length <= 0.0 { return (0.0, 0.0) }
    return (-dy / length, dx / length)
}

#: The same run walked backwards, so the second side offsets the other way.
fn _reversed_run(run: list[float]) -> list[float] {
    var out = []
    var index = run.len() // 2 - 1
    while index >= 0 {
        out.push(run[index * 2])
        out.push(run[index * 2 + 1])
        index -= 1
    }
    return out
}

#: Push the two ends outward by `half`, which is what a square cap is.
fn _extend_ends(run: list[float], half: float) -> void {
    let count = run.len() // 2
    if count < 2 { return }
    let (hx, hy) = _direction(run, 1, 0)
    run[0] = run[0] + hx * half
    run[1] = run[1] + hy * half
    let (tx, ty) = _direction(run, count - 2, count - 1)
    run[(count - 1) * 2] = run[(count - 1) * 2] + tx * half
    run[(count - 1) * 2 + 1] = run[(count - 1) * 2 + 1] + ty * half
}

#: The unit direction from one vertex to another.
fn _direction(run: list[float], from_index: int, to_index: int) -> tuple[float, float] {
    let dx = run[to_index * 2] - run[from_index * 2]
    let dy = run[to_index * 2 + 1] - run[from_index * 2 + 1]
    let length = math.hypot(dx, dy)
    if length <= 0.0 { return (0.0, 0.0) }
    return (dx / length, dy / length)
}

#: Report whether the corner at `vertex` leaves a gap worth filling.
#:
#: The notch a bevelled join leaves is about `half * (1 - cos(turn / 2))` deep.
#: Below a quarter of a pixel nothing is visible, so the disc is skipped.
fn _turns_visibly(run: list[float], vertex: int, half: float) -> bool {
    let ax = run[vertex * 2] - run[(vertex - 1) * 2]
    let ay = run[vertex * 2 + 1] - run[(vertex - 1) * 2 + 1]
    let bx = run[(vertex + 1) * 2] - run[vertex * 2]
    let by = run[(vertex + 1) * 2 + 1] - run[vertex * 2 + 1]
    let first = math.hypot(ax, ay)
    let second = math.hypot(bx, by)
    if first <= 0.0 or second <= 0.0 { return false }

    let cosine = (ax * bx + ay * by) / (first * second)
    let clamped = max(-1.0, min(1.0, cosine))
    # cos(turn/2) from cos(turn), avoiding an arccos and a second cosine.
    let half_cosine = math.sqrt(max(0.0, (1.0 + clamped) * 0.5))
    return half * (1.0 - half_cosine) > 0.25
}

#: Append a filled disc, approximated by a polygon fine enough for its size.
fn _add_disc(outline: Path, x: float, y: float, radius: float) -> void {
    if radius <= 0.0 { return }
    let steps = max(6, min(24, int(radius * 4.0)))
    var points = []
    for index in 0..steps {
        let angle = math.TAU * index.to_float() / steps.to_float()
        points.push((x + radius * math.cos(angle), y + radius * math.sin(angle)))
    }
    _add_ring(outline, points)
}

#: Append a ring, turned so every piece of a stroke winds the same way.
#:
#: Non-zero winding cancels where two rings of opposite orientation overlap. A
#: stroke's quads take their orientation from the direction of their segment, so
#: a curve that doubles back produces opposing quads and the overlap punches a
#: hole in the line. Normalising the orientation here is what keeps a stroked
#: path solid.
fn _add_ring(outline: Path, points: list[tuple[float, float]]) -> void {
    if points.len() < 3 { return }
    var twice_area = 0.0
    for index in 0..points.len() {
        let (x0, y0) = points[index]
        let (x1, y1) = points[(index + 1) % points.len()]
        twice_area += x0 * y1 - x1 * y0
    }
    if twice_area < 0.0 {
        var reversed = []
        var index = points.len() - 1
        while index >= 0 {
            reversed.push(points[index])
            index -= 1
        }
        let _ring = outline.polygon(reversed)
        return
    }
    let _ring = outline.polygon(points)
}
