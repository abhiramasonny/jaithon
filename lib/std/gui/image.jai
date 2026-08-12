#: An offscreen pixel buffer, split out of `std.gui.canvas` because it shares
#: no state with `Canvas`: `Canvas.init` copies a target's buffer and size
#: once and never touches the `PixelTarget` again, so `Image` and `Canvas` are
#: only ever connected through that trait, never through each other's fields.
from std.gui import PixelTarget

#: An offscreen pixel buffer: a sprite, a tile sheet, or a canvas for a test.
#:
#: ```
#: from std.gui.color import rgb
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
