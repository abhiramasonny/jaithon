#: The interactive backend: a figure in a window you can pan and zoom.
#:
#: The figure is re-rendered whenever the view changes rather than the pixels
#: being transformed, so a zoomed axis re-ticks and re-labels itself instead of
#: showing magnified text. That costs a render per interaction and is what makes
#: the picture stay correct at every scale.
#:
#: Dragging with the left button pans. The wheel zooms about the pointer, so
#: whatever is under the cursor stays under it. `r` resets, `s` saves a PNG
#: beside the program, `escape` closes.
#:
#:     from jaiplot import Figure
#:     from jaiplot.backend.window import show
#:
#:     let figure = Figure(width: 900, height: 600)
#:     figure.axes().line(xs, ys)
#:     show(figure)
import std.io
import std.math
from std.gui import Window
from std.gui.canvas import Canvas, Image, rgb
from std.gui.input import Input, Key, MouseButton

from ..axes import Axes
from ..figure import Figure
from ..frame import Frame
from .png import encode
from .raster import draw

#: How much one wheel notch scales the view.
#:
#: Below about a tenth per notch zooming feels unresponsive, and much above it
#: overshoots what the reader was aiming at.
pub const ZOOM_RATE: float = 0.12

#: Height of the readout strip along the bottom, in device pixels.
pub const READOUT_HEIGHT = 16

#: Open a window on `figure` and run until it is closed.
#:
#: Raises `RuntimeError` when no display is available, rather than appearing to
#: succeed and drawing into a buffer nobody sees.
pub fn show(figure: Figure, title: str = "jaiplot") -> void {
    let panels = figure.all_axes()
    if panels.len() == 0 {
        throw ValueError("the figure has no axes to show")
    }

    let window = Window(figure.width, figure.height, title)
    defer { window.close() }
    let canvas = Canvas(window)
    let input = Input(window)

    # The view is kept as limit overrides on the first set of axes. Panning and
    # zooming write here and the figure is rendered again, which is what makes a
    # zoomed axis re-tick rather than magnify.
    let axes = panels[0]
    let (home_x_low, home_x_high, home_y_low, home_y_high) = axes.limits()
    var view = _View(home_x_low, home_x_high, home_y_low, home_y_high)
    var dragging = false
    var last_x = 0
    var last_y = 0
    var dirty = true
    var readout = ""

    loop {
        if not window.poll() { break }
        input.update()

        if input.just_pressed(Key.Escape) { break }
        if input.just_pressed(Key.R) {
            view = _View(home_x_low, home_x_high, home_y_low, home_y_high)
            dirty = true
        }
        if input.just_pressed(Key.S) {
            let _written = _snapshot(figure, axes, view)
            readout = "saved jaiplot.png"
            dirty = true
        }

        let _applied = axes.set_xlim(view.x_low, view.x_high)
        let _also = axes.set_ylim(view.y_low, view.y_high)
        let plot = axes.plot_area(figure.device())
        let frame = axes.frame(figure.device())

        if input.mouse.just_pressed(MouseButton.Left) {
            dragging = true
            last_x = input.mouse.x()
            last_y = input.mouse.y()
        }
        if input.mouse.just_released(MouseButton.Left) { dragging = false }

        if dragging {
            let dx = input.mouse.x() - last_x
            let dy = input.mouse.y() - last_y
            if dx != 0 or dy != 0 {
                view = _pan(view, frame, dx, dy)
                last_x = input.mouse.x()
                last_y = input.mouse.y()
                dirty = true
            }
        }

        let (_wheel_x, wheel_y) = input.mouse.wheel()
        if wheel_y != 0.0 {
            view = _zoom(
                view,
                frame,
                input.mouse.x().to_float(),
                input.mouse.y().to_float(),
                wheel_y
            )
            dirty = true
        }

        if input.mouse.in_rect(
            int(plot.left()),
            int(plot.bottom()),
            int(plot.width()),
            int(plot.height())
        ) {
            let (x, y) = frame.locate(input.mouse.x().to_float(), input.mouse.y().to_float())
            let fresh = f"x={x:.4g}  y={y:.4g}"
            if fresh != readout {
                readout = fresh
                dirty = true
            }
        }

        if dirty {
            canvas.clear(rgb(255, 255, 255))
            draw(figure.render(), canvas)
            _draw_readout(canvas, figure, readout)
            dirty = false
        }
        window.present()
    }
}

#: The limits currently in view.
class _View {
    pub var x_low: float
    pub var x_high: float
    pub var y_low: float
    pub var y_high: float

    pub fn init(self, x_low: float, x_high: float, y_low: float, y_high: float) {
        self.x_low = x_low
        self.x_high = x_high
        self.y_low = y_low
        self.y_high = y_high
    }
}

#: Shift the view by a drag in device pixels.
#:
#: The shift is computed by inverting two device points rather than scaling the
#: pixel delta, so it stays correct on a logarithmic axis where a fixed number
#: of pixels means a different amount of data at each end.
fn _pan(view: _View, frame: Frame, dx: int, dy: int) -> _View {
    let (x0, y0) = frame.locate(0.0, 0.0)
    let (x1, y1) = frame.locate(dx.to_float(), dy.to_float())
    let shift_x = x0 - x1
    let shift_y = y0 - y1
    return _View(
        view.x_low + shift_x,
        view.x_high + shift_x,
        view.y_low + shift_y,
        view.y_high + shift_y
    )
}

#: Scale the view about the pointer, keeping what is under it in place.
fn _zoom(view: _View, frame: Frame, px: float, py: float, notches: float) -> _View {
    let factor = math.exp(-notches * ZOOM_RATE)
    let (anchor_x, anchor_y) = frame.locate(px, py)
    return _View(
        anchor_x + (view.x_low - anchor_x) * factor,
        anchor_x + (view.x_high - anchor_x) * factor,
        anchor_y + (view.y_low - anchor_y) * factor,
        anchor_y + (view.y_high - anchor_y) * factor
    )
}

fn _draw_readout(canvas: Canvas, figure: Figure, readout: str) -> void {
    if readout.len() == 0 { return }
    canvas.text(6, figure.height - READOUT_HEIGHT + 4, readout, rgb(90, 90, 100))
}

#: Write the current view out beside the program.
fn _snapshot(figure: Figure, axes: Axes, view: _View) -> bool {
    let _applied = axes.set_xlim(view.x_low, view.x_high)
    let _also = axes.set_ylim(view.y_low, view.y_high)
    let image = _render_offscreen(figure)
    io.write_bytes("jaiplot.png", encode(image))
    return true
}

fn _render_offscreen(figure: Figure) -> Image {
    let image = Image(figure.width, figure.height, rgb(255, 255, 255))
    draw(figure.render(), Canvas(image))
    return image
}

#: Run `figure` as an animation, calling `update` before each frame.
#:
#: `update` receives the frame number and mutates the series through their
#: `set_data`. Returning false ends the animation. The figure is re-rendered
#: every frame, so an axis that ranges itself follows the data; call `set_xlim`
#: and `set_ylim` first to hold the view still, which is usually what an
#: animation wants.
#:
#: The window paces itself, so `frames_per_second` is an upper bound rather
#: than a promise: a figure that takes longer than its budget to draw simply
#: runs slower rather than falling behind.
#:
#:     let figure = Figure(width: 640, height: 480)
#:     let axes = figure.axes()
#:     let _limits = axes.set_ylim(-1.5, 1.5)
#:     let wave = axes.line(x, ys)
#:     animate(figure, fn(frame: int) -> bool {
#:         wave.set_data(x, shifted(frame))
#:         return frame < 600
#:     })
pub fn animate(
    figure: Figure,
    update: fn(int) -> bool,
    title: str = "jaiplot",
    frames_per_second: int = 60
) -> void {
    if frames_per_second <= 0 {
        throw ValueError(f"a frame rate must be positive, got {frames_per_second}")
    }
    let window = Window(figure.width, figure.height, title, target_fps: frames_per_second)
    defer { window.close() }
    let canvas = Canvas(window)
    let input = Input(window)

    # A frozen figure draws its background once into an image and blits it
    # each frame. For a filter running over a fixed map that is the map, the
    # grid and every label -- most of the frame, redrawn for nothing.
    let cached = _background(figure)

    var frame = 0
    loop {
        if not window.poll() { break }
        input.update()
        if input.just_pressed(Key.Escape) { break }
        if not update(frame) { break }

        _paint(figure, canvas, cached)
        window.present()
        frame += 1
    }
}

#: The cached background of a frozen figure, or null when it has none.
fn _background(figure: Figure) -> Image? {
    if not figure.is_frozen() { return null }
    let image = Image(figure.width, figure.height, rgb(255, 255, 255))
    let (static_list, _dynamic) = figure.render_layers()
    draw(static_list, Canvas(image))
    return image
}

#: Draw one frame, blitting the cached background when there is one.
fn _paint(figure: Figure, canvas: Canvas, cached: Image?) -> void {
    if cached == null {
        canvas.clear(rgb(255, 255, 255))
        draw(figure.render(), canvas)
        return
    }
    canvas.blit_pixels(
        cached.pixels(),
        cached.width(),
        cached.height(),
        0,
        0,
        cached.width(),
        cached.height()
    )
    draw(figure.render_dynamic(), canvas)
}

#: Render an animation to a numbered sequence of PNG files.
#:
#: The headless counterpart of `animate`, for building a video or a GIF with
#: whatever tool is to hand. Files are written as `<prefix>0000.png` and upward,
#: zero-padded so they sort in order.
#:
#: Returns how many frames were written.
pub fn save_frames(
    figure: Figure,
    update: fn(int) -> bool,
    frames: int,
    prefix: str = "frame"
) -> int {
    if frames <= 0 {
        throw ValueError(f"a frame count must be positive, got {frames}")
    }
    let image = Image(figure.width, figure.height, rgb(255, 255, 255))
    let canvas = Canvas(image)
    let cached = _background(figure)

    var written = 0
    for frame in 0..frames {
        if not update(frame) { break }
        _paint(figure, canvas, cached)
        io.write_bytes(f"{prefix}{_padded(frame)}.png", encode(image))
        written += 1
    }
    return written
}

#: A frame number padded to four digits so the files sort in order.
fn _padded(frame: int) -> str {
    var text = str(frame)
    while text.len() < 4 { text = "0" + text }
    return text
}
