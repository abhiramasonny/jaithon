#: Showing images on screen, and the keyboard, mouse, and sliders that go with
#: them.
#:
#: The names are OpenCV's, so the loop everyone writes reads the same here:
#: `imshow` a frame, `wait_key` a millisecond, break on a key. Windows are
#: opened on first use and named the way OpenCV names them, so a second
#: `imshow` with the same name replaces what is in that window rather than
#: opening another.
#:
#: This sits on `std.gui`, which draws into a software back buffer. A window
#: with trackbars keeps them in a strip above the image and handles the mouse
#: for them, so `create_trackbar` behaves the way its OpenCV counterpart does
#: without a widget toolkit underneath.
#:
#: ```
#: import jaicv as cv
#:
#: let camera = cv.VideoCapture(0)
#: while true {
#:     let (ok, frame) = camera.read()
#:     if not ok { break }
#:     cv.imshow("camera", frame)
#:     if cv.wait_key(1) == 27 { break }
#: }
#: cv.destroy_all_windows()
#: ```
import std.gui as gui
from std.time import Instant
from std.gui import Event, Window
from std.gui.canvas import Canvas
from ..core.mat import Mat
from ..core.types import Rect

#: A window that cannot be resized, sized to the first image shown in it.
pub const WINDOW_AUTOSIZE = 1
#: A window the user can resize; the image is scaled to fit.
pub const WINDOW_NORMAL = 0
#: Keep the image's aspect ratio when the window is resized.
pub const WINDOW_KEEPRATIO = 0

#: The mouse moved.
pub const EVENT_MOUSEMOVE = 0
#: The left button went down.
pub const EVENT_LBUTTONDOWN = 1
#: The right button went down.
pub const EVENT_RBUTTONDOWN = 2
#: The middle button went down.
pub const EVENT_MBUTTONDOWN = 3
#: The left button came up.
pub const EVENT_LBUTTONUP = 4
#: The right button came up.
pub const EVENT_RBUTTONUP = 5
#: The middle button came up.
pub const EVENT_MBUTTONUP = 6
#: The wheel turned.
pub const EVENT_MOUSEWHEEL = 10

#: How tall one trackbar's strip is.
pub const TRACKBAR_HEIGHT = 26

class Trackbar {
    pub let name: str
    pub var value: int
    pub let count: int
    pub var on_change: fn(int) -> void

    pub fn init(self, name: str, value: int, count: int, on_change: fn(int) -> void) {
        self.name = name
        self.value = value
        self.count = max(count, 1)
        self.on_change = on_change
    }
}

class Pane {
    pub let name: str
    pub var window: Window?
    pub var flags: int
    pub var width: int
    pub var height: int
    pub var trackbars: list[Trackbar]
    pub var on_mouse: fn(int, int, int, int) -> void
    pub var dragging: int
    pub var last_key: int
    pub var closed: bool

    pub fn init(self, name: str, flags: int) {
        self.name = name
        self.window = null
        self.flags = flags
        self.width = 0
        self.height = 0
        self.trackbars = []
        self.on_mouse = fn(_event: int, _x: int, _y: int, _flags: int) -> void {}
        self.dragging = -1
        self.last_key = -1
        self.closed = false
    }

    pub fn strip_height(self) -> int { return self.trackbars.len() * TRACKBAR_HEIGHT }
}

var _panes: dict[str, Pane] = {}
var _order: list[str] = []

fn pane_for(name: str) -> Pane {
    let existing = _panes.get(name, null)
    if existing is not null { return existing }
    let made = Pane(name, WINDOW_AUTOSIZE)
    _panes[name] = made
    _order.push(name)
    return made
}

#: Open a window, or note the flags a later `imshow` should use.
pub fn named_window(name: str, flags: int = WINDOW_AUTOSIZE) -> void {
    let pane = pane_for(name)
    pane.flags = flags
}

fn ensure_window(pane: Pane, width: int, height: int) -> Window? {
    if pane.closed { return null }
    if not gui.is_available() { return null }
    let wanted_height = height + pane.strip_height()
    let existing = pane.window
    if existing is not null {
        if pane.width == width and pane.height == wanted_height { return existing }
        #: The back buffer is fixed at the size the window was opened with, so
        #: a different image means a different window.
        existing.close()
    }
    let opened = Window(width, wanted_height, pane.name)
    pane.window = opened
    pane.width = width
    pane.height = wanted_height
    return opened
}

fn packed(image: Mat) -> list[int] {
    #: One `0xAARRGGBB` int per pixel, which is what the back buffer holds.
    let cn = image.channels()
    let values = image.values()
    var out: list[int] = []
    let count = image.rows * image.cols
    for index in 0..count {
        var b = 0
        var g = 0
        var r = 0
        var a = 255
        if cn == 1 {
            b = clamp_byte(values[index])
            g = b
            r = b
        } else {
            b = clamp_byte(values[index * cn])
            g = clamp_byte(values[index * cn + 1])
            r = clamp_byte(values[index * cn + 2])
            if cn >= 4 { a = clamp_byte(values[index * cn + 3]) }
        }
        out.push(a << 24 | r << 16 | g << 8 | b)
    }
    return out
}

fn clamp_byte(value: float) -> int {
    let rounded = int(value + 0.5)
    if rounded < 0 { return 0 }
    if rounded > 255 { return 255 }
    return rounded
}

fn draw_trackbars(pane: Pane, canvas: Canvas) -> void {
    for index in 0..pane.trackbars.len() {
        let bar = pane.trackbars[index]
        let top = index * TRACKBAR_HEIGHT
        canvas.fill_rect(0, top, pane.width, TRACKBAR_HEIGHT, 0xFF202020)
        let track_y = top + TRACKBAR_HEIGHT // 2
        canvas.fill_rect(8, track_y - 1, pane.width - 16, 3, 0xFF606060)
        let span = max(pane.width - 16 - 8, 1)
        let at = 8 + span * bar.value // bar.count
        canvas.fill_rect(at, track_y - 7, 8, 15, 0xFFD0D0D0)
        canvas.text(10, top + 3, f"{bar.name} {bar.value}", 0xFFFFFFFF, 1)
    }
}

#: Show an image in a window, opening it if it is not open yet.
pub fn imshow(name: str, image: Mat) -> void {
    let pane = pane_for(name)
    image.flush()
    let window = ensure_window(pane, image.cols, image.rows)
    if window is null { return }
    let canvas = Canvas(window)
    let strip = pane.strip_height()
    canvas.blit_pixels(packed(image), image.cols, image.rows, 0, strip, image.cols, image.rows)
    if strip > 0 { draw_trackbars(pane, canvas) }
}

fn key_from(code: int, text: str) -> int {
    #: A composed character is what OpenCV reports, so it wins; a key with no
    #: character — an arrow, escape — falls back to its usage id offset out of
    #: the way of the printable range, which is what `wait_key_ex` returns.
    if text.len() > 0 { return ord(text[0]) }
    if code == 41 { return 27 }
    if code == 40 or code == 88 { return 13 }
    if code == 43 { return 9 }
    if code == 42 { return 8 }
    if code == 44 { return 32 }
    return 0x100000 + code
}

fn handle_events(pane: Pane, window: Window) -> void {
    let strip = pane.strip_height()
    for event in window.events() {
        match event {
            Event.Close => {
                pane.closed = true
                window.close()
                pane.window = null
            }
            Event.KeyDown(code, repeat) => {
                let _unused = repeat
                pane.last_key = key_from(code, "")
            }
            Event.Text(text) => { pane.last_key = key_from(0, text) }
            Event.MouseDown(button, x, y) => {
                if y < strip {
                    pane.dragging = y // TRACKBAR_HEIGHT
                    move_trackbar(pane, x)
                } else {
                    pane.on_mouse(mouse_down_event(button), x, y - strip, 0)
                }
            }
            Event.MouseUp(button, x, y) => {
                pane.dragging = -1
                if y >= strip { pane.on_mouse(mouse_up_event(button), x, y - strip, 0) }
            }
            Event.MouseMove(x, y) => {
                if pane.dragging >= 0 {
                    move_trackbar(pane, x)
                } elif y >= strip {
                    pane.on_mouse(EVENT_MOUSEMOVE, x, y - strip, 0)
                }
            }
            Event.Wheel(dx, dy) => {
                let _horizontal = dx
                pane.on_mouse(EVENT_MOUSEWHEEL, 0, 0, int(dy))
            }
            _ => { }
        }
    }
}

fn mouse_down_event(button: int) -> int {
    if button == 1 { return EVENT_MBUTTONDOWN }
    if button == 2 { return EVENT_RBUTTONDOWN }
    return EVENT_LBUTTONDOWN
}

fn mouse_up_event(button: int) -> int {
    if button == 1 { return EVENT_MBUTTONUP }
    if button == 2 { return EVENT_RBUTTONUP }
    return EVENT_LBUTTONUP
}

fn move_trackbar(pane: Pane, x: int) -> void {
    if pane.dragging < 0 or pane.dragging >= pane.trackbars.len() { return }
    let bar = pane.trackbars[pane.dragging]
    let span = max(pane.width - 24, 1)
    var value = (x - 8) * bar.count // span
    if value < 0 { value = 0 }
    if value > bar.count { value = bar.count }
    if value == bar.value { return }
    bar.value = value
    bar.on_change(value)
}

#: Draw every open window and wait for a key.
#:
#: `delay` is in milliseconds; zero waits until a key arrives. Returns the key,
#: or `-1` if the wait ran out. A window closed by the user counts as a key of
#: `-1` and leaves that window shut.
pub fn wait_key(delay: int = 0) -> int {
    if not gui.is_available() { return -1 }
    let started = Instant.now()
    while true {
        var any_open = false
        for name in _order {
            let pane = _panes.get(name, null)
            if pane is null { continue }
            let window = pane.window
            if window is null { continue }
            if not window.poll() {
                pane.window = null
                pane.closed = true
                continue
            }
            any_open = true
            handle_events(pane, window)
            window.present()
            if pane.last_key >= 0 {
                let key = pane.last_key
                pane.last_key = -1
                return key
            }
        }
        if not any_open { return -1 }
        if delay > 0 and Instant.now().duration_since(started).to_millis() >= delay { return -1 }
    }
    return -1
}

#: The same as `wait_key`, kept because OpenCV has both.
pub fn wait_key_ex(delay: int = 0) -> int { return wait_key(delay) }

#: Close one window.
pub fn destroy_window(name: str) -> void {
    let pane = _panes.get(name, null)
    if pane is null { return }
    let window = pane.window
    if window is not null { window.close() }
    pane.window = null
    pane.closed = true
    _panes.remove(name)
    var kept: list[str] = []
    for other in _order {
        if other != name { kept.push(other) }
    }
    _order = kept
}

#: Close every window.
pub fn destroy_all_windows() -> void {
    for name in [entry for entry in _order] { destroy_window(name) }
}

#: Rename a window's title bar without touching what is in it.
pub fn set_window_title(name: str, title: str) -> void {
    let pane = _panes.get(name, null)
    if pane is null { return }
    let window = pane.window
    if window is not null { window.set_title(title) }
}

#: Where the image sits inside its window.
pub fn get_window_image_rect(name: str) -> Rect {
    let pane = _panes.get(name, null)
    if pane is null { return Rect(0, 0, 0, 0) }
    let strip = pane.strip_height()
    return Rect(0, strip, pane.width, pane.height - strip)
}

#: Ask to be told when the mouse does something over a window.
#:
#: The callback is given the event, the position inside the image, and the
#: modifier flags, which is OpenCV's order.
pub fn set_mouse_callback(name: str, callback: fn(int, int, int, int) -> void) -> void {
    let pane = pane_for(name)
    pane.on_mouse = callback
}

#: Add a slider above a window's image.
pub fn create_trackbar(
    name: str,
    window_name: str,
    value: int,
    count: int,
    on_change: fn(int) -> void = fn(_value: int) -> void {}
) -> void {
    let pane = pane_for(window_name)
    for bar in pane.trackbars {
        if bar.name == name {
            bar.value = value
            return
        }
    }
    pane.trackbars.push(Trackbar(name, value, count, on_change))
    #: The strip grew, so the window has to be reopened at the new height.
    let window = pane.window
    if window is not null {
        window.close()
        pane.window = null
        pane.width = 0
        pane.height = 0
    }
}

#: Where a slider currently sits.
pub fn get_trackbar_pos(name: str, window_name: str) -> int {
    let pane = _panes.get(window_name, null)
    if pane is null { return 0 }
    for bar in pane.trackbars {
        if bar.name == name { return bar.value }
    }
    return 0
}

#: Move a slider from the program rather than the mouse.
pub fn set_trackbar_pos(name: str, window_name: str, value: int) -> void {
    let pane = _panes.get(window_name, null)
    if pane is null { return }
    for bar in pane.trackbars {
        if bar.name != name { continue }
        var wanted = value
        if wanted < 0 { wanted = 0 }
        if wanted > bar.count { wanted = bar.count }
        if wanted == bar.value { return }
        bar.value = wanted
        bar.on_change(wanted)
        return
    }
}

#: Whether a window with this name is open.
pub fn window_is_open(name: str) -> bool {
    let pane = _panes.get(name, null)
    if pane is null { return false }
    return pane.window is not null
}
