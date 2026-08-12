#: Windows, frame timing, and the raw event stream.
#:
#: A `Window` is a pixel buffer with a title bar. It owns no drawing code at
#: all: `pixels` hands out the back buffer as a live list of packed colours and
#: `std.gui.canvas` does every line, circle and glyph in Jaithon. The native
#: layer opens the window, pumps the OS event queue, and uploads the buffer.
#:
#: The frame loop is always the same four steps:
#:
#:     from std.gui import Window
#:     from std.gui.canvas import Canvas
#:     from std.gui.color import rgb
#:
#:     let window = Window(640, 480, "demo")
#:     defer { window.close() }
#:     let canvas = Canvas(window)
#:
#:     loop {
#:         if not window.poll() { break }        # 1. drain OS events
#:         canvas.clear(rgb(20, 20, 28))         # 2. draw the frame
#:         window.present()                      # 3. show it, 4. wait for 60 Hz
#:     }
#:
#: The back buffer keeps the size it was created with for the window's whole
#: life. Resizing the window scales the presented image instead of reallocating,
#: so a pointer into `pixels` — and every cached width in the drawing code —
#: stays valid. `Event.Resize` still reports the new on-screen size for a
#: program that wants to letterbox or re-lay out by hand.
#:
#: `Event` carries the platform's own key and button codes. `std.gui.input`
#: turns those into the named `Key` and `MouseButton` values that application
#: code should use; reach for the raw codes only when adding a key this library
#: does not name.
#:
#: ## Primitives used
#:
#: `gui_available`, `gui_window_open`, `gui_window_close`, `gui_poll`,
#: `gui_drain_events`, `gui_present`, `gui_pixels`, `gui_set_title`,
#: `time_mono_seconds` and `sleep`.

# Event tags, as the native layer writes them into each raw event record. They
# are matched with `==` rather than in a `match`, because a bare identifier in
# pattern position binds instead of comparing.
const _EVENT_CLOSE = 0
const _EVENT_KEY_DOWN = 1
const _EVENT_KEY_UP = 2
const _EVENT_TEXT = 3
const _EVENT_MOUSE_MOVE = 4
const _EVENT_MOUSE_DOWN = 5
const _EVENT_MOUSE_UP = 6
const _EVENT_WHEEL = 7
const _EVENT_RESIZE = 8
const _EVENT_FOCUS = 9

# How much of the new frame rate a smoothed reading takes on. Low enough that
# one slow frame does not make the counter jump, high enough to follow a real
# change within a few frames.
const _FPS_SMOOTHING: float = 0.1

#: A surface `std.gui.canvas` can draw into.
#:
#: A `Window` is one; so is an offscreen `Image`. Anything implementing it can
#: be handed to `Canvas`, which is what lets the same drawing code render to a
#: window, to a sprite, and to a test buffer with no display attached.
pub trait PixelTarget {
    #: Return the pixel buffer, `width() * height()` packed `0xAARRGGBB` ints.
    #:
    #: The list is live: assigning to an element changes that pixel. Row `y`
    #: starts at index `y * width()`.
    fn pixels(self) -> list[int]

    #: Width of the buffer in pixels.
    fn width(self) -> int

    #: Height of the buffer in pixels.
    fn height(self) -> int
}

#: Something that happened between two calls to `Window.poll`.
#:
#: Key codes are the platform's USB HID usage ids and button codes are 0 left,
#: 1 middle, 2 right — `std.gui.input` names both. `Text` carries the composed
#: characters an editor should insert, which is not derivable from key codes
#: once dead keys, IMEs and keyboard layouts are involved, so a text field must
#: use `Text` and never assemble characters out of `KeyDown`.
pub enum Event {
    #: The user asked to close the window.
    Close,
    #: A key went down. `repeat` is true for an auto-repeat, false for the
    #: first press, so a game reads presses and an editor reads both.
    KeyDown(code: int, repeat: bool),
    #: A key came back up.
    KeyUp(code: int),
    #: Text was composed and should be inserted at the caret.
    Text(text: str),
    #: The pointer moved to buffer position `(x, y)`.
    MouseMove(x: int, y: int),
    #: A mouse button went down at buffer position `(x, y)`.
    MouseDown(button: int, x: int, y: int),
    #: A mouse button came back up at buffer position `(x, y)`.
    MouseUp(button: int, x: int, y: int),
    #: The wheel or trackpad scrolled, in lines; positive `dy` scrolls up.
    Wheel(dx: float, dy: float),
    #: The window was resized on screen. The back buffer is unchanged.
    Resize(width: int, height: int),
    #: The window gained or lost keyboard focus.
    Focus(gained: bool),
}

#: An on-screen window with a software back buffer.
pub class Window: PixelTarget {
    let handle: int
    let buffer: list[int]
    let buffer_width: int
    let buffer_height: int
    let target_fps: int
    var frame_events: list[Event]
    var open: bool
    var frame_start: float
    var last_delta: float
    var smoothed_fps: float
    var frame_index: int

    #: Open a window with a `width` by `height` back buffer.
    #:
    #: `target_fps` is the rate `present` paces the loop to; 0 presents as fast
    #: as the program can draw. Raises `ValueError` on a non-positive size and
    #: `RuntimeError` when there is no display.
    pub fn init(self, width: int, height: int, title: str, target_fps: int = 60) {
        if width <= 0 or height <= 0 {
            throw ValueError(f"window size must be positive, got {width}x{height}")
        }
        if target_fps < 0 { throw ValueError(f"target_fps must be non-negative, got {target_fps}") }
        if not is_available() {
            throw RuntimeError("cannot open a window: no display is available")
        }
        self.handle = __prim__.gui_window_open(width, height, title)
        self.buffer = __prim__.gui_pixels(self.handle)
        self.buffer_width = width
        self.buffer_height = height
        self.target_fps = target_fps
        self.frame_events = []
        self.open = true
        self.frame_start = __prim__.time_mono_seconds()
        # Report the target frame time before the first `present` has measured
        # anything, so time-based animation does not jump on frame one.
        self.last_delta = if target_fps > 0 { 1.0 / target_fps } else { 0.0 }
        self.smoothed_fps = 0.0
        self.frame_index = 0
    }

    #: Return the back buffer, `width() * height()` packed `0xAARRGGBB` ints.
    #:
    #: The same live list every time: hold on to it and write pixels straight
    #: into it, which is exactly what `Canvas` does.
    pub fn pixels(self) -> list[int] { return self.buffer }

    #: Width of the back buffer in pixels.
    pub fn width(self) -> int { return self.buffer_width }

    #: Height of the back buffer in pixels.
    pub fn height(self) -> int { return self.buffer_height }

    #: Back buffer size as `(width, height)`.
    pub fn size(self) -> tuple[int, int] { return (self.buffer_width, self.buffer_height) }

    #: Drain the OS event queue and report whether the window is still open.
    #:
    #: Call it once at the top of each frame. It is what keeps the window
    #: responsive: a loop that draws without polling is a beachball. Once it
    #: has returned false it keeps returning false.
    pub fn poll(self) -> bool {
        if not self.open { return false }
        if not __prim__.gui_poll(self.handle) { self.open = false }
        let decoded: list[Event] = []
        for raw in __prim__.gui_drain_events(self.handle) {
            let event = _decode(raw)
            if event is null { continue }
            match event {
                Event.Close => { self.open = false }
                _ => { }
            }
            decoded.push(event)
        }
        self.frame_events = decoded
        return self.open
    }

    #: Events collected by the most recent `poll`, oldest first.
    #:
    #: The list is replaced by the next `poll`, so copy anything kept longer
    #: than the frame. `std.gui.input.Input` consumes exactly this list.
    pub fn events(self) -> list[Event] { return self.frame_events }

    #: Show the back buffer and wait out the rest of the frame budget.
    #:
    #: With a `target_fps` this is also the frame limiter: it sleeps for
    #: whatever is left of the budget after drawing, which is what keeps a
    #: simple loop from spinning a core at 4000 fps. Presenting a closed window
    #: does nothing.
    pub fn present(self) -> void {
        if not self.open { return }
        __prim__.gui_present(self.handle)
        if self.target_fps > 0 {
            let budget = 1.0 / self.target_fps
            let spent = __prim__.time_mono_seconds() - self.frame_start
            if spent < budget { __prim__.sleep(budget - spent) }
        }
        let ended = __prim__.time_mono_seconds()
        self.last_delta = ended - self.frame_start
        self.frame_start = ended
        self.frame_index += 1
        if self.last_delta > 0.0 {
            let instant = 1.0 / self.last_delta
            if self.frame_index == 1 {
                self.smoothed_fps = instant
            } else {
                self.smoothed_fps += (instant - self.smoothed_fps) * _FPS_SMOOTHING
            }
        }
    }

    #: Seconds the last full frame took, sleep included.
    #:
    #: Multiply every speed by it and the program runs at the same rate whether
    #: the frame limiter is on or the machine is struggling.
    pub fn delta_time(self) -> float { return self.last_delta }

    #: Frames per second, smoothed over roughly the last ten frames.
    #:
    #: Zero until the first `present` has completed.
    pub fn fps(self) -> float { return self.smoothed_fps }

    #: Number of frames presented so far.
    pub fn frame(self) -> int { return self.frame_index }

    #: Replace the text in the title bar.
    pub fn set_title(self, title: str) -> void {
        if not self.open { return }
        __prim__.gui_set_title(self.handle, title)
    }

    #: Report whether the window is still open.
    pub fn is_open(self) -> bool { return self.open }

    #: Close the window and release the native resources.
    #:
    #: Closing twice is harmless, so it belongs in a `defer` on the line after
    #: the window is opened.
    pub fn close(self) -> void {
        if not self.open { return }
        self.open = false
        __prim__.gui_window_close(self.handle)
    }

    fn __str__(self) -> str {
        return f"Window({self.buffer_width}x{self.buffer_height})"
    }
}

#: Report whether a display this process can open a window on exists.
#:
#: False over a bare ssh session or in a test runner with no window server, in
#: which case `Window` raises rather than hanging.
pub fn is_available() -> bool { return __prim__.gui_available() }

# One raw event record from the native layer: a tag followed by that tag's
# payload. An unrecognised tag yields null and is dropped, so a native layer
# that grows a new event kind does not break an older library.
fn _decode(raw: list[any]) -> Event? {
    let tag = raw[0]
    if tag == _EVENT_CLOSE { return Event.Close }
    if tag == _EVENT_KEY_DOWN { return Event.KeyDown(raw[1], raw[2]) }
    if tag == _EVENT_KEY_UP { return Event.KeyUp(raw[1]) }
    if tag == _EVENT_TEXT { return Event.Text(raw[1]) }
    if tag == _EVENT_MOUSE_MOVE { return Event.MouseMove(raw[1], raw[2]) }
    if tag == _EVENT_MOUSE_DOWN { return Event.MouseDown(raw[1], raw[2], raw[3]) }
    if tag == _EVENT_MOUSE_UP { return Event.MouseUp(raw[1], raw[2], raw[3]) }
    if tag == _EVENT_WHEEL { return Event.Wheel(raw[1], raw[2]) }
    if tag == _EVENT_RESIZE { return Event.Resize(raw[1], raw[2]) }
    if tag == _EVENT_FOCUS { return Event.Focus(raw[1]) }
    return null
}
