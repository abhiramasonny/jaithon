#: An immediate-mode user interface drawn on a `Canvas`.
#:
#: There is no widget tree and no retained state to keep in step with the
#: program: a widget is a function call that draws the control and returns what
#: the user did to it. A button call returns whether it was clicked this frame;
#: a slider call takes the current value and returns the new one. The only
#: state `UI` keeps between frames is which widget the pointer is over, which
#: one is being dragged, and which text field has focus — three strings.
#:
#: Every widget takes an `id` that must be unique among the widgets alive in
#: the same frame. Two controls sharing an id drag together.
#:
#: A complete tool, at 60 frames per second:
#:
#:     from std.gui import Window
#:     from std.gui.canvas import Canvas
#:     from std.gui.input import Input
#:     from std.gui.widgets import UI, Theme
#:
#:     fn main() -> int {
#:         let window = Window(420, 260, "mixer", target_fps: 60)
#:         defer { window.close() }
#:         let ui = UI(Canvas(window), Input(window))
#:
#:         var gain = 0.5
#:         var muted = false
#:         var name = "track 1"
#:
#:         loop {
#:             if not window.poll() { break }
#:             ui.begin()
#:             ui.canvas.clear(ui.theme.background)
#:
#:             ui.panel(10, 10, 400, 240, "mixer")
#:             name = ui.text_field("name", 24, 50, 250, name)
#:             gain = ui.slider("gain", 24, 90, 250, gain, 0.0, 1.0)
#:             ui.label(288, 92, f"{gain:.2f}")
#:             muted = ui.checkbox("mute", 24, 130, "mute", muted)
#:             if ui.button("reset", 24, 170, 90, 26, "reset") {
#:                 gain = 0.5
#:                 muted = false
#:             }
#:             ui.label(24, 216, f"{window.fps():.0f} fps")
#:
#:             ui.end()
#:             window.present()
#:         }
#:         return 0
#:     }
#:
#: `begin` must be called before the first widget of a frame and `end` after
#: the last: `begin` clears the hover state the widgets are about to set, and
#: `end` is what notices that the mouse button was let go somewhere else.
from std.gui.canvas import Canvas
from std.gui.color import rgb
from std.gui.font import GLYPH_HEIGHT, GLYPH_SPACING, GLYPH_WIDTH
from std.gui.input import Input, Key, MouseButton

#: Height of the widgets that size themselves: slider, checkbox, text field.
pub const WIDGET_HEIGHT = 22

#: Side of a checkbox's box, in pixels.
pub const CHECKBOX_SIZE = 16

# Horizontal advance of one character of the built-in font.
let _ADVANCE = GLYPH_WIDTH + GLYPH_SPACING

# Text inset from the left edge of a field or a button.
const _PADDING = 6

# Frames the caret spends visible, and again hidden. At 60 fps this is a blink
# a little under twice a second, which is the usual rate.
const _CARET_PERIOD = 30

#: The eight colours every widget draws itself from.
pub class Theme {
    #: Fill behind a whole window or panel.
    pub let background: int

    #: Face of a control at rest.
    pub let surface: int

    #: Face of a control under the pointer.
    pub let surface_hover: int

    #: Face of a control being pressed or dragged.
    pub let surface_active: int

    #: One-pixel outline around a control.
    pub let border: int

    #: Primary text.
    pub let text: int

    #: Secondary text: hints, units, disabled labels.
    pub let muted: int

    #: Filled slider track, focus ring, check mark.
    pub let accent: int

    #: Create a theme from its eight colours.
    pub fn init(
        self,
        background: int,
        surface: int,
        surface_hover: int,
        surface_active: int,
        border: int,
        text: int,
        muted: int,
        accent: int
    ) {
        self.background = background
        self.surface = surface
        self.surface_hover = surface_hover
        self.surface_active = surface_active
        self.border = border
        self.text = text
        self.muted = muted
        self.accent = accent
    }

    #: The default dark theme.
    pub static fn dark() -> Theme {
        return Theme(
            rgb(18, 18, 24),
            rgb(38, 38, 48),
            rgb(52, 52, 64),
            rgb(70, 70, 86),
            rgb(70, 70, 86),
            rgb(232, 232, 240),
            rgb(150, 150, 165),
            rgb(90, 160, 240)
        )
    }

    #: A light theme with the same contrast relationships.
    pub static fn light() -> Theme {
        return Theme(
            rgb(245, 245, 248),
            rgb(226, 226, 232),
            rgb(214, 214, 222),
            rgb(198, 198, 208),
            rgb(180, 180, 192),
            rgb(24, 24, 32),
            rgb(105, 105, 120),
            rgb(40, 110, 220)
        )
    }
}

#: One frame's worth of user interface over a canvas and its input.
pub class UI {
    #: The canvas the widgets draw on. Draw your own content on it too.
    pub let canvas: Canvas

    #: The input the widgets read. Ask it about keys the widgets do not use.
    pub let input: Input

    #: The colours in use.
    pub let theme: Theme

    var hot: str
    var active: str
    var focused: str
    var caret: int
    var frames: int

    #: Create a UI over `canvas`, driven by `input`, in `theme` or the dark one.
    pub fn init(self, canvas: Canvas, input: Input, theme: Theme? = null) {
        self.canvas = canvas
        self.input = input
        self.theme = theme ?? Theme.dark()
        self.hot = ""
        self.active = ""
        self.focused = ""
        self.caret = 0
        self.frames = 0
    }

    #: Start a frame. Call it before the first widget.
    pub fn begin(self) -> void {
        self.hot = ""
        self.frames += 1
    }

    #: Finish a frame. Call it after the last widget.
    #:
    #: Releasing the button anywhere ends a drag, which is why this cannot be
    #: left to the widget: the pointer is usually somewhere else by then.
    #: Pressing it over no widget at all drops text-field focus.
    pub fn end(self) -> void {
        if not self.input.mouse.is_down(MouseButton.Left) { self.active = "" }
        if self.input.mouse.just_pressed(MouseButton.Left) and self.hot.len() == 0 {
            self.focused = ""
        }
    }

    #: Report whether a text field currently has keyboard focus.
    pub fn has_focus(self) -> bool { return self.focused.len() > 0 }

    #: Draw a line of text, in `color` or the theme's text colour.
    pub fn label(self, x: int, y: int, text: str, color: int? = null) -> void {
        self.canvas.text(x, y, text, color ?? self.theme.text)
    }

    #: Draw a button and report whether it was clicked this frame.
    #:
    #: A click is a press and a release both inside the button, so dragging off
    #: it cancels — the behaviour every desktop toolkit has.
    pub fn button(self, id: str, x: int, y: int, width: int, height: int, label: str) -> bool {
        let hovering = self._hover(id, x, y, width, height)
        var clicked = false
        if self.active == id {
            if self.input.mouse.just_released(MouseButton.Left) {
                clicked = hovering
                self.active = ""
            }
        } elif hovering and self.input.mouse.just_pressed(MouseButton.Left) {
            self.active = id
            self.focused = ""
        }
        var face = self.theme.surface
        if hovering { face = self.theme.surface_hover }
        if hovering and self.active == id { face = self.theme.surface_active }
        self.canvas.fill_rounded_rect(x, y, width, height, 4, face)
        self.canvas.rounded_rect(x, y, width, height, 4, self.theme.border)
        let text_x = x + (width - self.canvas.text_width(label)) // 2
        self.canvas.text(text_x, y + (height - GLYPH_HEIGHT) // 2, label, self.theme.text)
        return clicked
    }

    #: Draw a checkbox with `label` to its right and return its new state.
    #:
    #: Clicking anywhere on the box or the label toggles it.
    pub fn checkbox(self, id: str, x: int, y: int, label: str, value: bool) -> bool {
        let width = CHECKBOX_SIZE + _PADDING + self.canvas.text_width(label)
        let hovering = self._hover(id, x, y, width, WIDGET_HEIGHT)
        var result = value
        if hovering and self.input.mouse.just_pressed(MouseButton.Left) {
            self.active = id
            self.focused = ""
        }
        if self.active == id and self.input.mouse.just_released(MouseButton.Left) {
            if hovering { result = not value }
            self.active = ""
        }
        let box_y = y + (WIDGET_HEIGHT - CHECKBOX_SIZE) // 2
        let face = if hovering { self.theme.surface_hover } else { self.theme.surface }
        self.canvas.fill_rounded_rect(x, box_y, CHECKBOX_SIZE, CHECKBOX_SIZE, 3, face)
        self.canvas.rounded_rect(x, box_y, CHECKBOX_SIZE, CHECKBOX_SIZE, 3, self.theme.border)
        if result {
            let left = x + 4
            let middle = x + 7
            let right = x + 12
            self.canvas.thick_line(left, box_y + 8, middle, box_y + 11, 2, self.theme.accent)
            self.canvas.thick_line(middle, box_y + 11, right, box_y + 5, 2, self.theme.accent)
        }
        let text_x = x + CHECKBOX_SIZE + _PADDING
        self.canvas.text(text_x, y + (WIDGET_HEIGHT - GLYPH_HEIGHT) // 2, label, self.theme.text)
        return result
    }

    #: Draw a horizontal slider and return its value after this frame's drag.
    #:
    #: The value is clamped into `[minimum, maximum]` on the way in, so a
    #: caller can hand back whatever it stored without checking it first.
    #: Raises `ValueError` when the range is empty.
    pub fn slider(
        self,
        id: str,
        x: int,
        y: int,
        width: int,
        value: float,
        minimum: float,
        maximum: float
    ) -> float {
        if maximum <= minimum {
            throw ValueError(f"slider range must be non-empty, got {minimum}..{maximum}")
        }
        let hovering = self._hover(id, x, y, width, WIDGET_HEIGHT)
        if hovering and self.input.mouse.just_pressed(MouseButton.Left) {
            self.active = id
            self.focused = ""
        }
        var result = min(max(value, minimum), maximum)
        let travel = max(width - CHECKBOX_SIZE, 1)
        if self.active == id {
            let offset = self.input.mouse.x() - (x + CHECKBOX_SIZE // 2)
            let dragged = min(max(float(offset) / float(travel), 0.0), 1.0)
            result = minimum + (maximum - minimum) * dragged
        }
        let fraction = (result - minimum) / (maximum - minimum)
        let knob_x = x + CHECKBOX_SIZE // 2 + int(travel * fraction)
        let track_y = y + WIDGET_HEIGHT // 2 - 2
        self.canvas.fill_rounded_rect(x, track_y, width, 4, 2, self.theme.surface)
        self.canvas.fill_rounded_rect(x, track_y, knob_x - x, 4, 2, self.theme.accent)
        var face = self.theme.surface
        if hovering or self.active == id { face = self.theme.surface_hover }
        self.canvas.fill_circle(knob_x, y + WIDGET_HEIGHT // 2, CHECKBOX_SIZE // 2, face)
        self.canvas.circle(knob_x, y + WIDGET_HEIGHT // 2, CHECKBOX_SIZE // 2, self.theme.border)
        return result
    }

    #: Draw an editable single-line text field and return its new contents.
    #:
    #: Clicking gives it focus and puts the caret where the click landed;
    #: clicking anywhere else takes focus away. Typing inserts what the
    #: platform composed — the user's real layout, dead keys and all — and
    #: Backspace, Delete, Left, Right, Home and End move and edit, with
    #: auto-repeat while held. Text longer than the field scrolls to keep the
    #: caret in view.
    pub fn text_field(self, id: str, x: int, y: int, width: int, value: str) -> str {
        let hovering = self._hover(id, x, y, width, WIDGET_HEIGHT)
        let inner_x = x + _PADDING
        let inner_width = width - 2 * _PADDING
        if hovering and self.input.mouse.just_pressed(MouseButton.Left) {
            self.focused = id
            let column = (self.input.mouse.x() - inner_x + _ADVANCE // 2) // _ADVANCE
            self.caret = min(max(column, 0), value.len())
        }
        var result = value
        if self.focused == id { result = self._edit(value) }
        let focused = self.focused == id
        let face = if hovering or focused { self.theme.surface_hover } else { self.theme.surface }
        self.canvas.fill_rounded_rect(x, y, width, WIDGET_HEIGHT, 4, face)
        let outline = if focused { self.theme.accent } else { self.theme.border }
        self.canvas.rounded_rect(x, y, width, WIDGET_HEIGHT, 4, outline)
        let caret_offset = self.caret * _ADVANCE
        let scroll = if focused { max(caret_offset - inner_width, 0) } else { 0 }
        let text_y = y + (WIDGET_HEIGHT - GLYPH_HEIGHT) // 2
        let (clip_x, clip_y, clip_width, clip_height) = self.canvas.clip()
        self.canvas.set_clip(inner_x, y, inner_width, WIDGET_HEIGHT)
        self.canvas.text(inner_x - scroll, text_y, result, self.theme.text)
        if focused and self.frames // _CARET_PERIOD % 2 == 0 {
            let caret_x = inner_x + caret_offset - scroll
            self.canvas.fill_rect(caret_x, text_y - 1, 1, GLYPH_HEIGHT + 2, self.theme.accent)
        }
        self.canvas.set_clip(clip_x, clip_y, clip_width, clip_height)
        return result
    }

    #: Draw a titled panel to group widgets, and return nothing.
    #:
    #: Draw it before the widgets that sit on it: it is opaque.
    pub fn panel(self, x: int, y: int, width: int, height: int, title: str) -> void {
        self.canvas.fill_rounded_rect(x, y, width, height, 6, self.theme.background)
        self.canvas.rounded_rect(x, y, width, height, 6, self.theme.border)
        if title.len() == 0 { return }
        let bar = GLYPH_HEIGHT + 2 * _PADDING
        self.canvas.text(x + _PADDING, y + _PADDING, title, self.theme.muted)
        self.canvas.fill_rect(x + 1, y + bar, width - 2, 1, self.theme.border)
    }

    fn _hover(self, id: str, x: int, y: int, width: int, height: int) -> bool {
        let inside = self.input.mouse.in_rect(x, y, width, height)
        if inside { self.hot = id }
        return inside
    }

    fn _edit(self, value: str) -> str {
        var result = value
        var caret = min(self.caret, result.len())
        let typed = self.input.text()
        if typed.len() > 0 {
            result = result[0:caret] + typed + result[caret:]
            caret += typed.len()
        }
        if self.input.keyboard.repeated(Key.Backspace) and caret > 0 {
            result = result[0:caret - 1] + result[caret:]
            caret -= 1
        }
        if self.input.keyboard.repeated(Key.Delete) and caret < result.len() {
            result = result[0:caret] + result[caret + 1:]
        }
        if self.input.keyboard.repeated(Key.Left) { caret = max(caret - 1, 0) }
        if self.input.keyboard.repeated(Key.Right) { caret = min(caret + 1, result.len()) }
        if self.input.keyboard.repeated(Key.Home) { caret = 0 }
        if self.input.keyboard.repeated(Key.End) { caret = result.len() }
        self.caret = caret
        return result
    }
}
