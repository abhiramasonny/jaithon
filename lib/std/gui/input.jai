#: Keyboard and mouse state, tracked from one frame to the next.
#:
#: Call `update` exactly once per frame, right after `Window.poll`: twice and
#: the second call sees no events, so every edge disappears; never, and the
#: state is frozen.
#:
#: ```
#: let keyboard = Keyboard()
#: keyboard.update([Event.KeyDown(4, false)])
#: keyboard.just_pressed(Key.A)    # true
#: keyboard.is_down(Key.A)         # true
#: keyboard.update([])
#: keyboard.just_pressed(Key.A)    # false
#: keyboard.is_down(Key.A)         # true
#: ```
from std.gui import Window, Event

#: Every key this library names.
#:
#: An unnamed key is dropped by `Keyboard`; add one as a single row in the
#: table at the bottom of this module.
#:
#: Letters are physical positions on a US layout (`Key.W` is above `Key.S`
#: everywhere, even AZERTY). Never build text from key codes — use
#: `Keyboard.text`, composed from the user's real layout.
pub enum Key {
    #: The key at the `A` position on a US layout.
    A,
    #: The key at the `B` position on a US layout.
    B,
    #: The key at the `C` position on a US layout.
    C,
    #: The key at the `D` position on a US layout.
    D,
    #: The key at the `E` position on a US layout.
    E,
    #: The key at the `F` position on a US layout.
    F,
    #: The key at the `G` position on a US layout.
    G,
    #: The key at the `H` position on a US layout.
    H,
    #: The key at the `I` position on a US layout.
    I,
    #: The key at the `J` position on a US layout.
    J,
    #: The key at the `K` position on a US layout.
    K,
    #: The key at the `L` position on a US layout.
    L,
    #: The key at the `M` position on a US layout.
    M,
    #: The key at the `N` position on a US layout.
    N,
    #: The key at the `O` position on a US layout.
    O,
    #: The key at the `P` position on a US layout.
    P,
    #: The key at the `Q` position on a US layout.
    Q,
    #: The key at the `R` position on a US layout.
    R,
    #: The key at the `S` position on a US layout.
    S,
    #: The key at the `T` position on a US layout.
    T,
    #: The key at the `U` position on a US layout.
    U,
    #: The key at the `V` position on a US layout.
    V,
    #: The key at the `W` position on a US layout.
    W,
    #: The key at the `X` position on a US layout.
    X,
    #: The key at the `Y` position on a US layout.
    Y,
    #: The key at the `Z` position on a US layout.
    Z,
    #: The `0` key on the number row above the letters, not `Keypad0`.
    Digit0,
    #: The `1` key on the number row above the letters, not `Keypad1`.
    Digit1,
    #: The `2` key on the number row above the letters, not `Keypad2`.
    Digit2,
    #: The `3` key on the number row above the letters, not `Keypad3`.
    Digit3,
    #: The `4` key on the number row above the letters, not `Keypad4`.
    Digit4,
    #: The `5` key on the number row above the letters, not `Keypad5`.
    Digit5,
    #: The `6` key on the number row above the letters, not `Keypad6`.
    Digit6,
    #: The `7` key on the number row above the letters, not `Keypad7`.
    Digit7,
    #: The `8` key on the number row above the letters, not `Keypad8`.
    Digit8,
    #: The `9` key on the number row above the letters, not `Keypad9`.
    Digit9,
    #: The main return key. The keypad has its own, `KeypadEnter`.
    Return,
    #: The escape key, at the top left of the main block.
    Escape,
    #: The backspace key, which erases to the left of the caret. The key
    #: labelled delete on a Mac keyboard reports as this one, not as `Delete`.
    Backspace,
    #: The tab key.
    Tab,
    #: The space bar. Read the space character itself from `Keyboard.text`,
    #: which the platform composes with the user's layout.
    Space,
    #: The `-` and `_` key, right of `Digit0` on the number row. The keypad
    #: has its own, `KeypadMinus`.
    Minus,
    #: The `=` and `+` key, right of `Minus`. The keypad's plus is `KeypadPlus`.
    Equals,
    #: The `[` and `{` key, right of `P` on a US layout.
    LeftBracket,
    #: The `]` and `}` key, right of `LeftBracket` on a US layout.
    RightBracket,
    #: The `\` and `|` key, above `Return` on a US layout.
    Backslash,
    #: The `;` and `:` key, right of `L` on a US layout.
    Semicolon,
    #: The `'` and `"` key, right of `Semicolon` on a US layout.
    Quote,
    #: The backtick and tilde key, left of `Digit1` on a US layout.
    Backquote,
    #: The `,` and `<` key, right of `M` on a US layout.
    Comma,
    #: The `.` and `>` key, right of `Comma`. The keypad's is `KeypadPeriod`.
    Period,
    #: The `/` and `?` key, right of `Period`. The keypad's is `KeypadDivide`.
    Slash,
    #: The caps lock key. It reports a press and a release like any other key;
    #: nothing here says whether the lock is currently on.
    CapsLock,
    #: The `F1` function key.
    F1,
    #: The `F2` function key.
    F2,
    #: The `F3` function key.
    F3,
    #: The `F4` function key.
    F4,
    #: The `F5` function key.
    F5,
    #: The `F6` function key.
    F6,
    #: The `F7` function key.
    F7,
    #: The `F8` function key.
    F8,
    #: The `F9` function key.
    F9,
    #: The `F10` function key.
    F10,
    #: The `F11` function key.
    F11,
    #: The `F12` function key.
    F12,
    #: The print screen key. A window system that drives its own screenshot
    #: tool from this key may take it before the window ever sees it.
    PrintScreen,
    #: The scroll lock key, absent from most compact keyboards.
    ScrollLock,
    #: The pause and break key, absent from most compact keyboards.
    Pause,
    #: The insert key, which toggles overtype in editors that have the mode.
    Insert,
    #: The home key, conventionally the start of the line or document.
    Home,
    #: The page up key.
    PageUp,
    #: The forward delete key, which erases to the right of the caret. It is
    #: not the key labelled delete on a Mac keyboard; that one is `Backspace`.
    Delete,
    #: The end key, conventionally the end of the line or document.
    End,
    #: The page down key.
    PageDown,
    #: The right arrow key.
    Right,
    #: The left arrow key.
    Left,
    #: The down arrow key.
    Down,
    #: The up arrow key.
    Up,
    #: The num lock key. The keypad keys keep their own names whichever way it
    #: is set, so an arrow pressed on the keypad arrives as `Keypad8`, not `Up`.
    NumLock,
    #: The keypad `/` key.
    KeypadDivide,
    #: The keypad `*` key.
    KeypadMultiply,
    #: The keypad `-` key.
    KeypadMinus,
    #: The keypad `+` key.
    KeypadPlus,
    #: The keypad's own enter key. The main one is `Return`.
    KeypadEnter,
    #: The keypad `1` key, not the number row's `Digit1`.
    Keypad1,
    #: The keypad `2` key, not the number row's `Digit2`.
    Keypad2,
    #: The keypad `3` key, not the number row's `Digit3`.
    Keypad3,
    #: The keypad `4` key, not the number row's `Digit4`.
    Keypad4,
    #: The keypad `5` key, not the number row's `Digit5`.
    Keypad5,
    #: The keypad `6` key, not the number row's `Digit6`.
    Keypad6,
    #: The keypad `7` key, not the number row's `Digit7`.
    Keypad7,
    #: The keypad `8` key, not the number row's `Digit8`.
    Keypad8,
    #: The keypad `9` key, not the number row's `Digit9`.
    Keypad9,
    #: The keypad `0` key, not the number row's `Digit0`.
    Keypad0,
    #: The keypad `.` key, which doubles as the keypad's delete key when num
    #: lock is off.
    KeypadPeriod,
    #: The context-menu key, right of the space bar on full-size PC keyboards.
    #: Mac keyboards do not have one.
    Menu,
    #: The left control key. A shortcut must test `RightControl` as well; no
    #: variant stands for either one.
    LeftControl,
    #: The left shift key. The platform has already applied shift to the text
    #: it composes, so read `Keyboard.text` rather than upper-casing by hand.
    LeftShift,
    #: The left alt key, labelled option on Mac keyboards.
    LeftAlt,
    #: The left super key: command on macOS, the Windows key on PC keyboards.
    LeftSuper,
    #: The right control key. Compact keyboards often leave it out, so a
    #: shortcut that only accepts this one is unreachable on them.
    RightControl,
    #: The right shift key.
    RightShift,
    #: The right alt key, which international layouts use as AltGr to reach a
    #: third character on a key.
    RightAlt,
    #: The right super key: command on macOS, the Windows key on PC keyboards.
    #: Compact keyboards often leave it out.
    RightSuper,

    #: Return the key a platform key code names, or null for an unknown code.
    #:
    #: ```
    #: Key.from_code(41)     # Key.Escape
    #: Key.from_code(9999)   # null
    #: ```
    pub static fn from_code(code: int) -> Key? { return _CODE_TO_KEY.get(code, null) }

    #: Return the platform key code for this key.
    #:
    #: ```
    #: Key.Escape.code()    # 41
    #: ```
    pub fn code(self) -> int { return _KEY_TO_CODE[self] }
}

#: The three mouse buttons this library names.
pub enum MouseButton {
    #: The primary button, button code 0 on an `Event.MouseDown`.
    Left,
    #: The wheel button, button code 1. Turning the wheel is an `Event.Wheel`
    #: and never a button press.
    Middle,
    #: The secondary button, button code 2, the one that opens a context menu.
    Right,

    #: Return the button a platform button code names, or null.
    pub static fn from_code(code: int) -> MouseButton? {
        if code == 0 { return MouseButton.Left }
        if code == 1 { return MouseButton.Middle }
        if code == 2 { return MouseButton.Right }
        return null
    }
}

#: Which keys are down, and which changed in the frame just polled.
pub class Keyboard {
    var down: set[Key]
    var pressed: set[Key]
    var released: set[Key]
    var repeats: set[Key]
    var typed: str

    #: Create a keyboard with nothing held down.
    pub fn init(self) {
        self.down = set()
        self.pressed = set()
        self.released = set()
        self.repeats = set()
        self.typed = ""
    }

    #: Fold one frame of events into the state.
    pub fn update(self, events: list[Event]) -> void {
        self.pressed = set()
        self.released = set()
        self.repeats = set()
        self.typed = ""
        for event in events {
            match event {
                Event.KeyDown(code, repeat) => {
                    let key = Key.from_code(code)
                    if key is not null {
                        self.repeats.add(key)
                        if not repeat {
                            self.down.add(key)
                            self.pressed.add(key)
                        }
                    }
                }
                Event.KeyUp(code) => {
                    let key = Key.from_code(code)
                    if key is not null {
                        if self.down.has(key) { self.down.remove(key) }
                        self.released.add(key)
                    }
                }
                Event.Text(text) => { self.typed += text }
                Event.Focus(gained) => { if not gained { self.down = set() } }
                # Keys held when the window loses focus never send their
                # release, so drop them rather than leaving a key stuck.
                _ => { }
            }
        }
    }

    #: Report whether `key` is held down right now.
    pub fn is_down(self, key: Key) -> bool { return self.down.has(key) }

    #: Report whether `key` went down during the frame just polled.
    pub fn just_pressed(self, key: Key) -> bool { return self.pressed.has(key) }

    #: Report whether `key` came up during the frame just polled.
    pub fn just_released(self, key: Key) -> bool { return self.released.has(key) }

    #: Report whether `key` fired this frame, first press or auto-repeat.
    #:
    #: Editors want this for Backspace and arrow keys (repeat keeps moving);
    #: games want `just_pressed` for actions like a jump (repeat should not).
    pub fn repeated(self, key: Key) -> bool { return self.repeats.has(key) }

    #: Text composed during the frame just polled, "" when none.
    #:
    #: The only correct source of characters — dead keys and input method
    #: composition make key codes insufficient.
    pub fn text(self) -> str { return self.typed }

    #: Report whether either shift key is held.
    pub fn shift(self) -> bool {
        return self.down.has(Key.LeftShift) or self.down.has(Key.RightShift)
    }

    #: Report whether either control key is held.
    pub fn control(self) -> bool {
        return self.down.has(Key.LeftControl) or self.down.has(Key.RightControl)
    }

    #: Report whether either alt (option) key is held.
    pub fn alt(self) -> bool {
        return self.down.has(Key.LeftAlt) or self.down.has(Key.RightAlt)
    }

    #: Report whether either command (super) key is held.
    pub fn meta(self) -> bool {
        return self.down.has(Key.LeftSuper) or self.down.has(Key.RightSuper)
    }
}

#: Where the pointer is, which buttons are down, and how far it moved.
pub class Mouse {
    var position_x: int
    var position_y: int
    var previous_x: int
    var previous_y: int
    var wheel_x: float
    var wheel_y: float
    var down: set[MouseButton]
    var pressed: set[MouseButton]
    var released: set[MouseButton]

    #: Create a mouse at the origin with no buttons down.
    pub fn init(self) {
        self.position_x = 0
        self.position_y = 0
        self.previous_x = 0
        self.previous_y = 0
        self.wheel_x = 0.0
        self.wheel_y = 0.0
        self.down = set()
        self.pressed = set()
        self.released = set()
    }

    #: Fold one frame of events into the state.
    pub fn update(self, events: list[Event]) -> void {
        self.previous_x = self.position_x
        self.previous_y = self.position_y
        self.pressed = set()
        self.released = set()
        self.wheel_x = 0.0
        self.wheel_y = 0.0
        for event in events {
            match event {
                Event.MouseMove(x, y) => {
                    self.position_x = x
                    self.position_y = y
                }
                Event.MouseDown(code, x, y) => {
                    self.position_x = x
                    self.position_y = y
                    let button = MouseButton.from_code(code)
                    if button is not null {
                        self.down.add(button)
                        self.pressed.add(button)
                    }
                }
                Event.MouseUp(code, x, y) => {
                    self.position_x = x
                    self.position_y = y
                    let button = MouseButton.from_code(code)
                    if button is not null {
                        if self.down.has(button) { self.down.remove(button) }
                        self.released.add(button)
                    }
                }
                Event.Wheel(dx, dy) => {
                    self.wheel_x += dx
                    self.wheel_y += dy
                }
                _ => { }
            }
        }
    }

    #: Pointer x in buffer pixels.
    pub fn x(self) -> int { return self.position_x }

    #: Pointer y in buffer pixels.
    pub fn y(self) -> int { return self.position_y }

    #: Pointer position as `(x, y)` in buffer pixels.
    pub fn position(self) -> tuple[int, int] { return (self.position_x, self.position_y) }

    #: How far the pointer moved during the frame just polled, as `(dx, dy)`.
    pub fn delta(self) -> tuple[int, int] {
        return (self.position_x - self.previous_x, self.position_y - self.previous_y)
    }

    #: Wheel movement during the frame just polled, in lines, positive dy up.
    pub fn wheel(self) -> tuple[float, float] { return (self.wheel_x, self.wheel_y) }

    #: Report whether `button` is held down right now.
    pub fn is_down(self, button: MouseButton) -> bool { return self.down.has(button) }

    #: Report whether `button` went down during the frame just polled.
    pub fn just_pressed(self, button: MouseButton) -> bool { return self.pressed.has(button) }

    #: Report whether `button` came up during the frame just polled.
    pub fn just_released(self, button: MouseButton) -> bool { return self.released.has(button) }

    #: Report whether the pointer is inside this rectangle.
    pub fn in_rect(self, x: int, y: int, width: int, height: int) -> bool {
        let inside_x = x <= self.position_x and self.position_x < x + width
        let inside_y = y <= self.position_y and self.position_y < y + height
        return inside_x and inside_y
    }
}

#: The keyboard and the mouse for one window, updated together once a frame.
pub class Input {
    #: Keyboard state. Reach through it for `text`, `repeated` and modifiers.
    pub let keyboard: Keyboard

    #: Mouse state.
    pub let mouse: Mouse

    let window: Window

    #: Create the input state for `window`.
    pub fn init(self, window: Window) {
        self.window = window
        self.keyboard = Keyboard()
        self.mouse = Mouse()
    }

    #: Fold the events of the frame just polled into the keyboard and mouse.
    #:
    #: Call it once per frame, right after `Window.poll`.
    pub fn update(self) -> void {
        let events = self.window.events()
        self.keyboard.update(events)
        self.mouse.update(events)
    }

    #: Report whether `key` is held down right now.
    pub fn is_down(self, key: Key) -> bool { return self.keyboard.is_down(key) }

    #: Report whether `key` went down during the frame just polled.
    pub fn just_pressed(self, key: Key) -> bool { return self.keyboard.just_pressed(key) }

    #: Report whether `key` came up during the frame just polled.
    pub fn just_released(self, key: Key) -> bool { return self.keyboard.just_released(key) }

    #: Text composed during the frame just polled, "" when none.
    pub fn text(self) -> str { return self.keyboard.text() }
}

# USB HID usage ids: the same numbers on every desktop platform, so one
# table covers all of them.
let _KEY_TABLE: list[tuple[int, Key]] = [
    (4, Key.A),
    (5, Key.B),
    (6, Key.C),
    (7, Key.D),
    (8, Key.E),
    (9, Key.F),
    (10, Key.G),
    (11, Key.H),
    (12, Key.I),
    (13, Key.J),
    (14, Key.K),
    (15, Key.L),
    (16, Key.M),
    (17, Key.N),
    (18, Key.O),
    (19, Key.P),
    (20, Key.Q),
    (21, Key.R),
    (22, Key.S),
    (23, Key.T),
    (24, Key.U),
    (25, Key.V),
    (26, Key.W),
    (27, Key.X),
    (28, Key.Y),
    (29, Key.Z),
    (30, Key.Digit1),
    (31, Key.Digit2),
    (32, Key.Digit3),
    (33, Key.Digit4),
    (34, Key.Digit5),
    (35, Key.Digit6),
    (36, Key.Digit7),
    (37, Key.Digit8),
    (38, Key.Digit9),
    (39, Key.Digit0),
    (40, Key.Return),
    (41, Key.Escape),
    (42, Key.Backspace),
    (43, Key.Tab),
    (44, Key.Space),
    (45, Key.Minus),
    (46, Key.Equals),
    (47, Key.LeftBracket),
    (48, Key.RightBracket),
    (49, Key.Backslash),
    (51, Key.Semicolon),
    (52, Key.Quote),
    (53, Key.Backquote),
    (54, Key.Comma),
    (55, Key.Period),
    (56, Key.Slash),
    (57, Key.CapsLock),
    (58, Key.F1),
    (59, Key.F2),
    (60, Key.F3),
    (61, Key.F4),
    (62, Key.F5),
    (63, Key.F6),
    (64, Key.F7),
    (65, Key.F8),
    (66, Key.F9),
    (67, Key.F10),
    (68, Key.F11),
    (69, Key.F12),
    (70, Key.PrintScreen),
    (71, Key.ScrollLock),
    (72, Key.Pause),
    (73, Key.Insert),
    (74, Key.Home),
    (75, Key.PageUp),
    (76, Key.Delete),
    (77, Key.End),
    (78, Key.PageDown),
    (79, Key.Right),
    (80, Key.Left),
    (81, Key.Down),
    (82, Key.Up),
    (83, Key.NumLock),
    (84, Key.KeypadDivide),
    (85, Key.KeypadMultiply),
    (86, Key.KeypadMinus),
    (87, Key.KeypadPlus),
    (88, Key.KeypadEnter),
    (89, Key.Keypad1),
    (90, Key.Keypad2),
    (91, Key.Keypad3),
    (92, Key.Keypad4),
    (93, Key.Keypad5),
    (94, Key.Keypad6),
    (95, Key.Keypad7),
    (96, Key.Keypad8),
    (97, Key.Keypad9),
    (98, Key.Keypad0),
    (99, Key.KeypadPeriod),
    (101, Key.Menu),
    (224, Key.LeftControl),
    (225, Key.LeftShift),
    (226, Key.LeftAlt),
    (227, Key.LeftSuper),
    (228, Key.RightControl),
    (229, Key.RightShift),
    (230, Key.RightAlt),
    (231, Key.RightSuper),
]

let _CODE_TO_KEY: dict[int, Key] = _index_by_code()
let _KEY_TO_CODE: dict[Key, int] = _index_by_key()

fn _index_by_code() -> dict[int, Key] {
    let table: dict[int, Key] = {}
    for (code, key) in _KEY_TABLE { table[code] = key }
    return table
}

fn _index_by_key() -> dict[Key, int] {
    let table: dict[Key, int] = {}
    for (code, key) in _KEY_TABLE { table[key] = code }
    return table
}
