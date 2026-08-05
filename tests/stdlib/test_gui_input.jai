#: The platform-keycode translation in the native layer.
#:
#: `std.gui.Event` carries USB HID usage ids and `std.gui.input.Key` decodes
#: them; the window system underneath has its own numbering — on macOS a
#: virtual keycode, where Escape is 53 rather than 41 — and `src/native/gui.m`
#: translates between the two at the boundary. Nothing above the native layer
#: ever sees a platform code, which is exactly why the translation needs
#: pinning: when it was missing, `Key.from_code` returned null for every key
#: and `Keyboard.update` dropped the lot, so no GUI program had a keyboard at
#: all and no test noticed.
#:
#: Two things are checked, because either alone would pass while the keyboard
#: stayed dead:
#:
#:   1. the table, row by row, against `Key.code()` — so the native table and
#:      `_KEY_TABLE` in `lib/std/gui/input.jai` cannot drift apart; and
#:   2. the whole path, by synthesising a key transition through the same
#:      native code a real key event runs and asserting the `Keyboard` on this
#:      side names the right `Key`.
#:
#: A GUI program cannot be driven headlessly, so (2) goes through
#: `__prim__.gui_test_key_event` and `__prim__.gui_test_key_state`, the
#: documented injection point in `src/native/gui.m`. They write to a scratch
#: window private to the native layer, never to a window a program opened, and
#: need no display.
#:
#: On a platform with no window system there is no translation to check and
#: every lookup answers 0; those tests return early rather than assert.
from std.gui import Event
from std.gui.input import Key, Keyboard

from std.test import assert_eq, assert_true, assert_false

# Every key `Key` names, paired with the macOS virtual keycode that produces
# it, in the order `lib/std/gui/input.jai` lists them. The HID id is never
# written down here: each row asserts against `Key.code()`, so this table
# checks the native layer against the library's own numbers rather than
# against a third copy of them.
#
# One row per `Key` variant. A key added to `Key` without a row here fails
# `test_the_translation_covers_every_named_key_exactly_once`.
let _PLATFORM_KEYS: list[tuple[int, Key]] = [
    # Letters. AppKit numbers them by 1984 keyboard position, not alphabetically.
    (0, Key.A),
    (11, Key.B),
    (8, Key.C),
    (2, Key.D),
    (14, Key.E),
    (3, Key.F),
    (5, Key.G),
    (4, Key.H),
    (34, Key.I),
    (38, Key.J),
    (40, Key.K),
    (37, Key.L),
    (46, Key.M),
    (45, Key.N),
    (31, Key.O),
    (35, Key.P),
    (12, Key.Q),
    (15, Key.R),
    (1, Key.S),
    (17, Key.T),
    (32, Key.U),
    (9, Key.V),
    (13, Key.W),
    (7, Key.X),
    (16, Key.Y),
    (6, Key.Z),

    # Digits, along the top row.
    (18, Key.Digit1),
    (19, Key.Digit2),
    (20, Key.Digit3),
    (21, Key.Digit4),
    (23, Key.Digit5),
    (22, Key.Digit6),
    (26, Key.Digit7),
    (28, Key.Digit8),
    (25, Key.Digit9),
    (29, Key.Digit0),

    # Editing and punctuation. AppKit calls the backspace key Delete.
    (36, Key.Return),
    (53, Key.Escape),
    (51, Key.Backspace),
    (48, Key.Tab),
    (49, Key.Space),
    (27, Key.Minus),
    (24, Key.Equals),
    (33, Key.LeftBracket),
    (30, Key.RightBracket),
    (42, Key.Backslash),
    (41, Key.Semicolon),
    (39, Key.Quote),
    (50, Key.Backquote),
    (43, Key.Comma),
    (47, Key.Period),
    (44, Key.Slash),
    (57, Key.CapsLock),

    # Function keys, and the three positions a Mac keyboard reaches through F13.
    (122, Key.F1),
    (120, Key.F2),
    (99, Key.F3),
    (118, Key.F4),
    (96, Key.F5),
    (97, Key.F6),
    (98, Key.F7),
    (100, Key.F8),
    (101, Key.F9),
    (109, Key.F10),
    (103, Key.F11),
    (111, Key.F12),
    (105, Key.PrintScreen),
    (107, Key.ScrollLock),
    (113, Key.Pause),

    # The navigation cluster. Insert is the Mac's Help key.
    (114, Key.Insert),
    (115, Key.Home),
    (116, Key.PageUp),
    (117, Key.Delete),
    (119, Key.End),
    (121, Key.PageDown),
    (124, Key.Right),
    (123, Key.Left),
    (125, Key.Down),
    (126, Key.Up),

    # The keypad. NumLock is the Mac's Clear key.
    (71, Key.NumLock),
    (75, Key.KeypadDivide),
    (67, Key.KeypadMultiply),
    (78, Key.KeypadMinus),
    (69, Key.KeypadPlus),
    (76, Key.KeypadEnter),
    (83, Key.Keypad1),
    (84, Key.Keypad2),
    (85, Key.Keypad3),
    (86, Key.Keypad4),
    (87, Key.Keypad5),
    (88, Key.Keypad6),
    (89, Key.Keypad7),
    (91, Key.Keypad8),
    (92, Key.Keypad9),
    (82, Key.Keypad0),
    (65, Key.KeypadPeriod),
    (110, Key.Menu),

    # Modifiers, left and right distinct on both sides of the translation.
    (59, Key.LeftControl),
    (56, Key.LeftShift),
    (58, Key.LeftAlt),
    (55, Key.LeftSuper),
    (62, Key.RightControl),
    (60, Key.RightShift),
    (61, Key.RightAlt),
    (54, Key.RightSuper),
]

# Escape's macOS keycode. `Key.Escape.code()` is 41, and the gap between the
# two is the whole bug this file exists for.
const _PLATFORM_ESCAPE = 53

# The event tags `__prim__.gui_test_key_event` returns, which are the tags
# `gui_drain_events` returns; `std.gui` keeps its own private copy for the same
# reason this file does, since neither is worth making public API.
const _EVENT_KEY_DOWN = 1
const _EVENT_KEY_UP = 2

# True when the native layer has a keycode table at all, which is to say when
# this build has a window system. Everywhere else every code translates to 0
# and there is nothing here to assert.
#
# Deliberately the weakest question that separates the two: whether *some* code
# translates to *something*, never whether it translates correctly. A guard
# phrased as `gui_key_from_platform(53) == Key.Escape.code()` reads better and
# is worthless — a build whose table was wrong would answer "unsupported" and
# skip every test that would have caught it. This one skips only the stub.
fn _translation_is_supported() -> bool {
    for (platform_code, _key) in _PLATFORM_KEYS {
        if __prim__.gui_key_from_platform(platform_code) != 0 { return true }
    }
    return false
}

# Push one key transition through the native path a real key event takes and
# return what came back out, decoded exactly as `std.gui.Window.poll` decodes
# it. This is the whole point of the file: `platform_code` goes in as the
# window system's own number and `Event` comes out carrying a HID id.
fn _inject(platform_code: int, down: bool, repeat: bool) -> list[Event] {
    let events: list[Event] = []
    for raw in __prim__.gui_test_key_event(platform_code, down, repeat) {
        if raw[0] == _EVENT_KEY_DOWN { events.push(Event.KeyDown(raw[1], raw[2])) }
        if raw[0] == _EVENT_KEY_UP { events.push(Event.KeyUp(raw[1])) }
    }
    return events
}

fn test_every_named_key_translates_from_its_platform_keycode() -> void {
    if not _translation_is_supported() { return }
    for (platform_code, key) in _PLATFORM_KEYS {
        assert_eq(
            __prim__.gui_key_from_platform(platform_code),
            key.code(),
            f"platform keycode {platform_code} should translate to {key}"
        )
    }
}

fn test_the_translation_covers_every_named_key_exactly_once() -> void {
    # Two keys sharing a HID id, or one named twice, would leave some other key
    # with no row at all — and a row-by-row check cannot see a missing row.
    # `Key` has 104 variants, so 104 distinct ids is the whole enum covered.
    let ids: set[int] = set()
    for (_platform_code, key) in _PLATFORM_KEYS { ids.add(key.code()) }
    assert_eq(ids.len(), _PLATFORM_KEYS.len())
    assert_eq(ids.len(), 104)
}

fn test_no_two_keys_share_a_platform_keycode() -> void {
    let codes: set[int] = set()
    for (platform_code, _key) in _PLATFORM_KEYS { codes.add(platform_code) }
    assert_eq(codes.len(), _PLATFORM_KEYS.len())
}

fn test_a_platform_keycode_naming_no_key_translates_to_nothing() -> void {
    # HID usage 0 is the reserved "no event" id and `Key.from_code(0)` is null,
    # so 0 is how the native layer says "this build names no such key".
    assert_true(Key.from_code(0) is null)
    # The fn key, the volume keys and the JIS-only keys are real macOS keycodes
    # that `Key` does not name.
    assert_eq(__prim__.gui_key_from_platform(63), 0)  # fn
    assert_eq(__prim__.gui_key_from_platform(72), 0)  # volume up
    assert_eq(__prim__.gui_key_from_platform(93), 0)  # JIS yen
    # And nothing outside the table's range may index it.
    assert_eq(__prim__.gui_key_from_platform(-1), 0)
    assert_eq(__prim__.gui_key_from_platform(1000000), 0)
}

fn test_a_synthesised_key_press_arrives_as_the_key_it_names() -> void {
    if not _translation_is_supported() { return }
    let keyboard = Keyboard()

    # macOS reports Escape as 53. Untranslated it would reach `Key.from_code`
    # as 53, which names Backquote — and before the table existed it reached
    # nothing at all, because `Keyboard.update` drops a code no key names.
    keyboard.update(_inject(_PLATFORM_ESCAPE, true, false))
    assert_true(keyboard.just_pressed(Key.Escape))
    assert_true(keyboard.is_down(Key.Escape))
    assert_false(keyboard.is_down(Key.Backquote))

    keyboard.update(_inject(_PLATFORM_ESCAPE, false, false))
    assert_true(keyboard.just_released(Key.Escape))
    assert_false(keyboard.is_down(Key.Escape))
}

fn test_every_key_snake_reads_survives_the_round_trip() -> void {
    # The keys `examples/snake.jai` binds, which is what the missing table
    # actually broke: the arrows, WASD, Q and R. Each is pressed and released
    # so nothing stays stuck down for the next assertion.
    if not _translation_is_supported() { return }
    let keyboard = Keyboard()
    let bound: list[tuple[int, Key]] = [
        (126, Key.Up),
        (125, Key.Down),
        (123, Key.Left),
        (124, Key.Right),
        (13, Key.W),
        (1, Key.S),
        (0, Key.A),
        (2, Key.D),
        (12, Key.Q),
        (15, Key.R),
    ]
    for (platform_code, key) in bound {
        keyboard.update(_inject(platform_code, true, false))
        assert_true(keyboard.just_pressed(key), f"{key} should have been pressed")
        keyboard.update(_inject(platform_code, false, false))
        assert_false(keyboard.is_down(key), f"{key} should have been released")
    }
}

fn test_an_auto_repeat_is_reported_as_a_repeat_not_a_press() -> void {
    if not _translation_is_supported() { return }
    let keyboard = Keyboard()
    keyboard.update(_inject(_PLATFORM_ESCAPE, true, true))
    assert_true(keyboard.repeated(Key.Escape))
    assert_false(keyboard.just_pressed(Key.Escape))
    keyboard.update(_inject(_PLATFORM_ESCAPE, false, false))
}

fn test_the_polled_key_state_is_indexed_by_hid_usage_id() -> void {
    # `gui_key_down` and the array behind it are the other consumer of the
    # translation, and a table applied only on the event path would leave them
    # answering questions about macOS numbers. macOS W is 13, which is `Key.J`
    # in HID — so an untranslated array reports J held and W not.
    if not _translation_is_supported() { return }
    let pressed = __prim__.gui_test_key_event(13, true, false)
    assert_eq(pressed.len(), 1)
    assert_true(__prim__.gui_test_key_state().contains(Key.W.code()))
    assert_false(__prim__.gui_test_key_state().contains(Key.J.code()))

    let released = __prim__.gui_test_key_event(13, false, false)
    assert_eq(released.len(), 1)
    assert_false(__prim__.gui_test_key_state().contains(Key.W.code()))
}

fn test_an_unnamed_platform_key_reaches_neither_the_queue_nor_the_state() -> void {
    # The fn key names no `Key`, so it must be dropped in the native layer
    # rather than recorded at index 0, where nothing would ever clear it.
    if not _translation_is_supported() { return }
    let events = __prim__.gui_test_key_event(63, true, false)
    assert_eq(events.len(), 0)
    assert_false(__prim__.gui_test_key_state().contains(0))
}
