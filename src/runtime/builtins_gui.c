/* builtins_gui.c — __prim__.gui_*, the surface std.gui is written over.
 *
 * std.gui owns every pixel of drawing; this file owns exactly four things: the
 * window, the back buffer, the event queue and the frame clock. The split is
 * why `Canvas` can draw to a window, to an offscreen image and to a test buffer
 * with no display attached — none of that code is here.
 *
 * The back buffer is the one interesting decision. `Window.pixels()` hands out
 * a `list[int]` and promises it is live: `Canvas` caches it once at startup and
 * assigns into it for the rest of the program's life, never asking for it
 * again. A Jaithon list is an array of tagged `Value`s and the native buffer is
 * an array of packed `uint32_t`, so the two cannot be the same memory. The list
 * is therefore the buffer of record and `gui_present` packs it down into the
 * ARGB buffer on the way out — one linear pass over the frame, which at
 * 640x480 is far cheaper than the upload that follows it. Handing back an
 * opaque handle plus a put_pixel primitive would be faster still per pixel, but
 * it would put a native call in the innermost loop of every drawing routine,
 * and that is the loop std.gui.canvas exists to keep in Jaithon.
 *
 * That list has to survive collections while only C holds it, and the GC has no
 * permanent-root API — only the temp-root stack, which unwinds. So every live
 * window's buffer is also parked in a list hanging off `__prim__`, which is a
 * module global and therefore a root the collector already traces.
 */

#include "runtime.h"
#include "handles.h"

#include "../native/native.h"
#include "../vm/gc.h"

/* Event records are drained in batches rather than one at a time so that a
 * frame's worth of input costs one native call. */
#define GUI_EVENT_BATCH 64

/* Keys are USB HID usage ids. The keyboard/keypad page stops well inside a
 * byte, so this bounds every scan over a window's polled key state. */
#define GUI_HID_CODE_LIMIT 256

typedef struct {
    JaiWindow *window;
    ObjList   *pixels;      /* the live back buffer; see the header comment */
    int        width;
    int        height;
    int        slot;        /* index into gKeepAlive */
} GuiWindow;

/* Every live window's pixel list, so the collector can see them. It is a
 * permanent GC root rather than a published name: as `__prim__.gui_buffers` it
 * was reachable from ordinary Jaithon, and clearing it there freed buffers the
 * native window still drew into. */
static ObjList *gKeepAlive;

static bool requireGui(const char *fnName) {
    if (jaiGuiAvailable()) return true;
    return jaiThrow(vm.cRuntimeError,
                    "%s(): no display is available on this machine", fnName);
}

static bool requireWindow(Value v, int index, const char *fnName,
                          GuiWindow **out) {
    if (!requireGui(fnName)) return false;
    void *ptr;
    if (!jaiHandleGet(v, index, HANDLE_WINDOW, fnName, &ptr)) return false;
    *out = (GuiWindow *)ptr;
    return true;
}

/* ------------------------------------------------------------------ */
/* Lifetime                                                            */
/* ------------------------------------------------------------------ */

static bool nGuiAvailable(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    *out = BOOL_VAL(jaiGuiAvailable());
    return true;
}

/* gui_window_open(width, height, title, target_fps = 0)
 *
 * std.gui paces its own frames with time_mono and sleep, so it opens the window
 * uncapped and the native limiter stays out of the way. The parameter is still
 * here because a caller that wants the native pacing can ask for it. */
static bool nGuiWindowOpen(int argc, Value *args, Value *out) {
    if (!requireGui("gui_window_open")) return false;

    int64_t width, height, targetFPS = 0;
    ObjString *title;
    if (!jaiArgInt(args[0], 1, "gui_window_open", &width)) return false;
    if (!jaiArgInt(args[1], 2, "gui_window_open", &height)) return false;
    if (!jaiArgString(args[2], 3, "gui_window_open", &title)) return false;
    if (argc >= 4 && !IS_NULL(args[3]) &&
        !jaiArgInt(args[3], 4, "gui_window_open", &targetFPS))
        return false;

    if (width <= 0 || height <= 0 || width > INT32_MAX || height > INT32_MAX)
        return jaiThrow(vm.cValueError,
                        "gui_window_open(): size must be positive, got %lldx%lld",
                        (long long)width, (long long)height);
    if (targetFPS < 0 || targetFPS > INT32_MAX)
        return jaiThrow(vm.cValueError,
                        "gui_window_open(): target_fps must be non-negative, got %lld",
                        (long long)targetFPS);

    JaiWindow *window = jaiWindowOpen((int)width, (int)height, title->chars,
                                      (int)targetFPS);
    if (window == NULL)
        return jaiThrow(vm.cRuntimeError,
                        "gui_window_open(): the window could not be created");

    /* Allocating the list can collect, and the window is not yet reachable from
     * anywhere the collector looks — but it is not a GC object either, so the
     * only thing at risk is the list itself, which is rooted below. */
    int64_t count = width * height;
    ObjList *pixels = jaiListNew((int)count);
    jaiGCPushRoot(OBJ_VAL(pixels));
    for (int64_t i = 0; i < count; i++) jaiListPush(pixels, INT_VAL(0));

    /* Reuse a slot a closed window left behind, so open/close in a loop does
     * not grow the keepalive list without bound. */
    int slot = -1;
    for (int i = 0; i < gKeepAlive->count; i++) {
        if (IS_NULL(gKeepAlive->items[i])) { slot = i; break; }
    }
    if (slot < 0) {
        slot = gKeepAlive->count;
        jaiListPush(gKeepAlive, OBJ_VAL(pixels));
    } else {
        gKeepAlive->items[slot] = OBJ_VAL(pixels);
    }
    jaiGCPopRoot();

    GuiWindow *record = JAI_ALLOC_ZEROED(GuiWindow, 1);
    record->window = window;
    record->pixels = pixels;
    record->width = (int)width;
    record->height = (int)height;
    record->slot = slot;

    *out = INT_VAL(jaiHandleAdd(HANDLE_WINDOW, record));
    return true;
}

static bool nGuiWindowClose(int argc, Value *args, Value *out) {
    (void)argc;
    GuiWindow *w;
    if (!requireWindow(args[0], 1, "gui_window_close", &w)) return false;

    /* Retire the handle first: a second close is then a clean error rather than
     * a double free, which is what makes `close` in a `defer` safe. */
    jaiHandleRelease(AS_INT(args[0]));
    if (w->slot >= 0 && w->slot < gKeepAlive->count)
        gKeepAlive->items[w->slot] = NULL_VAL;
    jaiWindowClose(w->window);
    JAI_FREE(GuiWindow, w);

    *out = NULL_VAL;
    return true;
}

/* ------------------------------------------------------------------ */
/* Back buffer                                                         */
/* ------------------------------------------------------------------ */

static bool nGuiPixels(int argc, Value *args, Value *out) {
    (void)argc;
    GuiWindow *w;
    if (!requireWindow(args[0], 1, "gui_pixels", &w)) return false;
    *out = OBJ_VAL(w->pixels);
    return true;
}

static bool nGuiPresent(int argc, Value *args, Value *out) {
    (void)argc;
    GuiWindow *w;
    if (!requireWindow(args[0], 1, "gui_present", &w)) return false;

    int bufferWidth = 0, bufferHeight = 0;
    uint32_t *target = jaiWindowPixels(w->window, &bufferWidth, &bufferHeight);
    if (target == NULL) {
        *out = NULL_VAL;
        return true;                    /* already closed; nothing to show */
    }

    int64_t needed = (int64_t)bufferWidth * bufferHeight;
    /* Resizing the list detaches it from the frame it is supposed to be, and
     * every pixel after the change would land in the wrong row. */
    if (w->pixels->count != needed)
        return jaiThrow(vm.cValueError,
                        "gui_present(): the pixel buffer holds %d values but "
                        "the window is %dx%d (%lld)",
                        w->pixels->count, bufferWidth, bufferHeight,
                        (long long)needed);

    const Value *items = w->pixels->items;
    for (int64_t i = 0; i < needed; i++) {
        if (!IS_INT(items[i]))
            return jaiThrow(vm.cTypeError,
                            "gui_present(): pixel %lld is %s, expected a packed "
                            "0xAARRGGBB int",
                            (long long)i, jaiTypeNameStatic(items[i]));
        target[i] = (uint32_t)(uint64_t)AS_INT(items[i]);
    }

    jaiWindowPresent(w->window);
    *out = NULL_VAL;
    return true;
}

/* ------------------------------------------------------------------ */
/* Events                                                              */
/* ------------------------------------------------------------------ */

static bool nGuiPoll(int argc, Value *args, Value *out) {
    (void)argc;
    GuiWindow *w;
    if (!requireWindow(args[0], 1, "gui_poll", &w)) return false;
    *out = BOOL_VAL(jaiWindowPoll(w->window));
    return true;
}

/* Append `count` values as one event record. The record is `[tag, ...payload]`,
 * which is what std.gui's `_decode` reads; a tag it does not know is dropped
 * there, so this may grow without breaking an older library. */
static bool pushRecord(ObjList *into, const Value *values, int count) {
    ObjList *record = jaiListNew(count);
    jaiGCPushRoot(OBJ_VAL(record));
    for (int i = 0; i < count; i++) jaiListPush(record, values[i]);
    jaiListPush(into, OBJ_VAL(record));
    jaiGCPopRoot();
    return true;
}

/* Move every queued event of `window` into `events` as one record apiece.
 * `events` must already be rooted: appending to it allocates. */
static void drainEventsInto(JaiWindow *window, ObjList *events) {
    JaiWindowEvent batch[GUI_EVENT_BATCH];
    int taken;
    do {
        taken = jaiWindowDrainEvents(window, batch, GUI_EVENT_BATCH);
        for (int i = 0; i < taken; i++) {
            const JaiWindowEvent *e = &batch[i];
            Value payload[4];
            payload[0] = INT_VAL(e->tag);

            switch ((JaiWindowEventTag)e->tag) {
            case JAI_EVENT_CLOSE:
                pushRecord(events, payload, 1);
                break;
            case JAI_EVENT_KEY_DOWN:
                payload[1] = INT_VAL(e->i0);
                payload[2] = BOOL_VAL(e->i1 != 0);
                pushRecord(events, payload, 3);
                break;
            case JAI_EVENT_KEY_UP:
                payload[1] = INT_VAL(e->i0);
                pushRecord(events, payload, 2);
                break;
            case JAI_EVENT_TEXT: {
                ObjString *text = jaiStringInternC(e->text);
                jaiGCPushRoot(OBJ_VAL(text));
                payload[1] = OBJ_VAL(text);
                pushRecord(events, payload, 2);
                jaiGCPopRoot();
                break;
            }
            case JAI_EVENT_MOUSE_MOVE:
                payload[1] = INT_VAL(e->i0);
                payload[2] = INT_VAL(e->i1);
                pushRecord(events, payload, 3);
                break;
            case JAI_EVENT_MOUSE_DOWN:
            case JAI_EVENT_MOUSE_UP:
                payload[1] = INT_VAL(e->i0);
                payload[2] = INT_VAL(e->i1);
                payload[3] = INT_VAL(e->i2);
                pushRecord(events, payload, 4);
                break;
            case JAI_EVENT_WHEEL:
                payload[1] = FLOAT_VAL(e->d0);
                payload[2] = FLOAT_VAL(e->d1);
                pushRecord(events, payload, 3);
                break;
            case JAI_EVENT_RESIZE:
                payload[1] = INT_VAL(e->i0);
                payload[2] = INT_VAL(e->i1);
                pushRecord(events, payload, 3);
                break;
            case JAI_EVENT_FOCUS:
                payload[1] = BOOL_VAL(e->i0 != 0);
                pushRecord(events, payload, 2);
                break;
            }
        }
    } while (taken == GUI_EVENT_BATCH);
}

static bool nGuiDrainEvents(int argc, Value *args, Value *out) {
    (void)argc;
    GuiWindow *w;
    if (!requireWindow(args[0], 1, "gui_drain_events", &w)) return false;

    ObjList *events = jaiListNew(0);
    jaiGCPushRoot(OBJ_VAL(events));
    drainEventsInto(w->window, events);
    jaiGCPopRoot();
    *out = OBJ_VAL(events);
    return true;
}

/* ------------------------------------------------------------------ */
/* Frame clock and polled input                                        */
/* ------------------------------------------------------------------ */

static bool nGuiDeltaTime(int argc, Value *args, Value *out) {
    (void)argc;
    GuiWindow *w;
    if (!requireWindow(args[0], 1, "gui_delta_time", &w)) return false;
    *out = FLOAT_VAL(jaiWindowDeltaTime(w->window));
    return true;
}

static bool nGuiFps(int argc, Value *args, Value *out) {
    (void)argc;
    GuiWindow *w;
    if (!requireWindow(args[0], 1, "gui_fps", &w)) return false;
    *out = FLOAT_VAL(jaiWindowFPS(w->window));
    return true;
}

static bool nGuiMousePos(int argc, Value *args, Value *out) {
    (void)argc;
    GuiWindow *w;
    if (!requireWindow(args[0], 1, "gui_mouse_pos", &w)) return false;
    double x = 0.0, y = 0.0;
    jaiWindowMousePos(w->window, &x, &y);
    Value pair[2] = {FLOAT_VAL(x), FLOAT_VAL(y)};
    *out = OBJ_VAL(jaiTupleNew(pair, 2));
    return true;
}

static bool nGuiMouseDown(int argc, Value *args, Value *out) {
    (void)argc;
    GuiWindow *w;
    if (!requireWindow(args[0], 1, "gui_mouse_down", &w)) return false;
    int64_t button;
    if (!jaiArgInt(args[1], 2, "gui_mouse_down", &button)) return false;
    *out = BOOL_VAL(button >= INT32_MIN && button <= INT32_MAX &&
                    jaiWindowMouseDown(w->window, (int)button));
    return true;
}

static bool nGuiKeyDown(int argc, Value *args, Value *out) {
    (void)argc;
    GuiWindow *w;
    if (!requireWindow(args[0], 1, "gui_key_down", &w)) return false;
    int64_t keycode;
    if (!jaiArgInt(args[1], 2, "gui_key_down", &keycode)) return false;
    *out = BOOL_VAL(keycode >= INT32_MIN && keycode <= INT32_MAX &&
                    jaiWindowKeyDown(w->window, (int)keycode));
    return true;
}

/* ------------------------------------------------------------------ */
/* Key code translation, and the test harness over it                  */
/* ------------------------------------------------------------------ */

/* gui_key_from_platform(platform_code) -> hid_code
 *
 * The window system's own number for a key — an AppKit virtual keycode on
 * macOS — as the USB HID usage id every event carries, or 0 for a key std.gui
 * does not name. Every event leaving the native layer is already translated,
 * so std.gui itself never calls this; it exists so the table can be checked
 * against `Key.code()` on a machine with no display. */
static bool nGuiKeyFromPlatform(int argc, Value *args, Value *out) {
    (void)argc;
    int64_t code;
    if (!jaiArgInt(args[0], 1, "gui_key_from_platform", &code)) return false;
    /* Out of int range is out of the table's range: 0, "names no key". */
    int platform = (code >= INT32_MIN && code <= INT32_MAX) ? (int)code : -1;
    *out = INT_VAL(jaiWindowKeyFromPlatform(platform));
    return true;
}

/* gui_test_key_event(platform_code, down, repeat = false) -> list[record]
 *
 * Test-only. Feeds one synthetic key transition through the same native path a
 * real key event takes and returns the events it produced, in exactly the
 * records `gui_drain_events` yields. A GUI program cannot be driven headlessly,
 * so this is the only way to prove a platform keycode reaches Jaithon as the
 * right `Key`; see tests/stdlib/test_gui_input.jai.
 *
 * It touches a scratch window private to the native layer, never a window the
 * program opened, and needs no display. On a platform with no window system it
 * injects nothing and returns an empty list. */
static bool nGuiTestKeyEvent(int argc, Value *args, Value *out) {
    int64_t code;
    bool down = false, repeat = false;
    if (!jaiArgInt(args[0], 1, "gui_test_key_event", &code)) return false;
    if (!jaiArgBool(args[1], 2, "gui_test_key_event", &down)) return false;
    if (argc >= 3 && !IS_NULL(args[2]) &&
        !jaiArgBool(args[2], 3, "gui_test_key_event", &repeat))
        return false;

    int platform = (code >= INT32_MIN && code <= INT32_MAX) ? (int)code : -1;
    jaiWindowTestInjectKey(platform, down, repeat);

    ObjList *events = jaiListNew(0);
    jaiGCPushRoot(OBJ_VAL(events));
    drainEventsInto(jaiWindowTestWindow(), events);
    jaiGCPopRoot();
    *out = OBJ_VAL(events);
    return true;
}

/* gui_test_key_state() -> list[int]
 *
 * Test-only companion to the above: the HID usage ids the scratch window's
 * polled arrays currently report down, ascending. This is `gui_key_down` read
 * over every code at once, and it is what proves those arrays are indexed by
 * HID id rather than by the platform's own number. */
static bool nGuiTestKeyState(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    JaiWindow *window = jaiWindowTestWindow();
    ObjList *held = jaiListNew(0);
    jaiGCPushRoot(OBJ_VAL(held));
    for (int hid = 0; hid < GUI_HID_CODE_LIMIT; hid++) {
        if (jaiWindowKeyDown(window, hid)) jaiListPush(held, INT_VAL(hid));
    }
    jaiGCPopRoot();
    *out = OBJ_VAL(held);
    return true;
}

static bool nGuiSetTitle(int argc, Value *args, Value *out) {
    (void)argc;
    GuiWindow *w;
    if (!requireWindow(args[0], 1, "gui_set_title", &w)) return false;
    ObjString *title;
    if (!jaiArgString(args[1], 2, "gui_set_title", &title)) return false;
    jaiWindowSetTitle(w->window, title->chars);
    *out = NULL_VAL;
    return true;
}

void jaiRegisterGuiPrimitives(void) {
    gKeepAlive = jaiListNew(0);
    jaiGCAddPermanentRoot(OBJ_VAL(gKeepAlive));

    jaiDefineNative("__prim__.gui_available",    nGuiAvailable,   0, 0);
    jaiDefineNative("__prim__.gui_window_open",  nGuiWindowOpen,  3, 4);
    jaiDefineNative("__prim__.gui_window_close", nGuiWindowClose, 1, 1);
    jaiDefineNative("__prim__.gui_pixels",       nGuiPixels,      1, 1);
    jaiDefineNative("__prim__.gui_present",      nGuiPresent,     1, 1);
    jaiDefineNative("__prim__.gui_poll",         nGuiPoll,        1, 1);
    jaiDefineNative("__prim__.gui_drain_events", nGuiDrainEvents, 1, 1);
    jaiDefineNative("__prim__.gui_delta_time",   nGuiDeltaTime,   1, 1);
    jaiDefineNative("__prim__.gui_fps",          nGuiFps,         1, 1);
    jaiDefineNative("__prim__.gui_mouse_pos",    nGuiMousePos,    1, 1);
    jaiDefineNative("__prim__.gui_mouse_down",   nGuiMouseDown,   2, 2);
    jaiDefineNative("__prim__.gui_key_down",     nGuiKeyDown,     2, 2);
    jaiDefineNative("__prim__.gui_set_title",    nGuiSetTitle,    2, 2);

    /* The translation and its test harness. Documented at each definition
     * above; `gui_test_*` exists for tests/stdlib/test_gui_input.jai and
     * reaches nothing but a scratch window inside the native layer. */
    jaiDefineNative("__prim__.gui_key_from_platform", nGuiKeyFromPlatform, 1, 1);
    jaiDefineNative("__prim__.gui_test_key_event",    nGuiTestKeyEvent,    2, 3);
    jaiDefineNative("__prim__.gui_test_key_state",    nGuiTestKeyState,    0, 0);
}
