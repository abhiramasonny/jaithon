/* gui.m — one window, one CPU-side ARGB back buffer, uploaded every present.
 *
 * Presenting copies the buffer into a BGRA8Unorm texture drawn over two
 * triangles, so resizing scales the image instead of reallocating — a pointer
 * from jaiWindowPixels stays valid for the window's whole life, which
 * std.gui.canvas assumes. Pixels are 0xAARRGGBB; little-endian that's B,G,R,A,
 * exactly MTLPixelFormatBGRA8Unorm, so the upload needs no swizzle.
 *
 * No drawing primitive lives here — Jaithon writes into the buffer directly.
 * AppKit is main-thread-only, so every entry point touching it checks first
 * and degrades to a no-op rather than corrupting state from a worker. */

#ifdef __APPLE__

#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import <QuartzCore/QuartzCore.h>

#include <time.h>

#include "native/native.h"

/* Keycodes leaving this file are USB HID usage ids, not AppKit virtual
 * keycodes; the keyboard page tops out at 231 (Right GUI), so a byte indexes it. */
#define JAI_KEY_COUNT 256

/* AppKit virtual keycodes are 7-bit; nothing on a keyboard reports above 0x7F. */
#define JAI_PLATFORM_KEY_COUNT 128

#define JAI_MOUSE_BUTTONS 3

/* Cocoa keeps no registry of our windows, so events — which arrive from one
 * global queue — are routed by matching NSWindow identity against this table. */
#define JAI_MAX_WINDOWS 16

/* Refuse absurd sizes before multiplying them out. 16384 is past any display. */
#define JAI_MAX_DIMENSION 16384

/* How much of the newest frame rate a reading takes on, so that one stalled
 * frame does not make a displayed counter jump. */
#define JAI_FPS_SMOOTHING 0.1

/* One frame of input at 60 Hz is a handful of events; this only has to absorb
 * the burst a caller who forgot to poll for a second leaves behind. */
#define JAI_EVENT_CAPACITY 256

static const char kBlitShaderSource[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "\n"
    "struct JaiBlitIO {\n"
    "    float4 position [[position]];\n"
    "    float2 uv;\n"
    "};\n"
    "\n"
    "vertex JaiBlitIO jaiBlitVertex(uint id [[vertex_id]]) {\n"
    "    float2 corners[6] = { float2(-1, -1), float2(1, -1), float2(-1, 1),\n"
    "                          float2(-1, 1), float2(1, -1), float2(1, 1) };\n"
    "    float2 uvs[6] = { float2(0, 1), float2(1, 1), float2(0, 0),\n"
    "                      float2(0, 0), float2(1, 1), float2(1, 0) };\n"
    "    JaiBlitIO out;\n"
    "    out.position = float4(corners[id], 0, 1);\n"
    "    out.uv = uvs[id];\n"
    "    return out;\n"
    "}\n"
    "\n"
    "fragment float4 jaiBlitFragment(JaiBlitIO in [[stage_in]],\n"
    "                                texture2d<float> src [[texture(0)]]) {\n"
    "    constexpr sampler nearest(mag_filter::nearest, min_filter::nearest);\n"
    "    return src.sample(nearest, in.uv);\n"
    "}\n";

@class JaiWindowState;

struct JaiWindow {
    /* The Objective-C half, held at +1 through a bridged cast: ARC will not
     * manage object fields of a jaiRealloc'd C struct for us. */
    void     *state;
    uint32_t *pixels;
    int       width;
    int       height;
    double    targetFrame;    /* seconds per frame; 0 means uncapped */
    double    nextFrame;      /* monotonic deadline of the frame being paced */
    double    lastPresent;    /* monotonic seconds, 0 before the first present */
    double    delta;
    double    fps;
    double    mouseX;
    double    mouseY;
    bool      mouse[JAI_MOUSE_BUTTONS];
    bool      keys[JAI_KEY_COUNT];
    /* A key pressed and released between two polls would otherwise be invisible
     * to a frame loop that only ever sees the up state. */
    bool      hits[JAI_KEY_COUNT];
    bool      open;
    /* Discrete events, oldest at `eventHead`. The polled state above cannot
     * express a press and a release in the same frame, an auto-repeat, or
     * composed text, so std.gui reads this queue instead. */
    JaiWindowEvent events[JAI_EVENT_CAPACITY];
    int            eventHead;
    int            eventCount;
};

/* Append one event, dropping the oldest when the queue is full: a caller far
 * enough behind to overflow it wants the recent input, not the stale input. */
static JaiWindowEvent *pushEvent(JaiWindow *w, int tag) {
    if (w == NULL) return NULL;
    int slot;
    if (w->eventCount == JAI_EVENT_CAPACITY) {
        slot = w->eventHead;
        w->eventHead = (w->eventHead + 1) % JAI_EVENT_CAPACITY;
    } else {
        slot = (w->eventHead + w->eventCount) % JAI_EVENT_CAPACITY;
        w->eventCount++;
    }
    JaiWindowEvent *e = &w->events[slot];
    memset(e, 0, sizeof *e);
    e->tag = tag;
    return e;
}

@interface JaiBlitView : MTKView
@property (nonatomic, assign) JaiWindow *owner;
@end

@interface JaiWindowState : NSObject <NSWindowDelegate>
@property (nonatomic, strong) NSWindow                   *window;
@property (nonatomic, strong) JaiBlitView                *view;
@property (nonatomic, strong) id<MTLDevice>               device;
@property (nonatomic, strong) id<MTLCommandQueue>         queue;
@property (nonatomic, strong) id<MTLRenderPipelineState>  pipeline;
@property (nonatomic, strong) id<MTLTexture>              texture;
@property (nonatomic, assign) JaiWindow                  *owner;
@end

@implementation JaiBlitView

- (BOOL)acceptsFirstResponder { return YES; }
- (BOOL)canBecomeKeyView { return YES; }
- (BOOL)isOpaque { return YES; }

/* Swallow key events. The responder chain would otherwise reach NSApp, which
 * beeps at every keystroke it cannot turn into a command. */
- (void)keyDown:(NSEvent *)event { (void)event; }
- (void)keyUp:(NSEvent *)event { (void)event; }

- (void)drawRect:(NSRect)dirty {
    (void)dirty;
    JaiWindow *w = self.owner;
    if (w == NULL || w->state == NULL) return;
    JaiWindowState *s = (__bridge JaiWindowState *)w->state;

    id<CAMetalDrawable> drawable = self.currentDrawable;
    MTLRenderPassDescriptor *pass = self.currentRenderPassDescriptor;
    if (drawable == nil || pass == nil) return;

    id<MTLCommandBuffer> commands = [s.queue commandBuffer];
    id<MTLRenderCommandEncoder> encoder =
        [commands renderCommandEncoderWithDescriptor:pass];
    if (encoder == nil) return;

    [encoder setRenderPipelineState:s.pipeline];
    [encoder setFragmentTexture:s.texture atIndex:0];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
    [encoder endEncoding];
    [commands presentDrawable:drawable];
    [commands commit];
}

@end

@implementation JaiWindowState

- (void)windowWillClose:(NSNotification *)note {
    (void)note;
    if (self.owner == NULL) return;
    /* Queue the close before clearing `open`: a frame loop polls once more
     * after the window goes away and must still see why. */
    (void)pushEvent(self.owner, JAI_EVENT_CLOSE);
    self.owner->open = false;
}

/* The back buffer keeps its size for the window's whole life — presenting
 * scales it — so a resize is reported and nothing is reallocated. */
- (void)windowDidResize:(NSNotification *)note {
    (void)note;
    if (self.owner == NULL || self.view == nil) return;
    NSRect backing = [self.view convertRectToBacking:[self.view bounds]];
    JaiWindowEvent *e = pushEvent(self.owner, JAI_EVENT_RESIZE);
    if (e == NULL) return;
    e->i0 = (int)backing.size.width;
    e->i1 = (int)backing.size.height;
}

- (void)windowDidBecomeKey:(NSNotification *)note {
    (void)note;
    JaiWindowEvent *e = pushEvent(self.owner, JAI_EVENT_FOCUS);
    if (e != NULL) e->i0 = 1;
}

/* Keys held when focus leaves would otherwise stay down forever: the key-up
 * goes to whichever window has focus now, not to this one. */
- (void)windowDidResignKey:(NSNotification *)note {
    (void)note;
    JaiWindow *w = self.owner;
    if (w == NULL) return;
    /* `keys` is indexed by HID usage id, so an index is already the code the
     * synthesised key-up should carry — no translation on the way out. */
    for (int hid = 0; hid < JAI_KEY_COUNT; hid++) {
        if (!w->keys[hid]) continue;
        w->keys[hid] = false;
        JaiWindowEvent *up = pushEvent(w, JAI_EVENT_KEY_UP);
        if (up != NULL) up->i0 = hid;
    }
    JaiWindowEvent *e = pushEvent(w, JAI_EVENT_FOCUS);
    if (e != NULL) e->i0 = 0;
}

@end

static JaiWindow *gWindows[JAI_MAX_WINDOWS];
static int        gWindowCount;

static void registerWindow(JaiWindow *w) {
    if (gWindowCount < JAI_MAX_WINDOWS) gWindows[gWindowCount++] = w;
}

static void unregisterWindow(JaiWindow *w) {
    for (int i = 0; i < gWindowCount; i++) {
        if (gWindows[i] != w) continue;
        gWindows[i] = gWindows[gWindowCount - 1];
        gWindows[--gWindowCount] = NULL;
        return;
    }
}

static JaiWindow *windowForNSWindow(NSWindow *native) {
    if (native == nil) return NULL;
    for (int i = 0; i < gWindowCount; i++) {
        JaiWindow *w = gWindows[i];
        if (w->state == NULL) continue;
        JaiWindowState *s = (__bridge JaiWindowState *)w->state;
        if (s.window == native) return w;
    }
    return NULL;
}

bool jaiGuiAvailable(void) {
    static bool checked = false;
    static bool available = false;
    if (checked) return available;
    checked = true;

    @autoreleasepool {
        if (MTLCreateSystemDefaultDevice() == nil) return false;
        /* A daemon or an ssh session has a Metal device but no window server,
         * where NSScreen reports no displays. */
        available = [[NSScreen screens] count] > 0;
    }
    return available;
}

/* NSApp has to exist and have finished launching before it will vend events. */
static void ensureApplication(void) {
    static bool done = false;
    if (done) return;
    done = true;

    [NSApplication sharedApplication];
    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
    [NSApp finishLaunching];
}

/* Modifier keycodes arrive as a state word, not key events, so they're named
 * here as Carbon's kVK_* constants — spelled out so this file avoids pulling
 * in <Carbon/Carbon.h> for nine integers. They also index the table below, so
 * a wrong value can't make the two disagree; it shows up as a keycode
 * translating to nothing (tests/stdlib/test_gui_input.jai). */
enum {
    JAI_VK_RIGHT_COMMAND = 54,
    JAI_VK_COMMAND       = 55,
    JAI_VK_SHIFT         = 56,
    JAI_VK_CAPS_LOCK     = 57,
    JAI_VK_OPTION        = 58,
    JAI_VK_CONTROL       = 59,
    JAI_VK_RIGHT_SHIFT   = 60,
    JAI_VK_RIGHT_OPTION  = 61,
    JAI_VK_RIGHT_CONTROL = 62,
};

/* AppKit virtual keycode -> USB HID usage id. AppKit numbers keys by 1984
 * Mac keyboard position (A=0, S=1, Escape=53); everything else in this tree
 * speaks USB HID (keyboard/keypad page 0x07), so translation happens once,
 * here, at the boundary.
 *
 * Exact inverse of `_KEY_TABLE` in lib/std/gui/input.jai, checked row-for-row
 * by tests/stdlib/test_gui_input.jai against `Key.code()` so the two can't
 * silently drift — adding a key means a row there, a row here, and a `Key`
 * variant.
 *
 * 0 means this platform key names no `Key` (fn, volume/media, JIS-only,
 * F16-F20); HID usage 0 is reserved for that, and `Key.from_code(0)` is null,
 * so an unmapped key is dropped rather than misreported. */
static const unsigned char kPlatformToHid[JAI_PLATFORM_KEY_COUNT] = {
    [0] = 4,     /* A */
    [11] = 5,    /* B */
    [8] = 6,     /* C */
    [2] = 7,     /* D */
    [14] = 8,    /* E */
    [3] = 9,     /* F */
    [5] = 10,    /* G */
    [4] = 11,    /* H */
    [34] = 12,   /* I */
    [38] = 13,   /* J */
    [40] = 14,   /* K */
    [37] = 15,   /* L */
    [46] = 16,   /* M */
    [45] = 17,   /* N */
    [31] = 18,   /* O */
    [35] = 19,   /* P */
    [12] = 20,   /* Q */
    [15] = 21,   /* R */
    [1] = 22,    /* S */
    [17] = 23,   /* T */
    [32] = 24,   /* U */
    [9] = 25,    /* V */
    [13] = 26,   /* W */
    [7] = 27,    /* X */
    [16] = 28,   /* Y */
    [6] = 29,    /* Z */

    /* HID orders digits 1..9 then 0, and so does `Key`. */
    [18] = 30,   /* 1 */
    [19] = 31,   /* 2 */
    [20] = 32,   /* 3 */
    [21] = 33,   /* 4 */
    [23] = 34,   /* 5 */
    [22] = 35,   /* 6 */
    [26] = 36,   /* 7 */
    [28] = 37,   /* 8 */
    [25] = 38,   /* 9 */
    [29] = 39,   /* 0 */

    /* AppKit's "Delete" is the backspace key; the one labelled Delete on a
     * full keyboard is ForwardDelete, further down. */
    [36] = 40,   /* Return */
    [53] = 41,   /* Escape */
    [51] = 42,   /* Backspace */
    [48] = 43,   /* Tab */
    [49] = 44,   /* Space */
    [27] = 45,   /* Minus */
    [24] = 46,   /* Equals */
    [33] = 47,   /* LeftBracket */
    [30] = 48,   /* RightBracket */
    [42] = 49,   /* Backslash */
    [41] = 51,   /* Semicolon */
    [39] = 52,   /* Quote */
    [50] = 53,   /* Backquote */
    [43] = 54,   /* Comma */
    [47] = 55,   /* Period */
    [44] = 56,   /* Slash */
    [JAI_VK_CAPS_LOCK] = 57,

    /* AppKit's order is not monotonic — F5 through F9 sit below F1 through
     * F4 — so each row is spelled out. */
    [122] = 58,  /* F1 */
    [120] = 59,  /* F2 */
    [99] = 60,   /* F3 */
    [118] = 61,  /* F4 */
    [96] = 62,   /* F5 */
    [97] = 63,   /* F6 */
    [98] = 64,   /* F7 */
    [100] = 65,  /* F8 */
    [101] = 66,  /* F9 */
    [109] = 67,  /* F10 */
    [103] = 68,  /* F11 */
    [111] = 69,  /* F12 */

    /* A Mac keyboard has no PrintScreen, ScrollLock or Pause; an external PC
     * keyboard reports those three positions as F13, F14 and F15, which is the
     * mapping every other Mac port of a cross-platform toolkit uses. */
    [105] = 70,  /* PrintScreen (F13) */
    [107] = 71,  /* ScrollLock (F14) */
    [113] = 72,  /* Pause (F15) */

    /* Insert is the Mac's Help key, in the same position. */
    [114] = 73,  /* Insert (Help) */
    [115] = 74,  /* Home */
    [116] = 75,  /* PageUp */
    [117] = 76,  /* Delete (ForwardDelete) */
    [119] = 77,  /* End */
    [121] = 78,  /* PageDown */
    [124] = 79,  /* Right */
    [123] = 80,  /* Left */
    [125] = 81,  /* Down */
    [126] = 82,  /* Up */

    /* The Mac's Clear key occupies the NumLock position. */
    [71] = 83,   /* NumLock (Clear) */
    [75] = 84,   /* KeypadDivide */
    [67] = 85,   /* KeypadMultiply */
    [78] = 86,   /* KeypadMinus */
    [69] = 87,   /* KeypadPlus */
    [76] = 88,   /* KeypadEnter */
    [83] = 89,   /* Keypad1 */
    [84] = 90,   /* Keypad2 */
    [85] = 91,   /* Keypad3 */
    [86] = 92,   /* Keypad4 */
    [87] = 93,   /* Keypad5 */
    [88] = 94,   /* Keypad6 */
    [89] = 95,   /* Keypad7 */
    [91] = 96,   /* Keypad8 */
    [92] = 97,   /* Keypad9 */
    [82] = 98,   /* Keypad0 */
    [65] = 99,   /* KeypadPeriod */

    [110] = 101, /* Menu */

    /* Left and right are distinct HID ids, and AppKit gives each side its own
     * virtual keycode, so the pairs survive the translation. */
    [JAI_VK_CONTROL] = 224,        /* LeftControl */
    [JAI_VK_SHIFT] = 225,          /* LeftShift */
    [JAI_VK_OPTION] = 226,         /* LeftAlt */
    [JAI_VK_COMMAND] = 227,        /* LeftSuper */
    [JAI_VK_RIGHT_CONTROL] = 228,  /* RightControl */
    [JAI_VK_RIGHT_SHIFT] = 229,    /* RightShift */
    [JAI_VK_RIGHT_OPTION] = 230,   /* RightAlt */
    [JAI_VK_RIGHT_COMMAND] = 231,  /* RightSuper */
};

int jaiWindowKeyFromPlatform(int platformCode) {
    if (platformCode < 0 || platformCode >= JAI_PLATFORM_KEY_COUNT) return 0;
    return kPlatformToHid[platformCode];
}

static void recordMouseMove(JaiWindow *w, JaiWindowState *s, NSEvent *event) {
    NSPoint inWindow = [event locationInWindow];
    NSPoint local = [s.view convertPoint:inWindow fromView:nil];
    NSRect bounds = [s.view bounds];

    /* The back buffer keeps its original size while the view can be resized, so
     * report the position in buffer pixels — the space the caller draws in. */
    double scaleX = bounds.size.width > 0.0 ? w->width / bounds.size.width : 1.0;
    double scaleY = bounds.size.height > 0.0 ? w->height / bounds.size.height : 1.0;
    double x = local.x * scaleX;
    double y = (bounds.size.height - local.y) * scaleY;   /* AppKit y grows up */

    if (x < 0.0) x = 0.0;
    if (y < 0.0) y = 0.0;
    if (x > w->width - 1) x = w->width - 1;
    if (y > w->height - 1) y = w->height - 1;
    w->mouseX = x;
    w->mouseY = y;
}

/* AppKit reports arrow/function keys as private-use scalars through the same
 * string; those are not text a caret should receive. */
static void recordText(JaiWindow *w, NSEvent *event) {
    NSString *characters = [event characters];
    if (characters == nil || [characters length] == 0) return;

    NSMutableString *kept = [NSMutableString stringWithCapacity:[characters length]];
    for (NSUInteger i = 0; i < [characters length]; i++) {
        unichar c = [characters characterAtIndex:i];
        if (c < 0x20 || c == 0x7F) continue;             /* control */
        if (c >= 0xF700 && c <= 0xF8FF) continue;        /* function keys */
        [kept appendFormat:@"%C", c];
    }
    if ([kept length] == 0) return;

    const char *utf8 = [kept UTF8String];
    if (utf8 == NULL) return;
    JaiWindowEvent *e = pushEvent(w, JAI_EVENT_TEXT);
    if (e == NULL) return;
    /* Longer than the buffer is a paste, not a keystroke; truncate on a scalar
     * boundary rather than emitting half a character. */
    size_t length = strlen(utf8);
    if (length >= sizeof e->text) {
        length = sizeof e->text - 1;
        while (length > 0 && ((unsigned char)utf8[length] & 0xC0) == 0x80) length--;
    }
    memcpy(e->text, utf8, length);
    e->text[length] = '\0';
}

/* The only place a keycode is translated, and the only place `keys`/`hits`
 * are written, so the arrays and the event queue never disagree on numbering
 * — all three are HID, since jaiWindowKeyDown and windowDidResignKey read
 * these arrays directly. An unnamed key is dropped here, not stored at index
 * 0, which nothing would ever clear. */
static void recordKey(JaiWindow *w, int platformCode, bool down, bool repeat) {
    int hid = jaiWindowKeyFromPlatform(platformCode);
    if (hid <= 0 || hid >= JAI_KEY_COUNT) return;

    w->keys[hid] = down;
    if (down) w->hits[hid] = true;

    JaiWindowEvent *e = pushEvent(w, down ? JAI_EVENT_KEY_DOWN : JAI_EVENT_KEY_UP);
    if (e == NULL) return;
    e->i0 = hid;
    e->i1 = repeat ? 1 : 0;
}

/* Modifiers arrive as a state word, not up/down pairs, so the edge has to be
 * found in the HID-indexed array, where the state now lives. */
static void recordModifier(JaiWindow *w, int platformCode, bool down) {
    int hid = jaiWindowKeyFromPlatform(platformCode);
    if (hid <= 0 || hid >= JAI_KEY_COUNT) return;
    if (w->keys[hid] == down) return;
    recordKey(w, platformCode, down, false);
}

/* AppKit orders its button events left, right, other; std.gui numbers them
 * left, middle, right, which is what `MouseButton.from_code` decodes. */
static void recordButton(JaiWindow *w, JaiWindowState *s, NSEvent *event,
                         int button, bool down) {
    recordMouseMove(w, s, event);
    if (button >= 0 && button < JAI_MOUSE_BUTTONS) w->mouse[button] = down;
    JaiWindowEvent *e = pushEvent(w, down ? JAI_EVENT_MOUSE_DOWN : JAI_EVENT_MOUSE_UP);
    if (e == NULL) return;
    e->i0 = button;
    e->i1 = (int)w->mouseX;
    e->i2 = (int)w->mouseY;
}

static void recordEvent(JaiWindow *w, JaiWindowState *s, NSEvent *event) {
    switch ([event type]) {
    case NSEventTypeKeyDown:
        recordKey(w, [event keyCode], true, [event isARepeat]);
        /* Outside recordKey on purpose: a key this build does not name can
         * still compose text, and a text field must receive it. */
        recordText(w, event);
        break;
    case NSEventTypeKeyUp:
        recordKey(w, [event keyCode], false, false);
        break;
    case NSEventTypeFlagsChanged: {
        /* Both the left and right keycode of each pair follow that one bit. */
        NSEventModifierFlags flags = [event modifierFlags];
        bool shift = (flags & NSEventModifierFlagShift) != 0;
        bool control = (flags & NSEventModifierFlagControl) != 0;
        bool option = (flags & NSEventModifierFlagOption) != 0;
        bool command = (flags & NSEventModifierFlagCommand) != 0;
        bool capsLock = (flags & NSEventModifierFlagCapsLock) != 0;
        recordModifier(w, JAI_VK_SHIFT, shift);
        recordModifier(w, JAI_VK_RIGHT_SHIFT, shift);
        recordModifier(w, JAI_VK_CONTROL, control);
        recordModifier(w, JAI_VK_RIGHT_CONTROL, control);
        recordModifier(w, JAI_VK_OPTION, option);
        recordModifier(w, JAI_VK_RIGHT_OPTION, option);
        recordModifier(w, JAI_VK_COMMAND, command);
        recordModifier(w, JAI_VK_RIGHT_COMMAND, command);
        recordModifier(w, JAI_VK_CAPS_LOCK, capsLock);
        break;
    }
    /* Every button event also carries a position, and a click on a stale
     * pointer position would report the wrong pixel. */
    case NSEventTypeLeftMouseDown:   recordButton(w, s, event, 0, true);  break;
    case NSEventTypeLeftMouseUp:     recordButton(w, s, event, 0, false); break;
    case NSEventTypeOtherMouseDown:  recordButton(w, s, event, 1, true);  break;
    case NSEventTypeOtherMouseUp:    recordButton(w, s, event, 1, false); break;
    case NSEventTypeRightMouseDown:  recordButton(w, s, event, 2, true);  break;
    case NSEventTypeRightMouseUp:    recordButton(w, s, event, 2, false); break;
    case NSEventTypeMouseMoved:
    case NSEventTypeLeftMouseDragged:
    case NSEventTypeRightMouseDragged:
    case NSEventTypeOtherMouseDragged: {
        recordMouseMove(w, s, event);
        JaiWindowEvent *e = pushEvent(w, JAI_EVENT_MOUSE_MOVE);
        if (e != NULL) {
            e->i0 = (int)w->mouseX;
            e->i1 = (int)w->mouseY;
        }
        break;
    }
    case NSEventTypeScrollWheel: {
        JaiWindowEvent *e = pushEvent(w, JAI_EVENT_WHEEL);
        if (e == NULL) break;
        /* deltaX/deltaY are line-based for both a wheel and a trackpad, which
         * is the unit std.gui.Event.Wheel documents. */
        e->d0 = [event deltaX];
        e->d1 = [event deltaY];
        break;
    }
    default:
        break;
    }
}

/* Runs for every open window at once: Cocoa has one event queue for the
 * process, not one per window. */
static void pumpEvents(void) {
    if (gWindowCount == 0) return;
    if (![NSThread isMainThread]) return;

    @autoreleasepool {
        NSEvent *event = nil;
        while ((event = [NSApp nextEventMatchingMask:NSEventMaskAny
                                           untilDate:[NSDate distantPast]
                                              inMode:NSDefaultRunLoopMode
                                             dequeue:YES]) != nil) {
            JaiWindow *target = windowForNSWindow([event window]);
            if (target != NULL && target->state != NULL) {
                recordEvent(target, (__bridge JaiWindowState *)target->state, event);
            }
            [NSApp sendEvent:event];
        }
        [NSApp updateWindows];

        /* windowWillClose: covers the close button, but a window can also be
         * ordered out from under us; either way it is no longer presentable. */
        for (int i = 0; i < gWindowCount; i++) {
            JaiWindow *w = gWindows[i];
            if (w->state == NULL) continue;
            JaiWindowState *s = (__bridge JaiWindowState *)w->state;
            if (![s.window isVisible]) w->open = false;
        }
    }
}

/* Everything the window owns, released in the order it was built. Safe to call
 * on a half-built window: every step checks. */
static void destroyWindow(JaiWindow *w) {
    unregisterWindow(w);

    @autoreleasepool {
        JaiWindowState *s = (__bridge_transfer JaiWindowState *)w->state;
        w->state = NULL;
        if (s != nil) {
            /* Break both back pointers first: closing the window fires
             * windowWillClose: on the delegate, which would otherwise write
             * through a pointer this function is about to free. */
            s.owner = NULL;
            s.view.owner = NULL;
            [s.window setDelegate:nil];
            [s.window orderOut:nil];
            [s.window close];
            s.window = nil;
            s.view = nil;
            s.texture = nil;
            s.pipeline = nil;
            s.queue = nil;
            s.device = nil;
        }
    }

    if (w->pixels != NULL) {
        JAI_FREE_ARRAY(uint32_t, w->pixels, (size_t)w->width * (size_t)w->height);
        w->pixels = NULL;
    }
    JAI_FREE(JaiWindow, w);
}

JaiWindow *jaiWindowOpen(int width, int height, const char *title, int targetFPS) {
    if (width <= 0 || height <= 0) return NULL;
    if (width > JAI_MAX_DIMENSION || height > JAI_MAX_DIMENSION) return NULL;
    if (!jaiGuiAvailable()) return NULL;
    /* AppKit forbids building a window off the main thread, and a caller that
     * tries has a bug that a crash deep inside Cocoa would not explain. */
    if (![NSThread isMainThread]) return NULL;
    if (gWindowCount >= JAI_MAX_WINDOWS) return NULL;

    JaiWindow *w = JAI_ALLOC_ZEROED(JaiWindow, 1);
    w->width = width;
    w->height = height;
    w->targetFrame = targetFPS > 0 ? 1.0 / (double)targetFPS : 0.0;

    size_t count = (size_t)width * (size_t)height;
    w->pixels = JAI_ALLOC(uint32_t, count);
    for (size_t i = 0; i < count; i++) w->pixels[i] = 0xFF000000u;  /* opaque black */

    @autoreleasepool {
        JaiWindowState *s = [[JaiWindowState alloc] init];
        s.owner = w;
        w->state = (__bridge_retained void *)s;

        s.device = MTLCreateSystemDefaultDevice();
        if (s.device == nil) { destroyWindow(w); return NULL; }

        s.queue = [s.device newCommandQueue];
        if (s.queue == nil) { destroyWindow(w); return NULL; }

        NSError *error = nil;
        id<MTLLibrary> library =
            [s.device newLibraryWithSource:@(kBlitShaderSource) options:nil error:&error];
        if (library == nil) { destroyWindow(w); return NULL; }

        id<MTLFunction> vertexFn = [library newFunctionWithName:@"jaiBlitVertex"];
        id<MTLFunction> fragmentFn = [library newFunctionWithName:@"jaiBlitFragment"];
        if (vertexFn == nil || fragmentFn == nil) { destroyWindow(w); return NULL; }

        MTLRenderPipelineDescriptor *pipeDesc = [[MTLRenderPipelineDescriptor alloc] init];
        pipeDesc.vertexFunction = vertexFn;
        pipeDesc.fragmentFunction = fragmentFn;
        pipeDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
        s.pipeline = [s.device newRenderPipelineStateWithDescriptor:pipeDesc error:&error];
        if (s.pipeline == nil) { destroyWindow(w); return NULL; }

        MTLTextureDescriptor *texDesc = [[MTLTextureDescriptor alloc] init];
        texDesc.pixelFormat = MTLPixelFormatBGRA8Unorm;
        texDesc.width = (NSUInteger)width;
        texDesc.height = (NSUInteger)height;
        texDesc.usage = MTLTextureUsageShaderRead;
        s.texture = [s.device newTextureWithDescriptor:texDesc];
        if (s.texture == nil) { destroyWindow(w); return NULL; }

        ensureApplication();

        NSRect frame = NSMakeRect(0, 0, width, height);
        NSUInteger style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                           NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable;
        s.window = [[NSWindow alloc] initWithContentRect:frame
                                               styleMask:style
                                                 backing:NSBackingStoreBuffered
                                                   defer:NO];
        if (s.window == nil) { destroyWindow(w); return NULL; }

        /* ARC owns the window through `s`; letting Cocoa release it on close
         * would leave `s.window` dangling. */
        [s.window setReleasedWhenClosed:NO];
        [s.window setAcceptsMouseMovedEvents:YES];
        [s.window setDelegate:s];
        [s.window center];

        NSString *name = title != NULL ? [NSString stringWithUTF8String:title] : nil;
        [s.window setTitle:name != nil ? name : @"jaithon"];

        s.view = [[JaiBlitView alloc] initWithFrame:frame device:s.device];
        s.view.owner = w;
        s.view.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
        s.view.framebufferOnly = YES;
        /* The frame loop drives drawing; the view must not also run a display
         * link, or it would present frames the caller has not finished. */
        s.view.paused = YES;
        s.view.enableSetNeedsDisplay = NO;
        /* Resizing then scales rather than reallocating, which is what keeps
         * jaiWindowPixels() stable. */
        s.view.autoResizeDrawable = NO;
        s.view.drawableSize = CGSizeMake(width, height);
        s.view.layer.magnificationFilter = kCAFilterNearest;
        s.view.layer.minificationFilter = kCAFilterNearest;

        [s.window setContentView:s.view];
        [s.window makeKeyAndOrderFront:nil];
        [s.window makeFirstResponder:s.view];
        [NSApp activateIgnoringOtherApps:YES];
    }

    w->open = true;
    w->lastPresent = jaiClockMonotonic();
    registerWindow(w);
    return w;
}

void jaiWindowClose(JaiWindow *w) {
    if (w == NULL) return;
    w->open = false;
    destroyWindow(w);
}

uint32_t *jaiWindowPixels(JaiWindow *w, int *outWidth, int *outHeight) {
    if (w == NULL) {
        if (outWidth != NULL) *outWidth = 0;
        if (outHeight != NULL) *outHeight = 0;
        return NULL;
    }
    if (outWidth != NULL) *outWidth = w->width;
    if (outHeight != NULL) *outHeight = w->height;
    return w->pixels;
}

static void sleepSeconds(double seconds) {
    if (seconds <= 0.0) return;
    struct timespec ts;
    ts.tv_sec = (time_t)seconds;
    ts.tv_nsec = (long)((seconds - (double)ts.tv_sec) * 1e9);
    if (ts.tv_nsec < 0) ts.tv_nsec = 0;
    if (ts.tv_nsec > 999999999L) ts.tv_nsec = 999999999L;
    /* A signal cutting the wait short only costs one early frame. */
    (void)nanosleep(&ts, NULL);
}

void jaiWindowPresent(JaiWindow *w) {
    if (w == NULL || !w->open || w->state == NULL) return;

    @autoreleasepool {
        JaiWindowState *s = (__bridge JaiWindowState *)w->state;
        MTLRegion region = MTLRegionMake2D(0, 0, (NSUInteger)w->width, (NSUInteger)w->height);
        [s.texture replaceRegion:region
                     mipmapLevel:0
                       withBytes:w->pixels
                     bytesPerRow:(NSUInteger)w->width * 4];

        if ([NSThread isMainThread]) {
            [s.view draw];
            pumpEvents();
        }
    }

    double now = jaiClockMonotonic();
    if (w->targetFrame > 0.0) {
        /* Paces against a fixed grid of deadlines, not the last present:
         * nanosleep overshoots, and measuring from the previous wait would let that error accumulate well below the target rate. */
        if (w->nextFrame == 0.0) w->nextFrame = w->lastPresent + w->targetFrame;
        if (now < w->nextFrame) {
            sleepSeconds(w->nextFrame - now);
            now = jaiClockMonotonic();
        }
        w->nextFrame += w->targetFrame;
        /* A frame slow enough to miss its whole slot resets the grid; catching
         * up by running the backlog at full speed would only stutter. */
        if (w->nextFrame < now) w->nextFrame = now + w->targetFrame;
    }

    double elapsed = now - w->lastPresent;
    if (elapsed < 0.0) elapsed = 0.0;   /* a clock that stepped backwards */

    w->delta = elapsed;
    w->lastPresent = now;

    double instant = elapsed > 0.0 ? 1.0 / elapsed : 0.0;
    w->fps = w->fps > 0.0 ? w->fps + JAI_FPS_SMOOTHING * (instant - w->fps) : instant;
}

bool jaiWindowPoll(JaiWindow *w) {
    if (w == NULL) return false;
    /* Taps are reported for exactly the frame they happened in, so the flags
     * are cleared here — at the frame boundary — and not by the pumps that
     * jaiWindowPresent runs. */
    memset(w->hits, 0, sizeof(w->hits));
    pumpEvents();
    return w->open;
}

double jaiWindowDeltaTime(JaiWindow *w) {
    return w != NULL ? w->delta : 0.0;
}

double jaiWindowFPS(JaiWindow *w) {
    return w != NULL ? w->fps : 0.0;
}

void jaiWindowMousePos(JaiWindow *w, double *x, double *y) {
    if (x != NULL) *x = w != NULL ? w->mouseX : 0.0;
    if (y != NULL) *y = w != NULL ? w->mouseY : 0.0;
}

bool jaiWindowMouseDown(JaiWindow *w, int button) {
    if (w == NULL || button < 0 || button >= JAI_MOUSE_BUTTONS) return false;
    return w->mouse[button];
}

/* `hidCode` is a USB HID usage id — `Key.code()` in std.gui.input — not an
 * AppKit virtual keycode. */
bool jaiWindowKeyDown(JaiWindow *w, int hidCode) {
    if (w == NULL || hidCode < 0 || hidCode >= JAI_KEY_COUNT) return false;
    return w->keys[hidCode] || w->hits[hidCode];
}

/* A GUI program can't be driven headlessly, so these test-only entry points
 * exist to let the key-path translation run without a display. They touch
 * only a scratch window with no NSWindow, no Metal device and no registration
 * in gWindows, so nothing pumps or draws it and no production path can reach it. */
static JaiWindow *gTestWindow;

JaiWindow *jaiWindowTestWindow(void) {
    if (gTestWindow == NULL) {
        gTestWindow = JAI_ALLOC_ZEROED(JaiWindow, 1);
        /* A size, so the struct is coherent; there is no buffer to go with it
         * and no entry point that reads one off this window. */
        gTestWindow->width = 1;
        gTestWindow->height = 1;
        gTestWindow->open = true;
    }
    return gTestWindow;
}

/* One call is one frame: tap flags are cleared first, exactly as
 * jaiWindowPoll does at a frame boundary, so a sequence of calls reads as a sequence of frames. */
void jaiWindowTestInjectKey(int platformCode, bool down, bool repeat) {
    JaiWindow *w = jaiWindowTestWindow();
    memset(w->hits, 0, sizeof w->hits);
    recordKey(w, platformCode, down, repeat);
}

void jaiWindowSetTitle(JaiWindow *w, const char *title) {
    if (w == NULL || w->state == NULL || title == NULL) return;
    if (![NSThread isMainThread]) return;

    @autoreleasepool {
        JaiWindowState *s = (__bridge JaiWindowState *)w->state;
        /* stringWithUTF8String: returns nil for invalid UTF-8; keep the old
         * title rather than raising inside AppKit. */
        NSString *name = [NSString stringWithUTF8String:title];
        if (name != nil) [s.window setTitle:name];
    }
}

int jaiWindowDrainEvents(JaiWindow *w, JaiWindowEvent *out, int max) {
    if (w == NULL || out == NULL || max <= 0) return 0;

    int taken = w->eventCount < max ? w->eventCount : max;
    for (int i = 0; i < taken; i++) {
        out[i] = w->events[(w->eventHead + i) % JAI_EVENT_CAPACITY];
    }
    w->eventHead = (w->eventHead + taken) % JAI_EVENT_CAPACITY;
    w->eventCount -= taken;
    return taken;
}

#endif /* __APPLE__ */
