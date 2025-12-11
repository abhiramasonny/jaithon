#import <Cocoa/Cocoa.h>
#include "runtime.h"

@interface JaiView : NSView
@property (nonatomic, assign) unsigned char* pixelData;
@property (nonatomic, assign) int pixelWidth;
@property (nonatomic, assign) int pixelHeight;
@end

@implementation JaiView

- (void)drawRect:(NSRect)dirtyRect {
    [super drawRect:dirtyRect];
    
    if (!self.pixelData) return;
    
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(
        self.pixelData,
        self.pixelWidth,
        self.pixelHeight,
        8,
        self.pixelWidth * 4,
        colorSpace,
        (CGBitmapInfo)kCGImageAlphaPremultipliedLast
    );
    
    if (ctx) {
        CGImageRef img = CGBitmapContextCreateImage(ctx);
        if (img) {
            NSGraphicsContext* nsCtx = [NSGraphicsContext currentContext];
            CGContextRef drawCtx = [nsCtx CGContext];
            
            CGContextTranslateCTM(drawCtx, 0, self.bounds.size.height);//coords
            CGContextScaleCTM(drawCtx, 1.0, -1.0);
            
            CGContextDrawImage(drawCtx, CGRectMake(0, 0, self.pixelWidth, self.pixelHeight), img);
            CGImageRelease(img);
        }
        CGContextRelease(ctx);
    }
    CGColorSpaceRelease(colorSpace);
}

- (BOOL)acceptsFirstResponder {
    return YES;
}

@end

@interface JaiWindowDelegate : NSObject <NSWindowDelegate>
@end

@implementation JaiWindowDelegate
- (void)windowWillClose:(NSNotification *)notification {
    (void)notification;
}
@end

static NSWindow* gWindow = nil;
static JaiView* gView = nil;
static JaiWindowDelegate* gDelegate = nil;
static unsigned char* gPixelBuffer = NULL;
static int gWidth = 0;
static int gHeight = 0;
static bool gWindowOpen = false;

Value native_gui_init(Value* args, int argc) {
    if (argc != 3) {
        runtimeError("gui_init expects 3 arguments: width, height, title");
        return makeNull();
    }
    
    if (args[0].type != VAL_NUMBER || args[1].type != VAL_NUMBER || args[2].type != VAL_STRING) {
        runtimeError("gui_init arguments must be (number, number, string)");
        return makeNull();
    }
    
    gWidth = (int)args[0].as.number;
    gHeight = (int)args[1].as.number;
    char* title = args[2].as.string;
    
    gPixelBuffer = (unsigned char*)calloc(gWidth * gHeight * 4, 1);//rgba
    if (!gPixelBuffer) {
        runtimeError("Failed to allocate pixel buffer");
        return makeNull();
    }
    
    @autoreleasepool {
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
        
        NSRect frame = NSMakeRect(0, 0, gWidth, gHeight);
        NSUInteger style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable;
        
        gWindow = [[NSWindow alloc] initWithContentRect:frame
                                              styleMask:style
                                                backing:NSBackingStoreBuffered
                                                  defer:NO];
        
        [gWindow setTitle:[NSString stringWithUTF8String:title]];
        [gWindow setReleasedWhenClosed:NO];
        [gWindow center];
        
        gDelegate = [[JaiWindowDelegate alloc] init];
        [gWindow setDelegate:gDelegate];
        
        gView = [[JaiView alloc] initWithFrame:frame];
        gView.pixelData = gPixelBuffer;
        gView.pixelWidth = gWidth;
        gView.pixelHeight = gHeight;
        [gWindow setContentView:gView];
        
        [gWindow makeKeyAndOrderFront:nil];
        [NSApp activateIgnoringOtherApps:YES];
        
        gWindowOpen = true;
    }
    
    return makeBool(true);
}

Value native_gui_update(Value* args, int argc) {
    if (argc != 1) {
        runtimeError("gui_update expects 1 argument: pixel_array");
        return makeNull();
    }
    
    if (args[0].type != VAL_ARRAY) {
        runtimeError("gui_update argument must be an array");
        return makeNull();
    }
    
    if (!gWindowOpen || !gWindow || !gPixelBuffer) {
        return makeBool(false);
    }
    
    @autoreleasepool {
        JaiArray* arr = args[0].as.array;
        int len = arr->length;
        if (len > gWidth * gHeight) len = gWidth * gHeight;
        
        for (int i = 0; i < len; i++) {
            Value v = arr->items[i];
            uint32_t color = 0;
            if (v.type == VAL_NUMBER) {
                color = (uint32_t)v.as.number;
            }
            
            int r = (color >> 16) & 0xFF;
            int g = (color >> 8) & 0xFF;
            int b = (color) & 0xFF;
            
            int idx = i * 4;
            gPixelBuffer[idx] = r;
            gPixelBuffer[idx + 1] = g;
            gPixelBuffer[idx + 2] = b;
            gPixelBuffer[idx + 3] = 255;
        }
        
        [gView setNeedsDisplay:YES];
        NSEvent* event;
        while ((event = [NSApp nextEventMatchingMask:NSEventMaskAny
                                           untilDate:[NSDate distantPast]
                                              inMode:NSDefaultRunLoopMode
                                             dequeue:YES])) {
            [NSApp sendEvent:event];
            [NSApp updateWindows];
        }
        
        // Check if window is still visible
        if (![gWindow isVisible]) {
            gWindowOpen = false;
        }
    }
    
    return makeBool(gWindowOpen);
}

Value native_gui_clear(Value* args, int argc) {
    if (argc != 1) {
        runtimeError("gui_clear expects 1 argument: color");
        return makeNull();
    }
    
    if (!gPixelBuffer || !gWindowOpen) {
        return makeBool(false);
    }
    
    uint32_t color = 0;
    if (args[0].type == VAL_NUMBER) {
        color = (uint32_t)args[0].as.number;
    }
    
    int r = (color >> 16) & 0xFF;
    int g = (color >> 8) & 0xFF;
    int b = (color) & 0xFF;
    
    int total = gWidth * gHeight;
    for (int i = 0; i < total; i++) {
        int idx = i * 4;
        gPixelBuffer[idx] = r;
        gPixelBuffer[idx + 1] = g;
        gPixelBuffer[idx + 2] = b;
        gPixelBuffer[idx + 3] = 255;
    }
    
    return makeBool(true);
}

Value native_gui_set_pixel(Value* args, int argc) {
    if (argc != 3) {
        runtimeError("gui_set_pixel expects 3 arguments: x, y, color");
        return makeNull();
    }
    
    if (!gPixelBuffer || !gWindowOpen) {
        return makeBool(false);
    }
    
    int x = (int)args[0].as.number;
    int y = (int)args[1].as.number;
    uint32_t color = (uint32_t)args[2].as.number;
    
    if (x < 0 || x >= gWidth || y < 0 || y >= gHeight) {
        return makeBool(false);
    }
    
    int r = (color >> 16) & 0xFF;
    int g = (color >> 8) & 0xFF;
    int b = (color) & 0xFF;
    
    int idx = (y * gWidth + x) * 4;
    gPixelBuffer[idx] = r;
    gPixelBuffer[idx + 1] = g;
    gPixelBuffer[idx + 2] = b;
    gPixelBuffer[idx + 3] = 255;
    
    return makeBool(true);
}

Value native_gui_fill_rect(Value* args, int argc) {
    if (argc != 5) {
        runtimeError("gui_fill_rect expects 5 arguments: x, y, w, h, color");
        return makeNull();
    }
    
    if (!gPixelBuffer || !gWindowOpen) {
        return makeBool(false);
    }
    
    int rx = (int)args[0].as.number;
    int ry = (int)args[1].as.number;
    int rw = (int)args[2].as.number;
    int rh = (int)args[3].as.number;
    uint32_t color = (uint32_t)args[4].as.number;
    
    int r = (color >> 16) & 0xFF;
    int g = (color >> 8) & 0xFF;
    int b = (color) & 0xFF;
    
    int x0 = rx < 0 ? 0 : rx;
    int y0 = ry < 0 ? 0 : ry;
    int x1 = (rx + rw) > gWidth ? gWidth : (rx + rw);
    int y1 = (ry + rh) > gHeight ? gHeight : (ry + rh);
    
    for (int y = y0; y < y1; y++) {
        for (int x = x0; x < x1; x++) {
            int idx = (y * gWidth + x) * 4;
            gPixelBuffer[idx] = r;
            gPixelBuffer[idx + 1] = g;
            gPixelBuffer[idx + 2] = b;
            gPixelBuffer[idx + 3] = 255;
        }
    }
    
    return makeBool(true);
}

Value native_gui_draw_line(Value* args, int argc) {
    if (argc != 5) {
        runtimeError("gui_draw_line expects 5 arguments: x0, y0, x1, y1, color");
        return makeNull();
    }
    
    if (!gPixelBuffer || !gWindowOpen) {
        return makeBool(false);
    }
    
    int x0 = (int)args[0].as.number;
    int y0 = (int)args[1].as.number;
    int x1 = (int)args[2].as.number;
    int y1 = (int)args[3].as.number;
    uint32_t color = (uint32_t)args[4].as.number;
    
    int r = (color >> 16) & 0xFF;
    int g = (color >> 8) & 0xFF;
    int b = (color) & 0xFF;
    
    int dx = abs(x1 - x0);
    int dy = -abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1;
    int sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    
    while (1) {
        if (x0 >= 0 && x0 < gWidth && y0 >= 0 && y0 < gHeight) {
            int idx = (y0 * gWidth + x0) * 4;
            gPixelBuffer[idx] = r;
            gPixelBuffer[idx + 1] = g;
            gPixelBuffer[idx + 2] = b;
            gPixelBuffer[idx + 3] = 255;
        }
        
        if (x0 == x1 && y0 == y1) break;
        
        int e2 = 2 * err;
        if (e2 >= dy) {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y0 += sy;
        }
    }
    
    return makeBool(true);
}

Value native_gui_draw_circle(Value* args, int argc) {
    if (argc != 4) {
        runtimeError("gui_draw_circle expects 4 arguments: xc, yc, radius, color");
        return makeNull();
    }
    
    if (!gPixelBuffer || !gWindowOpen) {
        return makeBool(false);
    }
    
    int xc = (int)args[0].as.number;
    int yc = (int)args[1].as.number;
    int radius = (int)args[2].as.number;
    uint32_t color = (uint32_t)args[3].as.number;
    
    int r = (color >> 16) & 0xFF;
    int g = (color >> 8) & 0xFF;
    int b = (color) & 0xFF;
    
    #define SET_PIXEL(px, py) do { \
        if ((px) >= 0 && (px) < gWidth && (py) >= 0 && (py) < gHeight) { \
            int idx = ((py) * gWidth + (px)) * 4; \
            gPixelBuffer[idx] = r; \
            gPixelBuffer[idx + 1] = g; \
            gPixelBuffer[idx + 2] = b; \
            gPixelBuffer[idx + 3] = 255; \
        } \
    } while(0)
    
    int x = 0;
    int y = radius;
    int d = 3 - 2 * radius;
    
    while (y >= x) {
        SET_PIXEL(xc + x, yc + y);
        SET_PIXEL(xc - x, yc + y);
        SET_PIXEL(xc + x, yc - y);
        SET_PIXEL(xc - x, yc - y);
        SET_PIXEL(xc + y, yc + x);
        SET_PIXEL(xc - y, yc + x);
        SET_PIXEL(xc + y, yc - x);
        SET_PIXEL(xc - y, yc - x);
        
        x++;
        if (d > 0) {
            y--;
            d = d + 4 * (x - y) + 10;
        } else {
            d = d + 4 * x + 6;
        }
    }
    
    #undef SET_PIXEL
    
    return makeBool(true);
}

Value native_gui_present(Value* args, int argc) {
    (void)args;
    (void)argc;
    
    if (!gWindowOpen || !gWindow) {
        return makeBool(false);
    }
    
    @autoreleasepool {
        [gView setNeedsDisplay:YES];
        
        NSEvent* event;
        while ((event = [NSApp nextEventMatchingMask:NSEventMaskAny
                                           untilDate:[NSDate distantPast]
                                              inMode:NSDefaultRunLoopMode
                                             dequeue:YES])) {
            [NSApp sendEvent:event];
            [NSApp updateWindows];
        }
        
        if (![gWindow isVisible]) {
            gWindowOpen = false;
        }
    }
    
    return makeBool(gWindowOpen);
}

void registerGuiFunctions(void) {
    setVariable("gui_init", makeNativeFunc(native_gui_init));
    setVariable("gui_update", makeNativeFunc(native_gui_update));
    setVariable("gui_clear", makeNativeFunc(native_gui_clear));
    setVariable("gui_set_pixel", makeNativeFunc(native_gui_set_pixel));
    setVariable("gui_fill_rect", makeNativeFunc(native_gui_fill_rect));
    setVariable("gui_draw_line", makeNativeFunc(native_gui_draw_line));
    setVariable("gui_draw_circle", makeNativeFunc(native_gui_draw_circle));
    setVariable("gui_present", makeNativeFunc(native_gui_present));
}
