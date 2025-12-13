#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#import <ImageIO/ImageIO.h>
#include "runtime.h"
#include <mach/mach_time.h>

/*
 * C/Objective-C
 *   - init: Create window with Metal layer
 *   - set_pixel: Write to CPU buffer
 *   - present: Upload buffer to GPU texture, render with vsync
 *   - get_width/get_height: Get dimensions
 *   - get_dt: Get delta time since last frame
 *   - get_fps: Get current FPS
 * 
 * Drawing algorithms are in Jaithon (lib/modules/gui/window.jai)
 */

static NSString* shaderSource = @R"(
#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float2 texCoord;
};

vertex VertexOut vertexShader(uint vertexID [[vertex_id]]) {
    float2 positions[6] = {
        float2(-1, -1), float2(1, -1), float2(-1, 1),
        float2(-1, 1), float2(1, -1), float2(1, 1)
    };
    float2 texCoords[6] = {
        float2(0, 1), float2(1, 1), float2(0, 0),
        float2(0, 0), float2(1, 1), float2(1, 0)
    };
    
    VertexOut out;
    out.position = float4(positions[vertexID], 0, 1);
    out.texCoord = texCoords[vertexID];
    return out;
}

fragment float4 fragmentShader(VertexOut in [[stage_in]],
                               texture2d<float> tex [[texture(0)]]) {
    constexpr sampler s(mag_filter::nearest, min_filter::nearest);
    return tex.sample(s, in.texCoord);
}
)";

@interface MetalView : NSView
@property (nonatomic, strong) CAMetalLayer* metalLayer;
@end

@implementation MetalView
- (CALayer*)makeBackingLayer {
    return [CAMetalLayer layer];
}
- (BOOL)wantsUpdateLayer { return YES; }
- (BOOL)acceptsFirstResponder { return YES; }
- (BOOL)canBecomeKeyView { return YES; }
- (void)keyDown:(NSEvent *)event {
    
    (void)event;
}
- (void)keyUp:(NSEvent *)event {
    (void)event;
}
@end

@interface JaiWindowDelegate : NSObject <NSWindowDelegate>
@end

@implementation JaiWindowDelegate
- (void)windowWillClose:(NSNotification *)notification {
    (void)notification;
}
@end

static id<MTLDevice> gDevice = nil;
static id<MTLCommandQueue> gCommandQueue = nil;
static id<MTLRenderPipelineState> gPipelineState = nil;
static id<MTLTexture> gTexture = nil;
static CAMetalLayer* gMetalLayer = nil;

static NSWindow* gWindow = nil;
static MetalView* gView = nil;
static JaiWindowDelegate* gDelegate = nil;
static unsigned char* gPixelBuffer = NULL;
static int gWidth = 0;
static int gHeight = 0;
static bool gWindowOpen = false;
static bool gKeyState[256] = {0};
static bool gKeyHit[256] = {0}; 
static bool gMouseState[3] = {0};
static int gMouseX = 0;
static int gMouseY = 0;

static uint64_t gLastFrameTime = 0;
static double gDeltaTime = 0.0;
static double gFPS = 0.0;
static mach_timebase_info_data_t gTimebaseInfo;
static double gTargetFrameTime = 1.0 / 60.0;
static int gTargetFPS = 60;


static bool isNumeric(Value v) {
    return v.type == VAL_NUMBER || v.type == VAL_DOUBLE || v.type == VAL_FLOAT ||
           v.type == VAL_INT || v.type == VAL_LONG || v.type == VAL_SHORT ||
           v.type == VAL_BYTE || v.type == VAL_CHAR;
}


static double toNum(Value v) {
    switch (v.type) {
        case VAL_NUMBER: return v.as.number;
        case VAL_DOUBLE: return v.as.f64;
        case VAL_FLOAT: return (double)v.as.f32;
        case VAL_INT: return (double)v.as.i32;
        case VAL_LONG: return (double)v.as.i64;
        case VAL_SHORT: return (double)v.as.i16;
        case VAL_BYTE: return (double)v.as.i8;
        case VAL_CHAR: return (double)(unsigned char)v.as.ch;
        case VAL_BOOL: return v.as.boolean ? 1.0 : 0.0;
        default: return 0.0;
    }
}

Value native_gui_init(Value* args, int argc) {
    if (argc < 3) {
        runtimeError("gui_init expects at least 3 arguments: width, height, title [, targetFPS]");
        return makeNull();
    }
    
    if (!isNumeric(args[0]) || !isNumeric(args[1]) || args[2].type != VAL_STRING) {
        runtimeError("gui_init arguments must be (number, number, string)");
        return makeNull();
    }
    
    gWidth = (int)toNum(args[0]);
    gHeight = (int)toNum(args[1]);
    char* title = args[2].as.string;
    
    if (argc >= 4 && isNumeric(args[3])) {
        gTargetFPS = (int)toNum(args[3]);
        if (gTargetFPS > 0) {
            gTargetFrameTime = 1.0 / gTargetFPS;
        }
    }
    
    mach_timebase_info(&gTimebaseInfo);
    gLastFrameTime = mach_absolute_time();
    
    gPixelBuffer = (unsigned char*)calloc(gWidth * gHeight * 4, 1);
    if (!gPixelBuffer) {
        runtimeError("Failed to allocate pixel buffer");
        return makeNull();
    }
    
    @autoreleasepool {
        gDevice = MTLCreateSystemDefaultDevice();
        if (!gDevice) {
            runtimeError("Metal is not supported on this device");
            free(gPixelBuffer);
            return makeNull();
        }
        
        gCommandQueue = [gDevice newCommandQueue];
        
        NSError* error = nil;
        id<MTLLibrary> library = [gDevice newLibraryWithSource:shaderSource options:nil error:&error];
        if (!library) {
            runtimeError("Failed to compile Metal shaders");
            free(gPixelBuffer);
            return makeNull();
        }
        
        id<MTLFunction> vertexFunc = [library newFunctionWithName:@"vertexShader"];
        id<MTLFunction> fragmentFunc = [library newFunctionWithName:@"fragmentShader"];
        
        MTLRenderPipelineDescriptor* pipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
        pipelineDesc.vertexFunction = vertexFunc;
        pipelineDesc.fragmentFunction = fragmentFunc;
        pipelineDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
        
        gPipelineState = [gDevice newRenderPipelineStateWithDescriptor:pipelineDesc error:&error];
        if (!gPipelineState) {
            runtimeError("Failed to create Metal pipeline");
            free(gPixelBuffer);
            return makeNull();
        }
        
        MTLTextureDescriptor* texDesc = [[MTLTextureDescriptor alloc] init];
        texDesc.pixelFormat = MTLPixelFormatRGBA8Unorm;
        texDesc.width = gWidth;
        texDesc.height = gHeight;
        texDesc.usage = MTLTextureUsageShaderRead;
        gTexture = [gDevice newTextureWithDescriptor:texDesc];
        
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
        [NSApp finishLaunching];
        
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
        
        gView = [[MetalView alloc] initWithFrame:frame];
        [gView setWantsLayer:YES];
        gMetalLayer = (CAMetalLayer*)[gView layer];
        gMetalLayer.device = gDevice;
        gMetalLayer.pixelFormat = MTLPixelFormatBGRA8Unorm;
        gMetalLayer.framebufferOnly = YES;
        gMetalLayer.drawableSize = CGSizeMake(gWidth, gHeight);
        gMetalLayer.displaySyncEnabled = YES; 
        
        [gWindow setContentView:gView];
        [gWindow makeKeyAndOrderFront:nil];
        [gWindow makeFirstResponder:gView];  
        [NSApp activateIgnoringOtherApps:YES];
        
        gWindowOpen = true;
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
    
    int x = (int)toNum(args[0]);
    int y = (int)toNum(args[1]);
    uint32_t color = (uint32_t)toNum(args[2]);
    
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

Value native_gui_clear(Value* args, int argc) {
    if (argc != 1) {
        runtimeError("gui_clear expects 1 argument: color");
        return makeNull();
    }
    
    if (!gPixelBuffer || !gWindowOpen) {
        return makeBool(false);
    }
    
    uint32_t color = (uint32_t)toNum(args[0]);
    int r = (color >> 16) & 0xFF;
    int g = (color >> 8) & 0xFF;
    int b = (color) & 0xFF;
    
    uint32_t pixel = (255 << 24) | (b << 16) | (g << 8) | r;
    uint32_t* buf32 = (uint32_t*)gPixelBuffer;
    int total = gWidth * gHeight;
    
    for (int i = 0; i < total; i++) {
        buf32[i] = pixel;
    }
    
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
    
    int rx = (int)toNum(args[0]);
    int ry = (int)toNum(args[1]);
    int rw = (int)toNum(args[2]);
    int rh = (int)toNum(args[3]);
    uint32_t color = (uint32_t)toNum(args[4]);
    
    int r = (color >> 16) & 0xFF;
    int g = (color >> 8) & 0xFF;
    int b = (color) & 0xFF;
    uint32_t pixel = (255 << 24) | (b << 16) | (g << 8) | r;
    
    int x0 = rx < 0 ? 0 : rx;
    int y0 = ry < 0 ? 0 : ry;
    int x1 = (rx + rw) > gWidth ? gWidth : (rx + rw);
    int y1 = (ry + rh) > gHeight ? gHeight : (ry + rh);
    
    uint32_t* buf32 = (uint32_t*)gPixelBuffer;
    
    for (int y = y0; y < y1; y++) {
        int rowStart = y * gWidth;
        for (int x = x0; x < x1; x++) {
            buf32[rowStart + x] = pixel;
        }
    }
    
    return makeBool(true);
}

static void pollEvents(void) {
    
    memset(gKeyHit, 0, sizeof(gKeyHit));
    
    @autoreleasepool {
        
        if (gWindow && ![gWindow isKeyWindow]) {
            [gWindow makeKeyWindow];
        }
        if (gView && [gWindow firstResponder] != gView) {
            [gWindow makeFirstResponder:gView];
        }
        
        NSEvent* event = nil;
        while ((event = [NSApp nextEventMatchingMask:NSEventMaskAny
                                            untilDate:[NSDate distantPast]
                                               inMode:NSDefaultRunLoopMode
                                              dequeue:YES])) {
            switch ([event type]) {
                case NSEventTypeKeyDown:
                    
                    if ([event keyCode] < 256) {
                        gKeyState[[event keyCode]] = true;
                        gKeyHit[[event keyCode]] = true;
                    }
                    break;
                case NSEventTypeKeyUp:
                    if ([event keyCode] < 256) gKeyState[[event keyCode]] = false;
                    break;
                case NSEventTypeFlagsChanged: {
                    
                    NSUInteger flags = [event modifierFlags];
                    gKeyState[56] = (flags & NSEventModifierFlagShift) != 0;    
                    gKeyState[60] = (flags & NSEventModifierFlagShift) != 0;    
                    gKeyState[59] = (flags & NSEventModifierFlagControl) != 0;  
                    gKeyState[62] = (flags & NSEventModifierFlagControl) != 0;  
                    gKeyState[58] = (flags & NSEventModifierFlagOption) != 0;   
                    gKeyState[61] = (flags & NSEventModifierFlagOption) != 0;   
                    gKeyState[55] = (flags & NSEventModifierFlagCommand) != 0;  
                    gKeyState[54] = (flags & NSEventModifierFlagCommand) != 0;  
                    break;
                }
                case NSEventTypeLeftMouseDown:
                    gMouseState[0] = true; break;
                case NSEventTypeLeftMouseUp:
                    gMouseState[0] = false; break;
                case NSEventTypeRightMouseDown:
                    gMouseState[1] = true; break;
                case NSEventTypeRightMouseUp:
                    gMouseState[1] = false; break;
                case NSEventTypeOtherMouseDown:
                    gMouseState[2] = true; break;
                case NSEventTypeOtherMouseUp:
                    gMouseState[2] = false; break;
                case NSEventTypeMouseMoved:
                case NSEventTypeLeftMouseDragged:
                case NSEventTypeRightMouseDragged:
                case NSEventTypeOtherMouseDragged: {
                    NSPoint p = [event locationInWindow];
                    if (gView) {
                        p = [gView convertPoint:p fromView:nil];
                    }
                    gMouseX = (int)p.x;
                    gMouseY = gHeight - (int)p.y - 1;
                    if (gMouseX < 0) gMouseX = 0;
                    if (gMouseY < 0) gMouseY = 0;
                    if (gMouseX >= gWidth) gMouseX = gWidth - 1;
                    if (gMouseY >= gHeight) gMouseY = gHeight - 1;
                    break;
                }
                default:
                    break;
            }
            [NSApp sendEvent:event];
        }
        [NSApp updateWindows];
    }
}

Value native_gui_present(Value* args, int argc) {
    (void)args;
    (void)argc;
    
    if (!gWindowOpen || !gWindow || !gTexture) {
        return makeBool(false);
    }
    
    @autoreleasepool {
        pollEvents();
        MTLRegion region = MTLRegionMake2D(0, 0, gWidth, gHeight);
        [gTexture replaceRegion:region mipmapLevel:0 withBytes:gPixelBuffer bytesPerRow:gWidth * 4];
        
        id<CAMetalDrawable> drawable = [gMetalLayer nextDrawable];
        if (!drawable) {
            return makeBool(gWindowOpen);
        }
        
        MTLRenderPassDescriptor* passDesc = [MTLRenderPassDescriptor renderPassDescriptor];
        passDesc.colorAttachments[0].texture = drawable.texture;
        passDesc.colorAttachments[0].loadAction = MTLLoadActionClear;
        passDesc.colorAttachments[0].storeAction = MTLStoreActionStore;
        passDesc.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
        
        id<MTLCommandBuffer> commandBuffer = [gCommandQueue commandBuffer];
        id<MTLRenderCommandEncoder> encoder = [commandBuffer renderCommandEncoderWithDescriptor:passDesc];
        
        [encoder setRenderPipelineState:gPipelineState];
        [encoder setFragmentTexture:gTexture atIndex:0];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
        [encoder endEncoding];
        
        [commandBuffer presentDrawable:drawable];
        [commandBuffer commit];
        
        pollEvents();
        
        if (![gWindow isVisible]) {
            gWindowOpen = false;
        }
        
        uint64_t currentTime = mach_absolute_time();
        uint64_t elapsed = currentTime - gLastFrameTime;
        double elapsedSeconds = (double)elapsed * gTimebaseInfo.numer / gTimebaseInfo.denom / 1e9;
        
        if (elapsedSeconds < gTargetFrameTime) {
            double sleepTime = gTargetFrameTime - elapsedSeconds;
            usleep((useconds_t)(sleepTime * 1e6));
            
            currentTime = mach_absolute_time();
            elapsed = currentTime - gLastFrameTime;
            elapsedSeconds = (double)elapsed * gTimebaseInfo.numer / gTimebaseInfo.denom / 1e9;
        }
        
        gDeltaTime = elapsedSeconds;
        gFPS = 1.0 / elapsedSeconds;
        gLastFrameTime = currentTime;
    }
    
    return makeBool(gWindowOpen);
}

Value native_gui_draw_line(Value* args, int argc) {
    if (argc != 5) {
        runtimeError("gui_draw_line expects 5 arguments: x0, y0, x1, y1, color");
        return makeNull();
    }
    
    if (!gPixelBuffer || !gWindowOpen) {
        return makeBool(false);
    }
    
    int x0 = (int)toNum(args[0]);
    int y0 = (int)toNum(args[1]);
    int x1 = (int)toNum(args[2]);
    int y1 = (int)toNum(args[3]);
    uint32_t color = (uint32_t)toNum(args[4]);
    
    int r = (color >> 16) & 0xFF;
    int g = (color >> 8) & 0xFF;
    int b = (color) & 0xFF;
    uint32_t pixel = (255 << 24) | (b << 16) | (g << 8) | r;
    uint32_t* buf32 = (uint32_t*)gPixelBuffer;
    
    int dx = abs(x1 - x0);
    int dy = -abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1;
    int sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    
    while (1) {
        if (x0 >= 0 && x0 < gWidth && y0 >= 0 && y0 < gHeight) {
            buf32[y0 * gWidth + x0] = pixel;
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

static inline void setPixelFast(uint32_t* buf, int x, int y, int w, int h, uint32_t pixel) {
    if (x >= 0 && x < w && y >= 0 && y < h) {
        buf[y * w + x] = pixel;
    }
}

Value native_gui_draw_circle(Value* args, int argc) {
    if (argc != 4) {
        runtimeError("gui_draw_circle expects 4 arguments: xc, yc, radius, color");
        return makeNull();
    }
    
    if (!gPixelBuffer || !gWindowOpen) {
        return makeBool(false);
    }
    
    int xc = (int)toNum(args[0]);
    int yc = (int)toNum(args[1]);
    int radius = (int)toNum(args[2]);
    uint32_t color = (uint32_t)toNum(args[3]);
    
    int r = (color >> 16) & 0xFF;
    int g = (color >> 8) & 0xFF;
    int b = (color) & 0xFF;
    uint32_t pixel = (255 << 24) | (b << 16) | (g << 8) | r;
    uint32_t* buf32 = (uint32_t*)gPixelBuffer;
    
    int x = 0;
    int y = radius;
    int d = 3 - 2 * radius;
    
    while (y >= x) {
        setPixelFast(buf32, xc + x, yc + y, gWidth, gHeight, pixel);
        setPixelFast(buf32, xc - x, yc + y, gWidth, gHeight, pixel);
        setPixelFast(buf32, xc + x, yc - y, gWidth, gHeight, pixel);
        setPixelFast(buf32, xc - x, yc - y, gWidth, gHeight, pixel);
        setPixelFast(buf32, xc + y, yc + x, gWidth, gHeight, pixel);
        setPixelFast(buf32, xc - y, yc + x, gWidth, gHeight, pixel);
        setPixelFast(buf32, xc + y, yc - x, gWidth, gHeight, pixel);
        setPixelFast(buf32, xc - y, yc - x, gWidth, gHeight, pixel);
        
        x++;
        if (d > 0) {
            y--;
            d = d + 4 * (x - y) + 10;
        } else {
            d = d + 4 * x + 6;
        }
    }
    
    return makeBool(true);
}

Value native_gui_fill_circle(Value* args, int argc) {
    if (argc != 4) {
        runtimeError("gui_fill_circle expects 4 arguments: xc, yc, radius, color");
        return makeNull();
    }
    
    if (!gPixelBuffer || !gWindowOpen) {
        return makeBool(false);
    }
    
    int xc = (int)toNum(args[0]);
    int yc = (int)toNum(args[1]);
    int radius = (int)toNum(args[2]);
    uint32_t color = (uint32_t)toNum(args[3]);
    
    int r = (color >> 16) & 0xFF;
    int g = (color >> 8) & 0xFF;
    int b = (color) & 0xFF;
    uint32_t pixel = (255 << 24) | (b << 16) | (g << 8) | r;
    uint32_t* buf32 = (uint32_t*)gPixelBuffer;
    
    int r2 = radius * radius;
    
    for (int dy = -radius; dy <= radius; dy++) {
        int py = yc + dy;
        if (py < 0 || py >= gHeight) continue;
        
        int dx_max = (int)sqrt(r2 - dy * dy);
        int x0 = xc - dx_max;
        int x1 = xc + dx_max;
        
        if (x0 < 0) x0 = 0;
        if (x1 >= gWidth) x1 = gWidth - 1;
        
        for (int px = x0; px <= x1; px++) {
            buf32[py * gWidth + px] = pixel;
        }
    }
    
    return makeBool(true);
}

Value native_gui_get_width(Value* args, int argc) {
    (void)args; (void)argc;
    return makeNumber(gWidth);
}

Value native_gui_get_height(Value* args, int argc) {
    (void)args; (void)argc;
    return makeNumber(gHeight);
}

Value native_gui_get_dt(Value* args, int argc) {
    (void)args; (void)argc;
    return makeNumber(gDeltaTime);
}

Value native_gui_get_fps(Value* args, int argc) {
    (void)args; (void)argc;
    return makeNumber(gFPS);
}

Value native_gui_get_time(Value* args, int argc) {
    (void)args; (void)argc;
    uint64_t now = mach_absolute_time();
    double seconds = (double)now * gTimebaseInfo.numer / gTimebaseInfo.denom / 1e9;
    return makeNumber(seconds);
}

Value native_gui_poll(Value* args, int argc) {
    (void)args; (void)argc;
    pollEvents();
    return makeNull();
}

Value native_gui_mouse_pos(Value* args, int argc) {
    (void)args; (void)argc;
    Value arr = makeArray(2);
    arrayPush(arr.as.array, makeNumber(gMouseX));
    arrayPush(arr.as.array, makeNumber(gMouseY));
    arr.as.array->length = 2;
    return arr;
}

Value native_gui_mouse_down(Value* args, int argc) {
    if (argc < 1 || !isNumeric(args[0])) return makeBool(false);
    int b = (int)toNum(args[0]);
    if (b < 0 || b > 2) return makeBool(false);
    return makeBool(gMouseState[b]);
}

Value native_gui_key_down(Value* args, int argc) {
    if (argc < 1 || !isNumeric(args[0])) return makeBool(false);
    int code = (int)toNum(args[0]);
    if (code < 0 || code >= 256) return makeBool(false);
    
    return makeBool(gKeyState[code] || gKeyHit[code]);
}

Value native_gui_get_keys(Value* args, int argc) {
    (void)args; (void)argc;
    Value arr = makeArray(0);
    for (int i = 0; i < 256; i++) {
        if (gKeyState[i] || gKeyHit[i]) {
            arrayPush(arr.as.array, makeNumber(i));
        }
    }
    return arr;
}

Value native_gui_load_png(Value* args, int argc) {
    if (argc < 1 || args[0].type != VAL_STRING) return makeNull();
    
    NSString* path = [NSString stringWithUTF8String:args[0].as.string];
    if (!path) return makeNull();
    
    NSData* data = [NSData dataWithContentsOfFile:path];
    if (!data) return makeNull();
    
    CGImageSourceRef source = CGImageSourceCreateWithData((__bridge CFDataRef)data, NULL);
    if (!source) return makeNull();
    
    CGImageRef image = CGImageSourceCreateImageAtIndex(source, 0, NULL);
    CFRelease(source);
    if (!image) return makeNull();
    
    size_t width = CGImageGetWidth(image);
    size_t height = CGImageGetHeight(image);
    size_t bytesPerPixel = 4;
    size_t bytesPerRow = bytesPerPixel * width;
    size_t totalBytes = bytesPerRow * height;
    
    unsigned char* raw = malloc(totalBytes);
    if (!raw) {
        CGImageRelease(image);
        return makeNull();
    }
    
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(raw, width, height, 8, bytesPerRow, colorSpace, kCGImageAlphaPremultipliedLast);
    if (!context) {
        free(raw);
        CGImageRelease(image);
        CGColorSpaceRelease(colorSpace);
        return makeNull();
    }
    
    CGContextDrawImage(context, CGRectMake(0, 0, width, height), image);
    CGContextRelease(context);
    CGColorSpaceRelease(colorSpace);
    CGImageRelease(image);
    
    
    Value pixels = makeArray((int)(width * height));
    pixels.as.array->length = 0;
    
    for (size_t i = 0; i < width * height; i++) {
        unsigned char* px = raw + (i * 4);
        int color = (px[0] << 16) | (px[1] << 8) | px[2]; 
        arrayPush(pixels.as.array, makeInt(color));
    }
    
    free(raw);
    
    Value result = makeArray(3);  
    arrayPush(result.as.array, makeInt((int)width));
    arrayPush(result.as.array, makeInt((int)height));
    arrayPush(result.as.array, pixels);
    result.as.array->length = 3;
    return result;
}


Value native_gui_blit(Value* args, int argc) {
    if (argc < 5) {
        runtimeError("gui_blit expects 5 args: x, y, w, h, pixels");
        return makeBool(false);
    }
    if (!gPixelBuffer || !gWindowOpen) return makeBool(false);
    
    int ox = (int)toNum(args[0]);
    int oy = (int)toNum(args[1]);
    int w = (int)toNum(args[2]);
    int h = (int)toNum(args[3]);
    if (args[4].type != VAL_ARRAY || w <= 0 || h <= 0) return makeBool(false);
    
    JaiArray* arr = args[4].as.array;
    int expected = w * h;
    if (arr->length < expected) return makeBool(false);
    
    uint32_t* dst = (uint32_t*)gPixelBuffer;
    
    for (int y = 0; y < h; y++) {
        int dy = oy + y;
        if (dy < 0 || dy >= gHeight) continue;
        int dstRow = dy * gWidth;
        int srcRow = y * w;
        for (int x = 0; x < w; x++) {
            int dx = ox + x;
            if (dx < 0 || dx >= gWidth) continue;
            Value v = arr->items[srcRow + x];
            uint32_t rgb = (uint32_t)toNum(v);
            uint32_t r = (rgb >> 16) & 0xFF;
            uint32_t g = (rgb >> 8) & 0xFF;
            uint32_t b = rgb & 0xFF;
            dst[dstRow + dx] = (255 << 24) | (b << 16) | (g << 8) | r;
        }
    }
    
    return makeBool(true);
}

void registerGuiFunctions(void) {
    setVariable("gui_init", makeNativeFunc(native_gui_init));
    setVariable("gui_set_pixel", makeNativeFunc(native_gui_set_pixel));
    setVariable("gui_clear", makeNativeFunc(native_gui_clear));
    setVariable("gui_fill_rect", makeNativeFunc(native_gui_fill_rect));
    setVariable("gui_draw_line", makeNativeFunc(native_gui_draw_line));
    setVariable("gui_draw_circle", makeNativeFunc(native_gui_draw_circle));
    setVariable("gui_fill_circle", makeNativeFunc(native_gui_fill_circle));
    setVariable("gui_present", makeNativeFunc(native_gui_present));
    setVariable("gui_get_width", makeNativeFunc(native_gui_get_width));
    setVariable("gui_get_height", makeNativeFunc(native_gui_get_height));
    setVariable("gui_get_dt", makeNativeFunc(native_gui_get_dt));
    setVariable("gui_get_fps", makeNativeFunc(native_gui_get_fps));
    setVariable("gui_get_time", makeNativeFunc(native_gui_get_time));
    setVariable("gui_poll", makeNativeFunc(native_gui_poll));
    setVariable("gui_mouse_pos", makeNativeFunc(native_gui_mouse_pos));
    setVariable("gui_mouse_down", makeNativeFunc(native_gui_mouse_down));
    setVariable("gui_key_down", makeNativeFunc(native_gui_key_down));
    setVariable("gui_get_keys", makeNativeFunc(native_gui_get_keys));
    setVariable("gui_load_png", makeNativeFunc(native_gui_load_png));
    setVariable("gui_blit", makeNativeFunc(native_gui_blit));
}
