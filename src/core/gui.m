#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
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
static bool gMouseState[3] = {0};
static int gMouseX = 0;
static int gMouseY = 0;

static uint64_t gLastFrameTime = 0;
static double gDeltaTime = 0.0;
static double gFPS = 0.0;
static mach_timebase_info_data_t gTimebaseInfo;
static double gTargetFrameTime = 1.0 / 60.0;
static int gTargetFPS = 60;

Value native_gui_init(Value* args, int argc) {
    if (argc < 3) {
        runtimeError("gui_init expects at least 3 arguments: width, height, title [, targetFPS]");
        return makeNull();
    }
    
    if (args[0].type != VAL_NUMBER || args[1].type != VAL_NUMBER || args[2].type != VAL_STRING) {
        runtimeError("gui_init arguments must be (number, number, string)");
        return makeNull();
    }
    
    gWidth = (int)args[0].as.number;
    gHeight = (int)args[1].as.number;
    char* title = args[2].as.string;
    
    if (argc >= 4 && args[3].type == VAL_NUMBER) {
        gTargetFPS = (int)args[3].as.number;
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
        gMetalLayer.displaySyncEnabled = YES; //vsync
        
        [gWindow setContentView:gView];
        [gWindow makeKeyAndOrderFront:nil];
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

Value native_gui_clear(Value* args, int argc) {
    if (argc != 1) {
        runtimeError("gui_clear expects 1 argument: color");
        return makeNull();
    }
    
    if (!gPixelBuffer || !gWindowOpen) {
        return makeBool(false);
    }
    
    uint32_t color = (uint32_t)args[0].as.number;
    int r = (color >> 16) & 0xFF;
    int g = (color >> 8) & 0xFF;
    int b = (color) & 0xFF;
    
    uint32_t pixel = (255 << 24) | (b << 16) | (g << 8) | r;//abgr
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
    
    int rx = (int)args[0].as.number;
    int ry = (int)args[1].as.number;
    int rw = (int)args[2].as.number;
    int rh = (int)args[3].as.number;
    uint32_t color = (uint32_t)args[4].as.number;
    
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
    NSEvent* event = nil;
    while ((event = [NSApp nextEventMatchingMask:NSEventMaskAny
                                        untilDate:[NSDate distantPast]
                                           inMode:NSDefaultRunLoopMode
                                          dequeue:YES])) {
        switch ([event type]) {
            case NSEventTypeKeyDown:
                if ([event keyCode] < 256) gKeyState[[event keyCode]] = true;
                break;
            case NSEventTypeKeyUp:
                if ([event keyCode] < 256) gKeyState[[event keyCode]] = false;
                break;
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
    
    int x0 = (int)args[0].as.number;
    int y0 = (int)args[1].as.number;
    int x1 = (int)args[2].as.number;
    int y1 = (int)args[3].as.number;
    uint32_t color = (uint32_t)args[4].as.number;
    
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
    
    int xc = (int)args[0].as.number;
    int yc = (int)args[1].as.number;
    int radius = (int)args[2].as.number;
    uint32_t color = (uint32_t)args[3].as.number;
    
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
    
    int xc = (int)args[0].as.number;
    int yc = (int)args[1].as.number;
    int radius = (int)args[2].as.number;
    uint32_t color = (uint32_t)args[3].as.number;
    
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
    if (argc < 1 || args[0].type != VAL_NUMBER) return makeBool(false);
    int b = (int)args[0].as.number;
    if (b < 0 || b > 2) return makeBool(false);
    return makeBool(gMouseState[b]);
}

Value native_gui_key_down(Value* args, int argc) {
    if (argc < 1 || args[0].type != VAL_NUMBER) return makeBool(false);
    int code = (int)args[0].as.number;
    if (code < 0 || code >= 256) return makeBool(false);
    return makeBool(gKeyState[code]);
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
}
