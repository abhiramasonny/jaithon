/* native_image.h — the contract between the core binary and the Apple images.
 *
 * On an arm64 Mac the Objective-C half of src/native (everything under apple/)
 * is linked into three dylibs beside the jaithon executable instead of into
 * it -- gpu (Metal, MPS, MPSGraph, CoreML), gui (AppKit, MetalKit) and camera
 * (AVFoundation) -- named jaithon-native-<image>-<uuid>.dylib. The core links
 * no Apple framework at all, so a program that never touches a GPU, a window,
 * a camera or a CoreML model never pays to load them. That load was 36% of the
 * cycles of a cached hello-world run: an empty C program costs 3.65M cycles,
 * the same program linked against the twelve frameworks 8.56M.
 *
 * Every function in native.h that the core calls is a three-instruction
 * trampoline (generated into build/gen/native_bridge.S from the names stubs.c
 * defines) that jumps through a slot. A slot starts at a binder; the first
 * call of a name opens the image that exports it, points every slot of that
 * image at the image's definitions, and continues into it. When the image is
 * missing or is not the one this binary was built with, its slots point at
 * stubs.c's definitions instead, which is exactly the behaviour of a machine
 * with no GPU. native_loader.c says where images are looked for.
 *
 * An image needs four things back from the core. Rather than resolving them
 * against the executable's exports -- which an LTO link is free to internalise
 * -- the core hands them over in this table when it opens the image, and
 * apple/image.m defines the four names as forwarders through it. */
#ifndef JAI_NATIVE_IMAGE_H
#define JAI_NATIVE_IMAGE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

/* Bump whenever the table below changes shape. An image refuses a core whose
 * number differs, on top of the UUID check the core makes of the image. */
#define JAI_NATIVE_IMAGE_ABI 1u

typedef struct {
    uint32_t abi;    /* JAI_NATIVE_IMAGE_ABI */
    uint32_t size;   /* sizeof(JaiNativeCoreApi) */
    void  *(*realloc)(void *ptr, size_t oldSize, size_t newSize);
    void  *(*calloc)(size_t elemSize, size_t count);
    double (*clockMonotonic)(void);
    void   (*parallelChunks)(size_t count, size_t leastPerChunk,
                             void (*body)(void *context, size_t start, size_t end),
                             void *context);
} JaiNativeCoreApi;

/* Exported by every image; the core finds it with dlsym. False means the
 * image will not run against this core, and the core falls back to the stubs. */
typedef bool (*JaiNativeImageInitFn)(const JaiNativeCoreApi *api);
#define JAI_NATIVE_IMAGE_INIT "jaiNativeImageInit"
bool jaiNativeImageInit(const JaiNativeCoreApi *api);

/* Core side (native_loader.c). Called by the generated binder the first time a
 * slot is used; returns what the slot now holds. */
void *jaiNativeBind(uint32_t index);

#endif /* JAI_NATIVE_IMAGE_H */
