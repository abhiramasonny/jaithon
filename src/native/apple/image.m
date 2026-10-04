/* image.m — the Apple image's side of native_image.h.
 *
 * Only compiled into anything when the build splits the Apple code into its
 * own dylib (JAI_NATIVE_SPLIT, set by the Makefile on arm64 Macs). In a
 * monolithic build the four names below are the core's own definitions and
 * this file is empty.
 *
 * The .m files call jaiRealloc, jaiCalloc, jaiClockMonotonic and
 * jaiParallelChunks as they always have. Inside the image those names resolve
 * here, and each one forwards to the core's definition through the table the
 * core passed to jaiNativeImageInit. Forwarding rather than duplicating
 * matters for two of them: jaiRealloc keeps the collector's byte count, and
 * jaiParallelChunks owns the one worker pool. Hidden, so that each image keeps
 * its own and exports only the native functions and jaiNativeImageInit. */

#ifdef JAI_NATIVE_SPLIT

#include "common/common.h"
#include "native/native_image.h"
#include "runtime/parallel.h"

static JaiNativeCoreApi gCore;

__attribute__((visibility("default")))
bool jaiNativeImageInit(const JaiNativeCoreApi *api) {
    if (api == NULL || api->abi != JAI_NATIVE_IMAGE_ABI ||
        api->size != (uint32_t)sizeof(JaiNativeCoreApi))
        return false;
    if (api->realloc == NULL || api->calloc == NULL ||
        api->clockMonotonic == NULL || api->parallelChunks == NULL)
        return false;
    gCore = *api;
    return true;
}

__attribute__((visibility("hidden")))
void *jaiRealloc(void *ptr, size_t oldSize, size_t newSize) {
    return gCore.realloc(ptr, oldSize, newSize);
}

__attribute__((visibility("hidden")))
void *jaiCalloc(size_t elemSize, size_t count) {
    return gCore.calloc(elemSize, count);
}

__attribute__((visibility("hidden")))
double jaiClockMonotonic(void) {
    return gCore.clockMonotonic();
}

__attribute__((visibility("hidden")))
void jaiParallelChunks(size_t count, size_t leastPerChunk,
                       void (*body)(void *context, size_t start, size_t end),
                       void *context) {
    gCore.parallelChunks(count, leastPerChunk, body, context);
}

#endif /* JAI_NATIVE_SPLIT */
