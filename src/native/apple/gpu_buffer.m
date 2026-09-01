/* gpu_buffer.m — the recycling pool, JaiGpuBuffer lifetime, and every
 * host/device transfer. */

#ifdef __APPLE__

#include "native/apple/gpu_internal.h"
#include "runtime/parallel.h"

/* Device buffers a freed tensor left behind, kept for the next allocation of
 * the same size.
 *
 * Every intermediate a network produces -- each activation, each gradient --
 * allocates a buffer and drops it a moment later, and the next step asks for
 * exactly the same sizes again. Handing back a fresh `MTLBuffer` each time
 * costs far more than the arithmetic that follows it: the same elementwise
 * kernel over four million floats runs at 373 GB/s writing into a buffer it
 * used before and 80 GB/s writing into a new one, because the new one's pages
 * are touched for the first time.
 *
 * Reuse is safe without any fence. There is one command queue, its command
 * buffers execute in the order they were committed, and the encoders are the
 * serial kind with hazard tracking -- so work that writes a recycled buffer is
 * always behind the work that read it. Contents are whatever the last owner
 * left, which is what `Buffer` has always promised.
 *
 * `JAITHON_GPU_POOL=0` turns it off, and `JAITHON_GPU_POISON=1` fills every
 * recycled buffer with a signalling NaN, so anything that quietly relied on a
 * fresh allocation arriving zeroed fails loudly instead of occasionally. */
/* MPS rejects a user buffer below its own alignment quantum. */
#define JAI_GPU_MIN_BYTES 256

#define JAI_POOL_MAX_ENTRIES 512
#define JAI_POOL_MAX_BYTES   (1536u * 1024u * 1024u)

typedef struct {
    size_t   bytes;
    void    *buffer;
    /* Carried across the pool with the memory it belongs to. A recycled buffer
     * whose previous owner left work queued against it is still waiting on
     * that work, and the new owner is the one who finds out. */
    uint64_t lastBatch;
} JaiPooledBuffer;

static JaiPooledBuffer gPool[JAI_POOL_MAX_ENTRIES];
static int             gPoolCount;
static size_t          gPoolBytes;
static int             gPoolMode;   /* 0 unknown, 1 on, 2 off */
static int             gPoolPoison; /* 0 unknown, 1 on, 2 off */

static bool poolEnabled(void) {
    if (gPoolMode == 0) {
        const char *setting = getenv("JAITHON_GPU_POOL");
        gPoolMode = (setting != NULL && strcmp(setting, "0") == 0) ? 2 : 1;
    }
    return gPoolMode == 1;
}

static bool poolPoisons(void) {
    if (gPoolPoison == 0) {
        const char *setting = getenv("JAITHON_GPU_POISON");
        gPoolPoison = (setting != NULL && strcmp(setting, "0") != 0) ? 1 : 2;
    }
    return gPoolPoison == 1;
}

/* The most recently parked buffer of exactly this size, or nil. Newest first,
 * because that one is likeliest to still be warm. */
/* Under this, a fresh allocation is cheaper than waiting for a parked buffer
 * to be free. Metal maps a small buffer in a few microseconds; waiting for a
 * kernel that is still reading one costs the better part of a millisecond. */
#define JAI_POOL_WAIT_RATHER_THAN_ALLOCATE (1u * 1024u * 1024u)

/* Hand back a parked buffer of exactly this size, preferring one whose work
 * has finished.
 *
 * Two of the same size are not interchangeable: one may still be the
 * destination or the source of a kernel that has not run, and its new owner
 * pays for that the first time it writes from the host -- 176 us against the
 * 4 us the dispatch it was feeding costs. A run of small buffers, each
 * uploaded and handed straight to one kernel, is the worst case: recycling the
 * busy one every time turns the run into a chain of round trips.
 *
 * So a small buffer refuses a busy one and lets the caller allocate. The pool
 * then grows to however many of that size are in flight at once and settles
 * there, which is the multi-buffering a caller would otherwise have to write.
 * A large one takes what it is given: the allocation is expensive, and the
 * wait is small beside the work such a buffer is usually part of. */
static id<MTLBuffer> poolTake(size_t bytes, uint64_t *lastBatch) {
    const uint64_t settled = doneBatch();
    const int passes = bytes < JAI_POOL_WAIT_RATHER_THAN_ALLOCATE ? 1 : 2;
    for (int pass = 0; pass < passes; pass++) {
        for (int i = gPoolCount - 1; i >= 0; i--) {
            if (gPool[i].bytes != bytes) continue;
            if (pass == 0 && gPool[i].lastBatch > settled) continue;
            id<MTLBuffer> buffer = (__bridge_transfer id<MTLBuffer>)gPool[i].buffer;
            if (lastBatch != NULL) *lastBatch = gPool[i].lastBatch;
            memmove(&gPool[i], &gPool[i + 1],
                    (size_t)(gPoolCount - i - 1) * sizeof(JaiPooledBuffer));
            gPoolCount--;
            gPoolBytes -= bytes;
            return buffer;
        }
    }
    return nil;
}

/* Park a buffer, evicting the oldest when there is no room. Evicting rather
 * than refusing keeps a workload that cycles through many sizes from filling
 * the pool with entries it will never ask for again. */
static bool poolGive(void *buffer, size_t bytes, uint64_t lastBatch) {
    if (bytes > JAI_POOL_MAX_BYTES) return false;
    while (gPoolCount > 0 &&
           (gPoolCount >= JAI_POOL_MAX_ENTRIES ||
            gPoolBytes + bytes > JAI_POOL_MAX_BYTES)) {
        CFBridgingRelease(gPool[0].buffer);
        gPoolBytes -= gPool[0].bytes;
        memmove(&gPool[0], &gPool[1],
                (size_t)(gPoolCount - 1) * sizeof(JaiPooledBuffer));
        gPoolCount--;
    }
    if (gPoolCount >= JAI_POOL_MAX_ENTRIES) return false;
    gPool[gPoolCount].bytes = bytes;
    gPool[gPoolCount].buffer = buffer;
    gPool[gPoolCount].lastBatch = lastBatch;
    gPoolCount++;
    gPoolBytes += bytes;
    return true;
}

void *jaiGpuBufferHandle(JaiGpuBuffer *b) {
    return b == NULL ? NULL : b->buffer;
}

JaiGpuBuffer *jaiGpuAlloc(size_t bytes) {
    if (bytes == 0 || !ensureDevice()) return NULL;
    if (bytes > gMaxBufferLength) return NULL;

    @autoreleasepool {
        id<MTLBuffer> buffer = nil;
        bool reused = false;
        uint64_t carried = 0;
        if (poolEnabled()) {
            @synchronized(gQueue) {
                buffer = poolTake(bytes, &carried);
            }
            reused = buffer != nil;
            if (reused && poolPoisons()) {
                jaiGpuSynchronize();
                float *slots = (float *)[buffer contents];
                const size_t count = bytes / sizeof(float);
                for (size_t i = 0; i < count; i++) slots[i] = NAN;
            }
        }
        if (buffer == nil) {
            /* Never smaller than MPS's own minimum. Several of its primitives
             * refuse a user buffer under a quantum of theirs, and a tensor of
             * four floats is a real thing to ask for; the allocation is
             * rounded up while `bytes` stays the size that was asked for, so
             * every bounds check still measures the real extent. */
            const size_t least = bytes < JAI_GPU_MIN_BYTES ? JAI_GPU_MIN_BYTES : bytes;
            buffer = [gDevice newBufferWithLength:least
                                          options:MTLResourceStorageModeShared];
        }
        if (buffer == nil) return NULL;

        JaiGpuBuffer *b = JAI_ALLOC(JaiGpuBuffer, 1);
        b->buffer = (__bridge_retained void *)buffer;
        b->bytes = bytes;
        b->recycled = reused;
        b->lastBatch = reused ? carried : 0;
        return b;
    }
}

/* Wait for queued work before the host writes over a buffer.
 *
 * One piece of GPU work needs no fence against the next -- there is one queue
 * and it runs in order. The host is not in that order. A buffer may still be
 * the destination of a kernel that has been encoded and not yet run, and that
 * kernel would land on top of whatever the host wrote: an optimizer's
 * momentum, zeroed by its new parameter and then filled in again by the
 * previous parameter's update.
 *
 * A buffer with nothing queued against it -- which is most of them, and every
 * one that has only ever been written from the host -- goes straight through.
 * The rest wait for their own batch and not for the queue, so a host write to
 * one buffer does not stall on work belonging to another. */
static void hostWriteBarrier(JaiGpuBuffer *b) {
    if (b == NULL) return;
    if (!b->recycled && b->lastBatch == 0) return;
    b->recycled = false;
    jaiGpuWaitFor(b);
}

void jaiGpuFree(JaiGpuBuffer *b) {
    if (b == NULL) return;

    @autoreleasepool {
        bool parked = false;
        if (poolEnabled() && b->buffer != NULL && gQueue != nil) {
            @synchronized(gQueue) {
                parked = poolGive(b->buffer, b->bytes, b->lastBatch);
            }
        }
        if (!parked) CFBridgingRelease(b->buffer);
    }

    JAI_FREE(JaiGpuBuffer, b);
}

/* Shared storage means the pointer is host-visible and coherent; there is no
 * separate staging copy and nothing to synchronise after a write. */
void jaiGpuUpload(JaiGpuBuffer *b, const void *src, size_t bytes, size_t offset) {
    if (b == NULL || b->buffer == NULL || src == NULL || bytes == 0) return;
    /* A partial copy would look like success and leave the tail stale, so an
     * oversized request copies nothing. Callers bound-check first. */
    if (offset > b->bytes || bytes > b->bytes - offset) return;
    hostWriteBarrier(b);

    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)b->buffer;
    memcpy((uint8_t *)[buffer contents] + offset, src, bytes);
}

void jaiGpuUploadU8(JaiGpuBuffer *b, const uint8_t *src, size_t count,
                    size_t offset, float scale) {
    if (b == NULL || b->buffer == NULL || src == NULL || count == 0) return;
    const size_t start = offset * sizeof(float);
    const size_t bytes = count * sizeof(float);
    if (start > b->bytes || bytes > b->bytes - start) return;
    /* Both routes below can write from the host, so the barrier covers the
     * whole function rather than the fallback alone. */
    hostWriteBarrier(b);
    if (count < JAI_GPU_MIN_WORK || count > UINT32_MAX || !ensureDevice() ||
        !ensureBuiltins() || gExpandU8 == nil) {
        float *destination =
            (float *)((__bridge id<MTLBuffer>)b->buffer).contents + offset;
        for (size_t i = 0; i < count; i++) destination[i] = (float)src[i] * scale;
        return;
    }

    @autoreleasepool {
        id<MTLBuffer> dest = (__bridge id<MTLBuffer>)b->buffer;
        id<MTLBuffer> staging = [gDevice newBufferWithBytes:src
                                                     length:count
                                                    options:MTLResourceStorageModeShared];
        if (staging == nil) {
            float *destination = (float *)[dest contents] + offset;
            for (size_t i = 0; i < count; i++) destination[i] = (float)src[i] * scale;
            return;
        }
        uint32_t n = (uint32_t)count;
        float scaleValue = scale;
        @synchronized(gQueue) {
            if (gAsyncCommands == nil) {
                gAsyncCommands = [gQueue commandBuffer];
                beginBatchLocked();
                if (gAsyncCommands == nil) {
                    float *destination = (float *)[dest contents] + offset;
                    for (size_t i = 0; i < count; i++) {
                        destination[i] = (float)src[i] * scale;
                    }
                    return;
                }
            }
            if (gAsyncEncoder == nil) {
                gAsyncEncoder = [gAsyncCommands computeCommandEncoder];
                if (gAsyncEncoder == nil) {
                    float *destination = (float *)[dest contents] + offset;
                    for (size_t i = 0; i < count; i++) {
                        destination[i] = (float)src[i] * scale;
                    }
                    return;
                }
            }
            markLocked(b);
            [gAsyncEncoder setComputePipelineState:gExpandU8];
            [gAsyncEncoder setBuffer:staging offset:0 atIndex:0];
            [gAsyncEncoder setBuffer:dest offset:start atIndex:1];
            [gAsyncEncoder setBytes:&n length:sizeof(n) atIndex:2];
            [gAsyncEncoder setBytes:&scaleValue length:sizeof(scaleValue) atIndex:3];
            encodeDispatch(gAsyncEncoder, gExpandU8, (NSUInteger)count, 256);
        }
    }
}

void jaiGpuFillUniform(JaiGpuBuffer *b, size_t elementOffset, size_t count,
                       float low, float high, uint64_t seed) {
    if (b == NULL || b->buffer == NULL || count == 0) return;
    const size_t start = elementOffset * sizeof(float);
    const size_t bytes = count * sizeof(float);
    if (start > b->bytes || bytes > b->bytes - start) return;
    hostWriteBarrier(b);
    float *destination =
        (float *)((__bridge id<MTLBuffer>)b->buffer).contents + elementOffset;
    uint64_t state = seed != 0 ? seed : 0x9E3779B97F4A7C15ull;
    const float scale = (high - low) * (1.0f / 16777216.0f);
    for (size_t i = 0; i < count; i++) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        destination[i] = low + scale * (float)((uint32_t)(state >> 40));
    }
}

void jaiGpuFillZero(JaiGpuBuffer *b, size_t elementOffset, size_t count) {
    if (b == NULL || b->buffer == NULL || count == 0) return;
    const size_t start = elementOffset * sizeof(float);
    const size_t bytes = count * sizeof(float);
    if (start > b->bytes || bytes > b->bytes - start) return;
    hostWriteBarrier(b);
    memset((uint8_t *)((__bridge id<MTLBuffer>)b->buffer).contents + start, 0, bytes);
}

void jaiGpuDownload(JaiGpuBuffer *b, void *dst, size_t bytes, size_t offset) {
    if (b == NULL || b->buffer == NULL || dst == NULL || bytes == 0) return;
    if (offset > b->bytes || bytes > b->bytes - offset) return;
    jaiGpuWaitFor(b);

    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)b->buffer;
    memcpy(dst, (const uint8_t *)[buffer contents] + offset, bytes);
}

typedef struct {
    const float *source;
    uint8_t     *dst;
    float        factor;
} JaiNarrowWork;

static void narrowRange(void *context, size_t start, size_t end) {
    const JaiNarrowWork *work = (const JaiNarrowWork *)context;
    for (size_t i = start; i < end; i++) {
        const float value = work->source[i] * work->factor;
        if (!(value > 0.0f)) {
            work->dst[i] = 0u;
            continue;
        }
        const float rounded = value + 0.5f;
        work->dst[i] = rounded > 255.0f ? 255u : (uint8_t)rounded;
    }
}

static void jaiParallelNarrow(const float *source, uint8_t *dst, size_t count,
                              float factor) {
    JaiNarrowWork work = {source, dst, factor};
    jaiParallelChunks(count, 32768, narrowRange, &work);
}

void jaiGpuDownloadU8(JaiGpuBuffer *b, uint8_t *dst, size_t count,
                      size_t offset, float scale) {
    if (b == NULL || b->buffer == NULL || dst == NULL || count == 0) return;
    const size_t start = offset * sizeof(float);
    const size_t bytes = count * sizeof(float);
    if (start > b->bytes || bytes > b->bytes - start) return;
    jaiGpuWaitFor(b);

    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)b->buffer;
    /* Indexed as floats rather than stepped as bytes and cast: `start` is a
     * whole number of them, and the byte cast only makes the compiler warn
     * about an alignment that is already guaranteed. */
    const float *source = (const float *)[buffer contents] + offset;
    /* Rounds half up and clamps, matching what a display expects and what the
     * Jaithon loop this replaces did. A NaN fails `> 0.0f` and lands on zero
     * rather than an undefined cast. */
    const float factor = scale != 0.0f ? 1.0f / scale : 1.0f;
    /* One frame is millions of these and nothing is shared between them. */
    jaiParallelNarrow(source, dst, count, factor);
}

/* The buffer's own memory, ready to read, after everything queued has run.
 *
 * Storage is shared, so a download is only a copy because the caller usually
 * wants one somewhere else. A caller that is going to walk the values anyway
 * -- turning them into list elements, say -- can read them where they are and
 * skip a staging array and a copy of the whole thing.
 *
 * The pointer is good until the next GPU work touches the buffer. */
const float *jaiGpuMapRead(JaiGpuBuffer *b, size_t elementOffset, size_t count) {
    if (b == NULL || b->buffer == NULL) return NULL;
    const size_t start = elementOffset * sizeof(float);
    const size_t bytes = count * sizeof(float);
    if (start > b->bytes || bytes > b->bytes - start) return NULL;
    jaiGpuWaitFor(b);
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)b->buffer;
    return (const float *)[buffer contents] + elementOffset;
}

float *jaiGpuMapWrite(JaiGpuBuffer *b, size_t elementOffset, size_t count) {
    if (b == NULL || b->buffer == NULL) return NULL;
    const size_t start = elementOffset * sizeof(float);
    const size_t bytes = count * sizeof(float);
    if (start > b->bytes || bytes > b->bytes - start) return NULL;
    hostWriteBarrier(b);
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)b->buffer;
    return (float *)[buffer contents] + elementOffset;
}

#endif /* __APPLE__ */
