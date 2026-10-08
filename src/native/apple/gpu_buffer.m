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
/* Hash slots for the size classes: a power of two, over twice the most
 * classes there can be (one per parked buffer). */
#define JAI_POOL_SLOTS       1024u

/* One parked buffer. It sits on two lists at once: every parked buffer in the
 * order it was parked, which is the eviction order, and the ones of its own
 * size in the same order, which is where an allocation looks. Both are
 * doubly linked through indices, so taking from either end, or from the
 * middle when an eviction reaches it, is O(1). */
typedef struct {
    size_t   bytes;
    void    *buffer;
    /* Carried across the pool with the memory it belongs to. A recycled buffer
     * whose previous owner left work queued against it is still waiting on
     * that work, and the new owner is the one who finds out. */
    uint64_t lastBatch;
    int      older, newer;            /* all parked buffers; `newer` links the free list */
    int      sizeOlder, sizeNewer;    /* parked buffers of this size */
} JaiPooledBuffer;

/* A size class: the oldest and newest parked buffer of exactly `bytes`.
 * `bytes` 0 marks an empty slot; open addressing, linear probing. */
typedef struct {
    size_t bytes;
    int    oldest, newest;
} JaiPoolClass;

static JaiPooledBuffer gPool[JAI_POOL_MAX_ENTRIES];
static JaiPoolClass    gClasses[JAI_POOL_SLOTS];
static int             gPoolFree = -1;     /* unused entries, through `newer` */
static int             gPoolOldest = -1;
static int             gPoolNewest = -1;
static bool            gPoolReady;
static int             gPoolCount;
static size_t          gPoolBytes;
static int             gPoolMode;   /* 0 unknown, 1 on, 2 off */
static int             gPoolPoison; /* 0 unknown, 1 on, 2 off */
static int             gPoolSwap;   /* 0 unknown, 1 on, 2 off */

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

/* Whether a small recycled buffer still busy with its previous owner's work
 * may be traded for a fresh one at its first host write instead of waited for
 * -- see hostWriteBarrier. JAITHON_GPU_POOL_SWAP=0 turns that off, and a
 * small allocation then refuses a busy buffer and allocates, as it used to. */
static bool poolSwaps(void) {
    if (gPoolSwap == 0) {
        const char *setting = getenv("JAITHON_GPU_POOL_SWAP");
        gPoolSwap = (setting != NULL && strcmp(setting, "0") == 0) ? 2 : 1;
    }
    return gPoolSwap == 1;
}

/* Under this, a fresh allocation is cheaper than waiting for a parked buffer
 * to be free. Metal maps a small buffer in a few microseconds; waiting for a
 * kernel that is still reading one costs the better part of a millisecond. */
#define JAI_POOL_WAIT_RATHER_THAN_ALLOCATE (1u * 1024u * 1024u)

static void poolInit(void) {
    if (gPoolReady) return;
    for (int i = 0; i < JAI_POOL_MAX_ENTRIES; i++) {
        gPool[i].newer = i + 1 < JAI_POOL_MAX_ENTRIES ? i + 1 : -1;
    }
    gPoolFree = 0;
    gPoolReady = true;
}

static unsigned classHome(size_t bytes) {
    uint64_t h = (uint64_t)bytes * 0x9E3779B97F4A7C15ull;
    return (unsigned)(h >> 40) & (JAI_POOL_SLOTS - 1u);
}

static int classFind(size_t bytes) {
    for (unsigned i = classHome(bytes), n = 0; n < JAI_POOL_SLOTS;
         i = (i + 1u) & (JAI_POOL_SLOTS - 1u), n++) {
        if (gClasses[i].bytes == bytes) return (int)i;
        if (gClasses[i].bytes == 0) return -1;
    }
    return -1;
}

static int classAdd(size_t bytes) {
    unsigned i = classHome(bytes);
    while (gClasses[i].bytes != 0) {
        if (gClasses[i].bytes == bytes) return (int)i;
        i = (i + 1u) & (JAI_POOL_SLOTS - 1u);
    }
    gClasses[i].bytes = bytes;
    gClasses[i].oldest = -1;
    gClasses[i].newest = -1;
    return (int)i;
}

/* Empty a slot without tombstones: shift back any later entry of the probe
 * run whose home does not lie cyclically after the hole. */
static void classRemove(int slot) {
    unsigned hole = (unsigned)slot;
    unsigned j = hole;
    for (;;) {
        j = (j + 1u) & (JAI_POOL_SLOTS - 1u);
        if (gClasses[j].bytes == 0) break;
        const unsigned home = classHome(gClasses[j].bytes);
        const bool stays = hole <= j ? (hole < home && home <= j)
                                     : (hole < home || home <= j);
        if (stays) continue;
        gClasses[hole] = gClasses[j];
        hole = j;
    }
    gClasses[hole].bytes = 0;
}

/* Take entry `e` off both lists and back onto the free list. */
static void poolUnlink(int e, int slot) {
    JaiPooledBuffer *p = &gPool[e];
    if (p->older >= 0) gPool[p->older].newer = p->newer; else gPoolOldest = p->newer;
    if (p->newer >= 0) gPool[p->newer].older = p->older; else gPoolNewest = p->older;
    JaiPoolClass *c = &gClasses[slot];
    if (p->sizeOlder >= 0) gPool[p->sizeOlder].sizeNewer = p->sizeNewer; else c->oldest = p->sizeNewer;
    if (p->sizeNewer >= 0) gPool[p->sizeNewer].sizeOlder = p->sizeOlder; else c->newest = p->sizeOlder;
    if (c->oldest < 0) classRemove(slot);
    gPoolCount--;
    gPoolBytes -= p->bytes;
    p->buffer = NULL;
    p->newer = gPoolFree;
    gPoolFree = e;
}

/* Hand back a parked buffer of exactly this size, preferring one whose work
 * has finished.
 *
 * Two of the same size are not interchangeable: one may still be the
 * destination or the source of a kernel that has not run. For GPU work that
 * does not matter -- the queue runs in order, so whatever the new owner
 * encodes lands behind it. It matters only to a host write, which is not in
 * that order, and hostWriteBarrier deals with it there: a large buffer waits
 * for the old work, a small untouched one is traded for a fresh one, which is
 * no dearer than allocating here would have been.
 *
 * Refusing a busy small buffer here instead, as this did, recycled nothing
 * in a chain that never waits: every freed buffer was still busy, the pool
 * filled with 512 of them and evicted one per free, and an elementwise op at
 * N=1024 cost 10.4 us against 8.5 with no pool at all. So a busy one is taken
 * when nothing settled is at either end of its size's list. The newest
 * settled one comes first, being the likeliest to be warm. */
static id<MTLBuffer> poolTake(size_t bytes, uint64_t *lastBatch) {
    const int slot = classFind(bytes);
    if (slot < 0) return nil;
    const uint64_t settled = doneBatch();
    const JaiPoolClass *c = &gClasses[slot];
    int e = c->newest;
    if (gPool[e].lastBatch > settled) {
        if (gPool[c->oldest].lastBatch <= settled) {
            e = c->oldest;
        } else if (bytes < JAI_POOL_WAIT_RATHER_THAN_ALLOCATE && !poolSwaps()) {
            return nil;
        }
    }
    id<MTLBuffer> buffer = (__bridge_transfer id<MTLBuffer>)gPool[e].buffer;
    if (lastBatch != NULL) *lastBatch = gPool[e].lastBatch;
    poolUnlink(e, slot);
    return buffer;
}

/* Park a buffer, evicting the oldest when there is no room. Evicting rather
 * than refusing keeps a workload that cycles through many sizes from filling
 * the pool with entries it will never ask for again. */
static bool poolGive(void *buffer, size_t bytes, uint64_t lastBatch) {
    if (bytes > JAI_POOL_MAX_BYTES) return false;
    poolInit();
    while (gPoolCount > 0 &&
           (gPoolCount >= JAI_POOL_MAX_ENTRIES ||
            gPoolBytes + bytes > JAI_POOL_MAX_BYTES)) {
        const int e = gPoolOldest;
        void *evicted = gPool[e].buffer;
        poolUnlink(e, classFind(gPool[e].bytes));
        CFBridgingRelease(evicted);
    }
    if (gPoolFree < 0) return false;
    const int e = gPoolFree;
    gPoolFree = gPool[e].newer;
    const int slot = classAdd(bytes);
    JaiPooledBuffer *p = &gPool[e];
    p->bytes = bytes;
    p->buffer = buffer;
    p->lastBatch = lastBatch;
    p->older = gPoolNewest;
    p->newer = -1;
    if (gPoolNewest >= 0) gPool[gPoolNewest].newer = e; else gPoolOldest = e;
    gPoolNewest = e;
    JaiPoolClass *c = &gClasses[slot];
    p->sizeOlder = c->newest;
    p->sizeNewer = -1;
    if (c->newest >= 0) gPool[c->newest].sizeNewer = e; else c->oldest = e;
    c->newest = e;
    gPoolCount++;
    gPoolBytes += bytes;
    return true;
}

void *jaiGpuBufferHandle(JaiGpuBuffer *b) {
    if (b == NULL) return NULL;
    /* The caller now holds the MTLBuffer itself and may write it without a
     * mark, so a later first host write must not trade it away. */
    b->untouched = false;
    return b->buffer;
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
        b->untouched = reused;
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
 * one buffer does not stall on work belonging to another.
 *
 * Except a small recycled one its new owner has not touched. The only work
 * still pending on it is the previous owner's, so nothing the new owner needs
 * is in those bytes: it trades the MTLBuffer for a fresh one and parks the
 * busy one again for GPU work, where queue order makes it safe. That is what
 * lets poolTake hand busy small buffers out at all -- a wait here would make
 * every upload into one a round trip. Touching it in any way (a mark, or
 * handing out the MTLBuffer) ends the trade, because from then on the bytes
 * or the handle belong to the new owner. */
static bool swapForFresh(JaiGpuBuffer *b) {
    const size_t least = b->bytes < JAI_GPU_MIN_BYTES ? JAI_GPU_MIN_BYTES : b->bytes;
    id<MTLBuffer> fresh = [gDevice newBufferWithLength:least
                                               options:MTLResourceStorageModeShared];
    if (fresh == nil) return false;
    void *busy = b->buffer;
    const uint64_t busyBatch = b->lastBatch;
    b->buffer = (__bridge_retained void *)fresh;
    b->lastBatch = 0;
    bool parked = false;
    @synchronized(gQueue) {
        parked = poolGive(busy, b->bytes, busyBatch);
    }
    if (!parked) CFBridgingRelease(busy);
    return true;
}

/* False when the wait failed -- the work is still queued or broke -- and the
 * caller must then not write: a write that went ahead would land under the
 * queued work it was meant to follow. The failure is pending, for the caller
 * to raise. */
static bool hostWriteBarrier(JaiGpuBuffer *b) {
    if (b == NULL) return true;
    if (!b->recycled && b->lastBatch == 0) return true;
    const bool tradable = b->recycled && b->untouched;
    if (tradable && b->bytes < JAI_POOL_WAIT_RATHER_THAN_ALLOCATE && poolSwaps() &&
        b->lastBatch > doneBatch()) {
        @autoreleasepool {
            if (swapForFresh(b)) {
                b->recycled = false;
                b->untouched = false;
                return true;
            }
        }
    }
    if (!jaiGpuWaitFor(b)) return false;
    b->recycled = false;
    b->untouched = false;
    return true;
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
bool jaiGpuUpload(JaiGpuBuffer *b, const void *src, size_t bytes, size_t offset) {
    if (b == NULL || b->buffer == NULL || src == NULL) return false;
    if (bytes == 0) return true;
    /* A partial copy would look like success and leave the tail stale, so an
     * oversized request copies nothing. Callers bound-check first. */
    if (offset > b->bytes || bytes > b->bytes - offset) return false;
    if (!hostWriteBarrier(b)) return false;

    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)b->buffer;
    memcpy((uint8_t *)[buffer contents] + offset, src, bytes);
    return true;
}

bool jaiGpuUploadU8(JaiGpuBuffer *b, const uint8_t *src, size_t count,
                    size_t offset, float scale) {
    if (b == NULL || b->buffer == NULL || src == NULL) return false;
    if (count == 0) return true;
    const size_t start = offset * sizeof(float);
    const size_t bytes = count * sizeof(float);
    if (start > b->bytes || bytes > b->bytes - start) return false;
    /* Both routes below can write from the host, so the barrier covers the
     * whole function rather than the fallback alone. */
    if (!hostWriteBarrier(b)) return false;
    if (count < JAI_GPU_MIN_WORK || count > UINT32_MAX || !ensureDevice() ||
        !ensureBuiltins() || gExpandU8 == nil) {
        float *destination =
            (float *)((__bridge id<MTLBuffer>)b->buffer).contents + offset;
        for (size_t i = 0; i < count; i++) destination[i] = (float)src[i] * scale;
        return true;
    }

    @autoreleasepool {
        id<MTLBuffer> dest = (__bridge id<MTLBuffer>)b->buffer;
        id<MTLBuffer> staging = [gDevice newBufferWithBytes:src
                                                     length:count
                                                    options:MTLResourceStorageModeShared];
        if (staging == nil) {
            float *destination = (float *)[dest contents] + offset;
            for (size_t i = 0; i < count; i++) destination[i] = (float)src[i] * scale;
            return true;
        }
        uint32_t n = (uint32_t)count;
        float scaleValue = scale;
        @synchronized(gQueue) {
            if (gAsyncCommands == nil) {
                gAsyncCommands = newAsyncCommandBuffer();
                beginBatchLocked();
                if (gAsyncCommands == nil) {
                    float *destination = (float *)[dest contents] + offset;
                    for (size_t i = 0; i < count; i++) {
                        destination[i] = (float)src[i] * scale;
                    }
                    return true;
                }
            }
            if (gAsyncEncoder == nil) {
                gAsyncEncoder = [gAsyncCommands computeCommandEncoder];
                if (gAsyncEncoder == nil) {
                    float *destination = (float *)[dest contents] + offset;
                    for (size_t i = 0; i < count; i++) {
                        destination[i] = (float)src[i] * scale;
                    }
                    return true;
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
    return true;
}

bool jaiGpuFillUniform(JaiGpuBuffer *b, size_t elementOffset, size_t count,
                       float low, float high, uint64_t seed) {
    if (b == NULL || b->buffer == NULL) return false;
    if (count == 0) return true;
    const size_t start = elementOffset * sizeof(float);
    const size_t bytes = count * sizeof(float);
    if (start > b->bytes || bytes > b->bytes - start) return false;
    if (!hostWriteBarrier(b)) return false;
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
    return true;
}

bool jaiGpuFillZero(JaiGpuBuffer *b, size_t elementOffset, size_t count) {
    if (b == NULL || b->buffer == NULL) return false;
    if (count == 0) return true;
    const size_t start = elementOffset * sizeof(float);
    const size_t bytes = count * sizeof(float);
    if (start > b->bytes || bytes > b->bytes - start) return false;
    if (!hostWriteBarrier(b)) return false;
    memset((uint8_t *)((__bridge id<MTLBuffer>)b->buffer).contents + start, 0, bytes);
    return true;
}

bool jaiGpuDownload(JaiGpuBuffer *b, void *dst, size_t bytes, size_t offset) {
    if (b == NULL || b->buffer == NULL || dst == NULL) return false;
    if (bytes == 0) return true;
    if (offset > b->bytes || bytes > b->bytes - offset) return false;
    if (!jaiGpuWaitFor(b)) return false;

    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)b->buffer;
    memcpy(dst, (const uint8_t *)[buffer contents] + offset, bytes);
    return true;
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

bool jaiGpuDownloadU8(JaiGpuBuffer *b, uint8_t *dst, size_t count,
                      size_t offset, float scale) {
    if (b == NULL || b->buffer == NULL || dst == NULL) return false;
    if (count == 0) return true;
    const size_t start = offset * sizeof(float);
    const size_t bytes = count * sizeof(float);
    if (start > b->bytes || bytes > b->bytes - start) return false;
    if (!jaiGpuWaitFor(b)) return false;

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
    return true;
}

/* The buffer's own memory, ready to read, after everything queued has run.
 *
 * Storage is shared, so a download is only a copy because the caller usually
 * wants one somewhere else. A caller that is going to walk the values anyway
 * -- turning them into list elements, say -- can read them where they are and
 * skip a staging array and a copy of the whole thing.
 *
 * The pointer is good until the next GPU work touches the buffer. NULL when
 * the wait failed, as well as out of range: bytes the queued work has not
 * finished writing are not a result. */
const float *jaiGpuMapRead(JaiGpuBuffer *b, size_t elementOffset, size_t count) {
    if (b == NULL || b->buffer == NULL) return NULL;
    const size_t start = elementOffset * sizeof(float);
    const size_t bytes = count * sizeof(float);
    if (start > b->bytes || bytes > b->bytes - start) return NULL;
    if (!jaiGpuWaitFor(b)) return NULL;
    b->untouched = false;   /* the caller holds a pointer into these bytes */
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)b->buffer;
    return (const float *)[buffer contents] + elementOffset;
}

float *jaiGpuMapWrite(JaiGpuBuffer *b, size_t elementOffset, size_t count) {
    if (b == NULL || b->buffer == NULL) return NULL;
    const size_t start = elementOffset * sizeof(float);
    const size_t bytes = count * sizeof(float);
    if (start > b->bytes || bytes > b->bytes - start) return NULL;
    if (!hostWriteBarrier(b)) return NULL;
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)b->buffer;
    return (float *)[buffer contents] + elementOffset;
}

#endif /* __APPLE__ */
