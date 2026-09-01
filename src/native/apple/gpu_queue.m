/* gpu_queue.m — the one open command buffer: batch numbering, in-flight
 * backpressure, flush and sync, and the kernels encoded into it. */

#ifdef __APPLE__

#include "native/apple/gpu_internal.h"

id<MTLCommandBuffer> gAsyncCommands;
id<MTLComputeCommandEncoder> gAsyncEncoder;
static NSMutableArray<id<MTLCommandBuffer>> *gInFlight;

static NSMutableDictionary<NSString *, id<MTLLibrary>> *gSourceLibraries;

#define JAI_GPU_MAX_IN_FLIGHT 16
#define JAI_GPU_AUTO_COMMIT 0

/* Which queued work a given buffer is waiting on.
 *
 * Everything the backend queues goes into one command buffer at a time, and
 * that command buffer is a "batch" with a number. A buffer records the number
 * of the last batch that could have written it, so reading it back waits for
 * that batch rather than for the whole queue to drain.
 *
 * The distinction is the difference between a pipeline and a stall. A live
 * loop that queues frame N's network and then reads frame N-1's pixels would,
 * draining the queue, wait for the network it just started -- which is exactly
 * the work it was trying to overlap with. Waiting per buffer, it does not.
 *
 * Marking a buffer that is only read by a kernel costs nothing but an extra
 * wait later, so every path that hands a buffer to the GPU marks it. Missing
 * one would be the other kind of wrong, so `JAITHON_GPU_FINE_SYNC=0` restores
 * the old drain-everything behaviour for bisecting. */
static uint64_t gBatchCounter;  /* last batch number handed out */
static uint64_t gOpenBatch;     /* number of gAsyncCommands, 0 when none open */
static unsigned gOpenEncoded;   /* dispatches encoded into it since it opened */
static int gAutoCommit = -1;
/* Every batch at or below this has finished. Written both by a thread that
 * waited for one and by Metal's own completion handler on a thread of its
 * own, so it is an atomic rather than something the queue lock covers -- the
 * handler must never have to take a lock a waiter might be holding. */
static _Atomic uint64_t gDoneBatch;
static NSMutableArray<NSNumber *> *gInFlightBatch;
static int gFineSync;           /* 0 unknown, 1 on, 2 off */

uint64_t doneBatch(void) {
    return atomic_load_explicit(&gDoneBatch, memory_order_acquire);
}

/* Raise the finished mark to `batch`, never lower it. */
void noteDone(uint64_t batch) {
    uint64_t seen = atomic_load_explicit(&gDoneBatch, memory_order_relaxed);
    while (seen < batch) {
        if (atomic_compare_exchange_weak_explicit(&gDoneBatch, &seen, batch,
                                                  memory_order_release,
                                                  memory_order_relaxed)) {
            return;
        }
    }
}

static bool fineSyncEnabled(void) {
    if (gFineSync == 0) {
        const char *setting = getenv("JAITHON_GPU_FINE_SYNC");
        gFineSync = (setting != NULL && strcmp(setting, "0") == 0) ? 2 : 1;
    }
    return gFineSync == 1;
}

static void commitOpenLocked(void);
static uint64_t takeFrontLocked(void);

/* How many dispatches may pile into one command buffer before it is sent.
 *
 * Nothing sends it until somebody waits, so the GPU sits idle for the whole
 * time the host spends encoding and only then starts -- the two never overlap.
 * Committing part way lets the card work on the front of a pipeline while the
 * host is still writing the back of it.
 *
 * Off by default, because it is only half a good idea. It is worth 5% on the
 * jaicv suite, where Jaithon interpreting between dispatches leaves the card
 * with nothing to do (2.72x to 2.85x against OpenCV at a threshold of eight).
 * It costs 26% on jaitensor, where the work is already back to back and a
 * graph split across command buffers loses the overlap MPSGraph arranges
 * inside one (1.76x to 1.31x against PyTorch MPS). The ML side is the one
 * that matters, so the default stays zero and the knob stays for tuning. */
static int autoCommitThreshold(void) {
    if (gAutoCommit < 0) {
        const char *setting = getenv("JAITHON_GPU_AUTO_COMMIT");
        gAutoCommit = setting != NULL ? atoi(setting) : JAI_GPU_AUTO_COMMIT;
        if (gAutoCommit < 0) gAutoCommit = 0;
    }
    return gAutoCommit;
}

static void ensureInFlight(void) {
    if (gInFlight == nil) gInFlight = [[NSMutableArray alloc] init];
    if (gInFlightBatch == nil) gInFlightBatch = [[NSMutableArray alloc] init];
}

/* Every site that opens gAsyncCommands calls this, so that a batch number
 * exists for the work about to be encoded into it. */
void beginBatchLocked(void) {
    gOpenBatch = ++gBatchCounter;
}

/* Note that whatever is being encoded now may write `b`.
 *
 * When nothing is open yet the batch is opened here, so that the number a
 * buffer records always belongs to a command buffer that will really be
 * committed -- a number guessed ahead of one would come loose if the work went
 * somewhere else instead. */
void markLocked(JaiGpuBuffer *b) {
    if (b == NULL || gQueue == nil) return;
    if (gOpenBatch == 0) {
        if (gAsyncCommands == nil) {
            gAsyncCommands = [gQueue commandBuffer];
            if (gAsyncCommands == nil) return;
        }
        beginBatchLocked();
    }
    b->lastBatch = gOpenBatch;
}

bool jaiGpuWaitFor(JaiGpuBuffer *b);

void mark(JaiGpuBuffer *b) {
    if (b == NULL || gQueue == nil) return;
    @synchronized(gQueue) {
        markLocked(b);
    }
}

/* For graphbuild.m and coreml.m, which reach past the JaiGpuBuffer for the
 * MTLBuffer inside it and encode against that directly. */
void jaiGpuBufferMark(JaiGpuBuffer *b) { mark(b); }

static void writeError(char *buf, size_t size, const char *fmt, ...) JAI_PRINTF(3, 4);

static void writeError(char *buf, size_t size, const char *fmt, ...) {
    if (buf == NULL || size == 0) return;
    va_list args;
    va_start(args, fmt);
    int written = vsnprintf(buf, size, fmt, args);
    va_end(args);
    if (written < 0) buf[0] = '\0';
}

JaiGpuKernel *jaiGpuCompile(const char *source, const char *entryPoint,
                            char *errBuf, size_t errBufSize) {
    if (errBuf != NULL && errBufSize > 0) errBuf[0] = '\0';

    if (source == NULL || entryPoint == NULL || entryPoint[0] == '\0') {
        writeError(errBuf, errBufSize, "kernel source and entry point are required");
        return NULL;
    }
    if (!ensureDevice()) {
        writeError(errBuf, errBufSize, "no Metal device is available");
        return NULL;
    }

    @autoreleasepool {
        NSString *text = [NSString stringWithUTF8String:source];
        NSString *entry = [NSString stringWithUTF8String:entryPoint];
        if (text == nil || entry == nil) {
            writeError(errBuf, errBufSize, "kernel source is not valid UTF-8");
            return NULL;
        }

        NSError *error = nil;
        id<MTLLibrary> library = nil;
        @synchronized(gQueue) {
            if (gSourceLibraries == nil) {
                gSourceLibraries = [[NSMutableDictionary alloc] init];
            }
            library = gSourceLibraries[text];
            if (library == nil) {
                library = [gDevice newLibraryWithSource:text options:nil error:&error];
                if (library != nil) gSourceLibraries[text] = library;
            }
        }
        if (library == nil) {
            /* The Metal front end packs its whole diagnostic listing — file,
             * line, caret — into the localised description. */
            const char *text8 = error != nil ? [[error localizedDescription] UTF8String] : NULL;
            writeError(errBuf, errBufSize, "%s",
                       text8 != NULL ? text8 : "Metal shader compilation failed");
            return NULL;
        }

        id<MTLFunction> function = [library newFunctionWithName:entry];
        if (function == nil) {
            writeError(errBuf, errBufSize, "no kernel function named '%s' in this source",
                       entryPoint);
            return NULL;
        }
        if ([function functionType] != MTLFunctionTypeKernel) {
            writeError(errBuf, errBufSize, "'%s' is not a kernel function", entryPoint);
            return NULL;
        }

        id<MTLComputePipelineState> pipeline =
            [gDevice newComputePipelineStateWithFunction:function error:&error];
        if (pipeline == nil) {
            const char *text8 = error != nil ? [[error localizedDescription] UTF8String] : NULL;
            writeError(errBuf, errBufSize, "%s",
                       text8 != NULL ? text8 : "could not build a compute pipeline");
            return NULL;
        }

        JaiGpuKernel *k = JAI_ALLOC_ZEROED(JaiGpuKernel, 1);
        k->pipeline = (__bridge_retained void *)pipeline;
        return k;
    }
}

/* `groupSize` 0 means the widest group the pipeline supports; a kernel with
 * a fixed-width threadgroup array must pass that width explicitly instead. */
void encodeDispatch(id<MTLComputeCommandEncoder> encoder,
                           id<MTLComputePipelineState> pipeline,
                           NSUInteger threads, NSUInteger groupSize) {
    if (threads == 0) return;

    const NSUInteger maxGroup = [pipeline maxTotalThreadsPerThreadgroup];
    const NSUInteger width = [pipeline threadExecutionWidth];

    if (groupSize == 0) {
        groupSize = maxGroup < JAI_DEFAULT_GROUP ? maxGroup : JAI_DEFAULT_GROUP;

        if (width > 1 && groupSize > width) {
            groupSize -= groupSize % width;
        }

        if (groupSize == 0)
            groupSize = width != 0 ? width : 1;
    } else if (groupSize > maxGroup) {
        groupSize = maxGroup;
    }

    if (groupSize > threads)
        groupSize = threads;

    if (groupSize == 0)
        groupSize = 1;

    const MTLSize group = MTLSizeMake(groupSize, 1, 1);

    if (gNonUniformThreadgroups) {
        [encoder dispatchThreads:MTLSizeMake(threads, 1, 1)
           threadsPerThreadgroup:group];
    } else {
        const NSUInteger groups = (threads + groupSize - 1) / groupSize;
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                threadsPerThreadgroup:group];
    }
}

void jaiGpuKernelFree(JaiGpuKernel *k) {
    if (k == NULL) return;
    @autoreleasepool {
        /* Hand the +1 from jaiGpuCompile back to ARC. */
        CFBridgingRelease(k->pipeline);
        k->pipeline = NULL;
    }
    JAI_FREE(JaiGpuKernel, k);
}

int jaiGpuMaxThreadsPerGroup(JaiGpuKernel *k) {
    if (k == NULL || k->pipeline == NULL) return 0;
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)k->pipeline;
    return (int)[pipeline maxTotalThreadsPerThreadgroup];
}

static bool dispatchKernel(JaiGpuKernel *k, JaiGpuBuffer **buffers, int count,
                           const uint32_t *scalars, int scalarCount,
                           int threads, int groupSize, const size_t *byteOffsets,
                           bool wait) {
    if (k == NULL || k->pipeline == NULL || threads <= 0) return false;
    if (count < 0 || (count > 0 && buffers == NULL)) return false;
    dispatchTraceTick(0);
    if (scalarCount < 0 || (scalarCount > 0 && scalars == NULL)) return false;
    if (groupSize < 0) return false;
    if (!ensureDevice()) return false;
    for (int i = 0; i < count; i++) {
        if (buffers[i] == NULL || buffers[i]->buffer == NULL) return false;
    }

    @autoreleasepool {
        id<MTLComputePipelineState> pipeline =
            (__bridge id<MTLComputePipelineState>)k->pipeline;
        if (!wait) {
            id<MTLCommandBuffer> oldest = nil;
            uint64_t oldestBatch = 0;
            @synchronized(gQueue) {
                if (gAsyncCommands == nil) {
                    gAsyncCommands = [gQueue commandBuffer];
                    if (gAsyncCommands == nil) return false;
                    beginBatchLocked();
                }
                if (gAsyncEncoder == nil) {
                    gAsyncEncoder = [gAsyncCommands computeCommandEncoder];
                    if (gAsyncEncoder == nil) {
                        gAsyncCommands = nil;
                        return false;
                    }
                }

                [gAsyncEncoder setComputePipelineState:pipeline];
                for (int i = 0; i < count; i++) markLocked(buffers[i]);
                for (int i = 0; i < count; i++)
                    [gAsyncEncoder setBuffer:(__bridge id<MTLBuffer>)buffers[i]->buffer
                                      offset:byteOffsets != NULL ? byteOffsets[i] : 0
                                     atIndex:(NSUInteger)i];
                for (int i = 0; i < scalarCount; i++)
                    [gAsyncEncoder setBytes:&scalars[i]
                                     length:sizeof scalars[i]
                                    atIndex:(NSUInteger)(count + i)];
                encodeDispatch(gAsyncEncoder, pipeline, (NSUInteger)threads,
                               (NSUInteger)groupSize);
                gOpenEncoded++;
                const int limit = autoCommitThreshold();
                if (limit > 0 && gOpenEncoded >= (unsigned)limit) {
                    commitOpenLocked();
                    if (gInFlight != nil && gInFlight.count > JAI_GPU_MAX_IN_FLIGHT) {
                        oldest = gInFlight[0];
                        oldestBatch = takeFrontLocked();
                    }
                }
            }
            /* The wait for the oldest happens outside the lock: it is the
             * backpressure that keeps the queue from growing without bound,
             * and holding the lock through it would stop every other thread
             * from encoding while this one sleeps. */
            if (oldest != nil) {
                [oldest waitUntilCompleted];
                if ([oldest status] != MTLCommandBufferStatusCompleted) return false;
                noteDone(oldestBatch);
            }
            return true;
        }

        /* A synchronous dispatch submitted after queued work must stay behind
         * it. Flush first, then use the low-overhead unretained command path. */
        if (!jaiGpuSynchronize()) return false;
        id<MTLCommandBuffer> commands = [gQueue commandBufferWithUnretainedReferences];
        if (commands == nil) return false;
        id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
        if (encoder == nil) return false;

        [encoder setComputePipelineState:pipeline];
        for (int i = 0; i < count; i++) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)buffers[i]->buffer
                        offset:byteOffsets != NULL ? byteOffsets[i] : 0
                       atIndex:(NSUInteger)i];
        }
        /* setBytes is the cheap path for an argument this small; it is exactly
         * what a `constant uint&` parameter binds against. */
        for (int i = 0; i < scalarCount; i++) {
            [encoder setBytes:&scalars[i]
                       length:sizeof scalars[i]
                      atIndex:(NSUInteger)(count + i)];
        }

        encodeDispatch(encoder, pipeline, (NSUInteger)threads,
                       (NSUInteger)groupSize);
        [encoder endEncoding];
        [commands commit];
        [commands waitUntilCompleted];
        meterNote(commands, 1u);
        return [commands status] == MTLCommandBufferStatusCompleted;
    }
}

bool jaiGpuDispatch(JaiGpuKernel *k, JaiGpuBuffer **buffers, int count,
                    const uint32_t *scalars, int scalarCount,
                    int threads, int groupSize, const size_t *byteOffsets) {
    return dispatchKernel(k, buffers, count, scalars, scalarCount,
                          threads, groupSize, byteOffsets, true);
}

bool jaiGpuDispatchAsync(JaiGpuKernel *k, JaiGpuBuffer **buffers, int count,
                         const uint32_t *scalars, int scalarCount,
                         int threads, int groupSize, const size_t *byteOffsets) {
    return dispatchKernel(k, buffers, count, scalars, scalarCount,
                          threads, groupSize, byteOffsets, false);
}

/* Commit whatever is open and file it under its batch number.
 *
 * The completion handler is what lets a batch be known finished without
 * anyone having waited for it: a buffer whose work is long done can then be
 * recycled and written to from the host with no wait at all. */
static void commitOpenLocked(void) {
    if (gAsyncCommands == nil) return;
    [gAsyncEncoder endEncoding];
    const uint64_t mine = gOpenBatch;
    const uint32_t encoded = gOpenEncoded;
    [gAsyncCommands addCompletedHandler:^(id<MTLCommandBuffer> done) {
        meterNote(done, encoded);
        noteDone(mine);
    }];
    [gAsyncCommands commit];
    ensureInFlight();
    [gInFlight addObject:gAsyncCommands];
    [gInFlightBatch addObject:@(gOpenBatch)];
    gAsyncEncoder = nil;
    gAsyncCommands = nil;
    gOpenBatch = 0;
    gOpenEncoded = 0;
}

/* Take the front of the queue off the books, returning the batch it holds.
 * The caller waits for it and then says so with noteDone: batches run in the
 * order they were committed, so one finishing means every earlier one has. */
static uint64_t takeFrontLocked(void) {
    const uint64_t batch = gInFlightBatch[0].unsignedLongLongValue;
    [gInFlight removeObjectAtIndex:0];
    [gInFlightBatch removeObjectAtIndex:0];
    return batch;
}

bool flushAsyncLocked(id<MTLCommandBuffer> *oldestOut, uint64_t *oldestBatch) {
    if (oldestOut != NULL) *oldestOut = nil;
    if (oldestBatch != NULL) *oldestBatch = 0;
    commitOpenLocked();
    if (gInFlight != nil && gInFlight.count > JAI_GPU_MAX_IN_FLIGHT) {
        if (oldestOut != NULL) {
            *oldestOut = gInFlight[0];
            const uint64_t batch = takeFrontLocked();
            if (oldestBatch != NULL) *oldestBatch = batch;
        }
    }
    return true;
}

bool jaiGpuFlush(void) {
    if (!ensureDevice()) return false;
    @autoreleasepool {
        id<MTLCommandBuffer> oldest = nil;
        uint64_t oldestBatch = 0;
        @synchronized(gQueue) {
            if (!commitMlpAccLocked()) return false;
            commitOpenLocked();
            if (gInFlight != nil && gInFlight.count > JAI_GPU_MAX_IN_FLIGHT) {
                oldest = gInFlight[0];
                oldestBatch = takeFrontLocked();
            }
        }
        if (oldest == nil) return true;
        [oldest waitUntilCompleted];
        if ([oldest status] != MTLCommandBufferStatusCompleted) return false;
        noteDone(oldestBatch);
        return true;
    }
}

bool jaiGpuSynchronize(void) {
    @autoreleasepool {
        @synchronized(gQueue) {
            if (!commitMlpAccLocked()) return false;
            if (!commitMlpWeightsLocked()) return false;
            if (gMlp3Side != 0) {
                if (!commitMlp3WeightsLocked()) return false;
            }
        }
    }
    if (!jaiGpuFlush()) return false;
    @autoreleasepool {
        NSArray<id<MTLCommandBuffer>> *pending;
        uint64_t newest = 0;
        @synchronized(gQueue) {
            if (gInFlight == nil || gInFlight.count == 0) return true;
            pending = [gInFlight copy];
            newest = gInFlightBatch.lastObject.unsignedLongLongValue;
            [gInFlight removeAllObjects];
            [gInFlightBatch removeAllObjects];
        }
        for (id<MTLCommandBuffer> commands in pending) {
            [commands waitUntilCompleted];
            if ([commands status] != MTLCommandBufferStatusCompleted) return false;
        }
        noteDone(newest);
        return true;
    }
}

/* Wait for the work that could have written `b`, and no more than that.
 *
 * Everything queued after it stays queued and keeps running while the caller
 * reads. That is what lets a loop hand the GPU the next frame's network before
 * reading this frame's result instead of after it.
 *
 * The staged MLP weights are the one thing a buffer number cannot describe --
 * they live in scratch that has to be committed as a set -- so a wait with any
 * of that outstanding falls back to draining the queue. */
bool jaiGpuWaitFor(JaiGpuBuffer *b) {
    if (b == NULL) return true;
    /* The other half of occupancy: an idle GPU and a blocked host are the two
     * ways a pipeline loses time, and only one of them shows up in the span
     * union above. */
    const double waitFrom = meterOn() ? meterNow() : 0.0;
    if (!fineSyncEnabled() || gQueue == nil) {
        const bool ok = jaiGpuSynchronize();
        if (waitFrom != 0.0) {
            meterHostWait((uint64_t)((meterNow() - waitFrom) * 1e9));
        }
        return ok;
    }

    uint64_t want = 0;
    @synchronized(gQueue) {
        if (gMlpSide != 0 || gMlpAccSide != 0 || gMlp3Side != 0) want = UINT64_MAX;
        else if (b->lastBatch > doneBatch()) want = b->lastBatch;
    }
    if (want == UINT64_MAX) {
        const bool ok = jaiGpuSynchronize();
        if (waitFrom != 0.0) meterHostWait((uint64_t)((meterNow() - waitFrom) * 1e9));
        return ok;
    }
    if (want == 0) return true;   /* nothing pending: not a wait */

    @autoreleasepool {
        NSArray<id<MTLCommandBuffer>> *pending = nil;
        uint64_t reached = 0;
        @synchronized(gQueue) {
            if (gOpenBatch != 0 && gOpenBatch <= want) commitOpenLocked();
            NSUInteger take = 0;
            for (NSUInteger i = 0; i < gInFlightBatch.count; i++) {
                take = i + 1;
                if (gInFlightBatch[i].unsignedLongLongValue >= want) break;
            }
            if (take > 0) {
                pending = [gInFlight subarrayWithRange:NSMakeRange(0, take)];
                reached = gInFlightBatch[take - 1].unsignedLongLongValue;
                [gInFlight removeObjectsInRange:NSMakeRange(0, take)];
                [gInFlightBatch removeObjectsInRange:NSMakeRange(0, take)];
            }
        }
        for (id<MTLCommandBuffer> commands in pending) {
            [commands waitUntilCompleted];
            if ([commands status] != MTLCommandBufferStatusCompleted) {
                if (waitFrom != 0.0) {
                    meterHostWait((uint64_t)((meterNow() - waitFrom) * 1e9));
                }
                return false;
            }
        }
        if (reached != 0) noteDone(reached);
    }
    if (waitFrom != 0.0) meterHostWait((uint64_t)((meterNow() - waitFrom) * 1e9));
    return true;
}

bool ensureAsyncCommandBuffer(void) {
    if (gAsyncEncoder != nil) {
        [gAsyncEncoder endEncoding];
        gAsyncEncoder = nil;
    }
    if (gAsyncCommands == nil) {
        gAsyncCommands = [gQueue commandBuffer];
        if (gAsyncCommands == nil) return false;
        beginBatchLocked();
    }
    return true;
}

#endif /* __APPLE__ */
