/* gpu_queue.m — the one open command buffer: batch numbering, in-flight
 * backpressure, flush and sync, and the kernels encoded into it. */

#ifdef __APPLE__

#include "native/apple/gpu_internal.h"

#include <errno.h>
#include <pthread.h>

id<MTLCommandBuffer> gAsyncCommands;
id<MTLComputeCommandEncoder> gAsyncEncoder;
static NSMutableArray<id<MTLCommandBuffer>> *gInFlight;

static NSMutableDictionary<NSString *, id<MTLLibrary>> *gSourceLibraries;

#define JAI_GPU_MAX_IN_FLIGHT 16
#define JAI_GPU_AUTO_COMMIT 0
/* See kernelCommitThreshold, batchCapUnits, batchCapSeconds and
 * waitTimeoutSeconds for what these bound and why. */
#define JAI_GPU_KERNEL_COMMIT 64
#define JAI_GPU_BATCH_UNITS 4096
#define JAI_GPU_BATCH_MS 250
#define JAI_GPU_WAIT_SECONDS 120.0

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
static unsigned gOpenGraphs;    /* MPSGraph encodes into it since it opened */
static bool gOpenHasGraph;      /* it holds at least one MPSGraph encode */
static int gAutoCommit = -1;
static int gKernelCommit = -1;
static int gBatchUnits = -1;
static double gBatchSeconds = -1.0;
static double gWaitSeconds = -1.0;
static int gEncoderStatus = -1;
/* GPU seconds per encoded unit (a dispatch or a graph encode), averaged over
 * finished batches. Written by Metal's completion thread, read by the
 * encoder, so it is an atomic; 0 until the first batch finishes. */
static _Atomic double gUnitSeconds;
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

/* Why the last wait failed, for the RuntimeError that reports it.
 *
 * A wait used to come back false on any status but Completed and drop
 * `commands.error` on the floor, so five failures in one session said only
 * "did not complete" and nothing could be learned from any of them. The text
 * is written from Metal's completion thread as well as from waiters, so it has
 * a lock of its own -- never the queue's, which a waiter may be holding. */
static pthread_mutex_t gErrorLock = PTHREAD_MUTEX_INITIALIZER;
static char gLastError[1024];
static char gLastErrorCopy[1024];
/* A batch failed and no caller has been told yet. Set by the completion
 * handler, so a failure in a batch nobody waits on individually (one retired
 * by backpressure inside an encode, say) still reaches the next synchronize. */
static atomic_bool gErrorPending;

static void recordGpuError(const char *fmt, ...) JAI_PRINTF(1, 2);
static void recordGpuError(const char *fmt, ...) {
    pthread_mutex_lock(&gErrorLock);
    va_list args;
    va_start(args, fmt);
    vsnprintf(gLastError, sizeof gLastError, fmt, args);
    va_end(args);
    pthread_mutex_unlock(&gErrorLock);
}

const char *jaiGpuLastError(void) {
    pthread_mutex_lock(&gErrorLock);
    memcpy(gLastErrorCopy, gLastError, sizeof gLastErrorCopy);
    pthread_mutex_unlock(&gErrorLock);
    return gLastErrorCopy;
}

/* Per-encoder status in a failed buffer's error costs Metal some bookkeeping
 * on every command buffer, so it is asked for only on request:
 * JAITHON_GPU_ENCODER_STATUS=1. */
static bool encoderStatusOn(void) {
    if (gEncoderStatus < 0) {
        const char *setting = getenv("JAITHON_GPU_ENCODER_STATUS");
        gEncoderStatus = setting != NULL && strcmp(setting, "0") != 0 ? 1 : 0;
    }
    return gEncoderStatus == 1;
}

id<MTLCommandBuffer> newAsyncCommandBuffer(void) {
    if (gQueue == nil) return nil;
    if (!encoderStatusOn()) return [gQueue commandBuffer];
    MTLCommandBufferDescriptor *desc = [MTLCommandBufferDescriptor new];
    desc.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
    return [gQueue commandBufferWithDescriptor:desc];
}

static const char *commandErrorName(NSInteger code) {
    switch (code) {
    case MTLCommandBufferErrorInternal:        return "internal error";
    case MTLCommandBufferErrorTimeout:         return "timeout";
    case MTLCommandBufferErrorPageFault:       return "page fault";
    case MTLCommandBufferErrorAccessRevoked:   return "access revoked";
    case MTLCommandBufferErrorNotPermitted:    return "not permitted";
    case MTLCommandBufferErrorOutOfMemory:     return "out of memory";
    case MTLCommandBufferErrorInvalidResource: return "invalid resource";
    case MTLCommandBufferErrorMemoryless:      return "memoryless";
    case MTLCommandBufferErrorDeviceRemoved:   return "device removed";
    case MTLCommandBufferErrorStackOverflow:   return "stack overflow";
    default:                                   return "unrecognised code";
    }
}

static const char *encoderStateName(MTLCommandEncoderErrorState state) {
    switch (state) {
    case MTLCommandEncoderErrorStateCompleted: return "completed";
    case MTLCommandEncoderErrorStateAffected:  return "affected";
    case MTLCommandEncoderErrorStatePending:   return "pending";
    case MTLCommandEncoderErrorStateFaulted:   return "faulted";
    default:                                   return "unknown";
    }
}

static const char *utf8Or(NSString *text, const char *fallback) {
    const char *raw = text != nil ? [text UTF8String] : NULL;
    return raw != NULL ? raw : fallback;
}

/* Domain, code, description and -- when JAITHON_GPU_ENCODER_STATUS was on as
 * the buffer was made -- which encoder faulted. */
static void recordCommandError(id<MTLCommandBuffer> done, uint64_t batch) {
    @autoreleasepool {
        NSError *error = done.error;
        NSMutableString *encoders = nil;
        id infos = error.userInfo[MTLCommandBufferEncoderInfoErrorKey];
        if ([infos isKindOfClass:[NSArray class]] && [(NSArray *)infos count] > 0) {
            encoders = [NSMutableString string];
            for (id<MTLCommandBufferEncoderInfo> info in (NSArray *)infos) {
                [encoders appendFormat:@"%@%@ %s", encoders.length > 0 ? @", " : @"",
                                       info.label.length > 0 ? info.label : @"(unlabelled)",
                                       encoderStateName(info.errorState)];
            }
        }
        recordGpuError("the Metal command buffer for batch %llu failed: %s code %ld (%s): %s%s%s",
                       (unsigned long long)batch,
                       error != nil ? utf8Or(error.domain, "?") : "no NSError",
                       error != nil ? (long)error.code : 0L,
                       error != nil ? commandErrorName(error.code) : "status error",
                       error != nil ? utf8Or(error.localizedDescription, "") : "",
                       encoders != nil ? "; encoders: "
                                       : (encoderStatusOn() ? ""
                                          : "; JAITHON_GPU_ENCODER_STATUS=1 adds which encoder failed"),
                       encoders != nil ? utf8Or(encoders, "") : "");
    }
}

/* How long a wait may go without its batch finishing before it gives up.
 *
 * waitUntilCompleted has no bound, and one run sat in it at 0% CPU for over
 * five minutes. A batch that takes this long is lost either way; an error
 * says so where a hang says nothing. The clock restarts for each buffer, so a
 * long queue that is making progress never trips it.
 * JAITHON_GPU_WAIT_TIMEOUT is in seconds; 0 waits forever. */
static double waitTimeoutSeconds(void) {
    if (gWaitSeconds < 0.0) {
        const char *setting = getenv("JAITHON_GPU_WAIT_TIMEOUT");
        gWaitSeconds = setting != NULL ? atof(setting) : JAI_GPU_WAIT_SECONDS;
        if (!(gWaitSeconds >= 0.0)) gWaitSeconds = JAI_GPU_WAIT_SECONDS;
    }
    return gWaitSeconds;
}

/* Every committed batch carries a completion handler that raises gDoneBatch
 * and then broadcasts here, so a waiter sleeps on a condition it can bound. */
static pthread_mutex_t gDoneLock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  gDoneCond = PTHREAD_COND_INITIALIZER;

static void signalDone(void) {
    pthread_mutex_lock(&gDoneLock);
    pthread_cond_broadcast(&gDoneCond);
    pthread_mutex_unlock(&gDoneLock);
}

/* JAITHON_GPU_DEBUG_STALL=1 makes the first committed batch wait on an event
 * nobody signals, so a test can watch a wait time out and report instead of
 * hanging. The event is signalled once the wait has given up, so the queue
 * drains and the process exits normally. Test-only; read once. */
static int gDebugStall = -1;
static id<MTLSharedEvent> gStallEvent;

static void injectStallLocked(id<MTLCommandBuffer> commands) {
    if (gDebugStall < 0) {
        const char *setting = getenv("JAITHON_GPU_DEBUG_STALL");
        gDebugStall = setting != NULL && strcmp(setting, "0") != 0 ? 1 : 0;
    }
    if (gDebugStall != 1 || gStallEvent != nil) return;
    gStallEvent = [gDevice newSharedEvent];
    if (gStallEvent == nil) return;
    [commands encodeWaitForEvent:gStallEvent value:1];
}

static void releaseStall(void) {
    if (gStallEvent != nil) gStallEvent.signaledValue = 1;
}

static bool waitBatch(id<MTLCommandBuffer> commands, uint64_t batch) {
    if (commands == nil) return true;
    if (doneBatch() < batch) {
        const double limit = waitTimeoutSeconds();
        bool timedOut = false;
        pthread_mutex_lock(&gDoneLock);
        if (limit <= 0.0) {
            while (doneBatch() < batch) pthread_cond_wait(&gDoneCond, &gDoneLock);
        } else {
            struct timespec deadline;
            clock_gettime(CLOCK_REALTIME, &deadline);
            const double whole = floor(limit);
            deadline.tv_sec += (time_t)whole;
            deadline.tv_nsec += (long)((limit - whole) * 1e9);
            if (deadline.tv_nsec >= 1000000000L) {
                deadline.tv_sec += 1;
                deadline.tv_nsec -= 1000000000L;
            }
            while (doneBatch() < batch) {
                if (pthread_cond_timedwait(&gDoneCond, &gDoneLock, &deadline) == ETIMEDOUT) {
                    timedOut = doneBatch() < batch;
                    break;
                }
            }
        }
        pthread_mutex_unlock(&gDoneLock);
        if (timedOut) {
            const MTLCommandBufferStatus status = [commands status];
            recordGpuError("GPU work queued in batch %llu did not finish within %g s "
                           "(command buffer status %ld); the device is hung or heavily "
                           "contended. JAITHON_GPU_WAIT_TIMEOUT sets the limit in seconds",
                           (unsigned long long)batch, limit, (long)status);
            releaseStall();
            return false;
        }
    }
    /* The batch is known finished, by its own handler or a later one's. A
     * later one's runs first only in principle, and then this returns at once;
     * it is here so the status read below is final. */
    [commands waitUntilCompleted];
    if ([commands status] == MTLCommandBufferStatusCompleted) return true;
    recordCommandError(commands, batch);
    atomic_store(&gErrorPending, false);
    return false;
}

/* A failure no wait has reported yet: one retired by backpressure inside an
 * encode, which has no caller to tell. */
static bool takePendingError(void) {
    return atomic_exchange(&gErrorPending, false);
}

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

/* The half of auto-commit that is a good idea: split a batch that holds only
 * kernels.
 *
 * What cost jaitensor 26% was splitting MPSGraph work, which loses the
 * overlap MPSGraph arranges inside one buffer. A batch of plain kernels has no
 * such overlap to lose, and committing it every so many dispatches buys two
 * things: the GPU starts on the front of a chain while the host writes the
 * back, and the buffers the chain frees settle, so the pool hands them back
 * instead of allocating (10.4 us an op at N=1024 against 3.1 with a commit
 * every 64). So a batch is committed after this many kernel dispatches, but
 * only while no graph has been encoded into it -- noteGraphEncodeLocked turns
 * that off for the rest of the batch.
 * JAITHON_GPU_KERNEL_COMMIT sets the count; 0 turns it off. */
static int kernelCommitThreshold(void) {
    if (gKernelCommit < 0) {
        const char *setting = getenv("JAITHON_GPU_KERNEL_COMMIT");
        gKernelCommit = setting != NULL ? atoi(setting) : JAI_GPU_KERNEL_COMMIT;
        if (gKernelCommit < 0) gKernelCommit = 0;
    }
    return gKernelCommit;
}

/* The most one command buffer may hold, whatever it holds.
 *
 * A loop that never reads queues everything into one buffer: one warm-up loop
 * put 36,947 dispatches and 10.8 s of GPU work into a single command buffer,
 * and that is the shape every "queued GPU work did not complete" in one
 * session had. So a batch is committed once it holds this many dispatches and
 * graph encodes (a training step is about 450), or once its estimated GPU time
 * passes batchCapSeconds, and backpressure then keeps at most
 * JAI_GPU_MAX_IN_FLIGHT of them queued. Only loops that never wait reach
 * either bound. JAITHON_GPU_BATCH_CAP sets the count; 0 turns the cap off. */
static int batchCapUnits(void) {
    if (gBatchUnits < 0) {
        const char *setting = getenv("JAITHON_GPU_BATCH_CAP");
        gBatchUnits = setting != NULL ? atoi(setting) : JAI_GPU_BATCH_UNITS;
        if (gBatchUnits < 0) gBatchUnits = 0;
    }
    return gBatchUnits;
}

/* Estimated from gUnitSeconds, the GPU time per unit of the batches that have
 * finished. JAITHON_GPU_BATCH_MS sets it in milliseconds; 0 turns it off. */
static double batchCapSeconds(void) {
    if (gBatchSeconds < 0.0) {
        const char *setting = getenv("JAITHON_GPU_BATCH_MS");
        gBatchSeconds = (setting != NULL ? atof(setting) : (double)JAI_GPU_BATCH_MS) / 1000.0;
        if (!(gBatchSeconds >= 0.0)) gBatchSeconds = 0.0;
    }
    return gBatchSeconds;
}

/* Whether the open batch should go now. `afterGraph` keeps the two kernel
 * thresholds to the dispatch site they apply at. */
static bool shouldCommitLocked(bool afterGraph) {
    if (gAsyncCommands == nil) return false;
    if (!afterGraph) {
        const int global = autoCommitThreshold();
        if (global > 0 && gOpenEncoded >= (unsigned)global) return true;
        const int kernels = kernelCommitThreshold();
        if (kernels > 0 && !gOpenHasGraph && gOpenEncoded >= (unsigned)kernels) return true;
    }
    const unsigned units = gOpenEncoded + gOpenGraphs;
    const int cap = batchCapUnits();
    if (cap > 0 && units >= (unsigned)cap) return true;
    const double capSeconds = batchCapSeconds();
    if (capSeconds > 0.0) {
        const double perUnit = atomic_load_explicit(&gUnitSeconds, memory_order_relaxed);
        if (perUnit > 0.0 && perUnit * (double)units >= capSeconds) return true;
    }
    return false;
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
    b->untouched = false;
    if (gOpenBatch == 0) {
        if (gAsyncCommands == nil) {
            gAsyncCommands = newAsyncCommandBuffer();
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
                    gAsyncCommands = newAsyncCommandBuffer();
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
                if (shouldCommitLocked(false)) {
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
                if (!waitBatch(oldest, oldestBatch)) return false;
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
        /* Off the batch books, so it is bounded by a semaphore of its own
         * rather than by gDoneBatch. */
        dispatch_semaphore_t finished = dispatch_semaphore_create(0);
        [commands addCompletedHandler:^(id<MTLCommandBuffer> done) {
            (void)done;
            dispatch_semaphore_signal(finished);
        }];
        [commands commit];
        const double limit = waitTimeoutSeconds();
        const dispatch_time_t until = limit > 0.0
            ? dispatch_time(DISPATCH_TIME_NOW, (int64_t)(limit * 1e9))
            : DISPATCH_TIME_FOREVER;
        if (dispatch_semaphore_wait(finished, until) != 0) {
            recordGpuError("a synchronous dispatch did not finish within %g s; the "
                           "device is hung or heavily contended", limit);
            return false;
        }
        [commands waitUntilCompleted];
        meterNote(commands, 1u);
        if ([commands status] == MTLCommandBufferStatusCompleted) return true;
        recordCommandError(commands, 0);
        return false;
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
    injectStallLocked(gAsyncCommands);
    const uint64_t mine = gOpenBatch;
    const uint32_t encoded = gOpenEncoded;
    const uint32_t units = gOpenEncoded + gOpenGraphs;
    [gAsyncCommands addCompletedHandler:^(id<MTLCommandBuffer> done) {
        meterNote(done, encoded);
        if ([done status] == MTLCommandBufferStatusError) {
            recordCommandError(done, mine);
            atomic_store(&gErrorPending, true);
        } else if (units > 0) {
            /* A swap MPSGraph made part way leaves only the tail's time on
             * this buffer, so this can only under-estimate; the dispatch cap
             * is the backstop for that. */
            const double took = done.GPUEndTime - done.GPUStartTime;
            if (took > 0.0) {
                const double perUnit = took / (double)units;
                const double seen = atomic_load_explicit(&gUnitSeconds, memory_order_relaxed);
                atomic_store_explicit(&gUnitSeconds,
                                      seen > 0.0 ? 0.75 * seen + 0.25 * perUnit : perUnit,
                                      memory_order_relaxed);
            }
        }
        /* Raised before the broadcast, so a woken waiter sees it. */
        noteDone(mine);
        signalDone();
    }];
    [gAsyncCommands commit];
    ensureInFlight();
    [gInFlight addObject:gAsyncCommands];
    [gInFlightBatch addObject:@(gOpenBatch)];
    gAsyncEncoder = nil;
    gAsyncCommands = nil;
    gOpenBatch = 0;
    gOpenEncoded = 0;
    gOpenGraphs = 0;
    gOpenHasGraph = false;
}

void noteGraphEncodeLocked(void) {
    gOpenHasGraph = true;
    gOpenGraphs++;
}

/* Inside the queue lock and after the graph is in the batch, so every buffer
 * the caller marked for it names the batch that is committed here. The wait
 * for the oldest is taken inside the lock: there is no caller to hand a
 * failure to, so it stays pending for the next synchronize. */
void afterGraphEncodeLocked(void) {
    if (!shouldCommitLocked(true)) return;
    commitOpenLocked();
    if (gInFlight != nil && gInFlight.count > JAI_GPU_MAX_IN_FLIGHT) {
        id<MTLCommandBuffer> oldest = gInFlight[0];
        const uint64_t batch = takeFrontLocked();
        if (waitBatch(oldest, batch)) {
            noteDone(batch);
        } else {
            atomic_store(&gErrorPending, true);
        }
    }
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
    if (!ensureDevice()) {
        recordGpuError("no Metal device is available");
        return false;
    }
    @autoreleasepool {
        id<MTLCommandBuffer> oldest = nil;
        uint64_t oldestBatch = 0;
        @synchronized(gQueue) {
            if (!commitMlpAccLocked()) {
                recordGpuError("the fused MLP's staged accumulators could not be committed");
                return false;
            }
            commitOpenLocked();
            if (gInFlight != nil && gInFlight.count > JAI_GPU_MAX_IN_FLIGHT) {
                oldest = gInFlight[0];
                oldestBatch = takeFrontLocked();
            }
        }
        if (oldest == nil) return true;
        if (!waitBatch(oldest, oldestBatch)) return false;
        noteDone(oldestBatch);
        return true;
    }
}

bool jaiGpuSynchronize(void) {
    @autoreleasepool {
        @synchronized(gQueue) {
            bool staged = commitMlpAccLocked() && commitMlpWeightsLocked();
            if (staged && gMlp3Side != 0) staged = commitMlp3WeightsLocked();
            if (!staged) {
                recordGpuError("the fused MLP's staged weights could not be committed");
                return false;
            }
        }
    }
    if (!jaiGpuFlush()) return false;
    @autoreleasepool {
        NSArray<id<MTLCommandBuffer>> *pending = nil;
        NSArray<NSNumber *> *batches = nil;
        @synchronized(gQueue) {
            if (gInFlight != nil && gInFlight.count > 0) {
                pending = [gInFlight copy];
                batches = [gInFlightBatch copy];
                [gInFlight removeAllObjects];
                [gInFlightBatch removeAllObjects];
            }
        }
        for (NSUInteger i = 0; i < pending.count; i++) {
            const uint64_t batch = batches[i].unsignedLongLongValue;
            if (!waitBatch(pending[i], batch)) return false;
            noteDone(batch);
        }
        if (takePendingError()) return false;
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
        NSArray<NSNumber *> *batches = nil;
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
                batches = [gInFlightBatch subarrayWithRange:NSMakeRange(0, take)];
                reached = gInFlightBatch[take - 1].unsignedLongLongValue;
                [gInFlight removeObjectsInRange:NSMakeRange(0, take)];
                [gInFlightBatch removeObjectsInRange:NSMakeRange(0, take)];
            }
        }
        for (NSUInteger i = 0; i < pending.count; i++) {
            if (!waitBatch(pending[i], batches[i].unsignedLongLongValue)) {
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
        gAsyncCommands = newAsyncCommandBuffer();
        if (gAsyncCommands == nil) return false;
        beginBatchLocked();
    }
    return true;
}

#endif /* __APPLE__ */
