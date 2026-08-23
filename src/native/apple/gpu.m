/* gpu.m — Metal compute: device buffers, kernels compiled from MSL, and the
 * five built-in operations std.gpu falls back on when no source is supplied.
 *
 * MSL has no `double`: buffers hold float32, built-ins round going in and
 * widen coming out; the CPU paths stay double, so GPU and CPU agree to a
 * tolerance, never to the bit (lib/std/gpu.jai documents this contract).
 *
 * Every built-in is total: it's correct with no device, a device too small
 * for the buffers, or a Metal call failing midway — each give-up path falls
 * back to the same scalar code the no-device build uses. */

#ifdef __APPLE__

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdatomic.h>
#include <string.h>

#include "native/native.h"
#include "runtime/parallel.h"

#include <stdlib.h>
static _Atomic long gDispatchTraceKernel = 0;
static _Atomic long gDispatchTraceGraph = 0;
static int gDispatchTraceOn = -1;
static void dispatchTraceDump(void) {
    fprintf(stderr, "JAI_GPU_DISPATCH_TRACE kernel=%ld graph=%ld total=%ld\n",
            gDispatchTraceKernel, gDispatchTraceGraph, gDispatchTraceKernel + gDispatchTraceGraph);
}
static void dispatchTraceTick(int isGraph) {
    if (gDispatchTraceOn < 0) {
        gDispatchTraceOn = getenv("JAI_GPU_DISPATCH_TRACE") != NULL ? 1 : 0;
        if (gDispatchTraceOn) atexit(dispatchTraceDump);
    }
    if (!gDispatchTraceOn) return;
    if (isGraph) gDispatchTraceGraph++; else gDispatchTraceKernel++;
}

/* Below this much arithmetic the upload, encode and queue wait cost more than
 * the work. Mirrors MIN_GPU_ELEMENTS in lib/std/gpu.jai. */
#define JAI_GPU_MIN_WORK 4096

/* The reduction kernel's threadgroup scratch is this width, so the host must
 * dispatch with exactly this group size. 256 is within every Metal device's guaranteed maximum. */
#define JAI_REDUCE_GROUP 256
#define JAI_REDUCE_LOADS 2
#define JAI_MATMUL_TILE 16
#define JAI_VECTOR_LANES 4
#define JAI_DEFAULT_GROUP 256

struct JaiGpuBuffer {
    void  *buffer;   /* id<MTLBuffer>, held at +1 */
    size_t bytes;
    /* Set when this came out of the recycling pool and the queue has not been
     * waited on since. Work already queued may still write these bytes, so the
     * first HOST write has to wait for it -- see hostWriteBarrier. */
    bool   recycled;
    /* The last batch of queued work that could have written these bytes; zero
     * when none has. Reading the buffer waits for that batch and no further --
     * see markLocked and jaiGpuWaitFor. */
    uint64_t lastBatch;
};

struct JaiGpuKernel {
    void *pipeline;  /* id<MTLComputePipelineState>, held at +1 */
};

static const char kBuiltinSource[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "\n"
    "constant uint JAI_GROUP = 256;\n"
    "constant uint JAI_TILE = 16;\n"
    "constant uint JAI_VECTOR_LANES = 4;\n"
    "\n"
    "kernel void jaiVectorAdd(device const float *a [[buffer(0)]],\n"
    "                         device const float *b [[buffer(1)]],\n"
    "                         device float *out [[buffer(2)]],\n"
    "                         constant uint &n [[buffer(3)]],\n"
    "                         uint tid [[thread_position_in_grid]]) {\n"
    "    uint base = tid * JAI_VECTOR_LANES;\n"
    "    if (base >= n) return;\n"
    "    out[base] = a[base] + b[base];\n"
    "    if (base + 1 < n) out[base + 1] = a[base + 1] + b[base + 1];\n"
    "    if (base + 2 < n) out[base + 2] = a[base + 2] + b[base + 2];\n"
    "    if (base + 3 < n) out[base + 3] = a[base + 3] + b[base + 3];\n"
    "}\n"
    "\n"
    "kernel void jaiVectorMul(device const float *a [[buffer(0)]],\n"
    "                         device const float *b [[buffer(1)]],\n"
    "                         device float *out [[buffer(2)]],\n"
    "                         constant uint &n [[buffer(3)]],\n"
    "                         uint tid [[thread_position_in_grid]]) {\n"
    "    uint base = tid * JAI_VECTOR_LANES;\n"
    "    if (base >= n) return;\n"
    "    out[base] = a[base] * b[base];\n"
    "    if (base + 1 < n) out[base + 1] = a[base + 1] * b[base + 1];\n"
    "    if (base + 2 < n) out[base + 2] = a[base + 2] * b[base + 2];\n"
    "    if (base + 3 < n) out[base + 3] = a[base + 3] * b[base + 3];\n"
    "}\n"
    "\n"
    "// 16x16 tiled row-major matrix multiply. Each A/B tile is loaded once\n"
    "// from device memory and reused by all 256 threads in the threadgroup.\n"
    "kernel void jaiMatMul(device const float *a [[buffer(0)]],\n"
    "                      device const float *b [[buffer(1)]],\n"
    "                      device float *out [[buffer(2)]],\n"
    "                      constant uint &rows [[buffer(3)]],\n"
    "                      constant uint &inner [[buffer(4)]],\n"
    "                      constant uint &columns [[buffer(5)]],\n"
    "                      uint2 lid [[thread_position_in_threadgroup]],\n"
    "                      uint2 group [[threadgroup_position_in_grid]]) {\n"
    "    threadgroup float tileA[JAI_TILE * JAI_TILE];\n"
    "    threadgroup float tileB[JAI_TILE * JAI_TILE];\n"
    "\n"
    "    uint row = group.y * JAI_TILE + lid.y;\n"
    "    uint col = group.x * JAI_TILE + lid.x;\n"
    "    uint local = lid.y * JAI_TILE + lid.x;\n"
    "    float total = 0.0f;\n"
    "\n"
    "    for (uint base = 0; base < inner; base += JAI_TILE) {\n"
    "        uint aCol = base + lid.x;\n"
    "        uint bRow = base + lid.y;\n"
    "        tileA[local] = (row < rows && aCol < inner)\n"
    "                         ? a[row * inner + aCol] : 0.0f;\n"
    "        tileB[local] = (bRow < inner && col < columns)\n"
    "                         ? b[bRow * columns + col] : 0.0f;\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "\n"
    "        for (uint t = 0; t < JAI_TILE; ++t) {\n"
    "            total += tileA[lid.y * JAI_TILE + t] *\n"
    "                     tileB[t * JAI_TILE + lid.x];\n"
    "        }\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    }\n"
    "\n"
    "    if (row < rows && col < columns)\n"
    "        out[row * columns + col] = total;\n"
    "}\n"
    "\n"
    "// Two input values per thread halves the number of reduction groups and\n"
    "// partials while retaining the fixed 256-thread scratch layout.\n"
    "kernel void jaiReduceSum(device const float *x [[buffer(0)]],\n"
    "                         device float *partials [[buffer(1)]],\n"
    "                         constant uint &n [[buffer(2)]],\n"
    "                         uint lid [[thread_index_in_threadgroup]],\n"
    "                         uint wid [[threadgroup_position_in_grid]]) {\n"
    "    threadgroup float scratch[JAI_GROUP];\n"
    "    uint base = wid * (JAI_GROUP * 2) + lid;\n"
    "    float sum = base < n ? x[base] : 0.0f;\n"
    "    uint second = base + JAI_GROUP;\n"
    "    if (second < n) sum += x[second];\n"
    "    scratch[lid] = sum;\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "\n"
    "    for (uint stride = JAI_GROUP / 2; stride > 0; stride >>= 1) {\n"
    "        if (lid < stride) scratch[lid] += scratch[lid + stride];\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    }\n"
    "    if (lid == 0) partials[wid] = scratch[0];\n"
    "}\n"
    "\n"
    "kernel void jaiExpandU8(device const uchar *src [[buffer(0)]],\n"
    "                        device float *dst [[buffer(1)]],\n"
    "                        constant uint &n [[buffer(2)]],\n"
    "                        constant float &scale [[buffer(3)]],\n"
    "                        uint tid [[thread_position_in_grid]]) {\n"
    "    if (tid >= n) return;\n"
    "    dst[tid] = (float)src[tid] * scale;\n"
    "}\n"
    "\n"
    "kernel void jaiSplitHeads(device const float *input [[buffer(0)]],\n"
    "                          device float *output [[buffer(1)]],\n"
    "                          constant uint &seq [[buffer(2)]],\n"
    "                          constant uint &heads [[buffer(3)]],\n"
    "                          constant uint &hd [[buffer(4)]],\n"
    "                          uint id [[thread_position_in_grid]]) {\n"
    "    uint vecs = hd / 4;\n"
    "    uint count = heads * seq * vecs;\n"
    "    if (id >= count) return;\n"
    "    uint d4 = id % vecs;\n"
    "    uint s = (id / vecs) % seq;\n"
    "    uint h = id / (vecs * seq);\n"
    "    uint dim = heads * hd;\n"
    "    device const float4 *in4 =\n"
    "        (device const float4 *)(input + s * dim + h * hd);\n"
    "    device float4 *out4 =\n"
    "        (device float4 *)(output + (h * seq + s) * hd);\n"
    "    out4[d4] = in4[d4];\n"
    "}\n"
    "\n"
    "kernel void jaiMergeHeads(device const float *input [[buffer(0)]],\n"
    "                          device float *output [[buffer(1)]],\n"
    "                          constant uint &seq [[buffer(2)]],\n"
    "                          constant uint &heads [[buffer(3)]],\n"
    "                          constant uint &hd [[buffer(4)]],\n"
    "                          uint id [[thread_position_in_grid]]) {\n"
    "    uint vecs = hd / 4;\n"
    "    uint count = heads * seq * vecs;\n"
    "    if (id >= count) return;\n"
    "    uint d4 = id % vecs;\n"
    "    uint s = (id / vecs) % seq;\n"
    "    uint h = id / (vecs * seq);\n"
    "    uint dim = heads * hd;\n"
    "    device const float4 *in4 =\n"
    "        (device const float4 *)(input + (h * seq + s) * hd);\n"
    "    device float4 *out4 =\n"
    "        (device float4 *)(output + s * dim + h * hd);\n"
    "    out4[d4] = in4[d4];\n"
    "}\n";

static id<MTLDevice>       gDevice;
static id<MTLCommandQueue> gQueue;
static bool                gDeviceReady;
static int                 gPreferredDevice = -1;
static bool                gMixedPrecision;
static bool                gNonUniformThreadgroups;
static size_t              gMaxBufferLength;

static id<MTLComputePipelineState> gVectorAdd;
static id<MTLComputePipelineState> gVectorMul;
static id<MTLComputePipelineState> gMatMul;
static id<MTLComputePipelineState> gReduceSum;
static id<MTLComputePipelineState> gExpandU8;
static id<MTLComputePipelineState> gSplitHeads;
static id<MTLComputePipelineState> gMergeHeads;
static id<MTLComputePipelineState> gFlashAttn32;
static id<MTLComputePipelineState> gFlashAttn64;
static id<MTLComputePipelineState> gFlashPack;
static id<MTLBuffer> gMhaHalfScratch;
static size_t gMhaHalfCap;
static id<MTLCommandBuffer> gAsyncCommands;
static id<MTLComputeCommandEncoder> gAsyncEncoder;
static NSMutableArray<id<MTLCommandBuffer>> *gInFlight;
static NSMutableDictionary<NSString *, id<MTLLibrary>> *gSourceLibraries;
static bool                        gBuiltinsReady;
static id<MTLBuffer> gMlpScratchW1, gMlpScratchB1, gMlpScratchW2, gMlpScratchB2;
static id<MTLBuffer> gMlpScratchAcc, gMlpScratchCorrect;
static size_t gMlpCapW1, gMlpCapB1, gMlpCapW2, gMlpCapB2, gMlpCapAcc, gMlpCapCorrect;
static int gMlpSide;
static int gMlpAccSide;
static JaiGpuBuffer *gMlpLiveW1, *gMlpLiveB1, *gMlpLiveW2, *gMlpLiveB2;
static JaiGpuBuffer *gMlpLiveAcc;
static size_t gMlpLiveW1Off, gMlpLiveB1Off, gMlpLiveW2Off, gMlpLiveB2Off;
static size_t gMlpLiveAccOff;
static size_t gMlpLiveW1Bytes, gMlpLiveB1Bytes, gMlpLiveW2Bytes, gMlpLiveB2Bytes;
static id<MTLBuffer> gMlp3ScratchW[4];
static id<MTLBuffer> gMlp3ScratchB[4];
static size_t gMlp3CapW[4];
static size_t gMlp3CapB[4];
static int gMlp3Side;
static JaiGpuBuffer *gMlp3LiveW[4];
static JaiGpuBuffer *gMlp3LiveB[4];
static size_t gMlp3LiveWOff[4];
static size_t gMlp3LiveBOff[4];
static size_t gMlp3LiveWBytes[4];
static size_t gMlp3LiveBBytes[4];

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

static uint64_t doneBatch(void) {
    return atomic_load_explicit(&gDoneBatch, memory_order_acquire);
}

/* Raise the finished mark to `batch`, never lower it. */
static void noteDone(uint64_t batch) {
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
static void beginBatchLocked(void) {
    gOpenBatch = ++gBatchCounter;
}

/* Note that whatever is being encoded now may write `b`.
 *
 * When nothing is open yet the batch is opened here, so that the number a
 * buffer records always belongs to a command buffer that will really be
 * committed -- a number guessed ahead of one would come loose if the work went
 * somewhere else instead. */
static void markLocked(JaiGpuBuffer *b) {
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

static void mark(JaiGpuBuffer *b) {
    if (b == NULL || gQueue == nil) return;
    @synchronized(gQueue) {
        markLocked(b);
    }
}

/* For graphbuild.m and coreml.m, which reach past the JaiGpuBuffer for the
 * MTLBuffer inside it and encode against that directly. */
void jaiGpuBufferMark(JaiGpuBuffer *b) { mark(b); }

static bool ensureDevice(void) {
    static dispatch_once_t once;

    dispatch_once(&once, ^{
        @autoreleasepool {
            id<MTLDevice> device = nil;
            if (gPreferredDevice >= 0) {
                NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
                if (devices != nil && gPreferredDevice < (int)devices.count) {
                    device = devices[(NSUInteger)gPreferredDevice];
                }
            }
            if (device == nil) device = MTLCreateSystemDefaultDevice();
            if (device == nil) return;

            id<MTLCommandQueue> queue = [device newCommandQueue];
            if (queue == nil) return;

            gDevice = device;
            gQueue = queue;
            gMaxBufferLength = (size_t)[device maxBufferLength];

            /* Checks the older Apple4/Mac2 capability (not just family 7+) to
             * stay valid across the widest set of Metal-capable hardware. */
            gNonUniformThreadgroups =
                [device supportsFamily:MTLGPUFamilyApple4] ||
                [device supportsFamily:MTLGPUFamilyMac2];

            gDeviceReady = true;
        }
    });

    return gDeviceReady;
}

/* Compiling the built-in library costs milliseconds, so it waits until one of
 * the built-ins is actually called rather than happening at device setup. */
static bool ensureBuiltins(void) {
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        if (!ensureDevice()) return;
        @autoreleasepool {
            NSError *error = nil;
            id<MTLLibrary> library = [gDevice newLibraryWithSource:@(kBuiltinSource)
                                                           options:nil
                                                             error:&error];
            if (library == nil) return;

            NSArray<NSString *> *names =
                @[@"jaiVectorAdd", @"jaiVectorMul", @"jaiMatMul", @"jaiReduceSum",
                  @"jaiExpandU8", @"jaiSplitHeads", @"jaiMergeHeads"];
            id<MTLComputePipelineState> built[7] = {nil, nil, nil, nil, nil, nil, nil};
            for (NSUInteger i = 0; i < 7; i++) {
                id<MTLFunction> fn = [library newFunctionWithName:names[i]];
                if (fn == nil) return;
                built[i] = [gDevice newComputePipelineStateWithFunction:fn error:&error];
                if (built[i] == nil) return;
            }

            gVectorAdd = built[0];
            gVectorMul = built[1];
            gMatMul = built[2];
            gReduceSum = built[3];
            gExpandU8 = built[4];
            gSplitHeads = built[5];
            gMergeHeads = built[6];
            gBuiltinsReady = true;
        }
    });
    return gBuiltinsReady;
}

/* Tiled FlashAttention-2 with simdgroup 8x8 MMA. Reads packed [seq, heads*hd]
 * directly so long sequences stay compute-bound instead of paying graph
 * transposes. Falls back to MPSGraph SDPA when the library does not compile. */
static const char kFlashAttnSource[] =
    "#include <metal_stdlib>\n"
    "#include <metal_simdgroup_matrix>\n"
    "using namespace metal;\n"
    "\n"
    "/* Packed [seq, heads*hd] float → BHSD [heads, seq, hd] half. One pass over\n"
    " * Q, K, and V so the prefill kernel streams contiguous half K/V instead of\n"
    " * converting fp32 on every tile. */\n"
    "kernel void jaiPackMhaHalf(device const float *Q [[buffer(0)]],\n"
    "                           device const float *K [[buffer(1)]],\n"
    "                           device const float *V [[buffer(2)]],\n"
    "                           device half *Qh [[buffer(3)]],\n"
    "                           device half *Kh [[buffer(4)]],\n"
    "                           device half *Vh [[buffer(5)]],\n"
    "                           constant uint &seq [[buffer(6)]],\n"
    "                           constant uint &heads [[buffer(7)]],\n"
    "                           constant uint &hd [[buffer(8)]],\n"
    "                           uint id [[thread_position_in_grid]]) {\n"
    "    uint vecs = hd / 4u;\n"
    "    uint count = heads * seq * vecs;\n"
    "    if (id >= count) return;\n"
    "    uint d4 = id % vecs;\n"
    "    uint s = (id / vecs) % seq;\n"
    "    uint h = id / (vecs * seq);\n"
    "    uint packed = s * (heads * hd) + h * hd + d4 * 4u;\n"
    "    uint bhsd = (h * seq + s) * hd + d4 * 4u;\n"
    "    float4 q = *reinterpret_cast<device const float4 *>(Q + packed);\n"
    "    float4 k = *reinterpret_cast<device const float4 *>(K + packed);\n"
    "    float4 v = *reinterpret_cast<device const float4 *>(V + packed);\n"
    "    *reinterpret_cast<device half4 *>(Qh + bhsd) = half4(q);\n"
    "    *reinterpret_cast<device half4 *>(Kh + bhsd) = half4(k);\n"
    "    *reinterpret_cast<device half4 *>(Vh + bhsd) = half4(v);\n"
    "}\n"
    "\n"
    "inline uint2 frag_coord(uint lane) {\n"
    "    uint qid = lane / 4u;\n"
    "    uint row = (qid & 4u) + ((lane / 2u) % 4u);\n"
    "    uint col = (qid & 2u) * 2u + (lane % 2u) * 2u;\n"
    "    return uint2(col, row);\n"
    "}\n"
    "\n"
    "#define PV_D(ACC, S, KOFF, DOFF) \\\n"
    "    simdgroup_load(vmat, KVs + (KOFF) * LDV + (DOFF), LDV); \\\n"
    "    simdgroup_multiply_accumulate(ACC, S, vmat, ACC);\n"
    "\n"
    "#define PV_K(S, KOFF, HD_) \\\n"
    "    PV_D(a0, S, KOFF, 0) \\\n"
    "    PV_D(a1, S, KOFF, 8) \\\n"
    "    PV_D(a2, S, KOFF, 16) \\\n"
    "    PV_D(a3, S, KOFF, 24) \\\n"
    "    if ((HD_) > 32) { \\\n"
    "        PV_D(a4, S, KOFF, 32) \\\n"
    "        PV_D(a5, S, KOFF, 40) \\\n"
    "        PV_D(a6, S, KOFF, 48) \\\n"
    "        PV_D(a7, S, KOFF, 56) \\\n"
    "    }\n"
    "\n"
    "#define PREFILL_KERNEL(NAME, HD) \\\n"
    "kernel void NAME(device const half *Q [[buffer(0)]], \\\n"
    "                 device const half *K [[buffer(1)]], \\\n"
    "                 device const half *V [[buffer(2)]], \\\n"
    "                 device float *Y [[buffer(3)]], \\\n"
    "                 constant uint &seq [[buffer(4)]], \\\n"
    "                 constant uint &heads [[buffer(5)]], \\\n"
    "                 constant float &scale [[buffer(6)]], \\\n"
    "                 uint lid [[thread_index_in_threadgroup]], \\\n"
    "                 uint2 tgpig [[threadgroup_position_in_grid]], \\\n"
    "                 uint sgitg [[simdgroup_index_in_threadgroup]]) { \\\n"
    "    constexpr uint BR = 64; \\\n"
    "    constexpr uint BC = 32; \\\n"
    "    constexpr uint LDQ = HD + 8; \\\n"
    "    constexpr uint LDK = BC + 8; \\\n"
    "    constexpr uint LDV = HD + 8; \\\n"
    "    constexpr uint KV0 = LDK * HD; \\\n"
    "    constexpr uint KV1 = BC * LDV; \\\n"
    "    constexpr uint KVN = KV0 > KV1 ? KV0 : KV1; \\\n"
    "    constexpr uint Q_TCOLS = 4; \\\n"
    "    constexpr uint Q_NREADS = HD / Q_TCOLS; \\\n"
    "    constexpr uint KV_TCOLS = 8; \\\n"
    "    constexpr uint KV_NREADS = HD / KV_TCOLS; \\\n"
    "    const uint q0 = tgpig.x * BR; \\\n"
    "    const uint head = tgpig.y; \\\n"
    "    const uint dim = heads * HD; \\\n"
    "    const uint qbase = head * HD; \\\n"
    "    const uint head_off = head * seq * HD; \\\n"
    "    const uint row0 = sgitg * 8; \\\n"
    "    const uint lane = lid & 31u; \\\n"
    "    const uint2 coord = frag_coord(lane); \\\n"
    "    const float scale2 = scale * 1.4426950408889634f; \\\n"
    "    threadgroup half Qs[BR * LDQ]; \\\n"
    "    threadgroup half KVs[KVN]; \\\n"
    "    const uint qrow_l = lid / Q_TCOLS; \\\n"
    "    const uint qd0 = (lid % Q_TCOLS) * Q_NREADS; \\\n"
    "    const uint kvrow_l = lid / KV_TCOLS; \\\n"
    "    const uint kvd0 = (lid % KV_TCOLS) * KV_NREADS; \\\n"
    "    { \\\n"
    "        uint qrow = q0 + qrow_l; \\\n"
    "        threadgroup half *dst = Qs + qrow_l * LDQ + qd0; \\\n"
    "        if (qrow < seq) { \\\n"
    "            device const half *src = Q + head_off + qrow * HD + qd0; \\\n"
    "            _Pragma(\"clang loop unroll(full)\") \\\n"
    "            for (uint j = 0; j < Q_NREADS; j += 4) { \\\n"
    "                half4 v = *reinterpret_cast<device const half4 *>(src + j); \\\n"
    "                dst[j + 0] = half(float(v.x) * scale2); \\\n"
    "                dst[j + 1] = half(float(v.y) * scale2); \\\n"
    "                dst[j + 2] = half(float(v.z) * scale2); \\\n"
    "                dst[j + 3] = half(float(v.w) * scale2); \\\n"
    "            } \\\n"
    "        } else { \\\n"
    "            _Pragma(\"clang loop unroll(full)\") \\\n"
    "            for (uint j = 0; j < Q_NREADS; ++j) dst[j] = half(0.0f); \\\n"
    "        } \\\n"
    "    } \\\n"
    "    simdgroup_float8x8 a0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \\\n"
    "    simdgroup_float8x8 a1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \\\n"
    "    simdgroup_float8x8 a2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \\\n"
    "    simdgroup_float8x8 a3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \\\n"
    "    simdgroup_float8x8 a4 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \\\n"
    "    simdgroup_float8x8 a5 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \\\n"
    "    simdgroup_float8x8 a6 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \\\n"
    "    simdgroup_float8x8 a7 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \\\n"
    "    float m_i = -INFINITY; \\\n"
    "    float l_i = 0.0f; \\\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup); \\\n"
    "    for (uint k0 = 0; k0 < seq; k0 += BC) { \\\n"
    "        { \\\n"
    "            uint kabs = k0 + kvrow_l; \\\n"
    "            if (kabs < seq) { \\\n"
    "                device const half *src = K + head_off + kabs * HD + kvd0; \\\n"
    "                _Pragma(\"clang loop unroll(full)\") \\\n"
    "                for (uint j = 0; j < KV_NREADS; j += 4) { \\\n"
    "                    half4 v = *reinterpret_cast<device const half4 *>(src + j); \\\n"
    "                    KVs[(kvd0 + j + 0) * LDK + kvrow_l] = v.x; \\\n"
    "                    KVs[(kvd0 + j + 1) * LDK + kvrow_l] = v.y; \\\n"
    "                    KVs[(kvd0 + j + 2) * LDK + kvrow_l] = v.z; \\\n"
    "                    KVs[(kvd0 + j + 3) * LDK + kvrow_l] = v.w; \\\n"
    "                } \\\n"
    "            } else { \\\n"
    "                _Pragma(\"clang loop unroll(full)\") \\\n"
    "                for (uint j = 0; j < KV_NREADS; ++j) { \\\n"
    "                    KVs[(kvd0 + j) * LDK + kvrow_l] = half(0.0f); \\\n"
    "                } \\\n"
    "            } \\\n"
    "        } \\\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup); \\\n"
    "        simdgroup_float8x8 s0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \\\n"
    "        simdgroup_float8x8 s1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \\\n"
    "        simdgroup_float8x8 s2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \\\n"
    "        simdgroup_float8x8 s3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \\\n"
    "        _Pragma(\"clang loop unroll(full)\") \\\n"
    "        for (uint d = 0; d < HD; d += 8) { \\\n"
    "            simdgroup_half8x8 qmat; \\\n"
    "            simdgroup_load(qmat, Qs + row0 * LDQ + d, LDQ); \\\n"
    "            simdgroup_half8x8 kmat; \\\n"
    "            simdgroup_load(kmat, KVs + d * LDK + 0, LDK); \\\n"
    "            simdgroup_multiply_accumulate(s0, qmat, kmat, s0); \\\n"
    "            simdgroup_load(kmat, KVs + d * LDK + 8, LDK); \\\n"
    "            simdgroup_multiply_accumulate(s1, qmat, kmat, s1); \\\n"
    "            simdgroup_load(kmat, KVs + d * LDK + 16, LDK); \\\n"
    "            simdgroup_multiply_accumulate(s2, qmat, kmat, s2); \\\n"
    "            simdgroup_load(kmat, KVs + d * LDK + 24, LDK); \\\n"
    "            simdgroup_multiply_accumulate(s3, qmat, kmat, s3); \\\n"
    "        } \\\n"
    "        thread auto &e0 = s0.thread_elements(); \\\n"
    "        thread auto &e1 = s1.thread_elements(); \\\n"
    "        thread auto &e2 = s2.thread_elements(); \\\n"
    "        thread auto &e3 = s3.thread_elements(); \\\n"
    "        const uint k_lim = (k0 + BC <= seq) ? BC : (seq - k0); \\\n"
    "        const uint c0 = coord.x; \\\n"
    "        if (c0 + 0 >= k_lim) e0[0] = -INFINITY; \\\n"
    "        if (c0 + 1 >= k_lim) e0[1] = -INFINITY; \\\n"
    "        if (c0 + 8 >= k_lim) e1[0] = -INFINITY; \\\n"
    "        if (c0 + 9 >= k_lim) e1[1] = -INFINITY; \\\n"
    "        if (c0 + 16 >= k_lim) e2[0] = -INFINITY; \\\n"
    "        if (c0 + 17 >= k_lim) e2[1] = -INFINITY; \\\n"
    "        if (c0 + 24 >= k_lim) e3[0] = -INFINITY; \\\n"
    "        if (c0 + 25 >= k_lim) e3[1] = -INFINITY; \\\n"
    "        float tilemax = max(max(e0[0], e0[1]), max(e1[0], e1[1])); \\\n"
    "        tilemax = max(tilemax, max(max(e2[0], e2[1]), max(e3[0], e3[1]))); \\\n"
    "        tilemax = max(tilemax, simd_shuffle_xor(tilemax, 1)); \\\n"
    "        tilemax = max(tilemax, simd_shuffle_xor(tilemax, 8)); \\\n"
    "        float newm = max(m_i, tilemax); \\\n"
    "        float alpha = (newm == -INFINITY) ? 1.0f : fast::exp2(m_i - newm); \\\n"
    "        m_i = newm; \\\n"
    "        e0[0] = (e0[0] == -INFINITY) ? 0.0f : fast::exp2(e0[0] - m_i); \\\n"
    "        e0[1] = (e0[1] == -INFINITY) ? 0.0f : fast::exp2(e0[1] - m_i); \\\n"
    "        e1[0] = (e1[0] == -INFINITY) ? 0.0f : fast::exp2(e1[0] - m_i); \\\n"
    "        e1[1] = (e1[1] == -INFINITY) ? 0.0f : fast::exp2(e1[1] - m_i); \\\n"
    "        e2[0] = (e2[0] == -INFINITY) ? 0.0f : fast::exp2(e2[0] - m_i); \\\n"
    "        e2[1] = (e2[1] == -INFINITY) ? 0.0f : fast::exp2(e2[1] - m_i); \\\n"
    "        e3[0] = (e3[0] == -INFINITY) ? 0.0f : fast::exp2(e3[0] - m_i); \\\n"
    "        e3[1] = (e3[1] == -INFINITY) ? 0.0f : fast::exp2(e3[1] - m_i); \\\n"
    "        float lpart = e0[0] + e0[1] + e1[0] + e1[1] + e2[0] + e2[1] + e3[0] + e3[1]; \\\n"
    "        lpart += simd_shuffle_xor(lpart, 1); \\\n"
    "        lpart += simd_shuffle_xor(lpart, 8); \\\n"
    "        l_i = l_i * alpha + lpart; \\\n"
    "        { thread auto &ae = a0.thread_elements(); ae[0] *= alpha; ae[1] *= alpha; } \\\n"
    "        { thread auto &ae = a1.thread_elements(); ae[0] *= alpha; ae[1] *= alpha; } \\\n"
    "        { thread auto &ae = a2.thread_elements(); ae[0] *= alpha; ae[1] *= alpha; } \\\n"
    "        { thread auto &ae = a3.thread_elements(); ae[0] *= alpha; ae[1] *= alpha; } \\\n"
    "        if (HD > 32) { \\\n"
    "            { thread auto &ae = a4.thread_elements(); ae[0] *= alpha; ae[1] *= alpha; } \\\n"
    "            { thread auto &ae = a5.thread_elements(); ae[0] *= alpha; ae[1] *= alpha; } \\\n"
    "            { thread auto &ae = a6.thread_elements(); ae[0] *= alpha; ae[1] *= alpha; } \\\n"
    "            { thread auto &ae = a7.thread_elements(); ae[0] *= alpha; ae[1] *= alpha; } \\\n"
    "        } \\\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup); \\\n"
    "        { \\\n"
    "            uint vabs = k0 + kvrow_l; \\\n"
    "            threadgroup half *dst = KVs + kvrow_l * LDV + kvd0; \\\n"
    "            if (vabs < seq) { \\\n"
    "                device const half *src = V + head_off + vabs * HD + kvd0; \\\n"
    "                _Pragma(\"clang loop unroll(full)\") \\\n"
    "                for (uint j = 0; j < KV_NREADS; j += 4) { \\\n"
    "                    half4 v = *reinterpret_cast<device const half4 *>(src + j); \\\n"
    "                    dst[j + 0] = v.x; \\\n"
    "                    dst[j + 1] = v.y; \\\n"
    "                    dst[j + 2] = v.z; \\\n"
    "                    dst[j + 3] = v.w; \\\n"
    "                } \\\n"
    "            } else { \\\n"
    "                _Pragma(\"clang loop unroll(full)\") \\\n"
    "                for (uint j = 0; j < KV_NREADS; ++j) dst[j] = half(0.0f); \\\n"
    "            } \\\n"
    "        } \\\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup); \\\n"
    "        simdgroup_half8x8 vmat; \\\n"
    "        PV_K(s0, 0, HD) \\\n"
    "        PV_K(s1, 8, HD) \\\n"
    "        PV_K(s2, 16, HD) \\\n"
    "        PV_K(s3, 24, HD) \\\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup); \\\n"
    "    } \\\n"
    "    const uint out_row = q0 + row0 + coord.y; \\\n"
    "    const uint out_col = coord.x; \\\n"
    "    const float inv = (l_i > 0.0f && m_i != -INFINITY) ? (1.0f / l_i) : 0.0f; \\\n"
    "    if (out_row < seq) { \\\n"
    "        device float *dst = Y + out_row * dim + qbase + out_col; \\\n"
    "        { thread auto &e = a0.thread_elements(); dst[0] = e[0] * inv; dst[1] = e[1] * inv; } \\\n"
    "        { thread auto &e = a1.thread_elements(); dst[8] = e[0] * inv; dst[9] = e[1] * inv; } \\\n"
    "        { thread auto &e = a2.thread_elements(); dst[16] = e[0] * inv; dst[17] = e[1] * inv; } \\\n"
    "        { thread auto &e = a3.thread_elements(); dst[24] = e[0] * inv; dst[25] = e[1] * inv; } \\\n"
    "        if (HD > 32) { \\\n"
    "            { thread auto &e = a4.thread_elements(); dst[32] = e[0] * inv; dst[33] = e[1] * inv; } \\\n"
    "            { thread auto &e = a5.thread_elements(); dst[40] = e[0] * inv; dst[41] = e[1] * inv; } \\\n"
    "            { thread auto &e = a6.thread_elements(); dst[48] = e[0] * inv; dst[49] = e[1] * inv; } \\\n"
    "            { thread auto &e = a7.thread_elements(); dst[56] = e[0] * inv; dst[57] = e[1] * inv; } \\\n"
    "        } \\\n"
    "    } \\\n"
    "}\n"
    "\n"
    "PREFILL_KERNEL(jaiPrefill32, 32)\n"
    "PREFILL_KERNEL(jaiPrefill64, 64)\n"
    "\n";
static bool ensureFlashAttn(void) {
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        if (!ensureDevice()) return;
        @autoreleasepool {
            NSError *error = nil;
            MTLCompileOptions *opts = [MTLCompileOptions new];
            if (@available(macOS 15.0, *)) {
                opts.mathMode = MTLMathModeFast;
            }
            id<MTLLibrary> library = [gDevice newLibraryWithSource:@(kFlashAttnSource)
                                                           options:opts
                                                             error:&error];
            if (library == nil) return;
            MTLComputePipelineDescriptor *desc = [MTLComputePipelineDescriptor new];
            desc.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
            id<MTLFunction> fn32 = [library newFunctionWithName:@"jaiPrefill32"];
            id<MTLFunction> fn64 = [library newFunctionWithName:@"jaiPrefill64"];
            id<MTLFunction> fnPack = [library newFunctionWithName:@"jaiPackMhaHalf"];
            if (fn32 == nil || fn64 == nil || fnPack == nil) return;
            desc.computeFunction = fn32;
            gFlashAttn32 = [gDevice newComputePipelineStateWithDescriptor:desc
                                                                  options:MTLPipelineOptionNone
                                                               reflection:nil
                                                                    error:&error];
            desc.computeFunction = fn64;
            gFlashAttn64 = [gDevice newComputePipelineStateWithDescriptor:desc
                                                                  options:MTLPipelineOptionNone
                                                               reflection:nil
                                                                    error:&error];
            gFlashPack = [gDevice newComputePipelineStateWithFunction:fnPack error:&error];
        }
    });
    return gFlashAttn32 != nil && gFlashAttn64 != nil && gFlashPack != nil;
}

bool jaiGpuAvailable(void) {
    return ensureDevice();
}

const char *jaiGpuDeviceName(void) {
    static char name[128];
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        if (!ensureDevice()) return;
        @autoreleasepool {
            const char *utf8 = [[gDevice name] UTF8String];
            if (utf8 != NULL) snprintf(name, sizeof(name), "%s", utf8);
        }
    });
    return name[0] != '\0' ? name : "none";
}

int jaiGpuDeviceCount(void) {
    @autoreleasepool {
        NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
        return devices != nil ? (int)devices.count : 0;
    }
}

bool jaiGpuSetDevice(int index) {
    if (gDeviceReady) return false;
    if (index < 0) return false;
    @autoreleasepool {
        NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
        if (devices == nil || index >= (int)devices.count) return false;
        gPreferredDevice = index;
        return true;
    }
}

void jaiGpuSetMixedPrecision(bool enabled) {
    gMixedPrecision = enabled;
}

bool jaiGpuMixedPrecision(void) {
    return gMixedPrecision;
}

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

/* The one device, queue and buffer the rest of the Apple backend shares.
 *
 * `graphbuild.m` builds whole networks against the same device and submits to
 * the same queue, so that its work is ordered against everything else exactly
 * as any other dispatch would be. */
id<MTLDevice> jaiGpuMetalDevice(void) {
    if (!ensureDevice()) return nil;
    return gDevice;
}

id<MTLCommandQueue> jaiGpuMetalQueue(void) {
    if (!ensureDevice()) return nil;
    return gQueue;
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

static void encodeDispatch(id<MTLComputeCommandEncoder> encoder,
                           id<MTLComputePipelineState> pipeline,
                           NSUInteger threads, NSUInteger groupSize);

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
static void encodeDispatch(id<MTLComputeCommandEncoder> encoder,
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

static bool commitMlpAccLocked(void);
static bool commitMlpWeightsLocked(void);
static bool commitMlp3WeightsLocked(void);
static bool blitMany(__unsafe_unretained id<MTLBuffer> *srcs, JaiGpuBuffer **dsts,
                    const size_t *offs, const size_t *bytes, int count);

static bool commitMlpAccLocked(void) {
    if (gMlpAccSide == 0 || gMlpLiveAcc == NULL) {
        gMlpAccSide = 0;
        return true;
    }
    __unsafe_unretained id<MTLBuffer> srcs[] = {gMlpScratchAcc};
    JaiGpuBuffer *dsts[] = {gMlpLiveAcc};
    size_t offs[] = {gMlpLiveAccOff};
    size_t sizes[] = {sizeof(float)};
    if (!blitMany(srcs, dsts, offs, sizes, 1)) return false;
    gMlpAccSide = 0;
    return true;
}

static bool commitMlpWeightsLocked(void) {
    if (gMlpSide == 0) return true;
    if (gMlpLiveW1 == NULL || gMlpLiveB1 == NULL || gMlpLiveW2 == NULL ||
        gMlpLiveB2 == NULL) {
        gMlpSide = 0;
        return true;
    }
    __unsafe_unretained id<MTLBuffer> srcs[] = {
        gMlpScratchW1, gMlpScratchB1, gMlpScratchW2, gMlpScratchB2
    };
    JaiGpuBuffer *dsts[] = {
        gMlpLiveW1, gMlpLiveB1, gMlpLiveW2, gMlpLiveB2
    };
    size_t offs[] = {
        gMlpLiveW1Off, gMlpLiveB1Off, gMlpLiveW2Off, gMlpLiveB2Off
    };
    size_t sizes[] = {
        gMlpLiveW1Bytes, gMlpLiveB1Bytes, gMlpLiveW2Bytes, gMlpLiveB2Bytes
    };
    if (!blitMany(srcs, dsts, offs, sizes, 4)) return false;
    gMlpSide = 0;
    return true;
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
    [gAsyncCommands addCompletedHandler:^(id<MTLCommandBuffer> done) {
        (void)done;
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

static bool flushAsyncLocked(id<MTLCommandBuffer> *oldestOut, uint64_t *oldestBatch) {
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
    if (!fineSyncEnabled() || gQueue == nil) return jaiGpuSynchronize();

    uint64_t want = 0;
    @synchronized(gQueue) {
        if (gMlpSide != 0 || gMlpAccSide != 0 || gMlp3Side != 0) want = UINT64_MAX;
        else if (b->lastBatch > doneBatch()) want = b->lastBatch;
    }
    if (want == UINT64_MAX) return jaiGpuSynchronize();
    if (want == 0) return true;

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
            if ([commands status] != MTLCommandBufferStatusCompleted) return false;
        }
        if (reached != 0) noteDone(reached);
    }
    return true;
}

static NSMutableDictionary<NSString *, id> *gMpsGraphs;

/* An MPSNDArray pads its innermost dimension out to a 16-byte row, so a packed
 * buffer read through one is misread whenever that dimension is not a multiple
 * of four floats. The whole-buffer initialiser has no such padding, which is
 * why offset zero is always safe and only a windowed feed can go wrong.
 *
 * NHWC activations with three channels are exactly that shape: twelve bytes a
 * row. Every batch after the first read the wrong pixels, silently -- the first
 * batch of an epoch trains on the right data and no later one does, so a
 * convolution over RGB images looked like a model that would not learn rather
 * than like a bug. Declining here sends the caller to its own kernels, which
 * address the buffer directly. */
static bool ndarrayWindowIsPacked(NSArray<NSNumber *> *shape) {
    if (shape.count == 0) return false;
    NSUInteger innermost = shape.lastObject.unsignedIntegerValue;
    return (innermost * sizeof(float)) % 16 == 0;
}

static MPSGraphTensorData *graphDataAt(JaiGpuBuffer *b, size_t offset,
                                      NSArray<NSNumber *> *shape) {
    NSUInteger count = 1;
    for (NSNumber *dim in shape) count *= dim.unsignedIntegerValue;
    const size_t bytes = (size_t)count * sizeof(float);
    if (offset + bytes > b->bytes) return nil;
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)b->buffer;
    if (offset == 0) {
        return [[MPSGraphTensorData alloc] initWithMTLBuffer:buf
                                                       shape:shape
                                                    dataType:MPSDataTypeFloat32];
    }
    if (!ndarrayWindowIsPacked(shape)) return nil;
    MPSNDArrayDescriptor *desc =
        [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32 shape:shape];
    if (desc == nil) return nil;
    MPSNDArray *array = [[MPSNDArray alloc] initWithBuffer:buf offset:offset descriptor:desc];
    if (array == nil) return nil;
    return [[MPSGraphTensorData alloc] initWithMPSNDArray:array];
}

static MPSGraphTensorData *graphData(JaiGpuBuffer *b, size_t offset,
                                     NSArray<NSNumber *> *shape) {
    mark(b);
    return graphDataAt(b, offset, shape);
}

/* The same window, without marking the buffer: a caller that decides for itself
 * which batch the work lands in has to mark it then, not now. Hands back an
 * MPSGraphTensorData at +1 for the caller to take over, or NULL when the window
 * cannot be addressed. */
void *jaiGpuTensorDataAt(JaiGpuBuffer *b, size_t offset, void *shape) {
    if (b == NULL || shape == NULL) return NULL;
    NSArray<NSNumber *> *dims = (__bridge NSArray<NSNumber *> *)shape;
    return (__bridge_retained void *)graphDataAt(b, offset, dims);
}

static MPSGraphTensorData *graphDataDesc(JaiGpuBuffer *b, size_t offset, size_t bytes,
                                         MPSNDArrayDescriptor *desc,
                                         NSArray<NSNumber *> *shape) {
    if (offset == 0) return graphData(b, 0, shape);
    if (desc == nil || b == NULL || b->buffer == NULL) return nil;
    if (!ndarrayWindowIsPacked(shape)) return nil;
    if (offset + bytes > b->bytes) return nil;
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)b->buffer;
    MPSNDArray *array = [[MPSNDArray alloc] initWithBuffer:buf offset:offset descriptor:desc];
    if (array == nil) return nil;
    return [[MPSGraphTensorData alloc] initWithMPSNDArray:array];
}

static bool prefetchBatchFeeds(JaiGpuBuffer *x, size_t xOff, size_t xStride, size_t xBytes,
                               MPSNDArrayDescriptor *xDesc, NSArray<NSNumber *> *xShape,
                               JaiGpuBuffer *labels, size_t labOff, size_t labStride,
                               size_t labBytes, MPSNDArrayDescriptor *yDesc,
                               NSArray<NSNumber *> *yShape, uint32_t steps,
                               NSMutableArray<MPSGraphTensorData *> *batchX,
                               NSMutableArray<MPSGraphTensorData *> *batchY) {
    for (uint32_t i = 0; i < steps; i++) {
        MPSGraphTensorData *dx =
            graphDataDesc(x, xOff + (size_t)i * xStride, xBytes, xDesc, xShape);
        MPSGraphTensorData *dy =
            graphDataDesc(labels, labOff + (size_t)i * labStride, labBytes, yDesc, yShape);
        if (dx == nil || dy == nil) return false;
        [batchX addObject:dx];
        [batchY addObject:dy];
    }
    return true;
}

static bool ensureAsyncCommandBuffer(void) {
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

static bool encodeGraphOnAsync(MPSGraph *graph,
                               NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds,
                               NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results) {
    dispatchTraceTick(1);
    if (!ensureAsyncCommandBuffer()) return false;
    MPSCommandBuffer *mps =
        [MPSCommandBuffer commandBufferWithCommandBuffer:gAsyncCommands];
    if (mps == nil) return false;
    static MPSGraphExecutionDescriptor *execDesc;
    static dispatch_once_t execOnce;
    dispatch_once(&execOnce, ^{
        execDesc = [MPSGraphExecutionDescriptor new];
        if (@available(macOS 12.3, *)) {
            MPSGraphCompilationDescriptor *comp = [MPSGraphCompilationDescriptor new];
            comp.optimizationLevel = MPSGraphOptimizationLevel1;
            execDesc.compilationDescriptor = comp;
        }
    });
    [graph encodeToCommandBuffer:mps
                           feeds:feeds
                targetOperations:nil
               resultsDictionary:results
             executionDescriptor:execDesc];
    gAsyncCommands = mps.rootCommandBuffer;
    gAsyncEncoder = nil;
    return gAsyncCommands != nil;
}

static id<MTLBuffer> growScratch(id<MTLBuffer> existing, size_t *cap, size_t bytes) {
    if (existing != nil && *cap >= bytes) return existing;
    *cap = bytes;
    return [gDevice newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

static bool blitMany(__unsafe_unretained id<MTLBuffer> *srcs, JaiGpuBuffer **dsts,
                    const size_t *offs, const size_t *bytes, int count) {
    if (count <= 0) return true;
    for (int i = 0; i < count; i++) {
        if (srcs[i] == nil || dsts[i] == NULL || dsts[i]->buffer == NULL) return false;
        if (offs[i] + bytes[i] > dsts[i]->bytes) return false;
    }
    if (!ensureAsyncCommandBuffer()) return false;
    for (int i = 0; i < count; i++) markLocked(dsts[i]);
    id<MTLBlitCommandEncoder> blit = [gAsyncCommands blitCommandEncoder];
    if (blit == nil) return false;
    for (int i = 0; i < count; i++) {
        [blit copyFromBuffer:srcs[i]
                sourceOffset:0
                    toBuffer:(__bridge id<MTLBuffer>)dsts[i]->buffer
           destinationOffset:offs[i]
                        size:bytes[i]];
    }
    [blit endEncoding];
    return true;
}

static MPSGraphTensor *ampMatMulForced(MPSGraph *graph, MPSGraphTensor *a, MPSGraphTensor *b,
                                       NSString *name) {
    MPSGraphTensor *a16 = [graph castTensor:a toType:MPSDataTypeFloat16
                                       name:[name stringByAppendingString:@"_a16"]];
    MPSGraphTensor *b16 = [graph castTensor:b toType:MPSDataTypeFloat16
                                       name:[name stringByAppendingString:@"_b16"]];
    MPSGraphTensor *c16 = [graph matrixMultiplicationWithPrimaryTensor:a16
                                                     secondaryTensor:b16
                                                                name:[name stringByAppendingString:@"_c16"]];
    return [graph castTensor:c16 toType:MPSDataTypeFloat32 name:name];
}

static MPSGraphTensor *ampMatMul(MPSGraph *graph, MPSGraphTensor *a, MPSGraphTensor *b,
                                 NSString *name) {
    if (!gMixedPrecision) {
        return [graph matrixMultiplicationWithPrimaryTensor:a secondaryTensor:b name:name];
    }
    return ampMatMulForced(graph, a, b, name);
}

static MPSGraphTensorData *graphDataMTL(id<MTLBuffer> buf, NSArray<NSNumber *> *shape) {
    if (buf == nil) return nil;
    return [[MPSGraphTensorData alloc] initWithMTLBuffer:buf
                                                   shape:shape
                                                dataType:MPSDataTypeFloat32];
}

static NSArray *cachedMpsGraph(uint32_t m, uint32_t k, uint32_t n, bool transA,
                               bool transB, bool useHalf) {
    if (gMpsGraphs == nil) gMpsGraphs = [[NSMutableDictionary alloc] init];
    NSString *key = [NSString stringWithFormat:@"%u:%u:%u:%d:%d:%d", m, k, n, transA, transB,
                     useHalf ? 1 : 0];
    NSArray *cached = gMpsGraphs[key];
    if (cached != nil) return cached;

    MPSGraph *graph = [[MPSGraph alloc] init];
    graph.options = MPSGraphOptionsNone;
    MPSGraphTensor *rawA = [graph
        placeholderWithShape:@[ @(transA ? k : m), @(transA ? m : k) ]
                    dataType:MPSDataTypeFloat32
                        name:@"A"];
    MPSGraphTensor *rawB = [graph
        placeholderWithShape:@[ @(transB ? n : k), @(transB ? k : n) ]
                    dataType:MPSDataTypeFloat32
                        name:@"B"];
    MPSGraphTensor *left = transA
        ? [graph transposeTensor:rawA permutation:@[ @1, @0 ] name:@"AT"]
        : rawA;
    MPSGraphTensor *right = transB
        ? [graph transposeTensor:rawB permutation:@[ @1, @0 ] name:@"BT"]
        : rawB;
    /* Whether to cast is the caller's to decide rather than the global
     * mixed-precision flag's. Inside a fused graph the cast is free: the
     * tensors it wraps never leave the graph, so MPSGraph folds the conversion
     * into the producer. Standing alone, both ends are the caller's float32
     * buffers, so casting means materialising a half copy of A, of B, and of
     * the whole m-by-n result and converting it back — which on a large output
     * costs more than the faster multiply saves, and on a small one does not.
     * Which way round it falls depends on the shape, so jaitensor measures
     * both and asks for the one that won. */
    MPSGraphTensor *product = useHalf
        ? ampMatMulForced(graph, left, right, @"C")
        : [graph matrixMultiplicationWithPrimaryTensor:left secondaryTensor:right name:@"C"];
    cached = @[ graph, rawA, rawB, product ];
    gMpsGraphs[key] = cached;
    return cached;
}

bool jaiGpuMatMulBuffers(JaiGpuBuffer *a, size_t aOffset, JaiGpuBuffer *b,
                         size_t bOffset, JaiGpuBuffer *out, size_t outOffset,
                         uint32_t m, uint32_t k, uint32_t n, bool transA,
                         bool transB, bool useHalf) {
    if (a == NULL || b == NULL || out == NULL) return false;
    if (a->buffer == NULL || b->buffer == NULL || out->buffer == NULL) return false;
    if (m == 0 || n == 0) return true;
    if (k == 0) return false;
    if (!ensureDevice()) return false;

    const NSUInteger aRows = transA ? k : m;
    const NSUInteger aCols = transA ? m : k;
    const NSUInteger bRows = transB ? n : k;
    const NSUInteger bCols = transB ? k : n;
    const NSUInteger aRowBytes = aCols * sizeof(float);
    const NSUInteger bRowBytes = bCols * sizeof(float);
    const NSUInteger cRowBytes = (NSUInteger)n * sizeof(float);
    if (aRowBytes % 16 != 0 || bRowBytes % 16 != 0 || cRowBytes % 16 != 0) {
        return false;
    }
    const size_t aBytes = (size_t)aRows * aRowBytes;
    const size_t bBytes = (size_t)bRows * bRowBytes;
    const size_t cBytes = (size_t)m * cRowBytes;
    if (aOffset + aBytes > a->bytes || bOffset + bBytes > b->bytes ||
        outOffset + cBytes > out->bytes) {
        return false;
    }

    @autoreleasepool {
        @synchronized(gQueue) {
            NSArray *cached = cachedMpsGraph(m, k, n, transA, transB, useHalf);
            if (cached == nil || cached.count != 4) return false;
            MPSGraph *graph = cached[0];
            MPSGraphTensor *rawA = cached[1];
            MPSGraphTensor *rawB = cached[2];
            MPSGraphTensor *product = cached[3];

            MPSGraphTensorData *dataA = graphData(a, aOffset, @[ @(aRows), @(aCols) ]);
            MPSGraphTensorData *dataB = graphData(b, bOffset, @[ @(bRows), @(bCols) ]);
            MPSGraphTensorData *dataC = graphData(out, outOffset, @[ @(m), @(n) ]);
            if (dataA == nil || dataB == nil || dataC == nil) return false;

            NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results =
                [@{product : dataC} mutableCopy];
            if (!encodeGraphOnAsync(graph, @{rawA : dataA, rawB : dataB}, results)) {
                return false;
            }
        }
        return true;
    }
}

static NSMutableDictionary<NSString *, NSArray *> *gMhaGraphs;

static bool encodeFlashAttn(id<MTLBuffer> qBuf, size_t qOff,
                            id<MTLBuffer> kBuf, size_t kOff,
                            id<MTLBuffer> vBuf, size_t vOff,
                            id<MTLBuffer> yBuf, size_t outOff,
                            uint32_t seq, uint32_t heads, uint32_t hd, float scale) {
    id<MTLComputePipelineState> pipe = hd == 32 ? gFlashAttn32 : hd == 64 ? gFlashAttn64 : nil;
    if (pipe == nil || gFlashPack == nil) return false;
    const size_t halfBytes = (size_t)seq * (size_t)heads * (size_t)hd * sizeof(uint16_t);
    gMhaHalfScratch = growScratch(gMhaHalfScratch, &gMhaHalfCap, halfBytes * 3u);
    if (gMhaHalfScratch == nil) return false;
    if (gAsyncCommands == nil) {
        gAsyncCommands = [gQueue commandBuffer];
        if (gAsyncCommands == nil) return false;
        beginBatchLocked();
    }
    if (gAsyncEncoder == nil) {
        gAsyncEncoder = [gAsyncCommands computeCommandEncoder];
        if (gAsyncEncoder == nil) return false;
    }
    [gAsyncEncoder setComputePipelineState:gFlashPack];
    [gAsyncEncoder setBuffer:qBuf offset:qOff atIndex:0];
    [gAsyncEncoder setBuffer:kBuf offset:kOff atIndex:1];
    [gAsyncEncoder setBuffer:vBuf offset:vOff atIndex:2];
    [gAsyncEncoder setBuffer:gMhaHalfScratch offset:0 atIndex:3];
    [gAsyncEncoder setBuffer:gMhaHalfScratch offset:halfBytes atIndex:4];
    [gAsyncEncoder setBuffer:gMhaHalfScratch offset:halfBytes * 2u atIndex:5];
    [gAsyncEncoder setBytes:&seq length:sizeof(seq) atIndex:6];
    [gAsyncEncoder setBytes:&heads length:sizeof(heads) atIndex:7];
    [gAsyncEncoder setBytes:&hd length:sizeof(hd) atIndex:8];
    const NSUInteger packThreads = (NSUInteger)heads * seq * (hd / 4u);
    encodeDispatch(gAsyncEncoder, gFlashPack, packThreads, 256);
    [gAsyncEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [gAsyncEncoder setComputePipelineState:pipe];
    [gAsyncEncoder setBuffer:gMhaHalfScratch offset:0 atIndex:0];
    [gAsyncEncoder setBuffer:gMhaHalfScratch offset:halfBytes atIndex:1];
    [gAsyncEncoder setBuffer:gMhaHalfScratch offset:halfBytes * 2u atIndex:2];
    [gAsyncEncoder setBuffer:yBuf offset:outOff atIndex:3];
    [gAsyncEncoder setBytes:&seq length:sizeof(seq) atIndex:4];
    [gAsyncEncoder setBytes:&heads length:sizeof(heads) atIndex:5];
    [gAsyncEncoder setBytes:&scale length:sizeof(scale) atIndex:6];
    const uint32_t qTiles = (seq + 63u) / 64u;
    [gAsyncEncoder dispatchThreadgroups:MTLSizeMake(qTiles, heads, 1)
                  threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
    return true;
}

static NSArray *cachedPackedMhaGraph(uint32_t seq, uint32_t heads, uint32_t hd, float scale) {
    if (seq == 0 || heads == 0 || hd == 0) return nil;
    if (gMhaGraphs == nil) gMhaGraphs = [[NSMutableDictionary alloc] init];
    NSString *key = [NSString stringWithFormat:@"mha-packed:%u:%u:%u:%.8f:%d",
                     seq, heads, hd, scale, gMixedPrecision ? 1 : 0];
    NSArray *cached = gMhaGraphs[key];
    if (cached != nil) return cached;
    if (@available(macOS 15.0, *)) {
        MPSGraph *graph = [MPSGraph new];
        graph.options = MPSGraphOptionsNone;
        NSArray *packed = @[ @1, @(seq), @(heads), @(hd) ];
        MPSGraphTensor *q = [graph placeholderWithShape:packed
                                               dataType:MPSDataTypeFloat32
                                                   name:@"Q"];
        MPSGraphTensor *k = [graph placeholderWithShape:packed
                                               dataType:MPSDataTypeFloat32
                                                   name:@"K"];
        MPSGraphTensor *v = [graph placeholderWithShape:packed
                                               dataType:MPSDataTypeFloat32
                                                   name:@"V"];
        MPSGraphTensor *qt = [graph transposeTensor:q permutation:@[ @0, @2, @1, @3 ]
                                               name:@"Qt"];
        MPSGraphTensor *kt = [graph transposeTensor:k permutation:@[ @0, @2, @1, @3 ]
                                               name:@"Kt"];
        MPSGraphTensor *vt = [graph transposeTensor:v permutation:@[ @0, @2, @1, @3 ]
                                               name:@"Vt"];
        MPSGraphTensor *qIn = qt;
        MPSGraphTensor *kIn = kt;
        MPSGraphTensor *vIn = vt;
        if (gMixedPrecision) {
            qIn = [graph castTensor:qt toType:MPSDataTypeFloat16 name:@"Q16"];
            kIn = [graph castTensor:kt toType:MPSDataTypeFloat16 name:@"K16"];
            vIn = [graph castTensor:vt toType:MPSDataTypeFloat16 name:@"V16"];
        }
        MPSGraphTensor *ctx =
            [graph scaledDotProductAttentionWithQueryTensor:qIn
                                                  keyTensor:kIn
                                                valueTensor:vIn
                                                      scale:scale
                                                       name:@"sdpa"];
        if (gMixedPrecision) {
            ctx = [graph castTensor:ctx toType:MPSDataTypeFloat32 name:@"C32"];
        }
        MPSGraphTensor *ct = [graph transposeTensor:ctx permutation:@[ @0, @2, @1, @3 ]
                                               name:@"Ct"];
        cached = @[ graph, q, k, v, ct ];
        gMhaGraphs[key] = cached;
        return cached;
    }
    return nil;
}

bool jaiGpuMhaPacked(JaiGpuBuffer *q, size_t qOff, JaiGpuBuffer *k, size_t kOff,
                     JaiGpuBuffer *v, size_t vOff, JaiGpuBuffer *out, size_t outOff,
                     uint32_t seq, uint32_t heads, uint32_t hd, float scale) {
    if (q == NULL || k == NULL || v == NULL || out == NULL) return false;
    if (q->buffer == NULL || k->buffer == NULL || v->buffer == NULL ||
        out->buffer == NULL) {
        return false;
    }
    if (seq == 0 || heads == 0 || hd == 0) return false;
    if (!isfinite(scale) || scale <= 0.0f) return false;
    const size_t bytes = (size_t)seq * (size_t)heads * (size_t)hd * sizeof(float);
    if (qOff + bytes > q->bytes || kOff + bytes > k->bytes ||
        vOff + bytes > v->bytes || outOff + bytes > out->bytes) {
        return false;
    }
    if (!ensureDevice()) return false;

    @autoreleasepool {
        @synchronized(gQueue) {
            if ((hd == 32 || hd == 64) && ensureFlashAttn()) {
                id<MTLBuffer> qBuf = (__bridge id<MTLBuffer>)q->buffer;
                id<MTLBuffer> kBuf = (__bridge id<MTLBuffer>)k->buffer;
                id<MTLBuffer> vBuf = (__bridge id<MTLBuffer>)v->buffer;
                id<MTLBuffer> yBuf = (__bridge id<MTLBuffer>)out->buffer;
                if (encodeFlashAttn(qBuf, qOff, kBuf, kOff, vBuf, vOff, yBuf, outOff,
                                    seq, heads, hd, scale)) {
                    markLocked(q);
                    markLocked(k);
                    markLocked(v);
                    markLocked(out);
                    return true;
                }
            }
            NSArray *packed = cachedPackedMhaGraph(seq, heads, hd, scale);
            if (packed != nil && packed.count == 5) {
                NSArray *shape = @[ @1, @(seq), @(heads), @(hd) ];
                MPSGraphTensorData *dq = graphData(q, qOff, shape);
                MPSGraphTensorData *dk = graphData(k, kOff, shape);
                MPSGraphTensorData *dv = graphData(v, vOff, shape);
                MPSGraphTensorData *dy = graphData(out, outOff, shape);
                if (dq != nil && dk != nil && dv != nil && dy != nil) {
                    NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results =
                        [@{packed[4] : dy} mutableCopy];
                    if (encodeGraphOnAsync(packed[0],
                                           @{packed[1] : dq, packed[2] : dk, packed[3] : dv},
                                           results)) {
                        return true;
                    }
                }
            }
            return false;
        }
    }
}

/* A graph compiled once and kept, rather than re-encoded per call.
 *
 * MPSGraph's own `encodeToCommandBuffer` re-plans the graph every time. For a
 * matmul that is lost in the multiply; for a convolution it is milliseconds of
 * CPU per encode, and a training step encodes three of them per layer. The
 * fused MLP path has compiled its graph since it was written for exactly this
 * reason -- `cachedMlpExecutable` -- and convolution had not, which is most of
 * why tests/bench/jaitensor's conv workloads were an order of magnitude behind
 * their peer with the arithmetic already on the same primitives.
 *
 * Returns nil when the graph will not compile, and every caller falls back to
 * encoding the graph directly. */
static MPSGraphExecutable *compiledGraph(MPSGraph *graph,
                                         NSArray<MPSGraphTensor *> *feeds,
                                         NSArray<NSArray<NSNumber *> *> *shapes,
                                         NSArray<MPSGraphTensor *> *targets) {
    if (graph == nil || feeds == nil || shapes == nil || targets == nil) return nil;
    if (feeds.count != shapes.count) return nil;
    if (@available(macOS 12.0, *)) {
        NSMutableDictionary<MPSGraphTensor *, MPSGraphShapedType *> *types =
            [NSMutableDictionary dictionaryWithCapacity:feeds.count];
        for (NSUInteger i = 0; i < feeds.count; i++) {
            types[feeds[i]] = [[MPSGraphShapedType alloc] initWithShape:shapes[i]
                                                              dataType:MPSDataTypeFloat32];
        }
        MPSGraphCompilationDescriptor *comp = nil;
        if (@available(macOS 12.3, *)) {
            comp = [MPSGraphCompilationDescriptor new];
            comp.optimizationLevel = MPSGraphOptimizationLevel1;
        }
        return [graph compileWithDevice:nil
                                  feeds:types
                          targetTensors:targets
                       targetOperations:nil
                  compilationDescriptor:comp];
    }
    return nil;
}

static bool encodeMlpExecutableOnAsync(
    MPSGraphExecutable *exec,
    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feedMap,
    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *resultMap);

bool jaiGpuConv2dBuffers(JaiGpuBuffer *input, size_t inputOffset,
                         JaiGpuBuffer *weights, size_t weightsOffset,
                         JaiGpuBuffer *bias, size_t biasOffset,
                         JaiGpuBuffer *out, size_t outOffset,
                         uint32_t n, uint32_t h, uint32_t w, uint32_t cin,
                         uint32_t cout, uint32_t kh, uint32_t kw,
                         uint32_t sh, uint32_t sw, uint32_t ph, uint32_t pw,
                         uint32_t activation, uint32_t layout) {
    if (input == NULL || weights == NULL || out == NULL) return false;
    if (layout > 1) return false;
    if (input->buffer == NULL || weights->buffer == NULL || out->buffer == NULL) return false;
    if (n == 0 || h == 0 || w == 0 || cin == 0 || cout == 0 || kh == 0 || kw == 0) return false;
    if (sh == 0 || sw == 0) return false;
    if (!ensureDevice()) return false;
    if (h + 2 * ph < kh || w + 2 * pw < kw) return false;
    const uint32_t outH = (h + 2 * ph - kh) / sh + 1;
    const uint32_t outW = (w + 2 * pw - kw) / sw + 1;
    const size_t inBytes = (size_t)n * h * w * cin * sizeof(float);
    const size_t wBytes = (size_t)kh * kw * cin * cout * sizeof(float);
    const size_t outBytes = (size_t)n * outH * outW * cout * sizeof(float);
    if (inputOffset + inBytes > input->bytes || weightsOffset + wBytes > weights->bytes ||
        outOffset + outBytes > out->bytes) {
        return false;
    }
    /* MPSNDArray rejects user buffers smaller than its internal alignment
     * quantum; fall back to the Metal im2col path for tiny activations. */
    if (input->bytes < 512 || weights->bytes < 512 || out->bytes < 512) return false;
    if (bias != NULL) {
        if (bias->buffer == NULL) return false;
        if (biasOffset + (size_t)cout * sizeof(float) > bias->bytes) return false;
    }

    @autoreleasepool {
        @synchronized(gQueue) {
            static NSMutableDictionary<NSString *, NSArray *> *graphs;
            if (graphs == nil) graphs = [[NSMutableDictionary alloc] init];
            NSString *key = [NSString stringWithFormat:
                @"conv:%u:%u:%u:%u:%u:%u:%u:%u:%u:%u:%u:%d:%d:%u:%u",
                n, h, w, cin, cout, kh, kw, sh, sw, ph, pw,
                bias != NULL ? 1 : 0, gMixedPrecision ? 1 : 0, activation, layout];
            NSArray *cached = graphs[key];
            if (cached == nil) {
                MPSGraph *graph = [[MPSGraph alloc] init];
                graph.options = MPSGraphOptionsNone;
                const bool chw = layout == 1;
                NSArray *srcShape = chw ? @[ @(n), @(cin), @(h), @(w) ]
                                        : @[ @(n), @(h), @(w), @(cin) ];
                NSArray *filtShape = chw ? @[ @(cout), @(cin), @(kh), @(kw) ]
                                         : @[ @(kh), @(kw), @(cin), @(cout) ];
                MPSGraphTensor *src = [graph placeholderWithShape:srcShape
                                                         dataType:MPSDataTypeFloat32
                                                             name:@"X"];
                MPSGraphTensor *filt = [graph placeholderWithShape:filtShape
                                                          dataType:MPSDataTypeFloat32
                                                              name:@"W"];
                MPSGraphConvolution2DOpDescriptor *desc =
                    [MPSGraphConvolution2DOpDescriptor
                        descriptorWithStrideInX:sw
                                      strideInY:sh
                                 dilationRateInX:1
                                 dilationRateInY:1
                                         groups:1
                                    paddingLeft:pw
                                   paddingRight:pw
                                     paddingTop:ph
                                  paddingBottom:ph
                                   paddingStyle:MPSGraphPaddingStyleExplicit
                                     dataLayout:(chw ? MPSGraphTensorNamedDataLayoutNCHW
                                                     : MPSGraphTensorNamedDataLayoutNHWC)
                                  weightsLayout:(chw ? MPSGraphTensorNamedDataLayoutOIHW
                                                     : MPSGraphTensorNamedDataLayoutHWIO)];
                /* Half precision for the multiply-accumulate, single for
                 * everything around it -- the same split the matmul and the
                 * attention paths already take when mixed precision is on.
                 * MPS accumulates a half-precision convolution into single
                 * internally, so what this trades is the width of the operands
                 * in memory, which is what a convolution of this size is
                 * limited by. The cast pair is inside the compiled graph, so
                 * it costs no dispatch of its own. */
                MPSGraphTensor *srcIn = src;
                MPSGraphTensor *filtIn = filt;
                if (gMixedPrecision) {
                    srcIn = [graph castTensor:src toType:MPSDataTypeFloat16 name:@"X16"];
                    filtIn = [graph castTensor:filt toType:MPSDataTypeFloat16 name:@"W16"];
                }
                MPSGraphTensor *conv = [graph convolution2DWithSourceTensor:srcIn
                                                              weightsTensor:filtIn
                                                                 descriptor:desc
                                                                       name:@"conv"];
                if (gMixedPrecision) {
                    conv = [graph castTensor:conv toType:MPSDataTypeFloat32 name:@"C32"];
                }
                MPSGraphTensor *product = conv;
                MPSGraphTensor *biasT = nil;
                if (bias != NULL) {
                    biasT = [graph placeholderWithShape:@[ @(cout) ]
                                               dataType:MPSDataTypeFloat32
                                                   name:@"b"];
                    /* The bias broadcasts along whichever axis holds the
                     * channels, so its shape follows the data layout. */
                    NSArray *biasShape = chw ? @[ @1, @(cout), @1, @1 ]
                                             : @[ @1, @1, @1, @(cout) ];
                    product = [graph additionWithPrimaryTensor:conv
                                               secondaryTensor:[graph reshapeTensor:biasT
                                                                          withShape:biasShape
                                                                               name:@"br"]
                                                          name:@"convb"];
                }
                /* Folding the activation into the same compiled graph as the
                 * convolution and its bias means the epilogue costs nothing
                 * beyond what the graph compiler already fuses on its own --
                 * one dispatch for conv+bias+activation instead of the
                 * elementwise kernel jaitensor would otherwise launch
                 * separately over the whole output. */
                if (activation == 1) {
                    product = [graph reLUWithTensor:product name:@"convrelu"];
                } else if (activation == 2) {
                    /* SiLU: x * sigmoid(x). Every modern detector puts one
                     * after every convolution, and as a separate jaitensor
                     * kernel it reads and writes the whole output again. */
                    product = [graph multiplicationWithPrimaryTensor:product
                                                     secondaryTensor:[graph sigmoidWithTensor:product
                                                                                         name:@"convsig"]
                                                                name:@"convsilu"];
                }
                NSMutableArray *built = [@[ graph, src, filt, product ] mutableCopy];
                if (biasT != nil) [built addObject:biasT];
                NSMutableArray *feeds = [@[ src, filt ] mutableCopy];
                NSMutableArray *shapes = [@[ srcShape, filtShape ] mutableCopy];
                if (biasT != nil) {
                    [feeds addObject:biasT];
                    [shapes addObject:@[ @(cout) ]];
                }
                MPSGraphExecutable *exec = compiledGraph(graph, feeds, shapes, @[ product ]);
                [built addObject:(exec != nil ? (id)exec : (id)[NSNull null])];
                cached = built;
                graphs[key] = cached;
            }

            MPSGraph *graph = cached[0];
            MPSGraphTensor *src = cached[1];
            MPSGraphTensor *filt = cached[2];
            MPSGraphTensor *product = cached[3];
            /* The same shapes the placeholders were built from -- the
             * buffers hold identical bytes either way, and only how the
             * dimensions are read off them differs. */
            const bool feedChw = layout == 1;
            MPSGraphTensorData *dataX =
                graphData(input, inputOffset,
                          feedChw ? @[ @(n), @(cin), @(h), @(w) ]
                                  : @[ @(n), @(h), @(w), @(cin) ]);
            MPSGraphTensorData *dataW =
                graphData(weights, weightsOffset,
                          feedChw ? @[ @(cout), @(cin), @(kh), @(kw) ]
                                  : @[ @(kh), @(kw), @(cin), @(cout) ]);
            MPSGraphTensorData *dataY =
                graphData(out, outOffset,
                          feedChw ? @[ @(n), @(cout), @(outH), @(outW) ]
                                  : @[ @(n), @(outH), @(outW), @(cout) ]);
            if (dataX == nil || dataW == nil || dataY == nil) return false;
            NSMutableDictionary *feeds = [@{src : dataX, filt : dataW} mutableCopy];
            if (bias != NULL) {
                if (cached.count < 5) return false;
                MPSGraphTensorData *dataB = graphData(bias, biasOffset, @[ @(cout) ]);
                if (dataB == nil) return false;
                feeds[cached[4]] = dataB;
            }
            NSMutableDictionary *results = [@{product : dataY} mutableCopy];
            id last = cached[cached.count - 1];
            if ([last isKindOfClass:[MPSGraphExecutable class]]) {
                if (encodeMlpExecutableOnAsync((MPSGraphExecutable *)last, feeds, results)) {
                    return true;
                }
            }
            if (!encodeGraphOnAsync(graph, feeds, results)) return false;
        }
        return true;
    }
}

/* The two convolution gradients, from the same MPSGraph the forward pass uses.
 *
 * jaitensor's own backward is im2col plus two products, which rebuilds a
 * [batch * outH * outW, kh * kw * cin] matrix per step: for the second layer of
 * tests/bench/jaitensor's conv-wide that is a hundred and fifty megabytes
 * allocated, filled and thrown away on every batch, and it is why convolution
 * was the one part of the library an order of magnitude behind its peer. These
 * hand the work to the same primitives PyTorch reaches, and the im2col path
 * stays as the fallback for a shape MPS will not take.
 *
 * `mode` picks which gradient: 0 the one with respect to the input, 1 the one
 * with respect to the weights. They differ only in which tensor is the second
 * input and what shape comes out, so one graph builder covers both and there is
 * one place to keep the descriptor in step with the forward pass. */
static bool convGradient(int mode,
                         JaiGpuBuffer *grad, size_t gradOffset,
                         JaiGpuBuffer *other, size_t otherOffset,
                         JaiGpuBuffer *out, size_t outOffset,
                         uint32_t n, uint32_t h, uint32_t w, uint32_t cin,
                         uint32_t cout, uint32_t kh, uint32_t kw,
                         uint32_t sh, uint32_t sw, uint32_t ph, uint32_t pw) {
    if (grad == NULL || other == NULL || out == NULL) return false;
    if (grad->buffer == NULL || other->buffer == NULL || out->buffer == NULL) return false;
    if (n == 0 || h == 0 || w == 0 || cin == 0 || cout == 0 || kh == 0 || kw == 0) return false;
    if (sh == 0 || sw == 0) return false;
    if (!ensureDevice()) return false;
    if (h + 2 * ph < kh || w + 2 * pw < kw) return false;

    const uint32_t outH = (h + 2 * ph - kh) / sh + 1;
    const uint32_t outW = (w + 2 * pw - kw) / sw + 1;
    const size_t gradBytes = (size_t)n * outH * outW * cout * sizeof(float);
    const size_t inBytes = (size_t)n * h * w * cin * sizeof(float);
    const size_t wBytes = (size_t)kh * kw * cin * cout * sizeof(float);
    const size_t otherBytes = mode == 0 ? wBytes : inBytes;
    const size_t resultBytes = mode == 0 ? inBytes : wBytes;
    if (gradOffset + gradBytes > grad->bytes) return false;
    if (otherOffset + otherBytes > other->bytes) return false;
    if (outOffset + resultBytes > out->bytes) return false;
    /* Same alignment quantum the forward pass respects. */
    if (grad->bytes < 512 || other->bytes < 512 || out->bytes < 512) return false;

    @autoreleasepool {
        @synchronized(gQueue) {
            static NSMutableDictionary<NSString *, NSArray *> *graphs;
            if (graphs == nil) graphs = [[NSMutableDictionary alloc] init];
            NSString *key = [NSString stringWithFormat:
                @"convgrad:%d:%u:%u:%u:%u:%u:%u:%u:%u:%u:%u:%u",
                mode, n, h, w, cin, cout, kh, kw, sh, sw, ph, pw];
            NSArray *cached = graphs[key];
            if (cached == nil) {
                MPSGraph *graph = [[MPSGraph alloc] init];
                graph.options = MPSGraphOptionsNone;
                NSArray *gradShape = @[ @(n), @(outH), @(outW), @(cout) ];
                NSArray *srcShape = @[ @(n), @(h), @(w), @(cin) ];
                NSArray *filtShape = @[ @(kh), @(kw), @(cin), @(cout) ];
                MPSGraphTensor *dy = [graph placeholderWithShape:gradShape
                                                        dataType:MPSDataTypeFloat32
                                                            name:@"dY"];
                MPSGraphTensor *second =
                    [graph placeholderWithShape:(mode == 0 ? filtShape : srcShape)
                                       dataType:MPSDataTypeFloat32
                                           name:@"other"];
                MPSGraphConvolution2DOpDescriptor *desc =
                    [MPSGraphConvolution2DOpDescriptor
                        descriptorWithStrideInX:sw
                                      strideInY:sh
                                 dilationRateInX:1
                                 dilationRateInY:1
                                         groups:1
                                    paddingLeft:pw
                                   paddingRight:pw
                                     paddingTop:ph
                                  paddingBottom:ph
                                   paddingStyle:MPSGraphPaddingStyleExplicit
                                     dataLayout:MPSGraphTensorNamedDataLayoutNHWC
                                  weightsLayout:MPSGraphTensorNamedDataLayoutHWIO];
                /* Single precision, deliberately: narrowing the two gradient
                 * convolutions the way the forward pass is narrowed measured
                 * two to three per cent SLOWER on every shape tried, and a
                 * gradient is the worse place to spend accuracy. */
                MPSGraphTensor *dyIn = dy;
                MPSGraphTensor *secondIn = second;
                MPSGraphTensor *result = nil;
                if (mode == 0) {
                    result = [graph convolution2DDataGradientWithIncomingGradientTensor:dyIn
                                                                         weightsTensor:secondIn
                                                                           outputShape:srcShape
                                                          forwardConvolutionDescriptor:desc
                                                                                  name:@"dX"];
                } else {
                    result = [graph convolution2DWeightsGradientWithIncomingGradientTensor:dyIn
                                                                             sourceTensor:secondIn
                                                                              outputShape:filtShape
                                                             forwardConvolutionDescriptor:desc
                                                                                     name:@"dW"];
                }
                if (result == nil) return false;
                NSArray *feedShapes = mode == 0
                    ? @[ gradShape, filtShape ]
                    : @[ gradShape, srcShape ];
                MPSGraphExecutable *exec =
                    compiledGraph(graph, @[ dy, second ], feedShapes, @[ result ]);
                cached = @[ graph, dy, second, result,
                            (exec != nil ? (id)exec : (id)[NSNull null]) ];
                graphs[key] = cached;
            }

            MPSGraphTensorData *dataG = graphData(grad, gradOffset,
                                                  @[ @(n), @(outH), @(outW), @(cout) ]);
            MPSGraphTensorData *dataO =
                graphData(other, otherOffset,
                          mode == 0 ? @[ @(kh), @(kw), @(cin), @(cout) ]
                                    : @[ @(n), @(h), @(w), @(cin) ]);
            MPSGraphTensorData *dataR =
                graphData(out, outOffset,
                          mode == 0 ? @[ @(n), @(h), @(w), @(cin) ]
                                    : @[ @(kh), @(kw), @(cin), @(cout) ]);
            if (dataG == nil || dataO == nil || dataR == nil) return false;
            NSMutableDictionary *feeds = [@{cached[1] : dataG, cached[2] : dataO} mutableCopy];
            NSMutableDictionary *results = [@{cached[3] : dataR} mutableCopy];
            if ([cached[4] isKindOfClass:[MPSGraphExecutable class]]) {
                if (encodeMlpExecutableOnAsync((MPSGraphExecutable *)cached[4], feeds, results)) {
                    return true;
                }
            }
            if (!encodeGraphOnAsync(cached[0], feeds, results)) return false;
        }
        return true;
    }
}

bool jaiGpuConv2dDataGradBuffers(JaiGpuBuffer *grad, size_t gradOffset,
                                 JaiGpuBuffer *weights, size_t weightsOffset,
                                 JaiGpuBuffer *out, size_t outOffset,
                                 uint32_t n, uint32_t h, uint32_t w, uint32_t cin,
                                 uint32_t cout, uint32_t kh, uint32_t kw,
                                 uint32_t sh, uint32_t sw, uint32_t ph, uint32_t pw) {
    return convGradient(0, grad, gradOffset, weights, weightsOffset, out, outOffset,
                        n, h, w, cin, cout, kh, kw, sh, sw, ph, pw);
}

bool jaiGpuConv2dWeightsGradBuffers(JaiGpuBuffer *grad, size_t gradOffset,
                                    JaiGpuBuffer *input, size_t inputOffset,
                                    JaiGpuBuffer *out, size_t outOffset,
                                    uint32_t n, uint32_t h, uint32_t w, uint32_t cin,
                                    uint32_t cout, uint32_t kh, uint32_t kw,
                                    uint32_t sh, uint32_t sw, uint32_t ph, uint32_t pw) {
    return convGradient(1, grad, gradOffset, input, inputOffset, out, outOffset,
                        n, h, w, cin, cout, kh, kw, sh, sw, ph, pw);
}

static NSMutableDictionary<NSString *, NSArray *> *gMlpGraphs;
static NSMutableDictionary<NSString *, MPSGraphExecutable *> *gMlpExecutables;

static NSString *mlpCacheKey(uint32_t B, uint32_t inFeatures, uint32_t H, uint32_t C,
                             float lr, bool trackCorrect) {
    return [NSString stringWithFormat:@"%u:%u:%u:%u:%.8f:%d:%d",
            B, inFeatures, H, C, lr, trackCorrect ? 1 : 0, gMixedPrecision ? 1 : 0];
}

static MPSGraphShapedType *mlpShapedType(NSArray<NSNumber *> *shape) {
    return [[MPSGraphShapedType alloc] initWithShape:shape dataType:MPSDataTypeFloat32];
}

static MPSGraphTensorShapedTypeDictionary *mlpFeedTypes(NSArray *cached, bool trackCorrect,
                                                          uint32_t B, uint32_t inFeatures,
                                                          uint32_t H, uint32_t C) {
    NSMutableDictionary *feeds = [@{
        cached[1] : mlpShapedType(@[ @(B), @(inFeatures) ]),
        cached[2] : mlpShapedType(@[ @(inFeatures), @(H) ]),
        cached[3] : mlpShapedType(@[ @(H) ]),
        cached[4] : mlpShapedType(@[ @(H), @(C) ]),
        cached[5] : mlpShapedType(@[ @(C) ]),
        cached[6] : mlpShapedType(@[ @(B) ]),
        cached[7] : mlpShapedType(@[ @1 ]),
    } mutableCopy];
    if (trackCorrect) feeds[cached[8]] = mlpShapedType(@[ @1 ]);
    return feeds;
}

static MPSGraphExecutable *cachedMlpExecutable(NSArray *cached, bool trackCorrect,
                                               uint32_t B, uint32_t inFeatures, uint32_t H,
                                               uint32_t C, float lr) {
    if (@available(macOS 12.0, *)) {
        if (gMlpExecutables == nil) {
            gMlpExecutables = [[NSMutableDictionary alloc] init];
        }
        NSString *key = mlpCacheKey(B, inFeatures, H, C, lr, trackCorrect);
        MPSGraphExecutable *exec = gMlpExecutables[key];
        if (exec != nil) return exec;

        MPSGraph *graph = cached[0];
        const int wBase = trackCorrect ? 9 : 8;
        const int targetCount = trackCorrect ? 6 : 5;
        NSMutableArray *targets =
            [NSMutableArray arrayWithCapacity:(NSUInteger)targetCount];
        for (int i = 0; i < targetCount; i++) {
            [targets addObject:cached[wBase + i]];
        }

        MPSGraphCompilationDescriptor *comp = nil;
        if (@available(macOS 12.3, *)) {
            comp = [MPSGraphCompilationDescriptor new];
            comp.optimizationLevel = MPSGraphOptimizationLevel1;
        }
        exec = [graph compileWithDevice:nil
                                  feeds:mlpFeedTypes(cached, trackCorrect, B, inFeatures, H, C)
                          targetTensors:targets
                       targetOperations:nil
                  compilationDescriptor:comp];
        if (exec == nil) return nil;
        exec.options = MPSGraphOptionsNone;
        gMlpExecutables[key] = exec;
        return exec;
    }
    return nil;
}

static NSMutableArray<MPSGraphTensorData *> *mlpMapArray(
    NSArray<MPSGraphTensor *> *tensors,
    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *map) {
    if (tensors == nil || map == nil) return nil;
    NSMutableArray<MPSGraphTensorData *> *out =
        [NSMutableArray arrayWithCapacity:tensors.count];
    for (MPSGraphTensor *tensor in tensors) {
        MPSGraphTensorData *data = map[tensor];
        if (data == nil) return nil;
        [out addObject:data];
    }
    return out;
}

static bool encodeMlpExecutableOnAsyncArrays(
    MPSGraphExecutable *exec,
    NSArray<MPSGraphTensorData *> *inputs,
    NSArray<MPSGraphTensorData *> *results) {
    if (exec == nil || inputs == nil || results == nil) return false;
    dispatchTraceTick(1);
    if (!ensureAsyncCommandBuffer()) return false;
    MPSCommandBuffer *mps =
        [MPSCommandBuffer commandBufferWithCommandBuffer:gAsyncCommands];
    if (mps == nil) return false;
    if (@available(macOS 12.0, *)) {
        [exec encodeToCommandBuffer:mps
                         inputsArray:inputs
                        resultsArray:results
                 executionDescriptor:nil];
    } else {
        return false;
    }
    gAsyncCommands = mps.rootCommandBuffer;
    gAsyncEncoder = nil;
    return gAsyncCommands != nil;
}

/* Encode a compiled executable into the batch everything else is queued on,
 * for the whole-network compiler next door. Keeping it on the shared command
 * buffer is what lets a compiled plan sit in the middle of ordinary work
 * without a fence on either side. */
bool jaiGpuEncodeExecutable(void *executable, void *inputs, void *results) {
    if (executable == NULL || inputs == NULL || results == NULL) return false;
    @synchronized(gQueue) {
        return encodeMlpExecutableOnAsyncArrays(
            (__bridge MPSGraphExecutable *)executable,
            (__bridge NSArray<MPSGraphTensorData *> *)inputs,
            (__bridge NSArray<MPSGraphTensorData *> *)results);
    }
}

static bool encodeMlpExecutableOnAsync(
    MPSGraphExecutable *exec,
    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feedMap,
    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *resultMap) {
    if (exec == nil || exec.feedTensors == nil || exec.targetTensors == nil) return false;
    NSMutableArray<MPSGraphTensorData *> *inputs = mlpMapArray(exec.feedTensors, feedMap);
    NSMutableArray<MPSGraphTensorData *> *results = mlpMapArray(exec.targetTensors, resultMap);
    if (inputs == nil || results == nil) return false;
    return encodeMlpExecutableOnAsyncArrays(exec, inputs, results);
}

static bool encodeEpochBatch(
    MPSGraphExecutable *exec,
    MPSGraph *graph,
    NSMutableDictionary *feeds,
    NSMutableDictionary *results,
    MPSGraphTensor *xTensor,
    MPSGraphTensor *yTensor,
    MPSGraphTensorData *dx,
    MPSGraphTensorData *dy,
    NSMutableArray<MPSGraphTensorData *> *__strong *inputsCache,
    NSArray<MPSGraphTensorData *> *__strong *resultsCache) {
    feeds[xTensor] = dx;
    feeds[yTensor] = dy;
    if (exec != nil && exec.feedTensors != nil && exec.targetTensors != nil) {
        if (*inputsCache == nil) {
            *inputsCache = mlpMapArray(exec.feedTensors, feeds);
            *resultsCache = mlpMapArray(exec.targetTensors, results);
            if (*inputsCache == nil || *resultsCache == nil) return false;
            return encodeMlpExecutableOnAsyncArrays(exec, *inputsCache, *resultsCache);
        }
        const NSUInteger xIdx = [exec.feedTensors indexOfObjectIdenticalTo:xTensor];
        const NSUInteger yIdx = [exec.feedTensors indexOfObjectIdenticalTo:yTensor];
        if (xIdx == NSNotFound || yIdx == NSNotFound) {
            return encodeMlpExecutableOnAsync(exec, feeds, results);
        }
        NSMutableArray<MPSGraphTensorData *> *inputs = [*inputsCache mutableCopy];
        [inputs replaceObjectAtIndex:xIdx withObject:dx];
        [inputs replaceObjectAtIndex:yIdx withObject:dy];
        *inputsCache = inputs;
        return encodeMlpExecutableOnAsyncArrays(exec, inputs, *resultsCache);
    }
    return encodeGraphOnAsync(graph, feeds, results);
}

static NSArray *cachedMlpGraph(uint32_t B, uint32_t inFeatures, uint32_t H, uint32_t C,
                               float lr, bool trackCorrect) {
    if (gMlpGraphs == nil) gMlpGraphs = [[NSMutableDictionary alloc] init];
    NSString *key = mlpCacheKey(B, inFeatures, H, C, lr, trackCorrect);
    NSArray *cached = gMlpGraphs[key];
    if (cached != nil) return cached;

    MPSGraph *graph = [[MPSGraph alloc] init];
    graph.options = MPSGraphOptionsNone;
    MPSGraphTensor *x = [graph placeholderWithShape:@[ @(B), @(inFeatures) ] dataType:MPSDataTypeFloat32 name:@"X"];
    MPSGraphTensor *w1 = [graph placeholderWithShape:@[ @(inFeatures), @(H) ] dataType:MPSDataTypeFloat32 name:@"W1"];
    MPSGraphTensor *b1 = [graph placeholderWithShape:@[ @(H) ] dataType:MPSDataTypeFloat32 name:@"b1"];
    MPSGraphTensor *w2 = [graph placeholderWithShape:@[ @(H), @(C) ] dataType:MPSDataTypeFloat32 name:@"W2"];
    MPSGraphTensor *b2 = [graph placeholderWithShape:@[ @(C) ] dataType:MPSDataTypeFloat32 name:@"b2"];
    MPSGraphTensor *labels = [graph placeholderWithShape:@[ @(B) ] dataType:MPSDataTypeFloat32 name:@"y"];
    MPSGraphTensor *acc = [graph placeholderWithShape:@[ @1 ] dataType:MPSDataTypeFloat32 name:@"acc"];
    MPSGraphTensor *correctAcc = trackCorrect
        ? [graph placeholderWithShape:@[ @1 ] dataType:MPSDataTypeFloat32 name:@"corr"]
        : nil;

    MPSGraphTensor *b1r = [graph reshapeTensor:b1 withShape:@[ @1, @(H) ] name:@"b1r"];
    MPSGraphTensor *hpre = [graph additionWithPrimaryTensor:ampMatMul(graph, x, w1, @"XW1")
                                           secondaryTensor:b1r
                                                      name:@"Hpre"];
    MPSGraphTensor *hidden = [graph reLUWithTensor:hpre name:@"H"];
    MPSGraphTensor *b2r = [graph reshapeTensor:b2 withShape:@[ @1, @(C) ] name:@"b2r"];
    MPSGraphTensor *logits = [graph additionWithPrimaryTensor:ampMatMul(graph, hidden, w2, @"HW2")
                                            secondaryTensor:b2r
                                                       name:@"logits"];
    MPSGraphTensor *idx = [graph castTensor:labels toType:MPSDataTypeInt32 name:@"idx"];
    MPSGraphTensor *onehot = [graph oneHotWithIndicesTensor:idx
                                                      depth:C
                                                   dataType:MPSDataTypeFloat32
                                                       name:@"oh"];
    MPSGraphTensor *loss = [graph softMaxCrossEntropyWithSourceTensor:logits
                                                        labelsTensor:onehot
                                                                axis:1
                                                       reductionType:MPSGraphLossReductionTypeSum
                                                                name:@"loss"];
    MPSGraphTensor *accOut = [graph additionWithPrimaryTensor:acc secondaryTensor:loss name:@"accOut"];
    MPSGraphTensor *correctOut = nil;
    if (trackCorrect) {
        MPSGraphTensor *pred = [graph reshapeTensor:[graph reductionArgMaximumWithTensor:logits
                                                                                   axis:1
                                                                                   name:@"pred"]
                                          withShape:@[ @(B) ]
                                               name:@"pred1"];
        MPSGraphTensor *hits = [graph castTensor:[graph equalWithPrimaryTensor:pred
                                                               secondaryTensor:idx
                                                                          name:@"hits"]
                                          toType:MPSDataTypeFloat32
                                            name:@"hitsF"];
        MPSGraphTensor *nCorrect = [graph reshapeTensor:[graph reductionSumWithTensor:hits
                                                                                axes:@[ @0 ]
                                                                                name:@"nC"]
                                              withShape:@[ @1 ]
                                                   name:@"nC1"];
        correctOut = [graph additionWithPrimaryTensor:correctAcc
                                     secondaryTensor:nCorrect
                                                name:@"cOut"];
    }
    MPSGraphTensor *ones = [graph constantWithScalar:1.0 shape:@[ @1 ] dataType:MPSDataTypeFloat32];
    MPSGraphTensor *dlogits = [graph softMaxCrossEntropyGradientWithIncomingGradientTensor:ones
                                                                              sourceTensor:logits
                                                                              labelsTensor:onehot
                                                                                      axis:1
                                                                             reductionType:MPSGraphLossReductionTypeSum
                                                                                      name:@"dlogits"];
    MPSGraphTensor *invB = [graph constantWithScalar:1.0 / (double)B dataType:MPSDataTypeFloat32];
    dlogits = [graph multiplicationWithPrimaryTensor:dlogits secondaryTensor:invB name:@"dlogitsMean"];

    MPSGraphTensor *w2t = [graph transposeTensor:w2 permutation:@[ @1, @0 ] name:@"W2T"];
    MPSGraphTensor *dH = ampMatMul(graph, dlogits, w2t, @"dH");
    MPSGraphTensor *dHpre = [graph reLUGradientWithIncomingGradient:dH sourceTensor:hpre name:@"dHpre"];
    MPSGraphTensor *ht = [graph transposeTensor:hidden permutation:@[ @1, @0 ] name:@"HT"];
    MPSGraphTensor *dW2 = ampMatMul(graph, ht, dlogits, @"dW2");
    MPSGraphTensor *db2 = [graph reshapeTensor:[graph reductionSumWithTensor:dlogits axis:0 name:@"db2s"]
                                     withShape:@[ @(C) ]
                                          name:@"db2"];
    MPSGraphTensor *xt = [graph transposeTensor:x permutation:@[ @1, @0 ] name:@"XT"];
    MPSGraphTensor *dW1 = ampMatMul(graph, xt, dHpre, @"dW1");
    MPSGraphTensor *db1 = [graph reshapeTensor:[graph reductionSumWithTensor:dHpre axis:0 name:@"db1s"]
                                     withShape:@[ @(H) ]
                                          name:@"db1"];
    MPSGraphTensor *lrT = [graph constantWithScalar:(double)lr dataType:MPSDataTypeFloat32];
    MPSGraphTensor *w1Out = [graph subtractionWithPrimaryTensor:w1
                                               secondaryTensor:[graph multiplicationWithPrimaryTensor:dW1 secondaryTensor:lrT name:@"lrW1"]
                                                          name:@"W1n"];
    MPSGraphTensor *b1Out = [graph subtractionWithPrimaryTensor:b1
                                               secondaryTensor:[graph multiplicationWithPrimaryTensor:db1 secondaryTensor:lrT name:@"lrB1"]
                                                          name:@"b1n"];
    MPSGraphTensor *w2Out = [graph subtractionWithPrimaryTensor:w2
                                               secondaryTensor:[graph multiplicationWithPrimaryTensor:dW2 secondaryTensor:lrT name:@"lrW2"]
                                                          name:@"W2n"];
    MPSGraphTensor *b2Out = [graph subtractionWithPrimaryTensor:b2
                                               secondaryTensor:[graph multiplicationWithPrimaryTensor:db2 secondaryTensor:lrT name:@"lrB2"]
                                                          name:@"b2n"];

    cached = trackCorrect
        ? @[
            graph, x, w1, b1, w2, b2, labels, acc, correctAcc,
            w1Out, b1Out, w2Out, b2Out, accOut, correctOut
        ]
        : @[
            graph, x, w1, b1, w2, b2, labels, acc,
            w1Out, b1Out, w2Out, b2Out, accOut
        ];
    gMlpGraphs[key] = cached;
    (void)cachedMlpExecutable(cached, trackCorrect, B, inFeatures, H, C, lr);
    return cached;
}

static NSMutableDictionary<NSString *, NSArray *> *gMlpBwdGraphs;
static NSMutableDictionary<NSString *, MPSGraphExecutable *> *gMlpBwdExecutables;

static NSString *mlpBwdCacheKey(uint32_t B, uint32_t inFeatures, uint32_t H, uint32_t C,
                                bool trackCorrect) {
    return [NSString stringWithFormat:@"bwd:%u:%u:%u:%u:%d:%d",
            B, inFeatures, H, C, trackCorrect ? 1 : 0, gMixedPrecision ? 1 : 0];
}

static MPSGraphExecutable *cachedMlpBwdExecutable(NSArray *cached, bool trackCorrect,
                                                  uint32_t B, uint32_t inFeatures, uint32_t H,
                                                  uint32_t C) {
    if (@available(macOS 12.0, *)) {
        if (gMlpBwdExecutables == nil) {
            gMlpBwdExecutables = [[NSMutableDictionary alloc] init];
        }
        NSString *key = mlpBwdCacheKey(B, inFeatures, H, C, trackCorrect);
        MPSGraphExecutable *exec = gMlpBwdExecutables[key];
        if (exec != nil) return exec;

        MPSGraph *graph = cached[0];
        const int wBase = trackCorrect ? 9 : 8;
        const int targetCount = trackCorrect ? 6 : 5;
        NSMutableArray *targets =
            [NSMutableArray arrayWithCapacity:(NSUInteger)targetCount];
        for (int i = 0; i < targetCount; i++) {
            [targets addObject:cached[wBase + i]];
        }

        MPSGraphCompilationDescriptor *comp = nil;
        if (@available(macOS 12.3, *)) {
            comp = [MPSGraphCompilationDescriptor new];
            comp.optimizationLevel = MPSGraphOptimizationLevel1;
        }
        exec = [graph compileWithDevice:nil
                                  feeds:mlpFeedTypes(cached, trackCorrect, B, inFeatures, H, C)
                          targetTensors:targets
                       targetOperations:nil
                  compilationDescriptor:comp];
        if (exec == nil) return nil;
        exec.options = MPSGraphOptionsNone;
        gMlpBwdExecutables[key] = exec;
        return exec;
    }
    return nil;
}

static NSArray *cachedMlpBwdGraph(uint32_t B, uint32_t inFeatures, uint32_t H, uint32_t C,
                                  bool trackCorrect) {
    if (gMlpBwdGraphs == nil) gMlpBwdGraphs = [[NSMutableDictionary alloc] init];
    NSString *key = mlpBwdCacheKey(B, inFeatures, H, C, trackCorrect);
    NSArray *cached = gMlpBwdGraphs[key];
    if (cached != nil) return cached;

    MPSGraph *graph = [[MPSGraph alloc] init];
    graph.options = MPSGraphOptionsNone;
    MPSGraphTensor *x = [graph placeholderWithShape:@[ @(B), @(inFeatures) ] dataType:MPSDataTypeFloat32 name:@"X"];
    MPSGraphTensor *w1 = [graph placeholderWithShape:@[ @(inFeatures), @(H) ] dataType:MPSDataTypeFloat32 name:@"W1"];
    MPSGraphTensor *b1 = [graph placeholderWithShape:@[ @(H) ] dataType:MPSDataTypeFloat32 name:@"b1"];
    MPSGraphTensor *w2 = [graph placeholderWithShape:@[ @(H), @(C) ] dataType:MPSDataTypeFloat32 name:@"W2"];
    MPSGraphTensor *b2 = [graph placeholderWithShape:@[ @(C) ] dataType:MPSDataTypeFloat32 name:@"b2"];
    MPSGraphTensor *labels = [graph placeholderWithShape:@[ @(B) ] dataType:MPSDataTypeFloat32 name:@"y"];
    MPSGraphTensor *acc = [graph placeholderWithShape:@[ @1 ] dataType:MPSDataTypeFloat32 name:@"acc"];
    MPSGraphTensor *correctAcc = trackCorrect
        ? [graph placeholderWithShape:@[ @1 ] dataType:MPSDataTypeFloat32 name:@"corr"]
        : nil;

    MPSGraphTensor *b1r = [graph reshapeTensor:b1 withShape:@[ @1, @(H) ] name:@"b1r"];
    MPSGraphTensor *hpre = [graph additionWithPrimaryTensor:ampMatMul(graph, x, w1, @"XW1")
                                           secondaryTensor:b1r
                                                      name:@"Hpre"];
    MPSGraphTensor *hidden = [graph reLUWithTensor:hpre name:@"H"];
    MPSGraphTensor *b2r = [graph reshapeTensor:b2 withShape:@[ @1, @(C) ] name:@"b2r"];
    MPSGraphTensor *logits = [graph additionWithPrimaryTensor:ampMatMul(graph, hidden, w2, @"HW2")
                                            secondaryTensor:b2r
                                                       name:@"logits"];
    MPSGraphTensor *idx = [graph castTensor:labels toType:MPSDataTypeInt32 name:@"idx"];
    MPSGraphTensor *onehot = [graph oneHotWithIndicesTensor:idx
                                                      depth:C
                                                   dataType:MPSDataTypeFloat32
                                                       name:@"oh"];
    MPSGraphTensor *loss = [graph softMaxCrossEntropyWithSourceTensor:logits
                                                        labelsTensor:onehot
                                                                axis:1
                                                       reductionType:MPSGraphLossReductionTypeSum
                                                                name:@"loss"];
    MPSGraphTensor *accOut = [graph additionWithPrimaryTensor:acc secondaryTensor:loss name:@"accOut"];
    MPSGraphTensor *correctOut = nil;
    if (trackCorrect) {
        MPSGraphTensor *pred = [graph reshapeTensor:[graph reductionArgMaximumWithTensor:logits
                                                                                   axis:1
                                                                                   name:@"pred"]
                                          withShape:@[ @(B) ]
                                               name:@"pred1"];
        MPSGraphTensor *hits = [graph castTensor:[graph equalWithPrimaryTensor:pred
                                                               secondaryTensor:idx
                                                                          name:@"hits"]
                                          toType:MPSDataTypeFloat32
                                            name:@"hitsF"];
        MPSGraphTensor *nCorrect = [graph reshapeTensor:[graph reductionSumWithTensor:hits
                                                                                axes:@[ @0 ]
                                                                                name:@"nC"]
                                              withShape:@[ @1 ]
                                                   name:@"nC1"];
        correctOut = [graph additionWithPrimaryTensor:correctAcc
                                     secondaryTensor:nCorrect
                                                name:@"cOut"];
    }
    MPSGraphTensor *ones = [graph constantWithScalar:1.0 shape:@[ @1 ] dataType:MPSDataTypeFloat32];
    MPSGraphTensor *dlogits = [graph softMaxCrossEntropyGradientWithIncomingGradientTensor:ones
                                                                              sourceTensor:logits
                                                                              labelsTensor:onehot
                                                                                      axis:1
                                                                             reductionType:MPSGraphLossReductionTypeSum
                                                                                      name:@"dlogits"];
    MPSGraphTensor *invB = [graph constantWithScalar:1.0 / (double)B dataType:MPSDataTypeFloat32];
    dlogits = [graph multiplicationWithPrimaryTensor:dlogits secondaryTensor:invB name:@"dlogitsMean"];

    MPSGraphTensor *w2t = [graph transposeTensor:w2 permutation:@[ @1, @0 ] name:@"W2T"];
    MPSGraphTensor *dH = ampMatMul(graph, dlogits, w2t, @"dH");
    MPSGraphTensor *dHpre = [graph reLUGradientWithIncomingGradient:dH sourceTensor:hpre name:@"dHpre"];
    MPSGraphTensor *ht = [graph transposeTensor:hidden permutation:@[ @1, @0 ] name:@"HT"];
    MPSGraphTensor *dW2 = ampMatMul(graph, ht, dlogits, @"dW2");
    MPSGraphTensor *db2 = [graph reshapeTensor:[graph reductionSumWithTensor:dlogits axis:0 name:@"db2s"]
                                     withShape:@[ @(C) ]
                                          name:@"db2"];
    MPSGraphTensor *xt = [graph transposeTensor:x permutation:@[ @1, @0 ] name:@"XT"];
    MPSGraphTensor *dW1 = ampMatMul(graph, xt, dHpre, @"dW1");
    MPSGraphTensor *db1 = [graph reshapeTensor:[graph reductionSumWithTensor:dHpre axis:0 name:@"db1s"]
                                     withShape:@[ @(H) ]
                                          name:@"db1"];

    cached = trackCorrect
        ? @[
            graph, x, w1, b1, w2, b2, labels, acc, correctAcc,
            dW1, db1, dW2, db2, accOut, correctOut
        ]
        : @[
            graph, x, w1, b1, w2, b2, labels, acc,
            dW1, db1, dW2, db2, accOut
        ];
    gMlpBwdGraphs[key] = cached;
    (void)cachedMlpBwdExecutable(cached, trackCorrect, B, inFeatures, H, C);
    return cached;
}

bool jaiGpuMlpBwdStep(JaiGpuBuffer *x, size_t xOff, JaiGpuBuffer *w1, size_t w1Off,
                      JaiGpuBuffer *b1, size_t b1Off, JaiGpuBuffer *w2, size_t w2Off,
                      JaiGpuBuffer *b2, size_t b2Off, JaiGpuBuffer *labels, size_t labOff,
                      JaiGpuBuffer *gW1, size_t gW1Off, JaiGpuBuffer *gB1, size_t gB1Off,
                      JaiGpuBuffer *gW2, size_t gW2Off, JaiGpuBuffer *gB2, size_t gB2Off,
                      JaiGpuBuffer *lossAcc, size_t lossOff, JaiGpuBuffer *correctAcc,
                      size_t correctOff, uint32_t B, uint32_t inFeatures, uint32_t H,
                      uint32_t C) {
    if (x == NULL || w1 == NULL || b1 == NULL || w2 == NULL || b2 == NULL ||
        labels == NULL || gW1 == NULL || gB1 == NULL || gW2 == NULL || gB2 == NULL ||
        lossAcc == NULL) {
        return false;
    }
    const bool trackCorrect = correctAcc != NULL;
    if (B == 0 || inFeatures == 0 || H == 0 || C == 0) return false;
    if (!ensureDevice()) return false;

    const size_t accBytes = sizeof(float);

    @autoreleasepool {
        @synchronized(gQueue) {
            NSArray *cached = cachedMlpBwdGraph(B, inFeatures, H, C, trackCorrect);
            const NSUInteger expected = trackCorrect ? 15u : 13u;
            if (cached == nil || cached.count != expected) return false;
            MPSGraph *graph = cached[0];
            gMlpScratchAcc = growScratch(gMlpScratchAcc, &gMlpCapAcc, accBytes);
            if (trackCorrect) {
                gMlpScratchCorrect = growScratch(gMlpScratchCorrect, &gMlpCapCorrect, accBytes);
            }
            NSArray *w1Shape = @[ @(inFeatures), @(H) ];
            NSArray *b1Shape = @[ @(H) ];
            NSArray *w2Shape = @[ @(H), @(C) ];
            NSArray *b2Shape = @[ @(C) ];
            MPSGraphTensorData *dx = graphData(x, xOff, @[ @(B), @(inFeatures) ]);
            MPSGraphTensorData *dw1 = graphData(w1, w1Off, w1Shape);
            MPSGraphTensorData *db1 = graphData(b1, b1Off, b1Shape);
            MPSGraphTensorData *dw2 = graphData(w2, w2Off, w2Shape);
            MPSGraphTensorData *db2 = graphData(b2, b2Off, b2Shape);
            MPSGraphTensorData *dy = graphData(labels, labOff, @[ @(B) ]);
            const bool accFromLive = !trackCorrect && gMlpAccSide == 0;
            MPSGraphTensorData *dacc = trackCorrect
                ? graphData(lossAcc, lossOff, @[ @1 ])
                : (accFromLive ? graphData(lossAcc, lossOff, @[ @1 ])
                               : graphDataMTL(gMlpScratchAcc, @[ @1 ]));
            MPSGraphTensorData *dcorr = trackCorrect
                ? graphData(correctAcc, correctOff, @[ @1 ])
                : nil;
            const int wBase = trackCorrect ? 9 : 8;
            MPSGraphTensorData *rgW1 = graphData(gW1, gW1Off, w1Shape);
            MPSGraphTensorData *rgB1 = graphData(gB1, gB1Off, b1Shape);
            MPSGraphTensorData *rgW2 = graphData(gW2, gW2Off, w2Shape);
            MPSGraphTensorData *rgB2 = graphData(gB2, gB2Off, b2Shape);
            MPSGraphTensorData *racc = trackCorrect
                ? graphDataMTL(gMlpScratchAcc, @[ @1 ])
                : (accFromLive ? graphDataMTL(gMlpScratchAcc, @[ @1 ])
                               : graphData(lossAcc, lossOff, @[ @1 ]));
            MPSGraphTensorData *rcorr = trackCorrect
                ? graphDataMTL(gMlpScratchCorrect, @[ @1 ])
                : nil;
            if (dx == nil || dw1 == nil || db1 == nil || dw2 == nil || db2 == nil ||
                dy == nil || dacc == nil || rgW1 == nil || rgB1 == nil ||
                rgW2 == nil || rgB2 == nil || racc == nil) {
                return false;
            }
            if (trackCorrect && (dcorr == nil || rcorr == nil)) return false;
            NSMutableDictionary *feeds = [@{
                cached[1] : dx,  cached[2] : dw1, cached[3] : db1,
                cached[4] : dw2, cached[5] : db2, cached[6] : dy,
                cached[7] : dacc
            } mutableCopy];
            if (trackCorrect) feeds[cached[8]] = dcorr;
            NSMutableDictionary *results = [@{
                cached[wBase + 0] : rgW1, cached[wBase + 1] : rgB1,
                cached[wBase + 2] : rgW2, cached[wBase + 3] : rgB2,
                cached[wBase + 4] : racc
            } mutableCopy];
            if (trackCorrect) results[cached[wBase + 5]] = rcorr;
            MPSGraphExecutable *exec =
                cachedMlpBwdExecutable(cached, trackCorrect, B, inFeatures, H, C);
            if (exec == nil ||
                !encodeMlpExecutableOnAsync(exec, feeds, results)) {
                if (!encodeGraphOnAsync(graph, feeds, results)) return false;
            }
            if (trackCorrect) {
                __unsafe_unretained id<MTLBuffer> accSrcs[] = {
                    gMlpScratchAcc, gMlpScratchCorrect
                };
                JaiGpuBuffer *accDsts[] = {lossAcc, correctAcc};
                size_t accOffs[] = {lossOff, correctOff};
                size_t accSizes[] = {accBytes, accBytes};
                if (!blitMany(accSrcs, accDsts, accOffs, accSizes, 2)) return false;
            }
            gMlpLiveAcc = lossAcc;
            gMlpLiveAccOff = lossOff;
            if (!trackCorrect) {
                gMlpAccSide = accFromLive ? 1 : 0;
            }
        }
        return true;
    }
}

static NSMutableDictionary<NSString *, NSArray *> *gMlp3Graphs;
static NSMutableDictionary<NSString *, MPSGraphExecutable *> *gMlp3Executables;
static NSMutableDictionary<NSString *, NSArray *> *gMlp3BwdGraphs;
static NSMutableDictionary<NSString *, MPSGraphExecutable *> *gMlp3BwdExecutables;

static NSString *mlp3CacheKey(uint32_t B, uint32_t inFeatures, uint32_t H1, uint32_t H2,
                              uint32_t H3, uint32_t C, float lr, bool trackCorrect) {
    return [NSString stringWithFormat:@"3:%u:%u:%u:%u:%u:%u:%.8f:%d:%d",
            B, inFeatures, H1, H2, H3, C, lr, trackCorrect ? 1 : 0, gMixedPrecision ? 1 : 0];
}

static NSString *mlp3BwdCacheKey(uint32_t B, uint32_t inFeatures, uint32_t H1, uint32_t H2,
                                   uint32_t H3, uint32_t C, bool trackCorrect) {
    return [NSString stringWithFormat:@"3bwd:%u:%u:%u:%u:%u:%u:%d:%d",
            B, inFeatures, H1, H2, H3, C, trackCorrect ? 1 : 0, gMixedPrecision ? 1 : 0];
}

static MPSGraphTensorShapedTypeDictionary *mlp3FeedTypes(NSArray *cached, bool trackCorrect,
                                                           uint32_t B, uint32_t inFeatures,
                                                           uint32_t H1, uint32_t H2, uint32_t H3,
                                                           uint32_t C) {
    NSMutableDictionary *feeds = [@{
        cached[1] : mlpShapedType(@[ @(B), @(inFeatures) ]),
        cached[2] : mlpShapedType(@[ @(inFeatures), @(H1) ]),
        cached[3] : mlpShapedType(@[ @(H1) ]),
        cached[4] : mlpShapedType(@[ @(H1), @(H2) ]),
        cached[5] : mlpShapedType(@[ @(H2) ]),
        cached[6] : mlpShapedType(@[ @(H2), @(H3) ]),
        cached[7] : mlpShapedType(@[ @(H3) ]),
        cached[8] : mlpShapedType(@[ @(H3), @(C) ]),
        cached[9] : mlpShapedType(@[ @(C) ]),
        cached[10] : mlpShapedType(@[ @(B) ]),
        cached[11] : mlpShapedType(@[ @1 ]),
    } mutableCopy];
    if (trackCorrect) feeds[cached[12]] = mlpShapedType(@[ @1 ]);
    return feeds;
}

static MPSGraphExecutable *cachedMlp3Executable(NSArray *cached, bool trackCorrect,
                                                uint32_t B, uint32_t inFeatures, uint32_t H1,
                                                uint32_t H2, uint32_t H3, uint32_t C, float lr) {
    if (@available(macOS 12.0, *)) {
        if (gMlp3Executables == nil) {
            gMlp3Executables = [[NSMutableDictionary alloc] init];
        }
        NSString *key = mlp3CacheKey(B, inFeatures, H1, H2, H3, C, lr, trackCorrect);
        MPSGraphExecutable *exec = gMlp3Executables[key];
        if (exec != nil) return exec;

        MPSGraph *graph = cached[0];
        const int wBase = trackCorrect ? 13 : 12;
        const int targetCount = trackCorrect ? 10 : 9;
        NSMutableArray *targets =
            [NSMutableArray arrayWithCapacity:(NSUInteger)targetCount];
        for (int i = 0; i < targetCount; i++) {
            [targets addObject:cached[wBase + i]];
        }

        MPSGraphCompilationDescriptor *comp = nil;
        if (@available(macOS 12.3, *)) {
            comp = [MPSGraphCompilationDescriptor new];
            comp.optimizationLevel = MPSGraphOptimizationLevel1;
        }
        exec = [graph compileWithDevice:nil
                                  feeds:mlp3FeedTypes(cached, trackCorrect, B, inFeatures,
                                                      H1, H2, H3, C)
                          targetTensors:targets
                       targetOperations:nil
                  compilationDescriptor:comp];
        if (exec == nil) return nil;
        exec.options = MPSGraphOptionsNone;
        gMlp3Executables[key] = exec;
        return exec;
    }
    return nil;
}

static MPSGraphExecutable *cachedMlp3BwdExecutable(NSArray *cached, bool trackCorrect,
                                                   uint32_t B, uint32_t inFeatures, uint32_t H1,
                                                   uint32_t H2, uint32_t H3, uint32_t C) {
    if (@available(macOS 12.0, *)) {
        if (gMlp3BwdExecutables == nil) {
            gMlp3BwdExecutables = [[NSMutableDictionary alloc] init];
        }
        NSString *key = mlp3BwdCacheKey(B, inFeatures, H1, H2, H3, C, trackCorrect);
        MPSGraphExecutable *exec = gMlp3BwdExecutables[key];
        if (exec != nil) return exec;

        MPSGraph *graph = cached[0];
        const int wBase = trackCorrect ? 13 : 12;
        const int targetCount = trackCorrect ? 10 : 9;
        NSMutableArray *targets =
            [NSMutableArray arrayWithCapacity:(NSUInteger)targetCount];
        for (int i = 0; i < targetCount; i++) {
            [targets addObject:cached[wBase + i]];
        }

        MPSGraphCompilationDescriptor *comp = nil;
        if (@available(macOS 12.3, *)) {
            comp = [MPSGraphCompilationDescriptor new];
            comp.optimizationLevel = MPSGraphOptimizationLevel1;
        }
        exec = [graph compileWithDevice:nil
                                  feeds:mlp3FeedTypes(cached, trackCorrect, B, inFeatures,
                                                      H1, H2, H3, C)
                          targetTensors:targets
                       targetOperations:nil
                  compilationDescriptor:comp];
        if (exec == nil) return nil;
        exec.options = MPSGraphOptionsNone;
        gMlp3BwdExecutables[key] = exec;
        return exec;
    }
    return nil;
}

static NSArray *buildMlp3GraphArrays(MPSGraph *graph, MPSGraphTensor *accOut,
                                     MPSGraphTensor *correctOut, MPSGraphTensor *dW1,
                                     MPSGraphTensor *db1, MPSGraphTensor *dW2,
                                     MPSGraphTensor *db2, MPSGraphTensor *dW3,
                                     MPSGraphTensor *db3, MPSGraphTensor *dW4,
                                     MPSGraphTensor *db4, MPSGraphTensor *w1Out,
                                     MPSGraphTensor *b1Out, MPSGraphTensor *w2Out,
                                     MPSGraphTensor *b2Out, MPSGraphTensor *w3Out,
                                     MPSGraphTensor *b3Out, MPSGraphTensor *w4Out,
                                     MPSGraphTensor *b4Out, MPSGraphTensor *x,
                                     MPSGraphTensor *w1, MPSGraphTensor *b1,
                                     MPSGraphTensor *w2, MPSGraphTensor *b2,
                                     MPSGraphTensor *w3, MPSGraphTensor *b3,
                                     MPSGraphTensor *w4, MPSGraphTensor *b4,
                                     MPSGraphTensor *labels, MPSGraphTensor *acc,
                                     MPSGraphTensor *correctAcc, bool trackCorrect,
                                     bool sgdUpdate) {
    if (trackCorrect) {
        if (sgdUpdate) {
            return @[
                graph, x, w1, b1, w2, b2, w3, b3, w4, b4, labels, acc, correctAcc,
                w1Out, b1Out, w2Out, b2Out, w3Out, b3Out, w4Out, b4Out, accOut, correctOut
            ];
        }
        return @[
            graph, x, w1, b1, w2, b2, w3, b3, w4, b4, labels, acc, correctAcc,
            dW1, db1, dW2, db2, dW3, db3, dW4, db4, accOut, correctOut
        ];
    }
    if (sgdUpdate) {
        return @[
            graph, x, w1, b1, w2, b2, w3, b3, w4, b4, labels, acc,
            w1Out, b1Out, w2Out, b2Out, w3Out, b3Out, w4Out, b4Out, accOut
        ];
    }
    return @[
        graph, x, w1, b1, w2, b2, w3, b3, w4, b4, labels, acc,
        dW1, db1, dW2, db2, dW3, db3, dW4, db4, accOut
    ];
}

static NSArray *cachedMlp3GraphCore(uint32_t B, uint32_t inFeatures, uint32_t H1, uint32_t H2,
                                    uint32_t H3, uint32_t C, float lr, bool trackCorrect,
                                    bool sgdUpdate, NSMutableDictionary *store, NSString *key) {
    NSArray *cached = store[key];
    if (cached != nil) return cached;

    MPSGraph *graph = [[MPSGraph alloc] init];
    graph.options = MPSGraphOptionsNone;
    MPSGraphTensor *x = [graph placeholderWithShape:@[ @(B), @(inFeatures) ] dataType:MPSDataTypeFloat32 name:@"X"];
    MPSGraphTensor *w1 = [graph placeholderWithShape:@[ @(inFeatures), @(H1) ] dataType:MPSDataTypeFloat32 name:@"W1"];
    MPSGraphTensor *b1 = [graph placeholderWithShape:@[ @(H1) ] dataType:MPSDataTypeFloat32 name:@"b1"];
    MPSGraphTensor *w2 = [graph placeholderWithShape:@[ @(H1), @(H2) ] dataType:MPSDataTypeFloat32 name:@"W2"];
    MPSGraphTensor *b2 = [graph placeholderWithShape:@[ @(H2) ] dataType:MPSDataTypeFloat32 name:@"b2"];
    MPSGraphTensor *w3 = [graph placeholderWithShape:@[ @(H2), @(H3) ] dataType:MPSDataTypeFloat32 name:@"W3"];
    MPSGraphTensor *b3 = [graph placeholderWithShape:@[ @(H3) ] dataType:MPSDataTypeFloat32 name:@"b3"];
    MPSGraphTensor *w4 = [graph placeholderWithShape:@[ @(H3), @(C) ] dataType:MPSDataTypeFloat32 name:@"W4"];
    MPSGraphTensor *b4 = [graph placeholderWithShape:@[ @(C) ] dataType:MPSDataTypeFloat32 name:@"b4"];
    MPSGraphTensor *labels = [graph placeholderWithShape:@[ @(B) ] dataType:MPSDataTypeFloat32 name:@"y"];
    MPSGraphTensor *acc = [graph placeholderWithShape:@[ @1 ] dataType:MPSDataTypeFloat32 name:@"acc"];
    MPSGraphTensor *correctAcc = trackCorrect
        ? [graph placeholderWithShape:@[ @1 ] dataType:MPSDataTypeFloat32 name:@"corr"]
        : nil;

    MPSGraphTensor *b1r = [graph reshapeTensor:b1 withShape:@[ @1, @(H1) ] name:@"b1r"];
    MPSGraphTensor *h1pre = [graph additionWithPrimaryTensor:ampMatMul(graph, x, w1, @"XW1")
                                           secondaryTensor:b1r
                                                      name:@"H1pre"];
    MPSGraphTensor *h1 = [graph reLUWithTensor:h1pre name:@"H1"];
    MPSGraphTensor *b2r = [graph reshapeTensor:b2 withShape:@[ @1, @(H2) ] name:@"b2r"];
    MPSGraphTensor *h2pre = [graph additionWithPrimaryTensor:ampMatMul(graph, h1, w2, @"H1W2")
                                           secondaryTensor:b2r
                                                      name:@"H2pre"];
    MPSGraphTensor *h2 = [graph reLUWithTensor:h2pre name:@"H2"];
    MPSGraphTensor *b3r = [graph reshapeTensor:b3 withShape:@[ @1, @(H3) ] name:@"b3r"];
    MPSGraphTensor *h3pre = [graph additionWithPrimaryTensor:ampMatMul(graph, h2, w3, @"H2W3")
                                           secondaryTensor:b3r
                                                      name:@"H3pre"];
    MPSGraphTensor *h3 = [graph reLUWithTensor:h3pre name:@"H3"];
    MPSGraphTensor *b4r = [graph reshapeTensor:b4 withShape:@[ @1, @(C) ] name:@"b4r"];
    MPSGraphTensor *logits = [graph additionWithPrimaryTensor:ampMatMul(graph, h3, w4, @"H3W4")
                                            secondaryTensor:b4r
                                                       name:@"logits"];
    MPSGraphTensor *idx = [graph castTensor:labels toType:MPSDataTypeInt32 name:@"idx"];
    MPSGraphTensor *onehot = [graph oneHotWithIndicesTensor:idx
                                                      depth:C
                                                   dataType:MPSDataTypeFloat32
                                                       name:@"oh"];
    MPSGraphTensor *loss = [graph softMaxCrossEntropyWithSourceTensor:logits
                                                        labelsTensor:onehot
                                                                axis:1
                                                       reductionType:MPSGraphLossReductionTypeSum
                                                                name:@"loss"];
    MPSGraphTensor *accOut = [graph additionWithPrimaryTensor:acc secondaryTensor:loss name:@"accOut"];
    MPSGraphTensor *correctOut = nil;
    if (trackCorrect) {
        MPSGraphTensor *pred = [graph reshapeTensor:[graph reductionArgMaximumWithTensor:logits
                                                                                   axis:1
                                                                                   name:@"pred"]
                                          withShape:@[ @(B) ]
                                               name:@"pred1"];
        MPSGraphTensor *hits = [graph castTensor:[graph equalWithPrimaryTensor:pred
                                                               secondaryTensor:idx
                                                                          name:@"hits"]
                                          toType:MPSDataTypeFloat32
                                            name:@"hitsF"];
        MPSGraphTensor *nCorrect = [graph reshapeTensor:[graph reductionSumWithTensor:hits
                                                                                axes:@[ @0 ]
                                                                                name:@"nC"]
                                              withShape:@[ @1 ]
                                                   name:@"nC1"];
        correctOut = [graph additionWithPrimaryTensor:correctAcc
                                     secondaryTensor:nCorrect
                                                name:@"cOut"];
    }
    MPSGraphTensor *ones = [graph constantWithScalar:1.0 shape:@[ @1 ] dataType:MPSDataTypeFloat32];
    MPSGraphTensor *dlogits = [graph softMaxCrossEntropyGradientWithIncomingGradientTensor:ones
                                                                              sourceTensor:logits
                                                                              labelsTensor:onehot
                                                                                      axis:1
                                                                             reductionType:MPSGraphLossReductionTypeSum
                                                                                      name:@"dlogits"];
    MPSGraphTensor *invB = [graph constantWithScalar:1.0 / (double)B dataType:MPSDataTypeFloat32];
    dlogits = [graph multiplicationWithPrimaryTensor:dlogits secondaryTensor:invB name:@"dlogitsMean"];

    MPSGraphTensor *w4t = [graph transposeTensor:w4 permutation:@[ @1, @0 ] name:@"W4T"];
    MPSGraphTensor *dH3 = ampMatMul(graph, dlogits, w4t, @"dH3");
    MPSGraphTensor *dH3pre = [graph reLUGradientWithIncomingGradient:dH3 sourceTensor:h3pre name:@"dH3pre"];
    MPSGraphTensor *h3t = [graph transposeTensor:h3 permutation:@[ @1, @0 ] name:@"H3T"];
    MPSGraphTensor *dW4 = ampMatMul(graph, h3t, dlogits, @"dW4");
    MPSGraphTensor *db4 = [graph reshapeTensor:[graph reductionSumWithTensor:dlogits axis:0 name:@"db4s"]
                                     withShape:@[ @(C) ]
                                          name:@"db4"];
    MPSGraphTensor *w3t = [graph transposeTensor:w3 permutation:@[ @1, @0 ] name:@"W3T"];
    MPSGraphTensor *dH2 = ampMatMul(graph, dH3pre, w3t, @"dH2");
    MPSGraphTensor *dH2pre = [graph reLUGradientWithIncomingGradient:dH2 sourceTensor:h2pre name:@"dH2pre"];
    MPSGraphTensor *h2t = [graph transposeTensor:h2 permutation:@[ @1, @0 ] name:@"H2T"];
    MPSGraphTensor *dW3 = ampMatMul(graph, h2t, dH3pre, @"dW3");
    MPSGraphTensor *db3 = [graph reshapeTensor:[graph reductionSumWithTensor:dH3pre axis:0 name:@"db3s"]
                                     withShape:@[ @(H3) ]
                                          name:@"db3"];
    MPSGraphTensor *w2t = [graph transposeTensor:w2 permutation:@[ @1, @0 ] name:@"W2T"];
    MPSGraphTensor *dH1 = ampMatMul(graph, dH2pre, w2t, @"dH1");
    MPSGraphTensor *dH1pre = [graph reLUGradientWithIncomingGradient:dH1 sourceTensor:h1pre name:@"dH1pre"];
    MPSGraphTensor *h1t = [graph transposeTensor:h1 permutation:@[ @1, @0 ] name:@"H1T"];
    MPSGraphTensor *dW2 = ampMatMul(graph, h1t, dH2pre, @"dW2");
    MPSGraphTensor *db2 = [graph reshapeTensor:[graph reductionSumWithTensor:dH2pre axis:0 name:@"db2s"]
                                     withShape:@[ @(H2) ]
                                          name:@"db2"];
    MPSGraphTensor *xt = [graph transposeTensor:x permutation:@[ @1, @0 ] name:@"XT"];
    MPSGraphTensor *dW1 = ampMatMul(graph, xt, dH1pre, @"dW1");
    MPSGraphTensor *db1 = [graph reshapeTensor:[graph reductionSumWithTensor:dH1pre axis:0 name:@"db1s"]
                                     withShape:@[ @(H1) ]
                                          name:@"db1"];

    MPSGraphTensor *w1Out = nil;
    MPSGraphTensor *b1Out = nil;
    MPSGraphTensor *w2Out = nil;
    MPSGraphTensor *b2Out = nil;
    MPSGraphTensor *w3Out = nil;
    MPSGraphTensor *b3Out = nil;
    MPSGraphTensor *w4Out = nil;
    MPSGraphTensor *b4Out = nil;
    if (sgdUpdate) {
        MPSGraphTensor *lrT = [graph constantWithScalar:(double)lr dataType:MPSDataTypeFloat32];
        w1Out = [graph subtractionWithPrimaryTensor:w1
                                  secondaryTensor:[graph multiplicationWithPrimaryTensor:dW1 secondaryTensor:lrT name:@"lrW1"]
                                             name:@"W1n"];
        b1Out = [graph subtractionWithPrimaryTensor:b1
                                  secondaryTensor:[graph multiplicationWithPrimaryTensor:db1 secondaryTensor:lrT name:@"lrB1"]
                                             name:@"b1n"];
        w2Out = [graph subtractionWithPrimaryTensor:w2
                                  secondaryTensor:[graph multiplicationWithPrimaryTensor:dW2 secondaryTensor:lrT name:@"lrW2"]
                                             name:@"W2n"];
        b2Out = [graph subtractionWithPrimaryTensor:b2
                                  secondaryTensor:[graph multiplicationWithPrimaryTensor:db2 secondaryTensor:lrT name:@"lrB2"]
                                             name:@"b2n"];
        w3Out = [graph subtractionWithPrimaryTensor:w3
                                  secondaryTensor:[graph multiplicationWithPrimaryTensor:dW3 secondaryTensor:lrT name:@"lrW3"]
                                             name:@"W3n"];
        b3Out = [graph subtractionWithPrimaryTensor:b3
                                  secondaryTensor:[graph multiplicationWithPrimaryTensor:db3 secondaryTensor:lrT name:@"lrB3"]
                                             name:@"b3n"];
        w4Out = [graph subtractionWithPrimaryTensor:w4
                                  secondaryTensor:[graph multiplicationWithPrimaryTensor:dW4 secondaryTensor:lrT name:@"lrW4"]
                                             name:@"W4n"];
        b4Out = [graph subtractionWithPrimaryTensor:b4
                                  secondaryTensor:[graph multiplicationWithPrimaryTensor:db4 secondaryTensor:lrT name:@"lrB4"]
                                             name:@"b4n"];
    }

    cached = buildMlp3GraphArrays(
        graph, accOut, correctOut, dW1, db1, dW2, db2, dW3, db3, dW4, db4,
        w1Out, b1Out, w2Out, b2Out, w3Out, b3Out, w4Out, b4Out,
        x, w1, b1, w2, b2, w3, b3, w4, b4, labels, acc, correctAcc,
        trackCorrect, sgdUpdate);
    store[key] = cached;
    if (sgdUpdate) {
        (void)cachedMlp3Executable(cached, trackCorrect, B, inFeatures, H1, H2, H3, C, lr);
    } else {
        (void)cachedMlp3BwdExecutable(cached, trackCorrect, B, inFeatures, H1, H2, H3, C);
    }
    return cached;
}

static NSArray *cachedMlp3Graph(uint32_t B, uint32_t inFeatures, uint32_t H1, uint32_t H2,
                                uint32_t H3, uint32_t C, float lr, bool trackCorrect) {
    if (gMlp3Graphs == nil) gMlp3Graphs = [[NSMutableDictionary alloc] init];
    NSString *key = mlp3CacheKey(B, inFeatures, H1, H2, H3, C, lr, trackCorrect);
    return cachedMlp3GraphCore(B, inFeatures, H1, H2, H3, C, lr, trackCorrect, true, gMlp3Graphs, key);
}

static NSArray *cachedMlp3BwdGraph(uint32_t B, uint32_t inFeatures, uint32_t H1, uint32_t H2,
                                   uint32_t H3, uint32_t C, bool trackCorrect) {
    if (gMlp3BwdGraphs == nil) gMlp3BwdGraphs = [[NSMutableDictionary alloc] init];
    NSString *key = mlp3BwdCacheKey(B, inFeatures, H1, H2, H3, C, trackCorrect);
    return cachedMlp3GraphCore(B, inFeatures, H1, H2, H3, C, 0.0f, trackCorrect, false, gMlp3BwdGraphs, key);
}

static bool mlp3WeightBytes(uint32_t inFeatures, uint32_t H1, uint32_t H2, uint32_t H3,
                            uint32_t C, size_t wBytes[4], size_t bBytes[4]) {
    wBytes[0] = (size_t)inFeatures * (size_t)H1 * sizeof(float);
    wBytes[1] = (size_t)H1 * (size_t)H2 * sizeof(float);
    wBytes[2] = (size_t)H2 * (size_t)H3 * sizeof(float);
    wBytes[3] = (size_t)H3 * (size_t)C * sizeof(float);
    bBytes[0] = (size_t)H1 * sizeof(float);
    bBytes[1] = (size_t)H2 * sizeof(float);
    bBytes[2] = (size_t)H3 * sizeof(float);
    bBytes[3] = (size_t)C * sizeof(float);
    return true;
}

static bool mlp3LiveMatches(JaiGpuBuffer *w[4], JaiGpuBuffer *b[4], size_t wOff[4],
                            size_t bOff[4]) {
    for (int i = 0; i < 4; i++) {
        if (w[i] != gMlp3LiveW[i] || b[i] != gMlp3LiveB[i] ||
            wOff[i] != gMlp3LiveWOff[i] || bOff[i] != gMlp3LiveBOff[i]) {
            return false;
        }
    }
    return true;
}

static bool commitMlp3WeightsLocked(void) {
    if (gMlp3Side == 0) return true;
    __unsafe_unretained id<MTLBuffer> srcs[8];
    JaiGpuBuffer *dsts[8];
    size_t offs[8];
    size_t sizes[8];
    for (int i = 0; i < 4; i++) {
        srcs[i * 2] = gMlp3ScratchW[i];
        srcs[i * 2 + 1] = gMlp3ScratchB[i];
        dsts[i * 2] = gMlp3LiveW[i];
        dsts[i * 2 + 1] = gMlp3LiveB[i];
        offs[i * 2] = gMlp3LiveWOff[i];
        offs[i * 2 + 1] = gMlp3LiveBOff[i];
        sizes[i * 2] = gMlp3LiveWBytes[i];
        sizes[i * 2 + 1] = gMlp3LiveBBytes[i];
    }
    if (!blitMany(srcs, dsts, offs, sizes, 8)) return false;
    gMlp3Side = 0;
    return true;
}

static bool encodeMlp3Step(
    NSArray *cached,
    bool trackCorrect,
    bool sgdUpdate,
    uint32_t B,
    uint32_t inFeatures,
    uint32_t H1,
    uint32_t H2,
    uint32_t H3,
    uint32_t C,
    float lr,
    JaiGpuBuffer *x,
    size_t xOff,
    JaiGpuBuffer *w[4],
    size_t wOff[4],
    JaiGpuBuffer *b[4],
    size_t bOff[4],
    JaiGpuBuffer *labels,
    size_t labOff,
    JaiGpuBuffer *lossAcc,
    size_t lossOff,
    JaiGpuBuffer *correctAcc,
    size_t correctOff,
    JaiGpuBuffer *gW[4],
    size_t gWOff[4],
    JaiGpuBuffer *gB[4],
    size_t gBOff[4]) {
    const NSUInteger expected = trackCorrect ? 23u : 21u;
    if (cached == nil || cached.count != expected) return false;
    MPSGraph *graph = cached[0];
    size_t wBytes[4];
    size_t bBytes[4];
    mlp3WeightBytes(inFeatures, H1, H2, H3, C, wBytes, bBytes);
    const size_t accBytes = sizeof(float);
    const bool fromLive = sgdUpdate && gMlp3Side == 0;
    gMlpScratchAcc = growScratch(gMlpScratchAcc, &gMlpCapAcc, accBytes);
    if (trackCorrect) {
        gMlpScratchCorrect = growScratch(gMlpScratchCorrect, &gMlpCapCorrect, accBytes);
    }
    if (sgdUpdate) {
        for (int i = 0; i < 4; i++) {
            gMlp3ScratchW[i] = growScratch(gMlp3ScratchW[i], &gMlp3CapW[i], wBytes[i]);
            gMlp3ScratchB[i] = growScratch(gMlp3ScratchB[i], &gMlp3CapB[i], bBytes[i]);
        }
    }
    NSArray *w1Shape = @[ @(inFeatures), @(H1) ];
    NSArray *w2Shape = @[ @(H1), @(H2) ];
    NSArray *w3Shape = @[ @(H2), @(H3) ];
    NSArray *w4Shape = @[ @(H3), @(C) ];
    NSArray *b1Shape = @[ @(H1) ];
    NSArray *b2Shape = @[ @(H2) ];
    NSArray *b3Shape = @[ @(H3) ];
    NSArray *cVec = @[ @(C) ];
    NSArray *wShapes[4] = {w1Shape, w2Shape, w3Shape, w4Shape};
    NSArray *bShapes[4] = {b1Shape, b2Shape, b3Shape, cVec};
    MPSGraphTensorData *dx = graphData(x, xOff, @[ @(B), @(inFeatures) ]);
    MPSGraphTensorData *dy = graphData(labels, labOff, @[ @(B) ]);
    const bool accFromLive = !trackCorrect && gMlpAccSide == 0;
    MPSGraphTensorData *dacc = trackCorrect
        ? graphData(lossAcc, lossOff, @[ @1 ])
        : (accFromLive ? graphData(lossAcc, lossOff, @[ @1 ])
                       : graphDataMTL(gMlpScratchAcc, @[ @1 ]));
    MPSGraphTensorData *dcorr = trackCorrect
        ? graphData(correctAcc, correctOff, @[ @1 ])
        : nil;
    MPSGraphTensorData *dws[4];
    MPSGraphTensorData *dbs[4];
    MPSGraphTensorData *rws[4];
    MPSGraphTensorData *rbs[4];
    for (int i = 0; i < 4; i++) {
        dws[i] = sgdUpdate && fromLive
            ? graphData(w[i], wOff[i], wShapes[i])
            : (sgdUpdate ? graphDataMTL(gMlp3ScratchW[i], wShapes[i])
                         : graphData(w[i], wOff[i], wShapes[i]));
        dbs[i] = sgdUpdate && fromLive
            ? graphData(b[i], bOff[i], bShapes[i])
            : (sgdUpdate ? graphDataMTL(gMlp3ScratchB[i], bShapes[i])
                         : graphData(b[i], bOff[i], bShapes[i]));
        if (sgdUpdate) {
            rws[i] = fromLive ? graphDataMTL(gMlp3ScratchW[i], wShapes[i])
                              : graphData(w[i], wOff[i], wShapes[i]);
            rbs[i] = fromLive ? graphDataMTL(gMlp3ScratchB[i], bShapes[i])
                              : graphData(b[i], bOff[i], bShapes[i]);
        } else {
            rws[i] = graphData(gW[i], gWOff[i], wShapes[i]);
            rbs[i] = graphData(gB[i], gBOff[i], bShapes[i]);
        }
        if (dws[i] == nil || dbs[i] == nil || rws[i] == nil || rbs[i] == nil) return false;
    }
    if (dx == nil || dy == nil || dacc == nil) return false;
    if (trackCorrect && (dcorr == nil)) return false;
    NSMutableDictionary *feeds = [NSMutableDictionary dictionaryWithCapacity:12];
    feeds[cached[1]] = dx;
    for (int i = 0; i < 4; i++) {
        feeds[cached[2 + i * 2]] = dws[i];
        feeds[cached[3 + i * 2]] = dbs[i];
    }
    feeds[cached[10]] = dy;
    feeds[cached[11]] = dacc;
    if (trackCorrect) feeds[cached[12]] = dcorr;
    const int wBase = trackCorrect ? 13 : 12;
    NSMutableDictionary *results = [NSMutableDictionary dictionaryWithCapacity:10];
    for (int i = 0; i < 4; i++) {
        results[cached[wBase + i * 2]] = rws[i];
        results[cached[wBase + i * 2 + 1]] = rbs[i];
    }
    MPSGraphTensorData *racc = trackCorrect
        ? graphDataMTL(gMlpScratchAcc, @[ @1 ])
        : (accFromLive ? graphDataMTL(gMlpScratchAcc, @[ @1 ])
                       : graphData(lossAcc, lossOff, @[ @1 ]));
    MPSGraphTensorData *rcorr = trackCorrect
        ? graphDataMTL(gMlpScratchCorrect, @[ @1 ])
        : nil;
    if (racc == nil) return false;
    results[cached[wBase + 8]] = racc;
    if (trackCorrect) {
        if (rcorr == nil) return false;
        results[cached[wBase + 9]] = rcorr;
    }
    MPSGraphExecutable *exec = sgdUpdate
        ? cachedMlp3Executable(cached, trackCorrect, B, inFeatures, H1, H2, H3, C, lr)
        : cachedMlp3BwdExecutable(cached, trackCorrect, B, inFeatures, H1, H2, H3, C);
    if (exec == nil || !encodeMlpExecutableOnAsync(exec, feeds, results)) {
        if (!encodeGraphOnAsync(graph, feeds, results)) return false;
    }
    if (trackCorrect) {
        __unsafe_unretained id<MTLBuffer> accSrcs[] = {
            gMlpScratchAcc, gMlpScratchCorrect
        };
        JaiGpuBuffer *accDsts[] = {lossAcc, correctAcc};
        size_t accOffs[] = {lossOff, correctOff};
        size_t accSizes[] = {accBytes, accBytes};
        if (!blitMany(accSrcs, accDsts, accOffs, accSizes, 2)) return false;
    }
    gMlpLiveAcc = lossAcc;
    gMlpLiveAccOff = lossOff;
    if (!trackCorrect) {
        gMlpAccSide = accFromLive ? 1 : 0;
    }
    if (sgdUpdate) {
        for (int i = 0; i < 4; i++) {
            gMlp3LiveW[i] = w[i];
            gMlp3LiveB[i] = b[i];
            gMlp3LiveWOff[i] = wOff[i];
            gMlp3LiveBOff[i] = bOff[i];
            gMlp3LiveWBytes[i] = wBytes[i];
            gMlp3LiveBBytes[i] = bBytes[i];
        }
        gMlp3Side = fromLive ? 1 : 0;
    }
    return true;
}

bool jaiGpuMlp3SgdStep(JaiGpuBuffer *x, size_t xOff, JaiGpuBuffer *w1, size_t w1Off,
                       JaiGpuBuffer *b1, size_t b1Off, JaiGpuBuffer *w2, size_t w2Off,
                       JaiGpuBuffer *b2, size_t b2Off, JaiGpuBuffer *w3, size_t w3Off,
                       JaiGpuBuffer *b3, size_t b3Off, JaiGpuBuffer *w4, size_t w4Off,
                       JaiGpuBuffer *b4, size_t b4Off, JaiGpuBuffer *labels, size_t labOff,
                       JaiGpuBuffer *lossAcc, size_t lossOff, JaiGpuBuffer *correctAcc,
                       size_t correctOff, uint32_t B, uint32_t inFeatures, uint32_t H1,
                       uint32_t H2, uint32_t H3, uint32_t C, float lr) {
    JaiGpuBuffer *w[4] = {w1, w2, w3, w4};
    JaiGpuBuffer *b[4] = {b1, b2, b3, b4};
    size_t wOff[4] = {w1Off, w2Off, w3Off, w4Off};
    size_t bOff[4] = {b1Off, b2Off, b3Off, b4Off};
    if (x == NULL || w1 == NULL || b1 == NULL || w2 == NULL || b2 == NULL ||
        w3 == NULL || b3 == NULL || w4 == NULL || b4 == NULL ||
        labels == NULL || lossAcc == NULL) {
        return false;
    }
    const bool trackCorrect = correctAcc != NULL;
    if (B == 0 || inFeatures == 0 || H1 == 0 || H2 == 0 || H3 == 0 || C == 0) return false;
    if (!ensureDevice()) return false;

    @autoreleasepool {
        @synchronized(gQueue) {
            if (gMlp3Side != 0 && !mlp3LiveMatches(w, b, wOff, bOff)) {
                if (!commitMlp3WeightsLocked()) return false;
            }
            NSArray *cached = cachedMlp3Graph(B, inFeatures, H1, H2, H3, C, lr, trackCorrect);
            if (!encodeMlp3Step(
                    cached, trackCorrect, true, B, inFeatures, H1, H2, H3, C, lr,
                    x, xOff, w, wOff, b, bOff, labels, labOff, lossAcc, lossOff,
                    correctAcc, correctOff, NULL, NULL, NULL, NULL)) {
                return false;
            }
        }
        return true;
    }
}

bool jaiGpuMlp3BwdStep(JaiGpuBuffer *x, size_t xOff, JaiGpuBuffer *w1, size_t w1Off,
                       JaiGpuBuffer *b1, size_t b1Off, JaiGpuBuffer *w2, size_t w2Off,
                       JaiGpuBuffer *b2, size_t b2Off, JaiGpuBuffer *w3, size_t w3Off,
                       JaiGpuBuffer *b3, size_t b3Off, JaiGpuBuffer *w4, size_t w4Off,
                       JaiGpuBuffer *b4, size_t b4Off, JaiGpuBuffer *labels, size_t labOff,
                       JaiGpuBuffer *gW1, size_t gW1Off, JaiGpuBuffer *gB1, size_t gB1Off,
                       JaiGpuBuffer *gW2, size_t gW2Off, JaiGpuBuffer *gB2, size_t gB2Off,
                       JaiGpuBuffer *gW3, size_t gW3Off, JaiGpuBuffer *gB3, size_t gB3Off,
                       JaiGpuBuffer *gW4, size_t gW4Off, JaiGpuBuffer *gB4, size_t gB4Off,
                       JaiGpuBuffer *lossAcc, size_t lossOff, JaiGpuBuffer *correctAcc,
                       size_t correctOff, uint32_t B, uint32_t inFeatures, uint32_t H1,
                       uint32_t H2, uint32_t H3, uint32_t C) {
    JaiGpuBuffer *w[4] = {w1, w2, w3, w4};
    JaiGpuBuffer *b[4] = {b1, b2, b3, b4};
    JaiGpuBuffer *gW[4] = {gW1, gW2, gW3, gW4};
    JaiGpuBuffer *gB[4] = {gB1, gB2, gB3, gB4};
    size_t wOff[4] = {w1Off, w2Off, w3Off, w4Off};
    size_t bOff[4] = {b1Off, b2Off, b3Off, b4Off};
    size_t gWOff[4] = {gW1Off, gW2Off, gW3Off, gW4Off};
    size_t gBOff[4] = {gB1Off, gB2Off, gB3Off, gB4Off};
    if (x == NULL || w1 == NULL || b1 == NULL || w2 == NULL || b2 == NULL ||
        w3 == NULL || b3 == NULL || w4 == NULL || b4 == NULL ||
        labels == NULL || gW1 == NULL || gB1 == NULL || gW2 == NULL || gB2 == NULL ||
        gW3 == NULL || gB3 == NULL || gW4 == NULL || gB4 == NULL || lossAcc == NULL) {
        return false;
    }
    const bool trackCorrect = correctAcc != NULL;
    if (B == 0 || inFeatures == 0 || H1 == 0 || H2 == 0 || H3 == 0 || C == 0) return false;
    if (!ensureDevice()) return false;

    @autoreleasepool {
        @synchronized(gQueue) {
            NSArray *cached = cachedMlp3BwdGraph(B, inFeatures, H1, H2, H3, C, trackCorrect);
            if (!encodeMlp3Step(
                    cached, trackCorrect, false, B, inFeatures, H1, H2, H3, C, 0.0f,
                    x, xOff, w, wOff, b, bOff, labels, labOff, lossAcc, lossOff,
                    correctAcc, correctOff, gW, gWOff, gB, gBOff)) {
                return false;
            }
        }
        return true;
    }
}

bool jaiGpuMlpSgdStep(JaiGpuBuffer *x, size_t xOff, JaiGpuBuffer *w1, size_t w1Off,
                      JaiGpuBuffer *b1, size_t b1Off, JaiGpuBuffer *w2, size_t w2Off,
                      JaiGpuBuffer *b2, size_t b2Off, JaiGpuBuffer *labels, size_t labOff,
                      JaiGpuBuffer *lossAcc, size_t lossOff, JaiGpuBuffer *correctAcc,
                      size_t correctOff, uint32_t B, uint32_t inFeatures, uint32_t H,
                      uint32_t C, float lr) {
    if (x == NULL || w1 == NULL || b1 == NULL || w2 == NULL || b2 == NULL ||
        labels == NULL || lossAcc == NULL) {
        return false;
    }
    const bool trackCorrect = correctAcc != NULL;
    if (B == 0 || inFeatures == 0 || H == 0 || C == 0) return false;
    if (!ensureDevice()) return false;

    const size_t w1Bytes = (size_t)inFeatures * (size_t)H * sizeof(float);
    const size_t b1Bytes = (size_t)H * sizeof(float);
    const size_t w2Bytes = (size_t)H * (size_t)C * sizeof(float);
    const size_t b2Bytes = (size_t)C * sizeof(float);
    const size_t accBytes = sizeof(float);

    @autoreleasepool {
        @synchronized(gQueue) {
            if (gMlpSide != 0 && (w1 != gMlpLiveW1 || b1 != gMlpLiveB1 ||
                                  w2 != gMlpLiveW2 || b2 != gMlpLiveB2 ||
                                  w1Off != gMlpLiveW1Off || b1Off != gMlpLiveB1Off ||
                                  w2Off != gMlpLiveW2Off || b2Off != gMlpLiveB2Off)) {
                __unsafe_unretained id<MTLBuffer> commitSrcs[] = {
                    gMlpScratchW1, gMlpScratchB1, gMlpScratchW2, gMlpScratchB2
                };
                JaiGpuBuffer *commitDsts[] = {gMlpLiveW1, gMlpLiveB1, gMlpLiveW2, gMlpLiveB2};
                size_t commitOffs[] = {
                    gMlpLiveW1Off, gMlpLiveB1Off, gMlpLiveW2Off, gMlpLiveB2Off
                };
                size_t commitSizes[] = {
                    gMlpLiveW1Bytes, gMlpLiveB1Bytes, gMlpLiveW2Bytes, gMlpLiveB2Bytes
                };
                if (!blitMany(commitSrcs, commitDsts, commitOffs, commitSizes, 4)) {
                    return false;
                }
                gMlpSide = 0;
            }
            NSArray *cached = cachedMlpGraph(B, inFeatures, H, C, lr, trackCorrect);
            const NSUInteger expected = trackCorrect ? 15u : 13u;
            if (cached == nil || cached.count != expected) return false;
            MPSGraph *graph = cached[0];
            gMlpScratchW1 = growScratch(gMlpScratchW1, &gMlpCapW1, w1Bytes);
            gMlpScratchB1 = growScratch(gMlpScratchB1, &gMlpCapB1, b1Bytes);
            gMlpScratchW2 = growScratch(gMlpScratchW2, &gMlpCapW2, w2Bytes);
            gMlpScratchB2 = growScratch(gMlpScratchB2, &gMlpCapB2, b2Bytes);
            gMlpScratchAcc = growScratch(gMlpScratchAcc, &gMlpCapAcc, accBytes);
            if (trackCorrect) {
                gMlpScratchCorrect = growScratch(gMlpScratchCorrect, &gMlpCapCorrect, accBytes);
            }
            const bool fromLive = gMlpSide == 0;
            NSArray *w1Shape = @[ @(inFeatures), @(H) ];
            NSArray *b1Shape = @[ @(H) ];
            NSArray *w2Shape = @[ @(H), @(C) ];
            NSArray *b2Shape = @[ @(C) ];
            MPSGraphTensorData *dx = graphData(x, xOff, @[ @(B), @(inFeatures) ]);
            MPSGraphTensorData *dw1 = fromLive ? graphData(w1, w1Off, w1Shape)
                                               : graphDataMTL(gMlpScratchW1, w1Shape);
            MPSGraphTensorData *db1 = fromLive ? graphData(b1, b1Off, b1Shape)
                                               : graphDataMTL(gMlpScratchB1, b1Shape);
            MPSGraphTensorData *dw2 = fromLive ? graphData(w2, w2Off, w2Shape)
                                               : graphDataMTL(gMlpScratchW2, w2Shape);
            MPSGraphTensorData *db2 = fromLive ? graphData(b2, b2Off, b2Shape)
                                               : graphDataMTL(gMlpScratchB2, b2Shape);
            MPSGraphTensorData *dy = graphData(labels, labOff, @[ @(B) ]);
            const bool accFromLive = !trackCorrect && gMlpAccSide == 0;
            MPSGraphTensorData *dacc = trackCorrect
                ? graphData(lossAcc, lossOff, @[ @1 ])
                : (accFromLive ? graphData(lossAcc, lossOff, @[ @1 ])
                               : graphDataMTL(gMlpScratchAcc, @[ @1 ]));
            MPSGraphTensorData *dcorr = trackCorrect
                ? graphData(correctAcc, correctOff, @[ @1 ])
                : nil;
            MPSGraphTensorData *rw1 = fromLive ? graphDataMTL(gMlpScratchW1, w1Shape)
                                               : graphData(w1, w1Off, w1Shape);
            MPSGraphTensorData *rb1 = fromLive ? graphDataMTL(gMlpScratchB1, b1Shape)
                                               : graphData(b1, b1Off, b1Shape);
            MPSGraphTensorData *rw2 = fromLive ? graphDataMTL(gMlpScratchW2, w2Shape)
                                               : graphData(w2, w2Off, w2Shape);
            MPSGraphTensorData *rb2 = fromLive ? graphDataMTL(gMlpScratchB2, b2Shape)
                                               : graphData(b2, b2Off, b2Shape);
            MPSGraphTensorData *racc = trackCorrect
                ? graphDataMTL(gMlpScratchAcc, @[ @1 ])
                : (accFromLive ? graphDataMTL(gMlpScratchAcc, @[ @1 ])
                               : graphData(lossAcc, lossOff, @[ @1 ]));
            MPSGraphTensorData *rcorr = trackCorrect
                ? graphDataMTL(gMlpScratchCorrect, @[ @1 ])
                : nil;
            if (dx == nil || dw1 == nil || db1 == nil || dw2 == nil || db2 == nil ||
                dy == nil || dacc == nil || rw1 == nil || rb1 == nil ||
                rw2 == nil || rb2 == nil || racc == nil) {
                return false;
            }
            if (trackCorrect && (dcorr == nil || rcorr == nil)) return false;
            NSMutableDictionary *feeds = [@{
                cached[1] : dx,  cached[2] : dw1, cached[3] : db1,
                cached[4] : dw2, cached[5] : db2, cached[6] : dy,
                cached[7] : dacc
            } mutableCopy];
            if (trackCorrect) feeds[cached[8]] = dcorr;
            const int wBase = trackCorrect ? 9 : 8;
            NSMutableDictionary *results = [@{
                cached[wBase + 0] : rw1, cached[wBase + 1] : rb1,
                cached[wBase + 2] : rw2, cached[wBase + 3] : rb2,
                cached[wBase + 4] : racc
            } mutableCopy];
            if (trackCorrect) results[cached[wBase + 5]] = rcorr;
            MPSGraphExecutable *exec =
                cachedMlpExecutable(cached, trackCorrect, B, inFeatures, H, C, lr);
            if (exec == nil ||
                !encodeMlpExecutableOnAsync(exec, feeds, results)) {
                if (!encodeGraphOnAsync(graph, feeds, results)) return false;
            }
            if (trackCorrect) {
                __unsafe_unretained id<MTLBuffer> accSrcs[] = {
                    gMlpScratchAcc, gMlpScratchCorrect
                };
                JaiGpuBuffer *accDsts[] = {lossAcc, correctAcc};
                size_t accOffs[] = {lossOff, correctOff};
                size_t accSizes[] = {accBytes, accBytes};
                if (!blitMany(accSrcs, accDsts, accOffs, accSizes, 2)) return false;
            }
            gMlpLiveAcc = lossAcc;
            gMlpLiveAccOff = lossOff;
            if (!trackCorrect) {
                gMlpAccSide = accFromLive ? 1 : 0;
            }
            gMlpLiveW1 = w1;
            gMlpLiveB1 = b1;
            gMlpLiveW2 = w2;
            gMlpLiveB2 = b2;
            gMlpLiveW1Off = w1Off;
            gMlpLiveB1Off = b1Off;
            gMlpLiveW2Off = w2Off;
            gMlpLiveB2Off = b2Off;
            gMlpLiveW1Bytes = w1Bytes;
            gMlpLiveB1Bytes = b1Bytes;
            gMlpLiveW2Bytes = w2Bytes;
            gMlpLiveB2Bytes = b2Bytes;
            gMlpSide = fromLive ? 1 : 0;
        }
        return true;
    }
}

bool jaiGpuMlpSgdEpoch(JaiGpuBuffer *x, size_t xOff, JaiGpuBuffer *w1, size_t w1Off,
                       JaiGpuBuffer *b1, size_t b1Off, JaiGpuBuffer *w2, size_t w2Off,
                       JaiGpuBuffer *b2, size_t b2Off, JaiGpuBuffer *labels, size_t labOff,
                       JaiGpuBuffer *lossAcc, size_t lossOff, JaiGpuBuffer *correctAcc,
                       size_t correctOff, uint32_t samples, uint32_t batch,
                       uint32_t inputs, uint32_t hidden, uint32_t classes, float lr,
                       uint32_t flushEvery, uint32_t *processed) {
    if (processed != NULL) *processed = 0;
    if (batch == 0 || samples < batch) return true;
    if (flushEvery == 0) flushEvery = 1;
    const uint32_t steps = samples / batch;
    const size_t xStride = (size_t)batch * (size_t)inputs * sizeof(float);
    const size_t labStride = (size_t)batch * sizeof(float);
    const size_t xBytes = xStride;
    const size_t labBytes = labStride;
    if (x == NULL || labels == NULL || w1 == NULL || b1 == NULL || w2 == NULL ||
        b2 == NULL || lossAcc == NULL) {
        return false;
    }
    if (xOff + (size_t)steps * xStride > x->bytes) return false;
    if (labOff + (size_t)steps * labStride > labels->bytes) return false;
    const bool trackCorrect = correctAcc != NULL;
    if (trackCorrect && (correctAcc->buffer == NULL)) return false;

    const size_t w1Bytes = (size_t)inputs * (size_t)hidden * sizeof(float);
    const size_t b1Bytes = (size_t)hidden * sizeof(float);
    const size_t w2Bytes = (size_t)hidden * (size_t)classes * sizeof(float);
    const size_t b2Bytes = (size_t)classes * sizeof(float);
    const size_t accBytes = sizeof(float);

    @autoreleasepool {
        NSArray *cached = nil;
        MPSGraphExecutable *exec = nil;
        MPSGraph *graph = nil;
        NSArray *w1Shape = nil;
        NSArray *b1Shape = nil;
        NSArray *w2Shape = nil;
        NSArray *b2Shape = nil;
        NSArray *xShape = nil;
        NSArray *yShape = nil;
        NSArray *accShape = nil;
        MPSNDArrayDescriptor *xDesc = nil;
        MPSNDArrayDescriptor *yDesc = nil;
        MPSGraphTensorData *liveW1 = nil, *scratchW1 = nil;
        MPSGraphTensorData *liveB1 = nil, *scratchB1 = nil;
        MPSGraphTensorData *liveW2 = nil, *scratchW2 = nil;
        MPSGraphTensorData *liveB2 = nil, *scratchB2 = nil;
        MPSGraphTensorData *liveAcc = nil, *scratchAcc = nil;
        MPSGraphTensorData *liveCorr = nil, *scratchCorr = nil;
        NSMutableDictionary *feedsA = nil, *feedsB = nil;
        NSMutableDictionary *resultsA = nil, *resultsB = nil;
        NSMutableArray<MPSGraphTensorData *> *inputsA = nil, *inputsB = nil;
        NSArray<MPSGraphTensorData *> *execResultsA = nil, *execResultsB = nil;
        const int wBase = trackCorrect ? 9 : 8;

        @synchronized(gQueue) {
            if (gMlpSide != 0 && (w1 != gMlpLiveW1 || b1 != gMlpLiveB1 ||
                                  w2 != gMlpLiveW2 || b2 != gMlpLiveB2 ||
                                  w1Off != gMlpLiveW1Off || b1Off != gMlpLiveB1Off ||
                                  w2Off != gMlpLiveW2Off || b2Off != gMlpLiveB2Off)) {
                __unsafe_unretained id<MTLBuffer> commitSrcs[] = {
                    gMlpScratchW1, gMlpScratchB1, gMlpScratchW2, gMlpScratchB2
                };
                JaiGpuBuffer *commitDsts[] = {gMlpLiveW1, gMlpLiveB1, gMlpLiveW2, gMlpLiveB2};
                size_t commitOffs[] = {
                    gMlpLiveW1Off, gMlpLiveB1Off, gMlpLiveW2Off, gMlpLiveB2Off
                };
                size_t commitSizes[] = {
                    gMlpLiveW1Bytes, gMlpLiveB1Bytes, gMlpLiveW2Bytes, gMlpLiveB2Bytes
                };
                if (!blitMany(commitSrcs, commitDsts, commitOffs, commitSizes, 4)) {
                    return false;
                }
                gMlpSide = 0;
            }
            cached = cachedMlpGraph(batch, inputs, hidden, classes, lr, trackCorrect);
            const NSUInteger expected = trackCorrect ? 15u : 13u;
            if (cached == nil || cached.count != expected) return false;
            graph = cached[0];
            exec = cachedMlpExecutable(cached, trackCorrect, batch, inputs, hidden, classes, lr);
            gMlpScratchW1 = growScratch(gMlpScratchW1, &gMlpCapW1, w1Bytes);
            gMlpScratchB1 = growScratch(gMlpScratchB1, &gMlpCapB1, b1Bytes);
            gMlpScratchW2 = growScratch(gMlpScratchW2, &gMlpCapW2, w2Bytes);
            gMlpScratchB2 = growScratch(gMlpScratchB2, &gMlpCapB2, b2Bytes);
            gMlpScratchAcc = growScratch(gMlpScratchAcc, &gMlpCapAcc, accBytes);
            if (trackCorrect) {
                gMlpScratchCorrect = growScratch(gMlpScratchCorrect, &gMlpCapCorrect, accBytes);
            }
            w1Shape = @[ @(inputs), @(hidden) ];
            b1Shape = @[ @(hidden) ];
            w2Shape = @[ @(hidden), @(classes) ];
            b2Shape = @[ @(classes) ];
            xShape = @[ @(batch), @(inputs) ];
            yShape = @[ @(batch) ];
            accShape = @[ @1 ];
            xDesc = [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32 shape:xShape];
            yDesc = [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32 shape:yShape];
            liveW1 = graphData(w1, w1Off, w1Shape);
            liveB1 = graphData(b1, b1Off, b1Shape);
            liveW2 = graphData(w2, w2Off, w2Shape);
            liveB2 = graphData(b2, b2Off, b2Shape);
            scratchW1 = graphDataMTL(gMlpScratchW1, w1Shape);
            scratchB1 = graphDataMTL(gMlpScratchB1, b1Shape);
            scratchW2 = graphDataMTL(gMlpScratchW2, w2Shape);
            scratchB2 = graphDataMTL(gMlpScratchB2, b2Shape);
            liveAcc = graphData(lossAcc, lossOff, accShape);
            scratchAcc = graphDataMTL(gMlpScratchAcc, accShape);
            if (trackCorrect) {
                liveCorr = graphData(correctAcc, correctOff, accShape);
                scratchCorr = graphDataMTL(gMlpScratchCorrect, accShape);
            }
            if (liveW1 == nil || liveB1 == nil || liveW2 == nil || liveB2 == nil ||
                scratchW1 == nil || scratchB1 == nil || scratchW2 == nil ||
                scratchB2 == nil || liveAcc == nil || scratchAcc == nil) {
                return false;
            }
            if (trackCorrect && (liveCorr == nil || scratchCorr == nil)) return false;

            feedsA = [NSMutableDictionary dictionaryWithCapacity:8];
            feedsB = [NSMutableDictionary dictionaryWithCapacity:8];
            resultsA = [NSMutableDictionary dictionaryWithCapacity:6];
            resultsB = [NSMutableDictionary dictionaryWithCapacity:6];
            feedsA[cached[2]] = liveW1;
            feedsA[cached[3]] = liveB1;
            feedsA[cached[4]] = liveW2;
            feedsA[cached[5]] = liveB2;
            feedsB[cached[2]] = scratchW1;
            feedsB[cached[3]] = scratchB1;
            feedsB[cached[4]] = scratchW2;
            feedsB[cached[5]] = scratchB2;
            resultsA[cached[wBase + 0]] = scratchW1;
            resultsA[cached[wBase + 1]] = scratchB1;
            resultsA[cached[wBase + 2]] = scratchW2;
            resultsA[cached[wBase + 3]] = scratchB2;
            resultsB[cached[wBase + 0]] = liveW1;
            resultsB[cached[wBase + 1]] = liveB1;
            resultsB[cached[wBase + 2]] = liveW2;
            resultsB[cached[wBase + 3]] = liveB2;
            if (trackCorrect) {
                feedsA[cached[7]] = liveAcc;
                feedsB[cached[7]] = liveAcc;
                feedsA[cached[8]] = liveCorr;
                feedsB[cached[8]] = liveCorr;
                resultsA[cached[wBase + 4]] = scratchAcc;
                resultsB[cached[wBase + 4]] = scratchAcc;
                resultsA[cached[wBase + 5]] = scratchCorr;
                resultsB[cached[wBase + 5]] = scratchCorr;
            } else {
                feedsA[cached[7]] = liveAcc;
                feedsB[cached[7]] = scratchAcc;
                resultsA[cached[wBase + 4]] = scratchAcc;
                resultsB[cached[wBase + 4]] = liveAcc;
            }
            gMlpLiveW1 = w1;
            gMlpLiveB1 = b1;
            gMlpLiveW2 = w2;
            gMlpLiveB2 = b2;
            gMlpLiveW1Off = w1Off;
            gMlpLiveB1Off = b1Off;
            gMlpLiveW2Off = w2Off;
            gMlpLiveB2Off = b2Off;
            gMlpLiveW1Bytes = w1Bytes;
            gMlpLiveB1Bytes = b1Bytes;
            gMlpLiveW2Bytes = w2Bytes;
            gMlpLiveB2Bytes = b2Bytes;
            gMlpLiveAcc = lossAcc;
            gMlpLiveAccOff = lossOff;
        }

        NSMutableArray<MPSGraphTensorData *> *batchX =
            [NSMutableArray arrayWithCapacity:steps];
        NSMutableArray<MPSGraphTensorData *> *batchY =
            [NSMutableArray arrayWithCapacity:steps];
        if (!prefetchBatchFeeds(x, xOff, xStride, xBytes, xDesc, xShape, labels, labOff,
                                labStride, labBytes, yDesc, yShape, steps, batchX, batchY)) {
            return false;
        }

        for (uint32_t i = 0; i < steps; i++) {
            @synchronized(gQueue) {
                const bool fromLive = gMlpSide == 0;
                NSMutableDictionary *feeds = fromLive ? feedsA : feedsB;
                NSMutableDictionary *results = fromLive ? resultsA : resultsB;
                if (!encodeEpochBatch(
                        exec, graph, feeds, results, cached[1], cached[6],
                        batchX[i], batchY[i],
                        fromLive ? &inputsA : &inputsB,
                        fromLive ? &execResultsA : &execResultsB)) {
                    return false;
                }
                if (trackCorrect) {
                    __unsafe_unretained id<MTLBuffer> accSrcs[] = {
                        gMlpScratchAcc, gMlpScratchCorrect
                    };
                    JaiGpuBuffer *accDsts[] = {lossAcc, correctAcc};
                    size_t accOffs[] = {lossOff, correctOff};
                    size_t accSizes[] = {accBytes, accBytes};
                    if (!blitMany(accSrcs, accDsts, accOffs, accSizes, 2)) return false;
                } else {
                    gMlpAccSide = fromLive ? 1 : 0;
                }
                gMlpSide = fromLive ? 1 : 0;
            }
            if ((i + 1) % flushEvery == 0 || i + 1 == steps) {
                id<MTLCommandBuffer> oldest = nil;
                uint64_t oldestBatch = 0;
                @synchronized(gQueue) {
                    if (!flushAsyncLocked(&oldest, &oldestBatch)) return false;
                }
                if (oldest != nil) {
                    [oldest waitUntilCompleted];
                    if ([oldest status] != MTLCommandBufferStatusCompleted) return false;
                    noteDone(oldestBatch);
                }
            }
        }
        @synchronized(gQueue) {
            /* Keep weight and accumulator ping-pong state aligned across epoch
             * calls.  Otherwise an odd number of batches leaves weights in the
             * scratch set while commitMlpAccLocked resets only the accumulator
             * to live, and the next epoch reads the previous scratch loss. */
            if (!trackCorrect && !commitMlpWeightsLocked()) return false;
            if (!commitMlpAccLocked()) return false;
            /* Committed and left on the books: a batch that leaves gInFlight
             * without being waited for is one no buffer can ever wait for. */
            if (!flushAsyncLocked(NULL, NULL)) return false;
        }
    }
    if (processed != NULL) *processed = steps * batch;
    return true;
}

bool jaiGpuMlp3SgdEpoch(JaiGpuBuffer *x, size_t xOff, JaiGpuBuffer *w1, size_t w1Off,
                        JaiGpuBuffer *b1, size_t b1Off, JaiGpuBuffer *w2, size_t w2Off,
                        JaiGpuBuffer *b2, size_t b2Off, JaiGpuBuffer *w3, size_t w3Off,
                        JaiGpuBuffer *b3, size_t b3Off, JaiGpuBuffer *w4, size_t w4Off,
                        JaiGpuBuffer *b4, size_t b4Off, JaiGpuBuffer *labels, size_t labOff,
                        JaiGpuBuffer *lossAcc, size_t lossOff, JaiGpuBuffer *correctAcc,
                        size_t correctOff, uint32_t samples, uint32_t batch,
                        uint32_t inputs, uint32_t hidden1, uint32_t hidden2,
                        uint32_t hidden3, uint32_t classes, float lr,
                        uint32_t flushEvery, uint32_t *processed) {
    if (processed != NULL) *processed = 0;
    if (batch == 0 || samples < batch) return true;
    if (flushEvery == 0) flushEvery = 1;
    const uint32_t steps = samples / batch;
    const size_t xStride = (size_t)batch * (size_t)inputs * sizeof(float);
    const size_t labStride = (size_t)batch * sizeof(float);
    const size_t xBytes = xStride;
    const size_t labBytes = labStride;
    if (x == NULL || labels == NULL || w1 == NULL || b1 == NULL || w2 == NULL ||
        b2 == NULL || w3 == NULL || b3 == NULL || w4 == NULL || b4 == NULL ||
        lossAcc == NULL) {
        return false;
    }
    if (xOff + (size_t)steps * xStride > x->bytes) return false;
    if (labOff + (size_t)steps * labStride > labels->bytes) return false;
    const bool trackCorrect = correctAcc != NULL;
    JaiGpuBuffer *w[4] = {w1, w2, w3, w4};
    JaiGpuBuffer *b[4] = {b1, b2, b3, b4};
    size_t wOff[4] = {w1Off, w2Off, w3Off, w4Off};
    size_t bOff[4] = {b1Off, b2Off, b3Off, b4Off};
    size_t wBytes[4];
    size_t bBytes[4];
    mlp3WeightBytes(inputs, hidden1, hidden2, hidden3, classes, wBytes, bBytes);
    const size_t accBytes = sizeof(float);
    const int wBase = trackCorrect ? 13 : 12;

    @autoreleasepool {
        NSArray *cached = nil;
        MPSGraphExecutable *exec = nil;
        MPSGraph *graph = nil;
        NSArray *xShape = nil;
        NSArray *yShape = nil;
        NSArray *accShape = nil;
        MPSNDArrayDescriptor *xDesc = nil;
        MPSNDArrayDescriptor *yDesc = nil;
        MPSGraphTensorData *liveW[4] = {nil, nil, nil, nil};
        MPSGraphTensorData *scratchW[4] = {nil, nil, nil, nil};
        MPSGraphTensorData *liveB[4] = {nil, nil, nil, nil};
        MPSGraphTensorData *scratchB[4] = {nil, nil, nil, nil};
        MPSGraphTensorData *liveAcc = nil, *scratchAcc = nil;
        MPSGraphTensorData *liveCorr = nil, *scratchCorr = nil;
        NSMutableDictionary *feedsA = nil, *feedsB = nil;
        NSMutableDictionary *resultsA = nil, *resultsB = nil;
        NSMutableArray<MPSGraphTensorData *> *inputsA = nil, *inputsB = nil;
        NSArray<MPSGraphTensorData *> *execResultsA = nil, *execResultsB = nil;

        @synchronized(gQueue) {
            if (gMlp3Side != 0 && !mlp3LiveMatches(w, b, wOff, bOff)) {
                if (!commitMlp3WeightsLocked()) return false;
            }
            cached = cachedMlp3Graph(batch, inputs, hidden1, hidden2, hidden3, classes, lr,
                                     trackCorrect);
            const NSUInteger expected = trackCorrect ? 23u : 21u;
            if (cached == nil || cached.count != expected) return false;
            graph = cached[0];
            exec = cachedMlp3Executable(cached, trackCorrect, batch, inputs, hidden1, hidden2,
                                        hidden3, classes, lr);
            gMlpScratchAcc = growScratch(gMlpScratchAcc, &gMlpCapAcc, accBytes);
            if (trackCorrect) {
                gMlpScratchCorrect = growScratch(gMlpScratchCorrect, &gMlpCapCorrect, accBytes);
            }
            NSArray *wShapes[4] = {
                @[ @(inputs), @(hidden1) ],
                @[ @(hidden1), @(hidden2) ],
                @[ @(hidden2), @(hidden3) ],
                @[ @(hidden3), @(classes) ]
            };
            NSArray *bShapes[4] = {
                @[ @(hidden1) ], @[ @(hidden2) ], @[ @(hidden3) ], @[ @(classes) ]
            };
            for (int i = 0; i < 4; i++) {
                gMlp3ScratchW[i] = growScratch(gMlp3ScratchW[i], &gMlp3CapW[i], wBytes[i]);
                gMlp3ScratchB[i] = growScratch(gMlp3ScratchB[i], &gMlp3CapB[i], bBytes[i]);
                liveW[i] = graphData(w[i], wOff[i], wShapes[i]);
                liveB[i] = graphData(b[i], bOff[i], bShapes[i]);
                scratchW[i] = graphDataMTL(gMlp3ScratchW[i], wShapes[i]);
                scratchB[i] = graphDataMTL(gMlp3ScratchB[i], bShapes[i]);
                if (liveW[i] == nil || liveB[i] == nil || scratchW[i] == nil ||
                    scratchB[i] == nil) {
                    return false;
                }
            }
            xShape = @[ @(batch), @(inputs) ];
            yShape = @[ @(batch) ];
            accShape = @[ @1 ];
            xDesc = [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32 shape:xShape];
            yDesc = [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32 shape:yShape];
            liveAcc = graphData(lossAcc, lossOff, accShape);
            scratchAcc = graphDataMTL(gMlpScratchAcc, accShape);
            if (liveAcc == nil || scratchAcc == nil) return false;
            if (trackCorrect) {
                liveCorr = graphData(correctAcc, correctOff, accShape);
                scratchCorr = graphDataMTL(gMlpScratchCorrect, accShape);
                if (liveCorr == nil || scratchCorr == nil) return false;
            }
            feedsA = [NSMutableDictionary dictionaryWithCapacity:12];
            feedsB = [NSMutableDictionary dictionaryWithCapacity:12];
            resultsA = [NSMutableDictionary dictionaryWithCapacity:10];
            resultsB = [NSMutableDictionary dictionaryWithCapacity:10];
            for (int i = 0; i < 4; i++) {
                feedsA[cached[2 + i * 2]] = liveW[i];
                feedsA[cached[3 + i * 2]] = liveB[i];
                feedsB[cached[2 + i * 2]] = scratchW[i];
                feedsB[cached[3 + i * 2]] = scratchB[i];
                resultsA[cached[wBase + i * 2]] = scratchW[i];
                resultsA[cached[wBase + i * 2 + 1]] = scratchB[i];
                resultsB[cached[wBase + i * 2]] = liveW[i];
                resultsB[cached[wBase + i * 2 + 1]] = liveB[i];
            }
            if (trackCorrect) {
                feedsA[cached[11]] = liveAcc;
                feedsB[cached[11]] = liveAcc;
                feedsA[cached[12]] = liveCorr;
                feedsB[cached[12]] = liveCorr;
                resultsA[cached[wBase + 8]] = scratchAcc;
                resultsB[cached[wBase + 8]] = scratchAcc;
                resultsA[cached[wBase + 9]] = scratchCorr;
                resultsB[cached[wBase + 9]] = scratchCorr;
            } else {
                feedsA[cached[11]] = liveAcc;
                feedsB[cached[11]] = scratchAcc;
                resultsA[cached[wBase + 8]] = scratchAcc;
                resultsB[cached[wBase + 8]] = liveAcc;
            }
            for (int i = 0; i < 4; i++) {
                gMlp3LiveW[i] = w[i];
                gMlp3LiveB[i] = b[i];
                gMlp3LiveWOff[i] = wOff[i];
                gMlp3LiveBOff[i] = bOff[i];
                gMlp3LiveWBytes[i] = wBytes[i];
                gMlp3LiveBBytes[i] = bBytes[i];
            }
            gMlpLiveAcc = lossAcc;
            gMlpLiveAccOff = lossOff;
        }

        NSMutableArray<MPSGraphTensorData *> *batchX =
            [NSMutableArray arrayWithCapacity:steps];
        NSMutableArray<MPSGraphTensorData *> *batchY =
            [NSMutableArray arrayWithCapacity:steps];
        if (!prefetchBatchFeeds(x, xOff, xStride, xBytes, xDesc, xShape, labels, labOff,
                                labStride, labBytes, yDesc, yShape, steps, batchX, batchY)) {
            return false;
        }

        for (uint32_t i = 0; i < steps; i++) {
            @synchronized(gQueue) {
                const bool fromLive = gMlp3Side == 0;
                NSMutableDictionary *feeds = fromLive ? feedsA : feedsB;
                NSMutableDictionary *results = fromLive ? resultsA : resultsB;
                if (!encodeEpochBatch(
                        exec, graph, feeds, results, cached[1], cached[10],
                        batchX[i], batchY[i],
                        fromLive ? &inputsA : &inputsB,
                        fromLive ? &execResultsA : &execResultsB)) {
                    return false;
                }
                if (trackCorrect) {
                    __unsafe_unretained id<MTLBuffer> accSrcs[] = {
                        gMlpScratchAcc, gMlpScratchCorrect
                    };
                    JaiGpuBuffer *accDsts[] = {lossAcc, correctAcc};
                    size_t accOffs[] = {lossOff, correctOff};
                    size_t accSizes[] = {accBytes, accBytes};
                    if (!blitMany(accSrcs, accDsts, accOffs, accSizes, 2)) return false;
                } else {
                    gMlpAccSide = fromLive ? 1 : 0;
                }
                gMlp3Side = fromLive ? 1 : 0;
            }
            if ((i + 1) % flushEvery == 0 || i + 1 == steps) {
                id<MTLCommandBuffer> oldest = nil;
                uint64_t oldestBatch = 0;
                @synchronized(gQueue) {
                    if (!flushAsyncLocked(&oldest, &oldestBatch)) return false;
                }
                if (oldest != nil) {
                    [oldest waitUntilCompleted];
                    if ([oldest status] != MTLCommandBufferStatusCompleted) return false;
                    noteDone(oldestBatch);
                }
            }
        }
        @synchronized(gQueue) {
            if (!trackCorrect && !commitMlp3WeightsLocked()) return false;
            if (!commitMlpAccLocked()) return false;
            /* Committed and left on the books: a batch that leaves gInFlight
             * without being waited for is one no buffer can ever wait for. */
            if (!flushAsyncLocked(NULL, NULL)) return false;
        }
    }
    if (processed != NULL) *processed = steps * batch;
    return true;
}

bool jaiGpuLabelsValid(JaiGpuBuffer *labels, size_t offset, uint32_t count,
                       uint32_t classes) {
    if (count == 0) return true;
    if (labels == NULL || labels->buffer == NULL) return false;
    if ((offset % sizeof(float)) != 0) return false;
    const size_t bytes = (size_t)count * sizeof(float);
    if (offset + bytes > labels->bytes) return false;
    if (!jaiGpuSynchronize()) return false;
    const float *values =
        (const float *)((__bridge id<MTLBuffer>)labels->buffer).contents +
        offset / sizeof(float);
    for (uint32_t i = 0; i < count; i++) {
        const float value = values[i];
        const int index = (int)value;
        if (value != (float)index || index < 0 || (uint32_t)index >= classes) {
            return false;
        }
    }
    return true;
}

/* Neumaier variant of Kahan summation — also captures the low bits dropped
 * when the running total is smaller than the term added. */
static double compensatedSum(const double *values, size_t n) {
    double total = 0.0;
    double correction = 0.0;
    for (size_t i = 0; i < n; i++) {
        double value = values[i];
        double next = total + value;
        if (fabs(total) >= fabs(value)) {
            correction += (total - next) + value;
        } else {
            correction += (value - next) + total;
        }
        total = next;
    }
    return total + correction;
}

static double compensatedSumF32(const float *values, size_t n) {
    double total = 0.0;
    double correction = 0.0;
    for (size_t i = 0; i < n; i++) {
        double value = (double)values[i];
        double next = total + value;
        if (fabs(total) >= fabs(value)) {
            correction += (total - next) + value;
        } else {
            correction += (value - next) + total;
        }
        total = next;
    }
    return total + correction;
}

/* Row/inner/column loop order keeps `b` and the output row walked forward;
 * same arithmetic as the textbook triple loop, different cache behavior. */
static void cpuMatMul(const double *a, const double *b, double *out,
                      size_t m, size_t k, size_t n) {
    memset(out, 0, m * n * sizeof(double));
    for (size_t row = 0; row < m; row++) {
        double *outRow = out + row * n;
        const double *aRow = a + row * k;
        for (size_t i = 0; i < k; i++) {
            double factor = aRow[i];
            if (factor == 0.0) continue;
            const double *bRow = b + i * n;
            for (size_t column = 0; column < n; column++) {
                outRow[column] += factor * bRow[column];
            }
        }
    }
}

/* The built-ins stage directly into Metal shared buffers so Apple silicon does
 * not allocate a second CPU float array and then copy it into unified memory. */
static inline bool fitsDeviceBuffer(size_t elements) {
    if (elements > SIZE_MAX / sizeof(float)) return false;
    return elements * sizeof(float) <= gMaxBufferLength;
}

static id<MTLBuffer> newInputBuffer(const double *src, size_t n) {
    if (src == NULL || n == 0 || !fitsDeviceBuffer(n))
        return nil;

    const size_t bytes = n * sizeof(float);

    /* Write-combined caching is ideal here: the CPU streams values in once and
     * never reads the input buffer back. */
    const MTLResourceOptions options =
        MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined;

    id<MTLBuffer> buffer =
        [gDevice newBufferWithLength:bytes options:options];

    if (buffer == nil)
        return nil;

    float *dst = (float *)[buffer contents];

#if defined(__clang__)
#  pragma clang loop vectorize(enable) interleave(enable)
#endif
    for (size_t i = 0; i < n; ++i)
        dst[i] = (float)src[i];

    return buffer;
}

static inline id<MTLBuffer> newOutputBuffer(size_t n) {
    if (n == 0 || !fitsDeviceBuffer(n))
        return nil;

    return [gDevice newBufferWithLength:n * sizeof(float)
                                options:MTLResourceStorageModeShared];
}

static void widenFloats(double *dst, const float *src, size_t n) {
#if defined(__clang__)
#  pragma clang loop vectorize(enable) interleave(enable)
#endif
    for (size_t i = 0; i < n; ++i)
        dst[i] = (double)src[i];
}

/* Both elementwise built-ins differ only in pipeline, so they share this
 * encoder. Returns false (nothing written) if any Metal call fails; the caller then falls back to the scalar path. */
static bool deviceElementwise(id<MTLComputePipelineState> pipeline,
                              const double *a, const double *b,
                              double *out, size_t n) {
    if (!fitsDeviceBuffer(n) || n > UINT32_MAX)
        return false;

    @autoreleasepool {
        id<MTLBuffer> aBuf = newInputBuffer(a, n);
        id<MTLBuffer> bBuf = newInputBuffer(b, n);
        id<MTLBuffer> outBuf = newOutputBuffer(n);

        if (aBuf == nil || bBuf == nil || outBuf == nil)
            return false;

        id<MTLCommandBuffer> commands =
            [gQueue commandBufferWithUnretainedReferences];
        if (commands == nil)
            return false;

        id<MTLComputeCommandEncoder> encoder =
            [commands computeCommandEncoder];
        if (encoder == nil)
            return false;

        const uint32_t count = (uint32_t)n;

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:aBuf offset:0 atIndex:0];
        [encoder setBuffer:bBuf offset:0 atIndex:1];
        [encoder setBuffer:outBuf offset:0 atIndex:2];
        [encoder setBytes:&count length:sizeof count atIndex:3];

        const NSUInteger workItems =
            ((NSUInteger)n + JAI_VECTOR_LANES - 1u) / JAI_VECTOR_LANES;

        encodeDispatch(encoder, pipeline, workItems, 0);
        [encoder endEncoding];

        [commands commit];
        [commands waitUntilCompleted];

        if ([commands status] != MTLCommandBufferStatusCompleted)
            return false;

        widenFloats(out, (const float *)[outBuf contents], n);
        return true;
    }
}

bool jaiGpuVectorAdd(const double *a, const double *b, double *out, size_t n) {
    if (a == NULL || b == NULL || out == NULL) return false;
    if (n == 0) return true;

    if (n >= JAI_GPU_MIN_WORK && ensureBuiltins() &&
        deviceElementwise(gVectorAdd, a, b, out, n)) {
        return true;
    }
    for (size_t i = 0; i < n; i++) out[i] = a[i] + b[i];
    return true;
}

bool jaiGpuVectorMul(const double *a, const double *b, double *out, size_t n) {
    if (a == NULL || b == NULL || out == NULL) return false;
    if (n == 0) return true;

    if (n >= JAI_GPU_MIN_WORK && ensureBuiltins() &&
        deviceElementwise(gVectorMul, a, b, out, n)) {
        return true;
    }
    for (size_t i = 0; i < n; i++) out[i] = a[i] * b[i];
    return true;
}

static inline void encodeMatMulDispatch(
    id<MTLComputeCommandEncoder> encoder, size_t rows, size_t columns) {
    const MTLSize threadsPerGroup =
        MTLSizeMake(JAI_MATMUL_TILE, JAI_MATMUL_TILE, 1);

    /* Always dispatches complete 16x16 groups: the tiled shader needs every
     * lane present to populate its threadgroup tiles, so partial edge groups would leave scratch entries unloaded. */
    const NSUInteger groupsX =
        ((NSUInteger)columns + JAI_MATMUL_TILE - 1u) / JAI_MATMUL_TILE;
    const NSUInteger groupsY =
        ((NSUInteger)rows + JAI_MATMUL_TILE - 1u) / JAI_MATMUL_TILE;

    [encoder dispatchThreadgroups:MTLSizeMake(groupsX, groupsY, 1)
            threadsPerThreadgroup:threadsPerGroup];
}

static bool deviceMatMul(const double *a, const double *b, double *out,
                         size_t m, size_t k, size_t n) {
    const size_t aCount = m * k;
    const size_t bCount = k * n;
    const size_t outCount = m * n;

    if (!fitsDeviceBuffer(aCount) ||
        !fitsDeviceBuffer(bCount) ||
        !fitsDeviceBuffer(outCount)) {
        return false;
    }

    if (m > UINT32_MAX || k > UINT32_MAX || n > UINT32_MAX)
        return false;

    if ([gMatMul maxTotalThreadsPerThreadgroup] <
        JAI_MATMUL_TILE * JAI_MATMUL_TILE) {
        return false;
    }

    @autoreleasepool {
        id<MTLBuffer> aBuf = newInputBuffer(a, aCount);
        id<MTLBuffer> bBuf = newInputBuffer(b, bCount);
        id<MTLBuffer> outBuf = newOutputBuffer(outCount);

        if (aBuf == nil || bBuf == nil || outBuf == nil)
            return false;

        id<MTLCommandBuffer> commands =
            [gQueue commandBufferWithUnretainedReferences];
        if (commands == nil)
            return false;

        id<MTLComputeCommandEncoder> encoder =
            [commands computeCommandEncoder];
        if (encoder == nil)
            return false;

        const uint32_t rows = (uint32_t)m;
        const uint32_t inner = (uint32_t)k;
        const uint32_t columns = (uint32_t)n;

        [encoder setComputePipelineState:gMatMul];
        [encoder setBuffer:aBuf offset:0 atIndex:0];
        [encoder setBuffer:bBuf offset:0 atIndex:1];
        [encoder setBuffer:outBuf offset:0 atIndex:2];

        /* Apple explicitly recommends setBytes for tiny transient arguments. */
        [encoder setBytes:&rows length:sizeof rows atIndex:3];
        [encoder setBytes:&inner length:sizeof inner atIndex:4];
        [encoder setBytes:&columns length:sizeof columns atIndex:5];

        encodeMatMulDispatch(encoder, m, n);
        [encoder endEncoding];

        [commands commit];
        [commands waitUntilCompleted];

        if ([commands status] != MTLCommandBufferStatusCompleted)
            return false;

        widenFloats(out, (const float *)[outBuf contents], outCount);
        return true;
    }
}

bool jaiGpuMatMul(const double *a, const double *b, double *out,
                  size_t m, size_t k, size_t n) {
    if (a == NULL || b == NULL || out == NULL) return false;
    if (m == 0 || n == 0) return true;
    /* Guard the index arithmetic below, not the device: m*k, k*n and m*n all
     * have to be representable before either path can touch them. */
    if (k != 0 && (m > SIZE_MAX / k || n > SIZE_MAX / k)) return false;
    if (m > SIZE_MAX / n) return false;

    if (k == 0) {
        memset(out, 0, m * n * sizeof(double));
        return true;
    }

    /* Work counts multiply-adds, not elements: a matmul does m*k*n of them and
     * it is the arithmetic, not the output size, that has to beat the upload. */
    size_t work = (m * n <= SIZE_MAX / k) ? m * n * k : SIZE_MAX;
    if (work >= JAI_GPU_MIN_WORK && ensureBuiltins() &&
        deviceMatMul(a, b, out, m, k, n)) {
        return true;
    }
    cpuMatMul(a, b, out, m, k, n);
    return true;
}

static bool deviceReduceSum(const double *a, size_t n, double *out) {
    if (!fitsDeviceBuffer(n) || n > UINT32_MAX)
        return false;

    const size_t valuesPerGroup =
        (size_t)JAI_REDUCE_GROUP * JAI_REDUCE_LOADS;
    const size_t groups =
        (n + valuesPerGroup - 1u) / valuesPerGroup;

    if (!fitsDeviceBuffer(groups))
        return false;

    if ([gReduceSum maxTotalThreadsPerThreadgroup] < JAI_REDUCE_GROUP)
        return false;

    @autoreleasepool {
        id<MTLBuffer> inBuf = newInputBuffer(a, n);
        id<MTLBuffer> partialBuf = newOutputBuffer(groups);

        if (inBuf == nil || partialBuf == nil)
            return false;

        id<MTLCommandBuffer> commands =
            [gQueue commandBufferWithUnretainedReferences];
        if (commands == nil)
            return false;

        id<MTLComputeCommandEncoder> encoder =
            [commands computeCommandEncoder];
        if (encoder == nil)
            return false;

        const uint32_t count = (uint32_t)n;

        [encoder setComputePipelineState:gReduceSum];
        [encoder setBuffer:inBuf offset:0 atIndex:0];
        [encoder setBuffer:partialBuf offset:0 atIndex:1];
        [encoder setBytes:&count length:sizeof count atIndex:2];

        [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(JAI_REDUCE_GROUP, 1, 1)];
        [encoder endEncoding];

        [commands commit];
        [commands waitUntilCompleted];

        if ([commands status] != MTLCommandBufferStatusCompleted)
            return false;

        *out = compensatedSumF32(
            (const float *)[partialBuf contents], groups);
        return true;
    }
}

bool jaiGpuReduceSum(const double *a, size_t n, double *out) {
    if (a == NULL || out == NULL) return false;
    if (n == 0) {
        *out = 0.0;
        return true;
    }

    if (n >= JAI_GPU_MIN_WORK && ensureBuiltins() && deviceReduceSum(a, n, out)) {
        return true;
    }
    *out = compensatedSum(a, n);
    return true;
}

#endif /* __APPLE__ */
