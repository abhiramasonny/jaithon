/* gpu_device.m — device, queue and library setup, the MSL shader text and
 * the pipelines built from it, and the process-wide occupancy meter. */

#ifdef __APPLE__

#include "native/apple/gpu_internal.h"

static _Atomic long gDispatchTraceKernel = 0;
static _Atomic long gDispatchTraceGraph = 0;
static int gDispatchTraceOn = -1;
static void dispatchTraceDump(void) {
    fprintf(stderr, "JAI_GPU_DISPATCH_TRACE kernel=%ld graph=%ld total=%ld\n",
            gDispatchTraceKernel, gDispatchTraceGraph, gDispatchTraceKernel + gDispatchTraceGraph);
}
void dispatchTraceTick(int isGraph) {
    if (gDispatchTraceOn < 0) {
        gDispatchTraceOn = getenv("JAI_GPU_DISPATCH_TRACE") != NULL ? 1 : 0;
        if (gDispatchTraceOn) atexit(dispatchTraceDump);
    }
    if (!gDispatchTraceOn) return;
    if (isGraph) gDispatchTraceGraph++; else gDispatchTraceKernel++;
}

/* ------------------------------------------------------------------ */
/* Occupancy meter                                                      */
/* ------------------------------------------------------------------ */

/* How much of the wall clock the GPU was actually working, per process.
 *
 * The question this exists to answer is not "which device should run this" --
 * that one is largely settled here, and settled correctly. It is "how much of
 * the time is either side idle waiting for the other", which nothing in this
 * tree could report. The knob that would fix an idle GPU
 * (JAITHON_GPU_AUTO_COMMIT, see autoCommitThreshold) is off by default
 * precisely because one global threshold is +5% on jaicv and -26% on
 * jaitensor, and there was no way to see which side of that a given workload
 * sits on.
 *
 * Off unless JAITHON_GPU_METER is set, and the increments themselves are
 * gated, not merely the printing, so an ordinary run pays one predictable
 * not-taken branch per command buffer.
 *
 * THE SUM IS NOT THE ANSWER. JAI_GPU_MAX_IN_FLIGHT is 16 and flushAsyncLocked
 * keeps that many buffers running at once, so adding their durations counts
 * the same wall nanosecond up to sixteen times and reports occupancy above
 * 1.0. Spans are recorded raw and merged at report time. */
#define JAI_METER_MAX 262144

typedef struct {
    double   start;      /* MTLCommandBuffer.GPUStartTime, seconds */
    double   end;
    uint32_t dispatches;
} MeterSpan;

static MeterSpan        gMeterSpans[JAI_METER_MAX];
static _Atomic uint32_t gMeterSpanCount;
static _Atomic uint64_t gMeterHostWaitNs;
static _Atomic uint32_t gMeterHostWaits;
static _Atomic uint64_t gMeterHostWaitMaxNs;
static _Atomic uint32_t gMeterDropped;
static _Atomic uint32_t gMeterMpsSwaps;
static double           gMeterWallStart;
static int              gMeterOn = -1;

double meterNow(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void meterReport(void);

int meterOn(void) {
    if (gMeterOn < 0) {
        gMeterOn = getenv("JAITHON_GPU_METER") != NULL ? 1 : 0;
        if (gMeterOn) {
            gMeterWallStart = meterNow();
            atexit(meterReport);
        }
    }
    return gMeterOn;
}

/* One finished command buffer. Called from Metal's completion thread for the
 * async path and inline for the four synchronous ones, so the slot claim is
 * atomic and the write is to a slot nobody else owns. */
void meterNote(id<MTLCommandBuffer> done, uint32_t dispatches) {
    if (!meterOn() || done == nil) return;
    const uint32_t slot = atomic_fetch_add(&gMeterSpanCount, 1u);
    if (slot >= JAI_METER_MAX) { atomic_fetch_add(&gMeterDropped, 1u); return; }
    gMeterSpans[slot].start      = done.GPUStartTime;
    gMeterSpans[slot].end        = done.GPUEndTime;
    gMeterSpans[slot].dispatches = dispatches;
}

/* MPSGraph is handed the open command buffer and may commit it and hand back a
 * different root (see the two MPSCommandBuffer sites). Anything it committed
 * that way never carried our completion handler, so its GPU time is not in the
 * span set. We cannot see that work; we can at least count the swaps and refuse
 * to report an occupancy that pretends otherwise. */
void meterNoteMpsSwap(void) {
    if (!meterOn()) return;
    atomic_fetch_add(&gMeterMpsSwaps, 1u);
}

void meterHostWait(uint64_t ns) {
    if (!meterOn()) return;
    atomic_fetch_add(&gMeterHostWaitNs, ns);
    atomic_fetch_add(&gMeterHostWaits, 1u);
    uint64_t seen = atomic_load(&gMeterHostWaitMaxNs);
    while (ns > seen &&
           !atomic_compare_exchange_weak(&gMeterHostWaitMaxNs, &seen, ns)) { }
}

static int meterSpanCmp(const void *a, const void *b) {
    const double x = ((const MeterSpan *)a)->start;
    const double y = ((const MeterSpan *)b)->start;
    return x < y ? -1 : (x > y ? 1 : 0);
}

static void meterReport(void) {
    uint32_t n = atomic_load(&gMeterSpanCount);
    if (n > JAI_METER_MAX) n = JAI_METER_MAX;
    const double wall = meterNow() - gMeterWallStart;

    /* Sort by start and merge overlaps -- see the note above on why a sum
     * would be wrong. Also reports the sum, because sum/union is exactly the
     * average depth of the pipeline and that is the number that says whether
     * more in-flight buffers would help. */
    qsort(gMeterSpans, n, sizeof gMeterSpans[0], meterSpanCmp);
    double busy = 0.0, raw = 0.0, curS = 0.0, curE = 0.0;
    uint64_t dispatches = 0;
    bool open = false;
    for (uint32_t i = 0; i < n; i++) {
        const double s = gMeterSpans[i].start, e = gMeterSpans[i].end;
        dispatches += gMeterSpans[i].dispatches;
        if (e <= s) continue;
        raw += e - s;
        if (!open) { curS = s; curE = e; open = true; continue; }
        if (s <= curE) { if (e > curE) curE = e; continue; }
        busy += curE - curS;
        curS = s; curE = e;
    }
    if (open) busy += curE - curS;

    const double waitS = (double)atomic_load(&gMeterHostWaitNs) * 1e-9;
    fprintf(stderr,
            "\n[gpu meter] wall %.1f ms | gpu busy %.1f ms | occupancy %.3f\n"
            "             buffers %u  dispatches %llu  pipeline depth %.2f\n"
            "             host waits %u  waiting %.1f ms  longest %.1f ms\n",
            wall * 1e3, busy * 1e3, wall > 0.0 ? busy / wall : 0.0,
            n, (unsigned long long)dispatches, busy > 0.0 ? raw / busy : 0.0,
            atomic_load(&gMeterHostWaits), waitS * 1e3,
            (double)atomic_load(&gMeterHostWaitMaxNs) * 1e-6);
    const uint32_t swaps = atomic_load(&gMeterMpsSwaps);
    if (swaps != 0) {
        fprintf(stderr, "             %u MPSGraph command-buffer swaps: that"
                        " work is NOT in the span set, so occupancy is a"
                        " LOWER bound\n", swaps);
    }
    const uint32_t dropped = atomic_load(&gMeterDropped);
    if (dropped != 0) {
        fprintf(stderr, "             %u buffers past the %d-span cap were not"
                        " counted; occupancy is a LOWER bound\n",
                dropped, JAI_METER_MAX);
    }
}

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

id<MTLDevice>       gDevice;
id<MTLCommandQueue> gQueue;
static bool                gDeviceReady;
static int                 gPreferredDevice = -1;
bool                gMixedPrecision;
bool                gNonUniformThreadgroups;
size_t              gMaxBufferLength;

id<MTLComputePipelineState> gVectorAdd;
id<MTLComputePipelineState> gVectorMul;
id<MTLComputePipelineState> gMatMul;
id<MTLComputePipelineState> gReduceSum;
id<MTLComputePipelineState> gExpandU8;
static id<MTLComputePipelineState> gSplitHeads;
static id<MTLComputePipelineState> gMergeHeads;
id<MTLComputePipelineState> gFlashAttn32;
id<MTLComputePipelineState> gFlashAttn64;
id<MTLComputePipelineState> gFlashPack;

static bool                        gBuiltinsReady;

bool ensureDevice(void) {
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
bool ensureBuiltins(void) {
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

bool ensureFlashAttn(void) {
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

#endif /* __APPLE__ */
