/* gpu.m — Metal compute: device buffers, kernels compiled from MSL, and the
 * four built-in operations std.gpu falls back on when no source is supplied.
 *
 * Precision. Metal Shading Language has no `double`, so a device buffer holds
 * 32-bit floats and the built-ins round on the way in and widen on the way out.
 * The CPU paths compute in full double precision — they are more accurate than
 * the device, not merely different — so the two agree to a tolerance and never
 * to the bit. lib/std/gpu.jai documents the same contract to callers.
 *
 * Every built-in is total: it produces the right answer with no device, with a
 * device too small for the buffers, and after any Metal call fails midway. The
 * GPU path is an optimisation that is allowed to give up at any point, and each
 * one that gives up falls into the same scalar code the no-device path uses.
 */

#ifdef __APPLE__

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <math.h>
#include <stdarg.h>

#include "native.h"

/* Below this much arithmetic the upload, the encode and the queue wait cost
 * more than the work. Mirrors MIN_GPU_ELEMENTS in lib/std/gpu.jai. */
#define JAI_GPU_MIN_WORK 4096

/* The reduction kernel declares its threadgroup scratch with this width, so the
 * host must dispatch it with exactly this group size. 256 is within the
 * guaranteed maximum on every Metal device. */
#define JAI_REDUCE_GROUP 256
#define JAI_REDUCE_LOADS 2
#define JAI_MATMUL_TILE 16
#define JAI_VECTOR_LANES 4
#define JAI_DEFAULT_GROUP 256

struct JaiGpuBuffer {
    void  *buffer;   /* id<MTLBuffer>, held at +1 */
    size_t bytes;
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
    "                         uint thread [[thread_position_in_grid]]) {\n"
    "    uint base = thread * JAI_VECTOR_LANES;\n"
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
    "                         uint thread [[thread_position_in_grid]]) {\n"
    "    uint base = thread * JAI_VECTOR_LANES;\n"
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
    "}\n";

/* ------------------------------------------------------------------ */
/* Device                                                              */
/* ------------------------------------------------------------------ */

static id<MTLDevice>       gDevice;
static id<MTLCommandQueue> gQueue;
static bool                gDeviceReady;
static bool                gNonUniformThreadgroups;
static size_t              gMaxBufferLength;

static id<MTLComputePipelineState> gVectorAdd;
static id<MTLComputePipelineState> gVectorMul;
static id<MTLComputePipelineState> gMatMul;
static id<MTLComputePipelineState> gReduceSum;
static bool                        gBuiltinsReady;

static bool ensureDevice(void) {
    static dispatch_once_t once;

    dispatch_once(&once, ^{
        @autoreleasepool {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (device == nil) return;

            id<MTLCommandQueue> queue = [device newCommandQueue];
            if (queue == nil) return;

            gDevice = device;
            gQueue = queue;
            gMaxBufferLength = (size_t)[device maxBufferLength];

            /*
             * Every Apple-silicon Mac is Apple GPU family 7 or newer, but keep
             * the older Apple4/Mac2 capability check so this remains valid for
             * the widest set of Metal-capable Apple hardware.
             */
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
                @[@"jaiVectorAdd", @"jaiVectorMul", @"jaiMatMul", @"jaiReduceSum"];
            id<MTLComputePipelineState> built[4] = {nil, nil, nil, nil};
            for (NSUInteger i = 0; i < 4; i++) {
                id<MTLFunction> fn = [library newFunctionWithName:names[i]];
                if (fn == nil) return;
                built[i] = [gDevice newComputePipelineStateWithFunction:fn error:&error];
                if (built[i] == nil) return;
            }

            gVectorAdd = built[0];
            gVectorMul = built[1];
            gMatMul = built[2];
            gReduceSum = built[3];
            gBuiltinsReady = true;
        }
    });
    return gBuiltinsReady;
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

/* ------------------------------------------------------------------ */
/* Buffers                                                             */
/* ------------------------------------------------------------------ */

JaiGpuBuffer *jaiGpuAlloc(size_t bytes) {
    if (bytes == 0 || !ensureDevice()) return NULL;
    if (bytes > gMaxBufferLength) return NULL;

    @autoreleasepool {
        id<MTLBuffer> buffer =
            [gDevice newBufferWithLength:bytes
                                 options:MTLResourceStorageModeShared];

        if (buffer == nil) return NULL;

        JaiGpuBuffer *b = JAI_ALLOC(JaiGpuBuffer, 1);
        b->buffer = (__bridge_retained void *)buffer;
        b->bytes = bytes;
        return b;
    }
}

void jaiGpuFree(JaiGpuBuffer *b) {
    if (b == NULL) return;

    @autoreleasepool {
        CFBridgingRelease(b->buffer);
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

    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)b->buffer;
    memcpy((uint8_t *)[buffer contents] + offset, src, bytes);
}

void jaiGpuDownload(JaiGpuBuffer *b, void *dst, size_t bytes, size_t offset) {
    if (b == NULL || b->buffer == NULL || dst == NULL || bytes == 0) return;
    if (offset > b->bytes || bytes > b->bytes - offset) return;

    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)b->buffer;
    memcpy(dst, (const uint8_t *)[buffer contents] + offset, bytes);
}

/* ------------------------------------------------------------------ */
/* Kernels                                                             */
/* ------------------------------------------------------------------ */

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
        id<MTLLibrary> library = [gDevice newLibraryWithSource:text options:nil error:&error];
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

/* One-dimensional dispatch of `threads` threads. `groupSize` 0 means the widest
 * group the pipeline supports; a kernel with a fixed-width threadgroup array
 * must pass that width instead. */
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

bool jaiGpuDispatch(JaiGpuKernel *k, JaiGpuBuffer **buffers, int count,
                    const uint32_t *scalars, int scalarCount,
                    int threads, int groupSize) {
    if (k == NULL || k->pipeline == NULL || threads <= 0) return false;
    if (count < 0 || (count > 0 && buffers == NULL)) return false;
    if (scalarCount < 0 || (scalarCount > 0 && scalars == NULL)) return false;
    if (groupSize < 0) return false;
    if (!ensureDevice()) return false;

    @autoreleasepool {
        id<MTLComputePipelineState> pipeline =
            (__bridge id<MTLComputePipelineState>)k->pipeline;
        id<MTLCommandBuffer> commands = [gQueue commandBufferWithUnretainedReferences];
        if (commands == nil) return false;
        id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
        if (encoder == nil) return false;

        [encoder setComputePipelineState:pipeline];
        for (int i = 0; i < count; i++) {
            JaiGpuBuffer *b = buffers[i];
            if (b == NULL || b->buffer == NULL) {
                [encoder endEncoding];
                return false;
            }
            [encoder setBuffer:(__bridge id<MTLBuffer>)b->buffer
                        offset:0
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
        /* std.gpu promises that every output buffer is readable once dispatch
         * returns, which is what makes a separate fence unnecessary. */
        [commands waitUntilCompleted];
        return [commands status] == MTLCommandBufferStatusCompleted;
    }
}

/* ------------------------------------------------------------------ */
/* Scalar implementations                                              */
/* ------------------------------------------------------------------ */

/* Neumaier's variant of Kahan summation: it also keeps the low bits of the
 * additions a plain Kahan loop drops, the ones where the running total is
 * smaller than the term being added. */
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

/* Row, inner, column order so that both `b` and the output row are walked
 * forwards; the arithmetic is the textbook triple loop's, the cache behaviour
 * is not. */
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

/* ------------------------------------------------------------------ */
/* Built-in kernels                                                    */
/* ------------------------------------------------------------------ */

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

    /*
     * Apple-silicon buffers are shared CPU/GPU memory. Write-combined caching
     * is ideal here because the CPU streams values in once and never reads the
     * input buffer back.
     */
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

/* Both elementwise built-ins differ only in their pipeline, so they share one
 * encoder. Returns false — having written nothing — if any Metal call fails,
 * and the caller then runs the scalar path. */
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

    /*
     * Always use complete 16x16 groups here. Unlike simple 1D kernels, the
     * tiled shader relies on every lane existing to populate threadgroup tiles,
     * so partial/non-uniform edge groups would leave scratch entries unloaded.
     */
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
