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

#include "native/apple/gpu_internal.h"

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
        meterNote(commands, 1u);

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
        meterNote(commands, 1u);

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
        meterNote(commands, 1u);

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
