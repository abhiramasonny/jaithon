/* gpu_graph.m — the MPSGraph plumbing every op family shares, and the plain
 * matmul that is the thinnest thing built on it. */

#ifdef __APPLE__

#include "native/apple/gpu_internal.h"

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

MPSGraphTensorData *graphData(JaiGpuBuffer *b, size_t offset,
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

bool prefetchBatchFeeds(JaiGpuBuffer *x, size_t xOff, size_t xStride, size_t xBytes,
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

bool encodeGraphOnAsync(MPSGraph *graph,
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
    if (mps.rootCommandBuffer != gAsyncCommands) meterNoteMpsSwap();
    gAsyncCommands = mps.rootCommandBuffer;
    gAsyncEncoder = nil;
    return gAsyncCommands != nil;
}

id<MTLBuffer> growScratch(id<MTLBuffer> existing, size_t *cap, size_t bytes) {
    if (existing != nil && *cap >= bytes) return existing;
    *cap = bytes;
    return [gDevice newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

bool blitMany(__unsafe_unretained id<MTLBuffer> *srcs, JaiGpuBuffer **dsts,
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

MPSGraphTensor *ampMatMul(MPSGraph *graph, MPSGraphTensor *a, MPSGraphTensor *b,
                                 NSString *name) {
    if (!gMixedPrecision) {
        return [graph matrixMultiplicationWithPrimaryTensor:a secondaryTensor:b name:name];
    }
    return ampMatMulForced(graph, a, b, name);
}

MPSGraphTensorData *graphDataMTL(id<MTLBuffer> buf, NSArray<NSNumber *> *shape) {
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
MPSGraphExecutable *compiledGraph(MPSGraph *graph,
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

MPSGraphShapedType *mlpShapedType(NSArray<NSNumber *> *shape) {
    return [[MPSGraphShapedType alloc] initWithShape:shape dataType:MPSDataTypeFloat32];
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
    if (mps.rootCommandBuffer != gAsyncCommands) meterNoteMpsSwap();
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

bool encodeMlpExecutableOnAsync(
    MPSGraphExecutable *exec,
    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feedMap,
    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *resultMap) {
    if (exec == nil || exec.feedTensors == nil || exec.targetTensors == nil) return false;
    NSMutableArray<MPSGraphTensorData *> *inputs = mlpMapArray(exec.feedTensors, feedMap);
    NSMutableArray<MPSGraphTensorData *> *results = mlpMapArray(exec.targetTensors, resultMap);
    if (inputs == nil || results == nil) return false;
    return encodeMlpExecutableOnAsyncArrays(exec, inputs, results);
}

bool encodeEpochBatch(
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

#endif /* __APPLE__ */
