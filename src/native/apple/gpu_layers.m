/* gpu_layers.m — the two layer primitives that are neither a plain matmul
 * nor the fused MLP: attention and convolution. */

#ifdef __APPLE__

#include "native/apple/gpu_internal.h"

static id<MTLBuffer> gMhaHalfScratch;
static size_t gMhaHalfCap;

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

#endif /* __APPLE__ */
