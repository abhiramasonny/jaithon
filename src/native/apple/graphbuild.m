/* graphbuild.m — building a whole network as one MPSGraph, rather than
 * dispatching its operators one at a time.
 *
 * jaicv's executor runs an imported model operator by operator, allocating a
 * tensor between each pair. That is the only way to run a graph whose shape is
 * not known until it runs, and it is what the fallback still does. But when
 * the whole graph IS known, handing all of it to MPSGraph at once is a
 * different proposition: the compiler fuses across operators, keeps
 * intermediates in registers instead of device memory, and -- at optimisation
 * level one -- places parts of the work on whichever of the GPU, the Neural
 * Engine and the CPU it judges best. That placement pass is the only route to
 * the Neural Engine that does not go through CoreML, and it is free here.
 *
 * The API is deliberately flat: a builder collects tensors by integer id, a
 * compile turns the builder into a plan, and a run feeds device buffers
 * through it. Everything that decides WHAT to build lives in Jaithon, next to
 * the importer that knows what the file said. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <string.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "native/native.h"

/* Defined in gpu.m: the one device and queue everything shares. */
id<MTLDevice> jaiGpuMetalDevice(void);
id<MTLCommandQueue> jaiGpuMetalQueue(void);
void *jaiGpuBufferHandle(JaiGpuBuffer *b);
void  jaiGpuBufferMark(JaiGpuBuffer *b);

struct JaiGraphBuilder {
    void *graph;    /* MPSGraph *, +1 */
    void *tensors;  /* NSMutableArray<MPSGraphTensor *> *, +1 */
    void *held;     /* NSMutableArray<NSData *> *, +1 -- constants' backing */
};

struct JaiGraphPlan {
    void *executable;   /* MPSGraphExecutable *, +1 */
    void *inputShapes;  /* NSArray<NSArray<NSNumber *> *> *, +1 */
    void *outputShapes; /* NSArray<NSArray<NSNumber *> *> *, +1 */
    int   inputCount;
    int   outputCount;
    /* Which way this plan reaches the hardware: 0 not decided yet, 1 encoded
     * into the shared command buffer, 2 run by MPSGraph itself. */
    int   route;
};

/* Encoding puts every operation into a Metal command buffer, which is a
 * promise the GPU executes all of it, and it batches with whatever else is
 * queued. Running hands the executable to MPSGraph, which schedules its own
 * work and is free to place parts of it on the Neural Engine or the CPU -- but
 * pays a submission of its own, and cannot batch.
 *
 * Which of those wins depends on the graph, so a plan tries both once and
 * keeps the faster. `JAITHON_GRAPH_ROUTE` set to `encode` or `run` decides it
 * instead, which is how the two are compared. */
enum { ROUTE_UNDECIDED = 0, ROUTE_ENCODE = 1, ROUTE_RUN = 2 };

/* How much quicker running has to be before it is worth losing the overlap
 * that encoding allows. See chooseRoute. */
#define JAI_GRAPH_RUN_MUST_BEAT 0.85

static int forcedRoute(void) {
    const char *setting = getenv("JAITHON_GRAPH_ROUTE");
    if (setting == NULL) return ROUTE_UNDECIDED;
    if (strcmp(setting, "encode") == 0) return ROUTE_ENCODE;
    if (strcmp(setting, "run") == 0) return ROUTE_RUN;
    return ROUTE_UNDECIDED;
}

static NSArray<NSNumber *> *shapeOf(const int64_t *dims, int rank) {
    NSMutableArray *shape = [NSMutableArray arrayWithCapacity:(NSUInteger)rank];
    for (int i = 0; i < rank; i++) [shape addObject:@(dims[i])];
    return shape;
}

/* The tensor an id names, or nil when the id is out of range. Every builder
 * entry point goes through this, so a bad id from the caller is a nil result
 * rather than a crash. */
static MPSGraphTensor *tensorAt(JaiGraphBuilder *b, int id) {
    if (b == NULL || id < 0) return nil;
    NSMutableArray *tensors = (__bridge NSMutableArray *)b->tensors;
    if ((NSUInteger)id >= tensors.count) return nil;
    return tensors[(NSUInteger)id];
}

static int record(JaiGraphBuilder *b, MPSGraphTensor *tensor) {
    if (tensor == nil) return -1;
    NSMutableArray *tensors = (__bridge NSMutableArray *)b->tensors;
    [tensors addObject:tensor];
    return (int)(tensors.count - 1);
}

JaiGraphBuilder *jaiGraphNew(void) {
    if (jaiGpuMetalDevice() == nil) return NULL;
    @autoreleasepool {
        MPSGraph *graph = [MPSGraph new];
        if (graph == nil) return NULL;
        graph.options = MPSGraphOptionsNone;
        JaiGraphBuilder *b = JAI_ALLOC(JaiGraphBuilder, 1);
        b->graph = (__bridge_retained void *)graph;
        b->tensors = (__bridge_retained void *)[NSMutableArray new];
        b->held = (__bridge_retained void *)[NSMutableArray new];
        return b;
    }
}

void jaiGraphFree(JaiGraphBuilder *b) {
    if (b == NULL) return;
    @autoreleasepool {
        CFBridgingRelease(b->graph);
        CFBridgingRelease(b->tensors);
        CFBridgingRelease(b->held);
    }
    JAI_FREE(JaiGraphBuilder, b);
}

int jaiGraphInput(JaiGraphBuilder *b, const int64_t *dims, int rank) {
    if (b == NULL || dims == NULL || rank <= 0) return -1;
    @autoreleasepool {
        MPSGraph *graph = (__bridge MPSGraph *)b->graph;
        return record(b, [graph placeholderWithShape:shapeOf(dims, rank)
                                            dataType:MPSDataTypeFloat32
                                                name:nil]);
    }
}

int jaiGraphConstant(JaiGraphBuilder *b, const float *values,
                     const int64_t *dims, int rank) {
    if (b == NULL || values == NULL || dims == NULL || rank <= 0) return -1;
    size_t count = 1;
    for (int i = 0; i < rank; i++) {
        if (dims[i] < 0) return -1;
        count *= (size_t)dims[i];
    }
    @autoreleasepool {
        /* Copied rather than referenced: the weights this comes from belong to
         * the importer, and the graph outlives any one call into it. */
        NSData *data = [NSData dataWithBytes:values length:count * sizeof(float)];
        [(__bridge NSMutableArray *)b->held addObject:data];
        MPSGraph *graph = (__bridge MPSGraph *)b->graph;
        return record(b, [graph constantWithData:data
                                           shape:shapeOf(dims, rank)
                                        dataType:MPSDataTypeFloat32]);
    }
}

int jaiGraphUnary(JaiGraphBuilder *b, int x, int op) {
    MPSGraphTensor *in = tensorAt(b, x);
    if (in == nil) return -1;
    @autoreleasepool {
        MPSGraph *g = (__bridge MPSGraph *)b->graph;
        MPSGraphTensor *out = nil;
        switch (op) {
            case 0:  out = [g reLUWithTensor:in name:nil]; break;
            case 1:  out = [g sigmoidWithTensor:in name:nil]; break;
            case 2:  out = [g tanhWithTensor:in name:nil]; break;
            case 3:  out = [g exponentWithTensor:in name:nil]; break;
            case 4:  out = [g logarithmWithTensor:in name:nil]; break;
            case 5:  out = [g squareRootWithTensor:in name:nil]; break;
            case 6:  out = [g negativeWithTensor:in name:nil]; break;
            case 7:  out = [g absoluteWithTensor:in name:nil]; break;
            case 8:  out = [g erfWithTensor:in name:nil]; break;
            /* SiLU has no operator of its own, and writing it here rather than
             * as two nodes lets the compiler see the whole shape of it. */
            case 9:  out = [g multiplicationWithPrimaryTensor:in
                                             secondaryTensor:[g sigmoidWithTensor:in name:nil]
                                                        name:nil]; break;
            case 10: out = [g floorWithTensor:in name:nil]; break;
            case 11: out = [g ceilWithTensor:in name:nil]; break;
            case 12: out = [g reciprocalWithTensor:in name:nil]; break;
            case 13: out = [g squareWithTensor:in name:nil]; break;
            /* `reverseSquareRoot` is marked deprecated but is still the only
             * spelling MPSGraph offers for it, so the warning is silenced
             * rather than the call replaced. */
            case 14: {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                out = [g reverseSquareRootWithTensor:in name:nil];
#pragma clang diagnostic pop
                break;
            }
            case 15: out = [g truncateWithTensor:in name:nil]; break;
            /* A cast to boolean, which ONNX writes as one and zero. The
             * comparison yields a boolean tensor, and every tensor that
             * leaves here is float, so it is cast back. */
            case 16: {
                MPSGraphTensor *flag =
                    [g notEqualWithPrimaryTensor:in
                                 secondaryTensor:[g constantWithScalar:0.0
                                                              dataType:in.dataType]
                                            name:nil];
                out = [g castTensor:flag toType:MPSDataTypeFloat32 name:nil];
                break;
            }
            default: return -1;
        }
        return record(b, out);
    }
}

/* `x` clamped into `[low, high]`, which is what ONNX's Clip is once its
 * bounds are known -- and they are constants in every model that uses it for
 * a bounded activation. */
int jaiGraphClamp(JaiGraphBuilder *b, int x, float low, float high) {
    MPSGraphTensor *in = tensorAt(b, x);
    if (in == nil || !(low <= high)) return -1;
    @autoreleasepool {
        MPSGraph *g = (__bridge MPSGraph *)b->graph;
        MPSGraphTensor *out =
            [g maximumWithPrimaryTensor:in
                        secondaryTensor:[g constantWithScalar:(double)low dataType:in.dataType]
                                   name:nil];
        out = [g minimumWithPrimaryTensor:out
                          secondaryTensor:[g constantWithScalar:(double)high dataType:in.dataType]
                                     name:nil];
        return record(b, out);
    }
}

int jaiGraphLeakyRelu(JaiGraphBuilder *b, int x, float slope) {
    MPSGraphTensor *in = tensorAt(b, x);
    if (in == nil) return -1;
    @autoreleasepool {
        MPSGraph *g = (__bridge MPSGraph *)b->graph;
        return record(b, [g leakyReLUWithTensor:in alpha:(double)slope name:nil]);
    }
}

int jaiGraphBinary(JaiGraphBuilder *b, int left, int right, int op) {
    MPSGraphTensor *a = tensorAt(b, left);
    MPSGraphTensor *c = tensorAt(b, right);
    if (a == nil || c == nil) return -1;
    @autoreleasepool {
        MPSGraph *g = (__bridge MPSGraph *)b->graph;
        MPSGraphTensor *out = nil;
        switch (op) {
            case 0: out = [g additionWithPrimaryTensor:a secondaryTensor:c name:nil]; break;
            case 1: out = [g subtractionWithPrimaryTensor:a secondaryTensor:c name:nil]; break;
            case 2: out = [g multiplicationWithPrimaryTensor:a secondaryTensor:c name:nil]; break;
            case 3: out = [g divisionWithPrimaryTensor:a secondaryTensor:c name:nil]; break;
            case 4: out = [g maximumWithPrimaryTensor:a secondaryTensor:c name:nil]; break;
            case 5: out = [g minimumWithPrimaryTensor:a secondaryTensor:c name:nil]; break;
            case 6: out = [g powerWithPrimaryTensor:a secondaryTensor:c name:nil]; break;
            /* The comparisons yield boolean tensors and everything that leaves
             * here is float, so each is cast back to the one and zero ONNX
             * says it produces. */
            case 7:  out = [g equalWithPrimaryTensor:a secondaryTensor:c name:nil]; break;
            case 8:  out = [g greaterThanWithPrimaryTensor:a secondaryTensor:c name:nil]; break;
            case 9:  out = [g lessThanWithPrimaryTensor:a secondaryTensor:c name:nil]; break;
            case 10: out = [g logicalANDWithPrimaryTensor:a secondaryTensor:c name:nil]; break;
            case 11: out = [g logicalORWithPrimaryTensor:a secondaryTensor:c name:nil]; break;
            default: return -1;
        }
        if (op >= 7 && op <= 11 && out.dataType != MPSDataTypeFloat32) {
            out = [g castTensor:out toType:MPSDataTypeFloat32 name:nil];
        }
        return record(b, out);
    }
}

/* Pick from `whenTrue` where `predicate` is not zero and from `whenFalse`
 * elsewhere -- ONNX's `Where`, whose predicate arrives here as a float. */
int jaiGraphSelect(JaiGraphBuilder *b, int predicate, int whenTrue, int whenFalse) {
    MPSGraphTensor *p = tensorAt(b, predicate);
    MPSGraphTensor *yes = tensorAt(b, whenTrue);
    MPSGraphTensor *no = tensorAt(b, whenFalse);
    if (p == nil || yes == nil || no == nil) return -1;
    @autoreleasepool {
        MPSGraph *g = (__bridge MPSGraph *)b->graph;
        MPSGraphTensor *flag =
            [g notEqualWithPrimaryTensor:p
                         secondaryTensor:[g constantWithScalar:0.0 dataType:p.dataType]
                                    name:nil];
        return record(b, [g selectWithPredicateTensor:flag
                                  truePredicateTensor:yes
                                 falsePredicateTensor:no
                                                 name:nil]);
    }
}

/* `p` is stride y, stride x, pad top, pad bottom, pad left, pad right,
 * dilation y, dilation x, groups. NCHW in, OIHW weights, which is the layout
 * every exported model already carries. */
int jaiGraphConv(JaiGraphBuilder *b, int x, int w, int bias, const int32_t *p) {
    MPSGraphTensor *in = tensorAt(b, x);
    MPSGraphTensor *filt = tensorAt(b, w);
    if (in == nil || filt == nil || p == NULL) return -1;
    @autoreleasepool {
        MPSGraph *g = (__bridge MPSGraph *)b->graph;
        MPSGraphConvolution2DOpDescriptor *d =
            [MPSGraphConvolution2DOpDescriptor
                descriptorWithStrideInX:(NSUInteger)p[1]
                              strideInY:(NSUInteger)p[0]
                        dilationRateInX:(NSUInteger)p[7]
                        dilationRateInY:(NSUInteger)p[6]
                                 groups:(NSUInteger)p[8]
                            paddingLeft:(NSUInteger)p[4]
                           paddingRight:(NSUInteger)p[5]
                             paddingTop:(NSUInteger)p[2]
                          paddingBottom:(NSUInteger)p[3]
                           paddingStyle:MPSGraphPaddingStyleExplicit
                             dataLayout:(p[9] != 0 ? MPSGraphTensorNamedDataLayoutNHWC
                                                   : MPSGraphTensorNamedDataLayoutNCHW)
                          weightsLayout:(p[10] != 0 ? MPSGraphTensorNamedDataLayoutHWIO
                                                    : MPSGraphTensorNamedDataLayoutOIHW)];
        if (d == nil) return -1;
        MPSGraphTensor *out = [g convolution2DWithSourceTensor:in
                                                 weightsTensor:filt
                                                    descriptor:d
                                                          name:nil];
        MPSGraphTensor *shift = tensorAt(b, bias);
        if (shift != nil) {
            /* A convolution's bias is one value a channel, so it is widened
             * to sit on the channel axis rather than left rank one, which
             * would try to line up with the width and not broadcast at all.
             * Which axis that is depends on the layout. */
            if (shift.shape.count == 1) {
                NSArray *widened = p[9] != 0
                    ? @[ @1, @1, @1, shift.shape[0] ]
                    : @[ @1, shift.shape[0], @1, @1 ];
                shift = [g reshapeTensor:shift withShape:widened name:nil];
            }
            out = [g additionWithPrimaryTensor:out secondaryTensor:shift name:nil];
        }
        return record(b, out);
    }
}

/* A transposed convolution, taking the same nine parameters as the forward
 * one it undoes plus the shape it has to produce. The weights are in the
 * forward layout, OIHW, which is not the layout ONNX writes them in -- the
 * caller swaps the two channel axes before it gets here. */
int jaiGraphConvTranspose(JaiGraphBuilder *b, int x, int w, int bias,
                          const int32_t *p, const int64_t *shape) {
    MPSGraphTensor *in = tensorAt(b, x);
    MPSGraphTensor *filt = tensorAt(b, w);
    if (in == nil || filt == nil || p == NULL || shape == NULL) return -1;
    @autoreleasepool {
        MPSGraph *g = (__bridge MPSGraph *)b->graph;
        MPSGraphConvolution2DOpDescriptor *d =
            [MPSGraphConvolution2DOpDescriptor
                descriptorWithStrideInX:(NSUInteger)p[1]
                              strideInY:(NSUInteger)p[0]
                        dilationRateInX:(NSUInteger)p[7]
                        dilationRateInY:(NSUInteger)p[6]
                                 groups:(NSUInteger)p[8]
                            paddingLeft:(NSUInteger)p[4]
                           paddingRight:(NSUInteger)p[5]
                             paddingTop:(NSUInteger)p[2]
                          paddingBottom:(NSUInteger)p[3]
                           paddingStyle:MPSGraphPaddingStyleExplicit
                             dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                          weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
        if (d == nil) return -1;
        MPSGraphTensor *out =
            [g convolutionTranspose2DWithSourceTensor:in
                                        weightsTensor:filt
                                          outputShape:shapeOf(shape, 4)
                                           descriptor:d
                                                 name:nil];
        if (out == nil) return -1;
        MPSGraphTensor *shift = tensorAt(b, bias);
        if (shift != nil) {
            if (shift.shape.count == 1) {
                shift = [g reshapeTensor:shift
                               withShape:@[ @1, shift.shape[0], @1, @1 ]
                                    name:nil];
            }
            out = [g additionWithPrimaryTensor:out secondaryTensor:shift name:nil];
        }
        return record(b, out);
    }
}

/* `p` is kernel y, kernel x, stride y, stride x, pad top, pad bottom,
 * pad left, pad right, ceil mode, and whether the padding counts toward an
 * average. `kind` 0 is maximum, 1 is average. */
int jaiGraphPool(JaiGraphBuilder *b, int x, const int32_t *p, int kind) {
    MPSGraphTensor *in = tensorAt(b, x);
    if (in == nil || p == NULL) return -1;
    @autoreleasepool {
        MPSGraph *g = (__bridge MPSGraph *)b->graph;
        MPSGraphPooling2DOpDescriptor *d =
            [MPSGraphPooling2DOpDescriptor descriptorWithKernelWidth:(NSUInteger)p[1]
                                                        kernelHeight:(NSUInteger)p[0]
                                                           strideInX:(NSUInteger)p[3]
                                                           strideInY:(NSUInteger)p[2]
                                                     dilationRateInX:1
                                                     dilationRateInY:1
                                                         paddingLeft:(NSUInteger)p[6]
                                                        paddingRight:(NSUInteger)p[7]
                                                          paddingTop:(NSUInteger)p[4]
                                                       paddingBottom:(NSUInteger)p[5]
                                                        paddingStyle:MPSGraphPaddingStyleExplicit
                                                          dataLayout:(p[10] != 0
                                                              ? MPSGraphTensorNamedDataLayoutNHWC
                                                              : MPSGraphTensorNamedDataLayoutNCHW)];
        if (d == nil) return -1;
        if (@available(macOS 12.0, *)) {
            d.ceilMode = p[8] != 0;
            d.includeZeroPadToAverage = p[9] != 0;
        } else if (p[8] != 0) {
            return -1;
        }
        MPSGraphTensor *out = kind == 0
            ? [g maxPooling2DWithSourceTensor:in descriptor:d name:nil]
            : [g avgPooling2DWithSourceTensor:in descriptor:d name:nil];
        return record(b, out);
    }
}

int jaiGraphConcat(JaiGraphBuilder *b, const int *ids, int count, int axis) {
    if (b == NULL || ids == NULL || count <= 0) return -1;
    @autoreleasepool {
        NSMutableArray<MPSGraphTensor *> *parts = [NSMutableArray arrayWithCapacity:(NSUInteger)count];
        for (int i = 0; i < count; i++) {
            MPSGraphTensor *part = tensorAt(b, ids[i]);
            if (part == nil) return -1;
            [parts addObject:part];
        }
        MPSGraph *g = (__bridge MPSGraph *)b->graph;
        return record(b, [g concatTensors:parts dimension:axis name:nil]);
    }
}

int jaiGraphReshape(JaiGraphBuilder *b, int x, const int64_t *dims, int rank) {
    MPSGraphTensor *in = tensorAt(b, x);
    if (in == nil || dims == NULL || rank <= 0) return -1;
    @autoreleasepool {
        MPSGraph *g = (__bridge MPSGraph *)b->graph;
        return record(b, [g reshapeTensor:in withShape:shapeOf(dims, rank) name:nil]);
    }
}

int jaiGraphTranspose(JaiGraphBuilder *b, int x, const int32_t *perm, int rank) {
    MPSGraphTensor *in = tensorAt(b, x);
    if (in == nil || perm == NULL || rank <= 0) return -1;
    @autoreleasepool {
        NSMutableArray *order = [NSMutableArray arrayWithCapacity:(NSUInteger)rank];
        for (int i = 0; i < rank; i++) [order addObject:@(perm[i])];
        MPSGraph *g = (__bridge MPSGraph *)b->graph;
        return record(b, [g transposeTensor:in permutation:order name:nil]);
    }
}

int jaiGraphSlice(JaiGraphBuilder *b, int x, const int32_t *starts,
                  const int32_t *ends, const int32_t *steps, int rank) {
    MPSGraphTensor *in = tensorAt(b, x);
    if (in == nil || starts == NULL || ends == NULL || steps == NULL || rank <= 0) return -1;
    @autoreleasepool {
        NSMutableArray *from = [NSMutableArray arrayWithCapacity:(NSUInteger)rank];
        NSMutableArray *to = [NSMutableArray arrayWithCapacity:(NSUInteger)rank];
        NSMutableArray *by = [NSMutableArray arrayWithCapacity:(NSUInteger)rank];
        for (int i = 0; i < rank; i++) {
            [from addObject:@(starts[i])];
            [to addObject:@(ends[i])];
            [by addObject:@(steps[i])];
        }
        MPSGraph *g = (__bridge MPSGraph *)b->graph;
        return record(b, [g sliceTensor:in starts:from ends:to strides:by name:nil]);
    }
}

/* `mode` is 0 constant, 1 reflect, 2 symmetric (ONNX's "edge" is 3). The two
 * padding arrays hold one value per axis. */
int jaiGraphPad(JaiGraphBuilder *b, int x, const int32_t *before, const int32_t *after,
                int rank, int mode, float value) {
    MPSGraphTensor *in = tensorAt(b, x);
    if (in == nil || before == NULL || after == NULL || rank <= 0) return -1;
    if (mode < 0 || mode > 3) return -1;
    @autoreleasepool {
        NSMutableArray *left = [NSMutableArray arrayWithCapacity:(NSUInteger)rank];
        NSMutableArray *right = [NSMutableArray arrayWithCapacity:(NSUInteger)rank];
        for (int i = 0; i < rank; i++) {
            if (before[i] < 0 || after[i] < 0) return -1;
            [left addObject:@(before[i])];
            [right addObject:@(after[i])];
        }
        MPSGraph *g = (__bridge MPSGraph *)b->graph;
        return record(b, [g padTensor:in
                      withPaddingMode:(MPSGraphPaddingMode)mode
                          leftPadding:left
                         rightPadding:right
                        constantValue:(double)value
                                 name:nil]);
    }
}

/* `largest` picks between the greatest and the least. The index comes back as
 * a float, which is what every tensor here is. */
int jaiGraphArgExtreme(JaiGraphBuilder *b, int x, int axis, int largest) {
    MPSGraphTensor *in = tensorAt(b, x);
    if (in == nil) return -1;
    @autoreleasepool {
        MPSGraph *g = (__bridge MPSGraph *)b->graph;
        if (@available(macOS 12.0, *)) {
            MPSGraphTensor *out = largest != 0
                ? [g reductionArgMaximumWithTensor:in axis:axis name:nil]
                : [g reductionArgMinimumWithTensor:in axis:axis name:nil];
            if (out == nil) return -1;
            if (out.dataType != MPSDataTypeFloat32) {
                out = [g castTensor:out toType:MPSDataTypeFloat32 name:nil];
            }
            return record(b, out);
        }
        return -1;
    }
}

int jaiGraphSoftmax(JaiGraphBuilder *b, int x, int axis) {
    MPSGraphTensor *in = tensorAt(b, x);
    if (in == nil) return -1;
    @autoreleasepool {
        MPSGraph *g = (__bridge MPSGraph *)b->graph;
        return record(b, [g softMaxWithTensor:in axis:axis name:nil]);
    }
}

/* `rounding` is MPSGraph's own enumeration: 0 round-prefer-ceil, 1
 * round-prefer-floor, 2 ceil, 3 floor. `center` and `corners` are how the
 * source coordinate is derived, which is what ONNX calls the coordinate
 * transformation mode. The caller maps its own spelling onto these and
 * refuses what it cannot map, because guessing here would be a silent
 * difference of one pixel rather than an error. `bilinear` picks between
 * the two samplings, and `rounding` means nothing for the smooth one. */
int jaiGraphResize(JaiGraphBuilder *b, int x, int height, int width,
                   int rounding, int center, int corners, int bilinear) {
    MPSGraphTensor *in = tensorAt(b, x);
    if (in == nil || height <= 0 || width <= 0) return -1;
    if (!bilinear && (rounding < 0 || rounding > 3)) return -1;
    @autoreleasepool {
        MPSGraph *g = (__bridge MPSGraph *)b->graph;
        if (@available(macOS 13.0, *)) {
            const int32_t wanted[2] = {(int32_t)height, (int32_t)width};
            NSData *data = [NSData dataWithBytes:wanted length:sizeof(wanted)];
            [(__bridge NSMutableArray *)b->held addObject:data];
            MPSGraphTensor *size = [g constantWithData:data
                                                 shape:@[ @2 ]
                                              dataType:MPSDataTypeInt32];
            if (bilinear) {
                return record(b, [g resizeBilinearWithTensor:in
                                                  sizeTensor:size
                                                centerResult:center != 0
                                                alignCorners:corners != 0
                                                      layout:MPSGraphTensorNamedDataLayoutNCHW
                                                        name:nil]);
            }
            return record(b, [g resizeNearestWithTensor:in
                                             sizeTensor:size
                                    nearestRoundingMode:(MPSGraphResizeNearestRoundingMode)rounding
                                           centerResult:center != 0
                                           alignCorners:corners != 0
                                                 layout:MPSGraphTensorNamedDataLayoutNCHW
                                                   name:nil]);
        }
        return -1;
    }
}

/* ONNX `Gather`: pick slices of `data` along `axis` at `indices`.
 *
 * The indices arrive as float, because that is the only element type this
 * runtime's tensors have, so they are narrowed to int32 for the operation. */
int jaiGraphGather(JaiGraphBuilder *b, int data, int indices, int axis) {
    MPSGraphTensor *values = tensorAt(b, data);
    MPSGraphTensor *at = tensorAt(b, indices);
    if (values == nil || at == nil || axis < 0) return -1;
    @autoreleasepool {
        MPSGraph *g = (__bridge MPSGraph *)b->graph;
        if (at.dataType != MPSDataTypeInt32) {
            at = [g castTensor:at toType:MPSDataTypeInt32 name:nil];
        }
        return record(b, [g gatherWithUpdatesTensor:values
                                      indicesTensor:at
                                               axis:(NSUInteger)axis
                                    batchDimensions:0
                                               name:nil]);
    }
}

int jaiGraphMatmul(JaiGraphBuilder *b, int left, int right) {
    MPSGraphTensor *a = tensorAt(b, left);
    MPSGraphTensor *c = tensorAt(b, right);
    if (a == nil || c == nil) return -1;
    @autoreleasepool {
        MPSGraph *g = (__bridge MPSGraph *)b->graph;
        return record(b, [g matrixMultiplicationWithPrimaryTensor:a secondaryTensor:c name:nil]);
    }
}

/* `kind` 0 sum, 1 mean, 2 maximum, 3 minimum. */
int jaiGraphReduce(JaiGraphBuilder *b, int x, const int32_t *axes, int count, int kind) {
    MPSGraphTensor *in = tensorAt(b, x);
    if (in == nil || axes == NULL || count <= 0) return -1;
    @autoreleasepool {
        NSMutableArray *over = [NSMutableArray arrayWithCapacity:(NSUInteger)count];
        for (int i = 0; i < count; i++) [over addObject:@(axes[i])];
        MPSGraph *g = (__bridge MPSGraph *)b->graph;
        MPSGraphTensor *out = nil;
        switch (kind) {
            case 0: out = [g reductionSumWithTensor:in axes:over name:nil]; break;
            case 1: out = [g meanOfTensor:in axes:over name:nil]; break;
            case 2: out = [g reductionMaximumWithTensor:in axes:over name:nil]; break;
            case 3: out = [g reductionMinimumWithTensor:in axes:over name:nil]; break;
            default: return -1;
        }
        return record(b, out);
    }
}

/* Layer normalisation over the given axes, built from the mean, the variance
 * and the affine the framework already has. `gamma` and `beta` may be -1.
 *
 * Every transformer opens each of its blocks with one of these, so a graph
 * compiler that cannot express it hands the whole model back to the
 * interpreter. */
int jaiGraphLayerNorm(JaiGraphBuilder *b, int x, int gamma, int beta,
                      const int32_t *axes, int count, float epsilon) {
    MPSGraphTensor *in = tensorAt(b, x);
    if (in == nil || axes == NULL || count <= 0) return -1;
    @autoreleasepool {
        MPSGraph *g = (__bridge MPSGraph *)b->graph;
        NSMutableArray *over = [NSMutableArray arrayWithCapacity:(NSUInteger)count];
        for (int i = 0; i < count; i++) [over addObject:@(axes[i])];
        MPSGraphTensor *mean = [g meanOfTensor:in axes:over name:nil];
        MPSGraphTensor *variance = [g varianceOfTensor:in meanTensor:mean axes:over name:nil];
        return record(b, [g normalizationWithTensor:in
                                         meanTensor:mean
                                     varianceTensor:variance
                                        gammaTensor:tensorAt(b, gamma)
                                         betaTensor:tensorAt(b, beta)
                                            epsilon:epsilon
                                               name:nil]);
    }
}

/* ONNX `Gemm`: `alpha * A' B' + beta * C`, with either operand optionally
 * transposed. `c` may be -1. */
int jaiGraphGemm(JaiGraphBuilder *b, int left, int right, int c,
                 int transposeLeft, int transposeRight, float alpha, float beta) {
    MPSGraphTensor *a = tensorAt(b, left);
    MPSGraphTensor *w = tensorAt(b, right);
    if (a == nil || w == nil) return -1;
    @autoreleasepool {
        MPSGraph *g = (__bridge MPSGraph *)b->graph;
        if (transposeLeft) a = [g transposeTensor:a dimension:0 withDimension:1 name:nil];
        if (transposeRight) w = [g transposeTensor:w dimension:0 withDimension:1 name:nil];
        MPSGraphTensor *out = [g matrixMultiplicationWithPrimaryTensor:a secondaryTensor:w name:nil];
        if (alpha != 1.0f) {
            MPSGraphTensor *scale = [g constantWithScalar:(double)alpha dataType:MPSDataTypeFloat32];
            out = [g multiplicationWithPrimaryTensor:out secondaryTensor:scale name:nil];
        }
        MPSGraphTensor *shift = tensorAt(b, c);
        if (shift != nil) {
            if (beta != 1.0f) {
                MPSGraphTensor *scale = [g constantWithScalar:(double)beta dataType:MPSDataTypeFloat32];
                shift = [g multiplicationWithPrimaryTensor:shift secondaryTensor:scale name:nil];
            }
            out = [g additionWithPrimaryTensor:out secondaryTensor:shift name:nil];
        }
        return record(b, out);
    }
}

/* A whole LSTM as one operation, rather than a loop over its timesteps.
 *
 * `source` is `[T, N, I]`, `recurrentWeight` `[4H, H]`, `inputWeight` `[4H, I]`
 * and `bias` `[4H]`, with the gates ordered i, f, z, o -- which is not the
 * order an ONNX file writes them in, so the caller permutes the weights before
 * it gets here. `initState` and `initCell` are `[N, H]` and may be -1.
 *
 * Fills `out` with the state and cell sequences, both `[T, N, H]`. Returns
 * false when the operation could not be built. */
bool jaiGraphLstm(JaiGraphBuilder *b, int source, int recurrentWeight, int inputWeight,
                  int bias, int initState, int initCell, int reverse, int *out) {
    MPSGraphTensor *x = tensorAt(b, source);
    MPSGraphTensor *r = tensorAt(b, recurrentWeight);
    MPSGraphTensor *w = tensorAt(b, inputWeight);
    if (x == nil || r == nil || w == nil || out == NULL) return false;
    if (@available(macOS 13.0, *)) {
        @autoreleasepool {
            MPSGraph *g = (__bridge MPSGraph *)b->graph;
            MPSGraphLSTMDescriptor *d = [MPSGraphLSTMDescriptor descriptor];
            if (d == nil) return false;
            d.reverse = reverse != 0;
            d.bidirectional = NO;
            d.produceCell = YES;
            d.training = NO;
            NSArray<MPSGraphTensor *> *made =
                [g LSTMWithSourceTensor:x
                        recurrentWeight:r
                            inputWeight:w
                                   bias:tensorAt(b, bias)
                              initState:tensorAt(b, initState)
                               initCell:tensorAt(b, initCell)
                             descriptor:d
                                   name:nil];
            if (made == nil || made.count < 2) return false;
            out[0] = record(b, made[0]);
            out[1] = record(b, made[1]);
            return out[0] >= 0 && out[1] >= 0;
        }
    }
    return false;
}

/* A whole GRU as one operation.
 *
 * The gates are ordered z, r, h, which is the order ONNX writes them in, so
 * unlike the LSTM nothing has to be permuted.
 *
 * Both reset variants map. Applying the reset after the recurrent weights --
 * ONNX's `linear_before_reset`, and what PyTorch exports -- computes the
 * candidate as `(b2 + h[t-1] R^T) r[t]`, so the candidate gate's recurrent
 * bias goes in on its own as `secondaryBias` while the update and reset gates
 * keep both of theirs summed into `bias`. */
int jaiGraphGru(JaiGraphBuilder *b, int source, int recurrentWeight, int inputWeight,
                int bias, int initState, int reverse, int resetAfter, int resetBias) {
    MPSGraphTensor *x = tensorAt(b, source);
    MPSGraphTensor *r = tensorAt(b, recurrentWeight);
    MPSGraphTensor *w = tensorAt(b, inputWeight);
    if (x == nil || r == nil || w == nil) return -1;
    if (@available(macOS 13.0, *)) {
        @autoreleasepool {
            MPSGraph *g = (__bridge MPSGraph *)b->graph;
            MPSGraphGRUDescriptor *d = [MPSGraphGRUDescriptor descriptor];
            if (d == nil) return -1;
            d.reverse = reverse != 0;
            d.bidirectional = NO;
            d.training = NO;
            /* All three left at their defaults, which is what agrees with
             * onnxruntime on every case the oracle records. `flipZ` in
             * particular reads from its documentation as though ONNX would
             * want it on; it does not, and the tests are the authority. */
            d.resetGateFirst = NO;
            d.resetAfter = resetAfter != 0;
            d.flipZ = NO;
            NSArray<MPSGraphTensor *> *made =
                [g GRUWithSourceTensor:x
                       recurrentWeight:r
                           inputWeight:w
                                  bias:tensorAt(b, bias)
                             initState:tensorAt(b, initState)
                                  mask:nil
                         secondaryBias:tensorAt(b, resetBias)
                            descriptor:d
                                  name:nil];
            if (made == nil || made.count < 1) return -1;
            return record(b, made[0]);
        }
    }
    return -1;
}

/* One-hot rows from integer-valued indices, `depth` wide.
 *
 * Built as its own operation rather than out of comparisons because a
 * comparison has no derivative and MPSGraph aborts the process when asked to
 * differentiate through one. */
int jaiGraphOneHot(JaiGraphBuilder *b, int indices, int depth) {
    MPSGraphTensor *in = tensorAt(b, indices);
    if (in == nil || depth <= 0) return -1;
    @autoreleasepool {
        MPSGraph *g = (__bridge MPSGraph *)b->graph;
        return record(b, [g oneHotWithIndicesTensor:in
                                              depth:(NSUInteger)depth
                                           dataType:MPSDataTypeFloat32
                                            onValue:1.0
                                           offValue:0.0
                                               name:nil]);
    }
}

/* Softmax cross-entropy against one-hot labels. `reduction` 0 none, 1 sum,
 * 2 mean.
 *
 * The platform's own, rather than a log-softmax and a pick built here: this
 * one has a derivative, and a hand-built one is exactly the sort of thing that
 * turns out not to. */
int jaiGraphSoftmaxCrossEntropy(JaiGraphBuilder *b, int logits, int labels, int axis,
                                int reduction) {
    MPSGraphTensor *source = tensorAt(b, logits);
    MPSGraphTensor *target = tensorAt(b, labels);
    if (source == nil || target == nil) return -1;
    if (reduction < 0 || reduction > 2) return -1;
    @autoreleasepool {
        MPSGraph *g = (__bridge MPSGraph *)b->graph;
        return record(b, [g softMaxCrossEntropyWithSourceTensor:source
                                                   labelsTensor:target
                                                           axis:axis
                                                  reductionType:(MPSGraphLossReductionType)reduction
                                                           name:nil]);
    }
}

/* The derivative of `loss` with respect to each of `wants`, as graph tensors.
 *
 * MPSGraph differentiates the graph it already holds, so a whole training step
 * -- the forward pass, the loss, the gradients and the update that follows
 * them -- becomes one compiled thing rather than a few hundred dispatches with
 * the processor between each of them. That is what the fused MLP path does by
 * hand and why it is eight times PyTorch; this is the same trick for any graph
 * that can be built here.
 *
 * EVERY tensor in `wants` has to be one the loss actually depends on. Asked
 * for the derivative with respect to something the forward pass never touched,
 * MPSGraph fails an assertion and takes the process down -- it does not return
 * nothing and it cannot be asked in advance. A caller building a training step
 * therefore collects its parameters as it consumes them, rather than listing
 * what a layer holds. The -1 below is for the case where the differentiation
 * runs but hands back no entry, which is not the same thing. */
bool jaiGraphGradients(JaiGraphBuilder *b, int loss, const int *wants, int count,
                       int *out) {
    if (b == NULL || wants == NULL || out == NULL || count <= 0) return false;
    MPSGraphTensor *target = tensorAt(b, loss);
    if (target == nil) return false;
    @autoreleasepool {
        MPSGraph *graph = (__bridge MPSGraph *)b->graph;
        NSMutableArray<MPSGraphTensor *> *against = [NSMutableArray new];
        for (int i = 0; i < count; i++) {
            MPSGraphTensor *want = tensorAt(b, wants[i]);
            if (want == nil) return false;
            [against addObject:want];
        }
        NSDictionary<MPSGraphTensor *, MPSGraphTensor *> *found =
            [graph gradientForPrimaryTensor:target withTensors:against name:nil];
        if (found == nil) return false;
        for (int i = 0; i < count; i++) {
            MPSGraphTensor *want = tensorAt(b, wants[i]);
            MPSGraphTensor *grad = found[want];
            out[i] = grad == nil ? -1 : record(b, grad);
        }
        return true;
    }
}

JaiGraphPlan *jaiGraphCompile(JaiGraphBuilder *b, const int *inputs, int inputCount,
                              const int *outputs, int outputCount) {
    if (b == NULL || inputs == NULL || outputs == NULL) return NULL;
    if (inputCount < 0 || outputCount <= 0) return NULL;
    @autoreleasepool {
        MPSGraph *graph = (__bridge MPSGraph *)b->graph;
        NSMutableDictionary<MPSGraphTensor *, MPSGraphShapedType *> *feeds = [NSMutableDictionary new];
        NSMutableArray *inShapes = [NSMutableArray new];
        for (int i = 0; i < inputCount; i++) {
            MPSGraphTensor *tensor = tensorAt(b, inputs[i]);
            if (tensor == nil) return NULL;
            feeds[tensor] = [[MPSGraphShapedType alloc] initWithShape:tensor.shape
                                                             dataType:MPSDataTypeFloat32];
            [inShapes addObject:tensor.shape];
        }
        NSMutableArray<MPSGraphTensor *> *targets = [NSMutableArray new];
        NSMutableArray *outShapes = [NSMutableArray new];
        for (int i = 0; i < outputCount; i++) {
            MPSGraphTensor *tensor = tensorAt(b, outputs[i]);
            if (tensor == nil) return NULL;
            [targets addObject:tensor];
            [outShapes addObject:(tensor.shape != nil ? tensor.shape : @[])];
        }

        MPSGraphCompilationDescriptor *descriptor = nil;
        if (@available(macOS 12.3, *)) {
            descriptor = [MPSGraphCompilationDescriptor new];
            /* The level whose placement pass may put parts of the graph on the
             * Neural Engine or the CPU rather than the GPU. */
            descriptor.optimizationLevel = MPSGraphOptimizationLevel1;
        }
        MPSGraphExecutable *executable = [graph compileWithDevice:nil
                                                            feeds:feeds
                                                    targetTensors:targets
                                                 targetOperations:nil
                                            compilationDescriptor:descriptor];
        if (executable == nil) return NULL;

        JaiGraphPlan *plan = JAI_ALLOC(JaiGraphPlan, 1);
        plan->executable = (__bridge_retained void *)executable;
        plan->inputShapes = (__bridge_retained void *)[inShapes copy];
        plan->outputShapes = (__bridge_retained void *)[outShapes copy];
        plan->inputCount = inputCount;
        plan->outputCount = outputCount;
        plan->route = forcedRoute();
        return plan;
    }
}

int jaiGraphPlanOutputRank(JaiGraphPlan *plan, int index) {
    if (plan == NULL || index < 0 || index >= plan->outputCount) return -1;
    NSArray *shapes = (__bridge NSArray *)plan->outputShapes;
    return (int)[shapes[(NSUInteger)index] count];
}

bool jaiGraphPlanOutputShape(JaiGraphPlan *plan, int index, int64_t *dims, int rank) {
    if (plan == NULL || dims == NULL || index < 0 || index >= plan->outputCount) return false;
    NSArray<NSArray<NSNumber *> *> *shapes = (__bridge NSArray *)plan->outputShapes;
    NSArray<NSNumber *> *shape = shapes[(NSUInteger)index];
    if ((int)shape.count != rank) return false;
    for (int i = 0; i < rank; i++) dims[i] = shape[(NSUInteger)i].longLongValue;
    return true;
}

/* One execution by each route, timed, so the plan can keep the faster. Both
 * wait for the work to land, which is the only way the numbers mean anything;
 * it costs two submissions once in a plan's life. */
/* Running hands the work to the queue and, left to itself, returns before it
 * is done -- the results are not in the buffers yet.
 *
 * That went unnoticed for as long as every read of a device buffer drained
 * the whole queue, because the graph submits to the same queue as everything
 * else and draining it swept the graph's own work up too. Once a read waited
 * only for the work that wrote the buffer it was reading, an Add over two fed
 * tensors started coming back with whatever the buffer held before.
 *
 * Waiting here rather than tracking the graph's command buffer keeps the
 * contract of this route what its name says: when it returns, the answer is
 * there. The route that does not wait is the encoding one, and a caller who
 * wants the overlap takes that. */
static MPSGraphExecutableExecutionDescriptor *blockingRun(void) {
    static MPSGraphExecutableExecutionDescriptor *descriptor;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        descriptor = [MPSGraphExecutableExecutionDescriptor new];
        descriptor.waitUntilCompleted = YES;
    });
    return descriptor;
}

static void chooseRoute(JaiGraphPlan *plan, NSArray *feeds, NSArray *results,
                        id<MTLCommandQueue> queue) {
    MPSGraphExecutable *executable = (__bridge MPSGraphExecutable *)plan->executable;

    /* Best of several, warmed first and taken in turns.
     *
     * Warmed because the GPU does not run at its working speed until it has
     * been busy for a while: the first few executions of either route measure
     * the clock coming up rather than the route. In turns because the two are
     * usually within a few per cent of each other, so a ramp part way through
     * a run of one route and not the other decides the answer -- one encode
     * probe that came out at 4.55 ms against its usual 3.18 handed a live loop
     * to the blocking route and cost it twenty frames a second. Alternating
     * puts both routes under whatever the clock is doing at the time. */
    const int WARM = 6;
    const int PROBES = 6;
    double encoded = INFINITY;
    double ran = INFINITY;

    for (int i = 0; i < WARM; i++) {
        jaiGpuEncodeExecutable((__bridge void *)executable, (__bridge void *)feeds,
                               (__bridge void *)results);
        jaiGpuSynchronize();
        [executable runWithMTLCommandQueue:queue inputsArray:feeds resultsArray:results
                       executionDescriptor:blockingRun()];
    }
    for (int i = 0; i < PROBES; i++) {
        NSDate *startedEncode = [NSDate date];
        jaiGpuEncodeExecutable((__bridge void *)executable, (__bridge void *)feeds,
                               (__bridge void *)results);
        jaiGpuSynchronize();
        const double tookEncode = -[startedEncode timeIntervalSinceNow];
        if (tookEncode < encoded) encoded = tookEncode;

        NSDate *startedRun = [NSDate date];
        [executable runWithMTLCommandQueue:queue inputsArray:feeds resultsArray:results
                       executionDescriptor:blockingRun()];
        const double tookRun = -[startedRun timeIntervalSinceNow];
        if (tookRun < ran) ran = tookRun;
    }

    /* Encoding wins ties, and then some.
     *
     * The two are not interchangeable at equal speed: running submits to the
     * queue and blocks until the answer is there, while encoding leaves the
     * work queued and returns, so the caller can do something else while the
     * GPU gets on with it. A live loop that hands over one frame's network and
     * then draws the frame before it costs the slower of the two halves when
     * the route encodes and their sum when it runs.
     *
     * Measured on YOLOv8n the two came out at 3.18 and 3.24 ms, close enough
     * that the probe picked differently from one run to the next and the loop
     * around it ran at either 128 or 105 frames a second depending. Running
     * has to be clearly quicker, not incidentally quicker, to be worth giving
     * that up for. */
    plan->route = ran < encoded * JAI_GRAPH_RUN_MUST_BEAT ? ROUTE_RUN : ROUTE_ENCODE;
    if (getenv("JAITHON_GRAPH_ROUTE_REPORT") != NULL) {
        fprintf(stderr, "[graph] encode %.2fms  run %.2fms  taking %s\n",
                encoded * 1000.0, ran * 1000.0,
                plan->route == ROUTE_RUN ? "run" : "encode");
    }
}

bool jaiGraphRun(JaiGraphPlan *plan, JaiGpuBuffer **ins, const size_t *inOffsets,
                 JaiGpuBuffer **outs, const size_t *outOffsets) {
    if (plan == NULL || outs == NULL || outOffsets == NULL) return false;
    if (plan->inputCount > 0 && (ins == NULL || inOffsets == NULL)) return false;
    id<MTLCommandQueue> queue = jaiGpuMetalQueue();
    if (queue == nil) return false;

    @autoreleasepool {
        NSArray<NSArray<NSNumber *> *> *inShapes = (__bridge NSArray *)plan->inputShapes;
        NSArray<NSArray<NSNumber *> *> *outShapes = (__bridge NSArray *)plan->outputShapes;
        /* Through the window the caller asked for, not from the start of the
         * buffer. A sliced batch is a view with an origin, and binding the
         * whole buffer instead feeds the first batch every time -- training
         * that reports a falling loss while it fits one batch over and over. */
        NSMutableArray<MPSGraphTensorData *> *feeds = [NSMutableArray new];
        for (int i = 0; i < plan->inputCount; i++) {
            MPSGraphTensorData *data = (__bridge_transfer MPSGraphTensorData *)
                jaiGpuTensorDataAt(ins[i], inOffsets[i],
                                   (__bridge void *)inShapes[(NSUInteger)i]);
            if (data == nil) return false;
            [feeds addObject:data];
        }
        NSMutableArray<MPSGraphTensorData *> *results = [NSMutableArray new];
        for (int i = 0; i < plan->outputCount; i++) {
            MPSGraphTensorData *data = (__bridge_transfer MPSGraphTensorData *)
                jaiGpuTensorDataAt(outs[i], outOffsets[i],
                                   (__bridge void *)outShapes[(NSUInteger)i]);
            if (data == nil) return false;
            [results addObject:data];
        }

        if (plan->route == ROUTE_UNDECIDED) chooseRoute(plan, feeds, results, queue);

        MPSGraphExecutable *executable = (__bridge MPSGraphExecutable *)plan->executable;
        if (plan->route == ROUTE_ENCODE) {
            /* Marked here and not a line earlier. The work goes into whatever
             * batch is open at the moment it is encoded, and choosing the
             * route drains the queue several times over on the way to a
             * decision -- a mark taken before that names a batch that has
             * been and gone, and a later read of the output waits for
             * nothing and sees whatever the buffer held before. */
            for (int i = 0; i < plan->inputCount; i++) jaiGpuBufferMark(ins[i]);
            for (int i = 0; i < plan->outputCount; i++) jaiGpuBufferMark(outs[i]);
            return jaiGpuEncodeExecutable((__bridge void *)executable,
                                          (__bridge void *)feeds,
                                          (__bridge void *)results);
        }
        /* Running submits straight to the queue, so anything already encoded
         * and not yet committed has to go first or it would land after. */
        if (!jaiGpuFlush()) return false;
        [executable runWithMTLCommandQueue:queue
                              inputsArray:feeds
                             resultsArray:results
                      executionDescriptor:blockingRun()];
        return true;
    }
}

void jaiGraphPlanFree(JaiGraphPlan *plan) {
    if (plan == NULL) return;
    @autoreleasepool {
        CFBridgingRelease(plan->executable);
        CFBridgingRelease(plan->inputShapes);
        CFBridgingRelease(plan->outputShapes);
    }
    JAI_FREE(JaiGraphPlan, plan);
}
