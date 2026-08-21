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
            /* A cast to boolean, which ONNX writes as one and zero. */
            case 16: out = [g notEqualWithPrimaryTensor:in
                                       secondaryTensor:[g constantWithScalar:0.0
                                                                    dataType:in.dataType]
                                                  name:nil]; break;
            default: return -1;
        }
        return record(b, out);
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
            default: return -1;
        }
        return record(b, out);
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
                             dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                          weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
        if (d == nil) return -1;
        MPSGraphTensor *out = [g convolution2DWithSourceTensor:in
                                                 weightsTensor:filt
                                                    descriptor:d
                                                          name:nil];
        MPSGraphTensor *shift = tensorAt(b, bias);
        if (shift != nil) {
            /* A convolution's bias is one value a channel, and the channel is
             * the second axis -- so it is widened to `[1, C, 1, 1]` rather
             * than left rank one, which would try to line up with the width
             * and not broadcast at all. */
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
 * pad left, pad right. `kind` 0 is maximum, 1 is average. */
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
                                                          dataLayout:MPSGraphTensorNamedDataLayoutNCHW];
        if (d == nil) return -1;
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
 * difference of one pixel rather than an error. */
int jaiGraphResizeNearest(JaiGraphBuilder *b, int x, int height, int width,
                          int rounding, int center, int corners) {
    MPSGraphTensor *in = tensorAt(b, x);
    if (in == nil || height <= 0 || width <= 0) return -1;
    if (rounding < 0 || rounding > 3) return -1;
    @autoreleasepool {
        MPSGraph *g = (__bridge MPSGraph *)b->graph;
        if (@available(macOS 13.0, *)) {
            const int32_t wanted[2] = {(int32_t)height, (int32_t)width};
            NSData *data = [NSData dataWithBytes:wanted length:sizeof(wanted)];
            [(__bridge NSMutableArray *)b->held addObject:data];
            MPSGraphTensor *size = [g constantWithData:data
                                                 shape:@[ @2 ]
                                              dataType:MPSDataTypeInt32];
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
static void chooseRoute(JaiGraphPlan *plan, NSArray *feeds, NSArray *results,
                        id<MTLCommandQueue> queue) {
    MPSGraphExecutable *executable = (__bridge MPSGraphExecutable *)plan->executable;

    /* Best of several, after a warm pass: the first execution of either route
     * pays for scheduling the ones after it do not, and a single sample of
     * either is well inside the noise of the other. */
    const int PROBES = 4;
    double encoded = INFINITY;
    double ran = INFINITY;

    jaiGpuEncodeExecutable((__bridge void *)executable, (__bridge void *)feeds,
                           (__bridge void *)results);
    jaiGpuSynchronize();
    for (int i = 0; i < PROBES; i++) {
        NSDate *started = [NSDate date];
        jaiGpuEncodeExecutable((__bridge void *)executable, (__bridge void *)feeds,
                               (__bridge void *)results);
        jaiGpuSynchronize();
        const double took = -[started timeIntervalSinceNow];
        if (took < encoded) encoded = took;
    }

    [executable runWithMTLCommandQueue:queue inputsArray:feeds resultsArray:results
                   executionDescriptor:nil];
    for (int i = 0; i < PROBES; i++) {
        NSDate *started = [NSDate date];
        [executable runWithMTLCommandQueue:queue inputsArray:feeds resultsArray:results
                       executionDescriptor:nil];
        const double took = -[started timeIntervalSinceNow];
        if (took < ran) ran = took;
    }

    plan->route = ran < encoded ? ROUTE_RUN : ROUTE_ENCODE;
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
        NSMutableArray<MPSGraphTensorData *> *feeds = [NSMutableArray new];
        for (int i = 0; i < plan->inputCount; i++) {
            id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)jaiGpuBufferHandle(ins[i]);
            if (buffer == nil) return false;
            MPSGraphTensorData *data =
                [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer
                                                        shape:inShapes[(NSUInteger)i]
                                                     dataType:MPSDataTypeFloat32];
            if (data == nil) return false;
            [feeds addObject:data];
        }
        NSMutableArray<MPSGraphTensorData *> *results = [NSMutableArray new];
        for (int i = 0; i < plan->outputCount; i++) {
            id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)jaiGpuBufferHandle(outs[i]);
            if (buffer == nil) return false;
            MPSGraphTensorData *data =
                [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer
                                                        shape:outShapes[(NSUInteger)i]
                                                     dataType:MPSDataTypeFloat32];
            if (data == nil) return false;
            [results addObject:data];
        }

        if (plan->route == ROUTE_UNDECIDED) chooseRoute(plan, feeds, results, queue);

        MPSGraphExecutable *executable = (__bridge MPSGraphExecutable *)plan->executable;
        if (plan->route == ROUTE_ENCODE) {
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
                      executionDescriptor:nil];
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
