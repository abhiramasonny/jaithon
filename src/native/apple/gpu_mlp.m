/* gpu_mlp.m — the fused two-layer MLP: its graphs, its compiled executables,
 * its weight and accumulator ping-pong, and the entry points. */

#ifdef __APPLE__

#include "native/apple/gpu_internal.h"

static id<MTLBuffer> gMlpScratchW1, gMlpScratchB1, gMlpScratchW2, gMlpScratchB2;
id<MTLBuffer> gMlpScratchAcc, gMlpScratchCorrect;
static size_t gMlpCapW1, gMlpCapB1, gMlpCapW2, gMlpCapB2;
size_t gMlpCapAcc, gMlpCapCorrect;
int gMlpSide;
int gMlpAccSide;
static JaiGpuBuffer *gMlpLiveW1, *gMlpLiveB1, *gMlpLiveW2, *gMlpLiveB2;
JaiGpuBuffer *gMlpLiveAcc;
static size_t gMlpLiveW1Off, gMlpLiveB1Off, gMlpLiveW2Off, gMlpLiveB2Off;
size_t gMlpLiveAccOff;
static size_t gMlpLiveW1Bytes, gMlpLiveB1Bytes, gMlpLiveW2Bytes, gMlpLiveB2Bytes;

bool commitMlpAccLocked(void) {
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

bool commitMlpWeightsLocked(void) {
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

static NSMutableDictionary<NSString *, NSArray *> *gMlpGraphs;
static NSMutableDictionary<NSString *, MPSGraphExecutable *> *gMlpExecutables;

static NSString *mlpCacheKey(uint32_t B, uint32_t inFeatures, uint32_t H, uint32_t C,
                             float lr, bool trackCorrect) {
    return [NSString stringWithFormat:@"%u:%u:%u:%u:%.8f:%d:%d",
            B, inFeatures, H, C, lr, trackCorrect ? 1 : 0, gMixedPrecision ? 1 : 0];
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

#endif /* __APPLE__ */
