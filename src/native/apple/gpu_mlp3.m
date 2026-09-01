/* gpu_mlp3.m — the fused three-layer MLP: gpu_mlp.m one layer deeper, and
 * the file that reads that one's shared loss and correct accumulator. */

#ifdef __APPLE__

#include "native/apple/gpu_internal.h"

static id<MTLBuffer> gMlp3ScratchW[4];
static id<MTLBuffer> gMlp3ScratchB[4];
static size_t gMlp3CapW[4];
static size_t gMlp3CapB[4];
int gMlp3Side;
static JaiGpuBuffer *gMlp3LiveW[4];
static JaiGpuBuffer *gMlp3LiveB[4];
static size_t gMlp3LiveWOff[4];
static size_t gMlp3LiveBOff[4];
static size_t gMlp3LiveWBytes[4];
static size_t gMlp3LiveBBytes[4];

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

bool commitMlp3WeightsLocked(void) {
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

#endif /* __APPLE__ */
