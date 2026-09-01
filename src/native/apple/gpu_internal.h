/* gpu_internal.h — what the Apple GPU backend's .m files share: the shader
 * constants, the two buffer types, and everything that crosses a file
 * boundary. A *Locked name means the caller already holds
 * @synchronized(gQueue); do not take the lock again inside one. */
#ifndef JAI_GPU_INTERNAL_H
#define JAI_GPU_INTERNAL_H

#ifdef __APPLE__

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "native/native.h"

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

/* Defined in gpu_device.m. */
extern id<MTLDevice>       gDevice;
extern id<MTLCommandQueue> gQueue;
extern bool                gMixedPrecision;
extern bool                gNonUniformThreadgroups;
extern size_t              gMaxBufferLength;

extern id<MTLComputePipelineState> gVectorAdd;
extern id<MTLComputePipelineState> gVectorMul;
extern id<MTLComputePipelineState> gMatMul;
extern id<MTLComputePipelineState> gReduceSum;
extern id<MTLComputePipelineState> gExpandU8;
extern id<MTLComputePipelineState> gFlashAttn32;
extern id<MTLComputePipelineState> gFlashAttn64;
extern id<MTLComputePipelineState> gFlashPack;

void   dispatchTraceTick(int isGraph);
double meterNow(void);
int    meterOn(void);
void   meterNote(id<MTLCommandBuffer> done, uint32_t dispatches);
void   meterNoteMpsSwap(void);
void   meterHostWait(uint64_t ns);
bool   ensureDevice(void);
bool   ensureBuiltins(void);
bool   ensureFlashAttn(void);

/* Defined in gpu_queue.m. */
extern id<MTLCommandBuffer>         gAsyncCommands;
extern id<MTLComputeCommandEncoder> gAsyncEncoder;

uint64_t doneBatch(void);
void     noteDone(uint64_t batch);
void     beginBatchLocked(void);
void     markLocked(JaiGpuBuffer *b);
void     mark(JaiGpuBuffer *b);
bool     flushAsyncLocked(id<MTLCommandBuffer> *oldestOut, uint64_t *oldestBatch);
bool     ensureAsyncCommandBuffer(void);
void     encodeDispatch(id<MTLComputeCommandEncoder> encoder,
                        id<MTLComputePipelineState> pipeline,
                        NSUInteger threads, NSUInteger groupSize);

/* Defined in gpu_graph.m. */
MPSGraphTensorData *graphData(JaiGpuBuffer *b, size_t offset,
                              NSArray<NSNumber *> *shape);
MPSGraphTensorData *graphDataMTL(id<MTLBuffer> buf, NSArray<NSNumber *> *shape);
MPSGraphShapedType *mlpShapedType(NSArray<NSNumber *> *shape);
MPSGraphTensor     *ampMatMul(MPSGraph *graph, MPSGraphTensor *a, MPSGraphTensor *b,
                              NSString *name);
MPSGraphExecutable *compiledGraph(MPSGraph *graph,
                                  NSArray<MPSGraphTensor *> *feeds,
                                  NSArray<NSArray<NSNumber *> *> *shapes,
                                  NSArray<MPSGraphTensor *> *targets);
id<MTLBuffer> growScratch(id<MTLBuffer> existing, size_t *cap, size_t bytes);
bool blitMany(__unsafe_unretained id<MTLBuffer> *srcs, JaiGpuBuffer **dsts,
              const size_t *offs, const size_t *bytes, int count);
bool prefetchBatchFeeds(JaiGpuBuffer *x, size_t xOff, size_t xStride, size_t xBytes,
                        MPSNDArrayDescriptor *xDesc, NSArray<NSNumber *> *xShape,
                        JaiGpuBuffer *labels, size_t labOff, size_t labStride,
                        size_t labBytes, MPSNDArrayDescriptor *yDesc,
                        NSArray<NSNumber *> *yShape, uint32_t steps,
                        NSMutableArray<MPSGraphTensorData *> *batchX,
                        NSMutableArray<MPSGraphTensorData *> *batchY);
bool encodeGraphOnAsync(MPSGraph *graph,
                        NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds,
                        NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results);
bool encodeMlpExecutableOnAsync(
    MPSGraphExecutable *exec,
    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feedMap,
    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *resultMap);
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
    NSArray<MPSGraphTensorData *> *__strong *resultsCache);

/* Defined in gpu_mlp.m. The three-layer path in gpu_mlp3.m deliberately shares
 * this one loss and correct accumulator and this one ping-pong flag. */
extern id<MTLBuffer>  gMlpScratchAcc, gMlpScratchCorrect;
extern size_t         gMlpCapAcc, gMlpCapCorrect;
extern int            gMlpSide;
extern int            gMlpAccSide;
extern JaiGpuBuffer  *gMlpLiveAcc;
extern size_t         gMlpLiveAccOff;

bool commitMlpAccLocked(void);
bool commitMlpWeightsLocked(void);

/* Defined in gpu_mlp3.m. */
extern int gMlp3Side;

bool commitMlp3WeightsLocked(void);

#endif /* __APPLE__ */
#endif /* JAI_GPU_INTERNAL_H */
