/* coreml.m — running a model on whichever silicon CoreML will give us.
 *
 * This is the only route to the neural accelerator. Metal cannot reach it, and
 * MPSGraph's placement pass does not in practice, so a model that should run
 * there has to go through CoreML.
 *
 * The accelerator is not faster than the GPU at this — a whole graph compiled
 * for the GPU beats it on every network measured. What it is, is SEPARATE: the
 * two run at the same time with almost no interference (0.45 ms of GPU work
 * and 0.73 ms of accelerator work together took 0.80 ms, not 1.18). So the
 * reason to have this is not to make one model faster, it is to run a model
 * and a pile of GPU work at once and pay for only the slower of them.
 *
 * Inputs and outputs are read and written straight out of the device buffers
 * everything else here uses. Storage is shared, so `MLMultiArray` can be
 * pointed at the same bytes rather than given a copy of them. */
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <Metal/Metal.h>

#include <stdlib.h>
#include <string.h>

#include "native/native.h"

const float *jaiGpuMapRead(JaiGpuBuffer *b, size_t elementOffset, size_t count);
void        *jaiGpuBufferHandle(JaiGpuBuffer *b);
bool         jaiGpuSynchronize(void);
bool         jaiGpuWaitFor(JaiGpuBuffer *b);

struct JaiCoreMLModel {
    void *model;    /* MLModel *, +1 */
    void *inputs;   /* NSArray<NSString *> *, +1 -- in declaration order */
    void *outputs;  /* NSArray<NSString *> *, +1 */
};

/* Names in a stable order, because the caller addresses them by position and a
 * dictionary's own order is not something to rely on. */
static NSArray<NSString *> *sortedNames(NSDictionary *descriptions) {
    return [[descriptions allKeys] sortedArrayUsingSelector:@selector(compare:)];
}

JaiCoreMLModel *jaiCoreMLOpen(const char *path, int units, char *errBuf, size_t errSize) {
    if (errBuf != NULL && errSize > 0) errBuf[0] = '\0';
    if (path == NULL) return NULL;
    if (units < 0 || units > 3) return NULL;

    @autoreleasepool {
        NSString *text = [NSString stringWithUTF8String:path];
        if (text == nil) return NULL;
        NSURL *url = [NSURL fileURLWithPath:text];
        NSError *error = nil;

        /* A `.mlmodel` or `.mlpackage` has to be compiled before it can be
         * loaded; a `.mlmodelc` already is. Compiling writes to a temporary
         * directory and takes a noticeable fraction of a second, which is why
         * the result is worth holding on to. */
        NSURL *compiled = url;
        if (![text hasSuffix:@".mlmodelc"]) {
            compiled = [MLModel compileModelAtURL:url error:&error];
            if (compiled == nil) {
                if (errBuf != NULL && errSize > 0) {
                    snprintf(errBuf, errSize, "%s",
                             error != nil ? [[error localizedDescription] UTF8String]
                                          : "the model could not be compiled");
                }
                return NULL;
            }
        }

        MLModelConfiguration *configuration = [MLModelConfiguration new];
        configuration.computeUnits = (MLComputeUnits)units;
        MLModel *model = [MLModel modelWithContentsOfURL:compiled configuration:configuration
                                                   error:&error];
        if (model == nil) {
            if (errBuf != NULL && errSize > 0) {
                snprintf(errBuf, errSize, "%s",
                         error != nil ? [[error localizedDescription] UTF8String]
                                      : "the model could not be loaded");
            }
            return NULL;
        }

        JaiCoreMLModel *held = JAI_ALLOC(JaiCoreMLModel, 1);
        held->model = (__bridge_retained void *)model;
        held->inputs = (__bridge_retained void *)sortedNames(model.modelDescription.inputDescriptionsByName);
        held->outputs = (__bridge_retained void *)sortedNames(model.modelDescription.outputDescriptionsByName);
        return held;
    }
}

void jaiCoreMLClose(JaiCoreMLModel *model) {
    if (model == NULL) return;
    @autoreleasepool {
        CFBridgingRelease(model->model);
        CFBridgingRelease(model->inputs);
        CFBridgingRelease(model->outputs);
    }
    JAI_FREE(JaiCoreMLModel, model);
}

int jaiCoreMLCount(JaiCoreMLModel *model, int outputs) {
    if (model == NULL) return -1;
    NSArray *names = (__bridge NSArray *)(outputs ? model->outputs : model->inputs);
    return (int)names.count;
}

const char *jaiCoreMLName(JaiCoreMLModel *model, int outputs, int index) {
    if (model == NULL || index < 0) return NULL;
    NSArray<NSString *> *names = (__bridge NSArray *)(outputs ? model->outputs : model->inputs);
    if ((NSUInteger)index >= names.count) return NULL;
    return [names[(NSUInteger)index] UTF8String];
}

/* The shape a name is declared with, or -1 when it has none that is fixed. */
int jaiCoreMLShape(JaiCoreMLModel *model, int outputs, int index, int64_t *dims, int room) {
    if (model == NULL || dims == NULL || index < 0) return -1;
    @autoreleasepool {
        MLModel *held = (__bridge MLModel *)model->model;
        NSArray<NSString *> *names = (__bridge NSArray *)(outputs ? model->outputs : model->inputs);
        if ((NSUInteger)index >= names.count) return -1;
        NSDictionary<NSString *, MLFeatureDescription *> *all =
            outputs ? held.modelDescription.outputDescriptionsByName
                    : held.modelDescription.inputDescriptionsByName;
        MLFeatureDescription *description = all[names[(NSUInteger)index]];
        if (description == nil || description.multiArrayConstraint == nil) return -1;
        NSArray<NSNumber *> *shape = description.multiArrayConstraint.shape;
        if ((int)shape.count > room) return -1;
        for (NSUInteger i = 0; i < shape.count; i++) dims[i] = shape[i].longLongValue;
        return (int)shape.count;
    }
}

/* Wrap a device buffer's own memory, rather than copying into a fresh array.
 * Storage is shared, so CoreML and the GPU are looking at the same bytes.
 * The deallocator does nothing because the buffer outlives the call. */
static MLMultiArray *wrap(JaiGpuBuffer *buffer, size_t elementOffset, NSArray<NSNumber *> *shape,
                          bool writing) {
    if (buffer == NULL) return nil;
    NSUInteger count = 1;
    for (NSNumber *dimension in shape) count *= dimension.unsignedIntegerValue;
    void *base = NULL;
    if (writing) {
        id<MTLBuffer> held = (__bridge id<MTLBuffer>)jaiGpuBufferHandle(buffer);
        if (held == nil) return nil;
        jaiGpuWaitFor(buffer);
        base = (float *)[held contents] + elementOffset;
    } else {
        base = (void *)jaiGpuMapRead(buffer, elementOffset, count);
    }
    if (base == NULL) return nil;

    NSMutableArray<NSNumber *> *strides = [NSMutableArray arrayWithCapacity:shape.count];
    NSUInteger step = 1;
    for (NSUInteger i = shape.count; i > 0; i--) {
        [strides insertObject:@(step) atIndex:0];
        step *= shape[i - 1].unsignedIntegerValue;
    }
    NSError *error = nil;
    return [[MLMultiArray alloc] initWithDataPointer:base
                                               shape:shape
                                            dataType:MLMultiArrayDataTypeFloat32
                                             strides:strides
                                         deallocator:^(void *bytes) { (void)bytes; }
                                               error:&error];
}

static NSArray<NSNumber *> *shapeFrom(const int64_t *dims, int rank) {
    NSMutableArray *shape = [NSMutableArray arrayWithCapacity:(NSUInteger)rank];
    for (int i = 0; i < rank; i++) [shape addObject:@(dims[i])];
    return shape;
}

/* A prediction already under way on another thread.
 *
 * Starting one and waiting for it separately is the whole point of having the
 * accelerator: between the two calls the caller can queue GPU work, and the
 * two run at once. */
struct JaiCoreMLTicket {
    void *done;     /* dispatch_semaphore_t, +1 */
    bool  ok;
    char  trouble[512];
};

/* Everything the prediction needs, gathered on the calling thread so that the
 * background one touches nothing that could move. */
typedef struct {
    id<MLFeatureProvider> provider;
    NSArray<NSString *> *names;
    float **destinations;
    size_t *counts;
    int     count;
} JaiCoreMLCall;

static bool prepare(JaiCoreMLModel *model,
                    JaiGpuBuffer **ins, const size_t *inOffsets,
                    const int64_t *inShapes, const int *inRanks,
                    JaiGpuBuffer **outs, const size_t *outOffsets,
                    const int64_t *outShapes, const int *outRanks,
                    JaiCoreMLCall *call, char *errBuf, size_t errSize) {
    NSArray<NSString *> *inNames = (__bridge NSArray *)model->inputs;
    NSArray<NSString *> *outNames = (__bridge NSArray *)model->outputs;

    NSMutableDictionary<NSString *, id> *feed = [NSMutableDictionary new];
    const int64_t *walk = inShapes;
    for (NSUInteger i = 0; i < inNames.count; i++) {
        MLMultiArray *array = wrap(ins[i], inOffsets[i], shapeFrom(walk, inRanks[i]), false);
        if (array == nil) return false;
        feed[inNames[i]] = array;
        walk += inRanks[i];
    }
    NSError *error = nil;
    call->provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:feed error:&error];
    if (call->provider == nil) {
        if (errBuf != NULL && errSize > 0) {
            snprintf(errBuf, errSize, "%s",
                     error != nil ? [[error localizedDescription] UTF8String]
                                  : "the inputs were refused");
        }
        return false;
    }

    call->names = outNames;
    call->count = (int)outNames.count;
    call->destinations = call->count > 0 ? JAI_ALLOC(float *, (size_t)call->count) : NULL;
    call->counts = call->count > 0 ? JAI_ALLOC(size_t, (size_t)call->count) : NULL;
    walk = outShapes;
    for (int i = 0; i < call->count; i++) {
        id<MTLBuffer> target = (__bridge id<MTLBuffer>)jaiGpuBufferHandle(outs[i]);
        if (target == nil) return false;
        size_t wanted = 1;
        for (int d = 0; d < outRanks[i]; d++) wanted *= (size_t)walk[d];
        call->destinations[i] = (float *)[target contents] + outOffsets[i];
        call->counts[i] = wanted;
        walk += outRanks[i];
    }
    /* The results land in memory the GPU also reads, so work already queued
     * against those buffers has to have finished before they are written. */
    for (int i = 0; i < call->count; i++) jaiGpuWaitFor(outs[i]);
    return true;
}

static void releaseCall(JaiCoreMLCall *call) {
    if (call->destinations != NULL) JAI_FREE_ARRAY(float *, call->destinations, (size_t)call->count);
    if (call->counts != NULL) JAI_FREE_ARRAY(size_t, call->counts, (size_t)call->count);
    call->destinations = NULL;
    call->counts = NULL;
}

/* Run the prepared call and copy its results out. */
static bool finish(MLModel *held, JaiCoreMLCall *call, char *errBuf, size_t errSize) {
    NSError *error = nil;
    id<MLFeatureProvider> produced = [held predictionFromFeatures:call->provider error:&error];
    if (produced == nil) {
        if (errBuf != NULL && errSize > 0) {
            snprintf(errBuf, errSize, "%s",
                     error != nil ? [[error localizedDescription] UTF8String]
                                  : "the model did not run");
        }
        return false;
    }
    for (int i = 0; i < call->count; i++) {
        MLFeatureValue *value = [produced featureValueForName:call->names[(NSUInteger)i]];
        if (value == nil || value.multiArrayValue == nil) return false;
        MLMultiArray *array = value.multiArrayValue;
        if ((size_t)array.count != call->counts[i]) return false;
        float *destination = call->destinations[i];
        const size_t bytes = call->counts[i] * sizeof(float);
        [array getBytesWithHandler:^(const void *source, NSInteger length) {
            memcpy(destination, source, (size_t)length < bytes ? (size_t)length : bytes);
        }];
    }
    return true;
}

JaiCoreMLTicket *jaiCoreMLStart(JaiCoreMLModel *model,
                                JaiGpuBuffer **ins, const size_t *inOffsets,
                                const int64_t *inShapes, const int *inRanks,
                                JaiGpuBuffer **outs, const size_t *outOffsets,
                                const int64_t *outShapes, const int *outRanks,
                                char *errBuf, size_t errSize) {
    if (errBuf != NULL && errSize > 0) errBuf[0] = '\0';
    if (model == NULL) return NULL;
    @autoreleasepool {
        JaiCoreMLCall call = {0};
        if (!prepare(model, ins, inOffsets, inShapes, inRanks,
                     outs, outOffsets, outShapes, outRanks, &call, errBuf, errSize)) {
            releaseCall(&call);
            return NULL;
        }
        JaiCoreMLTicket *ticket = JAI_ALLOC(JaiCoreMLTicket, 1);
        ticket->done = (__bridge_retained void *)dispatch_semaphore_create(0);
        ticket->ok = false;
        ticket->trouble[0] = '\0';

        MLModel *held = (__bridge MLModel *)model->model;
        __block JaiCoreMLCall running = call;
        dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
            @autoreleasepool {
                ticket->ok = finish(held, &running, ticket->trouble, sizeof(ticket->trouble));
                releaseCall(&running);
                dispatch_semaphore_signal((__bridge dispatch_semaphore_t)ticket->done);
            }
        });
        return ticket;
    }
}

bool jaiCoreMLWait(JaiCoreMLTicket *ticket, char *errBuf, size_t errSize) {
    if (errBuf != NULL && errSize > 0) errBuf[0] = '\0';
    if (ticket == NULL) return false;
    dispatch_semaphore_wait((__bridge dispatch_semaphore_t)ticket->done, DISPATCH_TIME_FOREVER);
    const bool ok = ticket->ok;
    if (!ok && errBuf != NULL && errSize > 0) snprintf(errBuf, errSize, "%s", ticket->trouble);
    CFBridgingRelease(ticket->done);
    JAI_FREE(JaiCoreMLTicket, ticket);
    return ok;
}

bool jaiCoreMLRun(JaiCoreMLModel *model,
                  JaiGpuBuffer **ins, const size_t *inOffsets,
                  const int64_t *inShapes, const int *inRanks,
                  JaiGpuBuffer **outs, const size_t *outOffsets,
                  const int64_t *outShapes, const int *outRanks,
                  char *errBuf, size_t errSize) {
    if (errBuf != NULL && errSize > 0) errBuf[0] = '\0';
    if (model == NULL) return false;
    @autoreleasepool {
        JaiCoreMLCall call = {0};
        if (!prepare(model, ins, inOffsets, inShapes, inRanks,
                     outs, outOffsets, outShapes, outRanks, &call, errBuf, errSize)) {
            releaseCall(&call);
            return false;
        }
        const bool ok = finish((__bridge MLModel *)model->model, &call, errBuf, errSize);
        releaseCall(&call);
        return ok;
    }
}
