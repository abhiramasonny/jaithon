#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "parallel.h"
#import <dispatch/dispatch.h>


static id<MTLDevice> metalDevice = nil;
static id<MTLCommandQueue> commandQueue = nil;
static id<MTLLibrary> library = nil;
static bool metalInitialized = false;
static bool metalAvailable = false;


static NSString* shaderSource = @R"(
#include <metal_stdlib>
using namespace metal;

kernel void reduceSum(device const float* input [[buffer(0)]],
                      device float* partialSums [[buffer(1)]],
                      constant uint& count [[buffer(2)]],
                      uint gid [[thread_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint blockDim [[threads_per_threadgroup]],
                      uint blockIdx [[threadgroup_position_in_grid]]) {
    
    threadgroup float shared[256];
    
    float sum = 0.0f;
    uint idx = gid;
    uint gridSize = blockDim * 64; // Assume max 64 threadgroups
    
    // Grid-stride loop for large arrays
    while (idx < count) {
        sum += input[idx];
        idx += gridSize;
    }
    
    shared[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction in shared memory
    for (uint s = blockDim / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < blockDim) {
            shared[tid] += shared[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write block result
    if (tid == 0) {
        partialSums[blockIdx] = shared[0];
    }
}

// Parallel product reduction
kernel void reduceProduct(device const float* input [[buffer(0)]],
                          device float* partialProds [[buffer(1)]],
                          constant uint& count [[buffer(2)]],
                          uint gid [[thread_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint blockDim [[threads_per_threadgroup]],
                          uint blockIdx [[threadgroup_position_in_grid]]) {
    
    threadgroup float shared[256];
    
    float prod = 1.0f;
    uint idx = gid;
    uint gridSize = blockDim * 64;
    
    while (idx < count) {
        prod *= input[idx];
        idx += gridSize;
    }
    
    shared[tid] = prod;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = blockDim / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < blockDim) {
            shared[tid] *= shared[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        partialProds[blockIdx] = shared[0];
    }
}

// Parallel min reduction
kernel void reduceMin(device const float* input [[buffer(0)]],
                      device float* partialMins [[buffer(1)]],
                      constant uint& count [[buffer(2)]],
                      uint gid [[thread_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint blockDim [[threads_per_threadgroup]],
                      uint blockIdx [[threadgroup_position_in_grid]]) {
    
    threadgroup float shared[256];
    
    float minVal = INFINITY;
    uint idx = gid;
    uint gridSize = blockDim * 64;
    
    while (idx < count) {
        minVal = min(minVal, input[idx]);
        idx += gridSize;
    }
    
    shared[tid] = minVal;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = blockDim / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < blockDim) {
            shared[tid] = min(shared[tid], shared[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        partialMins[blockIdx] = shared[0];
    }
}

// Parallel max reduction
kernel void reduceMax(device const float* input [[buffer(0)]],
                      device float* partialMaxs [[buffer(1)]],
                      constant uint& count [[buffer(2)]],
                      uint gid [[thread_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint blockDim [[threads_per_threadgroup]],
                      uint blockIdx [[threadgroup_position_in_grid]]) {
    
    threadgroup float shared[256];
    
    float maxVal = -INFINITY;
    uint idx = gid;
    uint gridSize = blockDim * 64;
    
    while (idx < count) {
        maxVal = max(maxVal, input[idx]);
        idx += gridSize;
    }
    
    shared[tid] = maxVal;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = blockDim / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < blockDim) {
            shared[tid] = max(shared[tid], shared[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        partialMaxs[blockIdx] = shared[0];
    }
}

// ==========================================================================
// Map Kernels (Element-wise operations)
// ==========================================================================

kernel void mapAdd(device const float* a [[buffer(0)]],
                   device const float* b [[buffer(1)]],
                   device float* result [[buffer(2)]],
                   constant uint& count [[buffer(3)]],
                   uint gid [[thread_position_in_grid]]) {
    if (gid < count) {
        result[gid] = a[gid] + b[gid];
    }
}

kernel void mapSub(device const float* a [[buffer(0)]],
                   device const float* b [[buffer(1)]],
                   device float* result [[buffer(2)]],
                   constant uint& count [[buffer(3)]],
                   uint gid [[thread_position_in_grid]]) {
    if (gid < count) {
        result[gid] = a[gid] - b[gid];
    }
}

kernel void mapMul(device const float* a [[buffer(0)]],
                   device const float* b [[buffer(1)]],
                   device float* result [[buffer(2)]],
                   constant uint& count [[buffer(3)]],
                   uint gid [[thread_position_in_grid]]) {
    if (gid < count) {
        result[gid] = a[gid] * b[gid];
    }
}

kernel void mapDiv(device const float* a [[buffer(0)]],
                   device const float* b [[buffer(1)]],
                   device float* result [[buffer(2)]],
                   constant uint& count [[buffer(3)]],
                   uint gid [[thread_position_in_grid]]) {
    if (gid < count) {
        result[gid] = a[gid] / b[gid];
    }
}

kernel void mapScale(device const float* a [[buffer(0)]],
                     constant float& scalar [[buffer(1)]],
                     device float* result [[buffer(2)]],
                     constant uint& count [[buffer(3)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid < count) {
        result[gid] = a[gid] * scalar;
    }
}

kernel void mapSqrt(device const float* a [[buffer(0)]],
                    device float* result [[buffer(1)]],
                    constant uint& count [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid < count) {
        result[gid] = sqrt(a[gid]);
    }
}

kernel void mapAbs(device const float* a [[buffer(0)]],
                   device float* result [[buffer(1)]],
                   constant uint& count [[buffer(2)]],
                   uint gid [[thread_position_in_grid]]) {
    if (gid < count) {
        result[gid] = abs(a[gid]);
    }
}

kernel void mapSin(device const float* a [[buffer(0)]],
                   device float* result [[buffer(1)]],
                   constant uint& count [[buffer(2)]],
                   uint gid [[thread_position_in_grid]]) {
    if (gid < count) {
        result[gid] = sin(a[gid]);
    }
}

kernel void mapCos(device const float* a [[buffer(0)]],
                   device float* result [[buffer(1)]],
                   constant uint& count [[buffer(2)]],
                   uint gid [[thread_position_in_grid]]) {
    if (gid < count) {
        result[gid] = cos(a[gid]);
    }
}

kernel void mapExp(device const float* a [[buffer(0)]],
                   device float* result [[buffer(1)]],
                   constant uint& count [[buffer(2)]],
                   uint gid [[thread_position_in_grid]]) {
    if (gid < count) {
        result[gid] = exp(a[gid]);
    }
}

kernel void mapLog(device const float* a [[buffer(0)]],
                   device float* result [[buffer(1)]],
                   constant uint& count [[buffer(2)]],
                   uint gid [[thread_position_in_grid]]) {
    if (gid < count) {
        result[gid] = log(a[gid]);
    }
}

// ==========================================================================
// Loop Pattern Kernels
// ==========================================================================

// Generic loop with modulo operation: sum += (i % divisor)
kernel void loopModuloSum(device float* output [[buffer(0)]],
                          constant uint& start [[buffer(1)]],
                          constant uint& divisor [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
    output[gid] = float((start + gid) % divisor);
}

// Generic arithmetic progression sum: sum += (start + i * step)
kernel void loopArithSum(device float* output [[buffer(0)]],
                         constant float& start [[buffer(1)]],
                         constant float& step [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    output[gid] = start + float(gid) * step;
}

// Dot product kernel
kernel void dotProduct(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* partialSums [[buffer(2)]],
                       constant uint& count [[buffer(3)]],
                       uint gid [[thread_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]],
                       uint blockDim [[threads_per_threadgroup]],
                       uint blockIdx [[threadgroup_position_in_grid]]) {
    
    threadgroup float shared[256];
    
    float sum = 0.0f;
    uint idx = gid;
    uint gridSize = blockDim * 64;
    
    while (idx < count) {
        sum += a[idx] * b[idx];
        idx += gridSize;
    }
    
    shared[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = blockDim / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < blockDim) {
            shared[tid] += shared[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        partialSums[blockIdx] = shared[0];
    }
}
)";


static void initMetal(void) {
    if (metalInitialized) return;
    metalInitialized = true;
    
    @autoreleasepool {
        
        metalDevice = MTLCreateSystemDefaultDevice();
        if (!metalDevice) {
            if (runtime.debug) {
                NSLog(@"[GPU] Metal not available, using CPU fallback");
            }
            return;
        }
        
        
        commandQueue = [metalDevice newCommandQueue];
        if (!commandQueue) {
            if (runtime.debug) {
                NSLog(@"[GPU] Failed to create Metal command queue");
            }
            metalDevice = nil;
            return;
        }
        
        
        NSError* error = nil;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.fastMathEnabled = YES;
        
        library = [metalDevice newLibraryWithSource:shaderSource 
                                            options:options 
                                              error:&error];
        if (!library) {
            if (runtime.debug) {
                NSLog(@"[GPU] Failed to compile Metal shaders: %@", error);
            }
            commandQueue = nil;
            metalDevice = nil;
            return;
        }
        
        metalAvailable = true;
        
        if (runtime.debug) {
            NSLog(@"[GPU] Metal initialized on %@", metalDevice.name);
        }
    }
}

bool gpuIsAvailable(void) {
    if (!metalInitialized) {
        initMetal();
    }
    return metalAvailable && parallelConfig.enableGPU;
}

bool gpuCanHandleWorkload(const AnalysisResult* analysis) {
    if (!gpuIsAvailable()) return false;
    if (!analysis->canUseGPU) return false;
    if (analysis->estimatedIterations < parallelConfig.minGPUWork) return false;
    
    
    
    
    
    return analysis->canParallelize && 
           !analysis->hasOrderDependentEffects;
}


Value gpuExecuteReduction(const double* data, int n, char op) {
    if (!gpuIsAvailable() || n < parallelConfig.minGPUWork) {
        
        return makeNumber(arraySum(data, n));
    }
    
    @autoreleasepool {
        NSError* error = nil;
        
        
        NSString* kernelName;
        switch (op) {
            case '+': kernelName = @"reduceSum"; break;
            case '*': kernelName = @"reduceProduct"; break;
            case '<': kernelName = @"reduceMin"; break;
            case '>': kernelName = @"reduceMax"; break;
            default:
                return makeNumber(arraySum(data, n));
        }
        
        id<MTLFunction> reduceFunc = [library newFunctionWithName:kernelName];
        if (!reduceFunc) {
            return makeNumber(arraySum(data, n));
        }
        
        id<MTLComputePipelineState> pipeline = 
            [metalDevice newComputePipelineStateWithFunction:reduceFunc error:&error];
        if (!pipeline) {
            return makeNumber(arraySum(data, n));
        }
        
        
        float* floatData = (float*)malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) {
            floatData[i] = (float)data[i];
        }
        
        
        NSUInteger threadGroupSize = MIN(256, pipeline.maxTotalThreadsPerThreadgroup);
        NSUInteger numGroups = (n + threadGroupSize - 1) / threadGroupSize;
        if (numGroups > 64) numGroups = 64;
        
        
        id<MTLBuffer> inputBuffer = [metalDevice newBufferWithBytes:floatData
                                                             length:n * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> partialBuffer = [metalDevice newBufferWithLength:numGroups * sizeof(float)
                                                               options:MTLResourceStorageModeShared];
        
        uint32_t count = (uint32_t)n;
        id<MTLBuffer> countBuffer = [metalDevice newBufferWithBytes:&count
                                                             length:sizeof(uint32_t)
                                                            options:MTLResourceStorageModeShared];
        
        
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:partialBuffer offset:0 atIndex:1];
        [encoder setBuffer:countBuffer offset:0 atIndex:2];
        
        MTLSize gridSize = MTLSizeMake(numGroups * threadGroupSize, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        
        float* partials = (float*)[partialBuffer contents];
        double result = 0;
        
        switch (op) {
            case '+':
                for (NSUInteger i = 0; i < numGroups; i++) result += partials[i];
                break;
            case '*':
                result = 1;
                for (NSUInteger i = 0; i < numGroups; i++) result *= partials[i];
                break;
            case '<':
                result = partials[0];
                for (NSUInteger i = 1; i < numGroups; i++) {
                    if (partials[i] < result) result = partials[i];
                }
                break;
            case '>':
                result = partials[0];
                for (NSUInteger i = 1; i < numGroups; i++) {
                    if (partials[i] > result) result = partials[i];
                }
                break;
        }
        
        free(floatData);
        
        return makeNumber(result);
    }
}


Value gpuExecuteMap(const double* input, int n, const char* operation) {
    if (!gpuIsAvailable() || n < parallelConfig.minGPUWork) {
        return makeNull();
    }
    
    @autoreleasepool {
        NSError* error = nil;
        
        
        NSString* kernelName = nil;
        if (strcmp(operation, "sqrt") == 0) kernelName = @"mapSqrt";
        else if (strcmp(operation, "abs") == 0) kernelName = @"mapAbs";
        else if (strcmp(operation, "sin") == 0) kernelName = @"mapSin";
        else if (strcmp(operation, "cos") == 0) kernelName = @"mapCos";
        else if (strcmp(operation, "exp") == 0) kernelName = @"mapExp";
        else if (strcmp(operation, "log") == 0) kernelName = @"mapLog";
        else return makeNull();
        
        id<MTLFunction> mapFunc = [library newFunctionWithName:kernelName];
        if (!mapFunc) return makeNull();
        
        id<MTLComputePipelineState> pipeline = 
            [metalDevice newComputePipelineStateWithFunction:mapFunc error:&error];
        if (!pipeline) return makeNull();
        
        
        float* floatInput = (float*)malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) {
            floatInput[i] = (float)input[i];
        }
        
        
        id<MTLBuffer> inputBuffer = [metalDevice newBufferWithBytes:floatInput
                                                             length:n * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> outputBuffer = [metalDevice newBufferWithLength:n * sizeof(float)
                                                              options:MTLResourceStorageModeShared];
        
        uint32_t count = (uint32_t)n;
        id<MTLBuffer> countBuffer = [metalDevice newBufferWithBytes:&count
                                                             length:sizeof(uint32_t)
                                                            options:MTLResourceStorageModeShared];
        
        
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:outputBuffer offset:0 atIndex:1];
        [encoder setBuffer:countBuffer offset:0 atIndex:2];
        
        NSUInteger threadGroupSize = MIN(256, pipeline.maxTotalThreadsPerThreadgroup);
        MTLSize gridSize = MTLSizeMake(n, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        
        Value result = makeArray(n);
        float* outputData = (float*)[outputBuffer contents];
        for (int i = 0; i < n; i++) {
            arrayPush(result.as.array, makeNumber(outputData[i]));
        }
        
        free(floatInput);
        
        return result;
    }
}


Value gpuExecuteLoop(int64_t start, int64_t end, const char* bodySrc,
                    const AnalysisResult* analysis) {
    if (!gpuIsAvailable()) {
        return makeNull();
    }
    
    int64_t count = end - start;
    if (count < parallelConfig.minGPUWork) {
        return makeNull();
    }
    
    @autoreleasepool {
        NSError* error = nil;
        
        
        
        if (analysis->hasReduction && analysis->reductionOp == '+' &&
            strstr(bodySrc, "%")) {
            
            
            const char* modPos = strstr(bodySrc, "%");
            if (!modPos) return makeNull();
            
            modPos++;
            while (*modPos && isspace(*modPos)) modPos++;
            
            int divisor = atoi(modPos);
            if (divisor <= 0) return makeNull();
            
            id<MTLFunction> loopFunc = [library newFunctionWithName:@"loopModuloSum"];
            if (!loopFunc) return makeNull();
            
            id<MTLComputePipelineState> pipeline = 
                [metalDevice newComputePipelineStateWithFunction:loopFunc error:&error];
            if (!pipeline) return makeNull();
            
            
            id<MTLBuffer> outputBuffer = [metalDevice newBufferWithLength:count * sizeof(float)
                                                                  options:MTLResourceStorageModeShared];
            
            uint32_t startVal = (uint32_t)start;
            uint32_t divisorVal = (uint32_t)divisor;
            
            id<MTLBuffer> startBuffer = [metalDevice newBufferWithBytes:&startVal
                                                                 length:sizeof(uint32_t)
                                                                options:MTLResourceStorageModeShared];
            id<MTLBuffer> divisorBuffer = [metalDevice newBufferWithBytes:&divisorVal
                                                                   length:sizeof(uint32_t)
                                                                  options:MTLResourceStorageModeShared];
            
            
            id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:outputBuffer offset:0 atIndex:0];
            [encoder setBuffer:startBuffer offset:0 atIndex:1];
            [encoder setBuffer:divisorBuffer offset:0 atIndex:2];
            
            NSUInteger threadGroupSize = MIN(256, pipeline.maxTotalThreadsPerThreadgroup);
            MTLSize gridSize = MTLSizeMake(count, 1, 1);
            MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
            
            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [encoder endEncoding];
            
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            
            
            float* results = (float*)[outputBuffer contents];
            
            
            id<MTLFunction> reduceFunc = [library newFunctionWithName:@"reduceSum"];
            id<MTLComputePipelineState> reducePipeline = 
                [metalDevice newComputePipelineStateWithFunction:reduceFunc error:&error];
            
            if (reducePipeline) {
                NSUInteger numGroups = (count + threadGroupSize - 1) / threadGroupSize;
                if (numGroups > 64) numGroups = 64;
                
                id<MTLBuffer> partialBuffer = [metalDevice newBufferWithLength:numGroups * sizeof(float)
                                                                       options:MTLResourceStorageModeShared];
                
                uint32_t countVal = (uint32_t)count;
                id<MTLBuffer> countBuffer = [metalDevice newBufferWithBytes:&countVal
                                                                     length:sizeof(uint32_t)
                                                                    options:MTLResourceStorageModeShared];
                
                commandBuffer = [commandQueue commandBuffer];
                encoder = [commandBuffer computeCommandEncoder];
                
                [encoder setComputePipelineState:reducePipeline];
                [encoder setBuffer:outputBuffer offset:0 atIndex:0];
                [encoder setBuffer:partialBuffer offset:0 atIndex:1];
                [encoder setBuffer:countBuffer offset:0 atIndex:2];
                
                MTLSize reduceGrid = MTLSizeMake(numGroups * threadGroupSize, 1, 1);
                [encoder dispatchThreads:reduceGrid threadsPerThreadgroup:threadgroupSize];
                [encoder endEncoding];
                
                [commandBuffer commit];
                [commandBuffer waitUntilCompleted];
                
                
                float* partials = (float*)[partialBuffer contents];
                double sum = 0;
                for (NSUInteger i = 0; i < numGroups; i++) {
                    sum += partials[i];
                }
                
                return makeNumber(sum);
            }
            
            
            double sum = 0;
            for (int64_t i = 0; i < count; i++) {
                sum += results[i];
            }
            
            return makeNumber(sum);
        }
        
        
        return makeNull();
    }
}
