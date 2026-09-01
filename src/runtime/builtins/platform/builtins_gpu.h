/* builtins_gpu.h — what the three builtins_gpu translation units share.
 *
 * __prim__.gpu_* is one registration table, built in builtins_gpu.c, which
 * also keeps the device queries and the buffer lifecycle. The kernels
 * (compile, dispatch and the built-in ones) live in builtins_gpu_kernels.c
 * and the fused MLP training primitives in builtins_gpu_train.c. This header
 * carries the buffer record, the two accessors those files need, and every
 * JaiNativeFn jaiRegisterGpuPrimitives binds from outside builtins_gpu.c.
 * Not a public interface: runtime.h declares what the rest of the VM calls.
 */
#ifndef JAI_BUILTINS_GPU_H
#define JAI_BUILTINS_GPU_H

#include "runtime/runtime.h"

#include "native/native.h"

typedef struct {
    JaiGpuBuffer *buffer;
    int64_t       count;      /* float slots, not bytes */
    int64_t       origin;     /* element offset into `buffer` */
    bool          owned;
} GpuBuffer;

/* --- shared plumbing (defined in builtins_gpu.c) -------------------- */

bool requireGpu(const char *fnName);

bool requireBuffer(Value v, int index, const char *fnName, GpuBuffer **out);

/* --- kernel primitives defined in builtins_gpu_kernels.c ------------ */

bool nGpuCompile(int argc, Value *args, Value *out);
bool nGpuMaxThreadsPerGroup(int argc, Value *args, Value *out);
bool nGpuDispatch(int argc, Value *args, Value *out);
bool nGpuDispatchAsync(int argc, Value *args, Value *out);
bool nGpuFlush(int argc, Value *args, Value *out);
bool nGpuSynchronize(int argc, Value *args, Value *out);
bool nGpuKernelFree(int argc, Value *args, Value *out);
bool nGpuVectorAdd(int argc, Value *args, Value *out);
bool nGpuVectorMul(int argc, Value *args, Value *out);
bool nGpuMatMul(int argc, Value *args, Value *out);
bool nGpuMatMulBuffers(int argc, Value *args, Value *out);
bool nGpuMhaBuffers(int argc, Value *args, Value *out);
bool nGpuConv2dBuffers(int argc, Value *args, Value *out);
bool nGpuConv2dDataGrad(int argc, Value *args, Value *out);
bool nGpuConv2dWeightsGrad(int argc, Value *args, Value *out);
bool nGpuReduceSum(int argc, Value *args, Value *out);

/* --- training primitives defined in builtins_gpu_train.c ------------ */

bool nGpuMlpSgdStep(int argc, Value *args, Value *out);
bool nGpuMlpSgdEpoch(int argc, Value *args, Value *out);
bool nGpuMlpBwdStep(int argc, Value *args, Value *out);
bool nGpuMlp3SgdStep(int argc, Value *args, Value *out);
bool nGpuMlp3SgdEpoch(int argc, Value *args, Value *out);
bool nGpuMlp3BwdStep(int argc, Value *args, Value *out);
bool nGpuLabelsValid(int argc, Value *args, Value *out);

#endif /* JAI_BUILTINS_GPU_H */
