/* builtins_gpu_kernels.c — compiling and dispatching Metal kernels, and the
 * built-in kernels that take host lists or buffer handles. */

#include "runtime/builtins/platform/builtins_gpu.h"
#include "runtime/handles.h"

#include "native/native.h"
#include "vm/gc.h"

/* Metal's diagnostic listing for a failed compile: file, line, caret per error. */
#define GPU_ERROR_BUFFER 4096

static bool requireKernel(Value v, int index, const char *fnName,
                          JaiGpuKernel **out) {
    if (!requireGpu(fnName)) return false;
    void *ptr;
    if (!jaiHandleGet(v, index, HANDLE_GPU_KERNEL, fnName, &ptr)) return false;
    *out = (JaiGpuKernel *)ptr;
    return true;
}

static double *numbersOf(ObjList *list, const char *fnName, int index) {
    if (list->count == 0) return NULL;
    double *values = JAI_ALLOC(double, list->count);
    for (int i = 0; i < list->count; i++) {
        if (!IS_NUMBER(jaiListGet(list, i))) {
            JAI_FREE_ARRAY(double, values, list->count);
            (void)jaiThrow(vm.cTypeError,
                           "%s() argument %d: element %d is %s, expected a number",
                           fnName, index, i, jaiTypeNameStatic(jaiListGet(list, i)));
            return NULL;
        }
        values[i] = jaiAsDouble(jaiListGet(list, i));
    }
    return values;
}

static ObjList *listOfDoubles(const double *values, int64_t count) {
    ObjList *list = jaiListNew((int)count);
    if (list == NULL || !jaiListReserveExact(list, (int)count)) return list;
    for (int64_t i = 0; i < count; i++) jaiListPut(list, i, FLOAT_VAL(values[i]));
    list->count = (int)count;
    list->version++;
    return list;
}

bool nGpuCompile(int argc, Value *args, Value *out) {
    (void)argc;
    if (!requireGpu("gpu_compile")) return false;

    ObjString *source, *entry;
    if (!jaiArgString(args[0], 1, "gpu_compile", &source)) return false;
    if (!jaiArgString(args[1], 2, "gpu_compile", &entry)) return false;

    /* Through jaiStringCStr, not `chars`: the Metal compiler reads until a NUL,
     * and a string somebody has since appended to is a VIEW of a buffer whose
     * next byte is no longer one. Concatenation writes past the view it was
     * handed and marks the older view unterminated, which is sound for every
     * Jaithon-level use and wrong for exactly this one -- jaitensor caches its
     * generated Metal source, and appending to that cached string made the very
     * next compile fail on the bytes that had been appended after it. */
    char errors[GPU_ERROR_BUFFER];
    /* Both rooted, and rooted BEFORE the second is made: a terminated copy is
     * reachable from nothing, so making the entry point's copy could collect
     * the source's. */
    ObjString *sourceText = jaiStringTerminated(source);
    if (sourceText == NULL) return false;
    jaiGCPushRoot(OBJ_VAL((Obj *)sourceText));
    ObjString *entryText = jaiStringTerminated(entry);
    if (entryText == NULL) { jaiGCPopRoot(); return false; }
    jaiGCPushRoot(OBJ_VAL((Obj *)entryText));
    JaiGpuKernel *kernel = jaiGpuCompile(sourceText->chars, entryText->chars,
                                         errors, sizeof errors);
    jaiGCPopRoots(2);
    if (kernel == NULL)
        return jaiThrow(vm.cValueError, "gpu_compile(): %s",
                        errors[0] != '\0' ? errors : "the kernel did not build");

    *out = INT_VAL(jaiHandleAdd(HANDLE_GPU_KERNEL, kernel));
    return true;
}

bool nGpuMaxThreadsPerGroup(int argc, Value *args, Value *out) {
    (void)argc;
    JaiGpuKernel *kernel;
    if (!requireKernel(args[0], 1, "gpu_max_threads_per_group", &kernel))
        return false;
    *out = INT_VAL(jaiGpuMaxThreadsPerGroup(kernel));
    return true;
}

static bool dispatchKernel(Value *args, Value *out, bool async) {
    const char *name = async ? "gpu_dispatch_async" : "gpu_dispatch";
    JaiGpuKernel *kernel;
    if (!requireKernel(args[0], 1, name, &kernel)) return false;

    ObjList *handles, *scalarList;
    int64_t threads, groupSize;
    if (!jaiArgList(args[1], 2, name, &handles)) return false;
    if (!jaiArgList(args[2], 3, name, &scalarList)) return false;
    if (!jaiArgInt(args[3], 4, name, &threads)) return false;
    if (!jaiArgInt(args[4], 5, name, &groupSize)) return false;

    if (threads <= 0 || threads > INT32_MAX)
        return jaiThrow(vm.cValueError,
                        "%s(): threads must be positive, got %lld", name,
                        (long long)threads);
    if (groupSize < 0 || groupSize > INT32_MAX)
        return jaiThrow(vm.cValueError,
                        "%s(): group_size must be non-negative, got %lld", name,
                        (long long)groupSize);

    JaiGpuBuffer **buffers = NULL;
    size_t *offsets = NULL;
    if (handles->count > 0) {
        buffers = JAI_ALLOC(JaiGpuBuffer *, handles->count);
        offsets = JAI_ALLOC(size_t, handles->count);
    }
    for (int i = 0; i < handles->count; i++) {
        void *ptr;
        if (!jaiHandleGet(jaiListGet(handles, i), 2, HANDLE_GPU_BUFFER, name, &ptr)) {
            if (buffers != NULL)
                JAI_FREE_ARRAY(JaiGpuBuffer *, buffers, handles->count);
            if (offsets != NULL)
                JAI_FREE_ARRAY(size_t, offsets, handles->count);
            return false;
        }
        GpuBuffer *record = (GpuBuffer *)ptr;
        buffers[i] = record->buffer;
        offsets[i] = (size_t)record->origin * sizeof(float);
    }

    uint32_t *scalars = NULL;
    if (scalarList->count > 0) scalars = JAI_ALLOC(uint32_t, scalarList->count);
    for (int i = 0; i < scalarList->count; i++) {
        int64_t scalar;
        if (!jaiArgInt(jaiListGet(scalarList, i), 3, name, &scalar) ||
            scalar < 0 || scalar > UINT32_MAX) {
            if (buffers != NULL)
                JAI_FREE_ARRAY(JaiGpuBuffer *, buffers, handles->count);
            if (offsets != NULL)
                JAI_FREE_ARRAY(size_t, offsets, handles->count);
            JAI_FREE_ARRAY(uint32_t, scalars, scalarList->count);
            if (vm.hasException) return false;
            return jaiThrow(vm.cValueError,
                            "%s(): scalar %d does not fit in a uint", name, i);
        }
        scalars[i] = (uint32_t)scalar;
    }

    bool ok = async
        ? jaiGpuDispatchAsync(kernel, buffers, handles->count, scalars,
                              scalarList->count, (int)threads, (int)groupSize,
                              offsets)
        : jaiGpuDispatch(kernel, buffers, handles->count, scalars,
                         scalarList->count, (int)threads, (int)groupSize,
                         offsets);
    if (buffers != NULL) JAI_FREE_ARRAY(JaiGpuBuffer *, buffers, handles->count);
    if (offsets != NULL) JAI_FREE_ARRAY(size_t, offsets, handles->count);
    if (scalars != NULL) JAI_FREE_ARRAY(uint32_t, scalars, scalarList->count);

    if (!ok)
        return jaiThrow(vm.cRuntimeError,
                        "%s(): the device did not accept or complete the kernel", name);

    *out = NULL_VAL;
    return true;
}

bool nGpuDispatch(int argc, Value *args, Value *out) {
    (void)argc;
    return dispatchKernel(args, out, false);
}

bool nGpuDispatchAsync(int argc, Value *args, Value *out) {
    (void)argc;
    return dispatchKernel(args, out, true);
}

bool nGpuFlush(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    if (!requireGpu("gpu_flush")) return false;
    if (!jaiGpuFlush())
        return jaiThrow(vm.cRuntimeError,
                        "gpu_flush(): queued GPU work did not complete");
    *out = NULL_VAL;
    return true;
}

bool nGpuSynchronize(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    if (!requireGpu("gpu_synchronize")) return false;
    if (!jaiGpuSynchronize())
        return jaiThrow(vm.cRuntimeError,
                        "gpu_synchronize(): queued GPU work did not complete");
    *out = NULL_VAL;
    return true;
}

bool nGpuKernelFree(int argc, Value *args, Value *out) {
    (void)argc;
    JaiGpuKernel *kernel;
    if (!requireKernel(args[0], 1, "gpu_kernel_free", &kernel)) return false;

    jaiHandleRelease(AS_INT(args[0]));
    jaiGpuKernelFree(kernel);

    *out = NULL_VAL;
    return true;
}

/* The built-in kernels below need no device: the native layer runs them on
 * the GPU when there is one and in compensated scalar arithmetic when there
 * is not, so they answer identically everywhere. */
static bool elementwise(Value *args, Value *out, const char *fnName,
                        bool (*run)(const double *, const double *, double *,
                                    size_t)) {
    ObjList *a, *b;
    if (!jaiArgList(args[0], 1, fnName, &a)) return false;
    if (!jaiArgList(args[1], 2, fnName, &b)) return false;
    if (a->count != b->count)
        return jaiThrow(vm.cValueError,
                        "%s(): operands are %d and %d long", fnName, a->count,
                        b->count);
    if (a->count == 0) {
        *out = OBJ_VAL(jaiListNew(0));
        return true;
    }

    double *left = numbersOf(a, fnName, 1);
    if (left == NULL) return false;
    double *right = numbersOf(b, fnName, 2);
    if (right == NULL) {
        JAI_FREE_ARRAY(double, left, a->count);
        return false;
    }

    double *result = JAI_ALLOC(double, a->count);
    bool ok = run(left, right, result, (size_t)a->count);
    ObjList *list = ok ? listOfDoubles(result, a->count) : NULL;

    JAI_FREE_ARRAY(double, left, a->count);
    JAI_FREE_ARRAY(double, right, b->count);
    JAI_FREE_ARRAY(double, result, a->count);

    if (!ok) return jaiThrow(vm.cRuntimeError, "%s(): the kernel failed", fnName);
    *out = OBJ_VAL(list);
    return true;
}

bool nGpuVectorAdd(int argc, Value *args, Value *out) {
    (void)argc;
    return elementwise(args, out, "gpu_vector_add", jaiGpuVectorAdd);
}

bool nGpuVectorMul(int argc, Value *args, Value *out) {
    (void)argc;
    return elementwise(args, out, "gpu_vector_mul", jaiGpuVectorMul);
}

bool nGpuMatMul(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *a, *b;
    int64_t m, k, n;
    if (!jaiArgList(args[0], 1, "gpu_matmul", &a)) return false;
    if (!jaiArgList(args[1], 2, "gpu_matmul", &b)) return false;
    if (!jaiArgInt(args[2], 3, "gpu_matmul", &m)) return false;
    if (!jaiArgInt(args[3], 4, "gpu_matmul", &k)) return false;
    if (!jaiArgInt(args[4], 5, "gpu_matmul", &n)) return false;

    if (m <= 0 || k <= 0 || n <= 0)
        return jaiThrow(vm.cValueError,
                        "gpu_matmul(): dimensions must be positive, got %lldx%lld "
                        "by %lldx%lld", (long long)m, (long long)k, (long long)k,
                        (long long)n);
    if ((int64_t)a->count != m * k || (int64_t)b->count != k * n)
        return jaiThrow(vm.cValueError,
                        "gpu_matmul(): a %dx1 and b %dx1 do not hold %lldx%lld "
                        "and %lldx%lld", a->count, b->count, (long long)m,
                        (long long)k, (long long)k, (long long)n);

    double *left = numbersOf(a, "gpu_matmul", 1);
    if (left == NULL) return false;
    double *right = numbersOf(b, "gpu_matmul", 2);
    if (right == NULL) {
        JAI_FREE_ARRAY(double, left, a->count);
        return false;
    }

    double *result = JAI_ALLOC(double, m * n);
    bool ok = jaiGpuMatMul(left, right, result, (size_t)m, (size_t)k, (size_t)n);
    ObjList *list = ok ? listOfDoubles(result, m * n) : NULL;

    JAI_FREE_ARRAY(double, left, a->count);
    JAI_FREE_ARRAY(double, right, b->count);
    JAI_FREE_ARRAY(double, result, m * n);

    if (!ok) return jaiThrow(vm.cRuntimeError, "gpu_matmul(): the kernel failed");
    *out = OBJ_VAL(list);
    return true;
}

bool nGpuMatMulBuffers(int argc, Value *args, Value *out) {
    (void)argc;
    GpuBuffer *a, *b, *result;
    if (!requireBuffer(args[0], 1, "gpu_matmul_buffers", &a)) return false;
    if (!requireBuffer(args[1], 2, "gpu_matmul_buffers", &b)) return false;
    if (!requireBuffer(args[2], 3, "gpu_matmul_buffers", &result)) return false;

    int64_t m, k, n;
    bool transA, transB, useHalf;
    if (!jaiArgInt(args[3], 4, "gpu_matmul_buffers", &m)) return false;
    if (!jaiArgInt(args[4], 5, "gpu_matmul_buffers", &k)) return false;
    if (!jaiArgInt(args[5], 6, "gpu_matmul_buffers", &n)) return false;
    if (!jaiArgBool(args[6], 7, "gpu_matmul_buffers", &transA)) return false;
    if (!jaiArgBool(args[7], 8, "gpu_matmul_buffers", &transB)) return false;
    if (!jaiArgBool(args[8], 9, "gpu_matmul_buffers", &useHalf)) return false;
    if (m < 0 || k < 0 || n < 0 || m > UINT32_MAX || k > UINT32_MAX || n > UINT32_MAX)
        return jaiThrow(vm.cValueError,
                        "gpu_matmul_buffers(): dimensions must fit in uint32, got "
                        "%lldx%lldx%lld", (long long)m, (long long)k, (long long)n);

    bool ok = jaiGpuMatMulBuffers(
        a->buffer, (size_t)a->origin * sizeof(float),
        b->buffer, (size_t)b->origin * sizeof(float),
        result->buffer, (size_t)result->origin * sizeof(float),
        (uint32_t)m, (uint32_t)k, (uint32_t)n, transA, transB, useHalf);
    if (!ok)
        return jaiThrow(vm.cRuntimeError, "gpu_matmul_buffers(): the kernel failed");
    *out = NULL_VAL;
    return true;
}

bool nGpuMhaBuffers(int argc, Value *args, Value *out) {
    (void)argc;
    GpuBuffer *q, *k, *v, *result;
    if (!requireBuffer(args[0], 1, "gpu_mha_buffers", &q)) return false;
    if (!requireBuffer(args[1], 2, "gpu_mha_buffers", &k)) return false;
    if (!requireBuffer(args[2], 3, "gpu_mha_buffers", &v)) return false;
    if (!requireBuffer(args[3], 4, "gpu_mha_buffers", &result)) return false;

    int64_t seq, heads, hd;
    double scale;
    if (!jaiArgInt(args[4], 5, "gpu_mha_buffers", &seq)) return false;
    if (!jaiArgInt(args[5], 6, "gpu_mha_buffers", &heads)) return false;
    if (!jaiArgInt(args[6], 7, "gpu_mha_buffers", &hd)) return false;
    if (!jaiArgNumber(args[7], 8, "gpu_mha_buffers", &scale)) return false;
    if (seq <= 0 || heads <= 0 || hd <= 0 ||
        seq > UINT32_MAX || heads > UINT32_MAX || hd > UINT32_MAX)
        return jaiThrow(vm.cValueError,
                        "gpu_mha_buffers(): dimensions must be positive uint32, "
                        "got %lldx%lldx%lld",
                        (long long)seq, (long long)heads, (long long)hd);
    if (!(scale > 0.0) || scale > 1e6)
        return jaiThrow(vm.cValueError, "gpu_mha_buffers(): scale is invalid");

    bool ok = jaiGpuMhaPacked(
        q->buffer, (size_t)q->origin * sizeof(float),
        k->buffer, (size_t)k->origin * sizeof(float),
        v->buffer, (size_t)v->origin * sizeof(float),
        result->buffer, (size_t)result->origin * sizeof(float),
        (uint32_t)seq, (uint32_t)heads, (uint32_t)hd, (float)scale);
    *out = BOOL_VAL(ok);
    return true;
}

bool nGpuConv2dBuffers(int argc, Value *args, Value *out) {
    GpuBuffer *input, *weights, *result, *bias = NULL;
    if (!requireBuffer(args[0], 1, "gpu_conv2d_buffers", &input)) return false;
    if (!requireBuffer(args[1], 2, "gpu_conv2d_buffers", &weights)) return false;
    if (!IS_NULL(args[2])) {
        if (!requireBuffer(args[2], 3, "gpu_conv2d_buffers", &bias)) return false;
    }
    if (!requireBuffer(args[3], 4, "gpu_conv2d_buffers", &result)) return false;

    /* Thirteen when the caller picks a data layout, twelve when it takes the
     * NHWC/HWIO default -- every caller that predates the NCHW path. */
    int64_t dims[13];
    dims[12] = 0;
    const int given = argc - 4;
    for (int i = 0; i < given; i++) {
        if (!jaiArgInt(args[4 + i], 5 + i, "gpu_conv2d_buffers", &dims[i])) return false;
        if (dims[i] < 0 || dims[i] > UINT32_MAX)
            return jaiThrow(vm.cValueError,
                            "gpu_conv2d_buffers(): dimension %d must fit in uint32",
                            i);
    }

    bool ok = jaiGpuConv2dBuffers(
        input->buffer, (size_t)input->origin * sizeof(float),
        weights->buffer, (size_t)weights->origin * sizeof(float),
        bias != NULL ? bias->buffer : NULL,
        bias != NULL ? (size_t)bias->origin * sizeof(float) : 0,
        result->buffer, (size_t)result->origin * sizeof(float),
        (uint32_t)dims[0], (uint32_t)dims[1], (uint32_t)dims[2], (uint32_t)dims[3],
        (uint32_t)dims[4], (uint32_t)dims[5], (uint32_t)dims[6],
        (uint32_t)dims[7], (uint32_t)dims[8], (uint32_t)dims[9], (uint32_t)dims[10],
        (uint32_t)dims[11], (uint32_t)dims[12]);
    if (!ok)
        return jaiThrow(vm.cRuntimeError, "gpu_conv2d_buffers(): the kernel failed");
    *out = NULL_VAL;
    return true;
}

/* Shared by both convolution gradients: they take the same eleven dimensions
 * and differ only in which buffer is the second input and which primitive
 * runs, so one argument decoder covers them. */
static bool convGradBuiltin(Value *args, Value *out, const char *name, bool weightsGrad) {
    GpuBuffer *grad, *other, *result;
    if (!requireBuffer(args[0], 1, name, &grad)) return false;
    if (!requireBuffer(args[1], 2, name, &other)) return false;
    if (!requireBuffer(args[2], 3, name, &result)) return false;

    int64_t dims[11];
    for (int i = 0; i < 11; i++) {
        if (!jaiArgInt(args[3 + i], 4 + i, name, &dims[i])) return false;
        if (dims[i] < 0 || dims[i] > UINT32_MAX)
            return jaiThrow(vm.cValueError, "%s(): dimension %d must fit in uint32",
                            name, i);
    }

    bool ok = (weightsGrad ? jaiGpuConv2dWeightsGradBuffers
                           : jaiGpuConv2dDataGradBuffers)(
        grad->buffer, (size_t)grad->origin * sizeof(float),
        other->buffer, (size_t)other->origin * sizeof(float),
        result->buffer, (size_t)result->origin * sizeof(float),
        (uint32_t)dims[0], (uint32_t)dims[1], (uint32_t)dims[2], (uint32_t)dims[3],
        (uint32_t)dims[4], (uint32_t)dims[5], (uint32_t)dims[6],
        (uint32_t)dims[7], (uint32_t)dims[8], (uint32_t)dims[9], (uint32_t)dims[10]);
    *out = BOOL_VAL(ok);
    return true;
}

bool nGpuConv2dDataGrad(int argc, Value *args, Value *out) {
    (void)argc;
    return convGradBuiltin(args, out, "gpu_conv2d_data_grad", false);
}

bool nGpuConv2dWeightsGrad(int argc, Value *args, Value *out) {
    (void)argc;
    return convGradBuiltin(args, out, "gpu_conv2d_weights_grad", true);
}

bool nGpuReduceSum(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *values;
    if (!jaiArgList(args[0], 1, "gpu_reduce_sum", &values)) return false;
    if (values->count == 0) {
        *out = FLOAT_VAL(0.0);
        return true;
    }

    double *raw = numbersOf(values, "gpu_reduce_sum", 1);
    if (raw == NULL) return false;

    double total = 0.0;
    bool ok = jaiGpuReduceSum(raw, (size_t)values->count, &total);
    JAI_FREE_ARRAY(double, raw, values->count);

    if (!ok)
        return jaiThrow(vm.cRuntimeError, "gpu_reduce_sum(): the kernel failed");
    *out = FLOAT_VAL(total);
    return true;
}
