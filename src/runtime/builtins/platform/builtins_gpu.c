/* builtins_gpu.c — __prim__.gpu_*, the surface std.gpu is written over.
 * Buffers and kernels are integer handles (not GC objects) for device memory. */

#include "runtime/runtime.h"
#include "runtime/handles.h"
#include <math.h>

#include "native/native.h"
#include "vm/gc.h"

/* Metal's diagnostic listing for a failed compile: file, line, caret per error. */
#define GPU_ERROR_BUFFER 4096

typedef struct {
    JaiGpuBuffer *buffer;
    int64_t       count;      /* float slots, not bytes */
} GpuBuffer;

static bool requireGpu(const char *fnName) {
    if (jaiGpuAvailable()) return true;
    return jaiThrow(vm.cRuntimeError,
                    "%s(): no GPU device is available on this machine", fnName);
}

static bool requireBuffer(Value v, int index, const char *fnName,
                          GpuBuffer **out) {
    if (!requireGpu(fnName)) return false;
    void *ptr;
    if (!jaiHandleGet(v, index, HANDLE_GPU_BUFFER, fnName, &ptr)) return false;
    *out = (GpuBuffer *)ptr;
    return true;
}

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
        if (!IS_NUMBER(list->items[i])) {
            JAI_FREE_ARRAY(double, values, list->count);
            (void)jaiThrow(vm.cTypeError,
                           "%s() argument %d: element %d is %s, expected a number",
                           fnName, index, i, jaiTypeNameStatic(list->items[i]));
            return NULL;
        }
        values[i] = jaiAsDouble(list->items[i]);
    }
    return values;
}

static ObjList *listOfDoubles(const double *values, int64_t count) {
    ObjList *list = jaiListNew((int)count);
    jaiGCPushRoot(OBJ_VAL(list));
    for (int64_t i = 0; i < count; i++) jaiListPush(list, FLOAT_VAL(values[i]));
    jaiGCPopRoot();
    return list;
}

static bool nGpuAvailable(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    *out = BOOL_VAL(jaiGpuAvailable());
    return true;
}

static bool nGpuDeviceName(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    const char *name = jaiGpuDeviceName();
    *out = OBJ_VAL(jaiStringInternC(name != NULL ? name : "none"));
    return true;
}

static bool nGpuBufferNew(int argc, Value *args, Value *out) {
    (void)argc;
    if (!requireGpu("gpu_buffer_new")) return false;

    int64_t count;
    if (!jaiArgInt(args[0], 1, "gpu_buffer_new", &count)) return false;
    if (count <= 0)
        return jaiThrow(vm.cValueError,
                        "gpu_buffer_new(): count must be positive, got %lld",
                        (long long)count);
    if (count > (int64_t)(SIZE_MAX / sizeof(float)))
        return jaiThrow(vm.cValueError,
                        "gpu_buffer_new(): %lld floats is more than this machine "
                        "can address", (long long)count);

    JaiGpuBuffer *buffer = jaiGpuAlloc((size_t)count * sizeof(float));
    if (buffer == NULL)
        return jaiThrow(vm.cRuntimeError,
                        "gpu_buffer_new(): the device refused an allocation of "
                        "%lld floats", (long long)count);

    GpuBuffer *record = JAI_ALLOC_ZEROED(GpuBuffer, 1);
    record->buffer = buffer;
    record->count = count;

    *out = INT_VAL(jaiHandleAdd(HANDLE_GPU_BUFFER, record));
    return true;
}

static bool nGpuBufferUpload(int argc, Value *args, Value *out) {
    (void)argc;
    GpuBuffer *b;
    if (!requireBuffer(args[0], 1, "gpu_buffer_upload", &b)) return false;

    ObjList *values;
    int64_t offset;
    if (!jaiArgList(args[1], 2, "gpu_buffer_upload", &values)) return false;
    if (!jaiArgInt(args[2], 3, "gpu_buffer_upload", &offset)) return false;

    if (offset < 0 || offset > b->count || values->count > b->count - offset)
        return jaiThrow(vm.cValueError,
                        "gpu_buffer_upload(): %d values at %lld exceed the "
                        "buffer's %lld", values->count, (long long)offset,
                        (long long)b->count);
    if (values->count == 0) {
        *out = NULL_VAL;
        return true;
    }

    float *narrowed = JAI_ALLOC(float, values->count);
    for (int i = 0; i < values->count; i++) {
        if (!IS_NUMBER(values->items[i])) {
            JAI_FREE_ARRAY(float, narrowed, values->count);
            return jaiThrow(vm.cTypeError,
                            "gpu_buffer_upload(): value %d is %s, expected a "
                            "number", i, jaiTypeNameStatic(values->items[i]));
        }
        narrowed[i] = (float)jaiAsDouble(values->items[i]);
    }
    jaiGpuUpload(b->buffer, narrowed, (size_t)values->count * sizeof(float),
                 (size_t)offset * sizeof(float));
    JAI_FREE_ARRAY(float, narrowed, values->count);

    *out = NULL_VAL;
    return true;
}

static bool nGpuBufferUploadU8(int argc, Value *args, Value *out) {
    (void)argc;
    GpuBuffer *b;
    if (!requireBuffer(args[0], 1, "gpu_buffer_upload_u8", &b)) return false;
    if (!IS_BYTES(args[1]))
        return jaiThrow(vm.cTypeError,
                        "gpu_buffer_upload_u8() argument 2 is %s, expected bytes",
                        jaiTypeNameStatic(args[1]));

    ObjBytes *bytes = AS_BYTES(args[1]);
    int64_t sourceOffset, count, destOffset;
    double scale;
    if (!jaiArgInt(args[2], 3, "gpu_buffer_upload_u8", &sourceOffset)) return false;
    if (!jaiArgInt(args[3], 4, "gpu_buffer_upload_u8", &count)) return false;
    if (!jaiArgInt(args[4], 5, "gpu_buffer_upload_u8", &destOffset)) return false;
    if (!jaiArgNumber(args[5], 6, "gpu_buffer_upload_u8", &scale)) return false;
    if (!isfinite(scale))
        return jaiThrow(vm.cValueError,
                        "gpu_buffer_upload_u8(): scale must be finite");

    if (sourceOffset < 0 || count < 0 || sourceOffset > bytes->length ||
        count > (int64_t)bytes->length - sourceOffset)
        return jaiThrow(vm.cValueError,
                        "gpu_buffer_upload_u8(): %lld bytes at %lld exceed source length %u",
                        (long long)count, (long long)sourceOffset, bytes->length);
    if (destOffset < 0 || destOffset > b->count || count > b->count - destOffset)
        return jaiThrow(vm.cValueError,
                        "gpu_buffer_upload_u8(): %lld values at %lld exceed buffer capacity %lld",
                        (long long)count, (long long)destOffset, (long long)b->count);
    if (count > 0)
        jaiGpuUploadU8(b->buffer, bytes->data + sourceOffset, (size_t)count,
                       (size_t)destOffset, (float)scale);
    *out = NULL_VAL;
    return true;
}

static bool nGpuBufferDownload(int argc, Value *args, Value *out) {
    (void)argc;
    GpuBuffer *b;
    if (!requireBuffer(args[0], 1, "gpu_buffer_download", &b)) return false;

    int64_t offset, wanted;
    if (!jaiArgInt(args[1], 2, "gpu_buffer_download", &offset)) return false;
    if (!jaiArgInt(args[2], 3, "gpu_buffer_download", &wanted)) return false;

    if (offset < 0 || wanted < 0 || offset > b->count ||
        wanted > b->count - offset)
        return jaiThrow(vm.cValueError,
                        "gpu_buffer_download(): %lld values at %lld exceed the "
                        "buffer's %lld", (long long)wanted, (long long)offset,
                        (long long)b->count);
    if (wanted == 0) {
        *out = OBJ_VAL(jaiListNew(0));
        return true;
    }

    float *raw = JAI_ALLOC(float, wanted);
    jaiGpuDownload(b->buffer, raw, (size_t)wanted * sizeof(float),
                   (size_t)offset * sizeof(float));

    ObjList *list = jaiListNew((int)wanted);
    jaiGCPushRoot(OBJ_VAL(list));
    for (int64_t i = 0; i < wanted; i++) jaiListPush(list, FLOAT_VAL(raw[i]));
    jaiGCPopRoot();
    JAI_FREE_ARRAY(float, raw, wanted);

    *out = OBJ_VAL(list);
    return true;
}

static bool nGpuBufferFree(int argc, Value *args, Value *out) {
    (void)argc;
    GpuBuffer *b;
    if (!requireBuffer(args[0], 1, "gpu_buffer_free", &b)) return false;

    jaiHandleRelease(AS_INT(args[0]));
    jaiGpuFree(b->buffer);
    JAI_FREE(GpuBuffer, b);

    *out = NULL_VAL;
    return true;
}

static bool nGpuCompile(int argc, Value *args, Value *out) {
    (void)argc;
    if (!requireGpu("gpu_compile")) return false;

    ObjString *source, *entry;
    if (!jaiArgString(args[0], 1, "gpu_compile", &source)) return false;
    if (!jaiArgString(args[1], 2, "gpu_compile", &entry)) return false;

    char errors[GPU_ERROR_BUFFER];
    JaiGpuKernel *kernel = jaiGpuCompile(source->chars, entry->chars, errors,
                                         sizeof errors);
    if (kernel == NULL)
        return jaiThrow(vm.cValueError, "gpu_compile(): %s",
                        errors[0] != '\0' ? errors : "the kernel did not build");

    *out = INT_VAL(jaiHandleAdd(HANDLE_GPU_KERNEL, kernel));
    return true;
}

static bool nGpuMaxThreadsPerGroup(int argc, Value *args, Value *out) {
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
    if (handles->count > 0) buffers = JAI_ALLOC(JaiGpuBuffer *, handles->count);
    for (int i = 0; i < handles->count; i++) {
        void *ptr;
        if (!jaiHandleGet(handles->items[i], 2, HANDLE_GPU_BUFFER, name, &ptr)) {
            if (buffers != NULL)
                JAI_FREE_ARRAY(JaiGpuBuffer *, buffers, handles->count);
            return false;
        }
        buffers[i] = ((GpuBuffer *)ptr)->buffer;
    }

    uint32_t *scalars = NULL;
    if (scalarList->count > 0) scalars = JAI_ALLOC(uint32_t, scalarList->count);
    for (int i = 0; i < scalarList->count; i++) {
        int64_t scalar;
        if (!jaiArgInt(scalarList->items[i], 3, name, &scalar) ||
            scalar < 0 || scalar > UINT32_MAX) {
            if (buffers != NULL)
                JAI_FREE_ARRAY(JaiGpuBuffer *, buffers, handles->count);
            JAI_FREE_ARRAY(uint32_t, scalars, scalarList->count);
            if (vm.hasException) return false;
            return jaiThrow(vm.cValueError,
                            "%s(): scalar %d does not fit in a uint", name, i);
        }
        scalars[i] = (uint32_t)scalar;
    }

    bool ok = async
        ? jaiGpuDispatchAsync(kernel, buffers, handles->count, scalars,
                              scalarList->count, (int)threads, (int)groupSize)
        : jaiGpuDispatch(kernel, buffers, handles->count, scalars,
                         scalarList->count, (int)threads, (int)groupSize);
    if (buffers != NULL) JAI_FREE_ARRAY(JaiGpuBuffer *, buffers, handles->count);
    if (scalars != NULL) JAI_FREE_ARRAY(uint32_t, scalars, scalarList->count);

    if (!ok)
        return jaiThrow(vm.cRuntimeError,
                        "%s(): the device did not accept or complete the kernel", name);

    *out = NULL_VAL;
    return true;
}

static bool nGpuDispatch(int argc, Value *args, Value *out) {
    (void)argc;
    return dispatchKernel(args, out, false);
}

static bool nGpuDispatchAsync(int argc, Value *args, Value *out) {
    (void)argc;
    return dispatchKernel(args, out, true);
}

static bool nGpuSynchronize(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    if (!requireGpu("gpu_synchronize")) return false;
    if (!jaiGpuSynchronize())
        return jaiThrow(vm.cRuntimeError,
                        "gpu_synchronize(): queued GPU work did not complete");
    *out = NULL_VAL;
    return true;
}

static bool nGpuKernelFree(int argc, Value *args, Value *out) {
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

static bool nGpuVectorAdd(int argc, Value *args, Value *out) {
    (void)argc;
    return elementwise(args, out, "gpu_vector_add", jaiGpuVectorAdd);
}

static bool nGpuVectorMul(int argc, Value *args, Value *out) {
    (void)argc;
    return elementwise(args, out, "gpu_vector_mul", jaiGpuVectorMul);
}

static bool nGpuMatMul(int argc, Value *args, Value *out) {
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

static bool nGpuReduceSum(int argc, Value *args, Value *out) {
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

void jaiRegisterGpuPrimitives(void) {
    jaiDefineNative("__prim__.gpu_available",   nGpuAvailable,   0, 0);
    jaiDefineNative("__prim__.gpu_device_name", nGpuDeviceName,  0, 0);

    jaiDefineNative("__prim__.gpu_buffer_new",      nGpuBufferNew,      1, 1);
    jaiDefineNative("__prim__.gpu_buffer_upload",   nGpuBufferUpload,   3, 3);
    jaiDefineNative("__prim__.gpu_buffer_upload_u8", nGpuBufferUploadU8, 6, 6);
    jaiDefineNative("__prim__.gpu_buffer_download", nGpuBufferDownload, 3, 3);
    jaiDefineNative("__prim__.gpu_buffer_free",     nGpuBufferFree,     1, 1);

    jaiDefineNative("__prim__.gpu_compile",               nGpuCompile,            2, 2);
    jaiDefineNative("__prim__.gpu_max_threads_per_group", nGpuMaxThreadsPerGroup, 1, 1);
    jaiDefineNative("__prim__.gpu_dispatch",              nGpuDispatch,           5, 5);
    jaiDefineNative("__prim__.gpu_dispatch_async",        nGpuDispatchAsync,      5, 5);
    jaiDefineNative("__prim__.gpu_synchronize",           nGpuSynchronize,        0, 0);
    jaiDefineNative("__prim__.gpu_kernel_free",           nGpuKernelFree,         1, 1);

    jaiDefineNative("__prim__.gpu_vector_add", nGpuVectorAdd, 2, 2);
    jaiDefineNative("__prim__.gpu_vector_mul", nGpuVectorMul, 2, 2);
    jaiDefineNative("__prim__.gpu_matmul",     nGpuMatMul,    5, 5);
    jaiDefineNative("__prim__.gpu_reduce_sum", nGpuReduceSum, 1, 1);
}
