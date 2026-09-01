/* builtins_gpu.c — __prim__.gpu_*, the surface std.gpu is written over.
 * Buffers and kernels are integer handles (not GC objects) for device memory. */

#include "runtime/builtins/platform/builtins_gpu.h"
#include "runtime/runtime.h"
#include "runtime/handles.h"
#include "runtime/parallel.h"
#include <limits.h>
#include <math.h>

#include "native/native.h"
#include "vm/gc.h"

bool requireGpu(const char *fnName) {
    if (jaiGpuAvailable()) return true;
    return jaiThrow(vm.cRuntimeError,
                    "%s(): no GPU device is available on this machine", fnName);
}

bool requireBuffer(Value v, int index, const char *fnName,
                   GpuBuffer **out) {
    if (!requireGpu(fnName)) return false;
    void *ptr;
    if (!jaiHandleGet(v, index, HANDLE_GPU_BUFFER, fnName, &ptr)) return false;
    *out = (GpuBuffer *)ptr;
    return true;
}

/* A buffer handle as the native buffer and the element offset into it, for
 * the graph primitives next door -- they need the same two facts and have no
 * business knowing what a `GpuBuffer` looks like. */
bool jaiGpuBufferOf(Value v, int index, const char *fnName,
                    JaiGpuBuffer **buffer, int64_t *origin) {
    GpuBuffer *held;
    if (!requireBuffer(v, index, fnName, &held)) return false;
    *buffer = held->buffer;
    *origin = held->origin;
    return true;
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

static bool nGpuDeviceCount(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    *out = INT_VAL(jaiGpuDeviceCount());
    return true;
}

static bool nGpuSetDevice(int argc, Value *args, Value *out) {
    (void)argc;
    int64_t index;
    if (!jaiArgInt(args[0], 1, "gpu_set_device", &index)) return false;
    if (index < 0 || index > INT_MAX)
        return jaiThrow(vm.cValueError, "gpu_set_device(): index out of range");
    if (!jaiGpuSetDevice((int)index))
        return jaiThrow(vm.cRuntimeError,
                        "gpu_set_device(): no device at %lld, or the GPU is already in use",
                        (long long)index);
    *out = NULL_VAL;
    return true;
}

static bool nGpuSetMixedPrecision(int argc, Value *args, Value *out) {
    (void)argc;
    bool enabled;
    if (!jaiArgBool(args[0], 1, "gpu_set_mixed_precision", &enabled)) return false;
    jaiGpuSetMixedPrecision(enabled);
    *out = NULL_VAL;
    return true;
}

static bool nGpuMixedPrecision(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    *out = BOOL_VAL(jaiGpuMixedPrecision());
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
    record->origin = 0;
    record->owned = true;

    *out = INT_VAL(jaiHandleAdd(HANDLE_GPU_BUFFER, record));
    return true;
}

static bool nGpuBufferView(int argc, Value *args, Value *out) {
    (void)argc;
    GpuBuffer *parent;
    if (!requireBuffer(args[0], 1, "gpu_buffer_view", &parent)) return false;

    int64_t offset, count;
    if (!jaiArgInt(args[1], 2, "gpu_buffer_view", &offset)) return false;
    if (!jaiArgInt(args[2], 3, "gpu_buffer_view", &count)) return false;
    if (offset < 0 || count <= 0 || offset > parent->count ||
        count > parent->count - offset)
        return jaiThrow(vm.cValueError,
                        "gpu_buffer_view(): %lld floats at %lld exceed the "
                        "buffer's %lld", (long long)count, (long long)offset,
                        (long long)parent->count);

    GpuBuffer *record = JAI_ALLOC_ZEROED(GpuBuffer, 1);
    record->buffer = parent->buffer;
    record->count = count;
    record->origin = parent->origin + offset;
    record->owned = false;
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
        if (!IS_NUMBER(jaiListGet(values, i))) {
            JAI_FREE_ARRAY(float, narrowed, values->count);
            return jaiThrow(vm.cTypeError,
                            "gpu_buffer_upload(): value %d is %s, expected a "
                            "number", i, jaiTypeNameStatic(jaiListGet(values, i)));
        }
        narrowed[i] = (float)jaiAsDouble(jaiListGet(values, i));
    }
    jaiGpuUpload(b->buffer, narrowed, (size_t)values->count * sizeof(float),
                 (size_t)(b->origin + offset) * sizeof(float));
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
                       (size_t)(b->origin + destOffset), (float)scale);
    *out = NULL_VAL;
    return true;
}

/* `gpu_buffer_download_u8(buffer, offset, count, scale)` -- float slots back
 * as clamped bytes.
 *
 * The inverse of `gpu_buffer_upload_u8`, and the cheap way to read pixels: the
 * ordinary download hands back a list, which boxes every element, and a 720p
 * frame is 2.8 million of them. */
static bool nGpuBufferDownloadU8(int argc, Value *args, Value *out) {
    (void)argc;
    GpuBuffer *b;
    if (!requireBuffer(args[0], 1, "gpu_buffer_download_u8", &b)) return false;
    int64_t offset, count;
    double scale;
    if (!jaiArgInt(args[1], 2, "gpu_buffer_download_u8", &offset)) return false;
    if (!jaiArgInt(args[2], 3, "gpu_buffer_download_u8", &count)) return false;
    if (!jaiArgNumber(args[3], 4, "gpu_buffer_download_u8", &scale)) return false;
    if (!isfinite(scale) || scale == 0.0)
        return jaiThrow(vm.cValueError,
                        "gpu_buffer_download_u8(): scale must be finite and non-zero");
    if (offset < 0 || count < 0 || offset > b->count || count > b->count - offset)
        return jaiThrow(vm.cValueError,
                        "gpu_buffer_download_u8(): %lld values at %lld exceed buffer "
                        "capacity %lld",
                        (long long)count, (long long)offset, (long long)b->count);

    ObjBytes *bytes = jaiBytesNew(NULL, (size_t)count);
    if (bytes == NULL) return false;
    if (count > 0)
        jaiGpuDownloadU8(b->buffer, bytes->data, (size_t)count,
                         (size_t)(b->origin + offset), (float)scale);
    *out = OBJ_VAL(bytes);
    return true;
}

/* One pixel, rounded and clamped into the 0xAARRGGBB a window's back buffer
 * holds. Matches `list_pack_argb`, including that a NaN lands on black rather
 * than on an undefined cast. */
static uint32_t packChannelF(float value) {
    if (!(value > 0.0f)) return 0u;
    const float rounded = value + 0.5f;
    return rounded > 255.0f ? 255u : (uint32_t)rounded;
}

typedef struct {
    const float *source;
    Value       *dst;
    float        factor;
    int          channels;
} JaiArgbWork;

static void packArgbRange(void *context, size_t start, size_t end) {
    const JaiArgbWork *work = (const JaiArgbWork *)context;
    const float *source = work->source;
    const float factor = work->factor;
    const int channels = work->channels;
    for (size_t i = start; i < end; i++) {
        const float *pixel = source + i * (size_t)channels;
        uint32_t r, g, b;
        if (channels == 1) {
            r = packChannelF(pixel[0] * factor);
            g = r;
            b = r;
        } else {
            /* Stored blue first, shown red first. */
            b = packChannelF(pixel[0] * factor);
            g = packChannelF(pixel[1] * factor);
            r = packChannelF(pixel[2] * factor);
        }
        work->dst[i] = INT_VAL((int64_t)(0xFF000000u | (r << 16) | (g << 8) | b));
    }
}

#define JAI_ARGB_CHUNK 32768

/* `gpu_buffer_pack_argb(buffer, offset, count, channels, scale)` -- device
 * pixels straight to the one integer a window wants per pixel.
 *
 * The two steps this replaces were a download that narrowed the floats into a
 * byte string and a pass that packed that string into integers: two walks over
 * a 720p frame and a 2.8 MB string in between, for a result that is neither. */
static bool nGpuBufferPackArgb(int argc, Value *args, Value *out) {
    (void)argc;
    GpuBuffer *b;
    if (!requireBuffer(args[0], 1, "gpu_buffer_pack_argb", &b)) return false;
    int64_t offset, count, channels;
    double scale;
    if (!jaiArgInt(args[1], 2, "gpu_buffer_pack_argb", &offset)) return false;
    if (!jaiArgInt(args[2], 3, "gpu_buffer_pack_argb", &count)) return false;
    if (!jaiArgInt(args[3], 4, "gpu_buffer_pack_argb", &channels)) return false;
    if (!jaiArgNumber(args[4], 5, "gpu_buffer_pack_argb", &scale)) return false;
    if (channels != 1 && channels != 3 && channels != 4)
        return jaiThrow(vm.cValueError,
                        "gpu_buffer_pack_argb(): channels must be 1, 3 or 4, got %lld",
                        (long long)channels);
    if (!isfinite(scale) || scale == 0.0)
        return jaiThrow(vm.cValueError,
                        "gpu_buffer_pack_argb(): scale must be finite and non-zero");
    if (offset < 0 || count < 0 || offset > b->count || count > b->count - offset)
        return jaiThrow(vm.cValueError,
                        "gpu_buffer_pack_argb(): %lld values at %lld exceed buffer "
                        "capacity %lld",
                        (long long)count, (long long)offset, (long long)b->count);
    if (count % channels != 0)
        return jaiThrow(vm.cValueError,
                        "gpu_buffer_pack_argb(): %lld values is not a whole number of "
                        "%lld-channel pixels",
                        (long long)count, (long long)channels);

    const int64_t pixels = count / channels;
    ObjList *list = jaiListNew((int)pixels);
    if (list == NULL) return false;
    if (pixels == 0) {
        *out = OBJ_VAL(list);
        return true;
    }
    jaiGCPushRoot(OBJ_VAL(list));
    const bool room = jaiListReserveExact(list, (int)pixels);
    jaiGCPopRoot();
    if (!room) return false;

    const float *source = jaiGpuMapRead(b->buffer, (size_t)(b->origin + offset),
                                        (size_t)count);
    if (source == NULL)
        return jaiThrow(vm.cRuntimeError, "gpu_buffer_pack_argb(): the buffer would not map");

    JaiArgbWork work = {source, list->items, scale != 0.0 ? (float)(1.0 / scale) : 1.0f,
                        (int)channels};
    jaiParallelChunks((size_t)pixels, JAI_ARGB_CHUNK, packArgbRange, &work);
    list->count = (int)pixels;
    list->version++;
    *out = OBJ_VAL(list);
    return true;
}

static bool nGpuBufferFillUniform(int argc, Value *args, Value *out) {
    (void)argc;
    GpuBuffer *b;
    if (!requireBuffer(args[0], 1, "gpu_buffer_fill_uniform", &b)) return false;
    double low, high;
    int64_t seed;
    if (!jaiArgNumber(args[1], 2, "gpu_buffer_fill_uniform", &low)) return false;
    if (!jaiArgNumber(args[2], 3, "gpu_buffer_fill_uniform", &high)) return false;
    if (!jaiArgInt(args[3], 4, "gpu_buffer_fill_uniform", &seed)) return false;
    if (!(low < high))
        return jaiThrow(vm.cValueError,
                        "gpu_buffer_fill_uniform(): low must be less than high");
    jaiGpuFillUniform(b->buffer, (size_t)b->origin, (size_t)b->count,
                      (float)low, (float)high, (uint64_t)seed);
    *out = NULL_VAL;
    return true;
}

static bool nGpuBufferFillZero(int argc, Value *args, Value *out) {
    (void)argc;
    GpuBuffer *b;
    if (!requireBuffer(args[0], 1, "gpu_buffer_fill_zero", &b)) return false;
    jaiGpuFillZero(b->buffer, (size_t)b->origin, (size_t)b->count);
    *out = NULL_VAL;
    return true;
}

/* Below this many elements one thread is quicker than waking others. */
#define JAI_DOWNLOAD_CHUNK 32768

typedef struct {
    const float *raw;
    Value       *items;
} DownloadWork;

/* Widening floats into list elements: three million of them for one frame,
 * each a sixteen-byte store, and nothing shared between the indices. */
static void widenRange(void *context, size_t start, size_t end) {
    const DownloadWork *work = (const DownloadWork *)context;
    for (size_t i = start; i < end; i++)
        work->items[i] = FLOAT_VAL(work->raw[i]);
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

    /* Read where the values already are. Storage is shared, so staging them
     * into an array first would be an allocation and a copy of the whole
     * buffer before the loop below even started. */
    const float *raw = jaiGpuMapRead(b->buffer, (size_t)(b->origin + offset),
                                     (size_t)wanted);
    if (raw == NULL)
        return jaiThrow(vm.cRuntimeError,
                        "gpu_buffer_download(): the buffer could not be read");

    /* Reserved once and written through, rather than pushed a value at a
     * time. The push path re-checks capacity on every element, and a 720p
     * frame is 2.8 million of them. */
    ObjList *list = jaiListNew((int)wanted);
    if (list == NULL || !jaiListReserveExact(list, (int)wanted)) return false;
    DownloadWork work = {raw, list->items};
    jaiParallelChunks((size_t)wanted, JAI_DOWNLOAD_CHUNK, widenRange, &work);
    list->count = (int)wanted;
    list->version++;

    *out = OBJ_VAL(list);
    return true;
}

static bool nGpuBufferFree(int argc, Value *args, Value *out) {
    (void)argc;
    GpuBuffer *b;
    if (!requireBuffer(args[0], 1, "gpu_buffer_free", &b)) return false;

    jaiHandleRelease(AS_INT(args[0]));
    if (b->owned) jaiGpuFree(b->buffer);
    JAI_FREE(GpuBuffer, b);

    *out = NULL_VAL;
    return true;
}

void jaiRegisterGpuPrimitives(void) {
    jaiDefineNative("__prim__.gpu_available",   nGpuAvailable,   0, 0);
    jaiDefineNative("__prim__.gpu_device_name", nGpuDeviceName,  0, 0);
    jaiDefineNative("__prim__.gpu_device_count", nGpuDeviceCount, 0, 0);
    jaiDefineNative("__prim__.gpu_set_device", nGpuSetDevice, 1, 1);
    jaiDefineNative("__prim__.gpu_set_mixed_precision", nGpuSetMixedPrecision, 1, 1);
    jaiDefineNative("__prim__.gpu_mixed_precision", nGpuMixedPrecision, 0, 0);

    jaiDefineNative("__prim__.gpu_buffer_new",      nGpuBufferNew,      1, 1);
    jaiDefineNative("__prim__.gpu_buffer_view",     nGpuBufferView,     3, 3);
    jaiDefineNative("__prim__.gpu_buffer_upload",   nGpuBufferUpload,   3, 3);
    jaiDefineNative("__prim__.gpu_buffer_download_u8", nGpuBufferDownloadU8, 4, 4);
    jaiDefineNative("__prim__.gpu_buffer_pack_argb", nGpuBufferPackArgb, 5, 5);
    jaiDefineNative("__prim__.gpu_buffer_upload_u8", nGpuBufferUploadU8, 6, 6);
    jaiDefineNative("__prim__.gpu_buffer_fill_uniform", nGpuBufferFillUniform, 4, 4);
    jaiDefineNative("__prim__.gpu_buffer_fill_zero", nGpuBufferFillZero, 1, 1);
    jaiDefineNative("__prim__.gpu_buffer_download", nGpuBufferDownload, 3, 3);
    jaiDefineNative("__prim__.gpu_buffer_free",     nGpuBufferFree,     1, 1);

    jaiDefineNative("__prim__.gpu_compile",               nGpuCompile,            2, 2);
    jaiDefineNative("__prim__.gpu_max_threads_per_group", nGpuMaxThreadsPerGroup, 1, 1);
    jaiDefineNative("__prim__.gpu_dispatch",              nGpuDispatch,           5, 5);
    jaiDefineNative("__prim__.gpu_dispatch_async",        nGpuDispatchAsync,      5, 5);
    jaiDefineNative("__prim__.gpu_flush",                 nGpuFlush,              0, 0);
    jaiDefineNative("__prim__.gpu_synchronize",           nGpuSynchronize,        0, 0);
    jaiDefineNative("__prim__.gpu_kernel_free",           nGpuKernelFree,         1, 1);

    jaiDefineNative("__prim__.gpu_vector_add", nGpuVectorAdd, 2, 2);
    jaiDefineNative("__prim__.gpu_vector_mul", nGpuVectorMul, 2, 2);
    jaiDefineNative("__prim__.gpu_matmul",     nGpuMatMul,    5, 5);
    jaiDefineNative("__prim__.gpu_matmul_buffers", nGpuMatMulBuffers, 9, 9);
    jaiDefineNative("__prim__.gpu_mha_buffers", nGpuMhaBuffers, 8, 8);
    jaiDefineNative("__prim__.gpu_conv2d_buffers", nGpuConv2dBuffers, 16, 17);
    jaiDefineNative("__prim__.gpu_conv2d_data_grad", nGpuConv2dDataGrad, 14, 14);
    jaiDefineNative("__prim__.gpu_conv2d_weights_grad", nGpuConv2dWeightsGrad, 14, 14);
    jaiDefineNative("__prim__.gpu_mlp_sgd_step", nGpuMlpSgdStep, 13, 13);
    jaiDefineNative("__prim__.gpu_mlp_sgd_epoch", nGpuMlpSgdEpoch, 15, 15);
    jaiDefineNative("__prim__.gpu_mlp_bwd_step", nGpuMlpBwdStep, 16, 16);
    jaiDefineNative("__prim__.gpu_mlp3_sgd_step", nGpuMlp3SgdStep, 19, 19);
    jaiDefineNative("__prim__.gpu_mlp3_sgd_epoch", nGpuMlp3SgdEpoch, 21, 21);
    jaiDefineNative("__prim__.gpu_mlp3_bwd_step", nGpuMlp3BwdStep, 26, 26);
    jaiDefineNative("__prim__.gpu_labels_valid", nGpuLabelsValid, 3, 3);
    jaiDefineNative("__prim__.gpu_reduce_sum", nGpuReduceSum, 1, 1);
}
