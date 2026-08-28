/* builtins_gpu.c — __prim__.gpu_*, the surface std.gpu is written over.
 * Buffers and kernels are integer handles (not GC objects) for device memory. */

#include "runtime/runtime.h"
#include "runtime/handles.h"
#include "runtime/parallel.h"
#include <limits.h>
#include <math.h>

#include "native/native.h"
#include "vm/gc.h"

/* Metal's diagnostic listing for a failed compile: file, line, caret per error. */
#define GPU_ERROR_BUFFER 4096

typedef struct {
    JaiGpuBuffer *buffer;
    int64_t       count;      /* float slots, not bytes */
    int64_t       origin;     /* element offset into `buffer` */
    bool          owned;
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

static bool nGpuCompile(int argc, Value *args, Value *out) {
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

static bool nGpuDispatch(int argc, Value *args, Value *out) {
    (void)argc;
    return dispatchKernel(args, out, false);
}

static bool nGpuDispatchAsync(int argc, Value *args, Value *out) {
    (void)argc;
    return dispatchKernel(args, out, true);
}

static bool nGpuFlush(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    if (!requireGpu("gpu_flush")) return false;
    if (!jaiGpuFlush())
        return jaiThrow(vm.cRuntimeError,
                        "gpu_flush(): queued GPU work did not complete");
    *out = NULL_VAL;
    return true;
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

static bool nGpuMatMulBuffers(int argc, Value *args, Value *out) {
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

static bool nGpuMhaBuffers(int argc, Value *args, Value *out) {
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

static bool nGpuConv2dBuffers(int argc, Value *args, Value *out) {
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

static bool nGpuConv2dDataGrad(int argc, Value *args, Value *out) {
    (void)argc;
    return convGradBuiltin(args, out, "gpu_conv2d_data_grad", false);
}

static bool nGpuConv2dWeightsGrad(int argc, Value *args, Value *out) {
    (void)argc;
    return convGradBuiltin(args, out, "gpu_conv2d_weights_grad", true);
}

static bool nGpuMlpSgdStep(int argc, Value *args, Value *out) {
    (void)argc;
    GpuBuffer *x, *w1, *b1, *w2, *b2, *labels, *lossAcc, *correctAcc;
    if (!requireBuffer(args[0], 1, "gpu_mlp_sgd_step", &x)) return false;
    if (!requireBuffer(args[1], 2, "gpu_mlp_sgd_step", &w1)) return false;
    if (!requireBuffer(args[2], 3, "gpu_mlp_sgd_step", &b1)) return false;
    if (!requireBuffer(args[3], 4, "gpu_mlp_sgd_step", &w2)) return false;
    if (!requireBuffer(args[4], 5, "gpu_mlp_sgd_step", &b2)) return false;
    if (!requireBuffer(args[5], 6, "gpu_mlp_sgd_step", &labels)) return false;
    if (!requireBuffer(args[6], 7, "gpu_mlp_sgd_step", &lossAcc)) return false;
    correctAcc = NULL;
    if (!IS_NULL(args[7])) {
        if (!requireBuffer(args[7], 8, "gpu_mlp_sgd_step", &correctAcc)) return false;
    }

    int64_t batch, inputs, hidden, classes;
    double lr;
    if (!jaiArgInt(args[8], 9, "gpu_mlp_sgd_step", &batch)) return false;
    if (!jaiArgInt(args[9], 10, "gpu_mlp_sgd_step", &inputs)) return false;
    if (!jaiArgInt(args[10], 11, "gpu_mlp_sgd_step", &hidden)) return false;
    if (!jaiArgInt(args[11], 12, "gpu_mlp_sgd_step", &classes)) return false;
    if (!jaiArgNumber(args[12], 13, "gpu_mlp_sgd_step", &lr)) return false;
    if (batch <= 0 || inputs <= 0 || hidden <= 0 || classes <= 0 ||
        batch > UINT32_MAX || inputs > UINT32_MAX || hidden > UINT32_MAX ||
        classes > UINT32_MAX)
        return jaiThrow(vm.cValueError,
                        "gpu_mlp_sgd_step(): dimensions must be positive uint32, "
                        "got %lldx%lldx%lldx%lld",
                        (long long)batch, (long long)inputs, (long long)hidden,
                        (long long)classes);
    if (!(lr > 0.0) || lr > 1e6)
        return jaiThrow(vm.cValueError, "gpu_mlp_sgd_step(): learning rate is invalid");

    bool ok = jaiGpuMlpSgdStep(
        x->buffer, (size_t)x->origin * sizeof(float),
        w1->buffer, (size_t)w1->origin * sizeof(float),
        b1->buffer, (size_t)b1->origin * sizeof(float),
        w2->buffer, (size_t)w2->origin * sizeof(float),
        b2->buffer, (size_t)b2->origin * sizeof(float),
        labels->buffer, (size_t)labels->origin * sizeof(float),
        lossAcc->buffer, (size_t)lossAcc->origin * sizeof(float),
        correctAcc != NULL ? correctAcc->buffer : NULL,
        correctAcc != NULL ? (size_t)correctAcc->origin * sizeof(float) : 0,
        (uint32_t)batch, (uint32_t)inputs, (uint32_t)hidden, (uint32_t)classes,
        (float)lr);
    if (!ok)
        return jaiThrow(vm.cRuntimeError, "gpu_mlp_sgd_step(): the kernel failed");
    *out = NULL_VAL;
    return true;
}

static bool nGpuMlpSgdEpoch(int argc, Value *args, Value *out) {
    (void)argc;
    GpuBuffer *x, *w1, *b1, *w2, *b2, *labels, *lossAcc, *correctAcc;
    if (!requireBuffer(args[0], 1, "gpu_mlp_sgd_epoch", &x)) return false;
    if (!requireBuffer(args[1], 2, "gpu_mlp_sgd_epoch", &w1)) return false;
    if (!requireBuffer(args[2], 3, "gpu_mlp_sgd_epoch", &b1)) return false;
    if (!requireBuffer(args[3], 4, "gpu_mlp_sgd_epoch", &w2)) return false;
    if (!requireBuffer(args[4], 5, "gpu_mlp_sgd_epoch", &b2)) return false;
    if (!requireBuffer(args[5], 6, "gpu_mlp_sgd_epoch", &labels)) return false;
    if (!requireBuffer(args[6], 7, "gpu_mlp_sgd_epoch", &lossAcc)) return false;
    correctAcc = NULL;
    if (!IS_NULL(args[7])) {
        if (!requireBuffer(args[7], 8, "gpu_mlp_sgd_epoch", &correctAcc)) return false;
    }
    int64_t samples, batch, inputs, hidden, classes, flushEvery;
    double lr;
    if (!jaiArgInt(args[8], 9, "gpu_mlp_sgd_epoch", &samples)) return false;
    if (!jaiArgInt(args[9], 10, "gpu_mlp_sgd_epoch", &batch)) return false;
    if (!jaiArgInt(args[10], 11, "gpu_mlp_sgd_epoch", &inputs)) return false;
    if (!jaiArgInt(args[11], 12, "gpu_mlp_sgd_epoch", &hidden)) return false;
    if (!jaiArgInt(args[12], 13, "gpu_mlp_sgd_epoch", &classes)) return false;
    if (!jaiArgNumber(args[13], 14, "gpu_mlp_sgd_epoch", &lr)) return false;
    if (!jaiArgInt(args[14], 15, "gpu_mlp_sgd_epoch", &flushEvery)) return false;
    if (samples <= 0 || batch <= 0 || inputs <= 0 || hidden <= 0 || classes <= 0 ||
        flushEvery <= 0 || samples > UINT32_MAX || batch > UINT32_MAX ||
        inputs > UINT32_MAX || hidden > UINT32_MAX || classes > UINT32_MAX ||
        flushEvery > UINT32_MAX)
        return jaiThrow(vm.cValueError, "gpu_mlp_sgd_epoch(): dimensions are invalid");
    if (!(lr > 0.0) || lr > 1e6)
        return jaiThrow(vm.cValueError, "gpu_mlp_sgd_epoch(): learning rate is invalid");
    uint32_t processed = 0;
    bool ok = jaiGpuMlpSgdEpoch(
        x->buffer, (size_t)x->origin * sizeof(float),
        w1->buffer, (size_t)w1->origin * sizeof(float),
        b1->buffer, (size_t)b1->origin * sizeof(float),
        w2->buffer, (size_t)w2->origin * sizeof(float),
        b2->buffer, (size_t)b2->origin * sizeof(float),
        labels->buffer, (size_t)labels->origin * sizeof(float),
        lossAcc->buffer, (size_t)lossAcc->origin * sizeof(float),
        correctAcc != NULL ? correctAcc->buffer : NULL,
        correctAcc != NULL ? (size_t)correctAcc->origin * sizeof(float) : 0,
        (uint32_t)samples, (uint32_t)batch, (uint32_t)inputs, (uint32_t)hidden,
        (uint32_t)classes, (float)lr, (uint32_t)flushEvery, &processed);
    if (!ok)
        return jaiThrow(vm.cRuntimeError, "gpu_mlp_sgd_epoch(): the kernel failed");
    *out = INT_VAL((int64_t)processed);
    return true;
}

static bool nGpuMlpBwdStep(int argc, Value *args, Value *out) {
    (void)argc;
    GpuBuffer *x, *w1, *b1, *w2, *b2, *labels, *gW1, *gB1, *gW2, *gB2, *lossAcc, *correctAcc;
    if (!requireBuffer(args[0], 1, "gpu_mlp_bwd_step", &x)) return false;
    if (!requireBuffer(args[1], 2, "gpu_mlp_bwd_step", &w1)) return false;
    if (!requireBuffer(args[2], 3, "gpu_mlp_bwd_step", &b1)) return false;
    if (!requireBuffer(args[3], 4, "gpu_mlp_bwd_step", &w2)) return false;
    if (!requireBuffer(args[4], 5, "gpu_mlp_bwd_step", &b2)) return false;
    if (!requireBuffer(args[5], 6, "gpu_mlp_bwd_step", &labels)) return false;
    if (!requireBuffer(args[6], 7, "gpu_mlp_bwd_step", &gW1)) return false;
    if (!requireBuffer(args[7], 8, "gpu_mlp_bwd_step", &gB1)) return false;
    if (!requireBuffer(args[8], 9, "gpu_mlp_bwd_step", &gW2)) return false;
    if (!requireBuffer(args[9], 10, "gpu_mlp_bwd_step", &gB2)) return false;
    if (!requireBuffer(args[10], 11, "gpu_mlp_bwd_step", &lossAcc)) return false;
    correctAcc = NULL;
    if (!IS_NULL(args[11])) {
        if (!requireBuffer(args[11], 12, "gpu_mlp_bwd_step", &correctAcc)) return false;
    }

    int64_t batch, inputs, hidden, classes;
    if (!jaiArgInt(args[12], 13, "gpu_mlp_bwd_step", &batch)) return false;
    if (!jaiArgInt(args[13], 14, "gpu_mlp_bwd_step", &inputs)) return false;
    if (!jaiArgInt(args[14], 15, "gpu_mlp_bwd_step", &hidden)) return false;
    if (!jaiArgInt(args[15], 16, "gpu_mlp_bwd_step", &classes)) return false;
    if (batch <= 0 || inputs <= 0 || hidden <= 0 || classes <= 0 ||
        batch > UINT32_MAX || inputs > UINT32_MAX || hidden > UINT32_MAX ||
        classes > UINT32_MAX)
        return jaiThrow(vm.cValueError,
                        "gpu_mlp_bwd_step(): dimensions must be positive uint32, "
                        "got %lldx%lldx%lldx%lld",
                        (long long)batch, (long long)inputs, (long long)hidden,
                        (long long)classes);

    bool ok = jaiGpuMlpBwdStep(
        x->buffer, (size_t)x->origin * sizeof(float),
        w1->buffer, (size_t)w1->origin * sizeof(float),
        b1->buffer, (size_t)b1->origin * sizeof(float),
        w2->buffer, (size_t)w2->origin * sizeof(float),
        b2->buffer, (size_t)b2->origin * sizeof(float),
        labels->buffer, (size_t)labels->origin * sizeof(float),
        gW1->buffer, (size_t)gW1->origin * sizeof(float),
        gB1->buffer, (size_t)gB1->origin * sizeof(float),
        gW2->buffer, (size_t)gW2->origin * sizeof(float),
        gB2->buffer, (size_t)gB2->origin * sizeof(float),
        lossAcc->buffer, (size_t)lossAcc->origin * sizeof(float),
        correctAcc != NULL ? correctAcc->buffer : NULL,
        correctAcc != NULL ? (size_t)correctAcc->origin * sizeof(float) : 0,
        (uint32_t)batch, (uint32_t)inputs, (uint32_t)hidden, (uint32_t)classes);
    if (!ok)
        return jaiThrow(vm.cRuntimeError, "gpu_mlp_bwd_step(): the kernel failed");
    *out = NULL_VAL;
    return true;
}

static bool nGpuMlp3SgdStep(int argc, Value *args, Value *out) {
    (void)argc;
    GpuBuffer *x, *w1, *b1, *w2, *b2, *w3, *b3, *w4, *b4, *labels, *lossAcc, *correctAcc;
    if (!requireBuffer(args[0], 1, "gpu_mlp3_sgd_step", &x)) return false;
    if (!requireBuffer(args[1], 2, "gpu_mlp3_sgd_step", &w1)) return false;
    if (!requireBuffer(args[2], 3, "gpu_mlp3_sgd_step", &b1)) return false;
    if (!requireBuffer(args[3], 4, "gpu_mlp3_sgd_step", &w2)) return false;
    if (!requireBuffer(args[4], 5, "gpu_mlp3_sgd_step", &b2)) return false;
    if (!requireBuffer(args[5], 6, "gpu_mlp3_sgd_step", &w3)) return false;
    if (!requireBuffer(args[6], 7, "gpu_mlp3_sgd_step", &b3)) return false;
    if (!requireBuffer(args[7], 8, "gpu_mlp3_sgd_step", &w4)) return false;
    if (!requireBuffer(args[8], 9, "gpu_mlp3_sgd_step", &b4)) return false;
    if (!requireBuffer(args[9], 10, "gpu_mlp3_sgd_step", &labels)) return false;
    if (!requireBuffer(args[10], 11, "gpu_mlp3_sgd_step", &lossAcc)) return false;
    correctAcc = NULL;
    if (!IS_NULL(args[11])) {
        if (!requireBuffer(args[11], 12, "gpu_mlp3_sgd_step", &correctAcc)) return false;
    }
    int64_t batch, inputs, hidden1, hidden2, hidden3, classes;
    double lr;
    if (!jaiArgInt(args[12], 13, "gpu_mlp3_sgd_step", &batch)) return false;
    if (!jaiArgInt(args[13], 14, "gpu_mlp3_sgd_step", &inputs)) return false;
    if (!jaiArgInt(args[14], 15, "gpu_mlp3_sgd_step", &hidden1)) return false;
    if (!jaiArgInt(args[15], 16, "gpu_mlp3_sgd_step", &hidden2)) return false;
    if (!jaiArgInt(args[16], 17, "gpu_mlp3_sgd_step", &hidden3)) return false;
    if (!jaiArgInt(args[17], 18, "gpu_mlp3_sgd_step", &classes)) return false;
    if (!jaiArgNumber(args[18], 19, "gpu_mlp3_sgd_step", &lr)) return false;
    if (batch <= 0 || inputs <= 0 || hidden1 <= 0 || hidden2 <= 0 || hidden3 <= 0 ||
        classes <= 0 || batch > UINT32_MAX || inputs > UINT32_MAX ||
        hidden1 > UINT32_MAX || hidden2 > UINT32_MAX || hidden3 > UINT32_MAX ||
        classes > UINT32_MAX)
        return jaiThrow(vm.cValueError,
                        "gpu_mlp3_sgd_step(): dimensions must be positive uint32");
    if (!(lr > 0.0) || lr > 1e6)
        return jaiThrow(vm.cValueError, "gpu_mlp3_sgd_step(): learning rate is invalid");
    bool ok = jaiGpuMlp3SgdStep(
        x->buffer, (size_t)x->origin * sizeof(float),
        w1->buffer, (size_t)w1->origin * sizeof(float),
        b1->buffer, (size_t)b1->origin * sizeof(float),
        w2->buffer, (size_t)w2->origin * sizeof(float),
        b2->buffer, (size_t)b2->origin * sizeof(float),
        w3->buffer, (size_t)w3->origin * sizeof(float),
        b3->buffer, (size_t)b3->origin * sizeof(float),
        w4->buffer, (size_t)w4->origin * sizeof(float),
        b4->buffer, (size_t)b4->origin * sizeof(float),
        labels->buffer, (size_t)labels->origin * sizeof(float),
        lossAcc->buffer, (size_t)lossAcc->origin * sizeof(float),
        correctAcc != NULL ? correctAcc->buffer : NULL,
        correctAcc != NULL ? (size_t)correctAcc->origin * sizeof(float) : 0,
        (uint32_t)batch, (uint32_t)inputs, (uint32_t)hidden1, (uint32_t)hidden2,
        (uint32_t)hidden3, (uint32_t)classes, (float)lr);
    if (!ok)
        return jaiThrow(vm.cRuntimeError, "gpu_mlp3_sgd_step(): the kernel failed");
    *out = NULL_VAL;
    return true;
}

static bool nGpuMlp3SgdEpoch(int argc, Value *args, Value *out) {
    (void)argc;
    GpuBuffer *x, *w1, *b1, *w2, *b2, *w3, *b3, *w4, *b4, *labels, *lossAcc, *correctAcc;
    if (!requireBuffer(args[0], 1, "gpu_mlp3_sgd_epoch", &x)) return false;
    if (!requireBuffer(args[1], 2, "gpu_mlp3_sgd_epoch", &w1)) return false;
    if (!requireBuffer(args[2], 3, "gpu_mlp3_sgd_epoch", &b1)) return false;
    if (!requireBuffer(args[3], 4, "gpu_mlp3_sgd_epoch", &w2)) return false;
    if (!requireBuffer(args[4], 5, "gpu_mlp3_sgd_epoch", &b2)) return false;
    if (!requireBuffer(args[5], 6, "gpu_mlp3_sgd_epoch", &w3)) return false;
    if (!requireBuffer(args[6], 7, "gpu_mlp3_sgd_epoch", &b3)) return false;
    if (!requireBuffer(args[7], 8, "gpu_mlp3_sgd_epoch", &w4)) return false;
    if (!requireBuffer(args[8], 9, "gpu_mlp3_sgd_epoch", &b4)) return false;
    if (!requireBuffer(args[9], 10, "gpu_mlp3_sgd_epoch", &labels)) return false;
    if (!requireBuffer(args[10], 11, "gpu_mlp3_sgd_epoch", &lossAcc)) return false;
    correctAcc = NULL;
    if (!IS_NULL(args[11])) {
        if (!requireBuffer(args[11], 12, "gpu_mlp3_sgd_epoch", &correctAcc)) return false;
    }
    int64_t samples, batch, inputs, hidden1, hidden2, hidden3, classes, flushEvery;
    double lr;
    if (!jaiArgInt(args[12], 13, "gpu_mlp3_sgd_epoch", &samples)) return false;
    if (!jaiArgInt(args[13], 14, "gpu_mlp3_sgd_epoch", &batch)) return false;
    if (!jaiArgInt(args[14], 15, "gpu_mlp3_sgd_epoch", &inputs)) return false;
    if (!jaiArgInt(args[15], 16, "gpu_mlp3_sgd_epoch", &hidden1)) return false;
    if (!jaiArgInt(args[16], 17, "gpu_mlp3_sgd_epoch", &hidden2)) return false;
    if (!jaiArgInt(args[17], 18, "gpu_mlp3_sgd_epoch", &hidden3)) return false;
    if (!jaiArgInt(args[18], 19, "gpu_mlp3_sgd_epoch", &classes)) return false;
    if (!jaiArgNumber(args[19], 20, "gpu_mlp3_sgd_epoch", &lr)) return false;
    if (!jaiArgInt(args[20], 21, "gpu_mlp3_sgd_epoch", &flushEvery)) return false;
    if (samples <= 0 || batch <= 0 || inputs <= 0 || hidden1 <= 0 || hidden2 <= 0 ||
        hidden3 <= 0 || classes <= 0 || flushEvery <= 0 || samples > UINT32_MAX ||
        batch > UINT32_MAX || inputs > UINT32_MAX || hidden1 > UINT32_MAX ||
        hidden2 > UINT32_MAX || hidden3 > UINT32_MAX || classes > UINT32_MAX ||
        flushEvery > UINT32_MAX)
        return jaiThrow(vm.cValueError, "gpu_mlp3_sgd_epoch(): dimensions are invalid");
    if (!(lr > 0.0) || lr > 1e6)
        return jaiThrow(vm.cValueError, "gpu_mlp3_sgd_epoch(): learning rate is invalid");
    uint32_t processed = 0;
    bool ok = jaiGpuMlp3SgdEpoch(
        x->buffer, (size_t)x->origin * sizeof(float),
        w1->buffer, (size_t)w1->origin * sizeof(float),
        b1->buffer, (size_t)b1->origin * sizeof(float),
        w2->buffer, (size_t)w2->origin * sizeof(float),
        b2->buffer, (size_t)b2->origin * sizeof(float),
        w3->buffer, (size_t)w3->origin * sizeof(float),
        b3->buffer, (size_t)b3->origin * sizeof(float),
        w4->buffer, (size_t)w4->origin * sizeof(float),
        b4->buffer, (size_t)b4->origin * sizeof(float),
        labels->buffer, (size_t)labels->origin * sizeof(float),
        lossAcc->buffer, (size_t)lossAcc->origin * sizeof(float),
        correctAcc != NULL ? correctAcc->buffer : NULL,
        correctAcc != NULL ? (size_t)correctAcc->origin * sizeof(float) : 0,
        (uint32_t)samples, (uint32_t)batch, (uint32_t)inputs, (uint32_t)hidden1,
        (uint32_t)hidden2, (uint32_t)hidden3, (uint32_t)classes, (float)lr,
        (uint32_t)flushEvery, &processed);
    if (!ok)
        return jaiThrow(vm.cRuntimeError, "gpu_mlp3_sgd_epoch(): the kernel failed");
    *out = INT_VAL((int64_t)processed);
    return true;
}

static bool nGpuMlp3BwdStep(int argc, Value *args, Value *out) {
    (void)argc;
    GpuBuffer *x, *w1, *b1, *w2, *b2, *w3, *b3, *w4, *b4, *labels;
    GpuBuffer *gW1, *gB1, *gW2, *gB2, *gW3, *gB3, *gW4, *gB4, *lossAcc, *correctAcc;
    if (!requireBuffer(args[0], 1, "gpu_mlp3_bwd_step", &x)) return false;
    if (!requireBuffer(args[1], 2, "gpu_mlp3_bwd_step", &w1)) return false;
    if (!requireBuffer(args[2], 3, "gpu_mlp3_bwd_step", &b1)) return false;
    if (!requireBuffer(args[3], 4, "gpu_mlp3_bwd_step", &w2)) return false;
    if (!requireBuffer(args[4], 5, "gpu_mlp3_bwd_step", &b2)) return false;
    if (!requireBuffer(args[5], 6, "gpu_mlp3_bwd_step", &w3)) return false;
    if (!requireBuffer(args[6], 7, "gpu_mlp3_bwd_step", &b3)) return false;
    if (!requireBuffer(args[7], 8, "gpu_mlp3_bwd_step", &w4)) return false;
    if (!requireBuffer(args[8], 9, "gpu_mlp3_bwd_step", &b4)) return false;
    if (!requireBuffer(args[9], 10, "gpu_mlp3_bwd_step", &labels)) return false;
    if (!requireBuffer(args[10], 11, "gpu_mlp3_bwd_step", &gW1)) return false;
    if (!requireBuffer(args[11], 12, "gpu_mlp3_bwd_step", &gB1)) return false;
    if (!requireBuffer(args[12], 13, "gpu_mlp3_bwd_step", &gW2)) return false;
    if (!requireBuffer(args[13], 14, "gpu_mlp3_bwd_step", &gB2)) return false;
    if (!requireBuffer(args[14], 15, "gpu_mlp3_bwd_step", &gW3)) return false;
    if (!requireBuffer(args[15], 16, "gpu_mlp3_bwd_step", &gB3)) return false;
    if (!requireBuffer(args[16], 17, "gpu_mlp3_bwd_step", &gW4)) return false;
    if (!requireBuffer(args[17], 18, "gpu_mlp3_bwd_step", &gB4)) return false;
    if (!requireBuffer(args[18], 19, "gpu_mlp3_bwd_step", &lossAcc)) return false;
    correctAcc = NULL;
    if (!IS_NULL(args[19])) {
        if (!requireBuffer(args[19], 20, "gpu_mlp3_bwd_step", &correctAcc)) return false;
    }
    int64_t batch, inputs, hidden1, hidden2, hidden3, classes;
    if (!jaiArgInt(args[20], 21, "gpu_mlp3_bwd_step", &batch)) return false;
    if (!jaiArgInt(args[21], 22, "gpu_mlp3_bwd_step", &inputs)) return false;
    if (!jaiArgInt(args[22], 23, "gpu_mlp3_bwd_step", &hidden1)) return false;
    if (!jaiArgInt(args[23], 24, "gpu_mlp3_bwd_step", &hidden2)) return false;
    if (!jaiArgInt(args[24], 25, "gpu_mlp3_bwd_step", &hidden3)) return false;
    if (!jaiArgInt(args[25], 26, "gpu_mlp3_bwd_step", &classes)) return false;
    if (batch <= 0 || inputs <= 0 || hidden1 <= 0 || hidden2 <= 0 || hidden3 <= 0 ||
        classes <= 0 || batch > UINT32_MAX || inputs > UINT32_MAX ||
        hidden1 > UINT32_MAX || hidden2 > UINT32_MAX || hidden3 > UINT32_MAX ||
        classes > UINT32_MAX)
        return jaiThrow(vm.cValueError,
                        "gpu_mlp3_bwd_step(): dimensions must be positive uint32");
    bool ok = jaiGpuMlp3BwdStep(
        x->buffer, (size_t)x->origin * sizeof(float),
        w1->buffer, (size_t)w1->origin * sizeof(float),
        b1->buffer, (size_t)b1->origin * sizeof(float),
        w2->buffer, (size_t)w2->origin * sizeof(float),
        b2->buffer, (size_t)b2->origin * sizeof(float),
        w3->buffer, (size_t)w3->origin * sizeof(float),
        b3->buffer, (size_t)b3->origin * sizeof(float),
        w4->buffer, (size_t)w4->origin * sizeof(float),
        b4->buffer, (size_t)b4->origin * sizeof(float),
        labels->buffer, (size_t)labels->origin * sizeof(float),
        gW1->buffer, (size_t)gW1->origin * sizeof(float),
        gB1->buffer, (size_t)gB1->origin * sizeof(float),
        gW2->buffer, (size_t)gW2->origin * sizeof(float),
        gB2->buffer, (size_t)gB2->origin * sizeof(float),
        gW3->buffer, (size_t)gW3->origin * sizeof(float),
        gB3->buffer, (size_t)gB3->origin * sizeof(float),
        gW4->buffer, (size_t)gW4->origin * sizeof(float),
        gB4->buffer, (size_t)gB4->origin * sizeof(float),
        lossAcc->buffer, (size_t)lossAcc->origin * sizeof(float),
        correctAcc != NULL ? correctAcc->buffer : NULL,
        correctAcc != NULL ? (size_t)correctAcc->origin * sizeof(float) : 0,
        (uint32_t)batch, (uint32_t)inputs, (uint32_t)hidden1, (uint32_t)hidden2,
        (uint32_t)hidden3, (uint32_t)classes);
    if (!ok)
        return jaiThrow(vm.cRuntimeError, "gpu_mlp3_bwd_step(): the kernel failed");
    *out = NULL_VAL;
    return true;
}

static bool nGpuLabelsValid(int argc, Value *args, Value *out) {
    (void)argc;
    GpuBuffer *labels;
    if (!requireBuffer(args[0], 1, "gpu_labels_valid", &labels)) return false;
    int64_t count, classes;
    if (!jaiArgInt(args[1], 2, "gpu_labels_valid", &count)) return false;
    if (!jaiArgInt(args[2], 3, "gpu_labels_valid", &classes)) return false;
    if (count < 0 || classes <= 0 || count > UINT32_MAX || classes > UINT32_MAX)
        return jaiThrow(vm.cValueError,
                        "gpu_labels_valid(): count and classes must fit in uint32");
    *out = BOOL_VAL(jaiGpuLabelsValid(
        labels->buffer, (size_t)labels->origin * sizeof(float),
        (uint32_t)count, (uint32_t)classes));
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
