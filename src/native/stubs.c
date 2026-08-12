/* stubs.c — windowing/GPU surface for platforms with neither; gui.m and gpu.m
 * own these symbols on macOS. The four arithmetic kernels are duplicated here
 * from gpu.m rather than shared, since the two files never compile together
 * and a shared object would be dead weight on both. */

#include "native/native.h"

/* Keeps the translation unit non-empty on Apple targets, where everything below
 * is compiled out. */
typedef int JaiNativeStubsUnit;

#ifndef __APPLE__

#include <math.h>

bool jaiGuiAvailable(void) {
    return false;
}

JaiWindow *jaiWindowOpen(int width, int height, const char *title, int targetFPS) {
    (void)width;
    (void)height;
    (void)title;
    (void)targetFPS;
    return NULL;
}

void jaiWindowClose(JaiWindow *w) {
    (void)w;
}

uint32_t *jaiWindowPixels(JaiWindow *w, int *outWidth, int *outHeight) {
    (void)w;
    if (outWidth != NULL) *outWidth = 0;
    if (outHeight != NULL) *outHeight = 0;
    return NULL;
}

void jaiWindowPresent(JaiWindow *w) {
    (void)w;
}

bool jaiWindowPoll(JaiWindow *w) {
    (void)w;
    return false;
}

double jaiWindowDeltaTime(JaiWindow *w) {
    (void)w;
    return 0.0;
}

double jaiWindowFPS(JaiWindow *w) {
    (void)w;
    return 0.0;
}

void jaiWindowMousePos(JaiWindow *w, double *x, double *y) {
    (void)w;
    if (x != NULL) *x = 0.0;
    if (y != NULL) *y = 0.0;
}

bool jaiWindowMouseDown(JaiWindow *w, int button) {
    (void)w;
    (void)button;
    return false;
}

bool jaiWindowKeyDown(JaiWindow *w, int hidCode) {
    (void)w;
    (void)hidCode;
    return false;
}

int jaiWindowKeyFromPlatform(int platformCode) {
    (void)platformCode;
    return 0;
}

JaiWindow *jaiWindowTestWindow(void) {
    return NULL;
}

void jaiWindowTestInjectKey(int platformCode, bool down, bool repeat) {
    (void)platformCode;
    (void)down;
    (void)repeat;
}

int jaiWindowDrainEvents(JaiWindow *w, JaiWindowEvent *out, int max) {
    (void)w;
    (void)out;
    (void)max;
    return 0;
}

void jaiWindowSetTitle(JaiWindow *w, const char *title) {
    (void)w;
    (void)title;
}

bool jaiGpuAvailable(void) {
    return false;
}

const char *jaiGpuDeviceName(void) {
    return "none";
}

JaiGpuBuffer *jaiGpuAlloc(size_t bytes) {
    (void)bytes;
    return NULL;
}

void jaiGpuFree(JaiGpuBuffer *b) {
    (void)b;
}

void jaiGpuUpload(JaiGpuBuffer *b, const void *src, size_t bytes, size_t offset) {
    (void)b;
    (void)src;
    (void)bytes;
    (void)offset;
}

void jaiGpuUploadU8(JaiGpuBuffer *b, const uint8_t *src, size_t count,
                    size_t offset, float scale) {
    (void)b;
    (void)src;
    (void)count;
    (void)offset;
    (void)scale;
}

void jaiGpuDownload(JaiGpuBuffer *b, void *dst, size_t bytes, size_t offset) {
    (void)b;
    (void)dst;
    (void)bytes;
    (void)offset;
}

JaiGpuKernel *jaiGpuCompile(const char *source, const char *entryPoint,
                            char *errBuf, size_t errBufSize) {
    (void)source;
    (void)entryPoint;
    if (errBuf != NULL && errBufSize > 0) {
        snprintf(errBuf, errBufSize, "no GPU device is available on this platform");
    }
    return NULL;
}

void jaiGpuKernelFree(JaiGpuKernel *k) {
    (void)k;   /* jaiGpuCompile never hands one out here */
}

int jaiGpuMaxThreadsPerGroup(JaiGpuKernel *k) {
    (void)k;
    return 0;
}

bool jaiGpuDispatch(JaiGpuKernel *k, JaiGpuBuffer **buffers, int count,
                    const uint32_t *scalars, int scalarCount,
                    int threads, int groupSize) {
    (void)k;
    (void)buffers;
    (void)count;
    (void)scalars;
    (void)scalarCount;
    (void)threads;
    (void)groupSize;
    return false;
}

bool jaiGpuDispatchAsync(JaiGpuKernel *k, JaiGpuBuffer **buffers, int count,
                         const uint32_t *scalars, int scalarCount,
                         int threads, int groupSize) {
    return jaiGpuDispatch(k, buffers, count, scalars, scalarCount, threads, groupSize);
}

bool jaiGpuSynchronize(void) {
    return false;
}

/* Neumaier variant of Kahan summation — also captures the low bits dropped
 * when the running total is smaller than the term added. Must stay identical to gpu.m's copy. */
static double compensatedSum(const double *values, size_t n) {
    double total = 0.0;
    double correction = 0.0;
    for (size_t i = 0; i < n; i++) {
        double value = values[i];
        double next = total + value;
        if (fabs(total) >= fabs(value)) {
            correction += (total - next) + value;
        } else {
            correction += (value - next) + total;
        }
        total = next;
    }
    return total + correction;
}

bool jaiGpuVectorAdd(const double *a, const double *b, double *out, size_t n) {
    if (a == NULL || b == NULL || out == NULL) return false;
    for (size_t i = 0; i < n; i++) out[i] = a[i] + b[i];
    return true;
}

bool jaiGpuVectorMul(const double *a, const double *b, double *out, size_t n) {
    if (a == NULL || b == NULL || out == NULL) return false;
    for (size_t i = 0; i < n; i++) out[i] = a[i] * b[i];
    return true;
}

/* Row/inner/column loop order keeps `b` and the output row walked forward;
 * same arithmetic as the textbook triple loop, different cache behavior. */
bool jaiGpuMatMul(const double *a, const double *b, double *out,
                  size_t m, size_t k, size_t n) {
    if (a == NULL || b == NULL || out == NULL) return false;
    if (m == 0 || n == 0) return true;
    if (k != 0 && (m > SIZE_MAX / k || n > SIZE_MAX / k)) return false;
    if (m > SIZE_MAX / n) return false;

    memset(out, 0, m * n * sizeof(double));
    for (size_t row = 0; row < m; row++) {
        double *outRow = out + row * n;
        const double *aRow = a + row * k;
        for (size_t i = 0; i < k; i++) {
            double factor = aRow[i];
            if (factor == 0.0) continue;
            const double *bRow = b + i * n;
            for (size_t column = 0; column < n; column++) {
                outRow[column] += factor * bRow[column];
            }
        }
    }
    return true;
}

bool jaiGpuReduceSum(const double *a, size_t n, double *out) {
    if (a == NULL || out == NULL) return false;
    *out = compensatedSum(a, n);
    return true;
}

#endif /* !__APPLE__ */
