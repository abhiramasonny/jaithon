/* stubs.c — the windowing and GPU surface on platforms that have neither.
 *
 * Only compiled in on non-Apple targets; gui.m and gpu.m own these symbols on
 * macOS. Everything that needs a device reports that there is none, so std.gui
 * and std.gpu see the same "no device" answer they would get from a Mac without
 * one, and the tree links on Linux.
 *
 * The four built-in kernels are the exception: they carry the same scalar code
 * their macOS counterparts fall back on, because a caller of jaiGpuVectorAdd
 * wants the sum, not a report that this machine has no Metal. Keeping them here
 * is what makes the built-ins produce identical answers on every platform. The
 * arithmetic is duplicated from gpu.m rather than shared, because the two files
 * are compiled on mutually exclusive platforms and a shared object file would
 * be dead weight on both.
 */

#include "native.h"

/* Keeps the translation unit non-empty on Apple targets, where everything below
 * is compiled out. */
typedef int JaiNativeStubsUnit;

#ifndef __APPLE__

#include <math.h>

/* ------------------------------------------------------------------ */
/* Windowing                                                           */
/* ------------------------------------------------------------------ */

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

/* No window system, so no platform key numbering to translate out of. 0 is the
 * "names no key" answer, which is what every code gets here. */
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

/* ------------------------------------------------------------------ */
/* GPU                                                                 */
/* ------------------------------------------------------------------ */

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

/* Neumaier's variant of Kahan summation: it also keeps the low bits of the
 * additions a plain Kahan loop drops, the ones where the running total is
 * smaller than the term being added. Must stay identical to gpu.m's copy. */
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

/* Row, inner, column order so that both `b` and the output row are walked
 * forwards; the arithmetic is the textbook triple loop's, the cache behaviour
 * is not. */
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
