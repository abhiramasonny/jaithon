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

int jaiGpuDeviceCount(void) {
    return 0;
}

bool jaiGpuSetDevice(int index) {
    (void)index;
    return false;
}

void jaiGpuSetMixedPrecision(bool enabled) {
    (void)enabled;
}

bool jaiGpuMixedPrecision(void) {
    return false;
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

const float *jaiGpuMapRead(JaiGpuBuffer *b, size_t elementOffset, size_t count) {
    (void)b; (void)elementOffset; (void)count; return NULL;
}

void jaiGpuDownloadU8(JaiGpuBuffer *b, uint8_t *dst, size_t count,
                      size_t offset, float scale) {
    (void)b;
    (void)dst;
    (void)count;
    (void)offset;
    (void)scale;
}

void jaiGpuFillUniform(JaiGpuBuffer *b, size_t elementOffset, size_t count,
                       float low, float high, uint64_t seed) {
    (void)b;
    (void)elementOffset;
    (void)count;
    (void)low;
    (void)high;
    (void)seed;
}

void jaiGpuFillZero(JaiGpuBuffer *b, size_t elementOffset, size_t count) {
    (void)b;
    (void)elementOffset;
    (void)count;
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
                    int threads, int groupSize, const size_t *byteOffsets) {
    (void)k;
    (void)buffers;
    (void)count;
    (void)scalars;
    (void)scalarCount;
    (void)threads;
    (void)groupSize;
    (void)byteOffsets;
    return false;
}

bool jaiGpuDispatchAsync(JaiGpuKernel *k, JaiGpuBuffer **buffers, int count,
                         const uint32_t *scalars, int scalarCount,
                         int threads, int groupSize, const size_t *byteOffsets) {
    return jaiGpuDispatch(k, buffers, count, scalars, scalarCount, threads,
                          groupSize, byteOffsets);
}

bool jaiGpuFlush(void) {
    return true;
}

bool jaiGpuSynchronize(void) {
    return false;
}

float *jaiGpuMapWrite(JaiGpuBuffer *b, size_t elementOffset, size_t count) {
    (void)b;
    (void)elementOffset;
    (void)count;
    return NULL;
}

bool jaiGpuWaitFor(JaiGpuBuffer *b) {
    (void)b;
    return false;
}

void jaiGpuBufferMark(JaiGpuBuffer *b) {
    (void)b;
}

void *jaiGpuTensorDataAt(JaiGpuBuffer *b, size_t offset, void *shape) {
    (void)b;
    (void)offset;
    (void)shape;
    return NULL;
}

bool jaiGpuMatMulBuffers(JaiGpuBuffer *a, size_t aOffset, JaiGpuBuffer *b,
                         size_t bOffset, JaiGpuBuffer *out, size_t outOffset,
                         uint32_t m, uint32_t k, uint32_t n, bool transA,
                         bool transB, bool useHalf) {
    (void)useHalf;
    (void)a;
    (void)aOffset;
    (void)b;
    (void)bOffset;
    (void)out;
    (void)outOffset;
    (void)m;
    (void)k;
    (void)n;
    (void)transA;
    (void)transB;
    return false;
}

bool jaiGpuMhaPacked(JaiGpuBuffer *q, size_t qOff, JaiGpuBuffer *k, size_t kOff,
                     JaiGpuBuffer *v, size_t vOff, JaiGpuBuffer *out, size_t outOff,
                     uint32_t seq, uint32_t heads, uint32_t hd, float scale) {
    (void)q;
    (void)qOff;
    (void)k;
    (void)kOff;
    (void)v;
    (void)vOff;
    (void)out;
    (void)outOff;
    (void)seq;
    (void)heads;
    (void)hd;
    (void)scale;
    return false;
}

bool jaiGpuConv2dBuffers(JaiGpuBuffer *input, size_t inputOffset,
                         JaiGpuBuffer *weights, size_t weightsOffset,
                         JaiGpuBuffer *bias, size_t biasOffset,
                         JaiGpuBuffer *out, size_t outOffset,
                         uint32_t n, uint32_t h, uint32_t w, uint32_t cin,
                         uint32_t cout, uint32_t kh, uint32_t kw,
                         uint32_t sh, uint32_t sw, uint32_t ph, uint32_t pw,
                         uint32_t activation, uint32_t layout) {
    (void)layout;
    (void)input;
    (void)inputOffset;
    (void)weights;
    (void)weightsOffset;
    (void)bias;
    (void)biasOffset;
    (void)out;
    (void)outOffset;
    (void)n;
    (void)h;
    (void)w;
    (void)cin;
    (void)cout;
    (void)kh;
    (void)kw;
    (void)sh;
    (void)sw;
    (void)ph;
    (void)pw;
    (void)activation;
    return false;
}

static bool stubConvGradient(JaiGpuBuffer *grad, size_t gradOffset,
                             JaiGpuBuffer *other, size_t otherOffset,
                             JaiGpuBuffer *out, size_t outOffset,
                             uint32_t n, uint32_t h, uint32_t w, uint32_t cin,
                             uint32_t cout, uint32_t kh, uint32_t kw,
                             uint32_t sh, uint32_t sw, uint32_t ph, uint32_t pw) {
    (void)grad;
    (void)gradOffset;
    (void)other;
    (void)otherOffset;
    (void)out;
    (void)outOffset;
    (void)n;
    (void)h;
    (void)w;
    (void)cin;
    (void)cout;
    (void)kh;
    (void)kw;
    (void)sh;
    (void)sw;
    (void)ph;
    (void)pw;
    return false;
}

bool jaiGpuConv2dDataGradBuffers(JaiGpuBuffer *grad, size_t gradOffset,
                                 JaiGpuBuffer *weights, size_t weightsOffset,
                                 JaiGpuBuffer *out, size_t outOffset,
                                 uint32_t n, uint32_t h, uint32_t w, uint32_t cin,
                                 uint32_t cout, uint32_t kh, uint32_t kw,
                                 uint32_t sh, uint32_t sw, uint32_t ph, uint32_t pw) {
    return stubConvGradient(grad, gradOffset, weights, weightsOffset, out, outOffset,
                            n, h, w, cin, cout, kh, kw, sh, sw, ph, pw);
}

bool jaiGpuConv2dWeightsGradBuffers(JaiGpuBuffer *grad, size_t gradOffset,
                                    JaiGpuBuffer *input, size_t inputOffset,
                                    JaiGpuBuffer *out, size_t outOffset,
                                    uint32_t n, uint32_t h, uint32_t w, uint32_t cin,
                                    uint32_t cout, uint32_t kh, uint32_t kw,
                                    uint32_t sh, uint32_t sw, uint32_t ph, uint32_t pw) {
    return stubConvGradient(grad, gradOffset, input, inputOffset, out, outOffset,
                            n, h, w, cin, cout, kh, kw, sh, sw, ph, pw);
}

bool jaiGpuMlpSgdStep(JaiGpuBuffer *x, size_t xOff, JaiGpuBuffer *w1, size_t w1Off,
                      JaiGpuBuffer *b1, size_t b1Off, JaiGpuBuffer *w2, size_t w2Off,
                      JaiGpuBuffer *b2, size_t b2Off, JaiGpuBuffer *labels, size_t labOff,
                      JaiGpuBuffer *lossAcc, size_t lossOff, JaiGpuBuffer *correctAcc,
                      size_t correctOff, uint32_t batch, uint32_t inputs, uint32_t hidden,
                      uint32_t classes, float lr) {
    (void)x;
    (void)xOff;
    (void)w1;
    (void)w1Off;
    (void)b1;
    (void)b1Off;
    (void)w2;
    (void)w2Off;
    (void)b2;
    (void)b2Off;
    (void)labels;
    (void)labOff;
    (void)lossAcc;
    (void)lossOff;
    (void)correctAcc;
    (void)correctOff;
    (void)batch;
    (void)inputs;
    (void)hidden;
    (void)classes;
    (void)lr;
    return false;
}

bool jaiGpuMlpBwdStep(JaiGpuBuffer *x, size_t xOff, JaiGpuBuffer *w1, size_t w1Off,
                      JaiGpuBuffer *b1, size_t b1Off, JaiGpuBuffer *w2, size_t w2Off,
                      JaiGpuBuffer *b2, size_t b2Off, JaiGpuBuffer *labels, size_t labOff,
                      JaiGpuBuffer *gW1, size_t gW1Off, JaiGpuBuffer *gB1, size_t gB1Off,
                      JaiGpuBuffer *gW2, size_t gW2Off, JaiGpuBuffer *gB2, size_t gB2Off,
                      JaiGpuBuffer *lossAcc, size_t lossOff, JaiGpuBuffer *correctAcc,
                      size_t correctOff, uint32_t batch, uint32_t inputs, uint32_t hidden,
                      uint32_t classes) {
    (void)x;
    (void)xOff;
    (void)w1;
    (void)w1Off;
    (void)b1;
    (void)b1Off;
    (void)w2;
    (void)w2Off;
    (void)b2;
    (void)b2Off;
    (void)labels;
    (void)labOff;
    (void)gW1;
    (void)gW1Off;
    (void)gB1;
    (void)gB1Off;
    (void)gW2;
    (void)gW2Off;
    (void)gB2;
    (void)gB2Off;
    (void)lossAcc;
    (void)lossOff;
    (void)correctAcc;
    (void)correctOff;
    (void)batch;
    (void)inputs;
    (void)hidden;
    (void)classes;
    return false;
}

bool jaiGpuMlpSgdEpoch(JaiGpuBuffer *x, size_t xOff, JaiGpuBuffer *w1, size_t w1Off,
                       JaiGpuBuffer *b1, size_t b1Off, JaiGpuBuffer *w2, size_t w2Off,
                       JaiGpuBuffer *b2, size_t b2Off, JaiGpuBuffer *labels, size_t labOff,
                       JaiGpuBuffer *lossAcc, size_t lossOff, JaiGpuBuffer *correctAcc,
                       size_t correctOff, uint32_t samples, uint32_t batch,
                       uint32_t inputs, uint32_t hidden, uint32_t classes, float lr,
                       uint32_t flushEvery, uint32_t *processed) {
    (void)x; (void)xOff; (void)w1; (void)w1Off; (void)b1; (void)b1Off;
    (void)w2; (void)w2Off; (void)b2; (void)b2Off; (void)labels; (void)labOff;
    (void)lossAcc; (void)lossOff; (void)correctAcc; (void)correctOff;
    (void)samples; (void)batch; (void)inputs; (void)hidden; (void)classes;
    (void)lr; (void)flushEvery;
    if (processed != NULL) *processed = 0;
    return false;
}

bool jaiGpuMlp3SgdStep(JaiGpuBuffer *x, size_t xOff, JaiGpuBuffer *w1, size_t w1Off,
                       JaiGpuBuffer *b1, size_t b1Off, JaiGpuBuffer *w2, size_t w2Off,
                       JaiGpuBuffer *b2, size_t b2Off, JaiGpuBuffer *w3, size_t w3Off,
                       JaiGpuBuffer *b3, size_t b3Off, JaiGpuBuffer *w4, size_t w4Off,
                       JaiGpuBuffer *b4, size_t b4Off, JaiGpuBuffer *labels, size_t labOff,
                       JaiGpuBuffer *lossAcc, size_t lossOff, JaiGpuBuffer *correctAcc,
                       size_t correctOff,                        uint32_t batch, uint32_t inputs, uint32_t hidden1,
                       uint32_t hidden2, uint32_t hidden3, uint32_t classes, float lr) {
    (void)x; (void)xOff; (void)w1; (void)w1Off; (void)b1; (void)b1Off;
    (void)w2; (void)w2Off; (void)b2; (void)b2Off; (void)w3; (void)w3Off;
    (void)b3; (void)b3Off; (void)w4; (void)w4Off; (void)b4; (void)b4Off;
    (void)labels; (void)labOff; (void)lossAcc; (void)lossOff; (void)correctAcc;
    (void)correctOff; (void)batch; (void)inputs; (void)hidden1; (void)hidden2;
    (void)hidden3; (void)classes; (void)lr;
    return false;
}

bool jaiGpuMlp3SgdEpoch(JaiGpuBuffer *x, size_t xOff, JaiGpuBuffer *w1, size_t w1Off,
                        JaiGpuBuffer *b1, size_t b1Off, JaiGpuBuffer *w2, size_t w2Off,
                        JaiGpuBuffer *b2, size_t b2Off, JaiGpuBuffer *w3, size_t w3Off,
                        JaiGpuBuffer *b3, size_t b3Off, JaiGpuBuffer *w4, size_t w4Off,
                        JaiGpuBuffer *b4, size_t b4Off, JaiGpuBuffer *labels, size_t labOff,
                        JaiGpuBuffer *lossAcc, size_t lossOff, JaiGpuBuffer *correctAcc,
                        size_t correctOff, uint32_t samples, uint32_t batch,
                        uint32_t inputs, uint32_t hidden1, uint32_t hidden2,
                        uint32_t hidden3, uint32_t classes, float lr,
                        uint32_t flushEvery, uint32_t *processed) {
    (void)x; (void)xOff; (void)w1; (void)w1Off; (void)b1; (void)b1Off;
    (void)w2; (void)w2Off; (void)b2; (void)b2Off; (void)w3; (void)w3Off;
    (void)b3; (void)b3Off; (void)w4; (void)w4Off; (void)b4; (void)b4Off;
    (void)labels; (void)labOff; (void)lossAcc; (void)lossOff; (void)correctAcc;
    (void)correctOff; (void)samples; (void)batch; (void)inputs; (void)hidden1;
    (void)hidden2; (void)hidden3; (void)classes; (void)lr; (void)flushEvery;
    if (processed != NULL) *processed = 0;
    return false;
}

bool jaiGpuMlp3BwdStep(JaiGpuBuffer *x, size_t xOff, JaiGpuBuffer *w1, size_t w1Off,
                       JaiGpuBuffer *b1, size_t b1Off, JaiGpuBuffer *w2, size_t w2Off,
                       JaiGpuBuffer *b2, size_t b2Off, JaiGpuBuffer *w3, size_t w3Off,
                       JaiGpuBuffer *b3, size_t b3Off, JaiGpuBuffer *w4, size_t w4Off,
                       JaiGpuBuffer *b4, size_t b4Off, JaiGpuBuffer *labels, size_t labOff,
                       JaiGpuBuffer *gW1, size_t gW1Off, JaiGpuBuffer *gB1, size_t gB1Off,
                       JaiGpuBuffer *gW2, size_t gW2Off, JaiGpuBuffer *gB2, size_t gB2Off,
                       JaiGpuBuffer *gW3, size_t gW3Off, JaiGpuBuffer *gB3, size_t gB3Off,
                       JaiGpuBuffer *gW4, size_t gW4Off, JaiGpuBuffer *gB4, size_t gB4Off,
                       JaiGpuBuffer *lossAcc, size_t lossOff, JaiGpuBuffer *correctAcc,
                       size_t correctOff, uint32_t batch, uint32_t inputs, uint32_t hidden1,
                       uint32_t hidden2, uint32_t hidden3, uint32_t classes) {
    (void)x; (void)xOff; (void)w1; (void)w1Off; (void)b1; (void)b1Off;
    (void)w2; (void)w2Off; (void)b2; (void)b2Off; (void)w3; (void)w3Off;
    (void)b3; (void)b3Off; (void)w4; (void)w4Off; (void)b4; (void)b4Off;
    (void)labels; (void)labOff; (void)gW1; (void)gW1Off; (void)gB1; (void)gB1Off;
    (void)gW2; (void)gW2Off; (void)gB2; (void)gB2Off; (void)gW3; (void)gW3Off;
    (void)gB3; (void)gB3Off; (void)gW4; (void)gW4Off; (void)gB4; (void)gB4Off;
    (void)lossAcc; (void)lossOff; (void)correctAcc; (void)correctOff;
    (void)batch; (void)inputs; (void)hidden1; (void)hidden2; (void)hidden3; (void)classes;
    return false;
}

bool jaiGpuLabelsValid(JaiGpuBuffer *labels, size_t offset, uint32_t count,
                       uint32_t classes) {
    (void)labels;
    (void)offset;
    (void)count;
    (void)classes;
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

int jaiCameraPermission(void) {
    return -2;
}

int jaiCameraDeviceCount(void) {
    return 0;
}

bool jaiCameraDeviceName(int index, char *buffer, size_t capacity) {
    (void)index;
    (void)buffer;
    (void)capacity;
    return false;
}

JaiCamera *jaiCameraOpen(int index, int width, int height, double fps) {
    (void)index;
    (void)width;
    (void)height;
    (void)fps;
    return NULL;
}

void jaiCameraClose(JaiCamera *camera) {
    (void)camera;
}

bool jaiCameraSize(JaiCamera *camera, int *width, int *height) {
    (void)camera;
    (void)width;
    (void)height;
    return false;
}

bool jaiCameraRead(JaiCamera *camera, uint8_t *destination, size_t capacity,
                   int *width, int *height, double timeoutSeconds) {
    (void)camera;
    (void)destination;
    (void)capacity;
    (void)width;
    (void)height;
    (void)timeoutSeconds;
    return false;
}

bool jaiGpuReduceSum(const double *a, size_t n, double *out) {
    if (a == NULL || out == NULL) return false;
    *out = compensatedSum(a, n);
    return true;
}

/* No CoreML off Apple, so nothing to reach an accelerator with. */
JaiCoreMLModel *jaiCoreMLOpen(const char *path, int units, char *errBuf, size_t errSize) {
    (void)path; (void)units;
    if (errBuf != NULL && errSize > 0) errBuf[0] = '\0';
    return NULL;
}
void jaiCoreMLClose(JaiCoreMLModel *model) { (void)model; }
int jaiCoreMLCount(JaiCoreMLModel *model, int outputs) { (void)model; (void)outputs; return -1; }
const char *jaiCoreMLName(JaiCoreMLModel *model, int outputs, int index) {
    (void)model; (void)outputs; (void)index; return NULL;
}
int jaiCoreMLShape(JaiCoreMLModel *model, int outputs, int index, int64_t *dims, int room) {
    (void)model; (void)outputs; (void)index; (void)dims; (void)room; return -1;
}
JaiCoreMLTicket *jaiCoreMLStart(JaiCoreMLModel *model,
                                JaiGpuBuffer **ins, const size_t *inOffsets,
                                const int64_t *inShapes, const int *inRanks,
                                JaiGpuBuffer **outs, const size_t *outOffsets,
                                const int64_t *outShapes, const int *outRanks,
                                char *errBuf, size_t errSize) {
    (void)model; (void)ins; (void)inOffsets; (void)inShapes; (void)inRanks;
    (void)outs; (void)outOffsets; (void)outShapes; (void)outRanks;
    if (errBuf != NULL && errSize > 0) errBuf[0] = '\0';
    return NULL;
}
bool jaiCoreMLWait(JaiCoreMLTicket *ticket, char *errBuf, size_t errSize) {
    (void)ticket;
    if (errBuf != NULL && errSize > 0) errBuf[0] = '\0';
    return false;
}
bool jaiCoreMLRun(JaiCoreMLModel *model,
                  JaiGpuBuffer **ins, const size_t *inOffsets,
                  const int64_t *inShapes, const int *inRanks,
                  JaiGpuBuffer **outs, const size_t *outOffsets,
                  const int64_t *outShapes, const int *outRanks,
                  char *errBuf, size_t errSize) {
    (void)model; (void)ins; (void)inOffsets; (void)inShapes; (void)inRanks;
    (void)outs; (void)outOffsets; (void)outShapes; (void)outRanks;
    if (errBuf != NULL && errSize > 0) errBuf[0] = '\0';
    return false;
}

/* No Metal, so no graph compiler either; every entry point refuses and the
 * caller runs its operators one at a time as it always could. */
JaiGraphBuilder *jaiGraphNew(void) { return NULL; }
void jaiGraphFree(JaiGraphBuilder *b) { (void)b; }
int jaiGraphInput(JaiGraphBuilder *b, const int64_t *dims, int rank) {
    (void)b; (void)dims; (void)rank; return -1;
}
int jaiGraphConstant(JaiGraphBuilder *b, const float *values, const int64_t *dims, int rank) {
    (void)b; (void)values; (void)dims; (void)rank; return -1;
}
int jaiGraphUnary(JaiGraphBuilder *b, int x, int op) { (void)b; (void)x; (void)op; return -1; }
int jaiGraphClamp(JaiGraphBuilder *b, int x, float lo, float hi) {
    (void)b; (void)x; (void)lo; (void)hi; return -1;
}
int jaiGraphLeakyRelu(JaiGraphBuilder *b, int x, float slope) {
    (void)b; (void)x; (void)slope; return -1;
}
int jaiGraphBinary(JaiGraphBuilder *b, int l, int r, int op) {
    (void)b; (void)l; (void)r; (void)op; return -1;
}
int jaiGraphConv(JaiGraphBuilder *b, int x, int w, int bias, const int32_t *p) {
    (void)b; (void)x; (void)w; (void)bias; (void)p; return -1;
}
int jaiGraphConvTranspose(JaiGraphBuilder *b, int x, int w, int bias,
                          const int32_t *p, const int64_t *shape) {
    (void)b; (void)x; (void)w; (void)bias; (void)p; (void)shape; return -1;
}
int jaiGraphPool(JaiGraphBuilder *b, int x, const int32_t *p, int kind) {
    (void)b; (void)x; (void)p; (void)kind; return -1;
}
int jaiGraphSelect(JaiGraphBuilder *b, int p, int a, int c) {
    (void)b; (void)p; (void)a; (void)c; return -1;
}
int jaiGraphConcat(JaiGraphBuilder *b, const int *ids, int count, int axis) {
    (void)b; (void)ids; (void)count; (void)axis; return -1;
}
int jaiGraphReshape(JaiGraphBuilder *b, int x, const int64_t *dims, int rank) {
    (void)b; (void)x; (void)dims; (void)rank; return -1;
}
int jaiGraphTranspose(JaiGraphBuilder *b, int x, const int32_t *perm, int rank) {
    (void)b; (void)x; (void)perm; (void)rank; return -1;
}
int jaiGraphSlice(JaiGraphBuilder *b, int x, const int32_t *s, const int32_t *e,
                  const int32_t *t, int rank) {
    (void)b; (void)x; (void)s; (void)e; (void)t; (void)rank; return -1;
}
int jaiGraphPad(JaiGraphBuilder *b, int x, const int32_t *before, const int32_t *after,
                int rank, int mode, float value) {
    (void)b; (void)x; (void)before; (void)after; (void)rank; (void)mode; (void)value;
    return -1;
}
int jaiGraphArgExtreme(JaiGraphBuilder *b, int x, int axis, int largest) {
    (void)b; (void)x; (void)axis; (void)largest; return -1;
}
int jaiGraphSoftmax(JaiGraphBuilder *b, int x, int axis) { (void)b; (void)x; (void)axis; return -1; }
int jaiGraphResize(JaiGraphBuilder *b, int x, int h, int w, int r, int c, int a, int bi) {
    (void)b; (void)x; (void)h; (void)w; (void)r; (void)c; (void)a; (void)bi; return -1;
}
int jaiGraphGather(JaiGraphBuilder *b, int d, int i, int axis) {
    (void)b; (void)d; (void)i; (void)axis; return -1;
}
int jaiGraphMatmul(JaiGraphBuilder *b, int l, int r) { (void)b; (void)l; (void)r; return -1; }
int jaiGraphReduce(JaiGraphBuilder *b, int x, const int32_t *axes, int count, int kind) {
    (void)b; (void)x; (void)axes; (void)count; (void)kind; return -1;
}
int jaiGraphLayerNorm(JaiGraphBuilder *b, int x, int gamma, int beta,
                      const int32_t *axes, int count, float epsilon) {
    (void)b; (void)x; (void)gamma; (void)beta; (void)axes; (void)count; (void)epsilon;
    return -1;
}
int jaiGraphGemm(JaiGraphBuilder *b, int l, int r, int c, int tl, int tr,
                 float alpha, float beta) {
    (void)b; (void)l; (void)r; (void)c; (void)tl; (void)tr; (void)alpha; (void)beta;
    return -1;
}
bool jaiGraphLstm(JaiGraphBuilder *b, int s, int r, int w, int bias,
                  int state, int cell, int reverse, int *out) {
    (void)b; (void)s; (void)r; (void)w; (void)bias; (void)state; (void)cell;
    (void)reverse; (void)out;
    return false;
}
int jaiGraphGru(JaiGraphBuilder *b, int s, int r, int w, int bias, int state, int reverse,
                int resetAfter, int resetBias) {
    (void)b; (void)s; (void)r; (void)w; (void)bias; (void)state; (void)reverse;
    (void)resetAfter; (void)resetBias; return -1;
}
int jaiGraphOneHot(JaiGraphBuilder *b, int indices, int depth) {
    (void)b; (void)indices; (void)depth; return -1;
}
int jaiGraphSoftmaxCrossEntropy(JaiGraphBuilder *b, int logits, int labels, int axis,
                                int reduction) {
    (void)b; (void)logits; (void)labels; (void)axis; (void)reduction; return -1;
}
bool jaiGraphGradients(JaiGraphBuilder *b, int loss, const int *wants, int count,
                       int *out) {
    (void)b; (void)loss; (void)wants; (void)count; (void)out; return false;
}
JaiGraphPlan *jaiGraphCompile(JaiGraphBuilder *b, const int *in, int inCount,
                              const int *out, int outCount) {
    (void)b; (void)in; (void)inCount; (void)out; (void)outCount; return NULL;
}
int jaiGraphPlanOutputRank(JaiGraphPlan *plan, int index) { (void)plan; (void)index; return -1; }
bool jaiGraphPlanOutputShape(JaiGraphPlan *plan, int index, int64_t *dims, int rank) {
    (void)plan; (void)index; (void)dims; (void)rank; return false;
}
bool jaiGraphRun(JaiGraphPlan *plan, JaiGpuBuffer **ins, const size_t *inOff,
                 JaiGpuBuffer **outs, const size_t *outOff) {
    (void)plan; (void)ins; (void)inOff; (void)outs; (void)outOff; return false;
}
void jaiGraphPlanFree(JaiGraphPlan *plan) { (void)plan; }

#endif /* !__APPLE__ */
