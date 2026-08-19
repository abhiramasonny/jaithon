/* native.h — platform subsystems behind a C ABI.
 * Stubs return false where a platform doesn't support them, so the rest of the tree never needs #ifdef __APPLE__.
 */
#ifndef JAI_NATIVE_H
#define JAI_NATIVE_H

#include "common/common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct JaiWindow JaiWindow;
typedef struct JaiCamera JaiCamera;

/* Video capture. Frames arrive as packed BGRA, the layout Core Video hands
 * over and the one an image type expands in a single pass. */
/* 1 granted, 0 not yet asked, -1 refused, -2 not allowed on this machine. */
int         jaiCameraPermission(void);
int         jaiCameraDeviceCount(void);
bool        jaiCameraDeviceName(int index, char *buffer, size_t capacity);
/* `width`/`height`/`fps` of zero mean "whatever the device prefers". Returns
 * NULL when there is no camera, or when the user refused access. */
JaiCamera  *jaiCameraOpen(int index, int width, int height, double fps);
void        jaiCameraClose(JaiCamera *camera);
bool        jaiCameraSize(JaiCamera *camera, int *width, int *height);
/* Blocks until a frame newer than the last one read arrives, or the timeout
 * passes; a negative timeout waits forever. */
bool        jaiCameraRead(JaiCamera *camera, uint8_t *destination, size_t capacity,
                          int *width, int *height, double timeoutSeconds);

bool        jaiGuiAvailable(void);
JaiWindow  *jaiWindowOpen(int width, int height, const char *title, int targetFPS);
void        jaiWindowClose(JaiWindow *w);
/* Direct access to the ARGB back buffer; drawing primitives live in Jaithon. */
uint32_t   *jaiWindowPixels(JaiWindow *w, int *outWidth, int *outHeight);
void        jaiWindowPresent(JaiWindow *w);
bool        jaiWindowPoll(JaiWindow *w);          /* false once closed */
double      jaiWindowDeltaTime(JaiWindow *w);
double      jaiWindowFPS(JaiWindow *w);
void        jaiWindowMousePos(JaiWindow *w, double *x, double *y);
bool        jaiWindowMouseDown(JaiWindow *w, int button);
/* `hidCode` is a USB HID usage id, the same numbering the key events below
 * carry and the same one `std.gui.input.Key.code()` returns. */
bool        jaiWindowKeyDown(JaiWindow *w, int hidCode);
void        jaiWindowSetTitle(JaiWindow *w, const char *title);

/* Maps a platform key code (an AppKit virtual keycode on macOS) to the USB
 * HID id std.gui uses; 0 if unknown, or on a platform with no window system.
 * Exported (despite being internal) so the table can be unit-tested headless. */
int         jaiWindowKeyFromPlatform(int platformCode);

/* Test-only key injection: drives a synthetic key transition through the real
 * event path on a scratch window (NULL where windowing is unsupported); one
 * injection is one frame, so tap flags are cleared first, as jaiWindowPoll does. */
JaiWindow  *jaiWindowTestWindow(void);
void        jaiWindowTestInjectKey(int platformCode, bool down, bool repeat);

/* std.gui.Event decodes these tags directly — append only, never renumber.
 * Button codes: 0 left, 1 middle, 2 right, matching jaiWindowMouseDown. */
typedef enum {
    JAI_EVENT_CLOSE = 0,
    JAI_EVENT_KEY_DOWN,     /* i0 keycode, i1 non-zero when auto-repeat */
    JAI_EVENT_KEY_UP,       /* i0 keycode                               */
    JAI_EVENT_TEXT,         /* text, composed by the input method       */
    JAI_EVENT_MOUSE_MOVE,   /* i0 x, i1 y, in back-buffer pixels        */
    JAI_EVENT_MOUSE_DOWN,   /* i0 button, i1 x, i2 y                    */
    JAI_EVENT_MOUSE_UP,     /* i0 button, i1 x, i2 y                    */
    JAI_EVENT_WHEEL,        /* d0 dx, d1 dy, in lines                   */
    JAI_EVENT_RESIZE,       /* i0 width, i1 height, on screen           */
    JAI_EVENT_FOCUS,        /* i0 non-zero when focus was gained        */
} JaiWindowEventTag;

typedef struct {
    int    tag;
    int    i0, i1, i2;
    double d0, d1;
    char   text[32];        /* JAI_EVENT_TEXT only; always NUL-terminated */
} JaiWindowEvent;

/* Drains up to `max` events, oldest first; the queue fills during both
 * jaiWindowPoll and jaiWindowPresent, so drain once per frame. */
int         jaiWindowDrainEvents(JaiWindow *w, JaiWindowEvent *out, int max);

typedef struct JaiGpuBuffer JaiGpuBuffer;
typedef struct JaiGpuKernel JaiGpuKernel;

bool          jaiGpuAvailable(void);
const char   *jaiGpuDeviceName(void);
int           jaiGpuDeviceCount(void);
bool          jaiGpuSetDevice(int index);
void          jaiGpuSetMixedPrecision(bool enabled);
bool          jaiGpuMixedPrecision(void);
JaiGpuBuffer *jaiGpuAlloc(size_t bytes);
void          jaiGpuFree(JaiGpuBuffer *b);
/* `offset` is a byte offset into the device buffer, so std.gpu can move a slice
 * of a buffer without a read-modify-write of the whole thing. */
void          jaiGpuUpload(JaiGpuBuffer *b, const void *src, size_t bytes,
                           size_t offset);
/* Expand unsigned bytes directly into float slots, applying `scale`. */
void          jaiGpuUploadU8(JaiGpuBuffer *b, const uint8_t *src, size_t count,
                             size_t offset, float scale);
void          jaiGpuFillUniform(JaiGpuBuffer *b, size_t elementOffset, size_t count,
                                float low, float high, uint64_t seed);
void          jaiGpuFillZero(JaiGpuBuffer *b, size_t elementOffset, size_t count);
void          jaiGpuDownload(JaiGpuBuffer *b, void *dst, size_t bytes,
                             size_t offset);
JaiGpuKernel *jaiGpuCompile(const char *source, const char *entryPoint,
                            char *errBuf, size_t errBufSize);
/* The kernel is opaque; this is the only way to free one. */
void          jaiGpuKernelFree(JaiGpuKernel *k);
/* The widest threadgroup this kernel can be dispatched with; 0 when unknown. */
int           jaiGpuMaxThreadsPerGroup(JaiGpuKernel *k);
/* Buffers bind at buffer(0) upward, then scalars follow as `constant uint&`.
 * `groupSize` 0 means the widest group the kernel supports. */
bool          jaiGpuDispatch(JaiGpuKernel *k, JaiGpuBuffer **buffers, int count,
                             const uint32_t *scalars, int scalarCount,
                             int threads, int groupSize, const size_t *byteOffsets);
/* Queue work without waiting. Commands submitted to the same device execute
 * in order; jaiGpuSynchronize waits for every queued dispatch. */
bool          jaiGpuDispatchAsync(JaiGpuKernel *k, JaiGpuBuffer **buffers, int count,
                                  const uint32_t *scalars, int scalarCount,
                                  int threads, int groupSize, const size_t *byteOffsets);
/* Commit queued async work so the GPU can start; does not wait. */
bool          jaiGpuFlush(void);
bool          jaiGpuSynchronize(void);
/* Device-buffer GEMM via Metal Performance Shaders. Encodes onto the async
 * command buffer (ending any open compute encoder). `transA` / `transB` treat
 * the physical buffers as transposed. Offsets are bytes into each buffer.
 * `useHalf` runs the product in half precision, casting on the way in and back
 * on the way out; whether that is faster depends on the shape, so the caller
 * decides rather than a global flag. */
bool          jaiGpuMatMulBuffers(JaiGpuBuffer *a, size_t aOffset,
                                  JaiGpuBuffer *b, size_t bOffset,
                                  JaiGpuBuffer *out, size_t outOffset,
                                  uint32_t m, uint32_t k, uint32_t n,
                                  bool transA, bool transB, bool useHalf);
/* Packed multi-head attention: Q/K/V/out are `[seq, heads*hd]` row-major. */
bool          jaiGpuMhaPacked(JaiGpuBuffer *q, size_t qOff,
                              JaiGpuBuffer *k, size_t kOff,
                              JaiGpuBuffer *v, size_t vOff,
                              JaiGpuBuffer *out, size_t outOff,
                              uint32_t seq, uint32_t heads, uint32_t hd,
                              float scale);
/* NHWC input, HWIO weights, optional bias. Output is NHWC. */
bool          jaiGpuConv2dBuffers(JaiGpuBuffer *input, size_t inputOffset,
                                  JaiGpuBuffer *weights, size_t weightsOffset,
                                  JaiGpuBuffer *bias, size_t biasOffset,
                                  JaiGpuBuffer *out, size_t outOffset,
                                  uint32_t n, uint32_t h, uint32_t w, uint32_t cin,
                                  uint32_t cout, uint32_t kh, uint32_t kw,
                                  uint32_t sh, uint32_t sw, uint32_t ph, uint32_t pw);
/* Two-layer ReLU + linear MLP: one fused SGD step. Updates weights in place
 * and adds the batch's total cross-entropy and correct-class count into
 * lossAcc / correctAcc. */
bool          jaiGpuMlpSgdStep(JaiGpuBuffer *x, size_t xOff,
                               JaiGpuBuffer *w1, size_t w1Off,
                               JaiGpuBuffer *b1, size_t b1Off,
                               JaiGpuBuffer *w2, size_t w2Off,
                               JaiGpuBuffer *b2, size_t b2Off,
                               JaiGpuBuffer *labels, size_t labOff,
                               JaiGpuBuffer *lossAcc, size_t lossOff,
                               JaiGpuBuffer *correctAcc, size_t correctOff,
                               uint32_t batch, uint32_t inputs, uint32_t hidden,
                               uint32_t classes, float lr);
/* Walk a rank-two feature matrix in native code so the VM is not entered
 * between fused SGD steps. Drops a ragged trailing batch. Returns how many
 * samples were consumed. */
bool          jaiGpuMlpSgdEpoch(JaiGpuBuffer *x, size_t xOff,
                                JaiGpuBuffer *w1, size_t w1Off,
                                JaiGpuBuffer *b1, size_t b1Off,
                                JaiGpuBuffer *w2, size_t w2Off,
                                JaiGpuBuffer *b2, size_t b2Off,
                                JaiGpuBuffer *labels, size_t labOff,
                                JaiGpuBuffer *lossAcc, size_t lossOff,
                                JaiGpuBuffer *correctAcc, size_t correctOff,
                                uint32_t samples, uint32_t batch, uint32_t inputs,
                                uint32_t hidden, uint32_t classes, float lr,
                                uint32_t flushEvery, uint32_t *processed);
/* Two-layer ReLU MLP: fused forward + backward into gradient buffers. */
bool          jaiGpuMlpBwdStep(JaiGpuBuffer *x, size_t xOff,
                               JaiGpuBuffer *w1, size_t w1Off,
                               JaiGpuBuffer *b1, size_t b1Off,
                               JaiGpuBuffer *w2, size_t w2Off,
                               JaiGpuBuffer *b2, size_t b2Off,
                               JaiGpuBuffer *labels, size_t labOff,
                               JaiGpuBuffer *gW1, size_t gW1Off,
                               JaiGpuBuffer *gB1, size_t gB1Off,
                               JaiGpuBuffer *gW2, size_t gW2Off,
                               JaiGpuBuffer *gB2, size_t gB2Off,
                               JaiGpuBuffer *lossAcc, size_t lossOff,
                               JaiGpuBuffer *correctAcc, size_t correctOff,
                               uint32_t batch, uint32_t inputs, uint32_t hidden,
                               uint32_t classes);
/* Three hidden ReLU layers plus a linear head: fused SGD step. */
bool          jaiGpuMlp3SgdStep(JaiGpuBuffer *x, size_t xOff,
                                JaiGpuBuffer *w1, size_t w1Off,
                                JaiGpuBuffer *b1, size_t b1Off,
                                JaiGpuBuffer *w2, size_t w2Off,
                                JaiGpuBuffer *b2, size_t b2Off,
                                JaiGpuBuffer *w3, size_t w3Off,
                                JaiGpuBuffer *b3, size_t b3Off,
                                JaiGpuBuffer *w4, size_t w4Off,
                                JaiGpuBuffer *b4, size_t b4Off,
                                JaiGpuBuffer *labels, size_t labOff,
                                JaiGpuBuffer *lossAcc, size_t lossOff,
                                JaiGpuBuffer *correctAcc, size_t correctOff,
                                uint32_t batch, uint32_t inputs,
                                uint32_t hidden1, uint32_t hidden2, uint32_t hidden3,
                                uint32_t classes, float lr);
bool          jaiGpuMlp3SgdEpoch(JaiGpuBuffer *x, size_t xOff,
                                 JaiGpuBuffer *w1, size_t w1Off,
                                 JaiGpuBuffer *b1, size_t b1Off,
                                 JaiGpuBuffer *w2, size_t w2Off,
                                 JaiGpuBuffer *b2, size_t b2Off,
                                 JaiGpuBuffer *w3, size_t w3Off,
                                 JaiGpuBuffer *b3, size_t b3Off,
                                 JaiGpuBuffer *w4, size_t w4Off,
                                 JaiGpuBuffer *b4, size_t b4Off,
                                 JaiGpuBuffer *labels, size_t labOff,
                                 JaiGpuBuffer *lossAcc, size_t lossOff,
                                 JaiGpuBuffer *correctAcc, size_t correctOff,
                                 uint32_t samples, uint32_t batch, uint32_t inputs,
                                 uint32_t hidden1, uint32_t hidden2, uint32_t hidden3,
                                 uint32_t classes, float lr, uint32_t flushEvery,
                                 uint32_t *processed);
/* Three hidden ReLU layers plus a linear head: fused forward/backward. */
bool          jaiGpuMlp3BwdStep(JaiGpuBuffer *x, size_t xOff,
                                JaiGpuBuffer *w1, size_t w1Off,
                                JaiGpuBuffer *b1, size_t b1Off,
                                JaiGpuBuffer *w2, size_t w2Off,
                                JaiGpuBuffer *b2, size_t b2Off,
                                JaiGpuBuffer *w3, size_t w3Off,
                                JaiGpuBuffer *b3, size_t b3Off,
                                JaiGpuBuffer *w4, size_t w4Off,
                                JaiGpuBuffer *b4, size_t b4Off,
                                JaiGpuBuffer *labels, size_t labOff,
                                JaiGpuBuffer *gW1, size_t gW1Off,
                                JaiGpuBuffer *gB1, size_t gB1Off,
                                JaiGpuBuffer *gW2, size_t gW2Off,
                                JaiGpuBuffer *gB2, size_t gB2Off,
                                JaiGpuBuffer *gW3, size_t gW3Off,
                                JaiGpuBuffer *gB3, size_t gB3Off,
                                JaiGpuBuffer *gW4, size_t gW4Off,
                                JaiGpuBuffer *gB4, size_t gB4Off,
                                JaiGpuBuffer *lossAcc, size_t lossOff,
                                JaiGpuBuffer *correctAcc, size_t correctOff,
                                uint32_t batch, uint32_t inputs,
                                uint32_t hidden1, uint32_t hidden2, uint32_t hidden3,
                                uint32_t classes);
/* Scan class-index labels in a shared buffer; no GPU round-trip. */
bool          jaiGpuLabelsValid(JaiGpuBuffer *labels, size_t offset,
                                uint32_t count, uint32_t classes);
/* Built-in kernels used by std.gpu when no custom source is supplied. */
bool jaiGpuVectorAdd(const double *a, const double *b, double *out, size_t n);
bool jaiGpuVectorMul(const double *a, const double *b, double *out, size_t n);
bool jaiGpuMatMul(const double *a, const double *b, double *out,
                  size_t m, size_t k, size_t n);
bool jaiGpuReduceSum(const double *a, size_t n, double *out);

typedef struct JaiThread JaiThread;
typedef struct JaiMutex  JaiMutex;
typedef struct JaiCond   JaiCond;

int         jaiCpuCount(void);
JaiThread  *jaiThreadSpawn(void *(*fn)(void *), void *arg);
bool        jaiThreadJoin(JaiThread *t, void **outResult);
void        jaiThreadDetach(JaiThread *t);
JaiMutex   *jaiMutexNew(void);
void        jaiMutexFree(JaiMutex *m);
void        jaiMutexLock(JaiMutex *m);
/* Non-blocking: true if the lock was taken, false if another thread holds it.
 * Any other failure panics (same as jaiMutexLock), not read as ordinary contention. */
bool        jaiMutexTryLock(JaiMutex *m);
void        jaiMutexUnlock(JaiMutex *m);
JaiCond    *jaiCondNew(void);
void        jaiCondFree(JaiCond *c);
void        jaiCondWait(JaiCond *c, JaiMutex *m);
void        jaiCondSignal(JaiCond *c);
void        jaiCondBroadcast(JaiCond *c);
int64_t     jaiAtomicAddI64(volatile int64_t *p, int64_t delta);
bool        jaiAtomicCasI64(volatile int64_t *p, int64_t expect, int64_t desired);

/* Tasks must not touch VM state — they operate only on raw buffers. */
typedef void (*JaiTaskFn)(void *arg, int index);
bool jaiParallelFor(int start, int end, JaiTaskFn fn, void *arg, int maxThreads);

int   jaiProcessRun(const char *command, char **outStdout, size_t *outLen);

/* Fields split by mode: waiting form only fills exitCode + the two buffers;
 * streaming form only fills pid and the three descriptors. */
typedef struct {
    int    pid;
    int    exitCode;
    int    stdinFd, stdoutFd, stderrFd;   /* -1 unless streaming */
    char  *out;  size_t outLen;           /* NUL-terminated, may hold NULs */
    char  *err;  size_t errLen;
} JaiSpawnResult;

/* JAI_SPAWN_EXEC means fork succeeded but exec failed, so outErrno is the
 * child's errno — how "no such program" is distinguished from "out of fds". */
typedef enum {
    JAI_SPAWN_OK = 0,
    JAI_SPAWN_SETUP,     /* pipe/fork failed in the parent */
    JAI_SPAWN_EXEC,
    JAI_SPAWN_IO,        /* a pipe read or the wait failed */
} JaiSpawnStatus;

/* argv[0] searches PATH if it holds no '/'; envp (if given) replaces the
 * environment. Caller frees out/err via JAI_FREE_ARRAY(char, p, len+1). */
JaiSpawnStatus jaiProcessSpawn(const char *const *argv, const char *cwd,
                               const char *const *envp,
                               const char *stdinText, size_t stdinLen,
                               bool stream, JaiSpawnResult *result,
                               int *outErrno);

/* `block` false polls: returns false with *outExit untouched while running,
 * or false with errno set on real error — that's how callers tell them apart. */
bool jaiProcessWait(int pid, bool block, int *outExit);

bool jaiProcessSignal(int pid, int sig);
char **jaiListDir(const char *path, int *outCount);   /* caller frees */
bool  jaiStatPath(const char *path, int64_t *size, int64_t *mtime, bool *isDir);
const char *jaiExecutablePath(void);

#ifdef __cplusplus
}
#endif

#endif /* JAI_NATIVE_H */
