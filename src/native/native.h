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
JaiGpuBuffer *jaiGpuAlloc(size_t bytes);
void          jaiGpuFree(JaiGpuBuffer *b);
/* `offset` is a byte offset into the device buffer, so std.gpu can move a slice
 * of a buffer without a read-modify-write of the whole thing. */
void          jaiGpuUpload(JaiGpuBuffer *b, const void *src, size_t bytes,
                           size_t offset);
/* Expand unsigned bytes directly into float slots, applying `scale`. */
void          jaiGpuUploadU8(JaiGpuBuffer *b, const uint8_t *src, size_t count,
                             size_t offset, float scale);
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
