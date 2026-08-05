/* native.h — platform subsystems behind a C ABI.
 *
 * These are the only places Jaithon touches the OS beyond libc. Each function
 * is a no-op stub returning false on platforms that do not support it, so the
 * rest of the tree never needs #ifdef __APPLE__.
 */
#ifndef JAI_NATIVE_H
#define JAI_NATIVE_H

#include "../common/common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Windowing and 2-D canvas (Metal/Cocoa on macOS)                      */
/* ------------------------------------------------------------------ */

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

/* Translate one of the window system's own key numbers — an AppKit virtual
 * keycode on macOS — into the USB HID usage id everything above speaks.
 * 0 when this platform's number names no key std.gui knows, which is also what
 * a platform with no window system returns for every input.
 *
 * The translation is applied inside the event pump, so nothing outside this
 * header needs it; it is exported only so the table can be tested without a
 * display, which is the one thing a GUI program cannot be made to do. */
int         jaiWindowKeyFromPlatform(int platformCode);

/* Test-only key injection, the counterpart to the above: push one synthetic
 * transition through the very code path a real key event takes, on a scratch
 * window with nothing behind it, then read the result back with the ordinary
 * jaiWindowDrainEvents and jaiWindowKeyDown. `jaiWindowTestWindow` returns that
 * window, creating it on first use, and NULL where windowing is unsupported.
 *
 * One injection is one frame: the tap flags are cleared first, as jaiWindowPoll
 * clears them. Exposed to Jaithon as `__prim__.gui_test_key_event` and
 * `__prim__.gui_test_key_state` for tests/stdlib/test_gui_input.jai. */
JaiWindow  *jaiWindowTestWindow(void);
void        jaiWindowTestInjectKey(int platformCode, bool down, bool repeat);

/* Discrete events, as opposed to the polled state above. std.gui.Event is
 * decoded straight from these tags, so they may be appended to but never
 * renumbered. Button codes are 0 left, 1 middle, 2 right — the same numbering
 * jaiWindowMouseDown uses. */
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
    int    tag;             /* a JaiWindowEventTag */
    int    i0, i1, i2;
    double d0, d1;
    char   text[32];        /* JAI_EVENT_TEXT only; always NUL-terminated */
} JaiWindowEvent;

/* Move up to `max` queued events into `out`, oldest first, and drop them from
 * the queue. Returns how many were written. The queue fills during
 * jaiWindowPoll and jaiWindowPresent, so drain it once per frame. */
int         jaiWindowDrainEvents(JaiWindow *w, JaiWindowEvent *out, int max);

/* ------------------------------------------------------------------ */
/* GPU compute (Metal on macOS)                                         */
/* ------------------------------------------------------------------ */

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
void          jaiGpuDownload(JaiGpuBuffer *b, void *dst, size_t bytes,
                             size_t offset);
/* Compile a kernel from Metal Shading Language source. */
JaiGpuKernel *jaiGpuCompile(const char *source, const char *entryPoint,
                            char *errBuf, size_t errBufSize);
/* Release a compiled pipeline. The kernel is opaque, so this is the only way to
 * free one — std.gpu's Kernel.free is written over it. */
void          jaiGpuKernelFree(JaiGpuKernel *k);
/* The widest threadgroup this kernel can be dispatched with; 0 when unknown. */
int           jaiGpuMaxThreadsPerGroup(JaiGpuKernel *k);
/* Buffers bind at buffer(0) upwards, then the scalars follow them one apiece as
 * `constant uint&` — the argument order std.gpu.Kernel documents. `groupSize` 0
 * means the widest group the kernel supports. */
bool          jaiGpuDispatch(JaiGpuKernel *k, JaiGpuBuffer **buffers, int count,
                             const uint32_t *scalars, int scalarCount,
                             int threads, int groupSize);
/* Built-in kernels used by std.gpu when no custom source is supplied. */
bool jaiGpuVectorAdd(const double *a, const double *b, double *out, size_t n);
bool jaiGpuVectorMul(const double *a, const double *b, double *out, size_t n);
bool jaiGpuMatMul(const double *a, const double *b, double *out,
                  size_t m, size_t k, size_t n);
bool jaiGpuReduceSum(const double *a, size_t n, double *out);

/* ------------------------------------------------------------------ */
/* Threads                                                              */
/* ------------------------------------------------------------------ */

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
/* Non-blocking: true when the lock was taken, false when another thread holds
 * it. A failure other than "busy" is the same caller bug jaiMutexLock panics
 * on, so it panics here too rather than reading as ordinary contention. */
bool        jaiMutexTryLock(JaiMutex *m);
void        jaiMutexUnlock(JaiMutex *m);
JaiCond    *jaiCondNew(void);
void        jaiCondFree(JaiCond *c);
void        jaiCondWait(JaiCond *c, JaiMutex *m);
void        jaiCondSignal(JaiCond *c);
void        jaiCondBroadcast(JaiCond *c);
int64_t     jaiAtomicAddI64(volatile int64_t *p, int64_t delta);
bool        jaiAtomicCasI64(volatile int64_t *p, int64_t expect, int64_t desired);

/* A fixed-size work-stealing pool used by std.thread's parallel map/reduce.
 * Tasks must not touch VM state; they operate on raw buffers. */
typedef void (*JaiTaskFn)(void *arg, int index);
bool jaiParallelFor(int start, int end, JaiTaskFn fn, void *arg, int maxThreads);

/* ------------------------------------------------------------------ */
/* Process and filesystem                                              */
/* ------------------------------------------------------------------ */

int   jaiProcessRun(const char *command, char **outStdout, size_t *outLen);

/* What jaiProcessSpawn hands back. In the waiting form only `exitCode`, the
 * two captured buffers and their lengths are meaningful; in the streaming form
 * only `pid` and the three descriptors are. */
typedef struct {
    int    pid;
    int    exitCode;
    int    stdinFd, stdoutFd, stderrFd;   /* -1 unless streaming */
    char  *out;  size_t outLen;           /* NUL-terminated, may hold NULs */
    char  *err;  size_t errLen;
} JaiSpawnResult;

/* Reasons a spawn did not produce a child. `JAI_SPAWN_EXEC` means the fork
 * succeeded and the exec did not, so `outErrno` is the child's errno — which
 * is how "no such program" is told apart from "out of file descriptors". */
typedef enum {
    JAI_SPAWN_OK = 0,
    JAI_SPAWN_SETUP,     /* pipe/fork failed in the parent */
    JAI_SPAWN_EXEC,      /* the child could not exec argv[0] */
    JAI_SPAWN_IO,        /* a pipe read or the wait failed */
} JaiSpawnStatus;

/* Start `argv` (NULL-terminated, argv[0] searched on PATH when it holds no
 * '/') and either wait for it or hand back its pipes.
 *
 *   cwd     directory to run in, or NULL to inherit
 *   envp    NULL-terminated "K=V" list to replace the environment, or NULL
 *   stdinText/stdinLen  written to the child, then its stdin is closed
 *   stream  false: wait, capture both output streams into the result;
 *           true:  return at once with pid and the three descriptors
 *
 * On JAI_SPAWN_OK in the waiting form the caller owns `out` and `err` and
 * frees each with JAI_FREE_ARRAY(char, p, len + 1). */
JaiSpawnStatus jaiProcessSpawn(const char *const *argv, const char *cwd,
                               const char *const *envp,
                               const char *stdinText, size_t stdinLen,
                               bool stream, JaiSpawnResult *result,
                               int *outErrno);

/* Wait for `pid`. `block` false polls: it answers false with *outExit
 * untouched while the child is still running. Returns false on error too, so
 * the caller distinguishes by errno being set. */
bool jaiProcessWait(int pid, bool block, int *outExit);

/* Send `sig` to `pid`. False when it could not be delivered. */
bool jaiProcessSignal(int pid, int sig);
char **jaiListDir(const char *path, int *outCount);   /* caller frees */
bool  jaiStatPath(const char *path, int64_t *size, int64_t *mtime, bool *isDir);
const char *jaiExecutablePath(void);

#ifdef __cplusplus
}
#endif

#endif /* JAI_NATIVE_H */
