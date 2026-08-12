/* thread.c — OS threads, mutexes, condition variables, atomics, and the
 * parallel-for pool behind std.thread. The VM is single-threaded: workers must
 * not touch VM state, allocate GC objects, or raise — tasks see only raw buffers, so failures are a status code, not a diagnostic. */

/* Feature macros must precede every include: sysconf's _SC_NPROCESSORS_ONLN is
 * not in the C11 headers on its own. */
#if !defined(_POSIX_C_SOURCE)
#  define _POSIX_C_SOURCE 200809L
#endif
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#  define _DARWIN_C_SOURCE
#endif

#include "native.h"

#include <errno.h>

#include <limits.h>
#include <pthread.h>
#include <unistd.h>

#if defined(__APPLE__)
#  include <sys/sysctl.h>
#endif

/* One of these three is always defined; the builtins are the only path taken
 * on the compilers Jaithon actually builds with. */
#if defined(__GNUC__) || defined(__clang__)
#  define JAI_ATOMIC_BUILTIN 1
#elif !defined(__STDC_NO_ATOMICS__)
#  include <stdatomic.h>
#  define JAI_ATOMIC_C11 1
#else
#  define JAI_ATOMIC_LOCKED 1
#endif

int jaiCpuCount(void) {
#if defined(_SC_NPROCESSORS_ONLN)
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n > 0) return n > (long)INT_MAX ? INT_MAX : (int)n;
#endif
#if defined(__APPLE__)
    /* sysconf can report -1 inside a sandbox where sysctl still answers. */
    int    count = 0;
    size_t size  = sizeof count;
    if (sysctlbyname("hw.logicalcpu", &count, &size, NULL, 0) == 0 && count > 0) {
        return count;
    }
#endif
    return 1;
}

/* Join and detach are the two disposal paths; both release the handle, so a
 * JaiThread* is dead after either one. There is no separate free. */
struct JaiThread {
    pthread_t handle;
};

JaiThread *jaiThreadSpawn(void *(*fn)(void *), void *arg) {
    if (fn == NULL) return NULL;

    JaiThread *t = JAI_ALLOC(JaiThread, 1);
    if (pthread_create(&t->handle, NULL, fn, arg) != 0) {
        JAI_FREE(JaiThread, t);
        return NULL;   /* EAGAIN: out of threads, not out of memory */
    }
    return t;
}

bool jaiThreadJoin(JaiThread *t, void **outResult) {
    if (outResult != NULL) *outResult = NULL;
    if (t == NULL) return false;

    void *result = NULL;
    int   err    = pthread_join(t->handle, &result);

    /* A failed join is EINVAL or ESRCH and neither becomes joinable later, so
     * the handle is released on both paths. */
    JAI_FREE(JaiThread, t);
    if (err != 0) return false;

    if (outResult != NULL) *outResult = result;
    return true;
}

void jaiThreadDetach(JaiThread *t) {
    if (t == NULL) return;
    pthread_detach(t->handle);
    JAI_FREE(JaiThread, t);
}

struct JaiMutex {
    pthread_mutex_t handle;
};

struct JaiCond {
    pthread_cond_t handle;
};

JaiMutex *jaiMutexNew(void) {
    JaiMutex *m = JAI_ALLOC(JaiMutex, 1);
    if (pthread_mutex_init(&m->handle, NULL) != 0) {
        JAI_FREE(JaiMutex, m);
        return NULL;
    }
    return m;
}

void jaiMutexFree(JaiMutex *m) {
    if (m == NULL) return;
    pthread_mutex_destroy(&m->handle);
    JAI_FREE(JaiMutex, m);
}

/* Failure here would otherwise silently corrupt the guarded state, so it's
 * fatal, not ignored — the only causes (destroyed mutex, self-deadlock) are caller bugs. */
void jaiMutexLock(JaiMutex *m) {
    if (m == NULL) return;
    int err = pthread_mutex_lock(&m->handle);
    if (JAI_UNLIKELY(err != 0)) JAI_PANIC("mutex lock failed: %s", strerror(err));
}

bool jaiMutexTryLock(JaiMutex *m) {
    if (m == NULL) return false;
    int err = pthread_mutex_trylock(&m->handle);
    if (err == 0) return true;
    if (err == EBUSY) return false;
    JAI_PANIC("mutex trylock failed: %s", strerror(err));
}

void jaiMutexUnlock(JaiMutex *m) {
    if (m == NULL) return;
    int err = pthread_mutex_unlock(&m->handle);
    if (JAI_UNLIKELY(err != 0)) JAI_PANIC("mutex unlock failed: %s", strerror(err));
}

JaiCond *jaiCondNew(void) {
    JaiCond *c = JAI_ALLOC(JaiCond, 1);
    if (pthread_cond_init(&c->handle, NULL) != 0) {
        JAI_FREE(JaiCond, c);
        return NULL;
    }
    return c;
}

void jaiCondFree(JaiCond *c) {
    if (c == NULL) return;
    pthread_cond_destroy(&c->handle);
    JAI_FREE(JaiCond, c);
}

/* Spurious wakeups are permitted: callers must re-test their predicate. */
void jaiCondWait(JaiCond *c, JaiMutex *m) {
    if (c == NULL || m == NULL) return;
    int err = pthread_cond_wait(&c->handle, &m->handle);
    if (JAI_UNLIKELY(err != 0)) JAI_PANIC("condition wait failed: %s", strerror(err));
}

void jaiCondSignal(JaiCond *c) {
    if (c == NULL) return;
    pthread_cond_signal(&c->handle);
}

void jaiCondBroadcast(JaiCond *c) {
    if (c == NULL) return;
    pthread_cond_broadcast(&c->handle);
}

#if defined(JAI_ATOMIC_LOCKED)
/* No atomics from the compiler: one global lock serialises every operation.
 * Correct, and slow enough that std.thread's counters stay honest. */
static pthread_mutex_t gAtomicLock = PTHREAD_MUTEX_INITIALIZER;
#endif

/* Returns the value *before* the add, matching C11 atomic_fetch_add. That is
 * what a claim-the-next-chunk cursor needs; the new value is `result + delta`. */
int64_t jaiAtomicAddI64(volatile int64_t *p, int64_t delta) {
    if (p == NULL) return 0;

#if defined(JAI_ATOMIC_BUILTIN)
    return __atomic_fetch_add(p, delta, __ATOMIC_SEQ_CST);
#elif defined(JAI_ATOMIC_C11)
    return atomic_fetch_add((_Atomic int64_t *)(int64_t *)p, delta);
#else
    pthread_mutex_lock(&gAtomicLock);
    int64_t old = *p;
    *p = old + delta;
    pthread_mutex_unlock(&gAtomicLock);
    return old;
#endif
}

/* Strong compare-and-swap: no spurious failures, so callers may test the
 * result directly instead of looping on it. */
bool jaiAtomicCasI64(volatile int64_t *p, int64_t expect, int64_t desired) {
    if (p == NULL) return false;

#if defined(JAI_ATOMIC_BUILTIN)
    return __atomic_compare_exchange_n(p, &expect, desired, false, __ATOMIC_SEQ_CST,
                                       __ATOMIC_SEQ_CST);
#elif defined(JAI_ATOMIC_C11)
    return atomic_compare_exchange_strong((_Atomic int64_t *)(int64_t *)p, &expect,
                                          desired);
#else
    pthread_mutex_lock(&gAtomicLock);
    bool swapped = (*p == expect);
    if (swapped) *p = desired;
    pthread_mutex_unlock(&gAtomicLock);
    return swapped;
#endif
}

/* Below this many iterations the spawn and join cost more than the body. */
#define JAI_PARALLEL_MIN_ITERS   1024
#define JAI_PARALLEL_MAX_THREADS 64
/* Chunks per thread: more than one lets a fast worker take over for a slow
 * one when the body's cost varies with the index; too many and the shared cursor becomes the bottleneck. */
#define JAI_PARALLEL_OVERSUBSCRIBE 4

typedef struct {
    volatile int64_t cursor;   /* first unclaimed offset; the only shared write */
    int64_t          total;
    int64_t          chunk;
    int              start;
    JaiTaskFn        fn;
    void            *arg;
} ParallelJob;

/* Offsets are tracked as int64 because end - start can exceed INT_MAX even
 * though every individual index fits in an int. */
static void parallelSerial(int start, int64_t total, JaiTaskFn fn, void *arg) {
    for (int64_t i = 0; i < total; i++) fn(arg, (int)((int64_t)start + i));
}

/* Each worker repeatedly claims a chunk with one fetch-add; whoever finishes
 * first simply claims more, which is what balances an uneven body. */
static void parallelRun(ParallelJob *job) {
    for (;;) {
        int64_t from = jaiAtomicAddI64(&job->cursor, job->chunk);
        if (from >= job->total) return;

        int64_t to = from + job->chunk;
        if (to > job->total) to = job->total;
        for (int64_t i = from; i < to; i++) {
            job->fn(job->arg, (int)((int64_t)job->start + i));
        }
    }
}

static void *parallelWorker(void *arg) {
    parallelRun((ParallelJob *)arg);
    return NULL;
}

bool jaiParallelFor(int start, int end, JaiTaskFn fn, void *arg, int maxThreads) {
    if (fn == NULL) return false;
    if (end <= start) return true;   /* an empty range is vacuously done */

    int64_t total = (int64_t)end - (int64_t)start;

    if (maxThreads <= 1 || total < JAI_PARALLEL_MIN_ITERS) {
        parallelSerial(start, total, fn, arg);
        return true;
    }

    int threads = maxThreads;
    int hw      = jaiCpuCount();
    if (threads > hw) threads = hw;
    if (threads > JAI_PARALLEL_MAX_THREADS) threads = JAI_PARALLEL_MAX_THREADS;

    int64_t chunk = total / ((int64_t)threads * JAI_PARALLEL_OVERSUBSCRIBE);
    if (chunk < 1) chunk = 1;

    /* Threads beyond the chunk count would only spawn and exit. */
    int64_t chunks = (total + chunk - 1) / chunk;
    if ((int64_t)threads > chunks) threads = (int)chunks;
    if (threads <= 1) {
        parallelSerial(start, total, fn, arg);
        return true;
    }

    ParallelJob job;
    job.cursor = 0;
    job.total  = total;
    job.chunk  = chunk;
    job.start  = start;
    job.fn     = fn;
    job.arg    = arg;

    /* `job` lives on this frame and every worker points at it, so this function
     * must not return until all of them are joined. */
    JaiThread **workers = JAI_ALLOC(JaiThread *, threads - 1);
    int spawned = 0;
    for (int i = 0; i < threads - 1; i++) {
        JaiThread *t = jaiThreadSpawn(parallelWorker, &job);
        /* A refused spawn costs throughput, not correctness: the remaining
         * workers pull the chunks it would have taken. */
        if (t == NULL) break;
        workers[spawned++] = t;
    }

    parallelRun(&job);   /* the caller is a worker, not a supervisor */

    bool ok = true;
    for (int i = 0; i < spawned; i++) {
        if (!jaiThreadJoin(workers[i], NULL)) ok = false;
    }
    JAI_FREE_ARRAY(JaiThread *, workers, threads - 1);
    return ok;
}
