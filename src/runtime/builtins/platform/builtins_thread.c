/* builtins_thread.c — threads, mutexes, condition variables and atomics
 * (spec Appendix C), plus cpu_count.
 *
 * Split out of builtins_io.c, which explains what else moved and why.
 * cpu_count is a Host fact, not a thread primitive, but it has always been
 * registered here rather than with the filesystem/environment group in
 * builtins_fs.c: the one thing that number is for is sizing a thread pool,
 * so its registration stayed next to what reads it, and the move kept that
 * call site intact rather than routing it through a cross-file declaration
 * for a two-line function.
 *
 * A spawned thread must not touch VM state. Nothing in the VM is
 * synchronised: the allocator keeps one byte counter, the collector
 * stops the world it can see, the intern table is a plain hash table,
 * and every Value is a pointer into a heap only the collector may
 * move through. A second thread inside any of that is a data race, not
 * a slow program.
 *
 * So this is not a documented rule, it is an enforced one: thread_spawn
 * refuses anything but a native primitive, and the worker receives two
 * plain integers — the address and the length of a private copy of the
 * caller's buffer. No Value it is handed refers to the heap, and only
 * an int comes back. A Jaithon closure cannot be passed at all, which
 * is the whole point: there is no way to express the unsafe thing.
 *
 * The thread, mutex, condition and atomic handles below come from the
 * shared table in handles.c.
 */

#include "runtime/runtime.h"
#include "runtime/handles.h"

#include "native/native.h"

typedef struct {
    JaiThread       *thread;
    JaiNativeFn      fn;
    uint8_t         *payload;      /* private copy, off the GC heap */
    size_t           payloadLength;
    int64_t          result;
    volatile int64_t finished;     /* the worker's only word of shared state */
} ThreadTask;

/* Detached tasks nobody will ever join. They are reclaimed the next time a
 * thread primitive runs and finds one whose worker has finished, which bounds
 * the outstanding memory by the number of threads still running. */
static ThreadTask **gDetached;
static int           gDetachedCount;
static int           gDetachedCapacity;

static void freeTask(ThreadTask *task) {
    if (task->payload != NULL)
        JAI_FREE_ARRAY(uint8_t, task->payload, task->payloadLength);
    JAI_FREE(ThreadTask, task);
}

static void reapDetached(void) {
    int kept = 0;
    for (int i = 0; i < gDetachedCount; i++) {
        ThreadTask *task = gDetached[i];
        /* An atomic add of zero is an atomic read of the flag. */
        if (jaiAtomicAddI64(&task->finished, 0) != 0) {
            freeTask(task);
            continue;
        }
        gDetached[kept++] = task;
    }
    gDetachedCount = kept;
}

static void rememberDetached(ThreadTask *task) {
    if (gDetachedCount + 1 > gDetachedCapacity) {
        int oldCapacity = gDetachedCapacity;
        gDetachedCapacity = JAI_GROW_CAP(oldCapacity);
        gDetached = JAI_GROW_ARRAY(ThreadTask *, gDetached, oldCapacity,
                                   gDetachedCapacity);
    }
    gDetached[gDetachedCount++] = task;
}

/* Runs on the worker thread. Everything it touches is either on its own stack
 * or in the private payload; the two Values it builds are integers, so no
 * heap object is read, written or allocated. */
static void *threadEntry(void *arg) {
    ThreadTask *task = (ThreadTask *)arg;
    Value args[2] = {
        INT_VAL((int64_t)(uintptr_t)task->payload),
        INT_VAL((int64_t)task->payloadLength),
    };
    Value result = NULL_VAL;
    if (task->fn(2, args, &result) && IS_INT(result)) task->result = AS_INT(result);
    (void)jaiAtomicAddI64(&task->finished, 1);
    return NULL;
}

static bool nThreadSpawn(int argc, Value *args, Value *out) {
    if (!IS_NATIVE(args[0]))
        return jaiThrow(vm.cTypeError,
                        "thread_spawn(): the worker must be a native primitive, "
                        "not %s; a Jaithon function cannot run off the VM thread",
                        jaiTypeNameStatic(args[0]));

    ObjNative *worker = AS_NATIVE(args[0]);
    if (worker->minArity > 2 || (worker->maxArity >= 0 && worker->maxArity < 2))
        return jaiThrow(vm.cTypeError,
                        "thread_spawn(): the worker '%s' must accept two "
                        "arguments, the buffer address and its length",
                        worker->name != NULL ? worker->name->chars : "?");

    const uint8_t *data = NULL;
    size_t length = 0;
    if (argc >= 2 && !IS_NULL(args[1])) {
        if (IS_BYTES(args[1])) {
            data = AS_BYTES(args[1])->data;
            length = AS_BYTES(args[1])->length;
        } else if (IS_STRING(args[1])) {
            data = (const uint8_t *)AS_STRING(args[1])->chars;
            length = AS_STRING(args[1])->length;
        } else {
            return jaiThrow(vm.cTypeError,
                            "thread_spawn() argument 2: expected bytes or str, "
                            "got %s", jaiTypeNameStatic(args[1]));
        }
    }

    reapDetached();

    ThreadTask *task = JAI_ALLOC(ThreadTask, 1);
    memset(task, 0, sizeof *task);
    task->fn = worker->fn;
    if (length > 0) {
        /* A copy, not the bytes object: the worker must not be able to reach
         * anything the collector owns. */
        task->payload = JAI_ALLOC(uint8_t, length);
        memcpy(task->payload, data, length);
        task->payloadLength = length;
    }

    task->thread = jaiThreadSpawn(threadEntry, task);
    if (task->thread == NULL) {
        freeTask(task);
        return jaiThrow(vm.cOSError,
                        "thread_spawn(): the operating system refused to start "
                        "a thread");
    }

    *out = INT_VAL(jaiHandleAdd(HANDLE_THREAD, task));
    return true;
}

static bool nThreadJoin(int argc, Value *args, Value *out) {
    (void)argc;
    void *ptr;
    if (!jaiHandleGet(args[0], 1, HANDLE_THREAD, "thread_join", &ptr)) return false;
    ThreadTask *task = (ThreadTask *)ptr;

    if (!jaiThreadJoin(task->thread, NULL))
        return jaiThrow(vm.cOSError, "thread_join(): the thread cannot be joined");

    int64_t result = task->result;
    jaiHandleRelease(AS_INT(args[0]));
    freeTask(task);
    reapDetached();

    *out = INT_VAL(result);
    return true;
}

static bool nThreadDetach(int argc, Value *args, Value *out) {
    (void)argc;
    void *ptr;
    if (!jaiHandleGet(args[0], 1, HANDLE_THREAD, "thread_detach", &ptr)) return false;
    ThreadTask *task = (ThreadTask *)ptr;

    jaiThreadDetach(task->thread);
    jaiHandleRelease(AS_INT(args[0]));
    /* The worker still owns its payload, so the record outlives the handle. */
    rememberDetached(task);
    reapDetached();

    *out = NULL_VAL;
    return true;
}

/* ------------------------------------------------------------------ */
/* Mutexes and condition variables                                      */
/* ------------------------------------------------------------------ */

static bool nMutexNew(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    JaiMutex *mutex = jaiMutexNew();
    if (mutex == NULL)
        return jaiThrow(vm.cOSError, "mutex_new(): cannot create a mutex");
    *out = INT_VAL(jaiHandleAdd(HANDLE_MUTEX, mutex));
    return true;
}

static bool nMutexLock(int argc, Value *args, Value *out) {
    (void)argc;
    void *ptr;
    if (!jaiHandleGet(args[0], 1, HANDLE_MUTEX, "mutex_lock", &ptr)) return false;
    jaiMutexLock((JaiMutex *)ptr);
    *out = NULL_VAL;
    return true;
}

static bool nMutexTryLock(int argc, Value *args, Value *out) {
    (void)argc;
    void *ptr;
    if (!jaiHandleGet(args[0], 1, HANDLE_MUTEX, "mutex_try_lock", &ptr)) return false;
    *out = BOOL_VAL(jaiMutexTryLock((JaiMutex *)ptr));
    return true;
}

static bool nMutexUnlock(int argc, Value *args, Value *out) {
    (void)argc;
    void *ptr;
    if (!jaiHandleGet(args[0], 1, HANDLE_MUTEX, "mutex_unlock", &ptr)) return false;
    jaiMutexUnlock((JaiMutex *)ptr);
    *out = NULL_VAL;
    return true;
}

/* Freeing a mutex another thread is waiting on is undefined, so the handle is
 * released first: a later use is then a clean "not a live handle" error rather
 * than a use-after-free. */
static bool nMutexFree(int argc, Value *args, Value *out) {
    (void)argc;
    void *ptr;
    if (!jaiHandleGet(args[0], 1, HANDLE_MUTEX, "mutex_free", &ptr)) return false;
    jaiHandleRelease(AS_INT(args[0]));
    jaiMutexFree((JaiMutex *)ptr);
    *out = NULL_VAL;
    return true;
}

static bool nCondNew(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    JaiCond *cond = jaiCondNew();
    if (cond == NULL)
        return jaiThrow(vm.cOSError, "cond_new(): cannot create a condition variable");
    *out = INT_VAL(jaiHandleAdd(HANDLE_COND, cond));
    return true;
}

/* The mutex must be held by this thread; it is released while waiting and
 * reacquired before the call returns, as the underlying primitive requires. */
static bool nCondWait(int argc, Value *args, Value *out) {
    (void)argc;
    void *condPtr, *mutexPtr;
    if (!jaiHandleGet(args[0], 1, HANDLE_COND, "cond_wait", &condPtr)) return false;
    if (!jaiHandleGet(args[1], 2, HANDLE_MUTEX, "cond_wait", &mutexPtr)) return false;
    jaiCondWait((JaiCond *)condPtr, (JaiMutex *)mutexPtr);
    *out = NULL_VAL;
    return true;
}

static bool nCondSignal(int argc, Value *args, Value *out) {
    (void)argc;
    void *ptr;
    if (!jaiHandleGet(args[0], 1, HANDLE_COND, "cond_signal", &ptr)) return false;
    jaiCondSignal((JaiCond *)ptr);
    *out = NULL_VAL;
    return true;
}

static bool nCondBroadcast(int argc, Value *args, Value *out) {
    (void)argc;
    void *ptr;
    if (!jaiHandleGet(args[0], 1, HANDLE_COND, "cond_broadcast", &ptr)) return false;
    jaiCondBroadcast((JaiCond *)ptr);
    *out = NULL_VAL;
    return true;
}

static bool nCondFree(int argc, Value *args, Value *out) {
    (void)argc;
    void *ptr;
    if (!jaiHandleGet(args[0], 1, HANDLE_COND, "cond_free", &ptr)) return false;
    jaiHandleRelease(AS_INT(args[0]));
    jaiCondFree((JaiCond *)ptr);
    *out = NULL_VAL;
    return true;
}

/* ------------------------------------------------------------------ */
/* Atomics                                                              */
/* ------------------------------------------------------------------ */

/* An atomic cell is one int64 outside the GC heap, so a worker thread can
 * update it without any collector interaction. */
static bool nAtomicNew(int argc, Value *args, Value *out) {
    int64_t initial = 0;
    if (argc >= 1 && !IS_NULL(args[0]) &&
        !jaiArgInt(args[0], 1, "atomic_new", &initial))
        return false;

    int64_t *cell = JAI_ALLOC(int64_t, 1);
    *cell = initial;
    *out = INT_VAL(jaiHandleAdd(HANDLE_ATOMIC, cell));
    return true;
}

static bool atomicCell(Value v, const char *fnName, volatile int64_t **out) {
    void *ptr;
    if (!jaiHandleGet(v, 1, HANDLE_ATOMIC, fnName, &ptr)) return false;
    *out = (volatile int64_t *)ptr;
    return true;
}

static bool nAtomicLoad(int argc, Value *args, Value *out) {
    (void)argc;
    volatile int64_t *cell;
    if (!atomicCell(args[0], "atomic_load", &cell)) return false;
    *out = INT_VAL(jaiAtomicAddI64(cell, 0));
    return true;
}

/* A store is a compare-and-swap loop rather than a plain write, so that a
 * concurrent add can never be lost between the read and the write. */
static bool exchangeCell(volatile int64_t *cell, int64_t value, int64_t *previous) {
    for (;;) {
        int64_t current = jaiAtomicAddI64(cell, 0);
        if (jaiAtomicCasI64(cell, current, value)) {
            *previous = current;
            return true;
        }
    }
}

static bool nAtomicStore(int argc, Value *args, Value *out) {
    (void)argc;
    volatile int64_t *cell;
    if (!atomicCell(args[0], "atomic_store", &cell)) return false;
    int64_t value;
    if (!jaiArgInt(args[1], 2, "atomic_store", &value)) return false;
    int64_t previous;
    (void)exchangeCell(cell, value, &previous);
    *out = NULL_VAL;
    return true;
}

static bool nAtomicExchange(int argc, Value *args, Value *out) {
    (void)argc;
    volatile int64_t *cell;
    if (!atomicCell(args[0], "atomic_exchange", &cell)) return false;
    int64_t value;
    if (!jaiArgInt(args[1], 2, "atomic_exchange", &value)) return false;
    int64_t previous = 0;
    (void)exchangeCell(cell, value, &previous);
    *out = INT_VAL(previous);
    return true;
}

/* Returns the value the cell held *before* the addition, as fetch-and-add
 * does everywhere else. */
static bool nAtomicAdd(int argc, Value *args, Value *out) {
    (void)argc;
    volatile int64_t *cell;
    if (!atomicCell(args[0], "atomic_add", &cell)) return false;
    int64_t delta;
    if (!jaiArgInt(args[1], 2, "atomic_add", &delta)) return false;
    *out = INT_VAL(jaiAtomicAddI64(cell, delta));
    return true;
}

static bool nAtomicCas(int argc, Value *args, Value *out) {
    (void)argc;
    volatile int64_t *cell;
    if (!atomicCell(args[0], "atomic_cas", &cell)) return false;
    int64_t expected, desired;
    if (!jaiArgInt(args[1], 2, "atomic_cas", &expected)) return false;
    if (!jaiArgInt(args[2], 3, "atomic_cas", &desired)) return false;
    *out = BOOL_VAL(jaiAtomicCasI64(cell, expected, desired));
    return true;
}

static bool nAtomicFree(int argc, Value *args, Value *out) {
    (void)argc;
    volatile int64_t *cell;
    if (!atomicCell(args[0], "atomic_free", &cell)) return false;
    jaiHandleRelease(AS_INT(args[0]));
    JAI_FREE_ARRAY(int64_t, (int64_t *)cell, 1);
    *out = NULL_VAL;
    return true;
}

/* ------------------------------------------------------------------ */
/* Host                                                                 */
/* ------------------------------------------------------------------ */

static bool nCpuCount(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    int count = jaiCpuCount();
    *out = INT_VAL(count > 0 ? count : 1);
    return true;
}

void jaiRegisterThreadPrimitives(void) {
    jaiDefineNative("__prim__.thread_spawn",  nThreadSpawn,  1, 2);
    jaiDefineNative("__prim__.thread_join",   nThreadJoin,   1, 1);
    jaiDefineNative("__prim__.thread_detach", nThreadDetach, 1, 1);

    jaiDefineNative("__prim__.mutex_new",    nMutexNew,    0, 0);
    jaiDefineNative("__prim__.mutex_lock",   nMutexLock,   1, 1);
    jaiDefineNative("__prim__.mutex_try_lock", nMutexTryLock, 1, 1);
    jaiDefineNative("__prim__.mutex_unlock", nMutexUnlock, 1, 1);
    jaiDefineNative("__prim__.mutex_free",   nMutexFree,   1, 1);

    jaiDefineNative("__prim__.cond_new",       nCondNew,       0, 0);
    jaiDefineNative("__prim__.cond_wait",      nCondWait,      2, 2);
    jaiDefineNative("__prim__.cond_signal",    nCondSignal,    1, 1);
    jaiDefineNative("__prim__.cond_broadcast", nCondBroadcast, 1, 1);
    jaiDefineNative("__prim__.cond_free",      nCondFree,      1, 1);

    jaiDefineNative("__prim__.atomic_new",      nAtomicNew,      0, 1);
    jaiDefineNative("__prim__.atomic_load",     nAtomicLoad,     1, 1);
    jaiDefineNative("__prim__.atomic_store",    nAtomicStore,    2, 2);
    jaiDefineNative("__prim__.atomic_exchange", nAtomicExchange, 2, 2);
    jaiDefineNative("__prim__.atomic_add",      nAtomicAdd,      2, 2);
    jaiDefineNative("__prim__.atomic_cas",      nAtomicCas,      3, 3);
    jaiDefineNative("__prim__.atomic_free",     nAtomicFree,     1, 1);

    jaiDefineNative("__prim__.cpu_count", nCpuCount, 0, 0);
}
