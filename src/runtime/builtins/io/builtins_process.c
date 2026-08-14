/* builtins_process.c — child processes: os_spawn and the record it builds
 * (spec Appendix C).
 *
 * Split out of builtins_io.c, which explains what else moved and why. This
 * piece is self-contained: os_spawn's own argument marshalling (CStrVec,
 * EnvVec), the wait/poll/signal callables bound to a pid, and the piped
 * stdin/stdout/stderr `file`s a streaming spawn hands back. The one thing it
 * borrows from elsewhere is jaiIODictPut (builtins_io.h), for the same reason
 * io_stat and gc_stats do: every primitive that answers with a record builds
 * it the same way.
 */

/* Feature macros must precede every include: fdopen and close reach further
 * than C11, and _DARWIN_C_SOURCE puts back what asking for POSIX takes away
 * on macOS. */
#if !defined(_POSIX_C_SOURCE)
#  define _POSIX_C_SOURCE 200809L
#endif
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#  define _DARWIN_C_SOURCE
#endif

#include <errno.h>
#include <unistd.h>

#include "runtime/runtime.h"
#include "runtime/builtins/io/builtins_io.h"

#include "native/native.h"
#include "vm/gc.h"

/* --- os_spawn ------------------------------------------------------ */

typedef struct {
    const char **items;
    int          count;      /* entries, not counting the NULL terminator */
} CStrVec;

static void cstrVecFree(CStrVec *v) {
    if (v->items != NULL) JAI_FREE_ARRAY(const char *, v->items, v->count + 1);
    v->items = NULL;
    v->count = 0;
}

static bool checkExecText(ObjString *s, const char *what) {
    if (memchr(s->chars, '\0', s->length) != NULL)
        return jaiThrow(vm.cValueError, "os_spawn(): %s contains a NUL byte", what);
    return true;
}

static bool argvFromList(ObjList *list, CStrVec *out) {
    if (list->count == 0)
        return jaiThrow(vm.cValueError,
                        "os_spawn(): the command must name a program to run");

    out->items = JAI_ALLOC(const char *, (size_t)list->count + 1);
    out->count = list->count;
    for (int i = 0; i < list->count; i++) {
        if (!IS_STRING(list->items[i])) {
            cstrVecFree(out);
            return jaiThrow(vm.cTypeError,
                            "os_spawn(): argument %d is '%s', not 'str'",
                            i, jaiTypeNameStatic(list->items[i]));
        }
        ObjString *s = AS_STRING(list->items[i]);
        if (!checkExecText(s, "an argument")) { cstrVecFree(out); return false; }
        out->items[i] = s->chars;
    }
    out->items[list->count] = NULL;
    return true;
}

typedef struct {
    CStrVec  vec;
    char   **owned;
    size_t  *ownedSizes;
    int      ownedCount;
} EnvVec;

static void envVecFree(EnvVec *e) {
    for (int i = 0; i < e->ownedCount; i++)
        JAI_FREE_ARRAY(char, e->owned[i], e->ownedSizes[i]);
    if (e->owned != NULL) JAI_FREE_ARRAY(char *, e->owned, e->vec.count);
    if (e->ownedSizes != NULL) JAI_FREE_ARRAY(size_t, e->ownedSizes, e->vec.count);
    cstrVecFree(&e->vec);
    e->owned = NULL;
    e->ownedSizes = NULL;
    e->ownedCount = 0;
}

static bool envFromDict(ObjDict *dict, EnvVec *out) {
    ObjList *items = jaiDictItems(dict);
    if (items == NULL) return false;
    jaiGCPushRoot(OBJ_VAL(items));

    out->vec.items   = JAI_ALLOC(const char *, (size_t)items->count + 1);
    out->vec.count   = items->count;
    out->owned       = JAI_ALLOC(char *, (size_t)items->count);
    out->ownedSizes  = JAI_ALLOC(size_t, (size_t)items->count);
    out->ownedCount  = 0;

    bool ok = true;
    for (int i = 0; i < items->count && ok; i++) {
        ObjTuple *pair = IS_TUPLE(items->items[i]) ? AS_TUPLE(items->items[i]) : NULL;
        if (pair == NULL || pair->count != 2 ||
            !IS_STRING(pair->items[0]) || !IS_STRING(pair->items[1])) {
            ok = jaiThrow(vm.cTypeError,
                          "os_spawn(): the environment must be dict[str, str]");
            break;
        }
        ObjString *key = AS_STRING(pair->items[0]);
        ObjString *val = AS_STRING(pair->items[1]);
        if (!checkExecText(key, "an environment name") ||
            !checkExecText(val, "an environment value")) { ok = false; break; }
        if (memchr(key->chars, '=', key->length) != NULL) {
            ok = jaiThrow(vm.cValueError,
                          "os_spawn(): the environment name '%s' contains '='",
                          key->chars);
            break;
        }
        size_t size = (size_t)key->length + 1 + (size_t)val->length + 1;
        char *entry = JAI_ALLOC(char, size);
        memcpy(entry, key->chars, key->length);
        entry[key->length] = '=';
        memcpy(entry + key->length + 1, val->chars, val->length);
        entry[size - 1] = '\0';
        out->owned[out->ownedCount] = entry;
        out->ownedSizes[out->ownedCount] = size;
        out->vec.items[out->ownedCount] = entry;
        out->ownedCount++;
    }
    out->vec.items[out->ownedCount] = NULL;

    jaiGCPopRoot();
    if (!ok) envVecFree(out);
    return ok;
}

static bool spawnOption(ObjDict *options, const char *key, Value *out) {
    *out = NULL_VAL;
    if (options == NULL) return false;
    ObjString *name = jaiStringInternC(key);
    return jaiDictGet(options, OBJ_VAL(name), out) && !IS_NULL(*out);
}

static bool nOsWait(int argc, Value *args, Value *out) {
    (void)argc;
    int64_t pid;
    if (!jaiArgInt(args[0], 1, "os_wait", &pid)) return false;
    int status = -1;
    errno = 0;
    if (!jaiProcessWait((int)pid, true, &status))
        return jaiThrow(vm.cOSError, "wait(): cannot wait for process %lld: %s",
                        (long long)pid, strerror(errno));
    *out = INT_VAL(status);
    return true;
}

static bool nOsPoll(int argc, Value *args, Value *out) {
    (void)argc;
    int64_t pid;
    if (!jaiArgInt(args[0], 1, "os_poll", &pid)) return false;
    int status = -1;
    errno = 0;
    if (jaiProcessWait((int)pid, false, &status)) {
        *out = INT_VAL(status);
        return true;
    }
    if (errno != 0)
        return jaiThrow(vm.cOSError, "poll(): cannot poll process %lld: %s",
                        (long long)pid, strerror(errno));
    *out = NULL_VAL;
    return true;
}

static bool nOsSignal(int argc, Value *args, Value *out) {
    (void)argc;
    int64_t pid, sig;
    if (!jaiArgInt(args[0], 1, "os_signal", &pid)) return false;
    if (!jaiArgInt(args[1], 2, "os_signal", &sig)) return false;
    errno = 0;
    if (!jaiProcessSignal((int)pid, (int)sig) && errno != ESRCH)
        return jaiThrow(vm.cOSError, "signal(): cannot signal process %lld: %s",
                        (long long)pid, strerror(errno));
    *out = NULL_VAL;
    return true;
}

static Value boundToPid(int pid, JaiNativeFn fn, const char *name,
                        int minArity, int maxArity) {
    ObjNative *native = jaiNativeNew(fn, name, minArity, maxArity, NULL);
    jaiGCPushRoot(OBJ_VAL(native));
    ObjBound *bound = jaiBoundNew(INT_VAL(pid), OBJ_VAL(native));
    jaiGCPopRoot();
    return OBJ_VAL(bound);
}

static ObjFile *fileFromFd(int fd, const char *mode, const char *label) {
    FILE *stream = fdopen(fd, mode);
    if (stream == NULL) { (void)close(fd); return NULL; }
    ObjString *name = jaiStringInternC(label);
    jaiGCPushRoot(OBJ_VAL(name));
    ObjFile *file = jaiFileNew(stream, name, mode);
    jaiGCPopRoot();
    return file;
}

static bool spawnStreamResult(const JaiSpawnResult *spawned, Value *out) {
    const int fd[3] = { spawned->stdinFd, spawned->stdoutFd, spawned->stderrFd };
    static const char *const mode[3] = { "w", "r", "r" };
    static const char *const label[3] = { "<stdin>", "<stdout>", "<stderr>" };
    ObjFile *file[3] = { NULL, NULL, NULL };

    int wrapped = 0;
    while (wrapped < 3) {
        file[wrapped] = fileFromFd(fd[wrapped], mode[wrapped], label[wrapped]);
        if (file[wrapped] == NULL) break;
        jaiGCPushRoot(OBJ_VAL(file[wrapped]));
        wrapped++;
    }
    if (wrapped < 3) {
        jaiGCPopRoots(wrapped);
        for (int i = wrapped + 1; i < 3; i++) (void)close(fd[i]);
        return jaiThrow(vm.cOSError, "os_spawn(): cannot open the child's pipes");
    }

    ObjFile *toChild = file[0], *fromChild = file[1], *fromErr = file[2];

    ObjDict *result = jaiDictNew();
    jaiGCPushRoot(OBJ_VAL(result));
    jaiIODictPut(result, "pid", INT_VAL(spawned->pid));
    jaiIODictPut(result, "stdin", OBJ_VAL(toChild));
    jaiIODictPut(result, "stdout", OBJ_VAL(fromChild));
    jaiIODictPut(result, "stderr", OBJ_VAL(fromErr));
    jaiIODictPut(result, "wait",
            boundToPid(spawned->pid, nOsWait, "wait", 1, 1));
    jaiIODictPut(result, "poll",
            boundToPid(spawned->pid, nOsPoll, "poll", 1, 1));
    jaiIODictPut(result, "signal",
            boundToPid(spawned->pid, nOsSignal, "signal", 2, 2));
    jaiGCPopRoots(4);

    *out = OBJ_VAL(result);
    return true;
}

static bool nOsSpawn(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *command;
    if (!jaiArgList(args[0], 1, "os_spawn", &command)) return false;
    if (!IS_DICT(args[1]))
        return jaiThrow(vm.cTypeError, "os_spawn(): the options must be a dict, not '%s'",
                        jaiTypeNameStatic(args[1]));
    ObjDict *options = AS_DICT(args[1]);

    CStrVec argv = { NULL, 0 };
    if (!argvFromList(command, &argv)) return false;

    const char *cwd = NULL;
    Value option;
    if (spawnOption(options, "cwd", &option)) {
        if (!IS_STRING(option)) {
            cstrVecFree(&argv);
            return jaiThrow(vm.cTypeError, "os_spawn(): `cwd` must be a str");
        }
        if (!checkExecText(AS_STRING(option), "the working directory")) {
            cstrVecFree(&argv);
            return false;
        }
        cwd = AS_STRING(option)->chars;
    }

    EnvVec env = { { NULL, 0 }, NULL, NULL, 0 };
    bool hasEnv = false;
    if (spawnOption(options, "env", &option)) {
        if (!IS_DICT(option)) {
            cstrVecFree(&argv);
            return jaiThrow(vm.cTypeError, "os_spawn(): `env` must be a dict");
        }
        if (!envFromDict(AS_DICT(option), &env)) { cstrVecFree(&argv); return false; }
        hasEnv = true;
    }

    const char *stdinText = NULL;
    size_t stdinLen = 0;
    if (spawnOption(options, "stdin", &option)) {
        if (!IS_STRING(option)) {
            cstrVecFree(&argv);
            if (hasEnv) envVecFree(&env);
            return jaiThrow(vm.cTypeError, "os_spawn(): `stdin` must be a str");
        }
        stdinText = AS_STRING(option)->chars;
        stdinLen = AS_STRING(option)->length;
    }

    bool stream = spawnOption(options, "stream", &option) && IS_BOOL(option) &&
                  AS_BOOL(option);

    JaiSpawnResult spawned;
    int failure = 0;
    JaiSpawnStatus status = jaiProcessSpawn(argv.items, cwd,
                                            hasEnv ? env.vec.items : NULL,
                                            stdinText, stdinLen, stream,
                                            &spawned, &failure);
    const char *program = argv.items[0];
    char programCopy[512];
    snprintf(programCopy, sizeof programCopy, "%s", program);
    cstrVecFree(&argv);
    if (hasEnv) envVecFree(&env);

    if (status == JAI_SPAWN_EXEC) {
        if (failure == ENOENT || failure == ENOTDIR)
            return jaiThrow(vm.cFileNotFoundError,
                            "os_spawn(): no such program: %s", programCopy);
        return jaiThrow(vm.cOSError, "os_spawn(): cannot run %s: %s",
                        programCopy, strerror(failure));
    }
    if (status != JAI_SPAWN_OK) {
        return jaiThrow(vm.cOSError, "os_spawn(): cannot run %s: %s",
                        programCopy,
                        failure != 0 ? strerror(failure) : "the child failed");
    }

    if (stream) return spawnStreamResult(&spawned, out);

    ObjString *outText = jaiStringTake(spawned.out, spawned.outLen);
    if (outText == NULL) return false;
    jaiGCPushRoot(OBJ_VAL(outText));
    ObjString *errText = jaiStringTake(spawned.err, spawned.errLen);
    if (errText == NULL) { jaiGCPopRoot(); return false; }
    jaiGCPushRoot(OBJ_VAL(errText));

    ObjDict *result = jaiDictNew();
    jaiGCPushRoot(OBJ_VAL(result));
    jaiIODictPut(result, "exit_code", INT_VAL(spawned.exitCode));
    jaiIODictPut(result, "stdout", OBJ_VAL(outText));
    jaiIODictPut(result, "stderr", OBJ_VAL(errText));
    jaiGCPopRoots(3);

    *out = OBJ_VAL(result);
    return true;
}

void jaiRegisterProcessPrimitives(void) {
    jaiDefineNative("__prim__.os_spawn", nOsSpawn, 2, 2);
}
