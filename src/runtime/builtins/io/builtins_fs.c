/* builtins_fs.c — the process's own environment and the filesystem it can
 * see: os_env, os_argv, os_exit, os_cwd/chdir, io_listdir/mkdir/remove/
 * rename/stat, and os_platform (spec Appendix C).
 *
 * Split out of builtins_io.c, which explains what else moved and why. What
 * belongs together here is everything that names a path or an environment
 * variable but never touches a `file` handle -- io_open and its kin stayed
 * behind for exactly the opposite reason. The errno-to-exception helpers
 * (jaiIOThrowErrno and kin), the NUL-safe path helpers, and the dict-building
 * helper are still defined in builtins_io.c and shared through builtins_io.h;
 * every syscall here that can fail on a bad path goes through them, so one
 * errno still produces one exception class and every message still names the
 * path.
 */

/* Feature macros must precede every include: setenv, getcwd and chdir are
 * POSIX, not C11, and _DARWIN_C_SOURCE puts back what asking for POSIX takes
 * away on macOS. */
#if !defined(_POSIX_C_SOURCE)
#  define _POSIX_C_SOURCE 200809L
#endif
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#  define _DARWIN_C_SOURCE
#endif

#include <errno.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "runtime/runtime.h"
#include "runtime/builtins/io/builtins_io.h"

#include "native/native.h"
#include "vm/gc.h"

/* POSIX guarantees the variable but not a declaration for it in <unistd.h>. */
extern char **environ;

/* ------------------------------------------------------------------ */
/* Environment                                                          */
/* ------------------------------------------------------------------ */

static bool checkVariableName(ObjString *name) {
    if (name->length == 0)
        return jaiThrow(vm.cValueError,
                        "os_env(): the variable name must not be empty");
    if (memchr(name->chars, '\0', name->length) != NULL ||
        memchr(name->chars, '=', name->length) != NULL)
        return jaiThrow(vm.cValueError,
                        "os_env(): the variable name may not contain '=' or a NUL byte");
    return true;
}

static bool environSnapshot(Value *out) {
    ObjDict *dict = jaiDictNew();
    jaiGCPushRoot(OBJ_VAL(dict));

    bool ok = true;
    for (char **entry = environ; ok && entry != NULL && *entry != NULL; entry++) {
        const char *equals = strchr(*entry, '=');
        if (equals == NULL) continue;    /* not a NAME=VALUE binding */
        size_t nameLen = (size_t)(equals - *entry);

        ObjString *key = jaiStringNew(*entry, nameLen);
        if (key == NULL) { ok = false; break; }
        jaiGCPushRoot(OBJ_VAL(key));
        ObjString *value = jaiStringNew(equals + 1, strlen(equals + 1));
        if (value == NULL) {
            jaiGCPopRoot();
            ok = false;
            break;
        }
        jaiGCPushRoot(OBJ_VAL(value));
        (void)jaiDictSet(dict, OBJ_VAL(key), OBJ_VAL(value));
        jaiGCPopRoots(2);
    }

    jaiGCPopRoot();
    if (!ok) return false;
    *out = OBJ_VAL(dict);
    return true;
}

/* os_env() is the whole environment, os_env(name) reads one variable, and
 * os_env(name, value) writes it — with a null value removing it. */
static bool nOsEnv(int argc, Value *args, Value *out) {
    if (argc == 0) return environSnapshot(out);

    ObjString *name;
    if (!jaiArgString(args[0], 1, "os_env", &name)) return false;
    if (!checkVariableName(name)) return false;

    if (argc == 1) {
        const char *value = getenv(name->chars);
        if (value == NULL) {
            *out = NULL_VAL;
            return true;
        }
        ObjString *text = jaiStringNew(value, strlen(value));
        if (text == NULL) return false;
        *out = OBJ_VAL(text);
        return true;
    }

    errno = 0;
    if (IS_NULL(args[1])) {
        if (unsetenv(name->chars) != 0)
            return jaiThrow(vm.cOSError, "os_env(): cannot unset '%s': %s",
                            name->chars, strerror(errno));
    } else {
        ObjString *value;
        if (!jaiArgString(args[1], 2, "os_env", &value)) return false;
        if (memchr(value->chars, '\0', value->length) != NULL)
            return jaiThrow(vm.cValueError,
                            "os_env(): the value contains a NUL byte");
        if (setenv(name->chars, value->chars, 1) != 0)
            return jaiThrow(vm.cOSError, "os_env(): cannot set '%s': %s",
                            name->chars, strerror(errno));
    }
    *out = NULL_VAL;
    return true;
}

/* The command line is not something libc hands back portably, so the CLI
 * publishes it as the builtin global `__argv__` — the same list spec §8.4
 * hands to main(). Until it does, the executable path is all there is to
 * report truthfully. */
static bool nOsArgv(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    ObjList *result = jaiListNew(0);
    jaiGCPushRoot(OBJ_VAL(result));

    Value stored;
    ObjString *key = jaiStringInternC("__argv__");
    bool published = vm.builtins != NULL &&
                     jaiTableGetInterned(&vm.builtins->globals, key, &stored) &&
                     IS_LIST(stored);
    if (published) {
        ObjList *source = AS_LIST(stored);
        for (int i = 0; i < source->count; i++) jaiListPush(result, source->items[i]);
    } else {
        const char *exe = jaiExecutablePath();
        if (exe != NULL && exe[0] != '\0') {
            ObjString *text = jaiStringNew(exe, strlen(exe));
            if (text == NULL) {
                jaiGCPopRoot();
                return false;
            }
            jaiGCPushRoot(OBJ_VAL(text));
            jaiListPush(result, OBJ_VAL(text));
            jaiGCPopRoot();
        }
    }

    jaiGCPopRoot();
    *out = OBJ_VAL(result);
    return true;
}

/* __prim__.module_path() -> list[str]
 *
 * The directories an import resolves against, exactly as this binary resolves
 * them: JAITHON_PATH, whatever -I added, and the library directories derived
 * from the executable's own location. `vm.modulePath` already mirrors all of
 * that for reflection (src/runtime/modules/module.c:152).
 *
 * The self-hosted front end needs it to resolve an import the same way the C
 * front end does. Without it, it was reduced to guessing from the importing
 * file's ancestors — which cannot find `lib` from `tests/lang`, and which no
 * heuristic fixes in general, because the answer depends on where the binary
 * was installed rather than on where the source sits. Two front ends that
 * disagree about where a module lives disagree about everything downstream of
 * it. */
static bool nModulePath(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    ObjList *result = jaiListNew(vm.modulePath.count);
    jaiGCPushRoot(OBJ_VAL(result));
    for (int i = 0; i < vm.modulePath.count; i++) {
        jaiListPush(result, OBJ_VAL(vm.modulePath.data[i]));
    }
    jaiGCPopRoot();
    *out = OBJ_VAL(result);
    return true;
}

static bool nOsExit(int argc, Value *args, Value *out) {
    (void)out;
    int64_t code = 0;
    if (argc >= 1 && !IS_NULL(args[0]) && !jaiArgInt(args[0], 1, "os_exit", &code))
        return false;
    fflush(stdout);
    fflush(stderr);
    exit((int)(code & 0xFF));
}

/* ------------------------------------------------------------------ */
/* Filesystem                                                           */
/* ------------------------------------------------------------------ */

static bool nOsCwd(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    char buf[JAI_MAX_PATH];
    errno = 0;
    if (getcwd(buf, sizeof buf) == NULL)
        return jaiIOThrowErrno(errno != 0 ? errno : EIO,
                          "cannot read the working directory", ".");
    ObjString *text = jaiStringNew(buf, strlen(buf));
    if (text == NULL) return false;
    *out = OBJ_VAL(text);
    return true;
}

static bool nOsChdir(int argc, Value *args, Value *out) {
    (void)argc;
    ObjString *path;
    if (!jaiArgString(args[0], 1, "os_chdir", &path)) return false;
    if (!jaiIOCheckPath(path, "os_chdir")) return false;

    errno = 0;
    if (chdir(path->chars) != 0)
        return jaiIOThrowErrno(errno != 0 ? errno : EIO, "cannot enter", path->chars);
    *out = NULL_VAL;
    return true;
}

/* jaiListDir hands back a jaiRealloc'd array of jaiStrdup'd names, which is
 * the tree-wide convention for an owned string: freeing each with its own
 * length keeps the allocator's accounting exact. */
static void freeDirEntries(char **entries, int count) {
    for (int i = 0; i < count; i++) {
        if (entries[i] == NULL) continue;
        JAI_FREE_ARRAY(char, entries[i], strlen(entries[i]) + 1);
    }
    /* count + 1: jaiListDir shrinks the array to exactly that, the extra slot
     * holding the NULL terminator. Freeing by `count` under-reports oldSize by
     * one pointer. */
    JAI_FREE_ARRAY(char *, entries, count + 1);
}

static bool nIoListdir(int argc, Value *args, Value *out) {
    (void)argc;
    ObjString *path;
    if (!jaiArgString(args[0], 1, "io_listdir", &path)) return false;
    if (!jaiIOCheckPath(path, "io_listdir")) return false;

    int count = 0;
    errno = 0;
    char **entries = jaiListDir(path->chars, &count);
    if (entries == NULL)
        return jaiIOThrowErrno(errno != 0 ? errno : ENOENT, "cannot list", path->chars);
    if (count < 0) count = 0;

    ObjList *names = jaiListNew(count);
    jaiGCPushRoot(OBJ_VAL(names));
    bool ok = true;
    for (int i = 0; i < count; i++) {
        if (entries[i] == NULL) continue;
        ObjString *name = jaiStringNew(entries[i], strlen(entries[i]));
        if (name == NULL) { ok = false; break; }
        jaiGCPushRoot(OBJ_VAL(name));
        jaiListPush(names, OBJ_VAL(name));
        jaiGCPopRoot();
    }
    jaiGCPopRoot();
    freeDirEntries(entries, count);

    if (!ok) return false;
    *out = OBJ_VAL(names);
    return true;
}

static int mkdirAt(ObjString *path) {
    char *tmp = NULL;
    int rc = mkdir(jaiIOPathCStr(path, &tmp), 0777);
    jaiIOPathDone(tmp);
    return rc;
}

static bool nIoMkdir(int argc, Value *args, Value *out) {
    ObjString *path;
    if (!jaiArgString(args[0], 1, "io_mkdir", &path)) return false;
    if (!jaiIOCheckPath(path, "io_mkdir")) return false;

    bool parents = false;
    if (argc >= 2 && !IS_NULL(args[1]) &&
        !jaiArgBool(args[1], 2, "io_mkdir", &parents))
        return false;

    errno = 0;
    if (parents) {
        if (!jaiMakeDirs(path->chars))
            return jaiIOThrowErrno(errno != 0 ? errno : EIO, "cannot create",
                              path->chars);
    } else if (mkdirAt(path) != 0) {
        return jaiIOThrowErrno(errno != 0 ? errno : EIO, "cannot create", path->chars);
    }
    *out = NULL_VAL;
    return true;
}

/* remove() unlinks a file and removes an empty directory, which is exactly the
 * pair a caller means by "remove this path". */
static bool nIoRemove(int argc, Value *args, Value *out) {
    (void)argc;
    ObjString *path;
    if (!jaiArgString(args[0], 1, "io_remove", &path)) return false;
    if (!jaiIOCheckPath(path, "io_remove")) return false;

    errno = 0;
    char *rmTmp = NULL;
    const char *rmPath = jaiIOPathCStr(path, &rmTmp);
    int rmRc = remove(rmPath);
    jaiIOPathDone(rmTmp);
    if (rmRc != 0)
        return jaiIOThrowErrno(errno != 0 ? errno : EIO, "cannot remove", path->chars);
    *out = NULL_VAL;
    return true;
}

static bool nIoRename(int argc, Value *args, Value *out) {
    (void)argc;
    ObjString *from, *to;
    if (!jaiArgString(args[0], 1, "io_rename", &from)) return false;
    if (!jaiArgString(args[1], 2, "io_rename", &to)) return false;
    if (!jaiIOCheckPath(from, "io_rename") || !jaiIOCheckPath(to, "io_rename")) return false;

    errno = 0;
    char *fromTmp = NULL, *toTmp = NULL;
    const char *fromP = jaiIOPathCStr(from, &fromTmp);
    const char *toP = jaiIOPathCStr(to, &toTmp);
    int mvRc = rename(fromP, toP);
    jaiIOPathDone(fromTmp);
    jaiIOPathDone(toTmp);
    if (mvRc != 0)
        return jaiIOThrowErrno2(errno != 0 ? errno : EIO, "cannot rename",
                           from->chars, to->chars);
    *out = NULL_VAL;
    return true;
}

/* io_stat answers null for a path that is not there rather than raising:
 * Path.exists() is written over it, and an exception is not an answer to
 * "is anything here". lstat by default, because the spec's four kinds include
 * "link" and a symlink has to be distinguishable from what it points at; the
 * second argument asks for stat instead, which is what "is a directory here"
 * means — `/tmp` is a link to `/private/tmp` on macOS and answering "link" to
 * `Path("/tmp").is_dir()` made every tool that writes under it fail. */
static bool nIoStat(int argc, Value *args, Value *out) {
    ObjString *path;
    if (!jaiArgString(args[0], 1, "io_stat", &path)) return false;
    if (!jaiIOCheckPath(path, "io_stat")) return false;

    bool follow = false;
    if (argc >= 2 && !IS_NULL(args[1]) &&
        !jaiArgBool(args[1], 2, "io_stat", &follow))
        return false;

    struct stat info;
    char *stTmp = NULL;
    const char *stPath = jaiIOPathCStr(path, &stTmp);
    int stRc = follow ? stat(stPath, &info) : lstat(stPath, &info);
    jaiIOPathDone(stTmp);
    if (stRc != 0) {
        /* Only "nothing there" is an answer; a broken lookup is still an error. */
        if (errno == ENOENT || errno == ENOTDIR) { *out = NULL_VAL; return true; }
        return jaiIOThrowErrno(errno, "cannot stat", path->chars);
    }

    const char *kind = S_ISREG(info.st_mode)    ? "file"
                       : S_ISDIR(info.st_mode)  ? "dir"
                       : S_ISLNK(info.st_mode)  ? "link"
                                                : "other";
    ObjString *kindName = jaiStringInternC(kind);
    if (kindName == NULL) return false;

#if defined(__APPLE__)
    struct timespec modified = info.st_mtimespec;
#else
    struct timespec modified = info.st_mtim;
#endif

    ObjDict *result = jaiDictNew();
    jaiGCPushRoot(OBJ_VAL(result));
    jaiIODictPut(result, "kind", OBJ_VAL(kindName));
    jaiIODictPut(result, "size", INT_VAL((int64_t)info.st_size));
    jaiIODictPut(result, "modified", INT_VAL((int64_t)modified.tv_sec * 1000000000 +
                                        (int64_t)modified.tv_nsec));
    jaiIODictPut(result, "mode", INT_VAL((int64_t)(info.st_mode & 07777)));
    jaiGCPopRoot();

    *out = OBJ_VAL(result);
    return true;
}

/* ------------------------------------------------------------------ */
/* Host                                                                 */
/* ------------------------------------------------------------------ */

static bool nOsPlatform(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
#if defined(__APPLE__)
    const char *name = "macos";
#elif defined(__linux__)
    const char *name = "linux";
#elif defined(__FreeBSD__)
    const char *name = "freebsd";
#elif defined(__OpenBSD__)
    const char *name = "openbsd";
#elif defined(__NetBSD__)
    const char *name = "netbsd";
#elif defined(_WIN32)
    const char *name = "windows";
#else
    const char *name = "unknown";
#endif
    *out = OBJ_VAL(jaiStringInternC(name));
    return true;
}

void jaiRegisterOSPrimitives(void) {
    jaiDefineNative("__prim__.os_env",      nOsEnv,      0, 2);
    jaiDefineNative("__prim__.os_argv",     nOsArgv,     0, 0);
    jaiDefineNative("__prim__.module_path", nModulePath,   0, 0);
    jaiDefineNative("__prim__.os_exit",     nOsExit,     0, 1);
    jaiDefineNative("__prim__.os_cwd",      nOsCwd,      0, 0);
    jaiDefineNative("__prim__.os_chdir",    nOsChdir,    1, 1);
    jaiDefineNative("__prim__.os_platform", nOsPlatform, 0, 0);

    /* The filesystem group is `io_*` in Appendix C, not `os_*`: `os_` is env,
     * argv, exit, cwd, chdir and the platform name; `os_spawn` is registered
     * by builtins_process.c's jaiRegisterProcessPrimitives instead, alongside
     * the rest of the child-process machinery it belongs with. `exists` and
     * `is_dir` are not primitives — std.io derives both from io_stat, and it
     * derives path joining and the temp directory the same way. */
    jaiDefineNative("__prim__.io_listdir", nIoListdir, 1, 1);
    jaiDefineNative("__prim__.io_mkdir",   nIoMkdir,   1, 2);
    jaiDefineNative("__prim__.io_remove",  nIoRemove,  1, 1);
    jaiDefineNative("__prim__.io_rename",  nIoRename,  2, 2);
    jaiDefineNative("__prim__.io_stat",    nIoStat,    1, 2);
}
