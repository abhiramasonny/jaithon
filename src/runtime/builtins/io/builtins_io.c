/* builtins_io.c — file streams and the `file` type: open, read, write, seek,
 * close, and the handful of methods bound to a `file` value (spec
 * Appendix C).
 *
 * This file used to hold the whole non-numeric primitive surface -- the
 * environment, the filesystem, child processes, the collector, reflection
 * and threads -- because nothing else under src/runtime/ was big enough yet
 * to make the seams worth drawing. It grew past the point where "everything
 * that isn't math" is a useful unit, so it split along its real concerns:
 *
 *   - builtins_fs.c      -- os_env, os_argv, os_exit, os_cwd/chdir,
 *                           io_listdir/mkdir/remove/rename/stat, os_platform.
 *                           The process's own environment and the paths it
 *                           can see; none of it touches a `file` handle.
 *   - builtins_process.c -- os_spawn and the child-process record it builds
 *                           (wait/poll/signal, the piped stdin/stdout/stderr
 *                           files).
 *   - builtins_gc.c      -- gc_collect/stats/enable/disable.
 *   - builtins_reflect.c -- compile/eval/exec/globals, and the build-id
 *                           primitive eval needs to stamp what it compiles.
 *   - builtins_thread.c  -- thread_spawn/join/detach, mutexes, condition
 *                           variables, atomics, and cpu_count -- registered
 *                           alongside them, as it always was, because sizing
 *                           a thread pool is the one thing that number is
 *                           for.
 *
 * What stayed here is one cohesive thing: a `file` owns a FILE*, and every
 * function below either resolves a `Stream` (a file or one of the three
 * standard descriptors) or acts on one that has already been resolved. None
 * of the five files above ever touch a FILE*.
 *
 * The errno-to-exception helpers, the NUL-safe path helpers, and the
 * dict-building helper stay defined here and are shared through
 * builtins_io.h: the rule they encode --
 *
 *   - Every failing syscall is reported by jaiIOThrowErrno() or
 *     jaiIOThrowErrno2(), so one errno always produces one exception class
 *     (§7.2) and every message names the path.
 *
 * -- is a rule about *this* file's central concern, syscalls that name a
 * path, so it is defined here and the other translation units answer to it
 * rather than each inventing their own.
 *
 * Two more rules hold throughout what is left in this file.
 *
 *   - A `file` owns its FILE*. The three standard streams are addressed by
 *     descriptor number — 0, 1, 2 — and are never closed here; that is what
 *     lets lib/std/fmt.jai write a carriage return straight to the terminal.
 *   - A path crossing into libc must be NUL-terminated, so a str holding a
 *     NUL is not the path the caller thinks it is; jaiIOCheckPath refuses it
 *     before any syscall sees it.
 */

#if !defined(_POSIX_C_SOURCE)
#  define _POSIX_C_SOURCE 200809L
#endif
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#  define _DARWIN_C_SOURCE
#endif

#include <errno.h>
#include <stdlib.h>
#include <sys/types.h>

#include "runtime/runtime.h"
#include "runtime/methods.h"
#include "runtime/builtins/io/builtins_io.h"

#include "vm/gc.h"

/* ------------------------------------------------------------------ */
/* errno -> exception                                                   */
/* ------------------------------------------------------------------ */

static ObjClass *classForErrno(int err) {
    switch (err) {
    case ENOENT: return vm.cFileNotFoundError;
    case EACCES:
    case EPERM:  return vm.cPermissionError;
    default:     return vm.cIOError;
    }
}

bool jaiIOThrowErrno(int err, const char *what, const char *path) {
    return jaiThrow(classForErrno(err), "%s '%s': %s", what,
                    path != NULL ? path : "?", strerror(err));
}

bool jaiIOThrowErrno2(int err, const char *what, const char *from,
                      const char *to) {
    return jaiThrow(classForErrno(err), "%s '%s' to '%s': %s", what, from, to,
                    strerror(err));
}

/* ------------------------------------------------------------------ */
/* Paths                                                                */
/* ------------------------------------------------------------------ */

const char *jaiIOPathCStr(ObjString *path, char **tmp) {
    *tmp = NULL;
    if (!JAI_STR_UNTERMINATED(path)) return path->chars;
    char *copy = (char *)malloc((size_t)path->length + 1);
    if (copy == NULL) return path->chars;   /* nothing better to do */
    memcpy(copy, path->chars, path->length);
    copy[path->length] = '\0';
    *tmp = copy;
    return copy;
}

void jaiIOPathDone(char *tmp) { free(tmp); }

bool jaiIOCheckPath(ObjString *path, const char *fnName) {
    if (path->length == 0)
        return jaiThrow(vm.cValueError, "%s(): the path is empty", fnName);
    if (memchr(path->chars, '\0', path->length) != NULL)
        return jaiThrow(vm.cValueError, "%s(): the path contains a NUL byte",
                        fnName);
    return true;
}

/* ------------------------------------------------------------------ */
/* Streams                                                              */
/* ------------------------------------------------------------------ */

typedef struct {
    FILE       *handle;
    ObjFile    *file;
    /* Only set for the three standard descriptors, whose names are C literals.
     * A `file`'s name is never cached here: it has to survive to wherever an
     * error is eventually thrown, which is arbitrarily far past this struct's
     * construction and past any number of GC-allocating calls in between --
     * exactly the "two calls in a row" hazard jaiStringTerminated warns about,
     * except stretched across a whole function instead of two adjacent lines.
     * streamDisplayName() below reads it fresh, immediately before the one
     * jaiThrow() call that consumes it, which is the only point a bare
     * jaiStringCStr() result is good for. */
    const char *name;
} Stream;

/* `f->path` may have lost its terminator to a later concatenation elsewhere
 * in the program since this file was opened, so every read of it for display
 * goes through here rather than `f->path->chars` directly. */
static const char *filePathDisplay(const ObjFile *f) {
    return f->path != NULL ? jaiStringCStr(f->path) : "<file>";
}

static bool throwClosed(const ObjFile *f, const char *fnName) {
    return jaiThrow(vm.cIOError, "%s(): the file '%s' is closed", fnName,
                    filePathDisplay(f));
}

/* The display name for `s`, resolved fresh: see the comment on Stream.name. */
static const char *streamDisplayName(const Stream *s) {
    return s->file != NULL ? filePathDisplay(s->file) : s->name;
}

static bool resolveStream(Value v, int index, const char *fnName, Stream *out) {
    if (IS_FILE(v)) {
        ObjFile *f = AS_FILE(v);
        if (f->closed || f->handle == NULL) return throwClosed(f, fnName);
        out->handle = f->handle;
        out->file = f;
        out->name = NULL;   /* unused for a file stream; see streamDisplayName */
        return true;
    }
    if (IS_INT(v)) {
        out->file = NULL;
        switch (AS_INT(v)) {
        case 0: out->handle = stdin;  out->name = "<stdin>";  return true;
        case 1: out->handle = stdout; out->name = "<stdout>"; return true;
        case 2: out->handle = stderr; out->name = "<stderr>"; return true;
        default:
            return jaiThrow(vm.cValueError,
                            "%s(): %lld names no standard stream (0, 1 or 2)",
                            fnName, (long long)AS_INT(v));
        }
    }
    return jaiThrow(vm.cTypeError,
                    "%s() argument %d: expected file or int, got %s", fnName,
                    index, jaiTypeNameStatic(v));
}

/* The open mode is authoritative: asking a "w" file to read fails here rather
 * than in libc, where the error would carry no useful errno. */
static bool requireStreamAccess(const Stream *s, bool wantRead,
                                const char *fnName) {
    bool ok = s->file == NULL ||
              (wantRead ? s->file->readable : s->file->writable);
    if (ok) return true;
    return jaiThrow(vm.cIOError, "%s(): '%s' is not open for %s", fnName,
                    streamDisplayName(s), wantRead ? "reading" : "writing");
}

/* Operations that manipulate the handle itself need the object, not a stream:
 * closing stdout or seeking a terminal is never what the caller meant. */
static bool requireFile(Value v, const char *fnName, ObjFile **out) {
    if (!IS_FILE(v))
        return jaiThrow(vm.cTypeError, "%s() argument 1: expected file, got %s",
                        fnName, jaiTypeNameStatic(v));
    *out = AS_FILE(v);
    return true;
}

/* ------------------------------------------------------------------ */
/* Reading                                                              */
/* ------------------------------------------------------------------ */

/* Both readers free `buf` when they fail, so a caller that gets false owns
 * nothing and can return straight away. */

static bool streamError(Stream *s, JaiBuf *buf, const char *what) {
    if (!ferror(s->handle)) return false;
    int err = errno;
    clearerr(s->handle);
    jaiBufFree(buf);
    (void)jaiIOThrowErrno(err != 0 ? err : EIO, what, streamDisplayName(s));
    return true;
}

/* A read cut short by a signal is not an error, and this VM raises signals of
 * its own: the JIT samples at 1 kHz on ITIMER_PROF, so any blocking read from
 * a pipe or a terminal is liable to be interrupted. `fread` reports that as a
 * short read with the stream's error flag set and errno EINTR, and
 * streamError below turns it into an IOError.
 *
 * It surfaced as "cannot read from '<stdout>': Interrupted system call" while
 * a parent process read 32 children's output, and got dramatically more likely
 * under JAITHON_JIT_DEOPT_STRESS, which lengthens the windows a read sits in.
 * Nothing about it is specific to that program -- any Jaithon code reading a
 * pipe while the sampler runs could see it. `SA_RESTART` is set on the SIGPROF
 * handler and did not prevent it, so the retry belongs here. */
static bool readInterrupted(Stream *s) {
    if (!ferror(s->handle) || errno != EINTR) return false;
    clearerr(s->handle);
    errno = 0;
    return true;
}

static bool readAllInto(Stream *s, JaiBuf *buf, const char *what) {
    char chunk[8192];
    for (;;) {
        errno = 0;
        size_t got = fread(chunk, 1, sizeof chunk, s->handle);
        if (got > 0) jaiBufAppend(buf, chunk, got);
        if (got < sizeof chunk) {
            if (readInterrupted(s)) continue;
            break;
        }
    }
    return !streamError(s, buf, what);
}

/* A short read is not an error: `count` is an upper bound, not a demand. */
static bool readCountInto(Stream *s, size_t want, JaiBuf *buf, const char *what) {
    char chunk[8192];
    while (want > 0) {
        size_t ask = want < sizeof chunk ? want : sizeof chunk;
        errno = 0;
        size_t got = fread(chunk, 1, ask, s->handle);
        if (got > 0) {
            jaiBufAppend(buf, chunk, got);
            want -= got;
        }
        if (got < ask) {
            if (readInterrupted(s)) continue;
            break;
        }
    }
    return !streamError(s, buf, what);
}

static bool isBinary(const Stream *s) {
    return s->file != NULL && s->file->binary;
}

static bool makePiece(const Stream *s, const char *data, size_t length,
                      Value *out) {
    if (isBinary(s)) {
        *out = OBJ_VAL(jaiBytesNew((const uint8_t *)data, length));
        return true;
    }
    ObjString *text = jaiStringNew(data, length);
    if (text == NULL) return false;
    *out = OBJ_VAL(text);
    return true;
}

static bool finishRead(const Stream *s, JaiBuf *buf, Value *out) {
    if (isBinary(s)) {
        ObjBytes *bytes = jaiBytesNew(buf->data, buf->count);
        jaiBufFree(buf);
        *out = OBJ_VAL(bytes);
        return true;
    }
    size_t length = 0;
    char *chars = jaiBufTakeCString(buf, &length);
    if (chars == NULL) return jaiThrow(vm.cRuntimeError, "out of memory reading");
    ObjString *text = jaiStringTake(chars, length);
    if (text == NULL) return false;
    *out = OBJ_VAL(text);
    return true;
}

/* ------------------------------------------------------------------ */
/* io_open                                                              */
/* ------------------------------------------------------------------ */

static const char *const kFileModes[] = {
    "r", "w", "a", "rb", "wb", "ab", "r+", "w+", "a+",
};

static bool modeIsKnown(const char *mode) {
    for (size_t i = 0; i < sizeof kFileModes / sizeof kFileModes[0]; i++) {
        if (strcmp(mode, kFileModes[i]) == 0) return true;
    }
    return false;
}

static bool nIoOpen(int argc, Value *args, Value *out) {
    ObjString *path;
    if (!jaiArgString(args[0], 1, "open", &path)) return false;
    if (!jaiIOCheckPath(path, "open")) return false;

    /* `mode` reaches here as an argument, not a literal: modeIsKnown() runs
     * strcmp against it, and an unterminated view would let that read past
     * the buffer it lives in looking for a NUL that isn't there. Through
     * jaiIOPathCStr like `path` below -- a malloc'd copy needs no GC root, so
     * unlike jaiStringTerminated it stays valid across every use in this
     * function without pairing a push with each return. */
    char *modeTmp = NULL;
    const char *mode = "r";
    if (argc >= 2 && !IS_NULL(args[1])) {
        ObjString *modeText;
        if (!jaiArgString(args[1], 2, "open", &modeText)) return false;
        mode = jaiIOPathCStr(modeText, &modeTmp);
    }
    if (!modeIsKnown(mode)) {
        bool thrown = jaiThrow(vm.cValueError,
                        "open(): invalid mode '%s'; expected one of "
                        "r w a rb wb ab r+ w+ a+", mode);
        jaiIOPathDone(modeTmp);
        return thrown;
    }

    errno = 0;
    char *pathTmp = NULL;
    const char *cpath = jaiIOPathCStr(path, &pathTmp);
    FILE *handle = fopen(cpath, mode);
    if (handle == NULL) {
        int err = errno;
        bool thrown = jaiIOThrowErrno(err != 0 ? err : EIO, "cannot open", cpath);
        jaiIOPathDone(pathTmp);
        jaiIOPathDone(modeTmp);
        return thrown;
    }
    jaiIOPathDone(pathTmp);

    /* jaiFileNew scans `mode` to a NUL of its own (readable/writable/binary
     * flags) but keeps no pointer into it, so modeTmp can be freed right
     * after this call returns. */
    *out = OBJ_VAL(jaiFileNew(handle, path, mode));
    jaiIOPathDone(modeTmp);
    return true;
}

/* ------------------------------------------------------------------ */
/* Reading: the io_read primitive, and the two line forms that are      */
/* methods of `file` rather than primitives                             */
/* ------------------------------------------------------------------ */

static bool nIoRead(int argc, Value *args, Value *out) {
    Stream s;
    if (!resolveStream(args[0], 1, "read", &s)) return false;
    if (!requireStreamAccess(&s, true, "read")) return false;

    int64_t count = -1;
    if (argc >= 2 && !IS_NULL(args[1]) &&
        !jaiArgInt(args[1], 2, "read", &count))
        return false;

    JaiBuf buf;
    jaiBufInit(&buf);
    bool ok = count < 0 ? readAllInto(&s, &buf, "cannot read from")
                        : readCountInto(&s, (size_t)count, &buf,
                                        "cannot read from");
    if (!ok) return false;

    ObjBytes *raw = jaiBytesNew(buf.data, buf.count);
    jaiBufFree(&buf);
    if (raw == NULL) return false;
    *out = OBJ_VAL(raw);
    return true;
}

static bool nIoReadLine(int argc, Value *args, Value *out) {
    (void)argc;
    Stream s;
    if (!resolveStream(args[0], 1, "read_line", &s)) return false;
    if (!requireStreamAccess(&s, true, "read_line")) return false;

    JaiBuf buf;
    jaiBufInit(&buf);
    bool sawAny = false;
    int c;
    while ((c = fgetc(s.handle)) != EOF) {
        sawAny = true;
        if (c == '\n') break;
        jaiBufPush(&buf, (uint8_t)c);
    }
    if (streamError(&s, &buf, "cannot read from")) return false;
    if (!sawAny) {
        jaiBufFree(&buf);
        *out = NULL_VAL;
        return true;
    }
    if (buf.count > 0 && buf.data[buf.count - 1] == '\r') buf.count--;
    return finishRead(&s, &buf, out);
}

static bool readLinesInto(Stream *s, ObjList *lines) {
    JaiBuf buf;
    jaiBufInit(&buf);
    if (!readAllInto(s, &buf, "cannot read from")) return false;

    const char *data = (const char *)buf.data;
    size_t total = buf.count;
    size_t start = 0;
    bool ok = true;

    for (size_t i = 0; ok && i <= total; i++) {
        if (i < total && data[i] != '\n') continue;
        if (i == total && start == total) break;   /* trailing newline */
        size_t end = i;
        if (end > start && data[end - 1] == '\r') end--;

        Value line;
        ok = makePiece(s, data + start, end - start, &line);
        if (ok) {
            jaiGCPushRoot(line);
            jaiListPush(lines, line);
            jaiGCPopRoot();
        }
        start = i + 1;
    }

    jaiBufFree(&buf);
    return ok;
}

static bool nIoReadLines(int argc, Value *args, Value *out) {
    (void)argc;
    Stream s;
    if (!resolveStream(args[0], 1, "read_lines", &s)) return false;
    if (!requireStreamAccess(&s, true, "read_lines")) return false;

    ObjList *lines = jaiListNew(0);
    jaiGCPushRoot(OBJ_VAL(lines));
    bool ok = readLinesInto(&s, lines);
    jaiGCPopRoot();
    if (!ok) return false;
    *out = OBJ_VAL(lines);
    return true;
}

/* ------------------------------------------------------------------ */
/* io_write / io_close / io_seek / io_tell / io_flush / io_eof          */
/* ------------------------------------------------------------------ */

static bool nIoWrite(int argc, Value *args, Value *out) {
    (void)argc;
    Stream s;
    if (!resolveStream(args[0], 1, "write", &s)) return false;
    if (!requireStreamAccess(&s, false, "write")) return false;

    const void *data;
    size_t length;
    if (IS_STRING(args[1])) {
        data = AS_STRING(args[1])->chars;
        length = AS_STRING(args[1])->length;
    } else if (IS_BYTES(args[1])) {
        data = AS_BYTES(args[1])->data;
        length = AS_BYTES(args[1])->length;
    } else {
        return jaiThrow(vm.cTypeError,
                        "write() argument 2: expected str or bytes, got %s",
                        jaiTypeNameStatic(args[1]));
    }

    errno = 0;
    size_t written = length > 0 ? fwrite(data, 1, length, s.handle) : 0;
    if (written != length) {
        int err = errno;
        clearerr(s.handle);
        return jaiIOThrowErrno(err != 0 ? err : EIO, "cannot write to", streamDisplayName(&s));
    }
    *out = INT_VAL((int64_t)written);
    return true;
}

static bool nIoClose(int argc, Value *args, Value *out) {
    (void)argc;
    ObjFile *file;
    if (!requireFile(args[0], "close", &file)) return false;

    *out = NULL_VAL;
    if (file->closed || file->handle == NULL) {
        file->closed = true;
        return true;
    }

    FILE *handle = file->handle;
    file->handle = NULL;
    file->closed = true;

    errno = 0;
    if (fclose(handle) != 0)
        return jaiIOThrowErrno(errno != 0 ? errno : EIO, "cannot close",
                          filePathDisplay(file));
    return true;
}

static bool nIoSeek(int argc, Value *args, Value *out) {
    ObjFile *file;
    if (!requireFile(args[0], "seek", &file)) return false;
    if (file->closed || file->handle == NULL) return throwClosed(file, "seek");

    int64_t offset;
    if (!jaiArgInt(args[1], 2, "seek", &offset)) return false;
    int64_t whence = 0;
    if (argc >= 3 && !IS_NULL(args[2]) &&
        !jaiArgInt(args[2], 3, "seek", &whence))
        return false;

    int origin;
    switch (whence) {
    case 0:  origin = SEEK_SET; break;
    case 1:  origin = SEEK_CUR; break;
    case 2:  origin = SEEK_END; break;
    default:
        return jaiThrow(vm.cValueError,
                        "seek(): whence must be 0, 1 or 2, got %lld",
                        (long long)whence);
    }

    /* Resolved fresh at each throw, not cached in a local: see the comment on
     * Stream.name above. */
    errno = 0;
    if (fseeko(file->handle, (off_t)offset, origin) != 0)
        return jaiIOThrowErrno(errno != 0 ? errno : EIO, "cannot seek in",
                          filePathDisplay(file));

    off_t position = ftello(file->handle);
    if (position < 0)
        return jaiIOThrowErrno(errno != 0 ? errno : EIO, "cannot seek in",
                          filePathDisplay(file));
    *out = INT_VAL((int64_t)position);
    return true;
}

static bool nIoTell(int argc, Value *args, Value *out) {
    (void)argc;
    ObjFile *file;
    if (!requireFile(args[0], "tell", &file)) return false;
    if (file->closed || file->handle == NULL) return throwClosed(file, "tell");

    errno = 0;
    off_t position = ftello(file->handle);
    if (position < 0)
        return jaiIOThrowErrno(errno != 0 ? errno : EIO, "cannot read the position of",
                          filePathDisplay(file));
    *out = INT_VAL((int64_t)position);
    return true;
}

static bool nIoFlush(int argc, Value *args, Value *out) {
    (void)argc;
    Stream s;
    if (!resolveStream(args[0], 1, "flush", &s)) return false;

    errno = 0;
    if (fflush(s.handle) != 0)
        return jaiIOThrowErrno(errno != 0 ? errno : EIO, "cannot flush", streamDisplayName(&s));
    *out = NULL_VAL;
    return true;
}

static bool nIoEof(int argc, Value *args, Value *out) {
    (void)argc;
    Stream s;
    if (!resolveStream(args[0], 1, "is_eof", &s)) return false;
    *out = BOOL_VAL(feof(s.handle) != 0);
    return true;
}

/* ------------------------------------------------------------------ */
/* File methods                                                         */
/* ------------------------------------------------------------------ */

static bool nFileLines(int argc, Value *args, Value *out) {
    Value lines;
    if (!nIoReadLines(argc, args, &lines)) return false;
    jaiGCPushRoot(lines);
    ObjIter *it = jaiIterNew(ITER_LIST, lines);
    jaiGCPopRoot();
    *out = OBJ_VAL(it);
    return true;
}

static bool nFileWriteLine(int argc, Value *args, Value *out) {
    Stream s;
    if (!resolveStream(args[0], 1, "write_line", &s)) return false;
    if (!requireStreamAccess(&s, false, "write_line")) return false;

    ObjString *text = NULL;
    if (argc >= 2 && !IS_NULL(args[1]) &&
        !jaiArgString(args[1], 2, "write_line", &text))
        return false;

    errno = 0;
    size_t length = text != NULL ? text->length : 0;
    size_t written = length > 0 ? fwrite(text->chars, 1, length, s.handle) : 0;
    bool ok = written == length && fputc('\n', s.handle) != EOF;
    if (!ok) {
        int err = errno;
        clearerr(s.handle);
        return jaiIOThrowErrno(err != 0 ? err : EIO, "cannot write to", streamDisplayName(&s));
    }
    *out = INT_VAL((int64_t)written + 1);
    return true;
}

static bool nFileIsClosed(int argc, Value *args, Value *out) {
    (void)argc;
    ObjFile *file;
    if (!requireFile(args[0], "is_closed", &file)) return false;
    *out = BOOL_VAL(file->closed || file->handle == NULL);
    return true;
}

static bool nFilePath(int argc, Value *args, Value *out) {
    (void)argc;
    ObjFile *file;
    if (!requireFile(args[0], "path", &file)) return false;
    *out = file->path != NULL ? OBJ_VAL(file->path)
                              : OBJ_VAL(jaiStringIntern("", 0));
    return true;
}

bool jaiFileMethod(Value receiver, ObjString *name, Value *out) {
    if (!IS_FILE(receiver) || name == NULL) return false;
    const char *text = name->chars;

#define FILE_METHOD(label, fn, minArity, maxArity)                             \
    if (strcmp(text, (label)) == 0) {                                          \
        *out = jaiBindNative(receiver, (label), (fn), (minArity), (maxArity),  \
                             NULL);                                            \
        return true;                                                           \
    }

    FILE_METHOD("read",       nIoRead,        1, 2)
    FILE_METHOD("read_line",  nIoReadLine,    1, 1)
    FILE_METHOD("read_lines", nIoReadLines,   1, 1)
    FILE_METHOD("lines",      nFileLines,     1, 1)
    FILE_METHOD("iter",       nFileLines,     1, 1)
    FILE_METHOD("write",      nIoWrite,       2, 2)
    FILE_METHOD("write_line", nFileWriteLine, 1, 2)
    FILE_METHOD("close",      nIoClose,       1, 1)
    FILE_METHOD("flush",      nIoFlush,       1, 1)
    FILE_METHOD("seek",       nIoSeek,        2, 3)
    FILE_METHOD("tell",       nIoTell,        1, 1)
    FILE_METHOD("is_eof",     nIoEof,         1, 1)
    FILE_METHOD("is_closed",  nFileIsClosed,  1, 1)
    FILE_METHOD("path",       nFilePath,      1, 1)

#undef FILE_METHOD
    return false;
}

void jaiRegisterIOPrimitives(void) {
    jaiDefineNative("__prim__.io_open",  nIoOpen,  1, 2);
    jaiDefineNative("__prim__.io_read",  nIoRead,  1, 2);
    jaiDefineNative("__prim__.io_write", nIoWrite, 2, 2);
    jaiDefineNative("__prim__.io_close", nIoClose, 1, 1);
    jaiDefineNative("__prim__.io_seek",  nIoSeek,  2, 3);
    jaiDefineNative("__prim__.io_flush", nIoFlush, 1, 1);
}

/* ------------------------------------------------------------------ */
/* Dict building                                                        */
/* ------------------------------------------------------------------ */

void jaiIODictPut(ObjDict *dict, const char *key, Value value) {
    jaiGCPushRoot(OBJ_VAL(dict));
    jaiGCPushRoot(value);
    ObjString *name = jaiStringInternC(key);
    jaiGCPushRoot(OBJ_VAL(name));
    (void)jaiDictSet(dict, OBJ_VAL(name), value);
    jaiGCPopRoots(3);
}
