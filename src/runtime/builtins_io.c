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

/* Feature macros must precede every include: fseeko is POSIX, not C11, and
 * _DARWIN_C_SOURCE puts back what asking for POSIX takes away on macOS. */
#if !defined(_POSIX_C_SOURCE)
#  define _POSIX_C_SOURCE 200809L
#endif
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#  define _DARWIN_C_SOURCE
#endif

#include <errno.h>
#include <stdlib.h>
#include <sys/types.h>

#include "runtime.h"
#include "methods.h"
#include "builtins_io.h"

#include "../vm/gc.h"

/* ------------------------------------------------------------------ */
/* errno -> exception                                                   */
/* ------------------------------------------------------------------ */

/* Spec §7.2 gives IOError exactly two subclasses, and these are the two errno
 * values that mean them. Everything else is an IOError with its own text: an
 * exhaustive errno table would be a list of synonyms for "the call failed". */
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
    /* Bounded by the length rather than by a terminator: a string may be a
     * view into a shared append buffer, in which case the byte after it
     * belongs to a later concatenation and strlen would run on past. */
    if (memchr(path->chars, '\0', path->length) != NULL)
        return jaiThrow(vm.cValueError, "%s(): the path contains a NUL byte",
                        fnName);
    return true;
}

/* ------------------------------------------------------------------ */
/* Streams                                                              */
/* ------------------------------------------------------------------ */

/* A resolved I/O target: a `file` value, or one of the standard streams named
 * by descriptor number. `file` is NULL for the latter, which is what decides
 * whether a read yields str or bytes and whether closing is even allowed. */
typedef struct {
    FILE       *handle;
    ObjFile    *file;
    const char *name;
} Stream;

static bool throwClosed(const ObjFile *f, const char *fnName) {
    return jaiThrow(vm.cIOError, "%s(): the file '%s' is closed", fnName,
                    f->path != NULL ? f->path->chars : "?");
}

static bool resolveStream(Value v, int index, const char *fnName, Stream *out) {
    if (IS_FILE(v)) {
        ObjFile *f = AS_FILE(v);
        if (f->closed || f->handle == NULL) return throwClosed(f, fnName);
        out->handle = f->handle;
        out->file = f;
        out->name = f->path != NULL ? f->path->chars : "<file>";
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
static bool requireReadable(const Stream *s, const char *fnName) {
    if (s->file != NULL && !s->file->readable)
        return jaiThrow(vm.cIOError, "%s(): '%s' is not open for reading",
                        fnName, s->name);
    return true;
}

static bool requireWritable(const Stream *s, const char *fnName) {
    if (s->file != NULL && !s->file->writable)
        return jaiThrow(vm.cIOError, "%s(): '%s' is not open for writing",
                        fnName, s->name);
    return true;
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
    (void)jaiIOThrowErrno(err != 0 ? err : EIO, what, s->name);
    return true;
}

static bool readAllInto(Stream *s, JaiBuf *buf, const char *what) {
    char chunk[8192];
    for (;;) {
        size_t got = fread(chunk, 1, sizeof chunk, s->handle);
        if (got > 0) jaiBufAppend(buf, chunk, got);
        if (got < sizeof chunk) break;
    }
    return !streamError(s, buf, what);
}

/* A short read is not an error: `count` is an upper bound, not a demand. */
static bool readCountInto(Stream *s, size_t want, JaiBuf *buf, const char *what) {
    char chunk[8192];
    while (want > 0) {
        size_t ask = want < sizeof chunk ? want : sizeof chunk;
        size_t got = fread(chunk, 1, ask, s->handle);
        if (got > 0) {
            jaiBufAppend(buf, chunk, got);
            want -= got;
        }
        if (got < ask) break;
    }
    return !streamError(s, buf, what);
}

/* A binary file yields bytes and a text file yields str; a standard stream has
 * no mode of its own and is text. */
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
    if (text == NULL) return false;      /* over the length limit; it threw */
    *out = OBJ_VAL(text);
    return true;
}

/* Consumes `buf` either way. */
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

/* Exactly the modes of the specification. "rb+" and friends are absent on
 * purpose: one spelling per mode, as everywhere else in the language. */
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

    const char *mode = "r";
    if (argc >= 2 && !IS_NULL(args[1])) {
        ObjString *modeText;
        if (!jaiArgString(args[1], 2, "open", &modeText)) return false;
        mode = modeText->chars;
    }
    if (!modeIsKnown(mode))
        return jaiThrow(vm.cValueError,
                        "open(): invalid mode '%s'; expected one of "
                        "r w a rb wb ab r+ w+ a+", mode);

    errno = 0;
    char *pathTmp = NULL;
    FILE *handle = fopen(jaiIOPathCStr(path, &pathTmp), mode);
    jaiIOPathDone(pathTmp);
    if (handle == NULL)
        return jaiIOThrowErrno(errno != 0 ? errno : EIO, "cannot open", path->chars);

    /* `path` is a native argument, so it is on the value stack and rooted for
     * the allocation inside jaiFileNew. */
    *out = OBJ_VAL(jaiFileNew(handle, path, mode));
    return true;
}

/* ------------------------------------------------------------------ */
/* Reading: the io_read primitive, and the two line forms that are      */
/* methods of `file` rather than primitives                             */
/* ------------------------------------------------------------------ */

static bool nIoRead(int argc, Value *args, Value *out) {
    Stream s;
    if (!resolveStream(args[0], 1, "read", &s)) return false;
    if (!requireReadable(&s, "read")) return false;

    /* A negative or absent count means "the rest of the stream". */
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

    /* Always bytes, whatever the mode: `io_read` is the raw primitive and
     * std.io decodes with `str_decode` (Appendix C). A count is a byte count, so
     * decoding here could also split a multi-byte sequence — which is exactly
     * the boundary std.io's readers are written to handle. */
    ObjBytes *raw = jaiBytesNew(buf.data, buf.count);
    jaiBufFree(&buf);
    if (raw == NULL) return false;
    *out = OBJ_VAL(raw);
    return true;
}

/* Returns null at end of input rather than an empty line, so that a read loop
 * ends on `null` and an empty line stays distinguishable from EOF. The
 * terminator is not part of the line. */
static bool nIoReadLine(int argc, Value *args, Value *out) {
    (void)argc;
    Stream s;
    if (!resolveStream(args[0], 1, "read_line", &s)) return false;
    if (!requireReadable(&s, "read_line")) return false;

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
    /* A CRLF ending leaves the CR behind; it is not part of the text. */
    if (buf.count > 0 && buf.data[buf.count - 1] == '\r') buf.count--;
    return finishRead(&s, &buf, out);
}

/* Splits what is left of the stream on '\n'. A final newline terminates the
 * last line rather than starting an empty one. */
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
        /* The buffer is raw memory, not a heap object, so allocating a string
         * per line cannot move or free it. */
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
    if (!requireReadable(&s, "read_lines")) return false;

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
    if (!requireWritable(&s, "write")) return false;

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
        return jaiIOThrowErrno(err != 0 ? err : EIO, "cannot write to", s.name);
    }
    *out = INT_VAL((int64_t)written);
    return true;
}

/* Closing twice is not an error: an explicit close next to a `defer` that also
 * closes is a normal shape, and the second one has nothing left to do. */
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
    /* Mark it closed before the fclose: even a failing fclose releases the
     * handle, so the collector must never touch it again. */
    file->handle = NULL;
    file->closed = true;

    errno = 0;
    if (fclose(handle) != 0)
        return jaiIOThrowErrno(errno != 0 ? errno : EIO, "cannot close",
                          file->path != NULL ? file->path->chars : "?");
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

    const char *name = file->path != NULL ? file->path->chars : "?";
    errno = 0;
    if (fseeko(file->handle, (off_t)offset, origin) != 0)
        return jaiIOThrowErrno(errno != 0 ? errno : EIO, "cannot seek in", name);

    off_t position = ftello(file->handle);
    if (position < 0)
        return jaiIOThrowErrno(errno != 0 ? errno : EIO, "cannot seek in", name);
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
                          file->path != NULL ? file->path->chars : "?");
    *out = INT_VAL((int64_t)position);
    return true;
}

static bool nIoFlush(int argc, Value *args, Value *out) {
    (void)argc;
    Stream s;
    if (!resolveStream(args[0], 1, "flush", &s)) return false;

    errno = 0;
    if (fflush(s.handle) != 0)
        return jaiIOThrowErrno(errno != 0 ? errno : EIO, "cannot flush", s.name);
    *out = NULL_VAL;
    return true;
}

/* C reports end of file only after a read has run into it, so this answers
 * "did the last read reach the end", not "is there anything left". */
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

/* `f.lines()` and `f.iter()` materialise the remaining lines and hand back a
 * list iterator: the iterator protocol has no file kind, and a lazy line
 * iterator would have to keep a raw FILE* alive inside a GC object. */
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
    if (!requireWritable(&s, "write_line")) return false;

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
        return jaiIOThrowErrno(err != 0 ? err : EIO, "cannot write to", s.name);
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

/* Arities count the receiver, which the VM passes as args[0]. */
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

/* Line reading, `tell` and the end-of-stream flag are methods of the `file`
 * type rather than primitives: each is either derivable from io_read or, in
 * the case of tell, exactly `io_seek(handle, 0, SEEK_CURRENT)`. */
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

/* Shared with builtins_fs.c (io_stat), builtins_process.c (os_spawn's
 * result) and builtins_gc.c (gc_stats) through builtins_io.h. */
void jaiIODictPut(ObjDict *dict, const char *key, Value value) {
    jaiGCPushRoot(OBJ_VAL(dict));
    jaiGCPushRoot(value);
    ObjString *name = jaiStringInternC(key);
    jaiGCPushRoot(OBJ_VAL(name));
    (void)jaiDictSet(dict, OBJ_VAL(name), value);
    jaiGCPopRoots(3);
}
