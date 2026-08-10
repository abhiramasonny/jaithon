/* builtins_io.c — the half of the primitive surface that talks to the machine:
 * files, the filesystem, the environment, child processes, the collector,
 * reflection, and threads (spec Appendix C).
 *
 * Four rules hold throughout this file.
 *
 *   - Every failing syscall is reported by throwErrno(), so one errno always
 *     produces one exception class (§7.2) and every message names the path.
 *   - A `file` owns its FILE*. The three standard streams are addressed by
 *     descriptor number — 0, 1, 2 — and are never closed here; that is what
 *     lets lib/std/fmt.jai write a carriage return straight to the terminal.
 *   - The reflection primitives all go through jaiCompileSource, so `eval`,
 *     the REPL and lib/jaithon/compile see exactly one front end.
 *   - Nothing here runs Jaithon code anywhere but on the thread that entered
 *     the VM. See the thread section for how that is enforced rather than
 *     merely documented.
 */

/* Feature macros must precede every include: fseeko, setenv, getcwd and
 * fileno are POSIX, not C11, and _DARWIN_C_SOURCE puts back what asking for
 * POSIX takes away on macOS. */
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

#include "runtime.h"
#include "handles.h"
#include "methods.h"

#include "../native/native.h"
#include "../vm/gc.h"
#include "../vm/serialize.h"

/* POSIX guarantees the variable but not a declaration for it in <unistd.h>. */
extern char **environ;

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

/* `what` reads as a verb phrase: "cannot open", "cannot read from". */
static bool throwErrno(int err, const char *what, const char *path) {
    return jaiThrow(classForErrno(err), "%s '%s': %s", what,
                    path != NULL ? path : "?", strerror(err));
}

static bool throwErrno2(int err, const char *what, const char *from,
                        const char *to) {
    return jaiThrow(classForErrno(err), "%s '%s' to '%s': %s", what, from, to,
                    strerror(err));
}

/* A path crossing into libc must be NUL-terminated, so a str holding a NUL is
 * not the path the caller thinks it is. Rejecting it keeps the error message
 * and the syscall talking about the same file. */
static bool checkPath(ObjString *path, const char *fnName) {
    if (path->length == 0)
        return jaiThrow(vm.cValueError, "%s(): the path is empty", fnName);
    if (strlen(path->chars) != path->length)
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
    (void)throwErrno(err != 0 ? err : EIO, what, s->name);
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
    if (!checkPath(path, "open")) return false;

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
    FILE *handle = fopen(path->chars, mode);
    if (handle == NULL)
        return throwErrno(errno != 0 ? errno : EIO, "cannot open", path->chars);

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
        return throwErrno(err != 0 ? err : EIO, "cannot write to", s.name);
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
        return throwErrno(errno != 0 ? errno : EIO, "cannot close",
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
        return throwErrno(errno != 0 ? errno : EIO, "cannot seek in", name);

    off_t position = ftello(file->handle);
    if (position < 0)
        return throwErrno(errno != 0 ? errno : EIO, "cannot seek in", name);
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
        return throwErrno(errno != 0 ? errno : EIO, "cannot read the position of",
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
        return throwErrno(errno != 0 ? errno : EIO, "cannot flush", s.name);
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
        return throwErrno(err != 0 ? err : EIO, "cannot write to", s.name);
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

/* Every primitive that answers with a record builds it this way, so the keys
 * are interned once and the dict is rooted for the whole construction. */
static void dictPut(ObjDict *dict, const char *key, Value value) {
    jaiGCPushRoot(OBJ_VAL(dict));
    jaiGCPushRoot(value);
    ObjString *name = jaiStringInternC(key);
    jaiGCPushRoot(OBJ_VAL(name));
    (void)jaiDictSet(dict, OBJ_VAL(name), value);
    jaiGCPopRoots(3);
}

/* ------------------------------------------------------------------ */
/* Environment                                                          */
/* ------------------------------------------------------------------ */

static bool checkVariableName(ObjString *name) {
    if (name->length == 0)
        return jaiThrow(vm.cValueError,
                        "os_env(): the variable name must not be empty");
    if (strlen(name->chars) != name->length || strchr(name->chars, '=') != NULL)
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
        if (strlen(value->chars) != value->length)
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
 * that for reflection (src/runtime/module.c:152).
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
        return throwErrno(errno != 0 ? errno : EIO,
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
    if (!checkPath(path, "os_chdir")) return false;

    errno = 0;
    if (chdir(path->chars) != 0)
        return throwErrno(errno != 0 ? errno : EIO, "cannot enter", path->chars);
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
    if (!checkPath(path, "io_listdir")) return false;

    int count = 0;
    errno = 0;
    char **entries = jaiListDir(path->chars, &count);
    if (entries == NULL)
        return throwErrno(errno != 0 ? errno : ENOENT, "cannot list", path->chars);
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

static bool nIoMkdir(int argc, Value *args, Value *out) {
    ObjString *path;
    if (!jaiArgString(args[0], 1, "io_mkdir", &path)) return false;
    if (!checkPath(path, "io_mkdir")) return false;

    bool parents = false;
    if (argc >= 2 && !IS_NULL(args[1]) &&
        !jaiArgBool(args[1], 2, "io_mkdir", &parents))
        return false;

    errno = 0;
    if (parents) {
        if (!jaiMakeDirs(path->chars))
            return throwErrno(errno != 0 ? errno : EIO, "cannot create",
                              path->chars);
    } else if (mkdir(path->chars, 0777) != 0) {
        return throwErrno(errno != 0 ? errno : EIO, "cannot create", path->chars);
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
    if (!checkPath(path, "io_remove")) return false;

    errno = 0;
    if (remove(path->chars) != 0)
        return throwErrno(errno != 0 ? errno : EIO, "cannot remove", path->chars);
    *out = NULL_VAL;
    return true;
}

static bool nIoRename(int argc, Value *args, Value *out) {
    (void)argc;
    ObjString *from, *to;
    if (!jaiArgString(args[0], 1, "io_rename", &from)) return false;
    if (!jaiArgString(args[1], 2, "io_rename", &to)) return false;
    if (!checkPath(from, "io_rename") || !checkPath(to, "io_rename")) return false;

    errno = 0;
    if (rename(from->chars, to->chars) != 0)
        return throwErrno2(errno != 0 ? errno : EIO, "cannot rename",
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
    if (!checkPath(path, "io_stat")) return false;

    bool follow = false;
    if (argc >= 2 && !IS_NULL(args[1]) &&
        !jaiArgBool(args[1], 2, "io_stat", &follow))
        return false;

    struct stat info;
    if ((follow ? stat(path->chars, &info) : lstat(path->chars, &info)) != 0) {
        /* Only "nothing there" is an answer; a broken lookup is still an error. */
        if (errno == ENOENT || errno == ENOTDIR) { *out = NULL_VAL; return true; }
        return throwErrno(errno, "cannot stat", path->chars);
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
    dictPut(result, "kind", OBJ_VAL(kindName));
    dictPut(result, "size", INT_VAL((int64_t)info.st_size));
    dictPut(result, "modified", INT_VAL((int64_t)modified.tv_sec * 1000000000 +
                                        (int64_t)modified.tv_nsec));
    dictPut(result, "mode", INT_VAL((int64_t)(info.st_mode & 07777)));
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

static bool nCpuCount(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    int count = jaiCpuCount();
    *out = INT_VAL(count > 0 ? count : 1);
    return true;
}

/* ------------------------------------------------------------------ */
/* Child processes                                                      */
/* ------------------------------------------------------------------ */

/* --- os_spawn ------------------------------------------------------ */

/* A NUL-terminated argv (or "K=V" environment) built out of Jaithon strings.
 * Each entry points into the ObjString it came from, so the strings must stay
 * rooted for as long as the array is used — they are, being the native's own
 * arguments and the dict reachable from them. */
typedef struct {
    const char **items;
    int          count;      /* entries, not counting the NULL terminator */
} CStrVec;

static void cstrVecFree(CStrVec *v) {
    if (v->items != NULL) JAI_FREE_ARRAY(const char *, v->items, v->count + 1);
    v->items = NULL;
    v->count = 0;
}

/* A NUL inside an argument would silently truncate it at the exec boundary,
 * where there is no length to carry, so it is refused here rather than
 * half-honoured. */
static bool checkExecText(ObjString *s, const char *what) {
    if (strlen(s->chars) != s->length)
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

/* The child's environment as "K=V" strings. The joined text is owned by this
 * vector, since neither half exists as one string anywhere else. */
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

/* The pid of the child a `wait`/`poll`/`signal` callable belongs to travels as
 * the bound receiver, which is why these are ObjBound over an int rather than
 * closures: the VM already passes a bound receiver as args[0]. */
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
    *out = NULL_VAL;      /* still running */
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

/* One end of a child's pipe as a `file`, so std.io.File can adopt it. Closing
 * the File closes the descriptor, which is what the class documents. */
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
    /* Each wrapper is rooted before the next one allocates, and each is
     * created only if the one before it succeeded — so `wrapped` is both the
     * number of roots to pop and the number of descriptors already owned by an
     * ObjFile, which the collector closes for us. Everything past it is a raw
     * descriptor nothing owns yet. */
    const int fd[3] = { spawned->stdinFd, spawned->stdoutFd, spawned->stderrFd };
    static const char *const mode[3] = { "w", "r", "r" };
    static const char *const label[3] = { "<stdin>", "<stdout>", "<stderr>" };
    ObjFile *file[3] = { NULL, NULL, NULL };

    int wrapped = 0;
    while (wrapped < 3) {
        file[wrapped] = fileFromFd(fd[wrapped], mode[wrapped], label[wrapped]);
        if (file[wrapped] == NULL) break;      /* it closed fd[wrapped] itself */
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
    dictPut(result, "pid", INT_VAL(spawned->pid));
    dictPut(result, "stdin", OBJ_VAL(toChild));
    dictPut(result, "stdout", OBJ_VAL(fromChild));
    dictPut(result, "stderr", OBJ_VAL(fromErr));
    dictPut(result, "wait",
            boundToPid(spawned->pid, nOsWait, "wait", 1, 1));
    dictPut(result, "poll",
            boundToPid(spawned->pid, nOsPoll, "poll", 1, 1));
    dictPut(result, "signal",
            boundToPid(spawned->pid, nOsSignal, "signal", 2, 2));
    jaiGCPopRoots(4);

    *out = OBJ_VAL(result);
    return true;
}

/* __prim__.os_spawn(argv, options) — spec Appendix C. */
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
        /* ENOENT here is "the program is not on PATH", which lib/std/os.jai
         * documents as FileNotFoundError; anything else is an OSError. */
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
    dictPut(result, "exit_code", INT_VAL(spawned.exitCode));
    dictPut(result, "stdout", OBJ_VAL(outText));
    dictPut(result, "stderr", OBJ_VAL(errText));
    jaiGCPopRoots(3);

    *out = OBJ_VAL(result);
    return true;
}

void jaiRegisterOSPrimitives(void) {
    jaiDefineNative("__prim__.os_env",      nOsEnv,      0, 2);
    jaiDefineNative("__prim__.os_argv",     nOsArgv,     0, 0);
    jaiDefineNative("__prim__.module_path", nModulePath,   0, 0);
    jaiDefineNative("__prim__.os_exit",     nOsExit,     0, 1);
    jaiDefineNative("__prim__.os_cwd",      nOsCwd,      0, 0);
    jaiDefineNative("__prim__.os_chdir",    nOsChdir,    1, 1);
    jaiDefineNative("__prim__.os_spawn",    nOsSpawn,    2, 2);
    jaiDefineNative("__prim__.os_platform", nOsPlatform, 0, 0);

    /* The filesystem group is `io_*` in Appendix C, not `os_*`: `os_` is env,
     * argv, exit, cwd, chdir, spawn and the platform name. `exists` and
     * `is_dir` are not primitives — std.io derives both from io_stat, and it
     * derives path joining and the temp directory the same way. */
    jaiDefineNative("__prim__.io_listdir", nIoListdir, 1, 1);
    jaiDefineNative("__prim__.io_mkdir",   nIoMkdir,   1, 2);
    jaiDefineNative("__prim__.io_remove",  nIoRemove,  1, 1);
    jaiDefineNative("__prim__.io_rename",  nIoRename,  2, 2);
    jaiDefineNative("__prim__.io_stat",    nIoStat,    1, 2);
}

/* ------------------------------------------------------------------ */
/* Garbage collector                                                    */
/* ------------------------------------------------------------------ */

static size_t heapBytes(void) {
    return jaiAllocatedBytes();
}

/* An explicit collect runs even when the collector is disabled: disabling it
 * means "do not collect behind my back", not "refuse to collect". */
static bool nGcCollect(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    size_t before = heapBytes();
    jaiGCCollect();
    size_t after = heapBytes();
    *out = INT_VAL(after < before ? (int64_t)(before - after) : 0);
    return true;
}

static bool nGcStats(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    GCState *gc = vm.gc;

    ObjDict *stats = jaiDictNew();
    jaiGCPushRoot(OBJ_VAL(stats));
    dictPut(stats, "bytes_allocated", INT_VAL((int64_t)heapBytes()));
    dictPut(stats, "next_gc", INT_VAL(gc != NULL ? (int64_t)gc->nextGC : 0));
    dictPut(stats, "min_heap", INT_VAL(gc != NULL ? (int64_t)gc->minHeap : 0));
    dictPut(stats, "grow_factor", FLOAT_VAL(gc != NULL ? gc->growFactor : 0.0));
    dictPut(stats, "collections", INT_VAL(gc != NULL ? (int64_t)gc->collections : 0));
    dictPut(stats, "total_freed", INT_VAL(gc != NULL ? (int64_t)gc->totalFreed : 0));
    dictPut(stats, "pause_seconds",
            FLOAT_VAL(gc != NULL ? gc->totalPauseSeconds : 0.0));
    dictPut(stats, "temp_roots", INT_VAL(gc != NULL ? gc->tempRootCount : 0));
    dictPut(stats, "enabled", BOOL_VAL(gc != NULL ? gc->enabled : false));
    dictPut(stats, "stress", BOOL_VAL(gc != NULL ? gc->stress : false));
    jaiGCPopRoot();

    *out = OBJ_VAL(stats);
    return true;
}

static bool nGcDisable(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    jaiGCEnable(false);
    *out = NULL_VAL;
    return true;
}

static bool nGcEnable(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    jaiGCEnable(true);
    *out = NULL_VAL;
    return true;
}

void jaiRegisterGCPrimitives(void) {
    jaiDefineNative("__prim__.gc_collect",   nGcCollect,  0, 0);
    jaiDefineNative("__prim__.gc_stats",     nGcStats,    0, 0);
    jaiDefineNative("__prim__.gc_disable",   nGcDisable,  0, 0);
    jaiDefineNative("__prim__.gc_enable",    nGcEnable,   0, 0);
}

/* ------------------------------------------------------------------ */
/* Reflection                                                           */
/* ------------------------------------------------------------------ */

/* The binding eval() compiles into and then removes. It is not of the form
 * `__x__`, which the parser reserves (§2.1), and it starts with an underscore,
 * which is what stops the checker warning that it is never read. */
static const char kEvalSlot[] = "__jai_eval";

/* A native runs on the caller's frame, so the caller's module is the top one.
 * That is what makes eval() and globals() see the names the caller sees. */
static ObjModule *callerModule(void) {
    if (vm.frameCount > 0 && vm.frames[vm.frameCount - 1].module != NULL)
        return vm.frames[vm.frameCount - 1].module;
    if (vm.mainModule != NULL) return vm.mainModule;
    return vm.builtins;
}

static CodegenOptions reflectOptions(void) {
    CodegenOptions opts = jaiCodegenDefaults();
    opts.optLevel = vm.optLevel;
    opts.stripAsserts = vm.releaseMode;
    return opts;
}

/* A compile error inside a running program cannot be printed — the caller is
 * mid-expression and expects an exception — so the first diagnostic becomes
 * the message and the bag is emptied rather than left to leak into whatever
 * flushes it next. */
static bool throwCompileError(const char *fnName) {
    char message[512];
    const JaiDiag *first = NULL;
    for (int i = 0; i < gDiags.diags.count && first == NULL; i++) {
        if (gDiags.diags.data[i].severity == JAI_SEV_ERROR)
            first = &gDiags.diags.data[i];
    }
    /* With --warnings-as-errors the failure may be recorded as a warning; it
     * is still the reason the compile produced nothing. */
    if (first == NULL && gDiags.diags.count > 0) first = &gDiags.diags.data[0];
    if (first != NULL) {
        snprintf(message, sizeof message, "%s(): %s: %s", fnName,
                 jaiDiagCodeString(first->code),
                 first->message != NULL ? first->message : "compilation failed");
    } else {
        snprintf(message, sizeof message, "%s(): compilation failed", fnName);
    }
    jaiDiagReset(&gDiags);
    return jaiThrow(vm.cParseError, "%s", message);
}

/* Compile a fragment as a module body of `module`. Returns NULL with the
 * exception already raised. */
static ObjFunction *compileFragment(const char *source, size_t length,
                                    const char *path, ObjModule *module,
                                    const char *fnName) {
    CodegenOptions opts = reflectOptions();
    ObjFunction *fn = jaiCompileSource(source, length, path, module, &opts);
    if (fn == NULL) {
        if (!vm.hasException) (void)throwCompileError(fnName);
        return NULL;
    }
    fn->module = module;
    return fn;
}

/* Runs a fragment by calling it like any other closure. jaiVMRunModule is not
 * an option here: it resets the value stack, which the native's own caller is
 * still standing on. */
static bool runFragment(ObjFunction *fn, Value *out) {
    Value ignored;
    if (out == NULL) out = &ignored;

    jaiGCPushRoot(OBJ_VAL(fn));
    ObjClosure *closure = jaiClosureNew(fn);
    jaiGCPopRoot();

    jaiGCPushRoot(OBJ_VAL(closure));
    bool ok = jaiCallValue(OBJ_VAL(closure), 0, NULL, out);
    jaiGCPopRoot();
    return ok;
}

static bool nReflectCompile(int argc, Value *args, Value *out) {
    ObjString *source;
    if (!jaiArgString(args[0], 1, "compile", &source)) return false;

    const char *path = "<compiled>";
    if (argc >= 2 && !IS_NULL(args[1])) {
        ObjString *name;
        if (!jaiArgString(args[1], 2, "compile", &name)) return false;
        if (name->length > 0) path = name->chars;
    }

    ObjModule *module = callerModule();
    ObjFunction *fn = compileFragment(source->chars, source->length, path,
                                      module, "compile");
    if (fn == NULL) return false;

    jaiGCPushRoot(OBJ_VAL(fn));
    ObjClosure *closure = jaiClosureNew(fn);
    jaiGCPopRoot();
    *out = OBJ_VAL(closure);
    return true;
}

/* An expression is evaluated by binding it to a module-level name and reading
 * that name back: the module body is the only thing the code generator emits,
 * and a body discards the value of an expression statement. The binding is
 * removed afterwards so eval leaves no trace in the module. */
static bool nReflectEval(int argc, Value *args, Value *out) {
    (void)argc;
    ObjString *source;
    if (!jaiArgString(args[0], 1, "eval", &source)) return false;

    JaiBuf text;
    jaiBufInit(&text);
    jaiBufAppendStr(&text, "let ");
    jaiBufAppendStr(&text, kEvalSlot);
    /* The expression gets a line of its own inside the parentheses so that a
     * trailing comment cannot swallow the closing one. */
    jaiBufAppendStr(&text, " = (\n");
    jaiBufAppend(&text, source->chars, source->length);
    jaiBufAppendStr(&text, "\n)\n");

    size_t length = 0;
    char *wrapped = jaiBufTakeCString(&text, &length);
    if (wrapped == NULL) return jaiThrow(vm.cRuntimeError, "out of memory in eval");

    ObjModule *module = callerModule();
    ObjFunction *fn = compileFragment(wrapped, length, "<eval>", module, "eval");
    JAI_FREE_ARRAY(char, wrapped, length + 1);
    if (fn == NULL) return false;

    if (!runFragment(fn, NULL)) return false;

    ObjString *slot = jaiStringInternC(kEvalSlot);
    Value value = NULL_VAL;
    if (module != NULL) {
        (void)jaiModuleGet(module, slot, &value);
        if (jaiTableDelete(&module->globals, OBJ_VAL(slot))) module->version++;
    }
    *out = value;
    return true;
}

/* The namespace a two-argument `exec` runs in: a module of its own, made on
 * first use and kept in vm.modules so that a later call sees what an earlier
 * one bound. jaithon.tool.test needs this — running a test file in the runner's
 * own namespace lets an `import std.str as str` in that file shadow the
 * runner's `str`, and the runner then fails inside its own reporting code. */
static ObjModule *execNamespace(ObjString *name) {
    /* vm.modules is keyed by pointer and `name` is whatever the caller passed. */
    name = jaiStringCanonical(name);
    Value existing;
    if (jaiTableGetInterned(&vm.modules, name, &existing) && IS_MODULE(existing))
        return AS_MODULE(existing);

    ObjModule *module = jaiModuleNew(name, name);
    if (module == NULL) {
        (void)jaiThrow(vm.cRuntimeError, "out of memory in exec");
        return NULL;
    }
    module->state = MOD_LOADED;      /* nothing may try to load it from disk */
    jaiGCPushRoot(OBJ_VAL(module));
    (void)jaiTableSetInterned(&vm.modules, name, OBJ_VAL(module));
    jaiGCPopRoot();
    return module;
}

static bool nReflectExec(int argc, Value *args, Value *out) {
    ObjString *source;
    if (!jaiArgString(args[0], 1, "exec", &source)) return false;

    ObjModule *module = callerModule();
    if (argc >= 2 && !IS_NULL(args[1])) {
        ObjString *space;
        if (!jaiArgString(args[1], 2, "exec", &space)) return false;
        module = execNamespace(space);
        if (module == NULL) return false;
    }

    ObjFunction *fn = compileFragment(source->chars, source->length, "<exec>",
                                      module, "exec");
    if (fn == NULL) return false;
    if (!runFragment(fn, NULL)) return false;
    *out = NULL_VAL;
    return true;
}

/* ------------------------------------------------------------------ */
/* Rendering a stream into a str                                        */
/* ------------------------------------------------------------------ */



static bool nReflectGlobals(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    ObjModule *module = callerModule();

    ObjDict *snapshot = jaiDictNew();
    jaiGCPushRoot(OBJ_VAL(snapshot));
    if (module != NULL) {
        int slot = 0;
        Value key, value;
        while (jaiTableNext(&module->globals, &slot, &key, &value)) {
            jaiGCPushRoot(key);
            jaiGCPushRoot(value);
            (void)jaiDictSet(snapshot, key, value);
            jaiGCPopRoots(2);
        }
    }
    jaiGCPopRoot();

    *out = OBJ_VAL(snapshot);
    return true;
}

/* The §7 `buildId` this binary writes into every .jaic and demands back out of
 * one. It is a hash of the C sources, so the self-hosted front end cannot
 * derive it and has to be told; a `.jaic` it writes without this stamp is
 * rejected by the reader with the header otherwise perfectly well formed. */
static bool nReflectBuildId(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    *out = INT_VAL((int64_t)jaiBuildId());
    return true;
}

void jaiRegisterReflectPrimitives(void) {
    jaiDefineNative("__prim__.jaic_build_id", nReflectBuildId,   0, 0);
    jaiDefineNative("__prim__.compile",     nReflectCompile,     1, 2);
    jaiDefineNative("__prim__.eval",        nReflectEval,        1, 1);
    jaiDefineNative("__prim__.exec",        nReflectExec,        1, 2);
    jaiDefineNative("__prim__.globals",     nReflectGlobals,     0, 0);
}

/* ------------------------------------------------------------------ */
/* Threads                                                              */
/*                                                                      */
/* A spawned thread must not touch VM state. Nothing in the VM is       */
/* synchronised: the allocator keeps one byte counter, the collector    */
/* stops the world it can see, the intern table is a plain hash table,  */
/* and every Value is a pointer into a heap only the collector may      */
/* move through. A second thread inside any of that is a data race, not */
/* a slow program.                                                      */
/*                                                                      */
/* So this is not a documented rule, it is an enforced one: thread_spawn */
/* refuses anything but a native primitive, and the worker receives two  */
/* plain integers — the address and the length of a private copy of the  */
/* caller's buffer. No Value it is handed refers to the heap, and only   */
/* an int comes back. A Jaithon closure cannot be passed at all, which   */
/* is the whole point: there is no way to express the unsafe thing.      */
/*                                                                      */
/* The thread, mutex, condition and atomic handles below come from the   */
/* shared table in handles.c.                                            */
/* ------------------------------------------------------------------ */

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
