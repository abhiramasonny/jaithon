/* builtins_io.h — what the I/O family of translation units share.
 *
 * Not a public interface. methods.h declares what the VM calls and runtime.h
 * what the registrar calls; this header exists only so that builtins_io.c,
 * builtins_fs.c, builtins_process.c and builtins_gc.c can agree on the
 * handful of helpers that genuinely straddle them, without any of those
 * growing a private entry. Everything here is defined in builtins_io.c.
 */
#ifndef JAI_BUILTINS_IO_H
#define JAI_BUILTINS_IO_H

#include "runtime/runtime.h"

/* --- errno -> exception ---------------------------------------------- */

bool jaiIOThrowErrno(int err, const char *what, const char *path);
bool jaiIOThrowErrno2(int err, const char *what, const char *from,
                      const char *to);

/* --- paths ------------------------------------------------------------- */

/* A NUL-terminated C string for `path`: `path->chars` itself when already
 * terminated, otherwise a malloc'd copy stashed in `*tmp` (NULL when none was
 * needed) for the caller to release with jaiIOPathDone. Despite the name this
 * is not path-specific -- anything reaching a native as an argument can be an
 * unterminated concatenation view (see jaiStringTerminated's comment in
 * object.h), and a plain malloc'd copy needs no GC root to stay valid across
 * however many further calls and allocations the rest of the native function
 * makes, unlike jaiStringTerminated's result. So this is reused across
 * builtins_io.c/fs.c/process.c for anything else that ends up in a
 * NUL-scanning libc call and has to survive more than one immediate use: an
 * open() mode, an os_env() name or value, an os_spawn() argv entry or cwd. */
const char *jaiIOPathCStr(ObjString *path, char **tmp);
void jaiIOPathDone(char *tmp);

bool jaiIOCheckPath(ObjString *path, const char *fnName);

/* --- dict building ------------------------------------------------------ */

void jaiIODictPut(ObjDict *dict, const char *key, Value value);

#endif /* JAI_BUILTINS_IO_H */
