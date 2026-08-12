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

/* `what` reads as a verb phrase: "cannot open", "cannot read from". Spec
 * §7.2 gives IOError exactly two subclasses; these two throw the right one
 * for a given errno, so one errno always produces one exception class and
 * every message names the path(s) involved. */
bool jaiIOThrowErrno(int err, const char *what, const char *path);
bool jaiIOThrowErrno2(int err, const char *what, const char *from,
                      const char *to);

/* --- paths ------------------------------------------------------------- */

/* A NUL-terminated pointer for a syscall.
 *
 * A string can be a view into a shared append buffer, in which case the byte
 * after it belongs to a later concatenation rather than being a terminator --
 * and `dir + "/" + name` is exactly how paths get built. Costs nothing for an
 * ordinary string; copies only for a view that a later append ran past. Pass
 * the same `tmp` to jaiIOPathDone when the call is finished. */
const char *jaiIOPathCStr(ObjString *path, char **tmp);
void jaiIOPathDone(char *tmp);

/* Rejects an empty path or one containing a NUL byte -- either would reach
 * libc as something other than the path the caller wrote. */
bool jaiIOCheckPath(ObjString *path, const char *fnName);

/* --- dict building ------------------------------------------------------ */

/* Every primitive that answers with a record builds it this way, so the keys
 * are interned once and the dict is rooted for the whole construction. */
void jaiIODictPut(ObjDict *dict, const char *key, Value value);

#endif /* JAI_BUILTINS_IO_H */
