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

const char *jaiIOPathCStr(ObjString *path, char **tmp);
void jaiIOPathDone(char *tmp);

bool jaiIOCheckPath(ObjString *path, const char *fnName);

/* --- dict building ------------------------------------------------------ */

void jaiIODictPut(ObjDict *dict, const char *key, Value value);

#endif /* JAI_BUILTINS_IO_H */
