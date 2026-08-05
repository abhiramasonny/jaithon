/* builtins_str.h — what the str, format and bytes translation units share.
 *
 * Not a public interface. methods.h declares what the VM calls and runtime.h
 * what the registrar calls; this header exists only so that builtins_str.c,
 * builtins_format.c and builtins_bytes.c can agree on the handful of helpers
 * that genuinely straddle them, without either of those growing a private
 * entry. Everything here is defined in builtins_str.c unless noted.
 */
#ifndef JAI_BUILTINS_STR_H
#define JAI_BUILTINS_STR_H

#include "methods.h"
#include "runtime.h"

#define JAI_COUNT_OF(a) (sizeof(a) / sizeof((a)[0]))

/* --- method tables ------------------------------------------------- */

typedef struct {
    const char *name;
    JaiNativeFn fn;
    int8_t      minArity;    /* the receiver counts as argument 0 */
    int8_t      maxArity;    /* -1 = variadic */
    /* Parameter names for a keyword call, args[0] ("self") first; NULL when
     * the method takes positional arguments only. */
    const char *const *params;
} JaiStrMethodEntry;

/* Hashes of the table names, so a lookup is a run of 64-bit compares. They are
 * computed rather than cached as ObjString pointers because interned strings
 * are weak GC roots: a cached pointer would dangle after a collection. */
typedef struct {
    const JaiStrMethodEntry *entries;
    size_t                   count;
    uint64_t                *hashes;
    bool                     ready;
} JaiStrMethodTable;

bool jaiStrLookupMethod(JaiStrMethodTable *table, Value receiver,
                        ObjString *name, Value *out);

/* --- scalars, byte offsets and text plumbing ----------------------- */

/* Byte offset of scalar `index`; clamped to the byte length when past the end. */
size_t jaiStrByteOffsetOf(ObjString *s, size_t index);

/* First occurrence of `needle` in `hay`, or NULL. */
const char *jaiStrFindBytes(const char *hay, size_t hayLen,
                            const char *needle, size_t needleLen);

/* Turns a finished buffer into a str Value, consuming the buffer either way. */
bool jaiStrTakeBuf(JaiBuf *buf, Value *out);

bool jaiStrWantStr(Value v, const char *method, const char *what,
                   ObjString **out);
bool jaiStrWantInt(Value v, const char *method, const char *what, int64_t *out);

/* An absent or null argument falls back; anything else must be an int. */
bool jaiStrOptInt(int argc, Value *args, int slot, const char *method,
                  const char *what, int64_t fallback, int64_t *out);

/* jaiListNew and jaiBytesNew take an int; text long enough to overflow one
 * simply starts with no reservation and grows. */
int jaiStrCapacityFor(size_t n);

/* Digit value of one character in any base up to 36, or -1. */
int jaiStrDigitValue(char c);

/* A null slice bound means "the natural end", which is what makes s[a:] work. */
bool jaiStrSliceBound(Value v, const char *name, int64_t fallback, int64_t *out);

/* --- the format engine (builtins_format.c) ------------------------- */

/* Renders one value under one format spec (spec §5.2), appending to `out`.
 * `where` names the caller in any diagnostic. */
bool jaiFormatValue(JaiBuf *out, Value value, const char *spec, size_t specLen,
                    const char *where);

/* Runs a whole `str.format` template over args[1..argc). */
bool jaiFormatTemplate(ObjString *tmpl, Value *args, int argc, Value *out,
                       const char *where);

/* --- registration -------------------------------------------------- */

/* Registers one primitive under both spellings the rest of the tree might use:
 * a member of the `__prim__` module, which is how `__prim__.str_cmp(a, b)` in
 * lib/std resolves, and a builtins global with the dotted name, for a front end
 * that folds the whole path into one identifier. */
void jaiStrDefinePrim(ObjModule *ns, const char *name, JaiNativeFn fn,
                      int minArity, int maxArity);

/* The bytes half of jaiRegisterStringPrimitives (builtins_bytes.c). */
void jaiBytesRegisterPrimitives(ObjModule *ns);

#endif /* JAI_BUILTINS_STR_H */
