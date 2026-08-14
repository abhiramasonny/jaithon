/* builtins_str.h — what the str, format and bytes translation units share;
 * not a public interface. Everything here is defined in builtins_str.c
 * unless noted. */
#ifndef JAI_BUILTINS_STR_H
#define JAI_BUILTINS_STR_H

#include "runtime/methods.h"
#include "runtime/runtime.h"

#define JAI_COUNT_OF(a) (sizeof(a) / sizeof((a)[0]))

/* --- method tables ------------------------------------------------- */

typedef struct {
    const char *name;
    JaiNativeFn fn;
    int8_t      minArity;
    int8_t      maxArity;    /* -1 = variadic */
    const char *const *params;
} JaiStrMethodEntry;

typedef struct {
    const JaiStrMethodEntry *entries;
    size_t                   count;
    uint64_t                *hashes;
    bool                     ready;
} JaiStrMethodTable;

bool jaiStrLookupMethod(JaiStrMethodTable *table, Value receiver,
                        ObjString *name, Value *out);

/* --- scalars, byte offsets and text plumbing ----------------------- */

size_t jaiStrByteOffsetOf(ObjString *s, size_t index);

const char *jaiStrFindBytes(const char *hay, size_t hayLen,
                            const char *needle, size_t needleLen);

bool jaiStrTakeBuf(JaiBuf *buf, Value *out);

bool jaiStrWantStr(Value v, const char *method, const char *what,
                   ObjString **out);
bool jaiStrWantInt(Value v, const char *method, const char *what, int64_t *out);

bool jaiStrOptInt(int argc, Value *args, int slot, const char *method,
                  const char *what, int64_t fallback, int64_t *out);

int jaiStrCapacityFor(size_t n);

int jaiStrDigitValue(char c);

bool jaiStrSliceBound(Value v, const char *name, int64_t fallback, int64_t *out);

/* --- the format engine (builtins_format.c) ------------------------- */

bool jaiFormatValue(JaiBuf *out, Value value, const char *spec, size_t specLen,
                    const char *where);

bool jaiFormatTemplate(ObjString *tmpl, Value *args, int argc, Value *out,
                       const char *where);

/* --- registration -------------------------------------------------- */

void jaiStrDefinePrim(ObjModule *ns, const char *name, JaiNativeFn fn,
                      int minArity, int maxArity);

void jaiBytesRegisterPrimitives(ObjModule *ns);

#endif /* JAI_BUILTINS_STR_H */
