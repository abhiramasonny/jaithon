/* builtins_seq.h — what the container method files (list/dict/set/tuple/
 * range/iterator) share; not a public interface. Everything here is defined
 * in builtins_seq.c. */
#ifndef JAI_BUILTINS_SEQ_H
#define JAI_BUILTINS_SEQ_H

#include "runtime/methods.h"
#include "runtime/runtime.h"

/* --- method tables ------------------------------------------------- */

/* Arity counts the receiver (`push(v)` is 2 arguments). Names compare by
 * length+memcmp, not cached interned pointers — the intern table holds only
 * weak references. */
typedef struct {
    const char *name;
    size_t      length;
    JaiNativeFn fn;
    int8_t      minArity;
    int8_t      maxArity;
    /* Parallel to args[0..maxArity-1] (so starts with "self"); only methods
     * with optional parameters declare it (spec §4.2). */
    const char *const *params;
} JaiSeqMethod;

#define JAI_METHOD(name, fn, min, max) \
    {(name), sizeof(name) - 1, (fn), (min), (max), NULL}
#define JAI_METHOD_KW(name, fn, min, max, params) \
    {(name), sizeof(name) - 1, (fn), (min), (max), (params)}

bool jaiSeqBindFrom(const JaiSeqMethod *table, int count, Value receiver,
                    ObjString *name, Value *out);

#define JAI_BIND_FROM(table, receiver, name, out)                              \
    jaiSeqBindFrom((table), (int)(sizeof(table) / sizeof((table)[0])),          \
                   (receiver), (name), (out))

/* --- receivers and arguments --------------------------------------- */

Value jaiSeqOptArg(int argc, Value *args, int index);

/* Reachable only when a bound method outlives the value it was bound to in a
 * way the VM cannot see; cheap enough to check, fatal enough to be worth it. */
bool jaiSeqReceiverError(const char *fnName, const char *expected, Value got);

/* Negative index counts from the end. */
bool jaiSeqIndexArg(Value v, int index, const char *fnName, int count,
                    int *out);

/* --- shared operations --------------------------------------------- */

/* Equality that reports a raising __eq__ instead of reading its failure as
 * "not equal". */
bool jaiSeqEqualsChecked(Value a, Value b, bool *equal);

/* Dict keys and set elements must hash, and must not be `null`. */
bool jaiSeqHashableKey(Value key, const char *fnName, const char *role);

/* Inlined: this runs on every `d[k]`/`d.get(k)`, and the out-of-line call was
 * 2.5% of dict_ops as pure call overhead (LTO didn't fold it). */
JAI_INLINE bool jaiSeqKeyCannotFail(Value key) {
    if (!IS_OBJ(key)) return !IS_NULL(key);
    switch (OBJ_TYPE(key)) {
        case OBJ_LIST: case OBJ_DICT: case OBJ_SET:
        case OBJ_TUPLE: case OBJ_ENUM_VAL: case OBJ_INSTANCE:
            return false;
        default:
            return true;
    }
}

/* jaiSeqHashableKey with the common answer settled without a call. */
#define JAI_SEQ_KEY_OK(key, fnName, role)                                      \
    (JAI_LIKELY(jaiSeqKeyCannotFail(key)) ||                                   \
     jaiSeqHashableKey((key), (fnName), (role)))

/* Result is unrooted — root it before allocating again. */
ObjList *jaiSeqCollectIterable(Value v);

/* `xs.iter()` is the explicit spelling of what `for x in xs` does implicitly. */
bool jaiSeqValueIter(int argc, Value *args, Value *out);

#endif /* JAI_BUILTINS_SEQ_H */
