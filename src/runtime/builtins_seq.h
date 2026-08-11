/* builtins_seq.h — what the container method files share.
 *
 * The methods of list, dict, set, tuple, range and iterator are three
 * translation units — builtins_list.c, builtins_dict.c and builtins_seq.c —
 * because each is a large, self-contained table with its own primitives. This
 * header is the small part that genuinely straddles them: the method-table
 * shape they all bind through, and the seven helpers more than one of them
 * calls. Everything here is defined in builtins_seq.c.
 *
 * Not a public interface: methods.h declares what the VM calls and runtime.h
 * what the registrar calls, and neither grows an entry for anything below.
 */
#ifndef JAI_BUILTINS_SEQ_H
#define JAI_BUILTINS_SEQ_H

#include "methods.h"
#include "runtime.h"

/* --- method tables ------------------------------------------------- */

/* Arities count the receiver, because that is what the VM passes: `push(v)` is
 * two arguments. The name length is a compile-time constant so a lookup is a
 * length test and a memcmp — the interned names cannot be cached here, since
 * the intern table holds only weak references. */
typedef struct {
    const char *name;
    size_t      length;
    JaiNativeFn fn;
    int8_t      minArity;
    int8_t      maxArity;
    /* Parameter names, parallel to args[0..maxArity-1] and so starting with
     * "self". Only the methods with optional parameters declare them, since
     * those are the ones a caller has any reason to name (spec §4.2). */
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

/* A concrete index into a sequence of `count` items, negatives counting from
 * the end. */
bool jaiSeqIndexArg(Value v, int index, const char *fnName, int count,
                    int *out);

/* --- shared operations --------------------------------------------- */

/* Equality that reports a raising __eq__ instead of reading its failure as
 * "not equal". */
bool jaiSeqEqualsChecked(Value a, Value b, bool *equal);

/* Dict keys and set elements must hash, and must not be `null`. */
bool jaiSeqHashableKey(Value key, const char *fnName, const char *role);

/* The answer for every key that cannot fail: a string, an int, a float, a
 * bool, a class, a function. Only the mutable containers and the composites
 * that may hold one need the real check, and only an instance can run user
 * code to answer it.
 *
 * Inline here because it decides the question on every `d[k]` and `d.get(k)`,
 * and jaiSeqHashableKey is a four-argument cross-translation-unit call that
 * LTO did not fold: it was 2.5% of dict_ops purely as call overhead. */
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

/* Drain any iterable into a fresh list. NULL with the exception pending on
 * failure. The result is unrooted: root it before allocating again. */
ObjList *jaiSeqCollectIterable(Value v);

/* `xs.iter()` is the explicit spelling of what `for x in xs` does implicitly;
 * one implementation serves every container. */
bool jaiSeqValueIter(int argc, Value *args, Value *out);

#endif /* JAI_BUILTINS_SEQ_H */
