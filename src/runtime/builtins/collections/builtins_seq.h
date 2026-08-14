/* builtins_seq.h — what the container method files (list/dict/set/tuple/
 * range/iterator) share; not a public interface. Everything here is defined
 * in builtins_seq.c. */
#ifndef JAI_BUILTINS_SEQ_H
#define JAI_BUILTINS_SEQ_H

#include "runtime/methods.h"
#include "runtime/runtime.h"

/* --- method tables ------------------------------------------------- */

typedef struct {
    const char *name;
    size_t      length;
    JaiNativeFn fn;
    int8_t      minArity;
    int8_t      maxArity;
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

bool jaiSeqReceiverError(const char *fnName, const char *expected, Value got);

bool jaiSeqIndexArg(Value v, int index, const char *fnName, int count,
                    int *out);

/* --- shared operations --------------------------------------------- */

bool jaiSeqEqualsChecked(Value a, Value b, bool *equal);

bool jaiSeqHashableKey(Value key, const char *fnName, const char *role);

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

#define JAI_SEQ_KEY_OK(key, fnName, role)                                      \
    (JAI_LIKELY(jaiSeqKeyCannotFail(key)) ||                                   \
     jaiSeqHashableKey((key), (fnName), (role)))

ObjList *jaiSeqCollectIterable(Value v);

bool jaiSeqValueIter(int argc, Value *args, Value *out);

#endif /* JAI_BUILTINS_SEQ_H */
