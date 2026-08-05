/* types.h — the semantic type system (spec §2).
 *
 * Types are interned: two structurally equal types are the same pointer, so
 * assignability checks are mostly pointer comparisons.
 */
#ifndef JAI_TYPES_H
#define JAI_TYPES_H

#include "../common/common.h"
#include "../common/diag.h"

typedef struct JaiType JaiType;

typedef enum {
    TY_ANY,          /* dynamic; assignable both ways with a runtime guard */
    TY_NEVER,        /* bottom; the type of `throw` and of a diverging branch */
    TY_NULL,
    TY_BOOL,
    TY_INT,
    TY_FLOAT,
    TY_STR,
    TY_BYTES,
    TY_LIST,         /* args[0] = element  */
    TY_DICT,         /* args[0] = key, args[1] = value */
    TY_SET,          /* args[0] = element  */
    TY_TUPLE,        /* args[..] = members */
    TY_RANGE,
    TY_FN,           /* args[0..n-1] = params, ret = return type */
    TY_CLASS,
    TY_TRAIT,
    TY_ENUM,
    TY_MODULE,
    TY_UNION,        /* args[..] = members, flattened and sorted */
    TY_GENERIC_PARAM,/* T inside a generic declaration */
    TY_ALIAS,        /* resolved lazily to `ret` */
} TypeKind;

/* Declaration info shared by class/trait/enum types. */
typedef struct TypeDecl TypeDecl;

struct JaiType {
    TypeKind    kind;
    const char *name;        /* interned; for class/trait/enum/generic/alias */
    JaiType   **args;
    int         argCount;
    JaiType    *ret;         /* TY_FN return; TY_ALIAS target */
    TypeDecl   *decl;        /* class/trait/enum declaration data */
    uint8_t     fnFlags;     /* variadic / kwrest */
    uint64_t    hash;
};

/* ------------------------------------------------------------------ */
/* The type universe                                                    */
/* ------------------------------------------------------------------ */

typedef struct {
    JaiArena arena;
    JAI_VEC(JaiType *) interned;
    /* Singletons, valid after jaiTypesInit. */
    JaiType *tAny, *tNever, *tNull, *tBool, *tInt, *tFloat, *tStr, *tBytes,
            *tRange, *tVoid;
} TypeUniverse;

extern TypeUniverse gTypes;

void jaiTypesInit(void);
void jaiTypesFree(void);

JaiType *jaiTypeList(JaiType *elem);
JaiType *jaiTypeDict(JaiType *key, JaiType *value);
JaiType *jaiTypeSet(JaiType *elem);
JaiType *jaiTypeTuple(JaiType **members, int count);
JaiType *jaiTypeFn(JaiType **params, int count, JaiType *ret, uint8_t flags);
JaiType *jaiTypeOptional(JaiType *inner);            /* T | null */
JaiType *jaiTypeUnion(JaiType **members, int count); /* flattens and sorts */
JaiType *jaiTypeNamed(TypeKind kind, const char *name, TypeDecl *decl);
JaiType *jaiTypeGenericParam(const char *name);

/* ------------------------------------------------------------------ */
/* Relations                                                            */
/* ------------------------------------------------------------------ */

/* Can a value of type `from` be stored where `to` is expected?
 * Sets *needsGuard when the assignment needs a conversion node in the tree:
 * either a runtime check (`from` is TY_ANY and `to` is concrete) or the int to
 * float widening below. */
bool jaiTypeAssignable(JaiType *from, JaiType *to, bool *needsGuard);
/* The float type an `int` must be converted to in order to be stored as `to`,
 * or NULL when no widening applies (spec §2.2). Top-level only: `list[int]`
 * does not widen to `list[float]`, and float never narrows to int. */
JaiType *jaiTypeWidenTarget(JaiType *from, JaiType *to);
bool jaiTypeEquals(JaiType *a, JaiType *b);
bool jaiTypeIsSubtype(JaiType *sub, JaiType *super);
/* Least upper bound; used for `if` expressions and list literals. */
JaiType *jaiTypeJoin(JaiType *a, JaiType *b);
/* Narrowed type of `t` after a successful `is C` / `!= null` test. */
JaiType *jaiTypeNarrow(JaiType *t, JaiType *by, bool positive);

bool jaiTypeIsNumeric(JaiType *t);
bool jaiTypeIsOptional(JaiType *t);
bool jaiTypeIsCallable(JaiType *t);
bool jaiTypeIsIterable(JaiType *t, JaiType **outElem);
bool jaiTypeIsIndexable(JaiType *t, JaiType **outIndex, JaiType **outResult);

/* Result type of `left op right`, or NULL when the operator does not apply. */
JaiType *jaiTypeBinaryResult(int opKind, JaiType *left, JaiType *right);
JaiType *jaiTypeUnaryResult(int opKind, JaiType *operand);

/* Substitute generic parameters: jaiTypeSubstitute(list[T], {T: int}) -> list[int] */
JaiType *jaiTypeSubstitute(JaiType *t, const char **names, JaiType **values,
                           int count);

/* Rendered form for diagnostics: "dict[str, list[int]]". Returned buffer is
 * owned by the type universe and valid until the next call on this thread. */
const char *jaiTypeToString(JaiType *t);

#endif /* JAI_TYPES_H */
