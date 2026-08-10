/* value.h — the Jaithon value representation.
 *
 * A Value is 16 bytes: an 8-byte tag word and an 8-byte payload. This is
 * deliberately not NaN-boxed: Jaithon distinguishes int from float as separate
 * types (spec §2.4), so a tagged union is both simpler and faster here than
 * NaN-boxing, which would have to unbox 64-bit ints anyway.
 */
#ifndef JAI_VALUE_H
#define JAI_VALUE_H

#include "../common/common.h"

typedef struct Obj        Obj;
typedef struct ObjString  ObjString;
typedef struct ObjList    ObjList;
typedef struct ObjDict    ObjDict;
typedef struct ObjSet     ObjSet;
typedef struct ObjTuple   ObjTuple;
typedef struct ObjRange   ObjRange;
typedef struct ObjBytes   ObjBytes;
typedef struct ObjFunction ObjFunction;
typedef struct ObjClosure ObjClosure;
typedef struct ObjUpvalue ObjUpvalue;
typedef struct ObjNative  ObjNative;
typedef struct ObjClass   ObjClass;
typedef struct ObjTrait   ObjTrait;
typedef struct ObjInstance ObjInstance;
typedef struct ObjBound   ObjBound;
typedef struct ObjModule  ObjModule;
typedef struct ObjEnum    ObjEnum;
typedef struct ObjEnumVal ObjEnumVal;
typedef struct ObjEnumCtor ObjEnumCtor;
typedef struct ObjIter    ObjIter;
typedef struct ObjFile    ObjFile;

typedef enum {
    VAL_NULL = 0,
    VAL_BOOL,
    VAL_INT,      /* int64_t  */
    VAL_FLOAT,    /* double   */
    VAL_OBJ,      /* heap object; see obj->type */
} ValueType;

typedef struct {
    ValueType type;
    union {
        bool    boolean;
        int64_t integer;
        double  number;
        Obj    *obj;
    } as;
} Value;

/* ------------------------------------------------------------------ */
/* Constructors and predicates                                         */
/* ------------------------------------------------------------------ */

#define NULL_VAL          ((Value){VAL_NULL,  {.integer = 0}})
#define BOOL_VAL(b)       ((Value){VAL_BOOL,  {.boolean = (b)}})
#define INT_VAL(i)        ((Value){VAL_INT,   {.integer = (int64_t)(i)}})
#define FLOAT_VAL(d)      ((Value){VAL_FLOAT, {.number  = (double)(d)}})
#define OBJ_VAL(o)        ((Value){VAL_OBJ,   {.obj     = (Obj *)(o)}})

#define IS_NULL(v)        ((v).type == VAL_NULL)
#define IS_BOOL(v)        ((v).type == VAL_BOOL)
#define IS_INT(v)         ((v).type == VAL_INT)
#define IS_FLOAT(v)       ((v).type == VAL_FLOAT)
#define IS_NUMBER(v)      ((v).type == VAL_INT || (v).type == VAL_FLOAT)
#define IS_OBJ(v)         ((v).type == VAL_OBJ)

#define AS_BOOL(v)        ((v).as.boolean)
#define AS_INT(v)         ((v).as.integer)
#define AS_FLOAT(v)       ((v).as.number)
#define AS_OBJ(v)         ((v).as.obj)

/* Which of the five kinds `v` is.
 *
 * A five-way branch on a value is a switch, and a switch needs something to
 * switch on. Reading `.type` directly works only while a Value has a tag field;
 * a NaN-boxed one does not, so every such switch goes through here instead and
 * the representation stays behind the macros. */
JAI_INLINE ValueType jaiValueType(Value v) { return v.type; }

/* Numeric coercion helper: int or float -> double. Undefined for other types. */
JAI_INLINE double jaiAsDouble(Value v) {
    return IS_INT(v) ? (double)AS_INT(v) : AS_FLOAT(v);
}

/* ------------------------------------------------------------------ */
/* Object header                                                       */
/* ------------------------------------------------------------------ */

typedef enum {
    OBJ_STRING, OBJ_BYTES, OBJ_LIST, OBJ_DICT, OBJ_SET, OBJ_TUPLE, OBJ_RANGE,
    OBJ_FUNCTION, OBJ_CLOSURE, OBJ_UPVALUE, OBJ_NATIVE, OBJ_BOUND,
    OBJ_CLASS, OBJ_TRAIT, OBJ_INSTANCE, OBJ_MODULE, OBJ_ENUM, OBJ_ENUM_VAL,
    OBJ_ITER, OBJ_FILE, OBJ_ENUM_CTOR,
    /* Appended, never inserted: the numbering is written into images. */
    OBJ_STRBUF,
    /* No OBJ_GENERIC: generics are erased at run time (spec §6.1), so no heap
     * object ever carries one. */
    OBJ_TYPE_COUNT
} ObjType;

struct Obj {
    ObjType  type;
    bool     isMarked;
    /* Three bytes of alignment padding sit between isMarked and next whatever
     * we do, so a subtype whose own bool would otherwise add a whole aligned
     * word to its header keeps it here instead. Only ObjString uses this, for
     * whether the string is in the intern table; that moves the flexible
     * character array eight bytes earlier on every string in the heap. */
    bool     subFlag;
    /* The third padding byte, which was going spare. ObjString uses it for
     * whether a later append into the same buffer overwrote this string's NUL
     * terminator; see jaiStringCStr. */
    bool     subFlag2;
    Obj     *next;        /* intrusive list of every heap object */
};

#define OBJ_TYPE(v)       (AS_OBJ(v)->type)
#define IS_OBJ_TYPE(v, t) (IS_OBJ(v) && AS_OBJ(v)->type == (t))

#define IS_STRING(v)      IS_OBJ_TYPE(v, OBJ_STRING)
#define IS_LIST(v)        IS_OBJ_TYPE(v, OBJ_LIST)
#define IS_DICT(v)        IS_OBJ_TYPE(v, OBJ_DICT)
#define IS_SET(v)         IS_OBJ_TYPE(v, OBJ_SET)
#define IS_TUPLE(v)       IS_OBJ_TYPE(v, OBJ_TUPLE)
#define IS_RANGE(v)       IS_OBJ_TYPE(v, OBJ_RANGE)
#define IS_BYTES(v)       IS_OBJ_TYPE(v, OBJ_BYTES)
#define IS_FUNCTION(v)    IS_OBJ_TYPE(v, OBJ_FUNCTION)
#define IS_CLOSURE(v)     IS_OBJ_TYPE(v, OBJ_CLOSURE)
#define IS_NATIVE(v)      IS_OBJ_TYPE(v, OBJ_NATIVE)
#define IS_CLASS(v)       IS_OBJ_TYPE(v, OBJ_CLASS)
#define IS_TRAIT(v)       IS_OBJ_TYPE(v, OBJ_TRAIT)
#define IS_INSTANCE(v)    IS_OBJ_TYPE(v, OBJ_INSTANCE)
#define IS_BOUND(v)       IS_OBJ_TYPE(v, OBJ_BOUND)
#define IS_MODULE(v)      IS_OBJ_TYPE(v, OBJ_MODULE)
#define IS_ENUM(v)        IS_OBJ_TYPE(v, OBJ_ENUM)
#define IS_ENUM_VAL(v)    IS_OBJ_TYPE(v, OBJ_ENUM_VAL)
#define IS_ENUM_CTOR(v)   IS_OBJ_TYPE(v, OBJ_ENUM_CTOR)
#define IS_ITER(v)        IS_OBJ_TYPE(v, OBJ_ITER)
#define IS_FILE(v)        IS_OBJ_TYPE(v, OBJ_FILE)

#define AS_STRING(v)      ((ObjString *)AS_OBJ(v))
#define AS_CSTRING(v)     (((ObjString *)AS_OBJ(v))->chars)
#define AS_LIST(v)        ((ObjList *)AS_OBJ(v))
#define AS_DICT(v)        ((ObjDict *)AS_OBJ(v))
#define AS_SET(v)         ((ObjSet *)AS_OBJ(v))
#define AS_TUPLE(v)       ((ObjTuple *)AS_OBJ(v))
#define AS_RANGE(v)       ((ObjRange *)AS_OBJ(v))
#define AS_BYTES(v)       ((ObjBytes *)AS_OBJ(v))
#define AS_FUNCTION(v)    ((ObjFunction *)AS_OBJ(v))
#define AS_CLOSURE(v)     ((ObjClosure *)AS_OBJ(v))
#define AS_NATIVE(v)      ((ObjNative *)AS_OBJ(v))
#define AS_CLASS(v)       ((ObjClass *)AS_OBJ(v))
#define AS_TRAIT(v)       ((ObjTrait *)AS_OBJ(v))
#define AS_INSTANCE(v)    ((ObjInstance *)AS_OBJ(v))
#define AS_BOUND(v)       ((ObjBound *)AS_OBJ(v))
#define AS_MODULE(v)      ((ObjModule *)AS_OBJ(v))
#define AS_ENUM(v)        ((ObjEnum *)AS_OBJ(v))
#define AS_ENUM_VAL(v)    ((ObjEnumVal *)AS_OBJ(v))
#define AS_ENUM_CTOR(v)   ((ObjEnumCtor *)AS_OBJ(v))
#define AS_ITER(v)        ((ObjIter *)AS_OBJ(v))
#define AS_FILE(v)        ((ObjFile *)AS_OBJ(v))

/* ------------------------------------------------------------------ */
/* Value array (constant pools, list storage)                          */
/* ------------------------------------------------------------------ */

typedef JAI_VEC(Value) ValueArray;

void jaiValueArrayInit(ValueArray *a);
void jaiValueArrayFree(ValueArray *a);
void jaiValueArrayPush(ValueArray *a, Value v);

/* ------------------------------------------------------------------ */
/* Core value operations                                               */
/* ------------------------------------------------------------------ */

/* Structural equality (`==`). Calls __eq__ for instances. May raise. */
bool jaiValuesEqual(Value a, Value b);
/* Identity (`is`). Never calls user code. */
bool jaiValuesIdentical(Value a, Value b);
/* Hash for dict/set keys. Raises TypeError for unhashable values. */
uint64_t jaiValueHash(Value v, bool *ok);
/* Ordering for <,<=,>,>=. Returns -1/0/1 in *out; false if not comparable. */
bool jaiValueCompare(Value a, Value b, int *out);

/* The name of a value's type as user code sees it: "int", "list[int]",
 * "Account". Returns an interned ObjString. */
ObjString *jaiTypeName(Value v);
/* Static type-name for a ValueType/ObjType pair, no allocation. */
const char *jaiTypeNameStatic(Value v);

/* Human-readable form (`str()`): strings render without quotes. */
ObjString *jaiValueToStr(Value v);
/* Debug form (`repr()`): strings render with quotes and escapes. */
ObjString *jaiValueToRepr(Value v);
/* Fast path used by the disassembler and traceback printer. */
void jaiPrintValue(FILE *out, Value v, bool repr);

/* Most parts one OP_FORMAT can join. An f-string with more falls back to the
 * build-a-list-and-join lowering, so this only has to cover what fits in the
 * instruction's u24 literal mask. */
#define JAI_FMT_MAX_PARTS 24

/* str() of every part, concatenated, in one allocation — what an f-string
 * lowers to. `parts` must be reachable from a root (OP_FORMAT leaves them on
 * the value stack) because building the result can collect. Returns NULL with
 * an exception pending when a user __str__ raises. */
ObjString *jaiValueFormat(const Value *parts, int count);

#endif /* JAI_VALUE_H */
