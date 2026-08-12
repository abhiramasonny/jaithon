/* value.h — the Jaithon value representation: a 16-byte {tag, payload} pair */
#ifndef JAI_VALUE_H
#define JAI_VALUE_H

#include "common/common.h"

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
    VAL_INT,
    VAL_FLOAT,
    VAL_OBJ,
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

JAI_INLINE ValueType jaiValueType(Value v) { return v.type; }

JAI_INLINE double jaiAsDouble(Value v) {
    return IS_INT(v) ? (double)AS_INT(v) : AS_FLOAT(v);
}

typedef enum {
    OBJ_STRING, OBJ_BYTES, OBJ_LIST, OBJ_DICT, OBJ_SET, OBJ_TUPLE, OBJ_RANGE,
    OBJ_FUNCTION, OBJ_CLOSURE, OBJ_UPVALUE, OBJ_NATIVE, OBJ_BOUND,
    OBJ_CLASS, OBJ_TRAIT, OBJ_INSTANCE, OBJ_MODULE, OBJ_ENUM, OBJ_ENUM_VAL,
    OBJ_ITER, OBJ_FILE, OBJ_ENUM_CTOR, OBJ_STRBUF, OBJ_TYPE_COUNT
} ObjType;

struct Obj {
    ObjType  type;
    bool     isMarked;
    bool     subFlag;
    bool     subFlag2;
    Obj     *next;
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

JAI_INLINE bool jaiValueIsInertGlobal(Value v) {
    if (!IS_OBJ(v)) return true;
    switch (AS_OBJ(v)->type) {
    case OBJ_STRING: case OBJ_BYTES:  case OBJ_LIST:  case OBJ_DICT:
    case OBJ_SET:    case OBJ_TUPLE:  case OBJ_RANGE: case OBJ_INSTANCE:
    case OBJ_ITER:   case OBJ_FILE:   case OBJ_STRBUF:
        return true;
    default:
        return false;
    }
}

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

typedef JAI_VEC(Value) ValueArray;

void jaiValueArrayInit(ValueArray *a);
void jaiValueArrayFree(ValueArray *a);
void jaiValueArrayPush(ValueArray *a, Value v);

bool jaiValuesEqual(Value a, Value b);
bool jaiValuesIdentical(Value a, Value b);
uint64_t jaiValueHash(Value v, bool *ok);
bool jaiValueCompare(Value a, Value b, int *out);
ObjString *jaiTypeName(Value v);
const char *jaiTypeNameStatic(Value v);
ObjString *jaiValueToStr(Value v);
ObjString *jaiValueToRepr(Value v);
void jaiPrintValue(FILE *out, Value v, bool repr);

#define JAI_FMT_MAX_PARTS 24

ObjString *jaiValueFormat(const Value *parts, int count);

#endif /* JAI_VALUE_H */
