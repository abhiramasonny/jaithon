// value.c houses equality, hashing, ordering, and rendering of Values.
#include <math.h>
#include <stdlib.h>

#include "vm/value.h"
#include "vm/object/object.h"
#include "vm/table.h"
#include "vm/vm.h"

#define JAI_I64_MIN_D (-9223372036854775808.0)
#define JAI_I64_SUP_D (9223372036854775808.0)

#define JAI_EQ_MAX_DEPTH   256
#define JAI_HASH_MAX_DEPTH 256
#define JAI_RENDER_MAX_DEPTH 64

void jaiValueArrayInit(ValueArray *a) { JAI_VEC_INIT(a); }
void jaiValueArrayFree(ValueArray *a) { JAI_VEC_FREE(Value, a); }
void jaiValueArrayPush(ValueArray *a, Value v) { JAI_VEC_PUSH(Value, a, v); }

JAI_INLINE bool tempRoot(Value v) {
    if (!IS_OBJ(v) || vm.gc == NULL) return false;
    jaiPushRoot(v);
    return true;
}

JAI_INLINE void tempUnroot(bool rooted) {
    if (rooted) jaiPopRoot();
}

JAI_INLINE bool doubleIsExactInt(double d, int64_t *out) {
    if (!(d >= JAI_I64_MIN_D && d < JAI_I64_SUP_D)) return false;
    int64_t i = (int64_t)d;
    if ((double)i != d) return false;
    *out = i;
    return true;
}

JAI_INLINE bool intEqualsDouble(int64_t i, double d) {
    int64_t di;
    return doubleIsExactInt(d, &di) && di == i;
}

static bool compareIntDouble(int64_t i, double d, int *out) {
    if (isnan(d)) return false;
    if (d >= JAI_I64_SUP_D) { *out = -1; return true; }   /* covers +inf */
    if (d < JAI_I64_MIN_D)  { *out = 1;  return true; }   /* covers -inf */

    int64_t di = (int64_t)d;
    if (i < di) { *out = -1; return true; }
    if (i > di) { *out = 1;  return true; }
    double integral = (double)di;
    *out = d > integral ? -1 : d < integral ? 1 : 0;
    return true;
}

static bool compareDoubles(double x, double y, int *out) {
    if (x < y) { *out = -1; return true; }
    if (x > y) { *out = 1; return true; }
    if (x == y) { *out = 0; return true; }
    return false;
}

// type names

static const char *classNameOf(const ObjClass *k) {
    if (k == NULL || k->name == NULL) return "object";
    return k->name->chars;
}

const char *jaiTypeNameStatic(Value v) {
    static const char *const scalarNames[VAL_OBJ] = {
        [VAL_NULL] = "null", [VAL_BOOL] = "bool",
        [VAL_INT] = "int",   [VAL_FLOAT] = "float",
    };
    static const char *const objectNames[OBJ_TYPE_COUNT] = {
        [OBJ_STRING] = "str",       [OBJ_STRBUF] = "str",
        [OBJ_BYTES] = "bytes",      [OBJ_LIST] = "list",
        [OBJ_DICT] = "dict",        [OBJ_SET] = "set",
        [OBJ_TUPLE] = "tuple",      [OBJ_RANGE] = "range",
        [OBJ_FUNCTION] = "fn",      [OBJ_CLOSURE] = "fn",
        [OBJ_NATIVE] = "fn",        [OBJ_BOUND] = "fn",
        [OBJ_CLASS] = "class",      [OBJ_TRAIT] = "trait",
        [OBJ_MODULE] = "module",    [OBJ_ENUM] = "enum",
        [OBJ_ITER] = "iterator",    [OBJ_FILE] = "file",
        [OBJ_UPVALUE] = "upvalue",  [OBJ_ENUM_CTOR] = "fn",
    };

    ValueType type = jaiValueType(v);
    if (JAI_LIKELY(type == VAL_INT)) return "int";
    if (type != VAL_OBJ) return scalarNames[type];

    ObjType objectType = OBJ_TYPE(v);
    if (objectType == OBJ_INSTANCE) return classNameOf(AS_INSTANCE(v)->klass);
    if (objectType == OBJ_ENUM_VAL) {
        ObjEnumVal *e = AS_ENUM_VAL(v);
        return e->type != NULL && e->type->name != NULL ? e->type->name->chars: "enum";
    }

    const char *name = objectType < OBJ_TYPE_COUNT ? objectNames[objectType] : NULL;
    return name != NULL ? name : "object";
}

ObjString *jaiTypeName(Value v) {
    return jaiStringInternC(jaiTypeNameStatic(v));
}

// equality
static int eqDepth = 0;

JAI_INLINE bool instanceHasEq(Value v) {
    if (!IS_OBJ(v)) return false;
    Obj *o = AS_OBJ(v);
    if (o->type != OBJ_INSTANCE) return false;
    ObjClass *k = ((ObjInstance *)o)->klass;
    return k != NULL && !IS_NULL(k->dunderEq);
}

static bool dispatchEq(Value a, Value b, bool *result, bool *missing) {
    Value out;
    Value arg = b;
    *missing = false;
    if (!jaiInvokeMethod(a, vm.strEq, 1, &arg, &out)) {
        if (!vm.hasException) *missing = true;
        return false;
    }
    if (!IS_BOOL(out)) {
        jaiThrow(vm.cTypeError, "__eq__ must return bool, not %s",
                 jaiTypeNameStatic(out));
        return false;
    }
    *result = AS_BOOL(out);
    return true;
}

static bool listsEqual(ObjList *a, ObjList *b) {
    if (a->count != b->count) return false;
    for (int i = 0; i < a->count; i++) {
        if (!jaiValuesEqual(a->items[i], b->items[i])) return false;
    }
    return true;
}

static bool tuplesEqual(ObjTuple *a, ObjTuple *b) {
    if (a->count != b->count) return false;
    for (uint32_t i = 0; i < a->count; i++) {
        if (!jaiValuesEqual(a->items[i], b->items[i])) return false;
    }
    return true;
}

static bool dictsEqual(ObjDict *a, ObjDict *b) {
    if (a->table.count != b->table.count) return false;
    int i = 0;
    Value k, v;
    while (jaiTableNext(&a->table, &i, &k, &v)) {
        Value other;
        if (!jaiTableGet(&b->table, k, &other)) return false;
        if (!jaiValuesEqual(v, other)) return false;
        if (vm.hasException) return false;
    }
    return true;
}

static bool setsEqual(ObjSet *a, ObjSet *b) {
    if (a->table.count != b->table.count) return false;
    int i = 0;
    Value k, v;
    while (jaiTableNext(&a->table, &i, &k, &v)) {
        if (!jaiSetHas(b, k)) return false;
        if (vm.hasException) return false;
    }
    return true;
}

static bool enumValsEqual(ObjEnumVal *a, ObjEnumVal *b) {
    if (a->type != b->type || a->tag != b->tag || a->count != b->count) return false;
    for (int i = 0; i < (int)a->count; i++) {
        if (!jaiValuesEqual(a->payload[i], b->payload[i])) return false;
    }
    return true;
}

static bool objectsEqual(Obj *ao, Obj *bo, Value a, Value b) {
    switch (ao->type) {
        case OBJ_LIST:     return listsEqual((ObjList *)ao, (ObjList *)bo);
        case OBJ_TUPLE:    return tuplesEqual((ObjTuple *)ao, (ObjTuple *)bo);
        case OBJ_DICT:     return dictsEqual((ObjDict *)ao, (ObjDict *)bo);
        case OBJ_SET:      return setsEqual((ObjSet *)ao, (ObjSet *)bo);
        case OBJ_ENUM_VAL: return enumValsEqual((ObjEnumVal *)ao, (ObjEnumVal *)bo);
        case OBJ_INSTANCE: {
            bool result = false, missing = false;
            if (instanceHasEq(a)) {
                if (dispatchEq(a, b, &result, &missing)) return result;
                if (!missing) return false;
            }
            return ao == bo;
        }
        default:
            return ao == bo;
    }
}

static bool valuesEqualInner(Value a, Value b) {
    ValueType ta = jaiValueType(a);
    ValueType tb = jaiValueType(b);
    if (ta == VAL_OBJ && tb ==VAL_OBJ) return objectsEqual(AS_OBJ(a), AS_OBJ(b), a, b);
    if (ta != tb) {
        bool result = false, missing = false;
        if (instanceHasEq(a)) {
            if (dispatchEq(a, b, &result, &missing)) return result;
            if (!missing) return false;
        }
        if (instanceHasEq(b)) {
            if (dispatchEq(b, a, &result, &missing)) return result;
            if (!missing) return false;
        }
        return false;
    }
    JAI_UNREACHABLE();
    return false;
}

bool jaiValuesEqual(Value a, Value b) {
    if (JAI_UNLIKELY(eqDepth >= JAI_EQ_MAX_DEPTH)) {
        if (vm.cRecursionError != NULL) {
            jaiThrow(vm.cRecursionError,
                     "maximum recursion depth exceeded while comparing values");
        }
        return false;
    }

    ValueType ta = jaiValueType(a);
    ValueType tb = jaiValueType(b);
    if (JAI_LIKELY(ta == VAL_INT && tb == VAL_INT))
        return AS_INT(a) == AS_INT(b);
    if (JAI_LIKELY(ta == tb)) {
        switch (ta) {
            case VAL_NULL:  return true;
            case VAL_BOOL:  return AS_BOOL(a) == AS_BOOL(b);
            case VAL_INT:   JAI_UNREACHABLE(); return false;
            case VAL_FLOAT: return AS_FLOAT(a) == AS_FLOAT(b);
            case VAL_OBJ: {
                Obj *ao = AS_OBJ(a), *bo = AS_OBJ(b);
                if (ao == bo && ao->type != OBJ_INSTANCE) return true;
                if (ao->type != bo->type) return false;

                switch (ao->type) {
                    case OBJ_STRING:
                        return jaiStringEquals((ObjString *)ao, (ObjString *)bo);
                    case OBJ_BYTES: {
                        ObjBytes *ba = (ObjBytes *)ao, *bb = (ObjBytes *)bo;
                        return ba->length == bb->length &&
                               memcmp(ba->data, bb->data, ba->length) == 0;
                    }
                    case OBJ_RANGE: {
                        ObjRange *ra = (ObjRange *)ao, *rb = (ObjRange *)bo;
                        return ra->start == rb->start && ra->stop == rb->stop &&
                               ra->step == rb->step &&
                               ra->inclusive == rb->inclusive;
                    }
                    case OBJ_LIST:
                    case OBJ_DICT:
                    case OBJ_SET:
                    case OBJ_TUPLE:
                    case OBJ_INSTANCE:
                    case OBJ_ENUM_VAL:
                        break;
                    default:
                        return false;
                }
                break;
            }
        }
    } else {
        if (ta == VAL_INT && tb == VAL_FLOAT) return intEqualsDouble(AS_INT(a), AS_FLOAT(b));
        if (ta == VAL_FLOAT && tb == VAL_INT) return intEqualsDouble(AS_INT(b), AS_FLOAT(a));
        if (ta == VAL_NULL || tb == VAL_NULL) return false;
        if (!instanceHasEq(a) && !instanceHasEq(b)) return false;
    }

    eqDepth++;
    bool result = valuesEqualInner(a, b);
    eqDepth--;
    return result;
}

bool jaiValuesIdentical(Value a, Value b) {
    ValueType type = jaiValueType(a);
    if (type != jaiValueType(b)) return false;
    switch (type) {
        case VAL_NULL:  return true;
        case VAL_BOOL:  return AS_BOOL(a) == AS_BOOL(b);
        case VAL_INT:   return AS_INT(a) == AS_INT(b);
        case VAL_FLOAT: {
            uint64_t x, y;
            double da = AS_FLOAT(a), db = AS_FLOAT(b);
            memcpy(&x, &da, sizeof x);
            memcpy(&y, &db, sizeof y);
            return x == y;
        }
        case VAL_OBJ:   return AS_OBJ(a) == AS_OBJ(b);
    }
    JAI_UNREACHABLE();
    return false;
}

// hashing

static int hashDepth = 0;

JAI_INLINE uint64_t hashDouble(double d) {
    int64_t i;
    if (doubleIsExactInt(d, &i)) return jaiHashU64((uint64_t)i);
    uint64_t bits;
    memcpy(&bits, &d, sizeof bits);
    return jaiHashU64(bits);
}

JAI_INLINE uint64_t hashPointer(const void *p) {
    return jaiHashU64((uint64_t)(uintptr_t)p);
}

JAI_INLINE uint64_t hashCombine(uint64_t acc, uint64_t h) {
    return (acc ^ h) * 0x100000001b3ULL;
}

static uint64_t hashTuple(ObjTuple *t, bool *ok) {
    if (t->hash != 0) return t->hash;
    uint64_t h = 0xcbf29ce484222325ULL;
    for (uint32_t i = 0; i < t->count; i++) {
        bool elemOk = true;
        uint64_t eh = jaiValueHash(t->items[i], &elemOk);
        if (!elemOk) { *ok = false; return 0; }
        h = hashCombine(h, eh);
    }
    h = jaiHashU64(h ^ (uint64_t)t->count);
    if (h == 0) h = 1; // 0 is the not compute marker
    t->hash = h;
    return h;
}

static uint64_t hashEnumVal(ObjEnumVal *e, bool *ok) {
    uint64_t h = hashCombine(hashPointer(e->type), jaiHashU64(e->tag));
    for (int i = 0; i < (int)e->count; i++) {
        bool elemOk = true;
        uint64_t eh = jaiValueHash(e->payload[i], &elemOk);
        if (!elemOk) { *ok = false; return 0; }
        h = hashCombine(h, eh);
    }
    return jaiHashU64(h);
}

static uint64_t hashInstance(ObjInstance *inst, bool *ok) {
    ObjClass *k = inst->klass;
    if (k == NULL || IS_NULL(k->dunderHash)) { *ok = false; return 0; }

    Value out;
    if (!jaiInvokeMethod(OBJ_VAL(inst), vm.strHash, 0, NULL, &out)) {
        *ok = false; //exception pending
        return 0;
    }
    if (!IS_INT(out)) {
        jaiThrow(vm.cTypeError, "__hash__ must return int, not %s",
                 jaiTypeNameStatic(out));
        *ok = false;
        return 0;
    }
    return jaiHashU64((uint64_t)AS_INT(out));
}

static uint64_t valueHashInner(Value v, bool *ok) {
    switch (OBJ_TYPE(v)) {
        case OBJ_TUPLE:    return hashTuple(AS_TUPLE(v), ok);
        case OBJ_ENUM_VAL: return hashEnumVal(AS_ENUM_VAL(v), ok);
        case OBJ_INSTANCE: return hashInstance(AS_INSTANCE(v), ok);
        default:
            JAI_UNREACHABLE();
            *ok = false;
            return 0;
    }
}

uint64_t jaiValueHash(Value v, bool *ok) {
    bool localOk = true;
    if (ok == NULL) ok = &localOk;
    *ok = true;
    if (JAI_UNLIKELY(hashDepth >= JAI_HASH_MAX_DEPTH)) {
        *ok = false;
        return 0;
    }

    ValueType type = jaiValueType(v);
    if (JAI_LIKELY(type == VAL_INT))
        return jaiHashU64((uint64_t)AS_INT(v));

    switch (type) {
        case VAL_NULL:
            *ok = false;
            return 0;
        case VAL_BOOL:
            return jaiHashU64(AS_BOOL(v) ? 0x5851F42D4C957F2DULL
                                         : 0x14057B7EF767814FULL);
        case VAL_INT:
            JAI_UNREACHABLE();
            return 0;
        case VAL_FLOAT:
            return hashDouble(AS_FLOAT(v));
        case VAL_OBJ: break;
    }

    switch (OBJ_TYPE(v)) {
        case OBJ_STRING:
            return jaiStringHash(AS_STRING(v));
        case OBJ_BYTES:
            return jaiHashBytes(AS_BYTES(v)->data, AS_BYTES(v)->length);
        case OBJ_TUPLE:
            if (AS_TUPLE(v)->hash != 0) return AS_TUPLE(v)->hash;
            break;
        case OBJ_ENUM_VAL: {
            ObjEnumVal *e = AS_ENUM_VAL(v);
            if (e->count != 0) break;
            return jaiHashU64(hashCombine(hashPointer(e->type),
                                          jaiHashU64(e->tag)));
        }
        case OBJ_INSTANCE:
            break;
        case OBJ_RANGE: {
            ObjRange *r = AS_RANGE(v);
            uint64_t h = jaiHashU64((uint64_t)r->start);
            h = hashCombine(h, jaiHashU64((uint64_t)r->stop));
            h = hashCombine(h, jaiHashU64((uint64_t)r->step));
            return jaiHashU64(h ^ (r->inclusive ? 1u : 0u));
        }
        case OBJ_LIST:
        case OBJ_DICT:
        case OBJ_SET:
            *ok = false;
            return 0;
        default: return hashPointer(AS_OBJ(v));
    }

    hashDepth++;
    uint64_t h = valueHashInner(v, ok);
    hashDepth--;
    return *ok ? h : 0;
}

// ordering
JAI_INLINE bool instanceHasLt(Value v) {
    if (!IS_OBJ(v)) return false;
    Obj *o = AS_OBJ(v);
    if (o->type != OBJ_INSTANCE) return false;
    ObjClass *k = ((ObjInstance *)o)->klass;
    return k != NULL && !IS_NULL(k->dunderLt);
}

static bool dispatchLt(Value a, Value b, bool *result, bool *missing) {
    Value out;
    Value arg = b;
    *missing = false;
    if (!jaiInvokeMethod(a, vm.strLt, 1, &arg, &out)) {
        if (!vm.hasException) *missing = true;
        return false;
    }
    if (!IS_BOOL(out)) {
        jaiThrow(vm.cTypeError, "__lt__ must return bool, not %s",
                 jaiTypeNameStatic(out));
        return false;
    }
    *result = AS_BOOL(out);
    return true;
}

static bool compareWithLt(Value a, Value b, int *out) {
    bool less = false, missing = false;
    bool aHasLt = instanceHasLt(a);

    if (aHasLt) {
        if (!dispatchLt(a, b, &less, &missing)) return false;
        if (less) { *out = -1; return true; }
    }

    bool bHasLt = instanceHasLt(b);
    if (bHasLt) {
        if (!dispatchLt(b, a, &less, &missing)) return false;
        if (less) { *out = 1; return true; }
        *out = 0;
        return true;
    }
    if (!aHasLt) return false;

    bool eq = jaiValuesEqual(a, b);
    if (vm.hasException) return false;
    *out = eq ? 0 : 1;
    return true;
}

static bool compareStrings(ObjString *a, ObjString *b, int *out) {
    if (a == b) { *out = 0; return true; }
    uint32_t n = a->length < b->length ? a->length : b->length;
    int c = n == 0 ? 0 : memcmp(a->chars, b->chars, n);
    if (c != 0) { *out = c < 0 ? -1 : 1; return true; }
    *out = a->length == b->length ? 0 : (a->length < b->length ? -1 : 1);
    return true;
}

bool jaiValueCompare(Value a, Value b, int *out) {
    ValueType ta = jaiValueType(a);
    ValueType tb = jaiValueType(b);

    if (JAI_LIKELY(ta == VAL_INT && tb == VAL_INT)) {
        int64_t x = AS_INT(a), y = AS_INT(b);
        *out = (x < y) ? -1 : (x > y) ? 1 : 0;
        return true;
    }
    if ((ta == VAL_INT || ta == VAL_FLOAT) &&
        (tb == VAL_INT || tb == VAL_FLOAT)) {
        if (ta == VAL_INT) return compareIntDouble(AS_INT(a), AS_FLOAT(b), out);
        if (tb == VAL_INT) {
            int inverted;
            if (!compareIntDouble(AS_INT(b), AS_FLOAT(a), &inverted)) return false;
            *out = -inverted;
            return true;
        }
        return compareDoubles(AS_FLOAT(a), AS_FLOAT(b), out);
    }

    if (ta == VAL_BOOL && tb == VAL_BOOL) {
        *out = (int)AS_BOOL(a) - (int)AS_BOOL(b);
        return true;
    }

    if (ta == VAL_OBJ && tb == VAL_OBJ &&
        AS_OBJ(a)->type == OBJ_STRING && AS_OBJ(b)->type == OBJ_STRING) {
        return compareStrings(AS_STRING(a), AS_STRING(b), out);
    }

    return compareWithLt(a, b, out);
}

typedef struct {
    JaiBuf *buf;
    FILE   *file;
} ValSink;

JAI_INLINE void sinkWrite(ValSink *s, const char *data, size_t n) {
    if (n == 0) return;
    if (s->buf != NULL) {
        JaiBuf *b = s->buf;
        if (JAI_UNLIKELY(b->capacity - b->count < n)) jaiBufReserve(b, n);
        memcpy(b->data + b->count, data, n);
        b->count += n;
    } else {
        size_t wrote = fwrite(data, 1, n, s->file);
        (void)wrote;
    }
}

JAI_INLINE void sinkStr(ValSink *s, const char *str) {
    sinkWrite(s, str, strlen(str));
}

JAI_INLINE void sinkReserve(ValSink *s, size_t n) {
    if (s->buf != NULL && s->buf->capacity - s->buf->count < n)
        jaiBufReserve(s->buf, n);
}

#define JAI_INT_DIGITS 24
#define JAI_FLOAT_CHARS 32

//magic sh*t
static const char digitPairs[] =
    "00010203040506070809"
    "10111213141516171819"
    "20212223242526272829"
    "30313233343536373839"
    "40414243444546474849"
    "50515253545556575859"
    "60616263646566676869"
    "70717273747576777879"
    "80818283848586878889"
    "90919293949596979899";

JAI_INLINE int decimalLength(uint64_t u) {
    if (u < 10000000000ULL) {
        if (u < 100000ULL) {
            if (u < 100ULL) return u < 10ULL ? 1 : 2;
            if (u < 1000ULL) return 3;
            if (u < 10000ULL) return 4;
            return 5;
        }
        if (u < 100000000ULL) {
            if (u < 1000000ULL) return 6;
            if (u < 10000000ULL) return 7;
            return 8;
        }
        return u < 1000000000ULL ? 9 : 10;
    }
    if (u < 1000000000000000ULL) {
        if (u < 1000000000000ULL)
            return u < 100000000000ULL ? 11 : 12;
        if (u < 10000000000000ULL) return 13;
        if (u < 100000000000000ULL) return 14;
        return 15;
    }
    if (u < 100000000000000000ULL)
        return u < 10000000000000000ULL ? 16 : 17;
    if (u < 10000000000000000000ULL)
        return u < 1000000000000000000ULL ? 18 : 19;
    return 20;
}

static int writeInt64(char *out, int64_t value) {
    uint64_t u = value < 0 ? (uint64_t)(-(value + 1)) + 1u : (uint64_t)value;
    int negative = value < 0;
    int len = negative + decimalLength(u);
    char *p = out + len;

    while (u >= 100) {
        uint64_t q = u / 100;
        unsigned r = (unsigned)(u - q * 100);
        p -= 2;
        p[0] = digitPairs[r * 2];
        p[1] = digitPairs[r * 2 + 1];
        u = q;
    }
    if (u < 10) {
        *--p = (char)('0' + (unsigned)u);
    } else {
        unsigned at = (unsigned)u * 2;
        p -= 2;
        p[0] = digitPairs[at];
        p[1] = digitPairs[at + 1];
    }
    if (negative) out[0] = '-';
    return len;
}

JAI_INLINE void sinkInt(ValSink *s, int64_t value) {
    char digits[JAI_INT_DIGITS];
    sinkWrite(s, digits, (size_t)writeInt64(digits, value));
}

static size_t formatDouble(char *out, size_t outSize, double d) {
    if (isnan(d)) {
        memcpy(out, "nan", 4);
        return 3;
    }
    if (isinf(d)) {
        if (d < 0) { memcpy(out, "-inf", 5); return 4; }
        memcpy(out, "inf", 4);
        return 3;
    }

    int64_t whole;
    if (d > -10000000000000000.0 && d < 10000000000000000.0 &&
        doubleIsExactInt(d, &whole)) {
        if (whole == 0 && signbit(d)) {
            memcpy(out, "-0.0", 5);
            return 4;
        }
        size_t len = (size_t)writeInt64(out, whole);
        memcpy(out + len, ".0", 3);
        return len + 2;
    }

    char sci[JAI_FLOAT_CHARS];
    int digits = 0, last = 0;
    for (int p = 1; p <= 3; p++) {
        snprintf(sci, sizeof sci, "%.*e", p - 1, d);
        last = p;
        if (strtod(sci, NULL) == d) { digits = p; break; }
    }
    if (digits == 0) {
        int low = 4, high = 17;
        while (low < high) {
            int p = low + (high - low) / 2;
            snprintf(sci, sizeof sci, "%.*e", p - 1, d);
            last = p;
            if (strtod(sci, NULL) == d) high = p;
            else low = p + 1;
        }
        digits = low;
        if (last != digits)
            snprintf(sci, sizeof sci, "%.*e", digits - 1, d);
    }

    const char *e = strchr(sci, 'e');
    int exp = 0;
    if (e != NULL) {
        const char *p = e + 1;
        bool negative = *p == '-';
        if (*p == '-' || *p == '+') p++;
        while (*p >= '0' && *p <= '9') exp = exp * 10 + (*p++ - '0');
        if (negative) exp = -exp;
    }

    if (exp < -4 || exp >= 16) {
        size_t len = strlen(sci);
        memcpy(out, sci, len + 1);
        return len;
    }
    int decimals = digits - 1 - exp;
    if (decimals < 0) decimals = 0;
    int wrote = snprintf(out, outSize, "%.*f", decimals, d);
    if (wrote < 0) { out[0] = '\0'; return 0; }
    size_t len = (size_t)wrote;
    if (strchr(out, '.') == NULL) {
        if (len + 3 <= outSize) {
            memcpy(out + len, ".0", 3);
            len += 2;
        }
    }
    return len;
}

static const char hexUpper[] = "0123456789ABCDEF";

JAI_INLINE void sinkByteEscape(ValSink *s, unsigned byte) {
    char escaped[4] = {'\\', 'x', hexUpper[(byte >> 4) & 15],
                      hexUpper[byte & 15]};
    sinkWrite(s, escaped, sizeof escaped);
}

JAI_INLINE void sinkControlEscape(ValSink *s, unsigned cp) {
    char escaped[6] = {'\\', 'u', '{', 0, 0, '}'};
    if (cp < 16) {
        escaped[3] = hexUpper[cp];
        escaped[4] = '}';
        sinkWrite(s, escaped, 5);
    } else {
        escaped[3] = hexUpper[(cp >> 4) & 15];
        escaped[4] = hexUpper[cp & 15];
        sinkWrite(s, escaped, sizeof escaped);
    }
}

static void emitQuotedString(ValSink *s, const char *chars, size_t length) {
    const char *p = chars;
    const char *run = chars;
    const char *end = chars + length;

    sinkWrite(s, "\"", 1);
    while (p < end) {
        unsigned char c = (unsigned char)*p;
        const char *escaped = NULL;
        switch (c) {
            case '\\': escaped = "\\\\"; break;
            case '"':  escaped = "\\\""; break;
            case '\n': escaped = "\\n";  break;
            case '\t': escaped = "\\t";  break;
            case '\r': escaped = "\\r";  break;
            default: break;
        }
        if (escaped != NULL) {
            sinkWrite(s, run, (size_t)(p - run));
            sinkWrite(s, escaped, 2);
            run = ++p;
            continue;
        }
        if (c < 0x80) {
            if (c < 0x20 || c == 0x7f) {
                sinkWrite(s, run, (size_t)(p - run));
                sinkControlEscape(s, (unsigned)c);
                run = ++p;
            } else {
                p++;
            }
            continue;
        }

        int len = 1;
        int32_t cp = jaiUtf8Decode(p, end, &len);
        if (len <= 0) len = 1;
        if (cp < 0 || (cp >= 0x80 && cp <= 0x9f)) {
            sinkWrite(s, run, (size_t)(p - run));
            if (cp < 0) sinkByteEscape(s, (unsigned)c); //invalid utf-8
            else sinkControlEscape(s, (unsigned)cp);
            p += len;
            run = p;
        } else {
            p += len;
        }
    }
    sinkWrite(s, run, (size_t)(end - run));
    sinkWrite(s, "\"", 1);
}

static void emitBytesLiteral(ValSink *s, const uint8_t *data, size_t length) {
    const uint8_t *p = data;
    const uint8_t *run = data;
    const uint8_t *end = data + length;

    sinkWrite(s, "b\"", 2);
    while (p < end) {
        uint8_t c = *p;
        const char *escaped = NULL;
        switch (c) {
            case '\\': escaped = "\\\\"; break;
            case '"':  escaped = "\\\""; break;
            case '\n': escaped = "\\n";  break;
            case '\t': escaped = "\\t";  break;
            case '\r': escaped = "\\r";  break;
            default: break;
        }
        if (escaped != NULL) {
            sinkWrite(s, (const char *)run, (size_t)(p - run));
            sinkWrite(s, escaped, 2);
            run = ++p;
        } else if (c >= 0x20 && c < 0x7f) {
            p++;
        } else {
            sinkWrite(s, (const char *)run, (size_t)(p - run));
            sinkByteEscape(s, (unsigned)c);
            run = ++p;
        }
    }
    sinkWrite(s, (const char *)run, (size_t)(end - run));
    sinkWrite(s, "\"", 1);
}

static const Obj *renderStack[JAI_RENDER_MAX_DEPTH];
static int renderDepth = 0;

JAI_INLINE bool renderInProgress(const Obj *o) {
    for (int i = renderDepth - 1; i >= 0; i--) {
        if (renderStack[i] == o) return true;
    }
    return false;
}

static bool renderValue(ValSink *s, Value v, bool repr, bool allowUser);

JAI_INLINE bool renderEnter(ValSink *s, const Obj *o, const char *elision) {
    if (renderInProgress(o) || renderDepth >= JAI_RENDER_MAX_DEPTH) {
        sinkStr(s, elision);
        return false;
    }
    renderStack[renderDepth++] = o;
    return true;
}

JAI_INLINE void renderLeave(void) {
    if (renderDepth > 0) renderDepth--;
}

static void renderFnLike(ValSink *s, Value v) {
    switch (OBJ_TYPE(v)) {
        case OBJ_FUNCTION: {
            ObjFunction *fn = AS_FUNCTION(v);
            sinkStr(s, "<fn ");
            sinkStr(s, fn->name != NULL ? fn->name->chars : "anonymous");
            sinkWrite(s, ">", 1);
            return;
        }
        case OBJ_CLOSURE: {
            ObjFunction *fn = AS_CLOSURE(v)->fn;
            const char *name = (fn != NULL && fn->name != NULL) ? fn->name->chars
                                                                : "anonymous";
            sinkStr(s, "<fn ");
            sinkStr(s, name);
            sinkWrite(s, ">", 1);
            return;
        }
        case OBJ_NATIVE: {
            ObjNative *n = AS_NATIVE(v);
            sinkStr(s, "<native fn ");
            sinkStr(s, n->name != NULL ? n->name->chars : "anonymous");
            sinkWrite(s, ">", 1);
            return;
        }
        case OBJ_BOUND: {
            ObjBound *b = AS_BOUND(v);
            const char *method = "anonymous";
            if (IS_CLOSURE(b->method)) {
                ObjFunction *fn = AS_CLOSURE(b->method)->fn;
                if (fn != NULL && fn->name != NULL) method = fn->name->chars;
            } else if (IS_NATIVE(b->method) && AS_NATIVE(b->method)->name != NULL) {
                method = AS_NATIVE(b->method)->name->chars;
            }
            sinkStr(s, "<bound method ");
            sinkStr(s, jaiTypeNameStatic(b->receiver));
            sinkWrite(s, ".", 1);
            sinkStr(s, method);
            sinkWrite(s, ">", 1);
            return;
        }
        default:
            sinkStr(s, "<fn>");
            return;
    }
}

static bool renderInstance(ValSink *s, ObjInstance *inst, bool repr, bool allowUser) {
    ObjClass *k = inst->klass;
    ObjString *dunder = NULL;

    if (allowUser && k != NULL) {
        if (repr) {
            if (!IS_NULL(k->dunderRepr)) dunder = vm.strRepr;
        } else if (!IS_NULL(k->dunderStr)) {
            dunder = vm.strStr;
        } else if (!IS_NULL(k->dunderRepr)) {
            dunder = vm.strRepr;
        }
    }
    if (dunder == NULL) {
        sinkWrite(s, "<", 1);
        sinkStr(s, classNameOf(k));
        sinkStr(s, " instance>");
        return true;
    }

    Value out;
    if (!jaiInvokeMethod(OBJ_VAL(inst), dunder, 0, NULL, &out)) {
        if (vm.hasException) return false;
        sinkWrite(s, "<", 1);
        sinkStr(s, classNameOf(k));
        sinkStr(s, " instance>");
        return true;
    }
    if (!IS_STRING(out)) {
        jaiThrow(vm.cTypeError, "%s must return str, not %s", dunder->chars,
                 jaiTypeNameStatic(out));
        return false;
    }

    ObjString *str = AS_STRING(out);
    bool rooted = tempRoot(out);
    sinkWrite(s, str->chars, str->length);
    tempUnroot(rooted);
    return true;
}

static bool renderList(ValSink *s, ObjList *l, bool allowUser) {
    if (!renderEnter(s, (const Obj *)l, "[...]")) return true;
    if (l->count > 0 && (size_t)l->count <= SIZE_MAX / 3)
        sinkReserve(s, (size_t)l->count * 3);
    sinkWrite(s, "[", 1);
    bool ok = true;

    for (int i = 0; i < l->count; i++) {
        if (i > 0) sinkWrite(s, ", ", 2);
        Value item = l->items[i];
        bool rooted = tempRoot(item);
        ok = renderValue(s, item, true, allowUser);
        tempUnroot(rooted);
        if (!ok) break;
    }
    if (ok) sinkWrite(s, "]", 1);
    renderLeave();
    return ok;
}

static bool renderTuple(ValSink *s, ObjTuple *t, bool allowUser) {
    if (!renderEnter(s, (const Obj *)t, "(...)")) return true;
    if (t->count > 0 && (size_t)t->count <= SIZE_MAX / 3)
        sinkReserve(s, (size_t)t->count * 3);
    sinkWrite(s, "(", 1);
    bool ok = true;
    for (uint32_t i = 0; i < t->count; i++) {
        if (i > 0) sinkWrite(s, ", ", 2);
        ok = renderValue(s, t->items[i], true, allowUser);
        if (!ok) break;
    }

    if (ok && t->count == 1) sinkWrite(s, ",", 1);
    if (ok) sinkWrite(s, ")", 1);
    renderLeave();
    return ok;
}

static bool renderDict(ValSink *s, ObjDict *d, bool allowUser) {
    if (!renderEnter(s, (const Obj *)d, "{...}")) return true;
    if (d->table.count > 0 && (size_t)d->table.count <= SIZE_MAX / 6)
        sinkReserve(s, (size_t)d->table.count * 6);
    sinkWrite(s, "{", 1);
    bool ok = true;
    int i = 0, n = 0;
    Value k, val;
    while (jaiTableNext(&d->table, &i, &k, &val)) {
        if (n++ > 0) sinkWrite(s, ", ", 2);
        bool rk = tempRoot(k), rv = tempRoot(val);
        ok = renderValue(s, k, true, allowUser);
        if (ok) {
            sinkWrite(s, ": ", 2);
            ok = renderValue(s, val, true, allowUser);
        }
        tempUnroot(rv);
        tempUnroot(rk);
        if (!ok) break;
    }
    if (ok) sinkWrite(s, "}", 1);
    renderLeave();
    return ok;
}

static bool renderSet(ValSink *s, ObjSet *set, bool allowUser) {
    if (set->table.count == 0) { sinkStr(s, "set()"); return true; }
    if (!renderEnter(s, (const Obj *)set, "{...}")) return true;
    if ((size_t)set->table.count <= SIZE_MAX / 3)
        sinkReserve(s, (size_t)set->table.count * 3);
    sinkWrite(s, "{", 1);
    bool ok = true;
    int i = 0, n = 0;
    Value k, val;
    while (jaiTableNext(&set->table, &i, &k, &val)) {
        if (n++ > 0) sinkWrite(s, ", ", 2);
        bool rooted = tempRoot(k);
        ok = renderValue(s, k, true, allowUser);
        tempUnroot(rooted);
        if (!ok) break;
    }
    if (ok) sinkWrite(s, "}", 1);
    renderLeave();
    return ok;
}

static bool renderEnumVal(ValSink *s, ObjEnumVal *e, bool allowUser) {
    const char *typeName = (e->type != NULL && e->type->name != NULL)
                               ? e->type->name->chars : "enum";
    const char *variant = "?";
    if (e->type != NULL && e->tag < e->type->variantCount &&
        e->type->variants != NULL && e->type->variants[e->tag].name != NULL) {
        variant = e->type->variants[e->tag].name->chars;
    }
    sinkStr(s, typeName);
    sinkWrite(s, ".", 1);
    sinkStr(s, variant);
    if (e->count == 0) return true;

    if (!renderEnter(s, (const Obj *)e, "(...)")) return true;
    if ((size_t)e->count <= SIZE_MAX / 3)
        sinkReserve(s, (size_t)e->count * 3);
    sinkWrite(s, "(", 1);
    bool ok = true;
    for (int i = 0; i < (int)e->count; i++) {
        if (i > 0) sinkWrite(s, ", ", 2);
        ok = renderValue(s, e->payload[i], true, allowUser);
        if (!ok) break;
    }
    if (ok) sinkWrite(s, ")", 1);
    renderLeave();
    return ok;
}

static void renderRange(ValSink *s, ObjRange *r) {
    sinkInt(s, r->start);
    sinkWrite(s, r->inclusive ? "..=" : "..", r->inclusive ? 3 : 2);
    sinkInt(s, r->stop);
    if (r->step != 1) {
        sinkWrite(s, ":", 1);
        sinkInt(s, r->step);
    }
}

static bool renderValue(ValSink *s, Value v, bool repr, bool allowUser) {
    switch (jaiValueType(v)) {
        case VAL_NULL: sinkStr(s, "null"); return true;
        case VAL_BOOL: sinkStr(s, AS_BOOL(v) ? "true" : "false"); return true;
        case VAL_INT:  sinkInt(s, AS_INT(v)); return true;
        case VAL_FLOAT: {
            char buf[JAI_FLOAT_CHARS];
            size_t len = formatDouble(buf, sizeof buf, AS_FLOAT(v));
            sinkWrite(s, buf, len);
            return true;
        }
        case VAL_OBJ: break;
    }

    switch (OBJ_TYPE(v)) {
        case OBJ_STRBUF: break;
        case OBJ_STRING: {
            ObjString *str = AS_STRING(v);
            if (repr) emitQuotedString(s, str->chars, str->length);
            else sinkWrite(s, str->chars, str->length);
            return true;
        }
        case OBJ_BYTES:
            emitBytesLiteral(s, AS_BYTES(v)->data, AS_BYTES(v)->length);
            return true;
        case OBJ_LIST:     return renderList(s, AS_LIST(v), allowUser);
        case OBJ_TUPLE:    return renderTuple(s, AS_TUPLE(v), allowUser);
        case OBJ_DICT:     return renderDict(s, AS_DICT(v), allowUser);
        case OBJ_SET:      return renderSet(s, AS_SET(v), allowUser);
        case OBJ_RANGE:    renderRange(s, AS_RANGE(v)); return true;
        case OBJ_ENUM_VAL: return renderEnumVal(s, AS_ENUM_VAL(v), allowUser);
        case OBJ_INSTANCE: return renderInstance(s, AS_INSTANCE(v), repr, allowUser);
        case OBJ_FUNCTION:
        case OBJ_CLOSURE:
        case OBJ_NATIVE:
        case OBJ_BOUND:
            renderFnLike(s, v);
            return true;
        case OBJ_CLASS:
            sinkStr(s, "<class ");
            sinkStr(s, classNameOf(AS_CLASS(v)));
            sinkWrite(s, ">", 1);
            return true;
        case OBJ_TRAIT: {
            ObjTrait *t = AS_TRAIT(v);
            sinkStr(s, "<trait ");
            sinkStr(s, t->name != NULL ? t->name->chars : "?");
            sinkWrite(s, ">", 1);
            return true;
        }
        case OBJ_MODULE: {
            ObjModule *m = AS_MODULE(v);
            sinkStr(s, "<module ");
            sinkStr(s, m->name != NULL ? m->name->chars : "?");
            sinkWrite(s, ">", 1);
            return true;
        }
        case OBJ_ENUM: {
            ObjEnum *e = AS_ENUM(v);
            sinkStr(s, "<enum ");
            sinkStr(s, e->name != NULL ? e->name->chars : "?");
            sinkWrite(s, ">", 1);
            return true;
        }
        case OBJ_FILE: {
            ObjFile *f = AS_FILE(v);
            const char *path = f->path != NULL ? f->path->chars : "?";
            sinkWrite(s, "<", 1);
            if (f->closed) sinkStr(s, "closed ");
            sinkStr(s, "file '");
            sinkStr(s, path);
            sinkStr(s, "'>");
            return true;
        }
        case OBJ_ENUM_CTOR: {
            ObjEnumCtor *c = AS_ENUM_CTOR(v);
            const char *en = (c->type != NULL && c->type->name != NULL)
                                 ? c->type->name->chars : "?";
            const char *vn = (c->type != NULL && c->tag < c->type->variantCount &&
                              c->type->variants[c->tag].name != NULL)
                                 ? c->type->variants[c->tag].name->chars : "?";
            sinkStr(s, "<enum constructor ");
            sinkStr(s, en);
            sinkWrite(s, ".", 1);
            sinkStr(s, vn);
            sinkWrite(s, ">", 1);
            return true;
        }
        case OBJ_ITER:    sinkStr(s, "<iterator>"); return true;
        case OBJ_UPVALUE: sinkStr(s, "<upvalue>");  return true;
        case OBJ_TYPE_COUNT: break;
    }
    sinkStr(s, "<object>");
    return true;
}

static ObjString *renderToString(Value v, bool repr) {
    switch (jaiValueType(v)) {
    case VAL_INT: {
        char digits[JAI_INT_DIGITS];
        return jaiStringNew(digits, (size_t)writeInt64(digits, AS_INT(v)));
    }
    case VAL_FLOAT: {
        char digits[JAI_FLOAT_CHARS];
        size_t len = formatDouble(digits, sizeof digits, AS_FLOAT(v));
        return jaiStringNew(digits, len);
    }
    case VAL_BOOL:
        return AS_BOOL(v) ? jaiStringNew("true", 4) : jaiStringNew("false", 5);
    case VAL_NULL:
        return jaiStringNew("null", 4);
    case VAL_OBJ:
        break;
    }

    JaiBuf buf;
    jaiBufInit(&buf);
    ValSink sink = {&buf, NULL};

    bool rooted = tempRoot(v);
    bool ok = renderValue(&sink, v, repr, true);
    tempUnroot(rooted);

    if (!ok) {
        jaiBufFree(&buf);
        return NULL;
    }
    ObjString *out = jaiStringNew(buf.data != NULL ? (const char *)buf.data : "",
                                  buf.count);
    jaiBufFree(&buf);
    return out;
}

//fstring
static ObjString *formatViaBuffer(const Value *parts, int count) {
    JaiBuf buf;
    jaiBufInit(&buf);
    ValSink sink = {&buf, NULL};

    for (int i = 0; i < count; i++) {
        if (!renderValue(&sink, parts[i], false, true)) {
            jaiBufFree(&buf);
            return NULL;
        }
    }
    ObjString *out = jaiStringNew(buf.data != NULL ? (const char *)buf.data : "",
                                  buf.count);
    jaiBufFree(&buf);
    return out;
}

ObjString *jaiValueFormat(const Value *parts, int count) {
    if (count <= 0) return jaiStringIntern("", 0);
    if (count > JAI_FMT_MAX_PARTS) return formatViaBuffer(parts, count);

    char        scratch[JAI_FMT_MAX_PARTS][JAI_FLOAT_CHARS];
    const char *runs[JAI_FMT_MAX_PARTS];
    uint32_t    lens[JAI_FMT_MAX_PARTS];
    size_t      total = 0;

    for (int i = 0; i < count; i++) {
        Value v = parts[i];
        switch (jaiValueType(v)) {
        case VAL_NULL:
            runs[i] = "null";
            lens[i] = 4;
            break;
        case VAL_BOOL:
            runs[i] = AS_BOOL(v) ? "true" : "false";
            lens[i] = AS_BOOL(v) ? 4u : 5u;
            break;
        case VAL_INT:
            lens[i] = (uint32_t)writeInt64(scratch[i], AS_INT(v));
            runs[i] = scratch[i];
            break;
        case VAL_FLOAT:
            lens[i] = (uint32_t)formatDouble(scratch[i], sizeof scratch[i],
                                             AS_FLOAT(v));
            runs[i] = scratch[i];
            break;
        case VAL_OBJ:
            if (!IS_STRING(v)) return formatViaBuffer(parts, count);
            runs[i] = AS_STRING(v)->chars;
            lens[i] = AS_STRING(v)->length;
            break;
        default:
            return formatViaBuffer(parts, count);
        }
        total += lens[i];
    }
    return jaiStringFromParts(runs, lens, count, total);
}

ObjString *jaiValueToStr(Value v) {
    if (IS_STRING(v)) return AS_STRING(v);
    return renderToString(v, false);
}

ObjString *jaiValueToRepr(Value v) {
    return renderToString(v, true);
}

void jaiPrintValue(FILE *out, Value v, bool repr) {
    if (out == NULL) return;
    ValSink sink = {NULL, out};
    (void)renderValue(&sink, v, repr, false);
}
