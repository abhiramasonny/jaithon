/* value.c — equality, hashing, ordering, and rendering of Values.
 *
 * Three invariants drive most of the code here:
 *
 *   1. int and float are distinct types (spec §2.4) but `==` compares them
 *      numerically, so equality and hashing must agree across the two: a float
 *      with an exact integral value hashes as that int.
 *   2. Every comparison of an int against a float is done exactly, never by
 *      widening the int to double — doubles cannot hold every int64.
 *   3. Rendering may re-enter the VM (__str__/__repr__), so it is written
 *      against a sink abstraction with a cycle/depth guard, and the fast path
 *      used by the disassembler and traceback printer runs it with user
 *      dispatch disabled so it can never recurse into the interpreter.
 */
#include <inttypes.h>
#include <math.h>
#include <stdlib.h>

#include "value.h"
#include "object.h"
#include "table.h"
#include "vm.h"

/* Exactly representable bounds of int64_t as doubles. -2^63 is exact; +2^63 is
 * the first double above INT64_MAX. */
#define JAI_I64_MIN_D (-9223372036854775808.0)
#define JAI_I64_SUP_D (9223372036854775808.0)

/* Comparison and hashing recurse through nested containers; cap the depth so a
 * pathological structure cannot smash the C stack. */
#define JAI_EQ_MAX_DEPTH   256
#define JAI_HASH_MAX_DEPTH 256
/* Rendering nests further (one level per container) but also has to stay well
 * inside the C stack while a user __repr__ may re-enter the VM. */
#define JAI_RENDER_MAX_DEPTH 64

/* ------------------------------------------------------------------ */
/* Value array                                                         */
/* ------------------------------------------------------------------ */

void jaiValueArrayInit(ValueArray *a) { JAI_VEC_INIT(a); }

void jaiValueArrayFree(ValueArray *a) { JAI_VEC_FREE(Value, a); }

void jaiValueArrayPush(ValueArray *a, Value v) { JAI_VEC_PUSH(Value, a, v); }

/* ------------------------------------------------------------------ */
/* Temporary GC roots                                                  */
/* ------------------------------------------------------------------ */

/* Root `v` if it is a heap object and the collector is running. Returns whether
 * a matching jaiPopRoot() is owed. Needed because the JaiBuf we render into is
 * plain memory, so an object we hold only in a C local can be collected by an
 * allocation performed while appending to it. */
static bool tempRoot(Value v) {
    if (!IS_OBJ(v) || vm.gc == NULL) return false;
    jaiPushRoot(v);
    return true;
}

static void tempUnroot(bool rooted) {
    if (rooted) jaiPopRoot();
}

/* ------------------------------------------------------------------ */
/* Exact int/float arithmetic predicates                               */
/* ------------------------------------------------------------------ */

/* True when `d` is finite, integral, and within int64 range. */
static bool doubleIsExactInt(double d, int64_t *out) {
    if (!isfinite(d) || d != floor(d)) return false;
    if (d < JAI_I64_MIN_D || d >= JAI_I64_SUP_D) return false;
    *out = (int64_t)d;
    return true;
}

static bool intEqualsDouble(int64_t i, double d) {
    int64_t di;
    return doubleIsExactInt(d, &di) && di == i;
}

/* Three-way compare of an int against a double without precision loss.
 * Returns false when `d` is NaN (unordered). */
static bool compareIntDouble(int64_t i, double d, int *out) {
    if (isnan(d)) return false;
    if (d >= JAI_I64_SUP_D) { *out = -1; return true; }   /* covers +inf */
    if (d < JAI_I64_MIN_D)  { *out = 1;  return true; }   /* covers -inf */

    double fl = floor(d);
    int64_t fi = (int64_t)fl;                             /* in range by the above */
    if (i < fi) { *out = -1; return true; }
    if (i > fi) { *out = 1;  return true; }
    *out = (d > fl) ? -1 : 0;                             /* equal floors: fraction decides */
    return true;
}

static bool compareDoubles(double x, double y, int *out) {
    if (isnan(x) || isnan(y)) return false;
    *out = (x < y) ? -1 : (x > y) ? 1 : 0;
    return true;
}

/* ------------------------------------------------------------------ */
/* Type names                                                          */
/* ------------------------------------------------------------------ */

static const char *classNameOf(const ObjClass *k) {
    if (k == NULL || k->name == NULL) return "object";
    return k->name->chars;
}

const char *jaiTypeNameStatic(Value v) {
    switch (jaiValueType(v)) {
        case VAL_NULL:  return "null";
        case VAL_BOOL:  return "bool";
        case VAL_INT:   return "int";
        case VAL_FLOAT: return "float";
        case VAL_OBJ:   break;
    }

    switch (OBJ_TYPE(v)) {
        case OBJ_STRING:   return "str";
        case OBJ_STRBUF:   return "str";   /* never reaches a Value */
        case OBJ_BYTES:    return "bytes";
        case OBJ_LIST:     return "list";
        case OBJ_DICT:     return "dict";
        case OBJ_SET:      return "set";
        case OBJ_TUPLE:    return "tuple";
        case OBJ_RANGE:    return "range";
        case OBJ_FUNCTION:
        case OBJ_CLOSURE:
        case OBJ_NATIVE:
        case OBJ_BOUND:    return "fn";
        case OBJ_CLASS:    return "class";
        case OBJ_TRAIT:    return "trait";
        case OBJ_MODULE:   return "module";
        case OBJ_ENUM:     return "enum";
        case OBJ_FILE:     return "file";
        case OBJ_ITER:     return "iterator";
        case OBJ_INSTANCE: return classNameOf(AS_INSTANCE(v)->klass);
        case OBJ_ENUM_VAL: {
            ObjEnumVal *e = AS_ENUM_VAL(v);
            if (e->type != NULL && e->type->name != NULL) return e->type->name->chars;
            return "enum";
        }
        case OBJ_UPVALUE:  return "upvalue";
        /* Typed as the function it is: `Shape.Circle` has type
         * `fn(float) -> Shape`, so `type_of` must not call it something else. */
        case OBJ_ENUM_CTOR: return "fn";
        case OBJ_TYPE_COUNT: break;
    }
    return "object";
}

ObjString *jaiTypeName(Value v) {
    return jaiStringInternC(jaiTypeNameStatic(v));
}

/* ------------------------------------------------------------------ */
/* Equality                                                            */
/* ------------------------------------------------------------------ */

static int eqDepth = 0;

/* The class's cached __eq__, or NULL_VAL when the value is not an instance or
 * its class does not define one. */
static bool instanceHasEq(Value v) {
    return IS_INSTANCE(v) && AS_INSTANCE(v)->klass != NULL &&
           !IS_NULL(AS_INSTANCE(v)->klass->dunderEq);
}

/* Dispatch a.__eq__(b). Returns false with the exception pending on error, or
 * with *missing set when the method vanished from under the cache. */
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

/* Dicts are equal as unordered key->value sets. */
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
    /* Identity short-circuits every container: it is both correct and what
     * makes a self-referential list compare equal to itself. Instances are
     * excluded because __eq__ may legitimately say otherwise. */
    if (ao == bo && ao->type != OBJ_INSTANCE) return true;
    if (ao->type != bo->type) return false;

    switch (ao->type) {
        case OBJ_STRING: {
            ObjString *sa = (ObjString *)ao, *sb = (ObjString *)bo;
            /* Two distinct interned strings can never have equal content. */
            if (JAI_STR_INTERNED(sa) && JAI_STR_INTERNED(sb)) return false;
            return jaiStringEquals(sa, sb);
        }
        case OBJ_BYTES: {
            ObjBytes *ba = (ObjBytes *)ao, *bb = (ObjBytes *)bo;
            return ba->length == bb->length &&
                   memcmp(ba->data, bb->data, ba->length) == 0;
        }
        case OBJ_LIST:     return listsEqual((ObjList *)ao, (ObjList *)bo);
        case OBJ_TUPLE:    return tuplesEqual((ObjTuple *)ao, (ObjTuple *)bo);
        case OBJ_DICT:     return dictsEqual((ObjDict *)ao, (ObjDict *)bo);
        case OBJ_SET:      return setsEqual((ObjSet *)ao, (ObjSet *)bo);
        case OBJ_ENUM_VAL: return enumValsEqual((ObjEnumVal *)ao, (ObjEnumVal *)bo);
        case OBJ_RANGE: {
            ObjRange *ra = (ObjRange *)ao, *rb = (ObjRange *)bo;
            return ra->start == rb->start && ra->stop == rb->stop &&
                   ra->step == rb->step && ra->inclusive == rb->inclusive;
        }
        case OBJ_INSTANCE: {
            bool result = false, missing = false;
            if (instanceHasEq(a)) {
                if (dispatchEq(a, b, &result, &missing)) return result;
                if (!missing) return false;          /* exception pending */
            }
            return ao == bo;
        }
        default:
            return ao == bo;                          /* functions, classes, ... */
    }
}

static bool valuesEqualInner(Value a, Value b) {
    if (jaiValueType(a) != jaiValueType(b)) {
        if (IS_NUMBER(a) && IS_NUMBER(b)) {
            /* The only cross-type equality: int vs float, compared exactly. */
            return IS_INT(a) ? intEqualsDouble(AS_INT(a), AS_FLOAT(b))
                             : intEqualsDouble(AS_INT(b), AS_FLOAT(a));
        }
        /* `null` is a unit type with one value (spec §2.1), so nothing else is
         * equal to it and no `__eq__` is consulted: `x == null` is the null
         * test the whole optional machinery rests on, and handing null to a
         * user comparison would break it in the other direction too. */
        if (IS_NULL(a) || IS_NULL(b)) return false;

        /* An instance may still claim equality with a value of another type. */
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

    switch (jaiValueType(a)) {
        case VAL_NULL:  return true;
        case VAL_BOOL:  return AS_BOOL(a) == AS_BOOL(b);
        case VAL_INT:   return AS_INT(a) == AS_INT(b);
        case VAL_FLOAT: return AS_FLOAT(a) == AS_FLOAT(b);   /* NaN != NaN */
        case VAL_OBJ:   return objectsEqual(AS_OBJ(a), AS_OBJ(b), a, b);
    }
    JAI_UNREACHABLE();
    return false;
}

bool jaiValuesEqual(Value a, Value b) {
    if (eqDepth >= JAI_EQ_MAX_DEPTH) {
        if (vm.cRecursionError != NULL) {
            jaiThrow(vm.cRecursionError,
                     "maximum recursion depth exceeded while comparing values");
        }
        return false;
    }
    eqDepth++;
    bool result = valuesEqualInner(a, b);
    eqDepth--;
    return result;
}

bool jaiValuesIdentical(Value a, Value b) {
    if (jaiValueType(a) != jaiValueType(b)) return false;
    switch (jaiValueType(a)) {
        case VAL_NULL:  return true;
        case VAL_BOOL:  return AS_BOOL(a) == AS_BOOL(b);
        case VAL_INT:   return AS_INT(a) == AS_INT(b);
        case VAL_FLOAT: {
            /* Bit identity, so `nan is nan` holds and `0.0 is -0.0` does not. */
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

/* ------------------------------------------------------------------ */
/* Hashing                                                             */
/* ------------------------------------------------------------------ */

static int hashDepth = 0;

/* A float that denotes an integer hashes as that integer, so that the equal
 * values 1 and 1.0 land in the same bucket. */
static uint64_t hashDouble(double d) {
    int64_t i;
    if (doubleIsExactInt(d, &i)) return jaiHashU64((uint64_t)i);
    uint64_t bits;
    memcpy(&bits, &d, sizeof bits);
    return jaiHashU64(bits);
}

static uint64_t hashPointer(const void *p) {
    return jaiHashU64((uint64_t)(uintptr_t)p);
}

static uint64_t hashCombine(uint64_t acc, uint64_t h) {
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
    if (h == 0) h = 1;      /* 0 is the "not computed" marker */
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
        *ok = false;                     /* exception pending, or no such method */
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
    switch (jaiValueType(v)) {
        /* `null` is not a valid key (spec §5.4). JaiEntry encodes an empty slot
         * as NULL_VAL, so allowing it would need a third slot state in every
         * table entry — and "is the key absent" and "is the key null" are the
         * same question anyway. jaiDictSet and jaiSetAdd turn *ok = false into
         * the TypeError; nothing else may reach the table with such a key. */
        case VAL_NULL:  *ok = false; return 0;
        case VAL_BOOL:  return jaiHashU64(AS_BOOL(v) ? 0x5851F42D4C957F2DULL : 0x14057B7EF767814FULL);
        case VAL_INT:   return jaiHashU64((uint64_t)AS_INT(v));
        case VAL_FLOAT: return hashDouble(AS_FLOAT(v));
        case VAL_OBJ:   break;
    }

    switch (OBJ_TYPE(v)) {
        case OBJ_STRING:   return jaiStringHash(AS_STRING(v));
        case OBJ_BYTES:    return jaiHashBytes(AS_BYTES(v)->data, AS_BYTES(v)->length);
        case OBJ_TUPLE:    return hashTuple(AS_TUPLE(v), ok);
        case OBJ_ENUM_VAL: return hashEnumVal(AS_ENUM_VAL(v), ok);
        case OBJ_INSTANCE: return hashInstance(AS_INSTANCE(v), ok);
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
            *ok = false;                 /* mutable containers are unhashable */
            return 0;
        default:
            return hashPointer(AS_OBJ(v));   /* functions, classes, modules, ... */
    }
}

uint64_t jaiValueHash(Value v, bool *ok) {
    bool localOk = true;
    if (ok == NULL) ok = &localOk;
    *ok = true;

    if (hashDepth >= JAI_HASH_MAX_DEPTH) { *ok = false; return 0; }
    hashDepth++;
    uint64_t h = valueHashInner(v, ok);
    hashDepth--;
    return *ok ? h : 0;
}

/* ------------------------------------------------------------------ */
/* Ordering                                                            */
/* ------------------------------------------------------------------ */

static bool instanceHasLt(Value v) {
    return IS_INSTANCE(v) && AS_INSTANCE(v)->klass != NULL &&
           !IS_NULL(AS_INSTANCE(v)->klass->dunderLt);
}

/* Evaluate a.__lt__(b). Returns false with the exception pending on error, or
 * with *missing set when the method could not be invoked. */
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

/* Order two values where at least one is an instance with __lt__: a < b decides
 * -1, otherwise b < a decides 1, otherwise they compare equal. */
static bool compareWithLt(Value a, Value b, int *out) {
    bool less = false, missing = false;

    if (instanceHasLt(a)) {
        if (!dispatchLt(a, b, &less, &missing)) return false;
        if (less) { *out = -1; return true; }
    }
    if (instanceHasLt(b)) {
        if (!dispatchLt(b, a, &less, &missing)) return false;
        if (less) { *out = 1; return true; }
        *out = 0;
        return true;
    }
    if (!instanceHasLt(a)) return false;

    /* Only `a` knows how to order itself; not-less means equal or greater. */
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
    if (IS_NUMBER(a) && IS_NUMBER(b)) {
        if (IS_INT(a) && IS_INT(b)) {
            int64_t x = AS_INT(a), y = AS_INT(b);
            *out = (x < y) ? -1 : (x > y) ? 1 : 0;
            return true;
        }
        if (IS_INT(a)) return compareIntDouble(AS_INT(a), AS_FLOAT(b), out);
        if (IS_INT(b)) {
            int inverted;
            if (!compareIntDouble(AS_INT(b), AS_FLOAT(a), &inverted)) return false;
            *out = -inverted;
            return true;
        }
        return compareDoubles(AS_FLOAT(a), AS_FLOAT(b), out);
    }

    if (IS_BOOL(a) && IS_BOOL(b)) {
        int x = AS_BOOL(a) ? 1 : 0, y = AS_BOOL(b) ? 1 : 0;
        *out = (x < y) ? -1 : (x > y) ? 1 : 0;
        return true;
    }

    if (IS_STRING(a) && IS_STRING(b)) {
        /* Byte order over UTF-8 is code-point order. */
        return compareStrings(AS_STRING(a), AS_STRING(b), out);
    }

    if (instanceHasLt(a) || instanceHasLt(b)) return compareWithLt(a, b, out);

    return false;
}

/* ------------------------------------------------------------------ */
/* Rendering                                                           */
/* ------------------------------------------------------------------ */

/* One renderer, two destinations: a growable buffer (str/repr) or a stream
 * (the disassembler and traceback printer, which must not allocate an
 * ObjString just to print one). */
typedef struct {
    JaiBuf *buf;
    FILE   *file;
} ValSink;

static void sinkWrite(ValSink *s, const char *data, size_t n) {
    if (n == 0) return;
    if (s->buf != NULL) {
        jaiBufAppend(s->buf, data, n);
    } else {
        size_t wrote = fwrite(data, 1, n, s->file);
        (void)wrote;
    }
}

static void sinkStr(ValSink *s, const char *str) { sinkWrite(s, str, strlen(str)); }

/* Decimal digits of an int64 into `out`, which must hold 20 bytes plus a sign.
 * Returns the length; nothing is NUL-terminated because every caller has the
 * length in hand. The negation goes through uint64_t so INT64_MIN, whose
 * absolute value is not representable, still comes out right. */
#define JAI_INT_DIGITS 24

static int writeInt64(char *out, int64_t value) {
    char rev[20];
    uint64_t u = value < 0 ? (uint64_t)(-(value + 1)) + 1u : (uint64_t)value;
    int n = 0;
    do {
        rev[n++] = (char)('0' + (int)(u % 10u));
        u /= 10u;
    } while (u != 0);

    int len = 0;
    if (value < 0) out[len++] = '-';
    while (n > 0) out[len++] = rev[--n];
    return len;
}

/* `sinkFmt(s, "%" PRId64, ...)` drags in the whole of vsnprintf — locale
 * lookup and all — to lay down at most twenty digits. Integers are the single
 * most rendered thing in the language, so they get their own path. */
static void sinkInt(ValSink *s, int64_t value) {
    char digits[JAI_INT_DIGITS];
    sinkWrite(s, digits, (size_t)writeInt64(digits, value));
}

static void sinkFmt(ValSink *s, const char *fmt, ...) JAI_PRINTF(2, 3);
static void sinkFmt(ValSink *s, const char *fmt, ...) {
    char stackBuf[256];
    va_list ap;

    va_start(ap, fmt);
    int n = vsnprintf(stackBuf, sizeof stackBuf, fmt, ap);
    va_end(ap);
    if (n < 0) return;

    if ((size_t)n < sizeof stackBuf) {
        sinkWrite(s, stackBuf, (size_t)n);
        return;
    }
    char *heap = JAI_ALLOC(char, (size_t)n + 1);
    va_start(ap, fmt);
    vsnprintf(heap, (size_t)n + 1, fmt, ap);
    va_end(ap);
    sinkWrite(s, heap, (size_t)n);
    JAI_FREE_ARRAY(char, heap, (size_t)n + 1);
}

/* Shortest decimal form that strtod maps back to the same double, and never
 * one that would read back as an int.
 *
 * The digit count and the choice of notation are decided separately: "%g"
 * would couple them and print 100.0 as "1e+02" once the shortest round-trip
 * turns out to be one significant digit. */
static void formatDouble(char *out, size_t outSize, double d) {
    if (isnan(d)) { snprintf(out, outSize, "nan"); return; }
    if (isinf(d)) { snprintf(out, outSize, d < 0 ? "-inf" : "inf"); return; }

    char sci[64];
    int digits = 17;
    for (int p = 1; p <= 17; p++) {
        snprintf(sci, sizeof sci, "%.*e", p - 1, d);
        if (strtod(sci, NULL) == d) { digits = p; break; }   /* 17 always ends this */
    }

    const char *e = strchr(sci, 'e');
    int exp = (e != NULL) ? (int)strtol(e + 1, NULL, 10) : 0;

    /* Same switch-over points as Python's float repr, so round numbers stay
     * readable and extremes stay short. */
    if (exp < -4 || exp >= 16) {
        snprintf(out, outSize, "%s", sci);
        return;
    }
    int decimals = digits - 1 - exp;
    if (decimals < 0) decimals = 0;
    snprintf(out, outSize, "%.*f", decimals, d);
    if (strchr(out, '.') == NULL) {
        size_t len = strlen(out);
        if (len + 3 <= outSize) memcpy(out + len, ".0", 3);
    }
}

static void emitQuotedString(ValSink *s, const char *chars, size_t length) {
    const char *p = chars;
    const char *end = chars + length;

    sinkWrite(s, "\"", 1);
    while (p < end) {
        unsigned char c = (unsigned char)*p;
        switch (c) {
            case '\\': sinkWrite(s, "\\\\", 2); p++; continue;
            case '"':  sinkWrite(s, "\\\"", 2); p++; continue;
            case '\n': sinkWrite(s, "\\n", 2);  p++; continue;
            case '\t': sinkWrite(s, "\\t", 2);  p++; continue;
            case '\r': sinkWrite(s, "\\r", 2);  p++; continue;
            default: break;
        }
        if (c < 0x80) {
            if (c < 0x20 || c == 0x7f) sinkFmt(s, "\\u{%X}", (unsigned)c);
            else sinkWrite(s, p, 1);
            p++;
            continue;
        }
        int len = 1;
        int32_t cp = jaiUtf8Decode(p, end, &len);
        if (len <= 0) len = 1;
        if (cp < 0) {
            sinkFmt(s, "\\x%02X", (unsigned)c);      /* not valid UTF-8 */
        } else if (cp >= 0x80 && cp <= 0x9f) {
            sinkFmt(s, "\\u{%X}", (unsigned)cp);     /* C1 controls */
        } else {
            sinkWrite(s, p, (size_t)len);
        }
        p += len;
    }
    sinkWrite(s, "\"", 1);
}

static void emitBytesLiteral(ValSink *s, const uint8_t *data, size_t length) {
    sinkWrite(s, "b\"", 2);
    for (size_t i = 0; i < length; i++) {
        uint8_t c = data[i];
        switch (c) {
            case '\\': sinkWrite(s, "\\\\", 2); continue;
            case '"':  sinkWrite(s, "\\\"", 2); continue;
            case '\n': sinkWrite(s, "\\n", 2);  continue;
            case '\t': sinkWrite(s, "\\t", 2);  continue;
            case '\r': sinkWrite(s, "\\r", 2);  continue;
            default: break;
        }
        if (c >= 0x20 && c < 0x7f) sinkWrite(s, (const char *)&data[i], 1);
        else sinkFmt(s, "\\x%02X", (unsigned)c);
    }
    sinkWrite(s, "\"", 1);
}

/* Containers currently being rendered, innermost last; a repeat is a cycle. */
static const Obj *renderStack[JAI_RENDER_MAX_DEPTH];
static int renderDepth = 0;

static bool renderInProgress(const Obj *o) {
    for (int i = 0; i < renderDepth; i++) {
        if (renderStack[i] == o) return true;
    }
    return false;
}

static bool renderValue(ValSink *s, Value v, bool repr, bool allowUser);

/* Enter a container: emits the Python-style elision and returns false when the
 * container is already being rendered or the nesting limit is reached. */
static bool renderEnter(ValSink *s, const Obj *o, const char *elision) {
    if (renderInProgress(o) || renderDepth >= JAI_RENDER_MAX_DEPTH) {
        sinkStr(s, elision);
        return false;
    }
    renderStack[renderDepth++] = o;
    return true;
}

static void renderLeave(void) {
    if (renderDepth > 0) renderDepth--;
}

static void renderFnLike(ValSink *s, Value v) {
    switch (OBJ_TYPE(v)) {
        case OBJ_FUNCTION: {
            ObjFunction *fn = AS_FUNCTION(v);
            sinkFmt(s, "<fn %s>", fn->name != NULL ? fn->name->chars : "anonymous");
            return;
        }
        case OBJ_CLOSURE: {
            ObjFunction *fn = AS_CLOSURE(v)->fn;
            const char *name = (fn != NULL && fn->name != NULL) ? fn->name->chars
                                                                : "anonymous";
            sinkFmt(s, "<fn %s>", name);
            return;
        }
        case OBJ_NATIVE: {
            ObjNative *n = AS_NATIVE(v);
            sinkFmt(s, "<native fn %s>", n->name != NULL ? n->name->chars : "anonymous");
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
            sinkFmt(s, "<bound method %s.%s>", jaiTypeNameStatic(b->receiver), method);
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
            dunder = vm.strRepr;      /* str() falls back to repr() */
        }
    }
    if (dunder == NULL) {
        sinkFmt(s, "<%s instance>", classNameOf(k));
        return true;
    }

    Value out;
    if (!jaiInvokeMethod(OBJ_VAL(inst), dunder, 0, NULL, &out)) {
        if (vm.hasException) return false;
        sinkFmt(s, "<%s instance>", classNameOf(k));   /* method vanished */
        return true;
    }
    if (!IS_STRING(out)) {
        jaiThrow(vm.cTypeError, "%s must return str, not %s", dunder->chars,
                 jaiTypeNameStatic(out));
        return false;
    }

    ObjString *str = AS_STRING(out);
    bool rooted = tempRoot(out);          /* sinkWrite may allocate and collect */
    sinkWrite(s, str->chars, str->length);
    tempUnroot(rooted);
    return true;
}

static bool renderList(ValSink *s, ObjList *l, bool allowUser) {
    if (!renderEnter(s, (const Obj *)l, "[...]")) return true;
    sinkWrite(s, "[", 1);
    bool ok = true;
    /* Re-read count each step: a user __repr__ may mutate the list. */
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
    sinkWrite(s, "(", 1);
    bool ok = true;
    for (uint32_t i = 0; i < t->count; i++) {
        if (i > 0) sinkWrite(s, ", ", 2);
        ok = renderValue(s, t->items[i], true, allowUser);
        if (!ok) break;
    }
    /* A one-tuple needs the trailing comma to stay distinguishable from (x). */
    if (ok && t->count == 1) sinkWrite(s, ",", 1);
    if (ok) sinkWrite(s, ")", 1);
    renderLeave();
    return ok;
}

static bool renderDict(ValSink *s, ObjDict *d, bool allowUser) {
    if (!renderEnter(s, (const Obj *)d, "{...}")) return true;
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
    /* `{}` is the empty dict literal, so the empty set has to spell itself. */
    if (set->table.count == 0) { sinkStr(s, "set()"); return true; }
    if (!renderEnter(s, (const Obj *)set, "{...}")) return true;
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
    sinkFmt(s, "%s.%s", typeName, variant);
    if (e->count == 0) return true;

    if (!renderEnter(s, (const Obj *)e, "(...)")) return true;
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
    sinkFmt(s, "%" PRId64 "%s%" PRId64, r->start, r->inclusive ? "..=" : "..",
            r->stop);
    /* A stepped range has no `..` spelling; borrow the slice notation. */
    if (r->step != 1) sinkFmt(s, ":%" PRId64, r->step);
}

/* Returns false only when a user dunder raised; the exception is then pending
 * and whatever was emitted so far is discarded by the caller. */
static bool renderValue(ValSink *s, Value v, bool repr, bool allowUser) {
    switch (jaiValueType(v)) {
        case VAL_NULL: sinkStr(s, "null"); return true;
        case VAL_BOOL: sinkStr(s, AS_BOOL(v) ? "true" : "false"); return true;
        case VAL_INT:  sinkInt(s, AS_INT(v)); return true;
        case VAL_FLOAT: {
            char buf[48];
            formatDouble(buf, sizeof buf, AS_FLOAT(v));
            sinkStr(s, buf);
            return true;
        }
        case VAL_OBJ: break;
    }

    switch (OBJ_TYPE(v)) {
        case OBJ_STRBUF: break;   /* never reaches a Value */
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
            sinkFmt(s, "<class %s>", classNameOf(AS_CLASS(v)));
            return true;
        case OBJ_TRAIT: {
            ObjTrait *t = AS_TRAIT(v);
            sinkFmt(s, "<trait %s>", t->name != NULL ? t->name->chars : "?");
            return true;
        }
        case OBJ_MODULE: {
            ObjModule *m = AS_MODULE(v);
            sinkFmt(s, "<module %s>", m->name != NULL ? m->name->chars : "?");
            return true;
        }
        case OBJ_ENUM: {
            ObjEnum *e = AS_ENUM(v);
            sinkFmt(s, "<enum %s>", e->name != NULL ? e->name->chars : "?");
            return true;
        }
        case OBJ_FILE: {
            ObjFile *f = AS_FILE(v);
            const char *path = f->path != NULL ? f->path->chars : "?";
            sinkFmt(s, "<%sfile '%s'>", f->closed ? "closed " : "", path);
            return true;
        }
        case OBJ_ENUM_CTOR: {
            ObjEnumCtor *c = AS_ENUM_CTOR(v);
            const char *en = (c->type != NULL && c->type->name != NULL)
                                 ? c->type->name->chars : "?";
            const char *vn = (c->type != NULL && c->tag < c->type->variantCount &&
                              c->type->variants[c->tag].name != NULL)
                                 ? c->type->variants[c->tag].name->chars : "?";
            sinkFmt(s, "<enum constructor %s.%s>", en, vn);
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
    /* A scalar's rendering is short, bounded, and identical under str and
     * repr, so it needs neither a growable buffer nor the root dance: going
     * straight to the string skips a malloc/free pair per f-string hole. */
    switch (jaiValueType(v)) {
    case VAL_INT: {
        char digits[JAI_INT_DIGITS];
        return jaiStringNew(digits, (size_t)writeInt64(digits, AS_INT(v)));
    }
    case VAL_FLOAT: {
        char digits[48];
        formatDouble(digits, sizeof digits, AS_FLOAT(v));
        return jaiStringNew(digits, strlen(digits));
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

    if (!ok) {                       /* exception pending; the caller unwinds */
        jaiBufFree(&buf);
        return NULL;
    }
    ObjString *out = jaiStringNew(buf.data != NULL ? (const char *)buf.data : "",
                                  buf.count);
    jaiBufFree(&buf);
    return out;
}

/* --- f-string assembly ------------------------------------------- */

/* Anything the measure-then-fill path cannot size up front — a list, a dict,
 * an instance with __str__ — goes through the ordinary renderer into a buffer.
 * The parts are rooted by the caller (OP_FORMAT leaves them on the value
 * stack), so a user dunder collecting mid-render is safe. */
static ObjString *formatViaBuffer(const Value *parts, int count) {
    JaiBuf buf;
    jaiBufInit(&buf);
    ValSink sink = {&buf, NULL};

    for (int i = 0; i < count; i++) {
        if (!renderValue(&sink, parts[i], false, true)) {
            jaiBufFree(&buf);
            return NULL;                 /* a user dunder raised */
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

    /* Measure first, then fill: every scalar's rendered length is known
     * without a buffer, so the result is one exactly-sized allocation. The
     * scalars are rendered once, here, into scratch that the assembly step
     * reads back. */
    char        scratch[JAI_FMT_MAX_PARTS][48];
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
            formatDouble(scratch[i], sizeof scratch[i], AS_FLOAT(v));
            lens[i] = (uint32_t)strlen(scratch[i]);
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
    if (IS_STRING(v)) return AS_STRING(v);   /* str() of a str is itself */
    return renderToString(v, false);
}

ObjString *jaiValueToRepr(Value v) {
    return renderToString(v, true);
}

void jaiPrintValue(FILE *out, Value v, bool repr) {
    if (out == NULL) return;
    ValSink sink = {NULL, out};
    /* allowUser = false: this runs from the disassembler and from traceback
     * printing, where re-entering the interpreter is not safe. */
    (void)renderValue(&sink, v, repr, false);
}
