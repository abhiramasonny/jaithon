/* builtins_core.c — the functions a Jaithon program can name without importing anything (spec §9). */

#include <ctype.h>
#include <math.h>
#include <stdlib.h>

#include "runtime/builtins/builtins.h"
#include "runtime/builtins/collections/builtins_seq.h"
#include "runtime/methods.h"
#include "runtime/runtime.h"

#include "vm/gc.h"

// Small shared helpers

static ObjString *dunderName(ObjString *cached, const char *text) {
    return cached != NULL ? cached : jaiStringInternC(text);
}

static bool valueLength(Value v, int64_t *out) {
    if (!IS_OBJ(v)) return false;
    switch (OBJ_TYPE(v)) {
    case OBJ_STRING: *out = (int64_t)jaiStringScalarCount(AS_STRING(v)); return true;
    case OBJ_BYTES:  *out = (int64_t)AS_BYTES(v)->length;                return true;
    case OBJ_LIST:   *out = (int64_t)AS_LIST(v)->count;                  return true;
    case OBJ_TUPLE:  *out = (int64_t)AS_TUPLE(v)->count;                 return true;
    case OBJ_DICT:   *out = (int64_t)AS_DICT(v)->table.count;            return true;
    case OBJ_SET:    *out = (int64_t)AS_SET(v)->table.count;             return true;
    case OBJ_RANGE:  *out = jaiRangeLength(AS_RANGE(v));                 return true;
    case OBJ_INSTANCE: {
        ObjInstance *inst = AS_INSTANCE(v);
        if (inst->klass == NULL || IS_NULL(inst->klass->dunderLen)) return false;
        Value result;
        if (!jaiInvokeMethod(v, dunderName(vm.strLen, "__len__"), 0, NULL, &result))
            return false;
        if (!IS_INT(result)) {
            jaiThrow(vm.cTypeError, "__len__ must return int, not %s",
                     jaiTypeNameStatic(result));
            return false;
        }
        if (AS_INT(result) < 0) {
            jaiThrow(vm.cValueError, "__len__ must return a non-negative int");
            return false;
        }
        *out = AS_INT(result);
        return true;
    }
    default:
        return false;
    }
}

static ObjList *collectIterable(Value v) {
    Value iterVal;
    if (!jaiGetIter(v, &iterVal)) return NULL;
    if (!IS_ITER(iterVal)) {
        jaiThrow(vm.cTypeError, "'%s' object is not iterable", jaiTypeNameStatic(v));
        return NULL;
    }

    jaiGCPushRoot(iterVal);
    ObjList *out = jaiListNew(0);
    jaiGCPushRoot(OBJ_VAL(out));

    Value item;
    while (jaiIterNext(AS_ITER(iterVal), &item)) {
        jaiGCPushRoot(item);
        jaiListPush(out, item);
        jaiGCPopRoot();
    }
    jaiGCPopRoots(2);

    return vm.hasException ? NULL : out;
}

/* Turns "no order between these two" into a TypeError, as every ordering builtin wants. */
static bool compareOrThrow(Value a, Value b, const char *fnName, int *out) {
    if (jaiValueCompare(a, b, out)) return true;
    if (vm.hasException) return false;
    return jaiThrow(vm.cTypeError, "%s(): cannot compare %s with %s", fnName,
                    jaiTypeNameStatic(a), jaiTypeNameStatic(b));
}

static bool requireHashable(Value v, const char *fnName) {
    bool ok = true;
    (void)jaiValueHash(v, &ok);
    if (ok) return true;
    if (vm.hasException) return false;
    return jaiThrow(vm.cTypeError, "%s(): unhashable type: '%s'", fnName,
                    jaiTypeNameStatic(v));
}

// Text to number

typedef enum { PARSE_OK, PARSE_BAD, PARSE_RANGE } ParseResult;

static int digitValue(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'z') return c - 'a' + 10;
    if (c >= 'A' && c <= 'Z') return c - 'A' + 10;
    return -1;
}

static bool matchPrefix(const char *s, size_t len, size_t i, char lower, char upper) {
    return i + 1 < len && s[i] == '0' && (s[i + 1] == lower || s[i + 1] == upper);
}

/* Accepts what the lexer accepts: sign, 0x/0o/0b prefixes when the base allows
 * them, and '_' separators between digits. base 0 means "detect from prefix". */
static ParseResult parseIntText(const char *s, size_t len, int base, int64_t *out) {
    size_t i = 0, end = len;
    while (i < end && isspace((unsigned char)s[i])) i++;
    while (end > i && isspace((unsigned char)s[end - 1])) end--;
    if (i == end) return PARSE_BAD;

    bool negative = false;
    if (s[i] == '+' || s[i] == '-') {
        negative = s[i] == '-';
        i++;
    }

    if (base == 0 || base == 16) {
        if (matchPrefix(s, end, i, 'x', 'X')) { base = 16; i += 2; }
    }
    if (base == 0 || base == 8) {
        if (matchPrefix(s, end, i, 'o', 'O')) { base = 8; i += 2; }
    }
    if (base == 0 || base == 2) {
        if (matchPrefix(s, end, i, 'b', 'B')) { base = 2; i += 2; }
    }
    if (base == 0) base = 10;

    uint64_t limit = negative ? (uint64_t)INT64_MAX + 1u : (uint64_t)INT64_MAX;
    uint64_t acc = 0;
    bool sawDigit = false;
    bool sawSeparator = true;   /* a leading '_' is not a separator */

    for (; i < end; i++) {
        char c = s[i];
        if (c == '_') {
            if (sawSeparator) return PARSE_BAD;
            sawSeparator = true;
            continue;
        }
        int d = digitValue(c);
        if (d < 0 || d >= base) return PARSE_BAD;
        if (acc > (limit - (uint64_t)d) / (uint64_t)base) return PARSE_RANGE;
        acc = acc * (uint64_t)base + (uint64_t)d;
        sawDigit = true;
        sawSeparator = false;
    }
    if (!sawDigit || sawSeparator) return PARSE_BAD;

    if (negative) {
        *out = acc == (uint64_t)INT64_MAX + 1u ? INT64_MIN : -(int64_t)acc;
    } else {
        *out = (int64_t)acc;
    }
    return PARSE_OK;
}

static ParseResult parseFloatText(const char *s, size_t len, double *out) {
    char stackBuf[64];
    char *buf = stackBuf;
    bool heap = false;
    if (len + 1 > sizeof stackBuf) {
        buf = JAI_ALLOC(char, len + 1);
        heap = true;
    }

    size_t n = 0;
    for (size_t i = 0; i < len; i++) {
        if (s[i] == '_') continue;
        buf[n++] = s[i];
    }
    buf[n] = '\0';

    const char *start = buf;
    while (*start != '\0' && isspace((unsigned char)*start)) start++;

    char *endPtr = NULL;
    double value = strtod(start, &endPtr);
    ParseResult result = PARSE_OK;
    if (endPtr == start) {
        result = PARSE_BAD;
    } else {
        while (*endPtr != '\0' && isspace((unsigned char)*endPtr)) endPtr++;
        if (*endPtr != '\0') result = PARSE_BAD;
    }

    if (heap) JAI_FREE_ARRAY(char, buf, len + 1);
    if (result == PARSE_OK) *out = value;
    return result;
}

// The copy a sort hands back

static ObjList *sortedCopy(ObjList *items, ObjList *keys, bool reverse,
                           const char *fnName) {
    int n = items->count;
    ObjList *result = jaiListNew(n);
    jaiGCPushRoot(OBJ_VAL(result));
    if (n <= 1) {
        for (int i = 0; i < n; i++) jaiListPush(result, jaiListGet(items, i));
        jaiGCPopRoot();
        return result;
    }

    int *idx = JAI_ALLOC(int, n);
    int *scratch = JAI_ALLOC(int, n);
    for (int i = 0; i < n; i++) idx[i] = i;

    bool ok = jaiSeqSortIndices(keys, idx, scratch, n, reverse, fnName);
    if (ok) {
        for (int i = 0; i < n; i++) jaiListPush(result, jaiListGet(items, idx[i]));
    }

    JAI_FREE_ARRAY(int, idx, n);
    JAI_FREE_ARRAY(int, scratch, n);
    jaiGCPopRoot();
    return ok ? result : NULL;
}

// Core builtins

static bool nPrint(int argc, Value *args, Value *out) {
    for (int i = 0; i < argc; i++) {
        ObjString *text = jaiValueToStr(args[i]);
        if (text == NULL) return false;
        if (i > 0) fputc(' ', stdout);
        (void)fwrite(text->chars, 1, text->length, stdout);
    }
    fputc('\n', stdout);
    *out = NULL_VAL;
    return true;
}

static bool nInput(int argc, Value *args, Value *out) {
    if (argc >= 1 && !IS_NULL(args[0])) {
        ObjString *prompt = jaiValueToStr(args[0]);
        if (prompt == NULL) return false;
        (void)fwrite(prompt->chars, 1, prompt->length, stdout);
    }
    fflush(stdout);

    JaiBuf line;
    jaiBufInit(&line);
    int c;
    bool sawAny = false;
    while ((c = fgetc(stdin)) != EOF) {
        sawAny = true;
        if (c == '\n') break;
        jaiBufPush(&line, (uint8_t)c);
    }
    if (!sawAny) {
        jaiBufFree(&line);
        return jaiThrow(vm.cIOError, "input(): end of input");
    }
    if (line.count > 0 && line.data[line.count - 1] == '\r') line.count--;

    ObjString *text = jaiStringNew(line.data != NULL ? (const char *)line.data : "",
                                   line.count);
    jaiBufFree(&line);
    if (text == NULL) return false;
    *out = OBJ_VAL(text);
    return true;
}

static bool nLen(int argc, Value *args, Value *out) {
    (void)argc;
    int64_t length;
    if (!valueLength(args[0], &length)) {
        if (vm.hasException) return false;
        return jaiThrow(vm.cTypeError, "object of type '%s' has no len()",
                        jaiTypeNameStatic(args[0]));
    }
    *out = INT_VAL(length);
    return true;
}

static bool nRange(int argc, Value *args, Value *out) {
    int64_t start = 0, stop = 0, step = 1;
    if (argc == 1) {
        if (!jaiArgInt(args[0], 1, "range", &stop)) return false;
    } else {
        if (!jaiArgInt(args[0], 1, "range", &start)) return false;
        if (!jaiArgInt(args[1], 2, "range", &stop)) return false;
        if (argc >= 3 && !jaiArgInt(args[2], 3, "range", &step)) return false;
    }
    if (step == 0) return jaiThrow(vm.cValueError, "range() step must not be zero");
    *out = OBJ_VAL(jaiRangeNew(start, stop, step, false));
    return true;
}

static bool nType(int argc, Value *args, Value *out) {
    (void)argc;
    *out = OBJ_VAL(jaiTypeName(args[0]));
    return true;
}

static bool nRepr(int argc, Value *args, Value *out) {
    (void)argc;
    ObjString *text = jaiValueToRepr(args[0]);
    if (text == NULL) return false;
    *out = OBJ_VAL(text);
    return true;
}

static bool nStr(int argc, Value *args, Value *out) {
    (void)argc;
    ObjString *text = jaiValueToStr(args[0]);
    if (text == NULL) return false;
    *out = OBJ_VAL(text);
    return true;
}

static bool floatToInt(double d, int64_t *out) {
    if (isnan(d)) return jaiThrow(vm.cValueError, "cannot convert float NaN to int");
    if (isinf(d)) return jaiThrow(vm.cOverflowError,
                                  "cannot convert float infinity to int");
    double truncated = trunc(d);
    if (truncated >= 9223372036854775808.0 || truncated < -9223372036854775808.0)
        return jaiThrow(vm.cOverflowError, "float %g is out of range for int", d);
    *out = (int64_t)truncated;
    return true;
}

static bool nIntConv(int argc, Value *args, Value *out) {
    Value v = args[0];

    if (argc >= 2 && !IS_NULL(args[1])) {
        int64_t base;
        if (!jaiArgInt(args[1], 2, "int", &base)) return false;
        if (!IS_STRING(v)) return jaiBuiltinArgTypeError(1, "int", "str", v);
        if (base != 0 && (base < 2 || base > 36))
            return jaiThrow(vm.cValueError,
                            "int() base must be 0 or between 2 and 36, got %lld",
                            (long long)base);
        ObjString *text = AS_STRING(v);
        int64_t parsed = 0;
        switch (parseIntText(text->chars, text->length, (int)base, &parsed)) {
        case PARSE_OK:
            *out = INT_VAL(parsed);
            return true;
        case PARSE_RANGE:
            return jaiThrow(vm.cOverflowError,
                            "int() literal out of range: '%.*s'",
                            (int)text->length, text->chars);
        case PARSE_BAD:
            break;
        }
        return jaiThrow(vm.cValueError,
                        "invalid literal for int() with base %lld: '%.*s'",
                        (long long)base, (int)text->length, text->chars);
    }

    switch (jaiValueType(v)) {
    case VAL_INT:
        *out = v;
        return true;
    case VAL_BOOL:
        *out = INT_VAL(AS_BOOL(v) ? 1 : 0);
        return true;
    case VAL_FLOAT: {
        int64_t truncated;
        if (!floatToInt(AS_FLOAT(v), &truncated)) return false;
        *out = INT_VAL(truncated);
        return true;
    }
    default:
        break;
    }

    if (IS_STRING(v)) {
        ObjString *text = AS_STRING(v);
        int64_t parsed = 0;
        switch (parseIntText(text->chars, text->length, 10, &parsed)) {
        case PARSE_OK:
            *out = INT_VAL(parsed);
            return true;
        case PARSE_RANGE:
            return jaiThrow(vm.cOverflowError,
                            "int() literal out of range: '%.*s'",
                            (int)text->length, text->chars);
        case PARSE_BAD:
            break;
        }
        return jaiThrow(vm.cValueError,
                        "invalid literal for int() with base 10: '%.*s'",
                        (int)text->length, text->chars);
    }
    return jaiBuiltinArgTypeError(1, "int", "int, float, bool or str", v);
}

static bool nFloatConv(int argc, Value *args, Value *out) {
    (void)argc;
    Value v = args[0];
    switch (jaiValueType(v)) {
    case VAL_FLOAT: *out = v; return true;
    case VAL_INT:   *out = FLOAT_VAL((double)AS_INT(v)); return true;
    case VAL_BOOL:  *out = FLOAT_VAL(AS_BOOL(v) ? 1.0 : 0.0); return true;
    default: break;
    }
    if (IS_STRING(v)) {
        ObjString *text = AS_STRING(v);
        double parsed = 0.0;
        if (parseFloatText(text->chars, text->length, &parsed) == PARSE_OK) {
            *out = FLOAT_VAL(parsed);
            return true;
        }
        return jaiThrow(vm.cValueError, "invalid literal for float(): '%.*s'",
                        (int)text->length, text->chars);
    }
    return jaiBuiltinArgTypeError(1, "float", "int, float, bool or str", v);
}

static bool nBoolConv(int argc, Value *args, Value *out) {
    (void)argc;
    Value v = args[0];
    switch (jaiValueType(v)) {
    case VAL_BOOL:  *out = v; return true;
    case VAL_NULL:  *out = BOOL_VAL(false); return true;
    case VAL_INT:   *out = BOOL_VAL(AS_INT(v) != 0); return true;
    case VAL_FLOAT: *out = BOOL_VAL(AS_FLOAT(v) != 0.0); return true;
    default: break;
    }
    int64_t length;
    if (valueLength(v, &length)) {
        *out = BOOL_VAL(length != 0);
        return true;
    }
    if (vm.hasException) return false;
    return jaiThrow(vm.cTypeError, "bool(): '%s' has no truth value",
                    jaiTypeNameStatic(v));
}

static bool nId(int argc, Value *args, Value *out) {
    (void)argc;
    Value v = args[0];
    switch (jaiValueType(v)) {
    case VAL_NULL:  *out = INT_VAL(0); return true;
    case VAL_BOOL:  *out = INT_VAL(AS_BOOL(v) ? 1 : 2); return true;
    case VAL_INT:   *out = INT_VAL(AS_INT(v)); return true;
    case VAL_FLOAT: {
        double d = AS_FLOAT(v);
        uint64_t bits;
        memcpy(&bits, &d, sizeof bits);
        *out = INT_VAL((int64_t)bits);
        return true;
    }
    case VAL_OBJ: {
        Obj *o = AS_OBJ(v);
        uint64_t address = 0;
        memcpy(&address, &o, sizeof o < sizeof address ? sizeof o : sizeof address);
        *out = INT_VAL((int64_t)address);
        return true;
    }
    }
    *out = INT_VAL(0);
    return true;
}

static bool nHash(int argc, Value *args, Value *out) {
    (void)argc;
    bool ok = true;
    uint64_t h = jaiValueHash(args[0], &ok);
    if (!ok) {
        if (vm.hasException) return false;
        return jaiThrow(vm.cTypeError, "unhashable type: '%s'",
                        jaiTypeNameStatic(args[0]));
    }
    *out = INT_VAL((int64_t)h);
    return true;
}

static bool nAbs(int argc, Value *args, Value *out) {
    (void)argc;
    Value v = args[0];
    if (IS_INT(v)) {
        int64_t n = AS_INT(v);
        if (n == INT64_MIN) return jaiBuiltinOverflowError("abs()");
        *out = INT_VAL(n < 0 ? -n : n);
        return true;
    }
    if (IS_FLOAT(v)) {
        *out = FLOAT_VAL(fabs(AS_FLOAT(v)));
        return true;
    }
    return jaiBuiltinArgTypeError(1, "abs", "int or float", v);
}

static bool extremum(int argc, Value *args, Value *out, bool wantMax) {
    const char *fnName = wantMax ? "max" : "min";

    ObjList *items = NULL;
    Value best;
    int start;

    if (argc == 1) {
        items = collectIterable(args[0]);
        if (items == NULL) return false;
        if (items->count == 0)
            return jaiThrow(vm.cValueError, "%s() argument is an empty sequence",
                            fnName);
        jaiGCPushRoot(OBJ_VAL(items));
        best = jaiListGet(items, 0);
        start = 1;
    } else {
        best = args[0];
        start = 1;
    }

    int count = items != NULL ? items->count : argc;
    bool ok = true;
    for (int i = start; i < count; i++) {
        Value candidate = items != NULL ? jaiListGet(items, i) : args[i];
        int order;
        if (!compareOrThrow(candidate, best, fnName, &order)) { ok = false; break; }
        if (wantMax ? order > 0 : order < 0) best = candidate;
    }

    if (items != NULL) jaiGCPopRoot();
    if (!ok) return false;
    *out = best;
    return true;
}

static bool nMin(int argc, Value *args, Value *out) { return extremum(argc, args, out, false); }
static bool nMax(int argc, Value *args, Value *out) { return extremum(argc, args, out, true); }

static bool nSum(int argc, Value *args, Value *out) {
    Value total = argc >= 2 ? args[1] : INT_VAL(0);
    if (!IS_NUMBER(total)) return jaiBuiltinArgTypeError(2, "sum", "int or float", total);

    ObjList *items = collectIterable(args[0]);
    if (items == NULL) return false;
    jaiGCPushRoot(OBJ_VAL(items));

    bool ok = true;
    for (int i = 0; i < items->count; i++) {
        Value item = jaiListGet(items, i);
        if (!IS_NUMBER(item)) {
            ok = jaiBuiltinArgTypeError(1, "sum", "an iterable of int or float", item);
            break;
        }
        if (IS_INT(total) && IS_INT(item)) {
            int64_t sum;
            if (!jaiBuiltinAddI64(AS_INT(total), AS_INT(item), &sum)) {
                ok = jaiBuiltinOverflowError("sum()");
                break;
            }
            total = INT_VAL(sum);
        } else {
            total = FLOAT_VAL(jaiAsDouble(total) + jaiAsDouble(item));
        }
    }
    jaiGCPopRoot();
    if (!ok) return false;
    *out = total;
    return true;
}

static bool nSorted(int argc, Value *args, Value *out) {
    Value source = args[0];
    Value keyFn = argc >= 2 ? args[1] : NULL_VAL;
    bool reverse = false;
    if (argc >= 3 && !jaiArgBool(args[2], 3, "sorted", &reverse)) return false;
    if (!IS_NULL(keyFn) && !jaiArgCallable(keyFn, 2, "sorted")) return false;

    ObjList *items = collectIterable(source);
    if (items == NULL) return false;
    jaiGCPushRoot(OBJ_VAL(items));
    jaiGCPushRoot(keyFn);

    ObjList *keys = items;
    bool ok = true;
    if (!IS_NULL(keyFn)) {
        keys = jaiListNew(items->count);
        jaiGCPushRoot(OBJ_VAL(keys));
        for (int i = 0; i < items->count; i++) {
            Value key;
            Value arg = jaiListGet(items, i);
            if (!jaiCallFn1(keyFn, arg, &key)) { ok = false; break; }
            jaiGCPushRoot(key);
            jaiListPush(keys, key);
            jaiGCPopRoot();
        }
    }

    ObjList *sorted = NULL;
    if (ok) sorted = sortedCopy(items, keys, reverse, "sorted");

    if (keys != items) jaiGCPopRoot();
    jaiGCPopRoots(2);

    if (!ok || sorted == NULL) return false;
    *out = OBJ_VAL(sorted);
    return true;
}

static bool nReversed(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *items = collectIterable(args[0]);
    if (items == NULL) return false;
    for (int i = 0, j = items->count - 1; i < j; i++, j--) {
        Value tmp = jaiListGet(items, i);
        jaiListPut(items, i, jaiListGet(items, j));
        jaiListPut(items, j, tmp);
    }
    *out = OBJ_VAL(items);
    return true;
}

static bool nEnumerate(int argc, Value *args, Value *out) {
    int64_t start = 0;
    if (argc >= 2 && !jaiArgInt(args[1], 2, "enumerate", &start)) return false;

    ObjList *items = collectIterable(args[0]);
    if (items == NULL) return false;
    jaiGCPushRoot(OBJ_VAL(items));

    ObjList *result = jaiListNew(items->count);
    jaiGCPushRoot(OBJ_VAL(result));
    bool ok = true;
    for (int i = 0; i < items->count; i++) {
        int64_t index;
        if (!jaiBuiltinAddI64(start, (int64_t)i, &index)) { ok = jaiBuiltinOverflowError("enumerate()"); break; }
        Value pair[2] = {INT_VAL(index), jaiListGet(items, i)};
        Value tuple = OBJ_VAL(jaiTupleNew(pair, 2));
        jaiGCPushRoot(tuple);
        jaiListPush(result, tuple);
        jaiGCPopRoot();
    }
    jaiGCPopRoots(2);
    if (!ok) return false;
    *out = OBJ_VAL(result);
    return true;
}

static ObjList *collectAll(int argc, Value *args, int from, int *outShortest) {
    ObjList *holder = jaiListNew(argc - from);
    jaiGCPushRoot(OBJ_VAL(holder));

    int shortest = -1;
    for (int i = from; i < argc; i++) {
        ObjList *one = collectIterable(args[i]);
        if (one == NULL) { jaiGCPopRoot(); return NULL; }
        jaiGCPushRoot(OBJ_VAL(one));
        jaiListPush(holder, OBJ_VAL(one));
        jaiGCPopRoot();
        if (shortest < 0 || one->count < shortest) shortest = one->count;
    }
    jaiGCPopRoot();

    *outShortest = shortest < 0 ? 0 : shortest;
    return holder;
}

static bool nZip(int argc, Value *args, Value *out) {
    int shortest = 0;
    ObjList *holder = collectAll(argc, args, 0, &shortest);
    if (holder == NULL) return false;
    jaiGCPushRoot(OBJ_VAL(holder));

    int width = holder->count;
    ObjList *result = jaiListNew(shortest);
    jaiGCPushRoot(OBJ_VAL(result));

    Value *slot = width > 0 ? JAI_ALLOC(Value, width) : NULL;
    for (int i = 0; i < shortest; i++) {
        for (int c = 0; c < width; c++)
            slot[c] = jaiListGet(AS_LIST(jaiListGet(holder, c)), i);
        Value tuple = OBJ_VAL(jaiTupleNew(slot, width));
        jaiGCPushRoot(tuple);
        jaiListPush(result, tuple);
        jaiGCPopRoot();
    }
    if (slot != NULL) JAI_FREE_ARRAY(Value, slot, width);

    jaiGCPopRoots(2);
    *out = OBJ_VAL(result);
    return true;
}

static bool nMap(int argc, Value *args, Value *out) {
    Value fn = args[0];
    if (!jaiArgCallable(fn, 1, "map")) return false;

    int shortest = 0;
    jaiGCPushRoot(fn);
    ObjList *holder = collectAll(argc, args, 1, &shortest);
    if (holder == NULL) { jaiGCPopRoot(); return false; }
    jaiGCPushRoot(OBJ_VAL(holder));

    int width = holder->count;
    ObjList *result = jaiListNew(shortest);
    jaiGCPushRoot(OBJ_VAL(result));

    Value *callArgs = width > 0 ? JAI_ALLOC(Value, width) : NULL;
    bool ok = true;
    for (int i = 0; i < shortest; i++) {
        for (int c = 0; c < width; c++)
            callArgs[c] = jaiListGet(AS_LIST(jaiListGet(holder, c)), i);
        Value mapped;
        if (!jaiCallValue(fn, width, callArgs, &mapped)) { ok = false; break; }
        jaiGCPushRoot(mapped);
        jaiListPush(result, mapped);
        jaiGCPopRoot();
    }
    if (callArgs != NULL) JAI_FREE_ARRAY(Value, callArgs, width);

    jaiGCPopRoots(3);
    if (!ok) return false;
    *out = OBJ_VAL(result);
    return true;
}

static bool nFilter(int argc, Value *args, Value *out) {
    (void)argc;
    Value fn = args[0];
    if (!jaiArgCallable(fn, 1, "filter")) return false;

    jaiGCPushRoot(fn);
    ObjList *items = collectIterable(args[1]);
    if (items == NULL) { jaiGCPopRoot(); return false; }
    jaiGCPushRoot(OBJ_VAL(items));

    ObjList *result = jaiListNew(0);
    jaiGCPushRoot(OBJ_VAL(result));

    bool ok = true;
    for (int i = 0; i < items->count; i++) {
        Value item = jaiListGet(items, i);
        Value verdict;
        if (!jaiCallFn1(fn, item, &verdict)) { ok = false; break; }
        if (!IS_BOOL(verdict)) {
            ok = jaiThrow(vm.cTypeError, "filter(): predicate must return bool, not %s",
                          jaiTypeNameStatic(verdict));
            break;
        }
        if (AS_BOOL(verdict)) {
            jaiGCPushRoot(item);
            jaiListPush(result, item);
            jaiGCPopRoot();
        }
    }
    jaiGCPopRoots(3);
    if (!ok) return false;
    *out = OBJ_VAL(result);
    return true;
}

static bool quantify(int argc, Value *args, Value *out, bool wantAny) {
    (void)argc;
    const char *fnName = wantAny ? "any" : "all";
    Value iterVal;
    if (!jaiGetIter(args[0], &iterVal)) return false;
    if (!IS_ITER(iterVal))
        return jaiThrow(vm.cTypeError, "'%s' object is not iterable",
                        jaiTypeNameStatic(args[0]));

    jaiGCPushRoot(iterVal);
    bool result = !wantAny;
    bool ok = true;
    Value item;
    while (jaiIterNext(AS_ITER(iterVal), &item)) {
        if (!IS_BOOL(item)) {
            ok = jaiBuiltinArgTypeError(1, fnName, "an iterable of bool", item);
            break;
        }
        if (AS_BOOL(item) == wantAny) { result = wantAny; break; }
    }
    jaiGCPopRoot();
    if (!ok || vm.hasException) return false;
    *out = BOOL_VAL(result);
    return true;
}

static bool nAny(int argc, Value *args, Value *out) { return quantify(argc, args, out, true); }
static bool nAll(int argc, Value *args, Value *out) { return quantify(argc, args, out, false); }

static bool nChr(int argc, Value *args, Value *out) {
    (void)argc;
    int64_t cp;
    if (!jaiArgInt(args[0], 1, "chr", &cp)) return false;
    if (cp < 0 || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF))
        return jaiThrow(vm.cValueError, "chr() argument out of range: %lld",
                        (long long)cp);

    char buf[4];
    int n = jaiUtf8Encode((int32_t)cp, buf);
    if (n == 0)
        return jaiThrow(vm.cValueError, "chr() argument is not a scalar value: %lld",
                        (long long)cp);
    *out = OBJ_VAL(jaiStringIntern(buf, (size_t)n));
    return true;
}

static bool nOrd(int argc, Value *args, Value *out) {
    (void)argc;
    ObjString *s;
    if (!jaiArgString(args[0], 1, "ord", &s)) return false;
    if (s->length == 0)
        return jaiThrow(vm.cValueError, "ord() expected one character, got \"\"");

    int len = 1;
    int32_t cp = jaiUtf8Decode(s->chars, s->chars + s->length, &len);
    if (cp < 0) return jaiThrow(vm.cValueError, "ord(): invalid UTF-8");
    if ((uint32_t)len != s->length)
        return jaiThrow(vm.cValueError,
                        "ord() expected one character, got a string of length %u",
                        jaiStringScalarCount(s));
    *out = INT_VAL(cp);
    return true;
}

bool jaiBuiltinMatchesType(Value v, Value t, bool *matched) {
    if (IS_CLASS(t)) {
        *matched = IS_INSTANCE(v) && jaiClassIsSubclassOf(AS_INSTANCE(v)->klass, AS_CLASS(t));
        return true;
    }
    if (IS_TRAIT(t)) {
        *matched = IS_INSTANCE(v) && jaiClassImplements(AS_INSTANCE(v)->klass, AS_TRAIT(t));
        return true;
    }
    if (IS_ENUM(t)) {
        *matched = IS_ENUM_VAL(v) && AS_ENUM_VAL(v)->type == AS_ENUM(t);
        return true;
    }
    if (IS_NATIVE(t) || IS_STRING(t)) {
        const char *want = IS_NATIVE(t) ? AS_NATIVE(t)->name->chars : AS_CSTRING(t);
        *matched = strcmp(jaiTypeNameStatic(v), want) == 0;
        if (!*matched && strcmp(want, "any") == 0) *matched = true;
        return true;
    }
    if (IS_LIST(t) || IS_TUPLE(t)) {
        int count = IS_LIST(t) ? AS_LIST(t)->count : (int)AS_TUPLE(t)->count;
        const Value *entries = IS_LIST(t) ? jaiListBox(AS_LIST(t))
                                          : AS_TUPLE(t)->items;
        for (int i = 0; i < count; i++) {
            if (!jaiBuiltinMatchesType(v, entries[i], matched)) return false;
            if (*matched) return true;
        }
        *matched = false;
        return true;
    }
    return jaiThrow(vm.cTypeError,
                    "isinstance() argument 2 must be a class, trait or type name, got %s",
                    jaiTypeNameStatic(t));
}

static bool nIsInstance(int argc, Value *args, Value *out) {
    (void)argc;
    bool matched = false;
    if (!jaiBuiltinMatchesType(args[0], args[1], &matched)) return false;
    *out = BOOL_VAL(matched);
    return true;
}

static bool nCallable(int argc, Value *args, Value *out) {
    (void)argc;
    *out = BOOL_VAL(jaiBuiltinIsCallable(args[0]));
    return true;
}

static void collectTableKeys(const JaiTable *table, ObjList *into) {
    int slot = 0;
    Value key, ignored;
    while (jaiTableNext(table, &slot, &key, &ignored)) {
        if (!IS_STRING(key)) continue;
        jaiGCPushRoot(key);
        jaiListPush(into, key);
        jaiGCPopRoot();
    }
}

static ObjList *memberNames(Value v) {
    if (IS_MODULE(v)) {
        ObjList *names = jaiListNew(0);
        jaiGCPushRoot(OBJ_VAL(names));
        collectTableKeys(&AS_MODULE(v)->globals, names);
        jaiGCPopRoot();
        return names;
    }
    if (IS_CLASS(v)) {
        ObjClass *k = AS_CLASS(v);
        ObjList *names = jaiListNew(0);
        jaiGCPushRoot(OBJ_VAL(names));
        collectTableKeys(&k->methods, names);
        collectTableKeys(&k->statics, names);
        collectTableKeys(&k->getters, names);
        collectTableKeys(&k->setters, names);
        jaiGCPopRoot();
        return names;
    }
    if (IS_INSTANCE(v)) {
        ObjClass *k = AS_INSTANCE(v)->klass;
        ObjList *names = jaiListNew(0);
        if (k == NULL) return names;
        jaiGCPushRoot(OBJ_VAL(names));
        collectTableKeys(&k->methods, names);
        collectTableKeys(&k->getters, names);
        collectTableKeys(&k->setters, names);
        for (uint16_t i = 0; i < k->fieldCount; i++) {
            ObjString *fieldName = k->fields[i].name;
            if (fieldName == NULL) continue;
            jaiGCPushRoot(OBJ_VAL(fieldName));
            jaiListPush(names, OBJ_VAL(fieldName));
            jaiGCPopRoot();
        }
        jaiGCPopRoot();
        return names;
    }
    if (IS_ENUM(v)) {
        ObjEnum *e = AS_ENUM(v);
        ObjList *names = jaiListNew(0);
        jaiGCPushRoot(OBJ_VAL(names));
        for (uint16_t i = 0; i < e->variantCount; i++) {
            ObjString *variant = e->variants[i].name;
            if (variant == NULL) continue;
            jaiGCPushRoot(OBJ_VAL(variant));
            jaiListPush(names, OBJ_VAL(variant));
            jaiGCPopRoot();
        }
        collectTableKeys(&e->methods, names);
        jaiGCPopRoot();
        return names;
    }
    return jaiBuiltinMethodNames(v);
}

static bool nDir(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *names = memberNames(args[0]);
    if (names == NULL || vm.hasException) return false;
    jaiGCPushRoot(OBJ_VAL(names));

    ObjList *sorted = sortedCopy(names, names, false, "dir");
    jaiGCPopRoot();
    if (sorted == NULL) return false;

    jaiGCPushRoot(OBJ_VAL(sorted));
    ObjList *unique = jaiListNew(sorted->count);
    jaiGCPushRoot(OBJ_VAL(unique));
    for (int i = 0; i < sorted->count; i++) {
        if (i > 0 && IS_STRING(jaiListGet(sorted, i)) && IS_STRING(jaiListGet(sorted, i - 1)) &&
            jaiStringEquals(AS_STRING(jaiListGet(sorted, i)), AS_STRING(jaiListGet(sorted, i - 1))))
            continue;
        jaiListPush(unique, jaiListGet(sorted, i));
    }
    jaiGCPopRoots(2);

    *out = OBJ_VAL(unique);
    return true;
}

static bool nAssertEq(int argc, Value *args, Value *out) {
    bool equal = jaiValuesEqual(args[0], args[1]);
    if (vm.hasException) return false;
    if (equal) {
        *out = NULL_VAL;
        return true;
    }

    ObjString *actual = jaiValueToRepr(args[0]);
    if (actual == NULL) return false;
    jaiGCPushRoot(OBJ_VAL(actual));

    ObjString *expected = jaiValueToRepr(args[1]);
    if (expected == NULL) { jaiGCPopRoot(); return false; }
    jaiGCPushRoot(OBJ_VAL(expected));

    ObjString *message = NULL;
    if (argc >= 3 && !IS_NULL(args[2])) {
        message = jaiValueToStr(args[2]);
        if (message == NULL) { jaiGCPopRoots(2); return false; }
        jaiGCPushRoot(OBJ_VAL(message));
    }

    if (message != NULL) {
        (void)jaiThrow(vm.cAssertionError, "%.*s: expected %.*s, got %.*s",
                       (int)message->length, message->chars,
                       (int)expected->length, expected->chars,
                       (int)actual->length, actual->chars);
    } else {
        (void)jaiThrow(vm.cAssertionError, "assert_eq failed: expected %.*s, got %.*s",
                       (int)expected->length, expected->chars,
                       (int)actual->length, actual->chars);
    }
    jaiGCPopRoots(message != NULL ? 3 : 2);
    return false;
}

static bool nExit(int argc, Value *args, Value *out) {
    (void)out;
    int64_t code = 0;
    if (argc >= 1 && !IS_NULL(args[0]) && !jaiArgInt(args[0], 1, "exit", &code))
        return false;
    fflush(stdout);
    fflush(stderr);
    exit((int)(code & 0xFF));
}

/* ------------------------------------------------------------------ */
/* Container constructors                                               */
/* ------------------------------------------------------------------ */

static bool nListConv(int argc, Value *args, Value *out) {
    if (argc == 0 || IS_NULL(args[0])) {
        *out = OBJ_VAL(jaiListNew(0));
        return true;
    }
    ObjList *items = collectIterable(args[0]);
    if (items == NULL) return false;
    *out = OBJ_VAL(items);
    return true;
}

static bool nTupleConv(int argc, Value *args, Value *out) {
    if (argc == 0 || IS_NULL(args[0])) {
        *out = OBJ_VAL(jaiTupleNew(NULL, 0));
        return true;
    }
    if (IS_TUPLE(args[0])) {
        *out = args[0];
        return true;
    }
    ObjList *items = collectIterable(args[0]);
    if (items == NULL) return false;
    jaiGCPushRoot(OBJ_VAL(items));
    ObjTuple *tuple = jaiTupleNew(jaiListBox(items), items->count);
    jaiGCPopRoot();
    *out = OBJ_VAL(tuple);
    return true;
}

static bool nSetConv(int argc, Value *args, Value *out) {
    ObjSet *set = jaiSetNew();
    if (argc == 0 || IS_NULL(args[0])) {
        *out = OBJ_VAL(set);
        return true;
    }
    jaiGCPushRoot(OBJ_VAL(set));
    ObjList *items = collectIterable(args[0]);
    if (items == NULL) { jaiGCPopRoot(); return false; }
    jaiGCPushRoot(OBJ_VAL(items));

    bool ok = true;
    for (int i = 0; i < items->count; i++) {
        if (!requireHashable(jaiListGet(items, i), "set")) { ok = false; break; }
        (void)jaiSetAdd(set, jaiListGet(items, i));
    }
    jaiGCPopRoots(2);
    if (!ok) return false;
    *out = OBJ_VAL(set);
    return true;
}

static bool dictFromPairs(ObjDict *dict, ObjList *pairs) {
    for (int i = 0; i < pairs->count; i++) {
        Value entry = jaiListGet(pairs, i);
        Value key, value;
        if (IS_TUPLE(entry) && AS_TUPLE(entry)->count == 2) {
            key = AS_TUPLE(entry)->items[0];
            value = AS_TUPLE(entry)->items[1];
        } else if (IS_LIST(entry) && AS_LIST(entry)->count == 2) {
            key = jaiListGet(AS_LIST(entry), 0);
            value = jaiListGet(AS_LIST(entry), 1);
        } else {
            return jaiThrow(vm.cValueError,
                            "dict() expects (key, value) pairs, got %s at index %d",
                            jaiTypeNameStatic(entry), i);
        }
        if (!requireHashable(key, "dict")) return false;
        (void)jaiDictSet(dict, key, value);
    }
    return true;
}

static bool nDictConv(int argc, Value *args, Value *out) {
    ObjDict *dict = jaiDictNew();
    if (argc == 0 || IS_NULL(args[0])) {
        *out = OBJ_VAL(dict);
        return true;
    }
    jaiGCPushRoot(OBJ_VAL(dict));

    bool ok = true;
    if (IS_DICT(args[0])) {
        int slot = 0;
        Value key, value;
        while (jaiTableNext(&AS_DICT(args[0])->table, &slot, &key, &value))
            (void)jaiDictSet(dict, key, value);
    } else {
        ObjList *pairs = collectIterable(args[0]);
        if (pairs == NULL) {
            ok = false;
        } else {
            jaiGCPushRoot(OBJ_VAL(pairs));
            ok = dictFromPairs(dict, pairs);
            jaiGCPopRoot();
        }
    }
    jaiGCPopRoot();
    if (!ok) return false;
    *out = OBJ_VAL(dict);
    return true;
}

static bool nBytesConv(int argc, Value *args, Value *out) {
    if (argc == 0 || IS_NULL(args[0])) {
        *out = OBJ_VAL(jaiBytesNew(NULL, 0));
        return true;
    }
    Value v = args[0];
    if (IS_BYTES(v)) { *out = v; return true; }
    if (IS_STRING(v)) {
        ObjString *s = AS_STRING(v);
        *out = OBJ_VAL(jaiBytesNew((const uint8_t *)s->chars, s->length));
        return true;
    }

    ObjList *items = collectIterable(v);
    if (items == NULL) return false;
    jaiGCPushRoot(OBJ_VAL(items));

    uint8_t stackBuf[64];
    uint8_t *buf = stackBuf;
    size_t n = (size_t)items->count;
    bool heap = false;
    if (n > sizeof stackBuf) {
        buf = JAI_ALLOC(uint8_t, n);
        heap = true;
    }

    bool ok = true;
    for (int i = 0; i < items->count; i++) {
        Value item = jaiListGet(items, i);
        if (!IS_INT(item)) {
            ok = jaiThrow(vm.cTypeError, "bytes(): expected int elements, got %s",
                          jaiTypeNameStatic(item));
            break;
        }
        if (AS_INT(item) < 0 || AS_INT(item) > 255) {
            ok = jaiThrow(vm.cValueError, "bytes(): value %lld is out of range 0..255",
                          (long long)AS_INT(item));
            break;
        }
        buf[i] = (uint8_t)AS_INT(item);
    }

    ObjBytes *bytes = ok ? jaiBytesNew(buf, n) : NULL;
    if (heap) JAI_FREE_ARRAY(uint8_t, buf, n);
    jaiGCPopRoot();
    if (!ok) return false;
    *out = OBJ_VAL(bytes);
    return true;
}

/* ------------------------------------------------------------------ */
/* ------------------------------------------------------------------ */
/* ------------------------------------------------------------------ */
/* Registration                                                         */
/* ------------------------------------------------------------------ */

void jaiRegisterCoreBuiltins(void) {
    jaiDefineNative("print",      nPrint,      0, -1);
    jaiDefineNative("input",      nInput,      0,  1);
    jaiDefineNative("len",        nLen,        1,  1);
    jaiDefineNative("range",      nRange,      1,  3);
    jaiDefineNative("type_of",    nType,       1,  1);
    jaiDefineNative("repr",       nRepr,       1,  1);
    jaiDefineNative("str",        nStr,        1,  1);
    jaiDefineNative("int",        nIntConv,    1,  2);
    jaiDefineNative("float",      nFloatConv,  1,  1);
    jaiDefineNative("bool",       nBoolConv,   1,  1);
    jaiDefineNative("id",         nId,         1,  1);
    jaiDefineNative("hash",       nHash,       1,  1);
    jaiDefineNative("abs",        nAbs,        1,  1);
    jaiDefineNative("min",        nMin,        1, -1);
    jaiDefineNative("max",        nMax,        1, -1);
    jaiDefineNative("sum",        nSum,        1,  2);
    jaiDefineNative("sorted",     nSorted,     1,  3);
    jaiDefineNative("reversed",   nReversed,   1,  1);
    jaiDefineNative("enumerate",  nEnumerate,  1,  2);
    jaiDefineNative("zip",        nZip,        1, -1);
    jaiDefineNative("map",        nMap,        2, -1);
    jaiDefineNative("filter",     nFilter,     2,  2);
    jaiDefineNative("any",        nAny,        1,  1);
    jaiDefineNative("all",        nAll,        1,  1);
    jaiDefineNative("chr",        nChr,        1,  1);
    jaiDefineNative("ord",        nOrd,        1,  1);
    jaiDefineNative("isinstance", nIsInstance, 2,  2);

    jaiDefineNative("__prim__.is_instance", nIsInstance, 2, 2);

    jaiDefineNative("callable",   nCallable,   1,  1);
    jaiDefineNative("dir",        nDir,        1,  1);
    jaiDefineNative("assert_eq",  nAssertEq,   2,  3);
    jaiDefineNative("exit",       nExit,       0,  1);

    jaiDefineNative("list",  nListConv,  0, 1);
    jaiDefineNative("dict",  nDictConv,  0, 1);
    jaiDefineNative("set",   nSetConv,   0, 1);
    jaiDefineNative("tuple", nTupleConv, 0, 1);
    jaiDefineNative("bytes", nBytesConv, 0, 1);

    jaiRegisterOperatorPrimitives();
}
