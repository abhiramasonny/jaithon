/* builtins_convert.c — the type constructors: int, float, bool, list, tuple,
 * set, dict and bytes. */

#include <ctype.h>
#include <math.h>
#include <stdlib.h>

#include "runtime/builtins/builtins_core.h"

#include "vm/gc.h"

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

bool nIntConv(int argc, Value *args, Value *out) {
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

bool nFloatConv(int argc, Value *args, Value *out) {
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

bool nBoolConv(int argc, Value *args, Value *out) {
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

/* ------------------------------------------------------------------ */
/* Container constructors                                               */
/* ------------------------------------------------------------------ */

bool nListConv(int argc, Value *args, Value *out) {
    if (argc == 0 || IS_NULL(args[0])) {
        *out = OBJ_VAL(jaiListNew(0));
        return true;
    }
    ObjList *items = collectIterable(args[0]);
    if (items == NULL) return false;
    *out = OBJ_VAL(items);
    return true;
}

bool nTupleConv(int argc, Value *args, Value *out) {
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

bool nSetConv(int argc, Value *args, Value *out) {
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

bool nDictConv(int argc, Value *args, Value *out) {
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

bool nBytesConv(int argc, Value *args, Value *out) {
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
