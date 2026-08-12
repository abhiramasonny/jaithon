/* builtins_str_convert.c — conversions between str and every other
 * representation: chars(), bytes(), code_points(), to_bytes(), the
 * parse_int/parse_float/to_int/to_float/to_str family, and the four
 * __prim__ functions (str_from_codepoint, str_to_codepoint, str_encode,
 * str_decode) that back them at the operator level.
 *
 * str_encode and to_bytes() do the identical thing — one is the method
 * spelling, one is the primitive the compiler emits for it — which is the
 * clearest sign these two surfaces belong in one file: every conversion a
 * program can ask for, whichever spelling it uses, is here.
 *
 * builtins_str.c keeps the method table, the rest of the __prim__ surface,
 * and the plumbing every method file calls through builtins_str_methods.h.
 */

#include "builtins_str_methods.h"

#include "../vm/gc.h"

#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* chars / bytes / code_points / to_bytes                               */
/* ------------------------------------------------------------------ */

bool strChars(int argc, Value *args, Value *out) {
    ObjString *s;
    if (!strReceiver(argc, args, "chars", &s)) return false;

    ObjList *list = jaiListNew(jaiStrCapacityFor(jaiStringScalarCount(s)));
    jaiGCPushRoot(OBJ_VAL(list));
    bool ok = true;
    const char *p = s->chars;
    const char *end = p + s->length;
    while (ok && p < end) {
        int len = 1;
        if ((unsigned char)*p >= 0x80u)
            (void)jaiUtf8Decode(p, end, &len);
        ok = pushSlice(list, p, (size_t)len);
        p += len;
    }
    jaiGCPopRoot();
    if (!ok) return false;
    *out = OBJ_VAL(list);
    return true;
}

bool strBytes(int argc, Value *args, Value *out) {
    ObjString *s;
    if (!strReceiver(argc, args, "bytes", &s)) return false;

    ObjList *list = jaiListNew(jaiStrCapacityFor(s->length));
    jaiGCPushRoot(OBJ_VAL(list));
    for (uint32_t i = 0; i < s->length; i++) {
        jaiListPush(list, INT_VAL((unsigned char)s->chars[i]));
    }
    jaiGCPopRoot();
    *out = OBJ_VAL(list);
    return true;
}

bool strCodePoints(int argc, Value *args, Value *out) {
    ObjString *s;
    if (!strReceiver(argc, args, "code_points", &s)) return false;

    ObjList *list = jaiListNew(jaiStrCapacityFor(jaiStringScalarCount(s)));
    jaiGCPushRoot(OBJ_VAL(list));
    const char *p = s->chars;
    const char *end = p + s->length;
    while (p < end) {
        int len = 1;
        int32_t cp;
        if ((unsigned char)*p < 0x80u) {
            cp = (unsigned char)*p;
        } else {
            cp = jaiUtf8Decode(p, end, &len);
        }
        /* An invalid byte reports as its negated value so that the list stays
         * the same length as chars() and the damage stays locatable. */
        jaiListPush(list, INT_VAL(cp));
        p += len;
    }
    jaiGCPopRoot();
    *out = OBJ_VAL(list);
    return true;
}

bool strToBytes(int argc, Value *args, Value *out) {
    ObjString *s;
    if (!strReceiver(argc, args, "to_bytes", &s)) return false;
    ObjBytes *b = jaiBytesNew((const uint8_t *)s->chars, s->length);
    if (b == NULL) return false;
    *out = OBJ_VAL(b);
    return true;
}

/* ------------------------------------------------------------------ */
/* Number parsing                                                       */
/* ------------------------------------------------------------------ */

typedef enum { PARSE_OK, PARSE_MALFORMED, PARSE_RANGE } ParseStatus;

static inline bool isAsciiSpace(char c) {
    const unsigned char u = (unsigned char)c;
    return u == ' ' || (u >= 0x09u && u <= 0x0Du);
}

int jaiStrDigitValue(char c) {
    const unsigned char u = (unsigned char)c;

    if ((unsigned)(u - '0') <= 9u)
        return (int)(u - '0');

    if ((unsigned)(u - 'a') <= 25u)
        return (int)(u - 'a') + 10;

    if ((unsigned)(u - 'A') <= 25u)
        return (int)(u - 'A') + 10;

    return -1;
}

/* Integer syntax of spec §2.3: an optional sign, an optional base prefix, and
 * digits that may be separated by single underscores. */
static ParseStatus parseIntText(const char *s, size_t len, int base,
                                int64_t *out) {
    size_t i = 0, end = len;

    while (i < end && isAsciiSpace(s[i])) ++i;
    while (end > i && isAsciiSpace(s[end - 1])) --end;
    if (i >= end) return PARSE_MALFORMED;

    bool negative = false;
    if (s[i] == '+' || s[i] == '-') {
        negative = s[i] == '-';
        ++i;
    }

    if (i + 1 < end && s[i] == '0') {
        const char marker = s[i + 1];
        const int implied =
            (marker == 'x' || marker == 'X') ? 16 :
            (marker == 'o' || marker == 'O') ? 8 :
            (marker == 'b' || marker == 'B') ? 2 : 0;

        if (implied != 0 && (base == 0 || base == implied)) {
            base = implied;
            i += 2;
        }
    }

    if (base == 0) base = 10;

    const uint64_t limit =
        negative ? (uint64_t)INT64_MAX + 1u : (uint64_t)INT64_MAX;
    const uint64_t radix = (uint64_t)base;

    /* One division/modulo per parse instead of one division per digit. */
    const uint64_t cutoff = limit / radix;
    const uint64_t cutlim = limit % radix;

    uint64_t value = 0;
    bool sawDigit = false;
    bool lastWasUnderscore = false;

    for (; i < end; ++i) {
        const char c = s[i];

        if (c == '_') {
            if (!sawDigit || lastWasUnderscore)
                return PARSE_MALFORMED;

            lastWasUnderscore = true;
            continue;
        }

        const int digit = jaiStrDigitValue(c);
        if (digit < 0 || digit >= base)
            return PARSE_MALFORMED;

        const uint64_t uDigit = (uint64_t)digit;
        if (value > cutoff || (value == cutoff && uDigit > cutlim))
            return PARSE_RANGE;

        value = value * radix + uDigit;
        sawDigit = true;
        lastWasUnderscore = false;
    }

    if (!sawDigit || lastWasUnderscore)
        return PARSE_MALFORMED;

    *out = negative ? (int64_t)(~value + 1u) : (int64_t)value;
    return PARSE_OK;
}

/* strtod over a copy with the underscores removed; the copy also guarantees
 * the NUL that strtod needs. */
static ParseStatus parseFloatText(const char *s, size_t len, double *out) {
    size_t i = 0, end = len;

    while (i < end && isAsciiSpace(s[i])) ++i;
    while (end > i && isAsciiSpace(s[end - 1])) --end;
    if (i >= end) return PARSE_MALFORMED;

    const char *const start = s + i;
    const char *const finish = s + end;
    const size_t span = end - i;

    /* ObjString text is NUL-terminated. The overwhelmingly common case has no
     * underscores, so let strtod read it directly without making a copy. */
    if (memchr(start, '_', span) == NULL) {
        char *stop = NULL;
        const double value = strtod(start, &stop);

        if (stop != finish)
            return PARSE_MALFORMED;

        *out = value;
        return PARSE_OK;
    }

    char stackBuf[64];
    char *text =
        span + 1 <= sizeof stackBuf ? stackBuf : JAI_ALLOC(char, span + 1);

    size_t w = 0;
    for (const char *p = start; p < finish; ++p) {
        if (*p != '_')
            text[w++] = *p;
    }
    text[w] = '\0';

    char *stop = NULL;
    const double value = strtod(text, &stop);
    const ParseStatus status =
        (w == 0 || stop != text + w) ? PARSE_MALFORMED : PARSE_OK;

    if (text != stackBuf)
        JAI_FREE_ARRAY(char, text, span + 1);

    if (status == PARSE_OK)
        *out = value;

    return status;
}

bool strParseInt(int argc, Value *args, Value *out) {
    ObjString *s;
    if (!strReceiver(argc, args, "parse_int", &s)) return false;
    int64_t base;
    if (!jaiStrOptInt(argc, args, 1, "parse_int", "the base", 10, &base)) return false;
    if (base != 0 && (base < 2 || base > 36)) {
        return jaiThrow(vm.cValueError,
                        "str.parse_int(): the base must be 0 or 2 to 36, got %lld",
                        (long long)base);
    }

    int64_t value = 0;
    switch (parseIntText(s->chars, s->length, (int)base, &value)) {
    case PARSE_OK:
        *out = INT_VAL(value);
        return true;
    case PARSE_RANGE:
        return jaiThrow(vm.cOverflowError,
                        "str.parse_int(): \"%.*s\" does not fit in an int",
                        (int)(s->length > 60 ? 60 : s->length), s->chars);
    default:
        return jaiThrow(vm.cValueError,
                        "str.parse_int(): \"%.*s\" is not an integer",
                        (int)(s->length > 60 ? 60 : s->length), s->chars);
    }
}

bool strParseFloat(int argc, Value *args, Value *out) {
    ObjString *s;
    if (!strReceiver(argc, args, "parse_float", &s)) return false;
    double value = 0.0;
    if (parseFloatText(s->chars, s->length, &value) != PARSE_OK) {
        return jaiThrow(vm.cValueError,
                        "str.parse_float(): \"%.*s\" is not a number",
                        (int)(s->length > 60 ? 60 : s->length), s->chars);
    }
    *out = FLOAT_VAL(value);
    return true;
}

/* The forgiving pair: null instead of an exception, for `text.to_int() ?? 0`. */
bool strToInt(int argc, Value *args, Value *out) {
    ObjString *s;
    if (!strReceiver(argc, args, "to_int", &s)) return false;
    int64_t base;
    if (!jaiStrOptInt(argc, args, 1, "to_int", "the base", 10, &base)) return false;
    if (base != 0 && (base < 2 || base > 36)) {
        return jaiThrow(vm.cValueError,
                        "str.to_int(): the base must be 0 or 2 to 36, got %lld",
                        (long long)base);
    }
    int64_t value = 0;
    *out = (parseIntText(s->chars, s->length, (int)base, &value) == PARSE_OK)
               ? INT_VAL(value) : NULL_VAL;
    return true;
}

bool strToFloat(int argc, Value *args, Value *out) {
    ObjString *s;
    if (!strReceiver(argc, args, "to_float", &s)) return false;
    double value = 0.0;
    *out = (parseFloatText(s->chars, s->length, &value) == PARSE_OK)
               ? FLOAT_VAL(value) : NULL_VAL;
    return true;
}

bool strToStr(int argc, Value *args, Value *out) {
    ObjString *s;
    if (!strReceiver(argc, args, "to_str", &s)) return false;
    *out = OBJ_VAL(s);
    return true;
}

/* ------------------------------------------------------------------ */
/* __prim__: codepoint and bytes conversions                            */
/* ------------------------------------------------------------------ */

bool primStrFromCodepoint(int argc, Value *args, Value *out) {
    int64_t cp;
    if (!jaiArgInt(args[0], 0, "str_from_codepoint", &cp)) return false;
    char utf8[4];
    int len = (cp >= 0 && cp <= 0x10FFFF) ? jaiUtf8Encode((int32_t)cp, utf8) : 0;
    if (len == 0) {
        return jaiThrow(vm.cValueError,
                        "str_from_codepoint(): %lld is not a Unicode scalar value",
                        (long long)cp);
    }
    ObjString *s = jaiStringNew(utf8, (size_t)len);
    if (s == NULL) return false;
    *out = OBJ_VAL(s);
    return true;
}

bool primStrToCodepoint(int argc, Value *args, Value *out) {
    (void)argc;

    ObjString *s;
    if (!jaiArgString(args[0], 0, "str_to_codepoint", &s)) return false;

    if (s->length == 1 && (unsigned char)s->chars[0] < 0x80u) {
        *out = INT_VAL((unsigned char)s->chars[0]);
        return true;
    }

    const uint32_t count = jaiStringScalarCount(s);
    if (count != 1) {
        return jaiThrow(vm.cValueError,
                        "str_to_codepoint(): expected exactly one character, got "
                        "%u",
                        count);
    }

    int len = 1;
    const int32_t cp = jaiUtf8Decode(s->chars, s->chars + s->length, &len);

    if (cp < 0)
        return jaiThrow(vm.cValueError,
                        "str_to_codepoint(): the string is not valid UTF-8");

    *out = INT_VAL(cp);
    return true;
}

bool primStrEncode(int argc, Value *args, Value *out) {
    ObjString *s;
    if (!jaiArgString(args[0], 0, "str_encode", &s)) return false;
    ObjBytes *b = jaiBytesNew((const uint8_t *)s->chars, s->length);
    if (b == NULL) return false;
    *out = OBJ_VAL(b);
    return true;
}

bool primStrDecode(int argc, Value *args, Value *out) {
    if (!IS_BYTES(args[0])) {
        return jaiThrow(vm.cTypeError, "str_decode(): expected bytes, got %s",
                        jaiTypeNameStatic(args[0]));
    }
    ObjBytes *b = AS_BYTES(args[0]);
    if (!jaiUtf8Validate((const char *)b->data, b->length)) {
        return jaiThrow(vm.cValueError, "str_decode(): the data is not valid UTF-8");
    }
    ObjString *s = jaiStringNew((const char *)b->data, b->length);
    if (s == NULL) return false;
    *out = OBJ_VAL(s);
    return true;
}
