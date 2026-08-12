/* builtins_format.c — the format-spec engine of spec §5.2, shared by f-string
 * holes (`value.__format__`, parser.c applyFormatSpec) and `str.format`;
 * std.fmt wraps both in Jaithon. */

#include "builtins_str.h"
#include "methods.h"


#include <math.h>
#include <stdlib.h>

/* ------------------------------------------------------------------ */
/* Format specs (spec §5.2)                                             */
/*                                                                      */
/*   [[fill]align][sign][#][0][width][group][.precision][type]          */
/*                                                                      */
/* ------------------------------------------------------------------ */

typedef struct {
    int32_t fill;
    bool    hasFill;     /* an explicit fill outranks the '0' flag */
    char    align;       /* 0 for "by type" */
    char    sign;
    bool    alt;
    bool    zero;        /* zeros pad between sign and digits, not before the sign */
    int64_t width;
    char    group;
    int64_t precision;   /* -1 when unset */
    char    type;        /* 0 for "by type" */
} FormatSpec;

static bool specError(const char *where, const char *spec, size_t specLen,
                      const char *why) {
    return jaiThrow(vm.cValueError, "%s(): %s in format spec \"%.*s\"", where,
                    why, (int)specLen, spec);
}

static bool parseSpec(const char *spec, size_t len, FormatSpec *fs,
                      const char *where) {
    fs->fill = ' ';
    fs->hasFill = false;
    fs->align = 0;
    fs->sign = '-';
    fs->alt = false;
    fs->zero = false;
    fs->width = 0;
    fs->group = 0;
    fs->precision = -1;
    fs->type = 0;

    size_t i = 0;
    /* The fill scalar is only a fill when an alignment follows it, so that a
     * lone `<` still reads as an alignment rather than as a fill. */
    if (len > 0) {
        int fillLen = 1;
        int32_t cp = jaiUtf8Decode(spec, spec + len, &fillLen);
        if (cp >= 0 && (size_t)fillLen < len) {
            char next = spec[fillLen];
            if (next == '<' || next == '>' || next == '^' || next == '=') {
                fs->fill = cp;
                fs->hasFill = true;
                fs->align = next;
                i = (size_t)fillLen + 1;
            }
        }
        if (fs->align == 0) {
            char first = spec[0];
            if (first == '<' || first == '>' || first == '^' || first == '=') {
                fs->align = first;
                i = 1;
            }
        }
    }

    if (i < len && (spec[i] == '+' || spec[i] == '-' || spec[i] == ' ')) {
        fs->sign = spec[i++];
    }
    if (i < len && spec[i] == '#') { fs->alt = true; i++; }
    if (i < len && spec[i] == '0') { fs->zero = true; i++; }

    while (i < len && spec[i] >= '0' && spec[i] <= '9') {
        /* Beyond the maximum string length the field could never be built. */
        if (fs->width > (int64_t)UINT32_MAX) {
            return specError(where, spec, len, "width is too large");
        }
        fs->width = fs->width * 10 + (spec[i++] - '0');
    }
    if (i < len && (spec[i] == ',' || spec[i] == '_')) fs->group = spec[i++];

    if (i < len && spec[i] == '.') {
        i++;
        if (i >= len || spec[i] < '0' || spec[i] > '9') {
            return specError(where, spec, len, "precision needs at least one digit");
        }
        fs->precision = 0;
        while (i < len && spec[i] >= '0' && spec[i] <= '9') {
            /* A precision beyond this is a typo, not an intent: snprintf would
             * happily try to materialise gigabytes of digits. */
            if (fs->precision > 100000) {
                return specError(where, spec, len, "precision is too large");
            }
            fs->precision = fs->precision * 10 + (spec[i++] - '0');
        }
    }

    if (i < len) {
        switch (spec[i]) {
        case 's': case 'r': case 'd': case 'b': case 'o': case 'x': case 'X':
        case 'c': case 'e': case 'E': case 'f': case 'F': case 'g': case 'G':
        case '%':
            fs->type = spec[i++];
            break;
        default:
            return specError(where, spec, len, "unknown presentation type");
        }
    }
    if (i != len) return specError(where, spec, len, "trailing characters");
    return true;
}

static bool typeIsInteger(char t) {
    return t == 'd' || t == 'b' || t == 'o' || t == 'x' || t == 'X' || t == 'c';
}

static bool typeIsFloat(char t) {
    return t == 'e' || t == 'E' || t == 'f' || t == 'F' || t == 'g' ||
           t == 'G' || t == '%';
}

/* Groups `head` characters of `digits` from the right, every `groupSize`
 * chars; caller sets `head` since only it knows whether a trailing 'e' is a
 * hex digit or the start of an exponent. */
static void appendGrouped(JaiBuf *out, const char *digits, size_t n,
                          size_t head, char sep, int groupSize) {
    if (sep == 0 || groupSize <= 0 || head == 0) {
        jaiBufAppend(out, digits, n);
        return;
    }
    if (head > n) head = n;

    size_t first = head % (size_t)groupSize;
    if (first == 0) first = (size_t)groupSize;
    size_t i = 0;
    while (i < head) {
        size_t take = (i == 0) ? first : (size_t)groupSize;
        if (take > head - i) take = head - i;
        if (i > 0) jaiBufPush(out, (uint8_t)sep);
        jaiBufAppend(out, digits + i, take);
        i += take;
    }
    jaiBufAppend(out, digits + head, n - head);
}

/* Writes sign, base prefix and grouped magnitude into `body`; *headLen counts
 * the sign and prefix, which is where '=' alignment inserts its padding. */
static void assembleNumber(JaiBuf *body, size_t *headLen, bool negative,
                           const char *prefix, const char *digits,
                           size_t digitLen, size_t groupable,
                           const FormatSpec *fs, int groupSize) {
    if (negative)            jaiBufPush(body, '-');
    else if (fs->sign == '+') jaiBufPush(body, '+');
    else if (fs->sign == ' ') jaiBufPush(body, ' ');
    if (prefix != NULL) jaiBufAppendStr(body, prefix);
    *headLen = body->count;
    appendGrouped(body, digits, digitLen, groupable, fs->group, groupSize);
}

/* The integer part of a rendered decimal number: everything before the point,
 * the exponent or a trailing '%'. */
static size_t decimalRun(const char *s, size_t n) {
    size_t i = 0;
    while (i < n && s[i] >= '0' && s[i] <= '9') i++;
    return i;
}

static void appendRadix(JaiBuf *out, uint64_t magnitude, int base, bool upper) {
    static const char *kLower = "0123456789abcdefghijklmnopqrstuvwxyz";
    static const char *kUpper = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    const char *alphabet = upper ? kUpper : kLower;

    char tmp[72];
    int n = 0;
    do {
        tmp[n++] = alphabet[magnitude % (uint64_t)base];
        magnitude /= (uint64_t)base;
    } while (magnitude != 0);
    while (n > 0) jaiBufPush(out, (uint8_t)tmp[--n]);
}

/* Magnitude of an int64 without overflowing on INT64_MIN. */
static uint64_t magnitudeOf(int64_t v) {
    return v < 0 ? (uint64_t)(-(v + 1)) + 1u : (uint64_t)v;
}

static bool renderInteger(JaiBuf *body, size_t *headLen, int64_t value,
                          const FormatSpec *fs, const char *where,
                          const char *spec, size_t specLen) {
    char type = fs->type ? fs->type : 'd';

    if (type == 'c') {
        if (fs->sign == '+' || fs->sign == ' ' || fs->group != 0) {
            return specError(where, spec, specLen,
                             "sign and grouping are not allowed with type 'c'");
        }
        char utf8[4];
        int n = (value >= 0 && value <= 0x10FFFF)
                    ? jaiUtf8Encode((int32_t)value, utf8) : 0;
        if (n == 0) {
            return jaiThrow(vm.cValueError,
                            "%s(): %lld is not a Unicode scalar value", where,
                            (long long)value);
        }
        *headLen = 0;
        jaiBufAppend(body, utf8, (size_t)n);
        return true;
    }

    int base = 10;
    const char *prefix = NULL;
    bool upper = false;
    switch (type) {
    case 'b': base = 2;  prefix = fs->alt ? "0b" : NULL; break;
    case 'o': base = 8;  prefix = fs->alt ? "0o" : NULL; break;
    case 'x': base = 16; prefix = fs->alt ? "0x" : NULL; break;
    case 'X': base = 16; prefix = fs->alt ? "0X" : NULL; upper = true; break;
    default:  break;
    }

    JaiBuf digits;
    jaiBufInit(&digits);
    appendRadix(&digits, magnitudeOf(value), base, upper);
    assembleNumber(body, headLen, value < 0, prefix, (const char *)digits.data,
                   digits.count, digits.count, fs, base == 10 ? 3 : 4);
    jaiBufFree(&digits);
    return true;
}

/* *zeroPad is cleared for nan/inf, else zero padding would produce "0000-inf". */
static void renderDouble(JaiBuf *body, size_t *headLen, double value,
                         const FormatSpec *fs, bool *zeroPad) {
    char type = fs->type ? fs->type : 'g';
    bool negative = signbit(value) != 0;
    double magnitude = fabs(value);

    if (isnan(magnitude) || isinf(magnitude)) {
        bool upper = (type == 'E' || type == 'F' || type == 'G');
        const char *text = isnan(magnitude) ? (upper ? "NAN" : "nan")
                                            : (upper ? "INF" : "inf");
        *zeroPad = false;
        /* NaN carries no sign in Jaithon's own str(), so neither does this. */
        FormatSpec bare = *fs;
        bare.group = 0;
        assembleNumber(body, headLen, negative && !isnan(magnitude), NULL, text,
                       strlen(text), 0, &bare, 0);
        return;
    }

    int precision = (fs->precision >= 0) ? (int)fs->precision : 6;
    JaiBuf digits;
    jaiBufInit(&digits);
    switch (type) {
    case 'e': jaiBufPrintf(&digits, "%.*e", precision, magnitude); break;
    case 'E': jaiBufPrintf(&digits, "%.*E", precision, magnitude); break;
    case 'f': jaiBufPrintf(&digits, "%.*f", precision, magnitude); break;
    case 'F': jaiBufPrintf(&digits, "%.*F", precision, magnitude); break;
    case 'G': jaiBufPrintf(&digits, "%.*G", precision ? precision : 1, magnitude); break;
    case '%': jaiBufPrintf(&digits, "%.*f%%", precision, magnitude * 100.0); break;
    default:  jaiBufPrintf(&digits, "%.*g", precision ? precision : 1, magnitude); break;
    }
    assembleNumber(body, headLen, negative, NULL, (const char *)digits.data,
                   digits.count, decimalRun((const char *)digits.data, digits.count),
                   fs, 3);
    jaiBufFree(&digits);
}

/* str()/repr() of a value, honouring `.precision` as a maximum length. */
static bool renderText(JaiBuf *body, Value value, const FormatSpec *fs,
                       const char *where, const char *spec, size_t specLen) {
    if (fs->sign != '-' || fs->alt || fs->group != 0) {
        return specError(where, spec, specLen,
                         "sign, '#' and grouping need a numeric type");
    }
    if (fs->align == '=' || (fs->zero && fs->align == 0)) {
        return specError(where, spec, specLen,
                         "'=' alignment needs a numeric type");
    }

    ObjString *text;
    if (fs->type == 'r') {
        text = jaiValueToRepr(value);
    } else if (IS_STRING(value)) {
        text = AS_STRING(value);
    } else {
        text = jaiValueToStr(value);
    }
    if (text == NULL) return false;         /* a user __str__ raised */

    size_t length = text->length;
    if (fs->precision >= 0) {
        size_t limit = jaiStrByteOffsetOf(text, (size_t)fs->precision);
        if (limit < length) length = limit;
    }
    jaiBufAppend(body, text->chars, length);
    return true;
}

/* Pads `body` into `out` to the spec's width, counting scalars. */
static bool padInto(JaiBuf *out, const JaiBuf *body, size_t headLen,
                    const FormatSpec *fs, bool numeric, bool zeroPad,
                    const char *where) {
    size_t scalars = jaiUtf8Length((const char *)body->data, body->count);
    if (fs->width <= 0 || scalars >= (size_t)fs->width) {
        jaiBufAppend(out, body->data, body->count);
        return true;
    }
    size_t pad = (size_t)fs->width - scalars;

    /* '0' is a shorthand for a '0' fill, so an explicit fill overrides it. */
    int32_t fill = (zeroPad && !fs->hasFill) ? '0' : fs->fill;
    char align = fs->align;
    if (align == 0) align = (numeric && zeroPad) ? '=' : (numeric ? '>' : '<');

    char fillBytes[4];
    int fillLen = jaiUtf8Encode(fill, fillBytes);
    if (fillLen <= 0) { fillBytes[0] = ' '; fillLen = 1; }

    if (pad * (size_t)fillLen + body->count > UINT32_MAX) {
        return jaiThrow(vm.cOverflowError,
                        "%s(): a field width of %lld exceeds the maximum string "
                        "length", where, (long long)fs->width);
    }

    size_t left = 0, right = 0;
    switch (align) {
    case '<': right = pad; break;
    case '^': left = pad / 2; right = pad - left; break;
    case '=':
        jaiBufAppend(out, body->data, headLen);
        for (size_t i = 0; i < pad; i++) jaiBufAppend(out, fillBytes, (size_t)fillLen);
        jaiBufAppend(out, body->data + headLen, body->count - headLen);
        return true;
    default:  left = pad; break;            /* '>' */
    }

    for (size_t i = 0; i < left; i++)  jaiBufAppend(out, fillBytes, (size_t)fillLen);
    jaiBufAppend(out, body->data, body->count);
    for (size_t i = 0; i < right; i++) jaiBufAppend(out, fillBytes, (size_t)fillLen);
    return true;
}

/* The whole engine for one value. `where` names the caller in diagnostics. */
bool jaiFormatValue(JaiBuf *out, Value value, const char *spec,
                        size_t specLen, const char *where) {
    FormatSpec fs;
    if (!parseSpec(spec, specLen, &fs, where)) return false;

    /* An int under a float type is promoted; a float under an int type is not
     * demoted, because silently truncating is never what the spec asked for. */
    bool numeric = false;
    if (typeIsInteger(fs.type)) {
        if (!IS_INT(value) && !IS_BOOL(value)) {
            return jaiThrow(vm.cValueError,
                            "%s(): presentation type '%c' needs an int, got %s",
                            where, fs.type, jaiTypeNameStatic(value));
        }
        numeric = true;
    } else if (typeIsFloat(fs.type)) {
        if (!IS_NUMBER(value)) {
            return jaiThrow(vm.cValueError,
                            "%s(): presentation type '%c' needs a number, got %s",
                            where, fs.type, jaiTypeNameStatic(value));
        }
        numeric = true;
    } else if (fs.type == 0) {
        numeric = IS_INT(value) || IS_FLOAT(value);
    }

    JaiBuf body;
    jaiBufInit(&body);
    size_t headLen = 0;
    bool zeroPad = fs.zero;
    bool ok = true;

    if (!numeric) {
        ok = renderText(&body, value, &fs, where, spec, specLen);
    } else if (typeIsFloat(fs.type)) {
        renderDouble(&body, &headLen, jaiAsDouble(value), &fs, &zeroPad);
    } else if (IS_FLOAT(value)) {
        /* Default spec on a float: the shortest round-tripping form, then the
         * ordinary sign, grouping and padding on top of it. */
        ObjString *text = jaiValueToStr(value);
        if (text == NULL) {
            ok = false;
        } else {
            bool negative = text->length > 0 && text->chars[0] == '-';
            const char *digits = text->chars + (negative ? 1 : 0);
            size_t digitLen = text->length - (negative ? 1u : 0u);
            assembleNumber(&body, &headLen, negative, NULL, digits, digitLen,
                           decimalRun(digits, digitLen), &fs, 3);
        }
    } else {
        int64_t i = IS_BOOL(value) ? (AS_BOOL(value) ? 1 : 0) : AS_INT(value);
        ok = renderInteger(&body, &headLen, i, &fs, where, spec, specLen);
    }

    if (ok) ok = padInto(out, &body, headLen, &fs, numeric, zeroPad, where);
    jaiBufFree(&body);
    return ok;
}

/* ------------------------------------------------------------------ */
/* str.format                                                          */
/* ------------------------------------------------------------------ */

/* `{name}` resolves against the dict arguments, searched left to right — a
 * native can't see the caller's bindings, hence no other namespace here. */
static bool namedArgument(Value *args, int argc, const char *name, size_t len,
                          Value *out) {
    ObjString *key = jaiStringIntern(name, len);
    if (key == NULL) return false;
    for (int i = 1; i < argc; i++) {
        if (!IS_DICT(args[i])) continue;
        if (jaiDictGet(AS_DICT(args[i]), OBJ_VAL(key), out)) return true;
    }
    return jaiThrow(vm.cKeyError, "format(): no argument named \"%.*s\"",
                    (int)len, name);
}

bool jaiFormatTemplate(ObjString *tmpl, Value *args, int argc, Value *out,
                      const char *where) {
    JaiBuf result;
    jaiBufInit(&result);

    const char *s = tmpl->chars;
    size_t len = tmpl->length;
    size_t i = 0;
    int autoIndex = 0;
    bool ok = true;

    while (ok && i < len) {
        char c = s[i];
        if (c == '{' && i + 1 < len && s[i + 1] == '{') { jaiBufPush(&result, '{'); i += 2; continue; }
        if (c == '}' && i + 1 < len && s[i + 1] == '}') { jaiBufPush(&result, '}'); i += 2; continue; }
        if (c == '}') {
            ok = jaiThrow(vm.cValueError,
                          "%s(): single '}' in the template; write '}}'", where);
            break;
        }
        if (c != '{') { jaiBufPush(&result, (uint8_t)c); i++; continue; }

        size_t fieldStart = i + 1;
        size_t j = fieldStart;
        while (j < len && s[j] != '}') {
            if (s[j] == '{') {
                ok = jaiThrow(vm.cValueError,
                              "%s(): nested '{' in a replacement field", where);
                break;
            }
            j++;
        }
        if (!ok) break;
        if (j >= len) {
            ok = jaiThrow(vm.cValueError,
                          "%s(): unmatched '{' in the template", where);
            break;
        }

        const char *field = s + fieldStart;
        size_t fieldLen = j - fieldStart;
        const char *colon = (const char *)memchr(field, ':', fieldLen);
        size_t nameLen = (colon != NULL) ? (size_t)(colon - field) : fieldLen;
        const char *spec = (colon != NULL) ? colon + 1 : "";
        size_t specLen = (colon != NULL) ? fieldLen - nameLen - 1 : 0;

        Value value = NULL_VAL;
        if (nameLen == 0) {
            int index = autoIndex++;
            if (index + 1 >= argc) {
                ok = jaiThrow(vm.cIndexError,
                              "%s(): replacement field %d has no argument "
                              "(%d given)", where, index, argc - 1);
                break;
            }
            value = args[index + 1];
        } else {
            bool digitsOnly = true;
            for (size_t k = 0; k < nameLen; k++) {
                if (field[k] < '0' || field[k] > '9') { digitsOnly = false; break; }
            }
            if (digitsOnly) {
                int64_t index = 0;
                for (size_t k = 0; k < nameLen; k++) {
                    if (index > (INT64_MAX - 9) / 10) { index = INT64_MAX; break; }
                    index = index * 10 + (field[k] - '0');
                }
                if (index + 1 >= (int64_t)argc) {
                    ok = jaiThrow(vm.cIndexError,
                                  "%s(): replacement field %lld has no argument "
                                  "(%d given)", where, (long long)index, argc - 1);
                    break;
                }
                value = args[index + 1];
            } else {
                ok = namedArgument(args, argc, field, nameLen, &value);
                if (!ok) break;
            }
        }

        ok = jaiFormatValue(&result, value, spec, specLen, where);
        i = j + 1;
    }

    if (!ok) {
        jaiBufFree(&result);
        return false;
    }
    return jaiStrTakeBuf(&result, out);
}

/* An f-string hole may hold a value of any type, so this isn't a str method —
 * jaiBuiltinMethod offers it for every receiver instead. */
static bool valueDunderFormat(int argc, Value *args, Value *out) {
    ObjString *spec;
    if (argc < 2) {
        return jaiThrow(vm.cTypeError,
                        "__format__() takes 1 argument but 0 were given");
    }
    if (!jaiStrWantStr(args[1], "__format__", "the format spec", &spec)) return false;

    JaiBuf buf;
    jaiBufInit(&buf);
    if (!jaiFormatValue(&buf, args[0], spec->chars, spec->length, "__format__")) {
        jaiBufFree(&buf);
        return false;
    }
    return jaiStrTakeBuf(&buf, out);
}

bool jaiValueFormatMethod(Value receiver, ObjString *name, Value *out) {
    if (name == NULL || strcmp(name->chars, "__format__") != 0) return false;
    *out = jaiBindNative(receiver, "__format__", valueDunderFormat, 2, 2, NULL);
    return true;
}
