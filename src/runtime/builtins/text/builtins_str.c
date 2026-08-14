/* builtins_str.c — the `str` method table and the low-level __prim__.str_*
 * operator surface (spec Appendix C).
 *
 * Every operation here counts Unicode scalar values, never bytes: an index, a
 * length, a width, a slice bound and a padding count all mean the same thing
 * to a program whatever the encoding of the text underneath. Byte offsets
 * exist only inside this file and its siblings, derived from scalar indices
 * on demand, with an identity fast path for the ASCII strings that dominate
 * real programs.
 *
 * The ~40 str methods are near-independent Value* functions grouped by
 * sub-concern into four sibling files that share builtins_str_methods.h with
 * this one and with each other:
 *
 *   builtins_str_case.c    upper/lower/title/capitalize, the is_* predicates,
 *                          strip/lstrip/rstrip, pad_left/pad_right/center,
 *                          repeat, and the Unicode case/classification
 *                          tables they are all built from.
 *   builtins_str_search.c  find/rfind/index/count, starts_with/ends_with,
 *                          contains.
 *   builtins_str_split.c   split/rsplit/splitlines, join, replace, format.
 *   builtins_str_convert.c chars/bytes/code_points/to_bytes, parse_int/
 *                          parse_float/to_int/to_float/to_str, and the
 *                          codepoint/bytes __prim__ conversions.
 *
 * What stays here is what ties that split together and what does not belong
 * to any one sub-concern: the scalar/byte-offset plumbing and the argument
 * checks every method file calls through builtins_str_methods.h; the method
 * table itself (kStrMethods necessarily names every method regardless of
 * which file defines it); and the rest of the __prim__ surface — the direct
 * operator primitives (len, get, slice, concat, cmp, find) the compiler
 * lowers `s[i]`, `s[a:b]`, `s+s2` and comparisons to, plus the registration
 * entry point that ties in the four codepoint/bytes prims from
 * builtins_str_convert.c and, from builtins_bytes.c, the whole `bytes` prim
 * surface.
 *
 * The format engine that `str.format` and every f-string hole run through is
 * builtins_format.c; `bytes`, which shares the argument plumbing and none of
 * the Unicode, is builtins_bytes.c. builtins_str.h is what all of these
 * agree on with builtins_format.c and builtins_bytes.c; builtins_str_methods.h
 * is the narrower contract just among this file and its four method-table
 * siblings.
 */

#include "runtime/builtins/text/builtins_str_methods.h"

#include "vm/gc.h"

#include <string.h>

/* ------------------------------------------------------------------ */
/* Scalars and byte offsets                                             */
/* ------------------------------------------------------------------ */

size_t jaiStrByteOffsetOf(ObjString *s, size_t index) {
    const size_t length = (size_t)s->length;

    if ((size_t)jaiStringScalarCount(s) == length)
        return index < length ? index : length;

    return jaiUtf8Offset(s->chars, length, index);
}

size_t scalarIndexOf(ObjString *s, size_t offset) {
    if ((size_t)jaiStringScalarCount(s) == (size_t)s->length)
        return offset;

    return jaiUtf8Length(s->chars, offset);
}

void resolveWindow(ObjString *s, int64_t start, int64_t end,
                   size_t *outStart, size_t *outEnd) {
    const size_t byteLength = (size_t)s->length;
    const size_t scalarCount = (size_t)jaiStringScalarCount(s);
    const int64_t n = (int64_t)scalarCount;

    if (start < 0) {
        start += n;
        if (start < 0) start = 0;
    }
    if (start > n) start = n;

    if (end < 0) {
        end += n;
        if (end < 0) end = 0;
    }
    if (end > n) end = n;
    if (end < start) end = start;

    if (scalarCount == byteLength) {
        *outStart = (size_t)start;
        *outEnd = (size_t)end;
        return;
    }

    *outStart = jaiUtf8Offset(s->chars, byteLength, (size_t)start);
    *outEnd = jaiUtf8Offset(s->chars, byteLength, (size_t)end);
}

const char *jaiStrFindBytes(const char *hay, size_t hayLen,
                            const char *needle, size_t needleLen) {
    if (needleLen == 0) return hay;
    if (needleLen > hayLen) return NULL;

    if (needleLen == 1)
        return (const char *)memchr(hay, (unsigned char)needle[0], hayLen);

    const unsigned char first = (unsigned char)needle[0];
    const unsigned char last = (unsigned char)needle[needleLen - 1];
    const char *p = hay;
    size_t remaining = hayLen - needleLen + 1;

    while (remaining) {
        const char *hit = (const char *)memchr(p, first, remaining);
        if (hit == NULL) return NULL;

        if ((unsigned char)hit[needleLen - 1] == last &&
            (needleLen == 2 ||
             memcmp(hit + 1, needle + 1, needleLen - 2) == 0))
            return hit;

        const size_t consumed = (size_t)(hit - p) + 1;
        p += consumed;
        remaining -= consumed;
    }

    return NULL;
}

const char *rfindBytes(const char *hay, size_t hayLen,
                       const char *needle, size_t needleLen) {
    if (needleLen == 0) return hay + hayLen;
    if (needleLen > hayLen) return NULL;

    if (needleLen == 1) {
        const unsigned char target = (unsigned char)needle[0];
        for (size_t i = hayLen; i != 0; --i) {
            if ((unsigned char)hay[i - 1] == target)
                return hay + i - 1;
        }
        return NULL;
    }

    const unsigned char first = (unsigned char)needle[0];
    const unsigned char last = (unsigned char)needle[needleLen - 1];
    size_t i = hayLen - needleLen;

    for (;;) {
        const char *candidate = hay + i;

        if ((unsigned char)candidate[0] == first &&
            (unsigned char)candidate[needleLen - 1] == last &&
            (needleLen == 2 ||
             memcmp(candidate + 1, needle + 1, needleLen - 2) == 0))
            return candidate;

        if (i == 0) break;
        --i;
    }

    return NULL;
}

/* ------------------------------------------------------------------ */
/* Native plumbing                                                      */
/* ------------------------------------------------------------------ */

bool strReceiver(int argc, Value *args, const char *method, ObjString **out) {
    if (argc >= 1 && args != NULL) {
        const Value self = args[0];
        if (IS_STRING(self)) {
            *out = AS_STRING(self);
            return true;
        }

        return jaiThrow(vm.cTypeError,
                        "str.%s() needs a str receiver, got %s",
                        method, jaiTypeNameStatic(self));
    }

    return jaiThrow(vm.cTypeError,
                    "str.%s() needs a str receiver, got nothing", method);
}

bool jaiStrTakeBuf(JaiBuf *buf, Value *out) {
    size_t length = 0;
    char *chars = jaiBufTakeCString(buf, &length);
    if (chars == NULL) {
        return jaiThrow(vm.cRuntimeError, "out of memory building a string");
    }
    ObjString *s = jaiStringTake(chars, length);
    if (s == NULL) return false;          /* over the length limit; it threw */
    *out = OBJ_VAL(s);
    return true;
}

bool jaiStrWantStr(Value v, const char *method, const char *what,
                    ObjString **out) {
    if (IS_STRING(v)) { *out = AS_STRING(v); return true; }
    return jaiThrow(vm.cTypeError, "str.%s(): %s must be a str, got %s", method,
                    what, jaiTypeNameStatic(v));
}

bool jaiStrWantInt(Value v, const char *method, const char *what,
                    int64_t *out) {
    if (IS_INT(v))  { *out = AS_INT(v); return true; }
    if (IS_BOOL(v)) { *out = AS_BOOL(v) ? 1 : 0; return true; }
    return jaiThrow(vm.cTypeError, "str.%s(): %s must be an int, got %s", method,
                    what, jaiTypeNameStatic(v));
}

bool jaiStrOptInt(int argc, Value *args, int slot, const char *method,
                   const char *what, int64_t fallback, int64_t *out) {
    if (argc <= slot || IS_NULL(args[slot])) { *out = fallback; return true; }
    return jaiStrWantInt(args[slot], method, what, out);
}

bool optStr(int argc, Value *args, int slot, const char *method,
           const char *what, ObjString **out) {
    if (argc <= slot || IS_NULL(args[slot])) {
        *out = NULL;
        return true;
    }

    return jaiStrWantStr(args[slot], method, what, out);
}

int jaiStrCapacityFor(size_t n) { return n > (size_t)INT32_MAX ? 0 : (int)n; }

/* ------------------------------------------------------------------ */
/* str methods with no sub-concern of their own                         */
/* ------------------------------------------------------------------ */

static bool strLen(int argc, Value *args, Value *out) {
    ObjString *s;
    if (!strReceiver(argc, args, "len", &s)) return false;
    *out = INT_VAL((int64_t)jaiStringScalarCount(s));
    return true;
}

/* ------------------------------------------------------------------ */
/* Method table                                                         */
/* ------------------------------------------------------------------ */

static const JaiStrMethodEntry kStrMethods[] = {
    {"len",         strLen,          1,  1, NULL},
    {"upper",       strUpper,        1,  1, NULL},
    {"lower",       strLower,        1,  1, NULL},
    {"title",       strTitle,        1,  1, NULL},
    {"capitalize",  strCapitalize,   1,  1, NULL},
    {"strip",       strStrip,        1,  2, NULL},
    {"lstrip",      strLstrip,       1,  2, NULL},
    {"rstrip",      strRstrip,       1,  2, NULL},
    {"split",       strSplit,        1,  3, NULL},
    {"rsplit",      strRsplit,       1,  3, NULL},
    {"splitlines",  strSplitlines,   1,  2, NULL},
    {"join",        strJoin,         2,  2, NULL},
    {"replace",     strReplace,      3,  4, NULL},
    {"find",        strFind,         2,  4, NULL},
    {"rfind",       strRfind,        2,  4, NULL},
    {"index",       strIndex,        2,  4, NULL},
    {"count",       strCount,        2,  4, NULL},
    {"starts_with", strStartsWith,   2,  4, NULL},
    {"ends_with",   strEndsWith,     2,  4, NULL},
    {"contains",    strContains,     2,  2, NULL},
    {"is_digit",    strIsDigit,      1,  1, NULL},
    {"is_alpha",    strIsAlpha,      1,  1, NULL},
    {"is_alnum",    strIsAlnum,      1,  1, NULL},
    {"is_space",    strIsSpace,      1,  1, NULL},
    {"is_upper",    strIsUpper,      1,  1, NULL},
    {"is_lower",    strIsLower,      1,  1, NULL},
    {"pad_left",    strPadLeft,      2,  3, NULL},
    {"pad_right",   strPadRight,     2,  3, NULL},
    {"center",      strCenter,       2,  3, NULL},
    {"repeat",      strRepeat,       2,  2, NULL},
    {"chars",       strChars,        1,  1, NULL},
    {"bytes",       strBytes,        1,  1, NULL},
    {"code_points", strCodePoints,   1,  1, NULL},
    {"to_bytes",    strToBytes,      1,  1, NULL},
    {"format",      strFormat,       1, -1, NULL},
    {"parse_int",   strParseInt,     1,  2, NULL},
    {"parse_float", strParseFloat,   1,  1, NULL},
    {"to_int",      strToInt,        1,  2, NULL},
    {"to_float",    strToFloat,      1,  1, NULL},
    {"to_str",      strToStr,        1,  1, NULL},
};
static uint64_t gStrHashes[JAI_COUNT_OF(kStrMethods)];
static uint8_t  gStrLengths[JAI_COUNT_OF(kStrMethods)];

#define STR_METHOD_INDEX_CAP 64u
static uint16_t gStrSlots[STR_METHOD_INDEX_CAP];
static bool     gStrIndexReady;

static JaiStrMethodTable gStrTable = {
    kStrMethods, JAI_COUNT_OF(kStrMethods), gStrHashes, false
};

static void initStrMethodIndex(void) {
    if (gStrIndexReady) return;

    for (size_t i = 0; i < gStrTable.count; ++i) {
        const JaiStrMethodEntry *const e = gStrTable.entries + i;
        const size_t length = strlen(e->name);
        const uint64_t hash = jaiHashBytes(e->name, length);

        gStrHashes[i] = hash;
        gStrLengths[i] = (uint8_t)length;

        uint32_t slot = (uint32_t)hash & (STR_METHOD_INDEX_CAP - 1u);
        while (gStrSlots[slot] != 0)
            slot = (slot + 1u) & (STR_METHOD_INDEX_CAP - 1u);

        gStrSlots[slot] = (uint16_t)(i + 1);
    }

    gStrTable.ready = true;
    gStrIndexReady = true;
}

bool jaiStrLookupMethod(JaiStrMethodTable *table, Value receiver,
                        ObjString *name, Value *out) {
    if (name == NULL) return false;

    if (table == &gStrTable) {
        initStrMethodIndex();

        uint32_t slot =
            (uint32_t)name->hash & (STR_METHOD_INDEX_CAP - 1u);

        for (;;) {
            const uint16_t encoded = gStrSlots[slot];
            if (encoded == 0)
                return false;

            const size_t i = (size_t)encoded - 1u;

            if (gStrHashes[i] == name->hash &&
                (size_t)gStrLengths[i] == (size_t)name->length) {
                const JaiStrMethodEntry *const e = kStrMethods + i;
                const size_t length = (size_t)gStrLengths[i];

                if (length == 0 ||
                    memcmp(e->name, name->chars, length) == 0) {
                    *out = jaiBindNative(receiver, e->name, e->fn,
                                         e->minArity, e->maxArity, e->params);
                    return true;
                }
            }

            slot = (slot + 1u) & (STR_METHOD_INDEX_CAP - 1u);
        }
    }

    if (!table->ready) {
        for (size_t i = 0; i < table->count; ++i) {
            const char *const methodName = table->entries[i].name;
            table->hashes[i] =
                jaiHashBytes(methodName, strlen(methodName));
        }
        table->ready = true;
    }

    for (size_t i = 0; i < table->count; ++i) {
        if (table->hashes[i] != name->hash)
            continue;

        const JaiStrMethodEntry *const e = table->entries + i;
        const size_t length = strlen(e->name);

        if (length != (size_t)name->length ||
            (length != 0 && memcmp(e->name, name->chars, length) != 0))
            continue;

        *out = jaiBindNative(receiver, e->name, e->fn,
                             e->minArity, e->maxArity, e->params);
        return true;
    }

    return false;
}

bool jaiStrMethod(Value receiver, ObjString *name, Value *out) {
    if (!IS_STRING(receiver)) return false;
    return jaiStrLookupMethod(&gStrTable, receiver, name, out);
}
/* ------------------------------------------------------------------ */
/* The __prim__ string surface (spec Appendix C)                        */
/* ------------------------------------------------------------------ */

static bool primStrLen(int argc, Value *args, Value *out) {
    ObjString *s;
    if (!jaiArgString(args[0], 0, "str_len", &s)) return false;
    *out = INT_VAL((int64_t)jaiStringScalarCount(s));
    return true;
}

static bool primStrGet(int argc, Value *args, Value *out) {
    (void)argc;

    ObjString *s;
    int64_t raw;

    if (!jaiArgString(args[0], 0, "str_get", &s)) return false;
    if (!jaiArgInt(args[1], 1, "str_get", &raw)) return false;

    const uint32_t scalarCount = jaiStringScalarCount(s);
    int index;

    if (!jaiNormalizeIndex(raw, (int)scalarCount, &index)) {
        return jaiThrow(vm.cIndexError,
                        "str_get(): index %lld is out of range for a string of "
                        "%u characters",
                        (long long)raw, scalarCount);
    }

    if (scalarCount == s->length) {
        ObjString *scalar = jaiStringNew(s->chars + index, 1);
        if (scalar == NULL) return false;
        *out = OBJ_VAL(scalar);
        return true;
    }

    const size_t at = jaiUtf8Offset(s->chars, s->length, (size_t)index);
    int len = 1;
    (void)jaiUtf8Decode(s->chars + at, s->chars + s->length, &len);

    ObjString *scalar = jaiStringNew(s->chars + at, (size_t)len);
    if (scalar == NULL) return false;

    *out = OBJ_VAL(scalar);
    return true;
}

bool jaiStrSliceBound(Value v, const char *name, int64_t fallback,
                           int64_t *out) {
    if (IS_NULL(v)) { *out = fallback; return true; }
    return jaiArgInt(v, 1, name, out);
}

static bool primStrSlice(int argc, Value *args, Value *out) {
    ObjString *s;
    if (!jaiArgString(args[0], 0, "str_slice", &s)) return false;
    int64_t step = 1;
    if (argc > 3 && !jaiStrSliceBound(args[3], "str_slice", 1, &step)) return false;
    int64_t defaultStart = (step < 0) ? INT64_MAX : 0;
    int64_t defaultStop  = (step < 0) ? INT64_MIN : INT64_MAX;
    int64_t start, stop;
    if (!jaiStrSliceBound(args[1], "str_slice", defaultStart, &start)) return false;
    if (!jaiStrSliceBound(args[2], "str_slice", defaultStop, &stop)) return false;

    ObjString *result = jaiStringSlice(s, start, stop, step);
    if (result == NULL) return false;
    *out = OBJ_VAL(result);
    return true;
}

static bool primStrConcat(int argc, Value *args, Value *out) {
    ObjString *a, *b;
    if (!jaiArgString(args[0], 0, "str_concat", &a)) return false;
    if (!jaiArgString(args[1], 1, "str_concat", &b)) return false;
    ObjString *result = jaiStringConcat(a, b);
    if (result == NULL) return false;
    *out = OBJ_VAL(result);
    return true;
}

static bool primStrCmp(int argc, Value *args, Value *out) {
    ObjString *a, *b;
    if (!jaiArgString(args[0], 0, "str_cmp", &a)) return false;
    if (!jaiArgString(args[1], 1, "str_cmp", &b)) return false;
    if (a == b) {
        *out = INT_VAL(0);
        return true;
    }

    size_t shared = a->length < b->length ? a->length : b->length;
    int cmp = shared > 0 ? memcmp(a->chars, b->chars, shared) : 0;
    if (cmp == 0) cmp = (a->length < b->length) ? -1 : (a->length > b->length);
    *out = INT_VAL(cmp < 0 ? -1 : (cmp > 0 ? 1 : 0));
    return true;
}

static bool primStrFind(int argc, Value *args, Value *out) {
    ObjString *s, *sub;
    if (!jaiArgString(args[0], 0, "str_find", &s)) return false;
    if (!jaiArgString(args[1], 1, "str_find", &sub)) return false;
    int64_t start = 0;
    if (argc > 2 && !jaiStrSliceBound(args[2], "str_find", 0, &start)) return false;

    size_t from, to;
    resolveWindow(s, start, INT64_MAX, &from, &to);
    const char *hit = jaiStrFindBytes(s->chars + from, to - from, sub->chars,
                                sub->length);
    *out = INT_VAL(hit == NULL ? -1
                               : (int64_t)scalarIndexOf(s, (size_t)(hit - s->chars)));
    return true;
}

static ObjModule *primNamespace(void) {
    ObjString *name = jaiStringInternC("__prim__");
    if (name == NULL || vm.builtins == NULL) return NULL;

    Value existing;
    if (jaiModuleGet(vm.builtins, name, &existing)) {
        return IS_MODULE(existing) ? AS_MODULE(existing) : NULL;
    }

    jaiGCPushRoot(OBJ_VAL(name));
    ObjString *path = jaiStringInternC("<prim>");
    ObjModule *m = (path != NULL) ? jaiModuleNew(name, path) : NULL;
    if (m != NULL) {
        m->state = MOD_LOADED;      /* nothing may try to load it from disk */
        jaiDefineGlobal("__prim__", OBJ_VAL(m));
    }
    jaiGCPopRoot();
    return m;
}

void jaiStrDefinePrim(ObjModule *ns, const char *name, JaiNativeFn fn,
                       int minArity, int maxArity) {
    if (ns != NULL) {
        ObjNative *native = jaiNativeNew(fn, name, minArity, maxArity, NULL);
        jaiGCPushRoot(OBJ_VAL(native));
        ObjString *key = jaiStringInternC(name);
        if (key != NULL) jaiModuleSet(ns, key, OBJ_VAL(native));
        jaiGCPopRoot();
    }

    char dotted[64];
    int written = snprintf(dotted, sizeof dotted, "__prim__.%s", name);
    if (written > 0 && (size_t)written < sizeof dotted) {
        jaiDefineNative(dotted, fn, minArity, maxArity);
    }
}

void jaiRegisterStringPrimitives(void) {
    ObjModule *ns = primNamespace();

    jaiStrDefinePrim(ns, "str_len",            primStrLen,            1, 1);
    jaiStrDefinePrim(ns, "str_get",            primStrGet,            2, 2);
    jaiStrDefinePrim(ns, "str_slice",          primStrSlice,          3, 4);
    jaiStrDefinePrim(ns, "str_concat",         primStrConcat,         2, 2);
    jaiStrDefinePrim(ns, "str_cmp",            primStrCmp,            2, 2);
    jaiStrDefinePrim(ns, "str_find",           primStrFind,           2, 3);
    jaiStrDefinePrim(ns, "str_from_codepoint", primStrFromCodepoint,  1, 1);
    jaiStrDefinePrim(ns, "str_to_codepoint",   primStrToCodepoint,    1, 1);
    jaiStrDefinePrim(ns, "str_encode",         primStrEncode,         1, 1);
    jaiStrDefinePrim(ns, "str_decode",         primStrDecode,         1, 1);

    jaiBytesRegisterPrimitives(ns);
}
