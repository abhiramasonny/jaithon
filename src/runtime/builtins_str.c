/* builtins_str.c — the `str` method table and the `__prim__.str_*` surface.
 *
 * Every operation here counts Unicode scalar values, never bytes; byte
 * offsets are derived from scalar indices on demand, with an ASCII fast path.
 */

#include "builtins_str.h"
#include "methods.h"
#include "runtime.h"

#include "../vm/gc.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Unicode: simple case mapping and classification                      */
/* ------------------------------------------------------------------ */

/* Covers algorithmic 1:1 case pairs only (no UnicodeData.txt); multi-scalar
 * mappings like ß->SS are locale-dependent and layered on top by std.str. */

typedef struct { int32_t lo, hi; } CpRange;

/* The range tables are sorted, so a miss costs log2(n) comparisons. */
static inline bool inRanges(int32_t cp, const CpRange *ranges, size_t count) {
    size_t lo = 0, hi = count;

    while (lo < hi) {
        const size_t mid = lo + ((hi - lo) >> 1);
        if (cp < ranges[mid].lo)
            hi = mid;
        else if (cp > ranges[mid].hi)
            lo = mid + 1;
        else
            return true;
    }

    return false;
}

/* Blocks laid out as (even = capital, odd = small) pairs, and the reverse. */
static inline int32_t evenPair(int32_t c, bool up) {
    return up ? ((c & 1) ? c - 1 : c) : ((c & 1) ? c : c + 1);
}

static inline int32_t oddPair(int32_t c, bool up) {
    return up ? ((c & 1) ? c : c - 1) : ((c & 1) ? c + 1 : c);
}

static int32_t caseMap(int32_t c, bool up) {
    if (c < 0) return c;

    if (c < 0x80) {
        if (up) return (c >= 'a' && c <= 'z') ? c - 32 : c;
        return (c >= 'A' && c <= 'Z') ? c + 32 : c;
    }
    if (c < 0x100) {
        if (c == 0xB5) return up ? 0x39C : c;
        if (c == 0xFF) return up ? 0x178 : c;
        if (c == 0xDF) return c;
        if (c >= 0xC0 && c <= 0xDE && c != 0xD7) return up ? c : c + 32;
        if (c >= 0xE0 && c <= 0xFE && c != 0xF7) return up ? c - 32 : c;
        return c;
    }
    if (c <= 0x17F) {
        if (c == 0x131) return up ? 'I' : c;
        if (c == 0x138 || c == 0x149) return c;
        if (c == 0x17F) return up ? 'S' : c;
        if (c == 0x178) return up ? c : 0xFF;
        if (c <= 0x137 || (c >= 0x14A && c <= 0x177)) return evenPair(c, up);
        return oddPair(c, up);
    }
    if (c <= 0x24F) {
        /* Only regular pair blocks; irregular Africanist/IPA codepoints stay
         * uncased. */
        if (c >= 0x1CD && c <= 0x1DC) return oddPair(c, up);
        if (c >= 0x1DE && c <= 0x1EF) return evenPair(c, up);
        if (c >= 0x1F8 && c <= 0x21F) return evenPair(c, up);
        if (c >= 0x222 && c <= 0x233) return evenPair(c, up);
        if (c >= 0x246 && c <= 0x24F) return evenPair(c, up);
        return c;
    }

    if (c >= 0x386 && c <= 0x3CE) {
        if (c == 0x386) return up ? c : 0x3AC;
        if (c >= 0x388 && c <= 0x38A) return up ? c : c + 0x25;
        if (c == 0x38C) return up ? c : 0x3CC;
        if (c == 0x38E || c == 0x38F) return up ? c : c + 0x3F;
        if (c >= 0x391 && c <= 0x3AB && c != 0x3A2) return up ? c : c + 0x20;
        if (c == 0x3AC) return up ? 0x386 : c;
        if (c >= 0x3AD && c <= 0x3AF) return up ? c - 0x25 : c;
        if (c == 0x3C2) return up ? 0x3A3 : c;
        if (c >= 0x3B1 && c <= 0x3CB) return up ? c - 0x20 : c;
        if (c == 0x3CC) return up ? 0x38C : c;
        if (c == 0x3CD || c == 0x3CE) return up ? c - 0x3F : c;
        return c;
    }
    if (c >= 0x400 && c <= 0x52F) {
        if (c <= 0x40F) return up ? c : c + 0x50;
        if (c <= 0x42F) return up ? c : c + 0x20;
        if (c <= 0x44F) return up ? c - 0x20 : c;
        if (c <= 0x45F) return up ? c - 0x50 : c;
        if (c >= 0x460 && c <= 0x481) return evenPair(c, up);
        if (c >= 0x48A && c <= 0x4BF) return evenPair(c, up);
        if (c >= 0x4C1 && c <= 0x4CE) return oddPair(c, up);
        if (c >= 0x4D0) return evenPair(c, up);
        return c;
    }
    if (c >= 0x531 && c <= 0x556) return up ? c : c + 0x30;
    if (c >= 0x561 && c <= 0x586) return up ? c - 0x30 : c;
    if (c >= 0xFF21 && c <= 0xFF3A) return up ? c : c + 0x20;
    if (c >= 0xFF41 && c <= 0xFF5A) return up ? c - 0x20 : c;
    if (c >= 0x10400 && c <= 0x10427) return up ? c : c + 0x28;
    if (c >= 0x10428 && c <= 0x1044F) return up ? c - 0x28 : c;
    return c;
}

static inline int32_t upperCp(int32_t c) {
    return caseMap(c, true);
}
static inline int32_t lowerCp(int32_t c) {
    return caseMap(c, false);
}

static inline bool isCasedCp(int32_t c) {
    return upperCp(c) != c || lowerCp(c) != c;
}
static inline bool isUpperCp(int32_t c) {
    return lowerCp(c) != c;
}
static inline bool isLowerCp(int32_t c) {
    return upperCp(c) != c;
}

static const CpRange kDigitRanges[] = {
    {0x0030, 0x0039}, {0x0660, 0x0669}, {0x06F0, 0x06F9}, {0x07C0, 0x07C9},
    {0x0966, 0x096F}, {0x09E6, 0x09EF}, {0x0A66, 0x0A6F}, {0x0AE6, 0x0AEF},
    {0x0B66, 0x0B6F}, {0x0BE6, 0x0BEF}, {0x0C66, 0x0C6F}, {0x0CE6, 0x0CEF},
    {0x0D66, 0x0D6F}, {0x0E50, 0x0E59}, {0x0ED0, 0x0ED9}, {0x0F20, 0x0F29},
    {0x1040, 0x1049}, {0x17E0, 0x17E9}, {0x1810, 0x1819}, {0xFF10, 0xFF19},
    {0x1D7CE, 0x1D7FF},
};

/* Uncased letters no case mapping reaches; cased scalars are covered by
 * isCasedCp and not repeated here. */
static const CpRange kLetterRanges[] = {
    {0x00AA, 0x00AA}, {0x00BA, 0x00BA}, {0x01BB, 0x01BB}, {0x01C0, 0x01C3},
    {0x0294, 0x0294}, {0x02B0, 0x02C1}, {0x0370, 0x0374}, {0x037A, 0x037A},
    {0x03F7, 0x03FB}, {0x0559, 0x0559}, {0x05D0, 0x05EA}, {0x05EF, 0x05F2},
    {0x0620, 0x064A}, {0x066E, 0x066F}, {0x0671, 0x06D3}, {0x06D5, 0x06D5},
    {0x0710, 0x072F}, {0x0904, 0x0939}, {0x093D, 0x093D}, {0x0950, 0x0950},
    {0x0958, 0x0961}, {0x0985, 0x098C}, {0x0993, 0x09A8}, {0x09AA, 0x09B0},
    {0x0A05, 0x0A0A}, {0x0B05, 0x0B0C}, {0x0C05, 0x0C0C}, {0x0D05, 0x0D0C},
    {0x0E01, 0x0E30}, {0x0E32, 0x0E33}, {0x0E40, 0x0E46}, {0x1100, 0x11FF},
    {0x1200, 0x1248}, {0x3005, 0x3006}, {0x3041, 0x3096}, {0x309D, 0x309F},
    {0x30A1, 0x30FA}, {0x30FC, 0x30FF}, {0x3400, 0x4DBF}, {0x4E00, 0x9FFF},
    {0xA000, 0xA48C}, {0xAC00, 0xD7A3}, {0xF900, 0xFA6D}, {0xFF66, 0xFFDC},
    {0x20000, 0x2A6DF},
};

static const CpRange kSpaceRanges[] = {
    {0x0009, 0x000D}, {0x001C, 0x001F}, {0x0020, 0x0020}, {0x0085, 0x0085},
    {0x00A0, 0x00A0}, {0x1680, 0x1680}, {0x2000, 0x200A}, {0x2028, 0x2029},
    {0x202F, 0x202F}, {0x205F, 0x205F}, {0x3000, 0x3000},
};

static inline bool isDigitCp(int32_t c) {
    if ((uint32_t)c < 0x80u)
        return c >= '0' && c <= '9';

    return inRanges(c, kDigitRanges, JAI_COUNT_OF(kDigitRanges));
}

static inline bool isAlphaCp(int32_t c) {
    if ((uint32_t)c < 0x80u)
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');

    if (isCasedCp(c))
        return true;

    return inRanges(c, kLetterRanges, JAI_COUNT_OF(kLetterRanges));
}

static inline bool isSpaceCp(int32_t c) {
    if ((uint32_t)c < 0x80u)
        return c == ' ' || (c >= 0x09 && c <= 0x0D);

    return inRanges(c, kSpaceRanges, JAI_COUNT_OF(kSpaceRanges));
}

/* ------------------------------------------------------------------ */
/* Scalars and byte offsets                                             */
/* ------------------------------------------------------------------ */

/* True when scalar index and byte offset coincide, which lets every offset
 * conversion below become a no-op for ASCII text. */
static inline bool isAscii(ObjString *s) {
    return (size_t)jaiStringScalarCount(s) == (size_t)s->length;
}

/* Clamped to the byte length when past the end. */
size_t jaiStrByteOffsetOf(ObjString *s, size_t index) {
    const size_t length = (size_t)s->length;

    if ((size_t)jaiStringScalarCount(s) == length)
        return index < length ? index : length;

    return jaiUtf8Offset(s->chars, length, index);
}

/* Scalar index of a byte offset that sits on a scalar boundary. */
static inline size_t scalarIndexOf(ObjString *s, size_t offset) {
    if ((size_t)jaiStringScalarCount(s) == (size_t)s->length)
        return offset;

    return jaiUtf8Length(s->chars, offset);
}

/* Resolves an optional [start, end) scalar window: negatives count from the
 * end, out-of-range bounds clamp, and an inverted window comes back empty. */
static inline void resolveWindow(ObjString *s, int64_t start, int64_t end,
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

/* memchr narrows scans to candidate starts, keeping this loop fast despite
 * its naive appearance. */
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

static const char *rfindBytes(const char *hay, size_t hayLen,
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

static inline bool cpInSet(int32_t cp, ObjString *set) {
    if ((uint32_t)cp < 0x80u)
        return memchr(set->chars, (unsigned char)cp, set->length) != NULL;

    const char *p = set->chars;
    const char *const end = p + set->length;

    while (p < end) {
        int len = 1;
        if (jaiUtf8Decode(p, end, &len) == cp)
            return true;
        p += len;
    }

    return false;
}

/* ------------------------------------------------------------------ */
/* Native plumbing                                                      */
/* ------------------------------------------------------------------ */

/* A bound native receives the receiver as args[0] and argc counts it, so every
 * body below reads its declared arguments from args[1] onwards. */
static inline bool strReceiver(int argc, Value *args, const char *method,
                               ObjString **out) {
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

/* Turns a finished buffer into a str Value, consuming the buffer either way. */
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

/* Absent or null yields NULL, which every caller reads as "the default
 * character set". */
static inline bool optStr(int argc, Value *args, int slot,
                          const char *method, const char *what,
                          ObjString **out) {
    if (argc <= slot || IS_NULL(args[slot])) {
        *out = NULL;
        return true;
    }

    return jaiStrWantStr(args[slot], method, what, out);
}

/* ------------------------------------------------------------------ */
/* str methods                                                          */
/* ------------------------------------------------------------------ */

static bool strLen(int argc, Value *args, Value *out) {
    ObjString *s;
    if (!strReceiver(argc, args, "len", &s)) return false;
    *out = INT_VAL((int64_t)jaiStringScalarCount(s));
    return true;
}

static bool mapCase(ObjString *s, bool up, Value *out) {
    const size_t length = (size_t)s->length;

    bool ascii = true;
    for (size_t i = 0; i < length; ++i) {
        if ((unsigned char)s->chars[i] & 0x80u) {
            ascii = false;
            break;
        }
    }

    if (ascii && length != 0) {
        ObjString *result = jaiStringReserve(length);
        if (result == NULL) return false;

        for (size_t i = 0; i < length; ++i) {
            unsigned char c = (unsigned char)s->chars[i];
            if (up) {
                if (c >= 'a' && c <= 'z') c = (unsigned char)(c - 32);
            } else {
                if (c >= 'A' && c <= 'Z') c = (unsigned char)(c + 32);
            }
            result->chars[i] = (char)c;
        }

        *out = OBJ_VAL(jaiStringSeal(result));
        return true;
    }

    JaiBuf buf;
    jaiBufInit(&buf);
    jaiBufReserve(&buf, length + 1);

    const char *p = s->chars;
    const char *const end = p + length;

    while (p < end) {
        const unsigned char c = (unsigned char)*p;

        if (c < 0x80u) {
            char mapped = (char)c;
            if (up) {
                if (mapped >= 'a' && mapped <= 'z') mapped -= 32;
            } else {
                if (mapped >= 'A' && mapped <= 'Z') mapped += 32;
            }
            jaiBufPush(&buf, mapped);
            ++p;
            continue;
        }

        int len = 1;
        const int32_t cp = jaiUtf8Decode(p, end, &len);

        if (cp < 0) {
            jaiBufAppend(&buf, p, (size_t)len);
        } else {
            char utf8[4];
            const int n = jaiUtf8Encode(caseMap(cp, up), utf8);
            if (n > 0) jaiBufAppend(&buf, utf8, (size_t)n);
        }

        p += len;
    }

    return jaiStrTakeBuf(&buf, out);
}

static bool strUpper(int argc, Value *args, Value *out) {
    ObjString *s;
    if (!strReceiver(argc, args, "upper", &s)) return false;
    return mapCase(s, true, out);
}

static bool strLower(int argc, Value *args, Value *out) {
    ObjString *s;
    if (!strReceiver(argc, args, "lower", &s)) return false;
    return mapCase(s, false, out);
}

static bool strTitle(int argc, Value *args, Value *out) {
    ObjString *s;
    if (!strReceiver(argc, args, "title", &s)) return false;

    const size_t length = (size_t)s->length;
    bool ascii = true;
    for (size_t i = 0; i < length; ++i) {
        if ((unsigned char)s->chars[i] & 0x80u) {
            ascii = false;
            break;
        }
    }

    if (ascii && length != 0) {
        ObjString *result = jaiStringReserve(length);
        if (result == NULL) return false;

        bool startOfWord = true;
        for (size_t i = 0; i < length; ++i) {
            unsigned char c = (unsigned char)s->chars[i];
            const bool alpha =
                (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
            const bool digit = c >= '0' && c <= '9';
            const bool inWord = alpha || digit;

            if (startOfWord && inWord) {
                if (c >= 'a' && c <= 'z') c = (unsigned char)(c - 32);
            } else {
                if (c >= 'A' && c <= 'Z') c = (unsigned char)(c + 32);
            }

            result->chars[i] = (char)c;
            startOfWord = !inWord;
        }

        *out = OBJ_VAL(jaiStringSeal(result));
        return true;
    }

    JaiBuf buf;
    jaiBufInit(&buf);
    jaiBufReserve(&buf, length + 1);

    const char *p = s->chars;
    const char *const end = p + length;
    bool startOfWord = true;

    while (p < end) {
        if ((unsigned char)*p < 0x80u) {
            unsigned char c = (unsigned char)*p++;
            const bool alpha =
                (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
            const bool digit = c >= '0' && c <= '9';
            const bool inWord = alpha || digit;

            if (startOfWord && inWord) {
                if (c >= 'a' && c <= 'z') c = (unsigned char)(c - 32);
            } else {
                if (c >= 'A' && c <= 'Z') c = (unsigned char)(c + 32);
            }

            jaiBufPush(&buf, (char)c);
            startOfWord = !inWord;
            continue;
        }

        int len = 1;
        const int32_t cp = jaiUtf8Decode(p, end, &len);

        if (cp < 0) {
            jaiBufAppend(&buf, p, (size_t)len);
            startOfWord = true;
        } else {
            const bool inWord = isAlphaCp(cp) || isDigitCp(cp);
            char utf8[4];
            const int n =
                jaiUtf8Encode(caseMap(cp, startOfWord && inWord), utf8);
            if (n > 0) jaiBufAppend(&buf, utf8, (size_t)n);
            startOfWord = !inWord;
        }

        p += len;
    }

    return jaiStrTakeBuf(&buf, out);
}

static bool strCapitalize(int argc, Value *args, Value *out) {
    ObjString *s;
    if (!strReceiver(argc, args, "capitalize", &s)) return false;

    const size_t length = (size_t)s->length;
    bool ascii = true;
    for (size_t i = 0; i < length; ++i) {
        if ((unsigned char)s->chars[i] & 0x80u) {
            ascii = false;
            break;
        }
    }

    if (ascii && length != 0) {
        ObjString *result = jaiStringReserve(length);
        if (result == NULL) return false;

        for (size_t i = 0; i < length; ++i) {
            unsigned char c = (unsigned char)s->chars[i];

            if (i == 0) {
                if (c >= 'a' && c <= 'z') c = (unsigned char)(c - 32);
            } else {
                if (c >= 'A' && c <= 'Z') c = (unsigned char)(c + 32);
            }

            result->chars[i] = (char)c;
        }

        *out = OBJ_VAL(jaiStringSeal(result));
        return true;
    }

    JaiBuf buf;
    jaiBufInit(&buf);
    jaiBufReserve(&buf, length + 1);

    const char *p = s->chars;
    const char *const end = p + length;
    bool first = true;

    while (p < end) {
        if ((unsigned char)*p < 0x80u) {
            unsigned char c = (unsigned char)*p++;
            if (first) {
                if (c >= 'a' && c <= 'z') c = (unsigned char)(c - 32);
            } else {
                if (c >= 'A' && c <= 'Z') c = (unsigned char)(c + 32);
            }
            jaiBufPush(&buf, (char)c);
            first = false;
            continue;
        }

        int len = 1;
        const int32_t cp = jaiUtf8Decode(p, end, &len);

        if (cp < 0) {
            jaiBufAppend(&buf, p, (size_t)len);
        } else {
            char utf8[4];
            const int n = jaiUtf8Encode(caseMap(cp, first), utf8);
            if (n > 0) jaiBufAppend(&buf, utf8, (size_t)n);
        }

        first = false;
        p += len;
    }

    return jaiStrTakeBuf(&buf, out);
}

static bool stripSides(ObjString *s, ObjString *set, bool left, bool right,
                       Value *out) {
    const char *begin = s->chars;
    const char *end = s->chars + s->length;

    while (left && begin < end) {
        const unsigned char c = (unsigned char)*begin;

        if (c < 0x80u) {
            const bool remove =
                set != NULL
                    ? memchr(set->chars, c, set->length) != NULL
                    : (c == ' ' || (c >= 0x09 && c <= 0x0D));

            if (!remove) break;
            ++begin;
            continue;
        }

        int len = 1;
        const int32_t cp = jaiUtf8Decode(begin, end, &len);
        if (cp < 0 || !(set == NULL ? isSpaceCp(cp) : cpInSet(cp, set)))
            break;

        begin += len;
    }

    while (right && end > begin) {
        const unsigned char c = (unsigned char)end[-1];

        if (c < 0x80u) {
            const bool remove =
                set != NULL
                    ? memchr(set->chars, c, set->length) != NULL
                    : (c == ' ' || (c >= 0x09 && c <= 0x0D));

            if (!remove) break;
            --end;
            continue;
        }

        const char *start = end - 1;
        while (start > begin &&
               ((unsigned char)*start & 0xC0u) == 0x80u)
            --start;

        int len = 1;
        const int32_t cp = jaiUtf8Decode(start, end, &len);

        if (cp < 0 || start + len != end ||
            !(set == NULL ? isSpaceCp(cp) : cpInSet(cp, set)))
            break;

        end = start;
    }

    if (begin == s->chars && end == s->chars + s->length) {
        *out = OBJ_VAL(s);
        return true;
    }

    ObjString *result = jaiStringNew(begin, (size_t)(end - begin));
    if (result == NULL) return false;

    *out = OBJ_VAL(result);
    return true;
}

static bool strStrip(int argc, Value *args, Value *out) {
    ObjString *s, *set;
    if (!strReceiver(argc, args, "strip", &s)) return false;
    if (!optStr(argc, args, 1, "strip", "the character set", &set)) return false;
    return stripSides(s, set, true, true, out);
}

static bool strLstrip(int argc, Value *args, Value *out) {
    ObjString *s, *set;
    if (!strReceiver(argc, args, "lstrip", &s)) return false;
    if (!optStr(argc, args, 1, "lstrip", "the character set", &set)) return false;
    return stripSides(s, set, true, false, out);
}

static bool strRstrip(int argc, Value *args, Value *out) {
    ObjString *s, *set;
    if (!strReceiver(argc, args, "rstrip", &s)) return false;
    if (!optStr(argc, args, 1, "rstrip", "the character set", &set)) return false;
    return stripSides(s, set, false, true, out);
}

/* jaiListNew takes an int; text long enough to overflow one simply starts with
 * no reservation and grows. */
int jaiStrCapacityFor(size_t n) { return n > (size_t)INT32_MAX ? 0 : (int)n; }

/* Appends one substring to a list under construction. The list is rooted by
 * the caller, so the new string is reachable the moment it exists. */
static bool pushSlice(ObjList *list, const char *chars, size_t length) {
    ObjString *piece = jaiStringNew(chars, length);
    if (piece == NULL) return false;
    jaiListPush(list, OBJ_VAL(piece));
    return true;
}

static bool splitWhitespace(ObjString *s, int64_t maxsplit, bool fromRight,
                            Value *out) {
    ObjList *list = jaiListNew(0);
    jaiGCPushRoot(OBJ_VAL(list));

    bool ok = true;
    const char *const base = s->chars;
    const char *const end = base + s->length;

    if (!fromRight) {
        const char *p = base;
        int64_t splits = 0;

        while (ok) {
            while (p < end) {
                const unsigned char c = (unsigned char)*p;

                if (c < 0x80u) {
                    if (!(c == ' ' || (c >= 0x09 && c <= 0x0D)))
                        break;
                    ++p;
                    continue;
                }

                int len = 1;
                const int32_t cp = jaiUtf8Decode(p, end, &len);
                if (cp < 0 || !isSpaceCp(cp))
                    break;
                p += len;
            }

            if (p >= end) break;

            if (maxsplit >= 0 && splits >= maxsplit) {
                ok = pushSlice(list, p, (size_t)(end - p));
                break;
            }

            const char *const fieldStart = p;

            while (p < end) {
                const unsigned char c = (unsigned char)*p;

                if (c < 0x80u) {
                    if (c == ' ' || (c >= 0x09 && c <= 0x0D))
                        break;
                    ++p;
                    continue;
                }

                int len = 1;
                const int32_t cp = jaiUtf8Decode(p, end, &len);
                if (cp >= 0 && isSpaceCp(cp))
                    break;
                p += len;
            }

            ok = pushSlice(list, fieldStart, (size_t)(p - fieldStart));
            ++splits;
        }
    } else {
        const char *p = end;
        int64_t splits = 0;

        while (ok) {
            while (p > base) {
                const unsigned char c = (unsigned char)p[-1];

                if (c < 0x80u) {
                    if (!(c == ' ' || (c >= 0x09 && c <= 0x0D)))
                        break;
                    --p;
                    continue;
                }

                const char *start = p - 1;
                while (start > base &&
                       ((unsigned char)*start & 0xC0u) == 0x80u)
                    --start;

                int len = 1;
                const int32_t cp = jaiUtf8Decode(start, p, &len);
                if (cp < 0 || !isSpaceCp(cp))
                    break;
                p = start;
            }

            if (p <= base) break;

            if (maxsplit >= 0 && splits >= maxsplit) {
                ok = pushSlice(list, base, (size_t)(p - base));
                break;
            }

            const char *const fieldEnd = p;

            while (p > base) {
                const unsigned char c = (unsigned char)p[-1];

                if (c < 0x80u) {
                    if (c == ' ' || (c >= 0x09 && c <= 0x0D))
                        break;
                    --p;
                    continue;
                }

                const char *start = p - 1;
                while (start > base &&
                       ((unsigned char)*start & 0xC0u) == 0x80u)
                    --start;

                int len = 1;
                const int32_t cp = jaiUtf8Decode(start, p, &len);
                if (cp >= 0 && isSpaceCp(cp))
                    break;
                p = start;
            }

            ok = pushSlice(list, p, (size_t)(fieldEnd - p));
            ++splits;
        }

        for (int i = 0, j = list->count - 1; i < j; ++i, --j) {
            const Value tmp = list->items[i];
            list->items[i] = list->items[j];
            list->items[j] = tmp;
        }
    }

    jaiGCPopRoot();

    if (!ok) return false;
    *out = OBJ_VAL(list);
    return true;
}

static bool splitSeparator(ObjString *s, ObjString *sep, int64_t maxsplit,
                           bool fromRight, Value *out) {
    ObjList *list = jaiListNew(0);
    jaiGCPushRoot(OBJ_VAL(list));
    bool ok = true;
    const char *base = s->chars;
    size_t total = s->length;

    if (!fromRight) {
        size_t pos = 0;
        int64_t splits = 0;
        while (ok && (maxsplit < 0 || splits < maxsplit)) {
            const char *hit = jaiStrFindBytes(base + pos, total - pos, sep->chars,
                                        sep->length);
            if (hit == NULL) break;
            ok = pushSlice(list, base + pos, (size_t)(hit - (base + pos)));
            pos = (size_t)(hit - base) + sep->length;
            splits++;
        }
        if (ok) ok = pushSlice(list, base + pos, total - pos);
    } else {
        size_t limit = total;
        int64_t splits = 0;
        while (ok && (maxsplit < 0 || splits < maxsplit)) {
            const char *hit = rfindBytes(base, limit, sep->chars, sep->length);
            if (hit == NULL) break;
            size_t at = (size_t)(hit - base);
            ok = pushSlice(list, base + at + sep->length, limit - at - sep->length);
            limit = at;
            splits++;
        }
        if (ok) ok = pushSlice(list, base, limit);
        for (int i = 0, j = list->count - 1; i < j; i++, j--) {
            Value tmp = list->items[i];
            list->items[i] = list->items[j];
            list->items[j] = tmp;
        }
    }

    jaiGCPopRoot();
    if (!ok) return false;
    *out = OBJ_VAL(list);
    return true;
}

static bool splitCommon(int argc, Value *args, Value *out, const char *method,
                        bool fromRight) {
    ObjString *s, *sep;
    if (!strReceiver(argc, args, method, &s)) return false;
    if (!optStr(argc, args, 1, method, "the separator", &sep)) return false;
    int64_t maxsplit;
    if (!jaiStrOptInt(argc, args, 2, method, "maxsplit", -1, &maxsplit)) return false;

    if (sep == NULL) return splitWhitespace(s, maxsplit, fromRight, out);
    if (sep->length == 0) {
        return jaiThrow(vm.cValueError, "str.%s(): the separator cannot be empty",
                        method);
    }
    return splitSeparator(s, sep, maxsplit, fromRight, out);
}

static bool strSplit(int argc, Value *args, Value *out) {
    return splitCommon(argc, args, out, "split", false);
}

static bool strRsplit(int argc, Value *args, Value *out) {
    return splitCommon(argc, args, out, "rsplit", true);
}

/* Recognises the full Unicode set of line terminators so text from any
 * platform round-trips. */
static inline size_t lineBreakAt(const char *p, const char *end) {
    const unsigned char c = (unsigned char)*p;

    if (c == '\r')
        return (p + 1 < end && p[1] == '\n') ? 2u : 1u;

    if (c == '\n' || c == 0x0B || c == 0x0C ||
        (c >= 0x1C && c <= 0x1E))
        return 1u;

    if (c == 0xC2 && p + 1 < end &&
        (unsigned char)p[1] == 0x85)
        return 2u;

    if (c == 0xE2 && p + 2 < end &&
        (unsigned char)p[1] == 0x80 &&
        ((unsigned char)p[2] == 0xA8 ||
         (unsigned char)p[2] == 0xA9))
        return 3u;

    return 0;
}

static bool strSplitlines(int argc, Value *args, Value *out) {
    ObjString *s;
    if (!strReceiver(argc, args, "splitlines", &s)) return false;
    bool keepends = false;
    if (argc > 1 && !IS_NULL(args[1])) {
        if (!IS_BOOL(args[1])) {
            return jaiThrow(vm.cTypeError,
                            "str.splitlines(): keepends must be a bool, got %s",
                            jaiTypeNameStatic(args[1]));
        }
        keepends = AS_BOOL(args[1]);
    }

    ObjList *list = jaiListNew(0);
    jaiGCPushRoot(OBJ_VAL(list));
    bool ok = true;
    const char *p = s->chars;
    const char *end = p + s->length;
    const char *lineStart = p;
    while (ok && p < end) {
        size_t brk = lineBreakAt(p, end);
        if (brk == 0) { p++; continue; }
        size_t length = (size_t)(p - lineStart) + (keepends ? brk : 0);
        ok = pushSlice(list, lineStart, length);
        p += brk;
        lineStart = p;
    }
    /* A trailing terminator ends the last line; it does not start a new one. */
    if (ok && lineStart < end) ok = pushSlice(list, lineStart, (size_t)(end - lineStart));
    jaiGCPopRoot();
    if (!ok) return false;
    *out = OBJ_VAL(list);
    return true;
}

static inline bool joinItem(JaiBuf *buf, Value item, int index) {
    if (!IS_STRING(item)) {
        return jaiThrow(vm.cTypeError,
                        "str.join(): item %d is a %s, but every item must be a str",
                        index, jaiTypeNameStatic(item));
    }

    ObjString *const s = AS_STRING(item);
    jaiBufAppend(buf, s->chars, s->length);
    return true;
}

/* A list/tuple can be sized before copying (one exact allocation); anything
 * else needs a growable buffer since an iterator's length is unknown. */
static bool joinSized(ObjString *sep, const Value *items, int count,
                      Value *out) {
    const size_t sepLength = (size_t)sep->length;
    size_t total = 0;

    for (int i = 0; i < count; ++i) {
        const Value itemValue = items[i];

        if (!IS_STRING(itemValue)) {
            return jaiThrow(vm.cTypeError,
                            "str.join(): item %d is a %s, but every item must "
                            "be a str",
                            i, jaiTypeNameStatic(itemValue));
        }

        ObjString *const item = AS_STRING(itemValue);

        if (i > 0) {
            if (sepLength > UINT32_MAX - total)
                return jaiThrow(vm.cOverflowError,
                                "joined string exceeds the maximum length");
            total += sepLength;
        }

        if ((size_t)item->length > UINT32_MAX - total)
            return jaiThrow(vm.cOverflowError,
                            "joined string exceeds the maximum length");

        total += item->length;
    }

    if (total == 0) {
        *out = OBJ_VAL(jaiStringIntern("", 0));
        return true;
    }

    ObjString *result = jaiStringReserve(total);
    if (result == NULL) return false;

    char *p = result->chars;

    for (int i = 0; i < count; ++i) {
        if (i > 0 && sepLength != 0) {
            memcpy(p, sep->chars, sepLength);
            p += sepLength;
        }

        ObjString *const item = AS_STRING(items[i]);
        if (item->length != 0) {
            memcpy(p, item->chars, item->length);
            p += item->length;
        }
    }

    *out = OBJ_VAL(jaiStringSeal(result));
    return true;
}

static bool strJoin(int argc, Value *args, Value *out) {
    ObjString *sep;
    if (!strReceiver(argc, args, "join", &sep)) return false;
    Value seq = args[1];

    if (IS_LIST(seq) || IS_TUPLE(seq)) {
        int count = IS_LIST(seq) ? AS_LIST(seq)->count : (int)AS_TUPLE(seq)->count;
        const Value *items = IS_LIST(seq) ? AS_LIST(seq)->items
                                          : AS_TUPLE(seq)->items;
        return joinSized(sep, items, count, out);
    }

    JaiBuf buf;
    jaiBufInit(&buf);
    bool ok = true;

    {
        Value iterVal;
        if (!jaiGetIter(seq, &iterVal)) {
            jaiBufFree(&buf);
            return false;
        }
        jaiGCPushRoot(iterVal);
        ObjIter *it = AS_ITER(iterVal);
        Value item;
        int i = 0;
        while (ok && jaiIterNext(it, &item)) {
            if (i > 0) jaiBufAppend(&buf, sep->chars, sep->length);
            ok = joinItem(&buf, item, i);
            i++;
        }
        if (ok && vm.hasException) ok = false;    /* the iterator itself failed */
        jaiGCPopRoot();
    }

    if (!ok) {
        jaiBufFree(&buf);
        return false;
    }
    return jaiStrTakeBuf(&buf, out);
}

static bool strReplace(int argc, Value *args, Value *out) {
    ObjString *s, *old, *replacement;
    if (!strReceiver(argc, args, "replace", &s)) return false;
    if (!jaiStrWantStr(args[1], "replace", "the text to replace", &old)) return false;
    if (!jaiStrWantStr(args[2], "replace", "the replacement", &replacement)) return false;
    int64_t limit;
    if (!jaiStrOptInt(argc, args, 3, "replace", "count", -1, &limit)) return false;
    if (limit == 0) { *out = OBJ_VAL(s); return true; }

    JaiBuf buf;
    jaiBufInit(&buf);
    jaiBufReserve(&buf, (size_t)s->length + 1);
    int64_t done = 0;

    if (old->length == 0) {
        /* An empty match sits between every pair of scalars, and at both ends. */
        const char *p = s->chars;
        const char *end = p + s->length;
        jaiBufAppend(&buf, replacement->chars, replacement->length);
        done++;
        while (p < end && (limit < 0 || done < limit)) {
            int len = 1;
            if ((unsigned char)*p >= 0x80u)
                (void)jaiUtf8Decode(p, end, &len);
            jaiBufAppend(&buf, p, (size_t)len);
            p += len;
            if (p <= end) {
                jaiBufAppend(&buf, replacement->chars, replacement->length);
                done++;
            }
        }
        jaiBufAppend(&buf, p, (size_t)(end - p));
    } else {
        size_t pos = 0;
        while (limit < 0 || done < limit) {
            const char *hit = jaiStrFindBytes(s->chars + pos, s->length - pos,
                                        old->chars, old->length);
            if (hit == NULL) break;
            size_t at = (size_t)(hit - s->chars);
            jaiBufAppend(&buf, s->chars + pos, at - pos);
            jaiBufAppend(&buf, replacement->chars, replacement->length);
            pos = at + old->length;
            done++;
        }
        jaiBufAppend(&buf, s->chars + pos, s->length - pos);
    }
    return jaiStrTakeBuf(&buf, out);
}

static bool searchIn(int argc, Value *args, const char *method, bool fromRight,
                     int64_t *outIndex) {
    ObjString *s, *sub;
    if (!strReceiver(argc, args, method, &s)) return false;
    if (!jaiStrWantStr(args[1], method, "the text to look for", &sub)) return false;
    int64_t start, end;
    if (!jaiStrOptInt(argc, args, 2, method, "start", 0, &start)) return false;
    if (!jaiStrOptInt(argc, args, 3, method, "end", INT64_MAX, &end)) return false;

    size_t from, to;
    resolveWindow(s, start, end, &from, &to);
    const char *hit = fromRight
        ? rfindBytes(s->chars + from, to - from, sub->chars, sub->length)
        : jaiStrFindBytes(s->chars + from, to - from, sub->chars, sub->length);
    *outIndex = (hit == NULL) ? -1 : (int64_t)scalarIndexOf(s, (size_t)(hit - s->chars));
    return true;
}

static bool strFind(int argc, Value *args, Value *out) {
    int64_t index;
    if (!searchIn(argc, args, "find", false, &index)) return false;
    *out = INT_VAL(index);
    return true;
}

static bool strRfind(int argc, Value *args, Value *out) {
    int64_t index;
    if (!searchIn(argc, args, "rfind", true, &index)) return false;
    *out = INT_VAL(index);
    return true;
}

static bool strIndex(int argc, Value *args, Value *out) {
    int64_t index;
    if (!searchIn(argc, args, "index", false, &index)) return false;
    if (index < 0) {
        ObjString *sub = AS_STRING(args[1]);
        return jaiThrow(vm.cValueError, "str.index(): \"%.*s\" is not present",
                        (int)(sub->length > 60 ? 60 : sub->length), sub->chars);
    }
    *out = INT_VAL(index);
    return true;
}

static bool strCount(int argc, Value *args, Value *out) {
    ObjString *s, *sub;
    if (!strReceiver(argc, args, "count", &s)) return false;
    if (!jaiStrWantStr(args[1], "count", "the text to count", &sub)) return false;
    int64_t start, end;
    if (!jaiStrOptInt(argc, args, 2, "count", "start", 0, &start)) return false;
    if (!jaiStrOptInt(argc, args, 3, "count", "end", INT64_MAX, &end)) return false;

    size_t from, to;
    resolveWindow(s, start, end, &from, &to);
    if (sub->length == 0) {
        /* The empty string sits at every scalar boundary in the window. */
        const size_t totalScalars = (size_t)jaiStringScalarCount(s);
        const size_t scalars =
            totalScalars == (size_t)s->length
                ? to - from
                : jaiUtf8Length(s->chars + from, to - from);
        *out = INT_VAL((int64_t)scalars + 1);
        return true;
    }

    int64_t found = 0;
    size_t pos = from;
    while (pos <= to) {
        const char *hit = jaiStrFindBytes(s->chars + pos, to - pos, sub->chars,
                                    sub->length);
        if (hit == NULL) break;
        found++;
        pos = (size_t)(hit - s->chars) + sub->length;   /* non-overlapping */
    }
    *out = INT_VAL(found);
    return true;
}

static bool affixCommon(int argc, Value *args, const char *method, bool prefix,
                        Value *out) {
    ObjString *s, *affix;
    if (!strReceiver(argc, args, method, &s)) return false;
    if (!jaiStrWantStr(args[1], method, "the affix", &affix)) return false;
    int64_t start, end;
    if (!jaiStrOptInt(argc, args, 2, method, "start", 0, &start)) return false;
    if (!jaiStrOptInt(argc, args, 3, method, "end", INT64_MAX, &end)) return false;

    size_t from, to;
    resolveWindow(s, start, end, &from, &to);
    if (affix->length > to - from) { *out = BOOL_VAL(false); return true; }
    const char *at = prefix ? s->chars + from : s->chars + to - affix->length;
    *out = BOOL_VAL(memcmp(at, affix->chars, affix->length) == 0);
    return true;
}

static bool strStartsWith(int argc, Value *args, Value *out) {
    return affixCommon(argc, args, "starts_with", true, out);
}

static bool strEndsWith(int argc, Value *args, Value *out) {
    return affixCommon(argc, args, "ends_with", false, out);
}

static bool strContains(int argc, Value *args, Value *out) {
    ObjString *s, *sub;
    if (!strReceiver(argc, args, "contains", &s)) return false;
    if (!jaiStrWantStr(args[1], "contains", "the text to look for", &sub)) return false;
    *out = BOOL_VAL(jaiStrFindBytes(s->chars, s->length, sub->chars, sub->length) != NULL);
    return true;
}

typedef enum { CLASS_DIGIT, CLASS_ALPHA, CLASS_ALNUM, CLASS_SPACE,
               CLASS_UPPER, CLASS_LOWER } CharClass;

static bool classifyAll(ObjString *s, CharClass kind, Value *out) {
    const char *p = s->chars;
    const char *const end = p + s->length;

    if (p == end) {
        *out = BOOL_VAL(false);
        return true;
    }

    bool sawCased = false;
    bool result = true;

    while (p < end) {
        int32_t cp;

        const unsigned char c = (unsigned char)*p;
        if (c < 0x80u) {
            cp = (int32_t)c;
            ++p;

            switch (kind) {
                case CLASS_DIGIT:
                    result = c >= '0' && c <= '9';
                    break;

                case CLASS_ALPHA:
                    result = (c >= 'a' && c <= 'z') ||
                             (c >= 'A' && c <= 'Z');
                    break;

                case CLASS_ALNUM:
                    result = (c >= '0' && c <= '9') ||
                             (c >= 'a' && c <= 'z') ||
                             (c >= 'A' && c <= 'Z');
                    break;

                case CLASS_SPACE:
                    result = c == ' ' || (c >= 0x09 && c <= 0x0D);
                    break;

                case CLASS_UPPER:
                    if (c >= 'a' && c <= 'z') result = false;
                    if ((c >= 'a' && c <= 'z') ||
                        (c >= 'A' && c <= 'Z'))
                        sawCased = true;
                    break;

                case CLASS_LOWER:
                    if (c >= 'A' && c <= 'Z') result = false;
                    if ((c >= 'a' && c <= 'z') ||
                        (c >= 'A' && c <= 'Z'))
                        sawCased = true;
                    break;
            }

            if (!result) break;
            continue;
        }

        int len = 1;
        cp = jaiUtf8Decode(p, end, &len);
        p += len;

        if (cp < 0) {
            result = false;
            break;
        }

        switch (kind) {
            case CLASS_DIGIT:
                result = isDigitCp(cp);
                break;

            case CLASS_ALPHA:
                result = isAlphaCp(cp);
                break;

            case CLASS_ALNUM:
                result = isAlphaCp(cp) || isDigitCp(cp);
                break;

            case CLASS_SPACE:
                result = isSpaceCp(cp);
                break;

            case CLASS_UPPER: {
                const int32_t upper = upperCp(cp);
                const int32_t lower = lowerCp(cp);
                if (upper != cp) result = false;
                if (upper != cp || lower != cp) sawCased = true;
                break;
            }

            case CLASS_LOWER: {
                const int32_t upper = upperCp(cp);
                const int32_t lower = lowerCp(cp);
                if (lower != cp) result = false;
                if (upper != cp || lower != cp) sawCased = true;
                break;
            }
        }

        if (!result) break;
    }

    if (kind == CLASS_UPPER || kind == CLASS_LOWER)
        result = result && sawCased;

    *out = BOOL_VAL(result);
    return true;
}

static bool strIsDigit(int argc, Value *args, Value *out) {
    ObjString *s;
    if (!strReceiver(argc, args, "is_digit", &s)) return false;
    return classifyAll(s, CLASS_DIGIT, out);
}

static bool strIsAlpha(int argc, Value *args, Value *out) {
    ObjString *s;
    if (!strReceiver(argc, args, "is_alpha", &s)) return false;
    return classifyAll(s, CLASS_ALPHA, out);
}

static bool strIsAlnum(int argc, Value *args, Value *out) {
    ObjString *s;
    if (!strReceiver(argc, args, "is_alnum", &s)) return false;
    return classifyAll(s, CLASS_ALNUM, out);
}

static bool strIsSpace(int argc, Value *args, Value *out) {
    ObjString *s;
    if (!strReceiver(argc, args, "is_space", &s)) return false;
    return classifyAll(s, CLASS_SPACE, out);
}

static bool strIsUpper(int argc, Value *args, Value *out) {
    ObjString *s;
    if (!strReceiver(argc, args, "is_upper", &s)) return false;
    return classifyAll(s, CLASS_UPPER, out);
}

static bool strIsLower(int argc, Value *args, Value *out) {
    ObjString *s;
    if (!strReceiver(argc, args, "is_lower", &s)) return false;
    return classifyAll(s, CLASS_LOWER, out);
}

static bool fillScalar(int argc, Value *args, int slot, const char *method,
                       const char **outBytes, int *outLen) {
    *outBytes = " ";
    *outLen = 1;
    if (argc <= slot || IS_NULL(args[slot])) return true;

    ObjString *fill;
    if (!jaiStrWantStr(args[slot], method, "the fill", &fill)) return false;
    if (jaiStringScalarCount(fill) != 1) {
        return jaiThrow(vm.cValueError,
                        "str.%s(): the fill must be exactly one character, got "
                        "\"%.*s\"", method,
                        (int)(fill->length > 40 ? 40 : fill->length), fill->chars);
    }
    *outBytes = fill->chars;
    *outLen = (int)fill->length;
    return true;
}

static inline char *writeRepeated(char *dst, const char *pattern,
                                  size_t patternLen, size_t count) {
    if (count == 0 || patternLen == 0)
        return dst;

    if (patternLen == 1) {
        memset(dst, (unsigned char)pattern[0], count);
        return dst + count;
    }

    const size_t total = patternLen * count;
    memcpy(dst, pattern, patternLen);

    size_t written = patternLen;
    while (written < total) {
        size_t chunk = written;
        const size_t remaining = total - written;
        if (chunk > remaining) chunk = remaining;

        memcpy(dst + written, dst, chunk);
        written += chunk;
    }

    return dst + total;
}

static void appendFill(JaiBuf *buf, const char *fill, int fillLen, int64_t n) {
    for (int64_t i = 0; i < n; i++)
        jaiBufAppend(buf, fill, (size_t)fillLen);
}

/* `side` is -1, 1 or 0. */
static bool padCommon(int argc, Value *args, const char *method, int side,
                      Value *out) {
    ObjString *s;
    if (!strReceiver(argc, args, method, &s)) return false;

    int64_t width;
    if (!jaiStrWantInt(args[1], method, "the width", &width)) return false;

    const char *fill;
    int fillLen;
    if (!fillScalar(argc, args, 2, method, &fill, &fillLen)) return false;

    const int64_t scalars = (int64_t)jaiStringScalarCount(s);
    if (width <= scalars) {
        *out = OBJ_VAL(s);
        return true;
    }

    const uint64_t pad = (uint64_t)(width - scalars);
    const uint64_t available = (uint64_t)UINT32_MAX - s->length;

    if (pad > available / (uint64_t)fillLen) {
        return jaiThrow(vm.cOverflowError,
                        "str.%s(): a width of %lld exceeds the maximum string "
                        "length",
                        method, (long long)width);
    }

    const size_t total =
        (size_t)(pad * (uint64_t)fillLen) + (size_t)s->length;

    ObjString *result = jaiStringReserve(total);
    if (result == NULL) return false;

    const size_t left =
        side < 0 ? (size_t)pad :
        side == 0 ? (size_t)(pad >> 1) : 0u;
    const size_t right = (size_t)pad - left;

    char *dst = result->chars;
    dst = writeRepeated(dst, fill, (size_t)fillLen, left);

    if (s->length != 0) {
        memcpy(dst, s->chars, s->length);
        dst += s->length;
    }

    (void)writeRepeated(dst, fill, (size_t)fillLen, right);

    *out = OBJ_VAL(jaiStringSeal(result));
    return true;
}

static bool strPadLeft(int argc, Value *args, Value *out) {
    return padCommon(argc, args, "pad_left", -1, out);
}

static bool strPadRight(int argc, Value *args, Value *out) {
    return padCommon(argc, args, "pad_right", 1, out);
}

static bool strCenter(int argc, Value *args, Value *out) {
    return padCommon(argc, args, "center", 0, out);
}

static bool strRepeat(int argc, Value *args, Value *out) {
    ObjString *s;
    if (!strReceiver(argc, args, "repeat", &s)) return false;

    int64_t times;
    if (!jaiStrWantInt(args[1], "repeat", "the repeat count", &times))
        return false;

    if (times <= 0 || s->length == 0) {
        ObjString *empty = jaiStringIntern("", 0);
        if (empty == NULL) return false;
        *out = OBJ_VAL(empty);
        return true;
    }

    if ((uint64_t)times > (uint64_t)UINT32_MAX / s->length) {
        return jaiThrow(vm.cOverflowError,
                        "str.repeat(): %lld copies exceed the maximum string "
                        "length",
                        (long long)times);
    }

    const size_t total = (size_t)times * s->length;
    ObjString *result = jaiStringReserve(total);
    if (result == NULL) return false;

    (void)writeRepeated(result->chars, s->chars, s->length, (size_t)times);

    *out = OBJ_VAL(jaiStringSeal(result));
    return true;
}

static bool strChars(int argc, Value *args, Value *out) {
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

static bool strBytes(int argc, Value *args, Value *out) {
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

static bool strCodePoints(int argc, Value *args, Value *out) {
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
        /* An invalid byte reports as its negated value, so the list stays the
         * same length as chars() and the damage stays locatable. */
        jaiListPush(list, INT_VAL(cp));
        p += len;
    }
    jaiGCPopRoot();
    *out = OBJ_VAL(list);
    return true;
}

static bool strToBytes(int argc, Value *args, Value *out) {
    ObjString *s;
    if (!strReceiver(argc, args, "to_bytes", &s)) return false;
    ObjBytes *b = jaiBytesNew((const uint8_t *)s->chars, s->length);
    if (b == NULL) return false;
    *out = OBJ_VAL(b);
    return true;
}

static bool strFormat(int argc, Value *args, Value *out) {
    ObjString *tmpl;
    if (!strReceiver(argc, args, "format", &tmpl)) return false;
    return jaiFormatTemplate(tmpl, args, argc, out, "format");
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

static bool strParseInt(int argc, Value *args, Value *out) {
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

static bool strParseFloat(int argc, Value *args, Value *out) {
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
static bool strToInt(int argc, Value *args, Value *out) {
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

static bool strToFloat(int argc, Value *args, Value *out) {
    ObjString *s;
    if (!strReceiver(argc, args, "to_float", &s)) return false;
    double value = 0.0;
    *out = (parseFloatText(s->chars, s->length, &value) == PARSE_OK)
               ? FLOAT_VAL(value) : NULL_VAL;
    return true;
}

static bool strToStr(int argc, Value *args, Value *out) {
    ObjString *s;
    if (!strReceiver(argc, args, "to_str", &s)) return false;
    *out = OBJ_VAL(s);
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

/* 40-ish methods at a 64-slot capacity keeps probing short while staying tiny.
 * Slots hold method-index + 1 so zero remains the empty marker. */
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

/* A null bound means "the natural end", which is what makes s[a:] work. */
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

    /* UTF-8 byte order is scalar order, so memcmp gives the code-point
     * ordering the language specifies without decoding anything. */
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

static bool primStrFromCodepoint(int argc, Value *args, Value *out) {
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

static bool primStrToCodepoint(int argc, Value *args, Value *out) {
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

static bool primStrEncode(int argc, Value *args, Value *out) {
    ObjString *s;
    if (!jaiArgString(args[0], 0, "str_encode", &s)) return false;
    ObjBytes *b = jaiBytesNew((const uint8_t *)s->chars, s->length);
    if (b == NULL) return false;
    *out = OBJ_VAL(b);
    return true;
}

static bool primStrDecode(int argc, Value *args, Value *out) {
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

/* The `__prim__` namespace object. It is looked up rather than assumed so that
 * whichever primitive group registers first creates it and the rest join in. */
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

/* Registers both a `__prim__` module member (what lib/std resolves against)
 * and a dotted builtins global (for a front end using one identifier). */
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

    /* Registered here, not their own entry point: Appendix C lists them beside
     * the str ones and runtime.h declares one registrar for the pair. */
    jaiBytesRegisterPrimitives(ns);
}
