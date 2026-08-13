/* builtins_str_case.c — case mapping, character classification, and the
 * str methods built directly on them: upper/lower/title/capitalize, the
 * is_* predicates, strip/lstrip/rstrip, pad_left/pad_right/center, and
 * repeat.
 *
 * The Unicode tables below are the reason this group is one file: caseMap
 * and the is*Cp classifiers are what upper()/lower()/title()/capitalize()
 * map through, what the is_* predicates test against, and what strip()
 * uses to find its default (whitespace) boundary. pad_left, pad_right,
 * center and repeat sit here too because they are the other methods built
 * from the same ASCII-fast-path-then-UTF-8-fallback shape as the case
 * mappers, not because they touch casing themselves.
 *
 * builtins_str.c keeps the method table, the scalar/byte-offset plumbing,
 * and the argument-checking helpers (strReceiver, optStr, ...) that every
 * method file, this one included, calls through builtins_str_methods.h.
 */

#include "runtime/builtins/text/builtins_str_methods.h"

#include <string.h>

/* ------------------------------------------------------------------ */
/* Unicode: simple case mapping and classification                      */
/* ------------------------------------------------------------------ */

/* The case tables below cover the scripts whose casing is algorithmic —
 * Latin, Greek, Cyrillic, Armenian, the fullwidth forms and Deseret — which is
 * every simple (1:1) case pair reachable without shipping UnicodeData.txt into
 * the core. Anything else maps to itself. Multi-scalar mappings (ß -> SS, the
 * Turkish dotted i) are deliberately absent: they are locale-dependent, and
 * std.str layers them on top where a locale is available. */

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
    if (c < 0x100) {                                  /* Latin-1 supplement */
        if (c == 0xB5) return up ? 0x39C : c;         /* micro sign -> capital mu */
        if (c == 0xFF) return up ? 0x178 : c;
        if (c == 0xDF) return c;                      /* ß: no 1:1 capital */
        if (c >= 0xC0 && c <= 0xDE && c != 0xD7) return up ? c : c + 32;
        if (c >= 0xE0 && c <= 0xFE && c != 0xF7) return up ? c - 32 : c;
        return c;
    }
    if (c <= 0x17F) {                                 /* Latin Extended-A */
        if (c == 0x131) return up ? 'I' : c;          /* dotless i */
        if (c == 0x138 || c == 0x149) return c;       /* ĸ, ŉ: uncased or 1:2 */
        if (c == 0x17F) return up ? 'S' : c;          /* long s */
        if (c == 0x178) return up ? c : 0xFF;
        if (c <= 0x137 || (c >= 0x14A && c <= 0x177)) return evenPair(c, up);
        return oddPair(c, up);                        /* 0x139-0x148, 0x179-0x17E */
    }
    if (c <= 0x24F) {                                 /* Latin Extended-B */
        /* Only the regular pair blocks; the Africanist and IPA additions in
         * between are individually irregular and stay uncased here. */
        if (c >= 0x1CD && c <= 0x1DC) return oddPair(c, up);
        if (c >= 0x1DE && c <= 0x1EF) return evenPair(c, up);
        if (c >= 0x1F8 && c <= 0x21F) return evenPair(c, up);
        if (c >= 0x222 && c <= 0x233) return evenPair(c, up);
        if (c >= 0x246 && c <= 0x24F) return evenPair(c, up);
        return c;
    }

    if (c >= 0x386 && c <= 0x3CE) {                   /* Greek */
        if (c == 0x386) return up ? c : 0x3AC;
        if (c >= 0x388 && c <= 0x38A) return up ? c : c + 0x25;
        if (c == 0x38C) return up ? c : 0x3CC;
        if (c == 0x38E || c == 0x38F) return up ? c : c + 0x3F;
        if (c >= 0x391 && c <= 0x3AB && c != 0x3A2) return up ? c : c + 0x20;
        if (c == 0x3AC) return up ? 0x386 : c;
        if (c >= 0x3AD && c <= 0x3AF) return up ? c - 0x25 : c;
        if (c == 0x3C2) return up ? 0x3A3 : c;        /* final sigma */
        if (c >= 0x3B1 && c <= 0x3CB) return up ? c - 0x20 : c;
        if (c == 0x3CC) return up ? 0x38C : c;
        if (c == 0x3CD || c == 0x3CE) return up ? c - 0x3F : c;
        return c;
    }
    if (c >= 0x400 && c <= 0x52F) {                   /* Cyrillic */
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
    if (c >= 0x531 && c <= 0x556) return up ? c : c + 0x30;   /* Armenian */
    if (c >= 0x561 && c <= 0x586) return up ? c - 0x30 : c;
    if (c >= 0xFF21 && c <= 0xFF3A) return up ? c : c + 0x20;  /* fullwidth */
    if (c >= 0xFF41 && c <= 0xFF5A) return up ? c - 0x20 : c;
    if (c >= 0x10400 && c <= 0x10427) return up ? c : c + 0x28; /* Deseret */
    if (c >= 0x10428 && c <= 0x1044F) return up ? c - 0x28 : c;
    return c;
}

static inline int32_t upperCp(int32_t c) {
    return caseMap(c, true);
}
static inline int32_t lowerCp(int32_t c) {
    return caseMap(c, false);
}

/* A scalar is cased when either mapping moves it. */
static inline bool isCasedCp(int32_t c) {
    return upperCp(c) != c || lowerCp(c) != c;
}
static const CpRange kDigitRanges[] = {
    {0x0030, 0x0039}, {0x0660, 0x0669}, {0x06F0, 0x06F9}, {0x07C0, 0x07C9},
    {0x0966, 0x096F}, {0x09E6, 0x09EF}, {0x0A66, 0x0A6F}, {0x0AE6, 0x0AEF},
    {0x0B66, 0x0B6F}, {0x0BE6, 0x0BEF}, {0x0C66, 0x0C6F}, {0x0CE6, 0x0CEF},
    {0x0D66, 0x0D6F}, {0x0E50, 0x0E59}, {0x0ED0, 0x0ED9}, {0x0F20, 0x0F29},
    {0x1040, 0x1049}, {0x17E0, 0x17E9}, {0x1810, 0x1819}, {0xFF10, 0xFF19},
    {0x1D7CE, 0x1D7FF},
};

/* Letters that no case mapping reaches: uncased scripts and the modifier
 * letters. Cased scalars are recognised by isCasedCp and are not repeated. */
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

/* Shared with builtins_str_split.c: split()/rsplit() in whitespace mode draw
 * the same boundary strip() does. */
bool isSpaceCp(int32_t c) {
    if ((uint32_t)c < 0x80u)
        return c == ' ' || (c >= 0x09 && c <= 0x0D);

    return inRanges(c, kSpaceRanges, JAI_COUNT_OF(kSpaceRanges));
}

/* ------------------------------------------------------------------ */
/* upper / lower / title / capitalize                                   */
/* ------------------------------------------------------------------ */

/* Shared body for upper/lower: `up` selects the mapping direction. */
static bool mapCase(ObjString *s, bool up, Value *out) {
    const size_t length = (size_t)s->length;

    /* Pure ASCII is overwhelmingly common and case mapping cannot change its
     * byte length, so write the result exactly once. */
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

bool strUpper(int argc, Value *args, Value *out) {
    ObjString *s;
    if (!strReceiver(argc, args, "upper", &s)) return false;
    return mapCase(s, true, out);
}

bool strLower(int argc, Value *args, Value *out) {
    ObjString *s;
    if (!strReceiver(argc, args, "lower", &s)) return false;
    return mapCase(s, false, out);
}

/* Word-initial scalars go up, the rest go down. A word starts wherever the
 * previous scalar is neither a letter nor a digit. */
bool strTitle(int argc, Value *args, Value *out) {
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

bool strCapitalize(int argc, Value *args, Value *out) {
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

/* ------------------------------------------------------------------ */
/* strip / lstrip / rstrip                                              */
/* ------------------------------------------------------------------ */

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

/* Shared body for strip/lstrip/rstrip. A NULL set means "whitespace". */
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

bool strStrip(int argc, Value *args, Value *out) {
    ObjString *s, *set;
    if (!strReceiver(argc, args, "strip", &s)) return false;
    if (!optStr(argc, args, 1, "strip", "the character set", &set)) return false;
    return stripSides(s, set, true, true, out);
}

bool strLstrip(int argc, Value *args, Value *out) {
    ObjString *s, *set;
    if (!strReceiver(argc, args, "lstrip", &s)) return false;
    if (!optStr(argc, args, 1, "lstrip", "the character set", &set)) return false;
    return stripSides(s, set, true, false, out);
}

bool strRstrip(int argc, Value *args, Value *out) {
    ObjString *s, *set;
    if (!strReceiver(argc, args, "rstrip", &s)) return false;
    if (!optStr(argc, args, 1, "rstrip", "the character set", &set)) return false;
    return stripSides(s, set, false, true, out);
}

/* ------------------------------------------------------------------ */
/* is_digit / is_alpha / is_alnum / is_space / is_upper / is_lower       */
/* ------------------------------------------------------------------ */

/* Shared body for the is_* predicates. `kind` selects the test; every one of
 * them is false for the empty string, since no scalar satisfies it. */
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

bool strIsDigit(int argc, Value *args, Value *out) {
    ObjString *s;
    if (!strReceiver(argc, args, "is_digit", &s)) return false;
    return classifyAll(s, CLASS_DIGIT, out);
}

bool strIsAlpha(int argc, Value *args, Value *out) {
    ObjString *s;
    if (!strReceiver(argc, args, "is_alpha", &s)) return false;
    return classifyAll(s, CLASS_ALPHA, out);
}

bool strIsAlnum(int argc, Value *args, Value *out) {
    ObjString *s;
    if (!strReceiver(argc, args, "is_alnum", &s)) return false;
    return classifyAll(s, CLASS_ALNUM, out);
}

bool strIsSpace(int argc, Value *args, Value *out) {
    ObjString *s;
    if (!strReceiver(argc, args, "is_space", &s)) return false;
    return classifyAll(s, CLASS_SPACE, out);
}

bool strIsUpper(int argc, Value *args, Value *out) {
    ObjString *s;
    if (!strReceiver(argc, args, "is_upper", &s)) return false;
    return classifyAll(s, CLASS_UPPER, out);
}

bool strIsLower(int argc, Value *args, Value *out) {
    ObjString *s;
    if (!strReceiver(argc, args, "is_lower", &s)) return false;
    return classifyAll(s, CLASS_LOWER, out);
}

/* ------------------------------------------------------------------ */
/* pad_left / pad_right / center / repeat                               */
/* ------------------------------------------------------------------ */

/* The fill argument of pad_left/pad_right/center: exactly one scalar. */
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

/* Shared body for pad_left/pad_right/center. `side` is -1, 1 or 0. */
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

bool strPadLeft(int argc, Value *args, Value *out) {
    return padCommon(argc, args, "pad_left", -1, out);
}

bool strPadRight(int argc, Value *args, Value *out) {
    return padCommon(argc, args, "pad_right", 1, out);
}

bool strCenter(int argc, Value *args, Value *out) {
    return padCommon(argc, args, "center", 0, out);
}

bool strRepeat(int argc, Value *args, Value *out) {
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
