/* builtins_str_split.c — split, rsplit, splitlines, join, replace, and
 * format: the str methods that cut a string into pieces or assemble one
 * from pieces.
 *
 * split()/rsplit() and join() are natural opposites of each other, and
 * replace() is a find-then-splice that shares jaiStrFindBytes with split's
 * separator mode; splitlines() is split()'s line-oriented cousin. format()
 * belongs here rather than in its own file because it is a one-line
 * dispatch into builtins_format.c's template engine — nothing about it is
 * separable on its own.
 *
 * builtins_str.c keeps the method table and the plumbing every method file
 * calls through builtins_str_methods.h.
 */

#include "runtime/builtins/text/builtins_str_methods.h"

#include "vm/gc.h"

#include <string.h>

/* Appends one substring to a list under construction. The list is rooted by
 * the caller, so the new string is reachable the moment it exists. Shared
 * with builtins_str_convert.c's chars(). */
bool pushSlice(ObjList *list, const char *chars, size_t length) {
    ObjString *piece = jaiStringNew(chars, length);
    if (piece == NULL) return false;
    jaiListPush(list, OBJ_VAL(piece));
    return true;
}

/* ------------------------------------------------------------------ */
/* split / rsplit                                                       */
/* ------------------------------------------------------------------ */

/* split()/rsplit() with no separator: fields are the runs of non-whitespace. */
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

bool strSplit(int argc, Value *args, Value *out) {
    return splitCommon(argc, args, out, "split", false);
}

bool strRsplit(int argc, Value *args, Value *out) {
    return splitCommon(argc, args, out, "rsplit", true);
}

/* ------------------------------------------------------------------ */
/* splitlines                                                           */
/* ------------------------------------------------------------------ */

/* Byte length of the line terminator starting at `p`, or 0. Recognises the
 * whole Unicode set so that text from any platform round-trips. */
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

bool strSplitlines(int argc, Value *args, Value *out) {
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

/* ------------------------------------------------------------------ */
/* join                                                                 */
/* ------------------------------------------------------------------ */

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

/* A list or tuple can be measured before it is copied, so the result is one
 * exactly-sized allocation written once. Everything else has to go through a
 * growable buffer, because an iterator's length is not known until it ends. */
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

bool strJoin(int argc, Value *args, Value *out) {
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

/* ------------------------------------------------------------------ */
/* replace                                                              */
/* ------------------------------------------------------------------ */

bool strReplace(int argc, Value *args, Value *out) {
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

/* ------------------------------------------------------------------ */
/* format                                                               */
/* ------------------------------------------------------------------ */

bool strFormat(int argc, Value *args, Value *out) {
    ObjString *tmpl;
    if (!strReceiver(argc, args, "format", &tmpl)) return false;
    return jaiFormatTemplate(tmpl, args, argc, out, "format");
}
