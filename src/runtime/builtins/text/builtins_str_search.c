/* builtins_str_search.c — the search/match str methods: find, rfind, index,
 * count, starts_with, ends_with, contains. */

#include "runtime/builtins/text/builtins_str_methods.h"

#include <string.h>

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

bool strFind(int argc, Value *args, Value *out) {
    int64_t index;
    if (!searchIn(argc, args, "find", false, &index)) return false;
    *out = INT_VAL(index);
    return true;
}

bool strRfind(int argc, Value *args, Value *out) {
    int64_t index;
    if (!searchIn(argc, args, "rfind", true, &index)) return false;
    *out = INT_VAL(index);
    return true;
}

bool strIndex(int argc, Value *args, Value *out) {
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

bool strCount(int argc, Value *args, Value *out) {
    ObjString *s, *sub;
    if (!strReceiver(argc, args, "count", &s)) return false;
    if (!jaiStrWantStr(args[1], "count", "the text to count", &sub)) return false;
    int64_t start, end;
    if (!jaiStrOptInt(argc, args, 2, "count", "start", 0, &start)) return false;
    if (!jaiStrOptInt(argc, args, 3, "count", "end", INT64_MAX, &end)) return false;

    size_t from, to;
    resolveWindow(s, start, end, &from, &to);
    if (sub->length == 0) {
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
        pos = (size_t)(hit - s->chars) + sub->length;
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

bool strStartsWith(int argc, Value *args, Value *out) {
    return affixCommon(argc, args, "starts_with", true, out);
}

bool strEndsWith(int argc, Value *args, Value *out) {
    return affixCommon(argc, args, "ends_with", false, out);
}

bool strContains(int argc, Value *args, Value *out) {
    ObjString *s, *sub;
    if (!strReceiver(argc, args, "contains", &s)) return false;
    if (!jaiStrWantStr(args[1], "contains", "the text to look for", &sub)) return false;
    *out = BOOL_VAL(jaiStrFindBytes(s->chars, s->length, sub->chars, sub->length) != NULL);
    return true;
}
