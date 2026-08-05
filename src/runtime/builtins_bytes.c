/* builtins_bytes.c — the `bytes` method table and the `__prim__.bytes_*`
 * surface (spec Appendix C).
 *
 * bytes is the one text-adjacent type with no Unicode in it: every index,
 * length and window here counts bytes, which is why it reads nothing from the
 * scalar machinery in builtins_str.c and shares only the small argument and
 * buffer helpers that builtins_str.h declares.
 */

#include "builtins_str.h"
#include "methods.h"
#include "runtime.h"

#include "../vm/gc.h"

/* ------------------------------------------------------------------ */
/* Receiver                                                             */
/* ------------------------------------------------------------------ */

/* A bound native receives the receiver as args[0] and argc counts it, so every
 * body below reads its declared arguments from args[1] onwards. */
static bool bytesReceiver(int argc, Value *args, const char *method,
                          ObjBytes **out) {
    if (argc >= 1 && args != NULL && IS_BYTES(args[0])) {
        *out = AS_BYTES(args[0]);
        return true;
    }
    return jaiThrow(vm.cTypeError, "bytes.%s() needs a bytes receiver, got %s",
                    method,
                    (argc >= 1 && args != NULL) ? jaiTypeNameStatic(args[0])
                                                : "nothing");
}

/* ------------------------------------------------------------------ */
/* bytes methods                                                        */
/* ------------------------------------------------------------------ */

/* Normalises a possibly-negative byte window, clamping both ends. */
static void resolveByteWindow(uint32_t length, int64_t start, int64_t end,
                              size_t *outStart, size_t *outEnd) {
    int64_t n = (int64_t)length;
    if (start < 0) { start += n; if (start < 0) start = 0; }
    if (start > n) start = n;
    if (end < 0)   { end += n;   if (end < 0)   end = 0; }
    if (end > n)   end = n;
    if (end < start) end = start;
    *outStart = (size_t)start;
    *outEnd = (size_t)end;
}

/* The needle of a bytes search: another bytes value, or one byte as an int. */
static bool bytesNeedle(Value v, const char *method, const uint8_t **outData,
                        size_t *outLen, uint8_t *scratch) {
    if (IS_BYTES(v)) {
        *outData = AS_BYTES(v)->data;
        *outLen = AS_BYTES(v)->length;
        return true;
    }
    if (IS_INT(v)) {
        int64_t byte = AS_INT(v);
        if (byte < 0 || byte > 255) {
            return jaiThrow(vm.cValueError,
                            "bytes.%s(): %lld is not a byte value (0 to 255)",
                            method, (long long)byte);
        }
        *scratch = (uint8_t)byte;
        *outData = scratch;
        *outLen = 1;
        return true;
    }
    return jaiThrow(vm.cTypeError,
                    "bytes.%s(): expected bytes or an int, got %s", method,
                    jaiTypeNameStatic(v));
}

static bool bytesLen(int argc, Value *args, Value *out) {
    ObjBytes *b;
    if (!bytesReceiver(argc, args, "len", &b)) return false;
    *out = INT_VAL((int64_t)b->length);
    return true;
}

static bool bytesDecode(int argc, Value *args, Value *out) {
    ObjBytes *b;
    if (!bytesReceiver(argc, args, "decode", &b)) return false;
    if (!jaiUtf8Validate((const char *)b->data, b->length)) {
        return jaiThrow(vm.cValueError,
                        "bytes.decode(): the data is not valid UTF-8");
    }
    ObjString *s = jaiStringNew((const char *)b->data, b->length);
    if (s == NULL) return false;
    *out = OBJ_VAL(s);
    return true;
}

static bool bytesToList(int argc, Value *args, Value *out) {
    ObjBytes *b;
    if (!bytesReceiver(argc, args, "to_list", &b)) return false;
    ObjList *list = jaiListNew(jaiStrCapacityFor(b->length));
    jaiGCPushRoot(OBJ_VAL(list));
    for (uint32_t i = 0; i < b->length; i++) jaiListPush(list, INT_VAL(b->data[i]));
    jaiGCPopRoot();
    *out = OBJ_VAL(list);
    return true;
}

static bool bytesHex(int argc, Value *args, Value *out) {
    ObjBytes *b;
    if (!bytesReceiver(argc, args, "hex", &b)) return false;
    static const char *kHex = "0123456789abcdef";

    JaiBuf buf;
    jaiBufInit(&buf);
    jaiBufReserve(&buf, (size_t)b->length * 2 + 1);
    for (uint32_t i = 0; i < b->length; i++) {
        jaiBufPush(&buf, (uint8_t)kHex[b->data[i] >> 4]);
        jaiBufPush(&buf, (uint8_t)kHex[b->data[i] & 0x0F]);
    }
    return jaiStrTakeBuf(&buf, out);
}

static bool bytesSlice(int argc, Value *args, Value *out) {
    ObjBytes *b;
    if (!bytesReceiver(argc, args, "slice", &b)) return false;
    int64_t start, end;
    if (!jaiStrOptInt(argc, args, 1, "slice", "start", 0, &start)) return false;
    if (!jaiStrOptInt(argc, args, 2, "slice", "end", INT64_MAX, &end)) return false;

    size_t from, to;
    resolveByteWindow(b->length, start, end, &from, &to);
    ObjBytes *result = jaiBytesNew(b->data + from, to - from);
    if (result == NULL) return false;
    *out = OBJ_VAL(result);
    return true;
}

static bool bytesFindCommon(int argc, Value *args, const char *method,
                            int64_t *outIndex) {
    ObjBytes *b;
    if (!bytesReceiver(argc, args, method, &b)) return false;
    uint8_t scratch = 0;
    const uint8_t *needle;
    size_t needleLen;
    if (!bytesNeedle(args[1], method, &needle, &needleLen, &scratch)) return false;
    int64_t start, end;
    if (!jaiStrOptInt(argc, args, 2, method, "start", 0, &start)) return false;
    if (!jaiStrOptInt(argc, args, 3, method, "end", INT64_MAX, &end)) return false;

    size_t from, to;
    resolveByteWindow(b->length, start, end, &from, &to);
    const char *hit = jaiStrFindBytes((const char *)b->data + from, to - from,
                                (const char *)needle, needleLen);
    *outIndex = (hit == NULL) ? -1 : (int64_t)(hit - (const char *)b->data);
    return true;
}

static bool bytesFind(int argc, Value *args, Value *out) {
    int64_t index;
    if (!bytesFindCommon(argc, args, "find", &index)) return false;
    *out = INT_VAL(index);
    return true;
}

static bool bytesIndex(int argc, Value *args, Value *out) {
    int64_t index;
    if (!bytesFindCommon(argc, args, "index", &index)) return false;
    if (index < 0) {
        return jaiThrow(vm.cValueError, "bytes.index(): the value is not present");
    }
    *out = INT_VAL(index);
    return true;
}

static bool bytesContains(int argc, Value *args, Value *out) {
    int64_t index;
    if (!bytesFindCommon(argc, args, "contains", &index)) return false;
    *out = BOOL_VAL(index >= 0);
    return true;
}

static bool bytesCount(int argc, Value *args, Value *out) {
    ObjBytes *b;
    if (!bytesReceiver(argc, args, "count", &b)) return false;
    uint8_t scratch = 0;
    const uint8_t *needle;
    size_t needleLen;
    if (!bytesNeedle(args[1], "count", &needle, &needleLen, &scratch)) return false;
    if (needleLen == 0) {
        *out = INT_VAL((int64_t)b->length + 1);
        return true;
    }

    int64_t found = 0;
    size_t pos = 0;
    while (pos <= b->length) {
        const char *hit = jaiStrFindBytes((const char *)b->data + pos, b->length - pos,
                                    (const char *)needle, needleLen);
        if (hit == NULL) break;
        found++;
        pos = (size_t)(hit - (const char *)b->data) + needleLen;
    }
    *out = INT_VAL(found);
    return true;
}

static bool bytesAffix(int argc, Value *args, const char *method, bool prefix,
                       Value *out) {
    ObjBytes *b;
    if (!bytesReceiver(argc, args, method, &b)) return false;
    uint8_t scratch = 0;
    const uint8_t *affix;
    size_t affixLen;
    if (!bytesNeedle(args[1], method, &affix, &affixLen, &scratch)) return false;
    if (affixLen > b->length) { *out = BOOL_VAL(false); return true; }
    const uint8_t *at = prefix ? b->data : b->data + b->length - affixLen;
    *out = BOOL_VAL(memcmp(at, affix, affixLen) == 0);
    return true;
}

static bool bytesStartsWith(int argc, Value *args, Value *out) {
    return bytesAffix(argc, args, "starts_with", true, out);
}

static bool bytesEndsWith(int argc, Value *args, Value *out) {
    return bytesAffix(argc, args, "ends_with", false, out);
}

static bool bytesRepeat(int argc, Value *args, Value *out) {
    ObjBytes *b;
    if (!bytesReceiver(argc, args, "repeat", &b)) return false;
    int64_t times;
    if (!jaiStrWantInt(args[1], "repeat", "the repeat count", &times)) return false;
    if (times <= 0 || b->length == 0) {
        ObjBytes *empty = jaiBytesNew(NULL, 0);
        if (empty == NULL) return false;
        *out = OBJ_VAL(empty);
        return true;
    }
    if ((uint64_t)times > (uint64_t)UINT32_MAX / b->length) {
        return jaiThrow(vm.cOverflowError,
                        "bytes.repeat(): %lld copies exceed the maximum bytes "
                        "length", (long long)times);
    }

    /* Allocated empty and filled in place: jaiBytesNew is the only allocation,
     * and `b` survives it because the receiver is a stack root. */
    size_t total = (size_t)times * b->length;
    ObjBytes *result = jaiBytesNew(NULL, total);
    if (result == NULL) return false;
    for (int64_t i = 0; i < times; i++) {
        memcpy(result->data + (size_t)i * b->length, b->data, b->length);
    }
    *out = OBJ_VAL(result);
    return true;
}

static bool bytesConcat(int argc, Value *args, Value *out) {
    ObjBytes *b;
    if (!bytesReceiver(argc, args, "concat", &b)) return false;
    if (!IS_BYTES(args[1])) {
        return jaiThrow(vm.cTypeError, "bytes.concat(): expected bytes, got %s",
                        jaiTypeNameStatic(args[1]));
    }
    ObjBytes *other = AS_BYTES(args[1]);
    size_t total = (size_t)b->length + other->length;
    if (total > UINT32_MAX) {
        return jaiThrow(vm.cOverflowError,
                        "bytes.concat(): the result exceeds the maximum bytes "
                        "length");
    }

    ObjBytes *result = jaiBytesNew(NULL, total);
    if (result == NULL) return false;
    memcpy(result->data, b->data, b->length);
    memcpy(result->data + b->length, other->data, other->length);
    *out = OBJ_VAL(result);
    return true;
}

/* ------------------------------------------------------------------ */
/* Method table                                                         */
/* ------------------------------------------------------------------ */

static const JaiStrMethodEntry kBytesMethods[] = {
    {"len",         bytesLen,        1,  1, NULL},
    {"decode",      bytesDecode,     1,  1, NULL},
    {"to_list",     bytesToList,     1,  1, NULL},
    {"hex",         bytesHex,        1,  1, NULL},
    {"slice",       bytesSlice,      1,  3, NULL},
    {"find",        bytesFind,       2,  4, NULL},
    {"index",       bytesIndex,      2,  4, NULL},
    {"contains",    bytesContains,   2,  2, NULL},
    {"count",       bytesCount,      2,  2, NULL},
    {"starts_with", bytesStartsWith, 2,  2, NULL},
    {"ends_with",   bytesEndsWith,   2,  2, NULL},
    {"repeat",      bytesRepeat,     2,  2, NULL},
    {"concat",      bytesConcat,     2,  2, NULL},
};
static uint64_t gBytesHashes[JAI_COUNT_OF(kBytesMethods)];

static JaiStrMethodTable gBytesTable = {kBytesMethods,
                                        JAI_COUNT_OF(kBytesMethods),
                                        gBytesHashes, false};

bool jaiBytesMethod(Value receiver, ObjString *name, Value *out) {
    if (!IS_BYTES(receiver)) return false;
    return jaiStrLookupMethod(&gBytesTable, receiver, name, out);
}

/* ------------------------------------------------------------------ */
/* The __prim__ bytes surface (spec Appendix C)                         */
/* ------------------------------------------------------------------ */

static bool primBytesNew(int argc, Value *args, Value *out) {
    ObjList *list;
    if (!jaiArgList(args[0], 0, "bytes_new", &list)) return false;

    /* Validated before allocating, so a bad item cannot leave a half-filled
     * object behind for the collector to walk. */
    for (int i = 0; i < list->count; i++) {
        if (!IS_INT(list->items[i]) || AS_INT(list->items[i]) < 0 ||
            AS_INT(list->items[i]) > 255) {
            return jaiThrow(vm.cValueError,
                            "bytes_new(): item %d is not a byte value (0 to 255)",
                            i);
        }
    }
    ObjBytes *b = jaiBytesNew(NULL, (size_t)list->count);
    if (b == NULL) return false;
    for (int i = 0; i < list->count; i++) b->data[i] = (uint8_t)AS_INT(list->items[i]);
    *out = OBJ_VAL(b);
    return true;
}

static bool primBytesReceiver(Value v, const char *name, ObjBytes **out) {
    if (IS_BYTES(v)) { *out = AS_BYTES(v); return true; }
    return jaiThrow(vm.cTypeError, "%s(): expected bytes, got %s", name,
                    jaiTypeNameStatic(v));
}

static bool primBytesLen(int argc, Value *args, Value *out) {
    ObjBytes *b;
    if (!primBytesReceiver(args[0], "bytes_len", &b)) return false;
    *out = INT_VAL((int64_t)b->length);
    return true;
}

static bool primBytesGet(int argc, Value *args, Value *out) {
    ObjBytes *b;
    int64_t raw;
    if (!primBytesReceiver(args[0], "bytes_get", &b)) return false;
    if (!jaiArgInt(args[1], 1, "bytes_get", &raw)) return false;
    int index;
    if (!jaiNormalizeIndex(raw, (int)b->length, &index)) {
        return jaiThrow(vm.cIndexError,
                        "bytes_get(): index %lld is out of range for %u bytes",
                        (long long)raw, b->length);
    }
    *out = INT_VAL(b->data[index]);
    return true;
}

static bool primBytesSlice(int argc, Value *args, Value *out) {
    ObjBytes *b;
    if (!primBytesReceiver(args[0], "bytes_slice", &b)) return false;
    int64_t start, stop;
    if (!jaiStrSliceBound(args[1], "bytes_slice", 0, &start)) return false;
    if (!jaiStrSliceBound(args[2], "bytes_slice", INT64_MAX, &stop)) return false;

    size_t from, to;
    resolveByteWindow(b->length, start, stop, &from, &to);
    ObjBytes *result = jaiBytesNew(b->data + from, to - from);
    if (result == NULL) return false;
    *out = OBJ_VAL(result);
    return true;
}

static bool primBytesConcat(int argc, Value *args, Value *out) {
    return bytesConcat(argc, args, out);
}

static bool primBytesCmp(int argc, Value *args, Value *out) {
    ObjBytes *a, *b;
    if (!primBytesReceiver(args[0], "bytes_cmp", &a)) return false;
    if (!primBytesReceiver(args[1], "bytes_cmp", &b)) return false;
    size_t shared = a->length < b->length ? a->length : b->length;
    int cmp = shared > 0 ? memcmp(a->data, b->data, shared) : 0;
    if (cmp == 0) cmp = (a->length < b->length) ? -1 : (a->length > b->length);
    *out = INT_VAL(cmp < 0 ? -1 : (cmp > 0 ? 1 : 0));
    return true;
}

static bool primBytesFind(int argc, Value *args, Value *out) {
    return bytesFind(argc, args, out);
}

static bool primBytesToList(int argc, Value *args, Value *out) {
    return bytesToList(argc, args, out);
}

static bool primBytesHex(int argc, Value *args, Value *out) {
    return bytesHex(argc, args, out);
}

static bool primBytesFromHex(int argc, Value *args, Value *out) {
    ObjString *s;
    if (!jaiArgString(args[0], 0, "bytes_from_hex", &s)) return false;
    if ((s->length & 1u) != 0) {
        return jaiThrow(vm.cValueError,
                        "bytes_from_hex(): a hex string needs an even number of "
                        "digits, got %u", s->length);
    }

    size_t count = s->length / 2;
    for (size_t i = 0; i < s->length; i++) {
        int digit = jaiStrDigitValue(s->chars[i]);
        if (digit < 0 || digit > 15) {
            return jaiThrow(vm.cValueError,
                            "bytes_from_hex(): \"%.*s\" is not a hex string",
                            (int)(s->length > 60 ? 60 : s->length), s->chars);
        }
    }
    ObjBytes *b = jaiBytesNew(NULL, count);
    if (b == NULL) return false;
    for (size_t i = 0; i < count; i++) {
        b->data[i] = (uint8_t)((jaiStrDigitValue(s->chars[i * 2]) << 4) |
                               jaiStrDigitValue(s->chars[i * 2 + 1]));
    }
    *out = OBJ_VAL(b);
    return true;
}
void jaiBytesRegisterPrimitives(ObjModule *ns) {
    jaiStrDefinePrim(ns, "bytes_new",      primBytesNew,      1, 1);
    jaiStrDefinePrim(ns, "bytes_len",      primBytesLen,      1, 1);
    jaiStrDefinePrim(ns, "bytes_get",      primBytesGet,      2, 2);
    jaiStrDefinePrim(ns, "bytes_slice",    primBytesSlice,    3, 3);
    jaiStrDefinePrim(ns, "bytes_concat",   primBytesConcat,   2, 2);
    jaiStrDefinePrim(ns, "bytes_cmp",      primBytesCmp,      2, 2);
    jaiStrDefinePrim(ns, "bytes_find",     primBytesFind,     2, 4);
    jaiStrDefinePrim(ns, "bytes_to_list",  primBytesToList,   1, 1);
    jaiStrDefinePrim(ns, "bytes_hex",      primBytesHex,      1, 1);
    jaiStrDefinePrim(ns, "bytes_from_hex", primBytesFromHex,  1, 1);
}
