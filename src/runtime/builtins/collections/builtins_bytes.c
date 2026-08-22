/* builtins_bytes.c — the `bytes` method table and `__prim__.bytes_*` surface (spec Appendix C); every index/length/window here counts bytes, not codepoints. */

#include <math.h>

#include "runtime/builtins/text/builtins_str.h"
#include "runtime/methods.h"
#include "runtime/parallel.h"
#include "runtime/runtime.h"

#include "vm/gc.h"

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
    if (list == NULL || !jaiListReserveExact(list, (int)b->length)) return false;
    for (uint32_t i = 0; i < b->length; i++) list->items[i] = INT_VAL(b->data[i]);
    list->count = (int)b->length;
    list->version++;
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

static bool primBytesNew(int argc, Value *args, Value *out) {
    ObjList *list;
    if (!jaiArgList(args[0], 0, "bytes_new", &list)) return false;

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
/* Clamp and round a list of floats into bytes, in one pass.
 *
 * Written in Jaithon this is a multiply, a NaN test, a `math.floor`, an `int`
 * and two compares per element, and the call to `math.floor` is what stops the
 * loop compiling: a 640x480 image took thirty-nine milliseconds, which is a
 * twenty-five frame ceiling on writing video before anything is encoded.
 *
 * `scale` multiplies before the clamp, which is what a caller holding
 * normalised floats wants. A NaN becomes zero rather than whatever the cast
 * would have produced. */
/* Below this many elements the loop stays on one thread: a small image is not
 * worth waking anything for. */
#define JAI_QUANTISE_CHUNK 32768

typedef struct {
    const Value *items;
    uint8_t     *dst;
    float        factor;
} QuantiseWork;

/* Single precision and no call to `floor`. The values are float32 to begin
 * with, and for a positive number truncation IS the floor -- which the clamp
 * has already established. `!(value > 0.0f)` catches the negatives and the
 * NaNs together, since a NaN fails every comparison. */
static void quantiseRange(void *context, size_t start, size_t end) {
    const QuantiseWork *work = (const QuantiseWork *)context;
    for (size_t i = start; i < end; i++) {
        const Value item = work->items[i];
        float value = IS_FLOAT(item) ? (float)AS_FLOAT(item) : (float)AS_INT(item);
        value = value * work->factor + 0.5f;
        if (!(value > 0.0f)) {
            work->dst[i] = 0;
            continue;
        }
        work->dst[i] = value >= 255.5f ? (uint8_t)255 : (uint8_t)value;
    }
}

static bool primBytesQuantise(int argc, Value *args, Value *out) {
    (void)argc;
    if (!IS_LIST(args[0])) {
        return jaiThrow(vm.cTypeError, "bytes_quantise(): expected a list, got %s",
                        jaiTypeNameStatic(args[0]));
    }
    double scale = 1.0;
    if (argc > 1 && !IS_NULL(args[1])) {
        if (IS_INT(args[1])) scale = (double)AS_INT(args[1]);
        else if (IS_FLOAT(args[1])) scale = AS_FLOAT(args[1]);
        else {
            return jaiThrow(vm.cTypeError, "bytes_quantise(): scale must be a number");
        }
    }

    ObjList *items = AS_LIST(args[0]);
    const size_t n = (size_t)items->count;
    ObjBytes *result = jaiBytesNew(NULL, n);
    if (result == NULL) return false;
    uint8_t *dst = result->data;

    /* A number that is not one is refused, but finding out costs a pass that
     * the parallel one below must not take -- it cannot raise from another
     * thread. Checking first is one cheap read of memory that is about to be
     * read again anyway. */
    for (size_t i = 0; i < n; i++) {
        if (!IS_FLOAT(items->items[i]) && !IS_INT(items->items[i])) {
            return jaiThrow(vm.cTypeError,
                            "bytes_quantise(): element %zu is %s, not a number",
                            i, jaiTypeNameStatic(items->items[i]));
        }
    }

    QuantiseWork work = {items->items, dst, (float)scale};
    jaiParallelChunks(n, JAI_QUANTISE_CHUNK, quantiseRange, &work);
    *out = OBJ_VAL(result);
    return true;
}

/* `list_filled(count, value)` -- a list of `count` copies of `value`.
 *
 * The idiom this replaces is `[0.0 for _slot in 0..n]`, which every
 * preallocation in jaicv and jaitensor is written as. That goes through the
 * comprehension's append path one element at a time and costs about 16 ns an
 * element -- 19 ms to size the 1.2-million-float buffer one 640x640 blob
 * needs, which was more than the convolution that consumed it. Reserving the
 * capacity once and writing straight into it is a memset in all but name.
 *
 * A negative count is refused rather than clamped: it always means the caller
 * computed a size wrongly, and a silent empty list hides that. */
static bool primListFilled(int argc, Value *args, Value *out) {
    (void)argc;
    int64_t count;
    if (!jaiStrWantInt(args[0], "list_filled", "the count", &count)) return false;
    if (count < 0) {
        return jaiThrow(vm.cValueError,
                        "list_filled(): a count cannot be negative, got %lld",
                        (long long)count);
    }
    if (count > INT32_MAX) {
        return jaiThrow(vm.cValueError,
                        "list_filled(): %lld is more elements than a list holds",
                        (long long)count);
    }

    Value fill = argc > 1 ? args[1] : INT_VAL(0);
    ObjList *list = jaiListNew((int)count);
    if (list == NULL) return false;
    /* Rooted across the reserve: growing the backing array can collect. */
    jaiGCPushRoot(OBJ_VAL(list));
    jaiListReserve(list, (int)count);
    jaiGCPopRoot();
    if (count > 0 && list->capacity < (int)count) return false;

    for (int64_t i = 0; i < count; i++) list->items[i] = fill;
    list->count = (int)count;
    list->version++;
    *out = OBJ_VAL(list);
    return true;
}

/* `list_fill_pattern(values, start, repeats, pattern)` -- write `pattern` into
 * `values` `repeats` times over, beginning at `start`.
 *
 * Every filled shape a drawing routine makes is rows of one colour, and a row
 * written a pixel at a time in Jaithon costs about 11 ns a pixel: a filled
 * rectangle over a 720p frame took 8.2 ms, which is more than the whole rest
 * of a camera frame put together. The same span written here is a short copy
 * repeated, and the pattern -- one pixel, so one to four numbers -- stays in
 * registers across the whole run.
 *
 * The list is written in place and handed back, so the caller keeps whatever
 * else it holds. Anything that would run past the end is refused rather than
 * clipped: a caller that computed the extent wrongly wants to hear about it.
 */
static bool primListFillPattern(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *values;
    ObjList *pattern;
    int64_t start, repeats;
    if (!jaiArgList(args[0], 1, "list_fill_pattern", &values)) return false;
    if (!jaiStrWantInt(args[1], "list_fill_pattern", "the start", &start)) return false;
    if (!jaiStrWantInt(args[2], "list_fill_pattern", "the repeat count", &repeats)) return false;
    if (!jaiArgList(args[3], 4, "list_fill_pattern", &pattern)) return false;

    const int64_t width = pattern->count;
    if (start < 0 || repeats < 0) {
        return jaiThrow(vm.cValueError,
                        "list_fill_pattern(): a start and a count cannot be negative, got %lld and %lld",
                        (long long)start, (long long)repeats);
    }
    if (width == 0 || repeats == 0) {
        *out = OBJ_VAL(values);
        return true;
    }
    if (repeats > (INT64_MAX / width) || start + repeats * width > values->count) {
        return jaiThrow(vm.cValueError,
                        "list_fill_pattern(): %lld copies of %lld from %lld runs past a list of %d",
                        (long long)repeats, (long long)width, (long long)start, values->count);
    }

    Value *write = values->items + start;
    if (width == 1) {
        const Value only = pattern->items[0];
        for (int64_t i = 0; i < repeats; i++) write[i] = only;
    } else if (width == 3) {
        const Value a = pattern->items[0], b = pattern->items[1], c = pattern->items[2];
        for (int64_t i = 0; i < repeats; i++) {
            write[0] = a;
            write[1] = b;
            write[2] = c;
            write += 3;
        }
    } else if (width == 4) {
        const Value a = pattern->items[0], b = pattern->items[1];
        const Value c = pattern->items[2], d = pattern->items[3];
        for (int64_t i = 0; i < repeats; i++) {
            write[0] = a;
            write[1] = b;
            write[2] = c;
            write[3] = d;
            write += 4;
        }
    } else {
        for (int64_t i = 0; i < repeats; i++) {
            memcpy(write, pattern->items, (size_t)width * sizeof(Value));
            write += width;
        }
    }
    values->version++;
    *out = OBJ_VAL(values);
    return true;
}

/* `list_push3(values, a, b, c)` -- append three numbers in one call.
 *
 * A rasteriser batching rows for the device appends one of these per row, and
 * a row of a thin line is one pixel, so this is once per pixel along every
 * stroke and every letter. Three `push` calls from Jaithon are three bounds
 * checks and three chances to grow; done here the capacity is settled once.
 */
static bool primListPush3(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *values;
    if (!jaiArgList(args[0], 1, "list_push3", &values)) return false;
    jaiListReserve(values, values->count + 3);
    if (values->capacity < values->count + 3) return false;
    values->items[values->count] = args[1];
    values->items[values->count + 1] = args[2];
    values->items[values->count + 2] = args[3];
    values->count += 3;
    values->version++;
    *out = OBJ_VAL(values);
    return true;
}

/* The pixels of one clipped, normalised line, in the order a rasteriser walks
 * them. Mirrors `walk` in jaicv's drawing module exactly, down to the error
 * term, so a line comes out on the same pixels either way.
 *
 * The Jaithon version built a point object per pixel and then wrote them one
 * at a time. A rectangle two pixels thick over a 720p frame is about four
 * thousand of those, and at roughly a hundred nanoseconds each the outlines
 * of a detector's boxes cost more than the network that found them.
 *
 * `emit` is what to do with a pixel, which is the only thing that differs
 * between drawing on the host and batching rows for the device. */
typedef struct {
    int x, y;
    int dx, dy;
    int majorX, majorY;
    int minorX, minorY;
    int connectivity;
} JaiLineWalk;

static bool lineWalk(const JaiLineWalk *w, bool (*emit)(void *at, int x, int y), void *at) {
    int x = w->x;
    int y = w->y;
    if (w->connectivity == 4) {
        int err = 0;
        const int plus = w->dx + w->dx + w->dy + w->dy;
        const int minus = -(w->dy + w->dy);
        const int steps = w->dx + w->dy + 1;
        for (int i = 0; i < steps; i++) {
            if (!emit(at, x, y)) return false;
            const bool sideways = err < 0;
            err += minus + (sideways ? plus : 0);
            x += sideways ? w->minorX : w->majorX;
            y += sideways ? w->minorY : w->majorY;
        }
        return true;
    }
    int err = w->dx - (w->dy + w->dy);
    const int plus = w->dx + w->dx;
    const int minus = -(w->dy + w->dy);
    const int steps = w->dx + 1;
    for (int i = 0; i < steps; i++) {
        if (!emit(at, x, y)) return false;
        const bool diagonal = err < 0;
        err += minus + (diagonal ? plus : 0);
        x += w->majorX + (diagonal ? w->minorX : 0);
        y += w->majorY + (diagonal ? w->minorY : 0);
    }
    return true;
}

/* Read the eight numbers every line primitive here takes, plus the frame it is
 * being clipped against. */
static bool readWalk(Value *args, int first, const char *fnName, JaiLineWalk *w,
                     int64_t *cols, int64_t *rows) {
    int64_t got[11];
    for (int i = 0; i < 11; i++) {
        if (!jaiStrWantInt(args[first + i], fnName, "a coordinate", &got[i])) return false;
    }
    w->x = (int)got[0];
    w->y = (int)got[1];
    w->dx = (int)got[2];
    w->dy = (int)got[3];
    w->majorX = (int)got[4];
    w->majorY = (int)got[5];
    w->minorX = (int)got[6];
    w->minorY = (int)got[7];
    w->connectivity = (int)got[8];
    *cols = got[9];
    *rows = got[10];
    if (w->dx < 0 || w->dy < 0) {
        return jaiThrow(vm.cValueError, "%s(): the deltas must not be negative", fnName);
    }
    return true;
}

typedef struct {
    ObjList *pending;
    int64_t  cols;
    int64_t  rows;
    int      count;
    bool     full;
    int64_t  widest;
} JaiSpanSink;

/* One row of one pixel per point, in the triples the span filler reads. */
static bool spanSink(void *at, int x, int y) {
    JaiSpanSink *sink = (JaiSpanSink *)at;
    if (x < 0 || y < 0 || x >= sink->cols || y >= sink->rows) return true;
    ObjList *list = sink->pending;
    jaiListReserve(list, list->count + 3);
    if (list->capacity < list->count + 3) {
        sink->full = true;
        return false;
    }
    list->items[list->count] = FLOAT_VAL((double)y);
    list->items[list->count + 1] = FLOAT_VAL((double)x);
    list->items[list->count + 2] = FLOAT_VAL(1.0);
    list->count += 3;
    sink->count++;
    return true;
}

/* `line_spans(pending, x, y, dx, dy, majorX, majorY, minorX, minorY,
 *  connectivity, cols, rows)` -- append one row per pixel of the line, and
 *  say how many. */
static bool primLineSpans(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *pending;
    if (!jaiArgList(args[0], 1, "line_spans", &pending)) return false;
    JaiLineWalk walk;
    int64_t cols, rows;
    if (!readWalk(args, 1, "line_spans", &walk, &cols, &rows)) return false;

    JaiSpanSink sink = {pending, cols, rows, 0, false, 1};
    lineWalk(&walk, spanSink, &sink);
    if (sink.full) return jaiThrow(vm.cRuntimeError, "line_spans(): out of room for the line");
    pending->version++;
    *out = INT_VAL(sink.count);
    return true;
}

typedef struct {
    ObjList *values;
    const Value *colour;
    int64_t cols;
    int64_t rows;
    int     cn;
} JaiPaintSink;

/* Straight into the host mirror, the same write `plot` makes. */
static bool paintSink(void *at, int x, int y) {
    JaiPaintSink *sink = (JaiPaintSink *)at;
    if (x < 0 || y < 0 || x >= sink->cols || y >= sink->rows) return true;
    const int64_t base = ((int64_t)y * sink->cols + x) * sink->cn;
    if (base < 0 || base + sink->cn > sink->values->count) return true;
    for (int c = 0; c < sink->cn; c++) sink->values->items[base + c] = sink->colour[c];
    return true;
}

/* `line_paint(values, colour, cn, x, y, dx, dy, majorX, majorY, minorX,
 *  minorY, connectivity, cols, rows)` -- write the line into a host mirror. */
static bool primLinePaint(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *values;
    ObjList *colour;
    int64_t channels;
    if (!jaiArgList(args[0], 1, "line_paint", &values)) return false;
    if (!jaiArgList(args[1], 2, "line_paint", &colour)) return false;
    if (!jaiStrWantInt(args[2], "line_paint", "the channel count", &channels)) return false;
    if (channels <= 0 || channels != colour->count) {
        return jaiThrow(vm.cValueError,
                        "line_paint(): %lld channels against a colour of %d",
                        (long long)channels, colour->count);
    }
    JaiLineWalk walk;
    int64_t cols, rows;
    if (!readWalk(args, 3, "line_paint", &walk, &cols, &rows)) return false;

    JaiPaintSink sink = {values, colour->items, cols, rows, (int)channels};
    lineWalk(&walk, paintSink, &sink);
    values->version++;
    *out = OBJ_VAL(values);
    return true;
}

/* The scanline fill of a convex polygon, in fixed point.
 *
 * Mirrors `fill_convex` in jaicv's drawing module: the same two edge chains
 * walked from the topmost vertex, the same truncating division for the slope,
 * the same rounding of each end of a row. OpenCV's arithmetic, so the spans
 * land on OpenCV's pixels.
 *
 * Every filled shape and every stroke wider than one pixel is made of these,
 * and a row of one cost about half a microsecond written in Jaithon -- a
 * rectangle two pixels thick over a 720p frame is some six hundred rows, which
 * is most of what drawing a frame of detections cost. */
#define JAI_XY_SHIFT 16
#define JAI_XY_ONE   (1 << JAI_XY_SHIFT)

static int64_t truncDiv(int64_t a, int64_t b) {
    int64_t quotient = a / b;
    /* C already truncates toward zero, which is what the Jaithon helper's
     * floor-plus-correction comes to. */
    return quotient;
}

typedef struct {
    const int64_t *xs;
    const int64_t *ys;
    int      count;
    int      shift;
    int64_t  delta1;
    int64_t  delta2;
    int64_t  cols;
    int64_t  rows;
} JaiConvexFill;

static void convexFill(const JaiConvexFill *poly,
                       void (*row)(void *at, int64_t y, int64_t left, int64_t right),
                       void *at) {
    const int count = poly->count;
    if (count < 3) return;
    const int64_t delta = (int64_t)1 << poly->shift >> 1;

    int64_t xMin = poly->xs[0], xMax = poly->xs[0];
    int64_t yMin = poly->ys[0], yMax = poly->ys[0];
    int top = 0;
    for (int i = 0; i < count; i++) {
        if (poly->ys[i] < yMin) { yMin = poly->ys[i]; top = i; }
        if (poly->ys[i] > yMax) yMax = poly->ys[i];
        if (poly->xs[i] > xMax) xMax = poly->xs[i];
        if (poly->xs[i] < xMin) xMin = poly->xs[i];
    }
    xMin = (xMin + delta) >> poly->shift;
    xMax = (xMax + delta) >> poly->shift;
    yMin = (yMin + delta) >> poly->shift;
    yMax = (yMax + delta) >> poly->shift;
    if (xMax < 0 || yMax < 0 || xMin >= poly->cols || yMin >= poly->rows) return;
    if (yMax > poly->rows - 1) yMax = poly->rows - 1;

    int indexOf[2] = {top, top};
    const int direction[2] = {1, count - 1};
    int64_t endY[2] = {yMin, yMin};
    int64_t edgeX[2] = {-JAI_XY_ONE, -JAI_XY_ONE};
    int64_t edgeDx[2] = {0, 0};
    int remaining = count;

    for (int64_t y = yMin; y <= yMax; y++) {
        bool exhausted = false;
        for (int side = 0; side < 2; side++) {
            if (y < endY[side]) continue;
            int idx = indexOf[side];
            const int di = direction[side];
            int64_t xs = 0;
            int64_t ty = 0;
            for (;;) {
                ty = (poly->ys[idx] + delta) >> poly->shift;
                if (ty > y || remaining == 0) break;
                xs = poly->xs[idx];
                idx += di;
                if (idx >= count) idx -= count;
                remaining--;
            }
            if (y >= ty) { exhausted = true; break; }
            endY[side] = ty;
            const int64_t start = xs << (JAI_XY_SHIFT - poly->shift);
            const int64_t finish = poly->xs[idx] << (JAI_XY_SHIFT - poly->shift);
            edgeDx[side] = truncDiv((finish - start) * 2 + (ty - y), 2 * (ty - y));
            edgeX[side] = start;
            indexOf[side] = idx;
        }
        if (y >= 0) {
            const int left = edgeX[0] > edgeX[1] ? 1 : 0;
            const int right = 1 - left;
            row(at, y, (edgeX[left] + poly->delta1) >> JAI_XY_SHIFT,
                (edgeX[right] + poly->delta2) >> JAI_XY_SHIFT);
        }
        if (exhausted) return;
        edgeX[0] += edgeDx[0];
        edgeX[1] += edgeDx[1];
    }
}

/* Clip a row to the frame and hand back what is left, or nothing. */
static bool rowInside(int64_t cols, int64_t *left, int64_t right, int64_t *run) {
    if (*left < 0) *left = 0;
    if (right > cols - 1) right = cols - 1;
    if (*left > right) return false;
    *run = right - *left + 1;
    return true;
}

static void convexSpanRow(void *at, int64_t y, int64_t left, int64_t right) {
    JaiSpanSink *sink = (JaiSpanSink *)at;
    if (sink->full || y < 0 || y >= sink->rows) return;
    int64_t run;
    if (!rowInside(sink->cols, &left, right, &run)) return;
    ObjList *list = sink->pending;
    jaiListReserve(list, list->count + 3);
    if (list->capacity < list->count + 3) { sink->full = true; return; }
    list->items[list->count] = FLOAT_VAL((double)y);
    list->items[list->count + 1] = FLOAT_VAL((double)left);
    list->items[list->count + 2] = FLOAT_VAL((double)run);
    list->count += 3;
    sink->count++;
    if (run > sink->widest) sink->widest = run;
}

static void convexPaintRow(void *at, int64_t y, int64_t left, int64_t right) {
    JaiPaintSink *sink = (JaiPaintSink *)at;
    if (y < 0 || y >= sink->rows) return;
    int64_t run;
    if (!rowInside(sink->cols, &left, right, &run)) return;
    int64_t base = ((int64_t)y * sink->cols + left) * sink->cn;
    if (base < 0 || base + run * sink->cn > sink->values->count) return;
    for (int64_t i = 0; i < run; i++) {
        for (int c = 0; c < sink->cn; c++) sink->values->items[base + c] = sink->colour[c];
        base += sink->cn;
    }
}

/* Read the polygon's corners, which arrive as one flat list of x then y. */
static bool readCorners(Value v, int index, const char *fnName, ObjList **flat, int *count) {
    if (!jaiArgList(v, index, fnName, flat)) return false;
    if ((*flat)->count % 2 != 0) {
        return jaiThrow(vm.cValueError, "%s(): %d numbers is not a list of corners",
                        fnName, (*flat)->count);
    }
    *count = (*flat)->count / 2;
    return true;
}

static bool cornersInto(ObjList *flat, int count, int64_t *xs, int64_t *ys,
                        const char *fnName) {
    for (int i = 0; i < count; i++) {
        Value vx = flat->items[i * 2];
        Value vy = flat->items[i * 2 + 1];
        if (!IS_INT(vx) || !IS_INT(vy)) {
            return jaiThrow(vm.cTypeError, "%s(): a corner is not a pair of integers", fnName);
        }
        xs[i] = AS_INT(vx);
        ys[i] = AS_INT(vy);
    }
    return true;
}

#define JAI_CONVEX_MAX_CORNERS 4096

/* `convex_spans(pending, corners, shift, delta1, delta2, cols, rows)` */
static bool primConvexSpans(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *pending;
    ObjList *flat;
    int count;
    int64_t shift, delta1, delta2, cols, rows;
    if (!jaiArgList(args[0], 1, "convex_spans", &pending)) return false;
    if (!readCorners(args[1], 2, "convex_spans", &flat, &count)) return false;
    if (!jaiStrWantInt(args[2], "convex_spans", "the shift", &shift)) return false;
    if (!jaiStrWantInt(args[3], "convex_spans", "the near rounding", &delta1)) return false;
    if (!jaiStrWantInt(args[4], "convex_spans", "the far rounding", &delta2)) return false;
    if (!jaiStrWantInt(args[5], "convex_spans", "the width", &cols)) return false;
    if (!jaiStrWantInt(args[6], "convex_spans", "the height", &rows)) return false;
    if (count < 3 || count > JAI_CONVEX_MAX_CORNERS || shift < 0 || shift > JAI_XY_SHIFT) {
        *out = INT_VAL(0);
        return true;
    }

    int64_t xs[JAI_CONVEX_MAX_CORNERS];
    int64_t ys[JAI_CONVEX_MAX_CORNERS];
    if (!cornersInto(flat, count, xs, ys, "convex_spans")) return false;

    JaiConvexFill poly = {xs, ys, count, (int)shift, delta1, delta2, cols, rows};
    JaiSpanSink sink = {pending, cols, rows, 0, false, 0};
    convexFill(&poly, convexSpanRow, &sink);
    if (sink.full) return jaiThrow(vm.cRuntimeError, "convex_spans(): out of room for the shape");
    pending->version++;
    ObjList *told = jaiListNew(2);
    if (told == NULL) return false;
    if (!jaiListReserveExact(told, 2)) return false;
    told->items[0] = INT_VAL(sink.count);
    told->items[1] = INT_VAL(sink.widest);
    told->count = 2;
    told->version++;
    *out = OBJ_VAL(told);
    return true;
}

/* `convex_paint(values, colour, cn, corners, shift, delta1, delta2, cols, rows)` */
static bool primConvexPaint(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *values;
    ObjList *colour;
    ObjList *flat;
    int count;
    int64_t channels, shift, delta1, delta2, cols, rows;
    if (!jaiArgList(args[0], 1, "convex_paint", &values)) return false;
    if (!jaiArgList(args[1], 2, "convex_paint", &colour)) return false;
    if (!jaiStrWantInt(args[2], "convex_paint", "the channel count", &channels)) return false;
    if (!readCorners(args[3], 4, "convex_paint", &flat, &count)) return false;
    if (!jaiStrWantInt(args[4], "convex_paint", "the shift", &shift)) return false;
    if (!jaiStrWantInt(args[5], "convex_paint", "the near rounding", &delta1)) return false;
    if (!jaiStrWantInt(args[6], "convex_paint", "the far rounding", &delta2)) return false;
    if (!jaiStrWantInt(args[7], "convex_paint", "the width", &cols)) return false;
    if (!jaiStrWantInt(args[8], "convex_paint", "the height", &rows)) return false;
    if (channels <= 0 || channels != colour->count) {
        return jaiThrow(vm.cValueError,
                        "convex_paint(): %lld channels against a colour of %d",
                        (long long)channels, colour->count);
    }
    if (count < 3 || count > JAI_CONVEX_MAX_CORNERS || shift < 0 || shift > JAI_XY_SHIFT) {
        *out = OBJ_VAL(values);
        return true;
    }

    int64_t xs[JAI_CONVEX_MAX_CORNERS];
    int64_t ys[JAI_CONVEX_MAX_CORNERS];
    if (!cornersInto(flat, count, xs, ys, "convex_paint")) return false;

    JaiConvexFill poly = {xs, ys, count, (int)shift, delta1, delta2, cols, rows};
    JaiPaintSink sink = {values, colour->items, cols, rows, (int)channels};
    convexFill(&poly, convexPaintRow, &sink);
    values->version++;
    *out = OBJ_VAL(values);
    return true;
}

/* One channel of a pixel, rounded and clamped the way a display wants it.
 *
 * `!(value > 0.0)` rather than `value < 0.0` so that a NaN -- which compares
 * false against everything -- lands on black instead of an undefined cast.
 * Anything that is not a number lands there too, which is what a caller who
 * passed one deserves and is cheaper than checking the whole list first. */
static uint32_t packChannel(Value v) {
    if (!IS_FLOAT(v) && !IS_INT(v)) return 0u;
    const double value = IS_FLOAT(v) ? AS_FLOAT(v) : (double)AS_INT(v);
    if (!(value > 0.0)) return 0u;
    const double rounded = value + 0.5;
    if (rounded > 255.0) return 255u;
    return (uint32_t)rounded;
}

/* `list_pack_argb(values, channels)` -- interleaved float channels to one
 * `0xAARRGGBB` int per pixel.
 *
 * The pixels come either as a list of floats, which are rounded and clamped
 * on the way through, or as `bytes` that are already in range -- which is what
 * `Mat.to_bytes` hands over, and it skips both the boxing and the arithmetic.
 *
 * `channels` is 1 for grey, 3 for BGR, 4 for BGRA; anything else is refused.
 * Grey fans the one value across the three colour channels, and a source
 * without an alpha channel gets an opaque one.
 *
 * This is what stands between a camera and its preview. Written as a Jaithon
 * loop it ran nine hundred thousand iterations and three and a half million
 * calls for one 720p frame -- about 18 ms, more than the whole rest of the
 * loop -- and it is the same arithmetic every time, so it belongs here. */
#define JAI_PACK_CHUNK 16384

typedef struct {
    const uint8_t *bytes;
    const Value   *values;
    Value         *out;
    int            channels;
} PackWork;

/* Bytes are already in range, so that path is a shuffle with no arithmetic at
 * all -- which is the whole reason to hand pixels over as bytes rather than as
 * a list of floats. */
static void packRange(void *context, size_t start, size_t end) {
    const PackWork *work = (const PackWork *)context;
    const int channels = work->channels;
    for (size_t pixel = start; pixel < end; pixel++) {
        uint32_t blue, green, red, alpha = 255u;
        if (work->bytes != NULL) {
            const uint8_t *channel = work->bytes + pixel * (size_t)channels;
            blue = channel[0];
            green = blue;
            red = blue;
            if (channels != 1) {
                green = channel[1];
                red = channel[2];
                if (channels == 4) alpha = channel[3];
            }
        } else {
            const Value *channel = &work->values[pixel * (size_t)channels];
            blue = packChannel(channel[0]);
            green = blue;
            red = blue;
            if (channels != 1) {
                green = packChannel(channel[1]);
                red = packChannel(channel[2]);
                if (channels == 4) alpha = packChannel(channel[3]);
            }
        }
        work->out[pixel] =
            INT_VAL((int64_t)((alpha << 24) | (red << 16) | (green << 8) | blue));
    }
}

static bool primListPackArgb(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *values = NULL;
    ObjBytes *raw = NULL;
    int64_t supplied;
    if (IS_BYTES(args[0])) {
        raw = AS_BYTES(args[0]);
        supplied = (int64_t)raw->length;
    } else if (jaiArgList(args[0], 1, "list_pack_argb", &values)) {
        supplied = (int64_t)values->count;
    } else {
        return false;
    }

    int64_t channels;
    if (!jaiStrWantInt(args[1], "list_pack_argb", "the channel count", &channels)) return false;
    if (channels != 1 && channels != 3 && channels != 4) {
        return jaiThrow(vm.cValueError,
                        "list_pack_argb(): channels must be 1, 3 or 4, got %lld",
                        (long long)channels);
    }
    if (supplied % channels != 0) {
        return jaiThrow(vm.cValueError,
                        "list_pack_argb(): %lld values is not a whole number of "
                        "%lld-channel pixels",
                        (long long)supplied, (long long)channels);
    }

    const int64_t count = supplied / channels;
    ObjList *list = jaiListNew((int)count);
    if (list == NULL) return false;
    /* Rooted across the reserve: growing the backing array can collect. */
    jaiGCPushRoot(OBJ_VAL(list));
    jaiListReserve(list, (int)count);
    jaiGCPopRoot();
    if (count > 0 && list->capacity < (int)count) return false;

    PackWork work = {raw != NULL ? raw->data : NULL,
                     values != NULL ? values->items : NULL,
                     list->items, (int)channels};
    jaiParallelChunks((size_t)count, JAI_PACK_CHUNK, packRange, &work);
    list->count = (int)count;
    list->version++;
    *out = OBJ_VAL(list);
    return true;
}

void jaiBytesRegisterPrimitives(ObjModule *ns) {
    jaiStrDefinePrim(ns, "bytes_quantise", primBytesQuantise, 1, 2);
    jaiStrDefinePrim(ns, "list_filled",    primListFilled,    1, 2);
    jaiStrDefinePrim(ns, "list_pack_argb", primListPackArgb,  2, 2);
    jaiStrDefinePrim(ns, "list_fill_pattern", primListFillPattern, 4, 4);
    jaiStrDefinePrim(ns, "list_push3",     primListPush3,     4, 4);
    jaiStrDefinePrim(ns, "line_spans",     primLineSpans,    12, 12);
    jaiStrDefinePrim(ns, "line_paint",     primLinePaint,    14, 14);
    jaiStrDefinePrim(ns, "convex_spans",   primConvexSpans,   7, 7);
    jaiStrDefinePrim(ns, "convex_paint",   primConvexPaint,   9, 9);
    jaiStrDefinePrim(ns, "bytes_new",      primBytesNew,      1, 1);
    jaiStrDefinePrim(ns, "bytes_len",      primBytesLen,      1, 1);
    jaiStrDefinePrim(ns, "bytes_get",      primBytesGet,      2, 2);
    jaiStrDefinePrim(ns, "bytes_slice",    primBytesSlice,    3, 3);
    jaiStrDefinePrim(ns, "bytes_concat",   bytesConcat,       2, 2);
    jaiStrDefinePrim(ns, "bytes_cmp",      primBytesCmp,      2, 2);
    jaiStrDefinePrim(ns, "bytes_find",     bytesFind,         2, 4);
    jaiStrDefinePrim(ns, "bytes_to_list",  bytesToList,       1, 1);
    jaiStrDefinePrim(ns, "bytes_hex",      bytesHex,          1, 1);
    jaiStrDefinePrim(ns, "bytes_from_hex", primBytesFromHex,  1, 1);
}
