/* builtins_bytes.c — the `bytes` method table and `__prim__.bytes_*` surface (spec Appendix C); every index/length/window here counts bytes, not codepoints. */

#include <math.h>

#include "runtime/builtins/text/builtins_str.h"
#include "runtime/methods.h"
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

    for (size_t i = 0; i < n; i++) {
        Value item = items->items[i];
        double value;
        if (IS_FLOAT(item)) value = AS_FLOAT(item);
        else if (IS_INT(item)) value = (double)AS_INT(item);
        else {
            return jaiThrow(vm.cTypeError,
                            "bytes_quantise(): element %zu is %s, not a number",
                            i, jaiTypeNameStatic(item));
        }
        value *= scale;
        /* NaN fails every comparison, so it lands here and nowhere else. */
        if (!(value == value)) { dst[i] = 0; continue; }
        double rounded = floor(value + 0.5);
        if (rounded <= 0.0) dst[i] = 0;
        else if (rounded >= 255.0) dst[i] = 255;
        else dst[i] = (uint8_t)rounded;
    }
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

    /* Bytes are already in range, so that path is a shuffle with no
     * arithmetic at all -- which is the whole reason to hand pixels over as
     * bytes rather than as a list of floats. */
    if (raw != NULL) {
        for (int64_t pixel = 0; pixel < count; pixel++) {
            const uint8_t *channel = raw->data + pixel * channels;
            uint32_t blue = channel[0], green = blue, red = blue, alpha = 255u;
            if (channels != 1) {
                green = channel[1];
                red = channel[2];
                if (channels == 4) alpha = channel[3];
            }
            list->items[pixel] =
                INT_VAL((int64_t)((alpha << 24) | (red << 16) | (green << 8) | blue));
        }
    } else {
        for (int64_t pixel = 0; pixel < count; pixel++) {
            const Value *channel = &values->items[pixel * channels];
            uint32_t blue, green, red, alpha = 255u;
            if (channels == 1) {
                blue = packChannel(channel[0]);
                green = blue;
                red = blue;
            } else {
                blue = packChannel(channel[0]);
                green = packChannel(channel[1]);
                red = packChannel(channel[2]);
                if (channels == 4) alpha = packChannel(channel[3]);
            }
            list->items[pixel] =
                INT_VAL((int64_t)((alpha << 24) | (red << 16) | (green << 8) | blue));
        }
    }
    list->count = (int)count;
    list->version++;
    *out = OBJ_VAL(list);
    return true;
}

void jaiBytesRegisterPrimitives(ObjModule *ns) {
    jaiStrDefinePrim(ns, "bytes_quantise", primBytesQuantise, 1, 2);
    jaiStrDefinePrim(ns, "list_filled",    primListFilled,    1, 2);
    jaiStrDefinePrim(ns, "list_pack_argb", primListPackArgb,  2, 2);
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
