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
    for (uint32_t i = 0; i < b->length; i++) jaiListPut(list, i, INT_VAL(b->data[i]));
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
        if (!IS_INT(jaiListGet(list, i)) || AS_INT(jaiListGet(list, i)) < 0 ||
            AS_INT(jaiListGet(list, i)) > 255) {
            return jaiThrow(vm.cValueError,
                            "bytes_new(): item %d is not a byte value (0 to 255)",
                            i);
        }
    }
    ObjBytes *b = jaiBytesNew(NULL, (size_t)list->count);
    if (b == NULL) return false;
    for (int i = 0; i < list->count; i++) b->data[i] = (uint8_t)AS_INT(jaiListGet(list, i));
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

/* `grid_nonzero(dst, src, rows, cols, stride, origin)` -- scatter a source
 * image into a padded destination as ones and zeros.
 *
 * `dst[origin + y * stride + x] = src[y * cols + x] != 0 ? 1 : 0`, which is
 * what every border follower, flood fill and connected-components pass in this
 * repository starts by building: a binary copy of the picture sitting inside a
 * ring of zeros, so the walk never has to ask whether a neighbour exists.
 *
 * It is here rather than in the library because the library's version is at the
 * language's floor and the floor is too high. Written as the obvious two loops
 * it is 4.95 ms for a 1080p frame -- 2.4 ns a cell for a load, a compare and a
 * store, which is what a compiled loop with two bounds checks and a boxed store
 * costs. Reading the source as `bytes` instead of `list[float]` measured 4.82
 * ms, so it is not the read: it is the per-element boxing and checking, and no
 * amount of rewriting inside the language removes that. This is the same
 * reasoning `bytes_quantise` and `list_filled` were added under.
 *
 * The source may be a `list` of numbers or `bytes`; the destination is a list
 * of ints long enough to hold the padded layout, which the caller allocates
 * once and reuses.
 */
/* Both sides are read through the list's own storage rather than as a `Value`
 * array, because either may be unboxed: a `list[int]` destination is an
 * `int64_t[]`, which is the layout this primitive wants anyway -- half the
 * stores and half the cache lines of the boxed form. The source is whichever
 * of the four a `list[float]`, a `list[int]` or an untyped list ended up as.
 *
 * The kinds are read ONCE, into this struct, rather than switched per element:
 * a list cannot change its storage while this runs (nothing here allocates or
 * calls back into the VM), so the branch belongs outside the loop. */
typedef struct {
    const void    *src;
    uint8_t        srcStore;
    const uint8_t *srcBytes;
    void          *dst;
    uint8_t        dstStore;
    size_t         cols;
    size_t         stride;
    size_t         origin;
} GridNonzeroWork;

/* Whether source element `i` is non-zero, whatever the source is made of. */
JAI_INLINE bool gridSourceSet(const GridNonzeroWork *work, size_t i) {
    if (work->srcBytes != NULL) return work->srcBytes[i] != 0;
    switch ((ListStore)work->srcStore) {
    case LIST_STORE_I64: return ((const int64_t *)work->src)[i] != 0;
    /* A NaN is not zero, and comparing it against zero says so correctly --
     * but only because the test is `!= 0`, not `> 0`. */
    case LIST_STORE_F64: return ((const double *)work->src)[i] != 0.0;
    case LIST_STORE_U8:  return ((const uint8_t *)work->src)[i] != 0;
    case LIST_STORE_BOXED: break;
    }
    const Value item = ((const Value *)work->src)[i];
    return IS_FLOAT(item) ? (AS_FLOAT(item) != 0.0) : (AS_INT(item) != 0);
}

static void gridNonzeroRows(void *context, size_t start, size_t end) {
    const GridNonzeroWork *work = (const GridNonzeroWork *)context;
    for (size_t y = start; y < end; y++) {
        const size_t from = y * work->cols;
        const size_t at = work->origin + y * work->stride;
        if ((ListStore)work->dstStore == LIST_STORE_I64) {
            int64_t *row = (int64_t *)work->dst + at;
            for (size_t x = 0; x < work->cols; x++) {
                row[x] = gridSourceSet(work, from + x) ? 1 : 0;
            }
            continue;
        }
        Value *row = (Value *)work->dst + at;
        for (size_t x = 0; x < work->cols; x++) {
            row[x] = INT_VAL(gridSourceSet(work, from + x) ? 1 : 0);
        }
    }
}

static bool primGridNonzero(int argc, Value *args, Value *out) {
    (void)argc;
    if (!IS_LIST(args[0])) {
        return jaiThrow(vm.cTypeError,
                        "grid_nonzero(): the destination must be a list, got %s",
                        jaiTypeNameStatic(args[0]));
    }
    const void *src = NULL;
    uint8_t srcStore = LIST_STORE_BOXED;
    const uint8_t *srcBytes = NULL;
    size_t srcCount = 0;
    if (IS_LIST(args[1])) {
        ObjList *items = AS_LIST(args[1]);
        src = items->items;
        srcStore = items->stg;
        srcCount = (size_t)items->count;
        /* Only a boxed source can hold something that is not a number; an
         * unboxed one is numbers by construction, so the pass is skipped
         * rather than boxing every element to ask. */
        if ((ListStore)srcStore == LIST_STORE_BOXED) {
            const Value *boxed = (const Value *)src;
            for (size_t i = 0; i < srcCount; i++) {
                if (!IS_FLOAT(boxed[i]) && !IS_INT(boxed[i])) {
                    return jaiThrow(vm.cTypeError,
                                    "grid_nonzero(): source element %zu is %s, not a number",
                                    i, jaiTypeNameStatic(boxed[i]));
                }
            }
        }
    } else if (IS_BYTES(args[1])) {
        ObjBytes *raw = AS_BYTES(args[1]);
        srcBytes = raw->data;
        srcCount = (size_t)raw->length;
    } else {
        return jaiThrow(vm.cTypeError,
                        "grid_nonzero(): the source must be a list or bytes, got %s",
                        jaiTypeNameStatic(args[1]));
    }

    int64_t rows, cols, stride, origin;
    if (!jaiStrWantInt(args[2], "grid_nonzero", "the row count", &rows)) return false;
    if (!jaiStrWantInt(args[3], "grid_nonzero", "the column count", &cols)) return false;
    if (!jaiStrWantInt(args[4], "grid_nonzero", "the destination stride", &stride)) return false;
    if (!jaiStrWantInt(args[5], "grid_nonzero", "the destination origin", &origin)) return false;
    if (rows < 0 || cols < 0 || stride < 0 || origin < 0) {
        return jaiThrow(vm.cValueError,
                        "grid_nonzero(): rows, cols, stride and origin cannot be negative");
    }
    if (cols > stride) {
        return jaiThrow(vm.cValueError,
                        "grid_nonzero(): %lld columns do not fit a stride of %lld",
                        (long long)cols, (long long)stride);
    }
    if ((size_t)(rows * cols) > srcCount) {
        return jaiThrow(vm.cValueError,
                        "grid_nonzero(): %lldx%lld needs %lld source values, got %zu",
                        (long long)rows, (long long)cols,
                        (long long)(rows * cols), srcCount);
    }
    ObjList *dst = AS_LIST(args[0]);
    /* The last cell written, which is what has to fit -- not the whole
     * rectangle, since the final row need not be padded. */
    int64_t reach = rows == 0 ? 0 : origin + (rows - 1) * stride + cols;
    if (reach > (int64_t)dst->count) {
        return jaiThrow(vm.cValueError,
                        "grid_nonzero(): the destination holds %d values, %lld are needed",
                        dst->count, (long long)reach);
    }

    /* Anything but a boxed or int-backed destination is boxed first: a float
     * or bool store cannot hold the ones and zeros this writes, and the
     * alternative -- a third arm -- would be code nothing calls. */
    if ((ListStore)dst->stg != LIST_STORE_BOXED &&
        (ListStore)dst->stg != LIST_STORE_I64) {
        jaiListBox(dst);
        /* jaiListBox frees the old array, and `src` was read before it ran.
         * grid_nonzero(xs, xs, ...) is one list, so that pointer is the one
         * just handed back to the allocator -- re-read it rather than hand a
         * freed block to the worker threads. */
        if (IS_LIST(args[1])) {
            ObjList *items = AS_LIST(args[1]);
            src = items->items;
            srcStore = items->stg;
        }
    }
    GridNonzeroWork work = {
        src, srcStore, srcBytes, dst->items, dst->stg,
        (size_t)cols, (size_t)stride, (size_t)origin,
    };
    /* Chunked by ROW, not by cell: a chunk boundary inside a row would have to
     * carry the row's base offsets, and a 1080p frame is a thousand rows --
     * enough work units for every core. */
    jaiParallelChunks((size_t)rows, cols > 0 ? (JAI_QUANTISE_CHUNK / (size_t)cols) + 1 : 1,
                      gridNonzeroRows, &work);
    dst->version++;
    *out = NULL_VAL;
    return true;
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
        if (!IS_FLOAT(jaiListGet(items, i)) && !IS_INT(jaiListGet(items, i))) {
            return jaiThrow(vm.cTypeError,
                            "bytes_quantise(): element %zu is %s, not a number",
                            i, jaiTypeNameStatic(jaiListGet(items, i)));
        }
    }

    QuantiseWork work = {jaiListBox(items), dst, (float)scale};
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

    for (int64_t i = 0; i < count; i++) jaiListPut(list, i, fill);
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
                     values != NULL ? jaiListBox(values) : NULL,
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
    jaiStrDefinePrim(ns, "grid_nonzero",   primGridNonzero,   6, 6);
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
