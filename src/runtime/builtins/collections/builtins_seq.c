/* builtins_seq.c — tuple, range and iterator methods, plus the pieces shared
 * with builtins_list.c and builtins_dict.c.
 *
 * The receiver arrives in args[0] (counted in argc); args[1] is the user's
 * first argument, reported as position 1. Anything that can re-enter Jaithon
 * (a callback, a user __eq__ or __hash__) can allocate and collect, so live
 * results stay GC-rooted and container bounds are re-read afterward.
 * sort/min/max copy the receiver before running a callback that might mutate
 * it. A mutator with no natural result returns the receiver so calls chain.
 */

#include <math.h>
#include <string.h>

#include "runtime/builtins/collections/builtins_seq.h"
#include "runtime/methods.h"
#include "runtime/runtime.h"

#include "vm/gc.h"

/* ------------------------------------------------------------------ */
/* Stable sort over an index permutation                                */
/* ------------------------------------------------------------------ */

/* A double carries every whole number up to this exactly. */
#define SORT_WHOLE_EXACT 9007199254740992LL

/* Below this a radix pass costs more in histograms than it saves. */
#define SORT_RADIX_FLOOR 128

#define SORT_RADIX_BITS   8
#define SORT_RADIX_BUCKETS (1 << SORT_RADIX_BITS)
#define SORT_RADIX_PASSES (64 / SORT_RADIX_BITS)

typedef struct {
    uint64_t bits;
    int      idx;
} SortPair;

/* IEEE 754 laid out so that comparing the bit patterns as unsigned integers
 * gives the same order as comparing the numbers: the sign bit is flipped for
 * a positive, and every bit for a negative. Reversing the sort is the
 * complement of that, which keeps equal keys in the order they arrived --
 * sorting ascending and turning the result round would not. */
static inline uint64_t sortableBits(double d, bool reverse) {
    uint64_t bits;
    memcpy(&bits, &d, sizeof(bits));
    bits = (bits & 0x8000000000000000ULL) ? ~bits : (bits | 0x8000000000000000ULL);
    return reverse ? ~bits : bits;
}

/* A key function almost always returns a number, and numbers can be sorted
 * by their digits instead of against each other -- which matters because a
 * comparison merge spends most of its time on a branch that real data
 * mispredicts half the time, and no amount of tightening the comparison
 * moved that.
 *
 * Two kinds are handed back rather than converted. A whole number past a
 * double's exact range would compare wrong, and a NaN has no order at all --
 * the general path raises for that, and it should keep doing so. Negative
 * zero is folded onto zero because the two compare equal, so the order they
 * arrived in is the order they have to keep. */
static bool numericBits(ObjList *keys, const int *idx, int n, bool reverse,
                        SortPair *into) {
    for (int i = 0; i < n; i++) {
        Value key = jaiListGet(keys, idx[i]);
        double d;
        if (IS_FLOAT(key)) {
            d = AS_FLOAT(key);
            if (isnan(d)) return false;
            if (d == 0.0) d = 0.0;
        } else if (IS_INT(key)) {
            int64_t whole = AS_INT(key);
            if (whole > SORT_WHOLE_EXACT || whole < -SORT_WHOLE_EXACT) return false;
            d = (double)whole;
        } else {
            return false;
        }
        into[i].bits = sortableBits(d, reverse);
        into[i].idx = idx[i];
    }
    return true;
}

/* Least significant digit first, which is what makes it stable. Every pass's
 * histogram is counted in the one walk over the keys, so a byte the whole
 * array agrees on can be skipped rather than shuffled for nothing -- with
 * keys of a similar size most of the top half goes that way. */
static SortPair *radixPairs(SortPair *pairs, SortPair *other, int n) {
    uint32_t counts[SORT_RADIX_PASSES][SORT_RADIX_BUCKETS];
    memset(counts, 0, sizeof(counts));
    for (int i = 0; i < n; i++) {
        uint64_t bits = pairs[i].bits;
        for (int pass = 0; pass < SORT_RADIX_PASSES; pass++) {
            counts[pass][(bits >> (pass * SORT_RADIX_BITS)) & (SORT_RADIX_BUCKETS - 1)]++;
        }
    }

    SortPair *from = pairs, *to = other;
    for (int pass = 0; pass < SORT_RADIX_PASSES; pass++) {
        uint32_t *bucket = counts[pass];
        int shift = pass * SORT_RADIX_BITS;
        if (bucket[(from[0].bits >> shift) & (SORT_RADIX_BUCKETS - 1)] == (uint32_t)n)
            continue;

        uint32_t at = 0;
        for (int b = 0; b < SORT_RADIX_BUCKETS; b++) {
            uint32_t here = bucket[b];
            bucket[b] = at;
            at += here;
        }
        for (int i = 0; i < n; i++) {
            to[bucket[(from[i].bits >> shift) & (SORT_RADIX_BUCKETS - 1)]++] = from[i];
        }
        SortPair *swap = from;
        from = to;
        to = swap;
    }
    return from;
}

static bool sortCompare(Value a, Value b, const char *fnName, int *out) {
    if (jaiValueCompare(a, b, out)) return true;
    if (vm.hasException) return false;
    return jaiThrow(vm.cTypeError, "%s(): cannot compare %s with %s", fnName,
                    jaiTypeNameStatic(a), jaiTypeNameStatic(b));
}

static bool mergeGeneral(ObjList *keys, int **src, int **dst, int n,
                         bool reverse, const char *fnName) {
    for (int width = 1; width < n; width *= 2) {
        const int *from = *src;
        int *to = *dst;
        for (int lo = 0; lo < n; lo += 2 * width) {
            int mid = lo + width < n ? lo + width : n;
            int hi = lo + 2 * width < n ? lo + 2 * width : n;
            int a = lo, b = mid, k = lo;
            while (a < mid && b < hi) {
                int order;
                if (!sortCompare(jaiListGet(keys, from[b]), jaiListGet(keys, from[a]),
                                 fnName, &order))
                    return false;
                if (reverse) order = -order;
                to[k++] = order < 0 ? from[b++] : from[a++];
            }
            while (a < mid) to[k++] = from[a++];
            while (b < hi)  to[k++] = from[b++];
        }
        int *swap = *src;
        *src = *dst;
        *dst = swap;
    }
    return true;
}

bool jaiSeqSortIndices(ObjList *keys, int *idx, int *scratch, int n,
                       bool reverse, const char *fnName) {
    if (n <= 1) return true;
    int *src = idx, *dst = scratch;

    /* One look at the first key says whether reading them all off stands a
     * chance, which spares a list of strings an allocation it cannot use. */
    bool ok = true;
    bool sorted = false;
    if (n >= SORT_RADIX_FLOOR && (IS_INT(jaiListGet(keys, 0)) || IS_FLOAT(jaiListGet(keys, 0)))) {
        SortPair *pairs = JAI_ALLOC(SortPair, 2 * n);
        if (numericBits(keys, idx, n, reverse, pairs)) {
            const SortPair *ranked = radixPairs(pairs, pairs + n, n);
            for (int i = 0; i < n; i++) idx[i] = ranked[i].idx;
            sorted = true;
        }
        JAI_FREE_ARRAY(SortPair, pairs, 2 * n);
    }
    if (!sorted) ok = mergeGeneral(keys, &src, &dst, n, reverse, fnName);

    if (ok && !sorted && src != idx) memcpy(idx, src, (size_t)n * sizeof(int));
    return ok;
}



/* ------------------------------------------------------------------ */
/* Receivers and arguments                                              */
/* ------------------------------------------------------------------ */

Value jaiSeqOptArg(int argc, Value *args, int index) {
    return index < argc ? args[index] : NULL_VAL;
}

bool jaiSeqReceiverError(const char *fnName, const char *expected, Value got) {
    return jaiThrow(vm.cTypeError, "%s() needs a %s receiver, not %s", fnName,
                    expected, jaiTypeNameStatic(got));
}

static inline bool selfTuple(Value *args, const char *fnName,
                             ObjTuple **out) {
    const Value self = args[0];
    if (!IS_TUPLE(self))
        return jaiSeqReceiverError(fnName, "tuple", self);

    *out = AS_TUPLE(self);
    return true;
}

static inline bool selfRange(Value *args, const char *fnName,
                             ObjRange **out) {
    const Value self = args[0];
    if (!IS_RANGE(self))
        return jaiSeqReceiverError(fnName, "range", self);

    *out = AS_RANGE(self);
    return true;
}

static inline bool selfIter(Value *args, const char *fnName,
                            ObjIter **out) {
    const Value self = args[0];
    if (!IS_ITER(self))
        return jaiSeqReceiverError(fnName, "iterator", self);

    *out = AS_ITER(self);
    return true;
}

bool jaiSeqIndexArg(Value v, int index, const char *fnName, int count,
                     int *out) {
    int64_t raw;
    if (!jaiArgInt(v, index, fnName, &raw)) return false;
    if (!jaiNormalizeIndex(raw, count, out)) {
        return jaiThrow(vm.cIndexError,
                        "%s(): index %lld out of range for length %d", fnName,
                        (long long)raw, count);
    }
    return true;
}

/* ------------------------------------------------------------------ */
/* Shared operations                                                    */
/* ------------------------------------------------------------------ */

bool jaiSeqEqualsChecked(Value a, Value b, bool *equal) {
    *equal = jaiValuesEqual(a, b);
    return !vm.hasException;
}

static inline bool hashMayFail(Value v) {
    if (!IS_OBJ(v)) return false;

    switch (OBJ_TYPE(v)) {
        case OBJ_LIST:
        case OBJ_DICT:
        case OBJ_SET:
        case OBJ_TUPLE:
        case OBJ_ENUM_VAL:
        case OBJ_INSTANCE:
            return true;
        default:
            return false;
    }
}

bool jaiSeqHashableKey(Value key, const char *fnName, const char *role) {
    if (IS_NULL(key)) {
        return jaiThrow(vm.cTypeError, "%s(): a %s cannot be null", fnName, role);
    }
    if (!hashMayFail(key)) return true;

    bool ok = true;
    (void)jaiValueHash(key, &ok);
    if (ok) return true;
    if (vm.hasException) return false;
    return jaiThrow(vm.cTypeError, "%s(): unhashable %s of type '%s'", fnName,
                    role, jaiTypeNameStatic(key));
}

ObjList *jaiSeqCollectIterable(Value v) {
    Value iterVal;
    if (!jaiGetIter(v, &iterVal)) return NULL;

    if (!IS_ITER(iterVal)) {
        jaiThrow(vm.cTypeError, "'%s' object is not iterable",
                 jaiTypeNameStatic(v));
        return NULL;
    }

    jaiGCPushRoot(iterVal);
    ObjIter *const it = AS_ITER(iterVal);

    int capacity = 0;
    if (IS_TUPLE(v)) {
        const uint32_t n = AS_TUPLE(v)->count;
        if (n <= INT32_MAX) capacity = (int)n;
    } else if (IS_RANGE(v)) {
        const int64_t n = jaiRangeLength(AS_RANGE(v));
        if (n >= 0 && n <= INT32_MAX) capacity = (int)n;
    }

    ObjList *result = jaiListNew(capacity);
    jaiGCPushRoot(OBJ_VAL(result));

    Value item;
    while (jaiIterNext(it, &item)) {
        jaiGCPushRoot(item);
        jaiListPush(result, item);
        jaiGCPopRoot();

        if (vm.hasException)
            break;
    }

    jaiGCPopRoots(2);
    return vm.hasException ? NULL : result;
}

bool jaiSeqValueIter(int argc, Value *args, Value *out) {
    (void)argc;
    return jaiGetIter(args[0], out);
}

/* ------------------------------------------------------------------ */
/* Method binding                                                       */
/* ------------------------------------------------------------------ */

bool jaiSeqBindFrom(const JaiSeqMethod *table, int count, Value receiver,
                    ObjString *name, Value *out) {
    const uint32_t length = name->length;
    if (length == 0) return false;

    const char *const chars = name->chars;
    const unsigned char first = (unsigned char)chars[0];
    const unsigned char last = (unsigned char)chars[length - 1];

    for (int i = 0; i < count; ++i) {
        const JaiSeqMethod *const method = table + i;

        if ((uint32_t)method->length != length)
            continue;

        if ((unsigned char)method->name[0] != first)
            continue;

        if (length > 1 &&
            (unsigned char)method->name[length - 1] != last)
            continue;

        if (length > 2 &&
            memcmp(chars + 1, method->name + 1, length - 2) != 0)
            continue;

        *out = jaiBindNative(receiver, method->name, method->fn,
                             method->minArity, method->maxArity,
                             method->params);
        return true;
    }

    return false;
}

/* ------------------------------------------------------------------ */
/* tuple                                                                */
/* ------------------------------------------------------------------ */

static bool tupleLen(int argc, Value *args, Value *out) {
    (void)argc;
    ObjTuple *self;
    if (!selfTuple(args, "tuple.len", &self)) return false;
    *out = INT_VAL((int64_t)self->count);
    return true;
}

static bool tupleGet(int argc, Value *args, Value *out) {
    (void)argc;
    ObjTuple *self;
    if (!selfTuple(args, "tuple.get", &self)) return false;
    int at;
    if (!jaiSeqIndexArg(args[1], 1, "tuple.get", (int)self->count, &at)) return false;
    *out = self->items[at];
    return true;
}

/* Same bounds check and IndexError shape as tuple.get, just under the name
 * the indexing operator's method form is meant to have -- see bytes.at
 * (builtins_bytes.c), which matches this exactly for its own receiver. */
static bool tupleAt(int argc, Value *args, Value *out) {
    (void)argc;
    ObjTuple *self;
    if (!selfTuple(args, "tuple.at", &self)) return false;
    int at;
    if (!jaiSeqIndexArg(args[1], 1, "tuple.at", (int)self->count, &at)) return false;
    *out = self->items[at];
    return true;
}

static bool tupleIsEmpty(int argc, Value *args, Value *out) {
    (void)argc;
    ObjTuple *self;
    if (!selfTuple(args, "tuple.is_empty", &self)) return false;
    *out = BOOL_VAL(self->count == 0);
    return true;
}

static bool tupleContains(int argc, Value *args, Value *out) {
    (void)argc;

    ObjTuple *self;
    if (!selfTuple(args, "tuple.contains", &self)) return false;

    const uint32_t count = self->count;
    const Value *const items = self->items;
    const Value needle = args[1];

    for (uint32_t i = 0; i < count; ++i) {
        bool same;
        if (!jaiSeqEqualsChecked(items[i], needle, &same))
            return false;

        if (same) {
            *out = BOOL_VAL(true);
            return true;
        }
    }

    *out = BOOL_VAL(false);
    return true;
}

static bool tupleIndex(int argc, Value *args, Value *out) {
    (void)argc;

    ObjTuple *self;
    if (!selfTuple(args, "tuple.index", &self)) return false;

    const uint32_t count = self->count;
    const Value *const items = self->items;
    const Value needle = args[1];

    for (uint32_t i = 0; i < count; ++i) {
        bool same;
        if (!jaiSeqEqualsChecked(items[i], needle, &same))
            return false;

        if (same) {
            *out = INT_VAL((int64_t)i);
            return true;
        }
    }

    return jaiThrow(vm.cValueError, "tuple.index(): value not in tuple");
}

static bool tupleCount(int argc, Value *args, Value *out) {
    (void)argc;

    ObjTuple *self;
    if (!selfTuple(args, "tuple.count", &self)) return false;

    const uint32_t count = self->count;
    const Value *const items = self->items;
    const Value needle = args[1];
    int64_t total = 0;

    for (uint32_t i = 0; i < count; ++i) {
        bool same;
        if (!jaiSeqEqualsChecked(items[i], needle, &same))
            return false;

        total += same;
    }

    *out = INT_VAL(total);
    return true;
}

static bool tupleToList(int argc, Value *args, Value *out) {
    (void)argc;

    ObjTuple *self;
    if (!selfTuple(args, "tuple.to_list", &self)) return false;

    const uint32_t count = self->count;
    ObjList *result = jaiListNew((int)count);

    if (count != 0) {
        JAI_ASSERT(result->capacity >= (int)count,
                   "jaiListNew did not reserve requested capacity");
        memcpy(result->items, self->items, (size_t)count * sizeof(Value));
        result->count = (int)count;
    }

    *out = OBJ_VAL(result);
    return true;
}

/* ------------------------------------------------------------------ */
/* range                                                                */
/* ------------------------------------------------------------------ */

static inline int64_t rangeLast(ObjRange *r, int64_t length) {
    const uint64_t offset =
        (uint64_t)(length - 1) * (uint64_t)r->step;
    return (int64_t)((uint64_t)r->start + offset);
}

static bool rangeLen(int argc, Value *args, Value *out) {
    (void)argc;
    ObjRange *self;
    if (!selfRange(args, "range.len", &self)) return false;
    *out = INT_VAL(jaiRangeLength(self));
    return true;
}

static bool rangeIsEmpty(int argc, Value *args, Value *out) {
    (void)argc;
    ObjRange *self;
    if (!selfRange(args, "range.is_empty", &self)) return false;
    *out = BOOL_VAL(jaiRangeLength(self) == 0);
    return true;
}

static bool rangeContains(int argc, Value *args, Value *out) {
    (void)argc;

    ObjRange *self;
    if (!selfRange(args, "range.contains", &self)) return false;

    if (!IS_INT(args[1])) {
        *out = BOOL_VAL(false);
        return true;
    }

    const int64_t length = jaiRangeLength(self);
    if (length == 0) {
        *out = BOOL_VAL(false);
        return true;
    }

    const int64_t value = AS_INT(args[1]);
    const int64_t start = self->start;
    const int64_t step = self->step;
    const int64_t last = rangeLast(self, length);

    bool within =
        step > 0 ? (value >= start && value <= last)
                 : (value <= start && value >= last);

    if (within && step != 1 && step != -1) {
        const uint64_t offset = (uint64_t)value - (uint64_t)start;
        const uint64_t stride =
            step > 0 ? (uint64_t)step : 0u - (uint64_t)step;

        within = (offset % stride) == 0;
    }

    *out = BOOL_VAL(within);
    return true;
}

static bool rangeToList(int argc, Value *args, Value *out) {
    (void)argc;

    ObjRange *self;
    if (!selfRange(args, "range.to_list", &self)) return false;

    const int64_t length = jaiRangeLength(self);
    if (length > INT32_MAX) {
        return jaiThrow(vm.cValueError,
                        "range.to_list(): %lld elements is more than a list can hold",
                        (long long)length);
    }

    const int count = (int)length;
    ObjList *result = jaiListNew(count);

    if (count != 0) {
        JAI_ASSERT(result->capacity >= count,
                   "jaiListNew did not reserve requested capacity");

        Value *const items = result->items;
        uint64_t value = (uint64_t)self->start;
        const uint64_t step = (uint64_t)self->step;

        for (int i = 0; i < count; ++i) {
            items[i] = INT_VAL((int64_t)value);
            value += step;
        }

        result->count = count;
    }

    *out = OBJ_VAL(result);
    return true;
}

static bool rangeStep(int argc, Value *args, Value *out) {
    (void)argc;
    ObjRange *self;
    if (!selfRange(args, "range.step", &self)) return false;
    *out = INT_VAL(self->step);
    return true;
}

static bool rangeStart(int argc, Value *args, Value *out) {
    (void)argc;
    ObjRange *self;
    if (!selfRange(args, "range.start", &self)) return false;
    *out = INT_VAL(self->start);
    return true;
}

static bool rangeStop(int argc, Value *args, Value *out) {
    (void)argc;
    ObjRange *self;
    if (!selfRange(args, "range.stop", &self)) return false;
    *out = INT_VAL(self->stop);
    return true;
}

static bool rangeReversed(int argc, Value *args, Value *out) {
    (void)argc;
    ObjRange *self;
    if (!selfRange(args, "range.reversed", &self)) return false;

    int64_t length = jaiRangeLength(self);
    if (length == 0) {
        *out = OBJ_VAL(jaiRangeNew(0, 0, 1, false));
        return true;
    }
    int64_t last = rangeLast(self, length);
    if (length == 1) {
        *out = OBJ_VAL(jaiRangeNew(last, last, 1, true));
        return true;
    }
    if (self->step == INT64_MIN) {
        return jaiThrow(vm.cOverflowError,
                        "range.reversed(): step %lld cannot be negated",
                        (long long)self->step);
    }
    *out = OBJ_VAL(jaiRangeNew(last, self->start, -self->step, true));
    return true;
}

/* ------------------------------------------------------------------ */
/* iterator                                                            */
/* ------------------------------------------------------------------ */

static bool iterNextMethod(int argc, Value *args, Value *out) {
    (void)argc;
    ObjIter *self;
    if (!selfIter(args, "iter.next", &self)) return false;

    Value item;
    if (jaiIterNext(self, &item)) {
        *out = item;
        return true;
    }
    if (vm.hasException) return false;
    *out = NULL_VAL;
    return true;
}

static bool iterTake(int argc, Value *args, Value *out) {
    (void)argc;

    ObjIter *self;
    if (!selfIter(args, "iter.take", &self)) return false;

    int64_t n;
    if (!jaiArgInt(args[1], 1, "iter.take", &n)) return false;

    if (n < 0) {
        return jaiThrow(vm.cValueError,
                        "iter.take(): count must be non-negative, got %lld",
                        (long long)n);
    }

    const int capacity = n <= 256 ? (int)n : 0;
    ObjList *result = jaiListNew(capacity);
    jaiGCPushRoot(OBJ_VAL(result));

    for (int64_t i = 0; i < n; ++i) {
        Value item;
        if (!jaiIterNext(self, &item))
            break;

        jaiGCPushRoot(item);
        jaiListPush(result, item);
        jaiGCPopRoot();

        if (vm.hasException)
            break;
    }

    jaiGCPopRoot();

    if (vm.hasException)
        return false;

    *out = OBJ_VAL(result);
    return true;
}

static bool iterDrop(int argc, Value *args, Value *out) {
    (void)argc;

    ObjIter *self;
    if (!selfIter(args, "iter.drop", &self)) return false;

    int64_t n;
    if (!jaiArgInt(args[1], 1, "iter.drop", &n)) return false;

    if (n < 0) {
        return jaiThrow(vm.cValueError,
                        "iter.drop(): count must be non-negative, got %lld",
                        (long long)n);
    }

    while (n-- > 0) {
        Value ignored;
        if (!jaiIterNext(self, &ignored))
            break;
    }

    if (vm.hasException)
        return false;

    *out = args[0];
    return true;
}

static bool iterCollect(int argc, Value *args, Value *out) {
    (void)argc;

    ObjIter *self;
    if (!selfIter(args, "iter.collect", &self)) return false;

    ObjList *result = jaiListNew(0);
    jaiGCPushRoot(OBJ_VAL(result));

    Value item;
    while (jaiIterNext(self, &item)) {
        jaiGCPushRoot(item);
        jaiListPush(result, item);
        jaiGCPopRoot();

        if (vm.hasException)
            break;
    }

    jaiGCPopRoot();

    if (vm.hasException)
        return false;

    *out = OBJ_VAL(result);
    return true;
}

/* ------------------------------------------------------------------ */
/* Method tables                                                        */
/* ------------------------------------------------------------------ */

static const JaiSeqMethod kTupleMethods[] = {
    JAI_METHOD("at",       tupleAt,       2, 2),
    JAI_METHOD("contains", tupleContains, 2, 2),
    JAI_METHOD("count",    tupleCount,    2, 2),
    JAI_METHOD("get",      tupleGet,      2, 2),
    JAI_METHOD("index",    tupleIndex,    2, 2),
    JAI_METHOD("is_empty", tupleIsEmpty,  1, 1),
    JAI_METHOD("iter",     jaiSeqValueIter,     1, 1),
    JAI_METHOD("len",      tupleLen,      1, 1),
    JAI_METHOD("to_list",  tupleToList,   1, 1),
};

static const JaiSeqMethod kRangeMethods[] = {
    JAI_METHOD("contains", rangeContains, 2, 2),
    JAI_METHOD("is_empty", rangeIsEmpty,  1, 1),
    JAI_METHOD("iter",     jaiSeqValueIter,     1, 1),
    JAI_METHOD("len",      rangeLen,      1, 1),
    JAI_METHOD("reversed", rangeReversed, 1, 1),
    JAI_METHOD("start",    rangeStart,    1, 1),
    JAI_METHOD("step",     rangeStep,     1, 1),
    JAI_METHOD("stop",     rangeStop,     1, 1),
    JAI_METHOD("to_list",  rangeToList,   1, 1),
};

static const JaiSeqMethod kIterMethods[] = {
    JAI_METHOD("collect", iterCollect,    1, 1),
    JAI_METHOD("drop",    iterDrop,       2, 2),
    JAI_METHOD("iter",    jaiSeqValueIter,      1, 1),
    JAI_METHOD("next",    iterNextMethod, 1, 1),
    JAI_METHOD("take",    iterTake,       2, 2),
};

bool jaiTupleMethod(Value receiver, ObjString *name, Value *out) {
    return JAI_BIND_FROM(kTupleMethods, receiver, name, out);
}

bool jaiRangeMethod(Value receiver, ObjString *name, Value *out) {
    return JAI_BIND_FROM(kRangeMethods, receiver, name, out);
}

bool jaiIterMethod(Value receiver, ObjString *name, Value *out) {
    return JAI_BIND_FROM(kIterMethods, receiver, name, out);
}
