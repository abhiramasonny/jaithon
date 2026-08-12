/* builtins_list.c — the `list` method table and the `__prim__.list_*` surface (spec Appendix C). */

#include "runtime/builtins/collections/builtins_seq.h"
#include "runtime/methods.h"
#include "runtime/runtime.h"

#include "vm/gc.h"

/* ------------------------------------------------------------------ */
/* Receivers and arguments                                              */
/* ------------------------------------------------------------------ */

static bool selfList(Value *args, const char *fnName, ObjList **out) {
    if (!IS_LIST(args[0])) return jaiSeqReceiverError(fnName, "list", args[0]);
    *out = AS_LIST(args[0]);
    return true;
}

/* An omitted argument and an explicit `null` mean the same thing for every
 * optional parameter here: use the default. */
static bool optIntArg(int argc, Value *args, int index, const char *fnName,
                      int64_t fallback, int64_t *out) {
    Value v = jaiSeqOptArg(argc, args, index);
    if (IS_NULL(v)) {
        *out = fallback;
        return true;
    }
    return jaiArgInt(v, index, fnName, out);
}

static bool optBoolArg(int argc, Value *args, int index, const char *fnName,
                       bool fallback, bool *out) {
    Value v = jaiSeqOptArg(argc, args, index);
    if (IS_NULL(v)) {
        *out = fallback;
        return true;
    }
    return jaiArgBool(v, index, fnName, out);
}

static bool optCallableArg(int argc, Value *args, int index, const char *fnName,
                           Value *out) {
    Value v = jaiSeqOptArg(argc, args, index);
    if (IS_NULL(v)) {
        *out = NULL_VAL;
        return true;
    }
    if (!jaiArgCallable(v, index, fnName)) return false;
    *out = v;
    return true;
}

/* ------------------------------------------------------------------ */
/* Shared operations                                                    */
/* ------------------------------------------------------------------ */

static bool addI64(int64_t a, int64_t b, int64_t *out) {
    if ((b > 0 && a > INT64_MAX - b) || (b < 0 && a < INT64_MIN - b)) return false;
    *out = a + b;
    return true;
}

/* "These two have no order" is a TypeError for every caller in this file. */
static bool compareOrThrow(Value a, Value b, const char *fnName, int *out) {
    if (jaiValueCompare(a, b, out)) return true;
    if (vm.hasException) return false;
    return jaiThrow(vm.cTypeError, "%s(): cannot compare %s with %s", fnName,
                    jaiTypeNameStatic(a), jaiTypeNameStatic(b));
}

/* No truthiness in Jaithon (spec §2.6): a predicate must return bool, not coerce. */
static bool callPredicate(Value pred, Value item, const char *fnName, bool *out) {
    Value arg = item, verdict;
    if (!jaiCallValue1(pred, arg, &verdict)) return false;
    if (!IS_BOOL(verdict)) {
        return jaiThrow(vm.cTypeError,
                        "%s(): the predicate must return bool, not %s", fnName,
                        jaiTypeNameStatic(verdict));
    }
    *out = AS_BOOL(verdict);
    return true;
}

/* ------------------------------------------------------------------ */
/* Stable sort over an index permutation                                */
/* ------------------------------------------------------------------ */

/* Values stay in GC-visible lists; only indices are raw C ints, so GC during a
 * user key/__lt__ call cannot invalidate anything here. */
static bool mergeSortIndices(ObjList *keys, int *idx, int *scratch, int n,
                             bool reverse, const char *fnName) {
    for (int width = 1; width < n; width *= 2) {
        for (int lo = 0; lo < n; lo += 2 * width) {
            int mid = lo + width < n ? lo + width : n;
            int hi = lo + 2 * width < n ? lo + 2 * width : n;
            int a = lo, b = mid, k = lo;
            while (a < mid && b < hi) {
                int order;
                if (!compareOrThrow(keys->items[idx[b]], keys->items[idx[a]],
                                    fnName, &order))
                    return false;
                if (reverse) order = -order;
                /* Take from the right half only when strictly smaller: keeps the
                 * sort stable and stops `reverse` from reordering equal elements. */
                scratch[k++] = order < 0 ? idx[b++] : idx[a++];
            }
            while (a < mid) scratch[k++] = idx[a++];
            while (b < hi)  scratch[k++] = idx[b++];
        }
        memcpy(idx, scratch, (size_t)n * sizeof(int));
    }
    return true;
}

/* Calls `keyFn` exactly once per element. Leaves two temp roots (items, keys)
 * on the stack on success, which the caller pops together. */
static bool snapshotAndKeys(ObjList *source, Value keyFn, ObjList **outItems,
                            ObjList **outKeys) {
    ObjList *items = jaiListNew(source->count);
    jaiGCPushRoot(OBJ_VAL(items));
    for (int i = 0; i < source->count && items->count < items->capacity; i++) {
        items->items[items->count++] = source->items[i];
    }

    if (IS_NULL(keyFn)) {
        /* Push the same list twice so the caller always pops two roots. */
        jaiGCPushRoot(OBJ_VAL(items));
        *outItems = items;
        *outKeys = items;
        return true;
    }

    ObjList *keys = jaiListNew(items->count);
    jaiGCPushRoot(OBJ_VAL(keys));
    for (int i = 0; i < items->count; i++) {
        Value arg = items->items[i], key;
        if (!jaiCallValue1(keyFn, arg, &key)) {
            jaiGCPopRoots(2);
            return false;
        }
        jaiGCPushRoot(key);
        jaiListPush(keys, key);          /* capacity reserved: cannot allocate */
        jaiGCPopRoot();
    }
    *outItems = items;
    *outKeys = keys;
    return true;
}

/* `items` and `keys` must be rooted by the caller and have the same length;
 * `keys` is `items` itself when there is no key function. */
static ObjList *sortedCopy(ObjList *items, ObjList *keys, bool reverse,
                           const char *fnName) {
    int n = items->count;
    ObjList *result = jaiListNew(n);
    jaiGCPushRoot(OBJ_VAL(result));
    if (n <= 1) {
        for (int i = 0; i < n; i++) result->items[result->count++] = items->items[i];
        jaiGCPopRoot();
        return result;
    }

    int *idx = JAI_ALLOC(int, n);
    int *scratch = JAI_ALLOC(int, n);
    for (int i = 0; i < n; i++) idx[i] = i;

    bool ok = mergeSortIndices(keys, idx, scratch, n, reverse, fnName);
    if (ok) {
        for (int i = 0; i < n; i++) result->items[result->count++] = items->items[idx[i]];
    }

    JAI_FREE_ARRAY(int, idx, n);
    JAI_FREE_ARRAY(int, scratch, n);
    jaiGCPopRoot();
    return ok ? result : NULL;
}

static ObjList *sortedList(ObjList *source, Value keyFn, bool reverse,
                           const char *fnName) {
    ObjList *items = NULL, *keys = NULL;
    jaiGCPushRoot(OBJ_VAL(source));
    jaiGCPushRoot(keyFn);
    if (!snapshotAndKeys(source, keyFn, &items, &keys)) {
        jaiGCPopRoots(2);
        return NULL;
    }
    ObjList *result = sortedCopy(items, keys, reverse, fnName);
    jaiGCPopRoots(4);
    return result;
}

/* ------------------------------------------------------------------ */
/* Shuffling                                                            */
/* ------------------------------------------------------------------ */

/* std.random owns the language-level generator; shuffle only needs a stream of
 * its own, seeded once. xoshiro256** over a splitmix64 seed. */
static uint64_t gShuffleState[4];
static bool     gShuffleSeeded;

static uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static uint64_t rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t shuffleNext(void) {
    if (!gShuffleSeeded) {
        uint64_t seed = (uint64_t)(jaiClockMonotonic() * 1e9);
        seed ^= (uint64_t)(uintptr_t)&gShuffleState;
        for (int i = 0; i < 4; i++) gShuffleState[i] = splitmix64(&seed);
        gShuffleSeeded = true;
    }

    uint64_t *s = gShuffleState;
    uint64_t result = rotl64(s[1] * 5, 7) * 9;
    uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl64(s[3], 45);
    return result;
}

/* Unbiased value in [0, bound): reject everything above the largest multiple
 * of `bound` that fits in 64 bits. */
static uint64_t shuffleBelow(uint64_t bound) {
    uint64_t zone = UINT64_MAX - (UINT64_MAX % bound);
    uint64_t r;
    do {
        r = shuffleNext();
    } while (r >= zone);
    return r % bound;
}

/* ------------------------------------------------------------------ */
/* Text assembly                                                        */
/* ------------------------------------------------------------------ */

/* Consumes `buf` either way, even on failure. */
static bool takeString(JaiBuf *buf, Value *out) {
    size_t length = 0;
    char *chars = jaiBufTakeCString(buf, &length);
    if (chars == NULL) {
        return jaiThrow(vm.cRuntimeError, "out of memory building a string");
    }
    ObjString *text = jaiStringTake(chars, length);
    if (text == NULL) return false;       /* over the length limit; it threw */
    *out = OBJ_VAL(text);
    return true;
}

/* ------------------------------------------------------------------ */
/* list                                                                 */
/* ------------------------------------------------------------------ */

static bool listLen(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *self;
    if (!selfList(args, "list.len", &self)) return false;
    *out = INT_VAL(self->count);
    return true;
}

static bool listPush(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *self;
    if (!selfList(args, "list.push", &self)) return false;
    jaiListPush(self, args[1]);
    if (vm.hasException) return false;
    *out = args[0];
    return true;
}

static bool listPop(int argc, Value *args, Value *out) {
    ObjList *self;
    if (!selfList(args, "list.pop", &self)) return false;
    if (self->count == 0) {
        return jaiThrow(vm.cIndexError, "list.pop(): the list is empty");
    }
    if (argc >= 2 && !IS_NULL(args[1])) {
        int at;
        if (!jaiSeqIndexArg(args[1], 1, "list.pop", self->count, &at)) return false;
        *out = jaiListRemove(self, at);
        return !vm.hasException;
    }
    *out = jaiListPop(self);
    return !vm.hasException;
}

/* An out-of-range index clamps to an end, as it does in Python: insert is the
 * one position argument for which "past the end" has an obvious meaning. */
static bool listInsert(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *self;
    if (!selfList(args, "list.insert", &self)) return false;
    int64_t raw;
    if (!jaiArgInt(args[1], 1, "list.insert", &raw)) return false;

    int count = self->count;
    int at;
    if (raw >= count)        at = count;
    else if (raw >= 0)       at = (int)raw;
    else if (raw <= -count)  at = 0;
    else                     at = (int)(raw + count);

    jaiListInsert(self, at, args[2]);
    if (vm.hasException) return false;
    *out = args[0];
    return true;
}

static bool listRemove(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *self;
    if (!selfList(args, "list.remove", &self)) return false;

    /* The count is re-read because a user __eq__ may resize the list. */
    for (int i = 0; i < self->count; i++) {
        bool same;
        if (!jaiSeqEqualsChecked(self->items[i], args[1], &same)) return false;
        if (!same) continue;
        (void)jaiListRemove(self, i);
        if (vm.hasException) return false;
        *out = args[0];
        return true;
    }
    return jaiThrow(vm.cValueError, "list.remove(): value not in list");
}

static bool listRemoveAt(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *self;
    if (!selfList(args, "list.remove_at", &self)) return false;
    int at;
    if (!jaiSeqIndexArg(args[1], 1, "list.remove_at", self->count, &at)) return false;
    *out = jaiListRemove(self, at);
    return !vm.hasException;
}

static bool listClear(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *self;
    if (!selfList(args, "list.clear", &self)) return false;
    /* The capacity is kept: a cleared list is usually about to be refilled,
     * and the stale slots past `count` are never traced. */
    self->count = 0;
    jaiListTouch(self);
    *out = args[0];
    return true;
}

static bool listExtend(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *self;
    if (!selfList(args, "list.extend", &self)) return false;

    /* Snapshot first, so that `xs.extend(xs)` terminates instead of tripping
     * the iterator's mutation check. */
    ObjList *items = jaiSeqCollectIterable(args[1]);
    if (items == NULL) return false;
    jaiGCPushRoot(OBJ_VAL(items));
    int64_t needed = (int64_t)self->count + items->count;
    if (needed <= INT32_MAX) jaiListReserve(self, (int)needed);
    for (int i = 0; i < items->count; i++) jaiListPush(self, items->items[i]);
    jaiGCPopRoot();
    if (vm.hasException) return false;
    *out = args[0];
    return true;
}

static bool listIndex(int argc, Value *args, Value *out) {
    ObjList *self;
    if (!selfList(args, "list.index", &self)) return false;
    int64_t from;
    if (!optIntArg(argc, args, 2, "list.index", 0, &from)) return false;
    if (from < 0) {
        from += self->count;
        if (from < 0) from = 0;
    }

    for (int64_t i = from; i < (int64_t)self->count; i++) {
        bool same;
        if (!jaiSeqEqualsChecked(self->items[i], args[1], &same)) return false;
        if (same) {
            *out = INT_VAL(i);
            return true;
        }
    }
    return jaiThrow(vm.cValueError, "list.index(): value not in list");
}

static bool listCount(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *self;
    if (!selfList(args, "list.count", &self)) return false;

    int64_t total = 0;
    for (int i = 0; i < self->count; i++) {
        bool same;
        if (!jaiSeqEqualsChecked(self->items[i], args[1], &same)) return false;
        if (same) total++;
    }
    *out = INT_VAL(total);
    return true;
}

static bool listContains(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *self;
    if (!selfList(args, "list.contains", &self)) return false;

    for (int i = 0; i < self->count; i++) {
        bool same;
        if (!jaiSeqEqualsChecked(self->items[i], args[1], &same)) return false;
        if (same) {
            *out = BOOL_VAL(true);
            return true;
        }
    }
    *out = BOOL_VAL(false);
    return true;
}

static bool listReverse(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *self;
    if (!selfList(args, "list.reverse", &self)) return false;
    for (int i = 0, j = self->count - 1; i < j; i++, j--) {
        Value tmp = self->items[i];
        self->items[i] = self->items[j];
        self->items[j] = tmp;
    }
    if (self->count > 1) jaiListTouch(self);
    *out = args[0];
    return true;
}

static bool listReversed(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *self;
    if (!selfList(args, "list.reversed", &self)) return false;

    ObjList *result = jaiListNew(self->count);
    for (int i = self->count - 1; i >= 0; i--) result->items[result->count++] = self->items[i];
    *out = OBJ_VAL(result);
    return true;
}

static bool listSort(int argc, Value *args, Value *out) {
    ObjList *self;
    if (!selfList(args, "list.sort", &self)) return false;
    Value keyFn;
    bool reverse;
    if (!optCallableArg(argc, args, 1, "list.sort", &keyFn)) return false;
    if (!optBoolArg(argc, args, 2, "list.sort", false, &reverse)) return false;

    int n = self->count;
    ObjList *sorted = sortedList(self, keyFn, reverse, "list.sort");
    if (sorted == NULL) return false;
    if (self->count != n) {
        return jaiThrow(vm.cRuntimeError, "list.sort(): list changed size during the sort");
    }
    /* Nothing allocates between here and the copy, so `sorted` stays alive
     * without a root. */
    if (n > 0) {
        memcpy(self->items, sorted->items, sizeof(Value) * (size_t)n);
        jaiListTouch(self);
    }
    *out = args[0];
    return true;
}

static bool listSorted(int argc, Value *args, Value *out) {
    ObjList *self;
    if (!selfList(args, "list.sorted", &self)) return false;
    Value keyFn;
    bool reverse;
    if (!optCallableArg(argc, args, 1, "list.sorted", &keyFn)) return false;
    if (!optBoolArg(argc, args, 2, "list.sorted", false, &reverse)) return false;

    ObjList *sorted = sortedList(self, keyFn, reverse, "list.sorted");
    if (sorted == NULL) return false;
    *out = OBJ_VAL(sorted);
    return true;
}

static bool listSlice(int argc, Value *args, Value *out) {
    ObjList *self;
    if (!selfList(args, "list.slice", &self)) return false;

    int64_t step;
    if (!optIntArg(argc, args, 3, "list.slice", 1, &step)) return false;
    if (step == 0) return jaiThrow(vm.cValueError, "list.slice(): step cannot be zero");

    /* The defaults depend on the direction of travel: INT64_MIN and INT64_MAX
     * clamp to "one before the start" and "one past the end". */
    int64_t start, stop;
    if (!optIntArg(argc, args, 1, "list.slice", step > 0 ? 0 : INT64_MAX, &start))
        return false;
    if (!optIntArg(argc, args, 2, "list.slice", step > 0 ? INT64_MAX : INT64_MIN, &stop))
        return false;

    ObjList *result = jaiListSlice(self, start, stop, step);
    if (result == NULL) return false;
    *out = OBJ_VAL(result);
    return true;
}

static bool listCopy(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *self;
    if (!selfList(args, "list.copy", &self)) return false;

    ObjList *result = jaiListNew(self->count);
    for (int i = 0; i < self->count && result->count < result->capacity; i++) {
        result->items[result->count++] = self->items[i];
    }
    *out = OBJ_VAL(result);
    return true;
}

static bool listMap(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *self;
    if (!selfList(args, "list.map", &self)) return false;
    if (!jaiArgCallable(args[1], 1, "list.map")) return false;

    ObjList *result = jaiListNew(self->count);
    jaiGCPushRoot(OBJ_VAL(result));
    bool ok = true;
    for (int i = 0; i < self->count; i++) {
        Value arg = self->items[i], mapped;
        if (!jaiCallValue1(args[1], arg, &mapped)) {
            ok = false;
            break;
        }
        /* No root needed: `result` is pre-sized and nothing else here allocates
         * (rooting per element cost 30% of this loop). Slow path covers callback growth. */
        if (JAI_LIKELY(result->count < result->capacity)) {
            result->items[result->count++] = mapped;
            result->version++;
        } else {
            jaiGCPushRoot(mapped);
            jaiListPush(result, mapped);
            jaiGCPopRoot();
        }
        if (vm.hasException) {
            ok = false;
            break;
        }
    }
    jaiGCPopRoot();
    if (!ok) return false;
    *out = OBJ_VAL(result);
    return true;
}

static bool listFilter(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *self;
    if (!selfList(args, "list.filter", &self)) return false;
    if (!jaiArgCallable(args[1], 1, "list.filter")) return false;

    ObjList *result = jaiListNew(0);
    jaiGCPushRoot(OBJ_VAL(result));
    bool ok = true;
    for (int i = 0; i < self->count; i++) {
        Value item = self->items[i];
        bool keep;
        if (!callPredicate(args[1], item, "list.filter", &keep)) {
            ok = false;
            break;
        }
        if (!keep) continue;
        jaiGCPushRoot(item);
        jaiListPush(result, item);
        jaiGCPopRoot();
        if (vm.hasException) {
            ok = false;
            break;
        }
    }
    jaiGCPopRoot();
    if (!ok) return false;
    *out = OBJ_VAL(result);
    return true;
}

/* reduce(f) folds from the first element; reduce(init, f) starts at `init` and
 * is total over the empty list. Combine is always last, matching std.core's fold. */
static bool listReduce(int argc, Value *args, Value *out) {
    ObjList *self;
    if (!selfList(args, "list.reduce", &self)) return false;
    Value combine = argc >= 3 ? args[2] : args[1];
    if (!jaiArgCallable(combine, argc >= 3 ? 2 : 1, "list.reduce")) return false;

    /* A null seed is the seedless form: that is what a keyword call leaves in
     * the hole when only `combine` is named. */
    int start = 0;
    Value acc;
    if (argc >= 3 && !IS_NULL(args[1])) {
        acc = args[1];
    } else {
        if (self->count == 0) {
            return jaiThrow(vm.cValueError,
                            "list.reduce(): empty list and no initial value");
        }
        acc = self->items[0];
        start = 1;
    }

    jaiGCPushRoot(acc);
    bool ok = true;
    for (int i = start; i < self->count; i++) {
        Value callArgs[2] = {acc, self->items[i]};
        Value next;
        if (!jaiCallValue(combine, 2, callArgs, &next)) {
            ok = false;
            break;
        }
        /* The accumulator is a fresh object more often than not, so it is
         * re-rooted rather than left behind on the previous root. */
        jaiGCPopRoot();
        acc = next;
        jaiGCPushRoot(acc);
    }
    jaiGCPopRoot();
    if (!ok) return false;
    *out = acc;
    return true;
}

static bool listForEach(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *self;
    if (!selfList(args, "list.for_each", &self)) return false;
    if (!jaiArgCallable(args[1], 1, "list.for_each")) return false;

    for (int i = 0; i < self->count; i++) {
        Value arg = self->items[i], ignored;
        if (!jaiCallValue1(args[1], arg, &ignored)) return false;
    }
    *out = NULL_VAL;
    return true;
}

static bool listQuantify(int argc, Value *args, Value *out, bool wantAny) {
    const char *fnName = wantAny ? "list.any" : "list.all";
    ObjList *self;
    if (!selfList(args, fnName, &self)) return false;
    Value pred;
    if (!optCallableArg(argc, args, 1, fnName, &pred)) return false;

    for (int i = 0; i < self->count; i++) {
        Value item = self->items[i];
        bool verdict;
        if (IS_NULL(pred)) {
            if (!IS_BOOL(item)) {
                return jaiThrow(vm.cTypeError,
                                "%s(): without a predicate the elements must be "
                                "bool, not %s", fnName, jaiTypeNameStatic(item));
            }
            verdict = AS_BOOL(item);
        } else if (!callPredicate(pred, item, fnName, &verdict)) {
            return false;
        }
        if (verdict == wantAny) {
            *out = BOOL_VAL(wantAny);
            return true;
        }
    }
    *out = BOOL_VAL(!wantAny);
    return true;
}

static bool listAny(int argc, Value *args, Value *out) {
    return listQuantify(argc, args, out, true);
}

static bool listAll(int argc, Value *args, Value *out) {
    return listQuantify(argc, args, out, false);
}

static bool listSum(int argc, Value *args, Value *out) {
    ObjList *self;
    if (!selfList(args, "list.sum", &self)) return false;

    Value total = jaiSeqOptArg(argc, args, 1);
    if (IS_NULL(total)) total = INT_VAL(0);
    if (!IS_NUMBER(total)) {
        return jaiThrow(vm.cTypeError,
                        "list.sum() argument 1: expected int or float, got %s",
                        jaiTypeNameStatic(total));
    }

    for (int i = 0; i < self->count; i++) {
        Value item = self->items[i];
        if (!IS_NUMBER(item)) {
            return jaiThrow(vm.cTypeError,
                            "list.sum(): element %d is %s, not a number", i,
                            jaiTypeNameStatic(item));
        }
        if (IS_INT(total) && IS_INT(item)) {
            int64_t sum;
            if (!addI64(AS_INT(total), AS_INT(item), &sum)) {
                return jaiThrow(vm.cOverflowError, "integer overflow in list.sum()");
            }
            total = INT_VAL(sum);
        } else {
            total = FLOAT_VAL(jaiAsDouble(total) + jaiAsDouble(item));
        }
    }
    *out = total;
    return true;
}

/* Each element's key is computed once, as for sort, not once per comparison. */
static bool listExtremum(int argc, Value *args, Value *out, bool wantMax) {
    const char *fnName = wantMax ? "list.max" : "list.min";
    ObjList *self;
    if (!selfList(args, fnName, &self)) return false;
    Value keyFn;
    if (!optCallableArg(argc, args, 1, fnName, &keyFn)) return false;
    if (self->count == 0) {
        return jaiThrow(vm.cValueError, "%s(): the list is empty", fnName);
    }

    ObjList *items = NULL, *keys = NULL;
    jaiGCPushRoot(keyFn);
    if (!snapshotAndKeys(self, keyFn, &items, &keys)) {
        jaiGCPopRoot();
        return false;
    }

    int best = 0;
    bool ok = true;
    for (int i = 1; i < items->count; i++) {
        int order;
        if (!compareOrThrow(keys->items[i], keys->items[best], fnName, &order)) {
            ok = false;
            break;
        }
        if (wantMax ? order > 0 : order < 0) best = i;
    }

    Value result = ok ? items->items[best] : NULL_VAL;
    jaiGCPopRoots(3);
    if (!ok) return false;
    *out = result;
    return true;
}

static bool listMin(int argc, Value *args, Value *out) {
    return listExtremum(argc, args, out, false);
}

static bool listMax(int argc, Value *args, Value *out) {
    return listExtremum(argc, args, out, true);
}

static bool listEnd(int argc, Value *args, Value *out, bool wantLast) {
    const char *fnName = wantLast ? "list.last" : "list.first";
    ObjList *self;
    if (!selfList(args, fnName, &self)) return false;
    if (self->count > 0) {
        *out = self->items[wantLast ? self->count - 1 : 0];
        return true;
    }
    if (argc >= 2) {
        *out = args[1];
        return true;
    }
    return jaiThrow(vm.cIndexError, "%s(): the list is empty", fnName);
}

static bool listFirst(int argc, Value *args, Value *out) {
    return listEnd(argc, args, out, false);
}

static bool listLast(int argc, Value *args, Value *out) {
    return listEnd(argc, args, out, true);
}

/* Elements go through str(), so join is total over any list (never a TypeError). */
static bool listJoin(int argc, Value *args, Value *out) {
    ObjList *self;
    if (!selfList(args, "list.join", &self)) return false;

    ObjString *sep = NULL;
    Value sepArg = jaiSeqOptArg(argc, args, 1);
    if (!IS_NULL(sepArg) && !jaiArgString(sepArg, 1, "list.join", &sep)) return false;

    JaiBuf buf;
    jaiBufInit(&buf);
    for (int i = 0; i < self->count; i++) {
        if (i > 0 && sep != NULL) jaiBufAppend(&buf, sep->chars, sep->length);
        Value item = self->items[i];
        if (IS_STRING(item)) {
            jaiBufAppend(&buf, AS_STRING(item)->chars, AS_STRING(item)->length);
            continue;
        }
        ObjString *text = jaiValueToStr(item);
        if (text == NULL) {
            jaiBufFree(&buf);
            return false;
        }
        jaiBufAppend(&buf, text->chars, text->length);
    }
    return takeString(&buf, out);
}

/* One level deep: every element must itself be iterable. */
static bool listFlatten(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *self;
    if (!selfList(args, "list.flatten", &self)) return false;

    ObjList *result = jaiListNew(0);
    jaiGCPushRoot(OBJ_VAL(result));
    bool ok = true;
    for (int i = 0; i < self->count; i++) {
        Value iterVal;
        if (!jaiGetIter(self->items[i], &iterVal) || !IS_ITER(iterVal)) {
            ok = false;
            break;
        }
        jaiGCPushRoot(iterVal);
        Value item;
        while (jaiIterNext(AS_ITER(iterVal), &item)) {
            jaiGCPushRoot(item);
            jaiListPush(result, item);
            jaiGCPopRoot();
            if (vm.hasException) break;
        }
        jaiGCPopRoot();
        if (vm.hasException) {
            ok = false;
            break;
        }
    }
    jaiGCPopRoot();
    if (!ok) return false;
    *out = OBJ_VAL(result);
    return true;
}

static bool listZip(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *self;
    if (!selfList(args, "list.zip", &self)) return false;

    ObjList *other = jaiSeqCollectIterable(args[1]);
    if (other == NULL) return false;
    jaiGCPushRoot(OBJ_VAL(other));

    int n = self->count < other->count ? self->count : other->count;
    ObjList *result = jaiListNew(n);
    jaiGCPushRoot(OBJ_VAL(result));
    for (int i = 0; i < n; i++) {
        Value pair[2] = {self->items[i], other->items[i]};
        ObjTuple *tuple = jaiTupleNew(pair, 2);
        result->items[result->count++] = OBJ_VAL(tuple);
    }
    jaiGCPopRoots(2);
    *out = OBJ_VAL(result);
    return true;
}

static bool listEnumerate(int argc, Value *args, Value *out) {
    ObjList *self;
    if (!selfList(args, "list.enumerate", &self)) return false;
    int64_t start;
    if (!optIntArg(argc, args, 1, "list.enumerate", 0, &start)) return false;

    int n = self->count;
    ObjList *result = jaiListNew(n);
    jaiGCPushRoot(OBJ_VAL(result));
    bool ok = true;
    for (int i = 0; i < n; i++) {
        int64_t index;
        if (!addI64(start, (int64_t)i, &index)) {
            ok = jaiThrow(vm.cOverflowError, "integer overflow in list.enumerate()");
            break;
        }
        Value pair[2] = {INT_VAL(index), self->items[i]};
        ObjTuple *tuple = jaiTupleNew(pair, 2);
        result->items[result->count++] = OBJ_VAL(tuple);
    }
    jaiGCPopRoot();
    if (!ok) return false;
    *out = OBJ_VAL(result);
    return true;
}

static bool listChunk(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *self;
    if (!selfList(args, "list.chunk", &self)) return false;
    int64_t size;
    if (!jaiArgInt(args[1], 1, "list.chunk", &size)) return false;
    if (size <= 0) {
        return jaiThrow(vm.cValueError, "list.chunk(): size must be positive, got %lld",
                        (long long)size);
    }

    int n = self->count;
    int step = size > (int64_t)n ? (n > 0 ? n : 1) : (int)size;
    ObjList *result = jaiListNew((n + step - 1) / step);
    jaiGCPushRoot(OBJ_VAL(result));
    for (int i = 0; i < n; i += step) {
        int width = (n - i < step) ? n - i : step;
        ObjList *chunk = jaiListNew(width);
        for (int k = 0; k < width; k++) chunk->items[chunk->count++] = self->items[i + k];
        jaiGCPushRoot(OBJ_VAL(chunk));
        jaiListPush(result, OBJ_VAL(chunk));
        jaiGCPopRoot();
        if (vm.hasException) break;
    }
    jaiGCPopRoot();
    if (vm.hasException) return false;
    *out = OBJ_VAL(result);
    return true;
}

static bool listWindow(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *self;
    if (!selfList(args, "list.window", &self)) return false;
    int64_t size;
    if (!jaiArgInt(args[1], 1, "list.window", &size)) return false;
    if (size <= 0) {
        return jaiThrow(vm.cValueError, "list.window(): size must be positive, got %lld",
                        (long long)size);
    }

    int n = self->count;
    bool fits = size <= (int64_t)n;
    int width = fits ? (int)size : 0;
    int count = fits ? n - width + 1 : 0;
    ObjList *result = jaiListNew(count);
    jaiGCPushRoot(OBJ_VAL(result));
    for (int i = 0; i < count; i++) {
        ObjList *window = jaiListNew(width);
        for (int k = 0; k < width; k++) window->items[window->count++] = self->items[i + k];
        jaiGCPushRoot(OBJ_VAL(window));
        jaiListPush(result, OBJ_VAL(window));
        jaiGCPopRoot();
        if (vm.hasException) break;
    }
    jaiGCPopRoot();
    if (vm.hasException) return false;
    *out = OBJ_VAL(result);
    return true;
}

/* `null` is tracked separately since it cannot be a set/table key. */
static bool listUnique(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *self;
    if (!selfList(args, "list.unique", &self)) return false;

    ObjSet *seen = jaiSetNew();
    jaiGCPushRoot(OBJ_VAL(seen));
    ObjList *result = jaiListNew(0);
    jaiGCPushRoot(OBJ_VAL(result));

    bool sawNull = false;
    bool ok = true;
    for (int i = 0; i < self->count; i++) {
        Value item = self->items[i];
        bool fresh;
        if (IS_NULL(item)) {
            fresh = !sawNull;
            sawNull = true;
        } else {
            if (!jaiSeqHashableKey(item, "list.unique", "list element")) {
                ok = false;
                break;
            }
            fresh = jaiSetAdd(seen, item);
            if (vm.hasException) {
                ok = false;
                break;
            }
        }
        if (!fresh) continue;
        jaiGCPushRoot(item);
        jaiListPush(result, item);
        jaiGCPopRoot();
        if (vm.hasException) {
            ok = false;
            break;
        }
    }
    jaiGCPopRoots(2);
    if (!ok) return false;
    *out = OBJ_VAL(result);
    return true;
}

static bool listShuffle(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *self;
    if (!selfList(args, "list.shuffle", &self)) return false;

    for (int i = self->count - 1; i > 0; i--) {
        int j = (int)shuffleBelow((uint64_t)i + 1);
        Value tmp = self->items[i];
        self->items[i] = self->items[j];
        self->items[j] = tmp;
    }
    if (self->count > 1) jaiListTouch(self);
    *out = args[0];
    return true;
}

/* ------------------------------------------------------------------ */
/* Method table                                                         */
/* ------------------------------------------------------------------ */

/* Names trailing optional params for keyword calls like `xs.sorted(reverse:
 * true)`; index 0 is the receiver "self", which callers can never name. */
static const char *const kSelfKey[]        = {"self", "key"};
static const char *const kSelfKeyReverse[] = {"self", "key", "reverse"};
static const char *const kSelfPredicate[]  = {"self", "predicate"};
static const char *const kSelfStart[]      = {"self", "start"};
static const char *const kSelfDefault[]    = {"self", "default"};
static const char *const kSelfSeparator[]  = {"self", "separator"};
static const char *const kSelfIndex[]      = {"self", "index"};
static const char *const kSelfValueFrom[]  = {"self", "value", "from"};
static const char *const kSelfInitialFn[]  = {"self", "initial", "combine"};
static const char *const kSelfSlice[]      = {"self", "start", "stop", "step"};

static const JaiSeqMethod kListMethods[] = {
    JAI_METHOD_KW("all",    listAll,    1, 2, kSelfPredicate),
    JAI_METHOD_KW("any",    listAny,    1, 2, kSelfPredicate),
    JAI_METHOD("chunk",     listChunk,     2, 2),
    JAI_METHOD("clear",     listClear,     1, 1),
    JAI_METHOD("contains",  listContains,  2, 2),
    JAI_METHOD("copy",      listCopy,      1, 1),
    JAI_METHOD("count",     listCount,     2, 2),
    JAI_METHOD_KW("enumerate", listEnumerate, 1, 2, kSelfStart),
    JAI_METHOD("extend",    listExtend,    2, 2),
    JAI_METHOD("filter",    listFilter,    2, 2),
    JAI_METHOD_KW("first",  listFirst,  1, 2, kSelfDefault),
    JAI_METHOD("flatten",   listFlatten,   1, 1),
    JAI_METHOD("for_each",  listForEach,   2, 2),
    JAI_METHOD_KW("index",  listIndex,  2, 3, kSelfValueFrom),
    JAI_METHOD("insert",    listInsert,    3, 3),
    JAI_METHOD("iter",      jaiSeqValueIter,     1, 1),
    JAI_METHOD_KW("join",   listJoin,   1, 2, kSelfSeparator),
    JAI_METHOD_KW("last",   listLast,   1, 2, kSelfDefault),
    JAI_METHOD("len",       listLen,       1, 1),
    JAI_METHOD("map",       listMap,       2, 2),
    JAI_METHOD_KW("max",    listMax,    1, 2, kSelfKey),
    JAI_METHOD_KW("min",    listMin,    1, 2, kSelfKey),
    JAI_METHOD_KW("pop",    listPop,    1, 2, kSelfIndex),
    JAI_METHOD("push",      listPush,      2, 2),
    JAI_METHOD_KW("reduce", listReduce, 2, 3, kSelfInitialFn),
    JAI_METHOD("remove",    listRemove,    2, 2),
    JAI_METHOD("remove_at", listRemoveAt,  2, 2),
    JAI_METHOD("reverse",   listReverse,   1, 1),
    JAI_METHOD("reversed",  listReversed,  1, 1),
    JAI_METHOD("shuffle",   listShuffle,   1, 1),
    JAI_METHOD_KW("slice",  listSlice,  1, 4, kSelfSlice),
    JAI_METHOD_KW("sort",   listSort,   1, 3, kSelfKeyReverse),
    JAI_METHOD_KW("sorted", listSorted, 1, 3, kSelfKeyReverse),
    JAI_METHOD_KW("sum",    listSum,    1, 2, kSelfStart),
    JAI_METHOD("unique",    listUnique,    1, 1),
    JAI_METHOD("window",    listWindow,    2, 2),
    JAI_METHOD("zip",       listZip,       2, 2),
};

bool jaiListMethod(Value receiver, ObjString *name, Value *out) {
    return JAI_BIND_FROM(kListMethods, receiver, name, out);
}

/* ------------------------------------------------------------------ */
/* __prim__.list_*                                                      */
/* ------------------------------------------------------------------ */

/* Thin layer std.list/std.dict are written over: one C op each, same errors as
 * the methods, no defaulting beyond an omitted slice bound. */

/* No receiver here, so args[i] is the user's argument i+1; methods above pass
 * the index straight through since their args[0] is the receiver. */
static bool optIntPrim(int argc, Value *args, int index, const char *fnName,
                       int64_t fallback, int64_t *out) {
    Value v = jaiSeqOptArg(argc, args, index);
    if (IS_NULL(v)) {
        *out = fallback;
        return true;
    }
    return jaiArgInt(v, index + 1, fnName, out);
}

static bool primListNew(int argc, Value *args, Value *out) {
    int64_t capacity;
    if (!optIntPrim(argc, args, 0, "list_new", 0, &capacity)) return false;
    if (capacity < 0) capacity = 0;
    if (capacity > INT32_MAX) {
        return jaiThrow(vm.cValueError, "list_new(): capacity %lld is too large",
                        (long long)capacity);
    }
    *out = OBJ_VAL(jaiListNew((int)capacity));
    return true;
}

static bool primListGet(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *list;
    if (!jaiArgList(args[0], 1, "list_get", &list)) return false;
    int at;
    if (!jaiSeqIndexArg(args[1], 2, "list_get", list->count, &at)) return false;
    *out = list->items[at];
    return true;
}

static bool primListSet(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *list;
    if (!jaiArgList(args[0], 1, "list_set", &list)) return false;
    int at;
    if (!jaiSeqIndexArg(args[1], 2, "list_set", list->count, &at)) return false;
    list->items[at] = args[2];
    jaiListTouch(list);
    *out = NULL_VAL;
    return true;
}

static bool primListPush(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *list;
    if (!jaiArgList(args[0], 1, "list_push", &list)) return false;
    jaiListPush(list, args[1]);
    if (vm.hasException) return false;
    *out = NULL_VAL;
    return true;
}

static bool primListPop(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *list;
    if (!jaiArgList(args[0], 1, "list_pop", &list)) return false;
    *out = jaiListPop(list);
    return !vm.hasException;
}

static bool primListInsert(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *list;
    if (!jaiArgList(args[0], 1, "list_insert", &list)) return false;
    int64_t raw;
    if (!jaiArgInt(args[1], 2, "list_insert", &raw)) return false;

    int count = list->count;
    int at;
    if (raw >= count)       at = count;
    else if (raw >= 0)      at = (int)raw;
    else if (raw <= -count) at = 0;
    else                    at = (int)(raw + count);

    jaiListInsert(list, at, args[2]);
    if (vm.hasException) return false;
    *out = NULL_VAL;
    return true;
}

static bool primListRemove(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *list;
    if (!jaiArgList(args[0], 1, "list_remove", &list)) return false;
    int at;
    if (!jaiSeqIndexArg(args[1], 2, "list_remove", list->count, &at)) return false;
    *out = jaiListRemove(list, at);
    return !vm.hasException;
}

static bool primListLen(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *list;
    if (!jaiArgList(args[0], 1, "list_len", &list)) return false;
    *out = INT_VAL(list->count);
    return true;
}

static bool primListSlice(int argc, Value *args, Value *out) {
    ObjList *list;
    if (!jaiArgList(args[0], 1, "list_slice", &list)) return false;

    int64_t step;
    if (!optIntPrim(argc, args, 3, "list_slice", 1, &step)) return false;
    if (step == 0) return jaiThrow(vm.cValueError, "list_slice(): step cannot be zero");

    int64_t start, stop;
    if (!optIntPrim(argc, args, 1, "list_slice", step > 0 ? 0 : INT64_MAX, &start))
        return false;
    if (!optIntPrim(argc, args, 2, "list_slice", step > 0 ? INT64_MAX : INT64_MIN, &stop))
        return false;

    ObjList *result = jaiListSlice(list, start, stop, step);
    if (result == NULL) return false;
    *out = OBJ_VAL(result);
    return true;
}

static bool primListConcat(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *a, *b;
    if (!jaiArgList(args[0], 1, "list_concat", &a)) return false;
    if (!jaiArgList(args[1], 2, "list_concat", &b)) return false;
    ObjList *result = jaiListConcat(a, b);
    if (result == NULL) return false;
    *out = OBJ_VAL(result);
    return true;
}

/* ------------------------------------------------------------------ */
/* Registration                                                         */
/* ------------------------------------------------------------------ */

void jaiRegisterListPrimitives(void) {
    jaiDefineNative("__prim__.list_new",    primListNew,    0, 1);
    jaiDefineNative("__prim__.list_get",    primListGet,    2, 2);
    jaiDefineNative("__prim__.list_set",    primListSet,    3, 3);
    jaiDefineNative("__prim__.list_push",   primListPush,   2, 2);
    jaiDefineNative("__prim__.list_pop",    primListPop,    1, 1);
    jaiDefineNative("__prim__.list_insert", primListInsert, 3, 3);
    jaiDefineNative("__prim__.list_remove", primListRemove, 2, 2);
    jaiDefineNative("__prim__.list_len",    primListLen,    1, 1);
    jaiDefineNative("__prim__.list_slice",  primListSlice,  1, 4);
    jaiDefineNative("__prim__.list_concat", primListConcat, 2, 2);
}
