/* builtins_seq.c — the methods of tuple, range and iterator, and the pieces
 * every container method file shares.
 *
 * list is builtins_list.c and dict and set are builtins_dict.c; the three are
 * split because each is a large table with its own `__prim__` surface, and
 * builtins_seq.h is the short list of things more than one of them needs.
 *
 * A built-in method is reached through an ObjBound, so the receiver arrives in
 * args[0] and argc counts it: args[1] is the first argument the user wrote and
 * its user-visible position is 1, which is what the jaiArg* helpers print.
 *
 * Three rules run through all three files.
 *
 *   - Anything that can re-enter Jaithon — a map callback, a sort key, a user
 *     __eq__ reached from a comparison, a user __hash__ reached from a dict
 *     insert — can allocate and therefore collect. Every partially built
 *     result is held in a GC temp root across such a call, and every loop
 *     bound over a container the caller can still reach is re-read each turn.
 *   - Methods that take a callback over a snapshot (sort, min, max) copy the
 *     receiver first. The callback is user code and may mutate the receiver;
 *     working from a private copy is what keeps the key array and the element
 *     array the same length.
 *   - A mutator with no natural result returns the receiver, so that
 *     `xs.push(1).push(2)` reads the way it looks. Everything else returns the
 *     answer to the question it was asked.
 */

#include "builtins_seq.h"
#include "methods.h"
#include "runtime.h"

#include "../vm/gc.h"

/* ------------------------------------------------------------------ */
/* Receivers and arguments                                              */
/* ------------------------------------------------------------------ */

Value jaiSeqOptArg(int argc, Value *args, int index) {
    return index < argc ? args[index] : NULL_VAL;
}

/* Reachable only when a bound method outlives the value it was bound to in a
 * way the VM cannot see; cheap enough to check, fatal enough to be worth it. */
bool jaiSeqReceiverError(const char *fnName, const char *expected, Value got) {
    return jaiThrow(vm.cTypeError, "%s() needs a %s receiver, not %s", fnName,
                    expected, jaiTypeNameStatic(got));
}

static bool selfTuple(Value *args, const char *fnName, ObjTuple **out) {
    if (!IS_TUPLE(args[0])) return jaiSeqReceiverError(fnName, "tuple", args[0]);
    *out = AS_TUPLE(args[0]);
    return true;
}

static bool selfRange(Value *args, const char *fnName, ObjRange **out) {
    if (!IS_RANGE(args[0])) return jaiSeqReceiverError(fnName, "range", args[0]);
    *out = AS_RANGE(args[0]);
    return true;
}

static bool selfIter(Value *args, const char *fnName, ObjIter **out) {
    if (!IS_ITER(args[0])) return jaiSeqReceiverError(fnName, "iterator", args[0]);
    *out = AS_ITER(args[0]);
    return true;
}

/* A concrete index into a sequence of `count` items, negatives counting from
 * the end. */
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

/* Equality that reports a raising __eq__ instead of reading its failure as
 * "not equal". */
bool jaiSeqEqualsChecked(Value a, Value b, bool *equal) {
    *equal = jaiValuesEqual(a, b);
    return !vm.hasException;
}

/* Hashing fails only for the mutable containers and for the composites that
 * may contain one — or for an instance with no (or a raising) __hash__. */
static bool hashMayFail(Value v) {
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

/* Dict keys and set elements must hash, and must not be `null`: the table
 * layer reserves null to mark an empty slot (table.c), so it has to be
 * rejected here rather than corrupting a probe.
 *
 * The hash check cannot be left to the table either, because the table
 * reports "this does not hash" exactly the way it reports "this key was
 * already present" — a false. Only the keys whose hash can fail at all are
 * probed here, which keeps a user __hash__ from being run twice for the keys
 * that dominate (int, str, bool, float). */
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

/* Drain any iterable into a fresh list. NULL with the exception pending on
 * failure. The result is unrooted: root it before allocating again. */
ObjList *jaiSeqCollectIterable(Value v) {
    Value iterVal;
    if (!jaiGetIter(v, &iterVal)) return NULL;
    if (!IS_ITER(iterVal)) {
        jaiThrow(vm.cTypeError, "'%s' object is not iterable", jaiTypeNameStatic(v));
        return NULL;
    }

    jaiGCPushRoot(iterVal);
    ObjList *out = jaiListNew(0);
    jaiGCPushRoot(OBJ_VAL(out));

    Value item;
    while (jaiIterNext(AS_ITER(iterVal), &item)) {
        jaiGCPushRoot(item);
        jaiListPush(out, item);
        jaiGCPopRoot();
        if (vm.hasException) break;
    }
    jaiGCPopRoots(2);

    return vm.hasException ? NULL : out;
}

/* `xs.iter()` is the explicit spelling of what `for x in xs` does implicitly;
 * one implementation serves every container. */
bool jaiSeqValueIter(int argc, Value *args, Value *out) {
    (void)argc;
    return jaiGetIter(args[0], out);
}

/* ------------------------------------------------------------------ */
/* Method binding                                                       */
/* ------------------------------------------------------------------ */

bool jaiSeqBindFrom(const JaiSeqMethod *table, int count, Value receiver,
                     ObjString *name, Value *out) {
    for (int i = 0; i < count; i++) {
        if (name->length != table[i].length) continue;
        if (memcmp(name->chars, table[i].name, table[i].length) != 0) continue;
        *out = jaiBindNative(receiver, table[i].name, table[i].fn,
                             table[i].minArity, table[i].maxArity, table[i].params);
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

static bool tupleContains(int argc, Value *args, Value *out) {
    (void)argc;
    ObjTuple *self;
    if (!selfTuple(args, "tuple.contains", &self)) return false;

    for (uint32_t i = 0; i < self->count; i++) {
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

static bool tupleIndex(int argc, Value *args, Value *out) {
    (void)argc;
    ObjTuple *self;
    if (!selfTuple(args, "tuple.index", &self)) return false;

    for (uint32_t i = 0; i < self->count; i++) {
        bool same;
        if (!jaiSeqEqualsChecked(self->items[i], args[1], &same)) return false;
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

    int64_t total = 0;
    for (uint32_t i = 0; i < self->count; i++) {
        bool same;
        if (!jaiSeqEqualsChecked(self->items[i], args[1], &same)) return false;
        if (same) total++;
    }
    *out = INT_VAL(total);
    return true;
}

static bool tupleToList(int argc, Value *args, Value *out) {
    (void)argc;
    ObjTuple *self;
    if (!selfTuple(args, "tuple.to_list", &self)) return false;

    ObjList *result = jaiListNew((int)self->count);
    for (uint32_t i = 0; i < self->count && result->count < result->capacity; i++) {
        result->items[result->count++] = self->items[i];
    }
    *out = OBJ_VAL(result);
    return true;
}

/* ------------------------------------------------------------------ */
/* range                                                                */
/* ------------------------------------------------------------------ */

/* The last value a range yields. Only called when the length is nonzero, and
 * computed in unsigned arithmetic because start + (n-1)*step is exact but its
 * intermediates need not be. */
static int64_t rangeLast(ObjRange *r, int64_t length) {
    uint64_t offset = (uint64_t)(length - 1) * (uint64_t)r->step;
    return (int64_t)((uint64_t)r->start + offset);
}

static bool rangeLen(int argc, Value *args, Value *out) {
    (void)argc;
    ObjRange *self;
    if (!selfRange(args, "range.len", &self)) return false;
    *out = INT_VAL(jaiRangeLength(self));
    return true;
}

/* Only an int can be a member; anything else is simply absent rather than an
 * error, so `x in r` stays usable on a heterogeneous value. */
static bool rangeContains(int argc, Value *args, Value *out) {
    (void)argc;
    ObjRange *self;
    if (!selfRange(args, "range.contains", &self)) return false;

    int64_t length = jaiRangeLength(self);
    if (!IS_INT(args[1]) || length == 0) {
        *out = BOOL_VAL(false);
        return true;
    }

    int64_t v = AS_INT(args[1]);
    int64_t last = rangeLast(self, length);
    bool within = (self->step > 0) ? (v >= self->start && v <= last)
                                   : (v <= self->start && v >= last);
    if (within) {
        /* The offset from the start is exact for any member, and the unsigned
         * difference is well defined even when the span exceeds INT64_MAX. */
        uint64_t offset = (uint64_t)v - (uint64_t)self->start;
        uint64_t stride = (self->step > 0) ? (uint64_t)self->step
                                           : 0u - (uint64_t)self->step;
        within = (offset % stride) == 0;
    }
    *out = BOOL_VAL(within);
    return true;
}

static bool rangeToList(int argc, Value *args, Value *out) {
    (void)argc;
    ObjRange *self;
    if (!selfRange(args, "range.to_list", &self)) return false;

    int64_t length = jaiRangeLength(self);
    if (length > INT32_MAX) {
        return jaiThrow(vm.cValueError,
                        "range.to_list(): %lld elements is more than a list can hold",
                        (long long)length);
    }

    ObjList *result = jaiListNew((int)length);
    for (int64_t i = 0; i < length; i++) {
        result->items[result->count++] =
            INT_VAL((int64_t)((uint64_t)self->start + (uint64_t)i * (uint64_t)self->step));
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

/* The same values in the other order, still as a range: start at the last
 * element and walk back, inclusively, to the original start. */
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
    /* Every other reversal walks back by -step, which INT64_MIN has no room
     * for. Nothing else in the language can represent that sequence, so it is
     * an error rather than a rounded answer. */
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

/* Exhaustion is `null`, matching the Iterator trait in std.core; a raising
 * __next__ is propagated instead. */
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

/* Iterators are pulled, not sliced, so take() materialises: at most `n`
 * elements, fewer if the source ends first. */
static bool iterTake(int argc, Value *args, Value *out) {
    (void)argc;
    ObjIter *self;
    if (!selfIter(args, "iter.take", &self)) return false;
    int64_t n;
    if (!jaiArgInt(args[1], 1, "iter.take", &n)) return false;
    if (n < 0) {
        return jaiThrow(vm.cValueError, "iter.take(): count must be non-negative, got %lld",
                        (long long)n);
    }

    ObjList *result = jaiListNew(0);
    jaiGCPushRoot(OBJ_VAL(result));
    for (int64_t i = 0; i < n; i++) {
        Value item;
        if (!jaiIterNext(self, &item)) break;
        jaiGCPushRoot(item);
        jaiListPush(result, item);
        jaiGCPopRoot();
        if (vm.hasException) break;
    }
    jaiGCPopRoot();
    if (vm.hasException) return false;
    *out = OBJ_VAL(result);
    return true;
}

/* drop advances the iterator and hands it back, so it can be chained. */
static bool iterDrop(int argc, Value *args, Value *out) {
    (void)argc;
    ObjIter *self;
    if (!selfIter(args, "iter.drop", &self)) return false;
    int64_t n;
    if (!jaiArgInt(args[1], 1, "iter.drop", &n)) return false;
    if (n < 0) {
        return jaiThrow(vm.cValueError, "iter.drop(): count must be non-negative, got %lld",
                        (long long)n);
    }

    for (int64_t i = 0; i < n; i++) {
        Value ignored;
        if (!jaiIterNext(self, &ignored)) break;
    }
    if (vm.hasException) return false;
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
        if (vm.hasException) break;
    }
    jaiGCPopRoot();
    if (vm.hasException) return false;
    *out = OBJ_VAL(result);
    return true;
}

/* ------------------------------------------------------------------ */
/* Method tables                                                        */
/* ------------------------------------------------------------------ */

static const JaiSeqMethod kTupleMethods[] = {
    JAI_METHOD("contains", tupleContains, 2, 2),
    JAI_METHOD("count",    tupleCount,    2, 2),
    JAI_METHOD("get",      tupleGet,      2, 2),
    JAI_METHOD("index",    tupleIndex,    2, 2),
    JAI_METHOD("iter",     jaiSeqValueIter,     1, 1),
    JAI_METHOD("len",      tupleLen,      1, 1),
    JAI_METHOD("to_list",  tupleToList,   1, 1),
};

static const JaiSeqMethod kRangeMethods[] = {
    JAI_METHOD("contains", rangeContains, 2, 2),
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
