/* builtins_iter.c — the builtins that walk an iterable: min, max, sum,
 * sorted, reversed, enumerate, zip, map, filter, any and all. */

#include "runtime/builtins/builtins_core.h"

#include "vm/gc.h"

/* Turns "no order between these two" into a TypeError, as every ordering builtin wants. */
static bool compareOrThrow(Value a, Value b, const char *fnName, int *out) {
    if (jaiValueCompare(a, b, out)) return true;
    if (vm.hasException) return false;
    return jaiThrow(vm.cTypeError, "%s(): cannot compare %s with %s", fnName,
                    jaiTypeNameStatic(a), jaiTypeNameStatic(b));
}

static bool extremum(int argc, Value *args, Value *out, bool wantMax) {
    const char *fnName = wantMax ? "max" : "min";

    ObjList *items = NULL;
    Value best;
    int start;

    if (argc == 1) {
        items = collectIterable(args[0]);
        if (items == NULL) return false;
        if (items->count == 0)
            return jaiThrow(vm.cValueError, "%s() argument is an empty sequence",
                            fnName);
        jaiGCPushRoot(OBJ_VAL(items));
        best = jaiListGet(items, 0);
        start = 1;
    } else {
        best = args[0];
        start = 1;
    }

    int count = items != NULL ? items->count : argc;
    bool ok = true;
    for (int i = start; i < count; i++) {
        Value candidate = items != NULL ? jaiListGet(items, i) : args[i];
        int order;
        if (!compareOrThrow(candidate, best, fnName, &order)) { ok = false; break; }
        if (wantMax ? order > 0 : order < 0) best = candidate;
    }

    if (items != NULL) jaiGCPopRoot();
    if (!ok) return false;
    *out = best;
    return true;
}

bool nMin(int argc, Value *args, Value *out) { return extremum(argc, args, out, false); }
bool nMax(int argc, Value *args, Value *out) { return extremum(argc, args, out, true); }

bool nSum(int argc, Value *args, Value *out) {
    Value total = argc >= 2 ? args[1] : INT_VAL(0);
    if (!IS_NUMBER(total)) return jaiBuiltinArgTypeError(2, "sum", "int or float", total);

    ObjList *items = collectIterable(args[0]);
    if (items == NULL) return false;
    jaiGCPushRoot(OBJ_VAL(items));

    bool ok = true;
    for (int i = 0; i < items->count; i++) {
        Value item = jaiListGet(items, i);
        if (!IS_NUMBER(item)) {
            ok = jaiBuiltinArgTypeError(1, "sum", "an iterable of int or float", item);
            break;
        }
        if (IS_INT(total) && IS_INT(item)) {
            int64_t sum;
            if (!jaiBuiltinAddI64(AS_INT(total), AS_INT(item), &sum)) {
                ok = jaiBuiltinOverflowError("sum()");
                break;
            }
            total = INT_VAL(sum);
        } else {
            total = FLOAT_VAL(jaiAsDouble(total) + jaiAsDouble(item));
        }
    }
    jaiGCPopRoot();
    if (!ok) return false;
    *out = total;
    return true;
}

bool nSorted(int argc, Value *args, Value *out) {
    Value source = args[0];
    Value keyFn = argc >= 2 ? args[1] : NULL_VAL;
    bool reverse = false;
    if (argc >= 3 && !jaiArgBool(args[2], 3, "sorted", &reverse)) return false;
    if (!IS_NULL(keyFn) && !jaiArgCallable(keyFn, 2, "sorted")) return false;

    ObjList *items = collectIterable(source);
    if (items == NULL) return false;
    jaiGCPushRoot(OBJ_VAL(items));
    jaiGCPushRoot(keyFn);

    ObjList *keys = items;
    bool ok = true;
    if (!IS_NULL(keyFn)) {
        keys = jaiListNew(items->count);
        jaiGCPushRoot(OBJ_VAL(keys));
        for (int i = 0; i < items->count; i++) {
            Value key;
            Value arg = jaiListGet(items, i);
            if (!jaiCallFn1(keyFn, arg, &key)) { ok = false; break; }
            jaiGCPushRoot(key);
            jaiListPush(keys, key);
            jaiGCPopRoot();
        }
    }

    ObjList *sorted = NULL;
    if (ok) sorted = sortedCopy(items, keys, reverse, "sorted");

    if (keys != items) jaiGCPopRoot();
    jaiGCPopRoots(2);

    if (!ok || sorted == NULL) return false;
    *out = OBJ_VAL(sorted);
    return true;
}

bool nReversed(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *items = collectIterable(args[0]);
    if (items == NULL) return false;
    for (int i = 0, j = items->count - 1; i < j; i++, j--) {
        Value tmp = jaiListGet(items, i);
        jaiListPut(items, i, jaiListGet(items, j));
        jaiListPut(items, j, tmp);
    }
    *out = OBJ_VAL(items);
    return true;
}

bool nEnumerate(int argc, Value *args, Value *out) {
    int64_t start = 0;
    if (argc >= 2 && !jaiArgInt(args[1], 2, "enumerate", &start)) return false;

    ObjList *items = collectIterable(args[0]);
    if (items == NULL) return false;
    jaiGCPushRoot(OBJ_VAL(items));

    ObjList *result = jaiListNew(items->count);
    jaiGCPushRoot(OBJ_VAL(result));
    bool ok = true;
    for (int i = 0; i < items->count; i++) {
        int64_t index;
        if (!jaiBuiltinAddI64(start, (int64_t)i, &index)) { ok = jaiBuiltinOverflowError("enumerate()"); break; }
        Value pair[2] = {INT_VAL(index), jaiListGet(items, i)};
        Value tuple = OBJ_VAL(jaiTupleNew(pair, 2));
        jaiGCPushRoot(tuple);
        jaiListPush(result, tuple);
        jaiGCPopRoot();
    }
    jaiGCPopRoots(2);
    if (!ok) return false;
    *out = OBJ_VAL(result);
    return true;
}

static ObjList *collectAll(int argc, Value *args, int from, int *outShortest) {
    ObjList *holder = jaiListNew(argc - from);
    jaiGCPushRoot(OBJ_VAL(holder));

    int shortest = -1;
    for (int i = from; i < argc; i++) {
        ObjList *one = collectIterable(args[i]);
        if (one == NULL) { jaiGCPopRoot(); return NULL; }
        jaiGCPushRoot(OBJ_VAL(one));
        jaiListPush(holder, OBJ_VAL(one));
        jaiGCPopRoot();
        if (shortest < 0 || one->count < shortest) shortest = one->count;
    }
    jaiGCPopRoot();

    *outShortest = shortest < 0 ? 0 : shortest;
    return holder;
}

bool nZip(int argc, Value *args, Value *out) {
    int shortest = 0;
    ObjList *holder = collectAll(argc, args, 0, &shortest);
    if (holder == NULL) return false;
    jaiGCPushRoot(OBJ_VAL(holder));

    int width = holder->count;
    ObjList *result = jaiListNew(shortest);
    jaiGCPushRoot(OBJ_VAL(result));

    Value *slot = width > 0 ? JAI_ALLOC(Value, width) : NULL;
    for (int i = 0; i < shortest; i++) {
        for (int c = 0; c < width; c++)
            slot[c] = jaiListGet(AS_LIST(jaiListGet(holder, c)), i);
        Value tuple = OBJ_VAL(jaiTupleNew(slot, width));
        jaiGCPushRoot(tuple);
        jaiListPush(result, tuple);
        jaiGCPopRoot();
    }
    if (slot != NULL) JAI_FREE_ARRAY(Value, slot, width);

    jaiGCPopRoots(2);
    *out = OBJ_VAL(result);
    return true;
}

bool nMap(int argc, Value *args, Value *out) {
    Value fn = args[0];
    if (!jaiArgCallable(fn, 1, "map")) return false;

    int shortest = 0;
    jaiGCPushRoot(fn);
    ObjList *holder = collectAll(argc, args, 1, &shortest);
    if (holder == NULL) { jaiGCPopRoot(); return false; }
    jaiGCPushRoot(OBJ_VAL(holder));

    int width = holder->count;
    ObjList *result = jaiListNew(shortest);
    jaiGCPushRoot(OBJ_VAL(result));

    Value *callArgs = width > 0 ? JAI_ALLOC(Value, width) : NULL;
    bool ok = true;
    for (int i = 0; i < shortest; i++) {
        for (int c = 0; c < width; c++)
            callArgs[c] = jaiListGet(AS_LIST(jaiListGet(holder, c)), i);
        Value mapped;
        if (!jaiCallValue(fn, width, callArgs, &mapped)) { ok = false; break; }
        jaiGCPushRoot(mapped);
        jaiListPush(result, mapped);
        jaiGCPopRoot();
    }
    if (callArgs != NULL) JAI_FREE_ARRAY(Value, callArgs, width);

    jaiGCPopRoots(3);
    if (!ok) return false;
    *out = OBJ_VAL(result);
    return true;
}

bool nFilter(int argc, Value *args, Value *out) {
    (void)argc;
    Value fn = args[0];
    if (!jaiArgCallable(fn, 1, "filter")) return false;

    jaiGCPushRoot(fn);
    ObjList *items = collectIterable(args[1]);
    if (items == NULL) { jaiGCPopRoot(); return false; }
    jaiGCPushRoot(OBJ_VAL(items));

    ObjList *result = jaiListNew(0);
    jaiGCPushRoot(OBJ_VAL(result));

    bool ok = true;
    for (int i = 0; i < items->count; i++) {
        Value item = jaiListGet(items, i);
        Value verdict;
        if (!jaiCallFn1(fn, item, &verdict)) { ok = false; break; }
        if (!IS_BOOL(verdict)) {
            ok = jaiThrow(vm.cTypeError, "filter(): predicate must return bool, not %s",
                          jaiTypeNameStatic(verdict));
            break;
        }
        if (AS_BOOL(verdict)) {
            jaiGCPushRoot(item);
            jaiListPush(result, item);
            jaiGCPopRoot();
        }
    }
    jaiGCPopRoots(3);
    if (!ok) return false;
    *out = OBJ_VAL(result);
    return true;
}

static bool quantify(int argc, Value *args, Value *out, bool wantAny) {
    (void)argc;
    const char *fnName = wantAny ? "any" : "all";
    Value iterVal;
    if (!jaiGetIter(args[0], &iterVal)) return false;
    if (!IS_ITER(iterVal))
        return jaiThrow(vm.cTypeError, "'%s' object is not iterable",
                        jaiTypeNameStatic(args[0]));

    jaiGCPushRoot(iterVal);
    bool result = !wantAny;
    bool ok = true;
    Value item;
    while (jaiIterNext(AS_ITER(iterVal), &item)) {
        if (!IS_BOOL(item)) {
            ok = jaiBuiltinArgTypeError(1, fnName, "an iterable of bool", item);
            break;
        }
        if (AS_BOOL(item) == wantAny) { result = wantAny; break; }
    }
    jaiGCPopRoot();
    if (!ok || vm.hasException) return false;
    *out = BOOL_VAL(result);
    return true;
}

bool nAny(int argc, Value *args, Value *out) { return quantify(argc, args, out, true); }
bool nAll(int argc, Value *args, Value *out) { return quantify(argc, args, out, false); }
