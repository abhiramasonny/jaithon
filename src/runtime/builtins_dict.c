/* builtins_dict.c — the `dict` and `set` method tables and the
 * `__prim__.dict_*` and `__prim__.set_*` surfaces (spec Appendix C).
 *
 * The two live together because they are one data structure seen twice: a set
 * is a table whose values are ignored, every set operation is written over the
 * same JaiTable walk, and both answer the same question about a key before
 * they touch it — does it hash, and is it non-null. builtins_seq.h holds what
 * they share with list, tuple and range.
 */

#include "builtins_seq.h"
#include "methods.h"
#include "runtime.h"

#include "../vm/gc.h"

/* ------------------------------------------------------------------ */
/* Receivers and arguments                                              */
/* ------------------------------------------------------------------ */

static bool selfDict(Value *args, const char *fnName, ObjDict **out) {
    if (!IS_DICT(args[0])) return jaiSeqReceiverError(fnName, "dict", args[0]);
    *out = AS_DICT(args[0]);
    return true;
}

static bool selfSet(Value *args, const char *fnName, ObjSet **out) {
    if (!IS_SET(args[0])) return jaiSeqReceiverError(fnName, "set", args[0]);
    *out = AS_SET(args[0]);
    return true;
}

static bool argSet(Value v, int index, const char *fnName, ObjSet **out) {
    if (!IS_SET(v)) {
        return jaiThrow(vm.cTypeError, "%s() argument %d: expected set, got %s",
                        fnName, index, jaiTypeNameStatic(v));
    }
    *out = AS_SET(v);
    return true;
}

/* ------------------------------------------------------------------ */
/* Shared operations                                                    */
/* ------------------------------------------------------------------ */

static bool missingKeyError(const char *fnName, const char *role, Value key) {
    ObjString *text = jaiValueToRepr(key);
    if (text == NULL) return false;      /* the repr raised; that error stands */
    return jaiThrow(vm.cKeyError, "%s(): %s not found: %s", fnName, role,
                    text->chars);
}

/* Every element of a set, in table order. Nothing here allocates after the
 * result is sized, so the walk cannot be disturbed by a collection. */
static ObjList *setElements(ObjSet *s) {
    jaiGCPushRoot(OBJ_VAL(s));
    ObjList *out = jaiListNew(s->table.count);
    jaiGCPushRoot(OBJ_VAL(out));

    int slot = 0;
    Value k, v;
    while (jaiTableNext(&s->table, &slot, &k, &v) && out->count < out->capacity) {
        out->items[out->count++] = k;
    }
    jaiGCPopRoots(2);
    return out;
}

/* The operand of a set operation: a set is taken as it is, any other iterable
 * is copied into a temporary one. Either way exactly one temp root is pushed,
 * which the caller pops when it is done. */
static bool operandSet(Value v, const char *fnName, ObjSet **out) {
    if (IS_SET(v)) {
        jaiGCPushRoot(v);
        *out = AS_SET(v);
        return true;
    }

    ObjList *items = jaiSeqCollectIterable(v);
    if (items == NULL) return false;
    jaiGCPushRoot(OBJ_VAL(items));
    ObjSet *s = jaiSetNew();
    jaiGCPushRoot(OBJ_VAL(s));

    bool ok = true;
    for (int i = 0; i < items->count; i++) {
        if (!jaiSeqHashableKey(items->items[i], fnName, "set element")) {
            ok = false;
            break;
        }
        (void)jaiSetAdd(s, items->items[i]);
        if (vm.hasException) {
            ok = false;
            break;
        }
    }
    jaiGCPopRoots(2);
    if (!ok) return false;

    /* No allocation happens between the pops and this push, so `s` cannot be
     * collected in the gap. */
    jaiGCPushRoot(OBJ_VAL(s));
    *out = s;
    return true;
}

/* ------------------------------------------------------------------ */
/* dict                                                                 */
/* ------------------------------------------------------------------ */

static bool dictLen(int argc, Value *args, Value *out) {
    (void)argc;
    ObjDict *self;
    if (!selfDict(args, "dict.len", &self)) return false;
    *out = INT_VAL(self->table.count);
    return true;
}

/* get(k) is null when the key is absent; get(k, d) is `d`. */
static bool dictGet(int argc, Value *args, Value *out) {
    ObjDict *self;
    if (!selfDict(args, "dict.get", &self)) return false;
    if (!jaiSeqHashableKey(args[1], "dict.get", "key")) return false;

    Value found;
    if (jaiDictGet(self, args[1], &found)) {
        *out = found;
        return true;
    }
    if (vm.hasException) return false;
    *out = jaiSeqOptArg(argc, args, 2);
    return true;
}

static bool dictSet(int argc, Value *args, Value *out) {
    (void)argc;
    ObjDict *self;
    if (!selfDict(args, "dict.set", &self)) return false;
    if (!jaiSeqHashableKey(args[1], "dict.set", "key")) return false;
    (void)jaiDictSet(self, args[1], args[2]);
    if (vm.hasException) return false;
    *out = args[0];
    return true;
}

static bool dictHas(int argc, Value *args, Value *out) {
    (void)argc;
    ObjDict *self;
    if (!selfDict(args, "dict.has", &self)) return false;
    if (!jaiSeqHashableKey(args[1], "dict.has", "key")) return false;
    bool found = jaiDictGet(self, args[1], NULL);
    if (vm.hasException) return false;
    *out = BOOL_VAL(found);
    return true;
}

/* delete answers whether anything was there; pop insists that there was. */
static bool dictDelete(int argc, Value *args, Value *out) {
    (void)argc;
    ObjDict *self;
    if (!selfDict(args, "dict.remove", &self)) return false;
    if (!jaiSeqHashableKey(args[1], "dict.remove", "key")) return false;
    bool removed = jaiDictDelete(self, args[1]);
    if (vm.hasException) return false;
    *out = BOOL_VAL(removed);
    return true;
}

static bool dictPop(int argc, Value *args, Value *out) {
    ObjDict *self;
    if (!selfDict(args, "dict.pop", &self)) return false;
    if (!jaiSeqHashableKey(args[1], "dict.pop", "key")) return false;

    Value found;
    if (jaiDictGet(self, args[1], &found)) {
        jaiGCPushRoot(found);
        (void)jaiDictDelete(self, args[1]);
        jaiGCPopRoot();
        if (vm.hasException) return false;
        *out = found;
        return true;
    }
    if (vm.hasException) return false;
    if (argc >= 3) {
        *out = args[2];
        return true;
    }
    return missingKeyError("dict.pop", "key", args[1]);
}

static bool dictClear(int argc, Value *args, Value *out) {
    (void)argc;
    ObjDict *self;
    if (!selfDict(args, "dict.clear", &self)) return false;
    jaiTableClear(&self->table);
    *out = args[0];
    return true;
}

static bool dictKeys(int argc, Value *args, Value *out) {
    (void)argc;
    ObjDict *self;
    if (!selfDict(args, "dict.keys", &self)) return false;
    ObjList *keys = jaiDictKeys(self);
    if (keys == NULL) return false;
    *out = OBJ_VAL(keys);
    return true;
}

static bool dictValues(int argc, Value *args, Value *out) {
    (void)argc;
    ObjDict *self;
    if (!selfDict(args, "dict.values", &self)) return false;
    ObjList *values = jaiDictValues(self);
    if (values == NULL) return false;
    *out = OBJ_VAL(values);
    return true;
}

static bool dictItems(int argc, Value *args, Value *out) {
    (void)argc;
    ObjDict *self;
    if (!selfDict(args, "dict.items", &self)) return false;
    ObjList *items = jaiDictItems(self);
    if (items == NULL) return false;
    *out = OBJ_VAL(items);
    return true;
}

static bool dictUpdate(int argc, Value *args, Value *out) {
    (void)argc;
    ObjDict *self;
    ObjDict *other;
    if (!selfDict(args, "dict.update", &self)) return false;
    if (!jaiArgDict(args[1], 1, "dict.update", &other)) return false;
    /* AddAll reuses the stored hashes, so no user __hash__ runs mid-copy. */
    jaiTableAddAll(&other->table, &self->table);
    *out = args[0];
    return true;
}

static bool dictCopy(int argc, Value *args, Value *out) {
    (void)argc;
    ObjDict *self;
    if (!selfDict(args, "dict.copy", &self)) return false;

    jaiGCPushRoot(args[0]);
    ObjDict *result = jaiDictNew();
    jaiGCPushRoot(OBJ_VAL(result));
    jaiTableAddAll(&self->table, &result->table);
    jaiGCPopRoots(2);
    *out = OBJ_VAL(result);
    return true;
}

/* merge is copy + update: `other` wins on the keys they share. */
static bool dictMerge(int argc, Value *args, Value *out) {
    (void)argc;
    ObjDict *self;
    ObjDict *other;
    if (!selfDict(args, "dict.merge", &self)) return false;
    if (!jaiArgDict(args[1], 1, "dict.merge", &other)) return false;

    jaiGCPushRoot(args[0]);
    jaiGCPushRoot(args[1]);
    ObjDict *result = jaiDictNew();
    jaiGCPushRoot(OBJ_VAL(result));
    jaiTableAddAll(&self->table, &result->table);
    jaiTableAddAll(&other->table, &result->table);
    jaiGCPopRoots(3);
    *out = OBJ_VAL(result);
    return true;
}

static bool dictGetOrInsert(int argc, Value *args, Value *out) {
    (void)argc;
    ObjDict *self;
    if (!selfDict(args, "dict.get_or_insert", &self)) return false;
    if (!jaiSeqHashableKey(args[1], "dict.get_or_insert", "key")) return false;

    Value found;
    if (jaiDictGet(self, args[1], &found)) {
        *out = found;
        return true;
    }
    if (vm.hasException) return false;
    (void)jaiDictSet(self, args[1], args[2]);
    if (vm.hasException) return false;
    *out = args[2];
    return true;
}

/* The entries are snapshotted first: the callback is user code and may insert
 * into the dict, which would rehash the table under the walk. */
static bool dictMapValues(int argc, Value *args, Value *out) {
    (void)argc;
    ObjDict *self;
    if (!selfDict(args, "dict.map_values", &self)) return false;
    if (!jaiArgCallable(args[1], 1, "dict.map_values")) return false;

    ObjList *entries = jaiDictItems(self);
    if (entries == NULL) return false;
    jaiGCPushRoot(OBJ_VAL(entries));
    ObjDict *result = jaiDictNew();
    jaiGCPushRoot(OBJ_VAL(result));

    bool ok = true;
    for (int i = 0; i < entries->count; i++) {
        ObjTuple *pair = AS_TUPLE(entries->items[i]);
        Value arg = pair->items[1], mapped;
        if (!jaiCallValue(args[1], 1, &arg, &mapped)) {
            ok = false;
            break;
        }
        jaiGCPushRoot(mapped);
        (void)jaiDictSet(result, pair->items[0], mapped);
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

/* filter's predicate takes the key and the value, in that order. */
static bool dictFilter(int argc, Value *args, Value *out) {
    (void)argc;
    ObjDict *self;
    if (!selfDict(args, "dict.filter", &self)) return false;
    if (!jaiArgCallable(args[1], 1, "dict.filter")) return false;

    ObjList *entries = jaiDictItems(self);
    if (entries == NULL) return false;
    jaiGCPushRoot(OBJ_VAL(entries));
    ObjDict *result = jaiDictNew();
    jaiGCPushRoot(OBJ_VAL(result));

    bool ok = true;
    for (int i = 0; i < entries->count; i++) {
        ObjTuple *pair = AS_TUPLE(entries->items[i]);
        Value callArgs[2] = {pair->items[0], pair->items[1]};
        Value verdict;
        if (!jaiCallValue(args[1], 2, callArgs, &verdict)) {
            ok = false;
            break;
        }
        if (!IS_BOOL(verdict)) {
            ok = jaiThrow(vm.cTypeError,
                          "dict.filter(): the predicate must return bool, not %s",
                          jaiTypeNameStatic(verdict));
            break;
        }
        if (!AS_BOOL(verdict)) continue;
        (void)jaiDictSet(result, pair->items[0], pair->items[1]);
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

/* ------------------------------------------------------------------ */
/* set                                                                  */
/* ------------------------------------------------------------------ */

static bool setLen(int argc, Value *args, Value *out) {
    (void)argc;
    ObjSet *self;
    if (!selfSet(args, "set.len", &self)) return false;
    *out = INT_VAL(self->table.count);
    return true;
}

static bool setAdd(int argc, Value *args, Value *out) {
    (void)argc;
    ObjSet *self;
    if (!selfSet(args, "set.add", &self)) return false;
    if (!jaiSeqHashableKey(args[1], "set.add", "set element")) return false;
    (void)jaiSetAdd(self, args[1]);
    if (vm.hasException) return false;
    *out = args[0];
    return true;
}

/* remove insists the element was there; discard reports whether it was. */
static bool setRemove(int argc, Value *args, Value *out) {
    (void)argc;
    ObjSet *self;
    if (!selfSet(args, "set.remove", &self)) return false;
    if (!jaiSeqHashableKey(args[1], "set.remove", "set element")) return false;
    bool removed = jaiSetDelete(self, args[1]);
    if (vm.hasException) return false;
    if (!removed) return missingKeyError("set.remove", "element", args[1]);
    *out = args[0];
    return true;
}

static bool setDiscard(int argc, Value *args, Value *out) {
    (void)argc;
    ObjSet *self;
    if (!selfSet(args, "set.discard", &self)) return false;
    if (!jaiSeqHashableKey(args[1], "set.discard", "set element")) return false;
    bool removed = jaiSetDelete(self, args[1]);
    if (vm.hasException) return false;
    *out = BOOL_VAL(removed);
    return true;
}

static bool setHasMethod(int argc, Value *args, Value *out) {
    (void)argc;
    ObjSet *self;
    if (!selfSet(args, "set.has", &self)) return false;
    if (!jaiSeqHashableKey(args[1], "set.has", "set element")) return false;
    bool found = jaiSetHas(self, args[1]);
    if (vm.hasException) return false;
    *out = BOOL_VAL(found);
    return true;
}

static bool setClear(int argc, Value *args, Value *out) {
    (void)argc;
    ObjSet *self;
    if (!selfSet(args, "set.clear", &self)) return false;
    jaiTableClear(&self->table);
    *out = args[0];
    return true;
}

typedef enum { SET_UNION, SET_INTERSECTION, SET_DIFFERENCE, SET_SYMMETRIC } SetOp;

/* The four algebraic operations, which differ only in which side contributes
 * an element and under what membership test. */
static bool setCombine(int argc, Value *args, Value *out, SetOp op,
                       const char *fnName) {
    (void)argc;
    ObjSet *self;
    if (!selfSet(args, fnName, &self)) return false;

    ObjSet *other;
    if (!operandSet(args[1], fnName, &other)) return false;   /* +1 root */

    ObjSet *result = jaiSetNew();
    jaiGCPushRoot(OBJ_VAL(result));

    bool ok = true;
    if (op == SET_UNION) {
        jaiTableAddAll(&self->table, &result->table);
        jaiTableAddAll(&other->table, &result->table);
    } else {
        ObjList *mine = setElements(self);
        jaiGCPushRoot(OBJ_VAL(mine));
        for (int i = 0; i < mine->count; i++) {
            bool inOther = jaiSetHas(other, mine->items[i]);
            if (vm.hasException) { ok = false; break; }
            bool keep = (op == SET_INTERSECTION) ? inOther : !inOther;
            if (!keep) continue;
            (void)jaiSetAdd(result, mine->items[i]);
            if (vm.hasException) { ok = false; break; }
        }
        jaiGCPopRoot();

        if (ok && op == SET_SYMMETRIC) {
            ObjList *theirs = setElements(other);
            jaiGCPushRoot(OBJ_VAL(theirs));
            for (int i = 0; i < theirs->count; i++) {
                bool inMine = jaiSetHas(self, theirs->items[i]);
                if (vm.hasException) { ok = false; break; }
                if (inMine) continue;
                (void)jaiSetAdd(result, theirs->items[i]);
                if (vm.hasException) { ok = false; break; }
            }
            jaiGCPopRoot();
        }
    }

    jaiGCPopRoots(2);
    if (!ok) return false;
    *out = OBJ_VAL(result);
    return true;
}

static bool setUnion(int argc, Value *args, Value *out) {
    return setCombine(argc, args, out, SET_UNION, "set.union");
}

static bool setIntersection(int argc, Value *args, Value *out) {
    return setCombine(argc, args, out, SET_INTERSECTION, "set.intersection");
}

static bool setDifference(int argc, Value *args, Value *out) {
    return setCombine(argc, args, out, SET_DIFFERENCE, "set.difference");
}

static bool setSymmetricDifference(int argc, Value *args, Value *out) {
    return setCombine(argc, args, out, SET_SYMMETRIC, "set.symmetric_difference");
}

/* Containment one way or the other: `wantSuper` swaps which side is walked. */
static bool setContainment(int argc, Value *args, Value *out, bool wantSuper,
                           const char *fnName) {
    (void)argc;
    ObjSet *self;
    if (!selfSet(args, fnName, &self)) return false;

    ObjSet *other;
    if (!operandSet(args[1], fnName, &other)) return false;   /* +1 root */

    ObjSet *walked = wantSuper ? other : self;
    ObjSet *tested = wantSuper ? self : other;

    ObjList *items = setElements(walked);
    jaiGCPushRoot(OBJ_VAL(items));
    bool result = true;
    bool ok = true;
    for (int i = 0; i < items->count; i++) {
        bool present = jaiSetHas(tested, items->items[i]);
        if (vm.hasException) { ok = false; break; }
        if (!present) { result = false; break; }
    }
    jaiGCPopRoots(2);
    if (!ok) return false;
    *out = BOOL_VAL(result);
    return true;
}

static bool setIsSubset(int argc, Value *args, Value *out) {
    return setContainment(argc, args, out, false, "set.is_subset");
}

static bool setIsSuperset(int argc, Value *args, Value *out) {
    return setContainment(argc, args, out, true, "set.is_superset");
}

static bool setCopy(int argc, Value *args, Value *out) {
    (void)argc;
    ObjSet *self;
    if (!selfSet(args, "set.copy", &self)) return false;

    jaiGCPushRoot(args[0]);
    ObjSet *result = jaiSetNew();
    jaiGCPushRoot(OBJ_VAL(result));
    jaiTableAddAll(&self->table, &result->table);
    jaiGCPopRoots(2);
    *out = OBJ_VAL(result);
    return true;
}

static bool setToList(int argc, Value *args, Value *out) {
    (void)argc;
    ObjSet *self;
    if (!selfSet(args, "set.to_list", &self)) return false;
    *out = OBJ_VAL(setElements(self));
    return true;
}

/* ------------------------------------------------------------------ */
/* Method tables                                                        */
/* ------------------------------------------------------------------ */

static const JaiSeqMethod kDictMethods[] = {
    JAI_METHOD("clear",         dictClear,       1, 1),
    /* `contains` alongside `has`: every other container spells membership that
     * way (list, str, bytes, tuple, range), and the spec's own §5.5 example
     * calls it on a set. `has` stays because all of lib/std was written to it. */
    JAI_METHOD("contains",      dictHas,         2, 2),
    JAI_METHOD("copy",          dictCopy,        1, 1),
    JAI_METHOD("filter",        dictFilter,      2, 2),
    JAI_METHOD("get",           dictGet,         2, 3),
    JAI_METHOD("get_or_insert", dictGetOrInsert, 3, 3),
    JAI_METHOD("has",           dictHas,         2, 2),
    JAI_METHOD("items",         dictItems,       1, 1),
    JAI_METHOD("iter",          jaiSeqValueIter,       1, 1),
    JAI_METHOD("keys",          dictKeys,        1, 1),
    JAI_METHOD("len",           dictLen,         1, 1),
    JAI_METHOD("map_values",    dictMapValues,   2, 2),
    JAI_METHOD("merge",         dictMerge,       2, 2),
    JAI_METHOD("pop",           dictPop,         2, 3),
    /* `remove`, not `delete`: list and set both spell it that way, and every
     * caller in lib/std was written against that name. */
    JAI_METHOD("remove",        dictDelete,      2, 2),
    JAI_METHOD("set",           dictSet,         3, 3),
    JAI_METHOD("update",        dictUpdate,      2, 2),
    JAI_METHOD("values",        dictValues,      1, 1),
};

static const JaiSeqMethod kSetMethods[] = {
    JAI_METHOD("add",                  setAdd,                 2, 2),
    JAI_METHOD("clear",                setClear,               1, 1),
    JAI_METHOD("contains",             setHasMethod,           2, 2),
    JAI_METHOD("copy",                 setCopy,                1, 1),
    JAI_METHOD("difference",           setDifference,          2, 2),
    JAI_METHOD("discard",              setDiscard,             2, 2),
    JAI_METHOD("has",                  setHasMethod,           2, 2),
    JAI_METHOD("intersection",         setIntersection,        2, 2),
    JAI_METHOD("is_subset",            setIsSubset,            2, 2),
    JAI_METHOD("is_superset",          setIsSuperset,          2, 2),
    JAI_METHOD("iter",                 jaiSeqValueIter,              1, 1),
    JAI_METHOD("len",                  setLen,                 1, 1),
    JAI_METHOD("remove",               setRemove,              2, 2),
    JAI_METHOD("symmetric_difference", setSymmetricDifference, 2, 2),
    JAI_METHOD("to_list",              setToList,              1, 1),
    JAI_METHOD("union",                setUnion,               2, 2),
};

bool jaiDictMethod(Value receiver, ObjString *name, Value *out) {
    return JAI_BIND_FROM(kDictMethods, receiver, name, out);
}

bool jaiSetMethod(Value receiver, ObjString *name, Value *out) {
    return JAI_BIND_FROM(kSetMethods, receiver, name, out);
}

/* ------------------------------------------------------------------ */
/* __prim__.dict_* and __prim__.set_*                                   */
/* ------------------------------------------------------------------ */

static bool primDictNew(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    *out = OBJ_VAL(jaiDictNew());
    return true;
}

/* dict_get(d, k) is null when absent; the third argument replaces that. */
static bool primDictGet(int argc, Value *args, Value *out) {
    ObjDict *dict;
    if (!jaiArgDict(args[0], 1, "dict_get", &dict)) return false;
    if (!jaiSeqHashableKey(args[1], "dict_get", "key")) return false;

    Value found;
    if (jaiDictGet(dict, args[1], &found)) {
        *out = found;
        return true;
    }
    if (vm.hasException) return false;
    *out = jaiSeqOptArg(argc, args, 2);
    return true;
}

static bool primDictSet(int argc, Value *args, Value *out) {
    (void)argc;
    ObjDict *dict;
    if (!jaiArgDict(args[0], 1, "dict_set", &dict)) return false;
    if (!jaiSeqHashableKey(args[1], "dict_set", "key")) return false;
    bool isNew = jaiDictSet(dict, args[1], args[2]);
    if (vm.hasException) return false;
    *out = BOOL_VAL(isNew);
    return true;
}

static bool primDictDel(int argc, Value *args, Value *out) {
    (void)argc;
    ObjDict *dict;
    if (!jaiArgDict(args[0], 1, "dict_del", &dict)) return false;
    if (!jaiSeqHashableKey(args[1], "dict_del", "key")) return false;
    bool removed = jaiDictDelete(dict, args[1]);
    if (vm.hasException) return false;
    *out = BOOL_VAL(removed);
    return true;
}

static bool primDictHas(int argc, Value *args, Value *out) {
    (void)argc;
    ObjDict *dict;
    if (!jaiArgDict(args[0], 1, "dict_has", &dict)) return false;
    if (!jaiSeqHashableKey(args[1], "dict_has", "key")) return false;
    bool found = jaiDictGet(dict, args[1], NULL);
    if (vm.hasException) return false;
    *out = BOOL_VAL(found);
    return true;
}

static bool primDictLen(int argc, Value *args, Value *out) {
    (void)argc;
    ObjDict *dict;
    if (!jaiArgDict(args[0], 1, "dict_len", &dict)) return false;
    *out = INT_VAL(dict->table.count);
    return true;
}

static bool primDictKeys(int argc, Value *args, Value *out) {
    (void)argc;
    ObjDict *dict;
    if (!jaiArgDict(args[0], 1, "dict_keys", &dict)) return false;
    ObjList *keys = jaiDictKeys(dict);
    if (keys == NULL) return false;
    *out = OBJ_VAL(keys);
    return true;
}

static bool primDictValues(int argc, Value *args, Value *out) {
    (void)argc;
    ObjDict *dict;
    if (!jaiArgDict(args[0], 1, "dict_values", &dict)) return false;
    ObjList *values = jaiDictValues(dict);
    if (values == NULL) return false;
    *out = OBJ_VAL(values);
    return true;
}

static bool primSetNew(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    *out = OBJ_VAL(jaiSetNew());
    return true;
}

static bool primSetAdd(int argc, Value *args, Value *out) {
    (void)argc;
    ObjSet *set;
    if (!argSet(args[0], 1, "set_add", &set)) return false;
    if (!jaiSeqHashableKey(args[1], "set_add", "set element")) return false;
    bool isNew = jaiSetAdd(set, args[1]);
    if (vm.hasException) return false;
    *out = BOOL_VAL(isNew);
    return true;
}

static bool primSetDel(int argc, Value *args, Value *out) {
    (void)argc;
    ObjSet *set;
    if (!argSet(args[0], 1, "set_del", &set)) return false;
    if (!jaiSeqHashableKey(args[1], "set_del", "set element")) return false;
    bool removed = jaiSetDelete(set, args[1]);
    if (vm.hasException) return false;
    *out = BOOL_VAL(removed);
    return true;
}

static bool primSetHas(int argc, Value *args, Value *out) {
    (void)argc;
    ObjSet *set;
    if (!argSet(args[0], 1, "set_has", &set)) return false;
    if (!jaiSeqHashableKey(args[1], "set_has", "set element")) return false;
    bool found = jaiSetHas(set, args[1]);
    if (vm.hasException) return false;
    *out = BOOL_VAL(found);
    return true;
}

static bool primSetLen(int argc, Value *args, Value *out) {
    (void)argc;
    ObjSet *set;
    if (!argSet(args[0], 1, "set_len", &set)) return false;
    *out = INT_VAL(set->table.count);
    return true;
}

/* ------------------------------------------------------------------ */
/* Registration                                                         */
/* ------------------------------------------------------------------ */

/* The set primitives ride along here: Appendix C lists them beside the dict
 * ones, and runtime.h declares no separate registrar for them. */
void jaiRegisterDictPrimitives(void) {
    jaiDefineNative("__prim__.dict_new",    primDictNew,    0, 0);
    jaiDefineNative("__prim__.dict_get",    primDictGet,    2, 3);
    jaiDefineNative("__prim__.dict_set",    primDictSet,    3, 3);
    jaiDefineNative("__prim__.dict_del",    primDictDel,    2, 2);
    jaiDefineNative("__prim__.dict_has",    primDictHas,    2, 2);
    jaiDefineNative("__prim__.dict_len",    primDictLen,    1, 1);
    jaiDefineNative("__prim__.dict_keys",   primDictKeys,   1, 1);
    jaiDefineNative("__prim__.dict_values", primDictValues, 1, 1);

    jaiDefineNative("__prim__.set_new", primSetNew, 0, 0);
    jaiDefineNative("__prim__.set_add", primSetAdd, 2, 2);
    jaiDefineNative("__prim__.set_del", primSetDel, 2, 2);
    jaiDefineNative("__prim__.set_has", primSetHas, 2, 2);
    jaiDefineNative("__prim__.set_len", primSetLen, 1, 1);
}
