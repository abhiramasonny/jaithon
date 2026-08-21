/* object_collection.c — the pure-data container kinds: ObjBytes, ObjList,
 * ObjTuple, ObjDict/ObjSet, and ObjRange.
 *
 * Grouped together (rather than five one-file-per-kind splits) because none
 * of them is more than a couple hundred lines and none has anything private
 * that another needs kept apart from it -- unlike, say, ObjString's interning
 * policy or ObjClass's dunder cache. What they share instead is a shape: a
 * fixed or growable Value payload with allocation, mutation and (for list and
 * dict) iteration-order operations, and no behaviour that calls back into the
 * interpreter or a user method. jaiDictGet/Set/Delete and jaiSetAdd/Has/Delete
 * both sit on the same JaiTable (table.h) and share the keyHash helper below,
 * which is the one piece of logic actually specific to more than one of these
 * kinds.
 *
 * jaiNormalizeIndex is defined here, at the end of the Lists section, because
 * that is where it was first needed, but it is generic over any indexable
 * length and is called from every sequence type's indexing path across the
 * runtime (builtins_str.c, builtins_bytes.c, builtins_seq.c, vm.c) as well as
 * from object_string.c's own scalar indexing.
 *
 * sliceCount (object_internal.h) is the other half of this file's contract
 * with object_string.c: jaiListSlice below and jaiStringSlice there are the
 * only two Python-step slice implementations in the object model, and they
 * share its start/stop/step clamping.
 */

#include "vm/object/object.h"
#include "vm/object/object_internal.h"   /* sliceCount, shared with object_string.c */

#include "vm/gc.h"
#include "vm/table.h"
#include "vm/vm.h"

/* ------------------------------------------------------------------ */
/* Bytes                                                                */
/* ------------------------------------------------------------------ */

ObjBytes *jaiBytesNew(const uint8_t *data, size_t length) {
    if (length > UINT32_MAX) {
        jaiThrow(vm.cValueError,
                 "bytes of %zu bytes exceeds the maximum length", length);
        return NULL;
    }

    ObjBytes *b = (ObjBytes *)jaiAllocateObjectRaw(
        sizeof(ObjBytes) + length, OBJ_BYTES);
    b->length = (uint32_t)length;

    if (length != 0) {
        if (data != NULL)
            memcpy(b->data, data, length);
        else
            memset(b->data, 0, length);
    }

    return b;
}

/* ------------------------------------------------------------------ */
/* Lists                                                                */
/* ------------------------------------------------------------------ */

ObjList *jaiListNew(int initialCapacity) {
    ObjList *list = JAI_ALLOCATE_OBJ(ObjList, OBJ_LIST);
    /* jaiAllocateObject already zeroed items/count/capacity/version. */

    if (initialCapacity > 0) {
        jaiGCPushRoot(OBJ_VAL(list));
        jaiListReserve(list, initialCapacity);
        jaiGCPopRoot();
    }

    return list;
}

void jaiListReserve(ObjList *list, int capacity) {
    if (capacity <= list->capacity) return;
    int oldCap = list->capacity;
    list->items = JAI_GROW_ARRAY(Value, list->items, oldCap, capacity);
    list->capacity = capacity;
}

bool jaiListReserveExact(ObjList *list, int count) {
    if (count < 0) return false;
    if (count == 0) return true;
    jaiGCPushRoot(OBJ_VAL(list));
    jaiListReserve(list, count);
    jaiGCPopRoot();
    return list->capacity >= count;
}

/* Grows to hold one more item, keeping `pending` (the value about to be
 * stored, which may be the only reference to a fresh object) alive. */
static bool listGrowFor(ObjList *list, Value pending) {
    if (list->capacity > INT32_MAX / 2) {
        jaiThrow(vm.cRuntimeError,
                 "list cannot grow beyond %d items", INT32_MAX);
        return false;
    }

    jaiGCPushRoot(OBJ_VAL(list));
    jaiGCPushRoot(pending);
    jaiListReserve(list, JAI_GROW_CAP(list->capacity));
    jaiGCPopRoots(2);
    return true;
}

/* A monotone counter, not a state hash: a live iterator only asks whether the
 * list is the one it started on, and wrapping after 2^32 mutations would take
 * a single loop longer than any program runs. */
void jaiListTouch(ObjList *list) {
    list->version++;
}

void jaiListPush(ObjList *list, Value v) {
    /* The guard belongs HERE, not in the `list.push` builtin: a typed receiver
     * and an `any` one reach the list through different entry points, and
     * guarding only the builtin caught the first and missed the second -- which
     * is the case that matters, since the `any` path is the one the checker
     * cannot see. Every caller already handles vm.hasException.
     *
     * A list with no declared element type carries FIELD_KIND_ANY and this is
     * one predictable not-taken compare. */
    if (JAI_UNLIKELY(list->elemKind != FIELD_KIND_ANY) &&
        !jaiCheckKind(list->elemKind, v, "an element")) {
        return;
    }
    if (JAI_UNLIKELY(list->count >= list->capacity) &&
        !listGrowFor(list, v))
        return;

    list->items[list->count++] = v;
    list->version++;
}

Value jaiListPop(ObjList *list) {
    if (list->count == 0) {
        jaiThrow(vm.cIndexError, "pop from empty list");
        return NULL_VAL;
    }
    list->version++;
    return list->items[--list->count];
}

void jaiListInsert(ObjList *list, int idx, Value v) {
    if (JAI_UNLIKELY(list->elemKind != FIELD_KIND_ANY) &&
        !jaiCheckKind(list->elemKind, v, "an element")) {
        return;
    }
    if (idx < 0) {
        idx += list->count;
        if (idx < 0) idx = 0;
    } else if (idx > list->count) {
        idx = list->count;
    }
    if (JAI_UNLIKELY(list->count >= list->capacity) &&
        !listGrowFor(list, v))
        return;

    if (idx < list->count) {
        memmove(&list->items[idx + 1], &list->items[idx],
                sizeof(Value) * (size_t)(list->count - idx));
    }
    list->items[idx] = v;
    list->count++;
    list->version++;
}

Value jaiListRemove(ObjList *list, int idx) {
    int at;
    if (!jaiNormalizeIndex(idx, list->count, &at)) {
        jaiThrow(vm.cIndexError, "list index %d out of range for length %d", idx,
                 list->count);
        return NULL_VAL;
    }
    Value removed = list->items[at];
    if (at + 1 < list->count) {
        memmove(&list->items[at], &list->items[at + 1],
                sizeof(Value) * (size_t)(list->count - at - 1));
    }
    list->count--;
    list->version++;
    return removed;
}

ObjList *jaiListSlice(ObjList *list, int64_t start,
                          int64_t stop, int64_t step) {
    if (step == 0) {
        jaiThrow(vm.cValueError, "slice step cannot be zero");
        return NULL;
    }

    const int64_t count =
        sliceCount((int64_t)list->count, &start, &stop, &step);

    jaiGCPushRoot(OBJ_VAL(list));
    ObjList *out = jaiListNew((int)count);

    if (count > 0) {
        if (step == 1) {
            memcpy(out->items, list->items + start,
                   sizeof(Value) * (size_t)count);
            out->count = (int)count;
        } else {
            Value *dst = out->items;
            for (int64_t i = 0, idx = start; i < count; ++i, idx += step)
                dst[i] = list->items[idx];
            out->count = (int)count;
        }
    }

    jaiGCPopRoot();
    return out;
}

ObjList *jaiListConcat(ObjList *a, ObjList *b) {
    const int aCount = a->count;
    const int bCount = b->count;
    const int64_t total = (int64_t)aCount + (int64_t)bCount;

    if (total > INT32_MAX) {
        jaiThrow(vm.cRuntimeError,
                 "list cannot grow beyond %d items", INT32_MAX);
        return NULL;
    }

    jaiGCPushRoot(OBJ_VAL(a));
    jaiGCPushRoot(OBJ_VAL(b));
    ObjList *out = jaiListNew((int)total);

    if (aCount != 0)
        memcpy(out->items, a->items, sizeof(Value) * (size_t)aCount);

    if (bCount != 0)
        memcpy(out->items + aCount, b->items,
               sizeof(Value) * (size_t)bCount);

    out->count = (int)total;
    jaiGCPopRoots(2);
    return out;
}

bool jaiNormalizeIndex(int64_t raw, int length, int *out) {
    if (raw < 0) raw += (int64_t)length;

    if ((uint64_t)raw >= (uint64_t)(unsigned)length)
        return false;

    *out = (int)raw;
    return true;
}

/* ------------------------------------------------------------------ */
/* Tuples                                                               */
/* ------------------------------------------------------------------ */

ObjTuple *jaiTupleNew(const Value *items, int count) {
    if (count < 0) count = 0;

    ObjTuple *t = (ObjTuple *)jaiAllocateObjectRaw(
        sizeof(ObjTuple) + sizeof(Value) * (size_t)count, OBJ_TUPLE);
    t->count = (uint32_t)count;
    t->hash = 0;

    if (count != 0) {
        if (items != NULL) {
            memcpy(t->items, items, sizeof(Value) * (size_t)count);
        } else {
            for (int i = 0; i < count; ++i)
                t->items[i] = NULL_VAL;
        }
    }

    return t;
}

/* ------------------------------------------------------------------ */
/* Dicts and sets                                                       */
/* ------------------------------------------------------------------ */

ObjDict *jaiDictNew(void) {
    ObjDict *d = JAI_ALLOCATE_OBJ(ObjDict, OBJ_DICT);
    jaiTableInit(&d->table);
    return d;
}

/* Hash a key, refusing the ones spec §5.4 does not allow: `null`, and anything
 * that does not hash at all. The table layer cannot make that judgement itself,
 * because it is also the VM's own symbol table, where such a key is a bug and
 * not a program's mistake; it asserts. So the rejection lives here instead, on
 * the two functions every dict write and set insertion funnels through, and the
 * hash is carried down so a user `__hash__` is not run a second time. */
static inline bool keyHash(Value key, const char *role, uint64_t *hash) {
    if (IS_NULL(key))
        return jaiThrow(vm.cTypeError, "a %s cannot be null", role);

    bool ok = true;
    *hash = jaiValueHashFast(key, &ok);
    if (ok) return true;
    if (vm.hasException) return false;

    return jaiThrow(vm.cTypeError, "unhashable type: '%s'",
                    jaiTypeNameStatic(key));
}

bool jaiDictGet(ObjDict *d, Value key, Value *out) {
    return jaiTableGet(&d->table, key, out);
}

bool jaiDictSet(ObjDict *d, Value key, Value value) {
    /* Same reasoning as jaiListPush: the lowest common point every writer
     * reaches, so an `any` receiver cannot slip past. */
    if (JAI_UNLIKELY(d->keyKind != FIELD_KIND_ANY) &&
        !jaiCheckKind(d->keyKind, key, "a key")) {
        return false;
    }
    if (JAI_UNLIKELY(d->valKind != FIELD_KIND_ANY) &&
        !jaiCheckKind(d->valKind, value, "a value")) {
        return false;
    }
    uint64_t hash;
    if (!keyHash(key, "dict key", &hash)) return false;
    return jaiTableSetHashed(&d->table, key, hash, value);
}

bool jaiDictDelete(ObjDict *d, Value key) {
    return jaiTableDelete(&d->table, key);
}

/* Collects one column of the dict into a fresh list. `wantValues` picks the
 * column; both are gathered in one pass so the table is walked once. */
static ObjList *dictColumn(ObjDict *d, bool wantValues) {
    jaiGCPushRoot(OBJ_VAL(d));
    ObjList *out = jaiListNew(d->table.count);
    jaiGCPushRoot(OBJ_VAL(out));

    int slot = 0;
    int count = 0;
    Value key, value;

    while (jaiTableNext(&d->table, &slot, &key, &value))
        out->items[count++] = wantValues ? value : key;

    out->count = count;
    jaiGCPopRoots(2);
    return out;
}

ObjList *jaiDictKeys(ObjDict *d)   { return dictColumn(d, false); }
ObjList *jaiDictValues(ObjDict *d) { return dictColumn(d, true); }

ObjList *jaiDictItems(ObjDict *d) {
    jaiGCPushRoot(OBJ_VAL(d));
    ObjList *out = jaiListNew(d->table.count);
    jaiGCPushRoot(OBJ_VAL(out));

    int slot = 0;
    int count = 0;
    Value key, value;

    while (jaiTableNext(&d->table, &slot, &key, &value)) {
        Value pair[2] = {key, value};
        ObjTuple *tuple = jaiTupleNew(pair, 2);
        out->items[count++] = OBJ_VAL(tuple);
        out->count = count;  /* tuple is reachable before the next allocation */
    }

    jaiGCPopRoots(2);
    return out;
}

ObjSet *jaiSetNew(void) {
    ObjSet *s = JAI_ALLOCATE_OBJ(ObjSet, OBJ_SET);
    jaiTableInit(&s->table);
    return s;
}

bool jaiSetAdd(ObjSet *s, Value v) {
    uint64_t hash;
    if (!keyHash(v, "set element", &hash)) return false;
    return jaiTableSetHashed(&s->table, v, hash, NULL_VAL);
}

bool jaiSetHas(ObjSet *s, Value v) {
    Value ignored;
    return jaiTableGet(&s->table, v, &ignored);
}

bool jaiSetDelete(ObjSet *s, Value v) {
    return jaiTableDelete(&s->table, v);
}

/* ------------------------------------------------------------------ */
/* Ranges                                                               */
/* ------------------------------------------------------------------ */

ObjRange *jaiRangeNew(int64_t start, int64_t stop, int64_t step,
                      bool inclusive) {
    ObjRange *r = JAI_ALLOCATE_OBJ(ObjRange, OBJ_RANGE);
    r->start = start;
    r->stop = stop;
    r->step = (step == 0) ? 1 : step;
    r->inclusive = inclusive;
    return r;
}

int64_t jaiRangeLength(ObjRange *r) {
    const int64_t step = r->step;
    if (step == 0) return 0;

    uint64_t span;

    if (step > 0) {
        if (r->start > r->stop) return 0;
        span = (uint64_t)r->stop - (uint64_t)r->start;
    } else {
        if (r->start < r->stop) return 0;
        span = (uint64_t)r->start - (uint64_t)r->stop;
    }

    if (span == 0) return r->inclusive ? 1 : 0;

    /* Unit-stride ranges are by far the common case and need no division. */
    if (step == 1 || step == -1) {
        uint64_t n = span;
        if (r->inclusive) {
            if (n == UINT64_MAX) return INT64_MAX;
            ++n;
        }
        return n > (uint64_t)INT64_MAX ? INT64_MAX : (int64_t)n;
    }

    const uint64_t stride =
        step > 0 ? (uint64_t)step : 0u - (uint64_t)step;

    uint64_t n;
    if (r->inclusive) {
        /* floor(span / stride) + 1 avoids span+1 overflow. */
        n = span / stride + 1u;
    } else {
        /* ceil(span / stride), written overflow-free. */
        n = span / stride + (span % stride != 0);
    }

    return n > (uint64_t)INT64_MAX ? INT64_MAX : (int64_t)n;
}
