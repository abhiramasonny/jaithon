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

#include <stdlib.h>

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
    size_t w = jaiListStoreWidth(list->stg);
    list->items = jaiRealloc(list->items, w * (size_t)list->capacity,
                             w * (size_t)capacity);
    list->capacity = capacity;
}

/* Picks the storage `list[T]` asks for. Called from OP_ELEM_KIND, on the
 * literal, the instant it is built -- so the usual case is an empty list and
 * the whole job is dropping a reservation made in the wrong width. A literal
 * with elements in it (`var xs: list[int] = [1, 2, 3]`) is converted instead,
 * and refuses if any element does not fit: FIELD_KIND_FLOAT accepts an int as
 * well as a float, so `[1, 2.5]` is a legal `list[float]` whose first element
 * would widen -- which is fine -- but nothing else may be reinterpreted.
 *
 * An unmodelled kind (str, list, instance, or `any` re-stamped over a list
 * that was already specialised) goes back to boxed rather than being ignored,
 * so `stg` never disagrees with `elemKind` about what the list may hold. */
/* An A/B switch, and the honest way to measure this: the machine runs several
 * agents at once and a load average of four moves a benchmark by more than the
 * change does, so before and after have to be the same binary minutes apart.
 * JAITHON_LIST_UNBOX=0 leaves every list boxed -- the tier then emits its
 * storage guards against BOXED and everything else is as it was. */
static bool unboxEnabled(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("JAITHON_LIST_UNBOX");
        on = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return on != 0;
}

bool jaiListUnboxOn(void) { return unboxEnabled(); }

void jaiListSpecialise(ObjList *list, uint8_t elemKind) {
    uint8_t stg;
    if (!unboxEnabled()) return;
    switch ((FieldKind)elemKind) {
    case FIELD_KIND_INT:   stg = LIST_STORE_I64; break;
    case FIELD_KIND_FLOAT: stg = LIST_STORE_F64; break;
    case FIELD_KIND_BOOL:  stg = LIST_STORE_U8;  break;
    default:               stg = LIST_STORE_BOXED; break;
    }
    if (stg == list->stg) return;
    if (stg == LIST_STORE_BOXED) { (void)jaiListBox(list); return; }

    /* EMPTY LISTS ONLY, and the reason is agreement rather than difficulty.
     *
     * The compiled tier has its own OP_ELEM_KIND arm, and it cannot convert a
     * list that already holds elements: the conversion reallocates, and a
     * compiled frame's other live values are not rooted at that point. If the
     * interpreter converted where the tier did not, the same literal would
     * come out unboxed or boxed depending on which tier happened to be running
     * -- and a loop form that pinned one storage is then DENIED ENTRY for
     * every list built by the other. tests/bench's heat_2d plate is built by
     * both, half its rows each way, and the stencil loop over it fell back to
     * the interpreter: 2.5 BILLION interpreted instructions against 2.9
     * million, a 765ms run turning into 8.5s.
     *
     * So neither tier converts, and `var xs: list[int] = [1, 2, 3]` stays
     * boxed while `var xs: list[int] = []` does not. The empty case is the one
     * that matters anyway: it is how this codebase builds every list it then
     * pushes onto. */
    if (list->count != 0) return;

    /* Whatever was reserved was reserved at the old width. Nothing has been
     * written into it, so dropping it costs an allocation the first push would
     * have made anyway. */
    if (list->capacity != 0) {
        JAI_FREE_ARRAY(char, list->items,
                       (size_t)list->capacity * jaiListStoreWidth(list->stg));
        list->items = NULL;
        list->capacity = 0;
    }
    list->stg = stg;
}

Value *jaiListBox(ObjList *list) {
    if (list->stg == LIST_STORE_BOXED) return (Value *)list->items;

    uint8_t was = list->stg;
    int n = list->count, cap = list->capacity;
    /* Read out of the old array before the new one is allocated: allocating
     * can collect, and a half-converted list is not a shape the marker can
     * read. The list itself is rooted for the same reason. */
    jaiGCPushRoot(OBJ_VAL(list));
    Value *boxed = cap > 0 ? JAI_GROW_ARRAY(Value, NULL, 0, cap) : NULL;
    jaiGCPopRoot();

    for (int i = 0; i < n; i++) boxed[i] = jaiListGet(list, i);

    JAI_FREE_ARRAY(char, list->items, (size_t)cap * jaiListStoreWidth(was));
    list->items = boxed;
    list->stg = LIST_STORE_BOXED;
    return boxed;
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

    if (JAI_UNLIKELY(!jaiListStoreAccepts(list, v))) (void)jaiListBox(list);
    jaiListSetRaw(list, list->count++, v);
    list->version++;
}

Value jaiListPop(ObjList *list) {
    if (list->count == 0) {
        jaiThrow(vm.cIndexError, "pop from empty list");
        return NULL_VAL;
    }
    list->version++;
    return jaiListGet(list, --list->count);
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

    if (JAI_UNLIKELY(!jaiListStoreAccepts(list, v))) (void)jaiListBox(list);
    size_t w = jaiListStoreWidth(list->stg);
    if (idx < list->count) {
        char *base = (char *)list->items;
        memmove(base + w * (size_t)(idx + 1), base + w * (size_t)idx,
                w * (size_t)(list->count - idx));
    }
    jaiListSetRaw(list, idx, v);
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
    Value removed = jaiListGet(list, at);
    if (at + 1 < list->count) {
        size_t w = jaiListStoreWidth(list->stg);
        char *base = (char *)list->items;
        memmove(base + w * (size_t)at, base + w * (size_t)(at + 1),
                w * (size_t)(list->count - at - 1));
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
    ObjList *out = jaiListNew(0);
    /* The slice of a typed list is a list of the same type, so it keeps the
     * storage rather than boxing on the way out -- `xs[a:b]` inside a loop
     * would otherwise be a de-specialisation the program never asked for.
     * Set before reserving: the width has to be right first.
     *
     * From the SOURCE'S STORAGE, not from its elemKind. The two disagree
     * whenever a list has been de-specialised -- jaiListBox leaves elemKind
     * saying `int` on a list whose array is now Values -- and the copy below
     * is a memcpy at the source's width into an array sized at the
     * destination's. Deriving it from elemKind wrote sixteen bytes per element
     * into an eight-byte-per-element block. */
    out->elemKind = list->elemKind;
    out->stg = list->stg;
    jaiGCPushRoot(OBJ_VAL(out));
    jaiListReserve(out, (int)count);
    jaiGCPopRoot();

    if (count > 0) {
        size_t w = jaiListStoreWidth(list->stg);
        if (step == 1) {
            memcpy(out->items, (const char *)list->items + w * (size_t)start,
                   w * (size_t)count);
        } else {
            for (int64_t i = 0, idx = start; i < count; ++i, idx += step)
                jaiListSetRaw(out, (int)i, jaiListGet(list, (int)idx));
        }
        out->count = (int)count;
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
    /* Two lists of the same storage concatenate into a third of that storage
     * and copy as bytes; mixed or boxed operands go through the boxed array,
     * which is what `jaiListBox` is for. */
    uint8_t stg = a->stg == b->stg ? a->stg : LIST_STORE_BOXED;
    if (stg == LIST_STORE_BOXED) {
        /* Before `out` exists: boxing allocates, and a fresh list nothing has
         * rooted yet would not survive the collection that allocation can
         * trigger. */
        (void)jaiListBox(a);
        (void)jaiListBox(b);
    }
    ObjList *out = jaiListNew(0);
    if (stg != LIST_STORE_BOXED) {
        out->elemKind = a->elemKind;
        out->stg = stg;
    }
    jaiGCPushRoot(OBJ_VAL(out));
    jaiListReserve(out, (int)total);
    jaiGCPopRoot();

    size_t w = jaiListStoreWidth(stg);
    if (aCount != 0)
        memcpy(out->items, a->items, w * (size_t)aCount);

    if (bCount != 0)
        memcpy((char *)out->items + w * (size_t)aCount, b->items,
               w * (size_t)bCount);

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
        jaiListBox(out)[count++] = wantValues ? value : key;

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
        jaiListBox(out)[count++] = OBJ_VAL(tuple);
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
