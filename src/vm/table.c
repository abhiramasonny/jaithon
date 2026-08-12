/* table.c — the one hash table (see table.h) and the string intern table. */

#include "table.h"

#include "gc.h"
#include "object.h"

/* Tombstone = VAL_OBJ{NULL}, distinct from NULL_VAL (never-used slot): a probe
 * must walk past a tombstone rather than stop, or a delete could hide keys
 * that collided with it. */
const Value JAI_TOMBSTONE = {VAL_OBJ, {.obj = NULL}};

/* Max load factor, counting tombstones. Half, not 3/4: measured on dict_ops
 * (30M lookups), probes drop from ~2.1 to ~1.3 and the benchmark is 7.5%
 * faster, at the cost of ~13% more peak RSS. NUM must stay DEN-1 so the
 * threshold `capacity - capacity/DEN` avoids a divide on the insert path. */
#define TABLE_LOAD_NUM 1
#define TABLE_LOAD_DEN 2

/* Beyond this the index arithmetic stops being meaningful; jaiRealloc would
 * OOM long before this anyway. */
#define TABLE_MAX_CAPACITY ((int64_t)1 << 30)

/* Probe state piggybacks on JaiEntry.order: -2 = never-used, -1 = tombstone,
 * >=0 = live insertion-order index. Keeps the hot probe path on a small
 * integer until a candidate survives the state/hash tests. */
#define ENTRY_EMPTY_ORDER     (-2)
#define ENTRY_TOMBSTONE_ORDER (-1)

/* ------------------------------------------------------------------ */
/* Slot predicates                                                     */
/* ------------------------------------------------------------------ */

JAI_INLINE JAI_UNUSED bool entryIsEmpty(const JaiEntry *e) {
    return e->order == ENTRY_EMPTY_ORDER;
}

JAI_INLINE JAI_UNUSED bool entryIsTombstone(const JaiEntry *e) {
    return e->order == ENTRY_TOMBSTONE_ORDER;
}

JAI_INLINE JAI_UNUSED bool entryIsLive(const JaiEntry *e) {
    return e->order >= 0;
}

/* NULL_VAL would read back as empty and a NULL Obj* as deleted, so `null` is
 * not a usable key; the caller must reject it before this layer. */
JAI_INLINE JAI_UNUSED bool keyIsUsable(Value key) {
    return !(IS_NULL(key) || (IS_OBJ(key) && AS_OBJ(key) == NULL));
}

/* ------------------------------------------------------------------ */
/* Capacity management                                                 */
/* ------------------------------------------------------------------ */

static void clearEntries(JaiEntry *entries, int capacity) {
    JaiEntry *e = entries;
    JaiEntry *const end = entries + capacity;

    for (; e != end; ++e) {
        e->key = NULL_VAL;
        e->order = ENTRY_EMPTY_ORDER;
    }
}

/* Called only when `order` is full, which needs `capacity` inserts since the
 * last compaction, so the O(capacity) walk amortises to O(1) per insert. */
static void compactOrder(JaiTable *t) {
    int32_t *const order = t->order;
    JaiEntry *const entries = t->entries;
    const int n = t->orderCount;
    int live = 0;

    for (int i = 0; i < n; ++i) {
        const int32_t slot = order[i];
        if (slot < 0) continue;

        order[live] = slot;
        entries[slot].order = live++;
    }

    t->orderCount = live;
}

/* Smallest power-of-two capacity that holds `liveEntries` without resizing. */
static int capacityFor(int64_t liveEntries) {
    if (liveEntries <= 6) return 8;

    const uint64_t need =
        ((uint64_t)liveEntries * TABLE_LOAD_DEN + TABLE_LOAD_NUM - 1) /
        TABLE_LOAD_NUM;

    if (liveEntries > TABLE_MAX_CAPACITY ||
        need > (uint64_t)TABLE_MAX_CAPACITY) {
        JAI_PANIC("hash table too large: %lld entries requested",
                  (long long)liveEntries);
    }

    uint32_t cap = (uint32_t)need - 1;
    cap |= cap >> 1;
    cap |= cap >> 2;
    cap |= cap >> 4;
    cap |= cap >> 8;
    cap |= cap >> 16;
    return (int)(cap + 1);
}

/* Destination slot during a rehash: the new array has no tombstones and the
 * keys are already known distinct, so no key comparison is needed -- hence no
 * chance of re-entering user code while the table is half-migrated. */
static inline JaiEntry *findEmptySlot(JaiEntry *entries, int capacity,
                                      uint64_t hash) {
    const uint32_t mask = (uint32_t)capacity - 1;
    uint32_t index = (uint32_t)hash & mask;

    while (entries[index].order != ENTRY_EMPTY_ORDER)
        index = (index + 1) & mask;

    return entries + index;
}

static void adjustCapacity(JaiTable *t, int capacity) {
    JaiEntry *entries = JAI_ALLOC(JaiEntry, capacity);
    int32_t *order = JAI_ALLOC(int32_t, capacity);

    JaiEntry *const oldEntries = t->entries;
    int32_t *const oldOrder = t->order;
    const int oldCapacity = t->capacity;
    const int oldOrderCount = t->orderCount;

    clearEntries(entries, capacity);

    int count = 0;
    for (int i = 0; i < oldOrderCount; ++i) {
        const int32_t slot = oldOrder[i];
        if (slot < 0) continue;

        const JaiEntry *const src = oldEntries + slot;
        JaiEntry *const dst = findEmptySlot(entries, capacity, src->hash);

        *dst = *src;
        dst->order = count;
        order[count++] = (int32_t)(dst - entries);
    }

    t->entries = entries;
    t->order = order;
    t->orderCount = count;
    t->capacity = capacity;
    t->count = count;
    t->tombstones = 0;
    ++t->version;
    ++t->keyVersion;   /* every live entry just changed address */

    JAI_FREE_ARRAY(JaiEntry, oldEntries, oldCapacity);
    JAI_FREE_ARRAY(int32_t, oldOrder, oldCapacity);
}
/* Rehash if one more entry would exceed the load factor. `key`/`value` are
 * GC-rooted since the resize allocates. Growth triggers on live-entry count,
 * not slot count, so a table full of tombstones (e.g. the intern table after
 * a GC pass) doesn't grow unbounded while live entries stay flat. */
static inline void ensureRoom(JaiTable *t, Value key, Value value) {
    const int capacity = t->capacity;
    const int count = t->count;

    if (count + t->tombstones + 1 <=
        capacity - capacity / TABLE_LOAD_DEN)
        return;

    const int newCapacity =
        count + 1 > (capacity >> 1) ? JAI_GROW_CAP(capacity) : capacity;

    jaiGCPushRoot(key);
    jaiGCPushRoot(value);
    adjustCapacity(t, newCapacity);
    jaiGCPopRoots(2);
}

/* ------------------------------------------------------------------ */
/* Probing                                                             */
/* ------------------------------------------------------------------ */
/* Deliberately kept out of line: merged into keyMatches, it was too big for
 * clang to inline into findExisting/findEntry, so every probe paid a call --
 * 6.6% of dict_ops by sample count, almost all of it the call itself. */
static JAI_NOINLINE bool keyEqualsOther(Value stored, Value key) {
    const ValueType type = jaiValueType(key);

    if (jaiValueType(stored) != type)
        return jaiValuesEqual(stored, key);

    switch (type) {
        case VAL_OBJ: {
            Obj *const a = AS_OBJ(stored);
            Obj *const b = AS_OBJ(key);

            if (a == b) return true;
            if (a->type == OBJ_STRING && b->type == OBJ_STRING)
                return jaiStringEquals((ObjString *)a, (ObjString *)b);

            return jaiValuesEqual(stored, key);
        }

        case VAL_INT:
            return AS_INT(stored) == AS_INT(key);

        case VAL_BOOL:
            return AS_BOOL(stored) == AS_BOOL(key);

        case VAL_NULL:
            return true;

        case VAL_FLOAT:
            return AS_FLOAT(stored) == AS_FLOAT(key) ||
                   jaiValuesIdentical(stored, key);
    }

    return jaiValuesEqual(stored, key);
}

/* Hash first (already in a register, rules out most slots), then the common
 * case -- same heap object, as an interned string key always is -- inline.
 * Anything else defers to keyEqualsOther. */
JAI_INLINE bool keyMatches(const JaiEntry *e, Value key, uint64_t hash) {
    if (e->hash != hash) return false;

    const Value stored = e->key;
    if (jaiValueType(stored) == VAL_OBJ && jaiValueType(key) == VAL_OBJ &&
        AS_OBJ(stored) == AS_OBJ(key))
        return true;

    return keyEqualsOther(stored, key);
}

/* Lookup-only: unlike findEntry this need not remember the first tombstone,
 * saving a dependency and branch on every get and every miss. */
static inline JaiEntry *findExisting(JaiEntry *entries, int capacity,
                                     Value key, uint64_t hash) {
    const uint32_t mask = (uint32_t)capacity - 1;
    uint32_t index = (uint32_t)hash & mask;

    for (;;) {
        JaiEntry *const e = entries + index;
        const int state = e->order;

        if (state == ENTRY_EMPTY_ORDER) return NULL;
        if (state >= 0 && keyMatches(e, key, hash)) return e;

        index = (index + 1) & mask;
    }
}

/* Returns the entry holding `key`, or the slot to insert into (preferring the
 * first tombstone seen, so deletions don't permanently cost space). */
static inline JaiEntry *findEntry(JaiEntry *entries, int capacity,
                                  Value key, uint64_t hash) {
    const uint32_t mask = (uint32_t)capacity - 1;
    uint32_t index = (uint32_t)hash & mask;
    JaiEntry *tombstone = NULL;

    for (;;) {
        JaiEntry *const e = entries + index;
        const int state = e->order;

        if (state < 0) {
            if (state == ENTRY_EMPTY_ORDER)
                return tombstone != NULL ? tombstone : e;
            if (tombstone == NULL) tombstone = e;
        } else if (keyMatches(e, key, hash)) {
            return e;
        }

        index = (index + 1) & mask;
    }
}

/* Same, comparing by pointer. Valid only when every key that could equal `key`
 * is the same interned ObjString (globals, exports, method tables). */
static inline JaiEntry *findEntryInterned(JaiEntry *entries, int capacity,
                                          ObjString *key) {
    const uint32_t mask = (uint32_t)capacity - 1;
    uint32_t index = (uint32_t)key->hash & mask;
    JaiEntry *tombstone = NULL;
    Obj *const needle = (Obj *)key;

    for (;;) {
        JaiEntry *const e = entries + index;
        const int state = e->order;

        if (state < 0) {
            if (state == ENTRY_EMPTY_ORDER)
                return tombstone != NULL ? tombstone : e;
            if (tombstone == NULL) tombstone = e;
        } else if (AS_OBJ(e->key) == needle) {
            return e;
        }

        index = (index + 1) & mask;
    }
}

static inline JaiEntry *findExistingInterned(JaiEntry *entries, int capacity,
                                             ObjString *key) {
    const uint32_t mask = (uint32_t)capacity - 1;
    uint32_t index = (uint32_t)key->hash & mask;
    Obj *const needle = (Obj *)key;

    for (;;) {
        JaiEntry *const e = entries + index;
        const int state = e->order;

        if (state == ENTRY_EMPTY_ORDER) return NULL;
        if (state >= 0 && AS_OBJ(e->key) == needle) return e;

        index = (index + 1) & mask;
    }
}

/* ------------------------------------------------------------------ */
/* Mutation primitives                                                 */
/* ------------------------------------------------------------------ */

static inline bool insertAt(JaiTable *t, JaiEntry *e, Value key,
                            uint64_t hash, Value value) {
    const int state = e->order;
    const bool isNew = state < 0;

    if (isNew) {
        if (state == ENTRY_TOMBSTONE_ORDER) --t->tombstones;
        ++t->count;

        if (t->orderCount >= t->capacity) compactOrder(t);

        const int orderIndex = t->orderCount++;
        e->key = key;
        e->hash = hash;
        e->order = orderIndex;
        t->order[orderIndex] = (int32_t)(e - t->entries);
        /* Key set changed; see JaiTable::keyVersion. */
        ++t->keyVersion;
    }

    e->value = value;
    ++t->version;
    return isNew;
}

static inline void removeEntry(JaiTable *t, JaiEntry *e) {
    const int orderIndex = e->order;
    JAI_ASSERT(orderIndex >= 0, "removing a non-live table entry");

    t->order[orderIndex] = -1;
    e->order = ENTRY_TOMBSTONE_ORDER;
    e->key = JAI_TOMBSTONE;
    e->value = NULL_VAL;
    e->hash = 0;

    --t->count;
    ++t->tombstones;
    ++t->version;
    ++t->keyVersion;   /* this address no longer holds this key */
}

/* Insert with an already-computed hash. Used by jaiTableAddAll too, which
 * must not re-hash (that could re-enter a user __hash__). */
static bool tableSetHashed(JaiTable *t, Value key, uint64_t hash, Value value) {
    const int capacity = t->capacity;

    if (t->count + t->tombstones + 1 <=
        capacity - capacity / TABLE_LOAD_DEN) {
        return insertAt(t, findEntry(t->entries, capacity, key, hash),
                        key, hash, value);
    }

    if (t->count != 0) {
        JaiEntry *const existing =
            findExisting(t->entries, t->capacity, key, hash);
        if (existing != NULL)
            return insertAt(t, existing, key, hash, value);
    }

    /* Equality can run user code, so the resize check must be recomputed. */
    ensureRoom(t, key, value);

    return insertAt(t,
                    findEntry(t->entries, t->capacity, key, hash),
                    key, hash, value);
}

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */

void jaiTableInit(JaiTable *t) {
    t->entries = NULL;
    t->order = NULL;
    t->orderCount = 0;
    t->count = 0;
    t->tombstones = 0;
    t->capacity = 0;
    t->version = 0;
    t->keyVersion = 0;
}

void jaiTableFree(JaiTable *t) {
    JAI_FREE_ARRAY(JaiEntry, t->entries, t->capacity);
    JAI_FREE_ARRAY(int32_t, t->order, t->capacity);
    jaiTableInit(t);
}

/* `minCapacity` is read as "this many live entries without a resize". */
void jaiTableReserve(JaiTable *t, int minCapacity) {
    if (minCapacity <= 0) return;
    int needed = capacityFor(minCapacity);
    if (needed > t->capacity) adjustCapacity(t, needed);
}

bool jaiTableGet(JaiTable *t, Value key, Value *out) {
    bool ok = true;
    /* Hash first: it can raise on an unhashable key and can run user code
     * that mutates `t`, so nothing may be cached across it. */
    uint64_t hash = jaiValueHashFast(key, &ok);
    if (!ok) return false;
    if (t->count == 0) return false;

    JaiEntry *e = findExisting(t->entries, t->capacity, key, hash);
    if (e == NULL) return false;
    if (out != NULL) *out = e->value;
    return true;
}

bool jaiTableSet(JaiTable *t, Value key, Value value) {
    JAI_ASSERT(keyIsUsable(key), "null is not a usable table key");
    bool ok = true;
    uint64_t hash = jaiValueHashFast(key, &ok);
    if (!ok) return false;
    return tableSetHashed(t, key, hash, value);
}

bool jaiTableSetHashed(JaiTable *t, Value key, uint64_t hash, Value value) {
    JAI_ASSERT(keyIsUsable(key), "null is not a usable table key");
    return tableSetHashed(t, key, hash, value);
}

bool jaiTableDelete(JaiTable *t, Value key) {
    bool ok = true;
    uint64_t hash = jaiValueHashFast(key, &ok);
    if (!ok) return false;
    if (t->count == 0) return false;

    JaiEntry *e = findExisting(t->entries, t->capacity, key, hash);
    if (e == NULL) return false;
    removeEntry(t, e);
    return true;
}

void jaiTableClear(JaiTable *t) {
    if (t->entries != NULL) clearEntries(t->entries, t->capacity);
    t->orderCount = 0;
    t->count = 0;
    t->tombstones = 0;
    t->version++;
    t->keyVersion++;
}

void jaiTableAddAll(const JaiTable *from, JaiTable *to) {
    if (from == to || from->entries == NULL || from->count == 0) return;

    const int needed = capacityFor((int64_t)to->count + from->count);
    if (needed > to->capacity) {
        adjustCapacity(to, needed);
    } else if ((int64_t)to->count + from->count + to->tombstones >
               to->capacity - to->capacity / TABLE_LOAD_DEN) {
        adjustCapacity(to, to->capacity);
    }

    const int32_t *const order = from->order;
    const JaiEntry *const entries = from->entries;
    const int n = from->orderCount;

    for (int i = 0; i < n; ++i) {
        const int32_t slot = order[i];
        if (slot < 0) continue;

        const JaiEntry *const e = entries + slot;
        insertAt(to,
                 findEntry(to->entries, to->capacity, e->key, e->hash),
                 e->key, e->hash, e->value);
    }
}

bool jaiTableGetInterned(JaiTable *t, ObjString *key, Value *out) {
    JAI_ASSERT(key != NULL, "interned key must not be NULL");
    if (t->count == 0) return false;

    JaiEntry *e = findExistingInterned(t->entries, t->capacity, key);
    if (e == NULL) return false;
    if (out != NULL) *out = e->value;
    return true;
}

/* insertAt, reporting what the slot held first (`e->order < 0` reads back
 * both an empty slot and a tombstone as absent). */
static inline bool insertAtPrev(JaiTable *t, JaiEntry *e, Value key,
                                uint64_t hash, Value value, Value *outPrev) {
    if (outPrev != NULL) *outPrev = e->order < 0 ? NULL_VAL : e->value;
    return insertAt(t, e, key, hash, value);
}

bool jaiTableSetInternedPrev(JaiTable *t, ObjString *key, Value value,
                             Value *outPrev) {
    JAI_ASSERT(key != NULL, "interned key must not be NULL");

    const Value k = OBJ_VAL(key);
    const uint64_t hash = key->hash;
    const int capacity = t->capacity;

    if (t->count + t->tombstones + 1 <=
        capacity - capacity / TABLE_LOAD_DEN) {
        return insertAtPrev(t,
                            findEntryInterned(t->entries, capacity, key),
                            k, hash, value, outPrev);
    }

    if (t->count != 0) {
        JaiEntry *const existing =
            findExistingInterned(t->entries, capacity, key);
        if (existing != NULL)
            return insertAtPrev(t, existing, k, hash, value, outPrev);
    }

    ensureRoom(t, k, value);
    return insertAtPrev(t,
                        findEntryInterned(t->entries, t->capacity, key),
                        k, hash, value, outPrev);
}

bool jaiTableSetInterned(JaiTable *t, ObjString *key, Value value) {
    return jaiTableSetInternedPrev(t, key, value, NULL);
}

JaiEntry *jaiTableFindEntryInterned(JaiTable *t, ObjString *key) {
    if (t->count == 0) return NULL;
    return findExistingInterned(t->entries, t->capacity, key);
}

int jaiTableFindIndex(JaiTable *t, Value key) {
    bool ok = true;
    uint64_t hash = jaiValueHashFast(key, &ok);
    if (!ok) return -1;
    if (t->count == 0) return -1;

    JaiEntry *e = findExisting(t->entries, t->capacity, key, hash);
    if (e == NULL) return -1;
    /* Stable until the next mutation, which the caller detects via version. */
    return (int)(e - t->entries);
}

/* `*i` is a cursor into the order array, not a slot number: callers only ever
 * start it at 0 and hand it back unchanged. */
bool jaiTableNext(const JaiTable *t, int *i, Value *outKey, Value *outValue) {
    if (t->entries == NULL) return false;

    if (t->count == 0) {
        *i = t->orderCount;
        return false;
    }

    int index = *i < 0 ? 0 : *i;
    for (; index < t->orderCount; ++index) {
        const int32_t slot = t->order[index];
        if (slot < 0) continue;

        const JaiEntry *const e = t->entries + slot;
        if (outKey != NULL) *outKey = e->key;
        if (outValue != NULL) *outValue = e->value;
        *i = index + 1;
        return true;
    }

    *i = index;
    return false;
}

void jaiTableMark(JaiTable *t) {
    if (t->entries == NULL) return;

    const int32_t *const order = t->order;
    JaiEntry *const entries = t->entries;
    const int n = t->orderCount;

    for (int i = 0; i < n; ++i) {
        const int32_t slot = order[i];
        if (slot < 0) continue;

        JaiEntry *const e = entries + slot;
        jaiGCMarkVal(e->key);
        jaiGCMarkVal(e->value);
    }
}

/* Called between marking and sweeping: an unmarked key is about to be freed,
 * so its entry goes too -- this is what makes the intern table weak. */
void jaiTableRemoveWhite(JaiTable *t) {
    if (t->entries == NULL) return;

    const int n = t->orderCount;
    for (int i = 0; i < n; ++i) {
        const int32_t slot = t->order[i];
        if (slot < 0) continue;

        JaiEntry *const e = t->entries + slot;
        if (IS_OBJ(e->key) && !AS_OBJ(e->key)->isMarked)
            removeEntry(t, e);
    }

    if (t->count >= (t->capacity >> 2)) return;

    const int wanted = capacityFor((int64_t)t->count * 2 + 1);
    if (wanted < t->capacity) adjustCapacity(t, wanted);
}

/* ------------------------------------------------------------------ */
/* String intern table                                                  */
/* ------------------------------------------------------------------ */

/* A set: keys are ObjString*, values are always NULL_VAL. Not file-static so
 * jaiInternTableCount() can be inlined -- a cross-TU call cost dict_ops 6%. */
JaiTable jaiInternTableStorage;
#define internTable jaiInternTableStorage

void jaiInternTableInit(void) {
    jaiTableInit(&internTable);
}

void jaiInternTableFree(void) {
    jaiTableFree(&internTable);
}

JaiTable *jaiInternTable(void) {
    return &internTable;
}

/* Packs a short string's identity into the value slot an intern-set entry
 * otherwise wastes: length in the top byte, first 7 content bytes below. For
 * length <= 7 this is exact, not a hash, so it settles equality without the
 * chars pointer-chase, which was 4.2% of dict_ops by itself. */
static inline uint64_t internFingerprint(const char *chars, size_t length) {
    uint64_t fp = (uint64_t)(length > 255 ? 255 : length) << 56;
    const size_t n = length < 7 ? length : 7;

    for (size_t i = 0; i < n; ++i)
        fp |= (uint64_t)(uint8_t)chars[i] << (i * 8);

    return fp;
}

/* Probes by (hash, length, bytes) rather than by Value: this runs during
 * string creation, before the ObjString being looked for exists. */
ObjString *jaiInternTableFind(const char *chars, size_t length, uint64_t hash) {
    JaiTable *const t = &internTable;
    if (t->count == 0) return NULL;

    const uint32_t mask = (uint32_t)t->capacity - 1;
    uint32_t index = (uint32_t)hash & mask;
    JaiEntry *const entries = t->entries;
    const uint64_t fp = internFingerprint(chars, length);

    for (;;) {
        JaiEntry *const e = entries + index;
        const int state = e->order;

        if (state == ENTRY_EMPTY_ORDER) return NULL;

        if (state >= 0 && e->hash == hash &&
            (uint64_t)AS_INT(e->value) == fp) {
            JAI_ASSERT(IS_STRING(e->key), "intern table holds only strings");
            ObjString *const s = (ObjString *)AS_OBJ(e->key);

            /* Seven bytes or fewer: the fingerprint carried all of them and
             * the length, so there is nothing left to compare. */
            if (length <= 7) return s;

            if ((size_t)s->length == length &&
                memcmp(s->chars, chars, length) == 0)
                return s;
        }

        index = (index + 1) & mask;
    }
}

void jaiInternTableAdd(ObjString *s) {
    JAI_ASSERT(s != NULL, "cannot intern NULL");
    JAI_ASSERT(s->hash != 0 || s->length == 0,
               "an interned string must carry its hash");
    /* Set the flag here so that "in the intern table" and "s->interned" cannot
     * disagree; the insert below may collect, and the collector reads it. */
    JAI_STR_INTERNED(s) = true;
    /* The value is the fingerprint jaiInternTableFind probes by. This is the
     * only writer, so nothing else has to know the encoding. */
    (void)jaiTableSetInterned(
        &internTable, s,
        INT_VAL((int64_t)internFingerprint(s->chars, s->length)));
}
