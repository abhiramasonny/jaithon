/* table.c — the one hash table (see table.h) and the string intern table.
 *
 * Open addressing, linear probing, power-of-two capacities, tombstones. The
 * table grows before an insert would push (live + tombstones) past 3/4 of
 * capacity. That bound is also what makes every probe loop terminate: at least
 * one never-used slot always remains, so a walk always hits an empty slot.
 *
 * Growth rehashes live entries only, so tombstones are the one kind of garbage
 * that a resize collects for free.
 */

#include "table.h"

#include "gc.h"
#include "object.h"

/* A tombstone is VAL_OBJ carrying a NULL pointer. No real Value can look like
 * that — OBJ_VAL is only ever built from an allocated Obj — and it is distinct
 * from NULL_VAL, which marks a slot that was never used. The distinction is
 * load-bearing: a probe stops at an empty slot but must walk past a tombstone,
 * or deleting a key would hide every key that collided with it. */
const Value JAI_TOMBSTONE = {VAL_OBJ, {.obj = NULL}};

/* Max load factor, counting tombstones: a tombstone still costs a probe step. */
#define TABLE_LOAD_NUM 3
#define TABLE_LOAD_DEN 4

/* Refuse to build a table so large that the index arithmetic stops being
 * meaningful; jaiRealloc would have run out of memory long before this. */
#define TABLE_MAX_CAPACITY ((int64_t)1 << 30)

/* ------------------------------------------------------------------ */
/* Slot predicates                                                     */
/* ------------------------------------------------------------------ */

JAI_INLINE bool entryIsEmpty(const JaiEntry *e) {
    return IS_NULL(e->key);
}

JAI_INLINE bool entryIsTombstone(const JaiEntry *e) {
    return IS_OBJ(e->key) && AS_OBJ(e->key) == NULL;
}

JAI_INLINE bool entryIsLive(const JaiEntry *e) {
    return !entryIsEmpty(e) && !entryIsTombstone(e);
}

/* Neither sentinel may be used as a key: NULL_VAL would read back as an empty
 * slot and a NULL Obj* as a deleted one. `null` is therefore not a usable dict
 * or set key; the caller must reject it before reaching this layer. */
JAI_INLINE JAI_UNUSED bool keyIsUsable(Value key) {
    return !(IS_NULL(key) || (IS_OBJ(key) && AS_OBJ(key) == NULL));
}

/* ------------------------------------------------------------------ */
/* Capacity management                                                 */
/* ------------------------------------------------------------------ */

static void clearEntries(JaiEntry *entries, int capacity) {
    for (int i = 0; i < capacity; i++) {
        entries[i].key = NULL_VAL;
        entries[i].value = NULL_VAL;
        entries[i].hash = 0;
        entries[i].order = -1;
    }
}

/* Squeeze the holes left by deletions out of the order array. Called only when
 * it is full, which needs `capacity` inserts since the last compaction, so the
 * O(capacity) walk is amortised to O(1) per insert. */
static void compactOrder(JaiTable *t) {
    int live = 0;
    for (int i = 0; i < t->orderCount; i++) {
        int32_t slot = t->order[i];
        if (slot < 0) continue;
        t->order[live] = slot;
        t->entries[slot].order = live;
        live++;
    }
    t->orderCount = live;
}

/* Smallest power-of-two capacity that holds `liveEntries` without resizing. */
static int capacityFor(int64_t liveEntries) {
    if (liveEntries > TABLE_MAX_CAPACITY) {
        JAI_PANIC("hash table too large: %lld entries requested",
                  (long long)liveEntries);
    }
    int64_t need = (liveEntries * TABLE_LOAD_DEN + TABLE_LOAD_NUM - 1) / TABLE_LOAD_NUM;
    int64_t cap = 8;
    while (cap < need) {
        cap *= 2;
        if (cap > TABLE_MAX_CAPACITY) {
            JAI_PANIC("hash table too large: %lld entries requested",
                      (long long)liveEntries);
        }
    }
    return (int)cap;
}

/* Destination slot during a rehash. The new array has no tombstones and the
 * keys being moved are already known to be distinct, so the first empty slot is
 * the answer — no key comparison, hence no chance of re-entering user code
 * while the table is half-migrated. */
static JaiEntry *findEmptySlot(JaiEntry *entries, int capacity, uint64_t hash) {
    uint32_t mask = (uint32_t)capacity - 1;
    uint32_t index = (uint32_t)(hash & (uint64_t)mask);
    while (!entryIsEmpty(&entries[index])) index = (index + 1) & mask;
    return &entries[index];
}

static void adjustCapacity(JaiTable *t, int capacity) {
    /* These allocations can run the collector. Until the swap below, `t` still
     * describes the old arrays, so marking it stays correct. */
    JaiEntry *entries = JAI_ALLOC(JaiEntry, capacity);
    clearEntries(entries, capacity);
    int32_t *order = JAI_ALLOC(int32_t, capacity);

    /* Migrating in insertion order rather than in slot order is what carries
     * the order across a rehash; it also drops the holes for free. */
    int count = 0;
    for (int i = 0; i < t->orderCount; i++) {
        int32_t slot = t->order[i];
        if (slot < 0) continue;
        JaiEntry *src = &t->entries[slot];
        if (!entryIsLive(src)) continue;   /* tombstones die here */
        JaiEntry *dest = findEmptySlot(entries, capacity, src->hash);
        dest->key = src->key;
        dest->value = src->value;
        dest->hash = src->hash;
        dest->order = count;
        order[count++] = (int32_t)(dest - entries);
    }

    JaiEntry *old = t->entries;
    int32_t *oldOrder = t->order;
    int oldCapacity = t->capacity;
    t->entries = entries;
    t->order = order;
    t->orderCount = count;
    t->capacity = capacity;
    t->count = count;
    t->tombstones = 0;
    t->version++;
    /* Freed only once the table points at the new arrays: a collection during
     * the free would otherwise walk memory we just released. */
    JAI_FREE_ARRAY(JaiEntry, old, oldCapacity);
    JAI_FREE_ARRAY(int32_t, oldOrder, oldCapacity);
}

/* Rehash if one more entry would exceed the load factor. `key` and `value` are
 * rooted across the resize because the resize allocates, and a caller that has
 * only just built them may have them nowhere else the collector can see.
 *
 * What filled the table may be tombstones rather than live entries — the weak
 * intern table replaces its whole contents on every collection. Doubling then
 * would grow the array without bound while the live count stayed flat, so the
 * capacity only grows when the live entries themselves need the room. Using
 * half the capacity as that trigger leaves at least a quarter of the slots
 * free after a rehash, so the O(capacity) walk is still amortised to O(1). */
static void ensureRoom(JaiTable *t, Value key, Value value) {
    int64_t used = (int64_t)t->count + t->tombstones + 1;
    if (used * TABLE_LOAD_DEN <= (int64_t)t->capacity * TABLE_LOAD_NUM) return;

    int capacity = t->capacity;
    if ((int64_t)(t->count + 1) * 2 > (int64_t)capacity) {
        capacity = JAI_GROW_CAP(capacity);
    }

    jaiGCPushRoot(key);
    jaiGCPushRoot(value);
    adjustCapacity(t, capacity);
    jaiGCPopRoots(2);
}

/* ------------------------------------------------------------------ */
/* Probing                                                             */
/* ------------------------------------------------------------------ */

static bool keyMatches(const JaiEntry *e, Value key, uint64_t hash) {
    if (e->hash != hash) return false;
    /* Identity first: it never calls user code, and it settles the interned
     * string and primitive cases that dominate. Only then the general path,
     * which may run __eq__. */
    if (jaiValuesIdentical(e->key, key)) return true;
    return jaiValuesEqual(e->key, key);
}

/* Returns the entry holding `key`, or the slot it should be inserted into —
 * preferring the first tombstone seen, so deletions do not permanently cost
 * space. Never returns NULL. */
static JaiEntry *findEntry(JaiEntry *entries, int capacity, Value key, uint64_t hash) {
    uint32_t mask = (uint32_t)capacity - 1;
    uint32_t index = (uint32_t)(hash & (uint64_t)mask);
    JaiEntry *tombstone = NULL;

    for (;;) {
        JaiEntry *e = &entries[index];
        if (entryIsEmpty(e)) return tombstone != NULL ? tombstone : e;
        if (entryIsTombstone(e)) {
            if (tombstone == NULL) tombstone = e;
        } else if (keyMatches(e, key, hash)) {
            return e;
        }
        index = (index + 1) & mask;
    }
}

/* Same, comparing by pointer. Valid only when every key that could equal `key`
 * is the same interned ObjString, i.e. on tables whose keys all come from the
 * intern table (globals, exports, method tables). */
static JaiEntry *findEntryInterned(JaiEntry *entries, int capacity, ObjString *key) {
    uint32_t mask = (uint32_t)capacity - 1;
    uint32_t index = (uint32_t)(key->hash & (uint64_t)mask);
    JaiEntry *tombstone = NULL;

    for (;;) {
        JaiEntry *e = &entries[index];
        if (entryIsEmpty(e)) return tombstone != NULL ? tombstone : e;
        if (entryIsTombstone(e)) {
            if (tombstone == NULL) tombstone = e;
        } else if (IS_OBJ(e->key) && AS_OBJ(e->key) == (Obj *)key) {
            return e;
        }
        index = (index + 1) & mask;
    }
}

/* ------------------------------------------------------------------ */
/* Mutation primitives                                                 */
/* ------------------------------------------------------------------ */

static bool insertAt(JaiTable *t, JaiEntry *e, Value key, uint64_t hash, Value value) {
    bool isNew = !entryIsLive(e);
    if (isNew) {
        if (entryIsTombstone(e)) t->tombstones--;
        t->count++;
        e->key = key;
        e->hash = hash;
        /* The load factor keeps count below capacity, so compacting always
         * frees at least one slot here. */
        if (t->orderCount >= t->capacity) compactOrder(t);
        e->order = t->orderCount;
        t->order[t->orderCount++] = (int32_t)(e - t->entries);
    }
    e->value = value;
    t->version++;
    return isNew;
}

static void removeEntry(JaiTable *t, JaiEntry *e) {
    e->key = JAI_TOMBSTONE;
    e->value = NULL_VAL;   /* drop the reference so the GC can reclaim it */
    e->hash = 0;
    if (e->order >= 0) t->order[e->order] = -1;
    e->order = -1;
    t->count--;
    t->tombstones++;
    t->version++;
}

/* Insert with an already-computed hash, comparing keys by value. Used by
 * jaiTableSet and by jaiTableAddAll, which must not re-hash (that could
 * re-enter a user __hash__). */
static bool tableSetHashed(JaiTable *t, Value key, uint64_t hash, Value value) {
    ensureRoom(t, key, value);
    return insertAt(t, findEntry(t->entries, t->capacity, key, hash), key, hash, value);
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
    /* Hash first: it can raise TypeError for an unhashable key, and it can run
     * user code that mutates `t`, so nothing may be cached across it. */
    uint64_t hash = jaiValueHashFast(key, &ok);
    if (!ok) return false;
    if (t->count == 0) return false;

    JaiEntry *e = findEntry(t->entries, t->capacity, key, hash);
    if (!entryIsLive(e)) return false;
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
    uint64_t hash = jaiValueHash(key, &ok);
    if (!ok) return false;
    if (t->count == 0) return false;

    JaiEntry *e = findEntry(t->entries, t->capacity, key, hash);
    if (!entryIsLive(e)) return false;
    removeEntry(t, e);
    return true;
}

void jaiTableClear(JaiTable *t) {
    if (t->entries != NULL) clearEntries(t->entries, t->capacity);
    t->orderCount = 0;
    t->count = 0;
    t->tombstones = 0;
    t->version++;
}

void jaiTableAddAll(const JaiTable *from, JaiTable *to) {
    if (from == to || from->entries == NULL || from->count == 0) return;

    /* Size the destination once so the copy loop does not allocate. */
    int needed = capacityFor((int64_t)to->count + from->count);
    if (needed > to->capacity) adjustCapacity(to, needed);

    /* In the source's insertion order, so a merged dict reads as the two
     * written end to end. */
    for (int i = 0; i < from->orderCount; i++) {
        int32_t slot = from->order[i];
        if (slot < 0) continue;
        const JaiEntry *e = &from->entries[slot];
        if (!entryIsLive(e)) continue;
        tableSetHashed(to, e->key, e->hash, e->value);
    }
}

bool jaiTableGetInterned(JaiTable *t, ObjString *key, Value *out) {
    JAI_ASSERT(key != NULL, "interned key must not be NULL");
    if (t->count == 0) return false;

    JaiEntry *e = findEntryInterned(t->entries, t->capacity, key);
    if (!entryIsLive(e)) return false;
    if (out != NULL) *out = e->value;
    return true;
}

bool jaiTableSetInterned(JaiTable *t, ObjString *key, Value value) {
    JAI_ASSERT(key != NULL, "interned key must not be NULL");
    Value k = OBJ_VAL(key);
    ensureRoom(t, k, value);
    JaiEntry *e = findEntryInterned(t->entries, t->capacity, key);
    return insertAt(t, e, k, key->hash, value);
}

int jaiTableFindIndex(JaiTable *t, Value key) {
    bool ok = true;
    uint64_t hash = jaiValueHash(key, &ok);
    if (!ok) return -1;
    if (t->count == 0) return -1;

    JaiEntry *e = findEntry(t->entries, t->capacity, key, hash);
    if (!entryIsLive(e)) return -1;
    /* Stable until the next mutation, which the caller detects via version. */
    return (int)(e - t->entries);
}

/* `*i` is a cursor into the order array, not a slot number: callers only ever
 * start it at 0 and hand it back unchanged. */
bool jaiTableNext(const JaiTable *t, int *i, Value *outKey, Value *outValue) {
    if (t->entries == NULL) return false;

    int index = *i < 0 ? 0 : *i;
    for (; index < t->orderCount; index++) {
        int32_t slot = t->order[index];
        if (slot < 0) continue;
        const JaiEntry *e = &t->entries[slot];
        if (!entryIsLive(e)) continue;
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
    for (int i = 0; i < t->capacity; i++) {
        JaiEntry *e = &t->entries[i];
        if (!entryIsLive(e)) continue;
        jaiGCMarkValue(e->key);
        jaiGCMarkValue(e->value);
    }
}

/* Called between marking and sweeping. An unmarked key is about to be freed,
 * so its entry has to go — this is exactly what makes the intern table a set of
 * weak references instead of a leak that pins every string ever created. */
void jaiTableRemoveWhite(JaiTable *t) {
    if (t->entries == NULL) return;
    for (int i = 0; i < t->capacity; i++) {
        JaiEntry *e = &t->entries[i];
        if (!entryIsLive(e)) continue;
        if (IS_OBJ(e->key) && !AS_OBJ(e->key)->isMarked) removeEntry(t, e);
    }

    /* The purge leaves one tombstone per collected string and nothing else
     * ever shrinks this table. Its arrays count towards the live bytes that
     * set the next GC threshold, so a string-churning loop would ratchet the
     * heap upward one collection at a time: a bigger table means a later
     * collection, which means more strings interned before the next purge,
     * which means a bigger table again. Rehash down to what survived, with
     * room for twice as many so the shrink cannot thrash against the growth
     * in ensureRoom. Safe here: adjustCapacity moves only marked entries and
     * allocates no objects, so it cannot re-enter the collector.
     *
     * Requiring three quarters of the slots to be dead before rehashing both
     * stops the shrink thrashing against the next round of inserts and keeps
     * `count * 2` well inside what capacityFor accepts. */
    if ((int64_t)t->count * 4 >= (int64_t)t->capacity) return;
    int wanted = capacityFor((int64_t)t->count * 2 + 1);
    if (wanted < t->capacity) adjustCapacity(t, wanted);
}

/* ------------------------------------------------------------------ */
/* String intern table                                                  */
/* ------------------------------------------------------------------ */

/* A set: keys are ObjString*, values are always NULL_VAL. There is exactly one
 * per process; the GC reaches it through jaiInternTable() and treats it weakly
 * via jaiTableRemoveWhite. Not file-static only so that jaiInternTableCount()
 * can be inlined — string construction consults the population on every short
 * string, and a cross-TU call for it cost dict_ops 6%. */
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

/* Probes by (hash, length, bytes) rather than by Value, because this runs
 * during string creation: the ObjString the caller is looking for may not exist
 * yet, so there is nothing for jaiValueHash to be called on. */
ObjString *jaiInternTableFind(const char *chars, size_t length, uint64_t hash) {
    JaiTable *t = &internTable;
    if (t->count == 0) return NULL;

    uint32_t mask = (uint32_t)t->capacity - 1;
    uint32_t index = (uint32_t)(hash & (uint64_t)mask);

    for (;;) {
        JaiEntry *e = &t->entries[index];
        if (entryIsEmpty(e)) return NULL;
        if (!entryIsTombstone(e) && e->hash == hash) {
            JAI_ASSERT(IS_STRING(e->key), "intern table holds only strings");
            ObjString *s = (ObjString *)AS_OBJ(e->key);
            if ((size_t)s->length == length &&
                (length == 0 || memcmp(s->chars, chars, length) == 0)) {
                return s;
            }
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
    (void)jaiTableSetInterned(&internTable, s, NULL_VAL);
}
