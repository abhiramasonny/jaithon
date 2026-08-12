// table.c is the hash table and string intern table for Jaithon

#include "vm/table.h"

#include "vm/gc.h"
#include "vm/object/object.h"

const Value JAI_TOMBSTONE = {VAL_OBJ, {.obj = NULL}};

#define TABLE_LOAD_NUM 1
#define TABLE_LOAD_DEN 2

#define TABLE_MAX_CAPACITY ((int64_t)1 << 30)

#define ENTRY_EMPTY_ORDER     (-2)
#define ENTRY_TOMBSTONE_ORDER (-1)

JAI_INLINE JAI_UNUSED bool entryIsEmpty(const JaiEntry *e) {
    return e->order == ENTRY_EMPTY_ORDER;
}

JAI_INLINE JAI_UNUSED bool entryIsTombstone(const JaiEntry *e) {
    return e->order == ENTRY_TOMBSTONE_ORDER;
}

JAI_INLINE JAI_UNUSED bool entryIsLive(const JaiEntry *e) {
    return e->order >= 0;
}

JAI_INLINE JAI_UNUSED bool keyIsUsable(Value key) {
    return !(IS_NULL(key) || (IS_OBJ(key) && AS_OBJ(key) == NULL));
}

//capacity management

static void clearEntries(JaiEntry *entries, int capacity) {
    JaiEntry *e = entries;
    JaiEntry *const end = entries + capacity;

    for (; e != end; ++e) {
        e->key = NULL_VAL;
        e->order = ENTRY_EMPTY_ORDER;
    }
}

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
    ++t->keyVersion;

    JAI_FREE_ARRAY(JaiEntry, oldEntries, oldCapacity);
    JAI_FREE_ARRAY(int32_t, oldOrder, oldCapacity);
}

//rehashing stuff
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

JAI_INLINE bool keyMatches(const JaiEntry *e, Value key, uint64_t hash) {
    if (e->hash != hash) return false;

    const Value stored = e->key;
    if (jaiValueType(stored) == VAL_OBJ && jaiValueType(key) == VAL_OBJ &&
        AS_OBJ(stored) == AS_OBJ(key))
        return true;

    return keyEqualsOther(stored, key);
}

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

//mutation prims

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
    ++t->keyVersion;
}

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

    ensureRoom(t, key, value);

    return insertAt(t,
                    findEntry(t->entries, t->capacity, key, hash),
                    key, hash, value);
}

// API

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

void jaiTableReserve(JaiTable *t, int minCapacity) {
    if (minCapacity <= 0) return;
    int needed = capacityFor(minCapacity);
    if (needed > t->capacity) adjustCapacity(t, needed);
}

bool jaiTableGet(JaiTable *t, Value key, Value *out) {
    bool ok = true;

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
    return (int)(e - t->entries);
}

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

// String Intern Table

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

static inline uint64_t internFingerprint(const char *chars, size_t length) {
    uint64_t fp = (uint64_t)(length > 255 ? 255 : length) << 56;
    const size_t n = length < 7 ? length : 7;

    for (size_t i = 0; i < n; ++i)
        fp |= (uint64_t)(uint8_t)chars[i] << (i * 8);

    return fp;
}

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
    JAI_STR_INTERNED(s) = true;

    (void)jaiTableSetInterned(
        &internTable, s,
        INT_VAL((int64_t)internFingerprint(s->chars, s->length)));
}
