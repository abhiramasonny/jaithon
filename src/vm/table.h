/* table.h — the one hash table used everywhere in Jaithon.
 *
 * Open addressing, linear probing, tombstones, 75% max load. Keys are Values;
 * hashing goes through jaiValueHash so dicts, sets, method tables, and global
 * tables all share one implementation.
 *
 * For the common case of ObjString* keys, jaiTableGetInterned skips hashing by
 * using the string's cached hash and compares by pointer.
 */
#ifndef JAI_TABLE_H
#define JAI_TABLE_H

#include "value.h"

typedef struct {
    Value    key;        /* NULL_VAL = empty slot; TOMBSTONE = deleted */
    Value    value;
    uint64_t hash;
    int32_t  order;      /* this entry's position in JaiTable.order, or -1 */
} JaiEntry;

typedef struct {
    JaiEntry *entries;
    /* Slot indices in insertion order, with -1 for an entry since deleted.
     * Iteration walks this rather than the slot array, because a dict keeps
     * insertion order (std.json documents it, and `{1: 1, 2: 4}` has to print
     * the way it was written). Holes are compacted on the next rehash. */
    int32_t  *order;
    int       orderCount; /* used slots of `order`, holes included */
    int       count;      /* live entries */
    int       tombstones;
    int       capacity;   /* always a power of two, or 0 */
    uint32_t  version;    /* bumped on every mutation; iterators snapshot it */
} JaiTable;

/* A dedicated sentinel distinguishing "deleted" from "never used". */
extern const Value JAI_TOMBSTONE;

void jaiTableInit(JaiTable *t);
void jaiTableFree(JaiTable *t);
void jaiTableReserve(JaiTable *t, int minCapacity);

/* Returns false if absent. */
bool jaiTableGet(JaiTable *t, Value key, Value *out);
/* Returns true if the key was newly inserted. */
bool jaiTableSet(JaiTable *t, Value key, Value value);
/* Same, with a hash the caller has already computed for `key`. A caller that
 * has to inspect the key first (jaiDictSet, which rejects the ones spec §5.4
 * forbids) uses this so a user `__hash__` runs once rather than twice. */
bool jaiTableSetHashed(JaiTable *t, Value key, uint64_t hash, Value value);
/* Returns true if a live entry was removed. */
bool jaiTableDelete(JaiTable *t, Value key);
void jaiTableClear(JaiTable *t);
void jaiTableAddAll(const JaiTable *from, JaiTable *to);

/* Fast path for interned-string keys: pointer comparison, cached hash. */
bool jaiTableGetInterned(JaiTable *t, ObjString *key, Value *out);
bool jaiTableSetInterned(JaiTable *t, ObjString *key, Value value);
/* Index of the live entry for `key`, or -1. Stable until the next mutation;
 * used by the inline caches for globals. */
int  jaiTableFindIndex(JaiTable *t, Value key);

/* Iteration: start with i = 0, repeat while it returns true.
 *   int i = 0; Value k, v;
 *   while (jaiTableNext(t, &i, &k, &v)) { ... }
 */
bool jaiTableNext(const JaiTable *t, int *i, Value *outKey, Value *outValue);

/* GC hooks. */
void jaiTableMark(JaiTable *t);
/* Remove entries whose ObjString key was not marked; used for the intern table. */
void jaiTableRemoveWhite(JaiTable *t);

/* ------------------------------------------------------------------ */
/* String intern table                                                  */
/* ------------------------------------------------------------------ */

/* Returns the interned string equal to [chars, chars+len), or NULL. */
ObjString *jaiInternTableFind(const char *chars, size_t length, uint64_t hash);
void       jaiInternTableAdd(ObjString *s);
void       jaiInternTableInit(void);
void       jaiInternTableFree(void);
JaiTable  *jaiInternTable(void);
/* Population, for callers deciding whether interning is still worth it. Owned
 * by table.c; read-only here, and inline because it is on the string-creation
 * path. */
extern JaiTable jaiInternTableStorage;
static inline int jaiInternTableCount(void) { return jaiInternTableStorage.count; }

#endif /* JAI_TABLE_H */
