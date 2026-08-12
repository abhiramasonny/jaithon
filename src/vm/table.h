/* table.h is the hash table used in Jaithon; keys are Values, which are
 * hashed through jaiValueHash. jaiTableGetInterned skips hashing for
 * ObjString* keys by using the string's cached hash and comparing by pointer. */

#ifndef JAI_TABLE_H
#define JAI_TABLE_H

#include "vm/value.h"

typedef struct {
    Value    key;
    Value    value;
    uint64_t hash;
    int32_t  order;
} JaiEntry;

typedef struct {
    JaiEntry *entries;
    int32_t  *order;
    int       orderCount;
    int       count;
    int       tombstones;
    int       capacity; //pow of 2
    uint32_t  version;
    uint32_t  keyVersion;
} JaiTable;

extern const Value JAI_TOMBSTONE;

void jaiTableInit(JaiTable *t);
void jaiTableFree(JaiTable *t);
void jaiTableReserve(JaiTable *t, int minCapacity);
bool jaiTableGet(JaiTable *t, Value key, Value *out);
bool jaiTableSet(JaiTable *t, Value key, Value value);
bool jaiTableSetHashed(JaiTable *t, Value key, uint64_t hash, Value value);
bool jaiTableDelete(JaiTable *t, Value key);
void jaiTableClear(JaiTable *t);
void jaiTableAddAll(const JaiTable *from, JaiTable *to);

bool jaiTableGetInterned(JaiTable *t, ObjString *key, Value *out);
bool jaiTableSetInterned(JaiTable *t, ObjString *key, Value value);
bool jaiTableSetInternedPrev(JaiTable *t, ObjString *key, Value value,
                             Value *outPrev);
int  jaiTableFindIndex(JaiTable *t, Value key);

JaiEntry *jaiTableFindEntryInterned(JaiTable *t, ObjString *key);

bool jaiTableNext(const JaiTable *t, int *i, Value *outKey, Value *outValue);

void jaiTableMark(JaiTable *t);
void jaiTableRemoveWhite(JaiTable *t);

//String intern table

ObjString *jaiInternTableFind(const char *chars, size_t length, uint64_t hash);
void       jaiInternTableAdd(ObjString *s);
void       jaiInternTableInit(void);
void       jaiInternTableFree(void);
JaiTable  *jaiInternTable(void);

extern JaiTable jaiInternTableStorage;
static inline int jaiInternTableCount(void) { return jaiInternTableStorage.count; }

#endif /* JAI_TABLE_H */
