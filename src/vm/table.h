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

/* jaiTableFindStr's third answer, besides an entry and NULL (absent). */
#define JAI_TABLE_SLOW ((JaiEntry *)(uintptr_t)1)
JaiEntry *jaiTableFindStr(JaiTable *t, ObjString *key);

bool jaiTableNext(const JaiTable *t, int *i, Value *outKey, Value *outValue);

void jaiTableMark(JaiTable *t);
void jaiTableRemoveWhite(JaiTable *t);

//String intern table

ObjString *jaiInternTableFind(const char *chars, size_t length, uint64_t hash);

/* What an intern entry keeps in its value slot: the length in the top byte
 * and the first seven bytes below it. A probe compares this before it touches
 * the string at all, and for a string of seven bytes or fewer it is the whole
 * string, so a hit costs no load from the string object. */
static inline uint64_t jaiInternFingerprint(const char *chars, size_t length) {
    uint64_t fp = (uint64_t)(length > 255 ? 255 : length) << 56;
    const size_t n = length < 7 ? length : 7;

    for (size_t i = 0; i < n; ++i)
        fp |= (uint64_t)(uint8_t)chars[i] << (i * 8);

    return fp;
}

/* The same fingerprint from one load, for a caller that guarantees eight
 * readable, initialised bytes at `chars` whatever `length` is, and a length
 * of at most 255. Little-endian, as the byte loop above is by construction. */
static inline uint64_t jaiInternFingerprintPadded(const char *chars,
                                                  size_t length) {
    uint64_t w;
    memcpy(&w, chars, sizeof w);
    const size_t n = length < 7 ? length : 7;
    return ((uint64_t)length << 56) | (w & ((UINT64_C(1) << (n * 8)) - 1));
}

void       jaiInternTableAdd(ObjString *s);
void       jaiInternTableInit(void);
void       jaiInternTableFree(void);
JaiTable  *jaiInternTable(void);

extern JaiTable jaiInternTableStorage;

/* The run-time interning policy's two numbers; object_string.c explains them. */
#define JAI_INTERN_MAX      32        /* longest run-time string worth a probe */
#define JAI_INTERN_SOFT_CAP (1 << 15) /* entries, past which run-time strings stop */
static inline int jaiInternTableCount(void) { return jaiInternTableStorage.count; }

#endif /* JAI_TABLE_H */
