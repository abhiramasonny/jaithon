/* table_inline.h -- the two string probes the hot paths want inlined.
 *
 * The callers that run once per iteration of a hot loop -- the compiled tier's
 * dict leaves (jit_runtime.c) and the f-string formatter's intern probe
 * (value.c) -- include this and get the loop in place of a call; table.c's
 * jaiInternTableFind is the same intern probe. A call each way was a sixth of
 * those leaves: the call, the argument moves and the frame the caller then
 * needs to keep its own values across it. Neither probe here calls anything,
 * which is the point -- a caller whose only calls are tail calls keeps no
 * frame at all.
 *
 * Its own header, not table.h, because both probes read ObjString fields and
 * object.h includes table.h for JaiTable. */
#ifndef JAI_TABLE_INLINE_H
#define JAI_TABLE_INLINE_H

#include "vm/object/object.h"
#include "vm/table.h"

/* JaiEntry::order of a slot that has never held anything. table.c's own
 * spelling is ENTRY_EMPTY_ORDER; the two are the same number. */
#define JAI_ENTRY_EMPTY_ORDER (-2)

/* jaiTableFindStr's probe with every call taken out, for the compiled tier's
 * dict leaves: it answers only what it can answer from the slots and the key's
 * cached hash -- the entry holding `key` itself, or NULL for absent -- and
 * JAI_TABLE_SLOW for everything else: a key whose lazy hash is not computed
 * yet, and a hash-equal slot holding some other object, which needs the byte
 * compare (or, for a key that is not a string, a user `__eq__`). The caller
 * hands JAI_TABLE_SLOW to jaiTableFindStr in a function of its own, so the
 * leaf that inlines this needs no frame at all: with the full compare inlined,
 * every caller saved and restored six register pairs on every probe for a
 * compare almost none of them reach. */
JAI_INLINE JaiEntry *jaiTableFindStrQuick(JaiTable *t, ObjString *key) {
    if (t->count == 0) return NULL;

    const uint64_t hash = key->hash;
    if (JAI_UNLIKELY(hash == 0)) return JAI_TABLE_SLOW;
    const uint32_t mask = (uint32_t)t->capacity - 1;
    uint32_t index = (uint32_t)hash & mask;
    JaiEntry *const entries = t->entries;

    for (;;) {
        JaiEntry *const e = entries + index;
        const int state = e->order;

        if (state == JAI_ENTRY_EMPTY_ORDER) return NULL;
        if (state >= 0 && e->hash == hash) {
            const Value stored = e->key;
            if (jaiValueType(stored) == VAL_OBJ && AS_OBJ(stored) == (Obj *)key)
                return e;
            return JAI_TABLE_SLOW;
        }

        index = (index + 1) & mask;
    }
}

/* Equal bytes, without a call: the run-time strings interning sees are at
 * most JAI_INTERN_MAX bytes, and a memcmp anywhere in the probe loop made the
 * probe save six register pairs on every call, hit or miss, for a compare most
 * probes never reach. */
JAI_INLINE bool jaiInternBytesEqual(const char *a, const char *b, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t x, y;
        memcpy(&x, a + i, 8);
        memcpy(&y, b + i, 8);
        if (x != y) return false;
    }
    for (; i < n; ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

/* The intern table's probe for a string of at most JAI_INTERN_MAX bytes,
 * fingerprint in hand (jaiInternFingerprint). `viaMemcmp` is a constant at
 * every call site, so each gets its own copy; table.c's long-string probe is
 * the one that passes true. */
JAI_INLINE ObjString *jaiInternProbeInline(const char *chars, size_t length,
                                           uint64_t hash, uint64_t fp,
                                           bool viaMemcmp) {
    JaiTable *const t = &jaiInternTableStorage;
    if (t->count == 0) return NULL;

    const uint32_t mask = (uint32_t)t->capacity - 1;
    uint32_t index = (uint32_t)hash & mask;
    JaiEntry *const entries = t->entries;

    for (;;) {
        JaiEntry *const e = entries + index;
        const int state = e->order;

        if (state == JAI_ENTRY_EMPTY_ORDER) return NULL;

        if (state >= 0 && e->hash == hash &&
            (uint64_t)AS_INT(e->value) == fp) {
            JAI_ASSERT(IS_STRING(e->key), "intern table holds only strings");
            ObjString *const s = (ObjString *)AS_OBJ(e->key);

            /* The fingerprint holds the length and every byte of a string
             * this short, and the hash agrees: nothing left to compare. */
            if (length <= 7) return s;

            if ((size_t)s->length == length &&
                (viaMemcmp ? memcmp(s->chars, chars, length) == 0
                           : jaiInternBytesEqual(s->chars, chars, length)))
                return s;
        }

        index = (index + 1) & mask;
    }
}

#endif /* JAI_TABLE_INLINE_H */
