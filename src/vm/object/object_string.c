/* object_string.c — ObjString: allocation, interning, concatenation and
 * Python-semantics slicing, plus the ObjStrBuf append buffer that backs a
 * growing concatenation chain.
 *
 * This is the largest single kind in the object model (see object.c for the
 * split's overview) because interning is a policy, not just a constructor:
 * every entry point that can produce a string -- literal, concat, f-string,
 * slice -- has to agree on when a result takes part in the intern table, and
 * that policy is easiest to get right, and to keep right, read as one file.
 *
 * The one piece of shared machinery: jaiAsciiChars, the 128-entry one-byte
 * cache, is read directly (not through a real function call) by the string
 * iterator's per-character fast path in object_iter.c, via the `static
 * inline` jaiAsciiCharTable() accessor declared alongside it in object.h.
 * Nothing else outside this file touches it.
 *
 * Nothing here interprets bytecode; the only call back into the VM is
 * jaiThrow for runtime errors.
 */

#include "vm/object/object.h"
#include "vm/object/object_internal.h"   /* sliceCount, shared with object_collection.c */

#include "vm/gc.h"
#include "vm/table.h"
#include "vm/vm.h"

/* Interning policy. Names — everything the compiler and the deserialiser
 * produce — are interned unconditionally, because pointer equality standing in
 * for string equality is what makes member and global lookup a pointer compare
 * (jaiTableGetInterned). Run-time strings are a different trade: interning
 * them pays off only when they repeat (tests/bench/dict_ops rebuilds the same
 * 10,000 keys 50 times each and wants the existing object back), and costs
 * when they do not (tests/bench/string_build makes 2,000,000 distinct ones;
 * growing the table for them and sweeping each as a weak reference on every
 * collection was 45% of its run). So a run-time string takes part only while
 * doing so still pays, and the signal is table population: a program whose
 * short strings repeat reaches a steady state (dict_ops settles around 20,000
 * entries and reuses them for the rest of its run) while one producing
 * distinct strings climbs without bound. Past the cap run-time strings stop
 * taking part altogether — no probe, no insert — so the table neither grows
 * nor costs a cache miss per string, and what is already in it goes on serving
 * the program that put it there. 1<<15 measured best; 1<<16 gave back 2.4%.
 *
 * A hit-rate window was tried instead and is worse, because the choice is
 * self-reinforcing: whichever mode is live decides which object a container
 * ends up holding, so switching modes orphans everything the other mode
 * interned and the table collapses. Measured, it flapped every few thousand
 * probes and cost dict_ops 24%.
 *
 * The cap has to clear the names as well, since they share the table; the
 * stdlib interns about 2,500. jaiStringCanonical is the way in for the few
 * reflective entry points that need identity for a string the caller built. */
#define JAI_INTERN_MAX      32        /* longest run-time string worth a probe */
#define JAI_INTERN_SOFT_CAP (1 << 15) /* entries, past which run-time strings stop */

/* ------------------------------------------------------------------ */
/* Strings                                                              */
/* ------------------------------------------------------------------ */

/* Raw ObjString of `length` payload bytes. The caller fills chars[] and sets
 * hash and interned. */
/* Raw allocation: every ObjString field is assigned here, and the characters
 * are written by the caller immediately after. */
static ObjString *allocString(size_t length) {
    ObjString *s = (ObjString *)jaiAllocateObjectRaw(JAI_STRING_ALLOC(length),
                                                     OBJ_STRING);
    s->chars = (char *)(s + 1);
    s->owner = NULL;
    JAI_STR_UNTERMINATED(s) = false;
    s->length = (uint32_t)length;
    s->scalars = UINT32_MAX;       /* not yet computed */
    s->cursorScalar = 0;           /* zero/zero is always a valid memo */
    s->cursorByte = 0;
    s->hash = 0;
    JAI_STR_INTERNED(s) = false;
    s->chars[length] = '\0';
    return s;
}

/* Adds `s` to the intern table. The table may grow, so `s` is rooted across
 * the insertion. */
static inline void internString(ObjString *s) {
    JAI_STR_INTERNED(s) = true;
    jaiGCPushRoot(OBJ_VAL(s));
    jaiInternTableAdd(s);
    jaiGCPopRoot();
}

ObjString *jaiStringIntern(const char *chars, size_t length) {
    if (length > UINT32_MAX) {
        jaiThrow(vm.cValueError,
                 "string of %zu bytes exceeds the maximum length", length);
        return NULL;
    }

    const uint64_t hash = jaiHashBytes(chars, length);
    ObjString *found = jaiInternTableFind(chars, length, hash);
    if (found != NULL) return found;

    ObjString *s = allocString(length);
    if (length != 0) memcpy(s->chars, chars, length);
    s->hash = hash;
    internString(s);
    return s;
}

ObjString *jaiStringInternC(const char *cstr) {
    return jaiStringIntern(cstr, cstr == NULL ? 0 : strlen(cstr));
}

ObjString *jaiStringCanonical(ObjString *s) {
    if (s == NULL || JAI_STR_INTERNED(s)) return s;
    /* Force the hash: an interned string's is what the table probes by, and a
     * run-time string may not have needed one yet. */
    uint64_t hash = jaiStringHash(s);
    ObjString *found = jaiInternTableFind(s->chars, s->length, hash);
    if (found != NULL) return found;
    internString(s);
    return s;
}

/* Should a run-time string of this length take part in interning at all —
 * both the probe and the insert? */
static inline bool runtimeInternable(size_t length) {
    return length <= JAI_INTERN_MAX &&
           jaiInternTableCount() < JAI_INTERN_SOFT_CAP;
}

/* The 128 one-byte ASCII strings, made once and kept forever.
 *
 * `s[i]` builds a string, and before this it built a fresh one every time:
 * allocate, hash the byte, probe the intern table, and give the collector
 * another object to walk. `str_search` reads a million characters and
 * `word_freq`'s scanner reads a quarter of a million, but so does every lexer,
 * parser and text-munging loop ever written in this language -- the front end
 * itself is one. Interning already made them one object; this makes them free.
 *
 * Held strongly here because the intern table's references are weak (see
 * jaiTableRemoveWhite in gc.c), so nothing else would keep them alive.
 *
 * External linkage (declared `extern` in object.h, where jaiAsciiCharTable()
 * is also defined, `static inline`, as the read side): the string iterator's
 * per-character fast path in object_iter.c needs to reach these slots without
 * a real function call, the same way jaiStringChar below does. This is the
 * only field in the object model any other object_*.c file reaches directly
 * rather than through a declared function. */
ObjString *jaiAsciiChars[128];

void jaiMarkAsciiChars(void) {
    for (unsigned i = 0; i < 128; ++i) {
        if (jaiAsciiChars[i] != NULL)
            jaiGCMarkObject((Obj *)jaiAsciiChars[i]);
    }
}

void jaiAsciiCharsFill(void) {
    for (unsigned i = 0; i < 128; ++i) (void)jaiStringChar((unsigned char)i);
}

void jaiAsciiCharsReset(void) {
    for (unsigned i = 0; i < 128; ++i) jaiAsciiChars[i] = NULL;
}

ObjString *jaiStringChar(unsigned char c) {
    if (c >= 128) return NULL;

    ObjString *cached = jaiAsciiChars[c];
    if (cached != NULL) return cached;

    const char one = (char)c;
    const uint64_t hash = jaiHashBytes(&one, 1);
    cached = jaiInternTableFind(&one, 1, hash);

    if (cached == NULL) {
        cached = allocString(1);
        cached->chars[0] = one;
        cached->hash = hash;
        internString(cached);
    }

    jaiAsciiChars[c] = cached;
    return cached;
}

ObjString *jaiStringNew(const char *chars, size_t length) {
    if (length == 1 && (unsigned char)chars[0] < 128)
        return jaiStringChar((unsigned char)chars[0]);

    if (length > UINT32_MAX) {
        jaiThrow(vm.cValueError,
                 "string of %zu bytes exceeds the maximum length", length);
        return NULL;
    }

    bool insert = false;
    uint64_t hash = 0;

    if (runtimeInternable(length)) {
        hash = jaiHashBytes(chars, length);
        ObjString *found = jaiInternTableFind(chars, length, hash);
        if (found != NULL) return found;
        insert = true;
    }

    ObjString *s = allocString(length);
    if (length != 0) memcpy(s->chars, chars, length);
    s->hash = hash;
    if (insert) internString(s);
    return s;
}

ObjString *jaiStringTake(char *chars, size_t length) {
    if (length > UINT32_MAX) {
        (void)jaiRealloc(chars, length + 1, 0);
        jaiThrow(vm.cValueError,
                 "string of %zu bytes exceeds the maximum length", length);
        return NULL;
    }

    if (length == 1 && (unsigned char)chars[0] < 128) {
        const unsigned char c = (unsigned char)chars[0];
        (void)jaiRealloc(chars, length + 1, 0);
        return jaiStringChar(c);
    }

    bool insert = false;
    uint64_t hash = 0;

    if (runtimeInternable(length)) {
        hash = jaiHashBytes(chars, length);
        ObjString *found = jaiInternTableFind(chars, length, hash);
        if (found != NULL) {
            (void)jaiRealloc(chars, length + 1, 0);
            return found;
        }
        insert = true;
    }

    ObjString *s = allocString(length);
    if (length != 0) memcpy(s->chars, chars, length);
    s->hash = hash;
    (void)jaiRealloc(chars, length + 1, 0);
    if (insert) internString(s);
    return s;
}

/* Growing a string by copying the whole accumulation each time makes the
 * ordinary `text = text + piece` loop quadratic: forty thousand steps over a
 * 280KB result move 5.6GB. These three give that loop spare capacity to grow
 * into, so it moves each byte once. */
static ObjStrBuf *strBufNew(size_t capacity) {
    if (capacity > UINT32_MAX) capacity = UINT32_MAX;

    ObjStrBuf *b = (ObjStrBuf *)jaiAllocateObjectRaw(
        sizeof(ObjStrBuf) + capacity + 1, OBJ_STRBUF);
    b->capacity = (uint32_t)capacity;
    b->used = 0;
    b->data[0] = '\0';
    return b;
}

/* A view of the first `length` bytes of `buf`. Header only: the bytes belong to
 * the buffer, which the collector keeps alive through `owner`. */
static ObjString *strSliceOver(ObjStrBuf *buf, size_t length) {
    ObjString *s = (ObjString *)jaiAllocateObjectRaw(sizeof(ObjString),
                                                     OBJ_STRING);
    s->chars = buf->data;
    s->owner = buf;
    s->length = (uint32_t)length;
    s->scalars = UINT32_MAX;
    s->cursorScalar = 0;
    s->cursorByte = 0;
    s->hash = 0;
    JAI_STR_INTERNED(s) = false;
    JAI_STR_UNTERMINATED(s) = false;
    return s;
}

const char *jaiStringCStr(ObjString *s) {
    if (!JAI_STR_UNTERMINATED(s)) return s->chars;
    /* Somebody appended past this view, so the byte after it is no longer a
     * NUL. The bytes up to `length` are still exactly right, so a fresh copy
     * of them is a correct C string. */
    ObjString *copy = jaiStringNew(s->chars, s->length);
    return copy != NULL ? copy->chars : "";
}

ObjString *jaiStringConcat(ObjString *a, ObjString *b) {
    size_t length = (size_t)a->length + (size_t)b->length;
    if (length > UINT32_MAX) {
        jaiThrow(vm.cOverflowError, "concatenated string of %zu bytes exceeds "
                 "the maximum length", length);
        return NULL;
    }
    if (length == 0) return jaiStringIntern("", 0);

    /* Short results go through the run-time intern policy: `a + b` in a loop
     * over a small alphabet repeats constantly. */
    if (length <= JAI_INTERN_MAX) {
        char buf[JAI_INTERN_MAX];
        memcpy(buf, a->chars, a->length);
        memcpy(buf + a->length, b->chars, b->length);
        return jaiStringNew(buf, length);
    }

    jaiGCPushRoot(OBJ_VAL(a));
    jaiGCPushRoot(OBJ_VAL(b));

    /* `a` is the newest view of its buffer and there is room: write `b` after
     * it. Every older view ends before this point and is untouched. */
    ObjStrBuf *buf = a->owner;
    if (buf != NULL && buf->used == a->length && buf->capacity >= length) {
        memcpy(buf->data + a->length, b->chars, b->length);
        buf->used = (uint32_t)length;
        buf->data[length] = '\0';
        /* `a`'s terminator was the byte just overwritten. */
        JAI_STR_UNTERMINATED(a) = true;
        ObjString *grown = strSliceOver(buf, length);
        jaiGCPopRoots(2);
        return grown;
    }

    /* No buffer, or somebody else already appended to it. Start one with room
     * to double, so a growing chain stops copying after this. */
    {
        size_t want;
        if (length < 32) {
            want = 64;
        } else if (length <= (size_t)UINT32_MAX / 2u) {
            want = length * 2u;
        } else {
            want = length;
        }
        ObjStrBuf *fresh = strBufNew(want);
        if (fresh != NULL) {
            jaiGCPushRoot(OBJ_VAL(fresh));
            memcpy(fresh->data, a->chars, a->length);
            memcpy(fresh->data + a->length, b->chars, b->length);
            fresh->used = (uint32_t)length;
            fresh->data[length] = '\0';
            ObjString *made = strSliceOver(fresh, length);
            jaiGCPopRoots(3);
            return made;
        }
    }

    ObjString *s = allocString(length);
    memcpy(s->chars, a->chars, a->length);
    memcpy(s->chars + a->length, b->chars, b->length);
    /* Over JAI_INTERN_MAX, so nothing is going to probe for it; leave the hash
     * to jaiStringHash if anyone ever asks. */
    jaiGCPopRoots(2);
    return s;
}

/* One allocation, sized exactly, for the n-way concatenation an f-string
 * performs. Short results still go through jaiStringNew so that the run-time
 * interning policy sees them; a long one is built in place, which is the point
 * — the pairwise OP_CONCAT lowering this replaces allocated an intermediate
 * string per hole and copied the prefix again for each one. */
ObjString *jaiStringFromParts(const char *const *runs, const uint32_t *lens,
                              int count, size_t total) {
    if (total > UINT32_MAX) {
        jaiThrow(vm.cOverflowError, "formatted string of %zu bytes exceeds the "
                 "maximum length", total);
        return NULL;
    }
    if (total == 0) return jaiStringIntern("", 0);

    if (runtimeInternable(total)) {
        /* Short enough to be worth a probe, so it is staged where the probe
         * can hash it before anything is allocated.
         *
         * Byte at a time rather than memcpy per run. The runs an f-string
         * produces are one to five bytes each -- `f"k{i}"` is a one-byte
         * literal and four digits -- and at that length the call into
         * _platform_memmove costs several times the copy. Two of them per
         * f-string were 6.6% of tests/bench/dict_ops by sample count. */
        char buf[JAI_INTERN_MAX];
        size_t o = 0;

        for (int i = 0; i < count; i++) {
            const char *const src = runs[i];
            for (uint32_t j = 0, n = lens[i]; j < n; ++j)
                buf[o++] = src[j];
        }
        return jaiStringNew(buf, total);
    }

    /* Nothing is going to probe for this one -- it is over JAI_INTERN_MAX, or
     * the intern table is already at its soft cap -- so there is nothing to
     * hash and the staging buffer would be a copy for its own sake. The runs
     * go straight into the string. tests/bench/string_build builds two million
     * thirteen-byte keys and all but the first thirty-two thousand take this
     * path; before, every one of them was copied twice.
     *
     * allocString may collect, but it collects *before* it allocates and the
     * runs belong to the caller's roots, so they are still there afterwards.
     * The hash is left to jaiStringHash. */
    ObjString *s = allocString(total);
    char *dst = s->chars;
    for (int i = 0; i < count; ++i) {
        const uint32_t len = lens[i];
        if (len == 0) continue;
        /* Same trade as above, and the same reason: a run this short is
         * cheaper copied here than handed to memcpy. */
        if (len <= 8) {
            const char *const src = runs[i];
            for (uint32_t j = 0; j < len; ++j) *dst++ = src[j];
        } else {
            memcpy(dst, runs[i], len);
            dst += len;
        }
    }
    return s;
}

/* Reserve/seal: for a caller that can size the result up front and would
 * otherwise build it in a JaiBuf and copy it in. str.join over a list is the
 * case — the old path grew a buffer by doubling, reallocated it to size, then
 * memcpy'd the whole thing into the string, which on the 2.3 MB join in
 * tests/bench/string_build meant three passes over the bytes instead of one. */
ObjString *jaiStringReserve(size_t length) {
    if (length > UINT32_MAX) {
        jaiThrow(vm.cValueError, "string of %zu bytes exceeds the maximum length",
                 length);
        return NULL;
    }
    return allocString(length);
}

ObjString *jaiStringSeal(ObjString *s) {
    if (s == NULL) return NULL;

    if (s->length == 1 && (unsigned char)s->chars[0] < 128)
        return jaiStringChar((unsigned char)s->chars[0]);

    if (!runtimeInternable(s->length))
        return s;

    const uint64_t hash = jaiHashBytes(s->chars, s->length);
    ObjString *found = jaiInternTableFind(s->chars, s->length, hash);
    if (found != NULL) return found;

    s->hash = hash;
    internString(s);
    return s;
}

uint32_t jaiStringScalarCount(ObjString *s) {
    uint32_t count = s->scalars;
    if (count == UINT32_MAX) {
        count = (uint32_t)jaiUtf8Length(s->chars, s->length);
        s->scalars = count;
    }
    return count;
}

bool jaiStringEqualsSlow(const ObjString *a, const ObjString *b) {
    if (a == b) return true;

    const uint32_t length = a->length;
    if (length != b->length) return false;
    if (length == 0) return true;

    if (jaiStringHash((ObjString *)a) != jaiStringHash((ObjString *)b))
        return false;

    return memcmp(a->chars, b->chars, length) == 0;
}

/* Byte offset of every scalar in `s`, plus a terminator entry holding the byte
 * length. Only needed for non-ASCII strings; the caller frees it. */
static uint32_t *buildScalarOffsets(const ObjString *s, int64_t n) {
    uint32_t *offsets = JAI_ALLOC(uint32_t, (size_t)n + 1);
    const char *p = s->chars;
    const char *const end = p + s->length;
    int64_t i = 0;

    while (p < end && i < n) {
        offsets[i++] = (uint32_t)(p - s->chars);

        if ((unsigned char)*p < 0x80u) {
            ++p;
        } else {
            int len = 1;
            (void)jaiUtf8Decode(p, end, &len);
            p += len;
        }
    }

    while (i <= n) offsets[i++] = s->length;
    return offsets;
}

ObjString *jaiStringSlice(ObjString *s, int64_t start, int64_t stop,
                          int64_t step) {
    if (step == 0) {
        jaiThrow(vm.cValueError, "slice step cannot be zero");
        return NULL;
    }

    const int64_t n = (int64_t)jaiStringScalarCount(s);
    const int64_t count = sliceCount(n, &start, &stop, &step);
    if (count <= 0) return jaiStringIntern("", 0);

    const bool ascii = (int64_t)s->length == n;

    if (ascii && step == 1) {
        if (count == 1)
            return jaiStringChar((unsigned char)s->chars[start]);

        /* Contiguous ASCII needs no scalar walk or scratch buffer. */
        jaiGCPushRoot(OBJ_VAL(s));
        ObjString *result = jaiStringNew(s->chars + start, (size_t)count);
        jaiGCPopRoot();
        return result;
    }

    if (!ascii && step == 1) {
        const char *const end = s->chars + s->length;
        int64_t cursor = (int64_t)s->cursorScalar;
        const int64_t viaMemo = cursor <= start ? start - cursor : cursor - start;
        int64_t at = 0;
        const char *p = s->chars;

        if (viaMemo <= start) {
            at = cursor;
            p = s->chars + s->cursorByte;
        }

        for (; at < start && p < end; ++at) {
            if ((unsigned char)*p < 0x80u) {
                ++p;
            } else {
                int len = 1;
                (void)jaiUtf8Decode(p, end, &len);
                p += len;
            }
        }

        for (; at > start; --at) {
            do {
                --p;
            } while (p > s->chars &&
                     ((unsigned char)*p & 0xC0u) == 0x80u);
        }

        s->cursorScalar = (uint32_t)at;
        s->cursorByte = (uint32_t)(p - s->chars);

        const char *const from = p;
        for (int64_t i = 0; i < count && p < end; ++i) {
            if ((unsigned char)*p < 0x80u) {
                ++p;
            } else {
                int len = 1;
                (void)jaiUtf8Decode(p, end, &len);
                p += len;
            }
        }

        jaiGCPushRoot(OBJ_VAL(s));
        ObjString *result = jaiStringNew(from, (size_t)(p - from));
        jaiGCPopRoot();
        return result;
    }

    uint32_t *offsets = ascii ? NULL : buildScalarOffsets(s, n);

    size_t bytes;
    if (ascii) {
        bytes = (size_t)count;
    } else {
        bytes = 0;
        for (int64_t i = 0, idx = start; i < count; ++i, idx += step)
            bytes += (size_t)(offsets[idx + 1] - offsets[idx]);
    }

    jaiGCPushRoot(OBJ_VAL(s));
    ObjString *result = jaiStringReserve(bytes);
    jaiGCPopRoot();

    if (result == NULL) {
        if (offsets != NULL)
            JAI_FREE_ARRAY(uint32_t, offsets, (size_t)n + 1);
        return NULL;
    }

    char *dst = result->chars;
    if (ascii) {
        for (int64_t i = 0, idx = start; i < count; ++i, idx += step)
            *dst++ = s->chars[idx];
    } else {
        for (int64_t i = 0, idx = start; i < count; ++i, idx += step) {
            const size_t off = offsets[idx];
            const size_t len = (size_t)(offsets[idx + 1] - offsets[idx]);
            memcpy(dst, s->chars + off, len);
            dst += len;
        }
        JAI_FREE_ARRAY(uint32_t, offsets, (size_t)n + 1);
    }

    return jaiStringSeal(result);
}
