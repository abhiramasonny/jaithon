/* object.c — heap object construction, destruction, and the pure-data
 * operations on them (slicing, class layout, iteration).
 *
 * Nothing here interprets bytecode; the only calls back into the VM are
 * jaiThrow for runtime errors and jaiInvokeMethod for user-defined
 * __iter__/__next__.
 */

#include "object.h"

#include "gc.h"
#include "table.h"
#include "vm.h"

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
/* Allocation and lifetime                                              */
/* ------------------------------------------------------------------ */

static inline Obj *allocObj(size_t size, ObjType type) {
    /* Collect *before* the allocation: afterwards the new object is not yet
     * reachable from any root and a collection would free it. Guarding this
     * with the inlined jaiGCWanted() test used to measure a wash, because the
     * allocation dwarfed the call; once small objects came from a free list
     * that stopped being true. */
    if (JAI_UNLIKELY(jaiGCWanted())) jaiGCMaybeCollect();

    Obj *obj = (Obj *)jaiRealloc(NULL, 0, size);
    obj->type = type;
    obj->isMarked = false;
    obj->next = NULL;
    jaiGCTrackObject(obj);
    vm.allocCount++;
    return obj;
}

/* Hands back `size` bytes with only the Obj header set and everything after it
 * uninitialised. Worth it only where the payload is about to be overwritten
 * wholesale, which means the variable-length types: zeroing a string's
 * characters and then memcpying over them turned the memset into a call into
 * libc with a runtime length, and that was 11% of tests/bench/string_build. */
Obj *jaiAllocateObjectRaw(size_t size, ObjType type) {
    return allocObj(size, type);
}

Obj *jaiAllocateObject(size_t size, ObjType type) {
    Obj *obj = allocObj(size, type);
    /* Everything past the header starts zeroed, so a constructor that does not
     * mention a field leaves it NULL or 0 rather than garbage. */
    memset((char *)obj + sizeof(Obj), 0, size - sizeof(Obj));
    return obj;
}

/* Push `o` as a temporary root, tolerating NULL so that push/pop stay paired
 * in constructors whose arguments are optional. */
static inline void pushObjRoot(void *o) {
    jaiGCPushRoot(o != NULL ? OBJ_VAL(o) : NULL_VAL);
}

void jaiFreeObject(Obj *obj) {
    if (obj == NULL) return;

    switch (obj->type) {
    case OBJ_STRING: {
        /* Flexible array: header and payload are one block. The intern table
         * holds only weak references and is purged by the sweep, so there is
         * nothing to unlink here. */
        ObjString *s = (ObjString *)obj;
        /* A slice carries no bytes of its own; the buffer is swept separately
         * once nothing views it. */
        (void)jaiRealloc(obj, s->owner != NULL ? sizeof(ObjString)
                                               : JAI_STRING_ALLOC(s->length), 0);
        return;
    }
    case OBJ_STRBUF: {
        ObjStrBuf *b = (ObjStrBuf *)obj;
        (void)jaiRealloc(obj, sizeof(ObjStrBuf) + b->capacity + 1, 0);
        return;
    }
    case OBJ_BYTES: {
        ObjBytes *b = (ObjBytes *)obj;
        (void)jaiRealloc(obj, sizeof(ObjBytes) + b->length, 0);
        return;
    }
    case OBJ_TUPLE: {
        ObjTuple *t = (ObjTuple *)obj;
        (void)jaiRealloc(obj, sizeof(ObjTuple) + sizeof(Value) * (size_t)t->count, 0);
        return;
    }
    case OBJ_INSTANCE: {
        ObjInstance *inst = (ObjInstance *)obj;
        (void)jaiRealloc(obj, sizeof(ObjInstance) + sizeof(Value) * (size_t)inst->fieldCount, 0);
        return;
    }
    case OBJ_ENUM_VAL: {
        ObjEnumVal *ev = (ObjEnumVal *)obj;
        (void)jaiRealloc(obj, sizeof(ObjEnumVal) + sizeof(Value) * (size_t)ev->count, 0);
        return;
    }
    case OBJ_LIST: {
        ObjList *l = (ObjList *)obj;
        JAI_FREE_ARRAY(Value, l->items, l->capacity);
        JAI_FREE(ObjList, obj);
        return;
    }
    case OBJ_DICT: {
        ObjDict *d = (ObjDict *)obj;
        jaiTableFree(&d->table);
        JAI_FREE(ObjDict, obj);
        return;
    }
    case OBJ_SET: {
        ObjSet *s = (ObjSet *)obj;
        jaiTableFree(&s->table);
        JAI_FREE(ObjSet, obj);
        return;
    }
    case OBJ_RANGE:
        JAI_FREE(ObjRange, obj);
        return;
    case OBJ_FUNCTION: {
        ObjFunction *fn = (ObjFunction *)obj;
        jaiChunkFree(&fn->chunk);
        JAI_FREE_ARRAY(ObjString *, fn->paramNames, fn->paramCount);
        JAI_FREE_ARRAY(ExceptionEntry, fn->exceptions, fn->exceptionCount);
        JAI_FREE_ARRAY(uint32_t, fn->defaultOffsets, fn->defaultCount);
        JAI_FREE(ObjFunction, obj);
        return;
    }
    case OBJ_CLOSURE: {
        ObjClosure *c = (ObjClosure *)obj;
        JAI_FREE_ARRAY(ObjUpvalue *, c->upvalues, c->upvalueCount);
        JAI_FREE(ObjClosure, obj);
        return;
    }
    case OBJ_UPVALUE:
        JAI_FREE(ObjUpvalue, obj);
        return;
    case OBJ_NATIVE:
        JAI_FREE(ObjNative, obj);
        return;
    case OBJ_BOUND:
        JAI_FREE(ObjBound, obj);
        return;
    case OBJ_CLASS: {
        ObjClass *c = (ObjClass *)obj;
        JAI_FREE_ARRAY(FieldInfo, c->fields, c->fieldCount);
        JAI_FREE_ARRAY(ObjTrait *, c->traits, c->traitCount);
        jaiTableFree(&c->methods);
        jaiTableFree(&c->statics);
        jaiTableFree(&c->getters);
        jaiTableFree(&c->setters);
        jaiTableFree(&c->restricted);
        JAI_FREE(ObjClass, obj);
        return;
    }
    case OBJ_TRAIT: {
        ObjTrait *t = (ObjTrait *)obj;
        jaiTableFree(&t->required);
        jaiTableFree(&t->defaults);
        JAI_FREE_ARRAY(ObjTrait *, t->supers, t->superCount);
        JAI_FREE(ObjTrait, obj);
        return;
    }
    case OBJ_MODULE: {
        ObjModule *m = (ObjModule *)obj;
        jaiTableFree(&m->globals);
        jaiTableFree(&m->exports);
        JAI_FREE(ObjModule, obj);
        return;
    }
    case OBJ_ENUM: {
        ObjEnum *e = (ObjEnum *)obj;
        for (uint16_t i = 0; i < e->variantCount; i++) {
            JAI_FREE_ARRAY(ObjString *, e->variants[i].fieldNames,
                           e->variants[i].arity);
        }
        JAI_FREE_ARRAY(EnumVariant, e->variants, e->variantCount);
        jaiTableFree(&e->methods);
        JAI_FREE(ObjEnum, obj);
        return;
    }
    case OBJ_ITER:
        JAI_FREE(ObjIter, obj);
        return;
    case OBJ_FILE: {
        ObjFile *f = (ObjFile *)obj;
        /* A file reaching the collector without being closed is closed here;
         * the standard streams are never ours to close. */
        if (f->handle != NULL && !f->closed && f->handle != stdin &&
            f->handle != stdout && f->handle != stderr) {
            (void)fclose(f->handle);
        }
        f->handle = NULL;
        f->closed = true;
        JAI_FREE(ObjFile, obj);
        return;
    }
    case OBJ_ENUM_CTOR:
        JAI_FREE(ObjEnumCtor, obj);
        return;

    case OBJ_TYPE_COUNT:
        break;   /* not a real tag; fall through to the panic below */
    }

    /* Every real tag returns from its own case. Reaching here means obj->type
     * is corrupt, and there is no size with which to free it correctly. */
    JAI_PANIC("jaiFreeObject: object with invalid type tag %d", (int)obj->type);
}

const char *jaiObjTypeName(ObjType t) {
    switch (t) {
    case OBJ_STRING:    return "str";
    case OBJ_STRBUF:    return "str";   /* never user-visible */
    case OBJ_BYTES:     return "bytes";
    case OBJ_LIST:      return "list";
    case OBJ_DICT:      return "dict";
    case OBJ_SET:       return "set";
    case OBJ_TUPLE:     return "tuple";
    case OBJ_RANGE:     return "range";
    case OBJ_FUNCTION:  return "function";
    case OBJ_CLOSURE:   return "function";
    case OBJ_UPVALUE:   return "upvalue";
    case OBJ_NATIVE:    return "native function";
    case OBJ_BOUND:     return "method";
    case OBJ_CLASS:     return "class";
    case OBJ_TRAIT:     return "trait";
    case OBJ_INSTANCE:  return "instance";
    case OBJ_MODULE:    return "module";
    case OBJ_ENUM:      return "enum";
    case OBJ_ENUM_VAL:  return "enum value";
    case OBJ_ENUM_CTOR: return "function";
    case OBJ_ITER:      return "iterator";
    case OBJ_FILE:      return "file";
    case OBJ_TYPE_COUNT: break;
    }
    return "object";
}

/* ------------------------------------------------------------------ */
/* Slice normalisation (Python semantics)                               */
/* ------------------------------------------------------------------ */

/* Clamps [start, stop) / step against a sequence of `n` items, rewriting all
 * three in place, and returns how many items the slice selects. `step` must be
 * nonzero. Clamping |step| to n is safe (a step larger than the sequence can
 * select at most one item) and keeps `start + i * step` from overflowing. */
static inline int64_t sliceCount(int64_t n, int64_t *pStart,
                                 int64_t *pStop, int64_t *pStep) {
    int64_t start = *pStart;
    int64_t stop = *pStop;
    int64_t step = *pStep;

    if (n <= 0) {
        *pStart = 0;
        *pStop = 0;
        *pStep = step > 0 ? 1 : -1;
        return 0;
    }

    if (step == 1) {
        if (start < 0) start = start < -n ? 0 : start + n;
        else if (start > n) start = n;
        if (stop < 0) stop = stop < -n ? 0 : stop + n;
        else if (stop > n) stop = n;

        const int64_t count = stop > start ? stop - start : 0;
        *pStart = start;
        *pStop = stop;
        *pStep = 1;
        return count;
    }

    if (step == -1) {
        if (start < 0) start = start < -n ? -1 : start + n;
        else if (start >= n) start = n - 1;
        if (stop < 0) stop = stop < -n ? -1 : stop + n;
        else if (stop >= n) stop = n - 1;

        const int64_t count = start > stop ? start - stop : 0;
        *pStart = start;
        *pStop = stop;
        *pStep = -1;
        return count;
    }

    if (step > n) step = n;
    if (step < -n) step = -n;

    int64_t count;
    if (step > 0) {
        if (start < 0) start = start < -n ? 0 : start + n;
        else if (start > n) start = n;
        if (stop < 0) stop = stop < -n ? 0 : stop + n;
        else if (stop > n) stop = n;

        if (stop > start) {
            const uint64_t span = (uint64_t)(stop - start);
            const uint64_t stride = (uint64_t)step;
            count = (int64_t)(span / stride + (span % stride != 0));
        } else {
            count = 0;
        }
    } else {
        if (start < 0) start = start < -n ? -1 : start + n;
        else if (start >= n) start = n - 1;
        if (stop < 0) stop = stop < -n ? -1 : stop + n;
        else if (stop >= n) stop = n - 1;

        if (start > stop) {
            const uint64_t span = (uint64_t)(start - stop);
            const uint64_t stride = (uint64_t)(-step);
            count = (int64_t)(span / stride + (span % stride != 0));
        } else {
            count = 0;
        }
    }

    *pStart = start;
    *pStop = stop;
    *pStep = step;
    return count;
}

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
 * jaiTableRemoveWhite in gc.c), so nothing else would keep them alive. */
static ObjString *sAsciiChar[128];

void jaiMarkAsciiChars(void) {
    for (unsigned i = 0; i < 128; ++i) {
        if (sAsciiChar[i] != NULL)
            jaiGCMarkObject((Obj *)sAsciiChar[i]);
    }
}

ObjString **jaiAsciiCharTable(void) { return sAsciiChar; }

ObjString *jaiStringChar(unsigned char c) {
    if (c >= 128) return NULL;

    ObjString *cached = sAsciiChar[c];
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

    sAsciiChar[c] = cached;
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

    if (total <= JAI_INTERN_MAX) {
        /* Byte at a time rather than memcpy per run. The runs an f-string
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

    /* allocString may collect, but it collects *before* it allocates and the
     * runs belong to the caller's roots, so they are still there afterwards.
     * Over JAI_INTERN_MAX nothing will probe for this, so the hash is left to
     * jaiStringHash. */
    ObjString *s = allocString(total);
    char *dst = s->chars;
    for (int i = 0; i < count; ++i) {
        const uint32_t len = lens[i];
        if (len != 0) {
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

/* ------------------------------------------------------------------ */
/* Functions, closures, natives                                         */
/* ------------------------------------------------------------------ */

ObjFunction *jaiFunctionNew(void) {
    ObjFunction *fn = JAI_ALLOCATE_OBJ(ObjFunction, OBJ_FUNCTION);
    /* All scalar/pointer bookkeeping starts at zero/NULL by allocator contract. */
    jaiChunkInit(&fn->chunk, -1);
    return fn;
}

bool jaiFunctionIsDeferThunk(const ObjFunction *fn) {
    return fn != NULL && fn->name != NULL && fn->name->length == 5 &&
           memcmp(fn->name->chars, "defer", 5) == 0;
}

ObjUpvalue *jaiUpvalueNew(Value *slot) {
    ObjUpvalue *up = JAI_ALLOCATE_OBJ(ObjUpvalue, OBJ_UPVALUE);
    up->location = slot;
    up->closed = NULL_VAL;
    return up;
}

ObjUpvalue *jaiUpvalueClosed(Value v) {
    const bool root = IS_OBJ(v);
    if (root) jaiGCPushRoot(v);

    ObjUpvalue *up = JAI_ALLOCATE_OBJ(ObjUpvalue, OBJ_UPVALUE);

    if (root) jaiGCPopRoot();

    up->closed = v;
    up->location = &up->closed;
    return up;
}

ObjClosure *jaiClosureNew(ObjFunction *fn) {
    const int count = fn == NULL ? 0 : (int)fn->upvalueCount;

    pushObjRoot(fn);
    ObjUpvalue **upvalues = NULL;
    if (count > 0)
        upvalues = JAI_ALLOC_ZEROED(ObjUpvalue *, count);

    ObjClosure *closure = JAI_ALLOCATE_OBJ(ObjClosure, OBJ_CLOSURE);
    jaiGCPopRoot();

    closure->fn = fn;
    closure->upvalues = upvalues;
    closure->upvalueCount = count;
    return closure;
}

ObjNative *jaiNativeNew(JaiNativeFn fn, const char *name,
                        int minArity, int maxArity,
                        const char *const *paramNames) {
    ObjString *interned = jaiStringInternC(name);
    pushObjRoot(interned);
    ObjNative *native = JAI_ALLOCATE_OBJ(ObjNative, OBJ_NATIVE);
    jaiGCPopRoot();

    native->fn = fn;
    native->name = interned;
    native->minArity = (int8_t)(minArity < -128 ? -128 :
                                 (minArity > 127 ? 127 : minArity));
    native->maxArity = (int8_t)(maxArity < -128 ? -128 :
                                 (maxArity > 127 ? 127 : maxArity));
    native->paramNames = paramNames;
    return native;
}

ObjBound *jaiBoundNew(Value receiver, Value method) {
    const bool rootReceiver = IS_OBJ(receiver);
    const bool rootMethod = IS_OBJ(method);
    int roots = 0;

    if (rootReceiver) {
        jaiGCPushRoot(receiver);
        ++roots;
    }
    if (rootMethod) {
        jaiGCPushRoot(method);
        ++roots;
    }

    ObjBound *bound = JAI_ALLOCATE_OBJ(ObjBound, OBJ_BOUND);

    if (roots != 0) jaiGCPopRoots(roots);

    bound->receiver = receiver;
    bound->method = method;
    return bound;
}

/* ------------------------------------------------------------------ */
/* Classes                                                              */
/* ------------------------------------------------------------------ */

/* Shape ids are the inline-cache key; 0 means "no shape", so ids start at 1
 * and are never reused. */
static uint32_t nextShapeId = 1;

uint32_t jaiFreshShapeId(void) {
    uint32_t id = nextShapeId++;
    if (nextShapeId == 0) nextShapeId = 1;   /* 0 is reserved */
    return id;
}

/* The dunder cache mirrors these method names into fixed ObjClass fields so
 * that operator dispatch never touches a hash table. */
typedef struct {
    const char *name;
    uint8_t     length;
    size_t      offset;       /* of the Value field inside ObjClass */
} DunderEntry;

#define DUNDER(field, text) \
    {(text), (uint8_t)(sizeof(text) - 1u), offsetof(ObjClass, field)}

static const DunderEntry kDunders[] = {
    DUNDER(dunderStr,      "__str__"),
    DUNDER(dunderRepr,     "__repr__"),
    DUNDER(dunderEq,       "__eq__"),
    DUNDER(dunderLt,       "__lt__"),
    DUNDER(dunderHash,     "__hash__"),
    DUNDER(dunderAdd,      "__add__"),
    DUNDER(dunderSub,      "__sub__"),
    DUNDER(dunderMul,      "__mul__"),
    DUNDER(dunderDiv,      "__div__"),
    DUNDER(dunderMod,      "__mod__"),
    DUNDER(dunderPow,      "__pow__"),
    DUNDER(dunderNeg,      "__neg__"),
    DUNDER(dunderLen,      "__len__"),
    DUNDER(dunderGetItem,  "__getitem__"),
    DUNDER(dunderSetItem,  "__setitem__"),
    DUNDER(dunderContains, "__contains__"),
    DUNDER(dunderIter,     "__iter__"),
    DUNDER(dunderNext,     "__next__"),
    DUNDER(dunderCall,     "__call__"),
};

#define DUNDER_COUNT ((int)(sizeof(kDunders) / sizeof(kDunders[0])))

static inline void setDunderField(ObjClass *c, size_t offset, Value v) {
    memcpy((char *)c + offset, &v, sizeof v);
}

ObjClass *jaiClassNew(ObjString *name, ObjClass *superclass) {
    pushObjRoot(name);
    pushObjRoot(superclass);
    ObjClass *c = JAI_ALLOCATE_OBJ(ObjClass, OBJ_CLASS);
    jaiGCPopRoots(2);

    c->name = name;
    c->qualifiedName = name;
    c->superclass = superclass;
    c->shapeId = jaiFreshShapeId();
    jaiTableInit(&c->methods);
    jaiTableInit(&c->statics);
    jaiTableInit(&c->getters);
    jaiTableInit(&c->setters);
    jaiTableInit(&c->restricted);
    c->initializer = NULL_VAL;

    for (int i = 0; i < DUNDER_COUNT; ++i)
        setDunderField(c, kDunders[i].offset, NULL_VAL);

    return c;
}

void jaiClassInherit(ObjClass *sub, ObjClass *super) {
    if (sub == NULL || super == NULL) return;
    sub->superclass = super;

    const uint16_t superCount = super->fieldCount;
    const uint16_t ownCount = sub->fieldCount;
    FieldInfo *const own = sub->fields;
    const uint32_t wide = (uint32_t)superCount + ownCount;

    if (wide > UINT16_MAX) {
        jaiThrow(vm.cRuntimeError,
                 "class '%s' would have %u fields, the limit is %u",
                 sub->name != NULL ? sub->name->chars : "?",
                 wide, UINT16_MAX);
        return;
    }

    const uint16_t total = (uint16_t)wide;
    FieldInfo *merged = NULL;

    if (total != 0) {
        merged = JAI_ALLOC(FieldInfo, total);

        if (superCount != 0)
            memcpy(merged, super->fields,
                   sizeof(FieldInfo) * (size_t)superCount);

        if (ownCount != 0) {
            memcpy(merged + superCount, own,
                   sizeof(FieldInfo) * (size_t)ownCount);

            for (uint16_t i = 0; i < ownCount; ++i)
                merged[superCount + i].slot = (uint16_t)(superCount + i);
        }
    }

    JAI_FREE_ARRAY(FieldInfo, own, ownCount);
    sub->fields = merged;
    sub->fieldCount = total;

    jaiTableAddAll(&super->methods, &sub->methods);
    jaiTableAddAll(&super->statics, &sub->statics);
    jaiTableAddAll(&super->getters, &sub->getters);
    jaiTableAddAll(&super->setters, &sub->setters);
    jaiTableAddAll(&super->restricted, &sub->restricted);

    /* Inheritance runs before the subclass declares its own methods, so the
     * fixed dunder cache can be copied directly instead of re-interning and
     * probing every dunder name. Later jaiClassAddMethod calls overwrite the
     * individual slots when the subclass declares an override. */
    for (int i = 0; i < DUNDER_COUNT; ++i) {
        Value inherited;
        memcpy(&inherited,
            (const char *)super + kDunders[i].offset,
            sizeof inherited);
        setDunderField(sub, kDunders[i].offset, inherited);
    }

    if (IS_NULL(sub->initializer))
        sub->initializer = super->initializer;
}

void jaiClassAddMethod(ObjClass *c, ObjString *name, Value method,
                       Visibility vis, uint32_t flags) {
    JaiTable *target;
    if (flags & FN_GETTER) target = &c->getters;
    else if (flags & FN_SETTER) target = &c->setters;
    else if (flags & FN_STATIC) target = &c->statics;
    else target = &c->methods;

    const uint32_t nameLength = name != NULL ? name->length : 0;
    const bool isDunder =
        name != NULL && nameLength > 4 &&
        name->chars[0] == '_' && name->chars[1] == '_' &&
        name->chars[nameLength - 2] == '_' &&
        name->chars[nameLength - 1] == '_';

    jaiGCPushRoot(OBJ_VAL(c));
    jaiGCPushRoot(method);
    pushObjRoot(name);
    (void)jaiTableSetInterned(target, name, method);

    if (vis == VIS_PUBLIC || isDunder) {
        if (c->restricted.count > 0)
            (void)jaiTableDelete(&c->restricted, OBJ_VAL(name));
    } else {
        (void)jaiTableSetInterned(
            &c->restricted, name,
            INT_VAL((int64_t)vis |
                    ((int64_t)(flags & 0xFFFFu) << 8) |
                    ((int64_t)c->shapeId << 24)));
    }

    jaiGCPopRoots(3);

    if (target != &c->methods) return;

    if ((flags & FN_INIT) ||
        (nameLength == 4 && memcmp(name->chars, "init", 4) == 0)) {
        c->initializer = method;
        return;
    }

    for (int i = 0; i < DUNDER_COUNT; ++i) {
        const DunderEntry *const dunder = kDunders + i;
        if (nameLength != dunder->length) continue;
        if (memcmp(name->chars, dunder->name, dunder->length) != 0) continue;
        setDunderField(c, dunder->offset, method);
        return;
    }
}

bool jaiClassRestrictedMethod(ObjClass *c, ObjString *name, MethodInfo *out) {
    /* The whole point of the side table: a class with no non-public method
     * answers here, before any hashing. */
    if (c == NULL || name == NULL || c->restricted.count == 0) return false;
    Value packed;
    if (!jaiTableGetInterned(&c->restricted, name, &packed)) return false;
    if (!IS_INT(packed)) return false;

    int64_t bits = AS_INT(packed);
    out->name = name;
    out->visibility = (Visibility)(bits & 0xFF);
    out->flags = (uint32_t)((bits >> 8) & 0xFFFF);

    /* `private` is private to the *declaring* class, not to whoever inherited
     * the entry, so the verdict needs that class. Its shapeId was packed in
     * when the method was added; recovering the pointer is a walk up the
     * superclass chain with no hashing at all. */
    uint32_t ownerShape = (uint32_t)((uint64_t)bits >> 24);
    out->owner = c;
    for (ObjClass *k = c; k != NULL; k = k->superclass) {
        if (k->shapeId == ownerShape) {
            out->owner = k;
            break;
        }
    }
    return true;
}

int jaiClassFieldSlot(ObjClass *c, ObjString *name) {
    const FieldInfo *info = jaiClassFieldInfo(c, name);
    return info == NULL ? -1 : (int)info->slot;
}

const FieldInfo *jaiClassFieldInfo(ObjClass *c, ObjString *name) {
    if (c == NULL || name == NULL) return NULL;

    const uint16_t count = c->fieldCount;
    const FieldInfo *const fields = c->fields;

    /* Compiler/deserializer names are canonical. For an interned lookup name,
     * equal field names must therefore be pointer-identical. */
    if (JAI_STR_INTERNED(name)) {
        for (uint16_t i = 0; i < count; ++i) {
            if (fields[i].name == name)
                return fields + i;
        }
        return NULL;
    }

    for (uint16_t i = 0; i < count; ++i) {
        const FieldInfo *const field = fields + i;
        if (field->name == name || jaiStringEquals(field->name, name))
            return field;
    }

    return NULL;
}

bool jaiClassIsSubclassOf(const ObjClass *sub, const ObjClass *super) {
    if (super == NULL) return false;
    for (const ObjClass *c = sub; c != NULL; c = c->superclass) {
        if (c == super) return true;
    }
    return false;
}

/* Depth-limited walk of a trait's supertrait DAG. The limit stops a cyclic
 * trait graph — which the checker rejects, but the VM must not hang on. */
static bool traitSatisfies(const ObjTrait *have, const ObjTrait *want,
                           int depth) {
    if (have == NULL || depth > 32) return false;
    if (have == want) return true;
    for (uint16_t i = 0; i < have->superCount; i++) {
        if (traitSatisfies(have->supers[i], want, depth + 1)) return true;
    }
    return false;
}

bool jaiClassImplements(const ObjClass *c, const ObjTrait *t) {
    if (t == NULL) return false;
    for (const ObjClass *k = c; k != NULL; k = k->superclass) {
        for (uint16_t i = 0; i < k->traitCount; i++) {
            if (traitSatisfies(k->traits[i], t, 0)) return true;
        }
    }
    return false;
}

void jaiClassRefreshDunders(ObjClass *c) {
    if (c == NULL) return;
    jaiGCPushRoot(OBJ_VAL(c));
    for (int i = 0; i < DUNDER_COUNT; i++) {
        /* Interning is idempotent: a dunder present in the table is keyed by
         * exactly this pointer, so the interned lookup is a pointer compare. */
        ObjString *name = jaiStringInternC(kDunders[i].name);
        Value m;
        if (name != NULL && jaiTableGetInterned(&c->methods, name, &m)) {
            setDunderField(c, kDunders[i].offset, m);
        } else {
            setDunderField(c, kDunders[i].offset, NULL_VAL);
        }
    }
    ObjString *init = jaiStringInternC("init");
    Value initFn;
    if (init != NULL && jaiTableGetInterned(&c->methods, init, &initFn)) {
        c->initializer = initFn;
    }
    jaiGCPopRoot();
}

/* ------------------------------------------------------------------ */
/* Traits                                                               */
/* ------------------------------------------------------------------ */

ObjTrait *jaiTraitNew(ObjString *name) {
    pushObjRoot(name);
    ObjTrait *trait = JAI_ALLOCATE_OBJ(ObjTrait, OBJ_TRAIT);
    jaiGCPopRoot();

    trait->name = name;
    jaiTableInit(&trait->required);
    jaiTableInit(&trait->defaults);
    return trait;
}

/* ------------------------------------------------------------------ */
/* Instances                                                            */
/* ------------------------------------------------------------------ */

ObjInstance *jaiInstanceNew(ObjClass *klass) {
    const uint16_t count = klass == NULL ? 0 : klass->fieldCount;

    pushObjRoot(klass);
    ObjInstance *inst = (ObjInstance *)jaiAllocateObjectRaw(
        sizeof(ObjInstance) + sizeof(Value) * (size_t)count,
        OBJ_INSTANCE);
    jaiGCPopRoot();

    inst->klass = klass;
    inst->fieldCount = count;

    for (uint16_t i = 0; i < count; ++i)
        inst->fields[i] = NULL_VAL;

    return inst;
}

/* ------------------------------------------------------------------ */
/* Enums                                                                */
/* ------------------------------------------------------------------ */

ObjEnum *jaiEnumNew(ObjString *name) {
    pushObjRoot(name);
    ObjEnum *e = JAI_ALLOCATE_OBJ(ObjEnum, OBJ_ENUM);
    jaiGCPopRoot();

    e->name = name;
    jaiTableInit(&e->methods);
    e->shapeId = jaiFreshShapeId();
    return e;
}

ObjEnumCtor *jaiEnumCtorNew(ObjEnum *e, uint16_t tag) {
    pushObjRoot(e);
    ObjEnumCtor *ctor = JAI_ALLOCATE_OBJ(ObjEnumCtor, OBJ_ENUM_CTOR);
    jaiGCPopRoot();

    ctor->type = e;
    ctor->tag = tag;
    return ctor;
}

ObjEnumVal *jaiEnumValNew(ObjEnum *e, uint16_t tag,
                          const Value *payload, int count) {
    if (count < 0) count = 0;
    if (count > 255) count = 255;

    pushObjRoot(e);
    ObjEnumVal *value = (ObjEnumVal *)jaiAllocateObjectRaw(
        sizeof(ObjEnumVal) + sizeof(Value) * (size_t)count,
        OBJ_ENUM_VAL);
    jaiGCPopRoot();

    value->type = e;
    value->tag = tag;
    value->count = (uint8_t)count;

    if (count != 0) {
        if (payload != NULL) {
            memcpy(value->payload, payload,
                   sizeof(Value) * (size_t)count);
        } else {
            for (int i = 0; i < count; ++i)
                value->payload[i] = NULL_VAL;
        }
    }

    return value;
}

/* ------------------------------------------------------------------ */
/* Modules                                                              */
/* ------------------------------------------------------------------ */

ObjModule *jaiModuleNew(ObjString *name, ObjString *path) {
    pushObjRoot(name);
    pushObjRoot(path);
    ObjModule *module = JAI_ALLOCATE_OBJ(ObjModule, OBJ_MODULE);
    jaiGCPopRoots(2);

    module->name = name;
    module->path = path;
    jaiTableInit(&module->globals);
    jaiTableInit(&module->exports);
    module->state = MOD_UNLOADED;
    module->sourceFileId = -1;
    return module;
}

bool jaiModuleGet(ObjModule *m, ObjString *name, Value *out) {
    return jaiTableGetInterned(&m->globals, name, out);
}

void jaiModuleSet(ObjModule *m, ObjString *name, Value v) {
    jaiGCPushRoot(OBJ_VAL(m));
    jaiGCPushRoot(v);
    Value prev = NULL_VAL;
    const bool added = jaiTableSetInternedPrev(&m->globals, name, v, &prev);
    jaiGCPopRoots(2);

    /* ObjModule::version retires compiled code, and compiled code resolves a
     * global by VALUE exactly four ways -- globalClass, globalFunction,
     * globalNative and globalIsSelf in jit_func.c -- each of which demands
     * IS_CLASS, IS_CLOSURE or IS_NATIVE. Everything else it resolves by
     * ADDRESS, re-loading the value behind a tag guard on every access, so an
     * update to an inert value needs no invalidation at all.
     *
     * jaiValueIsInertGlobal is the test, and it is written the safe way round:
     * it lists the types compiled code provably cannot bake and answers false
     * for everything else, so a new ObjType falls into the conservative arm.
     *
     * The key set and the table layout are NOT tracked here -- JaiTable bumps
     * keyVersion itself, so no writer can forget to. */
    if (added || ((IS_OBJ(prev) || IS_OBJ(v)) &&
                  (!jaiValueIsInertGlobal(prev) || !jaiValueIsInertGlobal(v)))) {
        m->version++;
    }
}

bool jaiModuleIsExported(ObjModule *m, ObjString *name) {
    Value ignored;
    return jaiTableGetInterned(&m->exports, name, &ignored);
}

/* ------------------------------------------------------------------ */
/* Iterators                                                            */
/* ------------------------------------------------------------------ */

/* Interned dunder names are cached on the VM at startup; fall back to
 * interning on demand so object.c works before jaiVMInit has run. */
static inline ObjString *iterName(void) {
    return vm.strIter != NULL ? vm.strIter : jaiStringInternC("__iter__");
}

static inline ObjString *nextName(void) {
    return vm.strNext != NULL ? vm.strNext : jaiStringInternC("__next__");
}

/* The trait spellings of the same two methods (std.core `Iterable`/`Iterator`).
 * They are ordinary names, so there is no cached intern for them. */
static inline ObjString *traitIterName(void) {
    return jaiStringInternC("iter");
}

static inline ObjString *traitNextName(void) {
    return jaiStringInternC("next");
}

/* True when instances of `v`'s class answer `name`. Inherited methods are
 * copied down at class creation, so one table lookup is the whole answer. */
ObjIter *jaiIterNew(IterKind kind, Value source) {
    const bool rootSource = IS_OBJ(source);
    if (rootSource) jaiGCPushRoot(source);

    ObjIter *it = JAI_ALLOCATE_OBJ(ObjIter, OBJ_ITER);

    if (rootSource) jaiGCPopRoot();

    it->kind = kind;
    it->source = source;

    switch (kind) {
        case ITER_LIST:
            if (IS_LIST(source)) {
                ObjList *const list = AS_LIST(source);
                it->limit = list->count;
                it->version = list->version;
            }
            break;

        case ITER_TUPLE:
            if (IS_TUPLE(source))
                it->limit = (int64_t)AS_TUPLE(source)->count;
            break;

        case ITER_STRING:
            if (IS_STRING(source))
                it->limit = (int64_t)AS_STRING(source)->length;
            break;

        case ITER_DICT_KEYS:
        case ITER_DICT_ITEMS:
            if (IS_DICT(source)) {
                JaiTable *const table = &AS_DICT(source)->table;
                it->limit = table->count;
                it->version = table->version;
            }
            break;

        case ITER_SET:
            if (IS_SET(source)) {
                JaiTable *const table = &AS_SET(source)->table;
                it->limit = table->count;
                it->version = table->version;
            }
            break;

        case ITER_RANGE:
            if (IS_RANGE(source))
                it->limit = jaiRangeLength(AS_RANGE(source));
            break;

        case ITER_USER:
        case ITER_TRAIT:
        case ITER_GENERATOR:
            break;
    }

    return it;
}

/* Both forms are fatal to the traversal, but they are reported apart because
 * the reader's next question differs: a resize invalidates the iterator's
 * bounds, while an in-place store leaves them valid and silently changes what
 * the loop sees. */
static inline bool iterMutated(bool resized) {
    return jaiThrow(vm.cRuntimeError,
                    resized ? "container changed size during iteration"
                            : "container was modified during iteration");
}

/* True when the pending exception is a StopIteration, i.e. an ordinary end of
 * a user-defined iterator rather than a failure. */
static inline bool pendingIsStopIteration(void) {
    if (!vm.hasException || vm.cStopIteration == NULL)
        return false;

    const Value exception = vm.pendingException;

    if (IS_INSTANCE(exception))
        return jaiClassIsSubclassOf(AS_INSTANCE(exception)->klass,
                                    vm.cStopIteration);

    if (IS_CLASS(exception))
        return jaiClassIsSubclassOf(AS_CLASS(exception), vm.cStopIteration);

    return false;
}

/* Two exhaustion protocols meet here. `__next__` (spec §7.1) ends by raising
 * StopIteration; `trait Iterator.next` (spec §9), which the whole standard
 * library is written against, ends by returning null. Which one applies is
 * decided by which method the object actually has, so neither has to know
 * about the other. */
static bool iterUserNext(ObjIter *it, Value *out) {
    Value result;
    if (!jaiInvokeMethod(it->source, nextName(), 0, NULL, &result)) {
        if (pendingIsStopIteration()) {
            jaiClearException();
            return false;
        }
        if (!vm.hasException) {
            jaiThrow(vm.cTypeError, "'%s' object has no __next__ method",
                     jaiTypeNameStatic(it->source));
        }
        return false;
    }
    *out = result;
    return true;
}

/* std.core's contract: `next` returns null at the end and must keep returning
 * null afterwards. An iterator over values that can themselves be null wraps
 * them, which is why null is safe to read as "done" here. */
static bool iterTraitNext(ObjIter *it, Value *out) {
    Value result;
    if (!jaiInvokeMethod(it->source, traitNextName(), 0, NULL, &result)) {
        if (pendingIsStopIteration()) {
            jaiClearException();
            return false;
        }
        if (!vm.hasException) {
            jaiThrow(vm.cTypeError, "'%s' object has no `next` method",
                     jaiTypeNameStatic(it->source));
        }
        return false;
    }
    if (IS_NULL(result)) return false;
    *out = result;
    return true;
}

bool jaiIterNext(ObjIter *it, Value *out) {
    switch (it->kind) {
        case ITER_LIST: {
            ObjList *const list = AS_LIST(it->source);
            if (JAI_UNLIKELY(list->version != it->version))
                return iterMutated((int64_t)list->count != it->limit);

            const int64_t index = it->index;
            if (index >= it->limit) return false;

            *out = list->items[index];
            it->index = index + 1;
            return true;
        }

        case ITER_TUPLE: {
            const int64_t index = it->index;
            if (index >= it->limit) return false;

            *out = AS_TUPLE(it->source)->items[index];
            it->index = index + 1;
            return true;
        }

        case ITER_STRING: {
            ObjString *const string = AS_STRING(it->source);
            const int64_t index = it->index;
            if (index >= it->limit) return false;

            const char *const p = string->chars + index;
            const unsigned char first = (unsigned char)*p;

            if (first < 0x80u) {
                ObjString *scalar = sAsciiChar[first];
                if (scalar == NULL) {
                    jaiGCPushRoot(OBJ_VAL(it));
                    scalar = jaiStringChar(first);
                    jaiGCPopRoot();
                    if (scalar == NULL) return false;
                }

                it->index = index + 1;
                *out = OBJ_VAL(scalar);
                return true;
            }

            int len = 1;
            (void)jaiUtf8Decode(p, string->chars + string->length, &len);
            if (index + len > it->limit)
                len = (int)(it->limit - index);

            jaiGCPushRoot(OBJ_VAL(it));
            ObjString *scalar = jaiStringNew(p, (size_t)len);
            jaiGCPopRoot();
            if (scalar == NULL) return false;

            it->index = index + len;
            *out = OBJ_VAL(scalar);
            return true;
        }

        case ITER_DICT_KEYS:
        case ITER_DICT_ITEMS: {
            ObjDict *const dict = AS_DICT(it->source);
            JaiTable *const table = &dict->table;

            if (JAI_UNLIKELY(table->version != it->version))
                return iterMutated((int64_t)table->count != it->limit);

            int slot = (int)it->index;
            Value key, value;
            if (!jaiTableNext(table, &slot, &key, &value)) {
                it->index = slot;
                return false;
            }
            it->index = slot;

            if (it->kind == ITER_DICT_KEYS) {
                *out = key;
                return true;
            }

            Value pair[2] = {key, value};
            jaiGCPushRoot(OBJ_VAL(it));
            ObjTuple *tuple = jaiTupleNew(pair, 2);
            jaiGCPopRoot();
            *out = OBJ_VAL(tuple);
            return true;
        }

        case ITER_SET: {
            ObjSet *const set = AS_SET(it->source);
            JaiTable *const table = &set->table;

            if (JAI_UNLIKELY(table->version != it->version))
                return iterMutated((int64_t)table->count != it->limit);

            int slot = (int)it->index;
            Value key, ignored;
            if (!jaiTableNext(table, &slot, &key, &ignored)) {
                it->index = slot;
                return false;
            }

            it->index = slot;
            *out = key;
            return true;
        }

        case ITER_RANGE: {
            const int64_t index = it->index;
            if (index >= it->limit) return false;

            ObjRange *const range = AS_RANGE(it->source);
            uint64_t value;

            if (range->step == 1) {
                value = (uint64_t)range->start + (uint64_t)index;
            } else if (range->step == -1) {
                value = (uint64_t)range->start - (uint64_t)index;
            } else {
                value = (uint64_t)range->start +
                        (uint64_t)index * (uint64_t)range->step;
            }

            *out = INT_VAL((int64_t)value);
            it->index = index + 1;
            return true;
        }

        case ITER_USER:
        case ITER_GENERATOR:
            return iterUserNext(it, out);

        case ITER_TRAIT:
            return iterTraitNext(it, out);
    }

    return false;
}

bool jaiGetIter(Value v, Value *out) {
    if (IS_OBJ(v)) {
        switch (OBJ_TYPE(v)) {
            case OBJ_ITER:
                *out = v;
                return true;

            case OBJ_LIST:
                *out = OBJ_VAL(jaiIterNew(ITER_LIST, v));
                return true;

            case OBJ_TUPLE:
                *out = OBJ_VAL(jaiIterNew(ITER_TUPLE, v));
                return true;

            case OBJ_STRING:
                *out = OBJ_VAL(jaiIterNew(ITER_STRING, v));
                return true;

            case OBJ_DICT:
                *out = OBJ_VAL(jaiIterNew(ITER_DICT_KEYS, v));
                return true;

            case OBJ_SET:
                *out = OBJ_VAL(jaiIterNew(ITER_SET, v));
                return true;

            case OBJ_RANGE:
                *out = OBJ_VAL(jaiIterNew(ITER_RANGE, v));
                return true;

            case OBJ_INSTANCE: {
                ObjInstance *const instance = AS_INSTANCE(v);
                ObjClass *const klass = instance->klass;

                const bool hasDunderIter =
                    klass != NULL && !IS_NULL(klass->dunderIter);
                ObjString *const method =
                    hasDunderIter ? iterName() : traitIterName();

                Value result;
                if (!jaiInvokeMethod(v, method, 0, NULL, &result)) {
                    if (!vm.hasException)
                        jaiThrow(vm.cTypeError,
                                 "'%s' object is not iterable",
                                 jaiTypeNameStatic(v));
                    return false;
                }

                if (IS_ITER(result)) {
                    *out = result;
                    return true;
                }

                if (IS_INSTANCE(result)) {
                    ObjInstance *const iterator = AS_INSTANCE(result);
                    ObjClass *const iteratorClass = iterator->klass;
                    const IterKind kind =
                        iteratorClass != NULL &&
                        !IS_NULL(iteratorClass->dunderNext)
                            ? ITER_USER
                            : ITER_TRAIT;

                    *out = OBJ_VAL(jaiIterNew(kind, result));
                    return true;
                }

                return jaiGetIter(result, out);
            }

            default:
                break;
        }
    }

    return jaiThrow(vm.cTypeError, "'%s' object is not iterable",
                    jaiTypeNameStatic(v));
}

/* ------------------------------------------------------------------ */
/* Files                                                                */
/* ------------------------------------------------------------------ */

ObjFile *jaiFileNew(FILE *handle, ObjString *path,
                    const char *mode) {
    pushObjRoot(path);
    ObjFile *file = JAI_ALLOCATE_OBJ(ObjFile, OBJ_FILE);
    jaiGCPopRoot();

    const char *p = mode != NULL ? mode : "r";
    bool readable = false;
    bool writable = false;
    bool binary = false;
    bool update = false;

    for (; *p != '\0'; ++p) {
        switch (*p) {
            case '+': update = true; break;
            case 'r': readable = true; break;
            case 'w':
            case 'a':
            case 'x': writable = true; break;
            case 'b': binary = true; break;
            default: break;
        }
    }

    file->handle = handle;
    file->path = path;
    file->readable = readable || update;
    file->writable = writable || update;
    file->binary = binary;
    file->closed = handle == NULL;
    return file;
}
