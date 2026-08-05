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
static void pushObjRoot(void *o) {
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
        (void)jaiRealloc(obj, JAI_STRING_ALLOC(s->length), 0);
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
static int64_t sliceCount(int64_t n, int64_t *pStart, int64_t *pStop,
                          int64_t *pStep) {
    int64_t start = *pStart, stop = *pStop, step = *pStep;

    if (n <= 0) {
        *pStart = 0;
        *pStop = 0;
        *pStep = (step > 0) ? 1 : -1;
        return 0;
    }
    if (step > n) step = n;
    if (step < -n) step = -n;

    int64_t count;
    if (step > 0) {
        if (start < 0)      start = (start < -n) ? 0 : start + n;
        else if (start > n) start = n;
        if (stop < 0)       stop = (stop < -n) ? 0 : stop + n;
        else if (stop > n)  stop = n;
        count = (stop > start) ? (stop - start + step - 1) / step : 0;
    } else {
        if (start < 0)        start = (start < -n) ? -1 : start + n;
        else if (start >= n)  start = n - 1;
        if (stop < 0)         stop = (stop < -n) ? -1 : stop + n;
        else if (stop >= n)   stop = n - 1;
        count = (start > stop) ? (start - stop + (-step) - 1) / (-step) : 0;
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
static void internString(ObjString *s) {
    JAI_STR_INTERNED(s) = true;
    jaiGCPushRoot(OBJ_VAL(s));
    jaiInternTableAdd(s);
    jaiGCPopRoot();
}

ObjString *jaiStringIntern(const char *chars, size_t length) {
    if (length > UINT32_MAX) {
        jaiThrow(vm.cValueError, "string of %zu bytes exceeds the maximum length",
                 length);
        return NULL;
    }
    uint64_t hash = jaiHashBytes(chars, length);
    ObjString *found = jaiInternTableFind(chars, length, hash);
    if (found != NULL) return found;

    ObjString *s = allocString(length);
    if (length > 0) memcpy(s->chars, chars, length);
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
static bool runtimeInternable(size_t length) {
    return length <= JAI_INTERN_MAX &&
           jaiInternTableCount() < JAI_INTERN_SOFT_CAP;
}

ObjString *jaiStringNew(const char *chars, size_t length) {
    if (length > UINT32_MAX) {
        jaiThrow(vm.cValueError, "string of %zu bytes exceeds the maximum length",
                 length);
        return NULL;
    }
    /* Hash only what the intern table is going to look at; a string the
     * policy excludes may never need one. */
    bool insert = false;
    uint64_t hash = 0;
    if (runtimeInternable(length)) {
        hash = jaiHashBytes(chars, length);
        ObjString *found = jaiInternTableFind(chars, length, hash);
        if (found != NULL) return found;
        insert = true;
    }

    ObjString *s = allocString(length);
    if (length > 0) memcpy(s->chars, chars, length);
    s->hash = hash;
    if (insert) internString(s);
    return s;
}

ObjString *jaiStringTake(char *chars, size_t length) {
    /* The buffer cannot become the payload — ObjString stores its bytes in a
     * flexible array — so ownership means "copy out, then free". */
    if (length > UINT32_MAX) {
        (void)jaiRealloc(chars, length + 1, 0);
        jaiThrow(vm.cValueError, "string of %zu bytes exceeds the maximum length",
                 length);
        return NULL;
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
    if (length > 0) memcpy(s->chars, chars, length);
    s->hash = hash;
    (void)jaiRealloc(chars, length + 1, 0);
    if (insert) internString(s);
    return s;
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
        char buf[JAI_INTERN_MAX];
        size_t o = 0;
        for (int i = 0; i < count; i++) {
            memcpy(buf + o, runs[i], lens[i]);
            o += lens[i];
        }
        return jaiStringNew(buf, total);
    }

    /* allocString may collect, but it collects *before* it allocates and the
     * runs belong to the caller's roots, so they are still there afterwards.
     * Over JAI_INTERN_MAX nothing will probe for this, so the hash is left to
     * jaiStringHash. */
    ObjString *s = allocString(total);
    size_t o = 0;
    for (int i = 0; i < count; i++) {
        memcpy(s->chars + o, runs[i], lens[i]);
        o += lens[i];
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
    if (!runtimeInternable(s->length)) return s;   /* hash it only on demand */

    uint64_t hash = jaiHashBytes(s->chars, s->length);
    ObjString *found = jaiInternTableFind(s->chars, s->length, hash);
    if (found != NULL) return found;     /* the reserved one becomes garbage */
    s->hash = hash;
    internString(s);
    return s;
}

uint32_t jaiStringScalarCount(ObjString *s) {
    if (s->scalars == UINT32_MAX) {
        s->scalars = (uint32_t)jaiUtf8Length(s->chars, s->length);
    }
    return s->scalars;
}

bool jaiStringEquals(const ObjString *a, const ObjString *b) {
    if (a == b) return true;
    if (a == NULL || b == NULL) return false;
    /* Two distinct interned strings are never equal, by construction. */
    if (JAI_STR_INTERNED(a) && JAI_STR_INTERNED(b)) return false;
    if (a->length != b->length) return false;
    if (jaiStringHash((ObjString *)a) != jaiStringHash((ObjString *)b))
        return false;
    return memcmp(a->chars, b->chars, a->length) == 0;
}

/* Byte offset of every scalar in `s`, plus a terminator entry holding the byte
 * length. Only needed for non-ASCII strings; the caller frees it. */
static uint32_t *buildScalarOffsets(const ObjString *s, int64_t n) {
    uint32_t *offsets = JAI_ALLOC(uint32_t, (size_t)n + 1);
    const char *p = s->chars;
    const char *end = s->chars + s->length;
    int64_t i = 0;
    while (p < end && i < n) {
        offsets[i++] = (uint32_t)(p - s->chars);
        int len = 1;
        (void)jaiUtf8Decode(p, end, &len);   /* len is 1 even for invalid bytes */
        p += len;
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
    int64_t n = (int64_t)jaiStringScalarCount(s);
    int64_t count = sliceCount(n, &start, &stop, &step);
    if (count <= 0) return jaiStringIntern("", 0);

    bool ascii = ((int64_t)s->length == n);

    /* A forward slice is bounded by two byte offsets, and a scan finds them
     * without the whole table. That matters: `text[i]` is a forward slice of
     * one scalar, a lexer does it once per character, and building an offset
     * table for the whole string each time makes lexing a file with a single
     * non-ASCII byte in it quadratic — 68 seconds for a 5 KB source. */
    if (!ascii && step == 1) {
        const char *end = s->chars + s->length;
        /* Reach `start` from whichever of the origin and the memo is nearer,
         * walking either way: UTF-8 is self-synchronising, so stepping back
         * over continuation bytes costs no more than stepping forwards. A memo
         * that only ran forwards was quadratic again on any interleaved walk —
         * a formatter reading a line and then a span inside that line reset it
         * to zero on every second call. */
        int64_t cursor = (int64_t)s->cursorScalar;
        int64_t viaMemo = cursor <= start ? start - cursor : cursor - start;
        int64_t at = 0;
        const char *p = s->chars;
        if (viaMemo <= start) {
            at = cursor;
            p = s->chars + s->cursorByte;
        }
        for (; at < start && p < end; at++) {
            int len = 1;
            (void)jaiUtf8Decode(p, end, &len);
            p += len;
        }
        for (; at > start; at--) {
            do { p--; } while (p > s->chars && ((unsigned char)*p & 0xC0) == 0x80);
        }
        s->cursorScalar = (uint32_t)at;
        s->cursorByte = (uint32_t)(p - s->chars);

        const char *from = p;
        for (int64_t i = 0; i < count && p < end; i++) {
            int len = 1;
            (void)jaiUtf8Decode(p, end, &len);
            p += len;
        }
        return jaiStringNew(from, (size_t)(p - from));
    }

    uint32_t *offsets = ascii ? NULL : buildScalarOffsets(s, n);

    size_t bytes = 0;
    for (int64_t i = 0, idx = start; i < count; i++, idx += step) {
        bytes += ascii ? 1u : (size_t)(offsets[idx + 1] - offsets[idx]);
    }

    char *buf = JAI_ALLOC(char, bytes + 1);
    size_t w = 0;
    for (int64_t i = 0, idx = start; i < count; i++, idx += step) {
        if (ascii) {
            buf[w++] = s->chars[idx];
        } else {
            size_t off = offsets[idx];
            size_t len = (size_t)(offsets[idx + 1] - offsets[idx]);
            memcpy(buf + w, s->chars + off, len);
            w += len;
        }
    }
    buf[bytes] = '\0';
    if (offsets != NULL) JAI_FREE_ARRAY(uint32_t, offsets, (size_t)n + 1);

    return jaiStringTake(buf, bytes);
}

/* ------------------------------------------------------------------ */
/* Bytes                                                                */
/* ------------------------------------------------------------------ */

ObjBytes *jaiBytesNew(const uint8_t *data, size_t length) {
    if (length > UINT32_MAX) {
        jaiThrow(vm.cValueError, "bytes of %zu bytes exceeds the maximum length",
                 length);
        return NULL;
    }
    ObjBytes *b = (ObjBytes *)jaiAllocateObject(sizeof(ObjBytes) + length,
                                                OBJ_BYTES);
    b->length = (uint32_t)length;
    if (data != NULL && length > 0) memcpy(b->data, data, length);
    return b;
}

/* ------------------------------------------------------------------ */
/* Lists                                                                */
/* ------------------------------------------------------------------ */

ObjList *jaiListNew(int initialCapacity) {
    ObjList *list = JAI_ALLOCATE_OBJ(ObjList, OBJ_LIST);
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
    list->version = 0;
    if (initialCapacity > 0) {
        /* Reserving allocates, which can collect; the list is not reachable
         * from any root yet. */
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
    if (list->count < list->capacity) return true;
    if (list->capacity > (INT32_MAX / 2)) {
        jaiThrow(vm.cRuntimeError, "list cannot grow beyond %d items", INT32_MAX);
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
    if (!listGrowFor(list, v)) return;
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
    if (!listGrowFor(list, v)) return;
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

ObjList *jaiListSlice(ObjList *list, int64_t start, int64_t stop, int64_t step) {
    if (step == 0) {
        jaiThrow(vm.cValueError, "slice step cannot be zero");
        return NULL;
    }
    int64_t count = sliceCount((int64_t)list->count, &start, &stop, &step);

    jaiGCPushRoot(OBJ_VAL(list));
    ObjList *out = jaiListNew((int)count);
    for (int64_t i = 0, idx = start; i < count; i++, idx += step) {
        out->items[out->count++] = list->items[idx];
    }
    jaiGCPopRoot();
    return out;
}

ObjList *jaiListConcat(ObjList *a, ObjList *b) {
    int64_t total = (int64_t)a->count + (int64_t)b->count;
    if (total > INT32_MAX) {
        jaiThrow(vm.cRuntimeError, "list cannot grow beyond %d items", INT32_MAX);
        return NULL;
    }
    jaiGCPushRoot(OBJ_VAL(a));
    jaiGCPushRoot(OBJ_VAL(b));
    ObjList *out = jaiListNew((int)total);
    if (a->count > 0) {
        memcpy(out->items, a->items, sizeof(Value) * (size_t)a->count);
    }
    if (b->count > 0) {
        memcpy(out->items + a->count, b->items, sizeof(Value) * (size_t)b->count);
    }
    out->count = (int)total;
    jaiGCPopRoots(2);
    return out;
}

bool jaiNormalizeIndex(int64_t raw, int length, int *out) {
    if (raw < 0) {
        if (raw < -(int64_t)length) return false;   /* also guards INT64_MIN */
        raw += length;
    }
    if (raw >= (int64_t)length) return false;
    *out = (int)raw;
    return true;
}

/* ------------------------------------------------------------------ */
/* Tuples                                                               */
/* ------------------------------------------------------------------ */

ObjTuple *jaiTupleNew(const Value *items, int count) {
    if (count < 0) count = 0;
    ObjTuple *t = (ObjTuple *)jaiAllocateObject(
        sizeof(ObjTuple) + sizeof(Value) * (size_t)count, OBJ_TUPLE);
    t->count = (uint32_t)count;
    t->hash = 0;                    /* computed lazily by jaiValueHash */
    if (items != NULL && count > 0) {
        memcpy(t->items, items, sizeof(Value) * (size_t)count);
    } else {
        for (int i = 0; i < count; i++) t->items[i] = NULL_VAL;
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
static bool keyHash(Value key, const char *role, uint64_t *hash) {
    if (IS_NULL(key)) {
        return jaiThrow(vm.cTypeError, "a %s cannot be null", role);
    }
    bool ok = true;
    *hash = jaiValueHashFast(key, &ok);
    if (ok) return true;
    if (vm.hasException) return false;      /* a user __hash__ raised */
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

    int i = 0;
    Value k, v;
    while (jaiTableNext(&d->table, &i, &k, &v)) {
        jaiListPush(out, wantValues ? v : k);
    }
    jaiGCPopRoots(2);
    return out;
}

ObjList *jaiDictKeys(ObjDict *d)   { return dictColumn(d, false); }
ObjList *jaiDictValues(ObjDict *d) { return dictColumn(d, true); }

ObjList *jaiDictItems(ObjDict *d) {
    jaiGCPushRoot(OBJ_VAL(d));
    ObjList *out = jaiListNew(d->table.count);
    jaiGCPushRoot(OBJ_VAL(out));

    int i = 0;
    Value k, v;
    while (jaiTableNext(&d->table, &i, &k, &v)) {
        Value pair[2] = {k, v};
        ObjTuple *t = jaiTupleNew(pair, 2);
        jaiListPush(out, OBJ_VAL(t));
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
    if (r->step == 0) return 0;

    /* Unsigned span arithmetic: stop - start can exceed INT64_MAX. */
    uint64_t span, stride;
    if (r->step > 0) {
        if (r->start > r->stop) return 0;
        span = (uint64_t)r->stop - (uint64_t)r->start;
        stride = (uint64_t)r->step;
    } else {
        if (r->start < r->stop) return 0;
        span = (uint64_t)r->start - (uint64_t)r->stop;
        stride = 0u - (uint64_t)r->step;   /* |step|, correct for INT64_MIN */
    }
    if (span == 0) return r->inclusive ? 1 : 0;
    if (r->inclusive) {
        if (span == UINT64_MAX) return INT64_MAX;
        span += 1;
    }

    uint64_t n = (span + stride - 1) / stride;
    return (n > (uint64_t)INT64_MAX) ? INT64_MAX : (int64_t)n;
}

/* ------------------------------------------------------------------ */
/* Functions, closures, natives                                         */
/* ------------------------------------------------------------------ */

ObjFunction *jaiFunctionNew(void) {
    ObjFunction *fn = JAI_ALLOCATE_OBJ(ObjFunction, OBJ_FUNCTION);
    fn->name = NULL;
    fn->qualifiedName = NULL;
    fn->arity = 0;
    fn->defaultCount = 0;
    fn->flags = 0;
    fn->maxSlots = 0;
    fn->upvalueCount = 0;
    fn->paramNames = NULL;
    fn->paramCount = 0;
    fn->exceptions = NULL;
    fn->exceptionCount = 0;
    fn->defaultOffsets = NULL;
    fn->module = NULL;
    fn->owner = NULL;
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
    up->next = NULL;
    return up;
}

ObjUpvalue *jaiUpvalueClosed(Value v) {
    jaiGCPushRoot(v);
    ObjUpvalue *u = JAI_ALLOCATE_OBJ(ObjUpvalue, OBJ_UPVALUE);
    jaiGCPopRoot();
    u->closed = v;
    u->location = &u->closed;
    u->next = NULL;      /* never on vm.openUpvalues: there is nothing to close */
    return u;
}

ObjClosure *jaiClosureNew(ObjFunction *fn) {
    int n = (fn == NULL) ? 0 : (int)fn->upvalueCount;

    pushObjRoot(fn);
    ObjUpvalue **upvalues = NULL;
    if (n > 0) {
        upvalues = JAI_ALLOC(ObjUpvalue *, n);
        for (int i = 0; i < n; i++) upvalues[i] = NULL;
    }
    ObjClosure *c = JAI_ALLOCATE_OBJ(ObjClosure, OBJ_CLOSURE);
    jaiGCPopRoot();

    c->fn = fn;
    c->upvalues = upvalues;
    c->upvalueCount = n;
    return c;
}

ObjNative *jaiNativeNew(JaiNativeFn fn, const char *name, int minArity,
                        int maxArity, const char *const *paramNames) {
    ObjString *interned = jaiStringInternC(name);
    pushObjRoot(interned);
    ObjNative *n = JAI_ALLOCATE_OBJ(ObjNative, OBJ_NATIVE);
    jaiGCPopRoot();

    n->fn = fn;
    n->name = interned;
    n->minArity = (int8_t)(minArity < -128 ? -128 : (minArity > 127 ? 127 : minArity));
    n->maxArity = (int8_t)(maxArity < -128 ? -128 : (maxArity > 127 ? 127 : maxArity));
    n->paramNames = paramNames;
    return n;
}

ObjBound *jaiBoundNew(Value receiver, Value method) {
    jaiGCPushRoot(receiver);
    jaiGCPushRoot(method);
    ObjBound *b = JAI_ALLOCATE_OBJ(ObjBound, OBJ_BOUND);
    jaiGCPopRoots(2);

    b->receiver = receiver;
    b->method = method;
    return b;
}

/* ------------------------------------------------------------------ */
/* Classes                                                              */
/* ------------------------------------------------------------------ */

/* Shape ids are the inline-cache key; 0 means "no shape", so ids start at 1
 * and are never reused. */
static uint32_t nextShapeId = 1;

/* The dunder cache mirrors these method names into fixed ObjClass fields so
 * that operator dispatch never touches a hash table. */
typedef struct {
    const char *name;
    size_t      offset;       /* of the Value field inside ObjClass */
} DunderEntry;

#define DUNDER(field, text) {(text), offsetof(ObjClass, field)}

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

static void setDunderField(ObjClass *c, size_t offset, Value v) {
    void *p = (char *)c + offset;
    *(Value *)p = v;
}

static bool nameIs(const ObjString *name, const char *literal) {
    size_t len = strlen(literal);
    return name != NULL && name->length == len &&
           memcmp(name->chars, literal, len) == 0;
}

ObjClass *jaiClassNew(ObjString *name, ObjClass *superclass) {
    pushObjRoot(name);
    pushObjRoot(superclass);
    ObjClass *c = JAI_ALLOCATE_OBJ(ObjClass, OBJ_CLASS);
    jaiGCPopRoots(2);

    c->name = name;
    c->qualifiedName = name;      /* refined by the code generator if nested */
    c->superclass = superclass;
    c->shapeId = nextShapeId++;
    if (nextShapeId == 0) nextShapeId = 1;   /* 0 is reserved */
    c->fields = NULL;
    c->fieldCount = 0;
    jaiTableInit(&c->methods);
    jaiTableInit(&c->statics);
    jaiTableInit(&c->getters);
    jaiTableInit(&c->setters);
    jaiTableInit(&c->restricted);
    c->traits = NULL;
    c->traitCount = 0;
    c->initializer = NULL_VAL;
    c->isAbstract = false;
    for (int i = 0; i < DUNDER_COUNT; i++) {
        setDunderField(c, kDunders[i].offset, NULL_VAL);
    }
    return c;
}

void jaiClassInherit(ObjClass *sub, ObjClass *super) {
    if (sub == NULL || super == NULL) return;
    sub->superclass = super;

    /* Parent fields keep their slot numbers and come first, so an inline cache
     * compiled against the parent stays valid for instances of the subclass. */
    uint16_t ownCount = sub->fieldCount;
    FieldInfo *own = sub->fields;
    uint32_t wide = (uint32_t)super->fieldCount + ownCount;
    if (wide > UINT16_MAX) {
        jaiThrow(vm.cRuntimeError, "class '%s' would have %u fields, the limit is %u",
                 (sub->name != NULL) ? sub->name->chars : "?", wide, UINT16_MAX);
        return;
    }
    uint16_t total = (uint16_t)wide;
    FieldInfo *merged = NULL;
    if (total > 0) {
        merged = JAI_ALLOC(FieldInfo, total);
        for (uint16_t i = 0; i < super->fieldCount; i++) {
            merged[i] = super->fields[i];
        }
        for (uint16_t i = 0; i < ownCount; i++) {
            merged[super->fieldCount + i] = own[i];
            merged[super->fieldCount + i].slot = (uint16_t)(super->fieldCount + i);
        }
    }
    JAI_FREE_ARRAY(FieldInfo, own, ownCount);
    sub->fields = merged;
    sub->fieldCount = total;

    /* Copy first, so members the subclass declares afterwards overwrite the
     * inherited entries. */
    jaiTableAddAll(&super->methods, &sub->methods);
    jaiTableAddAll(&super->statics, &sub->statics);
    jaiTableAddAll(&super->getters, &sub->getters);
    jaiTableAddAll(&super->setters, &sub->setters);
    /* Visibility travels with the method it describes; an override then
     * rewrites or clears the entry from jaiClassAddMethod. */
    jaiTableAddAll(&super->restricted, &sub->restricted);

    if (IS_NULL(sub->initializer)) sub->initializer = super->initializer;
    jaiClassRefreshDunders(sub);
}

void jaiClassAddMethod(ObjClass *c, ObjString *name, Value method,
                       Visibility vis, uint32_t flags) {
    JaiTable *target;
    if (flags & FN_GETTER)      target = &c->getters;
    else if (flags & FN_SETTER) target = &c->setters;
    else if (flags & FN_STATIC) target = &c->statics;
    else                        target = &c->methods;

    /* A dunder is the object protocol, not part of the class's surface: the
     * VM calls it on behalf of `print`, `==`, `for`, and the rest, from
     * wherever the operator was written. Recording `fn __str__` — which the
     * spec's own example leaves unmarked, i.e. private — as restricted would
     * make printing an instance from outside its class raise. */
    bool isDunder = name != NULL && name->length > 4 &&
                    memcmp(name->chars, "__", 2) == 0 &&
                    memcmp(name->chars + name->length - 2, "__", 2) == 0;

    jaiGCPushRoot(OBJ_VAL(c));
    jaiGCPushRoot(method);
    pushObjRoot(name);
    (void)jaiTableSetInterned(target, name, method);
    if (vis == VIS_PUBLIC || isDunder) {
        /* An override may widen: the inherited entry must not outlive it. */
        if (c->restricted.count > 0) {
            (void)jaiTableDelete(&c->restricted, OBJ_VAL(name));
        }
    } else {
        /* `c` is the declaring class by construction: a class only adds a
         * method it declares, and jaiClassInherit copies the parent's entries
         * — shapeId and all — before any of them run. */
        (void)jaiTableSetInterned(&c->restricted, name,
                                  INT_VAL((int64_t)vis |
                                          ((int64_t)(flags & 0xFFFFu) << 8) |
                                          ((int64_t)c->shapeId << 24)));
    }
    jaiGCPopRoots(3);

    if (target != &c->methods) return;

    if ((flags & FN_INIT) || nameIs(name, "init")) {
        c->initializer = method;
        return;
    }
    for (int i = 0; i < DUNDER_COUNT; i++) {
        if (nameIs(name, kDunders[i].name)) {
            setDunderField(c, kDunders[i].offset, method);
            return;
        }
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
    for (uint16_t i = 0; i < c->fieldCount; i++) {
        const FieldInfo *f = &c->fields[i];
        if (f->name == name || jaiStringEquals(f->name, name)) return f;
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
    ObjTrait *t = JAI_ALLOCATE_OBJ(ObjTrait, OBJ_TRAIT);
    jaiGCPopRoot();

    t->name = name;
    jaiTableInit(&t->required);
    jaiTableInit(&t->defaults);
    t->supers = NULL;
    t->superCount = 0;
    return t;
}

/* ------------------------------------------------------------------ */
/* Instances                                                            */
/* ------------------------------------------------------------------ */

ObjInstance *jaiInstanceNew(ObjClass *klass) {
    uint16_t n = (klass == NULL) ? 0 : klass->fieldCount;

    pushObjRoot(klass);
    ObjInstance *inst = (ObjInstance *)jaiAllocateObject(
        sizeof(ObjInstance) + sizeof(Value) * (size_t)n, OBJ_INSTANCE);
    jaiGCPopRoot();

    inst->klass = klass;
    inst->fieldCount = n;
    for (uint16_t i = 0; i < n; i++) inst->fields[i] = NULL_VAL;
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
    e->variants = NULL;
    e->variantCount = 0;
    jaiTableInit(&e->methods);
    return e;
}

ObjEnumCtor *jaiEnumCtorNew(ObjEnum *e, uint16_t tag) {
    pushObjRoot(e);
    ObjEnumCtor *c = JAI_ALLOCATE_OBJ(ObjEnumCtor, OBJ_ENUM_CTOR);
    jaiGCPopRoot();

    c->type = e;
    c->tag = tag;
    return c;
}

ObjEnumVal *jaiEnumValNew(ObjEnum *e, uint16_t tag, const Value *payload,
                          int count) {
    if (count < 0) count = 0;
    if (count > 255) count = 255;

    pushObjRoot(e);
    ObjEnumVal *ev = (ObjEnumVal *)jaiAllocateObject(
        sizeof(ObjEnumVal) + sizeof(Value) * (size_t)count, OBJ_ENUM_VAL);
    jaiGCPopRoot();

    ev->type = e;
    ev->tag = tag;
    ev->count = (uint8_t)count;
    for (int i = 0; i < count; i++) {
        ev->payload[i] = (payload != NULL) ? payload[i] : NULL_VAL;
    }
    return ev;
}

/* ------------------------------------------------------------------ */
/* Modules                                                              */
/* ------------------------------------------------------------------ */

ObjModule *jaiModuleNew(ObjString *name, ObjString *path) {
    pushObjRoot(name);
    pushObjRoot(path);
    ObjModule *m = JAI_ALLOCATE_OBJ(ObjModule, OBJ_MODULE);
    jaiGCPopRoots(2);

    m->name = name;
    m->path = path;
    jaiTableInit(&m->globals);
    jaiTableInit(&m->exports);
    m->version = 0;
    m->state = MOD_UNLOADED;
    m->body = NULL;
    m->sourceFileId = -1;
    return m;
}

bool jaiModuleGet(ObjModule *m, ObjString *name, Value *out) {
    return jaiTableGetInterned(&m->globals, name, out);
}

void jaiModuleSet(ObjModule *m, ObjString *name, Value v) {
    jaiGCPushRoot(OBJ_VAL(m));
    jaiGCPushRoot(v);
    (void)jaiTableSetInterned(&m->globals, name, v);
    jaiGCPopRoots(2);
    /* Every mutation invalidates the global inline caches keyed on version. */
    m->version++;
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
static ObjString *iterName(void) {
    return vm.strIter != NULL ? vm.strIter : jaiStringInternC("__iter__");
}

static ObjString *nextName(void) {
    return vm.strNext != NULL ? vm.strNext : jaiStringInternC("__next__");
}

/* The trait spellings of the same two methods (std.core `Iterable`/`Iterator`).
 * They are ordinary names, so there is no cached intern for them. */
static ObjString *traitIterName(void) { return jaiStringInternC("iter"); }
static ObjString *traitNextName(void) { return jaiStringInternC("next"); }

/* True when instances of `v`'s class answer `name`. Inherited methods are
 * copied down at class creation, so one table lookup is the whole answer. */
static bool instanceHasMethod(Value v, ObjString *name) {
    if (!IS_INSTANCE(v) || name == NULL) return false;
    Value ignored;
    return jaiTableGetInterned(&AS_INSTANCE(v)->klass->methods, name, &ignored);
}

ObjIter *jaiIterNew(IterKind kind, Value source) {
    jaiGCPushRoot(source);
    ObjIter *it = JAI_ALLOCATE_OBJ(ObjIter, OBJ_ITER);
    jaiGCPopRoot();

    it->kind = kind;
    it->source = source;
    it->index = 0;
    it->limit = 0;
    it->version = 0;

    switch (kind) {
    case ITER_LIST:
        if (IS_LIST(source)) {
            /* Two witnesses: the count catches a resize, the version catches an
             * in-place store the count cannot see (`xs[i] = v`, sort, reverse). */
            it->limit = AS_LIST(source)->count;
            it->version = AS_LIST(source)->version;
        }
        break;
    /* Tuples and strings are immutable, so neither carries a version and
     * neither needs a check; the limit is a bound, not a witness. */
    case ITER_TUPLE:
        if (IS_TUPLE(source)) it->limit = (int64_t)AS_TUPLE(source)->count;
        break;
    case ITER_STRING:
        if (IS_STRING(source)) {
            it->limit = (int64_t)AS_STRING(source)->length;   /* byte cursor */
        }
        break;
    case ITER_DICT_KEYS:
    case ITER_DICT_ITEMS:
        if (IS_DICT(source)) {
            it->limit = AS_DICT(source)->table.count;
            it->version = AS_DICT(source)->table.version;
        }
        break;
    case ITER_SET:
        if (IS_SET(source)) {
            it->limit = AS_SET(source)->table.count;
            it->version = AS_SET(source)->table.version;
        }
        break;
    case ITER_RANGE:
        /* A range is frozen at construction; nothing can invalidate the bound. */
        if (IS_RANGE(source)) it->limit = jaiRangeLength(AS_RANGE(source));
        break;
    /* A user iterator owns its own traversal state, so interference is its
     * own business and there is nothing here to witness. */
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
static bool iterMutated(bool resized) {
    return jaiThrow(vm.cRuntimeError, resized
                        ? "container changed size during iteration"
                        : "container was modified during iteration");
}

/* True when the pending exception is a StopIteration, i.e. an ordinary end of
 * a user-defined iterator rather than a failure. */
static bool pendingIsStopIteration(void) {
    if (!vm.hasException) return false;
    if (vm.cStopIteration == NULL) return false;
    Value e = vm.pendingException;
    if (IS_INSTANCE(e)) {
        return jaiClassIsSubclassOf(AS_INSTANCE(e)->klass, vm.cStopIteration);
    }
    if (IS_CLASS(e)) return jaiClassIsSubclassOf(AS_CLASS(e), vm.cStopIteration);
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
        ObjList *l = AS_LIST(it->source);
        if (l->version != it->version) {
            return iterMutated((int64_t)l->count != it->limit);
        }
        if (it->index >= it->limit) return false;
        *out = l->items[it->index++];
        return true;
    }
    case ITER_TUPLE: {
        ObjTuple *t = AS_TUPLE(it->source);
        if (it->index >= it->limit) return false;
        *out = t->items[it->index++];
        return true;
    }
    case ITER_STRING: {
        ObjString *s = AS_STRING(it->source);
        if (it->index >= it->limit) return false;
        const char *p = s->chars + it->index;
        int len = 1;
        (void)jaiUtf8Decode(p, s->chars + s->length, &len);
        if (it->index + len > it->limit) len = (int)(it->limit - it->index);
        jaiGCPushRoot(OBJ_VAL(it));            /* keeps `s` (and `p`) alive */
        ObjString *scalar = jaiStringNew(p, (size_t)len);
        jaiGCPopRoot();
        if (scalar == NULL) return false;      /* exception already pending */
        it->index += len;
        *out = OBJ_VAL(scalar);
        return true;
    }
    case ITER_DICT_KEYS:
    case ITER_DICT_ITEMS: {
        ObjDict *d = AS_DICT(it->source);
        if (d->table.version != it->version) {
            return iterMutated((int64_t)d->table.count != it->limit);
        }
        int slot = (int)it->index;
        Value k, v;
        if (!jaiTableNext(&d->table, &slot, &k, &v)) {
            it->index = slot;
            return false;
        }
        it->index = slot;
        if (it->kind == ITER_DICT_KEYS) {
            *out = k;
        } else {
            Value pair[2] = {k, v};
            jaiGCPushRoot(k);        /* the pair is unreachable until stored */
            jaiGCPushRoot(v);
            *out = OBJ_VAL(jaiTupleNew(pair, 2));
            jaiGCPopRoots(2);
        }
        return true;
    }
    case ITER_SET: {
        ObjSet *s = AS_SET(it->source);
        if (s->table.version != it->version) {
            return iterMutated((int64_t)s->table.count != it->limit);
        }
        int slot = (int)it->index;
        Value k, v;
        if (!jaiTableNext(&s->table, &slot, &k, &v)) {
            it->index = slot;
            return false;
        }
        it->index = slot;
        *out = k;
        return true;
    }
    case ITER_RANGE: {
        ObjRange *r = AS_RANGE(it->source);
        if (it->index >= it->limit) return false;
        *out = INT_VAL(r->start + it->index * r->step);
        it->index++;
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
            *out = v;                       /* an iterator is its own iterator */
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
            /* Two spellings reach the same place: the `__iter__` dunder of
             * spec §7.1 and the `iter` of std.core's `trait Iterable`, which
             * spec §5 requires of a user type in a `for`. */
            ObjString *method = instanceHasMethod(v, iterName()) ? iterName()
                                                                 : traitIterName();
            Value result;
            if (!jaiInvokeMethod(v, method, 0, NULL, &result)) {
                if (!vm.hasException) {
                    jaiThrow(vm.cTypeError, "'%s' object is not iterable",
                             jaiTypeNameStatic(v));
                }
                return false;
            }
            if (IS_ITER(result)) {
                *out = result;
                return true;
            }
            if (IS_INSTANCE(result)) {
                /* Whichever `next` the iterator answers to decides how the end
                 * is signalled: StopIteration for the dunder, null for the
                 * trait. Preferring the dunder keeps a class that has both
                 * behaving as it did before the trait was understood. */
                IterKind kind = instanceHasMethod(result, nextName()) ? ITER_USER
                                                                      : ITER_TRAIT;
                *out = OBJ_VAL(jaiIterNew(kind, result));
                return true;
            }
            return jaiGetIter(result, out);   /* terminates: not an instance */
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

ObjFile *jaiFileNew(FILE *handle, ObjString *path, const char *mode) {
    pushObjRoot(path);
    ObjFile *f = JAI_ALLOCATE_OBJ(ObjFile, OBJ_FILE);
    jaiGCPopRoot();

    const char *m = (mode != NULL) ? mode : "r";
    bool update = strchr(m, '+') != NULL;
    f->handle = handle;
    f->path = path;
    f->readable = update || strchr(m, 'r') != NULL;
    f->writable = update || strchr(m, 'w') != NULL || strchr(m, 'a') != NULL ||
                  strchr(m, 'x') != NULL;
    f->binary = strchr(m, 'b') != NULL;
    f->closed = (handle == NULL);
    return f;
}
