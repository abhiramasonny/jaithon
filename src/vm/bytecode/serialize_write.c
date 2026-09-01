/* serialize_write.c — the .jaic writer: the string table, the per-record
 * constant pool, and the container jaiSerializeModule emits. The format and
 * the rules the writer obeys are documented at the top of serialize.c. */

#include "vm/bytecode/serialize_internal.h"

/* ------------------------------------------------------------------ */
/* Writing: the constant pool of one function record                    */
/* ------------------------------------------------------------------ */

/* A record's pool is the chunk's pool plus whatever the record needs to name
 * by index and the chunk did not already hold: the function's own name, its
 * parameter names, and the names a class spec refers to. Appended entries land
 * after the chunk's own constants, so every existing u24 operand in the code
 * keeps pointing at the same value. */
typedef struct {
    const ValueArray *pool;
    ValueArray        extra;
} PoolWriter;

static uint32_t poolTotal(const PoolWriter *w) {
    return (uint32_t)w->pool->count + (uint32_t)w->extra.count;
}

static Value poolAt(const PoolWriter *w, uint32_t i) {
    if (i < (uint32_t)w->pool->count) return w->pool->data[i];
    return w->extra.data[i - (uint32_t)w->pool->count];
}

/* Exact identity for the values a record names by index: strings compare by
 * content (two equal literals must share one entry), everything else by
 * pointer. Never jaiValuesEqual, which runs user code and conflates int with
 * float. */
static bool poolSame(Value a, Value b) {
    if (!IS_OBJ(a) || !IS_OBJ(b)) return false;
    Obj *x = AS_OBJ(a), *y = AS_OBJ(b);
    if (x == y) return true;
    if (x == NULL || y == NULL || x->type != y->type) return false;
    if (x->type != OBJ_STRING) return false;

    const ObjString *sa = (const ObjString *)x, *sb = (const ObjString *)y;
    return sa->length == sb->length && sa->hash == sb->hash &&
           memcmp(sa->chars, sb->chars, sa->length) == 0;
}

static bool poolIndexOf(PoolWriter *w, Value v, uint32_t *out) {
    uint32_t total = poolTotal(w);
    for (uint32_t i = 0; i < total; i++) {
        if (poolSame(poolAt(w, i), v)) {
            *out = i;
            return true;
        }
    }
    if (total >= JAIC_CONST_MAX) return false;

    jaiValueArrayPush(&w->extra, v);
    *out = total;
    return true;
}

static bool poolIndexOfString(PoolWriter *w, ObjString *s, uint32_t *out) {
    if (s == NULL) return false;
    return poolIndexOf(w, OBJ_VAL(s), out);
}

/* ------------------------------------------------------------------ */
/* Writing: constants and function records                              */
/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/* Writing: the module string table                                     */
/* ------------------------------------------------------------------ */

/* Every string constant in the module, once, in first-use order.
 *
 * Strings are interned, so identity is equality and the dedup index can be a
 * pointer-keyed table. Two things pay for this: 62.5% of a .jaic's string
 * payload was the same text repeated across nested function pools (73.8% in the
 * seed), and interning at load drops from one call per occurrence -- 9,988
 * across lib -- to one per distinct text, 2,327. */
typedef struct {
    const ObjString **items;
    uint32_t          count;
    uint32_t          capacity;
    JaiTable          index;   /* interned string -> INT_VAL(position) */
} StrTab;

static void strTabInit(StrTab *st) {
    st->items = NULL;
    st->count = 0;
    st->capacity = 0;
    jaiTableInit(&st->index);
}

static void strTabFree(StrTab *st) {
    JAI_FREE_ARRAY(const ObjString *, st->items, st->capacity);
    jaiTableFree(&st->index);
    st->items = NULL;
    st->count = 0;
    st->capacity = 0;
}

static bool strTabIntern(StrTab *st, const ObjString *s, uint32_t *out) {
    if (s == NULL) return false;

    Value existing;
    if (jaiTableGetInterned(&st->index, (ObjString *)s, &existing) &&
        IS_INT(existing)) {
        *out = (uint32_t)AS_INT(existing);
        return true;
    }
    if (st->count >= JAIC_CONST_MAX) return false;

    if (st->capacity < st->count + 1) {
        uint32_t oldCapacity = st->capacity;
        st->capacity = (uint32_t)JAI_GROW_CAP((int)oldCapacity);
        st->items = JAI_GROW_ARRAY(const ObjString *, st->items, oldCapacity,
                                   st->capacity);
    }
    st->items[st->count] = s;
    *out = st->count;
    (void)jaiTableSetInterned(&st->index, (ObjString *)s, INT_VAL(st->count));
    st->count++;
    return true;
}

static void bufWriteUleb(JaiBuf *b, uint64_t v) {
    do {
        uint8_t byte = (uint8_t)(v & 0x7Fu);
        v >>= 7;
        if (v != 0) byte |= 0x80u;
        jaiBufPush(b, byte);
    } while (v != 0);
}

/* `count`, then each string as {uleb128 length, bytes}. Varint lengths because
 * 94.9% of strings are under 16 bytes: a flat u32 length cost 38,672 bytes
 * across lib to carry 75,717 bytes of payload. */
static void writeStrTab(JaiBuf *b, const StrTab *st) {
    bufWriteUleb(b, st->count);
    for (uint32_t i = 0; i < st->count; i++) {
        const ObjString *s = st->items[i];
        bufWriteUleb(b, s->length);
        jaiBufAppend(b, s->chars, s->length);
    }
}

/* When `sidecar` is non-NULL the line table is STRIPPED from the image and
 * written there instead: the .jaic gets a zero byte-length and the stream goes
 * to the .jaid beside it. The two stay in step because both are produced by
 * this same traversal, in this same order, one record per function -- including
 * functions with no line entries at all, which write a zero length rather than
 * nothing. */
static bool writeFunction(JaiBuf *b, JaiBuf *sidecar, StrTab *st,
                          const ObjFunction *fn, int depth);
static bool writeConstant(JaiBuf *b, JaiBuf *sidecar, PoolWriter *w, StrTab *st,
                          Value v, int depth);

/* §4. Refusing a value the format cannot express is always allowed: the
 * caller's fallback is to skip the cache and recompile. */
static bool writeConstant(JaiBuf *b, JaiBuf *sidecar, PoolWriter *w, StrTab *st,
                          Value v, int depth) {
    if (depth > JAIC_MAX_DEPTH) return false;

    switch (jaiValueType(v)) {
    case VAL_NULL:
        jaiBufPush(b, K_NULL);
        return true;
    case VAL_BOOL:
        jaiBufPush(b, K_BOOL);
        jaiBufPush(b, AS_BOOL(v) ? 1u : 0u);
        return true;
    case VAL_INT:
        jaiBufPush(b, K_INT);
        jaiBufWriteU64(b, (uint64_t)AS_INT(v));
        return true;
    case VAL_FLOAT:
        jaiBufPush(b, K_FLOAT);
        jaiBufWriteF64(b, AS_FLOAT(v));
        return true;
    case VAL_OBJ:
        break;
    }

    Obj *o = AS_OBJ(v);
    if (o == NULL) return false;

    switch (o->type) {
    case OBJ_STRING: {
        /* The bytes live once, in the module string table; the pool entry is
         * an index into it. */
        uint32_t index = 0;
        if (!strTabIntern(st, (const ObjString *)o, &index)) return false;
        jaiBufPush(b, K_STRREF);
        bufWriteUleb(b, index);
        return true;
    }
    case OBJ_BYTES: {
        const ObjBytes *bytes = (const ObjBytes *)o;
        jaiBufPush(b, K_BYTES);
        jaiBufWriteU32(b, bytes->length);
        jaiBufAppend(b, bytes->data, bytes->length);
        return true;
    }
    case OBJ_FUNCTION:
        jaiBufPush(b, K_FUNC);
        return writeFunction(b, sidecar, st, (const ObjFunction *)o, depth + 1);
    case OBJ_TUPLE: {
        const ObjTuple *t = (const ObjTuple *)o;
        jaiBufPush(b, K_TUPLE);
        jaiBufWriteU32(b, t->count);
        for (uint32_t i = 0; i < t->count; i++) {
            if (!writeConstant(b, sidecar, w, st, t->items[i], depth + 1)) {
                return false;
            }
        }
        return true;
    }
    default:
        /* Lists, dicts, closures, instances: not constants. Refusing beats
         * writing something the loader would have to guess at. */
        return false;
    }
}

/* FN_INIT is bit 8 and §5's flags field is one byte, so the bit is not stored;
 * it is recomputed on load from "a method named init", which is the same rule
 * jaiClassAddMethod already uses to pick a class's initialiser — a method
 * named init is one whether or not the bit is set. Only the case the name
 * cannot recover, a function marked FN_INIT under some other name, loses
 * information, and such a function is not written. */
static bool initFlagIsRecoverable(const ObjFunction *fn) {
    bool marked  = (fn->flags & FN_INIT) != 0;
    bool derived = (fn->flags & FN_METHOD) != 0 && fn->name != NULL &&
                   fn->name->length == 4 &&
                   memcmp(fn->name->chars, "init", 4) == 0;
    return derived || !marked;
}

/* §5. */
static bool writeFunction(JaiBuf *b, JaiBuf *sidecar, StrTab *st,
                          const ObjFunction *fn, int depth) {
    if (fn == NULL || depth > JAIC_MAX_DEPTH) return false;
    if (fn->flags & ~JAIC_KNOWN_FN_FLAGS) return false;
    if (!initFlagIsRecoverable(fn)) return false;
    if (fn->chunk.count < 0 || fn->chunk.lineStreamLen < 0) return false;
    /* A count without its array is a broken object, not a cacheable one. */
    if ((fn->chunk.count > 0 && fn->chunk.code == NULL) ||
        (fn->chunk.lineStreamLen > 0 && fn->chunk.lineStream == NULL) ||
        (fn->exceptionCount > 0 && fn->exceptions == NULL) ||
        (fn->defaultCount > 0 && fn->defaultOffsets == NULL) ||
        (fn->paramCount > 0 && fn->paramNames == NULL)) {
        return false;
    }

    PoolWriter w = { &fn->chunk.constants, { NULL, 0, 0 } };
    JaiBuf constants;
    uint32_t *paramIndices = NULL;
    uint32_t nameIndex = JAIC_NO_NAME;
    bool ok = true;

    jaiBufInit(&constants);

    /* Every index this record names must be resolved before the pool is
     * serialised, since resolving may append to it. Class specs append while
     * the pool is being written, which the loop below picks up. */
    if (fn->name != NULL && !poolIndexOfString(&w, fn->name, &nameIndex)) {
        ok = false;
    }
    if (ok && fn->paramCount > 0) {
        paramIndices = JAI_ALLOC(uint32_t, fn->paramCount);
        for (uint16_t i = 0; i < fn->paramCount && ok; i++) {
            ObjString *name = fn->paramNames[i];
            if (name == NULL) {
                paramIndices[i] = JAIC_NO_NAME;
            } else if (!poolIndexOfString(&w, name, &paramIndices[i])) {
                ok = false;
            }
        }
    }

    for (uint32_t i = 0; ok && i < poolTotal(&w); i++) {
        ok = writeConstant(&constants, sidecar, &w, st, poolAt(&w, i),
                           depth + 1);
    }

    uint32_t constCount = poolTotal(&w);
    if (ok && constCount >= JAIC_CONST_MAX) ok = false;

    if (ok) {
        jaiBufWriteU32(b, nameIndex);
        jaiBufPush(b, fn->arity);
        jaiBufWriteU16(b, (uint16_t)(fn->flags & 0xFFFFu));
        jaiBufWriteU16(b, fn->maxSlots);
        jaiBufWriteU16(b, fn->upvalueCount);

        jaiBufWriteU32(b, (uint32_t)fn->chunk.count);
        jaiBufAppend(b, fn->chunk.code, (size_t)fn->chunk.count);

        jaiBufWriteU32(b, constCount);
        jaiBufAppend(b, constants.data, constants.count);

        /* The length stays a BYTE count, which is what lets a reader that does
         * not understand the section skip it -- that property is why changing
         * what is inside it is a contained change. The bytes are now the LTV1
         * stream the chunk already holds, so writing the table is a memcpy. */
        if (sidecar != NULL) {
            /* Stripped: the image says "no spans", and the stream goes to the
             * sidecar. A release binary that ships without its .jaid loses
             * source spans in tracebacks and nothing else. */
            jaiBufWriteU32(b, 0);
            jaiBufWriteU32(sidecar, (uint32_t)fn->chunk.lineStreamLen);
            jaiBufAppend(sidecar, fn->chunk.lineStream,
                         (size_t)fn->chunk.lineStreamLen);
        } else {
            jaiBufWriteU32(b, (uint32_t)fn->chunk.lineStreamLen);
            jaiBufAppend(b, fn->chunk.lineStream,
                         (size_t)fn->chunk.lineStreamLen);
        }

        jaiBufWriteU16(b, fn->exceptionCount);
        for (uint16_t i = 0; i < fn->exceptionCount; i++) {
            const ExceptionEntry *e = &fn->exceptions[i];
            jaiBufWriteU32(b, e->start);
            jaiBufWriteU32(b, e->end);
            jaiBufWriteU32(b, e->handler);
            jaiBufWriteU32(b, e->typeConst);
        }

        jaiBufWriteU16(b, fn->defaultCount);
        for (uint8_t i = 0; i < fn->defaultCount; i++) {
            jaiBufWriteU32(b, fn->defaultOffsets[i]);
        }

        jaiBufWriteU16(b, fn->paramCount);
        for (uint16_t i = 0; i < fn->paramCount; i++) {
            jaiBufWriteU32(b, paramIndices[i]);
        }
    }

    if (paramIndices != NULL) JAI_FREE_ARRAY(uint32_t, paramIndices, fn->paramCount);
    jaiBufFree(&constants);
    jaiValueArrayFree(&w.extra);
    return ok;
}

/* ------------------------------------------------------------------ */
/* Writing: the container                                               */
/* ------------------------------------------------------------------ */

/* Last-modified time of the module's source, or 0 when it cannot be read.
 * mtime is informational: cache validity is decided by srcHash and crc32. */
static int64_t sourceMtime(const ObjModule *module) {
    struct stat st;
    if (module == NULL || module->path == NULL) return 0;
    if (stat(module->path->chars, &st) != 0) return 0;
    return (int64_t)st.st_mtime;
}

/* The module-level pool (§7) holds the export and import names; the module
 * body's own constants live in its function record. */
static bool writeModulePool(JaiBuf *b, StrTab *st, ObjModule *module,
                            uint32_t *outCount) {
    uint32_t count = 0;
    if (module != NULL) {
        int slot = 0;
        Value key, value;
        while (jaiTableNext(&module->exports, &slot, &key, &value)) {
            if (IS_STRING(key)) count++;
        }
    }
    if (count >= JAIC_CONST_MAX) return false;

    jaiBufWriteU32(b, count);
    if (module != NULL) {
        int slot = 0;
        Value key, value;
        while (jaiTableNext(&module->exports, &slot, &key, &value)) {
            if (!IS_STRING(key)) continue;
            uint32_t index = 0;
            if (!strTabIntern(st, AS_STRING(key), &index)) return false;
            jaiBufPush(b, K_STRREF);
            bufWriteUleb(b, index);
        }
    }
    *outCount = count;
    return true;
}

uint8_t *jaiSerializeModule(ObjModule *module, ObjFunction *body,
                            uint64_t sourceHash, uint32_t flags,
                            size_t *outSize, uint8_t **outSidecar,
                            size_t *outSidecarSize) {
    if (outSize != NULL) *outSize = 0;
    if (outSidecar != NULL) *outSidecar = NULL;
    if (outSidecarSize != NULL) *outSidecarSize = 0;
    if (body == NULL || flags > 0xFFFFu) return NULL;

    /* The line table is 37% of what a .jaic still weighs after LTV1, and a
     * release build has no use for it until something throws. Absent
     * JAIC_FLAG_DEBUG it goes to a .jaid beside the image, which a release
     * install can simply not ship. */
    bool strip = (flags & JAIC_FLAG_DEBUG) == 0 && outSidecar != NULL;
    JaiBuf sidecarBuf;
    jaiBufInit(&sidecarBuf);
    JaiBuf *sidecar = strip ? &sidecarBuf : NULL;

    JaiBuf b;
    jaiBufInit(&b);
    jaiBufAppend(&b, JAIC_MAGIC, 4);
    jaiBufWriteU16(&b, JAIC_VERSION);
    jaiBufWriteU16(&b, (uint16_t)flags);
    jaiBufWriteU32(&b, JAI_COMPILER_VERSION);
    jaiBufWriteU32(&b, JAI_BUILD_ID);
    jaiBufWriteU64(&b, sourceHash);

    const ObjString *path = module != NULL ? module->path : NULL;
    uint32_t pathLen = path != NULL ? path->length : 0;
    if (pathLen > UINT16_MAX) {
        jaiBufFree(&b);
        return NULL;
    }
    jaiBufWriteU16(&b, (uint16_t)pathLen);
    if (pathLen > 0) jaiBufAppend(&b, path->chars, pathLen);
    jaiBufWriteU64(&b, (uint64_t)sourceMtime(module));

    /* The body is built into its own buffer first, because the string table has
     * to be complete before it can be written and it is only complete once
     * every nested function record has been walked. The table then goes ahead
     * of the body, so the reader has it before the first K_STRREF. */
    StrTab st;
    strTabInit(&st);
    JaiBuf body_buf;
    jaiBufInit(&body_buf);

    uint32_t exportCount = 0;
    bool ok = writeModulePool(&body_buf, &st, module, &exportCount);
    if (ok) ok = writeFunction(&body_buf, sidecar, &st, body, 0);

    if (ok) {
        jaiBufWriteU32(&body_buf, exportCount);
        for (uint32_t i = 0; i < exportCount; i++) {
            /* the pool holds exactly the export names */
            jaiBufWriteU32(&body_buf, i);
        }
        /* ObjModule keeps no import list; the body's OP_IMPORT instructions
         * are what actually import. The section stays for format conformance. */
        jaiBufWriteU32(&body_buf, 0);
    }

    if (ok) {
        writeStrTab(&b, &st);
        jaiBufAppend(&b, body_buf.data, body_buf.count);
    }

    jaiBufFree(&body_buf);
    strTabFree(&st);

    if (!ok) {
        jaiBufFree(&b);
        jaiBufFree(&sidecarBuf);
        return NULL;
    }

    if (strip && sidecarBuf.count > 0) {
        /* Its own magic and checksum: a .jaid that does not match its .jaic is
         * ignored, never trusted to line up with the wrong image. srcHash is
         * what ties them together. */
        JaiBuf sc;
        jaiBufInit(&sc);
        jaiBufAppend(&sc, JAID_MAGIC, 4);
        jaiBufWriteU16(&sc, JAID_VERSION);
        jaiBufWriteU16(&sc, 0);
        jaiBufWriteU64(&sc, sourceHash);
        jaiBufAppend(&sc, sidecarBuf.data, sidecarBuf.count);
        jaiBufWriteU32(&sc, jaiCrc32(sc.data, sc.count));
        if (outSidecar != NULL) *outSidecar = sc.data;
        if (outSidecarSize != NULL) *outSidecarSize = sc.count;
    }
    jaiBufFree(&sidecarBuf);

    jaiBufWriteU32(&b, jaiCrc32(b.data, b.count));

    size_t size = b.count;
    uint8_t *data = b.data;
    if (b.capacity > size) data = (uint8_t *)jaiRealloc(b.data, b.capacity, size);
    jaiBufInit(&b);

    if (outSize != NULL) *outSize = size;
    return data;
}
