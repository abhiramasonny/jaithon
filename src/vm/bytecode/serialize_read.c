/* serialize_read.c — the .jaic reader: the bounds-checked cursor, the constant
 * and function records, and the container entry points. The format and the
 * rules the reader obeys are documented at the top of serialize.c. */

#include "vm/bytecode/serialize_internal.h"

/* ------------------------------------------------------------------ */
/* Reading: bounds-checked cursor                                       */
/* ------------------------------------------------------------------ */

typedef struct Cursor_ {
    const uint8_t *data;
    size_t         size;
    size_t         pos;
    bool           bad;   /* sticky: set by the first read that ran short */
    /* The container version this image declares. Carried on the cursor rather
     * than through every signature because only one section reads it -- the
     * line table, whose encoding changed at JAIC_VERSION_LTV1 -- and threading
     * it through readConstant/readTuple/readFunction would touch six
     * signatures to reach one branch. */
    uint16_t       version;
    /* The module string table, from JAIC_VERSION_STRTAB on: every K_STRREF is
     * an index into it. Interned once here rather than once per occurrence --
     * across lib that is 2,327 interns instead of 9,988. Owned by deserialize,
     * which frees the array (not the strings, which the intern table owns). */
    Value         *strings;
    uint32_t       stringCount;
    /* The .jaid's line-table stream, when the image was written stripped. Read
     * in lockstep with the records, because both were produced by the same
     * traversal in the same order. NULL means the spans are inline (a debug
     * image) or simply absent (a release install without its sidecar). */
    struct Cursor_ *sidecar;
} Cursor;

static size_t curLeft(const Cursor *c) {
    if (c->bad || c->pos > c->size) return 0;
    return c->size - c->pos;
}

/* Hands back `n` bytes at the cursor and advances, or marks the cursor bad.
 * Every other read is built on this one, so no read can escape the buffer. */
static bool curTake(Cursor *c, size_t n, const uint8_t **out) {
    if (c->bad || n > curLeft(c)) {
        c->bad = true;
        return false;
    }
    if (out != NULL) *out = c->data + c->pos;
    c->pos += n;
    return true;
}

static bool curSkip(Cursor *c, size_t n) { return curTake(c, n, NULL); }

/* True when `count` records of `each` bytes can still fit. Counts come from
 * the file, so this is what stops a corrupt length from sizing an allocation. */
static bool curFits(const Cursor *c, uint64_t count, uint64_t each) {
    return !c->bad && count <= (uint64_t)curLeft(c) / (each > 0 ? each : 1);
}

static uint8_t curU8(Cursor *c) {
    const uint8_t *p;
    if (!curTake(c, 1, &p)) return 0;
    return p[0];
}

static uint16_t curU16(Cursor *c) {
    const uint8_t *p;
    if (!curTake(c, 2, &p)) return 0;
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static uint32_t curU32(Cursor *c) {
    const uint8_t *p;
    if (!curTake(c, 4, &p)) return 0;
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint64_t curU64(Cursor *c) {
    const uint8_t *p;
    if (!curTake(c, 8, &p)) return 0;
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)p[i] << (8 * i);
    return v;
}

static double curF64(Cursor *c) {
    uint64_t bits = curU64(c);
    double d;
    memcpy(&d, &bits, sizeof d);   /* IEEE-754 bit pattern, as written */
    return d;
}

/* LEB128, bounded by the cursor. A malformed run refuses the image rather than
 * spinning: 10 bytes is the most a 64-bit value can occupy. */
static bool curUleb(Cursor *c, uint64_t *out) {
    uint64_t v = 0;
    int shift = 0;
    for (int i = 0; i < 10; i++) {
        const uint8_t *p = NULL;
        if (!curTake(c, 1, &p)) return false;
        if (shift < 64) v |= (uint64_t)(p[0] & 0x7Fu) << shift;
        shift += 7;
        if ((p[0] & 0x80u) == 0) {
            *out = v;
            return true;
        }
    }
    c->bad = true;
    return false;
}

/* The module string table: count, then {uleb128 length, bytes} each. Interning
 * happens exactly once per distinct text, here.
 *
 * The array is a GC ROOT RANGE for the whole of the rest of the load. The
 * intern table's references are weak (spec §10), so an interned string reachable
 * only from this C array would be swept by the first collection the remaining
 * reads trigger -- and they allocate a function object per record. Pushing the
 * range up front, pre-filled with NULL_VAL so a collection mid-fill marks a
 * well-formed range, is what keeps them alive until the pools point at them.
 * The caller pops it. */
static bool readStrTab(Cursor *c) {
    uint64_t count = 0;
    if (!curUleb(c, &count)) return false;
    if (count >= JAIC_CONST_MAX) return false;
    if (count == 0) return true;
    /* One byte minimum per entry (a zero-length string is one length byte), so
     * a corrupt count cannot size an allocation larger than the file. */
    if (!curFits(c, count, 1)) return false;

    c->strings = JAI_ALLOC(Value, (size_t)count);
    for (uint64_t i = 0; i < count; i++) c->strings[i] = NULL_VAL;
    c->stringCount = (uint32_t)count;
    jaiGCPushRootRange(c->strings, (int)count);

    for (uint64_t i = 0; i < count; i++) {
        uint64_t len = 0;
        if (!curUleb(c, &len)) return false;
        if (len > curLeft(c)) return false;
        const uint8_t *p = NULL;
        if (!curTake(c, (size_t)len, &p)) return false;
        ObjString *s = jaiStringIntern((const char *)p, (uint32_t)len);
        if (s == NULL) return false;
        c->strings[i] = OBJ_VAL(s);
    }
    return true;
}

uint32_t readU32At(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

/* ------------------------------------------------------------------ */
/* Reading: constants                                                   */
/* ------------------------------------------------------------------ */

static ObjFunction *readFunction(Cursor *c, ObjModule *module, int depth);

static ObjString *poolString(const ValueArray *pool, uint32_t index) {
    if (index >= (uint32_t)pool->count) return NULL;
    Value v = pool->data[index];
    return IS_STRING(v) ? AS_STRING(v) : NULL;
}

/* Reads a length-prefixed byte run, capping the length against what is left in
 * the buffer before anything is allocated. */
static bool readBlob(Cursor *c, const uint8_t **out, uint32_t *outLen) {
    uint32_t len = curU32(c);
    if (c->bad || len > curLeft(c)) {
        c->bad = true;
        return false;
    }
    const uint8_t *p = NULL;
    if (len > 0 && !curTake(c, len, &p)) return false;
    *out = p;
    *outLen = len;
    return true;
}

static bool readConstant(Cursor *c, ObjModule *module, Value *out, int depth);

static bool readTuple(Cursor *c, ObjModule *module, Value *out, int depth) {
    uint32_t count = curU32(c);
    if (c->bad || count > JAIC_CONST_MAX || !curFits(c, count, 1) ||
        count > (uint32_t)INT_MAX) {
        return false;
    }

    Value *items = count > 0 ? JAI_ALLOC(Value, count) : NULL;
    int rooted = 0;
    bool ok = true;

    for (uint32_t i = 0; i < count && ok; i++) {
        Value v;
        ok = readConstant(c, module, &v, depth + 1);
        if (!ok) break;
        items[i] = v;
        /* Held only in `items`, which the collector cannot see. */
        jaiGCPushRoot(v);
        rooted++;
    }

    ObjTuple *t = ok ? jaiTupleNew(items, (int)count) : NULL;
    jaiGCPopRoots(rooted);
    if (items != NULL) JAI_FREE_ARRAY(Value, items, count);

    if (!ok || t == NULL) return false;
    *out = OBJ_VAL(t);
    return true;
}

static bool readConstant(Cursor *c, ObjModule *module, Value *out, int depth) {
    if (depth > JAIC_MAX_DEPTH) {
        c->bad = true;
        return false;
    }

    uint8_t tag = curU8(c);
    if (c->bad) return false;

    switch (tag) {
    case K_NULL:
        *out = NULL_VAL;
        return true;
    case K_BOOL: {
        uint8_t v = curU8(c);
        if (c->bad || v > 1) return false;
        *out = BOOL_VAL(v != 0);
        return true;
    }
    case K_INT: {
        uint64_t bits = curU64(c);
        if (c->bad) return false;
        *out = INT_VAL((int64_t)bits);
        return true;
    }
    case K_FLOAT: {
        double d = curF64(c);
        if (c->bad) return false;
        *out = FLOAT_VAL(d);
        return true;
    }
    case K_STRREF: {
        uint64_t index = 0;
        if (!curUleb(c, &index)) return false;
        if (c->strings == NULL || index >= c->stringCount) return false;
        if (!IS_STRING(c->strings[index])) return false;   /* corrupt table */
        *out = c->strings[index];
        return true;
    }
    case K_BYTES: {
        const uint8_t *p = NULL;
        uint32_t len = 0;
        if (!readBlob(c, &p, &len)) return false;
        ObjBytes *bytes = jaiBytesNew(p, len);
        if (bytes == NULL) return false;
        *out = OBJ_VAL(bytes);
        return true;
    }
    case K_FUNC: {
        ObjFunction *fn = readFunction(c, module, depth + 1);
        if (fn == NULL) return false;
        *out = OBJ_VAL(fn);
        return true;
    }
    case K_TUPLE:
        return readTuple(c, module, out, depth);
    default:
        return false;
    }
}

/* Reads `count` constants into `out`. Every tag §4 defines is self-contained,
 * so one forward pass is enough: no constant refers to a later one.
 *
 * When `pushedRoots` is non-NULL the caller has no other reference to `out`,
 * so every stored value is also pushed as a GC temp root and the count is
 * reported back for the caller to pop, on the failure path too. */
static bool readConstantPool(Cursor *c, ObjModule *module, ValueArray *out,
                             uint32_t count, int depth, int *pushedRoots) {
    for (uint32_t i = 0; i < count; i++) {
        if (curLeft(c) == 0) return false;

        Value v = NULL_VAL;
        if (!readConstant(c, module, &v, depth)) return false;

        jaiValueArrayPush(out, v);
        if (pushedRoots != NULL) {
            jaiGCPushRoot(v);
            (*pushedRoots)++;
        }
    }
    return true;
}

/* ------------------------------------------------------------------ */
/* Reading: function records                                            */
/* ------------------------------------------------------------------ */

/* Inline-cache slots are not serialised — they must start empty — so their
 * number is recovered by decoding the code and taking the highest cache
 * operand. Walking the stream also proves it decodes cleanly; a file whose
 * instructions do not tile the code array exactly is rejected. */
static bool rebuildCaches(Chunk *chunk) {
    long maxIndex = -1;

    for (int off = 0; off < chunk->count;) {
        uint8_t raw = chunk->code[off];
        if (raw >= OP_COUNT) return false;
        OpCode op = (OpCode)raw;

        int size = jaiOpOperandSize(op);
        if (op == OP_CLOSURE) {
            if (off + 1 + 3 > chunk->count) return false;
            uint32_t k = jaiReadU24(chunk->code + off + 1);
            if (k >= (uint32_t)chunk->constants.count) return false;
            Value fn = chunk->constants.data[k];
            if (!IS_FUNCTION(fn)) return false;
            size = 3 + 3 * (int)AS_FUNCTION(fn)->upvalueCount;
        }
        if (size < 0 || off > chunk->count - 1 - size) return false;

        int cacheAt = jaiOpCacheOperand(op);
        if (cacheAt >= 0) {
            long idx = (long)jaiReadU16(chunk->code + off + 1 + cacheAt);
            if (idx > maxIndex) maxIndex = idx;
        }
        off += 1 + size;
    }

    /* Exactly maxIndex + 1 slots, not JAI_GROW_CAP's next power of two: the
     * count is known here, and the rounding cost 1.19 MB across the seed's
     * images. maxIndex starts at -1, so a chunk with no cache operands reserves
     * zero and allocates nothing. */
    return jaiChunkReserveCaches(chunk, (int)(maxIndex + 1)) &&
           chunk->cacheCount == (int)(maxIndex + 1);
}

/* §5. Returns an unrooted function; callers store it immediately, and no
 * allocation happens in between. */
static ObjFunction *readFunction(Cursor *c, ObjModule *module, int depth) {
    if (depth > JAIC_MAX_DEPTH) {
        c->bad = true;
        return NULL;
    }

    ObjFunction *fn = jaiFunctionNew();
    jaiGCPushRoot(OBJ_VAL(fn));
    int roots = 1;

    fn->module = module;
    fn->chunk.sourceFileId = module != NULL ? module->sourceFileId : -1;

    uint32_t nameIndex = curU32(c);
    uint8_t arity = curU8(c);
    uint32_t flags = c->version >= JAIC_VERSION_FNFLAGS16 ? curU16(c) : curU8(c);
    uint16_t maxSlots = curU16(c);
    uint16_t upvalueCount = curU16(c);
    if (c->bad || upvalueCount > JAI_MAX_UPVALUES) goto fail;

    {
        uint32_t codeLength = curU32(c);
        if (c->bad || codeLength > curLeft(c) || codeLength > (uint32_t)INT_MAX) {
            goto fail;
        }
        if (codeLength > 0) {
            const uint8_t *p = NULL;
            if (!curTake(c, codeLength, &p)) goto fail;
            fn->chunk.code = JAI_ALLOC(uint8_t, codeLength);
            memcpy(fn->chunk.code, p, codeLength);
            fn->chunk.count = (int)codeLength;
            fn->chunk.capacity = (int)codeLength;
        }
    }

    {
        uint32_t constCount = curU32(c);
        if (c->bad || constCount >= JAIC_CONST_MAX || !curFits(c, constCount, 1)) {
            goto fail;
        }
        /* The pool hangs off the rooted function, so its entries need no
         * roots of their own. */
        if (!readConstantPool(c, module, &fn->chunk.constants, constCount,
                              depth + 1, NULL)) {
            goto fail;
        }
    }

    {
        uint32_t lineBytes = curU32(c);
        if (c->bad || lineBytes > curLeft(c) || lineBytes > (uint32_t)INT_MAX) {
            goto fail;
        }
        /* A stripped image says zero here and the bytes live in the .jaid. The
         * sidecar is advanced for EVERY record, including ones with no entries,
         * so the two stay aligned. */
        Cursor *lineSrc = c;
        if (lineBytes == 0 && c->sidecar != NULL) {
            uint32_t sidecarBytes = curU32(c->sidecar);
            if (c->sidecar->bad || sidecarBytes > curLeft(c->sidecar)) {
                /* A malformed sidecar costs spans, not the load. */
                c->sidecar = NULL;
            } else {
                lineBytes = sidecarBytes;
                lineSrc = c->sidecar;
            }
        }
        if (lineBytes > 0) {
            /* The on-disk bytes ARE the runtime form, so loading the table
             * is one copy and no decode at all. */
            const uint8_t *p = NULL;
            if (!curTake(lineSrc, lineBytes, &p)) goto fail;
            fn->chunk.lineStream = JAI_ALLOC(uint8_t, lineBytes);
            memcpy(fn->chunk.lineStream, p, lineBytes);
            fn->chunk.lineStreamLen = (int)lineBytes;
            fn->chunk.lineStreamCap = (int)lineBytes;
        }
    }

    {
        uint16_t excCount = curU16(c);
        if (c->bad || !curFits(c, excCount, JAIC_EXC_ENTRY)) goto fail;
        if (excCount > 0) {
            fn->exceptions = JAI_ALLOC(ExceptionEntry, excCount);
            fn->exceptionCount = excCount;
            for (uint16_t i = 0; i < excCount; i++) {
                fn->exceptions[i].start = curU32(c);
                fn->exceptions[i].end = curU32(c);
                fn->exceptions[i].handler = curU32(c);
                fn->exceptions[i].typeConst = curU32(c);
            }
            if (c->bad) goto fail;
        }
    }

    {
        uint16_t defaultCount = curU16(c);
        if (c->bad || defaultCount > UINT8_MAX ||
            !curFits(c, defaultCount, JAIC_U32_ENTRY)) {
            goto fail;
        }
        if (defaultCount > 0) {
            fn->defaultOffsets = JAI_ALLOC(uint32_t, defaultCount);
            fn->defaultCount = (uint8_t)defaultCount;
            for (uint16_t i = 0; i < defaultCount; i++) {
                fn->defaultOffsets[i] = curU32(c);
            }
            if (c->bad) goto fail;
        }
    }

    {
        uint16_t paramCount = curU16(c);
        if (c->bad || !curFits(c, paramCount, JAIC_U32_ENTRY)) goto fail;
        if (paramCount > 0) {
            ObjString **names = JAI_ALLOC(ObjString *, paramCount);
            memset(names, 0, sizeof(ObjString *) * (size_t)paramCount);
            fn->paramNames = names;
            fn->paramCount = paramCount;
            for (uint16_t i = 0; i < paramCount; i++) {
                uint32_t idx = curU32(c);
                if (idx == JAIC_NO_NAME) continue;
                names[i] = poolString(&fn->chunk.constants, idx);
                if (names[i] == NULL) goto fail;
            }
            if (c->bad) goto fail;
        }
    }

    if (nameIndex != JAIC_NO_NAME) {
        ObjString *name = poolString(&fn->chunk.constants, nameIndex);
        if (name == NULL) goto fail;
        fn->name = name;
        fn->qualifiedName = name;   /* §5 stores no qualified name */
    }

    fn->arity = arity;
    fn->flags = flags;
    fn->maxSlots = maxSlots;
    fn->upvalueCount = upvalueCount;
    if ((flags & FN_METHOD) && fn->name != NULL && fn->name->length == 4 &&
        memcmp(fn->name->chars, "init", 4) == 0) {
        fn->flags |= FN_INIT;   /* see initFlagIsRecoverable */
    }

    /* Offsets the VM will branch to must land inside this chunk. The checksum
     * says the bytes are unaltered, but it says nothing about the compiler
     * that produced them; a handler pointing past the code would become a wild
     * jump. Full well-formedness is jaiVerifyChunk's job. */
    for (uint16_t i = 0; i < fn->exceptionCount; i++) {
        const ExceptionEntry *e = &fn->exceptions[i];
        if (e->start > e->end || e->end > (uint32_t)fn->chunk.count ||
            e->handler >= (uint32_t)fn->chunk.count) {
            goto fail;
        }
        if (e->typeConst != UINT32_MAX &&
            e->typeConst >= (uint32_t)fn->chunk.constants.count) {
            goto fail;
        }
    }
    for (uint8_t i = 0; i < fn->defaultCount; i++) {
        if (fn->defaultOffsets[i] >= (uint32_t)fn->chunk.count) goto fail;
    }

    if (!rebuildCaches(&fn->chunk)) goto fail;
    if ((fn->flags & FN_TRACE) && fn->chunk.caches != NULL) {
        for (int i = 0; i < fn->chunk.cacheCount; i++) {
            fn->chunk.caches[i].obsBudget = JAI_IC_OBS_BUDGET_TRACE;
        }
    }

    jaiGCPopRoots(roots);
    return fn;

fail:
    c->bad = true;
    jaiGCPopRoots(roots);
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Reading: the container                                               */
/* ------------------------------------------------------------------ */

/* Smallest byte count that could hold a header plus a checksum; anything
 * shorter cannot even be probed. */
#define JAIC_MIN_SIZE 32u

/* The escape hatch, and the only way to raise JAI_COMPILER_VERSION.
 *
 * Raising the version wedges the tree in two places at once, and relaxing the
 * seed alone is not enough. The seed serves the PREVIOUS compiler, which stamps
 * the previous version on everything it writes; those fresh images are then read
 * back as cache and refused for declaring the old version. Bumping the C
 * constant first fails that way, and bumping the self-hosted constant first
 * fails the other way, because an image from the future is refused outright. So
 * there is no ordering of a two-constant bump that works while both checks are
 * strict.
 *
 * With this set, both version checks are relaxed for one run -- long enough for
 * `make reseed` to build a seed that stamps the new version, after which nothing
 * needs it again. It deliberately does NOT relax the build id or the source
 * hash: those are what stop a stale image being replayed, and they are still
 * exactly right during a bootstrap.
 *
 * Checked once and warned about once, because the seed is loaded per module and
 * a warning per module would bury the build output it is meant to be visible
 * in. */
static bool jaiSeedAnyVersion(void) {
    static int state = -1;
    if (state < 0) {
        const char *v = getenv("JAITHON_SEED_ANY");
        state = (v != NULL && v[0] != '\0' && v[0] != '0') ? 1 : 0;
        if (state) {
            fprintf(stderr,
                    "warning: JAITHON_SEED_ANY is set -- the seed's minimum "
                    "version is not being enforced.\n"
                    "note: this exists to reseed a wedged tree; run `make "
                    "reseed` and then unset it.\n");
        }
    }
    return state == 1;
}

/* The build id is a *cache* key: it exists so that a rebuilt compiler does not
 * reuse `__jaicache__` entries the previous one wrote. The seed is not a cache.
 * It is compiled into this binary and is regenerated by the *previous* one, so
 * its recorded id is always the previous id and a strict check rejects it
 * always.
 *
 * That is not a small inconvenience: with the C front end gone the seed is the
 * only bootstrap, so a strict check made every edit under `src/vm` unbuildable
 * -- the new binary could not load the seed, and regenerating the seed needed a
 * binary that could run.
 *
 * JAI_COMPILER_VERSION used to be checked strictly for the seed too, on the
 * reasoning that changing the bytecode format means bumping it. That reasoning
 * is self-defeating: bumping it rejected the seed, which left no compiler able
 * to regenerate the seed, so the version could never be bumped and the rule it
 * enforced could never be exercised. The bump this file asks for was impossible
 * to perform.
 *
 * The seed is asked a DIFFERENT question from the cache, because the two want
 * different things:
 *
 *   cache -- must match exactly. It is an optimisation, and replaying bytecode
 *            emitted by a different compiler would run code the source no
 *            longer describes.
 *   seed  -- must merely be RUNNABLE by this binary. Opcodes are append-only
 *            (see OP_COUNT in chunk.h), so every opcode an older compiler could
 *            emit still exists here with the same number and meaning. An older
 *            seed is therefore executable, and that is the whole property
 *            bootstrapping needs.
 *
 * So the seed is accepted from JAI_SEED_MIN_VERSION up to the current version,
 * and refused above it -- an image from the future may use opcodes this binary
 * does not have. JAI_SEED_MIN_VERSION moves only for a change that append-only
 * cannot express (renumbering an opcode, widening an operand), and such a change
 * has to ship a reader for both formats for one release so that the seed can be
 * regenerated across the break. That is the migration this check now permits and
 * previously forbade.
 *
 * JAITHON_SEED_ANY=1 relaxes the floor for one run. It exists so a tree that has
 * wedged itself can always reseed its way out rather than needing a checkout; it
 * warns, and it is never the default. */
ObjFunction *deserializeWithSidecar(const uint8_t *data, size_t size,
                                    ObjModule *module,
                                    uint64_t expectedHash, bool fromSeed,
                                    const uint8_t *sidecar,
                                    size_t sidecarSize,
                                    size_t sidecarOffset);

static ObjFunction *deserialize(const uint8_t *data, size_t size,
                                ObjModule *module, uint64_t expectedHash,
                                bool fromSeed) {
    return deserializeWithSidecar(data, size, module, expectedHash, fromSeed,
                                  NULL, 0, 0);
}

ObjFunction *jaiDeserializeSeed(const uint8_t *data, size_t size,
                                ObjModule *module, uint64_t expectedHash) {
    return deserialize(data, size, module, expectedHash, true);
}

ObjFunction *jaiDeserializeModule(const uint8_t *data, size_t size,
                                  ObjModule *module, uint64_t expectedHash) {
    return deserialize(data, size, module, expectedHash, false);
}

ObjFunction *deserializeWithSidecar(const uint8_t *data, size_t size,
                                    ObjModule *module,
                                    uint64_t expectedHash, bool fromSeed,
                                    const uint8_t *sidecar,
                                    size_t sidecarSize,
                                    size_t sidecarOffset) {
    if (data == NULL || module == NULL || size < JAIC_MIN_SIZE) return NULL;
    if (memcmp(data, JAIC_MAGIC, 4) != 0) return NULL;

    /* Integrity first: nothing below parses a byte the checksum has not
     * already vouched for. */
    if (jaiCrc32(data, size - 4) != readU32At(data + size - 4)) return NULL;

    /* The cursor stops short of the checksum, so no parse can wander into it. */
    Cursor c = { data, size - 4, 4, false, 0, NULL, 0, NULL };
    Cursor sidecarCursor = { sidecar, sidecarSize, sidecarOffset, false, 0,
                             NULL, 0, NULL };
    if (sidecar != NULL) c.sidecar = &sidecarCursor;

    /* A RANGE, not an equality. A strict test here is evaluated before the
     * fromSeed branch below, so bumping JAIC_VERSION with it in place refuses
     * every image in boot/seed.bin -- written by the previous generation -- and
     * the tree wedges with "no front end" and no way to build one. */
    uint16_t version = curU16(&c);
    if (version < JAIC_VERSION_MIN || version > JAIC_VERSION) return NULL;
    c.version = version;
    (void)curU16(&c);                                  /* flags, informational */
    uint32_t recordedCompiler = curU32(&c);
    if (jaiSeedAnyVersion()) {
        /* Mid-bump: the running compiler is a generation behind this binary.
         * Everything below still applies, so a stale image is still refused. */
    } else if (fromSeed) {
        /* Runnable, not identical -- see the note above `deserialize`. */
        if (recordedCompiler > JAI_COMPILER_VERSION ||
            recordedCompiler < JAI_SEED_MIN_VERSION) {
            /* Say so. Refusing the seed leaves the tree with no compiler at
             * all, and the caller can only report that a front end is missing
             * -- which describes the symptom and not one word of the cause or
             * the way out. Once, not per module. */
            static bool told = false;
            if (!told) {
                told = true;
                fprintf(stderr,
                        "jaithon: the seed declares compiler version %u; this "
                        "build accepts %u..%u.\n"
                        "note: the seed is the only bootstrap, so nothing can "
                        "compile until this is resolved.\n"
                        "help: JAITHON_SEED_ANY=1 make reseed  -- then rebuild "
                        "and unset it.\n",
                        (unsigned)recordedCompiler,
                        (unsigned)JAI_SEED_MIN_VERSION,
                        (unsigned)JAI_COMPILER_VERSION);
            }
            return NULL;
        }
    } else if (recordedCompiler != JAI_COMPILER_VERSION) {
        return NULL;
    }
    uint32_t recordedBuildId = curU32(&c);
    if (!fromSeed && recordedBuildId != JAI_BUILD_ID) return NULL;
    uint64_t recordedHash = curU64(&c);
    /* The seed is allowed to be out of date with the source beside it. The
     * cache is not.
     *
     * A cache entry whose source has changed must be rejected: it is an
     * optimisation, and running it would run code the file no longer contains.
     * The seed is not an optimisation -- it is the only compiler a fresh tree
     * has, and it is asked for one precisely when the sources HAVE moved.
     * Editing the compiler changes its own source hash, so a hash-strict seed
     * refuses to load exactly when the compiler is being worked on, leaving
     * nothing able to rebuild it. That loop is what made lib/jaithon
     * unmodifiable.
     *
     * A one-generation-old compiler building the new sources is what
     * bootstrapping means, and `make fixpoint-check` is what proves the
     * generation it produces agrees with itself. */
    if (!fromSeed && recordedHash != expectedHash) return NULL;

    uint16_t pathLen = curU16(&c);
    if (!curSkip(&c, pathLen)) return NULL;            /* srcPath */
    (void)curU64(&c);                                  /* mtime, informational */
    if (c.bad) return NULL;

    ValueArray modulePool;
    jaiValueArrayInit(&modulePool);
    int poolRoots = 0;
    ObjFunction *body = NULL;
    bool ok = true;

    /* The string table comes before the pools, because every K_STRREF in them
     * indexes it. Its root range stays pushed for the whole read and is popped
     * on every exit below. */
    if (version >= JAIC_VERSION_STRTAB) {
        ok = readStrTab(&c);
    }

    uint32_t poolCount = ok ? curU32(&c) : 0;
    if (ok && (c.bad || poolCount >= JAIC_CONST_MAX ||
               !curFits(&c, poolCount, 1))) {
        ok = false;
    }
    if (ok) {
        ok = readConstantPool(&c, module, &modulePool, poolCount, 0, &poolRoots);
    }

    if (ok) {
        body = readFunction(&c, module, 0);
        ok = body != NULL;
    }
    if (ok) {
        jaiGCPushRoot(OBJ_VAL(body));
        poolRoots++;
    }

    /* Exports are collected before anything is written to `module`: a failure
     * anywhere must leave it exactly as it was. */
    uint32_t exportCount = 0;
    uint32_t *exportIndices = NULL;
    if (ok) {
        exportCount = curU32(&c);
        if (c.bad || !curFits(&c, exportCount, JAIC_U32_ENTRY)) {
            ok = false;
        } else if (exportCount > 0) {
            exportIndices = JAI_ALLOC(uint32_t, exportCount);
            for (uint32_t i = 0; i < exportCount; i++) {
                exportIndices[i] = curU32(&c);
                if (poolString(&modulePool, exportIndices[i]) == NULL) ok = false;
            }
            if (c.bad) ok = false;
        }
    }

    if (ok) {
        uint32_t importCount = curU32(&c);
        /* Nothing consumes the import list yet, but it must be present and
         * well-formed for the file to be considered whole. */
        if (c.bad || !curFits(&c, importCount, 2 * JAIC_U32_ENTRY)) {
            ok = false;
        } else {
            for (uint32_t i = 0; i < importCount && ok; i++) {
                uint32_t pathIndex = curU32(&c);
                uint32_t aliasIndex = curU32(&c);
                if (c.bad) { ok = false; break; }
                if (poolString(&modulePool, pathIndex) == NULL) ok = false;
                if (aliasIndex != JAIC_NO_NAME &&
                    poolString(&modulePool, aliasIndex) == NULL) {
                    ok = false;
                }
            }
        }
    }

    /* Trailing bytes between the last section and the checksum mean this is
     * not the file this reader thinks it is. */
    if (ok && (c.bad || c.pos != c.size)) ok = false;

    if (ok) {
        for (uint32_t i = 0; i < exportCount; i++) {
            ObjString *name = poolString(&modulePool, exportIndices[i]);
            (void)jaiTableSetInterned(&module->exports, name, BOOL_VAL(true));
        }
    }

    if (exportIndices != NULL) {
        JAI_FREE_ARRAY(uint32_t, exportIndices, exportCount);
    }
    jaiValueArrayFree(&modulePool);
    jaiGCPopRoots(poolRoots);
    if (c.strings != NULL) {
        /* Popped only here: every constant pool now holds its own reference to
         * each string, so the range has done its job. */
        jaiGCPopRootRange();
        JAI_FREE_ARRAY(Value, c.strings, c.stringCount);
    }
    return ok ? body : NULL;
}
