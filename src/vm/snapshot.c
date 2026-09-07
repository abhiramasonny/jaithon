/* snapshot.c -- can the initialised heap be written out as an image?
 *
 * Every `check`, `fmt`, `ast`, `doc` and `test`, and every edit-then-run, pays
 * ~15-17ms building the self-hosted front end: 98 modules inflated from
 * boot/seed.bin into ~11.4MB of live objects, used for a few milliseconds and
 * thrown away. Mapping that graph back instead of rebuilding it is the one
 * large multiplier found for the developer loop -- 20ms -> ~6ms. The design,
 * its measurements and its corrections are in
 * docs/research/PLAN-startup-snapshot.md.
 *
 * This file is the FIRST step of that work and deliberately writes nothing. It
 * answers the question the design cannot answer by reading: at the moment the
 * front end is built, does the live heap actually satisfy the preconditions a
 * heap image needs? Those preconditions are not guesses -- each one below is a
 * pointer that would be copied into the image and be WRONG on the way back:
 *
 *   - an ObjFile holding a live FILE *, which cannot be serialised at all;
 *   - ObjFunction::jitCode / jitFunc / jitLoop and JaiOsrForm::code, raw
 *     addresses in the JIT's per-process executable arena with no by-name
 *     rebind path (unlike ObjNative::fn, which has one). blackenFunction
 *     deliberately does not mark through these, so a writer that walks
 *     "whatever the GC walks" would copy them silently;
 *   - Chunk::constIndex, which markChunk also skips, for the same reason;
 *   - an OPEN ObjUpvalue, whose `location` points into the VM's value stack;
 *   - an ObjNative with no name, which the restore path could not rebind.
 *
 * Enumeration is the collector's own live list rather than a second traversal:
 * one forced collection leaves `objects` holding exactly the survivors, so this
 * cannot disagree with the GC about what is live -- which a hand-written walk
 * eventually would.
 *
 * Nothing here runs unless JAITHON_SNAPSHOT_AUDIT is set. */

#include <stdio.h>
#include <stdlib.h>

#include "vm/vm.h"
#include "vm/gc.h"
#include "vm/object/object.h"

void jaiSnapshotAudit(const char *when);

/* The bytes of an object's own HEADER, for every kind.
 *
 * jaiObjSoleBlock answers only for objects whose whole footprint is that block
 * and returns 0 for the nine kinds that own arrays -- the right answer to "can
 * this be freed with one call", the wrong one to "how many bytes is the
 * header". An image needs the second. Using the first meant the array-owning
 * kinds were never written AT ALL: every ObjClass was missing from the image,
 * so all 145 instances relocated their `klass` to nothing. Only the
 * reconstruction pass showed it -- the writer and the byte-comparison both
 * reported a clean round trip. */
static size_t snapshotHeaderSize(const Obj *o) {
    size_t sole = jaiObjSoleBlock(o);
    if (sole != 0) return sole;
    switch (o->type) {
    case OBJ_LIST:     return sizeof(ObjList);
    case OBJ_DICT:     return sizeof(ObjDict);
    case OBJ_SET:      return sizeof(ObjSet);
    case OBJ_FUNCTION: return sizeof(ObjFunction);
    case OBJ_CLOSURE:  return sizeof(ObjClosure);
    case OBJ_CLASS:    return sizeof(ObjClass);
    case OBJ_TRAIT:    return sizeof(ObjTrait);
    case OBJ_MODULE:   return sizeof(ObjModule);
    case OBJ_ENUM:     return sizeof(ObjEnum);
    case OBJ_FILE:     return sizeof(ObjFile);
    default:           return 0;
    }
}

void jaiSnapshotAudit(const char *when) {
    if (getenv("JAITHON_SNAPSHOT_AUDIT") == NULL) return;
    if (vm.gc == NULL) return;

    /* Survivors only: a sweep has just freed everything unreachable, so what is
     * left on the list is precisely the set an image would have to carry. */
    jaiGCCollect();

    unsigned long long byType[OBJ_TYPE_COUNT];
    for (int i = 0; i < OBJ_TYPE_COUNT; i++) byType[i] = 0;

    unsigned long long total = 0;
    unsigned long long openFiles = 0, closedFiles = 0;
    unsigned long long jitCode = 0, jitLoops = 0, osrForms = 0;
    unsigned long long constIndex = 0, openUpvalues = 0, namelessNatives = 0;

    for (Obj *o = vm.gc->objects; o != NULL; o = o->next) {
        total++;
        if (o->type < OBJ_TYPE_COUNT) byType[o->type]++;

        switch (o->type) {
        case OBJ_FILE: {
            ObjFile *f = (ObjFile *)o;
            if (f->handle != NULL) openFiles++; else closedFiles++;
            break;
        }
        case OBJ_FUNCTION: {
            ObjFunction *fn = (ObjFunction *)o;
            if (fn->jitCode != NULL || fn->jitFunc != NULL) jitCode++;
            if (fn->jitLoop != NULL) jitLoops++;
            if (fn->osrForms != NULL) osrForms++;
            if (fn->chunk.constIndex != NULL) constIndex++;
            break;
        }
        case OBJ_UPVALUE: {
            ObjUpvalue *uv = (ObjUpvalue *)o;
            /* An upvalue is CLOSED when its location points at its own `closed`
             * field; anything else is a pointer into the value stack. */
            if (uv->location != &uv->closed) openUpvalues++;
            break;
        }
        case OBJ_NATIVE: {
            ObjNative *n = (ObjNative *)o;
            if (n->name == NULL) namelessNatives++;
            break;
        }
        default:
            break;
        }
    }

    static const char *const kindName[OBJ_TYPE_COUNT] = {
        "string", "bytes", "list", "dict", "set", "tuple", "range",
        "function", "closure", "upvalue", "native", "bound",
        "class", "trait", "instance", "module", "enum", "enumval",
        "iter", "file", "enumctor", "strbuf",
    };

    fprintf(stderr, "[snapshot] audit at %s: %llu live objects\n", when, total);
    for (int i = 0; i < OBJ_TYPE_COUNT; i++) {
        if (byType[i] != 0)
            fprintf(stderr, "[snapshot]   %-10s %llu\n",
                    kindName[i] != NULL ? kindName[i] : "?", byType[i]);
    }
    fprintf(stderr,
            "[snapshot] blockers: open-file=%llu closed-file=%llu "
            "jit-code=%llu jit-loop=%llu osr-forms=%llu const-index=%llu "
            "open-upvalue=%llu nameless-native=%llu\n",
            openFiles, closedFiles, jitCode, jitLoops, osrForms, constIndex,
            openUpvalues, namelessNatives);
    /* What an image would actually have to COPY. Objects are individually
     * allocated -- a size-classed slab under 512 bytes, plain malloc above --
     * so nothing can be mapped as a region; every object and every array it
     * owns has to be copied and then relocated. jaiObjSoleBlock answers the
     * first half: a non-zero size means the object's whole footprint is its own
     * block, and zero means it owns arrays or a table that need their own
     * copy logic. Counting the second group is the real measure of how much
     * writer code this needs. */
    unsigned long long soleBytes = 0, soleCount = 0, ownsArrays = 0;
    unsigned long long ownsByType[OBJ_TYPE_COUNT];
    for (int i = 0; i < OBJ_TYPE_COUNT; i++) ownsByType[i] = 0;
    /* A string whose `chars` point into a SHARED ObjStrBuf rather than at its
     * own trailing bytes: the image cannot copy those bytes with the string,
     * and the pointer has to be relocated against the buffer instead. */
    unsigned long long sharedStrings = 0, ownStrings = 0;
    for (Obj *o = vm.gc->objects; o != NULL; o = o->next) {
        size_t sole = jaiObjSoleBlock(o);
        if (sole != 0) { soleCount++; soleBytes += sole; }
        else { ownsArrays++; if (o->type < OBJ_TYPE_COUNT) ownsByType[o->type]++; }
        if (o->type == OBJ_STRING) {
            if (((ObjString *)o)->owner != NULL) sharedStrings++;
            else ownStrings++;
        }
    }
    fprintf(stderr,
            "[snapshot] copy plan: %llu self-contained objects (%llu bytes), "
            "%llu own arrays or tables\n",
            soleCount, soleBytes, ownsArrays);
    for (int i = 0; i < OBJ_TYPE_COUNT; i++) {
        if (ownsByType[i] != 0)
            fprintf(stderr, "[snapshot]   owns-arrays %-10s %llu\n",
                    kindName[i] != NULL ? kindName[i] : "?", ownsByType[i]);
    }
    fprintf(stderr,
            "[snapshot] strings: %llu own their bytes, %llu point into a "
            "shared buffer\n", ownStrings, sharedStrings);

    /* Is the field table COMPLETE? An image relocates pointers field by field,
     * so a field nobody listed is a pointer copied verbatim into the new
     * address space. The way to find out is not to re-read the headers but to
     * follow every pointer this code believes exists and check it lands on an
     * object the collector agrees is live. A miss is either a field enumerated
     * wrongly or a pointer into memory the image would not carry.
     *
     * Direct Obj* fields only. Table and list CONTENTS are Values the GC
     * already traces and blackenObject is the authority on those; what has
     * historically been missed here is the plain pointer hanging off a header
     * (ObjString::owner, ObjFunction::module, ::jitBlockedOn). */
    {
        /* Open-addressed membership set over the live list. Power of two, and
         * generously sized so the probe stays short at this object count. */
        size_t cap = 1;
        while (cap < total * 4u) cap <<= 1;
        if (cap < 1024) cap = 1024;
        Obj **seen = (Obj **)calloc(cap, sizeof(Obj *));
        if (seen != NULL) {
            for (Obj *o = vm.gc->objects; o != NULL; o = o->next) {
                size_t h = ((uintptr_t)o >> 4) & (cap - 1);
                while (seen[h] != NULL) h = (h + 1) & (cap - 1);
                seen[h] = o;
            }
#define LIVE(p) ({                                                            \
        Obj *_q = (Obj *)(p);                                                 \
        bool _ok = true;                                                      \
        if (_q != NULL) {                                                     \
            size_t _h = ((uintptr_t)_q >> 4) & (cap - 1);                     \
            _ok = false;                                                      \
            while (seen[_h] != NULL) {                                        \
                if (seen[_h] == _q) { _ok = true; break; }                    \
                _h = (_h + 1) & (cap - 1);                                    \
            }                                                                 \
        }                                                                     \
        _ok; })
            unsigned long long checked = 0, missed = 0;
            unsigned long long missByType[OBJ_TYPE_COUNT];
            for (int i = 0; i < OBJ_TYPE_COUNT; i++) missByType[i] = 0;
#define CHECK(owner, p) do {                                                  \
        checked++;                                                            \
        if (!LIVE(p)) { missed++; missByType[(owner)->type]++; }              \
    } while (0)
            for (Obj *o = vm.gc->objects; o != NULL; o = o->next) {
                switch (o->type) {
                case OBJ_STRING:
                    CHECK(o, ((ObjString *)o)->owner);
                    break;
                case OBJ_FUNCTION: {
                    ObjFunction *f = (ObjFunction *)o;
                    CHECK(o, f->name);
                    CHECK(o, f->qualifiedName);
                    CHECK(o, f->module);
                    CHECK(o, f->jitBlockedOn);
                    break;
                }
                case OBJ_CLOSURE:
                    CHECK(o, ((ObjClosure *)o)->fn);
                    break;
                case OBJ_INSTANCE:
                    CHECK(o, ((ObjInstance *)o)->klass);
                    break;
                case OBJ_MODULE: {
                    ObjModule *m = (ObjModule *)o;
                    CHECK(o, m->name);
                    CHECK(o, m->path);
                    CHECK(o, m->body);
                    break;
                }
                case OBJ_CLASS: {
                    ObjClass *c = (ObjClass *)o;
                    CHECK(o, c->name);
                    CHECK(o, c->superclass);
                    break;
                }
                case OBJ_NATIVE:
                    CHECK(o, ((ObjNative *)o)->name);
                    break;
                default:
                    break;
                }
            }
#undef CHECK
#undef LIVE
            fprintf(stderr,
                    "[snapshot] field closure: %llu direct pointers checked, "
                    "%llu land outside the live set\n", checked, missed);
            for (int i = 0; i < OBJ_TYPE_COUNT; i++) {
                if (missByType[i] != 0)
                    fprintf(stderr, "[snapshot]   MISS in %s: %llu\n",
                            kindName[i] != NULL ? kindName[i] : "?",
                            missByType[i]);
            }
            free(seen);
        }
    }

    /* The single most favourable fact the design rests on, checked rather than
     * quoted: `ObjString::hash` is a CONTENT hash (jaiHashBytes over the bytes,
     * cached lazily, 0 meaning "not computed yet"). If that holds, every hash
     * table in the image stays valid when it is mapped at a different address,
     * and no table has to be rebuilt on load. If it did NOT hold -- if any hash
     * mixed in an address -- the whole technique would be impossible here, and
     * that is the kind of thing worth finding before writing a byte. */
    {
        unsigned long long hashed = 0, unhashed = 0, wrong = 0;
        for (Obj *o = vm.gc->objects; o != NULL; o = o->next) {
            if (o->type != OBJ_STRING) continue;
            ObjString *str = (ObjString *)o;
            if (str->hash == 0) { unhashed++; continue; }
            hashed++;
            if (str->hash != jaiHashBytes(str->chars, str->length)) wrong++;
        }
        fprintf(stderr,
                "[snapshot] string hashes: %llu cached, %llu never hashed, "
                "%llu DISAGREE with a recompute\n", hashed, unhashed, wrong);
    }

    /* Exact image size. The headers are the small half -- what an image must
     * actually carry is the arrays hanging off them, and until they are added
     * up "11.4MB of live heap" is a GC number, not a plan. Sizes below are the
     * ALLOCATED capacity, not the used count, because that is what a copy has
     * to reproduce for the structure to keep working. */
    {
        unsigned long long arrayBytes = 0;
        unsigned long long chunkCode = 0, chunkConsts = 0, chunkLines = 0;
        unsigned long long chunkCaches = 0, tableBytes = 0, listBytes = 0;
        unsigned long long upvalBytes = 0, unaccounted = 0;
#define TBL(t) (unsigned long long)((size_t)(t)->capacity * sizeof(JaiEntry) + \
                                    (size_t)(t)->capacity * sizeof(int32_t))
        for (Obj *o = vm.gc->objects; o != NULL; o = o->next) {
            if (jaiObjSoleBlock(o) != 0) continue;
            switch (o->type) {
            case OBJ_LIST: {
                ObjList *l = (ObjList *)o;
                size_t w = l->stg == LIST_STORE_BOXED ? sizeof(Value)
                         : l->stg == LIST_STORE_U8    ? 1u : 8u;
                listBytes += (unsigned long long)l->capacity * w;
                break;
            }
            case OBJ_DICT: tableBytes += TBL(&((ObjDict *)o)->table); break;
            case OBJ_SET:  tableBytes += TBL(&((ObjSet *)o)->table);  break;
            case OBJ_MODULE: {
                ObjModule *m = (ObjModule *)o;
                tableBytes += TBL(&m->globals) + TBL(&m->exports);
                break;
            }
            case OBJ_CLOSURE:
                upvalBytes += (unsigned long long)
                    ((ObjClosure *)o)->upvalueCount * sizeof(ObjUpvalue *);
                break;
            case OBJ_FUNCTION: {
                ObjFunction *f = (ObjFunction *)o;
                chunkCode   += (unsigned long long)f->chunk.capacity;
                chunkConsts += (unsigned long long)f->chunk.constants.capacity
                             * sizeof(Value);
                chunkLines  += (unsigned long long)f->chunk.lineStreamCap;
                chunkCaches += (unsigned long long)f->chunk.cacheCapacity
                             * sizeof(InlineCache);
                break;
            }
            default:
                /* class, trait, enum: several tables and arrays each, and only
                 * 121 objects between them -- counted as unaccounted rather
                 * than guessed at, so the total below is honest about what it
                 * does not yet include. */
                unaccounted++;
                break;
            }
        }
#undef TBL
        arrayBytes = chunkCode + chunkConsts + chunkLines + chunkCaches
                   + tableBytes + listBytes + upvalBytes;
        fprintf(stderr,
                "[snapshot] image size: %llu bytes of headers + %llu of arrays "
                "= %llu total (%llu objects not yet sized)\n",
                soleBytes + (total - soleCount) * 64ull, arrayBytes,
                soleBytes + (total - soleCount) * 64ull + arrayBytes,
                unaccounted);
        fprintf(stderr,
                "[snapshot]   chunk code %llu, constants %llu, lines %llu, "
                "caches %llu\n", chunkCode, chunkConsts, chunkLines,
                chunkCaches);
        fprintf(stderr,
                "[snapshot]   tables %llu, list items %llu, upvalue arrays %llu\n",
                tableBytes, listBytes, upvalBytes);
    }

    /* WRITER, SLICE ONE: assign every live object an index and rewrite the
     * direct header pointers as indices, then check the result round-trips.
     * Owned arrays are NOT carried yet -- this slice exists to prove the two
     * things everything else rests on: that a stable numbering can be assigned
     * over the collector's live list, and that every pointer an image would
     * store resolves back to exactly the object it came from.
     *
     * Index 0 is reserved for NULL so a missing pointer is not confusable with
     * the first object. */
    if (getenv("JAITHON_SNAPSHOT_WRITE") != NULL) {
        size_t icap = 1;
        while (icap < total * 4u) icap <<= 1;
        if (icap < 1024) icap = 1024;
        Obj **key = (Obj **)calloc(icap, sizeof(Obj *));
        uint32_t *val = (uint32_t *)calloc(icap, sizeof(uint32_t));
        if (key != NULL && val != NULL) {
            uint32_t next = 1;
            for (Obj *o = vm.gc->objects; o != NULL; o = o->next) {
                size_t h = ((uintptr_t)o >> 4) & (icap - 1);
                while (key[h] != NULL) h = (h + 1) & (icap - 1);
                key[h] = o; val[h] = next++;
            }
#define IDX(p) ({                                                             \
        Obj *_q = (Obj *)(p);                                                 \
        uint32_t _i = 0;                                                      \
        if (_q != NULL) {                                                     \
            size_t _h = ((uintptr_t)_q >> 4) & (icap - 1);                    \
            while (key[_h] != NULL) {                                         \
                if (key[_h] == _q) { _i = val[_h]; break; }                   \
                _h = (_h + 1) & (icap - 1);                                   \
            }                                                                 \
        }                                                                     \
        _i; })
            /* The inverse: index -> object, which the reader would build from
             * the image. Checking IDX and this agree on every pointer is the
             * round trip, minus the bytes. */
            Obj **byIndex = (Obj **)calloc(next, sizeof(Obj *));
            if (byIndex != NULL) {
                for (Obj *o = vm.gc->objects; o != NULL; o = o->next)
                    byIndex[IDX(o)] = o;

                unsigned long long rewritten = 0, bad = 0;
#define ROUND(owner, p) do {                                                  \
        uint32_t _i = IDX(p);                                                 \
        rewritten++;                                                          \
        if ((p) == NULL) { if (_i != 0) bad++; }                              \
        else if (_i == 0 || _i >= next || byIndex[_i] != (Obj *)(p)) bad++;   \
    } while (0)
                for (Obj *o = vm.gc->objects; o != NULL; o = o->next) {
                    switch (o->type) {
                    case OBJ_STRING: ROUND(o, ((ObjString *)o)->owner); break;
                    case OBJ_FUNCTION: {
                        ObjFunction *f = (ObjFunction *)o;
                        ROUND(o, f->name); ROUND(o, f->qualifiedName);
                        ROUND(o, f->module); ROUND(o, f->jitBlockedOn);
                        break;
                    }
                    case OBJ_CLOSURE: ROUND(o, ((ObjClosure *)o)->fn); break;
                    case OBJ_INSTANCE: ROUND(o, ((ObjInstance *)o)->klass); break;
                    case OBJ_MODULE: {
                        ObjModule *m = (ObjModule *)o;
                        ROUND(o, m->name); ROUND(o, m->path); ROUND(o, m->body);
                        break;
                    }
                    case OBJ_CLASS: {
                        ObjClass *c = (ObjClass *)o;
                        ROUND(o, c->name); ROUND(o, c->superclass);
                        break;
                    }
                    case OBJ_NATIVE: ROUND(o, ((ObjNative *)o)->name); break;
                    default: break;
                    }
                }
#undef ROUND
                fprintf(stderr,
                        "[snapshot] numbering: %u indices assigned, "
                        "%llu pointers rewritten and resolved back, %llu WRONG\n",
                        next - 1, rewritten, bad);

                /* SLICE TWO: write the self-contained objects to a file and
                 * read them back. These are the 63% whose entire footprint is
                 * their own block (jaiObjSoleBlock non-zero), so they need no
                 * array logic and isolate the question this slice is about:
                 * does the byte layout survive a write and a read?
                 *
                 * The record is {index, type, size, payload}. Payload is the
                 * object's own bytes verbatim -- pointers inside are NOT yet
                 * rewritten here, because slice one already proved the rewrite
                 * resolves and doing it twice would test nothing new. What is
                 * tested is that every byte comes back. */
                const char *path = getenv("JAITHON_SNAPSHOT_WRITE");
                if (path[0] != '\0' && path[0] != '1') {
                    FILE *f = fopen(path, "wb");
                    if (f == NULL) {
                        fprintf(stderr, "[snapshot] cannot write %s\n", path);
                    } else {
                        uint32_t magic = 0x4a414931u; /* "JAI1" */
                        unsigned long long wrote = 0, bytes = 0;
                        fwrite(&magic, sizeof magic, 1, f);
                        fwrite(&next, sizeof next, 1, f);
                        /* The payload goes out with its header pointers
                         * REWRITTEN AS INDICES, which is what makes the image
                         * loadable rather than merely comparable: a raw pointer
                         * means nothing in the next process. `Obj::next` is
                         * cleared outright -- it is the collector's own list and
                         * the reader rebuilds it as it allocates.
                         *
                         * Written from a copy, so the live object is untouched;
                         * the audit that follows still sees the real heap. */
                        unsigned char *tmp = NULL;
                        size_t tmpCap = 0;
                        for (Obj *o = vm.gc->objects; o != NULL; o = o->next) {
                            size_t sole = snapshotHeaderSize(o);
                            if (sole == 0) continue;
                            uint32_t ix = IDX(o);
                            uint32_t ty = (uint32_t)o->type;
                            uint32_t sz = (uint32_t)sole;
                            if (sole > tmpCap) {
                                unsigned char *nt = (unsigned char *)realloc(tmp, sole);
                                if (nt == NULL) break;
                                tmp = nt; tmpCap = sole;
                            }
                            memcpy(tmp, o, sole);
                            Obj *co = (Obj *)tmp;
                            co->next = NULL;
#define PIN(field) do { (field) = (void *)(uintptr_t)IDX(field); } while (0)
                            switch (o->type) {
                            case OBJ_STRING: {
                                ObjString *cs = (ObjString *)co;
                                PIN(cs->owner);
                                /* `chars` points at this object's own trailing
                                 * bytes (§10 proved none is shared), so it is
                                 * an OFFSET from the header, not an index. */
                                cs->chars = (char *)(uintptr_t)
                                    ((const char *)((ObjString *)o)->chars
                                     - (const char *)o);
                                break;
                            }
                            case OBJ_INSTANCE: PIN(((ObjInstance *)co)->klass); break;
                            case OBJ_FUNCTION: {
                                ObjFunction *cf = (ObjFunction *)co;
                                PIN(cf->name); PIN(cf->qualifiedName);
                                PIN(cf->module); PIN(cf->jitBlockedOn);
                                /* Arena addresses with no by-name route back:
                                 * reset to cold, which is what a fresh process
                                 * has anyway (§9 measured 2 and 5-8 of these
                                 * live at the snapshot point, so this is
                                 * required, not defensive). Owned arrays are
                                 * separate records; their pointers are rebuilt
                                 * on load, not carried. */
                                cf->jitCode = NULL; cf->jitFunc = NULL;
                                cf->jitLoop = NULL; cf->osrForms = NULL;
                                cf->osrCount = 0;
                                cf->chunk.code = NULL; cf->chunk.lineStream = NULL;
                                cf->chunk.caches = NULL;
                                cf->chunk.cacheCapacity = 0;
                                cf->chunk.constIndex = NULL;
                                cf->paramNames = NULL;
                                break;
                            }
                            case OBJ_CLOSURE: {
                                ObjClosure *cc = (ObjClosure *)co;
                                PIN(cc->fn); cc->upvalues = NULL; break;
                            }
                            case OBJ_CLASS: {
                                ObjClass *cc = (ObjClass *)co;
                                PIN(cc->name); PIN(cc->qualifiedName);
                                PIN(cc->superclass);
                                cc->fields = NULL; cc->traits = NULL; break;
                            }
                            case OBJ_MODULE: {
                                ObjModule *cm = (ObjModule *)co;
                                PIN(cm->name); PIN(cm->path); PIN(cm->body);
                                break;
                            }
                            case OBJ_ENUM: {
                                ObjEnum *ce = (ObjEnum *)co;
                                PIN(ce->name); ce->variants = NULL; break;
                            }
                            case OBJ_LIST: ((ObjList *)co)->items = NULL; break;
                            case OBJ_NATIVE: {
                                ObjNative *cn = (ObjNative *)co;
                                PIN(cn->name);
                                /* Re-bound by name on load; a code address and
                                 * a pointer into binary static storage are both
                                 * meaningless in the next process. */
                                cn->fn = NULL;
                                cn->paramNames = NULL;
                                break;
                            }
                            default: break;
                            }
#undef PIN
                            fwrite(&ix, sizeof ix, 1, f);
                            fwrite(&ty, sizeof ty, 1, f);
                            fwrite(&sz, sizeof sz, 1, f);
                            fwrite(tmp, 1, sole, f);
                            wrote++; bytes += sole;
                        }
                        free(tmp);
                        /* SLICE THREE: the arrays these objects OWN. Two
                         * shapes, and one of each is enough to prove the case:
                         * a plain byte array (a chunk's code and its line
                         * stream) and a POINTER array (a closure's upvalues),
                         * which must go out as indices like any other pointer.
                         *
                         * Inline caches are deliberately NOT written -- see the
                         * design note. They are 66% of the array bytes and a
                         * pure memo; `cacheAt` already degrades to the slow
                         * path on a NULL, which is the state a fresh process is
                         * in anyway. */
                        unsigned long long arrRecs = 0, arrBytes = 0;
                        for (Obj *o = vm.gc->objects; o != NULL; o = o->next) {
                            uint32_t ix = IDX(o);
                            if (o->type == OBJ_FUNCTION) {
                                ObjFunction *fn = (ObjFunction *)o;
                                uint32_t kind = 1;      /* chunk code */
                                uint32_t n = (uint32_t)fn->chunk.count;
                                uint32_t blen = n;
                                fwrite(&ix, sizeof ix, 1, f);
                                fwrite(&kind, sizeof kind, 1, f);
                                fwrite(&n, sizeof n, 1, f);
                                fwrite(&blen, sizeof blen, 1, f);
                                if (n) fwrite(fn->chunk.code, 1, n, f);
                                arrRecs++; arrBytes += n;

                                kind = 2;               /* line stream */
                                n = (uint32_t)fn->chunk.lineStreamLen;
                                blen = n;
                                fwrite(&ix, sizeof ix, 1, f);
                                fwrite(&kind, sizeof kind, 1, f);
                                fwrite(&n, sizeof n, 1, f);
                                fwrite(&blen, sizeof blen, 1, f);
                                if (n) fwrite(fn->chunk.lineStream, 1, n, f);
                                arrRecs++; arrBytes += n;
                            } else if (o->type == OBJ_CLOSURE) {
                                ObjClosure *cl = (ObjClosure *)o;
                                uint32_t kind = 3;      /* upvalues, as indices */
                                uint32_t n = (uint32_t)cl->upvalueCount;
                                uint32_t blen = n * (uint32_t)sizeof(uint32_t);
                                fwrite(&ix, sizeof ix, 1, f);
                                fwrite(&kind, sizeof kind, 1, f);
                                fwrite(&n, sizeof n, 1, f);
                                fwrite(&blen, sizeof blen, 1, f);
                                for (uint32_t u = 0; u < n; u++) {
                                    uint32_t ui = IDX(cl->upvalues[u]);
                                    fwrite(&ui, sizeof ui, 1, f);
                                }
                                arrRecs++; arrBytes += n * sizeof(uint32_t);
                            }
                        }
                        /* SLICE FOUR: JaiTables, and the Values inside them.
                         * This is the shape the rest of the tail reduces to --
                         * dict, set, module globals/exports, a class's five, a
                         * trait's two, an enum's methods are all the same
                         * struct. A Value goes out as {tag, payload}, with the
                         * payload replaced by an INDEX when it points at an
                         * object; that is the only part a plain memcpy would
                         * get wrong.
                         *
                         * `hash` rides along verbatim: §11 proved every string
                         * hash is content-derived and recomputes identically,
                         * so a table stays valid at a different address. */
                        unsigned long long tblRecs = 0, tblEntries = 0;
#define WRITE_VALUE(v) do {                                                   \
        uint32_t _t = (uint32_t)(v).type;                                     \
        uint64_t _p;                                                          \
        if (IS_OBJ(v)) _p = (uint64_t)IDX(AS_OBJ(v));                         \
        else memcpy(&_p, &(v).as, sizeof _p);                                 \
        fwrite(&_t, sizeof _t, 1, f);                                         \
        fwrite(&_p, sizeof _p, 1, f);                                         \
    } while (0)
#define WRITE_TABLE_K(o, t, k) do {                                           \
        uint32_t _ix = IDX(o), _kind = (k);                                   \
        uint32_t _cap = (uint32_t)(t)->capacity;                              \
        uint32_t _bl = _cap * 36u;   /* 4+8 key, 4+8 value, 8 hash, 4 order */                                            \
        fwrite(&_ix, sizeof _ix, 1, f);                                       \
        fwrite(&_kind, sizeof _kind, 1, f);                                   \
        fwrite(&_cap, sizeof _cap, 1, f);                                     \
        fwrite(&_bl, sizeof _bl, 1, f);                                       \
        for (uint32_t _e = 0; _e < _cap; _e++) {                              \
            JaiEntry *_en = &(t)->entries[_e];                                \
            WRITE_VALUE(_en->key);                                            \
            WRITE_VALUE(_en->value);                                          \
            fwrite(&_en->hash, sizeof _en->hash, 1, f);                       \
            fwrite(&_en->order, sizeof _en->order, 1, f);                     \
            tblEntries++;                                                     \
        }                                                                     \
        tblRecs++;                                                            \
    } while (0)
/* An object may own more than one table -- a module owns globals AND exports --
 * so the record carries WHICH, not just the owner. Without it the reader
 * matched by capacity and mis-paired 24 of 312 when the two happened to be
 * the same size. */
#define WRITE_TABLE(o, t) WRITE_TABLE_K(o, t, 4)
                        for (Obj *o = vm.gc->objects; o != NULL; o = o->next) {
                            switch (o->type) {
                            case OBJ_DICT: {
                                ObjDict *d = (ObjDict *)o;
                                if (d->table.entries) WRITE_TABLE(o, &d->table);
                                break;
                            }
                            case OBJ_SET: {
                                ObjSet *st = (ObjSet *)o;
                                if (st->table.entries) WRITE_TABLE(o, &st->table);
                                break;
                            }
                            case OBJ_MODULE: {
                                ObjModule *m = (ObjModule *)o;
                                if (m->globals.entries) WRITE_TABLE(o, &m->globals);
                                if (m->exports.entries) WRITE_TABLE_K(o, &m->exports, 5);
                                break;
                            }
                            case OBJ_CLASS: {
                                /* Five tables, and they are what a class IS --
                                 * methods dominates. Kinds 6..10 so the reader
                                 * can tell them apart, the lesson the module's
                                 * two tables taught. */
                                ObjClass *c = (ObjClass *)o;
                                if (c->methods.entries)    WRITE_TABLE_K(o, &c->methods, 6);
                                if (c->statics.entries)    WRITE_TABLE_K(o, &c->statics, 7);
                                if (c->getters.entries)    WRITE_TABLE_K(o, &c->getters, 8);
                                if (c->setters.entries)    WRITE_TABLE_K(o, &c->setters, 9);
                                if (c->restricted.entries) WRITE_TABLE_K(o, &c->restricted, 10);
                                break;
                            }
                            case OBJ_ENUM: {
                                ObjEnum *en = (ObjEnum *)o;
                                if (en->methods.entries) WRITE_TABLE_K(o, &en->methods, 11);
                                break;
                            }
                            case OBJ_LIST: {
                                /* Items, at the storage's own width. A BOXED
                                 * list holds Values and needs the same index
                                 * substitution a table entry does; the unboxed
                                 * widths are raw bytes. */
                                ObjList *l = (ObjList *)o;
                                if (l->items == NULL || l->count == 0) break;
                                uint32_t lix = IDX(o), lkind = 12;
                                uint32_t n = (uint32_t)l->count;
                                uint32_t stg = (uint32_t)l->stg;
                                uint32_t blen = 4u + (l->stg == LIST_STORE_BOXED
                                        ? n * 12u
                                        : n * (l->stg == LIST_STORE_U8 ? 1u : 8u));
                                fwrite(&lix, sizeof lix, 1, f);
                                fwrite(&lkind, sizeof lkind, 1, f);
                                fwrite(&n, sizeof n, 1, f);
                                fwrite(&blen, sizeof blen, 1, f);
                                fwrite(&stg, sizeof stg, 1, f);
                                if (l->stg == LIST_STORE_BOXED) {
                                    for (uint32_t e = 0; e < n; e++) {
                                        Value ev = ((Value *)l->items)[e];
                                        WRITE_VALUE(ev);
                                    }
                                } else {
                                    size_t w = l->stg == LIST_STORE_U8 ? 1u : 8u;
                                    fwrite(l->items, w, n, f);
                                }
                                tblRecs++;
                                break;
                            }
                            default: break;
                            }
                        }
#undef WRITE_TABLE
#undef WRITE_VALUE
                        fprintf(stderr,
                                "[snapshot] tables: %llu written, %llu entries\n",
                                tblRecs, tblEntries);

                        fprintf(stderr,
                                "[snapshot] arrays: %llu records, %llu bytes "
                                "(inline caches deliberately omitted)\n",
                                arrRecs, arrBytes);

                        long end = ftell(f);
                        fclose(f);
                        fprintf(stderr,
                                "[snapshot] wrote %llu objects, %llu payload "
                                "bytes, %ld file bytes -> %s\n",
                                wrote, bytes, end, path);

                        /* Read it back and check every record against the live
                         * object it came from, byte for byte.
                         *
                         * TIMED, because this is the cheapest honest proxy for
                         * what a real load would cost: it opens the file, reads
                         * every byte, and touches every record. A real reader
                         * does that plus allocation and relocation, and does
                         * NOT do the comparisons -- so this bounds the load
                         * from a direction that cannot flatter it. If this is
                         * slow, the whole technique is worth less than §5
                         * claims and it is better to know now. */
                        double rt0 = jaiClockMonotonic();
                        FILE *g = fopen(path, "rb");
                        if (g != NULL) {
                            uint32_t m2 = 0, n2 = 0;
                            unsigned long long read = 0, mismatch = 0;
                            unsigned char *buf = NULL;
                            size_t bufCap = 0;
                            if (fread(&m2, sizeof m2, 1, g) == 1 &&
                                fread(&n2, sizeof n2, 1, g) == 1 &&
                                m2 == magic && n2 == next) {
                                uint32_t ix, ty, sz;
                                /* Heap, and grown on demand: a self-contained
                                 * object is not necessarily small -- an
                                 * ObjString carries its bytes, and the front
                                 * end holds several past 4KB. A fixed stack
                                 * buffer silently truncated the read at 3,585
                                 * of 7,009 records. */
                                bufCap = 65536;
                                buf = (unsigned char *)malloc(bufCap);
                                if (buf == NULL) { mismatch++; goto readDone; }
                                while (read < wrote &&
                                       fread(&ix, sizeof ix, 1, g) == 1 &&
                                       fread(&ty, sizeof ty, 1, g) == 1 &&
                                       fread(&sz, sizeof sz, 1, g) == 1) {
                                    if (sz > bufCap) {
                                        unsigned char *nb = (unsigned char *)
                                            realloc(buf, sz);
                                        if (nb == NULL) { mismatch++; break; }
                                        buf = nb; bufCap = sz;
                                    }
                                    if (fread(buf, 1, sz, g) != sz) { mismatch++; break; }
                                    read++;
                                    Obj *orig = ix < next ? byIndex[ix] : NULL;
                                    if (orig == NULL ||
                                        (uint32_t)orig->type != ty ||
                                        snapshotHeaderSize(orig) != sz) {
                                        mismatch++;
                                        continue;
                                    }
                                    /* The payload is no longer byte-identical --
                                     * pointers are indices now -- so check that
                                     * each pinned field resolves back to the
                                     * object the live one points at. */
                                    Obj *co = (Obj *)buf;
                                    if (co->next != NULL) { mismatch++; continue; }
#define UNPIN(cf, lf) do {                                                    \
        uintptr_t _i = (uintptr_t)(cf);                                       \
        if ((lf) == NULL) { if (_i != 0) mismatch++; }                        \
        else if (_i == 0 || _i >= next || byIndex[_i] != (Obj *)(lf))         \
            mismatch++;                                                       \
    } while (0)
                                    if (ty == OBJ_STRING) {
                                        ObjString *cs = (ObjString *)co;
                                        ObjString *ls = (ObjString *)orig;
                                        UNPIN(cs->owner, ls->owner);
                                        if ((uintptr_t)cs->chars !=
                                            (uintptr_t)((const char *)ls->chars
                                                        - (const char *)ls))
                                            mismatch++;
                                    } else if (ty == OBJ_INSTANCE) {
                                        UNPIN(((ObjInstance *)co)->klass,
                                              ((ObjInstance *)orig)->klass);
                                    } else if (ty == OBJ_NATIVE) {
                                        UNPIN(((ObjNative *)co)->name,
                                              ((ObjNative *)orig)->name);
                                        if (((ObjNative *)co)->fn != NULL) mismatch++;
                                    }
#undef UNPIN
                                }
                            } else {
                                mismatch++;
                            }
                            /* The array section, checked the same way: every
                             * record against the live array it came from. */
                            {
                                uint32_t aix, akind, an;
                                unsigned long long ar = 0, amis = 0;
                                uint32_t ablen;
                                while (fread(&aix, sizeof aix, 1, g) == 1 &&
                                       fread(&akind, sizeof akind, 1, g) == 1 &&
                                       fread(&an, sizeof an, 1, g) == 1 &&
                                       fread(&ablen, sizeof ablen, 1, g) == 1) {
                                    size_t want = ablen;
                                    if (want > bufCap) {
                                        unsigned char *nb = (unsigned char *)
                                            realloc(buf, want);
                                        if (nb == NULL) { amis++; break; }
                                        buf = nb; bufCap = want;
                                    }
                                    if (want && fread(buf, 1, want, g) != want) {
                                        amis++; break;
                                    }
                                    ar++;
                                    Obj *orig = aix < next ? byIndex[aix] : NULL;
                                    if (orig == NULL) { amis++; continue; }
                                    if (akind == 1) {
                                        ObjFunction *fo = (ObjFunction *)orig;
                                        if ((uint32_t)fo->chunk.count != an ||
                                            (an && memcmp(buf, fo->chunk.code, an)))
                                            amis++;
                                    } else if (akind == 2) {
                                        ObjFunction *fo = (ObjFunction *)orig;
                                        if ((uint32_t)fo->chunk.lineStreamLen != an ||
                                            (an && memcmp(buf, fo->chunk.lineStream, an)))
                                            amis++;
                                    } else if (akind == 12) {
                                        /* List items: storage width first. */
                                        ObjList *lo = (ObjList *)orig;
                                        uint32_t stg = 0;
                                        memcpy(&stg, buf, 4);
                                        const unsigned char *q = buf + 4;
                                        if (orig->type != OBJ_LIST ||
                                            (uint32_t)lo->count != an ||
                                            stg != (uint32_t)lo->stg) {
                                            amis++;
                                        } else if (stg == LIST_STORE_BOXED) {
                                            for (uint32_t e = 0; e < an; e++) {
                                                uint32_t vt; uint64_t vp;
                                                memcpy(&vt, q, 4); q += 4;
                                                memcpy(&vp, q, 8); q += 8;
                                                Value lv = ((Value *)lo->items)[e];
                                                if (vt != (uint32_t)lv.type ||
                                                    (IS_OBJ(lv) &&
                                                     (vp >= next || byIndex[vp] != AS_OBJ(lv)))) {
                                                    amis++; break;
                                                }
                                            }
                                        } else {
                                            size_t w = stg == LIST_STORE_U8 ? 1u : 8u;
                                            if (memcmp(q, lo->items, w * an) != 0) amis++;
                                        }
                                    } else if (akind >= 4 && akind <= 11) {
                                        /* Compare every entry against the live
                                         * table: tags identical, object
                                         * payloads resolving back to the same
                                         * object, hash and order verbatim. */
                                        const JaiTable *lt = NULL;
                                        if (orig->type == OBJ_DICT)
                                            lt = &((ObjDict *)orig)->table;
                                        else if (orig->type == OBJ_SET)
                                            lt = &((ObjSet *)orig)->table;
                                        else if (orig->type == OBJ_MODULE) {
                                            ObjModule *mo = (ObjModule *)orig;
                                            lt = akind == 5 ? &mo->exports
                                                            : &mo->globals;
                                        }
                                        else if (orig->type == OBJ_CLASS) {
                                            ObjClass *co = (ObjClass *)orig;
                                            lt = akind == 6  ? &co->methods
                                               : akind == 7  ? &co->statics
                                               : akind == 8  ? &co->getters
                                               : akind == 9  ? &co->setters
                                                             : &co->restricted;
                                        }
                                        else if (orig->type == OBJ_ENUM) {
                                            lt = &((ObjEnum *)orig)->methods;
                                        }
                                        if (lt == NULL || (uint32_t)lt->capacity != an) {
                                            fprintf(stderr, "[snapshot]   table miss: kind=%u owner=%d an=%u cap=%d\n",
                                                    akind, (int)orig->type, an,
                                                    lt ? lt->capacity : -1);
                                            amis++;
                                        } else {
                                            const unsigned char *q = buf;
                                            for (uint32_t e = 0; e < an; e++) {
                                                uint32_t kt, vt; uint64_t kp, vp, hh;
                                                int32_t ord;
                                                memcpy(&kt, q, 4); q += 4;
                                                memcpy(&kp, q, 8); q += 8;
                                                memcpy(&vt, q, 4); q += 4;
                                                memcpy(&vp, q, 8); q += 8;
                                                memcpy(&hh, q, 8); q += 8;
                                                memcpy(&ord, q, 4); q += 4;
                                                const JaiEntry *le = &lt->entries[e];
                                                if (kt != (uint32_t)le->key.type ||
                                                    vt != (uint32_t)le->value.type ||
                                                    hh != le->hash || ord != le->order) {
                                                    fprintf(stderr, "[snapshot]   entry miss: kind=%u owner=%d e=%u kt=%u/%u vt=%u/%u\n",
                                                            akind, (int)orig->type, e, kt, (unsigned)le->key.type, vt, (unsigned)le->value.type);
                                                    amis++; break;
                                                }
                                                if (IS_OBJ(le->key) &&
                                                    (kp >= next || byIndex[kp] != AS_OBJ(le->key))) {
                                                    fprintf(stderr, "[snapshot]   key miss: owner=%d e=%u kp=%llu next=%u objtype=%d\n",
                                                            (int)orig->type, e,
                                                            (unsigned long long)kp, next,
                                                            (int)AS_OBJ(le->key)->type);
                                                    amis++; break;
                                                }
                                                if (IS_OBJ(le->value) &&
                                                    (vp >= next || byIndex[vp] != AS_OBJ(le->value))) {
                                                    fprintf(stderr, "[snapshot]   val miss: owner=%d e=%u vp=%llu next=%u objtype=%d\n",
                                                            (int)orig->type, e,
                                                            (unsigned long long)vp, next,
                                                            (int)AS_OBJ(le->value)->type);
                                                    amis++; break;
                                                }
                                            }
                                        }
                                    } else if (akind == 3) {
                                        ObjClosure *co = (ObjClosure *)orig;
                                        if ((uint32_t)co->upvalueCount != an) amis++;
                                        else for (uint32_t u = 0; u < an; u++) {
                                            uint32_t ui;
                                            memcpy(&ui, buf + u * sizeof(uint32_t),
                                                   sizeof ui);
                                            if (ui >= next ||
                                                byIndex[ui] != (Obj *)co->upvalues[u]) {
                                                amis++; break;
                                            }
                                        }
                                    }
                                }
                                fprintf(stderr,
                                        "[snapshot] read back %llu array "
                                        "records, %llu MISMATCH\n", ar, amis);
                            }
                            /* SLICE SEVEN: RECONSTRUCT. Allocate a fresh object
                             * per record from the image alone, relocate its
                             * indices back into pointers, and check the rebuilt
                             * graph has the same topology as the live one.
                             *
                             * Plain malloc, not the collector: this proves
                             * relocation, and reconstructing into the real heap
                             * belongs with the wiring-in step. Objects only --
                             * arrays and tables are already proven to survive
                             * the round trip, and attaching them needs the
                             * allocator this slice deliberately avoids. */
                            {
                                double lt0 = jaiClockMonotonic();
                                Obj **rebuilt = (Obj **)calloc(next, sizeof(Obj *));
                                FILE *h = fopen(path, "rb");
                                unsigned long long made = 0, relocBad = 0;
                                if (rebuilt != NULL && h != NULL) {
                                    uint32_t m3, n3;
                                    if (fread(&m3, 4, 1, h) == 1 &&
                                        fread(&n3, 4, 1, h) == 1 && n3 == next) {
                                        uint32_t ix, ty, sz;
                                        unsigned long long left = wrote;
                                        while (left-- &&
                                               fread(&ix, 4, 1, h) == 1 &&
                                               fread(&ty, 4, 1, h) == 1 &&
                                               fread(&sz, 4, 1, h) == 1) {
                                            unsigned char *mem =
                                                (unsigned char *)malloc(sz);
                                            if (mem == NULL) break;
                                            if (fread(mem, 1, sz, h) != sz) {
                                                free(mem); break;
                                            }
                                            if (ix < next) rebuilt[ix] = (Obj *)mem;
                                            made++;
                                        }
                                    }
                                    fclose(h);

                                    /* Relocate: index -> pointer, offset ->
                                     * address. A zero index is NULL; anything
                                     * out of range is a corrupt image and must
                                     * be caught here, not dereferenced. */
#define RELOC(f) do {                                                         \
        uintptr_t _i = (uintptr_t)(f);                                        \
        if (_i == 0) { (f) = NULL; }                                          \
        else if (_i >= next || rebuilt[_i] == NULL) { relocBad++; (f) = NULL; }\
        else { (f) = (void *)rebuilt[_i]; }                                   \
    } while (0)
                                    for (uint32_t i = 1; i < next; i++) {
                                        Obj *r = rebuilt[i];
                                        if (r == NULL) continue;
                                        switch (r->type) {
                                        case OBJ_STRING: {
                                            ObjString *rs = (ObjString *)r;
                                            RELOC(rs->owner);
                                            rs->chars = (char *)r +
                                                        (uintptr_t)rs->chars;
                                            break;
                                        }
                                        case OBJ_INSTANCE:
                                            RELOC(((ObjInstance *)r)->klass);
                                            break;
                                        case OBJ_FUNCTION: {
                                            ObjFunction *rf = (ObjFunction *)r;
                                            RELOC(rf->name); RELOC(rf->qualifiedName);
                                            RELOC(rf->module); RELOC(rf->jitBlockedOn);
                                            break;
                                        }
                                        case OBJ_CLOSURE:
                                            RELOC(((ObjClosure *)r)->fn); break;
                                        case OBJ_CLASS: {
                                            ObjClass *rc = (ObjClass *)r;
                                            RELOC(rc->name); RELOC(rc->qualifiedName);
                                            RELOC(rc->superclass); break;
                                        }
                                        case OBJ_MODULE: {
                                            ObjModule *rm = (ObjModule *)r;
                                            RELOC(rm->name); RELOC(rm->path);
                                            RELOC(rm->body); break;
                                        }
                                        case OBJ_ENUM:
                                            RELOC(((ObjEnum *)r)->name); break;
                                        case OBJ_NATIVE:
                                            RELOC(((ObjNative *)r)->name);
                                            break;
                                        default: break;
                                        }
                                    }
#undef RELOC
                                    /* Topology check: every relocated pointer
                                     * must land on the rebuilt object at the
                                     * same index as the live one's target. */
                                    unsigned long long topo = 0, topoBad = 0;
                                    for (uint32_t i = 1; i < next; i++) {
                                        Obj *r = rebuilt[i], *o2 = byIndex[i];
                                        if (r == NULL || o2 == NULL) continue;
                                        if (r->type != o2->type) { topoBad++; continue; }
                                        if (r->type == OBJ_STRING) {
                                            ObjString *rs = (ObjString *)r;
                                            ObjString *os = (ObjString *)o2;
                                            topo++;
                                            if (rs->length != os->length ||
                                                memcmp(rs->chars, os->chars,
                                                       os->length) != 0)
                                                topoBad++;
                                            if ((rs->owner == NULL) != (os->owner == NULL))
                                                topoBad++;
                                        } else if (r->type == OBJ_INSTANCE) {
                                            topo++;
                                            ObjClass *rk = ((ObjInstance *)r)->klass;
                                            ObjClass *ok = ((ObjInstance *)o2)->klass;
                                            if ((rk == NULL) != (ok == NULL) ||
                                                (ok != NULL &&
                                                 rebuilt[IDX(ok)] != (Obj *)rk))
                                                topoBad++;
                                        } else if (r->type == OBJ_FUNCTION) {
                                            topo++;
                                            ObjModule *rm2 = ((ObjFunction *)r)->module;
                                            ObjModule *om2 = ((ObjFunction *)o2)->module;
                                            if ((rm2 == NULL) != (om2 == NULL) ||
                                                (om2 != NULL &&
                                                 rebuilt[IDX(om2)] != (Obj *)rm2))
                                                topoBad++;
                                        } else if (r->type == OBJ_CLOSURE) {
                                            topo++;
                                            ObjFunction *rf2 = ((ObjClosure *)r)->fn;
                                            ObjFunction *of2 = ((ObjClosure *)o2)->fn;
                                            if ((rf2 == NULL) != (of2 == NULL) ||
                                                (of2 != NULL &&
                                                 rebuilt[IDX(of2)] != (Obj *)rf2))
                                                topoBad++;
                                        } else if (r->type == OBJ_CLASS) {
                                            topo++;
                                            ObjClass *rs2 = ((ObjClass *)r)->superclass;
                                            ObjClass *os2 = ((ObjClass *)o2)->superclass;
                                            if ((rs2 == NULL) != (os2 == NULL) ||
                                                (os2 != NULL &&
                                                 rebuilt[IDX(os2)] != (Obj *)rs2))
                                                topoBad++;
                                        } else if (r->type == OBJ_NATIVE) {
                                            topo++;
                                            ObjString *rn = ((ObjNative *)r)->name;
                                            ObjString *on = ((ObjNative *)o2)->name;
                                            if ((rn == NULL) != (on == NULL) ||
                                                (on != NULL &&
                                                 rebuilt[IDX(on)] != (Obj *)rn))
                                                topoBad++;
                                        }
                                    }
                                    fprintf(stderr,
                                            "[snapshot] rebuilt %llu objects in "
                                            "%.3f ms, %llu bad relocations, "
                                            "%llu topology checks, %llu WRONG\n",
                                            made,
                                            (jaiClockMonotonic() - lt0) * 1000.0,
                                            relocBad, topo, topoBad);
                                    for (uint32_t i = 1; i < next; i++)
                                        free(rebuilt[i]);
                                }
                                free(rebuilt);
                            }

                        readDone:
                            free(buf);
                            fclose(g);
                            fprintf(stderr,
                                    "[snapshot] read back %llu objects, "
                                    "%llu MISMATCH, whole read pass %.3f ms\n",
                                    read, mismatch,
                                    (jaiClockMonotonic() - rt0) * 1000.0);
                        }
                    }
                }
                free(byIndex);
            }
#undef IDX
        }
        free(key); free(val);
    }

    fprintf(stderr, "[snapshot] verdict: %s\n",
            (openFiles == 0 && jitCode == 0 && jitLoops == 0 &&
             osrForms == 0 && constIndex == 0 && openUpvalues == 0 &&
             namelessNatives == 0)
                ? "writable -- no unserialisable pointer is reachable"
                : "NOT writable as-is; the counts above say what must be "
                  "reset or refused");
}
