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
