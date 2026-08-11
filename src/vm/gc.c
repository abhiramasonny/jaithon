/* gc.c — precise mark-sweep collector (spec/BYTECODE.md §10).
 *
 * Tri-colour: white = unmarked, gray = marked and on the gray stack awaiting
 * tracing, black = marked and traced. Marking is precise — every reference is
 * reached through a typed field, never by scanning memory conservatively — so
 * anything the root set and the tracers below miss is a use-after-free.
 */
#include <stdlib.h>

#include "gc.h"
#include "chunk.h"
#include "object.h"
#include "table.h"
#include "vm.h"
#include "jit.h"

#define JAI_GC_DEFAULT_GROW_FACTOR 2.0
#define JAI_GC_DEFAULT_MIN_HEAP    ((size_t)1 << 20)

/* Every entry point but jaiGCInit/jaiGCFree is declared without a GCState
 * argument, so the active state is kept here as well as in vm.gc. Init writes
 * both; activeGC() prefers this copy so the collector keeps working even if
 * the VM struct is re-zeroed after installation. */
GCState *jaiGCActive;

/* Set while jaiGCCollect runs. A collection must never start another: the
 * heap is inconsistent between mark and sweep, and re-entering would blacken
 * objects a second pass has already begun to free. */
bool jaiGCInCollect;

static GCState *activeGC(void) { return jaiGCActive != NULL ? jaiGCActive : vm.gc; }

/* The back edge's one-word form of jaiGCWanted(). The expression below is the
 * old inline test rearranged, not a new policy: it reads jaiGCActive rather
 * than activeGC() and g->stress rather than gcStressOn(g) for exactly the
 * reason the old one did. */
size_t jaiGCLimit;

void jaiGCSyncLimit(void) {
    const GCState *g = jaiGCActive;
    jaiGCLimit = (g != NULL && g->enabled && !jaiGCInCollect && !g->stress)
                     ? g->nextGC
                     : 0;
}

/* The CLI sets its flags on the VM; honour them directly so startup order
 * cannot leave --gc-stress or --debug-gc silently inactive. */
static bool gcStressOn(const GCState *g)  { return g->stress  || vm.gcStress; }
static bool gcVerboseOn(const GCState *g) { return g->verbose || vm.debugGC; }

/* The collector used to keep its own byte total, mirrored from the allocator
 * by a hook fired on every jaiRealloc. Two counters holding the same number,
 * one of them reached through a function pointer, cost 3.5% of dict_ops for
 * nothing: the allocator's counter is the answer, so read it.
 *
 * The marker still never runs from inside jaiRealloc — that would hit
 * arbitrary points in object construction, where a half-built object is
 * reachable from no root yet. Collections happen only at the safepoints that
 * call jaiGCMaybeCollect. */
static size_t gcLiveBytes(const GCState *g) { (void)g; return jaiHeapBytes; }

static size_t gcNextThreshold(const GCState *g, size_t live) {
    double factor = g->growFactor > 1.0 ? g->growFactor : JAI_GC_DEFAULT_GROW_FACTOR;
    double target = (double)live * factor;
    size_t next = target >= (double)SIZE_MAX ? SIZE_MAX : (size_t)target;
    size_t floorBytes = g->minHeap != 0 ? g->minHeap : JAI_GC_DEFAULT_MIN_HEAP;
    return next > floorBytes ? next : floorBytes;
}

/* ------------------------------------------------------------------ */
/* Lifecycle                                                           */
/* ------------------------------------------------------------------ */

void jaiGCInit(GCState *gc) {
    if (gc == NULL) JAI_PANIC("jaiGCInit: NULL collector state");

    gc->objects = NULL;
    gc->grayStack = NULL;
    gc->grayCount = 0;
    gc->grayCapacity = 0;

    /* Whatever the front end allocated before the VM came up already counts
     * against the heap, so start from the allocator's running total. */
    gc->growFactor = JAI_GC_DEFAULT_GROW_FACTOR;
    gc->minHeap = JAI_GC_DEFAULT_MIN_HEAP;
    gc->nextGC = gcNextThreshold(gc, jaiHeapBytes);

    gc->tempRoots = NULL;
    gc->tempRootCount = 0;
    gc->tempRootCapacity = 0;

    gc->permanentRoots = NULL;
    gc->permanentRootCount = 0;
    gc->permanentRootCapacity = 0;

    gc->enabled = true;
    gc->stress = false;
    gc->verbose = false;

    gc->collections = 0;
    gc->totalFreed = 0;
    gc->totalPauseSeconds = 0.0;

    jaiGCActive = gc;
    vm.gc = gc;
    jaiGCInCollect = false;
    jaiGCSyncLimit();
}

void jaiGCFree(GCState *gc) {
    if (gc == NULL) return;

    Obj *object = gc->objects;
    while (object != NULL) {
        Obj *next = object->next;
        jaiFreeObject(object);
        object = next;
    }
    gc->objects = NULL;

    JAI_FREE_ARRAY(Value, gc->tempRoots, gc->tempRootCapacity);
    gc->tempRoots = NULL;
    gc->tempRootCount = 0;
    gc->tempRootCapacity = 0;

    JAI_FREE_ARRAY(Value, gc->permanentRoots, gc->permanentRootCapacity);
    gc->permanentRoots = NULL;
    gc->permanentRootCount = 0;
    gc->permanentRootCapacity = 0;

    if (jaiGCActive == gc) jaiGCActive = NULL;
    if (vm.gc == gc) vm.gc = NULL;
    jaiGCSyncLimit();

    /* Raw free: the gray stack never went through jaiRealloc. See below. */
    free(gc->grayStack);
    gc->grayStack = NULL;
    gc->grayCount = 0;
    gc->grayCapacity = 0;
}

void jaiGCEnable(bool enabled) {
    GCState *g = activeGC();
    if (g != NULL) g->enabled = enabled;
    jaiGCSyncLimit();
}

/* ------------------------------------------------------------------ */
/* Marking                                                             */
/* ------------------------------------------------------------------ */

void jaiGCMarkObject(Obj *obj) {
    if (obj == NULL || obj->isMarked) return;
    obj->isMarked = true;

    /* Leaf types own no references, so blackening them would do nothing and
     * they never enter the gray stack. */
    switch (obj->type) {
    case OBJ_STRING: {
        /* Leaf, except that a slice keeps its buffer alive. The buffer is
         * itself a leaf, so marking it here rather than graying this string
         * costs one call and keeps every string off the gray stack. */
        ObjString *s = (ObjString *)obj;
        if (s->owner != NULL) jaiGCMarkObject((Obj *)s->owner);
        return;
    }
    case OBJ_STRBUF:
    case OBJ_BYTES:
    case OBJ_RANGE:
        return;
    default:
        break;
    }

    GCState *g = activeGC();
    if (g == NULL) JAI_PANIC("jaiGCMarkObject called before jaiGCInit");

    if (g->grayCapacity < g->grayCount + 1) {
        int newCapacity = JAI_GROW_CAP(g->grayCapacity);
        if (newCapacity <= g->grayCapacity) JAI_PANIC("GC gray stack overflow");
        /* The gray stack is the one permitted exception to the jaiRealloc
         * rule: it is grown with the raw system allocator so that growing it
         * in the middle of a collection cannot feed the GC's allocation hook
         * and recurse back into the collector. */
        Obj **grown = realloc(g->grayStack, sizeof(Obj *) * (size_t)newCapacity);
        if (grown == NULL) JAI_PANIC("out of memory growing the GC gray stack");
        g->grayStack = grown;
        g->grayCapacity = newCapacity;
    }
    g->grayStack[g->grayCount++] = obj;
}

void jaiGCMarkValue(Value v) { jaiGCMarkVal(v); }

void jaiGCMarkArray(const ValueArray *a) {
    if (a == NULL || a->data == NULL) return;
    for (int i = 0; i < a->count; i++) jaiGCMarkVal(a->data[i]);
}

static void markValues(const Value *values, int count) {
    if (values == NULL) return;
    for (int i = 0; i < count; i++) jaiGCMarkVal(values[i]);
}

static void markStrings(ObjString *const *names, int count) {
    if (names == NULL) return;
    for (int i = 0; i < count; i++) jaiGCMark((Obj *)names[i]);
}

static void markChunk(Chunk *chunk) {
    jaiGCMarkArray(&chunk->constants);

    /* chunk->constIndex is deliberately not marked: its keys are constant
     * hashes and its values pool indices, both plain ints, and every constant
     * it names is already marked above. Marking it would be a no-op that
     * invites someone to start storing Values in it. */

    /* Inline caches hold strong references to what they memoise. Every way is
     * marked, not just the first `count`: a way written and later abandoned
     * would otherwise be freed while the slot still points at it, and the
     * next probe of that way would read a dangling Value. */
    for (int i = 0; i < chunk->cacheCount; i++) {
        InlineCache *ic = &chunk->caches[i];
        for (int w = 0; w < JAI_IC_WAYS; w++) jaiGCMarkVal(ic->cached[w]);
    }
}

static void blackenFunction(ObjFunction *fn) {
    jaiGCMark((Obj *)fn->name);
    jaiGCMark((Obj *)fn->qualifiedName);
    jaiGCMark((Obj *)fn->module);
    jaiGCMark((Obj *)fn->owner);
    markStrings(fn->paramNames, (int)fn->paramCount);
    markChunk(&fn->chunk);
}

static void blackenClass(ObjClass *c) {
    jaiGCMark((Obj *)c->name);
    jaiGCMark((Obj *)c->qualifiedName);
    jaiGCMark((Obj *)c->superclass);

    jaiTableMark(&c->methods);
    jaiTableMark(&c->statics);
    jaiTableMark(&c->getters);
    jaiTableMark(&c->setters);
    /* Values here are packed ints, but the keys are the method names and this
     * table can outlive the entry that put them in the intern table. */
    jaiTableMark(&c->restricted);

    for (int i = 0; i < (int)c->traitCount; i++)
        jaiGCMark((Obj *)c->traits[i]);

    /* Field names are only reachable from here; the FieldInfo array itself is
     * plain memory owned by the class. */
    for (int i = 0; i < (int)c->fieldCount; i++)
        jaiGCMark((Obj *)c->fields[i].name);

    jaiGCMarkVal(c->initializer);

    /* The dunder cache aliases entries in `methods`, but a method removed from
     * the table before the cache is refreshed would otherwise dangle. */
    jaiGCMarkVal(c->dunderStr);
    jaiGCMarkVal(c->dunderRepr);
    jaiGCMarkVal(c->dunderEq);
    jaiGCMarkVal(c->dunderLt);
    jaiGCMarkVal(c->dunderHash);
    jaiGCMarkVal(c->dunderAdd);
    jaiGCMarkVal(c->dunderSub);
    jaiGCMarkVal(c->dunderMul);
    jaiGCMarkVal(c->dunderDiv);
    jaiGCMarkVal(c->dunderMod);
    jaiGCMarkVal(c->dunderPow);
    jaiGCMarkVal(c->dunderNeg);
    jaiGCMarkVal(c->dunderLen);
    jaiGCMarkVal(c->dunderGetItem);
    jaiGCMarkVal(c->dunderSetItem);
    jaiGCMarkVal(c->dunderContains);
    jaiGCMarkVal(c->dunderIter);
    jaiGCMarkVal(c->dunderNext);
    jaiGCMarkVal(c->dunderCall);
}

static void blackenEnum(ObjEnum *e) {
    jaiGCMark((Obj *)e->name);
    jaiTableMark(&e->methods);
    if (e->variants == NULL) return;
    for (int i = 0; i < (int)e->variantCount; i++) {
        EnumVariant *v = &e->variants[i];
        jaiGCMark((Obj *)v->name);
        /* The cached payload-less value and the cached constructor are
         * reachable only from here; without this the second mention of
         * `Color.Red` hands back freed memory. */
        jaiGCMark((Obj *)v->unit);
        jaiGCMark((Obj *)v->ctor);
        markStrings(v->fieldNames, (int)v->arity);
    }
}

/* Trace one gray object's references, turning it black. */
static void blackenObject(Obj *obj) {
    switch (obj->type) {
    case OBJ_STRING:
    case OBJ_STRBUF:
    case OBJ_BYTES:
    case OBJ_RANGE:
        break;

    case OBJ_LIST: {
        ObjList *list = (ObjList *)obj;
        markValues(list->items, list->count);
        break;
    }
    case OBJ_DICT:
        jaiTableMark(&((ObjDict *)obj)->table);
        break;
    case OBJ_SET:
        jaiTableMark(&((ObjSet *)obj)->table);
        break;
    case OBJ_TUPLE: {
        ObjTuple *t = (ObjTuple *)obj;
        markValues(t->items, (int)t->count);
        break;
    }

    case OBJ_FUNCTION:
        blackenFunction((ObjFunction *)obj);
        break;
    case OBJ_CLOSURE: {
        ObjClosure *closure = (ObjClosure *)obj;
        jaiGCMark((Obj *)closure->fn);
        for (int i = 0; i < closure->upvalueCount; i++)
            jaiGCMark((Obj *)closure->upvalues[i]);
        break;
    }
    case OBJ_UPVALUE:
        /* Only the closed slot. While the upvalue is open its target is a live
         * stack slot, already covered by the stack scan; following `location`
         * here would also read slots above stackTop during frame teardown. */
        jaiGCMarkVal(((ObjUpvalue *)obj)->closed);
        break;
    case OBJ_NATIVE:
        jaiGCMark((Obj *)((ObjNative *)obj)->name);
        break;
    case OBJ_BOUND: {
        ObjBound *bound = (ObjBound *)obj;
        jaiGCMarkVal(bound->receiver);
        jaiGCMarkVal(bound->method);
        break;
    }

    case OBJ_CLASS:
        blackenClass((ObjClass *)obj);
        break;
    case OBJ_TRAIT: {
        ObjTrait *trait = (ObjTrait *)obj;
        jaiGCMark((Obj *)trait->name);
        jaiTableMark(&trait->required);
        jaiTableMark(&trait->defaults);
        for (int i = 0; i < (int)trait->superCount; i++)
            jaiGCMark((Obj *)trait->supers[i]);
        break;
    }
    case OBJ_INSTANCE: {
        ObjInstance *inst = (ObjInstance *)obj;
        jaiGCMark((Obj *)inst->klass);
        markValues(inst->fields, (int)inst->fieldCount);
        break;
    }

    case OBJ_MODULE: {
        ObjModule *module = (ObjModule *)obj;
        jaiGCMark((Obj *)module->name);
        jaiGCMark((Obj *)module->path);
        jaiGCMark((Obj *)module->body);
        jaiTableMark(&module->globals);
        jaiTableMark(&module->exports);
        break;
    }

    case OBJ_ENUM:
        blackenEnum((ObjEnum *)obj);
        break;
    case OBJ_ENUM_VAL: {
        ObjEnumVal *ev = (ObjEnumVal *)obj;
        jaiGCMark((Obj *)ev->type);
        markValues(ev->payload, (int)ev->count);
        break;
    }

    case OBJ_ITER:
        jaiGCMarkVal(((ObjIter *)obj)->source);
        break;
    case OBJ_FILE:
        jaiGCMark((Obj *)((ObjFile *)obj)->path);
        break;

    case OBJ_ENUM_CTOR:
        jaiGCMark((Obj *)((ObjEnumCtor *)obj)->type);
        break;

    case OBJ_TYPE_COUNT:
        JAI_UNREACHABLE();
        break;
    }
}

/* ------------------------------------------------------------------ */
/* Roots                                                               */
/* ------------------------------------------------------------------ */

static void markInternedNames(void) {
    ObjString *const names[] = {
        vm.strInit, vm.strStr, vm.strRepr, vm.strEq, vm.strLt, vm.strHash,
        vm.strLen, vm.strGetItem, vm.strSetItem, vm.strContains, vm.strIter,
        vm.strNext, vm.strCall, vm.strAdd, vm.strSub, vm.strMul, vm.strDiv,
        vm.strMod, vm.strPow, vm.strNeg, vm.strMain, vm.strSelf, vm.strMessage,
    };
    markStrings(names, (int)(sizeof names / sizeof names[0]));
}

static void markWellKnownClasses(void) {
    ObjClass *const classes[] = {
        vm.cError, vm.cTypeError, vm.cValueError, vm.cNameError, vm.cIndexError,
        vm.cKeyError, vm.cAttributeError, vm.cArithmeticError,
        vm.cDivisionByZeroError, vm.cOverflowError, vm.cIOError, vm.cOSError,
        vm.cRuntimeError, vm.cRecursionError, vm.cStopIteration,
        vm.cAssertionError, vm.cImportError, vm.cFileNotFoundError,
        vm.cPermissionError, vm.cParseError, vm.cLookupError,
    };
    for (size_t i = 0; i < sizeof classes / sizeof classes[0]; i++)
        jaiGCMark((Obj *)classes[i]);
}

static void markRoots(GCState *g) {
    if (vm.stack != NULL) {
        for (Value *slot = vm.stack; slot < vm.stackTop; slot++)
            jaiGCMarkVal(*slot);
    }

    if (vm.frames != NULL) {
        for (int i = 0; i < vm.frameCount; i++) {
            jaiGCMark((Obj *)vm.frames[i].closure);
            jaiGCMark((Obj *)vm.frames[i].module);
        }
    }

    for (ObjUpvalue *uv = vm.openUpvalues; uv != NULL; uv = uv->next)
        jaiGCMark((Obj *)uv);

    jaiMarkAsciiChars();
    jaiJitMarkFrames();

    markValues(vm.defers.data, vm.defers.count);
    markStrings(vm.modulePath.data, vm.modulePath.count);

    jaiTableMark(&vm.modules);
    jaiGCMark((Obj *)vm.mainModule);
    jaiGCMark((Obj *)vm.builtins);

    /* Marked whether or not hasException is set: a stale pending exception
     * that was never overwritten must not be left dangling. */
    jaiGCMarkVal(vm.pendingException);

    markInternedNames();
    markWellKnownClasses();

    markValues(g->tempRoots, g->tempRootCount);
    markValues(g->permanentRoots, g->permanentRootCount);
}

static void traceReferences(GCState *g) {
    while (g->grayCount > 0) blackenObject(g->grayStack[--g->grayCount]);
}

/* ------------------------------------------------------------------ */
/* Sweep                                                               */
/* ------------------------------------------------------------------ */

static int sweep(GCState *g) {
    int freedObjects = 0;
    Obj *previous = NULL;
    Obj *object = g->objects;

    while (object != NULL) {
        if (object->isMarked) {
            object->isMarked = false;   /* white again for the next cycle */
            previous = object;
            object = object->next;
            continue;
        }

        Obj *unreached = object;
        object = object->next;
        if (previous != NULL) {
            previous->next = object;
        } else {
            g->objects = object;
        }
        jaiFreeObject(unreached);
        freedObjects++;
    }
    return freedObjects;
}

/* ------------------------------------------------------------------ */
/* Collection                                                          */
/* ------------------------------------------------------------------ */

void jaiGCCollect(void) {
    GCState *g = activeGC();
    if (g == NULL || !g->enabled || jaiGCInCollect) return;

    jaiGCInCollect = true;
    jaiGCSyncLimit();
    double started = jaiClockMonotonic();
    bool verbose = gcVerboseOn(g);
    size_t before = gcLiveBytes(g);
    if (verbose) fprintf(stderr, "-- gc begin\n");

    markRoots(g);
    traceReferences(g);

    /* The intern table's references are weak: an interned string stays alive
     * only if something else marked it. Drop the white entries before sweep,
     * or the table would keep keys pointing at freed strings. */
    JaiTable *interned = jaiInternTable();
    if (interned != NULL) jaiTableRemoveWhite(interned);

    int freedObjects = sweep(g);

    size_t after = gcLiveBytes(g);
    size_t freedBytes = before > after ? before - after : 0;

    g->nextGC = gcNextThreshold(g, after);
    jaiGCSyncLimit();
    g->collections++;
    g->totalFreed += freedBytes;
    g->totalPauseSeconds += jaiClockMonotonic() - started;

    if (verbose) {
        fprintf(stderr, "-- gc sweep: freed %zu bytes in %d objects\n",
                freedBytes, freedObjects);
        fprintf(stderr, "-- gc end: %zu -> %zu bytes (next at %zu)\n",
                before, after, g->nextGC);
    }

    jaiGCInCollect = false;
    jaiGCSyncLimit();
}

void jaiGCMaybeCollect(void) {
    GCState *g = activeGC();
    if (g == NULL || !g->enabled || jaiGCInCollect) return;
    if (gcStressOn(g) || gcLiveBytes(g) > g->nextGC) jaiGCCollect();
}

/* ------------------------------------------------------------------ */
/* Temporary roots                                                     */
/* ------------------------------------------------------------------ */

/* The cold growth half, kept separate so jaiGCPushRoot stays a check and a
 * store on the path that does not grow. */
static void jaiGCGrowRoots(void);

void jaiGCTrackObject(Obj *obj) {
    GCState *g = jaiGCActive;
    if (JAI_UNLIKELY(g == NULL)) JAI_PANIC("jaiGCTrackObject before jaiGCInit");
    /* Anything born during a collection is black on arrival: sweep is already
     * walking this list and would otherwise reclaim it the moment it is linked
     * in. Nothing in the collector allocates objects, so this is a safety net
     * rather than a normal path. */
    obj->isMarked = jaiGCInCollect;
    obj->next = g->objects;
    g->objects = obj;
}

void jaiGCPushRoot(Value v) {
    GCState *g = jaiGCActive;
    if (JAI_UNLIKELY(g == NULL)) JAI_PANIC("jaiGCPushRoot before jaiGCInit");
    if (JAI_UNLIKELY(g->tempRootCapacity < g->tempRootCount + 1))
        jaiGCGrowRoots();
    g->tempRoots[g->tempRootCount++] = v;
}

void jaiGCPopRoots(int n) {
    GCState *g = jaiGCActive;
    if (g == NULL || n <= 0) return;
    JAI_ASSERT(n <= g->tempRootCount, "unbalanced jaiGCPushRoot/jaiGCPopRoot");
    g->tempRootCount = n < g->tempRootCount ? g->tempRootCount - n : 0;
}

void jaiGCPopRoot(void) { jaiGCPopRoots(1); }

void jaiGCAddPermanentRoot(Value v) {
    GCState *g = jaiGCActive;
    if (JAI_UNLIKELY(g == NULL))
        JAI_PANIC("jaiGCAddPermanentRoot before jaiGCInit");
    if (g->permanentRootCapacity < g->permanentRootCount + 1) {
        int oldCapacity = g->permanentRootCapacity;
        int newCapacity = JAI_GROW_CAP(oldCapacity);
        if (newCapacity <= oldCapacity) JAI_PANIC("GC permanent-root overflow");
        g->permanentRoots = JAI_GROW_ARRAY(Value, g->permanentRoots, oldCapacity,
                                           newCapacity);
        g->permanentRootCapacity = newCapacity;
    }
    g->permanentRoots[g->permanentRootCount++] = v;
}

static void jaiGCGrowRoots(void) {
    GCState *g = jaiGCActive;
    int oldCapacity = g->tempRootCapacity;
    int newCapacity = JAI_GROW_CAP(oldCapacity);
    if (newCapacity <= oldCapacity) JAI_PANIC("GC temp-root stack overflow");
    g->tempRoots = JAI_GROW_ARRAY(Value, g->tempRoots, oldCapacity, newCapacity);
    g->tempRootCapacity = newCapacity;
}

/* ------------------------------------------------------------------ */
/* Statistics                                                          */
/* ------------------------------------------------------------------ */

void jaiGCPrintStats(FILE *out) {
    if (out == NULL) return;
    GCState *g = activeGC();
    if (g == NULL) {
        fprintf(out, "gc: not initialised\n");
        return;
    }

    double totalMs = g->totalPauseSeconds * 1000.0;
    double averageMs = g->collections > 0 ? totalMs / (double)g->collections : 0.0;

    fprintf(out, "gc statistics:\n");
    fprintf(out, "  collections     : %llu\n", (unsigned long long)g->collections);
    fprintf(out, "  live bytes      : %zu\n", gcLiveBytes(g));
    fprintf(out, "  next collection : %zu\n", g->nextGC);
    fprintf(out, "  total freed     : %llu bytes\n",
            (unsigned long long)g->totalFreed);
    fprintf(out, "  total pause     : %.3f ms\n", totalMs);
    fprintf(out, "  average pause   : %.3f ms\n", averageMs);
    fprintf(out, "  gray capacity   : %d\n", g->grayCapacity);
    fprintf(out, "  temp roots      : %d\n", g->tempRootCount);
}
