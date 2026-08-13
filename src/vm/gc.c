// gc.c is a precise mark-sweep collector

#include <stdlib.h>

#include "vm/gc.h"
#include "vm/bytecode/chunk.h"
#include "vm/object/object.h"
#include "vm/table.h"
#include "vm/vm.h"
#include "vm/jit/jit.h"

#define JAI_GC_DEFAULT_GROW_FACTOR 2.0
#define JAI_GC_DEFAULT_MIN_HEAP    ((size_t)1 << 20)

GCState *jaiGCActive;
bool jaiGCInCollect;
static GCState *activeGC(void) { return jaiGCActive != NULL ? jaiGCActive : vm.gc; }
size_t jaiGCLimit;

void jaiGCSyncLimit(void) {
    const GCState *g = jaiGCActive;
    jaiGCLimit = (g != NULL && g->enabled && !jaiGCInCollect && !g->stress)
                     ? g->nextGC
                     : 0;
}

/* Non-const: it advances the cadence counter. Called once per allocation on
 * the slow path, which stress mode pins open by holding jaiGCLimit at zero. */
static bool gcStressDue(GCState *g) {
    if (!(g->stress || vm.gcStress)) return false;
    unsigned every = g->stressEvery ? g->stressEvery : vm.gcStressEvery;
    if (every <= 1) return true;
    if (++g->stressTick < every) return false;
    g->stressTick = 0;
    return true;
}
static bool gcVerboseOn(const GCState *g) { return g->verbose || vm.debugGC; }

static size_t gcLiveBytes(const GCState *g) { (void)g; return jaiHeapBytes; }

static size_t gcNextThreshold(const GCState *g, size_t live) {
    double factor = g->growFactor > 1.0 ? g->growFactor : JAI_GC_DEFAULT_GROW_FACTOR;
    double target = (double)live * factor;
    size_t next = target >= (double)SIZE_MAX ? SIZE_MAX : (size_t)target;
    size_t floorBytes = g->minHeap != 0 ? g->minHeap : JAI_GC_DEFAULT_MIN_HEAP;
    return next > floorBytes ? next : floorBytes;
}

void jaiGCInit(GCState *gc) {
    if (gc == NULL) JAI_PANIC("jaiGCInit: NULL collector state");

    gc->objects = NULL;
    gc->grayStack = NULL;
    gc->grayCount = 0;
    gc->grayCapacity = 0;

    gc->growFactor = JAI_GC_DEFAULT_GROW_FACTOR;
    gc->minHeap = JAI_GC_DEFAULT_MIN_HEAP;
    gc->nextGC = gcNextThreshold(gc, jaiHeapBytes);

    gc->tempRoots = NULL;
    gc->tempRootCount = 0;
    gc->tempRootCapacity = 0;
    gc->rootRanges = NULL;
    gc->rootRangeCount = 0;
    gc->rootRangeCapacity = 0;

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
    JAI_FREE_ARRAY(JaiGCRootRange, gc->rootRanges,
                   gc->rootRangeCapacity);
    gc->rootRanges = NULL;
    gc->rootRangeCount = 0;
    gc->rootRangeCapacity = 0;

    JAI_FREE_ARRAY(Value, gc->permanentRoots, gc->permanentRootCapacity);
    gc->permanentRoots = NULL;
    gc->permanentRootCount = 0;
    gc->permanentRootCapacity = 0;

    if (jaiGCActive == gc) jaiGCActive = NULL;
    if (vm.gc == gc) vm.gc = NULL;
    jaiGCSyncLimit();

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

//marking

/* The collector half of the allocation census (see object.c); same build.
 *
 * `total pause` alone says a collection is expensive without saying which of
 * its three phases is. Split, it says the same thing on every benchmark that
 * allocates: the sweep is 80-95% of the pause, 94-98% of the objects it visits
 * are corpses, and it costs ~2.75ns each. That is the number any nursery
 * proposal has to be priced against, so it is worth being able to re-derive. */
#ifdef JAI_ALLOC_CENSUS
uint64_t jaiGCMarked, jaiGCSwept, jaiGCSweptDead;
double   jaiGCMarkSec, jaiGCInternSec, jaiGCSweepSec;
#endif

void jaiGCMarkObject(Obj *obj) {
    if (obj == NULL || obj->isMarked) return;
    obj->isMarked = true;
#ifdef JAI_ALLOC_CENSUS
    jaiGCMarked++;
#endif

    switch (obj->type) {
    case OBJ_STRING: {
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

        //this is grown with the raw system allocator, not jaiRealloc
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
    jaiTableMark(&c->restricted);

    for (int i = 0; i < (int)c->traitCount; i++) jaiGCMark((Obj *)c->traits[i]);
    for (int i = 0; i < (int)c->fieldCount; i++) jaiGCMark((Obj *)c->fields[i].name);

    jaiGCMarkVal(c->initializer);
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
        jaiGCMark((Obj *)v->unit);
        jaiGCMark((Obj *)v->ctor);
        markStrings(v->fieldNames, (int)v->arity);
    }
}

// trace one gray objs refrences
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

//roots

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
    jaiGCMarkVal(vm.pendingException);

    markInternedNames();
    markWellKnownClasses();

    markValues(g->tempRoots, g->tempRootCount);
    for (int i = 0; i < g->rootRangeCount; i++) {
        markValues(g->rootRanges[i].values, g->rootRanges[i].count);
    }
    markValues(g->permanentRoots, g->permanentRootCount);
}

static void traceReferences(GCState *g) {
    while (g->grayCount > 0) blackenObject(g->grayStack[--g->grayCount]);
}

//sweep

static int sweep(GCState *g) {
    int freedObjects = 0;
    size_t freedBytes = 0;
    Obj *previous = NULL;
    Obj *object = g->objects;

    while (object != NULL) {
#ifdef JAI_ALLOC_CENSUS
        jaiGCSwept++;
        if (!object->isMarked) jaiGCSweptDead++;
#endif
        if (object->isMarked) {
            object->isMarked = false; //white again for the next cycle
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
        /* The overwhelming majority of what a sweep frees owns nothing but its
         * own block -- 94-98% of the objects it visits are corpses on every
         * benchmark that allocates, and they are instances, strings, tuples,
         * iterators and ranges. Those go straight onto the bin here rather
         * than through jaiFreeObject, which is 550 instructions and saves six
         * register pairs on entry it does not need for any of them.
         * jaiObjSoleBlock is the same size jaiFreeObject would have used. */
        size_t sole = jaiObjSoleBlock(unreached);
        if (JAI_LIKELY(sole != 0 && jaiSmallServes(sole))) {
            jaiSmallDeleteUnaccounted(unreached, sole);
            freedBytes += sole;   /* charged once, after the loop */
        } else {
            jaiFreeObject(unreached);
        }
        freedObjects++;
    }

    jaiHeapAccountFreed(freedBytes);
    return freedObjects;
}

//collection stuff

void jaiGCCollect(void) {
    GCState *g = activeGC();
    if (g == NULL || !g->enabled || jaiGCInCollect) return;

    jaiGCInCollect = true;
    jaiGCSyncLimit();
    double started = jaiClockMonotonic();
    bool verbose = gcVerboseOn(g);
    size_t before = gcLiveBytes(g);
    if (verbose) fprintf(stderr, "-- gc begin\n");

#ifdef JAI_ALLOC_CENSUS
    double t0 = jaiClockMonotonic();
#endif
    markRoots(g);
    traceReferences(g);
#ifdef JAI_ALLOC_CENSUS
    double t1 = jaiClockMonotonic();
#endif

    JaiTable *interned = jaiInternTable();
    if (interned != NULL) jaiTableRemoveWhite(interned);
#ifdef JAI_ALLOC_CENSUS
    double t2 = jaiClockMonotonic();
#endif

    int freedObjects = sweep(g);
#ifdef JAI_ALLOC_CENSUS
    double t3 = jaiClockMonotonic();
    jaiGCMarkSec += t1 - t0;
    jaiGCInternSec += t2 - t1;
    jaiGCSweepSec += t3 - t2;
#endif

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
    if (gcStressDue(g) || gcLiveBytes(g) > g->nextGC) jaiGCCollect();
}

static void jaiGCGrowRoots(void);

void jaiGCTrackObject(Obj *obj) {
    GCState *g = jaiGCActive;
    if (JAI_UNLIKELY(g == NULL)) JAI_PANIC("jaiGCTrackObject before jaiGCInit");
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

void jaiGCPushRootRange(const Value *values, int count) {
    GCState *g = jaiGCActive;
    if (JAI_UNLIKELY(g == NULL))
        JAI_PANIC("jaiGCPushRootRange before jaiGCInit");
    if (JAI_UNLIKELY(g->rootRangeCapacity < g->rootRangeCount + 1)) {
        int oldCapacity = g->rootRangeCapacity;
        int newCapacity = JAI_GROW_CAP(oldCapacity);
        if (newCapacity <= oldCapacity) JAI_PANIC("GC root-range overflow");
        g->rootRanges = JAI_GROW_ARRAY(JaiGCRootRange, g->rootRanges,
                                       oldCapacity, newCapacity);
        g->rootRangeCapacity = newCapacity;
    }
    g->rootRanges[g->rootRangeCount].values = values;
    g->rootRanges[g->rootRangeCount].count  = count;
    g->rootRangeCount++;
}

void jaiGCPopRootRange(void) {
    GCState *g = jaiGCActive;
    if (g == NULL || g->rootRangeCount <= 0) return;
    g->rootRangeCount--;
}

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
#ifdef JAI_ALLOC_CENSUS
    fprintf(out, "  marked total    : %llu\n", (unsigned long long)jaiGCMarked);
    fprintf(out, "  swept total     : %llu\n", (unsigned long long)jaiGCSwept);
    fprintf(out, "  swept dead      : %llu\n", (unsigned long long)jaiGCSweptDead);
    fprintf(out, "  mark ms         : %.3f\n", jaiGCMarkSec * 1000.0);
    fprintf(out, "  intern ms       : %.3f\n", jaiGCInternSec * 1000.0);
    fprintf(out, "  sweep ms        : %.3f\n", jaiGCSweepSec * 1000.0);
#endif
    fprintf(out, "  gray capacity   : %d\n", g->grayCapacity);
    fprintf(out, "  temp roots      : %d\n", g->tempRootCount);
}
