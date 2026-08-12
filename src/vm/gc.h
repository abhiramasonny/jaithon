/* gc.h — precise mark-sweep garbage collector (spec/BYTECODE.md §10). */
#ifndef JAI_GC_H
#define JAI_GC_H

#include "vm/value.h"

typedef struct { const Value *values; int count; } JaiGCRootRange;

typedef struct GCState {
    Obj      *objects;          /* intrusive list of every live object */
    Obj     **grayStack;
    int       grayCount;
    int       grayCapacity;

    size_t    nextGC;
    double    growFactor;       /* default 2.0 */
    size_t    minHeap;          /* never collect below this; default 1 MiB */

    /* Temporary roots pushed by C code that holds objects across allocations. */
    Value    *tempRoots;
    int       tempRootCount;
    int       tempRootCapacity;

    /* Ranges of Values already in contiguous memory the collector can't find
     * on its own (a compiled frame's JitCallDesc::roots). Pushing the range is
     * one entry regardless of count, instead of an O(roots) copy per call-out
     * into tempRoots; the descriptor outlives the call, so borrowing is sound. */
    JaiGCRootRange *rootRanges;
    int       rootRangeCount;
    int       rootRangeCapacity;

    /* Roots a native subsystem owns for the life of the VM. Never popped. */
    Value    *permanentRoots;
    int       permanentRootCount;
    int       permanentRootCapacity;

    bool      enabled;
    bool      stress;           /* collect on every allocation */
    bool      verbose;

    uint64_t  collections;
    uint64_t  totalFreed;
    double    totalPauseSeconds;
} GCState;

void jaiGCInit(GCState *gc);
void jaiGCFree(GCState *gc);

/* Called by jaiAllocateObject before handing out memory. */
void jaiGCMaybeCollect(void);
void jaiGCCollect(void);
void jaiGCEnable(bool enabled);

/* Collector internals, exported only so jaiGCWanted below can be inlined.
 * Nothing outside gc.c may write them. */
extern GCState *jaiGCActive;     /* NULL until jaiGCInit */
extern bool     jaiGCInCollect;  /* a collection must never start another */

/* One word standing in for the four-input test below, recomputed by
 * jaiGCSyncLimit() wherever an input changes -- inlining the four-input form
 * cost the interpreter's loop back edge seven dependent loads. */
extern size_t jaiGCLimit;

/* Conservative "a collection may be due" test, for the interpreter's loop
 * back edge (a call here cost 5.4% of a benchmark that never collects).
 * False proves jaiGCMaybeCollect would do nothing; true means ask it
 * properly. 0 means "ask every time", for a disabled/absent/stressed
 * collector. */
static inline bool jaiGCWanted(void) {
    return jaiHeapBytes > jaiGCLimit;
}

/* Recompute jaiGCLimit; call from anywhere that can change its inputs.
 * Forgetting one is benign: too high delays a collection, too low just
 * costs an extra no-op jaiGCMaybeCollect. */
void jaiGCSyncLimit(void);

/* Links a freshly built object into the collector's list. Deliberately out of
 * line: see the note on the root protocol below. */
void jaiGCTrackObject(Obj *obj);

void jaiGCMarkValue(Value v);
void jaiGCMarkObject(Obj *obj);
void jaiGCMarkArray(const ValueArray *a);

/* The already-black check moved to the call site: jaiGCMarkObject can't be a
 * leaf (graying may grow the gray stack), so its frame setup dominated
 * marking cost when most references are already black. Unlike jaiGCPushRoot
 * below, these run only from tracers, inside a collection, so inlining here
 * is cheap. */
JAI_INLINE void jaiGCMark(Obj *obj) {
    if (obj != NULL && !obj->isMarked) jaiGCMarkObject(obj);
}

JAI_INLINE void jaiGCMarkVal(Value v) {
    if (IS_OBJ(v)) jaiGCMark(AS_OBJ(v));
}

/* Temporary root protocol for C code:
 *     jaiGCPushRoot(v);  ... allocations ...  jaiGCPopRoot();
 */
/* Kept OUT OF LINE by measurement, not oversight: inlining these (a bounds
 * check + a store) was tried and was a consistent ~1-4% loss across repeated
 * A/B benchmark sweeps, from I-cache pressure at hot allocation sites.
 * Re-measure before inlining again. */
void jaiGCPushRoot(Value v);
void jaiGCPopRoots(int n);
/* Root a Value array in place, without copying it. `values` must stay valid
 * and unmoved until the matching pop. */
void jaiGCPushRootRange(const Value *values, int count);
void jaiGCPopRootRange(void);
void jaiGCPopRoot(void);

/* A root that lives as long as the VM does, for native code that parks a
 * Jaithon object in a C global -- publishing it under a name instead isn't a
 * root, since a program could drop that reference while native code still
 * uses it. */
void jaiGCAddPermanentRoot(Value v);

void jaiGCPrintStats(FILE *out);

#endif /* JAI_GC_H */
