/* gc.h — precise mark-sweep garbage collector (spec/BYTECODE.md §10). */
#ifndef JAI_GC_H
#define JAI_GC_H

#include "value.h"

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

/* One word standing in for the four inputs of the test below, recomputed by
 * jaiGCSyncLimit() wherever any of them changes. Inlining the four-input form
 * cost the loop back edge seven dependent loads -- jaiGCActive, ->enabled,
 * jaiGCInCollect, ->stress, ->nextGC and the heap total -- on the hottest path
 * in the interpreter, which the comment there described as "one predictable
 * branch". It was not. */
extern size_t jaiGCLimit;

/* Conservative "a collection may be due" test, for the interpreter's loop
 * back edge. Answering it through a call cost 5.4% of a benchmark that never
 * collects. False is a proof that jaiGCMaybeCollect would do nothing; true
 * only means the caller must ask it properly. A limit of 0 means "ask
 * properly every time", which is what a disabled, absent or stressed
 * collector needs; it is exact for every heap that holds anything at all. */
static inline bool jaiGCWanted(void) {
    return jaiHeapBytes > jaiGCLimit;
}

/* Recompute jaiGCLimit. Called from every place that can change any of its
 * inputs. Forgetting one is benign in one direction only: too high a limit
 * delays a collection, too low a one costs a jaiGCMaybeCollect that decides
 * to do nothing. */
void jaiGCSyncLimit(void);

/* Links a freshly built object into the collector's list. Deliberately out of
 * line: see the note on the root protocol below. */
void jaiGCTrackObject(Obj *obj);

void jaiGCMarkValue(Value v);
void jaiGCMarkObject(Obj *obj);
void jaiGCMarkArray(const ValueArray *a);

/* The two tests that decide a reference needs nothing done to it, moved to the
 * call site.
 *
 * A tracer touches every reference every reachable object holds, and the great
 * majority of them are already black by the time it gets there: the class each
 * of a million instances names, the name each of its methods names, the node
 * whichever subtree reached it first. jaiGCMarkObject cannot be a leaf -- it
 * grays, and graying can grow the gray stack -- so it sets up a four-register
 * frame before it is in a position to look at the byte that says there is
 * nothing to do. On binary_trees, whose live set is a quarter of a million
 * instances, that frame was most of what marking cost.
 *
 * This is NOT the inlining that gc.h warns about above. jaiGCPushRoot is
 * called from allocation sites, which are hot in every program and everywhere;
 * these are called only from tracers, which run only inside a collection. */
JAI_INLINE void jaiGCMark(Obj *obj) {
    if (obj != NULL && !obj->isMarked) jaiGCMarkObject(obj);
}

JAI_INLINE void jaiGCMarkVal(Value v) {
    if (IS_OBJ(v)) jaiGCMark(AS_OBJ(v));
}

/* Temporary root protocol for C code:
 *     jaiGCPushRoot(v);  ... allocations ...  jaiGCPopRoot();
 */
/* These stay OUT OF LINE, and that is a measured decision rather than an
 * oversight. They were moved into this header as `static inline` on the
 * argument that the bodies are a bounds check and a store and that natives run
 * the push/pop pair once per container element. Re-measured against the tree
 * as it now stands -- after the f-string opcode and the bound-method work
 * removed most of the surrounding allocation traffic -- inlining them is a
 * consistent LOSS. Two interleaved A/B sweeps of exactly this change against
 * the inlined build, 25 reps each, put the inlined build at 1.039x dict_ops,
 * 1.023x list_ops, 1.031x string_build and 1.009x overall; two further sweeps
 * using __attribute__((noinline)) rather than this layout agree, and the sign
 * was stable in every replicate. Inlining a bounds check and a store into
 * every allocation site pays for the saved call with I-cache pressure in
 * callers that are themselves hot. Re-measure before inlining them again. */
void jaiGCPushRoot(Value v);
void jaiGCPopRoots(int n);
void jaiGCPopRoot(void);

/* A root that lives as long as the VM does. Native code that parks a Jaithon
 * object in a C global registers it here; the alternative of publishing the
 * object under a name is not a root at all, because a program can then reach
 * the name and drop the reference while the native side still uses it. */
void jaiGCAddPermanentRoot(Value v);

void jaiGCPrintStats(FILE *out);

#endif /* JAI_GC_H */
