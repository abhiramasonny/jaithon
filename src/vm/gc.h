#ifndef JAI_GC_H
#define JAI_GC_H

#include "vm/value.h"

typedef struct { const Value *values; int count; } JaiGCRootRange;

typedef struct GCState {
    Obj      *objects; //list of every live obj
    Obj     **grayStack;
    int       grayCount;
    int       grayCapacity;

    size_t    nextGC;
    double    growFactor; //the default for this is 2
    size_t    minHeap; //default 1 MiB

    Value    *tempRoots;
    int       tempRootCount;
    int       tempRootCapacity;

    JaiGCRootRange *rootRanges;
    int       rootRangeCount;
    int       rootRangeCapacity;

    Value    *permanentRoots;
    int       permanentRootCount;
    int       permanentRootCapacity;

    bool      enabled;
    bool      stress;
    bool      verbose;
    /* Stress cadence: collect every Nth allocation rather than every one.
     * 0 or 1 means every allocation, which is what --gc-stress has always
     * done. Higher N exists because N=1 is quadratic -- the unit suites run
     * in 5.89s plain and tests/lang alone does not finish in 10 minutes under
     * N=1, so no gate covered them at all. `stress` stays the on/off flag
     * because it is one of jaiGCSyncLimit's four inputs. */
    unsigned  stressEvery;
    unsigned  stressTick;

    uint64_t  collections;
    uint64_t  totalFreed;
    double    totalPauseSeconds;
} GCState;

void jaiGCInit(GCState *gc);
void jaiGCFree(GCState *gc);

void jaiGCMaybeCollect(void);
void jaiGCCollect(void);
void jaiGCEnable(bool enabled);

extern GCState *jaiGCActive;
extern bool     jaiGCInCollect;
extern size_t jaiGCLimit;

static inline bool jaiGCWanted(void) {
    return jaiHeapBytes > jaiGCLimit;
}

void jaiGCSyncLimit(void);
void jaiGCTrackObject(Obj *obj);

/* What `isMarked` holds for a marked object. It flips at the start of every
 * collection, so every object is unmarked the moment a collection begins
 * without anything having to visit it to say so: a survivor keeps the value it
 * was marked with, an object allocated since is given the current value, and
 * after the flip both read as unmarked. That is what lets the sweep leave a
 * survivor untouched -- it used to clear the bit on every live object -- and
 * what lets the page space below skip the objects it frees entirely. */
extern bool jaiGCEpoch;

JAI_INLINE bool jaiGCIsMarked(const Obj *obj) {
    return obj->isMarked == jaiGCEpoch;
}

void jaiGCMarkValue(Value v);
void jaiGCMarkObject(Obj *obj);
void jaiGCMarkArray(const ValueArray *a);

JAI_INLINE void jaiGCMark(Obj *obj) {
    if (obj != NULL && obj->isMarked != jaiGCEpoch) jaiGCMarkObject(obj);
}

/* ------------------------------------------------------------------ */
/* The page space                                                       */
/* ------------------------------------------------------------------ */

/* Small objects that own nothing but their own block -- instances, strings,
 * tuples, iterators, bound methods, ranges and the rest jaiObjSoleBlock sizes
 * -- live in 64 KiB pages of one size class each, carved from a single
 * reserved range so that "is this a page object" is one subtract and compare.
 * They are NOT on GCState.objects. A page keeps a mark bitmap with one bit per
 * 16-byte grain, which the marker sets beside `isMarked`; the sweep turns the
 * mark bitmap into the page's live bitmap and the allocator hands out whatever
 * is not live. So a dead page object is never visited at all: not by the sweep
 * (it was 80-95% of every pause, ~2-20ns per corpse chasing `next`), and not by
 * the allocator, which pops a bit instead of loading a free-list link from the
 * block it is about to hand out.
 *
 * Lists, dicts, sets and closures own an array or a table, so a dead one has
 * to be visited to free it. They live in FINALIZING pages of their own, which
 * also keep an in-use bitmap: the sweep visits exactly in-use-and-unmarked,
 * in address order, and never a survivor -- where the list sweep chased `next`
 * through every one of them, live or dead, at 85ns apiece on `check
 * lib/jaithon`. Everything rarer that owns something (functions, classes,
 * modules, enums, traits, files) stays on the list and is swept as before.
 *
 * JAITHON_GC_PAGES=0 sends every object down the list path again, and
 * JAITHON_GC_FIN_PAGES=0 just the finalizing kinds. */
#define JAI_PAGE_SHIFT  16u
#define JAI_PAGE_BYTES  ((size_t)1 << JAI_PAGE_SHIFT)
#define JAI_PAGE_GRAINS (JAI_PAGE_BYTES >> 4)
#define JAI_PAGE_WORDS  (JAI_PAGE_GRAINS / 64u)

typedef struct JaiPage {
    uint64_t        mark[JAI_PAGE_WORDS];   /* set by the marker this cycle */
    uint64_t        live[JAI_PAGE_WORDS];   /* marked at the last collection */
    /* Finalizing pages only: every block holding an object -- live at the last
     * collection or handed out since. Zero on any other page. */
    uint64_t        inuse[JAI_PAGE_WORDS];
    struct JaiPage *next;                   /* its class's list, or the pool */
    uint32_t        cls;                    /* grains per block; 0 in the pool */
    uint32_t        liveBlocks;
    uint32_t        fin;                    /* a finalizing page */
} JaiPage;

/* The allocator's position in one size class. The first two fields are the
 * whole fast path. */
typedef struct {
    uint64_t freeMask;   /* block starts in the current word not yet handed out */
    char    *wordBase;   /* address of the current word's first grain */
    JaiPage *page;       /* the page the current word belongs to */
    JaiPage *nextPage;   /* the next page of `pages` to look in */
    JaiPage *pages;      /* every page of this class */
    uint64_t handedOut;  /* blocks made available since the last collection */
    uint64_t stash;      /* free blocks of the current word held back, one at a
                          * time, while jaiGCLimit is 0 (see jaiPageRefill) */
    uint32_t word;       /* index of the current word in `page` */
    uint16_t cls;        /* grains per block, fixed per cursor */
    uint16_t fin;        /* serves finalizing pages */
} JaiPageCursor;

extern JaiPageCursor jaiPageCursor[JAI_SMALL_CLASSES + 1];
extern JaiPageCursor jaiPageCursorFin[JAI_SMALL_CLASSES + 1];
/* Which kinds may live in a page: JAI_PAGE_PLAIN, JAI_PAGE_FIN, or 0. All 0
 * when the page space is off, so the allocator's one table load is the whole
 * cost of the switch. */
#define JAI_PAGE_PLAIN 1u
#define JAI_PAGE_FIN   2u
extern uint8_t   jaiPageKind[OBJ_TYPE_COUNT];
extern uintptr_t jaiPageBase;
extern uintptr_t jaiPageSpan;   /* 0 when off: nothing is in range */

void *jaiPageRefill(JaiPageCursor *pc);

JAI_INLINE bool jaiInPageSpace(const void *p) {
    return (uintptr_t)p - jaiPageBase < jaiPageSpan;
}

/* A block of `cls` grains, or NULL when the page space cannot serve one right
 * now (switched off, out of range, or mid-collection). Not zeroed. Already
 * accounted: jaiHeapBytes is charged a whole word's blocks when the refill
 * hands the word out, so the pop below is all an allocation costs -- and a
 * caller that does not ask jaiGCWanted() first still meets the collector at
 * the next refill, at most one 1 KiB word later. */
JAI_INLINE void *jaiPageNewAt(JaiPageCursor *pc) {
    uint64_t m = pc->freeMask;
    if (JAI_LIKELY(m != 0)) {
        pc->freeMask = m & (m - 1);
        return pc->wordBase + ((size_t)__builtin_ctzll(m) << 4);
    }
    return jaiPageRefill(pc);
}

JAI_INLINE void *jaiPageNew(unsigned cls) {
    return jaiPageNewAt(&jaiPageCursor[cls]);
}

/* Sets the page's mark bit for a page object. The marker's job, nobody else's. */
JAI_INLINE void jaiPageMark(const Obj *obj) {
    uintptr_t a = (uintptr_t)obj;
    JaiPage *pg = (JaiPage *)(a & ~(uintptr_t)(JAI_PAGE_BYTES - 1));
    unsigned g = (unsigned)((a >> 4) & (JAI_PAGE_GRAINS - 1));
    pg->mark[g >> 6] |= (uint64_t)1 << (g & 63u);
}

/* Frees what a dead finalizing-page object owns (object.c). */
void jaiObjFinalize(Obj *obj);

void jaiPageSpaceInit(void);
/* Called at the start of a collection, before anything is marked. */
void jaiPageCollectBegin(void);
/* Called after the marker and the list sweep; returns the bytes freed. */
size_t jaiPageCollectEnd(void);
/* Teardown: every page object is dead. Returns the bytes that were in use. */
size_t jaiPageSpaceReset(void);

JAI_INLINE void jaiGCMarkVal(Value v) {
    if (IS_OBJ(v)) jaiGCMark(AS_OBJ(v));
}

void jaiGCPushRoot(Value v);
void jaiGCPopRoots(int n);
void jaiGCPushRootRange(const Value *values, int count);
void jaiGCPopRootRange(void);
void jaiGCPopRoot(void);

void jaiGCAddPermanentRoot(Value v);

void jaiGCPrintStats(FILE *out);

#endif /* JAI_GC_H */
