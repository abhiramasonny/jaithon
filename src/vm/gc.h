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

void jaiGCMarkValue(Value v);
void jaiGCMarkObject(Obj *obj);
void jaiGCMarkArray(const ValueArray *a);

JAI_INLINE void jaiGCMark(Obj *obj) {
    if (obj != NULL && !obj->isMarked) jaiGCMarkObject(obj);
}

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
