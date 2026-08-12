/* object_internal.h — plumbing shared by the object_*.c files that implement
 * the heap object model, but with no business appearing in object.h: nothing
 * outside this directory calls either of these.
 *
 * Both are pure/stateless and safe to duplicate per translation unit, so both
 * are `static inline` here rather than living in exactly one of the .c files
 * with an extra declaration for the others to reach them through.
 */
#ifndef JAI_OBJECT_INTERNAL_H
#define JAI_OBJECT_INTERNAL_H

#include "object.h"
#include "gc.h"   /* jaiGCPushRoot, for pushObjRoot below */

/* Push `o` as a temporary GC root, tolerating NULL so that push/pop stay
 * paired in constructors whose arguments are optional. Used by every ObjX
 * constructor that roots an argument across an allocation that can collect. */
static inline void pushObjRoot(void *o) {
    jaiGCPushRoot(o != NULL ? OBJ_VAL(o) : NULL_VAL);
}

/* Clamps [start, stop) / step against a sequence of `n` items, rewriting all
 * three in place, and returns how many items the slice selects. `step` must be
 * nonzero. Clamping |step| to n is safe (a step larger than the sequence can
 * select at most one item) and keeps `start + i * step` from overflowing.
 *
 * Shared by jaiStringSlice (object_string.c) and jaiListSlice
 * (object_collection.c) — the two container kinds sliceable with Python
 * step semantics. Neither type's slicing logic is otherwise related to the
 * other's, so this is the only thing they share. */
static inline int64_t sliceCount(int64_t n, int64_t *pStart,
                                 int64_t *pStop, int64_t *pStep) {
    int64_t start = *pStart;
    int64_t stop = *pStop;
    int64_t step = *pStep;

    if (n <= 0) {
        *pStart = 0;
        *pStop = 0;
        *pStep = step > 0 ? 1 : -1;
        return 0;
    }

    if (step == 1) {
        if (start < 0) start = start < -n ? 0 : start + n;
        else if (start > n) start = n;
        if (stop < 0) stop = stop < -n ? 0 : stop + n;
        else if (stop > n) stop = n;

        const int64_t count = stop > start ? stop - start : 0;
        *pStart = start;
        *pStop = stop;
        *pStep = 1;
        return count;
    }

    if (step == -1) {
        if (start < 0) start = start < -n ? -1 : start + n;
        else if (start >= n) start = n - 1;
        if (stop < 0) stop = stop < -n ? -1 : stop + n;
        else if (stop >= n) stop = n - 1;

        const int64_t count = start > stop ? start - stop : 0;
        *pStart = start;
        *pStop = stop;
        *pStep = -1;
        return count;
    }

    if (step > n) step = n;
    if (step < -n) step = -n;

    int64_t count;
    if (step > 0) {
        if (start < 0) start = start < -n ? 0 : start + n;
        else if (start > n) start = n;
        if (stop < 0) stop = stop < -n ? 0 : stop + n;
        else if (stop > n) stop = n;

        if (stop > start) {
            const uint64_t span = (uint64_t)(stop - start);
            const uint64_t stride = (uint64_t)step;
            count = (int64_t)(span / stride + (span % stride != 0));
        } else {
            count = 0;
        }
    } else {
        if (start < 0) start = start < -n ? -1 : start + n;
        else if (start >= n) start = n - 1;
        if (stop < 0) stop = stop < -n ? -1 : stop + n;
        else if (stop >= n) stop = n - 1;

        if (start > stop) {
            const uint64_t span = (uint64_t)(start - stop);
            const uint64_t stride = (uint64_t)(-step);
            count = (int64_t)(span / stride + (span % stride != 0));
        } else {
            count = 0;
        }
    }

    *pStart = start;
    *pStop = stop;
    *pStep = step;
    return count;
}

#endif /* JAI_OBJECT_INTERNAL_H */
