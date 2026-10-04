/* builtins_shape.c — the convex hull behind jaicv's shape analysis.
 *
 * `points_hull(points, clockwise)` is `convex_hull` in `shape/hull.jai`: Andrew's
 * monotone chain over the points sorted by x and then y, duplicates dropped,
 * the ring read off the chain above from the right and then the chain below
 * from the left, and reversed for `clockwise` -- the same ring, starting at the
 * same point, made of the caller's own point objects.
 *
 * Why it is here. A hull is asked for once a contour, and a contour is tens of
 * points; at that size the whole cost is constant overhead, and in Jaithon
 * that overhead was four function bodies, one of them too large for the
 * compiled tier's buffer, so it ran interpreted. A 74-point frame contour took
 * 3.6 microseconds against OpenCV's 1.5. It also stands under `min_area_rect`,
 * where it was most of that operation's time too.
 *
 * Anything this does not take -- fewer than three distinct points, a
 * coordinate too large for the turn test to be exact, or a point that is not
 * an object with integer `x` and `y` -- comes back as null, and `convex_hull`
 * goes its own way. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "runtime/builtins/media/points.h"
#include "runtime/builtins/text/builtins_str.h"
#include "runtime/runtime.h"

#include "vm/gc.h"

/* `HULL_COORD_LIMIT` in hull.jai: below it a difference fits 31 bits, a
 * product of two 62, and the turn test's difference of products 63. */
#define JAI_HULL_COORD_LIMIT 1073741824LL

typedef struct {
    int64_t x, y;
    int32_t index;
} HullPoint;

/* By x, then y, then where the point came in, which is what a stable sort by
 * x and y leaves -- and so which of two equal points is kept. */
JAI_INLINE bool hullBefore(const HullPoint *a, const HullPoint *b) {
    if (a->x != b->x) return a->x < b->x;
    if (a->y != b->y) return a->y < b->y;
    return a->index < b->index;
}

/* Insertion sort for short runs, then bottom-up merges. A contour is nearly
 * sorted in long stretches, which suits both. */
static void hullSort(HullPoint *items, HullPoint *spare, size_t count) {
    const size_t run = 16;
    for (size_t start = 0; start < count; start += run) {
        const size_t end = start + run < count ? start + run : count;
        for (size_t i = start + 1; i < end; i++) {
            const HullPoint held = items[i];
            size_t j = i;
            while (j > start && hullBefore(&held, &items[j - 1])) {
                items[j] = items[j - 1];
                j--;
            }
            items[j] = held;
        }
    }
    HullPoint *from = items;
    HullPoint *to = spare;
    for (size_t width = run; width < count; width *= 2) {
        for (size_t left = 0; left < count; left += 2 * width) {
            const size_t mid = left + width < count ? left + width : count;
            const size_t right = left + 2 * width < count ? left + 2 * width : count;
            size_t i = left, j = mid, k = left;
            while (i < mid && j < right) to[k++] = hullBefore(&from[j], &from[i]) ? from[j++] : from[i++];
            while (i < mid) to[k++] = from[i++];
            while (j < right) to[k++] = from[j++];
        }
        HullPoint *swap = from;
        from = to;
        to = swap;
    }
    if (from != items) memcpy(items, from, count * sizeof(HullPoint));
}

/* The sign of the turn o -> a -> b; `cross_sign` in hull.jai. */
JAI_INLINE int hullTurn(const HullPoint *o, const HullPoint *a, const HullPoint *b) {
    const int64_t value = (a->x - o->x) * (b->y - o->y) - (a->y - o->y) * (b->x - o->x);
    return (value > 0) - (value < 0);
}

static bool primPointsHull(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *points;
    if (!jaiArgList(args[0], 1, "points_hull", &points)) return false;
    if (!IS_BOOL(args[1])) {
        return jaiThrow(vm.cTypeError, "points_hull(): clockwise must be a bool");
    }
    const bool clockwise = AS_BOOL(args[1]);
    const int count = points->count;
    *out = NULL_VAL;
    if (count < 3) return true;

    /* One allocation for the points, the merge buffer and both chains. */
    HullPoint *items = (HullPoint *)malloc((size_t)count * 2 * sizeof(HullPoint) +
                                           (size_t)count * 2 * sizeof(int32_t) + 16);
    if (items == NULL) return jaiThrow(vm.cRuntimeError, "points_hull(): out of memory");
    HullPoint *spare = items + count;
    int32_t *lower = (int32_t *)(spare + count);
    int32_t *upper = lower + count;

    JaiPointReader reader;
    jaiPointReaderInit(&reader);
    for (int i = 0; i < count; i++) {
        int64_t x, y;
        if (!jaiReadPoint(&reader, jaiListGet(points, i), &x, &y) ||
            x <= -JAI_HULL_COORD_LIMIT || x >= JAI_HULL_COORD_LIMIT ||
            y <= -JAI_HULL_COORD_LIMIT || y >= JAI_HULL_COORD_LIMIT) {
            free(items);
            return true;
        }
        items[i].x = x;
        items[i].y = y;
        items[i].index = i;
    }
    hullSort(items, spare, (size_t)count);

    /* Duplicates out, keeping the first of each: `sorted_unique`. */
    int unique = 1;
    for (int i = 1; i < count; i++) {
        if (items[i].x == items[unique - 1].x && items[i].y == items[unique - 1].y) continue;
        items[unique++] = items[i];
    }
    if (unique < 3) {
        free(items);
        return true;
    }

    /* `hull_by_sorting`: the chain below left to right, the chain above right
     * to left, a turn that is not strictly one way popping the stack. */
    int low = 0;
    for (int i = 0; i < unique; i++) {
        while (low >= 2 && hullTurn(&items[lower[low - 2]], &items[lower[low - 1]], &items[i]) <= 0) {
            low--;
        }
        lower[low++] = i;
    }
    int high = 0;
    for (int i = unique - 1; i >= 0; i--) {
        while (high >= 2 && hullTurn(&items[upper[high - 2]], &items[upper[high - 1]], &items[i]) <= 0) {
            high--;
        }
        upper[high++] = i;
    }

    /* Each chain without its last point, the chain above first. */
    const int ring = (high - 1) + (low - 1);
    ObjList *made = jaiListNew(ring);
    if (made == NULL) {
        free(items);
        return false;
    }
    jaiGCPushRoot(OBJ_VAL(made));
    const bool reserved = jaiListReserveExact(made, ring);
    jaiGCPopRoot();
    if (!reserved) {
        free(items);
        return jaiThrow(vm.cRuntimeError, "points_hull(): out of memory");
    }
    Value *slots = jaiListBox(made);
    int at = 0;
    for (int i = 0; i < high - 1; i++) slots[at++] = jaiListGet(points, items[upper[i]].index);
    for (int i = 0; i < low - 1; i++) slots[at++] = jaiListGet(points, items[lower[i]].index);
    free(items);
    if (clockwise && ring > 1) {
        /* The first point stays first; the rest run the other way. */
        for (int i = 1, j = ring - 1; i < j; i++, j--) {
            const Value held = slots[i];
            slots[i] = slots[j];
            slots[j] = held;
        }
    }
    made->count = ring;
    jaiListTouch(made);
    *out = OBJ_VAL(made);
    return true;
}

void jaiShapeRegisterPrimitives(ObjModule *ns) {
    jaiStrDefinePrim(ns, "points_hull", primPointsHull, 2, 2);
}
