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
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "runtime/builtins/media/points.h"
#include "runtime/builtins/text/builtins_str.h"
#include "runtime/runtime.h"

#include "vm/gc.h"

/* The floating point below has to round exactly as Jaithon's does, operation
 * by operation, and Jaithon never fuses a multiply into an add. */
#pragma STDC FP_CONTRACT OFF

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

/* The hull of `points` as positions in `items`, which comes back sorted, in
 * the order `convex_hull` gives it. 1 when it was taken, 0 when it is one this
 * does not take, -1 when memory ran out. `*itemsOut` is the caller's to free
 * whenever it is not NULL. */
static int hullRing(ObjList *points, bool clockwise, HullPoint **itemsOut, int32_t **ringOut,
                    int *ringCount) {
    *itemsOut = NULL;
    const int count = points->count;
    if (count < 3) return 0;

    /* One allocation for the points, the merge buffer, both chains and the
     * ring. */
    HullPoint *items = (HullPoint *)malloc((size_t)count * 2 * sizeof(HullPoint) +
                                           (size_t)count * 4 * sizeof(int32_t) + 16);
    if (items == NULL) return -1;
    *itemsOut = items;
    HullPoint *spare = items + count;
    int32_t *lower = (int32_t *)(spare + count);
    int32_t *upper = lower + count;
    int32_t *ring = upper + count;

    JaiPointReader reader;
    jaiPointReaderInit(&reader);
    for (int i = 0; i < count; i++) {
        int64_t x, y;
        if (!jaiReadPoint(&reader, jaiListGet(points, i), &x, &y) ||
            x <= -JAI_HULL_COORD_LIMIT || x >= JAI_HULL_COORD_LIMIT ||
            y <= -JAI_HULL_COORD_LIMIT || y >= JAI_HULL_COORD_LIMIT) {
            return 0;
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
    if (unique < 3) return 0;

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

    /* Each chain without its last point, the chain above first; for
     * `clockwise` the first point stays first and the rest run backwards. */
    int at = 0;
    for (int i = 0; i < high - 1; i++) ring[at++] = upper[i];
    for (int i = 0; i < low - 1; i++) ring[at++] = lower[i];
    if (clockwise && at > 1) {
        for (int i = 1, j = at - 1; i < j; i++, j--) {
            const int32_t held = ring[i];
            ring[i] = ring[j];
            ring[j] = held;
        }
    }
    *ringOut = ring;
    *ringCount = at;
    return 1;
}

static bool primPointsHull(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *points;
    if (!jaiArgList(args[0], 1, "points_hull", &points)) return false;
    if (!IS_BOOL(args[1])) {
        return jaiThrow(vm.cTypeError, "points_hull(): clockwise must be a bool");
    }
    *out = NULL_VAL;
    HullPoint *items;
    int32_t *ring = NULL;
    int count = 0;
    const int taken = hullRing(points, AS_BOOL(args[1]), &items, &ring, &count);
    if (taken <= 0) {
        free(items);
        return taken == 0 ? true : jaiThrow(vm.cRuntimeError, "points_hull(): out of memory");
    }
    ObjList *made = jaiListNew(count);
    if (made == NULL) {
        free(items);
        return false;
    }
    jaiGCPushRoot(OBJ_VAL(made));
    const bool reserved = jaiListReserveExact(made, count);
    jaiGCPopRoot();
    if (!reserved) {
        free(items);
        return jaiThrow(vm.cRuntimeError, "points_hull(): out of memory");
    }
    Value *slots = jaiListBox(made);
    for (int i = 0; i < count; i++) slots[i] = jaiListGet(points, items[ring[i]].index);
    free(items);
    made->count = count;
    jaiListTouch(made);
    *out = OBJ_VAL(made);
    return true;
}

/* `points_min_box(points, out)` -- `min_area_rect` in shape/enclosing.jai, the
 * smallest rotated rectangle around the points: the hull, then for each hull
 * edge the box flush with it, the smallest winning and a near tie going to
 * the edge nearest horizontal, then OpenCV's quarter turns into [-90, 0).
 * Every floating point operation is the one `best_box` and `min_area_rect`
 * perform, in their order, so the box is the same to the bit.
 *
 * Writes the centre's x and y, the width and height, and the angle into the
 * first five elements of `out` and returns true; returns false, writing
 * nothing, for a set `points_hull` would not take. */
static bool primPointsMinBox(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *points, *box;
    if (!jaiArgList(args[0], 1, "points_min_box", &points)) return false;
    if (!jaiArgList(args[1], 2, "points_min_box", &box)) return false;
    if (box->count < 5) {
        return jaiThrow(vm.cValueError, "points_min_box(): the box list holds %d of 5 values",
                        box->count);
    }
    HullPoint *items;
    int32_t *ring = NULL;
    int count = 0;
    const int taken = hullRing(points, false, &items, &ring, &count);
    if (taken <= 0) {
        free(items);
        if (taken < 0) return jaiThrow(vm.cRuntimeError, "points_min_box(): out of memory");
        *out = BOOL_VAL(false);
        return true;
    }

    /* `hull_floats`, then `best_box`. */
    double *hx = (double *)malloc((size_t)count * 2 * sizeof(double));
    if (hx == NULL) {
        free(items);
        return jaiThrow(vm.cRuntimeError, "points_min_box(): out of memory");
    }
    double *hy = hx + count;
    for (int i = 0; i < count; i++) {
        hx[i] = (double)items[ring[i]].x;
        hy[i] = (double)items[ring[i]].y;
    }
    free(items);

    double bestArea = 1e300;
    double bestFa = 1.0;
    double bestFb = -1.0;
    double found[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int index = 0; index < count; index++) {
        const int after = index + 1 == count ? 0 : index + 1;
        const double dx = hx[after] - hx[index];
        const double dy = hy[after] - hy[index];
        const double length = sqrt(dx * dx + dy * dy);
        if (length <= 1e-12) continue;
        const double ux = dx / length;
        const double uy = dy / length;
        double minU = 1e300, maxU = -1e300, minV = 1e300, maxV = -1e300;
        for (int at = 0; at < count; at++) {
            const double u = hx[at] * ux + hy[at] * uy;
            if (u < minU) minU = u;
            if (u > maxU) maxU = u;
            const double v = hy[at] * ux - hx[at] * uy;
            if (v < minV) minV = v;
            if (v > maxV) maxV = v;
        }
        const double area = (maxU - minU) * (maxV - minV);
        double fa = dx;
        double fb = dy;
        if (dx <= 0.0) {
            if (dy > 0.0) {
                fa = dy;
                fb = -dx;
            } else {
                fa = -dx;
                fb = -dy;
            }
        } else if (dy < 0.0) {
            fa = -dy;
            fb = dx;
        }
        bool take = false;
        if (area < bestArea - 1e-9) {
            take = true;
        } else if (area < bestArea + 1e-9) {
            if (fb * bestFa > bestFb * fa) take = true;
        }
        if (take) {
            if (area < bestArea) bestArea = area;
            bestFa = fa;
            bestFb = fb;
            found[0] = minU;
            found[1] = maxU;
            found[2] = minV;
            found[3] = maxV;
            found[4] = dx;
            found[5] = dy;
        }
    }
    free(hx);

    const double angle0 = atan2(found[5], found[4]);
    const double ux = cos(angle0);
    const double uy = sin(angle0);
    const double cu = (found[0] + found[1]) * 0.5;
    const double cv = (found[2] + found[3]) * 0.5;
    const double cx = cu * ux - cv * uy;
    const double cy = cu * uy + cv * ux;
    double width = found[1] - found[0];
    double height = found[3] - found[2];
    double angle = angle0 * 180.0 / 3.141592653589793;
    while (angle >= 0.0) {
        angle -= 90.0;
        const double swap = width;
        width = height;
        height = swap;
    }
    while (angle < -90.0) {
        angle += 90.0;
        const double swap = width;
        width = height;
        height = swap;
    }

    const double values[5] = {cx, cy, width, height, angle};
    for (int i = 0; i < 5; i++) jaiListPut(box, i, FLOAT_VAL(values[i]));
    jaiListTouch(box);
    *out = BOOL_VAL(true);
    return true;
}

void jaiShapeRegisterPrimitives(ObjModule *ns) {
    jaiStrDefinePrim(ns, "points_hull",    primPointsHull,   2, 2);
    jaiStrDefinePrim(ns, "points_min_box", primPointsMinBox, 2, 2);
}
