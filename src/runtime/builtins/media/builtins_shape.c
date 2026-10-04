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

/* The cosine and the sine as the two library calls Jaithon's `math.cos` and
 * `math.sin` make. Written plainly, a cosine and a sine of the same angle are
 * fused by the compiler into one `__sincos_stret`, and that does not always
 * return the same bits as the two functions: a reweighted line fit came out
 * one unit in the last place off. Calling through a volatile pointer keeps
 * them two calls. */
static double shapeCos(double x) {
    double (*volatile f)(double) = cos;
    return f(x);
}

static double shapeSin(double x) {
    double (*volatile f)(double) = sin;
    return f(x);
}

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

/* Columns, kept between calls, for putting points in order without sorting
 * them. A bit a column says whether any point landed there, and for each
 * column that has one, its lowest and its highest point -- the earliest of
 * any that tie, which is the one a stable sort keeps. Only the bits are
 * cleared afterwards; a column's two points are written fresh by the first
 * point to land in it. */
typedef struct {
    uint64_t *occupied;
    HullPoint *lowest;
    HullPoint *highest;
    size_t width;
} HullColumns;

static _Thread_local HullColumns tColumns;

static bool hullColumnsReserve(HullColumns *c, size_t width) {
    if (c->width >= width) return true;
    const size_t words = (width + 63) / 64;
    uint64_t *occupied = (uint64_t *)calloc(words, sizeof(uint64_t));
    HullPoint *lowest = (HullPoint *)malloc(width * sizeof(HullPoint));
    HullPoint *highest = (HullPoint *)malloc(width * sizeof(HullPoint));
    if (occupied == NULL || lowest == NULL || highest == NULL) {
        free(occupied);
        free(lowest);
        free(highest);
        return false;
    }
    free(c->occupied);
    free(c->lowest);
    free(c->highest);
    c->occupied = occupied;
    c->lowest = lowest;
    c->highest = highest;
    c->width = width;
    return true;
}

/* The points in sorted order with every point strictly between the lowest and
 * the highest of its column left out, written to `items`; returns how many.
 *
 * Why leaving them out changes nothing. The chain below meets a column's
 * points from the lowest up: each one above the lowest is pushed straight
 * after it, the next one up is collinear with the two before and pops it, and
 * the first point of the next column turns clockwise from the column's top
 * and pops whatever is above the lowest. So no point strictly inside a
 * column's span is ever on the chain when anything else is decided, and the
 * stack the next column meets is the one it meets without them; the chain
 * above is the same argument turned round. The two kept are the earliest of
 * any ties, as `sorted_unique` keeps them. */
static int hullByColumns(const HullPoint *read, int count, int64_t left, size_t width,
                         HullPoint *items) {
    HullColumns *c = &tColumns;
    for (int i = 0; i < count; i++) {
        const HullPoint *p = &read[i];
        const size_t column = (size_t)(p->x - left);
        uint64_t *word = &c->occupied[column >> 6];
        const uint64_t bit = (uint64_t)1 << (column & 63);
        if ((*word & bit) == 0) {
            *word |= bit;
            c->lowest[column] = *p;
            c->highest[column] = *p;
            continue;
        }
        /* Points come in by index, so the first of a tie is already held. */
        if (p->y < c->lowest[column].y) c->lowest[column] = *p;
        if (p->y > c->highest[column].y) c->highest[column] = *p;
    }
    int at = 0;
    const size_t words = (width + 63) / 64;
    for (size_t w = 0; w < words; w++) {
        uint64_t bits = c->occupied[w];
        c->occupied[w] = 0;
        while (bits != 0) {
            const size_t column = w * 64 + (size_t)__builtin_ctzll(bits);
            bits &= bits - 1;
            items[at++] = c->lowest[column];
            if (c->highest[column].y != c->lowest[column].y) items[at++] = c->highest[column];
        }
    }
    return at;
}

/* The sign of the turn o -> a -> b; `cross_sign` in hull.jai. */
JAI_INLINE int hullTurn(const HullPoint *o, const HullPoint *a, const HullPoint *b) {
    const int64_t value = (a->x - o->x) * (b->y - o->y) - (a->y - o->y) * (b->x - o->x);
    return (value > 0) - (value < 0);
}

/* Whether the hull puts points in order by column rather than by sorting.
 * Read once; JAICV_HULL_COLUMNS=0 sorts every time, which the column order is
 * checked against. */
static bool jaiHullColumnsOn(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("JAICV_HULL_COLUMNS");
        on = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return on == 1;
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
    int64_t left = 0, right = 0;
    for (int i = 0; i < count; i++) {
        int64_t x, y;
        if (!jaiReadPoint(&reader, jaiListGet(points, i), &x, &y) ||
            x <= -JAI_HULL_COORD_LIMIT || x >= JAI_HULL_COORD_LIMIT ||
            y <= -JAI_HULL_COORD_LIMIT || y >= JAI_HULL_COORD_LIMIT) {
            return 0;
        }
        if (i == 0 || x < left) left = x;
        if (i == 0 || x > right) right = x;
        items[i].x = x;
        items[i].y = y;
        items[i].index = i;
    }

    /* By column when the span is narrow enough to hold -- a contour's always
     * is, being pixels -- and by sorting otherwise. Sorting was half of what
     * a 74-point frame contour cost; the columns of that one are 1920 bits to
     * walk. */
    int unique;
    const size_t width = (size_t)(right - left) + 1;
    if (jaiHullColumnsOn() && width <= (size_t)count * 64 + 8192 &&
        hullColumnsReserve(&tColumns, width)) {
        memcpy(spare, items, (size_t)count * sizeof(HullPoint));
        unique = hullByColumns(spare, count, left, width, items);
    } else {
        hullSort(items, spare, (size_t)count);
        /* Duplicates out, keeping the first of each: `sorted_unique`. */
        unique = 1;
        for (int i = 1; i < count; i++) {
            if (items[i].x == items[unique - 1].x && items[i].y == items[unique - 1].y) continue;
            items[unique++] = items[i];
        }
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
    const double ux = shapeCos(angle0);
    const double uy = shapeSin(angle0);
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

/* --- curves --------------------------------------------------------- */

/* The points of `curve` as doubles, `float(point.x)` and `float(point.y)` in
 * Jaithon. NULL, having set `*readable` false, when a point is not one. */
static double *curveFloats(ObjList *curve, bool *readable) {
    const int count = curve->count;
    *readable = true;
    double *xy = (double *)malloc((size_t)(count > 0 ? count : 1) * 2 * sizeof(double));
    if (xy == NULL) return NULL;
    JaiPointReader reader;
    jaiPointReaderInit(&reader);
    for (int i = 0; i < count; i++) {
        int64_t x, y;
        if (!jaiReadPoint(&reader, jaiListGet(curve, i), &x, &y)) {
            *readable = false;
            free(xy);
            return NULL;
        }
        xy[i * 2] = (double)x;
        xy[i * 2 + 1] = (double)y;
    }
    return xy;
}

/* `furthest_from` in approx.jai: the point furthest from `origin`, the
 * earliest of a tie. */
static int curveFurthest(const double *xy, int count, int origin) {
    const double ox = xy[origin * 2];
    const double oy = xy[origin * 2 + 1];
    int far = 0;
    double furthest = -1.0;
    for (int index = 0; index < count; index++) {
        const double dx = xy[index * 2] - ox;
        const double dy = xy[index * 2 + 1] - oy;
        const double span = dx * dx + dy * dy;
        if (span > furthest) {
            furthest = span;
            far = index;
        }
    }
    return far;
}

/* One entry of `curveSimplify`'s walk: a stretch of the curve still to
 * simplify, from `first` to `last`, or, when `first` is negative, the point
 * at position `last` in `curve`, to keep. */
typedef struct {
    int first;
    int last;
} CurveSpan;

/* `douglas_peucker` in approx.jai, appending the kept points' positions in
 * `curve` to `kept`. The recursion is the same and so is the order of the
 * points it keeps: the stretch before the worst point, the worst point, the
 * stretch after it. Its stack is a heap array rather than C frames, since a
 * curve such as a spiral nests as deep as it has points and the C stack ran
 * out where the Jaithon raised a RecursionError. False when that array cannot
 * grow. */
static bool curveSimplify(const double *xy, int count, int start, int first, int last,
                          int32_t *kept, int *keptCount, double epsilon) {
    size_t capacity = 64;
    size_t depth = 0;
    CurveSpan *stack = (CurveSpan *)malloc(capacity * sizeof(CurveSpan));
    if (stack == NULL) return false;
    stack[depth++] = (CurveSpan){first, last};
    while (depth > 0) {
        const CurveSpan span = stack[--depth];
        if (span.first < 0) {
            kept[(*keptCount)++] = span.last;
            continue;
        }
        first = span.first;
        last = span.last;
        if (last <= first + 1) continue;
        int here = start + first;
        if (here >= count) here -= count;
        const double ax = xy[here * 2];
        const double ay = xy[here * 2 + 1];
        here = start + last;
        if (here >= count) here -= count;
        const double dx = xy[here * 2] - ax;
        const double dy = xy[here * 2 + 1] - ay;
        const double squared = dx * dx + dy * dy;
        double worst = 0.0;
        int worstAt = first;
        for (int index = first + 1; index < last; index++) {
            here = start + index;
            if (here >= count) here -= count;
            const double px = xy[here * 2];
            const double py = xy[here * 2 + 1];
            double gap = (px - ax) * (px - ax) + (py - ay) * (py - ay);
            if (squared > 1e-12) {
                const double dot = (px - ax) * dx + (py - ay) * dy;
                if (dot > 0.0) {
                    double t = 1.0;
                    if (dot < squared) t = dot / squared;
                    const double cx = ax + t * dx;
                    const double cy = ay + t * dy;
                    gap = (px - cx) * (px - cx) + (py - cy) * (py - cy);
                }
            }
            if (gap > worst) {
                worst = gap;
                worstAt = index;
            }
        }
        if (sqrt(worst) <= epsilon) continue;
        if (depth + 3 > capacity) {
            capacity *= 2;
            CurveSpan *grown = (CurveSpan *)realloc(stack, capacity * sizeof(CurveSpan));
            if (grown == NULL) {
                free(stack);
                return false;
            }
            stack = grown;
        }
        here = start + worstAt;
        if (here >= count) here -= count;
        /* Last in, first out: the stretch before the worst point is walked
         * whole before the point is kept and the stretch after it begun. */
        stack[depth++] = (CurveSpan){worstAt, last};
        stack[depth++] = (CurveSpan){-1, here};
        stack[depth++] = (CurveSpan){first, worstAt};
    }
    free(stack);
    return true;
}

/* `points_approx(curve, epsilon, closed)` -- `approx_poly_dp` for a curve of
 * three or more points and a tolerance that is not negative, which the caller
 * has checked: the curve cut at OpenCV's two furthest points if it is closed,
 * each half simplified by Douglas and Peucker, the result made of the
 * caller's own points in OpenCV's rotation. Null when a point is not an
 * object with integer `x` and `y`. */
static bool primPointsApprox(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *curve;
    if (!jaiArgList(args[0], 1, "points_approx", &curve)) return false;
    if (!IS_FLOAT(args[1]) || !IS_BOOL(args[2])) {
        return jaiThrow(vm.cTypeError, "points_approx(): wants a float tolerance and a bool");
    }
    const double epsilon = AS_FLOAT(args[1]);
    const bool closed = AS_BOOL(args[2]);
    const int count = curve->count;
    *out = NULL_VAL;
    if (count < 3 || !(epsilon >= 0.0)) return true;
    bool readable;
    double *xy = curveFloats(curve, &readable);
    if (xy == NULL) {
        return readable ? jaiThrow(vm.cRuntimeError, "points_approx(): out of memory") : true;
    }
    int32_t *kept = (int32_t *)malloc((size_t)(count + 2) * sizeof(int32_t));
    if (kept == NULL) {
        free(xy);
        return jaiThrow(vm.cRuntimeError, "points_approx(): out of memory");
    }
    int start = 0;
    if (closed) start = curveFurthest(xy, count, curveFurthest(xy, count, 0));
    const int last = closed ? count : count - 1;
    int keptCount = 0;
    kept[keptCount++] = start;
    const bool simplified = curveSimplify(xy, count, start, 0, last, kept, &keptCount, epsilon);
    free(xy);
    if (!simplified) {
        free(kept);
        return jaiThrow(vm.cRuntimeError, "points_approx(): out of memory");
    }
    if (!closed) kept[keptCount++] = count - 1;

    ObjList *made = jaiListNew(keptCount);
    if (made == NULL) {
        free(kept);
        return false;
    }
    jaiGCPushRoot(OBJ_VAL(made));
    const bool reserved = jaiListReserveExact(made, keptCount);
    jaiGCPopRoot();
    if (!reserved) {
        free(kept);
        return jaiThrow(vm.cRuntimeError, "points_approx(): out of memory");
    }
    Value *slots = jaiListBox(made);
    for (int i = 0; i < keptCount; i++) slots[i] = jaiListGet(curve, kept[i]);
    free(kept);
    made->count = keptCount;
    jaiListTouch(made);
    *out = OBJ_VAL(made);
    return true;
}

/* --- the enclosing circle ------------------------------------------- */

/* `points_min_circle(points, out)` -- `min_enclosing_circle` in
 * shape/enclosing.jai for two or more points: the same fixed shuffle, the
 * same widest pair moved to the front, Welzl's three nested walks with the
 * same slack, and the farthest point as the final say on the radius. Every
 * floating point operation is the one written there, in its order, so the
 * circle is the same to the bit.
 *
 * Writes the centre's x and y and the radius into the first three elements of
 * `out` and returns true; false, writing nothing, when a point is not an
 * object with integer `x` and `y` or a difference of two coordinates would
 * overflow, both of which the Jaithon then meets for itself. */
static bool primPointsMinCircle(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *points, *circle;
    if (!jaiArgList(args[0], 1, "points_min_circle", &points)) return false;
    if (!jaiArgList(args[1], 2, "points_min_circle", &circle)) return false;
    if (circle->count < 3) {
        return jaiThrow(vm.cValueError, "points_min_circle(): the circle list holds %d of 3 values",
                        circle->count);
    }
    const int count = points->count;
    *out = BOOL_VAL(false);
    if (count < 2) return true;
    int64_t *xs = (int64_t *)malloc((size_t)count * 2 * sizeof(int64_t));
    if (xs == NULL) return jaiThrow(vm.cRuntimeError, "points_min_circle(): out of memory");
    int64_t *ys = xs + count;
    JaiPointReader reader;
    jaiPointReaderInit(&reader);
    for (int i = 0; i < count; i++) {
        if (!jaiReadPoint(&reader, jaiListGet(points, i), &xs[i], &ys[i])) {
            free(xs);
            return true;
        }
    }

    /* `shuffle_points`: Fisher-Yates from the C standard's sample `rand`. */
    {
        int64_t state = 1;
        for (int index = count - 1; index > 0; index--) {
            state = (state * 1103515245 + 12345) % 2147483648LL;
            const int pick = (int)(state % (index + 1));
            const int64_t hx = xs[index], hy = ys[index];
            xs[index] = xs[pick];
            ys[index] = ys[pick];
            xs[pick] = hx;
            ys[pick] = hy;
        }
    }

    /* `lead_with_extremes`: the wider of the two extreme spans to the front. */
    {
        int64_t minX = xs[0], maxX = xs[0], minY = ys[0], maxY = ys[0];
        int left = 0, right = 0, top = 0, bottom = 0;
        for (int index = 1; index < count; index++) {
            const int64_t x = xs[index], y = ys[index];
            if (x < minX) {
                minX = x;
                left = index;
            }
            if (x > maxX) {
                maxX = x;
                right = index;
            }
            if (y < minY) {
                minY = y;
                top = index;
            }
            if (y > maxY) {
                maxY = y;
                bottom = index;
            }
        }
        int64_t wideI, tallI, acrossYI, acrossXI;
        if (__builtin_sub_overflow(maxX, minX, &wideI) || __builtin_sub_overflow(maxY, minY, &tallI) ||
            __builtin_sub_overflow(ys[right], ys[left], &acrossYI) ||
            __builtin_sub_overflow(xs[bottom], xs[top], &acrossXI)) {
            free(xs);
            return true;
        }
        const double wide = (double)wideI, tall = (double)tallI;
        const double acrossY = (double)acrossYI, acrossX = (double)acrossXI;
        int first = left, second = right;
        if (tall * tall + acrossX * acrossX > wide * wide + acrossY * acrossY) {
            first = top;
            second = bottom;
        }
        if (first != second) {
            int64_t hx = xs[0], hy = ys[0];
            xs[0] = xs[first];
            ys[0] = ys[first];
            xs[first] = hx;
            ys[first] = hy;
            if (second == 0) second = first;
            hx = xs[1];
            hy = ys[1];
            xs[1] = xs[second];
            ys[1] = ys[second];
            xs[second] = hx;
            ys[second] = hy;
        }
    }

    /* `welzl_walk`. */
    const double x0 = (double)xs[0];
    const double y0 = (double)ys[0];
    double cx = (x0 + (double)xs[1]) * 0.5;
    double cy = (y0 + (double)ys[1]) * 0.5;
    double r2 = (x0 - cx) * (x0 - cx) + (y0 - cy) * (y0 - cy);
    double limit = r2 + r2 * 1e-12 + 1e-12;
    for (int i = 2; i < count; i++) {
        const double ax = (double)xs[i];
        const double ay = (double)ys[i];
        const double ox = ax - cx;
        const double oy = ay - cy;
        if (ox * ox + oy * oy <= limit) continue;
        cx = (ax + x0) * 0.5;
        cy = (ay + y0) * 0.5;
        r2 = (ax - cx) * (ax - cx) + (ay - cy) * (ay - cy);
        limit = r2 + r2 * 1e-12 + 1e-12;
        for (int j = 1; j < i; j++) {
            const double bx = (double)xs[j];
            const double by = (double)ys[j];
            const double mx = bx - cx;
            const double my = by - cy;
            if (mx * mx + my * my <= limit) continue;
            cx = (ax + bx) * 0.5;
            cy = (ay + by) * 0.5;
            r2 = (ax - cx) * (ax - cx) + (ay - cy) * (ay - cy);
            limit = r2 + r2 * 1e-12 + 1e-12;
            for (int k = 0; k < j; k++) {
                const double px = (double)xs[k];
                const double py = (double)ys[k];
                const double ix = px - cx;
                const double iy = py - cy;
                if (ix * ix + iy * iy <= limit) continue;
                const double d = 2.0 * (ax * (by - py) + bx * (py - ay) + px * (ay - by));
                if (d > -1e-12 && d < 1e-12) {
                    const double ab = (ax - bx) * (ax - bx) + (ay - by) * (ay - by);
                    const double ap = (ax - px) * (ax - px) + (ay - py) * (ay - py);
                    const double bp = (bx - px) * (bx - px) + (by - py) * (by - py);
                    if (ab >= ap && ab >= bp) {
                        cx = (ax + bx) * 0.5;
                        cy = (ay + by) * 0.5;
                        r2 = ab * 0.25;
                    } else if (ap >= bp) {
                        cx = (ax + px) * 0.5;
                        cy = (ay + py) * 0.5;
                        r2 = ap * 0.25;
                    } else {
                        cx = (bx + px) * 0.5;
                        cy = (by + py) * 0.5;
                        r2 = bp * 0.25;
                    }
                } else {
                    const double a2 = ax * ax + ay * ay;
                    const double b2 = bx * bx + by * by;
                    const double p2 = px * px + py * py;
                    cx = (a2 * (by - py) + b2 * (py - ay) + p2 * (ay - by)) / d;
                    cy = (a2 * (px - bx) + b2 * (ax - px) + p2 * (bx - ax)) / d;
                    r2 = (ax - cx) * (ax - cx) + (ay - cy) * (ay - cy);
                }
                limit = r2 + r2 * 1e-12 + 1e-12;
            }
        }
    }

    /* `farthest_squared`, which has the last word on the radius. */
    double worst = 0.0;
    for (int index = 0; index < count; index++) {
        const double dx = (double)xs[index] - cx;
        const double dy = (double)ys[index] - cy;
        const double squared = dx * dx + dy * dy;
        if (squared > worst) worst = squared;
    }
    free(xs);
    if (worst > r2) r2 = worst;

    jaiListPut(circle, 0, FLOAT_VAL(cx));
    jaiListPut(circle, 1, FLOAT_VAL(cy));
    jaiListPut(circle, 2, FLOAT_VAL(sqrt(r2)));
    jaiListTouch(circle);
    *out = BOOL_VAL(true);
    return true;
}

/* --- measures over a contour ---------------------------------------- */

/* `points_area(contour, oriented)` -- `contour_area` for two or more points:
 * the shoelace sum in doubles, halved, its magnitude unless `oriented`. Null
 * when a point is not an object with integer `x` and `y`. */
static bool primPointsArea(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *contour;
    if (!jaiArgList(args[0], 1, "points_area", &contour)) return false;
    if (!IS_BOOL(args[1])) return jaiThrow(vm.cTypeError, "points_area(): oriented must be a bool");
    const int count = contour->count;
    *out = NULL_VAL;
    if (count < 2) return true;
    JaiPointReader reader;
    jaiPointReaderInit(&reader);
    int64_t px, py;
    if (!jaiReadPoint(&reader, jaiListGet(contour, count - 1), &px, &py)) return true;
    double total = 0.0;
    for (int i = 0; i < count; i++) {
        int64_t x, y;
        if (!jaiReadPoint(&reader, jaiListGet(contour, i), &x, &y)) return true;
        total += (double)px * (double)y - (double)x * (double)py;
        px = x;
        py = y;
    }
    total *= 0.5;
    *out = FLOAT_VAL(AS_BOOL(args[1]) ? total : fabs(total));
    return true;
}

/* `points_moments(contour, out)` -- the ten sums `moments_f` accumulates over
 * a polygon by Green's theorem, for a contour of one or more points, written
 * to `out` in its order: a00, a10, a01, a20, a11, a02, a30, a21, a12, a03.
 * Every product and sum is the one `moments_f` forms, in its order, over
 * `float(point.x)` and `float(point.y)`, so `moments` makes the same Moments
 * from them to the bit -- without the list of `Point2f` it used to build
 * first. False, writing nothing, when a point is not an object with integer
 * `x` and `y`. */
static bool primPointsMoments(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *contour, *sums;
    if (!jaiArgList(args[0], 1, "points_moments", &contour)) return false;
    if (!jaiArgList(args[1], 2, "points_moments", &sums)) return false;
    if (sums->count < 10) {
        return jaiThrow(vm.cValueError, "points_moments(): the sums list holds %d of 10 values",
                        sums->count);
    }
    *out = BOOL_VAL(false);
    const int count = contour->count;
    if (count == 0) return true;
    JaiPointReader reader;
    jaiPointReaderInit(&reader);
    int64_t ix, iy;
    if (!jaiReadPoint(&reader, jaiListGet(contour, count - 1), &ix, &iy)) return true;
    double a00 = 0.0, a10 = 0.0, a01 = 0.0, a20 = 0.0, a11 = 0.0;
    double a02 = 0.0, a30 = 0.0, a21 = 0.0, a12 = 0.0, a03 = 0.0;
    double px = (double)ix;
    double py = (double)iy;
    double px2 = px * px;
    double py2 = py * py;
    for (int index = 0; index < count; index++) {
        if (!jaiReadPoint(&reader, jaiListGet(contour, index), &ix, &iy)) return true;
        const double x = (double)ix;
        const double y = (double)iy;
        const double x2 = x * x;
        const double y2 = y * y;
        const double cross = px * y - x * py;
        const double sx = px + x;
        const double sy = py + y;
        a00 += cross;
        a10 += cross * sx;
        a01 += cross * sy;
        a20 += cross * (px * sx + x2);
        a11 += cross * (px * (sy + py) + x * (sy + y));
        a02 += cross * (py * sy + y2);
        a30 += cross * sx * (px2 + x2);
        a03 += cross * sy * (py2 + y2);
        a21 += cross * (px2 * (3.0 * py + y) + 2.0 * px * x * sy + x2 * (py + 3.0 * y));
        a12 += cross * (py2 * (3.0 * px + x) + 2.0 * py * y * sx + y2 * (px + 3.0 * x));
        px = x;
        py = y;
        px2 = x2;
        py2 = y2;
    }
    const double values[10] = {a00, a10, a01, a20, a11, a02, a30, a21, a12, a03};
    for (int i = 0; i < 10; i++) jaiListPut(sums, i, FLOAT_VAL(values[i]));
    jaiListTouch(sums);
    *out = BOOL_VAL(true);
    return true;
}

/* --- the ellipse fit ------------------------------------------------- */

/* `solve_dense` in geometric.jai: Gauss-Jordan elimination with partial
 * pivoting over an n by `width` system, `right` its last column, for the
 * small systems the fits make. `matrix` is n rows of width + 1. */
static void denseSolve(double *matrix, int n, int width, double *solution) {
    const int span = width + 1;
    int column = 0, pivotRow = 0;
    while (column < width && pivotRow < n) {
        int best = pivotRow;
        double bestSize = fabs(matrix[pivotRow * span + column]);
        for (int candidate = pivotRow + 1; candidate < n; candidate++) {
            const double size = fabs(matrix[candidate * span + column]);
            if (size > bestSize) {
                best = candidate;
                bestSize = size;
            }
        }
        if (bestSize < 1e-12) {
            column++;
            continue;
        }
        if (best != pivotRow) {
            for (int k = 0; k < span; k++) {
                const double held = matrix[pivotRow * span + k];
                matrix[pivotRow * span + k] = matrix[best * span + k];
                matrix[best * span + k] = held;
            }
        }
        const double leading = matrix[pivotRow * span + column];
        for (int k = column; k < span; k++) matrix[pivotRow * span + k] /= leading;
        for (int other = 0; other < n; other++) {
            if (other == pivotRow) continue;
            const double factor = matrix[other * span + column];
            if (factor == 0.0) continue;
            for (int k = column; k < span; k++) {
                matrix[other * span + k] -= factor * matrix[pivotRow * span + k];
            }
        }
        pivotRow++;
        column++;
    }
    for (int k = 0; k < width; k++) solution[k] = 0.0;
    int row = 0, lead = 0;
    while (row < n && lead < width) {
        if (fabs(matrix[row * span + lead]) < 1e-12) {
            lead++;
            continue;
        }
        solution[lead] = matrix[row * span + width];
        row++;
        lead++;
    }
}

/* `points_fit_ellipse(points, out)` -- `fit_ellipse` in shape/fit.jai for
 * five or more points: a conic by least squares about the centroid, a two by
 * two solve for its true centre, a second conic about that, and the axes and
 * angle from its eigenvalues. Every sum, product and division is the one the
 * Jaithon makes, in its order -- the normal equations accumulated row by row,
 * the same elimination with the same pivoting -- so the ellipse is the same
 * to the bit. Written there it built a list of five floats a point and walked
 * lists of lists, 159 microseconds for a 50-point blob against OpenCV's 8.
 *
 * Writes the centre's x and y, the shorter and the longer axis and the angle
 * into `out` and returns true; false, writing nothing, when a point is not an
 * object with integer `x` and `y`. A set that does not determine an ellipse
 * throws as the Jaithon does. */
static bool primPointsFitEllipse(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *points, *fitted;
    if (!jaiArgList(args[0], 1, "points_fit_ellipse", &points)) return false;
    if (!jaiArgList(args[1], 2, "points_fit_ellipse", &fitted)) return false;
    if (fitted->count < 5) {
        return jaiThrow(vm.cValueError, "points_fit_ellipse(): the out list holds %d of 5 values",
                        fitted->count);
    }
    const int count = points->count;
    *out = BOOL_VAL(false);
    if (count < 5) return true;
    double *xy = (double *)malloc((size_t)count * 2 * sizeof(double));
    if (xy == NULL) return jaiThrow(vm.cRuntimeError, "points_fit_ellipse(): out of memory");
    JaiPointReader reader;
    jaiPointReaderInit(&reader);
    for (int i = 0; i < count; i++) {
        int64_t x, y;
        if (!jaiReadPoint(&reader, jaiListGet(points, i), &x, &y)) {
            free(xy);
            return true;
        }
        xy[i * 2] = (double)x;
        xy[i * 2 + 1] = (double)y;
    }

    double cx = 0.0, cy = 0.0;
    for (int i = 0; i < count; i++) {
        cx += xy[i * 2];
        cy += xy[i * 2 + 1];
    }
    cx /= (double)count;
    cy /= (double)count;

    /* `least_squares` over [-u*u, -v*v, -u*v, u, v]. */
    double normal[5 * 6];
    for (int k = 0; k < 30; k++) normal[k] = 0.0;
    for (int p = 0; p < count; p++) {
        const double u = xy[p * 2] - cx;
        const double v = xy[p * 2 + 1] - cy;
        const double row[5] = {-u * u, -v * v, -u * v, u, v};
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) normal[i * 6 + j] += row[i] * row[j];
            normal[i * 6 + 5] += row[i];
        }
    }
    double conic[5];
    denseSolve(normal, 5, 5, conic);

    double centre[2 * 3] = {2.0 * conic[0], conic[2], conic[3], conic[2], 2.0 * conic[1], conic[4]};
    double offset[2];
    denseSolve(centre, 2, 2, offset);

    /* `least_squares` over [u*u, v*v, u*v] about the true centre. */
    double second[3 * 4];
    for (int k = 0; k < 12; k++) second[k] = 0.0;
    for (int p = 0; p < count; p++) {
        const double u = xy[p * 2] - cx - offset[0];
        const double v = xy[p * 2 + 1] - cy - offset[1];
        const double row[3] = {u * u, v * v, u * v};
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) second[i * 4 + j] += row[i] * row[j];
            second[i * 4 + 3] += row[i];
        }
    }
    free(xy);
    double shape[3];
    denseSolve(second, 3, 3, shape);
    const double a = shape[0], b = shape[1], c = shape[2];

    const double root = sqrt((a - b) * (a - b) + c * c);
    const double larger = (a + b + root) * 0.5;
    const double smaller = (a + b - root) * 0.5;
    if (fabs(larger) <= 1e-12 || fabs(smaller) <= 1e-12) {
        return jaiThrow(vm.cValueError, "these points do not determine an ellipse");
    }
    const double shortAxis = 2.0 / sqrt(fabs(larger));
    const double longAxis = 2.0 / sqrt(fabs(smaller));
    double angle = 0.5 * atan2(c, a - b) * 180.0 / 3.141592653589793 + 180.0;
    while (angle >= 180.0) angle -= 180.0;
    while (angle < 0.0) angle += 180.0;

    const double values[5] = {cx + offset[0], cy + offset[1], shortAxis, longAxis, angle};
    for (int i = 0; i < 5; i++) jaiListPut(fitted, i, FLOAT_VAL(values[i]));
    jaiListTouch(fitted);
    *out = BOOL_VAL(true);
    return true;
}

/* --- the line fit ---------------------------------------------------- */

/* `weighted_fit` in shape/fit.jai: the weighted centroid and the principal
 * direction of the weighted scatter about it. False when every weight came
 * to nothing, which the Jaithon reports as an error. */
static bool lineWeightedFit(const double *xy, const double *weights, int count, double result[4]) {
    double total = 0.0, sx = 0.0, sy = 0.0;
    for (int i = 0; i < count; i++) {
        const double w = weights[i];
        total += w;
        sx += w * xy[i * 2];
        sy += w * xy[i * 2 + 1];
    }
    if (total <= 1e-12) return false;
    const double cx = sx / total;
    const double cy = sy / total;
    double sxx = 0.0, syy = 0.0, sxy = 0.0;
    for (int i = 0; i < count; i++) {
        const double w = weights[i];
        const double dx = xy[i * 2] - cx;
        const double dy = xy[i * 2 + 1] - cy;
        sxx += w * dx * dx;
        syy += w * dy * dy;
        sxy += w * dx * dy;
    }
    const double difference = sxx - syy;
    const double angle = 0.5 * atan2(2.0 * sxy, difference);
    result[0] = shapeCos(angle);
    result[1] = shapeSin(angle);
    result[2] = cx;
    result[3] = cy;
    return true;
}

/* `loss_weight`: how much a point this far from the line counts next pass. */
static double lineLossWeight(int64_t distType, double r) {
    if (distType == 1) return 1.0 / (r >= 1e-6 ? r : 1e-6);
    if (distType == 5) return 1.0 / (1.0 + r / 1.3998);
    if (distType == 6) {
        const double t = r / 2.9846;
        return exp(-t * t);
    }
    if (distType == 7) {
        if (r < 1.345) return 1.0;
        return 1.345 / r;
    }
    return 1.0;
}

/* `points_fit_line(points, dist_type, out)` -- `fit_line` in shape/fit.jai
 * for two or more points: the least-squares line, then for any loss but L2
 * ten more fits each reweighted by how far every point fell from the last.
 * The same sums in the same order, so the line is the same to the bit.
 * Writes vx, vy, x0, y0 into `out` and returns true; false, writing nothing,
 * when a point is not an object with integer `x` and `y`. */
static bool primPointsFitLine(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *points, *line;
    int64_t distType;
    if (!jaiArgList(args[0], 1, "points_fit_line", &points)) return false;
    if (!jaiStrWantInt(args[1], "points_fit_line", "the distance type", &distType)) return false;
    if (!jaiArgList(args[2], 3, "points_fit_line", &line)) return false;
    if (line->count < 4) {
        return jaiThrow(vm.cValueError, "points_fit_line(): the out list holds %d of 4 values",
                        line->count);
    }
    const int count = points->count;
    *out = BOOL_VAL(false);
    if (count < 2) return true;
    double *xy = (double *)malloc((size_t)count * 3 * sizeof(double));
    if (xy == NULL) return jaiThrow(vm.cRuntimeError, "points_fit_line(): out of memory");
    double *weights = xy + count * 2;
    JaiPointReader reader;
    jaiPointReaderInit(&reader);
    for (int i = 0; i < count; i++) {
        int64_t x, y;
        if (!jaiReadPoint(&reader, jaiListGet(points, i), &x, &y)) {
            free(xy);
            return true;
        }
        xy[i * 2] = (double)x;
        xy[i * 2 + 1] = (double)y;
        weights[i] = 1.0;
    }
    double result[4];
    bool ok = lineWeightedFit(xy, weights, count, result);
    if (ok && distType != 2) {
        for (int pass = 0; pass < 10 && ok; pass++) {
            for (int i = 0; i < count; i++) {
                const double dx = xy[i * 2] - result[2];
                const double dy = xy[i * 2 + 1] - result[3];
                const double away = fabs(dx * -result[1] + dy * result[0]);
                weights[i] = lineLossWeight(distType, away);
            }
            ok = lineWeightedFit(xy, weights, count, result);
        }
    }
    free(xy);
    if (!ok) return jaiThrow(vm.cValueError, "every point was weighted to nothing");
    for (int i = 0; i < 4; i++) jaiListPut(line, i, FLOAT_VAL(result[i]));
    jaiListTouch(line);
    *out = BOOL_VAL(true);
    return true;
}

/* `points_hu(points)` -- the seven Hu invariants of a contour of one or more
 * points, as `hu_moments(moments(points))` makes them: the ten Green's
 * theorem sums, signed and scaled as `moments_of_sums` scales them, the
 * central and normalised moments as `Moments.init` derives them, and the
 * seven combinations `hu_moments` takes -- each operation the Jaithon's, in
 * its order. `match_shapes` asks for two of these a comparison, and making
 * each through two objects and their twenty-four fields was most of its cost:
 * 3600 comparisons took 21.7 ms against OpenCV's 2.9.
 *
 * Returns a new list of seven floats, or null when a point is not an object
 * with integer `x` and `y`. */
static bool contourHu(ObjList *contour, double hu[7]) {
    const int count = contour->count;
    if (count == 0) return false;
    JaiPointReader reader;
    jaiPointReaderInit(&reader);
    int64_t ix, iy;
    if (!jaiReadPoint(&reader, jaiListGet(contour, count - 1), &ix, &iy)) return false;
    double a00 = 0.0, a10 = 0.0, a01 = 0.0, a20 = 0.0, a11 = 0.0;
    double a02 = 0.0, a30 = 0.0, a21 = 0.0, a12 = 0.0, a03 = 0.0;
    double px = (double)ix, py = (double)iy;
    double px2 = px * px, py2 = py * py;
    for (int index = 0; index < count; index++) {
        if (!jaiReadPoint(&reader, jaiListGet(contour, index), &ix, &iy)) return false;
        const double x = (double)ix, y = (double)iy;
        const double x2 = x * x, y2 = y * y;
        const double cross = px * y - x * py;
        const double sx = px + x, sy = py + y;
        a00 += cross;
        a10 += cross * sx;
        a01 += cross * sy;
        a20 += cross * (px * sx + x2);
        a11 += cross * (px * (sy + py) + x * (sy + y));
        a02 += cross * (py * sy + y2);
        a30 += cross * sx * (px2 + x2);
        a03 += cross * sy * (py2 + y2);
        a21 += cross * (px2 * (3.0 * py + y) + 2.0 * px * x * sy + x2 * (py + 3.0 * y));
        a12 += cross * (py2 * (3.0 * px + x) + 2.0 * py * y * sx + y2 * (px + 3.0 * x));
        px = x;
        py = y;
        px2 = x2;
        py2 = y2;
    }

    /* `moments_of_sums`. */
    double m[10] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    if (!(fabs(a00) <= 1e-12)) {
        const double sign = a00 > 0.0 ? 1.0 : -1.0;
        m[0] = a00 * sign / 2.0;
        m[1] = a10 * sign / 6.0;
        m[2] = a01 * sign / 6.0;
        m[3] = a20 * sign / 12.0;
        m[4] = a11 * sign / 24.0;
        m[5] = a02 * sign / 12.0;
        m[6] = a30 * sign / 20.0;
        m[7] = a21 * sign / 60.0;
        m[8] = a12 * sign / 60.0;
        m[9] = a03 * sign / 20.0;
    }
    const double m00 = m[0], m10 = m[1], m01 = m[2], m20 = m[3], m11 = m[4];
    const double m02 = m[5], m30 = m[6], m21 = m[7], m12 = m[8], m03 = m[9];

    /* `Moments.init`. */
    double inverse = 0.0, cx = 0.0, cy = 0.0;
    if (fabs(m00) > 1e-12) {
        inverse = 1.0 / m00;
        cx = m10 * inverse;
        cy = m01 * inverse;
    }
    const double mu20 = m20 - m10 * cx;
    double mu11 = m11 - m10 * cy;
    const double mu02 = m02 - m01 * cy;
    const double kept11 = mu11;
    const double mu30 = m30 - cx * (3.0 * mu20 + cx * m10);
    mu11 = mu11 + mu11;
    const double mu21 = m21 - cx * (mu11 + cx * m01) - cy * mu20;
    const double mu12 = m12 - cy * (mu11 + cy * m10) - cx * mu02;
    const double mu03 = m03 - cy * (3.0 * mu02 + cy * m01);
    const double root = sqrt(fabs(inverse));
    const double s2 = inverse * inverse;
    const double s3 = s2 * root;
    const double n20 = mu20 * s2;
    const double n11 = kept11 * s2;
    const double n02 = mu02 * s2;
    const double n30 = mu30 * s3;
    const double n21 = mu21 * s3;
    const double n12 = mu12 * s3;
    const double n03 = mu03 * s3;

    /* `hu_moments`. */
    const double t0 = n30 + n12;
    const double t1 = n21 + n03;
    const double q0 = t0 * t0;
    const double q1 = t1 * t1;
    const double n4 = 4.0 * n11;
    const double s = n20 + n02;
    const double d = n20 - n02;
    const double sum0 = q0 - 3.0 * q1;
    const double sum1 = 3.0 * q0 - q1;
    const double values[7] = {
        s,
        d * d + n4 * n11,
        (n30 - 3.0 * n12) * (n30 - 3.0 * n12) + (3.0 * n21 - n03) * (3.0 * n21 - n03),
        q0 + q1,
        (n30 - 3.0 * n12) * t0 * sum0 + (3.0 * n21 - n03) * t1 * sum1,
        d * (q0 - q1) + n4 * t0 * t1,
        (3.0 * n21 - n03) * t0 * sum0 - (n30 - 3.0 * n12) * t1 * sum1,
    };
    for (int i = 0; i < 7; i++) hu[i] = values[i];
    return true;
}

static bool primPointsHu(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *contour;
    if (!jaiArgList(args[0], 1, "points_hu", &contour)) return false;
    *out = NULL_VAL;
    double hu[7];
    if (!contourHu(contour, hu)) return true;
    ObjList *made = jaiListNew(7);
    if (made == NULL) return false;
    jaiGCPushRoot(OBJ_VAL(made));
    for (int i = 0; i < 7; i++) jaiListPush(made, FLOAT_VAL(hu[i]));
    jaiGCPopRoot();
    *out = OBJ_VAL(made);
    return true;
}

/* `math.log10` as std.math writes it: the natural logarithm over LN_10,
 * snapped to the integer when that is within 1e-10 and ten to it is exactly
 * the argument, because two roundings leave exact powers of ten just short.
 * Only ever asked of a finite argument over 1e-30 here, or a NaN or an
 * infinity, which pass through as std.math passes them. */
static double shapeLog10(double x) {
    if (x != x) return x;
    if (isinf(x)) return x;
    const double approximate = log(x) / 2.302585092994046;
    const double nearest = round(approximate);
    if (fabs(approximate - nearest) < 1e-10 && !(fabs(nearest) > 22.0) && pow(10.0, nearest) == x) {
        return nearest;
    }
    return approximate;
}

/* `std.math.fabs`, which turns a negative zero positive and passes a NaN. */
static double shapeFabs(double x) {
    if (x < 0.0) return -x;
    if (x == 0.0) return 0.0;
    return x;
}

/* `points_match(a, b, method)` -- `match_shapes`: the two contours' Hu
 * invariants compared through their signed log magnitudes, by reciprocals
 * (1), differences (2) or the largest relative difference (3), each step the
 * Jaithon's. Null when either contour is empty or has a point that is not an
 * object with integer `x` and `y`. */
static bool primPointsMatch(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *a, *b;
    int64_t method;
    if (!jaiArgList(args[0], 1, "points_match", &a)) return false;
    if (!jaiArgList(args[1], 2, "points_match", &b)) return false;
    if (!jaiStrWantInt(args[2], "points_match", "the method", &method)) return false;
    *out = NULL_VAL;
    double ha[7], hb[7];
    if (!contourHu(a, ha) || !contourHu(b, hb)) return true;
    double total = 0.0, worst = 0.0;
    for (int i = 0; i < 7; i++) {
        const double ma = shapeFabs(ha[i]);
        const double mb = shapeFabs(hb[i]);
        if (ma <= 1e-30 || mb <= 1e-30) continue;
        const double sa = ha[i] < 0.0 ? -1.0 : 1.0;
        const double sb = hb[i] < 0.0 ? -1.0 : 1.0;
        const double la = sa * shapeLog10(ma);
        const double lb = sb * shapeLog10(mb);
        if (method == 1) {
            total += shapeFabs(1.0 / la - 1.0 / lb);
        } else if (method == 2) {
            total += shapeFabs(la - lb);
        } else {
            const double relative = shapeFabs((la - lb) / la);
            if (relative > worst) worst = relative;
        }
    }
    *out = FLOAT_VAL(method == 3 ? worst : total);
    return true;
}

void jaiShapeRegisterPrimitives(ObjModule *ns) {
    jaiStrDefinePrim(ns, "points_hull",    primPointsHull,   2, 2);
    jaiStrDefinePrim(ns, "points_min_box", primPointsMinBox, 2, 2);
    jaiStrDefinePrim(ns, "points_approx", primPointsApprox, 3, 3);
    jaiStrDefinePrim(ns, "points_min_circle", primPointsMinCircle, 2, 2);
    jaiStrDefinePrim(ns, "points_area", primPointsArea, 2, 2);
    jaiStrDefinePrim(ns, "points_moments", primPointsMoments, 2, 2);
    jaiStrDefinePrim(ns, "points_fit_ellipse", primPointsFitEllipse, 2, 2);
    jaiStrDefinePrim(ns, "points_fit_line", primPointsFitLine, 3, 3);
    jaiStrDefinePrim(ns, "points_hu", primPointsHu, 1, 1);
    jaiStrDefinePrim(ns, "points_match", primPointsMatch, 3, 3);
}
