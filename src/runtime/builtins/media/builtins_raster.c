/* builtins_raster.c — the pixel loops behind jaicv's drawing.
 *
 * Spans, lines and convex fills. Each is one call from Jaithon for a whole
 * shape rather than one per pixel, because a pixel written from up there costs
 * about a hundred nanoseconds and a frame of detections is tens of thousands
 * of them.
 *
 * Every routine writes to one of two places. One is a host mirror, a Jaithon
 * list of boxed numbers, which is where a picture already on the host is
 * edited. The other is a device buffer's own memory: storage is shared with
 * the GPU, so scribbling on a few thousand pixels of a frame needs no copy
 * over, no copy back, and no dispatch. Being ordinary host work it also runs
 * while the GPU is busy with something else, which drawing encoded onto the
 * queue cannot do -- the queue runs in order, so a kernel that draws a frame
 * waits behind whatever network was queued before it.
 *
 * The arithmetic is OpenCV's throughout, matching what jaicv's drawing module
 * computed when these loops lived there. `tests/test_against_opencv.jai` is
 * what holds them to it. */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "runtime/builtins/text/builtins_str.h"
#include "runtime/runtime.h"

#include "native/native.h"

#define JAI_XY_SHIFT 16
#define JAI_XY_ONE   (1 << JAI_XY_SHIFT)
#define JAI_MAX_CORNERS 4096

/* Truncating toward zero, which is what jaicv's `trunc_div` computes with its
 * floor division plus a correction, and what C's `/` already does. */
static int64_t truncDiv(int64_t a, int64_t b) { return a / b; }

/* Where a shape's pixels go: either boxed list elements or the floats a device
 * buffer holds. `origin` and `stride` place the first pixel and say how far
 * apart the rows are, so a view of a larger picture writes into the right part
 * of it. */
typedef struct {
    Value   *boxed;
    float   *raw;
    int64_t  capacity;
    int64_t  origin;
    int64_t  stride;
    int64_t  cols;
    int64_t  rows;
    int      cn;
    Value    colour[4];
    float    rawColour[4];
    bool     rawUniform;    /* every channel of rawColour holds the same bits */
} JaiSurface;

/* `run` pixels of the surface's colour, starting at (x, y). Out of range is
 * dropped rather than refused: a shape is clipped to the picture, and the
 * caller has already decided the shape is worth drawing. */
JAI_INLINE void surfaceRun(const JaiSurface *s, int64_t x, int64_t y, int64_t run) {
    if (run <= 0 || y < 0 || y >= s->rows) return;
    if (x < 0) {
        run += x;
        x = 0;
    }
    if (x + run > s->cols) run = s->cols - x;
    if (run <= 0) return;

    const int cn = s->cn;
    const int64_t base = s->origin + y * s->stride + x * (int64_t)cn;
    if (base < 0 || base + run * cn > s->capacity) return;
    /* The colour is read into locals before the first store. Read through
     * `s` inside the loop, every channel was loaded again after every store,
     * because a float store may alias the float array the colour lives in --
     * and the rows a thick stroke is made of are a few pixels each, so that
     * reload was most of the cost of drawing one. */
    if (s->raw != NULL) {
        float *write = s->raw + base;
        if (s->rawUniform) {
            /* Every channel the same, grey or white: one flat run, four
             * floats a store. Written out rather than left to the vectoriser,
             * whose checks and remainder loop cost more than the stores do on
             * the few pixels a row of a stroke is. */
            const float v = s->rawColour[0];
            int64_t n = run * cn;
#if defined(__aarch64__)
            const float32x4_t four = vdupq_n_f32(v);
            for (; n >= 4; n -= 4, write += 4) vst1q_f32(write, four);
#endif
            for (; n > 0; n--) *write++ = v;
            return;
        }
        const float c0 = s->rawColour[0], c1 = s->rawColour[1];
        const float c2 = s->rawColour[2], c3 = s->rawColour[3];
        if (cn == 3) {
            for (int64_t i = 0; i < run; i++, write += 3) {
                write[0] = c0;
                write[1] = c1;
                write[2] = c2;
            }
            return;
        }
        if (cn == 4) {
            for (int64_t i = 0; i < run; i++, write += 4) {
                write[0] = c0;
                write[1] = c1;
                write[2] = c2;
                write[3] = c3;
            }
            return;
        }
        for (int64_t i = 0; i < run; i++, write += 2) {
            write[0] = c0;
            write[1] = c1;
        }
        return;
    }
    Value colour[4];
    memcpy(colour, s->colour, sizeof colour);
    Value *write = s->boxed + base;
    for (int64_t i = 0; i < run; i++) {
        for (int c = 0; c < cn; c++) write[c] = colour[c];
        write += cn;
    }
}

/* Read the surface a drawing primitive writes to.
 *
 * `target` is a list when the picture is on the host and a device buffer when
 * it is not; the rest describes where in it the picture sits. */
static bool readSurface(Value *args, int first, const char *fnName, JaiSurface *s) {
    ObjList *colour;
    int64_t channels, origin, stride, cols, rows;
    if (!jaiArgList(args[first + 1], first + 2, fnName, &colour)) return false;
    if (!jaiStrWantInt(args[first + 2], fnName, "the channel count", &channels)) return false;
    if (!jaiStrWantInt(args[first + 3], fnName, "the first element", &origin)) return false;
    if (!jaiStrWantInt(args[first + 4], fnName, "the row stride", &stride)) return false;
    if (!jaiStrWantInt(args[first + 5], fnName, "the width", &cols)) return false;
    if (!jaiStrWantInt(args[first + 6], fnName, "the height", &rows)) return false;
    if (channels < 1 || channels > 4 || channels != colour->count) {
        return jaiThrow(vm.cValueError, "%s(): %lld channels against a colour of %d",
                        fnName, (long long)channels, colour->count);
    }
    if (origin < 0 || stride < 0 || cols < 0 || rows < 0) {
        return jaiThrow(vm.cValueError, "%s(): the layout must not be negative", fnName);
    }

    memset(s, 0, sizeof *s);
    s->cn = (int)channels;
    s->origin = origin;
    s->stride = stride;
    s->cols = cols;
    s->rows = rows;
    for (int c = 0; c < s->cn; c++) {
        Value v = jaiListGet(colour, c);
        if (!IS_FLOAT(v) && !IS_INT(v)) {
            return jaiThrow(vm.cTypeError, "%s(): the colour holds a non-number", fnName);
        }
        s->colour[c] = v;
        s->rawColour[c] = (float)(IS_FLOAT(v) ? AS_FLOAT(v) : (double)AS_INT(v));
    }
    s->rawUniform = true;
    for (int c = 1; c < s->cn; c++) {
        if (memcmp(&s->rawColour[c], &s->rawColour[0], sizeof(float)) != 0) s->rawUniform = false;
    }

    if (IS_LIST(args[first])) {
        ObjList *values = AS_LIST(args[first]);
        s->boxed = jaiListBox(values);
        s->capacity = values->count;
        values->version++;
        return true;
    }

    JaiGpuBuffer *buffer = NULL;
    int64_t base = 0;
    if (!jaiGpuBufferOf(args[first], first + 1, fnName, &buffer, &base)) return false;
    int64_t wanted = origin + cols * channels;
    if (rows > 0) wanted = origin + (rows - 1) * stride + cols * channels;
    float *raw = jaiGpuMapWrite(buffer, (size_t)base, (size_t)(wanted > 0 ? wanted : 0));
    if (raw == NULL) {
        return jaiThrow(vm.cRuntimeError, "%s(): the buffer would not map to write", fnName);
    }
    s->raw = raw;
    s->capacity = wanted;
    return true;
}

/* `fill_span(target, colour, cn, origin, stride, cols, rows, x, y, run)` --
 * one row of one colour.
 *
 * Every filled shape is rows of a single colour, and a row written a pixel at
 * a time in Jaithon cost about eleven nanoseconds a pixel: a filled rectangle
 * over a 720p frame took 8.2 ms, more than the rest of a camera frame put
 * together. */
static bool primFillSpan(int argc, Value *args, Value *out) {
    (void)argc;
    JaiSurface surface;
    if (!readSurface(args, 0, "fill_span", &surface)) return false;
    int64_t x, y, run;
    if (!jaiStrWantInt(args[7], "fill_span", "the column", &x)) return false;
    if (!jaiStrWantInt(args[8], "fill_span", "the row", &y)) return false;
    if (!jaiStrWantInt(args[9], "fill_span", "the length", &run)) return false;
    surfaceRun(&surface, x, y, run);
    *out = NULL_VAL;
    return true;
}

/* The pixels of one clipped, normalised segment, in the order OpenCV's line
 * iterator walks them. */
typedef struct {
    int x, y;
    int dx, dy;
    int majorX, majorY;
    int minorX, minorY;
    int connectivity;
} JaiLineWalk;

static void lineWalk(const JaiLineWalk *w, const JaiSurface *s) {
    int x = w->x;
    int y = w->y;
    if (w->connectivity == 4) {
        int err = 0;
        const int plus = w->dx + w->dx + w->dy + w->dy;
        const int minus = -(w->dy + w->dy);
        const int steps = w->dx + w->dy + 1;
        for (int i = 0; i < steps; i++) {
            surfaceRun(s, x, y, 1);
            const bool sideways = err < 0;
            err += minus + (sideways ? plus : 0);
            x += sideways ? w->minorX : w->majorX;
            y += sideways ? w->minorY : w->majorY;
        }
        return;
    }
    int err = w->dx - (w->dy + w->dy);
    const int plus = w->dx + w->dx;
    const int minus = -(w->dy + w->dy);
    const int steps = w->dx + 1;
    for (int i = 0; i < steps; i++) {
        surfaceRun(s, x, y, 1);
        const bool diagonal = err < 0;
        err += minus + (diagonal ? plus : 0);
        x += w->majorX + (diagonal ? w->minorX : 0);
        y += w->majorY + (diagonal ? w->minorY : 0);
    }
}

/* Clip a segment to the picture and settle which axis leads, the way jaicv's
 * `clip_line` and `line_points` do between them.
 *
 * Cohen-Sutherland with OpenCV's exact arithmetic, including that the segment
 * is normalised left to right *after* clipping rather than before -- clipping
 * first is what makes a line come out the same whichever end the caller names,
 * and the other order rounds ties differently on any segment running right to
 * left. False when none of it shows. */
static bool clipAndOrient(int64_t cols, int64_t rows, int64_t x1, int64_t y1,
                          int64_t x2, int64_t y2, int connectivity, JaiLineWalk *w) {
    if (cols <= 0 || rows <= 0) return false;
    const int64_t right = cols - 1;
    const int64_t bottom = rows - 1;

    int c1 = (x1 < 0 ? 1 : 0) + (x1 > right ? 2 : 0) + (y1 < 0 ? 4 : 0) + (y1 > bottom ? 8 : 0);
    int c2 = (x2 < 0 ? 1 : 0) + (x2 > right ? 2 : 0) + (y2 < 0 ? 4 : 0) + (y2 > bottom ? 8 : 0);

    if ((c1 & c2) == 0 && (c1 | c2) != 0) {
        if (c1 & 12) {
            const int64_t a = c1 < 8 ? 0 : bottom;
            x1 += truncDiv((a - y1) * (x2 - x1), y2 - y1);
            y1 = a;
            c1 = (x1 < 0 ? 1 : 0) + (x1 > right ? 2 : 0);
        }
        if (c2 & 12) {
            const int64_t a = c2 < 8 ? 0 : bottom;
            x2 += truncDiv((a - y2) * (x2 - x1), y2 - y1);
            y2 = a;
            c2 = (x2 < 0 ? 1 : 0) + (x2 > right ? 2 : 0);
        }
        if ((c1 & c2) == 0 && (c1 | c2) != 0) {
            if (c1) {
                const int64_t a = c1 == 1 ? 0 : right;
                y1 += truncDiv((a - x1) * (y2 - y1), x2 - x1);
                x1 = a;
                c1 = 0;
            }
            if (c2) {
                const int64_t a = c2 == 1 ? 0 : right;
                y2 += truncDiv((a - x2) * (y2 - y1), x2 - x1);
                x2 = a;
                c2 = 0;
            }
        }
    }
    if ((c1 | c2) != 0) return false;

    int64_t ax = x1, ay = y1, bx = x2, by = y2;
    if (bx < ax || (bx == ax && by < ay)) {
        ax = x2; ay = y2; bx = x1; by = y1;
    }
    int64_t dx = bx - ax;
    int64_t dy = by - ay;
    const int stepX = dx < 0 ? -1 : 1;
    const int stepY = dy < 0 ? -1 : 1;
    if (dx < 0) dx = -dx;
    if (dy < 0) dy = -dy;

    w->x = (int)ax;
    w->y = (int)ay;
    w->connectivity = connectivity;
    if (dy > dx) {
        w->dx = (int)dy;
        w->dy = (int)dx;
        w->majorX = 0;
        w->majorY = stepY;
        w->minorX = stepX;
        w->minorY = 0;
    } else {
        w->dx = (int)dx;
        w->dy = (int)dy;
        w->majorX = stepX;
        w->majorY = 0;
        w->minorX = 0;
        w->minorY = stepY;
    }
    return true;
}

/* `draw_line(target, colour, cn, origin, stride, cols, rows, x1, y1, x2, y2,
 *  connectivity)` -- one pixel wide, clipped and walked here.
 *
 * Building a point object for each pixel and writing them one by one cost
 * about a hundred nanoseconds a pixel, and a line of text is a few dozen
 * strokes of them. */
static bool primDrawLine(int argc, Value *args, Value *out) {
    (void)argc;
    JaiSurface surface;
    if (!readSurface(args, 0, "draw_line", &surface)) return false;
    int64_t ends[4];
    for (int i = 0; i < 4; i++) {
        if (!jaiStrWantInt(args[7 + i], "draw_line", "a coordinate", &ends[i])) return false;
    }
    int64_t connectivity;
    if (!jaiStrWantInt(args[11], "draw_line", "the connectivity", &connectivity)) return false;
    if (connectivity != 4 && connectivity != 8) {
        return jaiThrow(vm.cValueError, "draw_line(): connectivity must be 4 or 8, got %lld",
                        (long long)connectivity);
    }
    JaiLineWalk walk;
    if (clipAndOrient(surface.cols, surface.rows, ends[0], ends[1], ends[2], ends[3],
                      (int)connectivity, &walk)) {
        lineWalk(&walk, &surface);
    }
    *out = NULL_VAL;
    return true;
}

/* `draw_lines(target, colour, cn, origin, stride, cols, rows, ends,
 *  connectivity)` -- a batch of one-wide segments, `ends` holding x1, y1, x2,
 *  y2 for each in turn.
 *
 * A letter is a dozen or so strokes and a line of text a few hundred, and each
 * one arriving on its own meant reading the surface again for every one of
 * them. Handing the whole batch over reads it once. */
static bool primDrawLines(int argc, Value *args, Value *out) {
    (void)argc;
    JaiSurface surface;
    if (!readSurface(args, 0, "draw_lines", &surface)) return false;
    ObjList *ends;
    int64_t connectivity;
    if (!jaiArgList(args[7], 8, "draw_lines", &ends)) return false;
    if (!jaiStrWantInt(args[8], "draw_lines", "the connectivity", &connectivity)) return false;
    if (connectivity != 4 && connectivity != 8) {
        return jaiThrow(vm.cValueError, "draw_lines(): connectivity must be 4 or 8, got %lld",
                        (long long)connectivity);
    }
    if (ends->count % 4 != 0) {
        return jaiThrow(vm.cValueError,
                        "draw_lines(): %d numbers is not a whole number of segments",
                        ends->count);
    }

    const int segments = ends->count / 4;
    for (int i = 0; i < segments; i++) {
        int64_t at[4];
        for (int k = 0; k < 4; k++) {
            Value v = jaiListGet(ends, i * 4 + k);
            if (!IS_INT(v)) {
                return jaiThrow(vm.cTypeError,
                                "draw_lines(): segment %d holds a non-integer", i);
            }
            at[k] = AS_INT(v);
        }
        JaiLineWalk walk;
        if (clipAndOrient(surface.cols, surface.rows, at[0], at[1], at[2], at[3],
                          (int)connectivity, &walk)) {
            lineWalk(&walk, &surface);
        }
    }
    *out = NULL_VAL;
    return true;
}

/* The scanline fill of a convex polygon, in fixed point.
 *
 * Mirrors what jaicv's `fill_convex` computed: the same two edge chains walked
 * from the topmost vertex, the same truncating division for the slope, the
 * same rounding of each end of a row. A row of one cost about half a
 * microsecond written in Jaithon, and a rectangle two pixels thick over a 720p
 * frame is some six hundred rows -- most of what drawing a frame of detections
 * came to. */
static void convexFill(const JaiSurface *s, const int64_t *xs, const int64_t *ys,
                       int count, int shift, int64_t delta1, int64_t delta2) {
    if (count < 3) return;
    const int64_t delta = (int64_t)1 << shift >> 1;

    int64_t xMin = xs[0], xMax = xs[0];
    int64_t yMin = ys[0], yMax = ys[0];
    int top = 0;
    for (int i = 0; i < count; i++) {
        if (ys[i] < yMin) { yMin = ys[i]; top = i; }
        if (ys[i] > yMax) yMax = ys[i];
        if (xs[i] > xMax) xMax = xs[i];
        if (xs[i] < xMin) xMin = xs[i];
    }
    xMin = (xMin + delta) >> shift;
    xMax = (xMax + delta) >> shift;
    yMin = (yMin + delta) >> shift;
    yMax = (yMax + delta) >> shift;
    if (xMax < 0 || yMax < 0 || xMin >= s->cols || yMin >= s->rows) return;
    if (yMax > s->rows - 1) yMax = s->rows - 1;

    int indexOf[2] = {top, top};
    const int direction[2] = {1, count - 1};
    int64_t endY[2] = {yMin, yMin};
    int64_t edgeX[2] = {-JAI_XY_ONE, -JAI_XY_ONE};
    int64_t edgeDx[2] = {0, 0};
    int remaining = count;

    for (int64_t y = yMin; y <= yMax; y++) {
        bool exhausted = false;
        for (int side = 0; side < 2; side++) {
            if (y < endY[side]) continue;
            int idx = indexOf[side];
            const int di = direction[side];
            int64_t xAt = 0;
            int64_t ty = 0;
            for (;;) {
                ty = (ys[idx] + delta) >> shift;
                if (ty > y || remaining == 0) break;
                xAt = xs[idx];
                idx += di;
                if (idx >= count) idx -= count;
                remaining--;
            }
            if (y >= ty) {
                /* Both chains have reached the last vertex; this row is the
                 * bottom of the polygon and still belongs to it. */
                exhausted = true;
                break;
            }
            endY[side] = ty;
            const int64_t start = xAt << (JAI_XY_SHIFT - shift);
            const int64_t finish = xs[idx] << (JAI_XY_SHIFT - shift);
            edgeDx[side] = truncDiv((finish - start) * 2 + (ty - y), 2 * (ty - y));
            edgeX[side] = start;
            indexOf[side] = idx;
        }
        if (y >= 0) {
            const int left = edgeX[0] > edgeX[1] ? 1 : 0;
            const int right = 1 - left;
            int64_t from = (edgeX[left] + delta1) >> JAI_XY_SHIFT;
            int64_t to = (edgeX[right] + delta2) >> JAI_XY_SHIFT;
            if (from < 0) from = 0;
            if (to > s->cols - 1) to = s->cols - 1;
            if (from <= to) surfaceRun(s, from, y, to - from + 1);
        }
        if (exhausted) return;
        edgeX[0] += edgeDx[0];
        edgeX[1] += edgeDx[1];
    }
}

/* `fill_convex(target, colour, cn, origin, stride, cols, rows, corners, shift,
 *  delta1, delta2)` -- every row of a convex polygon. `corners` is the x and y
 *  of each in turn, in fixed point with `shift` fractional bits. */
static bool primFillConvex(int argc, Value *args, Value *out) {
    (void)argc;
    JaiSurface surface;
    if (!readSurface(args, 0, "fill_convex", &surface)) return false;
    ObjList *flat;
    int64_t shift, delta1, delta2;
    if (!jaiArgList(args[7], 8, "fill_convex", &flat)) return false;
    if (!jaiStrWantInt(args[8], "fill_convex", "the shift", &shift)) return false;
    if (!jaiStrWantInt(args[9], "fill_convex", "the near rounding", &delta1)) return false;
    if (!jaiStrWantInt(args[10], "fill_convex", "the far rounding", &delta2)) return false;
    if (flat->count % 2 != 0) {
        return jaiThrow(vm.cValueError, "fill_convex(): %d numbers is not a list of corners",
                        flat->count);
    }
    const int count = flat->count / 2;
    if (shift < 0 || shift > JAI_XY_SHIFT) {
        return jaiThrow(vm.cValueError, "fill_convex(): a shift of %lld is out of range",
                        (long long)shift);
    }
    /* More corners than there is room for is the caller's arithmetic gone
     * wrong, and drawing nothing would hide it. Fewer than three is a shape
     * with no area, which is a thing a caller legitimately asks for. */
    if (count > JAI_MAX_CORNERS) {
        return jaiThrow(vm.cValueError, "fill_convex(): %d corners is more than %d",
                        count, JAI_MAX_CORNERS);
    }
    if (count < 3) {
        *out = NULL_VAL;
        return true;
    }

    int64_t xs[JAI_MAX_CORNERS];
    int64_t ys[JAI_MAX_CORNERS];
    for (int i = 0; i < count; i++) {
        Value vx = jaiListGet(flat, i * 2);
        Value vy = jaiListGet(flat, i * 2 + 1);
        if (!IS_INT(vx) || !IS_INT(vy)) {
            return jaiThrow(vm.cTypeError,
                            "fill_convex(): a corner is not a pair of integers");
        }
        xs[i] = AS_INT(vx);
        ys[i] = AS_INT(vy);
    }

    convexFill(&surface, xs, ys, count, (int)shift, delta1, delta2);
    *out = NULL_VAL;
    return true;
}

static bool rasterOverflow(void) {
    return jaiThrow(vm.cOverflowError, "fill_strokes(): integer overflow");
}

/* The `x` and `y` of a point object, as integers. False when `v` is not an
 * instance with integer fields of those names, which sends the caller back to
 * its own loops: this reads points, it does not decide what one is. The slots
 * are looked up once a class. */
typedef struct {
    ObjClass  *klass;
    int        slotX, slotY;
    ObjString *nameX, *nameY;
} PointReader;

static bool readPoint(PointReader *r, Value v, int64_t *x, int64_t *y) {
    if (!IS_INSTANCE(v)) return false;
    ObjInstance *point = AS_INSTANCE(v);
    if (point->klass != r->klass) {
        r->klass = point->klass;
        r->slotX = jaiClassFieldSlot(point->klass, r->nameX);
        r->slotY = jaiClassFieldSlot(point->klass, r->nameY);
    }
    if (r->slotX < 0 || r->slotY < 0) return false;
    if (r->slotX >= point->fieldCount || r->slotY >= point->fieldCount) return false;
    const Value vx = point->fields[r->slotX];
    const Value vy = point->fields[r->slotY];
    if (!IS_INT(vx) || !IS_INT(vy)) return false;
    *x = AS_INT(vx);
    *y = AS_INT(vy);
    return true;
}

/* OpenCV's cvRound as `thick_contours` spells it out: the floor, then up when
 * the remainder is over a half, and to even when it is exactly one. */
static int64_t roundHalfEven(double value) {
    double floor = (double)(int64_t)value;
    if (floor > value) floor -= 1.0;
    int64_t whole = (int64_t)floor;
    const double part = value - floor;
    if (part > 0.5) {
        whole += 1;
    } else if (part == 0.5) {
        if (whole % 2 != 0) whole += 1;
    }
    return whole;
}

/* `fill_strokes(target, colour, cn, origin, stride, cols, rows, contours,
 *  offset_x, offset_y, thickness)` -- every contour outlined with a stroke
 *  `thickness` pixels wide, each closed back to its first point.
 *
 * jaicv's `thick_contours`, moved here whole: for each segment the
 * quadrilateral a pen of that width sweeps, then the round cap at each one's
 * far end, all with OpenCV's arithmetic -- the same `cvRound` of the same
 * products, the same corners, the same order of fills. What made it worth
 * moving is that its cost was all calls: a segment was a `fill_convex` and a
 * `fill_span` a cap row, each a fraction of a microsecond whatever it drew and
 * each mapping the picture again, and a frame contour is hundreds of segments.
 * Written in Jaithon, the passes that work out the corners were most of the
 * rest.
 *
 * `contours` is a list of lists of points, where a point is anything with
 * integer fields `x` and `y`. Returns false, having drawn nothing, if one is
 * not, so the caller can take its own path; every check is made before the
 * first pixel. Arithmetic that overflows in Jaithon throws here too, before
 * anything is drawn, as it would have there. */
static bool primFillStrokes(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *contours;
    int64_t offsetX, offsetY, thickness;
    if (!jaiArgList(args[7], 8, "fill_strokes", &contours)) return false;
    if (!jaiStrWantInt(args[8], "fill_strokes", "the x offset", &offsetX)) return false;
    if (!jaiStrWantInt(args[9], "fill_strokes", "the y offset", &offsetY)) return false;
    if (!jaiStrWantInt(args[10], "fill_strokes", "the thickness", &thickness)) return false;
    if (thickness < 2 || thickness > 32767) {
        return jaiThrow(vm.cValueError, "fill_strokes(): a thickness of %lld is out of range",
                        (long long)thickness);
    }

    /* Pass one, as `thick_contours`' first loop: every segment's two ends,
     * from the previous point to the current one, closing each contour, and
     * its squared length. */
    PointReader reader = {NULL, -1, -1, jaiStringInternC("x"), jaiStringInternC("y")};
    size_t total = 0;
    for (int c = 0; c < contours->count; c++) {
        const Value item = jaiListGet(contours, c);
        if (!IS_LIST(item)) {
            *out = BOOL_VAL(false);
            return true;
        }
        total += (size_t)AS_LIST(item)->count;
    }
    int64_t *ends = (int64_t *)malloc((total > 0 ? total : 1) * 4 * sizeof(int64_t));
    double *reach = (double *)malloc((total > 0 ? total : 1) * sizeof(double));
    if (ends == NULL || reach == NULL) {
        free(ends);
        free(reach);
        return jaiThrow(vm.cRuntimeError, "fill_strokes(): out of memory");
    }
    size_t segments = 0;
    bool readable = true;
    bool overflow = false;
    for (int c = 0; c < contours->count && readable && !overflow; c++) {
        ObjList *contour = AS_LIST(jaiListGet(contours, c));
        const int count = contour->count;
        if (count == 0) continue;
        int64_t px, py;
        if (!readPoint(&reader, jaiListGet(contour, count - 1), &px, &py)) {
            readable = false;
            break;
        }
        for (int i = 0; i < count; i++) {
            int64_t cx, cy;
            if (!readPoint(&reader, jaiListGet(contour, i), &cx, &cy)) {
                readable = false;
                break;
            }
            int64_t runX, runY, xx, yy, squared;
            int64_t *at = ends + segments * 4;
            if (__builtin_sub_overflow(px, cx, &runX) || __builtin_sub_overflow(cy, py, &runY) ||
                __builtin_mul_overflow(runX, runX, &xx) || __builtin_mul_overflow(runY, runY, &yy) ||
                __builtin_add_overflow(xx, yy, &squared) ||
                __builtin_add_overflow(px, offsetX, &at[0]) ||
                __builtin_add_overflow(py, offsetY, &at[1]) ||
                __builtin_add_overflow(cx, offsetX, &at[2]) ||
                __builtin_add_overflow(cy, offsetY, &at[3])) {
                overflow = true;
                break;
            }
            reach[segments] = (double)squared;
            segments++;
            px = cx;
            py = cy;
        }
    }
    if (!readable) {
        free(ends);
        free(reach);
        *out = BOOL_VAL(false);
        return true;
    }
    if (overflow) {
        free(ends);
        free(reach);
        return rasterOverflow();
    }

    /* Pass two: how far the quadrilateral stands off the segment, per unit of
     * its run. A segment of no length has no quadrilateral, only its cap. */
    const int64_t odd = thickness & 1;
    const int64_t halfWidth = thickness << (JAI_XY_SHIFT - 1);
    const double bias = (double)halfWidth + (double)odd * (double)JAI_XY_ONE * 0.5;
    const int64_t rounding = JAI_XY_ONE >> 1;
    for (size_t i = 0; i < segments; i++) {
        const double squared = reach[i];
        reach[i] = squared > 0 ? bias / sqrt(squared) : 0.0;
    }

    /* The cap disc, one span a row: `cap_rows` in contours.jai, the widest of
     * the up to four spans the midpoint circle names for each row. */
    const int64_t radius = (halfWidth + (JAI_XY_ONE >> 1)) >> JAI_XY_SHIFT;
    int64_t *widest = (int64_t *)malloc((size_t)(radius * 2 + 1) * sizeof(int64_t));
    if (widest == NULL) {
        free(ends);
        free(reach);
        return jaiThrow(vm.cRuntimeError, "fill_strokes(): out of memory");
    }
    for (int64_t i = 0; i < radius * 2 + 1; i++) widest[i] = -1;
    {
        int64_t err = 0, dx = radius, dy = 0, plus = 1, minus = (radius << 1) - 1;
        while (dx >= dy) {
            if (dx > widest[radius - dy]) widest[radius - dy] = dx;
            if (dx > widest[radius + dy]) widest[radius + dy] = dx;
            if (dy > widest[radius - dx]) widest[radius - dx] = dy;
            if (dy > widest[radius + dx]) widest[radius + dx] = dy;
            dy += 1;
            err += plus;
            plus += 2;
            if (err > 0) {
                err -= minus;
                dx -= 1;
                minus -= 2;
            }
        }
    }

    JaiSurface surface;
    if (!readSurface(args, 0, "fill_strokes", &surface)) {
        free(ends);
        free(reach);
        free(widest);
        return false;
    }

    /* The quadrilaterals, then the caps. */
    bool ok = true;
    for (size_t i = 0; i < segments && ok; i++) {
        const int64_t *at = ends + i * 4;
        int64_t riseY, runX;
        if (__builtin_sub_overflow(at[3], at[1], &riseY) ||
            __builtin_sub_overflow(at[0], at[2], &runX)) {
            ok = false;
            break;
        }
        const int64_t across = roundHalfEven((double)riseY * reach[i]);
        const int64_t along = roundHalfEven((double)runX * reach[i]);
        /* `<<` wraps in Jaithon; the sums are checked there and here. */
        const int64_t fromX = (int64_t)((uint64_t)at[0] << JAI_XY_SHIFT);
        const int64_t fromY = (int64_t)((uint64_t)at[1] << JAI_XY_SHIFT);
        const int64_t toX = (int64_t)((uint64_t)at[2] << JAI_XY_SHIFT);
        const int64_t toY = (int64_t)((uint64_t)at[3] << JAI_XY_SHIFT);
        int64_t xs[4], ys[4];
        if (__builtin_add_overflow(fromX, across, &xs[0]) ||
            __builtin_add_overflow(fromY, along, &ys[0]) ||
            __builtin_sub_overflow(fromX, across, &xs[1]) ||
            __builtin_sub_overflow(fromY, along, &ys[1]) ||
            __builtin_sub_overflow(toX, across, &xs[2]) ||
            __builtin_sub_overflow(toY, along, &ys[2]) ||
            __builtin_add_overflow(toX, across, &xs[3]) ||
            __builtin_add_overflow(toY, along, &ys[3])) {
            ok = false;
            break;
        }
        convexFill(&surface, xs, ys, 4, JAI_XY_SHIFT, rounding, rounding);
    }
    for (size_t i = 0; i < segments && ok; i++) {
        const int64_t toX = ends[i * 4 + 2];
        const int64_t toY = ends[i * 4 + 3];
        for (int64_t row = 0; row < radius * 2 + 1; row++) {
            const int64_t extent = widest[row];
            if (extent < 0) continue;
            int64_t x1, x2, y;
            if (__builtin_sub_overflow(toX, extent, &x1) ||
                __builtin_add_overflow(toX, extent, &x2) ||
                __builtin_add_overflow(toY, row - radius, &y)) {
                ok = false;
                break;
            }
            /* `Painter.span`: clipped to the picture's columns, nothing when
             * the clipped run is empty. */
            const int64_t start = x1 > 0 ? x1 : 0;
            const int64_t end = x2 < surface.cols - 1 ? x2 : surface.cols - 1;
            if (start > end) continue;
            surfaceRun(&surface, start, y, end - start + 1);
        }
    }
    free(ends);
    free(reach);
    free(widest);
    if (!ok) return rasterOverflow();
    *out = BOOL_VAL(true);
    return true;
}

void jaiRasterRegisterPrimitives(ObjModule *ns) {
    jaiStrDefinePrim(ns, "fill_span",     primFillSpan,     10, 10);
    jaiStrDefinePrim(ns, "draw_line",     primDrawLine,     12, 12);
    jaiStrDefinePrim(ns, "draw_lines",    primDrawLines,     9, 9);
    jaiStrDefinePrim(ns, "fill_convex",   primFillConvex,   11, 11);
    jaiStrDefinePrim(ns, "fill_strokes",  primFillStrokes,  11, 11);
}
