/* builtins_contours.c — the raster scan and border walk behind jaicv's
 * `find_contours`.
 *
 * `grid_borders` is Suzuki and Abe's border following over a whole picture in
 * one call: every border, outer and hole, in the order the raster scan meets
 * them, each with the number of the border that encloses it. It is the
 * algorithm `contours.jai` documents, moved here step for step -- the same
 * scan, the same anticlockwise first search, the same clockwise walk, the same
 * rule for when a pixel closes a border on its eastern side -- so it hands back
 * exactly what that code did. The retrieval modes, the hierarchy and the points
 * are still built up there.
 *
 * Why it is here at all. Written in Jaithon the scan was at the compiled tier's
 * floor: a load, a compare and a branch per cell over a grid of eight-byte ints,
 * about 3 ns a cell and two million cells a 1080p frame. A C transliteration of
 * that same scan over that same grid bought only 1.26x, because the cost was
 * the representation, not the language -- OpenCV is eight times quicker because
 * it reads the picture a byte at a time and skips runs of identical bytes
 * sixteen at a go. So the grid here is a byte a cell, and the scan asks a vector
 * of sixteen cells at once whether any of them is one it has to stop on. In a
 * frame of large shapes almost none are.
 *
 * The byte cannot hold a border's number, which needs as many bits as there can
 * be borders. It holds which of four states a cell is in -- clear, set and not
 * yet on any border, marked with a border's number, or marked with a border's
 * number negated -- and the number itself goes in a parallel array of ints that
 * is written only where a cell is marked and read only where the byte says it
 * was. Marked cells are the border pixels, a small fraction of the frame, so
 * that array is barely touched.
 *
 * Everything is kept between calls: the grid is filled in full every call, ring
 * included, so nothing from a previous picture survives into the next one, and
 * a border's number is only ever read under a mark this call wrote. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "runtime/builtins/text/builtins_str.h"
#include "runtime/parallel.h"
#include "runtime/runtime.h"

#include "native/native.h"
#include "vm/gc.h"

/* The four states a grid cell can be in. Clear and set are what the picture
 * says; the marks are what border following leaves behind, standing for the
 * `nbd` and `-nbd` Suzuki writes into the image itself. */
enum {
    CELL_CLEAR = 0,
    CELL_SET = 1,
    CELL_MARKED = 2,    /* +nbd */
    CELL_CLOSED = 3,    /* -nbd: the border ends at this pixel's eastern side */
};

/* Below this many cells the fill stays on one thread. */
#define JAI_BORDER_FILL_CHUNK 32768

/* A growable array of int64s, for the borders and points found. */
typedef struct {
    int64_t *items;
    size_t   count;
    size_t   capacity;
} I64Buf;

/* The working set, kept between calls. Growing these from nothing every call
 * was a realloc-and-copy cascade that measured a tenth of the whole scan. */
typedef struct {
    uint8_t *cells;
    int32_t *numbers;
    size_t   capacity;
    I64Buf   kinds;     /* per border number: outer or not, then its parent */
    I64Buf   meta;
    I64Buf   points;
    I64Buf   walk;
} BorderWork;

/* One per thread that calls in. The VM runs one Jaithon thread, so in practice
 * this is one; thread-local storage is what makes that an assumption nothing
 * depends on. */
static _Thread_local BorderWork tWork;

static bool borderGridReserve(BorderWork *work, size_t cells) {
    if (work->capacity >= cells) return true;
    uint8_t *bytes = (uint8_t *)malloc(cells);
    int32_t *numbers = (int32_t *)malloc(cells * sizeof(int32_t));
    if (bytes == NULL || numbers == NULL) {
        free(bytes);
        free(numbers);
        return false;
    }
    free(work->cells);
    free(work->numbers);
    work->cells = bytes;
    work->numbers = numbers;
    work->capacity = cells;
    return true;
}

static bool i64Grow(I64Buf *buf, size_t more) {
    if (JAI_LIKELY(buf->count + more <= buf->capacity)) return true;
    size_t cap = buf->capacity < 256 ? 256 : buf->capacity;
    while (cap < buf->count + more) cap *= 2;
    int64_t *items = (int64_t *)realloc(buf->items, cap * sizeof(int64_t));
    if (items == NULL) return false;
    buf->items = items;
    buf->capacity = cap;
    return true;
}

JAI_INLINE void i64PushUnchecked(I64Buf *buf, int64_t v) { buf->items[buf->count++] = v; }

JAI_INLINE bool i64Push(I64Buf *buf, int64_t v) {
    if (!i64Grow(buf, 1)) return false;
    buf->items[buf->count++] = v;
    return true;
}

/* --- the fill ------------------------------------------------------ */

typedef struct {
    const void    *src;
    uint8_t        srcStore;
    const uint8_t *srcBytes;
    const float   *srcFloats;
    size_t         origin;
    size_t         step;
    uint8_t       *grid;
    size_t         cols;
    size_t         stride;
    uint32_t      *bad;
} BorderFill;

/* Rows [start, end) of the picture, each with the ring cell either side of
 * it. The kinds are fixed for the call, so the switch is outside the loop and
 * every arm is a plain loop the compiler vectorises. */
static void borderFillRows(void *context, size_t start, size_t end) {
    const BorderFill *work = (const BorderFill *)context;
    const size_t cols = work->cols;
    for (size_t y = start; y < end; y++) {
        uint8_t *row = work->grid + (y + 1) * work->stride + 1;
        const size_t from = work->origin + y * work->step;
        row[-1] = CELL_CLEAR;
        row[cols] = CELL_CLEAR;
        if (work->srcFloats != NULL) {
            /* `!= 0` and not `> 0`: a NaN is not zero, and a negative value is
             * as set as a positive one. */
            const float *s = work->srcFloats + from;
            for (size_t x = 0; x < cols; x++) row[x] = s[x] != 0.0f;
            continue;
        }
        if (work->srcBytes != NULL) {
            const uint8_t *s = work->srcBytes + from;
            for (size_t x = 0; x < cols; x++) row[x] = s[x] != 0;
            continue;
        }
        switch ((ListStore)work->srcStore) {
        case LIST_STORE_F64: {
            const double *s = (const double *)work->src + from;
            for (size_t x = 0; x < cols; x++) row[x] = s[x] != 0.0;
            break;
        }
        case LIST_STORE_I64: {
            const int64_t *s = (const int64_t *)work->src + from;
            for (size_t x = 0; x < cols; x++) row[x] = s[x] != 0;
            break;
        }
        case LIST_STORE_U8: {
            const uint8_t *s = (const uint8_t *)work->src + from;
            for (size_t x = 0; x < cols; x++) row[x] = s[x] != 0;
            break;
        }
        case LIST_STORE_BOXED: {
            /* Checked as it is read rather than in a pass of its own: a boxed
             * 1080p frame is thirty-two megabytes, and reading it twice was
             * more than the whole scan. A float is set unless its bits are a
             * zero of either sign, which is the payload shifted past the sign;
             * an int is set unless its payload is zero. Anything else is
             * written as set and reported, and the caller throws. */
            const Value *s = (const Value *)work->src + from;
            uint32_t bad = 0;
            for (size_t x = 0; x < cols; x++) {
                const Value item = s[x];
                const uint64_t bits = (uint64_t)item.as.integer;
                const unsigned isFloat = item.type == VAL_FLOAT;
                bad |= (uint32_t)(!isFloat && item.type != VAL_INT);
                row[x] = (bits << isFloat) != 0;
            }
            if (bad != 0) __atomic_store_n(work->bad, 1u, __ATOMIC_RELAXED);
            break;
        }
        }
    }
}

/* --- the scan ------------------------------------------------------ */

/* The next cell in [c, limit) at which a border can start.
 *
 * An outer border starts at a set cell whose western neighbour is clear. A hole
 * border starts at a cell whose eastern neighbour is clear, if that cell is set
 * or carries a border's number unnegated. With `holes` false only the first
 * kind is asked for.
 *
 * The scan also has to know the number of the last marked cell it passed on
 * this row, which is Suzuki's LNBD. Stopping at every marked cell to read it
 * was most of the stops in a frame with holes in it -- each hole leaves four
 * marked cells across three rows -- so instead the vector loop remembers where
 * the last marked cell was, without a branch, and `lastMark` is left holding
 * the last marked cell in [c, the cell returned], or untouched if there was
 * none. Nothing writes the grid between here and the caller reading it.
 *
 * Reads one cell either side of the range, which the ring always provides:
 * `limit` is the row's eastern ring cell. */
JAI_INLINE int64_t borderNextStart(const uint8_t *g, int64_t c, int64_t limit, bool holes,
                                   int64_t *lastMark) {
#if defined(__aarch64__)
    const uint8x16_t one = vdupq_n_u8(CELL_SET);
    const uint8x16_t markBit = vdupq_n_u8(0xFE);
    int64_t markAt = -1;
    uint64_t markBits = 0;
    while (c + 16 <= limit) {
        /* Sixty-four cells at a go first. A run of identical clear or set
         * cells is nothing to the scan -- no mark, and no cell differs from
         * the one before it, which is where every start is -- so the block
         * only has to be read closely if some cell differs from its western
         * neighbour (or, for holes, the last from its eastern one) or carries
         * a mark. In a frame of large shapes that is a small minority of
         * blocks, and this test is a handful of instructions a block. */
        while (c + 64 <= limit) {
            const uint8x16_t h0 = vld1q_u8(g + c);
            const uint8x16_t h1 = vld1q_u8(g + c + 16);
            const uint8x16_t h2 = vld1q_u8(g + c + 32);
            const uint8x16_t h3 = vld1q_u8(g + c + 48);
            uint8x16_t any = vorrq_u8(veorq_u8(h0, vld1q_u8(g + c - 1)),
                                      veorq_u8(h1, vld1q_u8(g + c + 15)));
            any = vorrq_u8(any, vorrq_u8(veorq_u8(h2, vld1q_u8(g + c + 31)),
                                         veorq_u8(h3, vld1q_u8(g + c + 47))));
            if (holes) any = vorrq_u8(any, veorq_u8(h3, vld1q_u8(g + c + 49)));
            any = vorrq_u8(any, vandq_u8(vorrq_u8(vorrq_u8(h0, h1), vorrq_u8(h2, h3)), markBit));
            if (vmaxvq_u8(any) != 0) break;
            c += 64;
        }
        if (c + 16 > limit) break;
        const uint8x16_t here = vld1q_u8(g + c);
        const uint8x16_t westClear = vceqzq_u8(vld1q_u8(g + c - 1));
        const uint8x16_t isSet = vceqq_u8(here, one);
        uint8x16_t start = vandq_u8(isSet, westClear);
        if (holes) {
            const uint8x16_t eastClear = vceqzq_u8(vld1q_u8(g + c + 1));
            /* Set or marked unnegated: `here - 1` is 0 or 1. */
            const uint8x16_t open = vcleq_u8(vsubq_u8(here, one), one);
            start = vorrq_u8(start, vandq_u8(open, eastClear));
        }
        /* Four bits a lane, so a lane is a bit index over four. */
        const uint64_t marked = vget_lane_u64(
            vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(vcgtq_u8(here, one)), 4)), 0);
        const uint64_t starts = vget_lane_u64(
            vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(start), 4)), 0);
        if (starts != 0) {
            const uint64_t lowest = starts & (0 - starts);
            /* The marked lanes up to and including the start's. At lane 15 the
             * shift runs off the top and the mask is every lane, which is what
             * unsigned arithmetic gives. */
            const uint64_t before = marked & ((lowest << 4) - 1);
            if (before != 0) {
                *lastMark = c + (int64_t)((63 - __builtin_clzll(before)) >> 2);
            } else if (markAt >= 0) {
                *lastMark = markAt + (int64_t)((63 - __builtin_clzll(markBits)) >> 2);
            }
            return c + (int64_t)(__builtin_ctzll(starts) >> 2);
        }
        if (marked != 0) {
            markAt = c;
            markBits = marked;
        }
        c += 16;
    }
    if (markAt >= 0) *lastMark = markAt + (int64_t)((63 - __builtin_clzll(markBits)) >> 2);
#endif
    for (; c < limit; c++) {
        const uint8_t here = g[c];
        if (here > CELL_SET) *lastMark = c;
        if (here == CELL_SET && g[c - 1] == CELL_CLEAR) return c;
        if (holes && (here == CELL_SET || here == CELL_MARKED) && g[c + 1] == CELL_CLEAR) return c;
    }
    return limit;
}

/* The first marked cell in [c, limit), or `limit`. */
JAI_INLINE int64_t borderNextMark(const uint8_t *g, int64_t c, int64_t limit) {
#if defined(__aarch64__)
    const uint8x16_t one = vdupq_n_u8(CELL_SET);
    /* Clear and set are 0 and 1, so a block of them ORs to at most 1. */
    while (c + 64 <= limit) {
        const uint8x16_t both = vorrq_u8(vorrq_u8(vld1q_u8(g + c), vld1q_u8(g + c + 16)),
                                         vorrq_u8(vld1q_u8(g + c + 32), vld1q_u8(g + c + 48)));
        if (vmaxvq_u8(both) > CELL_SET) break;
        c += 64;
    }
    while (c + 16 <= limit) {
        const uint64_t marked = vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(
            vreinterpretq_u16_u8(vcgtq_u8(vld1q_u8(g + c), one)), 4)), 0);
        if (marked != 0) return c + (int64_t)(__builtin_ctzll(marked) >> 2);
        c += 16;
    }
#endif
    for (; c < limit; c++) {
        if (g[c] > CELL_SET) return c;
    }
    return limit;
}

/* The first cell in [c, limit) that is marked, or set with its western
 * neighbour clear. */
JAI_INLINE int64_t borderNextEdge(const uint8_t *g, int64_t c, int64_t limit) {
#if defined(__aarch64__)
    const uint8x16_t one = vdupq_n_u8(CELL_SET);
    const uint8x16_t markBit = vdupq_n_u8(0xFE);
    while (c + 16 <= limit) {
        /* As in `borderNextStart`: a block where no cell differs from its
         * western neighbour and none is marked has nothing in it. */
        while (c + 64 <= limit) {
            const uint8x16_t h0 = vld1q_u8(g + c);
            const uint8x16_t h1 = vld1q_u8(g + c + 16);
            const uint8x16_t h2 = vld1q_u8(g + c + 32);
            const uint8x16_t h3 = vld1q_u8(g + c + 48);
            uint8x16_t any = vorrq_u8(veorq_u8(h0, vld1q_u8(g + c - 1)),
                                      veorq_u8(h1, vld1q_u8(g + c + 15)));
            any = vorrq_u8(any, vorrq_u8(veorq_u8(h2, vld1q_u8(g + c + 31)),
                                         veorq_u8(h3, vld1q_u8(g + c + 47))));
            any = vorrq_u8(any, vandq_u8(vorrq_u8(vorrq_u8(h0, h1), vorrq_u8(h2, h3)), markBit));
            if (vmaxvq_u8(any) != 0) break;
            c += 64;
        }
        if (c + 16 > limit) break;
        const uint8x16_t here = vld1q_u8(g + c);
        const uint8x16_t edge = vorrq_u8(vandq_u8(vceqq_u8(here, one), vceqzq_u8(vld1q_u8(g + c - 1))),
                                         vcgtq_u8(here, one));
        const uint64_t bits = vget_lane_u64(
            vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(edge), 4)), 0);
        if (bits != 0) return c + (int64_t)(__builtin_ctzll(bits) >> 2);
        c += 16;
    }
#endif
    for (; c < limit; c++) {
        const uint8_t here = g[c];
        if (here > CELL_SET || (here == CELL_SET && g[c - 1] == CELL_CLEAR)) return c;
    }
    return limit;
}

/* The next outer border in [c, limit) that the frame encloses directly, for
 * RETR_EXTERNAL; see `primGridBorders` for why the last mark decides it.
 *
 * Two states. After an unnegated mark the scan is inside a shape already
 * walked, or in one of its holes, and nothing it meets can be a top-level
 * start until a mark says otherwise -- so it looks for the next mark and for
 * nothing else, which a vector does sixty-four cells at a time. Otherwise it
 * stops at the next mark or the next set cell with a clear cell to its west,
 * and the second is a start. A frame of large shapes, however many holes they
 * have, is two or three stops a row. */
JAI_INLINE int64_t borderNextTopStart(const uint8_t *g, int64_t c, int64_t limit, int64_t *lastMark) {
    for (;;) {
        if (*lastMark >= 0 && g[*lastMark] == CELL_MARKED) {
            c = borderNextMark(g, c, limit);
        } else {
            c = borderNextEdge(g, c, limit);
        }
        if (c >= limit) return limit;
        if (g[c] == CELL_SET) return c;
        *lastMark = c;
        c++;
    }
}

/* --- the walk ------------------------------------------------------ */

/* Mark a border pixel the way Suzuki's step 3.4 does. */
JAI_INLINE void borderMark(uint8_t *g, int32_t *numbers, int64_t cell, bool closes, int32_t nbd) {
    if (closes) {
        g[cell] = CELL_CLOSED;
        numbers[cell] = nbd;
    } else if (g[cell] == CELL_SET) {
        g[cell] = CELL_MARKED;
        numbers[cell] = nbd;
    }
}

/* Walk one border from the pixel it was entered at, appending its cells to
 * `walk`. Step for step `follow` in contours.jai: the first search turns one
 * way from the direction the border was entered from, every later one turns
 * the other way from the direction of the pixel just left, and a pixel closes
 * the border when the turn that would have looked east was reached and east
 * is clear. */
static bool borderFollow(uint8_t *g, int32_t *numbers, const int64_t offsets[8],
                         int64_t start, int entry, int32_t nbd, I64Buf *walk) {
    int64_t first = 0;
    int firstStep = -1;
    for (int turn = 1; turn <= 8; turn++) {
        const int step = (entry + turn) & 7;
        const int64_t cell = start + offsets[step];
        if (g[cell] != CELL_CLEAR) {
            first = cell;
            firstStep = step;
            break;
        }
    }
    if (firstStep < 0) {
        g[start] = CELL_CLOSED;
        numbers[start] = nbd;
        return i64Push(walk, start);
    }

    int back = firstStep;
    int64_t current = start;
    for (;;) {
        int64_t next = current;
        int nextStep = back;
        int foundAt = 8;
        for (int turn = 1; turn <= 8; turn++) {
            const int index = (back - turn + 8) & 7;
            const int64_t cell = current + offsets[index];
            if (g[cell] != CELL_CLEAR) {
                next = cell;
                nextStep = index;
                foundAt = turn;
                break;
            }
        }
        const int eastAt = back == 0 ? 8 : back;
        borderMark(g, numbers, current, foundAt >= eastAt && g[current + 1] == CELL_CLEAR, nbd);
        if (!i64Push(walk, current)) return false;
        if (next == start && current == first) return true;
        back = (nextStep + 4) & 7;
        current = next;
    }
}

/* --- output -------------------------------------------------------- */

/* Overwrite a caller's list with `count` ints. The list is the caller's and
 * may be typed or not; an int store takes the numbers as they are, a boxed one
 * takes them boxed, and anything else is boxed first. */
static bool borderWriteList(ObjList *list, const int64_t *items, size_t count) {
    if (count > (size_t)INT32_MAX) {
        return jaiThrow(vm.cRuntimeError, "grid_borders(): %zu results do not fit a list", count);
    }
    list->count = 0;
    if (!jaiListReserveExact(list, (int)count)) {
        return jaiThrow(vm.cRuntimeError, "grid_borders(): out of memory for %zu results", count);
    }
    if ((ListStore)list->stg == LIST_STORE_I64) {
        if (count > 0) memcpy(list->items, items, count * sizeof(int64_t));
    } else {
        Value *boxed = jaiListBox(list);
        for (size_t i = 0; i < count; i++) boxed[i] = INT_VAL(items[i]);
    }
    list->count = (int)count;
    jaiListTouch(list);
    return true;
}

/* The points of one walk, thinned to the ends of straight runs if asked. Two
 * steps between touching cells are the same step exactly when the two
 * differences in the padded numbering are equal, which is `compress` in
 * contours.jai. */
/* One step of a walk, from a cell to the neighbour after it, as the x and y it
 * moves by. The difference is one of the eight offsets `dy * stride + dx` with
 * `dx` and `dy` in -1..1, and with a stride of at least three the row is
 * whichever side of -1..1 the difference falls. */
JAI_INLINE void borderStep(int64_t difference, int64_t stride, int64_t *x, int64_t *y) {
    const int64_t down = difference > 1 ? 1 : (difference < -1 ? -1 : 0);
    *x += difference - down * stride;
    *y += down;
}

static bool borderEmit(BorderWork *work, int64_t stride, int64_t approx) {
    const size_t count = work->walk.count;
    const int64_t *cells = work->walk.items;
    I64Buf *points = &work->points;
    if (!i64Grow(points, count * 2)) return false;
    /* The first cell's x and y by division, and every later one's from the
     * step that reached it: a division a point was as much of a frame of
     * blobs as the walk itself. */
    const int64_t first = cells[0] / stride;
    int64_t x = cells[0] - first * stride - 1;
    int64_t y = first - 1;
    if (approx == 2 && count > 2) {
        int64_t previous = cells[count - 1];
        int64_t current = cells[0];
        for (size_t index = 0; index < count; index++) {
            const int64_t following = cells[index + 1 < count ? index + 1 : 0];
            if (current - previous != following - current) {
                i64PushUnchecked(points, x);
                i64PushUnchecked(points, y);
            }
            borderStep(following - current, stride, &x, &y);
            previous = current;
            current = following;
        }
        return true;
    }
    for (size_t index = 0; index < count; index++) {
        i64PushUnchecked(points, x);
        i64PushUnchecked(points, y);
        if (index + 1 < count) borderStep(cells[index + 1] - cells[index], stride, &x, &y);
    }
    return true;
}

/* `grid_borders(src, origin, step, rows, cols, approx, outer_only, meta, points)`
 *
 * `src` is the picture as a list of numbers, as bytes, or as a device buffer's
 * handle, and pixel (x, y) is element `origin + y * step + x` of it; a cell is
 * set when it is not zero. A device buffer is read where it lies -- storage is
 * shared -- which for a picture the GPU produced is the whole of the transfer:
 * nothing is downloaded into a list first, and a float is a quarter of what a
 * boxed list element is to read. `approx` is 1 to keep every border pixel and
 * 2 to keep only the ends of straight runs, OpenCV's CHAIN_APPROX_NONE and
 * _SIMPLE.
 *
 * Every border found is written to `meta` as three ints, in the order the scan
 * found them -- which makes border `i` the one Suzuki numbers `i + 2`, the
 * frame being 1: whether it is an outer border (1) or a hole (0), the number
 * of the border that encloses it, and where its points end in `points`, which
 * holds an x and a y per point. A border's points start where the previous
 * border's end.
 *
 * `outer_only` set asks for RETR_EXTERNAL's borders and no others: the outer
 * borders whose parent is the frame. Those come out exactly as the full scan
 * finds them, without tracing a single hole or anything inside one, which is
 * what OpenCV does too. Why that is sound: an outer border's walk depends only
 * on which cells are set, never on another border's marks, and the border
 * starts at its shape's first cell in raster order whatever else was traced.
 * So the question is only which starts to trace. A start is top-level exactly
 * when the clear cell to its west belongs to the background. With only
 * top-level shapes traced, the last mark on the row says which: a negated
 * mark is a pixel whose eastern neighbour Suzuki proved to be outside its
 * shape -- background, since the shape is top-level -- and no background cell
 * can lie between a positive mark and the start without an unmarked set cell
 * beside it on the background, which would be on a top-level outer border the
 * scan has already walked. No mark since the row began means the ring, which
 * is background. `test_find_contours.jai` holds this against the full scan. */
static bool primGridBorders(int argc, Value *args, Value *out) {
    (void)argc;
    const void *src = NULL;
    uint8_t srcStore = LIST_STORE_BOXED;
    const uint8_t *srcBytes = NULL;
    size_t srcCount = 0;
    JaiGpuBuffer *device = NULL;
    int64_t deviceBase = 0;
    if (IS_LIST(args[0])) {
        ObjList *items = AS_LIST(args[0]);
        src = items->items;
        srcStore = items->stg;
        srcCount = (size_t)items->count;
    } else if (IS_BYTES(args[0])) {
        ObjBytes *raw = AS_BYTES(args[0]);
        srcBytes = raw->data;
        srcCount = (size_t)raw->length;
    } else if (IS_INT(args[0])) {
        if (!jaiGpuBufferOf(args[0], 1, "grid_borders", &device, &deviceBase)) return false;
    } else {
        return jaiThrow(vm.cTypeError,
                        "grid_borders(): the source must be a list, bytes or a buffer, got %s",
                        jaiTypeNameStatic(args[0]));
    }

    int64_t origin, step, rows, cols, approx, outerOnly;
    if (!jaiStrWantInt(args[1], "grid_borders", "the origin", &origin)) return false;
    if (!jaiStrWantInt(args[2], "grid_borders", "the row step", &step)) return false;
    if (!jaiStrWantInt(args[3], "grid_borders", "the row count", &rows)) return false;
    if (!jaiStrWantInt(args[4], "grid_borders", "the column count", &cols)) return false;
    if (!jaiStrWantInt(args[5], "grid_borders", "the approximation", &approx)) return false;
    if (!jaiStrWantInt(args[6], "grid_borders", "the outer-only flag", &outerOnly)) return false;
    if (rows < 0 || cols < 0 || origin < 0 || step < 0) {
        return jaiThrow(vm.cValueError,
                        "grid_borders(): origin, step, rows and cols cannot be negative");
    }
    if (rows > 1 && step < cols) {
        return jaiThrow(vm.cValueError,
                        "grid_borders(): a row step of %lld is shorter than %lld columns",
                        (long long)step, (long long)cols);
    }
    if (approx != 1 && approx != 2) {
        return jaiThrow(vm.cValueError,
                        "grid_borders(): approximation %lld is neither 1 (every pixel) nor 2 (run ends)",
                        (long long)approx);
    }
    if (rows > INT32_MAX || cols > INT32_MAX || step > INT32_MAX || origin > INT32_MAX * 4LL) {
        return jaiThrow(vm.cValueError, "grid_borders(): %lldx%lld is too large",
                        (long long)rows, (long long)cols);
    }
    if (!IS_LIST(args[7]) || !IS_LIST(args[8])) {
        return jaiThrow(vm.cTypeError, "grid_borders(): meta and points must be lists");
    }
    ObjList *metaOut = AS_LIST(args[7]);
    ObjList *pointsOut = AS_LIST(args[8]);

    /* One past the last element read; nothing is read when there are no
     * pixels. */
    const int64_t reach = (rows == 0 || cols == 0) ? 0 : origin + (rows - 1) * step + cols;
    const float *srcFloats = NULL;
    if (device != NULL) {
        if (reach > 0) {
            srcFloats = jaiGpuMapRead(device, (size_t)deviceBase, (size_t)reach);
            if (srcFloats == NULL) {
                return jaiThrow(vm.cValueError,
                                "grid_borders(): %lldx%lld at %lld with a step of %lld is outside the buffer",
                                (long long)rows, (long long)cols, (long long)origin, (long long)step);
            }
        }
    } else if ((size_t)reach > srcCount) {
        return jaiThrow(vm.cValueError,
                        "grid_borders(): %lldx%lld at %lld with a step of %lld needs %lld source values, got %zu",
                        (long long)rows, (long long)cols, (long long)origin, (long long)step,
                        (long long)reach, srcCount);
    }

    const int64_t stride = cols + 2;
    const int64_t total = stride * (rows + 2);
    /* A border number is an int32 in the side array, and there cannot be more
     * borders than cells. */
    if (total > (int64_t)INT32_MAX) {
        return jaiThrow(vm.cValueError, "grid_borders(): %lldx%lld is too large",
                        (long long)rows, (long long)cols);
    }
    BorderWork *work = &tWork;
    if (!borderGridReserve(work, (size_t)total)) {
        return jaiThrow(vm.cRuntimeError, "grid_borders(): out of memory for a %lldx%lld grid",
                        (long long)rows, (long long)cols);
    }
    uint8_t *g = work->cells;
    int32_t *numbers = work->numbers;

    /* The ring's top and bottom rows; the fill writes its sides. */
    memset(g, CELL_CLEAR, (size_t)stride);
    memset(g + (rows + 1) * stride, CELL_CLEAR, (size_t)stride);
    uint32_t bad = 0;
    BorderFill fill = {src, srcStore, srcBytes, srcFloats, (size_t)origin, (size_t)step,
                       g, (size_t)cols, (size_t)stride, &bad};
    jaiParallelChunks((size_t)rows, cols > 0 ? (JAI_BORDER_FILL_CHUNK / (size_t)cols) + 1 : 1,
                      borderFillRows, &fill);
    if (bad != 0) {
        const Value *boxed = (const Value *)src;
        for (int64_t y = 0; y < rows; y++) {
            for (int64_t x = 0; x < cols; x++) {
                const size_t i = (size_t)(origin + y * step + x);
                if (!IS_FLOAT(boxed[i]) && !IS_INT(boxed[i])) {
                    return jaiThrow(vm.cTypeError,
                                    "grid_borders(): source element %zu is %s, not a number",
                                    i, jaiTypeNameStatic(boxed[i]));
                }
            }
        }
    }

    /* East, then clockwise with y down -- `STEP_X` and `STEP_Y` in
     * contours.jai -- as single additions in the padded numbering. */
    static const int stepY[8] = {0, 1, 1, 1, 0, -1, -1, -1};
    static const int stepX[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    int64_t offsets[8];
    for (int turn = 0; turn < 8; turn++) offsets[turn] = stepY[turn] * stride + stepX[turn];

    /* What each border number is: an outer border or a hole, and its parent.
     * Index 0 is the frame, number 1, a hole with no parent. */
    I64Buf *kinds = &work->kinds;
    I64Buf *meta = &work->meta;
    kinds->count = 0;
    meta->count = 0;
    work->points.count = 0;
    const bool holes = outerOnly == 0;
    bool ok = i64Push(kinds, 0) && i64Push(kinds, 0);
    int32_t nbd = 1;

    for (int64_t y = 0; ok && y < rows; y++) {
        const int64_t row = (y + 1) * stride + 1;
        const int64_t limit = row + cols;
        /* The last marked cell the scan has passed on this row; none means
         * the frame, number 1. */
        int64_t lastMark = -1;
        int64_t c = row;
        for (;;) {
            c = holes ? borderNextStart(g, c, limit, true, &lastMark)
                      : borderNextTopStart(g, c, limit, &lastMark);
            if (c >= limit) break;
            const uint8_t here = g[c];
            const bool outer = here == CELL_SET && g[c - 1] == CELL_CLEAR;
            int64_t parent = 1;
            if (holes) {
                const int32_t lastNbd = lastMark >= 0 ? numbers[lastMark] : 1;
                const int64_t ancestor = kinds->items[(lastNbd - 1) * 2 + 1];
                if (kinds->items[(lastNbd - 1) * 2] != 0) parent = outer ? ancestor : lastNbd;
                else parent = outer ? lastNbd : ancestor;
            }

            nbd++;
            work->walk.count = 0;
            if (!borderFollow(g, numbers, offsets, c, outer ? 4 : 0, nbd, &work->walk) ||
                !borderEmit(work, stride, approx) || !i64Grow(meta, 3) || !i64Grow(kinds, 2)) {
                ok = false;
                break;
            }
            i64PushUnchecked(kinds, outer ? 1 : 0);
            i64PushUnchecked(kinds, parent);
            i64PushUnchecked(meta, outer ? 1 : 0);
            i64PushUnchecked(meta, parent);
            i64PushUnchecked(meta, (int64_t)(work->points.count / 2));

            /* The walk marked the pixel it started from, and the row carries
             * that number forward from where the scan picks up. */
            lastMark = c;
            c++;
        }
    }

    if (!ok) return jaiThrow(vm.cRuntimeError, "grid_borders(): out of memory");
    jaiGCPushRoot(OBJ_VAL(metaOut));
    jaiGCPushRoot(OBJ_VAL(pointsOut));
    ok = borderWriteList(metaOut, meta->items, meta->count) &&
         borderWriteList(pointsOut, work->points.items, work->points.count);
    jaiGCPopRoots(2);
    if (!ok) return false;
    *out = NULL_VAL;
    return true;
}

/* --- points --------------------------------------------------------- */

/* Element `i` of a list of ints, whatever its storage. False when it is not
 * an int. */
JAI_INLINE bool borderInt(const ObjList *list, int i, int64_t *out) {
    if ((ListStore)list->stg == LIST_STORE_I64) {
        *out = ((const int64_t *)list->items)[i];
        return true;
    }
    const Value v = jaiListGet(list, i);
    if (!IS_INT(v)) return false;
    *out = AS_INT(v);
    return true;
}

/* `grid_border_points(klass, meta, xy, chosen, dx, dy)` -- the contours
 * `find_contours` hands back, built here: for each border index in `chosen`,
 * in that order, a list of `klass` objects, one a point of that border as
 * `grid_borders` wrote it, moved by (dx, dy).
 *
 * A point is made the way `__prim__.obj_new` and two field stores make one:
 * an instance of `klass` with `x` and `y` set and nothing else run. That is
 * only the same object `klass(x, y)` makes when the class is nothing but
 * those two integer fields and its `init` assigns them, which is what jaicv's
 * `Point` is -- so a class with any other field is refused, and `types.jai`
 * says beside `Point` that its `init` has to stay that way. Null for a class
 * this does not take; the caller then builds the points itself.
 *
 * Why. A frame of blobs is a few thousand contours and a couple of hundred
 * thousand points, and building each point through the class call was most of
 * `find_contours`: 178660 points over 3600 contours took 8.3 ms of a 10.3 ms
 * call, against 2 ms for the scan, the walk and everything else. */
static bool primGridBorderPoints(int argc, Value *args, Value *out) {
    (void)argc;
    if (!IS_CLASS(args[0])) {
        return jaiThrow(vm.cTypeError, "grid_border_points(): the first argument must be a class");
    }
    ObjClass *klass = AS_CLASS(args[0]);
    ObjList *meta, *xy, *chosen;
    int64_t dx, dy;
    if (!jaiArgList(args[1], 2, "grid_border_points", &meta)) return false;
    if (!jaiArgList(args[2], 3, "grid_border_points", &xy)) return false;
    if (!jaiArgList(args[3], 4, "grid_border_points", &chosen)) return false;
    if (!jaiStrWantInt(args[4], "grid_border_points", "the x offset", &dx)) return false;
    if (!jaiStrWantInt(args[5], "grid_border_points", "the y offset", &dy)) return false;
    *out = NULL_VAL;

    const int slotX = jaiClassFieldSlot(klass, jaiStringInternC("x"));
    const int slotY = jaiClassFieldSlot(klass, jaiStringInternC("y"));
    if (klass->fieldCount != 2 || slotX < 0 || slotY < 0) return true;

    /* Every range checked before anything is made. */
    const int borders = meta->count / 3;
    const int pairs = xy->count / 2;
    const int count = chosen->count;
    for (int c = 0; c < count; c++) {
        int64_t index, start = 0, stop;
        if (!borderInt(chosen, c, &index) || index < 0 || index >= borders) {
            return jaiThrow(vm.cValueError, "grid_border_points(): border %d is out of range", c);
        }
        if (index > 0 && !borderInt(meta, (int)index * 3 - 1, &start)) start = -1;
        if (!borderInt(meta, (int)index * 3 + 2, &stop)) stop = -1;
        if (start < 0 || stop < start || stop > pairs) {
            return jaiThrow(vm.cValueError, "grid_border_points(): border %lld's points are out of range",
                            (long long)index);
        }
    }

    ObjList *contours = jaiListNew(count);
    if (contours == NULL) return false;
    jaiGCPushRoot(OBJ_VAL(contours));
    contours->elemKind = FIELD_KIND_LIST;
    bool ok = jaiListReserveExact(contours, count);
    for (int c = 0; ok && c < count; c++) {
        int64_t index, start = 0, stop;
        borderInt(chosen, c, &index);
        if (index > 0) borderInt(meta, (int)index * 3 - 1, &start);
        borderInt(meta, (int)index * 3 + 2, &stop);
        const int length = (int)(stop - start);

        ObjList *points = jaiListNew(length);
        if (points == NULL) {
            ok = false;
            break;
        }
        /* Reachable from `contours` before anything else is allocated. */
        jaiListBox(contours)[c] = OBJ_VAL(points);
        contours->count = c + 1;
        if (!jaiListReserveExact(points, length)) {
            ok = false;
            break;
        }
        /* The collector does not move anything, so the array stays put while
         * the points below are allocated. */
        Value *slots = length > 0 ? jaiListBox(points) : NULL;
        for (int i = 0; i < length; i++) {
            int64_t x, y;
            const int at = (int)start + i;
            if (!borderInt(xy, at * 2, &x) || !borderInt(xy, at * 2 + 1, &y)) {
                jaiGCPopRoot();
                return jaiThrow(vm.cTypeError, "grid_border_points(): a point is not a pair of ints");
            }
            if (__builtin_add_overflow(x, dx, &x) || __builtin_add_overflow(y, dy, &y)) {
                jaiGCPopRoot();
                return jaiThrow(vm.cOverflowError, "grid_border_points(): integer overflow");
            }
            ObjInstance *point = jaiInstanceNew(klass);
            if (point == NULL) {
                ok = false;
                break;
            }
            point->fields[slotX] = INT_VAL(x);
            point->fields[slotY] = INT_VAL(y);
            slots[i] = OBJ_VAL(point);
            points->count = i + 1;
        }
        jaiListTouch(points);
    }
    jaiGCPopRoot();
    if (!ok) return jaiThrow(vm.cRuntimeError, "grid_border_points(): out of memory");
    jaiListTouch(contours);
    *out = OBJ_VAL(contours);
    return true;
}

void jaiContoursRegisterPrimitives(ObjModule *ns) {
    jaiStrDefinePrim(ns, "grid_borders", primGridBorders, 9, 9);
    jaiStrDefinePrim(ns, "grid_border_points", primGridBorderPoints, 6, 6);
}
