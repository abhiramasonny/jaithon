#: The Metal source behind every base primitive, and the cache in front of it.
from std.gpu import Kernel

let SOURCE = """#include <metal_stdlib>
using namespace metal;

constexpr constant uint GROUP = 256u;

/* Total `value` over a whole threadgroup, handing every thread the answer.
 *
 * Every column statistic here sums over the samples, the axis that is not
 * being written: one thread per column would leave a GPU running a whole
 * matrix's work on as many threads as the data has features. A threadgroup
 * strides the samples instead and reduces through this. Barriered at both
 * ends, so the same scratch carries the next reduction. */
static inline float group_total(
    float value,
    threadgroup float *lanes,
    uint lid,
    uint lane,
    uint warp
) {
    if (lid < 32u) lanes[lid] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    value = simd_sum(value);
    if (lane == 0u) lanes[warp] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint j = 0u; j < 32u; j++) total += lanes[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

/* The same shape for a bound. A thread that drew no elements passes the
 * identity, which never wins. */
static inline float group_bound(
    float value,
    threadgroup float *lanes,
    uint lid,
    uint lane,
    uint warp,
    bool largest
) {
    float empty = largest ? -INFINITY : INFINITY;
    if (lid < 32u) lanes[lid] = empty;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    value = largest ? simd_max(value) : simd_min(value);
    if (lane == 0u) lanes[warp] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float best = empty;
    for (uint j = 0u; j < 32u; j++) best = largest ? max(best, lanes[j]) : min(best, lanes[j]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return best;
}

/* Whether a float is a NaN, read out of its bits.
 *
 * `v != v` is the usual spelling and it does not work: the runtime compiles
 * every shader with the default `MTLCompileOptions`, whose fast math is on,
 * and that licenses the compiler to assume no operand is a NaN and fold the
 * comparison to false. Missing values in a design matrix arrive as NaN, so
 * `nan_mask` returned all zeros until this replaced it. */
static inline bool not_a_number(float v) {
    uint bits = as_type<uint>(v);
    return (bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0u;
}

/* A uniform draw in (0, 1) from two integers, so a kernel can be random with
 * no generator state carried between dispatches. splitmix64, because its
 * avalanche is good enough that neighbouring thread ids give uncorrelated
 * draws -- a plain xorshift of `id` does not. */
static inline float splitmix(uint a, uint b) {
    ulong state = ((ulong)a + 1ul) * 0x9E3779B97F4A7C15ul + (ulong)b * 0xBF58476D1CE4E5B9ul;
    state ^= state >> 30; state *= 0xBF58476D1CE4E5B9ul;
    state ^= state >> 27; state *= 0x94D049BB133111EBul;
    state ^= state >> 31;
    return (float((state >> 40) & 0xFFFFFFul) + 1.0f) * (1.0f / 16777217.0f);
}

/* A strided copy: `stride` 1 is a plain one, and `order + 1` over a square
 * matrix is its diagonal. */
kernel void base_copy(device const float* src [[buffer(0)]],
                      device float* dst [[buffer(1)]],
                      constant uint& count [[buffer(2)]],
                      constant uint& stride [[buffer(3)]],
                      uint id [[thread_position_in_grid]]) {
    if (id < count) dst[id] = src[id * stride];
}

kernel void base_gather_columns(device const float* x [[buffer(0)]],
                                device const float* picks [[buffer(1)]],
                                device float* out [[buffer(2)]],
                                constant uint& rows [[buffer(3)]],
                                constant uint& width [[buffer(4)]],
                                constant uint& kept [[buffer(5)]],
                                uint id [[thread_position_in_grid]]) {
    if (id >= rows * kept) return;
    uint row = id / kept;
    out[id] = x[row * width + uint(picks[id - row * kept])];
}

kernel void base_place_block(device const float* part [[buffer(0)]],
                             device float* out [[buffer(1)]],
                             constant uint& count [[buffer(2)]],
                             constant uint& part_width [[buffer(3)]],
                             constant uint& total_width [[buffer(4)]],
                             constant uint& offset [[buffer(5)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint row = id / part_width;
    out[row * total_width + offset + id - row * part_width] = part[id];
}

/* One threadgroup per column, striding the samples: mode 0 leaves the mean and
 * the variance, 1 the smallest and largest, 2 the mean over the entries
 * `mask` leaves at zero.
 *
 * The variance takes a second pass rather than `E[x^2] - E[x]^2`, which is one
 * pass and is what a first draft writes. On pixel bytes that puts the answer's
 * leading digits inside the cancellation, and a scaler then divides by a
 * variance good to two figures. The second pass rereads a matrix that is
 * already in cache. */
kernel void base_column_reduce(device const float* x [[buffer(0)]],
                               device const float* mask [[buffer(1)]],
                               device float* first [[buffer(2)]],
                               device float* second [[buffer(3)]],
                               constant uint& rows [[buffer(4)]],
                               constant uint& width [[buffer(5)]],
                               constant uint& mode [[buffer(6)]],
                               constant uint& ddof [[buffer(7)]],
                               uint column [[threadgroup_position_in_grid]],
                               uint lid [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup float lanes[32];
    if (column >= width) return;
    if (mode == 1u) {
        float low = INFINITY;
        float high = -INFINITY;
        for (uint row = lid; row < rows; row += GROUP) {
            float value = x[row * width + column];
            low = min(low, value);
            high = max(high, value);
        }
        float smallest = group_bound(low, lanes, lid, lane, warp, false);
        float largest = group_bound(high, lanes, lid, lane, warp, true);
        if (lid == 0u) { first[column] = smallest; second[column] = largest; }
        return;
    }
    float running = 0.0f;
    float seen = 0.0f;
    for (uint row = lid; row < rows; row += GROUP) {
        uint at = row * width + column;
        if (mode == 2u && mask[at] != 0.0f) continue;
        running += x[at];
        seen += 1.0f;
    }
    float total = group_total(running, lanes, lid, lane, warp);
    if (mode == 2u) {
        float count = group_total(seen, lanes, lid, lane, warp);
        /* A column with nothing left averages to zero and not to a NaN: an
         * imputer that returns NaN has not imputed anything. */
        if (lid == 0u) first[column] = count > 0.0f ? total / count : 0.0f;
        return;
    }
    float centre = total / float(max(rows, 1u));
    float spread = 0.0f;
    for (uint row = lid; row < rows; row += GROUP) {
        float gap = x[row * width + column] - centre;
        spread += gap * gap;
    }
    float squares = group_total(spread, lanes, lid, lane, warp);
    if (lid != 0u) return;
    first[column] = centre;
    second[column] = squares / float(max(rows - min(ddof, rows), 1u));
}

/* Fold one partial column statistic `(a, b, c)` into another.
 *
 * Mode 0 carries `(count, mean, M2)` and merges by Chan's rule, which is
 * Welford's update between two groups: one pass, and no partial ever holds a
 * raw sum of squares. That alone is NOT as stable as the two passes
 * `base_column_reduce` makes -- a float32 running mean rounds at the ulp of
 * the mean, so a column whose mean dwarfs its spread lost up to two hundred
 * times the variance's accuracy -- which is why every value is folded
 * relative to its column's first sample (see `base_column_partials`). Mode 1
 * carries `(min, max)` and mode 2 `(count, sum)`. */
static inline void column_fold(
    thread float &a,
    thread float &b,
    thread float &c,
    float a2,
    float b2,
    float c2,
    uint mode
) {
    if (mode == 1u) { a = min(a, a2); b = max(b, b2); return; }
    if (mode == 2u) { a += a2; b += b2; return; }
    float total = a + a2;
    if (total == 0.0f) return;
    float delta = b2 - b;
    b += delta * (a2 / total);
    c += c2 + delta * delta * (a * (a2 / total));
    a = total;
}

/* One tile of rows and one span of columns, as a partial statistic per
 * column.
 *
 * `base_column_reduce` puts a threadgroup on each column, which is as many
 * threadgroups as the data has features -- fifty, on a machine that wants
 * thousands of threads -- and has each read with the row width as its stride.
 * Here the threads of a group sit across the columns instead (`span` of them,
 * then the next row in the next `span`), so a group's reads are whole
 * consecutive rows. The groups form a grid of row tiles by column spans:
 * group `g` takes tile `g / chunks` and span `g % chunks`, so a wide matrix
 * with few rows still gets a group per span rather than a handful of groups
 * walking every span in turn. Each thread folds its column's values with
 * Welford's update, the group's lanes fold together in threadgroup memory,
 * and lane 0 writes the tile's partial for the column at
 * `partials[(column * tiles + tile) * 3]`.
 *
 * Mode 0 folds `value - x[column]`, the column's first sample: every tile
 * shares that shift, so Chan's merge is unchanged, and `base_column_finish`
 * adds it back to the mean. A shifted column's mean is of the order of its
 * spread, so the running mean no longer rounds at the scale of an offset the
 * variance does not depend on. */
kernel void base_column_partials(device const float* x [[buffer(0)]],
                                 device const float* mask [[buffer(1)]],
                                 device float* partials [[buffer(2)]],
                                 constant uint& rows [[buffer(3)]],
                                 constant uint& width [[buffer(4)]],
                                 constant uint& mode [[buffer(5)]],
                                 constant uint& tile [[buffer(6)]],
                                 constant uint& tiles [[buffer(7)]],
                                 uint group [[threadgroup_position_in_grid]],
                                 uint lid [[thread_index_in_threadgroup]]) {
    threadgroup float first[GROUP];
    threadgroup float second[GROUP];
    threadgroup float third[GROUP];
    uint span = min(width, GROUP);
    uint lanes = GROUP / span;
    uint lane = lid / span;
    uint offset = lid - lane * span;
    uint chunks = (width + span - 1u) / span;
    uint which = group / chunks;
    uint column = (group - which * chunks) * span + offset;
    uint start = which * tile;
    uint stop = min(start + tile, rows);
    bool live = lane < lanes && column < width;
    float a = mode == 1u ? INFINITY : 0.0f;
    float b = mode == 1u ? -INFINITY : 0.0f;
    float c = 0.0f;
    if (live) {
        float shift = mode == 0u && rows > 0u ? x[column] : 0.0f;
        for (uint row = start + lane; row < stop; row += lanes) {
            uint at = row * width + column;
            float value = x[at];
            if (mode == 1u) {
                a = min(a, value);
                b = max(b, value);
            } else if (mode == 2u) {
                if (mask[at] != 0.0f) continue;
                a += 1.0f;
                b += value;
            } else {
                value -= shift;
                a += 1.0f;
                float delta = value - b;
                b += delta / a;
                c += delta * (value - b);
            }
        }
    }
    first[lid] = a;
    second[lid] = b;
    third[lid] = c;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (live && lane == 0u) {
        for (uint other = 1u; other < lanes; ++other) {
            uint slot = other * span + offset;
            column_fold(a, b, c, first[slot], second[slot], third[slot], mode);
        }
        uint at = (column * tiles + which) * 3u;
        partials[at] = a;
        partials[at + 1u] = b;
        partials[at + 2u] = c;
    }
}

/* Every column's partials folded into its answer, a threadgroup a column:
 * each thread folds a stride of the partials and the group folds the threads
 * as a tree. Writes what `base_column_reduce` writes for the same mode. */
kernel void base_column_finish(device const float* partials [[buffer(0)]],
                               device float* out_first [[buffer(1)]],
                               device float* out_second [[buffer(2)]],
                               device const float* x [[buffer(3)]],
                               constant uint& width [[buffer(4)]],
                               constant uint& mode [[buffer(5)]],
                               constant uint& groups [[buffer(6)]],
                               constant uint& ddof [[buffer(7)]],
                               uint column [[threadgroup_position_in_grid]],
                               uint lid [[thread_index_in_threadgroup]]) {
    threadgroup float first[GROUP];
    threadgroup float second[GROUP];
    threadgroup float third[GROUP];
    if (column >= width) return;
    float a = mode == 1u ? INFINITY : 0.0f;
    float b = mode == 1u ? -INFINITY : 0.0f;
    float c = 0.0f;
    for (uint group = lid; group < groups; group += GROUP) {
        uint at = (column * groups + group) * 3u;
        column_fold(a, b, c, partials[at], partials[at + 1u], partials[at + 2u], mode);
    }
    first[lid] = a;
    second[lid] = b;
    third[lid] = c;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint half_span = GROUP / 2u; half_span > 0u; half_span >>= 1) {
        if (lid < half_span) {
            column_fold(a, b, c, first[lid + half_span], second[lid + half_span],
                        third[lid + half_span], mode);
            first[lid] = a;
            second[lid] = b;
            third[lid] = c;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid != 0u) return;
    if (mode == 1u) { out_first[column] = a; out_second[column] = b; return; }
    if (mode == 2u) {
        /* As in `base_column_reduce`: nothing left averages to zero. */
        out_first[column] = a > 0.0f ? b / a : 0.0f;
        return;
    }
    /* `base_column_partials` folded every value less the first sample. */
    out_first[column] = a > 0.0f ? b + x[column] : 0.0f;
    out_second[column] = c / max(a - float(min(ddof, uint(a))), 1.0f);
}

/* Every column copied into a power-of-two tall scratch, the pad set to
 * infinity so it sorts past the real values and never reaches an answer. */
kernel void base_pad_columns(device const float* x [[buffer(0)]],
                             device float* values [[buffer(1)]],
                             device float* order [[buffer(2)]],
                             constant uint& rows [[buffer(3)]],
                             constant uint& padded [[buffer(4)]],
                             constant uint& width [[buffer(5)]],
                             constant uint& keep_order [[buffer(6)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= padded * width) return;
    uint row = id / width;
    values[id] = row < rows ? x[id] : INFINITY;
    if (keep_order != 0u) order[id] = float(row);
}

/* One compare-exchange stage of a bitonic sort, every column at once.
 *
 * Bitonic because the network is fixed: the host walks log^2(padded) stages,
 * a couple of hundred dispatches for a million samples, and never touches the
 * data. The thread index is compacted onto the pairs that actually swap --
 * `i` is the slot with bit `step` clear -- so the dispatch is half a column
 * rather than a whole one with half its threads returning. Adjacent threads
 * take adjacent columns, which is what makes the strided reads coalesce. Ties
 * are not broken by row, so `argsort_columns` is not stable. */
kernel void base_bitonic_step(device float* values [[buffer(0)]],
                              device float* order [[buffer(1)]],
                              constant uint& padded [[buffer(2)]],
                              constant uint& width [[buffer(3)]],
                              constant uint& span [[buffer(4)]],
                              constant uint& step [[buffer(5)]],
                              constant uint& keep_order [[buffer(6)]],
                              uint id [[thread_position_in_grid]]) {
    if (id >= (padded >> 1) * width) return;
    uint column = id % width;
    uint slot = id / width;
    uint i = ((slot & ~(step - 1u)) << 1) | (slot & (step - 1u));
    uint here = i * width + column;
    uint there = (i | step) * width + column;
    float mine = values[here];
    float yours = values[there];
    if ((mine > yours) == ((i & span) == 0u)) {
        values[here] = yours;
        values[there] = mine;
        if (keep_order != 0u) {
            float tag = order[here];
            order[here] = order[there];
            order[there] = tag;
        }
    }
}

kernel void base_quantile_pick(device const float* sorted [[buffer(0)]],
                               device const float* fractions [[buffer(1)]],
                               device float* out [[buffer(2)]],
                               constant uint& rows [[buffer(3)]],
                               constant uint& width [[buffer(4)]],
                               constant uint& count [[buffer(5)]],
                               uint id [[thread_position_in_grid]]) {
    if (id >= count * width) return;
    uint which = id / width;
    uint column = id - which * width;
    float position = clamp(fractions[which], 0.0f, 1.0f) * float(rows - 1u);
    uint below = uint(floor(position));
    uint above = min(below + 1u, rows - 1u);
    out[id] = mix(sorted[below * width + column], sorted[above * width + column],
                  position - float(below));
}

/* `|a - b|^2 = |a|^2 + |b|^2 - 2 a.b`, applied in place to the product.
 *
 * Rounding drops a point's distance from itself a hair below zero, and a
 * square root of that is a NaN that spreads through the whole matrix, so the
 * clamp is not decoration. */
kernel void base_dist_finish(device float* out [[buffer(0)]],
                             device const float* left [[buffer(1)]],
                             device const float* right [[buffer(2)]],
                             constant uint& rows [[buffer(3)]],
                             constant uint& columns [[buffer(4)]],
                             constant uint& first [[buffer(5)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= rows * columns) return;
    uint row = id / columns;
    out[id] = max(left[first + row] + right[id - row * columns] - 2.0f * out[id], 0.0f);
}

kernel void base_gram_finish(device float* out [[buffer(0)]],
                             device const float* shape [[buffer(1)]],
                             constant uint& count [[buffer(2)]],
                             constant uint& kind [[buffer(3)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float value = out[id];
    if (kind == 1u) out[id] = exp(-shape[0] * value);
    else if (kind == 2u) out[id] = pow(shape[0] * value + shape[2], shape[1]);
    else if (kind == 3u) out[id] = tanh(shape[0] * value + shape[2]);
}

/* One thread per row, striding the columns.
 *
 * The reverse -- a threadgroup per row, which is what the column statistics do
 * -- is wrong here: these matrices are tall and narrow, `[n_samples,
 * n_classes]` or `[n_samples, n_clusters]`, so the rows alone fill the machine
 * and the columns would not fill a simdgroup. Ties go to the lowest column,
 * which is what every argmax in the package promises. */
kernel void base_row_extreme(device const float* m [[buffer(0)]],
                             device float* index [[buffer(1)]],
                             device float* value [[buffer(2)]],
                             constant uint& rows [[buffer(3)]],
                             constant uint& columns [[buffer(4)]],
                             constant uint& largest [[buffer(5)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= rows) return;
    float best = largest != 0u ? -INFINITY : INFINITY;
    uint pick = 0u;
    for (uint column = 0u; column < columns; ++column) {
        float here = m[id * columns + column];
        if (largest != 0u ? here > best : here < best) { best = here; pick = column; }
    }
    index[id] = float(pick);
    value[id] = best;
}

/* The `k` best entries of each row, in order.
 *
 * No per-thread scratch, because Metal will not size a thread array from a
 * kernel argument: each pass takes the best entry strictly below the previous
 * pick, comparing `(value, column)` as a pair so duplicates are handed out one
 * at a time instead of filling every slot. That is `k` passes over the row,
 * the right trade while `k` is the handful of neighbours a kNN asks for and
 * not a sort in disguise. */
kernel void base_row_topk(device const float* m [[buffer(0)]],
                          device float* values [[buffer(1)]],
                          device float* index [[buffer(2)]],
                          constant uint& rows [[buffer(3)]],
                          constant uint& columns [[buffer(4)]],
                          constant uint& k [[buffer(5)]],
                          constant uint& largest [[buffer(6)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= rows) return;
    float sign = largest != 0u ? 1.0f : -1.0f;
    float ceiling = 0.0f;
    uint ceiling_at = 0u;
    for (uint slot = 0u; slot < k; ++slot) {
        float best = -INFINITY;
        uint pick = columns;
        for (uint column = 0u; column < columns; ++column) {
            float here = sign * m[id * columns + column];
            if (slot != 0u && !(here < ceiling || (here == ceiling && column > ceiling_at))) {
                continue;
            }
            if (here > best || (here == best && column < pick)) { best = here; pick = column; }
        }
        values[id * k + slot] = sign * best;
        index[id * k + slot] = float(pick);
        ceiling = best;
        ceiling_at = pick;
    }
}

/* Whether `(s, i)` outranks `(t, j)`: the larger score, and on a tie the
 * smaller column. Every top-k here ranks by this one rule, so the tie order
 * is the same whichever kernel answered. */
static inline bool topk_before(float s, uint i, float t, uint j) {
    return s > t || (s == t && i < j);
}

/* The `k` best entries of each row in one pass, a threadgroup a row.
 *
 * `base_row_topk` gives a row one thread and `k` passes over it, so a
 * thousand-row distance matrix ran on a thousand threads making five strided
 * sweeps each. Here a row's threads take its columns a group-width apart --
 * adjacent threads, adjacent columns, so the reads coalesce -- and each keeps
 * its own best `K` in registers: `K` is a template constant so the insertion
 * below unrolls onto fixed slots, and a runtime `k` at most `K` reads only the
 * head of it. Then `k` rounds pick the best head across the group, a simd
 * butterfly and one slot per simdgroup in threadgroup memory, and the thread
 * that owned the winner pops it. Ranking is `topk_before` on `sign * value`,
 * so equal values come out smallest column first, as `base_row_topk` does.
 *
 * `fused` reads `m` as the raw product `a.bT` and ranks the distance
 * `max(left[first + row] + right[j] - 2 m, 0)` that `base_dist_finish` would
 * have written, so a neighbour search never makes that pass over the whole
 * matrix. Rows are written at `first + row`, which is how a search over bands
 * of queries fills one answer. A row with fewer than `k` ranked entries pads
 * with column `columns`, as `base_row_topk` does.
 *
 * `fused` 2 is the direct search for narrow rows: `m` is the training matrix
 * `b`, `left` the queries `a`, both `width <= DIRECT_WIDTH` wide, and the
 * ranked value is `sum((a_row - b_j)^2)` taken term by term. The expanded
 * form cancels catastrophically when the squared norms dwarf the distance --
 * one feature, or a few far from the origin -- and at this width the
 * difference costs no more than the product would. A query holding a NaN or
 * an infinity ranks nothing, so its row is all padding. */
constexpr constant uint DIRECT_WIDTH = 8u;

static inline bool not_finite(float v) {
    return (as_type<uint>(v) & 0x7F800000u) == 0x7F800000u;
}

template <uint K>
kernel void base_row_topk_group(device const float* m [[buffer(0)]],
                                device float* values [[buffer(1)]],
                                device float* index [[buffer(2)]],
                                device const float* left [[buffer(3)]],
                                device const float* right [[buffer(4)]],
                                constant uint& rows [[buffer(5)]],
                                constant uint& columns [[buffer(6)]],
                                constant uint& k [[buffer(7)]],
                                constant uint& largest [[buffer(8)]],
                                constant uint& fused [[buffer(9)]],
                                constant uint& first [[buffer(10)]],
                                constant uint& width [[buffer(11)]],
                                uint row [[threadgroup_position_in_grid]],
                                uint lid [[thread_index_in_threadgroup]],
                                uint threads [[threads_per_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup float heads[32];
    threadgroup uint owners[32];
    if (row >= rows) return;
    const uint none = 0xFFFFFFFFu;
    float sign = largest != 0u ? 1.0f : -1.0f;
    float best[K];
    uint at[K];
    #pragma unroll
    for (uint p = 0u; p < K; ++p) { best[p] = -INFINITY; at[p] = none; }
    device const float* line = m + ulong(row) * ulong(columns);
    float offset = fused == 1u ? left[first + row] : 0.0f;
    float q[DIRECT_WIDTH];
    uint reach = columns;
    if (fused == 2u) {
        bool bad = false;
        #pragma unroll
        for (uint c = 0u; c < DIRECT_WIDTH; ++c) {
            q[c] = c < width ? left[ulong(first + row) * ulong(width) + c] : 0.0f;
            bad = bad || not_finite(q[c]);
        }
        if (bad) reach = 0u;
    }
    for (uint j = lid; j < reach; j += threads) {
        float value;
        if (fused == 2u) {
            device const float* p = m + ulong(j) * ulong(width);
            value = 0.0f;
            #pragma unroll
            for (uint c = 0u; c < DIRECT_WIDTH; ++c) {
                if (c < width) {
                    float gap = q[c] - p[c];
                    value += gap * gap;
                }
            }
        } else {
            value = line[j];
            if (fused == 1u) value = max(offset + right[j] - 2.0f * value, 0.0f);
        }
        float s = sign * value;
        if (!topk_before(s, j, best[K - 1u], at[K - 1u])) continue;
        best[K - 1u] = s;
        at[K - 1u] = j;
        #pragma unroll
        for (uint p = K - 1u; p > 0u; --p) {
            if (topk_before(best[p], at[p], best[p - 1u], at[p - 1u])) {
                float ts = best[p]; best[p] = best[p - 1u]; best[p - 1u] = ts;
                uint ti = at[p]; at[p] = at[p - 1u]; at[p - 1u] = ti;
            }
        }
    }
    uint warps = (threads + 31u) / 32u;
    for (uint slot = 0u; slot < k; ++slot) {
        float s = best[0];
        uint i = at[0];
        for (ushort hop = 16; hop > 0; hop >>= 1) {
            float s2 = simd_shuffle_xor(s, hop);
            uint i2 = simd_shuffle_xor(i, hop);
            if (topk_before(s2, i2, s, i)) { s = s2; i = i2; }
        }
        if (lane == 0u) { heads[warp] = s; owners[warp] = i; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float top = heads[0];
        uint pick = owners[0];
        for (uint w = 1u; w < warps; ++w) {
            if (topk_before(heads[w], owners[w], top, pick)) { top = heads[w]; pick = owners[w]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0u) {
            ulong out = ulong(first + row) * ulong(k) + slot;
            values[out] = sign * top;
            index[out] = float(pick == none ? columns : pick);
        }
        if (at[0] == pick) {
            #pragma unroll
            for (uint p = 0u; p + 1u < K; ++p) { best[p] = best[p + 1u]; at[p] = at[p + 1u]; }
            best[K - 1u] = -INFINITY;
            at[K - 1u] = none;
        }
    }
}

/* Each query's `k` neighbours, found by the expanded form, given their exact
 * squared distances and put back in order, a thread a query.
 *
 * `|a|^2 + |b|^2 - 2 a.b` in float32 is wrong by about an ulp of the squared
 * norms, which is more than the distance between near neighbours once the
 * norms are large: an exact duplicate came back at a distance of five. The
 * search already holds the right rows; this recomputes `sum((a - b)^2)` for
 * just those `k` and sorts them by `(distance, index)`, the order every top-k
 * here uses. A query holding a NaN or an infinity has no neighbours: every
 * slot is padded with index `points` and distance infinity. */
kernel void base_nearest_refine(device const float* a [[buffer(0)]],
                                device const float* b [[buffer(1)]],
                                device float* values [[buffer(2)]],
                                device float* index [[buffer(3)]],
                                constant uint& queries [[buffer(4)]],
                                constant uint& width [[buffer(5)]],
                                constant uint& k [[buffer(6)]],
                                constant uint& points [[buffer(7)]],
                                uint id [[thread_position_in_grid]]) {
    if (id >= queries) return;
    device const float* query = a + ulong(id) * ulong(width);
    device float* near = values + ulong(id) * ulong(k);
    device float* at = index + ulong(id) * ulong(k);
    bool bad = false;
    for (uint c = 0u; c < width; ++c) bad = bad || not_finite(query[c]);
    for (uint j = 0u; j < k; ++j) {
        uint i = uint(at[j]);
        if (bad || i >= points) {
            near[j] = INFINITY;
            at[j] = float(points);
            continue;
        }
        device const float* row = b + ulong(i) * ulong(width);
        float total = 0.0f;
        for (uint c = 0u; c < width; ++c) {
            float gap = query[c] - row[c];
            total += gap * gap;
        }
        near[j] = total;
    }
    for (uint j = 1u; j < k; ++j) {
        float v = near[j];
        float i = at[j];
        uint p = j;
        while (p > 0u && (near[p - 1u] > v || (near[p - 1u] == v && at[p - 1u] > i))) {
            near[p] = near[p - 1u];
            at[p] = at[p - 1u];
            --p;
        }
        near[p] = v;
        at[p] = i;
    }
}

template [[host_name("base_row_topk4")]] [[kernel]] decltype(base_row_topk_group<4>) base_row_topk_group<4>;
template [[host_name("base_row_topk8")]] [[kernel]] decltype(base_row_topk_group<8>) base_row_topk_group<8>;
template [[host_name("base_row_topk16")]] [[kernel]] decltype(base_row_topk_group<16>) base_row_topk_group<16>;
template [[host_name("base_row_topk32")]] [[kernel]] decltype(base_row_topk_group<32>) base_row_topk_group<32>;

/* One total per class and column, a threadgroup striding the samples.
 *
 * `power` 0 counts each class and ignores `values`, 1 sums the column and 2
 * sums its square. Between them those are the sufficient statistics of every
 * prior, centroid and Gaussian in the package, so naive Bayes and LDA each fit
 * in three dispatches. */
kernel void base_class_moment(device const float* values [[buffer(0)]],
                              device const float* labels [[buffer(1)]],
                              device float* out [[buffer(2)]],
                              constant uint& rows [[buffer(3)]],
                              constant uint& width [[buffer(4)]],
                              constant uint& classes [[buffer(5)]],
                              constant uint& power [[buffer(6)]],
                              uint slot [[threadgroup_position_in_grid]],
                              uint lid [[thread_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup float lanes[32];
    if (slot >= classes * width) return;
    uint which = slot / width;
    uint column = slot - which * width;
    float running = 0.0f;
    for (uint row = lid; row < rows; row += GROUP) {
        if (uint(labels[row] + 0.5f) != which) continue;
        if (power == 0u) { running += 1.0f; continue; }
        float value = values[row * width + column];
        running += power == 1u ? value : value * value;
    }
    float total = group_total(running, lanes, lid, lane, warp);
    if (lid == 0u) out[slot] = total;
}

/* Rows added into the slots their indices name, with device atomics.
 *
 * Two samples landing in one slot arrive in an order nothing fixes, and float
 * addition is not associative, so the last bits of a total move between runs
 * on the same data. The same trade `jaitensor.gather_rows` documents: making
 * it exact means sorting the indices, which costs more than the
 * reproducibility is worth for totals that go on to be divided by thousands. */
kernel void base_scatter_add_rows(device const float* src [[buffer(0)]],
                                  device const float* index [[buffer(1)]],
                                  device float* dst [[buffer(2)]],
                                  constant uint& count [[buffer(3)]],
                                  constant uint& width [[buffer(4)]],
                                  constant uint& capacity [[buffer(5)]],
                                  uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint row = id / width;
    float asked = index[row];
    if (not_a_number(asked) || asked < 0.0f || uint(asked) >= capacity) return;
    atomic_fetch_add_explicit(
        (device atomic<float>*)(dst + uint(asked) * width + id - row * width),
        src[id],
        memory_order_relaxed
    );
}

kernel void base_affine_columns(device const float* x [[buffer(0)]],
                                device const float* scale [[buffer(1)]],
                                device const float* offset [[buffer(2)]],
                                device float* out [[buffer(3)]],
                                constant uint& count [[buffer(4)]],
                                constant uint& width [[buffer(5)]],
                                uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint column = id % width;
    out[id] = x[id] * scale[column] + offset[column];
}

kernel void base_mask_op(device const float* a [[buffer(0)]],
                         device const float* b [[buffer(1)]],
                         device float* out [[buffer(2)]],
                         constant uint& count [[buffer(3)]],
                         constant uint& op [[buffer(4)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    if (op == 0u) out[id] = a[id] == b[id] ? 1.0f : 0.0f;
    else out[id] = not_a_number(a[id]) ? 1.0f : 0.0f;
}

/* How far each sample is from one centre, kept against the best so far. */
kernel void base_seed_distance(device const float* x [[buffer(0)]],
                               device const float* centres [[buffer(1)]],
                               device float* nearest [[buffer(2)]],
                               constant uint& rows [[buffer(3)]],
                               constant uint& width [[buffer(4)]],
                               constant uint& centre [[buffer(5)]],
                               constant uint& replace [[buffer(6)]],
                               uint id [[thread_position_in_grid]]) {
    if (id >= rows) return;
    float total = 0.0f;
    for (uint column = 0u; column < width; ++column) {
        float gap = x[id * width + column] - centres[centre * width + column];
        total += gap * gap;
    }
    nearest[id] = replace != 0u ? total : min(nearest[id], total);
}

/* One key per sample, whose largest is a draw proportional to the distance.
 *
 * `-log(u) / w` is exponential with rate `w`, so the first of those to fire is
 * sample `i` with probability `w_i / sum(w)`: k-means++ exactly, without the
 * running total and the linear walk over the samples the textbook version does
 * on the host. That walk is what made jaicv's seeding two hundred of the two
 * hundred and fifty milliseconds a VGA frame took. The reciprocal is emitted
 * instead of the key itself so the winner is a maximum, which is the reduction
 * jaitensor already has tuned; a sample at no distance emits zero and so never
 * wins. */
kernel void base_seed_keys(device const float* nearest [[buffer(0)]],
                           device float* keys [[buffer(1)]],
                           constant uint& rows [[buffer(2)]],
                           constant uint& salt [[buffer(3)]],
                           constant uint& round [[buffer(4)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= rows) return;
    float weight = nearest[id];
    keys[id] = weight > 0.0f ? weight / -log(splitmix(id ^ (round * 0x9E3779B9u), salt)) : 0.0f;
}

/* Shifted by the row maximum, because a mixture's log densities run to the
 * hundreds and `exp` of one of those is an infinity long before the ratio the
 * caller wants is anything unusual. */
kernel void base_log_sum_exp_rows(device const float* m [[buffer(0)]],
                                  device float* out [[buffer(1)]],
                                  constant uint& rows [[buffer(2)]],
                                  constant uint& columns [[buffer(3)]],
                                  uint id [[thread_position_in_grid]]) {
    if (id >= rows) return;
    float peak = -INFINITY;
    for (uint column = 0u; column < columns; ++column) peak = max(peak, m[id * columns + column]);
    if (not_a_number(peak) || peak == INFINITY || peak == -INFINITY) {
        out[id] = peak;
        return;
    }
    float total = 0.0f;
    for (uint column = 0u; column < columns; ++column) total += exp(m[id * columns + column] - peak);
    out[id] = peak + log(total);
}

/* The distinct values of a sorted column, compacted with an atomic counter.
 *
 * A prefix sum would keep them in order and costs a two-level scan; the
 * counter costs four lines and returns the same set in an arbitrary one, which
 * the host sorts -- a list of at most `capacity` classes, small by
 * construction, where the samples behind it are not. */
kernel void base_unique_from_sorted(device const float* sorted [[buffer(0)]],
                                    device atomic_uint* counter [[buffer(1)]],
                                    device float* out [[buffer(2)]],
                                    constant uint& count [[buffer(3)]],
                                    constant uint& capacity [[buffer(4)]],
                                    uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    if (id > 0u && sorted[id] == sorted[id - 1u]) return;
    uint slot = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    if (slot < capacity) out[slot] = sorted[id];
}

/* The distinct values of a vector, gathered into an open-addressed table of
 * `slots` (a power of two) float bit patterns that starts out all `EMPTY`.
 *
 * Each sample hashes its bits and probes linearly: a slot that already holds
 * them is a plain read, so once the handful of classes are in, the samples
 * behind them cost one load each and no atomic; an empty slot is claimed by
 * compare-and-swap, and the winner bumps `counter[0]`. A NaN, which is no
 * class, or a sample that probes every slot without a home sets `counter[1]`.
 * `-0.0` is folded onto `0.0`, which is how the sorted route's `==` saw
 * them. */
kernel void base_distinct_hash(device const float* values [[buffer(0)]],
                               device atomic_uint* table [[buffer(1)]],
                               device atomic_uint* counter [[buffer(2)]],
                               constant uint& count [[buffer(3)]],
                               constant uint& slots [[buffer(4)]],
                               uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    const uint empty = 0xFFFFFFFFu;
    float value = values[id];
    if (not_a_number(value)) {
        atomic_store_explicit(&counter[1], 0x3F800000u, memory_order_relaxed);
        return;
    }
    uint bits = value == 0.0f ? 0u : as_type<uint>(value);
    /* The HIGH bits of the product: a product's low bits depend only on the
     * multiplicand's low bits, and an integer-valued float's low mantissa
     * bits are all zero, so taking the low ones put every label from 0 to
     * 1023 on one home slot and made each sample walk the whole chain. */
    uint at = slots > 1u ? (bits * 2654435761u) >> (32u - ctz(slots)) : 0u;
    for (uint probe = 0u; probe < slots; ++probe) {
        uint held = atomic_load_explicit(&table[at], memory_order_relaxed);
        if (held == bits) return;
        if (held == empty) {
            uint expected = empty;
            if (atomic_compare_exchange_weak_explicit(&table[at], &expected, bits,
                                                      memory_order_relaxed,
                                                      memory_order_relaxed)) {
                atomic_fetch_add_explicit(&counter[0], 1u, memory_order_relaxed);
                return;
            }
            if (expected == bits) return;
            /* Lost the slot to another value: look at this one again. */
            if (expected == empty) { probe--; continue; }
        }
        at = (at + 1u) & (slots - 1u);
    }
    atomic_store_explicit(&counter[1], 0x3F800000u, memory_order_relaxed);
}

/* `base_distinct_hash`'s table set to EMPTY and its two counters to zero. */
kernel void base_distinct_reset(device uint* table [[buffer(0)]],
                                device uint* counter [[buffer(1)]],
                                constant uint& slots [[buffer(2)]],
                                uint id [[thread_position_in_grid]]) {
    if (id < slots) table[id] = 0xFFFFFFFFu;
    if (id < 2u) counter[id] = 0u;
}

kernel void base_counter_value(device const atomic_uint* counter [[buffer(0)]],
                               device float* out [[buffer(1)]],
                               uint id [[thread_position_in_grid]]) {
    if (id == 0u) out[0] = float(atomic_load_explicit(counter, memory_order_relaxed));
}

/* Class values to codes and back. The table is short enough that a linear scan
 * beats anything with a search in it. An unknown value encodes to -1 rather
 * than to a plausible class, and the host checks for it once. */
kernel void base_map_labels(device const float* input [[buffer(0)]],
                            device const float* classes [[buffer(1)]],
                            device float* out [[buffer(2)]],
                            constant uint& count [[buffer(3)]],
                            constant uint& classes_n [[buffer(4)]],
                            constant uint& decode [[buffer(5)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float value = input[id];
    if (decode != 0u) {
        uint code = uint(max(value, 0.0f) + 0.5f);
        out[id] = code < classes_n ? classes[code] : NAN;
        return;
    }
    float found = -1.0f;
    for (uint which = 0u; which < classes_n; ++which) {
        if (classes[which] == value) { found = float(which); break; }
    }
    out[id] = found;
}

/* One column of a Cholesky factor: `phase` 0 is the diagonal on one thread,
 * `phase` 1 the entries below it on one thread each. They cannot share a
 * dispatch because there is no barrier across a grid. A non-positive pivot
 * makes the root a NaN, deliberately -- the host reads the finished diagonal
 * back and turns it into an exception. */
kernel void base_chol_step(device const float* a [[buffer(0)]],
                           device float* l [[buffer(1)]],
                           constant uint& order [[buffer(2)]],
                           constant uint& column [[buffer(3)]],
                           constant uint& phase [[buffer(4)]],
                           uint id [[thread_position_in_grid]]) {
    if (phase == 0u) {
        if (id != 0u) return;
        float total = a[column * order + column];
        for (uint k = 0u; k < column; ++k) {
            float seen = l[column * order + k];
            total -= seen * seen;
        }
        l[column * order + column] = sqrt(total);
        return;
    }
    uint row = column + 1u + id;
    if (row >= order) return;
    float total = a[row * order + column];
    for (uint k = 0u; k < column; ++k) total -= l[row * order + k] * l[column * order + k];
    l[row * order + column] = total / l[column * order + column];
}

/* One row of a forward or backward substitution, every right-hand side at
 * once. Sequential in the rows by nature, so `order` dispatches; the matrices
 * this runs on are as wide as the features or the components, never as the
 * samples. */
kernel void base_tri_row(device const float* l [[buffer(0)]],
                         device float* x [[buffer(1)]],
                         constant uint& order [[buffer(2)]],
                         constant uint& rhs [[buffer(3)]],
                         constant uint& row [[buffer(4)]],
                         constant uint& upper [[buffer(5)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= rhs) return;
    float total = x[row * rhs + id];
    if (upper == 0u) {
        for (uint k = 0u; k < row; ++k) total -= l[row * order + k] * x[k * rhs + id];
    } else {
        for (uint k = row + 1u; k < order; ++k) total -= l[k * order + row] * x[k * rhs + id];
    }
    x[row * rhs + id] = total / l[row * order + row];
}

kernel void base_box_muller(device const float* first [[buffer(0)]],
                            device const float* second [[buffer(1)]],
                            device float* out [[buffer(2)]],
                            constant uint& count [[buffer(3)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    out[id] = sqrt(-2.0f * log(first[id])) * cos(6.28318530718f * second[id]);
}
"""

var _kernels: dict[str, Kernel] = {}

#: The compiled pipeline for one entry point, compiled once per process.
#:
#: The cache is keyed on the entry name alone and shared by the whole package,
#: so every module must prefix its entry points with its own short name --
#: `nb_log_density`, `svm_smo_gradient`, `tree_histogram`. Two modules that
#: both called one `assign` would silently run the same kernel.
pub fn cached_kernel(source: str, entry: str) -> Kernel {
    let existing = _kernels.get(entry, null)
    if existing is not null { return existing }
    let compiled = Kernel.compile(source, entry)
    _kernels[entry] = compiled
    return compiled
}

pub fn base_kernel(entry: str) -> Kernel { return cached_kernel(SOURCE, entry) }
