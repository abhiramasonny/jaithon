#: `group_by` and everything a `GroupBy` handle does: aggregate, transform,
#: filter, and the row-picking selectors.
#:
#: One idea carries the whole module. `keys.jai:factorize` numbers every row
#: with the group it belongs to, and that lane of codes is what turns a
#: group-wise reduction into one segmented dispatch: a kernel reads a row, asks
#: the codes lane which output slot it belongs to, and writes there. Nothing in
#: here ever loops over the groups and reduces each one — that is the shape
#: pandas has to take because its groups are Python objects, and it is the
#: shape that would make a hundred thousand groups a hundred thousand
#: dispatches.
#:
#: There are two ways to spend the codes lane, and the handle picks between
#: them per call:
#:
#: - **Scatter** — one thread per row, atomically accumulating into
#:   `out[code]`. Nothing to set up, perfectly coalesced reads, and the only
#:   cost is that every row of a group contends for one slot.
#: - **Segment** — lay the rows out by code once, then give one thread each
#:   `(group, column)` pair and let it walk its own contiguous run. The layout
#:   costs two passes over the rows and a scan, and it is the only path that
#:   serves an aggregation the device has no atomic for: minimum, maximum,
#:   product, and the row-pickers.
#:
#: `SEGMENT_ROWS` is the crossover. Below it the groups are short, contention
#: is mild, and the scatter wins because it sets nothing up; above it one slot
#: is being fought over by enough rows that the dispatch serialises and paying
#: for the layout is cheaper. The layout is cached on the handle, so once
#: something has asked for a minimum every later aggregation takes the
#: segmented path for free.
#:
#: What is not here: `factorize` and the hashing under it are `keys.jai`,
#: `resample` is `timeseries.jai`, `pivot`/`unstack` are `reshape.jai`.
#:
#: ```
#: let sales = group_by(frame, "city")
#: defer { sales.free() }
#: let totals = sales.sum()
#: let ranked = sales.transform(Agg.Mean)
#: ```
import std.gpu as gpu
from std.gpu import Buffer, Kernel

from .column import Agg, Column, DataFrame, Dtype, Index, Series, Value
from .column import MAX_ROWS, NA
from .column import cached_kernel as column_kernel
from .column import concat_columns, f32_bits, prefix_sum
from .keys import NO_GROUP, factorize

#: Rows per group at which the segmented path starts to beat the scatter.
#:
#: A float atomic add to a contended address serialises, so a group of `n`
#: rows costs about `n` serialised adds on the scatter path and one thread's
#: walk of `n` rows on the segmented one — a wash, except that the segmented
#: path first has to lay the rows out, which is two passes over the rows and a
#: scan. Sixty-four rows a group is where that setup is repaid. It is reasoned,
#: not swept; a measurement should move it.
const SEGMENT_ROWS = 64

#: Threadgroup width of the segmented kernels. The shader stages `ORDER_TILE`
#: order entries per pass through threadgroup memory; that width is its own.
const SEGMENT_GROUP = 256

#: How wide the stable layout's histogram may get, in float slots.
#:
#: The layout counts each chunk's rows per group into a `chunks x groups`
#: matrix and scans it, so the matrix width is the parallelism and its area is
#: the scan's cost. Eight million slots is 32 MB: a thousand groups get 4096
#: chunks, a hundred thousand get 83, and a million get 8. Past that the layout
#: pass runs on very few threads — the documented limit of `cumcount`, `head`,
#: `tail` and `nth`, which are the calls a user reaches for when the groups are
#: few anyway.
const LAYOUT_SLOTS = 8388608
const LAYOUT_CHUNKS = 4096

#: Slices a tall group is split into, and the thread budget that picks the
#: count. One thread per `(group, column)` under-threads a frame with a handful
#: of groups, so a segment is cut into slices that reduce independently and a
#: fold of at most `MAX_SLICES` partials finishes it.
const THREAD_BUDGET = 65536
const MAX_SLICES = 64

#: Digits of the per-group quantile selection, and how many buckets that is.
const SELECT_BITS = 4
const SELECT_BUCKETS = 16

const OP_SUM = 0
const OP_COUNT = 1
const OP_NONZERO = 2
const OP_DEV2 = 3
const OP_PROD = 4
const OP_MIN = 5
const OP_MAX = 6
const OP_FIRST = 7
const OP_LAST = 8

const FINISH_DIV = 0
const FINISH_VAR = 1
const FINISH_STD = 2
const FINISH_ANY = 3
const FINISH_ALL = 4
const FINISH_LERP = 5

const PICK_HEAD = 0
const PICK_TAIL = 1
const PICK_NTH = 2

let SOURCE = """#include <metal_stdlib>
using namespace metal;

constexpr constant uint SEGMENT_GROUP = 256;
constexpr constant uint ORDER_TILE = 1024;
constexpr constant uint SELECT_BUCKETS = 16;
constexpr constant long WIDE_SCALE = 16777216L;

constexpr constant uint OP_SUM = 0u;
constexpr constant uint OP_COUNT = 1u;
constexpr constant uint OP_NONZERO = 2u;
constexpr constant uint OP_DEV2 = 3u;
constexpr constant uint OP_PROD = 4u;
constexpr constant uint OP_MIN = 5u;
constexpr constant uint OP_MAX = 6u;
constexpr constant uint OP_FIRST = 7u;
constexpr constant uint OP_LAST = 8u;

constexpr constant uint FINISH_DIV = 0u;
constexpr constant uint FINISH_VAR = 1u;
constexpr constant uint FINISH_STD = 2u;
constexpr constant uint FINISH_ANY = 3u;
constexpr constant uint FINISH_ALL = 4u;
constexpr constant uint FINISH_LERP = 5u;

constexpr constant uint PICK_HEAD = 0u;
constexpr constant uint PICK_TAIL = 1u;
constexpr constant uint PICK_NTH = 2u;

/* The two lanes of a wide dtype are recombined in `long`, exactly as
 * column.jai does it. A minimum over timestamps that went through `float`
 * would compare milliseconds a hundred thousand apart as equal. */
inline long wide_value(float low, float high) {
    return (long)high * WIDE_SCALE + (long)low;
}

inline void wide_store(device float* low, device float* high, uint at, long value) {
    long hi = value >> 24;
    low[at] = (float)(value - (hi << 24));
    high[at] = (float)hi;
}

inline float row_value(device const float* values,
                       device const float* wide,
                       uint has_wide,
                       uint row) {
    return (has_wide != 0u) ? (float)wide_value(values[row], wide[row]) : values[row];
}

/* A NaN in a lane with no validity is still a missing value — a column that
 * came out of arithmetic rather than out of a reader has no mask to say so.
 * Every reduction here skips both, which is pandas' `skipna=True`. */
inline bool row_present(device const float* valid, uint has_valid, uint row, float value) {
    if (has_valid != 0u && valid[row] == 0.0f) return false;
    return !isnan(value);
}

inline float fold_start(uint op) {
    return (op == OP_PROD) ? 1.0f : 0.0f;
}

inline float fold_step(uint op, float accumulated, float value, float centre) {
    if (op == OP_SUM) return accumulated + value;
    if (op == OP_COUNT) return accumulated + 1.0f;
    if (op == OP_NONZERO) return accumulated + ((value != 0.0f) ? 1.0f : 0.0f);
    if (op == OP_PROD) return accumulated * value;
    float deviation = value - centre;
    return accumulated + deviation * deviation;
}

inline float fold_join(uint op, float a, float b) {
    return (op == OP_PROD) ? (a * b) : (a + b);
}

inline bool picks_a_row(uint op) {
    return op >= OP_MIN;
}

/* Ties go to the earliest row, which is the row `idxmax` reports and the row
 * `first` means. Because the rule never looks at anything but the row number
 * to break a tie, a slice may walk its share of the segment in any order. */
inline bool pick_beats(uint op,
                       device const float* values,
                       device const float* wide,
                       uint has_wide,
                       uint row,
                       uint best) {
    if (op == OP_FIRST) return row < best;
    if (op == OP_LAST) return row > best;
    if (has_wide != 0u) {
        long a = wide_value(values[row], wide[row]);
        long b = wide_value(values[best], wide[best]);
        if (a != b) return (op == OP_MIN) ? (a < b) : (a > b);
        return row < best;
    }
    float a = values[row];
    float b = values[best];
    if (a != b) return (op == OP_MIN) ? (a < b) : (a > b);
    return row < best;
}

/* Floats in the unsigned order their bits already almost have: flip the sign
 * bit of a positive, invert a negative whole. Unsigned comparison on the
 * result is float comparison, which is what lets the quantile selection be a
 * radix walk rather than a sort. */
inline uint order_key(float value) {
    uint bits = as_type<uint>(value);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

inline float order_value(uint key) {
    uint bits = (key & 0x80000000u) ? (key & 0x7FFFFFFFu) : ~key;
    return as_type<float>(bits);
}

kernel void gb_group_counts(device const float* codes [[buffer(0)]],
                            device const float* keep [[buffer(1)]],
                            device float* counts [[buffer(2)]],
                            constant uint& count [[buffer(3)]],
                            constant uint& use_keep [[buffer(4)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (use_keep != 0u && keep[gid] == 0.0f) return;
    int code = (int)codes[gid];
    if (code < 0) return;
    atomic_fetch_add_explicit((device atomic<float>*)(counts + code), 1.0f, memory_order_relaxed);
}

/* Pass one of the stable layout: how many rows of each group live in each
 * chunk. The matrix is group-major, so the exclusive scan of the whole thing
 * is at once the group's start and its running offset within the group — the
 * previous groups' totals sit ahead of it by construction. That is why no
 * separate `starts` lane is needed here and why the layout comes out stable:
 * a chunk is a contiguous run of rows, and one thread walks it in order. */
kernel void gb_chunk_counts(device const float* codes [[buffer(0)]],
                            device float* histogram [[buffer(1)]],
                            constant uint& count [[buffer(2)]],
                            constant uint& chunks [[buffer(3)]],
                            constant uint& span [[buffer(4)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= chunks) return;
    uint begin = gid * span;
    if (begin >= count) return;
    uint end = min(begin + span, count);
    for (uint row = begin; row < end; row++) {
        int code = (int)codes[row];
        if (code < 0) continue;
        histogram[(uint)code * chunks + gid] += 1.0f;
    }
}

kernel void gb_chunk_place(device const float* codes [[buffer(0)]],
                           device float* offsets [[buffer(1)]],
                           device const float* starts [[buffer(2)]],
                           device float* order [[buffer(3)]],
                           device float* rank [[buffer(4)]],
                           constant uint& count [[buffer(5)]],
                           constant uint& chunks [[buffer(6)]],
                           constant uint& span [[buffer(7)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= chunks) return;
    uint begin = gid * span;
    if (begin >= count) return;
    uint end = min(begin + span, count);
    for (uint row = begin; row < end; row++) {
        int code = (int)codes[row];
        if (code < 0) {
            rank[row] = -1.0f;
            continue;
        }
        uint slot = (uint)offsets[(uint)code * chunks + gid];
        order[slot] = (float)row;
        rank[row] = (float)(slot - (uint)starts[code]);
        offsets[(uint)code * chunks + gid] = (float)(slot + 1u);
    }
}

/* The scatter path: one thread per row, one atomic per row.
 *
 * The quad path is taken when the payload is a single lane and the row count
 * divides by four. The four rows still need four atomics — their codes are
 * unrelated — but the value, mask and code loads become one transaction
 * apiece instead of twelve. */
inline void scatter_one(device float* out,
                        device const float* centres,
                        uint op,
                        int code,
                        float value,
                        bool present) {
    if (code < 0 || !present) return;
    float add = 1.0f;
    if (op == OP_SUM) add = value;
    else if (op == OP_NONZERO) add = (value != 0.0f) ? 1.0f : 0.0f;
    else if (op == OP_DEV2) {
        float deviation = value - centres[code];
        add = deviation * deviation;
    }
    atomic_fetch_add_explicit((device atomic<float>*)(out + code), add, memory_order_relaxed);
}

kernel void gb_scatter_add(device const float* values [[buffer(0)]],
                           device const float* wide [[buffer(1)]],
                           device const float* valid [[buffer(2)]],
                           device const float* codes [[buffer(3)]],
                           device const float* centres [[buffer(4)]],
                           device float* out [[buffer(5)]],
                           constant uint& count [[buffer(6)]],
                           constant uint& op [[buffer(7)]],
                           constant uint& has_wide [[buffer(8)]],
                           constant uint& has_valid [[buffer(9)]],
                           constant uint& quads [[buffer(10)]],
                           uint gid [[thread_position_in_grid]]) {
    if (quads != 0u) {
        if (gid * 4u >= count) return;
        float4 value = ((device const float4*)values)[gid];
        float4 code = ((device const float4*)codes)[gid];
        float4 mask = (has_valid != 0u) ? ((device const float4*)valid)[gid] : float4(1.0f);
        scatter_one(out, centres, op, (int)code.x, value.x, mask.x != 0.0f && !isnan(value.x));
        scatter_one(out, centres, op, (int)code.y, value.y, mask.y != 0.0f && !isnan(value.y));
        scatter_one(out, centres, op, (int)code.z, value.z, mask.z != 0.0f && !isnan(value.z));
        scatter_one(out, centres, op, (int)code.w, value.w, mask.w != 0.0f && !isnan(value.w));
        return;
    }
    if (gid >= count) return;
    float value = row_value(values, wide, has_wide, gid);
    scatter_one(out, centres, op, (int)codes[gid], value,
                row_present(valid, has_valid, gid, value));
}

/* The segmented path.
 *
 * A threadgroup covers 256 consecutive groups of one slice, so the order
 * entries its threads want are one contiguous stretch — neighbouring threads
 * read overlapping cache lines and nothing else in the dispatch touches them.
 * Staging that stretch through threadgroup memory turns 256 independent
 * scattered walks into cooperative coalesced loads.
 *
 * The grid is padded to a whole number of threadgroups per slice so that a
 * threadgroup never straddles two slices; the position is one division and a
 * subtraction, because the device has no integer divide to spend on a modulo
 * as well. */
kernel void gb_segment_reduce(device const float* values [[buffer(0)]],
                              device const float* wide [[buffer(1)]],
                              device const float* valid [[buffer(2)]],
                              device const float* order [[buffer(3)]],
                              device const float* starts [[buffer(4)]],
                              device const float* centres [[buffer(5)]],
                              device float* partials [[buffer(6)]],
                              device float* picks [[buffer(7)]],
                              constant uint& groups [[buffer(8)]],
                              constant uint& unit [[buffer(9)]],
                              constant uint& slices [[buffer(10)]],
                              constant uint& op [[buffer(11)]],
                              constant uint& has_wide [[buffer(12)]],
                              constant uint& has_valid [[buffer(13)]],
                              uint gid [[thread_position_in_grid]],
                              uint lid [[thread_index_in_threadgroup]],
                              uint wid [[threadgroup_position_in_grid]]) {
    threadgroup float tile[ORDER_TILE];
    uint slice = gid / unit;
    uint group = gid - slice * unit;
    uint block = wid * SEGMENT_GROUP;
    uint block_slice = block / unit;
    uint first = block - block_slice * unit;
    uint base = (first < groups) ? (uint)starts[first] : 0u;
    uint stop = (first < groups) ? (uint)starts[min(first + SEGMENT_GROUP, groups)] : 0u;
    bool live = group < groups;
    uint at = live ? (uint)starts[group] + slice : 0u;
    uint end = live ? (uint)starts[group + 1u] : 0u;
    float accumulated = fold_start(op);
    uint best = 0xFFFFFFFFu;
    while (base < stop) {
        uint staged = min(ORDER_TILE, stop - base);
        for (uint j = lid; j < staged; j += SEGMENT_GROUP) tile[j] = order[base + j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        while (at < base + staged && at < end) {
            uint row = (uint)tile[at - base];
            float value = row_value(values, wide, has_wide, row);
            if (row_present(valid, has_valid, row, value)) {
                if (picks_a_row(op)) {
                    if (best == 0xFFFFFFFFu || pick_beats(op, values, wide, has_wide, row, best)) {
                        best = row;
                    }
                } else {
                    accumulated = fold_step(op, accumulated, value,
                                            (op == OP_DEV2) ? centres[group] : 0.0f);
                }
            }
            at += slices;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        base += staged;
    }
    if (!live) return;
    partials[slice * groups + group] = accumulated;
    picks[slice * groups + group] = (best == 0xFFFFFFFFu) ? -1.0f : (float)best;
}

kernel void gb_segment_fold(device const float* partials [[buffer(0)]],
                            device float* out [[buffer(1)]],
                            constant uint& groups [[buffer(2)]],
                            constant uint& slices [[buffer(3)]],
                            constant uint& op [[buffer(4)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= groups) return;
    float accumulated = fold_start(op);
    for (uint slice = 0u; slice < slices; slice++) {
        accumulated = fold_join(op, accumulated, partials[slice * groups + gid]);
    }
    out[gid] = accumulated;
}

kernel void gb_pick_fold(device const float* values [[buffer(0)]],
                         device const float* wide [[buffer(1)]],
                         device const float* picks [[buffer(2)]],
                         device float* out [[buffer(3)]],
                         constant uint& groups [[buffer(4)]],
                         constant uint& slices [[buffer(5)]],
                         constant uint& op [[buffer(6)]],
                         constant uint& has_wide [[buffer(7)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= groups) return;
    uint best = 0xFFFFFFFFu;
    for (uint slice = 0u; slice < slices; slice++) {
        float candidate = picks[slice * groups + gid];
        if (candidate < 0.0f) continue;
        uint row = (uint)candidate;
        if (best == 0xFFFFFFFFu || pick_beats(op, values, wide, has_wide, row, best)) best = row;
    }
    out[gid] = (best == 0xFFFFFFFFu) ? -1.0f : (float)best;
}

/* One key column folded into a mixed-radix code, for `observed: false`.
 *
 * Called once per key over the same lane, so a row's code ends up the position
 * of its level tuple in the product of the level tables — which is a slot per
 * combination, whether or not any row landed there. That is exactly what an
 * unobserved category is. */
kernel void gb_product_codes(device const float* codes [[buffer(0)]],
                             device const float* valid [[buffer(1)]],
                             device float* out [[buffer(2)]],
                             constant uint& count [[buffer(3)]],
                             constant uint& stride [[buffer(4)]],
                             constant uint& levels [[buffer(5)]],
                             constant uint& has_valid [[buffer(6)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (out[gid] < 0.0f) return;
    if (has_valid != 0u && valid[gid] == 0.0f) {
        out[gid] = -1.0f;
        return;
    }
    int code = (int)codes[gid];
    if (code < 0 || code >= (int)levels) {
        out[gid] = -1.0f;
        return;
    }
    out[gid] += (float)((uint)code * stride);
}

/* The level this key takes at each slot of the product: one division to strip
 * the levels below it and one more to strip those above, with the remainder
 * from a subtraction. A modulo would be a third divide, and the device has
 * none to give. */
kernel void gb_product_levels(device float* out [[buffer(0)]],
                              constant uint& groups [[buffer(1)]],
                              constant uint& stride [[buffer(2)]],
                              constant uint& levels [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= groups) return;
    uint above = gid / stride;
    uint block = above / levels;
    out[gid] = (float)(above - block * levels);
}

kernel void gb_pack_int(device const float* lane [[buffer(0)]],
                        device float* values [[buffer(1)]],
                        device float* wide [[buffer(2)]],
                        constant uint& count [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    wide_store(values, wide, gid, (long)lane[gid]);
}

kernel void gb_compact(device const float* flags [[buffer(0)]],
                       device const float* offsets [[buffer(1)]],
                       device float* rows [[buffer(2)]],
                       constant uint& count [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (flags[gid] == 0.0f) return;
    rows[(uint)offsets[gid]] = (float)gid;
}

kernel void gb_group_flags(device const float* codes [[buffer(0)]],
                           device const float* per_group [[buffer(1)]],
                           device float* flags [[buffer(2)]],
                           constant uint& count [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    int code = (int)codes[gid];
    flags[gid] = (code >= 0 && per_group[code] != 0.0f) ? 1.0f : 0.0f;
}

/* `head`, `tail` and `nth` are the same question asked of the row's rank
 * within its group, which the stable layout already wrote. */
kernel void gb_rank_mask(device const float* codes [[buffer(0)]],
                         device const float* rank [[buffer(1)]],
                         device const float* sizes [[buffer(2)]],
                         device float* flags [[buffer(3)]],
                         constant uint& count [[buffer(4)]],
                         constant uint& mode [[buffer(5)]],
                         constant int& wanted [[buffer(6)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    int code = (int)codes[gid];
    if (code < 0) {
        flags[gid] = 0.0f;
        return;
    }
    int place = (int)rank[gid];
    int size = (int)sizes[code];
    bool keep = false;
    if (mode == PICK_HEAD) keep = place < wanted;
    else if (mode == PICK_TAIL) keep = place >= size - wanted;
    else keep = place == ((wanted < 0) ? (size + wanted) : wanted);
    flags[gid] = keep ? 1.0f : 0.0f;
}

kernel void gb_select_init(device const float* targets [[buffer(0)]],
                           device uint* prefix [[buffer(1)]],
                           device uint* want [[buffer(2)]],
                           constant uint& groups [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= groups) return;
    prefix[gid] = 0u;
    float target = targets[gid];
    want[gid] = (target < 0.0f) ? 0u : (uint)target;
}

/* One radix round of the per-group order statistic: how many of the values
 * still in the running fall in each of sixteen buckets. Eight rounds of four
 * bits cover the whole key, which is sixteen dispatches for a median — far
 * less than sorting every segment, and it needs no layout at all. */
kernel void gb_select_count(device const float* values [[buffer(0)]],
                            device const float* wide [[buffer(1)]],
                            device const float* valid [[buffer(2)]],
                            device const float* codes [[buffer(3)]],
                            device const uint* prefix [[buffer(4)]],
                            device uint* counts [[buffer(5)]],
                            constant uint& count [[buffer(6)]],
                            constant uint& shift [[buffer(7)]],
                            constant uint& has_wide [[buffer(8)]],
                            constant uint& has_valid [[buffer(9)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    int code = (int)codes[gid];
    if (code < 0) return;
    float value = row_value(values, wide, has_wide, gid);
    if (!row_present(valid, has_valid, gid, value)) return;
    uint key = order_key(value);
    if (shift < 28u && (key >> (shift + 4u)) != prefix[code]) return;
    uint bucket = (key >> shift) & (SELECT_BUCKETS - 1u);
    atomic_fetch_add_explicit((device atomic_uint*)(counts + (uint)code * SELECT_BUCKETS + bucket),
                              1u, memory_order_relaxed);
}

kernel void gb_select_step(device const uint* counts [[buffer(0)]],
                           device uint* prefix [[buffer(1)]],
                           device uint* want [[buffer(2)]],
                           constant uint& groups [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= groups) return;
    uint remaining = want[gid];
    uint chosen = SELECT_BUCKETS - 1u;
    for (uint bucket = 0u; bucket < SELECT_BUCKETS; bucket++) {
        uint here = counts[gid * SELECT_BUCKETS + bucket];
        if (remaining < here) {
            chosen = bucket;
            break;
        }
        remaining -= here;
    }
    prefix[gid] = (prefix[gid] << 4) | chosen;
    want[gid] = remaining;
}

kernel void gb_select_value(device const uint* prefix [[buffer(0)]],
                            device float* out [[buffer(1)]],
                            constant uint& groups [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= groups) return;
    out[gid] = order_value(prefix[gid]);
}

/* Which two order statistics a quantile sits between, and how far along. This
 * is pandas' `interpolation="linear"`: the position is `q * (n - 1)` and the
 * answer is the pair around it, weighted. A group with nothing present is
 * marked by a negative weight and comes out null. */
kernel void gb_quantile_targets(device const float* counts [[buffer(0)]],
                                device float* lower [[buffer(1)]],
                                device float* upper [[buffer(2)]],
                                device float* weight [[buffer(3)]],
                                constant uint& groups [[buffer(4)]],
                                constant uint& fraction [[buffer(5)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= groups) return;
    float present = counts[gid];
    if (present <= 0.0f) {
        lower[gid] = -1.0f;
        upper[gid] = -1.0f;
        weight[gid] = -1.0f;
        return;
    }
    float position = as_type<float>(fraction) * (present - 1.0f);
    float below = floor(position);
    lower[gid] = below;
    upper[gid] = min(below + 1.0f, present - 1.0f);
    weight[gid] = position - below;
}

kernel void gb_finish(device const float* a [[buffer(0)]],
                      device const float* b [[buffer(1)]],
                      device const float* c [[buffer(2)]],
                      device float* out [[buffer(3)]],
                      device float* valid [[buffer(4)]],
                      constant uint& count [[buffer(5)]],
                      constant uint& mode [[buffer(6)]],
                      constant int& shift [[buffer(7)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (mode == FINISH_ANY) {
        out[gid] = (a[gid] > 0.0f) ? 1.0f : 0.0f;
        valid[gid] = 1.0f;
        return;
    }
    if (mode == FINISH_ALL) {
        out[gid] = (a[gid] == b[gid]) ? 1.0f : 0.0f;
        valid[gid] = 1.0f;
        return;
    }
    if (mode == FINISH_LERP) {
        float fraction = c[gid];
        if (fraction < 0.0f) {
            out[gid] = 0.0f;
            valid[gid] = 0.0f;
            return;
        }
        out[gid] = a[gid] + fraction * (b[gid] - a[gid]);
        valid[gid] = 1.0f;
        return;
    }
    float divisor = b[gid] - (float)((mode == FINISH_DIV) ? 0 : shift);
    if (divisor <= 0.0f) {
        out[gid] = 0.0f;
        valid[gid] = 0.0f;
        return;
    }
    float result = a[gid] / divisor;
    out[gid] = (mode == FINISH_STD) ? sqrt(max(result, 0.0f)) : result;
    valid[gid] = 1.0f;
}

"""

var _kernels: dict[str, Kernel] = {}

fn kernel_for(entry: str) -> Kernel {
    let existing = _kernels.get(entry, null)
    if existing is not null { return existing }
    let compiled = Kernel.compile(SOURCE, entry)
    _kernels[entry] = compiled
    return compiled
}

fn flag(value: bool) -> int { return if value { 1 } else { 0 } }

fn u32(value: int) -> int { return value & 0xFFFFFFFF }

fn require_device(what: str) -> void {
    if not gpu.is_available() {
        throw RuntimeError(f"jaiframe needs a GPU to {what}")
    }
}

#: The frame a source is grouped as, and the name a `Series` source carried.
fn as_frame(source: any) -> (DataFrame, str?) {
    if source is DataFrame { return (source, null) }
    if source is Series {
        let name = source.name ?? "values"
        return (DataFrame([name], [source.values], index: source.index), name)
    }
    throw TypeError("group_by takes a DataFrame or a Series")
}

#: The cells of one index level or one key level, as host values.
#:
#: A download per lane rather than a `get` per row: this is the boundary where
#: a label stops being a device allocation, and the callers — `groups`,
#: `get_group`, the `by` callback — are the calls that already cost one.
fn level_values(column: Column) -> list[Value] {
    let missing = column.null_mask()
    var out: list[Value] = []
    if column.dtype.is_dictionary() {
        for text in column.to_strs() {
            out.push(if text is null {
                Value.null_of(column.dtype)
            } else {
                Value.of_str(text, column.dtype)
            })
        }
        return out
    }
    if column.dtype.is_wide() {
        for (at, number) in column.to_ints().enumerate() {
            out.push(if missing[at] {
                Value.null_of(column.dtype)
            } else {
                Value(column.dtype, integer: number)
            })
        }
        return out
    }
    for (at, number) in column.to_floats().enumerate() {
        out.push(if missing[at] {
            Value.null_of(column.dtype)
        } elif column.dtype == Dtype.Bool {
            Value.of_bool(number != 0.0)
        } else {
            Value.of_float(number)
        })
    }
    return out
}

#: One label per row, the levels joined the way `groups` and `get_group` spell
#: a multi-key group.
fn render_labels(levels: list[Column], length: int) -> list[str] {
    var out: list[str] = []
    for _row in 0..length { out.push("") }
    for (at, level) in levels.enumerate() {
        for (row, cell) in level_values(level).enumerate() {
            out[row] = if at == 0 { cell.to_str() } else { out[row] + "|" + cell.to_str() }
        }
    }
    return out
}

#: A `Column` that adopts `lane` rather than copying it.
#:
#: Every kernel here writes its result into a bare lane, and wrapping it costs
#: nothing: `Column.allocate` hands out a fresh buffer that is thrown away
#: immediately, which is one allocation, not one copy of the data.
fn adopt_lane(lane: Buffer, length: int, dtype: Dtype, valid: Buffer?) -> Column {
    let column = Column.allocate(dtype, length)
    column.values.free()
    column.values = lane
    column.valid = valid
    return column
}

#: The same lane read as `Int64`, which is exact because every integer a
#: reduction here produces is a row count and `MAX_ROWS` is 2^24.
fn adopt_int_lane(lane: Buffer, length: int, valid: Buffer?) -> Column {
    let column = Column.allocate(Dtype.Int64, length)
    column.valid = valid
    if length > 0 {
        kernel_for("gb_pack_int").dispatch_async(
            [lane, column.values, column.wide ?? column.values],
            length,
            [length]
        )
    }
    lane.free()
    return column
}

#: How a grouping was asked for, so `group_by` can name its levels.
enum KeySource {
    #: One or more of the frame's own columns.
    Columns,
    #: One or more levels of the frame's index.
    Levels,
    #: A callback over the index labels.
    Callback,
    #: A column handed in directly, belonging to nobody.
    Loose,
}

#: The handle `group_by` returns: a codes lane, the distinct keys, and the
#: frame the rows came from.
#:
#: Every aggregation is the same three steps — reduce the codes lane into one
#: value per group, wrap the result lane in a `Column`, hang the distinct keys
#: off it as an index. What differs between `sum` and `median` is only which
#: kernel does the middle step.
#:
#: The handle caches what it derives: the group sizes, their starts, and the
#: stable layout of the rows by code. Aggregating twice therefore costs one
#: layout, not two, and that is why the crossover in `use_scatter` asks
#: whether the layout already exists before it asks anything else.
pub class GroupBy {
    #: The rows being grouped, as a frame even when the source was a `Series`.
    pub var frame: DataFrame
    #: One code per row: which group it belongs to, `NO_GROUP` when dropped.
    pub var codes: Buffer
    #: The distinct key values, one column per key level, `ngroups` rows.
    pub var uniques: list[Column]
    #: A name per key level.
    pub var key_names: list[str?]
    #: Whether the keys become the result's index or its leading columns.
    pub var as_index: bool
    var group_total: int
    var key_positions: list[int]
    var source_name: str?
    var sizes_lane: Buffer?
    var starts_lane: Buffer?
    var order_lane: Buffer?
    var rank_lane: Buffer?
    var label_cache: list[str]?
    var freed: bool

    #: Build a handle over an existing factorisation.
    #:
    #: `group_by` is the constructor to reach for. This one is here because a
    #: module that already has a codes lane — a resample, a cut — should not
    #: have to hash its keys a second time to get an aggregator over them.
    pub fn init(
        self,
        frame: DataFrame,
        codes: Buffer,
        ngroups: int,
        uniques: list[Column],
        key_names: list[str?]? = null,
        key_positions: list[int]? = null,
        as_index: bool = true,
        source_name: str? = null
    ) {
        if uniques.len() == 0 {
            throw ValueError("a grouping needs at least one key level")
        }
        for level in uniques {
            if level.len() != ngroups {
                throw ValueError(f"a key level of {level.len()} cannot label {ngroups} groups")
            }
        }
        self.frame = frame
        self.codes = codes
        self.group_total = ngroups
        self.uniques = uniques
        self.key_names = key_names ?? uniques.map(|_level| null)
        self.key_positions = key_positions ?? []
        self.as_index = as_index
        self.source_name = source_name
        self.sizes_lane = null
        self.starts_lane = null
        self.order_lane = null
        self.rank_lane = null
        self.label_cache = null
        self.freed = false
    }

    #: How many groups the keys fell into.
    pub fn ngroups(self) -> int { return self.group_total }

    #: How many rows were grouped.
    pub fn nrows(self) -> int { return self.frame.len() }

    #: The distinct key values, one column per level.
    pub fn key_frame(self) -> DataFrame {
        var names: list[str] = []
        for (at, name) in self.key_names.enumerate() { names.push(name ?? f"key{at}") }
        return DataFrame(names, self.uniques.map(|level| level.copy()))
    }

    #: The row positions of every group, keyed by the group's label.
    #:
    #: A host materialisation: one boxed integer per row, which is what a
    #: `dict` of row lists costs however it is built. Reach for `get_group`
    #: when one group is wanted, and for `agg` when the rows themselves are
    #: not.
    pub fn groups(self) -> dict[str, list[int]] {
        self.check_live()
        let placed = self.layout().download()
        let bounds = self.starts().download()
        let names = self.labels()
        var out: dict[str, list[int]] = {}
        for group in 0..self.group_total {
            var rows: list[int] = []
            for at in int(bounds[group])..int(bounds[group + 1]) { rows.push(int(placed[at])) }
            out[names[group]] = rows
        }
        return out
    }

    #: How many rows each group holds, nulls included.
    pub fn size(self) -> Series {
        self.check_live()
        let lane = Buffer(self.group_total)
        copy_lane(self.sizes(), lane, self.group_total)
        return Series(
            adopt_int_lane(lane, self.group_total, null),
            index: self.key_index(),
            name: "size"
        )
    }

    #: How many rows of each column each group holds a value for.
    pub fn count(self) -> DataFrame { return self.aggregate(Agg.Count) }

    #: Which group each row landed in, `NO_GROUP` for a dropped one.
    pub fn ngroup(self) -> Series {
        self.check_live()
        let lane = Buffer(self.frame.len())
        copy_lane(self.codes, lane, self.frame.len())
        return Series(
            adopt_int_lane(lane, self.frame.len(), null),
            index: self.frame.index.copy(),
            name: "ngroup"
        )
    }

    #: How many rows of the same group came before each row.
    pub fn cumcount(self) -> Series {
        self.check_live()
        let lane = Buffer(self.frame.len())
        copy_lane(self.ranks(), lane, self.frame.len())
        return Series(
            adopt_int_lane(lane, self.frame.len(), null),
            index: self.frame.index.copy(),
            name: "cumcount"
        )
    }

    #: The first `n` rows of every group, in row order.
    pub fn head(self, n: int = 5) -> DataFrame { return self.rank_slice(PICK_HEAD, n) }

    #: The last `n` rows of every group, in row order.
    pub fn tail(self, n: int = 5) -> DataFrame { return self.rank_slice(PICK_TAIL, n) }

    #: Row `n` of every group, counting from the end when `n` is negative.
    pub fn nth(self, n: int) -> DataFrame { return self.rank_slice(PICK_NTH, n) }

    #: The rows of one group, named the way `groups` names it.
    pub fn get_group(self, key: any) -> DataFrame {
        self.check_live()
        let wanted = self.code_of(key)
        let chosen = Buffer(self.group_total)
        if self.group_total > 0 {
            chosen.fill_zero()
            chosen.upload([1.0], wanted)
        }
        let taken = self.take_flagged(chosen)
        chosen.free()
        return taken
    }

    #: Aggregate, in every shape pandas spells it.
    #:
    #: `spec` may be one aggregation, a list of them, or a dict of column name
    #: to one or a list. `named` is pandas' named aggregation: the language has
    #: no `**kwargs`, so `agg(total=("sales", "sum"))` is written
    #: `agg(named: {"total": ("sales", Agg.Sum)})`, and the dict's insertion
    #: order is the column order of the result.
    #:
    #: A single aggregation keeps the source column's name. A list of them
    #: makes two-level columns, which a frame here cannot hold, so they come
    #: out flattened as `sales_sum` — the same string pandas' own
    #: `columns.map("_".join)` produces.
    #:
    #: ```
    #: sales.agg(Agg.Mean)
    #: sales.agg([Agg.Min, Agg.Max])
    #: sales.agg({"units": Agg.Sum, "price": [Agg.Min, Agg.Max]})
    #: sales.agg(named: {"total": ("units", Agg.Sum)})
    #: ```
    pub fn agg(self, spec: any = null, named: dict[str, (str, Agg)]? = null) -> DataFrame {
        self.check_live()
        let requested = named
        if requested is not null { return self.agg_named(requested) }
        if spec is null { throw ValueError("agg needs an aggregation, a list, a dict or `named`") }
        if spec is dict { return self.agg_mapping(spec) }
        if spec is list { return self.agg_list(spec) }
        return self.aggregate(as_agg(spec))
    }

    #: Sum of the present values in each group.
    pub fn sum(self) -> DataFrame { return self.aggregate(Agg.Sum) }

    #: Arithmetic mean of the present values in each group.
    pub fn mean(self) -> DataFrame { return self.aggregate(Agg.Mean) }

    #: Smallest present value in each group.
    pub fn min(self) -> DataFrame { return self.aggregate(Agg.Min) }

    #: Largest present value in each group.
    pub fn max(self) -> DataFrame { return self.aggregate(Agg.Max) }

    #: Middle present value in each group.
    pub fn median(self) -> DataFrame { return self.aggregate(Agg.Median) }

    #: Sample standard deviation of each group.
    pub fn std(self, ddof: int = 1) -> DataFrame {
        return self.aggregate(Agg.Std, ddof: ddof)
    }

    #: Sample variance of each group.
    #:
    #: pandas spells this `var`, which is a keyword here.
    pub fn variance(self, ddof: int = 1) -> DataFrame {
        return self.aggregate(Agg.Var, ddof: ddof)
    }

    #: Product of the present values in each group.
    pub fn prod(self) -> DataFrame { return self.aggregate(Agg.Prod) }

    #: The `q` quantile of each group, interpolated linearly between the two
    #: order statistics it falls between, as pandas' default does.
    pub fn quantile(self, q: float = 0.5) -> DataFrame {
        if q < 0.0 or q > 1.0 { throw ValueError(f"a quantile is in [0, 1], got {q}") }
        return self.aggregate(Agg.Quantile, fraction: q)
    }

    #: How many distinct present values each group holds.
    pub fn nunique(self) -> DataFrame { return self.aggregate(Agg.Nunique) }

    #: Whether any present value in the group is truthy.
    pub fn any(self) -> DataFrame { return self.aggregate(Agg.Any) }

    #: Whether every present value in the group is truthy.
    pub fn all(self) -> DataFrame { return self.aggregate(Agg.All) }

    #: First present value of each group, in row order.
    pub fn first(self) -> DataFrame { return self.aggregate(Agg.First) }

    #: Last present value of each group, in row order.
    pub fn last(self) -> DataFrame { return self.aggregate(Agg.Last) }

    #: The index label where each group reaches its largest value.
    pub fn idxmax(self) -> DataFrame { return self.index_of_extreme(OP_MAX) }

    #: The index label where each group reaches its smallest value.
    pub fn idxmin(self) -> DataFrame { return self.index_of_extreme(OP_MIN) }

    #: Every row replaced by its group's result, keeping the frame's shape.
    #:
    #: One `take` on the codes lane does the broadcast: a row's code is
    #: already the position of its group in the result, so reading the group
    #: result back onto the rows is a gather, not a join. A dropped row has
    #: code `NO_GROUP` and comes out null.
    #:
    #: `how` is an aggregation, or a callback handed one group as a `Series`
    #: and returning the `Value` that group broadcasts — the escape hatch,
    #: which walks the groups on the host.
    pub fn transform(self, how: any) -> DataFrame {
        self.check_live()
        let positions = self.codes
        var names: list[str] = []
        var columns: list[Column] = []
        if how is Agg or how is str {
            let agg = as_agg(how)
            for at in self.value_columns(agg) {
                let reduced = self.aggregate_column(self.frame.columns[at], agg, 0.5, 1)
                names.push(self.frame.names[at])
                columns.push(reduced.take(positions, self.frame.len(), allow_missing: true))
                reduced.free()
            }
        } else {
            for at in self.every_value_column() {
                let reduced = self.apply_to_column(at, how)
                names.push(self.frame.names[at])
                columns.push(reduced.take(positions, self.frame.len(), allow_missing: true))
                reduced.free()
            }
        }
        return DataFrame(names, columns, index: self.frame.index.copy())
    }

    #: Every row of the groups `pred` keeps.
    #:
    #: `pred` is a `Bool` column of `ngroups` rows — the per-group answer, from
    #: wherever it came — or a callback handed one group as a frame. The
    #: column form is one kernel and one compaction whatever the group count;
    #: the callback form builds each group on the host, which is what makes it
    #: the slow path and the general one.
    pub fn filter(self, pred: any) -> DataFrame {
        self.check_live()
        if pred is Column {
            if pred.dtype != Dtype.Bool {
                throw TypeError(f"a group filter must be bool, not {pred.dtype.name()}")
            }
            if pred.len() != self.group_total {
                throw ValueError(f"{pred.len()} flags cannot answer for {self.group_total} groups")
            }
            return self.take_flagged(pred.values)
        }
        let bounds = self.starts().download()
        var verdicts: list[bool] = []
        for group in 0..self.group_total {
            let piece = self.group_frame(group, bounds)
            verdicts.push(as_bool(pred(piece)))
            piece.free()
        }
        let flags = Column.from_bools(verdicts)
        let taken = self.take_flagged(flags.values)
        flags.free()
        return taken
    }

    #: Run `fn_` once per group on the host — the documented escape hatch.
    #:
    #: Every other method here is one dispatch over the codes lane. This one is
    #: a loop over the groups, and each turn of it materialises a frame, so it
    #: costs a `take` per group per column. It is here because a per-group
    #: computation that is not an aggregation has nowhere else to go; reach for
    #: `agg`, `transform` or `filter` first.
    #:
    #: A callback returning a `Value` gives a `Series` over the group keys; one
    #: returning a `DataFrame` gives those frames stacked.
    pub fn apply(self, fn_: any) -> any {
        self.check_live()
        let bounds = self.starts().download()
        var cells: list[Value] = []
        var pieces: list[DataFrame] = []
        for group in 0..self.group_total {
            let piece = self.group_frame(group, bounds)
            let produced = fn_(piece)
            if produced is DataFrame {
                pieces.push(produced)
                piece.free()
                continue
            }
            if produced is Value { cells.push(produced) } else { cells.push(NA) }
            piece.free()
        }
        if pieces.len() > 0 { return stack_frames(pieces) }
        return Series(column_of_values(cells), index: self.key_index(), name: self.source_name)
    }

    #: Release the codes lane, the distinct keys and everything derived.
    #:
    #: The frame is not released: the handle borrowed it and the caller still
    #: owns it, exactly as `DataFrame._replace` leaves its inputs alone.
    pub fn free(self) -> void {
        if self.freed { return }
        self.freed = true
        self.codes.free()
        for level in self.uniques { level.free() }
        free_lane(self.sizes_lane)
        free_lane(self.starts_lane)
        free_lane(self.order_lane)
        free_lane(self.rank_lane)
    }

    #: Whether `free` has already run.
    pub fn is_freed(self) -> bool { return self.freed }

    fn check_live(self) -> void {
        if self.freed { throw RuntimeError("this GroupBy has already been released") }
    }

    fn labels(self) -> list[str] {
        let cached = self.label_cache
        if cached is not null { return cached }
        let rendered = render_labels(self.uniques, self.group_total)
        self.label_cache = rendered
        return rendered
    }

    fn key_index(self) -> Index {
        return Index.of(self.uniques.map(|level| level.copy()), [name for name in self.key_names])
    }

    #: How many rows fell into each group, counted once and kept.
    fn sizes(self) -> Buffer {
        let cached = self.sizes_lane
        if cached is not null { return cached }
        let lane = Buffer(self.group_total)
        if self.group_total > 0 { lane.fill_zero() }
        kernel_for("gb_group_counts").dispatch_async(
            [self.codes, self.codes, lane],
            self.frame.len(),
            [self.frame.len(), 0]
        )
        self.sizes_lane = lane
        return lane
    }

    #: Where each group's run begins once the rows are laid out by code.
    fn starts(self) -> Buffer {
        let cached = self.starts_lane
        if cached is not null { return cached }
        let scanned = prefix_sum(self.sizes(), self.group_total)
        self.starts_lane = scanned[0]
        return scanned[0]
    }

    #: The rows laid out by code, stable in row order within each group.
    #:
    #: Two passes over the rows with a scan between them. A chunk of rows is
    #: one thread's, so pass one counts each chunk's rows per group into a
    #: group-major matrix and pass two walks the chunk again handing rows out
    #: from the scan of it. Laying the matrix out group-major is what makes one
    #: scan do both jobs: the entry for `(group, chunk)` already has every
    #: earlier group's rows ahead of it, so it *is* the destination slot, and
    #: the layout comes out stable because a chunk is contiguous and its thread
    #: walks it in order.
    fn layout(self) -> Buffer {
        let cached = self.order_lane
        if cached is not null { return cached }
        self.build_layout()
        return self.order_lane ?? self.codes
    }

    #: How many rows of the same group came before each row.
    fn ranks(self) -> Buffer {
        let cached = self.rank_lane
        if cached is not null { return cached }
        self.build_layout()
        return self.rank_lane ?? self.codes
    }

    fn build_layout(self) -> void {
        let rows = self.frame.len()
        let groups = max(1, self.group_total)
        let chunks = bounded(LAYOUT_SLOTS // groups, 1, LAYOUT_CHUNKS)
        let span = if rows == 0 { 1 } else { (rows + chunks - 1) // chunks }
        let histogram = Buffer(chunks * groups)
        histogram.fill_zero()
        kernel_for("gb_chunk_counts").dispatch_async(
            [self.codes, histogram],
            chunks,
            [rows, chunks, span]
        )
        let scanned = prefix_sum(histogram, chunks * groups)
        let placed = scanned[1]
        let order = Buffer(max(1, placed))
        let rank = Buffer(max(1, rows))
        kernel_for("gb_chunk_place").dispatch_async(
            [self.codes, scanned[0], self.starts(), order, rank],
            chunks,
            [rows, chunks, span]
        )
        histogram.free()
        scanned[0].free()
        self.order_lane = order
        self.rank_lane = rank
    }

    #: Whether an aggregation should scatter with atomics or walk segments.
    #:
    #: The scatter needs nothing built, so it wins until a group is tall enough
    #: for the contention on its one slot to cost more than laying the rows
    #: out. Once something has laid them out the question is settled the other
    #: way: the layout is already paid for, and walking a run beats fighting
    #: over an address.
    fn use_scatter(self, op: int) -> bool {
        if op > OP_DEV2 { return false }
        if self.order_lane is not null { return false }
        return self.frame.len() < self.group_total * SEGMENT_ROWS
    }

    #: One value per group, by whichever path suits the aggregation.
    fn reduce_lane(self, column: Column, op: int, centres: Buffer?) -> Buffer {
        let out = Buffer(max(1, self.group_total))
        if self.group_total > 0 {
            out.fill(if op == OP_PROD { 1.0 } else { 0.0 })
        }
        let rows = self.frame.len()
        if rows == 0 or self.group_total == 0 { return out }
        if not self.use_scatter(op) {
            self.segment_into(column, op, centres, out)
            return out
        }
        let quads = not column.dtype.is_wide() and rows % 4 == 0
        let threads = if quads { rows // 4 } else { rows }
        kernel_for("gb_scatter_add").dispatch_async(
            [
                column.values,
                column.wide ?? column.values,
                column.valid ?? column.values,
                self.codes,
                centres ?? column.values,
                out,
            ],
            threads,
            [rows, op, flag(column.dtype.is_wide()), flag(column.valid is not null), flag(quads)]
        )
        return out
    }

    fn segment_into(self, column: Column, op: int, centres: Buffer?, out: Buffer) -> void {
        let groups = self.group_total
        let unit = (groups + SEGMENT_GROUP - 1) // SEGMENT_GROUP * SEGMENT_GROUP
        let slices = bounded(THREAD_BUDGET // max(1, groups), 1, MAX_SLICES)
        let wide = column.wide ?? column.values
        let has_wide = flag(column.dtype.is_wide())
        let partials = Buffer(slices * groups)
        let picks = Buffer(slices * groups)
        kernel_for("gb_segment_reduce").dispatch_async(
            [
                column.values,
                wide,
                column.valid ?? column.values,
                self.layout(),
                self.starts(),
                centres ?? column.values,
                partials,
                picks,
            ],
            slices * unit,
            [groups, unit, slices, op, has_wide, flag(column.valid is not null)],
            group_size: SEGMENT_GROUP
        )
        if op >= OP_MIN {
            kernel_for("gb_pick_fold").dispatch_async(
                [column.values, wide, picks, out],
                groups,
                [groups, slices, op, has_wide]
            )
        } else {
            kernel_for("gb_segment_fold").dispatch_async(
                [partials, out],
                groups,
                [groups, slices, op]
            )
        }
        partials.free()
        picks.free()
    }

    #: The row each group's extreme, first or last present value sits on.
    fn pick_lane(self, column: Column, op: int) -> Buffer {
        let out = Buffer(max(1, self.group_total))
        if self.group_total > 0 { out.fill(float(NO_GROUP)) }
        if self.frame.len() == 0 or self.group_total == 0 { return out }
        self.segment_into(column, op, null, out)
        return out
    }

    fn aggregate(self, agg: Agg, fraction: float = 0.5, ddof: int = 1) -> DataFrame {
        self.check_live()
        if agg == Agg.Size { return self.finish_frame(["size"], [self.sizes_column()]) }
        var names: list[str] = []
        var columns: list[Column] = []
        for at in self.value_columns(agg) {
            names.push(self.frame.names[at])
            columns.push(self.aggregate_column(self.frame.columns[at], agg, fraction, ddof))
        }
        return self.finish_frame(names, columns)
    }

    fn aggregate_column(self, column: Column, agg: Agg, fraction: float, ddof: int) -> Column {
        let groups = self.group_total
        return match agg {
            Agg.Size => self.sizes_column(),
            Agg.Count => adopt_int_lane(self.reduce_lane(column, OP_COUNT, null), groups, null),
            Agg.Sum => adopt_lane(self.reduce_lane(column, OP_SUM, null), groups, Dtype.Float64, null),
            Agg.Prod => adopt_lane(self.reduce_lane(column, OP_PROD, null), groups, Dtype.Float64, null),
            Agg.Mean => self.mean_column(column),
            Agg.Std => self.spread_column(column, ddof, true),
            Agg.Var => self.spread_column(column, ddof, false),
            Agg.Any => self.truth_column(column, false),
            Agg.All => self.truth_column(column, true),
            Agg.Median => self.quantile_column(column, 0.5),
            Agg.Quantile => self.quantile_column(column, fraction),
            Agg.Nunique => self.nunique_column(column),
            _ => self.picked_column(column, agg),
        }
    }

    fn sizes_column(self) -> Column {
        let lane = Buffer(max(1, self.group_total))
        copy_lane(self.sizes(), lane, self.group_total)
        return adopt_int_lane(lane, self.group_total, null)
    }

    fn picked_column(self, column: Column, agg: Agg) -> Column {
        let op = match agg {
            Agg.Min => OP_MIN,
            Agg.Max => OP_MAX,
            Agg.First => OP_FIRST,
            _ => OP_LAST,
        }
        let rows = self.pick_lane(column, op)
        let picked = column.take(rows, self.group_total, allow_missing: true)
        rows.free()
        return picked
    }

    #: A column with no missing lane has one present value in every row, so its
    #: count per group is the group's own size -- which the grouping already
    #: knows and caches, and which reads the codes rather than the values.
    #: Counting it again was a second full pass over the column: `mean` cost
    #: twice what `sum` did over the same two million rows, and a frame of
    #: several value columns paid for it once per column.
    fn mean_column(self, column: Column) -> Column {
        let groups = self.group_total
        let total = self.reduce_lane(column, OP_SUM, null)
        let counted = if column.has_validity() {
            self.reduce_lane(column, OP_COUNT, null)
        } else {
            null
        }
        let present = counted ?? self.sizes()
        let out = Buffer(max(1, groups))
        let valid = Buffer(max(1, groups))
        kernel_for("gb_finish").dispatch_async(
            [total, present, total, out, valid],
            groups,
            [groups, FINISH_DIV, 0]
        )
        total.free()
        #: `sizes()` belongs to the handle and outlives this call.
        if counted is not null { counted.free() }
        return adopt_lane(out, groups, Dtype.Float64, valid)
    }

    #: Variance from a second pass over the rows, not from `E[x^2] - E[x]^2`.
    #:
    #: The textbook identity subtracts two nearly equal float32 sums and keeps
    #: whatever is left, which for a column whose values sit far from zero is
    #: mostly rounding. Centring first costs one more read of a lane the
    #: previous dispatch just warmed.
    fn spread_column(self, column: Column, ddof: int, want_std: bool) -> Column {
        let groups = self.group_total
        let total = self.reduce_lane(column, OP_SUM, null)
        let present = self.reduce_lane(column, OP_COUNT, null)
        let centres = Buffer(max(1, groups))
        let centre_valid = Buffer(max(1, groups))
        kernel_for("gb_finish").dispatch_async(
            [total, present, total, centres, centre_valid],
            groups,
            [groups, FINISH_DIV, 0]
        )
        let deviation = self.reduce_lane(column, OP_DEV2, centres)
        let out = Buffer(max(1, groups))
        let valid = Buffer(max(1, groups))
        kernel_for("gb_finish").dispatch_async([deviation, present, deviation, out, valid], groups, [groups, if want_std {
            FINISH_STD
        } else {
            FINISH_VAR
        }, u32(ddof)])
        total.free()
        present.free()
        centres.free()
        centre_valid.free()
        deviation.free()
        return adopt_lane(out, groups, Dtype.Float64, valid)
    }

    fn truth_column(self, column: Column, want_all: bool) -> Column {
        let groups = self.group_total
        let truthy = self.reduce_lane(column, OP_NONZERO, null)
        let present = self.reduce_lane(column, OP_COUNT, null)
        let out = Buffer(max(1, groups))
        let valid = Buffer(max(1, groups))
        kernel_for("gb_finish").dispatch_async([truthy, present, truthy, out, valid], groups, [groups, if want_all {
            FINISH_ALL
        } else {
            FINISH_ANY
        }, 0])
        truthy.free()
        present.free()
        valid.free()
        return adopt_lane(out, groups, Dtype.Bool, null)
    }

    #: A quantile per group without sorting a single segment.
    #:
    #: Selecting the k-th smallest by radix is eight rounds of "how many of the
    #: values still in the running have each of sixteen leading digits", and
    #: every round is one pass over the rows keyed by code — the same segmented
    #: shape as every other reduction here. Sorting the segments instead would
    #: cost a sort per group, and a sort of one enormous group is the case that
    #: has no good answer.
    fn quantile_column(self, column: Column, fraction: float) -> Column {
        let groups = self.group_total
        let present = self.reduce_lane(column, OP_COUNT, null)
        let lower = Buffer(max(1, groups))
        let upper = Buffer(max(1, groups))
        let weight = Buffer(max(1, groups))
        kernel_for("gb_quantile_targets").dispatch_async(
            [present, lower, upper, weight],
            groups,
            [groups, f32_bits(fraction)]
        )
        let low = self.select_lane(column, lower)
        let high = self.select_lane(column, upper)
        let out = Buffer(max(1, groups))
        let valid = Buffer(max(1, groups))
        kernel_for("gb_finish").dispatch_async(
            [low, high, weight, out, valid],
            groups,
            [groups, FINISH_LERP, 0]
        )
        present.free()
        lower.free()
        upper.free()
        weight.free()
        low.free()
        high.free()
        return adopt_lane(out, groups, Dtype.Float64, valid)
    }

    fn select_lane(self, column: Column, targets: Buffer) -> Buffer {
        let groups = self.group_total
        let rows = self.frame.len()
        let prefix = Buffer(max(1, groups))
        let want = Buffer(max(1, groups))
        kernel_for("gb_select_init").dispatch_async([targets, prefix, want], groups, [groups])
        let counts = Buffer(max(1, groups * SELECT_BUCKETS))
        var shift = 32 - SELECT_BITS
        while shift >= 0 {
            counts.fill_zero()
            kernel_for("gb_select_count").dispatch_async(
                [
                    column.values,
                    column.wide ?? column.values,
                    column.valid ?? column.values,
                    self.codes,
                    prefix,
                    counts,
                ],
                rows,
                [rows, shift, flag(column.dtype.is_wide()), flag(column.valid is not null)]
            )
            kernel_for("gb_select_step").dispatch_async([counts, prefix, want], groups, [groups])
            shift -= SELECT_BITS
        }
        let out = Buffer(max(1, groups))
        kernel_for("gb_select_value").dispatch_async([prefix, out], groups, [groups])
        prefix.free()
        want.free()
        counts.free()
        return out
    }

    #: Distinct present values per group, from a second factorisation.
    #:
    #: Factorising `(group code, value)` numbers the distinct pairs, and the
    #: count of pairs belonging to a group is exactly its `nunique`. That reuses
    #: the hash table `keys.jai` already has rather than growing a second one
    #: here, and it costs one pass over the rows plus one over the pairs.
    fn nunique_column(self, column: Column) -> Column {
        let groups = self.group_total
        let counts = Buffer(max(1, groups))
        if groups > 0 { counts.fill_zero() }
        if self.frame.len() == 0 or groups == 0 {
            return adopt_int_lane(counts, groups, null)
        }
        let keyed = self.code_column()
        let paired = factorize_keys([keyed, column], false, false)
        let owner = paired[1][0]
        let payload = paired[1][1]
        let lane = owner.as_float_lane()
        kernel_for("gb_group_counts").dispatch_async(
            [lane, payload.valid ?? lane, counts],
            paired[2],
            [paired[2], flag(payload.valid is not null)]
        )
        lane.free()
        paired[0].free()
        for level in paired[1] { level.free() }
        keyed.free()
        return adopt_int_lane(counts, groups, null)
    }

    fn code_column(self) -> Column {
        let lane = Buffer(max(1, self.frame.len()))
        copy_lane(self.codes, lane, self.frame.len())
        return adopt_int_lane(lane, self.frame.len(), null)
    }

    fn index_of_extreme(self, op: int) -> DataFrame {
        self.check_live()
        if self.frame.index.nlevels() != 1 {
            throw ValueError(
                "idxmax over a multi-index would return a tuple, which a column cannot hold"
            )
        }
        let agg = if op == OP_MAX { Agg.Max } else { Agg.Min }
        var names: list[str] = []
        var columns: list[Column] = []
        for at in self.value_columns(agg) {
            let rows = self.pick_lane(self.frame.columns[at], op)
            let labelled = self.frame.index.take(rows, self.group_total, allow_missing: true)
            names.push(self.frame.names[at])
            columns.push(labelled.levels[0])
            rows.free()
        }
        return self.finish_frame(names, columns)
    }

    fn agg_list(self, spec: list) -> DataFrame {
        var names: list[str] = []
        var columns: list[Column] = []
        for at in self.every_value_column() {
            for entry in spec {
                let agg = as_agg(entry)
                if not supports(self.frame.columns[at].dtype, agg) { continue }
                names.push(f"{self.frame.names[at]}_{agg.name()}")
                columns.push(self.aggregate_column(self.frame.columns[at], agg, 0.5, 1))
            }
        }
        return self.finish_frame(names, columns)
    }

    fn agg_mapping(self, spec: dict) -> DataFrame {
        var names: list[str] = []
        var columns: list[Column] = []
        for key in spec {
            let at = self.frame.column_index(key)
            let entry = spec[key]
            if entry is list {
                for one in entry {
                    let agg = as_agg(one)
                    self.require_support(key, at, agg)
                    names.push(f"{key}_{agg.name()}")
                    columns.push(self.aggregate_column(self.frame.columns[at], agg, 0.5, 1))
                }
                continue
            }
            let agg = as_agg(entry)
            self.require_support(key, at, agg)
            names.push(key)
            columns.push(self.aggregate_column(self.frame.columns[at], agg, 0.5, 1))
        }
        return self.finish_frame(names, columns)
    }

    fn agg_named(self, named: dict[str, (str, Agg)]) -> DataFrame {
        var names: list[str] = []
        var columns: list[Column] = []
        for label in named {
            let pair = named[label]
            let at = self.frame.column_index(pair[0])
            self.require_support(pair[0], at, pair[1])
            names.push(label)
            columns.push(self.aggregate_column(self.frame.columns[at], pair[1], 0.5, 1))
        }
        return self.finish_frame(names, columns)
    }

    fn require_support(self, name: str, at: int, agg: Agg) -> void {
        let dtype = self.frame.columns[at].dtype
        if supports(dtype, agg) { return }
        throw TypeError(f"{agg.name()} is not defined over the {dtype.name()} column {name}")
    }

    #: The columns an aggregation runs over: everything that is not a key and
    #: that the aggregation is defined for.
    fn value_columns(self, agg: Agg) -> list[int] {
        var out: list[int] = []
        for at in self.every_value_column() {
            if supports(self.frame.columns[at].dtype, agg) { out.push(at) }
        }
        return out
    }

    fn every_value_column(self) -> list[int] {
        var out: list[int] = []
        for at in 0..self.frame.width() {
            if self.key_positions.contains(at) { continue }
            out.push(at)
        }
        return out
    }

    fn finish_frame(self, names: list[str], columns: list[Column]) -> DataFrame {
        if self.as_index { return DataFrame(names, columns, index: self.key_index()) }
        var all_names: list[str] = []
        var all_columns: list[Column] = []
        for (at, name) in self.key_names.enumerate() {
            all_names.push(name ?? f"key{at}")
            all_columns.push(self.uniques[at].copy())
        }
        for (at, name) in names.enumerate() {
            all_names.push(name)
            all_columns.push(columns[at])
        }
        return DataFrame(all_names, all_columns, index: Index.range(self.group_total))
    }

    fn rank_slice(self, mode: int, wanted: int) -> DataFrame {
        self.check_live()
        let rows = self.frame.len()
        let flags = Buffer(max(1, rows))
        kernel_for("gb_rank_mask").dispatch_async(
            [self.codes, self.ranks(), self.sizes(), flags],
            rows,
            [rows, mode, u32(wanted)]
        )
        let taken = self.take_flags(flags, rows)
        flags.free()
        return taken
    }

    fn take_flagged(self, per_group: Buffer) -> DataFrame {
        let rows = self.frame.len()
        let flags = Buffer(max(1, rows))
        kernel_for("gb_group_flags").dispatch_async([self.codes, per_group, flags], rows, [rows])
        let taken = self.take_flags(flags, rows)
        flags.free()
        return taken
    }

    fn take_flags(self, flags: Buffer, count: int) -> DataFrame {
        let scanned = prefix_sum(flags, count)
        let kept = scanned[1]
        let rows = Buffer(max(1, kept))
        kernel_for("gb_compact").dispatch_async([flags, scanned[0], rows], count, [count])
        scanned[0].free()
        let taken = self.take_rows(rows, kept)
        rows.free()
        return taken
    }

    fn take_rows(self, rows: Buffer, count: int) -> DataFrame {
        return DataFrame(
            [name for name in self.frame.names],
            self.frame.columns.map(|column| column.take(rows, count)),
            index: self.frame.index.take(rows, count)
        )
    }

    #: One group's rows as a frame — the per-group materialisation `apply` and
    #: the callback form of `filter` are built on, and the reason both are slow.
    fn group_frame(self, group: int, bounds: list[float]) -> DataFrame {
        let begin = int(bounds[group])
        let stop = int(bounds[group + 1])
        let rows = Buffer(max(1, stop - begin))
        copy_lane(self.layout(), rows, stop - begin, begin, 0)
        let taken = self.take_rows(rows, stop - begin)
        rows.free()
        return taken
    }

    fn apply_to_column(self, at: int, callback: any) -> Column {
        let bounds = self.starts().download()
        var cells: list[Value] = []
        for group in 0..self.group_total {
            let piece = self.group_frame(group, bounds)
            let produced = callback(
                Series(
                    piece.columns[at],
                    index: piece.index,
                    name: self.frame.names[at]
                )
            )
            cells.push(if produced is Value { produced } else { NA })
            piece.free()
        }
        return column_of_values(cells)
    }

    fn code_of(self, key: any) -> int {
        let wanted = label_of(key)
        for (group, name) in self.labels().enumerate() {
            if name == wanted { return group }
        }
        throw KeyError(f"no group labelled {wanted}")
    }
}

#: Group a frame or a series by columns, by index levels, or by a callback.
#:
#: `by` takes a column name, a list of names, a `Column` of the same height, or
#: a callback handed each row's index label. `level` takes an index level by
#: position or by name, or a list of them. Exactly one of the two is given.
#:
#: `sort` and `dropna` mean what they do in pandas and go straight to
#: `factorize`: whether the groups come out in key order, and whether a missing
#: key is a group of its own rather than a dropped row.
#:
#: `observed` is the other pandas flag, and it defaults to `true` here as it
#: does in pandas 2.1 and later: only the key tuples rows actually carry become
#: groups. `observed: false` asks for a group per combination of the keys'
#: level tables, which is a question only a `Category` can answer, so it needs
#: every key to be one.
#:
#: `as_index` says where the keys go: the result's index when true, its leading
#: columns when false.
#:
#: ```
#: group_by(frame, "city")
#: group_by(frame, ["city", "quarter"], sort: false)
#: group_by(frame, level: 0)
#: ```
pub fn group_by(
    source: any,
    by: any = null,
    level: any = null,
    as_index: bool = true,
    sort: bool = true,
    dropna: bool = true,
    observed: bool = true
) -> GroupBy {
    require_device("group a frame")
    let unpacked = as_frame(source)
    let frame = unpacked[0]
    if by is null and level is null {
        throw ValueError("group_by needs `by` or `level`")
    }
    if by is not null and level is not null {
        throw ValueError("group_by takes `by` or `level`, not both")
    }
    var keys: list[Column] = []
    var names: list[str?] = []
    var positions: list[int] = []
    var built: list[Column] = []
    if level is not null {
        let wanted = index_levels(frame.index, level)
        for (at, one) in frame.index.materialize().enumerate() {
            if not wanted.contains(at) {
                one.free()
                continue
            }
            keys.push(one)
            built.push(one)
            names.push(frame.index.names[at])
        }
    } else {
        for one in key_specs(by) {
            if one is str {
                let at = frame.column_index(one)
                keys.push(frame.columns[at])
                names.push(one)
                positions.push(at)
                continue
            }
            if one is Column {
                if one.len() != frame.len() {
                    throw ValueError(f"a key of {one.len()} rows cannot group {frame.len()}")
                }
                keys.push(one)
                names.push(null)
                continue
            }
            let derived = labelled_keys(frame.index, one)
            keys.push(derived)
            built.push(derived)
            names.push(null)
        }
    }
    if keys.len() == 0 { throw ValueError("group_by needs at least one key") }
    let grouped = if observed {
        factorize_keys(keys, sort, dropna)
    } else {
        category_product(keys, frame.len(), dropna)
    }
    for one in built { one.free() }
    return GroupBy(
        frame,
        grouped[0],
        grouped[2],
        grouped[1],
        key_names: names,
        key_positions: positions,
        as_index: as_index,
        source_name: unpacked[1]
    )
}

#: The aggregation a name or an `Agg` stands for.
fn as_agg(value: any) -> Agg {
    if value is Agg { return value }
    if value is str { return Agg.from_name(value) }
    throw TypeError("an aggregation is an Agg or its pandas name")
}

#: Whether an aggregation is defined over a dtype.
#:
#: `min` and `max` refuse text. The codes of a dictionary column are assignment
#: order, not alphabetical order, so comparing them would answer a question
#: nobody asked; sorting the level table is the business of whoever owns it.
#: `first` and `last` are positional and work over anything.
fn supports(dtype: Dtype, agg: Agg) -> bool {
    return match agg {
        Agg.Count | Agg.Size | Agg.First | Agg.Last | Agg.Nunique => true,
        Agg.Min | Agg.Max => dtype.is_numeric() or dtype == Dtype.Datetime,
        _ => dtype.is_numeric(),
    }
}

fn bounded(value: int, low: int, high: int) -> int { return max(low, min(high, value)) }

#: `count` slots of `src` copied into `dst`, through the column store's own
#: lane copy rather than a second kernel that would say the same thing.
fn copy_lane(
    src: Buffer,
    dst: Buffer,
    count: int,
    src_offset: int = 0,
    dst_offset: int = 0
) -> void {
    if count <= 0 { return }
    column_kernel("col_copy_lane").dispatch_async(
        [src, dst],
        count,
        [count, src_offset, dst_offset]
    )
}

fn free_lane(lane: Buffer?) -> void {
    if lane is not null { lane.free() }
}

#: The truth value a `filter` callback answered with.
fn as_bool(value: any) -> bool {
    if value is bool { return value }
    if value is Value { return not value.is_na() and value.to_bool() }
    throw TypeError("a group filter must answer with a bool")
}

#: A column holding host cells, typed by the first one that is not missing.
fn column_of_values(cells: list[Value]) -> Column {
    var dtype = Dtype.Float64
    for cell in cells {
        if not cell.is_na() {
            dtype = cell.dtype
            break
        }
    }
    if dtype.is_dictionary() {
        var text: list[str?] = []
        for cell in cells { text.push(if cell.is_na() { null } else { cell.to_str() }) }
        return Column.from_strs(text, dtype)
    }
    var missing: list[bool] = []
    for cell in cells { missing.push(cell.is_na()) }
    if dtype.is_wide() {
        var numbers: list[int] = []
        for cell in cells { numbers.push(if cell.is_na() { 0 } else { cell.to_int() }) }
        return Column.from_ints(numbers, nulls: missing, dtype: dtype)
    }
    if dtype == Dtype.Bool {
        var truths: list[bool] = []
        for cell in cells { truths.push(not cell.is_na() and cell.to_bool()) }
        return Column.from_bools(truths, nulls: missing)
    }
    var numbers: list[float] = []
    for cell in cells { numbers.push(cell.to_float()) }
    return Column.from_floats(numbers, nulls: missing)
}

#: The frames an `apply` produced, stacked.
#:
#: The result gets a fresh range index. pandas builds a hierarchical one from
#: the group keys and the pieces' own labels; that is a multi-index
#: construction, which belongs to `reshape.jai` and not to a call that has
#: already spent a host loop.
fn stack_frames(pieces: list[DataFrame]) -> DataFrame {
    let names = [name for name in pieces[0].names]
    var columns: list[Column] = []
    for name in names {
        var parts: list[Column] = []
        for piece in pieces { parts.push(piece.column(name)) }
        columns.push(concat_columns(parts))
    }
    var rows = 0
    for piece in pieces {
        rows += piece.len()
        piece.free()
    }
    return DataFrame(names, columns, index: Index.range(rows))
}

#: How `groups` and `get_group` spell one group's key.
fn label_of(key: any) -> str {
    if key is str { return key }
    if key is Value { return key.to_str() }
    if key is list {
        var parts: list[str] = []
        for part in key { parts.push(label_of(part)) }
        return "|".join(parts)
    }
    throw TypeError("a group key is a label, a Value, or a list of them")
}

#: The index levels `level` names, by position or by name.
fn index_levels(index: Index, level: any) -> list[int] {
    if level is list {
        var out: list[int] = []
        for one in level { out.push(index_levels(index, one)[0]) }
        return out
    }
    if level is int {
        let at = if level < 0 { index.nlevels() + level } else { level }
        if at < 0 or at >= index.nlevels() {
            throw IndexError(f"level {level} is outside an index of {index.nlevels()}")
        }
        return [at]
    }
    if level is str {
        for (at, name) in index.names.enumerate() {
            if name == level { return [at] }
        }
        throw KeyError(f"no index level named {level}")
    }
    throw TypeError("a level is a position or a name")
}

#: The key specifications `by` holds, always as a list.
fn key_specs(by: any) -> list[any] {
    if by is list { return by }
    return [by]
}

#: A key column from a callback over the index labels.
#:
#: The callback is Jaithon, so it runs once per row on the host — the same
#: bargain `access.jai:query` makes. Hand `by` a `Column` instead when the key
#: can be computed on the device.
fn labelled_keys(index: Index, callback: any) -> Column {
    let levels = index.materialize()
    var cells: list[list[Value]] = []
    for level in levels { cells.push(level_values(level)) }
    for level in levels { level.free() }
    var produced: list[Value] = []
    for row in 0..index.len() {
        var label: list[Value] = []
        for column in cells { label.push(column[row]) }
        let answer = if label.len() == 1 { callback(label[0]) } else { callback(label) }
        produced.push(coerce_key(answer))
    }
    return column_of_values(produced)
}

fn coerce_key(answer: any) -> Value {
    if answer is Value { return answer }
    if answer is str { return Value.of_str(answer) }
    if answer is int { return Value.of_int(answer) }
    if answer is float { return Value.of_float(answer) }
    if answer is bool { return Value.of_bool(answer) }
    throw TypeError("a group-by callback answers with a label")
}

#: `keys.jai:factorize`, reshaped to the three things this module wants.
#:
#: Every dependency on the key machinery goes through here: the codes lane,
#: one column per key level holding the distinct values, and how many groups
#: there are. Nothing else in this file knows how a key is hashed.
fn factorize_keys(columns: list[Column], sort: bool, dropna: bool) -> (Buffer, list[Column], int) {
    return factorize(columns, sort: sort, dropna: dropna)
}

#: Every combination of the key columns' levels, observed or not.
#:
#: `factorize` numbers the key tuples that rows actually carry, which is
#: `observed: true`. pandas' `observed: false` asks for a group per level
#: combination whether or not a row landed in it, and that is only a question a
#: `Category` can answer — its level table *is* the domain, where a `Str`
#: column's is just what turned up. So a mixed set of keys is refused rather
#: than half-expanded.
fn category_product(keys: list[Column], rows: int, dropna: bool) -> (Buffer, list[Column], int) {
    var total = 1
    for key in keys {
        if key.dtype != Dtype.Category {
            throw TypeError(
                f"observed: false needs every key to be a category, and one is {key.dtype.name()}"
            )
        }
        total *= max(1, key.level_count())
    }
    if not dropna {
        throw ValueError("observed: false has no slot for a missing level; pass dropna: true")
    }
    if total > MAX_ROWS {
        throw ValueError(f"{total} level combinations exceeds the {MAX_ROWS} a code lane holds")
    }
    var strides: list[int] = []
    for _key in keys { strides.push(1) }
    var running = 1
    for step in 0..keys.len() {
        let at = keys.len() - 1 - step
        strides[at] = running
        running *= max(1, keys[at].level_count())
    }
    let codes = Buffer(max(1, rows))
    if rows > 0 { codes.fill_zero() }
    for (at, key) in keys.enumerate() {
        kernel_for("gb_product_codes").dispatch_async(
            [key.values, key.valid ?? key.values, codes],
            rows,
            [rows, strides[at], key.level_count(), flag(key.valid is not null)]
        )
    }
    var uniques: list[Column] = []
    for (at, key) in keys.enumerate() {
        let level = Column.allocate(Dtype.Category, total)
        level.set_levels([name for name in key.levels ?? []])
        kernel_for("gb_product_levels").dispatch_async(
            [level.values],
            total,
            [total, strides[at], max(1, key.level_count())]
        )
        uniques.push(level)
    }
    return (codes, uniques, total)
}
