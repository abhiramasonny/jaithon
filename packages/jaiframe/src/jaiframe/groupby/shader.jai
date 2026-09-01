#: The Metal source every kernel in `groupby` is compiled from.
#:
#: One translation unit, so the scatter kernels, the segmented ones and the
#: layout pass share the same inline helpers and the same constants, and
#: `kernel_for` compiles it once whichever entry point is asked for first.
pub let SOURCE = """#include <metal_stdlib>
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
