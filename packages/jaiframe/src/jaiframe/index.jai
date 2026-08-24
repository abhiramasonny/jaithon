#: What an `Index` does once it has to answer questions about its labels:
#: where a label sits, which labels two indexes share, and how to put two
#: frames on one set of rows.
#:
#: Every row lookup goes through an `IndexLookup`, which picks one of three
#: plans and never builds more machinery than the labels need:
#:
#: - A range index is arithmetic. `start + i * step` inverts to a division, so
#:   the default index of every frame answers a lookup without storing a label,
#:   let alone a table.
#: - A monotonic single-level index bisects. The labels are already in the
#:   order a search wants, and standing 2n slots of hash table beside an index
#:   to answer what a bisect answers would cost more memory than the index.
#: - Anything else — unsorted, or a multi-index — gets one open-addressed table
#:   over the per-row hash `Column.hash_into` folds across the levels. A
#:   32-bit hash agreeing is not two labels agreeing, so every candidate the
#:   walk stops on is confirmed with `Column.equal_rows`, and the rare
#:   disagreement resumes the walk from where it stopped. The table is exact,
#:   not probabilistic.
#:
#: The table is built once and reused for as many probes as the caller makes,
#: which is why `IndexLookup` is a value you can hold rather than something
#: hidden inside `get_loc`. Building it costs a handful of dispatches and one
#: short download per settling round — normally one round.
#:
#: Text levels compare by dictionary code, so a text index is monotonic in the
#: order its levels were interned, not in alphabetical order. That is what
#: makes a string lookup an integer compare; `sort_index` in `keys.jai` is
#: where alphabetical order comes from, and sorting deliberately does not live
#: here. Set operations keep first-appearance order for the same reason:
#: pandas sorts its unions, and a caller who wants that sorts afterwards.
#:
#: ```
#: let joint = union(left.index, right.index)
#: defer { joint.free() }
#: let both = align(left, right, "outer")
#: let stacked = set_index(frame, ["city", "year"])
#: ```
import std.gpu as gpu
from std.gpu import Buffer, Kernel
from .column import Column, DataFrame, Dtype, Index, Series, Value
from .column import MAX_ROWS, NA
from .column import cached_kernel, concat_columns, f32_bits, iota, wide_pack

#: The label position that stands for "no such label", which is what `take`
#: reads as a null and what a caller tests an indexer against.
pub const NOT_FOUND = -1

#: Carry the last label at or before each target label.
pub const METHOD_PAD = "ffill"
#: Carry the first label at or after each target label.
pub const METHOD_BACKFILL = "bfill"
#: Take whichever neighbouring label is closer.
pub const METHOD_NEAREST = "nearest"

#: Where a tie goes in `searchsorted`: before the run of equal labels.
pub const SIDE_LEFT = "left"
#: Where a tie goes in `searchsorted`: after the run of equal labels.
pub const SIDE_RIGHT = "right"

#: Arithmetic: the labels are `start + i * step` and a lookup is a division.
pub const PLAN_RANGE = 0
#: Bisection over a monotonic increasing single level.
pub const PLAN_SORTED = 1
#: An open-addressed table over the per-row key hash.
pub const PLAN_HASHED = 2

const TALLY_GROUP = 256
const TALLY_GROUPS = 256
const PAIR_GROUP = 256

#: How many times the insert or the probe walk may be sent round again before
#: the loop is called broken.
#:
#: A round is only ever spent on a pair of labels whose 32-bit hashes agreed
#: and whose values did not. A million labels produce on the order of a hundred
#: such pairs, and each of them clears in the round after it is found, so two
#: rounds is the realistic ceiling. Sixty-four is a tripwire, not a budget.
const SETTLE_ROUNDS = 64

let SOURCE = """#include <metal_stdlib>
using namespace metal;

constexpr constant uint TALLY_GROUP = 256;
constexpr constant uint TALLY_GROUPS = 256;
constexpr constant uint PAIR_GROUP = 256;
constexpr constant long WIDE_SCALE = 16777216L;
constexpr constant uint UNSETTLED = 0xFFFFFFFFu;

/* The two lanes are only ever recombined in `long`, as they are in column.jai:
 * doing it in `float` throws away everything below the top 24 bits, which is
 * the whole reason the second lane exists. */
inline long wide_value(float low, float high) {
    return (long)high * WIDE_SCALE + (long)low;
}

kernel void idx_fill_u32(device uint* out [[buffer(0)]],
                         constant uint& count [[buffer(1)]],
                         constant uint& value [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out[gid] = value;
}

kernel void idx_start_probe(device const uint* hashes [[buffer(0)]],
                            device uint* cursor [[buffer(1)]],
                            constant uint& count [[buffer(2)]],
                            constant uint& mask [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    cursor[gid] = hashes[gid] & mask;
}

/* One thread per row, walking the table until it claims an empty slot or finds
 * one already holding a row whose key hash matches.
 *
 * The walk is in the kernel rather than one dispatch per probe step. A cluster
 * in a linear-probed table is walked without the host hearing about it, and
 * only the single candidate the walk stops on has to come back for an exact
 * comparison — one `equal_rows` chain for the whole column instead of one per
 * step. `idx_settle` sends the row round again, from where it stopped, when
 * the comparison says the hashes agreed but the labels did not.
 *
 * The slot holds `row + 1` so that zero can mean empty and the lane can be
 * cleared with a fill. A slot's key hash is read back through the row it
 * names rather than stored beside it, which is what keeps the claim a single
 * atomic word: a second word would have to be published after the claim, and
 * a reader could see the claim before the word. */
kernel void idx_insert(device atomic_uint* slots [[buffer(0)]],
                       device const uint* hashes [[buffer(1)]],
                       device uint* cursor [[buffer(2)]],
                       device uint* settled [[buffer(3)]],
                       device float* candidate [[buffer(4)]],
                       constant uint& count [[buffer(5)]],
                       constant uint& mask [[buffer(6)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (settled[gid] != UNSETTLED) return;
    uint pos = cursor[gid];
    uint mine = hashes[gid];
    for (uint step = 0u; step <= mask; step++) {
        uint seen = 0u;
        bool won = atomic_compare_exchange_weak_explicit(
            &slots[pos], &seen, gid + 1u, memory_order_relaxed, memory_order_relaxed);
        /* A weak exchange is allowed to fail with nothing there. Only a real
         * occupant ends the attempt on this slot. */
        while (!won && seen == 0u) {
            won = atomic_compare_exchange_weak_explicit(
                &slots[pos], &seen, gid + 1u, memory_order_relaxed, memory_order_relaxed);
        }
        if (won) {
            cursor[gid] = pos;
            settled[gid] = pos;
            candidate[gid] = -1.0f;
            return;
        }
        if (hashes[seen - 1u] == mine) {
            cursor[gid] = pos;
            candidate[gid] = (float)(seen - 1u);
            return;
        }
        pos = (pos + 1u) & mask;
    }
    /* Unreachable: the table has at least two slots per row, so an empty one
     * is always within a full lap. */
    cursor[gid] = pos;
    settled[gid] = pos;
    candidate[gid] = -1.0f;
}

kernel void idx_settle(device const float* flags [[buffer(0)]],
                       device const float* candidate [[buffer(1)]],
                       device uint* cursor [[buffer(2)]],
                       device uint* settled [[buffer(3)]],
                       device float* pending [[buffer(4)]],
                       constant uint& count [[buffer(5)]],
                       constant uint& mask [[buffer(6)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (settled[gid] != UNSETTLED) {
        pending[gid] = 0.0f;
        return;
    }
    if (candidate[gid] >= 0.0f && flags[gid] != 0.0f) {
        settled[gid] = cursor[gid];
        pending[gid] = 0.0f;
        return;
    }
    cursor[gid] = (cursor[gid] + 1u) & mask;
    pending[gid] = 1.0f;
}

/* Whoever won the race for a slot is whoever the scheduler ran first, so the
 * table alone cannot say which row of a repeated label came first. This
 * elects the lowest, which is what makes `unique` deterministic. */
kernel void idx_elect(device atomic_uint* rep [[buffer(0)]],
                      device const uint* settled [[buffer(1)]],
                      constant uint& count [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    atomic_fetch_min_explicit(&rep[settled[gid]], gid, memory_order_relaxed);
}

kernel void idx_first_of_kind(device const uint* rep [[buffer(0)]],
                              device const uint* settled [[buffer(1)]],
                              device float* out [[buffer(2)]],
                              constant uint& count [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out[gid] = (rep[settled[gid]] == gid) ? 1.0f : 0.0f;
}

/* The read side of the same walk. `answer` carries the state: -3 while the
 * query is still walking, -1 once it has run into an empty slot and is a miss,
 * and the row otherwise. */
kernel void idx_probe(device const uint* slots [[buffer(0)]],
                      device const uint* hashes [[buffer(1)]],
                      device const uint* queries [[buffer(2)]],
                      device uint* cursor [[buffer(3)]],
                      device float* answer [[buffer(4)]],
                      device float* candidate [[buffer(5)]],
                      constant uint& count [[buffer(6)]],
                      constant uint& mask [[buffer(7)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (answer[gid] > -2.5f) return;
    uint pos = cursor[gid];
    uint mine = queries[gid];
    for (uint step = 0u; step <= mask; step++) {
        uint seen = slots[pos];
        if (seen == 0u) {
            answer[gid] = -1.0f;
            candidate[gid] = -1.0f;
            return;
        }
        if (hashes[seen - 1u] == mine) {
            cursor[gid] = pos;
            candidate[gid] = (float)(seen - 1u);
            return;
        }
        pos = (pos + 1u) & mask;
    }
    answer[gid] = -1.0f;
    candidate[gid] = -1.0f;
}

kernel void idx_accept(device const float* flags [[buffer(0)]],
                       device const float* candidate [[buffer(1)]],
                       device const uint* rep [[buffer(2)]],
                       device uint* cursor [[buffer(3)]],
                       device float* answer [[buffer(4)]],
                       device float* pending [[buffer(5)]],
                       constant uint& count [[buffer(6)]],
                       constant uint& mask [[buffer(7)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (answer[gid] > -2.5f) {
        pending[gid] = 0.0f;
        return;
    }
    if (candidate[gid] >= 0.0f && flags[gid] != 0.0f) {
        answer[gid] = (float)rep[cursor[gid]];
        pending[gid] = 0.0f;
        return;
    }
    cursor[gid] = (cursor[gid] + 1u) & mask;
    pending[gid] = 1.0f;
}

/* A fixed grid striding the lane, not a thread per element: the fold costs
 * eight barriers, and paying them per element is what makes a count slower
 * than the read it is counting. Four at a time where the lane divides by
 * four — the loop is doing nothing but loading, so the load width is the
 * whole of its cost. */
kernel void idx_tally(device const float* flags [[buffer(0)]],
                      device float* partials [[buffer(1)]],
                      constant uint& count [[buffer(2)]],
                      uint gid [[thread_position_in_grid]],
                      uint lid [[thread_index_in_threadgroup]],
                      uint wid [[threadgroup_position_in_grid]]) {
    threadgroup float scratch[TALLY_GROUP];
    const uint span = TALLY_GROUP * TALLY_GROUPS;
    float total = 0.0f;
    if ((count & 3u) == 0u) {
        device const float4* quads = (device const float4*)flags;
        uint groups_of_four = count >> 2;
        for (uint i = gid; i < groups_of_four; i += span) total += dot(quads[i], float4(1.0f));
    } else {
        for (uint i = gid; i < count; i += span) total += flags[i];
    }
    scratch[lid] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = TALLY_GROUP / 2u; stride > 0u; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) partials[wid] = scratch[0];
}

/* How each adjacent pair of labels is ordered: -1 falling, 0 tied, 1 rising,
 * and 3 for a pair a null makes unorderable.
 *
 * Staged through a tile because thread `i` and thread `i - 1` both read row
 * `i`: without the tile every label crosses the memory system twice. The tile
 * is one element longer than the group so that the pair straddling the group
 * boundary is covered by the same load.
 *
 * A level only speaks where the levels above it tied, which is what makes a
 * multi-index compare lexicographically with one dispatch per level and no
 * per-level state. */
kernel void idx_pair_cmp(device const float* values [[buffer(0)]],
                         device const float* wide [[buffer(1)]],
                         device const float* valid [[buffer(2)]],
                         device float* cmp [[buffer(3)]],
                         constant uint& count [[buffer(4)]],
                         constant uint& has_wide [[buffer(5)]],
                         constant uint& has_valid [[buffer(6)]],
                         uint gid [[thread_position_in_grid]],
                         uint lid [[thread_index_in_threadgroup]],
                         uint wid [[threadgroup_position_in_grid]]) {
    threadgroup float low[PAIR_GROUP + 1];
    threadgroup float high[PAIR_GROUP + 1];
    uint base = wid * PAIR_GROUP;
    uint rows = count + 1u;
    if (base + lid < rows) {
        low[lid] = values[base + lid];
        if (has_wide != 0u) high[lid] = wide[base + lid];
    }
    if (lid == 0u && base + PAIR_GROUP < rows) {
        low[PAIR_GROUP] = values[base + PAIR_GROUP];
        if (has_wide != 0u) high[PAIR_GROUP] = wide[base + PAIR_GROUP];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gid >= count) return;
    if (cmp[gid] != 0.0f) return;
    if (has_valid != 0u && (valid[gid] == 0.0f || valid[gid + 1u] == 0.0f)) {
        cmp[gid] = 3.0f;
        return;
    }
    if (has_wide != 0u) {
        long before = wide_value(low[lid], high[lid]);
        long after = wide_value(low[lid + 1u], high[lid + 1u]);
        cmp[gid] = (after > before) ? 1.0f : ((after < before) ? -1.0f : 0.0f);
        return;
    }
    float before = low[lid];
    float after = low[lid + 1u];
    if (isnan(before) || isnan(after)) {
        cmp[gid] = 3.0f;
        return;
    }
    cmp[gid] = (after > before) ? 1.0f : ((after < before) ? -1.0f : 0.0f);
}

/* Rising pairs, falling pairs and tied pairs in one pass over the comparison
 * lane, since all three answers come from the same read and the lane is large
 * enough that reading it three times would be the whole cost.
 *
 * `step(a, b)` is `b >= a`, so a pair inside a range is the product of two
 * steps and no branch is needed to widen the count to four at a time. */
kernel void idx_order_tally(device const float* cmp [[buffer(0)]],
                            device float* partials [[buffer(1)]],
                            constant uint& count [[buffer(2)]],
                            uint gid [[thread_position_in_grid]],
                            uint lid [[thread_index_in_threadgroup]],
                            uint wid [[threadgroup_position_in_grid]]) {
    threadgroup float scratch[TALLY_GROUP];
    const uint span = TALLY_GROUP * TALLY_GROUPS;
    float rising = 0.0f;
    float falling = 0.0f;
    float tied = 0.0f;
    if ((count & 3u) == 0u) {
        device const float4* quads = (device const float4*)cmp;
        uint groups_of_four = count >> 2;
        for (uint i = gid; i < groups_of_four; i += span) {
            float4 v = quads[i];
            float4 not_below = step(0.0f, v);
            float4 not_above = step(v, 0.0f);
            rising += dot(not_below * step(v, 1.0f), float4(1.0f));
            falling += dot(step(-1.0f, v) * not_above, float4(1.0f));
            tied += dot(not_below * not_above, float4(1.0f));
        }
    } else {
        for (uint i = gid; i < count; i += span) {
            float v = cmp[i];
            if (v == 0.0f) {
                rising += 1.0f;
                falling += 1.0f;
                tied += 1.0f;
            } else if (v == 1.0f) {
                rising += 1.0f;
            } else if (v == -1.0f) {
                falling += 1.0f;
            }
        }
    }
    float carried[3] = {rising, falling, tied};
    for (uint slot = 0u; slot < 3u; slot++) {
        scratch[lid] = carried[slot];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = TALLY_GROUP / 2u; stride > 0u; stride >>= 1) {
            if (lid < stride) scratch[lid] += scratch[lid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lid == 0u) partials[slot * TALLY_GROUPS + wid] = scratch[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void idx_run_starts(device const float* cmp [[buffer(0)]],
                           device float* out [[buffer(1)]],
                           constant uint& count [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out[gid] = (gid == 0u || cmp[gid - 1u] != 0.0f) ? 1.0f : 0.0f;
}

/* One thread per query, a bisect apiece.
 *
 * `mode` 0 and 1 are searchsorted's two sides; mode 2 is the exact lookup,
 * which is the left side plus the one comparison that turns a position into a
 * hit or a miss. Folding the three into one kernel keeps the comparison — the
 * only part that has to know about the second lane — written once. */
kernel void idx_search(device const float* keys [[buffer(0)]],
                       device const float* keys_wide [[buffer(1)]],
                       device const float* wanted [[buffer(2)]],
                       device const float* wanted_wide [[buffer(3)]],
                       device const float* wanted_valid [[buffer(4)]],
                       device float* out [[buffer(5)]],
                       constant uint& count [[buffer(6)]],
                       constant uint& length [[buffer(7)]],
                       constant uint& has_wide [[buffer(8)]],
                       constant uint& has_valid [[buffer(9)]],
                       constant uint& mode [[buffer(10)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    /* A missing label sorts past every present one, the way NaN does, and can
     * never be an exact hit: this plan is only taken by an index that has no
     * null of its own to match it. */
    if (has_valid != 0u && wanted_valid[gid] == 0.0f) {
        out[gid] = (mode == 2u) ? -1.0f : (float)length;
        return;
    }
    long want_exact = (has_wide != 0u) ? wide_value(wanted[gid], wanted_wide[gid]) : 0L;
    float want = wanted[gid];
    uint lo = 0u;
    uint hi = length;
    while (lo < hi) {
        uint mid = (lo + hi) >> 1;
        bool before;
        if (has_wide != 0u) {
            long here = wide_value(keys[mid], keys_wide[mid]);
            before = (mode == 1u) ? (here <= want_exact) : (here < want_exact);
        } else {
            float here = keys[mid];
            before = (mode == 1u) ? (here <= want) : (here < want);
        }
        if (before) lo = mid + 1u; else hi = mid;
    }
    if (mode != 2u) {
        out[gid] = (float)lo;
        return;
    }
    bool hit = false;
    if (lo < length) {
        hit = (has_wide != 0u) ? (wide_value(keys[lo], keys_wide[lo]) == want_exact)
                               : (keys[lo] == want);
    }
    out[gid] = hit ? (float)lo : -1.0f;
}

/* Turn a left-side bisect into the neighbour a fill method asks for.
 *
 * Separate from the bisect because all three methods want the same position
 * and differ only in what they do with it, and because `nearest` needs both
 * neighbours, which the bisect already brackets. */
kernel void idx_method_pick(device const float* keys [[buffer(0)]],
                            device const float* keys_wide [[buffer(1)]],
                            device const float* wanted [[buffer(2)]],
                            device const float* wanted_wide [[buffer(3)]],
                            device const float* wanted_valid [[buffer(4)]],
                            device const float* bisect [[buffer(5)]],
                            device float* out [[buffer(6)]],
                            constant uint& count [[buffer(7)]],
                            constant uint& length [[buffer(8)]],
                            constant uint& has_wide [[buffer(9)]],
                            constant uint& has_valid [[buffer(10)]],
                            constant uint& method [[buffer(11)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (length == 0u || (has_valid != 0u && wanted_valid[gid] == 0.0f)) {
        out[gid] = -1.0f;
        return;
    }
    int at = (int)bisect[gid];
    bool exact = false;
    if (at < (int)length) {
        exact = (has_wide != 0u)
            ? (wide_value(keys[at], keys_wide[at]) == wide_value(wanted[gid], wanted_wide[gid]))
            : (keys[at] == wanted[gid]);
    }
    if (method == 0u) {
        out[gid] = exact ? (float)at : (float)(at - 1);
        return;
    }
    if (method == 1u) {
        out[gid] = (at < (int)length) ? (float)at : -1.0f;
        return;
    }
    int above = at;
    int below = at - 1;
    if (above >= (int)length) {
        out[gid] = (float)below;
        return;
    }
    if (below < 0) {
        out[gid] = (float)above;
        return;
    }
    /* A tie goes to the earlier label, so that a nearest reindex over evenly
     * spaced labels reads the same way a pad does. */
    if (has_wide != 0u) {
        long want = wide_value(wanted[gid], wanted_wide[gid]);
        long up = wide_value(keys[above], keys_wide[above]) - want;
        long down = want - wide_value(keys[below], keys_wide[below]);
        out[gid] = (down <= up) ? (float)below : (float)above;
        return;
    }
    float want = wanted[gid];
    float up = keys[above] - want;
    float down = want - keys[below];
    out[gid] = (down <= up) ? (float)below : (float)above;
}

/* A range index inverted. `start + i * step == label` is one division, and
 * the multiply back is what says whether the label was on the ramp at all.
 *
 * Metal's integer division truncates towards zero, so a label below `start`
 * lands one position high; the remainder's sign is what corrects it, and the
 * corrected remainder is also the "did it land on a label" test. */
kernel void idx_range_locate(device const float* values [[buffer(0)]],
                             device const float* wide [[buffer(1)]],
                             device const float* valid [[buffer(2)]],
                             device float* out [[buffer(3)]],
                             constant uint& count [[buffer(4)]],
                             constant uint& length [[buffer(5)]],
                             constant int& start [[buffer(6)]],
                             constant int& step [[buffer(7)]],
                             constant uint& has_wide [[buffer(8)]],
                             constant uint& has_valid [[buffer(9)]],
                             constant uint& mode [[buffer(10)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (has_valid != 0u && valid[gid] == 0.0f) {
        out[gid] = (mode == 0u) ? -1.0f : (float)length;
        return;
    }
    long want;
    bool integral = true;
    if (has_wide != 0u) {
        want = wide_value(values[gid], wide[gid]);
    } else {
        float raw = values[gid];
        if (isnan(raw)) {
            out[gid] = (mode == 0u) ? -1.0f : (float)length;
            return;
        }
        want = (long)floor(raw);
        integral = ((float)want == raw);
    }
    if (!integral) {
        if (mode == 0u) {
            out[gid] = -1.0f;
            return;
        }
        /* A label between two rungs of the ramp sorts where the next whole
         * one does, whichever side was asked for. */
        want += 1;
    }
    long delta = want - (long)start;
    long span = (long)step;
    long at = delta / span;
    long rest = delta - at * span;
    if (rest != 0 && ((rest < 0) != (span < 0))) {
        at -= 1;
        rest += span;
    }
    if (mode == 0u) {
        bool inside = rest == 0 && at >= 0 && at < (long)length;
        out[gid] = inside ? (float)at : -1.0f;
        return;
    }
    long position = (mode == 2u && integral) ? (at + 1) : ((rest == 0) ? at : at + 1);
    if (position < 0) position = 0;
    if (position > (long)length) position = (long)length;
    out[gid] = (float)position;
}

kernel void idx_fill_holes(device float* values [[buffer(0)]],
                           device float* wide [[buffer(1)]],
                           device float* valid [[buffer(2)]],
                           device const float* positions [[buffer(3)]],
                           constant uint& count [[buffer(4)]],
                           constant uint& has_wide [[buffer(5)]],
                           constant uint& low_bits [[buffer(6)]],
                           constant uint& high_bits [[buffer(7)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (positions[gid] >= 0.0f) return;
    values[gid] = as_type<float>(low_bits);
    if (has_wide != 0u) wide[gid] = as_type<float>(high_bits);
    valid[gid] = 1.0f;
}

kernel void idx_hit_mask(device const float* positions [[buffer(0)]],
                         device float* out [[buffer(1)]],
                         constant uint& count [[buffer(2)]],
                         constant uint& want [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    bool found = positions[gid] >= 0.0f;
    out[gid] = (found == (want != 0u)) ? 1.0f : 0.0f;
}

kernel void idx_match_scalar(device const float* values [[buffer(0)]],
                             device const float* wide [[buffer(1)]],
                             device const float* valid [[buffer(2)]],
                             device float* out [[buffer(3)]],
                             constant uint& count [[buffer(4)]],
                             constant uint& has_wide [[buffer(5)]],
                             constant uint& has_valid [[buffer(6)]],
                             constant uint& low_bits [[buffer(7)]],
                             constant uint& high_bits [[buffer(8)]],
                             constant uint& want_null [[buffer(9)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    bool missing = has_valid != 0u && valid[gid] == 0.0f;
    if (want_null != 0u) {
        out[gid] = missing ? 1.0f : 0.0f;
        return;
    }
    if (missing) {
        out[gid] = 0.0f;
        return;
    }
    bool same = values[gid] == as_type<float>(low_bits);
    if (same && has_wide != 0u) same = wide[gid] == as_type<float>(high_bits);
    out[gid] = same ? 1.0f : 0.0f;
}

/* The digits of a cartesian product: row `gid` takes digit
 * `(gid / stride_k) % size_k` from level `k`.
 *
 * Walked from the fastest-varying level so that the strides never have to be
 * formed. One division and one multiply-subtract per level carries the
 * remainder down; the device has no integer remainder instruction, so a
 * division and a modulo apiece would cost two divisions per level.
 *
 * The digits of every level go into one lane, `level * count` apart, because
 * a kernel cannot be handed a list of outputs whose length is only known at
 * dispatch. Each level then gathers through its own window of it. */
kernel void idx_product_digits(device const float* sizes [[buffer(0)]],
                               device float* out [[buffer(1)]],
                               constant uint& count [[buffer(2)]],
                               constant uint& levels [[buffer(3)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint rest = gid;
    for (uint level = levels; level > 0u; level--) {
        uint size = (uint)sizes[level - 1u];
        uint quotient = rest / size;
        out[(level - 1u) * count + gid] = (float)(rest - quotient * size);
        rest = quotient;
    }
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

fn require_device(what: str) -> void {
    if not gpu.is_available() {
        throw RuntimeError(f"jaiframe needs a GPU to {what}")
    }
}

fn flag(value: bool) -> int { return if value { 1 } else { 0 } }

#: Two's-complement bits of `value` as a non-negative int, because a kernel
#: scalar is a `uint32_t` and a negative one is refused before it reaches Metal.
fn u32(value: int) -> int { return value & 0xFFFFFFFF }

#: The three lanes a kernel binds for a column.
#:
#: A column that has no second lane and no validity lane binds its value lane
#: in their place and tells the kernel to ignore it with a flag, which is what
#: `Column` does internally and the only way to pass a lane that may not exist.
fn value_lane(column: Column) -> Buffer { return column.values }

fn wide_lane(column: Column) -> Buffer { return column.wide ?? column.values }

fn valid_lane(column: Column) -> Buffer { return column.valid ?? column.values }

#: Slots for a table over `rows` labels: a power of two, at least twice as many
#: as there are rows.
#:
#: Half full is what keeps the clusters in a linear-probed table short enough
#: that the in-kernel walk is a handful of reads. It also guarantees the walk
#: terminates, because an empty slot always exists.
fn table_capacity(rows: int) -> int {
    var capacity = 16
    while capacity < rows * 2 { capacity *= 2 }
    return capacity
}

#: How many entries of a 0/1 lane are set.
fn tally(flags: Buffer, count: int) -> int {
    if count == 0 { return 0 }
    let partials = Buffer(TALLY_GROUPS)
    kernel_for("idx_tally").dispatch_async(
        [flags, partials],
        TALLY_GROUPS * TALLY_GROUP,
        [count],
        group_size: TALLY_GROUP
    )
    var total = 0.0
    for part in partials.download() { total += part }
    partials.free()
    return int(total)
}

fn fill_u32(lane: Buffer, count: int, value: int) -> void {
    if count == 0 { return }
    kernel_for("idx_fill_u32").dispatch_async([lane], count, [count, u32(value)])
}

fn names_of(index: Index) -> list[str?] { return [name for name in index.names] }

#: A `Bool` column over a freshly written flag lane.
fn mask_column(length: int) -> Column { return Column.allocate(Dtype.Bool, length) }

#: The rows a mask keeps, as a positions lane and its length.
#:
#: Built by compressing the identity permutation, so the whole thing is the
#: scan and the scatter `Column.compress` already dispatches — nothing walks
#: the mask on the host.
fn selected_positions(mask: Column) -> (Buffer, int) {
    let ramp = Column.allocate(Dtype.Float64, mask.len())
    if mask.len() > 0 {
        cached_kernel("col_iota").dispatch_async([ramp.values], mask.len(), [mask.len()])
    }
    let kept = ramp.compress(mask)
    ramp.free()
    return (kept.values, kept.length)
}

#: True where an indexer found a label, or where it did not when `found` is
#: false.
fn hit_mask(positions: Buffer, count: int, found: bool) -> Column {
    let mask = mask_column(count)
    if count > 0 {
        kernel_for("idx_hit_mask").dispatch_async(
            [positions, mask.values],
            count,
            [count, flag(found)]
        )
    }
    return mask
}

#: How each adjacent pair of labels is ordered, and what that says about the
#: index as a whole.
#:
#: Returns the comparison lane — which `PLAN_SORTED` keeps, because the runs of
#: equal labels it marks are also the answer to `unique` — together with
#: whether every pair rises, whether every pair falls, and how many tie. The
#: caller owns the lane.
fn index_order(index: Index) -> (Buffer, bool, bool, int) {
    if index.is_range() {
        throw ValueError("a range index is ordered by its step, not by comparing labels")
    }
    require_device("order an index")
    let pairs = if index.len() > 0 { index.len() - 1 } else { 0 }
    let cmp = Buffer(if pairs > 0 { pairs } else { 1 })
    cmp.fill_zero()
    if pairs == 0 { return (cmp, true, true, 0) }
    let groups = (pairs + PAIR_GROUP - 1) // PAIR_GROUP
    for level in index.levels {
        kernel_for("idx_pair_cmp").dispatch_async(
            [value_lane(level), wide_lane(level), valid_lane(level), cmp],
            groups * PAIR_GROUP,
            [pairs, flag(level.dtype.is_wide()), flag(level.has_validity())],
            group_size: PAIR_GROUP
        )
    }
    let partials = Buffer(3 * TALLY_GROUPS)
    kernel_for("idx_order_tally").dispatch_async(
        [cmp, partials],
        TALLY_GROUPS * TALLY_GROUP,
        [pairs],
        group_size: TALLY_GROUP
    )
    let counts = partials.download()
    partials.free()
    var rising = 0.0
    var falling = 0.0
    var tied = 0.0
    for slot in 0..TALLY_GROUPS {
        rising += counts[slot]
        falling += counts[TALLY_GROUPS + slot]
        tied += counts[2 * TALLY_GROUPS + slot]
    }
    return (cmp, int(rising) == pairs, int(falling) == pairs, int(tied))
}

#: Whether the labels never decrease, which is what pandas means by monotonic
#: increasing: `[1, 1, 2]` is one and `[1, 0]` is not.
#:
#: A multi-index is compared lexicographically. A null orders against nothing,
#: so an index that has one is neither increasing nor decreasing.
pub fn is_monotonic_increasing(index: Index) -> bool {
    if index.is_range() { return index.len() <= 1 or index.step > 0 }
    let order = index_order(index)
    order[0].free()
    return order[1]
}

#: Whether the labels never increase.
pub fn is_monotonic_decreasing(index: Index) -> bool {
    if index.is_range() { return index.len() <= 1 or index.step < 0 }
    let order = index_order(index)
    order[0].free()
    return order[2]
}

#: Whether no label is repeated.
#:
#: Costs whatever the lookup for this index costs — nothing for a range index,
#: one pass over the comparison lane for a monotonic one, and the hash table
#: for anything else. Hold an `IndexLookup` when the answer is wanted beside
#: other lookups over the same index.
pub fn is_unique(index: Index) -> bool {
    let lookup = IndexLookup.of(index)
    defer { lookup.free() }
    return lookup.is_unique()
}

#: A prepared way of asking one index where a label sits.
#:
#: Held rather than hidden because building it is the expensive half of a
#: lookup and probing it is the cheap half: a join, a set operation and a
#: reindex all probe the same index several times, and each of them builds one
#: of these and reuses it. `get_loc` and friends build one per call, which is
#: the right cost for a single question and the wrong one for a loop.
#:
#: Which plan it takes is decided here, once, from the labels themselves. The
#: caller never chooses.
#:
#: ```
#: let lookup = IndexLookup.of(frame.index)
#: defer { lookup.free() }
#: let positions = lookup.locate(target)
#: ```
pub class IndexLookup {
    #: `PLAN_RANGE`, `PLAN_SORTED` or `PLAN_HASHED`.
    pub let plan: int
    #: How many labels the index carries.
    pub let length: int
    #: How many distinct labels it carries.
    pub let distinct: int

    var levels: list[Column]
    var order: Buffer?
    var hashes: Buffer?
    var slots: Buffer?
    var rep: Buffer?
    var settled: Buffer?
    var mask: int
    var start: int
    var step: int
    var freed: bool

    fn init(self) {
        self.plan = PLAN_RANGE
        self.length = 0
        self.distinct = 0
        self.levels = []
        self.order = null
        self.hashes = null
        self.slots = null
        self.rep = null
        self.settled = null
        self.mask = 0
        self.start = 0
        self.step = 1
        self.freed = false
    }

    #: Prepare a lookup over `index`.
    #:
    #: The level columns are borrowed, not copied: a lookup lives inside one
    #: operation and the index outlives it. Freeing the index while a lookup
    #: over it is still live is the one thing that breaks that.
    pub static fn of(index: Index) -> IndexLookup {
        require_device("look a label up")
        let lookup = IndexLookup()
        lookup.length = index.len()
        lookup.start = index.start
        lookup.step = index.step
        if index.is_range() {
            lookup.plan = PLAN_RANGE
            lookup.distinct = index.len()
            return lookup
        }
        lookup.levels = [level for level in index.levels]
        if index.levels.len() == 1 {
            let ordered = index_order(index)
            #: A null makes a pair unorderable, so an index with one is never
            #: reported as rising — except an index of a single row, which has
            #: no pair at all. The bisect reads the value lane without a mask,
            #: so that one row has to be ruled out here.
            if ordered[1] and index.levels[0].null_count() == 0 {
                lookup.plan = PLAN_SORTED
                lookup.order = ordered[0]
                lookup.distinct = index.len() - ordered[3]
                return lookup
            }
            ordered[0].free()
        }
        lookup.plan = PLAN_HASHED
        lookup.build()
        return lookup
    }

    #: Whether no label is repeated.
    pub fn is_unique(self) -> bool { return self.distinct == self.length }

    #: How many distinct labels the index carries.
    pub fn nunique(self) -> int { return self.distinct }

    #: True at each row that is the first to carry its label.
    #:
    #: The mask `unique` compresses by, and the one a caller wanting the
    #: duplicates inverts.
    pub fn unique_mask(self) -> Column {
        self.check_live()
        let mask = mask_column(self.length)
        if self.length == 0 { return mask }
        if self.plan == PLAN_RANGE {
            mask.values.fill(1.0)
            return mask
        }
        if self.plan == PLAN_SORTED {
            #: A monotonic index puts every repeat of a label next to the
            #: first of them, so the runs the comparison lane already marks are
            #: the answer and no table is needed to find them.
            kernel_for("idx_run_starts").dispatch_async(
                [self.order ?? mask.values, mask.values],
                self.length,
                [self.length]
            )
            return mask
        }
        kernel_for("idx_first_of_kind").dispatch_async(
            [self.rep ?? mask.values, self.settled ?? mask.values, mask.values],
            self.length,
            [self.length]
        )
        return mask
    }

    #: Where each label of `target` sits in this index, `NOT_FOUND` for a miss.
    #:
    #: The lane is `target.len()` long and is exactly what `Column.take` with
    #: `allow_missing` consumes, which is why a reindex is this call plus one
    #: take per column.
    pub fn locate(self, target: Index) -> Buffer {
        self.check_live()
        let count = target.len()
        let out = Buffer(count)
        if count == 0 { return out }
        if self.plan == PLAN_RANGE {
            self.locate_over_range(target, out, count)
            return out
        }
        let aligned = aligned_levels(self.levels, target)
        let probes = aligned[0]
        if self.plan == PLAN_SORTED {
            self.locate_by_bisect(probes[0], out, count)
        } else {
            self.locate_by_table(probes, out, count)
        }
        release_levels(probes, aligned[1])
        return out
    }

    #: Where one label sits, raising `KeyError` when the index has no such row.
    pub fn locate_one(self, key: list[Value]) -> int {
        self.check_live()
        let target = index_of_key(key, self.level_dtypes())
        let positions = self.locate(target)
        let at = int(positions.download(0, 1)[0])
        positions.free()
        target.free()
        if at < 0 { throw KeyError(f"{spell_key(key)} is not in the index") }
        return at
    }

    #: Release the table and the lanes behind it. Freeing twice is harmless.
    #:
    #: The level columns are the index's, not the lookup's, and are left alone.
    pub fn free(self) -> void {
        if self.freed { return }
        self.freed = true
        for lane in [self.order, self.hashes, self.slots, self.rep, self.settled] {
            if lane is not null { lane.free() }
        }
    }

    #: Whether `free` has already run.
    pub fn is_freed(self) -> bool { return self.freed }

    fn level_dtypes(self) -> list[Dtype] {
        if self.plan == PLAN_RANGE { return [Dtype.Int64] }
        return self.levels.map(|level| level.dtype)
    }

    fn check_live(self) -> void {
        if self.freed { throw RuntimeError("this IndexLookup has already been released") }
    }

    #: Fold the levels into one hash per row and settle every row into a slot.
    fn build(self) -> void {
        let rows = self.length
        self.mask = table_capacity(rows) - 1
        let hashes = Buffer(if rows > 0 { rows } else { 1 })
        let slots = Buffer(self.mask + 1)
        let rep = Buffer(self.mask + 1)
        let settled = Buffer(if rows > 0 { rows } else { 1 })
        hashes.fill_zero()
        slots.fill_zero()
        self.hashes = hashes
        self.slots = slots
        self.rep = rep
        self.settled = settled
        if rows == 0 {
            self.distinct = 0
            return
        }
        for (at, level) in self.levels.enumerate() { level.hash_into(hashes, seed: at + 1) }
        let cursor = Buffer(rows)
        let candidate = Buffer(rows)
        let flags = Buffer(rows)
        let pending = Buffer(rows)
        let rowwise = iota(rows)
        kernel_for("idx_start_probe").dispatch_async([hashes, cursor], rows, [rows, self.mask])
        fill_u32(settled, rows, 0xFFFFFFFF)
        var round = 0
        loop {
            kernel_for("idx_insert").dispatch_async(
                [slots, hashes, cursor, settled, candidate],
                rows,
                [rows, self.mask]
            )
            flags.fill(1.0)
            for level in self.levels { level.equal_rows(candidate, level, rowwise, rows, flags) }
            kernel_for("idx_settle").dispatch_async(
                [flags, candidate, cursor, settled, pending],
                rows,
                [rows, self.mask]
            )
            if tally(pending, rows) == 0 { break }
            round += 1
            if round > SETTLE_ROUNDS {
                throw RuntimeError(
                    f"an index of {rows} labels did not settle into its table in {round} rounds"
                )
            }
        }
        fill_u32(rep, self.mask + 1, 0x7FFFFFFF)
        kernel_for("idx_elect").dispatch_async([rep, settled], rows, [rows])
        kernel_for("idx_first_of_kind").dispatch_async([rep, settled, flags], rows, [rows])
        self.distinct = tally(flags, rows)
        cursor.free()
        candidate.free()
        flags.free()
        pending.free()
        rowwise.free()
    }

    fn locate_over_range(self, target: Index, out: Buffer, count: int) -> void {
        let borrowed = borrowed_levels(target)
        let raw = borrowed[0]
        let built = borrowed[1]
        if raw.len() != 1 {
            release_all(raw, built)
            throw ValueError(f"a range index has one level, not {raw.len()}")
        }
        let probe = raw[0]
        if probe.dtype.is_dictionary() {
            release_all(raw, built)
            throw TypeError(f"a range index cannot hold a {probe.dtype.name()} label")
        }
        kernel_for("idx_range_locate").dispatch_async(
            [value_lane(probe), wide_lane(probe), valid_lane(probe), out],
            count,
            [
                count,
                self.length,
                u32(self.start),
                u32(self.step),
                flag(probe.dtype.is_wide()),
                flag(probe.has_validity()),
                0,
            ]
        )
        release_all(raw, built)
    }

    fn locate_by_bisect(self, probe: Column, out: Buffer, count: int) -> void {
        let keys = self.levels[0]
        kernel_for("idx_search").dispatch_async(
            [
                value_lane(keys),
                wide_lane(keys),
                value_lane(probe),
                wide_lane(probe),
                valid_lane(probe),
                out,
            ],
            count,
            [count, self.length, flag(keys.dtype.is_wide()), flag(probe.has_validity()), 2]
        )
    }

    fn locate_by_table(self, probes: list[Column], out: Buffer, count: int) -> void {
        let slots = self.slots ?? out
        let hashes = self.hashes ?? out
        let rep = self.rep ?? out
        let queries = Buffer(count)
        let cursor = Buffer(count)
        let candidate = Buffer(count)
        let flags = Buffer(count)
        let pending = Buffer(count)
        let rowwise = iota(count)
        queries.fill_zero()
        for (at, level) in probes.enumerate() { level.hash_into(queries, seed: at + 1) }
        kernel_for("idx_start_probe").dispatch_async([queries, cursor], count, [count, self.mask])
        #: -3 is "still walking". A slot the walk ran off the end of writes -1,
        #: which is the miss the caller reads, and both are below every row.
        out.fill(-3.0)
        var round = 0
        loop {
            kernel_for("idx_probe").dispatch_async(
                [slots, hashes, queries, cursor, out, candidate],
                count,
                [count, self.mask]
            )
            flags.fill(1.0)
            for (at, level) in self.levels.enumerate() {
                level.equal_rows(candidate, probes[at], rowwise, count, flags)
            }
            kernel_for("idx_accept").dispatch_async(
                [flags, candidate, rep, cursor, out, pending],
                count,
                [count, self.mask]
            )
            if tally(pending, count) == 0 { break }
            round += 1
            if round > SETTLE_ROUNDS {
                throw RuntimeError(f"a probe of {count} labels did not finish in {round} rounds")
            }
        }
        queries.free()
        cursor.free()
        candidate.free()
        flags.free()
        pending.free()
        rowwise.free()
    }
}

#: Where one label sits, raising `KeyError` when the index has no such row.
#:
#: Builds a lookup, asks it one question and throws it away. A loop over labels
#: should not be written at all — that is what `get_indexer` is — but if one
#: is, hold the `IndexLookup` across it.
pub fn get_loc(index: Index, label: Value) -> int {
    return get_loc_tuple(index, [label])
}

#: Where one multi-index label sits, one `Value` per level.
pub fn get_loc_tuple(index: Index, key: list[Value]) -> int {
    let lookup = IndexLookup.of(index)
    defer { lookup.free() }
    return lookup.locate_one(key)
}

#: Where each of `labels` sits, `NOT_FOUND` for the ones that are not there.
#:
#: One level only; a multi-index is asked through `get_indexer_for_index`.
pub fn get_indexer(index: Index, labels: list[Value]) -> Buffer {
    let target = index_of_key_columns([labels], [level_dtype(index, 0)])
    let positions = get_indexer_for_index(index, target)
    target.free()
    return positions
}

#: Where each label of `target` sits in `index`, `NOT_FOUND` for a miss.
#:
#: The direction is pandas': the result is `target.len()` long and holds
#: positions into `index`.
pub fn get_indexer_for_index(index: Index, target: Index) -> Buffer {
    let lookup = IndexLookup.of(index)
    defer { lookup.free() }
    return lookup.locate(target)
}

#: Where each of `values` would be inserted to keep `index` sorted.
#:
#: `side` decides which end of a run of equal labels the answer sits at:
#: `SIDE_LEFT` before it, `SIDE_RIGHT` after. The index has to be monotonic
#: increasing, because the answer is defined by that order and a bisect over
#: anything else is a wrong number rather than a slow one.
#:
#: Text compares by dictionary code, so a value the index has never seen sorts
#: past everything — its code is assigned after the index's.
pub fn searchsorted(index: Index, values: Column, side: str = SIDE_LEFT) -> Buffer {
    let mode = match side {
        "left" => 0,
        "right" => 1,
        _ => throw ValueError(f"searchsorted takes side left or right, not {side}"),
    }
    require_device("search an index")
    let count = values.len()
    let out = Buffer(if count > 0 { count } else { 1 })
    if count == 0 { return out }
    if index.is_range() {
        if index.step <= 0 {
            throw ValueError("searchsorted needs a monotonic increasing index")
        }
        if values.dtype.is_dictionary() {
            throw TypeError(f"a range index cannot be searched for a {values.dtype.name()}")
        }
        kernel_for("idx_range_locate").dispatch_async(
            [value_lane(values), wide_lane(values), valid_lane(values), out],
            count,
            [
                count,
                index.len(),
                u32(index.start),
                u32(index.step),
                flag(values.dtype.is_wide()),
                flag(values.has_validity()),
                mode + 1,
            ]
        )
        return out
    }
    if index.levels.len() != 1 {
        throw ValueError("searchsorted reads one level, not a whole multi-index")
    }
    if not is_monotonic_increasing(index) {
        throw ValueError("searchsorted needs a monotonic increasing index")
    }
    let keys = index.levels[0]
    let probe = recoded_like(values, keys)
    kernel_for("idx_search").dispatch_async(
        [
            value_lane(keys),
            wide_lane(keys),
            value_lane(probe),
            wide_lane(probe),
            valid_lane(probe),
            out,
        ],
        count,
        [count, index.len(), flag(keys.dtype.is_wide()), flag(probe.has_validity()), mode]
    )
    if probe != values { probe.free() }
    return out
}

#: The labels with the repeats dropped, keeping the first of each.
pub fn unique(index: Index) -> Index {
    let lookup = IndexLookup.of(index)
    defer { lookup.free() }
    if lookup.is_unique() { return index.copy() }
    let mask = lookup.unique_mask()
    defer { mask.free() }
    let picked = selected_positions(mask)
    defer { picked[0].free() }
    return index.take(picked[0], picked[1])
}

#: The two indexes stacked end to end, repeats and all.
#:
#: Text levels are put on one dictionary as they are stacked, which is what
#: makes the codes of the result mean one thing.
pub fn append(left: Index, right: Index) -> Index {
    if left.nlevels() != right.nlevels() {
        throw ValueError(f"cannot append {right.nlevels()} index levels onto {left.nlevels()}")
    }
    let here = borrowed_levels(left)
    let there = borrowed_levels(right)
    var levels: list[Column] = []
    for (at, part) in here[0].enumerate() {
        levels.push(concat_columns([part, there[0][at]]))
    }
    release_all(here[0], here[1])
    release_all(there[0], there[1])
    return Index.of(levels, shared_names(left, right))
}

#: Every label of either index, each once, in the order they first appear.
#:
#: pandas sorts a union when it can. This one does not: sorting is
#: `sort_index`, one module over, and a caller who wants sorted labels can say
#: so. Order that comes from the inputs is also the only order a union of text
#: labels has, since dictionary codes are interning order, not alphabet.
pub fn union(left: Index, right: Index) -> Index {
    let stacked = append(left, right)
    let deduped = unique(stacked)
    stacked.free()
    return deduped
}

#: The labels both indexes carry, each once, in the left index's order.
pub fn intersection(left: Index, right: Index) -> Index {
    return filtered_by_membership(left, right, true)
}

#: The labels the left index carries and the right does not, each once.
pub fn difference(left: Index, right: Index) -> Index {
    return filtered_by_membership(left, right, false)
}

#: The labels exactly one of the two carries, the left index's first.
pub fn symmetric_difference(left: Index, right: Index) -> Index {
    let only_left = difference(left, right)
    let only_right = difference(right, left)
    let joined = append(only_left, only_right)
    only_left.free()
    only_right.free()
    return joined
}

#: Whether the two label the same rows with the same labels.
#:
#: Names are not compared — `identical` is the one that compares those. Nor
#: are dtypes promoted: an `Int64` label and a `Float64` label of the same
#: value are not equal here, where pandas would promote them. A range index
#: does compare equal to the stored ramp it stands for, because that is the
#: comparison every alignment makes.
pub fn equals(left: Index, right: Index) -> bool {
    if left.len() != right.len() { return false }
    if left.nlevels() != right.nlevels() { return false }
    if left.is_range() and right.is_range() {
        return left.len() == 0 or left.start == right.start and left.step == right.step
    }
    if left.len() == 0 { return true }
    require_device("compare two indexes")
    let here = borrowed_levels(left)
    let there = borrowed_levels(right)
    let above = here[0]
    let below = there[0]
    #: Two labels of different dtypes are not equal, and asking `equal_rows`
    #: to compare them would raise rather than answer. pandas promotes an int
    #: label to a float one; the promotion has a precision cost, so it belongs
    #: to a caller who asked for it and not to a comparison.
    var comparable = true
    for (at, level) in above.enumerate() {
        if level.dtype.is_dictionary() != below[at].dtype.is_dictionary() {
            comparable = false
        } elif not level.dtype.is_dictionary() and level.dtype != below[at].dtype {
            comparable = false
        }
    }
    var same = false
    if comparable {
        let rowwise = iota(left.len())
        let flags = Buffer(left.len())
        flags.fill(1.0)
        for (at, level) in above.enumerate() {
            let probe = recoded_like(below[at], level)
            level.equal_rows(rowwise, probe, rowwise, left.len(), flags)
            if probe != below[at] { probe.free() }
        }
        same = tally(flags, left.len()) == left.len()
        flags.free()
        rowwise.free()
    }
    release_all(above, here[1])
    release_all(below, there[1])
    return same
}

#: Whether the two are the same index: the same labels, the same names, and
#: the same kind — a range index is not identical to the ramp it stands for.
pub fn identical(left: Index, right: Index) -> bool {
    if left.is_range() != right.is_range() { return false }
    if names_of(left) != names_of(right) { return false }
    if not left.is_range() {
        for (at, level) in left.levels.enumerate() {
            if level.dtype != right.levels[at].dtype { return false }
        }
    }
    return equals(left, right)
}

#: The index two frames are put on to be combined.
#:
#: `how` is pandas': `outer` takes every label, `inner` the shared ones, `left`
#: and `right` keep one side's index untouched. Two indexes that already agree
#: come back unchanged whatever `how` says, which is the case an arithmetic
#: expression between two columns of one frame hits every time.
pub fn joint_index(left: Index, right: Index, how: str = "outer") -> Index {
    if equals(left, right) { return left.copy() }
    return match how {
        "outer" => union(left, right),
        "inner" => intersection(left, right),
        "left" => left.copy(),
        "right" => right.copy(),
        _ => throw ValueError(f"unknown join {how}"),
    }
}

#: The frame's rows put onto `target`, one take per column.
#:
#: A label of `target` the frame does not have becomes a missing row, or
#: `fill_value` when one is given. `method` fills those holes from a
#: neighbouring label instead: `METHOD_PAD` from the last label at or before,
#: `METHOD_BACKFILL` from the first at or after, `METHOD_NEAREST` from
#: whichever is closer — all three need a monotonic increasing index, since
#: that is what "before" means.
#:
#: The index has to be unique, as pandas' does: with a label on two rows there
#: is no one row to move.
pub fn reindex(
    frame: DataFrame,
    target: Index,
    method: str? = null,
    fill_value: Value? = null
) -> DataFrame {
    let positions = reindex_indexer(frame.index, target, method)
    var columns: list[Column] = []
    for column in frame.columns {
        let taken = column.take(positions, target.len(), allow_missing: true)
        fill_holes(taken, positions, fill_value)
        columns.push(taken)
    }
    positions.free()
    return DataFrame([name for name in frame.names], columns, index: target.copy())
}

#: The series' rows put onto `target`, on the same terms as `reindex`.
pub fn reindex_series(
    series: Series,
    target: Index,
    method: str? = null,
    fill_value: Value? = null
) -> Series {
    let positions = reindex_indexer(series.index, target, method)
    let taken = series.values.take(positions, target.len(), allow_missing: true)
    fill_holes(taken, positions, fill_value)
    positions.free()
    return Series(taken, index: target.copy(), name: series.name)
}

#: Both frames on one index, which is what an operation between them needs.
#:
#: The pair comes back in the order it went in. This is the call `arith.jai`
#: makes before every binary operation, so its shape is fixed: two frames in,
#: two frames on the same index out, and `how` deciding which labels survive.
pub fn align(left: DataFrame, right: DataFrame, how: str = "outer") -> (DataFrame, DataFrame) {
    let joint = joint_index(left.index, right.index, how)
    let a = reindex(left, joint)
    let b = reindex(right, joint)
    joint.free()
    return (a, b)
}

#: Both series on one index.
pub fn align_series(left: Series, right: Series, how: str = "outer") -> (Series, Series) {
    let joint = joint_index(left.index, right.index, how)
    let a = reindex_series(left, joint)
    let b = reindex_series(right, joint)
    joint.free()
    return (a, b)
}

#: A frame and a series on one index, for the row-wise half of frame
#: arithmetic.
pub fn align_frame_series(
    frame: DataFrame,
    series: Series,
    how: str = "outer"
) -> (DataFrame, Series) {
    let joint = joint_index(frame.index, series.index, how)
    let a = reindex(frame, joint)
    let b = reindex_series(series, joint)
    joint.free()
    return (a, b)
}

#: Where each label of `target` sits in `index`, honouring a fill method.
#:
#: Split out because `reindex` and `reindex_series` want the same lane and
#: because a caller reindexing several frames onto one target should compute
#: it once.
pub fn reindex_indexer(index: Index, target: Index, method: str? = null) -> Buffer {
    require_device("reindex")
    if equals(index, target) { return iota(index.len()) }
    let lookup = IndexLookup.of(index)
    defer { lookup.free() }
    if not lookup.is_unique() {
        throw ValueError("cannot reindex onto an index whose labels repeat")
    }
    if method is null { return lookup.locate(target) }
    return method_indexer(index, target, method ?? METHOD_PAD)
}

#: A multi-index over one column per level.
#:
#: The columns are copied, so the caller keeps whatever it passed in. pandas
#: spells this `MultiIndex.from_arrays`; `Index.of`, in column.jai, is the
#: adopting form for a caller that just built the columns.
pub fn from_arrays(arrays: list[Column], names: list[str?]? = null) -> Index {
    if arrays.len() == 0 { throw ValueError("an index needs at least one level") }
    return Index.of(arrays.map(|level| level.copy()), names)
}

#: A multi-index over one label tuple per row.
#:
#: Host data, so the cost is the boxed cell per level per row that the caller
#: already paid to write them down. Each level takes its dtype from its first
#: cell that is not missing.
pub fn from_tuples(tuples: list[list[Value]], names: list[str?]? = null) -> Index {
    if tuples.len() == 0 { throw ValueError("an index needs at least one label") }
    let width = tuples[0].len()
    if width == 0 { throw ValueError("an index label needs at least one level") }
    var levels: list[Column] = []
    for at in 0..width {
        var cells: list[Value] = []
        for row in tuples {
            if row.len() != width {
                for level in levels { level.free() }
                throw ValueError(f"a label of {row.len()} levels among labels of {width}")
            }
            cells.push(row[at])
        }
        levels.push(column_of(cells, inferred_dtype(cells)))
    }
    return Index.of(levels, names)
}

#: Every combination of the levels, the last varying fastest.
#:
#: The digits are worked out on the device — one division per level, walked
#: from the fastest-varying end so the remainder carries down and no level
#: needs a modulo. Each level then gathers itself through its own digits, so a
#: product of a thousand by a thousand is two dispatches and never a host loop
#: over the million rows it makes.
pub fn from_product(levels: list[Column], names: list[str?]? = null) -> Index {
    if levels.len() == 0 { throw ValueError("a product needs at least one level") }
    require_device("build a product index")
    var total = 1
    var sizes: list[float] = []
    for level in levels {
        if level.len() == 0 { total = 0 }
        sizes.push(float(level.len()))
    }
    if total != 0 {
        for level in levels { total *= level.len() }
    }
    if total > MAX_ROWS {
        throw ValueError(f"a product of {total} rows exceeds the {MAX_ROWS} a lane addresses")
    }
    if total == 0 {
        let nothing = iota(0)
        let empty = levels.map(|level| level.take(nothing, 0))
        nothing.free()
        return Index.of(empty, names)
    }
    let extents = Buffer(sizes.len())
    extents.upload(sizes)
    let digits = Buffer(levels.len() * total)
    kernel_for("idx_product_digits").dispatch_async([extents, digits], total, [total, levels.len()])
    var built: list[Column] = []
    for (at, level) in levels.enumerate() {
        let window = digits.view(at * total, total)
        built.push(level.take(window, total))
        window.free()
    }
    extents.free()
    digits.free()
    return Index.of(built, names)
}

#: The index with one level taken out.
pub fn droplevel(index: Index, level: any = 0) -> Index {
    let at = resolve_level(index, level)
    if index.nlevels() < 2 {
        throw ValueError("an index cannot drop its only level")
    }
    var levels: list[Column] = []
    var names: list[str?] = []
    for (position, column) in index.levels.enumerate() {
        if position == at { continue }
        levels.push(column.copy())
        names.push(index.names[position])
    }
    return Index.of(levels, names)
}

#: The index with two levels exchanged. The last two by default, as pandas
#: does, because that is the pair a `stack` and an `unstack` move.
pub fn swaplevel(index: Index, first: any = -2, second: any = -1) -> Index {
    let a = resolve_level(index, first)
    let b = resolve_level(index, second)
    var order: list[int] = []
    for position in 0..index.nlevels() {
        order.push(if position == a { b } elif position == b { a } else { position })
    }
    return reorder_levels(index, order)
}

#: The index with its levels in the order `order` names.
pub fn reorder_levels(index: Index, order: list[int]) -> Index {
    if index.is_range() {
        throw ValueError("a range index has one level, so there is nothing to reorder")
    }
    if order.len() != index.nlevels() {
        throw ValueError(f"{order.len()} positions cannot reorder {index.nlevels()} levels")
    }
    var levels: list[Column] = []
    var names: list[str?] = []
    var seen: dict[int, bool] = {}
    for position in order {
        let at = if position < 0 { index.nlevels() + position } else { position }
        if at < 0 or at >= index.nlevels() {
            for level in levels { level.free() }
            throw IndexError(f"level {position} is outside an index of {index.nlevels()}")
        }
        if seen.get(at, false) {
            for level in levels { level.free() }
            throw ValueError(f"level {at} appears twice in the new order")
        }
        seen[at] = true
        levels.push(index.levels[at].copy())
        names.push(index.names[at])
    }
    return Index.of(levels, names)
}

#: One level's labels as a column, freshly allocated.
#:
#: A range index materialises into the `Int64` ramp it stands for, which is the
#: one place the implied labels become storage.
pub fn get_level_values(index: Index, level: any = 0) -> Column {
    let at = resolve_level(index, level)
    if index.is_range() {
        let built = index.materialize()
        return built[0]
    }
    return index.levels[at].copy()
}

#: The index with its level names replaced.
pub fn set_names(index: Index, names: list[str?]) -> Index {
    if names.len() != index.nlevels() {
        throw ValueError(f"{names.len()} names cannot name {index.nlevels()} levels")
    }
    if index.is_range() {
        return Index.range(index.len(), start: index.start, step: index.step, name: names[0])
    }
    return Index.of(index.levels.map(|level| level.copy()), [name for name in names])
}

#: The frame with its index levels renamed.
#:
#: pandas takes a scalar for a single-level index; here a one-element list says
#: the same thing without a second spelling of the argument.
pub fn rename_axis(frame: DataFrame, names: list[str?]) -> DataFrame {
    return frame._replace(index: set_names(frame.index, names))
}

#: The rows whose label on one level is `key`, with that level taken out.
#:
#: The comparison against `key` is one dispatch — a scalar resolved once on the
#: host and compared as an integer, which for a text level means resolving it
#: to a dictionary code and, when the level has never seen it, answering with
#: no rows and no dispatch at all.
#:
#: A single-level index keeps its level whatever `drop_level` says: there would
#: be nothing left to label the rows with.
pub fn xs(frame: DataFrame, key: Value, level: any = 0, drop_level: bool = true) -> DataFrame {
    let at = resolve_level(frame.index, level)
    let labels = get_level_values(frame.index, at)
    let mask = matches_scalar(labels, key)
    labels.free()
    let picked = selected_positions(mask)
    mask.free()
    let rows = picked[1]
    var columns: list[Column] = []
    for column in frame.columns { columns.push(column.take(picked[0], rows)) }
    let taken = frame.index.take(picked[0], rows)
    picked[0].free()
    if not drop_level or taken.nlevels() < 2 {
        return DataFrame([name for name in frame.names], columns, index: taken)
    }
    let narrowed = droplevel(taken, at)
    taken.free()
    return DataFrame([name for name in frame.names], columns, index: narrowed)
}

#: The frame relabelled by some of its own columns.
#:
#: `drop` leaves those columns out of the result, which is what pandas does by
#: default; `append` puts them under the index the frame already had rather
#: than replacing it. The index takes copies, so the result and the original
#: can both be freed.
pub fn set_index(
    frame: DataFrame,
    keys: list[str],
    drop: bool = true,
    append: bool = false
) -> DataFrame {
    if keys.len() == 0 { throw ValueError("set_index needs at least one column") }
    var levels: list[Column] = []
    var names: list[str?] = []
    if append {
        for level in frame.index.materialize() { levels.push(level) }
        for name in frame.index.names { names.push(name) }
    }
    var moved: dict[str, bool] = {}
    for key in keys {
        if moved.get(key, false) {
            for level in levels { level.free() }
            throw ValueError(f"column {key} is named twice")
        }
        moved[key] = true
        levels.push(frame.column(key).copy())
        names.push(key)
    }
    var kept: list[str] = []
    var columns: list[Column] = []
    for (at, name) in frame.names.enumerate() {
        if drop and moved.get(name, false) { continue }
        kept.push(name)
        columns.push(frame.columns[at])
    }
    return DataFrame(kept, columns, index: Index.of(levels, names))
}

#: The frame's index levels moved back into columns.
#:
#: `level` names which of them to move, all of them when it is null. `drop`
#: throws them away instead of inserting them. The levels land at the front, in
#: index order, which is where pandas puts them. A level with no name becomes
#: `index` on a single-level index and `level_0`, `level_1` ... on a
#: multi-index, again as pandas does.
pub fn reset_index(frame: DataFrame, level: list[int]? = null, drop: bool = false) -> DataFrame {
    let total = frame.index.nlevels()
    var moving: list[int] = []
    if level is null {
        for at in 0..total { moving.push(at) }
    } else {
        for entry in level ?? [] { moving.push(resolve_level(frame.index, entry)) }
    }
    var taken: dict[int, bool] = {}
    for at in moving { taken[at] = true }
    var names: list[str] = []
    var columns: list[Column] = []
    if not drop {
        let materialized = frame.index.materialize()
        for at in moving {
            names.push(frame.index.names[at] ?? default_level_name(at, total))
            columns.push(materialized[at])
        }
        for (at, level_column) in materialized.enumerate() {
            if not taken.get(at, false) { level_column.free() }
        }
    }
    for (at, name) in frame.names.enumerate() {
        names.push(name)
        columns.push(frame.columns[at])
    }
    var remaining: list[Column] = []
    var remaining_names: list[str?] = []
    if not frame.index.is_range() {
        for (at, level_column) in frame.index.levels.enumerate() {
            if taken.get(at, false) { continue }
            remaining.push(level_column.copy())
            remaining_names.push(frame.index.names[at])
        }
    }
    let rebuilt = if remaining.len() == 0 {
        Index.range(frame.len())
    } else {
        Index.of(remaining, remaining_names)
    }
    return DataFrame(names, columns, index: rebuilt)
}

fn default_level_name(at: int, total: int) -> str {
    return if total == 1 { "index" } else { f"level_{at}" }
}

#: A level named by its position, negative from the end, or by its name.
fn resolve_level(index: Index, level: any) -> int {
    let total = index.nlevels()
    if level is int {
        let position: int = level
        let at = if position < 0 { total + position } else { position }
        if at < 0 or at >= total {
            throw IndexError(f"level {position} is outside an index of {total} levels")
        }
        return at
    }
    if level is str {
        for (at, name) in index.names.enumerate() {
            if name == level { return at }
        }
        throw KeyError(f"no index level named {level}")
    }
    throw TypeError("an index level is named by its position or by its name")
}

#: How one level of an index stores its labels. A range index stores none, and
#: the ramp it stands for is `Int64`.
fn level_dtype(index: Index, level: int) -> Dtype {
    if index.is_range() { return Dtype.Int64 }
    return index.levels[level].dtype
}

#: The name two indexes agree on for a level, `null` where they do not.
#:
#: pandas drops a name a set operation cannot justify keeping, and a
#: disagreement is exactly that case.
fn shared_names(left: Index, right: Index) -> list[str?] {
    var names: list[str?] = []
    for (at, name) in left.names.enumerate() {
        names.push(if at < right.names.len() and name == right.names[at] { name } else { null })
    }
    return names
}

#: The dtype a list of cells wants: the first one that is not missing, and
#: `Float64` when every one of them is.
fn inferred_dtype(cells: list[Value]) -> Dtype {
    for cell in cells {
        if not cell.is_na() { return cell.dtype }
    }
    return Dtype.Float64
}

#: Host cells as a column of `dtype`.
#:
#: The boundary where labels a program wrote down become a device column. The
#: validity list is left null when nothing is missing, so the column does not
#: carry a lane of ones for a kernel to read.
fn column_of(cells: list[Value], dtype: Dtype) -> Column {
    var missing = false
    for cell in cells {
        if cell.is_na() { missing = true }
    }
    if dtype.is_dictionary() {
        var text: list[str?] = []
        for cell in cells { text.push(if cell.is_na() { null } else { cell.to_str() }) }
        return Column.from_strs(text, dtype)
    }
    var nulls: list[bool]? = null
    if missing { nulls = cells.map(|cell| cell.is_na()) }
    if dtype.is_wide() {
        return Column.from_ints(cells.map(|cell| if cell.is_na() {
            0
        } else {
            cell.to_int()
        }), nulls, dtype)
    }
    if dtype == Dtype.Bool {
        return Column.from_bools(cells.map(|cell| not cell.is_na() and cell.to_bool()), nulls)
    }
    return Column.from_floats(cells.map(|cell| if cell.is_na() {
        0.0
    } else {
        cell.to_float()
    }), nulls)
}

#: A one-row index carrying one label, for the single-label lookups.
fn index_of_key(key: list[Value], dtypes: list[Dtype]) -> Index {
    if key.len() != dtypes.len() {
        throw ValueError(f"a label of {key.len()} levels cannot index {dtypes.len()}")
    }
    return index_of_key_columns(key.map(|cell| [cell]), dtypes)
}

#: An index over one host list of cells per level.
fn index_of_key_columns(cells: list[list[Value]], dtypes: list[Dtype]) -> Index {
    var levels: list[Column] = []
    for (at, column_cells) in cells.enumerate() {
        levels.push(column_of(column_cells, dtypes[at]))
    }
    return Index.of(levels)
}

fn spell_key(key: list[Value]) -> str {
    if key.len() == 1 { return key[0].to_str() }
    var parts: list[str] = []
    for cell in key { parts.push(cell.to_str()) }
    let joined = parts.join(", ")
    return f"({joined})"
}

#: `source` recoded onto `model`'s dictionary, so that the two columns' codes
#: mean the same string.
#:
#: The model's own codes are left alone — its table is the prefix of the merged
#: one — which is what lets a lookup hash an index once and reuse that hash
#: against any number of targets. A string the model has never seen is given a
#: code past the model's table, so it compares equal to nothing, which is
#: exactly the miss it should be.
#:
#: Returns `source` itself when there is nothing to do, so the caller has to
#: compare before freeing.
fn recoded_like(source: Column, model: Column) -> Column {
    if not model.dtype.is_dictionary() {
        if source.dtype != model.dtype {
            throw TypeError(
                f"cannot look a {source.dtype.name()} label up in a {model.dtype.name()} index"
            )
        }
        return source
    }
    if not source.dtype.is_dictionary() {
        throw TypeError(
            f"cannot look a {source.dtype.name()} label up in a {model.dtype.name()} index"
        )
    }
    var names: list[str] = [name for name in model.levels ?? []]
    var intern: dict[str, int] = {}
    for (code, name) in names.enumerate() { intern[name] = code }
    var novel = false
    for name in source.levels ?? [] {
        if intern.get(name, -1) >= 0 { continue }
        intern[name] = names.len()
        names.push(name)
        novel = true
    }
    if not novel and source.dtype == model.dtype and names.len() == source.level_count() {
        var identical_table = true
        for (code, name) in (source.levels ?? []).enumerate() {
            if names[code] != name { identical_table = false }
        }
        if identical_table { return source }
    }
    let out = Column.allocate(model.dtype, source.length, with_validity: source.has_validity())
    out.set_levels(names)
    let mask = source.valid
    let out_mask = out.valid
    if mask is not null and out_mask is not null and source.length > 0 {
        cached_kernel("col_copy_lane").dispatch_async(
            [mask, out_mask],
            source.length,
            [source.length, 0, 0]
        )
    }
    if source.length == 0 { return out }
    let table = source.levels ?? []
    if table.len() == 0 {
        cached_kernel("col_copy_lane").dispatch_async(
            [source.values, out.values],
            source.length,
            [source.length, 0, 0]
        )
        return out
    }
    var mapping: list[float] = []
    for name in table { mapping.push(float(intern.get(name, 0))) }
    let lane = Buffer(mapping.len())
    lane.upload(mapping)
    cached_kernel("col_remap").dispatch_async(
        [source.values, out.values, lane],
        source.length,
        [source.length, table.len()]
    )
    lane.free()
    return out
}

#: An index's levels as columns to read, and whether the caller has to free
#: them.
#:
#: A stored index lends its own columns; only a range index has to build the
#: ramp it stands for. Copying instead would put a full pass over the labels in
#: front of every lookup, alignment and comparison, which for an index that is
#: only ever read is a pass for nothing.
fn borrowed_levels(index: Index) -> (list[Column], bool) {
    if index.is_range() { return (index.materialize(), true) }
    return ([level for level in index.levels], false)
}

fn release_levels(levels: list[Column], owned: list[bool]) -> void {
    for (at, level) in levels.enumerate() {
        if owned[at] { level.free() }
    }
}

fn release_all(levels: list[Column], owned: bool) -> void {
    if not owned { return }
    for level in levels { level.free() }
}

#: `target`'s levels as columns comparable with `model`'s, with a flag per
#: level saying whether it was built here or borrowed from the index.
fn aligned_levels(model: list[Column], target: Index) -> (list[Column], list[bool]) {
    let borrowed = borrowed_levels(target)
    let raw = borrowed[0]
    let built = borrowed[1]
    if raw.len() != model.len() {
        release_all(raw, built)
        throw ValueError(f"an index of {raw.len()} levels cannot be looked up in {model.len()}")
    }
    var probes: list[Column] = []
    var owned: list[bool] = []
    for (at, level) in raw.enumerate() {
        var recoded = level
        try {
            recoded = recoded_like(level, model[at])
        } catch error: Error {
            release_levels(probes, owned)
            release_all(raw, built)
            throw error
        }
        if recoded != level {
            if built { level.free() }
            probes.push(recoded)
            owned.push(true)
        } else {
            probes.push(level)
            owned.push(built)
        }
    }
    return (probes, owned)
}

#: The unique labels of `left` that either are or are not in `right`.
fn filtered_by_membership(left: Index, right: Index, present: bool) -> Index {
    let deduped = unique(left)
    let positions = get_indexer_for_index(right, deduped)
    let mask = hit_mask(positions, deduped.len(), present)
    positions.free()
    let picked = selected_positions(mask)
    mask.free()
    let kept = deduped.take(picked[0], picked[1])
    picked[0].free()
    deduped.free()
    return kept
}

#: A `Bool` column, true where the column's label is `key`.
fn matches_scalar(column: Column, key: Value) -> Column {
    let mask = mask_column(column.len())
    if column.len() == 0 { return mask }
    if key.is_na() {
        if not column.has_validity() {
            mask.values.fill_zero()
            return mask
        }
        dispatch_match(column, mask, 0, 0, true)
        return mask
    }
    if column.dtype.is_dictionary() {
        let code = column.level_code(key.to_str())
        if code is null {
            #: A string the level has never held cannot label a row, and saying
            #: so costs a dictionary lookup rather than a pass over the column.
            mask.values.fill_zero()
            return mask
        }
        dispatch_match(column, mask, f32_bits(float(code ?? 0)), 0, false)
        return mask
    }
    if column.dtype.is_wide() {
        let split = wide_pack(key.to_int())
        dispatch_match(column, mask, f32_bits(split[0]), f32_bits(split[1]), false)
        return mask
    }
    dispatch_match(column, mask, f32_bits(key.to_float()), 0, false)
    return mask
}

fn dispatch_match(column: Column, mask: Column, low: int, high: int, want_null: bool) -> void {
    kernel_for("idx_match_scalar").dispatch_async(
        [value_lane(column), wide_lane(column), valid_lane(column), mask.values],
        column.len(),
        [
            column.len(),
            flag(column.dtype.is_wide()),
            flag(column.has_validity()),
            u32(low),
            u32(high),
            flag(want_null),
        ]
    )
}

#: Write `fill` into the rows a take left empty.
#:
#: Only the rows the indexer missed, not every null: a row that was already
#: missing in the source stays missing, which is what pandas' `fill_value`
#: means and what makes a reindex distinguishable from a `fillna`.
fn fill_holes(column: Column, positions: Buffer, fill: Value?) -> void {
    if fill is null { return }
    let cell = fill ?? NA
    if cell.is_na() or column.len() == 0 { return }
    var low = 0
    var high = 0
    if column.dtype.is_dictionary() {
        low = f32_bits(float(column.intern_level(cell.to_str())))
    } elif column.dtype.is_wide() {
        let split = wide_pack(cell.to_int())
        low = f32_bits(split[0])
        high = f32_bits(split[1])
    } else {
        low = f32_bits(cell.to_float())
    }
    kernel_for("idx_fill_holes").dispatch_async(
        [column.values, wide_lane(column), column.materialize_validity(), positions],
        column.len(),
        [column.len(), flag(column.dtype.is_wide()), u32(low), u32(high)]
    )
}

#: Where each label of `target` sits once a fill method has been allowed to
#: reach for a neighbour.
fn method_indexer(index: Index, target: Index, method: str) -> Buffer {
    let mode = match method {
        "ffill" | "pad" => 0,
        "bfill" | "backfill" => 1,
        "nearest" => 2,
        _ => throw ValueError(f"unknown reindex method {method}"),
    }
    if index.nlevels() != 1 or target.nlevels() != 1 {
        throw ValueError("a fill method reads one level, not a whole multi-index")
    }
    if not is_monotonic_increasing(index) {
        throw ValueError(f"reindex with method {method} needs a monotonic increasing index")
    }
    let source = borrowed_levels(index)
    let keys = source[0][0]
    let borrowed = borrowed_levels(target)
    let raw = borrowed[0]
    let probe = recoded_like(raw[0], keys)
    let count = target.len()
    let bisect = Buffer(if count > 0 { count } else { 1 })
    let out = Buffer(if count > 0 { count } else { 1 })
    if count > 0 {
        kernel_for("idx_search").dispatch_async(
            [
                value_lane(keys),
                wide_lane(keys),
                value_lane(probe),
                wide_lane(probe),
                valid_lane(probe),
                bisect,
            ],
            count,
            [count, index.len(), flag(keys.dtype.is_wide()), flag(probe.has_validity()), 0]
        )
        kernel_for("idx_method_pick").dispatch_async(
            [
                value_lane(keys),
                wide_lane(keys),
                value_lane(probe),
                wide_lane(probe),
                valid_lane(probe),
                bisect,
                out,
            ],
            count,
            [count, index.len(), flag(keys.dtype.is_wide()), flag(probe.has_validity()), mode]
        )
    }
    bisect.free()
    if probe != raw[0] { probe.free() }
    release_all(raw, borrowed[1])
    release_all(source[0], source[1])
    return out
}
