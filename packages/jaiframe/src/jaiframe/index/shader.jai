#: The Metal source every kernel in `index` is compiled from.
#:
#: One translation unit, so the table walk, the bisect and the level kernels
#: share the same inline helpers and the same constants, and `kernel_for`
#: compiles it once whichever entry point is asked for first.
pub let SOURCE = """#include <metal_stdlib>
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
