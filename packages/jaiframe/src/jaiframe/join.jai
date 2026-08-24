#: Combining frames: hash joins across the rows, concatenation along both axes.
#:
#: `merge` is the centre of it. `join`, the index alignment `concat(axis: 1)`
#: needs and `compare` are that machinery pointed at a different pair of key
#: columns. The two ordered merges are the exception: `merge_asof` and
#: `merge_ordered` bisect a sorted key lane instead, because a bucket answers
#: "which rows hold this key" and an asof merge is not asking that.
#:
#: A merge is a hash join, never a nested loop. Two frames of a million rows
#: are 10^12 key comparisons that way, which is hours; hashing both sides and
#: probing is two million hashes and about as many bucket reads. The nested
#: loop wins only when one side has a handful of rows, and it loses so badly
#: past that it is not worth the second code path.
#:
#: The table is built over whichever side has fewer rows and the other side
#: probes it. The table costs a bucket per build row — three lanes of it — so
#: building the smaller side is what keeps a join against a small dimension
#: table cheap in memory, and it leaves the long side streaming through the
#: probe, one bucket read per row, which is the access pattern a GPU wants.
#: The cost is that the output then comes out in probe order, and pandas' order
#: is the left frame's; when the probe is not the side the order follows, the
#: two position lanes are put back in order by a stable radix pass before a
#: single column is materialised. Reordering two lanes is far cheaper than
#: building the wrong table.
#:
#: The whole join is two lanes: left positions and right positions, with
#: `MISSING_POSITION` where a side has no row. Every result column is one
#: `Column.take(..., allow_missing: true)` off those lanes, so the width of the
#: frame costs a dispatch per column and nothing else.
#:
#: ```
#: let orders = pd.DataFrame(["city", "total"], [cities, totals])
#: let sizes = pd.DataFrame(["city", "population"], [names, people])
#: let joined = merge(orders, sizes, how: How.Left, on: "city")
#: defer { joined.free() }
#: ```
import std.gpu as gpu
from std.gpu import Buffer, Kernel
from .column import Column, DataFrame, Dtype, Index, MAX_ROWS, Value
from .column import cached_kernel, concat_columns, f32_bits, iota, prefix_sum
from .column import unify_levels, wide_pack
from .keys import factorize

#: Smallest hash table, so a tiny build side still spreads over a few buckets.
const MIN_BUCKETS = 16

let SOURCE = """#include <metal_stdlib>
using namespace metal;

constexpr constant long WIDE_SCALE = 16777216L;

/* Both lanes recombined in `long`, exactly as the column store does it. A join
 * key of milliseconds is past 2^24 on its first row, so a float comparison
 * here would put every timestamp in the same bucket of the search. */
inline long wide_value(float low, float high) {
    return (long)high * WIDE_SCALE + (long)low;
}

/* One comparison written once, because the asof and equal-range searches walk
 * the same lane pair and only differ in what they do with the answer. The
 * branch is uniform over the whole dispatch, so it costs a predicted jump. */
inline bool key_below(
    device const float* values,
    device const float* wide,
    uint has_wide,
    uint at,
    float target_f,
    long target_w
) {
    return (has_wide != 0u) ? (wide_value(values[at], wide[at]) < target_w)
                            : (values[at] < target_f);
}

inline bool key_above(
    device const float* values,
    device const float* wide,
    uint has_wide,
    uint at,
    float target_f,
    long target_w
) {
    return (has_wide != 0u) ? (wide_value(values[at], wide[at]) > target_w)
                            : (values[at] > target_f);
}

/* Which of the two neighbours `nearest` should take. The two gaps are compared
 * rather than returned, so each case stays in its own arithmetic: one shared
 * return type would have to be `long`, and truncating two sub-integer float
 * gaps to zero would send every near tie backward. */
inline bool forward_is_closer(
    device const float* values,
    device const float* wide,
    uint has_wide,
    uint back,
    uint fwd,
    float target_f,
    long target_w
) {
    if (has_wide != 0u) {
        long b = wide_value(values[back], wide[back]) - target_w;
        long f = wide_value(values[fwd], wide[fwd]) - target_w;
        return (f < 0L ? -f : f) < (b < 0L ? -b : b);
    }
    return fabs(values[fwd] - target_f) < fabs(values[back] - target_f);
}

/* pandas' tolerance is inclusive. A wide key's is counted in whole units of the
 * column — milliseconds for a timestamp — so it arrives as the same (low, high)
 * pair the lanes hold rather than as a float that stopped being exact at 2^24. */
inline bool within_tolerance(
    device const float* values,
    device const float* wide,
    uint has_wide,
    uint at,
    float target_f,
    long target_w,
    uint tol_lo,
    uint tol_hi,
    uint tol_bits
) {
    if (has_wide != 0u) {
        long d = wide_value(values[at], wide[at]) - target_w;
        return (d < 0L ? -d : d) <= (long)tol_hi * WIDE_SCALE + (long)tol_lo;
    }
    return fabs(values[at] - target_f) <= as_type<float>(tol_bits);
}

kernel void jn_bucket_of(device const uint* hashes [[buffer(0)]],
                         device uint* buckets [[buffer(1)]],
                         constant uint& count [[buffer(2)]],
                         constant uint& mask [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    buckets[gid] = hashes[gid] & mask;
}

kernel void jn_copy_uint(device const uint* src [[buffer(0)]],
                         device uint* dst [[buffer(1)]],
                         constant uint& count [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    dst[gid] = src[gid];
}

kernel void jn_radix_flags(device const uint* keys [[buffer(0)]],
                           device float* flags [[buffer(1)]],
                           constant uint& count [[buffer(2)]],
                           constant uint& bit [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    flags[gid] = (((keys[gid] >> bit) & 1u) == 0u) ? 1.0f : 0.0f;
}

/* The zero-bit entries keep their scanned rank and the one-bit entries follow
 * them, which is what makes the pass stable: within a bit value the input
 * order survives. The total sits in the last slot of the scan, so the host
 * never has to learn it to encode this dispatch. */
kernel void jn_radix_scatter(device const float* order_in [[buffer(0)]],
                             device const uint* keys_in [[buffer(1)]],
                             device const float* flags [[buffer(2)]],
                             device const float* offsets [[buffer(3)]],
                             device float* order_out [[buffer(4)]],
                             device uint* keys_out [[buffer(5)]],
                             constant uint& count [[buffer(6)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint zeros = (uint)offsets[count];
    uint rank = (uint)offsets[gid];
    uint at = (flags[gid] != 0.0f) ? rank : zeros + (gid - rank);
    order_out[at] = order_in[gid];
    keys_out[at] = keys_in[gid];
}

/* Bucket spans off the sorted bucket lane. Both tables start zeroed, so a
 * bucket nothing landed in reads back as the empty span [0, 0). */
kernel void jn_bucket_bounds(device const uint* sorted [[buffer(0)]],
                             device uint* starts [[buffer(1)]],
                             device uint* ends [[buffer(2)]],
                             constant uint& count [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint bucket = sorted[gid];
    if (gid == 0u || sorted[gid - 1u] != bucket) starts[bucket] = gid;
    if (gid + 1u == count || sorted[gid + 1u] != bucket) ends[bucket] = gid + 1u;
}

kernel void jn_probe_counts(device const uint* buckets [[buffer(0)]],
                            device const uint* starts [[buffer(1)]],
                            device const uint* ends [[buffer(2)]],
                            device float* counts [[buffer(3)]],
                            constant uint& count [[buffer(4)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint bucket = buckets[gid];
    counts[gid] = (float)(ends[bucket] - starts[bucket]);
}

/* One thread per probe row rather than per candidate: with a bucket per build
 * row the chain a row walks is one entry on average, and a thread per
 * candidate would have to binary-search back to find which row it belonged to.
 * A key repeated ten thousand times does make its thread walk ten thousand
 * entries, but that same key is about to produce ten thousand output rows. */
kernel void jn_probe_pairs(device const uint* buckets [[buffer(0)]],
                           device const uint* starts [[buffer(1)]],
                           device const uint* ends [[buffer(2)]],
                           device const float* table [[buffer(3)]],
                           device const float* offsets [[buffer(4)]],
                           device float* probe_out [[buffer(5)]],
                           device float* build_out [[buffer(6)]],
                           constant uint& count [[buffer(7)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint bucket = buckets[gid];
    uint at = (uint)offsets[gid];
    uint to = ends[bucket];
    for (uint j = starts[bucket]; j < to; j++) {
        probe_out[at] = (float)gid;
        build_out[at] = table[j];
        at++;
    }
}

/* How many output rows a probe row contributes. A row that matched nothing
 * still contributes one when its side is preserved, which is the whole
 * difference between an inner join and a left one. */
kernel void jn_emit_sizes(device const float* offsets [[buffer(0)]],
                          device const float* matched [[buffer(1)]],
                          device float* sizes [[buffer(2)]],
                          constant uint& count [[buffer(3)]],
                          constant uint& preserve [[buffer(4)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint to = (uint)offsets[gid + 1u];
    float hits = 0.0f;
    for (uint c = (uint)offsets[gid]; c < to; c++) hits += matched[c];
    sizes[gid] = (hits > 0.0f) ? hits : ((preserve != 0u) ? 1.0f : 0.0f);
}

kernel void jn_emit_pairs(device const float* offsets [[buffer(0)]],
                          device const float* matched [[buffer(1)]],
                          device const float* cand_build [[buffer(2)]],
                          device const float* out_at [[buffer(3)]],
                          device float* probe_out [[buffer(4)]],
                          device float* build_out [[buffer(5)]],
                          constant uint& count [[buffer(6)]],
                          constant uint& preserve [[buffer(7)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint to = (uint)offsets[gid + 1u];
    uint at = (uint)out_at[gid];
    bool any = false;
    for (uint c = (uint)offsets[gid]; c < to; c++) {
        if (matched[c] == 0.0f) continue;
        probe_out[at] = (float)gid;
        build_out[at] = cand_build[c];
        at++;
        any = true;
    }
    if (!any && preserve != 0u) {
        probe_out[at] = (float)gid;
        build_out[at] = -1.0f;
    }
}

/* Every matching pair stores the same 1 into its build row's slot, so the
 * store is atomic only to keep it from being a data race on paper; which
 * thread wins cannot change the answer. */
kernel void jn_mark_matched(device const float* cand_build [[buffer(0)]],
                            device const float* matched [[buffer(1)]],
                            device atomic_uint* seen [[buffer(2)]],
                            constant uint& count [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (matched[gid] == 0.0f) return;
    atomic_store_explicit(&seen[(uint)cand_build[gid]], 1u, memory_order_relaxed);
}

kernel void jn_unmatched_flags(device const uint* seen [[buffer(0)]],
                               device float* flags [[buffer(1)]],
                               constant uint& count [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    flags[gid] = (seen[gid] == 0u) ? 1.0f : 0.0f;
}

kernel void jn_append_unmatched(device const float* flags [[buffer(0)]],
                                device const float* offsets [[buffer(1)]],
                                device float* probe_out [[buffer(2)]],
                                device float* build_out [[buffer(3)]],
                                constant uint& count [[buffer(4)]],
                                constant uint& base [[buffer(5)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (flags[gid] == 0.0f) return;
    uint at = base + (uint)offsets[gid];
    probe_out[at] = -1.0f;
    build_out[at] = (float)gid;
}

/* One division for the row and a subtraction for the column. A second `%`
 * would cost a second reciprocal, because the device has no integer divide. */
kernel void jn_cross_pairs(device float* left_out [[buffer(0)]],
                           device float* right_out [[buffer(1)]],
                           constant uint& count [[buffer(2)]],
                           constant uint& width [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint row = gid / width;
    left_out[gid] = (float)row;
    right_out[gid] = (float)(gid - row * width);
}

/* Where an output row reads its merged key from, once the two key columns have
 * been stacked into one: the left row when there is one, else the right row
 * shifted past the left block. */
kernel void jn_key_positions(device const float* left_lane [[buffer(0)]],
                             device const float* right_lane [[buffer(1)]],
                             device float* out [[buffer(2)]],
                             constant uint& count [[buffer(3)]],
                             constant uint& left_rows [[buffer(4)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    int left = (int)left_lane[gid];
    int right = (int)right_lane[gid];
    out[gid] = (left >= 0) ? (float)left
                           : ((right >= 0) ? (float)((int)left_rows + right) : -1.0f);
}

kernel void jn_indicator_codes(device const float* left_lane [[buffer(0)]],
                               device const float* right_lane [[buffer(1)]],
                               device float* out [[buffer(2)]],
                               constant uint& count [[buffer(3)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    bool left = left_lane[gid] >= 0.0f;
    bool right = right_lane[gid] >= 0.0f;
    out[gid] = (left && right) ? 2.0f : (left ? 0.0f : 1.0f);
}

/* Rows the ordering side does not name sort last, so an outer join's
 * right-only rows trail the left ones instead of leading them. */
kernel void jn_order_key(device const float* lane [[buffer(0)]],
                         device uint* keys [[buffer(1)]],
                         constant uint& count [[buffer(2)]],
                         constant uint& absent [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    int at = (int)lane[gid];
    keys[gid] = (at >= 0) ? (uint)at : absent;
}

kernel void jn_gather_lane(device const float* src [[buffer(0)]],
                           device const float* perm [[buffer(1)]],
                           device float* dst [[buffer(2)]],
                           constant uint& count [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    dst[gid] = src[(uint)perm[gid]];
}

kernel void jn_gather_uint(device const uint* src [[buffer(0)]],
                           device const float* perm [[buffer(1)]],
                           device uint* dst [[buffer(2)]],
                           constant uint& count [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    dst[gid] = src[(uint)perm[gid]];
}

kernel void jn_map_positions(device const float* src [[buffer(0)]],
                             device const float* map [[buffer(1)]],
                             device float* dst [[buffer(2)]],
                             constant uint& count [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    int at = (int)src[gid];
    dst[gid] = (at < 0) ? -1.0f : map[at];
}

/* An order-preserving 32-bit image of a float lane, so a radix pass over the
 * image sorts the values themselves: flip the sign bit of a positive and every
 * bit of a negative. A missing row and a NaN both take the largest key, which
 * puts them last the way pandas sorts them. */
kernel void jn_sort_key_float(device const float* values [[buffer(0)]],
                              device const float* valid [[buffer(1)]],
                              device uint* keys [[buffer(2)]],
                              constant uint& count [[buffer(3)]],
                              constant uint& has_valid [[buffer(4)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (has_valid != 0u && valid[gid] == 0.0f) {
        keys[gid] = 0xFFFFFFFFu;
        return;
    }
    float value = values[gid];
    if (isnan(value)) {
        keys[gid] = 0xFFFFFFFFu;
        return;
    }
    uint bits = as_type<uint>(value == 0.0f ? 0.0f : value);
    keys[gid] = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

/* The two halves of a wide lane as two radix digits. The high half is biased
 * by 2^24 so a negative sorts below a positive, and a missing row takes 2^25,
 * one past every real high half, so nulls land last whatever their low half
 * happened to be. */
kernel void jn_sort_key_wide(device const float* values [[buffer(0)]],
                             device const float* wide [[buffer(1)]],
                             device const float* valid [[buffer(2)]],
                             device uint* keys [[buffer(3)]],
                             constant uint& count [[buffer(4)]],
                             constant uint& has_valid [[buffer(5)]],
                             constant uint& high_half [[buffer(6)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    bool missing = has_valid != 0u && valid[gid] == 0.0f;
    if (high_half != 0u) {
        keys[gid] = missing ? 0x2000000u : (uint)((int)wide[gid] + 16777216);
    } else {
        keys[gid] = missing ? 0xFFFFFFu : (uint)values[gid];
    }
}

kernel void jn_sort_key_code(device const float* codes [[buffer(0)]],
                             device const float* valid [[buffer(1)]],
                             device uint* keys [[buffer(2)]],
                             constant uint& count [[buffer(3)]],
                             constant uint& has_valid [[buffer(4)]],
                             constant uint& absent [[buffer(5)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    bool missing = has_valid != 0u && valid[gid] == 0.0f;
    keys[gid] = missing ? absent : (uint)codes[gid];
}

/* Where a lane stops being non-decreasing. A null counts as a break: an asof
 * or ordered merge has no answer for a row whose key is missing, and finding
 * that out here is better than searching for it. */
kernel void jn_key_breaks(device const float* values [[buffer(0)]],
                          device const float* wide [[buffer(1)]],
                          device const float* valid [[buffer(2)]],
                          device float* flags [[buffer(3)]],
                          constant uint& count [[buffer(4)]],
                          constant uint& has_wide [[buffer(5)]],
                          constant uint& has_valid [[buffer(6)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (has_valid != 0u && valid[gid] == 0.0f) {
        flags[gid] = 1.0f;
        return;
    }
    if (gid == 0u) {
        flags[gid] = 0.0f;
        return;
    }
    bool back;
    if (has_wide != 0u) {
        back = wide_value(values[gid], wide[gid]) < wide_value(values[gid - 1u], wide[gid - 1u]);
    } else {
        back = values[gid] < values[gid - 1u];
    }
    flags[gid] = back ? 1.0f : 0.0f;
}

/* One binary search per left row over a sorted stretch of the right key lane.
 * `log2(n)` reads against the whole-lane walk a hash would have replaced it
 * with — and the right side is already sorted, which is what asof means, so
 * there is nothing to pay for the ordering. */
kernel void jn_asof_search(device const float* l_values [[buffer(0)]],
                           device const float* l_wide [[buffer(1)]],
                           device const float* r_values [[buffer(2)]],
                           device const float* r_wide [[buffer(3)]],
                           device const float* run_begin [[buffer(4)]],
                           device const float* run_end [[buffer(5)]],
                           device float* out [[buffer(6)]],
                           constant uint& count [[buffer(7)]],
                           constant uint& has_wide [[buffer(8)]],
                           constant uint& direction [[buffer(9)]],
                           constant uint& has_tolerance [[buffer(10)]],
                           constant uint& tol_lo [[buffer(11)]],
                           constant uint& tol_hi [[buffer(12)]],
                           constant uint& tol_bits [[buffer(13)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out[gid] = -1.0f;
    int begin = (int)run_begin[gid];
    int end = (int)run_end[gid];
    if (begin < 0 || end <= begin) return;

    float target_f = l_values[gid];
    long target_w = (has_wide != 0u) ? wide_value(l_values[gid], l_wide[gid]) : 0L;

    uint low = (uint)begin;
    uint high = (uint)end;
    while (low < high) {
        uint mid = low + ((high - low) >> 1);
        if (key_below(r_values, r_wide, has_wide, mid, target_f, target_w)) low = mid + 1u;
        else high = mid;
    }
    uint lower = low;

    low = lower;
    high = (uint)end;
    while (low < high) {
        uint mid = low + ((high - low) >> 1);
        if (key_above(r_values, r_wide, has_wide, mid, target_f, target_w)) high = mid;
        else low = mid + 1u;
    }
    uint upper = low;

    int backward = (upper > (uint)begin) ? (int)(upper - 1u) : -1;
    int forward = (lower < (uint)end) ? (int)lower : -1;
    int chosen;
    if (direction == 0u) {
        chosen = backward;
    } else if (direction == 1u) {
        chosen = forward;
    } else if (backward < 0) {
        chosen = forward;
    } else if (forward < 0) {
        chosen = backward;
    } else {
        chosen = forward_is_closer(
            r_values, r_wide, has_wide, (uint)backward, (uint)forward, target_f, target_w
        ) ? forward : backward;
    }
    if (chosen < 0) return;
    if (has_tolerance != 0u) {
        bool near = within_tolerance(
            r_values, r_wide, has_wide, (uint)chosen, target_f, target_w, tol_lo, tol_hi, tol_bits
        );
        if (!near) return;
    }
    out[gid] = (float)chosen;
}

/* The half-open stretch of a sorted lane holding a key, for the two ordered
 * merges. Two searches rather than one, because a duplicated key is a range
 * and an ordered merge has to emit all of it. */
kernel void jn_equal_range(device const float* t_values [[buffer(0)]],
                           device const float* t_wide [[buffer(1)]],
                           device const float* s_values [[buffer(2)]],
                           device const float* s_wide [[buffer(3)]],
                           device float* lower_out [[buffer(4)]],
                           device float* upper_out [[buffer(5)]],
                           constant uint& count [[buffer(6)]],
                           constant uint& source_rows [[buffer(7)]],
                           constant uint& has_wide [[buffer(8)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    float target_f = t_values[gid];
    long target_w = (has_wide != 0u) ? wide_value(t_values[gid], t_wide[gid]) : 0L;
    uint low = 0u;
    uint high = source_rows;
    while (low < high) {
        uint mid = low + ((high - low) >> 1);
        if (key_below(s_values, s_wide, has_wide, mid, target_f, target_w)) low = mid + 1u;
        else high = mid;
    }
    uint lower = low;
    high = source_rows;
    while (low < high) {
        uint mid = low + ((high - low) >> 1);
        if (key_above(s_values, s_wide, has_wide, mid, target_f, target_w)) high = mid;
        else low = mid + 1u;
    }
    lower_out[gid] = (float)lower;
    upper_out[gid] = (float)low;
}

kernel void jn_span_sizes(device const float* lower [[buffer(0)]],
                          device const float* upper [[buffer(1)]],
                          device float* sizes [[buffer(2)]],
                          constant uint& count [[buffer(3)]],
                          constant uint& preserve [[buffer(4)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    float span = upper[gid] - lower[gid];
    sizes[gid] = (span > 0.0f) ? span : ((preserve != 0u) ? 1.0f : 0.0f);
}

/* Positions into the sorted side, not into the frame: the caller maps both
 * lanes back through the permutation that sorted them, in one pass over the
 * finished output, so an appended right-only row and a matched one go through
 * the same remap. */
kernel void jn_span_pairs(device const float* lower [[buffer(0)]],
                          device const float* upper [[buffer(1)]],
                          device const float* out_at [[buffer(2)]],
                          device float* probe_out [[buffer(3)]],
                          device float* build_out [[buffer(4)]],
                          constant uint& count [[buffer(5)]],
                          constant uint& preserve [[buffer(6)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint at = (uint)out_at[gid];
    uint from = (uint)lower[gid];
    uint to = (uint)upper[gid];
    if (from >= to) {
        if (preserve != 0u) {
            probe_out[at] = (float)gid;
            build_out[at] = -1.0f;
        }
        return;
    }
    for (uint j = from; j < to; j++) {
        probe_out[at] = (float)gid;
        build_out[at] = (float)j;
        at++;
    }
}

/* A stretch of a float code lane read back as the unsigned digits a group
 * table is keyed on. Both sides were factorized together, so the left codes are
 * the first block of one lane and the right codes the block after it. */
kernel void jn_code_lane(device const float* codes [[buffer(0)]],
                         device uint* out [[buffer(1)]],
                         constant uint& count [[buffer(2)]],
                         constant uint& base [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out[gid] = (uint)codes[base + gid];
}

/* The stretch of the group-ordered right side one left row may search. The
 * code indexes the bounds directly — dense codes are the whole reason the
 * groups come from a factorization rather than from a hash — so a group the
 * right side never had reads back as the empty span [0, 0). */
kernel void jn_group_span(device const uint* codes [[buffer(0)]],
                          device const uint* starts [[buffer(1)]],
                          device const uint* ends [[buffer(2)]],
                          device float* begin_out [[buffer(3)]],
                          device float* end_out [[buffer(4)]],
                          constant uint& count [[buffer(5)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint code = codes[gid];
    begin_out[gid] = (float)starts[code];
    end_out[gid] = (float)ends[code];
}

kernel void jn_not_flags(device const float* src [[buffer(0)]],
                         device float* dst [[buffer(1)]],
                         constant uint& count [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    dst[gid] = (src[gid] == 0.0f) ? 1.0f : 0.0f;
}

kernel void jn_compact_positions(device const float* flags [[buffer(0)]],
                                 device const float* offsets [[buffer(1)]],
                                 device float* out [[buffer(2)]],
                                 constant uint& count [[buffer(3)]],
                                 uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (flags[gid] == 0.0f) return;
    out[(uint)offsets[gid]] = (float)gid;
}

/* A forward fill as a positions lane: a present row takes itself, a missing
 * one takes the last present row before it, which the scan already counted.
 * No new scan kernel and no serial walk — `take` finishes the job. */
kernel void jn_fill_positions(device const float* flags [[buffer(0)]],
                              device const float* offsets [[buffer(1)]],
                              device const float* compact [[buffer(2)]],
                              device float* out [[buffer(3)]],
                              constant uint& count [[buffer(4)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (flags[gid] != 0.0f) {
        out[gid] = (float)gid;
        return;
    }
    uint before = (uint)offsets[gid];
    out[gid] = (before == 0u) ? -1.0f : compact[before - 1u];
}

kernel void jn_unequal_positions(device const float* same [[buffer(0)]],
                                 device const float* rows [[buffer(1)]],
                                 device float* out [[buffer(2)]],
                                 constant uint& count [[buffer(3)]],
                                 constant uint& keep_equal [[buffer(4)]],
                                 uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint row = (uint)rows[gid];
    bool equal = same[row] != 0.0f;
    out[gid] = (equal && keep_equal == 0u) ? -1.0f : rows[gid];
}

kernel void jn_count_pairs(device const float* lane [[buffer(0)]],
                           device atomic_uint* counts [[buffer(1)]],
                           constant uint& count [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    int at = (int)lane[gid];
    if (at < 0) return;
    atomic_fetch_add_explicit(&counts[at], 1u, memory_order_relaxed);
}

kernel void jn_over_one(device const uint* counts [[buffer(0)]],
                        device float* flags [[buffer(1)]],
                        constant uint& count [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    flags[gid] = (counts[gid] > 1u) ? 1.0f : 0.0f;
}
"""

var _kernels: dict[str, Kernel] = {}

#: The compiled pipeline for one entry point in the join shader library.
fn kernel_for(entry: str) -> Kernel {
    let existing = _kernels.get(entry, null)
    if existing is not null { return existing }
    let compiled = Kernel.compile(SOURCE, entry)
    _kernels[entry] = compiled
    return compiled
}

fn flag(value: bool) -> int { return if value { 1 } else { 0 } }

fn require_device(what: str) -> void {
    if not gpu.is_available() {
        throw RuntimeError(f"jaiframe needs a GPU to {what}")
    }
}

#: How the rows of two frames are paired up.
pub enum How {
    #: Only the rows whose keys appear on both sides.
    Inner,
    #: Every left row, with nulls where the right side has none.
    Left,
    #: Every right row, with nulls where the left side has none.
    Right,
    #: Every row of both sides.
    Outer,
    #: Every left row against every right row, on no key at all.
    Cross,

    #: The pandas spelling.
    pub fn name(self) -> str {
        return match self {
            How.Inner => "inner",
            How.Left => "left",
            How.Right => "right",
            How.Outer => "outer",
            How.Cross => "cross",
        }
    }

    #: The join `name` spells, so `merge(how: How.from_name("left"))` reads.
    pub static fn from_name(text: str) -> How {
        return match text {
            "inner" => How.Inner,
            "left" => How.Left,
            "right" => How.Right,
            "outer" | "full" => How.Outer,
            "cross" => How.Cross,
            _ => throw ValueError(f"unknown join {text}"),
        }
    }

    #: Whether a left row with no match still appears.
    pub fn keeps_left(self) -> bool {
        return match self {
            How.Left | How.Outer => true,
            _ => false,
        }
    }

    #: Whether a right row with no match still appears.
    pub fn keeps_right(self) -> bool {
        return match self {
            How.Right | How.Outer => true,
            _ => false,
        }
    }
}

#: The shape a merge's keys are promised to have.
#:
#: Checked over the keys that take part in the join rather than over the whole
#: frame, which is where a duplicate would change the answer: a key present on
#: only one side cannot multiply anything.
pub enum Validate {
    #: Every key at most once on each side.
    OneToOne,
    #: Every key at most once on the left, any number on the right.
    OneToMany,
    #: Any number on the left, at most once on the right.
    ManyToOne,
    #: No promise; the check is skipped.
    ManyToMany,

    #: The pandas spelling.
    pub fn name(self) -> str {
        return match self {
            Validate.OneToOne => "1:1",
            Validate.OneToMany => "1:m",
            Validate.ManyToOne => "m:1",
            Validate.ManyToMany => "m:m",
        }
    }

    #: The promise `name` spells.
    pub static fn from_name(text: str) -> Validate {
        return match text {
            "1:1" | "one_to_one" => Validate.OneToOne,
            "1:m" | "one_to_many" => Validate.OneToMany,
            "m:1" | "many_to_one" => Validate.ManyToOne,
            "m:m" | "many_to_many" => Validate.ManyToMany,
            _ => throw ValueError(f"unknown validation {text}"),
        }
    }

    #: Whether the left side promises unique keys.
    pub fn left_unique(self) -> bool {
        return match self {
            Validate.OneToOne | Validate.OneToMany => true,
            _ => false,
        }
    }

    #: Whether the right side promises unique keys.
    pub fn right_unique(self) -> bool {
        return match self {
            Validate.OneToOne | Validate.ManyToOne => true,
            _ => false,
        }
    }
}

#: Which way `merge_asof` looks when a key has no exact match.
pub enum AsofDirection {
    #: The last right row at or before the left key.
    Backward,
    #: The first right row at or after the left key.
    Forward,
    #: Whichever of those two is closer; a tie goes backward.
    Nearest,

    #: The pandas spelling.
    pub fn name(self) -> str {
        return match self {
            AsofDirection.Backward => "backward",
            AsofDirection.Forward => "forward",
            AsofDirection.Nearest => "nearest",
        }
    }

    #: The direction `name` spells.
    pub static fn from_name(text: str) -> AsofDirection {
        return match text {
            "backward" => AsofDirection.Backward,
            "forward" => AsofDirection.Forward,
            "nearest" => AsofDirection.Nearest,
            _ => throw ValueError(f"unknown asof direction {text}"),
        }
    }
}

#: The direction as `jn_asof_search` numbers them, which is not something a
#: caller should have to know about.
fn direction_code(direction: AsofDirection) -> int {
    return match direction {
        AsofDirection.Backward => 0,
        AsofDirection.Forward => 1,
        AsofDirection.Nearest => 2,
    }
}

#: How many bits it takes to write `value`.
fn bit_width(value: int) -> int {
    var bits = 0
    var rest = value
    while rest > 0 {
        bits += 1
        rest = rest >> 1
    }
    return bits
}

#: The power-of-two bucket count a build side of `rows` gets.
#:
#: One bucket per build row, so the chain a probe row walks is one entry in
#: expectation. Two per row would halve the collisions and double a table that
#: is already three lanes wide; at load one the extra candidates are cleared by
#: the same equality pass that has to run anyway.
fn bucket_count(rows: int) -> int {
    var size = MIN_BUCKETS
    while size < rows { size = size * 2 }
    return size
}

fn copy_lane(src: Buffer, dst: Buffer, count: int) -> void {
    if count == 0 { return }
    cached_kernel("col_copy_lane").dispatch_async([src, dst], count, [count, 0, 0])
}

#: A lane of hash bits copied as bits.
#:
#: `col_copy_lane` moves the same bytes but reads them as floats on the way,
#: and a hash is as likely as not to be a signalling NaN pattern that a float
#: move is allowed to quieten.
fn copy_uint(src: Buffer, dst: Buffer, count: int) -> void {
    if count == 0 { return }
    kernel_for("jn_copy_uint").dispatch_async([src, dst], count, [count])
}

#: One hash per row, folding every key column into the same lane.
fn hash_lane(keys: list[Column], rows: int, seed: int) -> Buffer {
    let lane = Buffer(rows)
    lane.fill_zero()
    for key in keys { key.hash_into(lane, seed) }
    return lane
}

#: A stable ascending ordering of `count` entries by an unsigned key lane.
#:
#: One binary LSD pass per significant bit: a scan of the "this bit is clear"
#: flags, then a scatter that keeps the clear entries in scanned order and puts
#: the set ones after them. Stability is the property everything here leans on
#: — ordering build rows by bucket leaves each bucket in row order, and
#: ordering output pairs by their left row leaves each left row's matches in
#: right order — so neither caller needs a second key.
#:
#: `bits` comes from the range the keys actually span, so a join against a
#: hundred-row table pays seven passes rather than thirty-two. The scan hands
#: its total back to the host, which is one four-byte read per pass and the
#: only place this waits on the device.
#:
#: Returns fresh `(order, keys)` lanes the caller owns; the inputs are untouched.
fn radix_order(order_in: Buffer, keys_in: Buffer, count: int, bits: int) -> (Buffer, Buffer) {
    var order_front = Buffer(count)
    var keys_front = Buffer(count)
    copy_lane(order_in, order_front, count)
    copy_uint(keys_in, keys_front, count)
    if count <= 1 or bits <= 0 { return (order_front, keys_front) }
    var order_back = Buffer(count)
    var keys_back = Buffer(count)
    let flags = Buffer(count)
    for bit in 0..bits {
        kernel_for("jn_radix_flags").dispatch_async([keys_front, flags], count, [count, bit])
        let scanned = prefix_sum(flags, count)
        kernel_for("jn_radix_scatter").dispatch_async(
            [order_front, keys_front, flags, scanned[0], order_back, keys_back],
            count,
            [count]
        )
        scanned[0].free()
        let spare_order = order_front
        let spare_keys = keys_front
        order_front = order_back
        keys_front = keys_back
        order_back = spare_order
        keys_back = spare_keys
    }
    flags.free()
    order_back.free()
    keys_back.free()
    return (order_front, keys_front)
}

#: The bucket table one side of a join is built into.
#:
#: `rows_of` holds the build rows grouped by bucket and ascending inside each
#: one; `starts` and `ends` are the half-open span of each bucket. A bucket
#: nothing landed in reads back as [0, 0), so the probe needs no emptiness test.
class BuildTable {
    #: How many buckets, always a power of two.
    pub var buckets: int
    #: How many build rows the table holds.
    pub var rows: int
    #: First entry of each bucket.
    pub var starts: Buffer
    #: One past the last entry of each bucket.
    pub var ends: Buffer
    #: Build rows, grouped by bucket.
    pub var rows_of: Buffer

    #: Adopt the lanes. `build_table` is the only correct caller.
    pub fn init(self, buckets: int, rows: int, starts: Buffer, ends: Buffer, rows_of: Buffer) {
        self.buckets = buckets
        self.rows = rows
        self.starts = starts
        self.ends = ends
        self.rows_of = rows_of
    }

    #: Release the lanes.
    pub fn free(self) -> void {
        self.starts.free()
        self.ends.free()
        self.rows_of.free()
    }
}

fn build_table(keys: list[Column], rows: int) -> BuildTable {
    let buckets = bucket_count(rows)
    let starts = Buffer(buckets)
    let ends = Buffer(buckets)
    starts.fill_zero()
    ends.fill_zero()
    if rows == 0 { return BuildTable(buckets, rows, starts, ends, Buffer(0)) }
    let hashes = hash_lane(keys, rows, 0)
    let lane = Buffer(rows)
    kernel_for("jn_bucket_of").dispatch_async([hashes, lane], rows, [rows, buckets - 1])
    hashes.free()
    let base = iota(rows)
    let sorted = radix_order(base, lane, rows, bit_width(buckets - 1))
    base.free()
    lane.free()
    kernel_for("jn_bucket_bounds").dispatch_async([sorted[1], starts, ends], rows, [rows])
    sorted[1].free()
    return BuildTable(buckets, rows, starts, ends, sorted[0])
}

#: The two position lanes a join produces.
#:
#: One entry per output row on each side, `MISSING_POSITION` where that side
#: contributes nothing. Every result column is one `take` off these.
pub class Pairing {
    #: The left row each output row came from.
    pub var left: Buffer
    #: The right row each output row came from.
    pub var right: Buffer
    #: How many output rows.
    pub var rows: int

    #: Adopt two lanes of `rows` positions.
    pub fn init(self, left: Buffer, right: Buffer, rows: int) {
        self.left = left
        self.right = right
        self.rows = rows
    }

    #: Release both lanes.
    pub fn free(self) -> void {
        self.left.free()
        self.right.free()
    }
}

fn cross_pairing(n_left: int, n_right: int) -> Pairing {
    let rows = n_left * n_right
    if rows > MAX_ROWS {
        throw ValueError(
            f"a cross join of {n_left} by {n_right} is more than the {MAX_ROWS} rows a position lane addresses"
        )
    }
    let left = Buffer(rows)
    let right = Buffer(rows)
    if rows > 0 {
        kernel_for("jn_cross_pairs").dispatch_async([left, right], rows, [rows, n_right])
    }
    return Pairing(left, right, rows)
}

fn permute_lane(lane: Buffer, perm: Buffer, rows: int) -> void {
    let scratch = Buffer(rows)
    kernel_for("jn_gather_lane").dispatch_async([lane, perm, scratch], rows, [rows])
    copy_lane(scratch, lane, rows)
    scratch.free()
}

#: Put the output rows back in the ordering side's row order, in place.
#:
#: The probe side comes out in its own order and pandas reports the left
#: frame's — the right frame's for a right join. A stable pass by that side's
#: position fixes both at once: rows the ordering side does not name take a key
#: one past its last row and so trail everything, and the pass's stability
#: leaves each ordering row's matches in the other side's order.
fn reorder_pairing(pairing: Pairing, by_right: bool, n_left: int, n_right: int) -> void {
    let rows = pairing.rows
    if rows <= 1 { return }
    let lane = if by_right { pairing.right } else { pairing.left }
    let absent = if by_right { n_right } else { n_left }
    let keys = Buffer(rows)
    kernel_for("jn_order_key").dispatch_async([lane, keys], rows, [rows, absent])
    let base = iota(rows)
    let sorted = radix_order(base, keys, rows, bit_width(absent))
    base.free()
    keys.free()
    sorted[1].free()
    permute_lane(pairing.left, sorted[0], rows)
    permute_lane(pairing.right, sorted[0], rows)
    sorted[0].free()
}

#: The rows of two frames paired up on their key columns.
#:
#: The table goes over whichever side is shorter and the other side probes it;
#: the result is then put back into the order `how` reports in. This is the one
#: place a merge, an index join and a `concat` alignment all go through.
pub fn pair_rows(
    left_keys: list[Column],
    right_keys: list[Column],
    n_left: int,
    n_right: int,
    how: How = How.Inner
) -> Pairing {
    require_device("join two frames")
    if how == How.Cross { return cross_pairing(n_left, n_right) }
    if left_keys.len() == 0 or left_keys.len() != right_keys.len() {
        throw ValueError(
            f"a join needs the same number of key columns on each side, not {left_keys.len()} and {right_keys.len()}"
        )
    }
    for (at, key) in left_keys.enumerate() {
        if key.dtype != right_keys[at].dtype {
            throw TypeError(
                f"key {at} is {key.dtype.name()} on the left and {right_keys[at].dtype.name()} on the right"
            )
        }
    }

    let build_left = n_left <= n_right
    let build_keys = if build_left { left_keys } else { right_keys }
    let probe_keys = if build_left { right_keys } else { left_keys }
    let n_build = if build_left { n_left } else { n_right }
    let n_probe = if build_left { n_right } else { n_left }
    let keep_build = if build_left { how.keeps_left() } else { how.keeps_right() }
    let keep_probe = if build_left { how.keeps_right() } else { how.keeps_left() }

    let table = build_table(build_keys, n_build)
    defer { table.free() }

    let probe_hashes = hash_lane(probe_keys, n_probe, 0)
    let probe_buckets = Buffer(n_probe)
    kernel_for("jn_bucket_of").dispatch_async(
        [probe_hashes, probe_buckets],
        n_probe,
        [n_probe, table.buckets - 1]
    )
    probe_hashes.free()

    let counts = Buffer(n_probe)
    kernel_for("jn_probe_counts").dispatch_async(
        [probe_buckets, table.starts, table.ends, counts],
        n_probe,
        [n_probe]
    )
    let scanned = prefix_sum(counts, n_probe)
    counts.free()
    let candidates = scanned[1]

    let cand_probe = Buffer(candidates)
    let cand_build = Buffer(candidates)
    kernel_for("jn_probe_pairs").dispatch_async(
        [probe_buckets, table.starts, table.ends, table.rows_of, scanned[0], cand_probe, cand_build],
        n_probe,
        [n_probe]
    )
    probe_buckets.free()

    #: Every candidate starts a match and each key column can only clear it, so
    #: chaining the columns leaves the pairs that agree on all of them. That is
    #: also what keeps a bucket collision from becoming a join.
    let matched = Buffer(candidates)
    if candidates > 0 { matched.fill(1.0) }
    for (at, key) in probe_keys.enumerate() {
        key.equal_rows(cand_probe, build_keys[at], cand_build, candidates, matched)
    }
    cand_probe.free()

    let sizes = Buffer(n_probe)
    kernel_for("jn_emit_sizes").dispatch_async(
        [scanned[0], matched, sizes],
        n_probe,
        [n_probe, flag(keep_probe)]
    )
    let emitted = prefix_sum(sizes, n_probe)
    sizes.free()
    let probe_rows = emitted[1]

    var extra = 0
    var unmatched: Buffer? = null
    var unmatched_at: Buffer? = null
    if keep_build and n_build > 0 {
        let seen = Buffer(n_build)
        seen.fill_zero()
        kernel_for("jn_mark_matched").dispatch_async(
            [cand_build, matched, seen],
            candidates,
            [candidates]
        )
        let flags = Buffer(n_build)
        kernel_for("jn_unmatched_flags").dispatch_async([seen, flags], n_build, [n_build])
        seen.free()
        let leftover = prefix_sum(flags, n_build)
        unmatched = flags
        unmatched_at = leftover[0]
        extra = leftover[1]
    }

    let rows = probe_rows + extra
    if rows > MAX_ROWS {
        throw ValueError(
            f"this join produces {rows} rows, past the {MAX_ROWS} a position lane addresses"
        )
    }
    let probe_lane = Buffer(rows)
    let build_lane = Buffer(rows)
    kernel_for("jn_emit_pairs").dispatch_async(
        [scanned[0], matched, cand_build, emitted[0], probe_lane, build_lane],
        n_probe,
        [n_probe, flag(keep_probe)]
    )
    let leftover_flags = unmatched
    let leftover_at = unmatched_at
    if leftover_flags is not null and leftover_at is not null {
        kernel_for("jn_append_unmatched").dispatch_async(
            [leftover_flags, leftover_at, probe_lane, build_lane],
            n_build,
            [n_build, probe_rows]
        )
        leftover_flags.free()
        leftover_at.free()
    }
    scanned[0].free()
    emitted[0].free()
    cand_build.free()
    matched.free()

    let pairing = if build_left {
        Pairing(build_lane, probe_lane, rows)
    } else {
        Pairing(probe_lane, build_lane, rows)
    }
    let by_right = how == How.Right
    let needs_order = if by_right { not build_left } else { build_left }
    if needs_order { reorder_pairing(pairing, by_right, n_left, n_right) }
    return pairing
}

#: How many positions in `lane` are named by more than one row.
fn duplicate_positions(lane: Buffer, rows: int, extent: int) -> int {
    if rows == 0 or extent == 0 { return 0 }
    let counts = Buffer(extent)
    counts.fill_zero()
    kernel_for("jn_count_pairs").dispatch_async([lane, counts], rows, [rows])
    let flags = Buffer(extent)
    kernel_for("jn_over_one").dispatch_async([counts, flags], extent, [extent])
    counts.free()
    let scanned = prefix_sum(flags, extent)
    flags.free()
    scanned[0].free()
    return scanned[1]
}

#: Refuse a pairing that breaks the promise `rule` made.
#:
#: A right row paired with two left rows is a duplicate key on the *left*, and
#: the other way round — so the multiplicity the build side's chains produced
#: is read straight off the finished pairing rather than counted again.
fn check_validate(pairing: Pairing, n_left: int, n_right: int, rule: Validate) -> void {
    if rule.left_unique() and duplicate_positions(pairing.right, pairing.rows, n_right) > 0 {
        throw ValueError(f"the left keys repeat, so this merge is not {rule.name()}")
    }
    if rule.right_unique() and duplicate_positions(pairing.left, pairing.rows, n_left) > 0 {
        throw ValueError(f"the right keys repeat, so this merge is not {rule.name()}")
    }
}

#: The column names an `on` / `left_on` / `right_on` argument spells.
fn names_of(spec: any?, what: str) -> list[str]? {
    if spec is null { return null }
    if spec is str { return [spec] }
    if spec is list {
        var out: list[str] = []
        for entry in spec {
            if not entry is str { throw TypeError(f"{what} must name columns") }
            out.push(entry)
        }
        if out.len() == 0 { throw ValueError(f"{what} names no columns") }
        return out
    }
    throw TypeError(f"{what} must be a column name or a list of them")
}

#: The key of every output row, taken from whichever side supplied one.
#:
#: The two key columns are stacked into one and read with a single `take`, so
#: an outer join's right-only rows get their key without a second pass and
#: without a kernel that has to know what a dtype's lanes are. Stacking also
#: puts two dictionary columns on one table, which they have to be for the
#: codes to mean the same thing.
fn merged_key_column(left_key: Column, right_key: Column, pairing: Pairing) -> Column {
    let stacked = concat_columns([left_key, right_key])
    defer { stacked.free() }
    let lane = Buffer(pairing.rows)
    kernel_for("jn_key_positions").dispatch_async(
        [pairing.left, pairing.right, lane],
        pairing.rows,
        [pairing.rows, left_key.length]
    )
    let out = stacked.take(lane, pairing.rows)
    lane.free()
    return out
}

#: pandas' `_merge` column: which side each output row came from.
fn indicator_column(pairing: Pairing) -> Column {
    let out = Column.allocate(Dtype.Category, pairing.rows)
    out.set_levels(["left_only", "right_only", "both"])
    if pairing.rows > 0 {
        kernel_for("jn_indicator_codes").dispatch_async(
            [pairing.left, pairing.right, out.values],
            pairing.rows,
            [pairing.rows]
        )
    }
    return out
}

#: A dictionary column's codes remapped to the rank of their level.
#:
#: Sorting text means sorting the strings, and the codes are in
#: first-appearance order. The host loop is over the distinct levels, not the
#: rows; a kernel does the rows.
fn level_rank_lane(column: Column) -> Buffer {
    let levels = column.levels ?? []
    var rank: dict[str, int] = {}
    for (at, name) in sorted(levels).enumerate() { rank[name] = at }
    var mapping: list[float] = []
    for name in levels { mapping.push(float(rank.get(name, 0))) }
    let table = Buffer(mapping.len())
    if mapping.len() > 0 { table.upload(mapping) }
    let lane = Buffer(column.length)
    cached_kernel("col_remap").dispatch_async(
        [column.values, lane, table],
        column.length,
        [column.length, levels.len()]
    )
    table.free()
    return lane
}

fn wide_key_lane(column: Column, high_half: bool) -> Buffer {
    let raw = Buffer(column.length)
    kernel_for("jn_sort_key_wide").dispatch_async(
        [column.values, column.wide ?? column.values, column.valid ?? column.values, raw],
        column.length,
        [column.length, flag(column.valid is not null), flag(high_half)]
    )
    return raw
}

#: One stable radix pass of `order` against a raw key lane. Frees `order`.
fn apply_sort_pass(order: Buffer, raw_keys: Buffer, rows: int, bits: int) -> Buffer {
    let lane = Buffer(rows)
    kernel_for("jn_gather_uint").dispatch_async([raw_keys, order, lane], rows, [rows])
    let sorted = radix_order(order, lane, rows, bits)
    lane.free()
    sorted[1].free()
    order.free()
    return sorted[0]
}

#: The passes one key column contributes to a lexicographic ordering.
#:
#: A wide column is two digits, low half first, because 48 bits do not fit one
#: radix key. A dictionary column sorts on the rank of its level rather than on
#: its code. Everything else is one 32-bit order-preserving image.
fn sort_column_passes(order: Buffer, column: Column, rows: int) -> Buffer {
    var next = order
    let has_valid = flag(column.valid is not null)
    if column.dtype.is_wide() {
        let low = wide_key_lane(column, false)
        next = apply_sort_pass(next, low, rows, 24)
        low.free()
        let high = wide_key_lane(column, true)
        next = apply_sort_pass(next, high, rows, 26)
        high.free()
        return next
    }
    let raw = Buffer(rows)
    if column.dtype.is_dictionary() {
        let ranks = level_rank_lane(column)
        kernel_for("jn_sort_key_code").dispatch_async(
            [ranks, column.valid ?? column.values, raw],
            rows,
            [rows, has_valid, column.level_count()]
        )
        ranks.free()
        next = apply_sort_pass(next, raw, rows, bit_width(column.level_count()))
    } elif column.dtype == Dtype.Bool {
        kernel_for("jn_sort_key_code").dispatch_async(
            [column.values, column.valid ?? column.values, raw],
            rows,
            [rows, has_valid, 2]
        )
        next = apply_sort_pass(next, raw, rows, 2)
    } else {
        kernel_for("jn_sort_key_float").dispatch_async(
            [column.values, column.valid ?? column.values, raw],
            rows,
            [rows, has_valid]
        )
        next = apply_sort_pass(next, raw, rows, 32)
    }
    raw.free()
    return next
}

#: A stable ordering of `rows` rows by `keys`, read left to right.
#:
#: Least significant digit last, so the columns are sorted back to front and
#: each pass's stability carries the earlier columns' decision. Missing values
#: take the largest key in every image, which puts them last as pandas does.
pub fn key_order(keys: list[Column], rows: int) -> Buffer {
    var order = iota(rows)
    if rows <= 1 { return order }
    var at = keys.len() - 1
    while at >= 0 {
        if keys[at].length != rows {
            throw ValueError(f"a key of {keys[at].length} rows cannot order {rows}")
        }
        order = sort_column_passes(order, keys[at], rows)
        at -= 1
    }
    return order
}

#: The output rows put into key order, position lanes and keys together.
#:
#: The lanes move in place and the keys are re-taken rather than re-derived, so
#: a caller cannot end up with a frame whose key column disagrees with the rows
#: it labels. The pass is stable, which is what lets an ordered merge append
#: its right-only rows after the matched ones and still have them land between
#: the right neighbours once this runs.
fn sort_pairing(pairing: Pairing, keys: list[Column]) -> list[Column] {
    if pairing.rows <= 1 or keys.len() == 0 { return keys }
    let perm = key_order(keys, pairing.rows)
    permute_lane(pairing.left, perm, pairing.rows)
    permute_lane(pairing.right, perm, pairing.rows)
    var out: list[Column] = []
    for key in keys { out.push(key.take(perm, pairing.rows)) }
    for key in keys { key.free() }
    perm.free()
    return out
}

#: Two dictionary key columns put on one table, or the pair unchanged.
fn unified_pair(left_key: Column, right_key: Column, owned: list[Column]) -> (Column, Column) {
    if not left_key.dtype.is_dictionary() { return (left_key, right_key) }
    let pair = unify_levels([left_key, right_key])
    owned.push(pair[0])
    owned.push(pair[1])
    return (pair[0], pair[1])
}

#: Rows of `left` paired with rows of `right` on their keys.
#:
#: pandas' `merge`, with pandas' defaults. `on` names key columns present on
#: both sides and puts one merged key column in the result; `left_on` and
#: `right_on` name them separately and keep both; `left_index` and
#: `right_index` use the row labels instead. Naming none of them falls back to
#: the columns the two frames share, which is what pandas does.
#:
#: The result carries a fresh range index unless both sides joined on their
#: index, where the merged labels are kept.
pub fn merge(
    left: DataFrame,
    right: DataFrame,
    how: How = How.Inner,
    on: any? = null,
    left_on: any? = null,
    right_on: any? = null,
    left_index: bool = false,
    right_index: bool = false,
    suffixes: tuple[str, str] = ("_x", "_y"),
    indicator: bool = false,
    indicator_name: str = "_merge",
    validate: Validate? = null,
    sort: bool = false
) -> DataFrame {
    require_device("merge two frames")
    var owned: list[Column] = []
    defer {
        for column in owned { column.free() }
    }

    var left_names = names_of(on, "on")
    var right_names = left_names
    if how == How.Cross {
        let keyed = on is not null or left_on is not null or right_on is not null
        if keyed or left_index or right_index {
            throw ValueError("a cross join takes no keys")
        }
        left_names = null
        right_names = null
    } elif on is not null {
        if left_on is not null or right_on is not null or left_index or right_index {
            throw ValueError("on cannot be combined with left_on, right_on or an index key")
        }
    } else {
        left_names = if left_index { null } else { names_of(left_on, "left_on") }
        right_names = if right_index { null } else { names_of(right_on, "right_on") }
        if left_names is null and not left_index and right_names is null and not right_index {
            var shared: list[str] = []
            for name in left.names {
                if right.has(name) { shared.push(name) }
            }
            if shared.len() == 0 {
                throw ValueError("the two frames share no column to merge on")
            }
            left_names = shared
            right_names = shared
        }
        if left_names is null and not left_index {
            throw ValueError("left_on or left_index has to say what the left side keys on")
        }
        if right_names is null and not right_index {
            throw ValueError("right_on or right_index has to say what the right side keys on")
        }
    }

    var left_keys: list[Column] = []
    var right_keys: list[Column] = []
    if how != How.Cross {
        if left_index {
            for level in left.index.materialize() {
                owned.push(level)
                left_keys.push(level)
            }
        } else {
            for name in left_names ?? [] { left_keys.push(left.column(name)) }
        }
        if right_index {
            for level in right.index.materialize() {
                owned.push(level)
                right_keys.push(level)
            }
        } else {
            for name in right_names ?? [] { right_keys.push(right.column(name)) }
        }
        if left_keys.len() != right_keys.len() {
            throw ValueError(
                f"{left_keys.len()} left keys cannot pair with {right_keys.len()} right ones"
            )
        }
        var unified_left: list[Column] = []
        var unified_right: list[Column] = []
        for (at, key) in left_keys.enumerate() {
            let pair = unified_pair(key, right_keys[at], owned)
            unified_left.push(pair[0])
            unified_right.push(pair[1])
        }
        left_keys = unified_left
        right_keys = unified_right
    }

    let pairing = pair_rows(left_keys, right_keys, left.len(), right.len(), how)
    defer { pairing.free() }
    let rule = validate
    if rule is not null { check_validate(pairing, left.len(), right.len(), rule) }

    var merged_keys: list[Column] = []
    for (at, key) in left_keys.enumerate() {
        merged_keys.push(merged_key_column(key, right_keys[at], pairing))
    }
    if sort { merged_keys = sort_pairing(pairing, merged_keys) }

    #: A key named the same on both sides collapses to one column, as pandas
    #: does; keys named differently stay two columns and are carried by the
    #: ordinary column loop below.
    var merged_at: dict[str, int] = {}
    let named_left = left_names
    let named_right = right_names
    if named_left is not null and named_right is not null {
        for (at, name) in named_left.enumerate() {
            if name == named_right[at] { merged_at[name] = at }
        }
    }

    var clashing: dict[str, int] = {}
    for name in right.names {
        if left.has(name) and merged_at.get(name, -1) < 0 { clashing[name] = 1 }
    }
    if clashing.len() > 0 and suffixes[0].len() == 0 and suffixes[1].len() == 0 {
        throw ValueError("the two frames share a column name and neither suffix would rename it")
    }

    let rows = pairing.rows
    let left_missing = how.keeps_right()
    let right_missing = how.keeps_left()
    var out_names: list[str] = []
    var out_columns: list[Column] = []
    for (at, name) in left.names.enumerate() {
        let merged = merged_at.get(name, -1)
        if merged >= 0 {
            out_names.push(name)
            out_columns.push(merged_keys[merged].copy())
            continue
        }
        out_names.push(if clashing.get(name, 0) > 0 { f"{name}{suffixes[0]}" } else { name })
        out_columns.push(left.columns[at].take(pairing.left, rows, allow_missing: left_missing))
    }
    for (at, name) in right.names.enumerate() {
        if merged_at.get(name, -1) >= 0 { continue }
        out_names.push(if clashing.get(name, 0) > 0 { f"{name}{suffixes[1]}" } else { name })
        out_columns.push(right.columns[at].take(pairing.right, rows, allow_missing: right_missing))
    }
    if indicator {
        out_names.push(indicator_name)
        out_columns.push(indicator_column(pairing))
    }

    #: pandas passes an index on when one was used as a key: both sides keyed
    #: on their index gives the merged labels, and one side keyed on its index
    #: gives the other side's labels, which are the ones the pairing left
    #: intact. Two frames keyed on columns get a fresh range index, because
    #: neither set of labels survived.
    var index = Index.range(rows)
    if left_index and right_index and merged_keys.len() > 0 {
        index = Index.of(merged_keys.map(|level| level.copy()), [name for name in left.index.names])
    } elif left_index {
        index = right.index.take(pairing.right, rows, allow_missing: right_missing)
    } elif right_index {
        index = left.index.take(pairing.left, rows, allow_missing: left_missing)
    }
    for key in merged_keys { key.free() }
    return DataFrame(out_names, out_columns, index: index)
}

#: `left` joined to `right` on the row labels, pandas' `DataFrame.join`.
#:
#: With no `on` both sides key on their index; with `on` the named left columns
#: key against the right's index. Either way the left frame's labels are the
#: ones the result carries, which is the whole point of an index join.
pub fn join(
    left: DataFrame,
    right: DataFrame,
    on: any? = null,
    how: How = How.Left,
    lsuffix: str = "",
    rsuffix: str = ""
) -> DataFrame {
    return merge(
        left,
        right,
        how: how,
        left_on: on,
        left_index: on is null,
        right_index: true,
        suffixes: (lsuffix, rsuffix)
    )
}

#: Every missing row filled from the last present row before it.
#:
#: The scan over the validity lane already counts the present rows before each
#: row, so compacting their positions and reading that count back out of the
#: compacted lane is a forward fill without a serial walk and without a scan
#: kernel of its own.
pub fn fill_forward(column: Column) -> Column {
    let mask = column.valid
    if mask is null or column.length == 0 { return column.copy() }
    let scanned = prefix_sum(mask, column.length)
    let compact = Buffer(scanned[1])
    kernel_for("jn_compact_positions").dispatch_async(
        [mask, scanned[0], compact],
        column.length,
        [column.length]
    )
    let lane = Buffer(column.length)
    kernel_for("jn_fill_positions").dispatch_async(
        [mask, scanned[0], compact, lane],
        column.length,
        [column.length]
    )
    scanned[0].free()
    compact.free()
    let out = column.take(lane, column.length, allow_missing: true)
    lane.free()
    return out
}

fn range_like(index: Index, other: Index) -> bool {
    if not index.is_range() or not other.is_range() { return false }
    return index.length == other.length and index.start == other.start and index.step == other.step
}

#: The column names `axis: 0` stacking produces, and where each part has them.
fn concat_names(objects: list[DataFrame], inner: bool, sort: bool) -> list[str] {
    var out: list[str] = []
    var seen: dict[str, int] = {}
    for frame in objects {
        for name in frame.names {
            if seen.get(name, -1) >= 0 { continue }
            if inner {
                var everywhere = true
                for other in objects {
                    if not other.has(name) { everywhere = false }
                }
                if not everywhere { continue }
            }
            seen[name] = out.len()
            out.push(name)
        }
    }
    return if sort { sorted(out) } else { out }
}

#: The dtype the parts agree on for `name`, and a refusal when they do not.
fn concat_dtype(objects: list[DataFrame], name: str) -> Dtype {
    var found: Dtype? = null
    for frame in objects {
        if not frame.has(name) { continue }
        let dtype = frame.column(name).dtype
        let earlier = found
        if earlier is not null and earlier != dtype {
            throw TypeError(
                f"column {name} is {earlier.name()} in one part and {dtype.name()} in another"
            )
        }
        found = dtype
    }
    let dtype = found
    if dtype is null { throw KeyError(f"no part has a column named {name}") }
    return dtype
}

#: The index levels of the parts stacked, with `keys` as an outer level.
fn concat_index(
    objects: list[DataFrame],
    keys: list[str]?,
    names: list[str?]?,
    total: int
) -> Index {
    let levels = objects[0].index.nlevels()
    for frame in objects {
        if frame.index.nlevels() != levels {
            throw ValueError(
                f"an index of {frame.index.nlevels()} levels cannot stack onto one of {levels}"
            )
        }
    }
    var stacked: list[Column] = []
    var labels: list[str?] = []
    let tags = keys
    if tags is not null {
        if tags.len() != objects.len() {
            throw ValueError(f"{tags.len()} keys cannot label {objects.len()} parts")
        }
        var parts: list[Column] = []
        for (at, frame) in objects.enumerate() {
            parts.push(Column.full(Dtype.Str, frame.len(), Value.of_str(tags[at])))
        }
        stacked.push(concat_columns(parts))
        for part in parts { part.free() }
        labels.push(null)
    }
    for level in 0..levels {
        var parts: list[Column] = []
        var owned: list[Column] = []
        for frame in objects {
            let materialised = frame.index.materialize()
            for extra in materialised { owned.push(extra) }
            parts.push(materialised[level])
        }
        stacked.push(concat_columns(parts))
        for part in owned { part.free() }
        labels.push(objects[0].index.names[level])
    }
    if total != stacked[0].length {
        throw ValueError(f"the stacked index has {stacked[0].length} labels for {total} rows")
    }
    let named = names
    if named is not null {
        if named.len() != labels.len() {
            throw ValueError(f"{named.len()} names cannot name {labels.len()} index levels")
        }
        labels = [name for name in named]
    }
    return Index.of(stacked, labels)
}

fn concat_rows(
    objects: list[DataFrame],
    inner: bool,
    keys: list[str]?,
    names: list[str?]?,
    ignore_index: bool,
    sort: bool
) -> DataFrame {
    var total = 0
    for frame in objects { total += frame.len() }
    let wanted = concat_names(objects, inner, sort)
    var columns: list[Column] = []
    for name in wanted {
        let dtype = concat_dtype(objects, name)
        var parts: list[Column] = []
        var filler: list[Column] = []
        for frame in objects {
            if frame.has(name) {
                parts.push(frame.column(name))
                continue
            }
            #: A part with no such column contributes missing rows, which is
            #: what an outer concatenation means and what pandas fills in.
            let gap = Column.nulls(dtype, frame.len())
            filler.push(gap)
            parts.push(gap)
        }
        columns.push(concat_columns(parts))
        for gap in filler { gap.free() }
    }
    let index = if ignore_index {
        Index.range(total)
    } else {
        concat_index(objects, keys, names, total)
    }
    return DataFrame(wanted, columns, index: index)
}

fn concat_columns_axis(
    objects: list[DataFrame],
    inner: bool,
    keys: list[str]?,
    names: list[str?]?,
    ignore_index: bool,
    sort: bool
) -> DataFrame {
    let how = if inner { How.Inner } else { How.Outer }
    var lanes: list[Buffer] = []
    var labels: list[Column] = objects[0].index.materialize()
    var rows = objects[0].len()
    lanes.push(iota(rows))

    for at in 1..objects.len() {
        let frame = objects[at]
        #: Two frames already labelled 0..n-1 the same way need no pairing, and
        #: side-by-side concatenation of default-indexed frames is the common
        #: case by a wide margin.
        if range_like(objects[0].index, frame.index) and rows == frame.len() {
            lanes.push(iota(rows))
            continue
        }
        let next = frame.index.materialize()
        let pairing = pair_rows(labels, next, rows, frame.len(), how)
        var merged: list[Column] = []
        for (level, column) in labels.enumerate() {
            merged.push(merged_key_column(column, next[level], pairing))
        }
        var updated: list[Buffer] = []
        for lane in lanes {
            let mapped = Buffer(pairing.rows)
            kernel_for("jn_map_positions").dispatch_async(
                [pairing.left, lane, mapped],
                pairing.rows,
                [pairing.rows]
            )
            updated.push(mapped)
            lane.free()
        }
        let taken = Buffer(pairing.rows)
        copy_lane(pairing.right, taken, pairing.rows)
        updated.push(taken)
        for column in labels { column.free() }
        for column in next { column.free() }
        pairing.free()
        lanes = updated
        labels = merged
        rows = pairing.rows
    }

    if sort and rows > 1 and labels.len() > 0 {
        let perm = key_order(labels, rows)
        var ordered: list[Column] = []
        for column in labels { ordered.push(column.take(perm, rows)) }
        for column in labels { column.free() }
        labels = ordered
        for lane in lanes { permute_lane(lane, perm, rows) }
        perm.free()
    }

    let tags = keys
    if tags is not null and tags.len() != objects.len() {
        throw ValueError(f"{tags.len()} keys cannot label {objects.len()} parts")
    }
    var out_names: list[str] = []
    var out_columns: list[Column] = []
    for (at, frame) in objects.enumerate() {
        for (position, name) in frame.names.enumerate() {
            #: A flat name list cannot hold pandas' two-level column index, so
            #: a key becomes a prefix instead of an outer level.
            let labelled = if tags is null { name } else { f"{tags[at]}_{name}" }
            out_names.push(labelled)
            out_columns.push(frame.columns[position].take(lanes[at], rows, allow_missing: true))
        }
    }
    for lane in lanes { lane.free() }
    let index = if ignore_index {
        for column in labels {
            column.free()
        }
        Index.range(rows)
    } else {
        let named = names
        var level_names: list[str?] = [name for name in objects[0].index.names]
        if named is not null {
            level_names = [name for name in named]
        }
        Index.of(labels, level_names)
    }
    return DataFrame(out_names, out_columns, index: index)
}

#: The parts combined, pandas' `concat`.
#:
#: `axis: 0` stacks rows, matching columns by name — `join: How.Outer` keeps
#: every name and fills the gaps with nulls, `How.Inner` keeps only the names
#: every part has. `axis: 1` puts the parts side by side, aligning them on
#: their row labels, which is the same pairing a merge does with the index as
#: the key.
#:
#: `keys` labels the parts: an outer index level when stacking rows, a prefix
#: on the column names when stacking columns, because a frame's column names
#: are a flat list here and cannot carry pandas' second level.
pub fn concat(
    objects: list[DataFrame],
    axis: int = 0,
    join: How = How.Outer,
    keys: list[str]? = null,
    names: list[str?]? = null,
    ignore_index: bool = false,
    sort: bool = false
) -> DataFrame {
    if objects.len() == 0 { throw ValueError("a concatenation needs at least one frame") }
    if join != How.Outer and join != How.Inner {
        throw ValueError(f"concat joins outer or inner, not {join.name()}")
    }
    let inner = join == How.Inner
    if axis == 0 { return concat_rows(objects, inner, keys, names, ignore_index, sort) }
    if axis == 1 {
        return concat_columns_axis(objects, inner, keys, names, ignore_index, sort)
    }
    throw ValueError(f"a frame has axes 0 and 1, not {axis}")
}

#: `other`'s rows added under `frame`'s, pandas' `DataFrame.append`.
pub fn append(frame: DataFrame, other: DataFrame, ignore_index: bool = false) -> DataFrame {
    return concat([frame, other], axis: 0, ignore_index: ignore_index)
}

#: Whether a key lane never steps backwards and never goes missing.
#:
#: One dispatch and one scan, which is what makes it cheap enough to check on
#: every ordered merge rather than take the caller's word for it.
fn is_ascending(column: Column) -> bool {
    if column.length <= 1 { return true }
    let flags = Buffer(column.length)
    kernel_for("jn_key_breaks").dispatch_async(
        [column.values, column.wide ?? column.values, column.valid ?? column.values, flags],
        column.length,
        [column.length, flag(column.dtype.is_wide()), flag(column.valid is not null)]
    )
    let scanned = prefix_sum(flags, column.length)
    flags.free()
    scanned[0].free()
    return scanned[1] == 0
}

#: Refuse a key the two ordered merges cannot bisect.
fn require_searchable(column: Column, side: str, name: str) -> void {
    if column.dtype.is_dictionary() {
        throw TypeError(
            f"an ordered merge bisects the key lane, and {name} is {column.dtype.name()}, whose codes are in first-appearance order rather than the text's — merge(sort: true) is the call that joins text keys"
        )
    }
    if column.null_count() > 0 {
        throw ValueError(
            f"the {side} key {name} has missing rows, which have no place in a key order"
        )
    }
}

#: A side's key in ascending order, with the permutation that put it there.
#:
#: Already ascending is the case these merges are written for, and it costs the
#: monotonicity scan and an `iota`. Anything else pays one radix sort, which is
#: still less than refusing and making the caller reorder the whole frame.
fn ascending_side(column: Column) -> (Column, Buffer) {
    if is_ascending(column) { return (column.copy(), iota(column.length)) }
    let perm = key_order([column], column.length)
    return (column.take(perm, column.length), perm)
}

#: A tolerance as the scalars `jn_asof_search` compares in: the two halves of a
#: wide key's limit, or the float bits of everything else's.
#:
#: A wide limit stays integral all the way to the device because a `Datetime`
#: gap is milliseconds and passing it as a float would stop being exact after
#: four and a half hours.
fn tolerance_scalars(column: Column, limit: any) -> (int, int, int) {
    if column.dtype.is_wide() {
        let span = whole_units(limit, column)
        if span < 0 { throw ValueError("a tolerance cannot be negative") }
        let halves = wide_pack(span)
        return (int(halves[0]), int(halves[1]), 0)
    }
    let span = real_units(limit)
    if span < 0.0 { throw ValueError("a tolerance cannot be negative") }
    return (0, 0, f32_bits(span))
}

fn whole_units(limit: any, column: Column) -> int {
    if limit is int { return limit }
    if limit is Value { return limit.to_int() }
    throw TypeError(
        f"a {column.dtype.name()} tolerance is counted in whole units of the column, so it has to be an int"
    )
}

fn real_units(limit: any) -> float {
    if limit is float { return limit }
    if limit is int { return float(limit) }
    if limit is Value { return limit.to_float() }
    throw TypeError("a tolerance has to be a number")
}

#: The spans an asof search with no `by` uses: the whole right side for every
#: left row, and the identity permutation that leaves it where it is.
fn whole_side_spans(n_left: int, n_right: int) -> (Buffer, Buffer, Buffer) {
    let begin = Buffer(n_left)
    let end = Buffer(n_left)
    begin.fill(0.0)
    end.fill(float(n_right))
    return (iota(n_right), begin, end)
}

#: Where each left row may search: the stretch of the group-ordered right side
#: carrying its `by` group, and the ordering that made those stretches.
#:
#: The two sides are factorized together, over one stacked column per `by`
#: level, so a code means the same thing on both — which two independent hashes
#: would only manage probabilistically, and a wrong group is a wrong number
#: rather than a slow one. Dense codes then index the bounds table directly, so
#: there is no probe to write, and the sort that gathers the right side into
#: group runs is `bit_width(groups)` radix passes rather than the sixty-four a
#: pair of hash lanes would have cost.
#:
#: That sort is stable and the right side is already in key order, so each
#: group's rows stay in key order inside its run and the bisection is searching
#: sorted data.
fn group_spans(left: DataFrame, right: DataFrame, names: list[str]) -> (Buffer, Buffer, Buffer) {
    let n_left = left.len()
    let n_right = right.len()
    var stacked: list[Column] = []
    for name in names {
        let mine = left.column(name)
        let theirs = right.column(name)
        if mine.dtype != theirs.dtype {
            throw TypeError(
                f"by column {name} is {mine.dtype.name()} on the left and {theirs.dtype.name()} on the right"
            )
        }
        stacked.push(concat_columns([mine, theirs]))
    }
    let grouped = factorize(stacked)
    for column in stacked { column.free() }
    for column in grouped[1] { column.free() }
    let codes = grouped[0]
    let groups = grouped[2]

    let left_codes = Buffer(n_left)
    kernel_for("jn_code_lane").dispatch_async([codes, left_codes], n_left, [n_left, 0])
    let right_codes = Buffer(n_right)
    kernel_for("jn_code_lane").dispatch_async([codes, right_codes], n_right, [n_right, n_left])
    codes.free()

    let base = iota(n_right)
    let width = if groups > 1 { bit_width(groups - 1) } else { 0 }
    let sorted = radix_order(base, right_codes, n_right, width)
    base.free()
    right_codes.free()

    let starts = Buffer(groups)
    let ends = Buffer(groups)
    starts.fill_zero()
    ends.fill_zero()
    kernel_for("jn_bucket_bounds").dispatch_async([sorted[1], starts, ends], n_right, [n_right])
    sorted[1].free()

    let begin = Buffer(n_left)
    let end = Buffer(n_left)
    kernel_for("jn_group_span").dispatch_async(
        [left_codes, starts, ends, begin, end],
        n_left,
        [n_left]
    )
    left_codes.free()
    starts.free()
    ends.free()
    return (sorted[0], begin, end)
}

#: Every left row paired with the nearest right row, pandas' `merge_asof`.
#:
#: An asof merge is a left join on an inequality, which no hash table can
#: answer: two keys being close is not two keys being equal, and a bucket says
#: nothing at all about a neighbour. Both sides being sorted on the key — what
#: "asof" means, and what pandas requires — makes it one bisection per left row
#: over the right key lane instead, `log2(n)` reads where a nested loop would
#: have walked the whole side.
#:
#: `by` restricts each search to the rows sharing a group. `tolerance` drops a
#: match further than that from the left key, counted in whole units of a wide
#: column — milliseconds for a `Datetime` — and inclusive at the limit, as
#: pandas' is.
#:
#: The result carries the left frame's rows and labels unchanged, because every
#: left row appears exactly once whether it matched or not.
pub fn merge_asof(
    left: DataFrame,
    right: DataFrame,
    on: str,
    by: any? = null,
    direction: AsofDirection = AsofDirection.Backward,
    tolerance: any? = null,
    suffixes: tuple[str, str] = ("_x", "_y")
) -> DataFrame {
    require_device("merge two frames as of a key")
    let left_key = left.column(on)
    let right_key = right.column(on)
    if left_key.dtype != right_key.dtype {
        throw TypeError(
            f"{on} is {left_key.dtype.name()} on the left and {right_key.dtype.name()} on the right"
        )
    }
    require_searchable(left_key, "left", on)
    require_searchable(right_key, "right", on)
    if not is_ascending(left_key) {
        throw ValueError(f"merge_asof needs the left frame sorted on {on}")
    }
    if not is_ascending(right_key) {
        throw ValueError(f"merge_asof needs the right frame sorted on {on}")
    }

    let n_left = left.len()
    let n_right = right.len()
    let by_names = names_of(by, "by") ?? []
    let spans = if by_names.len() == 0 {
        whole_side_spans(n_left, n_right)
    } else {
        group_spans(left, right, by_names)
    }
    let perm = spans[0]
    let begin = spans[1]
    let end = spans[2]
    defer {
        perm.free()
        begin.free()
        end.free()
    }

    let sorted_key = right_key.take(perm, n_right)
    defer { sorted_key.free() }
    var tol_lo = 0
    var tol_hi = 0
    var tol_bits = 0
    let limit = tolerance
    if limit is not null {
        let packed = tolerance_scalars(left_key, limit)
        tol_lo = packed[0]
        tol_hi = packed[1]
        tol_bits = packed[2]
    }
    let found = Buffer(n_left)
    kernel_for("jn_asof_search").dispatch_async(
        [
            left_key.values,
            left_key.wide ?? left_key.values,
            sorted_key.values,
            sorted_key.wide ?? sorted_key.values,
            begin,
            end,
            found,
        ],
        n_left,
        [
            n_left,
            flag(left_key.dtype.is_wide()),
            direction_code(direction),
            flag(limit is not null),
            tol_lo,
            tol_hi,
            tol_bits,
        ]
    )
    let right_lane = Buffer(n_left)
    kernel_for("jn_map_positions").dispatch_async([found, perm, right_lane], n_left, [n_left])
    found.free()
    defer { right_lane.free() }

    #: The right side's copy of the key and of every `by` column is dropped:
    #: they agree with the left's by construction, and pandas drops them too.
    var skipped: dict[str, int] = {}
    skipped[on] = 1
    for name in by_names { skipped[name] = 1 }
    var clashing: dict[str, int] = {}
    for name in right.names {
        if skipped.get(name, 0) > 0 { continue }
        if left.has(name) { clashing[name] = 1 }
    }
    if clashing.len() > 0 and suffixes[0].len() == 0 and suffixes[1].len() == 0 {
        throw ValueError("the two frames share a column name and neither suffix would rename it")
    }

    var out_names: list[str] = []
    var out_columns: list[Column] = []
    for (at, name) in left.names.enumerate() {
        out_names.push(if clashing.get(name, 0) > 0 { f"{name}{suffixes[0]}" } else { name })
        out_columns.push(left.columns[at].copy())
    }
    for (at, name) in right.names.enumerate() {
        if skipped.get(name, 0) > 0 { continue }
        out_names.push(if clashing.get(name, 0) > 0 { f"{name}{suffixes[1]}" } else { name })
        out_columns.push(right.columns[at].take(right_lane, n_left, allow_missing: true))
    }
    return DataFrame(out_names, out_columns, index: left.index.copy())
}

#: The one column name an ordered merge keys on, from a pandas-shaped argument.
fn single_key(spec: any?, what: str) -> str? {
    let names = names_of(spec, what)
    if names is null { return null }
    if names.len() != 1 {
        throw ValueError(
            f"an ordered merge bisects one key lane, so {what} names one column, not {names.len()}"
        )
    }
    return names[0]
}

#: The key column each side of an ordered merge uses.
fn ordered_keys(
    left: DataFrame,
    right: DataFrame,
    on: any?,
    left_on: any?,
    right_on: any?
) -> (str, str) {
    let both = single_key(on, "on")
    if both is not null {
        if left_on is not null or right_on is not null {
            throw ValueError("on cannot be combined with left_on or right_on")
        }
        return (both, both)
    }
    let mine = single_key(left_on, "left_on")
    let theirs = single_key(right_on, "right_on")
    if mine is not null and theirs is not null { return (mine, theirs) }
    if mine is not null or theirs is not null {
        throw ValueError("left_on and right_on name a key each, or neither of them does")
    }
    var shared: list[str] = []
    for name in left.names {
        if right.has(name) { shared.push(name) }
    }
    if shared.len() != 1 {
        throw ValueError(
            f"an ordered merge keys on one column and the two frames share {shared.len()}"
        )
    }
    return (shared[0], shared[0])
}

#: A taken column as the fill method asks for it. Consumes its argument.
fn carried(column: Column, fill: bool) -> Column {
    if not fill { return column }
    let filled = fill_forward(column)
    column.free()
    return filled
}

#: The rows of two ordered frames interleaved on their key, pandas'
#: `merge_ordered`.
#:
#: The same shape as an outer `merge` with `sort: true`, but reached by
#: bisection rather than by a hash: both sides are put in key order once and
#: each left row's matches are the half-open stretch of the right side holding
#: its key, which two binary searches find. That is `log2(n)` reads a row
#: against a bucket read, and it is worth taking because the sides are usually
#: already sorted — an ordered merge is a time series call — so the ordering
#: costs a scan that finds nothing to do.
#:
#: `fill_method: "ffill"` carries the last present value down every result
#: column, which is what makes an outer ordered merge of two irregular series
#: into one regular one.
pub fn merge_ordered(
    left: DataFrame,
    right: DataFrame,
    on: any? = null,
    left_on: any? = null,
    right_on: any? = null,
    fill_method: str? = null,
    suffixes: tuple[str, str] = ("_x", "_y"),
    how: How = How.Outer
) -> DataFrame {
    require_device("merge two frames in key order")
    if how == How.Cross {
        throw ValueError("an ordered merge pairs rows on a key, so it cannot be a cross join")
    }
    let method = fill_method
    if method is not null and method != "ffill" and method != "pad" {
        throw ValueError(f"an ordered merge fills forward, not {method}")
    }
    let fill = method is not null
    let keyed = ordered_keys(left, right, on, left_on, right_on)
    let left_name = keyed[0]
    let right_name = keyed[1]
    let left_key = left.column(left_name)
    let right_key = right.column(right_name)
    if left_key.dtype != right_key.dtype {
        throw TypeError(
            f"{left_name} is {left_key.dtype.name()} and {right_name} is {right_key.dtype.name()}"
        )
    }
    require_searchable(left_key, "left", left_name)
    require_searchable(right_key, "right", right_name)

    let n_left = left.len()
    let n_right = right.len()
    let left_side = ascending_side(left_key)
    let right_side = ascending_side(right_key)
    defer {
        left_side[0].free()
        left_side[1].free()
        right_side[0].free()
        right_side[1].free()
    }
    let sorted_left = left_side[0]
    let sorted_right = right_side[0]
    let wide = flag(left_key.dtype.is_wide())

    let lower = Buffer(n_left)
    let upper = Buffer(n_left)
    kernel_for("jn_equal_range").dispatch_async(
        [
            sorted_left.values,
            sorted_left.wide ?? sorted_left.values,
            sorted_right.values,
            sorted_right.wide ?? sorted_right.values,
            lower,
            upper,
        ],
        n_left,
        [n_left, n_right, wide]
    )
    let sizes = Buffer(n_left)
    kernel_for("jn_span_sizes").dispatch_async(
        [lower, upper, sizes],
        n_left,
        [n_left, flag(how.keeps_left())]
    )
    let emitted = prefix_sum(sizes, n_left)
    sizes.free()
    let block = emitted[1]

    #: A right row is unmatched exactly when its own stretch of the left side
    #: is empty, which is the same two searches run the other way round.
    var extra = 0
    var absent: Buffer? = null
    var absent_at: Buffer? = null
    if how.keeps_right() and n_right > 0 {
        let back_lower = Buffer(n_right)
        let back_upper = Buffer(n_right)
        kernel_for("jn_equal_range").dispatch_async(
            [
                sorted_right.values,
                sorted_right.wide ?? sorted_right.values,
                sorted_left.values,
                sorted_left.wide ?? sorted_left.values,
                back_lower,
                back_upper,
            ],
            n_right,
            [n_right, n_left, wide]
        )
        let spans = Buffer(n_right)
        kernel_for("jn_span_sizes").dispatch_async(
            [back_lower, back_upper, spans],
            n_right,
            [n_right, 0]
        )
        back_lower.free()
        back_upper.free()
        let holes = Buffer(n_right)
        kernel_for("jn_not_flags").dispatch_async([spans, holes], n_right, [n_right])
        spans.free()
        let scanned = prefix_sum(holes, n_right)
        absent = holes
        absent_at = scanned[0]
        extra = scanned[1]
    }

    let rows = block + extra
    if rows > MAX_ROWS {
        throw ValueError(
            f"this merge produces {rows} rows, past the {MAX_ROWS} a position lane addresses"
        )
    }
    let left_sorted_lane = Buffer(rows)
    let right_sorted_lane = Buffer(rows)
    kernel_for("jn_span_pairs").dispatch_async(
        [lower, upper, emitted[0], left_sorted_lane, right_sorted_lane],
        n_left,
        [n_left, flag(how.keeps_left())]
    )
    lower.free()
    upper.free()
    emitted[0].free()
    let holes = absent
    let holes_at = absent_at
    if holes is not null and holes_at is not null {
        kernel_for("jn_append_unmatched").dispatch_async(
            [holes, holes_at, left_sorted_lane, right_sorted_lane],
            n_right,
            [n_right, block]
        )
        holes.free()
        holes_at.free()
    }

    #: Both lanes hold positions into the sorted sides, so both are put back
    #: into frame coordinates by the same pass — which is why the pair kernels
    #: above do not take a mapping of their own.
    let left_lane = Buffer(rows)
    kernel_for("jn_map_positions").dispatch_async(
        [left_sorted_lane, left_side[1], left_lane],
        rows,
        [rows]
    )
    let right_lane = Buffer(rows)
    kernel_for("jn_map_positions").dispatch_async(
        [right_sorted_lane, right_side[1], right_lane],
        rows,
        [rows]
    )
    left_sorted_lane.free()
    right_sorted_lane.free()
    let pairing = Pairing(left_lane, right_lane, rows)
    defer { pairing.free() }

    let ordered = sort_pairing(pairing, [merged_key_column(left_key, right_key, pairing)])
    let merged_key = ordered[0]
    defer { merged_key.free() }

    let shared_name = left_name == right_name
    var clashing: dict[str, int] = {}
    for name in right.names {
        if shared_name and name == right_name { continue }
        if left.has(name) { clashing[name] = 1 }
    }
    if clashing.len() > 0 and suffixes[0].len() == 0 and suffixes[1].len() == 0 {
        throw ValueError("the two frames share a column name and neither suffix would rename it")
    }

    var out_names: list[str] = []
    var out_columns: list[Column] = []
    for (at, name) in left.names.enumerate() {
        if shared_name and name == left_name {
            out_names.push(name)
            out_columns.push(merged_key.copy())
            continue
        }
        out_names.push(if clashing.get(name, 0) > 0 { f"{name}{suffixes[0]}" } else { name })
        out_columns.push(
            carried(left.columns[at].take(pairing.left, rows, allow_missing: true), fill)
        )
    }
    for (at, name) in right.names.enumerate() {
        if shared_name and name == right_name { continue }
        out_names.push(if clashing.get(name, 0) > 0 { f"{name}{suffixes[1]}" } else { name })
        out_columns.push(
            carried(right.columns[at].take(pairing.right, rows, allow_missing: true), fill)
        )
    }
    return DataFrame(out_names, out_columns, index: Index.range(rows))
}

#: How many slots of a 0/1 lane are clear.
fn lane_zeros(lane: Buffer, count: int) -> int {
    if count == 0 { return 0 }
    let flags = Buffer(count)
    kernel_for("jn_not_flags").dispatch_async([lane, flags], count, [count])
    let scanned = prefix_sum(flags, count)
    flags.free()
    scanned[0].free()
    return scanned[1]
}

#: The rows a comparison keeps: every row when the shape is kept, otherwise the
#: rows where at least one column disagreed.
fn compared_rows(all_same: Buffer, rows: int, keep_shape: bool) -> (Buffer, int) {
    if keep_shape { return (iota(rows), rows) }
    let flags = Buffer(rows)
    kernel_for("jn_not_flags").dispatch_async([all_same, flags], rows, [rows])
    let scanned = prefix_sum(flags, rows)
    let lane = Buffer(scanned[1])
    kernel_for("jn_compact_positions").dispatch_async([flags, scanned[0], lane], rows, [rows])
    flags.free()
    scanned[0].free()
    return (lane, scanned[1])
}

#: Where two frames disagree, pandas' `DataFrame.compare`.
#:
#: The two frames have to carry the same columns in the same order over the
#: same number of rows: a difference between differently labelled frames is a
#: merge, and asking for one here would hide the mistake. Each surviving column
#: contributes `name_self` and `name_other`, because a flat name list cannot
#: hold pandas' second column level.
#:
#: By default only the differing rows and columns come back and an agreeing
#: cell inside a surviving row reads as null — the shape of the answer is what
#: makes it readable. `keep_shape` keeps every row and column, `keep_equal`
#: keeps the agreeing values instead of nulling them.
#:
#: Equality is `Column.equal_rows`, the same test a join confirms a bucket
#: with, so two frames compare equal exactly when a merge on every column would
#: have paired their rows — nulls included, which agree with each other.
pub fn compare(
    a: DataFrame,
    b: DataFrame,
    keep_shape: bool = false,
    keep_equal: bool = false
) -> DataFrame {
    require_device("compare two frames")
    if a.len() != b.len() {
        throw ValueError(f"a frame of {a.len()} rows cannot be compared with one of {b.len()}")
    }
    if a.width() != b.width() {
        throw ValueError(
            f"a frame of {a.width()} columns cannot be compared with one of {b.width()}"
        )
    }
    for (at, name) in a.names.enumerate() {
        if name != b.names[at] {
            throw ValueError(f"column {at} is {name} in one frame and {b.names[at]} in the other")
        }
    }

    let rows = a.len()
    let positions = iota(rows)
    defer { positions.free() }
    var owned: list[Column] = []
    defer {
        for column in owned { column.free() }
    }
    let all_same = Buffer(rows)
    all_same.fill(1.0)
    defer { all_same.free() }

    var same: list[Buffer] = []
    defer {
        for lane in same { lane.free() }
    }
    for (at, name) in a.names.enumerate() {
        var mine = a.columns[at]
        var theirs = b.columns[at]
        if mine.dtype != theirs.dtype {
            throw TypeError(
                f"column {name} is {mine.dtype.name()} in one frame and {theirs.dtype.name()} in the other"
            )
        }
        #: Two dictionaries only compare once they are on one table, exactly as
        #: two join keys only hash alike once they are.
        if mine.dtype.is_dictionary() {
            let pair = unify_levels([mine, theirs])
            owned.push(pair[0])
            owned.push(pair[1])
            mine = pair[0]
            theirs = pair[1]
        }
        let lane = Buffer(rows)
        lane.fill(1.0)
        mine.equal_rows(positions, theirs, positions, rows, lane)
        mine.equal_rows(positions, theirs, positions, rows, all_same)
        same.push(lane)
    }

    let kept = compared_rows(all_same, rows, keep_shape)
    let sifted = kept[0]
    let kept_rows = kept[1]
    defer { sifted.free() }

    var out_names: list[str] = []
    var out_columns: list[Column] = []
    for (at, name) in a.names.enumerate() {
        if not keep_shape and lane_zeros(same[at], rows) == 0 { continue }
        let lane = Buffer(kept_rows)
        kernel_for("jn_unequal_positions").dispatch_async(
            [same[at], sifted, lane],
            kept_rows,
            [kept_rows, flag(keep_equal)]
        )
        out_names.push(f"{name}_self")
        out_columns.push(a.columns[at].take(lane, kept_rows, allow_missing: true))
        out_names.push(f"{name}_other")
        out_columns.push(b.columns[at].take(lane, kept_rows, allow_missing: true))
        lane.free()
    }
    return DataFrame(out_names, out_columns, index: a.index.take(sifted, kept_rows))
}
