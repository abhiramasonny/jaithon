#: The Metal source the join kernels are compiled from, and the cache of them.
#:
#: One translation unit for the whole of `join`, so the hash probe, the radix
#: passes and the asof bisection share the same inline key helpers, and
#: `kernel_for` compiles it once whichever entry point is asked for first.
import std.gpu as gpu
from std.gpu import Kernel

#: Smallest hash table, so a tiny build side still spreads over a few buckets.
pub const MIN_BUCKETS = 16

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
pub fn kernel_for(entry: str) -> Kernel {
    let existing = _kernels.get(entry, null)
    if existing is not null { return existing }
    let compiled = Kernel.compile(SOURCE, entry)
    _kernels[entry] = compiled
    return compiled
}

pub fn flag(value: bool) -> int { return if value { 1 } else { 0 } }

pub fn require_device(what: str) -> void {
    if not gpu.is_available() {
        throw RuntimeError(f"jaiframe needs a GPU to {what}")
    }
}
