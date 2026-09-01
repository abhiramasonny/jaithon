#: The Metal source every kernel in `reduce` is compiled from.
#:
#: One translation unit, so the reduction, scan and window kernels share the
#: same inline helpers and the same constants, and `kernel_for` compiles it
#: once whichever entry point is asked for first.
pub let SOURCE = """#include <metal_stdlib>
using namespace metal;

constexpr constant uint RGROUP = 256;
constexpr constant uint RGROUPS = 256;
constexpr constant uint STATS = 8;
constexpr constant uint FOLDED = 16;
constexpr constant uint SCAN_GROUP = 256;
constexpr constant uint WIN_GROUP = 256;
constexpr constant uint WIN_TILE = 1024;
constexpr constant uint PGROUP = 256;
constexpr constant long WIDE_SCALE = 16777216L;

constexpr constant uint AGG_SUM = 0u;
constexpr constant uint AGG_MEAN = 1u;
constexpr constant uint AGG_MIN = 2u;
constexpr constant uint AGG_MAX = 3u;
constexpr constant uint AGG_COUNT = 4u;
constexpr constant uint AGG_PROD = 5u;
constexpr constant uint AGG_STD = 6u;
constexpr constant uint AGG_VAR = 7u;
constexpr constant uint AGG_MEDIAN = 8u;
constexpr constant uint AGG_QUANTILE = 9u;
constexpr constant uint AGG_ANY = 10u;
constexpr constant uint AGG_ALL = 11u;
constexpr constant uint AGG_SEM = 12u;
constexpr constant uint AGG_FIRST = 13u;
constexpr constant uint AGG_LAST = 14u;
constexpr constant uint AGG_SIZE = 15u;
constexpr constant uint AGG_ARGMIN = 16u;
constexpr constant uint AGG_ARGMAX = 17u;
constexpr constant uint AGG_SKEW = 18u;
constexpr constant uint AGG_KURT = 19u;

inline float missing_value() { return as_type<float>(0x7FC00000u); }

/* The two lanes are only ever recombined in `long`, as they are in the column
 * store: doing it in `float` would throw away everything below the top 24
 * bits, which is the whole reason the second lane exists. */
inline float wide_number(float low, float high) {
    return (float)((long)high * WIDE_SCALE + (long)low);
}

/* A NaN in a slot the validity lane calls present still means missing, because
 * NaN is what a float column stores for one. Every read goes through here so
 * that the two spellings of missing cannot disagree. */
inline bool present_at(device const float* values, device const float* valid,
                       uint at, uint has_valid) {
    if (has_valid != 0u && valid[at] == 0.0f) return false;
    return !isnan(values[at]);
}

/* Float32 reinterpreted so that an unsigned compare is the float compare.
 *
 * Positives get their sign bit set, so they sort above everything; negatives
 * are inverted whole, which reverses the descending order their bit patterns
 * would otherwise have. This is what lets a quantile be found by bisecting a
 * 32-bit integer range instead of by sorting the values. */
inline uint order_key(float value) {
    uint bits = as_type<uint>(value);
    return (bits & 0x80000000u) != 0u ? ~bits : (bits | 0x80000000u);
}

inline float order_float(uint key) {
    uint bits = (key & 0x80000000u) != 0u ? (key & 0x7FFFFFFFu) : ~key;
    return as_type<float>(bits);
}

/* One threadgroup's values folded to a single number, in the scratch the
 * caller lends.
 *
 * Barriered at both ends so the same scratch carries the next fold: seven
 * accumulators through one 256-float array costs seven times eight barriers
 * once per group, against seven arrays permanently resident. Requires the
 * group to be exactly RGROUP wide, which every caller dispatches. */
inline float fold_group(float value, threadgroup float* scratch, uint lid, uint op) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    scratch[lid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = RGROUP / 2u; stride > 0u; stride >>= 1) {
        if (lid < stride) {
            float other = scratch[lid + stride];
            if (op == 0u) scratch[lid] += other;
            else if (op == 1u) scratch[lid] *= other;
            else if (op == 2u) scratch[lid] = min(scratch[lid], other);
            else scratch[lid] = max(scratch[lid], other);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float total = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

/* A value and the row it was found at, folded together.
 *
 * Folding the value alone loses the row, and the row is what `idxmin` is. A
 * thread that drew nothing marks its row negative and never wins; ties go to
 * the lower row, which is the one pandas reports. */
inline void fold_extreme(threadgroup float* best, threadgroup float* at,
                         uint lid, uint want_high) {
    for (uint stride = RGROUP / 2u; stride > 0u; stride >>= 1) {
        if (lid < stride) {
            float other = best[lid + stride];
            float other_at = at[lid + stride];
            bool better = false;
            if (other_at >= 0.0f) {
                if (at[lid] < 0.0f) {
                    better = true;
                } else if (want_high != 0u) {
                    better = other > best[lid] || (other == best[lid] && other_at < at[lid]);
                } else {
                    better = other < best[lid] || (other == best[lid] && other_at < at[lid]);
                }
            }
            if (better) {
                best[lid] = other;
                at[lid] = other_at;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

/* How many of a strided run are present, and how many of those are at or below
 * a key. Both counts come from one walk because every caller wants both. */
inline uint count_at_or_below(device const float* data, device const float* mask,
                              uint has_mask, uint base, uint stride, uint length,
                              uint key, thread uint* out_present) {
    uint seen = 0u;
    uint below = 0u;
    for (uint i = 0u; i < length; ++i) {
        uint at = base + i * stride;
        float value = data[at];
        if (isnan(value)) continue;
        if (has_mask != 0u && mask[at] == 0.0f) continue;
        seen += 1u;
        if (order_key(value) <= key) below += 1u;
    }
    *out_present = seen;
    return below;
}

/* The k-th smallest of a strided run, by bisecting the ordered-key space.
 *
 * Thirty-two counting passes and no scratch at all. A per-thread sort would
 * need a scratch array as wide as the widest window any caller might ask for,
 * and a per-thread insertion sort is quadratic in that width; this is 32w for
 * every width and the answer is one of the values, not an interpolation of
 * two. The whole-column path uses a batched radix histogram instead, which is
 * four passes rather than thirty-two — worth the extra kernel there, because
 * there the run is the whole column, and not worth it here, where every thread
 * would need its own histogram. */
inline uint kth_key_of_run(device const float* data, device const float* mask,
                           uint has_mask, uint base, uint stride, uint length, uint k) {
    uint low = 0u;
    uint high = 0xFFFFFFFFu;
    for (uint step = 0u; step < 32u && low < high; ++step) {
        uint mid = low + ((high - low) >> 1);
        uint seen = 0u;
        uint below = count_at_or_below(data, mask, has_mask, base, stride, length, mid, &seen);
        if (below >= k + 1u) high = mid; else low = mid + 1u;
    }
    return low;
}

/* A linearly interpolated quantile of a strided run.
 *
 * `q` lands between two order statistics. The lower one is bisected for; the
 * upper is the smallest value strictly above it, which one further walk finds
 * — a second bisection would double the cost for a neighbour that is one pass
 * away. `present` comes back so the caller can apply its min_periods. */
inline float run_quantile(device const float* data, device const float* mask,
                          uint has_mask, uint base, uint stride, uint length,
                          float q, thread uint* out_present) {
    uint seen = 0u;
    count_at_or_below(data, mask, has_mask, base, stride, length, 0u, &seen);
    *out_present = seen;
    if (seen == 0u) return missing_value();
    float position = q * (float)(seen - 1u);
    float floored = floor(position);
    float fraction = position - floored;
    uint k0 = (uint)floored;
    uint low_key = kth_key_of_run(data, mask, has_mask, base, stride, length, k0);
    float low_value = order_float(low_key);
    if (fraction <= 0.0f || k0 + 1u >= seen) return low_value;
    uint ignored = 0u;
    uint at_or_below = count_at_or_below(data, mask, has_mask, base, stride, length,
                                         low_key, &ignored);
    if (at_or_below >= k0 + 2u) return low_value;
    uint next_key = 0xFFFFFFFFu;
    for (uint i = 0u; i < length; ++i) {
        uint at = base + i * stride;
        float value = data[at];
        if (isnan(value)) continue;
        if (has_mask != 0u && mask[at] == 0.0f) continue;
        uint key = order_key(value);
        if (key > low_key && key < next_key) next_key = key;
    }
    float high_value = order_float(next_key);
    return low_value + fraction * (high_value - low_value);
}

/* Stage one of a column reduction: count, power sums, product and truth.
 *
 * A fixed grid rather than a thread per row, for jaicv's reason — a thread
 * that reads one float and then spends eight barriers folding it spends more
 * time in threadgroup memory than in the column. 256 groups walk the column in
 * strides accumulating in registers, and the fold happens once per group.
 *
 * The power sums are central, about the `center` the caller passes, not raw.
 * A skew off raw third moments subtracts numbers of the same magnitude and in
 * float32 there is nothing left underneath; a first pass learns the mean and a
 * second pass centres on it. `center` is zero on the first pass, which is what
 * makes the sum in slot one the sum.
 *
 * The quad path reads four rows at a time. It needs a column with no validity
 * lane and a row count that divides by four — a frame straight out of a reader,
 * which is the case worth widening. NaN still has to be excluded inside the
 * quad, which `select` does without branching. */
kernel void red_moments(device const float* values [[buffer(0)]],
                        device const float* valid [[buffer(1)]],
                        device float* partials [[buffer(2)]],
                        constant uint& count [[buffer(3)]],
                        constant uint& has_valid [[buffer(4)]],
                        constant uint& center_bits [[buffer(5)]],
                        constant uint& slot [[buffer(6)]],
                        constant uint& packed [[buffer(7)]],
                        uint gid [[thread_position_in_grid]],
                        uint lid [[thread_index_in_threadgroup]],
                        uint wid [[threadgroup_position_in_grid]]) {
    threadgroup float scratch[RGROUP];
    const uint span = RGROUP * RGROUPS;
    const float center = as_type<float>(center_bits);
    float seen = 0.0f;
    float s1 = 0.0f;
    float s2 = 0.0f;
    float s3 = 0.0f;
    float s4 = 0.0f;
    float product = 1.0f;
    float truthy = 0.0f;
    if (packed != 0u) {
        device const float4* quads = (device const float4*)values;
        uint wide = count / 4u;
        for (uint at = gid; at < wide; at += span) {
            float4 raw = quads[at];
            bool4 ok = !isnan(raw);
            float4 keep = select(float4(0.0f), float4(1.0f), ok);
            float4 shifted = select(float4(0.0f), raw - center, ok);
            float4 squares = shifted * shifted;
            seen += keep.x + keep.y + keep.z + keep.w;
            s1 += shifted.x + shifted.y + shifted.z + shifted.w;
            s2 += squares.x + squares.y + squares.z + squares.w;
            float4 cubes = squares * shifted;
            s3 += cubes.x + cubes.y + cubes.z + cubes.w;
            float4 quarts = squares * squares;
            s4 += quarts.x + quarts.y + quarts.z + quarts.w;
            float4 factors = select(float4(1.0f), raw, ok);
            product *= factors.x * factors.y * factors.z * factors.w;
            float4 hot = select(float4(0.0f), float4(1.0f), ok && raw != 0.0f);
            truthy += hot.x + hot.y + hot.z + hot.w;
        }
    } else {
        for (uint at = gid; at < count; at += span) {
            if (!present_at(values, valid, at, has_valid)) continue;
            float raw = values[at];
            float shifted = raw - center;
            float squared = shifted * shifted;
            seen += 1.0f;
            s1 += shifted;
            s2 += squared;
            s3 += squared * shifted;
            s4 += squared * squared;
            product *= raw;
            if (raw != 0.0f) truthy += 1.0f;
        }
    }
    seen = fold_group(seen, scratch, lid, 0u);
    s1 = fold_group(s1, scratch, lid, 0u);
    s2 = fold_group(s2, scratch, lid, 0u);
    s3 = fold_group(s3, scratch, lid, 0u);
    s4 = fold_group(s4, scratch, lid, 0u);
    product = fold_group(product, scratch, lid, 1u);
    truthy = fold_group(truthy, scratch, lid, 0u);
    if (lid != 0u) return;
    uint base = (slot * RGROUPS + wid) * STATS;
    partials[base + 0u] = seen;
    partials[base + 1u] = s1;
    partials[base + 2u] = s2;
    partials[base + 3u] = s3;
    partials[base + 4u] = s4;
    partials[base + 5u] = product;
    partials[base + 6u] = truthy;
    partials[base + 7u] = 0.0f;
}

/* Stage one of the extremes, the rows they sit at, and the edges of the data.
 *
 * The same fixed grid. Five things fold here rather than two, because they all
 * come off the same read: the smallest and largest values with their rows, the
 * first and last rows that hold a value, and the first row that does not. That
 * last one is what a `skipna=false` reduction needs to know, and asking for it
 * separately would be a second pass over the column.
 *
 * A missing row is INFINITY inside the thread and -1 in the partials, so the
 * fold has one test for "this group saw nothing" rather than a sentinel per
 * quantity. */
kernel void red_extrema(device const float* values [[buffer(0)]],
                        device const float* valid [[buffer(1)]],
                        device float* partials [[buffer(2)]],
                        constant uint& count [[buffer(3)]],
                        constant uint& has_valid [[buffer(4)]],
                        constant uint& slot [[buffer(5)]],
                        constant uint& packed [[buffer(6)]],
                        uint gid [[thread_position_in_grid]],
                        uint lid [[thread_index_in_threadgroup]],
                        uint wid [[threadgroup_position_in_grid]]) {
    threadgroup float scratch[RGROUP];
    threadgroup float lows[RGROUP];
    threadgroup float low_at[RGROUP];
    threadgroup float highs[RGROUP];
    threadgroup float high_at[RGROUP];
    const uint span = RGROUP * RGROUPS;
    float my_low = INFINITY;
    float my_high = -INFINITY;
    float my_low_at = -1.0f;
    float my_high_at = -1.0f;
    float first_present = INFINITY;
    float last_present = -1.0f;
    float first_missing = INFINITY;
    if (packed != 0u) {
        device const float4* quads = (device const float4*)values;
        uint wide = count / 4u;
        for (uint at = gid; at < wide; at += span) {
            float4 raw = quads[at];
            bool4 ok = !isnan(raw);
            float4 lowv = select(float4(INFINITY), raw, ok);
            float4 highv = select(float4(-INFINITY), raw, ok);
            float low = min(min(lowv.x, lowv.y), min(lowv.z, lowv.w));
            float high = max(max(highv.x, highv.y), max(highv.z, highv.w));
            float row = (float)(at * 4u);
            if (low < INFINITY) {
                uint which = lowv.x == low ? 0u : (lowv.y == low ? 1u : (lowv.z == low ? 2u : 3u));
                if (my_low_at < 0.0f || low < my_low) {
                    my_low = low;
                    my_low_at = row + (float)which;
                }
            }
            if (high > -INFINITY) {
                uint which = highv.x == high ? 0u : (highv.y == high ? 1u : (highv.z == high ? 2u : 3u));
                if (my_high_at < 0.0f || high > my_high) {
                    my_high = high;
                    my_high_at = row + (float)which;
                }
            }
            for (uint j = 0u; j < 4u; ++j) {
                bool good = j == 0u ? ok.x : (j == 1u ? ok.y : (j == 2u ? ok.z : ok.w));
                float here = row + (float)j;
                if (good) {
                    first_present = min(first_present, here);
                    last_present = max(last_present, here);
                } else {
                    first_missing = min(first_missing, here);
                }
            }
        }
    } else {
        for (uint at = gid; at < count; at += span) {
            float here = (float)at;
            if (!present_at(values, valid, at, has_valid)) {
                first_missing = min(first_missing, here);
                continue;
            }
            float raw = values[at];
            first_present = min(first_present, here);
            last_present = max(last_present, here);
            if (my_low_at < 0.0f || raw < my_low) {
                my_low = raw;
                my_low_at = here;
            }
            if (my_high_at < 0.0f || raw > my_high) {
                my_high = raw;
                my_high_at = here;
            }
        }
    }
    lows[lid] = my_low;
    low_at[lid] = my_low_at;
    highs[lid] = my_high;
    high_at[lid] = my_high_at;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    fold_extreme(lows, low_at, lid, 0u);
    fold_extreme(highs, high_at, lid, 1u);
    first_present = fold_group(first_present, scratch, lid, 2u);
    last_present = fold_group(last_present, scratch, lid, 3u);
    first_missing = fold_group(first_missing, scratch, lid, 2u);
    if (lid != 0u) return;
    uint base = (slot * RGROUPS + wid) * STATS;
    partials[base + 0u] = low_at[0] < 0.0f ? 0.0f : lows[0];
    partials[base + 1u] = high_at[0] < 0.0f ? 0.0f : highs[0];
    partials[base + 2u] = low_at[0];
    partials[base + 3u] = high_at[0];
    partials[base + 4u] = first_present < INFINITY ? first_present : -1.0f;
    partials[base + 5u] = last_present;
    partials[base + 6u] = first_missing < INFINITY ? first_missing : -1.0f;
    partials[base + 7u] = 0.0f;
}

/* The row an Int64 or Datetime column's extreme really sits at.
 *
 * `red_extrema` folds the flattened lane, and at 1.7e12 a float32 resolves to
 * the nearest 131072: every millisecond of a day and a half ties there, so the
 * fold hands back the first row of the block rather than the smallest
 * timestamp in it. Recombining the two lanes in `long` is the only compare
 * that answers `min` over stamps, which is what the column stores.
 *
 * Only the row comes out. The value is read back off the column afterwards,
 * which is what every aggregation that picks a row already does.
 *
 * The same fixed grid as the float pass, folded through a `long` scratch, with
 * the winner handed back split across the two slots it is stored in — a float
 * partial cannot carry it whole, which is the whole reason this kernel is
 * here. */
kernel void red_wide_extreme(device const float* values [[buffer(0)]],
                             device const float* wide [[buffer(1)]],
                             device const float* valid [[buffer(2)]],
                             device float* partials [[buffer(3)]],
                             constant uint& count [[buffer(4)]],
                             constant uint& has_valid [[buffer(5)]],
                             constant uint& want_high [[buffer(6)]],
                             uint gid [[thread_position_in_grid]],
                             uint lid [[thread_index_in_threadgroup]],
                             uint wid [[threadgroup_position_in_grid]]) {
    threadgroup long best[RGROUP];
    threadgroup float best_at[RGROUP];
    const uint span = RGROUP * RGROUPS;
    long mine = 0L;
    float mine_at = -1.0f;
    for (uint at = gid; at < count; at += span) {
        if (has_valid != 0u && valid[at] == 0.0f) continue;
        if (isnan(values[at]) || isnan(wide[at])) continue;
        long here = (long)wide[at] * WIDE_SCALE + (long)values[at];
        bool better = want_high != 0u ? here > mine : here < mine;
        if (mine_at < 0.0f || better) {
            mine = here;
            mine_at = (float)at;
        }
    }
    best[lid] = mine;
    best_at[lid] = mine_at;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = RGROUP / 2u; stride > 0u; stride >>= 1) {
        if (lid < stride) {
            long other = best[lid + stride];
            float other_at = best_at[lid + stride];
            if (other_at >= 0.0f) {
                bool wins = best_at[lid] < 0.0f;
                if (!wins) {
                    wins = want_high != 0u ? other > best[lid] : other < best[lid];
                    if (!wins) wins = other == best[lid] && other_at < best_at[lid];
                }
                if (wins) {
                    best[lid] = other;
                    best_at[lid] = other_at;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid != 0u) return;
    long won = best[0];
    long block = won >> 24;
    partials[wid * 3u + 0u] = (float)(won - (block << 24));
    partials[wid * 3u + 1u] = (float)block;
    partials[wid * 3u + 2u] = best_at[0];
}

/* The moment partials folded to one row of numbers per column.
 *
 * A thread to each column's each slot: the slots combine independently, and
 * with 256 groups a serial fold is 256 dependent adds. Slot five is the
 * product and multiplies; everything else adds. What matters is that only the
 * folded row crosses the bus. */
kernel void red_join_moments(device const float* partials [[buffer(0)]],
                             device float* out [[buffer(1)]],
                             constant uint& lanes [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= lanes * STATS) return;
    uint column = gid / STATS;
    uint field = gid - column * STATS;
    uint base = column * RGROUPS * STATS + field;
    float total = field == 5u ? 1.0f : 0.0f;
    for (uint g = 0u; g < RGROUPS; ++g) {
        float part = partials[base + g * STATS];
        if (field == 5u) total *= part; else total += part;
    }
    out[column * FOLDED + field] = total;
}

/* The extrema partials folded the same way, by comparison rather than by
 * adding, and with the row carried alongside the value. A group that drew
 * nothing left its rows at -1 and must not win. */
kernel void red_join_extrema(device const float* partials [[buffer(0)]],
                             device float* out [[buffer(1)]],
                             constant uint& lanes [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= lanes) return;
    uint base = gid * RGROUPS * STATS;
    float low = 0.0f;
    float high = 0.0f;
    float low_at = -1.0f;
    float high_at = -1.0f;
    float first_present = -1.0f;
    float last_present = -1.0f;
    float first_missing = -1.0f;
    for (uint g = 0u; g < RGROUPS; ++g) {
        uint at = base + g * STATS;
        float group_low = partials[at + 0u];
        float group_high = partials[at + 1u];
        float group_low_at = partials[at + 2u];
        float group_high_at = partials[at + 3u];
        if (group_low_at >= 0.0f
            && (low_at < 0.0f || group_low < low
                || (group_low == low && group_low_at < low_at))) {
            low = group_low;
            low_at = group_low_at;
        }
        if (group_high_at >= 0.0f
            && (high_at < 0.0f || group_high > high
                || (group_high == high && group_high_at < high_at))) {
            high = group_high;
            high_at = group_high_at;
        }
        float group_first = partials[at + 4u];
        if (group_first >= 0.0f && (first_present < 0.0f || group_first < first_present)) {
            first_present = group_first;
        }
        last_present = max(last_present, partials[at + 5u]);
        float group_gap = partials[at + 6u];
        if (group_gap >= 0.0f && (first_missing < 0.0f || group_gap < first_missing)) {
            first_missing = group_gap;
        }
    }
    uint slot = gid * FOLDED + STATS;
    out[slot + 0u] = low;
    out[slot + 1u] = high;
    out[slot + 2u] = low_at;
    out[slot + 3u] = high_at;
    out[slot + 4u] = first_present;
    out[slot + 5u] = last_present;
    out[slot + 6u] = first_missing;
    out[slot + 7u] = 0.0f;
}

/* One column's values and presence written into a row-major block of a matrix
 * that a per-row or a pairwise pass then reads.
 *
 * Wide columns are flattened here, so nothing downstream has to know about the
 * second lane. `center` is subtracted as it goes: a covariance taken from raw
 * sums subtracts two numbers of the same size, and in float32 that is where
 * the digits go. Shifting a column by a constant does not change any
 * covariance or variance computed from it, so the shortcut formula stays exact
 * in algebra and becomes well conditioned in arithmetic. The per-row path
 * passes a centre of zero, because there the values themselves are the
 * answer. */
kernel void red_pack(device const float* values [[buffer(0)]],
                     device const float* wide [[buffer(1)]],
                     device const float* valid [[buffer(2)]],
                     device float* matrix [[buffer(3)]],
                     device float* mask [[buffer(4)]],
                     constant uint& rows [[buffer(5)]],
                     constant uint& base [[buffer(6)]],
                     constant uint& has_wide [[buffer(7)]],
                     constant uint& has_valid [[buffer(8)]],
                     constant uint& center_bits [[buffer(9)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= rows) return;
    float raw = has_wide != 0u ? wide_number(values[gid], wide[gid]) : values[gid];
    bool present = !isnan(raw) && (has_valid == 0u || valid[gid] != 0.0f);
    matrix[base + gid] = present ? raw - as_type<float>(center_bits) : 0.0f;
    mask[base + gid] = present ? 1.0f : 0.0f;
}

/* A reduction across the columns of one row.
 *
 * One thread per row walking the width, which is the opposite shape to the
 * column reductions and the right one here: a frame is a handful of columns
 * and a great many rows, so the parallelism belongs on the rows. The matrix is
 * column-major, so the width-th read of neighbouring threads is contiguous.
 *
 * Skew, kurtosis and the quantiles each need a second walk of the width, which
 * costs a handful of extra reads inside a thread rather than a second
 * dispatch. */
kernel void red_axis(device const float* matrix [[buffer(0)]],
                     device const float* mask [[buffer(1)]],
                     device float* out_values [[buffer(2)]],
                     device float* out_valid [[buffer(3)]],
                     constant uint& rows [[buffer(4)]],
                     constant uint& width [[buffer(5)]],
                     constant uint& agg [[buffer(6)]],
                     constant uint& ddof [[buffer(7)]],
                     constant uint& q_bits [[buffer(8)]],
                     constant uint& skipna [[buffer(9)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= rows) return;
    if (agg == AGG_MEDIAN || agg == AGG_QUANTILE) {
        uint seen = 0u;
        float q = agg == AGG_MEDIAN ? 0.5f : as_type<float>(q_bits);
        float answer = run_quantile(matrix, mask, 1u, gid, rows, width, q, &seen);
        bool ok = seen > 0u && (skipna != 0u || seen == width);
        out_values[gid] = ok ? answer : missing_value();
        out_valid[gid] = ok ? 1.0f : 0.0f;
        return;
    }
    float seen = 0.0f;
    float total = 0.0f;
    float squares = 0.0f;
    float product = 1.0f;
    float truthy = 0.0f;
    float low = 0.0f;
    float high = 0.0f;
    float low_at = -1.0f;
    float high_at = -1.0f;
    float first = 0.0f;
    float last = 0.0f;
    bool started = false;
    for (uint c = 0u; c < width; ++c) {
        uint at = c * rows + gid;
        if (mask[at] == 0.0f) continue;
        float value = matrix[at];
        seen += 1.0f;
        total += value;
        squares += value * value;
        product *= value;
        if (value != 0.0f) truthy += 1.0f;
        if (!started || value < low) {
            low = value;
            low_at = (float)c;
        }
        if (!started || value > high) {
            high = value;
            high_at = (float)c;
        }
        if (!started) first = value;
        last = value;
        started = true;
    }
    bool complete = seen == (float)width;
    bool ok = seen > 0.0f && (skipna != 0u || complete);
    float answer = 0.0f;
    if (agg == AGG_COUNT) {
        out_values[gid] = seen;
        out_valid[gid] = 1.0f;
        return;
    }
    if (agg == AGG_SIZE) {
        out_values[gid] = (float)width;
        out_valid[gid] = 1.0f;
        return;
    }
    if (agg == AGG_SUM) answer = total;
    else if (agg == AGG_MEAN) answer = total / seen;
    else if (agg == AGG_MIN) answer = low;
    else if (agg == AGG_MAX) answer = high;
    else if (agg == AGG_PROD) answer = product;
    else if (agg == AGG_ANY) answer = truthy > 0.0f ? 1.0f : 0.0f;
    else if (agg == AGG_ALL) answer = truthy == seen ? 1.0f : 0.0f;
    else if (agg == AGG_FIRST) answer = first;
    else if (agg == AGG_LAST) answer = last;
    else if (agg == AGG_ARGMIN) answer = low_at;
    else if (agg == AGG_ARGMAX) answer = high_at;
    else if (agg == AGG_SKEW || agg == AGG_KURT) {
        /* A second walk of the width, about the mean the first one found.
         * Raw third and fourth moments subtract numbers of the same size and
         * float32 has nothing left underneath; the width is a handful, so the
         * second walk is cheaper than the digits it saves. */
        float needed = agg == AGG_SKEW ? 3.0f : 4.0f;
        float mean = total / seen;
        float m2 = 0.0f;
        float m3 = 0.0f;
        float m4 = 0.0f;
        for (uint c = 0u; c < width; ++c) {
            uint at = c * rows + gid;
            if (mask[at] == 0.0f) continue;
            float d = matrix[at] - mean;
            float dd = d * d;
            m2 += dd;
            m3 += dd * d;
            m4 += dd * dd;
        }
        float spread = m2 / (seen - 1.0f);
        if (seen < needed || spread <= 0.0f) {
            out_values[gid] = missing_value();
            out_valid[gid] = 0.0f;
            return;
        }
        if (agg == AGG_SKEW) {
            float deviation = sqrt(spread);
            answer = seen / ((seen - 1.0f) * (seen - 2.0f))
                   * (m3 / (deviation * deviation * deviation));
        } else {
            float lead = seen * (seen + 1.0f)
                       / ((seen - 1.0f) * (seen - 2.0f) * (seen - 3.0f));
            float tail = 3.0f * (seen - 1.0f) * (seen - 1.0f)
                       / ((seen - 2.0f) * (seen - 3.0f));
            answer = lead * (m4 / (spread * spread)) - tail;
        }
    }
    else {
        float degrees = seen - (float)ddof;
        if (degrees <= 0.0f) {
            out_values[gid] = missing_value();
            out_valid[gid] = 0.0f;
            return;
        }
        float mean = total / seen;
        float variance = (squares - mean * total) / degrees;
        variance = max(variance, 0.0f);
        if (agg == AGG_VAR) answer = variance;
        else if (agg == AGG_STD) answer = sqrt(variance);
        else answer = sqrt(variance / seen);
    }
    out_values[gid] = ok ? answer : missing_value();
    out_valid[gid] = ok ? 1.0f : 0.0f;
}

/* Every column pair's co-moments in one pass over the packed matrix.
 *
 * One group per (pair, chunk): the pairs are independent, and a dispatch each
 * would pay the launch and the fold once per pair — a frame of twenty columns
 * is two hundred and ten of them. The pair and the chunk come out of the group
 * index with one division and a subtraction, because the device has no integer
 * divide to spend twice.
 *
 * Pairwise-complete, as pandas is: a row counts for a pair only when both of
 * its columns hold a value, so two pairs of the same frame are computed over
 * different numbers of rows and no pair can be assembled from per-column
 * sums. */
kernel void red_pairwise(device const float* matrix [[buffer(0)]],
                         device const float* mask [[buffer(1)]],
                         device const float* pairs [[buffer(2)]],
                         device float* partials [[buffer(3)]],
                         constant uint& rows [[buffer(4)]],
                         constant uint& chunks [[buffer(5)]],
                         uint lid [[thread_index_in_threadgroup]],
                         uint wid [[threadgroup_position_in_grid]]) {
    threadgroup float scratch[PGROUP];
    uint pair = wid / chunks;
    uint chunk = wid - pair * chunks;
    uint left = (uint)pairs[pair * 2u] * rows;
    uint right = (uint)pairs[pair * 2u + 1u] * rows;
    float seen = 0.0f;
    float sx = 0.0f;
    float sy = 0.0f;
    float sxx = 0.0f;
    float syy = 0.0f;
    float sxy = 0.0f;
    for (uint r = chunk * PGROUP + lid; r < rows; r += chunks * PGROUP) {
        if (mask[left + r] == 0.0f || mask[right + r] == 0.0f) continue;
        float x = matrix[left + r];
        float y = matrix[right + r];
        seen += 1.0f;
        sx += x;
        sy += y;
        sxx += x * x;
        syy += y * y;
        sxy += x * y;
    }
    seen = fold_group(seen, scratch, lid, 0u);
    sx = fold_group(sx, scratch, lid, 0u);
    sy = fold_group(sy, scratch, lid, 0u);
    sxx = fold_group(sxx, scratch, lid, 0u);
    syy = fold_group(syy, scratch, lid, 0u);
    sxy = fold_group(sxy, scratch, lid, 0u);
    if (lid != 0u) return;
    uint base = wid * STATS;
    partials[base + 0u] = seen;
    partials[base + 1u] = sx;
    partials[base + 2u] = sy;
    partials[base + 3u] = sxx;
    partials[base + 4u] = syy;
    partials[base + 5u] = sxy;
    partials[base + 6u] = 0.0f;
    partials[base + 7u] = 0.0f;
}

kernel void red_join_pairwise(device const float* partials [[buffer(0)]],
                              device float* out [[buffer(1)]],
                              constant uint& lanes [[buffer(2)]],
                              constant uint& chunks [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= lanes * STATS) return;
    uint pair = gid / STATS;
    uint field = gid - pair * STATS;
    float total = 0.0f;
    for (uint c = 0u; c < chunks; ++c) total += partials[(pair * chunks + c) * STATS + field];
    out[gid] = total;
}

/* An expanding window's answers blanked where too few rows have arrived.
 *
 * The scans that answer an expanding minimum, maximum or product run from row
 * zero and have no idea what min_periods is; `tally` is the running count of
 * present rows, which the same pass over the column already produced. */
kernel void win_gate(device float* values [[buffer(0)]],
                     device float* valid [[buffer(1)]],
                     device const float* tally [[buffer(2)]],
                     constant uint& count [[buffer(3)]],
                     constant uint& min_periods [[buffer(4)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    float seen = tally[gid];
    if (seen > 0.0f && seen >= (float)min_periods) return;
    values[gid] = missing_value();
    valid[gid] = 0.0f;
}

/* One radix pass of a batched order-statistic search: how many values of each
 * target carry each of 256 digit values, among those that match the digits
 * already fixed.
 *
 * Four of these narrow the answer from 32 bits to none, so a whole-column
 * median is four dispatches and four downloads of 256 counts a target —
 * against a sort of the column, which nothing here needs for anything else.
 * Targets are batched because `describe` wants three quantiles of every
 * column and the pass over the rows is the same pass for all of them.
 *
 * Each group counts into its own bins in threadgroup memory and merges once,
 * which is jaicv's histogram trick: a million rows atomically adding into 256
 * device bins is a million contended writes, and this is one per group per
 * occupied bin. */
kernel void sel_histogram(device const float* matrix [[buffer(0)]],
                          device const float* mask [[buffer(1)]],
                          device const float* targets [[buffer(2)]],
                          device const float* prefixes [[buffer(3)]],
                          device float* bins [[buffer(4)]],
                          constant uint& rows [[buffer(5)]],
                          constant uint& groups [[buffer(6)]],
                          constant uint& shift [[buffer(7)]],
                          constant uint& has_mask [[buffer(8)]],
                          uint lid [[thread_index_in_threadgroup]],
                          uint wid [[threadgroup_position_in_grid]]) {
    threadgroup atomic_uint counts[256];
    atomic_store_explicit(&counts[lid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint target = wid / groups;
    uint chunk = wid - target * groups;
    uint base = (uint)targets[target] * rows;
    uint fixed = (uint)prefixes[target];
    for (uint r = chunk * RGROUP + lid; r < rows; r += groups * RGROUP) {
        float value = matrix[base + r];
        if (isnan(value)) continue;
        if (has_mask != 0u && mask[base + r] == 0.0f) continue;
        uint key = order_key(value);
        if (shift < 24u && (key >> (shift + 8u)) != fixed) continue;
        uint digit = (key >> shift) & 0xFFu;
        atomic_fetch_add_explicit(&counts[digit], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint mine = atomic_load_explicit(&counts[lid], memory_order_relaxed);
    if (mine == 0u) return;
    atomic_fetch_add_explicit((device atomic<float>*)(bins + target * 256u + lid),
                              (float)mine, memory_order_relaxed);
}

inline float scan_identity(uint op) {
    if (op == 0u) return 0.0f;
    if (op == 1u) return 1.0f;
    if (op == 2u) return -INFINITY;
    return INFINITY;
}

/* The four scan operators, all commutative, so the sweep may combine its two
 * operands in either order.
 *
 * max and min get an explicit NaN test because Metal's do not have one: they
 * return the other operand, and a cumulative maximum with `skipna` off has to
 * stop at the first missing row rather than step over it. */
inline float scan_join(float a, float b, uint op) {
    if (op == 0u) return a + b;
    if (op == 1u) return a * b;
    if (isnan(a) || isnan(b)) return missing_value();
    return op == 2u ? max(a, b) : min(a, b);
}

/* One block of a work-efficient scan: an upsweep to the block total, then a
 * downsweep that hands every element the total of everything before it.
 *
 * Blelloch rather than the doubling scan the column store uses for its filter
 * offsets. Doubling costs n log n combines a block; this costs 2n, and a scan
 * of a whole column runs this kernel over every block of it. The block total
 * goes out the side so the totals can be scanned by the same kernel and added
 * back, which is what makes 2^24 rows three levels of dispatch rather than a
 * host loop over sixty-five thousand block sums. */
kernel void scan_block(device const float* input [[buffer(0)]],
                       device float* out [[buffer(1)]],
                       device float* totals [[buffer(2)]],
                       constant uint& count [[buffer(3)]],
                       constant uint& op [[buffer(4)]],
                       constant uint& inclusive [[buffer(5)]],
                       uint lid [[thread_index_in_threadgroup]],
                       uint wid [[threadgroup_position_in_grid]]) {
    threadgroup float scratch[SCAN_GROUP];
    uint base = wid * SCAN_GROUP;
    float own = base + lid < count ? input[base + lid] : scan_identity(op);
    scratch[lid] = own;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1u; stride < SCAN_GROUP; stride <<= 1) {
        uint at = (lid + 1u) * (stride << 1) - 1u;
        if (at < SCAN_GROUP) scratch[at] = scan_join(scratch[at - stride], scratch[at], op);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) {
        totals[wid] = scratch[SCAN_GROUP - 1u];
        scratch[SCAN_GROUP - 1u] = scan_identity(op);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = SCAN_GROUP >> 1; stride > 0u; stride >>= 1) {
        uint at = (lid + 1u) * (stride << 1) - 1u;
        if (at < SCAN_GROUP) {
            float carried = scratch[at - stride];
            scratch[at - stride] = scratch[at];
            scratch[at] = scan_join(carried, scratch[at], op);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (base + lid >= count) return;
    out[base + lid] = inclusive != 0u ? scan_join(scratch[lid], own, op) : scratch[lid];
}

kernel void scan_carry(device float* out [[buffer(0)]],
                       device const float* carried [[buffer(1)]],
                       constant uint& count [[buffer(2)]],
                       constant uint& op [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out[gid] = scan_join(carried[gid / SCAN_GROUP], out[gid], op);
}

/* The lane a scan actually runs over.
 *
 * A missing row contributes the operator's identity when nulls are being
 * skipped, so the running total steps over it and comes out the same on the
 * far side, and a NaN when they are not, so that everything from the first
 * missing row onwards is missing too. That second case is why the operators
 * above test for NaN: it is the whole mechanism, and it costs no extra pass
 * over the column. */
kernel void scan_prepare(device const float* values [[buffer(0)]],
                         device const float* wide [[buffer(1)]],
                         device const float* valid [[buffer(2)]],
                         device float* out [[buffer(3)]],
                         constant uint& count [[buffer(4)]],
                         constant uint& has_wide [[buffer(5)]],
                         constant uint& has_valid [[buffer(6)]],
                         constant uint& op [[buffer(7)]],
                         constant uint& skipna [[buffer(8)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    float raw = has_wide != 0u ? wide_number(values[gid], wide[gid]) : values[gid];
    bool present = !isnan(raw) && (has_valid == 0u || valid[gid] != 0.0f);
    if (present) {
        out[gid] = raw;
        return;
    }
    out[gid] = skipna != 0u ? scan_identity(op) : missing_value();
}

kernel void scan_finish(device float* result [[buffer(0)]],
                        device const float* valid [[buffer(1)]],
                        device float* out_valid [[buffer(2)]],
                        constant uint& count [[buffer(3)]],
                        constant uint& has_valid [[buffer(4)]],
                        constant uint& skipna [[buffer(5)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    bool present = skipna != 0u
        ? (has_valid == 0u || valid[gid] != 0.0f)
        : !isnan(result[gid]);
    out_valid[gid] = present ? 1.0f : 0.0f;
    if (!present) result[gid] = missing_value();
}

kernel void scan_missing_flags(device const float* values [[buffer(0)]],
                               device const float* valid [[buffer(1)]],
                               device float* out [[buffer(2)]],
                               constant uint& count [[buffer(3)]],
                               constant uint& has_valid [[buffer(4)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    bool present = !isnan(values[gid]) && (has_valid == 0u || valid[gid] != 0.0f);
    out[gid] = present ? 0.0f : 1.0f;
}

/* A float lane split back into the two lanes an `Int64` column stores.
 *
 * Counts and row positions come out of these kernels as floats and go back
 * into a column as exact integers; every one of them is well under 2^24, so
 * the split is exact and the second lane ends up zero. */
kernel void red_split_wide(device const float* src [[buffer(0)]],
                           device float* low [[buffer(1)]],
                           device float* high [[buffer(2)]],
                           constant uint& count [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    long value = (long)src[gid];
    long top = value >> 24;
    low[gid] = (float)(value - (top << 24));
    high[gid] = (float)top;
}

/* The two lanes an exponentially weighted mean is a ratio of.
 *
 * Both obey the same recurrence, `y = decay * y_before + own`, so one scan
 * carries them together. With `adjust` the weights are 1 and the ratio is the
 * weighted mean pandas defines; without it the recurrence is
 * `y = (1 - alpha) y_before + alpha x`, which is the same ratio once the first
 * present row is given weight 1 instead of alpha — that seed is the only
 * difference between the two conventions, and it is why the first present row
 * has to be known before the scan starts. */
kernel void ewm_prepare(device const float* values [[buffer(0)]],
                        device const float* wide [[buffer(1)]],
                        device const float* valid [[buffer(2)]],
                        device float* nums [[buffer(3)]],
                        device float* dens [[buffer(4)]],
                        constant uint& count [[buffer(5)]],
                        constant uint& has_wide [[buffer(6)]],
                        constant uint& has_valid [[buffer(7)]],
                        constant uint& alpha_bits [[buffer(8)]],
                        constant uint& adjust [[buffer(9)]],
                        constant uint& first_row [[buffer(10)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    float raw = has_wide != 0u ? wide_number(values[gid], wide[gid]) : values[gid];
    bool present = !isnan(raw) && (has_valid == 0u || valid[gid] != 0.0f);
    if (!present) {
        nums[gid] = 0.0f;
        dens[gid] = 0.0f;
        return;
    }
    float weight = adjust != 0u || gid == first_row ? 1.0f : as_type<float>(alpha_bits);
    nums[gid] = weight * raw;
    dens[gid] = weight;
}

/* One block of the exponentially weighted recurrence, both lanes at once.
 *
 * Doubling here rather than the work-efficient sweep the plain scans use. The
 * operator is `y = decay^d * carried + own`, whose multiplier depends on how
 * far apart the two operands are; the downsweep's swap step moves a partial to
 * a different distance from its neighbour and does not carry that. Doubling
 * costs log(n) redundant combines a block against the sweep's two, which over
 * a block of 256 is the cheaper of the two prices. */
kernel void ewm_block(device const float* nums [[buffer(0)]],
                      device const float* dens [[buffer(1)]],
                      device float* out_num [[buffer(2)]],
                      device float* out_den [[buffer(3)]],
                      device float* total_num [[buffer(4)]],
                      device float* total_den [[buffer(5)]],
                      constant uint& count [[buffer(6)]],
                      constant uint& decay_bits [[buffer(7)]],
                      uint lid [[thread_index_in_threadgroup]],
                      uint wid [[threadgroup_position_in_grid]]) {
    threadgroup float running_num[SCAN_GROUP];
    threadgroup float running_den[SCAN_GROUP];
    uint base = wid * SCAN_GROUP;
    bool inside = base + lid < count;
    running_num[lid] = inside ? nums[base + lid] : 0.0f;
    running_den[lid] = inside ? dens[base + lid] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float step = as_type<float>(decay_bits);
    for (uint stride = 1u; stride < SCAN_GROUP; stride <<= 1) {
        float from_num = lid >= stride ? running_num[lid - stride] : 0.0f;
        float from_den = lid >= stride ? running_den[lid - stride] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        running_num[lid] += step * from_num;
        running_den[lid] += step * from_den;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        step *= step;
    }
    if (inside) {
        out_num[base + lid] = running_num[lid];
        out_den[base + lid] = running_den[lid];
    }
    if (lid == SCAN_GROUP - 1u) {
        total_num[wid] = running_num[lid];
        total_den[wid] = running_den[lid];
    }
}

kernel void ewm_carry(device float* out_num [[buffer(0)]],
                      device float* out_den [[buffer(1)]],
                      device const float* carry_num [[buffer(2)]],
                      device const float* carry_den [[buffer(3)]],
                      constant uint& count [[buffer(4)]],
                      constant uint& decay_bits [[buffer(5)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint block = gid / SCAN_GROUP;
    if (block == 0u) return;
    uint local = gid - block * SCAN_GROUP;
    float reach = pow(as_type<float>(decay_bits), (float)(local + 1u));
    out_num[gid] += reach * carry_num[block - 1u];
    out_den[gid] += reach * carry_den[block - 1u];
}

kernel void ewm_finish(device const float* nums [[buffer(0)]],
                       device const float* dens [[buffer(1)]],
                       device float* out_values [[buffer(2)]],
                       device float* out_valid [[buffer(3)]],
                       constant uint& count [[buffer(4)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    float weight = dens[gid];
    bool ok = weight > 0.0f;
    out_values[gid] = ok ? nums[gid] / weight : missing_value();
    out_valid[gid] = ok ? 1.0f : 0.0f;
}

/* One thread per output row, each walking its own window.
 *
 * Neighbouring threads read overlapping rows — a window of w shares w-1 of
 * them with its neighbour — so the group stages the union of its threads'
 * windows in threadgroup memory first and every row is read from device memory
 * once instead of w times. The tile is 1024 wide, which covers a group of 256
 * and any window up to 769; past that the kernel reads through, and past that
 * is also where the host stops calling this path at all and scans instead.
 *
 * The window of row i ends at i, or at i + (w-1)/2 when it is centred, which
 * is where pandas puts the label. An expanding window starts at row zero
 * instead, and then no tile can hold it. */
kernel void roll_direct(device const float* values [[buffer(0)]],
                        device const float* valid [[buffer(1)]],
                        device float* out_values [[buffer(2)]],
                        device float* out_valid [[buffer(3)]],
                        constant uint& count [[buffer(4)]],
                        constant uint& window [[buffer(5)]],
                        constant uint& min_periods [[buffer(6)]],
                        constant uint& lead [[buffer(7)]],
                        constant uint& agg [[buffer(8)]],
                        constant uint& ddof [[buffer(9)]],
                        constant uint& q_bits [[buffer(10)]],
                        constant uint& has_valid [[buffer(11)]],
                        constant uint& widening [[buffer(12)]],
                        constant uint& tiled [[buffer(13)]],
                        uint gid [[thread_position_in_grid]],
                        uint lid [[thread_index_in_threadgroup]],
                        uint wid [[threadgroup_position_in_grid]]) {
    threadgroup float tile[WIN_TILE];
    threadgroup float tile_mask[WIN_TILE];
    int origin = (int)(wid * WIN_GROUP) + (int)lead - (int)window + 1;
    if (tiled != 0u) {
        uint span = WIN_GROUP + window - 1u;
        for (uint t = lid; t < span; t += WIN_GROUP) {
            int at = origin + (int)t;
            bool inside = at >= 0 && at < (int)count;
            float value = inside ? values[at] : 0.0f;
            bool ok = inside && !isnan(value)
                && (has_valid == 0u || valid[at] != 0.0f);
            tile[t] = ok ? value : 0.0f;
            tile_mask[t] = ok ? 1.0f : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (gid >= count) return;
    int last = widening != 0u ? (int)gid : (int)gid + (int)lead;
    int first = widening != 0u ? 0 : last - (int)window + 1;
    int lo = max(first, 0);
    int hi = min(last, (int)count - 1);
    if (hi < lo) {
        out_values[gid] = missing_value();
        out_valid[gid] = 0.0f;
        return;
    }
    if (agg == AGG_MEDIAN || agg == AGG_QUANTILE) {
        uint seen = 0u;
        float q = agg == AGG_MEDIAN ? 0.5f : as_type<float>(q_bits);
        float answer = run_quantile(values, valid, has_valid, (uint)lo, 1u,
                                    (uint)(hi - lo + 1), q, &seen);
        bool ok = seen > 0u && seen >= min_periods;
        out_values[gid] = ok ? answer : missing_value();
        out_valid[gid] = ok ? 1.0f : 0.0f;
        return;
    }
    float seen = 0.0f;
    float total = 0.0f;
    float squares = 0.0f;
    float product = 1.0f;
    float truthy = 0.0f;
    float low = 0.0f;
    float high = 0.0f;
    float first_value = 0.0f;
    float last_value = 0.0f;
    bool started = false;
    for (int at = lo; at <= hi; ++at) {
        float value;
        float ok;
        if (tiled != 0u) {
            uint t = (uint)(at - origin);
            value = tile[t];
            ok = tile_mask[t];
        } else {
            value = values[at];
            bool good = !isnan(value) && (has_valid == 0u || valid[at] != 0.0f);
            ok = good ? 1.0f : 0.0f;
            if (!good) value = 0.0f;
        }
        if (ok == 0.0f) continue;
        seen += 1.0f;
        total += value;
        squares += value * value;
        product *= value;
        if (value != 0.0f) truthy += 1.0f;
        if (!started || value < low) low = value;
        if (!started || value > high) high = value;
        if (!started) first_value = value;
        last_value = value;
        started = true;
    }
    if (agg == AGG_COUNT) {
        out_values[gid] = seen;
        out_valid[gid] = 1.0f;
        return;
    }
    bool enough = seen > 0.0f && seen >= (float)min_periods;
    float answer = 0.0f;
    if (agg == AGG_SUM) answer = total;
    else if (agg == AGG_MEAN) answer = total / seen;
    else if (agg == AGG_MIN) answer = low;
    else if (agg == AGG_MAX) answer = high;
    else if (agg == AGG_PROD) answer = product;
    else if (agg == AGG_ANY) answer = truthy > 0.0f ? 1.0f : 0.0f;
    else if (agg == AGG_ALL) answer = truthy == seen ? 1.0f : 0.0f;
    else if (agg == AGG_FIRST) answer = first_value;
    else if (agg == AGG_LAST) answer = last_value;
    else {
        float degrees = seen - (float)ddof;
        if (degrees <= 0.0f) enough = false;
        else {
            float mean = total / seen;
            float spread = max((squares - mean * total) / degrees, 0.0f);
            if (agg == AGG_VAR) answer = spread;
            else if (agg == AGG_STD) answer = sqrt(spread);
            else answer = sqrt(spread / seen);
        }
    }
    out_values[gid] = enough ? answer : missing_value();
    out_valid[gid] = enough ? 1.0f : 0.0f;
}

/* The additive window statistics read off three inclusive scans.
 *
 * A window sum is a difference of two prefix entries however wide the window
 * is, so this costs one read per output against the direct path's w. It is not
 * free and it is not always better: the three scans are three more passes over
 * the column and three more allocations, and a difference of two running
 * totals cancels — a window whose own sum is small, taken from prefixes that
 * have grown to a million, has lost most of its digits. So the host takes the
 * direct path for narrow windows and this one for wide, and an expanding
 * window always comes here because its width is the whole column. */
kernel void roll_prefix(device const float* sums [[buffer(0)]],
                        device const float* squares [[buffer(1)]],
                        device const float* counts [[buffer(2)]],
                        device float* out_values [[buffer(3)]],
                        device float* out_valid [[buffer(4)]],
                        constant uint& count [[buffer(5)]],
                        constant uint& window [[buffer(6)]],
                        constant uint& min_periods [[buffer(7)]],
                        constant uint& lead [[buffer(8)]],
                        constant uint& agg [[buffer(9)]],
                        constant uint& ddof [[buffer(10)]],
                        constant uint& widening [[buffer(11)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    int last = widening != 0u ? (int)gid : (int)gid + (int)lead;
    int first = widening != 0u ? 0 : last - (int)window + 1;
    int lo = max(first, 0);
    int hi = min(last, (int)count - 1);
    if (hi < lo) {
        out_values[gid] = missing_value();
        out_valid[gid] = 0.0f;
        return;
    }
    float before_sum = lo > 0 ? sums[lo - 1] : 0.0f;
    float before_square = lo > 0 ? squares[lo - 1] : 0.0f;
    float before_count = lo > 0 ? counts[lo - 1] : 0.0f;
    float total = sums[hi] - before_sum;
    float square = squares[hi] - before_square;
    float seen = counts[hi] - before_count;
    if (agg == AGG_COUNT) {
        out_values[gid] = seen;
        out_valid[gid] = 1.0f;
        return;
    }
    bool enough = seen > 0.0f && seen >= (float)min_periods;
    float answer = 0.0f;
    if (agg == AGG_SUM) answer = total;
    else if (agg == AGG_MEAN) answer = total / seen;
    else {
        float degrees = seen - (float)ddof;
        if (degrees <= 0.0f) enough = false;
        else {
            float mean = total / seen;
            float spread = max((square - mean * total) / degrees, 0.0f);
            if (agg == AGG_VAR) answer = spread;
            else if (agg == AGG_STD) answer = sqrt(spread);
            else answer = sqrt(spread / seen);
        }
    }
    out_values[gid] = enough ? answer : missing_value();
    out_valid[gid] = enough ? 1.0f : 0.0f;
}

/* The value, the square and the presence flag a prefix window reads, in one
 * pass rather than three. */
kernel void roll_terms(device const float* values [[buffer(0)]],
                       device const float* wide [[buffer(1)]],
                       device const float* valid [[buffer(2)]],
                       device float* out_value [[buffer(3)]],
                       device float* out_square [[buffer(4)]],
                       device float* out_count [[buffer(5)]],
                       constant uint& count [[buffer(6)]],
                       constant uint& has_wide [[buffer(7)]],
                       constant uint& has_valid [[buffer(8)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    float raw = has_wide != 0u ? wide_number(values[gid], wide[gid]) : values[gid];
    bool present = !isnan(raw) && (has_valid == 0u || valid[gid] != 0.0f);
    float kept = present ? raw : 0.0f;
    out_value[gid] = kept;
    out_square[gid] = kept * kept;
    out_count[gid] = present ? 1.0f : 0.0f;
}

/* A column moved along its own index by `periods` rows.
 *
 * A gather with a constant offset rather than `take` over a positions lane:
 * the lane would be a whole extra allocation the length of the column, written
 * by one kernel so another could read it back. Both value lanes travel, so a
 * shifted `Int64` or `Datetime` column is still exact. */
kernel void lag_shift(device const float* src_values [[buffer(0)]],
                      device const float* src_wide [[buffer(1)]],
                      device const float* src_valid [[buffer(2)]],
                      device float* dst_values [[buffer(3)]],
                      device float* dst_wide [[buffer(4)]],
                      device float* dst_valid [[buffer(5)]],
                      constant uint& count [[buffer(6)]],
                      constant int& periods [[buffer(7)]],
                      constant uint& has_wide [[buffer(8)]],
                      constant uint& has_valid [[buffer(9)]],
                      constant uint& fill_low_bits [[buffer(10)]],
                      constant uint& fill_high_bits [[buffer(11)]],
                      constant uint& has_fill [[buffer(12)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    int at = (int)gid - periods;
    if (at < 0 || at >= (int)count) {
        dst_values[gid] = has_fill != 0u ? as_type<float>(fill_low_bits) : missing_value();
        if (has_wide != 0u) {
            dst_wide[gid] = has_fill != 0u ? as_type<float>(fill_high_bits) : 0.0f;
        }
        dst_valid[gid] = has_fill != 0u ? 1.0f : 0.0f;
        return;
    }
    dst_values[gid] = src_values[at];
    if (has_wide != 0u) dst_wide[gid] = src_wide[at];
    dst_valid[gid] = has_valid != 0u ? src_valid[at] : 1.0f;
}

/* A row against the row `periods` back: their difference, or that difference
 * over the earlier row. Either is missing when either row is.
 *
 * A wide column subtracts in `long` and converts afterwards. Converting first
 * would be useless for the case that matters: a millisecond timestamp is about
 * 1.7e12, which a float32 resolves to the nearest 131072 milliseconds, so the
 * difference between two adjacent days would come out zero.
 *
 * A ratio is the difference over the earlier row rather than the quotient less
 * one, which is the same number and keeps its digits when the two rows are
 * close. */
kernel void lag_diff(device const float* values [[buffer(0)]],
                     device const float* wide [[buffer(1)]],
                     device const float* valid [[buffer(2)]],
                     device float* out_values [[buffer(3)]],
                     device float* out_valid [[buffer(4)]],
                     constant uint& count [[buffer(5)]],
                     constant int& periods [[buffer(6)]],
                     constant uint& has_wide [[buffer(7)]],
                     constant uint& has_valid [[buffer(8)]],
                     constant uint& ratio [[buffer(9)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    int at = (int)gid - periods;
    bool inside = at >= 0 && at < (int)count;
    bool here_ok = !isnan(values[gid]) && (has_valid == 0u || valid[gid] != 0.0f);
    bool there_ok = inside && !isnan(values[at])
        && (has_valid == 0u || valid[at] != 0.0f);
    if (!here_ok || !there_ok) {
        out_values[gid] = missing_value();
        out_valid[gid] = 0.0f;
        return;
    }
    float change;
    float earlier;
    if (has_wide != 0u) {
        long here = (long)wide[gid] * WIDE_SCALE + (long)values[gid];
        long there = (long)wide[at] * WIDE_SCALE + (long)values[at];
        change = (float)(here - there);
        earlier = (float)there;
    } else {
        change = values[gid] - values[at];
        earlier = values[at];
    }
    out_values[gid] = ratio != 0u ? change / earlier : change;
    out_valid[gid] = 1.0f;
}

/* Which digit the wanted rank falls in, and what is left of the rank once the
 * digits below it are accounted for.
 *
 * A thread to each target, walking 256 bins. Doing this on the host would mean
 * downloading a thousand counts a target and four round trips of latency
 * between the passes; here the whole search stays on the device and only the
 * answer comes back. `prefixes` carries at most 24 bits between passes, which
 * a float32 holds exactly; the last pass would make it 32, so it emits the
 * value instead. */
kernel void sel_pick(device const float* bins [[buffer(0)]],
                     device float* ranks [[buffer(1)]],
                     device float* prefixes [[buffer(2)]],
                     device float* out [[buffer(3)]],
                     constant uint& lanes [[buffer(4)]],
                     constant uint& final_pass [[buffer(5)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= lanes) return;
    float wanted = ranks[gid];
    uint base = gid * 256u;
    float running = 0.0f;
    uint digit = 255u;
    for (uint d = 0u; d < 256u; ++d) {
        float here = bins[base + d];
        if (running + here > wanted) {
            digit = d;
            break;
        }
        running += here;
    }
    if (final_pass != 0u) {
        out[gid] = order_float(((uint)prefixes[gid] << 8u) | digit);
        return;
    }
    ranks[gid] = wanted - running;
    prefixes[gid] = prefixes[gid] * 256.0f + (float)digit;
}
"""
