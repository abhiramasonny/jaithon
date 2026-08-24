#: Whole-column reductions, scans, and the moving-window forms of the same.
#:
#: Everything here collapses a column or runs along it. A reduction turns a
#: column into one `Value`; a scan turns it into another column of the same
#: length; a window turns it into a column of per-window answers. Group-wise
#: versions of all three live in `groupby`, and the time-based window offsets
#: in `timeseries`; what those modules share with this one is the `Agg`
#: vocabulary and `reduce_column`, so `df.sum()`, `df.groupby(k).sum()` and
#: `df.rolling(7).sum()` mean the same thing by construction.
#:
#: Two shapes of kernel do most of the work.
#:
#: A whole-column reduction is a two-stage tree, copied from jaicv's
#: `core/stats.jai` for its reason: a fixed grid of 256 groups strides the
#: column accumulating in registers, folds once per group, and a second tiny
#: dispatch folds the 256 partials — so a column of any height costs two
#: dispatches and one download of sixteen floats, rather than one boxed float
#: per row. Downloading the partials and adding them on the host cost jaicv
#: more than the reduction did, and a frame has many columns.
#:
#: A scan is a work-efficient Blelloch sweep over blocks of 256 with the block
#: totals scanned the same way and combined back, so 2^24 rows is three levels
#: of dispatch. `cumsum`, `cumprod`, `cummax` and `cummin` are the same kernel
#: under four operators.
#:
#: ```
#: let daily = pd.Series(pd.Column.from_floats([3.0, 1.0, 4.0, 1.0, 5.0]))
#: defer { daily.free() }
#: mean(daily).to_float()                   # 2.8
#: reduce_column(daily.values, Agg.Median)  # 3.0
#: let smooth = rolling(daily, 3).mean()
#: ```
import std.gpu as gpu
import std.math
from std.gpu import Buffer, Kernel
from std.math import NAN, is_nan
from .column import Agg, Column, DataFrame, Dtype, Index, Series, Value
from .column import NA
from .column import cached_kernel, f32_bits, positions_buffer, wide_pack, wide_unpack

#: Threads in a reduction group, and groups in the fixed reduction grid.
#:
#: 65536 threads walk a column in strides. 256 groups rather than jaicv's 1024
#: because the join here is one thread folding them serially — 256 dependent
#: adds is microseconds, and it keeps the join a dozen lines instead of a
#: second simd tree.
const RGROUP = 256
const RGROUPS = 256

#: Floats a stage-one group writes, and floats a folded column occupies.
const STATS = 8
const FOLDED = 16

#: Threads in a scan block. The block totals are scanned by the same kernel,
#: so 256 covers 2^24 rows in three levels.
const SCAN_GROUP = 256

#: Threads in a window group, and the widest tile one can stage.
#:
#: A group of 256 whose threads each read a window of `w` covers `255 + w`
#: rows between them, so a 1024-float tile carries every window up to 769.
const WIN_GROUP = 256
const WIN_TILE = 1024

#: Groups per pair in the pairwise pass, and threads in each.
const PGROUP = 256
const PCHUNKS = 32

#: Where the host stops dispatching a window kernel and starts scanning.
#:
#: The direct path reads `window` rows per output; the prefix path reads three
#: and pays three extra passes over the column to build the scans. The crossover
#: is where `window` first exceeds what those passes cost, which is a few tens
#: of rows. 64 is on the safe side of it: below 64 the direct path also keeps
#: more digits, because a window sum taken as a difference of two running totals
#: cancels once the totals have grown far past the window's own sum.
const PREFIX_WINDOW = 64

#: The most rows `Window.apply` will run a host callback over.
#:
#: One boxed call per output row: at a million rows that is minutes. Every
#: named aggregation dispatches a kernel instead, so the cap is a signpost to
#: them rather than a limit on what the module can do.
const APPLY_MAX_ROWS = 4096

const AGG_SUM = 0
const AGG_MEAN = 1
const AGG_MIN = 2
const AGG_MAX = 3
const AGG_COUNT = 4
const AGG_PROD = 5
const AGG_STD = 6
const AGG_VAR = 7
const AGG_MEDIAN = 8
const AGG_QUANTILE = 9
const AGG_ANY = 10
const AGG_ALL = 11
const AGG_SEM = 12
const AGG_FIRST = 13
const AGG_LAST = 14
const AGG_SIZE = 15
const AGG_ARGMIN = 16
const AGG_ARGMAX = 17
const AGG_SKEW = 18
const AGG_KURT = 19

const SCAN_ADD = 0
const SCAN_MUL = 1
const SCAN_MAX = 2
const SCAN_MIN = 3

let SOURCE = """#include <metal_stdlib>
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

#: Everything one pass over a column learns about it.
#:
#: Every reduction here is arithmetic on one of these, which is why `describe`
#: — six statistics of every column — costs the same single round trip that
#: `mean` of one column does. The power sums are central about `center`; the
#: first pass leaves that at zero, so `total` is the sum and the rest are raw.
#:
#: Positions are row numbers, -1 where the column had nothing to report.
pub class ColumnStats {
    #: Rows in the column, missing ones included.
    pub var length: int
    #: Rows that hold a value.
    pub var present: int
    #: Sum of the present values less `center`.
    pub var total: float
    #: Sum of the squared deviations from `center`.
    pub var sum_squares: float
    #: Sum of the cubed deviations from `center`.
    pub var sum_cubes: float
    #: Sum of the fourth powers of the deviations from `center`.
    pub var sum_quads: float
    #: Product of the present values.
    pub var product: float
    #: How many present values are non-zero.
    pub var truthy: int
    #: Smallest present value, as a float.
    pub var lowest: float
    #: Largest present value, as a float.
    pub var highest: float
    #: The row `lowest` was found at, the earliest one on a tie.
    pub var lowest_at: int
    #: The row `highest` was found at, the earliest one on a tie.
    pub var highest_at: int
    #: First row holding a value.
    pub var first_present: int
    #: Last row holding a value.
    pub var last_present: int
    #: First row holding nothing, which is what a `skipna=false` reduction asks.
    pub var first_missing: int
    #: What the power sums were taken about.
    pub var center: float

    #: An empty summary. `column_stats` is the constructor.
    pub fn init(self) {
        self.length = 0
        self.present = 0
        self.total = 0.0
        self.sum_squares = 0.0
        self.sum_cubes = 0.0
        self.sum_quads = 0.0
        self.product = 1.0
        self.truthy = 0
        self.lowest = 0.0
        self.highest = 0.0
        self.lowest_at = -1
        self.highest_at = -1
        self.first_present = -1
        self.last_present = -1
        self.first_missing = -1
        self.center = 0.0
    }

    #: How many rows hold nothing.
    pub fn missing(self) -> int { return self.length - self.present }

    #: The arithmetic mean of the present values, NaN over none.
    pub fn mean(self) -> float {
        if self.present == 0 { return NAN }
        return self.center + self.total / float(self.present)
    }

    #: The variance about the mean, with `ddof` degrees of freedom spent.
    #:
    #: Off the centred second moment, corrected for the gap between `center`
    #: and the mean it turned out to be. Clamped at zero because that
    #: correction can land a hair below it on a column of identical values.
    pub fn variance(self, ddof: int) -> float {
        let degrees = self.present - ddof
        if self.present == 0 or degrees <= 0 { return NAN }
        let offset = self.total / float(self.present)
        let centred = self.sum_squares - offset * self.total
        return max(centred, 0.0) / float(degrees)
    }
}

fn require_device(what: str) -> void {
    if not gpu.is_available() { throw RuntimeError(f"jaiframe needs a GPU to {what}") }
}

#: Whether a lane can be read four floats at a time.
#:
#: Needs no validity lane — the quad path has no place to put a mask test —
#: and a row count that divides by four. A frame straight out of a reader with
#: no missing data is the case this is for.
fn packed_run(length: int, valid: Buffer?) -> bool {
    return valid is null and length > 0 and length % 4 == 0
}

#: Each column summarised, in one round trip for all of them.
#:
#: Two stage-one dispatches and one join per kind, then one download of sixteen
#: floats a column. Batching matters because the wait is what costs: a frame
#: reduced a column at a time pays the round trip once per column and spends
#: more on the bus than on the columns.
fn batched_stats(
    lanes: list[Buffer],
    valids: list[Buffer?],
    length: int,
    centers: list[float]
) -> list[ColumnStats] {
    let width = lanes.len()
    var out: list[ColumnStats] = []
    for at in 0..width {
        let empty = ColumnStats()
        empty.length = length
        empty.center = centers[at]
        out.push(empty)
    }
    if width == 0 or length == 0 { return out }
    require_device("reduce a column")
    let moments = Buffer(width * RGROUPS * STATS)
    let extremes = Buffer(width * RGROUPS * STATS)
    let folded = Buffer(width * FOLDED)
    for (at, lane) in lanes.enumerate() {
        let mask = valids[at]
        let quad = packed_run(length, mask)
        kernel_for("red_moments").dispatch_async(
            [lane, mask ?? lane, moments],
            RGROUPS * RGROUP,
            [length, flag(mask is not null), f32_bits(centers[at]), at, flag(quad)],
            group_size: RGROUP
        )
        kernel_for("red_extrema").dispatch_async(
            [lane, mask ?? lane, extremes],
            RGROUPS * RGROUP,
            [length, flag(mask is not null), at, flag(quad)],
            group_size: RGROUP
        )
    }
    kernel_for("red_join_moments").dispatch_async([moments, folded], width * STATS, [width])
    kernel_for("red_join_extrema").dispatch_async([extremes, folded], width, [width])
    let numbers = folded.download()
    moments.free()
    extremes.free()
    folded.free()
    for at in 0..width {
        let base = at * FOLDED
        var stats = out[at]
        stats.present = int(numbers[base + 0])
        stats.total = numbers[base + 1]
        stats.sum_squares = numbers[base + 2]
        stats.sum_cubes = numbers[base + 3]
        stats.sum_quads = numbers[base + 4]
        stats.product = numbers[base + 5]
        stats.truthy = int(numbers[base + 6])
        stats.lowest = numbers[base + 8]
        stats.highest = numbers[base + 9]
        stats.lowest_at = int(numbers[base + 10])
        stats.highest_at = int(numbers[base + 11])
        stats.first_present = int(numbers[base + 12])
        stats.last_present = int(numbers[base + 13])
        stats.first_missing = int(numbers[base + 14])
    }
    return out
}

#: The float lanes of every column, and which of them the caller must free.
#:
#: A wide column has to be flattened to be reduced, and that costs an
#: allocation; a plain one already is its own lane and must not be freed here.
fn borrow_lanes(columns: list[Column]) -> (list[Buffer], list[Buffer?], list[bool]) {
    var lanes: list[Buffer] = []
    var masks: list[Buffer?] = []
    var owned: list[bool] = []
    for column in columns {
        let borrowed = column.borrow_float_lane()
        lanes.push(borrowed[0])
        masks.push(column.valid)
        owned.push(borrowed[1])
    }
    return (lanes, masks, owned)
}

fn release_lanes(lanes: list[Buffer], owned: list[bool]) -> void {
    for (at, lane) in lanes.enumerate() {
        if owned[at] { lane.free() }
    }
}

#: One column summarised. `center` is what the power sums are taken about.
pub fn column_stats(column: Column, center: float = 0.0) -> ColumnStats {
    let borrowed = borrow_lanes([column])
    let summary = batched_stats(borrowed[0], borrowed[1], column.len(), [center])
    release_lanes(borrowed[0], borrowed[2])
    return summary[0]
}

#: Groups per target in a radix pass. 16384 threads is enough to saturate the
#: read while keeping the merge into 256 device bins one write per group.
const SEL_GROUPS = 64

#: The k-th smallest present value of one column of a packed matrix, for each
#: (column, rank) the caller names.
#:
#: Four radix passes narrow the answer from 32 bits of ordered key to none, and
#: the digit search between the passes runs on the device too, so the whole
#: thing is eight dispatches and one download whatever the row count. Nothing
#: sorts: a quantile wants one value out of a column, and ordering the other
#: million to find it is work the answer does not need. Targets are batched
#: because `describe` asks for three quantiles of every column and the pass
#: over the rows is the same pass for all of them.
fn select_order_stats(
    matrix: Buffer,
    mask: Buffer?,
    rows: int,
    columns: list[int],
    ranks: list[int]
) -> list[float] {
    let lanes = columns.len()
    if lanes == 0 or rows == 0 { return [] }
    require_device("find an order statistic")
    let targets = Buffer(lanes)
    targets.upload(columns.map(|at| float(at)))
    let wanted = Buffer(lanes)
    wanted.upload(ranks.map(|k| float(k)))
    let prefixes = Buffer(lanes)
    prefixes.fill_zero()
    let bins = Buffer(lanes * 256)
    let answers = Buffer(lanes)
    for pass in 0..4 {
        bins.fill_zero()
        kernel_for("sel_histogram").dispatch_async(
            [matrix, mask ?? matrix, targets, prefixes, bins],
            lanes * SEL_GROUPS * RGROUP,
            [rows, SEL_GROUPS, 24 - pass * 8, flag(mask is not null)],
            group_size: RGROUP
        )
        kernel_for("sel_pick").dispatch_async(
            [bins, wanted, prefixes, answers],
            lanes,
            [lanes, flag(pass == 3)]
        )
    }
    let out = answers.download()
    targets.free()
    wanted.free()
    prefixes.free()
    bins.free()
    answers.free()
    return out
}

#: The two order statistics a quantile sits between, interpolated linearly.
#:
#: pandas' default interpolation. Both ranks go into one batched search, so a
#: quantile costs what one order statistic costs.
fn quantile_from(
    matrix: Buffer,
    mask: Buffer?,
    rows: int,
    at: int,
    present: int,
    q: float
) -> float {
    if present == 0 { return NAN }
    if q < 0.0 or q > 1.0 { throw ValueError(f"a quantile is between 0 and 1, not {q}") }
    let position = q * float(present - 1)
    let lower = int(math.floor(position))
    let fraction = position - float(lower)
    let upper = if lower + 1 < present { lower + 1 } else { lower }
    let picks = select_order_stats(matrix, mask, rows, [at, at], [lower, upper])
    return picks[0] + fraction * (picks[1] - picks[0])
}

fn column_quantile(column: Column, present: int, q: float) -> float {
    let borrowed = column.borrow_float_lane()
    let answer = quantile_from(borrowed[0], column.valid, column.len(), 0, present, q)
    if borrowed[1] { borrowed[0].free() }
    return answer
}

#: Whether an aggregation means anything over a dtype, with the reason when it
#: does not.
#:
#: The rules are pandas': a count counts anything, an extreme or an edge picks
#: one of the inputs and so works over text, and arithmetic needs numbers. A
#: timestamp is the awkward one — its middle is a timestamp and its spread is a
#: duration, and durations are the timeseries module's business, not this one's.
fn check_agg(dtype: Dtype, agg: Agg) -> void {
    if agg == Agg.Nunique {
        throw ValueError("nunique counts distinct values through a hash table, which is keys.jai")
    }
    if agg == Agg.Count or agg == Agg.Size or agg == Agg.First or agg == Agg.Last { return }
    if dtype.is_dictionary() {
        if agg == Agg.Min or agg == Agg.Max {
            throw TypeError(f"a {dtype.name()} extreme orders the level table, which keys.jai owns")
        }
        throw TypeError(f"{agg.name()} is not defined over {dtype.name()}")
    }
    if dtype == Dtype.Datetime {
        let extreme = agg == Agg.Min or agg == Agg.Max
        let middle = agg == Agg.Mean or agg == Agg.Median or agg == Agg.Quantile
        if extreme or middle { return }
        throw TypeError(f"{agg.name()} of a timestamp is a duration, which datetime64[ms] is not")
    }
}

fn agg_accepts(dtype: Dtype, agg: Agg) -> bool {
    if agg == Agg.Nunique { return false }
    if agg == Agg.Count or agg == Agg.Size or agg == Agg.First or agg == Agg.Last { return true }
    if dtype.is_dictionary() { return false }
    if dtype == Dtype.Datetime {
        let extreme = agg == Agg.Min or agg == Agg.Max
        let middle = agg == Agg.Mean or agg == Agg.Median or agg == Agg.Quantile
        return extreme or middle
    }
    return true
}

#: What dtype a reduction's answer is stored as.
#:
#: Counts are exact integers and say so. An extreme or an edge is one of the
#: inputs and keeps their dtype. Everything else is a `Float64`, even over an
#: `Int64` column where pandas would hand back an int64 — the sums are
#: accumulated in float32 lanes and past 2^24 they are not that integer, and a
#: dtype that admits it is better than one that is quietly wrong.
fn result_dtype(dtype: Dtype, agg: Agg) -> Dtype {
    if agg.counts_rows() { return Dtype.Int64 }
    if agg == Agg.Any or agg == Agg.All { return Dtype.Bool }
    if agg.picks_a_row() { return dtype }
    if dtype == Dtype.Datetime { return Dtype.Datetime }
    return Dtype.Float64
}

fn number_cell(dtype: Dtype, value: float) -> Value {
    if is_nan(value) { return Value.null_of(dtype) }
    if dtype == Dtype.Datetime { return Value.of_datetime(int(math.round(value))) }
    if dtype == Dtype.Int64 { return Value.of_int(int(math.round(value))) }
    if dtype == Dtype.Bool { return Value.of_bool(value != 0.0) }
    return Value.of_float(value)
}

#: The cell at a row, or NA when the reduction found no row at all.
#:
#: Goes back to the column rather than using the float the reduction carried,
#: because for an `Int64` or a `Datetime` column that float came off the
#: flattened lane and is only exact to 2^24. One extra short download, and only
#: for the aggregations that pick a row.
fn cell_at(column: Column, row: int) -> Value {
    if row < 0 { return Value.null_of(column.dtype) }
    return column.get(row)
}

#: The row a two-lane column's extreme sits at, folded in `long`.
#:
#: One dispatch and one download of 768 floats, over the same fixed grid the
#: float reduction uses. The 256 group winners are folded on the host because
#: 256 of them is microseconds and a second kernel to save it would be a second
#: round trip.
fn wide_extreme_row(column: Column, want_high: bool) -> int {
    require_device("find the extreme of a two-lane column")
    let partials = Buffer(RGROUPS * 3)
    let high = column.wide ?? column.values
    let mask = column.valid ?? column.values
    kernel_for("red_wide_extreme").dispatch_async(
        [column.values, high, mask, partials],
        RGROUPS * RGROUP,
        [column.len(), flag(column.valid is not null), flag(want_high)],
        group_size: RGROUP
    )
    let numbers = partials.download()
    partials.free()
    var best = 0
    var row = -1
    for group in 0..RGROUPS {
        let at = int(numbers[group * 3 + 2])
        if at < 0 { continue }
        let value = wide_unpack(numbers[group * 3 + 0], numbers[group * 3 + 1])
        var wins = row < 0
        if not wins { wins = if want_high { value > best } else { value < best } }
        if not wins { wins = value == best and at < row }
        if wins {
            best = value
            row = at
        }
    }
    return row
}

#: The row an extreme sits at, exact whatever the column stores.
#:
#: An `Int64` or a `Datetime` column is two float32 lanes, and the summary was
#: folded from the one lane they flatten to — which past 2^24 is not the
#: number, so the rows that share a float32 all tie and the summary reports the
#: first of them. Those columns pay one more pass to be told which row it
#: really was; every other column already knows.
fn extreme_row(column: Column, stats: ColumnStats, want_high: bool) -> int {
    let folded = if want_high { stats.highest_at } else { stats.lowest_at }
    if folded < 0 or not column.dtype.is_wide() { return folded }
    return wide_extreme_row(column, want_high)
}

fn empty_result(dtype: Dtype, agg: Agg) -> Value {
    if agg == Agg.Count or agg == Agg.Size { return Value.of_int(0) }
    if agg == Agg.Sum { return Value.of_float(0.0) }
    if agg == Agg.Prod { return Value.of_float(1.0) }
    if agg == Agg.Any { return Value.of_bool(false) }
    if agg == Agg.All { return Value.of_bool(true) }
    return Value.null_of(result_dtype(dtype, agg))
}

#: A reduction finished off a summary that has already been computed.
#:
#: Split out from `reduce_column` so a frame can summarise every column in one
#: round trip and then finish them one at a time without going back.
fn finish_reduction(
    column: Column,
    stats: ColumnStats,
    agg: Agg,
    skipna: bool,
    ddof: int,
    q: float,
    quantile_value: float
) -> Value {
    if agg == Agg.Count { return Value.of_int(stats.present) }
    if agg == Agg.Size { return Value.of_int(stats.length) }
    let target = result_dtype(column.dtype, agg)
    if not skipna and stats.missing() > 0 { return Value.null_of(target) }
    if agg == Agg.Min { return cell_at(column, extreme_row(column, stats, false)) }
    if agg == Agg.Max { return cell_at(column, extreme_row(column, stats, true)) }
    if agg == Agg.First { return cell_at(column, stats.first_present) }
    if agg == Agg.Last { return cell_at(column, stats.last_present) }
    if agg == Agg.Any { return Value.of_bool(stats.truthy > 0) }
    if agg == Agg.All { return Value.of_bool(stats.truthy == stats.present) }
    if agg == Agg.Sum { return Value.of_float(stats.total) }
    if agg == Agg.Prod { return Value.of_float(stats.product) }
    if agg == Agg.Median or agg == Agg.Quantile {
        return number_cell(target, quantile_value)
    }
    if stats.present == 0 { return Value.null_of(target) }
    if agg == Agg.Mean { return number_cell(target, stats.mean()) }
    let spread = stats.variance(ddof)
    if agg == Agg.Var { return number_cell(target, spread) }
    if is_nan(spread) { return Value.null_of(target) }
    return number_cell(target, math.sqrt(spread))
}

#: The `agg` of one column, as one cell.
#:
#: This is the call groupby, resample and pivot_table name too, so that
#: `Agg.Mean` means the same arithmetic wherever it is written. `ddof` is spent
#: by `Std` and `Var`; `q` is read only by `Quantile`.
pub fn reduce_column(
    column: Column,
    agg: Agg,
    skipna: bool = true,
    ddof: int = 1,
    q: float = 0.5
) -> Value {
    check_agg(column.dtype, agg)
    if agg == Agg.Size { return Value.of_int(column.len()) }
    if column.len() == 0 { return empty_result(column.dtype, agg) }
    let stats = column_stats(column)
    var picked = NAN
    if agg == Agg.Median or agg == Agg.Quantile {
        picked = column_quantile(column, stats.present, if agg == Agg.Median { 0.5 } else { q })
    }
    return finish_reduction(column, stats, agg, skipna, ddof, q, picked)
}

#: The columns in one column-major matrix, with a presence matrix beside it.
#:
#: Column-major so that the width-th read of neighbouring rows is contiguous,
#: which is what the per-row and pairwise kernels do. `centers` is subtracted
#: as the copy happens; a covariance pass passes the column means there and a
#: per-row pass passes zeros.
fn pack_frame(columns: list[Column], rows: int, centers: list[float]) -> (Buffer, Buffer) {
    let width = columns.len()
    require_device("pack a frame")
    let matrix = Buffer(max(width * rows, 1))
    let mask = Buffer(max(width * rows, 1))
    for (at, column) in columns.enumerate() {
        if rows == 0 { continue }
        kernel_for("red_pack").dispatch_async(
            [
                column.values,
                column.wide ?? column.values,
                column.valid ?? column.values,
                matrix,
                mask,
            ],
            rows,
            [
                rows,
                at * rows,
                flag(column.dtype.is_wide()),
                flag(column.valid is not null),
                f32_bits(centers[at]),
            ]
        )
    }
    return (matrix, mask)
}

fn zeros(length: int) -> list[float] {
    var out: list[float] = []
    for _at in 0..length { out.push(0.0) }
    return out
}

#: A column of cells the host already holds, in the dtype they share.
fn cells_column(cells: list[Value], dtype: Dtype) -> Column {
    if dtype.is_dictionary() {
        var labels: list[str?] = []
        for cell in cells { labels.push(if cell.is_na() { null } else { cell.to_str() }) }
        return Column.from_strs(labels, dtype)
    }
    var gaps: list[bool] = []
    for cell in cells { gaps.push(cell.is_na()) }
    if dtype.is_wide() {
        var numbers: list[int] = []
        for cell in cells { numbers.push(if cell.is_na() { 0 } else { cell.to_int() }) }
        return Column.from_ints(numbers, gaps, dtype)
    }
    if dtype == Dtype.Bool {
        var truths: list[bool] = []
        for cell in cells { truths.push(not cell.is_na() and cell.to_bool()) }
        return Column.from_bools(truths, gaps)
    }
    var numbers: list[float] = []
    for cell in cells { numbers.push(cell.to_float()) }
    return Column.from_floats(numbers)
}

#: The one dtype a per-column result can be stored in.
#:
#: `Float64` absorbs a mix of numbers. Text does not mix with anything, because
#: the alternative is pandas' object column and everything downstream then has
#: to ask what is really in it.
fn common_dtype(dtypes: list[Dtype], agg: Agg) -> Dtype {
    if dtypes.len() == 0 { return Dtype.Float64 }
    var chosen = result_dtype(dtypes[0], agg)
    for dtype in dtypes {
        let here = result_dtype(dtype, agg)
        if here == chosen { continue }
        if chosen.is_dictionary() or here.is_dictionary() {
            throw TypeError(
                f"{agg.name()} over both {chosen.name()} and {here.name()} has no one dtype"
            )
        }
        chosen = Dtype.Float64
    }
    return chosen
}

#: A series whose labels are column names, which is what a per-column
#: reduction of a frame is indexed by.
fn named_series(names: list[str], values: Column) -> Series {
    var labels: list[str?] = []
    for name in names { labels.push(name) }
    var titles: list[str?] = [null]
    return Series(values, index: Index.of([Column.from_strs(labels)], titles))
}

#: Which columns an aggregation is defined over, in frame order.
fn accepted_columns(frame: DataFrame, agg: Agg) -> list[int] {
    var chosen: list[int] = []
    for (at, column) in frame.columns.enumerate() {
        if agg_accepts(column.dtype, agg) { chosen.push(at) }
    }
    return chosen
}

#: Every column reduced, in one round trip for the summaries and one more for
#: the quantiles when the aggregation is one.
fn frame_axis0(frame: DataFrame, agg: Agg, skipna: bool, ddof: int, q: float) -> Series {
    if agg == Agg.Nunique {
        throw ValueError("nunique counts distinct values through a hash table, which is keys.jai")
    }
    let chosen = accepted_columns(frame, agg)
    var names: list[str] = []
    var columns: list[Column] = []
    for at in chosen {
        names.push(frame.names[at])
        columns.push(frame.columns[at])
    }
    let rows = frame.len()
    var cells: list[Value] = []
    if agg == Agg.Size {
        for _at in chosen { cells.push(Value.of_int(rows)) }
        return named_series(names, cells_column(cells, Dtype.Int64))
    }
    let borrowed = borrow_lanes(columns)
    let summaries = batched_stats(borrowed[0], borrowed[1], rows, zeros(columns.len()))
    release_lanes(borrowed[0], borrowed[2])
    var picks: list[float] = []
    if agg == Agg.Median or agg == Agg.Quantile {
        picks = frame_quantiles(columns, summaries, rows, if agg == Agg.Median { 0.5 } else { q })
    } else {
        picks = zeros(columns.len())
    }
    for (at, column) in columns.enumerate() {
        if rows == 0 {
            cells.push(empty_result(column.dtype, agg))
            continue
        }
        cells.push(finish_reduction(column, summaries[at], agg, skipna, ddof, q, picks[at]))
    }
    return named_series(names, cells_column(cells, common_dtype(columns.map(|c| c.dtype), agg)))
}

#: One quantile of every column, batched into a single radix search.
fn frame_quantiles(
    columns: list[Column],
    summaries: list[ColumnStats],
    rows: int,
    q: float
) -> list[float] {
    if columns.len() == 0 or rows == 0 { return zeros(columns.len()) }
    if q < 0.0 or q > 1.0 { throw ValueError(f"a quantile is between 0 and 1, not {q}") }
    let packed = pack_frame(columns, rows, zeros(columns.len()))
    var lanes: list[int] = []
    var ranks: list[int] = []
    var fractions: list[float] = []
    for (at, summary) in summaries.enumerate() {
        let present = max(summary.present, 1)
        let position = q * float(present - 1)
        let lower = int(math.floor(position))
        fractions.push(position - float(lower))
        lanes.push(at)
        ranks.push(lower)
        lanes.push(at)
        ranks.push(if lower + 1 < present { lower + 1 } else { lower })
    }
    let found = select_order_stats(packed[0], packed[1], rows, lanes, ranks)
    packed[0].free()
    packed[1].free()
    var out: list[float] = []
    for (at, summary) in summaries.enumerate() {
        if summary.present == 0 {
            out.push(NAN)
            continue
        }
        let low = found[at * 2]
        let high = found[at * 2 + 1]
        out.push(low + fractions[at] * (high - low))
    }
    return out
}

fn agg_code(agg: Agg) -> int {
    return match agg {
        Agg.Sum => AGG_SUM,
        Agg.Mean => AGG_MEAN,
        Agg.Min => AGG_MIN,
        Agg.Max => AGG_MAX,
        Agg.Count => AGG_COUNT,
        Agg.Size => AGG_SIZE,
        Agg.Prod => AGG_PROD,
        Agg.Std => AGG_STD,
        Agg.Var => AGG_VAR,
        Agg.Median => AGG_MEDIAN,
        Agg.Quantile => AGG_QUANTILE,
        Agg.First => AGG_FIRST,
        Agg.Last => AGG_LAST,
        Agg.Any => AGG_ANY,
        Agg.All => AGG_ALL,
        _ => throw ValueError(f"{agg.name()} has no across-the-row form here"),
    }
}

#: The numeric columns of a frame, which is what a per-row pass reads.
fn numeric_columns(frame: DataFrame) -> (list[str], list[Column]) {
    var names: list[str] = []
    var columns: list[Column] = []
    for (at, column) in frame.columns.enumerate() {
        if not column.dtype.is_numeric() { continue }
        names.push(frame.names[at])
        columns.push(column)
    }
    return (names, columns)
}

#: One value per row, reduced across the frame's numeric columns.
fn frame_axis1(frame: DataFrame, code: int, skipna: bool, ddof: int, q: float) -> Column {
    let picked = numeric_columns(frame)
    let columns = picked[1]
    let rows = frame.len()
    let out = Column.allocate(Dtype.Float64, rows, with_validity: true)
    if rows == 0 { return out }
    if columns.len() == 0 {
        out.values.fill(NAN)
        (out.valid ?? out.values).fill_zero()
        return out
    }
    let packed = pack_frame(columns, rows, zeros(columns.len()))
    kernel_for("red_axis").dispatch_async(
        [packed[0], packed[1], out.values, out.valid ?? out.values],
        rows,
        [rows, columns.len(), code, ddof, f32_bits(q), flag(skipna)]
    )
    packed[0].free()
    packed[1].free()
    return out
}

#: A float lane re-read as an exact `Int64` column, for the counts.
fn as_int_column(source: Column) -> Column {
    let out = Column.allocate(Dtype.Int64, source.len())
    if source.len() == 0 { return out }
    kernel_for("red_split_wide").dispatch_async(
        [source.values, out.values, out.wide ?? out.values],
        source.len(),
        [source.len()]
    )
    return out
}

fn dispatch_reduce(target: any, agg: Agg, axis: int, skipna: bool, ddof: int, q: float) -> any {
    if target is Column { return reduce_column(target, agg, skipna, ddof, q) }
    if target is Series { return reduce_column(target.values, agg, skipna, ddof, q) }
    if target is not DataFrame {
        throw TypeError("a reduction takes a Column, a Series or a DataFrame")
    }
    if axis == 0 { return frame_axis0(target, agg, skipna, ddof, q) }
    if axis != 1 { throw ValueError(f"axis {axis} is neither 0 nor 1") }
    let raw = frame_axis1(target, agg_code(agg), skipna, ddof, q)
    if not agg.counts_rows() { return Series(raw, index: target.index.copy()) }
    let counted = as_int_column(raw)
    raw.free()
    return Series(counted, index: target.index.copy())
}

#: The three statistics that are arithmetic on a summary but have no `Agg`
#: spelling, because group-by does not name them.
const EXTRA_SEM = 0
const EXTRA_SKEW = 1
const EXTRA_KURT = 2

#: A shape statistic off a first pass and a pass centred on its mean.
#:
#: pandas' bias-corrected forms, which are scipy's with `bias=False`. Skew
#: needs three present rows and kurtosis four, because below that the
#: correction divides by zero — pandas answers NaN there and so does this.
fn extra_from(summary: ColumnStats, shaped: ColumnStats?, kind: int, ddof: int) -> Value {
    let rows = summary.present
    if kind == EXTRA_SEM {
        let spread = summary.variance(ddof)
        if rows == 0 or is_nan(spread) { return Value.null_of(Dtype.Float64) }
        return Value.of_float(math.sqrt(spread / float(rows)))
    }
    if shaped is null { return Value.null_of(Dtype.Float64) }
    let needed = if kind == EXTRA_SKEW { 3 } else { 4 }
    if rows < needed { return Value.null_of(Dtype.Float64) }
    let spread = shaped.sum_squares / float(rows - 1)
    if not spread > 0.0 { return Value.null_of(Dtype.Float64) }
    if kind == EXTRA_SKEW {
        let deviation = math.sqrt(spread)
        let scaled = shaped.sum_cubes / (deviation * deviation * deviation)
        return Value.of_float(float(rows) / float((rows - 1) * (rows - 2)) * scaled)
    }
    let lead = float(rows * (rows + 1)) / float((rows - 1) * (rows - 2) * (rows - 3))
    let tail = 3.0 * float((rows - 1) * (rows - 1)) / float((rows - 2) * (rows - 3))
    return Value.of_float(lead * (shaped.sum_quads / (spread * spread)) - tail)
}

fn column_extra(column: Column, kind: int, skipna: bool, ddof: int) -> Value {
    if not column.dtype.is_numeric() {
        throw TypeError(f"a shape statistic needs numbers, not {column.dtype.name()}")
    }
    if column.len() == 0 { return Value.null_of(Dtype.Float64) }
    let first = column_stats(column)
    if not skipna and first.missing() > 0 { return Value.null_of(Dtype.Float64) }
    if kind == EXTRA_SEM { return extra_from(first, null, kind, ddof) }
    return extra_from(first, column_stats(column, center: first.mean()), kind, ddof)
}

fn frame_extra(frame: DataFrame, kind: int, skipna: bool, ddof: int) -> Series {
    let picked = numeric_columns(frame)
    let columns = picked[1]
    let rows = frame.len()
    let borrowed = borrow_lanes(columns)
    let first = batched_stats(borrowed[0], borrowed[1], rows, zeros(columns.len()))
    var shaped: list[ColumnStats] = []
    if kind != EXTRA_SEM {
        shaped = batched_stats(borrowed[0], borrowed[1], rows, first.map(|one| one.mean()))
    }
    release_lanes(borrowed[0], borrowed[2])
    var cells: list[Value] = []
    for (at, summary) in first.enumerate() {
        if not skipna and summary.missing() > 0 {
            cells.push(Value.null_of(Dtype.Float64))
            continue
        }
        let second = if kind == EXTRA_SEM { null } else { shaped[at] }
        cells.push(extra_from(summary, second, kind, ddof))
    }
    return named_series(picked[0], cells_column(cells, Dtype.Float64))
}

fn dispatch_extra(target: any, kind: int, axis: int, skipna: bool, ddof: int) -> any {
    if target is Column { return column_extra(target, kind, skipna, ddof) }
    if target is Series { return column_extra(target.values, kind, skipna, ddof) }
    if target is not DataFrame {
        throw TypeError("a shape statistic takes a Column, a Series or a DataFrame")
    }
    if axis == 0 { return frame_extra(target, kind, skipna, ddof) }
    if axis != 1 { throw ValueError(f"axis {axis} is neither 0 nor 1") }
    let code = if kind == EXTRA_SEM {
        AGG_SEM
    } elif kind == EXTRA_SKEW {
        AGG_SKEW
    } else {
        AGG_KURT
    }
    return Series(frame_axis1(target, code, skipna, ddof, 0.5), index: target.index.copy())
}

#: The sum of the present values. 0 over a column that has none, as pandas is.
pub fn sum(target: any, axis: int = 0, skipna: bool = true) -> any {
    return dispatch_reduce(target, Agg.Sum, axis, skipna, 1, 0.5)
}

#: The arithmetic mean of the present values.
pub fn mean(target: any, axis: int = 0, skipna: bool = true) -> any {
    return dispatch_reduce(target, Agg.Mean, axis, skipna, 1, 0.5)
}

#: The smallest present value. Named for `min` being a builtin.
pub fn minimum(target: any, axis: int = 0, skipna: bool = true) -> any {
    return dispatch_reduce(target, Agg.Min, axis, skipna, 1, 0.5)
}

#: The largest present value. Named for `max` being a builtin.
pub fn maximum(target: any, axis: int = 0, skipna: bool = true) -> any {
    return dispatch_reduce(target, Agg.Max, axis, skipna, 1, 0.5)
}

#: The product of the present values. 1 over a column that has none.
pub fn product(target: any, axis: int = 0, skipna: bool = true) -> any {
    return dispatch_reduce(target, Agg.Prod, axis, skipna, 1, 0.5)
}

#: How many rows hold a value. Never missing, and never affected by `skipna` —
#: counting the present ones is what it is for.
pub fn count(target: any, axis: int = 0) -> any {
    return dispatch_reduce(target, Agg.Count, axis, true, 1, 0.5)
}

#: The sample standard deviation, one degree of freedom by default.
pub fn std(target: any, axis: int = 0, skipna: bool = true, ddof: int = 1) -> any {
    return dispatch_reduce(target, Agg.Std, axis, skipna, ddof, 0.5)
}

#: The sample variance. Spelled out because `var` is a keyword here.
pub fn variance(target: any, axis: int = 0, skipna: bool = true, ddof: int = 1) -> any {
    return dispatch_reduce(target, Agg.Var, axis, skipna, ddof, 0.5)
}

#: The standard error of the mean: the deviation over the root of the count.
pub fn sem(target: any, axis: int = 0, skipna: bool = true, ddof: int = 1) -> any {
    return dispatch_extra(target, EXTRA_SEM, axis, skipna, ddof)
}

#: The middle present value, interpolated between the two middles when the
#: count is even.
pub fn median(target: any, axis: int = 0, skipna: bool = true) -> any {
    return dispatch_reduce(target, Agg.Median, axis, skipna, 1, 0.5)
}

#: The value `q` of the way through the present ones, interpolated linearly
#: between its neighbours the way pandas does by default.
pub fn quantile(target: any, q: float = 0.5, axis: int = 0, skipna: bool = true) -> any {
    return dispatch_reduce(target, Agg.Quantile, axis, skipna, 1, q)
}

#: The bias-corrected third standardised moment. NaN under three present rows.
pub fn skew(target: any, axis: int = 0, skipna: bool = true) -> any {
    return dispatch_extra(target, EXTRA_SKEW, axis, skipna, 1)
}

#: The bias-corrected excess kurtosis, Fisher's — zero for a normal sample.
#: NaN under four present rows.
pub fn kurtosis(target: any, axis: int = 0, skipna: bool = true) -> any {
    return dispatch_extra(target, EXTRA_KURT, axis, skipna, 1)
}

#: Whether any present value is truthy. Named for `any` being a type.
pub fn any_(target: any, axis: int = 0, skipna: bool = true) -> any {
    return dispatch_reduce(target, Agg.Any, axis, skipna, 1, 0.5)
}

#: Whether every present value is truthy. True over none, as pandas is.
pub fn all_(target: any, axis: int = 0, skipna: bool = true) -> any {
    return dispatch_reduce(target, Agg.All, axis, skipna, 1, 0.5)
}

#: The labels at a list of rows, `MISSING_POSITION` giving a null.
#:
#: A multi-index gives its first level, which is what a single answer per
#: column can carry.
fn label_column(index: Index, rows: list[int]) -> Column {
    let levels = index.materialize()
    let lane = positions_buffer(rows)
    let out = levels[0].take(lane, rows.len(), allow_missing: true)
    lane.free()
    for level in levels { level.free() }
    return out
}

fn label_of(index: Index, row: int) -> Value {
    if row < 0 { return NA }
    return index.label_at(row)[0]
}

#: The label of the row holding the extreme, per column or per row.
fn dispatch_arg(target: any, want_high: bool, axis: int, skipna: bool) -> any {
    if target is Column {
        let stats = column_stats(target)
        if not skipna and stats.missing() > 0 { return NA }
        let row = extreme_row(target, stats, want_high)
        return if row < 0 { NA } else { Value.of_int(row) }
    }
    if target is Series {
        let stats = column_stats(target.values)
        if not skipna and stats.missing() > 0 { return NA }
        return label_of(target.index, extreme_row(target.values, stats, want_high))
    }
    if target is not DataFrame {
        throw TypeError("an extreme's label needs a Column, a Series or a DataFrame")
    }
    if axis == 1 {
        return Series(argmax_names(target, want_high, skipna), index: target.index.copy())
    }
    if axis != 0 { throw ValueError(f"axis {axis} is neither 0 nor 1") }
    let picked = numeric_columns(target)
    let columns = picked[1]
    let borrowed = borrow_lanes(columns)
    let summaries = batched_stats(borrowed[0], borrowed[1], target.len(), zeros(columns.len()))
    release_lanes(borrowed[0], borrowed[2])
    var rows: list[int] = []
    for (at, summary) in summaries.enumerate() {
        if not skipna and summary.missing() > 0 {
            rows.push(-1)
            continue
        }
        rows.push(extreme_row(columns[at], summary, want_high))
    }
    return named_series(picked[0], label_column(target.index, rows))
}

#: The name of the column holding each row's extreme, as a text column whose
#: level table is the frame's own column names.
fn argmax_names(frame: DataFrame, want_high: bool, skipna: bool) -> Column {
    let picked = numeric_columns(frame)
    let rows = frame.len()
    let code = if want_high { AGG_ARGMAX } else { AGG_ARGMIN }
    let raw = frame_axis1(frame, code, skipna, 1, 0.5)
    let out = Column.allocate(Dtype.Str, rows, with_validity: true)
    out.set_levels(picked[0])
    if rows > 0 {
        cached_kernel("col_copy_lane").dispatch_async([raw.values, out.values], rows, [rows, 0, 0])
        cached_kernel("col_copy_lane").dispatch_async(
            [raw.valid ?? raw.values, out.valid ?? out.values],
            rows,
            [rows, 0, 0]
        )
    }
    raw.free()
    return out
}

#: The label of the row holding the smallest value.
pub fn idxmin(target: any, axis: int = 0, skipna: bool = true) -> any {
    return dispatch_arg(target, false, axis, skipna)
}

#: The label of the row holding the largest value.
pub fn idxmax(target: any, axis: int = 0, skipna: bool = true) -> any {
    return dispatch_arg(target, true, axis, skipna)
}

fn frame_edge(frame: DataFrame, want_last: bool) -> int {
    let borrowed = borrow_lanes(frame.columns)
    let summaries = batched_stats(borrowed[0], borrowed[1], frame.len(), zeros(frame.width()))
    release_lanes(borrowed[0], borrowed[2])
    var edge = -1
    for summary in summaries {
        let here = if want_last { summary.last_present } else { summary.first_present }
        if here < 0 { continue }
        if edge < 0 {
            edge = here
        } elif want_last {
            edge = max(edge, here)
        } else {
            edge = min(edge, here)
        }
    }
    return edge
}

#: The label of the first row that holds a value, NA when none does.
#:
#: Over a frame this is the first row where any column holds one, which is
#: pandas' meaning and is the smallest of the per-column answers — so it comes
#: out of the same single pass that summarises the columns.
pub fn first_valid_index(target: any) -> Value {
    if target is Column {
        let row = column_stats(target).first_present
        return if row < 0 { NA } else { Value.of_int(row) }
    }
    if target is Series { return label_of(target.index, column_stats(target.values).first_present) }
    if target is not DataFrame {
        throw TypeError("an edge needs a Column, a Series or a DataFrame")
    }
    return label_of(target.index, frame_edge(target, false))
}

#: The label of the last row that holds a value, NA when none does.
pub fn last_valid_index(target: any) -> Value {
    if target is Column {
        let row = column_stats(target).last_present
        return if row < 0 { NA } else { Value.of_int(row) }
    }
    if target is Series { return label_of(target.index, column_stats(target.values).last_present) }
    if target is not DataFrame {
        throw TypeError("an edge needs a Column, a Series or a DataFrame")
    }
    return label_of(target.index, frame_edge(target, true))
}

#: An inclusive or exclusive scan of a lane, into a buffer the caller owns.
#:
#: Blocks of 256 swept in threadgroup memory, then the block totals scanned by
#: the same kernel and combined back — so 2^24 rows is three levels of dispatch
#: rather than a host loop over sixty-five thousand block totals. Asking each
#: level for its total would put a device wait between the levels, which is why
#: nothing here downloads anything.
fn scan_into(input: Buffer, out: Buffer, length: int, op: int, inclusive: bool) -> void {
    if length <= 0 { return }
    require_device("scan a column")
    let blocks = (length + SCAN_GROUP - 1) // SCAN_GROUP
    let totals = Buffer(blocks)
    kernel_for("scan_block").dispatch_async(
        [input, out, totals],
        blocks * SCAN_GROUP,
        [length, op, flag(inclusive)],
        group_size: SCAN_GROUP
    )
    if blocks > 1 {
        let carried = Buffer(blocks)
        scan_into(totals, carried, blocks, op, false)
        kernel_for("scan_carry").dispatch_async([out, carried], length, [length, op])
        carried.free()
    }
    totals.free()
}

fn scan_op(agg: Agg) -> int {
    return match agg {
        Agg.Sum => SCAN_ADD,
        Agg.Prod => SCAN_MUL,
        Agg.Max => SCAN_MAX,
        Agg.Min => SCAN_MIN,
        _ => throw ValueError(f"{agg.name()} does not scan; a scan needs an associative operator"),
    }
}

#: A running `agg` down one column.
#:
#: The answer is a `Float64` even over an `Int64` column, for the reason the
#: reductions give: the running total is accumulated in a float32 lane. A
#: missing row keeps the running value where it was and stays missing itself;
#: with `skipna` off, it and everything after it are missing, which the scan
#: gets for free by putting a NaN in the lane instead of the operator's
#: identity.
pub fn scan_column(column: Column, agg: Agg, skipna: bool = true) -> Column {
    if not column.dtype.is_numeric() and column.dtype != Dtype.Datetime {
        throw TypeError(f"a running {agg.name()} needs numbers, not {column.dtype.name()}")
    }
    let op = scan_op(agg)
    let rows = column.len()
    let out = Column.allocate(Dtype.Float64, rows, with_validity: true)
    if rows == 0 { return out }
    let terms = Buffer(rows)
    kernel_for("scan_prepare").dispatch_async(
        [column.values, column.wide ?? column.values, column.valid ?? column.values, terms],
        rows,
        [rows, flag(column.dtype.is_wide()), flag(column.valid is not null), op, flag(skipna)]
    )
    scan_into(terms, out.values, rows, op, true)
    terms.free()
    kernel_for("scan_finish").dispatch_async(
        [out.values, column.valid ?? out.values, out.valid ?? out.values],
        rows,
        [rows, flag(column.valid is not null), flag(skipna)]
    )
    return out
}

#: How many rows up to and including each one hold nothing.
#:
#: An inclusive sum-scan of the missing flags, which are exact integers under
#: 2^24 and so exact in a float32 lane. Never missing itself — every row has an
#: answer, including the missing ones.
pub fn cumcount_na_column(column: Column) -> Column {
    let rows = column.len()
    let out = Column.allocate(Dtype.Int64, rows)
    if rows == 0 { return out }
    let flags = Buffer(rows)
    kernel_for("scan_missing_flags").dispatch_async(
        [column.values, column.valid ?? column.values, flags],
        rows,
        [rows, flag(column.valid is not null)]
    )
    let running = Buffer(rows)
    scan_into(flags, running, rows, SCAN_ADD, true)
    kernel_for("red_split_wide").dispatch_async(
        [running, out.values, out.wide ?? out.values],
        rows,
        [rows]
    )
    flags.free()
    running.free()
    return out
}

#: A column-wise transform lifted to whatever the caller handed in.
#:
#: A frame keeps only the columns the transform is defined over — pandas
#: concatenates text under `cumsum` and raises under `cummax`, and neither is
#: a shape a later join can rely on.
fn dispatch_column_wise(target: any, body: fn(Column) -> Column, numeric_only: bool) -> any {
    if target is Column { return body(target) }
    if target is Series {
        return Series(body(target.values), index: target.index.copy(), name: target.name)
    }
    if target is not DataFrame { throw TypeError("this takes a Column, a Series or a DataFrame") }
    var names: list[str] = []
    var columns: list[Column] = []
    for (at, column) in target.columns.enumerate() {
        if numeric_only and not column.dtype.is_numeric() and column.dtype != Dtype.Datetime {
            continue
        }
        names.push(target.names[at])
        columns.push(body(column))
    }
    return DataFrame(names, columns, index: target.index.copy())
}

#: The running sum down each column.
pub fn cumsum(target: any, skipna: bool = true) -> any {
    return dispatch_column_wise(target, |column| scan_column(column, Agg.Sum, skipna), true)
}

#: The running product down each column.
pub fn cumprod(target: any, skipna: bool = true) -> any {
    return dispatch_column_wise(target, |column| scan_column(column, Agg.Prod, skipna), true)
}

#: The running largest value down each column.
pub fn cummax(target: any, skipna: bool = true) -> any {
    return dispatch_column_wise(target, |column| scan_column(column, Agg.Max, skipna), true)
}

#: The running smallest value down each column.
pub fn cummin(target: any, skipna: bool = true) -> any {
    return dispatch_column_wise(target, |column| scan_column(column, Agg.Min, skipna), true)
}

#: The running count of missing rows down each column.
pub fn cumcount_na(target: any) -> any {
    return dispatch_column_wise(target, |column| cumcount_na_column(column), false)
}

#: One column moved along its index by `periods` rows.
#:
#: Both value lanes travel, so a shifted `Int64` or `Datetime` column is still
#: exact, and a text column keeps its level table. The rows that fall off the
#: end are `fill_value`, or missing when there is none.
pub fn shift_column(column: Column, periods: int = 1, fill_value: Value? = null) -> Column {
    let rows = column.len()
    let out = Column.allocate(column.dtype, rows, with_validity: true)
    if column.dtype.is_dictionary() {
        out.set_levels([name for name in column.levels ?? []])
    }
    if rows == 0 { return out }
    var low = 0.0
    var high = 0.0
    var has_fill = false
    let fill = fill_value
    if fill is not null and not fill.is_na() {
        has_fill = true
        if column.dtype.is_dictionary() {
            low = float(out.intern_level(fill.to_str()))
        } elif column.dtype.is_wide() {
            let split = wide_pack(fill.to_int())
            low = split[0]
            high = split[1]
        } else {
            low = fill.to_float()
        }
    }
    kernel_for("lag_shift").dispatch_async(
        [
            column.values,
            column.wide ?? column.values,
            column.valid ?? column.values,
            out.values,
            out.wide ?? out.values,
            out.valid ?? out.values,
        ],
        rows,
        [
            rows,
            u32(periods),
            flag(column.dtype.is_wide()),
            flag(column.valid is not null),
            f32_bits(low),
            f32_bits(high),
            flag(has_fill),
        ]
    )
    return out
}

fn change_column(column: Column, periods: int, ratio: bool) -> Column {
    let numeric = column.dtype.is_numeric() or column.dtype == Dtype.Datetime
    if not numeric {
        throw TypeError(f"a difference needs numbers, not {column.dtype.name()}")
    }
    if ratio and column.dtype == Dtype.Datetime {
        throw TypeError("the ratio of two timestamps is not a quantity; take the difference")
    }
    let rows = column.len()
    let out = Column.allocate(Dtype.Float64, rows, with_validity: true)
    if rows == 0 { return out }
    kernel_for("lag_diff").dispatch_async(
        [
            column.values,
            column.wide ?? column.values,
            column.valid ?? column.values,
            out.values,
            out.valid ?? out.values,
        ],
        rows,
        [
            rows,
            u32(periods),
            flag(column.dtype.is_wide()),
            flag(column.valid is not null),
            flag(ratio),
        ]
    )
    return out
}

#: Every value moved `periods` rows later, or earlier when negative.
pub fn shift(target: any, periods: int = 1, fill_value: Value? = null) -> any {
    return dispatch_column_wise(target, |column| shift_column(column, periods, fill_value), false)
}

#: Each row less the row `periods` back. A `Datetime` column gives the gap in
#: milliseconds, because a duration is not a timestamp.
pub fn diff(target: any, periods: int = 1) -> any {
    return dispatch_column_wise(target, |column| change_column(column, periods, false), true)
}

#: The fractional change from the row `periods` back.
pub fn pct_change(target: any, periods: int = 1) -> any {
    return dispatch_column_wise(target, |column| change_column(column, periods, true), true)
}

fn percent_label(q: float) -> str {
    let scaled = q * 100.0
    if scaled == math.round(scaled) { return f"{int(math.round(scaled))}%" }
    return f"{scaled}%"
}

#: The count, the middle, the spread and the shape of every numeric column.
#:
#: The rows are pandas': count, mean, std, min, the percentiles, max. The
#: summaries come out of one round trip for the whole frame, and the
#: percentiles out of one batched radix search for every column and every
#: percentile together — so a frame of twenty columns described at three
#: percentiles is two round trips, not a hundred and twenty.
pub fn describe(frame: DataFrame, percentiles: list[float]? = null) -> DataFrame {
    let wanted = percentiles ?? [0.25, 0.5, 0.75]
    for q in wanted {
        if q < 0.0 or q > 1.0 { throw ValueError(f"a percentile is between 0 and 1, not {q}") }
    }
    let picked = numeric_columns(frame)
    let columns = picked[1]
    let rows = frame.len()
    let borrowed = borrow_lanes(columns)
    let summaries = batched_stats(borrowed[0], borrowed[1], rows, zeros(columns.len()))
    release_lanes(borrowed[0], borrowed[2])
    var found: list[float] = []
    var fractions: list[float] = []
    if columns.len() > 0 and rows > 0 {
        let packed = pack_frame(columns, rows, zeros(columns.len()))
        var lanes: list[int] = []
        var ranks: list[int] = []
        for (at, summary) in summaries.enumerate() {
            for q in wanted {
                let present = max(summary.present, 1)
                let position = q * float(present - 1)
                let lower = int(math.floor(position))
                fractions.push(position - float(lower))
                lanes.push(at)
                ranks.push(lower)
                lanes.push(at)
                ranks.push(if lower + 1 < present { lower + 1 } else { lower })
            }
        }
        found = select_order_stats(packed[0], packed[1], rows, lanes, ranks)
        packed[0].free()
        packed[1].free()
    }
    var stats: list[str?] = ["count", "mean", "std", "min"]
    for q in wanted { stats.push(percent_label(q)) }
    stats.push("max")
    var out: list[Column] = []
    for (at, summary) in summaries.enumerate() {
        var numbers: list[float] = []
        numbers.push(float(summary.present))
        numbers.push(summary.mean())
        let spread = summary.variance(1)
        numbers.push(if is_nan(spread) { NAN } else { math.sqrt(spread) })
        numbers.push(if summary.present == 0 { NAN } else { summary.lowest })
        for (which, _q) in wanted.enumerate() {
            let slot = (at * wanted.len() + which) * 2
            if summary.present == 0 or slot + 1 >= found.len() {
                numbers.push(NAN)
                continue
            }
            numbers.push(
                found[slot] + fractions[at * wanted.len() + which] * (found[slot + 1] - found[slot])
            )
        }
        numbers.push(if summary.present == 0 { NAN } else { summary.highest })
        out.push(Column.from_floats(numbers))
    }
    var titles: list[str?] = [null]
    return DataFrame(picked[0], out, index: Index.of([Column.from_strs(stats)], titles))
}

#: The co-moments of every named pair, in one pass over a packed matrix.
#:
#: The matrix is centred on each column's own mean first. A covariance from raw
#: sums subtracts two numbers of the same size and float32 has little left
#: underneath; shifting a column by a constant changes no covariance, so the
#: shortcut formula stays exact in algebra and becomes usable in arithmetic.
#: The centres come from the same batched summary the caller would have wanted
#: anyway.
fn pairwise_moments(columns: list[Column], rows: int, pairs: list[int]) -> list[float] {
    let lanes = pairs.len() // 2
    if lanes == 0 or rows == 0 or columns.len() == 0 { return [] }
    let borrowed = borrow_lanes(columns)
    let summaries = batched_stats(borrowed[0], borrowed[1], rows, zeros(columns.len()))
    release_lanes(borrowed[0], borrowed[2])
    var centers: list[float] = []
    for summary in summaries {
        let middle = summary.mean()
        centers.push(if is_nan(middle) { 0.0 } else { middle })
    }
    let packed = pack_frame(columns, rows, centers)
    let table = Buffer(pairs.len())
    table.upload(pairs.map(|at| float(at)))
    let partials = Buffer(lanes * PCHUNKS * STATS)
    kernel_for("red_pairwise").dispatch_async(
        [packed[0], packed[1], table, partials],
        lanes * PCHUNKS * PGROUP,
        [rows, PCHUNKS],
        group_size: PGROUP
    )
    let folded = Buffer(lanes * STATS)
    kernel_for("red_join_pairwise").dispatch_async(
        [partials, folded],
        lanes * STATS,
        [lanes, PCHUNKS]
    )
    let out = folded.download()
    packed[0].free()
    packed[1].free()
    table.free()
    partials.free()
    folded.free()
    return out
}

fn check_correlation_method(method: str) -> void {
    if method == "pearson" { return }
    if method == "spearman" {
        throw ValueError(
            "spearman is pearson over ranks; rank the frame with keys.jai and correlate that"
        )
    }
    if method == "kendall" {
        throw ValueError("kendall counts concordant pairs, which is quadratic and is not here")
    }
    throw ValueError(f"unknown correlation method {method}")
}

#: The correlation or covariance one folded pair carries.
fn pair_value(
    moments: list[float],
    slot: int,
    min_periods: int,
    ddof: int,
    correlate: bool
) -> float {
    let base = slot * STATS
    let rows = moments[base]
    if rows < float(min_periods) or rows <= 0.0 { return NAN }
    let cxy = moments[base + 5] - moments[base + 1] * moments[base + 2] / rows
    let cxx = moments[base + 3] - moments[base + 1] * moments[base + 1] / rows
    let cyy = moments[base + 4] - moments[base + 2] * moments[base + 2] / rows
    if correlate {
        if cxx <= 0.0 or cyy <= 0.0 { return NAN }
        return cxy / math.sqrt(cxx * cyy)
    }
    let degrees = rows - float(ddof)
    if degrees <= 0.0 { return NAN }
    return cxy / degrees
}

fn square_matrix(frame: DataFrame, correlate: bool, min_periods: int, ddof: int) -> DataFrame {
    let picked = numeric_columns(frame)
    let columns = picked[1]
    let width = columns.len()
    var pairs: list[int] = []
    var slots: dict[int, int] = {}
    for left in 0..width {
        for right in left..width {
            slots[left * width + right] = pairs.len() // 2
            pairs.push(left)
            pairs.push(right)
        }
    }
    let moments = pairwise_moments(columns, frame.len(), pairs)
    var out: list[Column] = []
    for right in 0..width {
        var numbers: list[float] = []
        for left in 0..width {
            let low = min(left, right)
            let high = max(left, right)
            let slot = slots.get(low * width + high, -1)
            if slot < 0 or (slot + 1) * STATS > moments.len() {
                numbers.push(NAN)
                continue
            }
            numbers.push(pair_value(moments, slot, min_periods, ddof, correlate))
        }
        out.push(Column.from_floats(numbers))
    }
    var labels: list[str?] = []
    for name in picked[0] { labels.push(name) }
    var titles: list[str?] = [null]
    return DataFrame(picked[0], out, index: Index.of([Column.from_strs(labels)], titles))
}

#: The correlation of every pair of numeric columns.
#:
#: Pairwise-complete, as pandas is: a row counts for a pair only when both of
#: its columns hold a value, so two entries of the matrix can rest on different
#: numbers of rows. Only `pearson` is here — `spearman` is `pearson` of the
#: ranks, and ranking is keys.jai's.
pub fn corr(frame: DataFrame, method: str = "pearson", min_periods: int = 1) -> DataFrame {
    check_correlation_method(method)
    return square_matrix(frame, true, min_periods, 1)
}

#: The covariance of every pair of numeric columns, `ddof` degrees spent.
pub fn cov(frame: DataFrame, min_periods: int = 1, ddof: int = 1) -> DataFrame {
    return square_matrix(frame, false, min_periods, ddof)
}

#: The correlation of each of this frame's columns with its counterpart.
#:
#: Against a `Series`, every column is correlated with that one series.
#: Against a `DataFrame`, columns are matched by name; `drop` says whether a
#: name that only one side has is left out or reported as missing.
pub fn corrwith(
    frame: DataFrame,
    other: any,
    method: str = "pearson",
    drop: bool = false,
    min_periods: int = 1
) -> Series {
    check_correlation_method(method)
    let picked = numeric_columns(frame)
    var names: list[str] = []
    var columns: list[Column] = []
    var pairs: list[int] = []
    if other is Series {
        if other.len() != frame.len() {
            throw ValueError(f"a series of {other.len()} rows cannot align with {frame.len()}")
        }
        for column in picked[1] { columns.push(column) }
        columns.push(other.values)
        for (at, name) in picked[0].enumerate() {
            names.push(name)
            pairs.push(at)
            pairs.push(picked[1].len())
        }
    } elif other is DataFrame {
        if other.len() != frame.len() {
            throw ValueError(f"a frame of {other.len()} rows cannot align with {frame.len()}")
        }
        for column in picked[1] { columns.push(column) }
        var missing: list[str] = []
        for (at, name) in picked[0].enumerate() {
            if not other.has(name) {
                if not drop { missing.push(name) }
                continue
            }
            names.push(name)
            pairs.push(at)
            pairs.push(columns.len())
            columns.push(other.column(name))
        }
        for name in missing { names.push(name) }
    } else {
        throw TypeError("corrwith needs a Series or a DataFrame to correlate against")
    }
    let moments = pairwise_moments(columns, frame.len(), pairs)
    var numbers: list[float] = []
    for at in 0..names.len() {
        if (at + 1) * STATS > moments.len() {
            numbers.push(NAN)
            continue
        }
        numbers.push(pair_value(moments, at, min_periods, 1, true))
    }
    return named_series(names, Column.from_floats(numbers))
}

#: The most rows an expanding window will answer a per-window search over.
#:
#: The aggregations that cannot be scanned — a median, a quantile, a truth test
#: — have one thread read the whole history of its row, which over a whole
#: column is quadratic. Everything a scan can answer is routed to one and has
#: no cap; this is the boundary of what is left.
const EXPAND_DIRECT_MAX = 8192

fn window_code(agg: Agg) -> int {
    if agg == Agg.Sum or agg == Agg.Mean or agg == Agg.Min or agg == Agg.Max {
        return agg_code(agg)
    }
    if agg == Agg.Count or agg == Agg.Prod or agg == Agg.Std or agg == Agg.Var {
        return agg_code(agg)
    }
    if agg == Agg.Median or agg == Agg.Quantile { return agg_code(agg) }
    if agg == Agg.Any or agg == Agg.All or agg == Agg.First or agg == Agg.Last {
        return agg_code(agg)
    }
    throw ValueError(f"{agg.name()} has no window form")
}

fn additive_code(code: int) -> bool {
    let plain = code == AGG_SUM or code == AGG_MEAN or code == AGG_COUNT
    let spread = code == AGG_STD or code == AGG_VAR or code == AGG_SEM
    return plain or spread
}

fn scannable_code(code: int) -> int {
    if code == AGG_MIN { return SCAN_MIN }
    if code == AGG_MAX { return SCAN_MAX }
    if code == AGG_PROD { return SCAN_MUL }
    if code == AGG_SUM { return SCAN_ADD }
    return -1
}

#: The running count of present rows, which several window paths gate on.
fn presence_scan(column: Column, rows: int) -> Buffer {
    let flags = Buffer(rows)
    let squares = Buffer(rows)
    let values = Buffer(rows)
    kernel_for("roll_terms").dispatch_async(
        [
            column.values,
            column.wide ?? column.values,
            column.valid ?? column.values,
            values,
            squares,
            flags,
        ],
        rows,
        [rows, flag(column.dtype.is_wide()), flag(column.valid is not null)]
    )
    let tally = Buffer(rows)
    scan_into(flags, tally, rows, SCAN_ADD, true)
    flags.free()
    squares.free()
    values.free()
    return tally
}

#: Window statistics off three inclusive scans.
fn window_by_prefix(
    column: Column,
    code: int,
    window: int,
    min_periods: int,
    lead: int,
    widening: bool,
    ddof: int
) -> Column {
    let rows = column.len()
    let out = Column.allocate(Dtype.Float64, rows, with_validity: true)
    let values = Buffer(rows)
    let squares = Buffer(rows)
    let flags = Buffer(rows)
    kernel_for("roll_terms").dispatch_async(
        [
            column.values,
            column.wide ?? column.values,
            column.valid ?? column.values,
            values,
            squares,
            flags,
        ],
        rows,
        [rows, flag(column.dtype.is_wide()), flag(column.valid is not null)]
    )
    let sums = Buffer(rows)
    let sum_squares = Buffer(rows)
    let tally = Buffer(rows)
    scan_into(values, sums, rows, SCAN_ADD, true)
    scan_into(squares, sum_squares, rows, SCAN_ADD, true)
    scan_into(flags, tally, rows, SCAN_ADD, true)
    kernel_for("roll_prefix").dispatch_async(
        [sums, sum_squares, tally, out.values, out.valid ?? out.values],
        rows,
        [rows, window, min_periods, lead, code, ddof, flag(widening)]
    )
    values.free()
    squares.free()
    flags.free()
    sums.free()
    sum_squares.free()
    tally.free()
    return out
}

#: Window statistics with one thread per output row, each walking its window.
fn window_direct(
    column: Column,
    code: int,
    window: int,
    min_periods: int,
    lead: int,
    widening: bool,
    ddof: int,
    q: float
) -> Column {
    let rows = column.len()
    let out = Column.allocate(Dtype.Float64, rows, with_validity: true)
    let borrowed = column.borrow_float_lane()
    let lane = borrowed[0]
    let mask = column.valid
    let blocks = (rows + WIN_GROUP - 1) // WIN_GROUP
    let tiled = not widening and window - 1 + WIN_GROUP <= WIN_TILE
    kernel_for("roll_direct").dispatch_async(
        [lane, mask ?? lane, out.values, out.valid ?? out.values],
        blocks * WIN_GROUP,
        [
            rows,
            window,
            min_periods,
            lead,
            code,
            ddof,
            f32_bits(q),
            flag(mask is not null),
            flag(widening),
            flag(tiled),
        ],
        group_size: WIN_GROUP
    )
    if borrowed[1] { lane.free() }
    return out
}

#: An expanding minimum, maximum or product, off the scan that is already the
#: answer, with the rows that have seen too little blanked afterwards.
fn window_by_scan(column: Column, code: int, min_periods: int) -> Column {
    let rows = column.len()
    let out = Column.allocate(Dtype.Float64, rows, with_validity: true)
    let op = scannable_code(code)
    let terms = Buffer(rows)
    kernel_for("scan_prepare").dispatch_async(
        [column.values, column.wide ?? column.values, column.valid ?? column.values, terms],
        rows,
        [rows, flag(column.dtype.is_wide()), flag(column.valid is not null), op, 1]
    )
    scan_into(terms, out.values, rows, op, true)
    terms.free()
    let tally = presence_scan(column, rows)
    (out.valid ?? out.values).fill(1.0)
    kernel_for("win_gate").dispatch_async(
        [out.values, out.valid ?? out.values, tally],
        rows,
        [rows, min_periods]
    )
    tally.free()
    return out
}

#: One column's window statistic, by whichever of the three paths fits.
#:
#: The direct path reads `window` rows per output and stages its group's rows
#: in threadgroup memory, so it is one device read per row however wide the
#: window; the prefix path reads three per output but pays three scans of the
#: column to build them. Narrow windows go direct — fewer passes, fewer
#: allocations, and no cancellation between two running totals that have grown
#: far past the window's own sum. Wide ones and every expanding window go
#: through the prefix path, where the direct path's per-output cost would grow
#: without bound. An expanding extreme or product is neither: the scan that
#: answers it is the answer.
fn window_column(
    column: Column,
    code: int,
    window: int,
    min_periods: int,
    center: bool,
    widening: bool,
    ddof: int,
    q: float
) -> Column {
    if not column.dtype.is_numeric() {
        throw TypeError(f"a window statistic needs numbers, not {column.dtype.name()}")
    }
    let rows = column.len()
    if rows == 0 { return Column.allocate(Dtype.Float64, 0, with_validity: true) }
    let lead = if center { (window - 1) // 2 } else { 0 }
    if additive_code(code) {
        if widening or window >= PREFIX_WINDOW {
            return window_by_prefix(column, code, window, min_periods, lead, widening, ddof)
        }
        return window_direct(column, code, window, min_periods, lead, widening, ddof, q)
    }
    if widening and scannable_code(code) >= 0 {
        return window_by_scan(column, code, min_periods)
    }
    if widening and rows > EXPAND_DIRECT_MAX {
        throw ValueError(
            f"an expanding window of that kind reads its whole history per row, which over {rows} rows is quadratic; the sums, extremes and products are scanned instead"
        )
    }
    return window_direct(column, code, window, min_periods, lead, widening, ddof, q)
}

#: A window over a series or a frame, waiting to be told which statistic.
#:
#: pandas hands one of these back from `rolling` and `expanding` so that the
#: window and the aggregation can be written apart, and so that one window can
#: answer several. It owns no device memory: every method dispatches from the
#: target it was built over, which must outlive it.
pub class Window {
    #: The `Column`, `Series` or `DataFrame` the window runs over.
    pub var target: any
    #: How many rows a window spans. 0 for an expanding one, whose span is the
    #: history.
    pub var window: int
    #: How many present rows a window needs before it answers.
    pub var min_periods: int
    #: Whether the answer is labelled at the middle of the window rather than
    #: at its end.
    pub var center: bool
    #: Whether the window starts at row zero and grows.
    pub var widening: bool

    #: `rolling` and `expanding` are the constructors.
    pub fn init(
        self,
        target: any,
        window: int,
        min_periods: int,
        center: bool = false,
        widening: bool = false
    ) {
        self.target = target
        self.window = window
        self.min_periods = min_periods
        self.center = center
        self.widening = widening
    }

    #: The named aggregation over every window.
    #:
    #: The same `Agg` the whole-column reductions and group-by take, so
    #: `rolling(7).agg(Agg.Mean)` and `mean(series)` are the same arithmetic
    #: over different spans.
    pub fn agg(self, kind: Agg, ddof: int = 1, q: float = 0.5) -> any {
        let code = window_code(kind)
        let span = self.window
        let least = self.min_periods
        let middle = self.center
        let growing = self.widening
        return dispatch_column_wise(
            self.target,
            |column| window_column(column, code, span, least, middle, growing, ddof, q),
            true
        )
    }

    #: The sum over every window.
    pub fn sum(self) -> any { return self.agg(Agg.Sum) }

    #: The mean over every window.
    pub fn mean(self) -> any { return self.agg(Agg.Mean) }

    #: The smallest value in every window.
    pub fn min(self) -> any { return self.agg(Agg.Min) }

    #: The largest value in every window.
    pub fn max(self) -> any { return self.agg(Agg.Max) }

    #: The sample standard deviation over every window.
    pub fn std(self, ddof: int = 1) -> any { return self.agg(Agg.Std, ddof: ddof) }

    #: The sample variance over every window. Spelled out because `var` is a
    #: keyword here.
    pub fn variance(self, ddof: int = 1) -> any { return self.agg(Agg.Var, ddof: ddof) }

    #: The middle value of every window.
    pub fn median(self) -> any { return self.agg(Agg.Median) }

    #: The value `q` of the way through every window.
    pub fn quantile(self, q: float = 0.5) -> any { return self.agg(Agg.Quantile, q: q) }

    #: How many rows of every window hold a value. Never missing.
    pub fn count(self) -> any { return self.agg(Agg.Count) }

    #: The present values of every window, handed to a callback.
    #:
    #: One boxed call per output row, on the host, which is why it is capped.
    #: Everything the named aggregations cover dispatches a kernel instead and
    #: has no cap; this is the escape hatch for the statistic that is not one
    #: of them, and it is worth writing a kernel rather than reaching for it on
    #: a column of any size.
    pub fn apply(self, body: fn(list[float]) -> float) -> any {
        let span = self.window
        let least = self.min_periods
        let middle = self.center
        let growing = self.widening
        return dispatch_column_wise(
            self.target,
            |column| apply_windows(column, body, span, least, middle, growing),
            true
        )
    }
}

fn apply_windows(
    column: Column,
    body: fn(list[float]) -> float,
    window: int,
    min_periods: int,
    center: bool,
    widening: bool
) -> Column {
    let rows = column.len()
    if rows > APPLY_MAX_ROWS {
        throw ValueError(
            f"apply runs a host callback once per row, and {rows} rows is past the {APPLY_MAX_ROWS} that is reasonable; a named aggregation dispatches a kernel instead"
        )
    }
    if rows == 0 { return Column.allocate(Dtype.Float64, 0, with_validity: true) }
    let numbers = column.to_floats()
    let lead = if center { (window - 1) // 2 } else { 0 }
    var out: list[float] = []
    for row in 0..rows {
        let last = if widening { row } else { row + lead }
        let first = if widening { 0 } else { last - window + 1 }
        let lo = max(first, 0)
        let hi = min(last, rows - 1)
        var slice: list[float] = []
        for at in lo..hi + 1 {
            if is_nan(numbers[at]) { continue }
            slice.push(numbers[at])
        }
        if slice.len() == 0 or slice.len() < min_periods {
            out.push(NAN)
            continue
        }
        out.push(body(slice))
    }
    return Column.from_floats(out)
}

#: A window of `window` rows ending at each row.
#:
#: `min_periods` defaults to the whole window, as pandas does, so the leading
#: rows are missing until one is full. `center` labels the answer at the middle
#: of the window instead of at its end.
pub fn rolling(target: any, window: int, min_periods: int? = null, center: bool = false) -> Window {
    if window <= 0 { throw ValueError(f"a rolling window of {window} rows has nothing in it") }
    let least = min_periods ?? window
    if least < 0 { throw ValueError(f"min_periods cannot be {least}") }
    return Window(target, window, least, center: center)
}

#: A window that starts at row zero and grows to each row.
pub fn expanding(target: any, min_periods: int = 1) -> Window {
    if min_periods < 0 { throw ValueError(f"min_periods cannot be {min_periods}") }
    return Window(target, 0, min_periods, widening: true)
}

#: The decay one of pandas' four spellings asks for.
fn ewm_alpha(alpha: float?, span: float?, halflife: float?) -> float {
    var given = 0
    if alpha is not null { given += 1 }
    if span is not null { given += 1 }
    if halflife is not null { given += 1 }
    if given != 1 {
        throw ValueError("ewm needs exactly one of alpha, span and halflife")
    }
    let direct = alpha
    if direct is not null {
        if direct <= 0.0 or direct > 1.0 { throw ValueError(f"alpha is in (0, 1], not {direct}") }
        return direct
    }
    let width = span
    if width is not null {
        if width < 1.0 { throw ValueError(f"a span is at least 1, not {width}") }
        return 2.0 / (width + 1.0)
    }
    let life = halflife ?? 0.0
    if life <= 0.0 { throw ValueError(f"a halflife is positive, not {life}") }
    return 1.0 - math.exp(-math.LN_2 / life)
}

#: The exponentially weighted recurrence, run as a scan.
#:
#: Both lanes decay by the same factor, so one pass carries them together, and
#: the block totals recur with the factor raised to the block width — which
#: underflows to zero for any brisk decay, and that is the right answer: a
#: block's carry-in really has no weight left by the time it reaches the end.
fn ewm_scan(nums: Buffer, dens: Buffer, length: int, decay: float) -> (Buffer, Buffer) {
    let out_num = Buffer(length)
    let out_den = Buffer(length)
    let blocks = (length + SCAN_GROUP - 1) // SCAN_GROUP
    let total_num = Buffer(blocks)
    let total_den = Buffer(blocks)
    kernel_for("ewm_block").dispatch_async(
        [nums, dens, out_num, out_den, total_num, total_den],
        blocks * SCAN_GROUP,
        [length, f32_bits(decay)],
        group_size: SCAN_GROUP
    )
    if blocks > 1 {
        let carried = ewm_scan(total_num, total_den, blocks, decay ** float(SCAN_GROUP))
        kernel_for("ewm_carry").dispatch_async(
            [out_num, out_den, carried[0], carried[1]],
            length,
            [length, f32_bits(decay)]
        )
        carried[0].free()
        carried[1].free()
    }
    total_num.free()
    total_den.free()
    return (out_num, out_den)
}

fn ewm_column(column: Column, alpha: float, adjust: bool) -> Column {
    if not column.dtype.is_numeric() {
        throw TypeError(f"an exponential mean needs numbers, not {column.dtype.name()}")
    }
    let rows = column.len()
    let out = Column.allocate(Dtype.Float64, rows, with_validity: true)
    if rows == 0 { return out }
    var seed = rows
    if not adjust {
        let found = column_stats(column).first_present
        if found >= 0 { seed = found }
    }
    let nums = Buffer(rows)
    let dens = Buffer(rows)
    kernel_for("ewm_prepare").dispatch_async(
        [column.values, column.wide ?? column.values, column.valid ?? column.values, nums, dens],
        rows,
        [
            rows,
            flag(column.dtype.is_wide()),
            flag(column.valid is not null),
            f32_bits(alpha),
            flag(adjust),
            seed,
        ]
    )
    let running = ewm_scan(nums, dens, rows, 1.0 - alpha)
    nums.free()
    dens.free()
    kernel_for("ewm_finish").dispatch_async(
        [running[0], running[1], out.values, out.valid ?? out.values],
        rows,
        [rows]
    )
    running[0].free()
    running[1].free()
    return out
}

#: The exponentially weighted mean, with the decay given any of pandas' ways.
#:
#: A missing row contributes nothing but does not stop the clock, which is
#: pandas' `ignore_na=False`: the weights follow position, not the count of
#: present rows. Rows before the first present one are missing, because there
#: is nothing yet to average.
#:
#: `adjust` chooses between the two conventions. With it, every present row
#: carries weight 1 and the answer is their weighted mean, so the early rows
#: are means of what has arrived; without it, the recurrence is
#: `y = (1 - alpha) y_before + alpha x` seeded at the first present row.
pub fn ewm(
    target: any,
    alpha: float? = null,
    span: float? = null,
    halflife: float? = null,
    adjust: bool = true
) -> any {
    let decay = ewm_alpha(alpha, span, halflife)
    return dispatch_column_wise(target, |column| ewm_column(column, decay, adjust), true)
}
