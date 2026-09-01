#: The shader library, and the scalars its dispatches take.
import std.gpu as gpu
from std.gpu import Kernel

#: Digits of a radix pass. Eight bits is four passes over a 32-bit lane, and
#: the local rank in `key_radix_scatter` costs the tile width per element — so
#: a narrower digit buys more passes than it saves scanning, and a wider one
#: needs a bucket array bigger than a threadgroup.
pub const RADIX_BITS = 8
pub const RADIX_SIZE = 256
pub const RADIX_GROUP = 256

pub const REDUCE_GROUP = 256
pub const REDUCE_GROUPS = 256

#: The code a row gets when its key is missing and the caller asked for those
#: rows to be dropped. pandas' `factorize` spells the same thing `-1`.
pub const NO_GROUP = -1

let SOURCE = """#include <metal_stdlib>
using namespace metal;

constexpr constant uint RADIX_SIZE = 256;
constexpr constant uint RADIX_GROUP = 256;
constexpr constant uint REDUCE_GROUP = 256;
constexpr constant uint WIDE_SCALE = 16777216u;
constexpr constant float FLOAT_CEILING = 3.402823e+38f;

/* Every lane of a packed key is uint bits living in a plain float32 `Buffer`.
 * The allocation is only bytes; which type it holds is the kernel's choice,
 * the same trick the Hough accumulators in jaicv/imgproc/features.jai use to
 * get an atomic counter out of a float buffer. Downloading one of these as
 * floats gives a pile of denormals, so nothing outside this file reads them. */

inline uint fmix32(uint h) {
    h ^= h >> 16;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13;
    h *= 0xC2B2AE35u;
    h ^= h >> 16;
    return h;
}

inline uint hash_combine(uint accumulated, uint incoming) {
    return accumulated ^ (incoming + 0x9E3779B9u + (accumulated << 6) + (accumulated >> 2));
}

/* The standard total order on IEEE floats read as unsigned integers: flip the
 * sign bit of a positive one, every bit of a negative one. That is what lets
 * one radix pass sort a float lane without ever comparing floats. -0.0 folds
 * onto 0.0 first, so the two group and sort as the one value a reader means. */
inline uint order_bits(float value) {
    float v = (value == 0.0f) ? 0.0f : value;
    uint bits = as_type<uint>(v);
    return ((bits & 0x80000000u) != 0u) ? ~bits : (bits | 0x80000000u);
}

/* Two rows are the same key when every lane agrees. That is the whole of row
 * equality here, and it is why the hash table needs no dtype: the encoding
 * already made a timestamp, a string and a float into the same thing. */
inline bool rows_equal(device const uint* packed, uint stride, uint lanes, uint a, uint b) {
    for (uint l = 0u; l < lanes; ++l) {
        uint at = l * stride;
        if (packed[at + a] != packed[at + b]) return false;
    }
    return true;
}

kernel void key_encode_plain(device const float* values [[buffer(0)]],
                             device const float* valid [[buffer(1)]],
                             device uint* packed [[buffer(2)]],
                             constant uint& count [[buffer(3)]],
                             constant uint& lane_at [[buffer(4)]],
                             constant uint& has_valid [[buffer(5)]],
                             constant uint& as_bool [[buffer(6)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    float v = values[gid];
    bool absent = (has_valid != 0u && valid[gid] == 0.0f) || isnan(v);
    /* A missing row takes one canonical pattern in every value lane, so two
     * missing rows are equal on all of them and the null lane is the only
     * thing that separates a missing row from a present one. */
    uint bits = 0u;
    if (!absent) bits = (as_bool != 0u) ? ((v != 0.0f) ? 1u : 0u) : order_bits(v);
    packed[lane_at + gid] = bits;
}

/* The two lanes of a 48-bit integer are already a positional pair with a
 * non-negative low digit, so biasing the high half into unsigned is the whole
 * of the ordering: compare high, then low, and that is the integer's order. No
 * `long` arithmetic, and 25 + 24 bits is seven radix passes where recombining
 * into one 49-bit number and re-splitting it would also be seven. */
kernel void key_encode_wide(device const float* values [[buffer(0)]],
                            device const float* wide [[buffer(1)]],
                            device const float* valid [[buffer(2)]],
                            device uint* packed [[buffer(3)]],
                            constant uint& count [[buffer(4)]],
                            constant uint& high_at [[buffer(5)]],
                            constant uint& low_at [[buffer(6)]],
                            constant uint& has_valid [[buffer(7)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    bool absent = has_valid != 0u && valid[gid] == 0.0f;
    packed[high_at + gid] = absent ? WIDE_SCALE : (uint)((int)wide[gid] + (int)WIDE_SCALE);
    packed[low_at + gid] = absent ? 0u : (uint)values[gid];
}

/* Dictionary codes are in first-appearance order, which is not an order a
 * reader would sort by, so the lane holds the level's rank instead. The rank
 * table is built once on the host over the distinct values -- never over the
 * rows -- and this remaps the rows in one dispatch. */
kernel void key_encode_code(device const float* values [[buffer(0)]],
                            device const float* valid [[buffer(1)]],
                            device const float* ranks [[buffer(2)]],
                            device uint* packed [[buffer(3)]],
                            constant uint& count [[buffer(4)]],
                            constant uint& lane_at [[buffer(5)]],
                            constant uint& has_valid [[buffer(6)]],
                            constant uint& levels [[buffer(7)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    bool absent = has_valid != 0u && valid[gid] == 0.0f;
    int code = (int)values[gid];
    if (code < 0 || code >= (int)levels) absent = true;
    packed[lane_at + gid] = absent ? 0u : (uint)ranks[code];
}

/* The null lane, one bit, always 1 for a missing row. Which end that sorts to
 * is `argsort`'s business, not the encoding's: `na_position` is absolute in
 * pandas -- nulls go last whether the sort is ascending or descending -- so
 * the lane is inverted at sort time rather than written twice. */
kernel void key_encode_missing(device const float* values [[buffer(0)]],
                               device const float* valid [[buffer(1)]],
                               device uint* packed [[buffer(2)]],
                               constant uint& count [[buffer(3)]],
                               constant uint& lane_at [[buffer(4)]],
                               constant uint& has_valid [[buffer(5)]],
                               constant uint& watch_nan [[buffer(6)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    bool absent = has_valid != 0u && valid[gid] == 0.0f;
    if (watch_nan != 0u && isnan(values[gid])) absent = true;
    packed[lane_at + gid] = absent ? 1u : 0u;
}

kernel void key_any_missing(device const uint* packed [[buffer(0)]],
                            device float* out [[buffer(1)]],
                            constant uint& count [[buffer(2)]],
                            constant uint& lane_at [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (packed[lane_at + gid] != 0u) out[gid] = 1.0f;
}

/* The same key over a subset of the rows, without going back to the columns.
 *
 * The uniques of a group-by are its head rows, and sorting them has to agree
 * with the encoding the grouping used -- re-encoding the taken columns would
 * rank a dictionary against a smaller level set and quietly disagree. One
 * dispatch moves every lane. */
kernel void key_subset_lanes(device const uint* packed [[buffer(0)]],
                             device const float* positions [[buffer(1)]],
                             device uint* out [[buffer(2)]],
                             constant uint& count [[buffer(3)]],
                             constant uint& lanes [[buffer(4)]],
                             constant uint& source [[buffer(5)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint row = (uint)positions[gid];
    for (uint l = 0u; l < lanes; ++l) out[l * count + gid] = packed[l * source + row];
}

kernel void key_hash_lanes(device const uint* packed [[buffer(0)]],
                           device uint* out [[buffer(1)]],
                           constant uint& count [[buffer(2)]],
                           constant uint& lanes [[buffer(3)]],
                           constant uint& seed [[buffer(4)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint h = fmix32(seed ^ 0x9E3779B9u);
    for (uint l = 0u; l < lanes; ++l) h = hash_combine(h, fmix32(packed[l * count + gid]));
    out[gid] = fmix32(h);
}

/* Open addressing with linear probing, one thread per row, the table bound as
 * `device atomic_uint*` over a float32 buffer.
 *
 * A slot holds `row + 1`, so zero is empty and `fill_zero` is the whole of the
 * setup. Every row with a given hash walks the same probe sequence and a slot
 * never empties again, so two threads carrying the same key converge on the
 * same slot however the races fall: the loser of a compare-and-exchange reads
 * the winner out of `expected` and finds its key equal. */
kernel void key_table_insert(device const uint* packed [[buffer(0)]],
                             device atomic_uint* slots [[buffer(1)]],
                             device const uint* hashes [[buffer(2)]],
                             device uint* slot_of_row [[buffer(3)]],
                             constant uint& count [[buffer(4)]],
                             constant uint& mask [[buffer(5)]],
                             constant uint& lanes [[buffer(6)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint pos = hashes[gid] & mask;
    for (uint step = 0u; step <= mask; ++step) {
        uint seen = atomic_load_explicit(&slots[pos], memory_order_relaxed);
        if (seen == 0u) {
            uint expected = 0u;
            bool won = atomic_compare_exchange_weak_explicit(
                &slots[pos], &expected, gid + 1u,
                memory_order_relaxed, memory_order_relaxed);
            if (won) {
                slot_of_row[gid] = pos;
                return;
            }
            /* A weak exchange is allowed to fail with nothing having changed.
             * Read the same slot again rather than counting that as a
             * collision and walking off to the next one. */
            if (expected == 0u) continue;
            seen = expected;
        }
        if (rows_equal(packed, count, lanes, gid, seen - 1u)) {
            slot_of_row[gid] = pos;
            return;
        }
        pos = (pos + 1u) & mask;
    }
    slot_of_row[gid] = pos;
}

/* Whichever thread won the exchange named the group, and that is whatever
 * order the device happened to run in. pandas numbers groups by first
 * appearance, so a second pass takes the minimum row instead. */
kernel void key_table_claim(device atomic_uint* slots [[buffer(0)]],
                            device const uint* slot_of_row [[buffer(1)]],
                            constant uint& count [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    atomic_fetch_min_explicit(&slots[slot_of_row[gid]], gid + 1u, memory_order_relaxed);
}

kernel void key_table_head(device const uint* slots [[buffer(0)]],
                           device const uint* slot_of_row [[buffer(1)]],
                           device const float* missing [[buffer(2)]],
                           device float* heads [[buffer(3)]],
                           device float* flags [[buffer(4)]],
                           constant uint& count [[buffer(5)]],
                           constant uint& drop [[buffer(6)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint head = slots[slot_of_row[gid]] - 1u;
    heads[gid] = (float)head;
    bool dropped = drop != 0u && missing[gid] != 0.0f;
    flags[gid] = (head == gid && !dropped) ? 1.0f : 0.0f;
}

kernel void key_group_codes(device const float* offsets [[buffer(0)]],
                            device const float* heads [[buffer(1)]],
                            device const float* missing [[buffer(2)]],
                            device float* codes [[buffer(3)]],
                            constant uint& count [[buffer(4)]],
                            constant uint& drop [[buffer(5)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (drop != 0u && missing[gid] != 0.0f) {
        codes[gid] = -1.0f;
        return;
    }
    codes[gid] = offsets[(uint)heads[gid]];
}

/* The rows a set flag names, packed by the scan that counted them. A filter,
 * the head rows of a group-by and the survivors of `drop_duplicates` are all
 * this call. */
kernel void key_positions_where(device const float* flags [[buffer(0)]],
                                device const float* offsets [[buffer(1)]],
                                device float* out [[buffer(2)]],
                                constant uint& count [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (flags[gid] == 0.0f) return;
    out[(uint)offsets[gid]] = (float)gid;
}

kernel void key_group_extents(device const float* codes [[buffer(0)]],
                              device atomic_uint* last [[buffer(1)]],
                              device atomic_uint* tally [[buffer(2)]],
                              constant uint& count [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    float code = codes[gid];
    if (code < 0.0f) return;
    uint group = (uint)code;
    atomic_fetch_max_explicit(&last[group], gid + 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(&tally[group], 1u, memory_order_relaxed);
}

kernel void key_group_tally(device const float* codes [[buffer(0)]],
                            device atomic_uint* tally [[buffer(1)]],
                            constant uint& count [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    float code = codes[gid];
    if (code < 0.0f) return;
    atomic_fetch_add_explicit(&tally[(uint)code], 1u, memory_order_relaxed);
}

kernel void key_duplicated(device const float* codes [[buffer(0)]],
                           device const float* heads [[buffer(1)]],
                           device const uint* last [[buffer(2)]],
                           device const uint* tally [[buffer(3)]],
                           device float* out [[buffer(4)]],
                           constant uint& count [[buffer(5)]],
                           constant uint& mode [[buffer(6)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    float code = codes[gid];
    if (code < 0.0f) {
        out[gid] = 0.0f;
        return;
    }
    uint group = (uint)code;
    bool duplicate = false;
    if (mode == 0u) duplicate = (uint)heads[group] != gid;
    else if (mode == 1u) duplicate = last[group] - 1u != gid;
    else duplicate = tally[group] > 1u;
    out[gid] = duplicate ? 1.0f : 0.0f;
}

kernel void key_uint_to_float(device const uint* src [[buffer(0)]],
                              device float* dst [[buffer(1)]],
                              constant uint& count [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    dst[gid] = (float)src[gid];
}

kernel void key_scale(device const float* src [[buffer(0)]],
                      device float* dst [[buffer(1)]],
                      constant uint& count [[buffer(2)]],
                      constant uint& factor_bits [[buffer(3)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    dst[gid] = src[gid] * as_type<float>(factor_bits);
}

kernel void key_not(device const float* src [[buffer(0)]],
                    device float* dst [[buffer(1)]],
                    constant uint& count [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    dst[gid] = (src[gid] == 0.0f) ? 1.0f : 0.0f;
}

kernel void key_gather_float(device const float* src [[buffer(0)]],
                             device const float* positions [[buffer(1)]],
                             device float* dst [[buffer(2)]],
                             constant uint& count [[buffer(3)]],
                             constant uint& length [[buffer(4)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    int at = (int)positions[gid];
    dst[gid] = (at < 0 || at >= (int)length) ? 0.0f : src[at];
}

kernel void key_invert_order(device const float* order [[buffer(0)]],
                             device float* out [[buffer(1)]],
                             constant uint& count [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out[(uint)order[gid]] = (float)gid;
}

kernel void key_descending(device float* out [[buffer(0)]],
                           constant uint& count [[buffer(1)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out[gid] = (float)(count - 1u - gid);
}

kernel void key_remap_codes(device const float* codes [[buffer(0)]],
                            device const float* mapping [[buffer(1)]],
                            device float* out [[buffer(2)]],
                            constant uint& count [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    float code = codes[gid];
    out[gid] = (code < 0.0f) ? -1.0f : mapping[(uint)code];
}

kernel void key_flag_equal(device const float* values [[buffer(0)]],
                           device float* out [[buffer(1)]],
                           constant uint& count [[buffer(2)]],
                           constant uint& target_bits [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out[gid] = (values[gid] == as_type<float>(target_bits)) ? 1.0f : 0.0f;
}

/* One lane of the packed key, read through the order built so far and, when
 * the column sorts descending, complemented within its own width.
 *
 * `mask` is `(1 << width) - 1` and every digit is below it, so `mask ^ digit`
 * is `mask - digit`: the reversal is exact and stays inside the passes the
 * driver is going to run. */
kernel void key_lane_gather(device const uint* packed [[buffer(0)]],
                            device const float* order [[buffer(1)]],
                            device uint* out [[buffer(2)]],
                            constant uint& count [[buffer(3)]],
                            constant uint& lane_at [[buffer(4)]],
                            constant uint& mask [[buffer(5)]],
                            constant uint& invert [[buffer(6)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint digit = packed[lane_at + (uint)order[gid]];
    out[gid] = (invert != 0u) ? (mask ^ digit) : digit;
}

kernel void key_radix_histogram(device const uint* keys [[buffer(0)]],
                                device uint* counts [[buffer(1)]],
                                constant uint& count [[buffer(2)]],
                                constant uint& shift [[buffer(3)]],
                                uint gid [[thread_position_in_grid]],
                                uint lid [[thread_index_in_threadgroup]],
                                uint wid [[threadgroup_position_in_grid]]) {
    threadgroup atomic_uint hist[RADIX_SIZE];
    atomic_store_explicit(&hist[lid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gid < count) {
        atomic_fetch_add_explicit(
            &hist[(keys[gid] >> shift) & (RADIX_SIZE - 1u)], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    /* Block-major, so the offsets pass below reads one digit across blocks
     * with consecutive threads on consecutive addresses. Digit-major would
     * have made every one of those reads a separate cache line. */
    counts[wid * RADIX_SIZE + lid] = atomic_load_explicit(&hist[lid], memory_order_relaxed);
}

/* Turn the per-block histograms into the address each block's digit run
 * starts at, in one threadgroup and without a device wait.
 *
 * Thread `lid` owns digit `lid` across every block: it runs the digit down the
 * blocks, then the 256 digit totals are scanned in threadgroup memory, then
 * the digit's base is added back. Two strided walks of `blocks`, which at
 * 2^24 rows is 65536 steps on 256 threads -- against a device round trip per
 * pass if the scan went back through the host. */
kernel void key_radix_offsets(device uint* counts [[buffer(0)]],
                              constant uint& blocks [[buffer(1)]],
                              uint lid [[thread_index_in_threadgroup]]) {
    threadgroup uint scratch[RADIX_SIZE];
    uint running = 0u;
    for (uint b = 0u; b < blocks; ++b) {
        uint at = b * RADIX_SIZE + lid;
        uint here = counts[at];
        counts[at] = running;
        running += here;
    }
    scratch[lid] = running;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1u; stride < RADIX_SIZE; stride <<= 1) {
        uint add = (lid >= stride) ? scratch[lid - stride] : 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        scratch[lid] += add;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint base = scratch[lid] - running;
    for (uint b = 0u; b < blocks; ++b) counts[b * RADIX_SIZE + lid] += base;
}

/* The stable placement. Every thread needs its rank among the earlier
 * elements of its own block that share its digit, which is the only part of a
 * radix sort that is not a scan -- and the part that decides whether the sort
 * is stable, which is what makes a multi-key sort a sequence of single-key
 * passes.
 *
 * The block's digits go into threadgroup memory once and every thread walks
 * the prefix of that tile, four at a time: neighbouring threads read almost
 * the same span, so it is one tile in fast memory rather than 256 device
 * reads apiece. `lid >> 2` is the number of whole quads before the thread and
 * `lid - (whole << 2)` the remainder, one shift and a subtraction, because the
 * device has no integer divide. */
kernel void key_radix_scatter(device const uint* keys_in [[buffer(0)]],
                              device const float* order_in [[buffer(1)]],
                              device uint* keys_out [[buffer(2)]],
                              device float* order_out [[buffer(3)]],
                              device const uint* counts [[buffer(4)]],
                              constant uint& count [[buffer(5)]],
                              constant uint& shift [[buffer(6)]],
                              uint gid [[thread_position_in_grid]],
                              uint lid [[thread_index_in_threadgroup]],
                              uint wid [[threadgroup_position_in_grid]]) {
    threadgroup uint4 quads[RADIX_GROUP / 4u];
    threadgroup uint* tile = (threadgroup uint*)quads;
    uint key = (gid < count) ? keys_in[gid] : 0u;
    /* A thread past the end parks on a digit no live thread can hold, so it
     * never adds to anybody's rank. */
    uint digit = (gid < count) ? ((key >> shift) & (RADIX_SIZE - 1u)) : RADIX_SIZE;
    tile[lid] = digit;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gid >= count) return;
    uint whole = lid >> 2;
    uint rest = lid - (whole << 2);
    uint4 want = uint4(digit);
    uint rank = 0u;
    for (uint v = 0u; v < whole; ++v) {
        uint4 hit = select(uint4(0u), uint4(1u), quads[v] == want);
        rank += hit.x + hit.y + hit.z + hit.w;
    }
    uint head = whole << 2;
    for (uint j = 0u; j < rest; ++j) rank += (tile[head + j] == digit) ? 1u : 0u;
    uint at = counts[wid * RADIX_SIZE + digit] + rank;
    keys_out[at] = key;
    order_out[at] = order_in[gid];
}

/* Where a run of equal keys starts, read along a finished order. Ranking and
 * `keep: all` both need the ties, and both get them from the same encoding
 * the sort used rather than by comparing values a second way. */
kernel void key_run_starts(device const uint* packed [[buffer(0)]],
                           device const float* order [[buffer(1)]],
                           device float* flags [[buffer(2)]],
                           constant uint& count [[buffer(3)]],
                           constant uint& lanes [[buffer(4)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (gid == 0u) {
        flags[0] = 1.0f;
        return;
    }
    uint here = (uint)order[gid];
    uint before = (uint)order[gid - 1u];
    flags[gid] = rows_equal(packed, count, lanes, here, before) ? 0.0f : 1.0f;
}

/* Which run of equal keys a sorted position falls in.
 *
 * The scan behind the runs is exclusive, so it counts the runs that *start*
 * before a position -- which is the run's own number only at the position that
 * starts it. Adding the position's own flag and dropping one turns the count
 * into the run it belongs to. */
kernel void key_dense_codes(device const float* offsets [[buffer(0)]],
                            device const float* flags [[buffer(1)]],
                            device float* out [[buffer(2)]],
                            constant uint& count [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out[gid] = offsets[gid] + flags[gid] - 1.0f;
}

kernel void key_rank_assign(device const float* order [[buffer(0)]],
                            device const float* dense [[buffer(1)]],
                            device const float* run_start [[buffer(2)]],
                            device float* out [[buffer(3)]],
                            device float* valid [[buffer(4)]],
                            constant uint& count [[buffer(5)]],
                            constant uint& distinct [[buffer(6)]],
                            constant uint& method [[buffer(7)]],
                            constant uint& present [[buffer(8)]],
                            constant uint& keep_na [[buffer(9)]],
                            constant uint& as_pct [[buffer(10)]],
                            constant uint& dense_total [[buffer(11)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint row = (uint)order[gid];
    if (keep_na != 0u && gid >= present) {
        out[row] = 0.0f;
        valid[row] = 0.0f;
        return;
    }
    uint group = (uint)dense[gid];
    uint low = (uint)run_start[group];
    uint high = (group + 1u < distinct) ? ((uint)run_start[group + 1u] - 1u) : (count - 1u);
    float value = (float)(group + 1u);
    if (method == 0u) value = (float)(low + high + 2u) * 0.5f;
    else if (method == 1u) value = (float)(low + 1u);
    else if (method == 2u) value = (float)(high + 1u);
    else if (method == 3u) value = (float)(gid + 1u);
    if (as_pct != 0u) {
        float over = (method == 4u) ? (float)dense_total
                                    : (float)((keep_na != 0u) ? present : count);
        value = (over > 0.0f) ? value / over : 0.0f;
    }
    out[row] = value;
    valid[row] = 1.0f;
}

kernel void key_bin_tally(device const float* values [[buffer(0)]],
                          device const float* valid [[buffer(1)]],
                          device atomic_uint* tally [[buffer(2)]],
                          constant uint& count [[buffer(3)]],
                          constant uint& bins [[buffer(4)]],
                          constant uint& has_valid [[buffer(5)]],
                          constant uint& low_bits [[buffer(6)]],
                          constant uint& inverse_bits [[buffer(7)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (has_valid != 0u && valid[gid] == 0.0f) return;
    float v = values[gid];
    if (isnan(v)) return;
    /* A reciprocal the host worked out once, not a divide per row: the device
     * has no integer divide and a float one is several times a multiply. */
    int at = (int)floor((v - as_type<float>(low_bits)) * as_type<float>(inverse_bits));
    at = clamp(at, 0, (int)bins - 1);
    atomic_fetch_add_explicit(&tally[(uint)at], 1u, memory_order_relaxed);
}

/* Sum, smallest or largest over a lane, skipping the rows a mask calls
 * missing. A fixed grid strides the lane, as `col_count_nulls` does, because
 * the fold costs eight barriers and paying that per row rather than per group
 * makes the reduction slower than the read it is reducing. */
kernel void key_reduce(device const float* values [[buffer(0)]],
                       device const float* valid [[buffer(1)]],
                       device float* partials [[buffer(2)]],
                       constant uint& count [[buffer(3)]],
                       constant uint& has_valid [[buffer(4)]],
                       constant uint& mode [[buffer(5)]],
                       constant uint& stride [[buffer(6)]],
                       uint gid [[thread_position_in_grid]],
                       uint lid [[thread_index_in_threadgroup]],
                       uint wid [[threadgroup_position_in_grid]]) {
    threadgroup float scratch[REDUCE_GROUP];
    float best = 0.0f;
    if (mode == 1u) best = FLOAT_CEILING;
    if (mode == 2u) best = -FLOAT_CEILING;
    for (uint i = gid; i < count; i += stride) {
        if (has_valid != 0u && valid[i] == 0.0f) continue;
        float v = values[i];
        if (isnan(v)) continue;
        if (mode == 0u) best += v;
        else if (mode == 1u) best = min(best, v);
        else best = max(best, v);
    }
    scratch[lid] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = REDUCE_GROUP / 2u; step > 0u; step >>= 1) {
        if (lid < step) {
            float other = scratch[lid + step];
            if (mode == 0u) scratch[lid] += other;
            else if (mode == 1u) scratch[lid] = min(scratch[lid], other);
            else scratch[lid] = max(scratch[lid], other);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) partials[wid] = scratch[0];
}
"""

var _kernels: dict[str, Kernel] = {}

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

#: Two's-complement bits of `value` as a non-negative int, for a kernel scalar
#: that is declared `constant uint&` but carries a signed number.
pub fn u32(value: int) -> int { return value & 0xFFFFFFFF }

#: How many bits it takes to hold every value below `span`.
pub fn bits_for(span: int) -> int {
    var width = 1
    while 1 << width < span { width += 1 }
    return width
}
