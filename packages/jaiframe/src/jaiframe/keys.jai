#: Row identity and row order — the substrate group-by, join, reshape and
#: sorting all stand on.
#:
#: Two questions come up in every one of those modules: *which rows carry the
#: same key*, and *what order do the rows go in*. Answering either one well is
#: a hash table or a sort on the device, and answering them four times over in
#: four modules is how a data frame library ends up with four subtly different
#: notions of what makes two rows equal. So both live here, over one encoding.
#:
#: `encode_keys` turns any list of key columns into a `KeyEncoding`: a packed
#: lane of uint32 digits per column, ordered most significant first, in which
#: unsigned integer order *is* the value's order and equality on every lane
#: *is* row equality. A float becomes its order-preserving bit pattern; the two
#: lanes of an `Int64` become a biased high digit and a raw low one; a
#: dictionary code becomes the rank of its level. After that one step nothing
#: downstream has to know a dtype: `factorize` hashes the lanes and probes an
#: open-addressed table, `argsort` runs a radix pass per digit, and both are
#: the same code for text, timestamps and floats.
#:
#: ```
#: let city = Column.from_strs(["dallas", "austin", "dallas"])
#: defer { city.free() }
#: let grouped = factorize([city])
#: defer { grouped[0].free() }
#: grouped[2]                    # 2 groups
#: ```
import std.gpu as gpu
from std.gpu import Buffer, Kernel
from .column import Column, DataFrame, Dtype, Index, Series
from .column import cached_kernel, f32_bits, iota, prefix_sum

#: Digits of a radix pass. Eight bits is four passes over a 32-bit lane, and
#: the local rank in `key_radix_scatter` costs the tile width per element — so
#: a narrower digit buys more passes than it saves scanning, and a wider one
#: needs a bucket array bigger than a threadgroup.
const RADIX_BITS = 8
const RADIX_SIZE = 256
const RADIX_GROUP = 256

const REDUCE_GROUP = 256
const REDUCE_GROUPS = 256

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

#: Two's-complement bits of `value` as a non-negative int, for a kernel scalar
#: that is declared `constant uint&` but carries a signed number.
fn u32(value: int) -> int { return value & 0xFFFFFFFF }

#: How many bits it takes to hold every value below `span`.
fn bits_for(span: int) -> int {
    var width = 1
    while 1 << width < span { width += 1 }
    return width
}

#: The key columns of a frame turned into digits: one packed lane of uint32 per
#: level of the key, most significant first.
#:
#: The encoding is the point of this module. Unsigned integer order over a lane
#: is the value's order, and equality on every lane is row equality — so a
#: radix sort and a hash table both stop caring what dtype went in. What it
#: costs is a second copy of the key columns as uints: `lane_count * rows`
#: words, which for a three-column key over a million rows is about 25 MB.
#:
#: Lanes are held in one allocation rather than one buffer apiece so that the
#: table's equality test is a strided walk of a single binding, and so that the
#: whole key is one `free`.
pub class KeyEncoding {
    #: How many rows are keyed.
    pub var length: int
    #: `lane_count() * length` uint32 digits, lane `l` starting at `l * length`.
    #: Device-side bits: downloading this as floats gives nonsense.
    pub var packed: Buffer
    #: Significant bits in each lane, which is how many radix passes it needs.
    pub var widths: list[int]
    #: Which key column each lane came from, so `ascending` can be per column.
    pub var owners: list[int]
    #: Which lanes are a column's missing flag rather than its value.
    pub var null_lanes: list[bool]
    #: How many key columns went in.
    pub var columns: int
    #: 1.0 where any key column is missing, `null` when none of them can be.
    pub var missing: Buffer?
    var freed: bool

    #: An empty encoding. `encode_keys` is the constructor.
    pub fn init(self) {
        self.length = 0
        self.packed = Buffer(0)
        self.widths = []
        self.owners = []
        self.null_lanes = []
        self.columns = 0
        self.missing = null
        self.freed = false
    }

    #: How many digit lanes the key spans.
    pub fn lane_count(self) -> int { return self.widths.len() }

    #: One uint32 hash per row, folded over every lane.
    #:
    #: This agrees with the equality the hash table here tests, because both
    #: read the same lanes. `Column.hash_into` is the other self-consistent
    #: pair, with `Column.equal_rows`; do not mix a hash from one with an
    #: equality from the other.
    #:
    #: Device-side uint bits, like `packed`.
    pub fn hash(self, seed: int = 0) -> Buffer {
        self.check_live()
        let out = Buffer(self.length)
        if self.length == 0 { return out }
        kernel_for("key_hash_lanes").dispatch_async(
            [self.packed, out],
            self.length,
            [self.length, self.lane_count(), u32(seed)]
        )
        return out
    }

    #: The permutation that sorts the rows, stable, least significant key first.
    #:
    #: A radix sort is stable, and stability is exactly what turns a multi-key
    #: sort into a sequence of single-key ones: sort by the last key, then by
    #: the one before it, and the second pass leaves rows that tie on it in the
    #: order the first pass left them. So there is no comparator that has to
    #: know about all the keys at once — the lanes are walked from least to
    #: most significant and each pass only ever looks at eight bits.
    #:
    #: `ascending` is one flag per key column, or a single flag for all of
    #: them, or `null` for all ascending. `na_position` is absolute the way
    #: pandas' is: `"last"` puts missing rows at the end whether the sort is
    #: ascending or descending, which is why a null lane is never flipped along
    #: with its column.
    pub fn order(self, ascending: list[bool]? = null, na_position: str = "last") -> Buffer {
        self.check_live()
        if na_position != "first" and na_position != "last" {
            throw ValueError(f"na_position must be first or last, not {na_position}")
        }
        let ups = spread_flags(ascending, self.columns)
        var forward = iota(self.length)
        if self.length == 0 { return forward }

        let blocks = (self.length + RADIX_GROUP - 1) // RADIX_GROUP
        var keys = Buffer(self.length)
        var spare_keys = Buffer(self.length)
        var spare = Buffer(self.length)
        let counts = Buffer(blocks * RADIX_SIZE)

        var lane = self.lane_count() - 1
        while lane >= 0 {
            let width = self.widths[lane]
            let mask = (1 << width) - 1
            let invert = if self.null_lanes[lane] {
                na_position == "first"
            } else {
                not ups[self.owners[lane]]
            }
            kernel_for("key_lane_gather").dispatch_async(
                [self.packed, forward, keys],
                self.length,
                [self.length, lane * self.length, mask, flag(invert)]
            )
            var shift = 0
            while shift < width {
                kernel_for("key_radix_histogram").dispatch_async(
                    [keys, counts],
                    blocks * RADIX_GROUP,
                    [self.length, shift],
                    group_size: RADIX_GROUP
                )
                kernel_for("key_radix_offsets").dispatch_async(
                    [counts],
                    RADIX_GROUP,
                    [blocks],
                    group_size: RADIX_GROUP
                )
                kernel_for("key_radix_scatter").dispatch_async(
                    [keys, forward, spare_keys, spare, counts],
                    blocks * RADIX_GROUP,
                    [self.length, shift],
                    group_size: RADIX_GROUP
                )
                let swapped_keys = keys
                keys = spare_keys
                spare_keys = swapped_keys
                let swapped = forward
                forward = spare
                spare = swapped
                shift += RADIX_BITS
            }
            lane -= 1
        }
        keys.free()
        spare_keys.free()
        spare.free()
        counts.free()
        return forward
    }

    #: 1.0 where a row starts a run of equal keys along `positions`.
    #:
    #: The caller owns the lane. Ties are read off the same encoding the sort
    #: used, so a rank and a `keep: all` cutoff cannot disagree with the order
    #: they were computed from.
    pub fn run_flags(self, positions: Buffer) -> Buffer {
        self.check_live()
        let flags = Buffer(self.length)
        if self.length == 0 { return flags }
        kernel_for("key_run_starts").dispatch_async(
            [self.packed, positions, flags],
            self.length,
            [self.length, self.lane_count()]
        )
        return flags
    }

    #: The same key restricted to the rows `positions` names.
    #:
    #: The digits are moved rather than recomputed, so a sort of a group-by's
    #: uniques agrees with the grouping by construction: re-encoding the taken
    #: columns would rank a dictionary against whatever level set survived the
    #: take.
    pub fn gather(self, positions: Buffer, count: int) -> KeyEncoding {
        self.check_live()
        let out = KeyEncoding()
        out.length = count
        out.columns = self.columns
        out.widths = [width for width in self.widths]
        out.owners = [owner for owner in self.owners]
        out.null_lanes = [is_null for is_null in self.null_lanes]
        out.packed = Buffer(self.lane_count() * count)
        let absent = self.missing
        if absent is not null {
            let taken = Buffer(count)
            if count > 0 {
                kernel_for("key_gather_float").dispatch_async(
                    [absent, positions, taken],
                    count,
                    [count, self.length]
                )
            }
            out.missing = taken
        }
        if count > 0 {
            kernel_for("key_subset_lanes").dispatch_async(
                [self.packed, positions, out.packed],
                count,
                [count, self.lane_count(), self.length]
            )
        }
        return out
    }

    #: Release the packed lanes. Freeing twice is harmless.
    pub fn free(self) -> void {
        if self.freed { return }
        self.freed = true
        self.packed.free()
        let absent = self.missing
        if absent is not null { absent.free() }
    }

    fn check_live(self) -> void {
        if self.freed { throw RuntimeError("this KeyEncoding has already been released") }
    }
}

#: The key columns encoded into one hashable, order-preserving lane set.
#:
#: Text is ranked by its level string, not by its dictionary code, because
#: codes are in first-appearance order and nobody sorts by that. The rank table
#: is a host sort over the distinct values — not over the rows, which one
#: dispatch remaps — so a column of a million rows over twenty cities costs
#: twenty comparisons on the host.
#:
#: Two dictionary columns only encode alike once `unify_levels` has put them on
#: one table, exactly as they only hash alike after it: a level set of
#: `["a", "c"]` ranks `"c"` at 1 and one of `["a", "b", "c"]` ranks it at 2.
#: A join across two frames unifies first.
pub fn encode_keys(columns: list[Column]) -> KeyEncoding {
    if columns.len() == 0 { throw ValueError("a key needs at least one column") }
    require_device("encode a key")
    let length = columns[0].len()
    for column in columns {
        if column.len() != length {
            throw ValueError(f"key columns disagree on length: {column.len()} against {length}")
        }
    }

    let encoding = KeyEncoding()
    encoding.length = length
    encoding.columns = columns.len()
    var widths: list[int] = []
    var owners: list[int] = []
    var null_lanes: list[bool] = []
    var any_nullable = false
    for (at, column) in columns.enumerate() {
        if nullable(column) {
            widths.push(1)
            owners.push(at)
            null_lanes.push(true)
            any_nullable = true
        }
        for width in value_widths(column) {
            widths.push(width)
            owners.push(at)
            null_lanes.push(false)
        }
    }
    encoding.widths = widths
    encoding.owners = owners
    encoding.null_lanes = null_lanes
    encoding.packed = Buffer(widths.len() * length)

    var absent: Buffer? = null
    if any_nullable {
        absent = Buffer(length)
        if length > 0 { (absent ?? encoding.packed).fill_zero() }
    }
    encoding.missing = absent

    if length == 0 { return encoding }
    var lane = 0
    for column in columns {
        let mask = column.valid ?? column.values
        let has_valid = flag(column.has_validity())
        if nullable(column) {
            kernel_for("key_encode_missing").dispatch_async(
                [column.values, mask, encoding.packed],
                length,
                [length, lane * length, has_valid, flag(column.dtype == Dtype.Float64)]
            )
            if absent is not null {
                kernel_for("key_any_missing").dispatch_async(
                    [encoding.packed, absent ?? encoding.packed],
                    length,
                    [length, lane * length]
                )
            }
            lane += 1
        }
        if column.dtype.is_wide() {
            kernel_for("key_encode_wide").dispatch_async(
                [column.values, column.wide ?? column.values, mask, encoding.packed],
                length,
                [length, lane * length, (lane + 1) * length, has_valid]
            )
            lane += 2
            continue
        }
        if column.dtype.is_dictionary() {
            let levels = column.levels ?? []
            let ranks = Buffer(if levels.len() > 0 { levels.len() } else { 1 })
            if levels.len() > 0 { ranks.upload(level_ranks(levels)) }
            kernel_for("key_encode_code").dispatch_async(
                [column.values, mask, ranks, encoding.packed],
                length,
                [length, lane * length, has_valid, levels.len()]
            )
            ranks.free()
            lane += 1
            continue
        }
        kernel_for("key_encode_plain").dispatch_async(
            [column.values, mask, encoding.packed],
            length,
            [length, lane * length, has_valid, flag(column.dtype == Dtype.Bool)]
        )
        lane += 1
    }
    return encoding
}

#: One uint32 hash per row over the key columns.
#:
#: Device-side bits, as `KeyEncoding.hash` returns: this is that call with the
#: encoding built and thrown away, for a caller that wants the hash and nothing
#: else. A caller that also wants equality or an order should keep the
#: encoding.
pub fn hash_keys(columns: list[Column], seed: int = 0) -> Buffer {
    let encoding = encode_keys(columns)
    defer { encoding.free() }
    return encoding.hash(seed)
}

#: Whether a column can hold a missing value at all, and so needs a null lane.
#:
#: A `Float64` column always gets one even with no validity lane, because a
#: NaN written by a kernel — a zero divide, a mean over an empty group — is
#: missing to everything downstream and has to sort and group as one. The lane
#: is one bit, so the cost of being wrong the safe way is a single radix pass.
fn nullable(column: Column) -> bool {
    return column.has_validity() or column.dtype == Dtype.Float64
}

fn value_widths(column: Column) -> list[int] {
    if column.dtype.is_wide() { return [25, 24] }
    if column.dtype.is_dictionary() {
        return [bits_for(if column.level_count() > 1 { column.level_count() } else { 2 })]
    }
    if column.dtype == Dtype.Bool { return [1] }
    return [32]
}

fn level_ranks(levels: list[str]) -> list[float] {
    var codes: list[int] = []
    for code in 0..levels.len() { codes.push(code) }
    let ordered = codes.sort(key: |code| levels[code])
    var ranks: list[float] = []
    for _code in 0..levels.len() { ranks.push(0.0) }
    for (rank, code) in ordered.enumerate() { ranks[code] = float(rank) }
    return ranks
}

fn spread_flags(flags: list[bool]?, wanted: int) -> list[bool] {
    if flags is null { return [true for _at in 0..wanted] }
    let given = flags ?? []
    if given.len() == 1 { return [given[0] for _at in 0..wanted] }
    if given.len() != wanted {
        throw ValueError(f"{given.len()} ascending flags cannot cover {wanted} keys")
    }
    return [value for value in given]
}

#: The rows that carry the same key, numbered by first appearance.
#:
#: The one call group-by and join both make. `codes[row]` is which group the
#: row belongs to, `uniques` is one row per group in code order, and
#: `group_count` is how many there are. `dropna` gives a row whose key is
#: missing the code `NO_GROUP` and keeps it out of the uniques, which is what
#: `groupby(dropna: true)` and pandas' `factorize` both mean by it.
#:
#: A hash table on the device: one probe sequence per row, a compare-and-
#: exchange to claim an empty slot, and a second pass that replaces whichever
#: thread happened to win with the lowest row in the group — because pandas
#: numbers groups by where they first appear and the device's scheduling is
#: not that order. The table is sized at least 1.5x the rows so a probe is a
#: step or two, which costs 8 MB per million rows and is the trade this makes
#: against a longer walk.
#:
#: `sort` renumbers the groups into key order afterwards. It sorts the uniques,
#: not the rows, so it costs a radix sort over the distinct values.
pub fn factorize(
    columns: list[Column],
    sort: bool = false,
    dropna: bool = false
) -> (Buffer, list[Column], int) {
    let keys = encode_keys(columns)
    defer { keys.free() }
    let grouped = group_rows(keys, dropna)
    let codes = grouped[0]
    let heads = grouped[1]
    let group_count = grouped[2]
    defer { heads.free() }

    let uniques = columns.map(|column| column.take(heads, group_count))
    if not sort or group_count == 0 { return (codes, uniques, group_count) }

    let distinct = keys.gather(heads, group_count)
    defer { distinct.free() }
    let order = distinct.order()
    defer { order.free() }
    let renumbered = Buffer(group_count)
    kernel_for("key_invert_order").dispatch_async([order, renumbered], group_count, [group_count])
    let mapped = Buffer(keys.length)
    if keys.length > 0 {
        kernel_for("key_remap_codes").dispatch_async(
            [codes, renumbered, mapped],
            keys.length,
            [keys.length]
        )
    }
    renumbered.free()
    codes.free()
    let sorted_uniques = uniques.map(|column| column.take(order, group_count))
    for column in uniques { column.free() }
    return (mapped, sorted_uniques, group_count)
}

#: The distinct key rows, in order of first appearance.
#:
#: pandas' `Series.unique`, which does not sort — `factorize(sort: true)` is
#: the call that does.
pub fn unique(target: any) -> Column {
    let columns = columns_of(target)
    if columns.len() != 1 {
        throw ValueError(f"unique takes one column, not {columns.len()}")
    }
    let found = factorize(columns, dropna: false)
    found[0].free()
    return found[1][0]
}

#: How many distinct key rows there are.
pub fn nunique(target: any, dropna: bool = true) -> int {
    let columns = columns_of(target)
    let found = factorize(columns, dropna: dropna)
    found[0].free()
    for column in found[1] { column.free() }
    return found[2]
}

#: How often each distinct key row occurs.
#:
#: The result is a `Series` of counts labelled by the values counted, named
#: `count` — or `proportion` when `normalize` divides through by the total, as
#: pandas names it. `bins` instead cuts a numeric column into that many
#: equal-width right-closed intervals and counts those, including the empty
#: ones; the first edge is nudged down by a thousandth of the range so the
#: smallest value falls inside it, which is what pandas does and why its first
#: interval always looks slightly too wide.
pub fn value_counts(
    target: any,
    normalize: bool = false,
    sort: bool = true,
    ascending: bool = false,
    dropna: bool = true,
    bins: any = null
) -> Series {
    let columns = columns_of(target)
    if bins is not null {
        if columns.len() != 1 {
            throw ValueError("bins counts one numeric column, not a whole frame")
        }
        return binned_counts(columns[0], bins, normalize, sort, ascending, label_of(target))
    }
    let keys = encode_keys(columns)
    defer { keys.free() }
    let grouped = group_rows(keys, dropna)
    let codes = grouped[0]
    let heads = grouped[1]
    let group_count = grouped[2]
    defer { codes.free() }
    defer { heads.free() }

    let tally = tally_groups(codes, keys.length, group_count)
    var counts = Column.allocate(Dtype.Int64, group_count)
    if group_count > 0 {
        kernel_for("key_uint_to_float").dispatch_async(
            [tally, counts.values],
            group_count,
            [group_count]
        )
        (counts.wide ?? counts.values).fill_zero()
    }
    tally.free()
    if normalize {
        let total = float(kept_rows(keys, dropna))
        let shares = Column.allocate(Dtype.Float64, group_count)
        if group_count > 0 {
            let scale = if total > 0.0 { 1.0 / total } else { 0.0 }
            kernel_for("key_scale").dispatch_async(
                [counts.values, shares.values],
                group_count,
                [group_count, f32_bits(scale)]
            )
        }
        counts.free()
        counts = shares
    }

    var labels = columns.map(|column| column.take(heads, group_count))
    let named = if normalize { "proportion" } else { "count" }
    if not sort { return counted_series(labels, counts, names_of(target), named) }

    let order = argsort([counts], [ascending])
    defer { order.free() }
    let ranked = labels.map(|column| column.take(order, group_count))
    for column in labels { column.free() }
    let ordered_counts = counts.take(order, group_count)
    counts.free()
    return counted_series(ranked, ordered_counts, names_of(target), named)
}

#: The most frequent value or values, ascending.
#:
#: Several when they tie, as pandas returns. Built on the sorted factorization
#: so the ties come out in value order without a second sort.
pub fn mode(target: any, dropna: bool = true) -> Column {
    let columns = columns_of(target)
    if columns.len() != 1 { throw ValueError("mode takes one column") }
    let keys = encode_keys(columns)
    defer { keys.free() }
    let grouped = group_rows(keys, dropna)
    let codes = grouped[0]
    let heads = grouped[1]
    let group_count = grouped[2]
    defer { codes.free() }
    defer { heads.free() }
    if group_count == 0 { return Column.allocate(columns[0].dtype, 0) }

    let tally = tally_groups(codes, keys.length, group_count)
    let counts = Buffer(group_count)
    kernel_for("key_uint_to_float").dispatch_async([tally, counts], group_count, [group_count])
    tally.free()
    defer { counts.free() }
    let best = reduce_lane(counts, null, group_count, 2)
    let flags = Buffer(group_count)
    kernel_for("key_flag_equal").dispatch_async(
        [counts, flags],
        group_count,
        [group_count, f32_bits(best)]
    )
    defer { flags.free() }
    let picked = positions_where(flags, group_count)
    let winners = picked[0]
    defer { winners.free() }
    let rows = columns[0].take(heads, group_count)
    defer { rows.free() }
    let tied = rows.take(winners, picked[1])
    defer { tied.free() }
    let order = argsort([tied])
    defer { order.free() }
    return tied.take(order, picked[1])
}

#: True where a row repeats one already seen.
#:
#: `keep` is `"first"` for every occurrence after the first, `"last"` for every
#: one before the last, and `false` for every row of a repeated key. Missing
#: keys count as values and as equal to each other, which is what pandas'
#: `duplicated` does and the opposite of what a `dropna` group-by does.
pub fn duplicated(frame: any, subset: any = null, keep: any = "first") -> Column {
    let columns = subset_columns(frame, subset)
    let mode_code = duplicate_mode(keep)
    let keys = encode_keys(columns)
    defer { keys.free() }
    let grouped = group_rows(keys, false)
    let codes = grouped[0]
    let heads = grouped[1]
    let group_count = grouped[2]
    defer { codes.free() }
    defer { heads.free() }

    let out = Column.allocate(Dtype.Bool, keys.length)
    if keys.length == 0 { return out }
    let last = Buffer(if group_count > 0 { group_count } else { 1 })
    let tally = Buffer(if group_count > 0 { group_count } else { 1 })
    last.fill_zero()
    tally.fill_zero()
    kernel_for("key_group_extents").dispatch_async([codes, last, tally], keys.length, [keys.length])
    kernel_for("key_duplicated").dispatch_async(
        [codes, heads, last, tally, out.values],
        keys.length,
        [keys.length, mode_code]
    )
    last.free()
    tally.free()
    return out
}

#: The frame with repeated key rows removed.
#:
#: The survivors are gathered rather than compressed, so the index travels with
#: them: one positions lane feeds `Column.take` and `Index.take` alike.
pub fn drop_duplicates(
    frame: any,
    subset: any = null,
    keep: any = "first",
    ignore_index: bool = false
) -> any {
    let marks = duplicated(frame, subset, keep)
    defer { marks.free() }
    let flags = Buffer(marks.len())
    if marks.len() > 0 {
        kernel_for("key_not").dispatch_async([marks.values, flags], marks.len(), [marks.len()])
    }
    defer { flags.free() }
    let picked = positions_where(flags, marks.len())
    let survivors = picked[0]
    defer { survivors.free() }
    return reorder(frame, survivors, picked[1], ignore_index)
}

#: The permutation that sorts the rows by several key columns at once.
#:
#: Stable, and least significant key first — which is not an implementation
#: detail but the reason the sort is simple. Because each pass leaves rows that
#: tie on its own key in the order the previous pass left them, sorting by the
#: last key and then by the first gives exactly the order a comparator over the
#: pair would: the earlier keys break ties the later ones left. So there is no
#: multi-key comparison anywhere, only a chain of eight-bit passes over one
#: digit lane at a time.
#:
#: `ascending` is one flag per column, one flag for all of them, or `null`.
#: `na_position` is absolute, as in pandas: `"last"` means missing rows end up
#: at the end however the columns are sorted.
pub fn argsort(
    columns: list[Column],
    ascending: list[bool]? = null,
    na_position: str = "last"
) -> Buffer {
    let keys = encode_keys(columns)
    defer { keys.free() }
    return keys.order(ascending, na_position)
}

#: The frame, series or column sorted by its key columns.
#:
#: `by` names the columns of a frame, and a series or a column sorts by its own
#: values and refuses one. `ascending` is a bool or one bool per key.
#: `ignore_index` replaces the labels with a fresh range, as pandas' does,
#: rather than carrying the old ones along in their new order.
pub fn sort_values(
    target: any,
    by: any = null,
    ascending: any = true,
    na_position: str = "last",
    ignore_index: bool = false
) -> any {
    let columns = sort_key_columns(target, by)
    let order = argsort(columns, spread_any(ascending, columns.len()), na_position)
    defer { order.free() }
    return reorder(target, order, row_count(target), ignore_index)
}

#: The frame or series sorted by its row labels.
#:
#: `level` picks which levels of a multi-index decide, by position or by name;
#: `null` uses all of them, outermost first. A range index is already in order,
#: so this recognises one instead of materialising the ramp it stands for and
#: sorting that.
pub fn sort_index(
    target: any,
    level: any = null,
    ascending: bool = true,
    na_position: str = "last",
    ignore_index: bool = false
) -> any {
    let index = index_of(target)
    let length = index.len()
    if index.is_range() {
        let forward = (index.step > 0) == ascending
        let order = if forward { iota(length) } else { descending_positions(length) }
        defer { order.free() }
        return reorder(target, order, length, ignore_index)
    }
    var levels: list[Column] = []
    for at in resolve_levels(index, level) { levels.push(index.levels[at]) }
    let order = argsort(levels, [ascending], na_position)
    defer { order.free() }
    return reorder(target, order, length, ignore_index)
}

#: The `n` rows with the largest keys, largest first.
#:
#: `keep` decides the ties at the cutoff: `"first"` takes the rows that came
#: first, `"last"` the ones that came last, and `"all"` widens the result past
#: `n` to take the whole run. A row missing any of its key columns is left out
#: entirely, the way pandas leaves a NaN out of an `nlargest`.
pub fn nlargest(target: any, n: int, columns: any = null, keep: str = "first") -> any {
    return extreme(target, n, columns, keep, true)
}

#: The `n` rows with the smallest keys, smallest first.
pub fn nsmallest(target: any, n: int, columns: any = null, keep: str = "first") -> any {
    return extreme(target, n, columns, keep, false)
}

#: Each row's position in the sorted order, one to `n`.
#:
#: `method` decides what a tie gets: `"average"` the midpoint of the positions
#: the tied rows occupy, `"min"` the first of them, `"max"` the last, `"first"`
#: the position each row actually landed in, and `"dense"` the run's number
#: among distinct values with no gaps after it. `pct` divides through by the
#: number of ranked rows — by the number of distinct values for `"dense"`,
#: which is what makes a dense percentile reach 1.0.
#:
#: `na_option` is `"keep"` for a missing rank on a missing value, `"top"` to
#: rank missing first and `"bottom"` to rank it last. The sort underneath is
#: the same one `argsort` runs; the ties come off the same encoding, so a rank
#: cannot disagree with the order it was read from.
pub fn rank(
    target: any,
    method: str = "average",
    ascending: bool = true,
    pct: bool = false,
    na_option: str = "keep"
) -> any {
    if target is DataFrame {
        let ranked = target.columns.map(
            |column| rank_column(column, method, ascending, pct, na_option)
        )
        return target._replace(columns: ranked)
    }
    if target is Series {
        return Series(
            rank_column(target.values, method, ascending, pct, na_option),
            index: target.index.copy(),
            name: target.name
        )
    }
    if target is Column { return rank_column(target, method, ascending, pct, na_option) }
    throw TypeError("rank takes a Column, a Series or a DataFrame")
}

fn rank_column(column: Column, method: str, ascending: bool, pct: bool, na_option: str) -> Column {
    let code = rank_method(method)
    if na_option != "keep" and na_option != "top" and na_option != "bottom" {
        throw ValueError(f"na_option must be keep, top or bottom, not {na_option}")
    }
    let keeps = na_option == "keep"
    let na_position = if na_option == "top" { "first" } else { "last" }
    let count = column.len()
    let keys = encode_keys([column])
    defer { keys.free() }
    let order = keys.order([ascending], na_position)
    defer { order.free() }
    let out = Column.allocate(Dtype.Float64, count, with_validity: keeps)
    if count == 0 { return out }

    let runs = dense_runs(keys, order, count)
    let dense = runs[0]
    let starts = runs[1]
    let distinct = runs[2]
    defer { dense.free() }
    defer { starts.free() }

    let absent = count - kept_rows(keys, true)
    let present = count - absent
    let dense_total = if keeps and absent > 0 { distinct - 1 } else { distinct }
    var mask = out.valid
    if mask is null { mask = Buffer(count) }
    kernel_for("key_rank_assign").dispatch_async(
        [order, dense, starts, out.values, mask ?? out.values],
        count,
        [count, distinct, code, present, flag(keeps), flag(pct), dense_total]
    )
    if out.valid is null and mask is not null { mask.free() }
    return out
}

fn rank_method(name: str) -> int {
    return match name {
        "average" => 0,
        "min" => 1,
        "max" => 2,
        "first" => 3,
        "dense" => 4,
        _ => throw ValueError(f"unknown rank method {name}"),
    }
}

fn extreme(target: any, n: int, columns: any, keep: str, largest: bool) -> any {
    if n < 0 { throw ValueError(f"cannot take {n} rows") }
    if keep != "first" and keep != "last" and keep != "all" {
        throw ValueError(f"keep must be first, last or all, not {keep}")
    }
    let ranked = sort_key_columns(target, columns)
    let length = row_count(target)
    var wanted = [column for column in ranked]
    var ups = [not largest for _column in ranked]
    var tiebreak: Column? = null
    if keep == "last" {
        let backwards = descending_column(length)
        tiebreak = backwards
        wanted.push(backwards)
        ups.push(true)
    }
    let keys = encode_keys(wanted)
    let order = keys.order(ups, "last")
    let present = kept_rows(keys, true)
    var limit = if n < present { n } else { present }
    if keep == "all" and limit > 0 and limit < present {
        limit = tie_boundary(keys, order, length, limit, present)
    }
    let head = Buffer(limit)
    if limit > 0 {
        cached_kernel("col_copy_lane").dispatch_async([order, head], limit, [limit, 0, 0])
    }
    order.free()
    keys.free()
    let spare = tiebreak
    if spare is not null { spare.free() }
    let taken = reorder(target, head, limit, false)
    head.free()
    return taken
}

#: Where the run of ties straddling the cutoff ends.
#:
#: Two one-element downloads rather than a scan on the host: the dense code at
#: the cutoff says which run it is in, and the run table says where the next
#: run starts.
fn tie_boundary(keys: KeyEncoding, order: Buffer, length: int, limit: int, present: int) -> int {
    let runs = dense_runs(keys, order, length)
    let dense = runs[0]
    let starts = runs[1]
    let distinct = runs[2]
    defer { dense.free() }
    defer { starts.free() }
    let run = int(dense.download(limit - 1, 1)[0])
    if run + 1 >= distinct { return present }
    let after = int(starts.download(run + 1, 1)[0])
    return if after < present { after } else { present }
}

#: Which run of equal keys each sorted position falls in, where each run
#: starts, and how many runs there are.
#:
#: Ranking and a `keep: all` cutoff both need the ties, and both take them off
#: the encoding the order came from rather than comparing the values a second
#: way — two answers read from one sort cannot disagree. The caller owns both
#: lanes.
fn dense_runs(keys: KeyEncoding, positions: Buffer, length: int) -> (Buffer, Buffer, int) {
    let flags = keys.run_flags(positions)
    defer { flags.free() }
    let scanned = prefix_sum(flags, length)
    let offsets = scanned[0]
    let distinct = scanned[1]
    defer { offsets.free() }
    let starts = Buffer(distinct)
    kernel_for("key_positions_where").dispatch_async([flags, offsets, starts], length, [length])
    let dense = Buffer(length)
    kernel_for("key_dense_codes").dispatch_async([offsets, flags, dense], length, [length])
    return (dense, starts, distinct)
}

#: The group each row falls in, the first row of each group, and how many
#: groups there are.
#:
#: The shared middle of `factorize`, `value_counts`, `mode` and `duplicated`.
#: The caller owns both lanes.
fn group_rows(keys: KeyEncoding, dropna: bool) -> (Buffer, Buffer, int) {
    let length = keys.length
    let codes = Buffer(length)
    if length == 0 { return (codes, Buffer(0), 0) }
    let hashes = keys.hash()
    defer { hashes.free() }
    let capacity = next_capacity(length)
    let slots = Buffer(capacity)
    slots.fill_zero()
    defer { slots.free() }
    let slot_of_row = Buffer(length)
    defer { slot_of_row.free() }
    kernel_for("key_table_insert").dispatch_async(
        [keys.packed, slots, hashes, slot_of_row],
        length,
        [length, capacity - 1, keys.lane_count()]
    )
    kernel_for("key_table_claim").dispatch_async([slots, slot_of_row], length, [length])

    let absent = keys.missing ?? keys.packed
    let drop = dropna and keys.missing is not null
    let heads_of_row = Buffer(length)
    defer { heads_of_row.free() }
    let flags = Buffer(length)
    defer { flags.free() }
    kernel_for("key_table_head").dispatch_async(
        [slots, slot_of_row, absent, heads_of_row, flags],
        length,
        [length, flag(drop)]
    )
    let scanned = prefix_sum(flags, length)
    let offsets = scanned[0]
    let group_count = scanned[1]
    defer { offsets.free() }
    kernel_for("key_group_codes").dispatch_async(
        [offsets, heads_of_row, absent, codes],
        length,
        [length, flag(drop)]
    )
    let heads = Buffer(group_count)
    if group_count > 0 {
        kernel_for("key_positions_where").dispatch_async([flags, offsets, heads], length, [length])
    }
    return (codes, heads, group_count)
}

#: A table at least 1.5x the rows, rounded up to a power of two so the probe
#: wraps with a mask instead of a remainder.
fn next_capacity(length: int) -> int {
    let wanted = length + length // 2 + 8
    var capacity = 16
    while capacity < wanted { capacity = capacity << 1 }
    return capacity
}

#: How many rows each group holds, as device uint bits.
fn tally_groups(codes: Buffer, length: int, group_count: int) -> Buffer {
    let tally = Buffer(if group_count > 0 { group_count } else { 1 })
    tally.fill_zero()
    if length == 0 or group_count == 0 { return tally }
    kernel_for("key_group_tally").dispatch_async([codes, tally], length, [length])
    return tally
}

#: How many rows survive a `dropna`.
fn kept_rows(keys: KeyEncoding, dropna: bool) -> int {
    let absent = keys.missing
    if not dropna or absent is null or keys.length == 0 { return keys.length }
    return keys.length - int(reduce_lane(absent ?? keys.packed, null, keys.length, 0) + 0.5)
}

#: Sum, smallest or largest over a lane, finished on the host over 256
#: partials rather than in a second dispatch — one download of 256 floats
#: against a kernel launch and a device wait.
fn reduce_lane(values: Buffer, valid: Buffer?, count: int, mode: int) -> float {
    if count == 0 { return 0.0 }
    let partials = Buffer(REDUCE_GROUPS)
    kernel_for("key_reduce").dispatch_async(
        [values, valid ?? values, partials],
        REDUCE_GROUPS * REDUCE_GROUP,
        [count, flag(valid is not null), mode, REDUCE_GROUPS * REDUCE_GROUP],
        group_size: REDUCE_GROUP
    )
    let read = partials.download()
    partials.free()
    var best = read[0]
    for at in 1..read.len() {
        if mode == 0 { best += read[at] }
        if mode == 1 and read[at] < best { best = read[at] }
        if mode == 2 and read[at] > best { best = read[at] }
    }
    return best
}

#: The rows a flag lane marks, packed, and how many there are.
fn positions_where(flags: Buffer, length: int) -> (Buffer, int) {
    let scanned = prefix_sum(flags, length)
    let offsets = scanned[0]
    let kept = scanned[1]
    defer { offsets.free() }
    let out = Buffer(kept)
    if length > 0 and kept > 0 {
        kernel_for("key_positions_where").dispatch_async([flags, offsets, out], length, [length])
    }
    return (out, kept)
}

fn columns_of(target: any) -> list[Column] {
    if target is Column { return [target] }
    if target is Series { return [target.values] }
    if target is DataFrame { return [column for column in target.columns] }
    throw TypeError("expected a Column, a Series or a DataFrame")
}

fn names_of(target: any) -> list[str?] {
    var out: list[str?] = []
    if target is Series {
        out.push(target.name)
        return out
    }
    if target is DataFrame {
        for name in target.names { out.push(name) }
        return out
    }
    out.push(null)
    return out
}

fn label_of(target: any) -> str? {
    if target is Series { return target.name }
    return null
}

fn counted_series(labels: list[Column], counts: Column, names: list[str?], named: str) -> Series {
    return Series(counts, index: Index.of(labels, names), name: named)
}

fn row_count(target: any) -> int {
    if target is Column { return target.len() }
    if target is Series { return target.len() }
    if target is DataFrame { return target.len() }
    throw TypeError("expected a Column, a Series or a DataFrame")
}

fn index_of(target: any) -> Index {
    if target is Series { return target.index }
    if target is DataFrame { return target.index }
    throw TypeError("a bare Column carries no labels to sort by")
}

fn named_columns(frame: DataFrame, names: any) -> list[Column] {
    if names is str { return [frame.column(names)] }
    if names is list {
        var out: list[Column] = []
        for name in names { out.push(frame.column(name)) }
        if out.len() == 0 { throw ValueError("an empty list names no columns") }
        return out
    }
    throw TypeError("columns are named by a string or a list of strings")
}

fn subset_columns(frame: any, subset: any) -> list[Column] {
    if subset is null { return columns_of(frame) }
    if frame is DataFrame { return named_columns(frame, subset) }
    throw TypeError("a subset names the columns of a frame")
}

fn sort_key_columns(target: any, by: any) -> list[Column] {
    if target is DataFrame {
        if by is null { throw ValueError("sorting a frame needs the columns to sort by") }
        return named_columns(target, by)
    }
    if by is not null {
        throw ValueError("a series or a column sorts by its own values, not by a name")
    }
    return columns_of(target)
}

fn spread_any(ascending: any, wanted: int) -> list[bool]? {
    if ascending is bool {
        var out: list[bool] = []
        for _at in 0..wanted { out.push(ascending) }
        return out
    }
    if ascending is list {
        var out: list[bool] = []
        for value in ascending { out.push(value) }
        return out
    }
    throw TypeError("ascending is a bool or one bool per key")
}

fn duplicate_mode(keep: any) -> int {
    if keep is bool {
        if not keep { return 2 }
        throw ValueError("keep is first, last or false")
    }
    if keep == "first" { return 0 }
    if keep == "last" { return 1 }
    throw ValueError(f"keep must be first, last or false, not {keep}")
}

#: The rows `positions` names, carrying whatever labels the target had.
#:
#: One positions lane serves the columns and the index alike, which is why
#: every reordering here ends in a `take` rather than a filter: `compress`
#: would leave the index behind.
fn reorder(target: any, positions: Buffer, length: int, ignore_index: bool) -> any {
    if target is Column { return target.take(positions, length) }
    if target is Series {
        let labels = if ignore_index {
            Index.range(length)
        } else {
            target.index.take(positions, length)
        }
        return Series(target.values.take(positions, length), index: labels, name: target.name)
    }
    if target is DataFrame {
        let labels = if ignore_index {
            Index.range(length)
        } else {
            target.index.take(positions, length)
        }
        return target._replace(
            columns: target.columns.map(|column| column.take(positions, length)),
            index: labels
        )
    }
    throw TypeError("expected a Column, a Series or a DataFrame")
}

fn level_position(index: Index, level: any) -> int {
    if level is int {
        let at = if level < 0 { index.nlevels() + level } else { level }
        if at < 0 or at >= index.nlevels() {
            throw IndexError(f"level {level} is outside an index of {index.nlevels()}")
        }
        return at
    }
    if level is str {
        for (at, name) in index.names.enumerate() {
            if name == level { return at }
        }
        throw KeyError(f"no index level named {level}")
    }
    throw TypeError("a level is named by a position or a name")
}

fn resolve_levels(index: Index, level: any) -> list[int] {
    if level is null {
        var all: list[int] = []
        for at in 0..index.nlevels() { all.push(at) }
        return all
    }
    if level is list {
        var chosen: list[int] = []
        for entry in level { chosen.push(level_position(index, entry)) }
        return chosen
    }
    return [level_position(index, level)]
}

fn descending_positions(length: int) -> Buffer {
    let lane = Buffer(length)
    if length > 0 { kernel_for("key_descending").dispatch_async([lane], length, [length]) }
    return lane
}

#: An `Int64` column holding `length - 1` down to 0.
#:
#: The tiebreak that turns `keep: last` into an ordinary key: a stable sort
#: already leaves ties in row order, so reversing the row number as the least
#: significant key is the whole of "prefer the later row".
fn descending_column(length: int) -> Column {
    let column = Column.allocate(Dtype.Int64, length)
    if length > 0 {
        kernel_for("key_descending").dispatch_async([column.values], length, [length])
        (column.wide ?? column.values).fill_zero()
    }
    return column
}

#: Counts over equal-width intervals of a numeric column.
#:
#: The edges and the labels are worked out on the host, over `bins` of them and
#: never over the rows: the rows are binned by one atomic tally, and the only
#: thing downloaded is the `bins` counters.
fn binned_counts(
    column: Column,
    bins: any,
    normalize: bool,
    sort: bool,
    ascending: bool,
    named: str?
) -> Series {
    if bins is not int { throw TypeError("bins is a number of intervals") }
    if bins <= 0 { throw ValueError(f"cannot cut a column into {bins} intervals") }
    if not column.dtype.is_numeric() {
        throw TypeError(f"cannot bin a {column.dtype.name()} column")
    }
    let count = column.len()
    let borrowed = column.borrow_float_lane()
    let values = borrowed[0]
    let owned = borrowed[1]
    let mask = column.valid
    var low = reduce_lane(values, mask, count, 1)
    var high = reduce_lane(values, mask, count, 2)
    #: No present row leaves the reductions at their identities, which are the
    #: two ends of float32 and would produce edges nothing could fall between.
    if high < low {
        low = 0.0
        high = 0.0
    }
    #: pandas widens the first interval by a thousandth of the range so the
    #: smallest value falls inside a right-closed one, which is why its first
    #: label always looks slightly too low.
    let span = high - low
    let pad = if span > 0.0 { span * 0.001 } else { 0.001 }
    let first_edge = low - pad
    let step = (high - first_edge) / float(bins)
    let inverse = if step > 0.0 { 1.0 / step } else { 0.0 }
    let tally = Buffer(bins)
    tally.fill_zero()
    if count > 0 {
        kernel_for("key_bin_tally").dispatch_async(
            [values, mask ?? values, tally],
            count,
            [count, bins, flag(mask is not null), f32_bits(first_edge), f32_bits(inverse)]
        )
    }
    let read = Buffer(bins)
    kernel_for("key_uint_to_float").dispatch_async([tally, read], bins, [bins])
    let counted = read.download()
    read.free()
    tally.free()
    if owned { values.free() }

    var labels: list[str?] = []
    var totals: list[float] = []
    let scale = if normalize and count > 0 { 1.0 / float(count) } else { 1.0 }
    for at in 0..bins {
        let lower = first_edge + float(at) * step
        let upper = if at == bins - 1 { high } else { first_edge + float(at + 1) * step }
        labels.push(f"({lower}, {upper}]")
        totals.push(counted[at] * scale)
    }
    var order: list[int] = []
    for at in 0..bins { order.push(at) }
    #: `bins` is a handful, so ordering them on the host is cheaper than the
    #: dispatch a device sort would cost.
    if sort { order = order.sort(key: |at| totals[at], reverse: not ascending) }
    var final_labels: list[str?] = []
    var final_counts: list[float] = []
    for at in order {
        final_labels.push(labels[at])
        final_counts.push(totals[at])
    }
    let edges = Column.from_strs(final_labels)
    let values_out = if normalize {
        Column.from_floats(final_counts)
    } else {
        Column.from_ints(final_counts.map(|total| int(total + 0.5)))
    }
    return Series(values_out, index: Index.of([edges], [named]), name: named_result(normalize))
}

fn named_result(normalize: bool) -> str {
    return if normalize { "proportion" } else { "count" }
}
