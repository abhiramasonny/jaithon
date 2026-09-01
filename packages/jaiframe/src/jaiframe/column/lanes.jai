#: The shader library and the untyped float32 lanes a column is built out of.
import std.gpu as gpu
from std.gpu import Buffer, Kernel

#: The largest number of rows a column may hold.
#:
#: Row positions travel between kernels in float32 lanes — `take`, `compress`,
#: the scan offsets, the sort permutation every later module builds — and
#: float32 counts integers exactly only to 2^24. A frame taller than this would
#: start aliasing rows onto each other silently, so it is refused at the
#: allocation instead.
pub const MAX_ROWS = 16777216

#: The radix of the two-lane integer encoding: a value is `wide * WIDE_SCALE +
#: values`, with `values` in `[0, WIDE_SCALE)`.
pub const WIDE_SCALE = 16777216

#: One past the largest magnitude the two lanes can hold exactly, 2^48.
pub const WIDE_LIMIT = 281474976710656

#: The position `take` reads as "no source row", which becomes a null.
#:
#: Joins mark unmatched rows with it, which is why it is a named constant and
#: not just a negative number.
pub const MISSING_POSITION = -1

const SCAN_GROUP = 256
pub const COUNT_GROUP = 256
pub const COUNT_GROUPS = 256

let SOURCE = """#include <metal_stdlib>
using namespace metal;

constexpr constant uint SCAN_GROUP = 256;
constexpr constant uint COUNT_GROUP = 256;
constexpr constant uint COUNT_GROUPS = 256;
constexpr constant long WIDE_SCALE = 16777216L;

/* The two lanes are only ever recombined in `long`. Doing it in `float` would
 * throw away everything below the top 24 bits, which is the whole reason the
 * second lane exists. */
inline long wide_value(float low, float high) {
    return (long)high * WIDE_SCALE + (long)low;
}

inline void wide_store(device float* low, device float* high, uint at, long value) {
    long hi = value >> 24;
    low[at] = (float)(value - (hi << 24));
    high[at] = (float)hi;
}

inline uint fmix32(uint h) {
    h ^= h >> 16;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13;
    h *= 0xC2B2AE35u;
    h ^= h >> 16;
    return h;
}

/* Order-sensitive, so a two-key hash of (a, b) differs from (b, a). Multi-key
 * group-by chains one call per key column over the same lane. */
inline uint hash_combine(uint accumulated, uint incoming) {
    return accumulated ^ (incoming + 0x9E3779B9u + (accumulated << 6) + (accumulated >> 2));
}

/* -0.0 hashes as 0.0 and every NaN hashes alike, so a key column groups the
 * way a reader expects rather than the way the bit pattern happens to fall. */
inline uint hash_slot(float value) {
    float v = value;
    if (v == 0.0f) v = 0.0f;
    if (isnan(v)) return fmix32(0x7FC00000u);
    return fmix32(as_type<uint>(v));
}

kernel void col_iota(device float* out [[buffer(0)]],
                     constant uint& count [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out[gid] = (float)gid;
}

kernel void col_ramp(device float* values [[buffer(0)]],
                     device float* wide [[buffer(1)]],
                     constant uint& count [[buffer(2)]],
                     constant int& start [[buffer(3)]],
                     constant int& step [[buffer(4)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    wide_store(values, wide, gid, (long)start + (long)gid * (long)step);
}

kernel void col_copy_lane(device const float* src [[buffer(0)]],
                          device float* dst [[buffer(1)]],
                          constant uint& count [[buffer(2)]],
                          constant uint& src_offset [[buffer(3)]],
                          constant uint& dst_offset [[buffer(4)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    dst[dst_offset + gid] = src[src_offset + gid];
}

kernel void col_gather(device const float* src_values [[buffer(0)]],
                       device const float* src_wide [[buffer(1)]],
                       device const float* src_valid [[buffer(2)]],
                       device const float* positions [[buffer(3)]],
                       device float* dst_values [[buffer(4)]],
                       device float* dst_wide [[buffer(5)]],
                       device float* dst_valid [[buffer(6)]],
                       constant uint& count [[buffer(7)]],
                       constant uint& src_length [[buffer(8)]],
                       constant uint& has_wide [[buffer(9)]],
                       constant uint& src_has_valid [[buffer(10)]],
                       constant uint& dst_has_valid [[buffer(11)]],
                       constant uint& allow_missing [[buffer(12)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    int at = (int)positions[gid];
    /* Clamped rather than trusted. A stray position would otherwise read off
     * the end of the allocation, and a device out-of-bounds read is not a
     * diagnosable failure, it is a wrong answer somewhere else later. */
    bool outside = at < 0 || at >= (int)src_length;
    int safe = outside ? 0 : at;
    if (outside && allow_missing != 0u) {
        dst_values[gid] = 0.0f;
        if (has_wide != 0u) dst_wide[gid] = 0.0f;
        if (dst_has_valid != 0u) dst_valid[gid] = 0.0f;
        return;
    }
    dst_values[gid] = src_values[safe];
    if (has_wide != 0u) dst_wide[gid] = src_wide[safe];
    if (dst_has_valid != 0u) {
        dst_valid[gid] = (src_has_valid != 0u) ? src_valid[safe] : 1.0f;
    }
}

kernel void col_scatter(device const float* src_values [[buffer(0)]],
                        device const float* src_wide [[buffer(1)]],
                        device const float* src_valid [[buffer(2)]],
                        device const float* flags [[buffer(3)]],
                        device const float* offsets [[buffer(4)]],
                        device float* dst_values [[buffer(5)]],
                        device float* dst_wide [[buffer(6)]],
                        device float* dst_valid [[buffer(7)]],
                        constant uint& count [[buffer(8)]],
                        constant uint& has_wide [[buffer(9)]],
                        constant uint& src_has_valid [[buffer(10)]],
                        constant uint& dst_has_valid [[buffer(11)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (flags[gid] == 0.0f) return;
    uint at = (uint)offsets[gid];
    dst_values[at] = src_values[gid];
    if (has_wide != 0u) dst_wide[at] = src_wide[gid];
    if (dst_has_valid != 0u) {
        dst_valid[at] = (src_has_valid != 0u) ? src_valid[gid] : 1.0f;
    }
}

kernel void col_mask_flags(device const float* values [[buffer(0)]],
                           device const float* valid [[buffer(1)]],
                           device float* out [[buffer(2)]],
                           constant uint& count [[buffer(3)]],
                           constant uint& has_valid [[buffer(4)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    bool keep = values[gid] != 0.0f;
    if (has_valid != 0u && valid[gid] == 0.0f) keep = false;
    out[gid] = keep ? 1.0f : 0.0f;
}

/* Hillis-Steele within a threadgroup, block totals out the side. The inputs
 * are 0/1 flags and MAX_ROWS is 2^24, so every partial sum is an integer a
 * float32 holds exactly and the scan is not an approximation. */
kernel void col_scan_block(device const float* input [[buffer(0)]],
                           device float* out [[buffer(1)]],
                           device float* totals [[buffer(2)]],
                           constant uint& count [[buffer(3)]],
                           uint gid [[thread_position_in_grid]],
                           uint lid [[thread_index_in_threadgroup]],
                           uint wid [[threadgroup_position_in_grid]]) {
    threadgroup float scratch[SCAN_GROUP];
    float own = (gid < count) ? input[gid] : 0.0f;
    scratch[lid] = own;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1u; stride < SCAN_GROUP; stride <<= 1) {
        float add = (lid >= stride) ? scratch[lid - stride] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        scratch[lid] += add;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (gid < count) out[gid] = scratch[lid] - own;
    if (lid == SCAN_GROUP - 1u) totals[wid] = scratch[lid];
}

kernel void col_scan_add(device float* out [[buffer(0)]],
                         device const float* offsets [[buffer(1)]],
                         constant uint& count [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out[gid] += offsets[gid / SCAN_GROUP];
}

/* The total belongs at the end of the offsets lane so that reading it is one
 * one-element download rather than two plus a host add. */
kernel void col_scan_total(device float* out [[buffer(0)]],
                           device const float* flags [[buffer(1)]],
                           constant uint& count [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid != 0u) return;
    out[count] = out[count - 1u] + flags[count - 1u];
}

/* A fixed grid that strides the column, not one thread per row: the fold costs
 * eight barriers, and paying that per row rather than per group is what makes
 * a naive count slower than the read it is counting. */
kernel void col_count_nulls(device const float* valid [[buffer(0)]],
                            device float* partials [[buffer(1)]],
                            constant uint& count [[buffer(2)]],
                            uint gid [[thread_position_in_grid]],
                            uint lid [[thread_index_in_threadgroup]],
                            uint wid [[threadgroup_position_in_grid]]) {
    threadgroup float scratch[COUNT_GROUP];
    float total = 0.0f;
    for (uint i = gid; i < count; i += COUNT_GROUP * COUNT_GROUPS) {
        if (valid[i] == 0.0f) total += 1.0f;
    }
    scratch[lid] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = COUNT_GROUP / 2u; stride > 0u; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) partials[wid] = scratch[0];
}

kernel void col_cast(device const float* src_values [[buffer(0)]],
                     device const float* src_wide [[buffer(1)]],
                     device float* dst_values [[buffer(2)]],
                     device float* dst_wide [[buffer(3)]],
                     constant uint& count [[buffer(4)]],
                     constant uint& from_wide [[buffer(5)]],
                     constant uint& to_wide [[buffer(6)]],
                     constant uint& to_bool [[buffer(7)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    long exact = (from_wide != 0u) ? wide_value(src_values[gid], src_wide[gid]) : 0L;
    float loose = (from_wide != 0u) ? (float)exact : src_values[gid];
    if (to_bool != 0u) {
        dst_values[gid] = (loose != 0.0f) ? 1.0f : 0.0f;
        return;
    }
    if (to_wide != 0u) {
        wide_store(dst_values, dst_wide, gid,
                   (from_wide != 0u) ? exact : (long)trunc(loose));
        return;
    }
    dst_values[gid] = loose;
}

kernel void col_wide_to_float(device const float* values [[buffer(0)]],
                              device const float* wide [[buffer(1)]],
                              device float* out [[buffer(2)]],
                              constant uint& count [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out[gid] = (float)wide_value(values[gid], wide[gid]);
}

kernel void col_remap(device const float* src [[buffer(0)]],
                      device float* dst [[buffer(1)]],
                      device const float* mapping [[buffer(2)]],
                      constant uint& count [[buffer(3)]],
                      constant uint& levels [[buffer(4)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    int code = (int)src[gid];
    dst[gid] = (code < 0 || code >= (int)levels) ? 0.0f : mapping[code];
}

/* The hash lane is uint bits, not numbers. It is written and read on the
 * device only; downloading it as floats gives nonsense, and a NaN pattern
 * would not survive the trip through a double anyway. */
kernel void col_hash(device const float* values [[buffer(0)]],
                     device const float* wide [[buffer(1)]],
                     device const float* valid [[buffer(2)]],
                     device uint* out [[buffer(3)]],
                     constant uint& count [[buffer(4)]],
                     constant uint& has_wide [[buffer(5)]],
                     constant uint& has_valid [[buffer(6)]],
                     constant uint& seed [[buffer(7)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint h;
    if (has_valid != 0u && valid[gid] == 0.0f) {
        h = fmix32(0xA5A5A5A5u ^ seed);
    } else {
        h = fmix32(hash_slot(values[gid]) ^ seed);
        if (has_wide != 0u) h = fmix32(h ^ hash_slot(wide[gid]));
    }
    out[gid] = hash_combine(out[gid], h);
}

kernel void col_equal_rows(device const float* a_values [[buffer(0)]],
                           device const float* a_wide [[buffer(1)]],
                           device const float* a_valid [[buffer(2)]],
                           device const float* b_values [[buffer(3)]],
                           device const float* b_wide [[buffer(4)]],
                           device const float* b_valid [[buffer(5)]],
                           device const float* left [[buffer(6)]],
                           device const float* right [[buffer(7)]],
                           device float* out [[buffer(8)]],
                           constant uint& count [[buffer(9)]],
                           constant uint& has_wide [[buffer(10)]],
                           constant uint& a_has_valid [[buffer(11)]],
                           constant uint& b_has_valid [[buffer(12)]],
                           constant uint& null_equal [[buffer(13)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (out[gid] == 0.0f) return;
    int la = (int)left[gid];
    int rb = (int)right[gid];
    if (la < 0 || rb < 0) {
        out[gid] = 0.0f;
        return;
    }
    bool a_null = (a_has_valid != 0u) && a_valid[la] == 0.0f;
    bool b_null = (b_has_valid != 0u) && b_valid[rb] == 0.0f;
    if (a_null || b_null) {
        out[gid] = (a_null && b_null && null_equal != 0u) ? 1.0f : 0.0f;
        return;
    }
    bool same = a_values[la] == b_values[rb];
    if (same && has_wide != 0u) same = a_wide[la] == b_wide[rb];
    out[gid] = same ? 1.0f : 0.0f;
}
"""

var _kernels: dict[str, Kernel] = {}

#: The compiled pipeline for one entry point in the column shader library.
pub fn cached_kernel(entry: str) -> Kernel {
    let existing = _kernels.get(entry, null)
    if existing is not null { return existing }
    let kernel = Kernel.compile(SOURCE, entry)
    _kernels[entry] = kernel
    return kernel
}

#: Reinterpret a float's bits as an integer, so a kernel can take it as a
#: `constant uint&` and read it back with `as_type<float>`.
pub fn f32_bits(value: float) -> int {
    if value != value { return 0x7FC00000 }
    var magnitude = value
    var sign = 0
    if magnitude < 0.0 {
        sign = 1
        magnitude = -magnitude
    }
    if magnitude == 0.0 { return sign << 31 }
    if magnitude >= 3.4028234663852886e38 { return sign << 31 | 0x7F800000 }
    if magnitude <= 1.401298464324817e-45 { return sign << 31 }
    let split = __prim__.f64_frexp(magnitude)
    let biased = split[1] + 126
    if biased <= 0 {
        let fraction = int(__prim__.f64_ldexp(magnitude, 149))
        return sign << 31 | fraction & 0x7FFFFF
    }
    if biased >= 255 { return sign << 31 | 0x7F800000 }
    let fraction = int((split[0] * 2.0 - 1.0) * 8388608.0 + 0.5)
    return sign << 31 | biased << 23 | fraction & 0x7FFFFF
}

#: Two's-complement bits of `value` as a non-negative int.
#:
#: Kernel scalars are `uint32_t` and a negative one is rejected before it
#: reaches Metal, so a signed scalar goes through here and is declared
#: `constant int&` on the other side.
pub fn u32(value: int) -> int { return value & 0xFFFFFFFF }

pub fn flag(value: bool) -> int { return if value { 1 } else { 0 } }

pub fn require_device(what: str) -> void {
    if not gpu.is_available() {
        throw RuntimeError(f"jaiframe needs a GPU to {what}")
    }
}

#: Split an integer into the `(values, wide)` slot pair that stores it.
#:
#: Floor division, not truncation, so a negative value lands with its low half
#: in `[0, WIDE_SCALE)` and the device's arithmetic shift agrees with the
#: host's.
pub fn wide_pack(value: int) -> (float, float) {
    if value < -WIDE_LIMIT or value >= WIDE_LIMIT {
        throw ValueError(f"{value} does not fit in the 48 bits an int64 column stores")
    }
    let high = value // WIDE_SCALE
    return (float(value - high * WIDE_SCALE), float(high))
}

#: Recombine a `(values, wide)` slot pair into the integer it encodes.
pub fn wide_unpack(low: float, high: float) -> int {
    return int(high) * WIDE_SCALE + int(low)
}

#: A positions lane holding 0, 1, ... `length - 1`.
#:
#: The identity permutation everything reordering starts from: a sort hands
#: back one of these permuted, a filter a shorter one, and `take` consumes it.
#: Built on the device because a host list of a million positions is a boxed
#: float apiece and costs seconds before the first kernel has run.
pub fn iota(length: int) -> Buffer {
    if length < 0 { throw ValueError(f"cannot number {length} rows") }
    if length > MAX_ROWS {
        throw ValueError(f"{length} rows exceeds the {MAX_ROWS} a float32 position lane addresses")
    }
    require_device("build a positions lane")
    let lane = Buffer(length)
    if length == 0 { return lane }
    cached_kernel("col_iota").dispatch_async([lane], length, [length])
    return lane
}

#: A positions lane holding the rows a host list names.
#:
#: For positions that were already on the host — the rows a test spells out,
#: the handful a `head` wants. Anything derived from a column should stay in a
#: lane the device wrote; this pays an upload and a boxed float per entry.
#:
#: `MISSING_POSITION` is allowed through, so an outer join can build its right
#: side here and let `take(allow_missing: true)` turn the gaps into nulls.
pub fn positions_buffer(rows: list[int]) -> Buffer {
    require_device("build a positions lane")
    var slots: list[float] = []
    for row in rows {
        if row < MISSING_POSITION or row >= MAX_ROWS {
            throw ValueError(f"row {row} is not a position a float32 lane holds")
        }
        slots.push(float(row))
    }
    let lane = Buffer(rows.len())
    if rows.len() > 0 { lane.upload(slots) }
    return lane
}

#: Exclusive prefix sum of a lane of 0/1 flags, and the total.
#:
#: The returned lane has `length + 1` slots: slot `i` counts the flags set
#: before row `i`, which is where a scatter should put row `i`, and the last
#: slot is the total, which is how long the result has to be. A filter, a
#: `dropna`, the bucket offsets of a group-by and the output sizing of a join
#: are all this call plus a scatter. The caller owns the buffer.
#:
#: One download, of the one number the host actually needs.
pub fn prefix_sum(flags: Buffer, length: int) -> (Buffer, int) {
    let offsets = scan_exclusive(flags, length)
    if length == 0 { return (offsets, 0) }
    cached_kernel("col_scan_total").dispatch_async([offsets, flags], 1, [length], group_size: 1)
    return (offsets, int(offsets.download(length, 1)[0]))
}

#: The scan without the total.
#:
#: Blocks of 256 scanned in threadgroup memory, then the block totals scanned
#: the same way and added back, so 2^24 rows is three levels of dispatch rather
#: than a host loop over 65536 block sums. Asking each level for its total
#: would put a device wait between the levels, which is why that lives in
#: `prefix_sum` and this recursion does without it.
fn scan_exclusive(flags: Buffer, length: int) -> Buffer {
    if length < 0 { throw ValueError(f"cannot scan {length} flags") }
    require_device("scan a flag lane")
    let offsets = Buffer(length + 1)
    if length == 0 {
        offsets.fill_zero()
        return offsets
    }
    let blocks = (length + SCAN_GROUP - 1) // SCAN_GROUP
    let totals = Buffer(blocks)
    cached_kernel("col_scan_block").dispatch_async(
        [flags, offsets, totals],
        blocks * SCAN_GROUP,
        [length],
        group_size: SCAN_GROUP
    )
    if blocks > 1 {
        let carried = scan_exclusive(totals, blocks)
        cached_kernel("col_scan_add").dispatch_async([offsets, carried], length, [length])
        carried.free()
    }
    totals.free()
    return offsets
}

pub fn copy_lane(
    src: Buffer,
    dst: Buffer,
    count: int,
    src_offset: int = 0,
    dst_offset: int = 0
) -> void {
    if count == 0 { return }
    cached_kernel("col_copy_lane").dispatch_async(
        [src, dst],
        count,
        [count, src_offset, dst_offset]
    )
}
