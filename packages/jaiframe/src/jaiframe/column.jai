#: The typed nullable column store every other jaiframe module is built on.
#:
#: Storage is one contiguous `std.gpu.Buffer` of float32 slots per lane,
#: exactly as jaicv's `Mat` is, so a whole column is one allocation and every
#: operation over it is one dispatch.
#:
#: A float32 slot is exact only to 2^24, and a pandas clone has to carry exact
#: int64 and exact millisecond timestamps. So a column is up to three lanes:
#:
#: - `values` — the payload for `Float64` and `Bool`, and the dictionary code
#:   for `Str` and `Category`.
#: - `wide` — present only for `Int64` and `Datetime`, where the value is
#:   `wide * 2^24 + values`. Both halves are integral and below 2^24, so both
#:   are exact in float32 and the pair is exact to 48 bits. Metal reconstructs
#:   it in `long`, never in `float`.
#: - `valid` — 1.0 where a row holds a value, 0.0 where it is missing. `null`
#:   means the column has no missing values at all, which is what lets a kernel
#:   skip the mask instead of reading a lane of ones.
#:
#: `Datetime` is milliseconds since the Unix epoch — datetime64[ms]. 48 bits of
#: milliseconds is +/-8900 years; 48 bits of nanoseconds would be +/-3 days,
#: which is not a calendar. `Float64` is a single float32 lane and is named for
#: source compatibility, the same compromise jaicv makes with `CV_64F`.
#:
#: `Str` and `Category` share one representation: codes in `values` and a
#: `levels` table on the side. That is the only encoding that keeps a string
#: column contiguous, and it is what makes string equality an integer compare
#: and therefore hashable by the same kernel that hashes numbers.
#:
#: Go through the accessors here. Nothing outside this file should read a lane
#: directly, because which lanes exist depends on the dtype and on whether the
#: column has ever held a null.
#:
#: ```
#: let ages = Column.from_ints([31, 47, 29])
#: defer { ages.free() }
#: ages.get(1).to_int()          # 47
#: ages.null_count()             # 0
#: ```
import std.gpu as gpu
from std.gpu import Buffer, Kernel
from std.core import Printable
from std.math import NAN, is_nan

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
const COUNT_GROUP = 256
const COUNT_GROUPS = 256

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
fn u32(value: int) -> int { return value & 0xFFFFFFFF }

fn flag(value: bool) -> int { return if value { 1 } else { 0 } }

fn require_device(what: str) -> void {
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

#: How a column stores its values and what operations mean over it.
pub enum Dtype {
    #: A single float32 lane. Named `Float64` for source compatibility with
    #: pandas, the same compromise jaicv's `CV_64F` makes; the storage is
    #: float32 and so is the precision.
    Float64,
    #: Two lanes, exact to 48 bits.
    Int64,
    #: A single lane holding 0.0 or 1.0.
    Bool,
    #: Dictionary codes in `values`, the strings in `levels`.
    Str,
    #: The same storage as `Str`. What differs is intent: the levels are the
    #: domain, so group-by may report an empty group for a level no row uses.
    Category,
    #: Two lanes of milliseconds since the Unix epoch — datetime64[ms].
    #: 48 bits of milliseconds is +/-8900 years. Nanoseconds, pandas' unit,
    #: would be +/-3 days.
    Datetime,

    #: The pandas spelling of this dtype.
    pub fn name(self) -> str {
        return match self {
            Dtype.Float64 => "float64",
            Dtype.Int64 => "int64",
            Dtype.Bool => "bool",
            Dtype.Str => "str",
            Dtype.Category => "category",
            Dtype.Datetime => "datetime64[ms]",
        }
    }

    #: Whether arithmetic and numeric reductions are defined over it.
    #:
    #: `Bool` counts, as it does in pandas, because summing a mask is the
    #: point of one. `Datetime` does not: differences of timestamps are
    #: durations, which is the timeseries module's business, not arithmetic's.
    pub fn is_numeric(self) -> bool {
        return match self {
            Dtype.Float64 | Dtype.Int64 | Dtype.Bool => true,
            _ => false,
        }
    }

    #: Whether a column of this dtype carries a second `wide` lane.
    pub fn is_wide(self) -> bool {
        return match self {
            Dtype.Int64 | Dtype.Datetime => true,
            _ => false,
        }
    }

    #: Whether `values` holds codes into a `levels` table rather than a payload.
    pub fn is_dictionary(self) -> bool {
        return match self {
            Dtype.Str | Dtype.Category => true,
            _ => false,
        }
    }

    #: The dtype `name` spells, for `astype("int64")` and for CSV headers.
    pub static fn from_name(text: str) -> Dtype {
        return match text {
            "float64" | "float" | "float32" | "f8" => Dtype.Float64,
            "int64" | "int" | "i8" => Dtype.Int64,
            "bool" | "boolean" => Dtype.Bool,
            "str" | "string" | "object" => Dtype.Str,
            "category" => Dtype.Category,
            "datetime64[ms]" | "datetime64[ns]" | "datetime" => Dtype.Datetime,
            _ => throw ValueError(f"unknown dtype {text}"),
        }
    }
}

#: One cell: a dtype tag, the payload for that dtype, and a missing flag.
#:
#: Which field carries the payload follows the dtype, the same way the lanes
#: do. `Float64` and `Bool` use `number`, `Int64` and `Datetime` use `integer`,
#: `Str` and `Category` use `text`. A missing cell leaves all three at their
#: zero and answers `is_na`.
pub class Value: Printable {
    #: What the cell would be if it were not missing.
    pub let dtype: Dtype
    #: Payload for `Float64` and `Bool`.
    pub let number: float
    #: Payload for `Int64` and `Datetime`.
    pub let integer: int
    #: Payload for `Str` and `Category`.
    pub let text: str?
    #: Whether the cell holds nothing.
    pub let missing: bool

    #: Build a cell directly. The `of_*` constructors are the ones to reach for.
    pub fn init(
        self,
        dtype: Dtype,
        number: float = 0.0,
        integer: int = 0,
        text: str? = null,
        missing: bool = false
    ) {
        self.dtype = dtype
        self.number = number
        self.integer = integer
        self.text = text
        self.missing = missing
    }

    #: A `Float64` cell.
    pub static fn of_float(value: float) -> Value {
        return Value(Dtype.Float64, number: value)
    }

    #: An `Int64` cell.
    pub static fn of_int(value: int) -> Value {
        return Value(Dtype.Int64, integer: value)
    }

    #: A `Bool` cell.
    pub static fn of_bool(value: bool) -> Value {
        return Value(Dtype.Bool, number: if value { 1.0 } else { 0.0 })
    }

    #: A `Str` cell, or a `Category` one when `dtype` says so.
    pub static fn of_str(value: str, dtype: Dtype = Dtype.Str) -> Value {
        if not dtype.is_dictionary() {
            throw ValueError(f"{dtype.name()} does not hold text")
        }
        return Value(dtype, text: value)
    }

    #: A `Datetime` cell, milliseconds since the Unix epoch.
    pub static fn of_datetime(millis: int) -> Value {
        return Value(Dtype.Datetime, integer: millis)
    }

    #: The missing cell of `dtype`.
    pub static fn null_of(dtype: Dtype) -> Value {
        return Value(dtype, missing: true)
    }

    #: Whether the cell holds nothing.
    pub fn is_na(self) -> bool { return self.missing }

    #: The payload as a float, whatever lane it came from.
    #:
    #: A missing cell is NaN, which is what a float column stores for one and
    #: what a plot or a mean has to see.
    pub fn to_float(self) -> float {
        if self.missing { return NAN }
        if self.dtype.is_wide() { return float(self.integer) }
        if self.dtype.is_dictionary() {
            throw TypeError(f"a {self.dtype.name()} value has no numeric payload")
        }
        return self.number
    }

    #: The payload as an integer, truncated when it came from a float lane.
    pub fn to_int(self) -> int {
        if self.missing { throw ValueError("a missing value has no integer") }
        if self.dtype.is_wide() { return self.integer }
        if self.dtype.is_dictionary() {
            throw TypeError(f"a {self.dtype.name()} value has no numeric payload")
        }
        return int(self.number)
    }

    #: The payload as a truth value: nonzero, or a non-empty string.
    pub fn to_bool(self) -> bool {
        if self.missing { throw ValueError("a missing value has no truth value") }
        if self.dtype.is_dictionary() { return (self.text ?? "").len() > 0 }
        if self.dtype.is_wide() { return self.integer != 0 }
        return self.number != 0.0
    }

    #: The cell as text. `NA` for a missing one.
    pub fn to_str(self) -> str {
        if self.missing { return "NA" }
        if self.dtype == Dtype.Bool { return if self.number != 0.0 { "true" } else { "false" } }
        return match self.dtype {
            Dtype.Str | Dtype.Category => self.text ?? "",
            Dtype.Int64 | Dtype.Datetime => f"{self.integer}",
            _ => f"{self.number}",
        }
    }

    #: Cells are equal when both are missing, or when the same dtype family
    #: carries the same payload.
    #:
    #: Two missing cells compare equal whatever their dtypes, which is what
    #: makes the dtype-less `NA` usable against a column of any type. Note this
    #: is `NA == NA`, unlike pandas' propagating `NA`; a cell comparison here
    #: is a test, not an element of an expression.
    pub fn __eq__(self, other: any) -> bool {
        if other is not Value { return false }
        if self.missing or other.missing { return self.missing and other.missing }
        if self.dtype.is_dictionary() != other.dtype.is_dictionary() { return false }
        if self.dtype.is_dictionary() { return self.text == other.text }
        if self.dtype.is_wide() and other.dtype.is_wide() {
            return self.integer == other.integer
        }
        return self.to_float() == other.to_float()
    }

    fn __str__(self) -> str { return self.to_str() }
}

#: The missing cell, of no particular dtype.
#:
#: It equals the missing cell of every dtype, so `column.get(i) == NA` is the
#: null test whatever the column holds.
pub let NA = Value(Dtype.Float64, missing: true)

#: The aggregations reduce, group-by, rolling, resample and pivot all name.
#:
#: One enum rather than one per module, because a user writes
#: `df.groupby("k").agg(Agg.Mean)` and `df.rolling(3).agg(Agg.Mean)` and expects
#: the same word to mean the same thing.
pub enum Agg {
    #: Sum of the present values; 0 over an all-null group.
    Sum,
    #: Arithmetic mean of the present values.
    Mean,
    #: Smallest present value.
    Min,
    #: Largest present value.
    Max,
    #: Number of present values.
    Count,
    #: Number of rows, nulls included. This is what `Count` is not.
    Size,
    #: Product of the present values.
    Prod,
    #: Sample standard deviation, one degree of freedom.
    Std,
    #: Sample variance, one degree of freedom.
    Var,
    #: Middle present value, averaged over the two middles when even.
    Median,
    #: A named quantile; the caller carries the fraction alongside.
    Quantile,
    #: First present value in row order.
    First,
    #: Last present value in row order.
    Last,
    #: Number of distinct present values.
    Nunique,
    #: Whether any present value is truthy.
    Any,
    #: Whether every present value is truthy.
    All,

    #: The pandas spelling, which is also the suffix a named aggregation gets.
    pub fn name(self) -> str {
        return match self {
            Agg.Sum => "sum",
            Agg.Mean => "mean",
            Agg.Min => "min",
            Agg.Max => "max",
            Agg.Count => "count",
            Agg.Size => "size",
            Agg.Prod => "prod",
            Agg.Std => "std",
            Agg.Var => "var",
            Agg.Median => "median",
            Agg.Quantile => "quantile",
            Agg.First => "first",
            Agg.Last => "last",
            Agg.Nunique => "nunique",
            Agg.Any => "any",
            Agg.All => "all",
        }
    }

    #: The aggregation `name` spells, so `df.agg("mean")` reaches the enum.
    pub static fn from_name(text: str) -> Agg {
        return match text {
            "sum" => Agg.Sum,
            "mean" | "avg" | "average" => Agg.Mean,
            "min" => Agg.Min,
            "max" => Agg.Max,
            "count" => Agg.Count,
            "size" => Agg.Size,
            "prod" | "product" => Agg.Prod,
            "std" => Agg.Std,
            "var" => Agg.Var,
            "median" => Agg.Median,
            "quantile" => Agg.Quantile,
            "first" => Agg.First,
            "last" => Agg.Last,
            "nunique" => Agg.Nunique,
            "any" => Agg.Any,
            "all" => Agg.All,
            _ => throw ValueError(f"unknown aggregation {text}"),
        }
    }

    #: Whether the result is a count and so is `Int64` whatever went in.
    pub fn counts_rows(self) -> bool {
        return match self {
            Agg.Count | Agg.Size | Agg.Nunique => true,
            _ => false,
        }
    }

    #: Whether the result is one of the inputs rather than a new number, and so
    #: keeps the input's dtype and works over text as well as numbers.
    pub fn picks_a_row(self) -> bool {
        return match self {
            Agg.Min | Agg.Max | Agg.First | Agg.Last => true,
            _ => false,
        }
    }
}

#: One column of a frame: a dtype, a length, and the lanes that dtype needs.
#:
#: Lane ownership is one column, one set of buffers. Nothing here hands out a
#: view onto another column's allocation, because a frame is reshaped far more
#: often than an image is cropped and a shared lane would make every `free` a
#: question. The cost is that `copy` really copies; the benefit is that every
#: module can `defer { column.free() }` without asking who else is looking.
pub class Column {
    #: How the lanes are to be read.
    pub let dtype: Dtype
    #: Number of rows.
    pub let length: int
    #: The payload lane, or the dictionary code lane.
    pub var values: Buffer
    #: The high half of a 48-bit integer, present only for a wide dtype.
    pub var wide: Buffer?
    #: 1.0 where a row holds a value. `null` means no row is missing, which
    #: lets a kernel skip the mask entirely.
    pub var valid: Buffer?
    #: The dictionary for `Str` and `Category`, `null` for every other dtype.
    pub var levels: list[str]?
    var intern: dict[str, int]?
    var freed: bool

    #: Allocate the lanes `dtype` needs, with the contents left undefined.
    #:
    #: The caller writes every slot. This is the constructor a kernel-producing
    #: operation uses; the `from_*` ones are for host data.
    pub static fn allocate(dtype: Dtype, length: int, with_validity: bool = false) -> Column {
        if length < 0 { throw ValueError(f"a column cannot have {length} rows") }
        if length > MAX_ROWS {
            throw ValueError(
                f"{length} rows exceeds the {MAX_ROWS} a float32 position lane addresses"
            )
        }
        require_device("allocate a Column")
        let column = Column()
        column.dtype = dtype
        column.length = length
        column.values = Buffer(length)
        column.wide = if dtype.is_wide() { Buffer(length) } else { null }
        column.valid = if with_validity { Buffer(length) } else { null }
        column.levels = if dtype.is_dictionary() { [] } else { null }
        column.intern = if dtype.is_dictionary() { {} } else { null }
        column.freed = false
        return column
    }

    #: An empty shell. `allocate` and the `from_*` constructors fill it in.
    pub fn init(self) {
        self.dtype = Dtype.Float64
        self.length = 0
        self.values = Buffer(0)
        self.wide = null
        self.valid = null
        self.levels = null
        self.intern = null
        self.freed = false
    }

    #: A `Float64` column from host floats. NaN counts as missing unless
    #: `nulls` says otherwise, which is what a CSV reader wants.
    pub static fn from_floats(values: list[float], nulls: list[bool]? = null) -> Column {
        let mask = validity_of(values.len(), nulls, values.map(|v| is_nan(v)))
        let column = Column.allocate(Dtype.Float64, values.len(), with_validity: mask is not null)
        if values.len() > 0 { column.values.upload(values) }
        column.upload_validity(mask)
        return column
    }

    #: An `Int64` column, or a `Datetime` one when `dtype` says so.
    pub static fn from_ints(
        values: list[int],
        nulls: list[bool]? = null,
        dtype: Dtype = Dtype.Int64
    ) -> Column {
        if not dtype.is_wide() { throw ValueError(f"{dtype.name()} is not stored as two lanes") }
        let mask = validity_of(values.len(), nulls, null)
        let column = Column.allocate(dtype, values.len(), with_validity: mask is not null)
        var low: list[float] = []
        var high: list[float] = []
        for value in values {
            let split = wide_pack(value)
            low.push(split[0])
            high.push(split[1])
        }
        if values.len() > 0 {
            column.values.upload(low)
            (column.wide ?? column.values).upload(high)
        }
        column.upload_validity(mask)
        return column
    }

    #: A `Bool` column from host booleans.
    pub static fn from_bools(values: list[bool], nulls: list[bool]? = null) -> Column {
        let mask = validity_of(values.len(), nulls, null)
        let column = Column.allocate(Dtype.Bool, values.len(), with_validity: mask is not null)
        if values.len() > 0 {
            column.values.upload(values.map(|v| if v { 1.0 } else { 0.0 }))
        }
        column.upload_validity(mask)
        return column
    }

    #: A `Str` column, or a `Category` one, dictionary-encoding as it goes.
    #:
    #: A null entry is a missing row, not the empty string. Levels come out in
    #: first-appearance order, the way `factorize` numbers them; a module that
    #: wants them sorted reorders them itself.
    pub static fn from_strs(values: list[str?], dtype: Dtype = Dtype.Str) -> Column {
        if not dtype.is_dictionary() { throw ValueError(f"{dtype.name()} does not hold text") }
        var levels: list[str] = []
        var intern: dict[str, int] = {}
        var codes: list[float] = []
        var present: list[bool] = []
        var any_missing = false
        for entry in values {
            if entry is null {
                codes.push(0.0)
                present.push(false)
                any_missing = true
                continue
            }
            var code = intern.get(entry, -1)
            if code < 0 {
                code = levels.len()
                levels.push(entry)
                intern[entry] = code
            }
            codes.push(float(code))
            present.push(true)
        }
        let column = Column.allocate(dtype, values.len(), with_validity: any_missing)
        if values.len() > 0 { column.values.upload(codes) }
        column.upload_validity(if any_missing { present } else { null })
        column.levels = levels
        column.intern = intern
        return column
    }

    #: `length` copies of one cell.
    pub static fn full(dtype: Dtype, length: int, value: Value) -> Column {
        if value.is_na() { return Column.nulls(dtype, length) }
        let column = Column.allocate(dtype, length)
        if length == 0 { return column }
        if dtype.is_dictionary() {
            column.levels = [value.to_str()]
            column.intern = {value.to_str(): 0}
            column.values.fill(0.0)
            return column
        }
        if dtype.is_wide() {
            let split = wide_pack(value.to_int())
            column.values.fill(split[0])
            (column.wide ?? column.values).fill(split[1])
            return column
        }
        column.values.fill(value.to_float())
        return column
    }

    #: `length` missing rows.
    pub static fn nulls(dtype: Dtype, length: int) -> Column {
        let column = Column.allocate(dtype, length, with_validity: true)
        if length == 0 { return column }
        column.values.fill_zero()
        let high = column.wide
        if high is not null { high.fill_zero() }
        let mask = column.valid
        if mask is not null { mask.fill_zero() }
        return column
    }

    #: Number of rows, so a column reads like the other containers.
    pub fn len(self) -> int { return self.length }

    #: Whether the column holds no rows.
    pub fn is_empty(self) -> bool { return self.length == 0 }

    #: Number of entries in the dictionary, 0 when there is none.
    pub fn level_count(self) -> int { return (self.levels ?? []).len() }

    #: The string a dictionary code stands for.
    pub fn level_at(self, code: int) -> str {
        let table = self.levels
        if table is null { throw TypeError(f"{self.dtype.name()} has no dictionary") }
        if code < 0 or code >= table.len() {
            throw IndexError(f"code {code} is outside the {table.len()} levels")
        }
        return table[code]
    }

    #: The code `text` already has, or `null` when the dictionary lacks it.
    #:
    #: This is what makes `column == "widget"` one integer compare: the scalar
    #: is resolved to a code once, on the host, and a missing one short-circuits
    #: the whole comparison to false without a dispatch.
    pub fn level_code(self, text: str) -> int? {
        let table = self.intern
        if table is null { throw TypeError(f"{self.dtype.name()} has no dictionary") }
        let code = table.get(text, -1)
        return if code < 0 { null } else { code }
    }

    #: The code for `text`, appending a level when the dictionary lacks it.
    pub fn intern_level(self, text: str) -> int {
        var table = self.intern
        var names = self.levels
        if table is null or names is null {
            throw TypeError(f"{self.dtype.name()} has no dictionary")
        }
        let existing = table.get(text, -1)
        if existing >= 0 { return existing }
        let code = names.len()
        names.push(text)
        table[text] = code
        return code
    }

    #: Replace the dictionary wholesale, for a reader that decoded the codes
    #: and the levels separately.
    pub fn set_levels(self, levels: list[str]) -> void {
        if not self.dtype.is_dictionary() {
            throw TypeError(f"{self.dtype.name()} has no dictionary")
        }
        var table: dict[str, int] = {}
        for (code, name) in levels.enumerate() { table[name] = code }
        self.levels = levels
        self.intern = table
    }

    #: Whether the column carries a validity lane at all.
    pub fn has_validity(self) -> bool { return self.valid is not null }

    #: The validity lane, creating a lane of ones when there was none.
    #:
    #: Call it when a kernel is about to write nulls into a column that has not
    #: had any. Reading validity does not need it — a `null` lane already means
    #: every row is present.
    pub fn materialize_validity(self) -> Buffer {
        self.check_live()
        let existing = self.valid
        if existing is not null { return existing }
        let mask = Buffer(self.length)
        if self.length > 0 { mask.fill(1.0) }
        self.valid = mask
        return mask
    }

    #: The same values under a different validity lane, which the result adopts.
    #:
    #: The value lanes are copied rather than shared. One column owning one set
    #: of buffers is the rule that makes `free` decidable, and a mask update is
    #: rare enough next to the arithmetic that produced it to pay for a copy.
    pub fn with_validity(self, valid: Buffer?) -> Column {
        let out = self.copy_lanes(with_validity: false)
        out.valid = valid
        return out
    }

    #: An independent column with the same contents.
    pub fn copy(self) -> Column { return self.copy_lanes(with_validity: true) }

    #: Release the lanes. Freeing twice is harmless.
    pub fn free(self) -> void {
        if self.freed { return }
        self.freed = true
        self.values.free()
        let high = self.wide
        if high is not null { high.free() }
        let mask = self.valid
        if mask is not null { mask.free() }
    }

    #: Whether `free` has already run.
    pub fn is_freed(self) -> bool { return self.freed }

    #: Every row as a host float, missing rows as NaN.
    #:
    #: One boxed float per row: this is the boundary where a column stops being
    #: a device allocation, and it costs what materialising it costs.
    pub fn to_floats(self) -> list[float] {
        self.check_live()
        if self.length == 0 { return [] }
        var out = if self.dtype.is_wide() {
            let lane = self.as_float_lane()
            let downloaded = lane.download()
            lane.free()
            downloaded
        } else {
            self.values.download()
        }
        let mask = self.valid
        if mask is null { return out }
        for (row, present) in mask.download().enumerate() {
            if present == 0.0 { out[row] = NAN }
        }
        return out
    }

    #: Every row as a host integer. A missing row reads as 0.
    pub fn to_ints(self) -> list[int] {
        self.check_live()
        if self.length == 0 { return [] }
        if not self.dtype.is_wide() {
            return self.to_floats().map(|v| if is_nan(v) { 0 } else { int(v) })
        }
        let low = self.values.download()
        let high = (self.wide ?? self.values).download()
        var out: list[int] = []
        for (row, slot) in low.enumerate() { out.push(wide_unpack(slot, high[row])) }
        return out
    }

    #: Every row as a host boolean. A missing row reads as false.
    pub fn to_bools(self) -> list[bool] {
        return self.to_floats().map(|v| not is_nan(v) and v != 0.0)
    }

    #: Every row as host text, missing rows as `null`.
    pub fn to_strs(self) -> list[str?] {
        self.check_live()
        let table = self.levels
        if table is null { throw TypeError(f"{self.dtype.name()} has no dictionary") }
        if self.length == 0 { return [] }
        let codes = self.values.download()
        let mask = if self.valid is null { [] } else { (self.valid ?? self.values).download() }
        var out: list[str?] = []
        for (row, code) in codes.enumerate() {
            if mask.len() > 0 and mask[row] == 0.0 {
                out.push(null)
                continue
            }
            let at = int(code)
            out.push(if at >= 0 and at < table.len() { table[at] } else { null })
        }
        return out
    }

    #: True where a row is missing.
    pub fn null_mask(self) -> list[bool] {
        self.check_live()
        let mask = self.valid
        if mask is null { return [false for _row in 0..self.length] }
        return mask.download().map(|v| v == 0.0)
    }

    #: How many rows are missing.
    #:
    #: Reduced on the device into a few hundred partials and finished on the
    #: host, so a column of any size costs one dispatch and one short download
    #: instead of a boxed bool per row.
    pub fn null_count(self) -> int {
        self.check_live()
        let mask = self.valid
        if mask is null or self.length == 0 { return 0 }
        let partials = Buffer(COUNT_GROUPS)
        cached_kernel("col_count_nulls").dispatch_async(
            [mask, partials],
            COUNT_GROUPS * COUNT_GROUP,
            [self.length],
            group_size: COUNT_GROUP
        )
        var total = 0.0
        for part in partials.download() { total += part }
        partials.free()
        return int(total)
    }

    #: One cell.
    #:
    #: A device round trip per call. Reading a whole column goes through
    #: `to_floats` and friends, which pay one round trip for all of it.
    pub fn get(self, position: int) -> Value {
        self.check_live()
        let at = self.resolve(position)
        let mask = self.valid
        if mask is not null and mask.download(at, 1)[0] == 0.0 {
            return Value.null_of(self.dtype)
        }
        let slot = self.values.download(at, 1)[0]
        if self.dtype.is_wide() {
            let high = (self.wide ?? self.values).download(at, 1)[0]
            return Value(self.dtype, integer: wide_unpack(slot, high))
        }
        if self.dtype.is_dictionary() { return Value.of_str(self.level_at(int(slot)), self.dtype) }
        if self.dtype == Dtype.Bool { return Value.of_bool(slot != 0.0) }
        return Value.of_float(slot)
    }

    #: Write one cell, interning the text when the column is a dictionary one.
    pub fn set(self, position: int, value: Value) -> void {
        self.check_live()
        let at = self.resolve(position)
        if value.is_na() {
            self.materialize_validity().upload([0.0], at)
            return
        }
        let mask = self.valid
        if mask is not null { mask.upload([1.0], at) }
        if self.dtype.is_dictionary() {
            self.values.upload([float(self.intern_level(value.to_str()))], at)
            return
        }
        if self.dtype.is_wide() {
            let split = wide_pack(value.to_int())
            self.values.upload([split[0]], at)
            (self.wide ?? self.values).upload([split[1]], at)
            return
        }
        self.values.upload([value.to_float()], at)
    }

    #: The rows `positions` names, in that order.
    #:
    #: This is the one reordering primitive: sort, filter by index, join, and
    #: every reindex are a positions lane plus this call. `allow_missing` turns
    #: a `MISSING_POSITION` entry into a null instead of an error, which is what
    #: an outer join needs.
    pub fn take(self, positions: Buffer, length: int, allow_missing: bool = false) -> Column {
        self.check_live()
        let out = Column.allocate(
            self.dtype,
            length,
            with_validity: allow_missing or self.valid is not null
        )
        out.adopt_dictionary(self)
        if length == 0 { return out }
        cached_kernel("col_gather").dispatch_async(
            [
                self.values,
                self.wide ?? self.values,
                self.valid ?? self.values,
                positions,
                out.values,
                out.wide ?? out.values,
                out.valid ?? out.values,
            ],
            length,
            [
                length,
                self.length,
                flag(self.dtype.is_wide()),
                flag(self.valid is not null),
                flag(out.valid is not null),
                flag(allow_missing),
            ]
        )
        return out
    }

    #: The rows where `mask` is true, in order.
    #:
    #: A null in the mask drops its row, which is what `dropna` and a filter
    #: over a column that has nulls both want.
    pub fn compress(self, mask: Column) -> Column {
        self.check_live()
        if mask.dtype != Dtype.Bool {
            throw TypeError(f"a filter mask must be bool, not {mask.dtype.name()}")
        }
        if mask.length != self.length {
            throw ValueError(f"a mask of {mask.length} cannot filter {self.length} rows")
        }
        if self.length == 0 { return self.copy() }
        let flags = Buffer(self.length)
        cached_kernel("col_mask_flags").dispatch_async(
            [mask.values, mask.valid ?? mask.values, flags],
            self.length,
            [self.length, flag(mask.valid is not null)]
        )
        let scanned = prefix_sum(flags, self.length)
        let offsets = scanned[0]
        let kept = scanned[1]
        let out = Column.allocate(self.dtype, kept, with_validity: self.valid is not null)
        out.adopt_dictionary(self)
        if kept > 0 {
            cached_kernel("col_scatter").dispatch_async(
                [
                    self.values,
                    self.wide ?? self.values,
                    self.valid ?? self.values,
                    flags,
                    offsets,
                    out.values,
                    out.wide ?? out.values,
                    out.valid ?? out.values,
                ],
                self.length,
                [
                    self.length,
                    flag(self.dtype.is_wide()),
                    flag(self.valid is not null),
                    flag(out.valid is not null),
                ]
            )
        }
        flags.free()
        offsets.free()
        return out
    }

    #: The same rows read as `target`.
    #:
    #: `Str` and `Category` differ only in intent, so that pair is a relabel.
    #: `Int64` and `Datetime` share the two-lane layout, so that pair is a lane
    #: copy. Parsing text into numbers is the reader's job and formatting
    #: numbers into text is the display module's, so neither crosses here.
    pub fn cast(self, target: Dtype) -> Column {
        self.check_live()
        if target == self.dtype { return self.copy() }
        if self.dtype.is_dictionary() or target.is_dictionary() {
            if self.dtype.is_dictionary() and target.is_dictionary() {
                let relabelled = self.copy_lanes(with_validity: true, dtype: target)
                return relabelled
            }
            throw TypeError(
                f"casting between {self.dtype.name()} and {target.name()} needs a parser or a formatter"
            )
        }
        let out = Column.allocate(target, self.length, with_validity: self.valid is not null)
        let mask = self.valid
        if mask is not null and self.length > 0 {
            cached_kernel("col_copy_lane").dispatch_async(
                [mask, out.valid ?? out.values],
                self.length,
                [self.length, 0, 0]
            )
        }
        if self.length > 0 {
            cached_kernel("col_cast").dispatch_async(
                [self.values, self.wide ?? self.values, out.values, out.wide ?? out.values],
                self.length,
                [
                    self.length,
                    flag(self.dtype.is_wide()),
                    flag(target.is_wide()),
                    flag(target == Dtype.Bool),
                ]
            )
        }
        return out
    }

    #: One float32 lane holding every row's numeric value, freshly allocated.
    #:
    #: Wide columns are flattened here, and that is lossy past 2^24 — which is
    #: the honest precision of a mean or a standard deviation over int64
    #: anyway. An exact reduction reads both lanes instead. Dictionary columns
    #: give their codes.
    pub fn as_float_lane(self) -> Buffer {
        self.check_live()
        let lane = Buffer(self.length)
        if self.length == 0 { return lane }
        if self.dtype.is_wide() {
            cached_kernel("col_wide_to_float").dispatch_async(
                [self.values, self.wide ?? self.values, lane],
                self.length,
                [self.length]
            )
            return lane
        }
        cached_kernel("col_copy_lane").dispatch_async(
            [self.values, lane],
            self.length,
            [self.length, 0, 0]
        )
        return lane
    }

    #: The float lane and whether the caller owns it.
    #:
    #: A non-wide column already is its float lane, so a reduction over one
    #: should read `values` rather than copy it. This is the call that lets a
    #: hot path skip the copy without knowing which case it is in: free the
    #: buffer only when the flag says it is yours.
    pub fn borrow_float_lane(self) -> (Buffer, bool) {
        if self.dtype.is_wide() { return (self.as_float_lane(), true) }
        self.check_live()
        return (self.values, false)
    }

    #: Mix this column's per-row hash into `out`, one uint per row.
    #:
    #: `out` is device-side uint bits and must start zeroed; each call folds a
    #: further key column in, order-sensitively, so a multi-key group-by is one
    #: call per key over the same lane. Never download `out` as floats.
    #:
    #: Dictionary columns hash their codes, so two columns only hash alike
    #: after `unify_levels` has put them on one table.
    pub fn hash_into(self, out: Buffer, seed: int = 0) -> void {
        self.check_live()
        if self.length == 0 { return }
        cached_kernel("col_hash").dispatch_async(
            [self.values, self.wide ?? self.values, self.valid ?? self.values, out],
            self.length,
            [self.length, flag(self.dtype.is_wide()), flag(self.valid is not null), u32(seed)]
        )
    }

    #: Narrow `out` to the pairs whose rows are equal.
    #:
    #: `left[i]` indexes this column and `right[i]` indexes `other`. `out` must
    #: start at 1.0 and is only ever cleared, so chaining one call per key
    #: column leaves the pairs that agree on all of them — which is how a hash
    #: join confirms a bucket collision is a real match.
    #:
    #: Nulls compare equal to each other by default, matching what pandas'
    #: `merge` does with NaN keys; `dropna` group-by passes `false`.
    pub fn equal_rows(
        self,
        left: Buffer,
        other: Column,
        right: Buffer,
        length: int,
        out: Buffer,
        null_equal: bool = true
    ) -> void {
        self.check_live()
        other.check_live()
        if self.dtype != other.dtype {
            throw TypeError(f"cannot compare {self.dtype.name()} rows against {other.dtype.name()}")
        }
        if length == 0 { return }
        cached_kernel("col_equal_rows").dispatch_async(
            [
                self.values,
                self.wide ?? self.values,
                self.valid ?? self.values,
                other.values,
                other.wide ?? other.values,
                other.valid ?? other.values,
                left,
                right,
                out,
            ],
            length,
            [
                length,
                flag(self.dtype.is_wide()),
                flag(self.valid is not null),
                flag(other.valid is not null),
                flag(null_equal),
            ]
        )
    }

    fn check_live(self) -> void {
        if self.freed { throw RuntimeError("this Column has already been released") }
    }

    fn resolve(self, position: int) -> int {
        let at = if position < 0 { self.length + position } else { position }
        if at < 0 or at >= self.length {
            throw IndexError(f"row {position} is outside a column of {self.length}")
        }
        return at
    }

    fn upload_validity(self, present: list[bool]?) -> void {
        if present is null { return }
        let mask = self.valid
        if mask is null or self.length == 0 { return }
        mask.upload(present.map(|p| if p { 1.0 } else { 0.0 }))
    }

    fn adopt_dictionary(self, source: Column) -> void {
        if not self.dtype.is_dictionary() { return }
        self.levels = [name for name in source.levels ?? []]
        var table: dict[str, int] = {}
        for (code, name) in (self.levels ?? []).enumerate() { table[name] = code }
        self.intern = table
    }

    fn copy_lanes(self, with_validity: bool, dtype: Dtype? = null) -> Column {
        self.check_live()
        let out = Column.allocate(
            dtype ?? self.dtype,
            self.length,
            with_validity: with_validity and self.valid is not null
        )
        out.adopt_dictionary(self)
        if self.length == 0 { return out }
        copy_lane(self.values, out.values, self.length)
        let high = self.wide
        let out_high = out.wide
        if high is not null and out_high is not null { copy_lane(high, out_high, self.length) }
        let mask = self.valid
        let out_mask = out.valid
        if mask is not null and out_mask is not null { copy_lane(mask, out_mask, self.length) }
        return out
    }
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

#: Every part's dictionary rebuilt onto one shared table.
#:
#: Codes are only meaningful against the table they were assigned from, so two
#: text columns cannot be compared, hashed, joined or concatenated until they
#: agree on one. Levels come out in first-appearance order, scanning the parts
#: in order, which leaves the first part's codes unchanged in the common case.
#:
#: Returns fresh columns the caller owns; the inputs are untouched. The table
#: is a host `list[str]`, so this costs a host loop over the distinct values —
#: not over the rows, which a kernel remaps.
pub fn unify_levels(parts: list[Column]) -> list[Column] {
    if parts.len() == 0 { return [] }
    let dtype = parts[0].dtype
    if not dtype.is_dictionary() {
        throw TypeError(f"{dtype.name()} has no dictionary to unify")
    }
    var levels: list[str] = []
    var intern: dict[str, int] = {}
    for part in parts {
        if part.dtype != dtype {
            throw TypeError(f"cannot unify {dtype.name()} with {part.dtype.name()}")
        }
        for name in part.levels ?? [] {
            if intern.get(name, -1) >= 0 { continue }
            intern[name] = levels.len()
            levels.push(name)
        }
    }
    var out: list[Column] = []
    for part in parts { out.push(remap_levels(part, levels, intern)) }
    return out
}

#: The parts stacked end to end into one new column.
#:
#: Every part must already carry the same dtype. Promoting an `Int64` beside a
#: `Float64` is a decision with a precision cost, so it belongs to the caller
#: and to `cast`, not to a concatenation that would make it quietly. Dictionary
#: parts are unified first, so the result's codes are on one table.
pub fn concat_columns(parts: list[Column]) -> Column {
    if parts.len() == 0 {
        throw ValueError("a concatenation needs at least one column to take a dtype from")
    }
    let dtype = parts[0].dtype
    var total = 0
    var any_missing = false
    for part in parts {
        if part.dtype != dtype {
            throw TypeError(f"cannot concatenate {part.dtype.name()} onto {dtype.name()}")
        }
        total += part.length
        if part.has_validity() { any_missing = true }
    }
    if total > MAX_ROWS {
        throw ValueError(f"{total} rows exceeds the {MAX_ROWS} a float32 position lane addresses")
    }
    if not dtype.is_dictionary() { return stack_lanes(parts, dtype, total, any_missing) }
    let unified = unify_levels(parts)
    let out = stack_lanes(unified, dtype, total, any_missing)
    for part in unified { part.free() }
    return out
}

#: The row labels of a `Series` or a `DataFrame` — all three pandas index kinds
#: in one class.
#:
#: An empty `levels` is a range index, whose labels are `start + i * step` and
#: cost no storage. One level is a plain index. Several levels are a
#: multi-index, where a row's label is the tuple across them.
#:
#: pandas spells those RangeIndex, Index and MultiIndex, and then every
#: operation that touches an index has to ask which one it was handed. One
#: class with a list of levels makes `stack`, `unstack` and a multi-key
#: group-by fall out of the same code that handles a single key, and it is what
#: lets `DataFrame` name its index field before the reshaping module exists.
pub class Index {
    #: One column per level, empty for a range index.
    pub var levels: list[Column]
    #: A name per level, `null` where a level is unnamed. `nlevels()` long.
    pub var names: list[str?]
    #: How many rows are labelled.
    pub var length: int
    #: First label of a range index; unread once `levels` is occupied.
    pub var start: int
    #: Distance between the labels of a range index.
    pub var step: int

    #: The empty range index. `range` and `of` are the constructors.
    pub fn init(self) {
        self.levels = []
        self.names = [null]
        self.length = 0
        self.start = 0
        self.step = 1
    }

    #: A range index — the default labels of a frame nobody has indexed.
    pub static fn range(length: int, start: int = 0, step: int = 1, name: str? = null) -> Index {
        if length < 0 { throw ValueError(f"an index cannot label {length} rows") }
        if step == 0 { throw ValueError("a range index of step 0 repeats one label") }
        let last = start + (if length > 0 { length - 1 } else { 0 }) * step
        if not fits_wide(start) or not fits_wide(last) {
            throw ValueError(
                f"a range index from {start} to {last} leaves the 48 bits an Int64 column stores"
            )
        }
        let index = Index()
        index.levels = []
        index.names = [name]
        index.length = length
        index.start = start
        index.step = step
        return index
    }

    #: An index over stored labels: one level for a plain index, several for a
    #: multi-index. The levels are adopted, not copied.
    pub static fn of(levels: list[Column], names: list[str?]? = null) -> Index {
        if levels.len() == 0 {
            throw ValueError("an index with no levels is a range index; build it with Index.range")
        }
        let length = levels[0].len()
        for level in levels {
            if level.len() != length {
                throw ValueError(f"index levels disagree on length: {level.len()} against {length}")
            }
        }
        var labels: list[str?] = []
        if names is null {
            for _level in 0..levels.len() { labels.push(null) }
        } else {
            for name in names ?? [] { labels.push(name) }
        }
        if labels.len() != levels.len() {
            throw ValueError(f"{labels.len()} names cannot name {levels.len()} levels")
        }
        let index = Index()
        index.levels = [level for level in levels]
        index.names = labels
        index.length = length
        index.start = 0
        index.step = 1
        return index
    }

    #: Whether the labels are implied rather than stored.
    pub fn is_range(self) -> bool { return self.levels.len() == 0 }

    #: How many levels a label spans. A range index has one, as pandas' does.
    pub fn nlevels(self) -> int {
        return if self.is_range() { 1 } else { self.levels.len() }
    }

    #: How many rows are labelled.
    pub fn len(self) -> int { return self.length }

    #: The levels as stored columns, freshly allocated and owned by the caller.
    #:
    #: A range index materialises into the `Int64` ramp it stands for. The
    #: point of not storing that ramp is that most frames never ask for it; the
    #: ones that do — a reshape, a group-by keyed on the index — ask here.
    pub fn materialize(self) -> list[Column] {
        if self.is_range() { return [range_column(self.length, self.start, self.step)] }
        return self.levels.map(|level| level.copy())
    }

    #: The rows `positions` names, as a new index.
    pub fn take(self, positions: Buffer, length: int, allow_missing: bool = false) -> Index {
        if self.is_range() {
            let ramp = range_column(self.length, self.start, self.step)
            let taken = ramp.take(positions, length, allow_missing)
            ramp.free()
            return Index.of([taken], [name for name in self.names])
        }
        return Index.of(
            self.levels.map(|level| level.take(positions, length, allow_missing)),
            [name for name in self.names]
        )
    }

    #: The label of one row, one `Value` per level.
    #:
    #: A list even for a single level, so code written for a multi-index reads
    #: a plain one without a case for it.
    pub fn label_at(self, position: int) -> list[Value] {
        let at = if position < 0 { self.length + position } else { position }
        if at < 0 or at >= self.length {
            throw IndexError(f"row {position} is outside an index of {self.length}")
        }
        if self.is_range() { return [Value.of_int(self.start + at * self.step)] }
        return self.levels.map(|level| level.get(at))
    }

    #: An independent index with the same labels.
    pub fn copy(self) -> Index {
        if self.is_range() {
            return Index.range(self.length, start: self.start, step: self.step, name: self.names[0])
        }
        return Index.of(self.levels.map(|level| level.copy()), [name for name in self.names])
    }

    #: Release the level columns. A range index holds none to release.
    pub fn free(self) -> void {
        for level in self.levels { level.free() }
    }
}

#: One column with its labels: values, an index, and a name.
#:
#: Structure only. Arithmetic, alignment, reductions and display are other
#: modules; what lives here is what a `DataFrame` field has to be able to name.
pub class Series {
    #: The values.
    pub var values: Column
    #: One label per row.
    pub var index: Index
    #: What the series is called, which becomes a column name in a frame.
    pub var name: str?

    #: A series over `values`. A null index gets the default range one.
    pub fn init(self, values: Column, index: Index? = null, name: str? = null) {
        let labels = index ?? Index.range(values.len())
        if labels.len() != values.len() {
            throw ValueError(f"an index of {labels.len()} cannot label {values.len()} rows")
        }
        self.values = values
        self.index = labels
        self.name = name
    }

    #: Number of rows.
    pub fn len(self) -> int { return self.values.len() }

    #: How the values are stored and what operations mean over them.
    pub fn dtype(self) -> Dtype { return self.values.dtype }

    #: An independent series with the same contents.
    pub fn copy(self) -> Series {
        return Series(self.values.copy(), index: self.index.copy(), name: self.name)
    }

    #: Release the values and the index.
    pub fn free(self) -> void {
        self.values.free()
        self.index.free()
    }
}

#: A table: named columns of equal length over one shared index.
#:
#: Only the structure is here — what a frame is made of, and how to reach a
#: column by name. Selecting, arithmetic, missing data, reductions, group-by,
#: joins, reshaping, timeseries and display are separate modules, and every one
#: of them builds its result through `_replace`.
pub class DataFrame {
    #: The columns, in display order.
    pub var columns: list[Column]
    #: A name per column, in the same order.
    pub var names: list[str]
    #: One label per row, shared by every column.
    pub var index: Index
    var lookup: dict[str, int]

    #: A frame over `columns`, named, with `index` labelling the rows.
    #:
    #: Nothing is copied: the frame adopts the columns and the index it is
    #: given, and only the two list containers are fresh.
    #:
    #: Duplicate names are refused. pandas allows them, and then `df["a"]` is a
    #: `Series` sometimes and a frame other times. Every module here addresses
    #: a column by name, so a duplicate would make `column(name)` wrong rather
    #: than expressive.
    pub fn init(self, names: list[str], columns: list[Column], index: Index? = null) {
        if names.len() != columns.len() {
            throw ValueError(f"{names.len()} names cannot name {columns.len()} columns")
        }
        var lookup: dict[str, int] = {}
        for (at, name) in names.enumerate() {
            if lookup.get(name, -1) >= 0 { throw ValueError(f"duplicate column name {name}") }
            lookup[name] = at
        }
        let labels = index ?? Index.range(if columns.len() > 0 { columns[0].len() } else { 0 })
        for column in columns {
            if column.len() != labels.len() {
                throw ValueError(
                    f"a column of {column.len()} rows does not fit an index of {labels.len()}"
                )
            }
        }
        self.names = [name for name in names]
        self.columns = [column for column in columns]
        self.index = labels
        self.lookup = lookup
    }

    #: Rows and columns.
    pub fn shape(self) -> (int, int) { return (self.index.len(), self.columns.len()) }

    #: Number of rows, which is the index's length even with no columns.
    pub fn len(self) -> int { return self.index.len() }

    #: Number of columns.
    pub fn width(self) -> int { return self.columns.len() }

    #: Whether a column of that name exists.
    pub fn has(self, name: str) -> bool { return self.lookup.get(name, -1) >= 0 }

    #: Where a named column sits, raising `KeyError` when there is none.
    #:
    #: Resolved through a table built at construction, so this is a lookup and
    #: not a scan. That table is why renaming a column means building a new
    #: frame through `_replace` rather than writing into `names`.
    pub fn column_index(self, name: str) -> int {
        let at = self.lookup.get(name, -1)
        if at < 0 { throw KeyError(f"no column named {name}") }
        return at
    }

    #: A named column, raising `KeyError` when there is none.
    pub fn column(self, name: str) -> Column { return self.columns[self.column_index(name)] }

    #: How each column is stored, in column order.
    pub fn dtypes(self) -> list[Dtype] { return self.columns.map(|column| column.dtype) }

    #: An independent frame with the same contents.
    pub fn copy(self) -> DataFrame {
        return DataFrame(
            [name for name in self.names],
            self.columns.map(|column| column.copy()),
            index: self.index.copy()
        )
    }

    #: Release every column and the index.
    pub fn free(self) -> void {
        for column in self.columns { column.free() }
        self.index.free()
    }

    #: The same frame with some of its parts swapped out.
    #:
    #: This is how every other module returns a result: build the columns, hand
    #: them over, keep whatever did not change. Nothing is copied, so the result
    #: shares whichever parts came from this frame. Freeing either frame frees
    #: those shared buffers for both — `free` is idempotent, so the danger is a
    #: read after one of them was released, not a double release.
    pub fn _replace(
        self,
        names: list[str]? = null,
        columns: list[Column]? = null,
        index: Index? = null
    ) -> DataFrame {
        return DataFrame(names ?? self.names, columns ?? self.columns, index: index ?? self.index)
    }
}

#: Whether an integer survives the two-lane encoding.
fn fits_wide(value: int) -> bool { return value > -WIDE_LIMIT and value < WIDE_LIMIT }

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

fn remap_levels(part: Column, levels: list[str], intern: dict[str, int]) -> Column {
    let mapped = Column.allocate(part.dtype, part.length, with_validity: part.has_validity())
    mapped.set_levels([name for name in levels])
    let mask = part.valid
    let out_mask = mapped.valid
    if mask is not null and out_mask is not null { copy_lane(mask, out_mask, part.length) }
    if part.length == 0 { return mapped }
    let table = part.levels ?? []
    if table.len() == 0 {
        copy_lane(part.values, mapped.values, part.length)
        return mapped
    }
    var mapping: list[float] = []
    for name in table { mapping.push(float(intern.get(name, 0))) }
    let lane = Buffer(mapping.len())
    lane.upload(mapping)
    cached_kernel("col_remap").dispatch_async(
        [part.values, mapped.values, lane],
        part.length,
        [part.length, table.len()]
    )
    lane.free()
    return mapped
}

fn stack_lanes(parts: list[Column], dtype: Dtype, total: int, any_missing: bool) -> Column {
    let out = Column.allocate(dtype, total, with_validity: any_missing)
    let out_mask = out.valid
    #: Ones first, then the parts that carry a mask write over their stretch. A
    #: part with no mask has no missing row, so its stretch is already right.
    if out_mask is not null and total > 0 { out_mask.fill(1.0) }
    if dtype.is_dictionary() and parts.len() > 0 {
        out.set_levels([name for name in parts[0].levels ?? []])
    }
    var at = 0
    for part in parts {
        if part.length == 0 { continue }
        copy_lane(part.values, out.values, part.length, 0, at)
        let high = part.wide
        let out_high = out.wide
        if high is not null and out_high is not null {
            copy_lane(high, out_high, part.length, 0, at)
        }
        let mask = part.valid
        if mask is not null and out_mask is not null {
            copy_lane(mask, out_mask, part.length, 0, at)
        }
        at += part.length
    }
    return out
}

#: The `Int64` ramp a range index stands for.
fn range_column(length: int, start: int, step: int) -> Column {
    let column = Column.allocate(Dtype.Int64, length)
    if length == 0 { return column }
    cached_kernel("col_ramp").dispatch_async(
        [column.values, column.wide ?? column.values],
        length,
        [length, u32(start), u32(step)]
    )
    return column
}

fn validity_of(length: int, nulls: list[bool]?, derived: list[bool]?) -> list[bool]? {
    if nulls is not null {
        if nulls.len() != length {
            throw ValueError(f"{nulls.len()} null flags cannot cover {length} rows")
        }
        return nulls.map(|missing| not missing)
    }
    if derived is null { return null }
    var any = false
    for missing in derived {
        if missing {
            any = true
            break
        }
    }
    if not any { return null }
    return derived.map(|missing| not missing)
}

fn copy_lane(
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
