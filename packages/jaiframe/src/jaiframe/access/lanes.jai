#: The shader library, the lanes a dispatch binds, and the scalars it takes.
from std.gpu import Buffer, Kernel

from ..column import Column, Value
from ..column import f32_bits, wide_pack

#: Threadgroup width of the tiled `isin` scan, and how many wanted values one
#: pass of it stages.
pub const WANTED_TILE = 256

let SOURCE = """#include <metal_stdlib>
using namespace metal;

constexpr constant uint WANTED_TILE = 256;
constexpr constant long WIDE_SCALE = 16777216L;

/* Recombined in `long`, never in `float`: the low 24 bits are the whole
 * reason the second lane exists. */
inline long wide_value(float low, float high) {
    return (long)high * WIDE_SCALE + (long)low;
}

kernel void acc_keep_flags(device const float* values [[buffer(0)]],
                           device const float* valid [[buffer(1)]],
                           device float* flags [[buffer(2)]],
                           constant uint& count [[buffer(3)]],
                           constant uint& has_valid [[buffer(4)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    bool keep = values[gid] != 0.0f;
    if (has_valid != 0u && valid[gid] == 0.0f) keep = false;
    flags[gid] = keep ? 1.0f : 0.0f;
}

/* The same, four rows at a time, for a mask whose length divides by four.
 * This kernel does nothing but move sixteen bytes in and four out, so the
 * quartered thread count is very nearly a quartered cost. */
kernel void acc_keep_flags4(device const float4* values [[buffer(0)]],
                            device const float4* valid [[buffer(1)]],
                            device float4* flags [[buffer(2)]],
                            constant uint& quads [[buffer(3)]],
                            constant uint& has_valid [[buffer(4)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= quads) return;
    float4 keep = select(float4(0.0f), float4(1.0f), values[gid] != 0.0f);
    if (has_valid != 0u) keep = select(float4(0.0f), keep, valid[gid] != 0.0f);
    flags[gid] = keep;
}

/* The scatter that turns a scanned flag lane into the positions it marks.
 * `offsets` is the exclusive scan, so a set row already knows where it lands
 * and no two threads collide. */
kernel void acc_take_positions(device const float* flags [[buffer(0)]],
                               device const float* offsets [[buffer(1)]],
                               device float* out [[buffer(2)]],
                               constant uint& count [[buffer(3)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (flags[gid] == 0.0f) return;
    out[(uint)offsets[gid]] = (float)gid;
}

kernel void acc_span_positions(device float* out [[buffer(0)]],
                               constant uint& count [[buffer(1)]],
                               constant int& first [[buffer(2)]],
                               constant int& step [[buffer(3)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out[gid] = (float)(first + (int)gid * step);
}

/* Four positions at a time, from one multiply: a strided slice is arithmetic
 * and nothing else, so the write width is the whole cost. */
kernel void acc_span_positions4(device float4* out [[buffer(0)]],
                                constant uint& quads [[buffer(1)]],
                                constant int& first [[buffer(2)]],
                                constant int& step [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= quads) return;
    int base = first + (int)(gid << 2u) * step;
    out[gid] = float4((float)base,
                      (float)(base + step),
                      (float)(base + 2 * step),
                      (float)(base + 3 * step));
}

/* Dispatched over the listed positions, not over the rows: dropping ten rows
 * from a million costs ten threads and leaves the rest of the lane alone. */
kernel void acc_mark_positions(device float* flags [[buffer(0)]],
                               device const float* positions [[buffer(1)]],
                               constant uint& count [[buffer(2)]],
                               constant uint& length [[buffer(3)]],
                               constant uint& mark [[buffer(4)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    int at = (int)positions[gid];
    if (at < 0 || at >= (int)length) return;
    flags[at] = (mark != 0u) ? 1.0f : 0.0f;
}

kernel void acc_floor_positions(device float* lane [[buffer(0)]],
                                constant uint& count [[buffer(1)]],
                                constant uint& length [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    float at = floor(lane[gid]);
    float last = (float)(length - 1u);
    if (at < 0.0f) at = 0.0f;
    if (at > last) at = last;
    lane[gid] = at;
}

/* The permutation that lifts the appended rows back to `at`. Three straight
 * ranges and no division at all, which is what makes an insert one gather
 * rather than three takes and two concatenations per column. */
kernel void acc_insert_order(device float* lane [[buffer(0)]],
                             constant uint& total [[buffer(1)]],
                             constant uint& held [[buffer(2)]],
                             constant uint& at [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= total) return;
    uint added = total - held;
    if (gid < at) {
        lane[gid] = (float)gid;
        return;
    }
    if (gid < at + added) {
        lane[gid] = (float)(held + (gid - at));
        return;
    }
    lane[gid] = (float)(gid - added);
}

kernel void acc_copy_span(device const float* src [[buffer(0)]],
                          device float* dst [[buffer(1)]],
                          constant uint& count [[buffer(2)]],
                          constant uint& dst_offset [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    dst[dst_offset + gid] = src[gid];
}

/* Membership for a dictionary column, and for anything else whose answer
 * depends only on the level: the host decides once per distinct value, which
 * there are few of, and the rows are a single gather from that verdict lane.
 * `between` over text goes through here too, because the codes are in
 * first-appearance order and comparing them would not be comparing strings. */
kernel void acc_codes_in_set(device const float* codes [[buffer(0)]],
                             device const float* valid [[buffer(1)]],
                             device const float* wanted [[buffer(2)]],
                             device float* out [[buffer(3)]],
                             constant uint& count [[buffer(4)]],
                             constant uint& levels [[buffer(5)]],
                             constant uint& has_valid [[buffer(6)]],
                             constant uint& want_missing [[buffer(7)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (has_valid != 0u && valid[gid] == 0.0f) {
        out[gid] = (want_missing != 0u) ? 1.0f : 0.0f;
        return;
    }
    int code = (int)codes[gid];
    out[gid] = (code >= 0 && code < (int)levels) ? wanted[code] : 0.0f;
}

/* Every thread in the group compares against the same wanted values, so they
 * are staged in threadgroup memory a tile at a time rather than re-read from
 * device memory once per row. The tile loop is uniform across the group —
 * `wanted_count` is a scalar — which is what makes the barriers legal; the
 * out-of-range threads are held with a flag instead of returning early for
 * the same reason. */
kernel void acc_isin_values(device const float* values [[buffer(0)]],
                            device const float* wide [[buffer(1)]],
                            device const float* valid [[buffer(2)]],
                            device const float* wanted_low [[buffer(3)]],
                            device const float* wanted_high [[buffer(4)]],
                            device float* out [[buffer(5)]],
                            constant uint& count [[buffer(6)]],
                            constant uint& wanted_count [[buffer(7)]],
                            constant uint& has_wide [[buffer(8)]],
                            constant uint& has_valid [[buffer(9)]],
                            constant uint& want_missing [[buffer(10)]],
                            uint gid [[thread_position_in_grid]],
                            uint lid [[thread_index_in_threadgroup]]) {
    threadgroup float tile_low[WANTED_TILE];
    threadgroup float tile_high[WANTED_TILE];
    bool live = gid < count;
    bool absent = live && has_valid != 0u && valid[gid] == 0.0f;
    bool present = live && !absent;
    float own_low = present ? values[gid] : 0.0f;
    float own_high = (present && has_wide != 0u) ? wide[gid] : 0.0f;
    bool hit = false;
    for (uint base = 0u; base < wanted_count; base += WANTED_TILE) {
        uint span = min(WANTED_TILE, wanted_count - base);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < span) {
            tile_low[lid] = wanted_low[base + lid];
            tile_high[lid] = wanted_high[base + lid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (present && !hit) {
            for (uint i = 0u; i < span; ++i) {
                bool same = tile_low[i] == own_low;
                if (same && has_wide != 0u) same = tile_high[i] == own_high;
                if (same) {
                    hit = true;
                    break;
                }
            }
        }
    }
    if (!live) return;
    if (absent) {
        out[gid] = (want_missing != 0u) ? 1.0f : 0.0f;
        return;
    }
    out[gid] = hit ? 1.0f : 0.0f;
}

kernel void acc_between(device const float* values [[buffer(0)]],
                        device const float* valid [[buffer(1)]],
                        device float* out [[buffer(2)]],
                        constant uint& count [[buffer(3)]],
                        constant uint& has_valid [[buffer(4)]],
                        constant uint& low_bits [[buffer(5)]],
                        constant uint& high_bits [[buffer(6)]],
                        constant uint& closed [[buffer(7)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (has_valid != 0u && valid[gid] == 0.0f) {
        out[gid] = 0.0f;
        return;
    }
    float v = values[gid];
    bool above = ((closed & 1u) != 0u) ? (v >= as_type<float>(low_bits))
                                       : (v > as_type<float>(low_bits));
    bool below = ((closed & 2u) != 0u) ? (v <= as_type<float>(high_bits))
                                       : (v < as_type<float>(high_bits));
    out[gid] = (above && below) ? 1.0f : 0.0f;
}

/* The bounds arrive as their two halves rather than as floats, so a timestamp
 * range compares at full 48-bit width instead of at the 24 a float32 bound
 * would have rounded them to. */
kernel void acc_between_wide(device const float* values [[buffer(0)]],
                             device const float* wide [[buffer(1)]],
                             device const float* valid [[buffer(2)]],
                             device float* out [[buffer(3)]],
                             constant uint& count [[buffer(4)]],
                             constant uint& has_valid [[buffer(5)]],
                             constant int& low_low [[buffer(6)]],
                             constant int& low_high [[buffer(7)]],
                             constant int& high_low [[buffer(8)]],
                             constant int& high_high [[buffer(9)]],
                             constant uint& closed [[buffer(10)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (has_valid != 0u && valid[gid] == 0.0f) {
        out[gid] = 0.0f;
        return;
    }
    long v = wide_value(values[gid], wide[gid]);
    long low = (long)low_high * WIDE_SCALE + (long)low_low;
    long high = (long)high_high * WIDE_SCALE + (long)high_low;
    bool above = ((closed & 1u) != 0u) ? (v >= low) : (v > low);
    bool below = ((closed & 2u) != 0u) ? (v <= high) : (v < high);
    out[gid] = (above && below) ? 1.0f : 0.0f;
}

kernel void acc_where(device const float* values [[buffer(0)]],
                      device const float* wide [[buffer(1)]],
                      device const float* valid [[buffer(2)]],
                      device const float* cond [[buffer(3)]],
                      device const float* cond_valid [[buffer(4)]],
                      device float* out_values [[buffer(5)]],
                      device float* out_wide [[buffer(6)]],
                      device float* out_valid [[buffer(7)]],
                      constant uint& count [[buffer(8)]],
                      constant uint& has_wide [[buffer(9)]],
                      constant uint& has_valid [[buffer(10)]],
                      constant uint& cond_has_valid [[buffer(11)]],
                      constant uint& out_has_valid [[buffer(12)]],
                      constant uint& invert [[buffer(13)]],
                      constant uint& other_low [[buffer(14)]],
                      constant uint& other_high [[buffer(15)]],
                      constant uint& other_missing [[buffer(16)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    bool keep = cond[gid] != 0.0f;
    if (cond_has_valid != 0u && cond_valid[gid] == 0.0f) keep = false;
    if (invert != 0u) keep = !keep;
    if (keep) {
        out_values[gid] = values[gid];
        if (has_wide != 0u) out_wide[gid] = wide[gid];
        if (out_has_valid != 0u) out_valid[gid] = (has_valid != 0u) ? valid[gid] : 1.0f;
        return;
    }
    out_values[gid] = (other_missing != 0u) ? 0.0f : as_type<float>(other_low);
    if (has_wide != 0u) {
        out_wide[gid] = (other_missing != 0u) ? 0.0f : as_type<float>(other_high);
    }
    if (out_has_valid != 0u) out_valid[gid] = (other_missing != 0u) ? 0.0f : 1.0f;
}

/* Writes into the column's own lanes. Assignment is the one operation here
 * that is not allowed to hand back a new column: a frame's readers are
 * holding the buffers, not the frame. */
kernel void acc_assign(device float* values [[buffer(0)]],
                       device float* wide [[buffer(1)]],
                       device float* valid [[buffer(2)]],
                       device const float* flags [[buffer(3)]],
                       constant uint& count [[buffer(4)]],
                       constant uint& has_wide [[buffer(5)]],
                       constant uint& has_valid [[buffer(6)]],
                       constant uint& low_bits [[buffer(7)]],
                       constant uint& high_bits [[buffer(8)]],
                       constant uint& missing [[buffer(9)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (flags[gid] == 0.0f) return;
    if (missing != 0u) {
        if (has_valid != 0u) valid[gid] = 0.0f;
        return;
    }
    values[gid] = as_type<float>(low_bits);
    if (has_wide != 0u) wide[gid] = as_type<float>(high_bits);
    if (has_valid != 0u) valid[gid] = 1.0f;
}

/* Narrows a flag lane to the rows whose level equals one label, and only ever
 * clears — so chaining one call per level leaves the rows that agree on all
 * of them, which is how a multi-index label is matched without a join. */
kernel void acc_label_match(device const float* values [[buffer(0)]],
                            device const float* wide [[buffer(1)]],
                            device const float* valid [[buffer(2)]],
                            device float* flags [[buffer(3)]],
                            constant uint& count [[buffer(4)]],
                            constant uint& has_wide [[buffer(5)]],
                            constant uint& has_valid [[buffer(6)]],
                            constant uint& low_bits [[buffer(7)]],
                            constant uint& high_bits [[buffer(8)]],
                            constant uint& want_missing [[buffer(9)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (flags[gid] == 0.0f) return;
    bool present = !(has_valid != 0u && valid[gid] == 0.0f);
    if (want_missing != 0u) {
        flags[gid] = present ? 0.0f : 1.0f;
        return;
    }
    if (!present) {
        flags[gid] = 0.0f;
        return;
    }
    bool same = values[gid] == as_type<float>(low_bits);
    if (same && has_wide != 0u) same = wide[gid] == as_type<float>(high_bits);
    flags[gid] = same ? 1.0f : 0.0f;
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

pub fn u32(value: int) -> int { return value & 0xFFFFFFFF }

#: The three lane accessors every dispatch here binds through.
#:
#: A kernel has to be handed a buffer for every slot it declares even when the
#: lane does not exist, so an absent `wide` or `valid` binds the value lane and
#: a `has_wide` or `has_valid` scalar tells the kernel not to read it. Going
#: through these three rather than reaching for the fields keeps that
#: substitution in one place, where the dtype rule it depends on is written
#: down.
pub fn value_lane(column: Column) -> Buffer { return column.values }

pub fn wide_lane(column: Column) -> Buffer { return column.wide ?? column.values }

pub fn valid_lane(column: Column) -> Buffer { return column.valid ?? column.values }

#: The bits of `value` as this column would store it, interning text into
#: `column`'s dictionary when it is not there yet.
pub fn scalar_bits(column: Column, value: Value) -> (int, int) {
    if value.is_na() { return (0, 0) }
    if column.dtype.is_dictionary() {
        return (f32_bits(float(column.intern_level(value.to_str()))), 0)
    }
    if column.dtype.is_wide() {
        let split = wide_pack(value.to_int())
        return (f32_bits(split[0]), f32_bits(split[1]))
    }
    return (f32_bits(value.to_float()), 0)
}
