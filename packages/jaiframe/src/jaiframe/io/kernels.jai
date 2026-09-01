#: The device side of the readers and writers: the Metal source they share and
#: the lane downloads that fold a validity mask into its payload.
from std.gpu import Buffer, Kernel

from ..column import Column

pub const BYTES_GROUP = 256
pub const BYTES_GROUPS = 256
const LEVEL_TILE = 1024

let SOURCE = """#include <metal_stdlib>
using namespace metal;

constexpr constant uint BYTES_GROUP = 256;
constexpr constant uint BYTES_GROUPS = 256;
constexpr constant uint LEVEL_TILE = 1024;

/* A missing row, folded into the payload lane so a writer's host pass has one
 * question to ask per cell instead of two. */
kernel void io_blank_nulls(device const float* src [[buffer(0)]],
                           device const float* valid [[buffer(1)]],
                           device float* out [[buffer(2)]],
                           constant uint& count [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out[gid] = (valid[gid] == 0.0f) ? NAN : src[gid];
}

/* The same four rows at a time, for a column whose length divides by four.
 * The kernel does nothing but move eight bytes in and four out, so a quarter
 * of the threads is very nearly a quarter of the cost. */
kernel void io_blank_nulls4(device const float4* src [[buffer(0)]],
                            device const float4* valid [[buffer(1)]],
                            device float4* out [[buffer(2)]],
                            constant uint& quads [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= quads) return;
    out[gid] = select(float4(NAN), src[gid], valid[gid] != 0.0f);
}

/* How many bytes a dictionary column's strings occupy, counting a level once
 * per row that uses it.
 *
 * The rows are codes, so this is a gather out of a table with one entry per
 * level, and neighbouring threads hit the same handful of entries over and
 * over -- ten million rows over eight levels would read eight addresses ten
 * million times. The table is staged in threadgroup memory once per group and
 * read from there; past LEVEL_TILE levels the staging would cost more than it
 * saves and the gather goes straight to device memory instead.
 *
 * The fold runs in `long` because the answer does not fit a float32. Ten
 * million rows of ten bytes is 10^8 and float32 counts integers exactly only
 * to 2^24, so each group reports its total as the same two-lane split an
 * Int64 column stores. */
kernel void io_text_bytes(device const float* codes [[buffer(0)]],
                          device const float* valid [[buffer(1)]],
                          device const float* lengths [[buffer(2)]],
                          device float* low [[buffer(3)]],
                          device float* high [[buffer(4)]],
                          constant uint& count [[buffer(5)]],
                          constant uint& levels [[buffer(6)]],
                          constant uint& has_valid [[buffer(7)]],
                          uint gid [[thread_position_in_grid]],
                          uint lid [[thread_index_in_threadgroup]],
                          uint wid [[threadgroup_position_in_grid]]) {
    threadgroup float table[LEVEL_TILE];
    threadgroup long scratch[BYTES_GROUP];
    bool staged = levels <= LEVEL_TILE;
    if (staged) {
        for (uint at = lid; at < levels; at += BYTES_GROUP) table[at] = lengths[at];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    long total = 0;
    for (uint row = gid; row < count; row += BYTES_GROUP * BYTES_GROUPS) {
        if (has_valid != 0u && valid[row] == 0.0f) continue;
        uint code = (uint)codes[row];
        if (code >= levels) continue;
        total += (long)(staged ? table[code] : lengths[code]);
    }
    scratch[lid] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = BYTES_GROUP / 2u; stride > 0u; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) {
        long sum = scratch[0];
        long hi = sum >> 24;
        low[wid] = (float)(sum - (hi << 24));
        high[wid] = (float)hi;
    }
}
"""

var _kernels: dict[str, Kernel] = {}

pub fn compiled(entry: str) -> Kernel {
    let existing = _kernels.get(entry, null)
    if existing is not null { return existing }
    let kernel = Kernel.compile(SOURCE, entry)
    _kernels[entry] = kernel
    return kernel
}

pub fn flag(value: bool) -> int { return if value { 1 } else { 0 } }

#: Every row of a plain lane as a host float, missing rows already NaN.
#:
#: `Column.to_floats` answers the same question by downloading the validity
#: lane alongside the payload and patching the result row by row. Folding the
#: mask into the payload on the device instead is one dispatch, one download
#: and no second host pass, which on a wide frame is one avoided pass per
#: column rather than one in total.
pub fn blanked(lane: Buffer, valid: Buffer?, length: int) -> list[float] {
    if length == 0 { return [] }
    let mask = valid
    if mask is null { return lane.download() }
    let out = Buffer(length)
    if length % 4 == 0 {
        compiled("io_blank_nulls4").dispatch_async([lane, mask, out], length // 4, [length // 4])
    } else {
        compiled("io_blank_nulls").dispatch_async([lane, mask, out], length, [length])
    }
    let values = out.download()
    out.free()
    return values
}

#: The two lanes of a wide column, with the high half NaN where a row is
#: missing.
#:
#: The value has to come back exact, so the low lane is downloaded untouched
#: and recombined on the host. Only the high half carries the missing flag,
#: which costs the writer one `is_nan` where reading a separate mask would
#: cost it a third download and a third pass.
pub fn blanked_wide(column: Column) -> (list[float], list[float]) {
    if column.length == 0 { return ([], []) }
    let high = column.wide ?? column.values
    return (column.values.download(), blanked(high, column.valid, column.length))
}
