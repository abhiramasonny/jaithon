#: The shader source every jainum kernel is built on, and the pipeline cache it
#: is compiled into.
import std.gpu as gpu
from std.gpu import Kernel

#: The Metal every jainum kernel is compiled on top of: the index walk, the
#: conversion and arithmetic rules, and the universal passes.
#:
#: A module writes `let SOURCE = PRELUDE + """ ... """` and asks `kernel_of`
#: for its entry points.
pub let PRELUDE = """#include <metal_stdlib>
using namespace metal;

constexpr constant uint ND_GROUP = 256u;
constexpr constant uint ND_WARPS = ND_GROUP / 32u;

/* One float block per dispatch, as `layout_buffer` writes it:
 *
 *   [0]                              rank of the iteration shape
 *   [1 .. rank]                      the iteration shape
 *   [1 + rank + k*(rank+1)]          operand k's element offset
 *   [2 + rank + k*(rank+1) + axis]   operand k's stride along `axis`
 *
 * Floats and not `constant uint&` scalars for two reasons. A stride is signed
 * here -- a flip and a negative-step slice are views, not copies -- and
 * `Kernel.dispatch` binds its trailing scalars as unsigned. And there are
 * rank * operands of them, which is not a fixed argument list. Exact to 2^24,
 * which is also where an element count stops being exact, so nothing that
 * fits in an array fails to fit in its metadata.
 *
 * Operand 0 is the destination, by convention everywhere below. */
inline int nd_rank(device const float* meta) { return int(meta[0]); }

/* A linear element index turned into one operand's float slot.
 *
 * The divmod per axis is the real cost of the general path, which is why
 * every bulk operation here also has a contiguous entry point that takes a
 * flat count and skips this entirely. Most calls take that one. */
inline int nd_offset(device const float* meta, uint operand, uint linear) {
    int rank = int(meta[0]);
    int base = 1 + rank + int(operand) * (rank + 1);
    int at = int(meta[base]);
    uint rest = linear;
    for (int axis = rank - 1; axis >= 0; axis--) {
        uint extent = uint(meta[1 + axis]);
        at += int(rest % extent) * int(meta[base + 1 + axis]);
        rest /= extent;
    }
    return at;
}

/* What an integer or boolean dtype means on float32 storage: rounding to
 * nearest even -- which `rint` is -- and a clamp, or a flattening to 0 and 1.
 * NaN is truthy, as it is in numpy. */
inline float nd_convert(float value, uint mode, float low, float high) {
    if (mode == 1u) return clamp(rint(value), low, high);
    if (mode == 2u) return value != 0.0f ? 1.0f : 0.0f;
    return value;
}

inline float nd_apply(float a, float b, uint op) {
    if (op == 0u) return a + b;
    if (op == 1u) return a - b;
    if (op == 2u) return a * b;
    return a / b;
}

/* Complex arithmetic on an interleaved pair. Division is the naive formula
 * rather than Smith's: the inputs are float32 and the intermediates are
 * float32, so the range Smith's rescaling protects is already gone by the
 * time the values arrive. */
inline float2 nd_apply2(float2 a, float2 b, uint op) {
    if (op == 0u) return a + b;
    if (op == 1u) return a - b;
    if (op == 2u) return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
    float divisor = b.x * b.x + b.y * b.y;
    return float2((a.x * b.x + a.y * b.y) / divisor, (a.y * b.x - a.x * b.y) / divisor);
}

/* A real scalar meets a complex array the way numpy has it: it is a complex
 * number with a zero imaginary part, so it lands on the real lane for a sum
 * and on both lanes for a product. */
inline float2 nd_apply2_scalar(float2 a, float s, uint op) {
    if (op == 0u) return float2(a.x + s, a.y);
    if (op == 1u) return float2(a.x - s, a.y);
    if (op == 2u) return a * s;
    return a / s;
}

/* Exclusive prefix sum of `value` over a 256-thread group, handing every
 * thread its running position and the group's total.
 *
 * `warp_totals` must hold ND_WARPS uints. Barriered at both ends so the same
 * scratch survives a second call. */
inline uint nd_group_scan(
    uint value,
    threadgroup uint* warp_totals,
    uint lane,
    uint warp,
    thread uint* group_total
) {
    uint within = simd_prefix_exclusive_sum(value);
    uint warp_sum = simd_sum(value);
    if (lane == 0u) warp_totals[warp] = warp_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base = 0u;
    uint total = 0u;
    for (uint j = 0u; j < ND_WARPS; j++) {
        if (j < warp) base += warp_totals[j];
        total += warp_totals[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    *group_total = total;
    return base + within;
}

/* Copy and cast, dense. `count` is float slots, so lanes need no mention. */
kernel void nd_copy(device float* dst [[buffer(0)]],
                    device const float* src [[buffer(1)]],
                    constant uint& count [[buffer(2)]],
                    constant uint& mode [[buffer(3)]],
                    constant uint& low_bits [[buffer(4)]],
                    constant uint& high_bits [[buffer(5)]],
                    uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    dst[id] = nd_convert(src[id], mode, as_type<float>(low_bits), as_type<float>(high_bits));
}

/* Copy, cast and materialise through the layout buffer, and the one pass that
 * changes lane count: a real source fills a complex destination's imaginary
 * lane with zero, and a complex source dropped into a real destination loses
 * it. Every other kernel in the package takes one lane count for both sides. */
kernel void nd_copy_strided(device float* dst [[buffer(0)]],
                            device const float* src [[buffer(1)]],
                            device const float* meta [[buffer(2)]],
                            constant uint& count [[buffer(3)]],
                            constant uint& dst_lanes [[buffer(4)]],
                            constant uint& src_lanes [[buffer(5)]],
                            constant uint& mode [[buffer(6)]],
                            constant uint& low_bits [[buffer(7)]],
                            constant uint& high_bits [[buffer(8)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    int to = nd_offset(meta, 0u, id);
    int from = nd_offset(meta, 1u, id);
    float low = as_type<float>(low_bits);
    float high = as_type<float>(high_bits);
    for (uint lane = 0u; lane < dst_lanes; lane++) {
        float value = lane < src_lanes ? src[from + int(lane)] : 0.0f;
        dst[to + int(lane)] = nd_convert(value, mode, low, high);
    }
}

/* Fill, dense. `count` is elements: a complex fill takes a real value, so the
 * imaginary lane goes to zero rather than to a copy of it. */
kernel void nd_fill(device float* dst [[buffer(0)]],
                    constant uint& count [[buffer(1)]],
                    constant uint& lanes [[buffer(2)]],
                    constant uint& bits [[buffer(3)]],
                    uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint at = id * lanes;
    dst[at] = as_type<float>(bits);
    for (uint lane = 1u; lane < lanes; lane++) dst[at + lane] = 0.0f;
}

kernel void nd_fill_strided(device float* dst [[buffer(0)]],
                            device const float* meta [[buffer(1)]],
                            constant uint& count [[buffer(2)]],
                            constant uint& lanes [[buffer(3)]],
                            constant uint& bits [[buffer(4)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    int at = nd_offset(meta, 0u, id);
    dst[at] = as_type<float>(bits);
    for (uint lane = 1u; lane < lanes; lane++) dst[at + int(lane)] = 0.0f;
}

/* `a[mask] = value`. Operand 1 is the mask, which broadcasts like any other
 * operand, so a whole row can be switched on by a column of flags. */
kernel void nd_fill_masked(device float* dst [[buffer(0)]],
                           device const float* mask [[buffer(1)]],
                           device const float* meta [[buffer(2)]],
                           constant uint& count [[buffer(3)]],
                           constant uint& lanes [[buffer(4)]],
                           constant uint& bits [[buffer(5)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    if (mask[nd_offset(meta, 1u, id)] == 0.0f) return;
    int at = nd_offset(meta, 0u, id);
    dst[at] = as_type<float>(bits);
    for (uint lane = 1u; lane < lanes; lane++) dst[at + int(lane)] = 0.0f;
}

/* The four arithmetic ops, dense. `count` is elements, because a complex
 * product is not a per-slot operation.
 *
 * `mode`, `low` and `high` are the destination dtype's conversion, the same
 * three `nd_copy` takes: a sum of two int32 arrays that leaves the int32
 * range saturates here rather than being stored as a float the dtype cannot
 * hold. Only the single-lane path converts -- a complex destination is never
 * an integer dtype, so its mode is always CAST_NONE. */
kernel void nd_binary(device float* dst [[buffer(0)]],
                      device const float* a [[buffer(1)]],
                      device const float* b [[buffer(2)]],
                      constant uint& count [[buffer(3)]],
                      constant uint& lanes [[buffer(4)]],
                      constant uint& op [[buffer(5)]],
                      constant uint& mode [[buffer(6)]],
                      constant uint& low_bits [[buffer(7)]],
                      constant uint& high_bits [[buffer(8)]],
                      constant uint& a_wrap [[buffer(9)]],
                      constant uint& b_wrap [[buffer(10)]],
                      uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    /* Zero is dense; anything else is an operand that repeats every `wrap`
     * elements, which is a row against every row of a matrix. That case used
     * to fall to the strided walk and read 67 GB/s where the dense pair reads
     * 355 for more bytes. One divide against a whole index walk. */
    uint ai = a_wrap == 0u ? id : id - (id / a_wrap) * a_wrap;
    uint bi = b_wrap == 0u ? id : id - (id / b_wrap) * b_wrap;
    if (lanes == 1u) {
        float value = nd_apply(a[ai], b[bi], op);
        dst[id] = nd_convert(value, mode, as_type<float>(low_bits), as_type<float>(high_bits));
        return;
    }
    uint at = id * 2u;
    uint aat = ai * 2u;
    uint bat = bi * 2u;
    float2 value = nd_apply2(float2(a[aat], a[aat + 1u]), float2(b[bat], b[bat + 1u]), op);
    dst[at] = value.x;
    dst[at + 1u] = value.y;
}

kernel void nd_binary_strided(device float* dst [[buffer(0)]],
                              device const float* a [[buffer(1)]],
                              device const float* b [[buffer(2)]],
                              device const float* meta [[buffer(3)]],
                              constant uint& count [[buffer(4)]],
                              constant uint& lanes [[buffer(5)]],
                              constant uint& op [[buffer(6)]],
                              constant uint& mode [[buffer(7)]],
                              constant uint& low_bits [[buffer(8)]],
                              constant uint& high_bits [[buffer(9)]],
                              uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    int to = nd_offset(meta, 0u, id);
    int left = nd_offset(meta, 1u, id);
    int right = nd_offset(meta, 2u, id);
    if (lanes == 1u) {
        float one = nd_apply(a[left], b[right], op);
        dst[to] = nd_convert(one, mode, as_type<float>(low_bits), as_type<float>(high_bits));
        return;
    }
    float2 value = nd_apply2(float2(a[left], a[left + 1]), float2(b[right], b[right + 1]), op);
    dst[to] = value.x;
    dst[to + 1] = value.y;
}

kernel void nd_binary_scalar(device float* dst [[buffer(0)]],
                             device const float* a [[buffer(1)]],
                             constant uint& count [[buffer(2)]],
                             constant uint& lanes [[buffer(3)]],
                             constant uint& op [[buffer(4)]],
                             constant uint& bits [[buffer(5)]],
                             constant uint& mode [[buffer(6)]],
                             constant uint& low_bits [[buffer(7)]],
                             constant uint& high_bits [[buffer(8)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float scalar = as_type<float>(bits);
    if (lanes == 1u) {
        float one = nd_apply(a[id], scalar, op);
        dst[id] = nd_convert(one, mode, as_type<float>(low_bits), as_type<float>(high_bits));
        return;
    }
    uint at = id * 2u;
    float2 value = nd_apply2_scalar(float2(a[at], a[at + 1u]), scalar, op);
    dst[at] = value.x;
    dst[at + 1u] = value.y;
}

kernel void nd_binary_scalar_strided(device float* dst [[buffer(0)]],
                                     device const float* a [[buffer(1)]],
                                     device const float* meta [[buffer(2)]],
                                     constant uint& count [[buffer(3)]],
                                     constant uint& lanes [[buffer(4)]],
                                     constant uint& op [[buffer(5)]],
                                     constant uint& bits [[buffer(6)]],
                                     constant uint& mode [[buffer(7)]],
                                     constant uint& low_bits [[buffer(8)]],
                                     constant uint& high_bits [[buffer(9)]],
                                     uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    int to = nd_offset(meta, 0u, id);
    int from = nd_offset(meta, 1u, id);
    float scalar = as_type<float>(bits);
    if (lanes == 1u) {
        float one = nd_apply(a[from], scalar, op);
        dst[to] = nd_convert(one, mode, as_type<float>(low_bits), as_type<float>(high_bits));
        return;
    }
    float2 value = nd_apply2_scalar(float2(a[from], a[from + 1]), scalar, op);
    dst[to] = value.x;
    dst[to + 1] = value.y;
}

/* dst[i] = src[index[i]], both sides dense. A negative index counts from the
 * end; anything still outside is clamped, because a kernel cannot raise. The
 * caller validates -- `take` does it with one max reduction, which is cheaper
 * than a per-thread flag buffer nobody reads on the happy path. */
kernel void nd_gather(device float* dst [[buffer(0)]],
                      device const float* src [[buffer(1)]],
                      device const float* index [[buffer(2)]],
                      constant uint& count [[buffer(3)]],
                      constant uint& lanes [[buffer(4)]],
                      constant uint& limit [[buffer(5)]],
                      uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    int at = int(index[id]);
    if (at < 0) at += int(limit);
    at = clamp(at, 0, int(limit) - 1);
    for (uint lane = 0u; lane < lanes; lane++) {
        dst[id * lanes + lane] = src[uint(at) * lanes + lane];
    }
}

/* dst[index[i]] = src[i * src_step], optionally gated by mask[i].
 *
 * The gate is what turns the scan below into a compaction: scan a mask into
 * offsets, then scatter the source through those offsets with the same mask,
 * and the survivors land packed. `src_step` of zero broadcasts one value
 * across every target. When `use_mask` is zero the mask argument is bound to
 * the index buffer, which is read-only and already resident. */
kernel void nd_scatter(device float* dst [[buffer(0)]],
                       device const float* index [[buffer(1)]],
                       device const float* src [[buffer(2)]],
                       device const float* mask [[buffer(3)]],
                       constant uint& count [[buffer(4)]],
                       constant uint& lanes [[buffer(5)]],
                       constant uint& limit [[buffer(6)]],
                       constant uint& use_mask [[buffer(7)]],
                       constant uint& src_step [[buffer(8)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    if (use_mask != 0u && mask[id] == 0.0f) return;
    int at = int(index[id]);
    if (at < 0) at += int(limit);
    at = clamp(at, 0, int(limit) - 1);
    for (uint lane = 0u; lane < lanes; lane++) {
        dst[uint(at) * lanes + lane] = src[id * src_step + lane];
    }
}

/* Pass one of the exclusive scan: how many non-zeros each group's chunk holds.
 *
 * A group owns `span` contiguous elements and each of its threads owns `per`
 * contiguous elements inside that, the same partition pass three re-walks --
 * a prefix has to be built in element order, so the strided read pattern a
 * pure total would prefer is not available to the pass that needs it, and
 * matching partitions is worth more than coalescing one of the two. */
kernel void nd_scan_totals(device const float* src [[buffer(0)]],
                           device float* partials [[buffer(1)]],
                           constant uint& count [[buffer(2)]],
                           constant uint& span [[buffer(3)]],
                           constant uint& per [[buffer(4)]],
                           uint group [[threadgroup_position_in_grid]],
                           uint lid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint warp_totals[ND_WARPS];
    uint start = group * span + lid * per;
    uint mine = 0u;
    for (uint k = 0u; k < per; k++) {
        uint at = start + k;
        if (at < count && src[at] != 0.0f) mine++;
    }
    uint total = 0u;
    nd_group_scan(mine, warp_totals, lane, warp, &total);
    if (lid == 0u) partials[group] = float(total);
}

/* Pass two: one group turns the per-group totals into per-group bases, and
 * reports the grand total. Serial in the number of groups, which is why the
 * grid is capped rather than one group per 256 elements. */
kernel void nd_scan_bases(device float* partials [[buffer(0)]],
                          device float* totals [[buffer(1)]],
                          constant uint& groups [[buffer(2)]],
                          constant uint& per [[buffer(3)]],
                          uint lid [[thread_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint warp_totals[ND_WARPS];
    uint start = lid * per;
    uint mine = 0u;
    for (uint k = 0u; k < per; k++) {
        uint at = start + k;
        if (at < groups) mine += uint(partials[at]);
    }
    uint grand = 0u;
    uint running = nd_group_scan(mine, warp_totals, lane, warp, &grand);
    for (uint k = 0u; k < per; k++) {
        uint at = start + k;
        if (at >= groups) break;
        uint held = uint(partials[at]);
        partials[at] = float(running);
        running += held;
    }
    if (lid == 0u) totals[0] = float(grand);
}

/* Pass three: each element's exclusive offset, which is where it would land
 * if the non-zeros were packed. */
kernel void nd_scan_write(device float* dst [[buffer(0)]],
                          device const float* src [[buffer(1)]],
                          device const float* partials [[buffer(2)]],
                          constant uint& count [[buffer(3)]],
                          constant uint& span [[buffer(4)]],
                          constant uint& per [[buffer(5)]],
                          uint group [[threadgroup_position_in_grid]],
                          uint lid [[thread_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint warp_totals[ND_WARPS];
    uint start = group * span + lid * per;
    uint mine = 0u;
    for (uint k = 0u; k < per; k++) {
        uint at = start + k;
        if (at < count && src[at] != 0.0f) mine++;
    }
    uint ignored = 0u;
    uint running = uint(partials[group]) + nd_group_scan(mine, warp_totals, lane, warp, &ignored);
    for (uint k = 0u; k < per; k++) {
        uint at = start + k;
        if (at >= count) break;
        dst[at] = float(running);
        if (src[at] != 0.0f) running++;
    }
}
"""

pub let SOURCE = PRELUDE

var _kernels: dict[str, Kernel] = {}

#: The compiled pipeline for one entry point, kept for the life of the process.
#:
#: Keyed on the entry name alone. `Kernel.compile` interns on the source text
#: as well, and hashing a thirty-kilobyte shader costs more than encoding the
#: dispatch that follows it; every entry point in jainum carries its module's
#: prefix, so the names cannot collide and the source need not be consulted.
pub fn kernel_of(source: str, entry: str) -> Kernel {
    let existing = _kernels.get(entry, null)
    if existing is not null { return existing }
    let kernel = Kernel.compile(source, entry)
    _kernels[entry] = kernel
    return kernel
}

pub fn require_device(what: str) -> void {
    if not gpu.is_available() { throw RuntimeError(f"jainum needs a GPU to {what}") }
}
