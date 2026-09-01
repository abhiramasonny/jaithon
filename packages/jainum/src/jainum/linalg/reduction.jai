#: The one kernel in linalg: the folds a norm is built out of.
from std.gpu import Buffer
from ..array import FLOAT32, GROUP, NDArray, PRELUDE
from ..array import f32_bits, fill, is_contiguous, kernel_of, layout_buffer

#: Reduction selector: running total of `|x|`.
pub const RED_SUM_ABS = 0
#: Reduction selector: running total of `x*x`.
pub const RED_SUM_SQ = 1
#: Reduction selector: largest `|x|`.
pub const RED_MAX_ABS = 2
#: Reduction selector: smallest `|x|`.
pub const RED_MIN_ABS = 3
#: Reduction selector: running total of `|x|^p`.
pub const RED_SUM_POW = 4
#: Reduction selector: how many elements are non-zero.
pub const RED_COUNT = 5

#: Chunks one output row's reduction is cut into at most.
#:
#: A single group per row leaves 256 threads on a whole-array norm, which is a
#: sixteenth of this device. 64 chunks puts 16384 threads on it and still
#: leaves the fold pass a run of 64 floats per row to walk.
const NORM_CHUNKS = 64

let SOURCE = PRELUDE + """
/* Norm reductions. `op` is one of the RED_* selectors and every kernel below
 * carries it rather than being compiled six times: the branch is uniform
 * across the grid, so it costs one predicted jump per thread and saves five
 * pipelines that would otherwise be built on first use. */
inline float nl_seed(uint op) {
    if (op == 3u) return INFINITY;
    return 0.0f;
}

inline float nl_join(float a, float b, uint op) {
    if (op == 2u) return max(a, b);
    if (op == 3u) return min(a, b);
    return a + b;
}

inline float nl_take(float acc, float value, uint op, float power) {
    float m = fabs(value);
    if (op == 0u) return acc + m;
    if (op == 1u) return acc + value * value;
    if (op == 2u) return max(acc, m);
    if (op == 3u) return min(acc, m);
    if (op == 4u) return acc + pow(m, power);
    return acc + (value != 0.0f ? 1.0f : 0.0f);
}

/* Four at a time. The adds are paired rather than chained so the two halves
 * issue independently; the accumulator is still serial across iterations,
 * which is where the dependency that matters lives. */
inline float nl_take4(float acc, float4 value, uint op, float power) {
    float4 m = fabs(value);
    if (op == 0u) return acc + ((m.x + m.y) + (m.z + m.w));
    if (op == 1u) return acc + dot(value, value);
    if (op == 2u) return max(acc, max(max(m.x, m.y), max(m.z, m.w)));
    if (op == 3u) return min(acc, min(min(m.x, m.y), min(m.z, m.w)));
    if (op == 4u) {
        float4 raised = pow(m, float4(power));
        return acc + ((raised.x + raised.y) + (raised.z + raised.w));
    }
    float4 hit = select(float4(0.0f), float4(1.0f), value != 0.0f);
    return acc + ((hit.x + hit.y) + (hit.z + hit.w));
}

/* Fold `value` across a whole threadgroup under `op`. `scratch` holds one
 * float per simdgroup; barriered at both ends so it can be reused. */
inline float nl_group_join(
    float value,
    threadgroup float* scratch,
    uint lane,
    uint warp,
    uint op
) {
    float folded = op == 2u ? simd_max(value) : (op == 3u ? simd_min(value) : simd_sum(value));
    if (lane == 0u) scratch[warp] = folded;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = scratch[0];
    for (uint j = 1u; j < ND_WARPS; j++) total = nl_join(total, scratch[j], op);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

/* Pass one: `chunks` groups per output row, each folding its stride of the
 * reduced run into one partial.
 *
 * Three read paths, narrowest first. `quads` non-zero means the run is dense
 * and both its base and its length are multiples of four, so a float4 load is
 * aligned; `packed` alone means dense but not divisible; neither means the
 * general strided walk through `nd_offset`, which is what a reduction over a
 * transposed view or a middle axis takes. */
kernel void nl_reduce_partial(device float* partials [[buffer(0)]],
                              device const float* src [[buffer(1)]],
                              device const float* meta [[buffer(2)]],
                              constant uint& inner [[buffer(3)]],
                              constant uint& chunks [[buffer(4)]],
                              constant uint& op [[buffer(5)]],
                              constant uint& power_bits [[buffer(6)]],
                              constant uint& base [[buffer(7)]],
                              constant uint& packed [[buffer(8)]],
                              constant uint& quads [[buffer(9)]],
                              uint group [[threadgroup_position_in_grid]],
                              uint lid [[thread_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[ND_WARPS];
    /* One divide and a subtract, not a divide and a modulo: the device has no
     * integer divide and pays the whole reciprocal sequence for each. */
    uint row = group / chunks;
    uint chunk = group - row * chunks;
    float power = as_type<float>(power_bits);
    float acc = nl_seed(op);
    uint start = chunk * ND_GROUP + lid;
    uint step = chunks * ND_GROUP;
    if (quads != 0u) {
        device const float4* run = (device const float4*)(src + base + row * inner);
        for (uint at = start; at < quads; at += step) acc = nl_take4(acc, run[at], op, power);
    } else if (packed != 0u) {
        uint from = base + row * inner;
        for (uint at = start; at < inner; at += step) {
            acc = nl_take(acc, src[from + at], op, power);
        }
    } else {
        for (uint at = start; at < inner; at += step) {
            acc = nl_take(acc, src[nd_offset(meta, 0u, row * inner + at)], op, power);
        }
    }
    float total = nl_group_join(acc, scratch, lane, warp, op);
    if (lid == 0u) partials[group] = total;
}

/* Pass two: one thread per output, folding that row's `chunks` partials and
 * applying whatever root the norm ends in. `finish` is 1 for a square root
 * and 2 for the general `1/p` power. */
kernel void nl_reduce_fold(device float* dst [[buffer(0)]],
                           device const float* partials [[buffer(1)]],
                           constant uint& outer [[buffer(2)]],
                           constant uint& chunks [[buffer(3)]],
                           constant uint& op [[buffer(4)]],
                           constant uint& finish [[buffer(5)]],
                           constant uint& root_bits [[buffer(6)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= outer) return;
    uint from = id * chunks;
    float acc = partials[from];
    for (uint k = 1u; k < chunks; k++) acc = nl_join(acc, partials[from + k], op);
    if (finish == 1u) acc = sqrt(acc);
    if (finish == 2u) acc = pow(acc, as_type<float>(root_bits));
    dst[id] = acc;
}
"""

#: A view with `axes` moved to the end, keeping the others in their order.
#:
#: A pure stride permutation, so nothing is copied: the reduction below wants
#: the reduced run last so a linear index splits into a row and a position
#: with one division.
pub fn _move_axes_last(a: NDArray, axes: list[int]) -> NDArray {
    var taken = [false for _axis in a.shape]
    for axis in axes { taken[axis] = true }
    var shape: list[int] = []
    var strides: list[int] = []
    for axis in 0..a.ndim() {
        if taken[axis] { continue }
        shape.push(a.shape[axis])
        strides.push(a.strides[axis])
    }
    for axis in axes {
        shape.push(a.shape[axis])
        strides.push(a.strides[axis])
    }
    return a.view(shape, strides, a.offset)
}

#: `shape` with `axes` dropped, or set to one when `keepdims`.
pub fn _kept_shape(shape: list[int], axes: list[int], keepdims: bool) -> list[int] {
    var taken = [false for _axis in shape]
    for axis in axes { taken[axis] = true }
    var out: list[int] = []
    for axis in 0..shape.len() {
        if not taken[axis] {
            out.push(shape[axis])
        } elif keepdims {
            out.push(1)
        }
    }
    return out
}

#: The root a reduction ends in: none, a square root, or `1/p`.
fn _finish_of(op: int) -> int {
    if op == RED_SUM_SQ { return 1 }
    if op == RED_SUM_POW { return 2 }
    return 0
}

#: Fold `axes` of `a` under `op`, into a fresh array shaped `out_shape`.
#:
#: `out_shape` is passed rather than derived because a matrix norm composes
#: two of these and only the second one knows where the `keepdims` axes go.
pub fn _reduce_axes(
    a: NDArray,
    axes: list[int],
    op: int,
    power: float,
    out_shape: list[int]
) -> NDArray {
    let view = _move_axes_last(a, axes)
    var inner = 1
    for axis in axes { inner *= a.shape[axis] }
    let out = NDArray(out_shape, FLOAT32)
    let outer = out.size()
    if outer == 0 { return out }
    if inner == 0 {
        fill(out, 0.0)
        return out
    }

    let packed = is_contiguous(view.shape, view.strides, 1)
    #: A float4 load must be sixteen-byte aligned. The run's length divides by
    #: four so every row starts a multiple of four along, and the base has to
    #: as well — a view sliced to an odd column would otherwise fault.
    let vectored = packed and inner % 4 == 0 and view.offset % 4 == 0
    let quads = if vectored { inner // 4 } else { 0 }
    let chunks = max(1, min(NORM_CHUNKS, (inner + GROUP * 4 - 1) // (GROUP * 4)))

    let partials = Buffer(outer * chunks)
    defer { partials.free() }
    let meta = layout_buffer([view])
    defer { meta.free() }
    kernel_of(SOURCE, "nl_reduce_partial").dispatch_async([partials, view.storage, meta], outer * chunks * GROUP, [inner, chunks, op, f32_bits(power), view.offset, if packed {
        1
    } else {
        0
    }, quads], GROUP)
    let root = if power != 0.0 { 1.0 / power } else { 0.0 }
    kernel_of(SOURCE, "nl_reduce_fold").dispatch_async(
        [out.storage, partials],
        outer,
        [outer, chunks, op, _finish_of(op), f32_bits(root)]
    )
    return out
}
