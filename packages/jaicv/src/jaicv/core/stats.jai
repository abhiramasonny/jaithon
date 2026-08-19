#: Reductions over a `Mat`: extremes, sums, averages, norms, and counts.
#:
#: Each one reduces on the device into a small block of partial results and
#: finishes the last few hundred values on the host, so a 1080p frame costs one
#: dispatch and one short download rather than six million boxed floats.
#:
#: ```
#: let (low, high, low_at, high_at) = min_max_loc(gray)
#: let total = count_non_zero(mask)
#: normalize(depth, 0.0, 255.0, NORM_MINMAX)
#: ```
import std.gpu as gpu
import std.math
from std.gpu import Buffer, Kernel
from .mat import Mat, f32_bits
from .arith import OP_SUBTRACT, UN_ABS, binary_op, unary_op
from .types import CV_32F, CV_MAKETYPE, CV_MAT_CN, CV_MAT_DEPTH
from .types import NORM_INF, NORM_L1, NORM_L2, NORM_L2SQR, NORM_MINMAX
from .types import REDUCE_AVG, REDUCE_MAX, REDUCE_MIN, REDUCE_SUM
from .types import Point, Scalar

const GROUP = 256

let SOURCE = """#include <metal_stdlib>
using namespace metal;

constexpr constant uint GROUP = 256;

inline uint plane_index(uint id, uint cols, uint cn, uint off, uint step) {
    uint width = cols * cn;
    uint y = id / width;
    uint rest = id - y * width;
    return off + y * step + rest;
}

/* Sums, absolute sums, squares and non-zero counts in one pass: four
 * accumulators cost far less than four dispatches over the same pixels. */
kernel void stats_reduce(device const float* src [[buffer(0)]],
                         device const float* mask [[buffer(1)]],
                         device float* partials [[buffer(2)]],
                         constant uint& count [[buffer(3)]],
                         constant uint& cols [[buffer(4)]],
                         constant uint& cn [[buffer(5)]],
                         constant uint& off [[buffer(6)]],
                         constant uint& step [[buffer(7)]],
                         constant uint& m_off [[buffer(8)]],
                         constant uint& m_step [[buffer(9)]],
                         constant uint& use_mask [[buffer(10)]],
                         constant uint& channel [[buffer(11)]],
                         uint gid [[thread_position_in_grid]],
                         uint lid [[thread_index_in_threadgroup]],
                         uint wid [[threadgroup_position_in_grid]]) {
    threadgroup float total[GROUP];
    threadgroup float absolute[GROUP];
    threadgroup float squares[GROUP];
    threadgroup float nonzero[GROUP];
    float value = 0.0f;
    bool live = false;
    if (gid < count) {
        uint pixel = gid;
        uint index = off + (pixel / cols) * step + (pixel % cols) * cn + channel;
        live = true;
        if (use_mask != 0u) {
            live = mask[plane_index(pixel, cols, 1u, m_off, m_step)] != 0.0f;
        }
        if (live) value = src[index];
    }
    total[lid] = live ? value : 0.0f;
    absolute[lid] = live ? fabs(value) : 0.0f;
    squares[lid] = live ? value * value : 0.0f;
    nonzero[lid] = (live && value != 0.0f) ? 1.0f : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = GROUP / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            total[lid] += total[lid + stride];
            absolute[lid] += absolute[lid + stride];
            squares[lid] += squares[lid + stride];
            nonzero[lid] += nonzero[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) {
        partials[wid * 4u + 0u] = total[0];
        partials[wid * 4u + 1u] = absolute[0];
        partials[wid * 4u + 2u] = squares[0];
        partials[wid * 4u + 3u] = nonzero[0];
    }
}

kernel void stats_extrema(device const float* src [[buffer(0)]],
                          device const float* mask [[buffer(1)]],
                          device float* partials [[buffer(2)]],
                          constant uint& count [[buffer(3)]],
                          constant uint& cols [[buffer(4)]],
                          constant uint& cn [[buffer(5)]],
                          constant uint& off [[buffer(6)]],
                          constant uint& step [[buffer(7)]],
                          constant uint& m_off [[buffer(8)]],
                          constant uint& m_step [[buffer(9)]],
                          constant uint& use_mask [[buffer(10)]],
                          constant uint& channel [[buffer(11)]],
                          uint gid [[thread_position_in_grid]],
                          uint lid [[thread_index_in_threadgroup]],
                          uint wid [[threadgroup_position_in_grid]]) {
    threadgroup float lows[GROUP];
    threadgroup float highs[GROUP];
    threadgroup float low_at[GROUP];
    threadgroup float high_at[GROUP];
    float value = 0.0f;
    bool live = false;
    if (gid < count) {
        uint index = off + (gid / cols) * step + (gid % cols) * cn + channel;
        live = true;
        if (use_mask != 0u) {
            live = mask[plane_index(gid, cols, 1u, m_off, m_step)] != 0.0f;
        }
        if (live) value = src[index];
    }
    lows[lid] = live ? value : INFINITY;
    highs[lid] = live ? value : -INFINITY;
    low_at[lid] = live ? float(gid) : -1.0f;
    high_at[lid] = live ? float(gid) : -1.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = GROUP / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            if (lows[lid + stride] < lows[lid]) {
                lows[lid] = lows[lid + stride];
                low_at[lid] = low_at[lid + stride];
            }
            if (highs[lid + stride] > highs[lid]) {
                highs[lid] = highs[lid + stride];
                high_at[lid] = high_at[lid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) {
        partials[wid * 4u + 0u] = lows[0];
        partials[wid * 4u + 1u] = highs[0];
        partials[wid * 4u + 2u] = low_at[0];
        partials[wid * 4u + 3u] = high_at[0];
    }
}

kernel void stats_reduce_axis(device const float* src [[buffer(0)]],
                              device float* dst [[buffer(1)]],
                              constant uint& count [[buffer(2)]],
                              constant uint& rows [[buffer(3)]],
                              constant uint& cols [[buffer(4)]],
                              constant uint& cn [[buffer(5)]],
                              constant uint& off [[buffer(6)]],
                              constant uint& step [[buffer(7)]],
                              constant uint& d_off [[buffer(8)]],
                              constant uint& d_step [[buffer(9)]],
                              constant uint& dim [[buffer(10)]],
                              constant uint& op [[buffer(11)]],
                              uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint channel = id % cn;
    uint lane = id / cn;
    uint length = dim == 0u ? rows : cols;
    float acc = op == 3u ? INFINITY : (op == 2u ? -INFINITY : 0.0f);
    for (uint i = 0; i < length; ++i) {
        uint y = dim == 0u ? i : lane;
        uint x = dim == 0u ? lane : i;
        float value = src[off + y * step + x * cn + channel];
        if (op == 0u || op == 1u) acc += value;
        else if (op == 2u) acc = max(acc, value);
        else acc = min(acc, value);
    }
    if (op == 1u) acc /= float(length);
    dst[d_off + (dim == 0u ? 0u : lane) * d_step + (dim == 0u ? lane : 0u) * cn + channel] = acc;
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

fn reduce_partials(entry: str, src: Mat, mask: Mat?, channel: int) -> list[float] {
    src.flush()
    let count = src.rows * src.cols
    if count == 0 { return [] }
    let groups = (count + GROUP - 1) // GROUP
    let partials = Buffer(groups * 4)
    let use_mask = if mask is null { 0 } else { 1 }
    let mask_mat = if mask is null { src } else { mask }
    if mask is not null { mask.flush() }
    kernel_for(entry).dispatch(
        [src.storage, mask_mat.storage, partials],
        groups * GROUP,
        [
            count,
            src.cols,
            src.channels(),
            src.offset,
            src.step_,
            mask_mat.offset,
            mask_mat.step_,
            use_mask,
            channel,
        ],
        group_size: GROUP
    )
    let values = partials.download()
    partials.free()
    return values
}

#: Sum, absolute sum, sum of squares, and non-zero count of one channel.
pub fn channel_stats(src: Mat, channel: int, mask: Mat? = null) -> tuple[float, float, float, int] {
    let partials = reduce_partials("stats_reduce", src, mask, channel)
    var total = 0.0
    var absolute = 0.0
    var squares = 0.0
    var nonzero = 0.0
    var index = 0
    while index < partials.len() {
        total += partials[index]
        absolute += partials[index + 1]
        squares += partials[index + 2]
        nonzero += partials[index + 3]
        index += 4
    }
    return (total, absolute, squares, int(nonzero))
}

#: Sum of every element, one entry per channel.
pub fn sum_mat(src: Mat, mask: Mat? = null) -> Scalar {
    var out = [0.0, 0.0, 0.0, 0.0]
    for channel in 0..min(src.channels(), 4) {
        let (total, _absolute, _squares, _nonzero) = channel_stats(src, channel, mask: mask)
        out[channel] = total
    }
    return Scalar(out[0], out[1], out[2], out[3])
}

#: Number of elements that are not zero, summed over channels.
pub fn count_non_zero(src: Mat, mask: Mat? = null) -> int {
    var total = 0
    for channel in 0..src.channels() {
        let (_total, _absolute, _squares, nonzero) = channel_stats(src, channel, mask: mask)
        total += nonzero
    }
    return total
}

#: Average of each channel.
pub fn mean(src: Mat, mask: Mat? = null) -> Scalar {
    let denominator = if mask is null {
        float(src.rows * src.cols)
    } else {
        float(count_non_zero(mask))
    }
    if denominator <= 0.0 { return Scalar(0.0, 0.0, 0.0, 0.0) }
    var out = [0.0, 0.0, 0.0, 0.0]
    for channel in 0..min(src.channels(), 4) {
        let (total, _absolute, _squares, _nonzero) = channel_stats(src, channel, mask: mask)
        out[channel] = total / denominator
    }
    return Scalar(out[0], out[1], out[2], out[3])
}

#: Average and population standard deviation of each channel.
pub fn mean_std_dev(src: Mat, mask: Mat? = null) -> tuple[Scalar, Scalar] {
    let denominator = if mask is null {
        float(src.rows * src.cols)
    } else {
        float(count_non_zero(mask))
    }
    if denominator <= 0.0 {
        return (Scalar(0.0, 0.0, 0.0, 0.0), Scalar(0.0, 0.0, 0.0, 0.0))
    }
    var averages = [0.0, 0.0, 0.0, 0.0]
    var deviations = [0.0, 0.0, 0.0, 0.0]
    for channel in 0..min(src.channels(), 4) {
        let (total, _absolute, squares, _nonzero) = channel_stats(src, channel, mask: mask)
        let average = total / denominator
        let variance = max(0.0, squares / denominator - average * average)
        averages[channel] = average
        deviations[channel] = math.sqrt(variance)
    }
    return (
        Scalar(averages[0], averages[1], averages[2], averages[3]),
        Scalar(deviations[0], deviations[1], deviations[2], deviations[3]),
    )
}

#: Smallest and largest value and where each one is, over a single channel.
#:
#: Matches OpenCV in taking single-channel input; split a colour image first.
pub fn min_max_loc(src: Mat, mask: Mat? = null) -> tuple[float, float, Point, Point] {
    if src.channels() != 1 {
        throw ValueError("min_max_loc needs a single-channel Mat; split it first")
    }
    let partials = reduce_partials("stats_extrema", src, mask, 0)
    var low = 0.0
    var high = 0.0
    var low_at = -1
    var high_at = -1
    var index = 0
    var seen = false
    while index < partials.len() {
        let group_low = partials[index]
        let group_high = partials[index + 1]
        let group_low_at = int(partials[index + 2])
        let group_high_at = int(partials[index + 3])
        if group_low_at >= 0 and (not seen or group_low < low) {
            low = group_low
            low_at = group_low_at
        }
        if group_high_at >= 0 and (not seen or group_high > high) {
            high = group_high
            high_at = group_high_at
        }
        if group_low_at >= 0 or group_high_at >= 0 { seen = true }
        index += 4
    }
    let low_point = if low_at < 0 {
        Point(-1, -1)
    } else {
        Point(low_at % src.cols, low_at // src.cols)
    }
    let high_point = if high_at < 0 {
        Point(-1, -1)
    } else {
        Point(high_at % src.cols, high_at // src.cols)
    }
    return (low, high, low_point, high_point)
}

#: Smallest and largest value, ignoring where they are.
pub fn min_max(src: Mat, mask: Mat? = null) -> tuple[float, float] {
    let (low, high, _low_at, _high_at) = min_max_loc(src, mask: mask)
    return (low, high)
}

#: Length of `src` under one of the `NORM_` codes, or the distance to `other`.
pub fn norm(src: Mat, other: Mat? = null, norm_type: int = NORM_L2, mask: Mat? = null) -> float {
    var target = src
    if other is not null {
        let wide = CV_MAKETYPE(CV_32F, src.channels())
        target = unary_op(binary_op(src.convert_to(wide), other.convert_to(wide), OP_SUBTRACT), UN_ABS)
    }
    var l1 = 0.0
    var l2 = 0.0
    var inf = 0.0
    for channel in 0..target.channels() {
        let (_total, absolute, squares, _nonzero) = channel_stats(target, channel, mask: mask)
        l1 += absolute
        l2 += squares
        let plane = if target.channels() == 1 { target } else { target.split()[channel] }
        let (low, high) = min_max(plane, mask: mask)
        inf = max(inf, max(math.fabs(low), math.fabs(high)))
    }
    if norm_type == NORM_L1 { return l1 }
    if norm_type == NORM_L2SQR { return l2 }
    if norm_type == NORM_INF { return inf }
    return math.sqrt(l2)
}

#: Rescale so the values span `[alpha, beta]`, or so the chosen norm equals
#: `alpha`.
pub fn normalize(
    src: Mat,
    alpha: float = 1.0,
    beta: float = 0.0,
    norm_type: int = NORM_L2,
    dtype: int = -1,
    dst: Mat? = null
) -> Mat {
    let target_type = if dtype >= 0 { dtype } else { src.flags }
    if norm_type == NORM_MINMAX {
        if src.channels() != 1 {
            throw ValueError("NORM_MINMAX normalize needs a single-channel Mat")
        }
        let (low, high) = min_max(src)
        let span = high - low
        let low_target = min(alpha, beta)
        let high_target = max(alpha, beta)
        let scale = if span == 0.0 { 0.0 } else { (high_target - low_target) / span }
        let shift = low_target - low * scale
        let out = src.convert_to(target_type, scale, shift)
        if dst is not null { return out.copy_to(dst) }
        return out
    }
    let length = norm(src, norm_type: norm_type)
    let scale = if length == 0.0 { 0.0 } else { alpha / length }
    let out = src.convert_to(target_type, scale, 0.0)
    if dst is not null { return out.copy_to(dst) }
    return out
}

#: Collapse one axis with a `REDUCE_` operation.
#:
#: `dim` 0 collapses the rows into one row; `dim` 1 collapses the columns into
#: one column.
pub fn reduce(src: Mat, dim: int, op: int = REDUCE_SUM, dtype: int = -1) -> Mat {
    src.flush()
    let cn = src.channels()
    let out_rows = if dim == 0 { 1 } else { src.rows }
    let out_cols = if dim == 0 { src.cols } else { 1 }
    let target_type = if dtype >= 0 {
        CV_MAKETYPE(dtype, cn)
    } else {
        CV_MAKETYPE(CV_32F, cn)
    }
    let out = Mat(out_rows, out_cols, target_type)
    let count = out_rows * out_cols * cn
    if count == 0 { return out }
    kernel_for("stats_reduce_axis").dispatch_async(
        [src.storage, out.storage],
        count,
        [
            count,
            src.rows,
            src.cols,
            cn,
            src.offset,
            src.step_,
            out.offset,
            out.step_,
            dim,
            op,
        ]
    )
    out.invalidate()
    return out
}

#: Average of every element, over every channel at once.
pub fn mean_value(src: Mat, mask: Mat? = null) -> float {
    let averages = mean(src, mask: mask)
    var total = 0.0
    for channel in 0..src.channels() { total += averages[channel] }
    return total / float(src.channels())
}

#: Trace of a square matrix.
pub fn trace(src: Mat) -> Scalar {
    let limit = min(src.rows, src.cols)
    var out = [0.0, 0.0, 0.0, 0.0]
    for index in 0..limit {
        for channel in 0..min(src.channels(), 4) { out[channel] += src.at(index, index, channel) }
    }
    return Scalar(out[0], out[1], out[2], out[3])
}
