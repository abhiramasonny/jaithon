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
const GROUPS = 1024

let SOURCE = """#include <metal_stdlib>
using namespace metal;

constexpr constant uint GROUP = 256;
constexpr constant uint GROUPS = 1024;

inline uint plane_index(uint id, uint cols, uint cn, uint off, uint step) {
    uint width = cols * cn;
    uint y = id / width;
    uint rest = id - y * width;
    return off + y * step + rest;
}

/* Sums, absolute sums, squares and non-zero counts in one pass: four
 * accumulators cost far less than four dispatches over the same pixels.
 *
 * A fixed grid, not one thread per pixel. A 1080p frame is eight thousand
 * groups the other way round, and each of those threads read one float and
 * then spent eight barriers folding it -- more time in threadgroup memory
 * than in the picture. Here a thousand groups walk the frame in strides and
 * accumulate in registers, and the fold happens once per group.
 *
 * The packed path reads four floats at a time. It needs the rows to run
 * together with nothing between them, a single channel, and no mask, which
 * is what a plain frame is. */
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
                         constant uint& span [[buffer(12)]],
                         constant uint& packed [[buffer(13)]],
                         constant uint& slot [[buffer(14)]],
                         constant uint& dense [[buffer(15)]],
                         uint gid [[thread_position_in_grid]],
                         uint lid [[thread_index_in_threadgroup]],
                         uint wid [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup float lanes[4][32];
    float total = 0.0f;
    float absolute = 0.0f;
    float squares = 0.0f;
    float nonzero = 0.0f;
    if (packed != 0u) {
        device const float4* quads = (device const float4*)(src + off);
        uint wide = count / 4u;
        for (uint at = gid; at < wide; at += span) {
            float4 value = quads[at];
            float4 size = fabs(value);
            total += value.x + value.y + value.z + value.w;
            absolute += size.x + size.y + size.z + size.w;
            squares += dot(value, value);
            nonzero += float(value.x != 0.0f) + float(value.y != 0.0f)
                     + float(value.z != 0.0f) + float(value.w != 0.0f);
        }
    } else {
        for (uint pixel = gid; pixel < count; pixel += span) {
            if (use_mask != 0u
                && mask[plane_index(pixel, cols, 1u, m_off, m_step)] == 0.0f) continue;
            uint index = dense != 0u
                ? off + pixel * cn + channel
                : off + (pixel / cols) * step + (pixel % cols) * cn + channel;
            float value = src[index];
            total += value;
            absolute += fabs(value);
            squares += value * value;
            if (value != 0.0f) nonzero += 1.0f;
        }
    }
    if (lid < 32u) {
        lanes[0][lid] = 0.0f;
        lanes[1][lid] = 0.0f;
        lanes[2][lid] = 0.0f;
        lanes[3][lid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = simd_sum(total);
    absolute = simd_sum(absolute);
    squares = simd_sum(squares);
    nonzero = simd_sum(nonzero);
    if (lane == 0u) {
        lanes[0][warp] = total;
        lanes[1][warp] = absolute;
        lanes[2][warp] = squares;
        lanes[3][warp] = nonzero;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid != 0u) return;
    float4 folded = float4(0.0f);
    for (uint j = 0u; j < 32u; j++) {
        folded.x += lanes[0][j];
        folded.y += lanes[1][j];
        folded.z += lanes[2][j];
        folded.w += lanes[3][j];
    }
    uint base = (slot * GROUPS + wid) * 4u;
    partials[base + 0u] = folded.x;
    partials[base + 1u] = folded.y;
    partials[base + 2u] = folded.z;
    partials[base + 3u] = folded.w;
}

/* The extremes, on the same fixed grid.
 *
 * The position has to travel with the value, and a `simd_min` on the value
 * alone loses it. So the value folds first, and then the positions of whoever
 * matches fold by `simd_min` in their own right -- which picks the earliest
 * pixel among equals, the tie OpenCV reports.
 *
 * A missing position is `INFINITY` rather than a negative, so the same
 * `simd_min` handles "saw nothing" without a second test. */
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
                          constant uint& span [[buffer(12)]],
                          constant uint& packed [[buffer(13)]],
                          constant uint& slot [[buffer(14)]],
                          constant uint& dense [[buffer(15)]],
                          uint gid [[thread_position_in_grid]],
                          uint lid [[thread_index_in_threadgroup]],
                          uint wid [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup float lows[32];
    threadgroup float highs[32];
    threadgroup float low_at[32];
    threadgroup float high_at[32];
    float my_low = INFINITY;
    float my_high = -INFINITY;
    float my_low_at = INFINITY;
    float my_high_at = INFINITY;
    if (packed != 0u) {
        device const float4* quads = (device const float4*)(src + off);
        uint wide = count / 4u;
        for (uint at = gid; at < wide; at += span) {
            float4 value = quads[at];
            float low = min(min(value.x, value.y), min(value.z, value.w));
            float high = max(max(value.x, value.y), max(value.z, value.w));
            if (low <= my_low) {
                uint which = value.x == low ? 0u : (value.y == low ? 1u : (value.z == low ? 2u : 3u));
                float where = float(at * 4u + which);
                if (low < my_low || where < my_low_at) {
                    my_low = low;
                    my_low_at = where;
                }
            }
            if (high >= my_high) {
                uint which = value.x == high ? 0u : (value.y == high ? 1u : (value.z == high ? 2u : 3u));
                float where = float(at * 4u + which);
                if (high > my_high || where < my_high_at) {
                    my_high = high;
                    my_high_at = where;
                }
            }
        }
    } else {
        for (uint pixel = gid; pixel < count; pixel += span) {
            if (use_mask != 0u
                && mask[plane_index(pixel, cols, 1u, m_off, m_step)] == 0.0f) continue;
            uint index = dense != 0u
                ? off + pixel * cn + channel
                : off + (pixel / cols) * step + (pixel % cols) * cn + channel;
            float value = src[index];
            bool lower = value < my_low;
            my_low = lower ? value : my_low;
            my_low_at = lower ? float(pixel) : my_low_at;
            bool higher = value > my_high;
            my_high = higher ? value : my_high;
            my_high_at = higher ? float(pixel) : my_high_at;
        }
    }
    if (lid < 32u) {
        lows[lid] = INFINITY;
        highs[lid] = -INFINITY;
        low_at[lid] = INFINITY;
        high_at[lid] = INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float best_low = simd_min(my_low);
    float best_high = simd_max(my_high);
    float won_low_at = simd_min(my_low == best_low ? my_low_at : INFINITY);
    float won_high_at = simd_min(my_high == best_high ? my_high_at : INFINITY);
    if (lane == 0u) {
        lows[warp] = best_low;
        highs[warp] = best_high;
        low_at[warp] = won_low_at;
        high_at[warp] = won_high_at;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid != 0u) return;
    float low = INFINITY;
    float high = -INFINITY;
    float low_where = INFINITY;
    float high_where = INFINITY;
    for (uint j = 0u; j < 32u; j++) {
        if (low_at[j] < INFINITY && (lows[j] < low || (lows[j] == low && low_at[j] < low_where))) {
            low = lows[j];
            low_where = low_at[j];
        }
        if (high_at[j] < INFINITY && (highs[j] > high || (highs[j] == high && high_at[j] < high_where))) {
            high = highs[j];
            high_where = high_at[j];
        }
    }
    uint base = (slot * GROUPS + wid) * 4u;
    partials[base + 0u] = low_where < INFINITY ? low : 0.0f;
    partials[base + 1u] = high_where < INFINITY ? high : 0.0f;
    partials[base + 2u] = low_where < INFINITY ? low_where : -1.0f;
    partials[base + 3u] = high_where < INFINITY ? high_where : -1.0f;
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
/* The partials folded down on the device, so only four numbers come back.
 *
 * A 1080p frame is eight thousand groups, and reading all of their partials
 * back to add them up on the host cost more than the reduction did: 0.5 ms to
 * sum an eight megabyte picture, against 0.03 ms of actual reading. One more
 * pass, and the download is four floats.
 *
 * A threadgroup to each of the four lanes -- total, absolute, squares, count
 * -- because they combine independently. */
kernel void stats_join(device const float* partials [[buffer(0)]],
                       device float* out [[buffer(1)]],
                       constant uint& groups [[buffer(2)]],
                       uint slot [[threadgroup_position_in_grid]],
                       uint lid [[thread_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup float lanes[32];
    float running = 0.0f;
    uint channel = slot / 4u;
    uint field = slot - channel * 4u;
    uint base = channel * GROUPS * 4u + field;
    for (uint g = lid; g < groups; g += GROUP) running += partials[g * 4u + base];
    if (lid < 32u) lanes[lid] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    running = simd_sum(running);
    if (lane == 0u) lanes[warp] = running;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid != 0u) return;
    float total = 0.0f;
    for (uint j = 0u; j < 32u; j++) total += lanes[j];
    out[slot] = total;
}

/* The same fold for the extremes, which combine by comparison rather than by
 * adding, and which carry the position the extreme was found at.
 *
 * A group that saw nothing -- every element masked out -- marks its positions
 * negative, and those must not win. */
kernel void stats_join_extrema(device const float* partials [[buffer(0)]],
                               device float* out [[buffer(1)]],
                               constant uint& groups [[buffer(2)]],
                               uint lid [[thread_index_in_threadgroup]]) {
    threadgroup float best_low[GROUP];
    threadgroup float best_high[GROUP];
    threadgroup float low_at[GROUP];
    threadgroup float high_at[GROUP];

    float my_low = 0.0f;
    float my_high = 0.0f;
    float my_low_at = -1.0f;
    float my_high_at = -1.0f;
    for (uint g = lid; g < groups; g += GROUP) {
        float group_low = partials[g * 4u + 0u];
        float group_high = partials[g * 4u + 1u];
        float group_low_at = partials[g * 4u + 2u];
        float group_high_at = partials[g * 4u + 3u];
        if (group_low_at >= 0.0f
            && (my_low_at < 0.0f || group_low < my_low
                || (group_low == my_low && group_low_at < my_low_at))) {
            my_low = group_low;
            my_low_at = group_low_at;
        }
        if (group_high_at >= 0.0f
            && (my_high_at < 0.0f || group_high > my_high
                || (group_high == my_high && group_high_at < my_high_at))) {
            my_high = group_high;
            my_high_at = group_high_at;
        }
    }
    best_low[lid] = my_low;
    best_high[lid] = my_high;
    low_at[lid] = my_low_at;
    high_at[lid] = my_high_at;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid != 0u) return;
    float low = 0.0f;
    float high = 0.0f;
    float low_where = -1.0f;
    float high_where = -1.0f;
    for (uint j = 0u; j < GROUP; j++) {
        if (low_at[j] >= 0.0f
            && (low_where < 0.0f || best_low[j] < low
                || (best_low[j] == low && low_at[j] < low_where))) {
            low = best_low[j];
            low_where = low_at[j];
        }
        if (high_at[j] >= 0.0f
            && (high_where < 0.0f || best_high[j] > high
                || (best_high[j] == high && high_at[j] < high_where))) {
            high = best_high[j];
            high_where = high_at[j];
        }
    }
    out[0] = low;
    out[1] = high;
    out[2] = low_where;
    out[3] = high_where;
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

#: Whether the frame can be read four floats at a time.
#:
#: Needs the rows to run together with nothing between them, one channel, no
#: mask, and a start that lands on a quad boundary -- a plain frame, which is
#: the case that matters.
fn packed_run(src: Mat, mask: Mat?, channel: int) -> int {
    if mask is not null { return 0 }
    if src.channels() != 1 or channel != 0 { return 0 }
    if src.step_ != src.cols { return 0 }
    if src.offset % 4 != 0 { return 0 }
    if src.rows * src.cols % 4 != 0 { return 0 }
    return 1
}

#: One channel's partials, queued into `slot` of a shared buffer.
fn reduce_into(
    entry: str,
    src: Mat,
    mask: Mat?,
    channel: int,
    slot: int,
    partials: Buffer
) -> void {
    let mask_mat = if mask is null { src } else { mask }
    kernel_for(entry).dispatch_async([src.storage, mask_mat.storage, partials], GROUPS * GROUP, [src.rows * src.cols, src.cols, src.channels(), src.offset, src.step_, mask_mat.offset, mask_mat.step_, if mask is null {
        0
    } else {
        1
    }, channel, GROUPS * GROUP, packed_run(src, mask, channel), slot, if src.step_ == src.cols * src.channels() {
        1
    } else {
        0
    }], group_size: GROUP)
}

#: Every channel reduced and folded, four numbers each, in one download.
#:
#: The channels go into one buffer and one join, because the download is what
#: costs: an empty dispatch and a wait is 0.14 ms on this machine, so a colour
#: frame reduced a channel at a time paid that three times over and spent more
#: on the round trip than on the picture.
fn reduced(entry: str, join: str, src: Mat, mask: Mat?, channels: list[int]) -> list[float] {
    if src.rows * src.cols == 0 { return [] }
    src.flush()
    if mask is not null { mask.flush() }
    let partials = Buffer(channels.len() * GROUPS * 4)
    for (slot, channel) in channels.enumerate() {
        reduce_into(entry, src, mask, channel, slot, partials)
    }
    let folded = Buffer(channels.len() * 4)
    let lanes = if join == "stats_join" { channels.len() * 4 } else { 1 }
    #: Everything queued, so the only wait is the short download at the end.
    kernel_for(join).dispatch_async([partials, folded], lanes * GROUP, [GROUPS], group_size: GROUP)
    let out = folded.download()
    partials.free()
    folded.free()
    return out
}

#: Sum, absolute sum, sum of squares, and non-zero count of one channel.
pub fn channel_stats(src: Mat, channel: int, mask: Mat? = null) -> tuple[float, float, float, int] {
    let folded = reduced("stats_reduce", "stats_join", src, mask, [channel])
    if folded.len() < 4 { return (0.0, 0.0, 0.0, 0) }
    return (folded[0], folded[1], folded[2], int(folded[3]))
}

#: The same four numbers for every channel at once, in one round trip.
fn every_channel_stats(src: Mat, mask: Mat? = null) -> list[list[float]] {
    var wanted: list[int] = []
    for channel in 0..src.channels() { wanted.push(channel) }
    let folded = reduced("stats_reduce", "stats_join", src, mask, wanted)
    var out: list[list[float]] = []
    for channel in wanted {
        if (channel + 1) * 4 > folded.len() {
            out.push([0.0, 0.0, 0.0, 0.0])
        } else {
            out.push(
                [
                    folded[channel * 4],
                    folded[channel * 4 + 1],
                    folded[channel * 4 + 2],
                    folded[channel * 4 + 3],
                ]
            )
        }
    }
    return out
}

#: Sum of every element, one entry per channel.
pub fn sum_mat(src: Mat, mask: Mat? = null) -> Scalar {
    var out = [0.0, 0.0, 0.0, 0.0]
    for (channel, stats) in every_channel_stats(src, mask: mask).enumerate() {
        if channel < 4 { out[channel] = stats[0] }
    }
    return Scalar(out[0], out[1], out[2], out[3])
}

#: Number of elements that are not zero, summed over channels.
pub fn count_non_zero(src: Mat, mask: Mat? = null) -> int {
    var total = 0.0
    for stats in every_channel_stats(src, mask: mask) { total += stats[3] }
    return int(total)
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
    for (channel, stats) in every_channel_stats(src, mask: mask).enumerate() {
        if channel < 4 { out[channel] = stats[0] / denominator }
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
    for (channel, stats) in every_channel_stats(src, mask: mask).enumerate() {
        if channel >= 4 { break }
        let total = stats[0]
        let squares = stats[2]
        let average = total / denominator
        let variance = max(0.0, squares / denominator - average * average)
        averages[channel] = average
        deviations[channel] = math.sqrt(variance)
    }
    return (
        Scalar(averages[0], averages[1], averages[2], averages[3]),
        Scalar(deviations[0], deviations[1], deviations[2], deviations[3])
    )
}

#: Smallest and largest value and where each one is, over a single channel.
#:
#: Matches OpenCV in taking single-channel input; split a colour image first.
pub fn min_max_loc(src: Mat, mask: Mat? = null) -> tuple[float, float, Point, Point] {
    if src.channels() != 1 {
        throw ValueError("min_max_loc needs a single-channel Mat; split it first")
    }
    let partials = reduced("stats_extrema", "stats_join_extrema", src, mask, [0])
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
        target = unary_op(
            binary_op(src.convert_to(wide), other.convert_to(wide), OP_SUBTRACT),
            UN_ABS
        )
    }
    #: Only the norm that was asked for. Working out all three means a second
    #: reduction over the whole picture for the infinity norm -- and a split
    #: per channel to feed it -- which doubled what an L2 cost.
    if norm_type == NORM_INF {
        var inf = 0.0
        for channel in 0..target.channels() {
            let plane = if target.channels() == 1 { target } else { target.split()[channel] }
            let (low, high) = min_max(plane, mask: mask)
            inf = max(inf, max(math.fabs(low), math.fabs(high)))
        }
        return inf
    }
    var l1 = 0.0
    var l2 = 0.0
    for channel in 0..target.channels() {
        let (_total, absolute, squares, _nonzero) = channel_stats(target, channel, mask: mask)
        l1 += absolute
        l2 += squares
    }
    if norm_type == NORM_L1 { return l1 }
    if norm_type == NORM_L2SQR { return l2 }
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
    let target_type = if dtype >= 0 { CV_MAKETYPE(dtype, cn) } else { CV_MAKETYPE(CV_32F, cn) }
    let out = Mat(out_rows, out_cols, target_type)
    let count = out_rows * out_cols * cn
    if count == 0 { return out }
    kernel_for("stats_reduce_axis").dispatch_async(
        [src.storage, out.storage],
        count,
        [count, src.rows, src.cols, cn, src.offset, src.step_, out.offset, out.step_, dim, op]
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
