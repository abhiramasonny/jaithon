#: Order statistics, histograms, and the helpers for data that arrived as
#: samples rather than as a formula.
#:
#: Three shapes of work live here and they want three different grids.
#:
#: An order statistic is a *sort* plus a couple of gathers. `sorting`'s
#: bitonic network does the expensive half, and the quantile kernel that
#: follows it is one thread per (probability, segment) doing two loads — the
#: grid looks starved and is not, because the elements were all touched by the
#: sort.
#:
#: A covariance is a *GEMM*. Centre the observations once and the whole matrix
#: is `Xc @ Xc.T`; the double loop over pairs of variables that the textbook
#: formula suggests is the wrong shape for this machine by two orders of
#: magnitude, and `matmul` already tiles it.
#:
#: A histogram is a *scatter with collisions*. One thread per sample, an
#: atomic into the bin it lands in, and — when the bins fit — a threadgroup
#: tally merged once at the end, so a million samples contend on 256 slots of
#: threadgroup memory instead of on device memory. That staging is the whole
#: difference: `jaicv`'s `hist_count` found the same thing on pictures.
#:
#: The sampled-data helpers (`diff`, `gradient`, `trapz`, `interp`,
#: `convolve`) are stencils. `diff` and `trapz` are strided views fed to the
#: arithmetic kernels `array` already owns, which is why they compile to no
#: new Metal at all. `gradient` and `convolve` do get kernels, because their
#: neighbouring threads read overlapping windows and a threadgroup tile is
#: what turns three reads per point into one.
#:
#: What is deliberately elsewhere: `mean`, `std`, `var`, `ptp` and the
#: nan-aware reductions are folds and live in `reduce`; `clip` is a ufunc;
#: `unique` and the set operations are in `sorting`; drawing samples is
#: `random`. This module consumes all five.
import std.math
from std.gpu import Buffer
from .array import BOOL, FLOAT32, INT32, INT64, GROUP, NDArray, PRELUDE
from .array import OP_ADD, OP_DIV, OP_MUL, OP_SUB
from .array import binary_arith, contiguous_strides, copy_into, element_count
from .array import f32_bits, fill, gather_flat, kernel_of, normalize_axis
from .array import promote, require_real
from .create import array, linspace, zeros
from .manipulate import concatenate, diff, flip, reshape
from .matmul import matmul
from .reduce import all, amax, amin, count_nonzero, cumsum, mean, sum
from .sorting import argsort, searchsorted, sort
from .ufunc import clip, isnan, remainder, sqrt, subtract

#: Quantile rule: the sample is a piecewise-linear function of its ranks.
#: numpy's default, and the one `median` is defined by.
const METHOD_LINEAR = 0
#: Quantile rule: the order statistic below the virtual index.
const METHOD_LOWER = 1
#: Quantile rule: the order statistic above it.
const METHOD_HIGHER = 2
#: Quantile rule: the average of those two.
const METHOD_MIDPOINT = 3
#: Quantile rule: the nearer of those two, ties to the even index — which is
#: what `rint` does and what numpy's `around` does.
const METHOD_NEAREST = 4
#: Quantile rule: Hyndman and Fan's plotting position 5, alpha = beta = 1/2.
const METHOD_HAZEN = 5
#: Quantile rule: Hyndman and Fan's 6, alpha = beta = 0.
const METHOD_WEIBULL = 6
#: Quantile rule: Hyndman and Fan's 8, alpha = beta = 1/3. Median-unbiased
#: whatever the distribution.
const METHOD_MEDIAN_UNBIASED = 7
#: Quantile rule: Hyndman and Fan's 9, alpha = beta = 3/8. Unbiased for the
#: order statistics of a normal sample.
const METHOD_NORMAL_UNBIASED = 8

#: Elements one `convolve` threadgroup can stage, halo included. 512 floats is
#: 2 KB of threadgroup memory and leaves room for a 256-thread group plus a
#: kernel of 257 taps, which covers every filter anyone writes by hand.
pub const CONV_TILE = 512

#: Bins a histogram can tally in threadgroup memory before it has to contend
#: on device atomics. 1024 is what `jaicv`'s `hist_count` settled on for the
#: same reason: past it the zero-and-merge sweeps cost more than they save.
pub const STAGED_BINS = 1024

let SOURCE = PRELUDE + """
constexpr constant uint NST_TILE = 512u;
constexpr constant uint NST_STAGED_BINS = 1024u;

constexpr constant uint NST_LINEAR = 0u;
constexpr constant uint NST_LOWER = 1u;
constexpr constant uint NST_HIGHER = 2u;
constexpr constant uint NST_MIDPOINT = 3u;
constexpr constant uint NST_NEAREST = 4u;
constexpr constant uint NST_HAZEN = 5u;
constexpr constant uint NST_WEIBULL = 6u;
constexpr constant uint NST_MEDIAN_UNBIASED = 7u;
constexpr constant uint NST_NORMAL_UNBIASED = 8u;

inline float nst_nan() { return as_type<float>(0x7fc00000u); }

/* The plotting position (alpha, beta) that separates the continuous quantile
 * rules from one another. Every one of them is
 *
 *     virtual = alpha + q * (m - alpha - beta + 1) - 1
 *
 * so the rules cost a branch here and share the interpolation below. The four
 * discontinuous ones take (1, 1) and then round the result their own way,
 * which is also how numpy defines them. */
inline float2 nst_plotting(uint method) {
    if (method == NST_HAZEN) return float2(0.5f, 0.5f);
    if (method == NST_WEIBULL) return float2(0.0f, 0.0f);
    if (method == NST_MEDIAN_UNBIASED) return float2(1.0f / 3.0f, 1.0f / 3.0f);
    if (method == NST_NORMAL_UNBIASED) return float2(0.375f, 0.375f);
    return float2(1.0f, 1.0f);
}

/* One quantile of one already-sorted segment.
 *
 * `valid[seg]` is how many of the segment's `n` slots are not NaN. The sort
 * put the NaNs last, so the live values are the prefix and the nan-aware
 * family needs no second arrangement -- it just reads a shorter row. With
 * `skip_nan` off, a segment that lost anything answers NaN, which is what
 * numpy's plain `median` does.
 *
 * A thread per output looks like the starved grid this package warns about,
 * and is not: the sort ahead of it already visited every element, and this
 * pass is two loads deep. */
kernel void nst_quantile(device float* dst [[buffer(0)]],
                         device const float* sorted [[buffer(1)]],
                         device const float* probabilities [[buffer(2)]],
                         device const float* valid [[buffer(3)]],
                         constant uint& outputs [[buffer(4)]],
                         constant uint& segments [[buffer(5)]],
                         constant uint& n [[buffer(6)]],
                         constant uint& method [[buffer(7)]],
                         constant uint& skip_nan [[buffer(8)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= outputs) return;
    uint which = id / segments;
    uint seg = id - which * segments;
    uint m = uint(valid[seg]);
    if (m == 0u || (skip_nan == 0u && m < n)) {
        dst[id] = nst_nan();
        return;
    }
    float q = probabilities[which];
    float2 ab = nst_plotting(method);
    float top = float(m) - 1.0f;
    float virt = clamp(ab.x + q * (float(m) - ab.x - ab.y + 1.0f) - 1.0f, 0.0f, top);

    float below = floor(virt);
    float above = min(below + 1.0f, top);
    float weight = virt - below;
    if (method == NST_LOWER) { above = below; weight = 0.0f; }
    if (method == NST_HIGHER) { below = ceil(virt); above = below; weight = 0.0f; }
    if (method == NST_NEAREST) { below = rint(virt); above = below; weight = 0.0f; }
    if (method == NST_MIDPOINT) { above = ceil(virt); weight = 0.5f; }

    device const float* row = sorted + seg * n;
    float low = row[uint(below)];
    dst[id] = low + weight * (row[uint(above)] - low);
}

/* Add into a bin nobody owns exclusively.
 *
 * Relaxed ordering: the bins are independent counters and no thread reads
 * another's, so there is nothing for an acquire to make visible. */
inline void nst_bump(device float* bins, uint at, float amount) {
    atomic_fetch_add_explicit((device atomic<float>*)(bins + at), amount, memory_order_relaxed);
}

/* Which bin one value falls in along one dimension, or -1 for outside.
 *
 * Even edges are a multiply and a truncate. That matters more than it looks:
 * the general path is a bisection, twelve dependent loads deep at 4096 bins,
 * and the device has no integer divide to shorten it. The top edge is closed
 * and every other is half-open, which is numpy's rule -- hence the explicit
 * equality rather than letting the multiply round a boundary value into a bin
 * that does not exist.
 *
 * A NaN fails `value >= low` and lands outside, counted nowhere. */
inline int nst_bin_of(
    device const float* edges,
    uint base,
    uint bins,
    uint even,
    float value
) {
    float low = edges[base];
    float high = edges[base + bins];
    if (!(value >= low) || value > high) return -1;
    if (value == high) return int(bins) - 1;
    if (even != 0u) {
        int at = int((value - low) * float(bins) / (high - low));
        return clamp(at, 0, int(bins) - 1);
    }
    uint lo = 0u;
    uint hi = bins;
    while (hi - lo > 1u) {
        uint mid = (lo + hi) >> 1;
        if (value < edges[base + mid]) hi = mid; else lo = mid;
    }
    return int(lo);
}

/* One sample's flat bin over every dimension, row-major so that dimension
 * zero is the slowest -- the layout numpy's `histogramdd` returns.
 *
 * `meta` is [dims, bins per dim, edge base per dim, evenness per dim]. The
 * sample block is dense [rows, dims], materialised by the caller, so the
 * position is a multiply and not a strided walk. */
inline int nst_flat_bin(
    device const float* sample,
    device const float* edges,
    device const float* meta,
    uint dims,
    uint row
) {
    int flat = 0;
    for (uint d = 0u; d < dims; d++) {
        uint bins = uint(meta[1u + d]);
        uint base = uint(meta[1u + dims + d]);
        uint even = uint(meta[1u + 2u * dims + d]);
        int at = nst_bin_of(edges, base, bins, even, sample[row * dims + d]);
        if (at < 0) return -1;
        flat = flat * int(bins) + at;
    }
    return flat;
}

/* Straight to device atomics. For a histogram too wide to stage, and for
 * every weighted one -- a threadgroup `atomic<float>` is not a thing this
 * repo relies on anywhere, and the merge below only totals integers. */
kernel void nst_hist(device float* counts [[buffer(0)]],
                     device const float* sample [[buffer(1)]],
                     device const float* edges [[buffer(2)]],
                     device const float* weights [[buffer(3)]],
                     device const float* meta [[buffer(4)]],
                     constant uint& rows [[buffer(5)]],
                     constant uint& dims [[buffer(6)]],
                     constant uint& use_weights [[buffer(7)]],
                     uint id [[thread_position_in_grid]]) {
    if (id >= rows) return;
    int flat = nst_flat_bin(sample, edges, meta, dims, id);
    if (flat < 0) return;
    nst_bump(counts, uint(flat), use_weights != 0u ? weights[id] : 1.0f);
}

/* Tallied in threadgroup memory and merged once.
 *
 * A grid-stride loop over a fixed grid rather than a thread per sample,
 * because every thread has to reach both barriers and an early return would
 * hang the group. */
kernel void nst_hist_staged(device float* counts [[buffer(0)]],
                            device const float* sample [[buffer(1)]],
                            device const float* edges [[buffer(2)]],
                            device const float* meta [[buffer(3)]],
                            constant uint& rows [[buffer(4)]],
                            constant uint& dims [[buffer(5)]],
                            constant uint& bins [[buffer(6)]],
                            constant uint& stride [[buffer(7)]],
                            uint id [[thread_position_in_grid]],
                            uint lid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint tally[NST_STAGED_BINS];
    for (uint slot = lid; slot < bins; slot += ND_GROUP) {
        atomic_store_explicit(&tally[slot], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = id; row < rows; row += stride) {
        int flat = nst_flat_bin(sample, edges, meta, dims, row);
        if (flat < 0) continue;
        atomic_fetch_add_explicit(&tally[uint(flat)], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint slot = lid; slot < bins; slot += ND_GROUP) {
        uint mine = atomic_load_explicit(&tally[slot], memory_order_relaxed);
        if (mine != 0u) nst_bump(counts, slot, float(mine));
    }
}

/* `bincount`: the bin *is* the value, so there is no edge search at all.
 *
 * `quad` says the samples are a multiple of four long and sixteen-byte
 * aligned, which lets one thread pull four of them in a single load and cut
 * the launched grid to a quarter. The atomics per sample are unchanged --
 * they are the floor of this operation -- but the loads are not, and on a
 * ten-million-element count that is the difference the vector path buys. */
kernel void nst_bincount(device float* counts [[buffer(0)]],
                         device const float* values [[buffer(1)]],
                         device const float* weights [[buffer(2)]],
                         constant uint& count [[buffer(3)]],
                         constant uint& bins [[buffer(4)]],
                         constant uint& use_weights [[buffer(5)]],
                         constant uint& quad [[buffer(6)]],
                         uint id [[thread_position_in_grid]]) {
    if (quad != 0u) {
        if (id * 4u >= count) return;
        float4 four = ((device const float4*)values)[id];
        float4 mass = use_weights != 0u
            ? ((device const float4*)weights)[id]
            : float4(1.0f);
        for (uint k = 0u; k < 4u; k++) {
            uint at = uint(four[k]);
            if (at < bins) nst_bump(counts, at, mass[k]);
        }
        return;
    }
    if (id >= count) return;
    uint at = uint(values[id]);
    if (at < bins) nst_bump(counts, at, use_weights != 0u ? weights[id] : 1.0f);
}

kernel void nst_bincount_staged(device float* counts [[buffer(0)]],
                                device const float* values [[buffer(1)]],
                                constant uint& count [[buffer(2)]],
                                constant uint& bins [[buffer(3)]],
                                constant uint& stride [[buffer(4)]],
                                uint id [[thread_position_in_grid]],
                                uint lid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint tally[NST_STAGED_BINS];
    for (uint slot = lid; slot < bins; slot += ND_GROUP) {
        atomic_store_explicit(&tally[slot], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = id; row < count; row += stride) {
        uint at = uint(values[row]);
        if (at < bins) atomic_fetch_add_explicit(&tally[at], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint slot = lid; slot < bins; slot += ND_GROUP) {
        uint mine = atomic_load_explicit(&tally[slot], memory_order_relaxed);
        if (mine != 0u) nst_bump(counts, slot, float(mine));
    }
}

/* The gradient along one axis of a dense [segments, n] block.
 *
 * The interior is the three-point formula for uneven spacing,
 *
 *     (hs^2 f[i+1] + (hd^2 - hs^2) f[i] - hd^2 f[i-1]) / (hs hd (hs + hd))
 *
 * which collapses to the familiar (f[i+1] - f[i-1]) / 2h when hs == hd and is
 * second order either way. The two ends are one-sided: a plain forward or
 * backward difference at `edge_order` 1, and the three-point one-sided
 * formula at 2.
 *
 * A group owns 256 consecutive points of one segment and stages them with a
 * halo of one on each side. Every interior thread reads i-1, i and i+1, so
 * the tile turns three device reads per point into one. The two edge threads
 * of the whole segment read from device instead -- there are two of them and
 * the halo would have to grow to three to hold what they want. */
kernel void nst_gradient(device float* dst [[buffer(0)]],
                         device const float* src [[buffer(1)]],
                         device const float* coords [[buffer(2)]],
                         constant uint& segments [[buffer(3)]],
                         constant uint& n [[buffer(4)]],
                         constant uint& chunks [[buffer(5)]],
                         constant uint& even [[buffer(6)]],
                         constant uint& step_bits [[buffer(7)]],
                         constant uint& edge_order [[buffer(8)]],
                         uint group [[threadgroup_position_in_grid]],
                         uint lid [[thread_index_in_threadgroup]]) {
    threadgroup float tile[ND_GROUP + 2u];
    uint seg = group / chunks;
    uint chunk = group - seg * chunks;
    uint start = chunk * ND_GROUP;
    uint base = seg * n;
    uint i = start + lid;
    bool live = seg < segments;

    tile[lid + 1u] = (live && i < n) ? src[base + i] : 0.0f;
    if (lid == 0u) {
        tile[0] = (live && start > 0u) ? src[base + start - 1u] : 0.0f;
        uint after = start + ND_GROUP;
        tile[ND_GROUP + 1u] = (live && after < n) ? src[base + after] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!live || i >= n) return;

    float step = as_type<float>(step_bits);
    if (i > 0u && i + 1u < n) {
        float hs = even != 0u ? step : coords[i] - coords[i - 1u];
        float hd = even != 0u ? step : coords[i + 1u] - coords[i];
        float back = tile[lid];
        float here = tile[lid + 1u];
        float ahead = tile[lid + 2u];
        dst[base + i] = (hs * hs * ahead + (hd * hd - hs * hs) * here - hd * hd * back)
            / (hs * hd * (hs + hd));
        return;
    }
    if (edge_order == 1u || n < 3u) {
        if (i == 0u) {
            float h = even != 0u ? step : coords[1] - coords[0];
            dst[base] = (src[base + 1u] - src[base]) / h;
        } else {
            float h = even != 0u ? step : coords[n - 1u] - coords[n - 2u];
            dst[base + i] = (src[base + n - 1u] - src[base + n - 2u]) / h;
        }
        return;
    }
    if (i == 0u) {
        float h0 = even != 0u ? step : coords[1] - coords[0];
        float h1 = even != 0u ? step : coords[2] - coords[1];
        dst[base] = -(2.0f * h0 + h1) / (h0 * (h0 + h1)) * src[base]
            + (h0 + h1) / (h0 * h1) * src[base + 1u]
            - h0 / (h1 * (h0 + h1)) * src[base + 2u];
        return;
    }
    float h1 = even != 0u ? step : coords[n - 2u] - coords[n - 3u];
    float h0 = even != 0u ? step : coords[n - 1u] - coords[n - 2u];
    dst[base + i] = h0 / (h1 * (h1 + h0)) * src[base + n - 3u]
        - (h1 + h0) / (h1 * h0) * src[base + n - 2u]
        + (2.0f * h0 + h1) / (h0 * (h1 + h0)) * src[base + n - 1u];
}

/* One query to a thread: bisect the abscissae, then one lerp.
 *
 * `left_mode` and `right_mode` of zero mean "hold the end value", which spares
 * the host a download of `fp`'s first and last elements just to pass them back
 * down as constants. */
kernel void nst_interp(device float* dst [[buffer(0)]],
                       device const float* x [[buffer(1)]],
                       device const float* xp [[buffer(2)]],
                       device const float* fp [[buffer(3)]],
                       constant uint& count [[buffer(4)]],
                       constant uint& samples [[buffer(5)]],
                       constant uint& left_mode [[buffer(6)]],
                       constant uint& left_bits [[buffer(7)]],
                       constant uint& right_mode [[buffer(8)]],
                       constant uint& right_bits [[buffer(9)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float value = x[id];
    if (isnan(value)) {
        dst[id] = value;
        return;
    }
    uint last = samples - 1u;
    if (value < xp[0]) {
        dst[id] = left_mode != 0u ? as_type<float>(left_bits) : fp[0];
        return;
    }
    if (value > xp[last]) {
        dst[id] = right_mode != 0u ? as_type<float>(right_bits) : fp[last];
        return;
    }
    uint lo = 0u;
    uint hi = last;
    while (hi - lo > 1u) {
        uint mid = (lo + hi) >> 1;
        if (value < xp[mid]) hi = mid; else lo = mid;
    }
    float span = xp[hi] - xp[lo];
    /* Repeated abscissae are legal input and would divide by zero here.
     * Holding the left value is what a zero-width interval means and is what
     * numpy answers. */
    float t = span != 0.0f ? (value - xp[lo]) / span : 0.0f;
    dst[id] = fp[lo] + t * (fp[hi] - fp[lo]);
}

/* out[o] = sum_k a[o - pad + k] * taps[k], zero outside `a`.
 *
 * Convolution and correlation are the same kernel: the caller reverses the
 * taps for one of them and picks `pad` from the mode.
 *
 * Neighbouring outputs read windows that overlap in all but one element, so
 * the group stages its whole span -- 256 outputs need 255 + m values -- and
 * the inner loop reads threadgroup memory. Taps come from device and are the
 * same address for every thread in the group, which is a broadcast; when
 * there are a multiple of four of them they come four at a time, the tap
 * buffer being a fresh contiguous copy and therefore aligned. */
kernel void nst_correlate(device float* dst [[buffer(0)]],
                          device const float* a [[buffer(1)]],
                          device const float* taps [[buffer(2)]],
                          constant uint& outputs [[buffer(3)]],
                          constant uint& n [[buffer(4)]],
                          constant uint& m [[buffer(5)]],
                          constant uint& pad [[buffer(6)]],
                          constant uint& staged [[buffer(7)]],
                          uint group [[threadgroup_position_in_grid]],
                          uint lid [[thread_index_in_threadgroup]]) {
    threadgroup float tile[NST_TILE];
    uint o = group * ND_GROUP + lid;
    if (staged != 0u) {
        int origin = int(group * ND_GROUP) - int(pad);
        uint span = ND_GROUP + m - 1u;
        for (uint k = lid; k < span; k += ND_GROUP) {
            int at = origin + int(k);
            tile[k] = (at >= 0 && at < int(n)) ? a[at] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (o >= outputs) return;

    float total = 0.0f;
    if (staged == 0u) {
        int origin = int(o) - int(pad);
        for (uint k = 0u; k < m; k++) {
            int at = origin + int(k);
            if (at >= 0 && at < int(n)) total += a[at] * taps[k];
        }
        dst[o] = total;
        return;
    }
    uint k = 0u;
    if ((m & 3u) == 0u) {
        uint quads = m >> 2;
        for (uint q = 0u; q < quads; q++) {
            float4 four = ((device const float4*)taps)[q];
            uint at = lid + q * 4u;
            total += tile[at] * four.x + tile[at + 1u] * four.y
                + tile[at + 2u] * four.z + tile[at + 3u] * four.w;
        }
        k = m;
    }
    for (; k < m; k++) total += tile[lid + k] * taps[k];
    dst[o] = total;
}
"""

fn _number(value: any) -> float {
    if value is float { return value }
    if value is int { return float(value) }
    if value is bool { return if value { 1.0 } else { 0.0 } }
    throw TypeError(f"expected a number, got {type_of(value)}")
}

fn _index(value: any) -> int {
    if value is int { return value }
    throw TypeError(f"expected an int, got {type_of(value)}")
}

#: A list or a tuple as a list. The VM hands a comma-separated argument over
#: as a tuple, and every spec here accepts either spelling.
fn _as_list(value: any, what: str) -> list[any] {
    if value is list { return [item for item in value] }
    if value is tuple { return [item for item in value] }
    throw TypeError(f"{what} takes a list or a tuple, not {type_of(value)}")
}

fn _pair(value: any, what: str) -> (float, float) {
    let items = _as_list(value, what)
    if items.len() != 2 { throw ValueError(f"{what} takes a low and a high, not {items.len()}") }
    return (_number(items[0]), _number(items[1]))
}

#: `a` as a dense 1-d array starting at slot zero, and whether that cost an
#: allocation the caller now owns.
fn _flat_1d(a: NDArray) -> (NDArray, bool) {
    if a.ndim() == 1 and a.offset == 0 and a.is_contiguous() { return (a, false) }
    let out = NDArray([a.size()], a.dtype)
    copy_into(out.view(a.shape, contiguous_strides(a.shape, a.lanes()), 0), a)
    return (out, true)
}

#: `value` as an owned 1-d array, whatever spelling it arrived in.
fn _as_1d(value: any, what: str) -> NDArray {
    if value is NDArray {
        let out = NDArray([value.size()], value.dtype)
        copy_into(out.view(value.shape, contiguous_strides(value.shape, value.lanes()), 0), value)
        return out
    }
    if value is list or value is tuple {
        return array([_number(item) for item in _as_list(value, what)])
    }
    let one = NDArray([1], FLOAT32)
    one.upload([_number(value)])
    return one
}

#: A view of `a` with `axis` innermost, every other axis keeping its order.
fn _axis_last(a: NDArray, axis: int) -> NDArray {
    if axis == a.ndim() - 1 { return a }
    var shape: list[int] = []
    var strides: list[int] = []
    for at in 0..a.ndim() {
        if at == axis { continue }
        shape.push(a.shape[at])
        strides.push(a.strides[at])
    }
    shape.push(a.shape[axis])
    strides.push(a.strides[axis])
    return a.view(shape, strides, a.offset)
}

#: `a` copied into a dense `[segments, n]` block with `axis` innermost.
#:
#: Every kernel below that walks an axis takes this shape, so the strided read
#: happens once in the copy rather than once per neighbour in the stencil.
fn _rows_of(a: NDArray, at: int) -> NDArray {
    let n = a.shape[at]
    let moved = _axis_last(a, at)
    let rows = NDArray([a.size() // n, n], FLOAT32)
    copy_into(rows.view(moved.shape, contiguous_strides(moved.shape, 1), 0), moved)
    return rows
}

#: The inverse: a dense `[segments, n]` block written back through `axis`.
fn _rows_back(rows: NDArray, like: NDArray, at: int) -> NDArray {
    let out = NDArray(like.shape, FLOAT32)
    let moved = _axis_last(out, at)
    copy_into(moved, rows.view(moved.shape, contiguous_strides(moved.shape, 1), 0))
    return out
}

#: A view of one axis narrowed to `count` elements from `first`.
fn _slice_axis(a: NDArray, axis: int, first: int, count: int) -> NDArray {
    var shape = [extent for extent in a.shape]
    shape[axis] = count
    return a.view(shape, [stride for stride in a.strides], a.offset + first * a.strides[axis])
}

#: The shape left when `at` is reduced away.
fn _dropped_shape(shape: list[int], at: int, keepdims: bool) -> list[int] {
    var out: list[int] = []
    for axis in 0..shape.len() {
        if axis != at {
            out.push(shape[axis])
        } elif keepdims {
            out.push(1)
        }
    }
    return out
}

#: The axes a call names: `null` is every one, an int is one, a list or a
#: tuple is several.
fn _axes_of(axis: any, ndim: int) -> list[int] {
    if axis is null { return [at for at in 0..ndim] }
    if axis is int { return [normalize_axis(axis, ndim)] }
    var out: list[int] = []
    for item in _as_list(axis, "axis") { out.push(normalize_axis(_index(item), ndim)) }
    return out
}

#: The dtype a two-operand sampled-data pass produces. `BOOL` has no reading
#: as a running total, so it widens the way arithmetic on it does.
fn _pair_dtype(a: int, b: int) -> int {
    let out = promote(a, b)
    return if out == BOOL { INT32 } else { out }
}

#: The kernel selector one of numpy's method names asks for.
#:
#: Nine of numpy's thirteen. The four missing ones -- `inverted_cdf`,
#: `averaged_inverted_cdf`, `closest_observation` and
#: `interpolated_inverted_cdf` -- all branch on whether the fractional part of
#: the virtual index is *exactly* zero, and that test is not decidable here:
#: the index is `q * m` in float32 and `0.7 * 10` is not 7. Answering with a
#: rule that silently disagrees with numpy at every tenth probability is worse
#: than saying the name is not supported.
fn _method_code(name: str) -> int {
    if name == "linear" { return METHOD_LINEAR }
    if name == "lower" { return METHOD_LOWER }
    if name == "higher" { return METHOD_HIGHER }
    if name == "midpoint" { return METHOD_MIDPOINT }
    if name == "nearest" { return METHOD_NEAREST }
    if name == "hazen" { return METHOD_HAZEN }
    if name == "weibull" { return METHOD_WEIBULL }
    if name == "median_unbiased" { return METHOD_MEDIAN_UNBIASED }
    if name == "normal_unbiased" { return METHOD_NORMAL_UNBIASED }
    throw ValueError(
        f"no quantile method named {name}; linear, lower, higher, midpoint, nearest, " +
            "hazen, weibull, median_unbiased and normal_unbiased are the ones here"
    )
}

#: The probabilities as a device array in `[0, 1]`, and whether the caller
#: asked for one rather than a list of them — which decides whether the result
#: grows a leading axis.
fn _probabilities(q: any, scale: float, what: str) -> (NDArray, bool) {
    if q is NDArray {
        if q.ndim() > 1 { throw ValueError(f"{what} takes a 0-d or 1-d q, not {q.ndim()}") }
        let scaled = binary_arith(q, 1.0 / scale, OP_MUL)
        let lowest = amin(scaled).item()
        let highest = amax(scaled).item()
        if lowest < 0.0 or highest > 1.0 {
            scaled.free()
            throw ValueError(f"{what} needs every q in [0, {scale}]")
        }
        return (scaled, q.ndim() == 0)
    }
    var values: list[float] = []
    var single = false
    if q is list or q is tuple {
        for item in _as_list(q, what) { values.push(_number(item) / scale) }
    } else {
        values.push(_number(q) / scale)
        single = true
    }
    for p in values {
        if not (p >= 0.0 and p <= 1.0) {
            throw ValueError(f"{what} needs every q in [0, {scale}], got {p * scale}")
        }
    }
    let out = NDArray([max(1, values.len())], FLOAT32)
    if values.len() == 0 { return (out.view([0], [1], 0), false) }
    out.upload(values)
    return (out, single)
}

#: Sort the axis, count what is not NaN, then one thread per answer.
#:
#: The sort is the expensive half and `sorting` owns it; NaNs land at the end
#: of every row, which is what lets the nan-aware family read a shorter prefix
#: of the same buffer instead of compacting first.
fn _quantile_core(
    a: NDArray,
    q: any,
    axis: any,
    method: str,
    keepdims: bool,
    skip_nan: bool,
    scale: float,
    what: str
) -> NDArray {
    require_real(a, what)
    let code = _method_code(method)
    let (probabilities, single) = _probabilities(q, scale, what)
    defer { probabilities.free() }

    let over_all = axis is null
    let source = if over_all { _flat_1d(a)[0] } else { a }
    let owns_source = over_all and not source == a
    defer { if owns_source { source.free() } }
    if source.ndim() == 0 { throw ValueError(f"{what} needs an axis and a 0-d array has none") }
    let at = if over_all { 0 } else { normalize_axis(_index(axis), source.ndim()) }
    let n = source.shape[at]
    if n == 0 { throw ValueError(f"{what} of a zero-size axis has no answer") }

    let rows = _rows_of(source, at)
    defer { rows.free() }
    let segments = rows.shape[0]
    let ordered = sort(rows, 1)
    defer { ordered.free() }

    let flags = isnan(rows)
    let missing = sum(flags, 1)
    flags.free()
    let live = subtract(float(n), missing)
    missing.free()
    defer { live.free() }

    var shape: list[int] = []
    if not single { shape.push(probabilities.size()) }
    if over_all {
        if keepdims {
            for _axis in 0..a.ndim() { shape.push(1) }
        }
    } else {
        for extent in _dropped_shape(source.shape, at, keepdims) { shape.push(extent) }
    }
    let out = NDArray(shape, FLOAT32)
    let outputs = out.size()
    if outputs == 0 { return out }
    kernel_of(SOURCE, "nst_quantile").dispatch_async([out.storage, ordered.storage, probabilities.storage, live.storage], outputs, [outputs, segments, n, code, if skip_nan {
        1
    } else {
        0
    }])
    return out
}

#: The `q`-th quantile of `a` along `axis`, with `q` in `[0, 1]`.
#:
#: `q` is a number, a list of them, or a 1-d array, and a list or array grows
#: a leading axis on the result exactly as numpy's does. See `_method_code`
#: for the nine `method` names and for the four that are deliberately absent.
#:
#: A NaN anywhere in a segment makes that segment's answer NaN;
#: `nanquantile` is the one that steps over them.
pub fn quantile(
    a: NDArray,
    q: any,
    axis: any = null,
    method: str = "linear",
    keepdims: bool = false
) -> NDArray {
    return _quantile_core(a, q, axis, method, keepdims, false, 1.0, "quantile")
}

#: `quantile` with the NaNs dropped from each segment. A segment holding
#: nothing else answers NaN.
pub fn nanquantile(
    a: NDArray,
    q: any,
    axis: any = null,
    method: str = "linear",
    keepdims: bool = false
) -> NDArray {
    return _quantile_core(a, q, axis, method, keepdims, true, 1.0, "nanquantile")
}

#: `quantile` with `q` in `[0, 100]` instead of `[0, 1]`.
pub fn percentile(
    a: NDArray,
    q: any,
    axis: any = null,
    method: str = "linear",
    keepdims: bool = false
) -> NDArray {
    return _quantile_core(a, q, axis, method, keepdims, false, 100.0, "percentile")
}

#: `percentile` with the NaNs dropped.
pub fn nanpercentile(
    a: NDArray,
    q: any,
    axis: any = null,
    method: str = "linear",
    keepdims: bool = false
) -> NDArray {
    return _quantile_core(a, q, axis, method, keepdims, true, 100.0, "nanpercentile")
}

#: The middle value along `axis`, averaging the two middles of an even-length
#: run. That is the linear quantile at one half, which is how numpy defines it
#: and why this is a delegation rather than its own pass.
pub fn median(a: NDArray, axis: any = null, keepdims: bool = false) -> NDArray {
    return _quantile_core(a, 0.5, axis, "linear", keepdims, false, 1.0, "median")
}

#: `median` with the NaNs dropped.
pub fn nanmedian(a: NDArray, axis: any = null, keepdims: bool = false) -> NDArray {
    return _quantile_core(a, 0.5, axis, "linear", keepdims, true, 1.0, "nanmedian")
}

#: `weights` shaped so it broadcasts against `a`, and whether that cost a view.
fn _weight_view(a: NDArray, axis: any, weights: NDArray) -> (NDArray, bool) {
    if weights.shape == a.shape { return (weights, false) }
    if weights.ndim() != 1 or axis is null {
        throw ValueError(
            f"weights of {weights.shape} against {a.shape} need a named axis and one dimension"
        )
    }
    let at = normalize_axis(_index(axis), a.ndim())
    if weights.size() != a.shape[at] {
        throw ValueError(f"{weights.size()} weights for an axis of {a.shape[at]}")
    }
    var shape = [1 for _axis in 0..a.ndim()]
    shape[at] = weights.size()
    var strides = [0 for _axis in 0..a.ndim()]
    strides[at] = weights.strides[0]
    return (weights.view(shape, strides, weights.offset), true)
}

#: The weighted mean along `axis`, or the plain one when no weights are given.
#:
#: numpy's `returned=True` is spelled by summing the weights yourself; a flag
#: that changed the return arity would make every other caller unpack a shape
#: it did not ask for.
pub fn average(
    a: NDArray,
    axis: any = null,
    weights: NDArray? = null,
    keepdims: bool = false
) -> NDArray {
    require_real(a, "average")
    if weights is null { return mean(a, axis, keepdims) }
    let (shaped, owned) = _weight_view(a, axis, weights)
    defer { if owned { shaped.free() } }
    let spread = shaped.broadcast_to(a.shape)
    let scaled = binary_arith(a, spread, OP_MUL)
    defer { scaled.free() }
    let top = sum(scaled, axis, keepdims)
    defer { top.free() }
    let bottom = sum(spread, axis, keepdims)
    defer { bottom.free() }
    return binary_arith(top, bottom, OP_DIV)
}

#: One operand of `cov` as a `[variables, observations]` view.
#:
#: A 1-d input is one variable whatever `rowvar` says, which is numpy's rule
#: and the reason `cov` of a vector is a single number.
fn _as_variables(m: NDArray, rowvar: bool, what: str) -> NDArray {
    require_real(m, what)
    if m.ndim() == 0 { throw ValueError(f"{what} needs at least one dimension") }
    if m.ndim() > 2 { throw ValueError(f"{what} takes a 1-d or 2-d array, not {m.ndim()}") }
    if m.ndim() == 1 { return m.view([1, m.size()], [m.strides[0], m.strides[0]], m.offset) }
    if rowvar { return m }
    return m.view([m.shape[1], m.shape[0]], [m.strides[1], m.strides[0]], m.offset)
}

#: The covariance matrix of the variables in `m`, and in `y` if a second set
#: is given.
#:
#: `Xc @ Xc.T` over the centred observations, divided once. The double loop
#: over pairs of variables that the definition suggests would be `variables^2`
#: separate reductions; this is one GEMM, which is the shape the device is
#: built for and the reason `matmul` is a dependency of this module.
#:
#: `rowvar` false reads the columns as variables. `ddof` of -1 means "unset",
#: which is 0 when `bias` is set and 1 otherwise — numpy's default of the
#: sample covariance.
pub fn cov(
    m: NDArray,
    y: NDArray? = null,
    rowvar: bool = true,
    bias: bool = false,
    ddof: int = -1
) -> NDArray {
    var parts = [_as_variables(m, rowvar, "cov")]
    if y is not null { parts.push(_as_variables(y, rowvar, "cov")) }
    if parts.len() == 2 and parts[0].shape[1] != parts[1].shape[1] {
        throw ValueError(f"{parts[0].shape[1]} observations against {parts[1].shape[1]}")
    }
    let joined = if parts.len() == 1 { parts[0] } else { concatenate(parts, 0) }
    let variables = joined.shape[0]
    let observations = joined.shape[1]
    let correction = if ddof >= 0 { ddof } else { if bias { 0 } else { 1 } }
    if observations - correction <= 0 {
        throw ValueError(
            f"{observations} observations leave no degrees of freedom at ddof {correction}"
        )
    }

    let centre = mean(joined, 1, true)
    let centred = binary_arith(joined, centre, OP_SUB)
    centre.free()
    if parts.len() == 2 { joined.free() }
    defer { centred.free() }

    let transposed = centred.view(
        [observations, variables],
        [centred.strides[1], centred.strides[0]],
        centred.offset
    )
    let gram = matmul(centred, transposed)
    let scaled = binary_arith(gram, float(observations - correction), OP_DIV)
    gram.free()
    #: numpy squeezes a one-variable covariance to a bare number, and code
    #: written against it indexes the result accordingly.
    if variables != 1 { return scaled }
    let one = scaled.view([], [], 0).copy()
    scaled.free()
    return one
}

#: The Pearson correlation matrix: `cov` normalised by the standard deviations
#: on its diagonal.
#:
#: Clamped to `[-1, 1]` at the end. The division is exact in exact arithmetic
#: and is not in float32, and a correlation of 1.0000001 breaks every caller
#: that feeds it to `arccos`.
pub fn corrcoef(x: NDArray, y: NDArray? = null, rowvar: bool = true) -> NDArray {
    let c = cov(x, y, rowvar)
    if c.ndim() == 0 {
        c.free()
        let one = NDArray([], FLOAT32)
        fill(one, 1.0)
        return one
    }
    defer { c.free() }
    let variables = c.shape[0]
    let sigma = sqrt(c.view([variables], [c.strides[0] + c.strides[1]], c.offset))
    defer { sigma.free() }
    let down = binary_arith(c, sigma.view([variables, 1], [sigma.strides[0], 0], 0), OP_DIV)
    let across = binary_arith(down, sigma.view([1, variables], [0, sigma.strides[0]], 0), OP_DIV)
    down.free()
    let bounded = clip(across, -1.0, 1.0)
    across.free()
    return bounded
}

#: One column of a dense `[rows, dims]` block, as a view.
fn _column(block: NDArray, at: int) -> NDArray {
    return block.view([block.shape[0]], [block.shape[1]], at)
}

#: The samples as a dense `[rows, dims]` block of float32 at slot zero, which
#: is the only layout the binning kernels read.
fn _sample_block(sample: NDArray, what: str) -> NDArray {
    if sample.ndim() > 2 { throw ValueError(f"{what} takes a 1-d or 2-d sample") }
    let rows = if sample.ndim() == 0 { 1 } else { sample.shape[0] }
    let dims = if sample.ndim() == 2 { sample.shape[1] } else { 1 }
    var strides = [1, 1]
    var offset = sample.offset
    if sample.ndim() >= 1 { strides[0] = sample.strides[0] }
    if sample.ndim() == 2 { strides[1] = sample.strides[1] }
    if sample.ndim() == 0 { offset = sample.offset }
    let block = NDArray([rows, dims], FLOAT32)
    copy_into(block, sample.view([rows, dims], strides, offset))
    return block
}

#: One `bins` argument split into one spec per dimension.
#:
#: An int repeats. A list or tuple is one entry per dimension, each of them an
#: int, an edge array, or a list of edges. `histogram` and `histogram2d`
#: rewrite their own arguments into this before calling through, because numpy
#: reads a bare sequence as *edges* there and as *one spec per dimension*
#: here, and both readings have to survive.
fn _bin_specs(bins: any, dims: int, what: str) -> list[any] {
    if bins is int { return [bins for _d in 0..dims] }
    if bins is NDArray {
        if dims != 1 { throw ValueError(f"{what} needs one bin spec per dimension") }
        return [bins]
    }
    let items = _as_list(bins, what)
    if items.len() != dims {
        throw ValueError(f"{items.len()} bin specs for {dims} dimensions")
    }
    return items
}

#: One `range` argument split the same way. numpy lets a one-dimensional call
#: write `(low, high)` where the general form wants `[(low, high)]`; the two
#: are distinguishable because the short form holds numbers.
fn _bounds_specs(bounds: any, dims: int, what: str) -> list[any] {
    if bounds is null { return [null for _d in 0..dims] }
    let items = _as_list(bounds, what)
    if items.len() == 2 and (items[0] is float or items[0] is int) {
        if dims != 1 { throw ValueError(f"one pair of bounds for {dims} dimensions") }
        return [items]
    }
    if items.len() != dims { throw ValueError(f"{items.len()} bounds for {dims} dimensions") }
    return items
}

#: The edges for one dimension, and whether they are evenly spaced — which is
#: what lets the kernel find a bin with a multiply instead of a bisection.
fn _edges_for(column: NDArray, spec: any, bound: any, what: str) -> (NDArray, bool) {
    if spec is NDArray or spec is list or spec is tuple {
        let given = _as_1d(spec, what)
        if given.size() < 2 {
            given.free()
            throw ValueError(f"{what} needs at least two edges to make a bin")
        }
        return (given, false)
    }
    let count = _index(spec)
    if count < 1 { throw ValueError(f"{what} needs at least one bin, not {count}") }
    var low = 0.0
    var high = 1.0
    if bound is not null {
        let (given_low, given_high) = _pair(bound, what)
        low = given_low
        high = given_high
    } elif column.size() > 0 {
        #: The extent of the data, which costs a reduction and a synchronise.
        #: numpy pays exactly the same: there is no placing even bins without
        #: first knowing where the samples stop.
        let lowest = amin(column)
        let highest = amax(column)
        low = lowest.item()
        high = highest.item()
        lowest.free()
        highest.free()
    }
    if not low - high <= 0.0 or not math.is_finite(low) or not math.is_finite(high) {
        throw ValueError(f"{what} cannot place bins over [{low}, {high}]")
    }
    #: numpy's rule for an extent of zero, which is also what an empty input
    #: falls into.
    if low == high {
        low = low - 0.5
        high = high + 0.5
    }
    return (linspace(low, high, count + 1), true)
}

#: The n-dimensional histogram of `sample`, shaped `[rows, dims]`, and the bin
#: edges it used.
#:
#: One thread per sample and an atomic into the bin it lands in. When the bins
#: fit in threadgroup memory the tally is staged there and merged once per
#: group, which is what keeps a million samples from serialising on a handful
#: of device slots — `jaicv`'s picture histograms found the same cliff.
#:
#: `bounds` is numpy's `range`, renamed because `range` is a type in this
#: language. `density` divides by the cell volume and by the total that landed
#: inside, so the result integrates to one.
pub fn histogramdd(
    sample: NDArray,
    bins: any = 10,
    bounds: any = null,
    weights: NDArray? = null,
    density: bool = false
) -> (NDArray, list[NDArray]) {
    require_real(sample, "histogramdd")
    let block = _sample_block(sample, "histogramdd")
    defer { block.free() }
    let rows = block.shape[0]
    let dims = block.shape[1]
    if dims == 0 { throw ValueError("histogramdd needs at least one dimension") }

    let specs = _bin_specs(bins, dims, "histogramdd")
    let limits = _bounds_specs(bounds, dims, "histogramdd")
    var edges: list[NDArray] = []
    var even: list[bool] = []
    var shape: list[int] = []
    for d in 0..dims {
        let (line, spaced) = _edges_for(_column(block, d), specs[d], limits[d], "histogramdd")
        edges.push(line)
        even.push(spaced)
        shape.push(line.size() - 1)
    }
    var counts = zeros(shape, FLOAT32)
    let total_bins = element_count(shape)
    if rows == 0 or total_bins == 0 { return (counts, edges) }

    let joined = concatenate(edges, 0)
    defer { joined.free() }
    var described = [float(dims)]
    for extent in shape { described.push(float(extent)) }
    var base = 0
    for d in 0..dims {
        described.push(float(base))
        base += edges[d].size()
    }
    for spaced in even { described.push(if spaced { 1.0 } else { 0.0 }) }
    let meta = Buffer(described.len())
    meta.upload(described)
    defer { meta.free() }

    var mass = block
    var owns_mass = false
    if weights is not null {
        let (flat, owned) = _flat_1d(weights)
        if flat.size() != rows {
            if owned { flat.free() }
            throw ValueError(f"{flat.size()} weights for {rows} samples")
        }
        mass = flat
        owns_mass = owned
    }
    defer { if owns_mass { mass.free() } }

    if weights is null and total_bins <= STAGED_BINS {
        let groups = min(_TALLY_GROUPS, (rows + GROUP - 1) // GROUP)
        kernel_of(SOURCE, "nst_hist_staged").dispatch_async(
            [counts.storage, block.storage, joined.storage, meta],
            groups * GROUP,
            [rows, dims, total_bins, groups * GROUP],
            GROUP
        )
    } else {
        kernel_of(SOURCE, "nst_hist").dispatch_async([counts.storage, block.storage, joined.storage, mass.storage, meta], rows, [rows, dims, if weights is null {
            0
        } else {
            1
        }])
    }
    if not density { return (counts, edges) }

    let inside = sum(counts)
    defer { inside.free() }
    for d in 0..dims {
        let widths = diff(edges[d], 1, 0)
        var spread = [1 for _k in 0..dims]
        spread[d] = shape[d]
        let column = widths.view(spread, contiguous_strides(spread, 1), 0)
        let next = binary_arith(counts, column, OP_DIV)
        widths.free()
        counts.free()
        counts = next
    }
    let scaled = binary_arith(counts, inside, OP_DIV)
    counts.free()
    return (scaled, edges)
}

#: Threadgroups a staged tally runs over at most. Past a thousand the merge
#: sweeps -- one per group, over every bin -- start to cost more than the
#: contention they avoid.
const _TALLY_GROUPS = 1024

#: The histogram of `a`, and the edges it used.
#:
#: `bins` is a count or the edges themselves; `bounds` is numpy's `range`.
#: Unweighted counts come back as `INT64`, as numpy's do; a weighted or a
#: density histogram is a float.
pub fn histogram(
    a: NDArray,
    bins: any = 10,
    bounds: any = null,
    weights: NDArray? = null,
    density: bool = false
) -> (NDArray, NDArray) {
    require_real(a, "histogram")
    let (flat, owns_flat) = _flat_1d(a)
    defer { if owns_flat { flat.free() } }
    #: A sequence here is the edges, where `histogramdd` reads a sequence as
    #: one spec per dimension. Wrapping it in a list of one keeps both.
    let (counts, edges) = histogramdd(flat, [bins], bounds, weights, density)
    if density or weights is not null { return (counts, edges[0]) }
    let tallied = counts.astype(INT64)
    counts.free()
    return (tallied, edges[0])
}

#: The joint histogram of two equally long samples, and both sets of edges.
pub fn histogram2d(
    x: NDArray,
    y: NDArray,
    bins: any = 10,
    bounds: any = null,
    weights: NDArray? = null,
    density: bool = false
) -> (NDArray, NDArray, NDArray) {
    require_real(x, "histogram2d")
    require_real(y, "histogram2d")
    if x.size() != y.size() {
        throw ValueError(f"histogram2d needs matching samples, got {x.size()} and {y.size()}")
    }
    let block = NDArray([x.size(), 2], FLOAT32)
    defer { block.free() }
    copy_into(_column(block, 0), x.view([x.size()], contiguous_strides(x.shape, 1), x.offset))
    copy_into(_column(block, 1), y.view([y.size()], contiguous_strides(y.shape, 1), y.offset))
    var specs: list[any] = [bins, bins]
    if not bins is int and not bins is NDArray { specs = _as_list(bins, "histogram2d") }
    let (counts, edges) = histogramdd(block, specs, bounds, weights, density)
    return (counts, edges[0], edges[1])
}

#: How many times each whole number from zero appears in `x`.
#:
#: One thread per sample and an atomic into its slot — the bin *is* the value,
#: so there is no search at all. `minlength` pads the result; the length is
#: otherwise one past the largest value, which costs a reduction and a
#: synchronise to learn, as it does in numpy.
pub fn bincount(x: NDArray, weights: NDArray? = null, minlength: int = 0) -> NDArray {
    require_real(x, "bincount")
    if x.ndim() > 1 { throw ValueError(f"bincount takes a 1-d array, not {x.ndim()}") }
    if minlength < 0 { throw ValueError(f"bincount cannot pad to {minlength}") }
    let (flat, owns_flat) = _flat_1d(x)
    defer { if owns_flat { flat.free() } }
    let count = flat.size()
    var length = minlength
    if count > 0 {
        let lowest = amin(flat)
        let highest = amax(flat)
        let low = lowest.item()
        let high = highest.item()
        lowest.free()
        highest.free()
        if not low >= 0.0 { throw ValueError(f"bincount has no bin for {low}") }
        if low != math.floor(low) or high != math.floor(high) {
            throw ValueError("bincount counts whole numbers; round or cast first")
        }
        length = max(length, int(high) + 1)
    }
    let counts = zeros([length], FLOAT32)
    if count == 0 or length == 0 {
        if weights is not null { return counts }
        let empty = counts.astype(INT64)
        counts.free()
        return empty
    }

    var mass = flat
    var owns_mass = false
    if weights is not null {
        let (given, owned) = _flat_1d(weights)
        if given.size() != count {
            if owned { given.free() }
            throw ValueError(f"{given.size()} weights for {count} samples")
        }
        mass = given
        owns_mass = owned
    }
    defer { if owns_mass { mass.free() } }

    if weights is null and length <= STAGED_BINS {
        let groups = min(_TALLY_GROUPS, (count + GROUP - 1) // GROUP)
        kernel_of(SOURCE, "nst_bincount_staged").dispatch_async(
            [counts.storage, flat.storage],
            groups * GROUP,
            [count, length, groups * GROUP],
            GROUP
        )
    } else {
        let quad = count % 4 == 0
        kernel_of(SOURCE, "nst_bincount").dispatch_async([counts.storage, flat.storage, mass.storage], if quad {
            count // 4
        } else {
            count
        }, [count, length, if weights is null {
            0
        } else {
            1
        }, if quad {
            1
        } else {
            0
        }])
    }
    if weights is not null { return counts }
    let tallied = counts.astype(INT64)
    counts.free()
    return tallied
}

#: Which bin of `bins` each element of `x` belongs to.
#:
#: A bisection per query, which is `searchsorted` — the whole of this function
#: is picking the side and turning a falling sequence over. `bins` may rise or
#: fall but must do one of them; the check is two reductions and is what stops
#: an unsorted array from returning quiet nonsense.
pub fn digitize(x: NDArray, bins: NDArray, right: bool = false) -> NDArray {
    require_real(x, "digitize")
    require_real(bins, "digitize")
    if bins.ndim() != 1 { throw ValueError(f"digitize takes 1-d bins, not {bins.ndim()}") }
    let side = if right { "left" } else { "right" }
    if bins.size() <= 1 { return searchsorted(bins, x, side) }

    let steps = diff(bins, 1, 0)
    defer { steps.free() }
    let rising = _never_negative(steps, false)
    if rising { return searchsorted(bins, x, side) }
    if not _never_negative(steps, true) {
        throw ValueError("digitize needs bins that rise or fall throughout")
    }
    let turned = flip(bins, 0).contiguous()
    defer { turned.free() }
    let at = searchsorted(turned, x, side)
    defer { at.free() }
    return subtract(bins.size(), at)
}

#: Whether every step has the sign the caller is asking about.
#:
#: `clip(v, 0, null) - v` is zero exactly where `v` is at least zero and
#: positive where it is not, so "never negative" is "that shortfall is nowhere
#: nonzero" -- which is a count, not a truth fold.
#:
#: It used to clip the shortfall to `[-0.0, 0.0]` and hand THAT to `all`. The
#: clip made it zero everywhere, and `all` over zeros is false, so this
#: answered false for every input it was ever given: `digitize` rejected bins
#: as "neither rising nor falling" whatever they were.
fn _never_negative(steps: NDArray, inverted: bool) -> bool {
    let signed = if inverted { binary_arith(steps, -1.0, OP_MUL) } else { steps }
    let lifted = clip(signed, 0.0, null)
    let shortfall = binary_arith(lifted, signed, OP_SUB)
    lifted.free()
    let counted = count_nonzero(shortfall)
    let wrong = counted.item()
    counted.free()
    shortfall.free()
    if inverted { signed.free() }
    return wrong == 0.0
}
