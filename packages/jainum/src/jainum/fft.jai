#: Discrete Fourier transforms — numpy's `fft` namespace, on the device.
#:
#: Complex arrays are `COMPLEX64` from `array.jai`: interleaved re/im, two
#: float slots per element, which is the layout jaicv's `CV_32FC2` spectra
#: already use. Every entry point here returns one, except the ones whose
#: whole point is a real answer (`irfft`, `hfft`, `fftfreq`).
#:
#: **Nothing in this file may fall back to a host loop.** `jaicv/core/dft.jai`
#: learned that the expensive way: its recursive host transform takes about
#: five seconds over a 512 by 512 picture where the device path takes about a
#: millisecond, and the only reason that host path still exists there is that
#: it predates the device one and covers the lengths the stages do not. There
#: is no length here that is not covered, so there is no host transform at
#: all. A pure Jaithon `for k { for j { ... } }` over a spectrum is seconds per
#: frame, and a "small enough" threshold below which it looks fine is a trap:
#: the small case is the one that runs in a loop.
#:
#: Three paths, in the order the length picks them:
#:
#: - **Power of two.** A Stockham radix-4 autosort chain, one dispatch per
#:   stage, ping-ponging between two buffers. Stockham writes each stage in an
#:   order that leaves the next one contiguous, so there is no bit-reversal
#:   pass at either end. An odd exponent takes one radix-2 stage first, at
#:   `m = 1`, where every twiddle is one.
#: - **3- and 5-smooth.** The same chain with radix-3 and radix-5 stages
#:   appended. Worth having and not only for tidiness: 1000, 1200 and 4800 are
#:   sample counts people actually transform, and Bluestein would turn a
#:   length-1000 transform into three length-2048 ones.
#: - **Everything else.** Bluestein's chirp-z, built on the power-of-two path:
#:   a length-`n` transform becomes two forward and one inverse transform of
#:   length `L`, the first power of two at or above `2n - 1`, plus three
#:   pointwise passes. Roughly six times the work of a same-size power of two,
#:   which is the price of never having a slow length.
#:
#: A multi-dimensional transform runs the one-dimensional one along each axis
#: in turn. Each pass moves its axis to the fast position with `moveaxis` from
#: `manipulate.jai` and lets the strided copy do the transposition on the
#: device, so the stage kernels only ever walk a packed `[lines, n]` block and
#: never carry an axis stride.
#:
#: ```
#: let spectrum = fft(signal)
#: let power = spectrum * conj(spectrum)
#: let back = ifft(spectrum)              #: within float32 of `signal`
#: let bins = rfftfreq(signal.shape[0], 1.0 / 48000.0)
#: ```
#:
#: Accuracy is float32's, and the transform is the good kind of float32: the
#: error grows like sqrt(log n) rather than like n, because a stage chain sums
#: in a tree. Bluestein is the exception — its chirp angles are computed from
#: `t^2` folded modulo `2n`, which keeps the argument small, but the three
#: transforms and two pointwise products it stacks still cost it a digit
#: against a direct length.
import std.math

from .array import COMPLEX64, FLOAT32, NDArray, PRELUDE, copy_into
from .array import dtype_is_complex, element_count, f32_bits, fill, kernel_of
from .array import layout_buffer, normalize_axes, normalize_axis, require_real
from .manipulate import moveaxis, roll

let SOURCE = PRELUDE + """
constexpr constant float NF_TAU = 6.283185307179586f;
constexpr constant float NF_PI = 3.141592653589793f;

inline float2 nf_mul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

/* exp(i * angle), which is every twiddle this file has. */
inline float2 nf_cis(float angle) {
    return float2(cos(angle), sin(angle));
}

/* The four outputs of one radix-4 butterfly, from four already-twiddled
 * inputs.
 *
 *   y[t] = sum over u of x[u] * W^(u*t),   W = exp(sign*2*pi*i/4) = sign*i
 *
 * written out rather than looped, because W^2 = -1 and W^3 = -W collapse the
 * sixteen products into four sums and one rotation. Multiplying by sign*i is
 * a swap and a negation, not a call to cos. */
inline void nf_butterfly4(float2 x0, float2 x1, float2 x2, float2 x3,
                          float sign, thread float2* y) {
    float2 p = x0 + x2;
    float2 q = x0 - x2;
    float2 r = x1 + x3;
    float2 d = x1 - x3;
    float2 s = float2(-sign * d.y, sign * d.x);
    y[0] = p + r;
    y[1] = q + s;
    y[2] = p - r;
    y[3] = q - s;
}

/* One radix-2 Stockham stage, and only ever the first one.
 *
 * A power of two with an odd exponent needs exactly one radix-2 stage, and
 * the plan puts it at the front where `m` is 1: every twiddle there is one,
 * so the whole stage is a load, an add and a subtract. The two outputs of
 * butterfly `j` land at 2j and 2j+1 -- adjacent -- so the write is a single
 * float4 even though the two reads are half the line apart.
 *
 * There is deliberately no general-`m` radix-2 kernel. No length reaches one:
 * the plan spends every remaining power of two on radix-4 stages. */
kernel void nf_radix2(device float2* dst [[buffer(0)]],
                      device const float2* src [[buffer(1)]],
                      constant uint& count [[buffer(2)]],
                      constant uint& span [[buffer(3)]],
                      uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint line = id / span;
    uint j = id - line * span;
    uint base = line * span * 2u;
    float2 a = src[base + j];
    float2 b = src[base + j + span];
    device float4* wide = (device float4*)dst;
    wide[line * span + j] = float4(a + b, a - b);
}

/* One radix-4 Stockham stage, a butterfly to a thread.
 *
 * `m` is the product of the radices already applied and `groups` is n/4. In
 * Stockham's layout butterfly `j` reads at `j`, `j + groups`, `j + 2*groups`
 * and `j + 3*groups`, and writes at `4*block*m + k + t*m` for t in 0..3,
 * where `j = block*m + k`. The read index being `j` itself is what makes the
 * pair kernel below possible.
 *
 * Two divisions per thread and no modulo: `line` off `id`, `block` off `j`,
 * and both remainders by subtraction. The device has no integer divide unit,
 * so a modulo beside each division would be four of them. */
kernel void nf_radix4(device float2* dst [[buffer(0)]],
                      device const float2* src [[buffer(1)]],
                      constant uint& count [[buffer(2)]],
                      constant uint& groups [[buffer(3)]],
                      constant uint& m [[buffer(4)]],
                      constant uint& sign_bits [[buffer(5)]],
                      uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint line = id / groups;
    uint j = id - line * groups;
    uint block = j / m;
    uint k = j - block * m;
    uint base = line * groups * 4u;
    float sign = as_type<float>(sign_bits);
    float step = sign * NF_TAU * float(k) / float(4u * m);

    float2 y[4];
    nf_butterfly4(
        src[base + j],
        nf_mul(src[base + j + groups], nf_cis(step)),
        nf_mul(src[base + j + 2u * groups], nf_cis(step * 2.0f)),
        nf_mul(src[base + j + 3u * groups], nf_cis(step * 3.0f)),
        sign,
        y
    );
    uint at = base + 4u * block * m + k;
    for (uint t = 0u; t < 4u; t++) dst[at + t * m] = y[t];
}

/* The same stage with two adjacent butterflies to a thread, which is the one
 * the fast lengths take.
 *
 * Butterfly `j` reads at index `j`, so a thread owning `j` and `j+1` reads
 * two neighbouring complex numbers -- one float4 -- from each of the four
 * quarters. That is the whole reason to pair: a radix-4 butterfly's own four
 * reads are a quarter of a line apart and can never coalesce with each other.
 *
 * The writes coalesce too whenever `m` is even, which is the caller's
 * precondition: `j` even and `m` even make `k` even, so `k+1 < m` keeps both
 * butterflies inside one block and every output index `4*block*m + k + t*m`
 * is even. The only radix-4 stage that misses this is the first one of a pure
 * power of four, where m = 1; that one goes to the scalar kernel above. */
kernel void nf_radix4_pair(device float2* dst [[buffer(0)]],
                           device const float2* src [[buffer(1)]],
                           constant uint& count [[buffer(2)]],
                           constant uint& groups [[buffer(3)]],
                           constant uint& m [[buffer(4)]],
                           constant uint& sign_bits [[buffer(5)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint pairs = groups >> 1u;
    uint line = id / pairs;
    uint j = (id - line * pairs) << 1u;
    uint block = j / m;
    uint k = j - block * m;
    uint base = line * groups * 4u;
    float sign = as_type<float>(sign_bits);
    float step = sign * NF_TAU / float(4u * m);

    device const float4* wide_src = (device const float4*)src;
    uint at = (base + j) >> 1u;
    uint quarter = groups >> 1u;
    float4 v0 = wide_src[at];
    float4 v1 = wide_src[at + quarter];
    float4 v2 = wide_src[at + 2u * quarter];
    float4 v3 = wide_src[at + 3u * quarter];

    float even = step * float(k);
    float odd = step * float(k + 1u);
    float2 lo[4];
    float2 hi[4];
    nf_butterfly4(
        v0.xy,
        nf_mul(v1.xy, nf_cis(even)),
        nf_mul(v2.xy, nf_cis(even * 2.0f)),
        nf_mul(v3.xy, nf_cis(even * 3.0f)),
        sign,
        lo
    );
    nf_butterfly4(
        v0.zw,
        nf_mul(v1.zw, nf_cis(odd)),
        nf_mul(v2.zw, nf_cis(odd * 2.0f)),
        nf_mul(v3.zw, nf_cis(odd * 3.0f)),
        sign,
        hi
    );
    device float4* wide_dst = (device float4*)dst;
    uint out = base + 4u * block * m + k;
    for (uint t = 0u; t < 4u; t++) wide_dst[(out + t * m) >> 1u] = float4(lo[t], hi[t]);
}

/* One radix-3 or radix-5 Stockham stage, a butterfly to a thread.
 *
 * The radix-4 kernels are this with the radix fixed and the sums written out;
 * keeping both means the common case does not pay for a loop over a runtime
 * bound, and this one does not pay for four specialisations that would each
 * run once per transform. The two agree by construction:
 *
 *   a[u] = x[block*m + k + u*groups] * exp(sign*2*pi*i*u*k/(radix*m))
 *   y[radix*block*m + k + t*m] = sum over u of a[u] * exp(sign*2*pi*i*u*t/radix)
 *
 * No float4 here. These stages sit at the end of the plan where `m` is
 * whatever the earlier radices multiplied to, so neither the reads nor the
 * writes have a guaranteed even run, and a radix-3 stage moves three floats
 * where a radix-4 one moves four. */
kernel void nf_radix_odd(device float2* dst [[buffer(0)]],
                         device const float2* src [[buffer(1)]],
                         constant uint& count [[buffer(2)]],
                         constant uint& groups [[buffer(3)]],
                         constant uint& m [[buffer(4)]],
                         constant uint& radix [[buffer(5)]],
                         constant uint& sign_bits [[buffer(6)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint line = id / groups;
    uint j = id - line * groups;
    uint block = j / m;
    uint k = j - block * m;
    uint base = line * groups * radix;
    float sign = as_type<float>(sign_bits);

    float2 a[5];
    float inner = sign * NF_TAU / float(radix * m);
    for (uint u = 0u; u < radix; u++) {
        a[u] = nf_mul(src[base + j + u * groups], nf_cis(inner * float(u * k)));
    }
    float outer = sign * NF_TAU / float(radix);
    uint at = base + radix * block * m + k;
    for (uint t = 0u; t < radix; t++) {
        float2 sum = float2(0.0f, 0.0f);
        for (uint u = 0u; u < radix; u++) {
            sum += nf_mul(a[u], nf_cis(outer * float(u * t)));
        }
        dst[at + t * m] = sum;
    }
}

/* exp(sign * pi * i * t^2 / n), Bluestein's chirp.
 *
 * `t*t` is folded modulo 2n before it becomes an angle, and that fold is not
 * optional: the chirp has period 2n in t^2, and for a length in the tens of
 * thousands the raw square leaves float32's exact range long before the
 * cosine sees it -- the angle it would produce is noise, not a rotation. A
 * 64-bit multiply is the only way to hold the square first. One 64-bit
 * remainder per element is a rounding error against the three power-of-two
 * transforms this chirp exists to feed.
 *
 * `2u * n` is a uint, so this holds for n below 2^31; an NDArray stops
 * counting exactly at 2^24 long before that. */
inline float2 nf_chirp(uint t, uint n, float sign) {
    ulong square = ulong(t) * ulong(t);
    uint folded = uint(square % ulong(2u * n));
    return nf_cis(sign * NF_PI * float(folded) / float(n));
}

/* Bluestein pass one: the signal premultiplied by the chirp and laid into the
 * head of a longer line. The destination's tail was zeroed by `fill` -- a
 * kernel that wrote the zeros itself would need a grid over the padding, and
 * the padding is most of the buffer. */
kernel void nf_chirp_pre(device float2* dst [[buffer(0)]],
                         device const float2* src [[buffer(1)]],
                         constant uint& count [[buffer(2)]],
                         constant uint& n [[buffer(3)]],
                         constant uint& ell [[buffer(4)]],
                         constant uint& sign_bits [[buffer(5)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint line = id / n;
    uint t = id - line * n;
    dst[line * ell + t] = nf_mul(src[id], nf_chirp(t, n, as_type<float>(sign_bits)));
}

/* Bluestein pass two: the convolution kernel, one line shared by every line
 * of the batch.
 *
 * The chirp is even in its offset, so the negative half of the filter is the
 * positive half read backwards -- which is what `min(t, ell - t)` picks out --
 * and everything at an offset of n or more is zero, which is exactly the
 * headroom that stops the circular convolution from wrapping onto itself. */
kernel void nf_chirp_filter(device float2* dst [[buffer(0)]],
                            constant uint& ell [[buffer(1)]],
                            constant uint& n [[buffer(2)]],
                            constant uint& sign_bits [[buffer(3)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= ell) return;
    uint offset = min(id, ell - id);
    if (offset >= n) {
        dst[id] = float2(0.0f, 0.0f);
        return;
    }
    dst[id] = nf_chirp(offset, n, -as_type<float>(sign_bits));
}

/* Bluestein pass three: the batch's spectra times the filter's, in place.
 *
 * In place because the destination of a pointwise product is the only buffer
 * that is already resident and about to be overwritten anyway; there is no
 * cross-thread dependency to make that unsafe. */
kernel void nf_spectrum_mul(device float2* dst [[buffer(0)]],
                            device const float2* filter [[buffer(1)]],
                            constant uint& count [[buffer(2)]],
                            constant uint& ell [[buffer(3)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint line = id / ell;
    uint t = id - line * ell;
    dst[id] = nf_mul(dst[id], filter[t]);
}

/* Bluestein pass four: the convolution's head, chirped back and scaled.
 *
 * The 1/ell the inverse transform owes is folded in here rather than run as
 * its own pass -- this kernel already touches every output element once. */
kernel void nf_chirp_post(device float2* dst [[buffer(0)]],
                          device const float2* src [[buffer(1)]],
                          constant uint& count [[buffer(2)]],
                          constant uint& n [[buffer(3)]],
                          constant uint& ell [[buffer(4)]],
                          constant uint& sign_bits [[buffer(5)]],
                          constant uint& scale_bits [[buffer(6)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint line = id / n;
    uint k = id - line * n;
    float2 value = nf_mul(src[line * ell + k], nf_chirp(k, n, as_type<float>(sign_bits)));
    dst[id] = value * as_type<float>(scale_bits);
}

/* A half spectrum widened back into the full one it stands for.
 *
 * A real signal's transform satisfies X[n-k] = conj(X[k]), so the bins past
 * n/2 are a mirrored conjugate of the bins the caller kept and no arithmetic
 * is needed to recover them. This is the packing direction that has to be a
 * kernel; the other one is a narrowed axis, which is a view. */
kernel void nf_hermitian(device float2* dst [[buffer(0)]],
                         device const float2* src [[buffer(1)]],
                         constant uint& count [[buffer(2)]],
                         constant uint& n [[buffer(3)]],
                         constant uint& bins [[buffer(4)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint line = id / n;
    uint k = id - line * n;
    if (k < bins) {
        dst[id] = src[line * bins + k];
        return;
    }
    float2 mirrored = src[line * bins + (n - k)];
    dst[id] = float2(mirrored.x, -mirrored.y);
}

/* The complex conjugate, dense and strided. Not `a * (1, -1)`: the arithmetic
 * kernels in `array.jai` take a real scalar or a whole second array, and
 * neither spelling of this is one of those. */
kernel void nf_conj(device float2* dst [[buffer(0)]],
                    device const float2* src [[buffer(1)]],
                    constant uint& count [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float2 value = src[id];
    dst[id] = float2(value.x, -value.y);
}

kernel void nf_conj_strided(device float* dst [[buffer(0)]],
                            device const float* src [[buffer(1)]],
                            device const float* meta [[buffer(2)]],
                            constant uint& count [[buffer(3)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    int to = nd_offset(meta, 0u, id);
    int from = nd_offset(meta, 1u, id);
    dst[to] = src[from];
    dst[to + 1] = -src[from + 1];
}

/* An in-place scale over a fresh contiguous allocation, four floats to a
 * thread with a scalar tail.
 *
 * Only ever called on a buffer this file allocated, which is why the float4
 * cast is safe without an alignment check: a Metal buffer starts on a 256
 * byte boundary and the offset is zero. */
kernel void nf_scale(device float* dst [[buffer(0)]],
                     constant uint& count [[buffer(1)]],
                     constant uint& bits [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {
    uint at = id * 4u;
    if (at >= count) return;
    float scale = as_type<float>(bits);
    if (at + 4u <= count) {
        device float4* wide = (device float4*)dst;
        wide[id] = wide[id] * scale;
        return;
    }
    for (uint k = at; k < count; k++) dst[k] = dst[k] * scale;
}

/* The sample frequencies of a transform of `count` bins.
 *
 * `wrap` is the first index that reads as a negative frequency, so passing
 * `count` for it turns this into the plain ramp `rfftfreq` wants and the two
 * entry points share one kernel. */
kernel void nf_fftfreq(device float* dst [[buffer(0)]],
                       constant uint& count [[buffer(1)]],
                       constant uint& wrap [[buffer(2)]],
                       constant uint& step_bits [[buffer(3)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float index = id < wrap ? float(id) : float(id) - float(count);
    dst[id] = index * as_type<float>(step_bits);
}
"""

#: The stage chain that transforms a length of `n`, or null when a prime past
#: five is left over and Bluestein has to take it.
#:
#: The order is deliberate. Radix 2 goes first because it is only ever needed
#: once and `m = 1` is where its twiddles vanish; the radix-4 stages follow so
#: their `m` stays a power of two, which is what keeps them on the paired
#: float4 path; the odd radices go last because they are scalar either way.
pub fn radix_plan(n: int) -> list[int]? {
    if n < 1 { return null }
    var left = n
    var twos = 0
    while left % 2 == 0 {
        twos += 1
        left = left // 2
    }
    var odd: list[int] = []
    for candidate in [3, 5] {
        while left % candidate == 0 {
            odd.push(candidate)
            left = left // candidate
        }
    }
    if left != 1 { return null }
    var plan: list[int] = []
    if twos % 2 == 1 { plan.push(2) }
    for _stage in 0..twos // 2 { plan.push(4) }
    for radix in odd { plan.push(radix) }
    return plan
}

#: Whether `n` runs on the stage chain rather than through Bluestein. The
#: numbers `next_fast_length` rounds up to.
pub fn is_fast_length(n: int) -> bool { return radix_plan(n) is not null }

#: The smallest length at or above `n` that the stage chain transforms
#: directly — numpy's `next_fast_len`, and the length to pad to when the data
#: does not care about its exact duration.
pub fn next_fast_length(n: int) -> int {
    var size = max(1, n)
    while radix_plan(size) is null { size += 1 }
    return size
}

#: Run one stage chain over a packed `[lines, n]` block, ping-ponging between
#: `work` and `scratch`, and report whether the answer landed in `scratch`.
#:
#: Neither array's shape is read — the stages address a packed line and take
#: their geometry as scalars — so a caller is free to hand in arrays shaped
#: like the array it means to return.
fn run_plan(
    work: NDArray,
    scratch: NDArray,
    lines: int,
    n: int,
    plan: list[int],
    sign: float
) -> bool {
    var source = work
    var target = scratch
    var flipped = false
    var m = 1
    let bits = f32_bits(sign)
    for radix in plan {
        let groups = n // radix
        let threads = lines * groups
        if radix == 2 {
            kernel_of(SOURCE, "nf_radix2").dispatch_async(
                [target.storage, source.storage],
                threads,
                [threads, groups]
            )
        } elif radix == 4 and groups % 2 == 0 and m % 2 == 0 {
            let pairs = lines * (groups // 2)
            kernel_of(SOURCE, "nf_radix4_pair").dispatch_async(
                [target.storage, source.storage],
                pairs,
                [pairs, groups, m, bits]
            )
        } elif radix == 4 {
            kernel_of(SOURCE, "nf_radix4").dispatch_async(
                [target.storage, source.storage],
                threads,
                [threads, groups, m, bits]
            )
        } else {
            kernel_of(SOURCE, "nf_radix_odd").dispatch_async(
                [target.storage, source.storage],
                threads,
                [threads, groups, m, radix, bits]
            )
        }
        let spare = source
        source = target
        target = spare
        flipped = not flipped
        m *= radix
    }
    return flipped
}

#: Bluestein's chirp-z transform of a packed `[lines, n]` block into `out`.
#:
#: The identity is `2*j*k = j^2 + k^2 - (j-k)^2`, which turns the transform's
#: `exp(sign*2*pi*i*j*k/n)` into a chirp on the way in, a convolution against
#: an even kernel, and a chirp on the way out. The convolution runs circularly
#: at length `ell`, the first power of two at or above `2n - 1`, and that
#: headroom is what stops the wrap from folding one end of the signal onto the
#: other.
#:
#: Five dispatches plus three stage chains. The filter's own transform is one
#: line and is recomputed per call rather than cached: caching it would key on
#: (n, sign) and hold device memory for the life of the process, and the line
#: is a thousandth of the batch it multiplies.
fn bluestein(out: NDArray, work: NDArray, lines: int, n: int, sign: float) -> void {
    let ell = next_power_of_two(2 * n - 1)
    let plan = radix_plan(ell) ?? []
    let signal = NDArray([lines, ell], COMPLEX64)
    let signal_scratch = NDArray([lines, ell], COMPLEX64)
    let filter = NDArray([ell], COMPLEX64)
    let filter_scratch = NDArray([ell], COMPLEX64)
    defer { signal.free() }
    defer { signal_scratch.free() }
    defer { filter.free() }
    defer { filter_scratch.free() }

    let bits = f32_bits(sign)
    let head = lines * n
    #: The chirp only writes the first `n` of each line; the rest is the
    #: padding the convolution needs and has to start at zero.
    fill(signal, 0.0)
    kernel_of(SOURCE, "nf_chirp_pre").dispatch_async(
        [signal.storage, work.storage],
        head,
        [head, n, ell, bits]
    )
    kernel_of(SOURCE, "nf_chirp_filter").dispatch_async([filter.storage], ell, [ell, n, bits])

    #: The convolution's own transforms are always the ordinary pair, whatever
    #: direction the caller asked for: the chirps carry the sign, and a
    #: forward-inverse pair is what makes the product a convolution.
    let signal_flipped = run_plan(signal, signal_scratch, lines, ell, plan, -1.0)
    let filter_flipped = run_plan(filter, filter_scratch, 1, ell, plan, -1.0)
    let spectrum = if signal_flipped { signal_scratch } else { signal }
    let spare = if signal_flipped { signal } else { signal_scratch }
    let filter_spectrum = if filter_flipped { filter_scratch } else { filter }

    let total = lines * ell
    kernel_of(SOURCE, "nf_spectrum_mul").dispatch_async(
        [spectrum.storage, filter_spectrum.storage],
        total,
        [total, ell]
    )
    let product_flipped = run_plan(spectrum, spare, lines, ell, plan, 1.0)
    let product = if product_flipped { spare } else { spectrum }
    kernel_of(SOURCE, "nf_chirp_post").dispatch_async(
        [out.storage, product.storage],
        head,
        [head, n, ell, bits, f32_bits(1.0 / float(ell))]
    )
}

#: The first power of two at or above `n`.
pub fn next_power_of_two(n: int) -> int {
    var size = 1
    while size < n { size *= 2 }
    return size
}

#: The one-dimensional transform of `a` along `axis`, resampled to `n` points
#: and scaled on the way out.
#:
#: The axis is moved to the last position and the data copied into a packed
#: complex block, which is one strided dispatch and does the transposition,
#: the real-to-complex widening, and the truncate-or-zero-pad to `n` all at
#: once. The stages then walk a block whose fast axis is the transformed one,
#: which is the only layout they know how to address.
fn transform_axis(a: NDArray, n: int, axis: int, sign: float, scale: float) -> NDArray {
    if a.ndim() == 0 { throw ValueError("a transform needs an axis; a 0-d array has none") }
    if n < 1 { throw ValueError(f"a transform length must be positive, not {n}") }
    let at = normalize_axis(axis, a.ndim())
    let last = a.ndim() - 1
    let moved = moveaxis(a, at, last)
    var packed = [extent for extent in moved.shape]
    packed[last] = n
    let lines = element_count(packed) // n

    let work = NDArray(packed, COMPLEX64)
    let take = min(n, moved.shape[last])
    #: Only the padded case owes a zero pass; a transform at the input's own
    #: length overwrites every slot with the copy that follows.
    if take < n { fill(work, 0.0) }
    if take > 0 { copy_into(head_of(work, take), head_of(moved, take)) }

    var result = work
    let plan = radix_plan(n)
    if plan is null {
        let chirped = NDArray(packed, COMPLEX64)
        bluestein(chirped, work, lines, n, sign)
        work.free()
        result = chirped
    } else {
        let scratch = NDArray(packed, COMPLEX64)
        let flipped = run_plan(work, scratch, lines, n, plan, sign)
        result = if flipped { scratch } else { work }
        let unused = if flipped { work } else { scratch }
        unused.free()
    }
    if scale != 1.0 { scale_in_place(result, scale) }
    return restore_axis(result, a.shape, at, n)
}

#: A view of the first `take` elements of the last axis.
fn head_of(a: NDArray, take: int) -> NDArray {
    let last = a.ndim() - 1
    if take == a.shape[last] { return a }
    var shape = [extent for extent in a.shape]
    shape[last] = take
    return a.view(shape, a.strides, a.offset)
}

#: Put a result whose transformed axis is last back where the caller's axis
#: order wants it.
#:
#: Free when the axis was already last, which is the default and most of the
#: traffic. Otherwise one strided copy — and not a returned view, because the
#: caller has to own what it frees and a view owns nothing.
fn restore_axis(result: NDArray, source_shape: list[int], at: int, n: int) -> NDArray {
    let last = source_shape.len() - 1
    if at == last { return result }
    var shape = [extent for extent in source_shape]
    shape[at] = n
    let out = NDArray(shape, result.dtype)
    copy_into(moveaxis(out, at, last), result)
    result.free()
    return out
}

#: Multiply a fresh contiguous array by a constant, in place.
fn scale_in_place(a: NDArray, scale: float) -> void {
    let count = a.size() * a.lanes()
    if count == 0 { return }
    kernel_of(SOURCE, "nf_scale").dispatch_async(
        [a.storage],
        (count + 3) // 4,
        [count, f32_bits(scale)]
    )
}

#: The factor a transform of `n` points carries under `norm`.
#:
#: numpy's three conventions: `backward` puts the whole 1/n on the inverse,
#: `forward` puts it on the forward transform, and `ortho` splits it so both
#: directions are unitary and Parseval holds without a factor.
fn norm_scale(norm: str?, n: int, inverse: bool) -> float {
    let name = norm ?? "backward"
    if name == "backward" { return if inverse { 1.0 / float(n) } else { 1.0 } }
    if name == "ortho" { return 1.0 / math.sqrt(float(n)) }
    if name == "forward" { return if inverse { 1.0 } else { 1.0 / float(n) } }
    throw ValueError(f"norm {name} is not backward, ortho or forward")
}

#: `norm` read from the other direction, which is how the Hermitian pair is
#: defined: `hfft` is `irfft` of the conjugate under the swapped convention,
#: and `ihfft` is the conjugate of `rfft` under it.
fn swap_norm(norm: str?) -> str {
    let name = norm ?? "backward"
    if name == "backward" { return "forward" }
    if name == "forward" { return "backward" }
    if name == "ortho" { return "ortho" }
    throw ValueError(f"norm {name} is not backward, ortho or forward")
}

#: Elements `a` has along `axis`.
fn axis_length(a: NDArray, axis: int) -> int {
    return a.shape[normalize_axis(axis, a.ndim())]
}

#: The one-dimensional discrete Fourier transform along `axis`.
#:
#: `n` truncates or zero-pads the axis first, as numpy's does. The result is
#: `COMPLEX64` whatever the input was.
pub fn fft(a: NDArray, n: int? = null, axis: int = -1, norm: str? = null) -> NDArray {
    let length = n ?? axis_length(a, axis)
    return transform_axis(a, length, axis, -1.0, norm_scale(norm, length, false))
}

#: The one-dimensional inverse transform along `axis`.
pub fn ifft(a: NDArray, n: int? = null, axis: int = -1, norm: str? = null) -> NDArray {
    let length = n ?? axis_length(a, axis)
    return transform_axis(a, length, axis, 1.0, norm_scale(norm, length, true))
}

#: The transform over several axes, one after another.
#:
#: Each axis is a full pass: a transposing copy in, a stage chain, and a
#: transposing copy back. Doing it that way rather than teaching the stages an
#: axis stride is what keeps every dispatch reading a packed line — a column
#: transform addressed in place would have every thread of a warp reading a
#: row apart, and the bandwidth that costs is more than the two copies.
pub fn fftn(
    a: NDArray,
    s: list[int]? = null,
    axes: list[int]? = null,
    norm: str? = null
) -> NDArray {
    return transform_many(a, s, axes, norm, false)
}

#: The inverse transform over several axes.
pub fn ifftn(
    a: NDArray,
    s: list[int]? = null,
    axes: list[int]? = null,
    norm: str? = null
) -> NDArray {
    return transform_many(a, s, axes, norm, true)
}

fn transform_many(
    a: NDArray,
    s: list[int]?,
    axes: list[int]?,
    norm: str?,
    inverse: bool
) -> NDArray {
    let rank = a.ndim()
    let order = normalize_axes(axes ?? default_axes(rank, s), rank, "axes")
    if s is not null and s.len() != order.len() {
        throw ValueError(f"{s.len()} lengths for {order.len()} axes")
    }
    var current = a
    var owned = false
    for k in 0..order.len() {
        let length = if s is null { axis_length(current, order[k]) } else { s[k] }
        let sign = if inverse { 1.0 } else { -1.0 }
        let next = transform_axis(
            current,
            length,
            order[k],
            sign,
            norm_scale(norm, length, inverse)
        )
        if owned { current.free() }
        current = next
        owned = true
    }
    #: Nothing to transform still owes the caller an owned complex array, not
    #: a view of the argument they are about to free.
    if not owned { return a.astype(COMPLEX64) }
    return current
}

#: Every axis, or the trailing ones a list of lengths names — numpy's rule
#: when `axes` is left out.
fn default_axes(rank: int, s: list[int]?) -> list[int] {
    if s is null { return [axis for axis in 0..rank] }
    if s.len() > rank { throw ValueError(f"{s.len()} lengths for an array of {rank} axes") }
    return [rank - s.len() + k for k in 0..s.len()]
}

#: The transform over the last two axes.
pub fn fft2(
    a: NDArray,
    s: list[int]? = null,
    axes: list[int]? = null,
    norm: str? = null
) -> NDArray {
    return fftn(a, s, axes ?? plane_axes(a), norm)
}

#: The inverse transform over the last two axes.
pub fn ifft2(
    a: NDArray,
    s: list[int]? = null,
    axes: list[int]? = null,
    norm: str? = null
) -> NDArray {
    return ifftn(a, s, axes ?? plane_axes(a), norm)
}

fn plane_axes(a: NDArray) -> list[int] {
    if a.ndim() < 2 { throw ValueError("a two-dimensional transform needs two axes") }
    return [a.ndim() - 2, a.ndim() - 1]
}

#: The transform of a real signal, keeping only the bins that are not the
#: mirror of another: `n // 2 + 1` of them.
#:
#: The full complex transform runs first and the redundant half is dropped.
#: The alternative — packing the `n` real samples into `n/2` complex ones,
#: transforming at half the length and untangling — would save about half the
#: stage work, and it is not taken here because the untangle is its own kernel
#: and its own class of off-by-one at the Nyquist bin. If a real-heavy
#: workload ever makes that half matter, this is the function to change and
#: `test_rfft_matches_the_full_transform` is the test that pins it.
pub fn rfft(a: NDArray, n: int? = null, axis: int = -1, norm: str? = null) -> NDArray {
    require_real(a, "rfft")
    let length = n ?? axis_length(a, axis)
    let full = fft(a, length, axis, norm)
    defer { full.free() }
    return half_spectrum(full, length, axis).copy()
}

#: The inverse of `rfft`: a half spectrum back to the `n` real samples it came
#: from.
#:
#: `n` defaults to `2 * (bins - 1)`, which is numpy's rule and the reason a
#: round trip through an odd length has to be told the length back.
pub fn irfft(a: NDArray, n: int? = null, axis: int = -1, norm: str? = null) -> NDArray {
    let bins = axis_length(a, axis)
    let length = n ?? 2 * (bins - 1)
    if length < 1 { throw ValueError(f"an inverse real transform of {bins} bins has no length") }
    let full = full_spectrum(a, length, axis)
    defer { full.free() }
    let signal = transform_axis(full, length, axis, 1.0, norm_scale(norm, length, true))
    defer { signal.free() }
    #: `.real()` is a view over every other float slot; the copy is what makes
    #: it a packed real array the caller owns.
    return signal.real().copy()
}

#: `rfft` over the last two axes: the real transform along the last, an
#: ordinary complex one along the other.
pub fn rfft2(
    a: NDArray,
    s: list[int]? = null,
    axes: list[int]? = null,
    norm: str? = null
) -> NDArray {
    let order = normalize_axes(axes ?? plane_axes(a), a.ndim(), "axes")
    if order.len() != 2 { throw ValueError(f"rfft2 takes two axes, not {order.len()}") }
    let last = if s is null { axis_length(a, order[1]) } else { s[1] }
    let half = rfft(a, last, order[1], norm)
    defer { half.free() }
    let first = if s is null { axis_length(a, order[0]) } else { s[0] }
    return fft(half, first, order[0], norm)
}

#: The inverse of `rfft2`, unwound in the other order.
pub fn irfft2(
    a: NDArray,
    s: list[int]? = null,
    axes: list[int]? = null,
    norm: str? = null
) -> NDArray {
    let order = normalize_axes(axes ?? plane_axes(a), a.ndim(), "axes")
    if order.len() != 2 { throw ValueError(f"irfft2 takes two axes, not {order.len()}") }
    let first = if s is null { axis_length(a, order[0]) } else { s[0] }
    let stage = ifft(a, first, order[0], norm)
    defer { stage.free() }
    let last = if s is null { 2 * (axis_length(a, order[1]) - 1) } else { s[1] }
    return irfft(stage, last, order[1], norm)
}

#: The transform of a signal that is Hermitian in the time domain, which
#: therefore has a real spectrum.
#:
#: `hfft(a, n)` is `irfft(conj(a), n)` under the opposite scaling convention —
#: the same identity numpy is defined by, and the reason both of these are
#: three lines rather than a fourth transform path.
pub fn hfft(a: NDArray, n: int? = null, axis: int = -1, norm: str? = null) -> NDArray {
    let bins = axis_length(a, axis)
    let length = n ?? 2 * (bins - 1)
    let mirrored = conj(a)
    defer { mirrored.free() }
    return irfft(mirrored, length, axis, swap_norm(norm))
}

#: The inverse of `hfft`: a real signal to the half spectrum of its Hermitian
#: counterpart.
pub fn ihfft(a: NDArray, n: int? = null, axis: int = -1, norm: str? = null) -> NDArray {
    require_real(a, "ihfft")
    let length = n ?? axis_length(a, axis)
    let spectrum = rfft(a, length, axis, swap_norm(norm))
    defer { spectrum.free() }
    return conj(spectrum)
}

#: The bins of a full spectrum that are not the mirror of another: 0 through
#: `n // 2`.
#:
#: Free, and a view of its argument — one narrowed axis is a shape and a
#: stride, so this is not a kernel and must not become one. `full_spectrum` is
#: the direction that needs one.
pub fn half_spectrum(spectrum: NDArray, n: int? = null, axis: int = -1) -> NDArray {
    let at = normalize_axis(axis, spectrum.ndim())
    let length = n ?? spectrum.shape[at]
    var shape = [extent for extent in spectrum.shape]
    shape[at] = min(length // 2 + 1, spectrum.shape[at])
    return spectrum.view(shape, spectrum.strides, spectrum.offset)
}

#: A half spectrum widened into the full `n`-bin one it stands for, by
#: `X[n - k] = conj(X[k])`.
#:
#: A short input is zero-padded to `n // 2 + 1` bins and a long one is cut to
#: it, which is what lets `irfft` be asked for any output length. The result
#: is a fresh allocation: the mirrored half is a different value at a
#: different slot, so no stride rewrite reaches it.
pub fn full_spectrum(half: NDArray, n: int, axis: int = -1) -> NDArray {
    if half.ndim() == 0 { throw ValueError("a spectrum needs an axis; a 0-d array has none") }
    if n < 1 { throw ValueError(f"a spectrum of {n} bins is empty") }
    let at = normalize_axis(axis, half.ndim())
    let last = half.ndim() - 1
    let moved = moveaxis(half, at, last)
    let bins = n // 2 + 1
    var packed = [extent for extent in moved.shape]
    packed[last] = bins
    let lines = element_count(packed) // bins

    let kept = NDArray(packed, COMPLEX64)
    defer { kept.free() }
    let take = min(bins, moved.shape[last])
    if take < bins { fill(kept, 0.0) }
    if take > 0 { copy_into(head_of(kept, take), head_of(moved, take)) }

    var shape = [extent for extent in packed]
    shape[last] = n
    let out = NDArray(shape, COMPLEX64)
    let total = lines * n
    kernel_of(SOURCE, "nf_hermitian").dispatch_async(
        [out.storage, kept.storage],
        total,
        [total, n, bins]
    )
    return restore_axis(out, half.shape, at, n)
}

#: The complex conjugate, as an array of its own.
#:
#: A real array conjugates to itself, and this still hands back a copy rather
#: than the argument: a caller that frees what it got must not be freeing what
#: it passed.
pub fn conj(a: NDArray) -> NDArray {
    if not dtype_is_complex(a.dtype) { return a.copy() }
    let out = NDArray(a.shape, COMPLEX64)
    let count = out.size()
    if count == 0 { return out }
    if a.offset == 0 and a.is_contiguous() {
        kernel_of(SOURCE, "nf_conj").dispatch_async([out.storage, a.storage], count, [count])
        return out
    }
    let meta = layout_buffer([out, a])
    defer { meta.free() }
    kernel_of(SOURCE, "nf_conj_strided").dispatch_async(
        [out.storage, a.storage, meta],
        count,
        [count]
    )
    return out
}

#: `conj` under numpy's longer name.
pub fn conjugate(a: NDArray) -> NDArray { return conj(a) }

#: The sample frequencies of an `n`-point transform, in cycles per unit of
#: `d`.
#:
#: The second half reads as negative, which is where the ordering `fftshift`
#: exists to fix comes from: `[0, 1, ..., n/2 - 1, -n/2, ..., -1] / (n * d)`.
pub fn fftfreq(n: int, d: float = 1.0) -> NDArray {
    if n < 1 { throw ValueError(f"a transform of {n} points has no frequencies") }
    let out = NDArray([n], FLOAT32)
    kernel_of(SOURCE, "nf_fftfreq").dispatch_async(
        [out.storage],
        n,
        [n, (n + 1) // 2, f32_bits(1.0 / (float(n) * d))]
    )
    return out
}

#: The sample frequencies `rfft` returns bins for: the non-negative half,
#: `n // 2 + 1` of them, still divided by the *signal's* `n` and not by the
#: bin count.
pub fn rfftfreq(n: int, d: float = 1.0) -> NDArray {
    if n < 1 { throw ValueError(f"a transform of {n} points has no frequencies") }
    let bins = n // 2 + 1
    let out = NDArray([bins], FLOAT32)
    kernel_of(SOURCE, "nf_fftfreq").dispatch_async(
        [out.storage],
        bins,
        [bins, bins, f32_bits(1.0 / (float(n) * d))]
    )
    return out
}

#: Zero frequency moved to the middle of each named axis.
#:
#: A roll, which `manipulate.jai` already writes as two strided copies per
#: axis. There is nothing for a kernel to do here that a shape and an offset
#: do not already do.
pub fn fftshift(a: NDArray, axes: any = null) -> NDArray {
    let order = shift_axes(a, axes)
    return roll(a, [a.shape[at] // 2 for at in order], order)
}

#: The inverse of `fftshift`, which is not the same roll back for an odd
#: length — hence two functions rather than a sign.
pub fn ifftshift(a: NDArray, axes: any = null) -> NDArray {
    let order = shift_axes(a, axes)
    return roll(a, [-(a.shape[at] // 2) for at in order], order)
}

fn shift_axes(a: NDArray, axes: any) -> list[int] {
    if axes is null { return [axis for axis in 0..a.ndim()] }
    if axes is int { return [normalize_axis(axes, a.ndim(), "axes")] }
    if axes is list { return normalize_axes([item for item in axes], a.ndim(), "axes") }
    throw TypeError(f"axes takes an int or a list of ints, not {type_of(axes)}")
}
