#: The discrete Fourier transform and the things built directly on it.
#:
#: `dft` takes a real or a two-channel complex `Mat` and returns the spectrum,
#: with OpenCV's flags deciding whether the output is complex, whether the
#: transform runs the other way, and whether it is scaled. Rows are transformed
#: then columns, which is what makes the two-dimensional transform separable
#: and is why `DFT_ROWS` exists at all.
#:
#: The transform is mixed-radix: a length that factors into twos, threes and
#: fives is split recursively, and whatever prime is left is transformed
#: directly. `get_optimal_dft_size` says how far to pad to stay on the fast
#: path, and pads to the same sizes OpenCV pads to.
#:
#: ```
#: let spectrum = dft(padded, DFT_COMPLEX_OUTPUT)
#: let filtered = idft(mul_spectrums(spectrum, kernel), DFT_SCALE | DFT_REAL_OUTPUT)
#: ```
import std.math
from std.gpu import Buffer, Kernel
from ..core.mat import Mat, f32_bits
from ..core.types import CV_32F, CV_MAKETYPE, CV_MAT_CN

#: Run the transform the other way.
pub const DFT_INVERSE = 1
#: Divide by the number of elements.
pub const DFT_SCALE = 2
#: Transform each row on its own.
pub const DFT_ROWS = 4
#: Return two channels even for a real input.
pub const DFT_COMPLEX_OUTPUT = 16
#: Return one channel, discarding an imaginary part that should be zero.
pub const DFT_REAL_OUTPUT = 32

let SOURCE = """#include <metal_stdlib>
using namespace metal;

/* One stage of a Stockham autosort transform.
 *
 * The classic Cooley-Tukey needs its input bit-reversed; Stockham writes each
 * stage to a second buffer in an order that leaves the next stage contiguous,
 * so the two buffers alternate and no permutation pass is needed. After
 * log2(n) stages the answer is in natural order.
 *
 * `base_stride` and `line_stride` are what let the same kernel take both axes:
 * along a row the elements are one apart and the rows are `cols` apart, and
 * down a column it is the other way round. Transposing between the two passes
 * would cost two more passes over the picture for nothing. */
kernel void dft_stage(device const float* in_real [[buffer(0)]],
                      device const float* in_imaginary [[buffer(1)]],
                      device float* out_real [[buffer(2)]],
                      device float* out_imaginary [[buffer(3)]],
                      constant uint& count [[buffer(4)]],
                      constant uint& n [[buffer(5)]],
                      constant uint& m [[buffer(6)]],
                      constant uint& base_stride [[buffer(7)]],
                      constant uint& line_stride [[buffer(8)]],
                      constant uint& sign_bits [[buffer(9)]],
                      uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    /* Not `half`: that names a type in this language. */
    uint span = n / 2u;
    uint line = id / span;
    uint j = id - line * span;
    uint k = j & (m - 1u);
    uint block = j / m;
    float sign = as_type<float>(sign_bits);
    float angle = sign * 6.283185307179586f * float(k) / float(2u * m);
    float wr = cos(angle);
    float wi = sin(angle);
    uint lb = line * line_stride;
    uint i0 = lb + (block * m + k) * base_stride;
    uint i1 = lb + (block * m + k + span) * base_stride;
    float ar = in_real[i0];
    float ai = in_imaginary[i0];
    float br = in_real[i1];
    float bi = in_imaginary[i1];
    float tr = wr * br - wi * bi;
    float ti = wr * bi + wi * br;
    uint o0 = lb + (2u * block * m + k) * base_stride;
    uint o1 = lb + (2u * block * m + k + m) * base_stride;
    out_real[o0] = ar + tr;
    out_imaginary[o0] = ai + ti;
    out_real[o1] = ar - tr;
    out_imaginary[o1] = ai - ti;
}

/* Pull a real or complex picture apart into the two planes the stages want. */
kernel void dft_split(device const float* src [[buffer(0)]],
                      device float* real [[buffer(1)]],
                      device float* imaginary [[buffer(2)]],
                      constant uint& count [[buffer(3)]],
                      constant uint& cn [[buffer(4)]],
                      uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    real[id] = src[id * cn];
    imaginary[id] = cn > 1u ? src[id * cn + 1u] : 0.0f;
}

/* And put them back, scaling on the way out when the flags asked for it. */
kernel void dft_pack(device const float* real [[buffer(0)]],
                     device const float* imaginary [[buffer(1)]],
                     device float* dst [[buffer(2)]],
                     constant uint& count [[buffer(3)]],
                     constant uint& cn [[buffer(4)]],
                     constant uint& scale_bits [[buffer(5)]],
                     uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float scale = as_type<float>(scale_bits);
    dst[id * cn] = real[id] * scale;
    if (cn > 1u) dst[id * cn + 1u] = imaginary[id] * scale;
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

#: Whether `n` is a power of two, which is what the staged transform handles.
fn is_power_of_two(n: int) -> bool { return n > 0 and n & n - 1 == 0 }

#: Transform every line of one axis, in place across a pair of buffers.
#:
#: Returns whichever buffer the answer landed in: the stages alternate, so with
#: an odd number of them it is the scratch one.
fn stage_axis(
    real: Buffer,
    imaginary: Buffer,
    scratch_real: Buffer,
    scratch_imaginary: Buffer,
    lines: int,
    n: int,
    base_stride: int,
    line_stride: int,
    inverse: bool
) -> tuple[Buffer, Buffer] {
    if n <= 1 { return (real, imaginary) }
    let sign = if inverse { 1.0 } else { -1.0 }
    let butterflies = lines * (n // 2)
    var from_real = real
    var from_imaginary = imaginary
    var into_real = scratch_real
    var into_imaginary = scratch_imaginary
    var m = 1
    while m < n {
        kernel_for("dft_stage").dispatch_async(
            [from_real, from_imaginary, into_real, into_imaginary],
            butterflies,
            [butterflies, n, m, base_stride, line_stride, f32_bits(sign)]
        )
        let swap_real = from_real
        let swap_imaginary = from_imaginary
        from_real = into_real
        from_imaginary = into_imaginary
        into_real = swap_real
        into_imaginary = swap_imaginary
        m = m * 2
    }
    return (from_real, from_imaginary)
}

fn transform_line(real: list[float], imaginary: list[float], inverse: bool) -> void {
    let n = real.len()
    if n <= 1 { return }

    var factor = 0
    for candidate in [2, 3, 5] {
        if n % candidate == 0 {
            factor = candidate
            break
        }
    }
    if factor == 0 {
        naive_line(real, imaginary, inverse)
        return
    }

    #: Cooley-Tukey: the transform of `n = factor * m` points is `factor`
    #: transforms of `m` points, taken over the interleaved subsequences, and
    #: then combined. Splitting on the smallest factor keeps the recursion
    #: shallow and the twiddle count down.
    let m = n // factor
    var parts_real: list[list[float]] = []
    var parts_imaginary: list[list[float]] = []
    for part in 0..factor {
        var pr: list[float] = []
        var pi: list[float] = []
        for index in 0..m {
            pr.push(real[index * factor + part])
            pi.push(imaginary[index * factor + part])
        }
        transform_line(pr, pi, inverse)
        parts_real.push(pr)
        parts_imaginary.push(pi)
    }

    let sign = if inverse { 1.0 } else { -1.0 }
    let step = sign * 2.0 * math.PI / float(n)
    for k in 0..n {
        var sum_real = 0.0
        var sum_imaginary = 0.0
        for part in 0..factor {
            let angle = step * float(part) * float(k)
            let c = math.cos(angle)
            let s = math.sin(angle)
            let pr = parts_real[part][k % m]
            let pi = parts_imaginary[part][k % m]
            sum_real += pr * c - pi * s
            sum_imaginary += pr * s + pi * c
        }
        real[k] = sum_real
        imaginary[k] = sum_imaginary
    }
}

fn naive_line(real: list[float], imaginary: list[float], inverse: bool) -> void {
    let n = real.len()
    let sign = if inverse { 1.0 } else { -1.0 }
    var out_real: list[float] = []
    var out_imaginary: list[float] = []
    for k in 0..n {
        var sum_real = 0.0
        var sum_imaginary = 0.0
        for index in 0..n {
            let angle = sign * 2.0 * math.PI * float(k) * float(index) / float(n)
            let c = math.cos(angle)
            let s = math.sin(angle)
            sum_real += real[index] * c - imaginary[index] * s
            sum_imaginary += real[index] * s + imaginary[index] * c
        }
        out_real.push(sum_real)
        out_imaginary.push(sum_imaginary)
    }
    for index in 0..n {
        real[index] = out_real[index]
        imaginary[index] = out_imaginary[index]
    }
}

fn to_complex(src: Mat) -> tuple[list[float], list[float]] {
    src.flush()
    let values = src.to_list()
    let cn = CV_MAT_CN(src.type_())
    var real: list[float] = []
    var imaginary: list[float] = []
    if cn == 1 {
        for value in values {
            real.push(value)
            imaginary.push(0.0)
        }
        return (real, imaginary)
    }
    if cn != 2 { throw ValueError("a transform takes one or two channels") }
    for index in 0..values.len() // 2 {
        real.push(values[index * 2])
        imaginary.push(values[index * 2 + 1])
    }
    return (real, imaginary)
}

#: The transform on the device, or null when this one has to go on the host.
#:
#: Only for power-of-two lengths: the stages are radix two, and a length with
#: any other factor keeps to the recursive host transform. That is not a small
#: difference in cost -- the host transform of a 512 by 512 picture takes about
#: five seconds and this takes about a millisecond -- so `get_optimal_dft_size`
#: is worth calling, though it still answers with threes and fives, which land
#: on the slow path.
fn gpu_dft(src: Mat, flags: int, inverse: bool) -> Mat? {
    let rows = src.rows
    let cols = src.cols
    let cn = CV_MAT_CN(src.type_())
    if cn != 1 and cn != 2 { return null }
    if rows < 1 or cols < 1 { return null }
    if not is_power_of_two(cols) { return null }
    let rows_only = flags & DFT_ROWS != 0 or rows == 1
    if not rows_only and not is_power_of_two(rows) { return null }
    #: A view carries an offset and a row gap; the stages address a packed
    #: plane, so anything else goes the other way.
    if src.offset != 0 or src.step_ != cols * cn { return null }

    src.flush()
    let count = rows * cols
    let real = Buffer(count)
    let imaginary = Buffer(count)
    let scratch_real = Buffer(count)
    let scratch_imaginary = Buffer(count)
    kernel_for("dft_split").dispatch_async([src.storage, real, imaginary], count, [count, cn])

    var (held_real, held_imaginary) = stage_axis(
        real,
        imaginary,
        scratch_real,
        scratch_imaginary,
        rows,
        cols,
        1,
        cols,
        inverse
    )
    if not rows_only {
        #: Down a column the elements are `cols` apart and the columns are one
        #: apart, which is the row pass with its two strides exchanged.
        let other_real = if held_real is real { scratch_real } else { real }
        let other_imaginary = if held_imaginary is imaginary {
            scratch_imaginary
        } else {
            imaginary
        }
        let (done_real, done_imaginary) = stage_axis(
            held_real,
            held_imaginary,
            other_real,
            other_imaginary,
            cols,
            rows,
            cols,
            1,
            inverse
        )
        held_real = done_real
        held_imaginary = done_imaginary
    }

    var scale = 1.0
    if flags & DFT_SCALE != 0 {
        scale = 1.0 / (if rows_only { float(cols) } else { float(rows * cols) })
    }
    let out_cn = if flags & DFT_REAL_OUTPUT != 0 { 1 } else { 2 }
    let out = Mat(rows, cols, CV_MAKETYPE(CV_32F, out_cn))
    kernel_for("dft_pack").dispatch_async(
        [held_real, held_imaginary, out.storage],
        count,
        [count, out_cn, f32_bits(scale)]
    )
    out.invalidate()
    real.free()
    imaginary.free()
    scratch_real.free()
    scratch_imaginary.free()
    return out
}

#: The discrete Fourier transform of an image.
pub fn dft(src: Mat, flags: int = 0, nonzero_rows: int = 0) -> Mat {
    let _unused = nonzero_rows
    let rows = src.rows
    let cols = src.cols
    let inverse = flags & DFT_INVERSE != 0
    let staged = gpu_dft(src, flags, inverse)
    if staged is not null { return staged }
    var (real, imaginary) = to_complex(src)

    for y in 0..rows {
        var line_real: list[float] = []
        var line_imaginary: list[float] = []
        for x in 0..cols {
            line_real.push(real[y * cols + x])
            line_imaginary.push(imaginary[y * cols + x])
        }
        transform_line(line_real, line_imaginary, inverse)
        for x in 0..cols {
            real[y * cols + x] = line_real[x]
            imaginary[y * cols + x] = line_imaginary[x]
        }
    }

    if flags & DFT_ROWS == 0 and rows > 1 {
        for x in 0..cols {
            var line_real: list[float] = []
            var line_imaginary: list[float] = []
            for y in 0..rows {
                line_real.push(real[y * cols + x])
                line_imaginary.push(imaginary[y * cols + x])
            }
            transform_line(line_real, line_imaginary, inverse)
            for y in 0..rows {
                real[y * cols + x] = line_real[y]
                imaginary[y * cols + x] = line_imaginary[y]
            }
        }
    }

    if flags & DFT_SCALE != 0 {
        let divisor = if flags & DFT_ROWS != 0 { float(cols) } else { float(rows * cols) }
        for index in 0..real.len() {
            real[index] /= divisor
            imaginary[index] /= divisor
        }
    }

    if flags & DFT_REAL_OUTPUT != 0 {
        return Mat.from_list(rows, cols, CV_MAKETYPE(CV_32F, 1), real)
    }
    var out: list[float] = []
    for index in 0..real.len() {
        out.push(real[index])
        out.push(imaginary[index])
    }
    return Mat.from_list(rows, cols, CV_MAKETYPE(CV_32F, 2), out)
}

#: The inverse transform, which is `dft` with `DFT_INVERSE` already set.
pub fn idft(src: Mat, flags: int = 0, nonzero_rows: int = 0) -> Mat {
    return dft(src, flags | DFT_INVERSE, nonzero_rows)
}

#: The next size at or above `size` that the transform handles quickly.
#:
#: That is the next number whose only prime factors are two, three and five,
#: which is the same rule OpenCV follows.
pub fn get_optimal_dft_size(size: int) -> int {
    if size <= 1 { return 1 }
    var candidate = size
    loop {
        var remaining = candidate
        for factor in [2, 3, 5] {
            while remaining % factor == 0 { remaining = remaining // factor }
        }
        if remaining == 1 { return candidate }
        candidate += 1
    }
    return size
}

#: Multiply two spectra element by element.
#:
#: With `conjugate` set the second is conjugated first, which turns the product
#: into a correlation rather than a convolution.
pub fn mul_spectrums(a: Mat, b: Mat, flags: int = 0, conjugate: bool = false) -> Mat {
    let _unused = flags
    if a.rows != b.rows or a.cols != b.cols {
        throw ValueError("two spectra have to be the same size")
    }
    let left = a.values()
    let right = b.values()
    let left_complex = CV_MAT_CN(a.type_()) == 2
    let right_complex = CV_MAT_CN(b.type_()) == 2
    var out: list[float] = []
    for index in 0..a.rows * a.cols {
        let ar = if left_complex { left[index * 2] } else { left[index] }
        let ai = if left_complex { left[index * 2 + 1] } else { 0.0 }
        let br = if right_complex { right[index * 2] } else { right[index] }
        var bi = if right_complex { right[index * 2 + 1] } else { 0.0 }
        if conjugate { bi = -bi }
        out.push(ar * br - ai * bi)
        out.push(ar * bi + ai * br)
    }
    return Mat.from_list(a.rows, a.cols, CV_MAKETYPE(CV_32F, 2), out)
}

#: The discrete cosine transform of an image, in its second form.
pub fn dct(src: Mat, flags: int = 0) -> Mat {
    src.flush()
    if CV_MAT_CN(src.type_()) != 1 { throw ValueError("a cosine transform takes one channel") }
    let inverse = flags & DFT_INVERSE != 0
    let rows = src.rows
    let cols = src.cols
    var values = src.to_list()

    for y in 0..rows {
        var line: list[float] = []
        for x in 0..cols { line.push(values[y * cols + x]) }
        let transformed = cosine_line(line, inverse)
        for x in 0..cols { values[y * cols + x] = transformed[x] }
    }
    if rows > 1 and flags & DFT_ROWS == 0 {
        for x in 0..cols {
            var line: list[float] = []
            for y in 0..rows { line.push(values[y * cols + x]) }
            let transformed = cosine_line(line, inverse)
            for y in 0..rows { values[y * cols + x] = transformed[y] }
        }
    }
    return Mat.from_list(rows, cols, CV_MAKETYPE(CV_32F, 1), values)
}

fn cosine_line(line: list[float], inverse: bool) -> list[float] {
    let n = line.len()
    var out: list[float] = []
    let root = math.sqrt(2.0 / float(n))
    if not inverse {
        for k in 0..n {
            var total = 0.0
            for index in 0..n {
                total += line[index] *
                    math.cos(math.PI * (float(index) + 0.5) * float(k) / float(n))
            }
            let weight = if k == 0 { math.sqrt(1.0 / float(n)) } else { root }
            out.push(total * weight)
        }
        return out
    }
    for index in 0..n {
        var total = 0.0
        for k in 0..n {
            let weight = if k == 0 { math.sqrt(1.0 / float(n)) } else { root }
            total += weight *
                line[k] *
                math.cos(math.PI * (float(index) + 0.5) * float(k) / float(n))
        }
        out.push(total)
    }
    return out
}
