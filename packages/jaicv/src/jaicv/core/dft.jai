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
from ..core.mat import Mat
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

#: The discrete Fourier transform of an image.
pub fn dft(src: Mat, flags: int = 0, nonzero_rows: int = 0) -> Mat {
    let _unused = nonzero_rows
    let rows = src.rows
    let cols = src.cols
    var (real, imaginary) = to_complex(src)
    let inverse = flags & DFT_INVERSE != 0

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
    while true {
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
    let left = a.to_list()
    let right = b.to_list()
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
