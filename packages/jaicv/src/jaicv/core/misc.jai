#: The rest of `core`: rearranging matrices, sorting them, filling them with
#: random numbers, and the handful of linear-algebra calls OpenCV puts here
#: rather than in a module of their own.
#:
#: Nothing here is deep. It is the surface a program written against OpenCV
#: reaches for between the interesting calls, and a port without it makes the
#: caller write the same twenty lines over and over.
#:
#: ```
#: let stacked = vconcat([top, bottom])
#: let turned = rotate(frame, ROTATE_90_CLOCKWISE)
#: let noise = randn(Size(64, 64), CV_32FC1, 0.0, 1.0)
#: ```
import std.math
from ..core.linalg import inverse, matmul, symmetric_eigen, transpose
from ..core.mat import Mat, merge
from ..core.types import CV_32F, CV_64F, CV_MAKETYPE, CV_MAT_CN, CV_MAT_DEPTH
from ..core.types import Point, Rect, Scalar, Size
from ..core.types import BORDER_CONSTANT, BORDER_REFLECT, BORDER_REFLECT_101
from ..core.types import BORDER_REPLICATE, BORDER_WRAP
from ..core.types import SORT_ASCENDING, SORT_DESCENDING, SORT_EVERY_COLUMN, SORT_EVERY_ROW

#: A quarter turn clockwise.
pub const ROTATE_90_CLOCKWISE = 0
#: A half turn.
pub const ROTATE_180 = 1
#: A quarter turn anticlockwise.
pub const ROTATE_90_COUNTERCLOCKWISE = 2

var _rng_state = 0x853C49E6748FEA

#: Fix the sequence the random fills draw from.
pub fn set_rng_seed(seed: int) -> void { _rng_state = if seed == 0 { 1 } else { seed } }

fn next_random() -> int {
    _rng_state = _rng_state *% 6364136223846793 +% 1442695040888963 & 0xFFFFFFFFFFFF
    return _rng_state >> 8
}

fn uniform_unit() -> float { return float(next_random() % 1000000007) / 1000000007.0 }

#: Lay matrices side by side.
pub fn hconcat(parts: list[Mat]) -> Mat {
    if parts.len() == 0 { throw ValueError("there is nothing to join") }
    let rows = parts[0].rows
    let cn = parts[0].channels()
    var width = 0
    for part in parts {
        if part.rows != rows { throw ValueError("every part has to have the same number of rows") }
        if part.channels() != cn { throw ValueError("every part has to have the same channels") }
        width += part.cols
    }
    var planes: list[list[float]] = []
    for part in parts { planes.push(part.values()) }
    var out: list[float] = []
    for y in 0..rows {
        for index in 0..parts.len() {
            let part = parts[index]
            for x in 0..part.cols {
                for c in 0..cn { out.push(planes[index][(y * part.cols + x) * cn + c]) }
            }
        }
    }
    return Mat.from_list(rows, width, parts[0].type_(), out)
}

#: Stack matrices on top of each other.
pub fn vconcat(parts: list[Mat]) -> Mat {
    if parts.len() == 0 { throw ValueError("there is nothing to join") }
    let cols = parts[0].cols
    let cn = parts[0].channels()
    var height = 0
    for part in parts {
        if part.cols != cols {
            throw ValueError("every part has to have the same number of columns")
        }
        if part.channels() != cn { throw ValueError("every part has to have the same channels") }
        height += part.rows
    }
    var out: list[float] = []
    for part in parts {
        for value in part.to_list() { out.push(value) }
    }
    return Mat.from_list(height, cols, parts[0].type_(), out)
}

#: Tile a matrix.
pub fn repeat(src: Mat, ny: int, nx: int) -> Mat {
    if ny <= 0 or nx <= 0 { throw ValueError("a tiling needs positive counts") }
    var rows: list[Mat] = []
    for _index in 0..nx { rows.push(src) }
    let wide = hconcat(rows)
    var stack: list[Mat] = []
    for _index in 0..ny { stack.push(wide) }
    return vconcat(stack)
}

#: Turn a matrix by a quarter or a half.
pub fn rotate(src: Mat, code: int) -> Mat {
    src.flush()
    let rows = src.rows
    let cols = src.cols
    let cn = src.channels()
    let values = src.to_list()
    var out: list[float] = []
    if code == ROTATE_180 {
        for index in 0..rows * cols {
            let source = rows * cols - 1 - index
            for c in 0..cn { out.push(values[source * cn + c]) }
        }
        return Mat.from_list(rows, cols, src.type_(), out)
    }
    for y in 0..cols {
        for x in 0..rows {
            let source = if code == ROTATE_90_CLOCKWISE {
                (rows - 1 - x) * cols + y
            } else {
                x * cols + (cols - 1 - y)
            }
            for c in 0..cn { out.push(values[source * cn + c]) }
        }
    }
    return Mat.from_list(cols, rows, src.type_(), out)
}

#: One channel of a multi-channel matrix, on its own.
pub fn extract_channel(src: Mat, index: int) -> Mat {
    let cn = src.channels()
    if index < 0 or index >= cn { throw ValueError(f"there is no channel {index}") }
    let values = src.to_list()
    var out: list[float] = []
    for at in 0..src.rows * src.cols { out.push(values[at * cn + index]) }
    return Mat.from_list(src.rows, src.cols, CV_MAKETYPE(src.depth(), 1), out)
}

#: Put a single-channel matrix into one channel of another.
pub fn insert_channel(source: Mat, target: Mat, index: int) -> Mat {
    let cn = target.channels()
    if index < 0 or index >= cn { throw ValueError(f"there is no channel {index}") }
    if source.rows != target.rows or source.cols != target.cols {
        throw ValueError("both have to be the same size")
    }
    let plane = source.values()
    var values = target.to_list()
    for at in 0..target.rows * target.cols { values[at * cn + index] = plane[at] }
    return Mat.from_list(target.rows, target.cols, target.type_(), values)
}

#: One line in order.
#:
#: Placing each value into a sorted list makes the work grow with the square of
#: the line's length, which for a picture's worth of rows is the difference
#: between milliseconds and minutes.
fn sorted_line(line: list[float], descending: bool) -> list[float] {
    return line.sort(reverse: descending)
}

#: Sort every row, or every column.
pub fn sort(src: Mat, flags: int = SORT_EVERY_ROW) -> Mat {
    src.flush()
    if src.channels() != 1 { throw ValueError("sorting needs a single-channel Mat") }
    let descending = flags & SORT_DESCENDING != 0
    let by_column = flags & SORT_EVERY_COLUMN != 0
    var values = src.to_list()
    if by_column {
        for x in 0..src.cols {
            var line: list[float] = []
            for y in 0..src.rows { line.push(values[y * src.cols + x]) }
            let sorted = sorted_line(line, descending)
            for y in 0..src.rows { values[y * src.cols + x] = sorted[y] }
        }
    } else {
        for y in 0..src.rows {
            var line: list[float] = []
            for x in 0..src.cols { line.push(values[y * src.cols + x]) }
            let sorted = sorted_line(line, descending)
            for x in 0..src.cols { values[y * src.cols + x] = sorted[x] }
        }
    }
    return Mat.from_list(src.rows, src.cols, src.type_(), values)
}

#: The order that would sort every row, or every column.
pub fn sort_idx(src: Mat, flags: int = SORT_EVERY_ROW) -> Mat {
    src.flush()
    if src.channels() != 1 { throw ValueError("sorting needs a single-channel Mat") }
    let descending = flags & SORT_DESCENDING != 0
    let by_column = flags & SORT_EVERY_COLUMN != 0
    let values = src.to_list()
    var out: list[float] = []
    for _index in 0..src.rows * src.cols { out.push(0.0) }

    #: Sorting the positions by what sits at them, rather than inserting each
    #: one into a list already in order -- see `sorted_line` for why.
    if by_column {
        for x in 0..src.cols {
            var order: list[int] = []
            for y in 0..src.rows { order.push(y) }
            let ranked = order.sort(key: |y| values[y * src.cols + x], reverse: descending)
            for y in 0..src.rows { out[y * src.cols + x] = float(ranked[y]) }
        }
        return Mat.from_list(src.rows, src.cols, CV_MAKETYPE(4, 1), out)
    }
    for y in 0..src.rows {
        var order: list[int] = []
        for x in 0..src.cols { order.push(x) }
        let ranked = order.sort(key: |x| values[y * src.cols + x], reverse: descending)
        for x in 0..src.cols { out[y * src.cols + x] = float(ranked[x]) }
    }
    return Mat.from_list(src.rows, src.cols, CV_MAKETYPE(4, 1), out)
}

#: A matrix filled with numbers drawn evenly between `low` and `high`.
pub fn randu(size: Size, type_code: int, low: float = 0.0, high: float = 1.0) -> Mat {
    let cn = CV_MAT_CN(type_code)
    var values: list[float] = []
    for _index in 0..size.width * size.height * cn {
        values.push(low + uniform_unit() * (high - low))
    }
    return Mat.from_list(size.height, size.width, type_code, values)
}

#: A matrix filled with numbers drawn from a bell curve.
pub fn randn(size: Size, type_code: int, mean: float = 0.0, stddev: float = 1.0) -> Mat {
    let cn = CV_MAT_CN(type_code)
    var values: list[float] = []
    var index = 0
    let total = size.width * size.height * cn
    while index < total {
        #: Box and Muller: two even draws become two bell-curve draws, which is
        #: why they come out in pairs.
        var u = uniform_unit()
        if u < 1e-12 { u = 1e-12 }
        let v = uniform_unit()
        let radius = math.sqrt(-2.0 * math.log(u))
        let angle = 2.0 * math.PI * v
        values.push(mean + stddev * radius * math.cos(angle))
        index += 1
        if index < total {
            values.push(mean + stddev * radius * math.sin(angle))
            index += 1
        }
    }
    return Mat.from_list(size.height, size.width, type_code, values)
}

#: Where every non-zero element is.
pub fn find_non_zero(src: Mat) -> list[Point] {
    src.flush()
    if src.channels() != 1 { throw ValueError("this needs a single-channel Mat") }
    let values = src.to_list()
    var out: list[Point] = []
    for y in 0..src.rows {
        for x in 0..src.cols {
            if values[y * src.cols + x] != 0.0 { out.push(Point(x, y)) }
        }
    }
    return out
}

#: The smallest and largest values, and where each of them is.
pub fn min_max_idx(src: Mat) -> tuple[float, float, Point, Point] {
    src.flush()
    let values = src.to_list()
    if values.len() == 0 { throw ValueError("an empty Mat has no extremes") }
    var low = values[0]
    var high = values[0]
    var low_at = 0
    var high_at = 0
    for index in 0..values.len() {
        if values[index] < low {
            low = values[index]
            low_at = index
        }
        if values[index] > high {
            high = values[index]
            high_at = index
        }
    }
    return (
        low,
        high,
        Point(low_at % src.cols, low_at // src.cols),
        Point(high_at % src.cols, high_at // src.cols)
    )
}

#: Turn `n` into an index inside `[0, length)` by the given border rule.
pub fn border_interpolate(position: int, length: int, border_type: int) -> int {
    if length <= 0 { return -1 }
    if position >= 0 and position < length { return position }
    if border_type == BORDER_CONSTANT { return -1 }
    if border_type == BORDER_REPLICATE {
        if position < 0 { return 0 }
        return length - 1
    }
    if border_type == BORDER_WRAP {
        var wrapped = position % length
        if wrapped < 0 { wrapped += length }
        return wrapped
    }
    if border_type == BORDER_REFLECT {
        var p = position
        while p < 0 or p >= length {
            if p < 0 { p = -p - 1 }
            if p >= length { p = 2 * length - p - 1 }
        }
        return p
    }
    if length == 1 { return 0 }
    var p = position
    while p < 0 or p >= length {
        if p < 0 { p = -p }
        if p >= length { p = 2 * length - p - 2 }
    }
    return p
}

#: An identity matrix scaled by `value`, written into a matrix of this size.
pub fn set_identity(size: Size, type_code: int, value: float = 1.0) -> Mat {
    var values: list[float] = []
    for y in 0..size.height {
        for x in 0..size.width { values.push(if x == y { value } else { 0.0 }) }
    }
    return Mat.from_list(size.height, size.width, type_code, values)
}

#: The determinant of a square matrix.
pub fn determinant(src: Mat) -> float {
    if src.rows != src.cols { throw ValueError("only a square matrix has a determinant") }
    let n = src.rows
    let values = src.to_list()
    var work: list[list[float]] = []
    for row in 0..n {
        var line: list[float] = []
        for column in 0..n { line.push(values[row * n + column]) }
        work.push(line)
    }
    var total = 1.0
    for column in 0..n {
        var pivot = column
        for row in column..n {
            if math.fabs(work[row][column]) > math.fabs(work[pivot][column]) { pivot = row }
        }
        if math.fabs(work[pivot][column]) < 1e-300 { return 0.0 }
        if pivot != column {
            let swap = work[pivot]
            work[pivot] = work[column]
            work[column] = swap
            total = -total
        }
        total *= work[column][column]
        for row in column + 1..n {
            let factor = work[row][column] / work[column][column]
            if factor == 0.0 { continue }
            for index in column..n { work[row][index] -= factor * work[column][index] }
        }
    }
    return total
}

#: The mean of a set of samples and how they vary about it.
#:
#: `samples` has one sample per row. Returns the covariance matrix and the mean
#: as a row, which is the pair OpenCV's `calcCovarMatrix` produces.
pub fn calc_covar_matrix(samples: Mat, normalise: bool = true) -> tuple[Mat, Mat] {
    samples.flush()
    let rows = samples.rows
    let width = samples.cols * samples.channels()
    if rows < 2 { throw ValueError("a covariance needs at least two samples") }
    let values = samples.to_list()

    var centre: list[float] = []
    for column in 0..width {
        var total = 0.0
        for row in 0..rows { total += values[row * width + column] }
        centre.push(total / float(rows))
    }
    var covariance: list[float] = []
    for _index in 0..width * width { covariance.push(0.0) }
    for row in 0..rows {
        for i in 0..width {
            let a = values[row * width + i] - centre[i]
            for j in 0..width {
                covariance[i * width + j] += a * (values[row * width + j] - centre[j])
            }
        }
    }
    if normalise {
        for index in 0..width * width { covariance[index] /= float(rows - 1) }
    }
    return (
        Mat.from_list(width, width, CV_MAKETYPE(CV_32F, 1), covariance),
        Mat.from_list(1, width, CV_MAKETYPE(CV_32F, 1), centre)
    )
}

#: How far apart two vectors are, measured in the units the data itself varies
#: in rather than in raw distance.
pub fn mahalanobis(a: Mat, b: Mat, inverse_covariance: Mat) -> float {
    let left = a.values()
    let right = b.values()
    if left.len() != right.len() { throw ValueError("both vectors have to be the same length") }
    let width = left.len()
    let precision = inverse_covariance.values()
    var total = 0.0
    for i in 0..width {
        var row = 0.0
        for j in 0..width { row += precision[i * width + j] * (left[j] - right[j]) }
        total += (left[i] - right[i]) * row
    }
    return math.sqrt(max(total, 0.0))
}

#: `src` times its own transpose, or the other way round.
pub fn mul_transposed(src: Mat, a_ta: bool = true) -> Mat {
    if a_ta { return matmul(transpose(src), src) }
    return matmul(src, transpose(src))
}

#: The eigenvalues and eigenvectors of a symmetric matrix, largest first.
pub fn eigen(src: Mat) -> tuple[list[float], Mat] { return symmetric_eigen(src) }

#: The real roots of a polynomial, given its coefficients from the constant up.
#:
#: Found by Durand-Kerner, which starts every root at a different point on a
#: circle and lets them repel each other into place; the ones that end up real
#: are returned in order.
pub fn solve_poly(coefficients: list[float], iterations: int = 300) -> list[float] {
    var trimmed = [value for value in coefficients]
    while trimmed.len() > 1 and math.fabs(trimmed[trimmed.len() - 1]) < 1e-14 { trimmed.pop() }
    let degree = trimmed.len() - 1
    if degree < 1 { return [] }

    var real: list[float] = []
    var imaginary: list[float] = []
    for index in 0..degree {
        let angle = 2.0 * math.PI * float(index) / float(degree) + 0.4
        real.push(0.4 * math.cos(angle))
        imaginary.push(0.4 * math.sin(angle))
    }

    let lead = trimmed[degree]
    for _round in 0..iterations {
        var moved = 0.0
        for index in 0..degree {
            var value_real = trimmed[degree] / lead
            var value_imaginary = 0.0
            var power = degree - 1
            while power >= 0 {
                let nr = value_real * real[index] - value_imaginary * imaginary[index] +
                    trimmed[power] / lead
                let ni = value_real * imaginary[index] + value_imaginary * real[index]
                value_real = nr
                value_imaginary = ni
                power -= 1
            }
            var denominator_real = 1.0
            var denominator_imaginary = 0.0
            for other in 0..degree {
                if other == index { continue }
                let dr = real[index] - real[other]
                let di = imaginary[index] - imaginary[other]
                let nr = denominator_real * dr - denominator_imaginary * di
                let ni = denominator_real * di + denominator_imaginary * dr
                denominator_real = nr
                denominator_imaginary = ni
            }
            let size = denominator_real * denominator_real +
                denominator_imaginary * denominator_imaginary
            if size < 1e-30 { continue }
            let step_real = (value_real * denominator_real + value_imaginary * denominator_imaginary) /
                size
            let step_imaginary = (value_imaginary * denominator_real - value_real * denominator_imaginary) /
                size
            real[index] -= step_real
            imaginary[index] -= step_imaginary
            moved += step_real * step_real + step_imaginary * step_imaginary
        }
        if moved < 1e-24 { break }
    }

    var roots: list[float] = []
    for index in 0..degree {
        if math.fabs(imaginary[index]) > 1e-6 { continue }
        var at = roots.len()
        while at > 0 and real[index] < roots[at - 1] { at -= 1 }
        roots.insert(at, real[index])
    }
    return roots
}

#: Whether every element is finite and, if asked, inside a range.
pub fn check_range(src: Mat, low: float = -1e300, high: float = 1e300) -> bool {
    for value in src.to_list() {
        if value != value { return false }
        if value < low or value >= high { return false }
    }
    return true
}

#: Replace anything that is not a number with `value`.
pub fn patch_nans(src: Mat, value: float = 0.0) -> Mat {
    let values = src.to_list()
    var out: list[float] = []
    for entry in values { out.push(if entry != entry { value } else { entry }) }
    return Mat.from_list(src.rows, src.cols, src.type_(), out)
}
