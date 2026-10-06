#: Small dense linear algebra: the pieces geometry needs and nothing more.
#:
#: Everything here works on `Mat`. Most of it runs on the host: the matrices
#: geometry gives it are three by three, nine by nine, or a Jacobian a few
#: dozen wide -- sizes where a kernel launch costs more than the arithmetic,
#: and where being able to read the code matters more than the last
#: microsecond. A product big enough to be worth a dispatch, and the scatter
#: and projection a principal component analysis of a whole dataset needs, go
#: to the device through jaitensor's GEMM instead.
#:
#: ```
#: let (values, vectors) = symmetric_eigen(a)
#: let h = null_space(design)
#: let fitted = levenberg_marquardt(residual, jacobian, start, 40)
#: ```
import std.math
import jaitensor as jt
from jaitensor import Tensor
from std.os import env_or
from ..core.mat import Mat
from ..core.types import CV_32F, CV_MAKETYPE

#: Products of at least this many multiply-adds go to the device; smaller
#: ones stay on the host.
pub let DEVICE_WORK = 1 << 20

#: `JAICV_DEVICE_LINALG=0` keeps every product, covariance and projection on
#: the host whatever its size.
let _DEVICE_LINALG = env_or("JAICV_DEVICE_LINALG", "1") != "0"

#: Whether `work` multiply-adds should go to the device.
pub fn on_device(work: int) -> bool { return _DEVICE_LINALG and work >= DEVICE_WORK }

#: A tensor reading `mat`'s values where they lie, or null when they are not
#: one unbroken run of single-channel values. Freeing the tensor releases the
#: view, not the pixels.
pub fn device_view(mat: Mat) -> Tensor? {
    if mat.channels() != 1 or not mat.is_continuous() or mat.rows * mat.cols == 0 { return null }
    let storage = mat.buffer()
    return Tensor([mat.rows, mat.cols], storage: storage.view(mat.offset, mat.rows * mat.cols))
}

#: A float matrix holding `product`'s values, which it takes over.
fn mat_of(product: Tensor) -> Mat {
    return Mat(product.shape[0], product.shape[1], CV_MAKETYPE(CV_32F, 1), storage: product.buffer())
}

#: `a` (or its transpose) times `b` (or its transpose) on the device, or null
#: when either is not a plain run of floats.
pub fn device_product(a: Mat, b: Mat, trans_a: bool = false, trans_b: bool = false) -> Mat? {
    let left = device_view(a)
    if left is null { return null }
    let right = device_view(b)
    if right is null {
        left.free()
        return null
    }
    let product = jt.matmul(left, right, trans_a: trans_a, trans_b: trans_b)
    left.free()
    right.free()
    return mat_of(product)
}

#: The mean of the rows of `data` and their scatter about it -- the
#: covariance before it is divided by a count -- formed on the device: the
#: column totals by a reduction, then the centred rows' Gram matrix by a
#: GEMM. Null when `data` is not a plain run of floats.
#:
#: The device works in float32 where the host loop it replaces worked in
#: double. The scatter comes back exactly symmetric, as the eigen solver
#: expects.
pub fn device_scatter(data: Mat) -> tuple[list[float], list[float]]? {
    let rows = data.rows
    let width = data.cols
    let x = device_view(data)
    if x is null { return null }
    let turned = jt.transpose(x)
    let totals = jt.sum(turned, axis: -1)
    turned.free()
    let sums = totals.download()
    totals.free()
    var centre: list[float] = []
    var shift: list[float] = []
    for column in 0..width {
        let value = sums[column] / float(rows)
        centre.push(value)
        shift.push(-value)
    }
    let offset = Tensor([width], values: shift)
    let centred = jt.add(x, offset)
    offset.free()
    x.free()
    let gram = jt.matmul(centred, centred, trans_a: true)
    centred.free()
    let raw = gram.download()
    gram.free()
    var scatter: list[float] = []
    for i in 0..width {
        for j in 0..width { scatter.push((raw[i * width + j] + raw[j * width + i]) * 0.5) }
    }
    return (centre, scatter)
}

#: Each row of `data` less `centre`, times the transpose of `basis`, on the
#: device: the rows moved to the origin and written in the rows of `basis`.
#: Null when either is not a plain run of floats.
pub fn device_centred_product(data: Mat, centre: list[float], basis: Mat) -> Mat? {
    let x = device_view(data)
    if x is null { return null }
    let axes = device_view(basis)
    if axes is null {
        x.free()
        return null
    }
    var shift: list[float] = []
    for value in centre { shift.push(-value) }
    let offset = Tensor([centre.len()], values: shift)
    let centred = jt.add(x, offset)
    offset.free()
    x.free()
    let product = jt.matmul(centred, axes, trans_b: true)
    centred.free()
    axes.free()
    return mat_of(product)
}

#: Matrix product.
#:
#: On the device once it is big enough to be worth a dispatch: a host triple
#: loop took 183 ms to project 100,000 samples of fifty features onto ten.
pub fn matmul(a: Mat, b: Mat) -> Mat {
    if a.cols != b.rows { throw ValueError(f"cannot multiply {a} by {b}") }
    if on_device(a.rows * a.cols * b.cols) {
        let product = device_product(a, b)
        if product is not null { return product }
    }
    let left = a.values()
    let right = b.values()
    var out: list[float] = []
    for row in 0..a.rows {
        for column in 0..b.cols {
            var total = 0.0
            for index in 0..a.cols {
                total += left[row * a.cols + index] * right[index * b.cols + column]
            }
            out.push(total)
        }
    }
    return Mat.from_list(a.rows, b.cols, CV_MAKETYPE(CV_32F, 1), out)
}

#: Matrix transpose.
pub fn transpose(a: Mat) -> Mat {
    let values = a.to_list()
    var out: list[float] = []
    for column in 0..a.cols {
        for row in 0..a.rows { out.push(values[row * a.cols + column]) }
    }
    return Mat.from_list(a.cols, a.rows, CV_MAKETYPE(CV_32F, 1), out)
}

#: The identity, `size` by `size`.
pub fn identity(size: int) -> Mat {
    var out: list[float] = []
    for row in 0..size {
        for column in 0..size { out.push(if row == column { 1.0 } else { 0.0 }) }
    }
    return Mat.from_list(size, size, CV_MAKETYPE(CV_32F, 1), out)
}

#: Elementwise sum.
pub fn add(a: Mat, b: Mat) -> Mat {
    let left = a.values()
    let right = b.values()
    var out: list[float] = []
    for index in 0..left.len() { out.push(left[index] + right[index]) }
    return Mat.from_list(a.rows, a.cols, CV_MAKETYPE(CV_32F, 1), out)
}

#: Elementwise difference.
pub fn subtract(a: Mat, b: Mat) -> Mat {
    let left = a.values()
    let right = b.values()
    var out: list[float] = []
    for index in 0..left.len() { out.push(left[index] - right[index]) }
    return Mat.from_list(a.rows, a.cols, CV_MAKETYPE(CV_32F, 1), out)
}

#: Multiply every entry by `factor`.
pub fn scale(a: Mat, factor: float) -> Mat {
    var out: list[float] = []
    for value in a.to_list() { out.push(value * factor) }
    return Mat.from_list(a.rows, a.cols, CV_MAKETYPE(CV_32F, 1), out)
}

#: Matrix inverse, by Gauss-Jordan with partial pivoting.
pub fn inverse(a: Mat) -> Mat {
    if a.rows != a.cols { throw ValueError("only a square matrix can be inverted") }
    let n = a.rows
    let source = a.values()
    var work: list[list[float]] = []
    for row in 0..n {
        var line: list[float] = []
        for column in 0..n { line.push(source[row * n + column]) }
        for column in 0..n { line.push(if row == column { 1.0 } else { 0.0 }) }
        work.push(line)
    }
    for column in 0..n {
        var pivot = column
        for row in column..n {
            if math.fabs(work[row][column]) > math.fabs(work[pivot][column]) { pivot = row }
        }
        if math.fabs(work[pivot][column]) < 1e-14 { throw ValueError("this matrix has no inverse") }
        if pivot != column {
            let swap = work[pivot]
            work[pivot] = work[column]
            work[column] = swap
        }
        let divisor = work[column][column]
        for index in 0..2 * n { work[column][index] /= divisor }
        for row in 0..n {
            if row == column { continue }
            let factor = work[row][column]
            if factor == 0.0 { continue }
            for index in 0..2 * n { work[row][index] -= factor * work[column][index] }
        }
    }
    var out: list[float] = []
    for row in 0..n {
        for column in 0..n { out.push(work[row][n + column]) }
    }
    return Mat.from_list(n, n, CV_MAKETYPE(CV_32F, 1), out)
}

#: Solve `a x = b` for a square `a`.
pub fn solve(a: Mat, b: Mat) -> Mat { return matmul(inverse(a), b) }

#: Solve the least-squares problem `a x = b` through the normal equations.
pub fn solve_least_squares(a: Mat, b: Mat) -> Mat {
    let at = transpose(a)
    return solve(matmul(at, a), matmul(at, b))
}

#: The eigenvalues and eigenvectors of a symmetric matrix.
#:
#: Jacobi rotations: zero the largest off-diagonal entry, repeat. It converges
#: quadratically once the matrix is nearly diagonal and needs no shifts or
#: deflation, which for the sizes here is the right trade.
#:
#: Returns the eigenvalues in descending order and a matrix whose rows are the
#: matching eigenvectors.
pub fn symmetric_eigen(a: Mat) -> tuple[list[float], Mat] {
    if a.rows != a.cols { throw ValueError("only a square matrix has eigenvectors here") }
    let n = a.rows
    let (values, vectors) = symmetric_eigen_of(a.values(), n)
    return (values, Mat.from_list(n, n, CV_MAKETYPE(CV_32F, 1), vectors))
}

#: `symmetric_eigen` over a plain row-major list in double precision, with the
#: eigenvectors as rows of a row-major list.
#:
#: The rotations work on flat row-major lists, each step in a function of its
#: own: written as one body over lists of rows, it had more loops than the
#: tier records and ran interpreted, which a RANSAC calling it two thousand
#: times a fit could not afford. The arithmetic and its order are unchanged.
pub fn symmetric_eigen_of(source: list[float], n: int) -> tuple[list[float], list[float]] {
    var m: list[float] = []
    var v: list[float] = []
    _eigen_start(source, n, m, v)

    #: Enough rotations for the matrix in hand, not a hundred.
    #:
    #: Each turn of this loop zeroes ONE off-diagonal, so a hundred of them
    #: is a hundred rotations -- not a hundred sweeps. A 12 by 12 has
    #: sixty-six off-diagonals and classical Jacobi wants six to ten passes
    #: over them, so a hundred was about a sixth of the work. `pose_from_dlt`
    #: solves a 12 by 12. The loop still leaves the moment the largest
    #: off-diagonal is negligible, so this only raises the ceiling for the
    #: sizes that were reaching it.
    #:
    #: The pivot is still the largest off-diagonal, the first of equal ones in
    #: row order, but it is no longer found by scanning all of them: each row
    #: keeps where its own largest is, and a rotation of `p` and `q` only
    #: changes rows `p` and `q` and columns `p` and `q`, so only those entries
    #: are looked at again -- and a whole row only when its largest was one
    #: of them. Scanning everything per rotation made a 50 by 50 take 237 ms;
    #: the rotations, and so every value and vector, are the same as before.
    let sweeps = max(100, 12 * n * (n - 1) // 2)
    var best: list[int] = []
    for row in 0..n { best.push(_row_largest(m, n, row)) }
    for _sweep in 0..sweeps {
        if n < 2 { break }
        let at = _largest_tracked(m, n, best)
        if at < 0 { break }
        let p = at // n
        let q = at % n
        _jacobi_rotate(m, v, n, p, q)
        _track_rotation(m, n, best, p, q)
    }

    var order: list[int] = []
    for index in 0..n {
        var at = order.len()
        while at > 0 and m[index * n + index] > m[order[at - 1] * n + order[at - 1]] { at -= 1 }
        order.insert(at, index)
    }
    var values: list[float] = []
    var vectors: list[float] = []
    for index in order {
        values.push(m[index * n + index])
        for row in 0..n { vectors.push(v[row * n + index]) }
    }
    return (values, vectors)
}

#: Copy `source` into `m` and make `v` the identity, both flat.
fn _eigen_start(source: list[float], n: int, m: list[float], v: list[float]) -> void {
    for index in 0..n * n {
        m.push(source[index])
        let row = index // n
        v.push(if index - row * n == row { 1.0 } else { 0.0 })
    }
}

#: `math.fabs` without the call into `math`, for the pivot search's loops.
#: The same for every number but -0.0, which compares equal either way.
fn _magnitude(x: float) -> float { return if x < 0.0 { -x } else { x } }

#: The column of the largest entry right of the diagonal in `row` -- the
#: first of equal ones -- or -1 when there is none to pick.
fn _row_largest(m: list[float], n: int, row: int) -> int {
    var largest = -1.0
    var pick = -1
    for column in row + 1..n {
        let size = _magnitude(m[row * n + column])
        if size > largest {
            largest = size
            pick = column
        }
    }
    return pick
}

#: Where the largest off-diagonal entry above the diagonal is, as
#: `row * n + column` -- the first of equal ones in row order -- or -1 once it
#: is negligible, read from each row's largest.
fn _largest_tracked(m: list[float], n: int, best: list[int]) -> int {
    var largest = 0.0
    var p = 0
    var q = 1
    for row in 0..n {
        let column = best[row]
        if column < 0 { continue }
        let size = _magnitude(m[row * n + column])
        if size > largest {
            largest = size
            p = row
            q = column
        }
    }
    if largest < 1e-14 { return -1 }
    return p * n + q
}

#: Bring each row's largest up to date after rotating `p` and `q`.
fn _track_rotation(m: list[float], n: int, best: list[int], p: int, q: int) -> void {
    for row in 0..q {
        if row == p or best[row] == p or best[row] == q {
            best[row] = _row_largest(m, n, row)
            continue
        }
        var pick = best[row]
        var largest = if pick < 0 { -1.0 } else { _magnitude(m[row * n + pick]) }
        if p > row {
            let size = _magnitude(m[row * n + p])
            if size > largest or size == largest and p < pick {
                largest = size
                pick = p
            }
        }
        let size = _magnitude(m[row * n + q])
        if size > largest or size == largest and q < pick {
            largest = size
            pick = q
        }
        best[row] = pick
    }
    best[q] = _row_largest(m, n, q)
}

#: One Jacobi rotation zeroing `m[p][q]`, carried into the eigenvectors `v`.
fn _jacobi_rotate(m: list[float], v: list[float], n: int, p: int, q: int) -> void {
    let app = m[p * n + p]
    let aqq = m[q * n + q]
    let apq = m[p * n + q]
    let theta = 0.5 * math.atan2(2.0 * apq, app - aqq)
    let c = math.cos(theta)
    let s = math.sin(theta)
    let t = -s
    var kp = p
    var kq = q
    for _k in 0..n {
        let mkp = m[kp]
        let mkq = m[kq]
        m[kp] = c * mkp + s * mkq
        m[kq] = t * mkp + c * mkq
        kp += n
        kq += n
    }
    let pk = p * n
    let qk = q * n
    for k in 0..n {
        let mpk = m[pk + k]
        let mqk = m[qk + k]
        m[pk + k] = c * mpk + s * mqk
        m[qk + k] = t * mpk + c * mqk
    }
    kp = p
    kq = q
    for _k in 0..n {
        let vkp = v[kp]
        let vkq = v[kq]
        v[kp] = c * vkp + s * vkq
        v[kq] = t * vkp + c * vkq
        kp += n
        kq += n
    }
}

#: The direction that `a x` sends closest to zero.
#:
#: This is the right singular vector of the smallest singular value, found as
#: the eigenvector of `aᵀa` with the smallest eigenvalue. Every homogeneous fit
#: in geometry — a homography, a fundamental matrix, a conic — ends here.
pub fn null_space(a: Mat) -> list[float] {
    let turned = transpose(a)
    let normal = matmul(turned, a)
    turned.release()
    let (_values, vectors) = symmetric_eigen(normal)
    let n = normal.rows
    let flat = vectors.values()
    normal.release()
    vectors.release()
    var out: list[float] = []
    for column in 0..n { out.push(flat[(n - 1) * n + column]) }
    return out
}

#: `null_space` of a `rows` by `cols` row-major list, in double precision and
#: without a matrix: for a fit run thousands of times, such as every round of
#: a RANSAC, where each matrix is a device buffer to fill and read back.
pub fn null_space_of(a: list[float], rows: int, cols: int) -> list[float] {
    var normal: list[float] = []
    _gram(a, rows, cols, normal)
    let (_values, vectors) = symmetric_eigen_of(normal, cols)
    var out: list[float] = []
    for column in 0..cols { out.push(vectors[(cols - 1) * cols + column]) }
    return out
}

#: `aᵀa` of a `rows` by `cols` row-major list, appended to `normal` row-major.
fn _gram(a: list[float], rows: int, cols: int, normal: list[float]) -> void {
    for index in 0..cols * cols {
        let i = index // cols
        let j = index - i * cols
        var total = 0.0
        for r in 0..rows { total += a[r * cols + i] * a[r * cols + j] }
        normal.push(total)
    }
}

#: Singular value decomposition, as `(u, s, vt)` with `a = u diag(s) vt`.
#:
#: Built on `symmetric_eigen` rather than on a bidiagonalisation: the right
#: singular vectors are the eigenvectors of `aᵀa` and the singular values the
#: square roots of its eigenvalues, and `u` follows by projection. That is the
#: textbook route and it squares the condition number, which for the sizes this
#: is asked about — a three by three essential matrix, a nine-column design —
#: costs half the digits of a double and leaves plenty.
#:
#: The columns come back in decreasing order of singular value. A column whose
#: singular value is zero has no direction to project, so it is filled by
#: completing the basis, which is what makes `u` orthogonal even for a matrix
#: that is not full rank — the case every essential matrix is.
pub fn svd(a: Mat) -> tuple[Mat, list[float], Mat] {
    let rows = a.rows
    let columns = a.cols
    if rows < columns {
        #: Decompose the transpose and swap the two sides back, so the
        #: eigenproblem is always over the shorter dimension.
        let (u, s, vt) = svd(transpose(a))
        return (transpose(vt), s, transpose(u))
    }
    let (eigenvalues, vectors) = symmetric_eigen(matmul(transpose(a), a))
    var singular: list[float] = []
    for value in eigenvalues { singular.push(math.sqrt(max(value, 0.0))) }

    #: `symmetric_eigen` hands back its vectors a ROW each, which is OpenCV's
    #: convention and already the transpose this wants. Reading it the other
    #: way round rebuilds the matrix correctly and leaves `u` not orthogonal,
    #: which is the sort of wrong that only shows up two functions later.
    let vt_values = vectors.values()
    let source = a.values()
    var u: list[float] = []
    for _index in 0..rows * columns { u.push(0.0) }

    #: Relative, not absolute. A singular value a millionth of the largest is
    #: the square root of an eigenvalue that float32 could not tell from zero,
    #: and dividing by it does not recover a direction -- it amplifies the
    #: noise into one. Such a value is reported as the zero it is, which is
    #: also what makes a rank test on the result mean anything.
    let floor = singular[0] * 1e-6
    var established: list[bool] = []
    for _index in 0..columns { established.push(false) }
    for column in 0..columns {
        if singular[column] <= floor {
            singular[column] = 0.0
            continue
        }
        var projected: list[float] = []
        var length = 0.0
        for row in 0..rows {
            var total = 0.0
            for k in 0..columns {
                total += source[row * columns + k] * vt_values[column * columns + k]
            }
            let value = total / singular[column]
            projected.push(value)
            length += value * value
        }
        #: The projection must come out a unit vector. When it does not, the
        #: singular value it divided by was noise rather than a length -- which
        #: is what happens to a matrix BUILT to be rank deficient, where the
        #: eigenvalue that should be zero is a rounding error whose square root
        #: is large enough to pass any threshold set ahead of time. The column
        #: is completed below instead. `find_essential_mat` produces exactly
        #: such a matrix, and without this its translation came back as noise.
        length = math.sqrt(length)
        if math.fabs(length - 1.0) > 1e-3 {
            singular[column] = 0.0
            continue
        }
        for row in 0..rows { u[row * columns + column] = projected[row] }
        established[column] = true
    }

    #: A direction the matrix sends to nothing has nothing to project, so the
    #: column is completed instead: a trial axis, orthogonalised against every
    #: column already established, twice — once is not enough in float32 when
    #: the axis nearly lies in the span already.
    for column in 0..columns {
        if established[column] { continue }
        for axis in 0..rows {
            var candidate: list[float] = []
            for row in 0..rows { candidate.push(if row == axis { 1.0 } else { 0.0 }) }
            for _pass in 0..2 {
                for other in 0..columns {
                    if not established[other] { continue }
                    var dot = 0.0
                    for row in 0..rows { dot += candidate[row] * u[row * columns + other] }
                    for row in 0..rows { candidate[row] -= dot * u[row * columns + other] }
                }
            }
            var length = 0.0
            for value in candidate { length += value * value }
            length = math.sqrt(length)
            if length < 1e-3 { continue }
            for row in 0..rows { u[row * columns + column] = candidate[row] / length }
            established[column] = true
            break
        }
    }

    return (
        Mat.from_list(rows, columns, CV_MAKETYPE(CV_32F, 1), u),
        singular,
        Mat.from_list(columns, columns, CV_MAKETYPE(CV_32F, 1), vt_values)
    )
}

#: Fit parameters by Levenberg-Marquardt.
#:
#: `residual` returns how far the current parameters are from fitting, one
#: number per observation, and `jacobian` returns how each residual changes
#: with each parameter. The damping starts small and grows on a step that makes
#: things worse, which is what lets this start further from the answer than
#: Gauss-Newton could.
pub fn levenberg_marquardt(
    residual: fn(list[float]) -> list[float],
    jacobian: fn(list[float]) -> Mat,
    start: list[float],
    iterations: int = 40,
    epsilon: float = 1e-10
) -> list[float] {
    var parameters = [value for value in start]
    var damping = 1e-3
    var best = cost_of(residual(parameters))

    for _step in 0..iterations {
        let j = jacobian(parameters)
        let r = residual(parameters)
        let residual_matrix = Mat.from_list(r.len(), 1, CV_MAKETYPE(CV_32F, 1), r)
        let jt = transpose(j)
        let normal = matmul(jt, j)
        let gradient = matmul(jt, residual_matrix)

        var improved = false
        for _attempt in 0..12 {
            var damped = normal.to_list()
            let n = normal.rows
            for index in 0..n { damped[index * n + index] *= 1.0 + damping }
            var step: list[float] = []
            var solvable = true
            try {
                let solution = solve(Mat.from_list(n, n, CV_MAKETYPE(CV_32F, 1), damped), gradient)
                step = solution.to_list()
            } catch ValueError {
                solvable = false
            }
            if not solvable {
                damping *= 10.0
                continue
            }

            var candidate: list[float] = []
            for index in 0..parameters.len() { candidate.push(parameters[index] - step[index]) }
            let trial = cost_of(residual(candidate))
            if trial < best {
                #: A step that helped means the quadratic model is trustworthy
                #: here, so lean on it harder next time.
                parameters = candidate
                if best - trial < epsilon * max(best, 1e-12) {
                    best = trial
                    improved = true
                    break
                }
                best = trial
                damping = max(damping * 0.1, 1e-12)
                improved = true
                break
            }
            damping *= 10.0
        }
        if not improved { break }
    }
    return parameters
}

fn cost_of(residual: list[float]) -> float {
    var total = 0.0
    for value in residual { total += value * value }
    return total
}
