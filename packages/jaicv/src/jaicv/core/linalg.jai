#: Small dense linear algebra: the pieces geometry needs and nothing more.
#:
#: Everything here works on `Mat` and runs on the host. The matrices these are
#: given are three by three, nine by nine, or a Jacobian a few dozen wide —
#: sizes where a kernel launch costs more than the arithmetic, and where being
#: able to read the code matters more than the last microsecond.
#:
#: ```
#: let (values, vectors) = symmetric_eigen(a)
#: let h = null_space(design)
#: let fitted = levenberg_marquardt(residual, jacobian, start, 40)
#: ```
import std.math
from ..core.mat import Mat
from ..core.types import CV_32F, CV_MAKETYPE

#: Matrix product.
pub fn matmul(a: Mat, b: Mat) -> Mat {
    if a.cols != b.rows { throw ValueError(f"cannot multiply {a} by {b}") }
    let left = a.to_list()
    let right = b.to_list()
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
    let left = a.to_list()
    let right = b.to_list()
    var out: list[float] = []
    for index in 0..left.len() { out.push(left[index] + right[index]) }
    return Mat.from_list(a.rows, a.cols, CV_MAKETYPE(CV_32F, 1), out)
}

#: Elementwise difference.
pub fn subtract(a: Mat, b: Mat) -> Mat {
    let left = a.to_list()
    let right = b.to_list()
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
    let source = a.to_list()
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
    let source = a.to_list()
    var m: list[list[float]] = []
    for row in 0..n {
        var line: list[float] = []
        for column in 0..n { line.push(source[row * n + column]) }
        m.push(line)
    }
    var v: list[list[float]] = []
    for row in 0..n {
        var line: list[float] = []
        for column in 0..n { line.push(if row == column { 1.0 } else { 0.0 }) }
        v.push(line)
    }

    for _sweep in 0..100 {
        var largest = 0.0
        var p = 0
        var q = 1
        for row in 0..n {
            for column in row + 1..n {
                let size = math.fabs(m[row][column])
                if size > largest {
                    largest = size
                    p = row
                    q = column
                }
            }
        }
        if largest < 1e-14 or n < 2 { break }

        let app = m[p][p]
        let aqq = m[q][q]
        let apq = m[p][q]
        let theta = 0.5 * math.atan2(2.0 * apq, app - aqq)
        let c = math.cos(theta)
        let s = math.sin(theta)

        for k in 0..n {
            let mkp = m[k][p]
            let mkq = m[k][q]
            m[k][p] = c * mkp + s * mkq
            m[k][q] = -s * mkp + c * mkq
        }
        for k in 0..n {
            let mpk = m[p][k]
            let mqk = m[q][k]
            m[p][k] = c * mpk + s * mqk
            m[q][k] = -s * mpk + c * mqk
        }
        for k in 0..n {
            let vkp = v[k][p]
            let vkq = v[k][q]
            v[k][p] = c * vkp + s * vkq
            v[k][q] = -s * vkp + c * vkq
        }
    }

    var order: list[int] = []
    for index in 0..n {
        var at = order.len()
        while at > 0 and m[index][index] > m[order[at - 1]][order[at - 1]] { at -= 1 }
        order.insert(at, index)
    }
    var values: list[float] = []
    var vectors: list[float] = []
    for index in order {
        values.push(m[index][index])
        for row in 0..n { vectors.push(v[row][index]) }
    }
    return (values, Mat.from_list(n, n, CV_MAKETYPE(CV_32F, 1), vectors))
}

#: The direction that `a x` sends closest to zero.
#:
#: This is the right singular vector of the smallest singular value, found as
#: the eigenvector of `aᵀa` with the smallest eigenvalue. Every homogeneous fit
#: in geometry — a homography, a fundamental matrix, a conic — ends here.
pub fn null_space(a: Mat) -> list[float] {
    let normal = matmul(transpose(a), a)
    let (_values, vectors) = symmetric_eigen(normal)
    let n = normal.rows
    let flat = vectors.to_list()
    var out: list[float] = []
    for column in 0..n { out.push(flat[(n - 1) * n + column]) }
    return out
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
