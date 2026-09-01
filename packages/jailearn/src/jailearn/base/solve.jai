#: The small dense solves: a Cholesky factor, substitution, and a diagonal.
import std.math

from jaitensor import Tensor

from .kernels import base_kernel
from .reduce import copy_into

#: The lower-triangular `L` with `L LT = a`.
#:
#: For the normal equations and for a full-covariance Gaussian, both square in
#: the features or the components and never in the samples. Column by column,
#: two dispatches each because there is no barrier across a grid; a matrix that
#: is not positive definite is caught when the diagonal comes back with a NaN.
pub fn cholesky_factor(a: Tensor) -> Tensor {
    let order = square_order(a)
    let l = Tensor.zeros([order, order])
    let step = base_kernel("base_chol_step")
    for column in 0..order {
        step.dispatch_async([a.buffer(), l.buffer()], 1, [order, column, 0])
        let below = order - column - 1
        if below > 0 { step.dispatch_async([a.buffer(), l.buffer()], below, [order, column, 1]) }
    }
    let pivots = diagonal(l)
    let values = pivots.download()
    pivots.free()
    for (index, value) in values.enumerate() {
        if math.is_nan(value) or value <= 0.0 {
            l.free()
            throw ValueError(f"the matrix is not positive definite; pivot {index} is {value}")
        }
    }
    return l
}

#: Solve `L x = b`, or `LT x = b` when `transposed`. `b` may be `[order]` or
#: `[order, n_rhs]`, and the answer keeps its shape.
pub fn triangular_solve(l: Tensor, b: Tensor, transposed: bool = false) -> Tensor {
    let order = square_order(l)
    if b.rank() < 1 or b.rank() > 2 {
        throw ValueError(f"b must be a vector or a matrix, got {b.shape}")
    }
    if b.shape[0] != order {
        throw ValueError(f"b has {b.shape[0]} rows against an order-{order} factor")
    }
    let rhs = if b.rank() == 1 { 1 } else { b.shape[1] }
    let x = Tensor(b.shape)
    copy_into(b, x, b.size)
    let solve = base_kernel("base_tri_row")
    let upper = if transposed { 1 } else { 0 }
    for step in 0..order {
        let row = if transposed { order - 1 - step } else { step }
        solve.dispatch_async([l.buffer(), x.buffer()], rhs, [order, rhs, row, upper])
    }
    return x
}

#: Solve `a x = b` for a symmetric positive definite `a`.
pub fn cholesky_solve(a: Tensor, b: Tensor) -> Tensor {
    let l = cholesky_factor(a)
    let forward = triangular_solve(l, b, false)
    let x = triangular_solve(l, forward, true)
    l.free()
    forward.free()
    return x
}

#: The diagonal of a square matrix, as `[order]`.
pub fn diagonal(m: Tensor) -> Tensor {
    let order = square_order(m)
    let out = Tensor([order])
    copy_into(m, out, order, stride: order + 1)
    return out
}

fn square_order(m: Tensor) -> int {
    if m.rank() != 2 or m.shape[0] != m.shape[1] {
        throw ValueError(f"expected a square matrix, got {m.shape}")
    }
    if m.shape[0] == 0 { throw ValueError("the matrix is empty") }
    return m.shape[0]
}
