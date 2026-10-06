#: The small dense solves: a Cholesky factor, substitution, and a diagonal.
import std.math
from std.os import env_or

from jaitensor import Tensor

from .kernels import base_kernel
from .reduce import copy_into

#: The largest order factored and substituted on the host.
#:
#: The device route is a chain of dependent dispatches -- two per column for
#: the factor and one per row for each substitution -- so a 50x50 normal
#: equation cost about 200 of them, 1.83 ms, while the arithmetic is 21k
#: multiply-adds and the system is 10 KB to read back. Up to this order the
#: read back plus a host loop in double precision wins; past it the cubic
#: host loop does not, and the device route stays.
pub const HOST_SOLVE_ORDER: int = 256

#: `JAILEARN_HOST_SOLVE=0` keeps every factor and substitution on the device.
let _HOST_SOLVE = env_or("JAILEARN_HOST_SOLVE", "1") != "0"

fn on_host(order: int) -> bool { return _HOST_SOLVE and order <= HOST_SOLVE_ORDER }

#: The lower-triangular `L` with `L LT = a`.
#:
#: For the normal equations and for a full-covariance Gaussian, both square in
#: the features or the components and never in the samples. A small system is
#: factored on the host (see `HOST_SOLVE_ORDER`). A larger one runs column by
#: column on the device, two dispatches each because there is no barrier
#: across a grid; a matrix that is not positive definite is caught when the
#: diagonal comes back with a NaN.
pub fn cholesky_factor(a: Tensor) -> Tensor {
    let order = square_order(a)
    if on_host(order) {
        let factor = host_factor(a.download(), order)
        return Tensor([order, order], factor)
    }
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
            refuse(index, value)
        }
    }
    return l
}

fn refuse(index: int, value: float) -> void {
    throw ValueError(f"the matrix is not positive definite; pivot {index} is {value}")
}

#: The factor of a row-major `order x order` matrix, read from its lower
#: triangle only, as a row-major list with zeros above the diagonal.
#:
#: The first pivot that is not positive refuses, with the value the device
#: route would have reported for it: the root of a negative is a NaN.
fn host_factor(a: list[float], order: int) -> list[float] {
    var l = [0.0 for _slot in 0..order * order]
    for column in 0..order {
        let head = column * order
        var total = a[head + column]
        for k in 0..column {
            let seen = l[head + k]
            total -= seen * seen
        }
        if math.is_nan(total) or total <= 0.0 {
            refuse(column, if total == 0.0 { 0.0 } else { math.NAN })
        }
        let pivot = math.sqrt(total)
        l[head + column] = pivot
        for row in column + 1..order {
            let base = row * order
            var rest = a[base + column]
            for k in 0..column { rest -= l[base + k] * l[head + k] }
            l[base + column] = rest / pivot
        }
    }
    return l
}

#: Substitute through a row-major factor in place: `L x = b`, or `LT x = b`
#: when `transposed`. `x` holds `rhs` interleaved right-hand sides, row-major.
fn host_substitute(l: list[float], x: list[float], order: int, rhs: int, transposed: bool) -> void {
    for step in 0..order {
        let row = if transposed { order - 1 - step } else { step }
        let pivot = l[row * order + row]
        for column in 0..rhs {
            var total = x[row * rhs + column]
            if transposed {
                for k in row + 1..order { total -= l[k * order + row] * x[k * rhs + column] }
            } else {
                let base = row * order
                for k in 0..row { total -= l[base + k] * x[k * rhs + column] }
            }
            x[row * rhs + column] = total / pivot
        }
    }
}

fn rhs_count(b: Tensor, order: int) -> int {
    if b.rank() < 1 or b.rank() > 2 {
        throw ValueError(f"b must be a vector or a matrix, got {b.shape}")
    }
    if b.shape[0] != order {
        throw ValueError(f"b has {b.shape[0]} rows against an order-{order} factor")
    }
    return if b.rank() == 1 { 1 } else { b.shape[1] }
}

#: Solve `L x = b`, or `LT x = b` when `transposed`. `b` may be `[order]` or
#: `[order, n_rhs]`, and the answer keeps its shape.
pub fn triangular_solve(l: Tensor, b: Tensor, transposed: bool = false) -> Tensor {
    let order = square_order(l)
    let rhs = rhs_count(b, order)
    if on_host(order) {
        var x = b.download()
        host_substitute(l.download(), x, order, rhs, transposed)
        return Tensor(b.shape, x)
    }
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
#:
#: A small system is read back once and factored and substituted on the host
#: in double precision, so the answer crosses back to the device once.
pub fn cholesky_solve(a: Tensor, b: Tensor) -> Tensor {
    let order = square_order(a)
    let rhs = rhs_count(b, order)
    if on_host(order) {
        let l = host_factor(a.download(), order)
        var x = b.download()
        host_substitute(l, x, order, rhs, false)
        host_substitute(l, x, order, rhs, true)
        return Tensor(b.shape, x)
    }
    let l = cholesky_factor(a)
    let forward = triangular_solve(l, b, false)
    let x = triangular_solve(l, forward, true)
    l.free()
    forward.free()
    return x
}

#: Solve `a x = b` on the host for a symmetric positive definite `a` given as
#: a row-major list, for callers that already hold the system there.
#:
#: `damping` is added to the diagonal first, which is how an iterative fit
#: keeps a Hessian that fp32 accumulation left barely definite solvable.
pub fn host_cholesky_solve(
    a: list[float],
    b: list[float],
    order: int,
    damping: float = 0.0
) -> list[float] {
    if a.len() != order * order { throw ValueError(f"a has {a.len()} values for order {order}") }
    if b.len() % order != 0 { throw ValueError(f"b has {b.len()} values for order {order}") }
    var system = a
    if damping != 0.0 {
        system = [value for value in a]
        for index in 0..order { system[index * order + index] += damping }
    }
    let l = host_factor(system, order)
    var x = [value for value in b]
    let rhs = b.len() // order
    host_substitute(l, x, order, rhs, false)
    host_substitute(l, x, order, rhs, true)
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
