#: Pairwise squared distances, and the four kernel matrices built on them.
import jaitensor as jt

from jaitensor import Tensor
from std.gpu import Buffer

from .kernels import base_kernel
from .params import check_matrix

#: The squared length of every row. jaitensor's row reduction rather than a
#: kernel here: it puts a threadgroup on each row, which matters as soon as the
#: features outnumber a simdgroup, and it is already tuned.
fn row_sqnorm(x: Tensor) -> Tensor {
    let squared = jt.square(x)
    let out = jt.sum(squared, axis: -1)
    squared.free()
    return out
}

#: Squared euclidean distance from rows of `a` to every row of `b`.
#:
#: `|a|^2 + |b|^2 - 2 a.b`, so the cost is one GEMM and not the triple loop the
#: definition suggests. The product is written straight into the answer, so the
#: distance array exists once; `a.bT` into a temporary and a second pass over
#: it would hold two of them at the same moment, which is where a kNN over a
#: hundred thousand samples runs out of device memory.
#:
#: `first` and `rows` take one horizontal band instead of the whole matrix.
#: That is the form a neighbour search, DBSCAN or t-SNE wants -- `n^2` is never
#: materialised, the caller reduces each band before asking for the next --
#: while the default computes all of it.
pub fn pairwise_sqdist(a: Tensor, b: Tensor, first: int = 0, rows: int = 0) -> Tensor {
    check_matrix(a, "a")
    check_matrix(b, "b")
    if a.shape[1] != b.shape[1] {
        throw ValueError(f"a has {a.shape[1]} features against b's {b.shape[1]}")
    }
    let band = if rows > 0 { rows } else { a.shape[0] - first }
    if first < 0 or band <= 0 or first + band > a.shape[0] {
        throw IndexError(f"rows {first}..{first + band} run past a's {a.shape[0]} samples")
    }
    let out = Tensor([band, b.shape[0]])
    let block = a.view(first * a.shape[1], [band, a.shape[1]])
    let product = jt.matmul(block, b, trans_b: true, out: out)
    let left = row_sqnorm(a)
    let right = row_sqnorm(b)
    base_kernel("base_dist_finish").dispatch_async(
        [product.buffer(), left.buffer(), right.buffer()],
        out.size,
        [band, b.shape[0], first]
    )
    left.free()
    right.free()
    block.free()
    return out
}

#: `k(a_i, b_j) = a_i . b_j`.
pub const GRAM_LINEAR: int = 0
#: `k(a_i, b_j) = exp(-gamma |a_i - b_j|^2)`.
pub const GRAM_RBF: int = 1
#: `k(a_i, b_j) = (gamma a_i . b_j + coef0)^degree`.
pub const GRAM_POLY: int = 2
#: `k(a_i, b_j) = tanh(gamma a_i . b_j + coef0)`.
pub const GRAM_SIGMOID: int = 3

#: The kernel matrix between two sets of rows, `[n_a, n_b]`.
#:
#: Everything kernelised in the package -- the SVMs, kernel ridge, kernel PCA,
#: spectral clustering -- comes through here, so the four kernels are written
#: once and the estimators only pick a `kind`.
pub fn gram(
    a: Tensor,
    b: Tensor,
    kind: int = GRAM_LINEAR,
    gamma: float = 1.0,
    degree: float = 3.0,
    coef0: float = 0.0
) -> Tensor {
    if kind < GRAM_LINEAR or kind > GRAM_SIGMOID { throw ValueError(f"unknown gram kind {kind}") }
    if kind == GRAM_RBF {
        let distances = pairwise_sqdist(a, b)
        shape_gram(distances, kind, gamma, degree, coef0)
        return distances
    }
    check_matrix(a, "a")
    check_matrix(b, "b")
    if a.shape[1] != b.shape[1] {
        throw ValueError(f"a has {a.shape[1]} features against b's {b.shape[1]}")
    }
    let product = jt.matmul(a, b, trans_b: true)
    if kind != GRAM_LINEAR { shape_gram(product, kind, gamma, degree, coef0) }
    return product
}

fn shape_gram(m: Tensor, kind: int, gamma: float, degree: float, coef0: float) -> void {
    let shape = Buffer.of([gamma, degree, coef0])
    base_kernel("base_gram_finish").dispatch_async([m.buffer(), shape], m.size, [m.size, kind])
    shape.free()
}
