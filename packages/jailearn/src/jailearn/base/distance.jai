#: Pairwise squared distances, and the four kernel matrices built on them.
import jaitensor as jt

from jaitensor import Tensor
from std.gpu import Buffer

from .kernels import base_kernel
from .params import check_matrix
from std.os import env_or

from .reduce import DIRECT_WIDTH, row_topk, row_topk_into, topk_one_pass

#: `JAILEARN_NEAREST_EXACT=0` returns `nearest_rows` to the expanded form
#: alone: no term-by-term search for narrow rows, and no exact distances for
#: the rows a wide search picks.
let _NEAREST_EXACT = env_or("JAILEARN_NEAREST_EXACT", "1") != "0"

#: Whether the exact search is on (`JAILEARN_NEAREST_EXACT`); a neighbour
#: index that centres and refines its own search asks this first.
pub fn exact_search() -> bool { return _NEAREST_EXACT }

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

#: Queries per band in `nearest_rows` are chosen so a band's product holds
#: about this many floats (256 MB), whatever the size of `b`.
pub const NEAREST_BAND_FLOATS: int = 67108864

#: The `k` rows of `b` nearest each row of `a`: `(squared distances,
#: indices)`, both `[n_a, k]`, nearest first, equal distances by the smaller
#: index.
#:
#: Rows of at most `DIRECT_WIDTH` features are searched term by term, one
#: pass that ranks `sum((a_i - b_j)^2)` directly. Wider rows take one GEMM per
#: band and the expanded form `|a|^2 + |b|^2 - 2 a.b`, ranked as the top-k
#: reads the product, so the distance matrix is never finished by a separate
#: pass and `a` is taken in bands of rows sized by `NEAREST_BAND_FLOATS`. The
#: expanded form is only good to an ulp of the squared norms, so the distances
#: of the rows it picks are then recomputed term by term and re-sorted
#: (`refine_nearest`): an exact duplicate comes back at zero. What the
#: expanded form can still get wrong is WHICH rows: two rows closer together
#: than its rounding may be picked out of order. Data far from the origin
#: makes that rounding large; `KNeighborsClassifier` centres both sides first,
#: which is what removes it.
#:
#: `exact: false` (or `JAILEARN_NEAREST_EXACT=0`) is the expanded form alone,
#: which is `row_topk(pairwise_sqdist(a, b), k, largest: false)` exactly. A
#: `k` past `TOPK_ONE_PASS_MAX` takes that composition, band by band.
#: `band_floats` overrides the band size. A query holding a NaN or an
#: infinity has no neighbours under the exact search: its row is padded with
#: index `n_b` and distance infinity.
pub fn nearest_rows(
    a: Tensor,
    b: Tensor,
    k: int,
    band_floats: int = NEAREST_BAND_FLOATS,
    exact: bool = true
) -> tuple[Tensor, Tensor] {
    check_matrix(a, "a")
    check_matrix(b, "b")
    if a.shape[1] != b.shape[1] {
        throw ValueError(f"a has {a.shape[1]} features against b's {b.shape[1]}")
    }
    let queries = a.shape[0]
    let points = b.shape[0]
    if k <= 0 or k > points { throw ValueError(f"cannot take {k} of {points} rows") }
    let values = Tensor([queries, k])
    let index = Tensor([queries, k])
    let one_pass = topk_one_pass(k)
    let refine = exact and _NEAREST_EXACT
    if refine and one_pass and a.shape[1] <= DIRECT_WIDTH {
        row_topk_into(
            b.buffer(),
            queries,
            points,
            k,
            false,
            values,
            index,
            left: a.buffer(),
            direct_width: a.shape[1]
        )
        return (values, index)
    }
    let band = max(1, min(queries, band_floats // max(points, 1)))
    let left = row_sqnorm(a)
    let right = row_sqnorm(b)
    let product = Tensor([band, points])
    var first = 0
    while first < queries {
        let rows = min(band, queries - first)
        let block = a.view(first * a.shape[1], [rows, a.shape[1]])
        let target = product.view(0, [rows, points])
        jt.matmul(block, b, trans_b: true, out: target)
        if one_pass {
            row_topk_into(
                target.buffer(),
                rows,
                points,
                k,
                false,
                values,
                index,
                first: first,
                left: left.buffer(),
                right: right.buffer()
            )
        } else {
            base_kernel("base_dist_finish").dispatch_async(
                [target.buffer(), left.buffer(), right.buffer()],
                rows * points,
                [rows, points, first]
            )
            let (near, at) = row_topk(target, k, largest: false)
            copy_rows(near, values, first * k)
            copy_rows(at, index, first * k)
            near.free()
            at.free()
        }
        block.free()
        target.free()
        first += rows
    }
    product.free()
    left.free()
    right.free()
    if refine { refine_nearest(a, b, values, index) }
    return (values, index)
}

#: Recompute, term by term, the squared distances of the neighbours a search
#: of `b` for each row of `a` picked, and re-sort each row of `(values,
#: index)` by them, equal distances by the smaller index. Both are
#: `[n_a, k]` and are rewritten in place.
pub fn refine_nearest(a: Tensor, b: Tensor, values: Tensor, index: Tensor) -> void {
    let k = values.shape[1]
    base_kernel("base_nearest_refine").dispatch_async(
        [a.buffer(), b.buffer(), values.buffer(), index.buffer()],
        a.shape[0],
        [a.shape[0], a.shape[1], k, b.shape[0]]
    )
}

fn copy_rows(part: Tensor, whole: Tensor, offset: int) -> void {
    let window = whole.view(offset, [part.size])
    base_kernel("base_copy").dispatch_async(
        [part.buffer(), window.buffer()],
        part.size,
        [part.size, 1]
    )
    window.free()
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
