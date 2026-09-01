#: The vocabulary every jailearn estimator speaks, and the device primitives
#: they are built out of.
#:
#: One data convention holds everywhere in the package: `X` is a
#: `jaitensor.Tensor` of shape `[n_samples, n_features]`, `y` is `[n_samples]`,
#: both float32 and resident on the GPU, and class labels travel as float codes
#: rather than as integers or strings. `LabelIndex` maps whatever codes a
#: caller used onto `0..k-1` and back, and every classifier, `StratifiedKFold`
#: and the metrics go through it.
#:
#: The other rule this module exists to enforce is that no fit walks its
#: samples on the host, where a loop over more than a few thousand elements
#: costs seconds. Column statistics, sorting, distances, class totals, seeding
#: and the small dense solves are all kernels below, general enough that the
#: estimators compose them instead of writing their own.
#:
#: ```
#: import jailearn as jl
#: let x = jl.matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
#: defer { x.free() }
#: let (mean, variance) = jl.column_moments(x)
#: defer { mean.free() }
#: defer { variance.free() }
#: jl.to_values(mean)    # [2.0, 3.0]
#: ```
import jaitensor as jt
import std.math
from jaitensor import Tensor
from std.algo.sort import merge_sort
from std.random import Random
from .base.kernels import base_kernel, cached_kernel
from .base.params import Params, check_fitted, check_matrix, check_pair, check_width
from .base.params import param_bool, param_float, param_int, param_str, params_of
from .base.params import require_target
from .base.data import column, hstack, index_tensor, matrix, select_columns, take_rows
from .base.data import to_rows, to_values, vector
from .base.reduce import argsort_columns, column_min_max, column_moments, column_quantiles
from .base.reduce import copy_into, log_sum_exp_rows, masked_column_mean, row_argmax
from .base.reduce import row_argmin, row_extreme, row_topk, sort_columns
from .base.distance import GRAM_LINEAR, GRAM_POLY, GRAM_RBF, GRAM_SIGMOID, gram
from .base.distance import pairwise_sqdist
from .base.classes import bincount, class_sums, class_sumsq, scatter_add_rows

export { cached_kernel }
export { Params, check_fitted, check_matrix, check_pair, check_width, param_bool }
export { param_float, param_int, param_str, params_of, require_target }
export { column, hstack, index_tensor, matrix, select_columns, take_rows, to_rows }
export { to_values, vector }
export { argsort_columns, column_min_max, column_moments, column_quantiles }
export { log_sum_exp_rows, masked_column_mean, row_argmax, row_argmin, row_extreme }
export { row_topk, sort_columns }
export { GRAM_LINEAR, GRAM_POLY, GRAM_RBF, GRAM_SIGMOID, gram, pairwise_sqdist }
export { bincount, class_sums, class_sumsq, scatter_add_rows }

#: How `model_selection` measures a fitted estimator: higher is better, always.
#: It takes one of these rather than importing `metrics`, so a search can run
#: against a score that does not exist yet.
pub type Scorer = fn(Estimator, Tensor, Tensor) -> float

#: What an estimator produces, which decides which optional methods it has to
#: implement and what `default_score` means for it.
pub enum EstimatorKind {
    #: `predict` returns class codes; `score` is accuracy.
    Classifier,
    #: `predict` returns real values; `score` is the coefficient of determination.
    Regressor,
    #: `transform` maps `X` to another matrix; `score` has no meaning.
    Transformer,
    #: `fit` assigns each sample a cluster and `predict` returns it.
    Clusterer,
    #: `score_samples` returns a log density; `score` has no meaning.
    Density,
}

#: Everything in jailearn that learns from data.
#:
#: The seven required methods are the ones no estimator can do without. The
#: rest are defaulted, and a default a subclass has no business supporting
#: throws `RuntimeError` rather than returning something plausible -- the
#: choice `jaitensor.Layer.static_forward` makes, for the same reason: a
#: regressor that quietly answered `predict_proba` would be found out much
#: later than one that refuses.
#:
#: An estimator owns device tensors, so it needs `free`; freeing twice is
#: harmless, and `defer { model.free() }` beside the constructor is the habit
#: every example in the package follows.
pub trait Estimator {
    #: What this estimator produces.
    fn kind(self) -> EstimatorKind

    #: Learn from `x`, and from `y` when the estimator is supervised.
    fn fit(self, x: Tensor, y: Tensor? = null) -> void

    #: The constructor arguments, under the names `set_params` accepts.
    fn get_params(self) -> Params

    #: Change constructor arguments. Fitted state is not carried over; a search
    #: sets parameters on an unfitted clone.
    fn set_params(self, params: Params) -> void

    #: An unfitted estimator with the same parameters and no device memory of
    #: its own. This is what cross-validation refits, once per fold.
    fn clone_unfitted(self) -> Estimator

    #: Whether `fit` has run.
    fn is_fitted(self) -> bool

    #: Release the device tensors this estimator owns.
    fn free(self) -> void

    #: Map `x` into this estimator's output space.
    fn transform(self, x: Tensor) -> Tensor {
        throw RuntimeError("this estimator does not transform")
    }

    #: Map a transformed matrix back, as far as the transform allows.
    fn inverse_transform(self, x: Tensor) -> Tensor {
        throw RuntimeError("this estimator does not invert its transform")
    }

    #: Fit and transform together. Estimators that already hold the answer at
    #: the end of `fit` -- anything with a factorisation in hand -- override it.
    fn fit_transform(self, x: Tensor, y: Tensor? = null) -> Tensor {
        self.fit(x, y)
        return self.transform(x)
    }

    #: One prediction per row of `x`, shape `[n_samples]`.
    fn predict(self, x: Tensor) -> Tensor {
        throw RuntimeError("this estimator does not predict")
    }

    #: Class probabilities, shape `[n_samples, n_classes]`.
    fn predict_proba(self, x: Tensor) -> Tensor {
        throw RuntimeError("this estimator does not produce probabilities")
    }

    #: Unnormalised scores: `[n_samples]` for two classes, `[n_samples,
    #: n_classes]` beyond.
    fn decision_function(self, x: Tensor) -> Tensor {
        throw RuntimeError("this estimator does not expose a decision function")
    }

    #: The width `fit` saw, or 0 when the estimator records none. `check_width`
    #: reads 0 as "unknown" and lets the call through.
    fn feature_count(self) -> int { return 0 }

    #: Accuracy for a classifier, the coefficient of determination for a
    #: regressor, a refusal for anything else.
    fn score(self, x: Tensor, y: Tensor) -> float { return default_score(self, x, y) }
}

#: The score `Estimator.score` falls back to.
#:
#: Accuracy compares class codes for exact equality, which is safe only because
#: a classifier's `predict` returns values `LabelIndex.decode` copied out of the
#: class table rather than values it computed.
pub fn default_score(estimator: Estimator, x: Tensor, y: Tensor) -> float {
    check_pair(x, y)
    let kind = estimator.kind()
    if kind != EstimatorKind.Classifier and kind != EstimatorKind.Regressor {
        throw RuntimeError(f"{kind} has no default score; pass a Scorer")
    }
    let predicted = estimator.predict(x)
    defer { predicted.free() }
    if predicted.size != y.size {
        throw ValueError(f"predict returned {predicted.size} values for {y.size} samples")
    }
    if kind == EstimatorKind.Classifier {
        return float(count_equal(predicted, y)) / float(y.size)
    }
    let residual = jt.subtract(predicted, y)
    let squared = jt.square(residual)
    let unexplained = sum_to_host(squared)
    residual.free()
    squared.free()
    #: The total sum of squares comes from `column_moments` and not from
    #: `sum(y*y) - sum(y)^2/n`, which loses most of its digits when the mean is
    #: large next to the spread. House prices in dollars are the usual way to
    #: watch an R^2 of 0.97 come out as 0.2.
    let column = y.view(0, [y.size, 1])
    let (mean, variance) = column_moments(column)
    let spread = to_values(variance)[0] * float(y.size)
    mean.free()
    variance.free()
    column.free()
    if spread <= 0.0 { return 0.0 }
    return 1.0 - unexplained / spread
}

#: `x[i][j] * scale[j] + offset[j]`, which is every scaler's apply step.
pub fn affine_columns(x: Tensor, scale: Tensor, offset: Tensor) -> Tensor {
    check_matrix(x)
    let width = x.shape[1]
    if scale.size != width or offset.size != width {
        throw ValueError(
            f"scale and offset must be [{width}], got {scale.shape} and {offset.shape}"
        )
    }
    let out = Tensor(x.shape)
    base_kernel("base_affine_columns").dispatch_async(
        [x.buffer(), scale.buffer(), offset.buffer(), out.buffer()],
        x.size,
        [x.size, width]
    )
    return out
}

#: How many positions hold the same value in both tensors.
#:
#: Counted by summing a mask in float32, so the answer is exact up to about
#: sixteen million matches and then starts rounding. Accuracy over a test set
#: that size is not the measurement anybody is taking.
pub fn count_equal(a: Tensor, b: Tensor) -> int {
    if a.size != b.size { throw ValueError(f"{a.shape} and {b.shape} have different sizes") }
    let mask = Tensor([a.size])
    base_kernel("base_mask_op").dispatch_async(
        [a.buffer(), b.buffer(), mask.buffer()],
        a.size,
        [a.size, 0]
    )
    let total = sum_to_host(mask)
    mask.free()
    return int(total + 0.5)
}

#: 1 where `x` holds a NaN and 0 elsewhere, same shape as `x`.
pub fn nan_mask(x: Tensor) -> Tensor {
    let mask = Tensor(x.shape)
    base_kernel("base_mask_op").dispatch_async(
        [x.buffer(), x.buffer(), mask.buffer()],
        x.size,
        [x.size, 1]
    )
    return mask
}

fn sum_to_host(t: Tensor) -> float {
    let total = jt.sum(t)
    let value = total.download()[0]
    total.free()
    return value
}

#: `k` starting centres spread out by k-means++, as `[k, n_features]`.
#:
#: The first is uniform; each after it is drawn with probability proportional
#: to its squared distance from the nearest centre already chosen. Distances
#: and draw are both dispatches, so the host loop is over `k` and nothing else.
#: Clustering and the Gaussian mixtures both start here.
pub fn kmeanspp_seed(x: Tensor, k: int, seed: int = 0) -> Tensor {
    check_matrix(x)
    let rows = x.shape[0]
    let width = x.shape[1]
    if k <= 0 or k > rows { throw ValueError(f"{rows} samples cannot make {k} centres") }
    let centres = Tensor([k, width])
    let rng = rng_for(seed)
    copy_row(x, centres, rng.int_range(0, rows), 0, width)
    let nearest = Tensor([rows])
    let keys = Tensor([rows])
    let distance = base_kernel("base_seed_distance")
    let key_step = base_kernel("base_seed_keys")
    let salt = device_seed(seed, 0x5DEECE66)
    distance.dispatch_async(
        [x.buffer(), centres.buffer(), nearest.buffer()],
        rows,
        [rows, width, 0, 1]
    )
    for centre in 1..k {
        key_step.dispatch_async([nearest.buffer(), keys.buffer()], rows, [rows, salt, centre])
        let chosen = int(peak_index(keys) + 0.5)
        copy_row(x, centres, chosen, centre, width)
        distance.dispatch_async(
            [x.buffer(), centres.buffer(), nearest.buffer()],
            rows,
            [rows, width, centre, 0]
        )
    }
    nearest.free()
    keys.free()
    return centres
}

#: Where the largest value of a vector sits.
#:
#: Through jaitensor's row reduction, which puts a threadgroup on the vector:
#: `row_extreme` over a one-row matrix would put a single thread on it, and a
#: single thread walking a hundred thousand samples is the mistake this module
#: exists to keep out of the estimators.
fn peak_index(v: Tensor) -> float {
    let row = v.view(0, [1, v.size])
    let (values, indices) = jt.max_with_index(row)
    let at = indices.download()[0]
    values.free()
    indices.free()
    row.free()
    return at
}

fn copy_row(source: Tensor, destination: Tensor, taken: int, placed: int, width: int) -> void {
    let src = source.view(taken * width, [width])
    let dst = destination.view(placed * width, [width])
    copy_into(src, dst, width)
    src.free()
    dst.free()
}

#: The generator an estimator's `random_state` names.
#:
#: Seed 0 means unseeded, and it means that everywhere in the package: the
#: device fills below draw their own seed from a fresh generator when they get
#: a 0, rather than inheriting `Buffer.fill_uniform`'s habit of turning it into
#: a constant and handing back the same numbers every run.
pub fn rng_for(seed: int) -> Random {
    if seed == 0 { return Random() }
    return Random.with_seed(seed)
}

fn mix_seed(seed: int, salt: int) -> int {
    return seed *% 6364136223846793 +% salt *% 1442695040888963 & 0xFFFFFFFF
}

fn device_seed(seed: int, salt: int) -> int {
    return mix_seed(if seed == 0 { Random().next_int() } else { seed }, salt)
}

#: A tensor of uniform draws in `[lo, hi)`, filled on the device.
pub fn random_uniform(shape: list[int], lo: float = 0.0, hi: float = 1.0, seed: int = 0) -> Tensor {
    let out = Tensor(shape)
    out.buffer().fill_uniform(lo, hi, device_seed(seed, 0x27D4EB2F))
    return out
}

#: A tensor of standard normal draws, Box-Muller over two uniform fills.
#:
#: The first fill starts just above zero rather than at it: `log(0)` is an
#: infinity, and one of those in an initial weight matrix is a model that never
#: trains.
pub fn random_normal(shape: list[int], seed: int = 0) -> Tensor {
    let out = Tensor(shape)
    let first = Tensor(shape)
    let second = Tensor(shape)
    first.buffer().fill_uniform(1.0e-7, 1.0, device_seed(seed, 0x9E3779B9))
    second.buffer().fill_uniform(0.0, 1.0, device_seed(seed, 0x85EBCA6B))
    base_kernel("base_box_muller").dispatch_async(
        [first.buffer(), second.buffer(), out.buffer()],
        out.size,
        [out.size]
    )
    first.free()
    second.free()
    return out
}

#: The most distinct class values `distinct_values` will report.
pub const MAX_CLASSES: int = 4096

#: The distinct values of a vector, ascending.
#:
#: Sorted on the device and compacted with an atomic counter, so the host only
#: touches the answer -- at most `MAX_CLASSES` long -- and never the samples.
pub fn distinct_values(y: Tensor) -> list[float] {
    if y.size == 0 { throw ValueError("cannot take the distinct values of an empty vector") }
    let column = y.view(0, [y.size, 1])
    let sorted = sort_columns(column)
    column.free()
    let counter = Tensor.zeros([1])
    let found = Tensor([MAX_CLASSES])
    let total = Tensor([1])
    base_kernel("base_unique_from_sorted").dispatch_async(
        [sorted.buffer(), counter.buffer(), found.buffer()],
        y.size,
        [y.size, MAX_CLASSES]
    )
    base_kernel("base_counter_value").dispatch_async([counter.buffer(), total.buffer()], 1, [])
    let count = int(total.download()[0] + 0.5)
    let values = found.download()
    sorted.free()
    counter.free()
    found.free()
    total.free()
    if count > MAX_CLASSES {
        throw ValueError(f"{count} distinct values exceeds the {MAX_CLASSES} this supports")
    }
    var kept: list[float] = []
    for index in 0..count { kept.push(values[index]) }
    return merge_sort(kept)
}

#: Arbitrary class values mapped onto the codes `0..k-1` and back.
#:
#: Every classifier fits against codes, `StratifiedKFold` counts them and the
#: metrics compare them, so the mapping lives here once and each of those takes
#: a `LabelIndex` rather than deriving its own.
pub class LabelIndex {
    #: The class values, ascending. Position `i` is code `i`.
    pub let classes: list[float]
    let table: Tensor

    #: Build an index over class values that are already known.
    pub fn init(self, classes: list[float]) {
        if classes.len() == 0 { throw ValueError("a LabelIndex needs at least one class") }
        self.classes = [value for value in classes]
        self.table = Tensor([classes.len()], self.classes)
    }

    #: Build an index over whatever values a label vector holds.
    pub static fn from_labels(y: Tensor) -> LabelIndex {
        return LabelIndex(distinct_values(y))
    }

    #: How many classes there are.
    pub fn count(self) -> int { return self.classes.len() }

    #: The class values as a `[k]` device tensor, for kernels that need them.
    pub fn class_tensor(self) -> Tensor { return self.table }

    #: Class values to codes `0..k-1`.
    pub fn encode(self, y: Tensor) -> Tensor {
        let out = Tensor(y.shape)
        self._map(y, out, 0)
        #: A value in no class encodes to -1, and a silent -1 turns into a
        #: wrong class much further downstream, so it is caught here for the
        #: price of one pass and one scalar read. Negated and maximised rather
        #: than minimised because that is the reduction jaitensor spreads over
        #: threadgroups; the minimum would be one thread walking the samples.
        let flipped = jt.neg(out)
        let peak = jt.amax(flipped)
        let lowest = -peak.download()[0]
        flipped.free()
        peak.free()
        if lowest < 0.0 {
            out.free()
            throw ValueError(f"a label is outside the {self.count()} known classes")
        }
        return out
    }

    #: Codes `0..k-1` back to class values.
    pub fn decode(self, codes: Tensor) -> Tensor {
        let out = Tensor(codes.shape)
        self._map(codes, out, 1)
        return out
    }

    fn _map(self, input: Tensor, out: Tensor, decode: int) -> void {
        base_kernel("base_map_labels").dispatch_async(
            [input.buffer(), self.table.buffer(), out.buffer()],
            input.size,
            [input.size, self.count(), decode]
        )
    }

    #: Release the device copy of the class table. Freeing twice is harmless.
    pub fn free(self) -> void { self.table.free() }

    fn __str__(self) -> str { return f"LabelIndex(classes={self.classes})" }
}

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
