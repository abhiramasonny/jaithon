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
from jaitensor import Tensor
from .base.kernels import cached_kernel
from .base.params import Params, check_fitted, check_matrix, check_pair, check_width
from .base.params import param_bool, param_float, param_int, param_str, params_of
from .base.params import require_target
from .base.data import column, hstack, index_tensor, matrix, select_columns, take_rows
from .base.data import to_rows, to_values, vector
from .base.reduce import argsort_columns, column_min_max, column_moments, column_quantiles
from .base.reduce import log_sum_exp_rows, masked_column_mean, row_argmax, row_argmin
from .base.reduce import row_extreme, row_topk, sort_columns
from .base.distance import GRAM_LINEAR, GRAM_POLY, GRAM_RBF, GRAM_SIGMOID, gram
from .base.distance import pairwise_sqdist
from .base.classes import bincount, class_sums, class_sumsq, scatter_add_rows
from .base.elementwise import affine_columns, count_equal, nan_mask, sum_to_host
from .base.random import kmeanspp_seed, random_normal, random_uniform, rng_for
from .base.labels import LabelIndex, MAX_CLASSES, distinct_values
from .base.solve import cholesky_factor, cholesky_solve, diagonal, triangular_solve

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
export { affine_columns, count_equal, nan_mask }
export { kmeanspp_seed, random_normal, random_uniform, rng_for }
export { LabelIndex, MAX_CLASSES, distinct_values }
export { cholesky_factor, cholesky_solve, diagonal, triangular_solve }

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
