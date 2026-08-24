#: Classical machine learning on the GPU, with scikit-learn's shape.
#:
#: `X` is a `jaitensor.Tensor` of shape `[n_samples, n_features]`, `y` is
#: `[n_samples]`, both float32 and device-resident; class labels travel as
#: float codes. Every estimator implements `Estimator`, so a pipeline, a grid
#: search and a cross-validation take any of them without knowing which.
#:
#: Estimators own device memory, so every one of them has `free`.
#:
#: ```
#: import jailearn as jl
#: let x = jl.matrix([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.5]])
#: let y = jl.vector([0.0, 0.0, 1.0, 1.0])
#: defer { x.free() }
#: defer { y.free() }
#: let (mean, variance) = jl.column_moments(x)
#: defer { mean.free() }
#: defer { variance.free() }
#: let labels = jl.LabelIndex.from_labels(y)
#: defer { labels.free() }
#: labels.classes    # [0.0, 1.0]
#: ```
import std.gpu as gpu

from .base import Estimator, EstimatorKind, LabelIndex, Params, Scorer
from .base import GRAM_LINEAR, GRAM_POLY, GRAM_RBF, GRAM_SIGMOID, MAX_CLASSES
from .base import affine_columns, argsort_columns, bincount, cached_kernel, cholesky_factor
from .base import cholesky_solve, check_fitted, check_matrix, check_pair, check_width
from .base import class_sums, class_sumsq, column, column_min_max, column_moments
from .base import column_quantiles, count_equal, default_score, diagonal, distinct_values
from .base import gram, hstack, index_tensor, kmeanspp_seed, log_sum_exp_rows
from .base import masked_column_mean, matrix, nan_mask, pairwise_sqdist, param_bool
from .base import param_float, param_int, param_str, params_of, random_normal
from .base import random_uniform, require_target, rng_for, row_argmax, row_argmin
from .base import row_extreme, row_topk, scatter_add_rows, select_columns, sort_columns
from .base import take_rows, to_rows, to_values, triangular_solve, vector

export { Estimator, EstimatorKind, LabelIndex, Params, Scorer, default_score }
export { matrix, vector, to_rows, to_values, column, select_columns, hstack, take_rows }
export { index_tensor, param_int, param_float, param_bool, param_str, params_of }
export { check_matrix, check_pair, require_target, check_fitted, check_width }
export { column_moments, column_min_max, masked_column_mean, nan_mask, affine_columns }
export { sort_columns, argsort_columns, column_quantiles, pairwise_sqdist }
export { GRAM_LINEAR, GRAM_RBF, GRAM_POLY, GRAM_SIGMOID, gram }
export { row_extreme, row_argmax, row_argmin, row_topk, log_sum_exp_rows }
export { bincount, class_sums, class_sumsq, scatter_add_rows, count_equal }
export { kmeanspp_seed, rng_for, random_uniform, random_normal }
export { MAX_CLASSES, distinct_values }
export { cholesky_factor, triangular_solve, cholesky_solve, diagonal }
export { cached_kernel }

#: The package version, as the manifest declares it.
pub let VERSION = "0.1.0"

#: Whether the machine has a GPU backend that jailearn can use. Everything in
#: the package needs one; there is no host fallback.
pub fn is_available() -> bool { return gpu.is_available() }

#: Name of the active GPU, or `"none"` when none exists.
pub fn device_name() -> str { return gpu.device_name() }
