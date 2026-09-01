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
from .base.elementwise import affine_columns, count_equal, nan_mask
from .base.random import kmeanspp_seed, random_normal, random_uniform, rng_for
from .base.labels import LabelIndex, MAX_CLASSES, distinct_values
from .base.solve import cholesky_factor, cholesky_solve, diagonal, triangular_solve
from .base.estimator import Estimator, EstimatorKind, Scorer, default_score

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
export { Estimator, EstimatorKind, Scorer, default_score }
