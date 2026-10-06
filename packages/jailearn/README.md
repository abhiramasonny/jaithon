# jailearn

Classical machine learning, on the GPU.

```jai
import jailearn as jl

let x = jl.matrix([[0.0, 0.0], [1.0, 0.5], [4.0, 4.0], [5.0, 4.5]])
let y = jl.vector([0.0, 0.0, 1.0, 1.0])
defer { x.free() }
defer { y.free() }

let (mean, variance) = jl.column_moments(x)
defer { mean.free() }
defer { variance.free() }

let classes = jl.LabelIndex.from_labels(y)
defer { classes.free() }
let codes = classes.encode(y)
defer { codes.free() }
let counts = jl.bincount(codes, classes.count())
defer { counts.free() }
jl.to_values(counts)    # [2.0, 2.0]
```

Names follow scikit-learn where scikit-learn has a name — `fit`, `predict`,
`transform`, `fit_transform`, `score`, `get_params`, `set_params` — so a
program written against it translates line for line into snake case.

## The data convention

`X` is a `jaitensor.Tensor` of shape `[n_samples, n_features]`. `y` is
`[n_samples]`. Both are float32 and live on the GPU, and class labels travel
as float codes rather than as integers or strings; `LabelIndex` maps whatever
codes a caller used onto `0..k-1` and back, and every classifier,
`StratifiedKFold` and the metrics go through it.

`jl.matrix`, `jl.vector`, `jl.to_rows` and `jl.to_values` cross between host
and device for hand-written data. They walk every element on the host, so they
are for examples and tests; real data should arrive as a `Tensor` already.

jailearn needs a Metal device. `jl.is_available()` says whether there is one;
there is no host fallback.

## Memory

An estimator owns device tensors, and device memory is not garbage collected.
Every estimator has `free`, freeing twice is harmless, and the habit
throughout is a `defer` beside the constructor:

```jai
let model = jl.Ridge(alpha: 1.0)
defer { model.free() }
model.fit(x, y)
```

The same applies to every tensor a call here hands back.

## Why it is fast

A host loop over more than a few thousand elements costs seconds in this
language, so anything whose cost grows with `n_samples` is a Metal kernel
dispatched over the data. `base.jai` holds that kit — column moments and
bounds, per-column bitonic sorting and quantiles, tiled pairwise distances and
the four kernel matrices, row argmax/argmin/top-k, class totals, scatter-add,
k-means++ seeding, log-sum-exp and small dense Cholesky solves — and every
module composes it rather than writing its own.

That is the difference from the two estimators this package supersedes.
jaicv's `DTrees` sorts every feature at every node on the host; jaicv's `PCA`
forms the covariance in a triple host loop. Both are correct and both are
orders of magnitude slower than the same fit dispatched.

## Layout

`base.jai` is the foundation: the `Estimator` trait, the `Params` and `Scorer`
types, the validation helpers, `LabelIndex`, and the device kit.

`shared.jai` is the second layer — the machinery more than one estimator
family needs, so those families do not have to import each other. Binning
(`bin_edges`, `bin_codes`, `bin_centres`), the atomic histogram every split
search reduces to (`bin_histograms`), the split rule and its threadgroup scan
(`SplitRule`, `split_gains`, `best_split`), the device form of a fitted tree
(`TreeArrays`, `tree_accumulate`, `forest_mean`), the samplers
(`permutation`, `bootstrap_rows`, `choose_features`, `sample_weights`), the
linear helpers (`add_bias`, `linear_scores`, `soft_threshold`,
`column_norms`), and `Descent`, which is why every iterative fit in the
package reports `n_iter` and `converged` the same way.

`linear.jai` holds `StandardScaler`, `Ridge` and `linear_regression`;
`neighbors.jai` holds `KNeighborsClassifier` and `KNeighborsRegressor`, a
brute-force search built on `nearest_rows` -- one GEMM per band of queries and
a one-pass top-k that ranks the raw product, so the distance matrix is never
finished or held whole.

`pipeline.jai` composes estimators: `Pipeline`, `ColumnTransformer` and
`FeatureUnion`, with `__` routing a parameter down to a named step.

An estimator module imports `base` and `shared` and nothing else of
jailearn's, which is what lets the modules be written and reviewed
independently.

Metal entry points are prefixed with their module's short name —
`base_column_reduce`, `sh_histogram`, `svm_smo_gradient` — because
`base.cached_kernel(source, entry)` caches on the entry name alone and two
modules that both picked `assign` would silently share one kernel.

## Switches

Fast paths with an older path beside them, kept one environment variable away
so a change can be measured as an A/B in one binary. Each is read once, when
its module loads, and is on unless set to `0`.

| Switch | Off restores |
| --- | --- |
| `JAILEARN_HOST_SOLVE` | the device Cholesky factor and substitutions (two dispatches a column, one a row) for every order, instead of reading a system of order up to `HOST_SOLVE_ORDER` (256) back and solving it on the host in double precision; 50x50 goes from about 200 dependent dispatches to one read back |
| `JAILEARN_COLUMN_TILES` | one threadgroup per column striding the samples, two passes, for `column_moments`, `column_min_max` and `masked_column_mean`, instead of row tiles read as whole consecutive rows, folded per tile with Welford's update and merged by Chan's rule in a second small dispatch |
| `JAILEARN_ROW_TOPK` | one thread per row making `k` passes (`base_row_topk`) for every `k`, instead of a threadgroup per row folding its columns in one coalesced pass into per-thread registers and merging the heads, for `k` up to `TOPK_ONE_PASS_MAX` (32); the same tie order, so the same answer. It also turns `nearest_rows` back into a finished distance matrix and a separate top-k |
| `JAILEARN_RIDGE_PACKED` | `Ridge.fit` as a centred copy of `X`, then `XT X` and `XT y` as two products at the feature width (off the GEMM's fast route at widths not divisible by four), with the means read back before centring and the target summed on the host, instead of one product over a packed `[Xc \| yc \| 0..]` four-aligned operand and one wait |
| `JAILEARN_LOGIT_FUSED` | a separate product for `LogisticRegression`'s margins each Newton iteration, instead of taking each row's margin inside the row kernel |
| `JAILEARN_SCALER_ON_DEVICE` | reading the moments back and building `StandardScaler`'s state on the host, two waits on the device per `fit_transform`, instead of one in-place kernel and no wait |

## Tests

```sh
JAITHON_PATH=$PWD/lib ./jaithon test packages/jailearn/tests
```

Every export has a caller there, and the numeric assertions are against values
worked out by hand rather than against a previous run.
