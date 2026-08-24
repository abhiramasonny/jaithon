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
types, the validation helpers, `LabelIndex`, and the device kit. Every other
module imports it and nothing else of jailearn's, which is what lets the
modules be written and reviewed independently.

Metal entry points are prefixed with their module's short name —
`base_column_reduce`, `tree_histogram`, `svm_smo_gradient` — because
`base.cached_kernel(source, entry)` caches on the entry name alone and two
modules that both picked `assign` would silently share one kernel.

## Tests

```sh
JAITHON_PATH=$PWD/lib ./jaithon test packages/jailearn/tests
```

Every export has a caller there, and the numeric assertions are against values
worked out by hand rather than against a previous run.
