# jaisci

Scientific computing with scipy's API, on the GPU where it pays.

```jai
import jaisci as sci

let fit = sci.minimize(rosenbrock, [0.0, 0.0])
print(fit.x)

let spline = sci.CubicSpline(x, y)
let (frequencies, power) = sci.welch(samples, fs: 1000.0)
let (statistic, pvalue) = sci.ttest_ind(before, after)
```

Names, argument order and result fields follow scipy in snake case:
`minimize` returns an `OptimizeResult` with `x`, `fun` and `success`,
`linregress` returns slope, intercept, rvalue, pvalue and both standard errors,
and every hypothesis test returns a statistic and a p-value. A program written
against scipy translates line for line.

## The two policies

Everything in this package obeys both, and `src/jaisci/core.jai` states them at
the top of the file.

**Host arithmetic is f64, device arithmetic is f32.** A Jaithon `float` is a
double; a `std.gpu.Buffer` holds singles. Small dense work — a spline system, a
Jacobian, a covariance of a few hundred rows — stays on the host, where the
digits survive and where a kernel launch would cost more than the arithmetic.
That is why the `dense_*` factorisations in `core` take and return
`list[float]` and never touch a buffer, and it is the same call jaicv's
`core/linalg.jai` already made.

**Only O(n²) and up becomes a kernel.** Distance matrices, convolutions, sparse
products, Jacobian assembly, a distribution evaluated over a whole array: those
dispatch. A Newton step on twelve parameters does not. The rule that catches the
common mistake is the corollary — *a host loop over more than a few thousand
elements is a bug*, not a slow path, because in this language it costs seconds.

## Layout

`core.jai` is the floor every other module stands on: the `Array` type and its
kernels, `KernelCache`, the host `dense_*` factorisations, the result records,
and `approx_fprime`. Read its module doc before adding to any of the others.

| File | Covers |
| --- | --- |
| `core.jai` | `Array`, kernel cache, dense host factorisations, result records |
| `linalg.jai` | `scipy.linalg`: solves, factorisations, eigenvalues, `expm` |
| `sparse.jai` | CSR and COO matrices, `spsolve`, Krylov solvers |
| `minimize.jai` | `minimize`, `minimize_scalar`, line search, scalar roots |
| `least_squares.jai` | `least_squares`, `curve_fit`, `nnls`, Levenberg-Marquardt |
| `linprog.jai` | `linprog` |
| `integrate.jai` | `quad` and friends, `solve_ivp`, `odeint` |
| `interpolate.jai` | `interp1d`, splines, `griddata`, `interpn` |
| `spectral.jai` | `scipy.fft`, plus `welch`, `stft` and the spectral estimates |
| `filters.jai` | filter design and application, `get_window`, `savgol_filter` |
| `convolve.jai` | `convolve`, `correlate`, and their 2-D and FFT forms |
| `distributions.jai` | distributions and the special functions behind them |
| `stats.jai` | descriptive statistics, correlations, hypothesis tests |
| `spatial.jai` | distances, `KDTree`, hulls, Delaunay, Voronoi |

`src/jaisci/mod.jai` re-exports all of it, and is the contract each module owes
the package. Each file also stays independently importable —
`from jaisci.core import Array` keeps working.

## The array

`Array` is dense, row-major float32 in one `std.gpu.Buffer`, with a
`shape: list[int]` and nothing else: no strides, no channels, no autograd. A
`view` is a window on the parent's buffer and a `reshape` is the same buffer
read with different extents, so both alias and neither copies. Device memory is
not collected, so call `free`, usually from a `defer`.

It exists here because nothing else in the repository offers a plain numeric
array: `jaitensor.Tensor` carries autograd and a training shape, `jaicv.Mat` is
a two-dimensional image with depth and channels. `Array` and `core`'s op set
are `jainum`'s charter rather than this package's, and when jainum grows an
equivalent, jaisci should re-export it and delete that half of `core.jai`.

## What is deliberately absent

Distribution `.fit`, bootstrap and permutation tests, mixed-integer programming,
DAEs and PDEs, sparse eigensolvers, automatic knot placement, Clough-Tocher
cubic `griddata`, convex hulls above three dimensions, three-dimensional
Delaunay, and the DCT/DST family. Each is a reasonable followup; none is in
scope for 0.1.

## Tests

```
JAITHON_PATH=$PWD/lib ./jaithon test packages/jaisci/tests
```

One file per module, named after it. The device tests return early when
`jaisci.is_available()` is false; the `dense_*` tests always run, because pure
host arithmetic is the point of them.
