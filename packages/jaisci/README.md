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

## The array is jainum's

`jainum.NDArray` is the array type. jaisci does not define one, does not define
creation helpers, reductions, sorts or an FFT, and re-exports none of them:
`import jainum as nn` alongside jaisci and use both.

```jai
import jainum as nn
import jaisci as sci

let signal = nn.linspace(0.0, 1.0, 4096)
let spectrum = nn.fft.rfft(signal)          #: the transform is jainum's
let (f, power) = sci.welch(signal, fs: 4096.0)   #: the estimator is jaisci's
```

Two names stay in jainum on purpose. The transforms — `fft`, `rfft`, `fftfreq`
and the rest — are `jainum.fft`, and jaisci's `spectral.jai` is the estimators
built on top of them. `norm` the matrix norm is `jainum.linalg.norm`; the name
`norm` in scipy's other sense, the normal distribution, is `sci.normal` here,
because a flat facade cannot hold both and scipy only manages it by keeping
them in separate modules.

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
dispatch. A Newton step on twelve parameters does not. The rule that catches
the common mistake is the corollary — *a host loop over more than a few
thousand elements is a bug*, not a slow path, because in this language it costs
seconds.

## The calling convention

Stated once in `core.jai` and obeyed everywhere:

| Kind of value | Type |
| --- | --- |
| Bulk data: a signal, a sample, a matrix, a point set | `NDArray` |
| A parameter vector, filter coefficients, a small result | `list[float]` |
| A scalar objective | `fn(list[float]) -> float` |
| A model evaluated over data | `fn(NDArray, list[float]) -> NDArray` |

`as_array`, `host`, `device_vector` and `host_matrix` are the four crossings
between the two halves. A module that writes its own crossing has almost
certainly got the precision wrong.

## Kernel names

`jainum.kernel_of` caches compiled pipelines on the **entry name alone** — it
never hashes the source, because that costs more than the dispatch it precedes.
Two modules that both name an entry `reduce` therefore share one pipeline and
produce wrong answers with no error anywhere. Every entry point in jaisci
carries its module's reserved prefix:

| Prefix | Module | Prefix | Module |
| --- | --- | --- | --- |
| `sci_` | core | `sod_` | ode |
| `sla_` | linalg | `sit_` | interpolate |
| `ssp_` | sparse | `snd_` | interpnd |
| `ssk_` | sparse_solve | `scv_` | convolve |
| `sop_` | minimize | `ssg_` | spectral |
| `sro_` | roots | `sfl_` | filters |
| `sfi_` | fitting | `sdi_` | distributions |
| `slp_` | linprog | `sst_` | stats |
| `squ_` | integrate | `ssa_` | spatial |
| | | `sgo_` | geometry |

None of them collides with jainum's `nd_`, `nf_`, `nr_`, `nc_`, `nds_`, `nm_`
or `nl_`.

## Layout

`core.jai` is the floor every other module stands on: the policies, the calling
convention, the kernel idiom, the bridge helpers, the tiled GEMM, the host
`dense_*` factorisations, `approx_fprime` and the two shared result records.
Read its module doc before adding to any of the others.

| File | Covers |
| --- | --- |
| `core.jai` | the bridge, `matmul`, host dense factorisations, `OptimizeResult` |
| `linalg.jai` | `scipy.linalg`: solves, factorisations, eigenvalues, `expm` |
| `sparse.jai` | CSR and COO matrices and their products |
| `sparse_solve.jai` | `spsolve`, `cg`, `gmres` |
| `minimize.jai` | `minimize`, `minimize_scalar`, line search |
| `roots.jai` | `brentq` and the other scalar roots, `fsolve` |
| `fitting.jai` | `least_squares`, `curve_fit`, `nnls`, Levenberg-Marquardt |
| `linprog.jai` | `linprog` |
| `integrate.jai` | `quad` and friends, the fixed-sample rules |
| `ode.jai` | `solve_ivp`, `odeint` |
| `interpolate.jai` | `interp1d`, `CubicSpline`, `PchipInterpolator`, splines |
| `interpnd.jai` | `RegularGridInterpolator`, `griddata`, `interpn` |
| `convolve.jai` | `convolve`, `correlate`, and their 2-D and FFT forms |
| `spectral.jai` | `welch`, `stft` and the spectral estimates |
| `filters.jai` | filter design and application, `get_window`, `savgol_filter` |
| `distributions.jai` | distributions and the special functions behind them |
| `stats.jai` | descriptive statistics, correlations, hypothesis tests |
| `spatial.jai` | distances, `cdist`, `pdist`, `KDTree` |
| `geometry.jai` | `ConvexHull`, `Delaunay`, `Voronoi` |

`src/jaisci/mod.jai` re-exports all of it, and is the contract each module owes
the package. Each file also stays independently importable —
`from jaisci.core import matmul` keeps working.

## What is deliberately absent

Distribution `.fit`, bootstrap and permutation tests, mixed-integer and
interior-point LP, DAEs, BVPs and PDEs, sparse eigensolvers and sparse
Cholesky, `bicgstab`, `dblquad` and `tplquad`, the Akima, barycentric and
Lagrange interpolators, Clough-Tocher `griddata`, `cheby2`, `ellip`, `bessel`,
`decimate`, `freqz`, `istft`, the DCT/DST family, convex hulls above three
dimensions, three-dimensional Delaunay, and scipy's trust-region, SLSQP and
COBYLA minimisers. Each is a reasonable followup; none is in scope for 0.1, and
each module's own doc names the ones it would have owned.

The elementwise `arr_*` helpers in `core.jai` — `arr_exp`, `arr_log`,
`arr_sqrt` and the rest — are stopgaps with a FOLLOWUP apiece: jainum's ufunc
set is the right home for them, and they should be deleted and re-exported the
moment it lands.

## Tests

```
JAITHON_PATH=$PWD/lib ./jaithon test packages/jaisci/tests
```

One file per module, named after it. The device tests return early when
`jaisci.is_available()` is false; the `dense_*` tests always run, because pure
host arithmetic is the point of them.
