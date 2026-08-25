#: Scientific computing — this language's answer to scipy.
#:
#: Names, argument order and result fields follow scipy in snake case, so a
#: program written against it translates line for line. `minimize` returns an
#: `OptimizeResult` with `x`, `fun` and `success`; `linregress` returns slope,
#: intercept, rvalue, pvalue and both standard errors; a hypothesis test
#: returns a statistic and a p-value.
#:
#: ```
#: import jaisci as sci
#:
#: let (lower, pivots) = sci.dense_lu(a)
#: let x = sci.dense_lu_solve(lower, pivots, b)
#: ```
#:
#: `sci.minimize`, `sci.CubicSpline` and `sci.ttest_ind` are here. The other
#: example this doc used to carry -- `sci.welch` -- names the surface the
#: signal module owes when it is written.
#:
#: **The array is `jainum.NDArray`, and jaisci does not define one.** Bulk data
#: — signals, samples, matrices, point sets — is an `NDArray`; parameter
#: vectors, filter coefficients and small results are `list[float]`, because
#: they are short and host arithmetic is f64. `core.jai` states that convention
#: in full, along with the two policies the package obeys, and `as_array`,
#: `host`, `device_vector` and `host_matrix` are the four crossings between the
#: two halves. Read that module doc before adding to any of the others.
#:
#: Two names are jainum's and are deliberately absent here rather than
#: shadowed: the transforms (`fft`, `rfft`, `fftfreq` and the rest live in
#: `jainum.fft`, and jaisci's `spectral` is the estimators built on top of
#: them) and `norm` the matrix norm (`jainum.linalg.norm`). `normal` below is
#: scipy's normal *distribution*, which scipy also spells `norm` and which
#: cannot share a flat namespace with the other one.
#:
#: This facade is the contract each module owes the package: the names below
#: are what a caller reaches through `import jaisci`, and a module that renames
#: one of them breaks it. Each file stays independently importable too —
#: `from jaisci.core import matmul` keeps working — so the facade is for
#: callers who want the package rather than the file.
#:
#: **`core`, `linalg`, `optimize`, `integrate`, `interpolate` and `stats`
#: exist today.** This facade named nineteen
#: modules -- linalg, optimize, integrate, interpolate, signal, stats, sparse
#: and the rest -- and eighteen of them were never written, so `import jaisci`
#: raised `cannot find module '.linalg'` for every caller. The package's own
#: tests did not catch it because they import `jaisci.core` directly. The
#: surface below is what is actually here; the list above stands as the
#: contract the other modules owe when they arrive, not as a claim that they
#: have.
#:
import std.gpu as gpu

from .core import OptimizeResult, RootResult
from .core import approx_fprime, approx_jacobian, as_array, dense_cholesky, dense_lu
from .core import dense_lu_solve, dense_qr, dense_triangular, dense_tridiagonal
from .core import device_matrix, device_vector, dot, givens, host, host_matrix
from .core import is_available, matmul, matvec, require_gpu
from .core import arr_abs, arr_clip, arr_exp, arr_log, arr_maximum, arr_minimum, arr_pow
from .core import arr_recip, arr_sign, arr_sqrt

from .linalg import cho_factor, cho_solve, cholesky, det, inv, lstsq, lu_factor
from .linalg import lu_solve, qr, solve, solve_triangular

from .optimize import MAXITER, RTOL, XTOL
from .optimize import bisect, brentq, brentq_full, curve_fit, least_squares
from .optimize import minimize, minimize_scalar, newton

from .integrate import IVPResult
from .integrate import cumulative_trapezoid, fixed_quad, quad, simpson, solve_ivp
from .integrate import trapezoid

from .interpolate import CubicSpline, PchipInterpolator, interp

from .stats import DescribeResult, LinregressResult, Normal
from .stats import betainc, describe, lgamma, linregress, normal, pearsonr
from .stats import spearmanr, t_cdf, t_sf, ttest_1samp, ttest_ind, ttest_rel

export { as_array, host, host_matrix, device_vector, device_matrix, require_gpu }
export { arr_exp, arr_log, arr_sqrt, arr_abs, arr_pow, arr_recip, arr_sign, arr_clip }
export { arr_maximum, arr_minimum, matmul, matvec, dot }
export { dense_lu, dense_lu_solve, dense_cholesky, dense_qr, dense_triangular }
export { dense_tridiagonal, givens, approx_fprime, approx_jacobian }
export { OptimizeResult, RootResult }
export { solve, solve_triangular, lu_factor, lu_solve, cholesky, cho_factor, cho_solve }
export { qr, inv, det, lstsq }
export { bisect, brentq, brentq_full, newton, minimize_scalar, minimize }
export { least_squares, curve_fit, XTOL, RTOL, MAXITER }
export { quad, fixed_quad, trapezoid, simpson, cumulative_trapezoid }
export { solve_ivp, IVPResult }
export { interp, CubicSpline, PchipInterpolator }
export { linregress, pearsonr, spearmanr, describe, LinregressResult, DescribeResult }
export { ttest_1samp, ttest_ind, ttest_rel, normal, Normal, t_cdf, t_sf }
export { betainc, lgamma }
export { is_available }

#: The package version, which `jaithon.package.json` also carries.
pub let VERSION = "0.1.0"

#: The normal distribution, under a name that does not collide.
#:
#: scipy calls it `scipy.stats.norm` and calls the matrix norm
#: `scipy.linalg.norm`, keeping them apart by module. A flat facade has no room
#: for that, and the matrix norm is `jainum.linalg.norm` anyway, so the
#: distribution is the one that gets renamed. It comes back with `distributions`.

#: Name of the active GPU, or `"none"` when none exists.
pub fn device_name() -> str { return gpu.device_name() }

#: Number of Metal devices.
pub fn device_count() -> int { return gpu.device_count() }

#: Select the Metal device later allocations live on.
pub fn set_device(index: int) -> void { gpu.set_device(index) }
