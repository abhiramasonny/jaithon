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
#: let fit = sci.minimize(rosenbrock, [0.0, 0.0])
#: print(fit.x)
#:
#: let (frequencies, power) = sci.welch(signal, fs: 1000.0)
#: let spline = sci.CubicSpline(x, y)
#: let (statistic, pvalue) = sci.ttest_ind(before, after)
#: ```
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
import std.gpu as gpu

from .core import OptimizeResult, RootResult
from .core import approx_fprime, approx_jacobian, as_array, dense_cholesky, dense_lu
from .core import dense_lu_solve, dense_qr, dense_triangular, dense_tridiagonal
from .core import device_matrix, device_vector, dot, givens, host, host_matrix
from .core import is_available, matmul, matvec, require_gpu
from .core import arr_abs, arr_clip, arr_exp, arr_log, arr_maximum, arr_minimum, arr_pow
from .core import arr_recip, arr_sign, arr_sqrt
from .linalg import block_diag, cho_factor, cho_solve, cholesky, dense_banded, det, eig, eigh
from .linalg import eigvals, eigvalsh, expm, inv, kron, lstsq, lu, lu_factor, lu_solve
from .linalg import matrix_rank, null_space, orth, pinv, qr, solve, solve_banded
from .linalg import solve_triangular, svd, toeplitz
from .sparse import CooMatrix, CsrMatrix, coo_matrix, csr_matrix, diags, sparse_eye
from .sparse import sparse_kron, spdiags
from .sparse_solve import cg, gmres, spsolve
from .minimize import Bounds, differential_evolution, line_search, minimize, minimize_scalar
from .roots import bisect, brenth, brentq, fsolve, newton, ridder, root_scalar
from .fitting import curve_fit, least_squares, leastsq, levenberg_marquardt, lsq_linear, nnls
from .linprog import LinprogResult, linprog
from .integrate import cumulative_trapezoid, fixed_quad, quad, romberg, simpson, trapezoid
from .ode import OdeResult, odeint, solve_ivp
from .interpolate import CubicSpline, PchipInterpolator, interp1d, splev, splrep
from .interpnd import RegularGridInterpolator, griddata, interpn
from .convolve import choose_conv_method, convolve, convolve2d, correlate, correlate2d
from .convolve import deconvolve, fftconvolve, oaconvolve
from .spectral import coherence, csd, periodogram, spectrogram, stft, welch
from .filters import butter, cheby1, detrend, filtfilt, firwin, get_window, hilbert, iirfilter
from .filters import lfilter, medfilt, resample, savgol_filter, sosfilt, sosfiltfilt, tf2sos
from .filters import zpk2tf
from .distributions import Distribution, beta, betainc, betaln, binom, chi2, erf, erfc, expon
from .distributions import f, gamma, gammainc, gammaincc, gammaln, lognorm, ndtr, ndtri
from .distributions import norm as norm_dist
from .distributions import poisson, t, uniform
from .stats import DescribeResult, LinregressResult, TestResult, chi2_contingency, chisquare
from .stats import describe, entropy, f_oneway, gaussian_kde, gmean, hmean, iqr, kendalltau
from .stats import kruskal, ks_2samp, kstest, kurtosis, linregress, mannwhitneyu, mode
from .stats import pearsonr, sem, shapiro, skew, spearmanr, trim_mean, ttest_1samp, ttest_ind
from .stats import ttest_rel, wilcoxon, zscore
from .spatial import KDTree, cdist, chebyshev, cityblock, distance_matrix, euclidean
from .spatial import mahalanobis, minkowski, pdist, squareform
from .geometry import ConvexHull, Delaunay, Voronoi

export { as_array, host, host_matrix, device_vector, device_matrix, require_gpu }
export { arr_exp, arr_log, arr_sqrt, arr_abs, arr_pow, arr_recip, arr_sign, arr_clip }
export { arr_maximum, arr_minimum, matmul, matvec, dot }
export { dense_lu, dense_lu_solve, dense_cholesky, dense_qr, dense_triangular }
export { dense_tridiagonal, dense_banded, givens, approx_fprime, approx_jacobian }
export { OptimizeResult, RootResult, OdeResult, LinprogResult }
export { TestResult, DescribeResult, LinregressResult }
export { solve, solve_triangular, solve_banded, lu, lu_factor, lu_solve, cholesky }
export { cho_factor, cho_solve, qr, svd, eig, eigh, eigvals, eigvalsh, inv, pinv, det }
export { lstsq, matrix_rank, expm, null_space, orth, kron, block_diag, toeplitz }
export { CsrMatrix, CooMatrix, csr_matrix, coo_matrix, diags, spdiags, sparse_eye, sparse_kron }
export { spsolve, cg, gmres }
export { Bounds, minimize, minimize_scalar, line_search, differential_evolution }
export { bisect, brentq, brenth, ridder, newton, root_scalar, fsolve }
export { least_squares, curve_fit, leastsq, lsq_linear, nnls, levenberg_marquardt }
export { linprog }
export { quad, fixed_quad, romberg, simpson, trapezoid, cumulative_trapezoid }
export { solve_ivp, odeint }
export { interp1d, CubicSpline, PchipInterpolator, splrep, splev }
export { RegularGridInterpolator, griddata, interpn }
export { convolve, correlate, convolve2d, correlate2d, fftconvolve, oaconvolve, deconvolve }
export { choose_conv_method }
export { periodogram, welch, csd, coherence, stft, spectrogram }
export { get_window, butter, cheby1, firwin, iirfilter, lfilter, filtfilt }
export { sosfilt, sosfiltfilt, tf2sos, zpk2tf, savgol_filter, medfilt, detrend, resample }
export { hilbert }
export { Distribution, uniform, expon, gamma, beta, chi2, t, f, binom, poisson, lognorm }
export { erf, erfc, gammaln, gammainc, gammaincc, betainc, betaln, ndtr, ndtri }
export { describe, gmean, hmean, sem, zscore, skew, kurtosis, mode, trim_mean }
export { iqr, entropy, pearsonr, spearmanr, kendalltau, linregress }
export { ttest_1samp, ttest_ind, ttest_rel, chisquare, chi2_contingency, f_oneway }
export { mannwhitneyu, wilcoxon, kruskal, ks_2samp, kstest, shapiro, gaussian_kde }
export { cdist, pdist, squareform, distance_matrix, euclidean, cityblock, minkowski }
export { chebyshev, mahalanobis, KDTree }
export { ConvexHull, Delaunay, Voronoi }
export { is_available }

#: The package version, which `jaithon.package.json` also carries.
pub let VERSION = "0.1.0"

#: The normal distribution, under a name that does not collide.
#:
#: scipy calls it `scipy.stats.norm` and calls the matrix norm
#: `scipy.linalg.norm`, keeping them apart by module. A flat facade has no room
#: for that, and the matrix norm is `jainum.linalg.norm` anyway, so the
#: distribution is the one that gets renamed.
pub let normal = norm_dist

#: Name of the active GPU, or `"none"` when none exists.
pub fn device_name() -> str { return gpu.device_name() }

#: Number of Metal devices.
pub fn device_count() -> int { return gpu.device_count() }

#: Select the Metal device later allocations live on.
pub fn set_device(index: int) -> void { gpu.set_device(index) }
