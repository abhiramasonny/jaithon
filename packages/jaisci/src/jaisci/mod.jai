#: Scientific computing — this language's answer to scipy.
#:
#: Names, argument order and result fields follow scipy in snake case, so a
#: program written against it translates line for line. `minimize` returns an
#: `OptimizeResult` with `x`, `fun` and `success`; `linregress` returns slope,
#: intercept, rvalue, pvalue and both standard errors; a hypothesis test returns
#: a statistic and a p-value.
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
#: `core` carries the pieces every other module stands on: the `Array` type and
#: its kernels, the `KernelCache` idiom, the host `dense_*` factorisations, and
#: the result records. It also states the two policies the package obeys —
#: host arithmetic in f64, device arithmetic in f32, and only O(n^2) and up
#: becomes a kernel. Read `core.jai`'s module doc before adding to any of the
#: others.
#:
#: This facade is the contract each module owes the package: the names below are
#: what a caller reaches through `import jaisci`, and a module that renames one
#: of them breaks it. Each file stays independently importable too —
#: `from jaisci.core import Array` keeps working — so the facade is for callers
#: who want the package rather than the file.
import std.gpu as gpu

from .core import Array, KernelCache, LinregressResult, OdeResult, OptimizeResult
from .core import RootResult, TestResult
from .core import abs, add, amax, amin, approx_fprime, argmax, argmin, argsort, clamp, concat
from .core import cumsum, dense_banded, dense_cholesky, dense_lu, dense_lu_solve, dense_qr
from .core import dense_triangular, dense_tridiagonal, divide, dot, element_count, exp, f32_bits
from .core import givens, log, matmul, maximum, mean, minimum, multiply, neg, offset, power
from .core import rank_data, scale, searchsorted, sort_values, sqrt, subtract, sum, take
from .core import transpose
from .linalg import block_diag, cho_factor, cho_solve, cholesky, det, eig, eigh, eigvals
from .linalg import eigvalsh, expm, inv, kron, lstsq, lu, lu_factor, lu_solve, matrix_rank, norm
from .linalg import null_space, orth, pinv, qr, solve, solve_banded, solve_triangular, svd
from .linalg import toeplitz
from .sparse import CooMatrix, CsrMatrix, bicgstab, cg, coo_matrix, csr_matrix, diags, gmres
from .sparse import sparse_eye, sparse_kron, spdiags, spsolve
from .minimize import Bounds, bisect, brenth, brentq, differential_evolution, fsolve, line_search
from .minimize import minimize, minimize_scalar, newton, ridder, root_scalar
from .least_squares import curve_fit, least_squares, leastsq, levenberg_marquardt, lsq_linear
from .least_squares import nnls
from .linprog import linprog
from .integrate import cumulative_trapezoid, dblquad, fixed_quad, odeint, quad, romberg, simpson
from .integrate import solve_ivp, trapezoid
from .interpolate import Akima1DInterpolator, BarycentricInterpolator, CubicSpline
from .interpolate import PchipInterpolator, RegularGridInterpolator, griddata, interp1d, interpn
from .interpolate import lagrange, splev, splrep
from .spectral import coherence, csd, fft, fft2, fftfreq, fftshift, ifft, ifft2, ifftshift, irfft
from .spectral import istft, periodogram, rfft, rfftfreq, spectrogram, stft, welch
from .filters import butter, cheby1, cheby2, decimate, detrend, filtfilt, firwin, freqz, get_window
from .filters import hilbert, iirfilter, lfilter, medfilt, resample, savgol_filter, sosfilt
from .filters import sosfiltfilt, tf2sos, zpk2tf
from .convolve import choose_conv_method, convolve, convolve2d, correlate, correlate2d, deconvolve
from .convolve import fftconvolve, oaconvolve
from .distributions import Distribution, beta, betainc, betaln, binom, chi2, erf, erfc, expon, f
from .distributions import gamma, gammainc, gammaincc, gammaln, lognorm, ndtr, ndtri
from .distributions import norm as norm_dist
from .distributions import poisson, t, uniform
from .stats import DescribeResult, chi2_contingency, chisquare, describe, entropy, f_oneway
from .stats import gaussian_kde, gmean, hmean, iqr, kendalltau, kruskal, ks_2samp, kstest
from .stats import kurtosis, linregress, mannwhitneyu, mode, pearsonr, sem, shapiro, skew
from .stats import spearmanr, trim_mean, ttest_1samp, ttest_ind, ttest_rel, wilcoxon, zscore
from .spatial import ConvexHull, Delaunay, KDTree, Voronoi, cdist, chebyshev, cityblock
from .spatial import distance_matrix, euclidean, mahalanobis, minkowski, pdist, squareform

export { Array, KernelCache, element_count, f32_bits }
export { add, subtract, multiply, divide, scale, offset, neg, exp, log, sqrt, abs, power }
export { maximum, minimum, clamp, sum, mean, amin, amax, argmin, argmax, dot, cumsum }
export { matmul, transpose, concat, take }
export { sort_values, argsort, rank_data, searchsorted }
export { dense_lu, dense_lu_solve, dense_cholesky, dense_qr, dense_triangular }
export { dense_tridiagonal, dense_banded, givens, approx_fprime }
export { OptimizeResult, RootResult, OdeResult, TestResult, LinregressResult }
export { solve, solve_triangular, solve_banded, lu, lu_factor, lu_solve, cholesky }
export { cho_factor, cho_solve, qr, svd, eig, eigh, eigvals, eigvalsh, inv, pinv, det }
export { norm, lstsq, matrix_rank, expm, null_space, orth, kron, block_diag, toeplitz }
export { CsrMatrix, CooMatrix, csr_matrix, coo_matrix, diags, spdiags, sparse_eye, sparse_kron }
export { spsolve, cg, gmres, bicgstab }
export { Bounds, minimize, minimize_scalar, line_search, differential_evolution }
export { bisect, brentq, brenth, ridder, newton, root_scalar, fsolve }
export { least_squares, curve_fit, leastsq, lsq_linear, nnls, levenberg_marquardt }
export { linprog }
export { quad, fixed_quad, romberg, simpson, trapezoid, cumulative_trapezoid, dblquad }
export { solve_ivp, odeint }
export { interp1d, CubicSpline, PchipInterpolator, Akima1DInterpolator, BarycentricInterpolator }
export { splrep, splev, RegularGridInterpolator, griddata, interpn, lagrange }
export { fft, ifft, rfft, irfft, fft2, ifft2, fftfreq, rfftfreq, fftshift, ifftshift }
export { periodogram, welch, csd, coherence, stft, istft, spectrogram }
export { get_window, butter, cheby1, cheby2, firwin, iirfilter, freqz, lfilter, filtfilt }
export { sosfilt, sosfiltfilt, tf2sos, zpk2tf, savgol_filter, medfilt, detrend, resample }
export { decimate, hilbert }
export { convolve, correlate, convolve2d, correlate2d, fftconvolve, oaconvolve, deconvolve }
export { choose_conv_method }
export { Distribution, uniform, expon, gamma, beta, chi2, t, f, binom, poisson, lognorm }
export { erf, erfc, gammaln, gammainc, gammaincc, betainc, betaln, ndtr, ndtri }
export { DescribeResult, describe, gmean, hmean, sem, zscore, skew, kurtosis, mode, trim_mean }
export { iqr, entropy, pearsonr, spearmanr, kendalltau, linregress }
export { ttest_1samp, ttest_ind, ttest_rel, chisquare, chi2_contingency, f_oneway }
export { mannwhitneyu, wilcoxon, kruskal, ks_2samp, kstest, shapiro, gaussian_kde }
export { cdist, pdist, squareform, distance_matrix, euclidean, cityblock, minkowski }
export { chebyshev, mahalanobis, KDTree, ConvexHull, Delaunay, Voronoi }

#: The package version, which `jaithon.package.json` also carries.
pub let VERSION = "0.1.0"

#: The normal distribution, under the name scipy gives it.
#:
#: Imported aliased because `linalg.norm` is the matrix norm and both belong in
#: the facade; scipy keeps them apart by module and this package cannot.
pub let normal = norm_dist

#: Whether the machine has a GPU backend that jaisci can use.
#:
#: The `dense_*` factorisations and the scalar solvers run without one; `Array`
#: and everything built on it do not.
pub fn is_available() -> bool { return gpu.is_available() }

#: Name of the active GPU, or `"none"` when none exists.
pub fn device_name() -> str { return gpu.device_name() }
