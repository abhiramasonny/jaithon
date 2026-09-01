#: Factorizations, solvers and norms — the decompositions everything else in
#: numerical work is written on top of.
#:
#: ```
#: let (q, r) = qr(a)
#: let x = solve(a, b)
#: let (u, s, vh) = svd(a)
#: print(norm(a, "fro").item())
#: ```
#:
#: **Everything but `norm` runs on downloaded host values.** One `to_list` on
#: the way in, one `upload` on the way out, whatever the batch holds. That is
#: the same call `jaicv/core/linalg.jai` makes, and for the same reason: it is
#: handed three-by-threes, nine-by-nines and Jacobians a few dozen wide, sizes
#: where a kernel launch per panel costs more than the arithmetic inside the
#: panel. A Householder step on a 64-wide matrix is four thousand multiplies,
#: which an M2 Max finishes in less time than it takes to encode the dispatch
#: that would have done it. The reasoning holds up to a few hundred on a side.
#:
#: **Past that the trailing-matrix update is the whole cost**, and it is a
#: GEMM: an `n` by `n` LU is `n^3/3` operations of which all but `O(n^2)` live
#: in that one update. `blocked_panel` is the hook that switches to it — see
#: its note for the crossover and what still has to be wired.
#:
#: **`norm` is the exception and is a kernel**, because it is the one entry
#: point here that walks the whole array rather than one small matrix at a
#: time: a Frobenius norm of a ten-million-element array must not become ten
#: million host iterations, and a batched row-sum is a reduction like any
#: other. Two dispatches, partials then a fold.
#:
#: Stacked inputs loop over the leading axes: an `(..., m, n)` array is a
#: batch of `m` by `n` matrices and every entry point here takes one. The
#: batch axes of a solver's two arguments broadcast against each other.
#:
#: `eig` is real only. A real matrix can have complex conjugate eigenvalues,
#: and there is no `COMPLEX64` answer this can give without carrying complex
#: arithmetic through the whole Schur iteration, so it raises with the pair
#: named rather than returning the real part and calling it an eigenvalue.
#: `eigh` has no such case — a symmetric matrix's spectrum is real.
#:
#: A singular matrix raises `ValueError`, as `jaicv`'s `inverse` does; numpy's
#: `LinAlgError` has no counterpart in this language's error hierarchy.
from .linalg.common import BLOCKED_FROM, BLOCKED_PANEL, FLOAT32_EPS, blocked_panel
from .linalg.decompose import eig, eigh, eigvals, eigvalsh, qr, svd, svdvals
from .linalg.lstsq import cond, lstsq, matrix_rank, pinv
from .linalg.norm import norm
from .linalg.solve import cholesky, det, inv, lu_factor, lu_solve, slogdet, solve, triangular_solve

export { BLOCKED_FROM, BLOCKED_PANEL, FLOAT32_EPS, blocked_panel, norm }
export { cholesky, det, inv, lu_factor, lu_solve, slogdet, solve, triangular_solve }
export { eig, eigh, eigvals, eigvalsh, qr, svd, svdvals }
export { cond, lstsq, matrix_rank, pinv }
