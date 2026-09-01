#: The entry points that hand back a factorization: QR, SVD and the two
#: eigenvalue problems.
from ..array import NDArray, element_count
from .common import _emit, _extend, _require_square, _slice_matrix, _stacked
from .eigen import _eig_host, _jacobi_eigh, _symmetrize
from .qr import _accumulate_reflectors, _householder_qr, _upper_triangle
from .svd import _svd_host

#: `A = QR` with orthonormal `Q`, by Householder reflections.
#:
#: `"reduced"` gives `Q` as `m` by `k` and `R` as `k` by `n` for
#: `k = min(m, n)`; `"complete"` widens `Q` to `m` by `m` and `R` to `m` by
#: `n`. numpy's `"r"` and `"raw"` have no spelling here — a tuple return
#: cannot change its arity — so take the second element of `"reduced"`.
pub fn qr(a: NDArray, mode: str = "reduced") -> (NDArray, NDArray) {
    let (values, batch, m, n) = _stacked(a, "qr")
    if mode != "reduced" and mode != "complete" {
        throw ValueError(f"qr does not know the mode {mode}; it takes \"reduced\" or \"complete\"")
    }
    let k = min(m, n)
    let wide = if mode == "complete" { m } else { k }
    var bases: list[float] = []
    var triangles: list[float] = []
    for index in 0..element_count(batch) {
        var work = _slice_matrix(values, index * m * n, m * n)
        let (vs, norms) = _householder_qr(work, m, n, k)
        _extend(bases, _accumulate_reflectors(vs, norms, m, k, wide))
        _extend(triangles, _upper_triangle(work, n, wide))
    }
    return (_emit(bases, batch + [m, wide]), _emit(triangles, batch + [wide, n]))
}

#: `A = U S V^T`, singular values descending.
#:
#: `full_matrices` widens `U` and `V^T` to the square bases that span the two
#: whole spaces rather than just the range and the row space. numpy's
#: `compute_uv` is `svdvals` here, for the same reason `qr` has no `"r"`.
pub fn svd(a: NDArray, full_matrices: bool = true) -> (NDArray, NDArray, NDArray) {
    let (values, batch, m, n) = _stacked(a, "svd")
    let k = min(m, n)
    let wide = if full_matrices { m } else { k }
    let tall = if full_matrices { n } else { k }
    var lefts: list[float] = []
    var spectra: list[float] = []
    var rights: list[float] = []
    for index in 0..element_count(batch) {
        let matrix = _slice_matrix(values, index * m * n, m * n)
        let (u, s, vt) = _svd_host(matrix, m, n, full_matrices, true)
        _extend(lefts, u)
        _extend(spectra, s)
        _extend(rights, vt)
    }
    return (
        _emit(lefts, batch + [m, wide]),
        _emit(spectra, batch + [k]),
        _emit(rights, batch + [tall, n])
    )
}

#: The singular values alone, descending.
#:
#: Worth its own entry point rather than dropping two of `svd`'s three: the
#: bases are the quadratic-by-linear part of the work, and every caller here
#: that only needs the values — `matrix_rank`, `cond`, the 2 and nuclear
#: norms — skips accumulating them.
pub fn svdvals(a: NDArray) -> NDArray {
    let (values, batch, m, n) = _stacked(a, "svdvals")
    var out: list[float] = []
    for index in 0..element_count(batch) {
        let matrix = _slice_matrix(values, index * m * n, m * n)
        let (_u, s, _vt) = _svd_host(matrix, m, n, false, false)
        _extend(out, s)
    }
    return _emit(out, batch + [min(m, n)])
}

fn _uplo_lower(uplo: str) -> bool {
    if uplo == "L" or uplo == "l" { return true }
    if uplo == "U" or uplo == "u" { return false }
    throw ValueError(f"uplo is \"L\" or \"U\", not {uplo}")
}

#: Eigenvalues and eigenvectors of a symmetric matrix, ascending.
#:
#: Only the triangle `uplo` names is read. The eigenvectors are the columns of
#: the second result, so `a @ v[:, i] == w[i] * v[:, i]`.
pub fn eigh(a: NDArray, uplo: str = "L") -> (NDArray, NDArray) {
    let (values, batch, m, n) = _stacked(a, "eigh")
    _require_square(m, n, "eigh")
    let lower = _uplo_lower(uplo)
    var spectra: list[float] = []
    var bases: list[float] = []
    for index in 0..element_count(batch) {
        let matrix = _symmetrize(values, index * n * n, n, lower)
        let (w, q) = _jacobi_eigh(matrix, n, true)
        _extend(spectra, w)
        _extend(bases, q)
    }
    return (_emit(spectra, batch + [n]), _emit(bases, batch + [n, n]))
}

#: The eigenvalues of a symmetric matrix, ascending, without their vectors.
pub fn eigvalsh(a: NDArray, uplo: str = "L") -> NDArray {
    let (values, batch, m, n) = _stacked(a, "eigvalsh")
    _require_square(m, n, "eigvalsh")
    let lower = _uplo_lower(uplo)
    var out: list[float] = []
    for index in 0..element_count(batch) {
        let matrix = _symmetrize(values, index * n * n, n, lower)
        let (w, _q) = _jacobi_eigh(matrix, n, false)
        _extend(out, w)
    }
    return _emit(out, batch + [n])
}

#: Eigenvalues and right eigenvectors of a general matrix — real ones only.
#:
#: Hessenberg reduction, then shifted QR to the real Schur form, then back
#: substitution for the vectors. A real matrix can have complex conjugate
#: eigenvalues and this raises `ValueError` naming the pair when it meets one:
#: there is no `COMPLEX64` answer it could give without carrying complex
#: arithmetic through the whole iteration, and returning the real part would
#: be a wrong answer rather than a missing one. `eigh` has no such case.
pub fn eig(a: NDArray) -> (NDArray, NDArray) {
    let (values, batch, m, n) = _stacked(a, "eig")
    _require_square(m, n, "eig")
    var spectra: list[float] = []
    var bases: list[float] = []
    for index in 0..element_count(batch) {
        let (w, z) = _eig_host(_slice_matrix(values, index * n * n, n * n), n, true)
        _extend(spectra, w)
        _extend(bases, z)
    }
    return (_emit(spectra, batch + [n]), _emit(bases, batch + [n, n]))
}

#: The eigenvalues of a general matrix, without their vectors. Real only, as
#: `eig` is.
pub fn eigvals(a: NDArray) -> NDArray {
    let (values, batch, m, n) = _stacked(a, "eigvals")
    _require_square(m, n, "eigvals")
    var out: list[float] = []
    for index in 0..element_count(batch) {
        let (w, _z) = _eig_host(_slice_matrix(values, index * n * n, n * n), n, false)
        _extend(out, w)
    }
    return _emit(out, batch + [n])
}
