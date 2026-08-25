#: The estimators every scikit-learn program starts with.
#:
#: `StandardScaler` and `Ridge` between them exercise both halves of the
#: `Estimator` trait -- one transforms, one predicts -- and both are built
#: entirely from primitives that were already in `base.jai` and `shared.jai`.
#: That is why they were written first: the scaffolding had been here since the
#: package was created and nothing had ever gone through it, so the first
#: estimators are as much a test of the trait as of themselves.
#:
#: `Ridge` solves the normal equations rather than factorising `X`. That is
#: scikit-learn's `solver="cholesky"`, and it is the right choice here for the
#: same reasons: `cholesky_solve` is the primitive in hand, the matrix it forms
#: is `features x features` rather than `samples x features`, and ridge's own
#: penalty is what keeps it conditioned. A caller with a rank-deficient `X` and
#: an `alpha` of zero gets the refusal `cholesky_factor` raises, which is the
#: honest answer from this solver rather than a plausible wrong one.
import std.math

import jaitensor as jt
from jaitensor import Tensor

from .base import EstimatorKind, Params
from .base import affine_columns, check_fitted, check_matrix, check_pair, check_width
from .base import cholesky_solve, column_moments, param_bool, param_float, params_of
from .shared import linear_scores

#: Centre every column and scale it to unit variance.
pub class StandardScaler {
    pub var with_mean: bool
    pub var with_std: bool
    pub var mean_: Tensor?
    pub var scale_: Tensor?
    var width: int

    pub fn init(self, with_mean: bool = true, with_std: bool = true) {
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = null
        self.scale_ = null
        self.width = 0
    }

    pub fn kind(self) -> EstimatorKind { return EstimatorKind.Transformer }

    #: A column that does not vary is left alone rather than divided by zero.
    #: scikit-learn does the same and for the same reason: a constant column
    #: carries no information, and scaling it to unit variance would turn
    #: whatever rounding noise it holds into a feature.
    pub fn fit(self, x: Tensor, y: Tensor? = null) -> void {
        check_matrix(x)
        self.free()
        self.width = x.shape[1]
        let (mean, variance) = column_moments(x)
        defer { mean.free() }
        defer { variance.free() }

        let centres = mean.download()
        let spreads = variance.download()
        var offsets: list[float] = []
        var factors: list[float] = []
        for index in 0..self.width {
            offsets.push(if self.with_mean { centres[index] } else { 0.0 })
            let deviation = math.sqrt(spreads[index])
            let usable = self.with_std and deviation > 1.0e-12
            factors.push(if usable { 1.0 / deviation } else { 1.0 })
        }
        self.mean_ = Tensor([self.width], offsets)
        self.scale_ = Tensor([self.width], factors)
    }

    #: `affine_columns` computes `x * scale + offset`, so the centring has to
    #: be folded into the offset as `-mean * scale` rather than subtracted
    #: first -- one pass instead of two.
    pub fn transform(self, x: Tensor) -> Tensor {
        check_fitted(self.is_fitted(), "StandardScaler")
        check_width(x, self.width, "StandardScaler")
        let scale = self.scale_
        let mean = self.mean_
        if scale is null or mean is null { throw RuntimeError("StandardScaler lost its state") }
        let factors = scale.download()
        let centres = mean.download()
        var shift: list[float] = []
        for index in 0..self.width { shift.push(-centres[index] * factors[index]) }
        let offset = Tensor([self.width], shift)
        defer { offset.free() }
        return affine_columns(x, scale, offset)
    }

    #: Exact, because the transform is affine.
    pub fn inverse_transform(self, x: Tensor) -> Tensor {
        check_fitted(self.is_fitted(), "StandardScaler")
        check_width(x, self.width, "StandardScaler")
        let scale = self.scale_
        let mean = self.mean_
        if scale is null or mean is null { throw RuntimeError("StandardScaler lost its state") }
        let factors = scale.download()
        var widen: list[float] = []
        for value in factors { widen.push(1.0 / value) }
        let back = Tensor([self.width], widen)
        defer { back.free() }
        return affine_columns(x, back, mean)
    }

    pub fn fit_transform(self, x: Tensor, y: Tensor? = null) -> Tensor {
        self.fit(x, y)
        return self.transform(x)
    }

    pub fn get_params(self) -> Params {
        return params_of([("with_mean", self.with_mean), ("with_std", self.with_std)])
    }

    pub fn set_params(self, params: Params) -> void {
        self.with_mean = param_bool(params, "with_mean", self.with_mean)
        self.with_std = param_bool(params, "with_std", self.with_std)
    }

    pub fn clone_unfitted(self) -> StandardScaler {
        return StandardScaler(self.with_mean, self.with_std)
    }

    pub fn is_fitted(self) -> bool { return self.scale_ is not null }

    pub fn feature_count(self) -> int { return self.width }

    pub fn free(self) -> void {
        let mean = self.mean_
        if mean is not null { mean.free() }
        let scale = self.scale_
        if scale is not null { scale.free() }
        self.mean_ = null
        self.scale_ = null
    }
}

#: Least squares with an L2 penalty, through the normal equations.
pub class Ridge {
    pub var alpha: float
    pub var fit_intercept: bool
    pub var coef_: Tensor?
    pub var intercept_: float
    var width: int

    pub fn init(self, alpha: float = 1.0, fit_intercept: bool = true) {
        if alpha < 0.0 { throw ValueError(f"Ridge needs a non-negative alpha, got {alpha}") }
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = null
        self.intercept_ = 0.0
        self.width = 0
    }

    pub fn kind(self) -> EstimatorKind { return EstimatorKind.Regressor }

    #: The intercept is fitted by centring, not by a column of ones.
    #:
    #: Both give the same line, but a ones column would be penalised along with
    #: everything else and scikit-learn does not penalise the intercept. Taking
    #: the means out first, solving for the slopes, and recovering the
    #: intercept as `mean(y) - mean(x) . coef` keeps the penalty off it.
    pub fn fit(self, x: Tensor, y: Tensor? = null) -> void {
        check_matrix(x)
        let target = y
        if target is null { throw ValueError("Ridge needs a target") }
        check_pair(x, target)
        self.free()
        self.width = x.shape[1]
        let rows = x.shape[0]

        var centres = [0.0 for _index in 0..self.width]
        var middle = 0.0
        var design = x

        #: `reshape` hands back a VIEW -- its storage is `input.buffer().view(0,
        #: size)` -- and `Tensor.free` releases storage unconditionally, with no
        #: base pointer to tell an owner from a borrower. So freeing a reshape
        #: of the caller's `y` frees the caller's `y`, and the crash lands
        #: wherever they touch it next. Nothing derived from an argument is
        #: freed here; only tensors this function allocated are.
        let column = jt.reshape(target, [rows, 1])
        var response = column
        var owns_design = false
        var owns_response = false

        if self.fit_intercept {
            let (mean, variance) = column_moments(x)
            variance.free()
            centres = mean.download()
            mean.free()
            let ones = Tensor([self.width], [1.0 for _index in 0..self.width])
            let shift = Tensor([self.width], [-value for value in centres])
            design = affine_columns(x, ones, shift)
            owns_design = true
            ones.free()
            shift.free()

            var total = 0.0
            for value in target.download() { total += value }
            middle = total / float(rows)
            let scale_one = Tensor([1], [1.0])
            let drop = Tensor([1], [-middle])
            response = affine_columns(column, scale_one, drop)
            owns_response = true
            scale_one.free()
            drop.free()
        }

        #: `X^T X`, penalised on the diagonal, against `X^T y`.
        let normal = jt.matmul(design, design, trans_a: true)
        var system = normal
        if self.alpha > 0.0 {
            let unit = jt.eye(self.width)
            let penalty = jt.scale(unit, self.alpha)
            unit.free()
            system = jt.add(normal, penalty)
            penalty.free()
            normal.free()
        }
        let rhs = jt.matmul(design, response, trans_a: true)
        let weights = cholesky_solve(system, rhs)
        system.free()
        rhs.free()

        if self.fit_intercept {
            let slopes = weights.download()
            var offset = middle
            for index in 0..self.width { offset -= centres[index] * slopes[index] }
            self.intercept_ = offset
        }

        #: The reshape shares `weights`' storage, so `coef_` becomes its owner
        #: and `weights` must not be freed as well -- see the note above.
        self.coef_ = jt.reshape(weights, [1, self.width])
        if owns_design { design.free() }
        if owns_response { response.free() }
    }

    pub fn predict(self, x: Tensor) -> Tensor {
        check_fitted(self.is_fitted(), "Ridge")
        check_width(x, self.width, "Ridge")
        let coef = self.coef_
        if coef is null { throw RuntimeError("Ridge lost its coefficients") }
        let bias = Tensor([1], [self.intercept_])
        defer { bias.free() }
        #: Again a view over `scores`, so `scores` is not freed: the caller
        #: owns what comes back and freeing it releases the one allocation.
        let scores = linear_scores(x, coef, bias)
        return jt.reshape(scores, [x.shape[0]])
    }

    pub fn get_params(self) -> Params {
        return params_of([("alpha", self.alpha), ("fit_intercept", self.fit_intercept)])
    }

    pub fn set_params(self, params: Params) -> void {
        self.alpha = param_float(params, "alpha", self.alpha)
        self.fit_intercept = param_bool(params, "fit_intercept", self.fit_intercept)
    }

    pub fn clone_unfitted(self) -> Ridge { return Ridge(self.alpha, self.fit_intercept) }

    pub fn is_fitted(self) -> bool { return self.coef_ is not null }

    pub fn feature_count(self) -> int { return self.width }

    pub fn free(self) -> void {
        let coef = self.coef_
        if coef is not null { coef.free() }
        self.coef_ = null
        self.intercept_ = 0.0
    }
}

#: Ordinary least squares -- `Ridge` with no penalty, under scikit-learn's name.
pub fn linear_regression(fit_intercept: bool = true) -> Ridge {
    return Ridge(0.0, fit_intercept)
}
