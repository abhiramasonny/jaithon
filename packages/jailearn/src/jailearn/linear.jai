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
from std.gpu import Kernel
from std.os import env_or

import jaitensor as jt
from jaitensor import Tensor

from .base import Estimator, EstimatorKind, Params
from .base import affine_columns, cached_kernel, check_fitted, check_matrix, check_pair
from .base import check_width, cholesky_solve, column_moments, param_bool, param_float
from .base import HOST_SOLVE_ORDER, LabelIndex, host_cholesky_solve, param_int, params_of
from .base import require_target
from .shared import Descent, linear_scores

let SOURCE = """#include <metal_stdlib>
using namespace metal;

/* A scaler's fitted state from the column moments, in place: the mean
 * becomes the offset `mean_` records (zero when not centring) and the
 * variance becomes the reciprocal deviation `scale_` records (one for a
 * column that does not vary, or when not scaling). Precise square root and
 * division, so the state matches the double-precision host arithmetic this
 * replaced to the last bit float32 can hold. */
kernel void lin_scaler_state(device float* mean [[buffer(0)]],
                             device float* variance [[buffer(1)]],
                             constant uint& width [[buffer(2)]],
                             constant uint& with_mean [[buffer(3)]],
                             constant uint& with_std [[buffer(4)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= width) return;
    float deviation = precise::sqrt(variance[id]);
    bool usable = with_std != 0u && deviation > 1.0e-12f;
    variance[id] = usable ? precise::divide(1.0f, deviation) : 1.0f;
    if (with_mean == 0u) mean[id] = 0.0f;
}

/* `X` copied into a design `padded` columns wide: its own columns, a column
 * of ones (the intercept, and the row that sums everything else in a
 * product), then zeros to a multiple of four so the products below take the
 * GEMM's fast route. Zero columns add nothing to any product. */
kernel void lin_augment(device const float* x [[buffer(0)]],
                        device float* design [[buffer(1)]],
                        constant uint& rows [[buffer(2)]],
                        constant uint& width [[buffer(3)]],
                        constant uint& padded [[buffer(4)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= rows * padded) return;
    uint row = id / padded;
    uint column = id - row * padded;
    design[id] = column < width ? x[row * width + column] : (column == width ? 1.0f : 0.0f);
}

/* One Newton step's right-hand side for a binary logistic loss, a row at a
 * time: `p = sigmoid(z)`, then the row of `[s * design | p - t | loss | 0..]`
 * with `s = p (1 - p)`, so one product `designT . this` holds the weighted
 * Hessian, the gradient and -- in the ones row -- the total loss together.
 * Both `p` and `1 - p` come from `exp(-|z|)`, which never overflows and never
 * loses `1 - p` to cancellation when `p` is near one. */
kernel void lin_logit_rows(device const float* design [[buffer(0)]],
                           device const float* z [[buffer(1)]],
                           device const float* t [[buffer(2)]],
                           device float* out [[buffer(3)]],
                           constant uint& rows [[buffer(4)]],
                           constant uint& padded [[buffer(5)]],
                           constant uint& stride [[buffer(6)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= rows * stride) return;
    uint row = id / stride;
    uint column = id - row * stride;
    float margin = z[row];
    float e = exp(-fabs(margin));
    float near = 1.0f / (1.0f + e);
    float far = e / (1.0f + e);
    float p = margin >= 0.0f ? near : far;
    float q = margin >= 0.0f ? far : near;
    float value = 0.0f;
    if (column < padded) {
        value = p * q * design[row * padded + column];
    } else if (column == padded) {
        value = t[row] > 0.5f ? -q : p;
    } else if (column == padded + 1u) {
        float signed_margin = t[row] > 0.5f ? margin : -margin;
        value = max(-signed_margin, 0.0f) + log(1.0f + exp(-fabs(signed_margin)));
    }
    out[id] = value;
}

/* `lin_logit_rows` with the margin taken here, one thread a row: the row's
 * dot product with `theta`, then its whole output row. Saves the product
 * that would otherwise write the margins and its dispatch. */
kernel void lin_logit_rows_fused(device const float* design [[buffer(0)]],
                                 device const float* theta [[buffer(1)]],
                                 device const float* t [[buffer(2)]],
                                 device float* out [[buffer(3)]],
                                 constant uint& rows [[buffer(4)]],
                                 constant uint& padded [[buffer(5)]],
                                 constant uint& stride [[buffer(6)]],
                                 uint row [[thread_position_in_grid]]) {
    if (row >= rows) return;
    device const float4* line = (device const float4*)(design + ulong(row) * padded);
    device const float4* weights = (device const float4*)theta;
    float4 acc = 0.0f;
    for (uint j = 0u; j < padded / 4u; ++j) acc += line[j] * weights[j];
    float margin = acc.x + acc.y + acc.z + acc.w;
    float e = exp(-fabs(margin));
    float near = 1.0f / (1.0f + e);
    float far = e / (1.0f + e);
    float p = margin >= 0.0f ? near : far;
    float q = margin >= 0.0f ? far : near;
    float s = p * q;
    device float4* target = (device float4*)(out + ulong(row) * stride);
    for (uint j = 0u; j < padded / 4u; ++j) target[j] = s * line[j];
    bool positive = t[row] > 0.5f;
    float signed_margin = positive ? margin : -margin;
    float loss = max(-signed_margin, 0.0f) + log(1.0f + exp(-fabs(signed_margin)));
    target[padded / 4u] = float4(positive ? -q : p, loss, 0.0f, 0.0f);
    for (uint j = padded / 4u + 1u; j < stride / 4u; ++j) target[j] = 0.0f;
}

/* The normal equations' operand for a ridge fit, `stride` columns a row:
 * the row less the column means, then the target less its mean, then zeros
 * to a multiple of four, so that ONE product `thisT . this` holds `XcT Xc`
 * and `XcT yc` together and takes the GEMM's fast route whatever the width.
 * `centre` 0 leaves both uncentred (no intercept). */
kernel void lin_ridge_pack(device const float* x [[buffer(0)]],
                           device const float* y [[buffer(1)]],
                           device const float* mean [[buffer(2)]],
                           device const float* target_mean [[buffer(3)]],
                           device float* out [[buffer(4)]],
                           constant uint& rows [[buffer(5)]],
                           constant uint& width [[buffer(6)]],
                           constant uint& stride [[buffer(7)]],
                           constant uint& centre [[buffer(8)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= rows * stride) return;
    uint row = id / stride;
    uint column = id - row * stride;
    float value = 0.0f;
    if (column < width) {
        value = x[row * width + column] - (centre != 0u ? mean[column] : 0.0f);
    } else if (column == width) {
        value = y[row] - (centre != 0u ? target_mean[0] : 0.0f);
    }
    out[id] = value;
}

/* `sigmoid(z)`, stably, as the probability of the second class beside the
 * first's. */
kernel void lin_logit_proba(device const float* z [[buffer(0)]],
                            device float* out [[buffer(1)]],
                            constant uint& rows [[buffer(2)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= rows) return;
    float margin = z[id];
    float e = exp(-fabs(margin));
    float near = 1.0f / (1.0f + e);
    float far = e / (1.0f + e);
    float p = margin >= 0.0f ? near : far;
    out[id * 2u] = margin >= 0.0f ? far : near;
    out[id * 2u + 1u] = p;
}

kernel void lin_threshold(device const float* z [[buffer(0)]],
                          device float* out [[buffer(1)]],
                          constant uint& rows [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    if (id < rows) out[id] = z[id] > 0.0f ? 1.0f : 0.0f;
}

/* `-mean * scale`, the offset `affine_columns` adds after it multiplies. */
kernel void lin_scaler_shift(device const float* mean [[buffer(0)]],
                             device const float* scale [[buffer(1)]],
                             device float* shift [[buffer(2)]],
                             constant uint& width [[buffer(3)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= width) return;
    shift[id] = -mean[id] * scale[id];
}
"""

fn linear_kernel(entry: str) -> Kernel { return cached_kernel(SOURCE, entry) }

#: `JAILEARN_SCALER_ON_DEVICE=0` restores reading the moments back and
#: building the scaler's state on the host, which waits for the device twice
#: per `fit_transform`.
let _SCALER_ON_DEVICE = env_or("JAILEARN_SCALER_ON_DEVICE", "1") != "0"

#: `JAILEARN_LOGIT_FUSED=0` takes a logistic fit's margins with a separate
#: product (`jt.matmul`) before the row kernel, instead of inside it.
let _LOGIT_FUSED = env_or("JAILEARN_LOGIT_FUSED", "1") != "0"

#: `JAILEARN_RIDGE_PACKED=0` restores `Ridge.fit` as two products over a
#: centred copy of `X` (`XT X`, then `XT y`) with the means read back before
#: centring and the target summed on the host, instead of one product over a
#: packed `[Xc | yc | 0..]` and a single wait.
let _RIDGE_PACKED = env_or("JAILEARN_RIDGE_PACKED", "1") != "0"

#: Centre every column and scale it to unit variance.
pub class StandardScaler: Estimator {
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
        if _SCALER_ON_DEVICE {
            linear_kernel("lin_scaler_state").dispatch_async(
                [mean.buffer(), variance.buffer()],
                self.width,
                [
                    self.width,
                    if self.with_mean { 1 } else { 0 },
                    if self.with_std { 1 } else { 0 },
                ]
            )
            self.mean_ = mean
            self.scale_ = variance
            return
        }
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
        if _SCALER_ON_DEVICE {
            let shift = Tensor([self.width])
            defer { shift.free() }
            linear_kernel("lin_scaler_shift").dispatch_async(
                [mean.buffer(), scale.buffer(), shift.buffer()],
                self.width,
                [self.width]
            )
            return affine_columns(x, scale, shift)
        }
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
pub class Ridge: Estimator {
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
        if _RIDGE_PACKED {
            self.fit_packed(x, target)
            return
        }

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

    #: The fit as one product: `lin_ridge_pack` lays out `[Xc | yc | 0..]`
    #: and its gram holds both sides of the normal equations, read back once
    #: and solved on the host when the width allows (`HOST_SOLVE_ORDER`).
    #: Centred on the device with `column_moments`, so the precision argument
    #: for centring first holds exactly as before; only the means cross back,
    #: after the wait the solve needed anyway.
    fn fit_packed(self, x: Tensor, target: Tensor) -> void {
        let rows = x.shape[0]
        let width = self.width
        let stride = round_up4(width + 1)
        var mean: Tensor? = null
        var target_mean: Tensor? = null
        if self.fit_intercept {
            let column = target.view(0, [rows, 1])
            let (centres, spread) = column_moments(x)
            let (middle, target_spread) = column_moments(column)
            spread.free()
            target_spread.free()
            column.free()
            mean = centres
            target_mean = middle
        }
        let packed = Tensor([rows, stride])
        let anything = x.buffer()
        linear_kernel("lin_ridge_pack").dispatch_async(
            [
                x.buffer(),
                target.buffer(),
                if mean is not null { mean.buffer() } else { anything },
                if target_mean is not null { target_mean.buffer() } else { anything },
                packed.buffer(),
            ],
            rows * stride,
            [rows, width, stride, if self.fit_intercept { 1 } else { 0 }]
        )
        let product = jt.matmul(packed, packed, trans_a: true)
        packed.free()
        let g = product.download()
        product.free()

        var system = [0.0 for _slot in 0..width * width]
        var rhs = [0.0 for _slot in 0..width]
        for a in 0..width {
            for b in 0..width { system[a * width + b] = g[a * stride + b] }
            system[a * width + a] += self.alpha
            rhs[a] = g[a * stride + width]
        }
        var slopes: list[float] = []
        if width <= HOST_SOLVE_ORDER {
            slopes = host_cholesky_solve(system, rhs, width)
        } else {
            let a = Tensor([width, width], system)
            let b = Tensor([width], rhs)
            let solved = cholesky_solve(a, b)
            slopes = solved.download()
            a.free()
            b.free()
            solved.free()
        }
        if mean is not null and target_mean is not null {
            let centres = mean.download()
            var offset = target_mean.download()[0]
            for index in 0..width { offset -= centres[index] * slopes[index] }
            self.intercept_ = offset
            mean.free()
            target_mean.free()
        }
        self.coef_ = Tensor([1, width], slopes)
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

fn round_up4(n: int) -> int { return (n + 3) // 4 * 4 }

#: Two-class logistic regression with scikit-learn's objective:
#: `C * sum(log(1 + exp(-y z))) + |w|^2 / 2`, the intercept unpenalised.
#:
#: Fitted by Newton's method rather than scikit-learn's L-BFGS. With the
#: features in the tens, the Hessian is small enough to solve on the host
#: (`host_cholesky_solve`) and costs one product to form, so each iteration is
#: two dispatches over the samples and one read back of a `(d+1) x (d+3)`
#: matrix: `lin_logit_rows` lays out `[s * X1 | p - t | loss]` beside the
#: design `X1 = [X | 1]`, and `X1T` times that is the weighted Hessian, the
#: gradient and the loss at once. Newton's quadratic convergence then takes
#: around eight of those where L-BFGS takes tens of gradient evaluations.
#:
#: A step that raises the objective is halved until it does not, which is all
#: the globalisation a strictly convex objective needs; a Hessian float32 left
#: barely definite is damped on its diagonal until it factors.
#:
#: The fit stops when the largest gradient component of the MEAN loss --
#: scikit-learn's scale for `tol` -- is below `tol / 100`, or when a step no
#: longer moves the coefficients; tighter than scikit-learn's own stop, so the
#: two land on the same optimum rather than on two points near it.
pub class LogisticRegression: Estimator {
    pub var C: float
    pub var fit_intercept: bool
    pub var max_iter: int
    pub var tol: float
    pub var coef_: Tensor?
    pub var intercept_: float
    pub var classes_: list[float]
    pub var n_iter_: int
    pub var converged_: bool
    var width: int
    var labels: LabelIndex?

    pub fn init(
        self,
        C: float = 1.0,
        fit_intercept: bool = true,
        max_iter: int = 100,
        tol: float = 1.0e-4
    ) {
        if C <= 0.0 { throw ValueError(f"LogisticRegression needs a positive C, got {C}") }
        if max_iter <= 0 { throw ValueError(f"max_iter must be positive, got {max_iter}") }
        self.C = C
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = null
        self.intercept_ = 0.0
        self.classes_ = []
        self.n_iter_ = 0
        self.converged_ = false
        self.width = 0
        self.labels = null
    }

    pub fn kind(self) -> EstimatorKind { return EstimatorKind.Classifier }

    pub fn fit(self, x: Tensor, y: Tensor? = null) -> void {
        check_matrix(x)
        let target = require_target(y, "LogisticRegression")
        check_pair(x, target)
        self.free()
        let labels = LabelIndex.from_labels(target)
        if labels.count() != 2 {
            let count = labels.count()
            labels.free()
            throw ValueError(f"LogisticRegression here is two-class; y has {count} classes")
        }
        self.labels = labels
        self.classes_ = labels.classes
        let codes = labels.encode(target)
        defer { codes.free() }

        let rows = x.shape[0]
        let width = x.shape[1]
        self.width = width
        let padded = round_up4(width + 1)
        let stride = round_up4(padded + 2)
        let order = if self.fit_intercept { width + 1 } else { width }
        let design = Tensor([rows, padded])
        defer { design.free() }
        linear_kernel("lin_augment").dispatch_async(
            [x.buffer(), design.buffer()],
            rows * padded,
            [rows, width, padded]
        )
        let rhs = Tensor([rows, stride])
        defer { rhs.free() }
        let weights = Tensor([padded, 1])
        defer { weights.free() }

        let c = self.C
        let mean_scale = 1.0 / (c * float(rows))
        let floor = self.tol / 100.0
        var theta = [0.0 for _slot in 0..padded]
        var previous = [0.0 for _slot in 0..padded]
        var direction = [0.0 for _slot in 0..padded]
        var step = 1.0
        var best = 0.0
        var started = false

        let descent = Descent(self.max_iter, 0.0)
        descent.run(fn(iteration: int) -> float {
            weights.upload(theta)
            if _LOGIT_FUSED {
                linear_kernel("lin_logit_rows_fused").dispatch_async(
                    [design.buffer(), weights.buffer(), codes.buffer(), rhs.buffer()],
                    rows,
                    [rows, padded, stride]
                )
            } else {
                let margins = jt.matmul(design, weights)
                linear_kernel("lin_logit_rows").dispatch_async(
                    [design.buffer(), margins.buffer(), codes.buffer(), rhs.buffer()],
                    rows * stride,
                    [rows, padded, stride]
                )
                margins.free()
            }
            let product = jt.matmul(design, rhs, trans_a: true)
            let g = product.download()
            product.free()

            var penalty = 0.0
            for index in 0..width { penalty += theta[index] * theta[index] }
            let objective = c * g[width * stride + padded + 1] + 0.5 * penalty
            if math.is_nan(objective) { return math.NAN }
            if started and objective > best + 1.0e-6 * math.fabs(best) and step > 1.0e-6 {
                step *= 0.5
                for index in 0..order { theta[index] = previous[index] + step * direction[index] }
                return 1.0
            }

            var gradient = [0.0 for _slot in 0..order]
            var largest = 0.0
            for a in 0..order {
                let ridge = if a < width { theta[a] } else { 0.0 }
                gradient[a] = c * g[a * stride + padded] + ridge
                largest = max(largest, math.fabs(gradient[a]) * mean_scale)
            }
            if largest <= floor { return 0.0 }

            var hessian = [0.0 for _slot in 0..order * order]
            for a in 0..order {
                for b in 0..order { hessian[a * order + b] = c * g[a * stride + b] }
                if a < width { hessian[a * order + a] += 1.0 }
            }
            let newton = newton_direction(hessian, gradient, order)
            var moved = 0.0
            var size = 1.0
            for index in 0..order {
                previous[index] = theta[index]
                direction[index] = -newton[index]
                theta[index] -= newton[index]
                moved = max(moved, math.fabs(newton[index]))
                size = max(size, math.fabs(theta[index]))
            }
            best = objective
            started = true
            step = 1.0
            return if moved <= 1.0e-7 * size { 0.0 } else { moved }
        })
        self.n_iter_ = descent.n_iter
        self.converged_ = descent.converged
        self.coef_ = Tensor([1, width], theta[0:width])
        self.intercept_ = if self.fit_intercept { theta[width] } else { 0.0 }
    }

    #: `X . coef_ + intercept_`, the log-odds of the second class.
    pub fn decision_function(self, x: Tensor) -> Tensor {
        check_fitted(self.is_fitted(), "LogisticRegression")
        check_width(x, self.width, "LogisticRegression")
        let coef = self.coef_
        if coef is null { throw RuntimeError("LogisticRegression lost its coefficients") }
        let bias = Tensor([1], [self.intercept_])
        defer { bias.free() }
        let scores = linear_scores(x, coef, bias)
        return jt.reshape(scores, [x.shape[0]])
    }

    #: `[n_samples, 2]`, the columns in the order of `classes_`.
    pub fn predict_proba(self, x: Tensor) -> Tensor {
        let scores = self.decision_function(x)
        defer { scores.free() }
        let out = Tensor([x.shape[0], 2])
        linear_kernel("lin_logit_proba").dispatch_async(
            [scores.buffer(), out.buffer()],
            x.shape[0],
            [x.shape[0]]
        )
        return out
    }

    #: The second class where the log-odds are positive, as scikit-learn.
    pub fn predict(self, x: Tensor) -> Tensor {
        let scores = self.decision_function(x)
        defer { scores.free() }
        let codes = Tensor([x.shape[0]])
        defer { codes.free() }
        linear_kernel("lin_threshold").dispatch_async(
            [scores.buffer(), codes.buffer()],
            x.shape[0],
            [x.shape[0]]
        )
        let labels = self.labels
        if labels is null { throw RuntimeError("LogisticRegression lost its classes") }
        return labels.decode(codes)
    }

    pub fn get_params(self) -> Params {
        return params_of([
            ("C", self.C),
            ("fit_intercept", self.fit_intercept),
            ("max_iter", self.max_iter),
            ("tol", self.tol),
        ])
    }

    pub fn set_params(self, params: Params) -> void {
        self.C = param_float(params, "C", self.C)
        self.fit_intercept = param_bool(params, "fit_intercept", self.fit_intercept)
        self.max_iter = param_int(params, "max_iter", self.max_iter)
        self.tol = param_float(params, "tol", self.tol)
    }

    pub fn clone_unfitted(self) -> LogisticRegression {
        return LogisticRegression(self.C, self.fit_intercept, self.max_iter, self.tol)
    }

    pub fn is_fitted(self) -> bool { return self.coef_ is not null }

    pub fn feature_count(self) -> int { return self.width }

    pub fn free(self) -> void {
        let coef = self.coef_
        if coef is not null { coef.free() }
        let labels = self.labels
        if labels is not null { labels.free() }
        self.coef_ = null
        self.labels = null
        self.intercept_ = 0.0
    }
}

#: Solve `H d = g`, damping the diagonal by growing multiples of its largest
#: entry until float32's rounding no longer leaves `H` indefinite.
fn newton_direction(hessian: list[float], gradient: list[float], order: int) -> list[float] {
    var scale = 0.0
    for index in 0..order { scale = max(scale, math.fabs(hessian[index * order + index])) }
    var damping = 0.0
    for _attempt in 0..12 {
        try {
            return host_cholesky_solve(hessian, gradient, order, damping: damping)
        } catch _refused: ValueError {
            damping = if damping == 0.0 { 1.0e-7 * max(scale, 1.0) } else { damping * 10.0 }
        }
    }
    throw RuntimeError("the logistic Hessian would not factor, even damped")
}
