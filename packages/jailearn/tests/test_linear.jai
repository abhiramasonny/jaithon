#: `StandardScaler` and `Ridge` against scikit-learn, which is where every
#: expected value below came from — `sklearn 1.9.0` run on the same inputs.
#:
#: These are the first estimators the package has had. The trait, the pipeline
#: and the primitives had all been here since it was created and nothing had
#: ever gone through them, so what is checked here is as much the scaffolding
#: as the two estimators: that `fit` then `transform` composes, that
#: `clone_unfitted` really is unfitted, that `get_params` round-trips through
#: `set_params`, and that an unfitted estimator refuses rather than guessing.
import std.gpu as gpu
import std.math
from std.test import assert_eq, assert_near, assert_throws, assert_true
from jaitensor import Tensor
from jailearn import Estimator, EstimatorKind, LogisticRegression, Params, Pipeline, Ridge
from jailearn import StandardScaler
from jailearn import linear_regression, make_pipeline
import jailearn as jl
from jailearn.base import matrix

fn have_gpu() -> bool { return gpu.is_available() }

fn sample() -> Tensor {
    return matrix([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 6.0]])
}

fn target() -> Tensor { return Tensor([5], [3.0, 4.0, 8.0, 9.0, 13.0]) }

fn assert_close(got: list[float], expected: list[float], what: str) -> void {
    assert_eq(got.len(), expected.len(), what)
    for (index, value) in got.enumerate() { assert_near(value, expected[index], 1e-5, what) }
}

fn test_standard_scaler_matches_sklearn() -> void {
    if not have_gpu() { return }
    let x = sample()
    defer { x.free() }
    let scaler = StandardScaler()
    defer { scaler.free() }

    let scaled = scaler.fit_transform(x)
    defer { scaled.free() }
    assert_eq(scaler.kind(), EstimatorKind.Transformer)

    let mean = scaler.mean_
    let scale = scaler.scale_
    assert_true(mean is not null and scale is not null, "fit records both")
    if mean is not null { assert_close(mean.download(), [3.0, 3.2], "mean_") }
    #: scikit-learn stores the deviation; this stores its reciprocal, because
    #: `affine_columns` multiplies. 1/sqrt(2) and 1/1.72046.
    if scale is not null { assert_close(scale.download(), [0.7071068, 0.5812382], "scale_") }

    assert_close(scaled.download()[0:2], [-1.4142136, -0.6974858], "transform")

    let back = scaler.inverse_transform(scaled)
    defer { back.free() }
    assert_close(back.download()[0:2], [1.0, 2.0], "inverse_transform")
}

#: A column that never varies must survive rather than divide by zero.
fn test_standard_scaler_leaves_a_constant_column_alone() -> void {
    if not have_gpu() { return }
    let flat = matrix([[1.0, 7.0], [2.0, 7.0], [3.0, 7.0]])
    defer { flat.free() }
    let scaler = StandardScaler()
    defer { scaler.free() }
    let scaled = scaler.fit_transform(flat)
    defer { scaled.free() }
    let values = scaled.download()
    for row in 0..3 {
        assert_near(values[row * 2 + 1], 0.0, 1e-6, "a constant column centres to zero")
    }
}

fn test_standard_scaler_can_be_told_not_to_centre_or_scale() -> void {
    if not have_gpu() { return }
    let x = sample()
    defer { x.free() }
    let plain = StandardScaler(false, false)
    defer { plain.free() }
    let same = plain.fit_transform(x)
    defer { same.free() }
    assert_close(same.download()[0:2], [1.0, 2.0], "neither centred nor scaled is the identity")
}

fn test_ridge_matches_sklearn() -> void {
    if not have_gpu() { return }
    let x = sample()
    defer { x.free() }
    let y = target()
    defer { y.free() }

    let ridge = Ridge(1.0)
    defer { ridge.free() }
    ridge.fit(x, y)
    assert_eq(ridge.kind(), EstimatorKind.Regressor)

    let coef = ridge.coef_
    assert_true(coef is not null, "fit records coefficients")
    if coef is not null { assert_close(coef.download(), [1.4769648, 0.8753387], "ridge coef_") }
    assert_near(ridge.intercept_, 0.1680217, 1e-5, "ridge intercept_")

    let predicted = ridge.predict(x)
    defer { predicted.free() }
    assert_eq(predicted.shape, [5], "predict is one value a row")
    assert_close(
        predicted.download(),
        [3.395664, 3.99729, 8.100271, 8.701897, 12.804878],
        "ridge predict"
    )

    #: The penalty is what makes this differ from least squares, so a zero
    #: penalty has to give least squares back exactly.
    let ols = linear_regression()
    defer { ols.free() }
    ols.fit(x, y)
    let slopes = ols.coef_
    if slopes is not null { assert_close(slopes.download(), [1.75, 0.75], "ols coef_") }
    assert_near(ols.intercept_, -0.25, 1e-5, "ols intercept_")
}

fn test_ridge_without_an_intercept_matches_sklearn() -> void {
    if not have_gpu() { return }
    let x = sample()
    defer { x.free() }
    let y = target()
    defer { y.free() }
    let ridge = Ridge(1.0, false)
    defer { ridge.free() }
    ridge.fit(x, y)
    let coef = ridge.coef_
    if coef is not null {
        assert_close(coef.download(), [1.5103093, 0.8865979], "no-intercept coef_")
    }
    assert_near(ridge.intercept_, 0.0, 1e-9, "no intercept means zero")
}

#: `fit` must not consume what it was given. It reshapes `y`, and a reshape in
#: jaitensor is a VIEW over the same storage, so an earlier version freed the
#: caller's target and the crash landed wherever they touched it next.
fn test_fit_does_not_free_what_it_was_given() -> void {
    if not have_gpu() { return }
    let x = sample()
    defer { x.free() }
    let y = target()
    defer { y.free() }
    let ridge = Ridge(1.0)
    defer { ridge.free() }
    ridge.fit(x, y)
    assert_close(y.download(), [3.0, 4.0, 8.0, 9.0, 13.0], "the target survives fit")
    assert_close(x.download()[0:2], [1.0, 2.0], "the design matrix survives fit")

    #: And fitting twice on the same data must give the same answer.
    ridge.fit(x, y)
    let coef = ridge.coef_
    if coef is not null { assert_close(coef.download(), [1.4769648, 0.8753387], "refit is stable") }
}

fn test_the_estimator_contract_holds() -> void {
    if not have_gpu() { return }
    let x = sample()
    defer { x.free() }
    let y = target()
    defer { y.free() }

    let ridge = Ridge(0.5, true)
    defer { ridge.free() }
    assert_true(not ridge.is_fitted(), "a fresh estimator is unfitted")
    assert_throws(RuntimeError, || ridge.predict(x))
    ridge.fit(x, y)
    assert_true(ridge.is_fitted(), "fit sets it")
    assert_eq(ridge.feature_count(), 2)

    #: A clone carries the parameters and none of the fit.
    let fresh = ridge.clone_unfitted()
    defer { fresh.free() }
    assert_true(not fresh.is_fitted(), "a clone is unfitted")
    assert_near(fresh.alpha, 0.5, 1e-9, "a clone carries alpha")

    #: Parameters round-trip.
    let params = ridge.get_params()
    let other = Ridge(9.0, false)
    defer { other.free() }
    other.set_params(params)
    assert_near(other.alpha, 0.5, 1e-9, "set_params reads alpha")
    assert_true(other.fit_intercept, "set_params reads fit_intercept")

    #: A width the estimator never saw is refused rather than broadcast.
    let narrow = matrix([[1.0], [2.0]])
    defer { narrow.free() }
    assert_throws(ValueError, || ridge.predict(narrow))
    assert_throws(ValueError, || Ridge(-1.0))
}

#: Ridge against the normal equations solved on the host in double
#: precision, at a width the host solve takes and one past it, with features
#: whose means sit far from their spread -- which is what centring on the
#: device, before the product, has to survive.
fn check_ridge_against_host(rows: int, width: int, offset: float) -> void {
    var values: list[float] = []
    var target: list[float] = []
    for i in 0..rows {
        var total = 0.25
        for j in 0..width {
            let wave = math.sin(0.917 * float(i * width + j) + 0.3)
            values.push(offset + wave)
            total += wave * math.cos(float(j))
        }
        target.push(total + 0.1 * math.sin(3.3 * float(i)))
    }
    let x = Tensor([rows, width], values)
    defer { x.free() }
    let y = Tensor([rows], target)
    defer { y.free() }
    let model = Ridge(2.0)
    defer { model.free() }
    model.fit(x, y)

    var centres = [0.0 for _j in 0..width]
    var middle = 0.0
    for i in 0..rows {
        for j in 0..width { centres[j] += values[i * width + j] / float(rows) }
        middle += target[i] / float(rows)
    }
    var system = [0.0 for _s in 0..width * width]
    var rhs = [0.0 for _s in 0..width]
    for i in 0..rows {
        for a in 0..width {
            let left = values[i * width + a] - centres[a]
            rhs[a] += left * (target[i] - middle)
            for b in 0..width {
                system[a * width + b] += left * (values[i * width + b] - centres[b])
            }
        }
    }
    for a in 0..width { system[a * width + a] += 2.0 }
    let want = jl.host_cholesky_solve(system, rhs, width)
    let coef = model.coef_
    if coef is not null {
        let got = coef.download()
        for j in 0..width { assert_near(got[j], want[j], 2e-3, f"{rows}x{width} coef {j}") }
    }
    var intercept = middle
    for j in 0..width { intercept -= centres[j] * want[j] }
    assert_near(
        model.intercept_,
        intercept,
        2e-3 * max(1.0, math.fabs(intercept)),
        f"{width} intercept"
    )
}

fn test_ridge_matches_the_host_normal_equations() -> void {
    if not have_gpu() { return }
    check_ridge_against_host(500, 7, 0.0)
    check_ridge_against_host(500, 7, 300.0)
    #: Few rows, so the product stays under the GEMM tuner's threshold.
    check_ridge_against_host(60, 259, 0.0)
}

#: Two hundred rows of five features and labels -1 and 2, built by a formula
#: so scikit-learn could be run on exactly the same numbers.
fn logistic_data() -> tuple[Tensor, Tensor] {
    let rows = 200
    let width = 5
    let truth = [1.5, -2.0, 0.7, 0.0, 1.1]
    var values: list[float] = []
    var labels: list[float] = []
    for i in 0..rows {
        var score = 0.8 * math.sin(5.1 * float(i)) + 0.3
        for j in 0..width {
            let wave = math.sin(1.3 * float(i) + 0.7 * float(j))
            let value = wave + 0.5 * math.cos(0.37 * float(i) * float(j))
            values.push(value)
            score += value * truth[j]
        }
        labels.push(if score > 0.0 { 2.0 } else { -1.0 })
    }
    return (Tensor([rows, width], values), Tensor([rows], labels))
}

#: Against scikit-learn's `LogisticRegression(C, fit_intercept, tol=1e-12)`,
#: which converged to the optimum these coefficients are, at three settings.
fn test_logistic_regression_matches_sklearn() -> void {
    if not have_gpu() { return }
    let (x, y) = logistic_data()
    defer { x.free() }
    defer { y.free() }
    let cases: list[tuple[float, bool, list[float], float]] = [
        (
            1.0,
            true,
            [
                0.5267232,
                -2.6057907,
                0.5334995,
                0.3912605,
                1.3343498,
            ],
            2.3092815
        ),
        (
            0.1,
            true,
            [
                -0.3577779,
                -0.9729115,
                -0.0144722,
                0.2627082,
                0.6384113,
            ],
            2.0821466
        ),
        (
            1.0,
            false,
            [
                2.8996225,
                -2.842107,
                0.2829699,
                0.6658263,
                2.0753875,
            ],
            0.0
        ),
    ]
    for (c, intercept, coef, bias) in cases {
        let model = LogisticRegression(C: c, fit_intercept: intercept)
        model.fit(x, y)
        let fitted = model.coef_
        assert_true(fitted is not null, "fit records coefficients")
        if fitted is not null {
            assert_eq(fitted.shape, [1, 5])
            let got = fitted.download()
            for (index, want) in coef.enumerate() {
                assert_near(
                    got[index],
                    want,
                    1e-3 * max(1.0, math.fabs(want)),
                    f"C {c} coef {index}"
                )
            }
        }
        assert_near(model.intercept_, bias, 1e-3 * max(1.0, math.fabs(bias)), f"C {c} intercept")
        assert_true(model.converged_, f"C {c} converged")
        assert_true(model.n_iter_ < 30, f"C {c} took {model.n_iter_} Newton steps")
        model.free()
    }

    let model = LogisticRegression()
    defer { model.free() }
    model.fit(x, y)
    assert_eq(model.kind(), EstimatorKind.Classifier)
    assert_close(model.classes_, [-1.0, 2.0], "classes_")
    let head = x.view(0, [6, 5])
    defer { head.free() }
    let predicted = model.predict(head)
    defer { predicted.free() }
    assert_close(predicted.download(), [2.0, -1.0, -1.0, 2.0, 2.0, 2.0], "predict")
    let proba = model.predict_proba(head)
    defer { proba.free() }
    assert_eq(proba.shape, [6, 2])
    let shares = proba.download()
    let want = [0.1159787, 0.8840213, 0.7546127, 0.2453873, 0.5703635, 0.4296365]
    for index in 0..6 { assert_near(shares[index], want[index], 1e-4, f"predict_proba {index}") }
    let scores = model.decision_function(head)
    defer { scores.free() }
    assert_near(scores.download()[0], math.log(0.8840213 / 0.1159787), 1e-3, "decision_function")
}

#: Three classes from the same rows: scikit-learn's multinomial fit
#: (`tol=1e-12`), at two settings, coefficients and intercepts.
fn test_multinomial_logistic_regression_matches_sklearn() -> void {
    if not have_gpu() { return }
    let (x, _two) = logistic_data()
    defer { x.free() }
    defer { _two.free() }
    let weights = [
        [1.0, -0.5, 0.3, 0.0, 0.8],
        [-0.7, 1.2, 0.0, 0.4, -0.3],
        [0.2, 0.1, -1.1, 0.9, 0.0],
    ]
    let values = x.download()
    var labels: list[float] = []
    for i in 0..200 {
        var best = 0
        var top = -1.0e30
        for k in 0..3 {
            var score = 0.6 * math.sin(2.7 * float(i) + 1.9 * float(k))
            for j in 0..5 { score += values[i * 5 + j] * weights[k][j] }
            if score > top {
                top = score
                best = k
            }
        }
        labels.push([5.0, 7.0, 9.0][best])
    }
    let y = Tensor([200], labels)
    defer { y.free() }
    let cases: list[tuple[float, bool, list[float], list[float]]] = [
        (
            1.0,
            true,
            [
                0.921495,
                -1.1577,
                0.745035,
                -0.690657,
                0.773432,
                -0.912808,
                1.500597,
                0.829982,
                -0.074597,
                -0.483631,
                -0.008687,
                -0.342898,
                -1.575017,
                0.765254,
                -0.289802,
            ],
            [
                0.46118,
                -0.646521,
                0.185341,
            ]
        ),
        (
            0.3,
            false,
            [
                1.050079,
                -0.897648,
                0.564756,
                -0.404177,
                0.72196,
                -1.047837,
                1.129401,
                0.582024,
                -0.140759,
                -0.505767,
                -0.002242,
                -0.231753,
                -1.14678,
                0.544936,
                -0.216193,
            ],
            [0.0, 0.0, 0.0]
        ),
    ]
    for (c, intercept, coef, bias) in cases {
        let model = LogisticRegression(C: c, fit_intercept: intercept, tol: 1.0e-7, max_iter: 500)
        model.fit(x, y)
        assert_true(model.converged_, f"C {c} converged in {model.n_iter_}")
        let fitted = model.coef_
        if fitted is not null {
            assert_eq(fitted.shape, [3, 5])
            let got = fitted.download()
            for (index, want) in coef.enumerate() {
                assert_near(
                    got[index],
                    want,
                    1e-3 * max(1.0, math.fabs(want)),
                    f"C {c} coef {index}"
                )
            }
        }
        for (index, want) in bias.enumerate() {
            assert_near(model.intercepts_[index], want, 1e-3, f"C {c} intercept {index}")
        }
        model.free()
    }

    let model = LogisticRegression(tol: 1.0e-7, max_iter: 500)
    defer { model.free() }
    model.fit(x, y)
    assert_close(model.classes_, [5.0, 7.0, 9.0], "classes_")
    let head = x.view(0, [6, 5])
    defer { head.free() }
    let predicted = model.predict(head)
    defer { predicted.free() }
    assert_close(predicted.download(), [7.0, 7.0, 9.0, 9.0, 5.0, 5.0], "multinomial predict")
    let proba = model.predict_proba(head)
    defer { proba.free() }
    assert_eq(proba.shape, [6, 3])
    let shares = proba.download()
    let want = [0.273362, 0.69505, 0.031588, 0.276119, 0.667641, 0.05624]
    for index in 0..6 {
        assert_near(shares[index], want[index], 1e-4, f"multinomial proba {index}")
    }
    let scores = model.decision_function(head)
    defer { scores.free() }
    assert_eq(scores.shape, [6, 3])
}

fn test_logistic_regression_keeps_the_contract() -> void {
    if not have_gpu() { return }
    let (x, y) = logistic_data()
    defer { x.free() }
    defer { y.free() }
    let model = LogisticRegression(C: 0.5, max_iter: 50)
    defer { model.free() }
    assert_throws(RuntimeError, || model.predict(x))
    model.fit(x, y)
    assert_eq(model.feature_count(), 5)
    let other = LogisticRegression()
    defer { other.free() }
    other.set_params(model.get_params())
    assert_near(other.C, 0.5, 1e-12, "set_params reads C")
    assert_eq(other.max_iter, 50)
    let fresh = model.clone_unfitted()
    defer { fresh.free() }
    assert_true(not fresh.is_fitted(), "a clone is unfitted")
    assert_throws(ValueError, || LogisticRegression(C: 0.0))

    let one = Tensor([200], [4.0 for _i in 0..200])
    defer { one.free() }
    assert_throws(ValueError, || model.fit(x, one))

    #: One step cannot converge, and says so.
    let short = LogisticRegression(max_iter: 1)
    defer { short.free() }
    short.fit(x, y)
    assert_true(not short.converged_, "one Newton step is not convergence")

    let piped = make_pipeline([StandardScaler(), LogisticRegression()])
    defer { piped.free() }
    piped.fit(x, y)
    assert_true(piped.score(x, y) > 0.8, "a scaled logistic regression fits its own data")
}

#: The package's own estimators have to go into its own pipeline. Both of these
#: once lacked the `Estimator` conformance, so `make_pipeline` refused them
#: with a TypeError, while the suite stayed green because only test-local
#: estimators had ever been piped.
fn test_a_pipeline_takes_the_real_estimators() -> void {
    if not have_gpu() { return }
    let x = sample()
    defer { x.free() }
    let y = target()
    defer { y.free() }

    let scaler = StandardScaler()
    defer { scaler.free() }
    let ridge = Ridge(1.0)
    defer { ridge.free() }
    let scaled = scaler.fit_transform(x)
    defer { scaled.free() }
    ridge.fit(scaled, y)
    let alone = ridge.predict(scaled)
    defer { alone.free() }

    let piped = make_pipeline([StandardScaler(), Ridge(1.0)])
    defer { piped.free() }
    piped.fit(x, y)
    assert_eq(piped.kind(), EstimatorKind.Regressor)
    let together = piped.predict(x)
    defer { together.free() }
    assert_close(together.download(), alone.download(), "piped predictions")
    assert_near(piped.score(x, y), ridge.score(scaled, y), 1e-5, "piped score")

    #: Named steps, and a parameter routed down to one of them.
    let named = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(5.0))])
    defer { named.free() }
    let routed_alpha: Params = {"ridge__alpha": 1.0}
    named.set_params(routed_alpha)
    named.fit(x, y)
    let routed = named.predict(x)
    defer { routed.free() }
    assert_close(routed.download(), alone.download(), "routed alpha")

    #: And as plain trait values, which is what a search holds.
    let generic: Estimator = StandardScaler(false, true)
    defer { generic.free() }
    let copy = generic.clone_unfitted()
    defer { copy.free() }
    assert_true(not copy.is_fitted(), "a cloned scaler is unfitted")
}
