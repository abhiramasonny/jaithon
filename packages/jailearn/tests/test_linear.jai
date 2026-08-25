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
from std.test import assert_eq, assert_near, assert_throws, assert_true
from jaitensor import Tensor
from jailearn import EstimatorKind, Ridge, StandardScaler, linear_regression
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
