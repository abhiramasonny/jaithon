#: The package as its README spells it: `import jailearn as jl` and nothing
#: else, so `Estimator` is NOT a name in this module.
#:
#: That matters for the trait checks on a call like `jl.make_pipeline([...])`:
#: the list literal is checked against `list[Estimator]` here, in a module
#: that never imported the trait. Every other test file imports `Estimator`,
#: which is how `StandardScaler` and `Ridge` were able to reach a pipeline in
#: the tests while lacking the conformance, and how the fix for that looked
#: complete while the README's own spelling still threw "expected 'Estimator'
#: but got 'StandardScaler'". Keep this file free of `from jailearn import`.
import jailearn as jl
from jaitensor import Tensor
from std.test import assert_eq, assert_near, assert_true

fn data() -> tuple[Tensor, Tensor] {
    var rows: list[list[float]] = []
    var labels: list[float] = []
    for i in 0..60 {
        let a = float(i % 7) * 1.5 - 4.0
        let b = float(i % 11) * 0.25 + 100.0
        rows.push([a, b, a * 0.5 - float(i % 3)])
        labels.push(if a + 0.1 * (b - 101.0) > 0.0 { 1.0 } else { 0.0 })
    }
    return (jl.matrix(rows), jl.vector(labels))
}

fn assert_same(a: list[float], b: list[float], what: str) -> void {
    assert_eq(a.len(), b.len(), f"{what} length")
    for index in 0..a.len() { assert_near(a[index], b[index], 1e-4, f"{what} at {index}") }
}

fn test_make_pipeline_takes_every_public_estimator_through_the_alias() -> void {
    if not jl.is_available() { return }
    let (x, y) = data()
    defer { x.free() }
    defer { y.free() }

    let scaler = jl.StandardScaler()
    defer { scaler.free() }
    let scaled = scaler.fit_transform(x)
    defer { scaled.free() }

    let ridge = jl.Ridge(alpha: 1.0)
    defer { ridge.free() }
    ridge.fit(scaled, y)
    let alone = ridge.predict(scaled)
    defer { alone.free() }
    let piped = jl.make_pipeline([jl.StandardScaler(), jl.Ridge(alpha: 1.0)])
    defer { piped.free() }
    piped.fit(x, y)
    let together = piped.predict(x)
    defer { together.free() }
    assert_same(together.download(), alone.download(), "scaler then ridge")

    let logistic = jl.LogisticRegression()
    defer { logistic.free() }
    logistic.fit(scaled, y)
    let odds = logistic.predict_proba(scaled)
    defer { odds.free() }
    let classify = jl.make_pipeline([jl.StandardScaler(), jl.LogisticRegression()])
    defer { classify.free() }
    classify.fit(x, y)
    let piped_odds = classify.predict_proba(x)
    defer { piped_odds.free() }
    assert_same(piped_odds.download(), odds.download(), "scaler then logistic regression")

    let neighbours = jl.make_pipeline([jl.StandardScaler(), jl.KNeighborsClassifier(3)])
    defer { neighbours.free() }
    neighbours.fit(x, y)
    let votes = neighbours.predict(x)
    defer { votes.free() }
    assert_eq(votes.shape, [60], "scaler then neighbours")

    let regress = jl.make_pipeline([jl.StandardScaler(), jl.KNeighborsRegressor(2)])
    defer { regress.free() }
    regress.fit(x, y)
    let means = regress.predict(x)
    defer { means.free() }
    assert_eq(means.shape, [60], "scaler then neighbour regression")

    let twice = jl.make_pipeline([jl.StandardScaler(), jl.StandardScaler()])
    defer { twice.free() }
    let again = twice.fit_transform(x)
    defer { again.free() }
    assert_same(again.download(), scaled.download(), "a scaler twice is a scaler once")
    assert_true(twice.is_fitted(), "the pipeline is fitted")
}
