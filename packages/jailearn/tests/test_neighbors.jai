#: `KNeighborsClassifier` and `KNeighborsRegressor` against scikit-learn
#: 1.9.0 run on the same points, and the neighbour search against a host
#: brute force.
import jailearn as jl
import std.math
from jailearn import Estimator, EstimatorKind, KNeighborsClassifier, KNeighborsRegressor
from jailearn import make_pipeline, StandardScaler
from jaitensor import Tensor
from std.test import assert_eq, assert_near, assert_throws, assert_true

fn assert_values(actual: list[float], expected: list[float], what: str, tol: float = 1e-5) -> void {
    assert_eq(actual.len(), expected.len(), f"{what} length")
    for (index, want) in expected.enumerate() {
        assert_near(actual[index], want, tol, f"{what} at {index}")
    }
}

fn training() -> Tensor {
    return jl.matrix(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [5.0, 5.0],
            [6.0, 5.0],
            [5.0, 6.0],
            [6.0, 6.0],
            [0.0, 5.0],
            [1.0, 5.0],
            [0.0, 6.0],
            [1.0, 6.5],
        ]
    )
}

fn classes() -> Tensor {
    return jl.vector([3.0, 3.0, 3.0, 7.0, 7.0, 7.0, 7.0, 7.0, 9.0, 9.0, 9.0, 3.0])
}

fn targets() -> Tensor { return jl.vector([0.5 * float(i) + 0.3 for i in 0..12]) }

#: The fourth query is equidistant from three training rows and the fifth
#: sits exactly on one, which is what the distance weighting special-cases.
fn queries() -> Tensor {
    return jl.matrix([[0.2, 0.1], [5.5, 5.4], [0.4, 5.8], [3.0, 3.0], [1.0, 1.0]])
}

fn test_classifier_matches_sklearn() -> void {
    if not jl.is_available() { return }
    let x = training()
    defer { x.free() }
    let y = classes()
    defer { y.free() }
    let q = queries()
    defer { q.free() }

    let uniform = KNeighborsClassifier(3)
    defer { uniform.free() }
    uniform.fit(x, y)
    assert_eq(uniform.kind(), EstimatorKind.Classifier)
    assert_values(uniform.classes_, [3.0, 7.0, 9.0], "classes_")
    let predicted = uniform.predict(q)
    defer { predicted.free() }
    assert_values(jl.to_values(predicted), [3.0, 7.0, 9.0, 7.0, 3.0], "uniform predict")
    let proba = uniform.predict_proba(q)
    defer { proba.free() }
    assert_eq(proba.shape, [5, 3])
    assert_values(
        jl.to_values(proba),
        [
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.3333333,
            0.0,
            0.6666667,
            0.0,
            0.6666667,
            0.3333333,
            0.6666667,
            0.3333333,
            0.0,
        ],
        "uniform predict_proba"
    )

    let weighted = KNeighborsClassifier(3, weights: "distance")
    defer { weighted.free() }
    weighted.fit(x, y)
    let picked = weighted.predict(q)
    defer { picked.free() }
    #: The last query coincides with a class-7 row, so only that row votes.
    assert_values(jl.to_values(picked), [3.0, 7.0, 9.0, 7.0, 7.0], "distance predict")
    let shares = weighted.predict_proba(q)
    defer { shares.free() }
    assert_values(
        jl.to_values(shares),
        [
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.2443596,
            0.0,
            0.7556404,
            0.0,
            0.6666667,
            0.3333333,
            0.0,
            1.0,
            0.0,
        ],
        "distance predict_proba"
    )

    #: scikit-learn's 0.8333: the last row's two nearest are class 9.
    assert_near(uniform.score(x, y), 10.0 / 12.0, 1e-6, "training accuracy")
}

fn test_regressor_matches_sklearn() -> void {
    if not jl.is_available() { return }
    let x = training()
    defer { x.free() }
    let t = targets()
    defer { t.free() }
    let q = queries()
    defer { q.free() }

    let uniform = KNeighborsRegressor(3)
    defer { uniform.free() }
    uniform.fit(x, t)
    assert_eq(uniform.kind(), EstimatorKind.Regressor)
    let mean = uniform.predict(q)
    defer { mean.free() }
    assert_values(jl.to_values(mean), [0.8, 2.8, 5.1333333, 2.9666667, 1.3], "uniform predict")

    let weighted = KNeighborsRegressor(3, weights: "distance")
    defer { weighted.free() }
    weighted.fit(x, t)
    let blend = weighted.predict(q)
    defer { blend.free() }
    assert_values(
        jl.to_values(blend),
        [0.5508154, 2.7680542, 5.1702997, 2.9666667, 1.8],
        "distance predict",
        tol: 2e-5
    )
}

#: Distances are euclidean, nearest first, and equal distances come out by
#: the smaller training index -- which is a choice scikit-learn does not make
#: (it returned 9, 3, 4 for the fourth query), so only the sets are its.
fn test_kneighbors_orders_ties_by_index() -> void {
    if not jl.is_available() { return }
    let x = training()
    defer { x.free() }
    let y = classes()
    defer { y.free() }
    let q = queries()
    defer { q.free() }
    let model = KNeighborsClassifier(3)
    defer { model.free() }
    model.fit(x, y)
    let (distances, index) = model.kneighbors(q)
    defer { distances.free() }
    defer { index.free() }
    assert_values(
        jl.to_values(distances),
        [
            0.2236068,
            0.8062258,
            0.9219544,
            0.6403124,
            0.6403124,
            0.781025,
            0.4472136,
            0.8944272,
            0.9219544,
            2.8284271,
            2.8284271,
            2.8284271,
            0.0,
            1.0,
            1.0,
        ],
        "kneighbors distances",
        tol: 1e-4
    )
    let at = jl.to_values(index)
    assert_values(at[0:3], [0.0, 1.0, 2.0], "first query")
    assert_values(at[6:9], [10.0, 8.0, 11.0], "third query")
    assert_values(at[9:12], [3.0, 4.0, 9.0], "three-way tie, smallest index first")
    assert_values(at[12:15], [3.0, 1.0, 2.0], "an exact match, then a tie")
    let pair = [at[3], at[4]]
    assert_true(pair.contains(4.0) and pair.contains(5.0), "the near-tie holds rows 4 and 5")

    let (wide, wide_at) = model.kneighbors(q, n_neighbors: 12)
    defer { wide.free() }
    defer { wide_at.free() }
    assert_eq(wide_at.shape, [5, 12])
}

fn test_the_neighbour_estimators_keep_the_contract() -> void {
    if not jl.is_available() { return }
    let x = training()
    defer { x.free() }
    let y = classes()
    defer { y.free() }

    let model = KNeighborsClassifier(4, "distance")
    defer { model.free() }
    assert_true(not model.is_fitted(), "fresh is unfitted")
    assert_throws(RuntimeError, || model.predict(x))
    model.fit(x, y)
    assert_true(model.is_fitted(), "fit sets it")
    assert_eq(model.feature_count(), 2)
    let params = model.get_params()
    let other = KNeighborsClassifier()
    defer { other.free() }
    other.set_params(params)
    assert_eq(other.n_neighbors, 4)
    assert_eq(other.weights, "distance")
    let fresh = model.clone_unfitted()
    defer { fresh.free() }
    assert_true(not fresh.is_fitted(), "a clone is unfitted")

    let narrow = jl.matrix([[1.0], [2.0]])
    defer { narrow.free() }
    assert_throws(ValueError, || model.predict(narrow))
    assert_throws(ValueError, || KNeighborsClassifier(3, "nearest"))
    let too_many = KNeighborsRegressor(13)
    defer { too_many.free() }
    assert_throws(ValueError, || too_many.fit(x, y))

    #: Fit copies the training rows, so the caller may free theirs.
    let own = training()
    let kept = KNeighborsRegressor(2)
    defer { kept.free() }
    kept.fit(own, y)
    own.free()
    let q = queries()
    defer { q.free() }
    let after = kept.predict(q)
    defer { after.free() }
    assert_near(jl.to_values(after)[0], 3.0, 1e-5, "predict after the caller freed X")

    #: And into a pipeline, as a trait value.
    let piped = make_pipeline([StandardScaler(), KNeighborsClassifier(3)])
    defer { piped.free() }
    piped.fit(x, y)
    let labels = piped.predict(x)
    defer { labels.free() }
    assert_eq(labels.shape, [12])
}

#: The one-pass search against a double-precision brute force on the host,
#: at a size that spans several threads per row and with more queries than a
#: band holds when the band is small.
fn test_nearest_rows_matches_a_host_brute_force() -> void {
    if not jl.is_available() { return }
    let points = 1500
    let count = 64
    let width = 6
    var train: list[float] = []
    for index in 0..points * width { train.push(math.sin(float(index) * 1.618 + 0.3) * 3.0) }
    var query: list[float] = []
    for index in 0..count * width { query.push(math.cos(float(index) * 2.413 + 0.7) * 3.0) }
    let b = Tensor([points, width], train)
    defer { b.free() }
    let a = Tensor([count, width], query)
    defer { a.free() }
    #: The second case cuts the queries into bands of eleven, the last short.
    for (k, band) in [(1, 0), (7, 11 * points), (40, 11 * points), (5, 0)] {
        let (squared, index) = if band > 0 {
            jl.nearest_rows(a, b, k, band_floats: band)
        } else {
            jl.nearest_rows(a, b, k)
        }
        let got = jl.to_values(index)
        let got_distance = jl.to_values(squared)
        squared.free()
        index.free()
        var agree = 0
        for row in 0..count {
            var order = [column for column in 0..points]
            var exact: list[float] = []
            for column in 0..points {
                var total = 0.0
                for f in 0..width {
                    let gap = query[row * width + f] - train[column * width + f]
                    total += gap * gap
                }
                exact.push(total)
            }
            order.sort(key: |c| exact[c])
            for slot in 0..k {
                let picked = int(got[row * k + slot])
                if picked == order[slot] { agree += 1 }
                assert_near(got_distance[row * k + slot], exact[order[slot]], 1e-3, "distance")
                #: A disagreement must be a near-tie: this data is
                #: quasi-periodic and holds pairs a millionth apart, which
                #: float32 (numpy's included) cannot order.
                assert_near(exact[picked], exact[order[slot]], 1e-4, f"k {k} row {row} slot {slot}")
            }
        }
        assert_true(agree * 100 >= count * k * 95, f"k {k}: {agree} of {count * k} agree")
    }
}

fn uniform_cell(index: int) -> float {
    let s = math.sin(float(index) * 12.9898 + 78.233) * 43758.5453
    return s - math.floor(s)
}

#: How many of each query's `k` neighbours the estimator found, against a
#: double-precision brute force over the same float32 values. A miss must be
#: a near-tie (closer to the right distance than float32 resolves); returns
#: the share of exact matches.
fn neighbour_agreement(
    points: int,
    width: int,
    count: int,
    k: int,
    offset: float,
    spread: float
) -> float {
    var train: list[float] = []
    for index in 0..points * width { train.push(offset + spread * uniform_cell(index)) }
    var query: list[float] = []
    for index in 0..count * width { query.push(offset + spread * uniform_cell(index + 7777777)) }
    let b = Tensor([points, width], train)
    defer { b.free() }
    let a = Tensor([count, width], query)
    defer { a.free() }
    let rows = jl.to_values(b)
    let asked = jl.to_values(a)
    let y = jl.vector([float(i % 3) for i in 0..points])
    defer { y.free() }
    let model = KNeighborsClassifier(k)
    defer { model.free() }
    model.fit(b, y)
    let (distance, index) = model.kneighbors(a)
    let got = jl.to_values(index)
    let got_distance = jl.to_values(distance)
    distance.free()
    index.free()
    var agree = 0
    for row in 0..count {
        var exact: list[float] = []
        for column in 0..points {
            var total = 0.0
            for f in 0..width {
                let gap = asked[row * width + f] - rows[column * width + f]
                total += gap * gap
            }
            exact.push(total)
        }
        var order = [column for column in 0..points]
        order.sort(key: |c| exact[c])
        for slot in 0..k {
            let picked = int(got[row * k + slot])
            if picked == order[slot] { agree += 1 }
            let want = math.sqrt(exact[order[slot]])
            let scale = spread * spread * float(width)
            assert_near(exact[picked], exact[order[slot]], 1e-5 * scale, f"row {row} slot {slot}")
            assert_near(
                got_distance[row * k + slot],
                want,
                1e-4 * spread * math.sqrt(float(width)),
                f"distance {row} {slot}"
            )
        }
    }
    return float(agree) / float(count * k)
}

#: Far from the origin, or in one or two dimensions, the expanded distance
#: `|a|^2 + |b|^2 - 2 a.b` rounds at the scale of the squared norms rather
#: than of the distances: at fifty features a thousand from the origin it
#: found the right nearest row 17% of the time, and two features a hundred
#: out it found 2%. The search centres both sides on the training means and
#: compares narrow rows term by term.
fn test_neighbours_survive_an_offset_and_a_narrow_row() -> void {
    if not jl.is_available() { return }
    let far = neighbour_agreement(2000, 50, 40, 5, 1000.0, 1.0)
    assert_true(far >= 0.99, f"fifty features, offset 1000: {far} of neighbours right")
    let narrow = neighbour_agreement(20000, 2, 40, 5, 100.0, 1.0)
    assert_true(narrow >= 0.99, f"two features, offset 100: {narrow} of neighbours right")
    let line = neighbour_agreement(5000, 1, 60, 1, 0.0, 6.0)
    assert_true(line >= 0.99, f"one zero-mean feature: {line} of nearest rows right")
}

#: An exact duplicate is at distance zero, so distance weighting takes only
#: it -- which the expanded form missed by up to five on integer features in
#: the hundreds, so the rule never fired.
fn test_a_duplicate_row_is_at_distance_zero() -> void {
    if not jl.is_available() { return }
    let points = 500
    let width = 50
    var train: list[float] = []
    for index in 0..points * width { train.push(math.floor(uniform_cell(index) * 1000.0)) }
    let x = Tensor([points, width], train)
    defer { x.free() }
    let y = jl.vector([float(i) * 0.25 for i in 0..points])
    defer { y.free() }
    let q = Tensor([3, width], train[17 * width:20 * width])
    defer { q.free() }
    let model = KNeighborsRegressor(4, "distance")
    defer { model.free() }
    model.fit(x, y)
    let (distance, index) = model.kneighbors(q)
    let near = jl.to_values(distance)
    let at = jl.to_values(index)
    distance.free()
    index.free()
    for row in 0..3 {
        assert_eq(near[row * 4], 0.0, f"query {row} sits on a training row")
        assert_eq(at[row * 4], float(17 + row), f"query {row} found its own row")
    }
    let predicted = model.predict(q)
    defer { predicted.free() }
    assert_values(jl.to_values(predicted), [4.25, 4.5, 4.75], "the duplicate takes all the weight")
}

#: A query holding a NaN or an infinity has no neighbours: padded indices, a
#: NaN share, a NaN label and a NaN mean, and never a training row read
#: through a padded index.
fn test_a_query_that_is_not_finite_has_no_neighbours() -> void {
    if not jl.is_available() { return }
    for width in [2, 12] {
        var train: list[float] = []
        for index in 0..40 * width { train.push(uniform_cell(index)) }
        let x = Tensor([40, width], train)
        defer { x.free() }
        let y = jl.vector([float(i % 2) + 3.0 for i in 0..40])
        defer { y.free() }
        var asked = [0.5 for _slot in 0..3 * width]
        asked[width] = math.NAN
        asked[2 * width + 1] = math.INF
        let q = Tensor([3, width], asked)
        defer { q.free() }
        let classifier = KNeighborsClassifier(3)
        defer { classifier.free() }
        classifier.fit(x, y)
        let (distance, index) = classifier.kneighbors(q)
        let at = jl.to_values(index)
        distance.free()
        index.free()
        assert_true(at[0] < 40.0, f"width {width}: a finite query has neighbours")
        for slot in 3..9 { assert_eq(at[slot], 40.0, f"width {width}: slot {slot} is padded") }
        let proba = classifier.predict_proba(q)
        let shares = jl.to_values(proba)
        proba.free()
        assert_near(shares[0] + shares[1], 1.0, 1e-6, f"width {width}: finite shares")
        for slot in 2..6 { assert_true(math.is_nan(shares[slot]), f"width {width}: share {slot}") }
        let labels = classifier.predict(q)
        let named = jl.to_values(labels)
        labels.free()
        assert_true(named[0] == 3.0 or named[0] == 4.0, f"width {width}: a finite label")
        assert_true(math.is_nan(named[1]) and math.is_nan(named[2]), f"width {width}: NaN labels")
        let regressor = KNeighborsRegressor(3)
        defer { regressor.free() }
        regressor.fit(x, y)
        let mean = regressor.predict(q)
        let means = jl.to_values(mean)
        mean.free()
        assert_true(not math.is_nan(means[0]), f"width {width}: a finite mean")
        assert_true(math.is_nan(means[1]) and math.is_nan(means[2]), f"width {width}: NaN means")
    }
}
