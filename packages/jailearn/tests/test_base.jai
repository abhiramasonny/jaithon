#: The estimator protocol and the shared GPU kit every other jailearn module
#: is written against.
import jailearn as jl
import jaitensor as jt
import std.math
from jailearn import Estimator, EstimatorKind, LabelIndex, Params
from jaitensor import Tensor
from std.test import assert_eq, assert_near, assert_throws, assert_true

fn assert_values(
    actual: list[float],
    expected: list[float],
    where: str,
    tol: float = 1e-4
) -> void {
    assert_eq(actual.len(), expected.len(), f"{where} length")
    for (index, want) in expected.enumerate() {
        assert_near(actual[index], want, tol, f"{where} at {index}")
    }
}

#: Predicts the training mean, which makes its R^2 exactly zero and so pins
#: down `default_score` without a model that could be right by accident.
class MeanRegressor: Estimator {
    var value: float
    var fitted: bool
    var width: int

    pub fn init(self) {
        self.value = 0.0
        self.fitted = false
        self.width = 0
    }

    pub fn kind(self) -> EstimatorKind { return EstimatorKind.Regressor }

    pub fn fit(self, x: Tensor, y: Tensor? = null) -> void {
        let target = jl.require_target(y, "MeanRegressor")
        jl.check_pair(x, target)
        var total = 0.0
        for value in jl.to_values(target) { total += value }
        self.value = total / float(target.size)
        self.width = x.shape[1]
        self.fitted = true
    }

    pub fn get_params(self) -> Params { return jl.params_of([("width", self.width)]) }

    pub fn set_params(self, params: Params) -> void {
        self.width = jl.param_int(params, "width", self.width)
    }

    pub fn clone_unfitted(self) -> Estimator { return MeanRegressor() }

    pub fn is_fitted(self) -> bool { return self.fitted }

    pub fn free(self) -> void { self.fitted = false }

    pub fn feature_count(self) -> int { return self.width }

    pub fn predict(self, x: Tensor) -> Tensor {
        jl.check_fitted(self.fitted, "MeanRegressor")
        jl.check_width(x, self.width, "MeanRegressor")
        return jt.fill([x.shape[0]], self.value)
    }
}

#: Always answers with one class, so its accuracy is that class's share.
class OneClassClassifier: Estimator {
    let answer: float
    var fitted: bool

    pub fn init(self, answer: float) {
        self.answer = answer
        self.fitted = false
    }

    pub fn kind(self) -> EstimatorKind { return EstimatorKind.Classifier }

    pub fn fit(self, x: Tensor, y: Tensor? = null) -> void { self.fitted = true }

    pub fn get_params(self) -> Params { return jl.params_of([("answer", self.answer)]) }

    pub fn set_params(self, params: Params) -> void { }

    pub fn clone_unfitted(self) -> Estimator { return OneClassClassifier(self.answer) }

    pub fn is_fitted(self) -> bool { return self.fitted }

    pub fn free(self) -> void { self.fitted = false }

    pub fn predict(self, x: Tensor) -> Tensor { return jt.fill([x.shape[0]], self.answer) }
}

fn test_host_bridge_round_trips_matrices_and_vectors() -> void {
    if not jl.is_available() { return }
    assert_true(jl.device_name() != "", "a device that exists has a name")
    let x = jl.matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    defer { x.free() }
    assert_eq(x.shape, [2, 3])
    let rows = jl.to_rows(x)
    assert_values(rows[0], [1.0, 2.0, 3.0], "row 0")
    assert_values(rows[1], [4.0, 5.0, 6.0], "row 1")

    let y = jl.vector([7.0, 8.0])
    defer { y.free() }
    assert_values(jl.to_values(y), [7.0, 8.0], "vector")

    let second = jl.column(x, 1)
    defer { second.free() }
    assert_eq(second.shape, [2])
    assert_values(jl.to_values(second), [2.0, 5.0], "column 1")

    let picked = jl.select_columns(x, [2, 0])
    defer { picked.free() }
    assert_eq(picked.shape, [2, 2])
    assert_values(jl.to_values(picked), [3.0, 1.0, 6.0, 4.0], "select_columns")

    let as_column = second.view(0, [2, 1])
    defer { as_column.free() }
    let joined = jl.hstack([picked, as_column])
    defer { joined.free() }
    assert_eq(joined.shape, [2, 3])
    assert_values(jl.to_values(joined), [3.0, 1.0, 2.0, 6.0, 4.0, 5.0], "hstack")

    let order = jl.index_tensor([1, 0, 1])
    defer { order.free() }
    let taken = jl.take_rows(x, order)
    defer { taken.free() }
    assert_values(jl.to_values(taken), [4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "take_rows")

    assert_throws(ValueError, || jl.matrix([[1.0], [2.0, 3.0]]))
    assert_throws(IndexError, || jl.column(x, 9))
}

fn test_params_unpack_the_number_the_caller_actually_wrote() -> void {
    let params = jl.params_of([("depth", 5), ("gamma", 1), ("scaled", true), ("kind", "rbf")])
    assert_eq(jl.param_int(params, "depth", 0), 5)
    #: An integer where a float was meant is the mistake everybody makes once,
    #: and reading it back as the fallback would silently change a model.
    assert_near(jl.param_float(params, "gamma", 0.5), 1.0, 1e-9)
    assert_true(jl.param_bool(params, "scaled", false))
    assert_eq(jl.param_str(params, "kind", "linear"), "rbf")
    assert_eq(jl.param_int(params, "missing", 7), 7)
    assert_near(jl.param_float(params, "missing", 0.25), 0.25, 1e-9)
    assert_true(not jl.param_bool(params, "missing", false))
    assert_eq(jl.param_str(params, "missing", "linear"), "linear")
}

fn test_validation_names_the_shape_it_wanted() -> void {
    if not jl.is_available() { return }
    let x = jl.matrix([[1.0, 2.0], [3.0, 4.0]])
    defer { x.free() }
    let y = jl.vector([0.0, 1.0])
    defer { y.free() }
    jl.check_matrix(x)
    jl.check_pair(x, y)
    jl.check_width(x, 2, "estimator")
    #: A width of zero means the estimator records none, so the call goes on.
    jl.check_width(x, 0, "estimator")
    assert_throws(ValueError, || jl.check_matrix(y))
    assert_throws(ValueError, || jl.check_pair(x, x))
    assert_throws(ValueError, || jl.check_width(x, 5, "estimator"))
    assert_throws(RuntimeError, || jl.check_fitted(false, "estimator"))
    jl.check_fitted(true, "estimator")
    assert_true(jl.require_target(y, "estimator") is y)
    assert_throws(ValueError, || jl.require_target(null, "estimator"))
}

fn test_column_statistics_use_two_passes() -> void {
    if not jl.is_available() { return }
    let x = jl.matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    defer { x.free() }
    let (mean, variance) = jl.column_moments(x)
    defer { mean.free() }
    defer { variance.free() }
    assert_values(jl.to_values(mean), [2.0, 3.0], "mean")
    assert_values(jl.to_values(variance), [8.0 / 3.0, 8.0 / 3.0], "population variance")

    let (_unused, sample) = jl.column_moments(x, ddof: 1)
    defer { _unused.free() }
    defer { sample.free() }
    assert_values(jl.to_values(sample), [4.0, 4.0], "sample variance")

    let (low, high) = jl.column_min_max(x)
    defer { low.free() }
    defer { high.free() }
    assert_values(jl.to_values(low), [0.0, 1.0], "min")
    assert_values(jl.to_values(high), [4.0, 5.0], "max")

    #: The one-pass `E[x^2] - E[x]^2` this deliberately avoids returns a
    #: variance of about 0.03 here instead of 1.0, because the mean is seven
    #: digits clear of the spread.
    let shifted = jl.matrix([[1000000.0], [1000001.0], [1000002.0]])
    defer { shifted.free() }
    let (_centre, spread) = jl.column_moments(shifted, ddof: 1)
    defer { _centre.free() }
    defer { spread.free() }
    assert_near(jl.to_values(spread)[0], 1.0, 1e-3, "variance far from the origin")
}

fn test_masked_mean_and_nan_mask_make_an_imputer() -> void {
    if not jl.is_available() { return }
    let x = jl.matrix([[1.0, math.NAN], [3.0, 20.0]])
    defer { x.free() }
    let mask = jl.nan_mask(x)
    defer { mask.free() }
    assert_values(jl.to_values(mask), [0.0, 1.0, 0.0, 0.0], "nan_mask")
    let mean = jl.masked_column_mean(x, mask)
    defer { mean.free() }
    assert_values(jl.to_values(mean), [2.0, 20.0], "masked_column_mean")

    #: A column that is missing everywhere averages to zero, not to a NaN.
    let empty = jl.matrix([[math.NAN], [math.NAN]])
    defer { empty.free() }
    let all_missing = jl.nan_mask(empty)
    defer { all_missing.free() }
    let nothing = jl.masked_column_mean(empty, all_missing)
    defer { nothing.free() }
    assert_values(jl.to_values(nothing), [0.0], "an empty column")
}

fn test_sorting_and_quantiles_work_column_by_column() -> void {
    if not jl.is_available() { return }
    let x = jl.matrix([[3.0, 1.0], [1.0, 2.0], [2.0, 0.0]])
    defer { x.free() }
    let sorted = jl.sort_columns(x)
    defer { sorted.free() }
    assert_eq(sorted.shape, [3, 2])
    assert_values(jl.to_values(sorted), [1.0, 0.0, 2.0, 1.0, 3.0, 2.0], "sort_columns")

    let (values, origin) = jl.argsort_columns(x)
    defer { values.free() }
    defer { origin.free() }
    assert_values(jl.to_values(values), [1.0, 0.0, 2.0, 1.0, 3.0, 2.0], "argsort values")
    assert_values(jl.to_values(origin), [1.0, 2.0, 2.0, 0.0, 0.0, 1.0], "argsort origin")

    let quantiles = jl.column_quantiles(x, [0.0, 0.5, 1.0])
    defer { quantiles.free() }
    assert_eq(quantiles.shape, [3, 2])
    assert_values(jl.to_values(quantiles), [1.0, 0.0, 2.0, 1.0, 3.0, 2.0], "quantiles")

    #: A sample count that is not a power of two is the ordinary case, and the
    #: pad the bitonic network needs must not reach the answer.
    var ramp: list[list[float]] = []
    for index in 0..37 { ramp.push([float(36 - index)]) }
    let ragged = jl.matrix(ramp)
    defer { ragged.free() }
    let climbed = jl.sort_columns(ragged)
    defer { climbed.free() }
    let got = jl.to_values(climbed)
    for index in 0..37 { assert_near(got[index], float(index), 1e-6, f"ragged at {index}") }
    let median = jl.column_quantiles(ragged, [0.5])
    defer { median.free() }
    assert_near(jl.to_values(median)[0], 18.0, 1e-6, "ragged median")
}

fn test_distances_and_gram_kernels() -> void {
    if not jl.is_available() { return }
    let a = jl.matrix([[0.0, 0.0], [3.0, 4.0], [1.0, 0.0]])
    defer { a.free() }
    let b = jl.matrix([[0.0, 0.0], [1.0, 0.0]])
    defer { b.free() }
    let distances = jl.pairwise_sqdist(a, b)
    defer { distances.free() }
    assert_eq(distances.shape, [3, 2])
    assert_values(jl.to_values(distances), [0.0, 1.0, 25.0, 20.0, 1.0, 0.0], "pairwise_sqdist")

    let band = jl.pairwise_sqdist(a, b, first: 1, rows: 2)
    defer { band.free() }
    assert_eq(band.shape, [2, 2])
    assert_values(jl.to_values(band), [25.0, 20.0, 1.0, 0.0], "one band")

    let linear = jl.gram(a, b, jl.GRAM_LINEAR)
    defer { linear.free() }
    assert_values(jl.to_values(linear), [0.0, 0.0, 0.0, 3.0, 0.0, 1.0], "linear gram")

    let rbf = jl.gram(a, b, jl.GRAM_RBF, gamma: 1.0)
    defer { rbf.free() }
    assert_near(jl.to_values(rbf)[1], math.exp(-1.0), 1e-5, "rbf gram")

    let poly = jl.gram(a, b, jl.GRAM_POLY, gamma: 1.0, degree: 2.0, coef0: 1.0)
    defer { poly.free() }
    assert_near(jl.to_values(poly)[3], 16.0, 1e-4, "poly gram")

    let sigmoid = jl.gram(a, b, jl.GRAM_SIGMOID, gamma: 0.5, coef0: 0.25)
    defer { sigmoid.free() }
    assert_near(jl.to_values(sigmoid)[3], math.tanh(1.75), 1e-5, "sigmoid gram")

    assert_throws(ValueError, || jl.gram(a, b, 9))
}

fn test_row_reductions_break_ties_at_the_lowest_column() -> void {
    if not jl.is_available() { return }
    let m = jl.matrix([[1.0, 5.0, 3.0], [9.0, 2.0, 2.0]])
    defer { m.free() }
    let largest = jl.row_argmax(m)
    defer { largest.free() }
    assert_values(jl.to_values(largest), [1.0, 0.0], "row_argmax")
    let smallest = jl.row_argmin(m)
    defer { smallest.free() }
    assert_values(jl.to_values(smallest), [0.0, 1.0], "row_argmin, ties low")

    let (index, value) = jl.row_extreme(m, true)
    defer { index.free() }
    defer { value.free() }
    assert_values(jl.to_values(value), [5.0, 9.0], "row_extreme values")

    let (top, at) = jl.row_topk(m, 2)
    defer { top.free() }
    defer { at.free() }
    assert_values(jl.to_values(top), [5.0, 3.0, 9.0, 2.0], "row_topk values")
    assert_values(jl.to_values(at), [1.0, 2.0, 0.0, 1.0], "row_topk indices")

    #: Duplicates must be handed out one slot at a time rather than filling
    #: every slot with the same column, which is what a naive mask would do.
    let (nearest, where) = jl.row_topk(m, 2, largest: false)
    defer { nearest.free() }
    defer { where.free() }
    assert_values(jl.to_values(nearest), [1.0, 3.0, 2.0, 2.0], "smallest two")
    assert_values(jl.to_values(where), [0.0, 2.0, 1.0, 2.0], "smallest two indices")

    let totals = jl.log_sum_exp_rows(jl.matrix([[0.0, 0.0], [800.0, 800.0]]))
    defer { totals.free() }
    let logs = jl.to_values(totals)
    assert_near(logs[0], math.log(2.0), 1e-5, "log_sum_exp")
    #: Unshifted, `exp(800)` is an infinity and this row would come back as
    #: one; the shift by the row maximum is the whole point.
    assert_near(logs[1], 800.0 + math.log(2.0), 1e-2, "log_sum_exp stays finite")

    assert_throws(ValueError, || jl.row_topk(m, 4))
}

fn test_class_aggregation_gathers_the_sufficient_statistics() -> void {
    if not jl.is_available() { return }
    let labels = jl.vector([0.0, 1.0, 1.0, 2.0])
    defer { labels.free() }
    let counts = jl.bincount(labels, 3)
    defer { counts.free() }
    assert_values(jl.to_values(counts), [1.0, 2.0, 1.0], "bincount")

    let x = jl.matrix([[1.0], [2.0], [3.0], [4.0]])
    defer { x.free() }
    let sums = jl.class_sums(x, labels, 3)
    defer { sums.free() }
    assert_eq(sums.shape, [3, 1])
    assert_values(jl.to_values(sums), [1.0, 5.0, 4.0], "class_sums")
    let squares = jl.class_sumsq(x, labels, 3)
    defer { squares.free() }
    assert_values(jl.to_values(squares), [1.0, 13.0, 16.0], "class_sumsq")

    let source = jl.matrix([[1.0, 1.0], [2.0, 2.0]])
    defer { source.free() }
    let into = jl.index_tensor([0, 0])
    defer { into.free() }
    let scattered = jl.scatter_add_rows(source, into, 2)
    defer { scattered.free() }
    assert_values(jl.to_values(scattered), [3.0, 3.0, 0.0, 0.0], "scatter_add_rows")
}

fn test_affine_columns_and_count_equal() -> void {
    if not jl.is_available() { return }
    let x = jl.matrix([[1.0, 2.0], [3.0, 4.0]])
    defer { x.free() }
    let scale = jl.vector([2.0, 3.0])
    defer { scale.free() }
    let offset = jl.vector([1.0, -1.0])
    defer { offset.free() }
    let scaled = jl.affine_columns(x, scale, offset)
    defer { scaled.free() }
    assert_values(jl.to_values(scaled), [3.0, 5.0, 7.0, 11.0], "affine_columns")

    let left = jl.vector([1.0, 2.0, 3.0])
    defer { left.free() }
    let right = jl.vector([1.0, 9.0, 3.0])
    defer { right.free() }
    assert_eq(jl.count_equal(left, right), 2)
    assert_throws(ValueError, || jl.count_equal(left, scale))
}

fn test_kmeanspp_lands_one_centre_in_each_cluster() -> void {
    if not jl.is_available() { return }
    var rows: list[list[float]] = []
    for index in 0..90 {
        let group = index % 3
        rows.push([float(group) * 100.0 + float(index % 5) * 0.01, 0.0])
    }
    let x = jl.matrix(rows)
    defer { x.free() }
    #: Weighted by squared distance, three well-separated groups get one centre
    #: each essentially always; a uniform draw would put two in one group about
    #: a third of the time.
    let centres = jl.kmeanspp_seed(x, 3, seed: 20260823)
    defer { centres.free() }
    assert_eq(centres.shape, [3, 2])
    var seen: list[int] = []
    for row in jl.to_rows(centres) {
        let group = int(row[0] / 100.0 + 0.5)
        assert_true(not seen.contains(group), f"group {group} was seeded twice")
        seen.push(group)
    }
    assert_eq(seen.len(), 3, "one centre in each of the three groups")
    assert_throws(ValueError, || jl.kmeanspp_seed(x, 0))
}

fn test_random_draws_have_the_range_and_moments_asked_for() -> void {
    if not jl.is_available() { return }
    let rng = jl.rng_for(7)
    let first = rng.int_range(0, 100)
    let repeat = jl.rng_for(7).int_range(0, 100)
    assert_eq(first, repeat, "a seeded generator repeats")

    let uniform = jl.random_uniform([4096], lo: -2.0, hi: 5.0, seed: 11)
    defer { uniform.free() }
    for value in jl.to_values(uniform) {
        assert_true(value >= -2.0 and value < 5.0, f"{value} is outside [-2, 5)")
    }

    #: Seed 0 means unseeded everywhere in the package, so a named seed has to
    #: be the thing that makes a run repeat -- `Buffer.fill_uniform` on its own
    #: turns a 0 into a constant and repeats regardless.
    let repeated = jl.random_uniform([64], lo: -2.0, hi: 5.0, seed: 11)
    defer { repeated.free() }
    let prefix = uniform.view(0, [64])
    defer { prefix.free() }
    assert_eq(jl.count_equal(repeated, prefix), 64, "a seeded fill repeats")

    let normal = jl.random_normal([8192], seed: 13)
    defer { normal.free() }
    let (centre, spread) = jl.column_moments(normal.view(0, [8192, 1]))
    defer { centre.free() }
    defer { spread.free() }
    assert_near(jl.to_values(centre)[0], 0.0, 0.06, "standard normal mean")
    assert_near(jl.to_values(spread)[0], 1.0, 0.08, "standard normal variance")
}

fn test_label_index_maps_arbitrary_codes_both_ways() -> void {
    if not jl.is_available() { return }
    let y = jl.vector([5.0, 3.0, 5.0, 9.0, 3.0])
    defer { y.free() }
    assert_values(jl.distinct_values(y), [3.0, 5.0, 9.0], "distinct_values")

    let index = LabelIndex.from_labels(y)
    defer { index.free() }
    assert_eq(index.count(), 3)
    assert_values(index.classes, [3.0, 5.0, 9.0], "classes")
    assert_eq(index.class_tensor().shape, [3])

    let codes = index.encode(y)
    defer { codes.free() }
    assert_values(jl.to_values(codes), [1.0, 0.0, 1.0, 2.0, 0.0], "encode")
    let back = index.decode(codes)
    defer { back.free() }
    assert_values(jl.to_values(back), [5.0, 3.0, 5.0, 9.0, 3.0], "decode")
    assert_eq(jl.count_equal(back, y), 5, "decode is exact, so accuracy can compare codes")

    let stranger = jl.vector([4.0])
    defer { stranger.free() }
    assert_throws(ValueError, || index.encode(stranger))
    index.free()
}

fn test_cholesky_solves_the_normal_equations() -> void {
    if not jl.is_available() { return }
    let a = jl.matrix([[4.0, 2.0], [2.0, 3.0]])
    defer { a.free() }
    let factor = jl.cholesky_factor(a)
    defer { factor.free() }
    assert_values(jl.to_values(factor), [2.0, 0.0, 1.0, math.sqrt(2.0)], "cholesky_factor")
    let pivots = jl.diagonal(factor)
    defer { pivots.free() }
    assert_values(jl.to_values(pivots), [2.0, math.sqrt(2.0)], "diagonal")

    let b = jl.vector([1.0, 1.0])
    defer { b.free() }
    let solved = jl.cholesky_solve(a, b)
    defer { solved.free() }
    assert_eq(solved.shape, [2])
    assert_values(jl.to_values(solved), [0.125, 0.25], "cholesky_solve")

    let forward = jl.triangular_solve(factor, b)
    defer { forward.free() }
    assert_values(jl.to_values(forward), [0.5, 0.5 / math.sqrt(2.0)], "forward substitution")

    #: Several right-hand sides at once, which is what a multi-output ridge
    #: hands it.
    let many = jl.matrix([[1.0, 0.0], [1.0, 1.0]])
    defer { many.free() }
    let wide = jl.cholesky_solve(a, many)
    defer { wide.free() }
    assert_eq(wide.shape, [2, 2])
    assert_values(jl.to_values(wide), [0.125, -0.25, 0.25, 0.5], "cholesky_solve, two targets")

    let indefinite = jl.matrix([[1.0, 2.0], [2.0, 1.0]])
    defer { indefinite.free() }
    assert_throws(ValueError, || jl.cholesky_factor(indefinite))
}

fn test_the_estimator_protocol_defaults() -> void {
    if not jl.is_available() { return }
    let x = jl.matrix([[0.0], [1.0], [2.0], [3.0]])
    defer { x.free() }
    let y = jl.vector([1.0, 2.0, 3.0, 6.0])
    defer { y.free() }

    let model = MeanRegressor()
    defer { model.free() }
    assert_true(not model.is_fitted())
    assert_throws(RuntimeError, || model.predict(x))
    model.fit(x, y)
    assert_true(model.is_fitted())
    assert_eq(model.feature_count(), 1)
    #: A constant prediction at the mean explains none of the variance and no
    #: less than none, which is exactly R^2 = 0.
    assert_near(model.score(x, y), 0.0, 1e-5, "R^2 of the mean")
    assert_eq(jl.param_int(model.get_params(), "width", 0), 1)
    model.set_params(jl.params_of([("width", 1)]))

    let fresh = model.clone_unfitted()
    assert_true(not fresh.is_fitted(), "a clone is unfitted")
    assert_near(jl.default_score(model, x, y), 0.0, 1e-5, "default_score directly")

    #: Defaults a regressor has no business supporting refuse rather than
    #: answering, the way jaitensor's `Layer.static_forward` does.
    assert_throws(RuntimeError, || model.transform(x))
    assert_throws(RuntimeError, || model.inverse_transform(x))
    assert_throws(RuntimeError, || model.fit_transform(x, y))
    assert_throws(RuntimeError, || model.predict_proba(x))
    assert_throws(RuntimeError, || model.decision_function(x))

    let labels = jl.vector([1.0, 1.0, 1.0, 0.0])
    defer { labels.free() }
    let guesser = OneClassClassifier(1.0)
    defer { guesser.free() }
    guesser.fit(x, labels)
    assert_near(guesser.score(x, labels), 0.75, 1e-6, "accuracy")
    assert_eq(guesser.feature_count(), 0, "an estimator may record no width")
    assert_eq(guesser.kind(), EstimatorKind.Classifier)
}

fn test_cached_kernel_compiles_an_entry_point_once() -> void {
    if not jl.is_available() { return }
    let source = """#include <metal_stdlib>
using namespace metal;
kernel void jailearn_test_double(device const float* src [[buffer(0)]],
                                 device float* dst [[buffer(1)]],
                                 constant uint& count [[buffer(2)]],
                                 uint id [[thread_position_in_grid]]) {
    if (id < count) dst[id] = src[id] * 2.0f;
}
"""
    let first = jl.cached_kernel(source, "jailearn_test_double")
    let again = jl.cached_kernel(source, "jailearn_test_double")
    assert_true(first is again, "the cache is keyed on the entry name")
    let x = jl.vector([1.0, 2.0, 3.0])
    defer { x.free() }
    let out = Tensor([3])
    defer { out.free() }
    first.dispatch_async([x.buffer(), out.buffer()], 3, [3])
    assert_values(jl.to_values(out), [2.0, 4.0, 6.0], "a module's own kernel")
}
