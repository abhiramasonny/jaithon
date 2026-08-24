#: The layer between `base` and the estimator families: binning, histograms,
#: split choice, the device form of a tree, the samplers and the descent loop.
import jailearn as jl
import std.math
from jailearn import TreeArrays
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

#: A column of `0..n-1` as a one-feature design matrix, which makes every
#: binning strategy's answer something a reader can check by eye.
fn ramp(n: int) -> Tensor {
    var values: list[float] = []
    for step in 0..n { values.push(float(step)) }
    return Tensor([n, 1], values)
}

fn test_uniform_bins_split_the_range_evenly() -> void {
    let x = jl.matrix([[0.0, 10.0], [4.0, 30.0]])
    defer { x.free() }
    let edges = jl.bin_edges(x, 4, jl.BIN_UNIFORM)
    defer { edges.free() }
    assert_eq(edges.shape, [2, 3])
    assert_values(jl.to_values(edges), [1.0, 2.0, 3.0, 15.0, 20.0, 25.0], "uniform edges")

    let codes = jl.bin_codes(x, edges)
    defer { codes.free() }
    assert_values(jl.to_values(codes), [0.0, 0.0, 3.0, 3.0], "uniform codes")

    #: An edge belongs to the bin below it, so the cut at 2.0 codes to 1.
    let on_a_cut = jl.matrix([[2.0, 20.0]])
    defer { on_a_cut.free() }
    let boundary = jl.bin_codes(on_a_cut, edges)
    defer { boundary.free() }
    assert_values(jl.to_values(boundary), [1.0, 1.0], "a value sitting on a cut")

    let centres = jl.bin_centres(edges)
    defer { centres.free() }
    assert_eq(centres.shape, [2, 4])
    assert_values(
        jl.to_values(centres),
        [0.5, 1.5, 2.5, 3.5, 12.5, 17.5, 22.5, 27.5],
        "bin centres"
    )
}

fn test_quantile_bins_hold_the_same_count_and_uniform_ones_do_not() -> void {
    #: Seven samples bunched at zero and one far out: the case uniform bins get
    #: wrong, and the reason quantiles are the default.
    let x = jl.matrix([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [600.0]])
    defer { x.free() }

    let even = jl.bin_edges(x, 2, jl.BIN_UNIFORM)
    defer { even.free() }
    let even_codes = jl.bin_codes(x, even)
    defer { even_codes.free() }
    assert_values(jl.to_values(even_codes), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], "uniform codes")

    let quantile = jl.bin_edges(x, 2, jl.BIN_QUANTILE)
    defer { quantile.free() }
    assert_eq(quantile.shape, [1, 1])
    assert_values(jl.to_values(quantile), [3.0], "the median cut")
    let split = jl.bin_codes(x, quantile)
    defer { split.free() }
    assert_values(jl.to_values(split), [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0], "quantile codes")
}

fn test_kmeans_bins_follow_the_clusters() -> void {
    #: Two tight clumps a long way apart. Quantiles would cut inside a clump
    #: because that is where the counts are; k-means cuts between them.
    let x = jl.matrix([[0.0], [0.5], [1.0], [40.0], [40.5], [41.0]])
    defer { x.free() }
    let edges = jl.bin_edges(x, 2, jl.BIN_KMEANS)
    defer { edges.free() }
    let cut = jl.to_values(edges)[0]
    assert_true(cut > 1.0 and cut < 40.0, f"the k-means cut fell at {cut}")

    let codes = jl.bin_codes(x, edges)
    defer { codes.free() }
    assert_values(jl.to_values(codes), [0.0, 0.0, 0.0, 1.0, 1.0, 1.0], "k-means codes")
}

fn test_binning_refuses_what_it_cannot_hold() -> void {
    let x = ramp(8)
    defer { x.free() }
    assert_throws(ValueError, || jl.bin_edges(x, 1))
    assert_throws(ValueError, || jl.bin_edges(x, jl.MAX_BINS + 1))
    assert_throws(ValueError, || jl.bin_edges(x, 4, 9))

    let edges = jl.bin_edges(x, 4, jl.BIN_UNIFORM)
    defer { edges.free() }
    let wider = jl.matrix([[1.0, 2.0]])
    defer { wider.free() }
    assert_throws(ValueError, || jl.bin_codes(wider, edges))
}

fn test_a_missing_value_gets_no_bin() -> void {
    let x = jl.matrix([[0.0], [math.NAN], [3.0]])
    defer { x.free() }
    let edges = jl.matrix([[1.5]])
    defer { edges.free() }
    let codes = jl.bin_codes(x, edges)
    defer { codes.free() }
    assert_values(jl.to_values(codes), [0.0, -1.0, 1.0], "a NaN codes to -1")

    #: And the histogram drops it rather than counting it somewhere, so the
    #: two remaining samples are all that is totalled.
    let stats = jl.matrix([[1.0], [1.0], [1.0]])
    defer { stats.free() }
    let hist = jl.bin_histograms(codes, stats, null, 2)
    defer { hist.free() }
    assert_values(jl.to_values(hist), [1.0, 1.0, 1.0, 1.0], "counts skip the NaN")
}

fn test_histograms_total_every_statistic_and_the_count() -> void {
    let codes = jl.matrix([[0.0, 1.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
    defer { codes.free() }
    let stats = jl.matrix([[2.0], [3.0], [10.0], [20.0]])
    defer { stats.free() }
    let hist = jl.bin_histograms(codes, stats, null, 2)
    defer { hist.free() }
    assert_eq(hist.shape, [2, 2, 2])
    #: Feature 0 bin 0 holds rows 0 and 1, bin 1 holds rows 2 and 3; feature 1
    #: bin 0 holds row 1 alone. The second plane of each pair is the count the
    #: kernel writes, which no caller passed in.
    assert_values(jl.to_values(hist), [5.0, 2.0, 30.0, 2.0, 3.0, 1.0, 32.0, 3.0], "histogram")
}

fn test_a_row_subset_reweights_a_histogram_without_gathering() -> void {
    let codes = jl.matrix([[0.0], [0.0], [1.0], [1.0]])
    defer { codes.free() }
    let stats = jl.matrix([[1.0], [1.0], [1.0], [1.0]])
    defer { stats.free() }
    #: Row 3 drawn twice, as a bootstrap would, and rows 0 and 1 left out.
    let rows = jl.vector([2.0, 3.0, 3.0])
    defer { rows.free() }
    let hist = jl.bin_histograms(codes, stats, rows, 2)
    defer { hist.free() }
    assert_values(jl.to_values(hist), [0.0, 0.0, 3.0, 3.0], "the drawn rows only")
}

fn test_gini_finds_the_split_that_separates_the_classes() -> void {
    #: Feature 0 separates perfectly at bin 0; feature 1 tells the classes
    #: apart not at all, so the mask test below has something to hide.
    let codes = jl.matrix([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    defer { codes.free() }
    let one_hot = jl.matrix([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    defer { one_hot.free() }
    let hist = jl.bin_histograms(codes, one_hot, null, 2)
    defer { hist.free() }

    let rule = jl.SplitRule(jl.SPLIT_GINI)
    let gains = jl.split_gains(hist, rule)
    defer { gains.free() }
    assert_eq(gains.shape, [2, 2])
    let values = jl.to_values(gains)
    #: A pure two-way split of four samples: 2 + 2 - 4/2 = 2.
    assert_near(values[0], 2.0, 1e-4, "the perfect split")
    assert_true(values[1] == -math.INF, "the last bin is never a split")
    assert_true(values[2] == -math.INF, "a useless feature offers nothing")

    let (feature, bin, gain) = jl.best_split(hist, rule)
    assert_eq(feature, 0, "best feature")
    assert_eq(bin, 0, "best bin")
    assert_near(gain, 2.0, 1e-4, "best gain")
}

fn test_entropy_and_the_squared_error_agree_about_where_to_cut() -> void {
    let codes = jl.matrix([[0.0], [0.0], [1.0], [2.0]])
    defer { codes.free() }
    let one_hot = jl.matrix([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    defer { one_hot.free() }
    let counted = jl.bin_histograms(codes, one_hot, null, 3)
    defer { counted.free() }
    let (feature, bin, gain) = jl.best_split(counted, jl.SplitRule(jl.SPLIT_ENTROPY))
    assert_eq(feature, 0, "entropy feature")
    assert_eq(bin, 0, "entropy bin")
    #: Two pure children out of a balanced parent is the whole of the parent's
    #: entropy, 4 log 2.
    assert_near(gain, 4.0 * math.log(2.0), 1e-4, "entropy gain")

    let y = jl.matrix([[1.0, 1.0], [1.0, 1.0], [9.0, 81.0], [9.0, 81.0]])
    defer { y.free() }
    let regression = jl.bin_histograms(codes, y, null, 3)
    defer { regression.free() }
    let (where_, at, drop) = jl.best_split(regression, jl.SplitRule(jl.SPLIT_MSE))
    assert_eq(where_, 0, "mse feature")
    assert_eq(at, 0, "mse bin")
    #: 2 + 18 - 20 in means of squares: 2*1 + 2*81 against 4*25.
    assert_near(drop, 2.0 + 162.0 - 100.0, 1e-3, "mse gain")
}

fn test_a_gradient_split_is_damped_by_l2() -> void {
    let codes = jl.matrix([[0.0], [0.0], [1.0], [1.0]])
    defer { codes.free() }
    #: Gradients of -1 on the left and +1 on the right, unit hessians.
    let gh = jl.matrix([[-1.0, 1.0], [-1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    defer { gh.free() }
    let hist = jl.bin_histograms(codes, gh, null, 2)
    defer { hist.free() }

    let plain = jl.best_split(hist, jl.SplitRule(jl.SPLIT_GRADIENT))
    let (feature, bin, gain) = plain
    assert_eq(feature, 0, "gradient feature")
    assert_eq(bin, 0, "gradient bin")
    #: 0.5 * (4/2 + 4/2 - 0/4) with no ridge.
    assert_near(gain, 2.0, 1e-4, "gradient gain")

    let damped = jl.SplitRule(jl.SPLIT_GRADIENT, l2: 2.0)
    let (_f, _b, smaller) = jl.best_split(hist, damped)
    assert_near(smaller, 0.5 * (4.0 / 4.0 + 4.0 / 4.0), 1e-4, "the ridge shrinks the gain")
    assert_true(smaller < gain, "l2 can only shrink a gain")

    #: A hessian floor above what either child carries refuses every split.
    let starved = jl.SplitRule(jl.SPLIT_GRADIENT, min_child_weight: 3.0)
    let (none, nowhere, refused) = jl.best_split(hist, starved)
    assert_eq(none, -1, "no feature")
    assert_eq(nowhere, -1, "no bin")
    assert_true(refused == -math.INF, "no gain")
}

fn test_a_rule_refuses_small_children_masked_features_and_pure_nodes() -> void {
    let codes = jl.matrix([[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    defer { codes.free() }
    let one_hot = jl.matrix([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    defer { one_hot.free() }
    let hist = jl.bin_histograms(codes, one_hot, null, 2)
    defer { hist.free() }

    let (feature, bin, gain) = jl.best_split(hist, jl.SplitRule(jl.SPLIT_GINI))
    assert_eq(feature, 0, "the single-sample split is allowed at first")
    assert_eq(bin, 0, "at bin 0")
    assert_true(gain > 0.0, "and it is worth something")

    let strict = jl.SplitRule(jl.SPLIT_GINI, min_samples_leaf: 2)
    let (none, _nowhere, refused) = jl.best_split(hist, strict)
    assert_eq(none, -1, "a child of one is refused")
    assert_true(refused == -math.INF, "with no gain")

    #: The mask hides feature 0, which is the only one that could have won.
    let masked = jl.SplitRule(jl.SPLIT_GINI)
    let mask = jl.vector([0.0, 0.0])
    defer { mask.free() }
    masked.feature_mask = mask
    let (hidden, _bin, gone) = jl.best_split(hist, masked)
    assert_eq(hidden, -1, "a fully masked node has no split")
    assert_true(gone == -math.INF, "with no gain")

    #: A pure node's every split has gain exactly zero, and zero does not beat
    #: the default `min_gain`.
    let pure_codes = jl.matrix([[0.0], [1.0]])
    defer { pure_codes.free() }
    let pure_labels = jl.matrix([[1.0, 0.0], [1.0, 0.0]])
    defer { pure_labels.free() }
    let pure = jl.bin_histograms(pure_codes, pure_labels, null, 2)
    defer { pure.free() }
    let (nothing, _at, empty) = jl.best_split(pure, jl.SplitRule(jl.SPLIT_GINI))
    assert_eq(nothing, -1, "a pure node is a leaf")
    assert_true(empty == -math.INF, "with no gain")
}

fn test_split_gains_check_the_histogram_it_was_handed() -> void {
    let flat = jl.matrix([[1.0, 2.0]])
    defer { flat.free() }
    let rule = jl.SplitRule(jl.SPLIT_GINI)
    assert_throws(ValueError, || jl.split_gains(flat, rule))
    assert_throws(ValueError, || jl.SplitRule(9))

    let codes = jl.matrix([[0.0], [1.0]])
    defer { codes.free() }
    let single = jl.matrix([[1.0], [1.0]])
    defer { single.free() }
    let hist = jl.bin_histograms(codes, single, null, 2)
    defer { hist.free() }
    #: One statistic column plus the count is not a gradient and a hessian.
    let gradient = jl.SplitRule(jl.SPLIT_GRADIENT)
    assert_throws(ValueError, || jl.split_gains(hist, gradient))

    let wrong_mask = jl.vector([1.0, 1.0, 1.0])
    defer { wrong_mask.free() }
    let mismatched = jl.SplitRule(jl.SPLIT_GINI)
    mismatched.feature_mask = wrong_mask
    assert_throws(ValueError, || jl.split_gains(hist, mismatched))
}

#: A stump on feature 0 at 2.5, answering 10 on the left and 20 on the right.
fn stump() -> TreeArrays {
    return jl.TreeArrays(
        [0, -1, -1],
        [2.5, 0.0, 0.0],
        [1, -1, -1],
        [2, -1, -1],
        [[0.0], [10.0], [20.0]]
    )
}

fn test_a_tree_walks_every_sample_to_its_leaf() -> void {
    let tree = stump()
    defer { tree.free() }
    assert_eq(tree.node_count(), 3)
    assert_eq(tree.output_count(), 1)

    let x = jl.matrix([[0.0], [2.5], [3.0], [100.0]])
    defer { x.free() }
    let leaves = tree.apply(x)
    defer { leaves.free() }
    assert_values(jl.to_values(leaves), [1.0, 1.0, 2.0, 2.0], "leaf ids")

    let answer = tree.predict(x)
    defer { answer.free() }
    assert_eq(answer.shape, [4, 1])
    assert_values(jl.to_values(answer), [10.0, 10.0, 20.0, 20.0], "leaf values")
}

fn test_trees_accumulate_and_average_without_leaving_the_device() -> void {
    let first = stump()
    defer { first.free() }
    let second = jl.TreeArrays([-1], [0.0], [-1], [-1], [[4.0]])
    defer { second.free() }

    let x = jl.matrix([[0.0], [3.0]])
    defer { x.free() }
    let total = Tensor.zeros([2, 1])
    defer { total.free() }
    jl.tree_accumulate(first, x, total, 0.5)
    jl.tree_accumulate(second, x, total, 2.0)
    assert_values(jl.to_values(total), [5.0 + 8.0, 10.0 + 8.0], "weighted accumulation")

    let mean = jl.forest_mean([first, second], x, 1)
    defer { mean.free() }
    assert_values(jl.to_values(mean), [7.0, 12.0], "forest mean")
}

fn test_a_tree_refuses_arrays_that_do_not_describe_one() -> void {
    assert_throws(ValueError, || jl.TreeArrays([], [], [], [], []))
    assert_throws(ValueError, || jl.TreeArrays([0], [1.0, 2.0], [-1], [-1], [[0.0]]))
    assert_throws(
        ValueError,
        || jl.TreeArrays([-1, -1], [0.0, 0.0], [-1, -1], [-1, -1], [[1.0], [2.0, 3.0]])
    )
    assert_throws(IndexError, || jl.TreeArrays([0], [1.0], [7], [-1], [[0.0]]))

    let tree = stump()
    defer { tree.free() }
    let x = jl.matrix([[1.0]])
    defer { x.free() }
    let wrong = Tensor.zeros([2, 1])
    defer { wrong.free() }
    assert_throws(ValueError, || jl.tree_accumulate(tree, x, wrong, 1.0))
    assert_throws(ValueError, || jl.forest_mean([tree], x, 3))
    #: Freeing twice is harmless, which is what lets a `defer` sit beside a
    #: constructor that a later `free` also reaches.
    tree.free()
}

fn test_a_permutation_uses_every_index_once() -> void {
    let order = jl.permutation(64, seed: 7)
    defer { order.free() }
    assert_eq(order.shape, [64])
    var seen: list[bool] = []
    for _slot in 0..64 { seen.push(false) }
    for value in jl.to_values(order) {
        let at = int(value + 0.5)
        assert_true(at >= 0 and at < 64, f"index {at} is outside the permutation")
        assert_true(not seen[at], f"index {at} appeared twice")
        seen[at] = true
    }

    #: A seed reproduces itself, which is what `random_state` promises.
    let again = jl.permutation(64, seed: 7)
    defer { again.free() }
    assert_values(jl.to_values(again), jl.to_values(order), "the same seed")
}

fn test_bootstraps_and_feature_draws_stay_in_range() -> void {
    let drawn = jl.bootstrap_rows(10, 32, seed: 3)
    defer { drawn.free() }
    assert_eq(drawn.shape, [32])
    var repeated = false
    var counts: list[int] = []
    for _slot in 0..10 { counts.push(0) }
    for value in jl.to_values(drawn) {
        let at = int(value + 0.5)
        assert_true(at >= 0 and at < 10, f"row {at} is outside the data")
        counts[at] += 1
        if counts[at] > 1 { repeated = true }
    }
    #: Thirty-two draws from ten rows must repeat, which is the point of
    #: drawing with replacement.
    assert_true(repeated, "a bootstrap that never repeats is not one")
    assert_eq(jl.to_values(jl.bootstrap_rows(5, seed: 1)).len(), 5)

    let mask = jl.choose_features(9, 3, seed: 11)
    defer { mask.free() }
    var chosen = 0
    for value in jl.to_values(mask) {
        assert_true(value == 0.0 or value == 1.0, "a mask holds only ones and zeros")
        chosen += int(value + 0.5)
    }
    assert_eq(chosen, 3, "exactly k features")

    let everything = jl.choose_features(4, 0, seed: 2)
    defer { everything.free() }
    assert_values(jl.to_values(everything), [1.0, 1.0, 1.0, 1.0], "k of 0 means all")
}

fn test_sample_weights_default_to_one_and_are_copied() -> void {
    let ones = jl.sample_weights(null, 3)
    defer { ones.free() }
    assert_values(jl.to_values(ones), [1.0, 1.0, 1.0], "unweighted")

    let given = jl.vector([0.5, 2.0])
    defer { given.free() }
    let held = jl.sample_weights(given, 2)
    defer { held.free() }
    assert_values(jl.to_values(held), [0.5, 2.0], "weighted")
    #: A copy and not the caller's tensor, so freeing one does not empty the
    #: other.
    given.free()
    assert_values(jl.to_values(held), [0.5, 2.0], "still there after the source went")
    assert_throws(ValueError, || jl.sample_weights(held, 5))
}

fn test_the_linear_helpers() -> void {
    let x = jl.matrix([[1.0, 2.0], [3.0, 4.0]])
    defer { x.free() }
    let wide = jl.add_bias(x)
    defer { wide.free() }
    assert_eq(wide.shape, [2, 3])
    assert_values(jl.to_values(wide), [1.0, 2.0, 1.0, 3.0, 4.0, 1.0], "add_bias")

    let coef = jl.matrix([[1.0, 0.0], [0.0, 2.0]])
    defer { coef.free() }
    let intercept = jl.vector([10.0, 20.0])
    defer { intercept.free() }
    let scores = jl.linear_scores(x, coef, intercept)
    defer { scores.free() }
    assert_eq(scores.shape, [2, 2])
    assert_values(jl.to_values(scores), [11.0, 24.0, 13.0, 28.0], "linear_scores")

    let bare = jl.linear_scores(x, coef)
    defer { bare.free() }
    assert_values(jl.to_values(bare), [1.0, 4.0, 3.0, 8.0], "no intercept")
    assert_throws(ValueError, || jl.linear_scores(x, intercept.view(0, [1, 2]), coef))

    let raw = jl.vector([-3.0, -0.5, 0.0, 0.5, 3.0])
    defer { raw.free() }
    let shrunk = jl.soft_threshold(raw, 1.0)
    defer { shrunk.free() }
    assert_values(jl.to_values(shrunk), [-2.0, 0.0, 0.0, 0.0, 2.0], "soft_threshold")

    let norms = jl.column_norms(x)
    defer { norms.free() }
    assert_values(jl.to_values(norms), [math.sqrt(10.0), math.sqrt(20.0)], "column_norms")
    let squared = jl.column_norms(x, squared: true)
    defer { squared.free() }
    assert_values(jl.to_values(squared), [10.0, 20.0], "squared column_norms")
}

fn test_descent_reports_convergence_the_same_way_every_time() -> void {
    let slow = jl.Descent(max_iter: 50, tol: 0.01)
    slow.run(|step| 1.0 / float(step + 1))
    #: 1/100 is the first movement at or below the tolerance, so the hundredth
    #: step would converge -- but `max_iter` is fifty.
    assert_eq(slow.n_iter, 50, "ran to the cap")
    assert_true(not slow.converged, "and said so")

    let quick = jl.Descent(max_iter: 50, tol: 0.5)
    quick.run(|step| 1.0 / float(step + 1))
    assert_eq(quick.n_iter, 2, "stopped at the step that converged")
    assert_true(quick.converged, "and said so")

    quick.reset()
    assert_eq(quick.n_iter, 0, "reset forgets the run")
    assert_true(not quick.converged, "and the verdict with it")

    #: A fit that has left the representable range is stopped rather than run
    #: to the cap to produce a model made of NaNs.
    let diverging = jl.Descent(max_iter: 10, tol: 1e-6)
    assert_throws(RuntimeError, || diverging.run(|step| math.NAN))
    assert_throws(ValueError, || jl.Descent(max_iter: 0))
    assert_throws(ValueError, || jl.Descent(tol: -1.0))
}

fn test_a_wide_histogram_scans_and_reduces_correctly() -> void {
    #: Four features and thirty-two bins, which is past the point where the
    #: bin scan takes more than one Hillis-Steele step and where the split
    #: search has more than one feature to choose between. Feature 2 carries
    #: the label and the other three are noise that repeats every four rows.
    var rows: list[list[float]] = []
    var labels: list[list[float]] = []
    for row in 0..256 {
        let signal = float(row)
        rows.push([float(row % 4), float(row % 7), signal, float(row % 3)])
        labels.push(if row < 128 { [1.0, 0.0] } else { [0.0, 1.0] })
    }
    let x = jl.matrix(rows)
    defer { x.free() }
    let y = jl.matrix(labels)
    defer { y.free() }

    let edges = jl.bin_edges(x, 32, jl.BIN_UNIFORM)
    defer { edges.free() }
    assert_eq(edges.shape, [4, 31])
    let codes = jl.bin_codes(x, edges)
    defer { codes.free() }
    let hist = jl.bin_histograms(codes, y, null, 32)
    defer { hist.free() }
    assert_eq(hist.shape, [4, 32, 3])

    #: Every sample is counted once per feature, whichever bin it landed in.
    let planes = jl.to_values(hist)
    var counted = 0.0
    for bin in 0..32 { counted += planes[bin * 3 + 2] }
    assert_near(counted, 256.0, 1e-3, "feature 0 counts")

    let (feature, bin, gain) = jl.best_split(hist, jl.SplitRule(jl.SPLIT_GINI))
    assert_eq(feature, 2, "the feature that carries the label")
    #: 256 rows over 32 uniform bins is 8 to a bin, so the class boundary at
    #: row 128 falls between bins 15 and 16.
    assert_eq(bin, 15, "the bin the classes change at")
    assert_near(gain, 128.0, 1e-2, "a perfect split of 256 samples")

    #: The same histogram under the squared error, with the label as the
    #: target, must pick the same cut.
    var target: list[list[float]] = []
    for row in 0..256 {
        let value = if row < 128 { 0.0 } else { 1.0 }
        target.push([value, value * value])
    }
    let regression = jl.matrix(target)
    defer { regression.free() }
    let variance = jl.bin_histograms(codes, regression, null, 32)
    defer { variance.free() }
    let (where_, at, drop) = jl.best_split(variance, jl.SplitRule(jl.SPLIT_MSE))
    assert_eq(where_, 2, "mse picks the same feature")
    assert_eq(at, 15, "and the same bin")
    assert_near(drop, 64.0, 1e-2, "the whole of a balanced target's variance")
}
