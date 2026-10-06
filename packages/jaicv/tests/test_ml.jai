#: The classical models in `jaicv.ml` and the linear algebra under them, held
#: against the definitions they replaced or against what they promise.
import std.gpu as gpu
import std.math
from std.test import assert_eq, assert_near, assert_true
from jaicv.core.mat import Mat
from jaicv.core.types import CV_32FC1
from jaicv.ml.models import d_trees_create, r_trees_create, svm_create
from jaicv.ml.models import LogisticRegression, LOGISTIC_DEVICE_VALUES
from jaicv.core.linalg import matmul as linalg_matmul, on_device, symmetric_eigen_of
from jaicv.core.misc import calc_covar_matrix, mul_transposed
from jaicv.core.cluster import PCA, SOURCE as KMEANS_SOURCE
from std.gpu import Buffer, Kernel

fn have_gpu() -> bool { return gpu.is_available() }

fn next_state(state: int) -> int {
    return state *% 6364136223846793 +% 1442695040888963 & 0xFFFFFFFFFFFF
}

# --- the tree every threshold of every feature scored from scratch ----------
fn ref_impurity(targets: list[float], indices: list[int], regression: bool) -> float {
    if indices.len() == 0 { return 0.0 }
    if regression {
        var mean = 0.0
        for index in indices { mean += targets[index] }
        mean /= float(indices.len())
        var total = 0.0
        for index in indices {
            let gap = targets[index] - mean
            total += gap * gap
        }
        return total / float(indices.len())
    }
    var labels: list[float] = []
    var counts: list[int] = []
    for index in indices {
        let value = targets[index]
        var found = false
        for slot in 0..labels.len() {
            if labels[slot] != value { continue }
            counts[slot] += 1
            found = true
            break
        }
        if not found {
            labels.push(value)
            counts.push(1)
        }
    }
    var gini = 1.0
    for count in counts {
        let share = float(count) / float(indices.len())
        gini -= share * share
    }
    return gini
}

fn ref_answer(targets: list[float], indices: list[int], regression: bool) -> float {
    if indices.len() == 0 { return 0.0 }
    if regression {
        var total = 0.0
        for index in indices { total += targets[index] }
        return total / float(indices.len())
    }
    var labels: list[float] = []
    var counts: list[int] = []
    for index in indices {
        let value = targets[index]
        var found = false
        for slot in 0..labels.len() {
            if labels[slot] != value { continue }
            counts[slot] += 1
            found = true
            break
        }
        if not found {
            labels.push(value)
            counts.push(1)
        }
    }
    var best = labels[0]
    var most = counts[0]
    for slot in 1..labels.len() {
        if counts[slot] > most or counts[slot] == most and labels[slot] < best {
            most = counts[slot]
            best = labels[slot]
        }
    }
    return best
}

fn ref_grow(
    rows: list[list[float]],
    targets: list[float],
    indices: list[int],
    depth: int,
    max_depth: int,
    regression: bool,
    nodes: list[tuple[int, float, float, int, int]]
) -> int {
    let here = nodes.len()
    let value = ref_answer(targets, indices, regression)
    nodes.push((-1, 0.0, value, -1, -1))
    if depth >= max_depth or indices.len() < 4 { return here }
    var pure = true
    for index in indices {
        if targets[index] != targets[indices[0]] { pure = false }
    }
    if pure { return here }
    var best_feature = -1
    var best_threshold = 0.0
    var best_score = ref_impurity(targets, indices, regression)
    if best_score <= 0.0 { return here }
    for feature in 0..rows[0].len() {
        var values: list[float] = []
        for index in indices { values.push(rows[index][feature]) }
        values.sort()
        for at in 1..values.len() {
            if values[at] == values[at - 1] { continue }
            let threshold = (values[at] + values[at - 1]) * 0.5
            var left: list[int] = []
            var right: list[int] = []
            for index in indices {
                if rows[index][feature] <= threshold { left.push(index) } else { right.push(index) }
            }
            if left.len() == 0 or right.len() == 0 { continue }
            let score = (float(left.len()) * ref_impurity(targets, left, regression) + float(right.len()) * ref_impurity(targets, right, regression)) /
                float(indices.len())
            if score < best_score {
                best_score = score
                best_feature = feature
                best_threshold = threshold
            }
        }
    }
    if best_feature < 0 { return here }
    var left: list[int] = []
    var right: list[int] = []
    for index in indices {
        if rows[index][best_feature] <= best_threshold {
            left.push(index)
        } else {
            right.push(index)
        }
    }
    let left_at = ref_grow(rows, targets, left, depth + 1, max_depth, regression, nodes)
    let right_at = ref_grow(rows, targets, right, depth + 1, max_depth, regression, nodes)
    nodes[here] = (best_feature, best_threshold, value, left_at, right_at)
    return here
}

#: A small dataset whose features are coarse enough to tie: some columns are
#: small integers, one is sometimes a copy of another (so two features split
#: the samples identically), and the classes are few.
fn tree_dataset(seed: int, regression: bool) -> tuple[list[list[float]], list[float]] {
    var state = seed
    state = next_state(state)
    let count = 12 + (state >> 8) % 60
    state = next_state(state)
    let width = 1 + (state >> 8) % 5
    state = next_state(state)
    let levels = 2 + (state >> 8) % 6
    var rows: list[list[float]] = []
    var targets: list[float] = []
    for _sample in 0..count {
        var row: list[float] = []
        for feature in 0..width {
            state = next_state(state)
            let raw = (state >> 8) % 1000
            if feature == 2 and width > 3 {
                row.push(row[0])
            } elif feature % 2 == 0 {
                row.push(float(raw % levels))
            } else {
                row.push(float(raw) / 37.0 - 9.0)
            }
        }
        state = next_state(state)
        let noise = (state >> 8) % 4
        if regression {
            targets.push(row[0] * 1.5 - float(noise) * 0.25 + 1000.0)
        } else {
            targets.push(float((int(row[0]) + noise) % 3))
        }
        rows.push(row)
    }
    return (rows, targets)
}

fn flat(rows: list[list[float]]) -> list[float] {
    var out: list[float] = []
    for row in rows {
        for value in row { out.push(value) }
    }
    return out
}

fn test_trees_grow_node_for_node_what_scoring_every_threshold_grows() -> void {
    if not have_gpu() { return }
    for trial in 0..24 {
        let regression = trial % 3 == 2
        let (rows, targets) = tree_dataset(0x51ED + trial * 7919, regression)
        let width = rows[0].len()
        #: Through a float32 Mat and back, which is what the model sees.
        let samples = Mat.from_list(rows.len(), width, CV_32FC1, flat(rows))
        let responses = Mat.from_list(targets.len(), 1, CV_32FC1, targets)
        let seen = samples.to_list()
        var seen_rows: list[list[float]] = []
        for row in 0..rows.len() {
            var line: list[float] = []
            for column in 0..width { line.push(seen[row * width + column]) }
            seen_rows.push(line)
        }
        let seen_targets = responses.to_list()
        for max_depth in [2, 8] {
            var want: list[tuple[int, float, float, int, int]] = []
            var all: list[int] = []
            for index in 0..rows.len() { all.push(index) }
            ref_grow(seen_rows, seen_targets, all, 0, max_depth, regression, want)

            let tree = d_trees_create()
            tree.max_depth = max_depth
            tree.regression = regression
            tree.train(samples, responses)
            let got = tree.get_nodes()
            assert_eq(got.len(), want.len(), f"trial {trial} depth {max_depth}: node count")
            for at in 0..want.len() {
                assert_eq(got[at], want[at], f"trial {trial} depth {max_depth}: node {at}")
            }
        }
    }
}

fn test_a_forest_still_votes_for_the_class_it_was_shown() -> void {
    if not have_gpu() { return }
    var features: list[float] = []
    var labels: list[float] = []
    var state = 77
    for sample in 0..90 {
        let cluster = sample % 3
        state = next_state(state)
        let jitter = float((state >> 8) % 100) / 100.0 - 0.5
        features.push(float(cluster) * 3.0 + jitter)
        features.push(float(cluster) * -2.0 + jitter)
        labels.push(float(cluster))
    }
    let samples = Mat.from_list(90, 2, CV_32FC1, features)
    let forest = r_trees_create()
    forest.tree_count = 5
    forest.train(samples, Mat.from_list(90, 1, CV_32FC1, labels))
    let predicted = forest.predict(samples).to_list()
    for at in 0..90 { assert_eq(predicted[at], labels[at], f"sample {at}") }
}

# --- Jacobi with the pivot found by scanning every off-diagonal -------------
fn ref_eigen(source: list[float], n: int) -> tuple[list[float], list[float]] {
    var m: list[float] = []
    var v: list[float] = []
    for index in 0..n * n {
        m.push(source[index])
        v.push(if index % n == index // n { 1.0 } else { 0.0 })
    }
    for _turn in 0..max(100, 12 * n * (n - 1) // 2) {
        if n < 2 { break }
        var largest = 0.0
        var p = 0
        var q = 1
        for row in 0..n {
            for column in row + 1..n {
                let size = math.fabs(m[row * n + column])
                if size > largest {
                    largest = size
                    p = row
                    q = column
                }
            }
        }
        if largest < 1e-14 { break }
        let theta = 0.5 * math.atan2(2.0 * m[p * n + q], m[p * n + p] - m[q * n + q])
        let c = math.cos(theta)
        let s = math.sin(theta)
        for k in 0..n {
            let mkp = m[k * n + p]
            let mkq = m[k * n + q]
            m[k * n + p] = c * mkp + s * mkq
            m[k * n + q] = -s * mkp + c * mkq
        }
        for k in 0..n {
            let mpk = m[p * n + k]
            let mqk = m[q * n + k]
            m[p * n + k] = c * mpk + s * mqk
            m[q * n + k] = -s * mpk + c * mqk
        }
        for k in 0..n {
            let vkp = v[k * n + p]
            let vkq = v[k * n + q]
            v[k * n + p] = c * vkp + s * vkq
            v[k * n + q] = -s * vkp + c * vkq
        }
    }
    var order: list[int] = []
    for index in 0..n {
        var at = order.len()
        while at > 0 and m[index * n + index] > m[order[at - 1] * n + order[at - 1]] { at -= 1 }
        order.insert(at, index)
    }
    var values: list[float] = []
    var vectors: list[float] = []
    for index in order {
        values.push(m[index * n + index])
        for row in 0..n { vectors.push(v[row * n + index]) }
    }
    return (values, vectors)
}

#: A symmetric matrix; `coarse` ones are small integers, so equal entries --
#: and so ties for the pivot -- are everywhere.
fn symmetric(n: int, seed: int, coarse: bool) -> list[float] {
    var state = seed
    var out: list[float] = []
    for _index in 0..n * n { out.push(0.0) }
    for row in 0..n {
        for column in row..n {
            state = next_state(state)
            let raw = (state >> 8) % 2001
            let value = if coarse { float(raw % 5) - 2.0 } else { float(raw) / 1000.0 - 1.0 }
            out[row * n + column] = value
            out[column * n + row] = value
        }
    }
    return out
}

fn test_eigen_rotates_through_the_same_pivots_as_a_full_scan() -> void {
    for n in [1, 2, 3, 5, 8, 13, 24, 50] {
        for coarse in [false, true] {
            let source = symmetric(n, 0xE16E + n * 31 + (if coarse { 7 } else { 0 }), coarse)
            let (want_values, want_vectors) = ref_eigen(source, n)
            let (values, vectors) = symmetric_eigen_of(source, n)
            assert_eq(values, want_values, f"{n} by {n} coarse={coarse}: values")
            assert_eq(vectors, want_vectors, f"{n} by {n} coarse={coarse}: vectors")
        }
    }
}

# --- the support vector machine reaches the optimum it promises -------------

#: Two interleaved classes no line separates: the sign of `x * y`, with the
#: labels given as 3 and 7 so nothing leans on them being 0 and 1.
fn checkerboard(count: int, seed: int) -> tuple[Mat, Mat, list[float]] {
    var state = seed
    var features: list[float] = []
    var labels: list[float] = []
    for _sample in 0..count {
        state = next_state(state)
        let x = float((state >> 8) % 4001) / 1000.0 - 2.0
        state = next_state(state)
        let y = float((state >> 8) % 4001) / 1000.0 - 2.0
        features.push(x)
        features.push(y)
        labels.push(if x * y > 0.0 { 7.0 } else { 3.0 })
    }
    return (
        Mat.from_list(count, 2, CV_32FC1, features),
        Mat.from_list(count, 1, CV_32FC1, labels),
        labels
    )
}

fn test_the_svm_stops_at_the_optimum_of_its_dual() -> void {
    if not have_gpu() { return }
    let (samples, responses, labels) = checkerboard(240, 0x5B3)
    let model = svm_create()
    model.set_gamma(0.5)
    model.set_c(4.0)
    model.train(samples, responses)
    let (rho, weights, support) = model.get_decision_function()
    assert_true(support.len() > 0, "no support vectors")
    assert_eq(model.get_support_vectors().rows, support.len())

    #: The weights are the multipliers signed by class, and the multipliers of
    #: the two classes have to balance.
    var balance = 0.0
    var alpha: list[float] = []
    for _index in 0..labels.len() { alpha.push(0.0) }
    for at in 0..support.len() {
        balance += weights[at]
        alpha[support[at]] = math.fabs(weights[at])
    }
    assert_true(math.fabs(balance) < 1e-9, f"the multipliers do not balance: {balance}")

    #: The Karush-Kuhn-Tucker conditions, to the solver's tolerance: a sample
    #: with no weight is outside the margin, one at the bound is inside it, and
    #: one in between sits on it.
    let decisions = model.decision_function(samples).to_list()
    let slack = 2.0 * model.tolerance
    var free = 0
    for index in 0..labels.len() {
        let sign = if labels[index] == 7.0 { 1.0 } else { -1.0 }
        let margin = sign * decisions[index]
        if alpha[index] == 0.0 {
            assert_true(margin >= 1.0 - slack, f"sample {index}: weightless but margin {margin}")
        } elif alpha[index] >= model.c {
            assert_true(margin <= 1.0 + slack, f"sample {index}: at the bound but margin {margin}")
        } else {
            free += 1
            assert_true(
                math.fabs(margin - 1.0) <= slack,
                f"sample {index}: free but margin {margin}"
            )
        }
    }
    assert_true(free > 0, "no multiplier strictly inside the box")
    assert_true(rho == rho, "rho is not a number")

    #: And it has learned the pattern: fresh points from the same board.
    let (unseen, _responses, truth) = checkerboard(200, 0xA11)
    let predicted = model.predict(unseen).to_list()
    var right = 0
    for index in 0..truth.len() {
        if predicted[index] == truth[index] { right += 1 }
    }
    assert_true(right >= 180, f"only {right} of 200 unseen points classified correctly")
}

# --- products, covariances and PCA big enough to go to the device -----------
fn noise(count: int, seed: int, spread: float, shift: float) -> list[float] {
    var state = seed
    var out: list[float] = []
    for _index in 0..count {
        state = next_state(state)
        out.push((float((state >> 8) % 20001) / 10000.0 - 1.0) * spread + shift)
    }
    return out
}

fn test_a_large_product_agrees_with_the_host_sum() -> void {
    if not have_gpu() { return }
    let (rows, inner, cols) = (300, 200, 40)
    assert_true(on_device(rows * inner * cols), "this product is meant to reach the device")
    let left = noise(rows * inner, 11, 1.0, 0.0)
    let right = noise(inner * cols, 12, 1.0, 0.0)
    let got = linalg_matmul(
        Mat.from_list(rows, inner, CV_32FC1, left),
        Mat.from_list(inner, cols, CV_32FC1, right)
    )
    let a = Mat.from_list(rows, inner, CV_32FC1, left).to_list()
    let b = Mat.from_list(inner, cols, CV_32FC1, right).to_list()
    let values = got.to_list()
    var worst = 0.0
    for r in 0..rows {
        for c in 0..cols {
            var total = 0.0
            for k in 0..inner { total += a[r * inner + k] * b[k * cols + c] }
            worst = max(worst, math.fabs(values[r * cols + c] - total))
        }
    }
    assert_true(worst < 1e-3, f"device product is {worst} from the host sum")

    #: Both orientations of `mul_transposed`.
    let source = Mat.from_list(rows, inner, CV_32FC1, left)
    let gram = mul_transposed(source, true).to_list()
    let outer = mul_transposed(source, false).to_list()
    worst = 0.0
    for i in 0..inner {
        for j in 0..inner {
            var total = 0.0
            for k in 0..rows { total += a[k * inner + i] * a[k * inner + j] }
            worst = max(worst, math.fabs(gram[i * inner + j] - total))
        }
    }
    for i in 0..rows {
        for j in 0..rows {
            var total = 0.0
            for k in 0..inner { total += a[i * inner + k] * a[j * inner + k] }
            worst = max(worst, math.fabs(outer[i * rows + j] - total))
        }
    }
    assert_true(worst < 1e-3, f"device mul_transposed is {worst} from the host sum")
}

fn test_a_large_covariance_and_pca_agree_with_double_precision() -> void {
    if not have_gpu() { return }
    let (rows, width) = (20000, 9)
    assert_true(on_device(rows * width * width), "this covariance is meant to reach the device")
    var values: list[float] = []
    let base = noise(rows * width, 21, 1.0, 0.0)
    for r in 0..rows {
        for c in 0..width {
            #: Columns of different spread, mixed a little, far from the origin.
            let mixed = base[r * width + c] * (1.0 + float(c)) +
                0.3 * base[r * width + (c + 1) % width]
            values.push(mixed + 50.0)
        }
    }
    let samples = Mat.from_list(rows, width, CV_32FC1, values)
    let seen = samples.to_list()
    var centre: list[float] = []
    for c in 0..width {
        var total = 0.0
        for r in 0..rows { total += seen[r * width + c] }
        centre.push(total / float(rows))
    }
    var want: list[float] = []
    for i in 0..width {
        for j in 0..width {
            var total = 0.0
            for r in 0..rows {
                total += (seen[r * width + i] - centre[i]) * (seen[r * width + j] - centre[j])
            }
            want.push(total / float(rows - 1))
        }
    }
    let (covariance, mean) = calc_covar_matrix(samples)
    let got = covariance.to_list()
    let got_mean = mean.to_list()
    for c in 0..width { assert_near(got_mean[c], centre[c], 1e-4) }
    for index in 0..width * width {
        assert_true(
            math.fabs(got[index] - want[index]) <= 1e-4 * (1.0 + math.fabs(want[index])),
            f"covariance entry {index}: {got[index]} against {want[index]}"
        )
    }

    let (want_values, want_vectors) = symmetric_eigen_of(want, width)
    let analysis = PCA(samples, 4)
    let vectors = analysis.eigenvectors.to_list()
    for k in 0..4 {
        assert_true(
            math.fabs(analysis.eigenvalues[k] - want_values[k]) <= 1e-4 * want_values[k],
            f"eigenvalue {k}"
        )
        var dot = 0.0
        for c in 0..width { dot += vectors[k * width + c] * want_vectors[k * width + c] }
        #: Same direction AND the same sign as the double-precision solve.
        assert_true(dot > 0.9999, f"eigenvector {k} is at {dot} to the double-precision one")
    }

    #: The projection of every sample, against the host's.
    let projected = analysis.project(samples).to_list()
    var worst = 0.0
    for r in 0..rows {
        for k in 0..4 {
            var total = 0.0
            for c in 0..width {
                total += (seen[r * width + c] - got_mean[c]) * vectors[k * width + c]
            }
            worst = max(worst, math.fabs(projected[r * 4 + k] - total))
        }
    }
    assert_true(worst < 1e-3, f"device projection is {worst} from the host sum")
}

# --- logistic regression's device gradient takes the host's steps -----------
fn test_logistic_regression_on_the_device_takes_the_host_steps() -> void {
    if not have_gpu() { return }
    let (count, width) = (3000, 21)
    assert_true(
        count * (width + 1) >= LOGISTIC_DEVICE_VALUES,
        "this set is meant to reach the device"
    )
    let features = noise(count * width, 31, 2.0, 0.5)
    var labels: list[float] = []
    for r in 0..count {
        var score = 0.3
        for c in 0..width {
            score += features[r * width + c] * (if c % 3 == 0 { 1.0 } else { -0.4 })
        }
        labels.push(if score > 0.0 { 4.0 } else { -1.0 })
    }
    let samples = Mat.from_list(count, width, CV_32FC1, features)
    let responses = Mat.from_list(count, 1, CV_32FC1, labels)
    let model = LogisticRegression()
    model.iterations = 60
    model.train(samples, responses)
    let got = model.get_learnt_thetas().to_list()

    #: The same descent, with the gradient summed on the host in double.
    let seen = samples.to_list()
    var weights: list[float] = []
    for _index in 0..width + 1 { weights.push(0.0) }
    for _step in 0..60 {
        var gradient: list[float] = []
        for _index in 0..width + 1 { gradient.push(0.0) }
        for r in 0..count {
            var total = weights[width]
            for c in 0..width { total += weights[c] * seen[r * width + c] }
            let error = 1.0 / (1.0 + math.exp(-total)) - (if labels[r] == 4.0 { 1.0 } else { 0.0 })
            gradient[width] += error
            for c in 0..width { gradient[c] += error * seen[r * width + c] }
        }
        let scale = model.learning_rate / float(count)
        for c in 0..width {
            weights[c] -= scale * (gradient[c] + model.regularisation * weights[c])
        }
        weights[width] -= scale * gradient[width]
    }
    assert_near(got[0], weights[width], 1e-4)
    for c in 0..width { assert_near(got[c + 1], weights[c], 1e-4) }

    let predicted = model.predict(samples).to_list()
    var right = 0
    for r in 0..count {
        if predicted[r] == labels[r] { right += 1 }
    }
    assert_true(
        right > count * 9 // 10,
        f"only {right} of {count} training samples classified correctly"
    )
}

# --- the kmeans kernels that replaced the per-centre ones agree bit for bit --
fn test_kmeans_staged_and_column_kernels_match_the_originals_bit_for_bit() -> void {
    if not have_gpu() { return }
    for shape in [(5000, 7, 5), (20000, 50, 8), (3001, 3, 16), (777, 64, 1)] {
        let (count, width, k) = shape
        let values = Buffer.of(noise(count * width, count + width, 3.0, 1.0))
        let centres = Buffer.of(noise(k * width, k * 7 + 1, 3.0, 1.0))
        var held: list[Buffer] = []
        for _index in 0..6 { held.push(Buffer(count)) }
        let first_sums = Buffer(k * width + k + 1)
        let second_sums = Buffer(k * width + k + 1)
        Kernel.compile(KMEANS_SOURCE, "kmeans_assign").dispatch_async(
            [values, centres, held[0], held[1]],
            count,
            [count, width, k]
        )
        Kernel.compile(KMEANS_SOURCE, "kmeans_assign_staged").dispatch_async(
            [values, centres, held[2], held[3]],
            count,
            [count, width, k],
            group_size: 256
        )
        Kernel.compile(KMEANS_SOURCE, "kmeans_assign_tiled").dispatch_async(
            [values, centres, held[4], held[5]],
            (count + 255) // 256 * 256,
            [count, width, k],
            group_size: 256
        )
        Kernel.compile(KMEANS_SOURCE, "kmeans_gather").dispatch_async(
            [values, held[0], held[1], first_sums],
            (k * width + k + 1) * 256,
            [count, width, k],
            group_size: 256
        )
        Kernel.compile(KMEANS_SOURCE, "kmeans_gather_columns").dispatch_async(
            [values, held[0], held[1], second_sums],
            (width + 1) * 256,
            [count, width, k, 0],
            group_size: 256
        )
        let columns = Buffer(count * width)
        let third_sums = Buffer(k * width + k + 1)
        Kernel.compile(KMEANS_SOURCE, "kmeans_transpose").dispatch_async(
            [values, columns],
            count * width,
            [count, width]
        )
        Kernel.compile(KMEANS_SOURCE, "kmeans_gather_columns").dispatch_async(
            [columns, held[0], held[1], third_sums],
            (width + 1) * 256,
            [count, width, k, 1],
            group_size: 256
        )
        assert_eq(held[2].download(), held[0].download(), f"{shape}: labels")
        assert_eq(held[3].download(), held[1].download(), f"{shape}: distances")
        assert_eq(held[4].download(), held[0].download(), f"{shape}: tiled labels")
        assert_eq(held[5].download(), held[1].download(), f"{shape}: tiled distances")
        assert_eq(second_sums.download(), first_sums.download(), f"{shape}: totals")
        assert_eq(third_sums.download(), first_sums.download(), f"{shape}: totals by column")
        for buffer in [values, centres, first_sums, second_sums, third_sums, columns] {
            buffer.free()
        }
        for buffer in held { buffer.free() }
    }
}
