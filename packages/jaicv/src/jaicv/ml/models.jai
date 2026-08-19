#: The classifiers and regressors OpenCV's `ml` module offers.
#:
#: Every model here takes the same shape of data OpenCV's does — one sample per
#: row, one feature per column, and responses as a column — and every one has
#: `train` and `predict`. The factory functions are named the way OpenCV names
#: them, so `k_nearest_create()` stands where `cv2.ml.KNearest_create()` would.
#:
#: These run on the host. A model is trained once on data that fits in memory
#: and then asked about a handful of samples at a time; the arithmetic is small
#: next to the cost of getting it to the device and back.
#:
#: ```
#: let model = svm_create()
#: model.train(samples, responses)
#: let predicted = model.predict(unseen)
#: ```
import std.math
from ..core.linalg import inverse, matmul, transpose
from ..core.mat import Mat
from ..core.types import CV_32F, CV_MAKETYPE

#: Samples are rows, which is the layout everything here expects.
pub const ROW_SAMPLE = 0
#: Samples are columns.
pub const COL_SAMPLE = 1

#: Classification with a soft margin.
pub const SVM_C_SVC = 100
#: Regression with a tolerance band.
pub const SVM_EPS_SVR = 103

#: A dot product; the separating surface is a plane.
pub const SVM_LINEAR = 0
#: A radial basis function; the surface can curve.
pub const SVM_RBF = 2
#: A polynomial.
pub const SVM_POLY = 1

fn rows_of(data: Mat) -> list[list[float]] {
    data.flush()
    let values = data.to_list()
    let width = data.cols * data.channels()
    var out: list[list[float]] = []
    for row in 0..data.rows {
        var line: list[float] = []
        for column in 0..width { line.push(values[row * width + column]) }
        out.push(line)
    }
    return out
}

fn column_of(data: Mat) -> list[float] {
    data.flush()
    return data.to_list()
}

fn column_mat(values: list[float]) -> Mat {
    return Mat.from_list(values.len(), 1, CV_MAKETYPE(CV_32F, 1), values)
}

#: Classify a sample by what its nearest neighbours are.
#:
#: There is no training beyond remembering the data, which is what makes this
#: the first thing to try and the last thing to ship.
pub class KNearest {
    pub var default_k: int
    var samples: list[list[float]]
    var responses: list[float]

    pub fn init(self, default_k: int = 3) {
        self.default_k = default_k
        self.samples = []
        self.responses = []
    }

    #: Remember the training data.
    pub fn train(self, samples: Mat, responses: Mat) -> bool {
        self.samples = rows_of(samples)
        self.responses = column_of(responses)
        if self.samples.len() != self.responses.len() {
            throw ValueError("there has to be one response per sample")
        }
        return true
    }

    #: The `k` nearest neighbours of each sample, and what they say.
    #:
    #: Returns the prediction for each sample, the responses of the neighbours
    #: that decided it, and how far away each of those was — which is what
    #: OpenCV's `findNearest` returns.
    pub fn find_nearest(self, samples: Mat, k: int = 0) -> tuple[Mat, Mat, Mat] {
        let wanted = if k > 0 { k } else { self.default_k }
        let queries = rows_of(samples)
        if self.samples.len() == 0 { throw RuntimeError("this model has not been trained") }
        let neighbours = min(wanted, self.samples.len())

        var predictions: list[float] = []
        var responses: list[float] = []
        var distances: list[float] = []
        for query in queries {
            var order: list[int] = []
            var gaps: list[float] = []
            for index in 0..self.samples.len() {
                var total = 0.0
                for at in 0..query.len() {
                    let difference = query[at] - self.samples[index][at]
                    total += difference * difference
                }
                var slot = order.len()
                while slot > 0 and total < gaps[slot - 1] { slot -= 1 }
                order.insert(slot, index)
                gaps.insert(slot, total)
            }

            #: A vote among the neighbours, with ties going to the smaller
            #: response so the answer does not depend on the order they came in.
            var options: list[float] = []
            var counts: list[int] = []
            for at in 0..neighbours {
                let response = self.responses[order[at]]
                responses.push(response)
                distances.push(math.sqrt(gaps[at]))
                var found = false
                for slot in 0..options.len() {
                    if options[slot] != response { continue }
                    counts[slot] += 1
                    found = true
                    break
                }
                if not found {
                    options.push(response)
                    counts.push(1)
                }
            }
            var best = options[0]
            var most = counts[0]
            for slot in 1..options.len() {
                if counts[slot] > most or counts[slot] == most and options[slot] < best {
                    most = counts[slot]
                    best = options[slot]
                }
            }
            predictions.push(best)
        }
        return (
            column_mat(predictions),
            Mat.from_list(queries.len(), neighbours, CV_MAKETYPE(CV_32F, 1), responses),
            Mat.from_list(queries.len(), neighbours, CV_MAKETYPE(CV_32F, 1), distances)
        )
    }

    #: What each sample most likely is.
    pub fn predict(self, samples: Mat) -> Mat {
        let (predictions, _neighbours, _distances) = self.find_nearest(samples)
        return predictions
    }
}

#: Build a nearest-neighbour classifier.
pub fn k_nearest_create(default_k: int = 3) -> KNearest { return KNearest(default_k) }

#: Classify by assuming each class is a Gaussian cloud.
#:
#: The mean and covariance of every class are estimated from the training data,
#: and a sample is given to whichever class finds it least surprising.
pub class NormalBayesClassifier {
    var labels: list[float]
    var means: list[list[float]]
    var inverses: list[Mat]
    var offsets: list[float]

    pub fn init(self) {
        self.labels = []
        self.means = []
        self.inverses = []
        self.offsets = []
    }

    #: Estimate each class's cloud.
    pub fn train(self, samples: Mat, responses: Mat) -> bool {
        let rows = rows_of(samples)
        let targets = column_of(responses)
        if rows.len() != targets.len() {
            throw ValueError("there has to be one response per sample")
        }
        let width = if rows.len() > 0 { rows[0].len() } else { 0 }

        var labels: list[float] = []
        for value in targets {
            if not labels.contains(value) { labels.push(value) }
        }
        labels.sort()

        self.labels = labels
        self.means = []
        self.inverses = []
        self.offsets = []
        for label in labels {
            var members: list[list[float]] = []
            for index in 0..rows.len() {
                if targets[index] == label { members.push(rows[index]) }
            }
            var centre: list[float] = []
            for column in 0..width {
                var total = 0.0
                for member in members { total += member[column] }
                centre.push(total / float(max(members.len(), 1)))
            }
            var covariance: list[float] = []
            for _index in 0..width * width { covariance.push(0.0) }
            for member in members {
                for i in 0..width {
                    let a = member[i] - centre[i]
                    for j in 0..width { covariance[i * width + j] += a * (member[j] - centre[j]) }
                }
            }
            let divisor = float(max(members.len() - 1, 1))
            for index in 0..width * width { covariance[index] /= divisor }
            #: A ridge on the diagonal, because a class whose samples all agree
            #: on some feature would otherwise have a covariance with no
            #: inverse.
            for index in 0..width { covariance[index * width + index] += 1e-6 }
            let matrix = Mat.from_list(width, width, CV_MAKETYPE(CV_32F, 1), covariance)
            self.means.push(centre)
            self.inverses.push(inverse(matrix))
            self.offsets.push(math.log(float(members.len()) / float(rows.len())))
        }
        return true
    }

    #: What each sample most likely is.
    pub fn predict(self, samples: Mat) -> Mat {
        if self.labels.len() == 0 { throw RuntimeError("this model has not been trained") }
        let queries = rows_of(samples)
        var out: list[float] = []
        for query in queries {
            var best = self.labels[0]
            var score = -1e300
            for index in 0..self.labels.len() {
                let centre = self.means[index]
                let precision = self.inverses[index].to_list()
                let width = centre.len()
                var quadratic = 0.0
                for i in 0..width {
                    var row = 0.0
                    for j in 0..width { row += precision[i * width + j] * (query[j] - centre[j]) }
                    quadratic += (query[i] - centre[i]) * row
                }
                let candidate = self.offsets[index] - 0.5 * quadratic
                if candidate > score {
                    score = candidate
                    best = self.labels[index]
                }
            }
            out.push(best)
        }
        return column_mat(out)
    }
}

#: Build a Gaussian classifier.
pub fn normal_bayes_classifier_create() -> NormalBayesClassifier {
    return NormalBayesClassifier()
}

#: A support vector machine.
#:
#: Trained by sequential minimal optimisation: repeatedly pick the pair of
#: multipliers that most violates the conditions the solution has to satisfy,
#: and solve for those two exactly. Every step is a closed form, so there is no
#: inner solver and nothing to converge except the choice of pairs.
pub class SVM {
    pub var kernel_type: int
    pub var c: float
    pub var gamma: float
    pub var degree: float
    pub var coef0: float
    pub var iterations: int
    pub var tolerance: float
    var samples: list[list[float]]
    var targets: list[float]
    var alphas: list[float]
    var bias: float
    var positive: float
    var negative: float

    pub fn init(self) {
        self.kernel_type = SVM_RBF
        self.c = 1.0
        self.gamma = 0.1
        self.degree = 3.0
        self.coef0 = 0.0
        self.iterations = 200
        self.tolerance = 1e-3
        self.samples = []
        self.targets = []
        self.alphas = []
        self.bias = 0.0
        self.positive = 1.0
        self.negative = 0.0
    }

    #: Choose the kernel, which is what decides the shape of the boundary.
    pub fn set_kernel(self, kernel_type: int) -> void { self.kernel_type = kernel_type }

    #: How much a misclassified sample costs.
    pub fn set_c(self, c: float) -> void { self.c = c }

    #: How far a single sample's influence reaches, for the radial kernel.
    pub fn set_gamma(self, gamma: float) -> void { self.gamma = gamma }

    fn kernel(self, a: list[float], b: list[float]) -> float {
        if self.kernel_type == SVM_LINEAR {
            var total = 0.0
            for index in 0..a.len() { total += a[index] * b[index] }
            return total
        }
        if self.kernel_type == SVM_POLY {
            var total = 0.0
            for index in 0..a.len() { total += a[index] * b[index] }
            return (self.gamma * total + self.coef0) ** self.degree
        }
        var total = 0.0
        for index in 0..a.len() {
            let gap = a[index] - b[index]
            total += gap * gap
        }
        return math.exp(-self.gamma * total)
    }

    fn decide(self, query: list[float]) -> float {
        var total = self.bias
        for index in 0..self.samples.len() {
            if self.alphas[index] == 0.0 { continue }
            total += self.alphas[index] *
                self.targets[index] *
                self.kernel(self.samples[index], query)
        }
        return total
    }

    #: Fit the machine to two labelled classes.
    pub fn train(self, samples: Mat, responses: Mat) -> bool {
        let rows = rows_of(samples)
        let targets = column_of(responses)
        if rows.len() != targets.len() {
            throw ValueError("there has to be one response per sample")
        }

        var labels: list[float] = []
        for value in targets {
            if not labels.contains(value) { labels.push(value) }
        }
        labels.sort()
        if labels.len() != 2 { throw ValueError("this machine separates exactly two classes") }
        self.negative = labels[0]
        self.positive = labels[1]

        var signs: list[float] = []
        for value in targets { signs.push(if value == self.positive { 1.0 } else { -1.0 }) }
        self.samples = rows
        self.targets = signs
        self.alphas = []
        for _index in 0..rows.len() { self.alphas.push(0.0) }
        self.bias = 0.0

        let count = rows.len()
        for _pass in 0..self.iterations {
            var changed = 0
            for i in 0..count {
                let error_i = self.decide(rows[i]) - signs[i]
                let violates = signs[i] * error_i < -self.tolerance and self.alphas[i] < self.c or
                    signs[i] * error_i > self.tolerance and self.alphas[i] > 0.0
                if not violates { continue }
                #: The partner is chosen as the next sample along, which
                #: converges as surely as the heuristic pick and costs nothing
                #: to find.
                let j = (i + 1 + _pass) % count
                if j == i { continue }
                let error_j = self.decide(rows[j]) - signs[j]
                let old_i = self.alphas[i]
                let old_j = self.alphas[j]

                var low = 0.0
                var high = self.c
                if signs[i] != signs[j] {
                    low = max(0.0, old_j - old_i)
                    high = min(self.c, self.c + old_j - old_i)
                } else {
                    low = max(0.0, old_i + old_j - self.c)
                    high = min(self.c, old_i + old_j)
                }
                if high - low < 1e-12 { continue }

                let kii = self.kernel(rows[i], rows[i])
                let kjj = self.kernel(rows[j], rows[j])
                let kij = self.kernel(rows[i], rows[j])
                let curvature = 2.0 * kij - kii - kjj
                if curvature >= -1e-12 { continue }

                var moved_j = old_j - signs[j] * (error_i - error_j) / curvature
                if moved_j > high { moved_j = high }
                if moved_j < low { moved_j = low }
                if math.fabs(moved_j - old_j) < 1e-9 { continue }
                let moved_i = old_i + signs[i] * signs[j] * (old_j - moved_j)

                self.alphas[i] = moved_i
                self.alphas[j] = moved_j

                let b1 = self.bias -
                    error_i -
                    signs[i] * (moved_i - old_i) * kii -
                    signs[j] * (moved_j - old_j) * kij
                let b2 = self.bias -
                    error_j -
                    signs[i] * (moved_i - old_i) * kij -
                    signs[j] * (moved_j - old_j) * kjj
                if moved_i > 0.0 and moved_i < self.c {
                    self.bias = b1
                } elif moved_j > 0.0 and moved_j < self.c {
                    self.bias = b2
                } else {
                    self.bias = (b1 + b2) * 0.5
                }
                changed += 1
            }
            if changed == 0 { break }
        }
        return true
    }

    #: Which side of the boundary each sample falls on.
    pub fn predict(self, samples: Mat) -> Mat {
        if self.samples.len() == 0 { throw RuntimeError("this model has not been trained") }
        var out: list[float] = []
        for query in rows_of(samples) {
            out.push(if self.decide(query) >= 0.0 { self.positive } else { self.negative })
        }
        return column_mat(out)
    }

    #: How far each sample is from the boundary, signed.
    pub fn decision_function(self, samples: Mat) -> Mat {
        var out: list[float] = []
        for query in rows_of(samples) { out.push(self.decide(query)) }
        return column_mat(out)
    }
}

#: Build a support vector machine.
pub fn svm_create() -> SVM { return SVM() }

#: Fit a boundary that is a plane in the features, by gradient descent on the
#: logistic loss.
pub class LogisticRegression {
    pub var learning_rate: float
    pub var iterations: int
    pub var regularisation: float
    var weights: list[float]
    var positive: float
    var negative: float

    pub fn init(self) {
        self.learning_rate = 0.1
        self.iterations = 500
        self.regularisation = 1e-4
        self.weights = []
        self.positive = 1.0
        self.negative = 0.0
    }

    #: Fit the model.
    pub fn train(self, samples: Mat, responses: Mat) -> bool {
        let rows = rows_of(samples)
        let targets = column_of(responses)
        if rows.len() != targets.len() {
            throw ValueError("there has to be one response per sample")
        }
        var labels: list[float] = []
        for value in targets {
            if not labels.contains(value) { labels.push(value) }
        }
        labels.sort()
        if labels.len() != 2 { throw ValueError("this model separates exactly two classes") }
        self.negative = labels[0]
        self.positive = labels[1]

        let width = rows[0].len()
        self.weights = []
        for _index in 0..width + 1 { self.weights.push(0.0) }

        for _step in 0..self.iterations {
            var gradient: list[float] = []
            for _index in 0..width + 1 { gradient.push(0.0) }
            for index in 0..rows.len() {
                let wanted = if targets[index] == self.positive { 1.0 } else { 0.0 }
                let predicted = self.probability(rows[index])
                let error = predicted - wanted
                gradient[width] += error
                for column in 0..width { gradient[column] += error * rows[index][column] }
            }
            let scale = self.learning_rate / float(rows.len())
            for column in 0..width {
                #: The penalty pulls the weights back towards zero, which is
                #: what keeps a separable problem from running them off to
                #: infinity.
                self.weights[column] -= scale *
                    (gradient[column] + self.regularisation * self.weights[column])
            }
            self.weights[width] -= scale * gradient[width]
        }
        return true
    }

    fn probability(self, query: list[float]) -> float {
        var total = self.weights[self.weights.len() - 1]
        for index in 0..query.len() { total += self.weights[index] * query[index] }
        if total > 40.0 { return 1.0 }
        if total < -40.0 { return 0.0 }
        return 1.0 / (1.0 + math.exp(-total))
    }

    #: Which class each sample belongs to.
    pub fn predict(self, samples: Mat) -> Mat {
        if self.weights.len() == 0 { throw RuntimeError("this model has not been trained") }
        var out: list[float] = []
        for query in rows_of(samples) {
            out.push(if self.probability(query) >= 0.5 { self.positive } else { self.negative })
        }
        return column_mat(out)
    }

    #: How likely each sample is to be the positive class.
    pub fn predict_proba(self, samples: Mat) -> Mat {
        var out: list[float] = []
        for query in rows_of(samples) { out.push(self.probability(query)) }
        return column_mat(out)
    }
}

#: Build a logistic regression.
pub fn logistic_regression_create() -> LogisticRegression { return LogisticRegression() }

class TreeNode {
    pub var feature: int
    pub var threshold: float
    pub var value: float
    pub var left: int
    pub var right: int

    pub fn init(self, feature: int, threshold: float, value: float, left: int, right: int) {
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right
    }
}

#: A decision tree, split on one feature at a time.
pub class DTrees {
    pub var max_depth: int
    pub var min_sample_count: int
    pub var regression: bool
    var nodes: list[TreeNode]

    pub fn init(self) {
        self.max_depth = 8
        self.min_sample_count = 2
        self.regression = false
        self.nodes = []
    }

    #: Grow the tree.
    pub fn train(self, samples: Mat, responses: Mat) -> bool {
        let rows = rows_of(samples)
        let targets = column_of(responses)
        if rows.len() != targets.len() {
            throw ValueError("there has to be one response per sample")
        }
        self.nodes = []
        var indices: list[int] = []
        for index in 0..rows.len() { indices.push(index) }
        self.grow(rows, targets, indices, 0)
        return true
    }

    fn grow(
        self,
        rows: list[list[float]],
        targets: list[float],
        indices: list[int],
        depth: int
    ) -> int {
        let here = self.nodes.len()
        self.nodes.push(TreeNode(-1, 0.0, self.answer(targets, indices), -1, -1))
        if depth >= self.max_depth or indices.len() < self.min_sample_count * 2 { return here }
        if self.pure(targets, indices) { return here }

        let width = rows[0].len()
        var best_feature = -1
        var best_threshold = 0.0
        var best_score = self.impurity(targets, indices)
        if best_score <= 0.0 { return here }

        for feature in 0..width {
            var values: list[float] = []
            for index in indices { values.push(rows[index][feature]) }
            values.sort()
            for at in 1..values.len() {
                if values[at] == values[at - 1] { continue }
                let threshold = (values[at] + values[at - 1]) * 0.5
                var left: list[int] = []
                var right: list[int] = []
                for index in indices {
                    if rows[index][feature] <= threshold {
                        left.push(index)
                    } else {
                        right.push(index)
                    }
                }
                if left.len() == 0 or right.len() == 0 { continue }
                #: The split is scored by how much less mixed the two halves
                #: are than the whole, weighted by how many samples each holds.
                let score = (float(left.len()) * self.impurity(targets, left) + float(right.len()) * self.impurity(targets, right)) /
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
        self.nodes[here].feature = best_feature
        self.nodes[here].threshold = best_threshold
        self.nodes[here].left = self.grow(rows, targets, left, depth + 1)
        self.nodes[here].right = self.grow(rows, targets, right, depth + 1)
        return here
    }

    fn pure(self, targets: list[float], indices: list[int]) -> bool {
        if indices.len() == 0 { return true }
        let first = targets[indices[0]]
        for index in indices {
            if targets[index] != first { return false }
        }
        return true
    }

    fn impurity(self, targets: list[float], indices: list[int]) -> float {
        if indices.len() == 0 { return 0.0 }
        if self.regression {
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

    fn answer(self, targets: list[float], indices: list[int]) -> float {
        if indices.len() == 0 { return 0.0 }
        if self.regression {
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

    #: Walk one sample down the tree.
    pub fn predict_one(self, query: list[float]) -> float {
        if self.nodes.len() == 0 { throw RuntimeError("this model has not been trained") }
        var at = 0
        while self.nodes[at].feature >= 0 {
            let node = self.nodes[at]
            at = if query[node.feature] <= node.threshold { node.left } else { node.right }
        }
        return self.nodes[at].value
    }

    #: Walk every sample down the tree.
    pub fn predict(self, samples: Mat) -> Mat {
        var out: list[float] = []
        for query in rows_of(samples) { out.push(self.predict_one(query)) }
        return column_mat(out)
    }
}

#: Build a decision tree.
pub fn d_trees_create() -> DTrees { return DTrees() }

#: A forest of trees, each grown on a different sample of the data, voting.
pub class RTrees {
    pub var tree_count: int
    pub var max_depth: int
    pub var regression: bool
    var trees: list[DTrees]

    pub fn init(self) {
        self.tree_count = 20
        self.max_depth = 8
        self.regression = false
        self.trees = []
    }

    #: Grow the forest.
    pub fn train(self, samples: Mat, responses: Mat, seed: int = 0x2545F4914F6C) -> bool {
        let rows = rows_of(samples)
        let targets = column_of(responses)
        if rows.len() != targets.len() {
            throw ValueError("there has to be one response per sample")
        }
        self.trees = []
        var state = seed
        for _index in 0..self.tree_count {
            #: Each tree sees a bootstrap sample: the same size as the original
            #: but drawn with replacement, so the trees disagree enough for the
            #: vote to be worth taking.
            var picked_rows: list[float] = []
            var picked_targets: list[float] = []
            let width = rows[0].len()
            for _draw in 0..rows.len() {
                state = state *% 6364136223846793 +% 1442695040888963 & 0xFFFFFFFFFFFF
                let pick = (state >> 8) % rows.len()
                for column in 0..width { picked_rows.push(rows[pick][column]) }
                picked_targets.push(targets[pick])
            }
            let tree = DTrees()
            tree.max_depth = self.max_depth
            tree.regression = self.regression
            tree.train(
                Mat.from_list(rows.len(), width, CV_MAKETYPE(CV_32F, 1), picked_rows),
                column_mat(picked_targets)
            )
            self.trees.push(tree)
        }
        return true
    }

    #: Ask every tree and take the answer they mostly agree on.
    pub fn predict(self, samples: Mat) -> Mat {
        if self.trees.len() == 0 { throw RuntimeError("this model has not been trained") }
        var out: list[float] = []
        for query in rows_of(samples) {
            var answers: list[float] = []
            for tree in self.trees { answers.push(tree.predict_one(query)) }
            if self.regression {
                var total = 0.0
                for value in answers { total += value }
                out.push(total / float(answers.len()))
                continue
            }
            var labels: list[float] = []
            var counts: list[int] = []
            for value in answers {
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
            out.push(best)
        }
        return column_mat(out)
    }
}

#: Build a random forest.
pub fn r_trees_create() -> RTrees { return RTrees() }

#: A feed-forward network of the shape OpenCV's `ANN_MLP` takes.
pub class ANN_MLP {
    pub var layer_sizes: list[int]
    pub var learning_rate: float
    pub var iterations: int
    var weights: list[list[float]]
    var biases: list[list[float]]

    pub fn init(self) {
        self.layer_sizes = []
        self.learning_rate = 0.05
        self.iterations = 400
        self.weights = []
        self.biases = []
    }

    #: Say how wide each layer is, input and output included.
    pub fn set_layer_sizes(self, sizes: list[int]) -> void {
        if sizes.len() < 2 { throw ValueError("a network needs an input and an output layer") }
        self.layer_sizes = sizes
        self.weights = []
        self.biases = []
        var state = 0x9E3779B97F4A
        for layer in 1..sizes.len() {
            let fan_in = sizes[layer - 1]
            var matrix: list[float] = []
            #: Scaled so the signal neither grows nor dies as it goes through,
            #: which for a symmetric activation means one over the root of the
            #: fan-in.
            let spread = 1.0 / math.sqrt(float(fan_in))
            for _index in 0..fan_in * sizes[layer] {
                state = state *% 6364136223846793 +% 1442695040888963 & 0xFFFFFFFFFFFF
                matrix.push((float((state >> 8) % 2000) / 1000.0 - 1.0) * spread)
            }
            var offsets: list[float] = []
            for _index in 0..sizes[layer] { offsets.push(0.0) }
            self.weights.push(matrix)
            self.biases.push(offsets)
        }
    }

    fn forward(self, input: list[float]) -> list[list[float]] {
        var activations = [input]
        for layer in 0..self.weights.len() {
            let previous = activations[layer]
            let width = self.layer_sizes[layer + 1]
            let fan_in = self.layer_sizes[layer]
            var next: list[float] = []
            for unit in 0..width {
                var total = self.biases[layer][unit]
                for index in 0..fan_in {
                    total += previous[index] * self.weights[layer][index * width + unit]
                }
                next.push(math.tanh(total))
            }
            activations.push(next)
        }
        return activations
    }

    #: Fit the network by back-propagation.
    pub fn train(self, samples: Mat, responses: Mat) -> bool {
        if self.layer_sizes.len() == 0 { throw RuntimeError("set the layer sizes first") }
        let rows = rows_of(samples)
        let targets = rows_of(responses)
        if rows.len() != targets.len() {
            throw ValueError("there has to be one response per sample")
        }

        for _step in 0..self.iterations {
            for index in 0..rows.len() {
                let activations = self.forward(rows[index])
                let depth = self.weights.len()
                var deltas: list[list[float]] = []
                for _layer in 0..depth { deltas.push([]) }

                let output = activations[depth]
                var last: list[float] = []
                for unit in 0..output.len() {
                    let error = output[unit] - targets[index][unit]
                    last.push(error * (1.0 - output[unit] * output[unit]))
                }
                deltas[depth - 1] = last

                var layer = depth - 1
                while layer > 0 {
                    let width = self.layer_sizes[layer]
                    let outer = self.layer_sizes[layer + 1]
                    var here: list[float] = []
                    for unit in 0..width {
                        var total = 0.0
                        for out_unit in 0..outer {
                            total += deltas[layer][out_unit] *
                                self.weights[layer][unit * outer + out_unit]
                        }
                        let value = activations[layer][unit]
                        here.push(total * (1.0 - value * value))
                    }
                    deltas[layer - 1] = here
                    layer -= 1
                }

                for at in 0..depth {
                    let fan_in = self.layer_sizes[at]
                    let width = self.layer_sizes[at + 1]
                    for unit in 0..width {
                        let gradient = deltas[at][unit]
                        self.biases[at][unit] -= self.learning_rate * gradient
                        for feature in 0..fan_in {
                            self.weights[at][feature * width + unit] -= self.learning_rate *
                                gradient *
                                activations[at][feature]
                        }
                    }
                }
            }
        }
        return true
    }

    #: Run every sample through the network.
    pub fn predict(self, samples: Mat) -> Mat {
        if self.weights.len() == 0 { throw RuntimeError("this model has not been trained") }
        let width = self.layer_sizes[self.layer_sizes.len() - 1]
        var out: list[float] = []
        for query in rows_of(samples) {
            let activations = self.forward(query)
            for value in activations[activations.len() - 1] { out.push(value) }
        }
        return Mat.from_list(out.len() // width, width, CV_MAKETYPE(CV_32F, 1), out)
    }
}

#: Build a feed-forward network.
pub fn ann_mlp_create() -> ANN_MLP { return ANN_MLP() }
