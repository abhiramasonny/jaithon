#: The classifiers and regressors OpenCV's `ml` module offers.
#:
#: Every model here takes the same shape of data OpenCV's does — one sample per
#: row, one feature per column, and responses as a column — and every one has
#: `train` and `predict`. The factory functions are named the way OpenCV names
#: them, so `k_nearest_create()` stands where `cv2.ml.KNearest_create()` would.
#:
#: Most of the work here is on the host, and is made affordable by the
#: algorithm rather than the processor: trees sort each feature once and sweep
#: it, and the support vector machine is libsvm's solver with its gradient
#: kept up to date. Logistic regression's gradient over a large training set
#: is a pair of matrix products, so that one goes to the device. Nearest
#: neighbours still measures every query against every sample on the host:
#: 1,000 queries against 10,000 samples of fifty features take about 1.4 s.
#:
#: ```
#: let model = svm_create()
#: model.train(samples, responses)
#: let predicted = model.predict(unseen)
#: ```
import std.math
import jaitensor as jt
from jaitensor import Tensor
from std.os import env_or
from ..core.linalg import device_view, inverse, matmul, symmetric_eigen, transpose
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
            #: Only the `neighbours` nearest are ever read, so only that many
            #: are kept. Sorting the whole training set for each query -- which
            #: is what inserting every sample amounts to -- makes the work grow
            #: with the square of the set and then throws all but three of them
            #: away: two thousand samples took 3.7 seconds that way.
            #:
            #: A sample no nearer than the worst one held cannot displace it,
            #: and one exactly as near would sort after it, so both are passed
            #: over -- which is where the full sort would have left them too.
            var order: list[int] = []
            var gaps: list[float] = []
            for index in 0..self.samples.len() {
                var total = 0.0
                for at in 0..query.len() {
                    let difference = query[at] - self.samples[index][at]
                    total += difference * difference
                }
                if order.len() >= neighbours and total >= gaps[order.len() - 1] { continue }
                var slot = order.len()
                while slot > 0 and total < gaps[slot - 1] { slot -= 1 }
                order.insert(slot, index)
                gaps.insert(slot, total)
                if order.len() > neighbours {
                    order.pop()
                    gaps.pop()
                }
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
            #: The spread of the class as well as how often it occurs.
            #:
            #: A Gaussian's log density is `log prior - 0.5 log|C| - 0.5 d`,
            #: and only the last two terms were here. Without the middle one
            #: a wide class always looks likelier than a narrow one, because
            #: distance to its centre counts for less and nothing pays for
            #: the volume that buys: two classes on the same mean, one spread
            #: a hundred times wider, and every point but the mean itself was
            #: given to the wide one. The covariance is symmetric and has a
            #: ridge added, so its eigenvalues are positive and their logs
            #: sum to the log determinant.
            let (spread, _axes) = symmetric_eigen(matrix)
            var log_volume = 0.0
            for value in spread { log_volume += math.log(max(value, 1e-30)) }
            let prior = math.log(float(members.len()) / float(rows.len()))
            self.offsets.push(prior - 0.5 * log_volume)
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
                let precision = self.inverses[index].values()
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
#: Trained the way libsvm -- and so OpenCV, whose solver is libsvm's -- trains
#: one: sequential minimal optimisation of the dual, two multipliers at a
#: time, each pair solved in closed form. The pair is chosen by second-order
#: working-set selection: the multiplier that most violates the optimality
#: conditions, and the partner whose step would lower the objective most. The
#: gradient of the objective is kept up to date as the multipliers move, so
#: choosing a pair costs one pass over the samples rather than a decision
#: function per sample, and the kernel's rows are computed once when they are
#: first needed and kept while memory allows.
#:
#: The machine it replaced paired each sample with the next one along and
#: recomputed every decision from scratch: 500 samples took 21 seconds and
#: 2000 did not finish, and it stopped wherever its pass count ran out rather
#: than at an optimum.
pub class SVM {
    pub var kernel_type: int
    pub var c: float
    pub var gamma: float
    pub var degree: float
    pub var coef0: float
    #: The solver stops after `iterations` steps per sample if it has not met
    #: `tolerance` first.
    pub var iterations: int
    #: How far from optimal the solution may be: libsvm's `eps`.
    pub var tolerance: float
    var samples: list[list[float]]
    var targets: list[float]
    var alphas: list[float]
    var support: list[int]
    var bias: float
    var positive: float
    var negative: float
    var trained: bool

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
        self.support = []
        self.bias = 0.0
        self.positive = 1.0
        self.negative = 0.0
        self.trained = false
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
        let count = rows.len()
        let width = rows[0].len()
        var flat: list[float] = []
        for row in rows {
            for value in row { flat.push(value) }
        }
        let q = _SvmRows(
            self.kernel_type,
            self.gamma,
            self.degree,
            self.coef0,
            flat,
            count,
            width,
            signs
        )

        var alpha: list[float] = []
        var gradient: list[float] = []
        for _index in 0..count {
            alpha.push(0.0)
            gradient.push(-1.0)
        }
        let limit = max(10000, self.iterations * count)
        for _step in 0..limit {
            let i = _svm_first(signs, alpha, gradient, self.c)
            if i < 0 { break }
            let j = _svm_partner(
                q.row(i),
                q.diagonal,
                signs,
                alpha,
                gradient,
                self.c,
                i,
                self.tolerance
            )
            if j < 0 { break }
            _svm_step(q.row(i), q.row(j), q.diagonal, signs, alpha, gradient, self.c, i, j)
        }

        let rho = _svm_rho(signs, alpha, gradient, self.c)
        self.bias = -rho
        self.samples = []
        self.targets = []
        self.alphas = []
        self.support = []
        for index in 0..count {
            if alpha[index] <= 0.0 { continue }
            self.samples.push(rows[index])
            self.targets.push(signs[index])
            self.alphas.push(alpha[index])
            self.support.push(index)
        }
        self.trained = true
        return true
    }

    #: The support vectors, one per row -- OpenCV's `getSupportVectors`.
    pub fn get_support_vectors(self) -> Mat {
        var out: list[float] = []
        for row in self.samples {
            for value in row { out.push(value) }
        }
        let width = if self.samples.len() > 0 { self.samples[0].len() } else { 0 }
        return Mat.from_list(self.samples.len(), width, CV_MAKETYPE(CV_32F, 1), out)
    }

    #: OpenCV's `getDecisionFunction`: the offset `rho`, each support vector's
    #: signed weight, and which training sample each support vector was. The
    #: decision for `x` is the sum of `weight * kernel(sv, x)` less `rho`.
    pub fn get_decision_function(self) -> tuple[float, list[float], list[int]] {
        var weights: list[float] = []
        for index in 0..self.alphas.len() { weights.push(self.alphas[index] * self.targets[index]) }
        return (-self.bias, weights, self.support)
    }

    #: Which side of the boundary each sample falls on.
    pub fn predict(self, samples: Mat) -> Mat {
        if not self.trained { throw RuntimeError("this model has not been trained") }
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

#: How many entries of the signed kernel matrix training keeps at once.
let SVM_ROW_BUDGET = 1 << 23

#: Rows of the signed kernel matrix `Q[i][j] = y_i y_j K(x_i, x_j)`, each
#: computed the first time it is asked for and kept while they fit in
#: `SVM_ROW_BUDGET` entries; past that the oldest kept row is dropped.
class _SvmRows {
    pub var kernel_type: int
    pub var gamma: float
    pub var degree: float
    pub var coef0: float
    pub var flat: list[float]
    pub var count: int
    pub var width: int
    pub var signs: list[float]
    pub var rows: list[list[float]]
    pub var present: list[bool]
    pub var missing: list[float]
    pub var kept: list[int]
    pub var oldest: int
    pub var capacity: int
    pub var diagonal: list[float]

    pub fn init(
        self,
        kernel_type: int,
        gamma: float,
        degree: float,
        coef0: float,
        flat: list[float],
        count: int,
        width: int,
        signs: list[float]
    ) {
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.flat = flat
        self.count = count
        self.width = width
        self.signs = signs
        #: A row not kept holds a one-entry stand-in rather than nothing, so
        #: every element of `rows` is a list of floats the compiled code can
        #: read the kind of; `present` says which are real.
        self.missing = [0.0]
        self.rows = []
        self.present = []
        for _index in 0..count {
            self.rows.push(self.missing)
            self.present.push(false)
        }
        self.kept = []
        self.oldest = 0
        self.capacity = max(2, SVM_ROW_BUDGET // max(count, 1))
        self.diagonal = []
        for index in 0..count { self.diagonal.push(_svm_kernel_at(self, index, index)) }
    }

    pub fn row(self, i: int) -> list[float] {
        if self.present[i] { return self.rows[i] }
        if self.kept.len() - self.oldest >= self.capacity {
            let dropped = self.kept[self.oldest]
            self.rows[dropped] = self.missing
            self.present[dropped] = false
            self.oldest += 1
        }
        var out: list[float] = []
        _svm_fill_row(self, i, out)
        self.rows[i] = out
        self.present[i] = true
        self.kept.push(i)
        return out
    }
}

fn _svm_kernel_at(q: _SvmRows, i: int, j: int) -> float {
    let flat = q.flat
    let width = q.width
    let a = i * width
    let b = j * width
    var total = 0.0
    if q.kernel_type == SVM_LINEAR or q.kernel_type == SVM_POLY {
        for index in 0..width { total += flat[a + index] * flat[b + index] }
        if q.kernel_type == SVM_LINEAR { return total }
        return (q.gamma * total + q.coef0) ** q.degree
    }
    for index in 0..width {
        let gap = flat[a + index] - flat[b + index]
        total += gap * gap
    }
    return math.exp(-q.gamma * total)
}

fn _svm_fill_row(q: _SvmRows, i: int, out: list[float]) -> void {
    let signs = q.signs
    let sign = signs[i]
    if q.kernel_type != SVM_RBF {
        for j in 0..q.count { out.push(sign * signs[j] * _svm_kernel_at(q, i, j)) }
        return
    }
    _svm_rbf_row(q.flat, q.width, q.count, signs, q.present, -q.gamma, i, out)
    #: The matrix is symmetric, and a kept row already holds the entries
    #: skipped above, worked out from the same two samples the same way.
    let rows = q.rows
    let kept = q.kept
    for at in q.oldest..kept.len() {
        let j = kept[at]
        out[j] = rows[j][i]
    }
}

fn _svm_rbf_row(
    flat: list[float],
    width: int,
    count: int,
    signs: list[float],
    present: list[bool],
    scale: float,
    i: int,
    out: list[float]
) -> void {
    let sign = signs[i]
    let a = i * width
    var total = 0.0
    var gap = 0.0
    for j in 0..count {
        if present[j] {
            out.push(0.0)
            continue
        }
        let b = j * width
        total = 0.0
        for index in 0..width {
            gap = flat[a + index] - flat[b + index]
            total += gap * gap
        }
        out.push(sign * signs[j] * math.exp(scale * total))
    }
}

#: libsvm's second-order working-set selection, first half: the multiplier
#: that most violates the optimality conditions, or -1 if none can move.
fn _svm_first(signs: list[float], alpha: list[float], gradient: list[float], c: float) -> int {
    var largest = -1e300
    var pick = -1
    for t in 0..signs.len() {
        if signs[t] > 0.0 {
            if alpha[t] < c and -gradient[t] >= largest {
                largest = -gradient[t]
                pick = t
            }
        } elif alpha[t] > 0.0 and gradient[t] >= largest {
            largest = gradient[t]
            pick = t
        }
    }
    return pick
}

#: The second half: the partner for `pick` whose step lowers the objective
#: most, given `row`, the signed kernel row of `pick` -- or -1 once the
#: largest violation of the optimality conditions is under `tolerance`.
fn _svm_partner(
    row: list[float],
    diagonal: list[float],
    signs: list[float],
    alpha: list[float],
    gradient: list[float],
    c: float,
    pick: int,
    tolerance: float
) -> int {
    let largest = -signs[pick] * gradient[pick]
    let own = diagonal[pick]
    let sign = signs[pick]
    var second = -1e300
    var partner = -1
    var least = 1e300
    var difference = 0.0
    var curvature = 0.0
    var gain = 0.0
    for t in 0..signs.len() {
        if signs[t] > 0.0 {
            if alpha[t] <= 0.0 { continue }
            difference = largest + gradient[t]
            if gradient[t] >= second { second = gradient[t] }
            curvature = own + diagonal[t] - 2.0 * sign * row[t]
        } else {
            if alpha[t] >= c { continue }
            difference = largest - gradient[t]
            if -gradient[t] >= second { second = -gradient[t] }
            curvature = own + diagonal[t] + 2.0 * sign * row[t]
        }
        if difference <= 0.0 { continue }
        if curvature <= 0.0 { curvature = 1e-12 }
        gain = -(difference * difference) / curvature
        if gain <= least {
            least = gain
            partner = t
        }
    }
    if largest + second < tolerance { return -1 }
    return partner
}

#: Move the pair `i`, `j` to the optimum along the constraint, clipped to the
#: box, and carry the change into the gradient.
fn _svm_step(
    row_i: list[float],
    row_j: list[float],
    diagonal: list[float],
    signs: list[float],
    alpha: list[float],
    gradient: list[float],
    c: float,
    i: int,
    j: int
) -> void {
    let old_i = alpha[i]
    let old_j = alpha[j]
    if signs[i] != signs[j] {
        var curvature = diagonal[i] + diagonal[j] + 2.0 * row_i[j]
        if curvature <= 0.0 { curvature = 1e-12 }
        let delta = (-gradient[i] - gradient[j]) / curvature
        let difference = old_i - old_j
        alpha[i] = old_i + delta
        alpha[j] = old_j + delta
        if difference > 0.0 {
            if alpha[j] < 0.0 {
                alpha[j] = 0.0
                alpha[i] = difference
            }
        } elif alpha[i] < 0.0 {
            alpha[i] = 0.0
            alpha[j] = -difference
        }
        if difference > 0.0 {
            if alpha[i] > c {
                alpha[i] = c
                alpha[j] = c - difference
            }
        } elif alpha[j] > c {
            alpha[j] = c
            alpha[i] = c + difference
        }
    } else {
        var curvature = diagonal[i] + diagonal[j] - 2.0 * row_i[j]
        if curvature <= 0.0 { curvature = 1e-12 }
        let delta = (gradient[i] - gradient[j]) / curvature
        let total = old_i + old_j
        alpha[i] = old_i - delta
        alpha[j] = old_j + delta
        if total > c {
            if alpha[i] > c {
                alpha[i] = c
                alpha[j] = total - c
            }
        } elif alpha[j] < 0.0 {
            alpha[j] = 0.0
            alpha[i] = total
        }
        if total > c {
            if alpha[j] > c {
                alpha[j] = c
                alpha[i] = total - c
            }
        } elif alpha[i] < 0.0 {
            alpha[i] = 0.0
            alpha[j] = total
        }
    }
    _svm_carry(gradient, row_i, alpha[i] - old_i, row_j, alpha[j] - old_j)
}

fn _svm_carry(
    gradient: list[float],
    row_i: list[float],
    moved_i: float,
    row_j: list[float],
    moved_j: float
) -> void {
    for k in 0..gradient.len() { gradient[k] += row_i[k] * moved_i + row_j[k] * moved_j }
}

#: The offset: the average over the multipliers strictly inside the box,
#: or the middle of the range the bounded ones allow when there are none.
fn _svm_rho(signs: list[float], alpha: list[float], gradient: list[float], c: float) -> float {
    var upper = 1e300
    var lower = -1e300
    var free = 0
    var total = 0.0
    for index in 0..signs.len() {
        let value = signs[index] * gradient[index]
        if alpha[index] >= c {
            if signs[index] < 0.0 { upper = min(upper, value) } else { lower = max(lower, value) }
        } elif alpha[index] <= 0.0 {
            if signs[index] > 0.0 { upper = min(upper, value) } else { lower = max(lower, value) }
        } else {
            free += 1
            total += value
        }
    }
    if free > 0 { return total / float(free) }
    return (upper + lower) * 0.5
}

#: Build a support vector machine.
pub fn svm_create() -> SVM { return SVM() }

#: A training set with at least this many values, counting the intercept's
#: column, has its gradient formed on the device.
pub let LOGISTIC_DEVICE_VALUES = 1 << 16

#: `JAICV_LOGISTIC_DEVICE=0` forms every gradient on the host.
let _LOGISTIC_DEVICE = env_or("JAICV_LOGISTIC_DEVICE", "1") != "0"

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
    #:
    #: Full-batch gradient descent, the same steps whichever processor forms
    #: the gradient. Over a large training set the gradient goes to the
    #: device: it is the samples times the weights, a sigmoid, and the
    #: samples' transpose times the errors -- two matrix-vector products --
    #: where the host loop took 1.7 s for 10,000 samples of fifty features and
    #: 35 s for 100,000. The weights stay on the host, in double precision,
    #: and are updated there exactly as before; only the gradient is float32.
    pub fn train(self, samples: Mat, responses: Mat) -> bool {
        let targets = column_of(responses)
        if samples.rows != targets.len() {
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

        let width = samples.cols * samples.channels()
        self.weights = []
        for _index in 0..width + 1 { self.weights.push(0.0) }
        if _LOGISTIC_DEVICE and samples.rows * (width + 1) >= LOGISTIC_DEVICE_VALUES {
            if self.descend_on_device(samples, targets, width) { return true }
        }

        let rows = rows_of(samples)
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
            self.step(gradient, rows.len())
        }
        return true
    }

    #: Move the weights down `gradient`, summed over `count` samples.
    fn step(self, gradient: list[float], count: int) -> void {
        let width = self.weights.len() - 1
        let scale = self.learning_rate / float(count)
        for column in 0..width {
            #: The penalty pulls the weights back towards zero, which is
            #: what keeps a separable problem from running them off to
            #: infinity.
            self.weights[column] -= scale *
                (gradient[column] + self.regularisation * self.weights[column])
        }
        self.weights[width] -= scale * gradient[width]
    }

    #: Descend with the gradient formed on the device, or return false when
    #: the samples are not a plain run of floats there.
    #:
    #: A column of ones is appended to the samples once, so the intercept is
    #: one more weight and its gradient comes out of the same product.
    fn descend_on_device(self, samples: Mat, targets: list[float], width: int) -> bool {
        let x = device_view(samples)
        if x is null { return false }
        let count = samples.rows
        let ones = Tensor.full([count, 1], 1.0)
        let augmented = jt.cat([x, ones], axis: -1)
        x.free()
        ones.free()
        var wanted: list[float] = []
        for value in targets { wanted.push(if value == self.positive { 1.0 } else { 0.0 }) }
        let truth = Tensor([count, 1], values: wanted)
        for _step in 0..self.iterations {
            let weights = Tensor([width + 1, 1], values: self.weights)
            let scores = jt.matmul(augmented, weights)
            weights.free()
            let predicted = jt.sigmoid(scores)
            scores.free()
            let errors = jt.subtract(predicted, truth)
            predicted.free()
            let summed = jt.matmul(augmented, errors, trans_a: true)
            errors.free()
            let gradient = summed.download()
            summed.free()
            self.step(gradient, count)
        }
        augmented.free()
        truth.free()
        return true
    }

    #: The fitted parameters as one row, intercept first: the layout
    #: OpenCV's `getLearntThetas` uses, since it puts the column of ones in
    #: front of the samples.
    pub fn get_learnt_thetas(self) -> Mat {
        if self.weights.len() == 0 { throw RuntimeError("this model has not been trained") }
        let width = self.weights.len() - 1
        var out = [self.weights[width]]
        for column in 0..width { out.push(self.weights[column]) }
        return Mat.from_list(1, width + 1, CV_MAKETYPE(CV_32F, 1), out)
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

#: What every node of one tree's growth shares: the samples by row and by
#: column, the responses, each response's class (when classifying) and a
#: per-sample flag the split writes before it divides the sorted orders.
class _TreeData {
    pub var rows: list[list[float]]
    pub var targets: list[float]
    pub var columns: list[list[float]]
    pub var classes: list[int]
    pub var class_count: int
    pub var side: list[bool]

    pub fn init(
        self,
        rows: list[list[float]],
        targets: list[float],
        columns: list[list[float]],
        classes: list[int],
        class_count: int,
        side: list[bool]
    ) {
        self.rows = rows
        self.targets = targets
        self.columns = columns
        self.classes = classes
        self.class_count = class_count
        self.side = side
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
    #:
    #: Each feature is sorted once, here, and every node after that keeps its
    #: samples in that order by splitting the parent's lists, so finding a
    #: split is one sweep per feature with the class counts (or the sums)
    #: carried along it. Trying every threshold by partitioning the node and
    #: scoring both halves from scratch was the square of the node's size per
    #: feature: ten thousand samples of twenty features took 134 seconds to
    #: grow eight levels.
    pub fn train(self, samples: Mat, responses: Mat) -> bool {
        let rows = rows_of(samples)
        let targets = column_of(responses)
        if rows.len() != targets.len() {
            throw ValueError("there has to be one response per sample")
        }
        self.nodes = []
        var indices: list[int] = []
        for index in 0..rows.len() { indices.push(index) }
        if rows.len() == 0 {
            self.nodes.push(TreeNode(-1, 0.0, self.answer(targets, indices), -1, -1))
            return true
        }
        let width = rows[0].len()
        var columns: list[list[float]] = []
        var orders: list[list[int]] = []
        for feature in 0..width {
            var column: list[float] = []
            for row in rows { column.push(row[feature]) }
            var order: list[int] = []
            for index in 0..rows.len() { order.push(index) }
            order.sort(key: |at| column[at])
            columns.push(column)
            orders.push(order)
        }
        var labels: list[float] = []
        var classes: list[int] = []
        if not self.regression {
            for value in targets {
                if not labels.contains(value) { labels.push(value) }
            }
            labels.sort()
            for value in targets {
                var slot = 0
                while labels[slot] != value { slot += 1 }
                classes.push(slot)
            }
        }
        var side: list[bool] = []
        for _index in 0..rows.len() { side.push(false) }
        let tree = _TreeData(rows, targets, columns, classes, labels.len(), side)
        self.grow(tree, indices, orders, 0)
        return true
    }

    #: The tree as OpenCV's `getNodes` gives it: each node's split feature (or
    #: -1 for a leaf), threshold, value and children, in the order they were
    #: grown.
    pub fn get_nodes(self) -> list[tuple[int, float, float, int, int]] {
        var out: list[tuple[int, float, float, int, int]] = []
        for node in self.nodes {
            out.push((node.feature, node.threshold, node.value, node.left, node.right))
        }
        return out
    }

    #: Grow the node holding `indices` (in sample order), with `orders[f]` the
    #: same samples sorted by feature `f`.
    #:
    #: The split chosen is exactly the one scoring every threshold of every
    #: feature from scratch would choose: the lowest score, the first feature
    #: and then the first threshold among equal ones, and only if it beats the
    #: node as it is. The sweep's running totals round differently from
    #: scoring each half on its own, so the sweep only shortlists: every
    #: threshold within a hair of its best is scored again the old way, in the
    #: old order, and the comparison is made on those scores. A tie between
    #: features -- two columns splitting the samples identically -- is common
    #: in small integer data, and is broken the way it always was.
    fn grow(self, tree: _TreeData, indices: list[int], orders: list[list[int]], depth: int) -> int {
        let targets = tree.targets
        let here = self.nodes.len()
        self.nodes.push(TreeNode(-1, 0.0, self.answer(targets, indices), -1, -1))
        if depth >= self.max_depth or indices.len() < self.min_sample_count * 2 { return here }
        if self.pure(targets, indices) { return here }

        let parent = self.impurity(targets, indices)
        if parent <= 0.0 { return here }

        let width = tree.columns.len()
        let count = indices.len()
        let slack = 1e-8 * parent
        var near_feature: list[int] = []
        var near_threshold: list[float] = []
        var near_score: list[float] = []
        var forced: list[bool] = []
        var best_seen = parent

        if self.regression {
            var origin = 0.0
            for index in indices { origin += targets[index] }
            origin /= float(count)
            var all_sum = 0.0
            var all_squares = 0.0
            for index in indices {
                let gap = targets[index] - origin
                all_sum += gap
                all_squares += gap * gap
            }
            for feature in 0..width {
                let column = tree.columns[feature]
                let order = orders[feature]
                var left_sum = 0.0
                var left_squares = 0.0
                for at in 0..count - 1 {
                    let index = order[at]
                    let gap = targets[index] - origin
                    left_sum += gap
                    left_squares += gap * gap
                    let value = column[index]
                    let next = column[order[at + 1]]
                    if next == value { continue }
                    let threshold = (value + next) * 0.5
                    if threshold >= next {
                        near_feature.push(feature)
                        near_threshold.push(threshold)
                        near_score.push(best_seen)
                        forced.push(true)
                        continue
                    }
                    let on_left = float(at + 1)
                    let on_right = float(count - at - 1)
                    let right_sum = all_sum - left_sum
                    let left_error = max(left_squares - left_sum * left_sum / on_left, 0.0)
                    let right_error = max(
                        all_squares -
                            left_squares -
                            right_sum * right_sum / on_right,
                        0.0
                    )
                    let score = (left_error + right_error) / float(count)
                    if score > best_seen + slack { continue }
                    if score < best_seen { best_seen = score }
                    near_feature.push(feature)
                    near_threshold.push(threshold)
                    near_score.push(score)
                    forced.push(false)
                }
            }
        } else {
            let classes = tree.classes
            var totals: list[int] = []
            for _slot in 0..tree.class_count { totals.push(0) }
            for index in indices { totals[classes[index]] += 1 }
            var total_squares = 0
            for value in totals { total_squares += value * value }
            var left: list[int] = []
            var right: list[int] = []
            for _slot in 0..tree.class_count {
                left.push(0)
                right.push(0)
            }
            for feature in 0..width {
                let column = tree.columns[feature]
                let order = orders[feature]
                for slot in 0..tree.class_count {
                    left[slot] = 0
                    right[slot] = totals[slot]
                }
                var left_squares = 0
                var right_squares = total_squares
                for at in 0..count - 1 {
                    let index = order[at]
                    let label_slot = classes[index]
                    left_squares += 2 * left[label_slot] + 1
                    left[label_slot] += 1
                    right_squares -= 2 * right[label_slot] - 1
                    right[label_slot] -= 1
                    let value = column[index]
                    let next = column[order[at + 1]]
                    if next == value { continue }
                    let threshold = (value + next) * 0.5
                    if threshold >= next {
                        near_feature.push(feature)
                        near_threshold.push(threshold)
                        near_score.push(best_seen)
                        forced.push(true)
                        continue
                    }
                    let on_left = float(at + 1)
                    let on_right = float(count - at - 1)
                    let left_gini = on_left - float(left_squares) / on_left
                    let right_gini = on_right - float(right_squares) / on_right
                    let score = (left_gini + right_gini) / float(count)
                    if score > best_seen + slack { continue }
                    if score < best_seen { best_seen = score }
                    near_feature.push(feature)
                    near_threshold.push(threshold)
                    near_score.push(score)
                    forced.push(false)
                }
            }
        }

        var best_feature = -1
        var best_threshold = 0.0
        var best_score = parent
        let rows = tree.rows
        for slot in 0..near_feature.len() {
            if not forced[slot] and near_score[slot] > best_seen + slack { continue }
            let feature = near_feature[slot]
            let threshold = near_threshold[slot]
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
                float(count)
            if score < best_score {
                best_score = score
                best_feature = feature
                best_threshold = threshold
            }
        }
        if best_feature < 0 { return here }

        let side = tree.side
        var left: list[int] = []
        var right: list[int] = []
        for index in indices {
            let goes_left = rows[index][best_feature] <= best_threshold
            side[index] = goes_left
            if goes_left {
                left.push(index)
            } else {
                right.push(index)
            }
        }
        var left_orders: list[list[int]] = []
        var right_orders: list[list[int]] = []
        for order in orders {
            var to_left: list[int] = []
            var to_right: list[int] = []
            for index in order {
                if side[index] {
                    to_left.push(index)
                } else {
                    to_right.push(index)
                }
            }
            left_orders.push(to_left)
            right_orders.push(to_right)
        }
        self.nodes[here].feature = best_feature
        self.nodes[here].threshold = best_threshold
        self.nodes[here].left = self.grow(tree, left, left_orders, depth + 1)
        self.nodes[here].right = self.grow(tree, right, right_orders, depth + 1)
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
