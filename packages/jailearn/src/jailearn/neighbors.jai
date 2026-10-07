#: k-nearest-neighbour classification and regression, by brute force.
#:
#: scikit-learn's `algorithm="brute"`: every query against every training row,
#: as one GEMM per band of queries (`base.nearest_rows`) and a one-pass top-k
#: over each band. No tree is built, because on a GPU the brute search over a
#: hundred thousand rows takes milliseconds and a tree's walk does not
#: parallelise; scikit-learn reaches the same verdict for dense data of more
#: than a handful of features.
#:
#: Neighbours are ranked by squared euclidean distance with equal distances
#: broken by the smaller training index, and `weights="distance"` weighs each
#: by the inverse euclidean distance, a query that coincides with training
#: rows taking only those rows -- scikit-learn's rule for both.
#:
#: The distances are float32, where scikit-learn's brute force upcasts to
#: float64, so the search is arranged to lose as little as float32 must.
#: Rows of at most `DIRECT_WIDTH` features are compared term by term. Wider
#: ones go through the GEMM, whose expanded form `|a|^2 + |b|^2 - 2 a.b`
#: rounds at the scale of the squared norms: on data a thousand from the
#: origin it picked the wrong nearest row five times in six. So the index
#: centres the training rows on their column means once, at `fit`, and each
#: query batch on the same means; distance does not change under a shift, and
#: the norms shrink to the data's spread. The distances reported for the rows
#: picked are then recomputed term by term on the uncentred data
#: (`refine_nearest`), so a query that coincides with a training row is at
#: exactly zero from it.
#:
#: A query holding a NaN or an infinity has no neighbours: `kneighbors` pads
#: its row with index `n_samples` and distance infinity, `predict_proba` and
#: the regressor's `predict` give it NaN, and the classifier's `predict` a
#: NaN label. scikit-learn refuses such a query outright; refusing here would
#: cost a read back on every call.
from std.gpu import Kernel

from jaitensor import Tensor

from .base import DIRECT_WIDTH, Estimator, EstimatorKind, LabelIndex, Params, cached_kernel
from .base import check_fitted, check_matrix, check_pair, check_width, column_moments, exact_search
from .base import nearest_rows, param_int, param_str, params_of, refine_nearest, require_target
from .base.reduce import copy_into

let SOURCE = """#include <metal_stdlib>
using namespace metal;

/* A neighbour's weight: 1 under uniform weighting; under distance weighting
 * the inverse euclidean distance, unless the query sits on a training row,
 * when the rows at distance zero take weight one and the rest none. */
static inline float nb_weight(float squared, bool weighted, bool exact) {
    if (!weighted) return 1.0f;
    if (exact) return squared == 0.0f ? 1.0f : 0.0f;
    return 1.0f / sqrt(squared);
}

static inline bool nb_exact(device const float* squared, uint k, bool weighted) {
    if (!weighted) return false;
    for (uint j = 0u; j < k; ++j) if (squared[j] == 0.0f) return true;
    return false;
}

/* Whether the search padded any of a query's slots, which it does only for a
 * query holding a NaN or an infinity: such a query has no neighbours, and
 * its index slots must not be read as training rows. */
static inline bool nb_padded(device const float* at, uint k, uint points) {
    for (uint j = 0u; j < k; ++j) if (!(at[j] < float(points))) return true;
    return false;
}

/* `x` less the training column means, so the GEMM's expanded distance works
 * on the data's spread and not on its offset from the origin. */
kernel void nb_centre(device const float* x [[buffer(0)]],
                      device const float* mean [[buffer(1)]],
                      device float* out [[buffer(2)]],
                      constant uint& count [[buffer(3)]],
                      constant uint& width [[buffer(4)]],
                      uint id [[thread_position_in_grid]]) {
    if (id < count) out[id] = x[id] - mean[id % width];
}

/* The winning class code of each row of shares, the first on a tie, as
 * `row_argmax`; a row of NaN shares -- a query with no neighbours -- gets
 * `classes`, which `LabelIndex.decode` turns into a NaN label. */
kernel void nb_pick(device const float* proba [[buffer(0)]],
                    device float* codes [[buffer(1)]],
                    constant uint& queries [[buffer(2)]],
                    constant uint& classes [[buffer(3)]],
                    uint id [[thread_position_in_grid]]) {
    if (id >= queries) return;
    device const float* row = proba + ulong(id) * ulong(classes);
    uint bits = as_type<uint>(row[0]);
    if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0u) {
        codes[id] = float(classes);
        return;
    }
    float best = row[0];
    uint pick = 0u;
    for (uint c = 1u; c < classes; ++c) {
        if (row[c] > best) { best = row[c]; pick = c; }
    }
    codes[id] = float(pick);
}

/* Class shares among each query's neighbours, one thread a query. */
kernel void nb_vote(device const float* index [[buffer(0)]],
                    device const float* squared [[buffer(1)]],
                    device const float* codes [[buffer(2)]],
                    device float* proba [[buffer(3)]],
                    constant uint& queries [[buffer(4)]],
                    constant uint& k [[buffer(5)]],
                    constant uint& classes [[buffer(6)]],
                    constant uint& weighting [[buffer(7)]],
                    constant uint& points [[buffer(8)]],
                    uint id [[thread_position_in_grid]]) {
    if (id >= queries) return;
    device float* row = proba + ulong(id) * ulong(classes);
    device const float* near = squared + ulong(id) * ulong(k);
    device const float* at = index + ulong(id) * ulong(k);
    if (nb_padded(at, k, points)) {
        for (uint c = 0u; c < classes; ++c) row[c] = NAN;
        return;
    }
    for (uint c = 0u; c < classes; ++c) row[c] = 0.0f;
    bool weighted = weighting != 0u;
    bool exact = nb_exact(near, k, weighted);
    float total = 0.0f;
    for (uint j = 0u; j < k; ++j) {
        float w = nb_weight(near[j], weighted, exact);
        row[uint(codes[uint(at[j])] + 0.5f)] += w;
        total += w;
    }
    float scale = total > 0.0f ? 1.0f / total : 0.0f;
    for (uint c = 0u; c < classes; ++c) row[c] *= scale;
}

/* The (weighted) mean target of each query's neighbours. */
kernel void nb_mean(device const float* index [[buffer(0)]],
                    device const float* squared [[buffer(1)]],
                    device const float* targets [[buffer(2)]],
                    device float* out [[buffer(3)]],
                    constant uint& queries [[buffer(4)]],
                    constant uint& k [[buffer(5)]],
                    constant uint& weighting [[buffer(6)]],
                    constant uint& points [[buffer(7)]],
                    uint id [[thread_position_in_grid]]) {
    if (id >= queries) return;
    device const float* near = squared + ulong(id) * ulong(k);
    device const float* at = index + ulong(id) * ulong(k);
    if (nb_padded(at, k, points)) {
        out[id] = NAN;
        return;
    }
    bool weighted = weighting != 0u;
    bool exact = nb_exact(near, k, weighted);
    float total = 0.0f;
    float weight = 0.0f;
    for (uint j = 0u; j < k; ++j) {
        float w = nb_weight(near[j], weighted, exact);
        total += w * targets[uint(at[j])];
        weight += w;
    }
    out[id] = weight > 0.0f ? total / weight : 0.0f;
}

kernel void nb_sqrt(device float* values [[buffer(0)]],
                    constant uint& count [[buffer(1)]],
                    uint id [[thread_position_in_grid]]) {
    if (id < count) values[id] = sqrt(values[id]);
}
"""

fn neighbors_kernel(entry: str) -> Kernel { return cached_kernel(SOURCE, entry) }

fn weighting_code(weights: str) -> int {
    if weights == "uniform" { return 0 }
    if weights == "distance" { return 1 }
    throw ValueError(f"weights must be 'uniform' or 'distance', got '{weights}'")
}

fn copy_of(x: Tensor) -> Tensor {
    let out = Tensor(x.shape)
    copy_into(x, out, x.size)
    return out
}

#: `x` less the column means `mean`, as a new tensor.
fn centred(x: Tensor, mean: Tensor) -> Tensor {
    let out = Tensor(x.shape)
    neighbors_kernel("nb_centre").dispatch_async(
        [x.buffer(), mean.buffer(), out.buffer()],
        x.size,
        [x.size, x.shape[1]]
    )
    return out
}

#: The training rows both estimators search, and the search itself.
#:
#: Wider than `DIRECT_WIDTH`, it also keeps the rows centred on their column
#: means, which is what the GEMM search ranks (see the module comment).
class NeighborIndex {
    pub var rows: Tensor?
    var centre: Tensor?
    var shifted: Tensor?
    pub var width: int

    pub fn init(self) {
        self.rows = null
        self.centre = null
        self.shifted = null
        self.width = 0
    }

    pub fn fit(self, x: Tensor, k: int) -> void {
        check_matrix(x)
        if k <= 0 { throw ValueError(f"n_neighbors must be positive, got {k}") }
        if k > x.shape[0] {
            throw ValueError(f"n_neighbors {k} is more than the {x.shape[0]} samples fitted")
        }
        self.free()
        self.rows = copy_of(x)
        self.width = x.shape[1]
        if x.shape[1] > DIRECT_WIDTH and exact_search() {
            let (mean, variance) = column_moments(x)
            variance.free()
            self.centre = mean
            self.shifted = centred(x, mean)
        }
    }

    pub fn fitted(self) -> bool { return self.rows is not null }

    #: `(squared distances, indices)` of each query's `k` nearest rows.
    pub fn search(self, x: Tensor, k: int, what: str) -> tuple[Tensor, Tensor] {
        check_fitted(self.fitted(), what)
        check_width(x, self.width, what)
        let rows = self.rows
        if rows is null { throw RuntimeError(f"{what} lost its training rows") }
        if k <= 0 or k > rows.shape[0] {
            throw ValueError(f"n_neighbors {k} is outside 1..{rows.shape[0]}")
        }
        let mean = self.centre
        let shifted = self.shifted
        if mean is null or shifted is null { return nearest_rows(x, rows, k) }
        let queries = centred(x, mean)
        defer { queries.free() }
        let (squared, index) = nearest_rows(queries, shifted, k, exact: false)
        refine_nearest(x, rows, squared, index)
        return (squared, index)
    }

    #: Euclidean distances and indices, as `kneighbors` reports them.
    pub fn kneighbors(self, x: Tensor, k: int, what: str) -> tuple[Tensor, Tensor] {
        let (squared, index) = self.search(x, k, what)
        neighbors_kernel("nb_sqrt").dispatch_async([squared.buffer()], squared.size, [squared.size])
        return (squared, index)
    }

    pub fn points(self) -> int {
        let rows = self.rows
        return if rows is null { 0 } else { rows.shape[0] }
    }

    pub fn free(self) -> void {
        let rows = self.rows
        if rows is not null { rows.free() }
        let mean = self.centre
        if mean is not null { mean.free() }
        let shifted = self.shifted
        if shifted is not null { shifted.free() }
        self.rows = null
        self.centre = null
        self.shifted = null
    }
}

#: Classification by a vote of the `n_neighbors` nearest training rows.
pub class KNeighborsClassifier: Estimator {
    pub var n_neighbors: int
    pub var weights: str
    pub var classes_: list[float]
    var index: NeighborIndex
    var labels: LabelIndex?
    var codes: Tensor?

    pub fn init(self, n_neighbors: int = 5, weights: str = "uniform") {
        weighting_code(weights)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.classes_ = []
        self.index = NeighborIndex()
        self.labels = null
        self.codes = null
    }

    pub fn kind(self) -> EstimatorKind { return EstimatorKind.Classifier }

    pub fn fit(self, x: Tensor, y: Tensor? = null) -> void {
        let target = require_target(y, "KNeighborsClassifier")
        check_pair(x, target)
        weighting_code(self.weights)
        self.free()
        self.index.fit(x, self.n_neighbors)
        let (labels, codes) = LabelIndex.fit_codes(target)
        self.codes = codes
        self.classes_ = labels.classes
        self.labels = labels
    }

    #: `(distances, indices)`, both `[n_queries, k]`, nearest first; the
    #: distances are euclidean and the indices are training rows as floats.
    #: `n_neighbors` 0 means the estimator's own.
    pub fn kneighbors(self, x: Tensor, n_neighbors: int = 0) -> tuple[Tensor, Tensor] {
        let k = if n_neighbors > 0 { n_neighbors } else { self.n_neighbors }
        return self.index.kneighbors(x, k, "KNeighborsClassifier")
    }

    #: The share of each class among the neighbours, `[n_queries, n_classes]`
    #: in the order of `classes_`.
    pub fn predict_proba(self, x: Tensor) -> Tensor {
        let (squared, index) = self.index.search(x, self.n_neighbors, "KNeighborsClassifier")
        defer { squared.free() }
        defer { index.free() }
        let codes = self.codes
        if codes is null { throw RuntimeError("KNeighborsClassifier lost its labels") }
        let classes = self.classes_.len()
        let proba = Tensor([x.shape[0], classes])
        neighbors_kernel("nb_vote").dispatch_async(
            [index.buffer(), squared.buffer(), codes.buffer(), proba.buffer()],
            x.shape[0],
            [
                x.shape[0],
                self.n_neighbors,
                classes,
                weighting_code(self.weights),
                self.index.points(),
            ]
        )
        return proba
    }

    #: The most likely class, the first in `classes_` on a tie -- which is
    #: what scikit-learn's argmax over the same shares returns.
    pub fn predict(self, x: Tensor) -> Tensor {
        let proba = self.predict_proba(x)
        defer { proba.free() }
        let winners = Tensor([x.shape[0]])
        defer { winners.free() }
        neighbors_kernel("nb_pick").dispatch_async(
            [proba.buffer(), winners.buffer()],
            x.shape[0],
            [x.shape[0], self.classes_.len()]
        )
        let labels = self.labels
        if labels is null { throw RuntimeError("KNeighborsClassifier lost its labels") }
        return labels.decode(winners)
    }

    pub fn get_params(self) -> Params {
        return params_of([("n_neighbors", self.n_neighbors), ("weights", self.weights)])
    }

    pub fn set_params(self, params: Params) -> void {
        self.n_neighbors = param_int(params, "n_neighbors", self.n_neighbors)
        let weights = param_str(params, "weights", self.weights)
        weighting_code(weights)
        self.weights = weights
    }

    pub fn clone_unfitted(self) -> KNeighborsClassifier {
        return KNeighborsClassifier(self.n_neighbors, self.weights)
    }

    pub fn is_fitted(self) -> bool { return self.index.fitted() and self.labels is not null }

    pub fn feature_count(self) -> int { return self.index.width }

    pub fn free(self) -> void {
        self.index.free()
        let labels = self.labels
        if labels is not null { labels.free() }
        let codes = self.codes
        if codes is not null { codes.free() }
        self.labels = null
        self.codes = null
    }
}

#: Regression by the mean target of the `n_neighbors` nearest training rows.
pub class KNeighborsRegressor: Estimator {
    pub var n_neighbors: int
    pub var weights: str
    var index: NeighborIndex
    var targets: Tensor?

    pub fn init(self, n_neighbors: int = 5, weights: str = "uniform") {
        weighting_code(weights)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.index = NeighborIndex()
        self.targets = null
    }

    pub fn kind(self) -> EstimatorKind { return EstimatorKind.Regressor }

    pub fn fit(self, x: Tensor, y: Tensor? = null) -> void {
        let target = require_target(y, "KNeighborsRegressor")
        check_pair(x, target)
        weighting_code(self.weights)
        self.free()
        self.index.fit(x, self.n_neighbors)
        self.targets = copy_of(target)
    }

    #: As `KNeighborsClassifier.kneighbors`.
    pub fn kneighbors(self, x: Tensor, n_neighbors: int = 0) -> tuple[Tensor, Tensor] {
        let k = if n_neighbors > 0 { n_neighbors } else { self.n_neighbors }
        return self.index.kneighbors(x, k, "KNeighborsRegressor")
    }

    pub fn predict(self, x: Tensor) -> Tensor {
        let (squared, index) = self.index.search(x, self.n_neighbors, "KNeighborsRegressor")
        defer { squared.free() }
        defer { index.free() }
        let targets = self.targets
        if targets is null { throw RuntimeError("KNeighborsRegressor lost its targets") }
        let out = Tensor([x.shape[0]])
        neighbors_kernel("nb_mean").dispatch_async(
            [index.buffer(), squared.buffer(), targets.buffer(), out.buffer()],
            x.shape[0],
            [x.shape[0], self.n_neighbors, weighting_code(self.weights), self.index.points()]
        )
        return out
    }

    pub fn get_params(self) -> Params {
        return params_of([("n_neighbors", self.n_neighbors), ("weights", self.weights)])
    }

    pub fn set_params(self, params: Params) -> void {
        self.n_neighbors = param_int(params, "n_neighbors", self.n_neighbors)
        let weights = param_str(params, "weights", self.weights)
        weighting_code(weights)
        self.weights = weights
    }

    pub fn clone_unfitted(self) -> KNeighborsRegressor {
        return KNeighborsRegressor(self.n_neighbors, self.weights)
    }

    pub fn is_fitted(self) -> bool { return self.index.fitted() and self.targets is not null }

    pub fn feature_count(self) -> int { return self.index.width }

    pub fn free(self) -> void {
        self.index.free()
        let targets = self.targets
        if targets is not null { targets.free() }
        self.targets = null
    }
}
