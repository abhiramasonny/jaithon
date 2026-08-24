#: The vocabulary every jailearn estimator speaks, and the device primitives
#: they are built out of.
#:
#: One data convention holds everywhere in the package: `X` is a
#: `jaitensor.Tensor` of shape `[n_samples, n_features]`, `y` is `[n_samples]`,
#: both float32 and resident on the GPU, and class labels travel as float codes
#: rather than as integers or strings. `LabelIndex` maps whatever codes a
#: caller used onto `0..k-1` and back, and every classifier, `StratifiedKFold`
#: and the metrics go through it.
#:
#: The other rule this module exists to enforce is that no fit walks its
#: samples on the host, where a loop over more than a few thousand elements
#: costs seconds. Column statistics, sorting, distances, class totals, seeding
#: and the small dense solves are all kernels below, general enough that the
#: estimators compose them instead of writing their own.
#:
#: ```
#: import jailearn as jl
#: let x = jl.matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
#: defer { x.free() }
#: let (mean, variance) = jl.column_moments(x)
#: defer { mean.free() }
#: defer { variance.free() }
#: jl.to_values(mean)    # [2.0, 3.0]
#: ```
import jaitensor as jt
import std.math
from jaitensor import Tensor
from std.algo.sort import merge_sort
from std.gpu import Buffer, Kernel
from std.random import Random

#: A parameter dictionary, as `get_params` returns it and `set_params` takes
#: it. Read values back through `param_int` and friends, which accept an
#: integer where a float was meant rather than making every caller remember
#: which one they wrote.
pub type Params = dict[str, any]

#: How `model_selection` measures a fitted estimator: higher is better, always.
#: It takes one of these rather than importing `metrics`, so a search can run
#: against a score that does not exist yet.
pub type Scorer = fn(Estimator, Tensor, Tensor) -> float

#: What an estimator produces, which decides which optional methods it has to
#: implement and what `default_score` means for it.
pub enum EstimatorKind {
    #: `predict` returns class codes; `score` is accuracy.
    Classifier,
    #: `predict` returns real values; `score` is the coefficient of determination.
    Regressor,
    #: `transform` maps `X` to another matrix; `score` has no meaning.
    Transformer,
    #: `fit` assigns each sample a cluster and `predict` returns it.
    Clusterer,
    #: `score_samples` returns a log density; `score` has no meaning.
    Density,
}

#: Everything in jailearn that learns from data.
#:
#: The seven required methods are the ones no estimator can do without. The
#: rest are defaulted, and a default a subclass has no business supporting
#: throws `RuntimeError` rather than returning something plausible -- the
#: choice `jaitensor.Layer.static_forward` makes, for the same reason: a
#: regressor that quietly answered `predict_proba` would be found out much
#: later than one that refuses.
#:
#: An estimator owns device tensors, so it needs `free`; freeing twice is
#: harmless, and `defer { model.free() }` beside the constructor is the habit
#: every example in the package follows.
pub trait Estimator {
    #: What this estimator produces.
    fn kind(self) -> EstimatorKind

    #: Learn from `x`, and from `y` when the estimator is supervised.
    fn fit(self, x: Tensor, y: Tensor? = null) -> void

    #: The constructor arguments, under the names `set_params` accepts.
    fn get_params(self) -> Params

    #: Change constructor arguments. Fitted state is not carried over; a search
    #: sets parameters on an unfitted clone.
    fn set_params(self, params: Params) -> void

    #: An unfitted estimator with the same parameters and no device memory of
    #: its own. This is what cross-validation refits, once per fold.
    fn clone_unfitted(self) -> Estimator

    #: Whether `fit` has run.
    fn is_fitted(self) -> bool

    #: Release the device tensors this estimator owns.
    fn free(self) -> void

    #: Map `x` into this estimator's output space.
    fn transform(self, x: Tensor) -> Tensor {
        throw RuntimeError("this estimator does not transform")
    }

    #: Map a transformed matrix back, as far as the transform allows.
    fn inverse_transform(self, x: Tensor) -> Tensor {
        throw RuntimeError("this estimator does not invert its transform")
    }

    #: Fit and transform together. Estimators that already hold the answer at
    #: the end of `fit` -- anything with a factorisation in hand -- override it.
    fn fit_transform(self, x: Tensor, y: Tensor? = null) -> Tensor {
        self.fit(x, y)
        return self.transform(x)
    }

    #: One prediction per row of `x`, shape `[n_samples]`.
    fn predict(self, x: Tensor) -> Tensor {
        throw RuntimeError("this estimator does not predict")
    }

    #: Class probabilities, shape `[n_samples, n_classes]`.
    fn predict_proba(self, x: Tensor) -> Tensor {
        throw RuntimeError("this estimator does not produce probabilities")
    }

    #: Unnormalised scores: `[n_samples]` for two classes, `[n_samples,
    #: n_classes]` beyond.
    fn decision_function(self, x: Tensor) -> Tensor {
        throw RuntimeError("this estimator does not expose a decision function")
    }

    #: The width `fit` saw, or 0 when the estimator records none. `check_width`
    #: reads 0 as "unknown" and lets the call through.
    fn feature_count(self) -> int { return 0 }

    #: Accuracy for a classifier, the coefficient of determination for a
    #: regressor, a refusal for anything else.
    fn score(self, x: Tensor, y: Tensor) -> float { return default_score(self, x, y) }
}

#: The score `Estimator.score` falls back to.
#:
#: Accuracy compares class codes for exact equality, which is safe only because
#: a classifier's `predict` returns values `LabelIndex.decode` copied out of the
#: class table rather than values it computed.
pub fn default_score(estimator: Estimator, x: Tensor, y: Tensor) -> float {
    check_pair(x, y)
    let kind = estimator.kind()
    if kind != EstimatorKind.Classifier and kind != EstimatorKind.Regressor {
        throw RuntimeError(f"{kind} has no default score; pass a Scorer")
    }
    let predicted = estimator.predict(x)
    defer { predicted.free() }
    if predicted.size != y.size {
        throw ValueError(f"predict returned {predicted.size} values for {y.size} samples")
    }
    if kind == EstimatorKind.Classifier {
        return float(count_equal(predicted, y)) / float(y.size)
    }
    let residual = jt.subtract(predicted, y)
    let squared = jt.square(residual)
    let unexplained = sum_to_host(squared)
    residual.free()
    squared.free()
    #: The total sum of squares comes from `column_moments` and not from
    #: `sum(y*y) - sum(y)^2/n`, which loses most of its digits when the mean is
    #: large next to the spread. House prices in dollars are the usual way to
    #: watch an R^2 of 0.97 come out as 0.2.
    let column = y.view(0, [y.size, 1])
    let (mean, variance) = column_moments(column)
    let spread = to_values(variance)[0] * float(y.size)
    mean.free()
    variance.free()
    column.free()
    if spread <= 0.0 { return 0.0 }
    return 1.0 - unexplained / spread
}

let SOURCE = """#include <metal_stdlib>
using namespace metal;

constexpr constant uint GROUP = 256u;

/* Total `value` over a whole threadgroup, handing every thread the answer.
 *
 * Every column statistic here sums over the samples, the axis that is not
 * being written: one thread per column would leave a GPU running a whole
 * matrix's work on as many threads as the data has features. A threadgroup
 * strides the samples instead and reduces through this. Barriered at both
 * ends, so the same scratch carries the next reduction. */
static inline float group_total(
    float value,
    threadgroup float *lanes,
    uint lid,
    uint lane,
    uint warp
) {
    if (lid < 32u) lanes[lid] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    value = simd_sum(value);
    if (lane == 0u) lanes[warp] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint j = 0u; j < 32u; j++) total += lanes[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

/* The same shape for a bound. A thread that drew no elements passes the
 * identity, which never wins. */
static inline float group_bound(
    float value,
    threadgroup float *lanes,
    uint lid,
    uint lane,
    uint warp,
    bool largest
) {
    float empty = largest ? -INFINITY : INFINITY;
    if (lid < 32u) lanes[lid] = empty;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    value = largest ? simd_max(value) : simd_min(value);
    if (lane == 0u) lanes[warp] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float best = empty;
    for (uint j = 0u; j < 32u; j++) best = largest ? max(best, lanes[j]) : min(best, lanes[j]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return best;
}

/* Whether a float is a NaN, read out of its bits.
 *
 * `v != v` is the usual spelling and it does not work: the runtime compiles
 * every shader with the default `MTLCompileOptions`, whose fast math is on,
 * and that licenses the compiler to assume no operand is a NaN and fold the
 * comparison to false. Missing values in a design matrix arrive as NaN, so
 * `nan_mask` returned all zeros until this replaced it. */
static inline bool not_a_number(float v) {
    uint bits = as_type<uint>(v);
    return (bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0u;
}

/* A uniform draw in (0, 1) from two integers, so a kernel can be random with
 * no generator state carried between dispatches. splitmix64, because its
 * avalanche is good enough that neighbouring thread ids give uncorrelated
 * draws -- a plain xorshift of `id` does not. */
static inline float splitmix(uint a, uint b) {
    ulong state = ((ulong)a + 1ul) * 0x9E3779B97F4A7C15ul + (ulong)b * 0xBF58476D1CE4E5B9ul;
    state ^= state >> 30; state *= 0xBF58476D1CE4E5B9ul;
    state ^= state >> 27; state *= 0x94D049BB133111EBul;
    state ^= state >> 31;
    return (float((state >> 40) & 0xFFFFFFul) + 1.0f) * (1.0f / 16777217.0f);
}

/* A strided copy: `stride` 1 is a plain one, and `order + 1` over a square
 * matrix is its diagonal. */
kernel void base_copy(device const float* src [[buffer(0)]],
                      device float* dst [[buffer(1)]],
                      constant uint& count [[buffer(2)]],
                      constant uint& stride [[buffer(3)]],
                      uint id [[thread_position_in_grid]]) {
    if (id < count) dst[id] = src[id * stride];
}

kernel void base_gather_columns(device const float* x [[buffer(0)]],
                                device const float* picks [[buffer(1)]],
                                device float* out [[buffer(2)]],
                                constant uint& rows [[buffer(3)]],
                                constant uint& width [[buffer(4)]],
                                constant uint& kept [[buffer(5)]],
                                uint id [[thread_position_in_grid]]) {
    if (id >= rows * kept) return;
    uint row = id / kept;
    out[id] = x[row * width + uint(picks[id - row * kept])];
}

kernel void base_place_block(device const float* part [[buffer(0)]],
                             device float* out [[buffer(1)]],
                             constant uint& count [[buffer(2)]],
                             constant uint& part_width [[buffer(3)]],
                             constant uint& total_width [[buffer(4)]],
                             constant uint& offset [[buffer(5)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint row = id / part_width;
    out[row * total_width + offset + id - row * part_width] = part[id];
}

/* One threadgroup per column, striding the samples: mode 0 leaves the mean and
 * the variance, 1 the smallest and largest, 2 the mean over the entries
 * `mask` leaves at zero.
 *
 * The variance takes a second pass rather than `E[x^2] - E[x]^2`, which is one
 * pass and is what a first draft writes. On pixel bytes that puts the answer's
 * leading digits inside the cancellation, and a scaler then divides by a
 * variance good to two figures. The second pass rereads a matrix that is
 * already in cache. */
kernel void base_column_reduce(device const float* x [[buffer(0)]],
                               device const float* mask [[buffer(1)]],
                               device float* first [[buffer(2)]],
                               device float* second [[buffer(3)]],
                               constant uint& rows [[buffer(4)]],
                               constant uint& width [[buffer(5)]],
                               constant uint& mode [[buffer(6)]],
                               constant uint& ddof [[buffer(7)]],
                               uint column [[threadgroup_position_in_grid]],
                               uint lid [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup float lanes[32];
    if (column >= width) return;
    if (mode == 1u) {
        float low = INFINITY;
        float high = -INFINITY;
        for (uint row = lid; row < rows; row += GROUP) {
            float value = x[row * width + column];
            low = min(low, value);
            high = max(high, value);
        }
        float smallest = group_bound(low, lanes, lid, lane, warp, false);
        float largest = group_bound(high, lanes, lid, lane, warp, true);
        if (lid == 0u) { first[column] = smallest; second[column] = largest; }
        return;
    }
    float running = 0.0f;
    float seen = 0.0f;
    for (uint row = lid; row < rows; row += GROUP) {
        uint at = row * width + column;
        if (mode == 2u && mask[at] != 0.0f) continue;
        running += x[at];
        seen += 1.0f;
    }
    float total = group_total(running, lanes, lid, lane, warp);
    if (mode == 2u) {
        float count = group_total(seen, lanes, lid, lane, warp);
        /* A column with nothing left averages to zero and not to a NaN: an
         * imputer that returns NaN has not imputed anything. */
        if (lid == 0u) first[column] = count > 0.0f ? total / count : 0.0f;
        return;
    }
    float centre = total / float(max(rows, 1u));
    float spread = 0.0f;
    for (uint row = lid; row < rows; row += GROUP) {
        float gap = x[row * width + column] - centre;
        spread += gap * gap;
    }
    float squares = group_total(spread, lanes, lid, lane, warp);
    if (lid != 0u) return;
    first[column] = centre;
    second[column] = squares / float(max(rows - min(ddof, rows), 1u));
}

/* Every column copied into a power-of-two tall scratch, the pad set to
 * infinity so it sorts past the real values and never reaches an answer. */
kernel void base_pad_columns(device const float* x [[buffer(0)]],
                             device float* values [[buffer(1)]],
                             device float* order [[buffer(2)]],
                             constant uint& rows [[buffer(3)]],
                             constant uint& padded [[buffer(4)]],
                             constant uint& width [[buffer(5)]],
                             constant uint& keep_order [[buffer(6)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= padded * width) return;
    uint row = id / width;
    values[id] = row < rows ? x[id] : INFINITY;
    if (keep_order != 0u) order[id] = float(row);
}

/* One compare-exchange stage of a bitonic sort, every column at once.
 *
 * Bitonic because the network is fixed: the host walks log^2(padded) stages,
 * a couple of hundred dispatches for a million samples, and never touches the
 * data. The thread index is compacted onto the pairs that actually swap --
 * `i` is the slot with bit `step` clear -- so the dispatch is half a column
 * rather than a whole one with half its threads returning. Adjacent threads
 * take adjacent columns, which is what makes the strided reads coalesce. Ties
 * are not broken by row, so `argsort_columns` is not stable. */
kernel void base_bitonic_step(device float* values [[buffer(0)]],
                              device float* order [[buffer(1)]],
                              constant uint& padded [[buffer(2)]],
                              constant uint& width [[buffer(3)]],
                              constant uint& span [[buffer(4)]],
                              constant uint& step [[buffer(5)]],
                              constant uint& keep_order [[buffer(6)]],
                              uint id [[thread_position_in_grid]]) {
    if (id >= (padded >> 1) * width) return;
    uint column = id % width;
    uint slot = id / width;
    uint i = ((slot & ~(step - 1u)) << 1) | (slot & (step - 1u));
    uint here = i * width + column;
    uint there = (i | step) * width + column;
    float mine = values[here];
    float yours = values[there];
    if ((mine > yours) == ((i & span) == 0u)) {
        values[here] = yours;
        values[there] = mine;
        if (keep_order != 0u) {
            float tag = order[here];
            order[here] = order[there];
            order[there] = tag;
        }
    }
}

kernel void base_quantile_pick(device const float* sorted [[buffer(0)]],
                               device const float* fractions [[buffer(1)]],
                               device float* out [[buffer(2)]],
                               constant uint& rows [[buffer(3)]],
                               constant uint& width [[buffer(4)]],
                               constant uint& count [[buffer(5)]],
                               uint id [[thread_position_in_grid]]) {
    if (id >= count * width) return;
    uint which = id / width;
    uint column = id - which * width;
    float position = clamp(fractions[which], 0.0f, 1.0f) * float(rows - 1u);
    uint below = uint(floor(position));
    uint above = min(below + 1u, rows - 1u);
    out[id] = mix(sorted[below * width + column], sorted[above * width + column],
                  position - float(below));
}

/* `|a - b|^2 = |a|^2 + |b|^2 - 2 a.b`, applied in place to the product.
 *
 * Rounding drops a point's distance from itself a hair below zero, and a
 * square root of that is a NaN that spreads through the whole matrix, so the
 * clamp is not decoration. */
kernel void base_dist_finish(device float* out [[buffer(0)]],
                             device const float* left [[buffer(1)]],
                             device const float* right [[buffer(2)]],
                             constant uint& rows [[buffer(3)]],
                             constant uint& columns [[buffer(4)]],
                             constant uint& first [[buffer(5)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= rows * columns) return;
    uint row = id / columns;
    out[id] = max(left[first + row] + right[id - row * columns] - 2.0f * out[id], 0.0f);
}

kernel void base_gram_finish(device float* out [[buffer(0)]],
                             device const float* shape [[buffer(1)]],
                             constant uint& count [[buffer(2)]],
                             constant uint& kind [[buffer(3)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float value = out[id];
    if (kind == 1u) out[id] = exp(-shape[0] * value);
    else if (kind == 2u) out[id] = pow(shape[0] * value + shape[2], shape[1]);
    else if (kind == 3u) out[id] = tanh(shape[0] * value + shape[2]);
}

/* One thread per row, striding the columns.
 *
 * The reverse -- a threadgroup per row, which is what the column statistics do
 * -- is wrong here: these matrices are tall and narrow, `[n_samples,
 * n_classes]` or `[n_samples, n_clusters]`, so the rows alone fill the machine
 * and the columns would not fill a simdgroup. Ties go to the lowest column,
 * which is what every argmax in the package promises. */
kernel void base_row_extreme(device const float* m [[buffer(0)]],
                             device float* index [[buffer(1)]],
                             device float* value [[buffer(2)]],
                             constant uint& rows [[buffer(3)]],
                             constant uint& columns [[buffer(4)]],
                             constant uint& largest [[buffer(5)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= rows) return;
    float best = largest != 0u ? -INFINITY : INFINITY;
    uint pick = 0u;
    for (uint column = 0u; column < columns; ++column) {
        float here = m[id * columns + column];
        if (largest != 0u ? here > best : here < best) { best = here; pick = column; }
    }
    index[id] = float(pick);
    value[id] = best;
}

/* The `k` best entries of each row, in order.
 *
 * No per-thread scratch, because Metal will not size a thread array from a
 * kernel argument: each pass takes the best entry strictly below the previous
 * pick, comparing `(value, column)` as a pair so duplicates are handed out one
 * at a time instead of filling every slot. That is `k` passes over the row,
 * the right trade while `k` is the handful of neighbours a kNN asks for and
 * not a sort in disguise. */
kernel void base_row_topk(device const float* m [[buffer(0)]],
                          device float* values [[buffer(1)]],
                          device float* index [[buffer(2)]],
                          constant uint& rows [[buffer(3)]],
                          constant uint& columns [[buffer(4)]],
                          constant uint& k [[buffer(5)]],
                          constant uint& largest [[buffer(6)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= rows) return;
    float sign = largest != 0u ? 1.0f : -1.0f;
    float ceiling = 0.0f;
    uint ceiling_at = 0u;
    for (uint slot = 0u; slot < k; ++slot) {
        float best = -INFINITY;
        uint pick = columns;
        for (uint column = 0u; column < columns; ++column) {
            float here = sign * m[id * columns + column];
            if (slot != 0u && !(here < ceiling || (here == ceiling && column > ceiling_at))) {
                continue;
            }
            if (here > best || (here == best && column < pick)) { best = here; pick = column; }
        }
        values[id * k + slot] = sign * best;
        index[id * k + slot] = float(pick);
        ceiling = best;
        ceiling_at = pick;
    }
}

/* One total per class and column, a threadgroup striding the samples.
 *
 * `power` 0 counts each class and ignores `values`, 1 sums the column and 2
 * sums its square. Between them those are the sufficient statistics of every
 * prior, centroid and Gaussian in the package, so naive Bayes and LDA each fit
 * in three dispatches. */
kernel void base_class_moment(device const float* values [[buffer(0)]],
                              device const float* labels [[buffer(1)]],
                              device float* out [[buffer(2)]],
                              constant uint& rows [[buffer(3)]],
                              constant uint& width [[buffer(4)]],
                              constant uint& classes [[buffer(5)]],
                              constant uint& power [[buffer(6)]],
                              uint slot [[threadgroup_position_in_grid]],
                              uint lid [[thread_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup float lanes[32];
    if (slot >= classes * width) return;
    uint which = slot / width;
    uint column = slot - which * width;
    float running = 0.0f;
    for (uint row = lid; row < rows; row += GROUP) {
        if (uint(labels[row] + 0.5f) != which) continue;
        if (power == 0u) { running += 1.0f; continue; }
        float value = values[row * width + column];
        running += power == 1u ? value : value * value;
    }
    float total = group_total(running, lanes, lid, lane, warp);
    if (lid == 0u) out[slot] = total;
}

/* Rows added into the slots their indices name, with device atomics.
 *
 * Two samples landing in one slot arrive in an order nothing fixes, and float
 * addition is not associative, so the last bits of a total move between runs
 * on the same data. The same trade `jaitensor.gather_rows` documents: making
 * it exact means sorting the indices, which costs more than the
 * reproducibility is worth for totals that go on to be divided by thousands. */
kernel void base_scatter_add_rows(device const float* src [[buffer(0)]],
                                  device const float* index [[buffer(1)]],
                                  device float* dst [[buffer(2)]],
                                  constant uint& count [[buffer(3)]],
                                  constant uint& width [[buffer(4)]],
                                  constant uint& capacity [[buffer(5)]],
                                  uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint row = id / width;
    float asked = index[row];
    if (not_a_number(asked) || asked < 0.0f || uint(asked) >= capacity) return;
    atomic_fetch_add_explicit(
        (device atomic<float>*)(dst + uint(asked) * width + id - row * width),
        src[id],
        memory_order_relaxed
    );
}

kernel void base_affine_columns(device const float* x [[buffer(0)]],
                                device const float* scale [[buffer(1)]],
                                device const float* offset [[buffer(2)]],
                                device float* out [[buffer(3)]],
                                constant uint& count [[buffer(4)]],
                                constant uint& width [[buffer(5)]],
                                uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint column = id % width;
    out[id] = x[id] * scale[column] + offset[column];
}

kernel void base_mask_op(device const float* a [[buffer(0)]],
                         device const float* b [[buffer(1)]],
                         device float* out [[buffer(2)]],
                         constant uint& count [[buffer(3)]],
                         constant uint& op [[buffer(4)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    if (op == 0u) out[id] = a[id] == b[id] ? 1.0f : 0.0f;
    else out[id] = not_a_number(a[id]) ? 1.0f : 0.0f;
}

/* How far each sample is from one centre, kept against the best so far. */
kernel void base_seed_distance(device const float* x [[buffer(0)]],
                               device const float* centres [[buffer(1)]],
                               device float* nearest [[buffer(2)]],
                               constant uint& rows [[buffer(3)]],
                               constant uint& width [[buffer(4)]],
                               constant uint& centre [[buffer(5)]],
                               constant uint& replace [[buffer(6)]],
                               uint id [[thread_position_in_grid]]) {
    if (id >= rows) return;
    float total = 0.0f;
    for (uint column = 0u; column < width; ++column) {
        float gap = x[id * width + column] - centres[centre * width + column];
        total += gap * gap;
    }
    nearest[id] = replace != 0u ? total : min(nearest[id], total);
}

/* One key per sample, whose largest is a draw proportional to the distance.
 *
 * `-log(u) / w` is exponential with rate `w`, so the first of those to fire is
 * sample `i` with probability `w_i / sum(w)`: k-means++ exactly, without the
 * running total and the linear walk over the samples the textbook version does
 * on the host. That walk is what made jaicv's seeding two hundred of the two
 * hundred and fifty milliseconds a VGA frame took. The reciprocal is emitted
 * instead of the key itself so the winner is a maximum, which is the reduction
 * jaitensor already has tuned; a sample at no distance emits zero and so never
 * wins. */
kernel void base_seed_keys(device const float* nearest [[buffer(0)]],
                           device float* keys [[buffer(1)]],
                           constant uint& rows [[buffer(2)]],
                           constant uint& salt [[buffer(3)]],
                           constant uint& round [[buffer(4)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= rows) return;
    float weight = nearest[id];
    keys[id] = weight > 0.0f ? weight / -log(splitmix(id ^ (round * 0x9E3779B9u), salt)) : 0.0f;
}

/* Shifted by the row maximum, because a mixture's log densities run to the
 * hundreds and `exp` of one of those is an infinity long before the ratio the
 * caller wants is anything unusual. */
kernel void base_log_sum_exp_rows(device const float* m [[buffer(0)]],
                                  device float* out [[buffer(1)]],
                                  constant uint& rows [[buffer(2)]],
                                  constant uint& columns [[buffer(3)]],
                                  uint id [[thread_position_in_grid]]) {
    if (id >= rows) return;
    float peak = -INFINITY;
    for (uint column = 0u; column < columns; ++column) peak = max(peak, m[id * columns + column]);
    if (not_a_number(peak) || peak == INFINITY || peak == -INFINITY) {
        out[id] = peak;
        return;
    }
    float total = 0.0f;
    for (uint column = 0u; column < columns; ++column) total += exp(m[id * columns + column] - peak);
    out[id] = peak + log(total);
}

/* The distinct values of a sorted column, compacted with an atomic counter.
 *
 * A prefix sum would keep them in order and costs a two-level scan; the
 * counter costs four lines and returns the same set in an arbitrary one, which
 * the host sorts -- a list of at most `capacity` classes, small by
 * construction, where the samples behind it are not. */
kernel void base_unique_from_sorted(device const float* sorted [[buffer(0)]],
                                    device atomic_uint* counter [[buffer(1)]],
                                    device float* out [[buffer(2)]],
                                    constant uint& count [[buffer(3)]],
                                    constant uint& capacity [[buffer(4)]],
                                    uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    if (id > 0u && sorted[id] == sorted[id - 1u]) return;
    uint slot = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    if (slot < capacity) out[slot] = sorted[id];
}

kernel void base_counter_value(device const atomic_uint* counter [[buffer(0)]],
                               device float* out [[buffer(1)]],
                               uint id [[thread_position_in_grid]]) {
    if (id == 0u) out[0] = float(atomic_load_explicit(counter, memory_order_relaxed));
}

/* Class values to codes and back. The table is short enough that a linear scan
 * beats anything with a search in it. An unknown value encodes to -1 rather
 * than to a plausible class, and the host checks for it once. */
kernel void base_map_labels(device const float* input [[buffer(0)]],
                            device const float* classes [[buffer(1)]],
                            device float* out [[buffer(2)]],
                            constant uint& count [[buffer(3)]],
                            constant uint& classes_n [[buffer(4)]],
                            constant uint& decode [[buffer(5)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float value = input[id];
    if (decode != 0u) {
        uint code = uint(max(value, 0.0f) + 0.5f);
        out[id] = code < classes_n ? classes[code] : NAN;
        return;
    }
    float found = -1.0f;
    for (uint which = 0u; which < classes_n; ++which) {
        if (classes[which] == value) { found = float(which); break; }
    }
    out[id] = found;
}

/* One column of a Cholesky factor: `phase` 0 is the diagonal on one thread,
 * `phase` 1 the entries below it on one thread each. They cannot share a
 * dispatch because there is no barrier across a grid. A non-positive pivot
 * makes the root a NaN, deliberately -- the host reads the finished diagonal
 * back and turns it into an exception. */
kernel void base_chol_step(device const float* a [[buffer(0)]],
                           device float* l [[buffer(1)]],
                           constant uint& order [[buffer(2)]],
                           constant uint& column [[buffer(3)]],
                           constant uint& phase [[buffer(4)]],
                           uint id [[thread_position_in_grid]]) {
    if (phase == 0u) {
        if (id != 0u) return;
        float total = a[column * order + column];
        for (uint k = 0u; k < column; ++k) {
            float seen = l[column * order + k];
            total -= seen * seen;
        }
        l[column * order + column] = sqrt(total);
        return;
    }
    uint row = column + 1u + id;
    if (row >= order) return;
    float total = a[row * order + column];
    for (uint k = 0u; k < column; ++k) total -= l[row * order + k] * l[column * order + k];
    l[row * order + column] = total / l[column * order + column];
}

/* One row of a forward or backward substitution, every right-hand side at
 * once. Sequential in the rows by nature, so `order` dispatches; the matrices
 * this runs on are as wide as the features or the components, never as the
 * samples. */
kernel void base_tri_row(device const float* l [[buffer(0)]],
                         device float* x [[buffer(1)]],
                         constant uint& order [[buffer(2)]],
                         constant uint& rhs [[buffer(3)]],
                         constant uint& row [[buffer(4)]],
                         constant uint& upper [[buffer(5)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= rhs) return;
    float total = x[row * rhs + id];
    if (upper == 0u) {
        for (uint k = 0u; k < row; ++k) total -= l[row * order + k] * x[k * rhs + id];
    } else {
        for (uint k = row + 1u; k < order; ++k) total -= l[k * order + row] * x[k * rhs + id];
    }
    x[row * rhs + id] = total / l[row * order + row];
}

kernel void base_box_muller(device const float* first [[buffer(0)]],
                            device const float* second [[buffer(1)]],
                            device float* out [[buffer(2)]],
                            constant uint& count [[buffer(3)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    out[id] = sqrt(-2.0f * log(first[id])) * cos(6.28318530718f * second[id]);
}
"""

var _kernels: dict[str, Kernel] = {}

#: The compiled pipeline for one entry point, compiled once per process.
#:
#: The cache is keyed on the entry name alone and shared by the whole package,
#: so every module must prefix its entry points with its own short name --
#: `nb_log_density`, `svm_smo_gradient`, `tree_histogram`. Two modules that
#: both called one `assign` would silently run the same kernel.
pub fn cached_kernel(source: str, entry: str) -> Kernel {
    let existing = _kernels.get(entry, null)
    if existing is not null { return existing }
    let compiled = Kernel.compile(source, entry)
    _kernels[entry] = compiled
    return compiled
}

fn base_kernel(entry: str) -> Kernel { return cached_kernel(SOURCE, entry) }

#: A `[n_samples, n_features]` tensor from host rows.
#:
#: For hand-written data: examples, tests, the odd small table. Real data
#: should reach the device through `jaitensor`'s loaders, because this walks
#: every element on the host.
pub fn matrix(rows: list[list[float]]) -> Tensor {
    if rows.len() == 0 { throw ValueError("a design matrix needs at least one row") }
    let width = rows[0].len()
    if width == 0 { throw ValueError("a design matrix needs at least one feature") }
    var flat: list[float] = []
    for (index, row) in rows.enumerate() {
        if row.len() != width {
            throw ValueError(f"row {index} has {row.len()} features, expected {width}")
        }
        for value in row { flat.push(value) }
    }
    return Tensor([rows.len(), width], flat)
}

#: A `[n_samples]` tensor from host values, for targets and label vectors.
pub fn vector(values: list[float]) -> Tensor {
    if values.len() == 0 { throw ValueError("a target vector needs at least one value") }
    return Tensor([values.len()], values)
}

#: A matrix back as host rows, with the same warning as `matrix`.
pub fn to_rows(x: Tensor) -> list[list[float]] {
    check_matrix(x)
    let width = x.shape[1]
    let flat = x.download()
    var rows: list[list[float]] = []
    for row in 0..x.shape[0] {
        var out: list[float] = []
        for column in 0..width { out.push(flat[row * width + column]) }
        rows.push(out)
    }
    return rows
}

#: A tensor back as host values, whatever its shape.
pub fn to_values(v: Tensor) -> list[float] { return v.download() }

#: Column `j` of `x`, as a `[n_samples]` vector.
pub fn column(x: Tensor, j: int) -> Tensor {
    check_matrix(x)
    if j < 0 or j >= x.shape[1] {
        throw IndexError(f"column {j} is outside a {x.shape[1]}-feature matrix")
    }
    return gather_columns(x, [j], [x.shape[0]])
}

#: The named columns of `x`, in the order given, as `[n_samples, len(indices)]`.
pub fn select_columns(x: Tensor, indices: list[int]) -> Tensor {
    check_matrix(x)
    if indices.len() == 0 { throw ValueError("select_columns needs at least one column") }
    for index in indices {
        if index < 0 or index >= x.shape[1] {
            throw IndexError(f"column {index} is outside a {x.shape[1]}-feature matrix")
        }
    }
    return gather_columns(x, indices, [x.shape[0], indices.len()])
}

fn gather_columns(x: Tensor, indices: list[int], shape: list[int]) -> Tensor {
    let table = index_tensor(indices)
    let out = Tensor(shape)
    base_kernel("base_gather_columns").dispatch_async(
        [x.buffer(), table.buffer(), out.buffer()],
        out.size,
        [x.shape[0], x.shape[1], indices.len()]
    )
    table.free()
    return out
}

#: Matrices laid side by side, sharing a sample count.
pub fn hstack(parts: list[Tensor]) -> Tensor {
    if parts.len() == 0 { throw ValueError("hstack needs at least one matrix") }
    let rows = parts[0].shape[0]
    var total = 0
    for part in parts {
        check_matrix(part)
        if part.shape[0] != rows {
            throw ValueError(f"hstack got {part.shape[0]} rows against {rows}")
        }
        total += part.shape[1]
    }
    let out = Tensor([rows, total])
    var offset = 0
    for part in parts {
        base_kernel("base_place_block").dispatch_async(
            [part.buffer(), out.buffer()],
            part.size,
            [part.size, part.shape[1], total, offset]
        )
        offset += part.shape[1]
    }
    return out
}

#: The rows `indices` names, in that order.
#:
#: Folds, bootstraps and shuffles all come out of this, so the index vector
#: stays on the device rather than being rebuilt on the host once per fold.
pub fn take_rows(x: Tensor, indices: Tensor) -> Tensor {
    check_matrix(x)
    return jt.gather_rows(x, indices)
}

#: A `[count]` index vector from host integers, for index lists a caller builds
#: itself and that are short by construction.
pub fn index_tensor(indices: list[int]) -> Tensor {
    if indices.len() == 0 { throw ValueError("an index vector needs at least one index") }
    var values: list[float] = []
    for index in indices { values.push(float(index)) }
    return Tensor([values.len()], values)
}

#: Read an integer parameter, accepting the float a caller may have written.
pub fn param_int(params: Params, name: str, fallback: int) -> int {
    let value = params.get(name, null)
    if value is int { return value }
    if value is float { return int(value) }
    return fallback
}

#: Read a float parameter, accepting the integer a caller may have written --
#: `{"gamma": 1}` is the mistake everybody makes once.
pub fn param_float(params: Params, name: str, fallback: float) -> float {
    let value = params.get(name, null)
    if value is float { return value }
    if value is int { return float(value) }
    return fallback
}

#: Read a boolean parameter.
pub fn param_bool(params: Params, name: str, fallback: bool) -> bool {
    let value = params.get(name, null)
    if value is bool { return value }
    return fallback
}

#: Read a string parameter.
pub fn param_str(params: Params, name: str, fallback: str) -> str {
    let value = params.get(name, null)
    if value is str { return value }
    return fallback
}

#: A parameter dictionary from pairs, which reads better inside a search grid
#: than a dictionary literal in a list does.
pub fn params_of(pairs: list[tuple[str, any]]) -> Params {
    var out: Params = {}
    for (name, value) in pairs { out[name] = value }
    return out
}

#: Insist that `x` is a `[n_samples, n_features]` matrix with something in it.
pub fn check_matrix(x: Tensor, name: str = "X") -> void {
    if x.rank() != 2 { throw ValueError(f"{name} must be [n_samples, n_features], got {x.shape}") }
    if x.shape[0] == 0 or x.shape[1] == 0 { throw ValueError(f"{name} is empty, shape {x.shape}") }
}

#: Insist that `x` and `y` describe the same samples.
pub fn check_pair(x: Tensor, y: Tensor, name: str = "X") -> void {
    check_matrix(x, name)
    if y.rank() != 1 { throw ValueError(f"y must be [n_samples], got {y.shape}") }
    if y.shape[0] != x.shape[0] {
        throw ValueError(f"y has {y.shape[0]} values for {x.shape[0]} samples")
    }
}

#: Unwrap the target of a supervised `fit`, naming the estimator that needs it.
pub fn require_target(y: Tensor?, what: str) -> Tensor {
    let target = y
    if target is null { throw ValueError(f"{what} needs a target vector") }
    return target
}

#: Insist that `fit` has run before whatever is about to read its results.
pub fn check_fitted(fitted: bool, what: str) -> void {
    if not fitted { throw RuntimeError(f"{what} is not fitted; call fit first") }
}

#: Insist that `x` is as wide as `fit` saw. An `expected` of 0 means the
#: estimator records no width, and the call goes through.
pub fn check_width(x: Tensor, expected: int, what: str) -> void {
    check_matrix(x)
    if expected == 0 { return }
    if x.shape[1] != expected {
        throw ValueError(f"{what} was fitted on {expected} features, got {x.shape[1]}")
    }
}

fn column_reduce(x: Tensor, mask: Tensor, mode: int, ddof: int) -> tuple[Tensor, Tensor] {
    check_matrix(x)
    let width = x.shape[1]
    let first = Tensor([width])
    let second = Tensor([width])
    base_kernel("base_column_reduce").dispatch_async(
        [x.buffer(), mask.buffer(), first.buffer(), second.buffer()],
        width * 256,
        [x.shape[0], width, mode, max(ddof, 0)],
        group_size: 256
    )
    return (first, second)
}

#: The mean and variance of every column, as two `[n_features]` vectors.
#:
#: `ddof` 0 is the population variance a scaler wants, 1 the sample variance a
#: statistic wants.
pub fn column_moments(x: Tensor, ddof: int = 0) -> tuple[Tensor, Tensor] {
    return column_reduce(x, x, 0, ddof)
}

#: The smallest and largest value in every column.
pub fn column_min_max(x: Tensor) -> tuple[Tensor, Tensor] { return column_reduce(x, x, 1, 0) }

#: The mean of every column over the entries `mask` leaves at zero.
#:
#: `mask` is the missingness, so it pairs with `nan_mask` directly.
pub fn masked_column_mean(x: Tensor, mask: Tensor) -> Tensor {
    if not x.same_shape(mask) { throw ValueError(f"mask {mask.shape} does not match X {x.shape}") }
    let (mean, unused) = column_reduce(x, mask, 2, 0)
    unused.free()
    return mean
}

fn next_power_of_two(n: int) -> int {
    var size = 1
    while size < n { size *= 2 }
    return size
}

#: Every column sorted ascending in a power-of-two tall scratch. The first
#: `rows` of each column are the answer and the pad above them is `+inf`.
fn bitonic_columns(x: Tensor, keep_order: bool) -> tuple[Tensor, Tensor] {
    let rows = x.shape[0]
    let width = x.shape[1]
    let padded = next_power_of_two(rows)
    let values = Tensor([padded, width])
    let order = if keep_order { Tensor([padded, width]) } else { Tensor([1]) }
    let tag = if keep_order { 1 } else { 0 }
    base_kernel("base_pad_columns").dispatch_async(
        [x.buffer(), values.buffer(), order.buffer()],
        padded * width,
        [rows, padded, width, tag]
    )
    let stage = base_kernel("base_bitonic_step")
    let threads = padded // 2 * width
    var span = 2
    while span <= padded {
        var step = span // 2
        while step > 0 {
            stage.dispatch_async(
                [values.buffer(), order.buffer()],
                threads,
                [padded, width, span, step, tag]
            )
            step = step // 2
        }
        span *= 2
    }
    return (values, order)
}

fn copy_into(source: Tensor, destination: Tensor, count: int, stride: int = 1) -> void {
    base_kernel("base_copy").dispatch_async(
        [source.buffer(), destination.buffer()],
        count,
        [count, stride]
    )
}

#: Every column of `x` sorted ascending, independently of the others.
pub fn sort_columns(x: Tensor) -> Tensor {
    check_matrix(x)
    let (values, order) = bitonic_columns(x, false)
    let out = Tensor(x.shape)
    copy_into(values, out, out.size)
    values.free()
    order.free()
    return out
}

#: Every column sorted, and the row each sorted value came from.
#:
#: Returns `(sorted, origin)`, both `[n_samples, n_features]`: the sort has to
#: happen either way, so the values come back beside the permutation instead of
#: making the caller gather them again. Equal values are not kept in row order.
pub fn argsort_columns(x: Tensor) -> tuple[Tensor, Tensor] {
    check_matrix(x)
    let (values, order) = bitonic_columns(x, true)
    let sorted = Tensor(x.shape)
    let origin = Tensor(x.shape)
    copy_into(values, sorted, sorted.size)
    copy_into(order, origin, origin.size)
    values.free()
    order.free()
    return (sorted, origin)
}

#: The given quantiles of every column, as `[len(qs), n_features]`.
#:
#: Linear interpolation between the two nearest order statistics, which is what
#: numpy and sklearn do. `KBinsDiscretizer`'s edges and the trees' `BinMapper`
#: are both this call, which is why neither of those modules imports the other.
pub fn column_quantiles(x: Tensor, qs: list[float]) -> Tensor {
    check_matrix(x)
    if qs.len() == 0 { throw ValueError("column_quantiles needs at least one quantile") }
    let (values, order) = bitonic_columns(x, false)
    let fractions = Tensor([qs.len()], qs)
    let out = Tensor([qs.len(), x.shape[1]])
    #: Read straight out of the padded buffer with `rows` as the height: the
    #: real values occupy exactly the first `rows` of every column, so nothing
    #: has to be compacted first.
    base_kernel("base_quantile_pick").dispatch_async(
        [values.buffer(), fractions.buffer(), out.buffer()],
        out.size,
        [x.shape[0], x.shape[1], qs.len()]
    )
    values.free()
    order.free()
    fractions.free()
    return out
}

#: The squared length of every row. jaitensor's row reduction rather than a
#: kernel here: it puts a threadgroup on each row, which matters as soon as the
#: features outnumber a simdgroup, and it is already tuned.
fn row_sqnorm(x: Tensor) -> Tensor {
    let squared = jt.square(x)
    let out = jt.sum(squared, axis: -1)
    squared.free()
    return out
}

#: Squared euclidean distance from rows of `a` to every row of `b`.
#:
#: `|a|^2 + |b|^2 - 2 a.b`, so the cost is one GEMM and not the triple loop the
#: definition suggests. The product is written straight into the answer, so the
#: distance array exists once; `a.bT` into a temporary and a second pass over
#: it would hold two of them at the same moment, which is where a kNN over a
#: hundred thousand samples runs out of device memory.
#:
#: `first` and `rows` take one horizontal band instead of the whole matrix.
#: That is the form a neighbour search, DBSCAN or t-SNE wants -- `n^2` is never
#: materialised, the caller reduces each band before asking for the next --
#: while the default computes all of it.
pub fn pairwise_sqdist(a: Tensor, b: Tensor, first: int = 0, rows: int = 0) -> Tensor {
    check_matrix(a, "a")
    check_matrix(b, "b")
    if a.shape[1] != b.shape[1] {
        throw ValueError(f"a has {a.shape[1]} features against b's {b.shape[1]}")
    }
    let band = if rows > 0 { rows } else { a.shape[0] - first }
    if first < 0 or band <= 0 or first + band > a.shape[0] {
        throw IndexError(f"rows {first}..{first + band} run past a's {a.shape[0]} samples")
    }
    let out = Tensor([band, b.shape[0]])
    let block = a.view(first * a.shape[1], [band, a.shape[1]])
    let product = jt.matmul(block, b, trans_b: true, out: out)
    let left = row_sqnorm(a)
    let right = row_sqnorm(b)
    base_kernel("base_dist_finish").dispatch_async(
        [product.buffer(), left.buffer(), right.buffer()],
        out.size,
        [band, b.shape[0], first]
    )
    left.free()
    right.free()
    block.free()
    return out
}

#: `k(a_i, b_j) = a_i . b_j`.
pub const GRAM_LINEAR: int = 0
#: `k(a_i, b_j) = exp(-gamma |a_i - b_j|^2)`.
pub const GRAM_RBF: int = 1
#: `k(a_i, b_j) = (gamma a_i . b_j + coef0)^degree`.
pub const GRAM_POLY: int = 2
#: `k(a_i, b_j) = tanh(gamma a_i . b_j + coef0)`.
pub const GRAM_SIGMOID: int = 3

#: The kernel matrix between two sets of rows, `[n_a, n_b]`.
#:
#: Everything kernelised in the package -- the SVMs, kernel ridge, kernel PCA,
#: spectral clustering -- comes through here, so the four kernels are written
#: once and the estimators only pick a `kind`.
pub fn gram(
    a: Tensor,
    b: Tensor,
    kind: int = GRAM_LINEAR,
    gamma: float = 1.0,
    degree: float = 3.0,
    coef0: float = 0.0
) -> Tensor {
    if kind < GRAM_LINEAR or kind > GRAM_SIGMOID { throw ValueError(f"unknown gram kind {kind}") }
    if kind == GRAM_RBF {
        let distances = pairwise_sqdist(a, b)
        shape_gram(distances, kind, gamma, degree, coef0)
        return distances
    }
    check_matrix(a, "a")
    check_matrix(b, "b")
    if a.shape[1] != b.shape[1] {
        throw ValueError(f"a has {a.shape[1]} features against b's {b.shape[1]}")
    }
    let product = jt.matmul(a, b, trans_b: true)
    if kind != GRAM_LINEAR { shape_gram(product, kind, gamma, degree, coef0) }
    return product
}

fn shape_gram(m: Tensor, kind: int, gamma: float, degree: float, coef0: float) -> void {
    let shape = Buffer.of([gamma, degree, coef0])
    base_kernel("base_gram_finish").dispatch_async([m.buffer(), shape], m.size, [m.size, kind])
    shape.free()
}

#: The winning column of every row and the value that won it. Ties go to the
#: lowest column, matching every argmax in the package.
pub fn row_extreme(m: Tensor, largest: bool) -> tuple[Tensor, Tensor] {
    check_matrix(m, "m")
    let index = Tensor([m.shape[0]])
    let value = Tensor([m.shape[0]])
    let wanted = if largest { 1 } else { 0 }
    base_kernel("base_row_extreme").dispatch_async(
        [m.buffer(), index.buffer(), value.buffer()],
        m.shape[0],
        [m.shape[0], m.shape[1], wanted]
    )
    return (index, value)
}

#: The index of the largest entry in each row, as `[n_rows]` float codes.
pub fn row_argmax(m: Tensor) -> Tensor {
    let (index, value) = row_extreme(m, true)
    value.free()
    return index
}

#: The index of the smallest entry in each row.
pub fn row_argmin(m: Tensor) -> Tensor {
    let (index, value) = row_extreme(m, false)
    value.free()
    return index
}

#: The `k` best entries of every row: `(values, indices)`, both `[n_rows, k]`.
#: `largest` false picks the smallest, which is what a neighbour search wants
#: out of a distance matrix.
pub fn row_topk(m: Tensor, k: int, largest: bool = true) -> tuple[Tensor, Tensor] {
    check_matrix(m, "m")
    if k <= 0 or k > m.shape[1] { throw ValueError(f"cannot take {k} of {m.shape[1]} columns") }
    let values = Tensor([m.shape[0], k])
    let index = Tensor([m.shape[0], k])
    let wanted = if largest { 1 } else { 0 }
    base_kernel("base_row_topk").dispatch_async(
        [m.buffer(), values.buffer(), index.buffer()],
        m.shape[0],
        [m.shape[0], m.shape[1], k, wanted]
    )
    return (values, index)
}

#: `log(sum(exp(row)))` for every row, shifted by the row maximum.
pub fn log_sum_exp_rows(m: Tensor) -> Tensor {
    check_matrix(m, "m")
    let out = Tensor([m.shape[0]])
    base_kernel("base_log_sum_exp_rows").dispatch_async(
        [m.buffer(), out.buffer()],
        m.shape[0],
        [m.shape[0], m.shape[1]]
    )
    return out
}

fn class_moment(values: Tensor, labels: Tensor, width: int, classes: int, power: int) -> Tensor {
    if classes <= 0 { throw ValueError(f"classes must be positive, got {classes}") }
    let out = Tensor(if power == 0 { [classes] } else { [classes, width] })
    base_kernel("base_class_moment").dispatch_async(
        [values.buffer(), labels.buffer(), out.buffer()],
        classes * width * 256,
        [labels.size, width, classes, power],
        group_size: 256
    )
    return out
}

#: How many samples carry each code `0..classes-1`, as `[classes]`.
pub fn bincount(labels: Tensor, classes: int) -> Tensor {
    return class_moment(labels, labels, 1, classes, 0)
}

#: The column totals of each class, `[classes, n_features]`.
pub fn class_sums(x: Tensor, labels: Tensor, classes: int) -> Tensor {
    check_pair(x, labels)
    return class_moment(x, labels, x.shape[1], classes, 1)
}

#: The column totals of each class's squares, `[classes, n_features]`.
pub fn class_sumsq(x: Tensor, labels: Tensor, classes: int) -> Tensor {
    check_pair(x, labels)
    return class_moment(x, labels, x.shape[1], classes, 2)
}

#: Each row of `source` added into the row of the result its index names. The
#: order the additions land in is not fixed, so the last bits of a total vary
#: between runs on the same data.
pub fn scatter_add_rows(source: Tensor, indices: Tensor, rows: int) -> Tensor {
    check_matrix(source, "source")
    if indices.size != source.shape[0] {
        throw ValueError(f"{indices.size} indices for {source.shape[0]} rows")
    }
    if rows <= 0 { throw ValueError(f"destination rows must be positive, got {rows}") }
    let out = Tensor.zeros([rows, source.shape[1]])
    base_kernel("base_scatter_add_rows").dispatch_async(
        [source.buffer(), indices.buffer(), out.buffer()],
        source.size,
        [source.size, source.shape[1], rows]
    )
    return out
}

#: `x[i][j] * scale[j] + offset[j]`, which is every scaler's apply step.
pub fn affine_columns(x: Tensor, scale: Tensor, offset: Tensor) -> Tensor {
    check_matrix(x)
    let width = x.shape[1]
    if scale.size != width or offset.size != width {
        throw ValueError(
            f"scale and offset must be [{width}], got {scale.shape} and {offset.shape}"
        )
    }
    let out = Tensor(x.shape)
    base_kernel("base_affine_columns").dispatch_async(
        [x.buffer(), scale.buffer(), offset.buffer(), out.buffer()],
        x.size,
        [x.size, width]
    )
    return out
}

#: How many positions hold the same value in both tensors.
#:
#: Counted by summing a mask in float32, so the answer is exact up to about
#: sixteen million matches and then starts rounding. Accuracy over a test set
#: that size is not the measurement anybody is taking.
pub fn count_equal(a: Tensor, b: Tensor) -> int {
    if a.size != b.size { throw ValueError(f"{a.shape} and {b.shape} have different sizes") }
    let mask = Tensor([a.size])
    base_kernel("base_mask_op").dispatch_async(
        [a.buffer(), b.buffer(), mask.buffer()],
        a.size,
        [a.size, 0]
    )
    let total = sum_to_host(mask)
    mask.free()
    return int(total + 0.5)
}

#: 1 where `x` holds a NaN and 0 elsewhere, same shape as `x`.
pub fn nan_mask(x: Tensor) -> Tensor {
    let mask = Tensor(x.shape)
    base_kernel("base_mask_op").dispatch_async(
        [x.buffer(), x.buffer(), mask.buffer()],
        x.size,
        [x.size, 1]
    )
    return mask
}

fn sum_to_host(t: Tensor) -> float {
    let total = jt.sum(t)
    let value = total.download()[0]
    total.free()
    return value
}

#: `k` starting centres spread out by k-means++, as `[k, n_features]`.
#:
#: The first is uniform; each after it is drawn with probability proportional
#: to its squared distance from the nearest centre already chosen. Distances
#: and draw are both dispatches, so the host loop is over `k` and nothing else.
#: Clustering and the Gaussian mixtures both start here.
pub fn kmeanspp_seed(x: Tensor, k: int, seed: int = 0) -> Tensor {
    check_matrix(x)
    let rows = x.shape[0]
    let width = x.shape[1]
    if k <= 0 or k > rows { throw ValueError(f"{rows} samples cannot make {k} centres") }
    let centres = Tensor([k, width])
    let rng = rng_for(seed)
    copy_row(x, centres, rng.int_range(0, rows), 0, width)
    let nearest = Tensor([rows])
    let keys = Tensor([rows])
    let distance = base_kernel("base_seed_distance")
    let key_step = base_kernel("base_seed_keys")
    let salt = device_seed(seed, 0x5DEECE66)
    distance.dispatch_async(
        [x.buffer(), centres.buffer(), nearest.buffer()],
        rows,
        [rows, width, 0, 1]
    )
    for centre in 1..k {
        key_step.dispatch_async([nearest.buffer(), keys.buffer()], rows, [rows, salt, centre])
        let chosen = int(peak_index(keys) + 0.5)
        copy_row(x, centres, chosen, centre, width)
        distance.dispatch_async(
            [x.buffer(), centres.buffer(), nearest.buffer()],
            rows,
            [rows, width, centre, 0]
        )
    }
    nearest.free()
    keys.free()
    return centres
}

#: Where the largest value of a vector sits.
#:
#: Through jaitensor's row reduction, which puts a threadgroup on the vector:
#: `row_extreme` over a one-row matrix would put a single thread on it, and a
#: single thread walking a hundred thousand samples is the mistake this module
#: exists to keep out of the estimators.
fn peak_index(v: Tensor) -> float {
    let row = v.view(0, [1, v.size])
    let (values, indices) = jt.max_with_index(row)
    let at = indices.download()[0]
    values.free()
    indices.free()
    row.free()
    return at
}

fn copy_row(source: Tensor, destination: Tensor, taken: int, placed: int, width: int) -> void {
    let src = source.view(taken * width, [width])
    let dst = destination.view(placed * width, [width])
    copy_into(src, dst, width)
    src.free()
    dst.free()
}

#: The generator an estimator's `random_state` names.
#:
#: Seed 0 means unseeded, and it means that everywhere in the package: the
#: device fills below draw their own seed from a fresh generator when they get
#: a 0, rather than inheriting `Buffer.fill_uniform`'s habit of turning it into
#: a constant and handing back the same numbers every run.
pub fn rng_for(seed: int) -> Random {
    if seed == 0 { return Random() }
    return Random.with_seed(seed)
}

fn mix_seed(seed: int, salt: int) -> int {
    return seed *% 6364136223846793 +% salt *% 1442695040888963 & 0xFFFFFFFF
}

fn device_seed(seed: int, salt: int) -> int {
    return mix_seed(if seed == 0 { Random().next_int() } else { seed }, salt)
}

#: A tensor of uniform draws in `[lo, hi)`, filled on the device.
pub fn random_uniform(shape: list[int], lo: float = 0.0, hi: float = 1.0, seed: int = 0) -> Tensor {
    let out = Tensor(shape)
    out.buffer().fill_uniform(lo, hi, device_seed(seed, 0x27D4EB2F))
    return out
}

#: A tensor of standard normal draws, Box-Muller over two uniform fills.
#:
#: The first fill starts just above zero rather than at it: `log(0)` is an
#: infinity, and one of those in an initial weight matrix is a model that never
#: trains.
pub fn random_normal(shape: list[int], seed: int = 0) -> Tensor {
    let out = Tensor(shape)
    let first = Tensor(shape)
    let second = Tensor(shape)
    first.buffer().fill_uniform(1.0e-7, 1.0, device_seed(seed, 0x9E3779B9))
    second.buffer().fill_uniform(0.0, 1.0, device_seed(seed, 0x85EBCA6B))
    base_kernel("base_box_muller").dispatch_async(
        [first.buffer(), second.buffer(), out.buffer()],
        out.size,
        [out.size]
    )
    first.free()
    second.free()
    return out
}

#: The most distinct class values `distinct_values` will report.
pub const MAX_CLASSES: int = 4096

#: The distinct values of a vector, ascending.
#:
#: Sorted on the device and compacted with an atomic counter, so the host only
#: touches the answer -- at most `MAX_CLASSES` long -- and never the samples.
pub fn distinct_values(y: Tensor) -> list[float] {
    if y.size == 0 { throw ValueError("cannot take the distinct values of an empty vector") }
    let column = y.view(0, [y.size, 1])
    let sorted = sort_columns(column)
    column.free()
    let counter = Tensor.zeros([1])
    let found = Tensor([MAX_CLASSES])
    let total = Tensor([1])
    base_kernel("base_unique_from_sorted").dispatch_async(
        [sorted.buffer(), counter.buffer(), found.buffer()],
        y.size,
        [y.size, MAX_CLASSES]
    )
    base_kernel("base_counter_value").dispatch_async([counter.buffer(), total.buffer()], 1, [])
    let count = int(total.download()[0] + 0.5)
    let values = found.download()
    sorted.free()
    counter.free()
    found.free()
    total.free()
    if count > MAX_CLASSES {
        throw ValueError(f"{count} distinct values exceeds the {MAX_CLASSES} this supports")
    }
    var kept: list[float] = []
    for index in 0..count { kept.push(values[index]) }
    return merge_sort(kept)
}

#: Arbitrary class values mapped onto the codes `0..k-1` and back.
#:
#: Every classifier fits against codes, `StratifiedKFold` counts them and the
#: metrics compare them, so the mapping lives here once and each of those takes
#: a `LabelIndex` rather than deriving its own.
pub class LabelIndex {
    #: The class values, ascending. Position `i` is code `i`.
    pub let classes: list[float]
    let table: Tensor

    #: Build an index over class values that are already known.
    pub fn init(self, classes: list[float]) {
        if classes.len() == 0 { throw ValueError("a LabelIndex needs at least one class") }
        self.classes = [value for value in classes]
        self.table = Tensor([classes.len()], self.classes)
    }

    #: Build an index over whatever values a label vector holds.
    pub static fn from_labels(y: Tensor) -> LabelIndex {
        return LabelIndex(distinct_values(y))
    }

    #: How many classes there are.
    pub fn count(self) -> int { return self.classes.len() }

    #: The class values as a `[k]` device tensor, for kernels that need them.
    pub fn class_tensor(self) -> Tensor { return self.table }

    #: Class values to codes `0..k-1`.
    pub fn encode(self, y: Tensor) -> Tensor {
        let out = Tensor(y.shape)
        self._map(y, out, 0)
        #: A value in no class encodes to -1, and a silent -1 turns into a
        #: wrong class much further downstream, so it is caught here for the
        #: price of one pass and one scalar read. Negated and maximised rather
        #: than minimised because that is the reduction jaitensor spreads over
        #: threadgroups; the minimum would be one thread walking the samples.
        let flipped = jt.neg(out)
        let peak = jt.amax(flipped)
        let lowest = -peak.download()[0]
        flipped.free()
        peak.free()
        if lowest < 0.0 {
            out.free()
            throw ValueError(f"a label is outside the {self.count()} known classes")
        }
        return out
    }

    #: Codes `0..k-1` back to class values.
    pub fn decode(self, codes: Tensor) -> Tensor {
        let out = Tensor(codes.shape)
        self._map(codes, out, 1)
        return out
    }

    fn _map(self, input: Tensor, out: Tensor, decode: int) -> void {
        base_kernel("base_map_labels").dispatch_async(
            [input.buffer(), self.table.buffer(), out.buffer()],
            input.size,
            [input.size, self.count(), decode]
        )
    }

    #: Release the device copy of the class table. Freeing twice is harmless.
    pub fn free(self) -> void { self.table.free() }

    fn __str__(self) -> str { return f"LabelIndex(classes={self.classes})" }
}

#: The lower-triangular `L` with `L LT = a`.
#:
#: For the normal equations and for a full-covariance Gaussian, both square in
#: the features or the components and never in the samples. Column by column,
#: two dispatches each because there is no barrier across a grid; a matrix that
#: is not positive definite is caught when the diagonal comes back with a NaN.
pub fn cholesky_factor(a: Tensor) -> Tensor {
    let order = square_order(a)
    let l = Tensor.zeros([order, order])
    let step = base_kernel("base_chol_step")
    for column in 0..order {
        step.dispatch_async([a.buffer(), l.buffer()], 1, [order, column, 0])
        let below = order - column - 1
        if below > 0 { step.dispatch_async([a.buffer(), l.buffer()], below, [order, column, 1]) }
    }
    let pivots = diagonal(l)
    let values = pivots.download()
    pivots.free()
    for (index, value) in values.enumerate() {
        if math.is_nan(value) or value <= 0.0 {
            l.free()
            throw ValueError(f"the matrix is not positive definite; pivot {index} is {value}")
        }
    }
    return l
}

#: Solve `L x = b`, or `LT x = b` when `transposed`. `b` may be `[order]` or
#: `[order, n_rhs]`, and the answer keeps its shape.
pub fn triangular_solve(l: Tensor, b: Tensor, transposed: bool = false) -> Tensor {
    let order = square_order(l)
    if b.rank() < 1 or b.rank() > 2 {
        throw ValueError(f"b must be a vector or a matrix, got {b.shape}")
    }
    if b.shape[0] != order {
        throw ValueError(f"b has {b.shape[0]} rows against an order-{order} factor")
    }
    let rhs = if b.rank() == 1 { 1 } else { b.shape[1] }
    let x = Tensor(b.shape)
    copy_into(b, x, b.size)
    let solve = base_kernel("base_tri_row")
    let upper = if transposed { 1 } else { 0 }
    for step in 0..order {
        let row = if transposed { order - 1 - step } else { step }
        solve.dispatch_async([l.buffer(), x.buffer()], rhs, [order, rhs, row, upper])
    }
    return x
}

#: Solve `a x = b` for a symmetric positive definite `a`.
pub fn cholesky_solve(a: Tensor, b: Tensor) -> Tensor {
    let l = cholesky_factor(a)
    let forward = triangular_solve(l, b, false)
    let x = triangular_solve(l, forward, true)
    l.free()
    forward.free()
    return x
}

#: The diagonal of a square matrix, as `[order]`.
pub fn diagonal(m: Tensor) -> Tensor {
    let order = square_order(m)
    let out = Tensor([order])
    copy_into(m, out, order, stride: order + 1)
    return out
}

fn square_order(m: Tensor) -> int {
    if m.rank() != 2 or m.shape[0] != m.shape[1] {
        throw ValueError(f"expected a square matrix, got {m.shape}")
    }
    if m.shape[0] == 0 { throw ValueError("the matrix is empty") }
    return m.shape[0]
}
