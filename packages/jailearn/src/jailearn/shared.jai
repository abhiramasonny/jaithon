#: The second layer of jailearn: the machinery more than one estimator family
#: needs, so those families can be written without importing each other.
#:
#: `base` holds the protocol and the general device primitives. Here are the
#: pieces that are specific to jailearn and still shared -- binning, the
#: histogram every split search reduces to, the split rule and its scan, the
#: device form of a fitted tree, the samplers, the small linear helpers and one
#: convergence loop. Trees and boosting both want a histogram; a lasso and a
#: logistic regression both want a soft threshold and a convergence report.
#:
#: Nothing here is an `Estimator`. The families own their growth loops, their
#: factorisations and their parameters; they share the arithmetic underneath.
#:
#: ```
#: import jailearn as jl
#: let x = jl.matrix([[0.0], [1.0], [2.0], [3.0]])
#: defer { x.free() }
#: let edges = jl.bin_edges(x, 2, jl.BIN_UNIFORM)
#: defer { edges.free() }
#: let codes = jl.bin_codes(x, edges)
#: defer { codes.free() }
#: jl.to_values(codes)    # [0.0, 0.0, 1.0, 1.0]
#: ```
import jaitensor as jt
import std.math
from jaitensor import Tensor
from std.gpu import Buffer, Kernel
from .base import argsort_columns, cached_kernel, check_matrix, column_min_max
from .base import column_quantiles, rng_for, sort_columns

let SOURCE = """#include <metal_stdlib>
using namespace metal;

constexpr constant uint GROUP = 256u;

/* The widest histogram the split scan can hold in a threadgroup. Kept equal to
 * `MAX_BINS` on the host, which is where a caller's `n_bins` is checked. */
constexpr constant uint SCAN_WIDTH = 256u;

/* The identity of a maximum over gains. Not `-INFINITY`: see `sh_feature_best`
 * for why an infinity cannot be compared against in these shaders. */
constexpr constant float WORST = -3.4028235e38f;

/* `base.jai`'s two helpers again, because a Metal library is compiled per
 * source string and nothing crosses between them. The NaN test reads the bits
 * rather than writing `v != v`, which the default compile options fold to
 * false; the mixer is splitmix64, whose avalanche makes neighbouring thread
 * ids give uncorrelated draws. */
static inline bool not_a_number(float v) {
    uint bits = as_type<uint>(v);
    return (bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0u;
}

static inline float splitmix(uint a, uint b) {
    ulong state = ((ulong)a + 1ul) * 0x9E3779B97F4A7C15ul + (ulong)b * 0xBF58476D1CE4E5B9ul;
    state ^= state >> 30; state *= 0xBF58476D1CE4E5B9ul;
    state ^= state >> 27; state *= 0x94D049BB133111EBul;
    state ^= state >> 31;
    return (float((state >> 40) & 0xFFFFFFul) + 1.0f) * (1.0f / 16777217.0f);
}

/* The largest of a threadgroup's candidates and where it came from, ties to
 * the lowest slot, which is what every argmax in the package promises. */
static inline void group_argmax(threadgroup float *best, threadgroup uint *at, uint lid) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = GROUP >> 1; step > 0u; step >>= 1) {
        if (lid < step) {
            float other = best[lid + step];
            uint other_at = at[lid + step];
            if (other > best[lid] || (other == best[lid] && other_at < at[lid])) {
                best[lid] = other;
                at[lid] = other_at;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

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

kernel void sh_copy(device const float* src [[buffer(0)]],
                    device float* dst [[buffer(1)]],
                    constant uint& count [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {
    if (id < count) dst[id] = src[id];
}

/* Equally spaced cuts between a column's extremes.
 *
 * A constant column has every cut on top of the only value it holds, and
 * because `sh_bin_codes` counts the cuts strictly below a value, every sample
 * then lands in bin 0 and the histogram over that feature offers no split.
 * That is the right answer for a feature that carries nothing. */
kernel void sh_uniform_edges(device const float* low [[buffer(0)]],
                             device const float* high [[buffer(1)]],
                             device float* edges [[buffer(2)]],
                             constant uint& width [[buffer(3)]],
                             constant uint& cuts [[buffer(4)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= width * cuts) return;
    uint feature = id / cuts;
    uint cut = id - feature * cuts;
    float span = high[feature] - low[feature];
    edges[id] = low[feature] + span * float(cut + 1u) / float(cuts + 1u);
}

/* Which bin each element falls in, by a binary search over its feature's cuts.
 *
 * A search and not a walk: the cuts run to a couple of hundred and this is the
 * one kernel in the binning path that touches every element, so eight steps
 * against two hundred and fifty-six is a pass over the matrix against thirty.
 * A value on a cut goes to the bin below it -- the edges read as `(cut[b-1],
 * cut[b]]` -- and a NaN gets code -1, which every consumer here skips, so a
 * histogram can be built over a column an imputer has not reached yet. */
kernel void sh_bin_codes(device const float* x [[buffer(0)]],
                         device const float* edges [[buffer(1)]],
                         device float* codes [[buffer(2)]],
                         constant uint& count [[buffer(3)]],
                         constant uint& width [[buffer(4)]],
                         constant uint& cuts [[buffer(5)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float value = x[id];
    if (not_a_number(value)) { codes[id] = -1.0f; return; }
    device const float* row = edges + (id % width) * cuts;
    uint lo = 0u;
    uint hi = cuts;
    while (lo < hi) {
        uint mid = (lo + hi) >> 1;
        if (row[mid] < value) lo = mid + 1u; else hi = mid;
    }
    codes[id] = float(lo);
}

/* A representative value for every bin.
 *
 * The interior bins take the midpoint of the two cuts that bound them. The
 * outer two are open and have no midpoint, so they mirror the width of the bin
 * beside them; with a single cut there is no neighbouring width to mirror and
 * the half-unit is arbitrary, chosen only to keep the centres ordered. An
 * inverse transform that needs better than that should keep the column's
 * extremes, which this does not see. */
kernel void sh_bin_centres(device const float* edges [[buffer(0)]],
                           device float* centres [[buffer(1)]],
                           constant uint& width [[buffer(2)]],
                           constant uint& cuts [[buffer(3)]],
                           uint id [[thread_position_in_grid]]) {
    uint bins = cuts + 1u;
    if (id >= width * bins) return;
    uint feature = id / bins;
    uint bin = id - feature * bins;
    device const float* row = edges + feature * cuts;
    if (bin == 0u) {
        float step = cuts > 1u ? row[1] - row[0] : 1.0f;
        centres[id] = row[0] - 0.5f * step;
    } else if (bin == cuts) {
        float step = cuts > 1u ? row[cuts - 1u] - row[cuts - 2u] : 1.0f;
        centres[id] = row[cuts - 1u] + 0.5f * step;
    } else {
        centres[id] = 0.5f * (row[bin - 1u] + row[bin]);
    }
}

/* One Lloyd pass of the one-dimensional k-means behind `BIN_KMEANS`.
 *
 * Assignment and accumulation share a dispatch: which centre an element is
 * nearest is that thread's own business, and writing the answer out only to
 * read it back in the next kernel costs a whole extra pass over the matrix.
 * The atomics contend over `bins * width` slots -- a few hundred -- which is
 * the trade `base_scatter_add_rows` already takes, paid twenty times per fit
 * rather than once. */
kernel void sh_kmeans_totals(device const float* x [[buffer(0)]],
                             device const float* centres [[buffer(1)]],
                             device float* sums [[buffer(2)]],
                             device float* counts [[buffer(3)]],
                             constant uint& count [[buffer(4)]],
                             constant uint& width [[buffer(5)]],
                             constant uint& bins [[buffer(6)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float value = x[id];
    if (not_a_number(value)) return;
    uint feature = id % width;
    uint pick = 0u;
    float best = INFINITY;
    for (uint bin = 0u; bin < bins; ++bin) {
        float gap = fabs(value - centres[bin * width + feature]);
        if (gap < best) { best = gap; pick = bin; }
    }
    uint slot = pick * width + feature;
    atomic_fetch_add_explicit((device atomic<float>*)(sums + slot), value, memory_order_relaxed);
    atomic_fetch_add_explicit((device atomic<float>*)(counts + slot), 1.0f, memory_order_relaxed);
}

/* An empty cluster keeps the centre it had rather than collapsing to zero,
 * which would drag it across the column and reorder that feature's bins. */
kernel void sh_kmeans_update(device const float* sums [[buffer(0)]],
                             device const float* counts [[buffer(1)]],
                             device float* centres [[buffer(2)]],
                             constant uint& slots [[buffer(3)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= slots) return;
    float seen = counts[id];
    if (seen > 0.0f) centres[id] = sums[id] / seen;
}

kernel void sh_kmeans_edges(device const float* centres [[buffer(0)]],
                            device float* edges [[buffer(1)]],
                            constant uint& width [[buffer(2)]],
                            constant uint& cuts [[buffer(3)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= width * cuts) return;
    uint feature = id / cuts;
    uint cut = id - feature * cuts;
    edges[id] = 0.5f * (centres[cut * width + feature] + centres[(cut + 1u) * width + feature]);
}

/* Every statistic totalled per feature and bin, with device atomics.
 *
 * The one kernel that reads the samples during a split search: after it,
 * choosing a split is arithmetic over `n_features * n_bins` numbers and never
 * touches a row again. One thread per `(sample, feature)` pair, because the
 * pair is what decides a slot and a thread per statistic would multiply the
 * code reads for nothing.
 *
 * The count goes in behind the caller's columns. Left to the caller as a
 * column of ones, a criterion needing counts would silently read a class total
 * the first time somebody forgot; one more atomic per pair is cheaper.
 *
 * Additions land in an order nothing fixes and float addition is not
 * associative, so a total's last bits move between runs on the same data --
 * the trade `base_scatter_add_rows` documents. */
kernel void sh_histogram(device const float* codes [[buffer(0)]],
                         device const float* stats [[buffer(1)]],
                         device const float* picks [[buffer(2)]],
                         device float* out [[buffer(3)]],
                         constant uint& draws [[buffer(4)]],
                         constant uint& rows [[buffer(5)]],
                         constant uint& width [[buffer(6)]],
                         constant uint& n_stats [[buffer(7)]],
                         constant uint& bins [[buffer(8)]],
                         constant uint& subset [[buffer(9)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= draws * width) return;
    uint draw = id / width;
    uint feature = id - draw * width;
    uint row = draw;
    if (subset != 0u) {
        float asked = picks[draw];
        if (not_a_number(asked) || asked < 0.0f) return;
        row = uint(asked + 0.5f);
        if (row >= rows) return;
    }
    float code = codes[row * width + feature];
    if (not_a_number(code) || code < 0.0f) return;
    uint bin = uint(code + 0.5f);
    if (bin >= bins) return;
    uint planes = n_stats + 1u;
    device float* slot = out + (feature * bins + bin) * planes;
    for (uint s = 0u; s < n_stats; ++s) {
        atomic_fetch_add_explicit(
            (device atomic<float>*)(slot + s),
            stats[row * n_stats + s],
            memory_order_relaxed
        );
    }
    atomic_fetch_add_explicit(
        (device atomic<float>*)(slot + n_stats),
        1.0f,
        memory_order_relaxed
    );
}

/* Running totals up each feature's bins: a threadgroup per `(feature, plane)`,
 * a thread per bin.
 *
 * A split at bin `b` sends bins `0..b` left, so the gain needs the prefix, and
 * one plane's prefix is a scan of at most `SCAN_WIDTH` numbers -- small enough
 * to stay in threadgroup memory throughout. Hillis-Steele over a pair of
 * scratch rows: eight barriered steps against the `n_bins` serial additions a
 * thread-per-bin version would do. Threads past the real bin count carry
 * zero. */
kernel void sh_prefix_bins(device const float* hist [[buffer(0)]],
                           device float* left [[buffer(1)]],
                           constant uint& width [[buffer(2)]],
                           constant uint& bins [[buffer(3)]],
                           constant uint& planes [[buffer(4)]],
                           uint slot [[threadgroup_position_in_grid]],
                           uint lid [[thread_index_in_threadgroup]]) {
    threadgroup float scratch[2][SCAN_WIDTH];
    if (slot >= width * planes) return;
    uint feature = slot / planes;
    uint plane = slot - feature * planes;
    uint at = (feature * bins + lid) * planes + plane;
    scratch[0][lid] = lid < bins ? hist[at] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint src = 0u;
    for (uint step = 1u; step < SCAN_WIDTH; step <<= 1) {
        float carried = lid >= step ? scratch[src][lid - step] : 0.0f;
        scratch[1u - src][lid] = scratch[src][lid] + carried;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        src = 1u - src;
    }
    if (lid < bins) left[at] = scratch[src][lid];
}

/* What each candidate split is worth, one thread per `(feature, bin)`.
 *
 * Every criterion is `cost(parent) - cost(left) - cost(right)`, and the
 * parent's cost is the same for every bin of a feature, so a positive gain
 * always means a better partition. Gini and the squared error drop to sums of
 * squares over the totals, exact and needing no impurity of their own.
 *
 * Everything a rule refuses -- a masked feature, an undersized child, too
 * little hessian, too little gain -- leaves `-inf` rather than a flag beside a
 * number, so `best_split` is a maximum and nothing re-applies the rule. */
kernel void sh_split_gains(device const float* left [[buffer(0)]],
                           device const float* mask [[buffer(1)]],
                           device const float* limits [[buffer(2)]],
                           device float* gains [[buffer(3)]],
                           constant uint& width [[buffer(4)]],
                           constant uint& bins [[buffer(5)]],
                           constant uint& planes [[buffer(6)]],
                           constant uint& criterion [[buffer(7)]],
                           constant uint& min_leaf [[buffer(8)]],
                           constant uint& masked [[buffer(9)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= width * bins) return;
    uint feature = id / bins;
    uint bin = id - feature * bins;
    gains[id] = -INFINITY;
    if (masked != 0u && mask[feature] == 0.0f) return;
    /* The last bin puts every sample on the left, so it is not a split. */
    if (bin + 1u >= bins) return;
    uint n_stats = planes - 1u;
    device const float* here = left + (feature * bins + bin) * planes;
    device const float* whole = left + (feature * bins + bins - 1u) * planes;
    float count_l = here[n_stats];
    float count_all = whole[n_stats];
    float count_r = count_all - count_l;
    /* Both children have to hold something before anything is divided by their
     * counts, whatever `min_samples_leaf` was set to. */
    if (count_l <= 0.0f || count_r <= 0.0f) return;
    if (count_l < float(min_leaf) || count_r < float(min_leaf)) return;
    float min_child = limits[0];
    float l2 = limits[1];
    float min_gain = limits[2];
    float gain = 0.0f;
    if (criterion == 3u) {
        float gl = here[0];
        float hl = here[1];
        float gt = whole[0];
        float ht = whole[1];
        float gr = gt - gl;
        float hr = ht - hl;
        if (hl < min_child || hr < min_child) return;
        if (hl + l2 <= 0.0f || hr + l2 <= 0.0f || ht + l2 <= 0.0f) return;
        gain = 0.5f * (gl * gl / (hl + l2) + gr * gr / (hr + l2) - gt * gt / (ht + l2));
    } else {
        if (count_l < min_child || count_r < min_child) return;
        if (criterion == 2u) {
            float sl = here[0];
            float st = whole[0];
            float sr = st - sl;
            gain = sl * sl / count_l + sr * sr / count_r - st * st / count_all;
        } else {
            for (uint k = 0u; k < n_stats; ++k) {
                float l = here[k];
                float t = whole[k];
                float r = t - l;
                if (criterion == 0u) {
                    gain += l * l / count_l + r * r / count_r - t * t / count_all;
                } else {
                    if (l > 0.0f) gain += l * log(l / count_l);
                    if (r > 0.0f) gain += r * log(r / count_r);
                    if (t > 0.0f) gain -= t * log(t / count_all);
                }
            }
        }
    }
    /* Strictly greater, so a pure node's zero-gain split is refused and a
     * total that came back as a NaN never wins a node. */
    if (not_a_number(gain) || gain <= min_gain) return;
    gains[id] = gain;
}

/* The best bin of each feature, then the best feature, then both packed with
 * the gain so a node costs one read back of three floats.
 *
 * Two stages of its own rather than jaitensor's row reduction, which starts
 * its running maximum at -FLT_MAX and so cannot tell a refused split from a
 * merely bad one: a node whose rule rejected everything came back with a
 * plausible feature index and a gain of -3.4e38.
 *
 * Nothing here compares against an infinity either, for the same reason
 * jaitensor does not. The shaders are compiled with the default options, whose
 * fast math licenses the compiler to assume no operand is infinite, so `WORST`
 * is the identity and a `-inf` the split kernel wrote simply never wins a
 * comparison. An empty pick is carried as an index of -1 instead. */
kernel void sh_feature_best(device const float* gains [[buffer(0)]],
                            device float* value [[buffer(1)]],
                            device float* index [[buffer(2)]],
                            constant uint& width [[buffer(3)]],
                            constant uint& bins [[buffer(4)]],
                            uint feature [[threadgroup_position_in_grid]],
                            uint lid [[thread_index_in_threadgroup]]) {
    threadgroup float best[GROUP];
    threadgroup uint at[GROUP];
    if (feature >= width) return;
    float mine = WORST;
    uint pick = bins;
    for (uint bin = lid; bin < bins; bin += GROUP) {
        float here = gains[feature * bins + bin];
        if (here > mine || (here == mine && bin < pick)) { mine = here; pick = bin; }
    }
    best[lid] = mine;
    at[lid] = pick;
    group_argmax(best, at, lid);
    if (lid != 0u) return;
    value[feature] = best[0];
    index[feature] = at[0] < bins ? float(at[0]) : -1.0f;
}

kernel void sh_overall_best(device const float* value [[buffer(0)]],
                            device const float* index [[buffer(1)]],
                            device float* out [[buffer(2)]],
                            constant uint& width [[buffer(3)]],
                            uint lid [[thread_index_in_threadgroup]]) {
    threadgroup float best[GROUP];
    threadgroup uint at[GROUP];
    float mine = WORST;
    uint pick = width;
    for (uint feature = lid; feature < width; feature += GROUP) {
        if (index[feature] < 0.0f) continue;
        float here = value[feature];
        if (here > mine || (here == mine && feature < pick)) { mine = here; pick = feature; }
    }
    best[lid] = mine;
    at[lid] = pick;
    group_argmax(best, at, lid);
    if (lid != 0u) return;
    bool found = at[0] < width;
    out[0] = found ? float(at[0]) : -1.0f;
    out[1] = found ? index[at[0]] : -1.0f;
    out[2] = best[0];
}

/* One thread walks one sample down to its leaf.
 *
 * A thread per sample and not per level: the descent is a chain of dependent
 * loads no width of threads can shorten, and the samples are what there are
 * millions of. The loop is bounded by the node count rather than written
 * `while (true)` -- arrays built wrong would otherwise hang the command
 * buffer, and a hung GPU takes the process with it. */
kernel void sh_tree_walk(device const float* x [[buffer(0)]],
                         device const float* feature [[buffer(1)]],
                         device const float* threshold [[buffer(2)]],
                         device const float* left [[buffer(3)]],
                         device const float* right [[buffer(4)]],
                         device float* leaf [[buffer(5)]],
                         constant uint& rows [[buffer(6)]],
                         constant uint& width [[buffer(7)]],
                         constant uint& nodes [[buffer(8)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= rows) return;
    uint node = 0u;
    for (uint depth = 0u; depth < nodes; ++depth) {
        float which = feature[node];
        if (which < 0.0f) break;
        uint column = uint(which + 0.5f);
        if (column >= width) break;
        float value = x[id * width + column];
        float next = value <= threshold[node] ? left[node] : right[node];
        if (next < 0.0f) break;
        uint step = uint(next + 0.5f);
        if (step >= nodes) break;
        node = step;
    }
    leaf[id] = float(node);
}

/* The leaf values a walk landed on, written or accumulated. `accumulate` is
 * what a forest and a booster want: the answer stays on the device across
 * however many trees there are and is never read back between them. */
kernel void sh_tree_emit(device const float* leaf [[buffer(0)]],
                         device const float* value [[buffer(1)]],
                         device const float* weight [[buffer(2)]],
                         device float* out [[buffer(3)]],
                         constant uint& count [[buffer(4)]],
                         constant uint& outputs [[buffer(5)]],
                         constant uint& nodes [[buffer(6)]],
                         constant uint& accumulate [[buffer(7)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint row = id / outputs;
    uint slot = id - row * outputs;
    uint node = uint(max(leaf[row], 0.0f) + 0.5f);
    float held = node < nodes ? value[node * outputs + slot] : 0.0f;
    if (accumulate != 0u) out[id] += weight[0] * held;
    else out[id] = held;
}

kernel void sh_bootstrap(device float* out [[buffer(0)]],
                         constant uint& count [[buffer(1)]],
                         constant uint& rows [[buffer(2)]],
                         constant uint& salt [[buffer(3)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    out[id] = float(min(uint(splitmix(id, salt) * float(rows)), rows - 1u));
}

/* Exactly `k` features marked, by ranking one key per feature.
 *
 * `width^2` comparisons in a single dispatch. Sorting the keys is less
 * arithmetic but it is a chain of a couple of dozen dispatches, and this runs
 * once per node of a forest -- thousands of times in a fit -- where the chain
 * and not the arithmetic is what shows up. Ties break on the feature index, so
 * the ranks are a permutation and the mask always holds exactly `k` ones. */
kernel void sh_choose_features(device float* mask [[buffer(0)]],
                               constant uint& width [[buffer(1)]],
                               constant uint& k [[buffer(2)]],
                               constant uint& salt [[buffer(3)]],
                               uint id [[thread_position_in_grid]]) {
    if (id >= width) return;
    float mine = splitmix(id, salt);
    uint rank = 0u;
    for (uint other = 0u; other < width; ++other) {
        if (other == id) continue;
        float theirs = splitmix(other, salt);
        if (theirs < mine || (theirs == mine && other < id)) rank += 1u;
    }
    mask[id] = rank < k ? 1.0f : 0.0f;
}

kernel void sh_add_bias(device const float* x [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        constant uint& count [[buffer(2)]],
                        constant uint& width [[buffer(3)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint wide = width + 1u;
    uint row = id / wide;
    uint column = id - row * wide;
    out[id] = column < width ? x[row * width + column] : 1.0f;
}

kernel void sh_add_row_vector(device float* m [[buffer(0)]],
                              device const float* v [[buffer(1)]],
                              constant uint& count [[buffer(2)]],
                              constant uint& width [[buffer(3)]],
                              uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    m[id] += v[id % width];
}

kernel void sh_soft_threshold(device const float* v [[buffer(0)]],
                              device const float* amount [[buffer(1)]],
                              device float* out [[buffer(2)]],
                              constant uint& count [[buffer(3)]],
                              uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float value = v[id];
    float shrunk = fabs(value) - amount[0];
    out[id] = shrunk > 0.0f ? copysign(shrunk, value) : 0.0f;
}

/* One threadgroup per column striding the samples, for the same reason
 * `base_column_reduce` does it: a thread per column leaves a GPU running a
 * whole matrix's work on as many threads as the data has features. */
kernel void sh_column_norms(device const float* x [[buffer(0)]],
                            device float* out [[buffer(1)]],
                            constant uint& rows [[buffer(2)]],
                            constant uint& width [[buffer(3)]],
                            constant uint& squared [[buffer(4)]],
                            uint column [[threadgroup_position_in_grid]],
                            uint lid [[thread_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup float lanes[32];
    if (column >= width) return;
    float running = 0.0f;
    for (uint row = lid; row < rows; row += GROUP) {
        float value = x[row * width + column];
        running += value * value;
    }
    float total = group_total(running, lanes, lid, lane, warp);
    if (lid == 0u) out[column] = squared != 0u ? total : sqrt(total);
}
"""

fn shared_kernel(entry: str) -> Kernel { return cached_kernel(SOURCE, entry) }

fn copied(source: Tensor, shape: list[int]) -> Tensor {
    let out = Tensor(shape)
    shared_kernel("sh_copy").dispatch_async(
        [source.buffer(), out.buffer()],
        source.size,
        [source.size]
    )
    return out
}

#: A seed for a kernel that draws its own randomness.
#:
#: Seed 0 means unseeded everywhere in jailearn, so it picks one up from a
#: fresh generator rather than handing every run the same constant. `tag`
#: separates the draws of two kernels that share a caller's seed.
fn device_salt(seed: int, tag: int) -> int {
    let root = if seed == 0 { rng_for(0).next_int() } else { seed }
    return root *% 6364136223846793 +% tag *% 1442695040888963 & 0xFFFFFFFF
}

#: The most bins a histogram may hold, set by the threadgroup the split scan
#: runs in. Two hundred and fifty-six is well past where another bin changes a
#: tree, and it keeps the scan's scratch at two kilobytes.
pub const MAX_BINS: int = 256

#: Equal-width bins between each column's smallest and largest value.
pub const BIN_UNIFORM: int = 0
#: Equal-count bins: the cuts are the column's quantiles.
pub const BIN_QUANTILE: int = 1
#: Cuts halfway between the centres of a one-dimensional k-means per column.
pub const BIN_KMEANS: int = 2

#: Lloyd passes `BIN_KMEANS` takes.
#:
#: A host loop, bounded and small: one-dimensional k-means started from the
#: quantile grid moves in the first few passes and then stops, and the
#: difference between twenty passes and two hundred is under a part in a
#: thousand of a bin width on every column tried. The alternative -- reading a
#: convergence test back each pass -- costs a device sync per pass to save
#: passes that are two dispatches each.
const BIN_KMEANS_STEPS: int = 20

#: The `n_bins - 1` cuts that bin every column of `x`, as `[n_features,
#: n_bins - 1]` ascending.
#:
#: `BIN_QUANTILE` is the default because it is the strategy that survives a
#: skewed column: uniform bins over a feature with one outlier put every real
#: sample in the first bin and offer a tree nothing to split on. `BIN_KMEANS`
#: costs a couple of dozen passes over the matrix and buys a little over
#: quantiles only when a column is clustered rather than merely skewed.
pub fn bin_edges(x: Tensor, n_bins: int, strategy: int = BIN_QUANTILE) -> Tensor {
    check_matrix(x)
    if n_bins < 2 or n_bins > MAX_BINS {
        throw ValueError(f"n_bins must be between 2 and {MAX_BINS}, got {n_bins}")
    }
    let width = x.shape[1]
    let cuts = n_bins - 1
    if strategy == BIN_UNIFORM {
        let (low, high) = column_min_max(x)
        let edges = Tensor([width, cuts])
        shared_kernel("sh_uniform_edges").dispatch_async(
            [low.buffer(), high.buffer(), edges.buffer()],
            edges.size,
            [width, cuts]
        )
        low.free()
        high.free()
        return edges
    }
    if strategy == BIN_QUANTILE {
        var fractions: list[float] = []
        for cut in 0..cuts { fractions.push(float(cut + 1) / float(n_bins)) }
        #: `column_quantiles` answers `[n_quantiles, n_features]`, which is the
        #: transpose of what everything downstream indexes by feature.
        let rows_first = column_quantiles(x, fractions)
        let edges = jt.transpose(rows_first)
        rows_first.free()
        return edges
    }
    if strategy != BIN_KMEANS { throw ValueError(f"unknown binning strategy {strategy}") }
    return kmeans_edges(x, n_bins, width, cuts)
}

fn kmeans_edges(x: Tensor, n_bins: int, width: int, cuts: int) -> Tensor {
    #: Started from the bins' quantile midpoints rather than from a uniform
    #: grid: the init is already sorted per column and already where the mass
    #: is, which is most of what the passes below would otherwise spend
    #: themselves fixing.
    var midpoints: list[float] = []
    for bin in 0..n_bins { midpoints.push((float(bin) + 0.5) / float(n_bins)) }
    let centres = column_quantiles(x, midpoints)
    let totals = shared_kernel("sh_kmeans_totals")
    let update = shared_kernel("sh_kmeans_update")
    for _pass in 0..BIN_KMEANS_STEPS {
        let sums = Tensor.zeros([n_bins, width])
        let counts = Tensor.zeros([n_bins, width])
        totals.dispatch_async(
            [x.buffer(), centres.buffer(), sums.buffer(), counts.buffer()],
            x.size,
            [x.size, width, n_bins]
        )
        update.dispatch_async(
            [sums.buffer(), counts.buffer(), centres.buffer()],
            centres.size,
            [centres.size]
        )
        sums.free()
        counts.free()
    }
    #: Lloyd on one dimension keeps a sorted set of centres sorted, but an
    #: empty cluster holding its old position does not have to, and the binary
    #: search in `sh_bin_codes` reads a non-monotone edge list as garbage.
    #: Sorting the columns once is a few dispatches against that risk.
    let ordered = sort_columns(centres)
    centres.free()
    let edges = Tensor([width, cuts])
    shared_kernel("sh_kmeans_edges").dispatch_async(
        [ordered.buffer(), edges.buffer()],
        edges.size,
        [width, cuts]
    )
    ordered.free()
    return edges
}

#: Which bin every element of `x` falls in, same shape as `x`.
#:
#: Codes are floats like every other index in the package. A NaN codes to -1,
#: which `bin_histograms` skips and no bin ever matches.
pub fn bin_codes(x: Tensor, edges: Tensor) -> Tensor {
    check_matrix(x)
    check_matrix(edges, "edges")
    if edges.shape[0] != x.shape[1] {
        throw ValueError(f"edges cover {edges.shape[0]} features, X has {x.shape[1]}")
    }
    let out = Tensor(x.shape)
    shared_kernel("sh_bin_codes").dispatch_async(
        [x.buffer(), edges.buffer(), out.buffer()],
        x.size,
        [x.size, x.shape[1], edges.shape[1]]
    )
    return out
}

#: A representative value per bin, `[n_features, n_bins]`.
#:
#: What an inverse transform hands back for a code, and what a plot puts on an
#: axis. The two outer bins are open, so their centres are extrapolated from
#: the neighbouring bin's width.
pub fn bin_centres(edges: Tensor) -> Tensor {
    check_matrix(edges, "edges")
    let width = edges.shape[0]
    let cuts = edges.shape[1]
    let out = Tensor([width, cuts + 1])
    shared_kernel("sh_bin_centres").dispatch_async(
        [edges.buffer(), out.buffer()],
        out.size,
        [width, cuts]
    )
    return out
}

#: Every statistic totalled per feature and bin, `[n_features, n_bins,
#: n_stats + 1]`.
#:
#: `codes` is `bin_codes`' answer and `stats` is `[n_samples, n_stats]`: one-hot
#: class indicators for gini and entropy, `[y, y*y]` for the squared error,
#: `[g, h]` for a booster. `rows` names the samples to include and `null` means
#: all of them, which is how a node uses its parent's codes without gathering.
#:
#: The last plane is the sample count, written by the kernel, so no caller
#: passes a column of ones and no criterion guesses where its counts came
#: from.
pub fn bin_histograms(codes: Tensor, stats: Tensor, rows: Tensor?, n_bins: int) -> Tensor {
    check_matrix(codes, "codes")
    check_matrix(stats, "stats")
    if stats.shape[0] != codes.shape[0] {
        throw ValueError(f"stats has {stats.shape[0]} rows for {codes.shape[0]} samples")
    }
    if n_bins < 2 or n_bins > MAX_BINS {
        throw ValueError(f"n_bins must be between 2 and {MAX_BINS}, got {n_bins}")
    }
    let width = codes.shape[1]
    let n_stats = stats.shape[1]
    let picks = rows
    let subset = if picks is null { 0 } else { 1 }
    let index = if picks is null { codes } else { picks }
    let draws = if picks is null { codes.shape[0] } else { picks.size }
    if draws == 0 { throw ValueError("a histogram needs at least one sample") }
    let out = Tensor.zeros([width, n_bins, n_stats + 1])
    shared_kernel("sh_histogram").dispatch_async(
        [codes.buffer(), stats.buffer(), index.buffer(), out.buffer()],
        draws * width,
        [draws, codes.shape[0], width, n_stats, n_bins, subset]
    )
    return out
}

#: Squared class purity: `stats` are one-hot class indicators.
pub const SPLIT_GINI: int = 0
#: Information gain: `stats` are one-hot class indicators.
pub const SPLIT_ENTROPY: int = 1
#: Variance reduction: `stats` are `[y, y*y]` and only the first is read.
pub const SPLIT_MSE: int = 2
#: A booster's `G^2 / (H + l2)`: `stats` are `[g, h]`.
pub const SPLIT_GRADIENT: int = 3

#: Which splits a node will accept and what they are worth.
#:
#: One object rather than eight arguments through three calls, because a tree
#: builds it once and reuses it at every node. The feature mask is deliberately
#: a `var` the caller reassigns: a random forest draws a fresh one per node
#: from `choose_features`, and the rule points at whatever is current rather
#: than owning it. Nothing here frees the mask.
pub class SplitRule {
    #: One of `SPLIT_GINI`, `SPLIT_ENTROPY`, `SPLIT_MSE`, `SPLIT_GRADIENT`.
    pub let criterion: int
    #: The fewest samples either child may keep.
    pub var min_samples_leaf: int
    #: The least hessian a child may carry under `SPLIT_GRADIENT`, and the
    #: least weight under the others.
    pub var min_child_weight: float
    #: The ridge term in a booster's leaf denominator.
    pub var l2: float
    #: The gain a split must beat, strictly. Zero already refuses the zero-gain
    #: split a pure node offers.
    pub var min_gain: float
    #: A `[n_features]` mask of ones and zeros, or `null` for every feature.
    pub var feature_mask: Tensor?

    pub fn init(
        self,
        criterion: int,
        min_samples_leaf: int = 1,
        min_child_weight: float = 0.0,
        l2: float = 0.0,
        min_gain: float = 0.0,
        feature_mask: Tensor? = null
    ) {
        if criterion < SPLIT_GINI or criterion > SPLIT_GRADIENT {
            throw ValueError(f"unknown split criterion {criterion}")
        }
        self.criterion = criterion
        self.min_samples_leaf = max(min_samples_leaf, 1)
        self.min_child_weight = min_child_weight
        self.l2 = l2
        self.min_gain = min_gain
        self.feature_mask = feature_mask
    }

    fn __str__(self) -> str {
        return f"SplitRule(criterion={self.criterion}, min_samples_leaf={self.min_samples_leaf})"
    }
}

fn histogram_shape(hist: Tensor) -> tuple[int, int, int] {
    if hist.rank() != 3 {
        throw ValueError(f"a histogram must be [n_features, n_bins, n_stats + 1], got {hist.shape}")
    }
    if hist.shape[1] < 2 or hist.shape[1] > MAX_BINS {
        throw ValueError(f"a histogram has between 2 and {MAX_BINS} bins, got {hist.shape[1]}")
    }
    if hist.shape[2] < 2 {
        throw ValueError("a histogram needs at least one statistic beside its counts")
    }
    return (hist.shape[0], hist.shape[1], hist.shape[2])
}

#: What every candidate split is worth, `[n_features, n_bins]`.
#:
#: Entry `[f][b]` is the gain of sending bins `0..b` of feature `f` left. The
#: last bin of every feature, every split a rule refuses and every feature the
#: mask drops are all `-inf`, so a caller reduces rather than filters.
pub fn split_gains(hist: Tensor, rule: SplitRule) -> Tensor {
    let (width, bins, planes) = histogram_shape(hist)
    if rule.criterion == SPLIT_GRADIENT and planes < 3 {
        throw ValueError("a gradient split needs both a gradient and a hessian column")
    }
    let mask = rule.feature_mask
    if mask is not null and mask.size != width {
        throw ValueError(f"the feature mask has {mask.size} entries for {width} features")
    }
    let running = Tensor(hist.shape)
    shared_kernel("sh_prefix_bins").dispatch_async(
        [hist.buffer(), running.buffer()],
        width * planes * MAX_BINS,
        [width, bins, planes],
        group_size: MAX_BINS
    )
    let gains = Tensor([width, bins])
    let limits = Buffer.of([rule.min_child_weight, rule.l2, rule.min_gain])
    let table = if mask is null { gains } else { mask }
    let masked = if mask is null { 0 } else { 1 }
    shared_kernel("sh_split_gains").dispatch_async(
        [running.buffer(), table.buffer(), limits, gains.buffer()],
        gains.size,
        [width, bins, planes, rule.criterion, rule.min_samples_leaf, masked]
    )
    running.free()
    limits.free()
    return gains
}

#: The best split a histogram offers: `(feature, bin, gain)`.
#:
#: `(-1, -1, -inf)` when the rule refuses every candidate, which is a tree's
#: signal to make the node a leaf.
#:
#: The three floats read back here are the only host sync in a tree fit. The
#: reduction is two threadgroup stages -- a feature's bins, then the features
#: -- and packs the winner with its gain, so a node costs one download.
pub fn best_split(hist: Tensor, rule: SplitRule) -> tuple[int, int, float] {
    let width = hist.shape[0]
    let bins = hist.shape[1]
    let gains = split_gains(hist, rule)
    let value = Tensor([width])
    let index = Tensor([width])
    shared_kernel("sh_feature_best").dispatch_async(
        [gains.buffer(), value.buffer(), index.buffer()],
        width * 256,
        [width, bins],
        group_size: 256
    )
    let packed = Tensor([3])
    shared_kernel("sh_overall_best").dispatch_async(
        [value.buffer(), index.buffer(), packed.buffer()],
        256,
        [width],
        group_size: 256
    )
    let answer = packed.download()
    value.free()
    index.free()
    gains.free()
    packed.free()
    if answer[0] < 0.0 { return (-1, -1, -math.INF) }
    return (int(answer[0] + 0.5), int(answer[1] + 0.5), answer[2])
}

#: A grown tree in the form the device predicts from.
#:
#: Node `i` is a leaf when `feature[i]` is negative, and otherwise sends a
#: sample to `left[i]` when its value is at most `threshold[i]` and to
#: `right[i]` above it. Node 0 is the root and `value` carries a row per node.
#:
#: Built from host lists, once, at the end of a fit. Nodes number in the
#: hundreds or low thousands -- the growth loop is over nodes, never samples --
#: so walking them on the host costs nothing measurable, and growing in host
#: lists is what lets a tree append a node without reallocating five tensors.
pub class TreeArrays {
    let feature: Tensor
    let threshold: Tensor
    let left: Tensor
    let right: Tensor
    let value: Tensor
    let nodes: int
    let outputs: int

    pub fn init(
        self,
        feature: list[int],
        threshold: list[float],
        left: list[int],
        right: list[int],
        value: list[list[float]]
    ) {
        let nodes = feature.len()
        if nodes == 0 { throw ValueError("a tree needs at least a root") }
        if threshold.len() != nodes or left.len() != nodes or right.len() != nodes {
            throw ValueError("a tree's node arrays must all be the same length")
        }
        if value.len() != nodes {
            throw ValueError(f"{value.len()} value rows for {nodes} nodes")
        }
        let outputs = value[0].len()
        if outputs == 0 { throw ValueError("a tree node needs at least one output") }
        var flat: list[float] = []
        for (at, row) in value.enumerate() {
            if row.len() != outputs {
                throw ValueError(f"node {at} has {row.len()} outputs, expected {outputs}")
            }
            for held in row { flat.push(held) }
        }
        var features: list[float] = []
        var lefts: list[float] = []
        var rights: list[float] = []
        for at in 0..nodes {
            if left[at] >= nodes or right[at] >= nodes {
                throw IndexError(f"node {at} names a child outside the {nodes} nodes")
            }
            features.push(float(feature[at]))
            lefts.push(float(left[at]))
            rights.push(float(right[at]))
        }
        self.feature = Tensor([nodes], features)
        self.threshold = Tensor([nodes], threshold)
        self.left = Tensor([nodes], lefts)
        self.right = Tensor([nodes], rights)
        self.value = Tensor([nodes, outputs], flat)
        self.nodes = nodes
        self.outputs = outputs
    }

    #: How many nodes the tree has, leaves included.
    pub fn node_count(self) -> int { return self.nodes }

    #: How wide each node's value is: 1 for a regressor, `n_classes` for a
    #: classifier holding class shares.
    pub fn output_count(self) -> int { return self.outputs }

    #: The leaf value for every row, `[n_samples, n_outputs]`.
    pub fn predict(self, x: Tensor) -> Tensor {
        let leaf = self.apply(x)
        let out = Tensor([x.shape[0], self.outputs])
        self._emit(leaf, out, 0.0, 0)
        leaf.free()
        return out
    }

    #: The node each row lands in, `[n_samples]` float ids.
    pub fn apply(self, x: Tensor) -> Tensor {
        check_matrix(x)
        let leaf = Tensor([x.shape[0]])
        shared_kernel("sh_tree_walk").dispatch_async(
            [
                x.buffer(),
                self.feature.buffer(),
                self.threshold.buffer(),
                self.left.buffer(),
                self.right.buffer(),
                leaf.buffer(),
            ],
            x.shape[0],
            [x.shape[0], x.shape[1], self.nodes]
        )
        return leaf
    }

    #: `total += weight * predict(x)` in place. `tree_accumulate` is the name
    #: the ensembles call it by; this is where it happens.
    pub fn accumulate(self, x: Tensor, total: Tensor, weight: float) -> void {
        check_matrix(total, "total")
        if total.shape[0] != x.shape[0] or total.shape[1] != self.outputs {
            throw ValueError(f"total is {total.shape}, expected [{x.shape[0]}, {self.outputs}]")
        }
        let leaf = self.apply(x)
        self._emit(leaf, total, weight, 1)
        leaf.free()
    }

    fn _emit(self, leaf: Tensor, out: Tensor, weight: float, accumulate: int) -> void {
        let scale = Buffer.of([weight])
        shared_kernel("sh_tree_emit").dispatch_async(
            [leaf.buffer(), self.value.buffer(), scale, out.buffer()],
            out.size,
            [out.size, self.outputs, self.nodes, accumulate]
        )
        scale.free()
    }

    #: Release the node arrays. Freeing twice is harmless.
    pub fn free(self) -> void {
        self.feature.free()
        self.threshold.free()
        self.left.free()
        self.right.free()
        self.value.free()
    }

    fn __str__(self) -> str {
        return f"TreeArrays(nodes={self.nodes}, outputs={self.outputs})"
    }
}

#: `total += weight * tree.predict(x)`, without materialising the prediction.
#:
#: This is the inner step of every ensemble in the package. `total` stays on
#: the device from the first tree to the last, so a hundred trees cost a
#: hundred walks and no allocations beyond one leaf vector each.
pub fn tree_accumulate(tree: TreeArrays, x: Tensor, total: Tensor, weight: float) -> void {
    tree.accumulate(x, total, weight)
}

#: The mean prediction of a forest, `[n_samples, outputs]`.
#:
#: The host loop is over trees, which is `n_estimators` -- hundreds, and the
#: number the caller chose.
pub fn forest_mean(trees: list[TreeArrays], x: Tensor, outputs: int) -> Tensor {
    check_matrix(x)
    if trees.len() == 0 { throw ValueError("a forest needs at least one tree") }
    if outputs <= 0 { throw ValueError(f"outputs must be positive, got {outputs}") }
    let share = 1.0 / float(trees.len())
    let total = Tensor.zeros([x.shape[0], outputs])
    for (at, tree) in trees.enumerate() {
        if tree.output_count() != outputs {
            total.free()
            throw ValueError(f"tree {at} has {tree.output_count()} outputs, expected {outputs}")
        }
        tree_accumulate(tree, x, total, share)
    }
    return total
}

#: A random permutation of `0..n-1`, as `[n]` float indices.
#:
#: Random keys sorted on the device: the sort is `base`'s, so no sample reaches
#: the host, and the result is a permutation even when two float32 keys collide
#: -- a sort rearranges whatever it is handed, ties and all.
pub fn permutation(n: int, seed: int = 0) -> Tensor {
    if n <= 0 { throw ValueError(f"a permutation needs a positive length, got {n}") }
    let keys = Tensor([n, 1])
    keys.buffer().fill_uniform(0.0, 1.0, device_salt(seed, 0x51ED270B))
    let (sorted, origin) = argsort_columns(keys)
    keys.free()
    sorted.free()
    let out = copied(origin, [n])
    origin.free()
    return out
}

#: `size` row indices drawn with replacement from `0..n-1`.
#:
#: A `size` of 0 means `n`, which is what a bagging estimator wants. One
#: dispatch and no state: each draw is its own splitmix, so a bootstrap over a
#: million rows costs a single pass.
pub fn bootstrap_rows(n: int, size: int = 0, seed: int = 0) -> Tensor {
    if n <= 0 { throw ValueError(f"cannot draw from {n} rows") }
    let drawn = if size > 0 { size } else { n }
    let out = Tensor([drawn])
    shared_kernel("sh_bootstrap").dispatch_async(
        [out.buffer()],
        drawn,
        [drawn, n, device_salt(seed, 0x2545F491)]
    )
    return out
}

#: A `[width]` mask with exactly `k` ones, for the features a node may split
#: on. A `k` of 0 or `width` marks everything.
pub fn choose_features(width: int, k: int, seed: int = 0) -> Tensor {
    if width <= 0 { throw ValueError(f"cannot choose from {width} features") }
    let wanted = if k <= 0 or k >= width { width } else { k }
    let mask = Tensor([width])
    shared_kernel("sh_choose_features").dispatch_async(
        [mask.buffer()],
        width,
        [width, wanted, device_salt(seed, 0x68E31DA4)]
    )
    return mask
}

#: The sample weights a fit will use, `[n]`, always a fresh tensor the caller
#: owns and frees. `null` gives every sample a weight of one.
pub fn sample_weights(w: Tensor?, n: int) -> Tensor {
    if n <= 0 { throw ValueError(f"cannot weight {n} samples") }
    let given = w
    if given is null { return Tensor.full([n], 1.0) }
    if given.size != n {
        throw ValueError(f"{given.size} sample weights for {n} samples")
    }
    return copied(given, [n])
}

#: `x` with a column of ones on the right, `[n_samples, n_features + 1]`.
#:
#: For the solvers that would rather carry the intercept inside the coefficient
#: vector than fit it separately.
pub fn add_bias(x: Tensor) -> Tensor {
    check_matrix(x)
    let out = Tensor([x.shape[0], x.shape[1] + 1])
    shared_kernel("sh_add_bias").dispatch_async(
        [x.buffer(), out.buffer()],
        out.size,
        [out.size, x.shape[1]]
    )
    return out
}

#: `x @ coef.T + intercept`, `[n_samples, n_targets]`.
#:
#: `coef` is `[n_targets, n_features]` and `intercept` is `[n_targets]` or
#: `null`, which is scikit-learn's shape for both. A one-target model passes a
#: one-row `coef` and squeezes the answer rather than getting a different shape
#: back, so the linear models and the logistic regressions share this call.
pub fn linear_scores(x: Tensor, coef: Tensor, intercept: Tensor? = null) -> Tensor {
    check_matrix(x)
    check_matrix(coef, "coef")
    if coef.shape[1] != x.shape[1] {
        throw ValueError(f"coef covers {coef.shape[1]} features, X has {x.shape[1]}")
    }
    let bias = intercept
    if bias is not null and bias.size != coef.shape[0] {
        throw ValueError(f"{bias.size} intercepts for {coef.shape[0]} targets")
    }
    let out = jt.matmul(x, coef, trans_b: true)
    if bias is not null {
        shared_kernel("sh_add_row_vector").dispatch_async(
            [out.buffer(), bias.buffer()],
            out.size,
            [out.size, coef.shape[0]]
        )
    }
    return out
}

#: `sign(v) * max(|v| - amount, 0)`, elementwise.
#:
#: The proximal step of an L1 penalty, and so the one line that separates a
#: lasso's coordinate descent from a ridge's.
pub fn soft_threshold(v: Tensor, amount: float) -> Tensor {
    let out = Tensor(v.shape)
    let level = Buffer.of([amount])
    shared_kernel("sh_soft_threshold").dispatch_async(
        [v.buffer(), level, out.buffer()],
        v.size,
        [v.size]
    )
    level.free()
    return out
}

#: The euclidean length of every column, as `[n_features]`.
#:
#: `squared` skips the root, which is the form a coordinate descent divides by
#: and saves it squaring the answer back.
pub fn column_norms(x: Tensor, squared: bool = false) -> Tensor {
    check_matrix(x)
    let width = x.shape[1]
    let out = Tensor([width])
    let wanted = if squared { 1 } else { 0 }
    shared_kernel("sh_column_norms").dispatch_async(
        [x.buffer(), out.buffer()],
        width * 256,
        [x.shape[0], width, wanted],
        group_size: 256
    )
    return out
}

#: One iteration of an iterative fit, given its index. It returns how far the
#: parameters moved, and `Descent` stops when that is small enough.
pub type DescentStep = fn(int) -> float

#: The loop every iterative fit runs, so all of them report convergence alike.
#:
#: A coordinate descent, an EM, an SMO and an SGD differ in what a step does
#: and agree on everything around it: an iteration cap, a movement tolerance, a
#: count of what ran and whether the tolerance was reached. Estimators that
#: keep those four themselves end up disagreeing about whether `n_iter` counts
#: the step that converged, which reads to a user as a bug.
#:
#: The step returns a movement, not a loss, and the loop never reads it back
#: from the device -- how a fit measures its own movement is the fit's
#: business.
pub class Descent {
    #: The most steps `run` will take.
    pub let max_iter: int
    #: The movement below which the fit is converged.
    pub let tol: float
    #: How many steps the last `run` took.
    pub var n_iter: int
    #: Whether the last `run` reached `tol` before `max_iter`.
    pub var converged: bool

    pub fn init(self, max_iter: int = 100, tol: float = 1.0e-4) {
        if max_iter <= 0 { throw ValueError(f"max_iter must be positive, got {max_iter}") }
        if tol < 0.0 { throw ValueError(f"tol cannot be negative, got {tol}") }
        self.max_iter = max_iter
        self.tol = tol
        self.n_iter = 0
        self.converged = false
    }

    #: Run `step` until it reports a movement of at most `tol`, or until
    #: `max_iter` steps have been taken.
    pub fn run(self, step: DescentStep) -> void {
        self.n_iter = 0
        self.converged = false
        for iteration in 0..self.max_iter {
            let movement = step(iteration)
            self.n_iter = iteration + 1
            #: A NaN movement means the parameters have already left the
            #: representable range, and running to `max_iter` from there only
            #: buys the caller a fitted model made of NaNs.
            if math.is_nan(movement) {
                throw RuntimeError(f"the fit diverged at iteration {iteration}")
            }
            if movement <= self.tol {
                self.converged = true
                return
            }
        }
    }

    #: Forget the last run, for an estimator that refits in place.
    pub fn reset(self) -> void {
        self.n_iter = 0
        self.converged = false
    }

    fn __str__(self) -> str {
        return f"Descent(n_iter={self.n_iter}, converged={self.converged})"
    }
}
