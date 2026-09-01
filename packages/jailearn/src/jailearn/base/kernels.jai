#: The Metal source behind every base primitive, and the cache in front of it.
from std.gpu import Kernel

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

pub fn base_kernel(entry: str) -> Kernel { return cached_kernel(SOURCE, entry) }
