#: Grouping data and finding the directions it varies in: `kmeans` and `PCA`.
#:
#: Both take a `Mat` whose rows are samples and whose columns are features,
#: which is the layout OpenCV's `ml` module uses throughout, and both return
#: what OpenCV returns in the order it returns it.
#:
#: ```
#: let (compactness, labels, centres) = kmeans(samples, 4, 10)
#: let pca = PCA(samples, 2)
#: let reduced = pca.project(samples)
#: ```
import std.math
from std.gpu import Buffer, Kernel
from std.os import env_or
from ..core.linalg import device_centred_product, device_scatter, matmul, on_device
from ..core.linalg import symmetric_eigen, transpose
from ..core.mat import Mat
from ..core.types import CV_32F, CV_32S, CV_MAKETYPE

#: Start from centres picked at random.
pub const KMEANS_RANDOM_CENTERS = 0
#: Start from centres spread out by k-means++.
pub const KMEANS_PP_CENTERS = 2
#: Start from the centres the caller passed in.
pub const KMEANS_USE_INITIAL_LABELS = 1

fn squared_distance(values: list[float], a: int, b: int, width: int) -> float {
    var total = 0.0
    for index in 0..width {
        let gap = values[a * width + index] - values[b * width + index]
        total += gap * gap
    }
    return total
}

#: The kernels kmeans runs. Public so the tests can hold the replacement
#: kernels to the ones they replaced, bit for bit.
pub let SOURCE = """#include <metal_stdlib>
using namespace metal;

/* Which centre each sample belongs to, and how far away it is.
 *
 * A sample's nearest centre depends on nothing but the centres, so the whole
 * assignment is one dispatch. It is also the bulk of the work -- samples times
 * centres times width, every iteration. */
kernel void kmeans_assign(device const float* values [[buffer(0)]],
                          device const float* centres [[buffer(1)]],
                          device float* labels [[buffer(2)]],
                          device float* nearest [[buffer(3)]],
                          constant uint& samples [[buffer(4)]],
                          constant uint& width [[buffer(5)]],
                          constant uint& k [[buffer(6)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= samples) return;
    uint base = id * width;
    float best = INFINITY;
    uint pick = 0u;
    for (uint centre = 0u; centre < k; ++centre) {
        float total = 0.0f;
        uint at = centre * width;
        for (uint index = 0u; index < width; ++index) {
            float gap = values[base + index] - centres[at + index];
            total += gap * gap;
        }
        if (total < best) {
            best = total;
            pick = centre;
        }
    }
    labels[id] = float(pick);
    nearest[id] = best;
}

/* How far each sample is from one centre, kept against the best so far.
 *
 * The seeding walks the samples once per centre it adds, which is the same
 * shape as the assignment and the same reason to put it on the device: with
 * the iterations moved off the host it was what remained, two hundred of the
 * two hundred and fifty milliseconds a VGA frame took. */
kernel void kmeans_seed(device const float* values [[buffer(0)]],
                        device const float* centres [[buffer(1)]],
                        device float* distances [[buffer(2)]],
                        constant uint& samples [[buffer(3)]],
                        constant uint& width [[buffer(4)]],
                        constant uint& centre [[buffer(5)]],
                        constant uint& replace [[buffer(6)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= samples) return;
    uint base = id * width;
    uint at = centre * width;
    float total = 0.0f;
    for (uint index = 0u; index < width; ++index) {
        float gap = values[base + index] - centres[at + index];
        total += gap * gap;
    }
    distances[id] = replace != 0u ? total : min(distances[id], total);
}

/* One total per centre and column, a threadgroup striding the samples.
 *
 * A thread apiece would be as many threads as there are centres times columns,
 * which for eight colours is twenty-four -- see the same shape in jaitensor's
 * normalisations. Slot `k * width` counts the members of each centre. */
kernel void kmeans_gather(device const float* values [[buffer(0)]],
                          device const float* labels [[buffer(1)]],
                          device const float* nearest [[buffer(2)]],
                          device float* sums [[buffer(3)]],
                          constant uint& samples [[buffer(4)]],
                          constant uint& width [[buffer(5)]],
                          constant uint& k [[buffer(6)]],
                          uint slot [[threadgroup_position_in_grid]],
                          uint lid [[thread_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint warp [[simdgroup_index_in_threadgroup]]) {
    const uint GROUP = 256u;
    threadgroup float lanes[32];
    uint total_slots = k * width + k + 1u;
    if (slot >= total_slots) return;

    float running = 0.0f;
    if (slot < k * width) {
        uint centre = slot / width;
        uint column = slot - centre * width;
        for (uint sample = lid; sample < samples; sample += GROUP) {
            if (uint(labels[sample]) == centre) running += values[sample * width + column];
        }
    } else if (slot < k * width + k) {
        uint centre = slot - k * width;
        for (uint sample = lid; sample < samples; sample += GROUP) {
            if (uint(labels[sample]) == centre) running += 1.0f;
        }
    } else {
        for (uint sample = lid; sample < samples; sample += GROUP) running += nearest[sample];
    }

    if (lid < 32u) lanes[lid] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    running = simd_sum(running);
    if (lane == 0u) lanes[warp] = running;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid != 0u) return;
    float answer = 0.0f;
    for (uint j = 0u; j < 32u; j++) answer += lanes[j];
    sums[slot] = answer;
}

/* `kmeans_assign` with the centres staged in threadgroup memory and one
 * running total per centre in registers, so each sample's values are read
 * once rather than once per centre. The arithmetic per centre is the same,
 * in the same order, so the labels and distances are the same bit for bit.
 * For k up to 16 and k * width up to 4096. */
kernel void kmeans_assign_staged(device const float* values [[buffer(0)]],
                                 device const float* centres [[buffer(1)]],
                                 device float* labels [[buffer(2)]],
                                 device float* nearest [[buffer(3)]],
                                 constant uint& samples [[buffer(4)]],
                                 constant uint& width [[buffer(5)]],
                                 constant uint& k [[buffer(6)]],
                                 uint id [[thread_position_in_grid]],
                                 uint lid [[thread_index_in_threadgroup]],
                                 uint tg [[threads_per_threadgroup]]) {
    threadgroup float staged[4096];
    for (uint i = lid; i < k * width; i += tg) staged[i] = centres[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (id >= samples) return;
    float totals[16];
    for (uint c = 0u; c < 16u; ++c) totals[c] = 0.0f;
    uint base = id * width;
    for (uint index = 0u; index < width; ++index) {
        float value = values[base + index];
        for (uint c = 0u; c < 16u && c < k; ++c) {
            float gap = value - staged[c * width + index];
            totals[c] = fma(gap, gap, totals[c]);
        }
    }
    float best = INFINITY;
    uint pick = 0u;
    for (uint c = 0u; c < 16u && c < k; ++c) {
        if (totals[c] < best) {
            best = totals[c];
            pick = c;
        }
    }
    labels[id] = float(pick);
    nearest[id] = best;
}

/* `kmeans_assign_staged` with the samples read a tile at a time.
 *
 * A thread per sample reading its own row strides the device memory by the
 * row's width, so every load in a simdgroup touches a different cache line.
 * Here the threadgroup's 256 rows are copied eight columns at a time into
 * threadgroup memory, consecutive threads reading consecutive values, and
 * each thread then takes its row's eight from there. Every thread still adds
 * its row's values in column order, so the labels and distances are the same
 * bit for bit. Dispatched in groups of exactly 256. */
kernel void kmeans_assign_tiled(device const float* values [[buffer(0)]],
                                device const float* centres [[buffer(1)]],
                                device float* labels [[buffer(2)]],
                                device float* nearest [[buffer(3)]],
                                constant uint& samples [[buffer(4)]],
                                constant uint& width [[buffer(5)]],
                                constant uint& k [[buffer(6)]],
                                uint id [[thread_position_in_grid]],
                                uint lid [[thread_index_in_threadgroup]],
                                uint group [[threadgroup_position_in_grid]]) {
    const uint GROUP = 256u;
    const uint TILE = 8u;
    const uint STRIDE = TILE + 1u;
    threadgroup float staged[4096];
    threadgroup float tile[256 * 9];
    for (uint i = lid; i < k * width; i += GROUP) staged[i] = centres[i];
    float totals[16];
    for (uint c = 0u; c < 16u; ++c) totals[c] = 0.0f;
    uint first = group * GROUP;
    for (uint column = 0u; column < width; column += TILE) {
        uint span = min(TILE, width - column);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lid; i < GROUP * span; i += GROUP) {
            uint row = i / span;
            uint at = i - row * span;
            uint sample = first + row;
            tile[row * STRIDE + at] = sample < samples ? values[sample * width + column + at] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint at = 0u; at < span; ++at) {
            float value = tile[lid * STRIDE + at];
            uint index = column + at;
            for (uint c = 0u; c < 16u && c < k; ++c) {
                float gap = value - staged[c * width + index];
                totals[c] = fma(gap, gap, totals[c]);
            }
        }
    }
    if (id >= samples) return;
    float best = INFINITY;
    uint pick = 0u;
    for (uint c = 0u; c < 16u && c < k; ++c) {
        if (totals[c] < best) {
            best = totals[c];
            pick = c;
        }
    }
    labels[id] = float(pick);
    nearest[id] = best;
}

/* `kmeans_gather` in one pass over the labels per column instead of one per
 * centre and column.
 *
 * The old form gave every (centre, column) slot its own threadgroup, and each
 * of those 409 groups -- for eight centres of fifty columns -- read all the
 * labels to find its few members. Here a group takes a whole column (or the
 * counts and the cost, in the last group), and each thread keeps one running
 * total per centre in threadgroup memory as it walks its samples. Each thread
 * walks the same samples in the same order the old one did, and the totals
 * are combined across the group the same way, so every sum comes out the same
 * bit for bit. For k up to 16. */
kernel void kmeans_gather_columns(device const float* values [[buffer(0)]],
                                  device const float* labels [[buffer(1)]],
                                  device const float* nearest [[buffer(2)]],
                                  device float* sums [[buffer(3)]],
                                  constant uint& samples [[buffer(4)]],
                                  constant uint& width [[buffer(5)]],
                                  constant uint& k [[buffer(6)]],
                                  constant uint& by_column [[buffer(7)]],
                                  uint column [[threadgroup_position_in_grid]],
                                  uint lid [[thread_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint warp [[simdgroup_index_in_threadgroup]]) {
    const uint GROUP = 256u;
    threadgroup float running[256 * 17];
    threadgroup float lanes[17 * 32];
    uint slots = k + 1u;
    for (uint c = 0u; c < slots; ++c) running[lid * 17u + c] = 0.0f;
    for (uint j = lid; j < 17u * 32u; j += GROUP) lanes[j] = 0.0f;
    if (column < width) {
        for (uint sample = lid; sample < samples; sample += GROUP) {
            uint centre = uint(labels[sample]);
            float value = by_column != 0u ? values[column * samples + sample] : values[sample * width + column];
            if (centre < k) running[lid * 17u + centre] += value;
        }
    } else {
        for (uint sample = lid; sample < samples; sample += GROUP) {
            uint centre = uint(labels[sample]);
            if (centre < k) running[lid * 17u + centre] += 1.0f;
        }
        for (uint sample = lid; sample < samples; sample += GROUP) {
            running[lid * 17u + k] += nearest[sample];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint used = column < width ? k : slots;
    for (uint c = 0u; c < used; ++c) {
        float total = simd_sum(running[lid * 17u + c]);
        if (lane == 0u) lanes[c * 32u + warp] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid != 0u) return;
    for (uint c = 0u; c < used; ++c) {
        float answer = 0.0f;
        for (uint j = 0u; j < 32u; j++) answer += lanes[c * 32u + j];
        if (column < width) {
            sums[c * width + column] = answer;
        } else {
            sums[k * width + c] = answer;
        }
    }
}

/* The samples a column at a time: `target[column * samples + sample]`. With
 * the samples laid out this way `kmeans_gather_columns` reads each column in
 * one coalesced sweep rather than a value per row. */
kernel void kmeans_transpose(device const float* values [[buffer(0)]],
                             device float* target [[buffer(1)]],
                             constant uint& samples [[buffer(2)]],
                             constant uint& width [[buffer(3)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= samples * width) return;
    uint sample = id / width;
    uint column = id - sample * width;
    target[column * samples + sample] = values[id];
}

/* Copy `count` values from one buffer to another. */
kernel void kmeans_copy(device const float* source [[buffer(0)]],
                        device float* target [[buffer(1)]],
                        constant uint& count [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
    if (id < count) target[id] = source[id];
}
"""

var _kernels: dict[str, Kernel] = {}

fn kernel_for(entry: str) -> Kernel {
    let existing = _kernels.get(entry, null)
    if existing is not null { return existing }
    let compiled = Kernel.compile(SOURCE, entry)
    _kernels[entry] = compiled
    return compiled
}

fn squared_to_centre(
    values: list[float],
    sample: int,
    centres: list[float],
    centre: int,
    width: int
) -> float {
    var total = 0.0
    for index in 0..width {
        let gap = values[sample * width + index] - centres[centre * width + index]
        total += gap * gap
    }
    return total
}

#: `JAICV_KMEANS_STAGED=0` assigns samples with the kernel that reads each
#: sample once per centre, and `JAICV_KMEANS_TILED=0` with the staged one
#: that reads each sample's row straight from device memory;
#: `JAICV_KMEANS_COLUMNS=0` totals the groups with the one that reads every
#: label once per centre and column; and `JAICV_KMEANS_IN_PLACE=0` copies the
#: samples to the host and back rather than reading them where they lie. All
#: four give the same answer either way.
let _KMEANS_STAGED = env_or("JAICV_KMEANS_STAGED", "1") != "0"
let _KMEANS_TILED = env_or("JAICV_KMEANS_TILED", "1") != "0"
let _KMEANS_COLUMNS = env_or("JAICV_KMEANS_COLUMNS", "1") != "0"
let _KMEANS_IN_PLACE = env_or("JAICV_KMEANS_IN_PLACE", "1") != "0"
#: `JAICV_KMEANS_TRANSPOSED=0` totals the groups from the samples as they are
#: laid out, a row at a time, instead of from a copy laid out by column.
let _KMEANS_TRANSPOSED = env_or("JAICV_KMEANS_TRANSPOSED", "1") != "0"

#: Split samples into `k` groups, each around its own centre.
#:
#: Returns how tightly the groups sit — the summed squared distance from each
#: sample to its centre, which is what OpenCV calls compactness — along with
#: each sample's group and the centres themselves.
#:
#: The starting centres come from k-means++ by default: the first at random,
#: and each after it drawn with probability proportional to how far it is from
#: the nearest centre already chosen. That is what stops two centres landing in
#: the same cluster and leaving another with none.
pub fn kmeans(
    data: Mat,
    k: int,
    attempts: int = 3,
    iterations: int = 30,
    epsilon: float = 1e-4,
    flags: int = KMEANS_PP_CENTERS,
    seed: int = 0x243F6A8885A3
) -> tuple[float, Mat, Mat] {
    if k <= 0 { throw ValueError("k has to be at least one") }
    data.flush()
    let samples = data.rows
    let width = data.cols * data.channels()
    if samples < k { throw ValueError(f"{samples} samples cannot make {k} groups") }

    #: The samples are read where they already are when they lie in one run:
    #: copying them to the host as a list and back again cost 18 ms of a
    #: 212 ms run at 100,000 samples of fifty features, for nothing. The host
    #: copy is made only if an empty group needs it.
    let in_place = _KMEANS_IN_PLACE and data.is_continuous() and samples * width > 0
    var host_values: list[float]? = null
    let held_values = if in_place {
        data.storage.view(data.offset, samples * width)
    } else {
        let copy = Buffer(max(1, samples * width))
        let values = data.values()
        copy.upload(values)
        host_values = values
        copy
    }
    let held_centres = Buffer(max(1, k * width))
    let held_labels = Buffer(max(1, samples))
    let held_nearest = Buffer(max(1, samples))
    let gathered = Buffer(k * width + k + 1)
    let best_held = Buffer(max(1, samples))
    defer { held_values.free() }
    defer { held_centres.free() }
    defer { held_labels.free() }
    defer { held_nearest.free() }
    defer { gathered.free() }

    let staged = _KMEANS_STAGED and k <= 16 and k * width <= 4096
    let by_column = _KMEANS_COLUMNS and k <= 16
    #: The totals read the samples a column at a time, so they read them from
    #: a copy laid out that way, made once for every attempt and iteration.
    let held_columns: Buffer? = if by_column and _KMEANS_TRANSPOSED and width > 1 {
        let copy = Buffer(samples * width)
        kernel_for("kmeans_transpose").dispatch_async(
            [held_values, copy],
            samples * width,
            [samples, width]
        )
        copy
    } else {
        null
    }
    defer {
        if held_columns is not null { held_columns.free() }
    }
    var state = if seed == 0 { 1 } else { seed }
    var best_centres: list[float] = []
    var best_cost = 1e300

    for _attempt in 0..max(attempts, 1) {
        var centres: list[float] = []
        state = next_random(state)
        let first = state % samples
        for value in _sample_row(held_values, host_values, first, width) { centres.push(value) }

        #: How far each sample is from the nearest centre chosen so far, kept
        #: as centres are added rather than worked out again each time. Redoing
        #: it means every sample measured against every centre at every step,
        #: which is the seeding costing `k` times what it needs to -- and for
        #: the eight groups an image is usually quantised into, that is most of
        #: the work before the first iteration even starts.
        held_centres.upload(centres)
        kernel_for("kmeans_seed").dispatch_async(
            [held_values, held_centres, held_nearest],
            samples,
            [samples, width, 0, 1]
        )
        var distances = held_nearest.download()
        while centres.len() // width < k {
            var chosen = samples - 1
            if flags & KMEANS_PP_CENTERS != 0 {
                let total = _total(distances)
                if total > 0.0 {
                    state = next_random(state)
                    let target = float(state % 1000000) / 1000000.0 * total
                    chosen = _first_reaching(distances, target, samples - 1)
                } else {
                    state = next_random(state)
                    chosen = state % samples
                }
            } else {
                state = next_random(state)
                chosen = state % samples
            }
            let added = centres.len() // width
            for value in _sample_row(held_values, host_values, chosen, width) { centres.push(value) }
            if centres.len() // width < k {
                held_centres.upload(centres)
                kernel_for("kmeans_seed").dispatch_async(
                    [held_values, held_centres, held_nearest],
                    samples,
                    [samples, width, added, 0]
                )
                distances = held_nearest.download()
            }
        }

        var labels: list[int] = []
        var labels_are_current = false
        for _round in 0..iterations {
            var moved = 0.0
            #: Assignment and the totals it feeds both run on the device; only
            #: the centres, the totals and the counts come back, which is
            #: `k * width + k + 1` numbers however many samples there are.
            held_centres.upload(centres)
            if staged and _KMEANS_TILED {
                kernel_for("kmeans_assign_tiled").dispatch_async(
                    [held_values, held_centres, held_labels, held_nearest],
                    (samples + 255) // 256 * 256,
                    [samples, width, k],
                    group_size: 256
                )
            } elif staged {
                kernel_for("kmeans_assign_staged").dispatch_async(
                    [held_values, held_centres, held_labels, held_nearest],
                    samples,
                    [samples, width, k],
                    group_size: 256
                )
            } else {
                kernel_for("kmeans_assign").dispatch_async(
                    [held_values, held_centres, held_labels, held_nearest],
                    samples,
                    [samples, width, k]
                )
            }
            if by_column {
                kernel_for("kmeans_gather_columns").dispatch_async(
                    [held_columns ?? held_values, held_labels, held_nearest, gathered],
                    (width + 1) * 256,
                    [samples, width, k, if held_columns is not null { 1 } else { 0 }],
                    group_size: 256
                )
            } else {
                kernel_for("kmeans_gather").dispatch_async(
                    [held_values, held_labels, held_nearest, gathered],
                    (k * width + k + 1) * 256,
                    [samples, width, k],
                    group_size: 256
                )
            }

            var sums = gathered.download()
            var counts: list[int] = []
            for centre in 0..k { counts.push(int(sums[k * width + centre] + 0.5)) }
            labels_are_current = false
            for centre in 0..k {
                if counts[centre] == 0 {
                    #: An empty group is given the sample furthest from
                    #: everything, which is the one the grouping explains worst.
                    #: The only place the assignment itself is needed here, and
                    #: it almost never happens, so the labels come back only
                    #: when it does.
                    if not labels_are_current {
                        labels = []
                        for value in held_labels.download() { labels.push(int(value + 0.5)) }
                        labels_are_current = true
                    }
                    let values = host_values ?? data.values()
                    host_values = values
                    var furthest = 0
                    var worst = -1.0
                    for sample in 0..samples {
                        let gap = squared_to_centre(values, sample, centres, labels[sample], width)
                        if gap > worst {
                            worst = gap
                            furthest = sample
                        }
                    }
                    for index in 0..width {
                        centres[centre * width + index] = values[furthest * width + index]
                    }
                    continue
                }
                for index in 0..width {
                    let updated = sums[centre * width + index] / float(counts[centre])
                    let gap = updated - centres[centre * width + index]
                    moved += gap * gap
                    centres[centre * width + index] = updated
                }
            }
            if moved <= epsilon * epsilon { break }
        }

        #: The device totals the distances in float32, which is precise enough
        #: to steer the iterations but not to report: at a cost of a couple of
        #: million the last digits are gone. Summed once more here, in full
        #: precision, so what comes back is what the host used to return.
        let cost = _total(held_nearest.download())

        if cost < best_cost {
            best_cost = cost
            #: The labels stay on the device: they are already the floats a
            #: label matrix holds, so the best attempt's are copied aside there
            #: rather than read back, turned into integers and sent again.
            kernel_for("kmeans_copy").dispatch_async([held_labels, best_held], samples, [samples])
            best_centres = [value for value in centres]
        }
    }

    let labels_out = Mat(samples, 1, CV_MAKETYPE(CV_32S, 1), storage: best_held)
    return (
        best_cost,
        labels_out,
        Mat.from_list(k, width, CV_MAKETYPE(CV_32F, 1), best_centres)
    )
}

#: One sample's values: from the host copy when there is one, otherwise read
#: off the device.
fn _sample_row(held: Buffer, host: list[float]?, sample: int, width: int) -> list[float] {
    if host is not null {
        var out: list[float] = []
        for index in 0..width { out.push(host[sample * width + index]) }
        return out
    }
    return held.download(sample * width, width)
}

#: The sum of `values`, in order and in double precision.
fn _total(values: list[float]) -> float {
    var total = 0.0
    for value in values { total += value }
    return total
}

#: The first index at which the running sum of `values` reaches `target`, or
#: `otherwise` if it never does.
fn _first_reaching(values: list[float], target: float, otherwise: int) -> int {
    var running = 0.0
    for index in 0..values.len() {
        running += values[index]
        if running >= target { return index }
    }
    return otherwise
}

fn next_random(state: int) -> int {
    #: A small linear congruential generator, masked to stay inside the range
    #: Jaithon's integers trap at.
    return state *% 6364136223846793 +% 1442695040888963 & 0xFFFFFFFFFFFF
}

#: The directions a set of samples varies in most.
#:
#: The mean is subtracted, the covariance formed, and its eigenvectors kept in
#: order of how much variance each explains. `project` maps samples into that
#: basis and `back_project` maps them out again.
pub class PCA {
    #: The mean of the samples, as one row.
    pub var mean: Mat
    #: The kept directions, one per row.
    pub var eigenvectors: Mat
    #: How much variance each direction explains.
    pub var eigenvalues: list[float]

    pub fn init(self, data: Mat, components: int = 0) {
        data.flush()
        let samples = data.rows
        let width = data.cols * data.channels()
        if samples < 2 {
            throw ValueError("a principal component analysis needs at least two samples")
        }
        var centre: list[float] = []
        var covariance: list[float] = []
        #: A dataset of any size has its covariance formed on the device: the
        #: host loop below is samples times width squared multiply-adds, 650 ms
        #: of a 932 ms fit at 100,000 samples of fifty features. The device
        #: works in float32 rather than double, which the covariance is rounded
        #: to on its way into the eigen solver anyway.
        let scattered = if on_device(samples * width * width) { device_scatter(data) } else { null }
        if scattered is not null {
            let (found_centre, scatter) = scattered
            centre = found_centre
            for value in scatter { covariance.push(value / float(samples - 1)) }
        } else {
            let values = data.values()
            for index in 0..width {
                var total = 0.0
                for sample in 0..samples { total += values[sample * width + index] }
                centre.push(total / float(samples))
            }
            for _index in 0..width * width { covariance.push(0.0) }
            for sample in 0..samples {
                for i in 0..width {
                    let a = values[sample * width + i] - centre[i]
                    for j in 0..width {
                        covariance[i * width + j] += a * (values[sample * width + j] - centre[j])
                    }
                }
            }
            for index in 0..width * width { covariance[index] /= float(samples - 1) }
        }

        let (found, vectors) = symmetric_eigen(
            Mat.from_list(width, width, CV_MAKETYPE(CV_32F, 1), covariance)
        )
        let kept = if components > 0 { min(components, width) } else { width }
        var rows: list[float] = []
        var values_kept: list[float] = []
        let flat = vectors.values()
        for index in 0..kept {
            values_kept.push(found[index])
            for column in 0..width { rows.push(flat[index * width + column]) }
        }

        self.mean = Mat.from_list(1, width, CV_MAKETYPE(CV_32F, 1), centre)
        self.eigenvectors = Mat.from_list(kept, width, CV_MAKETYPE(CV_32F, 1), rows)
        self.eigenvalues = values_kept
    }

    #: Map samples into the kept basis.
    pub fn project(self, data: Mat) -> Mat {
        let width = self.mean.cols
        let centre = self.mean.to_list()
        if on_device(data.rows * width * self.eigenvectors.rows) and data.cols * data.channels() == width {
            let projected = device_centred_product(data, centre, self.eigenvectors)
            if projected is not null { return projected }
        }
        let values = data.values()
        var centred: list[float] = []
        for sample in 0..data.rows {
            for index in 0..width { centred.push(values[sample * width + index] - centre[index]) }
        }
        return matmul(
            Mat.from_list(data.rows, width, CV_MAKETYPE(CV_32F, 1), centred),
            transpose(self.eigenvectors)
        )
    }

    #: Map samples back out of the kept basis.
    pub fn back_project(self, data: Mat) -> Mat {
        let restored = matmul(data, self.eigenvectors)
        let width = self.mean.cols
        let centre = self.mean.to_list()
        let values = restored.values()
        var out: list[float] = []
        for sample in 0..restored.rows {
            for index in 0..width { out.push(values[sample * width + index] + centre[index]) }
        }
        return Mat.from_list(restored.rows, width, CV_MAKETYPE(CV_32F, 1), out)
    }
}
