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
from ..core.linalg import matmul, symmetric_eigen, transpose
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

let SOURCE = """#include <metal_stdlib>
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
    let values = data.values()

    #: One set of device buffers for every attempt and every iteration: the
    #: samples never change, and the rest are `k`-sized.
    let held_values = Buffer(max(1, samples * width))
    held_values.upload(values)
    let held_centres = Buffer(max(1, k * width))
    let held_labels = Buffer(max(1, samples))
    let held_nearest = Buffer(max(1, samples))
    let gathered = Buffer(k * width + k + 1)
    defer { held_values.free() }
    defer { held_centres.free() }
    defer { held_labels.free() }
    defer { held_nearest.free() }
    defer { gathered.free() }

    var state = if seed == 0 { 1 } else { seed }
    var best_labels: list[int] = []
    var best_centres: list[float] = []
    var best_cost = 1e300

    for _attempt in 0..max(attempts, 1) {
        var centres: list[float] = []
        state = next_random(state)
        let first = state % samples
        for index in 0..width { centres.push(values[first * width + index]) }

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
            var total = 0.0
            for sample in 0..samples { total += distances[sample] }
            var chosen = samples - 1
            if flags & KMEANS_PP_CENTERS != 0 and total > 0.0 {
                state = next_random(state)
                let target = float(state % 1000000) / 1000000.0 * total
                var running = 0.0
                for sample in 0..samples {
                    running += distances[sample]
                    if running >= target {
                        chosen = sample
                        break
                    }
                }
            } else {
                state = next_random(state)
                chosen = state % samples
            }
            let added = centres.len() // width
            for index in 0..width { centres.push(values[chosen * width + index]) }
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
        for _index in 0..samples { labels.push(0) }
        var cost = 0.0
        var labels_are_current = false
        for _round in 0..iterations {
            var moved = 0.0
            cost = 0.0
            #: Assignment and the totals it feeds both run on the device; only
            #: the centres, the totals and the counts come back, which is
            #: `k * width + k + 1` numbers however many samples there are.
            held_centres.upload(centres)
            kernel_for("kmeans_assign").dispatch_async(
                [held_values, held_centres, held_labels, held_nearest],
                samples,
                [samples, width, k]
            )
            kernel_for("kmeans_gather").dispatch_async(
                [held_values, held_labels, held_nearest, gathered],
                (k * width + k + 1) * 256,
                [samples, width, k],
                group_size: 256
            )

            var sums = gathered.download()
            var counts: list[int] = []
            for centre in 0..k { counts.push(int(sums[k * width + centre] + 0.5)) }
            cost = sums[k * width + k]
            labels_are_current = false
            for centre in 0..k {
                if counts[centre] == 0 {
                    #: An empty group is given the sample furthest from
                    #: everything, which is the one the grouping explains worst.
                    #: The only place the assignment itself is needed here, and
                    #: it almost never happens, so the labels come back only
                    #: when it does.
                    if not labels_are_current {
                        let read = held_labels.download()
                        for sample in 0..samples { labels[sample] = int(read[sample] + 0.5) }
                        labels_are_current = true
                    }
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
        let settled = held_nearest.download()
        cost = 0.0
        for sample in 0..samples { cost += settled[sample] }

        if cost < best_cost {
            best_cost = cost
            if not labels_are_current {
                let read = held_labels.download()
                for sample in 0..samples { labels[sample] = int(read[sample] + 0.5) }
            }
            best_labels = [label for label in labels]
            best_centres = [value for value in centres]
        }
    }

    var label_values: list[float] = []
    for label in best_labels { label_values.push(float(label)) }
    return (
        best_cost,
        Mat.from_list(samples, 1, CV_MAKETYPE(CV_32S, 1), label_values),
        Mat.from_list(k, width, CV_MAKETYPE(CV_32F, 1), best_centres)
    )
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
        let values = data.values()

        var centre: list[float] = []
        for index in 0..width {
            var total = 0.0
            for sample in 0..samples { total += values[sample * width + index] }
            centre.push(total / float(samples))
        }

        var covariance: list[float] = []
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
        let values = data.values()
        let centre = self.mean.to_list()
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
