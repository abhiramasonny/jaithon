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

    var state = if seed == 0 { 1 } else { seed }
    var best_labels: list[int] = []
    var best_centres: list[float] = []
    var best_cost = 1e300

    for _attempt in 0..max(attempts, 1) {
        var centres: list[float] = []
        state = next_random(state)
        let first = state % samples
        for index in 0..width { centres.push(values[first * width + index]) }

        while centres.len() // width < k {
            var distances: list[float] = []
            var total = 0.0
            for sample in 0..samples {
                var nearest = 1e300
                for centre in 0..centres.len() // width {
                    let gap = squared_to_centre(values, sample, centres, centre, width)
                    if gap < nearest { nearest = gap }
                }
                distances.push(nearest)
                total += nearest
            }
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
            for index in 0..width { centres.push(values[chosen * width + index]) }
        }

        var labels: list[int] = []
        for _index in 0..samples { labels.push(0) }
        var cost = 0.0
        for _round in 0..iterations {
            var moved = 0.0
            cost = 0.0
            for sample in 0..samples {
                var nearest = 1e300
                var pick = 0
                for centre in 0..k {
                    let gap = squared_to_centre(values, sample, centres, centre, width)
                    if gap < nearest {
                        nearest = gap
                        pick = centre
                    }
                }
                labels[sample] = pick
                cost += nearest
            }

            var sums: list[float] = []
            var counts: list[int] = []
            for _index in 0..k * width { sums.push(0.0) }
            for _index in 0..k { counts.push(0) }
            for sample in 0..samples {
                let label = labels[sample]
                counts[label] += 1
                for index in 0..width {
                    sums[label * width + index] += values[sample * width + index]
                }
            }
            for centre in 0..k {
                if counts[centre] == 0 {
                    #: An empty group is given the sample furthest from
                    #: everything, which is the one the grouping explains worst.
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

        if cost < best_cost {
            best_cost = cost
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
