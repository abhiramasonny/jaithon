#: Interactive foreground extraction: `grab_cut` (Rother, Kolmogorov and
#: Blake, 2004).
#:
#: A rectangle -- or a rough mask, for a second pass -- seeds two Gaussian
#: mixtures over colour, one for the background and one for the foreground.
#: Every pixel becomes a node in a graph, wired to a source and a sink by how
#: well its colour fits each mixture and to its eight neighbours by how close
#: their colours are; the minimum cut of that graph is the new segmentation,
#: and refitting the mixtures to it and cutting again is what "iter_count"
#: repeats.
#:
#: ```
#: let mask = Mat.zeros(image.rows, image.cols, CV_8UC1)
#: let (segmented, bgd_model, fgd_model) =
#:     grab_cut(image, mask, Rect(20, 15, 80, 100), Mat(0, 0, CV_64FC1), Mat(0, 0, CV_64FC1), 5)
#: ```
import std.math
from ..core.cluster import kmeans
from ..core.mat import Mat
from ..core.types import CV_32F, CV_64FC1, CV_8U, CV_MAKETYPE, Rect

#: Definitely background.
pub let GC_BGD = 0
#: Definitely foreground.
pub let GC_FGD = 1
#: Probably background, free to move during the cut.
pub let GC_PR_BGD = 2
#: Probably foreground, free to move during the cut.
pub let GC_PR_FGD = 3

#: Seed the mask from `rect`: background outside it, probable foreground in.
pub let GC_INIT_WITH_RECT = 0
#: Take the mask as the caller already set it.
pub let GC_INIT_WITH_MASK = 1
#: Refine an existing mask and model, refitting the mixtures as usual.
pub let GC_EVAL = 2
#: Refine an existing mask without refitting the mixtures -- one graph cut
#: under the model exactly as it stood.
pub let GC_EVAL_FREEZE_MODEL = 3

let GMM_COMPONENTS = 5
#: One weight, a three-element mean, then a row-major three by three
#: covariance.
let GMM_ENTRY = 13
let GMM_MODEL_SIZE = GMM_COMPONENTS * GMM_ENTRY

#: A directed flow network built one arc pair at a time. `max_flow` finds the
#: maximum flow from `source` to `sink` by Dinic's algorithm: a breadth-first
#: search layers every node by its distance from the source, and a
#: depth-first search finds a blocking flow inside that layering before the
#: layers are rebuilt, until the sink falls out of reach entirely.
#: `source_side` then reads the minimum cut back out of what max flow left
#: behind, by the max-flow min-cut theorem.
#:
#: Arcs are always added in pairs, at indices `k` and `k ^ 1`, so each is the
#: other's reverse. A one-way link -- every terminal link in GrabCut's graph
#: -- pairs a real capacity with a reverse of zero. A two-way link -- every
#: neighbour link -- pairs two real, equal capacities, which is the standard
#: reduction of an *undirected* edge of weight `w`: sending flow one way
#: raises how much the residual graph allows back the other, exactly
#: cancelling, so the two directions always sum to `w`.
pub class FlowGraph {
    #: How many nodes the graph has, source and sink included.
    pub let node_count: int
    var adjacency: list[list[int]]
    var to: list[int]
    var cap: list[float]

    pub fn init(self, node_count: int) {
        if node_count <= 0 { throw ValueError("a flow graph needs at least one node") }
        self.node_count = node_count
        var adjacency: list[list[int]] = []
        for _index in 0..node_count { adjacency.push([]) }
        self.adjacency = adjacency
        self.to = []
        self.cap = []
    }

    #: Add an arc `u -> v` with residual capacity `forward`, paired with its
    #: reverse `v -> u` at `backward`.
    pub fn add_edge(self, u: int, v: int, forward: float, backward: float) -> void {
        self.to.push(v)
        self.cap.push(forward)
        self.adjacency[u].push(self.to.len() - 1)
        self.to.push(u)
        self.cap.push(backward)
        self.adjacency[v].push(self.to.len() - 1)
    }

    fn bfs_levels(self, source: int, sink: int) -> list[int] {
        var level: list[int] = []
        for _index in 0..self.node_count { level.push(-1) }
        level[source] = 0
        var queue = [source]
        var head = 0
        while head < queue.len() {
            let u = queue[head]
            head += 1
            for arc in self.adjacency[u] {
                if self.cap[arc] <= 0.0 { continue }
                let v = self.to[arc]
                if level[v] >= 0 { continue }
                level[v] = level[u] + 1
                queue.push(v)
            }
        }
        return level
    }

    #: Push flow along one source-to-sink path of the level graph. Each
    #: node's entry in `iter` remembers how far it already searched, so a
    #: branch that led nowhere is never walked twice within one phase.
    fn send_flow(
        self,
        u: int,
        sink: int,
        pushed: float,
        level: list[int],
        iter: list[int]
    ) -> float {
        if u == sink { return pushed }
        while iter[u] < self.adjacency[u].len() {
            let arc = self.adjacency[u][iter[u]]
            let v = self.to[arc]
            if self.cap[arc] > 0.0 and level[v] == level[u] + 1 {
                let sent = self.send_flow(v, sink, min(pushed, self.cap[arc]), level, iter)
                if sent > 0.0 {
                    self.cap[arc] -= sent
                    self.cap[arc ^ 1] += sent
                    return sent
                }
            }
            iter[u] += 1
        }
        return 0.0
    }

    #: The maximum flow from `source` to `sink`. Leaves the residual
    #: capacities in place, which is what `source_side` reads the cut from.
    pub fn max_flow(self, source: int, sink: int) -> float {
        if source < 0 or source >= self.node_count or sink < 0 or sink >= self.node_count {
            throw ValueError(f"source and sink have to be inside 0..{self.node_count}")
        }
        var total = 0.0
        loop {
            let level = self.bfs_levels(source, sink)
            if level[sink] < 0 { break }
            var iter: list[int] = []
            for _index in 0..self.node_count { iter.push(0) }
            loop {
                let pushed = self.send_flow(source, sink, 1e300, level, iter)
                if pushed <= 0.0 { break }
                total += pushed
            }
        }
        return total
    }

    #: Which nodes stay reachable from `source` over positive residual
    #: capacity once no more flow can move -- the source side of a minimum
    #: cut. Call only after `max_flow`.
    pub fn source_side(self, source: int) -> list[bool] {
        var reached: list[bool] = []
        for _index in 0..self.node_count { reached.push(false) }
        reached[source] = true
        var queue = [source]
        var head = 0
        while head < queue.len() {
            let u = queue[head]
            head += 1
            for arc in self.adjacency[u] {
                if self.cap[arc] <= 0.0 { continue }
                let v = self.to[arc]
                if reached[v] { continue }
                reached[v] = true
                queue.push(v)
            }
        }
        return reached
    }
}

#: One Gaussian mixture over RGB, five components, fit and read the way
#: OpenCV's own `GMM` is, packed into the flat model `grab_cut`'s caller
#: keeps between calls.
#:
#: The packing groups by component rather than by field -- thirteen numbers
#: each, in a row: a weight, a three-element mean, then a row-major
#: covariance -- so one component can be read off without decoding the rest.
#: That is not OpenCV's own byte order (which groups by field: five weights,
#: then fifteen means, then forty-five covariance entries), but nothing here
#: reads another implementation's model, so what matters is that the layout
#: is fixed and a model this returns reads back the way it was written.
class GMM {
    var coefs: list[float]
    var mean: list[list[float]]
    var cov: list[list[float]]
    var inv_cov: list[list[float]]
    var det: list[float]
    #: `1 / sqrt(det)`, which every density evaluation wants and none of
    #: them changes. Working it out per pixel per component was a square
    #: root and a divide a hundred thousand times an iteration.
    var scale: list[float]
    var sums: list[list[float]]
    var prods: list[list[float]]
    var counts: list[int]
    var total: int

    pub fn init(self, model: list[float]) {
        var coefs: list[float] = []
        var mean: list[list[float]] = []
        var cov: list[list[float]] = []
        for ci in 0..GMM_COMPONENTS {
            let base = ci * GMM_ENTRY
            coefs.push(model[base])
            mean.push([model[base + 1], model[base + 2], model[base + 3]])
            var one: list[float] = []
            for index in 0..9 { one.push(model[base + 4 + index]) }
            cov.push(one)
        }
        self.coefs = coefs
        self.mean = mean
        self.cov = cov
        var inv_cov: list[list[float]] = []
        var det: list[float] = []
        for _ci in 0..GMM_COMPONENTS {
            inv_cov.push([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            det.push(0.0)
        }
        self.inv_cov = inv_cov
        self.det = det
        var scale: list[float] = []
        for _index in 0..GMM_COMPONENTS { scale.push(0.0) }
        self.scale = scale
        for ci in 0..GMM_COMPONENTS {
            if self.coefs[ci] > 0.0 { self.refresh(ci) }
        }
        self.start_learning()
    }

    #: Flatten back into the sixty-five numbers a caller keeps.
    pub fn to_model(self) -> list[float] {
        var out: list[float] = []
        for ci in 0..GMM_COMPONENTS {
            out.push(self.coefs[ci])
            for value in self.mean[ci] { out.push(value) }
            for value in self.cov[ci] { out.push(value) }
        }
        return out
    }

    #: Determinant and inverse of one component's covariance, by the closed
    #: form for a three by three: cofactors over the determinant. A
    #: covariance that has collapsed to a point -- every sample the
    #: component drew landed on the same colour -- is nudged off the
    #: diagonal first, the way OpenCV avoids the same singularity.
    fn refresh(self, ci: int) -> void {
        var c = self.cov[ci]
        var determinant = cofactor_det(c)
        if determinant <= 1e-12 {
            c = [c[0] + 0.01, c[1], c[2], c[3], c[4] + 0.01, c[5], c[6], c[7], c[8] + 0.01]
            self.cov[ci] = c
            determinant = cofactor_det(c)
        }
        self.det[ci] = determinant
        self.scale[ci] = 1.0 / math.sqrt(determinant)
        let inv_d = 1.0 / determinant
        self.inv_cov[ci] = [
            (c[4] * c[8] - c[5] * c[7]) * inv_d,
            (c[2] * c[7] - c[1] * c[8]) * inv_d,
            (c[1] * c[5] - c[2] * c[4]) * inv_d,
            (c[5] * c[6] - c[3] * c[8]) * inv_d,
            (c[0] * c[8] - c[2] * c[6]) * inv_d,
            (c[2] * c[3] - c[0] * c[5]) * inv_d,
            (c[3] * c[7] - c[4] * c[6]) * inv_d,
            (c[1] * c[6] - c[0] * c[7]) * inv_d,
            (c[0] * c[4] - c[1] * c[3]) * inv_d,
        ]
    }

    #: The Gaussian density of component `ci` at `color`, without the
    #: `(2 pi)^-3/2` normaliser -- OpenCV's own `GMM::operator()` leaves it
    #: out too, and since every use here only compares or sums these across
    #: the same five components, the missing constant is common to all of
    #: them and never changes which one wins or which side of the cut a
    #: pixel lands on.
    #: Everything but the exponential, which is where the arithmetic is.
    #:
    #: Split out because a call anywhere in a body stops the tier compiling
    #: it, and `math.exp` is a call that cannot be written out. Keeping the
    #: nine multiply-adds and the six list reads in their own body lets those
    #: compile, and leaves the interpreted part two calls long.
    fn component_exponent(self, ci: int, color: list[float]) -> float {
        let m = self.mean[ci]
        let d0 = color[0] - m[0]
        let d1 = color[1] - m[1]
        let d2 = color[2] - m[2]
        let inv = self.inv_cov[ci]
        let a = inv[0] * d0 + inv[1] * d1 + inv[2] * d2
        let b = inv[3] * d0 + inv[4] * d1 + inv[5] * d2
        let e = inv[6] * d0 + inv[7] * d1 + inv[8] * d2
        return -0.5 * (d0 * a + d1 * b + d2 * e)
    }

    fn component_density(self, ci: int, color: list[float]) -> float {
        if self.coefs[ci] <= 0.0 { return 0.0 }
        return self.scale[ci] * math.exp(self.component_exponent(ci, color))
    }

    #: The mixture density at `color`, weighted by each component's mixing
    #: coefficient -- what a terminal link's cost is the negative log of.
    pub fn density(self, color: list[float]) -> float {
        var total = 0.0
        for ci in 0..GMM_COMPONENTS { total += self.coefs[ci] * self.component_density(ci, color) }
        return total
    }

    #: Which component `color` is most likely under, ignoring the mixing
    #: weights -- the assignment step of the refit `learn_gmms` runs.
    pub fn which_component(self, color: list[float]) -> int {
        var best = 0
        var best_density = 0.0
        for ci in 0..GMM_COMPONENTS {
            let density = self.component_density(ci, color)
            if density > best_density {
                best_density = density
                best = ci
            }
        }
        return best
    }

    #: Clear the running sums a refit accumulates through `add_sample`.
    pub fn start_learning(self) -> void {
        var sums: list[list[float]] = []
        var prods: list[list[float]] = []
        var counts: list[int] = []
        for _ci in 0..GMM_COMPONENTS {
            sums.push([0.0, 0.0, 0.0])
            prods.push([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            counts.push(0)
        }
        self.sums = sums
        self.prods = prods
        self.counts = counts
        self.total = 0
    }

    #: Fold one pixel into component `ci`'s running mean and covariance sums.
    pub fn add_sample(self, ci: int, color: list[float]) -> void {
        let sums = self.sums[ci]
        sums[0] += color[0]
        sums[1] += color[1]
        sums[2] += color[2]
        let prods = self.prods[ci]
        for i in 0..3 {
            for j in 0..3 { prods[i * 3 + j] += color[i] * color[j] }
        }
        self.counts[ci] += 1
        self.total += 1
    }

    #: Turn the running sums into a mean, covariance and mixing weight per
    #: component. A component nobody was assigned to drops its weight to
    #: zero and keeps whatever mean and covariance it last had, rather than
    #: being reseeded -- the same choice OpenCV's `GMM::endLearning` makes.
    pub fn finish_learning(self) -> void {
        for ci in 0..GMM_COMPONENTS {
            let n = self.counts[ci]
            if n == 0 {
                self.coefs[ci] = 0.0
                continue
            }
            self.coefs[ci] = float(n) / float(self.total)
            let sums = self.sums[ci]
            let mean = [sums[0] / float(n), sums[1] / float(n), sums[2] / float(n)]
            self.mean[ci] = mean
            let prods = self.prods[ci]
            var cov: list[float] = []
            for i in 0..3 {
                for j in 0..3 { cov.push(prods[i * 3 + j] / float(n) - mean[i] * mean[j]) }
            }
            self.cov[ci] = cov
            self.refresh(ci)
        }
    }
}

fn cofactor_det(c: list[float]) -> float {
    return c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) +
        c[2] * (c[3] * c[7] - c[4] * c[6])
}

#: Everything outside `rect` is background, everything inside is probable
#: foreground -- OpenCV's own `initMaskWithRect`, clamped the same way to
#: whatever of `rect` actually overlaps the image.
fn rect_mask(rows: int, cols: int, rect: Rect) -> list[float] {
    let x = max(rect.x, 0)
    let y = max(rect.y, 0)
    let width = min(rect.width, cols - x)
    let height = min(rect.height, rows - y)
    if width <= 0 or height <= 0 {
        throw ValueError(f"{rect} does not overlap a {rows}x{cols} image")
    }
    var values: list[float] = []
    for py in 0..rows {
        for px in 0..cols {
            let inside = px >= x and px < x + width and py >= y and py < y + height
            values.push(if inside { float(GC_PR_FGD) } else { float(GC_BGD) })
        }
    }
    return values
}

fn check_mask(mask: Mat, rows: int, cols: int) -> void {
    if mask.rows != rows or mask.cols != cols {
        throw ValueError(
            f"the mask has to be {rows}x{cols} to match the image, got {mask.rows}x{mask.cols}"
        )
    }
    if mask.channels() != 1 { throw ValueError("the mask has to be single-channel") }
    for value in mask.values() {
        let label = int(value)
        if label != GC_BGD and label != GC_FGD and label != GC_PR_BGD and label != GC_PR_FGD {
            throw ValueError(f"mask value {label} is none of GC_BGD, GC_FGD, GC_PR_BGD, GC_PR_FGD")
        }
    }
}

fn model_from(model: Mat) -> list[float] {
    if model.empty() {
        var zeros: list[float] = []
        for _index in 0..GMM_MODEL_SIZE { zeros.push(0.0) }
        return zeros
    }
    if model.rows != 1 or model.cols != GMM_MODEL_SIZE {
        throw ValueError(
            f"a GrabCut model has to be 1x{GMM_MODEL_SIZE}, got {model.rows}x{model.cols}"
        )
    }
    return model.to_list()
}

fn fit_initial_gmm(colors: list[float]) -> GMM {
    let count = colors.len() // 3
    if count == 0 {
        throw ValueError(
            "grab_cut needs at least one background and one foreground pixel in the mask"
        )
    }
    var zeros: list[float] = []
    for _index in 0..GMM_MODEL_SIZE { zeros.push(0.0) }
    let gmm = GMM(zeros)
    #: k-means needs at least as many samples as components; a mask with a
    #: sliver of one side just grows fewer, emptier components instead.
    let k = min(GMM_COMPONENTS, count)
    let samples = Mat.from_list(count, 3, CV_MAKETYPE(CV_32F, 1), colors)
    let (_cost, labels, _centres) = kmeans(samples, k, 1, 10, 0.0)
    let label_values = labels.to_list()
    gmm.start_learning()
    for index in 0..count {
        let ci = int(label_values[index])
        gmm.add_sample(ci, [colors[index * 3], colors[index * 3 + 1], colors[index * 3 + 2]])
    }
    gmm.finish_learning()
    return gmm
}

#: Split the image into its background and foreground samples by the current
#: mask, and fit a fresh mixture to each by k-means, the way OpenCV's own
#: `initGMMs` seeds a mixture before the first cut.
fn init_gmms(colors: list[float], mask_values: list[float], pixels: int) -> tuple[GMM, GMM] {
    var bgd_colors: list[float] = []
    var fgd_colors: list[float] = []
    for index in 0..pixels {
        let label = int(mask_values[index])
        let base = index * 3
        if label == GC_BGD or label == GC_PR_BGD {
            bgd_colors.push(colors[base])
            bgd_colors.push(colors[base + 1])
            bgd_colors.push(colors[base + 2])
        } else {
            fgd_colors.push(colors[base])
            fgd_colors.push(colors[base + 1])
            fgd_colors.push(colors[base + 2])
        }
    }
    return (fit_initial_gmm(bgd_colors), fit_initial_gmm(fgd_colors))
}

fn assign_components(
    colors: list[float],
    mask_values: list[float],
    pixels: int,
    bgd_gmm: GMM,
    fgd_gmm: GMM
) -> list[int] {
    var comps: list[int] = []
    for index in 0..pixels {
        let label = int(mask_values[index])
        let base = index * 3
        let color = [colors[base], colors[base + 1], colors[base + 2]]
        let ci = if label == GC_BGD or label == GC_PR_BGD {
            bgd_gmm.which_component(color)
        } else {
            fgd_gmm.which_component(color)
        }
        comps.push(ci)
    }
    return comps
}

fn learn_gmms(
    colors: list[float],
    mask_values: list[float],
    comps: list[int],
    pixels: int,
    bgd_gmm: GMM,
    fgd_gmm: GMM
) -> void {
    bgd_gmm.start_learning()
    fgd_gmm.start_learning()
    for index in 0..pixels {
        let label = int(mask_values[index])
        let base = index * 3
        let color = [colors[base], colors[base + 1], colors[base + 2]]
        if label == GC_BGD or label == GC_PR_BGD {
            bgd_gmm.add_sample(comps[index], color)
        } else {
            fgd_gmm.add_sample(comps[index], color)
        }
    }
    bgd_gmm.finish_learning()
    fgd_gmm.finish_learning()
}

fn squared_diff3(values: list[float], a: int, b: int) -> float {
    let dr = values[a] - values[b]
    let dg = values[a + 1] - values[b + 1]
    let db = values[a + 2] - values[b + 2]
    return dr * dr + dg * dg + db * db
}

#: The smoothness term's `beta`, set from the mean squared colour difference
#: between neighbours over the whole image, as the paper prescribes: large
#: where colour usually jumps between neighbours, small where it does not, so
#: the boundary term keeps its bite either way.
fn calc_beta(colors: list[float], rows: int, cols: int) -> float {
    var total = 0.0
    var count = 0
    for y in 0..rows {
        for x in 0..cols {
            let here = (y * cols + x) * 3
            if x > 0 {
                total += squared_diff3(colors, here, here - 3)
                count += 1
            }
            if x > 0 and y > 0 {
                total += squared_diff3(colors, here, here - cols * 3 - 3)
                count += 1
            }
            if y > 0 {
                total += squared_diff3(colors, here, here - cols * 3)
                count += 1
            }
            if x < cols - 1 and y > 0 {
                total += squared_diff3(colors, here, here - cols * 3 + 3)
                count += 1
            }
        }
    }
    if count == 0 or total <= 1e-12 { return 0.0 }
    return 1.0 / (2.0 * (total / float(count)))
}

#: Wire up the source, the sink and every pixel's eight-neighbourhood into a
#: `FlowGraph`. A probable pixel's terminal links carry the negative log
#: likelihood under each mixture; a pixel the caller pinned to `GC_BGD` or
#: `GC_FGD` gets a link so large the cut never crosses it instead. Neighbour
#: links only need adding once each -- looking backward and left covers every
#: pair exactly once -- and carry `gamma * exp(-beta * colour gap^2)`, scaled
#: down by root two on the diagonal for the longer step.
fn construct_graph(
    colors: list[float],
    mask_values: list[float],
    rows: int,
    cols: int,
    bgd_gmm: GMM,
    fgd_gmm: GMM,
    beta: float,
    gamma: float,
    lambda_: float
) -> FlowGraph {
    let pixels = rows * cols
    let source = pixels
    let sink = pixels + 1
    let graph = FlowGraph(pixels + 2)
    let diagonal_gamma = gamma / math.sqrt(2.0)

    for y in 0..rows {
        for x in 0..cols {
            let index = y * cols + x
            let base = index * 3
            let color = [colors[base], colors[base + 1], colors[base + 2]]
            let label = int(mask_values[index])

            if label == GC_PR_BGD or label == GC_PR_FGD {
                let from_source = -math.log(max(bgd_gmm.density(color), 1e-300))
                let to_sink = -math.log(max(fgd_gmm.density(color), 1e-300))
                graph.add_edge(source, index, from_source, 0.0)
                graph.add_edge(index, sink, to_sink, 0.0)
            } elif label == GC_BGD {
                graph.add_edge(index, sink, lambda_, 0.0)
            } else {
                graph.add_edge(source, index, lambda_, 0.0)
            }

            if x > 0 {
                let w = gamma * math.exp(-beta * squared_diff3(colors, base, base - 3))
                graph.add_edge(index, index - 1, w, w)
            }
            if x > 0 and y > 0 {
                let w = diagonal_gamma *
                    math.exp(-beta * squared_diff3(colors, base, base - cols * 3 - 3))
                graph.add_edge(index, index - cols - 1, w, w)
            }
            if y > 0 {
                let w = gamma * math.exp(-beta * squared_diff3(colors, base, base - cols * 3))
                graph.add_edge(index, index - cols, w, w)
            }
            if x < cols - 1 and y > 0 {
                let w = diagonal_gamma *
                    math.exp(-beta * squared_diff3(colors, base, base - cols * 3 + 3))
                graph.add_edge(index, index - cols + 1, w, w)
            }
        }
    }
    return graph
}

#: Cut the graph and relabel every probable pixel by which side it landed on.
#: A pixel already pinned to `GC_BGD` or `GC_FGD` never moves, whichever side
#: its huge terminal link left it on.
fn estimate_segmentation(graph: FlowGraph, mask_values: list[float], rows: int, cols: int) -> void {
    let pixels = rows * cols
    let source = pixels
    let sink = pixels + 1
    let _flow = graph.max_flow(source, sink)
    let source_side = graph.source_side(source)
    for index in 0..pixels {
        let label = int(mask_values[index])
        if label == GC_PR_BGD or label == GC_PR_FGD {
            mask_values[index] = if source_side[index] {
                float(GC_PR_FGD)
            } else {
                float(GC_PR_BGD)
            }
        }
    }
}

#: Segment `image` into foreground and background.
#:
#: `mode` picks how the mask starts: `GC_INIT_WITH_RECT` builds it from
#: `rect` and fits fresh colour mixtures; `GC_INIT_WITH_MASK` takes `mask` as
#: given and fits fresh mixtures from it; `GC_EVAL` refines whatever `mask`,
#: `bgd_model` and `fgd_model` an earlier call left, refitting the mixtures
#: as it goes; `GC_EVAL_FREEZE_MODEL` does the same but holds the mixtures
#: fixed, running one cut under exactly the model handed in.
#:
#: Returns the updated mask and the two models, each a `1 x 65` `CV_64F` `Mat`
#: -- pass `Mat(0, 0, CV_64FC1)` for a model that has not been fit yet.
pub fn grab_cut(
    image: Mat,
    mask: Mat,
    rect: Rect,
    bgd_model: Mat,
    fgd_model: Mat,
    iter_count: int,
    mode: int = GC_INIT_WITH_RECT
) -> tuple[Mat, Mat, Mat] {
    if image.channels() != 3 { throw ValueError("grab_cut needs a three-channel colour image") }
    let rows = image.rows
    let cols = image.cols
    if rows <= 0 or cols <= 0 { throw ValueError("grab_cut needs a non-empty image") }
    image.flush()
    let colors = image.values()
    let pixels = rows * cols

    var mask_values: list[float] = []
    if mode == GC_INIT_WITH_RECT {
        mask_values = rect_mask(rows, cols, rect)
    } elif mode == GC_INIT_WITH_MASK or mode == GC_EVAL or mode == GC_EVAL_FREEZE_MODEL {
        check_mask(mask, rows, cols)
        mask_values = mask.to_list()
    } else {
        throw ValueError(
            f"mode {mode} is none of GC_INIT_WITH_RECT, GC_INIT_WITH_MASK, GC_EVAL, GC_EVAL_FREEZE_MODEL"
        )
    }

    let (bgd_gmm, fgd_gmm) = if mode == GC_INIT_WITH_RECT or mode == GC_INIT_WITH_MASK {
        init_gmms(colors, mask_values, pixels)
    } else {
        if bgd_model.empty() or fgd_model.empty() {
            throw ValueError("GC_EVAL needs bgd_model and fgd_model carried over from an earlier call")
        }
        (GMM(model_from(bgd_model)), GMM(model_from(fgd_model)))
    }

    if iter_count <= 0 {
        return (
            Mat.from_list(rows, cols, CV_MAKETYPE(CV_8U, 1), mask_values),
            Mat.from_list(1, GMM_MODEL_SIZE, CV_64FC1, bgd_gmm.to_model()),
            Mat.from_list(1, GMM_MODEL_SIZE, CV_64FC1, fgd_gmm.to_model())
        )
    }

    let rounds = if mode == GC_EVAL_FREEZE_MODEL { 1 } else { iter_count }
    let gamma = 50.0
    let lambda_ = 9.0 * gamma
    let beta = calc_beta(colors, rows, cols)

    for _round in 0..rounds {
        if mode != GC_EVAL_FREEZE_MODEL {
            let comps = assign_components(colors, mask_values, pixels, bgd_gmm, fgd_gmm)
            learn_gmms(colors, mask_values, comps, pixels, bgd_gmm, fgd_gmm)
        }
        let graph = construct_graph(
            colors,
            mask_values,
            rows,
            cols,
            bgd_gmm,
            fgd_gmm,
            beta,
            gamma,
            lambda_
        )
        estimate_segmentation(graph, mask_values, rows, cols)
    }

    return (
        Mat.from_list(rows, cols, CV_MAKETYPE(CV_8U, 1), mask_values),
        Mat.from_list(1, GMM_MODEL_SIZE, CV_64FC1, bgd_gmm.to_model()),
        Mat.from_list(1, GMM_MODEL_SIZE, CV_64FC1, fgd_gmm.to_model())
    )
}
