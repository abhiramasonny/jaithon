#: Splitting an image into regions: `flood_fill`, `connected_components`,
#: `distance_transform`, `watershed`, and the integral image the first three
#: are often paired with.
#:
#: These are the label-propagating half of `imgproc`, and they are sequential
#: in a way the filters are not — a label found on one row decides the labels
#: on the next. They run on the host for that reason, and read the image
#: through the same mirror everything else writes to.
#:
#: ```
#: let (count, labels, stats, centroids) = connected_components_with_stats(mask)
#: for index in 1..count {
#:     if stats[index].area < 40 { continue }
#:     rectangle_rect(frame, stats[index].box, Scalar(0.0, 255.0, 0.0), 1)
#: }
#: ```
import std.math
from std.core import Printable
from ..core.mat import Mat, f32_bits
from ..core.types import CV_32F, CV_32S, CV_8U, CV_MAKETYPE, Point, Point2f, Rect, Scalar
from std.gpu import Buffer, Kernel

#: Fill only where the neighbour's value is close to the seed's.
pub const FLOODFILL_FIXED_RANGE = 1 << 16
#: Write into the mask rather than the image.
pub const FLOODFILL_MASK_ONLY = 1 << 17

#: City-block distance, where a diagonal step costs two.
pub const DIST_L1 = 1
#: Euclidean distance.
pub const DIST_L2 = 2
#: Chessboard distance, where a diagonal step costs one.
pub const DIST_C = 3

#: A three by three chamfer window.
pub const DIST_MASK_3 = 3
#: A five by five chamfer window, which is closer to the true distance.
pub const DIST_MASK_5 = 5

#: The area and the box a connected component occupies.
pub class ComponentStats: Printable {
    #: The component's bounding box.
    pub let box: Rect
    #: How many pixels it holds.
    pub let area: int

    pub fn init(self, box: Rect, area: int) {
        self.box = box
        self.area = area
    }

    pub fn to_str(self) -> str { return f"ComponentStats({self.box}, area={self.area})" }

    fn __str__(self) -> str { return self.to_str() }
}

fn neighbourhood(connectivity: int) -> tuple[list[int], list[int]] {
    if connectivity == 4 { return ([-1, 0, 0, 1], [0, -1, 1, 0]) }
    if connectivity == 8 {
        return ([-1, -1, -1, 0, 0, 1, 1, 1], [-1, 0, 1, -1, 1, -1, 0, 1])
    }
    throw ValueError(f"connectivity {connectivity} is neither four nor eight")
}

#: Fill a connected region of similar colour, starting from one pixel.
#:
#: `lo_diff` and `up_diff` say how far below and above a pixel a neighbour may
#: be and still join it. Without `FLOODFILL_FIXED_RANGE` the comparison is
#: against the pixel already in the region, which lets the fill drift across a
#: gradient; with it, every pixel is compared against the seed.
#:
#: Returns the number of pixels filled and the box they occupy, which is what
#: OpenCV returns as well.
let SOURCE = """#include <metal_stdlib>
using namespace metal;

/* Where a pixel falls in the order components are first met.
 *
 * Four-connected scanning is pixel by pixel, so that is the linear index.
 * Eight-connected is two by two blocks, because every foreground pixel in
 * such a block is connected to every other -- which is how OpenCV numbers
 * them, and the numbering is what a caller sees. */
inline uint scan_position(uint y, uint x, uint cols, uint connectivity) {
    if (connectivity == 4u) return y * cols + x;
    uint blocks = (cols + 1u) / 2u;
    return ((y / 2u) * blocks + (x / 2u)) * 4u + (y % 2u) * 2u + (x % 2u);
}

/* Every foreground pixel starts as its own component, named for where it is.
 * Background keeps zero, which no pixel can be confused with because the
 * names are one-based. */
kernel void cc_seed(device const float* src [[buffer(0)]],
                    device uint* labels [[buffer(1)]],
                    constant uint& count [[buffer(2)]],
                    constant uint& cols [[buffer(3)]],
                    constant uint& off [[buffer(4)]],
                    constant uint& step [[buffer(5)]],
                    uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint y = id / cols;
    uint x = id - y * cols;
    labels[id] = src[off + y * step + x] != 0.0f ? id + 1u : 0u;
}

/* One round of joining: a pixel that can see a smaller name than its own
 * tells its own component's root about it.
 *
 * Writing to the root rather than to the neighbour is what makes this
 * converge -- two halves of a component that meet anywhere pull their whole
 * chains together, instead of the news travelling one pixel a round. */
kernel void cc_propagate(device atomic_uint* labels [[buffer(0)]],
                         device atomic_uint* changed [[buffer(1)]],
                         constant uint& count [[buffer(2)]],
                         constant uint& rows [[buffer(3)]],
                         constant uint& cols [[buffer(4)]],
                         constant uint& connectivity [[buffer(5)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint mine = atomic_load_explicit(&labels[id], memory_order_relaxed);
    if (mine == 0u) return;
    int y = int(id / cols);
    int x = int(id - uint(y) * cols);
    uint best = mine;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dy == 0 && dx == 0) continue;
            if (connectivity == 4u && dy != 0 && dx != 0) continue;
            int ny = y + dy;
            int nx = x + dx;
            if (ny < 0 || nx < 0 || ny >= int(rows) || nx >= int(cols)) continue;
            uint other = atomic_load_explicit(&labels[uint(ny) * cols + uint(nx)],
                                              memory_order_relaxed);
            if (other != 0u && other < best) best = other;
        }
    }
    if (best >= mine) return;
    atomic_fetch_min_explicit(&labels[mine - 1u], best, memory_order_relaxed);
    atomic_store_explicit(changed, 1u, memory_order_relaxed);
}

/* Follow every chain to its end and point straight at it, so the next round
 * of joining compares roots rather than whatever it was told last. */
kernel void cc_flatten(device atomic_uint* labels [[buffer(0)]],
                       constant uint& count [[buffer(1)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint mine = atomic_load_explicit(&labels[id], memory_order_relaxed);
    if (mine == 0u) return;
    uint root = mine;
    for (uint guard = 0u; guard < 64u; ++guard) {
        uint up = atomic_load_explicit(&labels[root - 1u], memory_order_relaxed);
        if (up == root || up == 0u) break;
        root = up;
    }
    if (root != mine) atomic_store_explicit(&labels[id], root, memory_order_relaxed);
}

kernel void cc_reset(device uint* slots [[buffer(0)]],
                     constant uint& count [[buffer(1)]],
                     uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    slots[id] = 0xFFFFFFFFu;
}

/* The earliest place each component is met, kept against its root. */
kernel void cc_first_seen(device const uint* labels [[buffer(0)]],
                          device atomic_uint* first [[buffer(1)]],
                          constant uint& count [[buffer(2)]],
                          constant uint& cols [[buffer(3)]],
                          constant uint& connectivity [[buffer(4)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint mine = labels[id];
    if (mine == 0u) return;
    uint y = id / cols;
    uint x = id - y * cols;
    atomic_fetch_min_explicit(&first[mine - 1u], scan_position(y, x, cols, connectivity),
                              memory_order_relaxed);
}

/* One entry per component: which pixel is its root, and where it was first
 * met. Only these travel to the host, and there are as many of them as there
 * are components rather than as there are pixels. */
kernel void cc_collect(device const uint* labels [[buffer(0)]],
                       device const uint* first [[buffer(1)]],
                       device atomic_uint* counter [[buffer(2)]],
                       device float* roots [[buffer(3)]],
                       device float* seen [[buffer(4)]],
                       constant uint& count [[buffer(5)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    if (labels[id] != id + 1u) return;
    uint slot = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    if (slot >= count) return;
    roots[slot] = float(id);
    seen[slot] = float(first[id]);
}

/* The new numbering scattered back against the roots it belongs to. */
kernel void cc_assign(device const float* ordered [[buffer(0)]],
                      device uint* remap [[buffer(1)]],
                      constant uint& count [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    remap[uint(ordered[id])] = id + 1u;
}

kernel void cc_apply(device const uint* labels [[buffer(0)]],
                     device const uint* remap [[buffer(1)]],
                     device float* dst [[buffer(2)]],
                     constant uint& count [[buffer(3)]],
                     constant uint& cols [[buffer(4)]],
                     constant uint& d_off [[buffer(5)]],
                     constant uint& d_step [[buffer(6)]],
                     uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint y = id / cols;
    uint x = id - y * cols;
    uint mine = labels[id];
    dst[d_off + y * d_step + x] = mine == 0u ? 0.0f : float(remap[mine - 1u]);
}

/* A sum too wide for one word, carried into a second.
 *
 * The x coordinates of every pixel of a 4K frame come to thirty-two billion,
 * so a single `uint` wraps. The carry costs one predictable branch that is
 * almost never taken. */
inline void add_wide(device atomic_uint* slot, uint value) {
    uint before = atomic_fetch_add_explicit(&slot[0], value, memory_order_relaxed);
    if (before + value < before) atomic_fetch_add_explicit(&slot[1], 1u, memory_order_relaxed);
}

/* Extremes start where nothing can beat them: the smallest at the top of the
 * range and the largest at the bottom. Seeding both ends the same way leaves
 * whichever comparison points the wrong way untouched for ever. */
kernel void cc_reset_box(device uint* boxes [[buffer(0)]],
                         constant uint& count [[buffer(1)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    boxes[id] = (id % 4u) < 2u ? 0xFFFFFFFFu : 0u;
}

/* Each component's extent and weight, gathered by the pixels themselves.
 *
 * A handful of atomics a pixel against a host loop over every pixel of the
 * frame, which for a 640 by 480 picture was most of what the labelling cost
 * once the labelling itself moved to the device. */
kernel void cc_stats(device const float* labels [[buffer(0)]],
                     device atomic_uint* boxes [[buffer(1)]],
                     device atomic_uint* sums [[buffer(2)]],
                     constant uint& count [[buffer(3)]],
                     constant uint& cols [[buffer(4)]],
                     constant uint& l_off [[buffer(5)]],
                     constant uint& l_step [[buffer(6)]],
                     constant uint& components [[buffer(7)]],
                     uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint y = id / cols;
    uint x = id - y * cols;
    uint label = uint(labels[l_off + y * l_step + x]);
    if (label >= components) return;
    device atomic_uint* box = &boxes[label * 4u];
    atomic_fetch_min_explicit(&box[0], x, memory_order_relaxed);
    atomic_fetch_min_explicit(&box[1], y, memory_order_relaxed);
    atomic_fetch_max_explicit(&box[2], x, memory_order_relaxed);
    atomic_fetch_max_explicit(&box[3], y, memory_order_relaxed);
    device atomic_uint* sum = &sums[label * 6u];
    add_wide(&sum[0], 1u);
    add_wide(&sum[2], x);
    add_wide(&sum[4], y);
}

/* A word in two halves, because a float32 cannot carry thirty-two bits of
 * integer and rounds the top of the range -- which is how an untouched
 * extreme came back as four billion and one rather than four billion. */
kernel void cc_split(device const uint* src [[buffer(0)]],
                     device float* out [[buffer(1)]],
                     constant uint& count [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint value = src[id];
    out[id * 2u + 0u] = float(value & 0xFFFFu);
    out[id * 2u + 1u] = float(value >> 16u);
}


/* The chamfer distance as a fixed point rather than as two sweeps.
 *
 * The sweeps are sequential by construction -- each pixel reads the one
 * before it -- so they cannot be spread over the device at all. Relaxing
 * every pixel against its whole neighbourhood at once converges on the same
 * answer, because the sweeps and the relaxation are both looking for the
 * shortest path through the chamfer mask and there is only one of those.
 *
 * Reading the previous round rather than this one keeps the result the same
 * whichever order the threads happen to run in. */
kernel void dt_seed(device const float* src [[buffer(0)]],
                    device float* dist [[buffer(1)]],
                    constant uint& count [[buffer(2)]],
                    constant uint& cols [[buffer(3)]],
                    constant uint& off [[buffer(4)]],
                    constant uint& step [[buffer(5)]],
                    uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint y = id / cols;
    uint x = id - y * cols;
    dist[id] = src[off + y * step + x] == 0.0f ? 0.0f : 1e10f;
}

inline float dt_reach(device const float* dist, int rows, int cols,
                      int y, int x, float weight, float best) {
    if (y < 0 || x < 0 || y >= rows || x >= cols) return best;
    return min(best, dist[y * cols + x] + weight);
}

kernel void dt_relax(device const float* dist [[buffer(0)]],
                     device float* next [[buffer(1)]],
                     device atomic_uint* changed [[buffer(2)]],
                     constant uint& count [[buffer(3)]],
                     constant uint& rows [[buffer(4)]],
                     constant uint& cols [[buffer(5)]],
                     constant uint& straight_bits [[buffer(6)]],
                     constant uint& diagonal_bits [[buffer(7)]],
                     constant uint& knight_bits [[buffer(8)]],
                     constant uint& wide [[buffer(9)]],
                     uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float here = dist[id];
    if (here == 0.0f) {
        next[id] = 0.0f;
        return;
    }
    int y = int(id / cols);
    int x = int(id - uint(y) * cols);
    int high = int(rows);
    int side = int(cols);
    /* Single steps only. Relaxing against a run of `n` of them at once, with
     * `n` times the weight, crosses an empty frame in far fewer rounds -- but
     * `n * w` and `w` added `n` times are not the same float, so the last bit
     * keeps moving and the rounds never settle. */
    float straight = as_type<float>(straight_bits);
    float diagonal = as_type<float>(diagonal_bits);
    float knight = as_type<float>(knight_bits);

    float best = here;
    best = dt_reach(dist, high, side, y, x - 1, straight, best);
    best = dt_reach(dist, high, side, y, x + 1, straight, best);
    best = dt_reach(dist, high, side, y - 1, x, straight, best);
    best = dt_reach(dist, high, side, y + 1, x, straight, best);
    best = dt_reach(dist, high, side, y - 1, x - 1, diagonal, best);
    best = dt_reach(dist, high, side, y - 1, x + 1, diagonal, best);
    best = dt_reach(dist, high, side, y + 1, x - 1, diagonal, best);
    best = dt_reach(dist, high, side, y + 1, x + 1, diagonal, best);
    if (wide != 0u) {
        best = dt_reach(dist, high, side, y - 2, x - 1, knight, best);
        best = dt_reach(dist, high, side, y - 2, x + 1, knight, best);
        best = dt_reach(dist, high, side, y - 1, x - 2, knight, best);
        best = dt_reach(dist, high, side, y - 1, x + 2, knight, best);
        best = dt_reach(dist, high, side, y + 2, x - 1, knight, best);
        best = dt_reach(dist, high, side, y + 2, x + 1, knight, best);
        best = dt_reach(dist, high, side, y + 1, x - 2, knight, best);
        best = dt_reach(dist, high, side, y + 1, x + 2, knight, best);
    }
    next[id] = best;
    if (best < here) atomic_store_explicit(changed, 1u, memory_order_relaxed);
}

kernel void dt_write(device const float* dist [[buffer(0)]],
                     device float* out [[buffer(1)]],
                     constant uint& count [[buffer(2)]],
                     constant uint& cols [[buffer(3)]],
                     constant uint& d_off [[buffer(4)]],
                     constant uint& d_step [[buffer(5)]],
                     uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint y = id / cols;
    uint x = id - y * cols;
    out[d_off + y * d_step + x] = dist[id];
}

kernel void cc_tally(device const uint* value [[buffer(0)]],
                     device float* out [[buffer(1)]],
                     constant uint& count [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    out[id] = float(value[id]);
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

pub fn flood_fill(
    image: Mat,
    seed: Point,
    new_value: Scalar,
    lo_diff: Scalar = Scalar(0.0, 0.0, 0.0, 0.0),
    up_diff: Scalar = Scalar(0.0, 0.0, 0.0, 0.0),
    flags: int = 4
) -> tuple[int, Rect] {
    let rows = image.rows
    let cols = image.cols
    let cn = image.channels()
    if seed.x < 0 or seed.y < 0 or seed.x >= cols or seed.y >= rows {
        throw ValueError(f"the seed {seed} is outside a {rows}x{cols} image")
    }
    let connectivity = if flags & 255 == 8 { 8 } else { 4 }
    let fixed = flags & FLOODFILL_FIXED_RANGE != 0
    let (offset_y, offset_x) = neighbourhood(connectivity)

    image.flush()
    let values = image.edit()
    var filled: list[bool] = []
    for _index in 0..rows * cols { filled.push(false) }

    var seed_colour: list[float] = []
    for channel in 0..cn { seed_colour.push(values[(seed.y * cols + seed.x) * cn + channel]) }

    var painted: list[float] = []
    for channel in 0..cn { painted.push(new_value[min(channel, 3)]) }

    var stack = [seed.y * cols + seed.x]
    filled[seed.y * cols + seed.x] = true
    var count = 0
    var left = seed.x
    var right = seed.x
    var top = seed.y
    var bottom = seed.y

    while stack.len() > 0 {
        let cell = stack.pop()
        let y = cell // cols
        let x = cell - y * cols
        count += 1
        if x < left { left = x }
        if x > right { right = x }
        if y < top { top = y }
        if y > bottom { bottom = y }

        var reference: list[float] = []
        for channel in 0..cn { reference.push(values[cell * cn + channel]) }
        for channel in 0..cn { values[cell * cn + channel] = painted[channel] }

        for step in 0..offset_y.len() {
            let ny = y + offset_y[step]
            let nx = x + offset_x[step]
            if ny < 0 or nx < 0 or ny >= rows or nx >= cols { continue }
            let next = ny * cols + nx
            if filled[next] { continue }
            var joins = true
            for channel in 0..cn {
                let against = if fixed { seed_colour[channel] } else { reference[channel] }
                let difference = values[next * cn + channel] - against
                if difference < -lo_diff[min(channel, 3)] or difference > up_diff[min(channel, 3)] {
                    joins = false
                    break
                }
            }
            if not joins { continue }
            filled[next] = true
            stack.push(next)
        }
    }
    image.flush()
    return (count, Rect(left, top, right - left + 1, bottom - top + 1))
}

#: Label each connected run of non-zero pixels.
#:
#: Returns how many labels there are, counting the background as one, and an
#: image of them. Label zero is the background; the rest are numbered in the
#: order a raster scan first meets them.
pub fn connected_components(image: Mat, connectivity: int = 8) -> tuple[int, Mat] {
    if image.channels() != 1 { throw ValueError("connected components need a single-channel Mat") }
    if connectivity != 4 and connectivity != 8 {
        throw ValueError(f"connectivity {connectivity} is neither four nor eight")
    }
    image.flush()
    let rows = image.rows
    let cols = image.cols
    let count = rows * cols
    let out = Mat(rows, cols, CV_MAKETYPE(CV_32S, 1))
    if count == 0 { return (1, out) }

    let labels = Buffer(count)
    kernel_for("cc_seed").dispatch_async(
        [image.storage, labels],
        count,
        [count, cols, image.offset, image.step_]
    )

    #: Join, then flatten, until a round finds nothing left to join. Each
    #: round is two dispatches and the flag costs one short read, which is
    #: why the check is not done per pixel.
    let changed = Buffer(1)
    let counted = Buffer(1)
    let propagate = kernel_for("cc_propagate")
    let flatten = kernel_for("cc_flatten")
    var rounds = 0
    while rounds < count {
        rounds += 1
        changed.fill_zero()
        propagate.dispatch_async([labels, changed], count, [count, rows, cols, connectivity])
        flatten.dispatch_async([labels], count, [count])
        kernel_for("cc_tally").dispatch_async([changed, counted], 1, [1])
        if counted.download()[0] == 0.0 { break }
    }

    #: The partition is settled; what remains is the numbering, and OpenCV
    #: numbers by where a component was first met. Only one entry per
    #: component travels for that, not one per pixel.
    let first = Buffer(count)
    kernel_for("cc_reset").dispatch_async([first], count, [count])
    kernel_for("cc_first_seen").dispatch_async([labels, first], count, [count, cols, connectivity])
    let tally = Buffer(1)
    tally.fill_zero()
    let roots = Buffer(count)
    let seen = Buffer(count)
    kernel_for("cc_collect").dispatch_async([labels, first, tally, roots, seen], count, [count])
    kernel_for("cc_tally").dispatch_async([tally, counted], 1, [1])
    let components = min(count, int(counted.download()[0]))

    let remap = Buffer(count)
    var total = 1
    if components > 0 {
        let root_list = roots.download(count: components)
        let seen_list = seen.download(count: components)
        var order: list[int] = []
        for index in 0..components { order.push(index) }
        let ordered = order.sort(key: |index| seen_list[index])
        var flat: list[float] = []
        for index in ordered { flat.push(root_list[index]) }
        let placed = Buffer(components)
        placed.upload(flat)
        kernel_for("cc_assign").dispatch_async([placed, remap], components, [components])
        placed.free()
        total = components + 1
    }
    kernel_for("cc_apply").dispatch_async(
        [labels, remap, out.storage],
        count,
        [count, cols, out.offset, out.step_]
    )
    out.invalidate()
    labels.free()
    changed.free()
    counted.free()
    first.free()
    tally.free()
    roots.free()
    seen.free()
    remap.free()
    return (total, out)
}

#: One device word, put back together from the halves it travelled in.
fn word_at(halves: list[float], index: int) -> float {
    return halves[index * 2] + halves[index * 2 + 1] * 65536.0
}

#: A word and the carry that followed it, which is how a sum wider than
#: thirty-two bits comes back.
fn wide_at(halves: list[float], index: int) -> float {
    return word_at(halves, index) + word_at(halves, index + 1) * 4294967296.0
}

#: Label the components and measure each one.
#:
#: Alongside the labels come the box and pixel count of every component and its
#: centre of mass, both indexed by label, with the background at index zero.
pub fn connected_components_with_stats(
    image: Mat,
    connectivity: int = 8
) -> tuple[int, Mat, list[ComponentStats], list[Point2f]] {
    let (count, labels) = connected_components(image, connectivity)
    let pixels = image.rows * image.cols
    let boxes = Buffer(count * 4)
    let sums = Buffer(count * 6)
    kernel_for("cc_reset_box").dispatch_async([boxes], count * 4, [count * 4])
    sums.fill_zero()
    if pixels > 0 {
        labels.flush()
        kernel_for("cc_stats").dispatch_async(
            [labels.storage, boxes, sums],
            pixels,
            [pixels, image.cols, labels.offset, labels.step_, count]
        )
    }
    let box_out = Buffer(count * 8)
    let sum_out = Buffer(count * 12)
    kernel_for("cc_split").dispatch_async([boxes, box_out], count * 4, [count * 4])
    kernel_for("cc_split").dispatch_async([sums, sum_out], count * 6, [count * 6])
    let extent = box_out.download()
    let weight = sum_out.download()
    boxes.free()
    sums.free()
    box_out.free()
    sum_out.free()

    var stats: list[ComponentStats] = []
    var centroids: list[Point2f] = []
    for index in 0..count {
        let area = int(wide_at(weight, index * 6))
        if area == 0 {
            stats.push(ComponentStats(Rect(0, 0, 0, 0), 0))
            centroids.push(Point2f(0.0, 0.0))
            continue
        }
        let left = int(word_at(extent, index * 4))
        let top = int(word_at(extent, index * 4 + 1))
        let right = int(word_at(extent, index * 4 + 2))
        let bottom = int(word_at(extent, index * 4 + 3))
        stats.push(ComponentStats(Rect(left, top, right - left + 1, bottom - top + 1), area))
        let total = float(area)
        centroids.push(
            Point2f(wide_at(weight, index * 6 + 2) / total, wide_at(weight, index * 6 + 4) / total)
        )
    }
    return (count, labels, stats, centroids)
}

fn chamfer_weights(distance_type: int, mask_size: int) -> list[float] {
    #: OpenCV's weights, which are chosen to keep the two-pass approximation
    #: close to the true Euclidean distance rather than to be exact steps.
    if distance_type == DIST_C { return [1.0, 1.0, 2.0] }
    if distance_type == DIST_L1 { return [1.0, 2.0, 3.0] }
    if mask_size == DIST_MASK_5 { return [1.0, 1.4, 2.1969] }
    return [0.955, 1.3693, 2.0]
}

#: The distance from every non-zero pixel to the nearest zero one.
#:
#: The chamfer distance, with the weights OpenCV uses, which is why the answer
#: is close to but not exactly the Euclidean distance for `DIST_L2`.
#:
#: Found by relaxing every pixel against its neighbours until nothing moves,
#: rather than by the forward and backward sweeps OpenCV uses, because a
#: sweep reads the pixel before it and so cannot be spread over the device at
#: all. Both are looking for the shortest path through the mask and there is
#: only one of those, so `DIST_L1` and `DIST_C`, whose weights are whole
#: numbers, come back identical; `DIST_L2` agrees to about seven digits,
#: which is single precision against the host's double.
#:
#: Distance travels one step a round, so the cost follows how far the
#: furthest pixel is from a zero rather than how large the frame is. On the
#: masks this is for -- edges, thresholded objects, watershed seeds -- that
#: is a handful of rounds and a 640 by 480 frame takes three milliseconds
#: against twenty-seven. A frame with a single zero pixel in it is the other
#: end of that trade and takes about twice what the sweeps did.
pub fn distance_transform(
    image: Mat,
    distance_type: int = DIST_L2,
    mask_size: int = DIST_MASK_3
) -> Mat {
    if image.channels() != 1 { throw ValueError("a distance transform needs a single-channel Mat") }
    image.flush()
    let rows = image.rows
    let cols = image.cols
    let count = rows * cols
    let out = Mat(rows, cols, CV_MAKETYPE(CV_32F, 1))
    if count == 0 { return out }
    let weights = chamfer_weights(distance_type, mask_size)
    let wide = if mask_size == DIST_MASK_5 and distance_type == DIST_L2 { 1 } else { 0 }

    var dist = Buffer(count)
    var next = Buffer(count)
    kernel_for("dt_seed").dispatch_async(
        [image.storage, dist],
        count,
        [count, cols, image.offset, image.step_]
    )
    let changed = Buffer(1)
    let counted = Buffer(1)
    let relax_kernel = kernel_for("dt_relax")
    let settings = [
        count,
        rows,
        cols,
        f32_bits(weights[0]),
        f32_bits(weights[1]),
        f32_bits(weights[2]),
        wide,
    ]
    #: Reading the flag every round would cost more than the rounds do, so a
    #: batch of them goes down together and only then is the queue asked
    #: whether anything moved.
    var rounds = 0
    let limit = rows + cols + 4
    while rounds < limit {
        changed.fill_zero()
        for _batch in 0..DT_BATCH {
            if rounds >= limit { break }
            rounds += 1
            relax_kernel.dispatch_async([dist, next, changed], count, settings)
            let swap = dist
            dist = next
            next = swap
        }
        kernel_for("cc_tally").dispatch_async([changed, counted], 1, [1])
        if counted.download()[0] == 0.0 { break }
    }

    kernel_for("dt_write").dispatch_async(
        [dist, out.storage],
        count,
        [count, cols, out.offset, out.step_]
    )
    out.invalidate()
    dist.free()
    next.free()
    changed.free()
    counted.free()
    return out
}

#: How many relaxation rounds go down before the queue is asked whether any
#: of them changed anything. A round over a 640 by 480 frame is a fortieth of
#: a millisecond and the question costs a seventh of one.
const DT_BATCH = 24

#: The running sum of an image, one row and column larger than it.
#:
#: `integral[y][x]` holds the sum of everything above and to the left of pixel
#: `(y - 1, x - 1)`, so the sum over any rectangle is four lookups.
pub fn integral(image: Mat) -> Mat {
    image.flush()
    let cn = image.channels()
    let rows = image.rows
    let cols = image.cols
    let source = image.values()
    var out: list[float] = []
    for _index in 0..(rows + 1) * (cols + 1) * cn { out.push(0.0) }
    for y in 0..rows {
        for x in 0..cols {
            for channel in 0..cn {
                let above = out[(y * (cols + 1) + x + 1) * cn + channel]
                let leftward = out[((y + 1) * (cols + 1) + x) * cn + channel]
                let corner = out[(y * (cols + 1) + x) * cn + channel]
                let here = source[(y * cols + x) * cn + channel]
                out[((y + 1) * (cols + 1) + x + 1) * cn + channel] = here + above + leftward -
                    corner
            }
        }
    }
    return Mat.from_list(rows + 1, cols + 1, CV_MAKETYPE(CV_32F, cn), out)
}

#: Grow the marked regions until they meet, and mark where they meet.
#:
#: `markers` is a single-channel `CV_32S` image where each region of interest
#: is labelled with a positive number, the known background with another, and
#: everything undecided with zero. On return the undecided pixels have joined a
#: region, and the pixels where two regions met are `-1`.
#:
#: This is Meyer's flooding: the pixels next to a marker go into a queue keyed
#: by how steep the image is there, and the flattest is always taken first, so
#: the regions meet along the ridges. It is the same method OpenCV uses and
#: puts the boundaries in the same places, but it is not checked against
#: OpenCV pixel for pixel the way the rest of this module is — where two
#: regions arrive at a pixel on the same step, which of them takes it comes
#: down to the order the queue happened to be built in.
pub fn watershed(image: Mat, markers: Mat) -> Mat {
    if image.rows != markers.rows or image.cols != markers.cols {
        throw ValueError("the markers have to be the same size as the image")
    }
    if markers.channels() != 1 { throw ValueError("the markers have to be single-channel") }
    image.flush()
    markers.flush()
    let rows = image.rows
    let cols = image.cols
    let cn = image.channels()
    let pixels = image.values()
    let labels = markers.edit()

    #: One bucket per possible gradient, so taking the flattest pixel next is a
    #: scan down the buckets rather than a sort.
    let levels = 256
    var buckets: list[list[int]] = []
    for _index in 0..levels { buckets.push([]) }
    var queued: list[bool] = []
    for _index in 0..rows * cols { queued.push(false) }

    let (offset_y, offset_x) = neighbourhood(4)
    for y in 0..rows {
        for x in 0..cols {
            let cell = y * cols + x
            if labels[cell] == 0.0 { continue }
            for step in 0..4 {
                let ny = y + offset_y[step]
                let nx = x + offset_x[step]
                if ny < 0 or nx < 0 or ny >= rows or nx >= cols { continue }
                let next = ny * cols + nx
                if labels[next] != 0.0 or queued[next] { continue }
                queued[next] = true
                buckets[gradient_at(pixels, rows, cols, cn, ny, nx, levels)].push(next)
            }
        }
    }

    var level = 0
    while level < levels {
        if buckets[level].len() == 0 {
            level += 1
            continue
        }
        let cell = buckets[level].pop()
        let y = cell // cols
        let x = cell - y * cols
        var mine = 0.0
        var conflict = false
        for step in 0..4 {
            let ny = y + offset_y[step]
            let nx = x + offset_x[step]
            if ny < 0 or nx < 0 or ny >= rows or nx >= cols { continue }
            let neighbour = labels[ny * cols + nx]
            if neighbour <= 0.0 { continue }
            if mine == 0.0 {
                mine = neighbour
            } elif neighbour != mine {
                conflict = true
            }
        }
        labels[cell] = if conflict { -1.0 } else { mine }
        if not conflict and mine != 0.0 {
            for step in 0..4 {
                let ny = y + offset_y[step]
                let nx = x + offset_x[step]
                if ny < 0 or nx < 0 or ny >= rows or nx >= cols { continue }
                let next = ny * cols + nx
                if labels[next] != 0.0 or queued[next] { continue }
                queued[next] = true
                let where = gradient_at(pixels, rows, cols, cn, ny, nx, levels)
                buckets[where].push(next)
                #: A flatter pixel than the one just taken has to be served
                #: before the level moves on, or the flooding would skip it.
                if where < level { level = where }
            }
        }
    }
    markers.flush()
    return markers
}

fn gradient_at(
    pixels: list[float],
    rows: int,
    cols: int,
    cn: int,
    y: int,
    x: int,
    levels: int
) -> int {
    var steepest = 0.0
    for channel in 0..cn {
        let here = pixels[(y * cols + x) * cn + channel]
        for step in 0..4 {
            let ny = y + (if step == 0 { -1 } elif step == 3 { 1 } else { 0 })
            let nx = x + (if step == 1 { -1 } elif step == 2 { 1 } else { 0 })
            if ny < 0 or nx < 0 or ny >= rows or nx >= cols { continue }
            #: The magnitude written out rather than called for. `math.fabs`
            #: is a call, and a call in this loop is what stops the tier
            #: compiling it -- which left the flood spending most of a
            #: watershed here.
            let difference = pixels[(ny * cols + nx) * cn + channel] - here
            let gap = if difference < 0.0 { -difference } else { difference }
            if gap > steepest { steepest = gap }
        }
    }
    let level = int(steepest)
    if level < 0 { return 0 }
    if level >= levels { return levels - 1 }
    return level
}

#: Turn a label image into something a person can look at.
pub fn colour_labels(labels: Mat, count: int) -> Mat {
    let values = labels.to_list()
    var out: list[float] = []
    for value in values {
        let label = int(value)
        if label <= 0 {
            out.push(0.0)
            out.push(0.0)
            out.push(0.0)
            continue
        }
        #: Three coprime strides through the byte range keep neighbouring
        #: labels far apart in colour.
        out.push(float(label * 71 % 251))
        out.push(float(label * 149 % 241))
        out.push(float(label * 199 % 233))
    }
    let _unused = count
    return Mat.from_list(labels.rows, labels.cols, CV_MAKETYPE(CV_8U, 3), out)
}
