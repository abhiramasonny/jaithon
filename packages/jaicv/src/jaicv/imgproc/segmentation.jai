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
from ..core.mat import Mat
from ..core.types import CV_32F, CV_32S, CV_8U, CV_MAKETYPE, Point, Point2f, Rect, Scalar

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

fn scan_order(rows: int, cols: int, connectivity: int) -> list[int] {
    var order: list[int] = []
    if connectivity == 4 {
        for cell in 0..rows * cols { order.push(cell) }
        return order
    }
    for block_y in 0..(rows + 1) // 2 {
        for block_x in 0..(cols + 1) // 2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    let y = block_y * 2 + dy
                    let x = block_x * 2 + dx
                    if y < rows and x < cols { order.push(y * cols + x) }
                }
            }
        }
    }
    return order
}

fn find_root(parent: list[int], start: int) -> int {
    var root = start
    while parent[root] != root { root = parent[root] }
    var walk = start
    while parent[walk] != walk {
        let next = parent[walk]
        parent[walk] = root
        walk = next
    }
    return root
}

fn union_labels(parent: list[int], a: int, b: int) -> void {
    let ra = find_root(parent, a)
    let rb = find_root(parent, b)
    if ra == rb { return }
    if ra < rb { parent[rb] = ra } else { parent[ra] = rb }
}

fn label_pass(
    values: list[float],
    rows: int,
    cols: int,
    connectivity: int
) -> tuple[int, list[int]] {
    var provisional: list[int] = []
    for _index in 0..rows * cols { provisional.push(0) }
    var parent = [0]
    var next_label = 1

    for y in 0..rows {
        for x in 0..cols {
            let cell = y * cols + x
            if values[cell] == 0.0 { continue }
            #: Only the neighbours already visited matter, which for a raster
            #: scan is the row above and the pixel to the left.
            var found: list[int] = []
            if x > 0 and provisional[cell - 1] != 0 { found.push(provisional[cell - 1]) }
            if y > 0 {
                if provisional[cell - cols] != 0 { found.push(provisional[cell - cols]) }
                if connectivity == 8 {
                    if x > 0 and provisional[cell - cols - 1] != 0 {
                        found.push(provisional[cell - cols - 1])
                    }
                    if x + 1 < cols and provisional[cell - cols + 1] != 0 {
                        found.push(provisional[cell - cols + 1])
                    }
                }
            }
            if found.len() == 0 {
                provisional[cell] = next_label
                parent.push(next_label)
                next_label += 1
                continue
            }
            var smallest = found[0]
            for label in found {
                if label < smallest { smallest = label }
            }
            provisional[cell] = smallest
            for label in found { union_labels(parent, smallest, label) }
        }
    }

    #: Labels are handed out in the order the components are first met, and
    #: which order that is depends on the connectivity, because OpenCV scans
    #: differently for each: pixel by pixel for four, and in two by two blocks
    #: for eight, since every foreground pixel in such a block is connected to
    #: every other. The partition is the same either way; only the numbering
    #: moves, and the numbering is what a caller sees.
    var renumbered: list[int] = []
    for _index in 0..next_label { renumbered.push(0) }
    var count = 1
    for cell in scan_order(rows, cols, connectivity) {
        if provisional[cell] == 0 { continue }
        let root = find_root(parent, provisional[cell])
        if renumbered[root] != 0 { continue }
        renumbered[root] = count
        count += 1
    }

    var labels: list[int] = []
    for cell in 0..rows * cols {
        if provisional[cell] == 0 {
            labels.push(0)
            continue
        }
        labels.push(renumbered[find_root(parent, provisional[cell])])
    }
    return (count, labels)
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
    let (count, labels) = label_pass(image.to_list(), image.rows, image.cols, connectivity)
    var values: list[float] = []
    for label in labels { values.push(float(label)) }
    return (count, Mat.from_list(image.rows, image.cols, CV_MAKETYPE(CV_32S, 1), values))
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
    let values = labels.to_list()
    var left: list[int] = []
    var top: list[int] = []
    var right: list[int] = []
    var bottom: list[int] = []
    var area: list[int] = []
    var sum_x: list[float] = []
    var sum_y: list[float] = []
    for _index in 0..count {
        left.push(image.cols)
        top.push(image.rows)
        right.push(-1)
        bottom.push(-1)
        area.push(0)
        sum_x.push(0.0)
        sum_y.push(0.0)
    }
    for y in 0..image.rows {
        for x in 0..image.cols {
            let label = int(values[y * image.cols + x])
            if x < left[label] { left[label] = x }
            if x > right[label] { right[label] = x }
            if y < top[label] { top[label] = y }
            if y > bottom[label] { bottom[label] = y }
            area[label] += 1
            sum_x[label] += float(x)
            sum_y[label] += float(y)
        }
    }
    var stats: list[ComponentStats] = []
    var centroids: list[Point2f] = []
    for index in 0..count {
        if right[index] < 0 {
            stats.push(ComponentStats(Rect(0, 0, 0, 0), 0))
            centroids.push(Point2f(0.0, 0.0))
            continue
        }
        stats.push(
            ComponentStats(
                Rect(
                    left[index],
                    top[index],
                    right[index] - left[index] + 1,
                    bottom[index] - top[index] + 1
                ),
                area[index]
            )
        )
        let total = float(area[index])
        centroids.push(Point2f(sum_x[index] / total, sum_y[index] / total))
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
#: The two-pass chamfer, with the weights OpenCV uses, which is why the answer
#: is close to but not exactly the Euclidean distance for `DIST_L2`.
pub fn distance_transform(
    image: Mat,
    distance_type: int = DIST_L2,
    mask_size: int = DIST_MASK_3
) -> Mat {
    if image.channels() != 1 { throw ValueError("a distance transform needs a single-channel Mat") }
    image.flush()
    let rows = image.rows
    let cols = image.cols
    let weights = chamfer_weights(distance_type, mask_size)
    let straight = weights[0]
    let diagonal = weights[1]
    let knight = weights[2]
    let wide = mask_size == DIST_MASK_5 and distance_type == DIST_L2

    let source = image.to_list()
    let far = 1e10
    var out: list[float] = []
    for value in source { out.push(if value == 0.0 { 0.0 } else { far }) }

    for y in 0..rows {
        for x in 0..cols {
            let cell = y * cols + x
            if out[cell] == 0.0 { continue }
            var best = out[cell]
            best = relax(out, best, rows, cols, y, x - 1, 0, straight)
            best = relax(out, best, rows, cols, y - 1, x, 0, straight)
            best = relax(out, best, rows, cols, y - 1, x - 1, 0, diagonal)
            best = relax(out, best, rows, cols, y - 1, x + 1, 0, diagonal)
            if wide {
                best = relax(out, best, rows, cols, y - 2, x - 1, 0, knight)
                best = relax(out, best, rows, cols, y - 2, x + 1, 0, knight)
                best = relax(out, best, rows, cols, y - 1, x - 2, 0, knight)
                best = relax(out, best, rows, cols, y - 1, x + 2, 0, knight)
            }
            out[cell] = best
        }
    }
    for step in 0..rows * cols {
        let y = rows - 1 - step // cols
        let x = cols - 1 - step % cols
        let cell = y * cols + x
        if out[cell] == 0.0 { continue }
        var best = out[cell]
        best = relax(out, best, rows, cols, y, x + 1, 0, straight)
        best = relax(out, best, rows, cols, y + 1, x, 0, straight)
        best = relax(out, best, rows, cols, y + 1, x + 1, 0, diagonal)
        best = relax(out, best, rows, cols, y + 1, x - 1, 0, diagonal)
        if wide {
            best = relax(out, best, rows, cols, y + 2, x - 1, 0, knight)
            best = relax(out, best, rows, cols, y + 2, x + 1, 0, knight)
            best = relax(out, best, rows, cols, y + 1, x - 2, 0, knight)
            best = relax(out, best, rows, cols, y + 1, x + 2, 0, knight)
        }
        out[cell] = best
    }
    return Mat.from_list(rows, cols, CV_MAKETYPE(CV_32F, 1), out)
}

fn relax(
    out: list[float],
    best: float,
    rows: int,
    cols: int,
    y: int,
    x: int,
    _unused: int,
    weight: float
) -> float {
    if y < 0 or x < 0 or y >= rows or x >= cols { return best }
    let candidate = out[y * cols + x] + weight
    if candidate < best { return candidate }
    return best
}

#: The running sum of an image, one row and column larger than it.
#:
#: `integral[y][x]` holds the sum of everything above and to the left of pixel
#: `(y - 1, x - 1)`, so the sum over any rectangle is four lookups.
pub fn integral(image: Mat) -> Mat {
    image.flush()
    let cn = image.channels()
    let rows = image.rows
    let cols = image.cols
    let source = image.to_list()
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
    let pixels = image.to_list()
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
            let gap = math.fabs(pixels[(ny * cols + nx) * cn + channel] - here)
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
