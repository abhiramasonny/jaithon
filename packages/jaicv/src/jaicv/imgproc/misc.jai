#: The corners of `imgproc` that do not belong with the filters, the shapes, or
#: the histograms: false colour, sub-pixel patches, running averages, phase
#: correlation, and the two segmenters that work on colour rather than on
#: edges.
#:
#: ```
#: let heat = apply_color_map(depth, COLORMAP_JET)
#: let patch = get_rect_sub_pix(frame, Size(31, 31), Point2f(120.5, 88.25))
#: let (shift, response) = phase_correlate(previous, current)
#: ```
import std.math
from ..core.mat import Mat
from ..core.types import CV_32F, CV_8U, CV_MAKETYPE, Point, Point2f, Rect, Scalar, Size
from ..core.dft import dft, idft, mul_spectrums, DFT_COMPLEX_OUTPUT, DFT_REAL_OUTPUT, DFT_SCALE
from .geometric import remap

#: Red through yellow to white.
pub const COLORMAP_AUTUMN = 0
#: A cold blue-grey ramp.
pub const COLORMAP_BONE = 1
#: Blue through green to red, the classic false-colour map.
pub const COLORMAP_JET = 2
#: Winter's blue to green.
pub const COLORMAP_WINTER = 3
#: A full sweep of hue.
pub const COLORMAP_RAINBOW = 4
#: Dark blue to white through the sea.
pub const COLORMAP_OCEAN = 5
#: Green to yellow.
pub const COLORMAP_SUMMER = 6
#: Magenta to yellow.
pub const COLORMAP_SPRING = 7
#: Cyan to magenta.
pub const COLORMAP_COOL = 8
#: The hue circle at full saturation.
pub const COLORMAP_HSV = 9
#: A pale sepia.
pub const COLORMAP_PINK = 10
#: Black through red and yellow to white.
pub const COLORMAP_HOT = 11

fn clamp_unit(value: float) -> float {
    if value < 0.0 { return 0.0 }
    if value > 1.0 { return 1.0 }
    return value
}

fn colour_at(map_id: int, t: float) -> tuple[float, float, float] {
    #: Each of these is the formula the map is defined by rather than a table
    #: of samples, so they hold at any resolution.
    if map_id == COLORMAP_AUTUMN { return (1.0, t, 0.0) }
    if map_id == COLORMAP_WINTER { return (0.0, t, 1.0 - t * 0.5) }
    if map_id == COLORMAP_SUMMER { return (t, 0.5 + t * 0.5, 0.4) }
    if map_id == COLORMAP_SPRING { return (1.0, t, 1.0 - t) }
    if map_id == COLORMAP_COOL { return (t, 1.0 - t, 1.0) }
    if map_id == COLORMAP_BONE {
        let r = if t < 0.75 { t * 7.0 / 8.0 } else { t * 11.0 / 8.0 - 3.0 / 8.0 }
        let g = if t < 0.375 {
            t * 7.0 / 8.0
        } elif t < 0.75 {
            t * 29.0 / 24.0 - 1.0 / 8.0
        } else {
            t * 7.0 / 8.0 + 1.0 / 8.0
        }
        let b = if t < 0.375 { t * 29.0 / 24.0 } else { t * 7.0 / 8.0 + 1.0 / 8.0 }
        return (clamp_unit(r), clamp_unit(g), clamp_unit(b))
    }
    if map_id == COLORMAP_HOT {
        let r = clamp_unit(t * 3.0)
        let g = clamp_unit(t * 3.0 - 1.0)
        let b = clamp_unit(t * 3.0 - 2.0)
        return (r, g, b)
    }
    if map_id == COLORMAP_OCEAN {
        let r = clamp_unit(t * 3.0 - 2.0)
        let g = clamp_unit(t * 1.5 - 0.5)
        return (r, g, t)
    }
    if map_id == COLORMAP_PINK {
        let r = math.sqrt(clamp_unit(t * 2.0 / 3.0 + 1.0 / 3.0))
        let g = math.sqrt(clamp_unit(if t < 0.5 {
            t * 2.0 / 3.0
        } else {
            t * 4.0 / 3.0 - 1.0 / 3.0
        }))
        let b = math.sqrt(clamp_unit(if t < 0.75 { t * 2.0 / 3.0 } else { t * 2.0 - 1.0 }))
        return (r, g, b)
    }
    if map_id == COLORMAP_HSV or map_id == COLORMAP_RAINBOW {
        let hue = if map_id == COLORMAP_HSV { t * 6.0 } else { (1.0 - t) * 4.0 }
        let sector = int(hue) % 6
        let fraction = hue - float(int(hue))
        let rising = fraction
        let falling = 1.0 - fraction
        if sector == 0 { return (1.0, rising, 0.0) }
        if sector == 1 { return (falling, 1.0, 0.0) }
        if sector == 2 { return (0.0, 1.0, rising) }
        if sector == 3 { return (0.0, falling, 1.0) }
        if sector == 4 { return (rising, 0.0, 1.0) }
        return (1.0, 0.0, falling)
    }
    #: Jet: the four ramps that make blue rise, then green, then red, with each
    #: falling as the next comes up.
    let r = clamp_unit(min(4.0 * t - 1.5, -4.0 * t + 4.5))
    let g = clamp_unit(min(4.0 * t - 0.5, -4.0 * t + 3.5))
    let b = clamp_unit(min(4.0 * t + 0.5, -4.0 * t + 2.5))
    return (r, g, b)
}

#: Give a single-channel image false colour.
#:
#: `map_id` is one of the `COLORMAP_` constants. Each of these is written as
#: the formula that defines the map rather than as a table of samples, which
#: is worth knowing because OpenCV ships some of them as tables: `AUTUMN`,
#: `COOL`, `SPRING`, `SUMMER` and `WINTER` come out identical to OpenCV's and
#: `JET` to within a level, while `BONE`, `HOT`, `OCEAN`, `PINK`, `HSV` and
#: `RAINBOW` follow the same shape but not the same sixty-four control points.
#: The perceptual maps OpenCV also offers — Viridis and its family — are
#: measured data with no formula behind them and are not here at all.
pub fn apply_color_map(src: Mat, map_id: int = COLORMAP_JET) -> Mat {
    src.flush()
    if src.channels() != 1 { throw ValueError("a colour map needs a single-channel Mat") }
    var table: list[float] = []
    for level in 0..256 {
        let (r, g, b) = colour_at(map_id, float(level) / 255.0)
        table.push(math.floor(b * 255.0 + 0.5))
        table.push(math.floor(g * 255.0 + 0.5))
        table.push(math.floor(r * 255.0 + 0.5))
    }
    let values = src.to_list()
    var out: list[float] = []
    for value in values {
        var level = int(value + 0.5)
        if level < 0 { level = 0 }
        if level > 255 { level = 255 }
        out.push(table[level * 3])
        out.push(table[level * 3 + 1])
        out.push(table[level * 3 + 2])
    }
    return Mat.from_list(src.rows, src.cols, CV_MAKETYPE(CV_8U, 3), out)
}

#: A patch of an image, centred anywhere, not only on a pixel.
pub fn get_rect_sub_pix(src: Mat, patch_size: Size, centre: Point2f) -> Mat {
    src.flush()
    let cn = src.channels()
    let values = src.to_list()
    let left = centre.x - float(patch_size.width - 1) * 0.5
    let top = centre.y - float(patch_size.height - 1) * 0.5
    var out: list[float] = []
    for y in 0..patch_size.height {
        for x in 0..patch_size.width {
            let sx = left + float(x)
            let sy = top + float(y)
            let x0 = int(math.floor(sx))
            let y0 = int(math.floor(sy))
            let fx = sx - float(x0)
            let fy = sy - float(y0)
            for c in 0..cn {
                let v00 = sample(values, src.rows, src.cols, cn, y0, x0, c)
                let v01 = sample(values, src.rows, src.cols, cn, y0, x0 + 1, c)
                let v10 = sample(values, src.rows, src.cols, cn, y0 + 1, x0, c)
                let v11 = sample(values, src.rows, src.cols, cn, y0 + 1, x0 + 1, c)
                let top_row = v00 + (v01 - v00) * fx
                let bottom_row = v10 + (v11 - v10) * fx
                out.push(top_row + (bottom_row - top_row) * fy)
            }
        }
    }
    return Mat.from_list(patch_size.height, patch_size.width, CV_MAKETYPE(CV_32F, cn), out)
}

fn sample(values: list[float], rows: int, cols: int, cn: int, y: int, x: int, c: int) -> float {
    var yy = y
    var xx = x
    if yy < 0 { yy = 0 }
    if xx < 0 { xx = 0 }
    if yy >= rows { yy = rows - 1 }
    if xx >= cols { xx = cols - 1 }
    return values[(yy * cols + xx) * cn + c]
}

fn combine(target: Mat, source: Mat, weight: float, keep: float) -> Mat {
    let a = target.to_list()
    let b = source.to_list()
    if a.len() != b.len() { throw ValueError("both have to be the same size") }
    var out: list[float] = []
    for index in 0..a.len() { out.push(a[index] * keep + b[index] * weight) }
    return Mat.from_list(target.rows, target.cols, target.type_(), out)
}

#: Add an image into a running total.
pub fn accumulate(src: Mat, total: Mat) -> Mat { return combine(total, src, 1.0, 1.0) }

#: Add an image's squares into a running total.
pub fn accumulate_square(src: Mat, total: Mat) -> Mat {
    let values = src.to_list()
    var squared: list[float] = []
    for value in values { squared.push(value * value) }
    return combine(total, Mat.from_list(src.rows, src.cols, total.type_(), squared), 1.0, 1.0)
}

#: Add the product of two images into a running total.
pub fn accumulate_product(first: Mat, second: Mat, total: Mat) -> Mat {
    let a = first.to_list()
    let b = second.to_list()
    var product: list[float] = []
    for index in 0..a.len() { product.push(a[index] * b[index]) }
    return combine(total, Mat.from_list(first.rows, first.cols, total.type_(), product), 1.0, 1.0)
}

#: Fold an image into a running average that forgets slowly.
pub fn accumulate_weighted(src: Mat, total: Mat, alpha: float) -> Mat {
    return combine(total, src, alpha, 1.0 - alpha)
}

#: A window that tapers to zero at its edges.
#:
#: Multiplying an image by this before transforming it stops the transform from
#: seeing the edges as a sudden step, which is what `phase_correlate` needs.
pub fn create_hanning_window(size: Size) -> Mat {
    var values: list[float] = []
    for y in 0..size.height {
        let wy = 0.5 * (1.0 - math.cos(2.0 * math.PI * float(y) / float(max(size.height - 1, 1))))
        for x in 0..size.width {
            let wx = 0.5 *
                (1.0 - math.cos(2.0 * math.PI * float(x) / float(max(size.width - 1, 1))))
            #: The square root is OpenCV's, and it matters: the window is
            #: applied to both images and multiplied together in the spectrum,
            #: so taking the root here makes the product the plain window.
            values.push(math.sqrt(wx * wy))
        }
    }
    return Mat.from_list(size.height, size.width, CV_MAKETYPE(CV_32F, 1), values)
}

#: How far one image has shifted relative to another.
#:
#: Both are transformed, their cross-power spectrum is taken, and the peak of
#: its inverse transform is where the shift is. Returns the shift and how sharp
#: the peak was, which says how much to trust it.
pub fn phase_correlate(first: Mat, second: Mat, window: Mat? = null) -> tuple[Point2f, float] {
    if first.rows != second.rows or first.cols != second.cols {
        throw ValueError("both images have to be the same size")
    }
    var a = first
    var b = second
    let taper = window
    if taper is not null {
        let wa = a.to_list()
        let wb = b.to_list()
        let w = taper.to_list()
        var pa: list[float] = []
        var pb: list[float] = []
        for index in 0..wa.len() {
            pa.push(wa[index] * w[index])
            pb.push(wb[index] * w[index])
        }
        a = Mat.from_list(first.rows, first.cols, CV_MAKETYPE(CV_32F, 1), pa)
        b = Mat.from_list(first.rows, first.cols, CV_MAKETYPE(CV_32F, 1), pb)
    }

    let fa = dft(a, DFT_COMPLEX_OUTPUT)
    let fb = dft(b, DFT_COMPLEX_OUTPUT)
    let cross = mul_spectrums(fa, fb, 0, true)
    #: Dividing by the magnitude leaves only the phase, which is where the
    #: shift lives; the magnitudes carry the content and would swamp it.
    let values = cross.to_list()
    var normalised: list[float] = []
    for index in 0..first.rows * first.cols {
        let re = values[index * 2]
        let im = values[index * 2 + 1]
        let size = math.sqrt(re * re + im * im)
        if size < 1e-12 {
            normalised.push(0.0)
            normalised.push(0.0)
            continue
        }
        normalised.push(re / size)
        normalised.push(im / size)
    }
    let raw = idft(Mat.from_list(first.rows, first.cols, CV_MAKETYPE(CV_32F, 2), normalised), DFT_SCALE | DFT_REAL_OUTPUT).to_list()

    #: The surface is shifted so zero lag sits in the middle, which is what
    #: lets the peak be read off as a displacement from the centre and lets a
    #: negative shift be found without wrapping.
    let rows = first.rows
    let cols = first.cols
    var surface: list[float] = []
    for _index in 0..rows * cols { surface.push(0.0) }
    for y in 0..rows {
        for x in 0..cols {
            let sy = (y + rows // 2) % rows
            let sx = (x + cols // 2) % cols
            surface[sy * cols + sx] = raw[y * cols + x]
        }
    }

    var peak = -1e300
    var peak_at = 0
    for index in 0..surface.len() {
        if surface[index] > peak {
            peak = surface[index]
            peak_at = index
        }
    }

    #: The peak is one pixel wide but the shift is not, so the centre of mass
    #: of the five by five around it is where the answer really is.
    let px = peak_at % cols
    let py = peak_at // cols
    var weight = 0.0
    var wx = 0.0
    var wy = 0.0
    for y in max(py - 2, 0)..min(py + 3, rows) {
        for x in max(px - 2, 0)..min(px + 3, cols) {
            let value = surface[y * cols + x]
            weight += value
            wx += float(x) * value
            wy += float(y) * value
        }
    }
    if math.fabs(weight) < 1e-12 { return (Point2f(0.0, 0.0), 0.0) }
    let cx = float(cols) * 0.5 - wx / weight
    let cy = float(rows) * 0.5 - wy / weight
    let response = weight / float(rows * cols)
    return (Point2f(cx, cy), response)
}

#: Flatten an image into regions of near-constant colour.
#:
#: Every pixel is moved to the average colour and position of the pixels near
#: it in both space and colour, repeatedly, until it stops moving. Regions of
#: similar colour collapse onto one value, which is what makes this a
#: segmenter rather than a filter.
pub fn pyr_mean_shift_filtering(
    src: Mat,
    spatial_radius: float,
    colour_radius: float,
    max_level: int = 1,
    iterations: int = 5
) -> Mat {
    let _unused = max_level
    src.flush()
    if src.channels() < 3 { throw ValueError("mean shift filtering needs a colour Mat") }
    let rows = src.rows
    let cols = src.cols
    let cn = src.channels()
    let values = src.to_list()
    let sr = int(spatial_radius)
    let cr2 = colour_radius * colour_radius

    var out: list[float] = []
    for y in 0..rows {
        for x in 0..cols {
            var cx = float(x)
            var cy = float(y)
            var colour = [0.0, 0.0, 0.0]
            for c in 0..3 { colour[c] = values[(y * cols + x) * cn + c] }

            for _round in 0..iterations {
                var sum_x = 0.0
                var sum_y = 0.0
                var sum_colour = [0.0, 0.0, 0.0]
                var count = 0.0
                let base_x = int(cx)
                let base_y = int(cy)
                for dy in -sr..sr + 1 {
                    for dx in -sr..sr + 1 {
                        let ny = base_y + dy
                        let nx = base_x + dx
                        if ny < 0 or nx < 0 or ny >= rows or nx >= cols { continue }
                        var gap = 0.0
                        for c in 0..3 {
                            let difference = values[(ny * cols + nx) * cn + c] - colour[c]
                            gap += difference * difference
                        }
                        if gap > cr2 { continue }
                        count += 1.0
                        sum_x += float(nx)
                        sum_y += float(ny)
                        for c in 0..3 { sum_colour[c] += values[(ny * cols + nx) * cn + c] }
                    }
                }
                if count == 0.0 { break }
                let moved_x = sum_x / count
                let moved_y = sum_y / count
                var moved = (moved_x - cx) * (moved_x - cx) + (moved_y - cy) * (moved_y - cy)
                cx = moved_x
                cy = moved_y
                for c in 0..3 {
                    let updated = sum_colour[c] / count
                    moved += (updated - colour[c]) * (updated - colour[c])
                    colour[c] = updated
                }
                if moved < 0.01 { break }
            }
            for c in 0..cn {
                if c < 3 {
                    out.push(math.floor(colour[c] + 0.5))
                } else {
                    out.push(values[(y * cols + x) * cn + c])
                }
            }
        }
    }
    return Mat.from_list(rows, cols, src.type_(), out)
}

fn inside_edge(point: Point2f, a: Point2f, b: Point2f) -> bool {
    return (b.x - a.x) * (point.y - a.y) - (b.y - a.y) * (point.x - a.x) >= -1e-9
}

fn edge_crossing(p: Point2f, q: Point2f, a: Point2f, b: Point2f) -> Point2f {
    let d1x = q.x - p.x
    let d1y = q.y - p.y
    let d2x = b.x - a.x
    let d2y = b.y - a.y
    let denominator = d1x * d2y - d1y * d2x
    if math.fabs(denominator) < 1e-12 { return q }
    let t = ((a.x - p.x) * d2y - (a.y - p.y) * d2x) / denominator
    return Point2f(p.x + d1x * t, p.y + d1y * t)
}

#: The overlap of two convex polygons, and how large it is.
#:
#: Sutherland and Hodgman's clip: take the first polygon and cut it against
#: each edge of the second in turn. Both have to be convex and wound the same
#: way, which is what `convex_hull` returns.
pub fn intersect_convex_convex(
    first: list[Point2f],
    second: list[Point2f]
) -> tuple[list[Point2f], float] {
    if first.len() < 3 or second.len() < 3 { return ([], 0.0) }
    var output = [point for point in first]
    for index in 0..second.len() {
        if output.len() == 0 { break }
        let a = second[index]
        let b = second[(index + 1) % second.len()]
        var clipped: list[Point2f] = []
        var previous = output[output.len() - 1]
        for point in output {
            let point_inside = inside_edge(point, a, b)
            let previous_inside = inside_edge(previous, a, b)
            if point_inside {
                if not previous_inside { clipped.push(edge_crossing(previous, point, a, b)) }
                clipped.push(point)
            } elif previous_inside {
                clipped.push(edge_crossing(previous, point, a, b))
            }
            previous = point
        }
        output = clipped
    }

    var area = 0.0
    for index in 0..output.len() {
        let a = output[index]
        let b = output[(index + 1) % output.len()]
        area += a.x * b.y - b.x * a.y
    }
    return (output, math.fabs(area) * 0.5)
}

#: The smallest triangle containing every point, and its area.
#:
#: A triangle enclosing a set of points is the intersection of three
#: half-planes, and once the three directions those planes face are fixed the
#: triangle itself is fixed too — each plane is pushed in until it touches. So
#: the search is over three angles rather than over three lines, which is what
#: makes it a smooth minimisation rather than a combinatorial one.
#:
#: A square is the case that shows why the obvious enumeration does not work:
#: its smallest enclosing triangle has one side touching only at a corner, so
#: no triple of its own edges produces it.
pub fn min_enclosing_triangle(points: list[Point2f]) -> tuple[list[Point2f], float] {
    let hull = convex_of(points)
    if hull.len() < 3 { return (hull, 0.0) }

    var best_angles = [0.0, 2.0 * math.PI / 3.0, 4.0 * math.PI / 3.0]
    var best_area = 1e300
    for index in 0..hull.len() {
        let a = hull[index]
        let b = hull[(index + 1) % hull.len()]
        #: Start with one side lying along a hull edge and the other two spread
        #: evenly, which is close enough to every optimum to fall into it.
        let facing = math.atan2(b.x - a.x, -(b.y - a.y))
        var angles = [facing, facing + 2.0 * math.PI / 3.0, facing + 4.0 * math.PI / 3.0]
        var area = triangle_area(hull, angles)
        var step = 0.6
        while step > 1e-6 {
            var improved = false
            for which in 0..3 {
                for direction in [1.0, -1.0] {
                    var moved = [angles[0], angles[1], angles[2]]
                    moved[which] += direction * step
                    let candidate = triangle_area(hull, moved)
                    if candidate < area - 1e-12 {
                        area = candidate
                        angles = moved
                        improved = true
                    }
                }
            }
            if not improved { step *= 0.5 }
        }
        if area < best_area {
            best_area = area
            best_angles = angles
        }
    }

    let corners = triangle_corners(hull, best_angles)
    if corners.len() != 3 { return (hull, 0.0) }
    return (corners, best_area)
}

fn support(hull: list[Point2f], angle: float) -> float {
    #: How far the hull reaches in this direction, which is where the side
    #: facing that way has to sit.
    let nx = math.cos(angle)
    let ny = math.sin(angle)
    var furthest = -1e300
    for point in hull {
        let reach = nx * point.x + ny * point.y
        if reach > furthest { furthest = reach }
    }
    return furthest
}

fn triangle_corners(hull: list[Point2f], angles: list[float]) -> list[Point2f] {
    var out: list[Point2f] = []
    for index in 0..3 {
        let a = angles[index]
        let b = angles[(index + 1) % 3]
        let determinant = math.cos(a) * math.sin(b) - math.sin(a) * math.cos(b)
        if math.fabs(determinant) < 1e-9 { return [] }
        let da = support(hull, a)
        let db = support(hull, b)
        out.push(
            Point2f(
                (da * math.sin(b) - db * math.sin(a)) / determinant,
                (db * math.cos(a) - da * math.cos(b)) / determinant
            )
        )
    }
    return out
}

fn triangle_area(hull: list[Point2f], angles: list[float]) -> float {
    let corners = triangle_corners(hull, angles)
    if corners.len() != 3 { return 1e300 }
    for point in hull {
        if not inside_triangle(point, corners) { return 1e300 }
    }
    return math.fabs((corners[1].x - corners[0].x) * (corners[2].y - corners[0].y) - (corners[2].x - corners[0].x) * (corners[1].y - corners[0].y)) *
        0.5
}

fn convex_of(points: list[Point2f]) -> list[Point2f] {
    #: Andrew's monotone chain, on floats, because the hull is what the search
    #: above walks and the points here are not on whole pixels.
    var sorted: list[Point2f] = []
    for point in points {
        var at = sorted.len()
        while at > 0 and
            (point.x < sorted[at - 1].x or point.x == sorted[at - 1].x and point.y < sorted[at - 1].y) {
            at -= 1
        }
        sorted.insert(at, point)
    }
    if sorted.len() < 3 { return sorted }

    var lower: list[Point2f] = []
    for point in sorted {
        while lower.len() >= 2 and
            turn(lower[lower.len() - 2], lower[lower.len() - 1], point) <= 0.0 {
            lower.pop()
        }
        lower.push(point)
    }
    var upper: list[Point2f] = []
    for index in 0..sorted.len() {
        let point = sorted[sorted.len() - 1 - index]
        while upper.len() >= 2 and
            turn(upper[upper.len() - 2], upper[upper.len() - 1], point) <= 0.0 {
            upper.pop()
        }
        upper.push(point)
    }
    var ring: list[Point2f] = []
    for index in 0..upper.len() - 1 { ring.push(upper[index]) }
    for index in 0..lower.len() - 1 { ring.push(lower[index]) }
    return ring
}

fn turn(o: Point2f, a: Point2f, b: Point2f) -> float {
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
}

fn inside_triangle(point: Point2f, corners: list[Point2f]) -> bool {
    var sign = 0
    for index in 0..3 {
        let a = corners[index]
        let b = corners[(index + 1) % 3]
        let cross = (b.x - a.x) * (point.y - a.y) - (b.y - a.y) * (point.x - a.x)
        if math.fabs(cross) < 1e-9 { continue }
        let here = if cross > 0.0 { 1 } else { -1 }
        if sign == 0 {
            sign = here
        } elif here != sign {
            return false
        }
    }
    return true
}
