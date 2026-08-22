#: Finding things in an image: template matching, corner responses, and the
#: Hough transforms.
#:
#: The three template-matching sums and the two corner responses run on the
#: device, because both are a fixed window swept over every pixel and that is
#: what a kernel is for. The Hough transforms build an accumulator and then
#: read peaks out of it, which is a scatter followed by a sort, and both halves
#: are on the host.
#:
#: ```
#: let scores = match_template(scene, patch, TM_CCOEFF_NORMED)
#: let (_low, best, _at_low, at_best) = min_max_loc(scores)
#:
#: let corners = good_features_to_track(gray, 50, 0.01, 8.0)
#: let lines = hough_lines(edges, 1.0, PI / 180.0, 90)
#: ```
import std.math
from std.core import Printable
from std.gpu import Kernel
from ..core.mat import Mat, f32_bits
from ..core.types import BORDER_DEFAULT, CV_32F, CV_MAKETYPE, Point, Point2f, Size
from .common import PRELUDE
from .filter import blur, box_filter, sobel

#: Sum of squared differences, where less is a better match.
pub const TM_SQDIFF = 0
#: Sum of squared differences over the energy of both, in `[0, 1]`.
pub const TM_SQDIFF_NORMED = 1
#: Cross-correlation, where more is a better match.
pub const TM_CCORR = 2
#: Cross-correlation over the energy of both, in `[0, 1]`.
pub const TM_CCORR_NORMED = 3
#: Cross-correlation after both have had their mean taken out.
pub const TM_CCOEFF = 4
#: The same, scaled into `[-1, 1]`.
pub const TM_CCOEFF_NORMED = 5

#: The classic Hough transform, over a line's distance and angle.
pub const HOUGH_STANDARD = 0
#: The probabilistic Hough transform, which returns segments.
pub const HOUGH_PROBABILISTIC = 1
#: The gradient method, which is what circles are found with.
pub const HOUGH_GRADIENT = 3

let SOURCE = PRELUDE + """
/* Every template-matching method is one of six reductions over the same three
 * sums, so one kernel computes all three and the choice is made at the end. */
kernel void feature_match(device const float* image [[buffer(0)]],
                          device const float* templ [[buffer(1)]],
                          device float* dst [[buffer(2)]],
                          device const float* t_stats [[buffer(3)]],
                          constant uint& count [[buffer(4)]],
                          constant uint& out_cols [[buffer(5)]],
                          constant uint& cn [[buffer(6)]],
                          constant uint& img_off [[buffer(7)]],
                          constant uint& img_step [[buffer(8)]],
                          constant uint& t_rows [[buffer(9)]],
                          constant uint& t_cols [[buffer(10)]],
                          constant uint& t_off [[buffer(11)]],
                          constant uint& t_step [[buffer(12)]],
                          constant uint& dst_off [[buffer(13)]],
                          constant uint& dst_step [[buffer(14)]],
                          constant uint& method [[buffer(15)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint oy = id / out_cols;
    uint ox = id - oy * out_cols;

    /* `t_stats` holds one sum per channel, then the template's total sum of
     * squares, then the norm the two centred methods divide by. Keeping the
     * per-channel sums apart is what makes a colour template's mean come out
     * the way OpenCV computes it: one mean per channel, not one overall. */
    float cross = 0.0f;
    float squared = 0.0f;
    float window_energy = 0.0f;
    float window[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint ty = 0; ty < t_rows; ++ty) {
        for (uint tx = 0; tx < t_cols; ++tx) {
            for (uint c = 0; c < cn; ++c) {
                float a = image[img_off + (oy + ty) * img_step + (ox + tx) * cn + c];
                float b = templ[t_off + ty * t_step + tx * cn + c];
                cross += a * b;
                window_energy += a * a;
                window[c] += a;
                float gap = a - b;
                squared += gap * gap;
            }
        }
    }

    float area = float(t_rows * t_cols);
    float t_energy = t_stats[4];
    float t_norm = t_stats[5];

    float result = 0.0f;
    if (method == 0u) {
        result = squared;
    } else if (method == 1u) {
        float below = sqrt(t_energy * window_energy);
        result = below > 0.0f ? squared / below : 0.0f;
    } else if (method == 2u) {
        result = cross;
    } else if (method == 3u) {
        float below = sqrt(t_energy * window_energy);
        result = below > 0.0f ? cross / below : 0.0f;
    } else {
        float centred = cross;
        float window_squares = 0.0f;
        for (uint c = 0; c < cn; ++c) {
            centred -= t_stats[c] / area * window[c];
            window_squares += window[c] * window[c];
        }
        if (method == 4u) {
            result = centred;
        } else {
            float spread = window_energy - window_squares / area;
            if (spread < 0.0f) spread = 0.0f;
            float below = sqrt(spread) * t_norm;
            result = below > 0.0f ? centred / below : 0.0f;
        }
    }
    dst[dst_off + oy * dst_step + ox] = result;
}

/* The three entries of the structure tensor, before they are summed over a
 * window: dx*dx, dx*dy, dy*dy. */
kernel void feature_covariance(device const float* dx [[buffer(0)]],
                               device const float* dy [[buffer(1)]],
                               device float* dst [[buffer(2)]],
                               constant uint& count [[buffer(3)]],
                               constant uint& cols [[buffer(4)]],
                               constant uint& dx_off [[buffer(5)]],
                               constant uint& dx_step [[buffer(6)]],
                               constant uint& dy_off [[buffer(7)]],
                               constant uint& dy_step [[buffer(8)]],
                               constant uint& dst_off [[buffer(9)]],
                               constant uint& dst_step [[buffer(10)]],
                               uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint y = id / cols;
    uint x = id - y * cols;
    float gx = dx[dx_off + y * dx_step + x];
    float gy = dy[dy_off + y * dy_step + x];
    uint base = dst_off + y * dst_step + x * 3u;
    dst[base] = gx * gx;
    dst[base + 1] = gx * gy;
    dst[base + 2] = gy * gy;
}

kernel void feature_harris(device const float* cov [[buffer(0)]],
                           device float* dst [[buffer(1)]],
                           constant uint& count [[buffer(2)]],
                           constant uint& cols [[buffer(3)]],
                           constant uint& cov_off [[buffer(4)]],
                           constant uint& cov_step [[buffer(5)]],
                           constant uint& dst_off [[buffer(6)]],
                           constant uint& dst_step [[buffer(7)]],
                           constant uint& k_bits [[buffer(8)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint y = id / cols;
    uint x = id - y * cols;
    uint base = cov_off + y * cov_step + x * 3u;
    float a = cov[base];
    float b = cov[base + 1];
    float c = cov[base + 2];
    float k = as_type<float>(k_bits);
    dst[dst_off + y * dst_step + x] = a * c - b * b - k * (a + c) * (a + c);
}

kernel void feature_min_eigen(device const float* cov [[buffer(0)]],
                              device float* dst [[buffer(1)]],
                              constant uint& count [[buffer(2)]],
                              constant uint& cols [[buffer(3)]],
                              constant uint& cov_off [[buffer(4)]],
                              constant uint& cov_step [[buffer(5)]],
                              constant uint& dst_off [[buffer(6)]],
                              constant uint& dst_step [[buffer(7)]],
                              uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint y = id / cols;
    uint x = id - y * cols;
    uint base = cov_off + y * cov_step + x * 3u;
    float a = cov[base] * 0.5f;
    float b = cov[base + 1];
    float c = cov[base + 2] * 0.5f;
    dst[dst_off + y * dst_step + x] = (a + c) - sqrt((a - c) * (a - c) + b * b);
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

#: Slide a template over an image and score every position.
#:
#: The result is `(image.rows - templ.rows + 1)` by `(image.cols - templ.cols
#: + 1)` and `CV_32F`. Which end of it is the best match depends on the method:
#: the two squared-difference forms are best at their smallest, the other four
#: at their largest.
pub fn match_template(image: Mat, templ: Mat, method: int = TM_CCOEFF_NORMED) -> Mat {
    if image.channels() != templ.channels() {
        throw ValueError("the template has to have as many channels as the image")
    }
    if templ.rows > image.rows or templ.cols > image.cols {
        throw ValueError("the template does not fit inside the image")
    }
    if method < 0 or method > 5 {
        throw ValueError(f"template matching method {method} is not one OpenCV defines")
    }
    image.flush()
    templ.flush()

    let rows = image.rows - templ.rows + 1
    let cols = image.cols - templ.cols + 1
    let out = Mat.zeros(rows, cols, CV_MAKETYPE(CV_32F, 1))
    let count = rows * cols
    if count == 0 { return out }

    let cn = templ.channels()
    let area = float(templ.rows * templ.cols)
    var sums = [0.0, 0.0, 0.0, 0.0]
    var energy = 0.0
    let flat = templ.values()
    for index in 0..flat.len() {
        sums[index % cn] += flat[index]
        energy += flat[index] * flat[index]
    }
    #: The norm the centred methods divide by is the template's spread over
    #: every channel at once, taken about each channel's own mean.
    var spread = energy
    for channel in 0..cn { spread -= sums[channel] * sums[channel] / area }
    if spread < 0.0 { spread = 0.0 }
    let stats = Mat.from_list(
        1,
        6,
        CV_MAKETYPE(CV_32F, 1),
        [sums[0], sums[1], sums[2], sums[3], energy, math.sqrt(spread)]
    )

    kernel_for("feature_match").dispatch_async(
        [image.storage, templ.storage, out.storage, stats.storage],
        count,
        [
            count,
            cols,
            image.channels(),
            image.offset,
            image.step_,
            templ.rows,
            templ.cols,
            templ.offset,
            templ.step_,
            out.offset,
            out.step_,
            method,
        ]
    )
    out.invalidate()
    return out
}

fn structure_tensor(src: Mat, block_size: int, aperture_size: int) -> Mat {
    #: OpenCV scales the derivatives so the response does not depend on the
    #: aperture, the block, or the depth, and the constant below is that
    #: scaling written out.
    var scale = float(1 << aperture_size - 1) * float(block_size)
    if src.depth() < CV_32F { scale *= 255.0 }
    scale = 1.0 / scale

    let dx = sobel(
        src,
        CV_32F,
        1,
        0,
        ksize: aperture_size,
        scale: scale,
        border_mode: BORDER_DEFAULT
    )
    let dy = sobel(
        src,
        CV_32F,
        0,
        1,
        ksize: aperture_size,
        scale: scale,
        border_mode: BORDER_DEFAULT
    )
    let cov = Mat.zeros(src.rows, src.cols, CV_MAKETYPE(CV_32F, 3))
    let count = src.rows * src.cols
    if count == 0 { return cov }
    kernel_for("feature_covariance").dispatch_async(
        [dx.storage, dy.storage, cov.storage],
        count,
        [count, src.cols, dx.offset, dx.step_, dy.offset, dy.step_, cov.offset, cov.step_]
    )
    cov.invalidate()
    #: The window sums rather than averages, which is why this asks the box
    #: filter not to normalise.
    return box_filter(
        cov,
        CV_32F,
        Size(block_size, block_size),
        normalize: false,
        border_mode: BORDER_DEFAULT
    )
}

#: Harris's corner response.
#:
#: High where the image bends in two directions at once, negative along an
#: edge, near zero on flat ground. `k` trades corners against edges; OpenCV's
#: examples use values between 0.04 and 0.06.
pub fn corner_harris(
    src: Mat,
    block_size: int = 3,
    aperture_size: int = 3,
    k: float = 0.04
) -> Mat {
    if src.channels() != 1 { throw ValueError("corner_harris needs a single-channel Mat") }
    src.flush()
    let cov = structure_tensor(src, block_size, aperture_size)
    let out = Mat.zeros(src.rows, src.cols, CV_MAKETYPE(CV_32F, 1))
    let count = src.rows * src.cols
    if count == 0 { return out }
    kernel_for("feature_harris").dispatch_async(
        [cov.storage, out.storage],
        count,
        [count, src.cols, cov.offset, cov.step_, out.offset, out.step_, f32_bits(k)]
    )
    out.invalidate()
    return out
}

#: The smaller eigenvalue of the structure tensor, which is Shi and Tomasi's
#: corner measure.
pub fn corner_min_eigen_val(src: Mat, block_size: int = 3, aperture_size: int = 3) -> Mat {
    if src.channels() != 1 { throw ValueError("corner_min_eigen_val needs a single-channel Mat") }
    src.flush()
    let cov = structure_tensor(src, block_size, aperture_size)
    let out = Mat.zeros(src.rows, src.cols, CV_MAKETYPE(CV_32F, 1))
    let count = src.rows * src.cols
    if count == 0 { return out }
    kernel_for("feature_min_eigen").dispatch_async(
        [cov.storage, out.storage],
        count,
        [count, src.cols, cov.offset, cov.step_, out.offset, out.step_]
    )
    out.invalidate()
    return out
}

#: The strongest corners, spread out.
#:
#: Corners weaker than `quality_level` times the strongest are dropped, then
#: the rest are taken in order of strength, skipping any that fall within
#: `min_distance` of one already taken. `use_harris` picks Harris's response
#: instead of Shi and Tomasi's.
pub fn good_features_to_track(
    image: Mat,
    max_corners: int,
    quality_level: float,
    min_distance: float,
    mask: Mat? = null,
    block_size: int = 3,
    use_harris: bool = false,
    k: float = 0.04
) -> list[Point2f] {
    if quality_level <= 0.0 { throw ValueError("the quality level has to be positive") }
    if min_distance < 0.0 { throw ValueError("the minimum distance cannot be negative") }
    let response = if use_harris {
        corner_harris(image, block_size, 3, k)
    } else {
        corner_min_eigen_val(image, block_size, 3)
    }
    let values = response.to_list()
    let rows = response.rows
    let cols = response.cols

    var strongest = 0.0
    for value in values {
        if value > strongest { strongest = value }
    }
    if strongest <= 0.0 { return [] }
    let floor_value = strongest * quality_level

    var mask_values: list[float] = []
    let mask_image = mask
    if mask_image is not null {
        mask_image.flush()
        mask_values = mask_image.to_list()
    }

    #: A corner has to be at least as strong as all eight of its neighbours, or
    #: one bump in the response would come back as a cluster of corners.
    var candidates: list[int] = []
    for y in 1..rows - 1 {
        for x in 1..cols - 1 {
            let cell = y * cols + x
            let here = values[cell]
            if here < floor_value { continue }
            if mask_image is not null and mask_values[cell] == 0.0 { continue }
            var peak = true
            for dy in -1..2 {
                for dx in -1..2 {
                    if dy == 0 and dx == 0 { continue }
                    if values[(y + dy) * cols + x + dx] > here {
                        peak = false
                        break
                    }
                }
                if not peak { break }
            }
            if peak { candidates.push(cell) }
        }
    }

    #: Strongest first. Inserting each candidate into a sorted list instead
    #: makes the work grow with the square of how many there are, and on a
    #: textured frame there are tens of thousands: 640 by 480 took 28 seconds
    #: that way, nearly all of it shuffling a list along.
    let ordered = candidates.sort(key: |cell| values[cell], reverse: true)

    var kept: list[Point2f] = []
    let squared = min_distance * min_distance
    for cell in ordered {
        if max_corners > 0 and kept.len() >= max_corners { break }
        let y = cell // cols
        let x = cell - y * cols
        var clear = true
        for taken in kept {
            let dx = taken.x - float(x)
            let dy = taken.y - float(y)
            if dx * dx + dy * dy < squared {
                clear = false
                break
            }
        }
        if clear { kept.push(Point2f(float(x), float(y))) }
    }
    return kept
}

#: Move corner estimates onto their true sub-pixel positions.
#:
#: Around a real corner every image gradient is perpendicular to the vector
#: from the corner, so the point where that holds best is found by solving a
#: small system, and iterating on it converges within a pixel or so.
pub fn corner_sub_pix(
    image: Mat,
    corners: list[Point2f],
    window: Size = Size(5, 5),
    iterations: int = 20,
    epsilon: float = 0.03
) -> list[Point2f] {
    if image.channels() != 1 { throw ValueError("corner_sub_pix needs a single-channel Mat") }
    image.flush()
    let values = image.to_list()
    let rows = image.rows
    let cols = image.cols
    var out: list[Point2f] = []

    for corner in corners {
        var cx = corner.x
        var cy = corner.y
        for _pass in 0..iterations {
            var a = 0.0
            var b = 0.0
            var c = 0.0
            var bx = 0.0
            var by = 0.0
            for dy in -window.height..window.height + 1 {
                for dx in -window.width..window.width + 1 {
                    let x = int(cx) + dx
                    let y = int(cy) + dy
                    if x < 1 or y < 1 or x >= cols - 1 or y >= rows - 1 { continue }
                    let gx = (values[y * cols + x + 1] - values[y * cols + x - 1]) * 0.5
                    let gy = (values[(y + 1) * cols + x] - values[(y - 1) * cols + x]) * 0.5
                    let weight = gx * gx + gy * gy
                    if weight <= 0.0 { continue }
                    a += gx * gx
                    b += gx * gy
                    c += gy * gy
                    bx += gx * gx * float(x) + gx * gy * float(y)
                    by += gx * gy * float(x) + gy * gy * float(y)
                }
            }
            let determinant = a * c - b * b
            if math.fabs(determinant) < 1e-12 { break }
            let nx = (c * bx - b * by) / determinant
            let ny = (a * by - b * bx) / determinant
            let moved = (nx - cx) * (nx - cx) + (ny - cy) * (ny - cy)
            cx = nx
            cy = ny
            if moved < epsilon * epsilon { break }
        }
        out.push(Point2f(cx, cy))
    }
    return out
}

#: A line as the Hough transform reports it: how far it passes from the origin
#: and at what angle.
pub class HoughLine: Printable {
    pub let rho: float
    pub let theta: float
    pub let votes: int

    pub fn init(self, rho: float, theta: float, votes: int) {
        self.rho = rho
        self.theta = theta
        self.votes = votes
    }

    pub fn to_str(self) -> str {
        return f"HoughLine(rho={self.rho}, theta={self.theta}, votes={self.votes})"
    }

    fn __str__(self) -> str { return self.to_str() }
}

#: Find lines by the classic Hough transform.
#:
#: `rho` and `theta` are the resolution of the accumulator; a pixel votes for
#: every line through it, and a cell with more than `threshold` votes that also
#: beats its four neighbours is reported. Lines come back strongest first.
pub fn hough_lines(
    image: Mat,
    rho: float,
    theta: float,
    threshold: int,
    lines_max: int = 0
) -> list[HoughLine] {
    if image.channels() != 1 { throw ValueError("hough_lines needs a single-channel Mat") }
    if rho <= 0.0 or theta <= 0.0 {
        throw ValueError("the accumulator resolution has to be positive")
    }
    image.flush()
    let rows = image.rows
    let cols = image.cols
    let values = image.to_list()

    let angles = int(math.floor(math.PI / theta + 0.5))
    let distances = int(math.floor(float((cols + rows) * 2 + 1) / rho + 0.5))
    if angles <= 0 or distances <= 0 { return [] }
    let inverse = 1.0 / rho

    var table_sin: list[float] = []
    var table_cos: list[float] = []
    for index in 0..angles {
        let angle = float(index) * theta
        table_sin.push(math.sin(angle) * inverse)
        table_cos.push(math.cos(angle) * inverse)
    }

    #: A one-cell border round the accumulator means the peak test never has to
    #: check whether it is at an edge.
    let width = distances + 2
    var accumulator: list[int] = []
    for _index in 0..(angles + 2) * width { accumulator.push(0) }
    let centre = (distances - 1) // 2

    for y in 0..rows {
        for x in 0..cols {
            if values[y * cols + x] == 0.0 { continue }
            for index in 0..angles {
                var distance = int(
                    math.floor(
                        float(x) * table_cos[index] +
                            float(y) * table_sin[index] +
                            0.5
                    )
                )
                distance += centre
                if distance < 0 or distance >= distances { continue }
                accumulator[(index + 1) * width + distance + 1] += 1
            }
        }
    }

    var found: list[HoughLine] = []
    for index in 0..angles {
        for distance in 0..distances {
            let base = (index + 1) * width + distance + 1
            let votes = accumulator[base]
            if votes <= threshold { continue }
            #: Strictly greater on the two neighbours behind and merely at
            #: least as great on the two ahead, which is what stops a plateau
            #: of equal votes being reported twice.
            if votes <= accumulator[base - 1] or votes < accumulator[base + 1] { continue }
            if votes <= accumulator[base - width] or votes < accumulator[base + width] { continue }
            found.push(
                HoughLine(
                    (float(distance) - float(distances - 1) * 0.5) * rho,
                    float(index) * theta,
                    votes
                )
            )
        }
    }

    #: Most votes first, sorted rather than inserted one at a time: a busy
    #: edge map produces thousands of lines, and placing each into an ordered
    #: list costs the square of that.
    let ordered = found.sort(key: |line| line.votes, reverse: true)
    if lines_max > 0 and ordered.len() > lines_max {
        var trimmed: list[HoughLine] = []
        for index in 0..lines_max { trimmed.push(ordered[index]) }
        return trimmed
    }
    return ordered
}

#: A line segment, as the probabilistic Hough transform reports it.
pub class HoughSegment: Printable {
    pub let start: Point
    pub let end: Point

    pub fn init(self, start: Point, end: Point) {
        self.start = start
        self.end = end
    }

    pub fn to_str(self) -> str { return f"HoughSegment({self.start}, {self.end})" }

    fn __str__(self) -> str { return self.to_str() }
}

#: Find line segments rather than infinite lines.
#:
#: The accumulator is filled the same way, but a peak is then walked back along
#: the image to find where the line actually starts and stops, and its pixels
#: are removed so the next peak is a different line.
#:
#: OpenCV shuffles the edge pixels with its own generator before doing this, so
#: which of several equally good segments it finds first is not reproducible
#: here; this walks the pixels in raster order instead, which is deterministic.
pub fn hough_lines_p(
    image: Mat,
    rho: float,
    theta: float,
    threshold: int,
    min_line_length: float = 0.0,
    max_line_gap: float = 0.0
) -> list[HoughSegment] {
    if image.channels() != 1 { throw ValueError("hough_lines_p needs a single-channel Mat") }
    image.flush()
    let rows = image.rows
    let cols = image.cols
    var alive = image.to_list()
    var out: list[HoughSegment] = []

    loop {
        let remaining = Mat.from_list(rows, cols, image.type_(), alive)
        let lines = hough_lines(remaining, rho, theta, threshold, 1)
        if lines.len() == 0 { break }
        let line = lines[0]
        let (start, end, cleared) = trace_segment(alive, rows, cols, line, max_line_gap)
        if not cleared { break }
        let dx = float(end.x - start.x)
        let dy = float(end.y - start.y)
        if math.sqrt(dx * dx + dy * dy) >= min_line_length { out.push(HoughSegment(start, end)) }
    }
    return out
}

fn trace_segment(
    alive: list[float],
    rows: int,
    cols: int,
    line: HoughLine,
    max_gap: float
) -> tuple[Point, Point, bool] {
    #: Every lit pixel within half a cell of the line belongs to it; the run
    #: between the first and last of them, allowing gaps up to `max_gap`, is
    #: the segment.
    let ct = math.cos(line.theta)
    let st = math.sin(line.theta)
    var on_line: list[int] = []
    for y in 0..rows {
        for x in 0..cols {
            if alive[y * cols + x] == 0.0 { continue }
            if math.fabs(float(x) * ct + float(y) * st - line.rho) > 0.71 { continue }
            on_line.push(y * cols + x)
        }
    }
    if on_line.len() == 0 { return (Point(0, 0), Point(0, 0), false) }

    var best_first = 0
    var best_last = 0
    var first = 0
    var previous = on_line[0]
    var best_length = -1.0
    for index in 0..on_line.len() {
        let cell = on_line[index]
        if index > 0 {
            let py = previous // cols
            let px = previous - py * cols
            let cy = cell // cols
            let cx = cell - cy * cols
            let gap = math.sqrt(float((cx - px) * (cx - px) + (cy - py) * (cy - py)))
            if gap > max_gap + 1.5 {
                let length = run_length(on_line[first], on_line[index - 1], cols)
                if length > best_length {
                    best_length = length
                    best_first = first
                    best_last = index - 1
                }
                first = index
            }
        }
        previous = cell
    }
    let length = run_length(on_line[first], on_line[on_line.len() - 1], cols)
    if length > best_length {
        best_first = first
        best_last = on_line.len() - 1
    }

    for index in best_first..best_last + 1 { alive[on_line[index]] = 0.0 }
    let a = on_line[best_first]
    let b = on_line[best_last]
    return (Point(a - a // cols * cols, a // cols), Point(b - b // cols * cols, b // cols), true)
}

fn run_length(from_cell: int, to_cell: int, cols: int) -> float {
    let ay = from_cell // cols
    let ax = from_cell - ay * cols
    let by = to_cell // cols
    let bx = to_cell - by * cols
    return math.sqrt(float((bx - ax) * (bx - ax) + (by - ay) * (by - ay)))
}

#: A circle the Hough transform found.
pub class HoughCircle: Printable {
    pub let centre: Point2f
    pub let radius: float
    pub let votes: int

    pub fn init(self, centre: Point2f, radius: float, votes: int) {
        self.centre = centre
        self.radius = radius
        self.votes = votes
    }

    pub fn to_str(self) -> str {
        return f"HoughCircle({self.centre}, r={self.radius}, votes={self.votes})"
    }

    fn __str__(self) -> str { return self.to_str() }
}

#: Find circles by the gradient Hough method.
#:
#: Each edge pixel votes along its own gradient, so centres accumulate where
#: the gradients point at each other; the radius is then chosen by a second
#: vote over the distances from the centre to the edge pixels around it.
#:
#: `dp` is how much coarser the centre accumulator is than the image,
#: `min_dist` how far apart two circles have to be, `param1` the upper Canny
#: threshold used to find the edges, and `param2` how many votes a centre
#: needs. This follows OpenCV's method but not its exact peak bookkeeping, so
#: the circles agree in position and radius rather than to the last decimal.
pub fn hough_circles(
    image: Mat,
    dp: float = 1.0,
    min_dist: float = 10.0,
    param1: float = 100.0,
    param2: float = 30.0,
    min_radius: int = 0,
    max_radius: int = 0
) -> list[HoughCircle] {
    if image.channels() != 1 { throw ValueError("hough_circles needs a single-channel Mat") }
    if dp <= 0.0 { throw ValueError("the accumulator scale has to be positive") }
    let smoothed = blur(image, Size(3, 3))
    let dx = sobel(smoothed, CV_32F, 1, 0, ksize: 3)
    let dy = sobel(smoothed, CV_32F, 0, 1, ksize: 3)
    let gradient_x = dx.values()
    let gradient_y = dy.values()

    let rows = image.rows
    let cols = image.cols
    let acc_rows = int(float(rows) / dp) + 1
    let acc_cols = int(float(cols) / dp) + 1
    var accumulator: list[int] = []
    for _index in 0..acc_rows * acc_cols { accumulator.push(0) }

    let upper = if max_radius > 0 { max_radius } else { max(rows, cols) }
    let lower = max(min_radius, 1)
    let edge_floor = param1 * 0.5

    var edges: list[int] = []
    for y in 0..rows {
        for x in 0..cols {
            let gx = gradient_x[y * cols + x]
            let gy = gradient_y[y * cols + x]
            let strength = math.sqrt(gx * gx + gy * gy)
            if strength < edge_floor { continue }
            edges.push(y * cols + x)
            let ux = gx / strength
            let uy = gy / strength
            for direction in 0..2 {
                let sign = if direction == 0 { 1.0 } else { -1.0 }
                for radius in lower..upper {
                    let cx = float(x) + sign * ux * float(radius)
                    let cy = float(y) + sign * uy * float(radius)
                    let ax = int(cx / dp)
                    let ay = int(cy / dp)
                    if ax < 0 or ay < 0 or ax >= acc_cols or ay >= acc_rows { break }
                    accumulator[ay * acc_cols + ax] += 1
                }
            }
        }
    }

    var centres: list[int] = []
    for y in 1..acc_rows - 1 {
        for x in 1..acc_cols - 1 {
            let here = accumulator[y * acc_cols + x]
            if float(here) < param2 { continue }
            var peak = true
            for dy2 in -1..2 {
                for dx2 in -1..2 {
                    if dy2 == 0 and dx2 == 0 { continue }
                    if accumulator[(y + dy2) * acc_cols + x + dx2] > here {
                        peak = false
                        break
                    }
                }
                if not peak { break }
            }
            if peak { centres.push(y * acc_cols + x) }
        }
    }

    #: Strongest first, sorted rather than inserted one at a time -- the same
    #: point as in `good_features_to_track`.
    let ordered = centres.sort(key: |cell| accumulator[cell], reverse: true)

    var out: list[HoughCircle] = []
    for cell in ordered {
        let ay = cell // acc_cols
        let ax = cell - ay * acc_cols
        let cx = (float(ax) + 0.5) * dp
        let cy = (float(ay) + 0.5) * dp
        var clear = true
        for existing in out {
            let ex = existing.centre.x - cx
            let ey = existing.centre.y - cy
            if math.sqrt(ex * ex + ey * ey) < min_dist {
                clear = false
                break
            }
        }
        if not clear { continue }

        #: The radius is whichever distance the most edge pixels around this
        #: centre agree on.
        var votes: list[int] = []
        for _index in 0..upper + 1 { votes.push(0) }
        for edge in edges {
            let ey = edge // cols
            let ex = edge - ey * cols
            let gap = math.sqrt(
                (float(ex) - cx) * (float(ex) - cx) + (float(ey) - cy) * (float(ey) - cy)
            )
            let radius = int(gap + 0.5)
            if radius < lower or radius > upper { continue }
            votes[radius] += 1
        }
        var best = lower
        for radius in lower..upper + 1 {
            if votes[radius] > votes[best] { best = radius }
        }
        if votes[best] == 0 { continue }
        out.push(HoughCircle(Point2f(cx, cy), float(best), accumulator[cell]))
    }
    return out
}
