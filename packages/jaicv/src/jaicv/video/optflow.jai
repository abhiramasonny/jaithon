#: Optical flow: where each part of the image went between two frames.
#:
#: `calc_optical_flow_pyr_lk` follows a handful of points and is what tracking
#: is usually built on; `calc_optical_flow_farneback` produces a vector for
#: every pixel. Both work down a pyramid, coarse level first, so a movement
#: larger than the window can still be found — which is the whole reason the
#: pyramid is there.
#:
#: Lucas and Kanade's method is small enough to run on the host: it touches
#: only the windows around the points being followed. Farneback's touches every
#: pixel several times over, so its two heavy steps — fitting a quadratic to
#: every neighbourhood, and turning those into a flow field — are kernels.
#:
#: ```
#: let (moved, found, error) = calc_optical_flow_pyr_lk(previous, current, points)
#: let flow = calc_optical_flow_farneback(previous, current)
#: ```
import std.math
from std.gpu import Kernel
from ..core.mat import Mat, f32_bits
from ..core.types import CV_32F, CV_MAKETYPE, Point2f, Size, TermCriteria
from ..imgproc.common import PRELUDE
from ..imgproc.filter import gaussian_blur
from ..imgproc.geometric import pyr_down, resize

#: Take the initial guess for each point from the value already in it.
pub const OPTFLOW_USE_INITIAL_FLOW = 4
#: Use the flow already in the output as the starting point.
pub const OPTFLOW_FARNEBACK_GAUSSIAN = 256

let SOURCE = PRELUDE + """
/* Fit a quadratic to the neighbourhood of every pixel and keep its five
 * coefficients. Farneback's method rests on this: two frames agree on where a
 * pixel went exactly when their two quadratics agree after a shift. */
kernel void flow_poly_expand(device const float* src [[buffer(0)]],
                             device float* dst [[buffer(1)]],
                             device const float* weights [[buffer(2)]],
                             constant uint& count [[buffer(3)]],
                             constant uint& rows [[buffer(4)]],
                             constant uint& cols [[buffer(5)]],
                             constant uint& src_off [[buffer(6)]],
                             constant uint& src_step [[buffer(7)]],
                             constant uint& dst_off [[buffer(8)]],
                             constant uint& dst_step [[buffer(9)]],
                             constant uint& radius [[buffer(10)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint y = id / cols;
    uint x = id - y * cols;

    /* The six sums of a weighted least-squares fit against the basis
     * {1, x, y, x*x, y*y, x*y}, which the separable weights make orthogonal
     * enough that the normal equations come apart into the closed form used
     * below. */
    float s0 = 0.0f, sx = 0.0f, sy = 0.0f, sxx = 0.0f, syy = 0.0f, sxy = 0.0f;
    float w0 = 0.0f, wxx = 0.0f, wxxxx = 0.0f, wxxyy = 0.0f;
    int r = int(radius);
    for (int dy = -r; dy <= r; ++dy) {
        for (int dx = -r; dx <= r; ++dx) {
            float w = weights[uint(dy + r)] * weights[uint(dx + r)];
            int yy = clamp(int(y) + dy, 0, int(rows) - 1);
            int xx = clamp(int(x) + dx, 0, int(cols) - 1);
            float v = src[src_off + uint(yy) * src_step + uint(xx)];
            float fx = float(dx);
            float fy = float(dy);
            s0 += w * v;
            sx += w * fx * v;
            sy += w * fy * v;
            sxx += w * fx * fx * v;
            syy += w * fy * fy * v;
            sxy += w * fx * fy * v;
            w0 += w;
            wxx += w * fx * fx;
            wxxxx += w * fx * fx * fx * fx;
            wxxyy += w * fx * fx * fy * fy;
        }
    }

    /* With separable symmetric weights the basis splits: x, y and x*y each
     * stand alone, while 1, x*x and y*y form a coupled three by three. Fitting
     * x*x against the constant alone, and ignoring that y*y pulls on it too,
     * biases both quadratic terms and tilts the flow that comes out of them.
     * The sum and the difference of the two diagonal terms separate that
     * system, which is why they are solved for rather than the terms. */
    float spread = w0 * (wxxxx + wxxyy) - 2.0f * wxx * wxx;
    float tilt = wxxxx - wxxyy;
    float total = spread != 0.0f ? (w0 * (sxx + syy) - 2.0f * wxx * s0) / spread : 0.0f;
    float difference = tilt != 0.0f ? (sxx - syy) / tilt : 0.0f;

    uint base = dst_off + y * dst_step + x * 5u;
    dst[base] = wxx != 0.0f ? sx / wxx : 0.0f;
    dst[base + 1] = wxx != 0.0f ? sy / wxx : 0.0f;
    dst[base + 2] = (total + difference) * 0.5f;
    dst[base + 3] = (total - difference) * 0.5f;
    dst[base + 4] = wxxyy != 0.0f ? sxy / wxxyy : 0.0f;
}

/* One step of the flow update: the two quadratics, taken at the position the
 * current flow points to, give a small linear system per pixel whose solution
 * is the correction. */
kernel void flow_update(device const float* first [[buffer(0)]],
                        device const float* second [[buffer(1)]],
                        device const float* flow_in [[buffer(2)]],
                        device float* flow_out [[buffer(3)]],
                        constant uint& count [[buffer(4)]],
                        constant uint& rows [[buffer(5)]],
                        constant uint& cols [[buffer(6)]],
                        constant uint& poly_step [[buffer(7)]],
                        constant uint& flow_step [[buffer(8)]],
                        constant uint& window [[buffer(9)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint y = id / cols;
    uint x = id - y * cols;

    float m11 = 0.0f, m12 = 0.0f, m22 = 0.0f, r1 = 0.0f, r2 = 0.0f;
    int w = int(window);
    /* Weighting the window rather than averaging it flat is what keeps one
     * bad pixel — the edge of the frame, where content has left it — from
     * dragging its whole neighbourhood along with it. */
    float sigma = max(float(w) * 0.3f, 0.5f);
    float falloff = -1.0f / (2.0f * sigma * sigma);
    for (int dy = -w; dy <= w; ++dy) {
        for (int dx = -w; dx <= w; ++dx) {
            float weight = exp(float(dx * dx + dy * dy) * falloff);
            int yy = clamp(int(y) + dy, 0, int(rows) - 1);
            int xx = clamp(int(x) + dx, 0, int(cols) - 1);
            uint here = uint(yy) * poly_step + uint(xx) * 5u;
            uint fhere = uint(yy) * flow_step + uint(xx) * 2u;

            float fx = flow_in[fhere];
            float fy = flow_in[fhere + 1];
            int ty = clamp(yy + int(rint(fy)), 0, int(rows) - 1);
            int tx = clamp(xx + int(rint(fx)), 0, int(cols) - 1);
            uint there = uint(ty) * poly_step + uint(tx) * 5u;

            /* The averaged quadratic of the two frames, and the difference of
             * their linear parts corrected for where the flow already points. */
            float a11 = (first[here + 2] + second[there + 2]) * 0.5f;
            float a22 = (first[here + 3] + second[there + 3]) * 0.5f;
            float a12 = (first[here + 4] + second[there + 4]) * 0.25f;
            float db1 = (first[here] - second[there]) * 0.5f;
            float db2 = (first[here + 1] - second[there + 1]) * 0.5f;
            db1 += a11 * fx + a12 * fy;
            db2 += a12 * fx + a22 * fy;

            m11 += weight * (a11 * a11 + a12 * a12);
            m12 += weight * a12 * (a11 + a22);
            m22 += weight * (a22 * a22 + a12 * a12);
            r1 += weight * (a11 * db1 + a12 * db2);
            r2 += weight * (a12 * db1 + a22 * db2);
        }
    }

    /* The small constant keeps a flat patch, where the system says nothing,
     * from turning into an arbitrarily large correction. */
    float determinant = m11 * m22 - m12 * m12 + 1e-3f;
    uint out_base = y * flow_step + x * 2u;
    flow_out[out_base] = (m22 * r1 - m12 * r2) / determinant;
    flow_out[out_base + 1] = (m11 * r2 - m12 * r1) / determinant;
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

fn sample_bilinear(values: list[float], rows: int, cols: int, x: float, y: float) -> float {
    var fx = x
    var fy = y
    if fx < 0.0 { fx = 0.0 }
    if fy < 0.0 { fy = 0.0 }
    if fx > float(cols - 1) { fx = float(cols - 1) }
    if fy > float(rows - 1) { fy = float(rows - 1) }
    let x0 = int(fx)
    let y0 = int(fy)
    let x1 = min(x0 + 1, cols - 1)
    let y1 = min(y0 + 1, rows - 1)
    let ax = fx - float(x0)
    let ay = fy - float(y0)
    let top = values[y0 * cols + x0] * (1.0 - ax) + values[y0 * cols + x1] * ax
    let bottom = values[y1 * cols + x0] * (1.0 - ax) + values[y1 * cols + x1] * ax
    return top * (1.0 - ay) + bottom * ay
}

fn shrink(image: Mat, width: int, height: int, scale: float) -> Mat {
    if width == image.cols and height == image.rows { return image }
    let sigma = (1.0 / scale - 1.0) * 0.5
    var window = int(math.floor(sigma * 5.0 + 0.5)) | 1
    if window < 3 { window = 3 }
    let smoothed = gaussian_blur(image, Size(window, window), sigma)
    return resize(smoothed, size: Size(width, height))
}

fn build_pyramid(image: Mat, levels: int) -> list[Mat] {
    var out = [image]
    var current = image
    for _level in 1..levels {
        if current.rows < 8 or current.cols < 8 { break }
        current = pyr_down(current)
        out.push(current)
    }
    return out
}

#: Follow a set of points from one frame to the next.
#:
#: Returns the points as they moved, whether each was found, and how far the
#: two windows still differ once it was. A point that leaves the image, or
#: whose window has no gradient to lock onto, comes back not found.
pub fn calc_optical_flow_pyr_lk(
    previous: Mat,
    current: Mat,
    points: list[Point2f],
    window: Size = Size(21, 21),
    max_level: int = 3,
    iterations: int = 30,
    epsilon: float = 0.01
) -> tuple[list[Point2f], list[bool], list[float]] {
    if previous.channels() != 1 or current.channels() != 1 {
        throw ValueError("optical flow needs single-channel frames")
    }
    if previous.rows != current.rows or previous.cols != current.cols {
        throw ValueError("both frames have to be the same size")
    }
    previous.flush()
    current.flush()

    let first = build_pyramid(previous, max_level + 1)
    let second = build_pyramid(current, max_level + 1)
    var planes_a: list[list[float]] = []
    var planes_b: list[list[float]] = []
    for level in 0..first.len() {
        planes_a.push(first[level].to_list())
        planes_b.push(second[level].to_list())
    }

    let half_x = window.width // 2
    let half_y = window.height // 2
    var moved: list[Point2f] = []
    var found: list[bool] = []
    var errors: list[float] = []

    for point in points {
        var guess_x = 0.0
        var guess_y = 0.0
        var alive = true
        var residual = 0.0

        var level = first.len() - 1
        while level >= 0 {
            let scale = float(1 << level)
            let px = point.x / scale
            let py = point.y / scale
            let rows = first[level].rows
            let cols = first[level].cols
            let a = planes_a[level]
            let b = planes_b[level]

            #: The window's own gradient matrix does not change while the
            #: point is being refined, so it is built once per level.
            var gxx = 0.0
            var gxy = 0.0
            var gyy = 0.0
            for dy in -half_y..half_y + 1 {
                for dx in -half_x..half_x + 1 {
                    let sx = px + float(dx)
                    let sy = py + float(dy)
                    let ix = (sample_bilinear(a, rows, cols, sx + 1.0, sy) - sample_bilinear(a, rows, cols, sx - 1.0, sy)) *
                        0.5
                    let iy = (sample_bilinear(a, rows, cols, sx, sy + 1.0) - sample_bilinear(a, rows, cols, sx, sy - 1.0)) *
                        0.5
                    gxx += ix * ix
                    gxy += ix * iy
                    gyy += iy * iy
                }
            }
            let determinant = gxx * gyy - gxy * gxy
            if math.fabs(determinant) < 1e-6 {
                alive = false
                level -= 1
                guess_x *= 2.0
                guess_y *= 2.0
                continue
            }

            for _pass in 0..iterations {
                var bx = 0.0
                var by = 0.0
                var total = 0.0
                for dy in -half_y..half_y + 1 {
                    for dx in -half_x..half_x + 1 {
                        let sx = px + float(dx)
                        let sy = py + float(dy)
                        let ix = (sample_bilinear(a, rows, cols, sx + 1.0, sy) - sample_bilinear(a, rows, cols, sx - 1.0, sy)) *
                            0.5
                        let iy = (sample_bilinear(a, rows, cols, sx, sy + 1.0) - sample_bilinear(a, rows, cols, sx, sy - 1.0)) *
                            0.5
                        let difference = sample_bilinear(a, rows, cols, sx, sy) -
                            sample_bilinear(b, rows, cols, sx + guess_x, sy + guess_y)
                        bx += difference * ix
                        by += difference * iy
                        total += math.fabs(difference)
                    }
                }
                let step_x = (gyy * bx - gxy * by) / determinant
                let step_y = (gxx * by - gxy * bx) / determinant
                guess_x += step_x
                guess_y += step_y
                residual = total / float(window.width * window.height)
                if step_x * step_x + step_y * step_y < epsilon * epsilon { break }
            }

            level -= 1
            if level >= 0 {
                guess_x *= 2.0
                guess_y *= 2.0
            }
        }

        let landed = Point2f(point.x + guess_x, point.y + guess_y)
        if landed.x < 0.0 or landed.y < 0.0 { alive = false }
        if landed.x > float(previous.cols - 1) or landed.y > float(previous.rows - 1) {
            alive = false
        }
        moved.push(landed)
        found.push(alive)
        errors.push(residual)
    }
    return (moved, found, errors)
}

fn gaussian_weights(radius: int, sigma: float) -> Mat {
    var values: list[float] = []
    var total = 0.0
    for index in -radius..radius + 1 {
        let value = math.exp(-float(index * index) / (2.0 * sigma * sigma))
        values.push(value)
        total += value
    }
    for index in 0..values.len() { values[index] /= total }
    return Mat.from_list(1, values.len(), CV_MAKETYPE(CV_32F, 1), values)
}

#: The five quadratic coefficients fitted to every neighbourhood: the two
#: linear terms, then the two squared terms and the cross term.
pub fn polynomial_expansion(image: Mat, radius: int, sigma: float) -> Mat {
    let weights = gaussian_weights(radius, sigma)
    let out = Mat.zeros(image.rows, image.cols, CV_MAKETYPE(CV_32F, 5))
    let count = image.rows * image.cols
    if count == 0 { return out }
    kernel_for("flow_poly_expand").dispatch_async(
        [image.storage, out.storage, weights.storage],
        count,
        [count, image.rows, image.cols, image.offset, image.step_, out.offset, out.step_, radius]
    )
    out.invalidate()
    return out
}

#: Dense optical flow by Farneback's method.
#:
#: Every pixel gets a quadratic fitted to its neighbourhood in both frames; two
#: such quadratics that describe the same patch differ by exactly the shift
#: between them, and solving for that shift over a window gives the flow. The
#: result is a two-channel `CV_32F` image holding the horizontal and vertical
#: movement of each pixel.
#:
#: This is Farneback's method and behaves like OpenCV's, but the constants
#: inside — the weighting of the fit and the averaging between iterations —
#: follow the paper rather than OpenCV's tuned version, so the flow fields are
#: close rather than identical. On a translated frame the flow comes back
#: within about a twentieth of a pixel over one or two levels. More levels find
#: larger movements but carry more of the frame's own edges inward with them,
#: where content has left the picture and there is no honest answer; if the
#: motion is small, fewer levels are more accurate.
pub fn calc_optical_flow_farneback(
    previous: Mat,
    current: Mat,
    pyramid_scale: float = 0.5,
    levels: int = 3,
    window: int = 15,
    iterations: int = 3,
    poly_n: int = 5,
    poly_sigma: float = 1.2
) -> Mat {
    if previous.channels() != 1 or current.channels() != 1 {
        throw ValueError("Farneback flow needs single-channel frames")
    }
    if previous.rows != current.rows or previous.cols != current.cols {
        throw ValueError("both frames have to be the same size")
    }
    if pyramid_scale <= 0.0 or pyramid_scale >= 1.0 {
        throw ValueError("the pyramid scale has to be between zero and one")
    }
    previous.flush()
    current.flush()

    let radius = poly_n // 2
    var flow: Mat? = null
    var level = levels - 1
    while level >= 0 {
        let scale = pyramid_scale ** float(level)
        let width = max(int(float(previous.cols) * scale), 8)
        let height = max(int(float(previous.rows) * scale), 8)
        #: A coarse level has to be blurred before it is shrunk. Sampling a
        #: textured image straight down by four folds its detail back on itself
        #: as something that looks like motion, and the coarse level's job is
        #: precisely to be believed about motion.
        let a = shrink(previous, width, height, scale)
        let b = shrink(current, width, height, scale)

        var here = Mat.zeros(a.rows, a.cols, CV_MAKETYPE(CV_32F, 2))
        let coarser = flow
        if coarser is not null {
            #: The flow from the coarser level is the starting guess here, and
            #: it has to grow with the image it now describes.
            let lifted = resize(coarser, size: Size(a.cols, a.rows))
            let values = lifted.to_list()
            let factor = float(a.cols) / float(coarser.cols)
            var scaled: list[float] = []
            for value in values { scaled.push(value * factor) }
            here = Mat.from_list(a.rows, a.cols, CV_MAKETYPE(CV_32F, 2), scaled)
        }

        let first = polynomial_expansion(a, radius, poly_sigma)
        let second = polynomial_expansion(b, radius, poly_sigma)
        let count = a.rows * a.cols
        for _pass in 0..iterations {
            let next = Mat.zeros(a.rows, a.cols, CV_MAKETYPE(CV_32F, 2))
            kernel_for("flow_update").dispatch_async(
                [first.storage, second.storage, here.storage, next.storage],
                count,
                [count, a.rows, a.cols, first.step_, here.step_, window // 2]
            )
            next.invalidate()
            here = next
        }
        flow = here
        level -= 1
    }

    let result = flow
    if result is null { return Mat.zeros(previous.rows, previous.cols, CV_MAKETYPE(CV_32F, 2)) }
    return result
}

#: Turn a flow field into something a person can look at.
#:
#: Direction becomes hue and speed becomes brightness, which is the usual way
#: these are shown.
pub fn flow_to_colour(flow: Mat, max_magnitude: float = 0.0) -> Mat {
    if flow.channels() != 2 { throw ValueError("a flow field has two channels") }
    let values = flow.to_list()
    var peak = max_magnitude
    if peak <= 0.0 {
        for index in 0..flow.rows * flow.cols {
            let fx = values[index * 2]
            let fy = values[index * 2 + 1]
            let size = math.sqrt(fx * fx + fy * fy)
            if size > peak { peak = size }
        }
    }
    if peak <= 0.0 { peak = 1.0 }

    var out: list[float] = []
    for index in 0..flow.rows * flow.cols {
        let fx = values[index * 2]
        let fy = values[index * 2 + 1]
        var angle = math.atan2(fy, fx) * 180.0 / math.PI
        if angle < 0.0 { angle += 360.0 }
        let strength = min(math.sqrt(fx * fx + fy * fy) / peak, 1.0)
        #: Straight from hue and value to blue, green, red, without going
        #: through a colour conversion for what is only ever a picture.
        let sector = angle / 60.0
        let index_of = int(sector) % 6
        let fraction = sector - float(int(sector))
        let v = strength * 255.0
        let p = 0.0
        let q = v * (1.0 - fraction)
        let t = v * fraction
        var r = 0.0
        var g = 0.0
        var b = 0.0
        if index_of == 0 {
            r = v
            g = t
            b = p
        } elif index_of == 1 {
            r = q
            g = v
            b = p
        } elif index_of == 2 {
            r = p
            g = v
            b = t
        } elif index_of == 3 {
            r = p
            g = q
            b = v
        } elif index_of == 4 {
            r = t
            g = p
            b = v
        } else {
            r = v
            g = p
            b = q
        }
        out.push(b)
        out.push(g)
        out.push(r)
    }
    return Mat.from_list(flow.rows, flow.cols, CV_MAKETYPE(0, 3), out)
}
