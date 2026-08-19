#: Depth from two cameras: block matching, semi-global matching, rectification,
#: and turning a disparity map into points in space.
#:
#: ```
#: let matcher = StereoBM(num_disparities: 64, block_size: 15)
#: let disparity = matcher.compute(left_gray, right_gray)
#: let cloud = reproject_image_to_3d(disparity, q)
#: ```
#:
#: Both matchers assume the pair is RECTIFIED: that a point in the left image
#: lies on the same row in the right one, so the search is along a row. That is
#: what `stereo_rectify` produces, and a pair that has not been through it will
#: match nothing useful.
#:
#: Disparities are returned as OpenCV returns them — sixteenths of a pixel in a
#: `CV_16S`-valued `Mat` — so `disparity / 16.0` is the number of pixels, and a
#: pixel with no match holds `(min_disparity - 1) * 16`.
#:
#: The cost volume and its aggregation run on the device: a disparity search is
#: the same small window swept over every pixel at every candidate shift, which
#: is exactly what a kernel is for. Choosing the winner and the sub-pixel fit
#: are on the host, over the aggregated volume.
import std.math
from std.gpu import Kernel
from ..core.linalg import inverse, matmul, transpose
from ..core.mat import Mat, f32_bits
from ..core.types import CV_16S, CV_32F, CV_MAKETYPE, Point3f, Rect, Size
from ..imgproc.common import PRELUDE
from .transform import rodrigues

#: What a pixel with no match holds, before the sixteenths are applied.
pub const STEREO_NO_MATCH = -1

let SOURCE = PRELUDE + """
/* One cost per pixel per candidate disparity: the sum of absolute differences
 * over a window, on the left image against the right shifted by `d`.
 *
 * The volume is laid out disparity-fastest so the aggregation below reads it
 * along a row with a stride of one for the whole disparity range, which is the
 * order every path wants. */
kernel void stereo_sad_volume(device const float* left [[buffer(0)]],
                              device const float* right [[buffer(1)]],
                              device float* cost [[buffer(2)]],
                              constant uint& count [[buffer(3)]],
                              constant uint& rows [[buffer(4)]],
                              constant uint& cols [[buffer(5)]],
                              constant uint& radius [[buffer(6)]],
                              constant uint& min_disp [[buffer(7)]],
                              constant uint& num_disp [[buffer(8)]],
                              constant uint& penalty_bits [[buffer(9)]],
                              uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint d = id % num_disp;
    uint pixel = id / num_disp;
    uint x = pixel % cols;
    uint y = pixel / cols;
    int shift = int(min_disp) + int(d);
    float unmatched = as_type<float>(penalty_bits);

    int rx = int(x) - shift;
    if (rx < 0 || rx >= int(cols)) {
        cost[id] = unmatched;
        return;
    }
    float total = 0.0f;
    uint seen = 0u;
    int r = int(radius);
    for (int dy = -r; dy <= r; ++dy) {
        int sy = int(y) + dy;
        if (sy < 0 || sy >= int(rows)) continue;
        for (int dx = -r; dx <= r; ++dx) {
            int lx = int(x) + dx;
            int qx = rx + dx;
            if (lx < 0 || lx >= int(cols) || qx < 0 || qx >= int(cols)) continue;
            total += fabs(left[sy * cols + lx] - right[sy * cols + qx]);
            seen += 1u;
        }
    }
    /* Averaged rather than summed, so a window clipped at the border is
     * comparable with a whole one and the edge of the picture does not always
     * prefer the smallest disparity. */
    cost[id] = seen == 0u ? unmatched : total / float(seen);
}

/* One semi-global path: the running cost along a direction, with a small
 * penalty for a disparity that steps by one and a large one for a jump.
 *
 * A thread takes one path from one border pixel and walks it to the far side,
 * which is what makes this sequential per path and parallel across them. The
 * eight directions are separate dispatches so each can index its own start. */
kernel void stereo_aggregate(device const float* cost [[buffer(0)]],
                             device float* out [[buffer(1)]],
                             constant uint& count [[buffer(2)]],
                             constant uint& rows [[buffer(3)]],
                             constant uint& cols [[buffer(4)]],
                             constant uint& num_disp [[buffer(5)]],
                             constant uint& step_y_biased [[buffer(6)]],
                             constant uint& step_x_biased [[buffer(7)]],
                             constant uint& p1_bits [[buffer(8)]],
                             constant uint& p2_bits [[buffer(9)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    /* The dispatch carries unsigned scalars, so a direction of -1, 0 or +1
     * travels as 0, 1 or 2 and is unbiased here. */
    int step_y = int(step_y_biased) - 1;
    int step_x = int(step_x_biased) - 1;
    float p1 = as_type<float>(p1_bits);
    float p2 = as_type<float>(p2_bits);

    /* Where this path starts: a border pixel on the side the direction comes
     * from. `count` is the number of paths, not of pixels. */
    int y = 0;
    int x = 0;
    if (step_y == 0) {
        y = int(id);
        x = step_x > 0 ? 0 : int(cols) - 1;
    } else if (step_x == 0) {
        x = int(id);
        y = step_y > 0 ? 0 : int(rows) - 1;
    } else {
        /* A diagonal starts on the top or bottom edge, then on the left or
         * right edge for the rest. */
        if (id < cols) {
            x = int(id);
            y = step_y > 0 ? 0 : int(rows) - 1;
        } else {
            y = int(id - cols);
            x = step_x > 0 ? 0 : int(cols) - 1;
        }
    }

    float previous[256];
    float best_previous = 0.0f;
    bool first = true;
    while (y >= 0 && y < int(rows) && x >= 0 && x < int(cols)) {
        uint base = (uint(y) * cols + uint(x)) * num_disp;
        float best_now = INFINITY;
        for (uint d = 0; d < num_disp; ++d) {
            float value = cost[base + d];
            if (!first) {
                float best = previous[d];
                float left = d > 0u ? previous[d - 1u] + p1 : INFINITY;
                float right = d + 1u < num_disp ? previous[d + 1u] + p1 : INFINITY;
                float jump = best_previous + p2;
                best = min(best, left);
                best = min(best, right);
                best = min(best, jump);
                /* The running minimum is subtracted so the cost cannot grow
                 * without bound along a long path. */
                value += best - best_previous;
            }
            previous[d] = value;
            out[base + d] += value;
            best_now = min(best_now, value);
        }
        best_previous = best_now;
        first = false;
        y += step_y;
        x += step_x;
    }
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

fn as_gray(image: Mat, side: str) -> Mat {
    if image.channels() != 1 {
        throw ValueError(f"the {side} image has to be single-channel, got {image.channels()}")
    }
    image.flush()
    return image
}

fn check_pair(left: Mat, right: Mat) -> void {
    if left.rows != right.rows or left.cols != right.cols {
        throw ValueError("both images have to be the same size")
    }
}

#: Pick the winning disparity for every pixel, with the sub-pixel refinement
#: and the uniqueness and texture tests OpenCV applies.
fn winners(
    cost: list[float],
    rows: int,
    cols: int,
    num_disp: int,
    min_disp: int,
    uniqueness: float,
    speckle_window: int,
    speckle_range: int
) -> Mat {
    var out: list[float] = []
    for _index in 0..rows * cols { out.push(float((min_disp - 1) * 16)) }

    for pixel in 0..rows * cols {
        let base = pixel * num_disp
        var best = 0
        var best_cost = cost[base]
        for d in 1..num_disp {
            if cost[base + d] < best_cost {
                best_cost = cost[base + d]
                best = d
            }
        }

        #: The winner has to beat every other disparity that is not its
        #: neighbour by the uniqueness margin. Without it a flat region matches
        #: everywhere equally and the answer is whichever rounding won.
        if uniqueness > 0.0 {
            var runner_up = 1e30
            for d in 0..num_disp {
                if d >= best - 1 and d <= best + 1 { continue }
                if cost[base + d] < runner_up { runner_up = cost[base + d] }
            }
            if runner_up < 1e29 and runner_up * (1.0 - uniqueness) < best_cost { continue }
        }

        #: A parabola through the winner and its two neighbours, which is what
        #: gives the sixteenths their meaning.
        var refined = float(best)
        if best > 0 and best < num_disp - 1 {
            let low = cost[base + best - 1]
            let mid = cost[base + best]
            let high = cost[base + best + 1]
            let denominator = low + high - 2.0 * mid
            if math.fabs(denominator) > 1e-12 {
                let step = (low - high) / (2.0 * denominator)
                if step > -1.0 and step < 1.0 { refined += step }
            }
        }
        out[pixel] = (float(min_disp) + refined) * 16.0
    }

    if speckle_window > 0 {
        remove_speckles(out, rows, cols, speckle_window, speckle_range, min_disp)
    }
    return Mat.from_list(rows, cols, CV_MAKETYPE(CV_16S, 1), out)
}

#: Drop connected regions smaller than `window` whose disparities all sit
#: within `range` of each other — the small wrong patches a matcher leaves in
#: texture-less areas.
fn remove_speckles(
    values: list[float],
    rows: int,
    cols: int,
    window: int,
    range_allowed: int,
    min_disp: int
) -> void {
    let invalid = float((min_disp - 1) * 16)
    var seen: list[bool] = []
    for _index in 0..rows * cols { seen.push(false) }
    let spread = float(range_allowed * 16)

    for start in 0..rows * cols {
        if seen[start] or values[start] == invalid { continue }
        var region = [start]
        var stack = [start]
        seen[start] = true
        while stack.len() > 0 {
            let at = stack.pop()
            let y = at // cols
            let x = at % cols
            for step in 0..4 {
                let ny = y + [0, 0, -1, 1][step]
                let nx = x + [-1, 1, 0, 0][step]
                if ny < 0 or ny >= rows or nx < 0 or nx >= cols { continue }
                let next = ny * cols + nx
                if seen[next] or values[next] == invalid { continue }
                if math.fabs(values[next] - values[at]) > spread { continue }
                seen[next] = true
                region.push(next)
                stack.push(next)
            }
        }
        if region.len() >= window { continue }
        for index in region { values[index] = invalid }
    }
}

#: Block matching: the sum of absolute differences over a window, per pixel per
#: disparity, with the smallest winning.
#:
#: Fast and local. It has no idea that neighbouring pixels are usually at
#: similar depths, which is what `StereoSGBM` adds and why a block matcher's
#: output is noisy where the picture has no texture.
pub class StereoBM {
    #: Where the search starts. Usually zero.
    pub var min_disparity: int
    #: How many disparities to try. A multiple of sixteen, as OpenCV requires.
    pub var num_disparities: int
    #: The matching window, in pixels. Odd.
    pub var block_size: int
    #: How much better the winner has to be than the next unrelated candidate,
    #: as a fraction. Zero turns the test off.
    pub var uniqueness_ratio: float
    #: Regions smaller than this many pixels are discarded. Zero turns it off.
    pub var speckle_window_size: int
    #: How far two disparities may differ and still be one region.
    pub var speckle_range: int

    pub fn init(self, num_disparities: int = 64, block_size: int = 15, min_disparity: int = 0) {
        if num_disparities <= 0 or num_disparities % 16 != 0 {
            throw ValueError("num_disparities has to be a positive multiple of sixteen")
        }
        if block_size < 3 or block_size % 2 == 0 {
            throw ValueError("block_size has to be odd and at least three")
        }
        if num_disparities > 256 {
            throw ValueError("this matcher searches at most 256 disparities")
        }
        self.num_disparities = num_disparities
        self.block_size = block_size
        self.min_disparity = min_disparity
        self.uniqueness_ratio = 0.15
        self.speckle_window_size = 0
        self.speckle_range = 1
    }

    #: The disparity of every pixel of `left`, in sixteenths of a pixel.
    pub fn compute(self, left: Mat, right: Mat) -> Mat {
        check_pair(left, right)
        let a = as_gray(left, "left")
        let b = as_gray(right, "right")
        let rows = a.rows
        let cols = a.cols
        let volume = Mat.zeros(rows * cols, self.num_disparities, CV_MAKETYPE(CV_32F, 1))
        kernel_for("stereo_sad_volume").dispatch_async(
            [a.storage, b.storage, volume.storage],
            rows * cols * self.num_disparities,
            [
                rows * cols * self.num_disparities,
                rows,
                cols,
                self.block_size // 2,
                self.min_disparity,
                self.num_disparities,
                f32_bits(1e9),
            ]
        )
        volume.invalidate()
        volume.flush()
        return winners(
            volume.values(),
            rows,
            cols,
            self.num_disparities,
            self.min_disparity,
            self.uniqueness_ratio,
            self.speckle_window_size,
            self.speckle_range
        )
    }
}

#: Semi-global matching: the same per-pixel costs, then aggregated along
#: several directions so that a disparity is only cheap if its neighbours agree.
#:
#: `p1` is what a step of one disparity costs and `p2` what a jump costs. The
#: two together are what make a surface come out smooth and an edge stay sharp,
#: and they are the only parameters that matter much.
pub class StereoSGBM {
    pub var min_disparity: int
    pub var num_disparities: int
    pub var block_size: int
    #: The penalty for a neighbour one disparity away.
    pub var p1: float
    #: The penalty for a neighbour further than one away.
    pub var p2: float
    pub var uniqueness_ratio: float
    pub var speckle_window_size: int
    pub var speckle_range: int
    #: How many directions to aggregate along: 4 or 8. Eight is better and
    #: twice the work.
    pub var directions: int

    pub fn init(
        self,
        num_disparities: int = 64,
        block_size: int = 5,
        min_disparity: int = 0,
        p1: float = 0.0,
        p2: float = 0.0
    ) {
        if num_disparities <= 0 or num_disparities % 16 != 0 {
            throw ValueError("num_disparities has to be a positive multiple of sixteen")
        }
        if block_size < 1 or block_size % 2 == 0 {
            throw ValueError("block_size has to be odd")
        }
        if num_disparities > 256 {
            throw ValueError("this matcher searches at most 256 disparities")
        }
        self.num_disparities = num_disparities
        self.block_size = block_size
        self.min_disparity = min_disparity
        #: OpenCV's defaults, scaled for costs that are means rather than sums:
        #: eight and thirty-two times the channel count times the window area,
        #: divided back out by the area.
        self.p1 = if p1 > 0.0 { p1 } else { 8.0 }
        self.p2 = if p2 > 0.0 { p2 } else { 32.0 }
        self.uniqueness_ratio = 0.10
        self.speckle_window_size = 100
        self.speckle_range = 2
        self.directions = 8
    }

    pub fn compute(self, left: Mat, right: Mat) -> Mat {
        check_pair(left, right)
        let a = as_gray(left, "left")
        let b = as_gray(right, "right")
        let rows = a.rows
        let cols = a.cols
        let count = rows * cols * self.num_disparities
        let volume = Mat.zeros(rows * cols, self.num_disparities, CV_MAKETYPE(CV_32F, 1))
        kernel_for("stereo_sad_volume").dispatch_async(
            [a.storage, b.storage, volume.storage],
            count,
            [
                count,
                rows,
                cols,
                self.block_size // 2,
                self.min_disparity,
                self.num_disparities,
                f32_bits(1e9),
            ]
        )
        volume.invalidate()

        let aggregated = Mat.zeros(rows * cols, self.num_disparities, CV_MAKETYPE(CV_32F, 1))
        var steps = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        if self.directions >= 8 {
            steps.push([1, 1])
            steps.push([1, -1])
            steps.push([-1, 1])
            steps.push([-1, -1])
        }
        for step in steps {
            let paths = if step[0] == 0 { rows } elif step[1] == 0 { cols } else { rows + cols }
            kernel_for("stereo_aggregate").dispatch_async(
                [volume.storage, aggregated.storage],
                paths,
                [
                    paths,
                    rows,
                    cols,
                    self.num_disparities,
                    step[0] + 1,
                    step[1] + 1,
                    f32_bits(self.p1),
                    f32_bits(self.p2),
                ]
            )
        }
        aggregated.invalidate()
        aggregated.flush()
        return winners(
            aggregated.values(),
            rows,
            cols,
            self.num_disparities,
            self.min_disparity,
            self.uniqueness_ratio,
            self.speckle_window_size,
            self.speckle_range
        )
    }
}

#: Turn a disparity map into points in space, given the reprojection matrix
#: `stereo_rectify` produced.
#:
#: Returns one `Point3f` a pixel, in the same order the image is stored. A
#: pixel with no disparity comes back at a very large depth, which is what
#: OpenCV does and what `handle_missing_values` is for.
pub fn reproject_image_to_3d(
    disparity: Mat,
    q: Mat,
    handle_missing_values: bool = false
) -> list[Point3f] {
    if q.rows != 4 or q.cols != 4 { throw ValueError("the reprojection matrix is four by four") }
    disparity.flush()
    let m = q.values()
    let values = disparity.to_list()
    var out: list[Point3f] = []
    for y in 0..disparity.rows {
        for x in 0..disparity.cols {
            let d = values[y * disparity.cols + x] / 16.0
            let fx = float(x)
            let fy = float(y)
            let w = m[12] * fx + m[13] * fy + m[14] * d + m[15]
            if math.fabs(w) < 1e-12 or handle_missing_values and d <= 0.0 {
                out.push(Point3f(10000.0, 10000.0, 10000.0))
                continue
            }
            out.push(
                Point3f(
                    (m[0] * fx + m[1] * fy + m[2] * d + m[3]) / w,
                    (m[4] * fx + m[5] * fy + m[6] * d + m[7]) / w,
                    (m[8] * fx + m[9] * fy + m[10] * d + m[11]) / w
                )
            )
        }
    }
    return out
}

#: The rotations and projections that put two cameras' rows in step, and the
#: matrix that turns a disparity back into depth.
#:
#: Returns `(r1, r2, p1, p2, q, roi1, roi2)` in OpenCV's order. `r` is the
#: rotation to apply to each camera, `p` its new projection matrix, and `q` the
#: four by four `reproject_image_to_3d` takes.
#:
#: Bouguet's method, which is what OpenCV uses when it is given no flags: half
#: the relative rotation is given to each camera, and the pair is then turned
#: so the baseline lies along the new x axis. What is not here is the alpha
#: parameter and the valid-region search behind it — the regions come back as
#: the whole image.
pub fn stereo_rectify(
    camera1: Mat,
    dist1: list[float],
    camera2: Mat,
    dist2: list[float],
    image_size: Size,
    rotation: Mat,
    translation: Mat
) -> tuple[Mat, Mat, Mat, Mat, Mat, Rect, Rect] {
    if camera1.rows != 3 or camera1.cols != 3 or camera2.rows != 3 or camera2.cols != 3 {
        throw ValueError("both camera matrices are three by three")
    }
    if rotation.rows != 3 or rotation.cols != 3 {
        throw ValueError("the relative rotation is three by three")
    }
    let _unused = (dist1, dist2)
    let t = translation.values()
    if t.len() != 3 { throw ValueError("the relative translation has three components") }

    #: Half the rotation each way, so neither camera is turned further than it
    #: has to be. The two rectified cameras must end up in the SAME
    #: orientation, which with `X2 = R X1 + T` means `R1 = R2 R`: taking `half`
    #: as the square root of `R`, camera one turns by `half` and camera two by
    #: its inverse. Giving them the other way round leaves the two views turned
    #: relative to each other by the whole of `R`, and a correspondence then
    #: sits tens of rows from its partner rather than on the same one.
    let half = half_rotation(rotation)
    let half_inv = transpose(half)
    #: The baseline in the intermediate frame. Camera two's centre in camera
    #: one's frame is `-Rᵀ T`, and `half` of that is `-half⁻¹ T` because `R` is
    #: `half²`.
    let inv = half_inv.values()
    var turned: list[float] = []
    for row in 0..3 {
        turned.push(-(inv[row * 3] * t[0] + inv[row * 3 + 1] * t[1] + inv[row * 3 + 2] * t[2]))
    }

    #: The rotation that puts the baseline on an axis, and turns the pair no
    #: further than it has to.
    #:
    #: The axis is whichever one the baseline mostly lies along, oriented the
    #: way the baseline points -- so a camera to the right stays to the right
    #: rather than the pair being turned through half a turn. What takes the
    #: baseline there is then the MINIMAL rotation: the axis-angle whose axis
    #: is `t × u` and whose angle is between them. Building an orthonormal
    #: frame by Gram-Schmidt instead also rectifies -- the rows still line up
    #: exactly -- but it differs from this by a turn about the baseline, which
    #: is the one degree of freedom rectification does not fix, and OpenCV
    #: resolves it this way.
    let baseline = math.sqrt(turned[0] * turned[0] + turned[1] * turned[1] + turned[2] * turned[2])
    if baseline < 1e-12 { throw ValueError("the two cameras are in the same place") }
    var dominant = 0
    for index in 1..3 {
        if math.fabs(turned[index]) > math.fabs(turned[dominant]) { dominant = index }
    }
    var axis = [0.0, 0.0, 0.0]
    axis[dominant] = if turned[dominant] > 0.0 { 1.0 } else { -1.0 }

    var cross = [
        turned[1] * axis[2] - turned[2] * axis[1],
        turned[2] * axis[0] - turned[0] * axis[2],
        turned[0] * axis[1] - turned[1] * axis[0],
    ]
    let cross_length = math.sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2])
    var vector = [0.0, 0.0, 0.0]
    if cross_length > 1e-12 {
        var dot = 0.0
        for index in 0..3 { dot += turned[index] * axis[index] }
        let angle = math.acos(max(-1.0, min(1.0, math.fabs(dot) / baseline)))
        for index in 0..3 { vector[index] = cross[index] * angle / cross_length }
    }
    let align = rodrigues(Mat.from_list(3, 1, CV_MAKETYPE(CV_32F, 1), vector))

    let r1 = matmul(align, half)
    let r2 = matmul(align, half_inv)

    #: One focal length and one principal point for both, so the rows line up.
    let k1 = camera1.values()
    let k2 = camera2.values()
    let fx = (k1[0] + k2[0]) / 2.0
    let fy = (k1[4] + k2[4]) / 2.0
    let cx = (k1[2] + k2[2]) / 2.0
    let cy = (k1[5] + k2[5]) / 2.0

    let p1 = Mat.from_list(
        3,
        4,
        CV_MAKETYPE(CV_32F, 1),
        [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    )
    #: The second camera sits one baseline along the new x axis, which is the
    #: whole of what a rectified pair is.
    let p2 = Mat.from_list(
        3,
        4,
        CV_MAKETYPE(CV_32F, 1),
        [fx, 0.0, cx, -fx * baseline, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    )
    let q = Mat.from_list(
        4,
        4,
        CV_MAKETYPE(CV_32F, 1),
        [
            1.0, 0.0, 0.0, -cx,
            0.0, 1.0, 0.0, -cy,
            0.0, 0.0, 0.0, fx,
            #: `1 / baseline`, not minus: OpenCV's `Q` is written in terms of
            #: `Tx`, the translation stored in `P2`, which is MINUS the
            #: baseline. Carrying the sign of the baseline instead gives every
            #: point a negative depth, which reprojects the scene behind the
            #: camera.
            0.0, 0.0, 1.0 / baseline, 0.0,
        ]
    )
    let whole = Rect(0, 0, image_size.width, image_size.height)
    return (r1, r2, p1, p2, q, whole, whole)
}

#: The rotation that is half of `rotation`, by halving its Rodrigues vector.
fn half_rotation(rotation: Mat) -> Mat {
    let m = rotation.values()
    let trace = m[0] + m[4] + m[8]
    var angle = math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0)))
    if angle < 1e-9 {
        return Mat.from_list(
            3,
            3,
            CV_MAKETYPE(CV_32F, 1),
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        )
    }
    let scale = 1.0 / (2.0 * math.sin(angle))
    var axis = [(m[7] - m[5]) * scale, (m[2] - m[6]) * scale, (m[3] - m[1]) * scale]
    angle /= 2.0
    let c = math.cos(angle)
    let s = math.sin(angle)
    let one = 1.0 - c
    let x = axis[0]
    let y = axis[1]
    let z = axis[2]
    return Mat.from_list(
        3,
        3,
        CV_MAKETYPE(CV_32F, 1),
        [
            c + x * x * one,
            x * y * one - z * s,
            x * z * one + y * s,
            y * x * one + z * s,
            c + y * y * one,
            y * z * one - x * s,
            z * x * one - y * s,
            z * y * one + x * s,
            c + z * z * one,
        ]
    )
}
