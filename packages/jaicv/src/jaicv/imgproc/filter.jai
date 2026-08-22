#: Convolution and the filters built on it: box and Gaussian blur, the median
#: and bilateral edge-preserving filters, and the Sobel/Scharr/Laplacian
#: derivatives.
#:
#: Kernels are `CV_32F` matrices, anchored at their centre unless told
#: otherwise, and correlate rather than convolve — the same choice OpenCV made,
#: so a kernel written for one works in the other.
#:
#: ```
#: let soft = gaussian_blur(frame, Size(5, 5), 1.4)
#: let dx = sobel(gray, CV_32F, 1, 0, ksize: 3)
#: ```
import std.math
from std.gpu import Kernel
from ..core.mat import Mat, f32_bits
from ..core.arith import UN_SQUARE, add, unary_op
from ..core.types import BORDER_CONSTANT, BORDER_DEFAULT, BORDER_REFLECT_101, BORDER_REPLICATE
from ..core.types import CV_32F, CV_MAKETYPE, CV_MAT_CN, CV_MAT_DEPTH
from ..core.types import Point, Scalar, Size
from .common import PRELUDE, saturation_args

let _SMALL_GAUSSIANS = [
    [1.0],
    [0.25, 0.5, 0.25],
    [0.0625, 0.25, 0.375, 0.25, 0.0625],
    [0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125],
]

#: Largest window `median_blur` will scan.
#:
#: The selection runs in a per-thread array, so the bound is real: an 11x11
#: window is 121 floats of thread-local storage per pixel and anything wider
#: spills far enough to be slower than doing it on the host.
pub const MAX_WINDOW = 11

let SOURCE = PRELUDE + """
kernel void filter_2d(device const float* src [[buffer(0)]],
                      device float* dst [[buffer(1)]],
                      device const float* weights [[buffer(2)]],
                      constant uint& count [[buffer(3)]],
                      constant uint& rows [[buffer(4)]],
                      constant uint& cols [[buffer(5)]],
                      constant uint& cn [[buffer(6)]],
                      constant uint& src_off [[buffer(7)]],
                      constant uint& src_step [[buffer(8)]],
                      constant uint& dst_off [[buffer(9)]],
                      constant uint& dst_step [[buffer(10)]],
                      constant uint& kernel_rows [[buffer(11)]],
                      constant uint& kernel_cols [[buffer(12)]],
                      constant uint& anchor_y [[buffer(13)]],
                      constant uint& anchor_x [[buffer(14)]],
                      constant uint& weights_off [[buffer(15)]],
                      constant uint& border [[buffer(16)]],
                      constant uint& delta_bits [[buffer(17)]],
                      constant uint& low_bits [[buffer(18)]],
                      constant uint& high_bits [[buffer(19)]],
                      constant uint& integral [[buffer(20)]],
                      uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint channel = id % cn;
    uint pixel = id / cn;
    uint y = pixel / cols;
    uint x = pixel - y * cols;
    float total = as_type<float>(delta_bits);
    for (uint ky = 0; ky < kernel_rows; ++ky) {
        int sy = int(y) + int(ky) - int(anchor_y);
        for (uint kx = 0; kx < kernel_cols; ++kx) {
            int sx = int(x) + int(kx) - int(anchor_x);
            float w = weights[weights_off + ky * kernel_cols + kx];
            if (w == 0.0f) continue;
            total += w * fetch(src, sy, sx, channel, rows, cols, cn,
                               src_off, src_step, border, 0.0f);
        }
    }
    put(dst, dst_off + y * dst_step + x * cn + channel, total,
        as_type<float>(low_bits), as_type<float>(high_bits), integral);
}

/* Separable filtering runs the row pass and the column pass in one kernel:
 * for the window sizes an image filter uses, one dispatch that reads a
 * neighbourhood twice beats two dispatches through device memory. */
/* A separable filter, run as the two passes that make it separable.
 *
 * Two things were tried against this pair and both lost, measured at 1080p in
 * milliseconds, kernel size against the split:
 *
 *   the whole square in one pass   3x3 0.29 / 0.29, 5x5 0.54 / 0.31,
 *                                  9x9 1.53 / 0.43, 11x11 2.23 / 0.50
 *   both passes fused, a group     3x3 0.36 / 0.29, 5x5 0.53 / 0.31,
 *   marching a band of rows with   9x9 0.89 / 0.43, 11x11 1.09 / 0.50
 *   the row-filtered lines in a
 *   threadgroup ring
 *
 * The square only ever draws level, at 3x3. The fused strip should have won --
 * it moves two passes over the picture instead of four -- and it loses because
 * of what it costs to march: two barriers per output row, a band that redoes
 * `col_len - 1` lines where it meets the next, and a hundred and thirty-odd
 * groups where the split pair has a thread per pixel. Passes over memory were
 * not the binding constraint; occupancy was.
 *
 * This used to sweep the whole `col_len` by `row_len` window per output pixel,
 * which is the work a general 2-D filter does and the work being separable is
 * meant to avoid: a 5 by 5 Gaussian fetched twenty-five samples a pixel where
 * ten will do, and a 15 by 15 fetched two hundred and twenty-five where thirty
 * will.
 *
 * Splitting is exact here rather than approximate. The border modes jaicv
 * offers -- replicate, reflect, wrap and a zero constant -- each extend the two
 * axes independently, so extending x and then y lands on the same sample the
 * two-dimensional fetch did. The intermediate stays float whatever the output
 * depth, so rounding happens once, at the end, exactly as before. */
kernel void filter_separable_row(device const float* src [[buffer(0)]],
                                 device float* tmp [[buffer(1)]],
                                 device const float* row_weights [[buffer(2)]],
                                 constant uint& count [[buffer(3)]],
                                 constant uint& rows [[buffer(4)]],
                                 constant uint& cols [[buffer(5)]],
                                 constant uint& cn [[buffer(6)]],
                                 constant uint& src_off [[buffer(7)]],
                                 constant uint& src_step [[buffer(8)]],
                                 constant uint& row_len [[buffer(9)]],
                                 constant uint& anchor_x [[buffer(10)]],
                                 constant uint& row_off [[buffer(11)]],
                                 constant uint& border [[buffer(12)]],
                                 uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint channel = id % cn;
    uint pixel = id / cn;
    uint y = pixel / cols;
    uint x = pixel - y * cols;
    float run = 0.0f;
    /* Away from the edges the window is entirely inside the picture, so the
     * addresses walk forward by `cn` and none of the border arithmetic is
     * reachable. That arithmetic is two bounds-resolutions a tap, which for a
     * three-tap blur is most of what the kernel does. */
    int left = int(x) - int(anchor_x);
    if (left >= 0 && left + int(row_len) <= int(cols)) {
        uint at = src_off + uint(y) * src_step + uint(left) * cn + channel;
        for (uint kx = 0; kx < row_len; ++kx) {
            run += row_weights[row_off + kx] * src[at];
            at += cn;
        }
    } else {
        for (uint kx = 0; kx < row_len; ++kx) {
            float wx = row_weights[row_off + kx];
            if (wx == 0.0f) continue;
            int sx = left + int(kx);
            run += wx * fetch(src, int(y), sx, channel, rows, cols, cn,
                              src_off, src_step, border, 0.0f);
        }
    }
    tmp[id] = run;
}

kernel void filter_separable_col(device const float* tmp [[buffer(0)]],
                                 device float* dst [[buffer(1)]],
                                 device const float* col_weights [[buffer(2)]],
                                 constant uint& count [[buffer(3)]],
                                 constant uint& rows [[buffer(4)]],
                                 constant uint& cols [[buffer(5)]],
                                 constant uint& cn [[buffer(6)]],
                                 constant uint& dst_off [[buffer(7)]],
                                 constant uint& dst_step [[buffer(8)]],
                                 constant uint& col_len [[buffer(9)]],
                                 constant uint& anchor_y [[buffer(10)]],
                                 constant uint& col_off [[buffer(11)]],
                                 constant uint& border [[buffer(12)]],
                                 constant uint& delta_bits [[buffer(13)]],
                                 constant uint& low_bits [[buffer(14)]],
                                 constant uint& high_bits [[buffer(15)]],
                                 constant uint& integral [[buffer(16)]],
                                 uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint channel = id % cn;
    uint pixel = id / cn;
    uint y = pixel / cols;
    uint x = pixel - y * cols;
    float total = as_type<float>(delta_bits);
    /* The intermediate is packed, so it is its own plane: no offset and a step
     * of `cols * cn`. Interior rows walk it by whole lines, as the row pass
     * walks its own by channels. */
    int top = int(y) - int(anchor_y);
    if (top >= 0 && top + int(col_len) <= int(rows)) {
        uint at = uint(top) * cols * cn + uint(x) * cn + channel;
        for (uint ky = 0; ky < col_len; ++ky) {
            total += col_weights[col_off + ky] * tmp[at];
            at += cols * cn;
        }
    } else {
        for (uint ky = 0; ky < col_len; ++ky) {
            float wy = col_weights[col_off + ky];
            if (wy == 0.0f) continue;
            total += wy * fetch(tmp, top + int(ky), int(x), channel, rows, cols, cn,
                                0u, cols * cn, border, 0.0f);
        }
    }
    put(dst, dst_off + y * dst_step + x * cn + channel, total,
        as_type<float>(low_bits), as_type<float>(high_bits), integral);
}

kernel void filter_median(device const float* src [[buffer(0)]],
                          device float* dst [[buffer(1)]],
                          constant uint& count [[buffer(2)]],
                          constant uint& rows [[buffer(3)]],
                          constant uint& cols [[buffer(4)]],
                          constant uint& cn [[buffer(5)]],
                          constant uint& src_off [[buffer(6)]],
                          constant uint& src_step [[buffer(7)]],
                          constant uint& dst_off [[buffer(8)]],
                          constant uint& dst_step [[buffer(9)]],
                          constant uint& ksize [[buffer(10)]],
                          constant uint& border [[buffer(11)]],
                          constant uint& low_bits [[buffer(12)]],
                          constant uint& high_bits [[buffer(13)]],
                          constant uint& integral [[buffer(14)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint channel = id % cn;
    uint pixel = id / cn;
    uint y = pixel / cols;
    uint x = pixel - y * cols;
    int radius = int(ksize) / 2;
    /* Selection over a fixed scratch array: the windows a median filter uses
     * are small enough that counting beats sorting. */
    float window[121];
    uint filled = 0;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            window[filled++] = fetch(src, int(y) + dy, int(x) + dx, channel,
                                     rows, cols, cn, src_off, src_step, border, 0.0f);
        }
    }
    uint middle = filled / 2u;
    for (uint i = 0; i <= middle; ++i) {
        uint smallest = i;
        for (uint j = i + 1u; j < filled; ++j) {
            if (window[j] < window[smallest]) smallest = j;
        }
        float swap = window[i];
        window[i] = window[smallest];
        window[smallest] = swap;
    }
    put(dst, dst_off + y * dst_step + x * cn + channel, window[middle],
        as_type<float>(low_bits), as_type<float>(high_bits), integral);
}

kernel void filter_bilateral(device const float* src [[buffer(0)]],
                             device float* dst [[buffer(1)]],
                             constant uint& count [[buffer(2)]],
                             constant uint& rows [[buffer(3)]],
                             constant uint& cols [[buffer(4)]],
                             constant uint& cn [[buffer(5)]],
                             constant uint& src_off [[buffer(6)]],
                             constant uint& src_step [[buffer(7)]],
                             constant uint& dst_off [[buffer(8)]],
                             constant uint& dst_step [[buffer(9)]],
                             constant uint& radius [[buffer(10)]],
                             constant uint& border [[buffer(11)]],
                             constant uint& colour_bits [[buffer(12)]],
                             constant uint& space_bits [[buffer(13)]],
                             constant uint& low_bits [[buffer(14)]],
                             constant uint& high_bits [[buffer(15)]],
                             constant uint& integral [[buffer(16)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint pixel = id / cn;
    uint channel = id % cn;
    uint y = pixel / cols;
    uint x = pixel - y * cols;
    float colour_coeff = as_type<float>(colour_bits);
    float space_coeff = as_type<float>(space_bits);
    int r = int(radius);
    /* The colour distance is measured over every channel at once, which is
     * what keeps a bilateral filter from bleeding across an edge that shows in
     * only one of them. */
    float centre[4];
    for (uint c = 0; c < cn && c < 4u; ++c) {
        centre[c] = fetch(src, int(y), int(x), c, rows, cols, cn,
                          src_off, src_step, border, 0.0f);
    }
    float total = 0.0f;
    float weight_total = 0.0f;
    for (int dy = -r; dy <= r; ++dy) {
        for (int dx = -r; dx <= r; ++dx) {
            float spatial = float(dy * dy + dx * dx);
            if (spatial > float(r * r)) continue;
            float difference = 0.0f;
            for (uint c = 0; c < cn && c < 4u; ++c) {
                float other = fetch(src, int(y) + dy, int(x) + dx, c, rows, cols, cn,
                                    src_off, src_step, border, 0.0f);
                float gap = other - centre[c];
                difference += gap * gap;
            }
            float w = exp(spatial * space_coeff + difference * colour_coeff);
            total += w * fetch(src, int(y) + dy, int(x) + dx, channel, rows, cols, cn,
                               src_off, src_step, border, 0.0f);
            weight_total += w;
        }
    }
    float value = weight_total <= 0.0f ? centre[min(channel, 3u)] : total / weight_total;
    put(dst, dst_off + y * dst_step + x * cn + channel, value,
        as_type<float>(low_bits), as_type<float>(high_bits), integral);
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

fn destination_type(src: Mat, ddepth: int) -> int {
    if ddepth < 0 { return src.flags }
    return CV_MAKETYPE(ddepth, src.channels())
}

#: Correlate `src` with an arbitrary kernel.
#:
#: `anchor` defaults to the kernel's centre. `delta` is added to every result
#: before it is written.
pub fn filter_2d(
    src: Mat,
    ddepth: int,
    weights: Mat,
    anchor: Point? = null,
    delta: float = 0.0,
    border_mode: int = BORDER_DEFAULT
) -> Mat {
    src.flush()
    weights.flush()
    if weights.channels() != 1 { throw ValueError("a filter kernel has one channel") }
    let point = anchor
    let anchor_y = if point is null { weights.rows // 2 } else { point.y }
    let anchor_x = if point is null { weights.cols // 2 } else { point.x }
    let out = Mat(src.rows, src.cols, destination_type(src, ddepth))
    let count = src.rows * src.cols * src.channels()
    if count == 0 { return out }
    var scalars = [
        count,
        src.rows,
        src.cols,
        src.channels(),
        src.offset,
        src.step_,
        out.offset,
        out.step_,
        weights.rows,
        weights.cols,
        anchor_y,
        anchor_x,
        weights.offset,
        border_mode,
        f32_bits(delta),
    ]
    for value in saturation_args(out.flags) { scalars.push(value) }
    kernel_for("filter_2d").dispatch_async(
        [src.storage, out.storage, weights.storage],
        count,
        scalars
    )
    out.invalidate()
    return out
}

#: Correlate with a kernel that is the outer product of two 1-D kernels.
pub fn sep_filter_2d(
    src: Mat,
    ddepth: int,
    kernel_x: Mat,
    kernel_y: Mat,
    anchor: Point? = null,
    delta: float = 0.0,
    border_mode: int = BORDER_DEFAULT
) -> Mat {
    src.flush()
    kernel_x.flush()
    kernel_y.flush()
    let row_len = kernel_x.rows * kernel_x.cols
    let col_len = kernel_y.rows * kernel_y.cols
    let point = anchor
    let anchor_x = if point is null { row_len // 2 } else { point.x }
    let anchor_y = if point is null { col_len // 2 } else { point.y }
    let out = Mat(src.rows, src.cols, destination_type(src, ddepth))
    let count = src.rows * src.cols * src.channels()
    if count == 0 { return out }
    #: Packed and always float, whatever the output depth: the two passes have
    #: to hand each other unrounded values, and rounding once at the end is what
    #: the single-pass version did.
    let middle = Mat(src.rows, src.cols, CV_MAKETYPE(CV_32F, src.channels()))
    kernel_for("filter_separable_row").dispatch_async(
        [src.storage, middle.storage, kernel_x.storage],
        count,
        [
            count,
            src.rows,
            src.cols,
            src.channels(),
            src.offset,
            src.step_,
            row_len,
            anchor_x,
            kernel_x.offset,
            border_mode,
        ]
    )
    var scalars = [
        count,
        src.rows,
        src.cols,
        src.channels(),
        out.offset,
        out.step_,
        col_len,
        anchor_y,
        kernel_y.offset,
        border_mode,
        f32_bits(delta),
    ]
    for value in saturation_args(out.flags) { scalars.push(value) }
    kernel_for("filter_separable_col").dispatch_async(
        [middle.storage, out.storage, kernel_y.storage],
        count,
        scalars
    )
    middle.release()
    out.invalidate()
    return out
}

#: A 1-D Gaussian of the given length. A `sigma` of zero derives one from the
#: size, exactly as `cv::getGaussianKernel` does.
pub fn get_gaussian_kernel(ksize: int, sigma: float) -> Mat {
    if ksize <= 0 or ksize % 2 == 0 {
        throw ValueError(f"a Gaussian kernel needs an odd positive size, got {ksize}")
    }
    #: With no sigma given, OpenCV answers the small sizes from a table of
    #: binomial rows rather than sampling the exponential, and a filter that
    #: samples instead is visibly softer. Larger sizes fall through.
    if sigma <= 0.0 and ksize <= 7 {
        let table = _SMALL_GAUSSIANS[(ksize - 1) // 2]
        return Mat.from_list(ksize, 1, CV_MAKETYPE(CV_32F, 1), [value for value in table])
    }
    let width = if sigma > 0.0 { sigma } else { 0.3 * (float(ksize - 1) * 0.5 - 1.0) + 0.8 }
    let scale = -0.5 / (width * width)
    var values: list[float] = []
    var total = 0.0
    let centre = float(ksize - 1) * 0.5
    for index in 0..ksize {
        let offset = float(index) - centre
        let value = math.exp(scale * offset * offset)
        values.push(value)
        total += value
    }
    for index in 0..ksize { values[index] /= total }
    return Mat.from_list(ksize, 1, CV_MAKETYPE(CV_32F, 1), values)
}

#: Blur with a Gaussian window.
pub fn gaussian_blur(
    src: Mat,
    ksize: Size,
    sigma_x: float,
    sigma_y: float = 0.0,
    border_mode: int = BORDER_DEFAULT
) -> Mat {
    let width = if ksize.width > 0 { ksize.width } else { _size_for(sigma_x) }
    let height = if ksize.height > 0 { ksize.height } else { _size_for(sigma_y) }
    let sy = if sigma_y > 0.0 { sigma_y } else { sigma_x }
    let row = get_gaussian_kernel(width, sigma_x)
    let column = get_gaussian_kernel(height, sy)
    return sep_filter_2d(src, -1, row, column, border_mode: border_mode)
}

fn _size_for(sigma: float) -> int {
    if sigma <= 0.0 { throw ValueError("gaussian_blur needs a size or a positive sigma") }
    let radius = int(math.floor(sigma * 3.0 + 0.5))
    return radius * 2 + 1
}

#: Average over a rectangular window.
pub fn box_filter(
    src: Mat,
    ddepth: int,
    ksize: Size,
    anchor: Point? = null,
    normalize: bool = true,
    border_mode: int = BORDER_DEFAULT
) -> Mat {
    let weight = if normalize { 1.0 / float(ksize.width * ksize.height) } else { 1.0 }
    var row: list[float] = []
    for _index in 0..ksize.width { row.push(weight) }
    var column: list[float] = []
    for _index in 0..ksize.height { column.push(1.0) }
    return sep_filter_2d(
        src,
        ddepth,
        Mat.from_list(ksize.width, 1, CV_MAKETYPE(CV_32F, 1), row),
        Mat.from_list(ksize.height, 1, CV_MAKETYPE(CV_32F, 1), column),
        anchor: anchor,
        border_mode: border_mode
    )
}

#: Average over a rectangular window, normalised.
pub fn blur(
    src: Mat,
    ksize: Size,
    anchor: Point? = null,
    border_mode: int = BORDER_DEFAULT
) -> Mat {
    return box_filter(src, -1, ksize, anchor: anchor, normalize: true, border_mode: border_mode)
}

#: Box filter of the squared values, which is what a local-variance estimate
#: needs alongside `box_filter` of the values themselves.
pub fn sqr_box_filter(
    src: Mat,
    ddepth: int,
    ksize: Size,
    normalize: bool = true,
    border_mode: int = BORDER_DEFAULT
) -> Mat {
    let widened = src.convert_to(CV_MAKETYPE(CV_32F, src.channels()))
    let squared = unary_op(widened, UN_SQUARE)
    return box_filter(squared, ddepth, ksize, normalize: normalize, border_mode: border_mode)
}

#: Replace each pixel with the median of its window, which removes speckle
#: without softening an edge.
pub fn median_blur(src: Mat, ksize: int, border_mode: int = BORDER_REPLICATE) -> Mat {
    src.flush()
    if ksize <= 0 or ksize % 2 == 0 {
        throw ValueError(f"median_blur needs an odd positive size, got {ksize}")
    }
    if ksize > MAX_WINDOW {
        throw ValueError(f"median_blur supports windows up to {MAX_WINDOW}, got {ksize}")
    }
    let out = Mat(src.rows, src.cols, src.flags)
    let count = src.rows * src.cols * src.channels()
    if count == 0 { return out }
    var scalars = [
        count,
        src.rows,
        src.cols,
        src.channels(),
        src.offset,
        src.step_,
        out.offset,
        out.step_,
        ksize,
        border_mode,
    ]
    for value in saturation_args(out.flags) { scalars.push(value) }
    kernel_for("filter_median").dispatch_async([src.storage, out.storage], count, scalars)
    out.invalidate()
    return out
}

#: Smooth while keeping edges, by weighting neighbours on both distance and
#: colour difference.
pub fn bilateral_filter(
    src: Mat,
    d: int,
    sigma_color: float,
    sigma_space: float,
    border_mode: int = BORDER_DEFAULT
) -> Mat {
    src.flush()
    let radius = if d > 0 { d // 2 } else { int(math.floor(sigma_space * 1.5 + 0.5)) }
    if radius <= 0 { throw ValueError("bilateral_filter needs a positive window") }
    let colour = if sigma_color <= 0.0 { 1.0 } else { sigma_color }
    let space = if sigma_space <= 0.0 { 1.0 } else { sigma_space }
    let out = Mat(src.rows, src.cols, src.flags)
    let count = src.rows * src.cols * src.channels()
    if count == 0 { return out }
    var scalars = [
        count,
        src.rows,
        src.cols,
        src.channels(),
        src.offset,
        src.step_,
        out.offset,
        out.step_,
        radius,
        border_mode,
        f32_bits(-0.5 / (colour * colour)),
        f32_bits(-0.5 / (space * space)),
    ]
    for value in saturation_args(out.flags) { scalars.push(value) }
    kernel_for("filter_bilateral").dispatch_async([src.storage, out.storage], count, scalars)
    out.invalidate()
    return out
}

#: The 1-D derivative and smoothing kernels a Sobel of a given size uses.
pub fn get_derivative_kernels(dx: int, dy: int, ksize: int) -> tuple[Mat, Mat] {
    return (_derivative_row(dx, ksize), _derivative_row(dy, ksize))
}

fn _derivative_row(order: int, ksize: int) -> Mat {
    if ksize == -1 {
        #: Scharr, which is the 3-tap kernel with the best rotational symmetry.
        let values = if order == 0 { [3.0, 10.0, 3.0] } else { [-1.0, 0.0, 1.0] }
        return Mat.from_list(3, 1, CV_MAKETYPE(CV_32F, 1), values)
    }
    if ksize % 2 == 0 or ksize < 1 {
        throw ValueError(f"a derivative kernel needs an odd positive size, got {ksize}")
    }
    if ksize == 1 {
        if order == 0 { return Mat.from_list(1, 1, CV_MAKETYPE(CV_32F, 1), [1.0]) }
        return Mat.from_list(3, 1, CV_MAKETYPE(CV_32F, 1), [-1.0, 0.0, 1.0])
    }
    #: A Sobel row is a binomial smoothing kernel differentiated `order` times,
    #: which is how OpenCV builds every size from Pascal's triangle.
    var smoothing = _binomial(ksize - 1 - order)
    var row = smoothing
    for _step in 0..order { row = _differentiate(row) }
    return Mat.from_list(row.len(), 1, CV_MAKETYPE(CV_32F, 1), row)
}

fn _binomial(order: int) -> list[float] {
    var row = [1.0]
    for _step in 0..order {
        var next: list[float] = [1.0]
        for index in 0..row.len() - 1 { next.push(row[index] + row[index + 1]) }
        next.push(1.0)
        row = next
    }
    return row
}

fn _differentiate(row: list[float]) -> list[float] {
    var next: list[float] = [-row[0]]
    for index in 0..row.len() - 1 { next.push(row[index] - row[index + 1]) }
    next.push(row[row.len() - 1])
    return next
}

#: Derivative of an image, smoothed across the other axis.
pub fn sobel(
    src: Mat,
    ddepth: int,
    dx: int,
    dy: int,
    ksize: int = 3,
    scale: float = 1.0,
    delta: float = 0.0,
    border_mode: int = BORDER_DEFAULT
) -> Mat {
    if dx < 0 or dy < 0 or dx == 0 and dy == 0 {
        throw ValueError("sobel needs a non-zero derivative order")
    }
    let (row, column) = get_derivative_kernels(dx, dy, ksize)
    let scaled = if scale == 1.0 { row } else { _scaled(row, scale) }
    return sep_filter_2d(src, ddepth, scaled, column, delta: delta, border_mode: border_mode)
}

#: The 3-tap derivative with the best rotational symmetry.
pub fn scharr(
    src: Mat,
    ddepth: int,
    dx: int,
    dy: int,
    scale: float = 1.0,
    delta: float = 0.0,
    border_mode: int = BORDER_DEFAULT
) -> Mat {
    return sobel(
        src,
        ddepth,
        dx,
        dy,
        ksize: -1,
        scale: scale,
        delta: delta,
        border_mode: border_mode
    )
}

fn _scaled(row: Mat, factor: float) -> Mat {
    var values: list[float] = []
    for value in row.to_list() { values.push(value * factor) }
    return Mat.from_list(row.rows, row.cols, row.flags, values)
}

#: Sum of the second derivatives, which responds to a blob rather than an edge.
pub fn laplacian(
    src: Mat,
    ddepth: int,
    ksize: int = 1,
    scale: float = 1.0,
    delta: float = 0.0,
    border_mode: int = BORDER_DEFAULT
) -> Mat {
    if ksize == 1 {
        let weights = Mat.from_list(
            3,
            3,
            CV_MAKETYPE(CV_32F, 1),
            [0.0, scale, 0.0, scale, -4.0 * scale, scale, 0.0, scale, 0.0]
        )
        return filter_2d(src, ddepth, weights, delta: delta, border_mode: border_mode)
    }
    let xx = sobel(src, ddepth, 2, 0, ksize: ksize, scale: scale, border_mode: border_mode)
    let yy = sobel(src, ddepth, 0, 2, ksize: ksize, scale: scale, border_mode: border_mode)
    return add(xx, yy)
}
