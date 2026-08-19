#: Resampling: `resize`, the affine and perspective warps, `remap`, the image
#: pyramid, and border padding.
#:
#: Coordinates follow OpenCV exactly — a destination pixel centre maps to
#: `(x + 0.5) * scale - 0.5` in the source, so a two-times upsample lines up
#: with what OpenCV produces rather than being half a pixel out.
#:
#: ```
#: let small = resize(frame, Size(320, 240))
#: let turned = warp_affine(frame, get_rotation_matrix_2d(centre, 30.0, 1.0), frame.size())
#: ```
import std.math
from std.gpu import Kernel
from ..core.mat import Mat, f32_bits
from ..core.types import BORDER_CONSTANT, BORDER_DEFAULT, BORDER_REFLECT_101, BORDER_REPLICATE
from ..core.types import CV_32F, CV_64F, CV_MAKETYPE, CV_MAT_CN
from ..core.types import INTER_AREA, INTER_CUBIC, INTER_LANCZOS4, INTER_LINEAR, INTER_NEAREST
from ..core.types import Point2f, Rect, Scalar, Size, WARP_INVERSE_MAP
from .common import PRELUDE, layout_of, saturation_args

let SOURCE = PRELUDE + """
kernel void geo_resize(device const float* src [[buffer(0)]],
                       device float* dst [[buffer(1)]],
                       constant uint& count [[buffer(2)]],
                       constant uint& rows [[buffer(3)]],
                       constant uint& cols [[buffer(4)]],
                       constant uint& cn [[buffer(5)]],
                       constant uint& src_off [[buffer(6)]],
                       constant uint& src_step [[buffer(7)]],
                       constant uint& out_rows [[buffer(8)]],
                       constant uint& out_cols [[buffer(9)]],
                       constant uint& dst_off [[buffer(10)]],
                       constant uint& dst_step [[buffer(11)]],
                       constant uint& interpolation [[buffer(12)]],
                       constant uint& scale_y_bits [[buffer(13)]],
                       constant uint& scale_x_bits [[buffer(14)]],
                       constant uint& low_bits [[buffer(15)]],
                       constant uint& high_bits [[buffer(16)]],
                       constant uint& integral [[buffer(17)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint channel = id % cn;
    uint pixel = id / cn;
    uint y = pixel / out_cols;
    uint x = pixel - y * out_cols;
    float scale_y = as_type<float>(scale_y_bits);
    float scale_x = as_type<float>(scale_x_bits);
    float value;
    if (interpolation == 3u) {
        /* INTER_AREA averages the source rectangle a destination pixel covers,
         * which is the only resampler that does not alias when shrinking. */
        float y0 = float(y) * scale_y;
        float y1 = float(y + 1u) * scale_y;
        float x0 = float(x) * scale_x;
        float x1 = float(x + 1u) * scale_x;
        int iy0 = int(floor(y0));
        int iy1 = max(iy0 + 1, int(ceil(y1)));
        int ix0 = int(floor(x0));
        int ix1 = max(ix0 + 1, int(ceil(x1)));
        float total = 0.0f;
        float weight_total = 0.0f;
        for (int sy = iy0; sy < iy1; ++sy) {
            float wy = min(float(sy + 1), y1) - max(float(sy), y0);
            if (wy <= 0.0f) continue;
            for (int sx = ix0; sx < ix1; ++sx) {
                float wx = min(float(sx + 1), x1) - max(float(sx), x0);
                if (wx <= 0.0f) continue;
                total += wy * wx * fetch(src, sy, sx, channel, rows, cols, cn,
                                         src_off, src_step, kBorderReplicate, 0.0f);
                weight_total += wy * wx;
            }
        }
        value = weight_total == 0.0f ? 0.0f : total / weight_total;
    } else if (interpolation == 0u) {
        /* INTER_NEAREST takes floor(dst * scale) with no half-pixel shift,
         * which is the one resampler OpenCV does not centre. */
        int sy = int(floor(float(y) * scale_y));
        int sx = int(floor(float(x) * scale_x));
        value = fetch(src, sy, sx, channel, rows, cols, cn, src_off, src_step,
                      kBorderReplicate, 0.0f);
    } else {
        float sy = (float(y) + 0.5f) * scale_y - 0.5f;
        float sx = (float(x) + 0.5f) * scale_x - 0.5f;
        value = sample_at(src, sy, sx, channel, rows, cols, cn, src_off, src_step,
                          kBorderReplicate, 0.0f, interpolation);
    }
    put(dst, dst_off + y * dst_step + x * cn + channel, value,
        as_type<float>(low_bits), as_type<float>(high_bits), integral);
}

kernel void geo_warp(device const float* src [[buffer(0)]],
                     device float* dst [[buffer(1)]],
                     device const float* matrix [[buffer(2)]],
                     constant uint& count [[buffer(3)]],
                     constant uint& rows [[buffer(4)]],
                     constant uint& cols [[buffer(5)]],
                     constant uint& cn [[buffer(6)]],
                     constant uint& src_off [[buffer(7)]],
                     constant uint& src_step [[buffer(8)]],
                     constant uint& out_cols [[buffer(9)]],
                     constant uint& dst_off [[buffer(10)]],
                     constant uint& dst_step [[buffer(11)]],
                     constant uint& matrix_off [[buffer(12)]],
                     constant uint& perspective [[buffer(13)]],
                     constant uint& interpolation [[buffer(14)]],
                     constant uint& border [[buffer(15)]],
                     constant uint& constant_bits [[buffer(16)]],
                     constant uint& low_bits [[buffer(17)]],
                     constant uint& high_bits [[buffer(18)]],
                     constant uint& integral [[buffer(19)]],
                     uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint channel = id % cn;
    uint pixel = id / cn;
    uint y = pixel / out_cols;
    uint x = pixel - y * out_cols;
    float fx = float(x);
    float fy = float(y);
    float m0 = matrix[matrix_off + 0u];
    float m1 = matrix[matrix_off + 1u];
    float m2 = matrix[matrix_off + 2u];
    float m3 = matrix[matrix_off + 3u];
    float m4 = matrix[matrix_off + 4u];
    float m5 = matrix[matrix_off + 5u];
    float sx = m0 * fx + m1 * fy + m2;
    float sy = m3 * fx + m4 * fy + m5;
    if (perspective != 0u) {
        float m6 = matrix[matrix_off + 6u];
        float m7 = matrix[matrix_off + 7u];
        float m8 = matrix[matrix_off + 8u];
        float w = m6 * fx + m7 * fy + m8;
        if (w == 0.0f) w = 1e-12f;
        sx /= w;
        sy /= w;
    }
    float value = sample_at(src, sy, sx, channel, rows, cols, cn, src_off, src_step,
                            border, as_type<float>(constant_bits), interpolation);
    put(dst, dst_off + y * dst_step + x * cn + channel, value,
        as_type<float>(low_bits), as_type<float>(high_bits), integral);
}

kernel void geo_remap(device const float* src [[buffer(0)]],
                      device float* dst [[buffer(1)]],
                      device const float* map_x [[buffer(2)]],
                      device const float* map_y [[buffer(3)]],
                      constant uint& count [[buffer(4)]],
                      constant uint& rows [[buffer(5)]],
                      constant uint& cols [[buffer(6)]],
                      constant uint& cn [[buffer(7)]],
                      constant uint& src_off [[buffer(8)]],
                      constant uint& src_step [[buffer(9)]],
                      constant uint& out_cols [[buffer(10)]],
                      constant uint& dst_off [[buffer(11)]],
                      constant uint& dst_step [[buffer(12)]],
                      constant uint& mx_off [[buffer(13)]],
                      constant uint& mx_step [[buffer(14)]],
                      constant uint& my_off [[buffer(15)]],
                      constant uint& my_step [[buffer(16)]],
                      constant uint& interpolation [[buffer(17)]],
                      constant uint& border [[buffer(18)]],
                      constant uint& constant_bits [[buffer(19)]],
                      constant uint& low_bits [[buffer(20)]],
                      constant uint& high_bits [[buffer(21)]],
                      constant uint& integral [[buffer(22)]],
                      uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint channel = id % cn;
    uint pixel = id / cn;
    uint y = pixel / out_cols;
    uint x = pixel - y * out_cols;
    float sx = map_x[mx_off + y * mx_step + x];
    float sy = map_y[my_off + y * my_step + x];
    float value = sample_at(src, sy, sx, channel, rows, cols, cn, src_off, src_step,
                            border, as_type<float>(constant_bits), interpolation);
    put(dst, dst_off + y * dst_step + x * cn + channel, value,
        as_type<float>(low_bits), as_type<float>(high_bits), integral);
}

/* The 5x5 Gaussian OpenCV builds its pyramid from, applied then decimated. */
constant float kPyramidKernel[5] = {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f};

kernel void geo_pyr_down(device const float* src [[buffer(0)]],
                         device float* dst [[buffer(1)]],
                         constant uint& count [[buffer(2)]],
                         constant uint& rows [[buffer(3)]],
                         constant uint& cols [[buffer(4)]],
                         constant uint& cn [[buffer(5)]],
                         constant uint& src_off [[buffer(6)]],
                         constant uint& src_step [[buffer(7)]],
                         constant uint& out_cols [[buffer(8)]],
                         constant uint& dst_off [[buffer(9)]],
                         constant uint& dst_step [[buffer(10)]],
                         constant uint& border [[buffer(11)]],
                         constant uint& low_bits [[buffer(12)]],
                         constant uint& high_bits [[buffer(13)]],
                         constant uint& integral [[buffer(14)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint channel = id % cn;
    uint pixel = id / cn;
    uint y = pixel / out_cols;
    uint x = pixel - y * out_cols;
    int cy = int(y) * 2;
    int cx = int(x) * 2;
    float total = 0.0f;
    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            total += kPyramidKernel[dy + 2] * kPyramidKernel[dx + 2] *
                     fetch(src, cy + dy, cx + dx, channel, rows, cols, cn,
                           src_off, src_step, border, 0.0f);
        }
    }
    put(dst, dst_off + y * dst_step + x * cn + channel, total,
        as_type<float>(low_bits), as_type<float>(high_bits), integral);
}

kernel void geo_pyr_up(device const float* src [[buffer(0)]],
                       device float* dst [[buffer(1)]],
                       constant uint& count [[buffer(2)]],
                       constant uint& rows [[buffer(3)]],
                       constant uint& cols [[buffer(4)]],
                       constant uint& cn [[buffer(5)]],
                       constant uint& src_off [[buffer(6)]],
                       constant uint& src_step [[buffer(7)]],
                       constant uint& out_cols [[buffer(8)]],
                       constant uint& dst_off [[buffer(9)]],
                       constant uint& dst_step [[buffer(10)]],
                       constant uint& border [[buffer(11)]],
                       constant uint& low_bits [[buffer(12)]],
                       constant uint& high_bits [[buffer(13)]],
                       constant uint& integral [[buffer(14)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint channel = id % cn;
    uint pixel = id / cn;
    uint y = pixel / out_cols;
    uint x = pixel - y * out_cols;
    /* Insert zeros between samples, then convolve with the same kernel scaled
     * by four so the average brightness is preserved.
     *
     * The border folds in the UPSAMPLED grid, not the source: reflecting a
     * source index instead disagrees with OpenCV along the right and bottom
     * edges, where the two lengths round differently. */
    float total = 0.0f;
    int up_rows = int(rows) * 2;
    int up_cols = int(cols) * 2;
    for (int dy = -2; dy <= 2; ++dy) {
        int sy2 = int(y) + dy;
        if ((sy2 & 1) != 0) continue;
        int fy = border_index(sy2, up_rows, border);
        if (fy < 0) continue;
        for (int dx = -2; dx <= 2; ++dx) {
            int sx2 = int(x) + dx;
            if ((sx2 & 1) != 0) continue;
            int fx = border_index(sx2, up_cols, border);
            if (fx < 0) continue;
            total += kPyramidKernel[dy + 2] * kPyramidKernel[dx + 2] *
                     fetch(src, fy / 2, fx / 2, channel, rows, cols, cn,
                           src_off, src_step, border, 0.0f);
        }
    }
    put(dst, dst_off + y * dst_step + x * cn + channel, total * 4.0f,
        as_type<float>(low_bits), as_type<float>(high_bits), integral);
}

kernel void geo_border(device const float* src [[buffer(0)]],
                       device float* dst [[buffer(1)]],
                       constant uint& count [[buffer(2)]],
                       constant uint& rows [[buffer(3)]],
                       constant uint& cols [[buffer(4)]],
                       constant uint& cn [[buffer(5)]],
                       constant uint& src_off [[buffer(6)]],
                       constant uint& src_step [[buffer(7)]],
                       constant uint& out_cols [[buffer(8)]],
                       constant uint& dst_off [[buffer(9)]],
                       constant uint& dst_step [[buffer(10)]],
                       constant uint& top [[buffer(11)]],
                       constant uint& left [[buffer(12)]],
                       constant uint& border [[buffer(13)]],
                       constant uint& c0 [[buffer(14)]],
                       constant uint& c1 [[buffer(15)]],
                       constant uint& c2 [[buffer(16)]],
                       constant uint& c3 [[buffer(17)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint channel = id % cn;
    uint pixel = id / cn;
    uint y = pixel / out_cols;
    uint x = pixel - y * out_cols;
    float fill = as_type<float>(c0);
    if (channel == 1u) fill = as_type<float>(c1);
    else if (channel == 2u) fill = as_type<float>(c2);
    else if (channel == 3u) fill = as_type<float>(c3);
    dst[dst_off + y * dst_step + x * cn + channel] =
        fetch(src, int(y) - int(top), int(x) - int(left), channel, rows, cols, cn,
              src_off, src_step, border, fill);
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

fn interpolation_code(flags: int) -> int { return flags & 7 }

#: Resample `src` to `size`, or by the given scale factors.
pub fn resize(
    src: Mat,
    size: Size? = null,
    fx: float = 0.0,
    fy: float = 0.0,
    interpolation: int = INTER_LINEAR
) -> Mat {
    src.flush()
    var out_cols = 0
    var out_rows = 0
    let wanted = size
    if wanted is not null and not wanted.empty() {
        out_cols = wanted.width
        out_rows = wanted.height
    } else {
        if fx <= 0.0 or fy <= 0.0 {
            throw ValueError("resize needs a target size or positive scale factors")
        }
        out_cols = int(math.floor(float(src.cols) * fx + 0.5))
        out_rows = int(math.floor(float(src.rows) * fy + 0.5))
    }
    if out_cols <= 0 or out_rows <= 0 {
        throw ValueError(f"resize target {out_cols}x{out_rows} is empty")
    }
    let out = Mat(out_rows, out_cols, src.flags)
    let count = out_rows * out_cols * src.channels()
    if count == 0 { return out }
    let scale_y = float(src.rows) / float(out_rows)
    let scale_x = float(src.cols) / float(out_cols)
    var mode = interpolation_code(interpolation)
    #: INTER_AREA only averages when it is shrinking; OpenCV degenerates it to
    #: nearest-neighbour the moment either axis grows.
    if mode == INTER_AREA and (scale_x < 1.0 or scale_y < 1.0) { mode = INTER_NEAREST }
    var scalars = [
        count,
        src.rows,
        src.cols,
        src.channels(),
        src.offset,
        src.step_,
        out_rows,
        out_cols,
        out.offset,
        out.step_,
        mode,
        f32_bits(scale_y),
        f32_bits(scale_x),
    ]
    for value in saturation_args(out.flags) { scalars.push(value) }
    kernel_for("geo_resize").dispatch_async([src.storage, out.storage], count, scalars)
    out.invalidate()
    return out
}

fn warp(
    src: Mat,
    matrix: Mat,
    size: Size,
    perspective: bool,
    flags: int,
    border_mode: int,
    border_value: Scalar
) -> Mat {
    src.flush()
    let inverse = if (flags & WARP_INVERSE_MAP) != 0 {
        matrix
    } else {
        invert_transform(matrix, perspective)
    }
    inverse.flush()
    let out = Mat(size.height, size.width, src.flags)
    let count = size.height * size.width * src.channels()
    if count == 0 { return out }
    var scalars = [
        count,
        src.rows,
        src.cols,
        src.channels(),
        src.offset,
        src.step_,
        size.width,
        out.offset,
        out.step_,
        inverse.offset,
        if perspective { 1 } else { 0 },
        interpolation_code(flags),
        border_mode,
        f32_bits(border_value[0]),
    ]
    for value in saturation_args(out.flags) { scalars.push(value) }
    kernel_for("geo_warp").dispatch_async(
        [src.storage, out.storage, inverse.storage],
        count,
        scalars
    )
    out.invalidate()
    return out
}

#: Apply a 2x3 affine transform, sampling the source through its inverse.
pub fn warp_affine(
    src: Mat,
    matrix: Mat,
    size: Size,
    flags: int = INTER_LINEAR,
    border_mode: int = BORDER_CONSTANT,
    border_value: Scalar = Scalar(0.0, 0.0, 0.0, 0.0)
) -> Mat {
    if matrix.rows != 2 or matrix.cols != 3 {
        throw ValueError(f"warp_affine needs a 2x3 matrix, got {matrix}")
    }
    return warp(src, matrix, size, false, flags, border_mode, border_value)
}

#: Apply a 3x3 perspective transform.
pub fn warp_perspective(
    src: Mat,
    matrix: Mat,
    size: Size,
    flags: int = INTER_LINEAR,
    border_mode: int = BORDER_CONSTANT,
    border_value: Scalar = Scalar(0.0, 0.0, 0.0, 0.0)
) -> Mat {
    if matrix.rows != 3 or matrix.cols != 3 {
        throw ValueError(f"warp_perspective needs a 3x3 matrix, got {matrix}")
    }
    return warp(src, matrix, size, true, flags, border_mode, border_value)
}

#: Sample `src` at the positions two coordinate maps name.
pub fn remap(
    src: Mat,
    map_x: Mat,
    map_y: Mat,
    interpolation: int = INTER_LINEAR,
    border_mode: int = BORDER_CONSTANT,
    border_value: Scalar = Scalar(0.0, 0.0, 0.0, 0.0)
) -> Mat {
    src.flush()
    map_x.flush()
    map_y.flush()
    if map_x.rows != map_y.rows or map_x.cols != map_y.cols {
        throw ValueError("remap needs two maps of one shape")
    }
    let out = Mat(map_x.rows, map_x.cols, src.flags)
    let count = out.rows * out.cols * src.channels()
    if count == 0 { return out }
    var scalars = [
        count,
        src.rows,
        src.cols,
        src.channels(),
        src.offset,
        src.step_,
        out.cols,
        out.offset,
        out.step_,
        map_x.offset,
        map_x.step_,
        map_y.offset,
        map_y.step_,
        interpolation_code(interpolation),
        border_mode,
        f32_bits(border_value[0]),
    ]
    for value in saturation_args(out.flags) { scalars.push(value) }
    kernel_for("geo_remap").dispatch_async(
        [src.storage, out.storage, map_x.storage, map_y.storage],
        count,
        scalars
    )
    out.invalidate()
    return out
}

#: Blur and halve, the down step of an image pyramid.
pub fn pyr_down(src: Mat, size: Size? = null, border_mode: int = BORDER_REFLECT_101) -> Mat {
    src.flush()
    let wanted = size
    let out_rows = if wanted is null { (src.rows + 1) // 2 } else { wanted.height }
    let out_cols = if wanted is null { (src.cols + 1) // 2 } else { wanted.width }
    let out = Mat(out_rows, out_cols, src.flags)
    let count = out_rows * out_cols * src.channels()
    if count == 0 { return out }
    var scalars = [
        count,
        src.rows,
        src.cols,
        src.channels(),
        src.offset,
        src.step_,
        out_cols,
        out.offset,
        out.step_,
        border_mode,
    ]
    for value in saturation_args(out.flags) { scalars.push(value) }
    kernel_for("geo_pyr_down").dispatch_async([src.storage, out.storage], count, scalars)
    out.invalidate()
    return out
}

#: Double and blur, the up step of an image pyramid.
pub fn pyr_up(src: Mat, size: Size? = null, border_mode: int = BORDER_REFLECT_101) -> Mat {
    src.flush()
    let wanted = size
    let out_rows = if wanted is null { src.rows * 2 } else { wanted.height }
    let out_cols = if wanted is null { src.cols * 2 } else { wanted.width }
    let out = Mat(out_rows, out_cols, src.flags)
    let count = out_rows * out_cols * src.channels()
    if count == 0 { return out }
    var scalars = [
        count,
        src.rows,
        src.cols,
        src.channels(),
        src.offset,
        src.step_,
        out_cols,
        out.offset,
        out.step_,
        border_mode,
    ]
    for value in saturation_args(out.flags) { scalars.push(value) }
    kernel_for("geo_pyr_up").dispatch_async([src.storage, out.storage], count, scalars)
    out.invalidate()
    return out
}

#: Pad `src` on every side, filling the new pixels by the border rule.
pub fn copy_make_border(
    src: Mat,
    top: int,
    bottom: int,
    left: int,
    right: int,
    border_mode: int = BORDER_CONSTANT,
    value: Scalar = Scalar(0.0, 0.0, 0.0, 0.0)
) -> Mat {
    src.flush()
    if top < 0 or bottom < 0 or left < 0 or right < 0 {
        throw ValueError("border widths must not be negative")
    }
    let out = Mat(src.rows + top + bottom, src.cols + left + right, src.flags)
    let count = out.rows * out.cols * src.channels()
    if count == 0 { return out }
    kernel_for("geo_border").dispatch_async(
        [src.storage, out.storage],
        count,
        [
            count,
            src.rows,
            src.cols,
            src.channels(),
            src.offset,
            src.step_,
            out.cols,
            out.offset,
            out.step_,
            top,
            left,
            border_mode,
            f32_bits(value[0]),
            f32_bits(value[1]),
            f32_bits(value[2]),
            f32_bits(value[3]),
        ]
    )
    out.invalidate()
    return out
}

#: The 2x3 matrix that rotates about `center` by `angle` degrees and scales.
pub fn get_rotation_matrix_2d(center: Point2f, angle: float, scale: float) -> Mat {
    let radians = angle * math.PI / 180.0
    let alpha = math.cos(radians) * scale
    let beta = math.sin(radians) * scale
    return Mat.from_list(
        2,
        3,
        CV_MAKETYPE(CV_32F, 1),
        [
            alpha,
            beta,
            (1.0 - alpha) * center.x - beta * center.y,
            -beta,
            alpha,
            beta * center.x + (1.0 - alpha) * center.y,
        ]
    )
}

#: The 2x3 affine transform taking three source points to three destinations.
pub fn get_affine_transform(source: list[Point2f], target: list[Point2f]) -> Mat {
    if source.len() != 3 or target.len() != 3 {
        throw ValueError("an affine transform is defined by three point pairs")
    }
    var rows: list[list[float]] = []
    var right: list[float] = []
    for index in 0..3 {
        rows.push([source[index].x, source[index].y, 1.0, 0.0, 0.0, 0.0])
        right.push(target[index].x)
        rows.push([0.0, 0.0, 0.0, source[index].x, source[index].y, 1.0])
        right.push(target[index].y)
    }
    let solution = solve_dense(rows, right)
    return Mat.from_list(2, 3, CV_MAKETYPE(CV_32F, 1), solution)
}

#: The 3x3 perspective transform taking four source points to four
#: destinations.
pub fn get_perspective_transform(source: list[Point2f], target: list[Point2f]) -> Mat {
    if source.len() != 4 or target.len() != 4 {
        throw ValueError("a perspective transform is defined by four point pairs")
    }
    var rows: list[list[float]] = []
    var right: list[float] = []
    for index in 0..4 {
        let sx = source[index].x
        let sy = source[index].y
        let tx = target[index].x
        let ty = target[index].y
        rows.push([sx, sy, 1.0, 0.0, 0.0, 0.0, -sx * tx, -sy * tx])
        right.push(tx)
        rows.push([0.0, 0.0, 0.0, sx, sy, 1.0, -sx * ty, -sy * ty])
        right.push(ty)
    }
    var solution = solve_dense(rows, right)
    solution.push(1.0)
    return Mat.from_list(3, 3, CV_MAKETYPE(CV_32F, 1), solution)
}

#: Solve a small dense system by Gaussian elimination with partial pivoting.
#:
#: The systems here are at most eight unknowns, so the cost is nothing and the
#: pivoting is what keeps a near-degenerate point set from blowing up.
pub fn solve_dense(rows: list[list[float]], right: list[float]) -> list[float] {
    let n = rows.len()
    if n == 0 { return [] }
    let width = rows[0].len()
    var matrix: list[list[float]] = []
    for index in 0..n {
        var row = [value for value in rows[index]]
        row.push(right[index])
        matrix.push(row)
    }
    var column = 0
    var pivot_row = 0
    while column < width and pivot_row < n {
        var best = pivot_row
        var best_size = math.fabs(matrix[pivot_row][column])
        for candidate in (pivot_row + 1)..n {
            let size = math.fabs(matrix[candidate][column])
            if size > best_size {
                best = candidate
                best_size = size
            }
        }
        if best_size < 1e-12 {
            column += 1
            continue
        }
        let swap = matrix[pivot_row]
        matrix[pivot_row] = matrix[best]
        matrix[best] = swap
        let leading = matrix[pivot_row][column]
        for index in column..(width + 1) { matrix[pivot_row][index] /= leading }
        for other in 0..n {
            if other == pivot_row { continue }
            let factor = matrix[other][column]
            if factor == 0.0 { continue }
            for index in column..(width + 1) {
                matrix[other][index] -= factor * matrix[pivot_row][index]
            }
        }
        pivot_row += 1
        column += 1
    }
    var solution: list[float] = []
    for index in 0..width { solution.push(0.0) }
    var row = 0
    var lead = 0
    while row < n and lead < width {
        if math.fabs(matrix[row][lead]) < 1e-12 {
            lead += 1
            continue
        }
        solution[lead] = matrix[row][width]
        row += 1
        lead += 1
    }
    return solution
}

#: The inverse of a 2x3 affine or 3x3 perspective transform.
pub fn invert_transform(matrix: Mat, perspective: bool) -> Mat {
    let values = matrix.to_list()
    if not perspective {
        let determinant = values[0] * values[4] - values[1] * values[3]
        if math.fabs(determinant) < 1e-12 {
            throw ValueError("this affine transform cannot be inverted")
        }
        let inverse = 1.0 / determinant
        let a = values[4] * inverse
        let b = -values[1] * inverse
        let d = -values[3] * inverse
        let e = values[0] * inverse
        return Mat.from_list(
            2,
            3,
            CV_MAKETYPE(CV_32F, 1),
            [a, b, -a * values[2] - b * values[5], d, e, -d * values[2] - e * values[5]]
        )
    }
    var rows: list[list[float]] = []
    for r in 0..3 {
        var row: list[float] = []
        for c in 0..3 { row.push(values[r * 3 + c]) }
        rows.push(row)
    }
    let inverse = invert3(rows)
    var flat: list[float] = []
    for r in 0..3 {
        for c in 0..3 { flat.push(inverse[r][c]) }
    }
    return Mat.from_list(3, 3, CV_MAKETYPE(CV_32F, 1), flat)
}

#: Inverse of a 3x3 matrix by cofactors.
pub fn invert3(m: list[list[float]]) -> list[list[float]] {
    let a = m[0][0]
    let b = m[0][1]
    let c = m[0][2]
    let d = m[1][0]
    let e = m[1][1]
    let f = m[1][2]
    let g = m[2][0]
    let h = m[2][1]
    let i = m[2][2]
    let determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if math.fabs(determinant) < 1e-14 { throw ValueError("this matrix cannot be inverted") }
    let inverse = 1.0 / determinant
    return [
        [(e * i - f * h) * inverse, (c * h - b * i) * inverse, (b * f - c * e) * inverse],
        [(f * g - d * i) * inverse, (a * i - c * g) * inverse, (c * d - a * f) * inverse],
        [(d * h - e * g) * inverse, (b * g - a * h) * inverse, (a * e - b * d) * inverse],
    ]
}

#: Apply a 3x3 perspective transform to a list of points.
pub fn perspective_transform(points: list[Point2f], matrix: Mat) -> list[Point2f] {
    let m = matrix.to_list()
    var out: list[Point2f] = []
    for point in points {
        let w = m[6] * point.x + m[7] * point.y + m[8]
        let scale = if math.fabs(w) < 1e-12 { 0.0 } else { 1.0 / w }
        out.push(
            Point2f(
                (m[0] * point.x + m[1] * point.y + m[2]) * scale,
                (m[3] * point.x + m[4] * point.y + m[5]) * scale
            )
        )
    }
    return out
}

#: Cut `rect` out of `src` as an independent image.
pub fn crop(src: Mat, rect: Rect) -> Mat { return src.roi(rect).clone() }
