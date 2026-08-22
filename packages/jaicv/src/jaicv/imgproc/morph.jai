#: Morphology: erosion, dilation, and the compound operations built from them.
#:
#: A structuring element is a `CV_8U` matrix; a non-zero entry means the pixel
#: under it takes part. `get_structuring_element` builds the three shapes
#: OpenCV names.
#:
#: ```
#: let element = get_structuring_element(MORPH_ELLIPSE, Size(5, 5))
#: let cleaned = morphology_ex(mask, MORPH_OPEN, element)
#: ```
import std.math
from std.gpu import Kernel
from ..core.mat import Mat, f32_bits
from ..core.arith import OP_SUBTRACT, binary_op
from ..core.types import BORDER_CONSTANT, BORDER_REPLICATE, CV_8U, CV_32F, CV_MAKETYPE
from ..core.types import Point, Scalar, Size
from .common import PRELUDE, saturation_args

#: A filled rectangle.
pub const MORPH_RECT = 0
#: A plus sign.
pub const MORPH_CROSS = 1
#: A filled ellipse inscribed in the rectangle.
pub const MORPH_ELLIPSE = 2

#: Shrink bright regions.
pub const MORPH_ERODE = 0
#: Grow bright regions.
pub const MORPH_DILATE = 1
#: Erode then dilate, which removes speckle without shrinking what is left.
pub const MORPH_OPEN = 2
#: Dilate then erode, which fills pinholes.
pub const MORPH_CLOSE = 3
#: Dilation minus erosion, an outline of every region.
pub const MORPH_GRADIENT = 4
#: The image minus its opening, which keeps what the opening removed.
pub const MORPH_TOPHAT = 5
#: The closing minus the image.
pub const MORPH_BLACKHAT = 6
#: Erode, then dilate twice, then erode: the morphological hit-or-miss.
pub const MORPH_HITMISS = 7

let SOURCE = PRELUDE + """
/* Erosion and dilation through a solid rectangle, which is separable.
 *
 * A minimum over a box is the minimum of the row minima, and a maximum likewise,
 * so a solid element costs two passes of K instead of one of K squared. Only a
 * solid one: an ellipse or a cross is not a product of its axes, and those keep
 * to `morph_rank`.
 *
 * The border constant reaches both passes, which is what makes the split exact
 * under BORDER_CONSTANT -- every sample the square would have taken from
 * outside is that value, and taking it in either pass gives the same extreme.
 */
kernel void morph_box_row(device const float* src [[buffer(0)]],
                          device float* tmp [[buffer(1)]],
                          constant uint& count [[buffer(2)]],
                          constant uint& rows [[buffer(3)]],
                          constant uint& cols [[buffer(4)]],
                          constant uint& cn [[buffer(5)]],
                          constant uint& src_off [[buffer(6)]],
                          constant uint& src_step [[buffer(7)]],
                          constant uint& element_cols [[buffer(8)]],
                          constant uint& anchor_x [[buffer(9)]],
                          constant uint& dilate [[buffer(10)]],
                          constant uint& border [[buffer(11)]],
                          constant uint& constant_bits [[buffer(12)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint channel = id % cn;
    uint pixel = id / cn;
    uint y = pixel / cols;
    uint x = pixel - y * cols;
    float best = dilate != 0u ? -INFINITY : INFINITY;
    for (uint kx = 0; kx < element_cols; ++kx) {
        int sx = int(x) + int(kx) - int(anchor_x);
        float value = fetch(src, int(y), sx, channel, rows, cols, cn, src_off, src_step,
                            border, as_type<float>(constant_bits));
        best = dilate != 0u ? max(best, value) : min(best, value);
    }
    tmp[id] = best;
}

kernel void morph_box_col(device const float* tmp [[buffer(0)]],
                          device float* dst [[buffer(1)]],
                          constant uint& count [[buffer(2)]],
                          constant uint& rows [[buffer(3)]],
                          constant uint& cols [[buffer(4)]],
                          constant uint& cn [[buffer(5)]],
                          constant uint& dst_off [[buffer(6)]],
                          constant uint& dst_step [[buffer(7)]],
                          constant uint& element_rows [[buffer(8)]],
                          constant uint& anchor_y [[buffer(9)]],
                          constant uint& dilate [[buffer(10)]],
                          constant uint& border [[buffer(11)]],
                          constant uint& constant_bits [[buffer(12)]],
                          constant uint& low_bits [[buffer(13)]],
                          constant uint& high_bits [[buffer(14)]],
                          constant uint& integral [[buffer(15)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint channel = id % cn;
    uint pixel = id / cn;
    uint y = pixel / cols;
    uint x = pixel - y * cols;
    float best = dilate != 0u ? -INFINITY : INFINITY;
    for (uint ky = 0; ky < element_rows; ++ky) {
        int sy = int(y) + int(ky) - int(anchor_y);
        float value = fetch(tmp, sy, int(x), channel, rows, cols, cn, 0u, cols * cn,
                            border, as_type<float>(constant_bits));
        best = dilate != 0u ? max(best, value) : min(best, value);
    }
    put(dst, dst_off + y * dst_step + x * cn + channel, best,
        as_type<float>(low_bits), as_type<float>(high_bits), integral);
}

kernel void morph_rank(device const float* src [[buffer(0)]],
                       device float* dst [[buffer(1)]],
                       device const float* element [[buffer(2)]],
                       constant uint& count [[buffer(3)]],
                       constant uint& rows [[buffer(4)]],
                       constant uint& cols [[buffer(5)]],
                       constant uint& cn [[buffer(6)]],
                       constant uint& src_off [[buffer(7)]],
                       constant uint& src_step [[buffer(8)]],
                       constant uint& dst_off [[buffer(9)]],
                       constant uint& dst_step [[buffer(10)]],
                       constant uint& element_rows [[buffer(11)]],
                       constant uint& element_cols [[buffer(12)]],
                       constant uint& anchor_y [[buffer(13)]],
                       constant uint& anchor_x [[buffer(14)]],
                       constant uint& element_off [[buffer(15)]],
                       constant uint& element_step [[buffer(16)]],
                       constant uint& dilate [[buffer(17)]],
                       constant uint& border [[buffer(18)]],
                       constant uint& constant_bits [[buffer(19)]],
                       constant uint& low_bits [[buffer(20)]],
                       constant uint& high_bits [[buffer(21)]],
                       constant uint& integral [[buffer(22)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint channel = id % cn;
    uint pixel = id / cn;
    uint y = pixel / cols;
    uint x = pixel - y * cols;
    /* Erosion takes the smallest value under the element and dilation the
     * largest, so the two differ only in the starting extreme and the compare. */
    float best = dilate != 0u ? -INFINITY : INFINITY;
    bool seen = false;
    for (uint ky = 0; ky < element_rows; ++ky) {
        for (uint kx = 0; kx < element_cols; ++kx) {
            if (element[element_off + ky * element_step + kx] == 0.0f) continue;
            int sy = int(y) + int(ky) - int(anchor_y);
            int sx = int(x) + int(kx) - int(anchor_x);
            float value = fetch(src, sy, sx, channel, rows, cols, cn, src_off, src_step,
                                border, as_type<float>(constant_bits));
            best = dilate != 0u ? max(best, value) : min(best, value);
            seen = true;
        }
    }
    if (!seen) best = fetch(src, int(y), int(x), channel, rows, cols, cn,
                            src_off, src_step, border, 0.0f);
    put(dst, dst_off + y * dst_step + x * cn + channel, best,
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

#: Build one of the named structuring-element shapes.
pub fn get_structuring_element(shape: int, ksize: Size, anchor: Point? = null) -> Mat {
    if ksize.width <= 0 or ksize.height <= 0 {
        throw ValueError(f"a structuring element needs a positive size, got {ksize}")
    }
    let point = anchor
    let anchor_x = if point is null { ksize.width // 2 } else { point.x }
    let anchor_y = if point is null { ksize.height // 2 } else { point.y }
    var values: list[float] = []
    #: The ellipse is drawn a row at a time, taking the half-width from the
    #: circle equation and rounding it to whole pixels. Testing the pixel
    #: centre against the ellipse instead fills the corners of a 3x3, where
    #: OpenCV draws a plus.
    let radius_y = ksize.height // 2
    let radius_x = ksize.width // 2
    let inv_r2 = if radius_y > 0 { 1.0 / float(radius_y * radius_y) } else { 0.0 }
    for y in 0..ksize.height {
        var left = 0
        var right = 0
        if shape == MORPH_ELLIPSE {
            let dy = y - radius_y
            if abs(dy) <= radius_y {
                let span = float(radius_x) *
                    math.sqrt(float(radius_y * radius_y - dy * dy) * inv_r2)
                let dx = int(math.floor(span + 0.5))
                left = max(radius_x - dx, 0)
                right = min(radius_x + dx + 1, ksize.width)
            }
        }
        for x in 0..ksize.width {
            var inside = false
            if shape == MORPH_RECT {
                inside = true
            } elif shape == MORPH_CROSS {
                inside = x == anchor_x or y == anchor_y
            } else {
                inside = x >= left and x < right
            }
            values.push(if inside { 1.0 } else { 0.0 })
        }
    }
    return Mat.from_list(ksize.height, ksize.width, CV_MAKETYPE(CV_8U, 1), values)
}

fn default_element(element: Mat?) -> Mat {
    let given = element
    if given is not null { return given }
    return get_structuring_element(MORPH_RECT, Size(3, 3))
}

#: Whether every place in the element is set, which is what lets the two axes
#: be taken one at a time.
#:
#: Read on the host, which costs nothing for the elements anyone actually
#: passes: `get_structuring_element` builds them from a list, so they are host
#: side already, and they are a few hundred values at most.
fn element_is_solid(element: Mat) -> bool {
    if element.channels() != 1 { return false }
    if element.offset != 0 or element.step_ != element.cols { return false }
    if element.rows * element.cols > 4096 { return false }
    for value in element.to_list() {
        if value == 0.0 { return false }
    }
    return true
}

fn rank_filter(
    src: Mat,
    element: Mat,
    dilate: bool,
    anchor: Point?,
    iterations: int,
    border_mode: int,
    border_value: Scalar
) -> Mat {
    src.flush()
    element.flush()
    let point = anchor
    let anchor_y = if point is null { element.rows // 2 } else { point.y }
    let anchor_x = if point is null { element.cols // 2 } else { point.x }
    let solid = element_is_solid(element)
    var current = src
    for _pass in 0..max(1, iterations) {
        let out = Mat(current.rows, current.cols, current.flags)
        let count = current.rows * current.cols * current.channels()
        if count == 0 { return out }
        let facing = if dilate { 1 } else { 0 }
        if solid {
            let middle = Mat(current.rows, current.cols, CV_MAKETYPE(CV_32F, current.channels()))
            kernel_for("morph_box_row").dispatch_async(
                [current.storage, middle.storage],
                count,
                [
                    count,
                    current.rows,
                    current.cols,
                    current.channels(),
                    current.offset,
                    current.step_,
                    element.cols,
                    anchor_x,
                    facing,
                    border_mode,
                    f32_bits(border_value[0]),
                ]
            )
            var tail = [
                count,
                current.rows,
                current.cols,
                current.channels(),
                out.offset,
                out.step_,
                element.rows,
                anchor_y,
                facing,
                border_mode,
                f32_bits(border_value[0]),
            ]
            for value in saturation_args(out.flags) { tail.push(value) }
            kernel_for("morph_box_col").dispatch_async([middle.storage, out.storage], count, tail)
            middle.release()
            out.invalidate()
            current = out
            continue
        }
        var scalars = [
            count,
            current.rows,
            current.cols,
            current.channels(),
            current.offset,
            current.step_,
            out.offset,
            out.step_,
            element.rows,
            element.cols,
            anchor_y,
            anchor_x,
            element.offset,
            element.step_,
            facing,
            border_mode,
            f32_bits(border_value[0]),
        ]
        for value in saturation_args(out.flags) { scalars.push(value) }
        kernel_for("morph_rank").dispatch_async(
            [current.storage, out.storage, element.storage],
            count,
            scalars
        )
        out.invalidate()
        current = out
    }
    return current
}

#: Shrink bright regions by the structuring element.
pub fn erode(
    src: Mat,
    element: Mat? = null,
    anchor: Point? = null,
    iterations: int = 1,
    border_mode: int = BORDER_REPLICATE,
    border_value: Scalar = Scalar(0.0, 0.0, 0.0, 0.0)
) -> Mat {
    return rank_filter(
        src,
        default_element(element),
        false,
        anchor,
        iterations,
        border_mode,
        border_value
    )
}

#: Grow bright regions by the structuring element.
pub fn dilate(
    src: Mat,
    element: Mat? = null,
    anchor: Point? = null,
    iterations: int = 1,
    border_mode: int = BORDER_REPLICATE,
    border_value: Scalar = Scalar(0.0, 0.0, 0.0, 0.0)
) -> Mat {
    return rank_filter(
        src,
        default_element(element),
        true,
        anchor,
        iterations,
        border_mode,
        border_value
    )
}

#: Any of the compound morphological operations.
pub fn morphology_ex(
    src: Mat,
    op: int,
    element: Mat? = null,
    anchor: Point? = null,
    iterations: int = 1,
    border_mode: int = BORDER_REPLICATE,
    border_value: Scalar = Scalar(0.0, 0.0, 0.0, 0.0)
) -> Mat {
    let shape = default_element(element)
    if op == MORPH_ERODE {
        return erode(
            src,
            element: shape,
            anchor: anchor,
            iterations: iterations,
            border_mode: border_mode,
            border_value: border_value
        )
    }
    if op == MORPH_DILATE {
        return dilate(
            src,
            element: shape,
            anchor: anchor,
            iterations: iterations,
            border_mode: border_mode,
            border_value: border_value
        )
    }
    if op == MORPH_OPEN {
        let eroded = erode(
            src,
            element: shape,
            anchor: anchor,
            iterations: iterations,
            border_mode: border_mode,
            border_value: border_value
        )
        return dilate(
            eroded,
            element: shape,
            anchor: anchor,
            iterations: iterations,
            border_mode: border_mode,
            border_value: border_value
        )
    }
    if op == MORPH_CLOSE {
        let dilated = dilate(
            src,
            element: shape,
            anchor: anchor,
            iterations: iterations,
            border_mode: border_mode,
            border_value: border_value
        )
        return erode(
            dilated,
            element: shape,
            anchor: anchor,
            iterations: iterations,
            border_mode: border_mode,
            border_value: border_value
        )
    }
    if op == MORPH_GRADIENT {
        let grown = dilate(src, element: shape, anchor: anchor, border_mode: border_mode)
        let shrunk = erode(src, element: shape, anchor: anchor, border_mode: border_mode)
        return binary_op(grown, shrunk, OP_SUBTRACT)
    }
    if op == MORPH_TOPHAT {
        let opened = morphology_ex(
            src,
            MORPH_OPEN,
            element: shape,
            anchor: anchor,
            iterations: iterations,
            border_mode: border_mode
        )
        return binary_op(src, opened, OP_SUBTRACT)
    }
    if op == MORPH_BLACKHAT {
        let closed = morphology_ex(
            src,
            MORPH_CLOSE,
            element: shape,
            anchor: anchor,
            iterations: iterations,
            border_mode: border_mode
        )
        return binary_op(closed, src, OP_SUBTRACT)
    }
    throw ValueError(f"morphology_ex does not know operation {op}")
}
