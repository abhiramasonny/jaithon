#: `canny` — gradient, non-maximum suppression, and hysteresis.
#:
#: The gradient magnitude is the L1 sum of the Sobel derivatives by default,
#: which is what OpenCV computes unless `l2_gradient` asks for the Euclidean
#: one. Hysteresis runs as a fixed-point iteration on the device: a weak edge
#: is promoted while any neighbour is strong, repeated until nothing changes.
#:
#: ```
#: let outline = canny(gray, 80.0, 200.0)
#: ```
import std.math
from std.gpu import Buffer, Kernel
from ..core.mat import Mat, f32_bits
from ..core.types import BORDER_REPLICATE, CV_8U, CV_32F, CV_MAKETYPE
from .common import PRELUDE, saturation_args
from .filter import sobel

#: Most passes hysteresis needs before it gives up chasing a chain of weak
#: edges. Each pass reaches one pixel further along a chain.
pub const MAX_HYSTERESIS_PASSES = 64

let SOURCE = PRELUDE + """
kernel void canny_suppress(device const float* dx [[buffer(0)]],
                           device const float* dy [[buffer(1)]],
                           device float* labels [[buffer(2)]],
                           constant uint& count [[buffer(3)]],
                           constant uint& rows [[buffer(4)]],
                           constant uint& cols [[buffer(5)]],
                           constant uint& dx_off [[buffer(6)]],
                           constant uint& dx_step [[buffer(7)]],
                           constant uint& dy_off [[buffer(8)]],
                           constant uint& dy_step [[buffer(9)]],
                           constant uint& out_off [[buffer(10)]],
                           constant uint& out_step [[buffer(11)]],
                           constant uint& l2 [[buffer(12)]],
                           constant uint& low_bits [[buffer(13)]],
                           constant uint& high_bits [[buffer(14)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint y = id / cols;
    uint x = id - y * cols;

    float gx = dx[dx_off + y * dx_step + x];
    float gy = dy[dy_off + y * dy_step + x];
    float magnitude = l2 != 0u ? sqrt(gx * gx + gy * gy) : fabs(gx) + fabs(gy);
    float high = as_type<float>(high_bits);
    float low = as_type<float>(low_bits);
    if (magnitude <= low) {
        labels[out_off + y * out_step + x] = 0.0f;
        return;
    }

    /* The gradient direction picks which two neighbours the magnitude has to
     * beat. tan(22.5) and tan(67.5) split the four sectors. */
    float ax = fabs(gx);
    float ay = fabs(gy);
    int nx = 0;
    int ny = 0;
    if (ay <= ax * 0.41421356f) { nx = 1; ny = 0; }
    else if (ax <= ay * 0.41421356f) { nx = 0; ny = 1; }
    else if ((gx > 0.0f) == (gy > 0.0f)) { nx = 1; ny = 1; }
    else { nx = 1; ny = -1; }

    float before = 0.0f;
    float after = 0.0f;
    int px = int(x) - nx;
    int py = int(y) - ny;
    if (px >= 0 && py >= 0 && px < int(cols) && py < int(rows)) {
        float bx = dx[dx_off + uint(py) * dx_step + uint(px)];
        float by = dy[dy_off + uint(py) * dy_step + uint(px)];
        before = l2 != 0u ? sqrt(bx * bx + by * by) : fabs(bx) + fabs(by);
    }
    int qx = int(x) + nx;
    int qy = int(y) + ny;
    if (qx >= 0 && qy >= 0 && qx < int(cols) && qy < int(rows)) {
        float cx = dx[dx_off + uint(qy) * dx_step + uint(qx)];
        float cy = dy[dy_off + uint(qy) * dy_step + uint(qx)];
        after = l2 != 0u ? sqrt(cx * cx + cy * cy) : fabs(cx) + fabs(cy);
    }
    if (magnitude < before || magnitude < after) {
        labels[out_off + y * out_step + x] = 0.0f;
        return;
    }
    labels[out_off + y * out_step + x] = magnitude > high ? 2.0f : 1.0f;
}

kernel void canny_promote(device float* labels [[buffer(0)]],
                          device float* changed [[buffer(1)]],
                          constant uint& count [[buffer(2)]],
                          constant uint& rows [[buffer(3)]],
                          constant uint& cols [[buffer(4)]],
                          constant uint& off [[buffer(5)]],
                          constant uint& step [[buffer(6)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint y = id / cols;
    uint x = id - y * cols;
    if (labels[off + y * step + x] != 1.0f) return;
    for (int dy = -1; dy <= 1; ++dy) {
        int sy = int(y) + dy;
        if (sy < 0 || sy >= int(rows)) continue;
        for (int dx = -1; dx <= 1; ++dx) {
            int sx = int(x) + dx;
            if (sx < 0 || sx >= int(cols)) continue;
            if (labels[off + uint(sy) * step + uint(sx)] == 2.0f) {
                labels[off + y * step + x] = 2.0f;
                atomic_fetch_add_explicit((device atomic<float>*)changed, 1.0f,
                                          memory_order_relaxed);
                return;
            }
        }
    }
}

kernel void canny_finish(device const float* labels [[buffer(0)]],
                         device float* dst [[buffer(1)]],
                         constant uint& count [[buffer(2)]],
                         constant uint& cols [[buffer(3)]],
                         constant uint& src_off [[buffer(4)]],
                         constant uint& src_step [[buffer(5)]],
                         constant uint& dst_off [[buffer(6)]],
                         constant uint& dst_step [[buffer(7)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint y = id / cols;
    uint x = id - y * cols;
    dst[dst_off + y * dst_step + x] =
        labels[src_off + y * src_step + x] == 2.0f ? 255.0f : 0.0f;
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

#: Find edges by the Canny method.
#:
#: `threshold1` and `threshold2` are the hysteresis pair; the smaller is the
#: level a pixel needs to survive at all, the larger the level that starts a
#: chain. `aperture_size` is the Sobel window.
pub fn canny(
    src: Mat,
    threshold1: float,
    threshold2: float,
    aperture_size: int = 3,
    l2_gradient: bool = false
) -> Mat {
    src.flush()
    if src.channels() != 1 { throw ValueError("canny needs a single-channel Mat") }
    let low = min(threshold1, threshold2)
    let high = max(threshold1, threshold2)
    let dx = sobel(src, CV_32F, 1, 0, ksize: aperture_size, border_mode: BORDER_REPLICATE)
    let dy = sobel(src, CV_32F, 0, 1, ksize: aperture_size, border_mode: BORDER_REPLICATE)
    let labels = Mat.zeros(src.rows, src.cols, CV_MAKETYPE(CV_32F, 1))
    let out = Mat.zeros(src.rows, src.cols, CV_MAKETYPE(CV_8U, 1))
    let count = src.rows * src.cols
    if count == 0 { return out }
    kernel_for("canny_suppress").dispatch_async([dx.storage, dy.storage, labels.storage], count, [count, src.rows, src.cols, dx.offset, dx.step_, dy.offset, dy.step_, labels.offset, labels.step_, if l2_gradient {
        1
    } else {
        0
    }, f32_bits(low), f32_bits(high)])
    labels.invalidate()

    #: Weak edges join a chain one ring at a time, so the pass count bounds how
    #: far a chain can reach; a counter says when a pass changed nothing.
    #:
    #: The counter is read every eighth pass, not every pass. Reading it is a
    #: round trip to the device and back, and the passes themselves are a
    #: single cheap kernel over the image -- so asking after each one spent
    #: almost the whole of Canny waiting: eighteen milliseconds at VGA, of
    #: which the arithmetic was a fraction. Promotion only ever turns a weak
    #: edge strong, so a few passes past convergence change nothing and cost
    #: only their own dispatch.
    let changed = Buffer(1)
    let promote = kernel_for("canny_promote")
    let CHECK_EVERY = 8
    var pass_number = 0
    while pass_number < MAX_HYSTERESIS_PASSES {
        changed.fill_zero()
        let block = min(CHECK_EVERY, MAX_HYSTERESIS_PASSES - pass_number)
        for _step in 0..block {
            promote.dispatch_async(
                [labels.storage, changed],
                count,
                [count, src.rows, src.cols, labels.offset, labels.step_]
            )
        }
        pass_number += block
        if changed.download()[0] == 0.0 { break }
    }
    changed.free()

    kernel_for("canny_finish").dispatch_async(
        [labels.storage, out.storage],
        count,
        [count, src.cols, labels.offset, labels.step_, out.offset, out.step_]
    )
    out.invalidate()
    return out
}
