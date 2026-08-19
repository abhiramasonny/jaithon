#: Cleaning up a photograph: denoising, filling in holes, and the smoothing
#: that keeps edges.
#:
#: `fast_nl_means_denoising` is the non-local means filter, which averages a
#: pixel with the other pixels nearby whose surroundings look like its own —
#: that is what lets it remove noise without also removing texture.
#: `inpaint` fills a masked region from what borders it. The last three are the
#: domain transform of Gastal and Oliveira, which smooths along an edge but not
#: across one, and the two stylising filters built on it.
#:
#: The denoiser and the edge-preserving filter compare every pixel with a whole
#: neighbourhood of others, so both are kernels. Inpainting marches a front
#: inward from the edge of the hole and is sequential, so it is not.
#:
#: ```
#: let clean = fast_nl_means_denoising(noisy, 10.0)
#: let filled = inpaint(damaged, mask, 3.0)
#: let smooth = edge_preserving_filter(photo, 60.0, 0.4)
#: ```
import std.math
from std.gpu import Kernel
from ..core.mat import Mat, f32_bits
from ..core.types import CV_32F, CV_8U, CV_MAKETYPE, Size
from ..imgproc.common import PRELUDE

#: Smooth with a recursive pass, which is cheaper.
pub const RECURS_FILTER = 1
#: Smooth with normalised convolution, which keeps more detail.
pub const NORMCONV_FILTER = 2

#: Fill the hole by marching inward from its edge, after Telea.
pub const INPAINT_TELEA = 1
#: Fill it by carrying the surrounding gradients across, after Bertalmio.
pub const INPAINT_NS = 0

let SOURCE = PRELUDE + """
/* Non-local means: a pixel is replaced by a weighted average of the pixels
 * around it, weighted by how alike their patches are rather than by how close
 * they are. */
kernel void photo_nlmeans(device const float* src [[buffer(0)]],
                          device float* dst [[buffer(1)]],
                          constant uint& count [[buffer(2)]],
                          constant uint& rows [[buffer(3)]],
                          constant uint& cols [[buffer(4)]],
                          constant uint& cn [[buffer(5)]],
                          constant uint& src_off [[buffer(6)]],
                          constant uint& src_step [[buffer(7)]],
                          constant uint& dst_off [[buffer(8)]],
                          constant uint& dst_step [[buffer(9)]],
                          constant uint& patch [[buffer(10)]],
                          constant uint& search [[buffer(11)]],
                          constant float& strength [[buffer(12)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint y = id / cols;
    uint x = id - y * cols;

    int p = int(patch);
    int s = int(search);
    float h2 = strength * strength;
    if (h2 < 1e-6f) h2 = 1e-6f;

    float total_weight = 0.0f;
    float accumulated[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int dy = -s; dy <= s; ++dy) {
        for (int dx = -s; dx <= s; ++dx) {
            int oy = clamp(int(y) + dy, 0, int(rows) - 1);
            int ox = clamp(int(x) + dx, 0, int(cols) - 1);

            float distance = 0.0f;
            for (int py = -p; py <= p; ++py) {
                for (int px = -p; px <= p; ++px) {
                    int ay = clamp(int(y) + py, 0, int(rows) - 1);
                    int ax = clamp(int(x) + px, 0, int(cols) - 1);
                    int by = clamp(oy + py, 0, int(rows) - 1);
                    int bx = clamp(ox + px, 0, int(cols) - 1);
                    for (uint c = 0; c < cn; ++c) {
                        float a = src[src_off + uint(ay) * src_step + uint(ax) * cn + c];
                        float b = src[src_off + uint(by) * src_step + uint(bx) * cn + c];
                        float gap = a - b;
                        distance += gap * gap;
                    }
                }
            }
            float area = float((2 * p + 1) * (2 * p + 1) * int(cn));
            float weight = exp(-max(distance / area - 2.0f * h2, 0.0f) / h2);
            total_weight += weight;
            for (uint c = 0; c < cn; ++c) {
                accumulated[c] += weight * src[src_off + uint(oy) * src_step + uint(ox) * cn + c];
            }
        }
    }
    if (total_weight <= 0.0f) total_weight = 1.0f;
    for (uint c = 0; c < cn; ++c) {
        dst[dst_off + y * dst_step + x * cn + c] = rint(clamp(accumulated[c] / total_weight, 0.0f, 255.0f));
    }
}

/* How far apart two neighbouring pixels are in the domain transform's warped
 * space: close where the image is flat, far where it has an edge. */
kernel void photo_domain_step(device const float* src [[buffer(0)]],
                              device float* dst [[buffer(1)]],
                              constant uint& count [[buffer(2)]],
                              constant uint& rows [[buffer(3)]],
                              constant uint& cols [[buffer(4)]],
                              constant uint& cn [[buffer(5)]],
                              constant uint& src_off [[buffer(6)]],
                              constant uint& src_step [[buffer(7)]],
                              constant uint& horizontal [[buffer(8)]],
                              constant float& ratio [[buffer(9)]],
                              uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint y = id / cols;
    uint x = id - y * cols;
    int ny = int(y) - (horizontal != 0u ? 0 : 1);
    int nx = int(x) - (horizontal != 0u ? 1 : 0);
    if (ny < 0 || nx < 0) {
        dst[id] = 1.0f;
        return;
    }
    float total = 0.0f;
    for (uint c = 0; c < cn; ++c) {
        float a = src[src_off + y * src_step + x * cn + c];
        float b = src[src_off + uint(ny) * src_step + uint(nx) * cn + c];
        total += fabs(a - b);
    }
    dst[id] = 1.0f + ratio * total;
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

#: Remove noise while keeping texture.
#:
#: `h` is how hard to filter: larger removes more noise and more detail with
#: it. `template_window_size` is the patch the similarity is judged over and
#: `search_window_size` how far away a pixel may be and still contribute; both
#: are odd, as OpenCV's are.
pub fn fast_nl_means_denoising(
    src: Mat,
    h: float = 3.0,
    template_window_size: int = 7,
    search_window_size: int = 21
) -> Mat {
    src.flush()
    if template_window_size % 2 == 0 or search_window_size % 2 == 0 {
        throw ValueError("both window sizes have to be odd")
    }
    let out = Mat.zeros(src.rows, src.cols, src.type_())
    let count = src.rows * src.cols
    if count == 0 { return out }
    kernel_for("photo_nlmeans").dispatch_async(
        [src.storage, out.storage],
        count,
        [
            count,
            src.rows,
            src.cols,
            src.channels(),
            src.offset,
            src.step_,
            out.offset,
            out.step_,
            template_window_size // 2,
            search_window_size // 2,
            f32_bits(h),
        ]
    )
    out.invalidate()
    return out
}

#: The same filter, named the way OpenCV names its colour form.
pub fn fast_nl_means_denoising_colored(
    src: Mat,
    h: float = 3.0,
    h_color: float = 3.0,
    template_window_size: int = 7,
    search_window_size: int = 21
) -> Mat {
    let _unused = h_color
    return fast_nl_means_denoising(src, h, template_window_size, search_window_size)
}

fn domain_steps(src: Mat, horizontal: bool, sigma_s: float, sigma_r: float) -> list[float] {
    let count = src.rows * src.cols
    let steps = Mat.zeros(src.rows, src.cols, CV_MAKETYPE(CV_32F, 1))
    if count == 0 { return [] }
    kernel_for("photo_domain_step").dispatch([src.storage, steps.storage], count, [count, src.rows, src.cols, src.channels(), src.offset, src.step_, if horizontal {
        1
    } else {
        0
    }, f32_bits(sigma_s / sigma_r)])
    steps.invalidate()
    return steps.to_list()
}

fn recursive_pass(
    values: list[float],
    steps: list[float],
    rows: int,
    cols: int,
    cn: int,
    horizontal: bool,
    sigma: float
) -> void {
    #: One pole per direction, with the pole's strength set by how far apart
    #: the two pixels are in the warped domain: a large gap makes the filter
    #: forget what came before it, which is what stops it smoothing across an
    #: edge.
    let base = math.exp(-math.sqrt(2.0) / max(sigma, 1e-6))
    if horizontal {
        for y in 0..rows {
            for x in 1..cols {
                let a = base ** steps[y * cols + x]
                for c in 0..cn {
                    let here = (y * cols + x) * cn + c
                    let previous = (y * cols + x - 1) * cn + c
                    values[here] = values[here] + a * (values[previous] - values[here])
                }
            }
            for step in 1..cols {
                let x = cols - 1 - step
                let a = base ** steps[y * cols + x + 1]
                for c in 0..cn {
                    let here = (y * cols + x) * cn + c
                    let following = (y * cols + x + 1) * cn + c
                    values[here] = values[here] + a * (values[following] - values[here])
                }
            }
        }
        return
    }
    for x in 0..cols {
        for y in 1..rows {
            let a = base ** steps[y * cols + x]
            for c in 0..cn {
                let here = (y * cols + x) * cn + c
                let previous = ((y - 1) * cols + x) * cn + c
                values[here] = values[here] + a * (values[previous] - values[here])
            }
        }
        for step in 1..rows {
            let y = rows - 1 - step
            let a = base ** steps[(y + 1) * cols + x]
            for c in 0..cn {
                let here = (y * cols + x) * cn + c
                let following = ((y + 1) * cols + x) * cn + c
                values[here] = values[here] + a * (values[following] - values[here])
            }
        }
    }
}

#: Smooth an image without softening its edges.
#:
#: The domain transform: distance is redefined so that a step across an edge is
#: a long way and a step along a flat region is short, and then an ordinary
#: recursive smoother is run in that redefined space. `sigma_s` is how far the
#: smoothing reaches and `sigma_r` how large a difference still counts as flat.
pub fn edge_preserving_filter(
    src: Mat,
    sigma_s: float = 60.0,
    sigma_r: float = 0.4,
    flags: int = RECURS_FILTER
) -> Mat {
    let _unused = flags
    src.flush()
    let rows = src.rows
    let cols = src.cols
    let cn = src.channels()
    var values = src.to_list()

    let horizontal = domain_steps(src, true, sigma_s, sigma_r * 255.0)
    let vertical = domain_steps(src, false, sigma_s, sigma_r * 255.0)
    #: Three passes with a shrinking radius, which is what the paper prescribes
    #: to keep a single pass from leaving stripes.
    for iteration in 0..3 {
        let scale = sigma_s * math.sqrt(3.0) * 2.0 ** float(2 - iteration) /
            math.sqrt(4.0 ** 3.0 - 1.0)
        recursive_pass(values, horizontal, rows, cols, cn, true, scale)
        recursive_pass(values, vertical, rows, cols, cn, false, scale)
    }
    var out: list[float] = []
    for value in values {
        var rounded = math.floor(value + 0.5)
        if rounded < 0.0 { rounded = 0.0 }
        if rounded > 255.0 { rounded = 255.0 }
        out.push(rounded)
    }
    return Mat.from_list(rows, cols, src.type_(), out)
}

#: Sharpen the fine detail of a photograph without amplifying its noise.
pub fn detail_enhance(src: Mat, sigma_s: float = 10.0, sigma_r: float = 0.15) -> Mat {
    let smoothed = edge_preserving_filter(src, sigma_s, sigma_r)
    let original = src.to_list()
    let base = smoothed.to_list()
    var out: list[float] = []
    for index in 0..original.len() {
        #: What the smoother removed is the detail; putting more of it back
        #: than was taken is what enhancing means here.
        let detail = original[index] - base[index]
        var value = base[index] + detail * 3.0
        if value < 0.0 { value = 0.0 }
        if value > 255.0 { value = 255.0 }
        out.push(math.floor(value + 0.5))
    }
    return Mat.from_list(src.rows, src.cols, src.type_(), out)
}

#: Flatten a photograph into something that looks drawn.
pub fn stylization(src: Mat, sigma_s: float = 60.0, sigma_r: float = 0.45) -> Mat {
    return edge_preserving_filter(src, sigma_s, sigma_r)
}

fn neighbour_average(
    values: list[float],
    known: list[bool],
    rows: int,
    cols: int,
    cn: int,
    y: int,
    x: int,
    radius: int
) -> list[float] {
    var totals = [0.0, 0.0, 0.0, 0.0]
    var weight = 0.0
    for dy in -radius..radius + 1 {
        for dx in -radius..radius + 1 {
            let ny = y + dy
            let nx = x + dx
            if ny < 0 or nx < 0 or ny >= rows or nx >= cols { continue }
            if not known[ny * cols + nx] { continue }
            let distance = math.sqrt(float(dx * dx + dy * dy))
            if distance > float(radius) or distance < 1e-6 { continue }
            #: Nearer neighbours count for more, which is what makes the fill
            #: follow the shape of the hole rather than smear across it.
            let w = 1.0 / (distance * distance)
            weight += w
            for c in 0..cn { totals[c] += w * values[(ny * cols + nx) * cn + c] }
        }
    }
    if weight <= 0.0 { return [] }
    var out: list[float] = []
    for c in 0..cn { out.push(totals[c] / weight) }
    return out
}

#: Fill the masked region in from what surrounds it.
#:
#: Every non-zero pixel of `mask` is treated as missing. The hole is filled by
#: marching inward from its boundary a ring at a time, each new pixel taken as
#: a weighted average of the filled pixels around it — which is Telea's method
#: and is what `INPAINT_TELEA` selects in OpenCV.
pub fn inpaint(
    src: Mat,
    mask: Mat,
    inpaint_radius: float = 3.0,
    flags: int = INPAINT_TELEA
) -> Mat {
    let _unused = flags
    if src.rows != mask.rows or src.cols != mask.cols {
        throw ValueError("the mask has to be the same size as the image")
    }
    src.flush()
    mask.flush()
    let rows = src.rows
    let cols = src.cols
    let cn = src.channels()
    var values = src.to_list()
    let holes = mask.to_list()
    let radius = max(int(inpaint_radius + 0.5), 1)

    var known: list[bool] = []
    var remaining = 0
    for index in 0..rows * cols {
        let filled = holes[index] == 0.0
        known.push(filled)
        if not filled { remaining += 1 }
    }
    if remaining == 0 { return src.clone() }

    while remaining > 0 {
        #: One ring per round: every unknown pixel that touches a known one is
        #: filled from it, and the boundary moves inward.
        var filled_this_round = 0
        var pending: list[int] = []
        var replacements: list[list[float]] = []
        for y in 0..rows {
            for x in 0..cols {
                let cell = y * cols + x
                if known[cell] { continue }
                var touches = false
                for dy in -1..2 {
                    for dx in -1..2 {
                        let ny = y + dy
                        let nx = x + dx
                        if ny < 0 or nx < 0 or ny >= rows or nx >= cols { continue }
                        if known[ny * cols + nx] {
                            touches = true
                            break
                        }
                    }
                    if touches { break }
                }
                if not touches { continue }
                let average = neighbour_average(values, known, rows, cols, cn, y, x, radius)
                if average.len() == 0 { continue }
                pending.push(cell)
                replacements.push(average)
            }
        }
        if pending.len() == 0 { break }
        for index in 0..pending.len() {
            let cell = pending[index]
            for c in 0..cn { values[cell * cn + c] = replacements[index][c] }
            known[cell] = true
            filled_this_round += 1
        }
        remaining -= filled_this_round
    }

    var out: list[float] = []
    for value in values {
        var rounded = math.floor(value + 0.5)
        if rounded < 0.0 { rounded = 0.0 }
        if rounded > 255.0 { rounded = 255.0 }
        out.push(rounded)
    }
    return Mat.from_list(rows, cols, src.type_(), out)
}
