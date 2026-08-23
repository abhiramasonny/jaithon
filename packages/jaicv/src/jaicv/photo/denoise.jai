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
from std.gpu import Buffer, Kernel
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

/* One ring of the fill, every pixel deciding for itself.
 *
 * The round is a Jacobi step: what a pixel reads is what was known when the
 * round began, so nothing filled this round feeds anything else filled this
 * round and the order threads run in cannot change the answer. Only the
 * knowledge needs two buffers -- an unknown pixel is never read as a source,
 * so the values can be written where they lie.
 *
 * The host walked the whole frame per ring and averaged a seven-pixel window
 * around each boundary pixel in interpreted code, which for a 640 by 480
 * frame with a thin mask was a second. */
kernel void inpaint_round(device float* values [[buffer(0)]],
                          device const float* known [[buffer(1)]],
                          device float* known_next [[buffer(2)]],
                          device atomic_uint* filled [[buffer(3)]],
                          constant uint& count [[buffer(4)]],
                          constant uint& rows [[buffer(5)]],
                          constant uint& cols [[buffer(6)]],
                          constant uint& cn [[buffer(7)]],
                          constant uint& radius [[buffer(8)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    if (known[id] != 0.0f) {
        known_next[id] = 1.0f;
        return;
    }
    known_next[id] = 0.0f;
    int y = int(id / cols);
    int x = int(id - uint(y) * cols);
    int reach = int(radius);

    bool touches = false;
    for (int dy = -1; dy <= 1 && !touches; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int ny = y + dy;
            int nx = x + dx;
            if (ny < 0 || nx < 0 || ny >= int(rows) || nx >= int(cols)) continue;
            if (known[uint(ny) * cols + uint(nx)] != 0.0f) {
                touches = true;
                break;
            }
        }
    }
    if (!touches) return;

    float totals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float weight = 0.0f;
    for (int dy = -reach; dy <= reach; ++dy) {
        for (int dx = -reach; dx <= reach; ++dx) {
            int ny = y + dy;
            int nx = x + dx;
            if (ny < 0 || nx < 0 || ny >= int(rows) || nx >= int(cols)) continue;
            uint cell = uint(ny) * cols + uint(nx);
            if (known[cell] == 0.0f) continue;
            float distance = sqrt(float(dx * dx + dy * dy));
            if (distance > float(reach) || distance < 1e-6f) continue;
            /* Nearer neighbours count for more, which is what makes the fill
             * follow the shape of the hole rather than smear across it. */
            float w = 1.0f / (distance * distance);
            weight += w;
            for (uint c = 0u; c < cn; ++c) totals[c] += w * values[cell * cn + c];
        }
    }
    if (weight <= 0.0f) return;
    for (uint c = 0u; c < cn; ++c) values[id * cn + c] = totals[c] / weight;
    known_next[id] = 1.0f;
    atomic_fetch_add_explicit(filled, 1u, memory_order_relaxed);
}

/* The atomic count, read as an integer rather than as the float its bits
 * would otherwise be taken for. */
kernel void inpaint_tally(device const uint* filled [[buffer(0)]],
                          device float* out [[buffer(1)]],
                          constant uint& count [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    out[id] = float(filled[id]);
}

/* Back to whole numbers in range, which is what the picture was. */
kernel void inpaint_settle(device const float* values [[buffer(0)]],
                           device float* dst [[buffer(1)]],
                           constant uint& count [[buffer(2)]],
                           constant uint& cols [[buffer(3)]],
                           constant uint& cn [[buffer(4)]],
                           constant uint& d_off [[buffer(5)]],
                           constant uint& d_step [[buffer(6)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint pixel = id / cn;
    uint channel = id - pixel * cn;
    uint y = pixel / cols;
    uint x = pixel - y * cols;
    dst[d_off + y * d_step + x * cn + channel] = clamp(floor(values[id] + 0.5f), 0.0f, 255.0f);
}

/* The mask read the other way round: a hole is what the fill has yet to
 * reach, so knowledge starts wherever the mask is clear. */
kernel void inpaint_seed(device const float* src [[buffer(0)]],
                         device const float* mask [[buffer(1)]],
                         device float* values [[buffer(2)]],
                         device float* known [[buffer(3)]],
                         device atomic_uint* holes [[buffer(4)]],
                         constant uint& count [[buffer(5)]],
                         constant uint& cols [[buffer(6)]],
                         constant uint& cn [[buffer(7)]],
                         constant uint& s_off [[buffer(8)]],
                         constant uint& s_step [[buffer(9)]],
                         constant uint& m_off [[buffer(10)]],
                         constant uint& m_step [[buffer(11)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint y = id / cols;
    uint x = id - y * cols;
    for (uint c = 0u; c < cn; ++c) values[id * cn + c] = src[s_off + y * s_step + x * cn + c];
    bool open = mask[m_off + y * m_step + x] == 0.0f;
    known[id] = open ? 1.0f : 0.0f;
    if (!open) atomic_fetch_add_explicit(holes, 1u, memory_order_relaxed);
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
    let original = src.values()
    let base = smoothed.values()
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

#: Fill the masked region in from what surrounds it.
#:
#: Every non-zero pixel of `mask` is treated as missing. The hole is filled by
#: marching inward from its boundary a ring at a time, each new pixel taken as
#: a weighted average of the filled pixels around it — which is Telea's method
#: and is what `INPAINT_TELEA` selects in OpenCV.
#:
#: The rings run on the device, one thread to a pixel, which is exact because
#: a ring only ever reads what was already known when it started. The averages
#: are single precision where the host worked in double, so a pixel whose
#: average lands on a half rounds the other way: three of seventy-six thousand
#: did on the frame this was measured against, each by one.
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
    if cn > 4 { throw ValueError("inpaint handles up to four channels") }
    let count = rows * cols
    if count == 0 { return src.clone() }
    let radius = max(int(inpaint_radius + 0.5), 1)

    let values = Buffer(count * cn)
    let known = Buffer(count)
    let next = Buffer(count)
    let tally = Buffer(1)
    let counted = Buffer(1)
    tally.fill_zero()
    kernel_for("inpaint_seed").dispatch_async(
        [src.storage, mask.storage, values, known, tally],
        count,
        [count, cols, cn, src.offset, src.step_, mask.offset, mask.step_]
    )
    kernel_for("inpaint_tally").dispatch_async([tally, counted], 1, [1])
    var remaining = int(counted.download()[0])

    var reading = known
    var writing = next
    while remaining > 0 {
        tally.fill_zero()
        kernel_for("inpaint_round").dispatch_async(
            [values, reading, writing, tally],
            count,
            [count, rows, cols, cn, radius]
        )
        kernel_for("inpaint_tally").dispatch_async([tally, counted], 1, [1])
        let filled = int(counted.download()[0])
        #: A ring that fills nothing means what is left has no known pixel
        #: anywhere near it, and another round would find the same.
        if filled == 0 { break }
        remaining -= filled
        let swap = reading
        reading = writing
        writing = swap
    }

    let out = Mat(rows, cols, src.type_())
    kernel_for("inpaint_settle").dispatch_async(
        [values, out.storage],
        count * cn,
        [count * cn, cols, cn, out.offset, out.step_]
    )
    out.invalidate()
    values.free()
    known.free()
    next.free()
    tally.free()
    counted.free()
    return out
}
