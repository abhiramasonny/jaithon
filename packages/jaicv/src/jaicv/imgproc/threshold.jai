#: Thresholding: the fixed forms, the adaptive forms, and the two automatic
#: threshold choosers.
#:
#: ```
#: let (level, mask) = threshold(gray, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU)
#: let local = adaptive_threshold(gray, 255.0, ADAPTIVE_THRESH_GAUSSIAN_C,
#:                               THRESH_BINARY, 11, 2.0)
#: ```
import std.math
from std.gpu import Kernel
from ..core.mat import Mat, f32_bits
from ..core.types import BORDER_REPLICATE, CV_8U, CV_32F, CV_MAKETYPE, CV_MAT_DEPTH
from ..core.types import Size, depth_max
from .common import PRELUDE, saturation_args
from .filter import blur, gaussian_blur

#: Above the level takes `max_value`, below takes zero.
pub const THRESH_BINARY = 0
#: Above the level takes zero, below takes `max_value`.
pub const THRESH_BINARY_INV = 1
#: Above the level is clipped to it, below is left alone.
pub const THRESH_TRUNC = 2
#: Below the level takes zero, above is left alone.
pub const THRESH_TOZERO = 3
#: Above the level takes zero, below is left alone.
pub const THRESH_TOZERO_INV = 4
#: Choose the level by Otsu's method, then apply the mode below it.
pub const THRESH_OTSU = 8
#: Choose the level by the triangle method.
pub const THRESH_TRIANGLE = 16

#: Average the neighbourhood.
pub const ADAPTIVE_THRESH_MEAN_C = 0
#: Average the neighbourhood, weighted by a Gaussian.
pub const ADAPTIVE_THRESH_GAUSSIAN_C = 1

let SOURCE = PRELUDE + """
kernel void threshold_fixed(device const float* src [[buffer(0)]],
                            device float* dst [[buffer(1)]],
                            constant uint& count [[buffer(2)]],
                            constant uint& cols [[buffer(3)]],
                            constant uint& cn [[buffer(4)]],
                            constant uint& src_off [[buffer(5)]],
                            constant uint& src_step [[buffer(6)]],
                            constant uint& dst_off [[buffer(7)]],
                            constant uint& dst_step [[buffer(8)]],
                            constant uint& mode [[buffer(9)]],
                            constant uint& level_bits [[buffer(10)]],
                            constant uint& max_bits [[buffer(11)]],
                            constant uint& low_bits [[buffer(12)]],
                            constant uint& high_bits [[buffer(13)]],
                            constant uint& integral [[buffer(14)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint width = cols * cn;
    uint y = id / width;
    uint rest = id - y * width;
    float value = src[src_off + y * src_step + rest];
    float level = as_type<float>(level_bits);
    float top = as_type<float>(max_bits);
    float out = value;
    if (mode == 0u) out = value > level ? top : 0.0f;
    else if (mode == 1u) out = value > level ? 0.0f : top;
    else if (mode == 2u) out = value > level ? level : value;
    else if (mode == 3u) out = value > level ? value : 0.0f;
    else if (mode == 4u) out = value > level ? 0.0f : value;
    put(dst, dst_off + y * dst_step + rest, out,
        as_type<float>(low_bits), as_type<float>(high_bits), integral);
}

kernel void threshold_adaptive(device const float* src [[buffer(0)]],
                               device const float* local [[buffer(1)]],
                               device float* dst [[buffer(2)]],
                               constant uint& count [[buffer(3)]],
                               constant uint& cols [[buffer(4)]],
                               constant uint& src_off [[buffer(5)]],
                               constant uint& src_step [[buffer(6)]],
                               constant uint& local_off [[buffer(7)]],
                               constant uint& local_step [[buffer(8)]],
                               constant uint& dst_off [[buffer(9)]],
                               constant uint& dst_step [[buffer(10)]],
                               constant uint& mode [[buffer(11)]],
                               constant uint& delta_bits [[buffer(12)]],
                               constant uint& max_bits [[buffer(13)]],
                               constant uint& low_bits [[buffer(14)]],
                               constant uint& high_bits [[buffer(15)]],
                               constant uint& integral [[buffer(16)]],
                               uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint y = id / cols;
    uint x = id - y * cols;
    float value = src[src_off + y * src_step + x];
    /* OpenCV rounds the neighbourhood average to an integer before taking the
     * offset away, so a threshold sits on the same side of a tie as it does
     * there. */
    float level = rint(local[local_off + y * local_step + x]) - as_type<float>(delta_bits);
    float top = as_type<float>(max_bits);
    float out = mode == 0u ? (value > level ? top : 0.0f) : (value > level ? 0.0f : top);
    put(dst, dst_off + y * dst_step + x, out,
        as_type<float>(low_bits), as_type<float>(high_bits), integral);
}

/* The counts live in an ordinary float buffer so the result reads back as a
 * Mat; the atomic add goes through a cast, which is what Metal offers for
 * float accumulation. */
/* A histogram counted per threadgroup and merged once.
 *
 * A thread per pixel, each adding into one of 256 shared bins, is two million
 * atomic writes contending over a quarter of a kilobyte -- and on device
 * memory, where an atomic is at its most expensive. Each group counts into its
 * own bins in threadgroup memory instead and adds those to the picture's at
 * the end, which turns two million device atomics into sixty-five thousand.
 *
 * A fixed grid, strided: the number of groups is what decides how many merges
 * there are, so it is chosen rather than derived from the pixel count. */
kernel void threshold_histogram(device const float* src [[buffer(0)]],
                                device float* histogram [[buffer(1)]],
                                constant uint& count [[buffer(2)]],
                                constant uint& cols [[buffer(3)]],
                                constant uint& src_off [[buffer(4)]],
                                constant uint& src_step [[buffer(5)]],
                                constant uint& stride [[buffer(6)]],
                                uint id [[thread_position_in_grid]],
                                uint lid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint bins[256];
    atomic_store_explicit(&bins[lid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint at = id; at < count; at += stride) {
        uint y = at / cols;
        uint x = at - y * cols;
        float value = src[src_off + y * src_step + x];
        uint slot = uint(clamp(rint(value), 0.0f, 255.0f));
        atomic_fetch_add_explicit(&bins[slot], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint mine = atomic_load_explicit(&bins[lid], memory_order_relaxed);
    if (mine == 0u) return;
    atomic_fetch_add_explicit((device atomic<float>*)(histogram + lid), float(mine),
                              memory_order_relaxed);
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

#: Counts of every 8-bit value in a single-channel image.
pub fn histogram_256(src: Mat) -> list[int] {
    src.flush()
    if src.channels() != 1 { throw ValueError("histogram_256 needs a single-channel Mat") }
    let counts = Mat.zeros(1, 256, CV_MAKETYPE(CV_32F, 1))
    let count = src.rows * src.cols
    if count == 0 { return [0 for _index in 0..256] }
    #: Two hundred and fifty-six groups of as many threads, each striding the
    #: picture: enough to fill the device and few enough that the merge is
    #: cheap.
    let threads = 256 * 256
    kernel_for("threshold_histogram").dispatch_async(
        [src.storage, counts.storage],
        threads,
        [count, src.cols, src.offset, src.step_, threads],
        group_size: 256
    )
    var out: list[int] = []
    for value in counts.to_list() { out.push(int(value)) }
    return out
}

#: The level Otsu's method picks: the one that most separates the two classes
#: of pixel by between-class variance.
pub fn otsu_threshold(src: Mat) -> float {
    let counts = histogram_256(src)
    var total = 0
    var weighted = 0.0
    for (value, count) in counts.enumerate() {
        total += count
        weighted += float(value) * float(count)
    }
    if total == 0 { return 0.0 }
    var below = 0
    var below_weight = 0.0
    var best = 0.0
    var best_level = 0.0
    for (value, count) in counts.enumerate() {
        below += count
        if below == 0 { continue }
        let above = total - below
        if above == 0 { break }
        below_weight += float(value) * float(count)
        let mean_below = below_weight / float(below)
        let mean_above = (weighted - below_weight) / float(above)
        let gap = mean_below - mean_above
        let score = float(below) * float(above) * gap * gap
        if score > best {
            best = score
            best_level = float(value)
        }
    }
    return best_level
}

#: The level the triangle method picks: the histogram value furthest from the
#: line joining the peak to the far end.
pub fn triangle_threshold(src: Mat) -> float {
    let counts = histogram_256(src)
    var left = 0
    while left < 256 and counts[left] == 0 { left += 1 }
    var right = 255
    while right > 0 and counts[right] == 0 { right -= 1 }
    if left >= right { return float(left) }
    var peak = left
    for index in left..right + 1 {
        if counts[index] > counts[peak] { peak = index }
    }
    #: The method assumes the long tail is to the right of the peak; when it is
    #: not, OpenCV mirrors the histogram, finds the level, and mirrors back.
    var flipped = false
    var values = [float(counts[index]) for index in 0..256]
    if peak - left < right - peak {
        flipped = true
        var mirrored: list[float] = []
        for index in 0..256 { mirrored.push(values[255 - index]) }
        values = mirrored
        let new_left = 255 - right
        let new_right = 255 - left
        left = new_left
        right = new_right
        peak = 255 - peak
    }
    let height = values[peak]
    let span = float(right - peak)
    #: Distance from the chord joining the peak to the last non-empty bin,
    #: which is what the method maximises.
    var best = -1.0
    var best_level = peak
    for index in peak..right + 1 {
        let numerator = math.fabs((values[index] - height) * span + height * float(index - peak))
        let score = numerator / math.sqrt(span * span + height * height)
        if score > best {
            best = score
            best_level = index
        }
    }
    if flipped { return float(255 - best_level) }
    return float(best_level)
}

#: Apply a threshold, returning the level used and the result.
#:
#: With `THRESH_OTSU` or `THRESH_TRIANGLE` in the mode, the level is chosen
#: from the image and the `thresh` argument is ignored.
pub fn threshold(src: Mat, thresh: float, max_value: float, mode: int) -> tuple[float, Mat] {
    src.flush()
    var level = thresh
    if mode & THRESH_OTSU != 0 {
        level = otsu_threshold(src)
    } elif mode & THRESH_TRIANGLE != 0 {
        level = triangle_threshold(src)
    }
    let plain = mode & 7
    let out = Mat(src.rows, src.cols, src.flags)
    let count = src.rows * src.cols * src.channels()
    if count == 0 { return (level, out) }
    var scalars = [
        count,
        src.cols,
        src.channels(),
        src.offset,
        src.step_,
        out.offset,
        out.step_,
        plain,
        f32_bits(level),
        f32_bits(max_value),
    ]
    for value in saturation_args(out.flags) { scalars.push(value) }
    kernel_for("threshold_fixed").dispatch_async([src.storage, out.storage], count, scalars)
    out.invalidate()
    return (level, out)
}

#: Threshold each pixel against a level computed from its own neighbourhood.
pub fn adaptive_threshold(
    src: Mat,
    max_value: float,
    adaptive_method: int,
    threshold_type: int,
    block_size: int,
    c: float
) -> Mat {
    src.flush()
    if src.channels() != 1 { throw ValueError("adaptive_threshold needs a single-channel Mat") }
    if block_size <= 1 or block_size % 2 == 0 {
        throw ValueError(f"adaptive_threshold needs an odd block size above one, got {block_size}")
    }
    let wide = src.convert_to(CV_MAKETYPE(CV_32F, 1))
    let local = if adaptive_method == ADAPTIVE_THRESH_GAUSSIAN_C {
        gaussian_blur(wide, Size(block_size, block_size), 0.0, border_mode: BORDER_REPLICATE)
    } else {
        blur(wide, Size(block_size, block_size), border_mode: BORDER_REPLICATE)
    }
    let out = Mat(src.rows, src.cols, src.flags)
    let count = src.rows * src.cols
    if count == 0 { return out }
    var scalars = [
        count,
        src.cols,
        src.offset,
        src.step_,
        local.offset,
        local.step_,
        out.offset,
        out.step_,
        threshold_type & 7,
        f32_bits(c),
        f32_bits(max_value),
    ]
    for value in saturation_args(out.flags) { scalars.push(value) }
    kernel_for("threshold_adaptive").dispatch_async(
        [src.storage, local.storage, out.storage],
        count,
        scalars
    )
    out.invalidate()
    return out
}
