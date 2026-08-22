#: Histograms: counting them, comparing them, projecting them back onto an
#: image, and the two equalisations built on them.
#:
#: `calc_hist` takes the same shape of arguments OpenCV's does — a list of
#: images, the channels to read from each, the bin counts, and the value range
#: per dimension — so a two-dimensional hue-saturation histogram is written the
#: same way here as there.
#:
#: The counting itself runs on the device for the common case of one
#: eight-bit channel over the full range; anything else is counted on the host,
#: where the indexing arithmetic is cheaper to express than to dispatch.
#:
#: ```
#: let hist = calc_hist([gray], [0], null, [256], [0.0, 256.0])
#: let flat = equalize_hist(gray)
#: let local = CLAHE(2.0, Size(8, 8)).apply(gray)
#: ```
import std.math
from std.gpu import Buffer, Kernel
from ..core.arith import lut
from ..core.mat import Mat, f32_bits
from ..core.types import CV_32F, CV_8U, CV_MAKETYPE, Size
from .threshold import histogram_256

#: Compare histograms by their correlation, where more is more alike.
pub const HISTCMP_CORREL = 0
#: Compare by the chi-squared statistic, where less is more alike.
pub const HISTCMP_CHISQR = 1
#: Compare by how much they overlap, where more is more alike.
pub const HISTCMP_INTERSECT = 2
#: Compare by the Bhattacharyya distance, where less is more alike.
pub const HISTCMP_BHATTACHARYYA = 3
#: The symmetric form of the chi-squared statistic.
pub const HISTCMP_CHISQR_ALT = 4
#: Compare by the Kullback-Leibler divergence.
pub const HISTCMP_KL_DIV = 5

fn bin_of(value: float, low: float, high: float, bins: int) -> int {
    if high <= low { return -1 }
    if value < low or value >= high { return -1 }
    let scaled = (value - low) * float(bins) / (high - low)
    let index = int(math.floor(scaled))
    if index < 0 or index >= bins { return -1 }
    return index
}

fn plain_8u(
    images: list[Mat],
    channels: list[int],
    bin_counts: list[int],
    ranges: list[float]
) -> bool {
    return images.len() == 1 and
        channels.len() == 1 and
        channels[0] == 0 and
        images[0].channels() == 1 and
        images[0].depth() == CV_8U and
        bin_counts.len() == 1 and
        bin_counts[0] == 256 and
        ranges[0] == 0.0 and
        ranges[1] == 256.0
}

#: Count how often each combination of values occurs.
#:
#: `channels` names one channel per dimension, indexed across `images` as if
#: they were laid side by side, `bin_counts` gives the bins per dimension, and
#: `ranges` gives a low and a high for each. `mask`, if given, restricts the
#: count to where it is non-zero.
#:
#: The result is a `CV_32F` matrix with the first dimension as its rows and, if
#: there are two, the second as its columns.
pub fn calc_hist(
    images: list[Mat],
    channels: list[int],
    mask: Mat?,
    bin_counts: list[int],
    ranges: list[float],
    accumulate: Mat? = null
) -> Mat {
    let dimensions = channels.len()
    if dimensions == 0 { throw ValueError("a histogram needs at least one channel") }
    if bin_counts.len() != dimensions {
        throw ValueError("there has to be one bin count per channel")
    }
    if ranges.len() != dimensions * 2 {
        throw ValueError("there has to be a low and a high per channel")
    }
    for image in images { image.flush() }

    var total = 1
    for count in bin_counts { total *= count }
    var counts: list[float] = []
    if accumulate is not null {
        counts = accumulate.to_list()
        if counts.len() != total { throw ValueError("the histogram to add to is the wrong size") }
    } else {
        for _index in 0..total { counts.push(0.0) }
    }

    #: The common case — one eight-bit channel over the whole range — is
    #: already a device kernel, so it does not come back to the host at all.
    if mask is null and accumulate is null and plain_8u(images, channels, bin_counts, ranges) {
        var from_device: list[float] = []
        for value in histogram_256(images[0]) { from_device.push(float(value)) }
        return Mat.from_list(256, 1, CV_MAKETYPE(CV_32F, 1), from_device)
    }

    #: One channel index runs across every image, as if their channels were
    #: laid end to end, which is how OpenCV reads the `channels` argument.
    var image_of: list[int] = []
    var channel_of: list[int] = []
    for wanted in channels {
        var remaining = wanted
        var found = false
        for index in 0..images.len() {
            let width = images[index].channels()
            if remaining < width {
                image_of.push(index)
                channel_of.push(remaining)
                found = true
                break
            }
            remaining -= width
        }
        if not found { throw ValueError(f"channel {wanted} is beyond the images given") }
    }

    let rows = images[0].rows
    let cols = images[0].cols
    for image in images {
        if image.rows != rows or image.cols != cols {
            throw ValueError("every image in a histogram has to be the same size")
        }
    }
    let mask_image = mask
    if mask_image is not null { mask_image.flush() }

    #: Counted on the device when the bins fit in a threadgroup, which is what
    #: `hist_count` stages them in. Anything bigger, or more than two channels,
    #: keeps to the walk below.
    let out_rows = bin_counts[0]
    let out_cols = if dimensions > 1 { total // bin_counts[0] } else { 1 }
    if dimensions <= 2 and total <= 1024 {
        let tally = Mat.from_list(out_rows, out_cols, CV_MAKETYPE(CV_32F, 1), counts)
        let a = images[image_of[0]]
        let b = images[image_of[dimensions - 1]]
        let guard = if mask_image is not null { mask_image } else { a }
        let threads = 256 * 256
        kernel_for("hist_count").dispatch_async([a.storage, b.storage, guard.storage, tally.storage], threads, [rows * cols, cols, dimensions, total, threads, if mask_image is null {
            0
        } else {
            1
        }, a.offset + channel_of[0], a.step_, a.channels(), b.offset + channel_of[dimensions - 1], b.step_, b.channels(), guard.offset, guard.step_, guard.channels(), bin_counts[0], if dimensions > 1 {
            bin_counts[1]
        } else {
            1
        }, f32_bits(ranges[0]), f32_bits(ranges[1]), f32_bits(if dimensions > 1 {
            ranges[2]
        } else {
            0.0
        }), f32_bits(if dimensions > 1 {
            ranges[3]
        } else {
            1.0
        })], group_size: 256)
        tally.invalidate()
        return tally
    }

    var planes: list[list[float]] = []
    for image in images { planes.push(image.values()) }
    var mask_values: list[float] = []
    if mask_image is not null { mask_values = mask_image.to_list() }

    for cell in 0..rows * cols {
        if mask_image is not null and mask_values[cell] == 0.0 { continue }
        var offset = 0
        var inside = true
        for axis in 0..dimensions {
            let source = images[image_of[axis]]
            let value = planes[image_of[axis]][cell * source.channels() + channel_of[axis]]
            let index = bin_of(value, ranges[axis * 2], ranges[axis * 2 + 1], bin_counts[axis])
            if index < 0 {
                inside = false
                break
            }
            offset = offset * bin_counts[axis] + index
        }
        if inside { counts[offset] += 1.0 }
    }
    return Mat.from_list(out_rows, out_cols, CV_MAKETYPE(CV_32F, 1), counts)
}

#: How alike two histograms are.
pub fn compare_hist(first: Mat, second: Mat, method: int = HISTCMP_CORREL) -> float {
    let a = first.to_list()
    let b = second.to_list()
    if a.len() != b.len() { throw ValueError("two histograms have to be the same size") }
    let count = a.len()
    if count == 0 { return 0.0 }

    if method == HISTCMP_CORREL {
        var mean_a = 0.0
        var mean_b = 0.0
        for index in 0..count {
            mean_a += a[index]
            mean_b += b[index]
        }
        mean_a /= float(count)
        mean_b /= float(count)
        var top = 0.0
        var left = 0.0
        var right = 0.0
        for index in 0..count {
            let da = a[index] - mean_a
            let db = b[index] - mean_b
            top += da * db
            left += da * da
            right += db * db
        }
        let below = math.sqrt(left * right)
        if below < 1e-20 { return 1.0 }
        return top / below
    }
    if method == HISTCMP_CHISQR {
        var total = 0.0
        for index in 0..count {
            if a[index] == 0.0 { continue }
            let gap = a[index] - b[index]
            total += gap * gap / a[index]
        }
        return total
    }
    if method == HISTCMP_CHISQR_ALT {
        var total = 0.0
        for index in 0..count {
            let sum = a[index] + b[index]
            if sum == 0.0 { continue }
            let gap = a[index] - b[index]
            total += gap * gap / sum
        }
        return total * 2.0
    }
    if method == HISTCMP_INTERSECT {
        var total = 0.0
        for index in 0..count { total += min(a[index], b[index]) }
        return total
    }
    if method == HISTCMP_BHATTACHARYYA {
        var joint = 0.0
        var sum_a = 0.0
        var sum_b = 0.0
        for index in 0..count {
            joint += math.sqrt(a[index] * b[index])
            sum_a += a[index]
            sum_b += b[index]
        }
        let product = sum_a * sum_b
        let scale = if math.fabs(product) > 1e-20 { 1.0 / math.sqrt(product) } else { 1.0 }
        return math.sqrt(max(1.0 - joint * scale, 0.0))
    }
    if method == HISTCMP_KL_DIV {
        var total = 0.0
        for index in 0..count {
            if a[index] == 0.0 { continue }
            let other = if b[index] == 0.0 { 1e-10 } else { b[index] }
            total += a[index] * math.log(a[index] / other)
        }
        return total
    }
    throw ValueError(f"histogram comparison {method} is not one jaicv implements")
}

#: Paint each pixel with how common its value is, according to a histogram.
#:
#: This is what turns a colour histogram taken from an object into a map of
#: where that object might be, which is how mean shift tracking is seeded.
pub fn calc_back_project(
    images: list[Mat],
    channels: list[int],
    hist: Mat,
    ranges: list[float],
    scale: float = 1.0
) -> Mat {
    let dimensions = channels.len()
    if ranges.len() != dimensions * 2 {
        throw ValueError("there has to be a low and a high per channel")
    }
    for image in images { image.flush() }

    var bin_counts: list[int] = []
    if dimensions == 1 {
        bin_counts.push(hist.rows * hist.cols)
    } else {
        bin_counts.push(hist.rows)
        bin_counts.push(hist.cols)
        for _axis in 2..dimensions {
            throw ValueError("back projection handles one or two channels")
        }
    }

    var image_of: list[int] = []
    var channel_of: list[int] = []
    for wanted in channels {
        var remaining = wanted
        var found = false
        for index in 0..images.len() {
            let width = images[index].channels()
            if remaining < width {
                image_of.push(index)
                channel_of.push(remaining)
                found = true
                break
            }
            remaining -= width
        }
        if not found { throw ValueError(f"channel {wanted} is beyond the images given") }
    }

    let rows = images[0].rows
    let cols = images[0].cols
    hist.flush()

    let out = Mat(rows, cols, CV_MAKETYPE(CV_8U, 1))
    let a = images[image_of[0]]
    let b = images[image_of[dimensions - 1]]
    kernel_for("hist_back_project").dispatch_async([a.storage, b.storage, hist.storage, out.storage], rows * cols, [rows * cols, cols, dimensions, f32_bits(scale), a.offset + channel_of[0], a.step_, a.channels(), b.offset + channel_of[dimensions - 1], b.step_, b.channels(), bin_counts[0], if dimensions > 1 {
        bin_counts[1]
    } else {
        1
    }, f32_bits(ranges[0]), f32_bits(ranges[1]), f32_bits(if dimensions > 1 {
        ranges[2]
    } else {
        0.0
    }), f32_bits(if dimensions > 1 {
        ranges[3]
    } else {
        1.0
    }), out.offset, out.step_])
    out.invalidate()
    return out
}

#: Spread an image's values evenly across the range.
#:
#: The lookup table is built from the running total of the histogram, skipping
#: the darkest occupied level so that the darkest pixel maps to zero, which is
#: the detail that makes this agree with OpenCV rather than merely resemble it.
pub fn equalize_hist(image: Mat) -> Mat {
    if image.channels() != 1 { throw ValueError("equalize_hist needs a single-channel Mat") }
    image.flush()
    let counts = histogram_256(image)
    let total = image.rows * image.cols
    if total == 0 { return image.clone() }

    var first = 0
    while first < 256 and counts[first] == 0 { first += 1 }
    if first == 256 { return image.clone() }
    if counts[first] == total {
        return Mat.full(image.rows, image.cols, image.type_(), float(first))
    }

    let scale = 255.0 / float(total - counts[first])
    var table: list[float] = []
    for _index in 0..first + 1 { table.push(0.0) }
    var running = 0
    for value in first + 1..256 {
        running += counts[value]
        var mapped = math.floor(float(running) * scale + 0.5)
        if mapped < 0.0 { mapped = 0.0 }
        if mapped > 255.0 { mapped = 255.0 }
        table.push(mapped)
    }

    #: The table is 256 entries and the picture is millions, so the mapping
    #: goes through the lookup kernel. Reading every pixel back to map it on
    #: the host, then handing the result to `Mat.from_list`, was eleven of the
    #: twelve milliseconds this used to take at 1080p.
    let mapping = Mat.from_list(1, 256, CV_MAKETYPE(image.depth(), 1), table)
    let out = lut(image, mapping)
    mapping.release()
    return out
}

let SOURCE = """#include <metal_stdlib>
using namespace metal;

/* Which bin a value falls in, or -1 when it falls outside the range.
 * The same rule the host `bin_of` follows. */
inline int hist_bin(float value, float low, float high, uint bins) {
    if (high <= low) return -1;
    if (value < low || value >= high) return -1;
    int index = int(floor((value - low) * float(bins) / (high - low)));
    if (index < 0 || index >= int(bins)) return -1;
    return index;
}

/* One or two channels counted into a histogram, a pixel to a thread.
 *
 * Counted into threadgroup bins and merged once, as in `threshold_histogram`:
 * a picture's worth of device atomics over a few hundred slots is what makes
 * the naive version slow, quite apart from it having been on the host. Only
 * for a histogram small enough to stage; a bigger one keeps to the host.
 *
 * Each plane is addressed as `base + y * step + x * cn`, with the channel
 * already folded into `base` by the caller. */
kernel void hist_count(device const float* first [[buffer(0)]],
                       device const float* second [[buffer(1)]],
                       device const float* mask [[buffer(2)]],
                       device float* counts [[buffer(3)]],
                       constant uint& count [[buffer(4)]],
                       constant uint& cols [[buffer(5)]],
                       constant uint& dims [[buffer(6)]],
                       constant uint& total_bins [[buffer(7)]],
                       constant uint& stride [[buffer(8)]],
                       constant uint& use_mask [[buffer(9)]],
                       constant uint& a_base [[buffer(10)]],
                       constant uint& a_step [[buffer(11)]],
                       constant uint& a_cn [[buffer(12)]],
                       constant uint& b_base [[buffer(13)]],
                       constant uint& b_step [[buffer(14)]],
                       constant uint& b_cn [[buffer(15)]],
                       constant uint& m_base [[buffer(16)]],
                       constant uint& m_step [[buffer(17)]],
                       constant uint& m_cn [[buffer(18)]],
                       constant uint& bins_a [[buffer(19)]],
                       constant uint& bins_b [[buffer(20)]],
                       constant uint& low_a [[buffer(21)]],
                       constant uint& high_a [[buffer(22)]],
                       constant uint& low_b [[buffer(23)]],
                       constant uint& high_b [[buffer(24)]],
                       uint id [[thread_position_in_grid]],
                       uint lid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint bins[1024];
    for (uint slot = lid; slot < total_bins; slot += 256u) {
        atomic_store_explicit(&bins[slot], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint at = id; at < count; at += stride) {
        uint y = at / cols;
        uint x = at - y * cols;
        if (use_mask != 0u && mask[m_base + y * m_step + x * m_cn] == 0.0f) continue;
        float value = first[a_base + y * a_step + x * a_cn];
        int index = hist_bin(value, as_type<float>(low_a), as_type<float>(high_a), bins_a);
        if (index < 0) continue;
        uint offset = uint(index);
        if (dims > 1u) {
            float other = second[b_base + y * b_step + x * b_cn];
            int next = hist_bin(other, as_type<float>(low_b), as_type<float>(high_b), bins_b);
            if (next < 0) continue;
            offset = offset * bins_b + uint(next);
        }
        atomic_fetch_add_explicit(&bins[offset], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint slot = lid; slot < total_bins; slot += 256u) {
        uint mine = atomic_load_explicit(&bins[slot], memory_order_relaxed);
        if (mine == 0u) continue;
        atomic_fetch_add_explicit((device atomic<float>*)(counts + slot), float(mine),
                                  memory_order_relaxed);
    }
}

/* The histogram read back onto the picture, a pixel to a thread. No atomics
 * and no size limit: every thread only reads. */
kernel void hist_back_project(device const float* first [[buffer(0)]],
                              device const float* second [[buffer(1)]],
                              device const float* counts [[buffer(2)]],
                              device float* dst [[buffer(3)]],
                              constant uint& count [[buffer(4)]],
                              constant uint& cols [[buffer(5)]],
                              constant uint& dims [[buffer(6)]],
                              constant uint& scale_bits [[buffer(7)]],
                              constant uint& a_base [[buffer(8)]],
                              constant uint& a_step [[buffer(9)]],
                              constant uint& a_cn [[buffer(10)]],
                              constant uint& b_base [[buffer(11)]],
                              constant uint& b_step [[buffer(12)]],
                              constant uint& b_cn [[buffer(13)]],
                              constant uint& bins_a [[buffer(14)]],
                              constant uint& bins_b [[buffer(15)]],
                              constant uint& low_a [[buffer(16)]],
                              constant uint& high_a [[buffer(17)]],
                              constant uint& low_b [[buffer(18)]],
                              constant uint& high_b [[buffer(19)]],
                              constant uint& out_base [[buffer(20)]],
                              constant uint& out_step [[buffer(21)]],
                              uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint y = id / cols;
    uint x = id - y * cols;
    uint where = out_base + y * out_step + x;

    float value = first[a_base + y * a_step + x * a_cn];
    int index = hist_bin(value, as_type<float>(low_a), as_type<float>(high_a), bins_a);
    if (index < 0) {
        dst[where] = 0.0f;
        return;
    }
    uint offset = uint(index);
    if (dims > 1u) {
        float other = second[b_base + y * b_step + x * b_cn];
        int next = hist_bin(other, as_type<float>(low_b), as_type<float>(high_b), bins_b);
        if (next < 0) {
            dst[where] = 0.0f;
            return;
        }
        offset = offset * bins_b + uint(next);
    }
    dst[where] = counts[offset] * as_type<float>(scale_bits);
}

/* One tile's histogram to a threadgroup.
 *
 * Every tile is independent and every pixel belongs to exactly one, so the
 * whole grid of them is a single dispatch with the counting done in
 * threadgroup memory. Walking the picture on the host to fill them took 560 ms
 * for a 640 by 480 frame.
 */
kernel void clahe_histograms(device const float* src [[buffer(0)]],
                             device float* counts [[buffer(1)]],
                             constant uint& cols [[buffer(2)]],
                             constant uint& src_off [[buffer(3)]],
                             constant uint& src_step [[buffer(4)]],
                             constant uint& tiles_x [[buffer(5)]],
                             constant uint& tile_w [[buffer(6)]],
                             constant uint& tile_h [[buffer(7)]],
                             uint tile [[threadgroup_position_in_grid]],
                             uint lid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint bins[256];
    atomic_store_explicit(&bins[lid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint tx = tile % tiles_x;
    uint ty = tile / tiles_x;
    uint area = tile_w * tile_h;
    for (uint at = lid; at < area; at += 256u) {
        uint y = ty * tile_h + at / tile_w;
        uint x = tx * tile_w + at - (at / tile_w) * tile_w;
        float value = src[src_off + y * src_step + x];
        uint slot = uint(clamp(value, 0.0f, 255.0f));
        atomic_fetch_add_explicit(&bins[slot], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    counts[tile * 256u + lid] = float(atomic_load_explicit(&bins[lid], memory_order_relaxed));
}

/* Clip each tile's histogram and turn it into a lookup table.
 *
 * A tile to a threadgroup, and one thread walks its 256 bins: the clipping and
 * the running total are sequential in the bins, and 256 steps for sixty-four
 * tiles at once is nothing next to reading them back to do it on the host --
 * which cost a download, sixteen thousand iterations of Jaithon, and an upload.
 *
 * The redistribution matches OpenCV's: what was clipped goes back evenly, and
 * the remainder is spread at a stride rather than piled onto the first bins. */
kernel void clahe_tables(device float* counts [[buffer(0)]],
                         device float* tables [[buffer(1)]],
                         constant uint& limit [[buffer(2)]],
                         constant uint& scale_bits [[buffer(3)]],
                         uint tile [[threadgroup_position_in_grid]],
                         uint lid [[thread_index_in_threadgroup]]) {
    threadgroup uint bins[256];
    bins[lid] = uint(counts[tile * 256u + lid] + 0.5f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid != 0u) return;

    if (limit > 0u) {
        uint clipped = 0u;
        for (uint index = 0u; index < 256u; ++index) {
            if (bins[index] > limit) {
                clipped += bins[index] - limit;
                bins[index] = limit;
            }
        }
        uint batch = clipped / 256u;
        uint residual = clipped - batch * 256u;
        for (uint index = 0u; index < 256u; ++index) bins[index] += batch;
        if (residual > 0u) {
            uint stride = max(256u / residual, 1u);
            uint index = 0u;
            while (index < 256u && residual > 0u) {
                bins[index] += 1u;
                residual -= 1u;
                index += stride;
            }
        }
    }

    float scale = as_type<float>(scale_bits);
    uint running = 0u;
    for (uint index = 0u; index < 256u; ++index) {
        running += bins[index];
        tables[tile * 256u + index] = clamp(floor(float(running) * scale + 0.5f), 0.0f, 255.0f);
    }
}

/* Each pixel blended from the four tile tables nearest it.
 *
 * The tables are worked out on the host -- sixty-four tiles of 256 entries is
 * nothing -- and only the blend, which is per pixel, comes here. */
kernel void clahe_blend(device const float* src [[buffer(0)]],
                        device const float* tables [[buffer(1)]],
                        device float* dst [[buffer(2)]],
                        constant uint& count [[buffer(3)]],
                        constant uint& rows [[buffer(4)]],
                        constant uint& cols [[buffer(5)]],
                        constant uint& src_off [[buffer(6)]],
                        constant uint& src_step [[buffer(7)]],
                        constant uint& dst_off [[buffer(8)]],
                        constant uint& dst_step [[buffer(9)]],
                        constant uint& tiles_x [[buffer(10)]],
                        constant uint& tiles_y [[buffer(11)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint y = id / cols;
    uint x = id - y * cols;
    float ty = float(y) * float(tiles_y) / float(rows) - 0.5f;
    float tx = float(x) * float(tiles_x) / float(cols) - 0.5f;
    int ty1 = int(floor(ty));
    int tx1 = int(floor(tx));
    float ya = ty - float(ty1);
    float xa = tx - float(tx1);
    int row1 = clamp(ty1, 0, int(tiles_y) - 1);
    int row2 = clamp(ty1 + 1, 0, int(tiles_y) - 1);
    int column1 = clamp(tx1, 0, int(tiles_x) - 1);
    int column2 = clamp(tx1 + 1, 0, int(tiles_x) - 1);

    uint value = uint(clamp(src[src_off + y * src_step + x], 0.0f, 255.0f));
    float blended =
        tables[(uint(row1) * tiles_x + uint(column1)) * 256u + value] * (1.0f - xa) * (1.0f - ya) +
        tables[(uint(row1) * tiles_x + uint(column2)) * 256u + value] * xa * (1.0f - ya) +
        tables[(uint(row2) * tiles_x + uint(column1)) * 256u + value] * (1.0f - xa) * ya +
        tables[(uint(row2) * tiles_x + uint(column2)) * 256u + value] * xa * ya;
    dst[dst_off + y * dst_step + x] = clamp(floor(blended + 0.5f), 0.0f, 255.0f);
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

#: Contrast-limited adaptive histogram equalisation.
#:
#: The image is cut into tiles, each tile equalised on its own, and the four
#: nearest tile mappings blended across every pixel so the tiling does not
#: show. `clip_limit` caps how much any one level may be stretched, which is
#: what stops flat regions turning into noise; the pixels clipped off are
#: handed back out evenly across the range.
pub class CLAHE {
    pub var clip_limit: float
    pub var tile_grid_size: Size

    pub fn init(self, clip_limit: float = 40.0, tile_grid_size: Size = Size(8, 8)) {
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
    }

    #: Change the cap without building a new one.
    pub fn set_clip_limit(self, limit: float) -> void { self.clip_limit = limit }

    #: Change the tiling without building a new one.
    pub fn set_tiles_grid_size(self, size: Size) -> void { self.tile_grid_size = size }

    #: Equalise `image`, which has to be single-channel and eight-bit.
    pub fn apply(self, image: Mat) -> Mat {
        if image.channels() != 1 { throw ValueError("CLAHE needs a single-channel Mat") }
        image.flush()
        let rows = image.rows
        let cols = image.cols
        let tiles_x = max(self.tile_grid_size.width, 1)
        let tiles_y = max(self.tile_grid_size.height, 1)
        let tile_w = cols // tiles_x
        let tile_h = rows // tiles_y
        if tile_w == 0 or tile_h == 0 { return image.clone() }
        let area = tile_w * tile_h

        var limit = 0
        if self.clip_limit > 0.0 {
            limit = int(self.clip_limit * float(area) / 256.0)
            if limit < 1 { limit = 1 }
        }
        let lut_scale = 255.0 / float(area)

        #: Every tile's histogram in one dispatch; the clipping and the running
        #: total that follow are sixty-four tiles of 256 numbers, which is
        #: nothing, and stay here.
        let tally = Buffer(tiles_x * tiles_y * 256)
        kernel_for("clahe_histograms").dispatch_async(
            [image.storage, tally],
            tiles_x * tiles_y * 256,
            [cols, image.offset, image.step_, tiles_x, tile_w, tile_h],
            group_size: 256
        )
        let tables = Buffer(tiles_x * tiles_y * 256)
        kernel_for("clahe_tables").dispatch_async(
            [tally, tables],
            tiles_x * tiles_y * 256,
            [limit, f32_bits(lut_scale)],
            group_size: 256
        )
        tally.free()

        #: The blend is per pixel and reads four tables. Nothing has come back
        #: to the host between the three passes, so there is no wait until the
        #: caller asks for the picture.
        let out = Mat(rows, cols, CV_MAKETYPE(CV_8U, 1))
        kernel_for("clahe_blend").dispatch_async(
            [image.storage, tables, out.storage],
            rows * cols,
            [
                rows * cols,
                rows,
                cols,
                image.offset,
                image.step_,
                out.offset,
                out.step_,
                tiles_x,
                tiles_y,
            ]
        )
        out.invalidate()
        tables.free()
        return out
    }
}

fn clamp_index(value: int, limit: int) -> int {
    if value < 0 { return 0 }
    if value >= limit { return limit - 1 }
    return value
}

#: Build a CLAHE with the given cap and tiling, the way OpenCV's factory reads.
pub fn create_clahe(clip_limit: float = 40.0, tile_grid_size: Size = Size(8, 8)) -> CLAHE {
    return CLAHE(clip_limit, tile_grid_size)
}
