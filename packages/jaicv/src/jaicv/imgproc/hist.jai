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
from ..core.arith import lut
from ..core.mat import Mat
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
    var planes: list[list[float]] = []
    for image in images { planes.push(image.values()) }
    var mask_values: list[float] = []
    let mask_image = mask
    if mask_image is not null {
        mask_image.flush()
        mask_values = mask_image.to_list()
    }

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

    let out_rows = bin_counts[0]
    let out_cols = if dimensions > 1 { total // bin_counts[0] } else { 1 }
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
    var planes: list[list[float]] = []
    for image in images { planes.push(image.values()) }
    let counts = hist.to_list()

    var out: list[float] = []
    for cell in 0..rows * cols {
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
        out.push(if inside { counts[offset] * scale } else { 0.0 })
    }
    return Mat.from_list(rows, cols, CV_MAKETYPE(CV_8U, 1), out)
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

        let source = image.values()
        var tables: list[list[float]] = []
        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                var counts: list[int] = []
                for _index in 0..256 { counts.push(0) }
                for y in ty * tile_h..(ty + 1) * tile_h {
                    for x in tx * tile_w..(tx + 1) * tile_w {
                        var value = int(source[y * cols + x])
                        if value < 0 { value = 0 }
                        if value > 255 { value = 255 }
                        counts[value] += 1
                    }
                }
                if limit > 0 {
                    var clipped = 0
                    for index in 0..256 {
                        if counts[index] > limit {
                            clipped += counts[index] - limit
                            counts[index] = limit
                        }
                    }
                    #: The clipped pixels go back evenly, and what is left over
                    #: is spread at a stride rather than piled on the first
                    #: bins, which is what OpenCV does.
                    let batch = clipped // 256
                    var residual = clipped - batch * 256
                    for index in 0..256 { counts[index] += batch }
                    if residual > 0 {
                        let stride = max(256 // residual, 1)
                        var index = 0
                        while index < 256 and residual > 0 {
                            counts[index] += 1
                            residual -= 1
                            index += stride
                        }
                    }
                }
                var table: list[float] = []
                var running = 0
                for index in 0..256 {
                    running += counts[index]
                    var mapped = math.floor(float(running) * lut_scale + 0.5)
                    if mapped < 0.0 { mapped = 0.0 }
                    if mapped > 255.0 { mapped = 255.0 }
                    table.push(mapped)
                }
                tables.push(table)
            }
        }

        var out: list[float] = []
        let inverse_w = float(tiles_x) / float(cols)
        let inverse_h = float(tiles_y) / float(rows)
        for y in 0..rows {
            let ty = float(y) * inverse_h - 0.5
            let ty1 = int(math.floor(ty))
            let ya = ty - float(ty1)
            let row1 = clamp_index(ty1, tiles_y)
            let row2 = clamp_index(ty1 + 1, tiles_y)
            for x in 0..cols {
                let tx = float(x) * inverse_w - 0.5
                let tx1 = int(math.floor(tx))
                let xa = tx - float(tx1)
                let column1 = clamp_index(tx1, tiles_x)
                let column2 = clamp_index(tx1 + 1, tiles_x)

                var value = int(source[y * cols + x])
                if value < 0 { value = 0 }
                if value > 255 { value = 255 }
                let blended = tables[row1 * tiles_x + column1][value] * (1.0 - xa) * (1.0 - ya) +
                    tables[row1 * tiles_x + column2][value] * xa * (1.0 - ya) +
                    tables[row2 * tiles_x + column1][value] * (1.0 - xa) * ya +
                    tables[row2 * tiles_x + column2][value] * xa * ya
                var rounded = math.floor(blended + 0.5)
                if rounded < 0.0 { rounded = 0.0 }
                if rounded > 255.0 { rounded = 255.0 }
                out.push(rounded)
            }
        }
        return Mat.from_list(rows, cols, CV_MAKETYPE(CV_8U, 1), out)
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
