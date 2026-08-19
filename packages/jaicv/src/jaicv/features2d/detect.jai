#: Keypoints and descriptors: FAST, ORB, BRIEF, blob detection, and the
#: brute-force matcher that pairs them up.
#:
#: The detectors here are OpenCV's, and `fast` reproduces its corner test and
#: its scores exactly, down to the order the keypoints come out in. The
#: descriptors are a different matter and worth being plain about: OpenCV's
#: BRIEF and ORB sample the patch at 256 point pairs that were learned offline
#: and shipped as a table, and that table is data rather than algorithm. This
#: builds an equivalent pattern from a fixed seed instead, so descriptors here
#: match each other run to run and match against each other correctly, but a
#: descriptor computed here will not compare against one computed by OpenCV.
#:
#: ```
#: let orb = ORB(500)
#: let (keypoints, descriptors) = orb.detect_and_compute(gray)
#: let matches = BFMatcher(NORM_HAMMING, true).match_(first, second)
#: ```
import std.math
from std.core import Printable
from ..core.mat import Mat
from ..core.types import CV_32F, CV_8U, CV_MAKETYPE, NORM_HAMMING, NORM_L1, NORM_L2
from ..core.types import Point, Point2f, Rect, Scalar, Size
from ..imgproc.drawing import circle, line
from ..imgproc.filter import gaussian_blur
from ..imgproc.geometric import resize
from ..imgproc.contours import find_contours, RETR_LIST, CHAIN_APPROX_NONE
from ..imgproc.shape import contour_area, moments
from ..imgproc.threshold import threshold, THRESH_BINARY

#: Nine of the sixteen ring pixels have to agree, which is the usual FAST.
pub const FAST_FEATURE_DETECTOR_TYPE_9_16 = 2
#: Seven of twelve, on a smaller ring.
pub const FAST_FEATURE_DETECTOR_TYPE_7_12 = 1
#: Five of eight, on the immediate neighbours.
pub const FAST_FEATURE_DETECTOR_TYPE_5_8 = 0

#: Rank ORB's keypoints by Harris's response.
pub const ORB_HARRIS_SCORE = 0
#: Rank them by their FAST score instead.
pub const ORB_FAST_SCORE = 1

#: Draw keypoints as plain dots.
pub const DRAW_MATCHES_FLAGS_DEFAULT = 0
#: Draw each keypoint at its own size and angle.
pub const DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS = 4

#: One detected point, with everything a descriptor needs to describe it.
pub class KeyPoint: Printable {
    #: Where it is, to sub-pixel precision.
    pub var pt: Point2f
    #: The diameter of the region it covers.
    pub var size: float
    #: Which way it faces, in degrees, or `-1` if it has none.
    pub var angle: float
    #: How strong it is; larger is stronger.
    pub var response: float
    #: Which pyramid level it was found on.
    pub var octave: int
    #: Free for the caller to group keypoints with.
    pub var class_id: int

    pub fn init(
        self,
        pt: Point2f,
        size: float = 7.0,
        angle: float = -1.0,
        response: float = 0.0,
        octave: int = 0,
        class_id: int = -1
    ) {
        self.pt = pt
        self.size = size
        self.angle = angle
        self.response = response
        self.octave = octave
        self.class_id = class_id
    }

    pub fn to_str(self) -> str {
        return f"KeyPoint({self.pt}, size={self.size}, response={self.response})"
    }

    fn __str__(self) -> str { return self.to_str() }
}

#: A pairing between one descriptor in each of two sets.
pub class DMatch: Printable {
    #: Index into the first set.
    pub var query_idx: int
    #: Index into the second set.
    pub var train_idx: int
    #: Which image of the second set, when there is more than one.
    pub var img_idx: int
    #: How far apart the two descriptors are.
    pub var distance: float

    pub fn init(self, query_idx: int, train_idx: int, distance: float, img_idx: int = 0) {
        self.query_idx = query_idx
        self.train_idx = train_idx
        self.img_idx = img_idx
        self.distance = distance
    }

    pub fn to_str(self) -> str {
        return f"DMatch({self.query_idx} -> {self.train_idx}, distance={self.distance})"
    }

    fn __str__(self) -> str { return self.to_str() }
}

#: The sixteen pixels of the Bresenham circle FAST reads, in the order OpenCV
#: walks them: clockwise from straight up.
let RING_X = [0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1]
let RING_Y = [3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3]

fn corner_score(differences: list[int], threshold: int) -> int {
    #: The score is the largest threshold at which the point would still be a
    #: corner, found by pushing the bright and dark cases as far as each will
    #: go. This is OpenCV's `cornerScore`, and its shape — the early `continue`
    #: before the second half of each run — is what keeps it linear.
    var brightest = threshold
    var k = 0
    while k < 16 {
        var a = min(differences[k + 1], differences[k + 2])
        a = min(a, differences[k + 3])
        if a > brightest {
            a = min(a, differences[k + 4])
            a = min(a, differences[k + 5])
            a = min(a, differences[k + 6])
            a = min(a, differences[k + 7])
            a = min(a, differences[k + 8])
            brightest = max(brightest, min(a, differences[k]))
            brightest = max(brightest, min(a, differences[k + 9]))
        }
        k += 2
    }

    var darkest = -brightest
    k = 0
    while k < 16 {
        var b = max(differences[k + 1], differences[k + 2])
        b = max(b, differences[k + 3])
        b = max(b, differences[k + 4])
        b = max(b, differences[k + 5])
        if b < darkest {
            b = max(b, differences[k + 6])
            b = max(b, differences[k + 7])
            b = max(b, differences[k + 8])
            darkest = min(darkest, max(b, differences[k]))
            darkest = min(darkest, max(b, differences[k + 9]))
        }
        k += 2
    }
    return -darkest - 1
}

fn is_corner(differences: list[int], threshold: int) -> bool {
    #: Nine in a row all brighter, or nine in a row all darker. The ring is
    #: walked twice over so a run that wraps is still nine consecutive entries.
    var run = 0
    for index in 0..25 {
        if differences[index] > threshold {
            run += 1
            if run >= 9 { return true }
        } else {
            run = 0
        }
    }
    run = 0
    for index in 0..25 {
        if differences[index] < -threshold {
            run += 1
            if run >= 9 { return true }
        } else {
            run = 0
        }
    }
    return false
}

#: Find corners by the FAST test.
#:
#: A pixel is a corner when nine of the sixteen pixels on a circle of radius
#: three around it are all brighter, or all darker, than it by more than
#: `threshold`. With `nonmax_suppression` on, a corner also has to beat all
#: eight of its neighbours' scores, which thins a cluster down to its peak.
pub fn fast(image: Mat, threshold: int = 10, nonmax_suppression: bool = true) -> list[KeyPoint] {
    if image.channels() != 1 { throw ValueError("fast needs a single-channel Mat") }
    image.flush()
    let rows = image.rows
    let cols = image.cols
    let values = image.values()
    var scores: list[int] = []
    for _index in 0..rows * cols { scores.push(0) }

    var differences: list[int] = []
    for _index in 0..25 { differences.push(0) }

    for y in 3..rows - 3 {
        for x in 3..cols - 3 {
            let centre = int(values[y * cols + x])
            for index in 0..16 {
                differences[index] = centre -
                    int(values[(y + RING_Y[index]) * cols + x + RING_X[index]])
            }
            for index in 16..25 { differences[index] = differences[index - 16] }
            if not is_corner(differences, threshold) { continue }
            #: Without suppression the score is never looked at, and OpenCV
            #: does not compute it either — it reports a response of zero.
            scores[y * cols + x] = if nonmax_suppression {
                corner_score(differences, threshold)
            } else {
                1
            }
        }
    }

    var out: list[KeyPoint] = []
    for y in 3..rows - 3 {
        for x in 3..cols - 3 {
            let score = scores[y * cols + x]
            if score == 0 { continue }
            if nonmax_suppression {
                var peak = true
                for dy in -1..2 {
                    for dx in -1..2 {
                        if dy == 0 and dx == 0 { continue }
                        if scores[(y + dy) * cols + x + dx] >= score {
                            peak = false
                            break
                        }
                    }
                    if not peak { break }
                }
                if not peak { continue }
            }
            let response = if nonmax_suppression { float(score) } else { 0.0 }
            out.push(KeyPoint(Point2f(float(x), float(y)), 7.0, -1.0, response))
        }
    }
    return out
}

fn xorshift(state: int) -> int {
    #: A small deterministic generator, so the sampling pattern is the same on
    #: every machine and every run without carrying a table around.
    var x = state
    x = (x ^ x << 13) & 0xFFFFFFFFFFFF
    x = x ^ x >> 7
    x = (x ^ x << 17) & 0xFFFFFFFFFFFF
    return x
}

fn sampling_pattern(bytes_wide: int, patch: int) -> list[int] {
    #: Four numbers per bit: the two points whose brightnesses are compared.
    #: They are drawn from a rough Gaussian about the patch centre, which is
    #: the distribution the BRIEF paper found worked best.
    let bits = bytes_wide * 8
    var out: list[int] = []
    var state = 0x2545F4914F6C
    let spread = float(patch) / 5.0
    for _index in 0..bits * 4 {
        var total = 0.0
        for _draw in 0..4 {
            state = xorshift(state)
            total += float(state % 2048) / 2048.0 - 0.5
        }
        var value = int(math.floor(total * spread + 0.5))
        let limit = patch // 2
        if value > limit { value = limit }
        if value < -limit { value = -limit }
        out.push(value)
    }
    return out
}

fn sample(values: list[float], rows: int, cols: int, x: int, y: int) -> float {
    var cx = x
    var cy = y
    if cx < 0 { cx = 0 }
    if cy < 0 { cy = 0 }
    if cx >= cols { cx = cols - 1 }
    if cy >= rows { cy = rows - 1 }
    return values[cy * cols + cx]
}

fn describe(
    values: list[float],
    rows: int,
    cols: int,
    keypoints: list[KeyPoint],
    pattern: list[int],
    bytes_wide: int,
    rotate: bool
) -> Mat {
    let bits = bytes_wide * 8
    var out: list[float] = []
    for point in keypoints {
        let cx = int(point.pt.x + 0.5)
        let cy = int(point.pt.y + 0.5)
        var cosine = 1.0
        var sine = 0.0
        if rotate and point.angle >= 0.0 {
            let radians = point.angle * math.PI / 180.0
            cosine = math.cos(radians)
            sine = math.sin(radians)
        }
        for byte in 0..bytes_wide {
            var packed = 0
            for bit in 0..8 {
                let index = (byte * 8 + bit) * 4
                let ax = float(pattern[index])
                let ay = float(pattern[index + 1])
                let bx = float(pattern[index + 2])
                let by = float(pattern[index + 3])
                let rax = cx + int(math.floor(ax * cosine - ay * sine + 0.5))
                let ray = cy + int(math.floor(ax * sine + ay * cosine + 0.5))
                let rbx = cx + int(math.floor(bx * cosine - by * sine + 0.5))
                let rby = cy + int(math.floor(bx * sine + by * cosine + 0.5))
                if sample(values, rows, cols, rax, ray) < sample(values, rows, cols, rbx, rby) {
                    packed = packed | 1 << bit
                }
            }
            out.push(float(packed))
        }
        let _unused = bits
    }
    return Mat.from_list(keypoints.len(), bytes_wide, CV_MAKETYPE(CV_8U, 1), out)
}

fn patch_angle(values: list[float], rows: int, cols: int, point: KeyPoint, radius: int) -> float {
    #: The intensity centroid: the vector from the keypoint to the centre of
    #: mass of the patch around it points a consistent way whichever way the
    #: image is turned, which is what makes the descriptor rotation-invariant.
    let cx = int(point.pt.x + 0.5)
    let cy = int(point.pt.y + 0.5)
    var m01 = 0.0
    var m10 = 0.0
    for dy in -radius..radius + 1 {
        for dx in -radius..radius + 1 {
            if dx * dx + dy * dy > radius * radius { continue }
            let weight = sample(values, rows, cols, cx + dx, cy + dy)
            m10 += float(dx) * weight
            m01 += float(dy) * weight
        }
    }
    var angle = math.atan2(m01, m10) * 180.0 / math.PI
    if angle < 0.0 { angle += 360.0 }
    return angle
}

#: Oriented FAST with a rotated binary descriptor.
#:
#: Keypoints are found with FAST on every level of a scale pyramid, ranked, cut
#: down to `n_features`, given an orientation from the intensity centroid of
#: their patch, and described by comparing pairs of points in that patch after
#: turning it to face the same way.
pub class ORB {
    pub var n_features: int
    pub var scale_factor: float
    pub var n_levels: int
    pub var edge_threshold: int
    pub var patch_size: int
    pub var fast_threshold: int
    var pattern: list[int]

    pub fn init(
        self,
        n_features: int = 500,
        scale_factor: float = 1.2,
        n_levels: int = 8,
        edge_threshold: int = 31,
        patch_size: int = 31,
        fast_threshold: int = 20
    ) {
        if scale_factor <= 1.0 { throw ValueError("the scale factor has to be greater than one") }
        self.n_features = n_features
        self.scale_factor = scale_factor
        self.n_levels = n_levels
        self.edge_threshold = edge_threshold
        self.patch_size = patch_size
        self.fast_threshold = fast_threshold
        self.pattern = sampling_pattern(32, patch_size)
    }

    #: How many bytes each descriptor takes.
    pub fn descriptor_size(self) -> int { return 32 }

    #: Find keypoints without describing them.
    pub fn detect(self, image: Mat, mask: Mat? = null) -> list[KeyPoint] {
        if image.channels() != 1 { throw ValueError("ORB needs a single-channel Mat") }
        image.flush()
        var found: list[KeyPoint] = []
        var scale = 1.0
        for level in 0..self.n_levels {
            let width = int(float(image.cols) / scale)
            let height = int(float(image.rows) / scale)
            if width < self.patch_size or height < self.patch_size { break }
            let scaled = if level == 0 { image } else { resize(image, size: Size(width, height)) }
            for point in fast(scaled, self.fast_threshold, true) {
                #: A keypoint too near the edge has no patch to describe, so it
                #: is dropped here rather than clamped later.
                let margin = float(self.edge_threshold)
                if point.pt.x < margin or point.pt.y < margin { continue }
                if point.pt.x >= float(width) - margin or point.pt.y >= float(height) - margin {
                    continue
                }
                found.push(
                    KeyPoint(
                        Point2f(point.pt.x * scale, point.pt.y * scale),
                        float(self.patch_size) * scale,
                        -1.0,
                        point.response,
                        level
                    )
                )
            }
            scale *= self.scale_factor
        }

        var mask_values: list[float] = []
        let mask_image = mask
        if mask_image is not null {
            mask_image.flush()
            mask_values = mask_image.to_list()
        }

        var ordered: list[KeyPoint] = []
        for point in found {
            if mask_image is not null {
                let mx = int(point.pt.x)
                let my = int(point.pt.y)
                if mx < 0 or my < 0 or mx >= image.cols or my >= image.rows { continue }
                if mask_values[my * image.cols + mx] == 0.0 { continue }
            }
            var at = ordered.len()
            while at > 0 and point.response > ordered[at - 1].response { at -= 1 }
            ordered.insert(at, point)
        }
        if self.n_features > 0 and ordered.len() > self.n_features {
            var trimmed: list[KeyPoint] = []
            for index in 0..self.n_features { trimmed.push(ordered[index]) }
            return trimmed
        }
        return ordered
    }

    #: Describe keypoints that have already been found.
    pub fn compute(self, image: Mat, keypoints: list[KeyPoint]) -> tuple[list[KeyPoint], Mat] {
        image.flush()
        let smoothed = gaussian_blur(image, Size(7, 7), 2.0)
        let values = smoothed.values()
        let radius = self.patch_size // 2
        for point in keypoints {
            point.angle = patch_angle(values, image.rows, image.cols, point, radius)
        }
        return (
            keypoints,
            describe(values, image.rows, image.cols, keypoints, self.pattern, 32, true)
        )
    }

    #: Find keypoints and describe them in one pass.
    pub fn detect_and_compute(self, image: Mat, mask: Mat? = null) -> tuple[list[KeyPoint], Mat] {
        return self.compute(image, self.detect(image, mask))
    }
}

#: The plain binary descriptor, without an orientation.
#:
#: Cheaper than ORB and just as discriminating when the images are the same way
#: up, which is the case it is for.
pub class BRIEF {
    pub var bytes_wide: int
    pub var patch_size: int
    var pattern: list[int]

    pub fn init(self, bytes_wide: int = 32, patch_size: int = 31) {
        self.bytes_wide = bytes_wide
        self.patch_size = patch_size
        self.pattern = sampling_pattern(bytes_wide, patch_size)
    }

    #: How many bytes each descriptor takes.
    pub fn descriptor_size(self) -> int { return self.bytes_wide }

    #: Describe the given keypoints.
    pub fn compute(self, image: Mat, keypoints: list[KeyPoint]) -> tuple[list[KeyPoint], Mat] {
        image.flush()
        let smoothed = gaussian_blur(image, Size(7, 7), 2.0)
        let values = smoothed.values()
        let described = describe(
            values,
            image.rows,
            image.cols,
            keypoints,
            self.pattern,
            self.bytes_wide,
            false
        )
        return (keypoints, described)
    }
}

fn popcount(value: int) -> int {
    var remaining = value
    var count = 0
    while remaining != 0 {
        count += remaining & 1
        remaining = remaining >> 1
    }
    return count
}

fn descriptor_distance(
    a: list[float],
    b: list[float],
    offset_a: int,
    offset_b: int,
    width: int,
    norm: int
) -> float {
    if norm == NORM_HAMMING {
        var total = 0
        for index in 0..width {
            total += popcount(int(a[offset_a + index]) ^ int(b[offset_b + index]))
        }
        return float(total)
    }
    if norm == NORM_L1 {
        var total = 0.0
        for index in 0..width { total += math.fabs(a[offset_a + index] - b[offset_b + index]) }
        return total
    }
    var total = 0.0
    for index in 0..width {
        let gap = a[offset_a + index] - b[offset_b + index]
        total += gap * gap
    }
    return math.sqrt(total)
}

#: Match descriptors by comparing every one against every other.
#:
#: `cross_check` keeps only the pairs that pick each other, which throws away
#: most of the wrong matches for the cost of a second pass.
pub class BFMatcher {
    pub var norm_type: int
    pub var cross_check: bool

    pub fn init(self, norm_type: int = NORM_L2, cross_check: bool = false) {
        self.norm_type = norm_type
        self.cross_check = cross_check
    }

    #: The single best match for each descriptor in `query`.
    pub fn match_(self, query: Mat, train: Mat) -> list[DMatch] {
        if query.cols != train.cols {
            throw ValueError("both sets of descriptors have to be the same width")
        }
        let a = query.values()
        let b = train.values()
        let width = query.cols

        var out: list[DMatch] = []
        for i in 0..query.rows {
            let (best, distance) = self.nearest(a, b, i * width, train.rows, width)
            if best < 0 { continue }
            if self.cross_check {
                let (back, _distance) = self.nearest(b, a, best * width, query.rows, width)
                if back != i { continue }
            }
            out.push(DMatch(i, best, distance))
        }
        var ordered: list[DMatch] = []
        for entry in out {
            var at = ordered.len()
            while at > 0 and entry.distance < ordered[at - 1].distance { at -= 1 }
            ordered.insert(at, entry)
        }
        return ordered
    }

    fn nearest(
        self,
        from_: list[float],
        to: list[float],
        offset: int,
        count: int,
        width: int
    ) -> tuple[int, float] {
        var best = -1
        var distance = 1e300
        for index in 0..count {
            let candidate = descriptor_distance(
                from_,
                to,
                offset,
                index * width,
                width,
                self.norm_type
            )
            if candidate < distance {
                distance = candidate
                best = index
            }
        }
        return (best, distance)
    }

    #: The `k` best matches for each descriptor in `query`.
    pub fn knn_match(self, query: Mat, train: Mat, k: int) -> list[list[DMatch]] {
        if query.cols != train.cols {
            throw ValueError("both sets of descriptors have to be the same width")
        }
        let a = query.values()
        let b = train.values()
        let width = query.cols
        var out: list[list[DMatch]] = []
        for i in 0..query.rows {
            var candidates: list[DMatch] = []
            for j in 0..train.rows {
                let distance = descriptor_distance(
                    a,
                    b,
                    i * width,
                    j * width,
                    width,
                    self.norm_type
                )
                var at = candidates.len()
                while at > 0 and distance < candidates[at - 1].distance { at -= 1 }
                candidates.insert(at, DMatch(i, j, distance))
            }
            var kept: list[DMatch] = []
            for index in 0..min(k, candidates.len()) { kept.push(candidates[index]) }
            out.push(kept)
        }
        return out
    }
}

#: What a blob has to look like to count as one.
pub class SimpleBlobDetectorParams {
    pub var min_threshold: float
    pub var max_threshold: float
    pub var threshold_step: float
    pub var min_repeatability: int
    pub var filter_by_area: bool
    pub var min_area: float
    pub var max_area: float
    pub var filter_by_circularity: bool
    pub var min_circularity: float
    pub var blob_color: int

    pub fn init(self) {
        self.min_threshold = 50.0
        self.max_threshold = 220.0
        self.threshold_step = 10.0
        self.min_repeatability = 2
        self.filter_by_area = true
        self.min_area = 25.0
        self.max_area = 5000.0
        self.filter_by_circularity = false
        self.min_circularity = 0.8
        self.blob_color = 0
    }
}

#: Find round dark or light patches.
#:
#: The image is thresholded at a series of levels and the shapes that appear at
#: several of them in the same place are reported, which is what makes this
#: robust to where exactly the threshold falls.
pub class SimpleBlobDetector {
    pub var params: SimpleBlobDetectorParams

    pub fn init(self, params: SimpleBlobDetectorParams? = null) {
        let given = params
        self.params = if given is not null { given } else { SimpleBlobDetectorParams() }
    }

    #: Find the blobs.
    pub fn detect(self, image: Mat) -> list[KeyPoint] {
        if image.channels() != 1 { throw ValueError("blob detection needs a single-channel Mat") }
        image.flush()
        var centres: list[Point2f] = []
        var sizes: list[float] = []
        var seen: list[int] = []

        var level = self.params.min_threshold
        while level < self.params.max_threshold {
            let (_used, binary) = threshold(image, level, 255.0, THRESH_BINARY)
            let hunted = if self.params.blob_color == 0 { subtract_from(binary) } else { binary }
            let (contours, _hierarchy) = find_contours(hunted, RETR_LIST, CHAIN_APPROX_NONE)
            for contour in contours {
                let area = contour_area(contour)
                if self.params.filter_by_area {
                    if area < self.params.min_area or area > self.params.max_area { continue }
                }
                if area <= 0.0 { continue }
                let m = moments(contour)
                let centre = m.centroid()
                if centre is null { continue }
                if self.params.filter_by_circularity {
                    let perimeter = perimeter_of(contour)
                    if perimeter <= 0.0 { continue }
                    let roundness = 4.0 * math.PI * area / (perimeter * perimeter)
                    if roundness < self.params.min_circularity { continue }
                }
                let radius = math.sqrt(area / math.PI)
                var joined = false
                for index in 0..centres.len() {
                    let dx = centres[index].x - centre.x
                    let dy = centres[index].y - centre.y
                    if math.sqrt(dx * dx + dy * dy) > sizes[index] { continue }
                    #: The same blob seen at another threshold: average the two
                    #: centres rather than reporting it twice.
                    let weight = float(seen[index])
                    centres[index] = Point2f(
                        (centres[index].x * weight + centre.x) / (weight + 1.0),
                        (centres[index].y * weight + centre.y) / (weight + 1.0)
                    )
                    sizes[index] = (sizes[index] * weight + radius) / (weight + 1.0)
                    seen[index] += 1
                    joined = true
                    break
                }
                if not joined {
                    centres.push(centre)
                    sizes.push(radius)
                    seen.push(1)
                }
            }
            level += self.params.threshold_step
        }

        var out: list[KeyPoint] = []
        for index in 0..centres.len() {
            if seen[index] < self.params.min_repeatability { continue }
            out.push(KeyPoint(centres[index], sizes[index] * 2.0, -1.0, float(seen[index])))
        }
        return out
    }
}

fn subtract_from(binary: Mat) -> Mat {
    let values = binary.values()
    var out: list[float] = []
    for value in values { out.push(255.0 - value) }
    return Mat.from_list(binary.rows, binary.cols, binary.type_(), out)
}

fn perimeter_of(contour: list[Point]) -> float {
    let count = contour.len()
    if count < 2 { return 0.0 }
    var total = 0.0
    var previous = contour[count - 1]
    for point in contour {
        let dx = float(point.x - previous.x)
        let dy = float(point.y - previous.y)
        total += math.sqrt(dx * dx + dy * dy)
        previous = point
    }
    return total
}

#: Draw keypoints onto a copy of an image.
pub fn draw_keypoints(
    image: Mat,
    keypoints: list[KeyPoint],
    colour: Scalar = Scalar(0.0, 255.0, 0.0),
    flags: int = DRAW_MATCHES_FLAGS_DEFAULT
) -> Mat {
    let canvas = if image.channels() == 3 { image.clone() } else { to_colour(image) }
    for point in keypoints {
        let centre = Point(int(point.pt.x + 0.5), int(point.pt.y + 0.5))
        if flags & DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS != 0 {
            let radius = max(int(point.size * 0.5), 1)
            circle(canvas, centre, radius, colour, 1)
            if point.angle >= 0.0 {
                let radians = point.angle * math.PI / 180.0
                let tip = Point(
                    centre.x + int(float(radius) * math.cos(radians)),
                    centre.y + int(float(radius) * math.sin(radians))
                )
                line(canvas, centre, tip, colour, 1)
            }
        } else {
            circle(canvas, centre, 2, colour, 1)
        }
    }
    return canvas
}

fn to_colour(image: Mat) -> Mat {
    let values = image.values()
    var out: list[float] = []
    for value in values {
        out.push(value)
        out.push(value)
        out.push(value)
    }
    return Mat.from_list(image.rows, image.cols, CV_MAKETYPE(CV_8U, 3), out)
}

#: Draw two images side by side with lines between matched keypoints.
pub fn draw_matches(
    first: Mat,
    first_keypoints: list[KeyPoint],
    second: Mat,
    second_keypoints: list[KeyPoint],
    matches: list[DMatch],
    colour: Scalar = Scalar(0.0, 255.0, 0.0)
) -> Mat {
    let left = if first.channels() == 3 { first } else { to_colour(first) }
    let right = if second.channels() == 3 { second } else { to_colour(second) }
    let rows = max(left.rows, right.rows)
    let cols = left.cols + right.cols
    let canvas = Mat.zeros(rows, cols, CV_MAKETYPE(CV_8U, 3))
    left.copy_to(canvas.roi(Rect(0, 0, left.cols, left.rows)))
    right.copy_to(canvas.roi(Rect(left.cols, 0, right.cols, right.rows)))

    for entry in matches {
        if entry.query_idx < 0 or entry.query_idx >= first_keypoints.len() { continue }
        if entry.train_idx < 0 or entry.train_idx >= second_keypoints.len() { continue }
        let a = first_keypoints[entry.query_idx].pt
        let b = second_keypoints[entry.train_idx].pt
        let from_ = Point(int(a.x + 0.5), int(a.y + 0.5))
        let to = Point(int(b.x + 0.5) + left.cols, int(b.y + 0.5))
        circle(canvas, from_, 3, colour, 1)
        circle(canvas, to, 3, colour, 1)
        line(canvas, from_, to, colour, 1)
    }
    return canvas
}
