#: Histograms of oriented gradients, and the sliding-window detector built on
#: them.
#:
#: `compute` produces the descriptor Dalal and Triggs describe and OpenCV
#: implements: the image is cut into cells, each cell holds a histogram of
#: gradient directions weighted by gradient strength, cells are grouped into
#: overlapping blocks, and each block is normalised on its own so that a
#: brighter part of the picture does not outvote a darker one.
#:
#: The detector is a plane through that descriptor. OpenCV ships the
#: coefficients of a people detector trained offline; those are data rather
#: than algorithm and are not here, so `set_svm_detector` takes coefficients
#: the caller has — from OpenCV, or from `svm_create` in `ml` trained on
#: descriptors this computed.
#:
#: ```
#: let hog = HOGDescriptor()
#: let features = hog.compute(window)
#: hog.set_svm_detector(coefficients)
#: for box in hog.detect_multi_scale(frame) { ... }
#: ```
import std.math
from ..core.mat import Mat
from ..core.types import CV_32F, CV_MAKETYPE, Point, Rect, Size
from ..imgproc.geometric import resize
from .cascade import group_rectangles

#: Signed gradients run the full turn; unsigned fold the two halves together.
pub const HOG_SIGNED = 1
#: The usual choice, and OpenCV's default.
pub const HOG_UNSIGNED = 0

#: The descriptor, and the detector that reads it.
pub class HOGDescriptor {
    #: The window a descriptor describes.
    pub var win_size: Size
    #: The block a normalisation covers.
    pub var block_size: Size
    #: How far a block moves between positions.
    pub var block_stride: Size
    #: The cell a histogram covers.
    pub var cell_size: Size
    #: How many directions a histogram splits into.
    pub var nbins: int
    #: How large a single bin may grow before it is clipped and the rest
    #: renormalised, which is what stops one strong edge dominating a block.
    pub var l2_hys_threshold: float
    #: Whether directions run the full turn.
    pub var signed_gradient: int
    var detector: list[float]

    pub fn init(
        self,
        win_size: Size = Size(64, 128),
        block_size: Size = Size(16, 16),
        block_stride: Size = Size(8, 8),
        cell_size: Size = Size(8, 8),
        nbins: int = 9
    ) {
        self.win_size = win_size
        self.block_size = block_size
        self.block_stride = block_stride
        self.cell_size = cell_size
        self.nbins = nbins
        self.l2_hys_threshold = 0.2
        self.signed_gradient = HOG_UNSIGNED
        self.detector = []
    }

    #: How many numbers a descriptor holds.
    pub fn get_descriptor_size(self) -> int {
        let across = (self.win_size.width - self.block_size.width) // self.block_stride.width + 1
        let down = (self.win_size.height - self.block_size.height) // self.block_stride.height + 1
        let cells_x = self.block_size.width // self.cell_size.width
        let cells_y = self.block_size.height // self.cell_size.height
        return across * down * cells_x * cells_y * self.nbins
    }

    #: Give the detector the plane it should compare against.
    #:
    #: The list is the descriptor's coefficients followed by one more, the
    #: bias, which is how OpenCV stores it too.
    pub fn set_svm_detector(self, coefficients: list[float]) -> void {
        let wanted = self.get_descriptor_size()
        if coefficients.len() != wanted + 1 and coefficients.len() != wanted {
            throw ValueError(
                f"this window needs {wanted} coefficients and a bias, got {coefficients.len()}"
            )
        }
        self.detector = coefficients
    }

    fn gradients(self, window: Mat) -> tuple[list[float], list[float]] {
        let rows = window.rows
        let cols = window.cols
        let cn = window.channels()
        let values = window.values()
        var magnitude: list[float] = []
        var direction: list[float] = []
        let turn = if self.signed_gradient == HOG_SIGNED { 360.0 } else { 180.0 }

        for y in 0..rows {
            for x in 0..cols {
                var best = 0.0
                var best_gx = 0.0
                var best_gy = 0.0
                for c in 0..cn {
                    #: A colour image takes whichever channel has the strongest
                    #: gradient at each pixel, which is what OpenCV does.
                    let left = values[(y * cols + max(x - 1, 0)) * cn + c]
                    let right = values[(y * cols + min(x + 1, cols - 1)) * cn + c]
                    let up = values[(max(y - 1, 0) * cols + x) * cn + c]
                    let down = values[(min(y + 1, rows - 1) * cols + x) * cn + c]
                    let gx = right - left
                    let gy = down - up
                    let size = math.sqrt(gx * gx + gy * gy)
                    if size > best {
                        best = size
                        best_gx = gx
                        best_gy = gy
                    }
                }
                magnitude.push(best)
                var angle = math.atan2(best_gy, best_gx) * 180.0 / math.PI
                if angle < 0.0 { angle += 360.0 }
                if turn == 180.0 and angle >= 180.0 { angle -= 180.0 }
                direction.push(angle)
            }
        }
        return (magnitude, direction)
    }

    #: The descriptor of one window.
    pub fn compute(self, window: Mat) -> list[float] {
        window.flush()
        if window.cols != self.win_size.width or window.rows != self.win_size.height {
            throw ValueError(f"this descriptor takes {self.win_size} windows, got {window.size()}")
        }
        let (magnitude, direction) = self.gradients(window)
        let cols = window.cols
        let cells_x = self.win_size.width // self.cell_size.width
        let cells_y = self.win_size.height // self.cell_size.height
        let turn = if self.signed_gradient == HOG_SIGNED { 360.0 } else { 180.0 }
        let width = turn / float(self.nbins)

        var cells: list[float] = []
        for _index in 0..cells_x * cells_y * self.nbins { cells.push(0.0) }
        for y in 0..self.win_size.height {
            let cy = y // self.cell_size.height
            if cy >= cells_y { continue }
            for x in 0..self.win_size.width {
                let cx = x // self.cell_size.width
                if cx >= cells_x { continue }
                let weight = magnitude[y * cols + x]
                if weight <= 0.0 { continue }
                #: The direction is split between the two nearest bins in
                #: proportion to how close it is to each, so a gradient that
                #: turns slowly moves smoothly between them rather than
                #: jumping.
                let position = direction[y * cols + x] / width - 0.5
                let lower = int(math.floor(position))
                let fraction = position - float(lower)
                let first = (lower % self.nbins + self.nbins) % self.nbins
                let second = (first + 1) % self.nbins
                let base = (cy * cells_x + cx) * self.nbins
                cells[base + first] += weight * (1.0 - fraction)
                cells[base + second] += weight * fraction
            }
        }

        let across = (self.win_size.width - self.block_size.width) // self.block_stride.width + 1
        let down = (self.win_size.height - self.block_size.height) // self.block_stride.height + 1
        let block_cells_x = self.block_size.width // self.cell_size.width
        let block_cells_y = self.block_size.height // self.cell_size.height

        var out: list[float] = []
        for by in 0..down {
            for bx in 0..across {
                var block: list[float] = []
                let ox = bx * self.block_stride.width // self.cell_size.width
                let oy = by * self.block_stride.height // self.cell_size.height
                for cy in 0..block_cells_y {
                    for cx in 0..block_cells_x {
                        let base = ((oy + cy) * cells_x + ox + cx) * self.nbins
                        for bin in 0..self.nbins { block.push(cells[base + bin]) }
                    }
                }
                #: L2-Hys: normalise, clip whatever is still too large, and
                #: normalise again. The clip is what makes it robust to a
                #: single very strong edge.
                normalise(block, self.l2_hys_threshold)
                for value in block { out.push(value) }
            }
        }
        return out
    }

    fn score(self, window: Mat) -> float {
        if self.detector.len() == 0 { throw RuntimeError("this detector has no coefficients") }
        let features = self.compute(window)
        var total = 0.0
        for index in 0..features.len() { total += features[index] * self.detector[index] }
        if self.detector.len() > features.len() { total += self.detector[features.len()] }
        return total
    }

    #: Find every window the detector accepts, at every scale.
    pub fn detect_multi_scale(
        self,
        image: Mat,
        hit_threshold: float = 0.0,
        win_stride: Size = Size(8, 8),
        scale_factor: float = 1.05,
        group_threshold: int = 2
    ) -> list[Rect] {
        if self.detector.len() == 0 { throw RuntimeError("this detector has no coefficients") }
        if scale_factor <= 1.0 { throw ValueError("the scale factor has to be greater than one") }
        image.flush()

        var found: list[Rect] = []
        var factor = 1.0
        loop {
            let width = int(float(image.cols) / factor + 0.5)
            let height = int(float(image.rows) / factor + 0.5)
            if width < self.win_size.width or height < self.win_size.height { break }
            let level = if factor == 1.0 { image } else { resize(image, size: Size(width, height)) }

            var y = 0
            while y + self.win_size.height <= height {
                var x = 0
                while x + self.win_size.width <= width {
                    let window = level.roi(Rect(x, y, self.win_size.width, self.win_size.height))
                    if self.score(window) > hit_threshold {
                        found.push(
                            Rect(
                                int(float(x) * factor + 0.5),
                                int(float(y) * factor + 0.5),
                                int(float(self.win_size.width) * factor + 0.5),
                                int(float(self.win_size.height) * factor + 0.5)
                            )
                        )
                    }
                    x += win_stride.width
                }
                y += win_stride.height
            }
            factor *= scale_factor
        }
        return group_rectangles(found, group_threshold, 0.2)
    }

    #: Find windows at the original scale only.
    pub fn detect(
        self,
        image: Mat,
        hit_threshold: float = 0.0,
        win_stride: Size = Size(8, 8)
    ) -> list[Point] {
        if self.detector.len() == 0 { throw RuntimeError("this detector has no coefficients") }
        image.flush()
        var out: list[Point] = []
        var y = 0
        while y + self.win_size.height <= image.rows {
            var x = 0
            while x + self.win_size.width <= image.cols {
                let window = image.roi(Rect(x, y, self.win_size.width, self.win_size.height))
                if self.score(window) > hit_threshold { out.push(Point(x, y)) }
                x += win_stride.width
            }
            y += win_stride.height
        }
        return out
    }
}

fn normalise(block: list[float], clip: float) -> void {
    var energy = 0.0
    for value in block { energy += value * value }
    var length = math.sqrt(energy + 1e-8)
    for index in 0..block.len() { block[index] /= length }
    var clipped = false
    for index in 0..block.len() {
        if block[index] > clip {
            block[index] = clip
            clipped = true
        }
    }
    if not clipped { return }
    energy = 0.0
    for value in block { energy += value * value }
    length = math.sqrt(energy + 1e-8)
    for index in 0..block.len() { block[index] /= length }
}
