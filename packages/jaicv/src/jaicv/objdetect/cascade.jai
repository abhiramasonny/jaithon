#: Cascade detection: the Viola-Jones detector, reading OpenCV's own trained
#: cascade files.
#:
#: `CascadeClassifier` loads the XML OpenCV writes, so the cascades that ship
#: with OpenCV — the frontal-face one and its siblings — work here unchanged.
#: The evaluation follows OpenCV's: a window is rejected as soon as any stage's
#: weighted vote falls short, features are read out of an integral image, and
#: the response is divided by the window's own standard deviation so that
#: lighting does not decide the answer.
#:
#: Scale is handled the way OpenCV handles it, by shrinking the image rather
#: than growing the features, so every scale evaluates the same fixed window.
#:
#: ```
#: let faces = CascadeClassifier("haarcascade_frontalface_default.xml")
#: for box in faces.detect_multi_scale(gray, 1.1, 3) {
#:     rectangle_rect(frame, box, Scalar(0.0, 255.0, 0.0), 2)
#: }
#: ```
import std.math
from std.io import Path
from ..core.mat import Mat
from ..core.types import Rect, Size
from ..imgproc.geometric import resize
from .xml import parse_xml, Node

#: Do not look for anything smaller than the objects already found.
pub const CASCADE_DO_CANNY_PRUNING = 1
#: Scale the image rather than the features, which is what this always does.
pub const CASCADE_SCALE_IMAGE = 2
#: Return only the largest object.
pub const CASCADE_FIND_BIGGEST_OBJECT = 4
#: Stop as soon as one object is found at a scale.
pub const CASCADE_DO_ROUGH_SEARCH = 8

class Feature {
    #: Up to three rectangles, each `x y width height weight`.
    pub var rects: list[list[float]]
    pub var tilted: bool

    pub fn init(self, rects: list[list[float]], tilted: bool) {
        self.rects = rects
        self.tilted = tilted
    }
}

class TreeNode {
    pub let left: int
    pub let right: int
    pub let feature: int
    pub let threshold: float

    pub fn init(self, left: int, right: int, feature: int, threshold: float) {
        self.left = left
        self.right = right
        self.feature = feature
        self.threshold = threshold
    }
}

class Weak {
    pub var nodes: list[TreeNode]
    pub var leaves: list[float]

    pub fn init(self, nodes: list[TreeNode], leaves: list[float]) {
        self.nodes = nodes
        self.leaves = leaves
    }
}

class Stage {
    pub var threshold: float
    pub var weak: list[Weak]

    pub fn init(self, threshold: float, weak: list[Weak]) {
        self.threshold = threshold
        self.weak = weak
    }
}

#: A trained cascade of boosted classifiers.
pub class CascadeClassifier {
    #: The window the cascade was trained on.
    pub var window: Size
    var features: list[Feature]
    var stages: list[Stage]
    var loaded: bool

    pub fn init(self, filename: str = "") {
        self.window = Size(0, 0)
        self.features = []
        self.stages = []
        self.loaded = false
        if filename.len() > 0 {
            if not self.load(filename) {
                throw ValueError(f"could not read a cascade from {filename}")
            }
        }
    }

    #: Whether anything has been loaded yet.
    pub fn empty(self) -> bool { return not self.loaded }

    #: Read a cascade from an OpenCV XML file.
    pub fn load(self, filename: str) -> bool {
        let path = Path(filename)
        if not path.exists() { return false }
        return self.read(path.read_text())
    }

    #: Read a cascade from XML already in memory.
    pub fn read(self, source: str) -> bool {
        let root = parse_xml(source)
        var cascade = root.find("cascade")
        if cascade is null {
            #: A file written by the old trainer names the cascade after itself
            #: rather than calling it `cascade`, so the one child of
            #: `opencv_storage` is taken instead.
            if root.children.len() == 1 { cascade = root.children[0] }
        }
        if cascade is null { return false }

        let kind = cascade.text_of("featureType", "HAAR")
        if kind != "HAAR" {
            throw ValueError(f"cascade feature type {kind} is not one jaicv reads; HAAR is")
        }
        let width = int(cascade.number_of("width", 0.0))
        let height = int(cascade.number_of("height", 0.0))
        if width <= 0 or height <= 0 { return false }
        self.window = Size(width, height)

        let features = cascade.find("features")
        if features is null { return false }
        var read_features: list[Feature] = []
        for entry in features.children {
            let rects = entry.find("rects")
            if rects is null { continue }
            var shapes: list[list[float]] = []
            for rect in rects.children {
                let numbers = rect.numbers()
                if numbers.len() < 5 { continue }
                shapes.push([numbers[0], numbers[1], numbers[2], numbers[3], numbers[4]])
            }
            read_features.push(Feature(shapes, entry.number_of("tilted", 0.0) != 0.0))
        }
        self.features = read_features

        let stages = cascade.find("stages")
        if stages is null { return false }
        var read_stages: list[Stage] = []
        for entry in stages.children {
            let classifiers = entry.find("weakClassifiers")
            if classifiers is null { continue }
            var weak: list[Weak] = []
            for classifier in classifiers.children {
                let internal = classifier.find("internalNodes")
                let leaf = classifier.find("leafValues")
                if internal is null or leaf is null { continue }
                let numbers = internal.numbers()
                var nodes: list[TreeNode] = []
                var at = 0
                while at + 4 <= numbers.len() {
                    nodes.push(
                        TreeNode(
                            int(numbers[at]),
                            int(numbers[at + 1]),
                            int(numbers[at + 2]),
                            numbers[at + 3]
                        )
                    )
                    at += 4
                }
                weak.push(Weak(nodes, leaf.numbers()))
            }
            read_stages.push(Stage(entry.number_of("stageThreshold", 0.0), weak))
        }
        self.stages = read_stages
        self.loaded = self.stages.len() > 0 and self.features.len() > 0
        return self.loaded
    }

    fn passes(
        self,
        integral: list[float],
        squared: list[float],
        stride: int,
        x: int,
        y: int
    ) -> bool {
        let w = self.window.width
        let h = self.window.height
        #: The variance is taken over the window inset by a pixel, which is
        #: what the cascade was trained against.
        let area = float((w - 2) * (h - 2))
        let sum = rect_sum(integral, stride, x + 1, y + 1, w - 2, h - 2)
        let sqsum = rect_sum(squared, stride, x + 1, y + 1, w - 2, h - 2)
        let spread = area * sqsum - sum * sum
        if spread <= 0.0 { return false }
        let normalise = 1.0 / math.sqrt(spread)

        for stage in self.stages {
            var vote = 0.0
            for weak in stage.weak {
                if weak.nodes.len() == 0 { continue }
                var index = 0
                while true {
                    let node = weak.nodes[index]
                    let value = self.feature_value(integral, stride, x, y, node.feature) * normalise
                    index = if value < node.threshold { node.left } else { node.right }
                    if index <= 0 { break }
                }
                vote += weak.leaves[-index]
            }
            if vote < stage.threshold { return false }
        }
        return true
    }

    fn feature_value(
        self,
        integral: list[float],
        stride: int,
        x: int,
        y: int,
        index: int
    ) -> float {
        let feature = self.features[index]
        var total = 0.0
        for rect in feature.rects {
            total += rect[4] *
                rect_sum(integral, stride, x + int(rect[0]), y + int(rect[1]), int(rect[2]), int(rect[3]))
        }
        return total
    }

    #: Find every object the cascade recognises, at every scale.
    #:
    #: `scale_factor` is how much the image shrinks between passes and
    #: `min_neighbors` how many overlapping detections a real object needs, both
    #: of which mean what they do in OpenCV.
    pub fn detect_multi_scale(
        self,
        image: Mat,
        scale_factor: float = 1.1,
        min_neighbors: int = 3,
        min_size: Size = Size(0, 0),
        max_size: Size = Size(0, 0)
    ) -> list[Rect] {
        if not self.loaded { throw RuntimeError("this cascade has nothing loaded in it") }
        if image.channels() != 1 {
            throw ValueError("cascade detection needs a single-channel Mat")
        }
        if scale_factor <= 1.0 { throw ValueError("the scale factor has to be greater than one") }
        image.flush()

        var found: list[Rect] = []
        var factor = 1.0
        while true {
            let width = int(float(image.cols) / factor + 0.5)
            let height = int(float(image.rows) / factor + 0.5)
            if width < self.window.width or height < self.window.height { break }

            let scaled_w = int(float(self.window.width) * factor + 0.5)
            let scaled_h = int(float(self.window.height) * factor + 0.5)
            let too_small = min_size.width > 0 and
                (scaled_w < min_size.width or scaled_h < min_size.height)
            let too_big = max_size.width > 0 and
                (scaled_w > max_size.width or scaled_h > max_size.height)
            if too_big { break }

            if not too_small {
                let level = if factor == 1.0 {
                    image
                } else {
                    resize(image, size: Size(width, height))
                }
                let (integral, squared) = integrals(level)
                let stride = width + 1
                #: OpenCV steps two pixels at a time until the window is large
                #: enough that one pixel of the level is two of the original.
                let step = if factor > 2.0 { 1 } else { 2 }
                var y = 0
                while y + self.window.height <= height {
                    var x = 0
                    while x + self.window.width <= width {
                        if self.passes(integral, squared, stride, x, y) {
                            found.push(
                                Rect(
                                    int(float(x) * factor + 0.5),
                                    int(float(y) * factor + 0.5),
                                    scaled_w,
                                    scaled_h
                                )
                            )
                        }
                        x += step
                    }
                    y += step
                }
            }
            factor *= scale_factor
        }
        return group_rectangles(found, min_neighbors, 0.2)
    }
}

fn rect_sum(integral: list[float], stride: int, x: int, y: int, width: int, height: int) -> float {
    let top_left = integral[y * stride + x]
    let top_right = integral[y * stride + x + width]
    let bottom_left = integral[(y + height) * stride + x]
    let bottom_right = integral[(y + height) * stride + x + width]
    return bottom_right - bottom_left - top_right + top_left
}

fn integrals(image: Mat) -> tuple[list[float], list[float]] {
    let rows = image.rows
    let cols = image.cols
    let values = image.values()
    let stride = cols + 1
    var plain: list[float] = []
    var squared: list[float] = []
    for _index in 0..(rows + 1) * stride {
        plain.push(0.0)
        squared.push(0.0)
    }
    for y in 0..rows {
        var row_sum = 0.0
        var row_squared = 0.0
        for x in 0..cols {
            let value = values[y * cols + x]
            row_sum += value
            row_squared += value * value
            plain[(y + 1) * stride + x + 1] = plain[y * stride + x + 1] + row_sum
            squared[(y + 1) * stride + x + 1] = squared[y * stride + x + 1] + row_squared
        }
    }
    return (plain, squared)
}

fn similar(a: Rect, b: Rect, epsilon: float) -> bool {
    let allowed = epsilon * float(min(a.width, b.width) + min(a.height, b.height)) * 0.5
    return math.fabs(float(a.x - b.x)) <= allowed and
        math.fabs(float(a.y - b.y)) <= allowed and
        math.fabs(float(a.x + a.width - b.x - b.width)) <= allowed and
        math.fabs(float(a.y + a.height - b.y - b.height)) <= allowed
}

#: Merge overlapping detections and drop the ones too few passes agreed on.
#:
#: This is OpenCV's `groupRectangles`: rectangles that are near each other are
#: put in one class, each class is averaged, and a class with fewer than
#: `group_threshold` members is thrown away as a false alarm.
pub fn group_rectangles(
    rectangles: list[Rect],
    group_threshold: int,
    epsilon: float = 0.2
) -> list[Rect] {
    if group_threshold <= 0 or rectangles.len() == 0 { return rectangles }
    let count = rectangles.len()
    var labels: list[int] = []
    for _index in 0..count { labels.push(-1) }
    var classes = 0
    for index in 0..count {
        if labels[index] >= 0 { continue }
        labels[index] = classes
        for other in index + 1..count {
            if labels[other] >= 0 { continue }
            if similar(rectangles[index], rectangles[other], epsilon) { labels[other] = classes }
        }
        classes += 1
    }

    var sums: list[list[float]] = []
    var members: list[int] = []
    for _index in 0..classes {
        sums.push([0.0, 0.0, 0.0, 0.0])
        members.push(0)
    }
    for index in 0..count {
        let label = labels[index]
        let rect = rectangles[index]
        sums[label][0] += float(rect.x)
        sums[label][1] += float(rect.y)
        sums[label][2] += float(rect.width)
        sums[label][3] += float(rect.height)
        members[label] += 1
    }

    var kept: list[Rect] = []
    for label in 0..classes {
        if members[label] < group_threshold { continue }
        let total = float(members[label])
        kept.push(
            Rect(
                int(sums[label][0] / total + 0.5),
                int(sums[label][1] / total + 0.5),
                int(sums[label][2] / total + 0.5),
                int(sums[label][3] / total + 0.5)
            )
        )
    }

    #: A rectangle wholly inside a larger surviving one is the same object seen
    #: at another scale, so only the outer one is reported.
    var out: list[Rect] = []
    for index in 0..kept.len() {
        var swallowed = false
        for other in 0..kept.len() {
            if index == other { continue }
            let inner = kept[index]
            let outer = kept[other]
            if outer.width <= inner.width or outer.height <= inner.height { continue }
            let margin = int(float(outer.width) * epsilon + 0.5)
            if inner.x >= outer.x - margin and
                inner.y >= outer.y - margin and
                inner.x + inner.width <= outer.x + outer.width + margin and
                inner.y + inner.height <= outer.y + outer.height + margin {
                swallowed = true
                break
            }
        }
        if not swallowed { out.push(kept[index]) }
    }
    return out
}
