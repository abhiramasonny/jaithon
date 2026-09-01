#: `QRCodeDetector`: OpenCV's method names over the steps beside this one.
import std.math
from ...core.mat import Mat
from ...core.types import CV_8U, CV_MAKETYPE, Point2f
from ...imgproc.threshold import otsu_threshold
from .decode import QRCode, decode_grid
from .finder import find_finders
from .sample import dimension_between, distance, order_finders, sample_grid

#: Find and read QR codes, with OpenCV's method names.
#:
#: ```
#: let detector = QRCodeDetector()
#: let (text, corners) = detector.detect_and_decode(frame)
#: ```
pub class QRCodeDetector {
    #: How far the three finder centres may be from a right isoceles triangle,
    #: as a fraction of the shorter side, before the triple is refused.
    pub var tolerance: float

    pub fn init(self, tolerance: float = 0.4) { self.tolerance = tolerance }

    fn prepared(self, image: Mat) -> tuple[list[float], int, int] {
        image.flush()
        let gray = if image.channels() == 1 {
            image
        } else {
            var values: list[float] = []
            let flat = image.values()
            let cn = image.channels()
            for index in 0..image.rows * image.cols {
                let b = flat[index * cn]
                let g = flat[index * cn + 1]
                let r = flat[index * cn + 2]
                values.push(0.114 * b + 0.587 * g + 0.299 * r)
            }
            Mat.from_list(image.rows, image.cols, CV_MAKETYPE(CV_8U, 1), values)
        }
        gray.flush()
        let level = otsu_threshold(gray)
        let raw = gray.to_list()
        var binary: list[float] = []
        #: Rewritten to a hard nought or two hundred and fifty-five so every
        #: test below is the same comparison against the midpoint, whatever the
        #: picture's own contrast was.
        for value in raw { binary.push(if value <= level { 0.0 } else { 255.0 }) }
        return (binary, gray.rows, gray.cols)
    }

    #: Every symbol in the image, as its four corners in the order OpenCV
    #: reports them: top-left, top-right, bottom-right, bottom-left.
    pub fn detect_multi(self, image: Mat) -> list[list[Point2f]] {
        var out: list[list[Point2f]] = []
        for found in self.read(image, false) { out.push(found[1]) }
        return out
    }

    #: The corners of the first symbol found, or null.
    pub fn detect(self, image: Mat) -> list[Point2f]? {
        let all = self.detect_multi(image)
        if all.len() == 0 { return null }
        return all[0]
    }

    #: Every symbol in the image, decoded.
    pub fn detect_and_decode_multi(self, image: Mat) -> tuple[list[str], list[list[Point2f]]] {
        var texts: list[str] = []
        var corners: list[list[Point2f]] = []
        for found in self.read(image, true) {
            let code = found[0]
            if code is null { continue }
            texts.push(code.text)
            corners.push(found[1])
        }
        return (texts, corners)
    }

    #: The first symbol's text and corners; the text is empty when nothing was
    #: found or nothing could be read, which is what OpenCV does too.
    pub fn detect_and_decode(self, image: Mat) -> tuple[str, list[Point2f]] {
        for found in self.read(image, true) {
            let code = found[0]
            if code is not null { return (code.text, found[1]) }
        }
        return ("", [])
    }

    #: Everything the reader learned about one symbol, for a caller that wants
    #: the version and correction level as well as the text.
    pub fn read_codes(self, image: Mat) -> list[QRCode] {
        var out: list[QRCode] = []
        for found in self.read(image, true) {
            let code = found[0]
            if code is not null { out.push(code) }
        }
        return out
    }

    fn read(self, image: Mat, decode: bool) -> list[tuple[QRCode?, list[Point2f]]] {
        let (values, rows, cols) = self.prepared(image)
        let finders = find_finders(values, rows, cols)
        var out: list[tuple[QRCode?, list[Point2f]]] = []
        if finders.len() < 3 { return out }

        var used: list[bool] = []
        for _index in 0..finders.len() { used.push(false) }

        for a in 0..finders.len() {
            for b in a + 1..finders.len() {
                for c in b + 1..finders.len() {
                    if used[a] or used[b] or used[c] { continue }
                    let (corner, top, side_finder) = order_finders(
                        finders[a],
                        finders[b],
                        finders[c]
                    )
                    let across = distance(corner.y, corner.x, top.y, top.x)
                    let down = distance(corner.y, corner.x, side_finder.y, side_finder.x)
                    if across <= 0.0 or down <= 0.0 { continue }
                    #: Three finders of one symbol are the same size. Checking
                    #: that before the geometry costs three comparisons and
                    #: throws out most triples, which each cost a sampling of
                    #: the whole symbol otherwise.
                    let widest = max(corner.width, max(top.width, side_finder.width))
                    let narrowest = min(corner.width, min(top.width, side_finder.width))
                    if widest > narrowest * 2.0 { continue }
                    let shorter = min(across, down)
                    #: The two sides of a symbol are equal and meet at a right
                    #: angle. Both are checked, because three finders in a row
                    #: pass the first test on their own.
                    if math.fabs(across - down) > shorter * self.tolerance { continue }
                    let diagonal = distance(top.y, top.x, side_finder.y, side_finder.x)
                    let wanted = math.sqrt(across * across + down * down)
                    if math.fabs(diagonal - wanted) > shorter * self.tolerance { continue }

                    let estimate = dimension_between(corner, top, side_finder)
                    if estimate < 21 or estimate > 177 { continue }
                    #: The module width is measured from runs crossing the
                    #: finders, and a symbol photographed at an angle makes
                    #: every such run wider than a module by the secant of the
                    #: angle -- which is enough to name the version one step
                    #: wrong. Rather than correct the width, the two
                    #: neighbouring sizes are tried as well: a wrong dimension
                    #: fails the format word or the parity, so accepting the
                    #: first that reads is as safe as getting it right first
                    #: time and costs two more samplings on a symbol that would
                    #: otherwise not be read at all.
                    var code: QRCode? = null
                    var dimension = 0
                    for offset in [0, 4, -4, 8, -8] {
                        let trying = estimate + offset
                        if trying < 21 or trying > 177 { continue }
                        let grid = sample_grid(values, rows, cols, corner, top, side_finder, trying)
                        if grid is null { continue }
                        if not decode {
                            dimension = trying
                            break
                        }
                        code = decode_grid(grid)
                        if code is not null {
                            dimension = trying
                            break
                        }
                    }
                    if dimension == 0 { continue }
                    if decode and code is null { continue }

                    let far = Point2f(
                        top.x + side_finder.x - corner.x,
                        top.y + side_finder.y - corner.y
                    )
                    out.push(
                        (
                            code,
                            [
                                Point2f(corner.x, corner.y),
                                Point2f(top.x, top.y),
                                far,
                                Point2f(side_finder.x, side_finder.y),
                            ]
                        )
                    )
                    used[a] = true
                    used[b] = true
                    used[c] = true
                }
            }
        }
        return out
    }
}
