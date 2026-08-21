#: Flatten a photographed page into a clean, square-on scan.
#:
#:     jaithon run examples/doc_scanner/scan.jai <photo> [scan.png]
#:
#: No weights, no downloads -- this is the classic pipeline, and every step of
#: it is a jaicv call:
#:
#:     grey -> blur -> Canny -> dilate -> contours -> largest quadrilateral
#:          -> perspective transform -> adaptive threshold
#:
#: The two steps that carry the whole thing are finding the quadrilateral and
#: ordering its corners. `approx_poly_dp` reduces a contour to its corners,
#: and a page is the largest four-corner convex contour in the picture; the
#: ordering matters because `get_perspective_transform` maps corner one to
#: corner one, so a rotated ordering produces a rotated -- or mirrored --
#: scan. Sorting by the sum and difference of the coordinates is the standard
#: trick and holds for any rotation short of 45 degrees.
import jaicv as cv
from jaicv import Mat, Point, Point2f, Scalar, Size
from std.os import argv
from std.time import Instant

#: A page has to occupy a decent share of the frame; anything smaller is a
#: label or a shadow, and accepting it produces a confident crop of nothing.
let MIN_AREA_FRACTION = 0.15

#: How far from the contour the simplified polygon may sit, as a fraction of
#: the perimeter. Two per cent keeps a slightly bowed page as four corners
#: while still refusing a circle.
let CORNER_TOLERANCE = 0.02

#: Corners as top-left, top-right, bottom-right, bottom-left.
#:
#: The sum `x + y` is smallest at the top-left and largest at the bottom-
#: right; the difference `x - y` is smallest at the bottom-left and largest at
#: the top-right. That holds however the quadrilateral is rotated, which a
#: sort by angle around the centroid also does -- but this needs no
#: trigonometry and no tie-breaking.
fn order_corners(quad: list[Point]) -> list[Point2f] {
    var top_left = quad[0]
    var bottom_right = quad[0]
    var top_right = quad[0]
    var bottom_left = quad[0]
    for corner in quad {
        let total = corner.x + corner.y
        let apart = corner.x - corner.y
        if total < top_left.x + top_left.y { top_left = corner }
        if total > bottom_right.x + bottom_right.y { bottom_right = corner }
        if apart > top_right.x - top_right.y { top_right = corner }
        if apart < bottom_left.x - bottom_left.y { bottom_left = corner }
    }
    return [
        Point2f(float(top_left.x), float(top_left.y)),
        Point2f(float(top_right.x), float(top_right.y)),
        Point2f(float(bottom_right.x), float(bottom_right.y)),
        Point2f(float(bottom_left.x), float(bottom_left.y)),
    ]
}

fn distance(a: Point2f, b: Point2f) -> float {
    let dx = a.x - b.x
    let dy = a.y - b.y
    return (dx * dx + dy * dy) ** 0.5
}

#: The largest four-cornered convex contour, or nothing.
fn find_page(edges: Mat, area: float) -> list[Point]? {
    let (contours, _hierarchy) = cv.find_contours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    var best: list[Point]? = null
    var best_area = area * MIN_AREA_FRACTION
    for contour in contours {
        if contour.len() < 4 { continue }
        let perimeter = cv.arc_length(contour, true)
        let quad = cv.approx_poly_dp(contour, CORNER_TOLERANCE * perimeter, true)
        if quad.len() != 4 { continue }
        if not cv.is_contour_convex(quad) { continue }
        let size = cv.contour_area(quad)
        if size > best_area {
            best_area = size
            best = quad
        }
    }
    return best
}

fn main() -> int {
    let args = argv()
    if args.len() < 2 {
        print("usage: jaithon run examples/doc_scanner/scan.jai <photo> [scan.png]")
        return 2
    }
    let target = if args.len() > 2 { args[2] } else { "scan.png" }

    var photo = cv.imread(args[1])
    if photo.empty() {
        print(f"could not read {args[1]}")
        return 1
    }
    let clock = Instant.now()

    let grey = cv.cvt_color(photo, cv.COLOR_BGR2GRAY)
    let blurred = cv.gaussian_blur(grey, Size(5, 5), 0.0)
    var edges = cv.canny(blurred, 60.0, 180.0)
    #: A page's border is often broken by a shadow or a low-contrast corner,
    #: and a contour has to close to be a quadrilateral. One dilation bridges
    #: the gaps without merging the page into what surrounds it.
    edges = cv.dilate(edges, cv.get_structuring_element(cv.MORPH_RECT, Size(3, 3)))

    let quad = find_page(edges, float(photo.rows * photo.cols))
    if quad is null {
        print("no page-shaped quadrilateral found")
        print("the page needs to be fully in frame and to contrast with what it sits on")
        return 1
    }

    let corners = order_corners(quad)
    #: The output keeps the longest edge of each pair, so a page photographed
    #: at an angle comes out at its true proportions rather than squashed to
    #: the shorter, further edge.
    let width = int(max(distance(corners[0], corners[1]), distance(corners[3], corners[2])))
    let height = int(max(distance(corners[0], corners[3]), distance(corners[1], corners[2])))
    if width < 8 or height < 8 {
        print("the quadrilateral found is too small to be a page")
        return 1
    }

    let transform = cv.get_perspective_transform(
        corners,
        [
            Point2f(0.0, 0.0),
            Point2f(float(width - 1), 0.0),
            Point2f(float(width - 1), float(height - 1)),
            Point2f(0.0, float(height - 1)),
        ]
    )
    let flat = cv.warp_perspective(photo, transform, Size(width, height))

    #: Adaptive rather than a single threshold: a photograph is lit unevenly,
    #: and one global cut turns the dim corner solid black.
    let flat_grey = cv.cvt_color(flat, cv.COLOR_BGR2GRAY)
    let scan = cv.adaptive_threshold(
        flat_grey,
        255.0,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY,
        25,
        12.0
    )

    print(
        f"page {width}x{height} from {photo.cols}x{photo.rows} in {clock.elapsed().to_seconds() * 1000.0:.0f}ms"
    )
    print(f"corners: {corners[0]} {corners[1]} {corners[2]} {corners[3]}")
    cv.imwrite(target, scan)
    print(f"wrote {target}")

    #: The outline drawn on the original, so a wrong crop is obvious rather
    #: than mysterious.
    var marked = photo.clone()
    cv.polylines(marked, [quad], true, Scalar(60.0, 220.0, 90.0), 3)
    for index in 0..corners.len() {
        cv.circle(
            marked,
            Point(int(corners[index].x), int(corners[index].y)),
            8,
            Scalar(40.0, 40.0, 240.0),
            -1
        )
        cv.put_text(
            marked,
            f"{index}",
            Point(int(corners[index].x) + 12, int(corners[index].y)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            Scalar(40.0, 40.0, 240.0),
            2
        )
    }
    cv.imwrite("scan_outline.png", marked)
    print("wrote scan_outline.png")
    return 0
}
