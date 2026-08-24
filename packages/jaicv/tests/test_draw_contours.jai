#: `draw_contours` against the general drawing path it used to be.
#:
#: A stroke wider than a pixel is drawn out segment by segment inside
#: `contours.jai` rather than handed to `polylines`, because at thirty
#: thousand segments the allocations and the calls the general path makes are
#: the whole of what it costs. The two are meant to put down exactly the same
#: pixels, and nothing weaker than exactly would do: they differ in how they
#: reach the rasteriser, not in what they ask it for.
#:
#: So the comparison here is against `polylines` itself, which is a second
#: implementation of the same drawing -- its own fixed-point conversion, its
#: own rounding, its own walk of the cap circle -- and every pixel of every
#: case below has to agree. `test_against_opencv.jai` is what holds
#: `polylines` to OpenCV in turn, so agreeing with it is agreeing with OpenCV.
#:
#: Two things are checked without reference to either. The cap template is
#: compared against the disc `circle` draws on its own, and the drawn pixels
#: are checked to lie where the geometry says they must: on the segment, and
#: no further from it than half the pen plus a pixel.
import std.math
from std.test import assert_eq, assert_true
from jaicv.core.mat import Mat
from jaicv.core.types import CV_8UC1, CV_8UC3, FILLED, LINE_4, LINE_8, Point, Scalar
from jaicv.imgproc.contours import draw_contours
from jaicv.imgproc.drawing import circle, polylines

let ROWS = 96
let COLS = 128
let INK = Scalar(0.0, 255.0, 0.0)

#: A ring of points around a centre, which is the shape a contour has.
fn ring(centre_x: int, centre_y: int, radius: int, corners: int) -> list[Point] {
    var out: list[Point] = []
    for step in 0..corners {
        let angle = 2.0 * math.PI * float(step) / float(corners)
        out.push(
            Point(
                centre_x + int(float(radius) * math.cos(angle)),
                centre_y + int(float(radius) * math.sin(angle))
            )
        )
    }
    return out
}

#: The cases worth drawing: a ring, a coarse ring whose segments are long, a
#: shape that runs off two edges so the clipping is exercised, a run of
#: repeated points, a pair, and a single point.
fn cases() -> list[list[Point]] {
    var out: list[list[Point]] = []
    out.push(ring(40, 40, 18, 24))
    out.push(ring(90, 60, 25, 5))
    out.push(ring(10, 88, 30, 13))
    out.push([Point(-20, 20), Point(150, 30), Point(60, 130), Point(-5, -5)])
    out.push([Point(70, 20), Point(70, 20), Point(70, 20), Point(71, 21)])
    out.push([Point(20, 70), Point(45, 74)])
    out.push([Point(100, 12)])
    return out
}

fn blank() -> Mat { return Mat.zeros(ROWS, COLS, CV_8UC3) }

fn shifted(contours: list[list[Point]], offset: Point) -> list[list[Point]] {
    var out: list[list[Point]] = []
    for contour in contours {
        var moved: list[Point] = []
        for point in contour { moved.push(Point(point.x + offset.x, point.y + offset.y)) }
        out.push(moved)
    }
    return out
}

fn differences(left: Mat, right: Mat) -> int {
    let a = left.values()
    let b = right.values()
    var apart = 0
    for index in 0..a.len() {
        if a[index] != b[index] { apart += 1 }
    }
    return apart
}

fn lit_count(image: Mat) -> int {
    let values = image.values()
    var lit = 0
    for index in 0..values.len() {
        if values[index] != 0.0 { lit += 1 }
    }
    return lit
}

#: Every thickness and line type, against `polylines` on the same points.
fn test_a_wide_stroke_matches_the_general_path() -> void {
    let contours = cases()
    for thickness in 2..10 {
        for kind in 0..2 {
            let line_type = if kind == 0 { LINE_8 } else { LINE_4 }
            let mine = draw_contours(blank(), contours, -1, INK, thickness, line_type)
            let theirs = polylines(blank(), contours, true, INK, thickness, line_type)
            let apart = differences(mine, theirs)
            assert_eq(apart, 0, f"thickness {thickness} line {line_type} differs in {apart}")
            assert_true(lit_count(mine) > 0, "a stroke that drew nothing proves nothing")
        }
    }
}

#: The same, with the contours moved by `offset` rather than by hand, which is
#: a separate path through the arithmetic.
fn test_an_offset_matches_moving_the_points() -> void {
    let contours = cases()
    for step in 0..3 {
        let offset = [Point(0, 0), Point(7, -5), Point(-13, 11)][step]
        let mine = draw_contours(blank(), contours, -1, INK, 3, LINE_8, offset)
        let theirs = polylines(blank(), shifted(contours, offset), true, INK, 3, LINE_8)
        let apart = differences(mine, theirs)
        assert_eq(apart, 0, f"offset ({offset.x}, {offset.y}) differs in {apart}")
    }
}

#: One contour at a time, so the `index` argument is exercised as well.
fn test_one_contour_matches_the_general_path() -> void {
    let contours = cases()
    for index in 0..contours.len() {
        let mine = draw_contours(blank(), contours, index, INK, 4)
        let theirs = polylines(blank(), [contours[index]], true, INK, 4)
        assert_eq(differences(mine, theirs), 0, f"contour {index} differs")
    }
}

#: A hairline and a fill go the way they always did, and are checked here so
#: that the branch which sends them there stays honest.
fn test_the_thin_and_filled_paths_are_untouched() -> void {
    let contours = cases()
    let thin = draw_contours(blank(), contours, -1, INK, 1)
    assert_eq(
        differences(thin, polylines(blank(), contours, true, INK, 1)),
        0,
        "a hairline no longer matches polylines"
    )
    let offset = Point(6, 4)
    let moved = draw_contours(blank(), contours, -1, INK, 1, LINE_8, offset)
    assert_eq(
        differences(moved, polylines(blank(), shifted(contours, offset), true, INK, 1)),
        0,
        "an offset hairline no longer matches polylines"
    )
    assert_true(
        lit_count(draw_contours(blank(), contours, -1, INK, FILLED)) > 0,
        "a filled contour drew nothing"
    )
}

#: The cap the wide path stamps at every vertex, against the disc `circle`
#: draws on its own. A one-point contour is nothing but its caps, so drawing
#: one is drawing the template.
fn test_the_cap_is_the_disc_circle_draws() -> void {
    for thickness in 2..12 {
        let radius = (thickness << 15) + 32768 >> 16
        let mine = draw_contours(blank(), [[Point(60, 48)]], -1, INK, thickness)
        let theirs = circle(blank(), Point(60, 48), radius, INK, FILLED)
        let apart = differences(mine, theirs)
        assert_eq(apart, 0, f"the cap for thickness {thickness} differs in {apart}")
    }
}

#: Where the pixels may be, worked out from the geometry rather than from
#: either implementation: a drawn pixel is no further from some segment than
#: the round cap reaches plus a pixel for the rasterising, and every vertex is
#: drawn. The cap, not half the thickness, is what sets the bound -- it is the
#: part of the stroke that sticks out past a corner.
fn test_the_pixels_lie_along_the_contour() -> void {
    let thickness = 5
    let contour = ring(64, 48, 26, 9)
    let drawn = draw_contours(blank(), [contour], -1, INK, thickness)
    let values = drawn.values()
    let allowed = float((thickness << 15) + 32768 >> 16) + 1.0

    let count = contour.len()
    for y in 0..ROWS {
        for x in 0..COLS {
            if values[(y * COLS + x) * 3 + 1] == 0.0 { continue }
            var nearest = 1.0e30
            var previous = contour[count - 1]
            for index in 0..count {
                let current = contour[index]
                let run_x = float(current.x - previous.x)
                let run_y = float(current.y - previous.y)
                let to_x = float(x - previous.x)
                let to_y = float(y - previous.y)
                let squared = run_x * run_x + run_y * run_y
                var t = 0.0
                if squared > 0.0 { t = (to_x * run_x + to_y * run_y) / squared }
                if t < 0.0 { t = 0.0 }
                if t > 1.0 { t = 1.0 }
                let gap_x = to_x - t * run_x
                let gap_y = to_y - t * run_y
                let gap = math.sqrt(gap_x * gap_x + gap_y * gap_y)
                if gap < nearest { nearest = gap }
                previous = current
            }
            assert_true(
                nearest <= allowed,
                f"({x}, {y}) is {nearest} from the contour, further than {allowed}"
            )
        }
    }
    for point in contour {
        assert_true(
            values[(point.y * COLS + point.x) * 3 + 1] != 0.0,
            f"the vertex ({point.x}, {point.y}) was not drawn"
        )
    }
}

#: An empty list and an index past the end.
fn test_the_edges_of_the_argument() -> void {
    let empty: list[list[Point]] = []
    assert_eq(lit_count(draw_contours(blank(), empty, -1, INK, 3)), 0, "no contours drew something")
    assert_eq(
        lit_count(draw_contours(blank(), [[]], -1, INK, 3)),
        0,
        "an empty contour drew something"
    )
}
