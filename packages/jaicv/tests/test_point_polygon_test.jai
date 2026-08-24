#: `point_polygon_test` checked twice over: against a second implementation
#: written the obvious way, and against a shape whose answer can be written
#: down.
#:
#: The first check is exact. The reference below is the algorithm as it stood
#: before the loop was rearranged for the arm64 tier -- a square root an edge,
#: a call an edge, the ray cast spelt with a comparison between two booleans --
#: and every rearrangement since was chosen to keep the last bit, so the two
#: are required to agree on the bit and not to a tolerance. A rearrangement
#: that moves one is a rearrangement that has to be argued for, and this is
#: what asks for the argument.
import std.math
from std.test import assert_eq, assert_near, assert_true
from jaicv.core.types import Point, Point2f
from jaicv.imgproc.shape import point_polygon_test

fn reference_gap(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float {
    let dx = bx - ax
    let dy = by - ay
    let squared = dx * dx + dy * dy
    if squared <= 1e-12 { return math.sqrt((px - ax) * (px - ax) + (py - ay) * (py - ay)) }
    var t = ((px - ax) * dx + (py - ay) * dy) / squared
    if t < 0.0 { t = 0.0 }
    if t > 1.0 { t = 1.0 }
    let cx = ax + t * dx
    let cy = ay + t * dy
    return math.sqrt((px - cx) * (px - cx) + (py - cy) * (py - cy))
}

#: The same question answered the plain way, one edge at a time.
fn reference_test(contour: list[Point], point: Point2f, measure_dist: bool) -> float {
    let count = contour.len()
    if count == 0 { return -1.0 }
    var inside = false
    var nearest = 1e300
    var on_edge = false

    var previous = contour[count - 1]
    for index in 0..count {
        let current = contour[index]
        let ax = float(previous.x)
        let ay = float(previous.y)
        let bx = float(current.x)
        let by = float(current.y)
        let gap = reference_gap(point.x, point.y, ax, ay, bx, by)
        if gap < nearest { nearest = gap }
        if gap <= 1e-9 { on_edge = true }
        let above_a = ay > point.y
        let above_b = by > point.y
        if above_a != above_b {
            let cut = ax + (point.y - ay) / (by - ay) * (bx - ax)
            if point.x < cut { inside = not inside }
        }
        previous = current
    }
    if on_edge { return 0.0 }
    if not measure_dist { return if inside { 1.0 } else { -1.0 } }
    return if inside { nearest } else { -nearest }
}

fn ring(count: int, radius: float, wobble: float) -> list[Point] {
    var out: list[Point] = []
    for index in 0..count {
        let angle = 6.283185307179586 * float(index) / float(count)
        let span = radius + wobble * math.sin(angle * 7.0)
        out.push(Point(int(200.0 + span * math.cos(angle)), int(200.0 + span * math.sin(angle))))
    }
    return out
}

fn test_the_answer_matches_the_plain_implementation_bit_for_bit() -> void {
    let shapes = [
        ring(1128, 130.0, 30.0),
        ring(97, 80.0, 0.0),
        ring(6, 40.0, 10.0),
        [Point(10, 10), Point(60, 10), Point(60, 60), Point(10, 60)],
        [Point(0, 0), Point(20, 0), Point(20, 0), Point(20, 20), Point(0, 20)],
        [Point(5, 5), Point(5, 5), Point(5, 5)],
        [Point(3, 7)],
        [],
    ]
    var state = 20240824
    for shape in shapes {
        for _trial in 0..400 {
            state = (state * 1103515245 + 12345) % 2147483648
            let x = float(state % 5000) / 10.0 - 100.0
            state = (state * 1103515245 + 12345) % 2147483648
            let y = float(state % 5000) / 10.0 - 100.0
            let query = Point2f(x, y)
            assert_eq(point_polygon_test(shape, query, true), reference_test(shape, query, true))
            assert_eq(point_polygon_test(shape, query, false), reference_test(shape, query, false))
        }
    }
}

#: Every vertex, and every midpoint of an edge, lies on the contour.
fn test_a_point_on_the_outline_reads_as_zero() -> void {
    let shape = [Point(10, 10), Point(70, 10), Point(70, 50), Point(10, 50)]
    for index in 0..shape.len() {
        let here = shape[index]
        let next = shape[(index + 1) % shape.len()]
        assert_eq(point_polygon_test(shape, Point2f(float(here.x), float(here.y)), true), 0.0)
        let midx = float(here.x + next.x) / 2.0
        let midy = float(here.y + next.y) / 2.0
        assert_eq(point_polygon_test(shape, Point2f(midx, midy), true), 0.0)
    }
}

#: A rectangle's answer can be written down: inside, the nearest of the four
#: sides; outside, the distance to the rectangle itself.
fn test_a_rectangle_agrees_with_the_distance_written_down() -> void {
    let x0 = 10.0
    let y0 = 20.0
    let x1 = 90.0
    let y1 = 60.0
    let shape = [Point(10, 20), Point(90, 20), Point(90, 60), Point(10, 60)]
    var state = 7
    for _trial in 0..500 {
        state = (state * 1103515245 + 12345) % 2147483648
        let px = float(state % 1400) / 10.0 - 20.5
        state = (state * 1103515245 + 12345) % 2147483648
        let py = float(state % 1400) / 10.0 - 20.5

        var outx = 0.0
        if px < x0 { outx = x0 - px }
        if px > x1 { outx = px - x1 }
        var outy = 0.0
        if py < y0 { outy = y0 - py }
        if py > y1 { outy = py - y1 }

        var wanted = 0.0
        if outx > 0.0 or outy > 0.0 {
            wanted = -math.sqrt(outx * outx + outy * outy)
        } else {
            var inner = px - x0
            if x1 - px < inner { inner = x1 - px }
            if py - y0 < inner { inner = py - y0 }
            if y1 - py < inner { inner = y1 - py }
            wanted = inner
        }
        assert_near(point_polygon_test(shape, Point2f(px, py), true), wanted, 1e-12)
    }
}

fn test_the_sign_says_which_side_the_point_is_on() -> void {
    let shape = ring(64, 50.0, 0.0)
    assert_eq(point_polygon_test(shape, Point2f(200.0, 200.0), false), 1.0)
    assert_eq(point_polygon_test(shape, Point2f(400.0, 400.0), false), -1.0)
    assert_true(point_polygon_test(shape, Point2f(200.0, 200.0), true) > 0.0)
    assert_true(point_polygon_test(shape, Point2f(400.0, 400.0), true) < 0.0)
}
