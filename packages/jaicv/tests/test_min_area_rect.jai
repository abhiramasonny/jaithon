#: `min_area_rect` checked against the smallest box found by brute force.
#:
#: The reference below shares no step with the calipers. It takes every
#: ordered pair of distinct points as a direction, projects every point onto
#: it, and keeps the smallest area any of them enclose: no hull, no sort, no
#: tie-break, and cubic work rather than linear. It is exact rather than
#: approximate, because a minimal box always has a side flush with a hull edge
#: and every hull edge is one of the pairs it tries.
#:
#: What is required to agree is the area, and separately that the box the
#: operation reports really does hold every point. Not the corners: two boxes
#: of the same area can both be minimal -- a square has four -- so demanding
#: the same corners would be demanding the same tie-break, which is a choice
#: rather than a property of the answer.
#:
#: The containment test reads the box the way its own docstring defines it,
#: through the centre, the angle and the two sides, so it checks the triple
#: that comes back rather than some intermediate the operation happened to
#: keep.
#:
#: The cases are the seams of the hull underneath: a shape narrow enough to
#: bucket by column, one that touches every column of a 1080p frame and used
#: to be handed to the sort instead, coordinates too large to pack into a
#: bucket, collinear points, repeated points, and fewer points than a hull
#: needs.
import std.math
from std.test import assert_true
from jaicv.core.types import Point, RotatedRect
from jaicv.imgproc.shape import min_area_rect

#: The smallest area any direction through two of the points encloses.
fn brute_area(points: list[Point]) -> float {
    let count = points.len()
    var best = 1e300
    for first in 0..count {
        for second in 0..count {
            let dx = float(points[second].x - points[first].x)
            let dy = float(points[second].y - points[first].y)
            let length = math.sqrt(dx * dx + dy * dy)
            if length <= 0.0 { continue }
            let ux = dx / length
            let uy = dy / length
            var min_u = 1e300
            var max_u = -1e300
            var min_v = 1e300
            var max_v = -1e300
            for at in 0..count {
                let cx = float(points[at].x)
                let cy = float(points[at].y)
                let u = cx * ux + cy * uy
                let v = cy * ux - cx * uy
                if u < min_u { min_u = u }
                if u > max_u { max_u = u }
                if v < min_v { min_v = v }
                if v > max_v { max_v = v }
            }
            let area = (max_u - min_u) * (max_v - min_v)
            if area < best { best = area }
        }
    }
    return best
}

#: Whether every point is within the box, read off the box's own centre,
#: angle and sides.
fn holds_them_all(box: RotatedRect, points: list[Point], slack: float) -> bool {
    let radians = box.angle * math.PI / 180.0
    let ux = math.cos(radians)
    let uy = math.sin(radians)
    let half_width = box.size.width * 0.5 + slack
    let half_height = box.size.height * 0.5 + slack
    for point in points {
        let dx = float(point.x) - box.center.x
        let dy = float(point.y) - box.center.y
        let along = dx * ux + dy * uy
        let across = dy * ux - dx * uy
        if math.fabs(along) > half_width { return false }
        if math.fabs(across) > half_height { return false }
    }
    return true
}

#: The largest coordinate in the set, which is what the slack has to scale
#: with: a projection of a billion carries a rounding error a hundred thousand
#: times the one a projection of ten carries.
fn reach_of(points: list[Point]) -> float {
    var reach = 1.0
    for point in points {
        let x = math.fabs(float(point.x))
        let y = math.fabs(float(point.y))
        if x > reach { reach = x }
        if y > reach { reach = y }
    }
    return reach
}

fn check(points: list[Point]) -> void {
    let box = min_area_rect(points)
    let reach = reach_of(points)
    assert_true(holds_them_all(box, points, reach * 1e-9 + 1e-9))
    assert_true(box.angle >= -90.0)
    assert_true(box.angle <= 0.0)
    assert_true(box.size.width >= 0.0)
    assert_true(box.size.height >= 0.0)
    #: Zero degrees is reserved for the hull that came back a single point,
    #: which has no direction to report; everything else lands in `[-90, 0)`.
    if box.size.width > 0.0 { assert_true(box.angle < 0.0) }
    if points.len() < 3 { return }
    let want = brute_area(points)
    if want > 1e299 { return }
    let got = box.size.width * box.size.height
    assert_true(math.fabs(got - want) <= reach * reach * 1e-9 + 1e-9)
}

var _seed = 20240824

#: The high bits and not the low ones. This generator's bottom bit alternates,
#: so `% 2` on the raw state gave forty clouds that were each a single point
#: repeated -- a case worth having, but not forty of them.
fn next_random(limit: int) -> int {
    _seed = (_seed * 1103515245 + 12345) % 2147483648
    return _seed // 1024 % limit
}

fn cloud(count: int, span: int) -> list[Point] {
    var out: list[Point] = []
    for _index in 0..count { out.push(Point(next_random(span), next_random(span))) }
    return out
}

#: The border of a frame, walked the way a contour walks it, with a notch a
#: pixel deep every so often so that the hull has to throw points away.
fn frame_border(width: int, height: int, step: int) -> list[Point] {
    var out: list[Point] = []
    var x = 0
    while x < width {
        out.push(Point(x, 0))
        out.push(Point(x + 1, 1))
        x += step
    }
    var y = 0
    while y < height {
        out.push(Point(width - 1, y))
        out.push(Point(width - 2, y + 1))
        y += step
    }
    out.push(Point(width - 1, height - 1))
    out.push(Point(0, height - 1))
    return out
}

fn test_the_box_matches_a_brute_force_search_on_random_clouds() -> void {
    for _trial in 0..60 {
        let count = 3 + next_random(20)
        let span = 2 + next_random(200)
        check(cloud(count, span))
    }
}

fn test_the_box_matches_a_brute_force_search_on_dense_clouds() -> void {
    for _trial in 0..40 { check(cloud(3 + next_random(24), 2 + next_random(6))) }
}

fn test_the_box_matches_a_brute_force_search_on_a_frame_contour() -> void {
    #: The shape that sent the hull to the sorting path and cost this
    #: operation all but a microsecond of its time: a couple of dozen points
    #: spread across every column of a 1080p frame.
    check(frame_border(1920, 1080, 240))
    check(frame_border(1920, 1080, 500))
    check(frame_border(640, 480, 90))
}

fn test_the_box_matches_a_brute_force_search_where_the_buckets_are_refused() -> void {
    #: Wider than the buckets will hold, so the hull underneath goes back to
    #: sorting and this has to come out the same anyway. Not wider still: the
    #: turn test that path uses multiplies two differences, and a coordinate of
    #: a couple of billion overflows it -- a limit this operation inherits
    #: rather than one it sets.
    check([Point(0, 0), Point(500000, 3), Point(250000, 900000)])
    check([Point(-900000, -900000), Point(900000, -899999), Point(0, 900000), Point(-3, 7)])
    #: A dozen points strung across a frame, which is the span the buckets
    #: decline as not worth holding.
    var strung: list[Point] = []
    for index in 0..12 { strung.push(Point(index * 160000, index * 7 % 5)) }
    check(strung)
}

fn test_the_box_matches_a_brute_force_search_on_collinear_and_repeated_points() -> void {
    check([Point(1, 1), Point(2, 2), Point(3, 3), Point(4, 4)])
    check([Point(5, 0), Point(5, 1), Point(5, 2), Point(5, 3)])
    check([Point(0, 7), Point(9, 7), Point(4, 7)])
    check([Point(2, 3), Point(2, 3), Point(2, 3), Point(9, 4)])
    var repeated: list[Point] = []
    for index in 0..30 { repeated.push(Point(index % 3, index % 2)) }
    check(repeated)
}

fn test_the_box_of_a_square_is_the_square() -> void {
    let square = [Point(0, 0), Point(10, 0), Point(10, 10), Point(0, 10)]
    let box = min_area_rect(square)
    assert_true(math.fabs(box.size.width - 10.0) < 1e-9)
    assert_true(math.fabs(box.size.height - 10.0) < 1e-9)
    assert_true(math.fabs(box.center.x - 5.0) < 1e-9)
    assert_true(math.fabs(box.center.y - 5.0) < 1e-9)
    assert_true(math.fabs(box.angle + 90.0) < 1e-9)
    check(square)
}

fn test_a_point_and_a_segment_have_no_area() -> void {
    let alone = min_area_rect([Point(4, 9)])
    assert_true(alone.center.x == 4.0)
    assert_true(alone.center.y == 9.0)
    assert_true(alone.size.width == 0.0)
    assert_true(alone.size.height == 0.0)
    let pair = min_area_rect([Point(1, 2), Point(4, 6)])
    assert_true(math.fabs(pair.size.width * pair.size.height) < 1e-9)
    assert_true(holds_them_all(pair, [Point(1, 2), Point(4, 6)], 1e-9))
    assert_true(min_area_rect([]).size.width == 0.0)
}

fn test_one_shape_leaves_no_trace_in_the_next() -> void {
    #: The hull underneath keeps its buckets between calls, so a run of
    #: differing shapes has to leave the answer for the first one unchanged.
    let first = frame_border(1920, 1080, 300)
    let wanted = min_area_rect(first)
    let others = [
        cloud(40, 9),
        frame_border(31, 29, 3),
        [Point(0, 0), Point(500000, 3), Point(250000, 900000)],
        cloud(5, 4000),
    ]
    for other in others {
        let _ = min_area_rect(other)
        let again = min_area_rect(first)
        assert_true(again.center.x == wanted.center.x)
        assert_true(again.center.y == wanted.center.y)
        assert_true(again.size.width == wanted.size.width)
        assert_true(again.size.height == wanted.size.height)
        assert_true(again.angle == wanted.angle)
    }
}
