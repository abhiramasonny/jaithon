#: `convex_hull` checked against a hull found a completely different way.
#:
#: The reference below is a gift wrap -- Jarvis's march -- which shares no step
#: with the column buckets and monotone chains `convex_hull` uses: it never
#: sorts, never buckets, and finds each corner by asking every remaining point
#: which of them turns furthest. The two are required to agree on the ring
#: exactly, corner for corner and in the same rotational order, not merely on
#: the set of corners or on the area.
#:
#: The cases below are chosen for the seams. `convex_hull` packs a point's
#: index into the same integer as its y and buckets by column, so it hands
#: shapes that break either of those back to the sorting path, and every one of
#: those hand-offs is exercised here: a column of points, a spread too wide to
#: bucket, coordinates too large to pack. The buckets and both chain stacks are
#: kept between calls, so a run of differing shapes is checked to leave no
#: trace of one call in the next.
import std.math
from std.test import assert_eq, assert_true
from jaicv.core.types import Point
from jaicv.imgproc.shape import convex_hull, HULL_COORD_LIMIT

#: The distinct points, in the order they were first seen.
fn distinct(points: list[Point]) -> list[Point] {
    var pool: list[Point] = []
    for point in points {
        var seen = false
        for held in pool {
            if held.x == point.x and held.y == point.y { seen = true }
        }
        if seen { continue }
        pool.push(point)
    }
    return pool
}

#: Two points have no rotational order for a wrap to find, and OpenCV does not
#: take one from the input: `convexHull` returns a degenerate hull descending
#: by x and then by y whichever way round the pair arrives. That is also the
#: direction its counter-clockwise hull of a collinear cloud comes back in, so
#: the reference has to say the same thing or it disagrees with OpenCV rather
#: than with `convex_hull`.
fn degenerate(pool: list[Point]) -> list[Point] {
    var out: list[Point] = []
    for point in pool { out.push(point) }
    for _pass in 0..out.len() {
        for index in 1..out.len() {
            let before = out[index - 1]
            let here = out[index]
            let swap = here.x > before.x or here.x == before.x and here.y > before.y
            if swap {
                out[index - 1] = here
                out[index] = before
            }
        }
    }
    return out
}

#: The hull by gift wrapping: from the leftmost point, take the point every
#: other point lies to one side of, and go on from there.
#:
#: Collinear points are dropped by preferring the furthest of the ones that tie
#: on the turn, which leaves corners only -- the same thing `convex_hull`'s
#: chains do by popping on a zero cross product.
fn wrapped(points: list[Point]) -> list[Point] {
    let pool = distinct(points)
    let count = pool.len()
    if count < 3 { return degenerate(pool) }
    var start = 0
    for index in 0..count {
        let here = pool[index]
        let best = pool[start]
        if here.x < best.x {
            start = index
        } elif here.x == best.x and here.y < best.y {
            start = index
        }
    }
    var ring: list[Point] = []
    var current = start
    for _step in 0..count + 1 {
        ring.push(pool[current])
        var next = (current + 1) % count
        for index in 0..count {
            if index == current { continue }
            let origin = pool[current]
            let held = pool[next]
            let rival = pool[index]
            let turn = (held.x - origin.x) * (rival.y - origin.y) -
                (held.y - origin.y) * (rival.x - origin.x)
            if turn < 0 {
                next = index
            } elif turn == 0 {
                let far = (held.x - origin.x) * (held.x - origin.x) +
                    (held.y - origin.y) * (held.y - origin.y)
                let further = (rival.x - origin.x) * (rival.x - origin.x) +
                    (rival.y - origin.y) * (rival.y - origin.y)
                if further > far { next = index }
            }
        }
        current = next
        if current == start { break }
    }
    if ring.len() < 3 { return degenerate(ring) }
    return ring
}

fn twice_area(ring: list[Point]) -> int {
    let count = ring.len()
    var total = 0
    for index in 0..count {
        let here = ring[index]
        let after = ring[(index + 1) % count]
        total += here.x * after.y - after.x * here.y
    }
    return total
}

fn reversed_ring(ring: list[Point]) -> list[Point] {
    var out: list[Point] = []
    for index in 0..ring.len() { out.push(ring[ring.len() - 1 - index]) }
    return out
}

#: The same ring rotated so that its lowest corner comes first, which lets two
#: rings that start at different corners be compared corner for corner.
fn from_lowest(ring: list[Point]) -> list[Point] {
    let count = ring.len()
    var first = 0
    for index in 0..count {
        let here = ring[index]
        let best = ring[first]
        if here.x < best.x {
            first = index
        } elif here.x == best.x and here.y < best.y {
            first = index
        }
    }
    var out: list[Point] = []
    for index in 0..count { out.push(ring[(first + index) % count]) }
    return out
}

fn same_ring(found: list[Point], wanted: list[Point]) -> bool {
    if found.len() != wanted.len() { return false }
    if found.len() < 3 {
        for index in 0..found.len() {
            if found[index].x != wanted[index].x { return false }
            if found[index].y != wanted[index].y { return false }
        }
        return true
    }
    var turned = wanted
    if twice_area(found) < 0 and twice_area(wanted) > 0 { turned = reversed_ring(wanted) }
    if twice_area(found) > 0 and twice_area(wanted) < 0 { turned = reversed_ring(wanted) }
    let mine = from_lowest(found)
    let theirs = from_lowest(turned)
    for index in 0..mine.len() {
        if mine[index].x != theirs[index].x { return false }
        if mine[index].y != theirs[index].y { return false }
    }
    return true
}

fn agrees(points: list[Point]) -> bool {
    let found = convex_hull(points)
    if not same_ring(found, wrapped(points)) { return false }

    #: Every corner is one of the points given, and every point given is on the
    #: hull or inside it. Neither follows from agreeing with the gift wrap --
    #: both would survive the two implementations sharing a mistake -- so they
    #: are asked separately.
    for corner in found {
        var known = false
        for point in points {
            if point.x == corner.x and point.y == corner.y { known = true }
        }
        if not known { return false }
    }
    let count = found.len()
    if count < 3 { return true }
    let sign = twice_area(found)
    for point in points {
        for index in 0..count {
            let here = found[index]
            let after = found[(index + 1) % count]
            let side = (after.x - here.x) * (point.y - here.y) -
                (after.y - here.y) * (point.x - here.x)
            if sign > 0 and side < 0 { return false }
            if sign < 0 and side > 0 { return false }
        }
    }
    return true
}

fn blob(seed: int, corners: int, spread: int, height: int) -> list[Point] {
    var state = seed
    var points: list[Point] = []
    for _index in 0..corners {
        state = (state * 1103515245 + 12345) % 2147483648
        let x = state % spread
        state = (state * 1103515245 + 12345) % 2147483648
        let y = state % height
        points.push(Point(x, y))
    }
    return points
}

fn test_the_hull_matches_a_gift_wrap_on_random_clouds() -> void {
    for trial in 0..120 {
        let points = blob(trial * 7919 + 3, 4 + trial % 40, 3 + trial % 37, 2 + trial % 29)
        assert_true(agrees(points))
    }
}

fn test_the_hull_matches_a_gift_wrap_on_dense_clouds() -> void {
    #: Narrow enough that most columns hold several points, which is what makes
    #: the column buckets drop the ones between a column's two extremes.
    for trial in 0..60 {
        let points = blob(trial * 104729 + 11, 60 + trial, 5 + trial % 9, 400)
        assert_true(agrees(points))
    }
}

fn test_the_hull_matches_a_gift_wrap_on_a_ring_of_pixels() -> void {
    for radius in 3..40 {
        var points: list[Point] = []
        for step in 0..radius * 8 {
            let angle = 6.283185307179586 * float(step) / float(radius * 8)
            let x = int(float(radius) * math.cos(angle))
            let y = int(float(radius) * math.sin(angle))
            points.push(Point(x + 500, y - 200))
        }
        assert_true(agrees(points))
    }
}

fn test_the_hull_matches_a_gift_wrap_where_the_buckets_are_refused() -> void {
    #: A column of points, which has no width to bucket by.
    assert_true(agrees([Point(4, 1), Point(4, 9), Point(4, 5), Point(4, 3)]))
    assert_true(agrees([Point(4, 1), Point(4, 9)]))
    assert_true(agrees([Point(4, 7), Point(4, 7), Point(4, 7)]))
    #: Two columns, which is the fewest that can carry a ring.
    assert_true(agrees([Point(0, 0), Point(0, 4), Point(6, 2)]))
    assert_true(agrees([Point(0, 0), Point(0, 4), Point(6, 0), Point(6, 4)]))
    assert_true(agrees([Point(0, 0), Point(6, 3)]))
    #: Far more columns than points, which would cost more to bucket than to
    #: sort.
    assert_true(agrees([Point(0, 0), Point(500000, 3), Point(250000, 900000)]))
    #: A y too large to share a word with an index. One past the limit, not
    #: far past it: the wrap this is checked against multiplies two coordinates
    #: together for its turn test, and at nine million million that product
    #: leaves a sixty-four bit integer and the reference raises before
    #: `convex_hull` is ever asked anything. Just over the limit refuses the
    #: packing all the same, which is the whole point of the case.
    let huge = HULL_COORD_LIMIT + 1
    assert_true(agrees([Point(0, huge), Point(9, 0 - huge), Point(4, 7), Point(2, 1)]))
}

fn test_the_hull_matches_a_gift_wrap_on_collinear_and_repeated_points() -> void {
    assert_true(agrees([Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3)]))
    assert_true(agrees([Point(0, 5), Point(1, 5), Point(2, 5), Point(3, 5)]))
    assert_true(agrees([Point(3, 3), Point(3, 3), Point(3, 3)]))
    assert_true(agrees([Point(0, 0), Point(2, 0), Point(1, 0), Point(1, 0)]))
    assert_true(agrees([Point(7, 2)]))
    assert_true(agrees([]))
}

fn test_the_hull_turns_the_other_way_when_asked() -> void {
    let points = blob(20250824, 40, 60, 60)
    let ring = convex_hull(points)
    let turned = convex_hull(points, true)
    assert_eq(turned.len(), ring.len())
    assert_eq(turned[0].x, ring[0].x)
    assert_eq(turned[0].y, ring[0].y)
    for index in 1..ring.len() {
        assert_eq(turned[index].x, ring[ring.len() - index].x)
        assert_eq(turned[index].y, ring[ring.len() - index].y)
    }
}

fn test_one_shape_leaves_no_trace_in_the_next() -> void {
    #: The buckets and both chain stacks outlive the call that used them, so a
    #: shape is asked for again after wider, narrower and refused shapes have
    #: been through them, and has to come back the same both times.
    let first = blob(991, 50, 40, 40)
    let wanted = convex_hull(first)
    let others = [
        blob(17, 300, 900, 900),
        blob(18, 4, 3, 3),
        [Point(4, 1), Point(4, 9), Point(4, 5)],
        [Point(0, 0), Point(500000, 3), Point(250000, 900000)],
        blob(19, 120, 7, 500),
    ]
    for other in others {
        let _ = convex_hull(other)
        let again = convex_hull(first)
        assert_eq(again.len(), wanted.len())
        for index in 0..wanted.len() {
            assert_eq(again[index].x, wanted[index].x)
            assert_eq(again[index].y, wanted[index].y)
        }
    }
}
