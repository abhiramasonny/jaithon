#: Where a point falls against a contour, and the simplification of a curve.
import std.math
from ...core.types import Point, Point2f
from .moments import EPSILON

#: The nearest SQUARED gap between `(px, py)` and the closed polygon of the
#: first `count` points, signed positive when the point is inside it.
#:
#: Split out from `point_polygon_test`, squared rather than rooted, and free of
#: any call at all, because that is what it takes for the arm64 tier to accept
#: a body. One `math.sqrt` in the loop is an OP_GET_FIELD on the module and
#: `contour.len()` is an OP_INVOKE; either one on its own made the tier decline
#: the WHOLE function, and 1128 edges then walked interpreted at 400 ns each.
#:
#: Signing the square is what lets one loop answer both halves of the question
#: without a tuple. Zero is not ambiguous: a point on the boundary is `0.0`
#: whichever side of it the winding says it is on.
#:
#: The scratch names `u`, `v`, `w`, `z` are reused rather than given a name per
#: quantity, and that is deliberate. Every distinct float in the body wants one
#: of the eight FP homes the tier has; the version of this loop that spelt them
#: all out needed thirteen, and the five that missed out went to the frame and
#: took it from 6.3 ms to 50 ms on the 1000-query run. Adding a local here is
#: not free.
fn nearest_gap_squared(contour: list[Point], count: int, px: float, py: float) -> float {
    var inside = false
    var nearest = 1e300
    var ax = float(contour[count - 1].x)
    var ay = float(contour[count - 1].y)
    for index in 0..count {
        let current = contour[index]
        let bx = float(current.x)
        let by = float(current.y)
        var u = px - ax
        var v = py - ay
        var gap = u * u + v * v
        var w = bx - ax
        var z = by - ay

        #: Straddling, written as the one comparison it is.
        #:
        #: This was two nested tests with the body spelt twice, because the
        #: tier had no arm for a comparison BETWEEN two booleans and declined
        #: the whole function on one. It has one now, and the two forms measure
        #: the same: a 100000-point crossing loop, forty passes, is 7.23 ms
        #: written this way against 7.67 ms written the other, in one binary.
        #: The comment that stood here asked for exactly that measurement
        #: before the form was put back.
        #:
        #: The forms agree on every input, NaN included. Where they disagree
        #: about straddling -- `by` is NaN and `ay > py` -- the edge's `z` is
        #: NaN too, so the cut is NaN, `px < cut` is false, and neither
        #: toggles.
        if (ay > py) != (by > py) {
            if px < ax + v / z * w { inside = not inside }
        }

        #: The clamp of the projection parameter is branches rather than
        #: arithmetic, so that the divide sits in the only arm that needs one.
        #: The foot of the perpendicular lands beyond an end for all but one
        #: edge in a thousand -- 624382 short of the start and 502269 past the
        #: end against 1349 between them, counted over a million edge tests on
        #: a 1128-point contour -- so almost every edge now escapes the one
        #: instruction in this body that is not pipelined.
        #:
        #: Both arms reproduce what the clamped parameter produced, digit for
        #: digit. A dot product of zero or less is exactly the case clamped to
        #: zero, whose foot is the start and whose gap is the `u * u + v * v`
        #: computed above; one at or beyond the squared length is exactly the
        #: case clamped to one, and scaling the edge by that 1.0 is exact, so
        #: the arm below covers it unchanged.
        u = u * w + v * z
        if u > 0.0 {
            v = w * w + z * z
            if v > 1e-12 {
                if u < v { u = u / v } else { u = 1.0 }
                w = px - (ax + u * w)
                z = py - (ay + u * z)
                gap = w * w + z * z
            }
        }
        if gap < nearest { nearest = gap }
        ax = bx
        ay = by
    }
    var answer = -nearest
    if inside { answer = nearest }
    return answer
}

#: Where a point falls relative to a contour.
#:
#: With `measure_dist` off the answer is `1.0` inside, `-1.0` outside, and
#: `0.0` exactly on the edge. With it on the answer is the distance to the
#: nearest edge, positive inside and negative outside.
pub fn point_polygon_test(
    contour: list[Point],
    point: Point2f,
    measure_dist: bool = false
) -> float {
    let count = contour.len()
    if count == 0 { return -1.0 }

    #: One square root for the contour rather than one an edge. `sqrt` is
    #: monotone, so the edge holding the smallest square holds the smallest
    #: gap, and the root of the winner is the same double the per-edge root
    #: gave; and the on-edge test asked whether ANY gap fell within 1e-9,
    #: which is the same question as whether the smallest one does.
    #:
    #: A compare in place of `math.fabs`, because this half runs interpreted
    #: -- `contour.len()` is an OP_INVOKE and the tier stops at one -- and
    #: there a module read and a call cost more than the negation does.
    #: `** 0.5` was tried in place of `math.sqrt` for the same reason and put
    #: back: the interpreter reaches it through `pow`, which came out one ulp
    #: from `sqrt` (-56.697174668448994 against -56.697174668449), and
    #: `tests/test_point_polygon_test.jai` asks for the bit.
    let signed = nearest_gap_squared(contour, count, point.x, point.y)
    var square = signed
    if square < 0.0 { square = -square }
    let distance = math.sqrt(square)
    if distance <= 1e-9 { return 0.0 }
    if measure_dist {
        if signed > 0.0 { return distance }
        return -distance
    }
    if signed > 0.0 { return 1.0 }
    return -1.0
}

#: The index of the point of `curve` furthest from `curve[origin]`.
#:
#: Ties go to the earliest index, which is what puts a closed curve's cut in
#: OpenCV's rotation.
fn furthest_from(curve: list[Point], count: int, origin: int) -> int {
    let ox = float(curve[origin].x)
    let oy = float(curve[origin].y)
    var far = 0
    var furthest = -1.0
    for index in 0..count {
        let point = curve[index]
        let dx = float(point.x) - ox
        let dy = float(point.y) - oy
        let span = dx * dx + dy * dy
        if span > furthest {
            furthest = span
            far = index
        }
    }
    return far
}

fn douglas_peucker(
    curve: list[Point],
    count: int,
    start: int,
    first: int,
    last: int,
    out: list[Point],
    epsilon: float
) -> void {
    if last <= first + 1 { return }

    #: One index and one point, both declared before anything else and both
    #: still read at the bottom, rather than a `head` and a `tail` that die
    #: where the endpoints are loaded. The compiler hands a dead local's slot
    #: to the next declaration, so `head` and `tail` -- ints, dead after the
    #: two loads -- had their slots taken by the point and by `dx`; slot 8
    #: then held an int and a Point and slot 9 an int and a float, and the
    #: tier declines a body with a slot of two kinds. That decline is what
    #: cost the whole scan below: it ran interpreted at 24 ns a point.
    var here = start + first
    if here >= count { here -= count }
    var point = curve[here]
    let ax = float(point.x)
    let ay = float(point.y)
    here = start + last
    if here >= count { here -= count }
    point = curve[here]
    let dx = float(point.x) - ax
    let dy = float(point.y) - ay
    let squared = dx * dx + dy * dy
    let tiny = EPSILON

    #: Every working value the loop uses is declared before it, and none
    #: inside, for the same reason.
    var px = 0.0
    var py = 0.0
    var gap = 0.0
    var dot = 0.0
    var t = 0.0
    var cx = 0.0
    var cy = 0.0
    var worst = 0.0
    var worst_at = first
    for index in first + 1..last {
        here = start + index
        if here >= count { here -= count }
        point = curve[here]
        px = float(point.x)
        py = float(point.y)
        gap = (px - ax) * (px - ax) + (py - ay) * (py - ay)
        if squared > tiny {
            dot = (px - ax) * dx + (py - ay) * dy
            if dot > 0.0 {
                t = 1.0
                if dot < squared { t = dot / squared }
                cx = ax + t * dx
                cy = ay + t * dy
                gap = (px - cx) * (px - cx) + (py - cy) * (py - cy)
            }
        }
        if gap > worst {
            worst = gap
            worst_at = index
        }
    }
    if worst ** 0.5 <= epsilon { return }
    douglas_peucker(curve, count, start, first, worst_at, out, epsilon)
    here = start + worst_at
    if here >= count { here -= count }
    out.push(curve[here])
    douglas_peucker(curve, count, start, worst_at, last, out, epsilon)
}

#: Simplify a curve, dropping points that sit within `epsilon` of the line
#: through their neighbours.
pub fn approx_poly_dp(curve: list[Point], epsilon: float, closed: bool) -> list[Point] {
    let count = curve.len()
    if count < 3 { return curve }
    if epsilon < 0.0 { throw ValueError("an approximation tolerance cannot be negative") }

    #: A closed curve has no natural ends, so two are made: the point furthest
    #: from the first, and then the point furthest from that one. The curve is
    #: cut there and each half simplified on its own. Taking the second of the
    #: two as the start is what puts the result in OpenCV's rotation.
    var start = 0
    if closed { start = furthest_from(curve, count, furthest_from(curve, count, 0)) }

    #: A closed curve runs one point past its end, back onto the point it
    #: began at; an open one stops at the last point it was given. Either way
    #: both ends are kept, and the recursion decides everything between them.
    let last = if closed { count } else { count - 1 }
    var out: list[Point] = []
    out.push(curve[start])
    douglas_peucker(curve, count, start, 0, last, out, epsilon)
    if not closed { out.push(curve[count - 1]) }
    return out
}
