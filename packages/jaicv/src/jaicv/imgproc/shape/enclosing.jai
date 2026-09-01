#: The smallest rotated rectangle and the smallest circle over a set of points.
import std.math
from ...core.types import Point, Point2f, RotatedRect, Size2f
from .hull import convex_hull

fn normalised_angle(degrees: float) -> float {
    var angle = degrees
    while angle >= 0.0 { angle -= 90.0 }
    while angle < -90.0 { angle += 90.0 }
    return angle
}

#: pi and the arc tangent, read once at import rather than through the module
#: on every call.
#:
#: `math.PI` is a property read on a module object and not a constant the
#: compiler folds, and `math.atan2(...)` is that read followed by the call.
#: Measured on this machine: 0.35 microseconds for the property, 0.48 for the
#: qualified call against 0.15 for the same function imported by name. The four
#: qualified spellings `min_area_rect` used to hold -- `atan2`, `cos`, `sin`
#: and `PI` -- came to 1.7 microseconds of the 7.3 it took. `BOX_PI` is the
#: same double `math.PI` is, so nothing it divides moves by a bit.
let BOX_PI = math.PI

#: The hull as floats and the winning edge's box, kept between calls.
#:
#: Not held to save the copying but to save the allocation: three lists a call
#: is three allocations for a hull of five corners, and the operation's whole
#: budget is a couple of microseconds. They only ever grow. Nothing outside a
#: single `min_area_rect` call ever sees them and the call makes no call that
#: could re-enter it, which is the same bargain `convex_hull` strikes with its
#: buckets a few hundred lines up.
var _box_hx: list[float] = []
var _box_hy: list[float] = []
var _box_found: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

#: The convex hull of a contour as two lists of floats, which is the shape the
#: calipers below walk.
#:
#: The two lists arrive as parameters and outlive the call; this fills the
#: first `count` of each and leaves whatever a longer hull left behind, which
#: is why the count is returned rather than read off a length.
#:
#: `convex_hull` and nothing of its own. This used to bucket the points by
#: column here, in seven functions that were a second copy of the ones behind
#: `convex_hull`, and the copy went stale: it refused a span of more than four
#: columns a point, so the 74-point contour around a 1080p frame -- which
#: touches every column of it -- went to the sorting path instead. That is
#: where all but a microsecond of this operation used to go. The buckets behind
#: `convex_hull` are kept between calls and indexed by block, and they take
#: that contour by column, so there is one hull left and one bench row that
#: says whether it is still fast.
@trace
fn hull_floats(points: list[Point], hx: list[float], hy: list[float]) -> int {
    let ring = convex_hull(points)
    let count = ring.len()
    while hx.len() < count {
        hx.push(0.0)
        hy.push(0.0)
    }
    for index in 0..count {
        let corner = ring[index]
        hx[index] = float(corner.x)
        hy[index] = float(corner.y)
    }
    return count
}

#: The hull edge whose box is smallest, and that box's reach.
#:
#: Nothing in here is a call, on purpose. `sqrt`, `atan2` and `fabs` all stayed
#: behind in the caller: one call anywhere in a body makes the tier decline the
#: whole of it, and with them here the walk over the hull ran interpreted and
#: cost more than the hull itself did.
#:
#: Being call-free was not enough, and for a long time this was declined
#: anyway: three spellings in it were each on their own a reason to refuse the
#: body, so the walk stayed interpreted and cost 5.2 microseconds of the 7.6 a
#: five-corner hull took, against 0.2 once the tier took it. `JAI_JIT_WHY=1`
#: names the opcode each time. The three were these.
#:
#: The projection loop held ten live floats -- two coordinates, two components
#: of the direction, the two projections and the four reaches -- and the tier
#: has eight FP registers. Two of them spilled, and an operand that has spilled
#: is a refusal under a guarded compare rather than a reload. Reading the two
#: coordinates out of the lists where they are used drops the loop to eight and
#: it compiles. `hy[at] * ux - hx[at] * uy` is `-cx * uy + cy * ux` to the bit:
#: a subtraction IS the addition of a negation in IEEE, and addition commutes.
#:
#: `if dx <= 0.0 and dy > 0.0` was refused for a float borrow reaching the
#: second comparison's guard, so the quadrant fold is nested ifs; and `if not
#: take and area < best_area + 1e-9` for the same reason, so the flag is an
#: integer and the two tests are an if and an elif.
#:
#: `out` comes back as the four reaches and then the direction of the winning
#: edge -- the direction and not its angle, because the caller wants `atan2` of
#: it once rather than once an edge. The winner goes there rather than into six
#: more locals for the same reason the projection loop lost its two: a local
#: here is a scarce thing.
#:
#: A square has two minimal boxes and a nearly square one has two that tie
#: numerically, so the tie is broken on the angle: the edge lying closest to
#: the horizontal wins, which is the one OpenCV reports. That is held as a
#: direction folded into the first quadrant rather than as an angle. The angle
#: was only ever wanted for the comparison, and it was taken modulo a quarter
#: turn before being compared -- which is what the folding does. What is left
#: is a comparison of two slopes, `fb / fa` against the best so far, and
#: cross-multiplying it gives whole numbers small enough to be exact. A slope
#: of minus one stands for no edge yet.
fn best_box(hx: list[float], hy: list[float], count: int, out: list[float]) -> void {
    var best_area = 1e300
    var best_fa = 1.0
    var best_fb = -1.0
    out[0] = 0.0
    out[1] = 0.0
    out[2] = 0.0
    out[3] = 0.0
    out[4] = 0.0
    out[5] = 0.0

    for index in 0..count {
        var after = index + 1
        if after == count { after = 0 }
        let dx = hx[after] - hx[index]
        let dy = hy[after] - hy[index]
        #: `** 0.5` and not `sqrt`, which is a call: the two agree for every
        #: argument this can produce, since the sum of two squares is never
        #: negative and never a negative zero.
        let length = (dx * dx + dy * dy) ** 0.5
        if length <= 1e-12 { continue }
        let ux = dx / length
        let uy = dy / length

        var min_u = 1e300
        var max_u = -1e300
        var min_v = 1e300
        var max_v = -1e300
        for at in 0..count {
            let u = hx[at] * ux + hy[at] * uy
            if u < min_u { min_u = u }
            if u > max_u { max_u = u }
            let v = hy[at] * ux - hx[at] * uy
            if v < min_v { min_v = v }
            if v > max_v { max_v = v }
        }
        let area = (max_u - min_u) * (max_v - min_v)

        var fa = dx
        var fb = dy
        if dx <= 0.0 {
            if dy > 0.0 {
                fa = dy
                fb = -dx
            } else {
                fa = -dx
                fb = -dy
            }
        } else {
            if dy < 0.0 {
                fa = -dy
                fb = dx
            }
        }
        var take = 0
        if area < best_area - 1e-9 {
            take = 1
        } elif area < best_area + 1e-9 {
            if fb * best_fa > best_fb * fa { take = 1 }
        }
        if take == 1 {
            if area < best_area { best_area = area }
            best_fa = fa
            best_fb = fb
            out[0] = min_u
            out[1] = max_u
            out[2] = min_v
            out[3] = max_v
            out[4] = dx
            out[5] = dy
        }
    }
}

#: The smallest rotated rectangle containing every point.
#:
#: Rotating calipers over the convex hull: every minimal box has a side flush
#: with a hull edge, so trying each edge in turn is enough.
pub fn min_area_rect(points: list[Point]) -> RotatedRect {
    var hx: list[float] = []
    var hy: list[float] = []
    let count = hull_floats(points, hx, hy)
    if count == 0 { return RotatedRect(Point2f(0.0, 0.0), Size2f(0.0, 0.0), 0.0) }
    if count == 1 { return RotatedRect(Point2f(hx[0], hy[0]), Size2f(0.0, 0.0), 0.0) }

    var found = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_box(hx, hy, count, found)
    let best_min_u = found[0]
    let best_max_u = found[1]
    let best_min_v = found[2]
    let best_max_v = found[3]
    let best_angle = math.atan2(found[5], found[4])

    let ux = math.cos(best_angle)
    let uy = math.sin(best_angle)
    let cu = (best_min_u + best_max_u) * 0.5
    let cv = (best_min_v + best_max_v) * 0.5
    let centre = Point2f(cu * ux - cv * uy, cu * uy + cv * ux)
    var width = best_max_u - best_min_u
    var height = best_max_v - best_min_v
    #: OpenCV measures the angle with `y` pointing up and reports it in
    #: `[-90, 0)`, rotating by a quarter turn and swapping the sides until it
    #: lands there, so one box always comes back the one way.
    var angle = best_angle * 180.0 / math.PI
    while angle >= 0.0 {
        angle -= 90.0
        let swap = width
        width = height
        height = swap
    }
    while angle < -90.0 {
        angle += 90.0
        let swap = width
        width = height
        height = swap
    }
    return RotatedRect(centre, Size2f(width, height), angle)
}

#: The four corners of a rotated rectangle.
pub fn box_points(box: RotatedRect) -> list[Point2f] {
    let radians = box.angle * math.PI / 180.0
    let ux = math.cos(radians) * 0.5
    let uy = math.sin(radians) * 0.5
    let w = box.size.width
    let h = box.size.height
    let px = ux * w
    let py = uy * w
    let qx = -uy * h
    let qy = ux * h
    return [
        Point2f(box.center.x - px - qx, box.center.y - py - qy),
        Point2f(box.center.x + px - qx, box.center.y + py - qy),
        Point2f(box.center.x + px + qx, box.center.y + py + qy),
        Point2f(box.center.x - px + qx, box.center.y - py + qy),
    ]
}

#: Fisher-Yates over a copy of the contour, from a fixed generator.
#:
#: The walk below is linear in expectation only over a random order, and a
#: contour arrives in boundary order, which is close to the worst one: the
#: points march around the outside, so each new one tends to fall outside the
#: circle its predecessors made and every step descends all three levels. The
#: walk makes 452 tests on a shuffled 452-point disk and 19144 on the same one
#: in contour order; on a 1080-point triangle it is 1126 against 7.4 million.
#:
#: The generator is fixed, not seeded from the clock, so that two runs on the
#: same contour return the same circle. Its constants are the ones from the C
#: standard's sample `rand`; nothing here needs more than that.
fn shuffle_points(order: list[Point], count: int) -> int {
    var state = 1
    var index = count - 1
    while index > 0 {
        state = (state * 1103515245 + 12345) % 2147483648
        let pick = state % (index + 1)
        let hold = order[index]
        order[index] = order[pick]
        order[pick] = hold
        index -= 1
    }
    return count
}

#: Move a widely separated pair to the front.
#:
#: The walk starts from the circle on the first two points and only descends
#: when a later one falls outside it, so opening with a pair that is nearly a
#: diameter of the answer means almost nothing descends. The pair is the wider
#: of the two extreme spans, horizontal and vertical: one pass and four running
#: maxima.
#:
#: On a 452-point disk this takes the walk from 1273 tests and 183 descents to
#: 450 tests and none, 9.8 us down to 0.9 us, and it is not a special case of
#: round shapes -- a 1080-point triangle goes from 11185 tests to 1126. It does
#: not replace the shuffle: extremes first on an unshuffled contour still runs
#: 35913 tests on a nine-lobed blob, against 3696 with both.
fn lead_with_extremes(order: list[Point], count: int) -> int {
    var min_x = order[0].x
    var max_x = min_x
    var min_y = order[0].y
    var max_y = min_y
    var left = 0
    var right = 0
    var top = 0
    var bottom = 0
    var index = 1
    while index < count {
        let point = order[index]
        let x = point.x
        let y = point.y
        if x < min_x {
            min_x = x
            left = index
        }
        if x > max_x {
            max_x = x
            right = index
        }
        if y < min_y {
            min_y = y
            top = index
        }
        if y > max_y {
            max_y = y
            bottom = index
        }
        index += 1
    }

    let wide = float(max_x - min_x)
    let tall = float(max_y - min_y)
    let across_y = float(order[right].y - order[left].y)
    let across_x = float(order[bottom].x - order[top].x)
    var first = left
    var second = right
    if tall * tall + across_x * across_x > wide * wide + across_y * across_y {
        first = top
        second = bottom
    }
    #: Every point is the same one, so no pair is wider than any other and the
    #: order the walk sees cannot matter.
    if first == second { return count }

    let head = order[0]
    order[0] = order[first]
    order[first] = head
    #: The point bound for slot one was sitting in slot zero, and the swap
    #: above has just moved it to where slot zero's occupant came from.
    if second == 0 { second = first }
    let next = order[1]
    order[1] = order[second]
    order[second] = next
    return count
}

#: Welzl's minidisk over `order`, as three nested forward walks.
#:
#: Hold a circle over the points seen so far. When the next one falls outside
#: it, that point has to sit on the boundary of the answer, so rebuild the
#: circle with it pinned and walk the points behind it again; pinning a second
#: point descends one level further, and a third fixes the circle outright.
#: Over a random order a point ends up pinned with probability 3/i, so the
#: descents thin out and the expected cost is linear. None of the three walks
#: ever restarts: the circle a level returns already holds everything before
#: the point that forced it.
#:
#: Written out flat, with the circumcircle inline and no helper anywhere,
#: because the arm64 tier declines a whole body over a single call and this is
#: where all the arithmetic is. Squared distances throughout for the same
#: reason -- `math.sqrt` is an OP_INVOKE -- and the root is taken once by the
#: caller. Flat and compiled this runs 0.9 us on a 452-point contour; split
#: into `span`/`circumcircle`/`level` helpers, which reads far better, the tier
#: took only the scans and it ran 106 us.
#:
#: The slack on `limit` is spelt as two literals rather than as module
#: constants because comparing against a module-level `let` is an
#: OP_JUMP_IF_CMP_FALSE the tier will not take, and that one comparison
#: declined this entire function. It is there because a point the walk has
#: pinned has to read as inside its own circle, and after rounding it sometimes
#: does not; without it the walk fits a circle to a point already on the
#: boundary.
#:
#: Returns the squared radius and writes the centre into `out`, rather than a
#: tuple, so that nothing here allocates.
fn welzl_walk(order: list[Point], count: int, out: list[float]) -> float {
    let head = order[0]
    let x0 = float(head.x)
    let y0 = float(head.y)
    let second = order[1]
    var cx = (x0 + float(second.x)) * 0.5
    var cy = (y0 + float(second.y)) * 0.5
    var r2 = (x0 - cx) * (x0 - cx) + (y0 - cy) * (y0 - cy)
    var limit = r2 + r2 * 1e-12 + 1e-12
    var i = 2
    while i < count {
        let outer = order[i]
        let ax = float(outer.x)
        let ay = float(outer.y)
        let ox = ax - cx
        let oy = ay - cy
        if ox * ox + oy * oy <= limit {
            i += 1
            continue
        }
        cx = (ax + x0) * 0.5
        cy = (ay + y0) * 0.5
        r2 = (ax - cx) * (ax - cx) + (ay - cy) * (ay - cy)
        limit = r2 + r2 * 1e-12 + 1e-12
        var j = 1
        while j < i {
            let middle = order[j]
            let bx = float(middle.x)
            let by = float(middle.y)
            let mx = bx - cx
            let my = by - cy
            if mx * mx + my * my <= limit {
                j += 1
                continue
            }
            cx = (ax + bx) * 0.5
            cy = (ay + by) * 0.5
            r2 = (ax - cx) * (ax - cx) + (ay - cy) * (ay - cy)
            limit = r2 + r2 * 1e-12 + 1e-12
            var k = 0
            while k < j {
                let inner = order[k]
                let px = float(inner.x)
                let py = float(inner.y)
                let ix = px - cx
                let iy = py - cy
                if ix * ix + iy * iy <= limit {
                    k += 1
                    continue
                }
                let d = 2.0 * (ax * (by - py) + bx * (py - ay) + px * (ay - by))
                if d > -1e-12 and d < 1e-12 {
                    #: Three pinned points are never exactly collinear in exact
                    #: arithmetic, so this only fires on a repeated point or on
                    #: rounding. The smallest circle over three points on a line
                    #: is the widest pair as a diameter, and taking it keeps the
                    #: division below off a determinant that has rounded away.
                    let ab = (ax - bx) * (ax - bx) + (ay - by) * (ay - by)
                    let ap = (ax - px) * (ax - px) + (ay - py) * (ay - py)
                    let bp = (bx - px) * (bx - px) + (by - py) * (by - py)
                    if ab >= ap and ab >= bp {
                        cx = (ax + bx) * 0.5
                        cy = (ay + by) * 0.5
                        r2 = ab * 0.25
                    } elif ap >= bp {
                        cx = (ax + px) * 0.5
                        cy = (ay + py) * 0.5
                        r2 = ap * 0.25
                    } else {
                        cx = (bx + px) * 0.5
                        cy = (by + py) * 0.5
                        r2 = bp * 0.25
                    }
                } else {
                    let a2 = ax * ax + ay * ay
                    let b2 = bx * bx + by * by
                    let p2 = px * px + py * py
                    cx = (a2 * (by - py) + b2 * (py - ay) + p2 * (ay - by)) / d
                    cy = (a2 * (px - bx) + b2 * (ax - px) + p2 * (bx - ax)) / d
                    r2 = (ax - cx) * (ax - cx) + (ay - cy) * (ay - cy)
                }
                limit = r2 + r2 * 1e-12 + 1e-12
                k += 1
            }
            j += 1
        }
        i += 1
    }
    out[0] = cx
    out[1] = cy
    return r2
}

#: The squared distance from `(cx, cy)` to the farthest of the first `stop`
#: points. Squared, and rooted by the caller, so this body stays call-free too.
fn farthest_squared(order: list[Point], stop: int, cx: float, cy: float) -> float {
    var worst = 0.0
    var index = 0
    while index < stop {
        let point = order[index]
        let dx = float(point.x) - cx
        let dy = float(point.y) - cy
        let squared = dx * dx + dy * dy
        if squared > worst { worst = squared }
        index += 1
    }
    return worst
}

#: The smallest circle containing every point.
#:
#: What this replaced tried every pair and every triple of hull points and
#: checked each candidate against every point. A 60-point hull is thirty-five
#: thousand triples, and with a `math.sqrt` in the innermost line the check ran
#: interpreted: 85 ms against OpenCV's 3.4 us on a 452-point contour.
#:
#: The walk runs over a copy because it reorders what it is given, and over the
#: `Point`s themselves rather than over two float arrays: the copy is one bulk
#: operation at 1.4 us where filling two arrays a `push` at a time was 4.3, and
#: a test that reads a field off the object costs 2.3 ns against 1.9.
pub fn min_enclosing_circle(points: list[Point]) -> tuple[Point2f, float] {
    let count = points.len()
    if count == 0 { return (Point2f(0.0, 0.0), 0.0) }
    if count == 1 { return (Point2f(float(points[0].x), float(points[0].y)), 0.0) }

    var order = points.copy()
    let _mixed = shuffle_points(order, count)
    let _led = lead_with_extremes(order, count)
    var centre = [0.0, 0.0]
    var squared = welzl_walk(order, count, centre)

    #: The slack in the walk lets it finish with a point a fraction outside,
    #: and the radius the circumcircle formula reports carries its own rounding
    #: either way. The distance to the farthest point is both the tighter
    #: answer and the one that cannot leave a point uncovered, and one more
    #: linear pass is nothing next to the walk.
    let reach = farthest_squared(order, count, centre[0], centre[1])
    if reach > squared { squared = reach }
    return (Point2f(centre[0], centre[1]), math.sqrt(squared))
}
