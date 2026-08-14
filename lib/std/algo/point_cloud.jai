#: Algorithms over a bare `list[Point2]`, with no ring or edge structure
#: assumed: `convex_hull`, `closest_pair`, `diameter`, `centroid`,
#: `bounding_box`, and `minimum_enclosing_circle`.
#:
#: Split out of `geometry.jai`. This group has no ties to `Line`/`Segment` or
#: to `Polygon` -- it operates on the points themselves, not on any shape
#: built from them -- so it separates cleanly. The one thing it shares with
#: `polygon.jai` is `bounds_of`, formerly the private `_bounds_of`: a
#: polygon's vertex ring is a point cloud too, and `Polygon.bounding_box`
#: reuses this file's `bounding_box` machinery rather than duplicating it, so
#: the helper was promoted to `pub` the same way `sign_within` and
#: `within_bounds` were in `geometry.jai`.
import std.math
from std.algo.sort import sort_by
from std.algo.vectors import Point2, EPSILON
from std.algo.geometry import orientation
from std.algo.shapes import Rect, Circle, circle_from_points

#: Return the vertices of the convex hull, counter-clockwise.
#:
#: Andrew's monotone chain: sort the points by x then y, sweep left to right
#: building the lower chain and right to left building the upper one, popping
#: any vertex that is not a strict left turn. O(n log n), all of it in the
#: sort. Duplicate points are removed and collinear points are dropped, so the
#: result is the minimal strictly convex ring. Fewer than three distinct
#: points, or all of them collinear, gives the one or two extreme points back
#: — a hull, just not a polygon.
#:
#: ```
#: let cloud = [Point2(0.0, 0.0), Point2(2.0, 0.0), Point2(1.0, 1.0), Point2(1.0, 0.2)]
#: let hull = convex_hull(cloud)
#: hull.len()      # 3
#: hull[0].x       # 0.0
#: ```
pub fn convex_hull(points: list[Point2]) -> list[Point2] {
    let ordered = _sorted_distinct(points)
    let count = ordered.len()
    if count <= 2 { return ordered }
    let lower: list[Point2] = []
    for point in ordered {
        _push_turning_left(lower, point)
    }
    let upper: list[Point2] = []
    var index = count - 1
    while index >= 0 {
        _push_turning_left(upper, ordered[index])
        index -= 1
    }
    lower.pop()
    upper.pop()
    return lower + upper
}

#: Return the two closest points and the distance between them.
#:
#: Divide and conquer in O(n log n): split the points at the median x, solve
#: both halves, then check only the points within the better of the two
#: distances of the dividing line. That strip is scanned in y order, where a
#: geometric packing argument caps the work at a constant number of comparisons
#: per point — no pair further than seven positions apart in y can beat the
#: distance already found. The brute-force alternative is O(n^2).
#:
#: Both index lists are sorted once up front and partitioned on the way down,
#: which is what keeps the recursion linear per level rather than re-sorting.
#: Raises `ValueError` for fewer than two points. Duplicate points are a
#: legitimate answer at distance zero.
#:
#: ```
#: let cloud = [Point2(0.0, 0.0), Point2(5.0, 5.0), Point2(1.0, 1.0)]
#: let (first, second, gap) = closest_pair(cloud)
#: first.x                                # 0.0
#: second.x                               # 1.0
#: math.approx_eq(gap, math.SQRT_2)       # true
#: ```
pub fn closest_pair(points: list[Point2]) -> tuple[Point2, Point2, float] {
    let count = points.len()
    if count < 2 {
        throw ValueError(f"closest_pair needs at least 2 points, got {count}")
    }
    let indices = [index for index in 0..count]
    let by_x = sort_by(indices, |left, right| _compare(points[left], points[right], false))
    let by_y = sort_by(indices, |left, right| _compare(points[left], points[right], true))
    let in_left = [false for _ in 0..count]
    let (first, second, distance) = _closest_pair_of(points, by_x, by_y, in_left)
    return (points[first], points[second], distance)
}

#: Return the arithmetic mean of the points.
#:
#: The centre of the *vertices*, not of the shape they outline: adding a point
#: pulls it, and subdividing an edge of a polygon moves it. For the centre of
#: mass of a filled polygon use `polygon_centroid`. This one is what a
#: clustering step or an angular sort wants.
#:
#: Raises `ValueError` for an empty list.
#:
#: ```
#: centroid([Point2(0.0, 0.0), Point2(2.0, 0.0), Point2(1.0, 3.0)]).y    # 1.0
#: ```
pub fn centroid(points: list[Point2]) -> Point2 {
    if points.len() == 0 { throw ValueError("centroid of an empty point set") }
    var total_x = 0.0
    var total_y = 0.0
    for point in points {
        total_x += point.x
        total_y += point.y
    }
    let count = float(points.len())
    return Point2(total_x / count, total_y / count)
}

#: Return the smallest axis-aligned rectangle containing every point.
#:
#: Raises `ValueError` for an empty list.
#:
#: ```
#: bounding_box([Point2(1.0, 5.0), Point2(-2.0, 0.0)]).width()    # 3.0
#: ```
pub fn bounding_box(points: list[Point2]) -> Rect { return bounds_of(points) }

#: Return the two furthest-apart points and the distance between them.
#:
#: Rotating calipers over the convex hull. The diameter is always realised by
#: two hull vertices, and specifically by an antipodal pair — two vertices
#: with parallel supporting lines. Walking one edge of the hull forward while
#: advancing the opposite vertex only as far as the triangle area keeps growing
#: enumerates every antipodal pair in a single loop, so the whole search after
#: the hull is O(h) rather than the O(h^2) of comparing all pairs.
#:
#: Raises `ValueError` for an empty list. Identical points give that point
#: twice at distance zero.
#:
#: ```
#: let cloud = [Point2(0.0, 0.0), Point2(3.0, 0.0), Point2(0.0, 4.0), Point2(1.0, 1.0)]
#: let (first, second, span) = diameter(cloud)
#: span            # 5.0
#: first.x         # 3.0
#: second.y        # 4.0
#: ```
pub fn diameter(points: list[Point2]) -> tuple[Point2, Point2, float] {
    if points.len() == 0 { throw ValueError("diameter of an empty point set") }
    let hull = convex_hull(points)
    let count = hull.len()
    if count == 1 { return (hull[0], hull[0], 0.0) }
    if count == 2 { return (hull[0], hull[1], hull[0].distance(hull[1])) }
    var best_first = hull[0]
    var best_second = hull[1]
    var best = hull[0].distance(hull[1])
    var opposite = 1
    for index in 0..count {
        let next = (index + 1) % count
        var steps = 0
        loop {
            if steps >= count { break }
            let ahead = _twice_triangle_area(hull[index], hull[next], hull[(opposite + 1) % count])
            let here = _twice_triangle_area(hull[index], hull[next], hull[opposite])
            if ahead <= here { break }
            opposite = (opposite + 1) % count
            steps += 1
        }
        let from_start = hull[index].distance(hull[opposite])
        if from_start > best {
            best = from_start
            best_first = hull[index]
            best_second = hull[opposite]
        }
        let from_end = hull[next].distance(hull[opposite])
        if from_end > best {
            best = from_end
            best_first = hull[next]
            best_second = hull[opposite]
        }
    }
    return (best_first, best_second, best)
}

#: Return the smallest circle containing every point.
#:
#: Welzl's algorithm in its incremental form. The key fact is that the answer
#: is fixed by at most three of the points, so the circle is grown one point at
#: a time and, whenever a point falls outside, rebuilt with that point pinned
#: to the boundary — recursively, to a depth of three. A point processed after
#: the circle already contains it costs one distance test, and a random order
#: makes the expensive rebuild rare enough that the whole thing is O(n)
#: expected. Processing the points in their given order is what would make it
#: quadratic, so they are permuted first; the permutation comes from a fixed
#: internal generator, so the same input always yields the same circle.
#:
#: Collinear inputs are handled: with no circumcircle to be had, the two
#: furthest-apart of the three candidate points define a diametral circle.
#:
#: Raises `ValueError` for an empty list.
#:
#: ```
#: let corners = [Point2(0.0, 0.0), Point2(2.0, 0.0), Point2(2.0, 2.0), Point2(0.0, 2.0)]
#: let circle = minimum_enclosing_circle(corners)
#: circle.center.x                              # 1.0
#: math.approx_eq(circle.radius, math.SQRT_2)   # true
#: ```
pub fn minimum_enclosing_circle(points: list[Point2]) -> Circle {
    if points.len() == 0 {
        throw ValueError("minimum_enclosing_circle of an empty point set")
    }
    let ordered = _pseudo_shuffled(points)
    var circle = Circle(ordered[0], 0.0)
    for index in 1..ordered.len() {
        if not _circle_holds(circle, ordered[index]) {
            circle = _enclosing_pinning_one(ordered, index, ordered[index])
        }
    }
    return circle
}

# --- Private helpers ---------------------------------------------------------
#: Lexicographically compare two points, `y` before `x` when `y_first`.
fn _compare(a: Point2, b: Point2, y_first: bool) -> int {
    if y_first {
        if a.y < b.y { return -1 }
        if a.y > b.y { return 1 }
        if a.x < b.x { return -1 }
        if a.x > b.x { return 1 }
        return 0
    }
    if a.x < b.x { return -1 }
    if a.x > b.x { return 1 }
    if a.y < b.y { return -1 }
    if a.y > b.y { return 1 }
    return 0
}

#: Return the points sorted by x then y, with exact duplicates removed.
fn _sorted_distinct(points: list[Point2]) -> list[Point2] {
    let ordered = sort_by(points, |a, b| _compare(a, b, false))
    let distinct: list[Point2] = []
    for point in ordered {
        if distinct.len() == 0 or not distinct[distinct.len() - 1].approx_eq(point) {
            distinct.push(point)
        }
    }
    return distinct
}

#: Append `point` to a hull chain, popping vertices that do not turn left.
fn _push_turning_left(chain: list[Point2], point: Point2) -> void {
    loop {
        if chain.len() < 2 { break }
        let last = chain.len() - 1
        if orientation(chain[last - 1], chain[last], point) > 0 { break }
        chain.pop()
    }
    chain.push(point)
}

#: Return the closest pair among the indices in `by_x`, as (i, j, distance).
#:
#: `by_x` and `by_y` hold the same indices ordered by x and by y. `in_left` is
#: scratch space shared by the whole recursion so that the partition costs one
#: pass instead of a set lookup per element.
fn _closest_pair_of(
    points: list[Point2],
    by_x: list[int],
    by_y: list[int],
    in_left: list[bool]
) -> tuple[int, int, float] {
    let count = by_x.len()
    if count <= 3 { return _closest_pair_brute(points, by_x) }
    let middle = count // 2
    let split_x = points[by_x[middle]].x
    for position in 0..middle { in_left[by_x[position]] = true }
    let left_y: list[int] = []
    let right_y: list[int] = []
    for index in by_y {
        if in_left[index] { left_y.push(index) } else { right_y.push(index) }
    }
    for position in 0..middle { in_left[by_x[position]] = false }
    let (left_a, left_b, left_gap) = _closest_pair_of(points, by_x[0:middle], left_y, in_left)
    let (right_a, right_b, right_gap) = _closest_pair_of(
        points,
        by_x[middle:count],
        right_y,
        in_left
    )
    var best_a = left_a
    var best_b = left_b
    var best = left_gap
    if right_gap < best {
        best_a = right_a
        best_b = right_b
        best = right_gap
    }
    let strip: list[int] = []
    for index in by_y {
        if math.fabs(points[index].x - split_x) < best { strip.push(index) }
    }
    for i in 0..strip.len() {
        var j = i + 1
        while j < strip.len() and points[strip[j]].y - points[strip[i]].y < best {
            let gap = points[strip[i]].distance(points[strip[j]])
            if gap < best {
                best = gap
                best_a = strip[i]
                best_b = strip[j]
            }
            j += 1
        }
    }
    return (best_a, best_b, best)
}

#: Return the closest pair among two or three indices, by direct comparison.
#:
#: The base case of the recursion: below four points the divide step cannot
#: split them usefully, and three comparisons settle it.
fn _closest_pair_brute(points: list[Point2], indices: list[int]) -> tuple[int, int, float] {
    var best_a = indices[0]
    var best_b = indices[1]
    var best = points[best_a].distance(points[best_b])
    for i in 0..indices.len() {
        for j in i + 1..indices.len() {
            let gap = points[indices[i]].distance(points[indices[j]])
            if gap < best {
                best = gap
                best_a = indices[i]
                best_b = indices[j]
            }
        }
    }
    return (best_a, best_b, best)
}

#: Return the smallest axis-aligned rectangle containing every point.
#:
#: `pub` rather than the file-local `fn` its name would suggest: it is the
#: shared body behind both `bounding_box` above and `Polygon.bounding_box` in
#: `polygon.jai`, which calls this directly rather than the free
#: `bounding_box`, that being a function with the same name as the method it
#: would be calling from.
pub fn bounds_of(points: list[Point2]) -> Rect {
    if points.len() == 0 { throw ValueError("bounding_box of an empty point set") }
    var min_x = points[0].x
    var min_y = points[0].y
    var max_x = points[0].x
    var max_y = points[0].y
    for point in points {
        min_x = min(min_x, point.x)
        min_y = min(min_y, point.y)
        max_x = max(max_x, point.x)
        max_y = max(max_y, point.y)
    }
    return Rect(Point2(min_x, min_y), Point2(max_x, max_y))
}

fn _twice_triangle_area(a: Point2, b: Point2, c: Point2) -> float {
    return math.fabs((b - a).cross(c - a))
}

#: Report whether `circle` contains `point`, tolerantly.
fn _circle_holds(circle: Circle, point: Point2) -> bool {
    return circle.center.distance(point) <= circle.radius + EPSILON * (1.0 + circle.radius)
}

#: Return the smallest circle with `first` and `second` on its boundary.
fn _circle_through_two(first: Point2, second: Point2) -> Circle {
    return Circle(first.lerp(second, 0.5), first.distance(second) / 2.0)
}

#: Return the smallest circle through three points, collinear ones included.
fn _circle_through_three(a: Point2, b: Point2, c: Point2) -> Circle {
    if orientation(a, b, c) != 0 { return circle_from_points(a, b, c) }
    var best = _circle_through_two(a, b)
    let with_c = _circle_through_two(a, c)
    if with_c.radius > best.radius { best = with_c }
    let other = _circle_through_two(b, c)
    if other.radius > best.radius { best = other }
    return best
}

#: Return the smallest circle over `points[0..count]` with `pinned` on its edge.
fn _enclosing_pinning_one(points: list[Point2], count: int, pinned: Point2) -> Circle {
    var circle = _circle_through_two(points[0], pinned)
    for index in 1..count {
        if not _circle_holds(circle, points[index]) {
            circle = _enclosing_pinning_two(points, index, pinned, points[index])
        }
    }
    return circle
}

#: Return the smallest circle over `points[0..count]` with two points on its edge.
fn _enclosing_pinning_two(
    points: list[Point2],
    count: int,
    first: Point2,
    second: Point2
) -> Circle {
    var circle = _circle_through_two(first, second)
    for index in 0..count {
        if not _circle_holds(circle, points[index]) {
            circle = _circle_through_three(first, second, points[index])
        }
    }
    return circle
}

#: Return the points in a fixed pseudo-random order.
#:
#: Fisher-Yates driven by a 31-bit linear congruential generator with the
#: constants from the C standard. The seed is fixed, so the permutation — and
#: therefore every circle `minimum_enclosing_circle` returns — is reproducible.
#: The generator's low bits are notoriously correlated, so the index comes from
#: the high half of the state.
fn _pseudo_shuffled(points: list[Point2]) -> list[Point2] {
    let out = points[0:points.len()]
    var state = 20250804
    var index = out.len() - 1
    while index > 0 {
        state = (state * 1103515245 + 12345) % 2147483648
        let target = state // 65536 % (index + 1)
        let held = out[index]
        out[index] = out[target]
        out[target] = held
        index -= 1
    }
    return out
}
