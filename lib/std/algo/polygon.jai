#: `Polygon` and every algorithm over a closed vertex ring: `winding_number`,
#: `point_in_polygon[_crossing]`, `polygon_signed_area/area/perimeter/
#: centroid`, `polygon_clip`, and `triangulate`.
#:
#: Split out of `geometry.jai`. This is the one group in the split that did
#: not divide any further: every function here shares one family of private
#: helpers (`_require_ring`, `_is_convex_ring`, `_counter_clockwise`,
#: `_cleaned_ring`, `_is_ear`, `_in_triangle`, `_line_crossing`) that nothing
#: outside this file calls, and `Polygon`'s own methods are thin wrappers
#: around the free functions sitting right below it -- `Polygon.area()` calls
#: `polygon_area()`, `Polygon.contains()` calls `point_in_polygon()`, and so
#: on. Pulling the class away from the functions it delegates to would not
#: have simplified anything.
import std.math
from std.core import Printable
from std.algo.vectors import Point2, EPSILON
from std.algo.geometry import orientation, within_bounds
from std.algo.shapes import Segment, Rect
from std.algo.point_cloud import bounds_of

#: A closed polygon, as a ring of vertices.
#:
#: The ring is implicitly closed: the last vertex joins back to the first, and
#: repeating the first vertex at the end would create a zero-length edge. The
#: class validates the ring once at construction so that the free functions
#: below, which take a bare `list[Point2]`, need not keep re-checking it.
pub class Polygon: Printable {
    let vertices: list[Point2]

    #: Create the polygon with the given vertex ring.
    #:
    #: Raises `ValueError` for fewer than three vertices.
    pub fn init(self, vertices: list[Point2]) {
        if vertices.len() < 3 {
            throw ValueError(f"a polygon needs at least 3 vertices, got {vertices.len()}")
        }
        self.vertices = vertices[0:vertices.len()]
    }

    #: Return the number of vertices.
    pub fn vertex_count(self) -> int { return self.vertices.len() }

    #: Return the vertex at `index`.
    #:
    #: Raises `IndexError` when `index` is out of range.
    pub fn vertex(self, index: int) -> Point2 {
        if index < 0 or index >= self.vertices.len() {
            throw IndexError(f"vertex index must be in 0..{self.vertices.len()}, got {index}")
        }
        return self.vertices[index]
    }

    #: Return a copy of the vertex ring, safe to hand to the free functions.
    pub fn to_list(self) -> list[Point2] { return self.vertices[0:self.vertices.len()] }

    #: Return the edges as segments, the closing edge last.
    pub fn edges(self) -> list[Segment] {
        let count = self.vertices.len()
        return [Segment(self.vertices[i], self.vertices[(i + 1) % count]) for i in 0..count]
    }

    #: Return the signed area: positive when the ring runs counter-clockwise.
    pub fn signed_area(self) -> float { return polygon_signed_area(self.vertices) }

    #: Return the enclosed area, regardless of winding direction.
    #:
    #: ```
    #: Polygon([Point2(0.0, 0.0), Point2(4.0, 0.0), Point2(4.0, 3.0)]).area()    # 6.0
    #: ```
    pub fn area(self) -> float { return polygon_area(self.vertices) }

    #: Return the total edge length.
    pub fn perimeter(self) -> float { return polygon_perimeter(self.vertices) }

    #: Return the area-weighted centroid.
    pub fn centroid(self) -> Point2 { return polygon_centroid(self.vertices) }

    #: Return the smallest axis-aligned rectangle containing the polygon.
    #:
    #: Calls the shared helper rather than the free `bounding_box`, which has
    #: the same name as this method.
    pub fn bounding_box(self) -> Rect { return bounds_of(self.vertices) }

    #: Report whether `p` is inside the polygon or on its boundary.
    pub fn contains(self, p: Point2) -> bool { return point_in_polygon(self.vertices, p) }

    #: Report whether the ring turns the same way at every vertex.
    #:
    #: Collinear vertices do not break convexity. A self-intersecting ring is
    #: never convex by this test, since it must reverse somewhere.
    pub fn is_convex(self) -> bool { return _is_convex_ring(self.vertices) }

    #: Report whether the ring runs counter-clockwise.
    pub fn is_counter_clockwise(self) -> bool { return self.signed_area() > 0.0 }

    #: Report whether no two non-adjacent edges touch.
    #:
    #: O(n^2), comparing every pair of edges. Worth calling before
    #: `triangulate`, which assumes a simple polygon and can only report the
    #: failure late and vaguely.
    pub fn is_simple(self) -> bool {
        let sides = self.edges()
        let count = sides.len()
        for i in 0..count {
            for j in i + 1..count {
                # Adjacent edges are meant to share exactly one endpoint.
                if j == i + 1 or i == 0 and j == count - 1 { continue }
                if sides[i].intersects(sides[j]) { return false }
            }
        }
        return true
    }

    #: Return the polygon rendered as `Polygon(n vertices)`.
    pub fn to_str(self) -> str { return f"Polygon({self.vertices.len()} vertices)" }

    fn __len__(self) -> int { return self.vertices.len() }

    fn __str__(self) -> str { return self.to_str() }
}

#: Return how many times the polygon's boundary winds around `point`.
#:
#: Zero means outside. Any other value means inside, and its magnitude counts
#: the loops: a doubly wound ring reports 2, a ring wound the other way -1.
#: Points on the boundary are unspecified — ask `point_in_polygon`, which
#: settles them first.
#:
#: ```
#: let square = [Point2(0.0, 0.0), Point2(2.0, 0.0), Point2(2.0, 2.0), Point2(0.0, 2.0)]
#: winding_number(square, Point2(1.0, 1.0))     # 1
#: winding_number(square, Point2(3.0, 1.0))     # 0
#: ```
pub fn winding_number(ring: list[Point2], point: Point2) -> int {
    _require_ring(ring, "winding_number")
    let count = ring.len()
    var winding = 0
    for index in 0..count {
        let a = ring[index]
        let b = ring[(index + 1) % count]
        # Half-open in y so that a vertex exactly at the ray's height is
        # counted by one of its two edges and not by both.
        if a.y <= point.y {
            if b.y > point.y and orientation(a, b, point) > 0 { winding += 1 }
        } elif b.y <= point.y and orientation(a, b, point) < 0 {
            winding -= 1
        }
    }
    return winding
}

#: Report whether `point` is inside the polygon or on its boundary.
#:
#: The non-zero winding rule, and the one to reach for. It is the rule that
#: matches the intuition "the ring encloses this point": for a self-
#: intersecting ring, a region covered twice counts as inside, where the
#: even-odd rule of `point_in_polygon_crossing` punches it out again. It also
#: costs the same, and unlike the crossing test it answers boundary points
#: definitely — they are inside — instead of leaving them to the rounding.
#: On a simple polygon the two rules agree everywhere except the boundary.
#:
#: ```
#: let square = [Point2(0.0, 0.0), Point2(2.0, 0.0), Point2(2.0, 2.0), Point2(0.0, 2.0)]
#: point_in_polygon(square, Point2(1.0, 1.0))    # true
#: point_in_polygon(square, Point2(2.0, 1.0))    # true
#: point_in_polygon(square, Point2(2.5, 1.0))    # false
#: ```
pub fn point_in_polygon(ring: list[Point2], point: Point2) -> bool {
    _require_ring(ring, "point_in_polygon")
    let count = ring.len()
    for index in 0..count {
        let a = ring[index]
        let b = ring[(index + 1) % count]
        if orientation(a, b, point) == 0 and within_bounds(a, b, point) { return true }
    }
    return winding_number(ring, point) != 0
}

#: Report whether `point` is inside the polygon by the even-odd rule.
#:
#: Ray casting: shoot a ray to the right and count boundary crossings, odd
#: meaning inside. Use it when even-odd is the semantics you want — it is what
#: SVG's `fill-rule: evenodd` and most CAD formats mean by a hole cut into a
#: ring drawn the same way round. Otherwise prefer `point_in_polygon`, whose
#: winding rule handles self-intersection more intuitively and does not leave
#: boundary points to chance; this test says nothing useful about a point that
#: lies exactly on an edge.
#:
#: ```
#: let square = [Point2(0.0, 0.0), Point2(2.0, 0.0), Point2(2.0, 2.0), Point2(0.0, 2.0)]
#: point_in_polygon_crossing(square, Point2(1.0, 1.0))    # true
#: point_in_polygon_crossing(square, Point2(3.0, 1.0))    # false
#: ```
pub fn point_in_polygon_crossing(ring: list[Point2], point: Point2) -> bool {
    _require_ring(ring, "point_in_polygon_crossing")
    let count = ring.len()
    var inside = false
    for index in 0..count {
        let a = ring[index]
        let b = ring[(index + 1) % count]
        # The half-open comparison is what makes a vertex on the ray count
        # once: exactly one of the two edges meeting there straddles it.
        if (a.y > point.y) != (b.y > point.y) {
            let crossing = a.x + (point.y - a.y) / (b.y - a.y) * (b.x - a.x)
            if point.x < crossing { inside = not inside }
        }
    }
    return inside
}

#: Return the signed area of the ring: positive counter-clockwise.
#:
#: The shoelace formula, evaluated as a triangle fan from the first vertex so
#: that every coordinate is a local offset. Summing `x_i * y_(i+1)` on the raw
#: coordinates instead would, for a small polygon far from the origin, form
#: huge products and then subtract them — the classic cancellation, and the
#: reason a survey polygon in metres-from-the-equator can come out with the
#: wrong area entirely.
#:
#: ```
#: polygon_signed_area([Point2(0.0, 0.0), Point2(4.0, 0.0), Point2(4.0, 3.0)])    # 6.0
#: polygon_signed_area([Point2(0.0, 0.0), Point2(4.0, 3.0), Point2(4.0, 0.0)])    # -6.0
#: ```
pub fn polygon_signed_area(ring: list[Point2]) -> float {
    _require_ring(ring, "polygon_signed_area")
    let origin = ring[0]
    var total = 0.0
    for index in 1..ring.len() - 1 {
        total += (ring[index] - origin).cross(ring[index + 1] - origin)
    }
    return total * 0.5
}

#: Return the area enclosed by the ring, whichever way it winds.
#:
#: ```
#: polygon_area([Point2(0.0, 0.0), Point2(4.0, 0.0), Point2(4.0, 3.0)])    # 6.0
#: ```
pub fn polygon_area(ring: list[Point2]) -> float {
    return math.fabs(polygon_signed_area(ring))
}

#: Return the total length of the ring's edges, closing edge included.
#:
#: ```
#: polygon_perimeter([Point2(0.0, 0.0), Point2(3.0, 0.0), Point2(3.0, 4.0)])    # 12.0
#: ```
pub fn polygon_perimeter(ring: list[Point2]) -> float {
    _require_ring(ring, "polygon_perimeter")
    let count = ring.len()
    var total = 0.0
    for index in 0..count {
        total += ring[index].distance(ring[(index + 1) % count])
    }
    return total
}

#: Return the area-weighted centroid of the region the ring encloses.
#:
#: This is the centre of mass of the filled polygon, which is not the mean of
#: the vertices — see `centroid`. Subdividing an edge changes the vertex mean
#: and leaves this unmoved, which is the property that makes it the right
#: answer for a shape.
#:
#: Raises `ValueError` for a degenerate ring, whose zero area leaves the
#: weighted average undefined.
#:
#: ```
#: let square = [Point2(0.0, 0.0), Point2(2.0, 0.0), Point2(2.0, 2.0), Point2(0.0, 2.0)]
#: polygon_centroid(square).x    # 1.0
#: ```
pub fn polygon_centroid(ring: list[Point2]) -> Point2 {
    _require_ring(ring, "polygon_centroid")
    let origin = ring[0]
    var twice_area = 0.0
    var weighted_x = 0.0
    var weighted_y = 0.0
    for index in 1..ring.len() - 1 {
        let a = ring[index] - origin
        let b = ring[index + 1] - origin
        let cross = a.cross(b)
        twice_area += cross
        weighted_x += (a.x + b.x) * cross
        weighted_y += (a.y + b.y) * cross
    }
    if math.fabs(twice_area) <= EPSILON {
        throw ValueError("polygon_centroid is undefined for a polygon of zero area")
    }
    let scale = 3.0 * twice_area
    return Point2(origin.x + weighted_x / scale, origin.y + weighted_y / scale)
}

#: Return `subject` clipped to the convex region `window`.
#:
#: The Sutherland-Hodgman algorithm: run the whole subject ring through one
#: window edge at a time, keeping the vertices on the inside and adding a new
#: vertex wherever the ring crosses. n passes over the ring, so O(n * m).
#:
#: `window` must be convex — that is the algorithm's premise, and it is checked
#: — but `subject` need not be. A concave subject clipped by a window that cuts
#: it into two pieces comes back as one ring joined by a degenerate edge along
#: the window boundary; that is the known artefact of the method, and the price
#: of its simplicity. Greiner-Hormann is the algorithm to reach for when the
#: pieces must stay separate.
#:
#: ```
#: let square = [Point2(0.0, 0.0), Point2(4.0, 0.0), Point2(4.0, 4.0), Point2(0.0, 4.0)]
#: let window = [Point2(1.0, 1.0), Point2(3.0, 1.0), Point2(3.0, 3.0), Point2(1.0, 3.0)]
#: polygon_area(polygon_clip(square, window))    # 4.0
#: ```
pub fn polygon_clip(subject: list[Point2], window: list[Point2]) -> list[Point2] {
    _require_ring(subject, "polygon_clip")
    _require_ring(window, "polygon_clip")
    if not _is_convex_ring(window) {
        throw ValueError("polygon_clip requires a convex clipping window")
    }
    let bounds = _counter_clockwise(window)
    var output = subject[0:subject.len()]
    let edge_count = bounds.len()
    for index in 0..edge_count {
        if output.len() == 0 { break }
        let edge_start = bounds[index]
        let edge_end = bounds[(index + 1) % edge_count]
        let pending = output
        output = []
        var previous = pending[pending.len() - 1]
        for current in pending {
            let current_inside = orientation(edge_start, edge_end, current) >= 0
            let previous_inside = orientation(edge_start, edge_end, previous) >= 0
            if current_inside {
                if not previous_inside {
                    output.push(_line_crossing(edge_start, edge_end, previous, current))
                }
                output.push(current)
            } elif previous_inside {
                output.push(_line_crossing(edge_start, edge_end, previous, current))
            }
            previous = current
        }
    }
    return output
}

#: Return a triangulation of a simple polygon, as counter-clockwise triples.
#:
#: Ear clipping: repeatedly find a convex vertex whose triangle with its two
#: neighbours holds no other vertex, emit that triangle, and unlink the vertex.
#: Every simple polygon has at least two such ears at all times (the two-ears
#: theorem), so the search always finds one and the loop always makes progress.
#: O(n^2), which is the practical choice below a few thousand vertices;
#: Seidel's and Chazelle's algorithms do better asymptotically and are far
#: harder to make robust.
#:
#: The triangles tile the polygon exactly and none of them is degenerate.
#: Collinear vertices are dropped rather than turned into zero-area triangles,
#: both the ones present at the start and the ones that only become collinear
#: as their neighbours are clipped away, so the count is `m - 2` for the `m`
#: vertices that survive that pruning — at most `n - 2`, and fewer for a ring
#: with straight runs in it.
#:
#: Raises `ValueError` when the ring encloses no area, and when no ear can be
#: found, which means the ring intersects itself; `Polygon.is_simple`
#: diagnoses that ahead of time.
#:
#: ```
#: let square = [Point2(0.0, 0.0), Point2(2.0, 0.0), Point2(2.0, 2.0), Point2(0.0, 2.0)]
#: triangulate(square).len()    # 2
#: ```
pub fn triangulate(ring: list[Point2]) -> list[tuple[Point2, Point2, Point2]] {
    _require_ring(ring, "triangulate")
    let cleaned = _cleaned_ring(ring)
    if cleaned.len() < 3 {
        throw ValueError("triangulate expects a polygon with non-zero area")
    }
    let working = _counter_clockwise(cleaned)
    let count = working.len()
    let previous_of = [(index + count - 1) % count for index in 0..count]
    let next_of = [(index + 1) % count for index in 0..count]
    let triangles: list[tuple[Point2, Point2, Point2]] = []
    var remaining = count
    var current = 0
    var failures = 0
    while remaining > 3 {
        let previous = previous_of[current]
        let next = next_of[current]
        let turn = orientation(working[previous], working[current], working[next])
        if turn == 0 {
            # Clipping the neighbours has left this vertex in a straight line
            # between the two that remain. It bounds no area, so unlink it
            # without emitting anything: a zero-area triangle is not a
            # triangle, and leaving it in place deadlocks the search — a ring
            # of collinear leftovers has no ear at all.
            next_of[previous] = next
            previous_of[next] = previous
            remaining -= 1
            failures = 0
            current = next
        } elif turn > 0 and _is_ear(working, previous_of, next_of, previous, current, next) {
            triangles.push((working[previous], working[current], working[next]))
            next_of[previous] = next
            previous_of[next] = previous
            remaining -= 1
            failures = 0
            current = next
        } else {
            current = next
            failures += 1
            if failures > remaining {
                throw ValueError("triangulate found no ear; the polygon is not simple")
            }
        }
    }
    let last_previous = previous_of[current]
    let last_next = next_of[current]
    if orientation(working[last_previous], working[current], working[last_next]) != 0 {
        triangles.push((working[last_previous], working[current], working[last_next]))
    }
    return triangles
}

# --- Private helpers ---------------------------------------------------------
fn _require_ring(ring: list[Point2], function_name: str) -> void {
    if ring.len() < 3 {
        throw ValueError(f"{function_name} needs at least 3 vertices, got {ring.len()}")
    }
}

#: Report whether every turn of the ring goes the same way.
fn _is_convex_ring(ring: list[Point2]) -> bool {
    let count = ring.len()
    var sign = 0
    for index in 0..count {
        let turn = orientation(ring[index], ring[(index + 1) % count], ring[(index + 2) % count])
        if turn != 0 {
            if sign != 0 and turn != sign { return false }
            sign = turn
        }
    }
    return true
}

#: Return the ring wound counter-clockwise, reversing it if it is not.
fn _counter_clockwise(ring: list[Point2]) -> list[Point2] {
    if polygon_signed_area(ring) >= 0.0 { return ring[0:ring.len()] }
    let reversed_ring: list[Point2] = []
    var index = ring.len() - 1
    while index >= 0 {
        reversed_ring.push(ring[index])
        index -= 1
    }
    return reversed_ring
}

#: Return the ring without duplicate or collinear vertices.
fn _cleaned_ring(ring: list[Point2]) -> list[Point2] {
    let distinct: list[Point2] = []
    for point in ring {
        if distinct.len() == 0 or not distinct[distinct.len() - 1].approx_eq(point) {
            distinct.push(point)
        }
    }
    while distinct.len() > 1 and distinct[0].approx_eq(distinct[distinct.len() - 1]) {
        distinct.pop()
    }
    let count = distinct.len()
    if count < 3 { return distinct }
    let cleaned: list[Point2] = []
    for index in 0..count {
        let previous = distinct[(index + count - 1) % count]
        let next = distinct[(index + 1) % count]
        if orientation(previous, distinct[index], next) != 0 { cleaned.push(distinct[index]) }
    }
    return cleaned
}

#: Report whether the vertex `current` is an ear of the remaining ring.
fn _is_ear(
    ring: list[Point2],
    previous_of: list[int],
    next_of: list[int],
    previous: int,
    current: int,
    next: int
) -> bool {
    if orientation(ring[previous], ring[current], ring[next]) <= 0 { return false }
    var index = next_of[next]
    while index != previous {
        # Only a reflex vertex can be trapped inside another vertex's ear, and
        # skipping the convex ones also avoids rejecting an ear because a
        # neighbouring convex vertex grazes its boundary.
        let is_reflex = orientation(ring[previous_of[index]], ring[index], ring[next_of[index]]) < 0
        if is_reflex and _in_triangle(ring[index], ring[previous], ring[current], ring[next]) {
            return false
        }
        index = next_of[index]
    }
    return true
}

#: Report whether `p` is inside the counter-clockwise triangle, boundary included.
fn _in_triangle(p: Point2, a: Point2, b: Point2, c: Point2) -> bool {
    return orientation(a, b, p) >= 0 and orientation(b, c, p) >= 0 and orientation(c, a, p) >= 0
}

#: Return where the infinite lines `a`-`b` and `c`-`d` cross.
#:
#: The callers only reach this with a crossing that exists: one endpoint of
#: `c`-`d` is inside the half-plane of `a`-`b` and the other is not.
fn _line_crossing(a: Point2, b: Point2, c: Point2, d: Point2) -> Point2 {
    let along = d - c
    let edge = b - a
    let denominator = along.cross(edge)
    if denominator == 0.0 { return c }
    return c + along * ((a - c).cross(edge) / denominator)
}
