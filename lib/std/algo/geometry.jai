#: Computational geometry: the point and vector types, the primitive shapes,
#: and the classical planar algorithms over them.
#:
#: ## Floating-point robustness
#:
#: Almost every decision made here reduces to the sign of one cross product:
#: which side of a line a point falls on, whether a hull turns left, whether
#: three vertices are collinear, whether a triangle is an ear. In exact
#: arithmetic that sign is never in doubt. In binary64 it is. For three nearly
#: collinear points the determinant `(b - a) x (c - a)` is a difference of two
#: products that agree in almost every bit, so what survives the subtraction is
#: rounding error and its sign is arbitrary. Worse, it is *inconsistently*
#: arbitrary: two algebraically identical expressions can disagree, and then a
#: hull walk pops a vertex it should have kept, or a point tests inside a
#: polygon from one edge and outside from the next, and the loop never
#: terminates.
#:
#: There are two honest responses. The first is exact arithmetic — Shewchuk's
#: adaptive predicates evaluate the determinant to as many bits as the sign
#: needs and no more. The second is to decide in advance how close is too close
#: to call, and answer "collinear" inside that band. This module takes the
#: second. `orientation` compares the determinant against
#: `EPSILON * (1 + |x1*y2| + |y1*x2|)` — an absolute floor of `EPSILON` plus a
#: term that grows with the magnitude of the operands — and every other
#: predicate is expressed through `orientation` so that they cannot contradict
#: one another.
#:
#: What that buys, and what it costs:
#:
#:   - Answers are consistent. The same three points always get the same
#:     verdict, which is what keeps the loops here terminating.
#:   - Answers are not exact. A polygon whose vertices differ only in the
#:     fifteenth digit is reported as degenerate. That is usually the truth
#:     about the data rather than a defect in the test.
#:   - Accuracy is best for coordinates between about 1 and 1e6. Near 1e15 a
#:     single ulp already exceeds `EPSILON` and no tolerance can rescue the
#:     predicate; translate to a local origin before calling.
#:   - Every routine that can subtract a local origin before multiplying does
#:     so — `orientation`, the shoelace sum, the circumcentre, the closest-pair
#:     strip — because that removes the leading cancellation instead of
#:     tolerating it.
#:   - Never compare coordinates with `==`. `Point2.approx_eq` is the test.
#:
#: ## Where the pieces live
#:
#: This file used to be all of the above in one 1857-line module. It is now
#: split by concern, five files deep:
#:
#:   - `vectors.jai` — `Point2`, `Vec3`, and `EPSILON`. The two value types
#:     have no dependency on anything else in the module, which is what makes
#:     them the leaf of the split; `EPSILON` moved with them because their
#:     `approx_eq` methods default to it.
#:   - `geometry.jai` (here) — `orientation`, the predicate the module doc
#:     comment above is about, plus its two tolerant-comparison siblings,
#:     `sign_within` and `within_bounds`. Both were private (`_sign_within`,
#:     `_within_bounds`) in the original file; they are `pub` now because
#:     `shapes.jai`'s `Segment` and `polygon.jai`'s `point_in_polygon` both
#:     need the exact same tolerant checks `orientation` is built from, and
#:     jaithon has no visibility tier between "this file only" and "the whole
#:     module" — the same move `jaithon.compile.check` already made for
#:     `is_any`/`decl_type` and their kin. This file imports nothing from the
#:     three below it, so it stays the dependency-free centre they all import
#:     from.
#:   - `shapes.jai` — `Line`, `Segment`, `Rect`, `Circle`, and the free
#:     functions that build or cut a circle (`circle_from_points`,
#:     `line_circle_intersection`).
#:   - `polygon.jai` — `Polygon` and every ring algorithm: `winding_number`,
#:     `point_in_polygon[_crossing]`, the `polygon_*` measurements,
#:     `polygon_clip`, `triangulate`.
#:   - `point_cloud.jai` — the algorithms over a bare `list[Point2]`:
#:     `convex_hull`, `closest_pair`, `diameter`, `minimum_enclosing_circle`,
#:     `centroid`, `bounding_box`.
#:
#: Each of those files' own header explains why its particular grouping holds
#: together; see them for the rest of the story.
import std.math
from std.algo.vectors import Point2, EPSILON

#: Return 1 when `a`, `b`, `c` turn counter-clockwise, -1 clockwise, 0 collinear.
#:
#: The predicate the whole module rests on. It is the sign of the determinant
#: `(b - a) x (c - a)`, compared against a tolerance that grows with the
#: magnitude of the two products so that the answer is the same however far
#: from the origin the points sit. See the module doc comment for what that
#: tolerance can and cannot do.
#:
#: ```
#: orientation(Point2(0.0, 0.0), Point2(1.0, 0.0), Point2(1.0, 1.0))    # 1
#: orientation(Point2(0.0, 0.0), Point2(1.0, 0.0), Point2(2.0, 0.0))    # 0
#: ```
pub fn orientation(a: Point2, b: Point2, c: Point2) -> int {
    let x1 = b.x - a.x
    let y1 = b.y - a.y
    let x2 = c.x - a.x
    let y2 = c.y - a.y
    let first = x1 * y2
    let second = y1 * x2
    return sign_within(first - second, math.fabs(first) + math.fabs(second))
}

# --- Shared tolerant-comparison predicates -----------------------------------
#:
#: `orientation` above is the predicate every caller outside this file wants.
#: These two are the smaller pieces it (and `shapes.jai`'s `Segment`, and
#: `polygon.jai`'s ring algorithms) are built from — see "Where the pieces
#: live" in the module doc comment for why they are `pub` rather than private.

#: Return the sign of `value`, treating anything near enough to zero as zero.
#:
#: `scale` is the magnitude of the terms `value` was formed from, so the
#: tolerance is absolute near the origin and relative once the products grow.
pub fn sign_within(value: float, scale: float) -> int {
    let tolerance = EPSILON * (1.0 + scale)
    if value > tolerance { return 1 }
    if value < -tolerance { return -1 }
    return 0
}

#: Report whether `p` lies within the bounding box of `a` and `b`.
#:
#: Only meaningful once `p` is known to be collinear with them, which is how
#: every caller uses it.
pub fn within_bounds(a: Point2, b: Point2, p: Point2) -> bool {
    return min(a.x, b.x) - EPSILON <= p.x <= max(a.x, b.x) + EPSILON and
        min(a.y, b.y) - EPSILON <= p.y <= max(a.y, b.y) + EPSILON
}
