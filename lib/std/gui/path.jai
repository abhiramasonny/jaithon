#: Anti-aliased path geometry: the `Path` a shape is described with, and the
#: scanline machinery that turns it into per-pixel coverage.
#:
#: Split out of `std.gui.canvas` because `Path` shares no state with `Canvas`
#: — it is built and read on its own, independent of any drawing surface —
#: and the functions around it (`_edges_of`, `_crossings`, `_spans`,
#: `_outline_run` and the rest) are pure: they take a `Path` or a flat list of
#: coordinates in and hand geometry back, never touching a pixel buffer or a
#: clip rectangle. `Canvas.fill_path` and `Canvas.stroke_path` still call
#: several of them directly, which is why the ones it reaches for
#: (`_axis_aligned_rect`, `_edges_of`, `_crossings`, `_spans`, `_add_span`,
#: `_dashed`, `_outline_run`, `_turns_visibly`) are exported; they remain
#: implementation plumbing for `std.gui.canvas`; not a public path-algebra API
#: in their own right, hence the leading underscore despite being `pub`.
import std.math

#: How a filled path decides which side of an edge is inside.
#:
#: `NonZero` counts edge directions and fills wherever they do not cancel, so a
#: self-crossing outline is solid. `EvenOdd` fills where a ray crosses an odd
#: number of edges, which leaves a hole wherever the shape overlaps itself.
pub enum Winding { NonZero, EvenOdd }

#: How a stroke finishes at an open end.
pub enum Cap { Butt, Round, Square }

#: How two segments of a stroke meet.
pub enum Join { Miter, Round, Bevel }

#: Sub-scanlines sampled per output row when filling a path.
#:
#: Horizontal coverage is computed exactly, so only the vertical direction is
#: sampled. Four rows put the remaining error below what the eye resolves on a
#: chart while keeping the inner loop short.
pub const SUBSAMPLES = 4

#: Segments a curve is flattened into.
pub const CURVE_STEPS = 16

#: A sequence of straight and curved segments in float coordinates.
#:
#: Paths exist so a shape can be described once and then filled or stroked with
#: anti-aliasing, which the integer primitives above cannot do. A path may hold
#: several subpaths; `move_to` begins each one.
#:
#: ```
#: let path = Path()
#: path.move_to(0.0, 0.0)
#: path.line_to(10.0, 0.0)
#: path.line_to(10.0, 10.0)
#: path.close()
#: path.subpaths().len()      # 1
#: ```
pub class Path {
    var runs: list[list[float]]
    var current: list[float]

    #: Create an empty path.
    pub fn init(self) {
        self.runs = []
        self.current = []
    }

    #: Begin a new subpath at `(x, y)`.
    pub fn move_to(self, x: float, y: float) -> Path {
        self._flush()
        self.current = [x, y]
        return self
    }

    #: Extend the current subpath with a straight segment.
    #:
    #: Raises `ValueError` when no subpath has been started, since there is no
    #: point to draw from.
    pub fn line_to(self, x: float, y: float) -> Path {
        if self.current.len() == 0 {
            throw ValueError("line_to before move_to: the path has no current point")
        }
        self.current.push(x)
        self.current.push(y)
        return self
    }

    #: Extend the current subpath with a cubic Bezier, flattened as it is added.
    pub fn cubic_to(
        self,
        cx0: float,
        cy0: float,
        cx1: float,
        cy1: float,
        x: float,
        y: float
    ) -> Path {
        if self.current.len() == 0 {
            throw ValueError("cubic_to before move_to: the path has no current point")
        }
        let x0 = self.current[self.current.len() - 2]
        let y0 = self.current[self.current.len() - 1]
        for step in 1..=CURVE_STEPS {
            let t = step.to_float() / CURVE_STEPS.to_float()
            let u = 1.0 - t
            let a = u * u * u
            let b = 3.0 * u * u * t
            let c = 3.0 * u * t * t
            let d = t * t * t
            self.current.push(a * x0 + b * cx0 + c * cx1 + d * x)
            self.current.push(a * y0 + b * cy0 + c * cy1 + d * y)
        }
        return self
    }

    #: Close the current subpath, joining its end back to its start.
    pub fn close(self) -> Path {
        if self.current.len() >= 4 {
            self.current.push(self.current[0])
            self.current.push(self.current[1])
        }
        self._flush()
        return self
    }

    #: Add a whole subpath from a list of points, closing it.
    pub fn polygon(self, points: list[tuple[float, float]]) -> Path {
        if points.len() == 0 { return self }
        let (first_x, first_y) = points[0]
        let _started = self.move_to(first_x, first_y)
        for index in 1..points.len() {
            let (x, y) = points[index]
            let _extended = self.line_to(x, y)
        }
        return self.close()
    }

    #: The subpaths, each a flat list of alternating x and y coordinates.
    pub fn subpaths(self) -> list[list[float]] {
        var out = [run for run in self.runs]
        if self.current.len() >= 4 { out.push(self.current) }
        return out
    }

    #: Report whether the path holds nothing drawable.
    pub fn is_empty(self) -> bool {
        return self.subpaths().len() == 0
    }

    fn _flush(self) -> void {
        if self.current.len() >= 4 { self.runs.push(self.current) }
        self.current = []
    }

    fn __str__(self) -> str {
        return f"Path({self.subpaths().len()} subpaths)"
    }
}

#: One edge of a flattened path, kept in the order the path walked it.
class _Edge {
    pub let x0: float
    pub let y0: float
    pub let x1: float
    pub let y1: float
    pub let direction: int

    pub fn init(self, x0: float, y0: float, x1: float, y1: float) {
        # Edges are stored top to bottom so a scanline test is one comparison,
        # and the direction that was lost in the swap is kept for the winding
        # count.
        if y0 <= y1 {
            self.x0 = x0
            self.y0 = y0
            self.x1 = x1
            self.y1 = y1
            self.direction = 1
        } else {
            self.x0 = x1
            self.y0 = y1
            self.x1 = x0
            self.y1 = y0
            self.direction = -1
        }
    }

    #: Where this edge crosses the horizontal line at `y`.
    pub fn crossing(self, y: float) -> float {
        let span = self.y1 - self.y0
        if span == 0.0 { return self.x0 }
        return self.x0 + (y - self.y0) * (self.x1 - self.x0) / span
    }
}

#: The rectangle a path describes, when it is exactly one axis-aligned box.
#:
#: Returned as `(left, top, right, bottom)` in device order, or null when the
#: path is anything else.
pub fn _axis_aligned_rect(path: Path) -> tuple[float, float, float, float]? {
    let runs = path.subpaths()
    if runs.len() != 1 { return null }
    let run = runs[0]
    let count = run.len() // 2
    # Four corners, or five when `close` repeated the first.
    if count != 4 and count != 5 { return null }
    if count == 5 {
        if run[0] != run[8] or run[1] != run[9] { return null }
    }

    var left = run[0]
    var right = run[0]
    var top = run[1]
    var bottom = run[1]
    for index in 0..4 {
        left = min(left, run[index * 2])
        right = max(right, run[index * 2])
        top = min(top, run[index * 2 + 1])
        bottom = max(bottom, run[index * 2 + 1])
    }
    # Every corner must sit on the box, and every edge must be axis aligned.
    for index in 0..4 {
        let x = run[index * 2]
        let y = run[index * 2 + 1]
        if x != left and x != right { return null }
        if y != top and y != bottom { return null }
        let next = (index + 1) % 4
        let nx = run[next * 2]
        let ny = run[next * 2 + 1]
        if x != nx and y != ny { return null }
    }
    if right <= left or bottom <= top { return null }
    return (left, top, right, bottom)
}

#: Every edge of every subpath, implicitly closed.
#:
#: A fill treats an open subpath as closed, which is what makes a polygon
#: described by its corners alone come out solid.
pub fn _edges_of(path: Path) -> list[_Edge] {
    var edges = []
    for run in path.subpaths() {
        let count = run.len() // 2
        if count < 2 { continue }
        for index in 0..count {
            let next = (index + 1) % count
            let x0 = run[index * 2]
            let y0 = run[index * 2 + 1]
            let x1 = run[next * 2]
            let y1 = run[next * 2 + 1]
            # A horizontal edge contributes no crossing and would only divide by
            # zero, so it is dropped rather than special-cased downstream.
            if y0 != y1 { edges.push(_Edge(x0, y0, x1, y1)) }
        }
    }
    return edges
}

#: Where the edges cross the line at `y`, sorted left to right.
#:
#: Each entry is the x position paired with the edge's direction, which the
#: winding rule needs.
pub fn _crossings(edges: list[_Edge], y: float) -> list[tuple[float, int]] {
    var hits = []
    for edge in edges {
        # Half-open in y: a vertex shared by two edges is counted once, so
        # abutting shapes neither double-cover nor leave a seam.
        if y >= edge.y0 and y < edge.y1 { hits.push((edge.crossing(y), edge.direction)) }
    }
    hits.sort(|entry| entry[0])
    return hits
}

#: The intervals a sub-scanline covers, in device x.
#:
#: Separating this from the accumulation is what lets a row notice that its
#: interior is solid and hand a whole run to the native span writer instead of
#: incrementing a coverage value per pixel.
pub fn _spans(hits: list[tuple[float, int]], winding: Winding) -> list[tuple[float, float]] {
    var out = []
    var count = 0
    for index in 0..hits.len() - 1 {
        let (x, direction) = hits[index]
        count += match winding {
            Winding.NonZero => direction,
            Winding.EvenOdd => 1,
        }
        let inside = match winding {
            Winding.NonZero => count != 0,
            Winding.EvenOdd => count % 2 != 0,
        }
        if not inside { continue }
        let (next_x, _next_direction) = hits[index + 1]
        if next_x > x { out.push((x, next_x)) }
    }
    return out
}

#: Add one sub-scanline's spans into `coverage`, returning whether any landed.
fn _accumulate(
    coverage: list[float],
    hits: list[tuple[float, int]],
    winding: Winding,
    first_column: int,
    span: int,
    share: float
) -> bool {
    var touched = false
    var count = 0
    for index in 0..hits.len() - 1 {
        let (x, direction) = hits[index]
        count += match winding {
            Winding.NonZero => direction,
            Winding.EvenOdd => 1,
        }
        let inside = match winding {
            Winding.NonZero => count != 0,
            Winding.EvenOdd => count % 2 != 0,
        }
        if not inside { continue }
        let (next_x, _next_direction) = hits[index + 1]
        if _add_span(
            coverage,
            x - first_column.to_float(),
            next_x - first_column.to_float(),
            span,
            share
        ) {
            touched = true
        }
    }
    return touched
}

#: Add `share` of coverage over `[from, to)`, exactly at the fractional ends.
#:
#: The partial pixels at each end are what make the edge smooth: a span ending
#: three tenths of the way into a pixel contributes three tenths of the share to
#: it.
pub fn _add_span(coverage: list[float], start: float, end: float, span: int, share: float) -> bool {
    let left = max(start, 0.0)
    let right = min(end, span.to_float())
    if right <= left { return false }

    let first = int(math.floor(left))
    let last = int(math.ceil(right)) - 1
    if first > last { return false }

    for column in first..=last {
        if column < 0 or column >= span { continue }
        let low = max(left, column.to_float())
        let high = min(right, column.to_float() + 1.0)
        if high > low { coverage[column] += share * (high - low) }
    }
    return true
}

#: Split one subpath into the pieces a dash pattern leaves visible.
#:
#: An empty pattern returns the run unchanged, which is the solid case and the
#: common one.
pub fn _dashed(run: list[float], dash: list[float]) -> list[list[float]] {
    if dash.len() == 0 { return [run] }
    var total = 0.0
    for length in dash { total += length }
    if total <= 0.0 { return [run] }

    var pieces = []
    var current = []
    var index = 0
    var remaining = dash[0]
    var on = true

    let count = run.len() // 2
    for segment in 0..count - 1 {
        let x0 = run[segment * 2]
        let y0 = run[segment * 2 + 1]
        let x1 = run[(segment + 1) * 2]
        let y1 = run[(segment + 1) * 2 + 1]
        let length = math.hypot(x1 - x0, y1 - y0)
        if length <= 0.0 { continue }
        let ux = (x1 - x0) / length
        let uy = (y1 - y0) / length

        var travelled = 0.0
        while travelled < length {
            let step = min(length - travelled, remaining)
            if on {
                if current.len() == 0 {
                    current.push(x0 + ux * travelled)
                    current.push(y0 + uy * travelled)
                }
                current.push(x0 + ux * (travelled + step))
                current.push(y0 + uy * (travelled + step))
            }
            travelled += step
            remaining -= step
            # Floating point will not land exactly on zero, and a pattern that
            # never advanced would spin here for ever.
            if remaining <= 1e-9 {
                if on and current.len() >= 4 { pieces.push(current) }
                if on { current = [] }
                on = not on
                index = (index + 1) % dash.len()
                remaining = dash[index]
            }
        }
    }
    if current.len() >= 4 { pieces.push(current) }
    return pieces
}

#: Add the outline of one stroked run into `outline`.
#:
#: The run becomes a single closed ring: out along one offset side and back
#: along the other. That is half the edges of a quad per segment, and one
#: subpath instead of hundreds, which is most of the cost of stroking a sampled
#: curve. Sharp corners still get a disc, because a mitre there would either
#: shoot off to infinity or leave a notch.
pub fn _outline_run(outline: Path, run: list[float], width: float, cap: Cap, join: Join) -> void {
    let half = width * 0.5
    let count = run.len() // 2
    if count < 2 { return }

    var extended = [value for value in run]
    if cap == Cap.Square {
        _extend_ends(extended, half)
    }

    let left = _offset_side(extended, half)
    let right = _offset_side(_reversed_run(extended), half)
    if left.len() < 2 or right.len() < 2 { return }

    var ring = [point for point in left]
    for point in right { ring.push(point) }
    _add_ring(outline, ring)

    if join != Join.Bevel and count > 2 {
        for vertex in 1..count - 1 {
            # A join only has a wedge to fill when the path actually turns. A
            # sampled curve turns by a fraction of a degree between samples, so
            # this is rare and the ring covers the rest.
            if _turns_visibly(extended, vertex, half) {
                _add_disc(outline, extended[vertex * 2], extended[vertex * 2 + 1], half)
            }
        }
    }
    if cap == Cap.Round {
        _add_disc(outline, extended[0], extended[1], half)
        _add_disc(outline, extended[(count - 1) * 2], extended[(count - 1) * 2 + 1], half)
    }
}

#: One side of a stroke, offset by `half` along each vertex's bisector.
fn _offset_side(run: list[float], half: float) -> list[tuple[float, float]] {
    let count = run.len() // 2
    var out = []
    for index in 0..count {
        let (nx, ny) = _vertex_normal(run, index, count)
        if nx == 0.0 and ny == 0.0 { continue }
        out.push((run[index * 2] + nx * half, run[index * 2 + 1] + ny * half))
    }
    return out
}

#: The outward normal at a vertex: the segment's own at an end, the bisector of
#: its two segments in the middle.
#:
#: The bisector is scaled so the offset side stays `half` from both segments,
#: and clamped because that scale grows without bound as a corner tightens.
fn _vertex_normal(run: list[float], index: int, count: int) -> tuple[float, float] {
    var before_x = 0.0
    var before_y = 0.0
    var after_x = 0.0
    var after_y = 0.0
    if index > 0 {
        (before_x, before_y) = _unit_normal(run, index - 1, index)
    }
    if index < count - 1 {
        (after_x, after_y) = _unit_normal(run, index, index + 1)
    }
    if index == 0 { return (after_x, after_y) }
    if index == count - 1 { return (before_x, before_y) }

    var mx = before_x + after_x
    var my = before_y + after_y
    let length = math.hypot(mx, my)
    if length < 1e-9 { return (after_x, after_y) }
    mx = mx / length
    my = my / length
    let scale = min(4.0, 1.0 / max(1e-6, mx * after_x + my * after_y))
    return (mx * scale, my * scale)
}

#: The unit normal of the segment between two vertices.
fn _unit_normal(run: list[float], from_index: int, to_index: int) -> tuple[float, float] {
    let dx = run[to_index * 2] - run[from_index * 2]
    let dy = run[to_index * 2 + 1] - run[from_index * 2 + 1]
    let length = math.hypot(dx, dy)
    if length <= 0.0 { return (0.0, 0.0) }
    return (-dy / length, dx / length)
}

#: The same run walked backwards, so the second side offsets the other way.
fn _reversed_run(run: list[float]) -> list[float] {
    var out = []
    var index = run.len() // 2 - 1
    while index >= 0 {
        out.push(run[index * 2])
        out.push(run[index * 2 + 1])
        index -= 1
    }
    return out
}

#: Push the two ends outward by `half`, which is what a square cap is.
fn _extend_ends(run: list[float], half: float) -> void {
    let count = run.len() // 2
    if count < 2 { return }
    let (hx, hy) = _direction(run, 1, 0)
    run[0] = run[0] + hx * half
    run[1] = run[1] + hy * half
    let (tx, ty) = _direction(run, count - 2, count - 1)
    run[(count - 1) * 2] = run[(count - 1) * 2] + tx * half
    run[(count - 1) * 2 + 1] = run[(count - 1) * 2 + 1] + ty * half
}

#: The unit direction from one vertex to another.
fn _direction(run: list[float], from_index: int, to_index: int) -> tuple[float, float] {
    let dx = run[to_index * 2] - run[from_index * 2]
    let dy = run[to_index * 2 + 1] - run[from_index * 2 + 1]
    let length = math.hypot(dx, dy)
    if length <= 0.0 { return (0.0, 0.0) }
    return (dx / length, dy / length)
}

#: Report whether the corner at `vertex` leaves a gap worth filling.
#:
#: The notch a bevelled join leaves is about `half * (1 - cos(turn / 2))` deep.
#: Below a quarter of a pixel nothing is visible, so the disc is skipped.
pub fn _turns_visibly(run: list[float], vertex: int, half: float) -> bool {
    let ax = run[vertex * 2] - run[(vertex - 1) * 2]
    let ay = run[vertex * 2 + 1] - run[(vertex - 1) * 2 + 1]
    let bx = run[(vertex + 1) * 2] - run[vertex * 2]
    let by = run[(vertex + 1) * 2 + 1] - run[vertex * 2 + 1]
    let first = math.hypot(ax, ay)
    let second = math.hypot(bx, by)
    if first <= 0.0 or second <= 0.0 { return false }

    let cosine = (ax * bx + ay * by) / (first * second)
    let clamped = max(-1.0, min(1.0, cosine))
    # cos(turn/2) from cos(turn), avoiding an arccos and a second cosine.
    let half_cosine = math.sqrt(max(0.0, (1.0 + clamped) * 0.5))
    return half * (1.0 - half_cosine) > 0.25
}

#: Append a filled disc, approximated by a polygon fine enough for its size.
fn _add_disc(outline: Path, x: float, y: float, radius: float) -> void {
    if radius <= 0.0 { return }
    let steps = max(6, min(24, int(radius * 4.0)))
    var points = []
    for index in 0..steps {
        let angle = math.TAU * index.to_float() / steps.to_float()
        points.push((x + radius * math.cos(angle), y + radius * math.sin(angle)))
    }
    _add_ring(outline, points)
}

#: Append a ring, turned so every piece of a stroke winds the same way.
#:
#: Non-zero winding cancels where two rings of opposite orientation overlap. A
#: stroke's quads take their orientation from the direction of their segment, so
#: a curve that doubles back produces opposing quads and the overlap punches a
#: hole in the line. Normalising the orientation here is what keeps a stroked
#: path solid.
fn _add_ring(outline: Path, points: list[tuple[float, float]]) -> void {
    if points.len() < 3 { return }
    var twice_area = 0.0
    for index in 0..points.len() {
        let (x0, y0) = points[index]
        let (x1, y1) = points[(index + 1) % points.len()]
        twice_area += x0 * y1 - x1 * y0
    }
    if twice_area < 0.0 {
        var reversed = []
        var index = points.len() - 1
        while index >= 0 {
            reversed.push(points[index])
            index -= 1
        }
        let _ring = outline.polygon(reversed)
        return
    }
    let _ring = outline.polygon(points)
}
