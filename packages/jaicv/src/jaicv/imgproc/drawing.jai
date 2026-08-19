#: Lines, rectangles, circles, ellipses, and polygons.
#:
#: These are the one family in jaicv that runs on the host rather than the
#: device. A shape touches a handful of pixels scattered across the image, so a
#: kernel launch per shape would cost more than the drawing does; instead the
#: rasteriser writes into the host mirror and one upload puts the result back.
#: A `Mat` that is only drawn on never leaves the host at all.
#:
#: The algorithms are OpenCV's, down to the rounding: Bresenham for `LINE_8`
#: and `LINE_4`, the midpoint circle, `ellipse2Poly` followed by a polygon
#: fill, and a thick line as a filled quadrilateral with a disc at each end.
#: Coordinates are carried in fixed point with sixteen fractional bits, which
#: is the space `shift` selects into.
#:
#: Everything one pixel wide, every filled polygon, and every filled rectangle
#: or circle comes out pixel for pixel identical to OpenCV. Two cases do not
#: quite: a stroke wider than one pixel and a filled ellipse, both of which go
#: through the fixed-point convex fill, land in the same place and at the same
#: width but disagree on a boundary pixel here and there — OpenCV's span
#: rounding along the two long edges does not follow from its algorithm as
#: published. `tests/test_against_opencv.jai` measures how far, and holds the
#: gap where it is.
#:
#: `LINE_AA` departs further: coverage here comes from the distance to the
#: ideal shape rather than from OpenCV's slope-correction tables, so an
#: anti-aliased shape agrees in position and width but not in every
#: intermediate coverage level.
#:
#: ```
#: rectangle(frame, Point(10, 10), Point(90, 60), Scalar(0.0, 255.0, 0.0), 2)
#: circle(frame, Point(50, 35), 12, Scalar(0.0, 0.0, 255.0), FILLED)
#: ```
import std.math
from ..core.mat import Mat
from ..core.types import depth_is_integer, depth_max, depth_min
from ..core.types import LINE_4, LINE_8, LINE_AA, Point, Point2f, Rect, RotatedRect, Scalar, Size

#: Fractional bits in the fixed-point coordinates the rasteriser works in.
pub const XY_SHIFT = 16
#: One whole pixel in fixed point.
pub const XY_ONE = 65536

#: A crosshair.
pub const MARKER_CROSS = 0
#: A crosshair rotated by forty-five degrees.
pub const MARKER_TILTED_CROSS = 1
#: A star, which is both crosses at once.
pub const MARKER_STAR = 2
#: A diamond.
pub const MARKER_DIAMOND = 3
#: A square.
pub const MARKER_SQUARE = 4
#: A triangle pointing up.
pub const MARKER_TRIANGLE_UP = 5
#: A triangle pointing down.
pub const MARKER_TRIANGLE_DOWN = 6

#: Largest `thickness` any of these accept, which is OpenCV's cap.
pub const MAX_THICKNESS = 32767

#: Ask a thick line for a disc at the end it starts from.
pub const CAP_START = 1
#: Ask a thick line for a disc at the end it finishes at.
pub const CAP_END = 2

fn trunc_div(a: int, b: int) -> int {
    let quotient = a // b
    if a % b != 0 and (a < 0) != (b < 0) { return quotient + 1 }
    return quotient
}

#: Round to nearest, ties to even, which is what `cvRound` does and what every
#: coordinate below is rounded with.
fn cv_round(value: float) -> int {
    let down = math.floor(value)
    let fraction = value - down
    let lower = int(down)
    if fraction > 0.5 { return lower + 1 }
    if fraction < 0.5 { return lower }
    return if lower % 2 == 0 { lower } else { lower + 1 }
}

fn to_fixed(point: Point, shift: int) -> Point {
    return Point(point.x << XY_SHIFT - shift, point.y << XY_SHIFT - shift)
}

fn from_fixed(point: Point) -> Point {
    return Point(point.x + (XY_ONE >> 1) >> XY_SHIFT, point.y + (XY_ONE >> 1) >> XY_SHIFT)
}

#: The surface a shape draws onto: an image, its host mirror, and one colour
#: already saturated into the image's depth.
#:
#: Every primitive below goes through this, so bounds checking and the blend
#: rule are written once.
pub class Painter {
    pub let image: Mat
    pub let values: list[float]
    pub let rows: int
    pub let cols: int
    pub let cn: int
    pub let low: float
    pub let high: float
    pub let integral: bool
    pub let colour: list[float]

    pub fn init(self, image: Mat, colour: Scalar) {
        self.image = image
        self.values = image.edit()
        self.rows = image.rows
        self.cols = image.cols
        self.cn = image.channels()
        let depth = image.depth()
        self.low = depth_min(depth)
        self.high = depth_max(depth)
        self.integral = depth_is_integer(depth)
        var saturated: list[float] = []
        for channel in 0..self.cn { saturated.push(self.saturate(colour[min(channel, 3)])) }
        self.colour = saturated
    }

    fn saturate(self, value: float) -> float {
        if not self.integral { return value }
        var rounded = float(cv_round(value))
        if rounded < self.low { rounded = self.low }
        if rounded > self.high { rounded = self.high }
        return rounded
    }

    #: Write one pixel opaquely.
    pub fn plot(self, x: int, y: int) -> void {
        if x < 0 or y < 0 or x >= self.cols or y >= self.rows { return }
        let base = (y * self.cols + x) * self.cn
        for channel in 0..self.cn { self.values[base + channel] = self.colour[channel] }
    }

    #: Write one pixel with `alpha` of the colour over what is already there.
    pub fn blend(self, x: int, y: int, alpha: float) -> void {
        if x < 0 or y < 0 or x >= self.cols or y >= self.rows { return }
        if alpha <= 0.0 { return }
        if alpha >= 1.0 {
            self.plot(x, y)
            return
        }
        let base = (y * self.cols + x) * self.cn
        for channel in 0..self.cn {
            let mixed = self.values[base + channel] * (1.0 - alpha) + self.colour[channel] * alpha
            self.values[base + channel] = self.saturate(mixed)
        }
    }

    #: Fill `[x1, x2]` on row `y`, inclusive at both ends.
    pub fn span(self, x1: int, x2: int, y: int) -> void {
        if y < 0 or y >= self.rows { return }
        var start = max(x1, 0)
        let end = min(x2, self.cols - 1)
        while start <= end {
            self.plot(start, y)
            start += 1
        }
    }

    #: Push everything written back to the device.
    pub fn finish(self) -> void { self.image.flush() }
}

#: Clip a segment to a `width x height` rectangle.
#:
#: Returns whether any of it survives, along with the clipped endpoints. This
#: is OpenCV's `clipLine`, so the endpoints land on the same pixels.
pub fn clip_line(size: Size, pt1: Point, pt2: Point) -> tuple[bool, Point, Point] {
    if size.width <= 0 or size.height <= 0 { return (false, pt1, pt2) }
    let right = size.width - 1
    let bottom = size.height - 1

    var x1 = pt1.x
    var y1 = pt1.y
    var x2 = pt2.x
    var y2 = pt2.y
    var c1 = (if x1 < 0 { 1 } else { 0 }) + (if x1 > right { 2 } else { 0 })
    c1 += (if y1 < 0 { 4 } else { 0 }) + (if y1 > bottom { 8 } else { 0 })
    var c2 = (if x2 < 0 { 1 } else { 0 }) + (if x2 > right { 2 } else { 0 })
    c2 += (if y2 < 0 { 4 } else { 0 }) + (if y2 > bottom { 8 } else { 0 })

    if c1 & c2 == 0 and c1 | c2 != 0 {
        if c1 & 12 != 0 {
            let a = if c1 < 8 { 0 } else { bottom }
            x1 += trunc_div((a - y1) * (x2 - x1), y2 - y1)
            y1 = a
            c1 = (if x1 < 0 { 1 } else { 0 }) + (if x1 > right { 2 } else { 0 })
        }
        if c2 & 12 != 0 {
            let a = if c2 < 8 { 0 } else { bottom }
            x2 += trunc_div((a - y2) * (x2 - x1), y2 - y1)
            y2 = a
            c2 = (if x2 < 0 { 1 } else { 0 }) + (if x2 > right { 2 } else { 0 })
        }
        if c1 & c2 == 0 and c1 | c2 != 0 {
            if c1 != 0 {
                let a = if c1 == 1 { 0 } else { right }
                y1 += trunc_div((a - x1) * (y2 - y1), x2 - x1)
                x1 = a
                c1 = 0
            }
            if c2 != 0 {
                let a = if c2 == 1 { 0 } else { right }
                y2 += trunc_div((a - x2) * (y2 - y1), x2 - x1)
                x2 = a
                c2 = 0
            }
        }
    }
    return (c1 | c2 == 0, Point(x1, y1), Point(x2, y2))
}

#: Every pixel a `LINE_8` or `LINE_4` segment covers, in order, already clipped
#: to `size`.
#:
#: This is OpenCV's `LineIterator`, and walking it is how a line one pixel wide
#: is drawn.
pub fn line_points(size: Size, pt1: Point, pt2: Point, connectivity: int = 8) -> list[Point] {
    let (visible, clipped_a, clipped_b) = clip_line(size, pt1, pt2)
    var out: list[Point] = []
    if not visible { return out }

    #: OpenCV walks every segment left to right, and does so after clipping
    #: rather than before, which is what makes its lines come out the same
    #: whichever end the caller names first. Both halves matter: normalising
    #: before the clip picks a different pair of endpoints on a segment that
    #: leaves through two borders, and not normalising rounds ties the other
    #: way on any segment that runs right to left.
    var a = clipped_a
    var b = clipped_b
    if b.x < a.x or b.x == a.x and b.y < a.y {
        a = clipped_b
        b = clipped_a
    }

    var dx = b.x - a.x
    var dy = b.y - a.y
    let step_x = if dx < 0 { -1 } else { 1 }
    let step_y = if dy < 0 { -1 } else { 1 }
    if dx < 0 { dx = -dx }
    if dy < 0 { dy = -dy }

    #: The major axis leads; swapping the deltas and the steps together is what
    #: lets one loop serve all eight octants.
    if dy > dx { return walk(a, dy, dx, 0, step_y, step_x, 0, connectivity) }
    return walk(a, dx, dy, step_x, 0, 0, step_y, connectivity)
}

fn walk(
    start: Point,
    dx: int,
    dy: int,
    major_x: int,
    major_y: int,
    minor_x: int,
    minor_y: int,
    connectivity: int
) -> list[Point] {
    var out: list[Point] = []
    var x = start.x
    var y = start.y
    if connectivity == 4 {
        var err = 0
        let plus_delta = dx + dx + dy + dy
        let minus_delta = -(dy + dy)
        for _index in 0..dx + dy + 1 {
            out.push(Point(x, y))
            let sideways = err < 0
            err += minus_delta + (if sideways { plus_delta } else { 0 })
            if sideways {
                x += minor_x
                y += minor_y
            } else {
                x += major_x
                y += major_y
            }
        }
        return out
    }
    var err = dx - (dy + dy)
    let plus_delta = dx + dx
    let minus_delta = -(dy + dy)
    for _index in 0..dx + 1 {
        out.push(Point(x, y))
        let diagonal = err < 0
        err += minus_delta + (if diagonal { plus_delta } else { 0 })
        x += major_x + (if diagonal { minor_x } else { 0 })
        y += major_y + (if diagonal { minor_y } else { 0 })
    }
    return out
}

fn draw_thin_line(painter: Painter, pt1: Point, pt2: Point, line_type: int) -> void {
    if line_type == LINE_AA {
        draw_line_aa(painter, pt1, pt2)
        return
    }
    let connectivity = if line_type == LINE_4 { 4 } else { 8 }
    for point in line_points(Size(painter.cols, painter.rows), pt1, pt2, connectivity) {
        painter.plot(point.x, point.y)
    }
}

fn draw_line_aa(painter: Painter, pt1: Point, pt2: Point) -> void {
    let dx = float(pt2.x - pt1.x)
    let dy = float(pt2.y - pt1.y)
    let steps = max(int(math.fabs(dx)), int(math.fabs(dy)))
    if steps == 0 {
        painter.blend(pt1.x, pt1.y, 1.0)
        return
    }
    let step_x = dx / float(steps)
    let step_y = dy / float(steps)
    var x = float(pt1.x)
    var y = float(pt1.y)
    for _index in 0..steps + 1 {
        let ix = int(math.floor(x))
        let iy = int(math.floor(y))
        let fx = x - float(ix)
        let fy = y - float(iy)
        painter.blend(ix, iy, (1.0 - fx) * (1.0 - fy))
        painter.blend(ix + 1, iy, fx * (1.0 - fy))
        painter.blend(ix, iy + 1, (1.0 - fx) * fy)
        painter.blend(ix + 1, iy + 1, fx * fy)
        x += step_x
        y += step_y
    }
}

#: One edge of a polygon, held in the active list a scanline fill walks.
class PolyEdge {
    pub var y0: int
    pub var y1: int
    pub var x: int
    pub var dx: int
    pub var next: PolyEdge?

    pub fn init(self, y0: int = 0, y1: int = 0, x: int = 0, dx: int = 0) {
        self.y0 = y0
        self.y1 = y1
        self.x = x
        self.dx = dx
        self.next = null
    }
}

fn collect_edges(
    painter: Painter,
    points: list[Point],
    edges: list[PolyEdge],
    line_type: int,
    shift: int
) -> void {
    let count = points.len()
    if count == 0 { return }
    let delta = 1 << shift >> 1
    var previous_x = points[count - 1].x << XY_SHIFT - shift
    var previous_y = points[count - 1].y + delta >> shift
    for index in 0..count {
        let current_x = points[index].x << XY_SHIFT - shift
        let current_y = points[index].y + delta >> shift
        #: The outline is drawn as well as the fill, which is what stops a
        #: nearly horizontal edge from falling between two scanlines.
        if line_type != LINE_AA {
            let a = Point(previous_x + (XY_ONE >> 1) >> XY_SHIFT, previous_y)
            let b = Point(current_x + (XY_ONE >> 1) >> XY_SHIFT, current_y)
            draw_thin_line(painter, a, b, line_type)
        }
        if previous_y != current_y {
            let slope = trunc_div(current_x - previous_x, current_y - previous_y)
            let edge = if previous_y < current_y {
                PolyEdge(previous_y, current_y, previous_x, slope)
            } else {
                PolyEdge(current_y, previous_y, current_x, slope)
            }
            edges.push(edge)
        }
        previous_x = current_x
        previous_y = current_y
    }
}

fn edge_precedes(a: PolyEdge, b: PolyEdge) -> bool {
    if a.y0 != b.y0 { return a.y0 < b.y0 }
    if a.x != b.x { return a.x < b.x }
    return a.dx < b.dx
}

fn sort_edges(edges: list[PolyEdge]) -> list[PolyEdge] {
    var out: list[PolyEdge] = []
    for edge in edges {
        var at = out.len()
        while at > 0 and edge_precedes(edge, out[at - 1]) { at -= 1 }
        out.insert(at, edge)
    }
    return out
}

fn fill_edges(painter: Painter, unsorted: list[PolyEdge]) -> void {
    if unsorted.len() < 2 { return }
    let edges = sort_edges(unsorted)
    let total = edges.len()

    var y_min = edges[0].y0
    var y_max = edges[0].y1
    for edge in edges {
        if edge.y0 < y_min { y_min = edge.y0 }
        if edge.y1 > y_max { y_max = edge.y1 }
    }
    if y_max < 0 or y_min >= painter.rows { return }
    y_max = min(y_max, painter.rows)

    #: `head` is a sentinel, so an edge can leave the active list without a
    #: special case for the front of it.
    let head = PolyEdge()
    var next_edge = 0
    var y = edges[0].y0
    while y < y_max {
        var previous = head
        var current = head.next
        var draw = false
        while true {
            let waiting = next_edge < total and edges[next_edge].y0 == y
            if current is null and not waiting { break }
            let live = current
            if live is not null and live.y1 == y {
                previous.next = live.next
                current = live.next
                continue
            }
            let keep = previous
            let entering = next_edge < total
            let take_live = live is not null and
                (not entering or edges[next_edge].y0 > y or live.x < edges[next_edge].x)
            if take_live {
                let step = live
                if step is null { break }
                previous = step
                current = step.next
            } elif entering {
                let arrival = edges[next_edge]
                previous.next = arrival
                arrival.next = live
                previous = arrival
                next_edge += 1
            } else {
                break
            }
            if draw {
                if y >= 0 {
                    var x1 = 0
                    var x2 = 0
                    if keep.x > previous.x {
                        x1 = previous.x + XY_ONE - 1 >> XY_SHIFT
                        x2 = keep.x >> XY_SHIFT
                    } else {
                        x1 = keep.x + XY_ONE - 1 >> XY_SHIFT
                        x2 = previous.x >> XY_SHIFT
                    }
                    painter.span(x1, x2, y)
                }
                keep.x += keep.dx
                previous.x += previous.dx
            }
            draw = not draw
        }
        resort_active(head)
        y += 1
    }
}

fn resort_active(head: PolyEdge) -> void {
    #: The active list has to stay sorted by `x`; a bubble pass is enough,
    #: because one scanline moves each edge by at most its own slope.
    var swapped = true
    while swapped {
        swapped = false
        var previous = head
        var current = head.next
        while current is not null {
            let node = current
            if node is null { break }
            let following = node.next
            if following is null { break }
            if node.x > following.x {
                previous.next = following
                node.next = following.next
                following.next = node
                previous = following
                swapped = true
            } else {
                previous = node
                current = following
            }
        }
    }
}

fn fill_convex(painter: Painter, points: list[Point], line_type: int, shift: int) -> void {
    let count = points.len()
    if count < 3 { return }
    let delta = 1 << shift >> 1
    let delta1 = if line_type == LINE_AA { XY_ONE - 1 } else { XY_ONE >> 1 }
    let delta2 = if line_type == LINE_AA { 0 } else { XY_ONE >> 1 }

    var x_min = points[0].x
    var x_max = points[0].x
    var y_min = points[0].y
    var y_max = points[0].y
    var top = 0
    for index in 0..count {
        let p = points[index]
        if p.y < y_min {
            y_min = p.y
            top = index
        }
        if p.y > y_max { y_max = p.y }
        if p.x > x_max { x_max = p.x }
        if p.x < x_min { x_min = p.x }
    }

    #: Only an unshifted polygon gets its outline drawn; a shifted one is
    #: already sub-pixel and the spans alone cover it.
    if shift == 0 {
        var previous = points[count - 1]
        for index in 0..count {
            draw_thin_line(painter, previous, points[index], line_type)
            previous = points[index]
        }
    }

    x_min = x_min + delta >> shift
    x_max = x_max + delta >> shift
    y_min = y_min + delta >> shift
    y_max = y_max + delta >> shift
    if x_max < 0 or y_max < 0 or x_min >= painter.cols or y_min >= painter.rows { return }
    y_max = min(y_max, painter.rows - 1)

    var index_of = [top, top]
    let direction = [1, count - 1]
    var end_y = [y_min, y_min]
    var edge_x = [-XY_ONE, -XY_ONE]
    var edge_dx = [0, 0]
    var remaining = count
    var y = y_min

    while y <= y_max {
        var exhausted = false
        for side in 0..2 {
            if y < end_y[side] { continue }
            var idx = index_of[side]
            let di = direction[side]
            var xs = 0
            var ty = 0
            while true {
                ty = points[idx].y + delta >> shift
                if ty > y or remaining == 0 { break }
                xs = points[idx].x
                idx += di
                if idx >= count { idx -= count }
                remaining -= 1
            }
            if y >= ty {
                #: Both chains have reached the last vertex; this row is the
                #: bottom of the polygon and still belongs to it.
                exhausted = true
                break
            }
            end_y[side] = ty
            let start = xs << XY_SHIFT - shift
            let finish = points[idx].x << XY_SHIFT - shift
            edge_dx[side] = trunc_div((finish - start) * 2 + (ty - y), 2 * (ty - y))
            edge_x[side] = start
            index_of[side] = idx
        }
        if y >= 0 {
            let left = if edge_x[0] > edge_x[1] { 1 } else { 0 }
            let right = 1 - left
            painter.span(edge_x[left] + delta1 >> XY_SHIFT, edge_x[right] + delta2 >> XY_SHIFT, y)
        }
        if exhausted { return }
        edge_x[0] += edge_dx[0]
        edge_x[1] += edge_dx[1]
        y += 1
    }
}

fn draw_circle(painter: Painter, centre: Point, radius: int, filled: bool, line_type: int) -> void {
    if radius < 0 { return }
    if line_type == LINE_AA {
        draw_circle_aa(painter, centre, radius, filled)
        return
    }
    #: The midpoint circle, stepping one octant and mirroring it eight ways.
    var err = 0
    var dx = radius
    var dy = 0
    var plus = 1
    var minus = (radius << 1) - 1

    while dx >= dy {
        let y11 = centre.y - dy
        let y12 = centre.y + dy
        let y21 = centre.y - dx
        let y22 = centre.y + dx
        let x11 = centre.x - dx
        let x12 = centre.x + dx
        let x21 = centre.x - dy
        let x22 = centre.x + dy

        if filled {
            painter.span(x11, x12, y11)
            painter.span(x11, x12, y12)
            painter.span(x21, x22, y21)
            painter.span(x21, x22, y22)
        } else {
            painter.plot(x11, y11)
            painter.plot(x12, y11)
            painter.plot(x11, y12)
            painter.plot(x12, y12)
            painter.plot(x21, y21)
            painter.plot(x22, y21)
            painter.plot(x21, y22)
            painter.plot(x22, y22)
        }

        dy += 1
        err += plus
        plus += 2
        if err > 0 {
            err -= minus
            dx -= 1
            minus -= 2
        }
    }
}

fn clamp01(value: float) -> float {
    if value < 0.0 { return 0.0 }
    if value > 1.0 { return 1.0 }
    return value
}

fn draw_circle_aa(painter: Painter, centre: Point, radius: int, filled: bool) -> void {
    let r = float(radius)
    let reach = radius + 2
    for offset_y in -reach..reach + 1 {
        for offset_x in -reach..reach + 1 {
            let distance = math.sqrt(float(offset_x * offset_x + offset_y * offset_y))
            let coverage = if filled {
                clamp01(r + 0.5 - distance)
            } else {
                clamp01(1.0 - math.fabs(distance - r))
            }
            if coverage > 0.0 { painter.blend(centre.x + offset_x, centre.y + offset_y, coverage) }
        }
    }
}

fn draw_thick(
    painter: Painter,
    p0: Point,
    p1: Point,
    thickness: int,
    line_type: int,
    caps: int
) -> void {
    if thickness <= 1 {
        draw_thin_line(painter, from_fixed(p0), from_fixed(p1), line_type)
        return
    }
    #: A thick segment is the quadrilateral a pen of that width sweeps, filled
    #: in fixed point so its edges land where OpenCV puts them, with a disc at
    #: whichever ends `caps` names so that joins come out round.
    let dx = float(p0.x - p1.x) / float(XY_ONE)
    let dy = float(p1.y - p0.y) / float(XY_ONE)
    let squared = dx * dx + dy * dy
    let odd = thickness & 1
    let half_width = thickness << XY_SHIFT - 1
    if squared > 1e-12 {
        let reach = (float(half_width) + float(odd) * float(XY_ONE) * 0.5) / math.sqrt(squared)
        let ox = cv_round(dy * reach)
        let oy = cv_round(dx * reach)
        fill_convex(
            painter,
            [
                Point(p0.x + ox, p0.y + oy),
                Point(p0.x - ox, p0.y - oy),
                Point(p1.x - ox, p1.y - oy),
                Point(p1.x + ox, p1.y + oy),
            ],
            line_type,
            XY_SHIFT
        )
    }
    let radius = half_width + (XY_ONE >> 1) >> XY_SHIFT
    var end = p0
    for index in 0..2 {
        if caps & 1 << index != 0 { draw_circle(painter, from_fixed(end), radius, true, line_type) }
        end = p1
    }
}

fn poly_line(
    painter: Painter,
    points: list[Point],
    closed: bool,
    thickness: int,
    line_type: int,
    shift: int
) -> void {
    let count = points.len()
    if count == 0 { return }
    if count == 1 {
        let only = to_fixed(points[0], shift)
        draw_thick(painter, only, only, thickness, line_type, CAP_START | CAP_END)
        return
    }
    #: A closed run caps only the far end of each segment, since the near end
    #: was capped by the segment before it; an open run caps both on the first.
    var caps = if closed { CAP_END } else { CAP_START | CAP_END }
    var previous = if closed { points[count - 1] } else { points[0] }
    let first = if closed { 0 } else { 1 }
    for index in first..count {
        let current = points[index]
        draw_thick(
            painter,
            to_fixed(previous, shift),
            to_fixed(current, shift),
            thickness,
            line_type,
            caps
        )
        previous = current
        caps = CAP_END
    }
}

fn arc_points(
    centre_x: float,
    centre_y: float,
    axis_x: float,
    axis_y: float,
    angle: int,
    arc_start: int,
    arc_end: int,
    delta: int
) -> list[Point2f] {
    if delta <= 0 or delta > 180 { throw ValueError("an arc step has to be between 1 and 180") }
    var rotation = angle
    while rotation < 0 { rotation += 360 }
    while rotation > 360 { rotation -= 360 }

    var start = arc_start
    var end = arc_end
    if start > end {
        let swap = start
        start = end
        end = swap
    }
    while start < 0 {
        start += 360
        end += 360
    }
    while end > 360 {
        end -= 360
        start -= 360
    }
    if end - start > 360 {
        start = 0
        end = 360
    }

    let radians = float(rotation) * math.PI / 180.0
    let alpha = math.cos(radians)
    let beta = math.sin(radians)

    var out: list[Point2f] = []
    var step = start
    while step < end + delta {
        var at = step
        if at > end { at = end }
        if at < 0 { at += 360 }
        let theta = float(at) * math.PI / 180.0
        let x = axis_x * math.cos(theta)
        let y = axis_y * math.sin(theta)
        out.push(Point2f(centre_x + x * alpha - y * beta, centre_y + x * beta + y * alpha))
        step += delta
    }
    if out.len() == 1 { out.push(Point2f(centre_x, centre_y)) }
    return out
}

fn rounded_unique(raw: list[Point2f]) -> list[Point] {
    var out: list[Point] = []
    var have = false
    var previous = Point(0, 0)
    for point in raw {
        let rounded = Point(cv_round(point.x), cv_round(point.y))
        if have and rounded.x == previous.x and rounded.y == previous.y { continue }
        out.push(rounded)
        previous = rounded
        have = true
    }
    return out
}

#: The polygon that approximates an elliptical arc.
#:
#: `angle` rotates the ellipse, `arc_start` and `arc_end` bound the arc, and
#: `delta` is the angular step in degrees. This is OpenCV's `ellipse2Poly`,
#: including its habit of dropping a point that rounds onto the one before it.
pub fn ellipse_2_poly(
    centre: Point,
    axes: Size,
    angle: int,
    arc_start: int,
    arc_end: int,
    delta: int
) -> list[Point] {
    return rounded_unique(
        arc_points(
            float(centre.x),
            float(centre.y),
            float(axes.width),
            float(axes.height),
            angle,
            arc_start,
            arc_end,
            delta
        )
    )
}

fn arc_step(reach: int) -> int {
    if reach < 3 { return 90 }
    if reach < 10 { return 30 }
    if reach < 15 { return 18 }
    return 5
}

fn draw_ellipse(
    painter: Painter,
    centre: Point,
    axes: Size,
    angle: int,
    arc_start: int,
    arc_end: int,
    thickness: int,
    line_type: int
) -> void {
    let width = abs(axes.width)
    let height = abs(axes.height)
    let delta = arc_step(max(width, height) + (XY_ONE >> 1) >> XY_SHIFT)
    let outline = rounded_unique(
        arc_points(
            float(centre.x),
            float(centre.y),
            float(width),
            float(height),
            angle,
            arc_start,
            arc_end,
            delta
        )
    )
    if outline.len() == 0 { return }
    if thickness >= 0 {
        poly_line(painter, outline, false, thickness, line_type, XY_SHIFT)
        return
    }
    if arc_end - arc_start >= 360 {
        fill_convex(painter, outline, line_type, XY_SHIFT)
        return
    }
    #: A partial arc closes through the centre, which makes it a wedge rather
    #: than a chord, and a wedge is not convex.
    var wedge = [point for point in outline]
    wedge.push(centre)
    var edges: list[PolyEdge] = []
    collect_edges(painter, wedge, edges, line_type, XY_SHIFT)
    fill_edges(painter, edges)
}

#: Draw a line segment.
pub fn line(
    img: Mat,
    pt1: Point,
    pt2: Point,
    colour: Scalar,
    thickness: int = 1,
    line_type: int = LINE_8,
    shift: int = 0
) -> Mat {
    if thickness <= 0 { throw ValueError("a line needs a thickness of at least one") }
    if thickness > MAX_THICKNESS { throw ValueError(f"a thickness of {thickness} is too large") }
    let painter = Painter(img, colour)
    draw_thick(
        painter,
        to_fixed(pt1, shift),
        to_fixed(pt2, shift),
        thickness,
        line_type,
        CAP_START | CAP_END
    )
    painter.finish()
    return img
}

#: Draw a rectangle between two opposite corners.
#:
#: A negative `thickness`, which `FILLED` spells, fills it instead.
pub fn rectangle(
    img: Mat,
    pt1: Point,
    pt2: Point,
    colour: Scalar,
    thickness: int = 1,
    line_type: int = LINE_8,
    shift: int = 0
) -> Mat {
    if thickness > MAX_THICKNESS { throw ValueError(f"a thickness of {thickness} is too large") }
    let corners = [pt1, Point(pt2.x, pt1.y), pt2, Point(pt1.x, pt2.y)]
    let painter = Painter(img, colour)
    if thickness >= 0 {
        poly_line(painter, corners, true, thickness, line_type, shift)
    } else {
        fill_convex(painter, corners, line_type, shift)
    }
    painter.finish()
    return img
}

#: Draw a rectangle given as a `Rect`.
pub fn rectangle_rect(
    img: Mat,
    rect: Rect,
    colour: Scalar,
    thickness: int = 1,
    line_type: int = LINE_8,
    shift: int = 0
) -> Mat {
    let far = Point(rect.x + rect.width - 1, rect.y + rect.height - 1)
    return rectangle(img, Point(rect.x, rect.y), far, colour, thickness, line_type, shift)
}

#: Draw a circle.
pub fn circle(
    img: Mat,
    centre: Point,
    radius: int,
    colour: Scalar,
    thickness: int = 1,
    line_type: int = LINE_8,
    shift: int = 0
) -> Mat {
    if radius < 0 { throw ValueError("a circle needs a radius of at least zero") }
    if thickness > MAX_THICKNESS { throw ValueError(f"a thickness of {thickness} is too large") }
    let painter = Painter(img, colour)
    if thickness > 1 or line_type == LINE_AA or shift > 0 {
        #: Anything but a hairline is the ellipse path, so a thick ring comes
        #: out the same width all the way round.
        let scaled = radius << XY_SHIFT - shift
        draw_ellipse(
            painter,
            to_fixed(centre, shift),
            Size(scaled, scaled),
            0,
            0,
            360,
            thickness,
            line_type
        )
    } else {
        draw_circle(painter, centre, radius, thickness < 0, line_type)
    }
    painter.finish()
    return img
}

#: Draw an ellipse or an elliptical arc.
pub fn ellipse(
    img: Mat,
    centre: Point,
    axes: Size,
    angle: float,
    start_angle: float,
    end_angle: float,
    colour: Scalar,
    thickness: int = 1,
    line_type: int = LINE_8,
    shift: int = 0
) -> Mat {
    let painter = Painter(img, colour)
    draw_ellipse(
        painter,
        to_fixed(centre, shift),
        Size(axes.width << XY_SHIFT - shift, axes.height << XY_SHIFT - shift),
        cv_round(angle),
        cv_round(start_angle),
        cv_round(end_angle),
        thickness,
        line_type
    )
    painter.finish()
    return img
}

#: Draw an ellipse described by a `RotatedRect`.
pub fn ellipse_rotated(
    img: Mat,
    box: RotatedRect,
    colour: Scalar,
    thickness: int = 1,
    line_type: int = LINE_8
) -> Mat {
    let axes = Size(cv_round(box.size.width * 0.5), cv_round(box.size.height * 0.5))
    let centre = Point(cv_round(box.center.x), cv_round(box.center.y))
    return ellipse(img, centre, axes, box.angle, 0.0, 360.0, colour, thickness, line_type, 0)
}

#: Draw one or more open or closed polylines.
pub fn polylines(
    img: Mat,
    polygons: list[list[Point]],
    closed: bool,
    colour: Scalar,
    thickness: int = 1,
    line_type: int = LINE_8,
    shift: int = 0
) -> Mat {
    if thickness > MAX_THICKNESS { throw ValueError(f"a thickness of {thickness} is too large") }
    let painter = Painter(img, colour)
    for polygon in polygons { poly_line(painter, polygon, closed, thickness, line_type, shift) }
    painter.finish()
    return img
}

#: Fill a convex polygon.
pub fn fill_convex_poly(
    img: Mat,
    points: list[Point],
    colour: Scalar,
    line_type: int = LINE_8,
    shift: int = 0
) -> Mat {
    let painter = Painter(img, colour)
    fill_convex(painter, points, line_type, shift)
    painter.finish()
    return img
}

#: Fill one or more polygons, which may be concave and may overlap.
pub fn fill_poly(
    img: Mat,
    polygons: list[list[Point]],
    colour: Scalar,
    line_type: int = LINE_8,
    shift: int = 0,
    offset: Point = Point(0, 0)
) -> Mat {
    let painter = Painter(img, colour)
    var edges: list[PolyEdge] = []
    for polygon in polygons {
        if polygon.len() == 0 { continue }
        var moved: list[Point] = []
        for point in polygon { moved.push(Point(point.x + offset.x, point.y + offset.y)) }
        collect_edges(painter, moved, edges, line_type, shift)
    }
    fill_edges(painter, edges)
    painter.finish()
    return img
}

#: Draw a line with an arrowhead at its far end.
pub fn arrowed_line(
    img: Mat,
    pt1: Point,
    pt2: Point,
    colour: Scalar,
    thickness: int = 1,
    line_type: int = LINE_8,
    shift: int = 0,
    tip_length: float = 0.1
) -> Mat {
    let dx = float(pt1.x - pt2.x)
    let dy = float(pt1.y - pt2.y)
    let tip = math.sqrt(dx * dx + dy * dy) * tip_length
    line(img, pt1, pt2, colour, thickness, line_type, shift)
    let angle = math.atan2(dy, dx)
    for side in 0..2 {
        let turn = angle + (if side == 0 { math.PI * 0.25 } else { -math.PI * 0.25 })
        let barb = Point(
            cv_round(float(pt2.x) + tip * math.cos(turn)),
            cv_round(float(pt2.y) + tip * math.sin(turn))
        )
        line(img, barb, pt2, colour, thickness, line_type, shift)
    }
    return img
}

#: Draw one of the predefined markers, centred on a point.
pub fn draw_marker(
    img: Mat,
    position: Point,
    colour: Scalar,
    marker_type: int = MARKER_CROSS,
    marker_size: int = 20,
    thickness: int = 1,
    line_type: int = LINE_8
) -> Mat {
    let half = marker_size // 2
    let x = position.x
    let y = position.y
    if marker_type == MARKER_CROSS or marker_type == MARKER_STAR {
        line(img, Point(x - half, y), Point(x + half, y), colour, thickness, line_type)
        line(img, Point(x, y - half), Point(x, y + half), colour, thickness, line_type)
    }
    if marker_type == MARKER_TILTED_CROSS {
        line(
            img,
            Point(x - half, y - half),
            Point(x + half, y + half),
            colour,
            thickness,
            line_type
        )
        line(
            img,
            Point(x + half, y - half),
            Point(x - half, y + half),
            colour,
            thickness,
            line_type
        )
        return img
    }
    if marker_type == MARKER_STAR {
        line(
            img,
            Point(x - half, y - half),
            Point(x + half, y + half),
            colour,
            thickness,
            line_type
        )
        line(
            img,
            Point(x + half, y - half),
            Point(x - half, y + half),
            colour,
            thickness,
            line_type
        )
        return img
    }
    if marker_type == MARKER_CROSS { return img }
    if marker_type == MARKER_DIAMOND {
        line(img, Point(x, y - half), Point(x + half, y), colour, thickness, line_type)
        line(img, Point(x + half, y), Point(x, y + half), colour, thickness, line_type)
        line(img, Point(x, y + half), Point(x - half, y), colour, thickness, line_type)
        line(img, Point(x - half, y), Point(x, y - half), colour, thickness, line_type)
        return img
    }
    if marker_type == MARKER_SQUARE {
        line(
            img,
            Point(x - half, y - half),
            Point(x + half, y - half),
            colour,
            thickness,
            line_type
        )
        line(
            img,
            Point(x + half, y - half),
            Point(x + half, y + half),
            colour,
            thickness,
            line_type
        )
        line(
            img,
            Point(x + half, y + half),
            Point(x - half, y + half),
            colour,
            thickness,
            line_type
        )
        line(
            img,
            Point(x - half, y + half),
            Point(x - half, y - half),
            colour,
            thickness,
            line_type
        )
        return img
    }
    if marker_type == MARKER_TRIANGLE_UP {
        line(
            img,
            Point(x - half, y + half),
            Point(x + half, y + half),
            colour,
            thickness,
            line_type
        )
        line(img, Point(x + half, y + half), Point(x, y - half), colour, thickness, line_type)
        line(img, Point(x, y - half), Point(x - half, y + half), colour, thickness, line_type)
        return img
    }
    if marker_type == MARKER_TRIANGLE_DOWN {
        line(
            img,
            Point(x - half, y - half),
            Point(x + half, y - half),
            colour,
            thickness,
            line_type
        )
        line(img, Point(x + half, y - half), Point(x, y + half), colour, thickness, line_type)
        line(img, Point(x, y + half), Point(x - half, y - half), colour, thickness, line_type)
        return img
    }
    throw ValueError(f"marker type {marker_type} is not one jaicv defines")
}
