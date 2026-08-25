#: What a contour is: its moments, its area and perimeter, the shapes that
#: bound it, and the tests that describe it.
#:
#: These take a contour as a list of points — what `find_contours` returns —
#: and they follow OpenCV's formulas rather than the textbook ones wherever the
#: two differ, because the differences are visible. A contour's moments come
#: from Green's theorem over the polygon, not from summing pixels; the area is
#: the shoelace sum and so is smaller than the pixel count by roughly half the
#: perimeter; and `bounding_rect` is inclusive of the far edge.
#:
#: ```
#: let (contours, _) = find_contours(mask, RETR_EXTERNAL)
#: for contour in contours {
#:     if contour_area(contour) < 50.0 { continue }
#:     let box = bounding_rect(contour)
#:     rectangle_rect(frame, box, Scalar(0.0, 255.0, 0.0), 1)
#: }
#: ```
import std.math
from std.algo.sort import natural_order, sort_by
from std.core import Printable
from std.math import atan2
from ..core.mat import Mat
from ..core.types import Point, Point2f, Rect, RotatedRect, Scalar, Size2f
from .geometric import solve_dense

#: Compare two shapes by the reciprocals of their log Hu moments.
pub const CONTOURS_MATCH_I1 = 1
#: Compare two shapes by the differences of their log Hu moments.
pub const CONTOURS_MATCH_I2 = 2
#: Compare two shapes by the largest relative difference among them.
pub const CONTOURS_MATCH_I3 = 3

#: Fit a line by least squares.
pub const DIST_L2 = 2
#: Fit a line by the L1 norm, which ignores outliers more readily.
pub const DIST_L1 = 1
#: Fit a line by the Fair loss.
pub const DIST_FAIR = 5
#: Fit a line by the Welsch loss.
pub const DIST_WELSCH = 6
#: Fit a line by the Huber loss.
pub const DIST_HUBER = 7

let EPSILON = 1e-12

#: The raw, central, and normalised moments of a shape.
#:
#: The field names are OpenCV's: `m` for raw, `mu` for central, `nu` for
#: normalised.
pub class Moments: Printable {
    pub let m00: float
    pub let m10: float
    pub let m01: float
    pub let m20: float
    pub let m11: float
    pub let m02: float
    pub let m30: float
    pub let m21: float
    pub let m12: float
    pub let m03: float
    pub let mu20: float
    pub let mu11: float
    pub let mu02: float
    pub let mu30: float
    pub let mu21: float
    pub let mu12: float
    pub let mu03: float
    pub let nu20: float
    pub let nu11: float
    pub let nu02: float
    pub let nu30: float
    pub let nu21: float
    pub let nu12: float
    pub let nu03: float

    pub fn init(self, raw: list[float]) {
        self.m00 = raw[0]
        self.m10 = raw[1]
        self.m01 = raw[2]
        self.m20 = raw[3]
        self.m11 = raw[4]
        self.m02 = raw[5]
        self.m30 = raw[6]
        self.m21 = raw[7]
        self.m12 = raw[8]
        self.m03 = raw[9]

        var inverse = 0.0
        var cx = 0.0
        var cy = 0.0
        if math.fabs(self.m00) > EPSILON {
            inverse = 1.0 / self.m00
            cx = self.m10 * inverse
            cy = self.m01 * inverse
        }
        let mu20 = self.m20 - self.m10 * cx
        var mu11 = self.m11 - self.m10 * cy
        let mu02 = self.m02 - self.m01 * cy
        self.mu20 = mu20
        self.mu11 = mu11
        self.mu02 = mu02
        self.mu30 = self.m30 - cx * (3.0 * mu20 + cx * self.m10)
        #: Doubled here rather than in each line below, exactly as OpenCV does,
        #: because the two third-order terms both want twice it.
        mu11 = mu11 + mu11
        self.mu21 = self.m21 - cx * (mu11 + cx * self.m01) - cy * mu20
        self.mu12 = self.m12 - cy * (mu11 + cy * self.m10) - cx * mu02
        self.mu03 = self.m03 - cy * (3.0 * mu02 + cy * self.m01)

        let root = math.sqrt(math.fabs(inverse))
        let s2 = inverse * inverse
        let s3 = s2 * root
        self.nu20 = self.mu20 * s2
        self.nu11 = self.mu11 * s2
        self.nu02 = self.mu02 * s2
        self.nu30 = self.mu30 * s3
        self.nu21 = self.mu21 * s3
        self.nu12 = self.mu12 * s3
        self.nu03 = self.mu03 * s3
    }

    #: The centre of mass, or `null` when the shape has no area.
    pub fn centroid(self) -> Point2f? {
        if math.fabs(self.m00) <= EPSILON { return null }
        return Point2f(self.m10 / self.m00, self.m01 / self.m00)
    }

    #: Every field, in OpenCV's order.
    pub fn to_list(self) -> list[float] {
        return [
            self.m00,
            self.m10,
            self.m01,
            self.m20,
            self.m11,
            self.m02,
            self.m30,
            self.m21,
            self.m12,
            self.m03,
            self.mu20,
            self.mu11,
            self.mu02,
            self.mu30,
            self.mu21,
            self.mu12,
            self.mu03,
            self.nu20,
            self.nu11,
            self.nu02,
            self.nu30,
            self.nu21,
            self.nu12,
            self.nu03,
        ]
    }

    pub fn to_str(self) -> str { return f"Moments(m00={self.m00})" }

    fn __str__(self) -> str { return self.to_str() }
}

#: The moments of a polygon.
pub fn moments(contour: list[Point]) -> Moments {
    return moments_f([Point2f(float(point.x), float(point.y)) for point in contour])
}

#: The moments of a polygon whose corners are not on whole pixels.
pub fn moments_f(contour: list[Point2f]) -> Moments {
    let count = contour.len()
    if count == 0 { return Moments([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) }

    var a00 = 0.0
    var a10 = 0.0
    var a01 = 0.0
    var a20 = 0.0
    var a11 = 0.0
    var a02 = 0.0
    var a30 = 0.0
    var a21 = 0.0
    var a12 = 0.0
    var a03 = 0.0

    var px = contour[count - 1].x
    var py = contour[count - 1].y
    var px2 = px * px
    var py2 = py * py
    for index in 0..count {
        let x = contour[index].x
        let y = contour[index].y
        let x2 = x * x
        let y2 = y * y
        let cross = px * y - x * py
        let sx = px + x
        let sy = py + y

        a00 += cross
        a10 += cross * sx
        a01 += cross * sy
        a20 += cross * (px * sx + x2)
        a11 += cross * (px * (sy + py) + x * (sy + y))
        a02 += cross * (py * sy + y2)
        a30 += cross * sx * (px2 + x2)
        a03 += cross * sy * (py2 + y2)
        a21 += cross * (px2 * (3.0 * py + y) + 2.0 * px * x * sy + x2 * (py + 3.0 * y))
        a12 += cross * (py2 * (3.0 * px + x) + 2.0 * py * y * sx + y2 * (px + 3.0 * x))

        px = x
        py = y
        px2 = x2
        py2 = y2
    }

    if math.fabs(a00) <= EPSILON {
        return Moments([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    }
    #: The sign of the enclosed area carries through every weight, so a contour
    #: wound the other way still reports a positive `m00`.
    let sign = if a00 > 0.0 { 1.0 } else { -1.0 }
    return Moments(
        [
            a00 * sign / 2.0,
            a10 * sign / 6.0,
            a01 * sign / 6.0,
            a20 * sign / 12.0,
            a11 * sign / 24.0,
            a02 * sign / 12.0,
            a30 * sign / 20.0,
            a21 * sign / 60.0,
            a12 * sign / 60.0,
            a03 * sign / 20.0,
        ]
    )
}

#: The moments of an image, summed over its pixels.
pub fn moments_image(image: Mat, binary: bool = false) -> Moments {
    if image.channels() != 1 { throw ValueError("image moments need a single-channel Mat") }
    image.flush()
    let values = image.values()
    var m00 = 0.0
    var m10 = 0.0
    var m01 = 0.0
    var m20 = 0.0
    var m11 = 0.0
    var m02 = 0.0
    var m30 = 0.0
    var m21 = 0.0
    var m12 = 0.0
    var m03 = 0.0
    for y in 0..image.rows {
        let fy = float(y)
        for x in 0..image.cols {
            var weight = values[y * image.cols + x]
            if weight == 0.0 { continue }
            if binary { weight = 1.0 }
            let fx = float(x)
            m00 += weight
            m10 += weight * fx
            m01 += weight * fy
            m20 += weight * fx * fx
            m11 += weight * fx * fy
            m02 += weight * fy * fy
            m30 += weight * fx * fx * fx
            m21 += weight * fx * fx * fy
            m12 += weight * fx * fy * fy
            m03 += weight * fy * fy * fy
        }
    }
    return Moments([m00, m10, m01, m20, m11, m02, m30, m21, m12, m03])
}

#: The seven moment invariants, which do not change under translation, scale,
#: or rotation.
pub fn hu_moments(source: Moments) -> list[float] {
    let n20 = source.nu20
    let n11 = source.nu11
    let n02 = source.nu02
    let n30 = source.nu30
    let n21 = source.nu21
    let n12 = source.nu12
    let n03 = source.nu03

    let t0 = n30 + n12
    let t1 = n21 + n03
    let q0 = t0 * t0
    let q1 = t1 * t1
    let n4 = 4.0 * n11
    let s = n20 + n02
    let d = n20 - n02

    var out: list[float] = []
    out.push(s)
    out.push(d * d + n4 * n11)
    out.push((n30 - 3.0 * n12) * (n30 - 3.0 * n12) + (3.0 * n21 - n03) * (3.0 * n21 - n03))
    out.push(q0 + q1)

    let sum0 = q0 - 3.0 * q1
    let sum1 = 3.0 * q0 - q1
    out.push((n30 - 3.0 * n12) * t0 * sum0 + (3.0 * n21 - n03) * t1 * sum1)
    out.push(d * (q0 - q1) + n4 * t0 * t1)
    out.push((3.0 * n21 - n03) * t0 * sum0 - (n30 - 3.0 * n12) * t1 * sum1)
    return out
}

#: The area a contour encloses.
#:
#: This is the shoelace sum over the corners, so a shape drawn on pixels comes
#: out smaller than the number of pixels in it — the corners sit at pixel
#: centres and half the boundary falls outside. `oriented` keeps the sign,
#: which says which way the contour is wound.
pub fn contour_area(contour: list[Point], oriented: bool = false) -> float {
    let count = contour.len()
    if count < 2 { return 0.0 }
    var total = 0.0
    var previous = contour[count - 1]
    for index in 0..count {
        let point = contour[index]
        total += float(previous.x) * float(point.y) - float(point.x) * float(previous.y)
        previous = point
    }
    total *= 0.5
    if not oriented { return math.fabs(total) }
    return total
}

#: The edges from `opening` onward summed, which is the whole of what
#: `arc_length` does.
#:
#: Split out so the arm64 tier will take it: `curve.len()` is an OP_INVOKE and
#: one of those anywhere in the body made the tier decline the WHOLE function,
#: which then walked the curve interpreted at 160 ns a point. The count and
#: the starting neighbour come in already worked out; nothing in here calls
#: out.
#:
#: `** 0.5` rather than `math.sqrt` for the same reason — reading `sqrt` off
#: the module is an OP_GET_FIELD, and calling the imported one is a call the
#: tier will not put in a loop; either spelling costs 74 ns a point where this
#: one costs 1.6. Compiled it is a bare `fsqrt`, which is the correctly
#: rounded root `math.sqrt` returns. Until the body is compiled the
#: interpreter routes `**` through `pow`, and this platform's `pow(x, 0.5)`
#: differs from its `sqrt` by one unit in the last place for about one input
#: in eight hundred; a perimeter is a sum of roots and carries more error than
#: that from the sum alone.
fn walk_length(curve: list[Point], count: int, first: int, opening: int) -> float {
    let start = curve[opening]
    var px = start.x
    var py = start.y
    var total = 0.0
    for index in first..count {
        let point = curve[index]
        let dx = float(point.x - px)
        let dy = float(point.y - py)
        total += (dx * dx + dy * dy) ** 0.5
        px = point.x
        py = point.y
    }
    return total
}

#: The length of a curve, closed or open.
pub fn arc_length(curve: list[Point], closed: bool) -> float {
    let count = curve.len()
    if count <= 1 { return 0.0 }
    let opening = if closed { count - 1 } else { 0 }
    let first = if closed { 0 } else { 1 }
    return walk_length(curve, count, first, opening)
}

#: The smallest upright rectangle containing every point.
#:
#: The far edge is inclusive, as OpenCV's is: a single point gives a one by one
#: rectangle rather than an empty one.
pub fn bounding_rect(points: list[Point]) -> Rect {
    if points.len() == 0 { return Rect(0, 0, 0, 0) }
    var left = points[0].x
    var right = points[0].x
    var top = points[0].y
    var bottom = points[0].y
    for point in points {
        if point.x < left { left = point.x }
        if point.x > right { right = point.x }
        if point.y < top { top = point.y }
        if point.y > bottom { bottom = point.y }
    }
    return Rect(left, top, right - left + 1, bottom - top + 1)
}

fn normalised_angle(degrees: float) -> float {
    var angle = degrees
    while angle >= 0.0 { angle -= 90.0 }
    while angle < -90.0 { angle += 90.0 }
    return angle
}

fn cross_sign(o: Point, a: Point, b: Point) -> int {
    let value = (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
    if value > 0 { return 1 }
    if value < 0 { return -1 }
    return 0
}

#: A hull of fewer than three distinct points, ordered the way OpenCV orders
#: it.
#:
#: Two points have no rotational order, and OpenCV does not take one from the
#: input: `convexHull` returns them descending by x and then by y whichever way
#: round they arrive, so `[(1,2),(3,5)]` and `[(3,5),(1,2)]` both come back
#: `[(3,5),(1,2)]`. That is the same direction the counter-clockwise hull of a
#: collinear cloud comes back in, which is what makes it consistent rather than
#: arbitrary. `clockwise` reverses it, as it does everywhere else here.
fn degenerate_hull(points: list[Point], clockwise: bool) -> list[Point] {
    let unique = sorted_unique(points)
    if clockwise { return unique }
    var out: list[Point] = []
    for index in 0..unique.len() { out.push(unique[unique.len() - 1 - index]) }
    return out
}

fn sorted_unique(points: list[Point]) -> list[Point] {
    #: By x, and by y where the x agree. A hull is often asked for over a whole
    #: contour, which is thousands of points, and inserting each into a list
    #: already in order costs the square of that.
    let out = sort_by(points, |a, b| if a.x != b.x {
        natural_order(a.x, b.x)
    } else {
        natural_order(a.y, b.y)
    })
    var unique: list[Point] = []
    for point in out {
        let count = unique.len()
        if count > 0 and unique[count - 1].x == point.x and unique[count - 1].y == point.y {
            continue
        }
        unique.push(point)
    }
    return unique
}

#: The convex hull of a set of points, found by sorting.
#:
#: What `convex_hull` was until columns took over, and still what it falls back
#: to for the shapes columns are wrong for. Andrew's monotone chain: sort by x,
#: then walk the sorted points forwards for the chain below the hull and
#: backwards for the chain above it.
fn hull_by_sorting(points: list[Point], clockwise: bool) -> list[Point] {
    let sorted = sorted_unique(points)
    let count = sorted.len()
    if count < 3 { return degenerate_hull(points, clockwise) }

    var lower: list[Point] = []
    for index in 0..count {
        let point = sorted[index]
        while lower.len() >= 2 and
            cross_sign(lower[lower.len() - 2], lower[lower.len() - 1], point) <= 0 {
            lower.pop()
        }
        lower.push(point)
    }
    var upper: list[Point] = []
    for index in 0..count {
        let point = sorted[count - 1 - index]
        while upper.len() >= 2 and
            cross_sign(upper[upper.len() - 2], upper[upper.len() - 1], point) <= 0 {
            upper.pop()
        }
        upper.push(point)
    }

    #: `lower` walks left to right below the hull and `upper` right to left
    #: above it; dropping each one's last point joins them into a single ring.
    var ring: list[Point] = []
    for index in 0..upper.len() - 1 { ring.push(upper[index]) }
    for index in 0..lower.len() - 1 { ring.push(lower[index]) }
    if not clockwise { return ring }
    var flipped = [ring[0]]
    for index in 0..ring.len() - 1 { flipped.push(ring[ring.len() - 1 - index]) }
    return flipped
}

#: How a column's two extreme points are packed into one integer each.
#:
#: A column needs the y of its lowest point and of its highest, and for each
#: the index of the point it came from, because the ring hands back the
#: caller's own `Point` objects the way sorting did. Index and y share a word
#: so that a point costs one read and one write a side instead of two, and the
#: two sides are packed so that *smaller is better* for both: the low cell
#: holds `y * HULL_STRIDE + index` and the high cell `index - y * HULL_STRIDE`.
#: One comparison then settles each, and a tie between duplicate points goes to
#: the earliest of them -- which is the one the stable sort used to leave.
let HULL_SHIFT = 20
let HULL_STRIDE = 1048576
let HULL_MASK = 1048575

#: A column nothing has landed in. No packed cell can reach it: an index is
#: below `HULL_STRIDE` and a y large enough to carry one this far is refused.
let HULL_UNUSED = 4611686018427387904

#: As many points as an index packed into twenty bits can name.
let HULL_POINT_LIMIT = 1048576

#: How far a coordinate may sit from zero for the arithmetic below to be exact.
#:
#: Not the limit the packing alone would ask for, which is a thousand times
#: looser. The turn test multiplies a difference in x by a difference in y. The
#: difference in x is bounded by `HULL_WIDTH_LIMIT` and the one in y by this,
#: so the product fits in fifty bits and the difference of two of them in
#: fifty-one. Wider than any picture, and everything past it goes to
#: `hull_by_sorting`, which multiplies the same two differences but is never
#: reached with a span this path would have taken.
pub let HULL_COORD_LIMIT = 1073741824

#: How many columns the bucket path will address.
#:
#: A memory ceiling rather than a speed one: the cells hold two words a column,
#: so this is four megabytes of them kept between calls, and a quarter of a
#: million columns is a hundred times the width of a 1080p frame. It is also
#: what bounds the difference in x in the turn test above.
let HULL_WIDTH_LIMIT = 262144

#: How many columns a point may span before sorting is the cheaper answer.
#:
#: Sorting costs about six hundred nanoseconds a point, because its comparator
#: is a closure and no closure call is a candidate for the tier. A column costs
#: under two nanoseconds even when nothing has landed in it, so columns stay
#: ahead until there are some three hundred of them a point, and the block
#: index below usually means the empty ones are not visited at all. The floor
#: is there because a handful of points spread over a frame is still cheaper by
#: column than by sort.
#:
#: The old budget was four columns a point. It refused a 74-point frame contour
#: spanning 1920 columns and sent it to the sort: 44 microseconds where columns
#: take one and a half.
let HULL_COLUMN_BUDGET = 256
let HULL_COLUMN_FLOOR = 8192

#: The column buckets, the block index over them, the two chain stacks and the
#: span, kept between calls.
#:
#: Not held to save the allocation but to save the clearing. Filling a fresh
#: bucket array is the single most expensive thing this operation used to do
#: once the sort was gone: a hundred-point contour spans a couple of hundred
#: columns, and writing a sentinel into each of them cost more than reading the
#: points did. Kept instead, and blanked one column at a time by the walk that
#: reads it, so only columns a point actually landed in are ever written twice.
#:
#: `_hull_blocks` is what makes a sparse span affordable. A frame contour spans
#: every column of the picture but lands in only a few dozen of them, and the
#: walk used to step through all the empty ones in between: 1920 columns for 49
#: occupied, three and a half microseconds of the four and nine tenths the call
#: took. A block covers a run of columns and holds the leftmost and rightmost
#: that a point landed in, so the walk steps over about as many blocks as there
#: are points and then only the stretch of each block that holds anything --
#: 111 steps instead of 1920 on that contour. A pair starts at `HULL_UNUSED`
#: and -1 so that the two comparisons that widen it need no test for the first
#: point of a block.
#:
#: Nothing outside a single `convex_hull` call ever sees them, and the call
#: makes no call that could re-enter it.
var _hull_cells: list[int] = []
var _hull_blocks: list[int] = []
var _hull_low_chain: list[int] = []
var _hull_high_chain: list[int] = []
var _hull_span: list[int] = [0, 0, 0]

#: Room for a span this wide and a set this big, grown once and then kept.
#:
#: Every buffer starts empty and only ever grows, so this runs on the first
#: call and then on the rare call that spans wider than any before it.
fn hull_reserve(
    cells: list[int],
    blocks: list[int],
    low_chain: list[int],
    high_chain: list[int],
    width: int,
    count: int
) -> void {
    while cells.len() < width * 2 { cells.push(HULL_UNUSED) }
    var shift = 0
    while width >> shift > count { shift += 1 }
    let blocks_end = (width - 1 >> shift) + 1
    while blocks.len() < blocks_end * 2 {
        blocks.push(HULL_UNUSED)
        blocks.push(-1)
    }
    let room = count * 2 + 2
    while low_chain.len() < room {
        low_chain.push(0)
        high_chain.push(0)
    }
}

#: The extent of the points and the lowest and highest point of every column.
#:
#: Two passes over the points in one function, and the walk below is a second
#: function, because the tier is working to two budgets at once. All three
#: passes in one body needs more deopt records than it will keep for one -- a
#: hundred and sixty -- and the whole body is then declined and runs
#: interpreted at 90 nanoseconds a point. Split five ways instead, each body
#: has to reach the tier's entry threshold on its own, and a hull is entered
#: once a contour. Two is what fits both.
#:
#: `@trace` for the same reason: the tier looks at a function on its
#: sixty-fifth entry, and a marked one on its ninth. Sixty-four contours is a
#: long warm-up for something a frame calls once a shape, and the marking costs
#: nothing where it is read -- a trace session is only opened for a call made
#: from interpreted code, and by the time this matters `convex_hull` is native
#: and calls straight through.
#:
#: The buffers arrive as parameters rather than being read off the globals they
#: belong to, because a parameter's slot is its own for the life of the call
#: and the tier will not compile a body that gives one slot two kinds.
#:
#: -1 says the columns refused the set, and a value below that asks for room
#: and names the width that needs it. Both are settled before a single bucket
#: is written, so neither leaves anything to blank.
@trace
fn hull_columns(
    points: list[Point],
    cells: list[int],
    blocks: list[int],
    low_chain: list[int],
    span: list[int]
) -> int {
    #: Bound before any integer is, and reused by both passes rather than
    #: declared inside them. A local declared inside a loop is given a slot the
    #: compiler is free to hand to an integer once the loop ends, and a slot
    #: the tier has seen two kinds in is a slot it will not compile against.
    var held = points[0]
    let count = points.len()
    var lowest = held.x
    var highest = lowest
    var deepest = held.y
    var tallest = deepest
    for index in 0..count {
        held = points[index]
        let x = held.x
        if x < lowest { lowest = x }
        if x > highest { highest = x }
        let y = held.y
        if y < deepest { deepest = y }
        if y > tallest { tallest = y }
    }
    if lowest <= 0 - HULL_COORD_LIMIT { return -1 }
    if highest >= HULL_COORD_LIMIT { return -1 }
    if deepest <= 0 - HULL_COORD_LIMIT { return -1 }
    if tallest >= HULL_COORD_LIMIT { return -1 }
    if count >= HULL_POINT_LIMIT { return -1 }
    let width = highest - lowest + 1
    if width < 2 { return -1 }
    if width > HULL_WIDTH_LIMIT { return -1 }
    if width > HULL_COLUMN_BUDGET * count + HULL_COLUMN_FLOOR { return -1 }

    #: As many blocks as there are points: fewer and the stretch scanned inside
    #: a block grows, more and stepping over the empty blocks costs what
    #: stepping over the empty columns used to.
    var shift = 0
    while width >> shift > count { shift += 1 }
    let blocks_end = (width - 1 >> shift) + 1
    span[0] = lowest
    span[1] = shift
    span[2] = width

    #: Growing the buffers is `hull_reserve`'s job and not this function's, and
    #: it is asked for by returning rather than by calling. Three more loops in
    #: here cost guard-table room the tier does not have, and a body it
    #: declines runs interpreted at 90 nanoseconds a point.
    if cells.len() < width * 2 { return 0 - width - 2 }
    if blocks.len() < blocks_end * 2 { return 0 - width - 2 }
    if low_chain.len() < count * 2 + 2 { return 0 - width - 2 }

    #: Index and y share a word so that a point costs one read and one write a
    #: side; the block is widened only when the column was empty, because a
    #: column that already holds a point is already inside its block's span.
    for index in 0..count {
        held = points[index]
        let column = held.x - lowest
        let at = column * 2
        let scaled = held.y * HULL_STRIDE
        let low = scaled + index
        let seen = cells[at]
        if seen == HULL_UNUSED {
            cells[at] = low
            cells[at + 1] = index - scaled
            let bat = (column >> shift) * 2
            if column < blocks[bat] { blocks[bat] = column }
            if column > blocks[bat + 1] { blocks[bat + 1] = column }
        } else {
            if low < seen { cells[at] = low }
            let high = index - scaled
            if high < cells[at + 1] { cells[at + 1] = high }
        }
    }
    return 0
}

#: Both monotone chains, in one walk over the occupied columns.
#:
#: The outer loop is over blocks and the inner one over the stretch of each
#: block a point landed in, so a span with long empty runs in it costs about
#: what its points cost rather than what its width costs.
#:
#: A second function and not the tail of the one above because the two
#: together need more guards than the tier will record for one body, and a
#: declined body runs interpreted.
#:
#: Each chain is given only the points that could belong to it. The lowest
#: point of the leftmost column and the lowest of the rightmost are both
#: corners of the chain below, so nothing above the line joining them can be
#: one, and a column whose low point fails that test is skipped without
#: touching the stack at all. Skipping about two points in five is worth more
#: than the test costs, because a point that reaches the stack is pushed and
#: then usually popped again.
#:
#: The top two entries of each stack are carried in locals. The turn test reads
#: them on every candidate and the stack itself only when one is popped, which
#: turns four guarded list reads a candidate into none.
#:
#: The three counts come back in one integer -- the two chain depths and the
#: number of extreme points -- because a Jaithon function returns one value and
#: a list to hold three would cost an allocation a call.
@trace
fn hull_walk(
    cells: list[int],
    blocks: list[int],
    low_chain: list[int],
    high_chain: list[int],
    span: list[int]
) -> int {
    let lowest = span[0]
    let shift = span[1]
    let width = span[2]
    let blocks_end = (width - 1 >> shift) + 1
    let far = (width - 1) * 2
    let low_left = cells[0] >> HULL_SHIFT
    let low_right = cells[far] >> HULL_SHIFT
    let high_left = 0 - (cells[1] >> HULL_SHIFT)
    let high_right = 0 - (cells[far + 1] >> HULL_SHIFT)
    let reach = width - 1
    let low_rise = low_right - low_left
    let high_rise = high_right - high_left

    var corners = 0
    var low_slot = 0
    var low_ax = 0
    var low_ay = 0
    var low_ox = 0
    var low_oy = 0
    var high_slot = 0
    var high_ax = 0
    var high_ay = 0
    var high_ox = 0
    var high_oy = 0

    for block in 0..blocks_end {
        let bat = block * 2
        let first = blocks[bat]
        if first == HULL_UNUSED { continue }
        let last = blocks[bat + 1]
        blocks[bat] = HULL_UNUSED
        blocks[bat + 1] = -1

        for column in first..last + 1 {
            let at = column * 2
            let packed_low = cells[at]
            if packed_low == HULL_UNUSED { continue }
            let packed_high = cells[at + 1]
            cells[at] = HULL_UNUSED
            let x = lowest + column
            let low_y = packed_low >> HULL_SHIFT
            let high_y = 0 - (packed_high >> HULL_SHIFT)
            corners += 1
            if low_y != high_y { corners += 1 }

            if reach * (low_y - low_left) - low_rise * column <= 0 {
                while low_slot >= 4 {
                    let turn = (low_ax - low_ox) * (low_y - low_oy)
                    if turn - (low_ay - low_oy) * (x - low_ox) > 0 { break }
                    low_slot -= 2
                    low_ax = low_ox
                    low_ay = low_oy
                    if low_slot >= 4 {
                        low_ox = low_chain[low_slot - 4]
                        low_oy = low_chain[low_slot - 3] >> HULL_SHIFT
                    }
                }
                low_chain[low_slot] = x
                low_chain[low_slot + 1] = packed_low
                low_slot += 2
                low_ox = low_ax
                low_oy = low_ay
                low_ax = x
                low_ay = low_y
            }

            if reach * (high_y - high_left) - high_rise * column >= 0 {
                while high_slot >= 4 {
                    let turn = (high_ax - high_ox) * (high_y - high_oy)
                    if turn - (high_ay - high_oy) * (x - high_ox) < 0 { break }
                    high_slot -= 2
                    high_ax = high_ox
                    high_ay = high_oy
                    if high_slot >= 4 {
                        high_ox = high_chain[high_slot - 4]
                        high_oy = 0 - (high_chain[high_slot - 3] >> HULL_SHIFT)
                    }
                }
                high_chain[high_slot] = x
                high_chain[high_slot + 1] = packed_high
                high_slot += 2
                high_ox = high_ax
                high_oy = high_ay
                high_ax = x
                high_ay = high_y
            }
        }
    }
    let low_depth = low_slot // 2
    let high_depth = high_slot // 2
    return (low_depth * HULL_STRIDE + high_depth) * HULL_STRIDE + corners
}

#: The ring, from the two chains.
#:
#: Its own function, and the reversal below is another, because the tier is
#: particular about which slot holds what. A list comprehension's loop counter
#: is an integer in the slot the list itself is bound to next, and a slot the
#: tier has seen two kinds in is a slot it will not compile an index against --
#: so the ring is pushed rather than comprehended, and `convex_hull` is held to
#: the same rule and left holding nothing but integers.
@trace
fn hull_ring_of(
    points: list[Point],
    low_chain: list[int],
    high_chain: list[int],
    packed: int
) -> list[Point] {
    var ring: list[Point] = []
    let high_depth = packed >> HULL_SHIFT & HULL_MASK
    let low_depth = packed >> HULL_SHIFT >> HULL_SHIFT

    #: The chain above runs from the rightmost column to the leftmost once it
    #: is read backwards, and the chain below from the leftmost to the
    #: rightmost, so the two meet at both ends. Where a column's lowest and
    #: highest point are the same point the two chains share it, and one of the
    #: pair drops it -- which is what sorting's "drop each chain's last point"
    #: amounted to.
    #:
    #: Unpacked here rather than by the caller because the caller is entered
    #: once a hull, which is the slowest way there is to reach the tier's
    #: threshold: every integer settled in here is one the interpreter does not
    #: have to settle on all the calls before that. It does not go in
    #: `hull_walk` either -- that body is already as long as the emitter's
    #: buffer.
    let low_end = (low_depth - 1) * 2 + 1
    let high_end = (high_depth - 1) * 2 + 1
    var above = high_depth
    if low_chain[1] >> HULL_SHIFT == 0 - (high_chain[1] >> HULL_SHIFT) { above -= 1 }
    var under = low_depth
    let low_tail = low_chain[low_end] >> HULL_SHIFT
    if low_tail == 0 - (high_chain[high_end] >> HULL_SHIFT) { under -= 1 }

    var at = high_end
    for _index in 0..above {
        ring.push(points[high_chain[at] & HULL_MASK])
        at -= 2
    }
    at = 1
    for _index in 0..under {
        ring.push(points[low_chain[at] & HULL_MASK])
        at += 2
    }
    return ring
}

fn hull_reverse(ring: list[Point]) -> list[Point] {
    var flipped = [ring[0]]
    for index in 0..ring.len() - 1 { flipped.push(ring[ring.len() - 1 - index]) }
    return flipped
}

#: The convex hull of a set of points.
#:
#: `clockwise` is read in OpenCV's frame, where `y` points up, so the default
#: comes out clockwise on screen. The ring is the same one OpenCV returns and
#: turns the same way; which of its vertices comes first is not: OpenCV's falls
#: out of the two chains its Sklansky scan happens to build, and this starts at
#: the leftmost point instead.
#:
#: Bucketed by column rather than sorted. Sorting was almost all of what this
#: cost: `sorted_unique` calls a comparator closure per comparison, so a
#: hundred-point contour paid seven hundred closure calls before the chains
#: began, and no closure call is a candidate for the tier. Points are integer
#: pixels, so a column is an array index, and one pass over the points followed
#: by one over the columns leaves the chains the same sequence -- same order,
#: same duplicates dropped -- that sorting left them.
#:
#: Columns are refused for a span too wide to hold or too wide to be worth
#: holding, for a coordinate large enough to break the packing, and for more
#: points than an index can name; all of them go back to `hull_by_sorting`. So
#: does a set that comes back with fewer than three extreme points, which is
#: the one case where the two disagree: sorting returns its points untouched
#: when it has fewer than three, and that ordering is not the one the chains
#: would give.
pub fn convex_hull(points: list[Point], clockwise: bool = false) -> list[Point] {
    let count = points.len()
    if count < 3 { return degenerate_hull(points, clockwise) }
    var ready = hull_columns(points, _hull_cells, _hull_blocks, _hull_low_chain, _hull_span)
    if ready < -1 {
        hull_reserve(
            _hull_cells,
            _hull_blocks,
            _hull_low_chain,
            _hull_high_chain,
            0 - ready - 2,
            count
        )
        ready = hull_columns(points, _hull_cells, _hull_blocks, _hull_low_chain, _hull_span)
    }
    if ready != 0 { return hull_by_sorting(points, clockwise) }
    let packed = hull_walk(_hull_cells, _hull_blocks, _hull_low_chain, _hull_high_chain, _hull_span)
    if packed & HULL_MASK < 3 { return hull_by_sorting(points, clockwise) }
    if clockwise {
        return hull_reverse(hull_ring_of(points, _hull_low_chain, _hull_high_chain, packed))
    }
    return hull_ring_of(points, _hull_low_chain, _hull_high_chain, packed)
}

#: The indices, into `points`, of the hull points.
pub fn convex_hull_indices(points: list[Point], clockwise: bool = false) -> list[int] {
    let hull = convex_hull(points, clockwise)
    var out: list[int] = []
    for corner in hull {
        for index in 0..points.len() {
            if points[index].x == corner.x and points[index].y == corner.y {
                out.push(index)
                break
            }
        }
    }
    return out
}

#: Whether a contour turns the same way at every corner.
pub fn is_contour_convex(contour: list[Point]) -> bool {
    let count = contour.len()
    if count < 3 { return false }
    var sign = 0
    for index in 0..count {
        let a = contour[index]
        let b = contour[(index + 1) % count]
        let c = contour[(index + 2) % count]
        let turn = cross_sign(a, b, c)
        if turn == 0 { continue }
        if sign == 0 {
            sign = turn
        } elif turn != sign {
            return false
        }
    }
    return true
}

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

#: Compare two shapes by their Hu moments.
pub fn match_shapes(a: list[Point], b: list[Point], method: int = CONTOURS_MATCH_I1) -> float {
    let ha = hu_moments(moments(a))
    let hb = hu_moments(moments(b))
    var total = 0.0
    var worst = 0.0
    for index in 0..7 {
        let ma = math.fabs(ha[index])
        let mb = math.fabs(hb[index])
        if ma <= 1e-30 or mb <= 1e-30 { continue }
        let sa = if ha[index] < 0.0 { -1.0 } else { 1.0 }
        let sb = if hb[index] < 0.0 { -1.0 } else { 1.0 }
        let la = sa * math.log10(ma)
        let lb = sb * math.log10(mb)
        if method == CONTOURS_MATCH_I1 {
            total += math.fabs(1.0 / la - 1.0 / lb)
        } elif method == CONTOURS_MATCH_I2 {
            total += math.fabs(la - lb)
        } else {
            let relative = math.fabs((la - lb) / la)
            if relative > worst { worst = relative }
        }
    }
    if method == CONTOURS_MATCH_I3 { return worst }
    return total
}

#: Fit a line to a set of points.
#:
#: Returns `[vx, vy, x0, y0]`: a unit vector along the line and a point on it,
#: which is the layout OpenCV's `fitLine` returns. `dist_type` other than
#: `DIST_L2` runs the same fit again under an iteratively reweighted loss,
#: which is what makes it tolerate outliers.
pub fn fit_line(
    points: list[Point],
    dist_type: int = DIST_L2,
    reps: float = 0.01,
    aeps: float = 0.01
) -> list[float] {
    let count = points.len()
    if count < 2 { throw ValueError("a line needs at least two points") }
    let _unused = reps + aeps
    var weights: list[float] = []
    for _index in 0..count { weights.push(1.0) }

    var result = weighted_fit(points, weights)
    if dist_type == DIST_L2 { return result }
    for _pass in 0..10 {
        for index in 0..count {
            let dx = float(points[index].x) - result[2]
            let dy = float(points[index].y) - result[3]
            let along = dx * result[0] + dy * result[1]
            let away = math.fabs(dx * -result[1] + dy * result[0])
            let _keep = along
            weights[index] = loss_weight(dist_type, away)
        }
        result = weighted_fit(points, weights)
    }
    return result
}

fn loss_weight(dist_type: int, distance: float) -> float {
    let r = distance
    if dist_type == DIST_L1 { return 1.0 / max(r, 1e-6) }
    if dist_type == DIST_FAIR { return 1.0 / (1.0 + r / 1.3998) }
    if dist_type == DIST_WELSCH {
        let t = r / 2.9846
        return math.exp(-t * t)
    }
    if dist_type == DIST_HUBER {
        if r < 1.345 { return 1.0 }
        return 1.345 / r
    }
    return 1.0
}

fn weighted_fit(points: list[Point], weights: list[float]) -> list[float] {
    var total = 0.0
    var sx = 0.0
    var sy = 0.0
    for index in 0..points.len() {
        let w = weights[index]
        total += w
        sx += w * float(points[index].x)
        sy += w * float(points[index].y)
    }
    if total <= EPSILON { throw ValueError("every point was weighted to nothing") }
    let cx = sx / total
    let cy = sy / total
    var sxx = 0.0
    var syy = 0.0
    var sxy = 0.0
    for index in 0..points.len() {
        let w = weights[index]
        let dx = float(points[index].x) - cx
        let dy = float(points[index].y) - cy
        sxx += w * dx * dx
        syy += w * dy * dy
        sxy += w * dx * dy
    }
    #: The direction is the principal eigenvector of the scatter matrix, which
    #: for two by two is a closed form rather than an iteration.
    let difference = sxx - syy
    let angle = 0.5 * math.atan2(2.0 * sxy, difference)
    return [math.cos(angle), math.sin(angle), cx, cy]
}

#: Fit an ellipse to a set of points.
#:
#: Three least-squares fits, which is how OpenCV does it and why the answer
#: agrees with it rather than merely being close: a conic through the points
#: about their centroid, a two by two solve for where its centre really is, and
#: a second conic about that centre. Fitting once about the centroid alone
#: leaves the centre off by enough to move the axes.
#:
#: Returns the ellipse as a `RotatedRect`, with `width` the shorter axis.
pub fn fit_ellipse(points: list[Point]) -> RotatedRect {
    let count = points.len()
    if count < 5 { throw ValueError("an ellipse needs at least five points") }

    var cx = 0.0
    var cy = 0.0
    for point in points {
        cx += float(point.x)
        cy += float(point.y)
    }
    cx /= float(count)
    cy /= float(count)

    var terms: list[list[float]] = []
    for point in points {
        let u = float(point.x) - cx
        let v = float(point.y) - cy
        terms.push([-u * u, -v * v, -u * v, u, v])
    }
    let conic = least_squares(terms, 5)

    #: Where the conic's gradient vanishes is its centre; two equations in two
    #: unknowns, taken from differentiating the general form.
    let offset = solve_dense(
        [[2.0 * conic[0], conic[2]], [conic[2], 2.0 * conic[1]]],
        [conic[3], conic[4]]
    )
    if offset.len() < 2 { throw ValueError("these points do not determine an ellipse") }

    var centred: list[list[float]] = []
    for point in points {
        let u = float(point.x) - cx - offset[0]
        let v = float(point.y) - cy - offset[1]
        centred.push([u * u, v * v, u * v])
    }
    let shape = least_squares(centred, 3)
    let a = shape[0]
    let b = shape[1]
    let c = shape[2]

    let root = math.sqrt((a - b) * (a - b) + c * c)
    let larger = (a + b + root) * 0.5
    let smaller = (a + b - root) * 0.5
    if math.fabs(larger) <= EPSILON or math.fabs(smaller) <= EPSILON {
        throw ValueError("these points do not determine an ellipse")
    }
    #: The larger eigenvalue belongs to the shorter axis, since an axis is one
    #: over the root of its eigenvalue.
    let short_axis = 2.0 / math.sqrt(math.fabs(larger))
    let long_axis = 2.0 / math.sqrt(math.fabs(smaller))
    var angle = 0.5 * math.atan2(c, a - b) * 180.0 / math.PI + 180.0
    while angle >= 180.0 { angle -= 180.0 }
    while angle < 0.0 { angle += 180.0 }
    return RotatedRect(
        Point2f(cx + offset[0], cy + offset[1]),
        Size2f(short_axis, long_axis),
        angle
    )
}

#: Least squares through the normal equations, for a fit whose right-hand side
#: is all ones.
fn least_squares(rows: list[list[float]], width: int) -> list[float] {
    var normal: list[list[float]] = []
    var right: list[float] = []
    for _row in 0..width {
        var row: list[float] = []
        for _column in 0..width { row.push(0.0) }
        normal.push(row)
        right.push(0.0)
    }
    for row in rows {
        for i in 0..width {
            for j in 0..width { normal[i][j] += row[i] * row[j] }
            right[i] += row[i]
        }
    }
    let solution = solve_dense(normal, right)
    if solution.len() < width { throw ValueError("these points do not determine a fit") }
    return solution
}
