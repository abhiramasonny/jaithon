#: The convex hull of a set of points, by column buckets over a sort.
from std.algo.sort import natural_order, sort_by
from ...core.types import Point

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
