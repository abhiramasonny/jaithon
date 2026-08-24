#: Border following: `find_contours` and `draw_contours`.
#:
#: The tracer is Suzuki and Abe's, which is the one OpenCV uses, and it is
#: reproduced closely enough that the contours come out in the same order, with
#: the same points in the same rotation, and with the same hierarchy — for
#: every retrieval mode and both chain approximations.
#:
#: Two details of that algorithm are easy to get wrong and are worth naming,
#: because both are load-bearing here. The border is walked anticlockwise in
#: image coordinates, not clockwise. And a border pixel is marked as closing a
#: border only when the pixel to its right was actually one of the neighbours
#: examined on the way round, not merely because that pixel is zero — mark on
#: the weaker rule and every hole inside a shape goes missing.
#:
#: This runs on the host: border following is inherently sequential, and the
#: images it runs on are already on their way to the host anyway.
#:
#: ```
#: let (contours, hierarchy) = find_contours(mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
#: draw_contours(frame, contours, -1, Scalar(0.0, 255.0, 0.0), 2)
#: ```
import std.math
from std.core import Printable
from ..core.mat import Mat
from ..core.types import LINE_8, LINE_AA, Point, Scalar
from .drawing import fill_poly, polylines, Painter, MAX_THICKNESS, XY_ONE, XY_SHIFT

#: Only the outermost contours.
pub const RETR_EXTERNAL = 0
#: Every contour, with no hierarchy between them.
pub const RETR_LIST = 1
#: Every contour, arranged as shapes and the holes directly inside them.
pub const RETR_CCOMP = 2
#: Every contour, with the full nesting.
pub const RETR_TREE = 3

#: Keep every border pixel.
pub const CHAIN_APPROX_NONE = 1
#: Keep only the ends of each straight run.
pub const CHAIN_APPROX_SIMPLE = 2

#: Where one contour sits among the others.
#:
#: The four fields are OpenCV's four, in its order, and index into the list of
#: contours returned alongside. `-1` means there is nothing there.
pub class Hierarchy: Printable {
    #: The next contour at this level.
    pub let next: int
    #: The previous contour at this level.
    pub let previous: int
    #: The first contour one level down.
    pub let first_child: int
    #: The contour one level up.
    pub let parent: int

    pub fn init(self, next: int, previous: int, first_child: int, parent: int) {
        self.next = next
        self.previous = previous
        self.first_child = first_child
        self.parent = parent
    }

    #: The four fields in OpenCV's order.
    pub fn to_list(self) -> list[int] {
        return [self.next, self.previous, self.first_child, self.parent]
    }

    pub fn to_str(self) -> str {
        return f"Hierarchy({self.next}, {self.previous}, {self.first_child}, {self.parent})"
    }

    fn __str__(self) -> str { return self.to_str() }
}

#: The eight neighbours in clockwise order, starting due east, with `y` down.
let STEP_Y = [0, 1, 1, 1, 0, -1, -1, -1]
let STEP_X = [1, 1, 0, -1, -1, -1, 0, 1]

class Border {
    pub var nbd: int
    #: The border's pixels as cells of the padded grid, not as points. See
    #: `follow` for why they are kept in that numbering all the way out.
    pub var cells: list[int]
    pub var outer: bool
    pub var parent_nbd: int

    pub fn init(self, nbd: int, cells: list[int], outer: bool, parent_nbd: int) {
        self.nbd = nbd
        self.cells = cells
        self.outer = outer
        self.parent_nbd = parent_nbd
    }
}

#: Walk a border all the way round, from the pixel it was entered at.
#:
#: Everything here is in the padded grid's own numbering: a cell is one
#: index, a step is one addition, and the ring of zeros around the picture
#: means no step can leave it. The bounds test that took its place ran eight
#: times a boundary pixel and there are two hundred thousand of those in a
#: frame of edges.
#:
#: The walk hands back those cell indices rather than points, and it never
#: works out an `x` and a `y` at all. Three quarters of a million border
#: pixels in a 1080p frame of rings is three quarters of a million `Point`
#: objects that `CHAIN_APPROX_SIMPLE` then throws all but a fifteenth of
#: away; the caller builds points once, after the thinning.
#:
#: Two shapes here are for the tier rather than for the reader. `cells` is
#: declared before the anticlockwise search below even though nothing fills
#: it until afterwards, and "no first neighbour" is a negative `first_step`
#: rather than a flag beside it. Both keep a slot from holding two kinds of
#: value across the function — a list where a bool was, an int where a bool
#: was — which is the one thing that stops the tier taking it at all.
#: Declined, the walk runs interpreted, and it is most of the call.
fn follow(grid: list[int], offsets: list[int], start_cell: int, entry: int, nbd: int) -> list[int] {
    var cells: list[int] = []

    #: Step 3.1: look anticlockwise from the pixel the border was entered from
    #: for the first neighbour that is set.
    var first_cell = 0
    var first_step = -1
    for offset in 1..9 {
        let step = (entry + offset) % 8
        let cell = start_cell + offsets[step]
        if grid[cell] != 0 {
            first_cell = cell
            first_step = step
            break
        }
    }
    if first_step < 0 {
        grid[start_cell] = -nbd
        cells.push(start_cell)
        return cells
    }

    #: Which way the pixel we came from lies, kept as a direction rather than
    #: worked back out of two coordinates every time round.
    var back = first_step
    var current_cell = start_cell
    loop {
        #: Step 3.3: from where we came, look clockwise round the current
        #: pixel for the next set neighbour. `back - offset` is lifted over
        #: zero before the remainder rather than after, because a negative
        #: remainder needs a branch and this is the innermost loop there is.
        var next_cell = 0
        var next_step = 0
        var found_at = 0
        for offset in 1..9 {
            let index = (back - offset + 8) % 8
            let cell = current_cell + offsets[index]
            if grid[cell] != 0 {
                next_cell = cell
                next_step = index
                found_at = offset
                break
            }
        }

        #: Step 3.4: a pixel whose eastern neighbour was examined and is clear
        #: closes the border on that side. East is one cell along, always
        #: inside the padding. Which turn would have looked east is fixed by
        #: `back` alone, so whether it was reached is one comparison against
        #: the turn the search stopped on rather than a test every turn.
        let east_at = if back == 0 { 8 } else { back }
        if found_at >= east_at and grid[current_cell + 1] == 0 {
            grid[current_cell] = -nbd
        } elif grid[current_cell] == 1 {
            grid[current_cell] = nbd
        }
        cells.push(current_cell)

        if next_cell == start_cell and current_cell == first_cell { break }
        back = (next_step + 4) % 8
        current_cell = next_cell
    }
    return cells
}

fn trace_all(values: list[float], rows: int, cols: int) -> list[Border] {
    #: A ring of zeros around the picture, so that neither the scan nor the
    #: walk has to ask whether a neighbour exists. The row length grows by two
    #: and every cell moves by one row and one column.
    #:
    #: Filled in two passes rather than pushed a cell at a time: a two
    #: megapixel frame is two million `push` calls, and one allocation of the
    #: whole thing followed by a write only where the picture is set costs
    #: about half as much and touches nothing where it is clear.
    let stride = cols + 2
    var grid = [0] * (stride * (rows + 2))
    for y in 0..rows {
        let base = y * cols
        let target = (y + 1) * stride + 1
        for x in 0..cols {
            if values[base + x] != 0.0 { grid[target + x] = 1 }
        }
    }

    #: The eight steps as single additions in that numbering.
    var offsets: list[int] = []
    for step in 0..8 { offsets.push(STEP_Y[step] * stride + STEP_X[step]) }

    #: The image is treated as sitting inside a frame border numbered one,
    #: which is a hole border; every top-level shape is a child of it. Held as
    #: ones and zeros rather than as bools for the same reason the scan below
    #: keeps clear of them.
    var kind_outer = [0]
    var kind_parent = [0]
    var found: list[Border] = []
    var nbd = 1

    #: Which of the two kinds of border starts here, as the direction the walk
    #: is entered from: four for an outer border, nought for a hole, and minus
    #: one for a pixel that starts neither.
    #:
    #: This used to read as three lines of `outer`, `hole` and `not outer and
    #: not hole`, which says what it means far better than the branches do.
    #: It cost eight times the whole rest of the scan. `not` applied to a bool
    #: the loop is holding, inside an `and`, is a shape the tier will not take
    #: — the same decision written as a chain of comparisons ran a two
    #: megapixel frame in 3.4 ms and written with that one `not` in it took
    #: 28 ms. There is nothing else in the difference; measure before tidying
    #: this back into the readable form.
    for y in 0..rows {
        var last_nbd = 1
        let row = (y + 1) * stride + 1
        for x in 0..cols {
            let cell = row + x
            let here = grid[cell]
            if here == 0 { continue }
            var entry = -1
            if here == 1 and grid[cell - 1] == 0 {
                entry = 4
            } elif here >= 1 and grid[cell + 1] == 0 {
                entry = 0
                if here > 1 { last_nbd = here }
            } else {
                #: `abs` is a global, and reading one from inside a loop this
                #: hot is enough on its own to stop the tier compiling the
                #: whole scan; the branch costs nothing and keeps it.
                if here != 1 { last_nbd = if here < 0 { -here } else { here } }
                continue
            }
            nbd += 1
            let ancestor = kind_parent[last_nbd - 1]
            let parent = if kind_outer[last_nbd - 1] == 1 {
                if entry == 4 {
                    ancestor
                } else {
                    last_nbd
                }
            } else {
                if entry == 4 {
                    last_nbd
                } else {
                    ancestor
                }
            }
            let cells = follow(grid, offsets, cell, entry, nbd)
            kind_outer.push(if entry == 4 { 1 } else { 0 })
            kind_parent.push(parent)
            found.push(Border(nbd, cells, entry == 4, parent))
            let after = grid[cell]
            if after != 1 { last_nbd = if after < 0 { -after } else { after } }
        }
    }
    return found
}

#: Drop the pixels in the middle of a straight run.
#:
#: The test is on the padded cell indices, not on an `x` and a `y`: for two
#: pixels that touch, `dy * stride + dx` tells the eight directions apart on
#: its own, so two steps are the same step exactly when the two differences
#: are equal. Consecutive border pixels always touch, and the last touches the
#: first, which is what makes the wrap at either end sound.
fn compress(cells: list[int]) -> list[int] {
    let count = cells.len()
    if count <= 2 { return cells }
    var out: list[int] = []
    var previous = cells[count - 1]
    var current = cells[0]
    for index in 0..count {
        let following = cells[if index + 1 < count { index + 1 } else { 0 }]
        if current - previous != following - current { out.push(current) }
        previous = current
        current = following
    }
    return out
}

#: Which contour encloses each one, as an index into `borders`.
#:
#: Suzuki's scan already knows the answer — it assigns a border's parent by
#: its number as it starts the walk — so all this has to do is turn a border
#: number into a position. That used to be a search of the whole list once per
#: contour, which is quadratic and, at the eight thousand contours a frame of
#: rings has, was two thirds of the entire call.
fn parents_of(borders: list[Border], mode: int) -> list[int] {
    let count = borders.len()
    var top = 1
    for border in borders {
        if border.nbd > top { top = border.nbd }
    }
    var at = [-1] * (top + 1)
    for index in 0..count { at[borders[index].nbd] = index }
    var out: list[int] = []
    for border in borders {
        if mode == RETR_CCOMP and border.outer {
            out.push(-1)
        } elif border.parent_nbd >= 0 and border.parent_nbd <= top {
            out.push(at[border.parent_nbd])
        } else {
            out.push(-1)
        }
    }
    return out
}

fn emit_order(borders: list[Border], mode: int) -> list[int] {
    #: A parent comes out before its children, and siblings come out in the
    #: reverse of the order the raster scan found them. That is the shape of
    #: the tree OpenCV builds by prepending each new contour to its parent.
    let parents = parents_of(borders, mode)
    let count = parents.len()

    #: The children of each contour as a chain, built by walking backwards so
    #: that each chain comes out in discovery order. Sweeping the whole list
    #: to collect one node's children is the other half of the quadratic.
    var first_child = [-1] * (count + 1)
    var next_sibling = [-1] * count
    var index = count - 1
    while index >= 0 {
        let slot = parents[index] + 1
        next_sibling[index] = first_child[slot]
        first_child[slot] = index
        index -= 1
    }

    var order: list[int] = []
    var stack = [-1]
    while stack.len() > 0 {
        let parent = stack.pop()
        if parent >= 0 { order.push(parent) }
        #: Pushed in discovery order so they come off reversed, and each child
        #: is walked before the sibling that precedes it.
        var child = first_child[parent + 1]
        while child >= 0 {
            stack.push(child)
            child = next_sibling[child]
        }
    }
    return order
}

fn linked(count: int) -> list[Hierarchy] {
    var out: list[Hierarchy] = []
    for index in 0..count {
        out.push(Hierarchy(if index + 1 < count {
            index + 1
        } else {
            -1
        }, if index > 0 {
            index - 1
        } else {
            -1
        }, -1, -1))
    }
    return out
}

#: Turn a parent for each contour into OpenCV's four links.
#:
#: One sweep, ascending: the previous sibling is whichever contour with the
#: same parent came last, and naming it also names the sibling after it. The
#: search over every other contour this replaces ran twice per contour and was
#: the single most expensive thing in the whole call.
fn relate(parents: list[int]) -> list[Hierarchy] {
    let count = parents.len()
    var previous = [-1] * count
    var next = [-1] * count
    var eldest = [-1] * count
    var latest = [-1] * (count + 1)
    for index in 0..count {
        let parent = parents[index]
        let earlier = latest[parent + 1]
        previous[index] = earlier
        if earlier >= 0 { next[earlier] = index }
        latest[parent + 1] = index
        if parent >= 0 and eldest[parent] < 0 { eldest[parent] = index }
    }
    var out: list[Hierarchy] = []
    for index in 0..count {
        out.push(Hierarchy(next[index], previous[index], eldest[index], parents[index]))
    }
    return out
}

#: Find the contours in a binary image.
#:
#: `mode` is one of the `RETR_` constants and `method` one of the `CHAIN_`
#: ones. Every non-zero pixel counts as part of a shape. Returns the contours
#: and, alongside, where each one sits among the others.
pub fn find_contours(
    image: Mat,
    mode: int = RETR_LIST,
    method: int = CHAIN_APPROX_SIMPLE,
    offset: Point = Point(0, 0)
) -> tuple[list[list[Point]], list[Hierarchy]] {
    if image.channels() != 1 { throw ValueError("find_contours needs a single-channel Mat") }
    if method != CHAIN_APPROX_NONE and method != CHAIN_APPROX_SIMPLE {
        throw ValueError(f"chain approximation {method} is not one jaicv implements")
    }
    image.flush()
    let stride = image.cols + 2
    let found = trace_all(image.values(), image.rows, image.cols)

    var chosen: list[Border] = []
    if mode == RETR_EXTERNAL {
        #: A top-level shape is one whose enclosing border is the frame.
        for index in 0..found.len() {
            let border = found[found.len() - 1 - index]
            if border.outer and border.parent_nbd == 1 { chosen.push(border) }
        }
    } elif mode == RETR_LIST {
        for index in 0..found.len() { chosen.push(found[found.len() - 1 - index]) }
    } elif mode == RETR_CCOMP or mode == RETR_TREE {
        for index in emit_order(found, mode) { chosen.push(found[index]) }
    } else {
        throw ValueError(f"retrieval mode {mode} is not one jaicv implements")
    }

    #: The one place a cell becomes a point. A row is a division and a column
    #: is the remainder that division already worked out, so the offset the
    #: caller asked for rides along for nothing.
    let shift_y = offset.y - 1
    let shift_x = offset.x - 1
    var contours: list[list[Point]] = []
    for border in chosen {
        let cells = if method == CHAIN_APPROX_SIMPLE {
            compress(border.cells)
        } else {
            border.cells
        }
        var points: list[Point] = []
        for cell in cells {
            let row = cell // stride
            points.push(Point(cell - row * stride + shift_x, row + shift_y))
        }
        contours.push(points)
    }

    if mode == RETR_EXTERNAL or mode == RETR_LIST {
        return (contours, linked(contours.len()))
    }
    return (contours, relate(parents_of(chosen, mode)))
}

#: The disc a thick stroke caps a segment with, as one span a row.
#:
#: `draw_circle` walks one octant of the midpoint circle and mirrors it eight
#: ways, which names the same row up to four times over; all four of those
#: spans are centred on the vertex, so the widest covers the rest and keeping
#: one a row writes the same pixels in a quarter of the calls. The shape does
#: not depend on where the disc lands, so it is worked out once for a whole
#: picture rather than once a vertex.
#:
#: Returns the row offsets and, for each, how far its span reaches either side.
fn cap_rows(radius: int) -> tuple[list[int], list[int]] {
    var widest: list[int] = []
    for _index in 0..radius + radius + 1 { widest.push(-1) }
    var err = 0
    var dx = radius
    var dy = 0
    var plus = 1
    var minus = (radius << 1) - 1
    while dx >= dy {
        if dx > widest[radius - dy] { widest[radius - dy] = dx }
        if dx > widest[radius + dy] { widest[radius + dy] = dx }
        if dy > widest[radius - dx] { widest[radius - dx] = dy }
        if dy > widest[radius + dx] { widest[radius + dx] = dy }
        dy += 1
        err += plus
        plus += 2
        if err > 0 {
            err -= minus
            dx -= 1
            minus -= 2
        }
    }
    var rows: list[int] = []
    var reach: list[int] = []
    for index in 0..radius + radius + 1 {
        if widest[index] >= 0 {
            rows.push(index - radius)
            reach.push(widest[index])
        }
    }
    return (rows, reach)
}

#: Outline every contour with a stroke more than one pixel wide.
#:
#: The pixels are `polylines`', segment by segment: the quadrilateral a pen of
#: that width sweeps, and a disc at the far end of each so that joins come out
#: round. What is different is the shape of the loop, because at this count
#: that is the whole of the cost. Drawing 576 contours over a 1080p frame is
#: thirty thousand segments, and the general path builds eight `Point`s and
#: rounds through two function calls for every one of them.
#:
#: The work is in passes rather than in one loop because a call anywhere in a
#: loop body makes the JIT tier decline the whole body, and so does a body
#: with too much live in it. The arithmetic here measured 4 ns an iteration
#: compiled against 218 ns interpreted, so `math.sqrt` is kept to a pass of
#: its own and every other pass is small enough to be compiled: that frame
#: went from 72.1 ms to 35.5 ms.
#:
#: What is left is the primitive calls, and they are now nearly all of it. A
#: `fill_span` costs about 140 ns and a `fill_convex` about 430 ns whatever
#: they draw, almost none of which is the drawing; a segment needs one of the
#: second for its quadrilateral and one of the first a row of its cap, which
#: is 25 ms of the 35. OpenCV draws the same frame in 4.8 ms, and it is not
#: doing anything cleverer -- it is doing all thirty thousand segments inside
#: one call. Closing the rest of that gap wants a primitive that takes a
#: batch of thick segments the way `draw_lines` takes a batch of thin ones;
#: the same contours as hairlines, which do go over in one call a contour,
#: draw in 1.1 ms.
fn thick_contours(
    painter: Painter,
    contours: list[list[Point]],
    offset_x: int,
    offset_y: int,
    thickness: int
) -> void {
    let odd = thickness & 1
    let half_width = thickness << XY_SHIFT - 1
    let bias = float(half_width) + float(odd) * float(XY_ONE) * 0.5
    let (cap_dy, cap_reach) = cap_rows(half_width + (XY_ONE >> 1) >> XY_SHIFT)
    let cap_count = cap_dy.len()
    let rounding = XY_ONE >> 1

    #: Every segment's two ends, four numbers at a time, and its squared
    #: length. Flattening them is not tidiness: it is what leaves the passes
    #: below few enough live objects to be compiled.
    var ends: list[int] = []
    var lengths: list[int] = []
    for contour in contours {
        let count = contour.len()
        if count == 0 { continue }
        var previous = contour[count - 1]
        for index in 0..count {
            let current = contour[index]
            let run_x = previous.x - current.x
            let run_y = current.y - previous.y
            lengths.push(run_x * run_x + run_y * run_y)
            ends.push(previous.x + offset_x)
            ends.push(previous.y + offset_y)
            ends.push(current.x + offset_x)
            ends.push(current.y + offset_y)
            previous = current
        }
    }
    let segments = lengths.len()

    #: A segment of no length has no quadrilateral, only its cap; `0.0` is how
    #: the passes below are told so without asking a second time.
    var reaches: list[float] = []
    for squared in lengths {
        let real = squared > 0
        reaches.push(if real { bias / math.sqrt(float(squared)) } else { 0.0 })
    }

    #: How far the quadrilateral stands off the segment, rounded the way
    #: OpenCV rounds it. `cv_round` is written out here, twice over, because
    #: reaching it as a function is the one call that would cost this pass its
    #: compilation -- and interpreted it was the dearest pass of the five.
    var offsets: list[int] = []
    for index in 0..segments {
        let reach = reaches[index]
        let base = index * 4
        let across = float(ends[base + 3] - ends[base + 1]) * reach
        var floor_across = float(int(across))
        if floor_across > across { floor_across -= 1.0 }
        var offset_across = int(floor_across)
        let part_across = across - floor_across
        if part_across > 0.5 {
            offset_across += 1
        } elif part_across == 0.5 {
            if offset_across % 2 != 0 { offset_across += 1 }
        }
        let along = float(ends[base] - ends[base + 2]) * reach
        var floor_along = float(int(along))
        if floor_along > along { floor_along -= 1.0 }
        var offset_along = int(floor_along)
        let part_along = along - floor_along
        if part_along > 0.5 {
            offset_along += 1
        } elif part_along == 0.5 {
            if offset_along % 2 != 0 { offset_along += 1 }
        }
        offsets.push(offset_across)
        offsets.push(offset_along)
    }

    #: One corner list, refilled rather than rebuilt: the primitive reads it
    #: and keeps nothing of it. A segment of no length is handed over too --
    #: its four corners coincide, and the scanline walk leaves a quadrilateral
    #: with no rows in it alone -- because asking first costs more than the
    #: call it would save at this ratio of empty segments to real ones.
    var corners = [0, 0, 0, 0, 0, 0, 0, 0]
    for index in 0..segments {
        let base = index * 4
        let across = offsets[index * 2]
        let along = offsets[index * 2 + 1]
        let from_x = ends[base] << XY_SHIFT
        let from_y = ends[base + 1] << XY_SHIFT
        let to_x = ends[base + 2] << XY_SHIFT
        let to_y = ends[base + 3] << XY_SHIFT
        corners[0] = from_x + across
        corners[1] = from_y + along
        corners[2] = from_x - across
        corners[3] = from_y - along
        corners[4] = to_x - across
        corners[5] = to_y - along
        corners[6] = to_x + across
        corners[7] = to_y + along
        painter.fill(corners, XY_SHIFT, rounding, rounding)
    }

    #: The caps go over in a pass of their own for the same reason. Nothing
    #: here reads a pixel back and there is one colour, so which pass writes a
    #: shared pixel first cannot be seen.
    for index in 0..segments {
        let base = index * 4
        let to_x = ends[base + 2]
        let to_y = ends[base + 3]
        for row in 0..cap_count {
            let extent = cap_reach[row]
            painter.span(to_x - extent, to_x + extent, to_y + cap_dy[row])
        }
    }
}

#: Draw contours onto an image.
#:
#: `index` selects one contour, or `-1` draws all of them. A negative
#: `thickness`, which `FILLED` spells, fills them instead of outlining them.
pub fn draw_contours(
    image: Mat,
    contours: list[list[Point]],
    index: int,
    colour: Scalar,
    thickness: int = 1,
    line_type: int = LINE_8,
    offset: Point = Point(0, 0)
) -> Mat {
    var chosen: list[list[Point]] = []
    if index < 0 {
        chosen = contours
    } elif index < contours.len() {
        chosen = [contours[index]]
    } else {
        throw IndexError(f"contour {index} is outside a list of {contours.len()}")
    }
    if chosen.len() == 0 { return image }
    if thickness < 0 { return fill_poly(image, chosen, colour, line_type, 0, offset) }
    if thickness > MAX_THICKNESS { throw ValueError(f"a thickness of {thickness} is too large") }
    #: A hairline run already goes over as one batch a contour, and an
    #: anti-aliased one has to read pixels back; only the wide, hard-edged
    #: case is worth writing out, and it is the one a detector draws.
    if thickness > 1 and line_type != LINE_AA {
        let painter = Painter(image, colour)
        thick_contours(painter, chosen, offset.x, offset.y, thickness)
        painter.finish()
        return image
    }
    if offset.x == 0 and offset.y == 0 {
        return polylines(image, chosen, true, colour, thickness, line_type)
    }
    var moved: list[list[Point]] = []
    for contour in chosen {
        var shifted: list[Point] = []
        for point in contour { shifted.push(Point(point.x + offset.x, point.y + offset.y)) }
        moved.push(shifted)
    }
    return polylines(image, moved, true, colour, thickness, line_type)
}
