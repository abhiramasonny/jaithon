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
from std.core import Printable
from ..core.mat import Mat
from ..core.types import LINE_8, Point, Scalar
from .drawing import fill_poly, polylines

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

#: The inverse of `STEP_Y`/`STEP_X`, indexed by `(dy + 1) * 3 + (dx + 1)`.
#: The centre cell is -1 because no step stands still.
let STEP_INDEX = [5, 6, 7, 4, -1, 0, 3, 2, 1]

#: The lookup with no check that the step is a real one. `follow` uses this
#: because every step it takes is to a pixel it has just found adjacent.
#: Keeping the check out of the traversal matters twice over: it runs once a
#: boundary pixel, and reading `ValueError` for the throw is a global the JIT
#: tier has no slot for -- so the check alone left the whole loop interpreted.
fn step_index(dy: int, dx: int) -> int {
    return STEP_INDEX[(dy + 1) * 3 + dx + 1]
}

#: The checked form, for a caller that has not established the step is one.
fn direction_of(dy: int, dx: int) -> int {
    if dy >= -1 and dy <= 1 and dx >= -1 and dx <= 1 {
        let index = step_index(dy, dx)
        if index >= 0 { return index }
    }
    throw ValueError(f"({dy}, {dx}) is not a step to a neighbouring pixel")
}

class Border {
    pub var nbd: int
    pub var points: list[Point]
    pub var outer: bool
    pub var parent_nbd: int

    pub fn init(self, nbd: int, points: list[Point], outer: bool, parent_nbd: int) {
        self.nbd = nbd
        self.points = points
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
fn follow(
    grid: list[int],
    offsets: list[int],
    start_cell: int,
    start_y: int,
    start_x: int,
    entry: int,
    nbd: int
) -> list[Point] {
    #: Step 3.1: look anticlockwise from the pixel the border was entered from
    #: for the first neighbour that is set.
    var first_cell = 0
    var first_step = 0
    var have_first = false
    for offset in 1..9 {
        let step = (entry + offset) % 8
        let cell = start_cell + offsets[step]
        if grid[cell] != 0 {
            first_cell = cell
            first_step = step
            have_first = true
            break
        }
    }
    if not have_first {
        grid[start_cell] = -nbd
        return [Point(start_x, start_y)]
    }

    #: Which way the pixel we came from lies, kept as a direction rather than
    #: worked back out of two coordinates every time round.
    var back = first_step
    var current_cell = start_cell
    var current_y = start_y
    var current_x = start_x
    var points: list[Point] = []
    loop {
        #: Step 3.3: from where we came, look clockwise round the current pixel
        #: for the next set neighbour, remembering whether the pixel to the
        #: east was one of the ones looked at.
        var next_cell = 0
        var next_step = 0
        var saw_east = false
        for offset in 1..9 {
            let step = (back - offset) % 8
            let index = if step < 0 { step + 8 } else { step }
            if index == 0 { saw_east = true }
            let cell = current_cell + offsets[index]
            if grid[cell] != 0 {
                next_cell = cell
                next_step = index
                break
            }
        }

        #: Step 3.4: a pixel whose eastern neighbour was examined and is clear
        #: closes the border on that side. East is one cell along, always
        #: inside the padding.
        if saw_east and grid[current_cell + 1] == 0 {
            grid[current_cell] = -nbd
        } elif grid[current_cell] == 1 {
            grid[current_cell] = nbd
        }
        points.push(Point(current_x, current_y))

        if next_cell == start_cell and current_cell == first_cell { break }
        back = (next_step + 4) % 8
        current_cell = next_cell
        current_y += STEP_Y[next_step]
        current_x += STEP_X[next_step]
    }
    return points
}

fn trace_all(values: list[float], rows: int, cols: int) -> list[Border] {
    #: A ring of zeros around the picture, so that neither the scan nor the
    #: walk has to ask whether a neighbour exists. The row length grows by two
    #: and every cell moves by one row and one column.
    let stride = cols + 2
    var grid: list[int] = []
    for _index in 0..stride { grid.push(0) }
    for y in 0..rows {
        let base = y * cols
        grid.push(0)
        for x in 0..cols { grid.push(if values[base + x] != 0.0 { 1 } else { 0 }) }
        grid.push(0)
    }
    for _index in 0..stride { grid.push(0) }

    #: The eight steps as single additions in that numbering.
    var offsets: list[int] = []
    for step in 0..8 { offsets.push(STEP_Y[step] * stride + STEP_X[step]) }

    #: The image is treated as sitting inside a frame border numbered one,
    #: which is a hole border; every top-level shape is a child of it.
    var kind_outer = [false]
    var kind_parent = [0]
    var found: list[Border] = []
    var nbd = 1

    for y in 0..rows {
        var last_nbd = 1
        let row = (y + 1) * stride + 1
        for x in 0..cols {
            let cell = row + x
            let here = grid[cell]
            if here == 0 { continue }
            let west = grid[cell - 1]
            let east = grid[cell + 1]
            let outer = here == 1 and west == 0
            let hole = not outer and here >= 1 and east == 0
            if not outer and not hole {
                if here != 1 { last_nbd = abs(here) }
                continue
            }
            nbd += 1
            var entry = 4
            if not outer {
                entry = 0
                if here > 1 { last_nbd = here }
            }
            let previous_outer = kind_outer[last_nbd - 1]
            let parent = if outer {
                if previous_outer {
                    kind_parent[last_nbd - 1]
                } else {
                    last_nbd
                }
            } else {
                if previous_outer {
                    last_nbd
                } else {
                    kind_parent[last_nbd - 1]
                }
            }
            let points = follow(grid, offsets, cell, y, x, entry, nbd)
            kind_outer.push(outer)
            kind_parent.push(parent)
            found.push(Border(nbd, points, outer, parent))
            let after = grid[cell]
            if after != 1 { last_nbd = abs(after) }
        }
    }
    return found
}

fn compress(points: list[Point]) -> list[Point] {
    let count = points.len()
    if count <= 2 { return points }
    var out: list[Point] = []
    for index in 0..count {
        let previous = points[(index + count - 1) % count]
        let current = points[index]
        let following = points[(index + 1) % count]
        let in_x = current.x - previous.x
        let in_y = current.y - previous.y
        let out_x = following.x - current.x
        let out_y = following.y - current.y
        if in_x != out_x or in_y != out_y { out.push(current) }
    }
    return out
}

fn parent_index(borders: list[Border], border: Border, mode: int) -> int {
    if mode == RETR_CCOMP and border.outer { return -1 }
    for index in 0..borders.len() {
        if borders[index].nbd == border.parent_nbd { return index }
    }
    return -1
}

fn emit_order(borders: list[Border], mode: int) -> list[int] {
    #: A parent comes out before its children, and siblings come out in the
    #: reverse of the order the raster scan found them. That is the shape of
    #: the tree OpenCV builds by prepending each new contour to its parent.
    var parents: list[int] = []
    for border in borders { parents.push(parent_index(borders, border, mode)) }

    var order: list[int] = []
    var stack = [-1]
    while stack.len() > 0 {
        let parent = stack.pop()
        if parent >= 0 { order.push(parent) }
        var children: list[int] = []
        for index in 0..borders.len() {
            if parents[index] == parent { children.push(index) }
        }
        #: Pushed in discovery order so they come off reversed, and each child
        #: is walked before the sibling that precedes it.
        for child in children { stack.push(child) }
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

fn relate(parents: list[int]) -> list[Hierarchy] {
    let count = parents.len()
    var out: list[Hierarchy] = []
    for index in 0..count {
        let parent = parents[index]
        var previous = -1
        var next = -1
        var seen = false
        for other in 0..count {
            if parents[other] != parent { continue }
            if other == index {
                seen = true
                continue
            }
            if not seen { previous = other } elif next < 0 { next = other }
        }
        var first_child = -1
        for other in 0..count {
            if parents[other] == index {
                first_child = other
                break
            }
        }
        out.push(Hierarchy(next, previous, first_child, parent))
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

    var contours: list[list[Point]] = []
    for border in chosen {
        let points = if method == CHAIN_APPROX_SIMPLE {
            compress(border.points)
        } else {
            border.points
        }
        if offset.x == 0 and offset.y == 0 {
            contours.push(points)
        } else {
            var moved: list[Point] = []
            for point in points { moved.push(Point(point.x + offset.x, point.y + offset.y)) }
            contours.push(moved)
        }
    }

    if mode == RETR_EXTERNAL or mode == RETR_LIST {
        return (contours, linked(contours.len()))
    }
    var parents: list[int] = []
    for border in chosen { parents.push(parent_index(chosen, border, mode)) }
    return (contours, relate(parents))
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
    var moved: list[list[Point]] = []
    for contour in chosen {
        var shifted: list[Point] = []
        for point in contour { shifted.push(Point(point.x + offset.x, point.y + offset.y)) }
        moved.push(shifted)
    }
    return polylines(image, moved, true, colour, thickness, line_type)
}
