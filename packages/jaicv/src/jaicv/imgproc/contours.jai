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

fn direction_of(dy: int, dx: int) -> int {
    for index in 0..8 {
        if STEP_Y[index] == dy and STEP_X[index] == dx { return index }
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

fn at(grid: list[int], rows: int, cols: int, y: int, x: int) -> int {
    if y < 0 or x < 0 or y >= rows or x >= cols { return 0 }
    return grid[y * cols + x]
}

fn follow(
    grid: list[int],
    rows: int,
    cols: int,
    start_y: int,
    start_x: int,
    from_y: int,
    from_x: int,
    nbd: int
) -> list[Point] {
    #: Step 3.1: look anticlockwise from the pixel the border was entered from
    #: for the first neighbour that is set.
    let entry = direction_of(from_y - start_y, from_x - start_x)
    var first_y = 0
    var first_x = 0
    var have_first = false
    for offset in 1..9 {
        let step = (entry + offset) % 8
        let y = start_y + STEP_Y[step]
        let x = start_x + STEP_X[step]
        if at(grid, rows, cols, y, x) != 0 {
            first_y = y
            first_x = x
            have_first = true
            break
        }
    }
    if not have_first {
        grid[start_y * cols + start_x] = -nbd
        return [Point(start_x, start_y)]
    }

    var previous_y = first_y
    var previous_x = first_x
    var current_y = start_y
    var current_x = start_x
    var points: list[Point] = []
    while true {
        #: Step 3.3: from where we came, look clockwise round the current pixel
        #: for the next set neighbour, remembering whether the pixel to the
        #: east was one of the ones looked at.
        let back = direction_of(previous_y - current_y, previous_x - current_x)
        var next_y = 0
        var next_x = 0
        var saw_east = false
        for offset in 1..9 {
            let step = (back - offset) % 8
            let index = if step < 0 { step + 8 } else { step }
            if index == 0 { saw_east = true }
            let y = current_y + STEP_Y[index]
            let x = current_x + STEP_X[index]
            if at(grid, rows, cols, y, x) != 0 {
                next_y = y
                next_x = x
                break
            }
        }

        #: Step 3.4: a pixel whose eastern neighbour was examined and is clear
        #: closes the border on that side.
        let cell = current_y * cols + current_x
        if saw_east and at(grid, rows, cols, current_y, current_x + 1) == 0 {
            grid[cell] = -nbd
        } elif grid[cell] == 1 {
            grid[cell] = nbd
        }
        points.push(Point(current_x, current_y))

        if next_y == start_y and next_x == start_x and current_y == first_y and current_x == first_x {
            break
        }
        previous_y = current_y
        previous_x = current_x
        current_y = next_y
        current_x = next_x
    }
    return points
}

fn trace_all(values: list[float], rows: int, cols: int) -> list[Border] {
    var grid: list[int] = []
    for value in values { grid.push(if value != 0.0 { 1 } else { 0 }) }

    #: The image is treated as sitting inside a frame border numbered one,
    #: which is a hole border; every top-level shape is a child of it.
    var kind_outer = [false]
    var kind_parent = [0]
    var found: list[Border] = []
    var nbd = 1

    for y in 0..rows {
        var last_nbd = 1
        for x in 0..cols {
            let here = grid[y * cols + x]
            if here == 0 { continue }
            let west = if x == 0 { 0 } else { grid[y * cols + x - 1] }
            let east = if x == cols - 1 { 0 } else { grid[y * cols + x + 1] }
            let outer = here == 1 and west == 0
            let hole = not outer and here >= 1 and east == 0
            if not outer and not hole {
                if here != 1 { last_nbd = abs(here) }
                continue
            }
            nbd += 1
            var from_y = y
            var from_x = x - 1
            if not outer {
                from_x = x + 1
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
            let points = follow(grid, rows, cols, y, x, from_y, from_x, nbd)
            kind_outer.push(outer)
            kind_parent.push(parent)
            found.push(Border(nbd, points, outer, parent))
            let after = grid[y * cols + x]
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
