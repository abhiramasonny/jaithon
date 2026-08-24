#: `find_contours` checked against counts arrived at a completely different
#: way, and against itself run twice.
#:
#: Border following and flood filling have nothing in common but the answer.
#: Suzuki's scan hands back one outer border per 8-connected run of set pixels
#: and one hole border per 4-connected pocket of clear ones that the frame does
#: not reach, so a flood fill that knows nothing about borders can say how many
#: contours there have to be and which component each one belongs to. That is
#: what these tests ask, because it is the one question the tracer cannot get
#: wrong in the same direction as its own reference would.
#:
#: The last test is about the scratch grid the tracer keeps between calls. It
#: is filled unconditionally, so nothing a previous picture marked can survive
#: into the next one -- but that is an argument, and an argument about reused
#: memory is worth a test that runs the same picture on either side of a
#: different one and demands the same answer twice.
from std.test import assert_eq, assert_true
from jaicv.core.mat import Mat
from jaicv.core.types import Point
from jaicv.imgproc.contours import CHAIN_APPROX_NONE, CHAIN_APPROX_SIMPLE, RETR_EXTERNAL, RETR_LIST, find_contours

let CV_8UC1 = 0

#: A picture as a flat list of ones and zeros, so the reference below can read
#: it without going through a `Mat` at all.
fn picture(rows: int, cols: int, kind: int) -> list[int] {
    var cells: list[int] = []
    var state = 9781 + kind * 137
    for index in 0..rows * cols {
        let y = index // cols
        let x = index - y * cols
        state = (state * 1103515245 + 12345) % 2147483648
        var on = 0
        if kind == 0 {
            if (x // 3 + y // 4) % 3 != 0 { on = 1 }
        } elif kind == 1 {
            if state % 100 < 55 { on = 1 }
        } elif kind == 2 {
            let radius = (x - cols // 2) * (x - cols // 2) + (y - rows // 2) * (y - rows // 2)
            let outer = radius < cols * cols // 9
            let inner = radius < cols * cols // 36
            if outer and inner == false { on = 1 }
        } elif kind == 3 {
            if index % 17 != 0 { on = 1 }
        } elif kind == 4 {
            if x % 7 < 4 and y % 5 < 3 { on = 1 }
        } else {
            on = 1
        }
        cells.push(on)
    }
    return cells
}

fn as_mat(cells: list[int], rows: int, cols: int) -> Mat {
    var values: list[float] = []
    for cell in cells { values.push(if cell != 0 { 255.0 } else { 0.0 }) }
    return Mat.from_list(rows, cols, CV_8UC1, values)
}

#: One label per set pixel, runs that touch at a corner counted as one.
#:
#: A plain flood fill off an explicit stack. Nothing here knows what a border
#: is, which is the point of it.
fn label_set(cells: list[int], rows: int, cols: int) -> list[int] {
    var labels = [-1] * (rows * cols)
    var count = 0
    for seed in 0..rows * cols {
        if cells[seed] == 0 { continue }
        if labels[seed] >= 0 { continue }
        labels[seed] = count
        var stack = [seed]
        while stack.len() > 0 {
            let at = stack.pop()
            let y = at // cols
            let x = at - y * cols
            for dy in -1..2 {
                for dx in -1..2 {
                    let ny = y + dy
                    let nx = x + dx
                    if ny < 0 or ny >= rows { continue }
                    if nx < 0 or nx >= cols { continue }
                    let next = ny * cols + nx
                    if cells[next] == 0 { continue }
                    if labels[next] >= 0 { continue }
                    labels[next] = count
                    stack.push(next)
                }
            }
        }
        count += 1
    }
    return labels
}

#: How many pockets of clear pixels the frame cannot reach, edges only.
fn count_pockets(cells: list[int], rows: int, cols: int) -> int {
    var seen = [0] * (rows * cols)
    var pockets = 0
    for seed in 0..rows * cols {
        if cells[seed] != 0 { continue }
        if seen[seed] != 0 { continue }
        var enclosed = true
        var stack = [seed]
        seen[seed] = 1
        while stack.len() > 0 {
            let at = stack.pop()
            let y = at // cols
            let x = at - y * cols
            let edge = y == 0 or y == rows - 1
            if edge or x == 0 or x == cols - 1 { enclosed = false }
            for step in 0..4 {
                var ny = y
                var nx = x
                if step == 0 {
                    ny = y - 1
                } elif step == 1 {
                    ny = y + 1
                } elif step == 2 {
                    nx = x - 1
                } else {
                    nx = x + 1
                }
                if ny < 0 or ny >= rows { continue }
                if nx < 0 or nx >= cols { continue }
                let next = ny * cols + nx
                if cells[next] != 0 { continue }
                if seen[next] != 0 { continue }
                seen[next] = 1
                stack.push(next)
            }
        }
        if enclosed { pockets += 1 }
    }
    return pockets
}

#: The clear pixels the frame can reach, edges only, with outside the picture
#: counted as clear. A set component that touches one of these is a shape
#: nothing encloses, which is what `RETR_EXTERNAL` keeps.
fn reach_from_frame(cells: list[int], rows: int, cols: int) -> list[int] {
    var seen = [0] * (rows * cols)
    var stack: list[int] = []
    for y in 0..rows {
        for x in 0..cols {
            let edge = y == 0 or y == rows - 1
            if edge == false and x != 0 and x != cols - 1 { continue }
            let at = y * cols + x
            if cells[at] != 0 { continue }
            if seen[at] != 0 { continue }
            seen[at] = 1
            stack.push(at)
        }
    }
    while stack.len() > 0 {
        let at = stack.pop()
        let y = at // cols
        let x = at - y * cols
        for step in 0..4 {
            var ny = y
            var nx = x
            if step == 0 {
                ny = y - 1
            } elif step == 1 {
                ny = y + 1
            } elif step == 2 {
                nx = x - 1
            } else {
                nx = x + 1
            }
            if ny < 0 or ny >= rows { continue }
            if nx < 0 or nx >= cols { continue }
            let next = ny * cols + nx
            if cells[next] != 0 { continue }
            if seen[next] != 0 { continue }
            seen[next] = 1
            stack.push(next)
        }
    }
    return seen
}

#: Which components nothing encloses, as a flag each.
fn top_level_of(
    cells: list[int],
    labels: list[int],
    rows: int,
    cols: int,
    components: int
) -> list[int] {
    let outside = reach_from_frame(cells, rows, cols)
    var top = [0] * (components + 1)
    for y in 0..rows {
        for x in 0..cols {
            let at = y * cols + x
            if cells[at] == 0 { continue }
            for dy in -1..2 {
                for dx in -1..2 {
                    let ny = y + dy
                    let nx = x + dx
                    let beyond = ny < 0 or ny >= rows
                    if beyond or nx < 0 or nx >= cols {
                        top[labels[at]] = 1
                    } elif outside[ny * cols + nx] != 0 {
                        top[labels[at]] = 1
                    }
                }
            }
        }
    }
    return top
}

fn highest(values: list[int]) -> int {
    var top = -1
    for value in values {
        if value > top { top = value }
    }
    return top
}

let ROWS = 41
let COLS = 57

#: One outer contour a component, and every point of it inside that component.
fn test_external_finds_each_unenclosed_component_once() -> void {
    for kind in 0..6 {
        let cells = picture(ROWS, COLS, kind)
        let labels = label_set(cells, ROWS, COLS)
        let components = highest(labels) + 1
        let top = top_level_of(cells, labels, ROWS, COLS, components)
        var wanted = 0
        for index in 0..components { wanted += top[index] }
        let image = as_mat(cells, ROWS, COLS)
        let (contours, _tree) = find_contours(image, RETR_EXTERNAL, CHAIN_APPROX_NONE)
        assert_eq(contours.len(), wanted)

        var claimed = [0] * (components + 1)
        for contour in contours {
            assert_true(contour.len() > 0)
            let owner = labels[contour[0].y * COLS + contour[0].x]
            assert_true(owner >= 0)
            assert_eq(top[owner], 1)
            claimed[owner] += 1
            for point in contour {
                let at = point.y * COLS + point.x
                assert_eq(cells[at], 1)
                assert_eq(labels[at], owner)
            }
        }
        for index in 0..components { assert_eq(claimed[index], top[index]) }
    }
}

#: Every border, counted as components plus enclosed pockets.
fn test_list_finds_one_border_a_component_and_a_pocket() -> void {
    for kind in 0..6 {
        let cells = picture(ROWS, COLS, kind)
        let components = highest(label_set(cells, ROWS, COLS)) + 1
        let pockets = count_pockets(cells, ROWS, COLS)
        let image = as_mat(cells, ROWS, COLS)
        let (contours, _tree) = find_contours(image, RETR_LIST, CHAIN_APPROX_NONE)
        assert_eq(contours.len(), components + pockets)
    }
}

#: A border is a closed ring of pixels that each touch a clear one.
fn test_every_border_pixel_touches_the_background() -> void {
    for kind in 0..6 {
        let cells = picture(ROWS, COLS, kind)
        let image = as_mat(cells, ROWS, COLS)
        let (contours, _tree) = find_contours(image, RETR_LIST, CHAIN_APPROX_NONE)
        for contour in contours {
            let count = contour.len()
            var previous = contour[count - 1]
            for point in contour {
                var clear = false
                for dy in -1..2 {
                    for dx in -1..2 {
                        let ny = point.y + dy
                        let nx = point.x + dx
                        let outside = ny < 0 or ny >= ROWS
                        if outside or nx < 0 or nx >= COLS {
                            clear = true
                        } elif cells[ny * COLS + nx] == 0 {
                            clear = true
                        }
                    }
                }
                assert_true(clear)
                if count > 1 {
                    let gap_y = if point.y > previous.y {
                        point.y - previous.y
                    } else {
                        previous.y - point.y
                    }
                    let gap_x = if point.x > previous.x {
                        point.x - previous.x
                    } else {
                        previous.x - point.x
                    }
                    assert_true(gap_y <= 1)
                    assert_true(gap_x <= 1)
                    assert_true(gap_y + gap_x > 0)
                }
                previous = point
            }
        }
    }
}

fn shape_of(contours: list[list[Point]]) -> list[int] {
    var out: list[int] = []
    for contour in contours {
        out.push(contour.len())
        for point in contour {
            out.push(point.x)
            out.push(point.y)
        }
    }
    return out
}

#: The tracer keeps its padded grid between calls; nothing may cross.
fn test_a_picture_traces_the_same_after_another_one() -> void {
    let first = as_mat(picture(ROWS, COLS, 1), ROWS, COLS)
    let other = as_mat(picture(23, 91, 2), 23, 91)
    let same_size = as_mat(picture(ROWS, COLS, 4), ROWS, COLS)
    for mode in [RETR_EXTERNAL, RETR_LIST] {
        let (before, _a) = find_contours(first, mode, CHAIN_APPROX_SIMPLE)
        let (_b, _c) = find_contours(other, mode, CHAIN_APPROX_SIMPLE)
        let (_d, _e) = find_contours(same_size, mode, CHAIN_APPROX_SIMPLE)
        let (after, _f) = find_contours(first, mode, CHAIN_APPROX_SIMPLE)
        assert_eq(shape_of(after), shape_of(before))
    }
}
