#: From three finder centres to a sampled module grid.
import std.math
from ...core.types import Point2f
from ...imgproc.geometric import get_perspective_transform, perspective_transform
from .finder import Finder
from .symbol import alignment_centres

pub fn distance(ay: float, ax: float, by: float, bx: float) -> float {
    let dy = ay - by
    let dx = ax - bx
    return math.sqrt(dy * dy + dx * dx)
}

#: Order three finders as (corner, along the top, down the side).
#:
#: The corner is the one opposite the longest side, which is the diagonal; of
#: the other two, the one that turns clockwise from the corner is the top.
pub fn order_finders(a: Finder, b: Finder, c: Finder) -> tuple[Finder, Finder, Finder] {
    let ab = distance(a.y, a.x, b.y, b.x)
    let bc = distance(b.y, b.x, c.y, c.x)
    let ca = distance(c.y, c.x, a.y, a.x)
    var corner = a
    var first = b
    var second = c
    if ab >= bc and ab >= ca {
        corner = c
        first = a
        second = b
    } elif bc >= ab and bc >= ca {
        corner = a
        first = b
        second = c
    } else {
        corner = b
        first = c
        second = a
    }
    let cross = (first.x - corner.x) * (second.y - corner.y) -
        (first.y - corner.y) * (second.x - corner.x)
    if cross < 0.0 { return (corner, second, first) }
    return (corner, first, second)
}

#: Sample a symbol's modules through the perspective its three finders imply.
#:
#: The three finder centres sit three and a half modules in from their corners,
#: which fixes three of the four points a perspective needs. The fourth is the
#: opposite corner, estimated from the other three and then refined against the
#: alignment pattern that sits near it -- without that refinement a symbol
#: photographed at an angle samples progressively off-centre towards the far
#: corner, which is where every module of a large version is.
pub fn sample_grid(
    values: list[float],
    rows: int,
    cols: int,
    corner: Finder,
    top: Finder,
    side_finder: Finder,
    dimension: int
) -> list[list[bool]]? {
    let edge = float(dimension) - 3.5
    var source = [Point2f(3.5, 3.5), Point2f(edge, 3.5), Point2f(3.5, edge), Point2f(edge, edge)]
    var target = [
        Point2f(corner.x, corner.y),
        Point2f(top.x, top.y),
        Point2f(side_finder.x, side_finder.y),
        Point2f(top.x + side_finder.x - corner.x, top.y + side_finder.y - corner.y),
    ]

    #: The alignment pattern nearest the far corner, if this version has one.
    #: It is looked for where the three-finder estimate says it should be --
    #: mapped through that estimate rather than measured from the corner --
    #: and only within a couple of modules of there. A wider search finds a
    #: data module that happens to look like one, and a wrong fourth point is
    #: worse than none: it warps every sample towards the far corner, where
    #: most of a large symbol's modules are.
    let centres = alignment_centres((dimension - 17) // 4)
    if centres.len() >= 2 {
        let centre_module = float(centres[centres.len() - 1]) + 0.5
        let rough = get_perspective_transform(source, target)
        let guessed = perspective_transform([Point2f(centre_module, centre_module)], rough)[0]
        let scale = distance(target[0].y, target[0].x, target[1].y, target[1].x) / (edge - 3.5)
        let refined = refine_alignment(values, rows, cols, guessed, scale)
        if refined is not null {
            source[3] = Point2f(centre_module, centre_module)
            target[3] = refined
        }
    }

    let matrix = get_perspective_transform(source, target)
    var probes: list[Point2f] = []
    for y in 0..dimension {
        for x in 0..dimension { probes.push(Point2f(float(x) + 0.5, float(y) + 0.5)) }
    }
    let mapped = perspective_transform(probes, matrix)

    #: A module is decided by a vote over the pixels around its centre, not by
    #: the one pixel there. One flipped pixel at a centre corrupts a whole
    #: module, and a module is the unit the parity budget is spent in: at two
    #: per cent salt-and-pepper noise a single-pixel sample lost every symbol
    #: below level M, and a vote over a third of a module reads them all.
    let unit = distance(target[0].y, target[0].x, target[1].y, target[1].x) / (edge - 3.5)
    let reach = if unit >= 6.0 { 2 } elif unit >= 3.0 { 1 } else { 0 }

    var grid: list[list[bool]] = []
    var at = 0
    for _y in 0..dimension {
        var row: list[bool] = []
        for _x in 0..dimension {
            let point = mapped[at]
            at += 1
            let py = int(point.y + 0.5)
            let px = int(point.x + 0.5)
            if py < 0 or py >= rows or px < 0 or px >= cols { return null }
            var dark = 0
            var seen = 0
            for dy in -reach..reach + 1 {
                let sy = py + dy
                if sy < 0 or sy >= rows { continue }
                for dx in -reach..reach + 1 {
                    let sx = px + dx
                    if sx < 0 or sx >= cols { continue }
                    seen += 1
                    if values[sy * cols + sx] < 128.0 { dark += 1 }
                }
            }
            row.push(dark * 2 > seen)
        }
        grid.push(row)
    }
    return grid
}

#: The centre of the alignment pattern nearest `guess`, or null.
fn refine_alignment(
    values: list[float],
    rows: int,
    cols: int,
    guess: Point2f,
    unit: float
) -> Point2f? {
    if unit < 1.0 { return null }
    let reach = int(unit * 2.0) + 2
    let cy = int(guess.y + 0.5)
    let cx = int(guess.x + 0.5)
    var best: Point2f? = null
    var best_distance = 0.0

    for dy in -reach..reach + 1 {
        let y = cy + dy
        if y < 1 or y >= rows - 1 { continue }
        for dx in -reach..reach + 1 {
            let x = cx + dx
            if x < 1 or x >= cols - 1 { continue }
            if values[y * cols + x] >= 128.0 { continue }
            #: A one-unit dark centre inside a one-unit light ring: the
            #: pixel is dark, its neighbours a unit away are light, and its
            #: neighbours two modules away are dark again.
            let step = int(unit + 0.5)
            if step < 1 { continue }
            if y - 2 * step < 0 or y + 2 * step >= rows { continue }
            if x - 2 * step < 0 or x + 2 * step >= cols { continue }
            if values[(y - step) * cols + x] < 128.0 { continue }
            if values[(y + step) * cols + x] < 128.0 { continue }
            if values[y * cols + x - step] < 128.0 { continue }
            if values[y * cols + x + step] < 128.0 { continue }
            if values[(y - 2 * step) * cols + x] >= 128.0 { continue }
            if values[(y + 2 * step) * cols + x] >= 128.0 { continue }
            if values[y * cols + x - 2 * step] >= 128.0 { continue }
            if values[y * cols + x + 2 * step] >= 128.0 { continue }
            let away = distance(float(y), float(x), guess.y, guess.x)
            if best is null or away < best_distance {
                best = Point2f(float(x), float(y))
                best_distance = away
            }
        }
    }
    return best
}

#: How many modules across a symbol is, from the distance between its finder
#: centres. Snapped to the nearest legal dimension, since only sizes one more
#: than a multiple of four exist.
pub fn dimension_between(corner: Finder, top: Finder, side_finder: Finder) -> int {
    let across = distance(corner.y, corner.x, top.y, top.x)
    let down = distance(corner.y, corner.x, side_finder.y, side_finder.x)
    let unit = (corner.width + top.width + side_finder.width) / 3.0
    if unit <= 0.0 { return 0 }
    let estimate = (across + down) / 2.0 / unit + 7.0
    var dimension = int(estimate + 0.5)
    let remainder = (dimension - 17) % 4
    if remainder == 1 { dimension -= 1 }
    if remainder == 2 { dimension += 2 }
    if remainder == 3 { dimension += 1 }
    return dimension
}
