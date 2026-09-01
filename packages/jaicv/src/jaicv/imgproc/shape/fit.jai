#: Matching two shapes, and fitting a line or an ellipse to points.
import std.math
from ..geometric import solve_dense
from ...core.types import Point, Point2f, RotatedRect, Size2f
from .moments import EPSILON, hu_moments, moments

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
