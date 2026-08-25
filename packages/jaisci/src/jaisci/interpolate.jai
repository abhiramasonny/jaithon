#: Interpolation under scipy's names — `scipy.interpolate`.
#:
#: Host f64 on `list[float]`, for the reason `core.jai` states: a spline is
#: built by solving a tridiagonal system, which is a sequential elimination,
#: and evaluated by a binary search per query. Neither has the width a device
#: wants, and the knot counts that reach an interpolator are small.
#:
#: The knots must be strictly increasing, and every routine here says so by
#: name rather than producing a quietly wrong curve — a duplicated knot divides
#: by zero in the slope and an out-of-order one makes the search meaningless.
#:
#: ```
#: import jaisci as sci
#:
#: let curve = sci.CubicSpline(xs, ys)
#: let mid = curve.at(1.5)
#: ```
import std.math
from std.core import Printable

#: Where `value` sits among `knots`: the index `i` with
#: `knots[i] <= value < knots[i + 1]`, clamped to a real interval at both ends.
#:
#: Clamped rather than refused because every routine here extrapolates with the
#: end interval's polynomial, which is what scipy does with its default
#: `extrapolate=True`. Binary search rather than a scan: an evaluation is one
#: search and a handful of multiplies, so a linear scan would dominate for any
#: interesting number of knots.
fn interval_of(knots: list[float], value: float) -> int {
    let last = knots.len() - 2
    if value <= knots[0] { return 0 }
    if value >= knots[knots.len() - 1] { return last }
    var low = 0
    var high = knots.len() - 1
    while high - low > 1 {
        let middle = (low + high) // 2
        if knots[middle] <= value {
            low = middle
        } else {
            high = middle
        }
    }
    if low > last { return last }
    return low
}

fn check_knots(x: list[float], y: list[float], least: int, what: str) -> void {
    if x.len() != y.len() {
        throw ValueError(f"{what} got {x.len()} x values and {y.len()} y values")
    }
    if x.len() < least {
        throw ValueError(f"{what} needs at least {least} points, got {x.len()}")
    }
    for index in 1..x.len() {
        if x[index] <= x[index - 1] {
            throw ValueError(f"{what} needs increasing x, but x[{index}] <= x[{index - 1}]")
        }
    }
}

#: Piecewise linear interpolation, as `numpy.interp` does it.
#:
#: Outside the knots it holds the end VALUE rather than extrapolating the end
#: slope, which is numpy's behaviour and differs from every spline below --
#: worth knowing, because the two are otherwise interchangeable.
pub fn interp(query: list[float], x: list[float], y: list[float]) -> list[float] {
    check_knots(x, y, 2, "interp")
    var out: list[float] = []
    for value in query {
        if value <= x[0] {
            out.push(y[0])
        } elif value >= x[x.len() - 1] {
            out.push(y[y.len() - 1])
        } else {
            let index = interval_of(x, value)
            let span = x[index + 1] - x[index]
            let t = (value - x[index]) / span
            out.push(y[index] + t * (y[index + 1] - y[index]))
        }
    }
    return out
}

#: Solve a tridiagonal system in place by the Thomas algorithm.
#:
#: `lower`, `diagonal` and `upper` are the three bands, `rhs` the right-hand
#: side; the answer replaces `rhs`. No pivoting, which is sound here because
#: every system these splines build is diagonally dominant by construction --
#: the diagonal is twice the sum of the neighbouring spans.
fn thomas(
    lower: list[float],
    diagonal: list[float],
    upper: list[float],
    rhs: list[float]
) -> list[float] {
    let n = diagonal.len()
    var c = [value for value in upper]
    var d = [value for value in rhs]
    var b = [value for value in diagonal]
    for index in 1..n {
        let factor = lower[index] / b[index - 1]
        b[index] -= factor * c[index - 1]
        d[index] -= factor * d[index - 1]
    }
    var out = [0.0] * n
    out[n - 1] = d[n - 1] / b[n - 1]
    var index = n - 2
    while index >= 0 {
        out[index] = (d[index] - c[index] * out[index + 1]) / b[index]
        index -= 1
    }
    return out
}

#: A cubic spline through the knots, as `scipy.interpolate.CubicSpline`.
#:
#: `bc_type` is `"not-a-knot"` (scipy's default) or `"natural"`. The two differ
#: only in the first and last equations of the system, and they differ a LOT in
#: the end intervals: on `y = x^3` sampled at 0..4, not-a-knot reproduces the
#: cubic exactly and natural does not, because forcing the second derivative to
#: zero at the ends is wrong for that curve. That is why scipy's default is
#: not-a-knot and why this keeps the same default.
pub class CubicSpline: Printable {
    #: The knots, strictly increasing.
    pub let x: list[float]
    #: The value at each knot.
    pub let y: list[float]
    #: The first derivative at each knot, which is what the system solves for
    #: and what every evaluation below is written in terms of.
    pub let slope: list[float]

    pub fn init(self, x: list[float], y: list[float], bc_type: str = "not-a-knot") {
        check_knots(x, y, 2, "CubicSpline")
        self.x = [value for value in x]
        self.y = [value for value in y]
        self.slope = _spline_slopes(x, y, bc_type)
    }

    #: The curve at `value`, extrapolating with the end interval's cubic.
    pub fn at(self, value: float) -> float {
        let index = interval_of(self.x, value)
        let span = self.x[index + 1] - self.x[index]
        let t = (value - self.x[index]) / span
        let y0 = self.y[index]
        let y1 = self.y[index + 1]
        let m0 = self.slope[index] * span
        let m1 = self.slope[index + 1] * span
        #: Hermite form: the two values and the two scaled slopes. Written this
        #: way rather than as four stored coefficients because the slopes are
        #: what the system produced, and re-deriving coefficients from them
        #: only loses digits.
        let a = 2.0 * (y0 - y1) + m0 + m1
        let b = 3.0 * (y1 - y0) - 2.0 * m0 - m1
        return ((a * t + b) * t + m0) * t + y0
    }

    #: The first derivative at `value`.
    pub fn slope_at(self, value: float) -> float {
        let index = interval_of(self.x, value)
        let span = self.x[index + 1] - self.x[index]
        let t = (value - self.x[index]) / span
        let y0 = self.y[index]
        let y1 = self.y[index + 1]
        let m0 = self.slope[index] * span
        let m1 = self.slope[index + 1] * span
        let a = 2.0 * (y0 - y1) + m0 + m1
        let b = 3.0 * (y1 - y0) - 2.0 * m0 - m1
        return ((3.0 * a * t + 2.0 * b) * t + m0) / span
    }

    #: The curve at every value of `query`.
    pub fn map(self, query: list[float]) -> list[float] {
        var out: list[float] = []
        for value in query { out.push(self.at(value)) }
        return out
    }

    pub fn to_str(self) -> str { return f"CubicSpline({self.x.len()} knots)" }

    fn __str__(self) -> str { return self.to_str() }
}

fn _spline_slopes(x: list[float], y: list[float], bc_type: str) -> list[float] {
    let n = x.len()
    let kind = bc_type.lower()
    if kind != "not-a-knot" and kind != "natural" {
        throw ValueError(f"CubicSpline does not implement bc_type '{bc_type}'")
    }
    if n == 2 {
        #: One interval: the only cubic through two points with any end
        #: condition is the straight line.
        let slope = (y[1] - y[0]) / (x[1] - x[0])
        return [slope, slope]
    }

    var span = [0.0] * (n - 1)
    var divided = [0.0] * (n - 1)
    for index in 0..n - 1 {
        span[index] = x[index + 1] - x[index]
        divided[index] = (y[index + 1] - y[index]) / span[index]
    }

    var lower = [0.0] * n
    var diagonal = [0.0] * n
    var upper = [0.0] * n
    var rhs = [0.0] * n
    for index in 1..n - 1 {
        lower[index] = span[index]
        diagonal[index] = 2.0 * (span[index - 1] + span[index])
        upper[index] = span[index - 1]
        rhs[index] = 3.0 * (span[index] * divided[index - 1] + span[index - 1] * divided[index])
    }

    if kind == "natural" {
        diagonal[0] = 2.0
        upper[0] = 1.0
        rhs[0] = 3.0 * divided[0]
        lower[n - 1] = 1.0
        diagonal[n - 1] = 2.0
        rhs[n - 1] = 3.0 * divided[n - 2]
    } else {
        #: Not-a-knot: the third derivative is continuous across the second and
        #: second-to-last knots, which is the same as saying the first two
        #: intervals share one cubic. Three points make that condition
        #: degenerate -- the whole curve is then a single parabola -- so that
        #: case gets its own pair of equations, which is what scipy does too.
        if n == 3 {
            diagonal[0] = 1.0
            upper[0] = 1.0
            rhs[0] = 2.0 * divided[0]
            lower[n - 1] = 1.0
            diagonal[n - 1] = 1.0
            rhs[n - 1] = 2.0 * divided[n - 2]
        } else {
            let d0 = span[0]
            let d1 = span[1]
            diagonal[0] = d1
            upper[0] = d0 + d1
            rhs[0] = ((d0 + 2.0 * (d0 + d1)) * d1 * divided[0] + d0 * d0 * divided[1]) / (d0 + d1)

            let e0 = span[n - 2]
            let e1 = span[n - 3]
            lower[n - 1] = e0 + e1
            diagonal[n - 1] = e1
            rhs[n - 1] = (e0 * e0 * divided[n - 3] + (2.0 * (e0 + e1) + e0) * e1 * divided[n - 2]) /
                (e0 + e1)
        }
    }
    return thomas(lower, diagonal, upper, rhs)
}

#: Piecewise cubic Hermite, monotone by construction — `PchipInterpolator`.
#:
#: The slopes are Fritsch and Carlson's harmonic mean rather than a solved
#: system, which is what makes it LOCAL: moving one knot changes the curve only
#: in the two intervals beside it, where a cubic spline redraws the whole
#: thing. It also cannot overshoot -- between two knots the curve stays between
#: their values -- which is the reason to reach for it over `CubicSpline` on
#: data that is monotone and should stay that way.
pub class PchipInterpolator: Printable {
    pub let x: list[float]
    pub let y: list[float]
    pub let slope: list[float]

    pub fn init(self, x: list[float], y: list[float]) {
        check_knots(x, y, 2, "PchipInterpolator")
        self.x = [value for value in x]
        self.y = [value for value in y]
        self.slope = _pchip_slopes(x, y)
    }

    pub fn at(self, value: float) -> float {
        let index = interval_of(self.x, value)
        let span = self.x[index + 1] - self.x[index]
        let t = (value - self.x[index]) / span
        let y0 = self.y[index]
        let y1 = self.y[index + 1]
        let m0 = self.slope[index] * span
        let m1 = self.slope[index + 1] * span
        let a = 2.0 * (y0 - y1) + m0 + m1
        let b = 3.0 * (y1 - y0) - 2.0 * m0 - m1
        return ((a * t + b) * t + m0) * t + y0
    }

    pub fn map(self, query: list[float]) -> list[float] {
        var out: list[float] = []
        for value in query { out.push(self.at(value)) }
        return out
    }

    pub fn to_str(self) -> str { return f"PchipInterpolator({self.x.len()} knots)" }

    fn __str__(self) -> str { return self.to_str() }
}

fn _pchip_slopes(x: list[float], y: list[float]) -> list[float] {
    let n = x.len()
    if n == 2 {
        let slope = (y[1] - y[0]) / (x[1] - x[0])
        return [slope, slope]
    }

    var span = [0.0] * (n - 1)
    var divided = [0.0] * (n - 1)
    for index in 0..n - 1 {
        span[index] = x[index + 1] - x[index]
        divided[index] = (y[index + 1] - y[index]) / span[index]
    }

    var slope = [0.0] * n
    for index in 1..n - 1 {
        let left = divided[index - 1]
        let right = divided[index]
        #: A sign change or a flat side pins the slope to zero, which is what
        #: stops the curve overshooting through an extremum.
        if left * right <= 0.0 {
            slope[index] = 0.0
            continue
        }
        let hl = span[index - 1]
        let hr = span[index]
        let w1 = 2.0 * hr + hl
        let w2 = hr + 2.0 * hl
        slope[index] = (w1 + w2) / (w1 / left + w2 / right)
    }
    slope[0] = _pchip_end(span[0], span[1], divided[0], divided[1])
    slope[n - 1] = _pchip_end(span[n - 2], span[n - 3], divided[n - 2], divided[n - 3])
    return slope
}

#: The one-sided end slope, three-point and then clamped.
#:
#: The clamp is the whole point: the raw three-point formula can overshoot at
#: an end, and Fritsch and Carlson's rule is to fall back to zero on a sign
#: change and to cap at three times the neighbouring secant otherwise.
fn _pchip_end(h0: float, h1: float, d0: float, d1: float) -> float {
    var slope = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
    if slope * d0 <= 0.0 { return 0.0 }
    if d0 * d1 <= 0.0 and math.fabs(slope) > math.fabs(3.0 * d0) { return 3.0 * d0 }
    return slope
}
