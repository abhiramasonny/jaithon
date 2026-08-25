#: Root finding and minimisation under scipy's names — `scipy.optimize`.
#:
#: Everything here works in host f64 on `list[float]`, which is the convention
#: `core.jai` states for parameter vectors: an optimiser's vector is short, its
#: arithmetic is sequential and conditioning-sensitive, and a round trip to the
#: device per step would cost far more than the step. The OBJECTIVE may be as
#: expensive as it likes and may itself be a device pass — what these routines
#: promise is a bounded number of calls to it, not that they are cheap
#: themselves.
#:
#: Argument order, defaults and result fields follow scipy so a program
#: translates line for line: `brentq(f, a, b)` gives the root as a float,
#: `brentq_full` gives it inside a `RootResult`, and `minimize(f, x0)` gives an
#: `OptimizeResult` with `x`, `fun` and `success`.
#:
#: ```
#: import jaisci as sci
#:
#: let root = sci.brentq(fn(x: float) -> float { return x * x - 2.0 }, 0.0, 2.0)
#: let fit = sci.minimize(rosenbrock, [-1.2, 1.0], method: "bfgs")
#: ```
import std.math

from .core import OptimizeResult, RootResult, approx_fprime, approx_jacobian
from .core import dense_lu, dense_lu_solve

#: scipy's own defaults, so a call that names none of them behaves as scipy's.
pub const XTOL = 2e-12
pub const RTOL = 8.881784197001252e-16
pub const MAXITER = 100

#: Half the width a bracket has to close to: an absolute floor plus a relative
#: term, so a root near zero and a root near 1e12 both stop somewhere sensible.
fn half_tol(x: float, xtol: float, rtol: float) -> float {
    return 0.5 * (xtol + rtol * math.fabs(x))
}

fn opposite_signs(fa: float, fb: float) -> bool {
    if fa == 0.0 or fb == 0.0 { return false }
    return (fa > 0.0) != (fb > 0.0)
}

# ------------------------------------------------------------------
# Scalar roots
# ------------------------------------------------------------------

#: A sign-change bracket, halved until it closes.
#:
#: The slowest of the three and the only one that cannot fail on a continuous
#: function, so it is what a pathological objective should be given. Each step
#: halves the bracket, so `maxiter` halvings reach float64's resolution on any
#: bracket narrower than `2^100` — the default is never the binding constraint.
pub fn bisect(
    f: fn(float) -> float,
    a: float,
    b: float,
    xtol: float = XTOL,
    rtol: float = RTOL,
    maxiter: int = MAXITER
) -> RootResult {
    var low = a
    var high = b
    var f_low = f(low)
    var f_high = f(high)
    var calls = 2
    if f_low == 0.0 { return RootResult(low, 0, calls) }
    if f_high == 0.0 { return RootResult(high, 0, calls) }
    if not opposite_signs(f_low, f_high) {
        return RootResult(low, 0, calls, false, "no sign change")
    }
    var step = 0
    while step < maxiter {
        let middle = low + (high - low) * 0.5
        if math.fabs(high - low) < 2.0 * half_tol(middle, xtol, rtol) {
            return RootResult(middle, step, calls)
        }
        let f_middle = f(middle)
        calls += 1
        step += 1
        if f_middle == 0.0 { return RootResult(middle, step, calls) }
        if opposite_signs(f_low, f_middle) {
            high = middle
            f_high = f_middle
        } else {
            low = middle
            f_low = f_middle
        }
    }
    return RootResult(low + (high - low) * 0.5, maxiter, calls, false, "max iterations")
}

#: Brent's method: inverse quadratic interpolation behind a bisection guard.
#:
#: The guard is the whole algorithm. Interpolation converges superlinearly and
#: can also step outside the bracket or crawl; Brent accepts an interpolated
#: step only when it is less than half the step before last AND lands inside
#: the bracket by a margin, and bisects otherwise. That buys secant speed on a
#: smooth function and bisection's guarantee on anything else, which is why
#: scipy makes it the default and why this is the one to reach for unless the
#: derivative is free.
#:
#: Transcribed from scipy's `brentq.c` rather than from a textbook, including
#: the "step at least `delta`" floor at the end: without it the iterate can
#: stop moving while the bracket is still wider than the tolerance, and the
#: loop then burns every remaining iteration on the same point.
pub fn brentq_full(
    f: fn(float) -> float,
    a: float,
    b: float,
    xtol: float = XTOL,
    rtol: float = RTOL,
    maxiter: int = MAXITER
) -> RootResult {
    var x_pre = a
    var x_cur = b
    var f_pre = f(x_pre)
    var f_cur = f(x_cur)
    var calls = 2
    if f_pre == 0.0 { return RootResult(x_pre, 0, calls) }
    if f_cur == 0.0 { return RootResult(x_cur, 0, calls) }
    if not opposite_signs(f_pre, f_cur) {
        return RootResult(x_pre, 0, calls, false, "no sign change")
    }

    #: The far end of the bracket, whose sign differs from the current best.
    var x_blk = 0.0
    var f_blk = 0.0
    #: The last two steps, which is what Brent's acceptance test compares.
    var s_pre = 0.0
    var s_cur = 0.0

    var step = 0
    while step < maxiter {
        if opposite_signs(f_pre, f_cur) {
            x_blk = x_pre
            f_blk = f_pre
            s_cur = x_cur - x_pre
            s_pre = s_cur
        }
        if math.fabs(f_blk) < math.fabs(f_cur) {
            x_pre = x_cur
            x_cur = x_blk
            x_blk = x_pre
            f_pre = f_cur
            f_cur = f_blk
            f_blk = f_pre
        }
        let delta = half_tol(x_cur, xtol, rtol)
        let s_bisect = 0.5 * (x_blk - x_cur)
        if f_cur == 0.0 or math.fabs(s_bisect) < delta {
            return RootResult(x_cur, step, calls)
        }

        if math.fabs(s_pre) > delta and math.fabs(f_cur) < math.fabs(f_pre) {
            var s_try = 0.0
            if x_pre == x_blk {
                #: Two distinct points only: the secant step.
                s_try = -f_cur * (x_cur - x_pre) / (f_cur - f_pre)
            } else {
                #: Three: inverse quadratic through them.
                let d_pre = (f_pre - f_cur) / (x_pre - x_cur)
                let d_blk = (f_blk - f_cur) / (x_blk - x_cur)
                s_try = -f_cur * (f_blk * d_blk - f_pre * d_pre) / (d_blk * d_pre * (f_blk - f_pre))
            }
            let bound = min(math.fabs(s_pre), 3.0 * math.fabs(s_bisect) - delta)
            if 2.0 * math.fabs(s_try) < bound {
                s_pre = s_cur
                s_cur = s_try
            } else {
                s_pre = s_bisect
                s_cur = s_bisect
            }
        } else {
            s_pre = s_bisect
            s_cur = s_bisect
        }

        x_pre = x_cur
        f_pre = f_cur
        if math.fabs(s_cur) > delta {
            x_cur += s_cur
        } else {
            x_cur += if s_bisect > 0.0 { delta } else { -delta }
        }
        f_cur = f(x_cur)
        calls += 1
        step += 1
    }
    return RootResult(x_cur, maxiter, calls, false, "max iterations")
}

#: The root alone, which is what `scipy.optimize.brentq` returns by default.
pub fn brentq(
    f: fn(float) -> float,
    a: float,
    b: float,
    xtol: float = XTOL,
    rtol: float = RTOL,
    maxiter: int = MAXITER
) -> float {
    let found = brentq_full(f, a, b, xtol: xtol, rtol: rtol, maxiter: maxiter)
    if not found.converged { throw RuntimeError(f"brentq did not converge: {found.flag}") }
    return found.root
}

#: Newton's method when `fprime` is given, the secant method when it is not.
#:
#: Neither is bracketed, so neither is guaranteed: a flat derivative or a bad
#: start diverges, and the result says so rather than throwing. Use `brentq`
#: whenever a sign-change bracket is available — this is for the case where it
#: is not, or where the derivative is analytic and cheap.
pub fn newton(
    f: fn(float) -> float,
    x0: float,
    fprime: (fn(float) -> float)? = null,
    tol: float = 1.48e-08,
    maxiter: int = 50
) -> RootResult {
    var x = x0
    var calls = 0
    if fprime is not null {
        let derivative = fprime
        var step = 0
        while step < maxiter {
            let value = f(x)
            calls += 1
            if value == 0.0 { return RootResult(x, step, calls) }
            let slope = derivative(x)
            if slope == 0.0 {
                return RootResult(x, step, calls, false, "derivative went flat")
            }
            let moved = x - value / slope
            step += 1
            if math.fabs(moved - x) < tol { return RootResult(moved, step, calls) }
            x = moved
        }
        return RootResult(x, maxiter, calls, false, "max iterations")
    }

    #: scipy's secant start: a small relative nudge off `x0`, signed away from
    #: zero so a start of exactly zero still moves.
    var x_pre = x
    var x_cur = if x >= 0.0 { x * 1.0001 + 0.0001 } else { x * 1.0001 - 0.0001 }
    var f_pre = f(x_pre)
    var f_cur = f(x_cur)
    calls = 2
    var step = 0
    while step < maxiter {
        if f_cur == 0.0 { return RootResult(x_cur, step, calls) }
        if f_cur == f_pre {
            return RootResult(0.5 * (x_pre + x_cur), step, calls, false, "tolerance reached")
        }
        let moved = x_cur - f_cur * (x_cur - x_pre) / (f_cur - f_pre)
        step += 1
        if math.fabs(moved - x_cur) < tol { return RootResult(moved, step, calls) }
        x_pre = x_cur
        f_pre = f_cur
        x_cur = moved
        f_cur = f(x_cur)
        calls += 1
    }
    return RootResult(x_cur, maxiter, calls, false, "max iterations")
}

# ------------------------------------------------------------------
# Scalar minimisation
# ------------------------------------------------------------------

#: The golden ratio's conjugate, which is the fraction Brent's minimiser falls
#: back to when a parabola will not do.
const GOLDEN = 0.3819660112501051

#: Brent's minimiser: parabolic interpolation behind a golden-section guard.
#:
#: The same shape as the root finder and for the same reason -- a parabola
#: through the three best points converges superlinearly near a smooth minimum
#: and is useless away from one, so it is accepted only when the step is inside
#: the bracket and less than half the step before last.
#:
#: `bounds` is the interval to search, which makes this
#: `scipy.optimize.minimize_scalar(..., bounds=(a, b), method="bounded")` and
#: NOT its two-element `bracket=`. The names are worth keeping apart: scipy's
#: `bracket` is a pair of downhill SEEDS that it then expands outward until it
#: has a triple straddling a minimum, and that expansion can run away -- scipy
#: itself raises "no valid bracket was found" on `sin(x) + x/2` seeded with
#: (-3, 0), which searching the interval answers in nine calls. An interval is
#: also the thing a caller usually has.
pub fn minimize_scalar(
    f: fn(float) -> float,
    bounds: tuple[float, float] = (0.0, 1.0),
    xtol: float = 1.48e-08,
    maxiter: int = 500
) -> OptimizeResult {
    let (start, stop) = bounds
    var low = min(start, stop)
    var high = max(start, stop)

    var x = low + GOLDEN * (high - low)
    var w = x
    var v = x
    var fx = f(x)
    var fw = fx
    var fv = fx
    var calls = 1
    #: The last two steps, as in the root finder.
    var step_pre = 0.0
    var step_cur = 0.0

    var iteration = 0
    while iteration < maxiter {
        let middle = 0.5 * (low + high)
        let tol1 = xtol * math.fabs(x) + 1e-11
        let tol2 = 2.0 * tol1
        if math.fabs(x - middle) <= tol2 - 0.5 * (high - low) {
            return OptimizeResult([x], fx, true, 0, "converged", iteration, calls)
        }

        var use_golden = true
        if math.fabs(step_pre) > tol1 {
            #: The parabola through (v, fv), (w, fw), (x, fx).
            let r = (x - w) * (fx - fv)
            let q0 = (x - v) * (fx - fw)
            var p = (x - v) * q0 - (x - w) * r
            var q = 2.0 * (q0 - r)
            if q > 0.0 { p = -p }
            q = math.fabs(q)
            let step_before = step_pre
            step_pre = step_cur
            if math.fabs(p) < math.fabs(0.5 * q * step_before) and
                p > q * (low - x) and
                p < q * (high - x) {
                step_cur = p / q
                let moved = x + step_cur
                if moved - low < tol2 or high - moved < tol2 {
                    step_cur = if x < middle { tol1 } else { -tol1 }
                }
                use_golden = false
            }
        }
        if use_golden {
            step_pre = if x >= middle { low - x } else { high - x }
            step_cur = GOLDEN * step_pre
        }

        let moved = if math.fabs(step_cur) >= tol1 {
            x + step_cur
        } else {
            x + (if step_cur > 0.0 {
                tol1
            } else {
                -tol1
            })
        }
        let f_moved = f(moved)
        calls += 1
        iteration += 1

        if f_moved <= fx {
            if moved >= x { low = x } else { high = x }
            v = w
            fv = fw
            w = x
            fw = fx
            x = moved
            fx = f_moved
        } else {
            if moved < x { low = moved } else { high = moved }
            if f_moved <= fw or w == x {
                v = w
                fv = fw
                w = moved
                fw = f_moved
            } elif f_moved <= fv or v == x or v == w {
                v = moved
                fv = f_moved
            }
        }
    }
    return OptimizeResult([x], fx, false, 1, "max iterations", maxiter, calls)
}

# ------------------------------------------------------------------
# Vector minimisation
# ------------------------------------------------------------------
fn axpy(a: list[float], scale: float, b: list[float]) -> list[float] {
    var out: list[float] = []
    for index in 0..a.len() { out.push(a[index] + scale * b[index]) }
    return out
}

fn inner(a: list[float], b: list[float]) -> float {
    var total = 0.0
    for index in 0..a.len() { total += a[index] * b[index] }
    return total
}

fn max_abs(a: list[float]) -> float {
    var largest = 0.0
    for value in a {
        let size = math.fabs(value)
        if size > largest { largest = size }
    }
    return largest
}

#: Nelder and Mead's simplex, with scipy's initial simplex and its four
#: coefficients.
#:
#: Derivative free and slow, and the right answer when the objective is noisy,
#: discontinuous or expensive to differentiate -- which is why scipy makes it
#: the default. The initial simplex is scipy's: each vertex nudges one
#: coordinate by 5%, or by 0.00025 when that coordinate is zero, because a
#: relative nudge cannot move a zero.
fn nelder_mead(
    f: fn(list[float]) -> float,
    x0: list[float],
    xatol: float,
    fatol: float,
    maxiter: int
) -> OptimizeResult {
    let n = x0.len()
    if n == 0 { throw ValueError("minimize needs at least one parameter") }
    var calls = 0
    let counted = fn(v: list[float]) -> float {
        calls += 1
        return f(v)
    }

    var points: list[list[float]] = [[value for value in x0]]
    for index in 0..n {
        var vertex = [value for value in x0]
        if vertex[index] != 0.0 {
            vertex[index] = vertex[index] * 1.05
        } else {
            vertex[index] = 0.00025
        }
        points.push(vertex)
    }
    var values: list[float] = []
    for point in points { values.push(counted(point)) }

    var iteration = 0
    while iteration < maxiter {
        #: An insertion sort of n + 1 vertices, which for the sizes this method
        #: is used at is cheaper than building an index permutation.
        for i in 1..points.len() {
            let point = points[i]
            let value = values[i]
            var j = i - 1
            while j >= 0 and values[j] > value {
                points[j + 1] = points[j]
                values[j + 1] = values[j]
                j -= 1
            }
            points[j + 1] = point
            values[j + 1] = value
        }

        var widest = 0.0
        var worst_gap = 0.0
        for i in 1..points.len() {
            let gap = math.fabs(values[i] - values[0])
            if gap > worst_gap { worst_gap = gap }
            for k in 0..n {
                let spread = math.fabs(points[i][k] - points[0][k])
                if spread > widest { widest = spread }
            }
        }
        if widest <= xatol and worst_gap <= fatol {
            return OptimizeResult(points[0], values[0], true, 0, "converged", iteration, calls)
        }

        #: The centroid of everything but the worst vertex.
        var centre: list[float] = []
        for k in 0..n {
            var total = 0.0
            for i in 0..n { total += points[i][k] }
            centre.push(total / float(n))
        }
        let worst = points[n]
        let f_worst = values[n]

        let reflected = axpy(centre, 1.0, axpy(centre, -1.0, worst))
        let f_reflected = counted(reflected)

        if f_reflected < values[0] {
            let expanded = axpy(centre, 2.0, axpy(centre, -1.0, worst))
            let f_expanded = counted(expanded)
            if f_expanded < f_reflected {
                points[n] = expanded
                values[n] = f_expanded
            } else {
                points[n] = reflected
                values[n] = f_reflected
            }
        } elif f_reflected < values[n - 1] {
            points[n] = reflected
            values[n] = f_reflected
        } else {
            var shrink = true
            if f_reflected < f_worst {
                let outside = axpy(centre, 0.5, axpy(centre, -1.0, worst))
                let f_outside = counted(outside)
                if f_outside <= f_reflected {
                    points[n] = outside
                    values[n] = f_outside
                    shrink = false
                }
            } else {
                let inside = axpy(centre, -0.5, axpy(centre, -1.0, worst))
                let f_inside = counted(inside)
                if f_inside < f_worst {
                    points[n] = inside
                    values[n] = f_inside
                    shrink = false
                }
            }
            if shrink {
                for i in 1..points.len() {
                    points[i] = axpy(points[0], 0.5, axpy(points[i], -1.0, points[0]))
                    values[i] = counted(points[i])
                }
            }
        }
        iteration += 1
    }

    var best = 0
    for i in 1..points.len() {
        if values[i] < values[best] { best = i }
    }
    return OptimizeResult(points[best], values[best], false, 1, "max iterations", maxiter, calls)
}

#: BFGS with a backtracking Armijo line search.
#:
#: scipy's BFGS uses a strong-Wolfe search, which costs more objective calls
#: per step and buys a curvature condition the update needs to stay positive
#: definite. This uses Armijo backtracking and SKIPS the rank-two update when
#: the curvature `s . y` is not positive -- which is the same guarantee by a
#: cheaper route: a skipped update leaves the previous inverse Hessian, which
#: is already positive definite, so the direction is always a descent one.
#:
#: `jac` is optional and defaults to a forward difference, which is `n + 1`
#: objective calls per gradient. A cheap analytic gradient is worth supplying:
#: on a ten-parameter problem it is the difference between eleven calls a step
#: and one.
fn bfgs(
    f: fn(list[float]) -> float,
    x0: list[float],
    jac: (fn(list[float]) -> list[float])?,
    gtol: float,
    maxiter: int
) -> OptimizeResult {
    let n = x0.len()
    if n == 0 { throw ValueError("minimize needs at least one parameter") }

    #: Every call goes through the counter, including the ones the difference
    #: quotient makes. `nfev` is the number a caller is choosing between when
    #: they decide whether to write a gradient by hand, so counting only the
    #: line search's calls would report the two as equal cost -- which is the
    #: opposite of true: a forward difference is `n + 1` calls a step.
    var calls = 0
    let counted = fn(v: list[float]) -> float {
        calls += 1
        return f(v)
    }
    let gradient = jac ?? fn(v: list[float]) -> list[float] { return approx_fprime(v, counted) }

    var x = [value for value in x0]
    var fx = counted(x)
    var g = gradient(x)

    #: The inverse Hessian, held whole because `n` here is small; a limited
    #: memory form would be the answer past a few hundred parameters.
    var h: list[list[float]] = []
    for i in 0..n {
        var row: list[float] = []
        for j in 0..n { row.push(if i == j { 1.0 } else { 0.0 }) }
        h.push(row)
    }

    var iteration = 0
    while iteration < maxiter {
        if max_abs(g) <= gtol {
            return OptimizeResult(x, fx, true, 0, "converged", iteration, calls, g)
        }

        var direction: list[float] = []
        for i in 0..n {
            var total = 0.0
            for j in 0..n { total -= h[i][j] * g[j] }
            direction.push(total)
        }
        let slope = inner(g, direction)
        if slope >= 0.0 {
            #: The direction stopped being a descent one, which a skipped
            #: update should prevent; reset to steepest descent rather than
            #: give up.
            direction = [-value for value in g]
        }

        var alpha = 1.0
        var moved = axpy(x, alpha, direction)
        var f_moved = counted(moved)
        var tries = 0
        while f_moved > fx + 1e-4 * alpha * inner(g, direction) and tries < 40 {
            alpha *= 0.5
            moved = axpy(x, alpha, direction)
            f_moved = counted(moved)
            tries += 1
        }
        if tries >= 40 {
            return OptimizeResult(x, fx, false, 2, "line search failed", iteration, calls, g)
        }

        let g_moved = gradient(moved)
        var s: list[float] = []
        var y: list[float] = []
        for i in 0..n {
            s.push(moved[i] - x[i])
            y.push(g_moved[i] - g[i])
        }
        let curvature = inner(s, y)

        x = moved
        fx = f_moved
        g = g_moved
        iteration += 1

        #: Skipped when the curvature is not positive, which is what keeps `h`
        #: positive definite without a Wolfe condition.
        if curvature > 1e-12 {
            let rho = 1.0 / curvature
            var hy: list[float] = []
            for i in 0..n {
                var total = 0.0
                for j in 0..n { total += h[i][j] * y[j] }
                hy.push(total)
            }
            let y_h_y = inner(y, hy)
            for i in 0..n {
                for j in 0..n {
                    h[i][j] = h[i][j] - rho * (hy[i] * s[j] + s[i] * hy[j]) +
                        rho * rho * y_h_y * s[i] * s[j] +
                        rho * s[i] * s[j]
                }
            }
        }
    }
    return OptimizeResult(x, fx, false, 1, "max iterations", maxiter, calls, g)
}

#: Minimise `f` over a parameter vector, as `scipy.optimize.minimize` does.
#:
#: `method` is `"nelder-mead"` (the default, derivative free) or `"bfgs"`.
#: scipy chooses between them by whether bounds and a gradient are present;
#: choosing explicitly is clearer and is what most callers do anyway.
pub fn minimize(
    f: fn(list[float]) -> float,
    x0: list[float],
    method: str = "nelder-mead",
    jac: (fn(list[float]) -> list[float])? = null,
    tol: float = 1e-08,
    maxiter: int = 1000
) -> OptimizeResult {
    let lowered = method.lower()
    if lowered == "nelder-mead" or lowered == "neldermead" {
        return nelder_mead(f, x0, tol, tol, maxiter)
    }
    if lowered == "bfgs" { return bfgs(f, x0, jac, 1e-05, maxiter) }
    throw ValueError(f"minimize does not implement method '{method}'")
}

# ------------------------------------------------------------------
# Least squares
# ------------------------------------------------------------------

#: Levenberg–Marquardt on a vector residual, as `scipy.optimize.least_squares`
#: with `method="lm"` does.
#:
#: The normal equations `(JᵀJ + λ diag(JᵀJ)) δ = -Jᵀr` rather than a QR of `J`:
#: squaring the condition number is the price, and for the problem sizes a
#: host-side optimiser sees it is paid back by not factorising an `m x n`
#: matrix every step. `lambda` is scaled by the diagonal rather than by the
#: identity, which is Marquardt's refinement and is what makes the method
#: scale-invariant in the parameters.
pub fn least_squares(
    residual: fn(list[float]) -> list[float],
    x0: list[float],
    jac: (fn(list[float]) -> list[float])? = null,
    ftol: float = 1e-08,
    xtol: float = 1e-08,
    maxiter: int = 100
) -> OptimizeResult {
    let n = x0.len()
    if n == 0 { throw ValueError("least_squares needs at least one parameter") }
    var calls = 0
    let counted = fn(v: list[float]) -> list[float] {
        calls += 1
        return residual(v)
    }
    let jacobian = jac ?? fn(v: list[float]) -> list[float] { return approx_jacobian(v, counted) }

    var x = [value for value in x0]
    var r = counted(x)
    let m = r.len()
    if m < n {
        throw ValueError(f"least_squares got {m} residuals for {n} parameters")
    }
    var cost = 0.5 * inner(r, r)
    var damping = 1e-03

    var iteration = 0
    while iteration < maxiter {
        let j = jacobian(x)

        #: `JᵀJ` and `Jᵀr`, both accumulated in one pass over the rows.
        var jtj: list[float] = []
        for _cell in 0..n * n { jtj.push(0.0) }
        var jtr: list[float] = []
        for _cell in 0..n { jtr.push(0.0) }
        for row in 0..m {
            let base = row * n
            for a in 0..n {
                let value = j[base + a]
                jtr[a] += value * r[row]
                for b in 0..n { jtj[a * n + b] += value * j[base + b] }
            }
        }

        var accepted = false
        var tries = 0
        while not accepted and tries < 30 {
            var damped = [value for value in jtj]
            for a in 0..n {
                let diagonal = jtj[a * n + a]
                damped[a * n + a] = diagonal + damping * (if diagonal > 0.0 {
                    diagonal
                } else {
                    1.0
                })
            }
            #: `dense_lu`'s third value is the SIGN of the permutation and
            #: is zero only when a column was exactly zero -- so zero is the
            #: singular case and 1 or -1 are both fine. Reading it the other
            #: way round declined every well-conditioned system, which is a
            #: silent "no step reduced the cost" rather than an error.
            let (lu, pivots, sign) = dense_lu(damped, n)
            if sign == 0 {
                damping *= 10.0
                tries += 1
                continue
            }
            let delta = dense_lu_solve(lu, pivots, [-value for value in jtr], n)
            let trial = axpy(x, 1.0, delta)
            let r_trial = counted(trial)
            let cost_trial = 0.5 * inner(r_trial, r_trial)
            if cost_trial < cost {
                let moved = max_abs(delta)
                let dropped = cost - cost_trial
                x = trial
                r = r_trial
                cost = cost_trial
                damping = max(damping * 0.1, 1e-12)
                accepted = true
                if moved <= xtol * (1.0 + max_abs(x)) or dropped <= ftol * (1.0 + cost) {
                    return OptimizeResult(x, cost, true, 0, "converged", iteration + 1, calls)
                }
            } else {
                damping *= 10.0
                tries += 1
            }
        }
        if not accepted {
            return OptimizeResult(x, cost, false, 2, "no step reduced the cost", iteration, calls)
        }
        iteration += 1
    }
    return OptimizeResult(x, cost, false, 1, "max iterations", maxiter, calls)
}

#: Fit `f(x, *params)` to `(xdata, ydata)`, as `scipy.optimize.curve_fit` does.
#:
#: The model takes the whole `xdata` at once rather than a scalar, which is
#: what lets it be a device pass: `least_squares` calls it `n + 1` times a
#: step for the Jacobian, and a per-point model would make that `m (n + 1)`
#: calls instead of `n + 1`.
#:
#: Returns the parameters alone. The covariance scipy also returns needs
#: `(JᵀJ)⁻¹` scaled by the residual variance, which is a second factorisation
#: of a matrix this routine has already thrown away — it belongs here when
#: something needs it, not before.
pub fn curve_fit(
    f: fn(list[float], list[float]) -> list[float],
    xdata: list[float],
    ydata: list[float],
    p0: list[float],
    ftol: float = 1e-08,
    xtol: float = 1e-08,
    maxiter: int = 100
) -> list[float] {
    if xdata.len() != ydata.len() {
        throw ValueError(f"curve_fit got {xdata.len()} x values and {ydata.len()} y values")
    }
    let fitted = least_squares(fn(params: list[float]) -> list[float] {
        let predicted = f(xdata, params)
        if predicted.len() != ydata.len() {
            throw ValueError(f"the model returned {predicted.len()} values, not {ydata.len()}")
        }
        var out: list[float] = []
        for index in 0..ydata.len() {
            out.push(predicted[index] - ydata[index])
        }
        return out
    }, p0, ftol: ftol, xtol: xtol, maxiter: maxiter)
    return fitted.x
}
