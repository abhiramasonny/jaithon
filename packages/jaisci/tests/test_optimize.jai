#: `jaisci.optimize` against scipy, which is where every expected value came
#: from. The reference calls are named beside each one so they can be re-run.
#:
#: Two places this deliberately differs from scipy, and both are checked here
#: rather than glossed:
#:
#: - `minimize_scalar` takes `bounds`, the interval to search, which is scipy's
#:   `method="bounded"`. scipy's two-element `bracket=` means something else --
#:   downhill seeds it expands outward -- and that expansion fails on
#:   `sin(x) + x/2` seeded (-3, 0), which the bounded form answers.
#: - `minimize(method: "bfgs")` uses an Armijo backtracking line search where
#:   scipy uses strong Wolfe, so the two take different step counts to the same
#:   minimum. What is checked is the minimum, to a tolerance that a different
#:   line search cannot move.
import std.math
from std.test import assert_eq, assert_near, assert_true, assert_false, assert_throws
from jaisci.optimize import bisect, brentq, brentq_full, newton, minimize_scalar
from jaisci.optimize import minimize, least_squares, curve_fit

fn quadratic(x: float) -> float { return x * x - 2.0 }

fn cubic(x: float) -> float { return x * x * x - 2.0 * x - 5.0 }

fn cosine_minus_x(x: float) -> float { return math.cos(x) - x }

#: scipy: optimize.brentq(lambda x: x*x - 2, 0, 2) -> 1.4142135623731364
#: The value is bit for bit scipy's, which is what a faithful transcription of
#: brentq.c should give and is the strongest check in this file.
fn test_brentq_matches_scipy_bit_for_bit() -> void {
    assert_eq(brentq(quadratic, 0.0, 2.0), 1.4142135623731364, "sqrt 2")
    assert_eq(brentq(cubic, 2.0, 3.0), 2.094551481542327, "the cubic's real root")
    assert_eq(brentq(cosine_minus_x, 0.0, 1.0), 0.7390851332151559, "the Dottie number")
}

fn test_brentq_reports_its_work() -> void {
    let found = brentq_full(quadratic, 0.0, 2.0)
    assert_true(found.converged, "brentq converged")
    assert_true(found.function_calls <= 12, "brentq took a handful of calls")
    assert_near(found.root, 1.4142135623730951, 1e-12, "the root")
}

#: An endpoint that is already the root is answered without iterating, which is
#: scipy's behaviour and the one case where a bracket test would reject a
#: perfectly good answer.
fn test_a_root_on_the_boundary_is_returned_as_is() -> void {
    let line = fn(x: float) -> float { return x - 1.0 }
    assert_eq(brentq(line, 1.0, 5.0), 1.0, "the low end")
    assert_eq(brentq(line, -3.0, 1.0), 1.0, "the high end")
}

fn test_no_sign_change_is_reported_not_thrown() -> void {
    let found = brentq_full(quadratic, 2.0, 3.0)
    assert_false(found.converged, "no bracket")
    assert_eq(found.flag, "no sign change", "the reason")
    assert_throws(RuntimeError, fn() -> void {
        let _ = brentq(quadratic, 2.0, 3.0)
    }, "brentq raises")
}

#: scipy: optimize.bisect(lambda x: x*x - 2, 0, 2) -> 1.4142135623715149.
#: Bisection stops on the bracket width, so the two differ in the last few
#: digits by exactly how the width test is written; both are inside the
#: tolerance either asked for.
fn test_bisect_converges_to_the_same_root() -> void {
    let found = bisect(quadratic, 0.0, 2.0)
    assert_true(found.converged, "bisect converged")
    assert_near(found.root, 1.4142135623730951, 1e-11, "sqrt 2")
}

#: scipy: optimize.newton(f, 1.0) -> 1.414213562373095 (secant, no derivative)
#: scipy: optimize.newton(f, 1.0, fprime=...) -> 1.4142135623730951
fn test_newton_with_and_without_a_derivative() -> void {
    let secant = newton(quadratic, 1.0)
    assert_true(secant.converged, "the secant converged")
    assert_near(secant.root, 1.414213562373095, 1e-12, "sqrt 2 by secant")

    let tangent = newton(quadratic, 1.0, fprime: fn(x: float) -> float { return 2.0 * x })
    assert_true(tangent.converged, "newton converged")
    assert_eq(tangent.root, 1.4142135623730951, "sqrt 2 by newton")
    assert_true(tangent.iterations < secant.iterations, "the derivative is worth having")
}

fn test_a_flat_derivative_is_reported_not_thrown() -> void {
    let found = newton(fn(x: float) -> float {
        return x * x + 1.0
    }, 0.0, fprime: fn(x: float) -> float {
        return 2.0 * x
    })
    assert_false(found.converged, "a flat derivative stops the search")
    assert_eq(found.flag, "derivative went flat", "the reason")
}

#: Every one from scipy's minimize_scalar(..., method="bounded"):
#:   (x-3)^2+1     on (0, 10)  -> x=3.0,                f=1.0
#:   (x+0.5)^4+3   on (-5, 5)  -> x=-0.5000532772618896 f=3.0
#:   sin(x)+x/2    on (-3, 0)  -> x=-2.0943942664279884 f=-1.9132229549807338
#:   exp(x)-2x     on (-2, 3)  -> x=0.6931455398130545  f=0.6137056388828015
fn test_minimize_scalar_matches_scipys_bounded_brent() -> void {
    let parabola = minimize_scalar(fn(x: float) -> float {
        return (x - 3.0) * (x - 3.0) + 1.0
    }, bounds: (0.0, 10.0))
    assert_true(parabola.success, "the parabola converged")
    assert_near(parabola.x[0], 3.0, 1e-06, "the parabola's minimum")
    assert_near(parabola.fun, 1.0, 1e-12, "the parabola's value")

    #: A quartic is flat at its minimum, so the ARGUMENT is only good to a
    #: quarter of the tolerance while the VALUE is exact -- which is why scipy
    #: lands at -0.50005 and not -0.5, and why this checks them differently.
    let quartic = minimize_scalar(fn(x: float) -> float {
        let shifted = x + 0.5
        return shifted * shifted * shifted * shifted + 3.0
    }, bounds: (-5.0, 5.0))
    assert_near(quartic.x[0], -0.5, 1e-03, "the quartic's minimum")
    assert_near(quartic.fun, 3.0, 1e-09, "the quartic's value")

    let wave = minimize_scalar(fn(x: float) -> float {
        return math.sin(x) + 0.5 * x
    }, bounds: (-3.0, 0.0))
    assert_near(wave.x[0], -2.0943942664279884, 1e-06, "the wave's minimum")
    assert_near(wave.fun, -1.9132229549807338, 1e-09, "the wave's value")

    #: Checked against the TRUE minimum, log 2, rather than against scipy's
    #: iterate: `exp(x) - 2x` is quadratic at its minimum, so the argument is
    #: only ever good to about the square root of the tolerance and the two
    #: implementations land on either side of it -- scipy at 0.69314554, this
    #: at 0.69314717, the truth at 0.69314718. The VALUE is what both get
    #: right, and it is what a minimiser is asked for.
    let curve = minimize_scalar(fn(x: float) -> float {
        return math.exp(x) - 2.0 * x
    }, bounds: (-2.0, 3.0))
    assert_near(curve.x[0], 0.6931471805599453, 1e-05, "log 2")
    assert_near(curve.fun, 0.6137056388828015, 1e-09, "the value there")
}

fn rosenbrock(v: list[float]) -> float {
    let a = 1.0 - v[0]
    let b = v[1] - v[0] * v[0]
    return a * a + 100.0 * b * b
}

fn rosenbrock_gradient(v: list[float]) -> list[float] {
    let b = v[1] - v[0] * v[0]
    return [-2.0 * (1.0 - v[0]) - 400.0 * v[0] * b, 200.0 * b]
}

#: scipy Nelder-Mead from (-1.2, 1.0) reaches f=1.0990889519195732e-18 in 117
#: iterations; scipy BFGS reaches 2.0488413879144732e-11 in 32. Both are
#: checked against the MINIMUM rather than against each other's step counts.
fn test_both_methods_find_the_rosenbrock_minimum() -> void {
    let simplex = minimize(rosenbrock, [-1.2, 1.0], maxiter: 2000)
    assert_true(simplex.success, "nelder-mead converged")
    assert_near(simplex.x[0], 1.0, 1e-06, "nelder-mead x")
    assert_near(simplex.x[1], 1.0, 1e-06, "nelder-mead y")
    assert_true(simplex.fun < 1e-12, "nelder-mead reached the floor")

    let quasi = minimize(rosenbrock, [-1.2, 1.0], method: "bfgs", maxiter: 2000)
    assert_true(quasi.success, "bfgs converged")
    assert_near(quasi.x[0], 1.0, 1e-04, "bfgs x")
    assert_near(quasi.x[1], 1.0, 1e-04, "bfgs y")
    assert_true(quasi.fun < 1e-09, "bfgs reached the floor")
}

#: An analytic gradient has to reach the same place as the difference
#: quotient, and in no more steps.
fn test_an_analytic_gradient_agrees_with_the_difference_quotient() -> void {
    let guessed = minimize(rosenbrock, [-1.2, 1.0], method: "bfgs", maxiter: 2000)
    let given = minimize(
        rosenbrock,
        [-1.2, 1.0],
        method: "bfgs",
        jac: rosenbrock_gradient,
        maxiter: 2000
    )
    assert_true(given.success, "bfgs with a gradient converged")
    assert_near(given.x[0], 1.0, 1e-04, "x")
    assert_near(given.x[1], 1.0, 1e-04, "y")
    assert_true(given.nfev < guessed.nfev, "a given gradient costs fewer calls")
}

#: scipy on (x-1.5)^2 + (y+2)^2 + 3:
#:   Nelder-Mead -> [1.50001992, -1.9999703], f=3.0000000012785297
#:   BFGS        -> [1.49999997, -2.00000001], f=3.000000000000001
fn test_a_separable_quadratic_lands_where_scipy_lands() -> void {
    let bowl = fn(v: list[float]) -> float {
        let a = v[0] - 1.5
        let b = v[1] + 2.0
        return a * a + b * b + 3.0
    }
    for method in ["nelder-mead", "bfgs"] {
        let found = minimize(bowl, [0.0, 0.0], method: method)
        assert_true(found.success, f"{method} converged")
        assert_near(found.x[0], 1.5, 1e-04, f"{method} x")
        assert_near(found.x[1], -2.0, 1e-04, f"{method} y")
        assert_near(found.fun, 3.0, 1e-08, f"{method} value")
    }
}

fn test_an_unknown_method_is_named_in_the_error() -> void {
    assert_throws(ValueError, fn() -> void {
        let _ = minimize(rosenbrock, [0.0, 0.0], method: "powell")
    }, "an unimplemented method raises")
}

#: A straight line fitted to six points, which `numpy.polyfit` and scipy's
#: curve_fit both put at slope 1.00285714, intercept 2.04285714.
fn test_curve_fit_recovers_a_line() -> void {
    let xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    let ys = [2.1, 2.9, 4.2, 5.1, 5.8, 7.2]
    let line = fn(x: list[float], p: list[float]) -> list[float] {
        var out: list[float] = []
        for value in x {
            out.push(p[0] * value + p[1])
        }
        return out
    }
    let fitted = curve_fit(line, xs, ys, [0.0, 0.0])
    assert_near(fitted[0], 1.00285714, 1e-07, "the slope")
    assert_near(fitted[1], 2.04285714, 1e-07, "the intercept")
}

#: Exact data from `2.5 * exp(-0.8 x)`, which scipy recovers to the printed
#: digits. A nonlinear model is what separates Levenberg-Marquardt from a
#: normal-equations solve, so this is the one that exercises the damping.
fn test_curve_fit_recovers_an_exponential() -> void {
    var xs: list[float] = []
    var ys: list[float] = []
    for index in 0..9 {
        let x = float(index) * 0.5
        xs.push(x)
        ys.push(2.5 * math.exp(-0.8 * x))
    }
    let decay = fn(x: list[float], p: list[float]) -> list[float] {
        var out: list[float] = []
        for value in x {
            out.push(p[0] * math.exp(-p[1] * value))
        }
        return out
    }
    let fitted = curve_fit(decay, xs, ys, [1.0, 1.0])
    assert_near(fitted[0], 2.5, 1e-06, "the amplitude")
    assert_near(fitted[1], 0.8, 1e-06, "the rate")
}

fn test_curve_fit_refuses_mismatched_data() -> void {
    let line = fn(x: list[float], p: list[float]) -> list[float] { return x }
    assert_throws(ValueError, fn() -> void {
        let _ = curve_fit(line, [1.0, 2.0], [1.0], [0.0])
    }, "x and y of different lengths raise")
}

#: least_squares reports HALF the sum of squares, as scipy's `cost` does, so a
#: perfect fit is zero and a known residual is checkable directly.
fn test_least_squares_reports_scipys_cost() -> void {
    let residual = fn(p: list[float]) -> list[float] {
        return [p[0] - 3.0, p[1] + 1.0, p[0] + p[1] - 2.0]
    }
    let found = least_squares(residual, [0.0, 0.0])
    assert_true(found.success, "least_squares converged")
    let r = residual(found.x)
    var cost = 0.0
    for value in r { cost += value * value }
    assert_near(found.fun, 0.5 * cost, 1e-12, "the cost is half the sum of squares")
    assert_true(found.fun < 0.34, "the overdetermined system is solved as well as it can be")
}

fn test_least_squares_refuses_fewer_residuals_than_parameters() -> void {
    assert_throws(ValueError, fn() -> void {
        let _ = least_squares(fn(p: list[float]) -> list[float] {
            return [p[0]]
        }, [0.0, 0.0])
    }, "an underdetermined system raises")
}
