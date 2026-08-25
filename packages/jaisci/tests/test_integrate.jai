#: `jaisci.integrate` against scipy, which is where every expected value came
#: from, and against the closed forms where one exists.
#:
#: Where this differs from scipy it is checked against the TRUTH rather than
#: against scipy's iterate, and the difference is named:
#:
#: - `quad` is QUADPACK's adaptive QK15 without the epsilon-algorithm
#:   extrapolation QAGS adds, so its error ESTIMATE is conservative -- on
#:   `1/(1+x^2)` it reports 5e-11 for an answer that is right to 2e-15. The
#:   value agrees with scipy to fourteen digits; the estimate is an upper
#:   bound, which is what an error estimate is for.
#: - `solve_ivp` is the same Dormand-Prince 5(4) tableau scipy uses and lands
#:   in the same place, but takes about a fifth more derivative calls to get
#:   there: scipy refines its first step and its rejection factor in ways this
#:   does not. Accuracy is checked; call counts are checked only as a bound.
import std.math
from std.test import assert_eq, assert_near, assert_true, assert_throws
from jaisci.integrate import quad, fixed_quad, trapezoid, simpson
from jaisci.integrate import cumulative_trapezoid, solve_ivp

#: scipy: integrate.quad(...) -> (value, abserr)
#:   exp(-x^2) on [0, 1]   -> 0.7468241328124271
#:   sin       on [0, pi]  -> 2.0
#:   1/(1+x^2) on [0, 1]   -> 0.7853981633974483  (= pi/4)
#:   x^3       on [0, 2]   -> 4.0
#:   exp       on [0, 5]   -> 147.41315910257663
fn test_quad_matches_scipy() -> void {
    let (gaussian, gaussian_error) = quad(fn(x: float) -> float {
        return math.exp(-x * x)
    }, 0.0, 1.0)
    assert_near(gaussian, 0.7468241328124271, 1e-13, "the error function's half")
    assert_true(gaussian_error < 1e-09, "and an estimate that bounds it")

    let (wave, _wave_error) = quad(fn(x: float) -> float { return math.sin(x) }, 0.0, math.PI)
    assert_near(wave, 2.0, 1e-13, "sin over a half period")

    let (arc, _arc_error) = quad(fn(x: float) -> float { return 1.0 / (1.0 + x * x) }, 0.0, 1.0)
    assert_near(arc, math.PI / 4.0, 1e-13, "atan 1")

    let (cube, _cube_error) = quad(fn(x: float) -> float { return x * x * x }, 0.0, 2.0)
    assert_near(cube, 4.0, 1e-12, "a polynomial the rule is exact on")

    let (growth, _growth_error) = quad(fn(x: float) -> float { return math.exp(x) }, 0.0, 5.0)
    assert_near(growth, 147.41315910257663, 1e-10, "exp over five")
}

#: The integral over a reversed interval is the negated one, which is what
#: every calculus convention and scipy both say. Worth pinning because the
#: adaptive loop has no reason to notice on its own.
fn test_a_reversed_interval_negates_the_integral() -> void {
    let f = fn(x: float) -> float { return x * x }
    let (forward, _fe) = quad(f, 0.0, 3.0)
    let (backward, _be) = quad(f, 3.0, 0.0)
    assert_near(forward, 9.0, 1e-12, "forward")
    assert_near(backward, -9.0, 1e-12, "backward")
    let (empty, empty_error) = quad(f, 2.0, 2.0)
    assert_eq(empty, 0.0, "an empty interval")
    assert_eq(empty_error, 0.0, "with no error")
}

#: A five-point Gauss rule is exact for degree nine, so a cubic is exact --
#: which is also scipy's answer, 4.000000000000002.
fn test_fixed_quad_is_exact_on_a_low_degree_polynomial() -> void {
    let cube = fn(x: float) -> float { return x * x * x }
    for order in [2, 3, 4, 5, 7] {
        assert_near(fixed_quad(cube, 0.0, 2.0, n: order), 4.0, 1e-12, f"the {order}-point rule")
    }
}

fn test_an_untabulated_order_is_named_in_the_error() -> void {
    assert_throws(ValueError, fn() -> void {
        let _ = fixed_quad(fn(x: float) -> float {
            return x
        }, 0.0, 1.0, n: 6)
    }, "an order with no table raises")
}

#: On [0, 1, 4, 9, 16, 25] with unit spacing -- five intervals, so Simpson
#: leaves one over and how that is closed is the whole difference between
#: implementations.
#:   numpy.trapezoid       -> 42.5
#:   scipy.simpson         -> 41.66666666666667   (= 125/3, and exact)
#:   cumulative_trapezoid  -> [0, 0.5, 3, 9.5, 22, 42.5]
fn test_the_sampled_rules_match_scipy() -> void {
    let y = [0.0, 1.0, 4.0, 9.0, 16.0, 25.0]
    assert_near(trapezoid(y), 42.5, 1e-12, "the trapezoid rule")
    assert_near(simpson(y), 125.0 / 3.0, 1e-12, "Simpson, exact on a quadratic")

    let running = cumulative_trapezoid(y)
    let wanted = [0.0, 0.5, 3.0, 9.5, 22.0, 42.5]
    assert_eq(running.len(), wanted.len(), "one value per sample")
    for index in 0..wanted.len() {
        assert_near(running[index], wanted[index], 1e-12, f"the running total at {index}")
    }
}

#: The last value of a running trapezoid is the whole trapezoid, on any
#: spacing. A property rather than a table, because it holds for every input
#: and so cannot be satisfied by a coincidence.
fn test_the_running_total_ends_at_the_whole_integral() -> void {
    let xs = [0.0, 0.5, 1.5, 1.7, 4.0]
    let ys = [1.0, 3.0, 2.0, 8.0, -1.0]
    let running = cumulative_trapezoid(ys, x: xs)
    assert_near(running[running.len() - 1], trapezoid(ys, x: xs), 1e-12, "the last value")
}

fn test_the_sampled_rules_refuse_mismatched_lengths() -> void {
    assert_throws(ValueError, fn() -> void {
        let _ = trapezoid([1.0, 2.0], x: [0.0])
    }, "trapezoid with a short x")
    assert_throws(ValueError, fn() -> void {
        let _ = cumulative_trapezoid([1.0, 2.0], x: [0.0])
    }, "cumulative_trapezoid with a short x")
}

#: scipy: solve_ivp(decay, (0, 5), [2.0], rtol=1e-6, atol=1e-9) ends at
#: 0.16417010146129843, against the closed form 2 exp(-2.5) =
#: 0.1641699972477976. Both are checked, at different tolerances and for
#: different reasons.
#:
#: The two implementations agree to 2.3e-10 -- far tighter than either agrees
#: with the truth, which both miss by 1.04e-7. That is not an error in either:
#: `rtol` was 1e-6 and the answer is 0.164, so 1.6e-7 is the accuracy that was
#: ASKED FOR, and landing at the same place inside it is what says the tableau
#: is right rather than merely self-consistent. Tightening the closed-form
#: assertion past what rtol buys would only pin the arithmetic of one machine.
fn test_solve_ivp_integrates_exponential_decay() -> void {
    let found = solve_ivp(fn(t: float, s: list[float]) -> list[float] {
        return [-0.5 * s[0]]
    }, (0.0, 5.0), [2.0], rtol: 1e-06, atol: 1e-09)
    assert_true(found.success, "the integration reached the end")
    let last = found.y[found.y.len() - 1]
    assert_near(last[0], 2.0 * math.exp(-2.5), 5e-07, "the closed form, to the rtol asked for")
    assert_near(last[0], 0.16417010146129843, 1e-09, "scipy's answer, far more closely")
    assert_near(found.t[found.t.len() - 1], 5.0, 1e-12, "the last time")
    assert_true(found.nfev < 300, "a bounded number of derivative calls")
}

#: A harmonic oscillator round one full period, which returns to where it
#: started -- the shape that catches a tableau whose orders do not match,
#: because a wrong one drifts in amplitude rather than merely in phase.
#: scipy ends at (0.9999999921021394, -1.2080661332403153e-09).
fn test_solve_ivp_returns_an_oscillator_to_its_start() -> void {
    let found = solve_ivp(fn(t: float, s: list[float]) -> list[float] {
        return [s[1], -s[0]]
    }, (0.0, 2.0 * math.PI), [1.0, 0.0], rtol: 1e-08, atol: 1e-10)
    assert_true(found.success, "the integration reached the end")
    let last = found.y[found.y.len() - 1]
    assert_near(last[0], 1.0, 1e-07, "back to the starting position")
    assert_near(last[1], 0.0, 1e-07, "and the starting velocity")
    assert_true(found.nfev < 1200, "a bounded number of derivative calls")
}

#: Integrating backwards is the same problem with the sign of the step
#: flipped, and the controller has to keep that straight -- getting it wrong
#: gives a step that never advances rather than a wrong answer.
fn test_solve_ivp_integrates_backwards() -> void {
    let found = solve_ivp(fn(t: float, s: list[float]) -> list[float] {
        return [-0.5 * s[0]]
    }, (5.0, 0.0), [2.0 * math.exp(-2.5)], rtol: 1e-08, atol: 1e-11)
    assert_true(found.success, "the backwards integration reached the end")
    assert_near(found.t[found.t.len() - 1], 0.0, 1e-12, "the last time")
    let last = found.y[found.y.len() - 1]
    assert_near(last[0], 2.0, 1e-06, "back to the initial condition")
}

fn test_an_empty_span_is_the_initial_condition() -> void {
    let found = solve_ivp(fn(t: float, s: list[float]) -> list[float] {
        return [1.0]
    }, (3.0, 3.0), [7.0])
    assert_true(found.success, "an empty span succeeds")
    assert_eq(found.t.len(), 1, "one time")
    assert_eq(found.y[0][0], 7.0, "the initial condition")
}

fn test_solve_ivp_needs_a_state() -> void {
    assert_throws(ValueError, fn() -> void {
        let _ = solve_ivp(fn(t: float, s: list[float]) -> list[float] {
            return []
        }, (0.0, 1.0), [])
    }, "an empty state raises")
}
