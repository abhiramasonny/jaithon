#: `jaisci.interpolate` against scipy, which is where every expected value came
#: from. The reference calls are named beside each block so they can be re-run.
#:
#: The tolerances are tight on purpose -- 1e-12 relative on numbers of order
#: one -- because these are closed-form constructions rather than iterations:
#: if the end conditions are scipy's, the answers agree to the last few digits,
#: and if they are not, they disagree in the FIRST few. A loose tolerance here
#: would pass a natural spline where not-a-knot was asked for.
import std.math
from std.test import assert_eq, assert_near, assert_true, assert_throws
from jaisci.interpolate import interp, CubicSpline, PchipInterpolator

let X = [0.0, 1.0, 2.0, 3.0, 4.0]
let CUBE = [0.0, 1.0, 8.0, 27.0, 64.0]
let Q = [0.5, 1.5, 2.5, 3.5, 0.0, 4.0]

fn assert_all_near(got: list[float], wanted: list[float], tol: float, what: str) -> void {
    assert_eq(got.len(), wanted.len(), f"{what}: one value per query")
    for index in 0..wanted.len() {
        assert_near(got[index], wanted[index], tol, f"{what} at {index}")
    }
}

#: numpy.interp holds the end VALUE outside the knots rather than extrapolating
#: the end slope, which is the one place it differs from every spline here.
fn test_linear_interpolation_matches_numpy() -> void {
    assert_all_near(interp(Q, X, CUBE), [0.5, 4.5, 17.5, 45.5, 0.0, 64.0], 1e-12, "interp")
    assert_all_near(interp([-5.0, 9.0], X, CUBE), [0.0, 64.0], 1e-12, "held outside")
}

#: scipy: CubicSpline(x, y)(q) with the default bc_type="not-a-knot".
#: On y = x^3 the not-a-knot spline reproduces the cubic almost exactly and the
#: natural one does not -- 0.125 against 0.098 at x = 0.5 -- so this pair also
#: says which end condition was used.
fn test_the_default_spline_is_not_a_knot() -> void {
    let wanted = [0.12500000000000033, 3.375, 15.625, 42.875, 0.0, 64.0]
    assert_all_near(CubicSpline(X, CUBE).map(Q), wanted, 1e-12, "not-a-knot")

    let natural = CubicSpline(X, CUBE, bc_type: "natural")
    let wanted_natural = [
        0.09821428571428564,
        3.455357142857143,
        15.330357142857142,
        43.97321428571429,
        0.0,
        64.0,
    ]
    assert_all_near(natural.map(Q), wanted_natural, 1e-12, "natural")
}

#: scipy: CubicSpline(x, y)(q, 1)
fn test_the_spline_derivative_matches_scipy() -> void {
    let curve = CubicSpline(X, CUBE)
    var slopes: list[float] = []
    for value in Q { slopes.push(curve.slope_at(value)) }
    assert_all_near(slopes, [0.75, 6.75, 18.75, 36.75, 0.0, 48.0], 1e-11, "the derivative")
}

#: Uneven knots and a curve that is not a polynomial, which is where an end
#: condition written from a half-remembered formula stops agreeing.
#: scipy on x = [0, 0.5, 1.7, 2, 4.5, 6], y = [1, -2, 0.5, 0.5, 3.25, -1],
#: queried at [-1, 0.25, 1, 1.85, 3, 5, 7] -- the ends extrapolate.
fn test_uneven_knots_agree_with_scipy() -> void {
    let xs = [0.0, 0.5, 1.7, 2.0, 4.5, 6.0]
    let ys = [1.0, -2.0, 0.5, 0.5, 3.25, -1.0]
    let q = [-1.0, 0.25, 1.0, 1.85, 3.0, 5.0, 7.0]

    assert_all_near(
        CubicSpline(xs, ys).map(q),
        [
            27.87003146886358,
            -1.1045831297272033,
            -1.435342233264513,
            0.5406463292782908,
            1.3772673078749644,
            2.890910897375012,
            -11.007590811458435,
        ],
        1e-11,
        "not-a-knot on uneven knots"
    )

    let natural = CubicSpline(xs, ys, bc_type: "natural")
    assert_all_near(
        natural.map(q),
        [
            2.382100671140954,
            -0.7886187080536915,
            -1.6793742232165045,
            0.5465599496644296,
            1.5739852348993288,
            2.4011372110365397,
            -4.401137211036539,
        ],
        1e-11,
        "natural on uneven knots"
    )

    assert_all_near(
        PchipInterpolator(xs, ys).map(q),
        [
            -7.602941176470576,
            -1.0235906862745099,
            -1.0596064814814814,
            0.5,
            1.4680000000000002,
            2.626851851851852,
            -4.782407407407408,
        ],
        1e-11,
        "pchip on uneven knots"
    )

    assert_all_near(
        interp(q, xs, ys),
        [1.0, -0.5, -0.9583333333333333, 0.5, 1.6, 1.8333333333333333, -1.0],
        1e-12,
        "linear on uneven knots"
    )
}

#: Three knots make the not-a-knot condition degenerate -- the whole curve is
#: one parabola -- so it gets its own pair of equations. scipy does the same,
#: and these are its numbers.
fn test_three_knots_take_the_degenerate_branch() -> void {
    let x3 = [0.0, 1.0, 3.0]
    let y3 = [2.0, 5.0, 4.0]
    let q3 = [0.5, 2.0, -1.0, 4.0]
    assert_all_near(
        CubicSpline(x3, y3).map(q3),
        [3.791666666666667, 5.666666666666666, -3.3333333333333304, 0.0],
        1e-11,
        "not-a-knot"
    )
    assert_all_near(
        CubicSpline(x3, y3, bc_type: "natural").map(q3),
        [3.7187500000000004, 5.375, -1.0000000000000009, 2.625000000000001],
        1e-11,
        "natural"
    )
    assert_all_near(
        PchipInterpolator(x3, y3).map(q3),
        [4.020833333333333, 4.875, 0.33333333333333215, 1.625],
        1e-11,
        "pchip"
    )
}

#: Two knots: the only cubic through two points under any end condition is the
#: straight line, and it extrapolates as one. scipy agrees.
fn test_two_knots_are_a_straight_line() -> void {
    let curve = CubicSpline([1.0, 4.0], [2.0, 8.0])
    assert_all_near(curve.map([2.0, 0.0, 5.0]), [4.0, 0.0, 10.0], 1e-12, "two knots")
}

#: What pchip is FOR: on data that rises, flattens and rises again, the cubic
#: spline dips below zero and overshoots past one, and pchip does neither.
#: scipy's own numbers for both, at thirteen points across [0, 3].
fn test_pchip_does_not_overshoot_where_the_spline_does() -> void {
    let xs = [0.0, 1.0, 2.0, 3.0]
    let ys = [0.0, 0.0, 1.0, 1.0]
    var q: list[float] = []
    for index in 0..13 { q.push(float(index) * 0.25) }

    let monotone = PchipInterpolator(xs, ys).map(q)
    assert_all_near(
        monotone,
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.15625, 0.5, 0.84375, 1.0, 1.0, 1.0, 1.0, 1.0],
        1e-12,
        "pchip"
    )
    for value in monotone {
        assert_true(value >= -1e-12 and value <= 1.0 + 1e-12, "pchip stayed inside the data")
    }

    let spline = CubicSpline(xs, ys).map(q)
    assert_all_near(
        spline,
        [
            0.0,
            -0.203125,
            -0.25,
            -0.171875,
            0.0,
            0.234375,
            0.5,
            0.765625,
            1.0,
            1.171875,
            1.25,
            1.203125,
            1.0,
        ],
        1e-12,
        "the cubic spline"
    )
    assert_true(spline[2] < -0.2, "the spline dips below the data")
    assert_true(spline[10] > 1.2, "and overshoots above it")
}

#: Every interpolator passes through its own knots exactly. A property rather
#: than a table: it holds for every input, so it cannot be satisfied by a
#: coincidence, and it is the one thing all four have in common.
fn test_every_interpolator_passes_through_its_knots() -> void {
    let xs = [0.0, 0.5, 1.7, 2.0, 4.5, 6.0]
    let ys = [1.0, -2.0, 0.5, 0.5, 3.25, -1.0]
    let notaknot = CubicSpline(xs, ys)
    let natural = CubicSpline(xs, ys, bc_type: "natural")
    let monotone = PchipInterpolator(xs, ys)
    for index in 0..xs.len() {
        assert_near(notaknot.at(xs[index]), ys[index], 1e-12, f"not-a-knot at knot {index}")
        assert_near(natural.at(xs[index]), ys[index], 1e-12, f"natural at knot {index}")
        assert_near(monotone.at(xs[index]), ys[index], 1e-12, f"pchip at knot {index}")
    }
    assert_all_near(interp(xs, xs, ys), ys, 1e-12, "interp at its knots")
}

fn test_knots_out_of_order_are_refused() -> void {
    assert_throws(ValueError, fn() -> void {
        let _ = CubicSpline([0.0, 2.0, 1.0], [1.0, 2.0, 3.0])
    }, "decreasing x")
    assert_throws(ValueError, fn() -> void {
        let _ = PchipInterpolator([0.0, 1.0, 1.0], [1.0, 2.0, 3.0])
    }, "a repeated knot")
    assert_throws(ValueError, fn() -> void {
        let _ = interp([0.5], [0.0, 2.0, 1.0], [1.0, 2.0, 3.0])
    }, "interp with decreasing x")
}

fn test_mismatched_or_short_inputs_are_refused() -> void {
    assert_throws(ValueError, fn() -> void {
        let _ = CubicSpline([0.0, 1.0, 2.0], [1.0, 2.0])
    }, "more x than y")
    assert_throws(ValueError, fn() -> void { let _ = CubicSpline([0.0], [1.0]) }, "one knot")
}

fn test_an_unknown_end_condition_is_named_in_the_error() -> void {
    assert_throws(ValueError, fn() -> void {
        let _ = CubicSpline(X, CUBE, bc_type: "clamped")
    }, "an unimplemented bc_type")
}
