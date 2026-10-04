#: `2.0 * x` compiled as `x + x`.
#:
#: The two are the same double for every x -- the product by two is exact, so
#: there is no rounding to differ, and NaN, the infinities, -0.0 and overflow
#: to infinity all come out of an add the way they come out of the multiply.
#: The add is a cycle shorter on a dependency chain, which is what mandelbrot's
#: `y = 2.0 * xy + y0` sits on (JAITHON_JIT_FTWO). Every expectation here is
#: the interpreter's (`JAITHON_NO_JIT=1`); the loops run long enough to be
#: compiled by both tiers.
from std.math import INF, NAN, is_nan
from std.test import assert_eq, assert_true

fn doubled_left(xs: list[float]) -> list[float] {
    var out: list[float] = []
    for x in xs { out.push(2.0 * x) }
    return out
}

fn doubled_right(xs: list[float]) -> list[float] {
    var out: list[float] = []
    for x in xs { out.push(x * 2.0) }
    return out
}

fn chain(seed: float, n: int) -> float {
    var y = seed
    for _i in 0..n { y = 2.0 * y - y * 2.0 + y * 0.5 + 2.0 * 0.25 }
    return y
}

fn test_doubling_matches_for_every_kind_of_value() -> void {
    let xs = [0.0, -0.0, 1.5, -3.25, 1e308, -1e308, 5e-324, INF, -INF, NAN, 0.1, 123456.789]
    for _round in 0..200 {
        let a = doubled_left(xs)
        let b = doubled_right(xs)
        for i in 0..xs.len() {
            let want = xs[i] + xs[i]
            if is_nan(want) {
                assert_true(is_nan(a[i]) and is_nan(b[i]), "NaN stays NaN")
            } else {
                assert_eq(str(a[i]), str(want), f"2.0 * {xs[i]}")
                assert_eq(str(b[i]), str(want), f"{xs[i]} * 2.0")
            }
        }
    }
    assert_eq(str(doubled_left([-0.0])[0]), "-0.0", "the sign of zero")
    assert_eq(doubled_left([1e308])[0], INF, "overflow to infinity")
}

fn test_a_doubling_on_a_loop_carried_chain() -> void {
    for _round in 0..100 {
        assert_eq(chain(1.0, 1000), chain(1.0, 1000), "deterministic")
    }
    var y = 1.0
    for _i in 0..1000 { y = (y + y) - (y + y) + y * 0.5 + 0.5 }
    assert_eq(chain(1.0, 1000), y, "matches the spelled-out adds")
}
