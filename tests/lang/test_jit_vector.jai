#: The vectoriser (src/vm/jit/jit_vector.c) runs a `for j in a..b` whose body
#: is one float store two to eight elements at a time, in front of the scalar
#: loop, and must leave exactly what the scalar loop would have: the same bits
#: in every element (NaN, infinities and -0.0 included), the same recurrence
#: when the stored list is also read, the same wrap of a negative index, the
#: same IndexError at the same element, and nothing at all when a list's
#: storage is not unboxed floats. Each case compares the shape the vectoriser
#: takes against a `while` loop doing the same arithmetic in the same order,
#: which it never takes.
from std.test import assert_eq, assert_true, assert_throws
from std.math import NAN, INF, is_nan, copysign

fn same(a: float, b: float) -> bool {
    if is_nan(a) { return is_nan(b) }
    return a == b and copysign(1.0, a) == copysign(1.0, b)
}

fn same_lists(a: list[float], b: list[float]) -> bool {
    if a.len() != b.len() { return false }
    for i in 0..a.len() {
        if not same(a[i], b[i]) { return false }
    }
    return true
}

fn stencil(src: list[float], dst: list[float], lo: int, hi: int) -> void {
    for j in lo..hi {
        dst[j] = 0.25 * (src[j - 1] + src[j] * 2.0 + src[j + 1]) - src[j]
    }
}

fn stencil_ref(src: list[float], dst: list[float], lo: int, hi: int) -> void {
    var j = lo
    while j < hi {
        dst[j] = 0.25 * (src[j - 1] + src[j] * 2.0 + src[j + 1]) - src[j]
        j += 1
    }
}

fn axpy(a: float, x: list[float], y: list[float], out: list[float], n: int) -> void {
    for i in 0..n { out[i] = a * x[i] + y[i] }
}

fn axpy_ref(a: float, x: list[float], y: list[float], out: list[float], n: int) -> void {
    var i = 0
    while i < n {
        out[i] = a * x[i] + y[i]
        i += 1
    }
}

fn shift(a: list[float], b: list[float], n: int) -> void {
    for j in 0..n { b[j + 1] = a[j] }
}

fn shift_ref(a: list[float], b: list[float], n: int) -> void {
    var j = 0
    while j < n {
        b[j + 1] = a[j]
        j += 1
    }
}

fn running(a: list[float], n: int) -> void {
    for j in 1..n { a[j] = a[j - 1] + a[j] }
}

fn running_ref(a: list[float], n: int) -> void {
    var j = 1
    while j < n {
        a[j] = a[j - 1] + a[j]
        j += 1
    }
}

fn untyped(src, dst, lo: int, hi: int) -> void {
    for j in lo..hi { dst[j] = src[j] * 0.5 + src[j + 1] }
}

fn untyped_ref(src, dst, lo: int, hi: int) -> void {
    var j = lo
    while j < hi {
        dst[j] = src[j] * 0.5 + src[j + 1]
        j += 1
    }
}

fn specials(n: int) -> list[float] {
    let pool = [1.5, -0.0, 0.0, NAN, INF, -INF, 1e-310, -2.5e-308, 1e308,
                -1e308, 0.1, 3.0, -7.25, 2.0, 1e-300, 4.9e-324]
    var out: list[float] = []
    for i in 0..n { out.push(pool[(i * 7 + 3) % pool.len()]) }
    return out
}

fn zeros(n: int) -> list[float] {
    var out: list[float] = []
    for _ in 0..n { out.push(0.0) }
    return out
}

fn copy(xs: list[float]) -> list[float] {
    var out: list[float] = []
    for x in xs { out.push(x) }
    return out
}

fn test_every_length_and_window_is_bit_identical() -> void {
    for _round in 0..40 {
        for n in 2..24 {
            let src = specials(n)
            for lo in 1..n {
                for hi in lo..n {
                    var got = zeros(n)
                    var want = zeros(n)
                    stencil(src, got, lo, hi)
                    stencil_ref(src, want, lo, hi)
                    assert_true(same_lists(got, want), f"n={n} lo={lo} hi={hi}")
                }
            }
        }
    }
}

fn test_a_float_local_is_an_invariant() -> void {
    for _round in 0..200 {
        for n in 0..21 {
            let x = specials(n + 3)
            let y = specials(n + 5)
            var got = zeros(n + 1)
            var want = zeros(n + 1)
            axpy(-1.75, x, y, got, n)
            axpy_ref(-1.75, x, y, want, n)
            assert_true(same_lists(got, want), f"n={n}")
        }
    }
}

fn test_a_list_stored_into_itself_at_an_offset_is_a_recurrence() -> void {
    for _round in 0..200 {
        for n in 0..19 {
            var got = specials(n + 1)
            var want = copy(got)
            shift(got, got, n)
            shift_ref(want, want, n)
            assert_true(same_lists(got, want), f"shift n={n}")
            var g2 = specials(n + 2)
            var w2 = copy(g2)
            running(g2, n + 1)
            running_ref(w2, n + 1)
            assert_true(same_lists(g2, w2), f"running n={n}")
        }
    }
}

fn test_a_negative_index_still_wraps() -> void {
    for _round in 0..200 {
        let src = specials(17)
        var got = zeros(17)
        var want = zeros(17)
        stencil(src, got, 0, 16)
        stencil_ref(src, want, 0, 16)
        assert_true(same_lists(got, want))
    }
}

fn test_an_index_past_the_end_raises_at_the_same_element() -> void {
    for _round in 0..200 {
        let src = specials(13)
        var got = zeros(13)
        var want = zeros(13)
        assert_throws(IndexError, || stencil(src, got, 1, 13))
        assert_throws(IndexError, || stencil_ref(src, want, 1, 13))
        assert_true(same_lists(got, want))
    }
}

fn test_lists_that_are_not_unboxed_floats_take_the_scalar_loop() -> void {
    for _round in 0..200 {
        var mixed = [1.0, 2, 3.5, -0.0, 4, 5.25, 6, 7.0, 8, 9.5, 10, 11.0]
        var got = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        var want = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        untyped(mixed, got, 0, 11)
        untyped_ref(mixed, want, 0, 11)
        for i in 0..12 { assert_eq(got[i], want[i]) }
        var fl = specials(12)
        var g2 = zeros(12)
        var w2 = zeros(12)
        untyped(fl, g2, 0, 11)
        untyped_ref(fl, w2, 0, 11)
        assert_true(same_lists(g2, w2))
    }
}
