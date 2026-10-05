#: `list.filter` over a call-free predicate runs as a kernel: the predicate's
#: body compiled with the loop inside it, each true verdict copying its
#: element into the result. The result keeps exactly the elements the
#: per-call path keeps, in order; a predicate that raises or deoptimises part
#: way leaves the kernel on that element, with the ones before it kept.
from std.math import NAN
from std.test import assert_eq, assert_throws, assert_true

fn ints(n: int) -> list[int] {
    var xs: list[int] = []
    for i in 0..n { xs.push((i * 7919) % 1000 - 300) }
    return xs
}

fn floats(n: int) -> list[float] {
    var xs: list[float] = []
    for i in 0..n { xs.push(float((i * 7919) % 1000) / 7.0) }
    return xs
}

fn test_int_filter_matches_a_loop() -> void {
    let xs = ints(600)
    var want: list[int] = []
    for x in xs {
        if x % 3 == 1 { want.push(x) }
    }
    for _round in 0..40 {
        let got = xs.filter(|x| x % 3 == 1)
        assert_eq(len(got), len(want))
        assert_eq(got, want)
    }
}

fn test_float_filter_and_empty_result() -> void {
    let fs = floats(400)
    var want: list[float] = []
    for f in fs {
        if f > 70.0 { want.push(f) }
    }
    for _round in 0..40 {
        assert_eq(fs.filter(|f| f > 70.0), want)
        assert_eq(len(fs.filter(|f| f > 1.0e9)), 0)
    }
}

#: An overflow in the predicate on element 150 raises, as it would per call.
fn test_raise_mid_run() -> void {
    var xs: list[int] = []
    for i in 0..300 { xs.push(i) }
    for _round in 0..30 { let _k = xs.filter(|x| x * 2 > 100) }
    xs[150] = 9223372036854775807
    assert_throws(OverflowError, || xs.filter(|x| x * 2 > 100))
}

#: A NaN reaching an ordering compare deoptimises that element and raises.
fn test_deopt_mid_run() -> void {
    for round in 0..30 {
        var fs = floats(200)
        if round >= 20 {
            fs[90] = NAN
            assert_throws(TypeError, || fs.filter(|f| f > 70.0))
        } else {
            let kept = fs.filter(|f| f > 70.0)
            assert_eq(len(kept) > 0, true)
        }
    }
}

#: The result is an untyped list and takes anything afterwards.
fn test_result_takes_any_element() -> void {
    let xs = ints(200)
    var ys = xs.filter(|x| x > 0)
    for _round in 0..30 { ys = xs.filter(|x| x > 0) }
    let n = len(ys)
    ys.push("tail")
    assert_eq(ys[n], "tail")
    assert_eq(ys[0] > 0, true)
}

#: A result that kept few of many elements gives back the room it was
#: reserved, and stays an ordinary list: it grows, takes any element, and an
#: empty one is still a list.
fn test_a_sparse_result_shrinks_and_still_grows() -> void {
    var xs: list[int] = []
    for i in 0..5000 { xs.push((i * 7919) % 10007) }
    for _round in 0..30 {
        var ys = xs.filter(|x| x < 5)
        let none = xs.filter(|x| x < 0)
        assert_eq(len(none), 0)
        let n = len(ys)
        for v in ys { assert_true(v < 5) }
        ys.push(99)
        ys.insert(0, -1)
        ys.extend([7, 8])
        ys.push(2.5)
        assert_eq(len(ys), n + 5)
        assert_eq(ys[0], -1)
        assert_eq(ys[n + 1], 99)
        assert_eq(ys[n + 4], 2.5)
        var grown = none
        for k in 0..100 { grown.push(k) }
        assert_eq(grown[99], 99)
    }
}
