#: `list.map` over a call-free lambda runs as a map kernel: the lambda's body
#: compiled once more with the loop inside it, so each element costs the body
#: and not a call. Every way out of the body other than its return -- an
#: overflow, a deopt, a raise -- has to leave the kernel on exactly the element
#: it was on, with the elements before it stored and the one after it never
#: started. Each case maps enough elements, often enough, that the lambda has
#: compiled and the kernel is the path actually taken.
from std.math import NAN
from std.test import assert_eq, assert_throws, assert_true

var SCALE = 2

fn ints(n: int) -> list[int] {
    var xs: list[int] = []
    for i in 0..n { xs.push(i * 3 - 7) }
    return xs
}

fn floats(n: int) -> list[float] {
    var xs: list[float] = []
    for i in 0..n { xs.push(float(i) * 0.25) }
    return xs
}

fn test_int_map_matches_a_loop() -> void {
    let xs = ints(500)
    for _round in 0..40 {
        let ys = xs.map(|x| x * 3 + 1)
        assert_eq(len(ys), 500)
        for i in 0..500 { assert_eq(ys[i], xs[i] * 3 + 1) }
    }
}

fn test_float_and_mixed_kinds() -> void {
    let fs = floats(300)
    let xs = ints(300)
    for _round in 0..40 {
        let a = fs.map(|x| x * 0.5 + 1.0)
        let b = xs.map(|x| float(x) / 2.0)
        for i in 0..300 {
            assert_eq(a[i], fs[i] * 0.5 + 1.0)
            assert_eq(b[i], float(xs[i]) / 2.0)
        }
    }
}

#: An overflow on element 150 of 300 raises, as the per-call path does.
fn test_overflow_mid_run_raises() -> void {
    var xs: list[int] = []
    for i in 0..300 { xs.push(i) }
    for _round in 0..30 { let _ys = xs.map(|x| x * 4) }
    xs[150] = 4611686018427387904
    assert_throws(OverflowError, || xs.map(|x| x * 4))
    xs[150] = 150
    let ys = xs.map(|x| x * 4)
    assert_eq(ys[299], 1196)
}

#: A NaN reaching an ordering compare deoptimises that element to the
#: interpreter, which raises on it; the same map without it comes out of the
#: kernel whole.
fn test_deopt_mid_run_raises_on_that_element() -> void {
    for round in 0..30 {
        var fs = floats(200)
        if round >= 20 { fs[77] = NAN }
        if round >= 20 {
            assert_throws(TypeError, || fs.map(|x| if x > 10.0 { 1.0 } else { 2.0 }))
        } else {
            let ys = fs.map(|x| if x > 10.0 { 1.0 } else { 2.0 })
            assert_eq(len(ys), 200)
            assert_eq(ys[5], 2.0)
            assert_eq(ys[77], 1.0)
            assert_eq(ys[199], 1.0)
        }
    }
}

#: The result is an untyped list: it takes anything afterwards.
fn test_result_takes_any_element() -> void {
    let xs = ints(100)
    var ys = xs.map(|x| x + 1)
    for _round in 0..30 { ys = xs.map(|x| x + 1) }
    ys.push("tail")
    assert_eq(ys[0], -6)
    assert_eq(ys[100], "tail")
    assert_eq(len(ys), 101)
}

#: A global the lambda reads is read on every element of every run.
fn test_global_read_is_live() -> void {
    let xs = ints(100)
    for _round in 0..30 { let _ys = xs.map(|x| x * SCALE) }
    SCALE = 5
    let ys = xs.map(|x| x * SCALE)
    assert_eq(ys[10], xs[10] * 5)
    SCALE = 2
}

#: A boxed source list takes the per-element path and agrees with it.
fn test_boxed_source() -> void {
    var xs = []
    for i in 0..100 { xs.push(i) }
    xs.push(2.5)
    for _round in 0..30 {
        let ys = xs.map(|x| x * 2)
        assert_eq(ys[99], 198)
        assert_eq(ys[100], 5.0)
    }
}

#: An empty and a short list never reach a kernel and still map.
fn test_short_lists() -> void {
    let empty: list[int] = []
    for _round in 0..30 {
        assert_eq(len(empty.map(|x| x + 1)), 0)
        assert_eq([1, 2, 3].map(|x| x + 1), [2, 3, 4])
    }
    assert_true(true)
}
