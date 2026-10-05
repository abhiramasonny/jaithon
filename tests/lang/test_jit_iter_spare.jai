#: A nested `for x in xs` finished in compiled code hands its iterator back,
#: and the next loop's iterator is that same object, reset. Every case here
#: nests loops so one is always being opened while another just closed, and
#: checks the iteration still visits exactly its own list's elements -- with
#: a deopt in the middle of an inner loop, after which the interpreter owns
#: that iterator and the compiled code must not have handed it out again.
from std.test import assert_eq

fn rows(n: int, w: int) -> list[list[int]] {
    var out: list[list[int]] = []
    for i in 0..n {
        var r: list[int] = []
        for j in 0..w + i % 3 { r.push(i * 10 + j) }
        out.push(r)
    }
    return out
}

fn nested_sum(g: list[list[int]]) -> int {
    var t = 0
    for r in g {
        for x in r {
            for y in r { t += x * 3 + y }
        }
        for x in r { t -= x }
    }
    return t
}

fn reference(g: list[list[int]]) -> int {
    var t = 0
    for i in 0..len(g) {
        let r = g[i]
        for a in 0..len(r) {
            for b in 0..len(r) { t += r[a] * 3 + r[b] }
        }
        for a in 0..len(r) { t -= r[a] }
    }
    return t
}

fn test_nested_loops_keep_their_own_elements() -> void {
    let g = rows(60, 4)
    let want = reference(g)
    for _round in 0..150 { assert_eq(nested_sum(g), want) }
}

#: One row turns into floats half way: the inner loop deoptimises on it and
#: finishes interpreted, while later rows open new iterators in compiled code.
fn mixed_sum(g: list[any]) -> float {
    var t = 0.0
    for r in g {
        for x in r { t += float(x) }
    }
    return t
}

fn test_deopt_inside_an_inner_loop() -> void {
    var g: list[any] = []
    for i in 0..80 {
        var r: list[int] = []
        for j in 0..5 { r.push(i + j) }
        g.push(r)
    }
    var want = 0.0
    for i in 0..80 {
        for j in 0..5 { want += float(i + j) }
    }
    for _round in 0..150 { assert_eq(mixed_sum(g), want) }
    var odd: list[any] = []
    for i in 0..80 { odd.push(g[i]) }
    odd[40] = [40.0, 41.0, 42.0, 43.0, 44.0]
    assert_eq(mixed_sum(odd), want)
    for _round in 0..20 { assert_eq(mixed_sum(g), want) }
}
