#: A negative subscript after a push in the same loop counts from the NEW end.
#:
#: A push that does not grow the list raises its count without calling out, so
#: a count kept in a register across the loop would go stale-low and `xs[-1]`
#: would normalise against it to an earlier element (JAITHON_JIT_HOIST_PUSH).
#: Each shape runs in the function tier (a helper called many times) and in
#: the loop tier (one long loop), the pushed list named by the same local or
#: by another. Every expectation is the interpreter's (`JAITHON_NO_JIT=1`).
from std.test import assert_eq

fn dedupe(xs: list[int]) -> list[int] {
    var out: list[int] = [xs[0]]
    for i in 1..xs.len() {
        let x = xs[i]
        if out[-1] != x { out.push(x) }
    }
    return out
}

fn last_after_push(xs: list[int], n: int) -> int {
    var s = 0
    for i in 0..n {
        xs.push(i)
        s += xs[-1]
    }
    return s
}

fn set_last_after_push(xs: list[int], n: int) -> list[int] {
    for i in 0..n {
        xs.push(0)
        xs[-1] = i + 100
    }
    return xs
}

fn last_after_push_alias(xs: list[int], ys: list[int], n: int) -> int {
    var s = 0
    for i in 0..n {
        ys.push(i * 2)
        s += xs[-1] + xs[0]
    }
    return s
}

fn test_dedupe_compares_with_the_last_pushed() -> void {
    let xs = [1, 1, 2, 2, 2, 3, 1, 1, 4, 4, 5]
    for _round in 0..500 {
        assert_eq(dedupe(xs), [1, 2, 3, 1, 4, 5], "dedupe")
    }
}

fn test_a_read_of_the_last_element_after_a_push() -> void {
    for _round in 0..300 {
        assert_eq(last_after_push([1, 2, 3, 4, 5, 6, 7, 8], 4), 6, "read")
    }
}

fn test_a_store_to_the_last_element_after_a_push() -> void {
    for _round in 0..300 {
        assert_eq(
            set_last_after_push([1, 2, 3, 4, 5, 6, 7, 8], 4),
            [1, 2, 3, 4, 5, 6, 7, 8, 100, 101, 102, 103],
            "store"
        )
    }
}

fn test_a_push_through_another_local() -> void {
    for _round in 0..300 {
        let xs = [7, 1, 2, 3, 4, 5, 6, 9]
        # 0 + 2 + 4 + 6 pushed, each read back as xs[-1], plus xs[0] each time.
        assert_eq(last_after_push_alias(xs, xs, 4), 12 + 4 * 7, "aliased")
        let fresh = [7, 1, 2, 3, 4, 5, 6, 9]
        let ys: list[int] = []
        assert_eq(last_after_push_alias(fresh, ys, 4), 4 * 9 + 4 * 7, "distinct")
    }
}

fn test_a_long_loop_that_pushes_then_reads_back() -> void {
    let n = 300000
    var xs: list[int] = [1, 2, 3, 4, 5, 6, 7, 8]
    var s = 0
    for i in 0..n {
        xs.push(i * 3)
        s += xs[-1] + xs[i % 8]
    }
    assert_eq(s, 3 * (n * (n - 1) // 2) + n // 8 * 36, "int")
    var ys: list[float] = [0.5, 1.5]
    var f = 0.0
    for i in 0..n {
        ys.push(float(i))
        f += ys[-2] + ys[0]
    }
    # ys[-2] is 1.5 first, then the previous push: 0, 1, ..., n - 2.
    let want = 1.5 + float((n - 1) * (n - 2) // 2) + 0.5 * float(n)
    assert_eq(f, want, "float")
    var zs: list[int] = [0, 0]
    for i in 0..n {
        zs.push(0)
        zs[-1] = i
        zs[-2] += 1
    }
    # Each push is set to i, then bumped once by the next round's zs[-2].
    assert_eq(zs[n + 1], n - 1, "store, newest")
    assert_eq(zs[n], n - 1, "store, one before")
    assert_eq(zs[2], 1, "store, first pushed")
    assert_eq(zs[1], 1, "store, the seed")
}
