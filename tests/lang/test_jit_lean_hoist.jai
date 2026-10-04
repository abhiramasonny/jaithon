#: Hoisted list headers that keep only `items`, a stored-into list whose bumped
#: version is held in a register, and subscripts that load and store through
#: `[items, idx, lsl #3]`.
#:
#: The version is what an iterator reads to notice its list changed under it,
#: so these are the shapes that would show a version written at the wrong time
#: or not at all: an iterator over the very list an inner loop stores into, an
#: inner loop that runs zero times, two locals naming one list. Also every
#: unboxed storage through the direct forms (int, float, bool) and a subscript
#: the head's guard does not cover, which has to load its own count. Every
#: expectation is the interpreter's (`JAITHON_NO_JIT=1`).
from std.test import assert_eq, assert_true

#: The inner loop stores into the list the outer loop iterates. The interpreter
#: raises at the outer loop's next step after the first store.
fn store_under_iterator(xs: list[float], n: int) -> int {
    var steps = 0
    try {
        for x in xs {
            steps += 1
            for j in 0..n { xs[j] = x + float(j) }
        }
    } catch e: RuntimeError {
        steps += 100
    }
    return steps
}

fn test_a_store_under_a_live_iterator_is_seen() -> void {
    # Few calls over long lists reach the loop tier, which is where headers
    # are hoisted; many calls over short ones reach the function tier.
    for size in [64, 64, 64, 30000, 30000] {
        var big: list[float] = []
        for k in 0..size { big.push(float(k)) }
        assert_eq(store_under_iterator(big, size), 101, "long: raised on the second step")
        var none: list[float] = []
        for k in 0..size { none.push(float(k)) }
        assert_eq(store_under_iterator(none, 0), size, "long: no store, no raise")
    }
    for _round in 0..120 {
        var xs: list[float] = []
        for k in 0..64 { xs.push(float(k)) }
        assert_eq(store_under_iterator(xs, 64), 101, "raised on the second step")
        var ys: list[float] = []
        for k in 0..64 { ys.push(float(k)) }
        # Zero trips: nothing stored, so nothing to notice.
        assert_eq(store_under_iterator(ys, 0), 64, "no store, no raise")
    }
}

#: The store happens only on the loop's LAST trip, so a loop entered in the
#: compiled tier after its first trip stores nothing until well inside the
#: region -- the version it writes then has to differ from the snapshot the
#: caller's iterator took, which it would not if the bump had been skipped.
fn store_last(xs: list[float], n: int) -> void {
    for j in 0..n {
        if j == n - 1 { xs[0] = 5.0 }
    }
}

fn iterate_and_store(xs: list[float], n: int) -> int {
    var steps = 0
    try {
        for _x in xs {
            steps += 1
            store_last(xs, n)
        }
    } catch e: RuntimeError {
        steps += 100
    }
    return steps
}

fn test_a_store_late_in_a_compiled_loop_is_seen_by_the_caller() -> void {
    for n in [50, 2000000, 2000000, 2000000] {
        var xs: list[float] = [1.0, 2.0, 3.0]
        assert_eq(iterate_and_store(xs, n), 101, f"n = {n}")
    }
}

fn aliased(a: list[float], b: list[float], n: int) -> float {
    for j in 0..n {
        a[j] = a[j] + 1.0
        b[j] = b[j] * 2.0
    }
    var s = 0.0
    for j in 0..n { s += a[j] }
    return s
}

fn test_two_locals_naming_one_list() -> void {
    for _round in 0..3 {
        var xs: list[float] = []
        for k in 0..200000 { xs.push(float(k % 9)) }
        var want = 0.0
        for k in 0..200000 { want += (float(k % 9) + 1.0) * 2.0 }
        assert_eq(aliased(xs, xs, 200000), want, "long: same list twice")
    }
    for _round in 0..80 {
        var xs: list[float] = []
        for k in 0..200 { xs.push(float(k % 9)) }
        var want = 0.0
        for k in 0..200 { want += (float(k % 9) + 1.0) * 2.0 }
        assert_eq(aliased(xs, xs, 200), want, "same list twice")
        var ys: list[float] = []
        var zs: list[float] = []
        for k in 0..200 {
            ys.push(float(k % 9))
            zs.push(1.0)
        }
        var want2 = 0.0
        for k in 0..200 { want2 += float(k % 9) + 1.0 }
        assert_eq(aliased(ys, zs, 200), want2, "two lists")
        assert_eq(zs[17], 2.0, "the second list was written")
    }
}

fn shift_ints(src: list[int], dst: list[int], n: int) -> int {
    for j in 1..n - 1 { dst[j] = src[j - 1] + src[j + 1] }
    var t = 0
    for j in 0..n { t += dst[j] }
    return t
}

fn flip_bools(src: list[bool], dst: list[bool], n: int) -> int {
    for j in 0..n { dst[j] = not src[j] }
    var c = 0
    for j in 0..n { if dst[j] { c += 1 } }
    return c
}

fn test_every_unboxed_storage() -> void {
    for _round in 0..3 {
        var src: list[int] = []
        var dst: list[int] = []
        var bs: list[bool] = []
        var bd: list[bool] = []
        for k in 0..200000 {
            src.push(k % 1000)
            dst.push(0)
            bs.push(k % 3 == 0)
            bd.push(false)
        }
        var want = 0
        for k in 1..199999 { want += (k - 1) % 1000 + (k + 1) % 1000 }
        assert_eq(shift_ints(src, dst, 200000), want, "long ints")
        assert_eq(flip_bools(bs, bd, 200000), 133333, "long bools")
    }
    for _round in 0..80 {
        var src: list[int] = []
        var dst: list[int] = []
        for k in 0..300 {
            src.push(k * 3 - 100)
            dst.push(0)
        }
        var want = 0
        for k in 1..299 { want += (k - 1) * 3 - 100 + (k + 1) * 3 - 100 }
        assert_eq(shift_ints(src, dst, 300), want, "ints")
        var bs: list[bool] = []
        var bd: list[bool] = []
        for k in 0..300 {
            bs.push(k % 3 == 0)
            bd.push(false)
        }
        assert_eq(flip_bools(bs, bd, 300), 200, "bools")
    }
}

#: `xs[k]` with a `k` the loop head knows nothing about: no guard covers it, so
#: it checks its own bounds against a count it loads itself.
fn uncovered(xs: list[float], ks: list[int], n: int) -> float {
    var s = 0.0
    for j in 0..n {
        let k = ks[j]
        xs[j] = xs[j] + 1.0
        s += xs[k]
    }
    return s
}

fn test_an_uncovered_subscript_checks_its_own_bounds() -> void {
    for _round in 0..3 {
        var xs: list[float] = []
        var ks: list[int] = []
        var bad: list[int] = []
        for k in 0..200000 {
            xs.push(float(k))
            ks.push(k * 7 % 200000)
            bad.push(if k == 150000 { 200000 } else { k })
        }
        assert_true(uncovered(xs, ks, 200000) > 0.0, "long: ran")
        var raised = false
        try {
            uncovered(xs, bad, 200000)
        } catch e: IndexError {
            raised = true
        }
        assert_true(raised, "long: one past the end still raises")
    }
    for _round in 0..80 {
        var xs: list[float] = []
        var ks: list[int] = []
        for k in 0..100 {
            xs.push(float(k))
            ks.push(k * 7 % 100)
        }
        let got = uncovered(xs, ks, 100)
        assert_true(got > 0.0, "ran")
        var bad: list[int] = []
        for k in 0..100 { bad.push(if k == 60 { 100 } else { k }) }
        var raised = false
        try {
            uncovered(xs, bad, 100)
        } catch e: IndexError {
            raised = true
        }
        assert_true(raised, "an index one past the end still raises")
    }
}
