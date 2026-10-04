#: A loop's one `push` once kept its list's count and bumped version in
#: registers, writing both back after every push but never loading them.
#: That arm gave way to JAITHON_JIT_GROW_KEEPS; the shapes stay as a check.
#:
#: The shapes here are the ones where a count held in a register could drift
#: from the list: another local naming the same list and reading its length or
#: elements mid-loop, an iterator over the list the loop pushes to, a loop left
#: by an exception or a deopt part-way, nested loops, and growth across many
#: capacity boundaries. Every expectation is the interpreter's
#: (`JAITHON_NO_JIT=1`).
from std.test import assert_eq, assert_true

fn build(n: int) -> list[int] {
    var out: list[int] = []
    for i in 0..n { out.push(i * 3) }
    return out
}

fn build_watched(out: list[int], alias: list[int], n: int) -> int {
    var seen = 0
    for i in 0..n {
        out.push(i)
        seen += alias.len() + alias[i]
    }
    return seen
}

fn nested(n: int, m: int) -> list[int] {
    var out: list[int] = []
    for i in 0..n {
        for j in 0..m { out.push(i * m + j) }
    }
    return out
}

fn test_counts_and_contents_survive_growth() -> void {
    for n in [0, 1, 7, 8, 9, 1000, 100000] {
        let xs = build(n)
        assert_eq(xs.len(), n, f"len {n}")
        if n > 0 {
            assert_eq(xs[n - 1], (n - 1) * 3, "last")
            assert_eq(xs[0], 0, "first")
        }
    }
    let ys = nested(300, 70)
    assert_eq(ys.len(), 21000, "nested len")
    for k in [0, 1, 69, 70, 20999] { assert_eq(ys[k], k, "nested value") }
}

fn test_an_alias_reads_the_live_count() -> void {
    for n in [10, 1000, 50000] {
        var xs: list[int] = []
        # i-th step: len is i + 1 and element i is i.
        var want = 0
        for i in 0..n { want += i + 1 + i }
        assert_eq(build_watched(xs, xs, n), want, "alias")
        assert_eq(xs.len(), n, "pushed")
    }
}

fn push_while_iterating(xs: list[int], ys: list[int]) -> int {
    var steps = 0
    try {
        for x in xs {
            steps += 1
            ys.push(x)
        }
    } catch _e: RuntimeError {
        steps += 1000
    }
    return steps
}

fn test_pushing_onto_the_list_being_iterated() -> void {
    for _round in 0..200 {
        var xs = [1, 2, 3]
        assert_eq(push_while_iterating(xs, xs), 1001, "raised on the second step")
        assert_eq(xs.len(), 4, "the one push landed")
        var other: list[int] = []
        assert_eq(push_while_iterating([1, 2, 3], other), 3, "distinct")
        assert_eq(other.len(), 3, "distinct pushes")
    }
}

fn push_until(out: list[int], n: int, stop: int) -> int {
    for i in 0..n {
        if i == stop { throw ValueError("stop") }
        out.push(i)
    }
    return out.len()
}

fn test_a_loop_left_by_an_exception_leaves_the_right_count() -> void {
    for _round in 0..100 {
        var out: list[int] = []
        var raised = false
        try {
            push_until(out, 5000, 3333)
        } catch _e: ValueError {
            raised = true
        }
        assert_true(raised, "raised")
        assert_eq(out.len(), 3333, "count at the raise")
        assert_eq(out[3332], 3332, "last element")
        assert_eq(push_until(out, 10, 99), 3343, "and it keeps counting")
    }
}

fn mixed_push(out: list[any], vals: list[any]) -> int {
    for v in vals { out.push(v) }
    return out.len()
}

fn test_a_push_that_deopts_part_way() -> void {
    for _round in 0..100 {
        var out: list[any] = []
        let vals: list[any] = [1, 2, "three", 4.0, 5, null, 7]
        assert_eq(mixed_push(out, vals), 7, "count")
        assert_eq(str(out), "[1, 2, \"three\", 4.0, 5, null, 7]", "contents")
    }
}
