#: A compiled for-in over a list hands its iterator back when the loop is
#: exhausted, and the next OP_GET_ITER reuses it instead of allocating
#: (JAITHON_JIT_ITER_RECYCLE).
#:
#: So these are the shapes where a reused iterator could carry something over
#: from its last loop, or be handed out while something still holds it: nested
#: loops, a `break` that leaves an iterator unexhausted, a loop left by an
#: exception, a mutation that must still be seen, an element kind that changes
#: mid-loop and sends the loop back to the interpreter, and recursion that
#: reaches the same loop from inside it. Every expectation is the
#: interpreter's (`JAITHON_NO_JIT=1`).
from std.test import assert_eq, assert_true

fn total(xs: list[int]) -> int {
    var t = 0
    for x in xs { t += x }
    return t
}

fn nested(rows: list[list[int]]) -> int {
    var t = 0
    for row in rows {
        for x in row { t += x }
        for x in row { t -= x * 2 }
    }
    return t
}

fn test_reused_iterators_start_fresh() -> void {
    let a = [1, 2, 3, 4]
    let b = [10, 20]
    let c: list[int] = []
    for _round in 0..300 {
        assert_eq(total(a), 10, "a")
        assert_eq(total(b), 30, "b")
        assert_eq(total(c), 0, "empty")
        assert_eq(total(a), 10, "a again")
    }
    let rows = [[1, 2], [3], [], [4, 5, 6]]
    for _round in 0..300 { assert_eq(nested(rows), 0 - 21, "nested") }
}

fn first_over(xs: list[int], limit: int) -> int {
    for x in xs {
        if x > limit { return x }
    }
    return -1
}

fn broken(xs: list[int], stop: int) -> int {
    var seen = 0
    for x in xs {
        if x == stop { break }
        seen += 1
    }
    return seen
}

fn test_loops_left_early_do_not_recycle_a_live_iterator() -> void {
    let xs = [5, 8, 1, 9, 3]
    for _round in 0..300 {
        assert_eq(first_over(xs, 7), 8, "return from inside")
        assert_eq(broken(xs, 9), 3, "break")
        assert_eq(total(xs), 26, "a fresh loop after both")
    }
}

fn raises_inside(xs: list[int]) -> int {
    var t = 0
    for x in xs {
        if x == 0 { throw ValueError("zero") }
        t += 100 // x
    }
    return t
}

fn test_a_loop_left_by_an_exception() -> void {
    for _round in 0..300 {
        var caught = false
        try {
            raises_inside([4, 2, 0, 1])
        } catch _e: ValueError {
            caught = true
        }
        assert_true(caught, "raised")
        assert_eq(raises_inside([4, 2, 1]), 25 + 50 + 100, "and the next loop is whole")
    }
}

fn mutate_while_iterating(xs: list[int]) -> int {
    var n = 0
    try {
        for x in xs {
            n += 1
            if x == 2 { xs.push(9) }
        }
    } catch _e: RuntimeError {
        n += 1000
    }
    return n
}

fn test_a_mutation_is_still_seen_by_a_reused_iterator() -> void {
    for _round in 0..300 {
        assert_eq(total([1, 1, 1]), 3, "warm a spare")
        assert_eq(mutate_while_iterating([1, 2, 3]), 1002, "raised at the step after the push")
    }
}

fn mixed(xs: list[any]) -> int {
    var t = 0
    for x in xs {
        if x is int { t += 1 } else { t += 100 }
    }
    return t
}

fn test_an_element_kind_change_mid_loop() -> void {
    for _round in 0..300 {
        assert_eq(mixed([1, 2, 3]), 3, "ints")
        assert_eq(mixed([1, "two", 3]), 102, "a string in the middle")
        assert_eq(total([7, 7]), 14, "after the deopt")
    }
}

fn depth_sum(xs: list[int], depth: int) -> int {
    var t = 0
    for x in xs {
        t += x
        if depth > 0 { t += depth_sum(xs, depth - 1) }
    }
    return t
}

fn test_recursion_inside_the_loop() -> void {
    for _round in 0..100 {
        # 6 + 3 * (6 + 3 * 6) = 78
        assert_eq(depth_sum([1, 2, 3], 2), 78, "recursive")
    }
}
