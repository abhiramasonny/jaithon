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

#: `break` out of a for-in drops the iterator and jumps to the exit; the walk
#: has to carry on after that jump with the iterator back on its model, or the
#: whole function declines (JAITHON_JIT_BREAK_ITER). Nested, with work after
#: the break, and with a break that is never taken.
fn find_pair(rows: list[list[int]], want: int) -> int {
    var found = -1
    var r = 0
    for row in rows {
        for v in row {
            if v == want {
                found = r
                break
            }
            if v < 0 { break }
        }
        if found >= 0 { break }
        r += 1
    }
    return found * 100 + r
}

fn test_break_out_of_nested_for_in() -> void {
    let rows = [[1, 2, 3], [4, -1, 9], [7, 8, 9], [10]]
    for _round in 0..300 {
        assert_eq(find_pair(rows, 8), 202, "found in row 2")
        assert_eq(find_pair(rows, 9), 202, "the -1 hides row 1's 9")
        assert_eq(find_pair(rows, 42), -100 + 4, "never found")
        assert_eq(broken([1, 2, 3], 7), 3, "break never taken")
    }
}

#: The iterator's index is stored back after every step, so a
#: `continue`, a deopt part-way and a store under the iterator must all still
#: see the index the interpreter would.
fn odd_total(xs: list[int]) -> int {
    var t = 0
    for x in xs {
        if x % 2 == 0 { continue }
        t += x
    }
    return t
}

fn store_under(xs: list[int]) -> int {
    var n = 0
    try {
        for x in xs {
            n += 1
            if x == 3 { xs[0] = 99 }
        }
    } catch _e: RuntimeError {
        n += 1000
    }
    return n
}

fn test_an_index_held_in_a_register() -> void {
    var long: list[int] = []
    for k in 0..200000 { long.push(k % 10) }
    for _round in 0..40 {
        assert_eq(odd_total([1, 2, 3, 4, 5, 6, 7]), 16, "continue")
        assert_eq(odd_total(long), 500000, "long continue")
        assert_eq(store_under([1, 2, 3, 4, 5]), 1003, "store under the iterator")
        assert_eq(mixed([5, 6, "x", 8]), 103, "deopt part-way")
    }
}

#: A `break` out of the very loop an OSR region was compiled for: the iterator
#: is not on that region's model, so the break is a jump to the iterator's own
#: exit, whose stub has the interpreter drop it. One long first call so the
#: loop is entered compiled, over each head kind the region can have.
fn index_of_first(xs: list[int], limit: int) -> int {
    var i = 0
    for x in xs {
        if x >= limit { break }
        i += 1
    }
    return i
}

fn range_until(n: int, stop: int) -> int {
    var t = 0
    for i in 0..n {
        if i == stop { break }
        t += i
    }
    return t
}

fn items_until(d: dict[str, int], stop: int) -> int {
    var t = 0
    for (_k, v) in d.items() {
        if v == stop { break }
        t += v
    }
    return t
}

fn enumerate_until(xs: list[int], stop: int) -> int {
    var t = 0
    for (i, x) in xs.enumerate() {
        if x == stop { break }
        t += i * x
    }
    return t
}

fn test_break_out_of_an_osr_region() -> void {
    var long: list[int] = []
    for k in 0..300000 { long.push(k % 1000) }
    long.push(5000)
    var d: dict[str, int] = {}
    for k in 0..50000 { d[f"k{k}"] = k }
    var small: list[int] = []
    for k in 0..200000 { small.push(k % 7) }
    small.push(99)
    var want = 0
    for k in 0..200000 { want += k * (k % 7) }
    for _round in 0..3 {
        assert_eq(index_of_first(long, 5000), 300000, "list, break at the end")
        assert_eq(index_of_first(long, 999), 999, "list, break early")
        assert_eq(index_of_first(long, 6000), 300001, "list, never")
        assert_eq(range_until(1000000, 400000), 79999800000, "range")
        assert_eq(range_until(10, 400000), 45, "range, never")
        assert_eq(items_until(d, 40000), 799980000, "items")
        assert_eq(enumerate_until(small, 99), want, "enumerate")
        assert_eq(enumerate_until(small, 6), 55, "enumerate early")
    }
}
