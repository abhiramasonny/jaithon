#: Overflow checks the compiled tier drops because a comparison proves them
#: dead (jit_range.c), and the ones it must keep.
#:
#: `i < n` puts `i` at most INT64_MAX - 1, so `i += 1` straight after cannot
#: overflow; `n >= 2` puts `n - 2` far from INT64_MIN. Everything here runs at
#: the edge of the int range, so a check dropped where the comparison did NOT
#: prove it shows up as a wrapped number instead of an OverflowError.
#:
#: Each function is called enough times to reach the compiled tier, and each
#: answer is checked against what the interpreter computes.
# jit-compiles: count_up, dec2, count_up_inclusive, after_write
from std.test import assert_eq, assert_throws

const MAX = 9223372036854775807
const MIN = -9223372036854775807 - 1

#: `i < n` bounds `i`, so the increment is unchecked -- and stays right.
fn count_up(start: int, n: int) -> int {
    var i = start
    while i < n {
        i += 1
    }
    return i
}

fn test_strict_bound_counts_to_the_top() {
    for _r in 0..200 {
        assert_eq(count_up(MAX - 10, MAX), MAX)
        assert_eq(count_up(0, 50), 50)
    }
}

#: `<=` proves nothing about `i + 1`: at i == MAX it overflows, and must say so.
fn count_up_inclusive(start: int, n: int) -> int {
    var i = start
    while i <= n {
        i += 1
    }
    return i
}

fn test_inclusive_bound_still_overflows() {
    for _r in 0..200 {
        assert_eq(count_up_inclusive(0, 40), 41)
    }
    assert_throws(OverflowError, || count_up_inclusive(MAX - 3, MAX))
}

#: The false edge of `n < 2` is `n >= 2`; the true edge proves nothing about a
#: lower bound, so a subtraction there keeps its check.
fn dec2(n: int) -> int {
    if n < 2 {
        return n - 1
    }
    return n - 2
}

fn test_both_edges_of_a_constant_compare() {
    for _r in 0..200 {
        assert_eq(dec2(10), 8)
        assert_eq(dec2(2), 0)
        assert_eq(dec2(1), 0)
        assert_eq(dec2(MIN + 1), MIN)
    }
    assert_throws(OverflowError, || dec2(MIN))
}

#: A write between the comparison and the arithmetic retires the fact.
fn after_write(i: int, n: int, big: int) -> int {
    var j = i
    if j < n {
        j = big
        j += 1
    }
    return j
}

fn test_a_write_retires_the_bound() {
    for _r in 0..200 {
        assert_eq(after_write(0, 10, 5), 6)
        assert_eq(after_write(20, 10, 5), 20)
    }
    assert_throws(OverflowError, || after_write(0, 10, MAX))
}

#: A comparison against a float says nothing an int step can lean on.
fn float_bound(start: int, limit: float) -> int {
    var i = start
    var n = 0
    while n < 3 {
        if i < limit {
            i += 1
        }
        n += 1
    }
    return i
}

fn test_float_compare_keeps_the_check() {
    for _r in 0..200 {
        assert_eq(float_bound(0, 10.0), 3)
    }
    assert_throws(OverflowError, || float_bound(MAX, 1.0e300))
}
