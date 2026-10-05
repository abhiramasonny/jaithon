#: `for c in lo..hi` puts `c` at most INT64_MAX - 1 (the range is exclusive)
#: and, when every range that starts its counter starts at a literal, at least
#: the smallest literal -- so `c + 1` and `c - 1` cannot overflow and are not
#: checked. Each case here is one where the bound does NOT hold, run after the
#: body has compiled, and must still raise exactly where the interpreter does.
from std.test import assert_eq, assert_throws

let MAX = 9223372036854775807
let MIN = -9223372036854775807 - 1

fn up_sum(a: int, b: int) -> int {
    var t = 0
    for i in a..b { t += (i + 1) % 7 }
    return t
}

fn up2_sum(a: int, b: int) -> int {
    var t = 0
    for i in a..b { t += (i + 2) % 7 }
    return t
}

fn incl_sum(a: int, b: int) -> int {
    var t = 0
    for i in a..=b { t += (i + 1) % 7 }
    return t
}

fn down_sum(a: int, b: int) -> int {
    var t = 0
    for i in a..b { t += (i - 1) % 7 }
    return t
}

fn lit_down_sum(n: int) -> int {
    var t = 0
    for i in 1..n { t += (i - 1) % 7 }
    return t
}

#: The same counter temporaries can serve two loops in one body; the second
#: one's start is not a literal, so neither gets a lower bound from it.
fn two_loops(a: int, n: int) -> int {
    var t = 0
    for i in 5..n { t += (i - 1) % 7 }
    for j in a..n { t += (j - 1) % 7 }
    return t
}

fn branchy_start(flag: bool, n: int) -> int {
    var t = 0
    for i in (if flag { MIN } else { 0 })..n { t += (i - 1) % 7 }
    return t
}

fn warm() -> void {
    for _r in 0..200 {
        let _a = up_sum(0, 50)
        let _b = up2_sum(0, 50)
        let _c = incl_sum(0, 50)
        let _d = down_sum(1, 50)
        let _e = lit_down_sum(50)
        let _f = two_loops(3, 50)
        let _g = branchy_start(false, 50)
    }
}

fn test_exclusive_top_cannot_overflow_plus_one() -> void {
    warm()
    assert_eq(up_sum(MAX - 3, MAX), up_sum(MAX - 3, MAX))
    assert_throws(OverflowError, || up2_sum(MAX - 3, MAX))
}

fn test_inclusive_top_still_checked() -> void {
    warm()
    assert_throws(OverflowError, || incl_sum(MAX - 3, MAX))
}

fn test_start_not_a_literal_still_checked() -> void {
    warm()
    assert_throws(OverflowError, || down_sum(MIN, MIN + 3))
    assert_throws(OverflowError, || two_loops(MIN, MIN + 3))
    assert_throws(OverflowError, || branchy_start(true, MIN + 3))
}

fn test_literal_start_matches_interpreter() -> void {
    warm()
    var want = 0
    for i in 1..1000 { want += (i - 1) % 7 }
    assert_eq(lit_down_sum(1000), want)
}
