#: A call that raises inside a `try` in a loop: the OSR tier compiles the
#: loop and the raise unwinds to the handler (see emitProtectedThrew), rather
#: than declining the loop the way the function tier still does.
#:
#: Each test warms its loop past the OSR threshold with calls that do not
#: raise, so the raise is taken in compiled code, then checks the handler ran
#: (and only the handler), the loop carried on, and values on both sides of
#: the raise are exact.
from std.test import assert_eq, assert_throws

fn checked(x: int) -> int {
    if x % 8 == 3 { throw ValueError(f"bad value {x}") }
    return x + 1
}

fn storm(n: int) -> int {
    var ok = 0
    var bad = 0
    var i = 0
    while i < n {
        try {
            ok += checked(i)
        } catch e: ValueError {
            bad += 1
        }
        i += 1
    }
    return ok - bad
}

fn storm_message(n: int) -> str {
    var seen = ""
    var i = 0
    while i < n {
        try {
            checked(i)
        } catch e: ValueError {
            seen = e.message
        }
        i += 1
    }
    return seen
}

fn checked_at(i: int, raise_at: int) -> int {
    if i == raise_at { throw ValueError(f"bad value {i}") }
    return i + 1
}

fn storm_uncaught(n: int, raise_at: int) -> int {
    var ok = 0
    var i = 0
    while i < n {
        try {
            ok += checked_at(i, raise_at)
        } catch _e: TypeError {
            return -1000
        }
        i += 1
    }
    return ok
}

fn storm_nested(n: int) -> int {
    var ok = 0
    var bad = 0
    var i = 0
    while i < n {
        try {
            try {
                ok += checked(i)
            } catch _inner: TypeError {
                return -1000
            }
        } catch _e: ValueError {
            bad += 1
        }
        i += 1
    }
    return ok - bad
}

fn test_storm_counts_match_the_interpreter() -> void {
    # 0..19999: 2500 throws, ok = 175010000, bad = 2500.
    assert_eq(storm(20000), 175007500)
}

fn test_handler_sees_the_message() -> void {
    assert_eq(storm_message(20), "bad value 19")
}

fn test_unmatched_handler_propagates() -> void {
    # Warm the loop with no raise so the raise below fires in compiled code,
    # then prove an unmatched handler lets it escape instead of catching it.
    assert_eq(storm_uncaught(20000, -1), 200010000)
    assert_throws(ValueError, storm_uncaught_raising)
}

fn storm_uncaught_raising() -> int {
    return storm_uncaught(20000, 19999)
}

fn test_nested_handlers_pick_the_match() -> void {
    assert_eq(storm_nested(20000), 175007500)
}

fn main() -> int {
    test_storm_counts_match_the_interpreter()
    test_handler_sees_the_message()
    test_unmatched_handler_propagates()
    test_nested_handlers_pick_the_match()
    print("protected-call ok")
    return 0
}
