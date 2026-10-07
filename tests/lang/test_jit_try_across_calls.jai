#: A `try` around an indirect call whose compiled callee raises.
#:
#: A compiled body has no interpreter frame, so a raise that leaves it through
#: the exception exit unwinds from its CALLER: the handler the body's own `try`
#: installed is never searched. Every call arm therefore asks whether it sits
#: inside a protected region and declines there -- except the arm for a closure
#: held in a local, which lost the question when its bail became a deopt. So
#: `try { return f(x) } catch { ... }` compiled, the callee's OverflowError
#: went to the exception exit, and the `catch` was skipped: the error escaped
#: to whoever called the function holding the `try`.
#:
#: OSR loops are the exception to the decline: a call that writes nothing
#: (direct, indirect, or descriptor) records its offset and unwinds to the
#: handler from the compiled loop instead of stopping it (see
#: emitProtectedThrew). The function tier, and callees that write, still
#: decline -- which is what every test below pins.
#:
#: The interpreter, and a body that has not compiled yet, get every one of
#: these right, so each test warms the try-holder past the call threshold with
#: a call that does not raise before making the one that does.
# jit-compiles: doubled_past_range
from std.test import assert_eq

let WARM = 300

#: 2^62 at depth zero, doubled once per level on the way out, so any depth
#: past zero overflows -- from compiled code once this has compiled. Recursive,
#: so it is too big to inline and the caller has to make a real call.
fn doubled_past_range(depth: int) -> int {
    if depth < 1 {
        return 4611686018427387904
    }
    return doubled_past_range(depth - 1) * 2
}

fn one_level_up(depth: int) -> int {
    return doubled_past_range(depth) + 1
}

fn throws_past_zero(n: int) -> int {
    if n > 0 {
        throw ValueError(f"{n} is past zero")
    }
    return n + 1
}

fn guarded_param_call(f: fn(int) -> int, x: int) -> int {
    try {
        return f(x)
    } catch _failure {
        return -1
    }
}

fn guarded_closure_call(f: fn(int) -> int, x: int) -> int {
    try {
        return f(x)
    } catch _failure {
        return -1
    }
}

fn guarded_two_deep(f: fn(int) -> int, x: int) -> int {
    try {
        return f(x)
    } catch _failure {
        return -1
    }
}

fn guarded_throw(f: fn(int) -> int, x: int) -> int {
    try {
        return f(x)
    } catch failure: ValueError {
        assert_eq(failure.message, "1 is past zero")
        return -1
    }
}

#: The inner handler does not match an OverflowError, so the outer one must.
fn guarded_nested(f: fn(int) -> int, x: int) -> int {
    try {
        try {
            return f(x)
        } catch _wrong: ValueError {
            return -2
        }
    } catch _failure {
        return -1
    }
}

#: The `try` is inside the loop, so the loop head is outside the region.
fn caught_in_loop(f: fn(int) -> int, n: int, raise_at: int) -> int {
    var caught = 0
    var total = 0
    for i in 0..n {
        try {
            total += f(if i == raise_at { 1 } else { 0 }) % 7
        } catch _failure {
            caught += 1
        }
    }
    return caught * 1000 + total % 1000
}

fn test_a_parameter_callee_that_overflows() -> void {
    var warm = 0
    for _i in 0..WARM {
        warm += guarded_param_call(doubled_past_range, 0) % 7
    }
    assert_eq(
        guarded_param_call(doubled_past_range, 1),
        -1,
        "the callee's overflow escaped the try"
    )
}

fn test_a_closure_with_an_upvalue() -> void {
    let big = 4611686018427387904
    let grow = fn(n: int) -> int {
        if n > 0 {
            return big * 2
        }
        return n
    }
    var warm = 0
    for _i in 0..WARM {
        warm += guarded_closure_call(grow, 0)
    }
    assert_eq(guarded_closure_call(grow, 1), -1, "the closure's overflow escaped the try")
}

fn test_a_raise_two_calls_down() -> void {
    var warm = 0
    for _i in 0..WARM {
        warm += guarded_two_deep(one_level_up, 0) % 7
    }
    assert_eq(guarded_two_deep(one_level_up, 1), -1, "an overflow two calls down escaped the try")
}

fn test_a_thrown_error() -> void {
    var warm = 0
    for _i in 0..WARM {
        warm += guarded_throw(throws_past_zero, 0)
    }
    assert_eq(guarded_throw(throws_past_zero, 1), -1, "the callee's throw escaped the try")
}

fn test_the_outer_of_two_handlers() -> void {
    var warm = 0
    for _i in 0..WARM {
        warm += guarded_nested(doubled_past_range, 0) % 7
    }
    assert_eq(guarded_nested(doubled_past_range, 1), -1, "the overflow escaped both handlers")
}

fn test_a_try_inside_a_loop() -> void {
    var warm = 0
    for _i in 0..WARM {
        warm += caught_in_loop(doubled_past_range, 20, -1)
    }
    assert_eq(
        caught_in_loop(doubled_past_range, 20, 19),
        1076,
        "the overflow escaped the try inside the loop"
    )
}
