#: A `throw` on a branch that never runs, inside a compiled body.
#:
#: Reading the global that names the error class is something the tier cannot
#: emit, and it used to decline the whole function for it -- so every argument
#: check written the ordinary way made its function uncompilable. It now takes
#: the unarmed deopt instead: compiled up to the throw, interpreted from there.
#: What these check is that the cold path still WORKS, because a guard that
#: stops raising is far worse than a slow one.
from std.test import assert_eq, assert_true

fn guarded(value: int, lowest: int) -> int {
    if value < lowest { throw ValueError(f"{value} is below {lowest}") }
    return value * 2 + 1
}

fn test_the_guard_still_raises_after_the_body_compiles() -> void {
    var warm = 0
    for index in 0..40000 { warm += guarded(index % 50 + 5, 0) }

    var caught = 0
    var last = ""
    for index in 0..500 {
        try {
            let _unused = guarded(-index - 1, 0)
            assert_true(false, "the guard did not raise")
        } catch failure {
            caught += 1
            last = failure.message
        }
    }
    assert_eq(caught, 500, "a compiled body stopped raising from its cold branch")
    assert_eq(last, "-500 is below 0", "the raised message is wrong")
}

fn test_the_hot_path_is_unchanged_by_having_thrown() -> void {
    var before = 0
    for index in 0..40000 { before += guarded(index % 50 + 5, 0) }
    try {
        let _unused = guarded(-1, 0)
    } catch _failure {
    }
    var after = 0
    for index in 0..40000 { after += guarded(index % 50 + 5, 0) }
    assert_eq(after, before, "throwing changed what the hot path returns")
}

fn test_a_guard_that_starts_firing_midway() -> void {
    var total = 0
    var raised = 0
    for index in 0..30000 {
        let lowest = if index < 20000 { 0 } else { 1000000 }
        try {
            total += guarded(index % 50 + 5, lowest)
        } catch _failure {
            raised += 1
        }
    }
    assert_eq(raised, 10000, "the guard did not start firing when the bound moved")
}
