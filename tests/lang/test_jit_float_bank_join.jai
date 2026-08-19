#: A branch merge must not land inside the FP bank's settle.
#:
#: The compiled tier keeps float operand-stack entries unboxed in v16..v25 and
#: writes each one back to its X register with an `fmov` when something that
#: cannot read the bank comes next (`fpSyncAll` in `src/vm/jit/jit_func.c`).
#: `branchTo` empties the bank before it records a fixup, so an edge arriving
#: at a join holds every entry in X already — and the settle the fall-through
#: edge needs has to be emitted BEFORE the offset map records the instruction,
#: or the arriving edge runs it too and copies a `d` register it never wrote
#: over a value that was already correct.
#:
#: `if flag { a[i] } else { a[0] }` is the shape that separates them. One arm
#: ends at the jump, so `fpWorthLoading` sees a branch next and leaves its
#: element in X; the other flows straight into the `OP_ADD` and leaves its
#: element in the bank. Before the fix the taken arm's sum came back as
#: garbage — `inf`, or a number around 1e+122 — and only for `float`, since an
#: `int` list never enters the bank at all.
#:
#: Every expectation here is the interpreter's, so `JAITHON_NO_JIT=1` proves
#: them and the plain run proves the compiled code agrees. Each function is
#: called well past `JAI_JIT_THRESHOLD` so the compiled body is what answers.
from std.test import assert_eq, assert_near

fn pick_index(values: list[float], n: int, flag: bool) -> float {
    var total = 0.0
    for i in 0..n { total = total + (if flag { values[i] } else { values[0] }) }
    return total
}

fn pick_constant(values: list[float], n: int, flag: bool) -> float {
    var total = 0.0
    for i in 0..n { total = total + (if flag { values[i] } else { 1.0 }) }
    return total
}

fn pick_reversed(values: list[float], n: int, flag: bool) -> float {
    var total = 0.0
    for i in 0..n {
        let value = if flag { values[n - 1 - i] } else { values[i] }
        total = total + value
    }
    return total
}

fn pick_int(values: list[int], n: int, flag: bool) -> int {
    var total = 0
    for i in 0..n { total = total + (if flag { values[i] } else { values[0] }) }
    return total
}

fn ramp(n: int) -> list[float] {
    var values: list[float] = []
    for i in 0..n { values.push(float(i)) }
    return values
}

fn test_a_branch_arm_that_ends_at_the_jump_keeps_its_element() -> void {
    let values = ramp(64)
    var taken = 0.0
    var other = 0.0
    for _repeat in 0..200 {
        taken = pick_index(values, 64, true)
        other = pick_index(values, 64, false)
    }
    #: 0 + 1 + ... + 63, against 64 copies of values[0], which is 0.
    assert_near(taken, 2016.0, 1e-9)
    assert_near(other, 0.0, 1e-9)
}

fn test_a_constant_arm_merges_with_an_element_arm() -> void {
    let values = ramp(64)
    var taken = 0.0
    var other = 0.0
    for _repeat in 0..200 {
        taken = pick_constant(values, 64, true)
        other = pick_constant(values, 64, false)
    }
    assert_near(taken, 2016.0, 1e-9)
    assert_near(other, 64.0, 1e-9)
}

fn test_both_arms_reading_elements_agree_whichever_runs() -> void {
    let values = ramp(64)
    var forward = 0.0
    var backward = 0.0
    for _repeat in 0..200 {
        backward = pick_reversed(values, 64, true)
        forward = pick_reversed(values, 64, false)
    }
    #: The same 64 elements either way, so a bank left holding one edge's copy
    #: shows up as the two disagreeing.
    assert_near(backward, 2016.0, 1e-9)
    assert_near(forward, 2016.0, 1e-9)
}

fn test_the_integer_form_is_unchanged() -> void {
    var values: list[int] = []
    for i in 0..64 { values.push(i) }
    var taken = 0
    for _repeat in 0..200 { taken = pick_int(values, 64, true) }
    assert_eq(taken, 2016)
}
