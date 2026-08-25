#: A loop in the OSR tier whose body calls a compiled callee that gives up.
#:
#: The tier's bail protocol -- "could not continue, throw the computation
#: away" -- rests on partial execution being invisible. That holds for the
#: FUNCTION tier: its locals are registers and its frame is entered fresh. It
#: does not hold for the OSR tier, whose locals ARE the interpreter's frame
#: slots and whose every way out syncs them back before returning. An OSR bail
#: part way through an iteration therefore left the loop variable advanced and
#: the locals written, and the interpreter resumed at the LOOP HEAD -- which
#: advanced the iterator again and silently dropped the rest of that
#: iteration's work.
#:
#: Once per bail, and a compiled callee bails when it deopts and its
#: continuation cannot finish in the tier -- which under `--gc-stress` happens
#: regularly. So `count += rule(value)` came back short, `pre += 1` before the
#: call did not, and the shortfall scaled inversely with the collection
#: period: 1326 lost at --gc-stress=100 against 25 unstressed, on the same
#: 150000-iteration loop.
#:
#: The indirect call now takes a deopt at the CALL OFFSET instead, which is
#: what the direct-call arm next to it always did: the record is taken with
#: the callee and its arguments still on the model's stack, so the interpreter
#: re-runs the call rather than losing it.
#:
#: `pre` is the control. It is incremented BEFORE the call, so it counted
#: correctly even when the bug was live; `first` and `second` come after, and
#: were both short by exactly the same amount. All three must agree.
# jit-compiles: through_a_closure
from std.test import assert_eq

let ITEMS = 150_000

#: Calls a global, so it is too big for the tier to inline into its caller and
#: has to be a real call -- which is the shape under test. A closure the caller
#: can inline never reaches the bail at all.
fn step(value: int) -> int {
    return 1
}

fn through_a_closure(items: list[int], rule: fn(int) -> int) -> int {
    var pre = 0
    var first = 0
    var second = 0
    for value in items {
        pre += 1
        let r = rule(value)
        first += r
        second += r
    }
    if pre != first { return -1 }
    if pre != second { return -2 }
    return first
}

fn build(n: int) -> list[int] {
    var items: list[int] = []
    var i = 0
    while i < n {
        items.push(i % 37)
        i += 1
    }
    return items
}

fn test_no_iteration_is_lost_when_the_callee_gives_up() -> void {
    assert_eq(through_a_closure(build(ITEMS), |v| step(v)), ITEMS)
}
