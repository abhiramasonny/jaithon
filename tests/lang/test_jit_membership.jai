#: `x in c`, `x not in c` and tuple construction, none of which had an arm.
#:
#: Both ENDED THE WALK, so a loop containing either ran interpreted from that
#: point on. Measured in one binary through `JAITHON_JIT_MEMBERSHIP` and
#: `JAITHON_JIT_TUPLE`:
#:
#:   (i & 15, i & 7)     24,018,742 -> 46,378       0.12s -> 0.055s  2.2x
#:   (i & 7) in xs        9,824,657 -> 1,363,630    0.08s -> 0.03s   2.7x
#:
#: The switched-off side takes the unarmed path, so these are ratios against no
#: arm at all -- see the note above `unarmedOpcode`.
#:
#: Neither arm makes the operation itself faster. `in` calls out to the very
#: same `jaiContainsOp` the interpreter runs, and the tuple to `jaiTupleNew`;
#: what the arms buy is the body around them. That is the whole test for a row
#: over a call -- the call has to be cheap next to the loop it sits in, and
#: membership in a short list or a dict is.
#:
#: Calling the interpreter's own containment is also what keeps the throw
#: identical. `1 in 5` reports `TypeError: 'in' is not supported for 'int'`
#: from compiled code and from the interpreter, one message with one source.
# jit-compiles: count_present, count_absent, sum_pairs
from std.test import assert_eq, assert_true, assert_false, assert_throws

#: Above JAI_JIT_THRESHOLD entries, so the bodies below reach the compiled tier.
let CALLS = 200

fn count_present(xs: list, rounds: int) -> int {
    var seen = 0
    var i = 0
    while i < rounds {
        if i & 7 in xs { seen += 1 }
        i += 1
    }
    return seen
}

fn count_absent(xs: list, rounds: int) -> int {
    var seen = 0
    var i = 0
    while i < rounds {
        if i & 7 not in xs { seen += 1 }
        i += 1
    }
    return seen
}

fn sum_pairs(rounds: int) -> int {
    var t = 0
    var i = 0
    while i < rounds {
        let p = (i & 15, i & 7)
        #: `len` and not `p.get(0)`: the arm pushes a plain SLOT_OBJ, and an
        #: invoke on an object the model has no sample for is declined
        #: ("an invoke on an object with nothing to look at"). So the built
        #: tuple can be measured but not yet read from compiled code -- which
        #: is the next thing worth an exemplar, not a reason to skip the arm.
        t += len(p)
        i += 1
    }
    return t
}

pub fn test_membership_over_a_list() -> void {
    let xs = [1, 2, 3]
    var present = 0
    var absent = 0
    var call = 0
    while call < CALLS {
        #: i & 7 walks 0..7, of which exactly three are in xs.
        present += count_present(xs, 8)
        absent += count_absent(xs, 8)
        call += 1
    }
    assert_eq(present, 3 * CALLS, "present")
    assert_eq(absent, 5 * CALLS, "absent")
}

pub fn test_membership_answers_the_same_as_the_interpreter() -> void {
    let xs = [1, 2, 3]
    assert_true(2 in xs, "present")
    assert_false(9 in xs, "absent")
    assert_false(2 not in xs, "not in present")
    assert_true(9 not in xs, "not in absent")
    assert_false(0 in [], "the empty list holds nothing")
}

pub fn test_membership_over_the_other_containers() -> void {
    #: The arm does not know what a container is -- it hands both operands to
    #: jaiContainsOp -- so every container the interpreter supports still works.
    let d = {"a": 1, "b": 2}
    assert_true("a" in d, "dict key")
    assert_false("z" in d, "dict miss")
    assert_true("ell" in "hello", "substring")
    assert_false("xyz" in "hello", "absent substring")
}

pub fn test_membership_throws_the_interpreters_error() -> void {
    assert_throws(TypeError, fn() -> void { let _ = 1 in 5 }, "an int contains nothing")
}

pub fn test_tuple_construction() -> void {
    var total = 0
    var call = 0
    while call < CALLS {
        #: Two elements per round, eight rounds.
        total += sum_pairs(8)
        call += 1
    }
    assert_eq(total, 16 * CALLS, "sum")
}

pub fn test_a_built_tuple_is_a_tuple() -> void {
    let p = (4, 7)
    assert_eq(len(p), 2, "len")
    assert_eq(p.get(0), 4, "first")
    assert_eq(p.get(1), 7, "second")
    assert_true(p == (4, 7), "equality is by value")
    assert_false(p == (7, 4), "and is ordered")
}
