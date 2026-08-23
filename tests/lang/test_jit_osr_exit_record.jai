#: A loop that leaves compiled code the ordinary way, after having called a
#: compiled function that deoptimised part way through.
#:
#: The deopt record `gDeopt` is one global, on the reasoning written above it
#: that only one body can be deoptimising at a time. That holds for a body and
#: its own stubs. It does not hold across a call: a compiled callee deopts,
#: its own return path puts it back on its feet, and its record stays in the
#: global. The OSR return path then read `nstack` after EVERY way out of the
#: region rather than only after a deopt -- so a region that had called such a
#: callee and then finished its loop normally pushed the callee's leftover
#: operand stack onto this frame's.
#:
#: What that looks like from here: `for n in sizes` holds its iterator on the
#: operand stack for the length of the loop, and the two stray values landed
#: on top of it. The next `OP_FOR_ITER_BIND` peeked one of them and the loop
#: died with `TypeError: for-loop expected an iterator, not 'int'` -- pointing
#: at the outer loop, which has nothing wrong with it.
#:
#: Reproducing it needs all three: an enclosing `for` over a list (a range
#: keeps no iterator on the stack), an inner loop the tier enters by OSR, and
#: a call inside that inner loop to something compiled that actually deopts.
#: `--gc-stress` is what makes the callee deopt reliably; without it this
#: failed about one run in twenty.
import std.math
from std.random import Random
from std.test import assert_eq, assert_true

let SIZES = [3, 4, 6, 9, 12, 16]

#: `Random.float_range` is the callee that deopts: it is compiled, and it has
#: the shape that makes the tier guard -- two cold throws and a loop around a
#: call to `next_float`.
fn test_an_ordinary_exit_does_not_inherit_a_callees_deopt() -> void {
    var seen = 0
    for n in SIZES {
        let rng = Random.with_seed(31 + n)
        var total = 0.0
        for _index in 0..n * n { total += rng.float_range(-1.0, 1.0) }
        assert_true(math.is_finite(total), f"total for {n} was {total}")
        seen += n
    }
    #: Reaching here at all is most of the test; the sum says every iteration
    #: ran rather than the loop having ended early.
    assert_eq(seen, 50)
}

#: The same shape with the values kept, so a stack left one deep would be
#: caught by a wrong answer as well as by a raised error.
fn test_the_enclosing_loop_binds_every_element() -> void {
    var got: list[int] = []
    for n in SIZES {
        let rng = Random.with_seed(7 + n)
        var total = 0.0
        for _index in 0..n * 8 { total += rng.float_range(0.0, 1.0) }
        if total > 0.0 { got.push(n) }
    }
    assert_eq(got, SIZES)
}
