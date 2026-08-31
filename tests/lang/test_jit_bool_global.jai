#: A module-level `bool` read from a compiled body.
#:
#: `BOOL_VAL` writes the one-byte `bool` member of the Value union and leaves
#: the other seven bytes of the payload indeterminate. OP_GET_GLOBAL's
#: compiled read loaded the payload eight bytes wide whatever its kind, so a
#: bool global came back as whatever those seven bytes happened to hold, and
#: every consumer tests the whole register against zero.
#:
#: It was a SILENT WRONG ANSWER, not a crash, and it survived because nothing
#: exercised it: `std.fmt`'s `_colors_enabled` is true, and `green()` returned
#: uncoloured text from the 65th call onward -- exactly JAI_JIT_THRESHOLD, the
#: call at which the body first compiles. It surfaced by accident when a
#: parallel test runner made the same `green()` hot enough in one process.
#: The test uses `std.fmt.green` itself rather than a fresh global of its own:
#: a bool written into a slot that has only ever held bools leaves the upper
#: payload bytes zero, so the wide load happens to give the right answer and
#: the bug hides. `_colors_enabled` sits in a slot whose bytes are dirty,
#: which is why that one shows it.
from std.fmt import colors_enabled, green
from std.test import assert_eq

#: Above JAI_JIT_THRESHOLD, so the bodies below reach the compiled tier.
let CALLS = 200

fn painted() -> int {
    #: Longer than the text it wraps exactly when colour is on.
    return green("x").len()
}

pub fn test_a_bool_global_reads_the_same_after_it_compiles() -> void {
    if not colors_enabled() { return }
    var wrapped = 0
    var call = 0
    while call < CALLS {
        if painted() > 1 { wrapped += 1 }
        call += 1
    }
    #: Was 64 -- every call up to JAI_JIT_THRESHOLD, and none after it.
    assert_eq(wrapped, CALLS, "colour survives the compile")
}
