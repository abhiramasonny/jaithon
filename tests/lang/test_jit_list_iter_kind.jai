#: `for x in xs` over a list whose elements change kind after the loop was
#: compiled.
#:
#: The compiled tier takes the loop variable's kind from one element it can see
#: at compile time, so a list that is homogeneous when it is sampled and
#: heterogeneous afterwards is the case that has to be guarded rather than
#: refused. A mixed list is declined at admission ("list element kind unknown"),
#: and an annotated `list[int]` is enforced on push, but `var xs: list = []` has
#: no element kind at all and `LANGUAGE.md` is explicit that a container reached
#: through `any` is not element-checked, so this is reachable from ordinary
#: code.
#:
#: It was not guarded. OP_FOR_ITER_BIND's list arm called out to the runtime for
#: each element and then read the payload half of the returned Value without
#: ever looking at its tag, so a `str` pushed onto a list that had been sampled
#: as int bound the ObjString's ADDRESS as an integer -- `41080394656` where the
#: interpreter raises TypeError -- and a float bound its IEEE-754 bits as an
#: integer. No crash, no diagnostic, and a raw heap pointer in program-visible
#: data. `JAITHON_JIT_DEOPT_STRESS=1` did not catch it either: the arm had no
#: guard for the stress mode to force.
#:
#: Every case below warms a loop until the tier compiles it specialised to one
#: kind and then feeds it another. The answers here are the interpreter's, so
#: `JAITHON_NO_JIT=1` proves the expectations and the plain run proves the
#: compiled code agrees. A deopt does not retire compiled code, so one summing
#: function keeps its teeth across all of them.
#:
#: total/total_f/count are the guarded loop bodies; scripts/jit_compile_check.py
#: fails if any of them stops reaching the compiled tier under JAI_JIT_WHY=1.
# jit-compiles: total, total_f, count
from std.test import assert_eq, assert_near, assert_throws

class Marker {
    pub var n: int = 7
}

#: Sums a list. `any` rather than `int` because the whole point is that an
#: element may widen the accumulator out from under the compiled kind.
fn total(xs: list) -> any {
    var acc = 0
    for x in xs { acc += x }
    return acc
}

#: Same body over a float accumulator, for the list that is sampled as float.
fn total_f(xs: list) -> any {
    var acc = 0.0
    for x in xs { acc += x }
    return acc
}

#: Never reads the element, so only the guard and the resume point decide its
#: answer. The loop variable is still bound, and still specialised to int.
fn count(xs: list) -> any {
    var seen = 0
    for _x in xs { seen += 1 }
    return seen
}

#: Long enough that the sampling tier compiles the loop. Hotness is measured in
#: SIGPROF ticks on back edges, not in iterations, so this is a duration and not
#: a count -- generous on purpose. Nothing here asserts that a compile happened:
#: the test is correct under `JAITHON_NO_JIT=1` too, it just has less to say.
fn warm_int(xs: list) -> void {
    for _r in 0..8000 { total(xs) }
}

fn warm_float(xs: list) -> void {
    for _r in 0..8000 { total_f(xs) }
}

fn warm_count(xs: list) -> void {
    for _r in 0..8000 { count(xs) }
}

fn ints() -> list {
    var xs: list = []
    for i in 0..64 { xs.push(i) }
    return xs
}

fn floats() -> list {
    var xs: list = []
    for i in 0..64 { xs.push(float(i) + 0.5) }
    return xs
}

fn test_a_str_after_ints_raises_rather_than_binding_a_pointer() -> void {
    let xs = ints()
    warm_int(xs)
    assert_eq(total(xs), 2016)
    xs.push("boom")
    #: The bug: an ObjString's address, added as an integer.
    assert_throws(TypeError, || total(xs))
    assert_throws(TypeError, || total(xs))
}

fn test_a_float_after_ints_widens_rather_than_binding_its_bits() -> void {
    let xs = ints()
    warm_int(xs)
    assert_eq(total(xs), 2016)
    xs.push(0.5)
    #: int + float is legal and widens, so a missing guard here is a wrong
    #: NUMBER and not a throw -- 4602678819172648928 is 0.5's bit pattern.
    assert_near(total(xs), 2016.5)
    assert_near(total(xs), 2016.5)
}

fn test_an_instance_after_ints_raises() -> void {
    let xs = ints()
    warm_int(xs)
    assert_eq(total(xs), 2016)
    xs.push(Marker())
    assert_throws(TypeError, || total(xs))
    assert_throws(TypeError, || total(xs))
}

fn test_a_null_after_ints_raises() -> void {
    let xs = ints()
    warm_int(xs)
    assert_eq(total(xs), 2016)
    xs.push(null)
    assert_throws(TypeError, || total(xs))
    assert_throws(TypeError, || total(xs))
}

fn test_an_int_after_floats_is_added_as_an_int() -> void {
    #: The reverse order: sampled as float, so a missing guard reads an int
    #: payload through `fmov d, x` and 7 becomes 3.5e-323, i.e. nothing.
    let xs = floats()
    warm_float(xs)
    assert_near(total_f(xs), 2048.0)
    xs.push(7)
    assert_near(total_f(xs), 2055.0)
    assert_near(total_f(xs), 2055.0)
}

fn test_the_loop_still_sees_every_element_after_a_guard_fires() -> void {
    #: A guard placed after the iterator had already advanced would SKIP an
    #: element rather than repeat it, and no sum can tell that apart from a
    #: wrong kind. Counting can. Under JAITHON_JIT_DEOPT_STRESS=1 every guard
    #: fires on every iteration, so this is the case that fails -- with a count
    #: of 0 or 1 rather than 65 -- if the resume point is wrong.
    let xs = ints()
    warm_count(xs)
    assert_eq(count(xs), 64)
    xs.push("tail")
    assert_eq(count(xs), 65)
    assert_eq(count(xs), 65)
}
