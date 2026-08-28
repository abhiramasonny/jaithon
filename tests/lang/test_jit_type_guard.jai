#: A declared boundary whose guard provably has nothing to do.
#:
#: `OP_TYPE_GUARD` is emitted at every declared parameter and return type. Its
#: arm settled only `float`, `int` and `bool`, so a body carrying any other
#: annotation declined WHOLE -- for a guard that, in these three cases, the
#: interpreter itself treats as a no-op:
#:
#:   - `any` -- `jaiValueMatchesType` returns true for that name before it
#:     looks at the subject at all.
#:   - `list` on a SLOT_LIST -- SLOT_LIST is a guarded fact, not a hope: every
#:     arm that puts one on the stack has already proved OBJ_LIST, because
#:     VAL_OBJ covers every heap object. `jaiTypeNameStatic` calls an ObjList
#:     "list", which is the name the guard compares against.
#:   - a class name matching the PINNED class of a SLOT_INST -- same argument,
#:     the shapeId check that pinned it is what proves it.
#:
#: An exact name match is sufficient and not necessary: the interpreter also
#: accepts a subclass and a trait implementer, so anything this does not
#: recognise still declines rather than being waved through.
#:
#: **This measures ZERO on the clock, and that is recorded rather than hidden.**
#: `check --no-cache` on resolve.jai and parser.jai, `jainum/arrayops` and
#: `jaicv/imgproc` are all flat, five to six samples each, interleaved through
#: `JAITHON_JIT_ANY_GUARD`. What it does move is coverage: **+14 compiled
#: bodies across `tests/bench`** (773 -> 787) and +6 on resolve.jai, and it
#: causes no new refusal anywhere.
#:
#: It is kept on that basis and not on a speculative one -- declining a whole
#: function over a guard that does nothing is wrong regardless of what today's
#: workloads happen to measure. The `?.` arm, which also measured zero, was
#: REVERTED because it additionally pushed a body into a whole-body decline.
#: Zero and harmless is not the same as zero and harmful.
#: NO discriminating marker, and the reason is worth writing down: the checker
#: ELIDES OP_TYPE_GUARD wherever it can prove the type statically, which in a
#: small self-contained file is everywhere -- even routing the argument through
#: an `any` did not put one back. The guards that reach this arm come from
#: boundaries the checker cannot see across, which is why they turn up in the
#: packages (`Tensor`, `Mat`, `NDArray`) and not here. So the three functions
#: below compile under either switch setting and a marker on them would prove
#: nothing.
#:
#: What this file guards is that the answers do not move -- across both switch
#: states, the interpreter and the stress modes -- and that a wrong type is
#: still rejected. The tier evidence is the +14 compiled bodies across
#: `tests/bench` quoted above.
# jit-compiles: sum_any, count_list, total_boxes
from std.test import assert_eq, assert_true, assert_throws

#: Above JAI_JIT_THRESHOLD entries, so these reach the compiled tier by a count.
let CALLS = 200

class Box {
    pub var v: int

    pub fn init(self, v: int) -> void {
        self.v = v
    }
}

#: `any` on the boundary: satisfied by everything.
fn sum_any(values: list[int], extra: any) -> int {
    var total = 0
    var index = 0
    while index < values.len() {
        total += values[index]
        index += 1
    }
    return total
}

#: `list` on a value the tier has already proved is a list.
fn count_list(values: list[int]) -> int {
    var seen = 0
    var index = 0
    while index < values.len() {
        seen += 1
        index += 1
    }
    return seen
}

#: A class name matching the pinned class.
fn total_boxes(a: Box, b: Box) -> int {
    return a.v + b.v
}

fn as_any(v: any) -> any {
    return v
}

pub fn test_an_any_boundary_is_no_constraint() -> void {
    let values = [1, 2, 3, 4]
    var total = 0
    var call = 0
    while call < CALLS {
        total += sum_any(as_any(values), "a string is fine")
        call += 1
    }
    assert_eq(total, 10 * CALLS, "sum")
    #: Every kind satisfies it, which is the whole point.
    assert_eq(sum_any(values, 7), 10, "an int")
    assert_eq(sum_any(values, null), 10, "null")
    assert_eq(sum_any(values, [1]), 10, "a list")
    assert_eq(sum_any(values, Box(1)), 10, "an instance")
}

pub fn test_a_list_boundary_on_a_proved_list() -> void {
    let values = [1, 2, 3, 4, 5]
    var total = 0
    var call = 0
    while call < CALLS {
        total += count_list(as_any(values))
        call += 1
    }
    assert_eq(total, 5 * CALLS, "count")
    assert_eq(count_list(as_any([])), 0, "the empty list")
}

pub fn test_a_class_boundary_on_a_pinned_instance() -> void {
    let a = Box(3)
    let b = Box(4)
    var total = 0
    var call = 0
    while call < CALLS {
        total += total_boxes(as_any(a), as_any(b))
        call += 1
    }
    assert_eq(total, 7 * CALLS, "sum")
}

pub fn test_a_boundary_still_rejects_the_wrong_type() -> void {
    #: Reaching the guard with the wrong type takes an `any` in between: the
    #: checker rejects `count_list(7)` outright (E0400), so from checked code
    #: the runtime guard is a backstop rather than the thing that reports the
    #: error. The arm proves the guard is a no-op or declines; it never waves a
    #: wrong value through, and the message is the interpreter's own.
    assert_throws(TypeError, || count_list(as_any(7)), "an int, not a list")
    assert_throws(TypeError, || total_boxes(Box(1), as_any(2)), "an int, not a Box")
}
