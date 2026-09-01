#: Reading a dynamic local inside an OSR loop, and what the tag says about it.
#:
#: A slot two paths disagree about is made `dynamic`: it keeps its frame home,
#: carries a run-time tag beside the payload, and the walk stops refusing the
#: clash. What pays for that is the read -- the compile-time kind of a dynamic
#: slot is a speculation, and every read is supposed to check the tag before
#: trusting it. The function tier did. The OSR tier took the widening and read
#: the slot as a bare payload word, so the kind was never confirmed against
#: anything.
#:
#: Nothing in the frame is corrupted by that; the write side always stored the
#: right tag. What goes wrong is the operand-stack entry the read pushes. It is
#: stamped with the compile-time kind, and when the region ends at an unarmed
#: opcode the deopt stub rebuilds that entry's tag from the kind -- writing a
#: non-null tag over a null payload, or VAL_OBJ over an integer. The
#: interpreter then resumes holding a value whose tag is a lie.
#:
#: Three ingredients, and dropping any one of them stops the case reproducing:
#:
#:   * a local given two kinds inside the loop, one of them null-bearing, which
#:     is what drives the retry to `dynamic` in the first place;
#:   * the two binds ordered so the kind the walk keeps -- last writer in
#:     bytecode order, since a dynamic slot gets no join at the merge -- is the
#:     wrong one on the path that skips the `if`;
#:   * a read of that local inside the compiled region followed by an unarmed
#:     opcode, so the region ends at a deopt whose record has to synthesise the
#:     tag. `??` supplies that for free: OP_JUMP_IF_NULL has no arm.
#:
#: Both functions are called once with a long loop, which is what puts the work
#: in the OSR tier rather than the whole-function one: the function tier
#: compiles on call count and never sees a body called this rarely, so under
#: JAITHON_JIT_THRESHOLD=1 these pass whatever the OSR tier does.

#: The null-ness is what is lost. `m` is bound null at the top of the body and
#: 32640 inside the `if`, so the kind the walk keeps is int; on the iterations
#: where the `if` does not fire the slot really holds null, and reading it as an
#: int made `?? 100` hand back the payload 0 instead of the default.
fn defaulted(n: int) -> int {
    var total = 0
    var i = 0
    while i < n {
        var w = 0
        while w < 6 {
            w = w + 1
            var m: int? = null
            if w < 4 { m = 32640 }
            total = total + (m ?? 100)
        }
        i = i + 1
    }
    return total
}

#: The same slot with the binds the other way round, which is the more dangerous
#: half: the kind the walk keeps is the null-bearing one, so the tag it rebuilds
#: is computed from the payload and an integer 32640 comes back tagged as an
#: object. `??` then treats it as present and the value is read as a pointer.
fn reversed(n: int) -> int {
    var total = 0
    var i = 0
    while i < n {
        var w = 0
        while w < 6 {
            w = w + 1
            var m: int? = 32640
            if w < 3 { m = null }
            total = total + (m ?? 100)
        }
        i = i + 1
    }
    return total
}

fn main() -> int {
    print(defaulted(40000))
    print(reversed(40000))
    return 0
}
