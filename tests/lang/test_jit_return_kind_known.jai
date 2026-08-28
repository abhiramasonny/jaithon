#: A callee whose walk never reached an OP_RETURN must not be a direct-call
#: target.
#:
#: `ObjFunction::jitReturnKind` is stored unconditionally at the end of a
#: compile, so a body that compiled a PREFIX and took the unarmed path before
#: any return leaves it at the SLOT_INT a zeroed `Emit` starts on -- and that is
#: indistinguishable from a proven int. `emitGlobalCall` already knew this and
#: asks the interpreter's own observation first. `emitDirectCall` did not, and
#: it is the site where it matters most: it branches straight in and takes the
#: callee's raw x0 with no tag guard, so there the kind is a CONTRACT rather
#: than a prediction.
#:
#: The shape that found it, in the lexer, was a chain three links long:
#:
#:   1. `_is_ident_start` walks only to OP_GET_GLOBAL at offset 0 -- a cold
#:      `throw` on its first instruction -- so it compiles nothing and still
#:      advertises "returns int".
#:   2. `_is_ident_cont`, which is `_is_ident_start(c) or _is_digit(c)`, reads
#:      that and declines: "a branch on a int, not a bool".
#:   3. `_ident_run_end`'s OSR loop then retried `_is_ident_cont` EIGHTY times,
#:      waiting for a function that could never compile.
#:
#: Refusing the direct call breaks the chain: the caller falls back to the
#: guarded `emitGlobalCall`, which asks the observation and gets `bool`.
#: `_is_ident_cont` compiles, and `_ident_run_end` goes from 80 blocked
#: attempts to 0. On `check --no-cache lib/jaithon/compile/parser.jai`,
#: interleaved through `JAITHON_JIT_RETURN_KNOWN` and repeated: **0.565s ->
#: 0.545s**, and the bench corpus compiles three more bodies (769 -> 772).
#:
#: NO `# jit-compiles:` MARKER, deliberately, and it is worth saying why rather
#: than leaving one that lies. Reproducing the shape needs the CALLEE to stop
#: before its return while the CALLER is compiled afterwards, and which of the
#: two the tier reaches first is a property of the run, not of the source: in
#: this file it is `decide` that stops at OP_GET_GLOBAL at 0, waiting for
#: `stops_early`, and both then compile under either switch setting. A marker
#: here would pass with the fix reverted and prove nothing.
#:
#: What this file guards is that the answers do not move -- across both switch
#: states, the interpreter and the stress modes. The tier evidence is the
#: decline counts and the clock above, on `lib/jaithon/compile/parser.jai`,
#: where the ordering does arise.
from std.test import assert_eq, assert_true, assert_false, assert_throws

#: Above JAI_JIT_THRESHOLD entries, so the bodies below reach the compiled tier.
let CALLS = 200

#: The first instruction reads a global that only the cold branch needs, which
#: is what makes the walk stop at offset 0 -- the same shape as an argument
#: check that throws.
fn stops_early(c: int) -> bool {
    if c < 0 { throw ValueError("negative") }
    return c > 10
}

fn also_bool(c: int) -> bool {
    return c & 1 == 0
}

#: `or` branches on the first call's result. If the tier believes that result
#: is an int, this whole body declines.
fn decide(c: int) -> bool {
    return stops_early(c) or also_bool(c)
}

pub fn test_the_caller_still_answers() -> void {
    var seen = 0
    var call = 0
    while call < CALLS {
        if decide(12) { seen += 1 }
        call += 1
    }
    assert_eq(seen, CALLS, "12 is over ten")
}

pub fn test_both_arms_of_the_or() -> void {
    assert_true(decide(12), "the first arm")
    assert_true(decide(4), "the second arm, 4 being even")
    assert_false(decide(5), "neither: 5 is under ten and odd")
    assert_true(decide(11), "the first arm again")
}

pub fn test_the_cold_branch_still_throws() -> void {
    assert_throws(ValueError, || decide(-1), "the walk-stopping branch")
}
