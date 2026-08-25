#: `a + "x"` inside a body the tier INLINES into its caller.
#:
#: OP_ADD sits on `inlinableBody`'s whitelist because every arm it used to have
#: emitted straight-line code. The string-concat arm calls out to
#: `jaiStringConcat`, and an inlined body that calls breaks two things at once:
#: the inlined entries live in x0..x8 when the caller's bank is callee-saved
#: (`inlineOwnBank`), which is exactly the register file a call destroys, and
#: `emitDescriptor` reads its arguments through `valueBankReg`, which does not
#: know about that bank -- so the descriptor is filled from callee-saved
#: registers holding something else and the callee dereferences them.
#:
#: It is a SEGFAULT, not a wrong answer, and it took eleven lines of ordinary
#: code to reach: `concat_one` below crashed the process before the arm learned
#: to decline while `e->inlining` is set. Declining costs nothing -- the inline
#: fails and both tiers retry the whole body with inlining off, where the arm
#: fires normally -- which is why nothing here checks for speed, only for an
#: answer.
#:
#: The `any` parameter matters: it is what keeps the operand SLOT_OBJ rather
#: than something the checker has already narrowed, so the concat arm is the
#: one that fires.
# jit-compiles: concat_one, concat_ab
from std.test import assert_eq

#: Enough separate entries that the whole-function tier takes `concat_one` and
#: then inlines it into the loop below; at a few thousand it is only ever
#: interpreted and this file tests nothing.
let REPS = 400_000

#: The same total inner iterations, in rounds of 32.
let ROUNDS = 12_500

fn concat_one(a: any) -> any {
    return a + "x"
}

fn drive_concat_one(reps: int) -> int {
    var total = 0
    var i = 0
    while i < reps {
        let s = concat_one("a")
        total += s.len()
        i += 1
    }
    return total
}

fn test_a_concat_inside_an_inlined_body() -> void {
    assert_eq(drive_concat_one(REPS), REPS * 2)
}

#: The accumulating shape as well, so the operand on the left is a growing
#: string rather than a literal every time. Built in fixed rounds of 32 so the
#: answer is a whole string to compare against rather than a residue.
#: The literal on the right rather than a second parameter: `acc + piece` is
#: OP_ADD_LOCALS, a fused opcode the concat arm does not cover at all, so that
#: spelling declines before it can reach the shape under test here.
fn concat_ab(acc: any) -> any {
    return acc + "ab"
}

fn build_round() -> str {
    var acc = ""
    var i = 0
    while i < 32 {
        acc = concat_ab(acc)
        i += 1
    }
    return acc
}

fn drive_concat_onto(rounds: int) -> int {
    var whole = 0
    var r = 0
    while r < rounds {
        if build_round().len() == 64 { whole += 1 }
        r += 1
    }
    return whole
}

fn test_an_accumulating_concat_inside_an_inlined_body() -> void {
    assert_eq(drive_concat_onto(ROUNDS), ROUNDS)
    assert_eq(build_round(), "abababababababababababababababababababababababababababababababab")
}
