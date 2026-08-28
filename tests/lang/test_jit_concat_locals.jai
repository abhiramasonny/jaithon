#: `out = out + piece` on two locals -- string building, fused.
#:
#: The stack form of `+` has had a guarded concat arm for a while.
#: OP_ADD_LOCALS, the FUSED form, did not -- and the fused form is the one real
#: code emits. So a loop that builds a string declined WHOLE, and everything
#: else in it ran interpreted with it.
#:
#: That is what makes the number large. The concat itself is not made faster --
#: it is the same `jaiStringConcat` behind the same call -- but a loop doing
#: ordinary work with an occasional concat went from entirely interpreted to
#: compiled:
#:
#:   a lexer-shaped loop, 3M rounds   63,377,191 -> 91,272   0.15s -> 0.01s  15x
#:   a bare concat loop, 400k rounds   4,001,920 -> 22,940   0.04s -> 0.04s   --
#:
#: The second row is the honest control: with nothing else in the loop, the
#: string allocation is the whole cost and compiling the loop around it buys
#: nothing measurable. The win is the body, not the operator.
#:
#: The operands arrive as slot numbers and `emitStringConcat` wants stack
#: entries, so they are pushed first -- the cost the fusion existed to avoid,
#: and nothing beside a call that allocates. Both operands are guarded inside
#: `emitStringConcat`, so a site whose locals stop being strings deopts at this
#: instruction with both of them untouched; a site handed strings, then lists,
#: then ints answers the same as the interpreter under every stress mode.
# jit-compiles: scan
from std.test import assert_eq, assert_true

#: Above JAI_JIT_THRESHOLD entries.
let CALLS = 200

fn scan(rounds: int) -> int {
    var out = ""
    let piece = "x"
    var total = 0
    var i = 0
    while i < rounds {
        let c = i & 15
        total += c * c + (c ^ 3)
        if c == 0 { out = out + piece }
        i += 1
    }
    return total + out.len()
}

fn add2(a: any, b: any) -> any {
    return a + b
}

pub fn test_a_loop_that_builds_a_string() -> void {
    var total = 0
    var call = 0
    while call < CALLS {
        total += scan(16)
        call += 1
    }
    #: c walks 0..15 once: sum of c*c is 1240, sum of c^3 is 120, plus the one
    #: character appended when c is 0.
    assert_eq(total, (1240 + 120 + 1) * CALLS, "sum")
}

pub fn test_the_string_is_actually_built() -> void {
    assert_eq(scan(0), 0, "no rounds")
    assert_eq(scan(1), 0 + 3 + 1, "one round appends once")
    assert_eq(scan(17), 1240 + 120 + 0 + 3 + 2, "seventeen appends twice")
}

pub fn test_the_same_site_handed_other_types() -> void {
    #: Warm it on strings so the arm is the one that compiled, then hand it
    #: operands the guard has to reject.
    var call = 0
    var s = ""
    while call < CALLS {
        s = add2(s, "x")
        if s.len() > 50 { s = "" }
        call += 1
    }
    assert_true(s.len() > 0, "still building")
    assert_eq(len(add2([1], [2])), 2, "two lists concatenate")
    assert_eq(add2(3, 4), 7, "two ints add")
    assert_true(add2(1.5, 2.5) == 4.0, "two floats add")
    assert_eq(add2("a", "b"), "ab", "and strings still concatenate")
}
