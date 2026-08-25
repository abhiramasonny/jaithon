#: `x is null` in a loop hot enough to reach the compiled tier.
#:
#: The construct spent a long time declining the whole enclosing body for a
#: reason that reads right and is not: the arm asked for SLOT_NULL, which is
#: what a `-> void` call leaves behind, while the `null` written in source is
#: OP_NULL and pushes SLOT_MAYBE_INST -- a null literal and an instance have to
#: agree on a kind or `var x: Box? = null` gives its local two of them. So the
#: kind alone cannot tell `x is null` from `x is y`, and `stackNullLit` carries
#: the half it cannot.
#:
#: That flag is a PROOF, not a prediction: it deletes the general `is` rather
#: than guarding it, so it must not survive a branch join, where the other edge
#: carries a value this walk never saw. `null_or_class_in_else` below is that
#: case, written the way the equality merge in test_string_char_compare.jai is
#: -- the literal in the ELSE arm, because the walk's state at the join is the
#: else arm's.
# jit-compiles: tally_null_links, tally_live_links
from std.test import assert_eq, assert_true

class Cell {
    pub var v: int

    pub fn init(self, v: int) -> void {
        self.v = v
    }
}

#: Sparse on purpose. A list dense in nulls is refused at OSR entry (the
#: element guard bails per null), and the refusal is a density test -- one in
#: eight is well past it, one in a thousand well under, and both have to give
#: the interpreter's answer.
fn build(n: int, every: int) -> list[Cell?] {
    var xs: list[Cell?] = []
    var i = 0
    while i < n {
        if every > 0 and i % every == 0 {
            xs.push(null)
        } else {
            xs.push(Cell(i))
        }
        i += 1
    }
    return xs
}

fn tally_null_links(xs: list[Cell?]) -> int {
    var t = 0
    for x in xs {
        if x is null { t += 1 }
    }
    return t
}

fn tally_live_links(xs: list[Cell?]) -> int {
    var t = 0
    for x in xs {
        if x is not null { t += 1 }
    }
    return t
}

fn test_a_sparse_list_counts_both_ways() -> void {
    let xs = build(200_000, 1000)
    var r = 0
    while r < 6 {
        assert_eq(tally_null_links(xs), 200)
        assert_eq(tally_live_links(xs), 199_800)
        r += 1
    }
}

fn test_a_dense_list_counts_both_ways() -> void {
    let xs = build(200_000, 8)
    var r = 0
    while r < 6 {
        assert_eq(tally_null_links(xs), 25_000)
        assert_eq(tally_live_links(xs), 175_000)
        r += 1
    }
}

fn test_a_list_with_no_nulls_counts_both_ways() -> void {
    let xs = build(200_000, 0)
    var r = 0
    while r < 6 {
        assert_eq(tally_null_links(xs), 0)
        assert_eq(tally_live_links(xs), 200_000)
        r += 1
    }
}

#: The merge, and it has to be on the RIGHT of `is`. The proof says "this
#: operand is the null literal", and the arm reads it only to decide whether to
#: fire at all -- the answer it then emits is a test on the SUBJECT. So a proof
#: wrongly surviving on the subject's side changes nothing, and only a proof
#: wrongly surviving on the right side can turn `a is b` into `a is null`.
#:
#: The literal goes in the ELSE arm for the reason the equality merge in
#: test_string_char_compare.jai gives: the walk is linear -- condition, then
#: arm, jump, else arm, join -- so the state arriving at the join is the else
#: arm's, and only a literal there leaves a proof where the other arm's value
#: will be.
#:
#: With the subject null and the taken arm a live Cell, the two answers differ:
#: `null is cell` is false, while the `is null` the stale proof would compile
#: is true for every iteration.
#:
#: HONEST LIMIT: this body does not reach the compiled tier today. Its loop
#: head is `OP_GET_LOCAL2`, which the tier has no arm for, so it stops there
#: before it ever walks as far as the join -- `JAI_JIT_WHY=1` says
#: `osr at 21 stopped: OP_GET_LOCAL2`. The bytecode IS the hazard shape
#: (`0041 OP_JUMP -> 0045 / 0044 OP_NULL / 0045 OP_IS`, so the join is a branch
#: target), and the arm is written for it, but nothing here can currently
#: exercise it. It is left in as the net for the day OP_GET_LOCAL2 compiles,
#: and deliberately absent from the jit-compiles marker above so the gate does
#: not claim teeth this does not have.
fn null_is_merged(subject: Cell?, other: Cell, take_other: bool, reps: int) -> int {
    var hits = 0
    var r = 0
    while r < reps {
        if subject is (if take_other { other } else { null }) { hits += 1 }
        r += 1
    }
    return hits
}

fn test_a_merge_does_not_inherit_the_null_literals_proof() -> void {
    let cell = Cell(7)
    #: Warmed the same way as the other merge tests: the assertion is only
    #: wrong once the loop is compiled, so it has to run compiled from its
    #: first iteration rather than from wherever the sampler caught it.
    var w = 0
    while w < 8 {
        let warm = null_is_merged(null, cell, true, 150_000)
        assert_true(warm >= 0)
        w += 1
    }
    assert_eq(null_is_merged(null, cell, true, 150_000), 0)
    assert_eq(null_is_merged(null, cell, false, 150_000), 150_000)
    assert_eq(null_is_merged(cell, cell, true, 150_000), 150_000)
}
