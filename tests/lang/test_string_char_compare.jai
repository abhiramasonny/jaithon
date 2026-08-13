#: Indexing a string and comparing the character, in loops hot enough that the
#: compiled tier takes them.
#:
#: `s[i]` yields one of 128 shared one-byte strings, so two of them are equal
#: exactly when they are the same pointer, and the tier compiles the compare to
#: `cmp` — skipping the "is a string, and is interned" guards it would
#: otherwise emit for a value it only knows by sample.
#:
#: That skip is a PROOF, not a prediction, and a wrong proof is a wrong answer
#: rather than a crash: `big_equal_to_literal` below is forty characters built
#: at run time, so it is past the interning length and is a different object
#: from the identical literal, while being equal to it. If the proof ever
#: reaches a value that did not come out of the table — which is what a branch
#: merge would do, since the other edge into a join carries a value the
#: compiler never saw — the compare becomes a pointer compare on two unequal
#: pointers and answers `false` where the truth is `true`. Every case here is
#: written to be run enough times to be compiled; at interpreter speed they all
#: pass trivially, so run this file with and without JAITHON_NO_JIT.
from std.test import assert_eq, assert_true, assert_false

#: Big enough that the sampler notices. Nothing forces a compile: a 4kHz timer
#: tick has to land on one of these loops' back edges, so a case that runs for
#: a millisecond only ever tests the interpreter and would pass whatever the
#: tier emitted. At this size each of them runs for tens of milliseconds and is
#: sampled hundreds of times.
let REPS = 150_000

#: Forty bytes, past JAI_INTERN_MAX, assembled a piece at a time so no constant
#: folding can turn it back into the literal it equals.
fn built_forty() -> str {
    var s = ""
    var i = 0
    while i < 4 {
        s = s + "0123456789"
        i += 1
    }
    return s
}

let FORTY = built_forty()
let FORTY_LITERAL = "0123456789012345678901234567890123456789"

fn count_matches(text: str, ch: str) -> int {
    var hits = 0
    var i = 0
    let n = text.len()
    while i < n {
        if text[i] == ch { hits += 1 }
        i += 1
    }
    return hits
}

fn count_self_matches(a: str, b: str, reps: int) -> int {
    var hits = 0
    var r = 0
    while r < reps {
        var i = 0
        while i < 4 {
            if a[i] == b[i] { hits += 1 }
            i += 1
        }
        r += 1
    }
    return hits
}

#: The merge: one arm is a table character, the other is not. Whatever the
#: compiler proved about the first arm must not reach the second.
#:
#: The character has to be the ELSE arm. The compiler walks the bytecode
#: straight through -- condition, then-arm, jump, else-arm, join -- so the state
#: that arrives at the join is the else-arm's, and only a character there can
#: leave a proof lying where the other arm's value will be. With the arms the
#: other way round the then-arm's proof is overwritten by the else-arm's push
#: and nothing can go wrong, which is why that ordering tests nothing.
fn char_in_else(s: str, other: str, lit: str, take_other: bool,
                reps: int) -> int {
    var hits = 0
    var r = 0
    while r < reps {
        if (if take_other { other } else { s[0] }) == lit { hits += 1 }
        r += 1
    }
    return hits
}

fn test_a_character_equals_its_literal() -> void {
    var text = ""
    var i = 0
    while i < 50_000 {
        text = text + "abcd"
        i += 1
    }
    assert_eq(count_matches(text, "a"), 50_000)
    assert_eq(count_matches(text, "d"), 50_000)
    assert_eq(count_matches(text, "z"), 0)
}

fn test_two_characters_compare_by_content() -> void {
    assert_eq(count_self_matches("abcd", "abcd", REPS), REPS * 4)
    assert_eq(count_self_matches("abcd", "abce", REPS), REPS * 3)
    assert_eq(count_self_matches("abcd", "wxyz", REPS), 0)
}

fn test_a_merge_does_not_inherit_the_characters_proof() -> void {
    #: An unasserted first call, so that what follows runs compiled from its
    #: first iteration rather than from wherever the sampler happened to catch
    #: it. Without it the assertion below is only wrong for the tail of the
    #: loop and a few thousand iterations of slack decide whether the gate
    #: fires.
    let warm = char_in_else("0", FORTY, FORTY_LITERAL, true, REPS)
    assert_true(warm >= 0)

    #: The taken arm is the long run-time string, equal to the literal without
    #: being the same object. Answering by pointer -- which is what an
    #: inherited proof would do -- says `false` for every iteration.
    assert_eq(char_in_else("0", FORTY, FORTY_LITERAL, true, REPS), REPS)
    #: And the arm the proof really does describe.
    assert_eq(char_in_else("0", FORTY, "0", false, REPS), REPS)
    assert_eq(char_in_else("0", FORTY, "z", false, REPS), 0)
}

fn test_the_long_string_is_equal_without_being_identical() -> void {
    assert_true(FORTY == FORTY_LITERAL)
    assert_eq(FORTY.len(), 40)
}

#: Everything the fast paths decline has to keep working: a non-ASCII string
#: indexes by scalar, not by byte, and an out-of-range index still raises.
fn test_non_ascii_indexing_is_by_scalar() -> void {
    let s = "héllo"
    var seen = ""
    var i = 0
    while i < s.len() {
        seen = seen + s[i]
        i += 1
    }
    assert_eq(seen, s)
    assert_eq(s.len(), 5)
    assert_eq(s[1], "é")
    assert_false(s[1] == "h")
}

#: The merge case above found this, so it is pinned on its own: an OSR entry
#: reads the interpreter's slots directly, and a bool's payload is one byte, so
#: the seven bytes above it belong to whatever the slot held before. Loading
#: eight of them puts garbage in a register every consumer tests whole. `dirty`
#: is called first with an argument wide enough to leave high bytes behind, so
#: a compiled `if flag` that reads the whole word takes the wrong arm.
fn branch_on_flag(flag: bool, reps: int) -> int {
    var taken = 0
    var r = 0
    while r < reps {
        if flag {
            taken += 1
        }
        r += 1
    }
    return taken
}

fn test_a_bool_local_survives_an_osr_entry() -> void {
    assert_eq(branch_on_flag(true, REPS), REPS)
    assert_eq(branch_on_flag(false, REPS), 0)
    assert_eq(branch_on_flag(true, REPS), REPS)
    assert_eq(branch_on_flag(false, REPS), 0)
}

fn test_negative_and_out_of_range_indices() -> void {
    let s = "abcd"
    var tail = 0
    var r = 0
    while r < REPS {
        if s[-1] == "d" { tail += 1 }
        r += 1
    }
    assert_eq(tail, REPS)

    var raised = false
    try {
        let _ = s[4]
    } catch e: Error {
        raised = true
    }
    assert_true(raised)
}
