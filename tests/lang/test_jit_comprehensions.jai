#: List comprehensions in a loop hot enough to reach the compiled tier.
#:
#: Every spelling of a comprehension declined, and the chain took three fixes,
#: none of which measures anything on its own -- a refusal is per-body, so
#: clearing one link only moves the body to the next.
#:
#: 1. `[expr for x in [a, b, c]]` iterates a list the body BUILT, which has no
#:    live sample to read an element kind off. `OP_BUILD_LIST` now records an
#:    exemplar of what went into it (Emit::stackElem): a synthesised zero for a
#:    scalar, the element's own sample for an object.
#: 2. The loop variable and the comprehension's RESULT share one local slot --
#:    the front end reuses it, since the loop variable is dead by the bind. So
#:    the loop tier, which seeds each slot with what it holds at entry, saw a
#:    list from the previous iteration and refused to bind an int into it.
#: 3. That refusal is not fatal: a slot given two kinds asks to carry its tag
#:    and the walk runs again with that decided (`needDynamic`). The function
#:    tier had honoured that request since it grew one; the loop tier only ever
#:    honoured the nullable half. Threading the dynamic half through is what
#:    made the whole chain pay.
#:
#: `[0 for _i in 0..8]` twenty million times: 2.54s before, 0.67s after
#: (**3.8x**, five runs each). Interpreted instructions, one million rounds:
#: `[x * 2 for x in [1, 2, 3]]` 27,000,320 to 2,220,393; `[0 for _i in 0..8]`
#: 45,283,921 to 10,716,483; `[x * 2 for x in src]` 28,312,358 to 3,461,012.
# jit-compiles: over_a_literal, over_a_range, over_a_variable, of_strings
from std.test import assert_eq

let CALLS = 100

fn over_a_literal(rounds: int) -> int {
    var total = 0
    var i = 0
    while i < rounds {
        let ys = [x * 2 for x in [1, 2, 3]]
        total += ys.len()
        i += 1
    }
    return total
}

fn over_a_range(rounds: int) -> int {
    var total = 0
    var i = 0
    while i < rounds {
        let ys = [0 for _i in 0..8]
        total += ys.len()
        i += 1
    }
    return total
}

fn over_a_variable(src: list[int], rounds: int) -> int {
    var total = 0
    var i = 0
    while i < rounds {
        let ys = [x * 2 for x in src]
        total += ys.len()
        i += 1
    }
    return total
}

#: An exemplar that is an object rather than a synthesised scalar: the elements'
#: own sample is carried, and the guard at each read is what makes it sound.
fn of_strings(rounds: int) -> int {
    var total = 0
    var i = 0
    while i < rounds {
        let ys = [s for s in ["ab", "cde"]]
        total += ys.len()
        i += 1
    }
    return total
}

pub fn test_over_a_literal_list() -> void {
    var total = 0
    var call = 0
    while call < CALLS {
        total += over_a_literal(10)
        call += 1
    }
    assert_eq(total, 3 * 10 * CALLS)
}

pub fn test_over_a_range() -> void {
    var total = 0
    var call = 0
    while call < CALLS {
        total += over_a_range(10)
        call += 1
    }
    assert_eq(total, 8 * 10 * CALLS)
}

pub fn test_over_a_variable() -> void {
    let src = [5, 6, 7, 8]
    var total = 0
    var call = 0
    while call < CALLS {
        total += over_a_variable(src, 10)
        call += 1
    }
    assert_eq(total, 4 * 10 * CALLS)
}

pub fn test_of_strings() -> void {
    var total = 0
    var call = 0
    while call < CALLS {
        total += of_strings(10)
        call += 1
    }
    assert_eq(total, 2 * 10 * CALLS)
}

#: What the four above deliberately do NOT do: subscript the result. A list the
#: body built has no live sample either, so `ys[2]` declines at
#: "OP_GET_INDEX: no live list to read an element kind off" -- the next link in
#: this chain, and not one this change reaches. Kept as a correctness check of
#: what the comprehensions actually produce, which the length assertions above
#: cannot see.
pub fn test_the_elements_are_right() -> void {
    assert_eq([x * 2 for x in [1, 2, 3]], [2, 4, 6])
    assert_eq([0 for _i in 0..8], [0, 0, 0, 0, 0, 0, 0, 0])
    assert_eq([x * 2 for x in [5, 6, 7, 8]], [10, 12, 14, 16])
    assert_eq([s for s in ["ab", "cde"]], ["ab", "cde"])
}

pub fn test_a_mixed_source_still_answers() -> void {
    #: The exemplar walk refuses to predict when the elements disagree, so the
    #: body is declined and the interpreter answers. Here to prove the refusal
    #: is a refusal and not a wrong prediction.
    var total = 0
    var call = 0
    while call < CALLS {
        #: Not a numeric mix: `[1, "ab", 3.5]` unifies its numbers, so the
        #: int arrives as 1.0 and the lengths stop saying what they look like.
        let ys = [str(x).len() for x in [1, "ab", true]]
        total += ys[0] + ys[1] + ys[2]
        call += 1
    }
    assert_eq(total, (1 + 2 + 4) * CALLS)
}
