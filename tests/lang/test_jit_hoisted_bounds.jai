#: A subscript whose index is the loop variable plus a constant does not need
#: its own bounds check: the range head already says what the variable will
#: reach, so one compare above the loop covers every element.
#:
#: The five-point stencil below is the shape it was built for -- `xs[j]`,
#: `xs[j - 1]`, `xs[j + 1]` over three rows and a write to a fourth -- and it
#: went from ten instructions of checking per cell to none, worth 1.30x.
#:
#: What these pin is the part that is easy to get wrong. The guard is only
#: sound while the loop variable really is the counter, the list really is the
#: one the head measured, and the range really does fit -- so:
#:
#:   - a range that runs PAST the end of the list must still raise IndexError,
#:     with the same index in the message the interpreter would report, rather
#:     than reading off the end of the array;
#:   - a body that ASSIGNS the loop variable indexes with a value no head ever
#:     bounded, and must keep its own check;
#:   - a negative index must still normalise, since the hoisted form skips the
#:     normalisation too;
#:   - and every answer must equal the interpreter's.
#:
#: Run with JAITHON_NO_JIT=1 as well: none of these may move.
# jit-compiles: stencil_sum, walk_forward, walk_reassigned
from std.test import assert_eq, assert_throws

let ROUNDS = 300

#: Above JAI_JIT_THRESHOLD entries, so the bodies the marker names reach the
#: compiled tier by a count rather than by a timer.
let WARM_CALLS = 200

fn build(count: int) -> list[int] {
    var values: list[int] = []
    for index in 0..count { values.push(index * 3 % 17) }
    return values
}

fn stencil_sum(row: list[int], count: int, rounds: int) -> int {
    var total = 0
    var round = 0
    while round < rounds {
        for index in 1..count - 1 {
            total = total + row[index] + row[index - 1] + row[index + 1]
        }
        round += 1
    }
    return total
}

fn walk_forward(values: list[int], count: int, rounds: int) -> int {
    var total = 0
    var round = 0
    while round < rounds {
        for index in 0..count { total = total + values[index] }
        round += 1
    }
    return total
}

fn walk_reassigned(values: list[int], count: int) -> int {
    #: The loop variable is not the index here -- `pick` is -- so nothing the
    #: head knows about `index` bounds what this reads.
    var total = 0
    for index in 0..count {
        var pick = index
        if index > 3 { pick = 1 }
        total = total + values[pick]
    }
    return total
}

pub fn test_stencil_matches_the_long_way_round() -> void {
    let row = build(64)
    #: Every interior cell counted by hand: element j appears once as itself,
    #: once as its left neighbour's right, and once as its right neighbour's
    #: left, except at the two ends.
    var expected = 0
    for index in 1..63 {
        expected = expected + row[index] + row[index - 1] + row[index + 1]
    }
    #: Entered WARM_CALLS times for the same reason as walk_forward below: the
    #: marker names the function tier, and the function tier counts entries.
    var call = 0
    while call < WARM_CALLS {
        assert_eq(stencil_sum(row, 64, 1), expected)
        call += 1
    }
    assert_eq(stencil_sum(row, 64, ROUNDS), expected * ROUNDS)
}

pub fn test_a_range_past_the_end_still_raises() -> void {
    let values = build(40)
    #: Hot first, so the loop is compiled with a head guard measured against a
    #: list of forty, and only then handed a range of sixty.
    var expected = 0
    for index in 0..40 { expected = expected + values[index] }
    #: ENTERED many times, not merely looped many times. `walk_forward` is
    #: called three times in this file, so only the OSR tier could ever compile
    #: it -- and OSR fires on a 250us sampling timer, which twelve thousand fast
    #: iterations do not trip on a quiet machine. The marker above therefore
    #: asserted something timing-dependent and had silently stopped holding.
    #: The function tier compiles at a COUNT of entries, so this is stable.
    var call = 0
    while call < WARM_CALLS {
        assert_eq(walk_forward(values, 40, 1), expected)
        call += 1
    }
    assert_eq(walk_forward(values, 40, ROUNDS), expected * ROUNDS)
    assert_throws(IndexError, fn() {
        walk_forward(values, 60, 1)
    }, "a range reaching past the list must raise, not read past the end")
    #: And the index in the message is the one the program asked for.
    try {
        walk_forward(values, 60, 1)
    } catch error: IndexError {
        assert_eq(error.message, "list index 40 out of range for length 40")
    }
}

pub fn test_an_index_the_head_never_bounded() -> void {
    let values = build(40)
    var expected = 0
    for index in 0..40 {
        var pick = index
        if index > 3 { pick = 1 }
        expected = expected + values[pick]
    }
    var round = 0
    while round < ROUNDS {
        assert_eq(walk_reassigned(values, 40), expected)
        round += 1
    }
}

pub fn test_negative_indices_still_normalise() -> void {
    let values = build(40)
    var total = 0
    var round = 0
    while round < ROUNDS {
        total = values[-1] + values[-40] + values[0] + values[39]
        round += 1
    }
    assert_eq(total, values[39] + values[0] + values[0] + values[39])
    assert_throws(IndexError, fn() {
        let _ = values[-41]
    }, "a negative index past the front still raises")
}
