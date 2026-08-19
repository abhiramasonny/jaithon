#: A `continue` must not cut the loop short for the compiled tier.
#:
#: A loop has one back edge for the ordinary path off the bottom and one more
#: for every `continue`, and a `continue` is written near the TOP of the body,
#: which is the point of it. The tier found the loop's end by taking the first
#: back edge, so the compiled region was the prefix before the `continue` and
#: everything after it ran interpreted -- which made the loop slower the LESS
#: often the `continue` fired: over three hundred thousand elements, 0.25ms
#: when every iteration took it against 7.6ms when none did, for a loop the
#: same code written with an `if` ran in 0.29ms.
#:
#: The bug was silent. What it produced was correct, only slow, so no test in
#: the tree could have noticed. These check the ANSWER, since that is what a
#: test can check: a loop whose tail is reached must still add up, whether the
#: tail was compiled or interpreted.
#:
#: No `jit-compiles:` marker, deliberately. These loops are reached through the
#: OSR tier, which is sampler-driven -- scripts/jit_compile_check.py says so at
#: length -- and which of the three is compiled varies run to run. A marker
#: here would be a coin toss. What proves the fix is the benchmark: 5.98ms to
#: 0.35ms over three hundred thousand elements.
from std.test import assert_eq

#: The tail after the `continue` is the only thing that accumulates, so a
#: compiled region that stopped at the `continue` would return zero.
fn sum_after_continue(values: list[int]) -> int {
    var total = 0
    for index in 0..values.len() {
        let value = values[index]
        if value < 0 { continue }
        total += value
        total += 1
    }
    return total
}

#: Two `continue`s, so the loop has three back edges and the last one is the
#: only correct end.
fn count_pairs(values: list[int]) -> int {
    var seen = 0
    for index in 0..values.len() {
        let value = values[index]
        if value % 2 != 0 { continue }
        if value == 0 { continue }
        seen += 1
    }
    return seen
}

#: A `continue` in the inner loop of a nested pair, which is the shape every
#: image algorithm has.
fn nested_continue(rows: int, cols: int) -> int {
    var total = 0
    for y in 0..rows {
        for x in 0..cols {
            if (x + y) % 3 == 0 { continue }
            total += x + y
        }
    }
    return total
}

fn test_a_continue_does_not_truncate_the_compiled_body() -> void {
    var values: list[int] = []
    var expected = 0
    for index in 0..40000 {
        let value = index % 7 - 2
        values.push(value)
        if value >= 0 { expected += value + 1 }
    }
    assert_eq(sum_after_continue(values), expected)
}

fn test_two_continues_leave_the_last_back_edge_as_the_end() -> void {
    var values: list[int] = []
    var expected = 0
    for index in 0..300000 {
        let value = index % 5
        values.push(value)
        if value % 2 == 0 and value != 0 { expected += 1 }
    }
    assert_eq(count_pairs(values), expected)
}

fn test_a_continue_in_a_nested_loop() -> void {
    var expected = 0
    for y in 0..600 {
        for x in 0..600 {
            if (x + y) % 3 == 0 { continue }
            expected += x + y
        }
    }
    assert_eq(nested_continue(600, 600), expected)
}
