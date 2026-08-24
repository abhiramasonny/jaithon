#: Numbers are sorted by their digits rather than against each other, which
#: is a different implementation from the comparison merge everything else
#: uses. It only runs above a length floor, so the sizes here straddle it,
#: and the cases are the ones a bit pattern gets wrong: negatives, zero's two
#: spellings, the infinities, and whole numbers past a double's exact range.
from std.math import INF, NAN
from std.test import assert_eq, assert_true

let FLOOR = 128

#: Long enough to take the digit sort, short enough to stay readable.
fn ramp(n: int, span: int) -> list[int] {
    var values: list[int] = []
    for index in 0..n { values.push(index * 7919 % span) }
    return values
}

fn is_ordered(values: list[float], reverse: bool) -> bool {
    for index in 1..values.len() {
        let before = values[index - 1]
        let after = values[index]
        if reverse and before < after { return false }
        if not reverse and before > after { return false }
    }
    return true
}

fn test_lengths_either_side_of_the_digit_sort() -> void {
    for n in [0, 1, 2, 3, 17, FLOOR - 1, FLOOR, FLOOR + 1, 5000] {
        let values = ramp(n, 101)
        let up = values.sorted()
        let down = values.sorted(reverse: true)
        assert_eq(up.len(), n)
        assert_eq(down.len(), n)
        for index in 1..n {
            assert_true(up[index - 1] <= up[index], f"ascending at {n}")
            assert_true(down[index - 1] >= down[index], f"descending at {n}")
        }
        assert_eq(down, up.reversed())
    }
}

fn test_negatives_and_zero_and_the_infinities() -> void {
    let big = INF
    var values: list[float] = [big, -big, 0.0, -0.0, 1.0, -1.0, 1.0e-300, -1.0e-300]
    for index in 0..FLOOR { values.push(float(index % 17) - 8.0) }
    assert_true(is_ordered(values.sorted(), false), "ascending")
    assert_true(is_ordered(values.sorted(reverse: true), true), "descending")
    assert_eq(values.sorted()[0], -big)
    assert_eq(values.sorted(reverse: true)[0], big)
}

#: Zero and negative zero compare equal, so they have to come back in the
#: order they were given -- a bit pattern alone would separate them.
fn test_the_two_spellings_of_zero_keep_their_order() -> void {
    var values: list[tuple[float, int]] = []
    for index in 0..FLOOR * 2 {
        values.push((if index % 2 == 0 { 0.0 } else { -0.0 }, index))
    }
    let ordered = values.sorted(key: |pair| pair[0])
    for index in 0..ordered.len() { assert_eq(ordered[index][1], index) }
}

fn test_equal_keys_keep_the_order_they_arrived_in() -> void {
    var values: list[tuple[int, int]] = []
    for index in 0..FLOOR * 3 { values.push((index % 7, index)) }
    let ordered = values.sorted(key: |pair| pair[0])
    var last = -1
    var previous = -1
    for pair in ordered {
        if pair[0] != last {
            last = pair[0]
            previous = -1
        }
        assert_true(pair[1] > previous, f"key {pair[0]} came back out of order")
        previous = pair[1]
    }
}

#: Past a double's exact range the digits of a converted key collide, so
#: these have to go the long way round and still come back exact.
fn test_whole_numbers_past_a_doubles_exact_range() -> void {
    let base = 9007199254740992
    var values: list[int] = []
    for index in 0..FLOOR { values.push(base + index * 7919 % 1000) }
    let ordered = values.sorted()
    for index in 1..ordered.len() {
        assert_true(ordered[index - 1] <= ordered[index], "exactness was lost")
    }
    assert_eq(ordered.len(), FLOOR)
}

#: Ints and floats mixed in one list order against each other, not in blocks.
fn test_ints_and_floats_in_the_same_list() -> void {
    var values: list[float] = []
    for index in 0..FLOOR { values.push(if index % 2 == 0 { index } else { float(index) + 0.5 }) }
    assert_true(is_ordered(values.sorted(), false), "mixed kinds")
}

#: A NaN has no order, and the digit sort has to hand that back rather than
#: inventing one.
fn test_a_nan_key_is_still_refused() -> void {
    var values: list[float] = []
    for index in 0..FLOOR { values.push(float(index)) }
    values.push(NAN)
    var raised = false
    try {
        let _ordered = values.sorted()
    } catch _error {
        raised = true
    }
    assert_true(raised, "a NaN key should have been refused")
}

#: Anything that is not a number still goes through the comparison path.
fn test_other_kinds_still_sort() -> void {
    var words: list[str] = []
    for index in 0..FLOOR { words.push(f"{(index * 7919) % 997:04}") }
    let ordered = words.sorted()
    for index in 1..ordered.len() {
        assert_true(ordered[index - 1] <= ordered[index], "text")
    }
}
