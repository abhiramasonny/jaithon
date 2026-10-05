#: An element of a list of lists is confirmed to be a list by its object type
#: before anything reads it as one, and the pointer that check loaded is the
#: element handed on. Compiled against lists, then handed a row that is not a
#: list: the check must still deoptimise and the interpreter raise.
from std.test import assert_eq, assert_throws

fn sum_firsts(rows: list[any]) -> int {
    var t = 0
    for i in 0..len(rows) { t += rows[i][0] }
    return t
}

fn rows_of(n: int) -> list[any] {
    var out: list[any] = []
    for i in 0..n { out.push([i, i + 1]) }
    return out
}

fn test_rows_are_read_through_the_checked_pointer() -> void {
    let rows = rows_of(300)
    var want = 0
    for i in 0..300 { want += i }
    for _round in 0..200 { assert_eq(sum_firsts(rows), want) }
}

fn test_a_row_that_is_not_a_list_still_raises() -> void {
    let rows = rows_of(300)
    for _round in 0..200 { let _s = sum_firsts(rows) }
    var mixed = rows_of(300)
    mixed[150] = "not a list"
    assert_throws(TypeError, || sum_firsts(mixed))
}
