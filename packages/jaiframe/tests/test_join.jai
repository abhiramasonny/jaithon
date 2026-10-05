#: Joins, checked against the host's definition of the join rather than
#: against numbers this module produced.
#:
#: The sizes matter here the way they do for the key sorts: the radix passes a
#: join orders its buckets and its output pairs with only stitch blocks
#: together past one threadgroup, and only switch to the parallel offsets scan
#: past 128 blocks, so the cases are 70000 rows.
import std.gpu as gpu
from std.test import assert_eq, assert_true
from jaiframe.column import Column, DataFrame
from jaiframe.join import How, merge

fn keyed_frames(rows: int) -> (DataFrame, DataFrame) {
    var keys: list[float] = []
    var values: list[float] = []
    for at in 0..rows {
        keys.push(float(at * 37 % 500))
        values.push(float(at))
    }
    var right_keys: list[float] = []
    var tags: list[float] = []
    for key in 0..400 {
        right_keys.push(float(399 - key))
        tags.push(float((399 - key) * 2 + 1))
    }
    let left = DataFrame(["key", "value"], [Column.from_floats(keys), Column.from_floats(values)])
    let right = DataFrame(["key", "tag"], [Column.from_floats(right_keys), Column.from_floats(tags)])
    return (left, right)
}

#: An inner join over a right side with unique keys keeps exactly the left
#: rows that find a partner, in left order, each with its partner's tag.
fn test_an_inner_join_keeps_the_matching_left_rows_in_order() -> void {
    if not gpu.is_available() { return }
    let rows = 70000
    let (left, right) = keyed_frames(rows)
    defer { left.free() }
    defer { right.free() }
    let out = merge(left, right, on: "key")
    defer { out.free() }
    var expected: list[float] = []
    for at in 0..rows {
        if at * 37 % 500 < 400 { expected.push(float(at)) }
    }
    assert_eq(out.len(), expected.len())
    let got = out.column("value").to_floats()
    let keys = out.column("key").to_floats()
    let tags = out.column("tag").to_floats()
    var wrong = 0
    for at in 0..expected.len() {
        if got[at] != expected[at] { wrong += 1 }
        if tags[at] != keys[at] * 2.0 + 1.0 { wrong += 1 }
    }
    assert_eq(wrong, 0)
}

#: A left join keeps every left row in order and leaves the tag missing where
#: the key has no partner.
fn test_a_left_join_keeps_every_left_row() -> void {
    if not gpu.is_available() { return }
    let rows = 70000
    let (left, right) = keyed_frames(rows)
    defer { left.free() }
    defer { right.free() }
    let out = merge(left, right, how: How.Left, on: "key")
    defer { out.free() }
    assert_eq(out.len(), rows)
    let got = out.column("value").to_floats()
    let tags = out.column("tag").to_floats()
    var wrong = 0
    for at in 0..rows {
        let key = at * 37 % 500
        if got[at] != float(at) { wrong += 1 }
        let tag = tags[at]
        if key < 400 {
            if tag != float(key * 2 + 1) { wrong += 1 }
        } elif tag == tag {
            wrong += 1
        }
    }
    assert_eq(wrong, 0)
}

#: `sort: true` orders the output by key, and a stable order leaves the rows
#: of one key in left order.
fn test_a_sorted_join_orders_by_key_and_keeps_ties_in_left_order() -> void {
    if not gpu.is_available() { return }
    let rows = 70000
    let (left, right) = keyed_frames(rows)
    defer { left.free() }
    defer { right.free() }
    let out = merge(left, right, on: "key", sort: true)
    defer { out.free() }
    let keys = out.column("key").to_floats()
    let got = out.column("value").to_floats()
    var wrong = 0
    for at in 1..keys.len() {
        if keys[at] < keys[at - 1] { wrong += 1 }
        if keys[at] == keys[at - 1] and got[at] < got[at - 1] { wrong += 1 }
    }
    assert_eq(wrong, 0)
    assert_true(keys.len() > 50000, "the join kept the matching rows")
}
