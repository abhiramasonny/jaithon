#: gc-stress-every: 50
#: A list element of raw object type must not decline the enclosing function.
#:
#: `OP_GET_INDEX`'s list-element arm classified a string element into
#: SLOT_OBJ but every other raw-holdable kind -- dict, set, tuple -- still hit
#: its `else return false`, declining the whole function the same way an
#: unclassified field used to (see jit_field_obj.jai). Widened from
#: `IS_STRING(elem)` to the same `rawObjValue` predicate the field and global
#: sites already use.
#:
#: Two things have to hold, and neither is visible without the JIT:
#:   * a list of dicts read and invoked on (`.len()`, through the inline
#:     cache) sees the right element each iteration;
#:   * the same compiled body, called again with a list of a different raw
#:     object kind (set rather than dict), deoptimises on the sample mismatch
#:     rather than reading a set's layout as a dict's.
fn total_len(rows: list[any], n: int) -> int {
    var t = 0
    var i = 0
    while i < n {
        t = t + rows[i % rows.len()].len()
        i = i + 1
    }
    return t
}

let dicts: list[any] = [{"a": 1, "b": 2}, {"c": 3}, {}]
print(total_len(dicts, 200000) // 200000)

let sets: list[any] = [{1, 2, 3}, {4}, {5, 6}]
print(total_len(sets, 200000) // 200000)

print(total_len(dicts, 200000) // 200000)
