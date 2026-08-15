#: gc-stress-every: 50
#: A `for` loop's element classification must admit every raw-holdable
#: element kind `OP_GET_INDEX` already does, and must guard `SLOT_LIST`
#: exactly as strictly.
#:
#: `OP_FOR_ITER_BIND` (both its ordinary and OSR-loop-head arms) classified a
#: list element into `SLOT_INT`/`SLOT_FLOAT`/`SLOT_INST` only (plus
#: `SLOT_BOOL` in the OSR arm) and declined the whole loop for anything else
#: -- a list of strings, dicts, sets, tuples, or nested lists never compiled.
#: Widened to match `OP_GET_INDEX`'s full ladder, including the `SLOT_LIST`
#: guard that confirms `Obj.type == OBJ_LIST` before a consumer trusts it
#: with no check of its own (see `tests/golden/jit_heterogeneous_list.jai`
#: for the bug that guard exists to prevent -- `OP_GET_INDEX` had it,
#: `OP_FOR_ITER_BIND` did not, until this fix).
#:
#: Four element kinds, one loop shape each, plus the heterogeneous case: a
#: list mixing a nested list with a string must correctly decline to admit a
#: single loop-variable kind (`adoptLocalKindSeen` sees two different kinds
#: across iterations) rather than misreading one element's bytes as the
#: other's -- the outer loop still compiles, only the inner element read
#: does not, and the answer must be correct either way.
fn sum_str_lens(xs: list[any], n: int) -> int {
    var t = 0
    var i = 0
    while i < n {
        for s in xs { t = t + s.len() }
        i = i + 1
    }
    return t
}

fn count_true(xs: list[any], n: int) -> int {
    var t = 0
    var i = 0
    while i < n {
        for b in xs {
            if b { t = t + 1 }
        }
        i = i + 1
    }
    return t
}

fn sum_inner_lens(xs: list[any], n: int) -> int {
    var t = 0
    var i = 0
    while i < n {
        for row in xs { t = t + row.len() }
        i = i + 1
    }
    return t
}

fn sum_dict_lens(xs: list[any], n: int) -> int {
    var t = 0
    var i = 0
    while i < n {
        for d in xs { t = t + d.len() }
        i = i + 1
    }
    return t
}

fn sum_mixed_lens(xs: list[any], n: int) -> int {
    var t = 0
    var i = 0
    while i < n {
        for e in xs { t = t + e.len() }
        i = i + 1
    }
    return t
}

let strs: list[any] = ["a", "bb", "ccc"]
print(sum_str_lens(strs, 50000))

let bools: list[any] = [true, false, true, true]
print(count_true(bools, 50000))

let rows: list[any] = [[1, 2], [3, 4, 5], []]
print(sum_inner_lens(rows, 50000))

let dicts: list[any] = [{"a": 1}, {"b": 2, "c": 3}]
print(sum_dict_lens(dicts, 50000))

let mixed: list[any] = [
    [1, 2],
    "a longer string than the list to make the shapes clearly different",
]
print(sum_mixed_lens(mixed, 200000))
