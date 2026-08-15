#: gc-stress-every: 50
#: A list-of-lists read must not admit a DIFFERENT raw object kind through
#: the same guard.
#:
#: `OP_GET_INDEX`'s list-element arm samples element 0 to decide a kind for
#: the whole read site. For `SLOT_LIST` it only checked the generic VAL_OBJ
#: value tag before this fix -- the same tag every heap object carries -- and
#: never confirmed the specific object was actually `Obj.type == OBJ_LIST`,
#: unlike every other `SLOT_LIST` producer in the tier (`OP_GET_FIELD_LOCAL`,
#: `OP_GET_FIELD`, `OP_GET_GLOBAL`'s `globalKind` path), which all confirm it.
#:
#: A list holding both a nested list and some other raw object kind (string,
#: dict, set, tuple) therefore passed the same weak guard on either element,
#: and code compiled for "this is a list" read the other object's bytes
#: through `ObjList`'s field offsets -- `items` at +16, `count`/`capacity` at
#: +24 -- silently, with no crash and no diagnostic. Confirmed as a real,
#: reproducible miscompile before the fix: JIT-enabled and interpreter-only
#: runs of `total_len` below disagreed (1685108 against the correct
#: 15750000), and `JAI_JIT_DEOPT_STRESS`/`JAI_JIT_SPLIT_STRESS` did not
#: surface it on their own -- only comparing against the interpreter did.
fn total_len(rows: list[any], n: int) -> int {
    var t = 0
    var i = 0
    while i < n {
        t = t + rows[i % rows.len()].len()
        i = i + 1
    }
    return t
}

let mixed: list[any] = [[1, 2], "hello world this is a longer string than a small one for sure"]
print(total_len(mixed, 500000))
