#: gc-stress-every: 50
#: `xs[a:b]` on a tuple must not decline the enclosing function.
#:
#: `OP_GET_SLICE`'s container-type guard recognized `SLOT_LIST` and a
#: string held as `SLOT_OBJ`, but not a tuple held as `SLOT_OBJ` -- even
#: though the C helper it calls out to, `jitGetSlice`, is a thin wrapper
#: over `jaiSliceGet`, which already handles a tuple container exactly like
#: a list or a string. Only this arm's own guard was narrower than what the
#: call it makes actually supports.
fn slice_sum(t: any, n: int) -> int {
    var total = 0
    var i = 0
    while i < n {
        let s = t[1:3]
        total = total + s.len()
        i = i + 1
    }
    return total
}

let t: any = (10, 20, 30, 40, 50)
print(slice_sum(t, 200000))
