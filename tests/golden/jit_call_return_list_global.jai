#: gc-stress-every: 50
#: A global call whose predicted return kind is `SLOT_LIST` must not read
#: a differently-typed result as an `ObjList`.
#:
#: `emitGlobalCall`'s post-call guard checked the return tag (`VAL_OBJ`)
#: and, for `SLOT_INST`, went on to confirm the object's own `Obj.type`
#: before trusting `klass` -- but had no equivalent arm for `SLOT_LIST`.
#: `pick` below gets a compiled specialisation whose `jitReturnKind` is
#: `SLOT_LIST` (from being warmed with a list argument). `boom` calls it
#: with a `str` argument instead: `directCallArgsMatch` declines the
#: direct call over the kind mismatch, the descriptor path is taken with
#: `rk == SLOT_LIST`, and at that point `pick` actually runs interpreted
#: and hands back the `str` untouched. VAL_OBJ alone does not prove that
#: `str` is a list -- without the missing `Obj.type == OBJ_LIST` check,
#: its header bytes at `ObjList.count`'s offset are unrelated `ObjString`
#: fields, so `.len()` on the "list" reads garbage instead of the
#: string's real length.
fn pick(xs: any, n: int) -> any {
    if n > 1000000000 { return xs }
    return xs
}

fn boom(s: any, n: int) -> int {
    var t = 0
    for i in 0..n {
        t = t + pick(s, i).len()
    }
    return t
}

fn warm_pick(xs: any, n: int) -> int {
    var t = 0
    for i in 0..n {
        t = t + pick(xs, i).len()
    }
    return t
}

fn main() -> int {
    let xs = [1, 2, 3]
    var w = 0
    for _r in 0..4000 { w = w + warm_pick(xs, 8) }

    let s = "hello"
    var t = 0
    for _r in 0..4000 { t = t + boom(s, 8) }
    print(t)
    return 0
}
