#: gc-stress-every: 50
#: A dynamic local's read must confirm its specific object type, not just
#: the generic `VAL_OBJ` value tag.
#:
#: A local reassigned different `SlotKind`s across its lifetime (here: a
#: list on one path, a dict on another) is compiled as a "dynamic local" --
#: `dynamicLocal[slot]` -- and every read of it guards against a runtime
#: fact stored alongside it (`localTagFor`) before trusting the value.
#: `localTagFor` maps `SLOT_LIST`, `SLOT_OBJ` and `SLOT_INST` all to the same
#: `VAL_OBJ` tag, since that is genuinely all a bare value tag can say. A
#: read compiled for `SLOT_LIST` therefore passed that guard just as easily
#: for a dict the slot held on a later call, and proceeded to read the
#: dict's bytes through `ObjList`'s field offsets -- `items` at +16,
#: `count`/`capacity` at +24 -- the same shape as
#: `jit_heterogeneous_list.jai`'s bug, one level up (a local, not an
#: element). Confirmed as a real miscompile before this fix: JIT gave
#: 212672, the interpreter gives the correct 4160000.
fn f(n: int, lst: any, d: any) -> int {
    var a = lst
    var t = 0
    var i = 0
    while i < n {
        t = t + a[0]
        a = d
        i = i + 1
    }
    return t
}

let lst: any = [10, 20, 30]
let d: any = {0: 99}
var total = 0
var k = 0
while k < 20000 {
    total = total + f(3, lst, d)
    k = k + 1
}
print(total)
