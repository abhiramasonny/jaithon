#: gc-stress-every: 50
#: Same hazard as jit_call_return_list_global.jai, at the OP_INVOKE
#: method-call site instead of a global call.
#:
#: `OP_INVOKE`'s post-call guard had the identical shape as
#: `emitGlobalCall`'s: it confirmed `Obj.type == OBJ_INSTANCE` for a
#: `SLOT_INST` result before reading `klass`, but had no equivalent arm
#: for `SLOT_LIST`. `Holder.pick` gets a compiled specialisation whose
#: `jitReturnKind` is `SLOT_LIST` (warmed with a list argument); `boom`
#: invokes it with a `str` argument, the kind mismatch declines the
#: direct call, the descriptor path predicts `SLOT_LIST`, and the method
#: actually runs interpreted and returns the `str` untouched. Without the
#: `Obj.type == OBJ_LIST` check, `.len()` on that "list" reads
#: `ObjString` fields at `ObjList.count`'s offset instead of the string's
#: real length.
class Holder {
    pub var tag: int = 0

    pub fn pick(self, xs: any, n: int) -> any {
        if n > 1000000000 { return xs }
        return xs
    }
}

fn boom(h: Holder, s: any, n: int) -> int {
    var t = 0
    for i in 0..n {
        t = t + h.pick(s, i).len()
    }
    return t
}

fn warm(h: Holder, xs: any, n: int) -> int {
    var t = 0
    for i in 0..n {
        t = t + h.pick(xs, i).len()
    }
    return t
}

fn main() -> int {
    let h = Holder()
    let xs = [1, 2, 3]
    var w = 0
    for _r in 0..4000 { w = w + warm(h, xs, 8) }

    let s = "hello"
    var t = 0
    for _r in 0..4000 { t = t + boom(h, s, 8) }
    print(t)
    return 0
}
