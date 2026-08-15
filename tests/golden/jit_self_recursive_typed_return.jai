#: gc-stress-every: 50
#: Regression coverage for `OP_CALL`'s self-recursive-call slow stub, which
#: is where the compiler emits a direct call to the function currently
#: being compiled (distinct from `emitDirectCall`, used for calls to an
#: already-fully-compiled other function).
#:
#: That stub used to trust a SLOT_LIST or SLOT_INST result's `VAL_OBJ` tag
#: alone (`retType = -1`, "no type check"), reasoning that the function is
#: its own callee, so its own returns already bound `e->returnKind`. But
#: `e->returnKind` is built by `mergeReturnKind` at each `OP_RETURN` the
#: compile-time walk actually visits -- and `emitUnarmedDeopt` can skip a
#: whole run of instructions wholesale (by raw instruction length, never
#: through the opcode switch) when it meets something this tier can't
#: compile. A `return` inside a skipped run is invisible to
#: `mergeReturnKind`, so it is never proven to agree with `e->returnKind`,
#: yet it is still live code the interpreter runs after a deopt into that
#: span. `retType`/`retShape` are now set the same way
#: `emitDirectCall`'s sibling stub already does, closing the gap.
#:
#: No live case forcing that divergent path was constructed (the natural
#: repro is circular: it needs an opcode this tier declines mid-body,
#: which is rare in code that otherwise compiles a clean self-recursion),
#: so this is positive-path coverage: ordinary self-recursion returning a
#: list or an instance must still answer correctly with the extra guard
#: in place, at every opt level and every stress mode.
class Box {
    pub var n: int

    fn init(self, n: int) -> void { self.n = n }
}

fn count_down_list(n: int) -> any {
    if n <= 0 { return [n] }
    return count_down_list(n - 1)
}

fn count_down_inst(n: int) -> any {
    if n <= 0 { return Box(n) }
    return count_down_inst(n - 1)
}

fn main() -> int {
    var total = 0
    var i = 0
    while i < 200000 {
        let lst: any = count_down_list(5)
        total = total + lst.len()
        let b: any = count_down_inst(5)
        total = total + b.n
        i = i + 1
    }
    print(total)
    return 0
}
