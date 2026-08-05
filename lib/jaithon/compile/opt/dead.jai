#: Dead-code elimination: pass 2 of `spec/BYTECODE.md` §8.
#:
#: The Jaithon half of `jaiOptDeadCode` (`src/codegen/optimize.c:576`). Runs at
#: every optimisation level, `--O0` included: §8 disables passes 3-7, not this
#: one.
#:
#: The shape is the one every pass in this package has — `decode`, transform,
#: `rebuild` — and the transform is `mark_unreachable`, which lives in
#: `jaithon.compile.opt.chunk` because the peephole alternates with it.
#:
#: What this deletes in practice is an epilogue nobody asked for. Both emitters
#: close a block-bodied function with `OP_RETURN_NULL` whichever way the block
#: ended (`jaiEmitFunctionNode`, `src/codegen/codegen_decl.c:535`, and
#: `Emitter._function_constant` here), because whether the last statement
#: returns on every path is a question about the whole body and the emitter is
#: walking it forwards. A body that already ended in `return` therefore carries
#: a trailing `OP_RETURN_NULL` that `OP_RETURN` never falls through to, and this
#: pass is what removes it. Running at `--O0` is what makes that safe to rely
#: on: the epilogue goes at every level, so the two front ends agree about the
#: length of every function at every level, and `make bootstrap` compares them
#: with no allowance for one.
#:
#: Everything else it removes follows from the same walk: statements after a
#: `return` or a `throw`, the code after a loop whose only exit edge an earlier
#: round of the peephole folded away, and the handler of a `try` whose protected
#: region is itself unreachable.
from jaithon.compile.emit import FuncProto
from .chunk import decode, mark_unreachable, rebuild

#: Delete every instruction unreachable from an entry point; answer how many.
#:
#: Answers 0 and leaves the chunk untouched when the chunk cannot be decoded or
#: the result cannot be re-encoded.
#:
#: The entry points and the handler rule are `mark_unreachable`'s: offset 0 and
#: every default-value thunk are reachable unconditionally, a handler only once
#: something in its protected region is. Nothing here may seed a handler on its
#: own; doing so resurrects the `try` of a branch the front end already proved
#: dead, and through the backward jump that closes an enclosing loop it takes
#: the loop's `OP_FOR_ITER` with it, whose iterator was produced by setup code
#: that stayed dead.
pub fn dead_code(proto: FuncProto) -> int {
    let code = decode(proto)
    if code == null { return 0 }

    var removed = mark_unreachable(code)

    # A rebuild that answers false is a capacity verdict and not a diagnosis:
    # the chunk is left exactly as it was and the caller hears about no
    # changes, so a program that compiles at `--O0` cannot fail at `--O2`.
    if removed > 0 and not rebuild(code) { removed = 0 }
    return removed
}
