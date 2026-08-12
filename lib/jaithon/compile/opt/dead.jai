#: Dead-code elimination: pass 2 of `spec/BYTECODE.md` §8.
#:
#: The Jaithon half of `jaiOptDeadCode` (`src/codegen/optimize.c:576`). Runs at
#: every optimisation level, `--O0` included: §8 disables passes 3-7, not this
#: one.
#:
#: In practice this mostly deletes an unwanted epilogue: both emitters
#: unconditionally close a block-bodied function with a trailing
#: `OP_RETURN_NULL`, since the emitter walks forward and can't know in advance
#: that every path already returns. That epilogue exists at every optimisation
#: level in both front ends, which is what lets `make bootstrap` compare them
#: byte-for-byte with no allowance for it.
#:
#: Everything else follows the same walk: code after `return`/`throw`, a dead
#: loop's unreachable body, an unreachable `try`'s handler.
from jaithon.compile.emit import FuncProto
from .chunk import decode, mark_unreachable, rebuild

#: Delete every instruction unreachable from an entry point; answer how many.
#:
#: Answers 0 and leaves the chunk untouched when the chunk cannot be decoded or
#: the result cannot be re-encoded.
#:
#: The entry points and handler rule are `mark_unreachable`'s: nothing here may
#: seed a handler on its own, since doing so can resurrect code (e.g. a dead
#: loop's `OP_FOR_ITER` setup) the front end already proved dead.
pub fn dead_code(proto: FuncProto) -> int {
    let code = decode(proto)
    if code == null { return 0 }

    var removed = mark_unreachable(code)

    # False is a capacity verdict, not a diagnosis: the chunk is left as it
    # was, so a program that compiles at `--O0` cannot fail at `--O2`.
    if removed > 0 and not rebuild(code) { removed = 0 }
    return removed
}
