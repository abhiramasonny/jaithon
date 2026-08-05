#: The optimiser: the pass pipeline of `spec/BYTECODE.md` §8, over a chunk the
#: emitter has just finished.
#:
#: The Jaithon half of `jaiOptimize` / `optimizeFunction`
#: (`src/codegen/optimize.c:1808`). The order below is not a suggestion: the
#: two front ends are compared instruction for instruction by `make bootstrap`,
#: so the pipeline is part of the specification and every line of it is the
#: same line the C driver runs.
#:
#: `optimize` is called once per function, as codegen finishes it — that is
#: where `jaiCgFinishFunction` (`src/codegen/codegen.c`) calls `jaiOptimize`,
#: and `jaithon.compile.compile_source` is what hands this module to the
#: emitter. `_optimize_nested` then walks the constant pool and optimises every
#: function record it finds there, so a front end that only reached the module
#: body would still get a fully optimised program — and a front end that does
#: both, as this one does, runs the pipeline over a nested function once per
#: enclosing function. That repetition is deliberate: it is what the C driver
#: does, and the two front ends have to agree byte for byte, so it is not a
#: round that can be dropped for being redundant.
from jaithon.compile.emit import ConstTag, FuncProto
from .chunk import is_defer_thunk
from .coalesce import coalesce_slots
from .dead import dead_code
from .fuse import superinstructions
from .hoist import hoist_globals
from .peephole import peephole

#: Deepest nesting `optimize` descends into the constant pool.
#:
#: A pool that names its own enclosing function would otherwise recurse
#: forever; the identity test below catches the direct case and this catches
#: everything else.
pub const MAX_DEPTH: int = 64

#: Rounds the byte-level passes alternate for before the pipeline moves on.
pub const FIXPOINT_ROUNDS: int = 4

#: Optimise `proto` and every function record in its constant pool.
#:
#: `level` is the `--O` level: 0 runs dead-code elimination only, anything
#: above it runs the whole list. 2 is the default the CLI passes and the level
#: `make bootstrap` compares at.
pub fn optimize(proto: FuncProto, level: int = 2) -> void {
    _optimize_function(proto, level, 0)
}

# Nested functions live in the constant pool, so optimising them from here
# means a caller that only reaches the module body still gets a fully optimised
# program. A record never names itself, but a malformed pool could, so the
# identity test stays.
fn _optimize_nested(proto: FuncProto, level: int, depth: int) -> void {
    for value in proto.chunk.constants {
        if value.tag != ConstTag.Func { continue }
        let nested = value.value
        if nested is not proto { _optimize_function(nested, level, depth + 1) }
    }
}

fn _optimize_function(proto: FuncProto, level: int, depth: int) -> void {
    if depth > MAX_DEPTH { return }
    _optimize_nested(proto, level, depth)

    # Unreachable code is removed at every level: spec §8 disables passes 3-7
    # under --O0, not pass 2.
    dead_code(proto)

    if level <= 0 { return }

    # Fusing exposes new windows and deleting exposes new fusions, so run the
    # byte-level passes to a fixpoint before the ones that depend on the final
    # instruction stream.
    for _round in 0..FIXPOINT_ROUNDS {
        var changes = peephole(proto)
        changes += superinstructions(proto)
        if changes == 0 { break }
    }
    if not is_defer_thunk(proto) and hoist_globals(proto) > 0 {
        peephole(proto)
        superinstructions(proto)
    }

    # Coalescing runs last so that it sees the final set of locals, including
    # the ones hoisting just introduced. Running it earlier leaves a hoisted
    # slot unpacked until the next call, which makes the pipeline disagree with
    # itself about the frame size.
    #
    # A defer thunk is exempt: its slot numbers are the defining frame's, so
    # renumbering them against its own (empty) parameter list would make it read
    # the wrong local.
    if not is_defer_thunk(proto) { coalesce_slots(proto) }
}
