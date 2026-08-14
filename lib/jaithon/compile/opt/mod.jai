#: The optimiser: the pass pipeline of `spec/BYTECODE.md` §8, over a chunk the
#: emitter has just finished.
#:
#: The Jaithon half of `jaiOptimize` / `optimizeFunction`
#: (`src/codegen/optimize.c:1808`). The order below is not a suggestion: the
#: two front ends are compared instruction for instruction by `make bootstrap`,
#: so it is part of the specification.
#:
#: `_optimize_nested` walks the constant pool and re-optimises every nested
#: function once per enclosing function -- that repetition looks redundant but
#: is deliberate: it's what the C driver does, and the two front ends must
#: agree byte for byte.
from jaithon.compile.emit import ConstTag, FuncProto
from .chunk import compact, decode, is_defer_thunk, mark_unreachable, rebuild
from .coalesce import coalesce_slots, coalesce_slots_on
from .dead import dead_code
from .fuse import superinstructions, superinstructions_on
from .hoist import hoist_globals, hoist_globals_on
from .peephole import peephole, peephole_on

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

fn _optimize_nested(proto: FuncProto, level: int, depth: int) -> void {
    for value in proto.chunk.constants {
        if value.tag != ConstTag.Func { continue }
        let nested = value.value
        if nested is not proto { _optimize_function(nested, level, depth + 1) }
    }
}

#: Run the passes over one function, decoding once and re-encoding once.
#:
#: Each pass used to decode and rebuild on its own -- about six round trips per
#: function once the fixpoint loop is counted, and it dominated the optimiser:
#: measured over the standard library, decode+rebuild is 317ms against 2.9s for
#: every rewrite combined. No pass reads `chunk.code`, so the decode moved here
#: and passes gained `_on` entry points taking the `Code` already in hand.
#:
#: One rebuild at the end also makes failure all-or-nothing: it leaves the
#: chunk untouched when unrepresentable, so `--O0` can't fail to compile at
#: `--O2`. `max_slots` is the one state the passes write outside `code`, so it
#: is saved and restored on that path.
fn _optimize_function(proto: FuncProto, level: int, depth: int) -> void {
    if depth > MAX_DEPTH { return }
    _optimize_nested(proto, level, depth)

    let code = decode(proto)
    if code == null { return }
    let slots_before = proto.max_slots

    var changes = mark_unreachable(code)
    if changes > 0 { compact(code) }

    if level > 0 {
        for _round in 0..FIXPOINT_ROUNDS {
            var round = peephole_on(code)
            if round > 0 { compact(code) }
            let fused = superinstructions_on(code)
            if fused > 0 { compact(code) }
            round += fused
            if round == 0 { break }
            changes += round
        }
        if not is_defer_thunk(proto) and hoist_globals_on(code, proto) > 0 {
            changes += 1
            compact(code)
            if peephole_on(code) > 0 {
                compact(code)
                changes += 1
            }
            if superinstructions_on(code) > 0 {
                compact(code)
                changes += 1
            }
        }

        if not is_defer_thunk(proto) { changes += coalesce_slots_on(code) }
    }

    if changes > 0 and not rebuild(code) { proto.max_slots = slots_before }
}
