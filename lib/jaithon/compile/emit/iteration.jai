#: The bodies of `while`, `loop` and `for`, the specialised range form of
#: `for`, and the `break` and `continue` that leave them.
#:
#: Named `iteration` and not `loop` because `loop` is a KEYWORD: the import
#: path `jaithon.compile.emit.loop` does not parse.
#:
#: What is NOT here: the loop frame itself. `_enter_loop` and `_close_loop` are
#: on `Emitter`, because `while` and `loop` are lowered inline by `_stmt` and
#: push their frames from there.
#:
#: The seam. A loop body is a statement, so this file calls `lower_stmt`.
from jaithon.ast import Node, NodeKind, Span
from jaithon.compile.emit.opcode import Op
from jaithon.compile.emit.state import Emitter, lower_stmt
from jaithon.compile.emit.exception import _finally_chain_for_loop
from jaithon.compile.emit.expr import _expr, _items_iterable
from jaithon.compile.emit.shape import _is_items_call

pub fn _body_of_loop(
    e: Emitter,
    node: Node,
    continue_target: int,
    continue_depth: int,
    break_depth: int
) -> void {
    let base = e.resolution.capture_base_of(node)
    e._enter_loop(node.opt_text("label"), continue_target, base, continue_depth, break_depth)
    _loop_body(e, node.child("body"))
    e._close_iteration(base, node.span)
}

pub fn _loop_body(e: Emitter, body: Node?) -> void {
    if body == null { return }
    if body.kind != NodeKind.Block {
        lower_stmt(e, body)
        return
    }
    for statement in body.child_list("stmts") { lower_stmt(e, statement) }
}

#: `for x in a..b` with no ObjRange and no ObjIter: two int temporaries hold
#: the counter and the end, and nothing is pushed at all.
#:
#: The saving is per loop ENTRY, which is why it is worth an opcode pair: an
#: allocation census found the two objects a range loop's header builds were
#: 99.95% of everything `tests/bench/nbody` allocated and 98.5% of
#: `matrix_mul`'s. Measured at ~15.7ns per entry on an M2 Max, two thirds of
#: it the allocation itself and a third the collections it forces.
#:
#: Emitted here rather than fused from bytecode — unlike `FOR_ITER_PAIR`,
#: which is a peephole rule — because the loop stops occupying an operand
#: stack slot, and `break` unwinds to a depth only this function knows.
#:
#: Answers false, having emitted nothing, for a loop that is not this shape:
#: the range must be written out at the loop (one reaching it through a
#: variable already built its object, so there is nothing left to save) and
#: it must bind to a single frame slot.
pub fn _for_range(e: Emitter, node: Node, iterable: Node?) -> bool {
    if iterable == null or iterable.kind != NodeKind.Range { return false }
    let pattern = node.child("pattern")
    if e._range_bind_slot(pattern, false) < 0 { return false }
    let span = node.span
    let base = e._depth()
    let wildcard = pattern != null and pattern.kind == NodeKind.PatWildcard
    let counter = e._alloc_temp()
    let end = e._alloc_temp()
    let slot = e._range_bind_slot(pattern, wildcard)
    _expr(e, iterable.child("start"))
    _expr(e, iterable.child("stop"))
    e._chunk().write_op(Op.IterRange, span)
    e._chunk().write_byte(if iterable.flag("inclusive") { 1 } else { 0 })
    e._chunk().write_u16(counter)
    e._chunk().write_u16(end)

    let start = e._chunk().here()
    e._chunk().write_op(Op.ForRangeBind, span)
    let exit = e._chunk().here()
    e._chunk().write_u16(0)
    e._chunk().write_u16(slot)
    e._chunk().write_u16(counter)
    e._chunk().write_u16(end)

    let capture_base = e.resolution.capture_base_of(node)
    e._enter_loop(node.opt_text("label"), start, capture_base, base, base)
    _loop_body(e, node.child("body"))
    e._close_iteration(capture_base, span)
    e._loop_back(start, span)
    e._set_depth(base)
    # The displacement is measured from the end of the instruction, which is
    # six bytes past its own operand rather than the usual zero.
    e._patch_from(exit, exit + 8)
    e._close_loop()
    e._free_temps(if wildcard { 3 } else { 2 })
    return true
}

pub fn _for(e: Emitter, node: Node) -> void {
    let span = node.span
    let base = e._depth()
    let iterable = node.child("iterable")
    if _for_range(e, node, iterable) { return }
    if _is_items_call(iterable) {
        _items_iterable(e, iterable, span)
    } else {
        _expr(e, iterable)
        e._op(Op.GetIter, span)
    }
    let start = e._chunk().here()
    let exit = e._emit_jump(Op.ForIter, span)
    e._adjust(1)
    let capture_base = e.resolution.capture_base_of(node)
    e._enter_loop(node.opt_text("label"), start, capture_base, base + 1, base)
    e._bind_pattern(node.child("pattern"), span)
    _loop_body(e, node.child("body"))
    e._close_iteration(capture_base, span)
    e._loop_back(start, span)
    e._set_depth(base)
    e._patch(exit)
    e._close_loop()
}

pub fn _jump(e: Emitter, node: Node) -> void {
    let loops = e._ctx().loops
    let label = node.opt_text("label")
    var index = loops.len() - 1
    if label != null {
        while index >= 0 and loops[index].label != label { index -= 1 }
    }
    if index < 0 {
        e._unsupported(node.span, f"`break`/`continue` to `{label}`")
        return
    }
    let frame = loops[index]
    _finally_chain_for_loop(e, index)
    let saved = e._depth()
    var want = frame.continue_depth
    if node.kind == NodeKind.Break { want = frame.break_depth }
    e._pop_n(saved - want, node.span)
    e._close_iteration(e._close_base_through(index), node.span)
    if node.kind == NodeKind.Break {
        frame.breaks.push(e._emit_jump(Op.Jump, node.span))
    } else {
        e._loop_back(frame.continue_target, node.span)
    }
    e._set_depth(saved)
}
