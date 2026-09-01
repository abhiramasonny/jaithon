#: `match` and the tests its arms compile to: the dispatch over the arms, and
#: one test per pattern form -- an alternation, a sequence with an optional
#: rest, an enum variant with its payload, a class with its fields.
#:
#: Every test reads the subject from a FRAME SLOT rather than the operand
#: stack, which is what lets a test fail half way through destructuring
#: without leaving anything to unwind. `_pattern_in_slot` is the dispatch the
#: sub-tests recurse back into.
#:
#: What is NOT here: `_bind_pattern`, the irrefutable form used by `let` and a
#: loop header, which is on `Emitter` because it cannot fail and so needs none
#: of this file's miss bookkeeping.
#:
#: The seam. An arm body is a statement, so this file calls `lower_stmt`.
from jaithon.ast import Node, NodeKind, Span, is_statement
from jaithon.compile.emit.opcode import Op
from jaithon.compile.emit.spec import Const, ConstTag
from jaithon.compile.emit.state import Emitter, lower_stmt
from jaithon.compile.emit.expr import _block_expr, _expr
from jaithon.compile.emit.shape import _field_name

pub fn _match(e: Emitter, node: Node, as_expression: bool) -> void {
    let span = node.span
    let base = e._depth()

    _expr(e, node.child("subject"))
    let subject = e._alloc_temp()
    e._bind_local(subject, span)

    var done: list[int] = []
    for arm in node.records("arms") {
        let arm_span: Span = arm["span"]
        var misses: list[int] = []
        var guard_miss: list[int] = []

        e._set_depth(base)
        _pattern_in_slot(e, arm["pattern"], subject, misses)

        if arm["guard"] != null {
            let guard: Node = arm["guard"]
            _expr(e, guard)
            guard_miss.push(e._emit_jump(Op.JumpIfFalse, guard.span))
        }

        let body: Node? = arm["body"]
        if as_expression {
            if body != null and body.kind == NodeKind.Block {
                _block_expr(e, body)
            } else {
                _expr(e, body)
            }
        } elif body != null {
            if body.kind == NodeKind.Block or is_statement(body.kind) {
                lower_stmt(e, body)
            } else {
                _expr(e, body)
                e._op(Op.Pop, arm_span)
            }
        }
        done.push(e._emit_jump(Op.Jump, arm_span))

        # A pattern miss arrives with the failed candidate still pushed.
        e._set_depth(base + 1)
        e._patch_all(misses)
        e._op(Op.Pop, arm_span)
        e._set_depth(base)
        e._patch_all(guard_miss)
    }

    e._set_depth(base)
    if as_expression {
        # The checker proves exhaustiveness; this covers `any` subjects and
        # keeps the join depth honest.
        e._op_k(Op.AssertFail, e._string("no match arm matched"), span)
        e._op(Op.Null, span)
    }

    e._set_depth(base + (if as_expression { 1 } else { 0 }))
    e._patch_all(done)
    e._free_temps(1)
}

pub fn _pattern_in_slot(e: Emitter, pattern: Node?, slot: int, misses: list[int]) -> void {
    if pattern == null { return }
    let span = pattern.span
    let kind = pattern.kind

    if kind == NodeKind.PatWildcard {
        return
    } elif kind == NodeKind.PatBind {
        let declared = pattern.type_field("type")
        if declared != null {
            e._get_local(slot, span)
            e._chunk().write_op(Op.MatchTypePop, span)
            e._chunk().write_u24(e._string(declared.name ?? "any"))
            misses.push(e._reserve_jump())
        }
        let symbol = e.resolution.symbol_of(pattern)
        if symbol == null and pattern.opt_text("name") == null { return }
        e._get_local(slot, span)
        e._store_symbol(symbol, pattern.opt_text("name") ?? "_", span, symbol == null)
    } elif kind == NodeKind.PatLiteral {
        # The success path always pops the subject copy right here -- see
        # spec/BYTECODE.md's `MATCH_CONST_POP` entry -- so it is emitted
        # fused rather than as MatchConst followed by a separate Pop.
        e._get_local(slot, span)
        e._chunk().write_op(Op.MatchConstPop, span)
        e._chunk().write_u24(e._literal_constant(pattern.child("value")))
        misses.push(e._reserve_jump())
    } elif kind == NodeKind.PatRange {
        e._get_local(slot, span)
        e._chunk().write_op(Op.MatchRangePop, span)
        e._chunk().write_u24(e._literal_constant(pattern.child("lo")))
        e._chunk().write_u24(e._literal_constant(pattern.child("hi")))
        e._chunk().write_byte(if pattern.flag("inclusive") { 1 } else { 0 })
        misses.push(e._reserve_jump())
    } elif kind == NodeKind.PatOr {
        _test_or(e, pattern, slot, misses)
    } elif kind == NodeKind.PatTuple or kind == NodeKind.PatList {
        _test_sequence(e, pattern, slot, misses)
    } elif kind == NodeKind.PatEnum {
        _test_enum(e, pattern, slot, misses)
    } elif kind == NodeKind.PatClass {
        _test_class(e, pattern, slot, misses)
    } else {
        e._unsupported(span, f"the pattern {kind}")
    }
}

pub fn _sub_pattern(e: Emitter, sub: Node?, misses: list[int], span: Span) -> void {
    if sub == null or sub.kind == NodeKind.PatWildcard {
        e._op(Op.Pop, span)
        return
    }
    if sub.kind == NodeKind.PatBind and sub.type_field("type") == null {
        e._bind_pattern(sub, span)
        return
    }
    let temp = e._alloc_temp()
    e._bind_local(temp, span)
    _pattern_in_slot(e, sub, temp, misses)
    e._free_temps(1)
}

pub fn _test_or(e: Emitter, pattern: Node, slot: int, misses: list[int]) -> void {
    let alternatives = pattern.child_list("elems")
    if alternatives.len() == 0 { return }
    var matched: list[int] = []
    let base = e._depth()

    for (index, alternative) in alternatives.enumerate() {
        if index == alternatives.len() - 1 {
            _pattern_in_slot(e, alternative, slot, misses)
            break
        }
        var local: list[int] = []
        _pattern_in_slot(e, alternative, slot, local)
        matched.push(e._emit_jump(Op.Jump, pattern.span))
        e._set_depth(base + 1)  # the failing candidate is still pushed
        e._patch_all(local)
        e._op(Op.Pop, pattern.span)
    }

    e._set_depth(base)
    e._patch_all(matched)
}

pub fn _test_sequence(e: Emitter, pattern: Node, slot: int, misses: list[int]) -> void {
    let span = pattern.span
    let elements = pattern.child_list("elems")
    let rest = pattern.number("restIndex")
    let count = elements.len()
    let has_rest = rest >= 0
    let fixed = if has_rest { count - 1 } else { count }
    let tail = if has_rest { count - rest - 1 } else { 0 }

    e._get_local(slot, span)
    e._chunk().write_op(Op.MatchSeqPop, span)
    e._chunk().write_byte(if fixed > 255 { 255 } else { fixed })
    e._chunk().write_byte(if has_rest { 1 } else { 0 })
    misses.push(e._reserve_jump())

    for (index, element) in elements.enumerate() {
        if has_rest and index == rest {
            e._get_local(slot, span)
            e._emit_int(rest, span)
            var flags = 1
            var pushed = 1
            if tail > 0 {
                e._emit_int(-tail, span)
                flags = flags | 2
                pushed += 1
            }
            e._op_u8(Op.GetSlice, flags, span)
            e._adjust(-pushed)
        } else {
            let offset = if has_rest and index > rest { index - count } else { index }
            e._get_local(slot, span)
            e._emit_int(offset, span)
            e._op(Op.GetIndex, span)
        }
        _sub_pattern(e, element, misses, span)
    }
}

pub fn _test_enum(e: Emitter, pattern: Node, slot: int, misses: list[int]) -> void {
    let span = pattern.span
    let owner = pattern.opt_text("typeName")
    let variant = pattern.opt_text("variantName")
    let tag = e._variant_tag(pattern)

    e._get_local(slot, span)
    e._chunk().write_op(Op.MatchTypePop, span)
    if tag < 0 {
        e._chunk().write_u24(e._string(f"{owner ?? "?"}.{variant ?? "?"}"))
    } else {
        e._chunk().write_u24(e._string(owner ?? "any"))
    }
    misses.push(e._reserve_jump())

    if tag >= 0 {
        e._get_local(slot, span)
        e._op(Op.EnumTag, span)
        e._op(Op.SwapPop, span)
        e._chunk().write_op(Op.MatchConstPop, span)
        e._chunk().write_u24(e._chunk().add_constant(Const(ConstTag.Int, tag)))
        misses.push(e._reserve_jump())
    }

    for (index, element) in pattern.child_list("subPatterns").enumerate() {
        e._get_local(slot, span)
        e._op_u8(Op.EnumField, index, span)
        e._op(Op.SwapPop, span)
        _sub_pattern(e, element, misses, span)
    }
}

pub fn _test_class(e: Emitter, pattern: Node, slot: int, misses: list[int]) -> void {
    let span = pattern.span
    let sub_patterns = pattern.child_list("subPatterns")

    e._get_local(slot, span)
    e._chunk().write_op(Op.MatchTypePop, span)
    e._chunk().write_u24(e._string(pattern.opt_text("typeName") ?? "any"))
    misses.push(e._reserve_jump())
    if sub_patterns.len() == 0 { return }

    var names: list[Const] = []
    for index in range(sub_patterns.len()) {
        names.push(Const(ConstTag.Str, _field_name(pattern, index)))
    }
    e._get_local(slot, span)
    e._chunk().write_op(Op.MatchFields, span)
    e._chunk().write_u24(e._chunk().add_constant(Const(ConstTag.Tuple, names)))
    misses.push(e._reserve_jump())
    e._adjust(sub_patterns.len())

    var temps: list[int] = []
    for _index in range(sub_patterns.len()) { temps.push(0) }
    var index = sub_patterns.len() - 1
    while index >= 0 {
        temps[index] = e._alloc_temp()
        e._bind_local(temps[index], span)
        index -= 1
    }
    e._op(Op.Pop, span)

    for (position, element) in sub_patterns.enumerate() {
        _pattern_in_slot(e, element, temps[position], misses)
    }
    e._free_temps(sub_patterns.len())
}
