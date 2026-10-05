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
from jaithon.compile.emit.opcode import Op, opcode
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
    let arms = node.records("arms")
    var index = 0
    while index < arms.len() {
        let stop = _enum_run_end(e, arms, index)
        if stop > index {
            _enum_run(e, arms, index, stop, subject, base, as_expression, done)
            index = stop
            continue
        }
        let arm = arms[index]
        let arm_span: Span = arm["span"]
        var misses: list[int] = []
        var guard_miss: list[int] = []

        e._set_depth(base)
        _pattern_in_slot(e, arm["pattern"], subject, misses)
        _arm_rest(e, arm, as_expression, guard_miss, done)

        # A pattern miss arrives with the failed candidate still pushed.
        e._set_depth(base + 1)
        e._patch_all(misses)
        e._op(Op.Pop, arm_span)
        e._set_depth(base)
        e._patch_all(guard_miss)
        index += 1
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

#: An arm's guard and body, once its pattern has matched: a failed guard
#: adds to `guard_miss`, and the body ends in a jump added to `done`.
fn _arm_rest(
    e: Emitter,
    arm: dict[str, any],
    as_expression: bool,
    guard_miss: list[int],
    done: list[int]
) -> void {
    let arm_span: Span = arm["span"]
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
}

#: The enum an arm's pattern names when it is a payload-free variant of an
#: enum the emitter can see -- `Op.Add`, or `Op.Add | Op.Sub` -- and null for
#: every other pattern.
fn _enum_owner(e: Emitter, pattern: Node?) -> str? {
    if pattern == null { return null }
    if pattern.kind == NodeKind.PatOr {
        let alternatives = pattern.child_list("elems")
        if alternatives.len() == 0 { return null }
        return _enum_owner(e, alternatives[0])
    }
    if pattern.kind != NodeKind.PatEnum { return null }
    if pattern.child_list("subPatterns").len() > 0 { return null }
    if e._variant_tag(pattern) < 0 { return null }
    return pattern.opt_text("typeName")
}

#: Appends the tags `pattern` accepts, in source order, and answers true when
#: every alternative in it is a payload-free variant of `owner`. On false,
#: `tags` may be left half filled.
fn _enum_tags(e: Emitter, pattern: Node?, owner: str, tags: list[int]) -> bool {
    if pattern == null { return false }
    if pattern.kind == NodeKind.PatOr {
        let alternatives = pattern.child_list("elems")
        if alternatives.len() == 0 { return false }
        for alternative in alternatives {
            if not _enum_tags(e, alternative, owner, tags) { return false }
        }
        return true
    }
    if pattern.kind != NodeKind.PatEnum { return false }
    if pattern.child_list("subPatterns").len() > 0 { return false }
    if pattern.opt_text("typeName") != owner { return false }
    let tag = e._variant_tag(pattern)
    if tag < 0 { return false }
    tags.push(tag)
    return true
}

#: Where a run of enum-constant arms starting at `start` ends: one past the
#: last consecutive arm whose pattern is only variants of the first arm's
#: enum. `start` itself when the run would test fewer than two tags, which
#: the per-arm lowering does as cheaply.
fn _enum_run_end(e: Emitter, arms: list[dict[str, any]], start: int) -> int {
    let owner = _enum_owner(e, arms[start]["pattern"])
    if owner == null { return start }
    var stop = start
    var count = 0
    while stop < arms.len() {
        var tags: list[int] = []
        if not _enum_tags(e, arms[stop]["pattern"], owner, tags) { break }
        count += tags.len()
        stop += 1
    }
    if count < 2 { return start }
    return stop
}

#: A run of arms that only test variants of one enum, lowered to ONE type
#: test and ONE tag read into a frame slot, then a single fused
#: compare-and-branch per alternative -- where the per-arm lowering spends
#: seven instructions on every alternative it rejects.
#:
#: Arm order, guards and the arms after the run are unchanged: a failed guard
#: falls to the next arm's test exactly as before, and a subject that is not
#: this enum skips the whole run, which is what each of its arms would have
#: done one at a time. Every miss edge between arms lands with the operand
#: stack at `base`, because the tag lives in a slot rather than on the stack.
fn _enum_run(
    e: Emitter,
    arms: list[dict[str, any]],
    start: int,
    stop: int,
    subject: int,
    base: int,
    as_expression: bool,
    done: list[int]
) -> void {
    let first: Node = arms[start]["pattern"]
    let owner = _enum_owner(e, first) ?? "any"
    let span = first.span

    e._set_depth(base)
    e._get_local(subject, span)
    e._chunk().write_op(Op.MatchTypePop, span)
    e._chunk().write_u24(e._string(owner))
    let not_enum = e._reserve_jump()
    e._get_local(subject, span)
    e._op(Op.EnumTag, span)
    e._op(Op.SwapPop, span)
    let tag_slot = e._alloc_temp()
    e._bind_local(tag_slot, span)

    var misses: list[int] = []
    for index in range(start, stop) {
        let arm = arms[index]
        let arm_span: Span = arm["span"]
        e._set_depth(base)
        e._patch_all(misses)
        misses = []

        var tags: list[int] = []
        _enum_tags(e, arm["pattern"], owner, tags)
        var matched: list[int] = []
        for (position, tag) in tags.enumerate() {
            let last = position == tags.len() - 1
            # Jumps when the comparison is FALSE: `!=` on the way to the
            # body, `==` on the way to the next arm.
            e._chunk().write_op(Op.JumpIfCmpLocalK, arm_span)
            e._chunk().write_byte(opcode(if last { Op.Eq } else { Op.Ne }))
            e._chunk().write_u16(tag_slot)
            e._chunk().write_u24(e._chunk().add_constant(Const(ConstTag.Int, tag)))
            if last {
                misses.push(e._reserve_jump())
            } else {
                matched.push(e._reserve_jump())
            }
        }
        e._patch_all(matched)
        _arm_rest(e, arm, as_expression, misses, done)
    }

    # The last arm's misses read the subject back, only so that the slot is
    # still live after every tag test: slot coalescing would otherwise hand
    # the subject's slot to the tag, and one slot holding an enum and then
    # an int is a kind clash the compiled tier pays for on every read. The
    # pop it shares with `not_enum` -- the subject MATCH_TYPE_POP left
    # behind -- is a branch target, so the peephole keeps the pair.
    e._set_depth(base)
    e._patch_all(misses)
    e._get_local(subject, span)
    e._patch(not_enum)
    e._op(Op.Pop, span)
    e._set_depth(base)
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
