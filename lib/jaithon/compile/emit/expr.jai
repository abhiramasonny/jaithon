#: Every expression form: the literals and the operators, f-strings, the
#: short-circuiting pair, a comparison chain, calls in all four shapes
#: (ordinary, invoke, keyword and spread), `super`, a block or an `if` used for
#: its value, and comprehensions.
#:
#: `_items_iterable` is here rather than in `iteration.jai` because it EMITS an
#: expression -- the receiver of `X.items()` -- and both callers that want it,
#: a comprehension clause here and a `for` header there, are above it.
#:
#: The seams. Three calls run backwards out of this file: a block statement
#: (`lower_stmt`), `match` in expression position (`lower_match`, which
#: `pattern.jai` implements above), and the closure a lambda builds
#: (`lower_closure`, which `function.jai` implements above).
from jaithon.ast import CompKind, Node, NodeKind, OpKind, Span
from jaithon.compile.emit.opcode import FMT_MAX_PARTS, Op
from jaithon.compile.emit.spec import Const, ConstTag
from jaithon.compile.emit.state import Emitter, lower_closure, lower_match, lower_stmt
from jaithon.compile.emit.exception import _type_guard_const
from jaithon.compile.emit.operator import _binary_op, _unary_op
from jaithon.compile.emit.shape import _else_of, _is_items_call, _items_receiver
from jaithon.compile.emit.shape import _then_of

pub fn _each_expr(e: Emitter, nodes: list[Node]) -> void {
    for node in nodes { _expr(e, node) }
}

pub fn _expr(e: Emitter, node: Node?) -> void {
    if node == null {
        e._op(Op.Null, Span.none())
        return
    }
    let span = node.span
    match node.kind {
        NodeKind.IntLit => { e._emit_int(node.number("intLit"), span) }
        NodeKind.FloatLit => {
            e._op_k(Op.Const, e._chunk().add_constant(Const(ConstTag.Float, node.get("floatLit"))), span)
        }
        NodeKind.StrLit => { e._op_k(Op.Const, e._string(node.text("chars")), span) }
        NodeKind.BoolLit => { e._op(if node.flag("boolLit") { Op.True } else { Op.False }, span) }
        NodeKind.NullLit => { e._op(Op.Null, span) }
        NodeKind.FString => { _fstring(e, node) }
        NodeKind.Ident => { e._load_name(node, node.text("name")) }
        NodeKind.Self | NodeKind.Super => {
            let captured = e.resolution.symbol_of(node)
            if captured == null {
                e._op_u16(Op.GetLocal, 0, span)
            } else {
                e._load_symbol(captured, "self", span)
            }
        }
        NodeKind.ListLit => { _sequence_literal(e, node, Op.BuildList) }
        NodeKind.SetLit => { _sequence_literal(e, node, Op.BuildSet) }
        NodeKind.TupleLit => { _sequence_literal(e, node, Op.BuildTuple) }
        NodeKind.DictLit => {
            let keys = node.child_list("keys")
            let values = node.child_list("values")
            for (index, key) in keys.enumerate() {
                _expr(e, key)
                _expr(e, values[index])
            }
            e._op_u16(Op.BuildDict, keys.len(), span)
            e._adjust(-2 * keys.len() + 1)
        }
        NodeKind.Unary => {
            _expr(e, node.child("operand"))
            e._op(_unary_op(node.op()), span)
        }
        NodeKind.Binary => {
            _expr(e, node.child("left"))
            _expr(e, node.child("right"))
            e._op(_binary_op(node.op()), span)
        }
        NodeKind.Logical => { _logical(e, node) }
        NodeKind.CompareChain => { _chain(e, node) }
        NodeKind.Ternary | NodeKind.IfExpr => {
            _expr(e, node.child("cond"))
            let otherwise = e._emit_jump(Op.JumpIfFalse, span)
            let base = e._depth()
            _branch_value(e, _then_of(node), span)
            let done = e._emit_jump(Op.Jump, span)
            e._set_depth(base)
            e._patch(otherwise)
            _branch_value(e, _else_of(node), span)
            e._set_depth(base + 1)
            e._patch(done)
        }
        NodeKind.Coalesce => {
            _expr(e, node.child("left"))
            let absent = e._emit_jump(Op.JumpIfNull, span)
            let done = e._emit_jump(Op.Jump, span)
            let base = e._depth() - 1
            e._set_depth(base + 1)
            e._patch(absent)
            e._op(Op.Pop, span)
            _expr(e, node.child("right"))
            e._set_depth(base + 1)
            e._patch(done)
        }
        NodeKind.Call => { _call(e, node) }
        NodeKind.Index => {
            _expr(e, node.child("object"))
            if not e.resolution.is_type_args(node) {
                _expr(e, node.child("index"))
                e._op(Op.GetIndex, span)
            }
        }
        NodeKind.Slice => {
            _expr(e, node.child("object"))
            let before = e._depth()
            e._op_u8(Op.GetSlice, _slice_parts(e, node), span)
            e._set_depth(before)  # object out, one slice in
        }
        NodeKind.Member => {
            let object = node.require("object")
            if object.kind == NodeKind.Super {
                e._op_u16(Op.GetLocal, 0, span)
                e._op_k(Op.GetSuper, e._string(node.text("name")), span)
            } else {
                _expr(e, object)
                e._op_cached(Op.GetField, e._string(node.text("name")), span)
            }
        }
        NodeKind.OptMember => {
            _expr(e, node.child("object"))
            let absent = e._emit_jump(Op.JumpIfNull, span)
            e._op_cached(Op.GetField, e._string(node.text("name")), span)
            e._patch(absent)
        }
        NodeKind.Lambda | NodeKind.AnonFn => { lower_closure(e, node, null, 0) }
        NodeKind.Comprehension => { _comprehension(e, node) }
        NodeKind.Range => {
            _expr(e, node.child("start"))
            _expr(e, node.child("stop"))
            e._op_u8(Op.BuildRange, if node.flag("inclusive") { 1 } else { 0 }, span)
        }
        NodeKind.MatchExpr => { lower_match(e, node, true) }
        NodeKind.Cast => {
            _expr(e, node.child("operand"))
            if node.flag("widen") {
                e._op(Op.ToFloat, span)
            } else {
                let target = node.type_field("target")
                e._op_k(Op.TypeGuard, _type_guard_const(e, target), span)
            }
        }
        NodeKind.ThrowExpr => {
            _expr(e, node.child("operand"))
            e._op(Op.Throw, span)
        }
        NodeKind.Block => { _block_expr(e, node) }
        NodeKind.Yield => {
            e._unsupported(span, "`yield`: the instruction set has no generators")
            e._op(Op.Null, span)
        }
        NodeKind.Await => {
            e._unsupported(span, "`await`: the instruction set has no coroutines")
            e._op(Op.Null, span)
        }
        _ => {
            e._unsupported(span, f"{node.kind} in expression position")
            e._op(Op.Null, span)
        }
    }
}

pub fn _fstring(e: Emitter, node: Node) -> void {
    let span = node.span
    let parts = node.child_list("parts")
    if parts.len() == 0 {
        e._op_k(Op.Const, e._string(""), span)
        return
    }
    if parts.len() == 1 and parts[0].kind == NodeKind.StrLit {
        _expr(e, parts[0])
        return
    }

    if parts.len() <= FMT_MAX_PARTS {
        var litmask = 0
        for (index, part) in parts.enumerate() {
            if part.kind == NodeKind.StrLit { litmask = litmask | 1 << index }
            _expr(e, part)
        }
        e._chunk().write_op(Op.Format, span)
        e._chunk().write_byte(parts.len())
        e._chunk().write_u24(litmask)
        e._chunk().write_u24(e._string("str"))
        e._chunk().write_u16(e._chunk().add_cache())
        e._adjust(-parts.len() + 1)
        return
    }

    e._op_k(Op.Const, e._string(""), span)
    for part in parts { _fstring_part(e, part) }
    e._op_u16(Op.BuildList, parts.len(), span)
    e._adjust(-parts.len() + 1)
    e._invoke_name("join", 1, span)
}

pub fn _fstring_part(e: Emitter, part: Node) -> void {
    if part.kind == NodeKind.StrLit {
        _expr(e, part)
        return
    }
    e._op_cached(Op.GetGlobal, e._string("str"), part.span)
    _expr(e, part)
    e._op_u8(Op.Call, 1, part.span)
    e._adjust(-1)
}

pub fn _logical(e: Emitter, node: Node) -> void {
    _expr(e, node.child("left"))
    var keep = Op.JumpIfTrueKeep
    if node.op() == OpKind.And { keep = Op.JumpIfFalseKeep }
    let short_circuit = e._emit_jump(keep, node.span)
    e._op(Op.Pop, node.span)
    _expr(e, node.child("right"))
    e._patch(short_circuit)
}

pub fn _chain(e: Emitter, node: Node) -> void {
    let span = node.span
    let operands = node.child_list("operands")
    let ops: list[OpKind] = node.get("ops")
    if ops.len() == 0 {
        _expr(e, operands[0])
        return
    }

    let temp = if ops.len() > 1 { e._alloc_temp() } else { -1 }
    var short_circuit: list[int] = []
    let base = e._depth()

    _expr(e, operands[0])
    for (index, op) in ops.enumerate() {
        _expr(e, operands[index + 1])
        if index + 1 < ops.len() {
            e._op_u16(Op.SetLocal, temp, span)  # peeks: keeps the operand
        }
        e._op(_binary_op(op), span)
        if index + 1 < ops.len() {
            short_circuit.push(e._emit_jump(Op.JumpIfFalseKeep, span))
            e._op(Op.Pop, span)
            e._get_local(temp, span)
        }
    }

    e._set_depth(base + 1)
    e._patch_all(short_circuit)
    if temp >= 0 { e._free_temps(1) }
}

pub fn _sequence_literal(e: Emitter, node: Node, build: Op) -> void {
    let items = node.child_list("items")
    _each_expr(e, items)
    e._op_u16(build, items.len(), node.span)
    e._adjust(-items.len() + 1)
}

pub fn _slice_parts(e: Emitter, node: Node) -> int {
    var flags = 0
    if node.child("start") != null {
        _expr(e, node.child("start"))
        flags = flags | 1
    }
    if node.child("stop") != null {
        _expr(e, node.child("stop"))
        flags = flags | 2
    }
    if node.child("step") != null {
        _expr(e, node.child("step"))
        flags = flags | 4
    }
    return flags
}

pub fn _call(e: Emitter, node: Node, tail: bool = false) -> void {
    let span = node.span
    let callee = node.require("callee")
    var keywords: list[str] = []
    var positional: list[Node] = []
    var spread: Node? = null

    for argument in node.records("args") {
        let value: Node = argument["value"]
        if argument["isSpread"] {
            spread = value
        } elif argument["name"] != null {
            keywords.push(argument["name"])
        } else {
            positional.push(value)
        }
    }

    if callee.kind == NodeKind.Super {
        _super_call(e, node, "init", positional, span)
        return
    }
    if callee.kind == NodeKind.Member and callee.require("object").kind == NodeKind.Super {
        _super_call(e, node, callee.text("name"), positional, span)
        return
    }

    #: `return self.m(x)` is an INVOKE, not a tail call.
    #:
    #: `Op.TailCall` reuses the caller's frame only when the callee passes
    #: `IS_CLOSURE`, and `self.m` puts a BOUND METHOD in the callee slot --
    #: an ObjBound is not a closure, so the test fails and the call is
    #: handed straight back to the ordinary path. Checked, not assumed: a
    #: class whose method tail-calls itself two million times still dies at
    #: 1024 frames, exactly as it does with this change. So for a genuine
    #: method the tail form bought nothing and cost the compiled tier the
    #: whole enclosing body -- `self._at` lowers as a FIELD read, the
    #: function tier's `jaiClassFieldInfo` does not find a method there, and
    #: the body declines at `OP_GET_FIELD_LOCAL`. The invoke shape compiles
    #: through the direct-method path that already existed.
    #:
    #: Measured old against new, alternating, best of 30 timed runs each,
    #: under scripts/gpu_lock.sh: a method-per-iteration probe 1369ms ->
    #: 413ms (3.3x); tests/bench/json_parse, which has two `return
    #: self.method()` sites, 955ms -> 709ms (1.35x); a cold `check` of
    #: parser.jai, where the front end's own `_peek`/`_at` are the inner
    #: loop, 881ms -> 842ms (4.5%).
    #:
    #: NARROW ON PURPOSE, and narrower than `self.` alone. A member of
    #: anything else may be a module function or a static, and a member of
    #: `self` may be a callable FIELD -- `lib/std/iter/mod.jai:712`,
    #: `lib/std/core.jai:578` and `lib/jaithon/compile/check/ctx.jai:551`
    #: all hold one. Those ARE plain closures and DO reuse the frame today,
    #: so turning them into invokes would both give up real tail calls and
    #: silently change which callee a subclass field shadows. Requiring the
    #: name to be a method of the enclosing class costs nothing that exists
    #: -- an inherited method simply keeps the old shape -- and keeps the
    #: whole measured win, because every site that pays for it is a genuine
    #: method.
    #:
    #: Ruled out as the alternative: fusing the field read and the call
    #: inside the JIT instead. It needs the model to hold a receiver where
    #: the interpreter's stack holds a bound method, and every guard between
    #: the two writes that model out to a deopt record, so the record would
    #: have to rematerialise an ObjBound from a stub that allocates.
    var self_method = false
    if callee.kind == NodeKind.Member and callee.require("object").kind == NodeKind.Self {
        if e.class_methods.len() > 0 {
            let own = e.class_methods[e.class_methods.len() - 1]
            self_method = callee.text("name") in own
        }
    }
    if callee.kind == NodeKind.Member and
        keywords.len() == 0 and
        spread == null and
        (not tail or self_method) {
        _expr(e, callee.require("object"))
        _each_expr(e, positional)
        e._invoke_name(callee.text("name"), positional.len(), span)
        return
    }

    var opt_skip = -1
    var is_tail = tail
    if callee.kind == NodeKind.OptMember {
        _expr(e, callee.require("object"))
        opt_skip = e._emit_jump(Op.JumpIfNull, span)
        e._op_cached(Op.GetField, e._string(callee.text("name")), span)
        is_tail = false
    } else {
        _expr(e, callee)
    }

    _each_expr(e, positional)
    if spread != null {
        _expr(e, spread)
        e._op_u8(Op.CallSpread, positional.len() + 1, span)
        e._adjust(-(positional.len() + 1))
        if opt_skip >= 0 { e._patch(opt_skip) }
        return
    }
    if keywords.len() > 0 {
        var names: list[Const] = []
        for name in keywords { names.push(Const(ConstTag.Str, name)) }
        var keyword_count = 0
        for argument in node.records("args") {
            if argument["name"] != null {
                _expr(e, argument["value"])
                keyword_count += 1
            }
        }
        e._chunk().write_op(Op.CallKw, span)
        e._chunk().write_byte(positional.len())
        e._chunk().write_u24(e._chunk().add_constant(Const(ConstTag.Tuple, names)))
        e._adjust(-(positional.len() + keyword_count))
        if opt_skip >= 0 { e._patch(opt_skip) }
        return
    }
    e._op_u8(if is_tail { Op.TailCall } else { Op.Call }, positional.len(), span)
    e._adjust(-positional.len())
    if opt_skip >= 0 { e._patch(opt_skip) }
}

pub fn _super_call(
    e: Emitter,
    node: Node,
    method: str,
    positional: list[Node],
    span: Span
) -> void {
    e._op_u16(Op.GetLocal, 0, span)
    _each_expr(e, positional)
    e._chunk().write_op(Op.SuperInvoke, span)
    e._chunk().write_u24(e._string(method))
    e._chunk().write_byte(positional.len())
    e._adjust(-positional.len())
}

pub fn _block_expr(e: Emitter, node: Node) -> void {
    let close_at = e.resolution.capture_base_of(node)
    let saved = e._enter_close_scope(close_at)
    _block_expr_contents(e, node)
    e._close_scope(close_at, node.span)
    e._ctx().close_base = saved
}

pub fn _block_expr_contents(e: Emitter, node: Node) -> void {
    let stmts = node.child_list("stmts")
    if stmts.len() == 0 {
        e._op(Op.Null, node.span)
        return
    }
    var index = 0
    while index < stmts.len() - 1 {
        lower_stmt(e, stmts[index])
        index += 1
    }
    let last = stmts[stmts.len() - 1]
    if last.kind == NodeKind.ExprStmt {
        _expr(e, last.child("expr"))
        return
    }
    if last.kind == NodeKind.Match or last.kind == NodeKind.If and last.child("elseBranch") != null {
        _branch_value(e, last, last.span)
        return
    }
    lower_stmt(e, last)
    e._op(Op.Null, node.span)
}

pub fn _branch_value(e: Emitter, branch: Node?, span: Span) -> void {
    if branch == null {
        e._op(Op.Null, span)
        return
    }
    if branch.kind == NodeKind.Block {
        _block_expr(e, branch)
        return
    }
    if branch.kind == NodeKind.Match {
        lower_match(e, branch, true)
        return
    }
    if branch.kind == NodeKind.If {
        let base = e._depth()
        _expr(e, branch.child("cond"))
        let otherwise = e._emit_jump(Op.JumpIfFalse, span)
        _branch_value(e, branch.child("thenBranch"), span)
        let done = e._emit_jump(Op.Jump, span)
        e._set_depth(base)
        e._patch(otherwise)
        _branch_value(e, branch.child("elseBranch"), span)
        e._set_depth(base + 1)
        e._patch(done)
        return
    }
    _expr(e, branch)
}

pub fn _comprehension(e: Emitter, node: Node) -> void {
    let span = node.span
    let clauses = node.records("clauses")
    let kind: CompKind = node.get("compKind")
    if kind == CompKind.Generator {
        e._unsupported(span, "a generator expression: it needs a coroutine")
    }
    if kind == CompKind.Dict {
        e._op_u16(Op.BuildDict, 0, span)
    } elif kind == CompKind.Set {
        e._op_u16(Op.BuildSet, 0, span)
    } else {
        e._op_u16(Op.BuildList, 0, span)
    }
    e._adjust(1)
    let container_depth = e._depth() - 1

    var starts: list[int] = []
    var exits: list[int] = []
    var before_iter: list[int] = []
    var skips: list[list[int]] = []
    var is_range: list[bool] = []
    var range_temps: list[int] = []
    for clause in clauses {
        before_iter.push(e._depth())
        let iterable: Node = clause["iterable"]
        let pattern: Node = clause["pattern"]
        if iterable.kind == NodeKind.Range and e._range_bind_slot(pattern, false) >= 0 {
            is_range.push(true)
            let wildcard = pattern.kind == NodeKind.PatWildcard
            let counter = e._alloc_temp()
            let end = e._alloc_temp()
            let slot = e._range_bind_slot(pattern, wildcard)
            _expr(e, iterable.child("start"))
            _expr(e, iterable.child("stop"))
            e._chunk().write_op(Op.IterRange, span)
            e._chunk().write_byte(if iterable.flag("inclusive") { 1 } else { 0 })
            e._chunk().write_u16(counter)
            e._chunk().write_u16(end)
            starts.push(e._chunk().here())
            e._chunk().write_op(Op.ForRangeBind, span)
            exits.push(e._chunk().here())
            e._chunk().write_u16(0)
            e._chunk().write_u16(slot)
            e._chunk().write_u16(counter)
            e._chunk().write_u16(end)
            range_temps.push(if wildcard { 3 } else { 2 })
        } else {
            is_range.push(false)
            range_temps.push(0)
            if _is_items_call(iterable) {
                _items_iterable(e, iterable, span)
            } else {
                _expr(e, iterable)
                e._op(Op.GetIter, span)
            }
            starts.push(e._chunk().here())
            exits.push(e._emit_jump(Op.ForIter, span))
            e._adjust(1)
            e._bind_pattern(pattern, span)
        }
        var failures: list[int] = []
        let conditions: list[Node] = clause["conditions"]
        for condition in conditions {
            _expr(e, condition)
            failures.push(e._emit_jump(Op.JumpIfFalse, span))
        }
        skips.push(failures)
    }

    if kind == CompKind.Dict {
        _expr(e, node.child("keyExpr"))
        _expr(e, node.child("element"))
        e._op_u16(Op.DictInsert, e._depth() - 1 - container_depth, span)
    } elif kind == CompKind.Set {
        _expr(e, node.child("element"))
        e._op_u16(Op.SetAdd, e._depth() - 1 - container_depth, span)
    } else {
        _expr(e, node.child("element"))
        e._op_u16(Op.ListAppend, e._depth() - 1 - container_depth, span)
    }

    var index = clauses.len() - 1
    while index >= 0 {
        e._patch_all(skips[index])
        e._loop_back(starts[index], span)
        e._set_depth(before_iter[index])
        if is_range[index] {
            e._patch_from(exits[index], exits[index] + 8)
            e._free_temps(range_temps[index])
        } else {
            e._patch(exits[index])
        }
        index -= 1
    }
}

#: `X.items()` as a loop iterable: take the lazy path when X turns out to
#: be a dict, and otherwise fall through to the ordinary call so a user
#: class's own `items()` still runs. The branch is the opcode's, because
#: this file cannot know X's type — see `_is_items_call`.
pub fn _items_iterable(e: Emitter, iterable: Node?, span: Span) -> void {
    _expr(e, _items_receiver(iterable))
    let skip = e._emit_jump(Op.GetIterItems, span)
    e._invoke_name("items", 0, span)
    e._op(Op.GetIter, span)
    e._patch(skip)
}
