#: The statement dispatch: one arm per `NodeKind` that can stand as a
#: statement, plus assignment in its five target shapes and `from ... import`.
#:
#: This is the top of `emit/`. Every other production file is below it, which
#: is why it needs no seam of its own -- and why the seam the others use is
#: `_stmt` from here.
#:
#: `while` and `loop` are lowered inline rather than in `iteration.jai`: both
#: are a jump, a body and a jump back, and the frame they push is `Emitter`'s
#: `_enter_loop`. `for` is not, because the range form rewrites the whole
#: header.
from jaithon.ast import Node, NodeKind, Span, Visibility
from jaithon.compile.emit.opcode import Op
from jaithon.compile.emit.spec import Const, ConstTag
from jaithon.compile.emit.chunk import FuncProto
from jaithon.compile.emit.state import Emitter, FnCtx
from jaithon.compile.emit.exception import _finally_chain_for_return, _try
from jaithon.compile.emit.expr import _call, _expr, _slice_parts
from jaithon.compile.emit.function import _closure
from jaithon.compile.emit.decl import _class_decl, _define, _enum_decl, _trait_decl
from jaithon.compile.emit.pattern import _match
from jaithon.compile.emit.iteration import _body_of_loop, _for, _jump
from jaithon.compile.emit.field_kind import _container_kind_code
from jaithon.compile.emit.operator import _binary_op
from jaithon.compile.emit.assert_text import _assert_message
from jaithon.compile.emit.shape import _last_segment

pub fn _stmt(e: Emitter, node: Node?) -> void {
    if node == null { return }
    let span = node.span
    match node.kind {
        NodeKind.ExprStmt => {
            _expr(e, node.child("expr"))
            e._op(Op.Pop, span)
        }
        NodeKind.VarDecl => {
            _expr(e, node.child("init"))
            let pattern = node.child("pattern")
            if pattern != null and pattern.kind == NodeKind.PatBind {
                let packed = _container_kind_code(node.type_field("declaredType"))
                if packed != 0 {
                    e._chunk().write_op(Op.ElemKind, span)
                    e._chunk().write_byte(packed)
                }
            }
            e._bind_pattern(pattern, span)
            let symbol = e.resolution.symbol_of(node)
            if symbol != null and
                symbol.visibility == Visibility.Public and
                pattern != null and
                pattern.kind == NodeKind.PatBind {
                e._op_k(Op.Export, e._string(symbol.name), span)
            }
        }
        NodeKind.Assign => { _assign(e, node) }
        NodeKind.Block => {
            let close_at = e.resolution.capture_base_of(node)
            let saved = e._enter_close_scope(close_at)
            for statement in node.child_list("stmts") { _stmt(e, statement) }
            e._close_scope(close_at, span)
            e._ctx().close_base = saved
        }
        NodeKind.If => {
            let base = e._depth()
            _expr(e, node.child("cond"))
            let otherwise = e._emit_jump(Op.JumpIfFalse, span)
            _stmt(e, node.child("thenBranch"))
            if node.child("elseBranch") == null {
                e._set_depth(base)
                e._patch(otherwise)
            } else {
                let done = e._emit_jump(Op.Jump, span)
                e._set_depth(base)
                e._patch(otherwise)
                _stmt(e, node.child("elseBranch"))
                e._set_depth(base)
                e._patch(done)
            }
        }
        NodeKind.While => {
            let base = e._depth()
            let start = e._chunk().here()
            _expr(e, node.child("cond"))
            let exit = e._emit_jump(Op.JumpIfFalse, span)
            _body_of_loop(e, node, start, base, base)
            e._loop_back(start, span)
            e._set_depth(base)
            e._patch(exit)
            e._close_loop()
        }
        NodeKind.Loop => {
            let base = e._depth()
            let start = e._chunk().here()
            _body_of_loop(e, node, start, base, base)
            e._loop_back(start, span)
            e._close_loop()
        }
        NodeKind.For => { _for(e, node) }
        NodeKind.Match => { _match(e, node, false) }
        NodeKind.Break | NodeKind.Continue => { _jump(e, node) }
        NodeKind.Return => {
            let value = node.child("value")
            if value == null {
                e._close_scope(e._return_close_base(), span)
                _finally_chain_for_return(e)
                e._run_defers(span)
                e._op(Op.ReturnNull, span)
            } elif e._can_tail_call(value) {
                _call(e, value, true)
                e._op(Op.Return, span)
            } else {
                _expr(e, value)
                e._close_scope(e._return_close_base(), span)
                _finally_chain_for_return(e)
                e._run_defers(span)
                e._op(Op.Return, span)
            }
        }
        NodeKind.Throw => {
            _expr(e, node.child("value"))
            e._op(Op.Throw, span)
        }
        NodeKind.Try => { _try(e, node) }
        NodeKind.Defer => {
            let parent = e._ctx()
            let proto = FuncProto("defer", 0, 0)
            proto.upvalue_count = parent.proto.upvalue_count
            e.stack.push(FnCtx(proto, parent.next_temp))
            e._ctx().max_slot = parent.max_slot
            e._ctx().protect_depth = 1
            _stmt(e, node.child("body"))
            e._op(Op.ReturnNull, span)
            parent.proto.chunk.raise_max_depth(proto.chunk.max_depth)
            e._finish()
            e._op_k(Op.PushDefer, e._chunk().add_constant(Const(ConstTag.Func, proto)), span)
            parent.has_defer = true
        }
        NodeKind.Assert => {
            if e.strip_asserts { return }
            _expr(e, node.child("cond"))
            let ok = e._emit_jump(Op.JumpIfTrue, span)
            e._op_k(Op.AssertFail, e._string(_assert_message(node, e.source)), span)
            e._patch(ok)
        }
        NodeKind.FnDecl => {
            _closure(e, node, node.opt_text("name"), 0)
            _define(e, node, node.opt_text("name"), span)
        }
        NodeKind.ClassDecl => { _class_decl(e, node) }
        NodeKind.EnumDecl => { _enum_decl(e, node) }
        NodeKind.TraitDecl => { _trait_decl(e, node) }
        NodeKind.Import => {
            e._op_k(Op.Import, e._string(node.text("path")), span)
            _define(e, node, _last_segment(node.opt_text("alias") ?? node.text("path")), span)
        }
        NodeKind.FromImport => { _from_import(e, node) }
        NodeKind.Export => {
            let names: list[str?] = node.get("names")
            for name in names {
                if name != null { e._op_k(Op.Export, e._string(name), span) }
            }
        }
        # Aliases erase at use sites (spec §2.3), but `from M import Alias`
        # resolves against M's globals/exports at run time, so a public
        # alias still needs a binding here or import fails with E0802.
        NodeKind.TypeDecl => {
            if node.visibility() == Visibility.Public {
                let index = e._string(node.text("name"))
                e._op(Op.Null, span)
                e._op_k(Op.DefGlobal, index, span)
                e._op_k(Op.Export, index, span)
            }
        }
        NodeKind.ModuleDecl => { }
        _ => {
            _expr(e, node)
            e._op(Op.Pop, span)
        }
    }
}

pub fn _assign(e: Emitter, node: Node) -> void {
    let span = node.span
    let target = node.require("target")
    let value = node.child("value")
    let compound = node.flag("isCompound")

    if target.kind == NodeKind.Ident {
        if compound {
            e._load_name(target, target.text("name"))
            _expr(e, value)
            e._op(_binary_op(node.op()), span)
        } else {
            _expr(e, value)
        }
        e._store_symbol(e.resolution.symbol_of(target), target.text("name"), span, false)
    } elif target.kind == NodeKind.Member {
        let field = e._string(target.text("name"))
        _expr(e, target.child("object"))
        if compound {
            e._op(Op.Dup, span)
            e._op_cached(Op.GetField, field, span)
            _expr(e, value)
            e._op(_binary_op(node.op()), span)
        } else {
            _expr(e, value)
        }
        e._op_cached(Op.SetField, field, span)
    } elif target.kind == NodeKind.Index {
        _expr(e, target.child("object"))
        _expr(e, target.child("index"))
        if compound {
            e._op(Op.Dup2, span)
            e._op(Op.GetIndex, span)
            _expr(e, value)
            e._op(_binary_op(node.op()), span)
        } else {
            _expr(e, value)
        }
        e._op(Op.SetIndex, span)
    } elif target.kind == NodeKind.Slice {
        _expr(e, target.child("object"))
        let before = e._depth()
        let flags = _slice_parts(e, target)
        _expr(e, value)
        e._op_u8(Op.SetSlice, flags, span)
        e._set_depth(before - 1)  # object, bounds and value all go
    } else {
        _expr(e, value)
        e._bind_pattern(target, span)
    }
}

pub fn _from_import(e: Emitter, node: Node) -> void {
    let span = node.span
    e._op_k(Op.Import, e._string(node.text("path")), span)
    if node.flag("isWildcard") {
        e._op(Op.Pop, span)
        e._unsupported(span, "`import *`: the loader needs an explicit name list")
        return
    }
    for item in node.records("items") {
        let item_span: Span = item["span"]
        e._op_k(Op.ImportFrom, e._string(item["name"]), item_span)
        let bound: str = item["alias"] ?? item["name"]
        e._op_k(Op.DefGlobal, e._string(bound), item_span)
    }
    e._op(Op.Pop, span)
}
