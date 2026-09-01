#: `try`, its handlers and its `finally`: the exception table entries a static
#: handler needs, the run-time dispatch a dynamic one needs, and the copies of
#: a `finally` block that an explicit `return`, `break` or `continue` has to
#: inline on its way out.
#:
#: The two `_finally_chain_*` functions live here rather than beside the exits
#: that call them because what they walk -- `FnCtx.finallys` and the loop depth
#: each was entered at -- is this file's bookkeeping, pushed and popped by
#: `_try`. `iteration.jai` and `stmt.jai` sit above and reach them directly.
#:
#: The seam. Every handler body is a nested statement, so this file calls
#: `lower_stmt`; `stmt.jai` is the last module of `emit/` and cannot be
#: imported from here.
from jaithon.ast import Node, Span, TypeExpr, TypeKind
from jaithon.compile.emit.opcode import Op
from jaithon.compile.emit.spec import CATCH_ALL, Const, ConstTag, ExceptionEntry
from jaithon.compile.emit.state import Emitter, lower_stmt

pub fn _finally_chain_for_loop(e: Emitter, index: int) -> void {
    let ctx = e._ctx()
    var position = ctx.finallys.len() - 1
    while position >= 0 and ctx.finally_loop_depth[position] > index + 1 {
        lower_stmt(e, ctx.finallys[position])
        position -= 1
    }
}

pub fn _finally_chain_for_return(e: Emitter) -> void {
    let ctx = e._ctx()
    var position = ctx.finallys.len() - 1
    while position >= 0 {
        lower_stmt(e, ctx.finallys[position])
        position -= 1
    }
}

pub fn _push_handler(e: Emitter, type_const: int, span: Span) -> int {
    e._chunk().write_op(Op.PushHandler, span)
    let operand = e._chunk().here()
    e._chunk().write_u16(0)
    e._chunk().write_u24(type_const)
    return operand
}

pub fn _type_guard_const(e: Emitter, caught: TypeExpr?) -> int {
    if caught is not null and caught.kind == TypeKind.Union {
        var names: list[Const] = []
        for member in caught.args {
            names.push(Const(ConstTag.Str, member.name ?? "any"))
        }
        return e._chunk().add_constant(Const(ConstTag.Tuple, names))
    }
    return e._string(caught?.name ?? "any")
}

pub fn _try(e: Emitter, node: Node) -> void {
    let span = node.span
    let ctx = e._ctx()
    let finally_block = node.child("finallyBlock")
    let catches = node.records("catches")
    let base = e._depth()

    let dynamic = catches.len() > 0 and (base > 0 or finally_block != null)

    var finally_entry = -1
    if finally_block != null {
        finally_entry = e._emit_jump(Op.PushFinally, span)
        ctx.finallys.push(finally_block)
        ctx.finally_loop_depth.push(ctx.loops.len())
    }
    var dispatch_entry = -1
    if dynamic { dispatch_entry = _push_handler(e, e._string("any"), span) }

    let saved_deep = ctx.deep_close_base
    ctx.deep_close_base = -1

    let guarded_start = e._chunk().here()
    ctx.protect_depth += 1
    lower_stmt(e, node.child("body"))
    ctx.protect_depth -= 1
    if dynamic { e._op(Op.PopHandler, span) }
    let guarded_end = e._chunk().here()
    let body_deep = ctx.deep_close_base

    var done: list[int] = [e._emit_jump(Op.Jump, span)]

    if dynamic {
        e._patch_from(dispatch_entry, dispatch_entry + 5)
        e._close_scope(body_deep, span)
        _catch_dispatch(e, catches, base, done, span)
    } else {
        for handler in catches {
            let handler_span: Span = handler["span"]
            let entry = e._chunk().here()
            e._set_depth(base)
            e._close_scope(body_deep, handler_span)

            let types: list[TypeExpr] = handler["types"]
            if types.len() == 0 {
                e._chunk().exceptions.push(
                    ExceptionEntry(guarded_start, guarded_end, entry, CATCH_ALL)
                )
            } else {
                for caught in types {
                    e._chunk().exceptions.push(
                        ExceptionEntry(
                            guarded_start,
                            guarded_end,
                            entry,
                            _type_guard_const(e, caught)
                        )
                    )
                }
            }
            _catch_body(e, handler, handler_span)
            done.push(e._emit_jump(Op.Jump, handler_span))
        }
    }

    let catches_end = e._chunk().here()

    if finally_block != null {
        ctx.finallys.pop()
        ctx.finally_loop_depth.pop()

        e._set_depth(base)
        e._patch_all(done)
        e._patch(finally_entry)

        e._chunk().exceptions.push(
            ExceptionEntry(guarded_start, catches_end, e._chunk().here(), CATCH_ALL)
        )
        e._close_scope(ctx.deep_close_base, span)

        ctx.protect_depth += 1
        lower_stmt(e, finally_block)
        ctx.protect_depth -= 1
        e._op(Op.EndFinally, span)
    } else {
        e._set_depth(base)
        e._patch_all(done)
    }

    if saved_deep >= 0 and (ctx.deep_close_base < 0 or saved_deep < ctx.deep_close_base) {
        ctx.deep_close_base = saved_deep
    }
    e._set_depth(base)
}

pub fn _catch_dispatch(
    e: Emitter,
    catches: list[any],
    base: int,
    done: list[int],
    span: Span
) -> void {
    var entries: list[list[int]] = []
    e._set_depth(base)
    for handler in catches {
        let handler_span: Span = handler["span"]
        var hits: list[int] = []
        let types: list[TypeExpr] = handler["types"]
        if types.len() == 0 {
            hits.push(e._emit_jump(Op.Jump, handler_span))
        } else {
            for caught in types {
                e._op_k(Op.MatchExc, _type_guard_const(e, caught), handler_span)
                hits.push(e._emit_jump(Op.JumpIfTrue, handler_span))
            }
        }
        entries.push(hits)
    }
    e._op(Op.Reraise, span)

    for (index, handler) in catches.enumerate() {
        let handler_span: Span = handler["span"]
        e._set_depth(base)
        e._patch_all(entries[index])
        _catch_body(e, handler, handler_span)
        done.push(e._emit_jump(Op.Jump, handler_span))
    }
}

pub fn _catch_body(e: Emitter, handler: any, handler_span: Span) -> void {
    e._op(Op.GetExc, handler_span)
    let symbol = e.resolution.symbol_of(handler)
    if symbol != null or handler["name"] != null {
        e._store_symbol(symbol, handler["name"] ?? "_", handler_span, symbol == null)
    } else {
        e._op(Op.Pop, handler_span)
    }
    let ctx = e._ctx()
    ctx.protect_depth += 1
    lower_stmt(e, handler["body"])
    ctx.protect_depth -= 1
}
