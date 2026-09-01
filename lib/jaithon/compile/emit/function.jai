#: A function value: the `FuncProto` constant its body compiles into, the
#: `OP_CLOSURE` that captures the upvalues the resolver recorded, the thunks a
#: default argument is evaluated by, and the field initialisers a class's
#: `init` runs ahead of its body.
#:
#: `_function_constant` pushes the new `FnCtx` and `_finish` pops it, which is
#: where a nested function's frame window is decided. The one other place that
#: pushes an `FnCtx` is `_stmt`'s `defer`, which builds a thunk rather than a
#: function.
#:
#: The seam. A function body is a statement, so this file calls `lower_stmt`.
from jaithon.ast import Node, NodeKind, Span
from jaithon.compile.emit.opcode import Op
from jaithon.compile.emit.spec import Const, ConstTag, FN_ASYNC, FN_GENERATOR
from jaithon.compile.emit.spec import FN_GPU_KERNEL, FN_KWREST, FN_STATIC
from jaithon.compile.emit.spec import FN_TRACE, FN_VARIADIC
from jaithon.compile.emit.chunk import FuncProto
from jaithon.compile.emit.state import Emitter, FnCtx, lower_stmt
from jaithon.compile.emit.expr import _expr

pub fn _closure(e: Emitter, node: Node, name: str?, extra_flags: int) -> int {
    let index = _function_constant(e, node, name, extra_flags)
    _emit_closure(e, node, index)
    return index
}

pub fn _emit_closure(e: Emitter, node: Node, index: int) -> void {
    e._op_k(Op.Closure, index, node.span)
    let scope = e.resolution.scope_of(node)
    if scope == null { return }
    for upvalue in scope.upvalues {
        var how = if upvalue.is_local { 1 } else { 0 }
        if upvalue.by_value { how = how | 2 }
        e._chunk().write_byte(how)
        e._chunk().write_u16(upvalue.index)
    }
}

pub fn _function_constant(
    e: Emitter,
    node: Node,
    name: str?,
    extra_flags: int,
    init_fields: list[any]? = null
) -> int {
    let scope = e.resolution.scope_of(node)
    var arity = 0
    var flags = extra_flags
    if node.flag("isStatic") { flags = flags | FN_STATIC }
    if node.flag("isGenerator") { flags = flags | FN_GENERATOR }
    if node.flag("isAsync") { flags = flags | FN_ASYNC }
    if node.has("decorators") {
        let decorators = node.get("decorators")
        if decorators is not null {
            for decorator in decorators {
                if decorator == "trace" { flags = flags | FN_TRACE }
                if decorator == "gpu_kernel" or decorator == "mps_graph" {
                    flags = flags | FN_GPU_KERNEL
                }
            }
        }
    }

    var names: list[str] = []
    var defaults: list[Node] = []
    for param in node.records("params") {
        if param["name"] == "self" { continue }
        names.push(param["name"])
        if param["isVariadic"] {
            flags = flags | FN_VARIADIC
        } elif param["isKwRest"] {
            flags = flags | FN_KWREST
        } else {
            arity += 1
            if param["defaultValue"] != null { defaults.push(param["defaultValue"]) }
        }
    }

    let proto = FuncProto((name ?? node.opt_text("name")) ?? "<anonymous>", arity, flags)
    proto.upvalue_count = if scope == null { 0 } else { scope.upvalues.len() }
    proto.param_names = names

    var locals = e.resolution.slot_count(node)
    if locals < node.records("params").len() + 1 {
        locals = node.records("params").len() + 1
    }
    e.stack.push(FnCtx(proto, locals))

    _default_thunks(e, proto, defaults, node)
    if init_fields != null { _field_defaults(e, init_fields) }

    let body = node.child("body")
    if body == null {
        e._op(Op.ReturnNull, node.span)
    } elif node.flag("isExprBody") or body.kind != NodeKind.Block {
        _expr(e, body)
        e._run_defers(node.span)
        e._op(Op.Return, node.span)
    } else {
        lower_stmt(e, body)
        e._run_defers(node.span)
        e._op(Op.ReturnNull, node.span)
    }

    e._finish()
    return e._chunk().add_constant(Const(ConstTag.Func, proto))
}

pub fn _default_thunks(e: Emitter, proto: FuncProto, defaults: list[Node], node: Node) -> void {
    if defaults.len() == 0 { return }
    let to_body = e._emit_jump(Op.Jump, node.span)
    for default_value in defaults {
        proto.default_offsets.push(e._chunk().here())
        let depth = e._depth()
        _expr(e, default_value)
        e._op(Op.Return, default_value.span)
        e._set_depth(depth)
    }
    e._patch(to_body)
}

pub fn _field_defaults(e: Emitter, fields: list[any]) -> void {
    for field in fields {
        if field["defaultValue"] == null or field["isStatic"] { continue }
        let field_span: Span = field["span"]
        e._op_u16(Op.GetLocal, 0, field_span)
        _expr(e, field["defaultValue"])
        e._op_cached(Op.SetField, e._string(field["name"]), field_span)
    }
}
