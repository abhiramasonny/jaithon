#: The three declarations that build a run-time type -- a class, an enum and a
#: trait -- and the binding step every declaration ends with.
#:
#: All three share `_method` here and `Emitter`'s `_class_spec`, which is why
#: they are one file: an enum is a class whose spec carries variants, and a
#: trait is a class whose spec carries the methods an implementor must supply.
#:
#: No seam. Everything a declaration lowers -- a field default, a method body's
#: `FuncProto` -- is reached through `expr.jai` and `function.jai`, both below.
from jaithon.ast import Node, NodeKind, Span, Visibility
from jaithon.compile.emit.opcode import Op
from jaithon.compile.emit.spec import Const, ConstTag, FN_ASYNC, FN_GENERATOR
from jaithon.compile.emit.spec import FN_GETTER, FN_METHOD, FN_SETTER, FN_STATIC
from jaithon.compile.emit.spec import SPEC_CLASS, SPEC_ENUM, SPEC_TRAIT
from jaithon.compile.emit.state import Emitter
from jaithon.compile.emit.expr import _expr
from jaithon.compile.emit.function import _emit_closure, _function_constant
from jaithon.compile.emit.field_kind import _field_kind_code, _visibility_code

pub fn _class_decl(e: Emitter, node: Node) -> void {
    let span = node.span
    let name = node.text("name")
    let superclass = node.type_field("superclass")
    let super_name = superclass?.name

    if super_name != null {
        e._op_cached(Op.GetGlobal, e._string(super_name), span)
    }
    e._op_k(
        Op.Class,
        e._class_spec(SPEC_CLASS, name, super_name, node.flag("isAbstract"), [], []),
        span
    )
    if super_name != null {
        e._op(Op.Inherit, span)
        e._op(Op.SwapPop, span)
    }

    for trait_type in node.type_list("traits") {
        e._op_k(Op.ImplTrait, e._string(trait_type.name ?? ""), span)
    }

    var type_params: list[str] = []
    for generic in node.records("generics") {
        let generic_name = generic["name"]
        if generic_name != null { type_params.push(generic_name) }
    }

    let fields = node.records("fields")
    for field in fields {
        var bits = _visibility_code(field["visibility"]) & 0x3
        if field["isStatic"] { bits = bits | 4 }
        if field["isLet"] { bits = bits | 8 }
        bits = bits | (_field_kind_code(field["type"], type_params) & 0xF) << 4
        e._chunk().write_op(Op.FieldDef, field["span"])
        e._chunk().write_u24(e._string(field["name"]))
        e._chunk().write_byte(bits)
    }

    for field in fields {
        if not field["isStatic"] or field["defaultValue"] == null { continue }
        let field_span: Span = field["span"]
        e._op(Op.Dup, field_span)
        _expr(e, field["defaultValue"])
        e._op_cached(Op.SetField, e._string(field["name"]), field_span)
    }

    var init: Node? = null
    for method in node.child_list("methods") {
        if method.opt_text("name") == "init" {
            init = method
            break
        }
    }
    var needs_init = init == null
    if needs_init {
        needs_init = false
        for field in fields {
            if field["defaultValue"] != null and not field["isStatic"] {
                needs_init = true
                break
            }
        }
    }

    var own_methods: list[str] = []
    for method in node.child_list("methods") {
        let method_name = method.opt_text("name")
        if method_name != null { own_methods.push(method_name) }
    }
    e.class_methods.push(own_methods)
    for method in node.child_list("methods") {
        let own = if method is init { fields } else { null }
        _method(e, method, FN_METHOD, own)
    }
    for getter in node.child_list("getters") {
        _method(e, getter, FN_METHOD | FN_GETTER, null)
    }
    for setter in node.child_list("setters") {
        _method(e, setter, FN_METHOD | FN_SETTER, null)
    }
    e.class_methods.pop()

    if needs_init {
        let synthetic = Node(NodeKind.FnDecl, {"name": "init"}, span)
        let index = _function_constant(e, synthetic, "init", FN_METHOD, fields)
        e._op_k(Op.Closure, index, span)
        e._chunk().write_op(Op.Method, span)
        e._chunk().write_u24(e._string("init"))
        e._chunk().write_byte(FN_METHOD)
        e._chunk().write_byte(_visibility_code(Visibility.Public))
    }

    _define(e, node, name, span)
}

pub fn _method(e: Emitter, method: Node, flags: int, init_fields: list[any]?) -> void {
    let span = method.span
    var combined = flags
    if method.flag("isStatic") { combined = combined | FN_STATIC }
    if method.flag("isGenerator") { combined = combined | FN_GENERATOR }
    if method.flag("isAsync") { combined = combined | FN_ASYNC }
    let name = method.text("name")
    let index = _function_constant(e, method, name, combined, init_fields)
    _emit_closure(e, method, index)
    e._chunk().write_op(Op.Method, span)
    e._chunk().write_u24(e._string(name))
    e._chunk().write_byte(combined & 0xFF)
    e._chunk().write_byte(_visibility_code(method.visibility()))
}

pub fn _enum_decl(e: Emitter, node: Node) -> void {
    let span = node.span
    let name = node.text("name")
    e.enums[name] = node

    var variants: list[Const] = []
    for variant in node.records("variants") {
        variants.push(Const(ConstTag.Str, variant["name"]))
        var field_names: list[Const] = []
        for param in variant["params"] {
            field_names.push(Const(ConstTag.Str, param["name"] ?? ""))
        }
        variants.push(Const(ConstTag.Tuple, field_names))
    }
    e._op_k(Op.Class, e._class_spec(SPEC_ENUM, name, null, false, [], variants), span)
    for method in node.child_list("methods") { _method(e, method, FN_METHOD, null) }
    _define(e, node, name, span)
}

pub fn _trait_decl(e: Emitter, node: Node) -> void {
    let span = node.span
    let name = node.text("name")

    var required: list[Const] = []
    for method in node.child_list("methods") {
        if method.child("body") != null { continue }
        required.push(Const(ConstTag.Str, method.opt_text("name") ?? ""))
        required.push(Const(ConstTag.Int, method.records("params").len()))
    }
    e._op_k(Op.Class, e._class_spec(SPEC_TRAIT, name, null, false, required, []), span)
    for super_trait in node.type_list("supers") {
        e._op_k(Op.ImplTrait, e._string(super_trait.name ?? ""), span)
    }
    for method in node.child_list("methods") {
        if method.child("body") == null { continue }  # defaults only
        _method(e, method, FN_METHOD, null)
    }
    _define(e, node, name, span)
}

pub fn _define(e: Emitter, node: Node, name: str?, span: Span) -> void {
    let symbol = e.resolution.symbol_of(node)
    e._store_symbol(symbol, name ?? "_", span, true)
    if symbol != null and symbol.visibility == Visibility.Public {
        e._op_k(Op.Export, e._string(symbol.name), span)
    }
}
