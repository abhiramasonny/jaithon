#: `jaithon.compile.check.decl` — signatures and the function bodies behind them.
from std.test import assert_eq, assert_false, assert_len, assert_true
from jaithon.ast import DeclKind, Node, NodeKind, Span, TypeExpr, TypeKind, Visibility, generic, param
from jaithon.compile.check.decl import E0408_RETURN_TYPE_MISMATCH, E0601_MISSING_RETURN, check_body, declared_return_type, function_type, install, is_init_method, register_decl, signature_of
from jaithon.compile.check.ctx import Ctx
from jaithon.compile.check.expr import E0400_TYPE_MISMATCH
from jaithon.compile.check.kinds import TyKind
from jaithon.compile.check.render import render
from jaithon.compile.check.ty import Type, TypeDecl
from jaithon.compile.check.union import optional_of, union_of
from jaithon.compile.check.universe import Universe, dict_of, fn_of, instantiate, list_of, set_of, tuple_of
from jaithon.compile.symbol import Symbol, SymbolKind

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------
fn _resolver(c: Ctx, written: TypeExpr?) -> Type? {
    if written is null { return null }
    let u = c.universe

    match written.kind {
        TypeKind.Infer => { return null }

        TypeKind.Name => { return _named_type(c, written.name) }

        TypeKind.Generic => {
            var args: list[Type] = []
            for argument in written.args { args.push(_resolver(c, argument) ?? u.t_any) }
            let name = written.name
            if name == "list" and args.len() == 1 { return list_of(u, args[0]) }
            if name == "set" and args.len() == 1 { return set_of(u, args[0]) }
            if name == "dict" and args.len() == 2 { return dict_of(u, args[0], args[1]) }
            let base = _named_type(c, name)
            if base is null { return null }
            return instantiate(u, base, args)
        }

        TypeKind.Optional => { return optional_of(u, _resolver(c, written.inner)) }

        TypeKind.Union => {
            var members: list[Type?] = []
            for member in written.args { members.push(_resolver(c, member)) }
            return union_of(u, members)
        }

        TypeKind.Fn => {
            var params: list[Type] = []
            for parameter in written.args { params.push(_resolver(c, parameter) ?? u.t_any) }
            return fn_of(u, params, _resolver(c, written.inner) ?? u.t_any, 0)
        }

        TypeKind.Tuple => {
            var members: list[Type] = []
            for member in written.args { members.push(_resolver(c, member) ?? u.t_any) }
            return tuple_of(u, members)
        }

        TypeKind.Const => { return u.intern(TyKind.Int, written.name) }
    }
}

fn _named_type(c: Ctx, name: str?) -> Type? {
    if name is null { return null }
    let scoped = c.scoped_type_name(name)
    if scoped is not null { return scoped }
    let decl = c.decl_named(name)
    if decl is not null { return decl.ty }

    let u = c.universe
    if name == "any" { return u.t_any }
    if name == "void" { return u.t_void }
    if name == "null" { return u.t_null }
    if name == "bool" { return u.t_bool }
    if name == "int" { return u.t_int }
    if name == "float" { return u.t_float }
    if name == "str" { return u.t_str }
    if name == "bytes" { return u.t_bytes }
    if name == "never" { return u.t_never }
    return null
}

fn _ctx() -> Ctx {
    let c = Ctx(Universe())
    install(c)
    c.resolve_ast_type = _resolver
    return c
}

fn _code(c: Ctx, at: int) -> str {
    return c.bag.items[at].code
}

fn _message(c: Ctx, at: int) -> str {
    return c.bag.items[at].message
}

fn _help(c: Ctx, at: int) -> str? {
    return c.bag.items[at].help
}

fn _label(c: Ctx, at: int, index: int) -> str? {
    return c.bag.items[at].labels[index][1]
}

fn _fn(
    name: str,
    params: list[dict[str, any]] = [],
    returns: TypeExpr? = null,
    body: Node? = null,
    is_static: bool = false,
    visibility: Visibility = Visibility.Private
) -> Node {
    return Node(
        NodeKind.FnDecl,
        {
            "name": name,
            "params": params,
            "returnType": returns,
            "body": body,
            "isStatic": is_static,
            "visibility": visibility,
        }
    )
}

fn _method(
    name: str,
    params: list[dict[str, any]] = [],
    returns: TypeExpr? = null,
    is_static: bool = false,
    visibility: Visibility = Visibility.Private
) -> Node {
    return _fn(name, params, returns, Node.block([]), is_static, visibility)
}

fn _class(
    name: str,
    fields: list[dict[str, any]] = [],
    methods: list[Node] = [],
    superclass: TypeExpr? = null,
    traits: list[TypeExpr] = [],
    is_abstract: bool = false,
    generics: list[dict[str, any]] = [],
    getters: list[Node] = [],
    setters: list[Node] = []
) -> Node {
    return Node(
        NodeKind.ClassDecl,
        {
            "name": name,
            "generics": generics,
            "superclass": superclass,
            "traits": traits,
            "fields": fields,
            "methods": methods,
            "getters": getters,
            "setters": setters,
            "isAbstract": is_abstract,
        }
    )
}

fn _declare(c: Ctx, node: Node) -> TypeDecl {
    return register_decl(c, node) ?? TypeDecl("?", node.span)
}

fn _self() -> dict[str, any] {
    return param("self")
}

# ------------------------------------------------------------------
# Signatures
# ------------------------------------------------------------------
fn test_a_method_signature_drops_the_receiver_and_a_function_does_not() -> void {
    let c = _ctx()
    let node = _method("scale", [_self(), param("by", TypeExpr.named("float"))])

    assert_eq(render(signature_of(c, node, true)), "fn(float) -> any")
    assert_eq(render(signature_of(c, node, false)), "fn(any, float) -> any")
    assert_eq(render(function_type(c, node)), "fn(any, float) -> any")
    assert_eq(render(function_type(c, null)), "any")
}

fn test_a_static_method_keeps_every_parameter() -> void {
    let c = _ctx()
    let node = _method("make", [param("n", TypeExpr.named("int"))], is_static: true)
    assert_eq(render(signature_of(c, node, true)), "fn(int) -> any")
}

fn test_the_variadic_and_kwrest_flags_reach_the_signature() -> void {
    let c = _ctx()
    let node = _fn(
        "f",
        [
            param("n", TypeExpr.named("int")),
            param("rest", TypeExpr.named("str"), variadic: true),
        ]
    )
    let signature = signature_of(c, node, false)
    assert_true(signature.is_variadic())
    assert_false(signature.has_kwrest())

    let kw = _fn("g", [param("opts", TypeExpr.named("int"), kw_rest: true)])
    assert_true(signature_of(c, kw, false).has_kwrest())
}

fn test_a_generic_parameter_is_in_scope_while_the_signature_resolves() -> void {
    let c = _ctx()
    let node = Node(
        NodeKind.FnDecl,
        {
            "name": "identity",
            "generics": [generic("T")],
            "params": [param("value", TypeExpr.named("T"))],
            "returnType": TypeExpr.named("T"),
            "body": Node.block([]),
        }
    )

    assert_eq(render(signature_of(c, node, false)), "fn(T) -> T")
    assert_true(c.scoped_type_name("T") is null)
}

fn test_a_constructor_returns_void_and_only_as_a_method() -> void {
    let c = _ctx()
    let init = _method("init", [_self()])

    assert_true(declared_return_type(c, init, true) == c.universe.t_void)
    assert_true(declared_return_type(c, init, false) == c.universe.t_any)
    assert_true(is_init_method(init))
    assert_false(is_init_method(_method("area", [_self()])))
}

fn test_a_written_return_type_wins_over_both() -> void {
    let c = _ctx()
    let node = _method("init", [_self()], TypeExpr.named("int"))
    assert_true(declared_return_type(c, node, true) == c.universe.t_int)
}

# ------------------------------------------------------------------
# Function bodies
# ------------------------------------------------------------------
fn test_a_parameter_default_is_checked_against_the_parameter() -> void {
    let c = _ctx()
    let node = _fn(
        "f",
        [param("n", TypeExpr.named("int"), Node.str_lit("x"))],
        body: Node.block([])
    )
    check_body(c, node, null, false)

    assert_len(c.bag.items, 1)
    assert_eq(_code(c, 0), E0400_TYPE_MISMATCH)
    assert_eq(_message(c, 0), "mismatched type for a default value: expected `int`, found `str`")
}

fn test_a_variadic_binds_a_list_and_a_kwrest_binds_a_dict() -> void {
    let c = _ctx()
    let rest = param("rest", TypeExpr.named("int"), variadic: true)
    let opts = param("opts", TypeExpr.named("str"), kw_rest: true)
    let node = _fn("f", [rest, opts], body: Node.block([]))

    let rest_symbol = Symbol(SymbolKind.Param, "rest", Span(0, 0, 4), DeclKind.Var)
    let opts_symbol = Symbol(SymbolKind.Param, "opts", Span(0, 5, 9), DeclKind.Var)
    c.resolution.bind(rest, rest_symbol)
    c.resolution.bind(opts, opts_symbol)
    check_body(c, node, null, false)

    assert_eq(render(c.symbol_type(rest_symbol)), "list[int]")
    assert_eq(render(c.symbol_type(opts_symbol)), "dict[str, str]")
    assert_eq(render(signature_of(c, node, false)), "fn(...int, **str) -> any")
}

fn test_the_receiver_binds_to_the_owning_declarations_type() -> void {
    let c = _ctx()
    let point = _declare(c, _class("Point"))
    let receiver = _self()
    let node = _fn("area", [receiver], body: Node.block([]))
    let symbol = Symbol(SymbolKind.Param, "self", Span(0, 0, 4), DeclKind.Let)
    c.resolution.bind(receiver, symbol)

    check_body(c, node, point, true)
    assert_eq(render(c.symbol_type(symbol)), "Point")
}

fn test_strict_reports_an_unannotated_parameter() -> void {
    let c = _ctx()
    c.strict = true
    check_body(c, _fn("f", [param("n")], body: Node.block([])), null, false)

    assert_len(c.bag.items, 1)
    assert_eq(_code(c, 0), E0400_TYPE_MISMATCH)
    assert_eq(_message(c, 0), "parameter `n` has no type annotation")
    assert_eq(_help(c, 0), "--strict requires every parameter to be annotated")
}

fn test_strict_exempts_a_kwrest_parameter_and_the_receiver() -> void {
    let c = _ctx()
    c.strict = true
    let point = _declare(c, _class("Point"))
    let node = _fn("f", [_self(), param("opts", null, kw_rest: true)], body: Node.block([]))
    check_body(c, node, point, true)

    assert_len(c.bag.items, 0)
}

fn test_an_annotated_function_must_not_fall_off_its_end() -> void {
    let c = _ctx()
    let returns = TypeExpr.named("int")
    let node = _fn("total", [], returns, Node.block([]))
    check_body(c, node, null, false)

    assert_len(c.bag.items, 1)
    assert_eq(_code(c, 0), E0601_MISSING_RETURN)
    assert_eq(_message(c, 0), "`total` must return `int` on every path")
    assert_eq(_label(c, 0, 0), "declared to return `int`")
    assert_eq(_help(c, 0), "add a `return` at the end, or make the last branch an `else`")
}

fn test_a_body_that_always_diverges_is_exempt() -> void {
    let c = _ctx()
    let body = Node.block([Node.return_stmt(Node.int_lit(1))])
    check_body(c, _fn("total", [], TypeExpr.named("int"), body), null, false)
    assert_len(c.bag.items, 0)
}

fn test_an_unannotated_or_optional_return_is_exempt() -> void {
    let c = _ctx()
    check_body(c, _fn("f", [], null, Node.block([])), null, false)
    check_body(c, _fn("g", [], TypeExpr.infer(), Node.block([])), null, false)
    check_body(
        c,
        _fn("h", [], TypeExpr.optional(TypeExpr.named("int")), Node.block([])),
        null,
        false
    )
    check_body(c, _fn("v", [], TypeExpr.named("void"), Node.block([])), null, false)
    assert_len(c.bag.items, 0)
}

fn test_a_generator_returns_through_yield_and_is_exempt() -> void {
    let c = _ctx()
    let node = _fn("counter", [], TypeExpr.named("int"), Node.block([]))
    node.set("isGenerator", true)
    check_body(c, node, null, false)
    assert_len(c.bag.items, 0)
}

fn test_an_expression_body_is_checked_against_the_annotation() -> void {
    let c = _ctx()
    let node = _fn("f", [], TypeExpr.named("int"), Node.str_lit("x"))
    node.set("isExprBody", true)
    check_body(c, node, null, false)

    assert_len(c.bag.items, 1)
    assert_eq(_code(c, 0), E0408_RETURN_TYPE_MISMATCH)
    assert_eq(_message(c, 0), "mismatched return type: expected `int`, found `str`")
}

fn test_an_unannotated_expression_body_takes_its_own_type() -> void {
    let c = _ctx()
    let node = _fn("f", [], null, Node.str_lit("x"))
    check_body(c, node, null, false)
    assert_len(c.bag.items, 0)
}

fn test_a_body_restores_the_enclosing_class_and_return_type() -> void {
    let c = _ctx()
    let point = _declare(c, _class("Point"))
    let outer = _declare(c, _class("Outer"))
    c.current_class = outer
    c.current_return_type = c.universe.t_str

    check_body(c, _fn("area", [_self()], TypeExpr.named("int"), Node.block([])), point, true)
    assert_true(c.current_class == outer)
    assert_true(c.current_return_type == c.universe.t_str)
}
