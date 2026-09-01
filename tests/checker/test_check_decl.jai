#: `jaithon.compile.check.decl` — layout, conformance, and declaration bodies.
#:
#: Three groups of case earn their place here. The first is the ones where
#: `check_decl.c` does something a re-derivation would not: `init` is exempt from
#: the override rule, a trait's members are public whether or not `pub` is
#: written, a parent that resolved to nothing is opaque rather than absent, and
#: trait conformance is verified *before* anything concludes the class is
#: abstract. The second is the layout arithmetic — a subclass's field slots
#: continue after its parent's without the parent's fields being copied, which is
#: the one place this port deliberately differs from the C and so the one place a
#: silent regression would go unnoticed. The third is substitution: a generic
#: trait's requirement has to arrive in the implementer's own parameters, and
#: `Self` has to become the implementer's type, before either is compared.
#:
#: The diagnostics are asserted by code **and by message text**, and by their
#: labels, notes and helps where the C attaches them, because all of it is a
#: contract with the C front end: Phase 4 turns `tests/errors` into goldens and a
#: reworded message fails there.
from std.test import assert_eq, assert_len
from jaithon.ast import DeclKind, Node, NodeKind, OpKind, Span, TypeExpr, TypeKind, Visibility, class_field, param, variant
from jaithon.compile.check.decl import E0303_NON_CONSTANT_CONST, E0304_MISSING_INITIALIZER, E0601_MISSING_RETURN, E0708_SUPER_INIT_NOT_FIRST, check_decl, check_var_decl, function_type, install, layout_decl, register_decl
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

fn _trait(
    name: str,
    methods: list[Node] = [],
    supers: list[TypeExpr] = [],
    generics: list[dict[str, any]] = []
) -> Node {
    return Node(
        NodeKind.TraitDecl,
        {"name": name, "generics": generics, "supers": supers, "methods": methods}
    )
}

fn _enum(name: str, variants: list[dict[str, any]] = [], methods: list[Node] = []) -> Node {
    return Node(NodeKind.EnumDecl, {"name": name, "variants": variants, "methods": methods})
}

fn _declare(c: Ctx, node: Node) -> TypeDecl {
    return register_decl(c, node) ?? TypeDecl("?", node.span)
}

fn _self() -> dict[str, any] {
    return param("self")
}

# ------------------------------------------------------------------
# Declaration bodies
# ------------------------------------------------------------------
fn test_a_field_default_is_checked_against_the_field() -> void {
    let c = _ctx()
    let node = _class("Point", [class_field("x", TypeExpr.named("int"), Node.str_lit("a"))])
    _declare(c, node)
    check_decl(c, node)

    assert_len(c.bag.items, 1)
    assert_eq(_code(c, 0), E0400_TYPE_MISMATCH)
    assert_eq(_message(c, 0), "mismatched type for a field default: expected `int`, found `str`")
}

fn test_a_field_default_widens_an_int_literal_into_a_float() -> void {
    let c = _ctx()
    let literal = Node.int_lit(1)
    let node = _class("Point", [class_field("x", TypeExpr.named("float"), literal)])
    _declare(c, node)
    check_decl(c, node)

    assert_len(c.bag.items, 0)
    assert_eq(literal.kind, NodeKind.FloatLit)
}

fn test_a_shadowed_declarations_body_is_not_checked() -> void {
    let c = _ctx()
    _declare(c, _class("Point"))
    let shadow = _class("Point", [class_field("x", TypeExpr.named("int"), Node.str_lit("a"))])
    let _second = register_decl(c, shadow)
    check_decl(c, shadow)

    assert_len(c.bag.items, 0)
}

fn test_super_must_open_a_constructor_when_the_parent_declares_one() -> void {
    let c = _ctx()
    _declare(c, _class("Base", methods: [_method("init", [_self()])]))

    let call = Node.call(Node(NodeKind.Super), [])
    let body = Node.block([Node.expr_stmt(Node.int_lit(1)), Node.expr_stmt(call)])
    let init = _fn("init", [_self()], null, body)
    let node = _class("Derived", methods: [init], superclass: TypeExpr.named("Base"))
    _declare(c, node)
    check_decl(c, node)

    assert_len(c.bag.items, 1)
    assert_eq(_code(c, 0), E0708_SUPER_INIT_NOT_FIRST)
    assert_eq(_message(c, 0), "`super(...)` must be the first statement of `init`")
    assert_eq(_label(c, 0, 0), "`Base` declares an `init`")
    assert_eq(_help(c, 0), "move the call to the top of `init`")
}

fn test_super_first_is_accepted() -> void {
    let c = _ctx()
    _declare(c, _class("Base", methods: [_method("init", [_self()])]))

    let call = Node.call(Node(NodeKind.Super), [])
    let init = _fn("init", [_self()], null, Node.block([Node.expr_stmt(call)]))
    let node = _class("Derived", methods: [init], superclass: TypeExpr.named("Base"))
    _declare(c, node)
    check_decl(c, node)

    assert_len(c.bag.items, 0)
}

fn test_a_parent_with_no_init_imposes_no_ordering() -> void {
    let c = _ctx()
    _declare(c, _class("Base"))

    let call = Node.call(Node(NodeKind.Super), [])
    let body = Node.block([Node.expr_stmt(Node.int_lit(1)), Node.expr_stmt(call)])
    let node = _class(
        "Derived",
        methods: [_fn("init", [_self()], null, body)],
        superclass: TypeExpr.named("Base")
    )
    _declare(c, node)
    check_decl(c, node)

    assert_len(c.bag.items, 0)
}

fn test_a_trait_requirement_has_no_body_to_check() -> void {
    let c = _ctx()
    let node = _trait("Shape", [_fn("area", [_self()], TypeExpr.named("int"))])
    _declare(c, node)
    check_decl(c, node)

    assert_len(c.bag.items, 0)
}

fn test_a_trait_default_body_is_checked() -> void {
    let c = _ctx()
    let node = _trait("Shape", [_method("area", [_self()], TypeExpr.named("int"))])
    _declare(c, node)
    check_decl(c, node)

    assert_len(c.bag.items, 1)
    assert_eq(_code(c, 0), E0601_MISSING_RETURN)
}

fn test_a_variant_payloads_default_is_evaluated() -> void {
    let c = _ctx()
    let default_value = Node.binary(OpKind.Add, Node.int_lit(1), Node.int_lit(2))
    let node = _enum(
        "Shape",
        [variant("Circle", [param("radius", TypeExpr.named("int"), default_value)])]
    )
    _declare(c, node)
    check_decl(c, node)

    assert_len(c.bag.items, 0)
    assert_eq(default_value.kind, NodeKind.IntLit)
    assert_eq(default_value.number("intLit"), 3)
}

fn test_a_function_declaration_gets_a_type_and_a_symbol_type() -> void {
    let c = _ctx()
    let node = _fn(
        "twice",
        [param("n", TypeExpr.named("int"))],
        TypeExpr.named("int"),
        Node.block([Node.return_stmt(Node.int_lit(1))])
    )
    let symbol = Symbol(SymbolKind.Global, "twice", Span(0, 0, 5), DeclKind.Let)
    c.resolution.bind(node, symbol)
    check_decl(c, node)

    assert_eq(render(c.type_of(node)), "fn(int) -> int")
    assert_eq(render(c.symbol_type(symbol)), "fn(int) -> int")
    assert_len(c.bag.items, 0)
}

fn test_a_type_alias_and_an_import_type_their_bindings() -> void {
    let c = _ctx()
    let alias = Node(NodeKind.TypeDecl, {"name": "Count", "aliased": TypeExpr.named("int")})
    let alias_symbol = Symbol(SymbolKind.TypeAlias, "Count", Span(0, 0, 5), DeclKind.Let)
    c.resolution.bind(alias, alias_symbol)
    check_decl(c, alias)
    assert_eq(render(c.symbol_type(alias_symbol)), "int")

    let imported = Node(NodeKind.Import, {"path": "std.math"})
    let module_symbol = Symbol(SymbolKind.Module, "math", Span(0, 0, 4), DeclKind.Let)
    c.resolution.bind(imported, module_symbol)
    check_decl(c, imported)
    assert_eq(render(c.symbol_type(module_symbol)), "module std.math")
}

# ------------------------------------------------------------------
# Value declarations
# ------------------------------------------------------------------
fn test_an_annotated_binding_takes_its_annotation() -> void {
    let c = _ctx()
    let node = Node.var_decl(DeclKind.Let, Node.bind("n"), TypeExpr.named("int"), Node.int_lit(1))
    check_var_decl(c, node)

    assert_eq(render(c.type_of(node)), "int")
    assert_len(c.bag.items, 0)
}

fn test_an_initialiser_that_does_not_fit_its_annotation_is_reported() -> void {
    let c = _ctx()
    let node = Node.var_decl(DeclKind.Let, Node.bind("n"), TypeExpr.named("int"), Node.str_lit("x"))
    check_var_decl(c, node)

    assert_len(c.bag.items, 1)
    assert_eq(_code(c, 0), E0400_TYPE_MISMATCH)
    assert_eq(_message(c, 0), "mismatched types in this binding: expected `int`, found `str`")
}

fn test_an_unannotated_binding_infers_its_initialiser() -> void {
    let c = _ctx()
    let node = Node.var_decl(DeclKind.Let, Node.bind("n"), null, Node.int_lit(1))
    check_var_decl(c, node)
    assert_eq(render(c.type_of(node)), "int")
}

fn test_an_unannotated_var_bound_to_null_is_dynamic() -> void {
    let c = _ctx()
    let variable = Node.var_decl(DeclKind.Var, Node.bind("x"), null, Node.null_lit())
    check_var_decl(c, variable)
    assert_eq(render(c.type_of(variable)), "any")

    let binding = Node.var_decl(DeclKind.Let, Node.bind("x"), null, Node.null_lit())
    check_var_decl(c, binding)
    assert_eq(render(c.type_of(binding)), "null")
}

fn test_a_binding_with_no_initialiser_is_any() -> void {
    let c = _ctx()
    let node = Node.var_decl(DeclKind.Var, Node.bind("x"))
    check_var_decl(c, node)
    assert_eq(render(c.type_of(node)), "any")
}

fn test_the_bindings_symbol_is_typed_through_the_pattern() -> void {
    let c = _ctx()
    let pattern = Node.bind("n")
    let node = Node.var_decl(DeclKind.Let, pattern, null, Node.int_lit(1))
    let symbol = Symbol(SymbolKind.Local, "n", Span(0, 0, 1), DeclKind.Let)
    c.resolution.bind(pattern, symbol)
    check_var_decl(c, node)

    assert_eq(render(c.symbol_type(symbol)), "int")
}

fn test_a_const_is_folded_and_recorded_for_its_uses() -> void {
    let c = _ctx()
    let pattern = Node.bind("K")
    let init = Node.binary(OpKind.Mul, Node.int_lit(6), Node.int_lit(7))
    let node = Node.var_decl(DeclKind.Const, pattern, null, init)
    let symbol = Symbol(SymbolKind.Global, "K", Span(0, 0, 1), DeclKind.Const)
    c.resolution.bind(pattern, symbol)
    check_var_decl(c, node)

    assert_len(c.bag.items, 0)
    assert_eq(init.kind, NodeKind.IntLit)
    assert_eq(init.number("intLit"), 42)
    assert_eq(c.consts["K"].number("intLit"), 42)
}

fn test_a_const_that_is_not_constant_is_reported() -> void {
    let c = _ctx()
    let node = Node.var_decl(DeclKind.Const, Node.bind("K"), null, Node.ident("runtime"))
    check_var_decl(c, node)

    assert_len(c.bag.items, 1)
    assert_eq(_code(c, 0), E0303_NON_CONSTANT_CONST)
    assert_eq(_message(c, 0), "the initialiser of a `const` must be a constant expression")
    assert_eq(_help(c, 0), "use `let` if the value is only known at run time")
}

fn test_a_const_with_no_initialiser_is_reported() -> void {
    let c = _ctx()
    check_var_decl(c, Node.var_decl(DeclKind.Const, Node.bind("K")))

    assert_len(c.bag.items, 1)
    assert_eq(_code(c, 0), E0304_MISSING_INITIALIZER)
    assert_eq(_message(c, 0), "a `const` declaration needs an initialiser")
}

fn test_a_fold_that_overflowed_reports_once_and_not_twice() -> void {
    let c = _ctx()
    let over = Node.binary(OpKind.Pow, Node.int_lit(2), Node.int_lit(70))
    check_var_decl(c, Node.var_decl(DeclKind.Const, Node.bind("K"), null, over))

    assert_len(c.bag.items, 1)
    assert_eq(_code(c, 0), "E0900")
}

# ------------------------------------------------------------------
# Installation
# ------------------------------------------------------------------
fn test_install_replaces_the_contexts_defaults() -> void {
    let c = Ctx(Universe())
    let node = _class("Box")
    let box = _declare(c, node)

    c.layout_decl(c, box)
    assert_len(c.layout_status, 0)

    install(c)
    c.layout_decl(c, box)
    assert_eq(c.layout_status[id(box)], 2)
    assert_eq(
        render(c.function_type(c, _fn("f", [param("n")], body: Node.block([])))),
        "fn(any) -> any"
    )
}
