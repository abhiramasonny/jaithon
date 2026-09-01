#: `jaithon.compile.check.decl` — inheritance links and override compatibility.
from std.test import assert_eq, assert_len, assert_true
from jaithon.ast import Node, NodeKind, Span, TypeExpr, TypeKind, Visibility, class_field, param
from jaithon.compile.check.decl import E0700_UNKNOWN_CLASS, E0704_INCOMPATIBLE_OVERRIDE, E0706_CYCLIC_INHERITANCE, install, layout_decl, register_decl
from jaithon.compile.check.ctx import Ctx
from jaithon.compile.check.kinds import TyKind
from jaithon.compile.check.ty import Type, TypeDecl
from jaithon.compile.check.union import optional_of, union_of
from jaithon.compile.check.universe import Universe, dict_of, fn_of, instantiate, list_of, set_of, tuple_of

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

fn _note(c: Ctx, at: int) -> str? {
    return c.bag.items[at].note
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

fn _laid(c: Ctx, node: Node) -> TypeDecl {
    let d = _declare(c, node)
    layout_decl(c, d)
    return d
}

fn _self() -> dict[str, any] {
    return param("self")
}

# ------------------------------------------------------------------
# Inheritance
# ------------------------------------------------------------------
fn test_a_superclass_that_this_unit_can_see_is_linked() -> void {
    let c = _ctx()
    let base = _declare(c, _class("Base"))
    let derived = _laid(c, _class("Derived", superclass: TypeExpr.named("Base")))

    assert_true(derived.superclass == base)
    assert_true(derived.opaque_super is null)
    assert_len(c.bag.items, 0)
}

fn test_a_superclass_this_unit_cannot_see_is_opaque_rather_than_absent() -> void {
    let c = _ctx()
    let derived = _laid(c, _class("Derived", superclass: TypeExpr.named("Imported")))

    assert_true(derived.superclass is null)
    assert_eq(derived.opaque_super, "Imported")
    assert_len(c.bag.items, 0)
}

fn test_a_trait_this_unit_cannot_see_is_named_rather_than_denied() -> void {
    let c = _ctx()
    let widget = _laid(c, _class("Widget", traits: [TypeExpr.named("Drawable")]))

    assert_len(widget.traits, 0)
    assert_eq(widget.opaque_traits, ["Drawable"])
    assert_len(c.bag.items, 0)
}

fn test_naming_a_class_where_a_trait_belongs_is_reported() -> void {
    let c = _ctx()
    _declare(c, _class("Base"))
    _laid(c, _class("Widget", traits: [TypeExpr.named("Base")]))

    assert_eq(_code(c, 0), E0700_UNKNOWN_CLASS)
    assert_eq(_message(c, 0), "`Base` is not a trait")
}

fn test_naming_a_trait_or_an_enum_where_a_class_belongs_is_reported() -> void {
    let c = _ctx()
    _declare(c, _trait("Shape"))
    _declare(c, _enum("Colour"))
    _laid(c, _class("A", superclass: TypeExpr.named("Shape")))
    _laid(c, _class("B", superclass: TypeExpr.named("Colour")))

    assert_eq(_code(c, 0), E0700_UNKNOWN_CLASS)
    assert_eq(_message(c, 0), "`Shape` is not a class")
    assert_eq(_message(c, 1), "`Colour` is not a class")
}

fn test_a_class_that_would_inherit_from_itself_is_reported() -> void {
    let c = _ctx()
    let loop_class = _laid(c, _class("Loop", superclass: TypeExpr.named("Loop")))

    assert_eq(_code(c, 0), E0706_CYCLIC_INHERITANCE)
    assert_eq(_message(c, 0), "`Loop` would inherit from itself")
    assert_eq(_label(c, 0, 0), "the cycle passes through here")
    assert_true(loop_class.superclass is null)
    # Reported, and then treated as a parent this unit cannot enumerate.
    assert_eq(loop_class.opaque_super, "Loop")
}

fn test_a_cycle_through_two_classes_is_caught_on_the_link_that_closes_it() -> void {
    let c = _ctx()
    _declare(c, _class("A", superclass: TypeExpr.named("B")))
    _declare(c, _class("B", superclass: TypeExpr.named("A")))
    layout_decl(c, c.decl_named("A"))

    assert_eq(_code(c, 0), E0706_CYCLIC_INHERITANCE)
    assert_eq(_message(c, 0), "`A` would inherit from itself")
}

fn test_layout_runs_once() -> void {
    let c = _ctx()
    let point = _laid(c, _class("Point", [class_field("x", TypeExpr.named("int"))]))
    layout_decl(c, point)
    layout_decl(c, point)

    assert_len(point.fields, 1)
    assert_eq(c.layout_status[id(point)], 2)
}

fn test_a_parent_is_laid_out_on_demand_from_the_child_that_names_it() -> void {
    let c = _ctx()
    _declare(c, _class("Base", [class_field("a", TypeExpr.named("int"))]))
    let derived = _laid(
        c,
        _class(
            "Derived",
            [class_field("b", TypeExpr.named("int"))],
            superclass: TypeExpr.named("Base")
        )
    )

    # `Base` was never laid out directly; resolving the header did it.
    let base = derived.superclass ?? TypeDecl("?", Span.none())
    assert_eq(c.layout_status[id(base)], 2)
    assert_eq(derived.fields[0].slot, 1)
}

# ------------------------------------------------------------------
# Overrides
# ------------------------------------------------------------------
fn test_an_override_may_widen_a_parameter_and_narrow_the_return() -> void {
    let c = _ctx()
    _declare(
        c,
        _class(
            "Base",
            methods: [_method("f", [_self(), param("n", TypeExpr.named("int"))], TypeExpr.named("any"))]
        )
    )
    _laid(
        c,
        _class(
            "Derived",
            methods: [_method("f", [_self(), param("n", TypeExpr.named("any"))], TypeExpr.named("int"))],
            superclass: TypeExpr.named("Base")
        )
    )

    assert_len(c.bag.items, 0)
}

fn test_an_override_that_narrows_a_parameter_is_reported() -> void {
    let c = _ctx()
    _declare(
        c,
        _class(
            "Base",
            methods: [_method("f", [_self(), param("n", TypeExpr.named("int"))], TypeExpr.named("int"))]
        )
    )
    _laid(
        c,
        _class(
            "Derived",
            methods: [_method("f", [_self(), param("n", TypeExpr.named("str"))], TypeExpr.named("int"))],
            superclass: TypeExpr.named("Base")
        )
    )

    assert_len(c.bag.items, 1)
    assert_eq(_code(c, 0), E0704_INCOMPATIBLE_OVERRIDE)
    assert_eq(
        _message(c, 0),
        "`f` overrides an incompatible signature: " + "a parameter type is not general enough"
    )
    assert_eq(_note(c, 0), "overriding `fn(int) -> int` with `fn(str) -> int`")
    assert_eq(_label(c, 0, 0), "the overridden declaration")
    assert_eq(_help(c, 0), "parameters may only widen and the return type may only narrow")
}

fn test_an_override_with_a_different_parameter_count_is_reported() -> void {
    let c = _ctx()
    _declare(c, _class("Base", methods: [_method("f", [_self(), param("n")])]))
    _laid(
        c,
        _class("Derived", methods: [_method("f", [_self()])], superclass: TypeExpr.named("Base"))
    )

    assert_eq(
        _message(c, 0),
        "`f` overrides an incompatible signature: " + "the parameter counts differ"
    )
}

fn test_an_override_that_widens_the_return_is_reported() -> void {
    let c = _ctx()
    _declare(c, _class("Base", methods: [_method("f", [_self()], TypeExpr.named("int"))]))
    _laid(
        c,
        _class(
            "Derived",
            methods: [_method("f", [_self()], TypeExpr.named("str"))],
            superclass: TypeExpr.named("Base")
        )
    )

    assert_eq(
        _message(c, 0),
        "`f` overrides an incompatible signature: " + "the return type is not compatible"
    )
}

fn test_init_is_not_an_override() -> void {
    let c = _ctx()
    _declare(c, _class("Base", methods: [_method("init", [_self(), param("n")])]))
    _laid(
        c,
        _class("Derived", methods: [_method("init", [_self()])], superclass: TypeExpr.named("Base"))
    )

    assert_len(c.bag.items, 0)
}

fn test_an_unannotated_parameter_fits_whatever_the_parent_declared() -> void {
    let c = _ctx()
    _declare(
        c,
        _class(
            "Base",
            methods: [_method("f", [_self(), param("n", TypeExpr.named("int"))], TypeExpr.named("int"))]
        )
    )
    _laid(
        c,
        _class(
            "Derived",
            methods: [_method("f", [_self(), param("n")])],
            superclass: TypeExpr.named("Base")
        )
    )

    assert_len(c.bag.items, 0)
}
