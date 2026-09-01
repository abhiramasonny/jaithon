#: `jaithon.compile.check.decl` — registration, generics, fields, members.
from std.test import assert_eq, assert_false, assert_len, assert_true
from jaithon.ast import DeclKind, Node, NodeKind, Span, TypeExpr, TypeKind, Visibility, class_field, generic, param
from jaithon.compile.check.decl import E0703_MISSING_SELF, E0707_STATIC_WITH_SELF, E0709_DUPLICATE_MEMBER, SELF_NAME, install, layout_decl, push_generics, push_self, register_decl
from jaithon.compile.check.ctx import Ctx
from jaithon.compile.check.expr import VIS_PRIVATE, VIS_PUBLIC
from jaithon.compile.check.kinds import TyKind
from jaithon.compile.check.nominal import find_field
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
# Registration
# ------------------------------------------------------------------
fn test_registering_a_class_interns_its_type_and_records_the_flags() -> void {
    let c = _ctx()
    let node = _class("Point")
    let point = _declare(c, node)

    assert_eq(point.name, "Point")
    assert_false(point.is_trait)
    assert_false(point.is_enum)
    assert_false(point.is_abstract)
    assert_eq(render(point.ty), "Point")
    assert_true(point.ty is not null and point.ty.kind == TyKind.Class)
    assert_true(point.ty is not null and point.ty.decl == point)
    assert_true(c.type_of(node) == point.ty)
}

fn test_a_trait_and_an_enum_get_their_own_kinds() -> void {
    let c = _ctx()
    let shape = _declare(c, _trait("Shape"))
    let colour = _declare(c, _enum("Colour"))

    assert_true(shape.is_trait)
    assert_true(shape.ty is not null and shape.ty.kind == TyKind.Trait)
    assert_true(colour.is_enum)
    assert_true(colour.ty is not null and colour.ty.kind == TyKind.Enum)
}

fn test_an_abstract_class_says_so() -> void {
    let c = _ctx()
    assert_true(_declare(c, _class("Shape", is_abstract: true)).is_abstract)
    assert_false(_declare(c, _class("Point")).is_abstract)
}

fn test_a_second_declaration_of_one_name_answers_the_first() -> void {
    let c = _ctx()
    let first = _declare(c, _class("Point"))
    let shadow = _class("Point")
    let second = register_decl(c, shadow)

    assert_true(second == first)
    assert_len(c.bag.items, 0)
    assert_true(c.decl_for_node(shadow) is null)
}

fn test_only_a_class_trait_or_enum_registers() -> void {
    let c = _ctx()
    assert_true(register_decl(c, Node.block([])) is null)
    assert_true(register_decl(c, null) is null)
    assert_true(register_decl(c, _class("")) is not null)
}

fn test_registration_stamps_the_declarations_symbol() -> void {
    let c = _ctx()
    let node = _class("Point")
    let symbol = Symbol(SymbolKind.Class, "Point", Span(0, 0, 5), DeclKind.Let)
    c.resolution.bind(node, symbol)

    let point = _declare(c, node)
    assert_true(c.symbol_type(symbol) == point.ty)
}

# ------------------------------------------------------------------
# Generic parameters and `Self`
# ------------------------------------------------------------------
fn test_generic_parameters_are_bound_and_unwound() -> void {
    let c = _ctx()
    let mark = c.type_name_mark()
    push_generics(c, [generic("T"), generic("U")])

    assert_eq(render(c.scoped_type_name("T")), "T")
    assert_eq(render(c.scoped_type_name("U")), "U")
    c.type_name_restore(mark)
    assert_true(c.scoped_type_name("T") is null)
}

fn test_a_generic_bound_is_resolved_with_the_parameters_in_scope() -> void {
    let c = _ctx()
    _declare(c, _trait("Comparable"))
    let mark = c.type_name_mark()
    push_generics(c, [generic("T", TypeExpr.generic("Comparable", [TypeExpr.named("T")]))])
    assert_eq(render(c.scoped_type_name("T")), "T")
    c.type_name_restore(mark)
    assert_len(c.bag.items, 0)
}

fn test_self_is_the_implementer_or_stays_abstract() -> void {
    let c = _ctx()
    push_self(c, null)
    assert_eq(render(c.scoped_type_name(SELF_NAME)), "Self")
    push_self(c, c.universe.t_int)
    assert_eq(render(c.scoped_type_name(SELF_NAME)), "int")
}

# ------------------------------------------------------------------
# Fields and slots
# ------------------------------------------------------------------
fn test_fields_take_slots_in_declaration_order() -> void {
    let c = _ctx()
    let point = _laid(
        c,
        _class(
            "Point",
            [
                class_field("x", TypeExpr.named("int")),
                class_field("y", TypeExpr.named("int")),
            ]
        )
    )

    assert_len(point.fields, 2)
    assert_eq(point.fields[0].slot, 0)
    assert_eq(point.fields[1].slot, 1)
    assert_eq(render(point.fields[0].ty), "int")
    assert_len(c.bag.items, 0)
}

fn test_a_static_field_occupies_no_slot() -> void {
    let c = _ctx()
    let counter = _laid(
        c,
        _class(
            "Counter",
            [
                class_field("total", TypeExpr.named("int"), is_static: true),
                class_field("n", TypeExpr.named("int")),
            ]
        )
    )

    assert_eq(counter.fields[0].slot, -1)
    assert_true(counter.fields[0].is_static)
    assert_eq(counter.fields[1].slot, 0)
}

fn test_a_subclass_numbers_its_fields_after_its_parents_without_copying_them() -> void {
    let c = _ctx()
    _declare(
        c,
        _class(
            "Base",
            [class_field("a", TypeExpr.named("int")), class_field("b", TypeExpr.named("int"))]
        )
    )
    let derived = _laid(
        c,
        _class(
            "Derived",
            [class_field("c", TypeExpr.named("int"))],
            superclass: TypeExpr.named("Base")
        )
    )

    assert_len(derived.fields, 1)
    assert_eq(derived.fields[0].name, "c")
    assert_eq(derived.fields[0].slot, 2)
    assert_true(derived.find_field("a") is null)
    assert_eq(find_field(derived, "a")?.slot, 0)
}

fn test_field_visibility_and_the_default_flag_are_recorded() -> void {
    let c = _ctx()
    let account = _laid(
        c,
        _class(
            "Account",
            [
                class_field(
                    "balance",
                    TypeExpr.named("int"),
                    Node.int_lit(0),
                    Visibility.Public
                ),
                class_field("pin", TypeExpr.named("int"), is_let: true),
            ]
        )
    )

    assert_eq(account.fields[0].visibility, VIS_PUBLIC)
    assert_true(account.fields[0].has_default)
    assert_eq(account.fields[1].visibility, VIS_PRIVATE)
    assert_true(account.fields[1].is_let)
    assert_false(account.fields[1].has_default)
}

fn test_a_field_declared_twice_is_reported() -> void {
    let c = _ctx()
    _laid(
        c,
        _class(
            "Point",
            [class_field("x", TypeExpr.named("int")), class_field("x", TypeExpr.named("int"))]
        )
    )

    assert_len(c.bag.items, 1)
    assert_eq(_code(c, 0), E0709_DUPLICATE_MEMBER)
    assert_eq(_message(c, 0), "`x` is already declared in `Point`")
    assert_eq(_label(c, 0, 0), "in this declaration")
}

fn test_redeclaring_a_parents_field_is_reported() -> void {
    let c = _ctx()
    _declare(c, _class("Base", [class_field("a", TypeExpr.named("int"))]))
    _laid(
        c,
        _class(
            "Derived",
            [class_field("a", TypeExpr.named("int"))],
            superclass: TypeExpr.named("Base")
        )
    )

    assert_eq(_code(c, 0), E0709_DUPLICATE_MEMBER)
    assert_eq(_message(c, 0), "`a` is already declared in `Derived`")
}

# ------------------------------------------------------------------
# Members
# ------------------------------------------------------------------
fn test_a_method_becomes_a_member_carrying_its_signature() -> void {
    let c = _ctx()
    let point = _laid(
        c,
        _class(
            "Point",
            methods: [
                _method(
                    "scale",
                    [_self(), param("by", TypeExpr.named("float"))],
                    TypeExpr.named("int"),
                    visibility: Visibility.Public
                ),
            ]
        )
    )

    assert_len(point.members, 1)
    let scale = point.members[0]
    assert_eq(scale.name, "scale")
    assert_eq(render(scale.ty), "fn(float) -> int")
    assert_eq(scale.visibility, VIS_PUBLIC)
    assert_false(scale.is_abstract)
    assert_false(scale.is_static)
    assert_len(c.bag.items, 0)
}

fn test_a_method_with_no_body_is_abstract() -> void {
    let c = _ctx()
    let shape = _laid(c, _trait("Shape", [_fn("area", [_self()], TypeExpr.named("float"))]))
    assert_true(shape.members[0].is_abstract)
}

fn test_a_traits_members_are_public_whether_or_not_pub_is_written() -> void {
    let c = _ctx()
    let shape = _laid(c, _trait("Shape", [_fn("area", [_self()], TypeExpr.named("float"))]))
    assert_eq(shape.members[0].visibility, VIS_PUBLIC)

    let point = _laid(c, _class("Point", methods: [_method("area", [_self()])]))
    assert_eq(point.members[0].visibility, VIS_PRIVATE)
}

fn test_an_instance_method_must_take_self() -> void {
    let c = _ctx()
    _laid(c, _class("Point", methods: [_method("area")]))

    assert_len(c.bag.items, 1)
    assert_eq(_code(c, 0), E0703_MISSING_SELF)
    assert_eq(_message(c, 0), "instance method `area` must take `self` as its first parameter")
}

fn test_a_static_method_must_not_take_self() -> void {
    let c = _ctx()
    _laid(c, _class("Point", methods: [_method("origin", [_self()], is_static: true)]))

    assert_len(c.bag.items, 1)
    assert_eq(_code(c, 0), E0707_STATIC_WITH_SELF)
    assert_eq(_message(c, 0), "`static fn origin` must not take `self`")
}

fn test_a_property_needs_no_self_and_types_as_what_it_reads_or_writes() -> void {
    let c = _ctx()
    let point = _laid(
        c,
        _class(
            "Point",
            getters: [_method("width", [], TypeExpr.named("int"))],
            setters: [_method("width", [param("value", TypeExpr.named("int"))])]
        )
    )

    assert_len(c.bag.items, 0)
    assert_len(point.members, 2)
    assert_true(point.members[0].is_getter)
    assert_eq(render(point.members[0].ty), "int")
    assert_true(point.members[1].is_setter)
    assert_eq(render(point.members[1].ty), "int")
}

fn test_a_method_declared_twice_is_reported() -> void {
    let c = _ctx()
    _laid(c, _class("Point", methods: [_method("area", [_self()]), _method("area", [_self()])]))

    assert_len(c.bag.items, 1)
    assert_eq(_code(c, 0), E0709_DUPLICATE_MEMBER)
    assert_eq(_message(c, 0), "`area` is already declared in `Point`")
    assert_eq(_label(c, 0, 0), "in this declaration")
}

fn test_a_method_that_collides_with_a_field_is_reported() -> void {
    let c = _ctx()
    _laid(
        c,
        _class(
            "Point",
            [class_field("area", TypeExpr.named("int"))],
            methods: [_method("area", [_self()])]
        )
    )

    assert_len(c.bag.items, 1)
    assert_eq(_code(c, 0), E0709_DUPLICATE_MEMBER)
    assert_eq(_message(c, 0), "`area` is already declared as a field of `Point`")
}

fn test_a_getter_and_a_setter_share_a_name_without_colliding() -> void {
    let c = _ctx()
    _laid(
        c,
        _class(
            "Point",
            getters: [_method("width", [], TypeExpr.named("int"))],
            setters: [_method("width", [param("value", TypeExpr.named("int"))])]
        )
    )
    assert_len(c.bag.items, 0)
}
