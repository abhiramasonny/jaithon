#: `jaithon.compile.check.decl` — trait conformance, substitution, and enums.
from std.test import assert_eq, assert_len, assert_true
from jaithon.ast import Node, NodeKind, TypeExpr, TypeKind, Visibility, generic, param, variant
from jaithon.compile.check.decl import E0705_TRAIT_NOT_IMPLEMENTED, E0709_DUPLICATE_MEMBER, install, layout_decl, register_decl
from jaithon.compile.check.ctx import Ctx
from jaithon.compile.check.expr import VIS_PUBLIC
from jaithon.compile.check.kinds import TyKind
from jaithon.compile.check.nominal import find_member
from jaithon.compile.check.render import render
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
# Trait conformance
# ------------------------------------------------------------------
fn test_an_unimplemented_requirement_is_reported_with_the_declaration_to_write() -> void {
    let c = _ctx()
    _laid(
        c,
        _trait(
            "Shape",
            [
                _fn(
                    "area",
                    [_self(), param("scale", TypeExpr.named("float"))],
                    TypeExpr.named("float")
                ),
            ]
        )
    )
    _laid(c, _class("Circle", traits: [TypeExpr.named("Shape")]))

    assert_len(c.bag.items, 1)
    assert_eq(_code(c, 0), E0705_TRAIT_NOT_IMPLEMENTED)
    assert_eq(_message(c, 0), "`Circle` does not implement `area` required by trait `Shape`")
    assert_eq(_label(c, 0, 0), "required here")
    assert_eq(_help(c, 0), "add `fn area(self, scale: float) -> float` to `Circle`")
}

fn test_a_trait_default_satisfies_its_own_requirement() -> void {
    let c = _ctx()
    _laid(c, _trait("Shape", [_method("area", [_self()], TypeExpr.named("float"))]))
    let circle = _laid(c, _class("Circle", traits: [TypeExpr.named("Shape")]))

    assert_len(c.bag.items, 0)
    assert_len(circle.members, 0)
    assert_eq(render(find_member(circle, "area")?.ty), "fn() -> float")
}

fn test_an_implementation_with_the_wrong_signature_is_reported() -> void {
    let c = _ctx()
    _laid(c, _trait("Shape", [_fn("area", [_self()], TypeExpr.named("float"))]))
    _laid(
        c,
        _class(
            "Circle",
            methods: [_method("area", [_self()], TypeExpr.named("str"))],
            traits: [TypeExpr.named("Shape")]
        )
    )

    assert_len(c.bag.items, 1)
    assert_eq(_code(c, 0), E0705_TRAIT_NOT_IMPLEMENTED)
    assert_eq(
        _message(c, 0),
        "`Circle` implements `area` as `fn() -> str`, but trait " +
            "`Shape` requires `fn() -> float`: the return type is not compatible"
    )
    assert_eq(_label(c, 0, 0), "required here")
    assert_true(_note(c, 0) is null)
}

fn test_a_requirement_written_self_becomes_the_implementers_own_type() -> void {
    let c = _ctx()
    _laid(c, _trait("Doubler", [_fn("twice", [_self()], TypeExpr.named("Self"))]))
    _laid(
        c,
        _class(
            "Bad",
            methods: [_method("twice", [_self()], TypeExpr.named("int"))],
            traits: [TypeExpr.named("Doubler")]
        )
    )

    assert_eq(_code(c, 0), E0705_TRAIT_NOT_IMPLEMENTED)
    assert_eq(
        _message(c, 0),
        "`Bad` implements `twice` as `fn() -> int`, but trait " +
            "`Doubler` requires `fn() -> Bad`: the return type is not compatible"
    )
    assert_eq(_note(c, 0), "`twice` is declared `fn() -> Self` in `Doubler`, where `Self` is `Bad`")
}

fn test_returning_self_satisfies_the_requirement() -> void {
    let c = _ctx()
    _laid(c, _trait("Doubler", [_fn("twice", [_self()], TypeExpr.named("Self"))]))
    _laid(
        c,
        _class(
            "Good",
            methods: [_method("twice", [_self()], TypeExpr.named("Good"))],
            traits: [TypeExpr.named("Doubler")]
        )
    )
    assert_len(c.bag.items, 0)
}

fn test_a_generic_traits_argument_is_substituted_into_its_requirement() -> void {
    let c = _ctx()
    _laid(
        c,
        _trait("Producer", [_fn("next", [_self()], TypeExpr.named("T"))], generics: [generic("T")])
    )
    _laid(
        c,
        _class(
            "Ints",
            methods: [_method("next", [_self()], TypeExpr.named("str"))],
            traits: [TypeExpr.generic("Producer", [TypeExpr.named("int")])]
        )
    )

    assert_eq(_code(c, 0), E0705_TRAIT_NOT_IMPLEMENTED)
    assert_eq(
        _message(c, 0),
        "`Ints` implements `next` as `fn() -> str`, but trait " +
            "`Producer` requires `fn() -> int`: the return type is not compatible"
    )
    assert_eq(
        _note(c, 0),
        "`next` is declared `fn() -> T` in `Producer`, which " +
            "`Ints` implements as `Producer[int]`"
    )
}

fn test_a_generic_traits_argument_is_satisfied_by_the_substituted_signature() -> void {
    let c = _ctx()
    _laid(
        c,
        _trait("Producer", [_fn("next", [_self()], TypeExpr.named("T"))], generics: [generic("T")])
    )
    _laid(
        c,
        _class(
            "Ints",
            methods: [_method("next", [_self()], TypeExpr.named("int"))],
            traits: [TypeExpr.generic("Producer", [TypeExpr.named("int")])]
        )
    )
    assert_len(c.bag.items, 0)
}

fn test_a_supertraits_requirement_is_restated_in_the_extending_traits_parameters() -> void {
    let c = _ctx()
    _laid(
        c,
        _trait(
            "Comparable",
            [_fn("compare", [_self(), param("other", TypeExpr.named("T"))], TypeExpr.named("int"))],
            generics: [generic("T")]
        )
    )
    let sorted = _laid(
        c,
        _trait(
            "Sorted",
            supers: [TypeExpr.generic("Comparable", [TypeExpr.named("U")])],
            generics: [generic("U")]
        )
    )

    # The one place this port copies: a requirement set is enumerated, so an
    # inherited requirement that was not restated would never be checked.
    assert_len(sorted.members, 1)
    assert_eq(render(sorted.members[0].ty), "fn(U) -> int")
}

fn test_an_inherited_requirement_is_verified_against_the_implementer() -> void {
    let c = _ctx()
    _laid(
        c,
        _trait(
            "Comparable",
            [_fn("compare", [_self(), param("other", TypeExpr.named("T"))], TypeExpr.named("int"))],
            generics: [generic("T")]
        )
    )
    _laid(
        c,
        _trait(
            "Sorted",
            supers: [TypeExpr.generic("Comparable", [TypeExpr.named("U")])],
            generics: [generic("U")]
        )
    )
    _laid(c, _class("Money", traits: [TypeExpr.generic("Sorted", [TypeExpr.named("int")])]))

    assert_len(c.bag.items, 1)
    assert_eq(_message(c, 0), "`Money` does not implement `compare` required by trait `Sorted`")
    assert_eq(_help(c, 0), "add `fn compare(self, other: int) -> int` to `Money`")
}

fn test_the_first_trait_that_carries_a_name_is_the_one_that_answers_for_it() -> void {
    let satisfied = _ctx()
    _laid(satisfied, _trait("Needs", [_fn("area", [_self()], TypeExpr.named("float"))]))
    _laid(satisfied, _trait("Has", [_method("area", [_self()], TypeExpr.named("float"))]))
    _laid(satisfied, _class("Good", traits: [TypeExpr.named("Has"), TypeExpr.named("Needs")]))
    assert_len(satisfied.bag.items, 0)

    let unsatisfied = _ctx()
    _laid(unsatisfied, _trait("Needs", [_fn("area", [_self()], TypeExpr.named("float"))]))
    _laid(unsatisfied, _trait("Has", [_method("area", [_self()], TypeExpr.named("float"))]))
    _laid(unsatisfied, _class("Bad", traits: [TypeExpr.named("Needs"), TypeExpr.named("Has")]))
    assert_len(unsatisfied.bag.items, 1)
    assert_eq(_message(unsatisfied, 0), "`Bad` does not implement `area` required by trait `Needs`")
}

fn test_an_abstract_class_and_a_trait_are_exempt_from_conformance() -> void {
    let c = _ctx()
    _laid(c, _trait("Shape", [_fn("area", [_self()], TypeExpr.named("float"))]))
    _laid(c, _class("Partial", traits: [TypeExpr.named("Shape")], is_abstract: true))
    _laid(c, _trait("Extended", supers: [TypeExpr.named("Shape")]))

    assert_len(c.bag.items, 0)
}

fn test_a_requirement_satisfied_by_the_superclass_is_not_reported() -> void {
    let c = _ctx()
    _laid(c, _trait("Shape", [_fn("area", [_self()], TypeExpr.named("float"))]))
    _declare(c, _class("Base", methods: [_method("area", [_self()], TypeExpr.named("float"))]))
    _laid(
        c,
        _class("Circle", superclass: TypeExpr.named("Base"), traits: [TypeExpr.named("Shape")])
    )

    assert_len(c.bag.items, 0)
}

fn test_an_opaque_trait_imposes_no_requirement_this_unit_can_check() -> void {
    let c = _ctx()
    _laid(c, _class("Printer", traits: [TypeExpr.named("Printable")]))
    assert_len(c.bag.items, 0)
}

# ------------------------------------------------------------------
# Enums
# ------------------------------------------------------------------
fn test_unit_variants_are_values_and_payload_variants_are_constructors() -> void {
    let c = _ctx()
    let shape = _laid(
        c,
        _enum(
            "Shape",
            [
                variant("Empty"),
                variant("Circle", [param("radius", TypeExpr.named("float"))]),
            ]
        )
    )

    assert_eq(shape.variant_count, 2)
    assert_eq(render(shape.fields[0].ty), "Shape")
    assert_eq(render(shape.fields[1].ty), "fn(float) -> Shape")
    assert_eq(shape.fields[0].slot, 0)
    assert_eq(shape.fields[1].slot, 1)
    assert_true(shape.fields[1].is_static)
    assert_eq(shape.fields[1].visibility, VIS_PUBLIC)
    assert_len(c.bag.items, 0)
}

fn test_a_repeated_variant_is_reported() -> void {
    let c = _ctx()
    let colour = _laid(c, _enum("Colour", [variant("Red"), variant("Red")]))

    assert_len(c.bag.items, 1)
    assert_eq(_code(c, 0), E0709_DUPLICATE_MEMBER)
    assert_eq(_message(c, 0), "`Red` is already a variant of `Colour`")
    assert_eq(colour.variant_count, 1)
}

fn test_an_enums_methods_are_members() -> void {
    let c = _ctx()
    let colour = _laid(
        c,
        _enum(
            "Colour",
            [variant("Red")],
            [_method("name", [_self()], TypeExpr.named("str"))]
        )
    )

    assert_len(colour.members, 1)
    assert_eq(render(colour.members[0].ty), "fn() -> str")
    assert_eq(colour.variant_count, 1)
}
