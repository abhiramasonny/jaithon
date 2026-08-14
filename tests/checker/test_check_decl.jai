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
from std.test import assert_eq, assert_false, assert_len, assert_true
from jaithon.ast import DeclKind, Node, NodeKind, OpKind, Span, TypeExpr, TypeKind
from jaithon.ast import Visibility
from jaithon.ast import class_field, generic, param, variant
from jaithon.compile.check.decl import E0303_NON_CONSTANT_CONST, E0304_MISSING_INITIALIZER
from jaithon.compile.check.decl import E0408_RETURN_TYPE_MISMATCH, E0601_MISSING_RETURN
from jaithon.compile.check.decl import E0700_UNKNOWN_CLASS, E0703_MISSING_SELF
from jaithon.compile.check.decl import E0704_INCOMPATIBLE_OVERRIDE
from jaithon.compile.check.decl import E0705_TRAIT_NOT_IMPLEMENTED, E0706_CYCLIC_INHERITANCE
from jaithon.compile.check.decl import E0707_STATIC_WITH_SELF, E0708_SUPER_INIT_NOT_FIRST
from jaithon.compile.check.decl import E0709_DUPLICATE_MEMBER, SELF_NAME
from jaithon.compile.check.decl import check_body, check_decl, check_var_decl
from jaithon.compile.check.decl import declared_return_type, function_type, install
from jaithon.compile.check.decl import is_init_method, layout_decl, push_generics, push_self
from jaithon.compile.check.decl import register_decl, signature_of
from jaithon.compile.check.ctx import Ctx
from jaithon.compile.check.expr import E0400_TYPE_MISMATCH, VIS_PRIVATE, VIS_PUBLIC
from jaithon.compile.check.kinds import TyKind
from jaithon.compile.check.nominal import find_field, find_member
from jaithon.compile.check.render import render
from jaithon.compile.check.ty import Type, TypeDecl
from jaithon.compile.check.union import optional_of, union_of
from jaithon.compile.check.universe import Universe, dict_of, fn_of, instantiate, list_of
from jaithon.compile.check.universe import set_of, tuple_of
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
