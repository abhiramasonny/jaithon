#: `jaithon.compile.check.stmt` — patterns, and the bindings each one makes.
from std.test import assert_eq
from jaithon.ast import DeclKind, Node, NodeKind, Span, TypeExpr
from jaithon.compile.check.ctx import Ctx
from jaithon.compile.check.expr import E0400_TYPE_MISMATCH, E0410_UNKNOWN_MEMBER, E0701_PRIVATE_ACCESS, VIS_PRIVATE, VIS_PUBLIC
from jaithon.compile.check.kinds import TyKind
from jaithon.compile.check.render import render
from jaithon.compile.check.stmt import E0402_UNKNOWN_TYPE, E0502_PATTERN_ARITY, E0503_DUPLICATE_BINDING_IN_PATTERN, MAX_BINDS, install, pattern
from jaithon.compile.check.ty import FieldInfo, Type, TypeDecl
from jaithon.compile.check.union import optional_of
from jaithon.compile.check.universe import Universe, fn_of, list_of, named, tuple_of
from jaithon.compile.symbol import Symbol, SymbolKind

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------
fn _ctx() -> Ctx {
    let c = Ctx(Universe())
    install(c)
    return c
}

fn _annotated() -> Ctx {
    let c = _ctx()
    c.resolve_ast_type = _named_types
    return c
}

fn _named_types(c: Ctx, written: TypeExpr?) -> Type? {
    if written is null { return null }
    let name = written.name
    if name == "int" { return c.universe.t_int }
    if name == "float" { return c.universe.t_float }
    if name == "str" { return c.universe.t_str }
    if name == "bool" { return c.universe.t_bool }
    let decl = c.decl_named(name)
    if decl is null { return null }
    return decl.ty
}

fn _symbol(c: Ctx, name: str, ty: Type?, kind: SymbolKind = SymbolKind.Local) -> Symbol {
    let symbol = Symbol(kind, name, Span(0, 0, 1), DeclKind.Let)
    c.set_symbol_type(symbol, ty)
    return symbol
}

fn _use(c: Ctx, symbol: Symbol) -> Node {
    let node = Node.ident(symbol.name)
    c.resolution.bind(node, symbol)
    return node
}

fn _binding(c: Ctx, name: str, ty: Type?) -> Node {
    return _use(c, _symbol(c, name, ty))
}

fn _class(c: Ctx, name: str) -> TypeDecl {
    let decl = TypeDecl(name, Span(0, 0, 1))
    decl.ty = named(c.universe, TyKind.Class, name, decl)
    c.register_decl(decl)
    return decl
}

fn _field(
    decl: TypeDecl,
    name: str,
    ty: Type?,
    vis: int = VIS_PUBLIC,
    is_let: bool = false
) -> void {
    decl.fields.push(FieldInfo(name, ty, -1, vis, false, is_let, false))
}

fn _enum(c: Ctx, name: str) -> TypeDecl {
    let decl = TypeDecl(name, Span(0, 0, 1))
    decl.is_enum = true
    decl.ty = named(c.universe, TyKind.Enum, name, decl)
    c.register_decl(decl)
    return decl
}

fn _variant(c: Ctx, decl: TypeDecl, name: str, payload: list[Type]) -> void {
    var ty: Type? = null
    if payload.len() > 0 { ty = fn_of(c.universe, payload, decl.ty) }
    decl.fields.push(FieldInfo(name, ty))
    decl.variant_count += 1
}

fn _no_payload() -> list[Type] {
    return []
}

fn _ty(decl: TypeDecl) -> Type {
    let t = decl.ty
    if t is null { throw ValueError(f"the `{decl.name}` fixture interned no type") }
    return t
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

fn _seq(kind: NodeKind, elems: list[Node], rest: int = -1) -> Node {
    return Node(kind, {"elems": elems, "restIndex": rest})
}

# ------------------------------------------------------------------
# Patterns — bindings
# ------------------------------------------------------------------
fn test_a_bind_pattern_takes_the_subject_type() -> void {
    let c = _ctx()
    let pat = Node.bind("n")
    let symbol = _symbol(c, "n", null)
    c.resolution.bind(pat, symbol)
    pattern(c, pat, c.universe.t_str)
    assert_eq(render(c.type_of(pat)), "str")
    assert_eq(render(c.symbol_type(symbol)), "str")
}

fn test_an_annotated_bind_pattern_takes_the_annotation() -> void {
    let c = _annotated()
    let pat = Node.bind("n", TypeExpr.named("int"))
    pattern(c, pat, c.universe.t_str)
    assert_eq(render(c.type_of(pat)), "int")
}

fn test_an_unresolvable_annotation_reads_as_any() -> void {
    let c = _ctx()
    let pat = Node.bind("n", TypeExpr.named("Whatever"))
    pattern(c, pat, c.universe.t_str)
    assert_eq(render(c.type_of(pat)), "any")
}

fn test_a_wildcard_takes_the_subject_and_binds_nothing() -> void {
    let c = _ctx()
    let pat = Node(NodeKind.PatWildcard)
    pattern(c, pat, c.universe.t_int)
    assert_eq(render(c.type_of(pat)), "int")
    assert_eq(c.bag.items.len(), 0)
}

fn test_one_name_bound_twice_in_a_pattern_is_reported_once() -> void {
    let c = _ctx()
    let pat = _seq(NodeKind.PatTuple, [Node.bind("x"), Node.bind("x")])
    pattern(c, pat, tuple_of(c.universe, [c.universe.t_int, c.universe.t_int]))
    assert_eq(c.bag.items.len(), 1)
    assert_eq(_code(c, 0), E0503_DUPLICATE_BINDING_IN_PATTERN)
    assert_eq(_message(c, 0), "`x` is bound twice in the same pattern")
    assert_eq(c.bag.items[0].labels[0][1], "first bound here")
}

fn test_underscore_may_be_bound_as_often_as_it_likes() -> void {
    let c = _ctx()
    let pat = _seq(NodeKind.PatTuple, [Node.bind("_"), Node.bind("_a"), Node.bind("_")])
    pattern(c, pat, c.universe.t_any)
    assert_eq(c.bag.items.len(), 0)
}

fn test_each_or_alternative_binds_in_its_own_scope() -> void {
    let c = _ctx()
    let pat = _seq(NodeKind.PatOr, [Node.bind("x"), Node.bind("x")])
    pattern(c, pat, c.universe.t_int)
    assert_eq(c.bag.items.len(), 0)
}

# ------------------------------------------------------------------
# Patterns — sequences
# ------------------------------------------------------------------
fn test_a_tuple_pattern_binds_each_member_by_position() -> void {
    let c = _ctx()
    let u = c.universe
    let first = Node.bind("a")
    let second = Node.bind("b")
    let pat = _seq(NodeKind.PatTuple, [first, second])
    pattern(c, pat, tuple_of(u, [u.t_int, u.t_str]))
    assert_eq(render(c.type_of(first)), "int")
    assert_eq(render(c.type_of(second)), "str")
}

fn test_a_tuple_pattern_of_the_wrong_length_is_reported() -> void {
    let c = _ctx()
    let u = c.universe
    let pat = _seq(NodeKind.PatTuple, [Node.bind("a")])
    pattern(c, pat, tuple_of(u, [u.t_int, u.t_str]))
    assert_eq(_code(c, 0), E0502_PATTERN_ARITY)
    assert_eq(_message(c, 0), "this pattern has 1 element but `(int, str)` has 2")
}

fn test_the_arity_message_pluralises() -> void {
    let c = _ctx()
    let u = c.universe
    let pat = _seq(NodeKind.PatTuple, [Node.bind("a"), Node.bind("b"), Node.bind("d")])
    pattern(c, pat, tuple_of(u, [u.t_int, u.t_str]))
    assert_eq(_message(c, 0), "this pattern has 3 elements but `(int, str)` has 2")
}

fn test_a_rest_element_suppresses_the_tuple_arity_check() -> void {
    let c = _ctx()
    let u = c.universe
    let rest = Node.bind("more")
    let pat = _seq(NodeKind.PatTuple, [Node.bind("a"), rest], 1)
    pattern(c, pat, tuple_of(u, [u.t_int, u.t_str, u.t_str]))
    assert_eq(c.bag.items.len(), 0)
    assert_eq(render(c.type_of(rest)), "list[any]")
}

fn test_a_list_pattern_binds_the_element_type_everywhere() -> void {
    let c = _ctx()
    let u = c.universe
    let head = Node.bind("head")
    let tail = Node.bind("tail")
    let pat = _seq(NodeKind.PatList, [head, tail], 1)
    pattern(c, pat, list_of(u, u.t_str))
    assert_eq(render(c.type_of(head)), "str")
    assert_eq(render(c.type_of(tail)), "list[str]")
}

fn test_a_sequence_pattern_over_an_unknown_subject_binds_any() -> void {
    let c = _ctx()
    let element = Node.bind("a")
    pattern(c, _seq(NodeKind.PatList, [element]), c.universe.t_any)
    assert_eq(render(c.type_of(element)), "any")
    assert_eq(c.bag.items.len(), 0)
}

# ------------------------------------------------------------------
# Patterns — literals and ranges
# ------------------------------------------------------------------
fn test_a_literal_pattern_that_can_never_match_is_reported() -> void {
    let c = _ctx()
    let pat = Node(NodeKind.PatLiteral, {"value": Node.str_lit("a")})
    pattern(c, pat, c.universe.t_int)
    assert_eq(_code(c, 0), E0400_TYPE_MISMATCH)
    assert_eq(_message(c, 0), "this pattern can never match: expected `int`, found `str`")
}

fn test_a_literal_pattern_that_fits_either_way_is_accepted() -> void {
    let c = _ctx()
    let u = c.universe
    let pat = Node(NodeKind.PatLiteral, {"value": Node.int_lit(1)})
    pattern(c, pat, optional_of(u, u.t_int))
    assert_eq(c.bag.items.len(), 0)
}

fn test_a_literal_pattern_against_an_unknown_subject_is_silent() -> void {
    let c = _ctx()
    let pat = Node(NodeKind.PatLiteral, {"value": Node.str_lit("a")})
    pattern(c, pat, c.universe.t_any)
    assert_eq(c.bag.items.len(), 0)
}

fn test_a_range_pattern_checks_both_bounds() -> void {
    let c = _ctx()
    let lo = Node.int_lit(1)
    let hi = Node.int_lit(9)
    pattern(c, Node(NodeKind.PatRange, {"lo": lo, "hi": hi}), c.universe.t_int)
    assert_eq(render(c.type_of(lo)), "int")
    assert_eq(render(c.type_of(hi)), "int")
    assert_eq(c.bag.items.len(), 0)
}

# ------------------------------------------------------------------
# Patterns — classes
# ------------------------------------------------------------------
fn test_a_class_pattern_binds_fields_by_position() -> void {
    let c = _ctx()
    let u = c.universe
    let decl = _class(c, "Point")
    _field(decl, "x", u.t_int)
    _field(decl, "y", u.t_int)
    let first = Node.bind("a")
    let pat = Node(NodeKind.PatClass, {"typeName": "Point", "subPatterns": [first]})
    pattern(c, pat, _ty(decl))
    assert_eq(render(c.type_of(first)), "int")
    assert_eq(c.bag.items.len(), 0)
}

fn test_a_class_pattern_sees_inherited_fields_parents_first() -> void {
    let c = _ctx()
    let u = c.universe
    let base = _class(c, "Shape")
    _field(base, "id", u.t_int)
    let derived = _class(c, "Circle")
    derived.superclass = base
    _field(derived, "radius", u.t_float)
    let first = Node.bind("a")
    let second = Node.bind("b")
    let pat = Node(NodeKind.PatClass, {"typeName": "Circle", "subPatterns": [first, second]})
    pattern(c, pat, _ty(derived))
    assert_eq(c.bag.items.len(), 0)
    assert_eq(render(c.type_of(first)), "int")
    assert_eq(render(c.type_of(second)), "float")
}

fn test_past_the_binding_cap_duplicates_stop_being_reported() -> void {
    let c = _ctx()
    var elems: list[Node] = []
    var at = 0
    while at < MAX_BINDS {
        elems.push(Node.bind(f"n{at}"))
        at += 1
    }
    elems.push(Node.bind("n0"))    # a duplicate of a name that was recorded
    elems.push(Node.bind("over"))  # past the cap, so never recorded...
    elems.push(Node.bind("over"))  # ...and its duplicate goes unseen
    pattern(c, _seq(NodeKind.PatList, elems), c.universe.t_any)
    assert_eq(c.bag.items.len(), 1)
    assert_eq(_message(c, 0), "`n0` is bound twice in the same pattern")
}

fn test_a_class_pattern_binds_fields_by_name() -> void {
    let c = _ctx()
    let u = c.universe
    let decl = _class(c, "Point")
    _field(decl, "x", u.t_int)
    _field(decl, "label", u.t_str)
    let bound = Node.bind("text")
    let pat = Node(
        NodeKind.PatClass,
        {"typeName": "Point", "fieldNames": ["label"], "subPatterns": [bound]}
    )
    pattern(c, pat, _ty(decl))
    assert_eq(render(c.type_of(bound)), "str")
}

fn test_a_named_field_with_no_sub_pattern_binds_the_name_itself() -> void {
    let c = _ctx()
    let decl = _class(c, "Point")
    _field(decl, "x", c.universe.t_int)
    let subs: list[Node?] = [null]
    let pat = Node(
        NodeKind.PatClass,
        {"typeName": "Point", "fieldNames": ["x"], "subPatterns": subs}
    )
    pattern(c, pat, _ty(decl))
    assert_eq(c.bag.items.len(), 0)
}

fn test_a_class_pattern_naming_an_unknown_field_is_reported() -> void {
    let c = _ctx()
    let decl = _class(c, "Point")
    _field(decl, "width", c.universe.t_int)
    let pat = Node(
        NodeKind.PatClass,
        {"typeName": "Point", "fieldNames": ["widht"], "subPatterns": [Node.bind("w")]}
    )
    pattern(c, pat, _ty(decl))
    assert_eq(_code(c, 0), E0410_UNKNOWN_MEMBER)
    assert_eq(_message(c, 0), "`Point` has no field `widht`")
    assert_eq(_help(c, 0), "did you mean `width`?")
}

fn test_a_class_pattern_binding_more_than_there_are_fields_is_reported() -> void {
    let c = _ctx()
    let decl = _class(c, "Unit")
    _field(decl, "only", c.universe.t_int)
    let pat = Node(
        NodeKind.PatClass,
        {"typeName": "Unit", "subPatterns": [Node.bind("a"), Node.bind("b")]}
    )
    pattern(c, pat, _ty(decl))
    assert_eq(_code(c, 0), E0502_PATTERN_ARITY)
    assert_eq(_message(c, 0), "`Unit` has 1 field but the pattern binds 2")
}

fn test_a_class_pattern_reaching_a_private_field_is_reported() -> void {
    let c = _ctx()
    let decl = _class(c, "Box")
    _field(decl, "secret", c.universe.t_int, VIS_PRIVATE)
    let pat = Node(NodeKind.PatClass, {"typeName": "Box", "subPatterns": [Node.bind("s")]})
    pattern(c, pat, _ty(decl))
    assert_eq(_code(c, 0), E0701_PRIVATE_ACCESS)
    assert_eq(_message(c, 0), "`secret` is private to `Box`")
}

fn test_a_class_pattern_naming_an_unknown_type_is_reported() -> void {
    let c = _ctx()
    let inner = Node.bind("a")
    let pat = Node(NodeKind.PatClass, {"typeName": "Nowhere", "subPatterns": [inner]})
    pattern(c, pat, c.universe.t_int)
    assert_eq(_code(c, 0), E0402_UNKNOWN_TYPE)
    assert_eq(_message(c, 0), "unknown type `Nowhere` in pattern")
    assert_eq(render(c.type_of(inner)), "any")
}

fn test_an_unknown_type_against_an_unknown_subject_is_silent() -> void {
    let c = _ctx()
    let pat = Node(NodeKind.PatClass, {"typeName": "Nowhere", "subPatterns": []})
    pattern(c, pat, c.universe.t_any)
    assert_eq(c.bag.items.len(), 0)
}

fn test_a_whole_enum_pattern_with_no_variant_binds_nothing() -> void {
    let c = _ctx()
    let decl = _enum(c, "Colour")
    _variant(c, decl, "Red", _no_payload())
    let pat = Node(NodeKind.PatClass, {"typeName": "Colour", "subPatterns": []})
    pattern(c, pat, _ty(decl))
    assert_eq(c.bag.items.len(), 0)
}

# ------------------------------------------------------------------
# Patterns — enum variants
# ------------------------------------------------------------------
fn test_an_enum_pattern_binds_its_payload() -> void {
    let c = _ctx()
    let u = c.universe
    let decl = _enum(c, "Shape")
    _variant(c, decl, "Point", _no_payload())
    _variant(c, decl, "Circle", [u.t_float])
    let radius = Node.bind("r")
    let pat = Node(
        NodeKind.PatEnum,
        {"typeName": "Shape", "variantName": "Circle", "subPatterns": [radius]}
    )
    pattern(c, pat, _ty(decl))
    assert_eq(c.bag.items.len(), 0)
    assert_eq(render(c.type_of(radius)), "float")
}

fn test_an_enum_pattern_of_the_wrong_arity_is_reported() -> void {
    let c = _ctx()
    let decl = _enum(c, "Shape")
    _variant(c, decl, "Point", _no_payload())
    let pat = Node(
        NodeKind.PatEnum,
        {"typeName": "Shape", "variantName": "Point", "subPatterns": [Node.bind("x")]}
    )
    pattern(c, pat, _ty(decl))
    assert_eq(_code(c, 0), E0502_PATTERN_ARITY)
    assert_eq(_message(c, 0), "`Shape.Point` takes 0 fields but the pattern binds 1")
}

fn test_an_unknown_variant_is_reported_with_a_suggestion() -> void {
    let c = _ctx()
    let decl = _enum(c, "Colour")
    _variant(c, decl, "Green", _no_payload())
    let pat = Node(NodeKind.PatEnum, {"typeName": "Colour", "variantName": "Greeen"})
    pattern(c, pat, _ty(decl))
    assert_eq(_code(c, 0), E0410_UNKNOWN_MEMBER)
    assert_eq(_message(c, 0), "`Colour` has no variant `Greeen`")
    assert_eq(_help(c, 0), "did you mean `Green`?")
}

fn test_a_static_member_is_not_a_variant() -> void {
    let c = _ctx()
    let decl = _enum(c, "Colour")
    _variant(c, decl, "Red", _no_payload())
    decl.fields.push(FieldInfo("DEFAULT", c.universe.t_int))
    let pat = Node(NodeKind.PatEnum, {"typeName": "Colour", "variantName": "DEFAULT"})
    pattern(c, pat, _ty(decl))
    assert_eq(_code(c, 0), E0410_UNKNOWN_MEMBER)
}

fn test_an_enum_pattern_finds_its_declaration_through_the_subject() -> void {
    let c = _ctx()
    let decl = TypeDecl("Colour", Span(0, 0, 1))
    decl.is_enum = true
    decl.ty = named(c.universe, TyKind.Enum, "Colour", decl)
    _variant(c, decl, "Red", _no_payload())
    let pat = Node(NodeKind.PatEnum, {"typeName": "Colour", "variantName": "Red"})
    pattern(c, pat, _ty(decl))
    assert_eq(c.bag.items.len(), 0)
}

# ------------------------------------------------------------------
# Patterns — expression targets
# ------------------------------------------------------------------
fn test_an_expression_in_pattern_position_keeps_its_own_type() -> void {
    let c = _ctx()
    let target = _binding(c, "n", c.universe.t_int)
    pattern(c, target, c.universe.t_str)
    assert_eq(render(c.type_of(target)), "int")
}
