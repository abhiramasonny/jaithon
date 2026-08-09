#: `jaithon.compile.check.stmt` — statements, control flow and narrowing.
#:
#: The cases worth writing down are the ones where the C does something a
#: re-derivation would not: `if x == null { return }` narrows the *rest of the
#: block* and nothing further, a store to a narrowed binding is checked against
#: the declaration rather than the narrowed view and then ends the fact, a
#: guarded match arm does not cover its own pattern, a match over a list is
#: total when `[]` and `[a, ...r]` are both present, each alternative of an `or`
#: pattern binds its names in its own scope, and a `var x = null` is `any` while
#: a `let x = null` is `null`. Each of those is a paragraph of
#: `src/sema/check_stmt.c` and `src/sema/check.c` that reads like an accident
#: and is not.
#:
#: Narrowing gets the most attention here, because a leaked fact is a wrong type
#: on code that never tested anything: every branch test asserts both what the
#: binding reads as *inside* the branch and what it reads as after.
#:
#: The diagnostics are asserted by code **and by message text**, because both
#: are a contract with the C front end: Phase 4 turns `tests/errors` into
#: goldens and a reworded message fails there.
from std.test import assert_eq, assert_false, assert_true
from jaithon.ast import CompKind, DeclKind, Node, NodeKind, OpKind, Span, TypeExpr, TypeKind
from jaithon.ast import catch_clause, comp_clause, match_arm, param
from jaithon.compile.check.expr import Ctx, E0400_TYPE_MISMATCH, E0401_CONDITION_NOT_BOOL
from jaithon.compile.check.expr import E0405_NOT_ITERABLE, E0406_BAD_OPERAND_TYPES
from jaithon.compile.check.expr import E0410_UNKNOWN_MEMBER, E0411_INT_FLOAT_MIX
from jaithon.compile.check.expr import E0701_PRIVATE_ACCESS, VIS_PRIVATE, VIS_PUBLIC
from jaithon.compile.check.expr import W0102_UNREACHABLE_CODE, expr
from jaithon.compile.check.fold import const_eval
from jaithon.compile.check.kinds import TyKind
from jaithon.compile.check.render import render
from jaithon.compile.check.stmt import E0301_ASSIGN_TO_IMMUTABLE, E0303_NON_CONSTANT_CONST
from jaithon.compile.check.stmt import E0304_MISSING_INITIALIZER, E0402_UNKNOWN_TYPE
from jaithon.compile.check.stmt import E0408_RETURN_TYPE_MISMATCH, E0500_UNREACHABLE_ARM
from jaithon.compile.check.stmt import E0501_NON_EXHAUSTIVE_MATCH, E0502_PATTERN_ARITY
from jaithon.compile.check.stmt import E0503_DUPLICATE_BINDING_IN_PATTERN, E0601_MISSING_RETURN
from jaithon.compile.check.stmt import E0702_UNDECLARED_FIELD, MAX_BINDS, block
from jaithon.compile.check.stmt import declared_return_type, function, function_type, install
from jaithon.compile.check.stmt import is_init_method
from jaithon.compile.check.stmt import match_expr, pattern, push_generics, stmt
from jaithon.compile.check.ty import FieldInfo, MemberInfo, Type, TypeDecl
from jaithon.compile.check.union import optional_of
from jaithon.compile.check.universe import Universe, fn_of, list_of, named, tuple_of
from jaithon.compile.resolve import Symbol, SymbolKind

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

# `install` is part of the file under test: without it an expression cannot
# reach a statement, and half of what is asserted below runs through that seam.
fn _ctx() -> Ctx {
    let c = Ctx(Universe())
    install(c)
    return c
}

# A `resolve_ast_type` that knows the four scalars and the registered classes,
# so an annotation is worth writing at all. `checker.jai` supplies the real one.
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

# A fresh use of `symbol`, bound the way the resolver would have bound it. Two
# uses share one symbol, which is what narrowing needs.
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

# A variant is a field whose type is the `Fn` that builds it, so its parameters
# are the fields a pattern destructures. A payload-free variant has no type.
fn _variant(c: Ctx, decl: TypeDecl, name: str, payload: list[Type]) -> void {
    var ty: Type? = null
    if payload.len() > 0 { ty = fn_of(c.universe, payload, decl.ty) }
    decl.fields.push(FieldInfo(name, ty))
    decl.variant_count += 1
}

fn _no_payload() -> list[Type] {
    return []
}

# `TypeDecl.ty` is optional because a declaration exists before it is interned;
# every fixture here interns one, so this is the unwrap.
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

# `x != null`, the condition every narrowing test is built from.
fn _not_null(c: Ctx, symbol: Symbol) -> Node {
    return Node.binary(OpKind.Ne, _use(c, symbol), Node.null_lit())
}

fn _is_null(c: Ctx, symbol: Symbol) -> Node {
    return Node.binary(OpKind.Eq, _use(c, symbol), Node.null_lit())
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
    # The default `resolve_ast_type` answers null, and `jaiChkOrAny` turns that
    # into `any` rather than falling back to the subject.
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
    # The second points back at the first.
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
    # `A(x) | B(x)` binds the same name twice by design.
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
    # `...rest` collects the remainder, so it is a list of elements — and for a
    # tuple the element is asked for with no position, which is `any`.
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
    # `1` against `int?` fits one way; that is enough.
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
    # `layoutClassFields` in the C copies a parent's entries in ahead of the
    # child's, and that order is what a positional pattern indexes.
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
    # The sub-patterns are still walked, so a name inside one is still bound.
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
    # Past `variant_count` the field list holds statics, which a pattern cannot
    # match: the C tests the index against `variantCount` for exactly this.
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
    # Deliberately not registered: the subject is the only route to it.
    let pat = Node(NodeKind.PatEnum, {"typeName": "Colour", "variantName": "Red"})
    pattern(c, pat, _ty(decl))
    assert_eq(c.bag.items.len(), 0)
}

# ------------------------------------------------------------------
# Patterns — expression targets
# ------------------------------------------------------------------
fn test_an_expression_in_pattern_position_keeps_its_own_type() -> void {
    let c = _ctx()
    # A destructuring assignment target reuses expression nodes.
    let target = _binding(c, "n", c.universe.t_int)
    pattern(c, target, c.universe.t_str)
    assert_eq(render(c.type_of(target)), "int")
}

# ------------------------------------------------------------------
# Match — the type it answers
# ------------------------------------------------------------------
fn test_a_match_statement_is_void_and_still_checks_its_arms() -> void {
    let c = _ctx()
    let u = c.universe
    let arms = [match_arm(Node(NodeKind.PatWildcard), Node.int_lit(1))]
    let node = Node(NodeKind.Match, {"subject": Node.str_lit("x"), "arms": arms})
    assert_true(match_expr(c, node, true) == u.t_void)
}

fn test_a_match_expression_joins_its_arm_bodies() -> void {
    let c = _ctx()
    let arms = [
        match_arm(
            Node(NodeKind.PatLiteral, {"value": Node.int_lit(1)}),
            Node.str_lit("one")
        ),
        match_arm(Node(NodeKind.PatWildcard), Node.str_lit("many")),
    ]
    let node = Node(NodeKind.MatchExpr, {"subject": Node.int_lit(1), "arms": arms})
    assert_eq(render(match_expr(c, node, false)), "str")
}

fn test_a_match_that_may_fall_through_yields_an_optional() -> void {
    let c = _ctx()
    let arms = [
        match_arm(
            Node(NodeKind.PatLiteral, {"value": Node.int_lit(1)}),
            Node.str_lit("one")
        ),
    ]
    let node = Node(
        NodeKind.MatchExpr,
        {"subject": _binding(c, "n", c.universe.t_int), "arms": arms}
    )
    assert_eq(render(match_expr(c, node, false)), "str?")
}

fn test_a_catch_all_arm_makes_a_match_total() -> void {
    let c = _ctx()
    let arms = [match_arm(Node.bind("other"), Node.str_lit("any"))]
    let node = Node(
        NodeKind.MatchExpr,
        {"subject": _binding(c, "n", c.universe.t_int), "arms": arms}
    )
    assert_eq(render(match_expr(c, node, false)), "str")
}

fn test_an_unknown_subject_is_not_evidence_of_a_missing_arm() -> void {
    let c = _ctx()
    let arms = [
        match_arm(
            Node(NodeKind.PatLiteral, {"value": Node.int_lit(1)}),
            Node.str_lit("one")
        ),
    ]
    let node = Node(NodeKind.MatchExpr, {"subject": Node.ident("mystery"), "arms": arms})
    # Nothing was reported for the coverage, so the arms are taken at their word
    # here too rather than infecting the result with `| null`.
    assert_eq(render(match_expr(c, node, false)), "str")
}

fn test_an_empty_match_is_any() -> void {
    let c = _ctx()
    let node = Node(NodeKind.MatchExpr, {"subject": Node.int_lit(1), "arms": []})
    assert_true(match_expr(c, node, false) == c.universe.t_any)
}

fn test_a_void_arm_body_contributes_nothing_to_the_join() -> void {
    let c = _ctx()
    let arms = [
        match_arm(
            Node(NodeKind.PatLiteral, {"value": Node.int_lit(1)}),
            Node.str_lit("one")
        ),
        match_arm(Node.bind("other"), Node.str_lit("many")),
    ]
    let node = Node(NodeKind.MatchExpr, {"subject": Node.int_lit(1), "arms": arms})
    assert_eq(render(match_expr(c, node, false)), "str")
}

fn test_sequence_arms_can_exhaust_a_list() -> void {
    let c = _ctx()
    let u = c.universe
    let arms = [
        match_arm(_seq(NodeKind.PatList, []), Node.str_lit("empty")),
        match_arm(
            _seq(NodeKind.PatList, [Node.bind("a"), Node.bind("r")], 1),
            Node.str_lit("more")
        ),
    ]
    let subject = _binding(c, "xs", list_of(u, u.t_int))
    let node = Node(NodeKind.MatchExpr, {"subject": subject, "arms": arms})
    # `[]` covers length zero and `[a, ...r]` takes everything from one up.
    assert_eq(render(match_expr(c, node, false)), "str")
}

fn test_a_refutable_sequence_arm_does_not_cover_its_length() -> void {
    let c = _ctx()
    let u = c.universe
    let one = Node(NodeKind.PatLiteral, {"value": Node.int_lit(1)})
    let arms = [
        match_arm(_seq(NodeKind.PatList, []), Node.str_lit("empty")),
        match_arm(_seq(NodeKind.PatList, [one, Node.bind("r")], 1), Node.str_lit("more")),
    ]
    let subject = _binding(c, "xs", list_of(u, u.t_int))
    let node = Node(NodeKind.MatchExpr, {"subject": subject, "arms": arms})
    # `[1, ...r]` still falls through for a list whose head is not 1.
    assert_eq(render(match_expr(c, node, false)), "str?")
}

# ------------------------------------------------------------------
# Match — coverage diagnostics
# ------------------------------------------------------------------
fn test_an_arm_after_a_catch_all_is_unreachable() -> void {
    let c = _ctx()
    let arms = [
        match_arm(Node.bind("other"), Node.int_lit(0)),
        match_arm(Node(NodeKind.PatWildcard), Node.int_lit(1)),
    ]
    let node = Node(NodeKind.Match, {"subject": Node.int_lit(1), "arms": arms})
    match_expr(c, node, true)
    assert_eq(_code(c, 0), E0500_UNREACHABLE_ARM)
    assert_eq(_message(c, 0), "unreachable match arm: an earlier arm already matches")
    assert_eq(_help(c, 0), "remove it, or move it above the arm that subsumes it")
}

fn test_only_a_catch_all_is_subsumed_by_a_catch_all() -> void {
    let c = _ctx()
    let arms = [
        match_arm(Node.bind("other"), Node.int_lit(0)),
        match_arm(Node(NodeKind.PatLiteral, {"value": Node.int_lit(1)}), Node.int_lit(1)),
    ]
    let node = Node(NodeKind.Match, {"subject": Node.int_lit(1), "arms": arms})
    match_expr(c, node, true)
    # `patternCovered` consults `sawCatchAll` only for a wildcard or a binding;
    # every other shape is asked about its own kind of coverage, so a literal
    # after a catch-all goes unreported. That is the C's behaviour and not an
    # oversight of this port.
    assert_eq(c.bag.items.len(), 0)
}

fn test_a_repeated_literal_arm_is_unreachable() -> void {
    let c = _ctx()
    let arms = [
        match_arm(
            Node(NodeKind.PatLiteral, {"value": Node.str_lit("a")}),
            Node.int_lit(0)
        ),
        match_arm(
            Node(NodeKind.PatLiteral, {"value": Node.str_lit("a")}),
            Node.int_lit(1)
        ),
        match_arm(Node.bind("rest"), Node.int_lit(2)),
    ]
    let node = Node(NodeKind.Match, {"subject": Node.str_lit("x"), "arms": arms})
    match_expr(c, node, true)
    assert_eq(c.bag.items.len(), 1)
    assert_eq(_code(c, 0), E0500_UNREACHABLE_ARM)
}

fn test_a_guarded_arm_does_not_cover_its_pattern() -> void {
    let c = _ctx()
    let guarded = match_arm(Node.bind("x"), Node.int_lit(0), Node.bool_lit(true))
    let arms = [guarded, match_arm(Node.bind("y"), Node.int_lit(1))]
    let node = Node(NodeKind.Match, {"subject": Node.int_lit(1), "arms": arms})
    match_expr(c, node, true)
    # The guard may be false, so the second arm is still reachable.
    assert_eq(c.bag.items.len(), 0)
}

fn test_a_match_guard_must_be_a_bool() -> void {
    let c = _ctx()
    let arms = [match_arm(Node.bind("x"), Node.int_lit(0), Node.int_lit(1))]
    let node = Node(NodeKind.Match, {"subject": Node.int_lit(1), "arms": arms})
    match_expr(c, node, true)
    assert_eq(_code(c, 0), E0401_CONDITION_NOT_BOOL)
    assert_eq(_message(c, 0), "the condition of a match guard must be `bool`, found `int`")
}

fn test_a_bool_subject_must_cover_both_values() -> void {
    let c = _ctx()
    let arms = [
        match_arm(
            Node(NodeKind.PatLiteral, {"value": Node.bool_lit(true)}),
            Node.int_lit(1)
        ),
    ]
    let node = Node(NodeKind.Match, {"subject": _binding(c, "b", c.universe.t_bool), "arms": arms})
    match_expr(c, node, true)
    assert_eq(_code(c, 0), E0501_NON_EXHAUSTIVE_MATCH)
    assert_eq(_message(c, 0), "non-exhaustive match: `false` is not covered")
    assert_eq(_help(c, 0), "add the missing arm, or a `_ => ...` arm")
}

fn test_an_enum_match_names_every_variant_it_missed() -> void {
    let c = _ctx()
    let decl = _enum(c, "Colour")
    _variant(c, decl, "Red", _no_payload())
    _variant(c, decl, "Green", _no_payload())
    _variant(c, decl, "Blue", _no_payload())
    let red = Node(NodeKind.PatEnum, {"typeName": "Colour", "variantName": "Red"})
    let arms = [match_arm(red, Node.int_lit(0))]
    let node = Node(NodeKind.Match, {"subject": _binding(c, "k", decl.ty), "arms": arms})
    match_expr(c, node, true)
    assert_eq(_code(c, 0), E0501_NON_EXHAUSTIVE_MATCH)
    assert_eq(_message(c, 0), "non-exhaustive match: `Colour.Green`, `Colour.Blue` not covered")
    assert_eq(_help(c, 0), "add the missing arms, or a `_ => ...` arm")
    assert_eq(c.bag.items[0].labels[0][1], "`Colour` is declared here")
}

fn test_one_missing_variant_reads_in_the_singular() -> void {
    let c = _ctx()
    let decl = _enum(c, "Colour")
    _variant(c, decl, "Red", _no_payload())
    _variant(c, decl, "Green", _no_payload())
    let red = Node(NodeKind.PatEnum, {"typeName": "Colour", "variantName": "Red"})
    let arms = [match_arm(red, Node.int_lit(0))]
    let node = Node(NodeKind.Match, {"subject": _binding(c, "k", decl.ty), "arms": arms})
    match_expr(c, node, true)
    assert_eq(_message(c, 0), "non-exhaustive match: `Colour.Green` not covered")
    assert_eq(_help(c, 0), "add the missing arm, or a `_ => ...` arm")
}

fn test_a_complete_enum_match_reports_nothing() -> void {
    let c = _ctx()
    let decl = _enum(c, "Colour")
    _variant(c, decl, "Red", _no_payload())
    _variant(c, decl, "Green", _no_payload())
    let red = Node(NodeKind.PatEnum, {"typeName": "Colour", "variantName": "Red"})
    let green = Node(NodeKind.PatEnum, {"typeName": "Colour", "variantName": "Green"})
    let arms = [match_arm(red, Node.int_lit(0)), match_arm(green, Node.int_lit(1))]
    let node = Node(NodeKind.Match, {"subject": _binding(c, "k", decl.ty), "arms": arms})
    match_expr(c, node, true)
    assert_eq(c.bag.items.len(), 0)
}

fn test_a_payload_variant_arm_is_never_subsumed() -> void {
    let c = _ctx()
    let u = c.universe
    let decl = _enum(c, "Shape")
    _variant(c, decl, "Circle", [u.t_float])
    let first = Node(
        NodeKind.PatEnum,
        {"typeName": "Shape", "variantName": "Circle", "subPatterns": [Node.bind("r")]}
    )
    let second = Node(
        NodeKind.PatEnum,
        {"typeName": "Shape", "variantName": "Circle", "subPatterns": [Node.bind("s")]}
    )
    let arms = [match_arm(first, Node.int_lit(0)), match_arm(second, Node.int_lit(1))]
    let node = Node(NodeKind.Match, {"subject": _binding(c, "k", decl.ty), "arms": arms})
    match_expr(c, node, true)
    # A payload arm may have narrowed on its sub-patterns, so only a
    # payload-free variant is fully covered by an earlier arm.
    assert_eq(c.bag.items.len(), 0)
}

fn test_an_or_pattern_is_covered_only_when_every_alternative_is() -> void {
    let c = _ctx()
    let one = Node(NodeKind.PatLiteral, {"value": Node.int_lit(1)})
    let two = Node(NodeKind.PatLiteral, {"value": Node.int_lit(2)})
    let alt = _seq(NodeKind.PatOr, [Node(NodeKind.PatLiteral, {"value": Node.int_lit(1)}), two])
    let arms = [
        match_arm(one, Node.int_lit(0)),
        match_arm(alt, Node.int_lit(1)),
        match_arm(Node.bind("rest"), Node.int_lit(2)),
    ]
    let node = Node(NodeKind.Match, {"subject": Node.int_lit(1), "arms": arms})
    match_expr(c, node, true)
    # `2` was never taken, so `1 | 2` is still reachable.
    assert_eq(c.bag.items.len(), 0)
}

# ------------------------------------------------------------------
# Match — narrowing
# ------------------------------------------------------------------
fn test_an_arm_body_cannot_leak_a_narrowing_into_the_next_arm() -> void {
    let c = _ctx()
    let u = c.universe
    let symbol = _symbol(c, "v", optional_of(u, u.t_int))
    # A bare `if` whose `then` diverges deliberately leaves its else-edge facts
    # applied for whatever follows; the arm's own restore is what stops that
    # from being the next arm.
    let guard = Node.if_stmt(_is_null(c, symbol), Node.block([Node.return_stmt()]))
    let after = _use(c, symbol)
    let arms = [
        match_arm(Node(NodeKind.PatLiteral, {"value": Node.int_lit(1)}), guard),
        match_arm(Node.bind("rest"), after),
    ]
    let node = Node(NodeKind.Match, {"subject": Node.int_lit(1), "arms": arms})
    match_expr(c, node, true)
    assert_eq(render(c.type_of(after)), "int?")
}

fn test_a_pattern_binding_rebinds_its_symbol_outright() -> void {
    let c = _ctx()
    let u = c.universe
    let symbol = _symbol(c, "v", optional_of(u, u.t_int))
    let pat = Node.bind("v")
    c.resolution.bind(pat, symbol)
    pattern(c, pat, u.t_str)
    # A binding is a declaration, not a fact: it writes `Symbol.type` directly
    # and the narrowing save stack never sees it, so no `narrow_restore` undoes
    # it. Nothing outer is clobbered in practice because the resolver gives a
    # pattern its own symbol in its own scope.
    assert_eq(render(c.symbol_type(symbol)), "str")
}

# ------------------------------------------------------------------
# let, var and const
# ------------------------------------------------------------------
fn test_a_let_takes_the_type_of_its_initialiser() -> void {
    let c = _ctx()
    let pat = Node.bind("n")
    let symbol = _symbol(c, "n", null)
    let node = Node.var_decl(DeclKind.Let, pat, null, Node.int_lit(3))
    c.resolution.bind(node, symbol)
    stmt(c, node)
    assert_eq(render(c.type_of(node)), "int")
    assert_eq(render(c.symbol_type(symbol)), "int")
    assert_eq(render(c.type_of(pat)), "int")
}

fn test_an_unannotated_var_initialised_to_null_is_any() -> void {
    let c = _ctx()
    let node = Node.var_decl(DeclKind.Var, Node.bind("v"), null, Node.null_lit())
    stmt(c, node)
    # Pinning it to `null` would reject every later assignment.
    assert_eq(render(c.type_of(node)), "any")
}

fn test_an_unannotated_let_initialised_to_null_stays_null() -> void {
    let c = _ctx()
    let node = Node.var_decl(DeclKind.Let, Node.bind("v"), null, Node.null_lit())
    stmt(c, node)
    assert_eq(render(c.type_of(node)), "null")
}

fn test_an_annotation_wins_over_the_initialiser() -> void {
    let c = _annotated()
    let node = Node.var_decl(DeclKind.Let, Node.bind("x"), TypeExpr.named("float"), Node.int_lit(1))
    stmt(c, node)
    assert_eq(render(c.type_of(node)), "float")
    # §2.2: the int literal is respelled rather than reported.
    assert_eq(c.bag.items.len(), 0)
    assert_true(node.child("init")?.kind == NodeKind.FloatLit)
}

fn test_a_binding_that_does_not_fit_its_annotation_is_reported() -> void {
    let c = _annotated()
    let node = Node.var_decl(DeclKind.Let, Node.bind("x"), TypeExpr.named("int"), Node.str_lit("a"))
    stmt(c, node)
    assert_eq(_code(c, 0), E0400_TYPE_MISMATCH)
    assert_eq(_message(c, 0), "mismatched types in this binding: expected `int`, found `str`")
}

fn test_a_declaration_with_no_initialiser_and_no_annotation_is_any() -> void {
    let c = _ctx()
    let node = Node.var_decl(DeclKind.Var, Node.bind("x"))
    stmt(c, node)
    assert_eq(render(c.type_of(node)), "any")
}

fn test_a_const_is_folded_and_kept_for_its_uses() -> void {
    let c = _ctx()
    let pat = Node.bind("K")
    let symbol = _symbol(c, "K", null)
    symbol.mutability = DeclKind.Const
    let init = Node.binary(OpKind.Add, Node.int_lit(2), Node.int_lit(3))
    let node = Node.var_decl(DeclKind.Const, pat, null, init)
    c.resolution.bind(node, symbol)
    stmt(c, node)
    assert_eq(c.bag.items.len(), 0)
    # The initialiser became the literal, and the value is where a use can find
    # it: `Ctx.consts` is this port's `Symbol.constValue`.
    assert_true(init.kind == NodeKind.IntLit)
    assert_eq(init.number("intLit"), 5)
    assert_eq(const_eval(c.consts["K"]).as_int(), 5)
}

fn test_a_const_use_inlines_the_folded_value() -> void {
    let c = _ctx()
    let pat = Node.bind("K")
    let symbol = _symbol(c, "K", null)
    symbol.mutability = DeclKind.Const
    let node = Node.var_decl(DeclKind.Const, pat, null, Node.int_lit(8))
    c.resolution.bind(node, symbol)
    stmt(c, node)
    let site = _use(c, symbol)
    expr(c, site)
    assert_true(site.kind == NodeKind.IntLit)
    assert_eq(site.number("intLit"), 8)
}

fn test_a_const_with_a_non_constant_initialiser_is_reported() -> void {
    let c = _ctx()
    let init = Node.call(_binding(c, "now", c.universe.t_any), [])
    let node = Node.var_decl(DeclKind.Const, Node.bind("K"), null, init)
    stmt(c, node)
    assert_eq(_code(c, 0), E0303_NON_CONSTANT_CONST)
    assert_eq(_message(c, 0), "the initialiser of a `const` must be a constant expression")
    assert_eq(_help(c, 0), "use `let` if the value is only known at run time")
}

fn test_a_const_with_no_initialiser_is_reported() -> void {
    let c = _ctx()
    stmt(c, Node.var_decl(DeclKind.Const, Node.bind("K")))
    assert_eq(_code(c, 0), E0304_MISSING_INITIALIZER)
    assert_eq(_message(c, 0), "a `const` declaration needs an initialiser")
}

fn test_an_overflowing_const_reports_rather_than_falling_silent() -> void {
    let c = _ctx()
    # A `const` is the one place a failed fold is worth E0900: elsewhere the VM
    # raises at run time and the checker leaves the expression alone.
    let init = Node.binary(OpKind.Pow, Node.int_lit(2), Node.int_lit(70))
    stmt(c, Node.var_decl(DeclKind.Const, Node.bind("K"), null, init))
    assert_eq(_code(c, 0), "E0900")
    # And E0303 is not piled on top of it.
    assert_eq(c.bag.items.len(), 1)
}

# ------------------------------------------------------------------
# Assignment
# ------------------------------------------------------------------
fn test_a_store_is_checked_against_the_declaration_not_the_narrowing() -> void {
    let c = _ctx()
    let u = c.universe
    let symbol = _symbol(c, "v", optional_of(u, u.t_int))
    let after = _use(c, symbol)
    let then = Node.block([Node.assign(_use(c, symbol), Node.null_lit()), Node.expr_stmt(after)])
    stmt(c, Node.if_stmt(_not_null(c, symbol), then))
    # Inside `if v != null` the binding reads `int`, but the store is into the
    # `int?` the declaration promised, so it is legal...
    assert_eq(c.bag.items.len(), 0)
    # ...and it ends the fact.
    assert_eq(render(c.type_of(after)), "int?")
}

fn test_a_store_that_does_not_fit_the_declaration_is_reported() -> void {
    let c = _ctx()
    let target = _binding(c, "n", c.universe.t_int)
    stmt(c, Node.assign(target, Node.str_lit("a")))
    assert_eq(_code(c, 0), E0400_TYPE_MISMATCH)
    assert_eq(_message(c, 0), "mismatched types in this assignment: expected `int`, found `str`")
}

fn test_a_compound_assignment_widens_an_int_operand_into_a_float_target() -> void {
    let c = _ctx()
    let u = c.universe
    let source = Node.int_lit(1)
    let node = Node.compound_assign(OpKind.Add, _binding(c, "x", u.t_float), source)
    stmt(c, node)
    assert_eq(c.bag.items.len(), 0)
    assert_true(source.kind == NodeKind.FloatLit)
}

fn test_a_float_result_does_not_fit_an_int_target() -> void {
    let c = _ctx()
    let u = c.universe
    let target = _binding(c, "total", u.t_int)
    stmt(c, Node.compound_assign(OpKind.Add, target, Node.float_lit(1.5)))
    assert_eq(_code(c, 0), E0411_INT_FLOAT_MIX)
    assert_eq(_message(c, 0), "cannot apply `+=` to `int` and `float`")
    assert_eq(
        _help(c, 0),
        "the result is `float` and `total` is `int`; declare it `float`, or truncate with `int(...)`"
    )
}

fn test_a_compound_assignment_with_no_such_operator_is_reported() -> void {
    let c = _ctx()
    let u = c.universe
    let target = _binding(c, "s", u.t_str)
    stmt(c, Node.compound_assign(OpKind.Sub, target, _binding(c, "t", u.t_str)))
    assert_eq(_code(c, 0), E0406_BAD_OPERAND_TYPES)
    assert_eq(_message(c, 0), "operator `-=` does not apply to `str` and `str`")
}

fn test_a_compound_assignment_on_an_unknown_operand_is_silent() -> void {
    let c = _ctx()
    let target = _binding(c, "x", c.universe.t_any)
    stmt(c, Node.compound_assign(OpKind.Add, target, Node.str_lit("a")))
    assert_eq(c.bag.items.len(), 0)
}

fn test_a_field_store_checks_the_field_type() -> void {
    let c = _ctx()
    let u = c.universe
    let decl = _class(c, "Point")
    _field(decl, "x", u.t_int)
    let target = Node.member(_binding(c, "p", decl.ty), "x")
    stmt(c, Node.assign(target, Node.str_lit("a")))
    assert_eq(_code(c, 0), E0400_TYPE_MISMATCH)
    assert_eq(_message(c, 0), "mismatched types in this assignment: expected `int`, found `str`")
}

fn test_a_store_into_a_let_field_from_outside_the_class_is_reported() -> void {
    let c = _ctx()
    let u = c.universe
    let decl = _class(c, "Point")
    _field(decl, "x", u.t_int, VIS_PUBLIC, true)
    let target = Node.member(_binding(c, "p", decl.ty), "x")
    stmt(c, Node.assign(target, Node.int_lit(1)))
    assert_eq(_code(c, 0), E0301_ASSIGN_TO_IMMUTABLE)
    assert_eq(_message(c, 0), "`x` is declared `let` and cannot be reassigned")
    assert_eq(_help(c, 0), "declare it `var` if it needs to change after `init`")
}

fn test_a_let_field_is_settable_from_inside_its_own_class() -> void {
    let c = _ctx()
    let u = c.universe
    let decl = _class(c, "Point")
    _field(decl, "x", u.t_int, VIS_PUBLIC, true)
    c.current_class = decl
    let target = Node.member(_binding(c, "p", decl.ty), "x")
    stmt(c, Node.assign(target, Node.int_lit(1)))
    assert_eq(c.bag.items.len(), 0)
}

fn test_a_store_into_a_private_field_is_reported() -> void {
    let c = _ctx()
    let u = c.universe
    let decl = _class(c, "Box")
    _field(decl, "secret", u.t_int, VIS_PRIVATE)
    let target = Node.member(_binding(c, "b", decl.ty), "secret")
    stmt(c, Node.assign(target, Node.int_lit(1)))
    assert_eq(_code(c, 0), E0701_PRIVATE_ACCESS)
    assert_eq(_message(c, 0), "`secret` is private to `Box`")
}

fn test_a_store_into_a_method_is_reported() -> void {
    let c = _ctx()
    let u = c.universe
    let decl = _class(c, "Point")
    decl.members.push(MemberInfo("area", fn_of(u, [], u.t_int)))
    let target = Node.member(_binding(c, "p", decl.ty), "area")
    stmt(c, Node.assign(target, Node.int_lit(1)))
    assert_eq(_code(c, 0), E0702_UNDECLARED_FIELD)
    assert_eq(_message(c, 0), "`area` is a method of `Point` and cannot be assigned")
}

fn test_a_store_into_a_field_that_does_not_exist_is_reported() -> void {
    let c = _ctx()
    let decl = _class(c, "Point")
    _field(decl, "width", c.universe.t_int)
    let target = Node.member(_binding(c, "p", decl.ty), "widht")
    stmt(c, Node.assign(target, Node.int_lit(1)))
    assert_eq(_code(c, 0), E0702_UNDECLARED_FIELD)
    assert_eq(_message(c, 0), "`Point` has no field `widht`")
    assert_eq(_help(c, 0), "declare it in the class body; fields are not created by assignment")
}

fn test_a_setter_accepts_a_store() -> void {
    let c = _ctx()
    let u = c.universe
    let decl = _class(c, "Temp")
    decl.members.push(MemberInfo("celsius", u.t_float, VIS_PUBLIC, false, false, true))
    let target = Node.member(_binding(c, "t", decl.ty), "celsius")
    stmt(c, Node.assign(target, Node.float_lit(1.0)))
    assert_eq(c.bag.items.len(), 0)
    assert_eq(render(c.type_of(target)), "float")
}

fn test_a_store_through_an_any_receiver_is_left_to_the_run_time() -> void {
    let c = _ctx()
    let target = Node.member(_binding(c, "thing", c.universe.t_any), "whatever")
    stmt(c, Node.assign(target, Node.int_lit(1)))
    assert_eq(c.bag.items.len(), 0)
    assert_eq(render(c.type_of(target)), "any")
}

fn test_an_indexed_store_checks_the_index_and_the_element() -> void {
    let c = _ctx()
    let u = c.universe
    let container = _binding(c, "xs", list_of(u, u.t_int))
    let target = Node.index(container, Node.str_lit("a"))
    stmt(c, Node.assign(target, Node.str_lit("b")))
    assert_eq(_code(c, 0), E0400_TYPE_MISMATCH)
    assert_eq(_message(c, 0), "invalid index type: expected `int`, found `str`")
    assert_eq(_code(c, 1), E0400_TYPE_MISMATCH)
    assert_eq(_message(c, 1), "mismatched types in this assignment: expected `int`, found `str`")
}

fn test_an_indexed_store_into_something_unsubscriptable_is_reported() -> void {
    let c = _ctx()
    let target = Node.index(_binding(c, "n", c.universe.t_int), Node.int_lit(0))
    stmt(c, Node.assign(target, Node.int_lit(1)))
    assert_eq(_message(c, 0), "`int` does not support indexed assignment")
}

fn test_a_slice_target_checks_both_sides() -> void {
    let c = _ctx()
    let u = c.universe
    let object = _binding(c, "xs", list_of(u, u.t_int))
    let target = Node(NodeKind.Slice, {"object": object, "start": Node.str_lit("a")})
    let source = Node.int_lit(1)
    stmt(c, Node.assign(target, source))
    assert_eq(_message(c, 0), "a slice bound must be an int: expected `int`, found `str`")
    assert_eq(render(c.type_of(source)), "int")
}

fn test_a_destructuring_assignment_binds_through_the_pattern() -> void {
    let c = _ctx()
    let u = c.universe
    let first = Node.bind("a")
    let target = _seq(NodeKind.PatTuple, [first, Node.bind("b")])
    let source = _binding(c, "pair", tuple_of(u, [u.t_int, u.t_str]))
    stmt(c, Node.assign(target, source))
    assert_eq(render(c.type_of(first)), "int")
}

# ------------------------------------------------------------------
# Blocks and reachability
# ------------------------------------------------------------------
fn test_code_after_a_diverging_statement_is_warned_about_once() -> void {
    let c = _ctx()
    let body = Node.block(
        [
            Node.return_stmt(),
            Node.expr_stmt(Node.int_lit(1)),
            Node.expr_stmt(Node.int_lit(2)),
        ]
    )
    block(c, body)
    assert_eq(c.bag.items.len(), 1)
    assert_eq(_code(c, 0), W0102_UNREACHABLE_CODE)
    assert_eq(_message(c, 0), "unreachable code: the previous statement always transfers control")
}

fn test_a_declaration_after_a_diverging_statement_is_not_dead() -> void {
    let c = _ctx()
    # Declarations are hoisted by the resolver, so only real statements after a
    # diverging one are dead.
    let body = Node.block([Node.return_stmt(), Node.fn_decl("helper", [], Node.block([]))])
    block(c, body)
    assert_eq(c.bag.items.len(), 0)
}

fn test_a_program_is_checked_like_a_block() -> void {
    let c = _ctx()
    stmt(c, Node.program([Node.if_stmt(Node.int_lit(1), Node.block([]))]))
    assert_eq(_code(c, 0), E0401_CONDITION_NOT_BOOL)
}

fn test_a_match_statement_records_void_on_its_node() -> void {
    let c = _ctx()
    let arms = [match_arm(Node(NodeKind.PatWildcard), Node.int_lit(1))]
    let node = Node(NodeKind.Match, {"subject": Node.str_lit("x"), "arms": arms})
    stmt(c, node)
    assert_true(c.type_of(node) == c.universe.t_void)
}

fn test_break_and_continue_diverge_but_check_nothing() -> void {
    let c = _ctx()
    let body = Node.block([Node(NodeKind.Break), Node.expr_stmt(Node.int_lit(1))])
    block(c, body)
    assert_eq(c.bag.items.len(), 1)
    assert_eq(_code(c, 0), W0102_UNREACHABLE_CODE)
}

# ------------------------------------------------------------------
# if — narrowing
# ------------------------------------------------------------------
fn test_an_if_narrows_its_then_branch_and_restores_after() -> void {
    let c = _ctx()
    let u = c.universe
    let symbol = _symbol(c, "v", optional_of(u, u.t_int))
    let inside = _use(c, symbol)
    stmt(c, Node.if_stmt(_not_null(c, symbol), Node.block([Node.expr_stmt(inside)])))
    assert_eq(render(c.type_of(inside)), "int")
    assert_eq(render(c.symbol_type(symbol)), "int?")
}

fn test_the_else_edge_is_narrowed_too() -> void {
    let c = _ctx()
    let u = c.universe
    let symbol = _symbol(c, "v", optional_of(u, u.t_int))
    let inside = _use(c, symbol)
    let node = Node.if_stmt(
        _is_null(c, symbol),
        Node.block([]),
        Node.block([Node.expr_stmt(inside)])
    )
    stmt(c, node)
    assert_eq(render(c.type_of(inside)), "int")
}

fn test_the_false_edge_of_a_disjunction_narrows_both_operands() -> void {
    let c = _ctx()
    let u = c.universe
    let first = _symbol(c, "a", optional_of(u, u.t_int))
    let second = _symbol(c, "b", optional_of(u, u.t_str))
    let cond = Node.logical(OpKind.Or, _is_null(c, first), _is_null(c, second))
    let a_use = _use(c, first)
    let b_use = _use(c, second)
    let node = Node.if_stmt(
        cond,
        Node.block([]),
        Node.block([Node.expr_stmt(a_use), Node.expr_stmt(b_use)])
    )
    stmt(c, node)
    # `a or b` proves nothing when true, but `not (a or b)` proves both parts.
    assert_eq(render(c.type_of(a_use)), "int")
    assert_eq(render(c.type_of(b_use)), "str")
}

fn test_a_diverging_guard_narrows_the_rest_of_the_block() -> void {
    let c = _ctx()
    let u = c.universe
    let symbol = _symbol(c, "v", optional_of(u, u.t_int))
    let after = _use(c, symbol)
    let guard = Node.if_stmt(_is_null(c, symbol), Node.block([Node.return_stmt()]))
    block(c, Node.block([guard, Node.expr_stmt(after)]))
    assert_eq(render(c.type_of(after)), "int")
    # The block put the declaration back when it ended.
    assert_eq(render(c.symbol_type(symbol)), "int?")
}

fn test_a_non_diverging_guard_narrows_nothing_after_it() -> void {
    let c = _ctx()
    let u = c.universe
    let symbol = _symbol(c, "v", optional_of(u, u.t_int))
    let after = _use(c, symbol)
    let guard = Node.if_stmt(_is_null(c, symbol), Node.block([]))
    block(c, Node.block([guard, Node.expr_stmt(after)]))
    assert_eq(render(c.type_of(after)), "int?")
}

fn test_an_if_condition_must_be_a_bool() -> void {
    let c = _ctx()
    stmt(c, Node.if_stmt(Node.int_lit(1), Node.block([])))
    assert_eq(_code(c, 0), E0401_CONDITION_NOT_BOOL)
    assert_eq(_message(c, 0), "the condition of an `if` must be `bool`, found `int`")
}

fn test_an_is_test_narrows_to_the_class() -> void {
    let c = _ctx()
    let u = c.universe
    let decl = _class(c, "Dog")
    let symbol = _symbol(c, "pet", optional_of(u, decl.ty))
    let name = _use(c, _symbol(c, "Dog", decl.ty, SymbolKind.Class))
    let cond = Node.binary(OpKind.Is, _use(c, symbol), name)
    let inside = _use(c, symbol)
    stmt(c, Node.if_stmt(cond, Node.block([Node.expr_stmt(inside)])))
    assert_eq(render(c.type_of(inside)), "Dog")
}

# ------------------------------------------------------------------
# Loops
# ------------------------------------------------------------------
fn test_a_while_narrows_its_body_and_restores_after() -> void {
    let c = _ctx()
    let u = c.universe
    let symbol = _symbol(c, "v", optional_of(u, u.t_int))
    let inside = _use(c, symbol)
    let node = Node.while_stmt(_not_null(c, symbol), Node.block([Node.expr_stmt(inside)]))
    stmt(c, node)
    assert_eq(render(c.type_of(inside)), "int")
    assert_eq(render(c.symbol_type(symbol)), "int?")
}

fn test_a_while_condition_must_be_a_bool() -> void {
    let c = _ctx()
    stmt(c, Node.while_stmt(Node.int_lit(1), Node.block([])))
    assert_eq(_message(c, 0), "the condition of a `while` must be `bool`, found `int`")
}

fn test_a_bare_loop_has_no_condition_to_check() -> void {
    let c = _ctx()
    let inside = Node.expr_stmt(Node.int_lit(1))
    stmt(c, Node(NodeKind.Loop, {"body": Node.block([inside])}))
    assert_eq(c.bag.items.len(), 0)
    assert_eq(render(c.type_of(inside.require("expr"))), "int")
}

fn test_a_for_binds_its_pattern_to_the_element_type() -> void {
    let c = _ctx()
    let u = c.universe
    let pat = Node.bind("item")
    let over = _binding(c, "xs", list_of(u, u.t_str))
    stmt(c, Node.for_stmt(pat, over, Node.block([])))
    assert_eq(render(c.type_of(pat)), "str")
}

fn test_a_for_over_something_unloopable_is_reported() -> void {
    let c = _ctx()
    let over = _binding(c, "n", c.universe.t_int)
    stmt(c, Node.for_stmt(Node.bind("i"), over, Node.block([])))
    assert_eq(_code(c, 0), E0405_NOT_ITERABLE)
    assert_eq(_message(c, 0), "`int` is not iterable")
}

fn test_a_for_body_cannot_leak_a_narrowing_past_the_loop() -> void {
    let c = _ctx()
    let u = c.universe
    let symbol = _symbol(c, "v", optional_of(u, u.t_int))
    let guard = Node.if_stmt(_is_null(c, symbol), Node.block([Node.return_stmt()]))
    let over = _binding(c, "xs", list_of(u, u.t_str))
    let after = _use(c, symbol)
    # The loop body is a bare `if`, so nothing but `check_for`'s own restore
    # stands between its else-edge facts and the statement after the loop.
    let iteration = Node.for_stmt(Node.bind("i"), over, guard)
    block(c, Node.block([iteration, Node.expr_stmt(after)]))
    assert_eq(render(c.type_of(after)), "int?")
}

# ------------------------------------------------------------------
# return, throw, try, assert, defer
# ------------------------------------------------------------------
fn test_a_return_is_checked_against_the_enclosing_result() -> void {
    let c = _ctx()
    c.current_return_type = c.universe.t_int
    stmt(c, Node.return_stmt(Node.str_lit("a")))
    assert_eq(_code(c, 0), E0408_RETURN_TYPE_MISMATCH)
    assert_eq(_message(c, 0), "mismatched return type: expected `int`, found `str`")
}

fn test_a_bare_return_from_a_value_function_is_reported() -> void {
    let c = _ctx()
    c.current_return_type = c.universe.t_int
    stmt(c, Node.return_stmt())
    assert_eq(_code(c, 0), E0408_RETURN_TYPE_MISMATCH)
    assert_eq(_message(c, 0), "this function must return `int`")
    assert_eq(_help(c, 0), "add a value: `return <int>`")
}

fn test_a_bare_return_from_an_optional_function_is_fine() -> void {
    let c = _ctx()
    let u = c.universe
    c.current_return_type = optional_of(u, u.t_int)
    stmt(c, Node.return_stmt())
    assert_eq(c.bag.items.len(), 0)
}

fn test_a_return_outside_every_function_checks_only_the_value() -> void {
    let c = _ctx()
    let returned = Node.int_lit(1)
    stmt(c, Node.return_stmt(returned))
    assert_eq(c.bag.items.len(), 0)
    assert_eq(render(c.type_of(returned)), "int")
}

fn test_a_returned_int_literal_widens_into_a_float_result() -> void {
    let c = _ctx()
    let returned = Node.int_lit(1)
    c.current_return_type = c.universe.t_float
    stmt(c, Node.return_stmt(returned))
    assert_eq(c.bag.items.len(), 0)
    assert_true(returned.kind == NodeKind.FloatLit)
}

fn test_throwing_something_that_is_not_a_class_is_reported() -> void {
    let c = _ctx()
    stmt(c, Node(NodeKind.Throw, {"value": Node.str_lit("boom")}))
    assert_eq(_code(c, 0), E0400_TYPE_MISMATCH)
    assert_eq(_message(c, 0), "only an `Error` instance can be thrown, found `str`")
    assert_eq(_help(c, 0), "throw a class deriving from `Error`, e.g. `ValueError(\"...\")`")
}

fn test_throwing_a_class_instance_is_accepted() -> void {
    let c = _ctx()
    let decl = _class(c, "ValueError")
    stmt(c, Node(NodeKind.Throw, {"value": _binding(c, "e", decl.ty)}))
    assert_eq(c.bag.items.len(), 0)
}

fn test_a_bare_throw_checks_nothing() -> void {
    let c = _ctx()
    stmt(c, Node(NodeKind.Throw, {"value": null}))
    assert_eq(c.bag.items.len(), 0)
}

fn test_an_annotated_catch_binds_the_type_it_names() -> void {
    let c = _annotated()
    let decl = _class(c, "IoError")
    let handler = catch_clause("e", Node.block([]), [TypeExpr.named("IoError")])
    let symbol = _symbol(c, "e", null)
    c.resolution.bind(handler, symbol)
    stmt(c, Node(NodeKind.Try, {"body": Node.block([]), "catches": [handler]}))
    assert_true(c.symbol_type(symbol) == decl.ty)
}

fn test_a_multi_type_catch_binds_their_union() -> void {
    let c = _annotated()
    let io = _class(c, "IoError")
    let os = _class(c, "OsError")
    let types = [TypeExpr.named("IoError"), TypeExpr.named("OsError")]
    let handler = catch_clause("e", Node.block([]), types)
    let symbol = _symbol(c, "e", null)
    c.resolution.bind(handler, symbol)
    stmt(c, Node(NodeKind.Try, {"body": Node.block([]), "catches": [handler]}))
    assert_eq(render(c.symbol_type(symbol)), "IoError | OsError")
    assert_true(io.ty is not null and os.ty is not null)
}

fn test_an_unannotated_catch_binds_error() -> void {
    let c = _annotated()
    let decl = _class(c, "Error")
    let handler = catch_clause("e", Node.block([]))
    let symbol = _symbol(c, "e", null)
    c.resolution.bind(handler, symbol)
    stmt(c, Node(NodeKind.Try, {"body": Node.block([]), "catches": [handler]}))
    assert_true(c.symbol_type(symbol) == decl.ty)
}

fn test_an_unannotated_catch_is_any_when_error_is_not_visible() -> void {
    let c = _ctx()
    let handler = catch_clause("e", Node.block([]))
    let symbol = _symbol(c, "e", null)
    c.resolution.bind(handler, symbol)
    stmt(c, Node(NodeKind.Try, {"body": Node.block([]), "catches": [handler]}))
    assert_eq(render(c.symbol_type(symbol)), "any")
}

fn test_every_part_of_a_try_is_checked() -> void {
    let c = _ctx()
    let node = Node(
        NodeKind.Try,
        {
            "body": Node.block([Node.if_stmt(Node.int_lit(1), Node.block([]))]),
            "catches": [catch_clause("e", Node.block([Node.expr_stmt(Node.int_lit(2))]))],
            "finallyBlock": Node.block([Node.if_stmt(Node.str_lit("a"), Node.block([]))]),
        }
    )
    stmt(c, node)
    assert_eq(c.bag.items.len(), 2)
    assert_eq(_message(c, 0), "the condition of an `if` must be `bool`, found `int`")
    assert_eq(_message(c, 1), "the condition of an `if` must be `bool`, found `str`")
}

fn test_an_assert_checks_its_condition_and_its_message() -> void {
    let c = _ctx()
    stmt(c, Node(NodeKind.Assert, {"cond": Node.int_lit(1), "message": Node.int_lit(2)}))
    assert_eq(_code(c, 0), E0401_CONDITION_NOT_BOOL)
    assert_eq(_message(c, 0), "the condition of an `assert` must be `bool`, found `int`")
    assert_eq(_code(c, 1), E0400_TYPE_MISMATCH)
    assert_eq(_message(c, 1), "an assertion message must be a `str`: expected `str`, found `int`")
}

fn test_a_defer_body_is_checked_like_any_other_statement() -> void {
    let c = _ctx()
    stmt(
        c,
        Node(NodeKind.Defer, {"body": Node.block([Node.if_stmt(Node.int_lit(1), Node.block([]))])})
    )
    assert_eq(_code(c, 0), E0401_CONDITION_NOT_BOOL)
}

# ------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------
fn test_is_init_method_names_only_the_constructor() -> void {
    assert_true(is_init_method(Node.fn_decl("init", [], Node.block([]))))
    assert_false(is_init_method(Node.fn_decl("area", [], Node.block([]))))
    assert_false(is_init_method(null))
    # A lambda has no name at all, so it is never one either.
    assert_false(is_init_method(Node(NodeKind.Lambda, {"name": "init"})))
}

fn test_an_unannotated_result_is_any_and_a_constructor_is_void() -> void {
    let c = _ctx()
    let plain = Node.fn_decl("area", [], Node.block([]))
    assert_true(declared_return_type(c, plain, false) == c.universe.t_any)
    let ctor = Node.fn_decl("init", [param("self")], Node.block([]))
    assert_true(declared_return_type(c, ctor, true) == c.universe.t_void)
    # Only as a method: a free function called `init` returns `any`.
    assert_true(declared_return_type(c, ctor, false) == c.universe.t_any)
}

fn test_a_function_type_is_its_parameters_and_its_result() -> void {
    let c = _annotated()
    let params = [param("a", TypeExpr.named("int")), param("b", TypeExpr.named("str"))]
    let node = Node.fn_decl("pair", params, Node.block([]), TypeExpr.named("bool"))
    assert_eq(render(function_type(c, node, false)), "fn(int, str) -> bool")
}

fn test_a_method_signature_drops_its_receiver() -> void {
    let c = _annotated()
    let params = [param("self"), param("n", TypeExpr.named("int"))]
    let node = Node.fn_decl("scale", params, Node.block([]), TypeExpr.named("int"))
    assert_eq(render(function_type(c, node, true)), "fn(int) -> int")
    # As a free function the receiver is just another parameter.
    assert_eq(render(function_type(c, node, false)), "fn(any, int) -> int")
}

fn test_a_signature_records_the_variadic_and_kwrest_flags() -> void {
    let c = _annotated()
    let rest = param("xs", TypeExpr.named("int"), null, true)
    let opts = param("kw", TypeExpr.named("str"), null, false, true)
    let node = Node.fn_decl("collect", [rest, opts], Node.block([]))
    let signature = function_type(c, node, false)
    assert_true(signature.is_variadic())
    assert_true(signature.has_kwrest())
}

fn test_a_generic_parameter_is_in_scope_for_the_signature_only() -> void {
    let c = _ctx()
    push_generics(c, [{"name": "T", "bound": null, "span": Span.none()}])
    assert_eq(render(c.scoped_type_name("T")), "T")
}

fn test_a_parameter_binding_is_the_declared_type() -> void {
    let c = _annotated()
    let p = param("n", TypeExpr.named("int"))
    let symbol = _symbol(c, "n", null)
    c.resolution.bind(p, symbol)
    function(c, Node.fn_decl("f", [p], Node.block([])), null, false)
    assert_eq(render(c.symbol_type(symbol)), "int")
}

fn test_a_variadic_parameter_binds_a_list_and_a_kwrest_binds_a_dict() -> void {
    let c = _annotated()
    let rest = param("xs", TypeExpr.named("int"), null, true)
    let opts = param("kw", TypeExpr.named("str"), null, false, true)
    let rest_symbol = _symbol(c, "xs", null)
    let opts_symbol = _symbol(c, "kw", null)
    c.resolution.bind(rest, rest_symbol)
    c.resolution.bind(opts, opts_symbol)
    function(c, Node.fn_decl("f", [rest, opts], Node.block([])), null, false)
    assert_eq(render(c.symbol_type(rest_symbol)), "list[int]")
    assert_eq(render(c.symbol_type(opts_symbol)), "dict[str, str]")
}

fn test_a_receiver_binds_the_class_being_checked() -> void {
    let c = _ctx()
    let decl = _class(c, "Point")
    let receiver = param("self")
    let symbol = _symbol(c, "self", null)
    c.resolution.bind(receiver, symbol)
    function(c, Node.fn_decl("area", [receiver], Node.block([])), decl, true)
    assert_eq(render(c.symbol_type(symbol)), "Point")
}

fn test_a_default_value_is_checked_against_its_parameter() -> void {
    let c = _annotated()
    let p = param("n", TypeExpr.named("int"), Node.str_lit("a"))
    function(c, Node.fn_decl("f", [p], Node.block([])), null, false)
    assert_eq(_code(c, 0), E0400_TYPE_MISMATCH)
    assert_eq(_message(c, 0), "mismatched type for a default value: expected `int`, found `str`")
}

fn test_strict_wants_every_parameter_annotated() -> void {
    let c = _annotated()
    c.strict = true
    function(c, Node.fn_decl("f", [param("n")], Node.block([])), null, false)
    assert_eq(_code(c, 0), E0400_TYPE_MISMATCH)
    assert_eq(_message(c, 0), "parameter `n` has no type annotation")
    assert_eq(_help(c, 0), "--strict requires every parameter to be annotated")
}

fn test_the_enclosing_result_type_is_restored_afterwards() -> void {
    let c = _annotated()
    c.current_return_type = c.universe.t_str
    let body = Node.block([Node.return_stmt(Node.int_lit(1))])
    function(c, Node.fn_decl("f", [], body, TypeExpr.named("int")), null, false)
    assert_eq(c.bag.items.len(), 0)
    assert_true(c.current_return_type == c.universe.t_str)
}

fn test_the_enclosing_class_is_restored_afterwards() -> void {
    let c = _ctx()
    let outer = _class(c, "Outer")
    let inner = _class(c, "Inner")
    c.current_class = outer
    function(c, Node.fn_decl("m", [param("self")], Node.block([])), inner, true)
    assert_true(c.current_class == outer)
}

fn test_a_generic_bound_is_resolved_with_the_parameter_in_scope() -> void {
    let c = _ctx()
    var seen: list[str] = []
    fn record(inner: Ctx, written: TypeExpr?) -> Type? {
        if written is not null { seen.push(written.name ?? "?") }
        return null
    }
    c.resolve_ast_type = record
    push_generics(c, [{"name": "T", "bound": TypeExpr.named("Comparable"), "span": Span.none()}])
    # `[T: Comparable[T]]` has to see itself, so the parameters go in first and
    # the bounds are resolved afterwards.
    assert_eq(seen, ["Comparable"])
    assert_eq(render(c.scoped_type_name("T")), "T")
}

fn test_an_annotated_function_must_return_on_every_path() -> void {
    let c = _annotated()
    let node = Node.fn_decl("total", [], Node.block([]), TypeExpr.named("int"))
    function(c, node, null, false)
    assert_eq(_code(c, 0), E0601_MISSING_RETURN)
    assert_eq(_message(c, 0), "`total` must return `int` on every path")
    assert_eq(_help(c, 0), "add a `return` at the end, or make the last branch an `else`")
    assert_eq(c.bag.items[0].labels[0][1], "declared to return `int`")
}

fn test_a_body_that_always_diverges_satisfies_the_result() -> void {
    let c = _annotated()
    let body = Node.block([Node.return_stmt(Node.int_lit(1))])
    function(c, Node.fn_decl("total", [], body, TypeExpr.named("int")), null, false)
    assert_eq(c.bag.items.len(), 0)
}

fn test_a_generator_returns_through_yield_and_is_exempt() -> void {
    let c = _annotated()
    let node = Node.fn_decl("stream", [], Node.block([]), TypeExpr.named("int"))
    node.set("isGenerator", true)
    function(c, node, null, false)
    assert_eq(c.bag.items.len(), 0)
}

fn test_an_unannotated_function_may_fall_off_its_end() -> void {
    let c = _annotated()
    function(c, Node.fn_decl("f", [], Node.block([])), null, false)
    assert_eq(c.bag.items.len(), 0)
}

fn test_an_optional_result_may_fall_off_its_end() -> void {
    let c = _annotated()
    let written = TypeExpr.optional(TypeExpr.named("int"))
    c.resolve_ast_type = _optional_types
    function(c, Node.fn_decl("f", [], Node.block([]), written), null, false)
    assert_eq(c.bag.items.len(), 0)
}

fn _optional_types(c: Ctx, written: TypeExpr?) -> Type? {
    if written is null { return null }
    if written.kind == TypeKind.Optional { return optional_of(c.universe, c.universe.t_int) }
    return _named_types(c, written)
}

fn test_a_lambda_body_is_its_return_value() -> void {
    let c = _annotated()
    let body = Node.str_lit("a")
    let node = Node(
        NodeKind.Lambda,
        {"name": null, "params": [], "returnType": TypeExpr.named("int"), "body": body}
    )
    function(c, node, null, false)
    assert_eq(_code(c, 0), E0408_RETURN_TYPE_MISMATCH)
    assert_eq(_message(c, 0), "mismatched return type: expected `int`, found `str`")
}

fn test_an_unannotated_lambda_body_is_checked_but_not_measured() -> void {
    let c = _ctx()
    let node = Node(NodeKind.Lambda, {"name": null, "params": [], "body": Node.str_lit("a")})
    function(c, node, null, false)
    assert_eq(c.bag.items.len(), 0)
}

# ------------------------------------------------------------------
# The seam
# ------------------------------------------------------------------
fn test_install_lets_an_expression_reach_a_match() -> void {
    let c = _ctx()
    let arms = [match_arm(Node.bind("other"), Node.str_lit("x"))]
    let node = Node(NodeKind.MatchExpr, {"subject": Node.int_lit(1), "arms": arms})
    # Without `install` the default answers `any` and checks nothing.
    assert_eq(render(expr(c, node)), "str")
}

fn test_install_lets_a_comprehension_bind_its_loop_variable() -> void {
    let c = _ctx()
    let u = c.universe
    let pat = Node.bind("item")
    let symbol = _symbol(c, "item", null)
    c.resolution.bind(pat, symbol)
    let over = _binding(c, "xs", list_of(u, u.t_str))
    let element = _use(c, symbol)
    let node = Node(
        NodeKind.Comprehension,
        {
            "compKind": CompKind.List,
            "element": element,
            "clauses": [comp_clause(pat, over)],
        }
    )
    assert_eq(render(expr(c, node)), "list[str]")
}

fn test_install_types_and_checks_a_lambda_in_expression_position() -> void {
    let c = _annotated()
    let node = Node(
        NodeKind.Lambda,
        {
            "name": null,
            "params": [param("n", TypeExpr.named("int"))],
            "returnType": TypeExpr.named("str"),
            "body": Node.int_lit(1),
        }
    )
    assert_eq(render(expr(c, node)), "fn(int) -> str")
    # And the body went through `function`, not merely the signature.
    assert_eq(_code(c, 0), E0408_RETURN_TYPE_MISMATCH)
}

fn test_install_lets_an_expression_reach_a_statement() -> void {
    let c = _ctx()
    let inner = Node.if_stmt(Node.int_lit(1), Node.block([]))
    expr(c, inner)
    assert_eq(_code(c, 0), E0401_CONDITION_NOT_BOOL)
}

fn test_a_declaration_inside_a_block_reaches_the_decl_callback() -> void {
    let c = _ctx()
    var seen = 0
    fn record(inner: Ctx, node: Node?) -> void {
        seen += 1
    }
    c.check_decl = record
    block(c, Node.block([Node.fn_decl("helper", [], Node.block([]))]))
    assert_eq(seen, 1)
}
