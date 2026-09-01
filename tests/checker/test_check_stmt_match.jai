#: `jaithon.compile.check.stmt` — match: the type it answers, coverage, narrowing.
from std.test import assert_eq, assert_true
from jaithon.ast import DeclKind, Node, NodeKind, OpKind, Span, match_arm
from jaithon.compile.check.ctx import Ctx
from jaithon.compile.check.expr import E0401_CONDITION_NOT_BOOL
from jaithon.compile.check.kinds import TyKind
from jaithon.compile.check.render import render
from jaithon.compile.check.stmt import E0500_UNREACHABLE_ARM, E0501_NON_EXHAUSTIVE_MATCH, block, install, match_expr, pattern
from jaithon.compile.check.ty import FieldInfo, Type, TypeDecl
from jaithon.compile.check.union import optional_of
from jaithon.compile.check.universe import Universe, fn_of, list_of, named
from jaithon.compile.symbol import Symbol, SymbolKind

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------
fn _ctx() -> Ctx {
    let c = Ctx(Universe())
    install(c)
    return c
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

fn _code(c: Ctx, at: int) -> str {
    return c.bag.items[at].code
}

fn _message(c: Ctx, at: int) -> str {
    return c.bag.items[at].message
}

fn _help(c: Ctx, at: int) -> str? {
    return c.bag.items[at].help
}

fn _is_null(c: Ctx, symbol: Symbol) -> Node {
    return Node.binary(OpKind.Eq, _use(c, symbol), Node.null_lit())
}

fn _seq(kind: NodeKind, elems: list[Node], rest: int = -1) -> Node {
    return Node(kind, {"elems": elems, "restIndex": rest})
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
    assert_eq(render(c.symbol_type(symbol)), "str")
}
