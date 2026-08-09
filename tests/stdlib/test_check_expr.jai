#: `jaithon.compile.check.expr` — the type of every expression form.
#:
#: The cases worth writing down are the ones where the C does something a
#: re-derivation would not: an unannotated `[1, 2.0]` converts its int element in
#: place while `{1, 2.0}` deliberately does not, a constant subscript on a tuple
#: answers with the member it selects rather than the join, `a?.m(x)` takes the
#: optional off the member load and puts it back on the call, an argument that
#: found no parameter is still checked exactly once, and a `??` on a
#: non-optional is a warning rather than an error. Each of those is a paragraph
#: of `src/sema/check_expr.c` that reads like an accident and is not.
#:
#: The diagnostics are asserted by code **and by message text**, because both
#: are a contract with the C front end: Phase 4 turns `tests/errors` into
#: goldens and a reworded message fails there.
from std.test import assert_eq, assert_false, assert_true
from jaithon.ast import CompKind, DeclKind, Node, NodeKind, OpKind, Span, TypeExpr
from jaithon.ast import arg, comp_clause, param
from jaithon.compile.check.expr import Ctx, E0400_TYPE_MISMATCH, E0401_CONDITION_NOT_BOOL
from jaithon.compile.check.expr import E0403_NOT_CALLABLE, E0404_NOT_INDEXABLE
from jaithon.compile.check.expr import E0405_NOT_ITERABLE, E0406_BAD_OPERAND_TYPES
from jaithon.compile.check.expr import E0410_UNKNOWN_MEMBER, E0602_BAD_CALL_ARITY
from jaithon.compile.check.expr import E0603_UNKNOWN_KEYWORD_ARG, E0604_DUPLICATE_ARGUMENT
from jaithon.compile.check.expr import E0605_POSITIONAL_AFTER_KEYWORD, E0701_PRIVATE_ACCESS
from jaithon.compile.check.expr import E0710_ABSTRACT_INSTANTIATION, VIS_PRIVATE, VIS_PROTECTED
from jaithon.compile.check.expr import VIS_PUBLIC, W0102_UNREACHABLE_CODE
from jaithon.compile.check.expr import condition, expr, find_init_decl, is_arithmetic
from jaithon.compile.check.expr import iterable_element, member, self_skip_of, slice
from jaithon.compile.check.expr import subject_of, value, visibility_word, visible_from
from jaithon.compile.check.expr import widens_operands
from jaithon.compile.check.kinds import TyKind
from jaithon.compile.check.render import render
from jaithon.compile.check.ty import FieldInfo, MemberInfo, Type, TypeDecl
from jaithon.compile.check.union import optional_of
from jaithon.compile.check.universe import Universe, dict_of, fn_of, list_of, named
from jaithon.compile.resolve import Symbol, SymbolKind

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------
fn _ctx() -> Ctx {
    return Ctx(Universe())
}

# A binding of `ty`, and the symbol behind it, so that two use sites can share
# one symbol — which is what narrowing needs.
fn _symbol(c: Ctx, name: str, ty: Type?, kind: SymbolKind = SymbolKind.Local) -> Symbol {
    let symbol = Symbol(kind, name, Span(0, 0, 1), DeclKind.Let)
    c.set_symbol_type(symbol, ty)
    return symbol
}

# A fresh use of `symbol`, bound the way the resolver would have bound it.
fn _use(c: Ctx, symbol: Symbol) -> Node {
    let node = Node.ident(symbol.name)
    c.resolution.bind(node, symbol)
    return node
}

# One binding used once: the common case.
fn _binding(c: Ctx, name: str, ty: Type?) -> Node {
    return _use(c, _symbol(c, name, ty))
}

# A registered class, with its type interned and attached.
fn _class(c: Ctx, name: str) -> TypeDecl {
    let decl = TypeDecl(name, Span(0, 0, 1))
    decl.ty = named(c.universe, TyKind.Class, name, decl)
    c.register_decl(decl)
    return decl
}

# The name of a class, as an expression: `Point` in `Point(1, 2)`.
fn _class_name(c: Ctx, decl: TypeDecl) -> Node {
    return _use(c, _symbol(c, decl.name, decl.ty, SymbolKind.Class))
}

# A top-level function, its symbol and a call site for it. The symbol's `decl`
# back-link is what makes the call bind against the real parameter list.
fn _function(c: Ctx, name: str, params: list[dict[str, any]], signature: Type) -> Symbol {
    let decl = Node.fn_decl(name, params, Node.block([]))
    let symbol = _symbol(c, name, signature, SymbolKind.Global)
    symbol.decl = decl
    c.resolution.bind(decl, symbol)
    return symbol
}

fn _call(c: Ctx, symbol: Symbol, args: list[dict[str, any]]) -> Node {
    return Node(NodeKind.Call, {"callee": _use(c, symbol), "args": args})
}

# A `resolve_ast_type` that knows the four scalars, so the argument tests can
# see a declared parameter type at all. `checker.jai` supplies the real one.
fn _scalar_types(c: Ctx, written: TypeExpr?) -> Type? {
    if written is null { return null }
    let name = written.name
    if name == "int" { return c.universe.t_int }
    if name == "float" { return c.universe.t_float }
    if name == "str" { return c.universe.t_str }
    if name == "bool" { return c.universe.t_bool }
    return null
}

fn _code(c: Ctx, at: int) -> str {
    return c.bag.items[at].code
}

fn _message(c: Ctx, at: int) -> str {
    return c.bag.items[at].message
}

# ------------------------------------------------------------------
# Literals, identifiers and the type side table
# ------------------------------------------------------------------
fn test_literals_answer_their_own_type_and_record_it() -> void {
    let c = _ctx()
    let u = c.universe
    let node = Node.int_lit(3)
    assert_true(expr(c, node) == u.t_int)
    # The answer is recorded, which is the C's `node->type`.
    assert_true(c.type_of(node) == u.t_int)
    assert_true(expr(c, Node.float_lit(1.5)) == u.t_float)
    assert_true(expr(c, Node.str_lit("a")) == u.t_str)
    assert_true(expr(c, Node.bool_lit(true)) == u.t_bool)
    assert_true(expr(c, Node.null_lit()) == u.t_null)
    assert_true(expr(c, null) == u.t_any)
    assert_eq(c.bag.items.len(), 0)
}

fn test_the_type_table_is_keyed_by_identity_not_by_value() -> void {
    let c = _ctx()
    let one = Node.int_lit(1)
    let other = Node.int_lit(1)
    c.set_type(one, c.universe.t_int)
    assert_true(c.type_of(one) == c.universe.t_int)
    # Two structurally equal nodes are two expressions, so `x` in two scopes
    # cannot share a type any more than it can share a symbol.
    assert_true(c.type_of(other) is null)
}

fn test_an_unresolved_name_is_any_and_reports_nothing() -> void {
    let c = _ctx()
    # The resolver already reported the undefined name; the checker does not
    # pile on.
    assert_true(expr(c, Node.ident("nowhere")) == c.universe.t_any)
    assert_eq(c.bag.items.len(), 0)
}

fn test_a_bound_name_answers_its_symbol_type() -> void {
    let c = _ctx()
    let u = c.universe
    assert_true(expr(c, _binding(c, "n", u.t_int)) == u.t_int)
}

fn test_a_class_name_picks_its_type_up_from_the_registry() -> void {
    let c = _ctx()
    let decl = _class(c, "Point")
    # The symbol has no type yet, which is the state a class referenced before
    # its layout pass is in.
    let symbol = Symbol(SymbolKind.Class, "Point", Span.none(), DeclKind.Let)
    let node = _use(c, symbol)
    assert_eq(render(expr(c, node)), "Point")
    # And it is remembered on the symbol, so the second use costs no lookup.
    assert_true(c.symbol_type(symbol) == decl.ty)
}

fn test_a_folded_const_is_inlined_at_its_use_site() -> void {
    let c = _ctx()
    let u = c.universe
    c.consts["K"] = Node.int_lit(8)
    let symbol = Symbol(SymbolKind.Global, "K", Span.none(), DeclKind.Const)
    let node = _use(c, symbol)
    assert_true(expr(c, node) == u.t_int)
    # Spec §3: the use site becomes the literal.
    assert_true(node.kind == NodeKind.IntLit)
    assert_eq(node.number("intLit"), 8)
}

fn test_const_inlining_is_off_when_folding_is() -> void {
    let c = _ctx()
    c.fold_constants = false
    c.consts["K"] = Node.int_lit(8)
    let node = _use(c, Symbol(SymbolKind.Global, "K", Span.none(), DeclKind.Const))
    expr(c, node)
    assert_true(node.kind == NodeKind.Ident)
}

# ------------------------------------------------------------------
# Container literals
# ------------------------------------------------------------------
fn test_an_unannotated_list_joins_its_elements() -> void {
    let c = _ctx()
    let node = Node(NodeKind.ListLit, {"items": [Node.int_lit(1), Node.str_lit("a")]})
    assert_eq(render(expr(c, node)), "list[int | str]")
}

fn test_an_empty_list_is_list_any() -> void {
    let c = _ctx()
    assert_eq(render(expr(c, Node(NodeKind.ListLit, {"items": []}))), "list[any]")
}

fn test_a_list_literal_fuses_an_int_with_a_float() -> void {
    let c = _ctx()
    let one = Node.int_lit(1)
    let node = Node(NodeKind.ListLit, {"items": [one, Node.float_lit(2.0)]})
    # Spec §2.2: the int element is converted in place, so the `list[float]`
    # the join reports is one the elements agree with.
    assert_eq(render(expr(c, node)), "list[float]")
    assert_true(one.kind == NodeKind.FloatLit)
}

fn test_a_set_literal_is_deliberately_not_fused() -> void {
    let c = _ctx()
    let one = Node.int_lit(1)
    let node = Node(NodeKind.SetLit, {"items": [one, Node.float_lit(2.0)]})
    # A number in a hash position decides which elements are the same element,
    # so an inferred conversion could silently shrink the set.
    assert_eq(render(expr(c, node)), "set[int | float]")
    assert_true(one.kind == NodeKind.IntLit)
}

fn test_an_annotated_list_checks_each_element_against_the_annotation() -> void {
    let c = _ctx()
    let u = c.universe
    let bad = Node.str_lit("a")
    let node = Node(NodeKind.ListLit, {"items": [Node.int_lit(1), bad]})
    c.apply_context(c, node, list_of(u, u.t_int))
    assert_true(expr(c, node) == list_of(u, u.t_int))
    # The offending element is reported, not the whole container failing
    # against its own annotation.
    assert_eq(_code(c, 0), E0400_TYPE_MISMATCH)
    assert_eq(_message(c, 0), "mismatched element type: expected `int`, found `str`")
}

fn test_an_annotated_empty_list_takes_the_annotation() -> void {
    let c = _ctx()
    let u = c.universe
    let node = Node(NodeKind.ListLit, {"items": []})
    c.apply_context(c, node, list_of(u, u.t_str))
    assert_eq(render(expr(c, node)), "list[str]")
}

fn test_a_container_context_of_the_wrong_shape_is_ignored_and_dropped() -> void {
    let c = _ctx()
    let u = c.universe
    let node = Node(NodeKind.ListLit, {"items": []})
    # A hint is taken only when it is the right kind and arity for the literal
    # under it; anything else is consumed and discarded, so it cannot leak into
    # the inference.
    c.set_type(node, dict_of(u, u.t_str, u.t_int))
    assert_eq(render(expr(c, node)), "list[any]")
    assert_eq(c.bag.items.len(), 0)
}

fn test_a_tuple_literal_keeps_its_members_apart() -> void {
    let c = _ctx()
    let node = Node(NodeKind.TupleLit, {"items": [Node.str_lit("a"), Node.int_lit(1)]})
    assert_eq(render(expr(c, node)), "(str, int)")
}

fn test_a_dict_literal_fuses_its_values_and_leaves_its_keys() -> void {
    let c = _ctx()
    let key = Node.int_lit(1)
    let held = Node.int_lit(2)
    let node = Node(
        NodeKind.DictLit,
        {"keys": [key, Node.float_lit(1.5)], "values": [held, Node.float_lit(2.5)]}
    )
    assert_eq(render(expr(c, node)), "dict[int | float, float]")
    assert_true(key.kind == NodeKind.IntLit)
    assert_true(held.kind == NodeKind.FloatLit)
}

# ------------------------------------------------------------------
# Operators
# ------------------------------------------------------------------
fn test_arithmetic_widens_an_int_operand_against_a_float_one() -> void {
    let c = _ctx()
    let u = c.universe
    let n = _binding(c, "n", u.t_int)
    let node = Node.binary(OpKind.Mul, n, Node.float_lit(1.5))
    assert_true(expr(c, node) == u.t_float)
    # The conversion goes into the tree, so the VM performs one conversion
    # rather than a type test on every evaluation.
    assert_true(n.kind == NodeKind.Cast)
    assert_true(c.is_widen_cast(n))
}

fn test_the_wrapping_and_bitwise_operators_do_not_widen() -> void {
    let c = _ctx()
    let u = c.universe
    assert_false(widens_operands(OpKind.AddWrap))
    assert_true(is_arithmetic(OpKind.AddWrap))
    assert_false(widens_operands(OpKind.Band))

    let node = Node.binary(OpKind.AddWrap, _binding(c, "n", u.t_int), Node.float_lit(1.5))
    assert_true(expr(c, node) == u.t_any)
    assert_eq(_code(c, 0), E0406_BAD_OPERAND_TYPES)
    assert_eq(_message(c, 0), "operator `+%` does not apply to `int` and `float`")
}

fn test_identity_never_constrains_its_operands() -> void {
    let c = _ctx()
    let u = c.universe
    let node = Node.binary(OpKind.Is, _binding(c, "s", u.t_str), _binding(c, "n", u.t_int))
    assert_true(expr(c, node) == u.t_bool)
    assert_eq(c.bag.items.len(), 0)
}

fn test_any_makes_an_operator_dynamic_and_a_comparison_still_bool() -> void {
    let c = _ctx()
    let u = c.universe
    let dynamic = _symbol(c, "x", u.t_any)
    assert_true(expr(c, Node.binary(OpKind.Add, _use(c, dynamic), Node.int_lit(1))) == u.t_any)
    assert_true(expr(c, Node.binary(OpKind.Lt, _use(c, dynamic), Node.int_lit(1))) == u.t_bool)
    assert_eq(c.bag.items.len(), 0)
}

fn test_a_bad_operator_names_both_operands_and_offers_the_dunder() -> void {
    let c = _ctx()
    let u = c.universe
    let decl = _class(c, "Money")
    let node = Node.binary(OpKind.Add, _binding(c, "m", decl.ty), Node.int_lit(1))
    assert_true(expr(c, node) == u.t_any)
    assert_eq(_code(c, 0), E0406_BAD_OPERAND_TYPES)
    assert_eq(_message(c, 0), "operator `+` does not apply to `Money` and `int`")
    assert_eq(c.bag.items[0].help, "define the matching dunder method on `Money`")
}

fn test_a_unary_operator_reports_its_one_operand() -> void {
    let c = _ctx()
    let u = c.universe
    assert_true(expr(c, Node.unary(OpKind.Neg, Node.float_lit(1.5))) == u.t_float)
    let node = Node.unary(OpKind.Bnot, _binding(c, "f", u.t_float))
    assert_true(expr(c, node) == u.t_any)
    assert_eq(_message(c, 0), "operator `~` does not apply to `float`")
}

fn test_not_takes_a_condition_rather_than_an_operand() -> void {
    let c = _ctx()
    let u = c.universe
    let node = Node.unary(OpKind.Not, _binding(c, "n", u.t_int))
    # `not` is a boolean position, so the failure is E0401 and not E0406.
    assert_true(expr(c, node) == u.t_bool)
    assert_eq(_code(c, 0), E0401_CONDITION_NOT_BOOL)
    assert_eq(_message(c, 0), "the condition of `not` must be `bool`, found `int`")
}

fn test_a_comparison_chain_is_bool_and_checks_every_operand_once() -> void {
    let c = _ctx()
    let u = c.universe
    let bad = Node.binary(OpKind.Add, _binding(c, "s", u.t_str), Node.int_lit(1))
    let node = Node(
        NodeKind.CompareChain,
        {
            "operands": [Node.int_lit(1), Node.int_lit(2), bad],
            "ops": [OpKind.Lt, OpKind.Lt],
        }
    )
    assert_true(expr(c, node) == u.t_bool)
    # The operand's own error, and nothing about the links: `binary_result`
    # answers `bool` for every comparison whatever the operands are — whether
    # they compare at all is `__eq__`/`__lt__`, which is a member lookup this
    # table does not do. A chain holds only comparisons, so the E0406 arm of
    # `_check_chain` is unreachable. Verified against the C front end, which
    # accepts `1 < 2 < "a"` for the same reason.
    assert_eq(c.bag.items.len(), 1)
    assert_eq(_code(c, 0), E0406_BAD_OPERAND_TYPES)
    assert_eq(_message(c, 0), "operator `+` does not apply to `str` and `int`")
}

fn test_a_ternary_joins_its_arms_and_narrows_each_edge() -> void {
    let c = _ctx()
    let u = c.universe
    let maybe = _symbol(c, "x", optional_of(u, u.t_int))
    let cond = Node.binary(OpKind.Ne, _use(c, maybe), Node.null_lit())
    let node = Node(
        NodeKind.Ternary,
        {"cond": cond, "thenExpr": _use(c, maybe), "elseExpr": Node.int_lit(0)}
    )
    # `x != null` is true on the then edge and false on the else edge, so the
    # then arm sees `int` and the join is `int`, not `int?`.
    assert_eq(render(expr(c, node)), "int")
    # The facts are undone afterwards.
    assert_eq(render(c.symbol_type(maybe)), "int?")
}

fn test_the_right_operand_of_and_sees_what_the_left_established() -> void {
    let c = _ctx()
    let u = c.universe
    let maybe = _symbol(c, "x", optional_of(u, u.t_bool))
    let node = Node.logical(
        OpKind.And,
        Node.binary(OpKind.Ne, _use(c, maybe), Node.null_lit()),
        _use(c, maybe)
    )
    assert_true(expr(c, node) == u.t_bool)
    # Without the narrowing the right operand would be `bool?` and E0401.
    assert_eq(c.bag.items.len(), 0)
}

fn test_or_collects_the_facts_of_the_false_edge() -> void {
    let c = _ctx()
    let u = c.universe
    let maybe = _symbol(c, "x", optional_of(u, u.t_bool))
    let node = Node.logical(
        OpKind.Or,
        Node.binary(OpKind.Eq, _use(c, maybe), Node.null_lit()),
        _use(c, maybe)
    )
    assert_true(expr(c, node) == u.t_bool)
    assert_eq(c.bag.items.len(), 0)
}

fn test_coalesce_on_an_optional_drops_the_null_arm() -> void {
    let c = _ctx()
    let u = c.universe
    let node = Node(
        NodeKind.Coalesce,
        {"left": _binding(c, "x", optional_of(u, u.t_int)), "right": Node.int_lit(0)}
    )
    assert_eq(render(expr(c, node)), "int")
    assert_eq(c.bag.items.len(), 0)
}

fn test_coalesce_on_a_non_optional_warns_rather_than_errs() -> void {
    let c = _ctx()
    let u = c.universe
    let node = Node(
        NodeKind.Coalesce,
        {"left": _binding(c, "n", u.t_int), "right": Node.int_lit(0)}
    )
    assert_true(expr(c, node) == u.t_int)
    assert_eq(_code(c, 0), W0102_UNREACHABLE_CODE)
    assert_eq(_message(c, 0), "the right operand of `??` is unreachable: `int` is never null")
    assert_eq(c.bag.error_count, 0)
    assert_eq(c.bag.warning_count, 1)
}

# ------------------------------------------------------------------
# Conditions
# ------------------------------------------------------------------
fn test_a_condition_help_is_written_with_the_condition_own_spelling() -> void {
    let c = _ctx()
    let u = c.universe
    condition(c, _binding(c, "items", list_of(u, u.t_int)), "an `if`")
    assert_eq(_code(c, 0), E0401_CONDITION_NOT_BOOL)
    assert_eq(_message(c, 0), "the condition of an `if` must be `bool`, found `list[int]`")
    assert_eq(c.bag.items[0].help, "test the length explicitly: `items.len() > 0`")

    condition(c, _binding(c, "found", optional_of(u, u.t_str)), "a ternary")
    assert_eq(c.bag.items[1].help, "test for absence explicitly: `found != null`")

    # Anything larger than a bare name gets a placeholder rather than a guess
    # at what the reader wrote.
    condition(c, Node.binary(OpKind.Add, Node.int_lit(1), _binding(c, "n", u.t_int)), "a ternary")
    assert_eq(c.bag.items[2].help, "compare explicitly: `n != 0`")
}

fn test_a_condition_of_an_unrelated_shape_gets_the_general_help() -> void {
    let c = _ctx()
    let decl = _class(c, "Box")
    condition(c, _binding(c, "b", decl.ty), "an `if`")
    assert_eq(c.bag.items[0].help, "Jaithon has no truthiness; a condition must be a `bool`")
}

fn test_subject_of_quotes_a_name_and_nothing_else() -> void {
    assert_eq(subject_of(Node.ident("total"), "x"), "total")
    assert_eq(subject_of(Node.int_lit(1), "x"), "x")
    assert_eq(subject_of(null, "xs"), "xs")
}

# ------------------------------------------------------------------
# Iteration
# ------------------------------------------------------------------
fn test_iterable_element_answers_what_a_loop_binds() -> void {
    let c = _ctx()
    let u = c.universe
    assert_true(iterable_element(c, null, list_of(u, u.t_str)) == u.t_str)
    assert_true(iterable_element(c, null, u.t_str) == u.t_str)
    assert_true(iterable_element(c, null, u.t_bytes) == u.t_int)
    assert_true(iterable_element(c, null, u.t_any) == u.t_any)
    assert_eq(c.bag.items.len(), 0)
}

fn test_a_class_that_declares_iter_is_iterable_with_element_any() -> void {
    let c = _ctx()
    let decl = _class(c, "Bag")
    decl.members.push(MemberInfo("__iter__"))
    assert_true(iterable_element(c, null, decl.ty) == c.universe.t_any)
    assert_eq(c.bag.items.len(), 0)
}

fn test_a_class_that_declares_next_is_iterable_too() -> void {
    let c = _ctx()
    let decl = _class(c, "Cursor")
    # `is_iterable` in `predicate.jai` knows only `iter` and `__iter__`; the
    # checker adds `__next__` and the `Iterable` trait on top of it.
    decl.members.push(MemberInfo("__next__"))
    assert_true(iterable_element(c, null, decl.ty) == c.universe.t_any)
    assert_eq(c.bag.items.len(), 0)
}

fn test_a_class_that_is_not_iterable_is_reported_with_the_trait_to_add() -> void {
    let c = _ctx()
    let decl = _class(c, "Point")
    assert_true(iterable_element(c, null, decl.ty) == c.universe.t_any)
    assert_eq(_code(c, 0), E0405_NOT_ITERABLE)
    assert_eq(_message(c, 0), "`Point` is not iterable")
    assert_eq(c.bag.items[0].help, "implement `trait Iterable` on `Point` to loop over it")
}

fn test_an_int_is_not_iterable_and_the_help_shows_a_range() -> void {
    let c = _ctx()
    iterable_element(c, null, c.universe.t_int)
    assert_eq(_message(c, 0), "`int` is not iterable")
    assert_eq(c.bag.items[0].help, "loop over a range: `for i in 0..n`")
}

# ------------------------------------------------------------------
# Indexing and slicing
# ------------------------------------------------------------------
fn test_a_constant_subscript_selects_the_tuple_member_it_names() -> void {
    let c = _ctx()
    let u = c.universe
    let pair = _symbol(c, "pair", u.intern(TyKind.Tuple, null, [u.t_str, u.t_int]))
    assert_true(expr(c, Node.index(_use(c, pair), Node.int_lit(0))) == u.t_str)
    assert_true(expr(c, Node.index(_use(c, pair), Node.int_lit(1))) == u.t_int)
    # A negative subscript counts from the end.
    assert_true(expr(c, Node.index(_use(c, pair), Node.int_lit(-1))) == u.t_int)
    assert_eq(c.bag.items.len(), 0)
}

fn test_a_computed_subscript_on_a_tuple_keeps_the_join() -> void {
    let c = _ctx()
    let u = c.universe
    let pair = _symbol(c, "pair", u.intern(TyKind.Tuple, null, [u.t_str, u.t_int]))
    let at = _binding(c, "i", u.t_int)
    assert_eq(render(expr(c, Node.index(_use(c, pair), at))), "int | str")
}

fn test_a_constant_subscript_past_the_end_is_reported() -> void {
    let c = _ctx()
    let u = c.universe
    let pair = _symbol(c, "pair", u.intern(TyKind.Tuple, null, [u.t_str, u.t_int]))
    assert_true(expr(c, Node.index(_use(c, pair), Node.int_lit(2))) == u.t_any)
    assert_eq(_code(c, 0), E0404_NOT_INDEXABLE)
    assert_eq(_message(c, 0), "index 2 is out of range for a 2-member tuple")
}

fn test_an_index_of_the_wrong_type_is_checked_against_the_container() -> void {
    let c = _ctx()
    let u = c.universe
    let xs = _binding(c, "xs", list_of(u, u.t_str))
    assert_true(expr(c, Node.index(xs, Node.str_lit("a"))) == u.t_str)
    assert_eq(_code(c, 0), E0400_TYPE_MISMATCH)
    assert_eq(_message(c, 0), "invalid index type: expected `int`, found `str`")
}

fn test_a_type_that_cannot_be_indexed_offers_the_dunder() -> void {
    let c = _ctx()
    let decl = _class(c, "Point")
    let node = Node.index(_binding(c, "p", decl.ty), Node.int_lit(0))
    assert_true(expr(c, node) == c.universe.t_any)
    assert_eq(_code(c, 0), E0404_NOT_INDEXABLE)
    assert_eq(_message(c, 0), "`Point` cannot be indexed")
    assert_eq(c.bag.items[0].help, "define `__getitem__` on `Point`")
}

fn test_getitem_makes_a_user_type_indexable_and_the_result_dynamic() -> void {
    let c = _ctx()
    let decl = _class(c, "Grid")
    decl.members.push(MemberInfo("__getitem__"))
    let node = Node.index(_binding(c, "g", decl.ty), Node.int_lit(0))
    assert_true(expr(c, node) == c.universe.t_any)
    assert_eq(c.bag.items.len(), 0)
}

fn test_type_argument_brackets_erase_to_the_bare_declared_type() -> void {
    let c = _ctx()
    let decl = _class(c, "Box")
    let node = Node.index(_class_name(c, decl), Node.ident("int"))
    c.resolution.set_type_args(node)
    # Generics are erased (spec §6.1), so `Box[int]` is `Box`.
    assert_eq(render(expr(c, node)), "Box")
    assert_eq(c.bag.items.len(), 0)
}

fn test_brackets_after_a_class_name_are_type_arguments_even_unmarked() -> void {
    let c = _ctx()
    let decl = _class(c, "Box")
    let node = Node.index(_class_name(c, decl), Node.ident("int"))
    assert_eq(render(expr(c, node)), "Box")
    assert_eq(c.bag.items.len(), 0)
}

fn test_a_slice_answers_the_container_it_was_taken_from() -> void {
    let c = _ctx()
    let u = c.universe
    let xs = list_of(u, u.t_int)
    let node = Node(NodeKind.Slice, {"object": _binding(c, "xs", xs)})
    assert_true(slice(c, node) == xs)
    assert_true(slice(c, Node(NodeKind.Slice, {"object": Node.str_lit("ab")})) == u.t_str)
}

fn test_a_slice_bound_must_be_an_int_or_null() -> void {
    let c = _ctx()
    let u = c.universe
    let node = Node(
        NodeKind.Slice,
        {
            "object": Node.str_lit("ab"),
            "start": Node.str_lit("x"),
            "stop": Node.null_lit(),
        }
    )
    assert_true(slice(c, node) == u.t_str)
    assert_eq(c.bag.items.len(), 1)
    assert_eq(_message(c, 0), "a slice bound must be an int: expected `int`, found `str`")
}

fn test_a_type_that_cannot_be_sliced_is_reported() -> void {
    let c = _ctx()
    let u = c.universe
    let node = Node(NodeKind.Slice, {"object": _binding(c, "n", u.t_int)})
    assert_true(slice(c, node) == u.t_any)
    assert_eq(_message(c, 0), "`int` cannot be sliced")
}

fn test_a_range_bound_must_be_an_int() -> void {
    let c = _ctx()
    let u = c.universe
    let good = Node(NodeKind.Range, {"start": Node.int_lit(0), "stop": Node.int_lit(4)})
    assert_true(expr(c, good) == u.t_range)
    let bad = Node(NodeKind.Range, {"start": Node.int_lit(0), "stop": Node.str_lit("x")})
    assert_true(expr(c, bad) == u.t_range)
    assert_eq(_message(c, 0), "a range bound must be an int: expected `int`, found `str`")
}

fn test_an_fstring_is_a_str_and_checks_its_holes() -> void {
    let c = _ctx()
    let u = c.universe
    let literal = Node.str_lit("n = ")
    let hole = _binding(c, "n", u.t_int)
    let node = Node(NodeKind.FString, {"parts": [literal, hole]})
    assert_true(expr(c, node) == u.t_str)
    assert_true(c.type_of(literal) == u.t_str)
    assert_true(c.type_of(hole) == u.t_int)
}

# ------------------------------------------------------------------
# Member access
# ------------------------------------------------------------------
fn test_a_field_load_answers_the_declared_field_type() -> void {
    let c = _ctx()
    let u = c.universe
    let decl = _class(c, "Point")
    decl.fields.push(FieldInfo("x", u.t_int, 0, VIS_PUBLIC))
    let (t, found) = member(c, Node.member(_binding(c, "p", decl.ty), "x"))
    assert_true(t == u.t_int)
    assert_true(found is null)
    assert_eq(c.bag.items.len(), 0)
}

fn test_a_method_load_hands_back_the_declaration_behind_it() -> void {
    let c = _ctx()
    let u = c.universe
    let decl = _class(c, "Point")
    let body = Node.fn_decl("norm", [param("self")], Node.block([]))
    decl.members.push(MemberInfo("norm", fn_of(u, [], u.t_float), VIS_PUBLIC, node: body))
    let (t, found) = member(c, Node.member(_binding(c, "p", decl.ty), "norm"))
    assert_eq(render(t), "fn() -> float")
    # The C's `outDecl`: the `fn` a call through the member is bound against.
    assert_true(found == body)
}

fn test_a_private_field_is_reported_with_the_declaration_that_owns_it() -> void {
    let c = _ctx()
    let u = c.universe
    let decl = _class(c, "Account")
    decl.fields.push(FieldInfo("balance", u.t_int, 0, VIS_PRIVATE))
    member(c, Node.member(_binding(c, "a", decl.ty), "balance"))
    assert_eq(_code(c, 0), E0701_PRIVATE_ACCESS)
    assert_eq(_message(c, 0), "`balance` is private to `Account`")
    assert_eq(c.bag.items[0].help, "mark it `pub` or add an accessor method")
    assert_eq(c.bag.items[0].labels.len(), 1)
}

fn test_a_protected_field_reaches_a_subclass_and_not_a_stranger() -> void {
    let c = _ctx()
    let u = c.universe
    let base = _class(c, "Shape")
    base.fields.push(FieldInfo("sides", u.t_int, 0, VIS_PROTECTED))
    let derived = _class(c, "Circle")
    derived.superclass = base

    assert_false(visible_from(c, base, VIS_PROTECTED))
    c.current_class = derived
    assert_true(visible_from(c, base, VIS_PROTECTED))
    assert_false(visible_from(c, base, VIS_PRIVATE))
    c.current_class = base
    assert_true(visible_from(c, base, VIS_PRIVATE))
    assert_true(visible_from(c, null, VIS_PRIVATE))
    assert_eq(visibility_word(VIS_PROTECTED), "protected")
    assert_eq(visibility_word(VIS_PUBLIC), "private")
}

fn test_an_unknown_member_names_the_declaration_and_suggests_a_near_miss() -> void {
    let c = _ctx()
    let u = c.universe
    let decl = _class(c, "Point")
    decl.fields.push(FieldInfo("width", u.t_int, 0, VIS_PUBLIC))
    let (t, _found) = member(c, Node.member(_binding(c, "p", decl.ty), "widht"))
    assert_true(t == u.t_any)
    assert_eq(_code(c, 0), E0410_UNKNOWN_MEMBER)
    assert_eq(_message(c, 0), "`Point` has no member `widht`")
    assert_eq(c.bag.items[0].help, "did you mean `width`?")
    assert_eq(c.bag.items[0].labels.len(), 1)
}

fn test_a_member_of_a_class_with_an_unseen_parent_is_dynamic() -> void {
    let c = _ctx()
    let decl = _class(c, "Button")
    # The parent's declaration is not visible here, so "no member" would be a
    # guess — and it was a wrong one: every subclass of an imported class lost
    # the whole inherited surface.
    decl.opaque_super = "Widget"
    let (t, _found) = member(c, Node.member(_binding(c, "b", decl.ty), "paint"))
    assert_true(t == c.universe.t_any)
    assert_eq(c.bag.items.len(), 0)
}

fn test_a_plain_dot_on_an_optional_receiver_is_dynamic_and_silent() -> void {
    let c = _ctx()
    let u = c.universe
    let decl = _class(c, "Point")
    decl.fields.push(FieldInfo("x", u.t_int, 0, VIS_PUBLIC))
    let (t, _found) = member(c, Node.member(_binding(c, "p", optional_of(u, decl.ty)), "x"))
    # The E0400 "may be null" arm is written and cannot fire, in this port and
    # in the C alike: `Point?` is a union, a union carries no declaration, and
    # the null-declaration escape hatch above the check answers `any` first.
    # Reproduced rather than corrected — `./jaithon check` on `var p: Point?
    # = null; p.x` accepts it, so a port that reported would diverge.
    assert_true(t == u.t_any)
    assert_eq(c.bag.items.len(), 0)
}

fn test_an_optional_member_load_narrows_in_and_wraps_out() -> void {
    let c = _ctx()
    let u = c.universe
    let decl = _class(c, "Point")
    decl.fields.push(FieldInfo("x", u.t_int, 0, VIS_PUBLIC))
    let node = Node(
        NodeKind.OptMember,
        {"object": _binding(c, "p", optional_of(u, decl.ty)), "name": "x"}
    )
    assert_eq(render(expr(c, node)), "int?")
    assert_eq(c.bag.items.len(), 0)
}

fn test_a_static_member_resolves_against_the_class_and_not_an_instance() -> void {
    let c = _ctx()
    let u = c.universe
    let decl = _class(c, "Account")
    decl.fields.push(FieldInfo("count", u.t_int, -1, VIS_PUBLIC, true))
    let node = Node.member(_class_name(c, decl), "count")
    assert_true(expr(c, node) == u.t_int)
}

fn test_a_missing_enum_variant_says_how_many_there_are() -> void {
    let c = _ctx()
    let decl = TypeDecl("Colour", Span(0, 0, 1))
    decl.ty = named(c.universe, TyKind.Enum, "Colour", decl)
    decl.is_enum = true
    decl.variant_count = 1
    c.register_decl(decl)
    member(c, Node.member(_class_name(c, decl), "Blue"))
    assert_eq(_message(c, 0), "`Colour` has no member `Blue`")
    assert_eq(c.bag.items[0].help, "`Colour` has 1 variant")
}

fn test_super_resolves_in_the_parent() -> void {
    let c = _ctx()
    let u = c.universe
    let base = _class(c, "Shape")
    base.fields.push(FieldInfo("sides", u.t_int, 0, VIS_PUBLIC))
    let derived = _class(c, "Circle")
    derived.superclass = base
    c.current_class = derived

    let node = Node.member(Node(NodeKind.Super), "sides")
    assert_true(expr(c, node) == u.t_int)
    # `self` and `super` in value position answer the class and its parent.
    assert_true(expr(c, Node(NodeKind.Self)) == derived.ty)
    assert_true(expr(c, Node(NodeKind.Super)) == base.ty)
}

fn test_a_member_of_a_builtin_container_is_dynamic() -> void {
    let c = _ctx()
    let u = c.universe
    # Builtin containers carry their methods in the runtime, not in a
    # declaration, so a load off one is not an unknown member.
    let node = Node.member(_binding(c, "xs", list_of(u, u.t_int)), "push")
    assert_true(expr(c, node) == u.t_any)
    assert_eq(c.bag.items.len(), 0)
}

# ------------------------------------------------------------------
# Calls
# ------------------------------------------------------------------
fn test_a_call_of_a_declared_function_binds_against_its_parameters() -> void {
    let c = _ctx()
    let u = c.universe
    c.resolve_ast_type = _scalar_types
    let signature = fn_of(u, [u.t_int], u.t_str)
    let f = _function(c, "f", [param("n", TypeExpr.named("int"))], signature)
    let node = _call(c, f, [arg(Node.str_lit("x"))])
    assert_true(expr(c, node) == u.t_str)
    assert_eq(_code(c, 0), E0400_TYPE_MISMATCH)
    # The label names the parameter it bound to, not the position.
    assert_eq(_message(c, 0), "argument `n` of `f`: expected `int`, found `str`")
}

fn test_too_many_arguments_names_the_declaration() -> void {
    let c = _ctx()
    let u = c.universe
    let f = _function(c, "f", [param("n")], fn_of(u, [u.t_int], u.t_str))
    expr(c, _call(c, f, [arg(Node.int_lit(1)), arg(Node.int_lit(2))]))
    assert_eq(_code(c, 0), E0602_BAD_CALL_ARITY)
    assert_eq(_message(c, 0), "`f` takes 1 argument but 2 were given")
    assert_eq(c.bag.items[0].labels.len(), 1)
}

fn test_a_missing_required_argument_names_the_first_one() -> void {
    let c = _ctx()
    let u = c.universe
    let f = _function(
        c,
        "f",
        [param("a"), param("b"), param("c", null, Node.int_lit(0))],
        fn_of(u, [u.t_int, u.t_int, u.t_int], u.t_int)
    )
    expr(c, _call(c, f, []))
    assert_eq(_code(c, 0), E0602_BAD_CALL_ARITY)
    # The one with a default is not missing.
    assert_eq(_message(c, 0), "`f` is missing 2 required arguments")
    assert_eq(c.bag.items[0].help, "the first one is `a`")
}

fn test_a_keyword_argument_naming_no_parameter_suggests_the_near_miss() -> void {
    let c = _ctx()
    let u = c.universe
    let f = _function(c, "f", [param("count")], fn_of(u, [u.t_int], u.t_int))
    expr(c, _call(c, f, [arg(Node.int_lit(1), "cont")]))
    assert_eq(_code(c, 0), E0603_UNKNOWN_KEYWORD_ARG)
    assert_eq(_message(c, 0), "`f` has no parameter named `cont`")
    assert_eq(c.bag.items[0].help, "did you mean `count`?")
}

fn test_one_argument_given_twice_is_reported_once_per_repeat() -> void {
    let c = _ctx()
    let u = c.universe
    let f = _function(c, "f", [param("n")], fn_of(u, [u.t_int], u.t_int))
    expr(c, _call(c, f, [arg(Node.int_lit(1), "n"), arg(Node.int_lit(2), "n")]))
    assert_eq(_code(c, 0), E0604_DUPLICATE_ARGUMENT)
    assert_eq(_message(c, 0), "argument `n` is given twice")
    assert_eq(c.bag.items[0].labels.len(), 1)
}

fn test_a_positional_argument_after_a_keyword_one_is_reported() -> void {
    let c = _ctx()
    let u = c.universe
    let f = _function(c, "f", [param("a"), param("b")], fn_of(u, [u.t_int, u.t_int], u.t_int))
    expr(c, _call(c, f, [arg(Node.int_lit(1), "b"), arg(Node.int_lit(2))]))
    assert_eq(_code(c, 0), E0605_POSITIONAL_AFTER_KEYWORD)
    assert_eq(_message(c, 0), "positional argument follows a keyword argument")
}

fn test_a_spread_suppresses_every_arity_complaint() -> void {
    let c = _ctx()
    let u = c.universe
    let f = _function(c, "f", [param("a"), param("b")], fn_of(u, [u.t_int, u.t_int], u.t_int))
    # Nothing here can know how many values the spread contributes.
    expr(c, _call(c, f, [arg(_binding(c, "xs", list_of(u, u.t_int)), null, true)]))
    assert_eq(c.bag.items.len(), 0)
}

fn test_a_variadic_parameter_swallows_the_rest() -> void {
    let c = _ctx()
    let u = c.universe
    let f = _function(
        c,
        "f",
        [param("a"), param("rest", null, null, true)],
        fn_of(u, [u.t_int, u.t_int], u.t_int)
    )
    expr(c, _call(c, f, [arg(Node.int_lit(1)), arg(Node.int_lit(2)), arg(Node.int_lit(3))]))
    assert_eq(c.bag.items.len(), 0)
}

fn test_every_argument_is_checked_exactly_once_even_when_it_found_no_home() -> void {
    let c = _ctx()
    let u = c.universe
    let f = _function(c, "f", [param("n")], fn_of(u, [u.t_int], u.t_int))
    let stray = Node.binary(OpKind.Add, _binding(c, "s", u.t_str), Node.int_lit(1))
    expr(c, _call(c, f, [arg(Node.int_lit(1)), arg(stray)]))
    # One arity error and one operand error: the argument that overflowed is
    # still type-checked, and only once.
    assert_eq(c.bag.items.len(), 2)
    assert_eq(_code(c, 0), E0602_BAD_CALL_ARITY)
    assert_eq(_code(c, 1), E0406_BAD_OPERAND_TYPES)
}

fn test_a_call_through_a_value_shows_the_signature_at_its_binding() -> void {
    let c = _ctx()
    let u = c.universe
    # A parameter of function type denotes no declaration, so the arity is
    # checked against the type and the label says where the value came from.
    let hook = _symbol(c, "hook", fn_of(u, [u.t_int], u.t_str), SymbolKind.Param)
    let node = Node(NodeKind.Call, {"callee": _use(c, hook), "args": []})
    assert_true(expr(c, node) == u.t_str)
    assert_eq(_code(c, 0), E0602_BAD_CALL_ARITY)
    assert_eq(_message(c, 0), "`hook` takes 1 argument but 0 were given")
    let (_span, label) = c.bag.items[0].labels[0]
    assert_eq(label, "`hook` is the parameter declared here, of type `fn(int) -> str`")
}

fn test_a_binding_that_shadows_a_method_says_so() -> void {
    let c = _ctx()
    let u = c.universe
    let decl = _class(c, "Grid")
    decl.members.push(MemberInfo("draw", fn_of(u, [u.t_int], u.t_int), VIS_PUBLIC))
    c.current_class = decl
    let shadow = _symbol(c, "draw", fn_of(u, [], u.t_int), SymbolKind.Local)
    expr(c, Node(NodeKind.Call, {"callee": _use(c, shadow), "args": [arg(Node.int_lit(1))]}))
    assert_eq(_code(c, 0), E0602_BAD_CALL_ARITY)
    assert_eq(
        c.bag.items[0].help,
        "this `draw` is the local binding, which shadows the method `Grid.draw`; write `self.draw(...)` to call the method"
    )
}

fn test_a_value_that_is_not_callable_is_reported() -> void {
    let c = _ctx()
    let u = c.universe
    let node = Node(NodeKind.Call, {"callee": _binding(c, "n", u.t_int), "args": []})
    assert_true(expr(c, node) == u.t_any)
    assert_eq(_code(c, 0), E0403_NOT_CALLABLE)
    assert_eq(_message(c, 0), "`int` is not callable")
}

fn test_a_class_with_call_is_callable_and_the_result_is_dynamic() -> void {
    let c = _ctx()
    let decl = TypeDecl("Op", Span(0, 0, 1))
    decl.ty = named(c.universe, TyKind.Trait, "Op", decl)
    decl.is_trait = true
    decl.members.push(MemberInfo("__call__"))
    c.register_decl(decl)
    let node = Node(NodeKind.Call, {"callee": _binding(c, "op", decl.ty), "args": []})
    assert_true(expr(c, node) == c.universe.t_any)
    assert_eq(c.bag.items.len(), 0)
}

fn test_a_class_name_before_a_paren_constructs() -> void {
    let c = _ctx()
    let decl = _class(c, "Point")
    let init = Node.fn_decl("init", [param("self"), param("x")], Node.block([]))
    decl.members.push(MemberInfo("init", null, VIS_PUBLIC, node: init))
    let node = Node(NodeKind.Call, {"callee": _class_name(c, decl), "args": [arg(Node.int_lit(1))]})
    assert_true(expr(c, node) == decl.ty)
    assert_eq(c.bag.items.len(), 0)
}

fn test_construction_skips_self_when_counting_the_init_parameters() -> void {
    let c = _ctx()
    let decl = _class(c, "Point")
    let init = Node.fn_decl("init", [param("self"), param("x")], Node.block([]))
    decl.members.push(MemberInfo("init", null, VIS_PUBLIC, node: init))
    let node = Node(NodeKind.Call, {"callee": _class_name(c, decl), "args": []})
    expr(c, node)
    assert_eq(_message(c, 0), "`Point` is missing 1 required argument")
    assert_eq(self_skip_of(init), 1)
}

fn test_a_class_with_no_init_takes_no_arguments() -> void {
    let c = _ctx()
    let decl = _class(c, "Empty")
    let node = Node(NodeKind.Call, {"callee": _class_name(c, decl), "args": [arg(Node.int_lit(1))]})
    assert_true(expr(c, node) == decl.ty)
    assert_eq(_code(c, 0), E0602_BAD_CALL_ARITY)
    assert_eq(_message(c, 0), "`Empty` has no `init`, so it takes no arguments")
}

fn test_an_init_is_inherited_from_the_nearest_ancestor_that_declares_one() -> void {
    let c = _ctx()
    let base = _class(c, "Shape")
    let init = Node.fn_decl("init", [param("self"), param("sides")], Node.block([]))
    base.members.push(MemberInfo("init", null, VIS_PUBLIC, node: init))
    let derived = _class(c, "Circle")
    derived.superclass = base

    let (found, owner) = find_init_decl(derived)
    assert_true(found == init)
    assert_eq(owner?.name, "Shape")
    assert_true(find_init_decl(null)[0] is null)
}

fn test_a_trait_and_an_abstract_class_cannot_be_instantiated() -> void {
    let c = _ctx()
    let trait_decl = TypeDecl("Drawable", Span(0, 0, 1))
    trait_decl.ty = named(c.universe, TyKind.Trait, "Drawable", trait_decl)
    trait_decl.is_trait = true
    c.register_decl(trait_decl)
    expr(c, Node(NodeKind.Call, {"callee": _class_name(c, trait_decl), "args": []}))
    assert_eq(_code(c, 0), E0710_ABSTRACT_INSTANTIATION)
    assert_eq(_message(c, 0), "`Drawable` is a trait and cannot be instantiated")

    let abstract_decl = _class(c, "Shape")
    abstract_decl.is_abstract = true
    abstract_decl.members.push(MemberInfo("area", null, VIS_PUBLIC, is_abstract: true))
    expr(c, Node(NodeKind.Call, {"callee": _class_name(c, abstract_decl), "args": []}))
    assert_eq(_message(c, 1), "`Shape` is abstract and cannot be instantiated")
    assert_eq(c.bag.items[1].help, "`area` has no implementation")
}

fn test_an_optional_call_chain_yields_an_optional_result() -> void {
    let c = _ctx()
    let u = c.universe
    let decl = _class(c, "Point")
    decl.members.push(MemberInfo("norm", fn_of(u, [], u.t_float), VIS_PUBLIC))
    let callee = Node(
        NodeKind.OptMember,
        {"object": _binding(c, "p", optional_of(u, decl.ty)), "name": "norm"}
    )
    let node = Node(NodeKind.Call, {"callee": callee, "args": []})
    # Spec §4.2: when `p` is null nothing is called and the call yields null,
    # so the optional comes off the member load and back onto the result.
    assert_eq(render(expr(c, node)), "float?")
    assert_eq(c.bag.items.len(), 0)
}

fn test_a_method_call_binds_against_the_declaration_the_member_found() -> void {
    let c = _ctx()
    let u = c.universe
    let decl = _class(c, "Point")
    let body = Node.fn_decl("scale", [param("self"), param("by")], Node.block([]))
    decl.members.push(MemberInfo("scale", fn_of(u, [u.t_int], u.t_float), VIS_PUBLIC, node: body))
    let callee = Node.member(_binding(c, "p", decl.ty), "scale")
    expr(c, Node(NodeKind.Call, {"callee": callee, "args": []}))
    assert_eq(_message(c, 0), "`scale` is missing 1 required argument")
}

# ------------------------------------------------------------------
# Comprehensions, lambdas and `if` expressions
# ------------------------------------------------------------------
fn test_a_list_comprehension_is_a_list_of_its_element() -> void {
    let c = _ctx()
    let u = c.universe
    let node = Node(
        NodeKind.Comprehension,
        {
            "compKind": CompKind.List,
            "element": Node.str_lit("x"),
            "clauses": [comp_clause(Node.bind("n"), _binding(c, "xs", list_of(u, u.t_int)))],
        }
    )
    assert_eq(render(expr(c, node)), "list[str]")
    assert_eq(c.bag.items.len(), 0)
}

fn test_a_dict_comprehension_takes_its_key_from_the_key_expression() -> void {
    let c = _ctx()
    let u = c.universe
    let node = Node(
        NodeKind.Comprehension,
        {
            "compKind": CompKind.Dict,
            "keyExpr": Node.str_lit("k"),
            "element": Node.int_lit(1),
            "clauses": [comp_clause(Node.bind("n"), _binding(c, "xs", list_of(u, u.t_int)))],
        }
    )
    assert_eq(render(expr(c, node)), "dict[str, int]")
}

fn test_a_comprehension_filter_is_a_boolean_position() -> void {
    let c = _ctx()
    let u = c.universe
    let clause = comp_clause(
        Node.bind("n"),
        _binding(c, "xs", list_of(u, u.t_int)),
        [_binding(c, "flag", u.t_int)]
    )
    let node = Node(
        NodeKind.Comprehension,
        {"compKind": CompKind.List, "element": Node.int_lit(1), "clauses": [clause]}
    )
    expr(c, node)
    assert_eq(_code(c, 0), E0401_CONDITION_NOT_BOOL)
    assert_eq(_message(c, 0), "the condition of a comprehension filter must be `bool`, found `int`")
}

fn test_a_comprehension_over_something_uniterable_is_reported() -> void {
    let c = _ctx()
    let u = c.universe
    let node = Node(
        NodeKind.Comprehension,
        {
            "compKind": CompKind.List,
            "element": Node.int_lit(1),
            "clauses": [comp_clause(Node.bind("n"), _binding(c, "n", u.t_int))],
        }
    )
    expr(c, node)
    assert_eq(_code(c, 0), E0405_NOT_ITERABLE)
}

fn test_an_annotated_comprehension_reports_the_element_that_does_not_fit() -> void {
    let c = _ctx()
    let u = c.universe
    let node = Node(
        NodeKind.Comprehension,
        {
            "compKind": CompKind.List,
            "element": Node.str_lit("x"),
            "clauses": [comp_clause(Node.bind("n"), _binding(c, "xs", list_of(u, u.t_int)))],
        }
    )
    c.apply_context(c, node, list_of(u, u.t_int))
    assert_true(expr(c, node) == list_of(u, u.t_int))
    assert_eq(_message(c, 0), "mismatched element type: expected `int`, found `str`")
}

fn test_a_generator_comprehension_is_dynamic() -> void {
    let c = _ctx()
    let u = c.universe
    let node = Node(
        NodeKind.Comprehension,
        {
            "compKind": CompKind.Generator,
            "element": Node.int_lit(1),
            "clauses": [comp_clause(Node.bind("n"), _binding(c, "xs", list_of(u, u.t_int)))],
        }
    )
    assert_true(expr(c, node) == u.t_any)
}

fn test_a_lambda_gets_a_signature_and_a_call_through_it_is_counted() -> void {
    let c = _ctx()
    c.resolve_ast_type = _scalar_types
    let lambda = Node(
        NodeKind.Lambda,
        {
            "params": [param("n", TypeExpr.named("int"))],
            "returnType": TypeExpr.named("str"),
            "body": Node.str_lit("x"),
        }
    )
    assert_eq(render(expr(c, lambda)), "fn(int) -> str")

    let f = _symbol(c, "f", c.type_of(lambda))
    expr(c, Node(NodeKind.Call, {"callee": _use(c, f), "args": []}))
    assert_eq(_message(c, 0), "`f` takes 1 argument but 0 were given")
}

fn test_an_if_expression_without_an_else_is_optional() -> void {
    let c = _ctx()
    let node = Node(NodeKind.IfExpr, {"cond": Node.bool_lit(true), "thenBranch": Node.int_lit(1)})
    assert_eq(render(expr(c, node)), "int?")
}

fn test_an_if_expression_joins_its_branches_and_narrows_each() -> void {
    let c = _ctx()
    let u = c.universe
    let maybe = _symbol(c, "x", optional_of(u, u.t_int))
    let node = Node(
        NodeKind.IfExpr,
        {
            "cond": Node.binary(OpKind.IsNot, _use(c, maybe), Node.null_lit()),
            "thenBranch": _use(c, maybe),
            "elseBranch": Node.str_lit("none"),
        }
    )
    assert_eq(render(expr(c, node)), "int | str")
    assert_eq(render(c.symbol_type(maybe)), "int?")
}

fn test_isinstance_narrows_the_way_is_does() -> void {
    let c = _ctx()
    let u = c.universe
    let decl = _class(c, "Circle")
    let shape = _symbol(c, "s", u.t_any)
    let cond = Node(
        NodeKind.Call,
        {
            "callee": Node.ident("isinstance"),
            "args": [arg(_use(c, shape)), arg(_class_name(c, decl))],
        }
    )
    let node = Node(
        NodeKind.IfExpr,
        {"cond": cond, "thenBranch": _use(c, shape), "elseBranch": Node.null_lit()}
    )
    assert_eq(render(expr(c, node)), "Circle?")
}

# ------------------------------------------------------------------
# The dispatcher and the seam
# ------------------------------------------------------------------
fn test_yield_and_await_check_their_operand_and_answer_any() -> void {
    let c = _ctx()
    let u = c.universe
    let inner = Node.binary(OpKind.Add, _binding(c, "s", u.t_str), Node.int_lit(1))
    assert_true(expr(c, Node(NodeKind.Yield, {"operand": inner})) == u.t_any)
    assert_eq(_code(c, 0), E0406_BAD_OPERAND_TYPES)
    assert_true(expr(c, Node(NodeKind.Await, {"operand": null})) == u.t_any)
}

fn test_a_cast_answers_its_target_and_still_checks_its_operand() -> void {
    let c = _ctx()
    let u = c.universe
    c.resolve_ast_type = _scalar_types
    let inner = Node.binary(OpKind.Add, _binding(c, "s", u.t_str), Node.int_lit(1))
    let node = Node(NodeKind.Cast, {"operand": inner, "target": TypeExpr.named("float")})
    assert_true(expr(c, node) == u.t_float)
    assert_eq(_code(c, 0), E0406_BAD_OPERAND_TYPES)
}

fn test_a_statement_in_expression_position_hands_back_to_the_statement_checker() -> void {
    let c = _ctx()
    var seen = 0
    c.check_stmt = fn(inner: Ctx, node: Node?) -> void { seen += 1 }
    let t = expr(c, Node.return_stmt(Node.int_lit(1)))
    assert_eq(seen, 1)
    # `void` is `null` under the name a return position uses.
    assert_true(t == c.universe.t_void)
}

fn test_the_match_expression_seam_is_asked_and_its_answer_is_recorded() -> void {
    let c = _ctx()
    let u = c.universe
    c.match_expr = fn(inner: Ctx, node: Node?, statement: bool) -> Type {
        return inner.universe.t_str
    }
    let node = Node(NodeKind.MatchExpr, {"subject": Node.int_lit(1)})
    assert_true(expr(c, node) == u.t_str)
    assert_true(c.type_of(node) == u.t_str)
}

fn test_the_module_member_seam_is_consulted_before_the_type_is() -> void {
    let c = _ctx()
    let u = c.universe
    c.is_module = fn(inner: Ctx, node: Node?) -> bool {
        return node is not null and node.kind == NodeKind.Ident and node.opt_text("name") == "m"
    }
    c.module_member = fn(inner: Ctx, node: Node?, name: str) -> (Type, Node?) {
        return (fn_of(inner.universe, [], inner.universe.t_int), null)
    }
    let node = Node.member(_binding(c, "m", u.t_any), "f")
    assert_eq(render(expr(c, node)), "fn() -> int")
}

fn test_narrowing_unwinds_exactly_to_its_mark() -> void {
    let c = _ctx()
    let u = c.universe
    let maybe = _symbol(c, "x", optional_of(u, u.t_int))
    let cond = Node.binary(OpKind.Ne, _use(c, maybe), Node.null_lit())

    let mark = c.narrow_mark()
    let facts = c.collect_facts(cond, false)
    assert_eq(facts.len(), 1)
    c.narrow_apply(facts, false)
    assert_eq(render(c.symbol_type(maybe)), "int")
    # The declaration is still reachable while the fact is live.
    assert_eq(render(c.declared_type(maybe)), "int?")
    c.narrow_apply(facts, true)
    c.narrow_restore(mark)
    assert_eq(render(c.symbol_type(maybe)), "int?")
}

fn test_a_disjunction_guarantees_no_single_fact() -> void {
    let c = _ctx()
    let u = c.universe
    let a = _symbol(c, "a", optional_of(u, u.t_int))
    let b = _symbol(c, "b", optional_of(u, u.t_int))
    let cond = Node.logical(
        OpKind.Or,
        Node.binary(OpKind.Ne, _use(c, a), Node.null_lit()),
        Node.binary(OpKind.Ne, _use(c, b), Node.null_lit())
    )
    assert_eq(c.collect_facts(cond, false).len(), 0)
    # `not (a or b)` is a conjunction on the false edge, so both facts hold.
    assert_eq(c.collect_facts(cond, true).len(), 2)
}

fn test_value_is_expr_for_everything_that_produces_one() -> void {
    let c = _ctx()
    let u = c.universe
    assert_true(value(c, Node.str_lit("hi")) == u.t_str)
    assert_true(value(c, null) == u.t_any)
}
