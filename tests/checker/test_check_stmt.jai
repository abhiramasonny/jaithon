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
from std.test import assert_eq, assert_true
from jaithon.ast import DeclKind, Node, NodeKind, OpKind, Span, TypeExpr, catch_clause, match_arm
from jaithon.compile.check.ctx import Ctx
from jaithon.compile.check.expr import E0400_TYPE_MISMATCH, E0401_CONDITION_NOT_BOOL, E0405_NOT_ITERABLE, W0102_UNREACHABLE_CODE, expr
from jaithon.compile.check.kinds import TyKind
from jaithon.compile.check.render import render
from jaithon.compile.check.stmt import E0408_RETURN_TYPE_MISMATCH, block, function, install, stmt
from jaithon.compile.check.ty import Type, TypeDecl
from jaithon.compile.check.union import optional_of
from jaithon.compile.check.universe import Universe, list_of, named
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

fn _code(c: Ctx, at: int) -> str {
    return c.bag.items[at].code
}

fn _message(c: Ctx, at: int) -> str {
    return c.bag.items[at].message
}

fn _help(c: Ctx, at: int) -> str? {
    return c.bag.items[at].help
}

fn _not_null(c: Ctx, symbol: Symbol) -> Node {
    return Node.binary(OpKind.Ne, _use(c, symbol), Node.null_lit())
}

fn _is_null(c: Ctx, symbol: Symbol) -> Node {
    return Node.binary(OpKind.Eq, _use(c, symbol), Node.null_lit())
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
