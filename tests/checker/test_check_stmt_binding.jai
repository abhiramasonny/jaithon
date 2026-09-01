#: `jaithon.compile.check.stmt` — let, var, const, and every form of store.
from std.test import assert_eq, assert_true
from jaithon.ast import DeclKind, Node, NodeKind, OpKind, Span, TypeExpr
from jaithon.compile.check.ctx import Ctx, E0411_INT_FLOAT_MIX
from jaithon.compile.check.expr import E0400_TYPE_MISMATCH, E0406_BAD_OPERAND_TYPES, E0701_PRIVATE_ACCESS, VIS_PRIVATE, VIS_PUBLIC, expr
from jaithon.compile.check.fold import const_eval
from jaithon.compile.check.kinds import TyKind
from jaithon.compile.check.render import render
from jaithon.compile.check.stmt import E0301_ASSIGN_TO_IMMUTABLE, E0303_NON_CONSTANT_CONST, E0304_MISSING_INITIALIZER, E0702_UNDECLARED_FIELD, block, install, stmt
from jaithon.compile.check.ty import FieldInfo, MemberInfo, Type, TypeDecl
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

fn _seq(kind: NodeKind, elems: list[Node], rest: int = -1) -> Node {
    return Node(kind, {"elems": elems, "restIndex": rest})
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
    let init = Node.binary(OpKind.Pow, Node.int_lit(2), Node.int_lit(70))
    stmt(c, Node.var_decl(DeclKind.Const, Node.bind("K"), null, init))
    assert_eq(_code(c, 0), "E0900")
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
    assert_eq(c.bag.items.len(), 0)
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
