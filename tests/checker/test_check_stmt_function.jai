#: `jaithon.compile.check.stmt` — function signatures, bodies, and the install seam.
from std.test import assert_eq, assert_false, assert_true
from jaithon.ast import CompKind, DeclKind, Node, NodeKind, Span, TypeExpr, TypeKind, comp_clause, match_arm, param
from jaithon.compile.check.ctx import Ctx
from jaithon.compile.check.expr import E0400_TYPE_MISMATCH, E0401_CONDITION_NOT_BOOL, expr
from jaithon.compile.check.kinds import TyKind
from jaithon.compile.check.render import render
from jaithon.compile.check.stmt import E0408_RETURN_TYPE_MISMATCH, E0601_MISSING_RETURN, block, declared_return_type, function, function_type, install, is_init_method, push_generics
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

# ------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------
fn test_is_init_method_names_only_the_constructor() -> void {
    assert_true(is_init_method(Node.fn_decl("init", [], Node.block([]))))
    assert_false(is_init_method(Node.fn_decl("area", [], Node.block([]))))
    assert_false(is_init_method(null))
    assert_false(is_init_method(Node(NodeKind.Lambda, {"name": "init"})))
}

fn test_an_unannotated_result_is_any_and_a_constructor_is_void() -> void {
    let c = _ctx()
    let plain = Node.fn_decl("area", [], Node.block([]))
    assert_true(declared_return_type(c, plain, false) == c.universe.t_any)
    let ctor = Node.fn_decl("init", [param("self")], Node.block([]))
    assert_true(declared_return_type(c, ctor, true) == c.universe.t_void)
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
