#: `jaithon.compile.check.fold` — the constant folder and evaluator.
#:
#: The first test is the important one: it walks all thirty-two `OpKind` codes
#: and pins what each folds to over two ints, two floats, two strings and one
#: int. Every other test here explains one row of it, or one of the two things
#: a folder can get wrong that nothing else catches.
#:
#: Those two are worth naming. **A fold must never wrap**, because the VM raises
#: `OverflowError` for every operation checked here, so a wrapped fold turns a
#: run-time error into a silently different program — hence a case per operator
#: and per direction. And **a fold must not lose the checker's type**: the
#: literal knows less than the annotation it replaced a subexpression of.
#:
#: The expected values in this file are not guesses. Every one of them was taken
#: from a run of the C checker over the same expression, which is
#: the contract the retired sema diff enforced.
from std.math import INT_MAX, INT_MIN
from std.test import assert_eq, assert_false, assert_throws, assert_true
from jaithon.ast import Node, NodeKind, OpKind, Span, TypeExpr, kind_name, op_name
from jaithon.compile.check.fold import ConstKind, ConstValue
from jaithon.compile.check.fold import const_eval, const_replace, render_const, try_fold
from jaithon.compile.check.ty import Type
from jaithon.compile.check.union import optional_of, union_of
from jaithon.compile.check.universe import Universe, list_of
from jaithon.compile.lexer import DiagnosticBag

# op, `7 op 3`, `7.0 op 3.0`, `"b" op "a"`, `op 5`. `<none>` is no constant.
fn _table() -> list[tuple[OpKind, str, str, str, str]] {
    return [
        (OpKind.Add, "10", "10", "\"ba\"", "<none>"),
        (OpKind.Sub, "4", "4", "<none>", "<none>"),
        (OpKind.Mul, "21", "21", "<none>", "<none>"),
        (OpKind.Div, "2.3333333333333335", "2.3333333333333335", "<none>", "<none>"),
        (OpKind.FloorDiv, "2", "2", "<none>", "<none>"),
        (OpKind.Mod, "1", "1", "<none>", "<none>"),
        (OpKind.Pow, "343", "343", "<none>", "<none>"),
        (OpKind.AddWrap, "10", "<none>", "<none>", "<none>"),
        (OpKind.SubWrap, "4", "<none>", "<none>", "<none>"),
        (OpKind.MulWrap, "21", "<none>", "<none>", "<none>"),
        (OpKind.Band, "3", "<none>", "<none>", "<none>"),
        (OpKind.Bor, "7", "<none>", "<none>", "<none>"),
        (OpKind.Bxor, "4", "<none>", "<none>", "<none>"),
        (OpKind.Shl, "56", "<none>", "<none>", "<none>"),
        (OpKind.Shr, "0", "<none>", "<none>", "<none>"),
        (OpKind.MatMul, "<none>", "<none>", "<none>", "<none>"),
        (OpKind.Eq, "false", "false", "false", "<none>"),
        (OpKind.Ne, "true", "true", "true", "<none>"),
        (OpKind.Lt, "false", "false", "false", "<none>"),
        (OpKind.Le, "false", "false", "false", "<none>"),
        (OpKind.Gt, "true", "true", "true", "<none>"),
        (OpKind.Ge, "true", "true", "true", "<none>"),
        (OpKind.Is, "<none>", "<none>", "<none>", "<none>"),
        (OpKind.IsNot, "<none>", "<none>", "<none>", "<none>"),
        (OpKind.In, "<none>", "<none>", "<none>", "<none>"),
        (OpKind.NotIn, "<none>", "<none>", "<none>", "<none>"),
        # `and` and `or` are `Logical` nodes, so a `Binary` carrying one folds
        # to nothing however constant its operands are.
        (OpKind.And, "<none>", "<none>", "<none>", "<none>"),
        (OpKind.Or, "<none>", "<none>", "<none>", "<none>"),
        (OpKind.Neg, "<none>", "<none>", "<none>", "-5"),
        (OpKind.Pos, "<none>", "<none>", "<none>", "5"),
        (OpKind.Not, "<none>", "<none>", "<none>", "<none>"),
        (OpKind.Bnot, "<none>", "<none>", "<none>", "-6"),
    ]
}

fn test_every_operator_code_has_a_row() -> void {
    let rows = _table()
    # One row per `OpKind`. A code added to the enum with no row here fails
    # this line, which is the point of counting.
    assert_eq(rows.len(), 32)

    for (op, on_ints, on_floats, on_strs, on_one_int) in rows {
        let where = op_name(op)
        assert_eq(_binary(op, Node.int_lit(7), Node.int_lit(3)), on_ints, where)
        assert_eq(_binary(op, Node.float_lit(7.0), Node.float_lit(3.0)), on_floats, where)
        assert_eq(_binary(op, Node.str_lit("b"), Node.str_lit("a")), on_strs, where)
        assert_eq(_unary(op, Node.int_lit(5)), on_one_int, where)
    }
}

fn test_an_ordinary_fold_reports_nothing() -> void {
    # The whole table again, with somewhere for a diagnostic to go. Only
    # overflow and division by zero report, and no row above does either.
    let bag = DiagnosticBag()
    for (op, _ints, _floats, _strs, _one_int) in _table() {
        const_eval(Node.binary(op, Node.int_lit(7), Node.int_lit(3)), bag)
        const_eval(Node.binary(op, Node.float_lit(7.0), Node.float_lit(3.0)), bag)
        const_eval(Node.unary(op, Node.int_lit(5)), bag)
    }
    assert_eq(bag.error_count, 0)
    assert_eq(bag.warning_count, 0)
}

# ------------------------------------------------------------------
# Overflow
# ------------------------------------------------------------------
fn test_overflow_upwards_folds_nothing_and_reports_e0900() -> void {
    # Every operator that can leave the top of the range.
    _assert_overflow(OpKind.Add, Node.int_lit(INT_MAX), Node.int_lit(1), "+")
    _assert_overflow(OpKind.Sub, Node.int_lit(INT_MAX), Node.int_lit(-1), "-")
    _assert_overflow(OpKind.Mul, Node.int_lit(INT_MAX), Node.int_lit(2), "*")
    _assert_overflow(OpKind.Pow, Node.int_lit(2), Node.int_lit(70), "**")
}

fn test_overflow_downwards_folds_nothing_and_reports_e0900() -> void {
    _assert_overflow(OpKind.Add, Node.int_lit(INT_MIN), Node.int_lit(-1), "+")
    _assert_overflow(OpKind.Sub, Node.int_lit(INT_MIN), Node.int_lit(1), "-")
    _assert_overflow(OpKind.Mul, Node.int_lit(INT_MIN), Node.int_lit(2), "*")
    # The one quotient with no `int` answer, and the one negation.
    _assert_overflow(OpKind.FloorDiv, Node.int_lit(INT_MIN), Node.int_lit(-1), "//")
    _assert_overflow(OpKind.Pow, Node.int_lit(-2), Node.int_lit(70), "**")
}

fn test_negating_the_smallest_int_overflows() -> void {
    let bag = DiagnosticBag()
    let node = Node.unary(OpKind.Neg, Node.int_lit(INT_MIN))
    assert_true(const_eval(node, bag).is_none())
    assert_eq(bag.items[0].code, "E0900")
    assert_eq(bag.items[0].message, "integer overflow in constant `-`")
}

fn test_the_edges_that_do_fit_still_fold() -> void {
    # One below each overflow above, so the tests there are pinning the edge
    # rather than a range that was never folded at all.
    assert_eq(_binary(OpKind.Add, Node.int_lit(INT_MAX - 1), Node.int_lit(1)), f"{INT_MAX}")
    assert_eq(_binary(OpKind.Sub, Node.int_lit(INT_MIN + 1), Node.int_lit(1)), f"{INT_MIN}")
    assert_eq(_binary(OpKind.Mul, Node.int_lit(INT_MIN // 2), Node.int_lit(2)), f"{INT_MIN}")
    assert_eq(_binary(OpKind.Pow, Node.int_lit(2), Node.int_lit(62)), "4611686018427387904")
    # `2 ** 63` overflows only in the square the loop does not use, which is
    # why the squaring is guarded: an unguarded one reports on this.
    assert_eq(_binary(OpKind.Pow, Node.int_lit(-2), Node.int_lit(63)), f"{INT_MIN}")
}

fn test_the_wrapping_operators_wrap_instead_of_reporting() -> void {
    # `+%` asked to wrap, so wrapping is the answer and not an overflow.
    let bag = DiagnosticBag()
    let node = Node.binary(OpKind.AddWrap, Node.int_lit(INT_MAX), Node.int_lit(1))
    assert_eq(render_const(const_eval(node, bag)), f"{INT_MIN}")
    assert_eq(_binary(OpKind.SubWrap, Node.int_lit(INT_MIN), Node.int_lit(1)), f"{INT_MAX}")
    assert_eq(_binary(OpKind.MulWrap, Node.int_lit(INT_MAX), Node.int_lit(2)), "-2")
    assert_eq(bag.error_count, 0)
}

fn test_a_float_overflow_is_a_value_and_not_an_error() -> void {
    # IEEE 754 answers `inf`, which is a number, so there is nothing to report.
    let bag = DiagnosticBag()
    let node = Node.binary(OpKind.Mul, Node.float_lit(1e300), Node.float_lit(10.0))
    assert_eq(render_const(const_eval(node, bag)), "1.0000000000000001e+301")
    assert_eq(bag.error_count, 0)
}

# ------------------------------------------------------------------
# Division by zero
# ------------------------------------------------------------------
fn test_division_by_zero_folds_nothing_and_reports_e0901() -> void {
    _assert_divide_by_zero(OpKind.Div, Node.int_lit(1), Node.int_lit(0), "/")
    _assert_divide_by_zero(OpKind.FloorDiv, Node.int_lit(1), Node.int_lit(0), "//")
    _assert_divide_by_zero(OpKind.Mod, Node.int_lit(1), Node.int_lit(0), "%")
    _assert_divide_by_zero(OpKind.Div, Node.float_lit(1.0), Node.float_lit(0.0), "/")
    _assert_divide_by_zero(OpKind.FloorDiv, Node.float_lit(1.0), Node.float_lit(0.0), "//")
    _assert_divide_by_zero(OpKind.Mod, Node.float_lit(1.0), Node.float_lit(0.0), "%")
}

fn test_a_negative_shift_count_is_reported_as_e0900() -> void {
    _assert_error(
        Node.binary(OpKind.Shl, Node.int_lit(1), Node.int_lit(-1)),
        "E0900",
        "negative shift count in constant `<<`"
    )
    _assert_error(
        Node.binary(OpKind.Shr, Node.int_lit(1), Node.int_lit(-1)),
        "E0900",
        "negative shift count in constant `>>`"
    )
}

fn test_a_shift_past_the_width_discards_bits_rather_than_failing() -> void {
    # Exactly what `bitwise()` in the VM computes, or a folded shift would mean
    # something different from the same shift behind a variable.
    let bag = DiagnosticBag()
    assert_eq(_binary(OpKind.Shl, Node.int_lit(1), Node.int_lit(64)), "0")
    assert_eq(_binary(OpKind.Shl, Node.int_lit(1), Node.int_lit(63)), f"{INT_MIN}")
    assert_eq(_binary(OpKind.Shr, Node.int_lit(-1), Node.int_lit(64)), "-1")
    assert_eq(_binary(OpKind.Shr, Node.int_lit(8), Node.int_lit(100)), "0")
    const_eval(Node.binary(OpKind.Shl, Node.int_lit(1), Node.int_lit(64)), bag)
    assert_eq(bag.error_count, 0)
}

fn test_without_a_bag_a_failed_fold_is_silent() -> void {
    # The C reports only where a constant is required. Here that is the bag:
    # with nowhere to put an error the fold gives up and the VM raises.
    assert_true(const_eval(Node.binary(OpKind.Div, Node.int_lit(1), Node.int_lit(0))).is_none())
    assert_true(const_eval(Node.binary(OpKind.Pow, Node.int_lit(2), Node.int_lit(70))).is_none())
}

fn test_the_error_lands_on_the_span_of_the_operator_node() -> void {
    let bag = DiagnosticBag()
    let span = Span(0, 12, 20)
    let node = Node.binary(OpKind.Div, Node.int_lit(1), Node.int_lit(0)).with_span(span)
    const_eval(node, bag)
    assert_true(bag.items[0].span == span)
}

# ------------------------------------------------------------------
# The arithmetic that is easy to get wrong
# ------------------------------------------------------------------
fn test_int_division_yields_a_float_and_floor_division_an_int() -> void {
    # `/` on two ints is float (BYTECODE §3.3); `//` is the int division.
    assert_eq(_binary(OpKind.Div, Node.int_lit(7), Node.int_lit(2)), "3.5")
    assert_eq(_binary(OpKind.FloorDiv, Node.int_lit(7), Node.int_lit(2)), "3")
}

fn test_floor_division_and_modulo_floor_towards_negative_infinity() -> void {
    assert_eq(_binary(OpKind.FloorDiv, Node.int_lit(-7), Node.int_lit(3)), "-3")
    assert_eq(_binary(OpKind.FloorDiv, Node.int_lit(7), Node.int_lit(-3)), "-3")
    assert_eq(_binary(OpKind.Mod, Node.int_lit(-7), Node.int_lit(3)), "2")
    assert_eq(_binary(OpKind.Mod, Node.int_lit(7), Node.int_lit(-3)), "-2")
    # The remainder takes the sign of the divisor for floats too, which `fmod`
    # on its own does not do.
    assert_eq(_binary(OpKind.Mod, Node.float_lit(7.0), Node.float_lit(-3.0)), "-2")
    assert_eq(_binary(OpKind.Mod, Node.float_lit(-7.0), Node.float_lit(3.0)), "2")
}

fn test_the_smallest_int_modulo_minus_one_is_zero() -> void {
    # The quotient overflows and the remainder does not, so `%` answers where
    # `//` reports.
    assert_eq(_binary(OpKind.Mod, Node.int_lit(INT_MIN), Node.int_lit(-1)), "0")
}

fn test_a_negative_exponent_is_left_to_the_vm() -> void {
    # It has no `int` result, and widening it to a float here would fold a
    # different value from the one the VM raises on.
    assert_eq(_binary(OpKind.Pow, Node.int_lit(2), Node.int_lit(-1)), "<none>")
    assert_eq(_binary(OpKind.Pow, Node.float_lit(2.0), Node.float_lit(-1.0)), "0.5")
}

fn test_a_mixed_pair_folds_as_two_floats() -> void {
    assert_eq(_binary(OpKind.Add, Node.int_lit(1), Node.float_lit(2.5)), "3.5")
    assert_eq(_binary(OpKind.Mul, Node.int_lit(3), Node.float_lit(1.5)), "4.5")
    assert_eq(_binary(OpKind.Lt, Node.int_lit(1), Node.float_lit(2.5)), "true")
}

# ------------------------------------------------------------------
# Strings
# ------------------------------------------------------------------
fn test_strings_concatenate_and_compare() -> void {
    assert_eq(_binary(OpKind.Add, Node.str_lit("ab"), Node.str_lit("cd")), "\"abcd\"")
    assert_eq(_binary(OpKind.Eq, Node.str_lit("ab"), Node.str_lit("ab")), "true")
    # A prefix sorts before what extends it, which is the length tie-break the
    # C reaches after `memcmp` runs out of common bytes.
    assert_eq(_binary(OpKind.Lt, Node.str_lit("ab"), Node.str_lit("abc")), "true")
    assert_eq(_binary(OpKind.Ge, Node.str_lit("b"), Node.str_lit("ab")), "true")
    assert_eq(_binary(OpKind.Le, Node.str_lit(""), Node.str_lit("")), "true")
}

fn test_a_string_against_a_non_string_folds_to_nothing() -> void {
    # `"ab" * 3` is a repetition this does not do, and `"ab" + 1` is an error
    # the checker reports; either way there is no constant here.
    assert_eq(_binary(OpKind.Mul, Node.str_lit("ab"), Node.int_lit(3)), "<none>")
    assert_eq(_binary(OpKind.Add, Node.str_lit("ab"), Node.int_lit(1)), "<none>")
    assert_eq(_binary(OpKind.Lt, Node.str_lit("ab"), Node.int_lit(1)), "<none>")
}

fn test_two_bools_fold_only_their_equality() -> void {
    assert_eq(_binary(OpKind.Eq, Node.bool_lit(true), Node.bool_lit(true)), "true")
    assert_eq(_binary(OpKind.Ne, Node.bool_lit(true), Node.bool_lit(false)), "true")
    assert_eq(_binary(OpKind.Lt, Node.bool_lit(true), Node.bool_lit(false)), "<none>")
    assert_eq(_binary(OpKind.Add, Node.bool_lit(true), Node.bool_lit(true)), "<none>")
}

# ------------------------------------------------------------------
# The shapes beyond a binary operator
# ------------------------------------------------------------------
fn test_a_logical_short_circuits_past_a_non_constant_operand() -> void {
    let unknown = Node.ident("x")
    let false_and = Node.logical(OpKind.And, Node.bool_lit(false), unknown)
    let true_or = Node.logical(OpKind.Or, Node.bool_lit(true), unknown)
    assert_eq(render_const(const_eval(false_and)), "false")
    assert_eq(render_const(const_eval(true_or)), "true")
    # The other way round there is nothing to decide without the right operand.
    assert_eq(
        render_const(const_eval(Node.logical(OpKind.And, Node.bool_lit(true), unknown))),
        "<none>"
    )
    assert_eq(
        render_const(
            const_eval(Node.logical(OpKind.And, Node.bool_lit(true), Node.bool_lit(false)))
        ),
        "false"
    )
}

fn test_a_ternary_folds_to_the_branch_its_condition_picks() -> void {
    let taken = _ternary(Node.bool_lit(true), Node.int_lit(1), Node.ident("x"))
    assert_eq(render_const(const_eval(taken)), "1")
    let other = _ternary(Node.bool_lit(false), Node.ident("x"), Node.int_lit(2))
    assert_eq(render_const(const_eval(other)), "2")
    # A condition that is not a constant bool decides nothing.
    let unknown = _ternary(Node.ident("c"), Node.int_lit(1), Node.int_lit(1))
    assert_true(const_eval(unknown).is_none())
}

fn test_coalesce_folds_to_the_left_unless_it_is_null() -> void {
    let null_left = Node(NodeKind.Coalesce, {"left": Node.null_lit(), "right": Node.int_lit(5)})
    assert_eq(render_const(const_eval(null_left)), "5")
    let value_left = Node(NodeKind.Coalesce, {"left": Node.int_lit(1), "right": Node.int_lit(5)})
    assert_eq(render_const(const_eval(value_left)), "1")
}

fn test_a_comparison_chain_is_every_link_and_ed_together() -> void {
    assert_eq(render_const(const_eval(_chain([1, 2, 3], [OpKind.Lt, OpKind.Lt]))), "true")
    # A false link ends the chain, which is the same short circuit the emitter
    # produces for one that is not constant.
    assert_eq(render_const(const_eval(_chain([3, 2, 1], [OpKind.Lt, OpKind.Lt]))), "false")
    assert_eq(render_const(const_eval(_chain([1, 2, 1], [OpKind.Lt, OpKind.Lt]))), "false")
}

fn test_a_malformed_chain_folds_to_nothing_rather_than_indexing_past_its_end() -> void {
    let no_ops = Node(NodeKind.CompareChain, {"operands": [Node.int_lit(1)], "ops": []})
    assert_true(const_eval(no_ops).is_none())
    let short = Node(NodeKind.CompareChain, {"operands": [Node.int_lit(1)], "ops": [OpKind.Lt]})
    assert_true(const_eval(short).is_none())
}

fn test_a_cast_folds_to_what_it_casts() -> void {
    let cast = Node(NodeKind.Cast, {"operand": Node.int_lit(7), "target": TypeExpr.named("int")})
    assert_eq(render_const(const_eval(cast)), "7")
}

fn test_the_literals_are_their_own_constants() -> void {
    assert_eq(render_const(const_eval(Node.int_lit(-5))), "-5")
    assert_eq(render_const(const_eval(Node.float_lit(0.5))), "0.5")
    assert_eq(render_const(const_eval(Node.bool_lit(false))), "false")
    assert_eq(render_const(const_eval(Node.str_lit("hi"))), "\"hi\"")
    assert_eq(render_const(const_eval(Node.null_lit())), "null")
}

fn test_a_null_node_is_not_a_constant() -> void {
    assert_true(const_eval(null).is_none())
}

# ------------------------------------------------------------------
# Constant propagation
# ------------------------------------------------------------------
fn test_a_const_binding_inlines_into_the_expressions_that_use_it() -> void {
    let consts: dict[str, Node] = {"K": Node.int_lit(8)}
    let node = Node.binary(OpKind.Mul, Node.ident("K"), Node.int_lit(2))
    assert_eq(render_const(const_eval(node, null, consts)), "16")
    # A name with no folded value behind it is not a constant.
    assert_true(const_eval(Node.ident("other"), null, consts).is_none())
}

fn test_one_const_may_be_written_in_terms_of_another() -> void {
    let consts: dict[str, Node] = {
        "A": Node.ident("B"),
        "B": Node.binary(OpKind.Add, Node.int_lit(1), Node.int_lit(2)),
    }
    assert_eq(render_const(const_eval(Node.ident("A"), null, consts)), "3")
}

fn test_a_const_that_refers_to_itself_stops_instead_of_looping() -> void {
    # Not reachable through the resolver, which rejects the declaration. It is
    # reachable through this dict, and a compiler that hangs is worse than one
    # that folds nothing.
    let consts: dict[str, Node] = {"A": Node.ident("A")}
    assert_true(const_eval(Node.ident("A"), null, consts).is_none())
}

# ------------------------------------------------------------------
# The rewrite
# ------------------------------------------------------------------
fn test_try_fold_replaces_the_node_in_place_and_keeps_its_span() -> void {
    let u = Universe()
    let span = Span(0, 4, 9)
    let node = Node.binary(OpKind.Add, Node.int_lit(2), Node.int_lit(3)).with_span(span)
    let (folded, ty) = try_fold(u, node, u.t_int)

    assert_true(folded)
    assert_eq(kind_name(node.kind), "AST_INT_LIT")
    assert_eq(node.number("intLit"), 5)
    assert_true(node.span == span)
    assert_true(ty == u.t_int)
    # The node is the literal outright: the operator's fields are gone rather
    # than left behind for the encoder to trip over.
    assert_false(node.has("left"))
}

fn test_each_of_the_six_operator_kinds_folds() -> void {
    let u = Universe()
    let nodes: list[Node] = [
        Node.unary(OpKind.Neg, Node.int_lit(1)),
        Node.binary(OpKind.Add, Node.int_lit(1), Node.int_lit(1)),
        Node.logical(OpKind.And, Node.bool_lit(true), Node.bool_lit(true)),
        _ternary(Node.bool_lit(true), Node.int_lit(1), Node.int_lit(2)),
        _chain([1, 2], [OpKind.Lt]),
        Node(
            NodeKind.Coalesce,
            {"left": Node.null_lit(), "right": Node.int_lit(1)}
        ),
    ]
    for node in nodes {
        let (folded, _ty) = try_fold(u, node)
        assert_true(folded, kind_name(node.kind))
    }
}

fn test_nothing_but_those_six_kinds_folds() -> void {
    # A literal is already a literal, and every other expression either has an
    # effect or is not one this walk can evaluate.
    let u = Universe()
    let nodes: list[Node] = [
        Node.int_lit(1),
        Node.str_lit("x"),
        Node.ident("x"),
        Node.call(Node.ident("f"), [Node.int_lit(1)]),
        Node.index(Node.ident("xs"), Node.int_lit(0)),
        Node.expr_stmt(Node.int_lit(1)),
    ]
    for node in nodes {
        let before = node.kind
        let (folded, ty) = try_fold(u, node, u.t_int)
        assert_false(folded, kind_name(before))
        assert_true(node.kind == before, kind_name(before))
        assert_true(ty == u.t_int, kind_name(before))
    }
}

fn test_a_non_constant_expression_folds_to_nothing() -> void {
    let u = Universe()
    let node = Node.binary(OpKind.Add, Node.ident("x"), Node.int_lit(1))
    let (folded, ty) = try_fold(u, node, u.t_int)

    assert_false(folded)
    assert_eq(kind_name(node.kind), "AST_BINARY")
    assert_true(ty == u.t_int)
    assert_true(const_eval(node).is_none())
}

fn test_an_overflowing_fold_leaves_the_node_alone() -> void {
    # The one failure a constant folder must never have: reporting the overflow
    # and then rewriting the node with whatever the wrap produced.
    let u = Universe()
    let bag = DiagnosticBag()
    let node = Node.binary(OpKind.Mul, Node.int_lit(INT_MAX), Node.int_lit(2))
    let (folded, _ty) = try_fold(u, node, u.t_int, bag)

    assert_false(folded)
    assert_eq(kind_name(node.kind), "AST_BINARY")
    assert_eq(bag.items[0].code, "E0900")
}

fn test_const_replace_writes_each_literal_kind_with_its_type() -> void {
    let u = Universe()
    let cases: list[tuple[ConstValue, str, Type]] = [
        (
            ConstValue.of_int(3),
            "AST_INT_LIT",
            u.t_int
        ),
        (
            ConstValue.of_float(0.5),
            "AST_FLOAT_LIT",
            u.t_float
        ),
        (
            ConstValue.of_bool(true),
            "AST_BOOL_LIT",
            u.t_bool
        ),
        (
            ConstValue.of_str("s"),
            "AST_STR_LIT",
            u.t_str
        ),
        (
            ConstValue.of_null(),
            "AST_NULL_LIT",
            u.t_null
        ),
    ]
    for (value, kind, ty) in cases {
        let node = Node.ident("x")
        assert_true(const_replace(u, node, value) == ty, kind)
        assert_eq(kind_name(node.kind), kind)
    }
    # A string literal carries its length, which nothing else fills in.
    let node = Node.ident("x")
    const_replace(u, node, ConstValue.of_str("abc"))
    assert_eq(node.number("length"), 3)
}

fn test_const_replace_does_nothing_for_a_value_that_is_not_one() -> void {
    let u = Universe()
    let node = Node.ident("x")
    assert_true(const_replace(u, node, ConstValue.none()) is null)
    assert_eq(kind_name(node.kind), "AST_IDENT")
}

# ------------------------------------------------------------------
# The type the folded node keeps
# ------------------------------------------------------------------
fn test_a_fold_keeps_the_checkers_type_when_the_literal_still_fits_it() -> void {
    # A union or an alias is more informative than the literal's own type, and
    # a fold that replaced it would narrow the program behind the checker's
    # back.
    let u = Universe()
    let wanted = union_of(u, [u.t_int, u.t_str])
    let (folded, ty) = try_fold(u, _sum(), wanted)
    assert_true(folded)
    assert_true(ty == wanted)

    let (_folded_any, any_ty) = try_fold(u, _sum(), u.t_any)
    assert_true(any_ty == u.t_any)

    let (_folded_opt, opt_ty) = try_fold(u, _sum(), optional_of(u, u.t_int))
    assert_true(opt_ty == optional_of(u, u.t_int))
}

fn test_a_fold_takes_the_literals_type_when_the_fit_needs_a_guard() -> void {
    # `int` reaching `float` is a conversion rather than a subtype relation, so
    # the checker's `float` is not a type this literal has: it is one something
    # would have had to convert it into.
    let u = Universe()
    let (_folded, ty) = try_fold(u, _sum(), u.t_float)
    assert_true(ty == u.t_int)
}

fn test_a_fold_takes_the_literals_type_when_it_does_not_fit_at_all() -> void {
    let u = Universe()
    let (_folded, ty) = try_fold(u, _sum(), list_of(u, u.t_int))
    assert_true(ty == u.t_int)
}

fn test_a_fold_with_no_checked_type_answers_the_literals_own() -> void {
    let u = Universe()
    let (_folded, ty) = try_fold(u, _sum())
    assert_true(ty == u.t_int)
}

# ------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------
fn test_a_folded_value_renders_the_way_the_c_renders_it() -> void {
    assert_eq(render_const(ConstValue.of_int(5)), "5")
    assert_eq(render_const(ConstValue.of_int(INT_MIN)), "-9223372036854775808")
    assert_eq(render_const(ConstValue.of_bool(true)), "true")
    assert_eq(render_const(ConstValue.of_bool(false)), "false")
    assert_eq(render_const(ConstValue.of_null()), "null")
    assert_eq(render_const(ConstValue.none()), "<none>")
}

fn test_a_float_renders_through_the_seventeen_digits_the_constant_pool_uses() -> void {
    # Not the shortest round-tripping form: `%.17g` is what the C writes, and
    # the dump is compared byte for byte.
    assert_eq(render_const(ConstValue.of_float(1.0 / 3.0)), "0.33333333333333331")
    assert_eq(render_const(ConstValue.of_float(0.1)), "0.10000000000000001")
    assert_eq(render_const(ConstValue.of_float(100.0)), "100")
    assert_eq(render_const(ConstValue.of_float(2.5)), "2.5")
}

fn test_a_string_renders_quoted_with_the_three_escapes_a_dump_line_needs() -> void {
    assert_eq(render_const(ConstValue.of_str("ab")), "\"ab\"")
    assert_eq(render_const(ConstValue.of_str("a\nb")), "\"a\\nb\"")
    assert_eq(render_const(ConstValue.of_str("a\"b")), "\"a\\\"b\"")
    assert_eq(render_const(ConstValue.of_str("a\\b")), "\"a\\\\b\"")
    # A tab is not one of the three: the C passes it through and so does this.
    assert_eq(render_const(ConstValue.of_str("a\tb")), "\"a\tb\"")
}

# ------------------------------------------------------------------
# ConstValue itself
# ------------------------------------------------------------------
fn test_a_const_value_is_its_kind_and_its_value() -> void {
    assert_true(ConstValue.of_int(5) == ConstValue.of_int(5))
    assert_false(ConstValue.of_int(5) == ConstValue.of_int(6))
    # An int and a float that compare equal are still different constants: one
    # of them rewrites the node to `AST_INT_LIT` and the other does not.
    assert_false(ConstValue.of_int(5) == ConstValue.of_float(5.0))
    assert_false(ConstValue.of_bool(true) == ConstValue.of_int(1))
    assert_true(ConstValue.none().kind == ConstKind.None)
    assert_true(ConstValue.of_null().kind == ConstKind.Null)
}

fn test_reading_a_constant_as_the_wrong_kind_raises() -> void {
    # A folded `int` read back as a `str` is a bug in the caller, not a value
    # to invent a default for.
    assert_throws(ValueError, || ConstValue.of_int(1).as_str())
    assert_throws(ValueError, || ConstValue.of_str("x").as_int())
    assert_throws(ValueError, || ConstValue.none().as_bool())
    assert_throws(ValueError, || ConstValue.of_bool(true).as_float())
    # An int does read as a float, which is the widening a mixed pair needs.
    assert_eq(ConstValue.of_int(2).as_float(), 2.0)
    assert_true(ConstValue.of_int(2).is_numeric())
    assert_false(ConstValue.of_str("x").is_numeric())
}

fn test_the_module_example_is_true() -> void {
    # The doc runner does not reach a module-level block, so its example is
    # asserted here instead of going unchecked.
    let plus = Node.binary(OpKind.Add, Node.int_lit(2), Node.int_lit(3))
    assert_eq(render_const(const_eval(plus)), "5")
    assert_true(const_eval(Node.ident("x")).is_none())
    assert_eq(const_eval(Node.ident("K"), null, {"K": Node.int_lit(8)}).as_int(), 8)
}

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
fn _binary(op: OpKind, left: Node, right: Node) -> str {
    return render_const(const_eval(Node.binary(op, left, right)))
}

fn _unary(op: OpKind, operand: Node) -> str {
    return render_const(const_eval(Node.unary(op, operand)))
}

fn _ternary(cond: Node, then_expr: Node, else_expr: Node) -> Node {
    return Node(NodeKind.Ternary, {"cond": cond, "thenExpr": then_expr, "elseExpr": else_expr})
}

fn _chain(values: list[int], ops: list[OpKind]) -> Node {
    return Node(
        NodeKind.CompareChain,
        {"operands": [Node.int_lit(value) for value in values], "ops": ops}
    )
}

# A fresh `2 + 3`, because `try_fold` consumes the node it is handed.
fn _sum() -> Node {
    return Node.binary(OpKind.Add, Node.int_lit(2), Node.int_lit(3))
}

fn _assert_overflow(op: OpKind, left: Node, right: Node, spelling: str) -> void {
    _assert_error(
        Node.binary(op, left, right),
        "E0900",
        f"integer overflow in constant `{spelling}`"
    )
}

fn _assert_divide_by_zero(op: OpKind, left: Node, right: Node, spelling: str) -> void {
    _assert_error(
        Node.binary(op, left, right),
        "E0901",
        f"division by zero in constant `{spelling}`"
    )
}

fn _assert_error(node: Node, code: str, message: str) -> void {
    let bag = DiagnosticBag()
    assert_true(const_eval(node, bag).is_none(), message)
    assert_eq(bag.error_count, 1, message)
    assert_eq(bag.items[0].code, code, message)
    assert_eq(bag.items[0].message, message)
}
