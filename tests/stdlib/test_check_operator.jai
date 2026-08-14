#: `jaithon.compile.check.operator` — what an operator makes of its operands.
#:
#: The first test is the important one: it walks all thirty-two `OpKind` codes
#: and pins the result of each against `int op int`, `float op float` and
#: `op int`. Every other test here explains one row of that table; the table
#: itself is what catches a code being left out, because a missing row fails on
#: the length assertion rather than by being quietly untested.
#:
#: Nothing in this file expects a diagnostic, because the module raises none.
#: A null result is the whole error channel, and `render` spells it
#: `<unknown>` — which is why the table's cells are strings.
from std.test import assert_eq, assert_true
from jaithon.ast import OpKind, Span, op_is_comparison, op_name
from jaithon.compile.check.kinds import MAX_DECL_DEPTH, TyKind
from jaithon.compile.check.operator import arith_dunder, binary_result, unary_result
from jaithon.compile.check.render import render
from jaithon.compile.check.ty import MemberInfo, Type, TypeDecl
from jaithon.compile.check.union import union_of
from jaithon.compile.check.universe import Universe, fn_of, list_of, named

fn _table() -> list[tuple[OpKind, str, str, str]] {
    return [
        (OpKind.Add, "int", "float", "<unknown>"),
        (OpKind.Sub, "int", "float", "<unknown>"),
        (OpKind.Mul, "int", "float", "<unknown>"),
        (OpKind.Div, "float", "float", "<unknown>"),
        (OpKind.FloorDiv, "int", "float", "<unknown>"),
        (OpKind.Mod, "int", "float", "<unknown>"),
        (OpKind.Pow, "int", "float", "<unknown>"),
        (OpKind.AddWrap, "int", "<unknown>", "<unknown>"),
        (OpKind.SubWrap, "int", "<unknown>", "<unknown>"),
        (OpKind.MulWrap, "int", "<unknown>", "<unknown>"),
        (OpKind.Band, "int", "<unknown>", "<unknown>"),
        (OpKind.Bor, "int", "<unknown>", "<unknown>"),
        (OpKind.Bxor, "int", "<unknown>", "<unknown>"),
        (OpKind.Shl, "int", "<unknown>", "<unknown>"),
        (OpKind.Shr, "int", "<unknown>", "<unknown>"),
        (OpKind.MatMul, "<unknown>", "<unknown>", "<unknown>"),
        (OpKind.Eq, "bool", "bool", "<unknown>"),
        (OpKind.Ne, "bool", "bool", "<unknown>"),
        (OpKind.Lt, "bool", "bool", "<unknown>"),
        (OpKind.Le, "bool", "bool", "<unknown>"),
        (OpKind.Gt, "bool", "bool", "<unknown>"),
        (OpKind.Ge, "bool", "bool", "<unknown>"),
        (OpKind.Is, "bool", "bool", "<unknown>"),
        (OpKind.IsNot, "bool", "bool", "<unknown>"),
        (OpKind.In, "bool", "bool", "<unknown>"),
        (OpKind.NotIn, "bool", "bool", "<unknown>"),
        (OpKind.And, "<unknown>", "<unknown>", "<unknown>"),
        (OpKind.Or, "<unknown>", "<unknown>", "<unknown>"),
        (OpKind.Neg, "<unknown>", "<unknown>", "int"),
        (OpKind.Pos, "<unknown>", "<unknown>", "int"),
        (OpKind.Not, "<unknown>", "<unknown>", "<unknown>"),
        (OpKind.Bnot, "<unknown>", "<unknown>", "int"),
    ]
}

fn test_every_operator_code_has_a_row() -> void {
    let u = Universe()
    let rows = _table()
    assert_eq(rows.len(), 32)

    for (op, on_ints, on_floats, on_one_int) in rows {
        let where = op_name(op)
        assert_eq(render(binary_result(u, op, u.t_int, u.t_int)), on_ints, where)
        assert_eq(render(binary_result(u, op, u.t_float, u.t_float)), on_floats, where)
        assert_eq(render(unary_result(u, op, u.t_int)), on_one_int, where)
    }
}

fn test_the_comparison_group_is_the_one_the_ast_names() -> void {
    let u = Universe()
    for (op, _ints, _floats, _one_int) in _table() {
        let is_bool = binary_result(u, op, u.t_str, u.t_int) == u.t_bool
        assert_eq(is_bool, op_is_comparison(op), op_name(op))
    }
}

fn test_int_widens_to_float_in_a_mixed_pair() -> void {
    let u = Universe()
    for op in [OpKind.Add, OpKind.Sub, OpKind.Mul, OpKind.FloorDiv, OpKind.Mod, OpKind.Pow] {
        let where = op_name(op)
        assert_true(binary_result(u, op, u.t_int, u.t_int) == u.t_int, where)
        assert_true(binary_result(u, op, u.t_int, u.t_float) == u.t_float, where)
        assert_true(binary_result(u, op, u.t_float, u.t_int) == u.t_float, where)
        assert_true(binary_result(u, op, u.t_float, u.t_float) == u.t_float, where)
    }
}

fn test_division_is_float_even_between_two_ints() -> void {
    let u = Universe()
    assert_true(binary_result(u, OpKind.Div, u.t_int, u.t_int) == u.t_float)
    assert_true(binary_result(u, OpKind.Div, u.t_int, u.t_float) == u.t_float)
    assert_true(binary_result(u, OpKind.FloorDiv, u.t_int, u.t_int) == u.t_int)
    assert_true(binary_result(u, OpKind.Div, u.t_str, u.t_int) == null)
}

fn test_pow_of_two_ints_stays_int() -> void {
    let u = Universe()
    assert_true(binary_result(u, OpKind.Pow, u.t_int, u.t_int) == u.t_int)
    assert_true(binary_result(u, OpKind.Pow, u.t_float, u.t_int) == u.t_float)
}

fn test_str_concatenates_and_repeats() -> void {
    let u = Universe()
    assert_true(binary_result(u, OpKind.Add, u.t_str, u.t_str) == u.t_str)
    assert_true(binary_result(u, OpKind.Mul, u.t_str, u.t_int) == u.t_str)
    assert_true(binary_result(u, OpKind.Mul, u.t_int, u.t_str) == u.t_str)
    assert_true(binary_result(u, OpKind.Mul, u.t_str, u.t_str) == null)
    assert_true(binary_result(u, OpKind.Add, u.t_str, u.t_int) == null)
    assert_true(binary_result(u, OpKind.Sub, u.t_str, u.t_str) == null)
}

fn test_bytes_concatenates_and_repeats() -> void {
    let u = Universe()
    assert_true(binary_result(u, OpKind.Add, u.t_bytes, u.t_bytes) == u.t_bytes)
    assert_true(binary_result(u, OpKind.Mul, u.t_bytes, u.t_int) == u.t_bytes)
    assert_true(binary_result(u, OpKind.Mul, u.t_int, u.t_bytes) == u.t_bytes)
    assert_true(binary_result(u, OpKind.Add, u.t_bytes, u.t_str) == null)
    assert_true(binary_result(u, OpKind.Add, u.t_str, u.t_bytes) == null)
}

fn test_list_concatenation_joins_the_element_types() -> void {
    let u = Universe()
    let ints = list_of(u, u.t_int)
    let strs = list_of(u, u.t_str)
    assert_true(binary_result(u, OpKind.Add, ints, ints) == ints)
    assert_eq(render(binary_result(u, OpKind.Add, ints, strs)), "list[int | str]")
    assert_true(binary_result(u, OpKind.Add, ints, u.t_int) == null)

    let base = TypeDecl("Base", Span.none())
    let derived = TypeDecl("Derived", Span.none())
    derived.superclass = base
    let bases = list_of(u, named(u, TyKind.Class, "Base", base))
    let deriveds = list_of(u, named(u, TyKind.Class, "Derived", derived))
    assert_true(binary_result(u, OpKind.Add, deriveds, bases) == bases)
}

fn test_list_concatenation_uses_the_join_it_is_given() -> void {
    let u = Universe()
    let joined = binary_result(
        u,
        OpKind.Add,
        list_of(u, u.t_int),
        list_of(u, u.t_str),
        _join_to_the_right
    )
    assert_true(joined == list_of(u, u.t_str))
}

fn test_a_join_past_the_cap_gives_up_and_answers_any() -> void {
    let u = Universe()
    var lefts: list[Type?] = []
    var rights: list[Type?] = []
    for index in 0..5 {
        lefts.push(named(u, TyKind.Class, f"L{index}"))
        rights.push(named(u, TyKind.Class, f"R{index}"))
    }
    let left = list_of(u, union_of(u, lefts))
    let right = list_of(u, union_of(u, rights))
    assert_true(binary_result(u, OpKind.Add, left, right) == list_of(u, u.t_any))
}

fn test_list_repetition_answers_the_list_operand() -> void {
    let u = Universe()
    let ints = list_of(u, u.t_int)
    assert_true(binary_result(u, OpKind.Mul, ints, u.t_int) == ints)
    assert_true(binary_result(u, OpKind.Mul, u.t_int, ints) == ints)
    assert_true(binary_result(u, OpKind.Mul, ints, ints) == null)
    assert_true(binary_result(u, OpKind.Mul, ints, u.t_float) == null)
}

fn test_bitwise_operators_reject_floats() -> void {
    let u = Universe()
    for op in [OpKind.Band, OpKind.Bor, OpKind.Bxor, OpKind.Shl, OpKind.Shr] {
        let where = op_name(op)
        assert_true(binary_result(u, op, u.t_int, u.t_int) == u.t_int, where)
        assert_true(binary_result(u, op, u.t_float, u.t_float) == null, where)
        assert_true(binary_result(u, op, u.t_int, u.t_float) == null, where)
        assert_true(binary_result(u, op, u.t_float, u.t_int) == null, where)
        assert_true(binary_result(u, op, u.t_bool, u.t_bool) == null, where)
    }
}

fn test_the_wrapping_forms_are_int_only() -> void {
    let u = Universe()
    for op in [OpKind.AddWrap, OpKind.SubWrap, OpKind.MulWrap] {
        let where = op_name(op)
        assert_true(binary_result(u, op, u.t_int, u.t_int) == u.t_int, where)
        assert_true(binary_result(u, op, u.t_int, u.t_float) == null, where)
        assert_true(binary_result(u, op, u.t_float, u.t_float) == null, where)
        assert_true(binary_result(u, op, u.t_str, u.t_str) == null, where)
    }
}

fn test_matmul_has_no_core_meaning() -> void {
    let u = Universe()
    assert_true(binary_result(u, OpKind.MatMul, u.t_int, u.t_int) == null)
    assert_true(binary_result(u, OpKind.MatMul, u.t_float, u.t_float) == null)
    let boxes = _class_with_dunder(u, "Matrix", "__matmul__", u.t_int)
    assert_true(binary_result(u, OpKind.MatMul, boxes, boxes) == null)
}

fn test_and_or_take_bools_and_nothing_else() -> void {
    let u = Universe()
    for op in [OpKind.And, OpKind.Or] {
        let where = op_name(op)
        assert_true(binary_result(u, op, u.t_bool, u.t_bool) == u.t_bool, where)
        assert_true(binary_result(u, op, u.t_int, u.t_bool) == null, where)
        assert_true(binary_result(u, op, u.t_bool, list_of(u, u.t_int)) == null, where)
        assert_true(binary_result(u, op, u.t_any, u.t_bool) == u.t_bool, where)
        assert_true(binary_result(u, op, u.t_bool, u.t_any) == u.t_bool, where)
        assert_true(binary_result(u, op, u.t_any, u.t_any) == u.t_bool, where)
    }
}

fn test_any_is_soft_accepted() -> void {
    let u = Universe()
    assert_true(binary_result(u, OpKind.Add, u.t_any, u.t_int) == u.t_any)
    assert_true(binary_result(u, OpKind.Add, u.t_str, u.t_any) == u.t_any)
    assert_true(binary_result(u, OpKind.Shl, u.t_any, u.t_float) == u.t_any)
    assert_true(binary_result(u, OpKind.MulWrap, u.t_any, u.t_str) == u.t_any)
    assert_true(binary_result(u, OpKind.MatMul, u.t_any, u.t_any) == u.t_any)
    assert_true(binary_result(u, OpKind.Lt, u.t_any, u.t_any) == u.t_bool)
}

fn test_a_null_operand_is_read_as_any() -> void {
    let u = Universe()
    assert_true(binary_result(u, OpKind.Add, null, u.t_int) == u.t_any)
    assert_true(binary_result(u, OpKind.Add, u.t_int, null) == u.t_any)
    assert_true(binary_result(u, OpKind.Add, null, null) == u.t_any)
    assert_true(unary_result(u, OpKind.Neg, null) == u.t_any)
    assert_true(unary_result(u, OpKind.Not, null) == u.t_bool)
}

fn test_six_operators_have_a_dunder_and_the_rest_do_not() -> void {
    let wired = {
        OpKind.Add: "__add__",
        OpKind.Sub: "__sub__",
        OpKind.Mul: "__mul__",
        OpKind.Div: "__div__",
        OpKind.Mod: "__mod__",
        OpKind.Pow: "__pow__",
    }
    for (op, _ints, _floats, _one_int) in _table() {
        assert_eq(arith_dunder(op), wired.get(op, null), op_name(op))
    }
    assert_true(arith_dunder(OpKind.FloorDiv) == null)
}

fn test_a_class_operand_falls_back_to_its_dunder() -> void {
    let u = Universe()
    let money = _class_with_dunder(u, "Money", "__add__", u.t_str)
    assert_true(binary_result(u, OpKind.Add, money, u.t_int) == u.t_str)
    # The dunder that is not defined is still an error.
    assert_true(binary_result(u, OpKind.Sub, money, u.t_int) == null)
}

fn test_only_the_left_operand_is_consulted() -> void {
    let u = Universe()
    let money = _class_with_dunder(u, "Money", "__add__", u.t_str)
    assert_true(binary_result(u, OpKind.Add, money, u.t_int) == u.t_str)
    assert_true(binary_result(u, OpKind.Add, u.t_int, money) == null)
}

fn test_a_class_cannot_overload_the_bitwise_or_wrapping_operators() -> void {
    let u = Universe()
    let bits = _class_with_dunder(u, "Bits", "__band__", u.t_int)
    assert_true(binary_result(u, OpKind.Band, bits, bits) == null)
    assert_true(binary_result(u, OpKind.Shl, bits, u.t_int) == null)
    assert_true(binary_result(u, OpKind.AddWrap, bits, bits) == null)
    assert_true(binary_result(u, OpKind.Lt, bits, bits) == u.t_bool)
}

fn test_an_unchecked_overload_answers_any() -> void {
    let u = Universe()
    let decl = TypeDecl("Pending", Span.none())
    decl.members.push(MemberInfo("__add__"))
    let pending = named(u, TyKind.Class, "Pending", decl)
    assert_true(binary_result(u, OpKind.Add, pending, u.t_int) == u.t_any)

    let field = _class_with_member(u, "Field", MemberInfo("__add__", u.t_int))
    assert_true(binary_result(u, OpKind.Add, field, u.t_int) == u.t_any)
    let bare = _class_with_dunder(u, "Bare", "__add__", null)
    assert_true(binary_result(u, OpKind.Add, bare, u.t_int) == u.t_any)
}

fn test_a_trait_or_enum_operand_falls_back_too() -> void {
    let u = Universe()
    let trait_decl = TypeDecl("Addable", Span.none())
    trait_decl.is_trait = true
    trait_decl.members.push(MemberInfo("__add__", fn_of(u, [u.t_any], u.t_str)))
    let addable = named(u, TyKind.Trait, "Addable", trait_decl)
    assert_true(binary_result(u, OpKind.Add, addable, u.t_int) == u.t_str)
    assert_true(unary_result(u, OpKind.Neg, addable) == null)

    let enum_decl = TypeDecl("Suit", Span.none())
    enum_decl.is_enum = true
    enum_decl.members.push(MemberInfo("__sub__", fn_of(u, [u.t_any], u.t_bytes)))
    enum_decl.members.push(MemberInfo("__neg__", fn_of(u, [u.t_any], u.t_bool)))
    let suit = named(u, TyKind.Enum, "Suit", enum_decl)
    assert_true(binary_result(u, OpKind.Sub, suit, u.t_int) == u.t_bytes)
    assert_true(unary_result(u, OpKind.Neg, suit) == u.t_bool)

    let module_type = named(u, TyKind.Module, "m", trait_decl)
    assert_true(binary_result(u, OpKind.Add, module_type, u.t_int) == null)
}

fn test_the_walk_takes_the_first_of_own_trait_superclass() -> void {
    let u = Universe()
    let sup = TypeDecl("Sup", Span.none())
    sup.members.push(MemberInfo("__add__", fn_of(u, [u.t_any], u.t_int)))
    let tr = TypeDecl("Tr", Span.none())
    tr.is_trait = true
    tr.members.push(MemberInfo("__add__", fn_of(u, [u.t_any], u.t_bytes)))

    let sub = TypeDecl("Sub", Span.none())
    sub.superclass = sup
    sub.traits.push(tr)
    assert_true(
        binary_result(u, OpKind.Add, named(u, TyKind.Class, "Sub", sub), u.t_int) == u.t_bytes
    )

    let own = TypeDecl("Own", Span.none())
    own.members.push(MemberInfo("__add__", fn_of(u, [u.t_any], u.t_float)))
    own.superclass = sup
    own.traits.push(tr)
    assert_true(
        binary_result(u, OpKind.Add, named(u, TyKind.Class, "Own", own), u.t_int) == u.t_float
    )
}

fn test_the_declaration_walk_stops_at_the_depth_cap() -> void {
    let u = Universe()
    assert_true(_chain_result(u, MAX_DECL_DEPTH + 1) == u.t_str)
    assert_true(_chain_result(u, MAX_DECL_DEPTH + 2) == null)

    let cyclic = TypeDecl("Cyc", Span.none())
    cyclic.superclass = cyclic
    assert_true(
        binary_result(u, OpKind.Add, named(u, TyKind.Class, "Cyc", cyclic), u.t_int) == null
    )

    let ring = TypeDecl("Ring", Span.none())
    ring.traits.push(ring)
    assert_true(binary_result(u, OpKind.Add, named(u, TyKind.Class, "Ring", ring), u.t_int) == null)
}

fn test_a_list_interned_without_an_element_still_answers() -> void {
    let u = Universe()
    let empty = u.intern(TyKind.List, null, [])
    assert_true(binary_result(u, OpKind.Add, empty, empty) == list_of(u, u.t_any))
    assert_true(binary_result(u, OpKind.Add, empty, list_of(u, u.t_int)) == list_of(u, u.t_int))
    assert_true(binary_result(u, OpKind.Add, list_of(u, u.t_int), empty) == list_of(u, u.t_int))
}

fn test_the_modules_own_summary_holds() -> void {
    let u = Universe()
    assert_true(binary_result(u, OpKind.Div, u.t_int, u.t_int) == u.t_float)
    assert_true(binary_result(u, OpKind.Band, u.t_int, u.t_float) == null)
    assert_true(unary_result(u, OpKind.Not, u.t_bool) == u.t_bool)
}

fn test_an_inherited_overload_is_found() -> void {
    let u = Universe()
    let base = TypeDecl("Base", Span.none())
    base.members.push(MemberInfo("__add__", fn_of(u, [u.t_any], u.t_str)))

    let derived = TypeDecl("Derived", Span.none())
    derived.superclass = base
    let child = named(u, TyKind.Class, "Derived", derived)
    assert_true(binary_result(u, OpKind.Add, child, u.t_int) == u.t_str)

    let addable = TypeDecl("Addable", Span.none())
    addable.is_trait = true
    addable.members.push(MemberInfo("__mul__", fn_of(u, [u.t_any], u.t_bytes)))
    let user = TypeDecl("User", Span.none())
    user.traits.push(addable)
    let uses = named(u, TyKind.Class, "User", user)
    assert_true(binary_result(u, OpKind.Mul, uses, u.t_int) == u.t_bytes)
}

fn test_a_class_with_no_declaration_or_no_dunder_has_no_result() -> void {
    let u = Universe()
    let forward = named(u, TyKind.Class, "Forward")
    assert_true(binary_result(u, OpKind.Add, forward, u.t_int) == null)
    let plain = _class_with_dunder(u, "Plain", "greet", u.t_str)
    assert_true(binary_result(u, OpKind.Add, plain, u.t_int) == null)
}

fn test_unary_minus_keeps_the_operand_type() -> void {
    let u = Universe()
    assert_true(unary_result(u, OpKind.Neg, u.t_int) == u.t_int)
    assert_true(unary_result(u, OpKind.Neg, u.t_float) == u.t_float)
    assert_true(unary_result(u, OpKind.Neg, u.t_any) == u.t_any)
    assert_true(unary_result(u, OpKind.Neg, u.t_bool) == null)
    assert_true(unary_result(u, OpKind.Neg, u.t_str) == null)
}

fn test_unary_minus_falls_back_to_its_dunder_and_unary_plus_does_not() -> void {
    let u = Universe()
    let vector = _class_with_dunder(u, "Vector", "__neg__", u.t_str)
    assert_true(unary_result(u, OpKind.Neg, vector) == u.t_str)
    # `+x` is a no-op on a number and nothing else; there is no `__pos__`.
    assert_true(unary_result(u, OpKind.Pos, vector) == null)
    assert_true(unary_result(u, OpKind.Pos, u.t_int) == u.t_int)
    assert_true(unary_result(u, OpKind.Pos, u.t_float) == u.t_float)
    assert_true(unary_result(u, OpKind.Pos, u.t_str) == null)
}

fn test_not_takes_a_bool_and_gives_a_bool() -> void {
    let u = Universe()
    assert_true(unary_result(u, OpKind.Not, u.t_bool) == u.t_bool)
    assert_true(unary_result(u, OpKind.Not, u.t_any) == u.t_bool)
    # No truthiness, so nothing else is a condition.
    assert_true(unary_result(u, OpKind.Not, u.t_int) == null)
    assert_true(unary_result(u, OpKind.Not, list_of(u, u.t_int)) == null)
    assert_true(unary_result(u, OpKind.Not, u.t_null) == null)
}

fn test_bnot_is_int_only() -> void {
    let u = Universe()
    assert_true(unary_result(u, OpKind.Bnot, u.t_int) == u.t_int)
    assert_true(unary_result(u, OpKind.Bnot, u.t_any) == u.t_any)
    assert_true(unary_result(u, OpKind.Bnot, u.t_float) == null)
    assert_true(unary_result(u, OpKind.Bnot, u.t_bool) == null)
    # And it is not overloadable.
    let bits = _class_with_dunder(u, "Bits", "__invert__", u.t_int)
    assert_true(unary_result(u, OpKind.Bnot, bits) == null)
}

fn test_the_two_tables_do_not_answer_each_others_codes() -> void {
    let u = Universe()
    for op in [OpKind.Neg, OpKind.Pos, OpKind.Not, OpKind.Bnot] {
        assert_true(binary_result(u, op, u.t_int, u.t_int) == null, op_name(op))
    }
    for op in [OpKind.Add, OpKind.Eq, OpKind.And, OpKind.Shl, OpKind.MatMul] {
        assert_true(unary_result(u, op, u.t_int) == null, op_name(op))
    }
}

fn _join_to_the_right(u: Universe, a: Type?, b: Type?) -> Type {
    return (b ?? a) ?? u.t_any
}

fn _class_with_dunder(u: Universe, name: str, dunder: str, returns: Type?) -> Type {
    return _class_with_member(u, name, MemberInfo(dunder, fn_of(u, [u.t_any], returns)))
}

fn _class_with_member(u: Universe, name: str, member: MemberInfo) -> Type {
    let decl = TypeDecl(name, Span.none())
    decl.members.push(member)
    return named(u, TyKind.Class, name, decl)
}

fn _chain_result(u: Universe, depth: int) -> Type? {
    var decls: list[TypeDecl] = []
    for level in 0..depth {
        let decl = TypeDecl(f"D{depth}_{level}", Span.none())
        decls.push(decl)
        if level > 0 { decls[level - 1].superclass = decl }
    }
    decls[depth - 1].members.push(MemberInfo("__add__", fn_of(u, [u.t_any], u.t_str)))
    return binary_result(u, OpKind.Add, named(u, TyKind.Class, f"Head{depth}", decls[0]), u.t_int)
}
