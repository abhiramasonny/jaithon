#: Numeric fusion in unannotated literals (spec §2.2, §2.5): a list or dict
#: literal mixing int and float infers a float element type, and the ints in it
#: really are converted rather than merely relabelled.
#:
#: The static type is the half of this that `assert_eq` cannot see, because
#: `[1, 2.0] == [1.0, 2.0]` is true and `hash(1) == hash(1.0)`. Lists are
#: invariant, so passing an inferred binding to a helper with a declared
#: parameter type compiles only when the inferred type matches exactly; that is
#: the oracle every case below uses. Bind to a name first and pass the name —
#: passing a literal directly lets the parameter type stamp itself onto the
#: literal and proves nothing — and never route a value through `any`, which
#: every helper would accept.
from std.test import assert_eq, assert_true, assert_len

fn wants_float_list(xs: list[float]) -> int { return xs.len() }

fn wants_int_list(xs: list[int]) -> int { return xs.len() }

fn wants_mixed_list(xs: list[int | float]) -> int { return xs.len() }

fn wants_int_str_list(xs: list[int | str]) -> int { return xs.len() }

fn test_a_mixed_list_literal_is_a_float_list() -> void {
    let xs = [1, 2.0]
    assert_eq(wants_float_list(xs), 2)
    assert_eq(xs, [1.0, 2.0])
    assert_eq(type_of(xs[0]), "float")
    assert_true(xs[0] is float)
}

fn test_the_float_may_come_anywhere() -> void {
    let leading = [2.0, 1]
    let middle = [1, 2.0, 3]
    let trailing = [1, 2, 3.0]
    assert_eq(wants_float_list(leading), 2)
    assert_eq(wants_float_list(middle), 3)
    assert_eq(wants_float_list(trailing), 3)
    assert_eq(leading, [2.0, 1.0])
    assert_eq(middle, [1.0, 2.0, 3.0])
    assert_eq(trailing, [1.0, 2.0, 3.0])
    for value in middle { assert_true(value is float) }
    for value in trailing { assert_true(value is float) }
}

fn wants_float_str_list(xs: list[float | str]) -> int { return xs.len() }

fn test_fusion_does_not_depend_on_element_order() -> void {
    let early = [1, "a", 2.0]
    let late = [1, 2.0, "a"]
    assert_eq(wants_float_str_list(early), 3)
    assert_eq(wants_float_str_list(late), 3)
    assert_eq(early, [1.0, "a", 2.0])
    assert_eq(late, [1.0, 2.0, "a"])
    assert_eq(type_of(early[0]), "float")
    assert_eq(type_of(late[0]), "float")
}

fn test_a_non_literal_int_element_widens_too() -> void {
    var n = 7
    let xs = [n, 2.0]
    assert_eq(wants_float_list(xs), 2)
    assert_eq(xs[0], 7.0)
    assert_eq(type_of(xs[0]), "float")
    assert_true(xs[0] is float)
}

fn returns_int() -> int { return 9 }

fn test_a_call_result_widens_too() -> void {
    let xs = [returns_int(), 2.0]
    assert_eq(wants_float_list(xs), 2)
    assert_eq(xs[0], 9.0)
    assert_eq(type_of(xs[0]), "float")
}

const SIDES = 5

fn test_a_negated_literal_and_folded_arithmetic_widen() -> void {
    let negated = [-1, 2.0]
    let folded = [1 + 1, 2.0]
    let constant = [SIDES, 2.0]
    assert_eq(wants_float_list(negated), 2)
    assert_eq(wants_float_list(folded), 2)
    assert_eq(wants_float_list(constant), 2)
    assert_eq(negated[0], -1.0)
    assert_eq(folded[0], 2.0)
    assert_eq(constant[0], 5.0)
    assert_eq(type_of(negated[0]), "float")
    assert_eq(type_of(folded[0]), "float")
    assert_eq(type_of(constant[0]), "float")
}

fn test_a_huge_int_rounds_when_it_fuses() -> void {
    let xs = [9007199254740993, 1.5]
    assert_eq(wants_float_list(xs), 2)
    assert_len(xs, 2)
    assert_eq(xs[0], 9007199254740992.0)
    assert_eq(type_of(xs[0]), "float")
}

fn test_a_comprehension_over_a_fused_list_is_float() -> void {
    let xs = [x for x in [1, 2.0]]
    assert_eq(wants_float_list(xs), 2)
    assert_eq(xs, [1.0, 2.0])
    assert_eq(type_of(xs[0]), "float")
}

fn wants_float_matrix(xs: list[list[float]]) -> int { return xs.len() }

fn test_a_nested_list_fuses_within_one_inner_literal() -> void {
    let xs = [[1, 2.0]]
    assert_eq(wants_float_matrix(xs), 1)
    assert_eq(xs, [[1.0, 2.0]])
    assert_eq(type_of(xs[0][0]), "float")
}

fn wants_str_float_dict(d: dict[str, float]) -> int { return d.len() }

fn test_dict_values_fuse() -> void {
    let d = {"n": 1, "f": 1.5}
    assert_eq(wants_str_float_dict(d), 2)
    assert_len(d, 2)
    assert_eq(d["n"], 1.0)
    assert_eq(type_of(d["n"]), "float")
    assert_true(d["n"] is float)
}

fn wants_any_list(xs: list[any]) -> int { return xs.len() }

fn test_an_empty_list_is_unaffected() -> void {
    let xs = []
    assert_eq(wants_any_list(xs), 0)
    assert_len(xs, 0)
}

fn test_an_all_int_list_is_unaffected() -> void {
    let xs = [1, 2, 3]
    assert_eq(wants_int_list(xs), 3)
    assert_eq(type_of(xs[0]), "int")
    assert_true(xs[0] is int)
}

fn test_an_all_float_list_is_unaffected() -> void {
    let xs = [1.0, 2.0]
    assert_eq(wants_float_list(xs), 2)
    assert_eq(type_of(xs[0]), "float")
}

fn test_a_non_numeric_mix_stays_a_union() -> void {
    let xs = [1, "a"]
    assert_eq(wants_int_str_list(xs), 2)
    assert_eq(type_of(xs[0]), "int")
    assert_true(xs[0] is int)
}

fn wants_optional_mixed_list(xs: list[int | float | null]) -> int { return xs.len() }

fn test_an_optional_int_beside_a_float_does_not_fuse() -> void {
    let maybe: int? = 3
    let xs = [maybe, 2.0]
    assert_eq(wants_optional_mixed_list(xs), 2)
    assert_eq(type_of(xs[0]), "int")
}

fn test_a_dynamic_element_does_not_fuse() -> void {
    let dyn: any = 1
    let xs = [dyn, 2.0]
    assert_eq(wants_any_list(xs), 2)
    assert_eq(type_of(xs[0]), "int")
}

fn wants_mixed_set(s: set[int | float]) -> int { return s.len() }

fn test_a_set_literal_does_not_fuse() -> void {
    let s = {1, 2.0}
    assert_eq(wants_mixed_set(s), 2)
    assert_len(s, 2)
    assert_true(1 in s)
}

fn test_a_set_keeps_its_cardinality_above_two_to_the_53() -> void {
    let s = {9007199254740993, 9007199254740992, 1.5}
    assert_len(s, 3)

    let asked: set[float] = {9007199254740993, 9007199254740992, 1.5}
    assert_len(asked, 2)
}

fn wants_mixed_key_dict(d: dict[int | float, str]) -> int { return d.len() }

fn test_dict_keys_do_not_fuse() -> void {
    let d = {1: "a", 2.0: "b"}
    assert_eq(wants_mixed_key_dict(d), 2)
    assert_len(d, 2)

    let big = {9007199254740993: "x", 9007199254740992: "y", 1.5: "z"}
    assert_len(big, 3)
}

fn wants_list_of_int_or_float_lists(xs: list[list[int] | list[float]]) -> int {
    return xs.len()
}

fn test_nested_lists_do_not_fuse_across_containers() -> void {
    let xs = [[1], [2.0]]
    assert_eq(wants_list_of_int_or_float_lists(xs), 2)
    assert_eq(xs, [[1], [2.0]])
}

fn test_a_named_int_list_still_does_not_fuse_at_a_call() -> void {
    let xs = [1, 2]
    assert_eq(wants_int_list(xs), 2)
    assert_eq(type_of(xs[0]), "int")
}

fn test_coalesce_does_not_fuse() -> void {
    let present: int? = 1
    let value = present ?? 2.0
    assert_eq(type_of(value), "int")
    assert_eq(value, 1)
}

fn test_a_tuple_does_not_fuse() -> void {
    let pair = (1, 2.0)
    assert_eq(type_of(pair[0]), "int")
    assert_eq(type_of(pair[1]), "float")
    assert_eq(pair, (1, 2.0))
}

fn test_iterating_a_mixed_tuple_does_not_fuse() -> void {
    var seen = []
    for value in (1, 2.0) { seen.push(type_of(value)) }
    assert_eq(seen, ["int", "float"])
}

fn test_concatenating_lists_does_not_fuse() -> void {
    let xs = [1] + [2.0]
    assert_eq(type_of(xs[0]), "int")
    assert_eq(xs, [1, 2.0])
}

fn test_a_ternary_element_does_not_fuse_yet() -> void {
    let pick = true
    let xs = [pick ? 1 : 2.0, 3.0]
    assert_eq(wants_mixed_list(xs), 2)
    assert_eq(type_of(xs[0]), "int")
}
