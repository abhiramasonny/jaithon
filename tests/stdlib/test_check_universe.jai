#: `jaithon.compile.check.universe` and `.union` — interning and normalisation.
#:
#: Interning is what buys back the property the C gets free from its arena:
#: two structurally equal types must be the same object, because every relation
#: the checker computes assumes identity is a comparison rather than a walk.
#:
#: The union expectations are the ones measured out of the C with
#: `jaithon check --dump-sema`, so they pin normalisation against the front end
#: this one has to agree with.
from std.test import assert_eq, assert_false, assert_true
from jaithon.compile.check.kinds import TyKind
from jaithon.compile.check.render import render
from jaithon.compile.check.ty import Type, TypeDecl
from jaithon.compile.check.union import compare_types, optional_of, union_of
from jaithon.compile.check.universe import Universe, dict_of, fn_of, generic_param
from jaithon.compile.check.universe import list_of, named, set_of, tuple_of
from jaithon.ast import Span

fn test_the_primitives_are_interned_in_c_order() -> void {
    let u = Universe()
    assert_eq(u.t_any.uid, 0)
    assert_eq(u.t_never.uid, 1)
    assert_eq(u.t_null.uid, 2)
    assert_eq(u.t_bool.uid, 3)
    assert_eq(u.t_int.uid, 4)
    assert_eq(u.t_float.uid, 5)
    assert_eq(u.t_str.uid, 6)
    assert_eq(u.t_bytes.uid, 7)
    assert_eq(u.t_range.uid, 8)
}

fn test_void_is_null_and_not_a_tenth_type() -> void {
    let u = Universe()
    assert_true(u.t_void == u.t_null)
    assert_eq(u.count(), 9)
}

fn test_equal_types_are_the_same_object() -> void {
    let u = Universe()
    assert_true(list_of(u, u.t_int) == list_of(u, u.t_int))
    assert_true(dict_of(u, u.t_str, u.t_int) == dict_of(u, u.t_str, u.t_int))
    assert_true(set_of(u, u.t_float) == set_of(u, u.t_float))
    assert_true(tuple_of(u, [u.t_int, u.t_str]) == tuple_of(u, [u.t_int, u.t_str]))
    assert_true(fn_of(u, [u.t_int], u.t_str) == fn_of(u, [u.t_int], u.t_str))
}

fn test_different_types_are_different_objects() -> void {
    let u = Universe()
    assert_false(list_of(u, u.t_int) == list_of(u, u.t_str))
    assert_false(dict_of(u, u.t_str, u.t_int) == dict_of(u, u.t_int, u.t_str))
    assert_false(fn_of(u, [u.t_int], u.t_str) == fn_of(u, [u.t_str], u.t_int))
    # Flags are part of the key: a variadic function is not the plain one.
    assert_false(fn_of(u, [u.t_int], u.t_str, 1) == fn_of(u, [u.t_int], u.t_str))
}

fn test_a_name_cannot_collide_with_a_uid() -> void {
    # The key joins its fields with a newline for this reason: a class called
    # `4` must not be the type whose first argument has uid 4.
    let u = Universe()
    let awkward = named(u, TyKind.Class, "4")
    assert_false(awkward == list_of(u, u.t_int))
    assert_eq(render(awkward), "4")
}

fn test_interning_a_named_type_twice_answers_one_object() -> void {
    let u = Universe()
    let first = named(u, TyKind.Class, "Box")
    let second = named(u, TyKind.Class, "Box")
    assert_true(first == second)
    # A trait called Box is a different type from a class called Box.
    assert_false(first == named(u, TyKind.Trait, "Box"))
}

fn test_a_declaration_is_back_patched_onto_the_type_that_already_existed() -> void {
    # A forward reference interns the type before any declaration is visible;
    # the declaration has to land on that same object when it arrives.
    let u = Universe()
    let forward = named(u, TyKind.Class, "Node")
    assert_true(forward.decl is null)

    let declaration = TypeDecl("Node", Span.none())
    let later = named(u, TyKind.Class, "Node", declaration)
    assert_true(later == forward)
    assert_false(forward.decl is null)
}

fn test_generic_parameters_intern_by_name() -> void {
    let u = Universe()
    assert_true(generic_param(u, "T") == generic_param(u, "T"))
    assert_false(generic_param(u, "T") == generic_param(u, "K"))
}

fn test_a_union_sorts_its_members() -> void {
    let u = Universe()
    assert_eq(render(union_of(u, [u.t_str, u.t_int])), "int | str")
    assert_eq(render(union_of(u, [u.t_float, u.t_bool, u.t_int])), "bool | int | float")
}

fn test_order_does_not_change_the_type() -> void {
    let u = Universe()
    assert_true(union_of(u, [u.t_str, u.t_int]) == union_of(u, [u.t_int, u.t_str]))
}

fn test_any_absorbs_and_never_is_dropped() -> void {
    let u = Universe()
    assert_true(union_of(u, [u.t_any, u.t_int]) == u.t_any)
    assert_true(union_of(u, [u.t_never, u.t_int]) == u.t_int)
    assert_true(union_of(u, [u.t_never]) == u.t_never)
    assert_true(union_of(u, []) == u.t_never)
}

fn test_members_are_deduplicated_and_a_single_member_collapses() -> void {
    let u = Universe()
    assert_true(union_of(u, [u.t_int, u.t_int]) == u.t_int)
    assert_eq(render(union_of(u, [u.t_int, u.t_str, u.t_int])), "int | str")
}

fn test_nested_unions_are_flattened() -> void {
    let u = Universe()
    let inner = union_of(u, [u.t_int, u.t_str])
    assert_eq(render(union_of(u, [inner, u.t_float])), "int | float | str")
    # And flattening makes the nested and flat spellings one type.
    assert_true(union_of(u, [inner, u.t_float]) == union_of(u, [u.t_int, u.t_str, u.t_float]))
}

fn test_a_union_with_null_renders_as_the_optional_sugar() -> void {
    let u = Universe()
    assert_eq(render(union_of(u, [u.t_null, u.t_int])), "int?")
    assert_eq(render(optional_of(u, u.t_int)), "int?")
    assert_true(optional_of(u, u.t_int) == union_of(u, [u.t_int, u.t_null]))
}

fn test_null_sorts_early_but_renders_last() -> void {
    let u = Universe()
    assert_eq(render(union_of(u, [u.t_str, u.t_null, u.t_int])), "int | str | null")
}

fn test_same_kind_members_break_ties_by_name() -> void {
    let u = Universe()
    let zebra = named(u, TyKind.Class, "Zebra")
    let apple = named(u, TyKind.Class, "Apple")
    assert_eq(render(union_of(u, [zebra, apple])), "Apple | Zebra")
}

fn test_containers_already_in_ordinal_order_are_left_alone() -> void {
    let u = Universe()
    let members: list[Type?] = [
        list_of(u, u.t_int),
        dict_of(u, u.t_str, u.t_int),
        set_of(u, u.t_int),
    ]
    assert_eq(render(union_of(u, members)), "list[int] | dict[str, int] | set[int]")
}

fn test_compare_is_a_total_order_on_the_primitives() -> void {
    let u = Universe()
    assert_true(compare_types(u.t_bool, u.t_int) < 0)
    assert_true(compare_types(u.t_int, u.t_bool) > 0)
    assert_eq(compare_types(u.t_int, u.t_int), 0)
    assert_true(compare_types(null, u.t_int) < 0)
    assert_true(compare_types(u.t_int, null) > 0)
    assert_eq(compare_types(null, null), 0)
}
