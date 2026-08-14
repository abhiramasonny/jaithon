#: `jaithon.compile.check.relate` — equality, the join and narrowing, and the
#: subtype relation they read.
#:
#: `is_subtype` is imported from `assign.jai`, where it is defined, because
#: `join` and `narrow` are the reason it exists and a test of them that did not
#: also pin its asymmetry would be testing half a relation.
#:
#: Two things are being pinned here, and they pull in opposite directions.
#:
#: The first is that these relations are **thin**. `is_subtype` reads one bit
#: off `assignable`, and `join` and `narrow` read `is_subtype`, so most of the
#: interesting behaviour is `assign.jai`'s and is tested there. What this file
#: has to catch is a wire crossed between the two: the guard bit dropped, the
#: two subtype tests in `join` swapped, `narrow`'s two directions collapsed into
#: one.
#:
#: The second is that they are **asymmetric**, in three places that look like
#: symmetry from a distance — `any` is a supertype of everything and a subtype
#: of nothing; `join` answers the wider of two related types but a union of two
#: unrelated ones; and `narrow`'s positive branch refines while its negative
#: branch only subtracts. Each has a test that would still pass if the asymmetry
#: were reversed in one direction only, and a partner test that would not.
#:
#: Assertions compare `Type` objects with `==`. That is identity, not structure
#: — `Type` defines no `__eq__` — which is the way the checker asks it.
from std.test import assert_eq, assert_false, assert_true
from jaithon.ast import Span
from jaithon.compile.check.assign import is_subtype
from jaithon.compile.check.kinds import JOIN_MAX_MEMBERS, TyKind
from jaithon.compile.check.relate import equals, join, join_all, narrow
from jaithon.compile.check.render import render
from jaithon.compile.check.ty import Type, TypeDecl
from jaithon.compile.check.union import optional_of, union_of
from jaithon.compile.check.universe import Universe, dict_of, fn_of, generic_param
from jaithon.compile.check.universe import instantiate, list_of, named, set_of
from jaithon.compile.check.universe import tuple_of

fn _shapes(u: Universe) -> tuple[Type, Type, Type] {
    let shape = TypeDecl("Shape", Span.none())
    let circle = TypeDecl("Circle", Span.none())
    let square = TypeDecl("Square", Span.none())
    circle.superclass = shape
    square.superclass = shape
    let shape_ty = named(u, TyKind.Class, "Shape", shape)
    let circle_ty = named(u, TyKind.Class, "Circle", circle)
    let square_ty = named(u, TyKind.Class, "Square", square)
    return (shape_ty, circle_ty, square_ty)
}

fn _classes(u: Universe, count: int) -> list[Type] {
    var out: list[Type] = []
    for index in 0..count { out.push(named(u, TyKind.Class, f"C{index}")) }
    return out
}

fn _union_of_all(u: Universe, members: list[Type]) -> Type {
    var widened: list[Type?] = []
    for member in members { widened.push(member) }
    return union_of(u, widened)
}

fn test_equals_is_object_identity() -> void {
    let u = Universe()
    assert_true(equals(u.t_int, u.t_int))
    assert_false(equals(u.t_int, u.t_float))
    assert_true(equals(list_of(u, u.t_int), list_of(u, u.t_int)))
    assert_true(equals(dict_of(u, u.t_str, u.t_int), dict_of(u, u.t_str, u.t_int)))
    assert_false(equals(list_of(u, u.t_int), set_of(u, u.t_int)))
}

fn test_equals_says_no_to_a_type_that_never_reached_the_intern_table() -> void {
    let u = Universe()
    let stranger = Type(TyKind.Int)
    assert_false(equals(stranger, u.t_int))
    assert_true(equals(stranger, stranger))
}

fn test_equals_of_two_missing_types() -> void {
    let u = Universe()
    assert_true(equals(null, null))
    assert_false(equals(null, u.t_int))
    assert_false(equals(u.t_int, null))
}

fn test_subtyping_is_reflexive_and_has_a_top_and_a_bottom() -> void {
    let u = Universe()
    for t in [u.t_int, u.t_str, u.t_null, u.t_any, u.t_never, list_of(u, u.t_int)] {
        assert_true(is_subtype(u, t, t), render(t))
        assert_true(is_subtype(u, t, u.t_any), render(t))
        assert_true(is_subtype(u, u.t_never, t), render(t))
    }
    assert_false(is_subtype(u, u.t_int, u.t_str))
}

fn test_any_is_a_supertype_of_everything_and_a_subtype_of_nothing() -> void {
    let u = Universe()
    assert_true(is_subtype(u, u.t_int, u.t_any))
    assert_false(is_subtype(u, u.t_any, u.t_int))
    assert_false(is_subtype(u, u.t_any, list_of(u, u.t_str)))
    assert_true(is_subtype(u, u.t_any, u.t_any))
}

fn test_a_generic_parameter_is_a_supertype_and_not_a_subtype() -> void {
    let u = Universe()
    let param = generic_param(u, "T")
    assert_true(is_subtype(u, u.t_int, param))
    assert_false(is_subtype(u, param, u.t_int))
}

fn test_a_subclass_is_a_subtype_and_a_sibling_is_not() -> void {
    let u = Universe()
    let (shape, circle, square) = _shapes(u)
    assert_true(is_subtype(u, circle, shape))
    assert_false(is_subtype(u, shape, circle))
    assert_false(is_subtype(u, circle, square))
}

fn test_a_member_is_a_subtype_of_its_union() -> void {
    let u = Universe()
    let maybe = optional_of(u, u.t_int)
    assert_true(is_subtype(u, u.t_int, maybe))
    assert_true(is_subtype(u, u.t_null, maybe))
    assert_false(is_subtype(u, maybe, u.t_int))
    assert_false(is_subtype(u, u.t_str, maybe))
    let wider = union_of(u, [u.t_int, u.t_str, u.t_null])
    assert_true(is_subtype(u, maybe, wider))
    assert_false(is_subtype(u, wider, maybe))
}

fn test_a_missing_type_is_a_subtype_of_anything_and_vice_versa() -> void {
    let u = Universe()
    assert_true(is_subtype(u, null, u.t_int))
    assert_true(is_subtype(u, u.t_int, null))
    assert_true(is_subtype(u, null, null))
}

fn test_join_of_a_type_with_itself_is_that_type() -> void {
    let u = Universe()
    for t in [u.t_int, u.t_any, u.t_never, list_of(u, u.t_str), optional_of(u, u.t_int)] {
        assert_true(join(u, t, t) == t, render(t))
    }
}

fn test_never_is_the_identity_of_the_join_and_any_absorbs() -> void {
    let u = Universe()
    assert_true(join(u, u.t_never, u.t_str) == u.t_str)
    assert_true(join(u, u.t_str, u.t_never) == u.t_str)
    assert_true(join(u, u.t_any, u.t_str) == u.t_any)
    assert_true(join(u, u.t_str, u.t_any) == u.t_any)
    assert_true(join(u, u.t_never, u.t_any) == u.t_any)
}

fn test_join_of_a_missing_type_answers_the_other_one() -> void {
    let u = Universe()
    assert_true(join(u, null, u.t_int) == u.t_int)
    assert_true(join(u, u.t_int, null) == u.t_int)
    assert_true(join(u, null, null) == u.t_any)
}

fn test_join_of_two_unrelated_types_is_their_union() -> void {
    let u = Universe()
    assert_eq(render(join(u, u.t_int, u.t_str)), "int | str")
    assert_eq(render(join(u, u.t_int, u.t_null)), "int?")
    assert_eq(render(join(u, list_of(u, u.t_int), u.t_str)), "str | list[int]")
}

fn test_join_of_two_siblings_is_their_union_and_not_their_ancestor() -> void {
    let u = Universe()
    let (shape, circle, square) = _shapes(u)
    assert_eq(render(join(u, circle, square)), "Circle | Square")
    assert_false(join(u, circle, square) == shape)
    assert_true(join(u, circle, shape) == shape)
    assert_true(join(u, shape, circle) == shape)
}

fn test_join_of_a_type_and_a_union_containing_it_is_the_union() -> void {
    let u = Universe()
    let maybe = optional_of(u, u.t_int)
    assert_true(join(u, u.t_int, maybe) == maybe)
    assert_true(join(u, maybe, u.t_int) == maybe)
    assert_eq(render(join(u, maybe, u.t_str)), "int | str | null")
}

fn test_join_merges_two_unions() -> void {
    let u = Universe()
    let left = union_of(u, [u.t_int, u.t_str])
    let right = union_of(u, [u.t_str, u.t_bool])
    assert_eq(render(join(u, left, right)), "bool | int | str")
}

fn test_a_join_past_the_cap_gives_up_and_answers_any() -> void {
    let u = Universe()
    let seven = _union_of_all(u, _classes(u, 7))
    let widened = join(u, seven, named(u, TyKind.Class, "C7"))
    assert_eq(widened.arg_count(), JOIN_MAX_MEMBERS)
    assert_true(join(u, widened, named(u, TyKind.Class, "C8")) == u.t_any)
}

fn test_a_union_at_the_cap_still_joins_with_itself() -> void {
    let u = Universe()
    let eight = _union_of_all(u, _classes(u, 8))
    assert_eq(eight.arg_count(), JOIN_MAX_MEMBERS)
    assert_true(join(u, eight, eight) == eight)
    assert_true(join(u, eight, u.t_never) == eight)
    assert_true(join(u, u.t_never, eight) == eight)
}

fn test_join_all_folds_from_never() -> void {
    let u = Universe()
    assert_eq(render(join_all(u, [u.t_int, u.t_int])), "int")
    assert_eq(render(join_all(u, [u.t_bool, u.t_int, u.t_str])), "bool | int | str")
    assert_true(join_all(u, [u.t_int]) == u.t_int)
    let pair = tuple_of(u, [u.t_str, u.t_int])
    assert_eq(render(join_all(u, pair.args)), "int | str")
}

fn test_join_all_of_nothing_is_never() -> void {
    let u = Universe()
    let nothing: list[Type] = []
    assert_true(join_all(u, nothing) == u.t_never)
}

fn test_join_all_folds_rather_than_collecting() -> void {
    let u = Universe()
    assert_true(join_all(u, _classes(u, 9)) == u.t_any)
    assert_eq(_union_of_all(u, _classes(u, 9)).arg_count(), 9)

    let eight = _union_of_all(u, _classes(u, 8))
    assert_true(join_all(u, [eight]) == eight)
}

fn test_narrow_keeps_the_members_the_test_can_succeed_for() -> void {
    let u = Universe()
    let mixed = union_of(u, [u.t_int, u.t_str, u.t_null])
    assert_eq(render(narrow(u, mixed, u.t_str, true)), "str")
    assert_eq(render(narrow(u, mixed, u.t_null, true)), "null")
    assert_eq(render(narrow(u, mixed, u.t_bool, true)), "never")
}

fn test_narrow_subtracts_only_definite_subtypes() -> void {
    let u = Universe()
    let maybe = optional_of(u, u.t_int)
    assert_eq(render(narrow(u, maybe, u.t_null, false)), "int")
    assert_eq(render(narrow(u, maybe, u.t_str, false)), "int?")

    let (shape, circle, square) = _shapes(u)
    let either = union_of(u, [circle, square])
    assert_eq(render(narrow(u, either, shape, false)), "never")
    let wider = union_of(u, [shape, square])
    assert_eq(render(narrow(u, wider, circle, false)), "Shape | Square")
}

fn test_narrow_downcasts_a_wider_member() -> void {
    let u = Universe()
    let (shape, circle, _square) = _shapes(u)
    assert_true(narrow(u, shape, circle, true) == circle)
    assert_true(narrow(u, shape, circle, false) == shape)
}

fn test_narrow_keeps_the_member_when_each_is_a_subtype_of_the_other() -> void {
    let u = Universe()
    let decl = TypeDecl("Box", Span.none())
    let box = named(u, TyKind.Class, "Box", decl)
    let box_int = instantiate(u, box, [u.t_int])
    let box_str = instantiate(u, box, [u.t_str])
    assert_false(equals(box_int, box_str))
    assert_true(is_subtype(u, box_int, box_str))
    assert_true(is_subtype(u, box_str, box_int))
    assert_true(narrow(u, box_int, box_str, true) == box_int)

    let first = generic_param(u, "T")
    let second = generic_param(u, "K")
    assert_false(equals(first, second))
    assert_true(is_subtype(u, first, second))
    assert_true(is_subtype(u, second, first))
    assert_true(narrow(u, first, second, true) == first)
}

fn test_narrow_downcasts_a_dynamic_binding() -> void {
    let u = Universe()
    let (_shape, circle, _square) = _shapes(u)
    assert_true(narrow(u, u.t_any, circle, true) == circle)
    assert_true(narrow(u, u.t_any, u.t_str, true) == u.t_str)
    assert_true(narrow(u, u.t_any, circle, false) == u.t_any)
}

fn test_narrow_of_a_single_type_that_is_already_specific_enough() -> void {
    let u = Universe()
    let (shape, circle, square) = _shapes(u)
    assert_true(narrow(u, circle, shape, true) == circle)
    assert_true(narrow(u, circle, shape, false) == u.t_never)
    assert_true(narrow(u, circle, square, true) == u.t_never)
    assert_true(narrow(u, circle, square, false) == circle)
}

fn test_narrow_answers_a_missing_type_only_for_a_missing_type() -> void {
    let u = Universe()
    assert_true(narrow(u, null, u.t_int, true) is null)
    assert_true(narrow(u, null, u.t_int, false) is null)
    assert_true(narrow(u, u.t_int, null, true) == u.t_int)
    assert_true(narrow(u, u.t_int, null, false) == u.t_int)
}

fn test_narrow_leaves_a_union_alone_when_every_member_survives() -> void {
    let u = Universe()
    let (shape, circle, square) = _shapes(u)
    let either = union_of(u, [circle, square])
    assert_true(narrow(u, either, shape, true) == either)
    assert_true(narrow(u, either, u.t_str, false) == either)
}

fn test_a_structural_type_narrows_by_the_same_two_rules() -> void {
    let u = Universe()
    let ints = list_of(u, u.t_int)
    let strs = list_of(u, u.t_str)
    let either = union_of(u, [ints, strs])
    assert_true(narrow(u, either, ints, true) == ints)
    assert_true(narrow(u, either, ints, false) == strs)
    assert_true(narrow(u, list_of(u, u.t_any), ints, true) == ints)
    let signature = fn_of(u, [u.t_int], u.t_str)
    assert_true(narrow(u, signature, u.t_any, true) == signature)
}

fn test_the_modules_own_summary_holds() -> void {
    let u = Universe()
    assert_true(equals(u.t_int, u.t_int))
    assert_true(is_subtype(u, u.t_int, u.t_any))
    assert_false(is_subtype(u, u.t_any, u.t_int))
    assert_true(join(u, u.t_int, u.t_never) == u.t_int)
}
