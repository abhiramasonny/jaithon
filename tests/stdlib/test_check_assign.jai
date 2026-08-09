#: `jaithon.compile.check.assign` — assignability and widening.
#:
#: Every assertion is about the *pair*, because the second half of it is what
#: the code generator reads: an accepted assignment that answers `GUARDED` puts
#: an `OP_TYPE_GUARD` into the bytecode and one that answers `FREE` does not.
#: A test that only asked whether the assignment was accepted would pass on both
#: sides of that difference, which is the difference this file exists to pin.
#:
#: The cases worth their length are the ones where the guard is nearly right:
#: `never` reaching `int` is free while `any` reaching `int` is guarded, a
#: generic parameter guards at the top level and never as a container element,
#: a union target prefers the member that fits without a guard even when the
#: guarded member sorts first, and `int` reaching `float` is accepted at the top
#: level while `list[int]` reaching `list[float]` is not accepted at all.
from std.test import assert_eq, assert_false, assert_true
from jaithon.ast import Span
from jaithon.compile.check.assign import assignable, is_assignable, is_subtype
from jaithon.compile.check.assign import widen_target
from jaithon.compile.check.kinds import FN_VARIADIC, FN_KWREST, TyKind
from jaithon.compile.check.render import render
from jaithon.compile.check.ty import Type, TypeDecl
from jaithon.compile.check.union import optional_of, union_of
from jaithon.compile.check.universe import Universe, dict_of, fn_of, generic_param
from jaithon.compile.check.universe import list_of, named, set_of, tuple_of

# The three answers `assignable` has, named so that an assertion says which one
# it is about rather than leaving a reader to decode two bools.
let FREE: (bool, bool) = (true, false)
let GUARDED: (bool, bool) = (true, true)
let REJECTED: (bool, bool) = (false, false)

fn _decl(name: str) -> TypeDecl {
    return TypeDecl(name, Span.none())
}

# `Animal` and its subclass `Dog`, as class types of `u`.
fn _animals(u: Universe) -> (Type, Type) {
    let animal = _decl("Animal")
    let dog = _decl("Dog")
    dog.superclass = animal
    return (named(u, TyKind.Class, "Animal", animal), named(u, TyKind.Class, "Dog", dog))
}

fn test_a_null_operand_accepts_without_a_guard() -> void {
    # An error was already reported upstream, and a second one about the same
    # mistake helps nobody.
    let u = Universe()
    assert_eq(assignable(u, null, u.t_int), FREE)
    assert_eq(assignable(u, u.t_int, null), FREE)
    assert_eq(assignable(u, null, null), FREE)
}

fn test_a_null_operand_reaches_the_two_derived_answers_too() -> void {
    # The rule above is the first line of the recursion, so both answers built
    # on it inherit it — including `is_subtype`, which `relate.jai` reads. The
    # test above pins `assignable` alone and would not notice a derived answer
    # that special-cased null the other way.
    let u = Universe()
    assert_true(is_assignable(u, null, u.t_int))
    assert_true(is_assignable(u, u.t_int, null))
    assert_true(is_subtype(u, null, u.t_int))
    assert_true(is_subtype(u, u.t_int, null))
    assert_true(is_subtype(u, null, null))
}

fn test_identity_accepts_without_a_guard() -> void {
    let u = Universe()
    assert_eq(assignable(u, u.t_int, u.t_int), FREE)
    assert_eq(assignable(u, u.t_any, u.t_any), FREE)
    assert_eq(assignable(u, u.t_never, u.t_never), FREE)
    assert_eq(assignable(u, list_of(u, u.t_int), list_of(u, u.t_int)), FREE)
    let param = generic_param(u, "T")
    assert_eq(assignable(u, param, param), FREE)
}

fn test_anything_reaches_any_without_a_guard() -> void {
    let u = Universe()
    assert_eq(assignable(u, u.t_int, u.t_any), FREE)
    assert_eq(assignable(u, list_of(u, u.t_str), u.t_any), FREE)
    assert_eq(assignable(u, u.t_never, u.t_any), FREE)
    assert_eq(assignable(u, generic_param(u, "T"), u.t_any), FREE)
    assert_eq(assignable(u, union_of(u, [u.t_int, u.t_str]), u.t_any), FREE)
}

fn test_never_reaches_anything_without_a_guard() -> void {
    # The bottom type inhabits nothing, so no run-time check could fail and
    # none is emitted. This is the test `any` answers differently.
    let u = Universe()
    let (animal, _dog) = _animals(u)
    assert_eq(assignable(u, u.t_never, u.t_int), FREE)
    assert_eq(assignable(u, u.t_never, list_of(u, u.t_str)), FREE)
    assert_eq(assignable(u, u.t_never, animal), FREE)
    assert_eq(assignable(u, u.t_never, generic_param(u, "T")), FREE)
}

fn test_any_reaches_a_concrete_type_behind_a_guard() -> void {
    let u = Universe()
    let (animal, _dog) = _animals(u)
    assert_eq(assignable(u, u.t_any, u.t_int), GUARDED)
    assert_eq(assignable(u, u.t_any, list_of(u, u.t_str)), GUARDED)
    assert_eq(assignable(u, u.t_any, animal), GUARDED)
    assert_eq(assignable(u, u.t_any, union_of(u, [u.t_int, u.t_str])), GUARDED)
    # `any` answers before the generic-parameter target does, so this is the
    # dynamic boundary and not the free acceptance a `T` target usually gives.
    assert_eq(assignable(u, u.t_any, generic_param(u, "T")), GUARDED)
}

fn test_a_generic_parameter_target_accepts_without_a_guard() -> void {
    # Generics are erased, so `T` names no run-time shape: without this every
    # call to a generic function would be a type error.
    let u = Universe()
    let param = generic_param(u, "T")
    let (_animal, dog) = _animals(u)
    assert_eq(assignable(u, u.t_int, param), FREE)
    assert_eq(assignable(u, list_of(u, u.t_str), param), FREE)
    assert_eq(assignable(u, dog, param), FREE)
}

fn test_a_generic_parameter_source_accepts_behind_a_guard() -> void {
    # The mirror of the rule above: reaching a concrete type from an erased one
    # is the same boundary `any` crosses, and it is checked at run time.
    let u = Universe()
    let param = generic_param(u, "T")
    assert_eq(assignable(u, param, u.t_int), GUARDED)
    assert_eq(assignable(u, param, list_of(u, u.t_str)), GUARDED)
    assert_eq(assignable(u, param, union_of(u, [u.t_int, u.t_str])), GUARDED)
}

fn test_a_generic_parameter_reaching_its_own_optional_is_free() -> void {
    # Why the two generic-parameter tests sit on opposite sides of the union
    # rules. `T` reaching `T | null` is `T` reaching `T`, which costs nothing;
    # testing the source before the unions were taken apart would report a
    # dynamic conversion on every optional generic slot.
    let u = Universe()
    let param = generic_param(u, "T")
    assert_eq(assignable(u, param, optional_of(u, param)), FREE)
    assert_eq(assignable(u, param, union_of(u, [param, u.t_int])), FREE)
}

fn test_a_free_member_of_a_union_target_wins_over_a_guarded_one() -> void {
    let u = Universe()
    let param = generic_param(u, "T")
    let target = union_of(u, [param, u.t_int])
    # Members are sorted by kind ordinal, so the guarded member is tried first
    # and the answer must still be free. Nothing a caller writes changes that
    # order, which is why the preference has to live in the rule.
    assert_eq(render(target), "int | T")
    assert_eq(assignable(u, param, target), FREE)

    # And a target whose members all need a guard reports one.
    assert_eq(assignable(u, param, union_of(u, [u.t_int, u.t_str])), GUARDED)
}

fn test_null_reaches_an_optional_and_nothing_else() -> void {
    # `T?` is `T | null`, so this is the union rule and not a case of its own.
    let u = Universe()
    assert_eq(assignable(u, u.t_null, optional_of(u, u.t_int)), FREE)
    assert_eq(assignable(u, u.t_null, u.t_any), FREE)
    assert_eq(assignable(u, u.t_null, u.t_int), REJECTED)
    assert_eq(assignable(u, u.t_null, union_of(u, [u.t_int, u.t_str])), REJECTED)
}

fn test_every_member_of_a_union_source_must_fit() -> void {
    let u = Universe()
    let both = union_of(u, [u.t_int, u.t_str])
    assert_eq(assignable(u, both, both), FREE)
    assert_eq(assignable(u, both, union_of(u, [u.t_int, u.t_str, u.t_float])), FREE)
    assert_eq(assignable(u, both, u.t_int), REJECTED)
    assert_eq(assignable(u, optional_of(u, u.t_int), u.t_int), REJECTED)
}

fn test_one_guarded_member_guards_the_whole_union_source() -> void {
    let u = Universe()
    let param = generic_param(u, "T")
    # `int` fits for free and `T` only behind a check, so the assignment as a
    # whole needs the check.
    assert_eq(assignable(u, union_of(u, [param, u.t_int]), u.t_int), GUARDED)
}

fn test_containers_are_invariant() -> void {
    # These are mutable: if `list[Dog]` reached `list[Animal]` the callee could
    # push a cat through the widened reference.
    let u = Universe()
    let (animal, dog) = _animals(u)
    assert_eq(assignable(u, list_of(u, dog), list_of(u, animal)), REJECTED)
    assert_eq(assignable(u, list_of(u, animal), list_of(u, dog)), REJECTED)
    assert_eq(assignable(u, list_of(u, u.t_int), list_of(u, u.t_float)), REJECTED)
    assert_eq(assignable(u, set_of(u, dog), set_of(u, animal)), REJECTED)
    assert_eq(assignable(u, dict_of(u, u.t_str, dog), dict_of(u, u.t_str, animal)), REJECTED)
    # The kind is part of it too, and so is the arity a kind implies.
    assert_eq(assignable(u, list_of(u, u.t_int), set_of(u, u.t_int)), REJECTED)
    assert_eq(assignable(u, list_of(u, u.t_int), u.t_int), REJECTED)
}

fn test_an_any_element_crosses_into_a_concrete_one_behind_a_guard() -> void {
    # What an untyped comprehension produces has to be able to reach a typed
    # slot, and `any` is the one type that converts both ways.
    let u = Universe()
    assert_eq(assignable(u, list_of(u, u.t_any), list_of(u, u.t_str)), GUARDED)
    assert_eq(assignable(u, set_of(u, u.t_any), set_of(u, u.t_int)), GUARDED)
    assert_eq(assignable(u, dict_of(u, u.t_any, u.t_any), dict_of(u, u.t_str, u.t_int)), GUARDED)
    # One dynamic element is enough to need the check, and the other position
    # being equal does not excuse it.
    assert_eq(assignable(u, dict_of(u, u.t_str, u.t_any), dict_of(u, u.t_str, u.t_int)), GUARDED)
}

fn test_a_concrete_element_widens_to_any_for_free() -> void {
    let u = Universe()
    assert_eq(assignable(u, list_of(u, u.t_str), list_of(u, u.t_any)), FREE)
    assert_eq(assignable(u, dict_of(u, u.t_str, u.t_int), dict_of(u, u.t_any, u.t_any)), FREE)
}

fn test_a_generic_parameter_element_never_guards() -> void {
    # The one place the top-level rule and the element rule disagree: an
    # element position guards for `any` reaching something concrete and for
    # nothing else, so an erased element passes free in both directions.
    let u = Universe()
    let param = generic_param(u, "T")
    assert_eq(assignable(u, param, u.t_int), GUARDED)
    assert_eq(assignable(u, list_of(u, param), list_of(u, u.t_int)), FREE)
    assert_eq(assignable(u, list_of(u, u.t_int), list_of(u, param)), FREE)
    assert_eq(assignable(u, dict_of(u, param, param), dict_of(u, u.t_str, u.t_int)), FREE)
}

fn test_never_is_free_at_the_top_level_and_stops_at_an_element() -> void {
    # `never` is not erased, so an invariant element position does not let it
    # past — the one place the bottom type and the top-level rule disagree.
    # `_is_erased` naming `any` and a generic parameter and nothing else is what
    # buys that, and without this case the two spellings are indistinguishable.
    let u = Universe()
    assert_eq(assignable(u, u.t_never, u.t_int), FREE)
    assert_eq(assignable(u, list_of(u, u.t_never), list_of(u, u.t_int)), REJECTED)
    assert_eq(assignable(u, set_of(u, u.t_never), set_of(u, u.t_int)), REJECTED)
    # The wanted element being erased is a different rule and still applies.
    assert_eq(assignable(u, list_of(u, u.t_never), list_of(u, u.t_any)), FREE)
    # A tuple is covariant, so there the recursion does reach the top-level
    # rule and `never` passes.
    assert_eq(assignable(u, tuple_of(u, [u.t_never]), tuple_of(u, [u.t_int])), FREE)
}

fn test_a_structural_element_recurses_and_a_nominal_one_does_not() -> void {
    let u = Universe()
    let (animal, dog) = _animals(u)
    let any_pair = tuple_of(u, [u.t_any, u.t_any])
    let typed_pair = tuple_of(u, [u.t_int, u.t_str])
    # `list[(any, any)]` is what an untyped comprehension of pairs produces,
    # and it has to reach `list[(K, V)]`.
    assert_eq(assignable(u, list_of(u, any_pair), list_of(u, typed_pair)), GUARDED)
    assert_eq(
        assignable(u, list_of(u, list_of(u, u.t_any)), list_of(u, list_of(u, u.t_int))),
        GUARDED
    )
    assert_eq(
        assignable(
            u,
            list_of(u, fn_of(u, [u.t_int], u.t_any)),
            list_of(u, fn_of(u, [u.t_int], u.t_str))
        ),
        GUARDED
    )
    # A nominal element is where the recursion stops, at every depth.
    assert_eq(assignable(u, list_of(u, dog), list_of(u, animal)), REJECTED)
    assert_eq(assignable(u, list_of(u, list_of(u, dog)), list_of(u, list_of(u, animal))), REJECTED)
    # Reached through a structural element, though, nominal covariance applies:
    # the recursion asks the same question, and a tuple is covariant.
    assert_eq(
        assignable(u, list_of(u, tuple_of(u, [dog])), list_of(u, tuple_of(u, [animal]))),
        FREE
    )
    # Two structural elements of different kinds do not recurse at all.
    assert_eq(
        assignable(u, list_of(u, list_of(u, u.t_str)), list_of(u, set_of(u, u.t_str))),
        REJECTED
    )
}

fn test_tuples_are_covariant_element_wise() -> void {
    # Immutable, so there is no widened reference to push anything through.
    let u = Universe()
    let (animal, dog) = _animals(u)
    assert_eq(assignable(u, tuple_of(u, [dog, u.t_int]), tuple_of(u, [animal, u.t_int])), FREE)
    assert_eq(assignable(u, tuple_of(u, [animal, u.t_int]), tuple_of(u, [dog, u.t_int])), REJECTED)
    assert_eq(
        assignable(u, tuple_of(u, [u.t_any, u.t_int]), tuple_of(u, [u.t_str, u.t_int])),
        GUARDED
    )
    assert_eq(assignable(u, tuple_of(u, [u.t_int]), tuple_of(u, [u.t_int, u.t_int])), REJECTED)
    assert_eq(assignable(u, tuple_of(u, [u.t_int]), list_of(u, u.t_int)), REJECTED)
}

fn test_functions_are_contravariant_in_their_parameters() -> void {
    # The target's caller passes the target's argument types, so the source has
    # to accept them: a function that takes any animal can stand in where one
    # that takes a dog was wanted, and not the other way round.
    let u = Universe()
    let (animal, dog) = _animals(u)
    let takes_animal = fn_of(u, [animal], u.t_void)
    let takes_dog = fn_of(u, [dog], u.t_void)
    assert_eq(assignable(u, takes_animal, takes_dog), FREE)
    assert_eq(assignable(u, takes_dog, takes_animal), REJECTED)
}

fn test_functions_are_covariant_in_their_return_type() -> void {
    let u = Universe()
    let (animal, dog) = _animals(u)
    assert_eq(assignable(u, fn_of(u, [], dog), fn_of(u, [], animal)), FREE)
    assert_eq(assignable(u, fn_of(u, [], animal), fn_of(u, [], dog)), REJECTED)
}

fn test_a_function_needs_equal_arity_and_equal_flags() -> void {
    # The flags change how a call is assembled rather than what it means: a
    # variadic reads one list where a fixed function reads two values, so no
    # relation between the parameter types could stand in for this test.
    let u = Universe()
    assert_eq(
        assignable(u, fn_of(u, [u.t_int], u.t_void), fn_of(u, [u.t_int, u.t_int], u.t_void)),
        REJECTED
    )
    let variadic = fn_of(u, [list_of(u, u.t_int)], u.t_void, FN_VARIADIC)
    let fixed = fn_of(u, [list_of(u, u.t_int)], u.t_void)
    let kwrest = fn_of(u, [list_of(u, u.t_int)], u.t_void, FN_KWREST)
    assert_eq(assignable(u, variadic, fixed), REJECTED)
    assert_eq(assignable(u, fixed, variadic), REJECTED)
    assert_eq(assignable(u, variadic, kwrest), REJECTED)
    assert_eq(assignable(u, variadic, variadic), FREE)
}

fn test_a_function_guards_through_a_parameter_or_a_return() -> void {
    let u = Universe()
    # Contravariance puts the dynamic side on the target's parameter: the stored
    # function will be handed an `any` where it declared an `int`.
    assert_eq(assignable(u, fn_of(u, [u.t_int], u.t_void), fn_of(u, [u.t_any], u.t_void)), GUARDED)
    # And the other direction costs nothing, because `int` reaching `any` is
    # free wherever it appears.
    assert_eq(assignable(u, fn_of(u, [u.t_any], u.t_void), fn_of(u, [u.t_int], u.t_void)), FREE)
    assert_eq(assignable(u, fn_of(u, [], u.t_any), fn_of(u, [], u.t_int)), GUARDED)
    assert_eq(assignable(u, fn_of(u, [], u.t_int), fn_of(u, [], u.t_any)), FREE)
}

fn test_a_class_reaches_its_superclass_and_not_the_other_way() -> void {
    let u = Universe()
    let (animal, dog) = _animals(u)
    let rock = named(u, TyKind.Class, "Rock", _decl("Rock"))
    assert_eq(assignable(u, dog, animal), FREE)
    assert_eq(assignable(u, animal, dog), REJECTED)
    assert_eq(assignable(u, dog, rock), REJECTED)
    assert_eq(assignable(u, dog, u.t_int), REJECTED)
}

fn test_a_class_reaches_a_trait_it_satisfies() -> void {
    let u = Universe()
    let comparable = _decl("Comparable")
    let money = _decl("Money")
    money.traits.push(comparable)
    assert_eq(
        assignable(
            u,
            named(u, TyKind.Class, "Money", money),
            named(u, TyKind.Trait, "Comparable", comparable)
        ),
        FREE
    )
    assert_eq(
        assignable(
            u,
            named(u, TyKind.Class, "Money", money),
            named(u, TyKind.Trait, "Printable", _decl("Printable"))
        ),
        REJECTED
    )
}

fn test_a_trait_this_unit_cannot_see_is_matched_by_name() -> void {
    # It arrived as a runtime object, so there is no declaration to compare —
    # only the name the class wrote in its header, which is still narrower than
    # letting every class satisfy it.
    let u = Universe()
    let widget = _decl("Widget")
    widget.opaque_traits.push("Drawable")
    let widget_type = named(u, TyKind.Class, "Widget", widget)
    assert_eq(assignable(u, widget_type, named(u, TyKind.Trait, "Drawable")), FREE)
    assert_eq(assignable(u, widget_type, named(u, TyKind.Trait, "Printable")), REJECTED)
}

fn test_an_enum_reaches_a_trait_but_not_an_unrelated_class() -> void {
    let u = Universe()
    let serialisable = _decl("Serialisable")
    let colour = _decl("Colour")
    colour.is_enum = true
    colour.traits.push(serialisable)
    let colour_type = named(u, TyKind.Enum, "Colour", colour)
    assert_eq(
        assignable(u, colour_type, named(u, TyKind.Trait, "Serialisable", serialisable)),
        FREE
    )
    let (animal, _dog) = _animals(u)
    assert_eq(assignable(u, colour_type, animal), REJECTED)
    assert_eq(assignable(u, colour_type, named(u, TyKind.Enum, "Suit", _decl("Suit"))), REJECTED)
}

fn test_a_trait_reaches_only_the_traits_it_extends() -> void {
    let u = Universe()
    let comparable = _decl("Comparable")
    let ordered = _decl("Ordered")
    ordered.traits.push(comparable)
    let ordered_type = named(u, TyKind.Trait, "Ordered", ordered)
    assert_eq(assignable(u, ordered_type, named(u, TyKind.Trait, "Comparable", comparable)), FREE)
    assert_eq(
        assignable(u, named(u, TyKind.Trait, "Comparable", comparable), ordered_type),
        REJECTED
    )
    # A trait-typed value names no class, so no class type can receive it.
    let (animal, _dog) = _animals(u)
    assert_eq(assignable(u, ordered_type, animal), REJECTED)
}

fn test_a_trait_source_reaches_an_opaque_trait_by_name_as_well() -> void {
    # The other half of the by-name rule. In the C the choice between the
    # declaration and the name is a ternary written out twice — once for a class
    # or enum reaching a trait, once for a trait reaching one — and only the
    # first copy is pinned above. This port routes both through
    # `satisfies_trait`, so replacing this branch with a declaration-only
    # `implements_trait` would still pass every other assertion in this file.
    let u = Universe()
    let styled = _decl("Styled")
    styled.is_trait = true
    styled.opaque_traits.push("Drawable")
    let styled_type = named(u, TyKind.Trait, "Styled", styled)
    assert_eq(assignable(u, styled_type, named(u, TyKind.Trait, "Drawable")), FREE)
    assert_eq(assignable(u, styled_type, named(u, TyKind.Trait, "Printable")), REJECTED)

    # And the name is found through the traits a declaration names, not only in
    # its own list: `Panel` never wrote `Drawable` itself.
    let panel = _decl("Panel")
    panel.traits.push(styled)
    assert_eq(
        assignable(u, named(u, TyKind.Class, "Panel", panel), named(u, TyKind.Trait, "Drawable")),
        FREE
    )
    # An enum reaches one too, through the arm it shares with a class.
    let shade = _decl("Shade")
    shade.is_enum = true
    shade.opaque_traits.push("Drawable")
    assert_eq(
        assignable(u, named(u, TyKind.Enum, "Shade", shade), named(u, TyKind.Trait, "Drawable")),
        FREE
    )
}

fn test_int_widens_to_float_at_the_top_level() -> void {
    let u = Universe()
    assert_eq(assignable(u, u.t_int, u.t_float), GUARDED)
    assert_eq(assignable(u, u.t_int, optional_of(u, u.t_float)), GUARDED)
    assert_eq(assignable(u, u.t_int, union_of(u, [u.t_float, u.t_str])), GUARDED)
    # The other direction loses the fraction and stays an error.
    assert_eq(assignable(u, u.t_float, u.t_int), REJECTED)
}

fn test_int_never_widens_below_the_top_level() -> void {
    # The reason the widening is not part of the relation. A `list[int]` that
    # passed as a `list[float]` would leave the elements unconverted, and a
    # `fn() -> int` that passed as a `fn() -> float` would leave a result that
    # does not exist yet unconverted. There is no node to hang the conversion
    # on in either case.
    let u = Universe()
    assert_eq(assignable(u, list_of(u, u.t_int), list_of(u, u.t_float)), REJECTED)
    assert_eq(assignable(u, set_of(u, u.t_int), set_of(u, u.t_float)), REJECTED)
    assert_eq(assignable(u, dict_of(u, u.t_str, u.t_int), dict_of(u, u.t_str, u.t_float)), REJECTED)
    assert_eq(assignable(u, tuple_of(u, [u.t_int]), tuple_of(u, [u.t_float])), REJECTED)
    assert_eq(assignable(u, fn_of(u, [], u.t_int), fn_of(u, [], u.t_float)), REJECTED)
    # Contravariance would make this the parameter case of the same rule.
    assert_eq(
        assignable(u, fn_of(u, [u.t_float], u.t_void), fn_of(u, [u.t_int], u.t_void)),
        REJECTED
    )
}

fn test_widen_target_answers_only_where_a_conversion_is_needed() -> void {
    let u = Universe()
    assert_true(widen_target(u, u.t_int, u.t_float) == u.t_float)
    # `float?` and its kin: the value still has to become a float to fit.
    assert_true(widen_target(u, u.t_int, optional_of(u, u.t_float)) == u.t_float)
    assert_true(widen_target(u, u.t_int, union_of(u, [u.t_float, u.t_str])) == u.t_float)
    # A union that already admits `int` was accepted by plain assignability and
    # needs no conversion at all.
    assert_true(widen_target(u, u.t_int, union_of(u, [u.t_int, u.t_float])) is null)
    assert_eq(assignable(u, u.t_int, union_of(u, [u.t_int, u.t_float])), FREE)

    assert_true(widen_target(u, u.t_int, u.t_str) is null)
    assert_true(widen_target(u, u.t_int, u.t_any) is null)
    assert_true(widen_target(u, u.t_int, list_of(u, u.t_float)) is null)
    assert_true(widen_target(u, u.t_float, u.t_int) is null)
    assert_true(widen_target(u, u.t_bool, u.t_float) is null)
    assert_true(widen_target(u, null, u.t_float) is null)
    assert_true(widen_target(u, u.t_int, null) is null)
}

fn test_widen_target_refuses_a_union_that_admits_any() -> void {
    # `union_of` cannot build this one — `any` absorbs the union it is a member
    # of — but a union interned directly still reaches `widen_target`, and the
    # C tests the member, so this port does too. Untested the branch reads as
    # dead code, and deleting it would turn `int` reaching such a union into a
    # conversion where the C leaves it alone.
    let u = Universe()
    let admits_any = u.intern(TyKind.Union, null, [u.t_float, u.t_any])
    assert_true(widen_target(u, u.t_int, admits_any) is null)
    # The same union with the `any` arm removed is the shape that answers.
    assert_true(widen_target(u, u.t_int, union_of(u, [u.t_float, u.t_str])) == u.t_float)
}

fn test_is_assignable_drops_the_guard_and_is_subtype_rejects_it() -> void {
    # Subtyping is assignability minus the dynamic escape hatch: `any` is not a
    # subtype of a concrete type, it converts into one, and so does `int`.
    let u = Universe()
    let (animal, dog) = _animals(u)
    assert_true(is_assignable(u, u.t_any, u.t_int))
    assert_false(is_subtype(u, u.t_any, u.t_int))
    assert_true(is_assignable(u, u.t_int, u.t_float))
    assert_false(is_subtype(u, u.t_int, u.t_float))
    assert_false(is_subtype(u, generic_param(u, "T"), u.t_int))

    assert_true(is_subtype(u, dog, animal))
    assert_true(is_subtype(u, u.t_int, u.t_any))
    assert_true(is_subtype(u, u.t_never, u.t_int))
    assert_true(is_subtype(u, u.t_int, generic_param(u, "T")))
    assert_false(is_assignable(u, animal, dog))
    assert_false(is_subtype(u, animal, dog))
}
