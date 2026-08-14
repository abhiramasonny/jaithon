#: `jaithon.compile.check.substitute` — replacing generic parameters by name.
#:
#: Almost every assertion here is about identity rather than about shape, and
#: that is the point. Two properties carry the weight:
#:
#: - a type that mentions no bound parameter comes back as **the same object**.
#:   For a type this universe interned that is nearly free — a rebuild would
#:   land on the same object — so the test that pins it hands over a type the
#:   universe never interned, which is where the two answers differ;
#: - a type that *was* rebuilt comes back **interned**, because a structural
#:   twin of a type is not that type, and every relation the checker computes
#:   assumes otherwise.
#:
#: So the assertions are mostly `==` between two `Type` objects. That is not a
#: structural comparison — `Type` defines no `__eq__` — it is the identity
#: question, asked the way the checker asks it.
from std.test import assert_eq, assert_false, assert_true, fail
from jaithon.compile.check.kinds import FN_KWREST, FN_VARIADIC, TyKind
from jaithon.compile.check.render import render
from jaithon.compile.check.substitute import mentions_generic_param, substitute
from jaithon.compile.check.ty import Type, TypeDecl
from jaithon.compile.check.union import optional_of, union_of
from jaithon.compile.check.universe import Universe, dict_of, fn_of, generic_param
from jaithon.compile.check.universe import instantiate, list_of, named, set_of
from jaithon.compile.check.universe import tuple_of
from jaithon.ast import Span

fn _sub(u: Universe, t: Type, bindings: dict[str, Type]) -> Type {
    let out = substitute(u, t, bindings)
    if out is not null { return out }
    fail("substitute answered null for a type that was not null")
    return u.t_never
}

fn test_a_bare_parameter_becomes_its_binding() -> void {
    let u = Universe()
    assert_true(_sub(u, generic_param(u, "T"), {"T": u.t_int}) == u.t_int)
}

fn test_an_unbound_parameter_is_left_alone() -> void {
    let u = Universe()
    let other = generic_param(u, "K")
    assert_true(_sub(u, other, {"T": u.t_int}) == other)
}

fn test_a_nameless_parameter_matches_nothing() -> void {
    let u = Universe()
    let nameless = generic_param(u, null)
    assert_true(_sub(u, nameless, {"T": u.t_int}) == nameless)
}

fn test_null_in_is_null_out() -> void {
    let u = Universe()
    assert_true(substitute(u, null, {"T": u.t_int}) is null)
}

fn test_an_empty_mapping_answers_the_type_itself() -> void {
    let u = Universe()
    let empty: dict[str, Type] = {}
    let t = list_of(u, generic_param(u, "T"))
    assert_true(substitute(u, t, empty) == t)
}

fn test_a_type_with_nothing_to_replace_is_the_same_object() -> void {
    let u = Universe()
    let bindings = {"T": u.t_int}
    assert_true(_sub(u, u.t_str, bindings) == u.t_str)
    let deep = fn_of(u, [list_of(u, u.t_str)], dict_of(u, u.t_int, set_of(u, u.t_bool)))
    assert_true(_sub(u, deep, bindings) == deep)
}

fn test_the_early_out_answers_the_argument() -> void {
    let u = Universe()
    let bindings = {"T": u.t_int}
    let foreign_list = Type(TyKind.List, null, [u.t_str])
    let foreign_set = Type(TyKind.Set, null, [u.t_str])
    let foreign_dict = Type(TyKind.Dict, null, [u.t_str, u.t_int])
    let foreign_tuple = Type(TyKind.Tuple, null, [u.t_str])
    assert_true(substitute(u, foreign_list, bindings) == foreign_list)
    assert_true(substitute(u, foreign_set, bindings) == foreign_set)
    assert_true(substitute(u, foreign_dict, bindings) == foreign_dict)
    assert_true(substitute(u, foreign_tuple, bindings) == foreign_tuple)
    assert_eq(u.count(), 9)
}

fn test_a_named_type_with_no_arguments_is_untouched() -> void {
    let u = Universe()
    let box = named(u, TyKind.Class, "Box")
    assert_true(_sub(u, box, {"T": u.t_int}) == box)
}

fn test_a_module_is_its_own_answer() -> void {
    let u = Universe()
    let os = named(u, TyKind.Module, "os")
    assert_true(_sub(u, os, {"T": u.t_int}) == os)
}

fn test_an_instantiation_with_nothing_to_replace_is_the_same_object() -> void {
    let u = Universe()
    let base = named(u, TyKind.Class, "Box")
    let box_str = instantiate(u, base, [u.t_str])
    assert_true(_sub(u, box_str, {"T": u.t_int}) == box_str)
}

fn test_the_containers_substitute_their_elements() -> void {
    let u = Universe()
    let param = generic_param(u, "T")
    let bindings = {"T": u.t_int}
    assert_true(_sub(u, list_of(u, param), bindings) == list_of(u, u.t_int))
    assert_true(_sub(u, set_of(u, param), bindings) == set_of(u, u.t_int))
    let mapping = dict_of(u, param, list_of(u, param))
    assert_true(_sub(u, mapping, bindings) == dict_of(u, u.t_int, list_of(u, u.t_int)))
}

fn test_substitution_reaches_nested_arguments() -> void {
    let u = Universe()
    let param = generic_param(u, "T")
    let before = list_of(u, dict_of(u, u.t_str, set_of(u, param)))
    let after = _sub(u, before, {"T": u.t_int})
    assert_eq(render(after), "list[dict[str, set[int]]]")
    assert_true(after == list_of(u, dict_of(u, u.t_str, set_of(u, u.t_int))))
}

fn test_several_parameters_are_replaced_in_one_walk() -> void {
    let u = Universe()
    let before = dict_of(u, generic_param(u, "K"), list_of(u, generic_param(u, "V")))
    let after = _sub(u, before, {"K": u.t_str, "V": u.t_int})
    assert_eq(render(after), "dict[str, list[int]]")
}

fn test_a_parameter_may_be_replaced_by_another_parameter() -> void {
    let u = Universe()
    let before = list_of(u, generic_param(u, "T"))
    let after = _sub(u, before, {"T": generic_param(u, "U")})
    assert_eq(render(after), "list[U]")
    assert_true(mentions_generic_param(after))
}

fn test_a_tuple_substitutes_member_by_member() -> void {
    let u = Universe()
    let before = tuple_of(u, [generic_param(u, "T"), u.t_str, generic_param(u, "K")])
    let after = _sub(u, before, {"T": u.t_int})
    assert_eq(render(after), "(int, str, K)")
    assert_true(after == tuple_of(u, [u.t_int, u.t_str, generic_param(u, "K")]))
}

fn test_a_function_substitutes_its_parameters_and_its_return() -> void {
    let u = Universe()
    let param = generic_param(u, "T")
    let before = fn_of(u, [param, list_of(u, param)], param)
    let after = _sub(u, before, {"T": u.t_int})
    assert_eq(render(after), "fn(int, list[int]) -> int")
    assert_true(after == fn_of(u, [u.t_int, list_of(u, u.t_int)], u.t_int))
}

fn test_a_changed_return_alone_is_enough_to_rebuild() -> void {
    let u = Universe()
    let before = fn_of(u, [u.t_str], generic_param(u, "T"))
    assert_true(_sub(u, before, {"T": u.t_int}) == fn_of(u, [u.t_str], u.t_int))
}

fn test_a_function_that_declared_no_return_still_declares_none() -> void {
    let u = Universe()
    let before = fn_of(u, [generic_param(u, "T")], null)
    let after = _sub(u, before, {"T": u.t_int})
    assert_eq(render(after), "fn(int) -> any")
    assert_true(after == fn_of(u, [u.t_int], null))
}

fn test_the_function_flags_survive_substitution() -> void {
    let u = Universe()
    let param = generic_param(u, "T")
    let flags = FN_VARIADIC | FN_KWREST
    let params = [param, list_of(u, param), dict_of(u, u.t_str, param)]
    let before = fn_of(u, params, u.t_null, flags)
    let after = _sub(u, before, {"T": u.t_int})
    assert_eq(render(after), "fn(int, ...list[int], **dict[str, int]) -> void")
    let want = [u.t_int, list_of(u, u.t_int), dict_of(u, u.t_str, u.t_int)]
    assert_true(after == fn_of(u, want, u.t_null, flags))
}

fn test_a_union_is_renormalised_after_substitution() -> void {
    let u = Universe()
    let members: list[Type?] = [generic_param(u, "T"), u.t_int]
    let before = union_of(u, members)
    assert_true(_sub(u, before, {"T": u.t_int}) == u.t_int)
}

fn test_substituting_into_a_union_keeps_the_optional_sugar() -> void {
    let u = Universe()
    let members: list[Type?] = [generic_param(u, "T"), u.t_null]
    let before = union_of(u, members)
    let after = _sub(u, before, {"T": u.t_int})
    assert_eq(render(after), "int?")
    assert_true(after == optional_of(u, u.t_int))
}

fn test_any_absorbs_the_union_it_is_substituted_into() -> void {
    let u = Universe()
    let members: list[Type?] = [generic_param(u, "T"), u.t_str]
    let before = union_of(u, members)
    assert_true(_sub(u, before, {"T": u.t_any}) == u.t_any)
}

fn test_a_generic_instantiation_keeps_its_name() -> void {
    let u = Universe()
    let base = named(u, TyKind.Class, "Box")
    let before = instantiate(u, base, [generic_param(u, "T")])
    let after = _sub(u, before, {"T": u.t_int})
    assert_eq(render(after), "Box[int]")
    assert_true(after == instantiate(u, base, [u.t_int]))
}

fn test_a_generic_instantiation_keeps_its_declaration() -> void {
    let u = Universe()
    let declaration = TypeDecl("Box", Span.none())
    let base = named(u, TyKind.Class, "Box", declaration)
    let before = instantiate(u, base, [generic_param(u, "T")])
    let after = _sub(u, before, {"T": u.t_int})
    assert_true(after.decl == declaration)
}

fn test_a_trait_and_an_enum_substitute_the_same_way() -> void {
    let u = Universe()
    let param = generic_param(u, "T")
    let trait_base = named(u, TyKind.Trait, "Iterable")
    let enum_base = named(u, TyKind.Enum, "Result")
    let iterable = instantiate(u, trait_base, [param])
    let result = instantiate(u, enum_base, [param, u.t_str])
    assert_eq(render(_sub(u, iterable, {"T": u.t_int})), "Iterable[int]")
    assert_eq(render(_sub(u, result, {"T": u.t_int})), "Result[int, str]")
}

fn test_self_is_a_generic_parameter_like_any_other() -> void {
    let u = Universe()
    let implementer = named(u, TyKind.Class, "Counter")
    let requirement = fn_of(u, [], generic_param(u, "Self"))
    assert_eq(render(_sub(u, requirement, {"Self": implementer})), "fn() -> Counter")
}

fn test_an_alias_is_answered_unchanged() -> void {
    let u = Universe()
    let alias = Type(TyKind.Alias, "Id", [], generic_param(u, "T"))
    assert_true(substitute(u, alias, {"T": u.t_int}) == alias)
}

fn test_mentions_is_false_for_a_type_with_no_parameter() -> void {
    let u = Universe()
    assert_false(mentions_generic_param(null))
    assert_false(mentions_generic_param(u.t_int))
    assert_false(mentions_generic_param(list_of(u, dict_of(u, u.t_str, u.t_bool))))
    assert_false(mentions_generic_param(fn_of(u, [u.t_int], u.t_str)))
}

fn test_mentions_finds_a_parameter_at_any_depth() -> void {
    let u = Universe()
    let param = generic_param(u, "T")
    assert_true(mentions_generic_param(param))
    assert_true(mentions_generic_param(list_of(u, set_of(u, param))))
    assert_true(mentions_generic_param(tuple_of(u, [u.t_int, param])))
    let members: list[Type?] = [param, u.t_null]
    assert_true(mentions_generic_param(union_of(u, members)))
    let base = named(u, TyKind.Class, "Box")
    assert_true(mentions_generic_param(instantiate(u, base, [list_of(u, param)])))
}

fn test_mentions_reads_a_return_type_only_on_a_function() -> void {
    let u = Universe()
    let param = generic_param(u, "T")
    assert_true(mentions_generic_param(fn_of(u, [u.t_bool], param)))
    let alias = Type(TyKind.Alias, "Id", [], param)
    assert_false(mentions_generic_param(alias))
}

fn test_a_guard_is_refused_exactly_while_a_parameter_is_unsubstituted() -> void {
    let u = Universe()
    let erased = list_of(u, generic_param(u, "T"))
    assert_true(mentions_generic_param(erased))
    assert_false(mentions_generic_param(_sub(u, erased, {"T": u.t_int})))
}
