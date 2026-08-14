#: `jaithon.compile.check.nominal` — the relations over `TypeDecl`.
#:
#: Hierarchies are built by hand here rather than parsed, because these
#: relations run against declarations that layout has not finished with, and a
#: hand-built one is the only way to reach the states a parser would never
#: produce: a cycle, a trait that names a trait nobody can see, a member whose
#: type has not been computed yet.
#:
#: Identity is object identity throughout. Two `TypeDecl`s with the same name
#: are two declarations, exactly as two modules each declaring a `Node` declare
#: two classes, so every assertion below compares the objects the test made.
from std.test import assert_eq, assert_false, assert_true
from jaithon.ast import Span
from jaithon.compile.check.kinds import TyKind
from jaithon.compile.check.nominal import find_accessor, find_field, find_member
from jaithon.compile.check.nominal import implements_trait, inherits_from, inherits_opaquely
from jaithon.compile.check.nominal import is_subclass_of, names_opaque_trait
from jaithon.compile.check.nominal import owner_of_field, owner_of_member, satisfies_trait
from jaithon.compile.check.ty import FieldInfo, MemberInfo, TypeDecl
from jaithon.compile.check.universe import Universe, named

fn _decl(name: str) -> TypeDecl {
    return TypeDecl(name, Span.none())
}

fn _trait(name: str) -> TypeDecl {
    let d = TypeDecl(name, Span.none())
    d.is_trait = true
    return d
}

fn _field(name: str, slot: int = 0) -> FieldInfo {
    return FieldInfo(name, null, slot)
}

fn _method(name: str) -> MemberInfo {
    return MemberInfo(name)
}

fn _chain(names: list[str]) -> list[TypeDecl] {
    let built = [_decl(name) for name in names]
    for i in range(0, built.len() - 1) {
        built[i].superclass = built[i + 1]
    }
    return built
}

fn test_a_declaration_is_its_own_subclass() -> void {
    let animal = _decl("Animal")
    assert_true(is_subclass_of(animal, animal))
}

fn test_a_subclass_reaches_every_ancestor() -> void {
    let line = _chain(["Cat", "Mammal", "Animal"])
    assert_true(is_subclass_of(line[0], line[1]))
    assert_true(is_subclass_of(line[0], line[2]))
    assert_true(is_subclass_of(line[1], line[2]))
}

fn test_subclassing_does_not_run_backwards() -> void {
    let line = _chain(["Cat", "Animal"])
    assert_false(is_subclass_of(line[1], line[0]))
}

fn test_two_declarations_of_one_name_are_two_classes() -> void {
    assert_false(is_subclass_of(_decl("Animal"), _decl("Animal")))
}

fn test_a_missing_declaration_is_never_a_subclass() -> void {
    let animal = _decl("Animal")
    assert_false(is_subclass_of(null, animal))
    assert_false(is_subclass_of(animal, null))
    assert_false(is_subclass_of(null, null))
}

fn test_a_trait_is_not_a_superclass() -> void {
    let drawable = _trait("Drawable")
    let widget = _decl("Widget")
    widget.traits = [drawable]
    assert_false(is_subclass_of(widget, drawable))
    assert_true(implements_trait(widget, drawable))
}

fn test_a_class_implements_the_traits_it_names() -> void {
    let drawable = _trait("Drawable")
    let widget = _decl("Widget")
    widget.traits = [drawable]
    assert_true(implements_trait(widget, drawable))
}

fn test_a_trait_implements_the_traits_it_extends() -> void {
    let comparable = _trait("Comparable")
    let ordered = _trait("Ordered")
    ordered.traits = [comparable]
    let money = _decl("Money")
    money.traits = [ordered]
    assert_true(implements_trait(ordered, comparable))
    assert_true(implements_trait(money, comparable))
}

fn test_a_trait_is_inherited_through_the_superclass() -> void {
    let drawable = _trait("Drawable")
    let line = _chain(["Button", "Widget"])
    line[1].traits = [drawable]
    assert_true(implements_trait(line[0], drawable))
}

fn test_implementing_is_reflexive_and_otherwise_earned() -> void {
    let drawable = _trait("Drawable")
    let printable = _trait("Printable")
    assert_true(implements_trait(drawable, drawable))
    assert_false(implements_trait(drawable, printable))
    assert_false(implements_trait(null, drawable))
    assert_false(implements_trait(drawable, null))
}

fn test_visibility_asks_the_same_closure_under_its_own_name() -> void {
    let drawable = _trait("Drawable")
    let line = _chain(["Button", "Widget"])
    line[1].traits = [drawable]
    assert_true(inherits_from(line[0], line[1]))
    assert_true(inherits_from(line[0], drawable))
    assert_true(implements_trait(line[0], line[1]))
    assert_false(inherits_from(line[1], line[0]))
}

fn test_an_invisible_trait_is_matched_by_its_name() -> void {
    let widget = _decl("Widget")
    widget.opaque_traits = ["Drawable"]
    assert_true(names_opaque_trait(widget, "Drawable"))
    assert_false(names_opaque_trait(widget, "Printable"))
}

fn test_an_invisible_trait_is_inherited_like_a_visible_one() -> void {
    let base = _trait("Styled")
    base.opaque_traits = ["Themed"]
    let line = _chain(["Button", "Widget"])
    line[1].opaque_traits = ["Drawable"]
    line[0].traits = [base]
    assert_true(names_opaque_trait(line[0], "Drawable"))
    assert_true(names_opaque_trait(line[0], "Themed"))
}

fn test_a_name_nobody_wrote_is_not_matched() -> void {
    let widget = _decl("Widget")
    assert_false(names_opaque_trait(widget, "Drawable"))
    assert_false(names_opaque_trait(null, "Drawable"))
    assert_false(names_opaque_trait(widget, null))
}

fn test_satisfying_a_trait_prefers_the_declaration() -> void {
    let u = Universe()
    let drawable = _trait("Drawable")
    let widget = _decl("Widget")
    widget.traits = [drawable]
    let seen = named(u, TyKind.Trait, "Drawable", drawable)
    assert_true(satisfies_trait(widget, seen))
    assert_false(satisfies_trait(_decl("Label"), seen))
}

fn test_satisfying_a_trait_falls_back_to_the_name() -> void {
    let u = Universe()
    let widget = _decl("Widget")
    widget.opaque_traits = ["Drawable"]
    let unseen = named(u, TyKind.Trait, "Drawable")
    assert_true(satisfies_trait(widget, unseen))
    assert_false(satisfies_trait(_decl("Label"), unseen))
}

fn test_an_invisible_parent_is_visible_as_a_gap() -> void {
    let imported = _decl("Widget")
    imported.opaque_super = "Control"
    assert_true(inherits_opaquely(imported))

    let named_trait = _decl("Button")
    named_trait.opaque_traits = ["Drawable"]
    assert_true(inherits_opaquely(named_trait))

    let line = _chain(["Toggle", "Button", "Widget"])
    line[2].opaque_super = "Control"
    assert_true(inherits_opaquely(line[0]))
}

fn test_a_fully_visible_hierarchy_inherits_nothing_opaquely() -> void {
    let drawable = _trait("Drawable")
    let line = _chain(["Button", "Widget"])
    line[1].traits = [drawable]
    assert_false(inherits_opaquely(line[0]))
    assert_false(inherits_opaquely(null))
}

fn test_a_field_is_found_on_the_declaration_itself() -> void {
    let animal = _decl("Animal")
    animal.fields = [_field("name", 0), _field("legs", 1)]
    assert_eq(find_field(animal, "legs")?.slot, 1)
}

fn test_a_field_is_found_on_an_ancestor() -> void {
    let line = _chain(["Cat", "Mammal", "Animal"])
    line[2].fields = [_field("name", 0)]
    line[0].fields = [_field("indoor", 1)]
    assert_eq(find_field(line[0], "name")?.name, "name")
    assert_eq(find_field(line[0], "indoor")?.slot, 1)
    assert_true(find_field(line[2], "indoor") is null)
}

fn test_a_field_is_not_looked_for_in_traits() -> void {
    let drawable = _trait("Drawable")
    drawable.fields = [_field("pixels", 0)]
    let widget = _decl("Widget")
    widget.traits = [drawable]
    assert_true(find_field(widget, "pixels") is null)
}

fn test_a_missing_field_is_null() -> void {
    let animal = _decl("Animal")
    assert_true(find_field(animal, "legs") is null)
    assert_true(find_field(null, "legs") is null)
}

fn test_a_member_is_found_on_the_declaration_itself() -> void {
    let animal = _decl("Animal")
    animal.members = [_method("speak")]
    assert_eq(find_member(animal, "speak")?.name, "speak")
}

fn test_an_override_wins_over_what_it_overrides() -> void {
    let line = _chain(["Cat", "Animal"])
    let inherited = _method("speak")
    let overriding = _method("speak")
    line[1].members = [inherited]
    line[0].members = [overriding]
    assert_true(find_member(line[0], "speak") == overriding)
    assert_true(find_member(line[1], "speak") == inherited)
}

fn test_the_declaration_is_searched_before_its_traits() -> void {
    let drawable = _trait("Drawable")
    let default_body = _method("describe")
    let overriding = _method("describe")
    drawable.members = [default_body]
    let widget = _decl("Widget")
    widget.traits = [drawable]
    widget.members = [overriding]
    assert_true(find_member(widget, "describe") == overriding)
}

fn test_a_trait_is_searched_before_the_superclass() -> void {
    let first = _trait("Styled")
    let second = _trait("Drawable")
    let from_trait = _method("describe")
    let from_parent = _method("describe")
    first.members = [from_trait]
    let line = _chain(["Button", "Widget"])
    line[1].members = [from_parent]
    line[0].traits = [first, second]
    assert_true(find_member(line[0], "describe") == from_trait)

    line[0].traits = [second]
    assert_true(find_member(line[0], "describe") == from_parent)
}

fn test_a_member_that_has_not_been_checked_yet_is_still_found() -> void {
    let animal = _decl("Animal")
    animal.members = [_method("speak")]
    let found = find_member(animal, "speak")
    assert_false(found is null)
    assert_true(found?.ty is null)
}

fn test_a_missing_member_is_null() -> void {
    let animal = _decl("Animal")
    assert_true(find_member(animal, "speak") is null)
    assert_true(find_member(null, "speak") is null)
}

fn test_a_getter_and_a_setter_share_one_name() -> void {
    let getter = MemberInfo("width", null, 0, false, true, false)
    let setter = MemberInfo("width", null, 0, false, false, true)
    let box = _decl("Box")
    box.members = [getter, setter]
    assert_true(find_accessor(box, "width", false) == getter)
    assert_true(find_accessor(box, "width", true) == setter)
    assert_true(find_member(box, "width") == getter)
    assert_true(find_accessor(box, "height", true) is null)

    let write_only = _decl("Gauge")
    write_only.members = [MemberInfo("level", null, 0, false, false, true)]
    assert_true(find_accessor(write_only, "level", false) is null)
    assert_false(find_member(write_only, "level") is null)
}

fn test_an_accessor_is_inherited() -> void {
    let setter = MemberInfo("width", null, 0, false, false, true)
    let line = _chain(["Button", "Widget"])
    line[1].members = [setter]
    assert_true(find_accessor(line[0], "width", true) == setter)
}

fn test_the_owner_is_the_topmost_declaration_that_has_it() -> void {
    let line = _chain(["Cat", "Mammal", "Animal"])
    line[2].fields = [_field("name")]
    line[1].fields = [_field("name")]
    line[2].members = [_method("speak")]
    line[0].members = [_method("speak")]
    assert_true(owner_of_field(line[0], "name") == line[2])
    assert_true(owner_of_member(line[0], "speak") == line[2])
}

fn test_a_trait_default_is_owned_by_the_class_that_implements_it() -> void {
    let drawable = _trait("Drawable")
    drawable.members = [_method("describe")]
    let line = _chain(["Toggle", "Button", "Widget"])
    line[2].traits = [drawable]
    assert_true(owner_of_member(line[1], "describe") == line[2])
    assert_true(owner_of_member(line[0], "describe") == line[2])

    assert_true(owner_of_member(line[2], "describe") == line[2])
}

fn test_the_owner_of_a_name_nobody_declares_is_the_one_asked() -> void {
    let line = _chain(["Cat", "Animal"])
    assert_true(owner_of_field(line[0], "name") == line[0])
    assert_true(owner_of_member(line[0], "speak") == line[0])
    assert_true(owner_of_field(null, "name") is null)
    assert_true(owner_of_member(null, "speak") is null)
}

fn test_the_two_depth_caps_differ_by_exactly_one() -> void {
    let line = _chain([f"D{i}" for i in range(0, 40)])
    assert_true(is_subclass_of(line[0], line[31]))
    assert_false(is_subclass_of(line[0], line[32]))
    assert_true(inherits_from(line[0], line[32]))
    assert_false(inherits_from(line[0], line[33]))
}

fn test_a_lookup_stops_at_the_cap_too() -> void {
    let line = _chain([f"D{i}" for i in range(0, 40)])
    line[32].fields = [_field("near")]
    line[33].fields = [_field("far")]
    line[32].members = [_method("near")]
    line[33].members = [_method("far")]
    assert_eq(find_field(line[0], "near")?.name, "near")
    assert_true(find_field(line[0], "far") is null)
    assert_eq(find_member(line[0], "near")?.name, "near")
    assert_true(find_member(line[0], "far") is null)
}

fn test_every_uncapped_walk_reaches_exactly_as_far() -> void {
    let last = _chain([f"D{i}" for i in range(0, 40)])
    last[32].opaque_super = "Control"
    last[32].opaque_traits = ["Drawable"]
    last[32].fields = [_field("edge")]
    last[32].members = [_method("edge")]
    assert_true(inherits_opaquely(last[0]))
    assert_true(names_opaque_trait(last[0], "Drawable"))
    assert_eq(find_field(last[0], "edge")?.name, "edge")
    assert_eq(find_member(last[0], "edge")?.name, "edge")
    assert_true(owner_of_field(last[0], "edge") == last[32])
    assert_true(owner_of_member(last[0], "edge") == last[32])

    let past = _chain([f"D{i}" for i in range(0, 40)])
    past[33].opaque_traits = ["Drawable"]
    past[33].fields = [_field("edge")]
    past[33].members = [_method("edge")]
    assert_false(inherits_opaquely(past[0]))
    assert_false(names_opaque_trait(past[0], "Drawable"))
    assert_true(find_field(past[0], "edge") is null)
    assert_true(find_member(past[0], "edge") is null)
    assert_true(owner_of_field(past[0], "edge") == past[0])
    assert_true(owner_of_member(past[0], "edge") == past[0])
}

fn test_a_cyclic_hierarchy_answers_instead_of_hanging() -> void {
    let odd = _decl("Odd")
    let even = _decl("Even")
    odd.superclass = even
    even.superclass = odd
    let stranger = _decl("Stranger")
    assert_false(is_subclass_of(odd, stranger))
    assert_false(inherits_from(odd, stranger))
    assert_false(names_opaque_trait(odd, "Drawable"))
    assert_false(inherits_opaquely(odd))
    assert_true(find_field(odd, "legs") is null)
    assert_true(find_member(odd, "speak") is null)
    assert_true(owner_of_field(odd, "legs") == odd)
    assert_true(is_subclass_of(odd, even))
}

fn test_a_cycle_through_traits_also_terminates() -> void {
    let first = _trait("First")
    let second = _trait("Second")
    first.traits = [second]
    second.traits = [first]
    assert_false(implements_trait(first, _trait("Third")))
    assert_true(implements_trait(first, second))
    assert_false(names_opaque_trait(first, "Third"))
}
