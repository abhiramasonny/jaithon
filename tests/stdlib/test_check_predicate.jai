#: `jaithon.compile.check.predicate` — the five type-shape predicates.
#:
#: The cases that cost something to get wrong are the ones where a shape almost
#: fits: a dict iterates its keys and not its pairs, a set is iterable but not
#: indexable, a union is neither however uniform its members are, and a class
#: that implements an iterator is iterable with element `any` rather than with
#: the element it really has. Each of those is a line of `src/sema/types.c` that
#: reads like an oversight and is not, so each has a test naming what it is.
from std.test import assert_eq, assert_false, assert_true
from jaithon.ast import Span
from jaithon.compile.check.kinds import MAX_DECL_DEPTH, TyKind
from jaithon.compile.check.predicate import is_callable, is_indexable, is_iterable
from jaithon.compile.check.predicate import is_numeric, is_optional
from jaithon.compile.check.render import render
from jaithon.compile.check.ty import MemberInfo, Type, TypeDecl
from jaithon.compile.check.union import optional_of, union_of
from jaithon.compile.check.universe import Universe, dict_of, fn_of, generic_param
from jaithon.compile.check.universe import list_of, named, set_of, tuple_of

# A join that answers a sentinel whatever it is handed, so a test can tell
# which arm of a predicate consulted it and which computed its own answer.
fn _sentinel_join(u: Universe, a: Type?, b: Type?) -> Type {
    return u.t_bool
}

# A declaration that declares `members` and nothing else.
fn _decl(name: str, members: list[str]) -> TypeDecl {
    let declaration = TypeDecl(name, Span.none())
    for member in members { declaration.members.push(MemberInfo(member)) }
    return declaration
}

# `length` declarations, each the superclass of the one before it, with `member`
# declared only on the deepest. The head is what gets asked.
fn _chain(prefix: str, length: int, member: str) -> TypeDecl {
    var walk = _decl(f"{prefix}{length - 1}", [member])
    var at = length - 2
    while at >= 0 {
        let above = _decl(f"{prefix}{at}", [])
        above.superclass = walk
        walk = above
        at -= 1
    }
    return walk
}

fn test_only_int_and_float_are_numeric() -> void {
    let u = Universe()
    assert_true(is_numeric(u.t_int))
    assert_true(is_numeric(u.t_float))
    # bool is stored as an integer and is still not a number: the language has
    # no bool arithmetic, and this is where saying so begins.
    assert_false(is_numeric(u.t_bool))
    assert_false(is_numeric(u.t_str))
    assert_false(is_numeric(u.t_bytes))
    assert_false(is_numeric(u.t_any))
    assert_false(is_numeric(u.t_null))
    assert_false(is_numeric(u.t_never))
    assert_false(is_numeric(null))
}

fn test_null_and_any_are_optional_with_no_union_around_them() -> void {
    let u = Universe()
    assert_true(is_optional(u.t_null))
    assert_true(is_optional(u.t_any))
    assert_false(is_optional(null))
}

fn test_a_union_is_optional_exactly_when_a_member_is_null() -> void {
    let u = Universe()
    assert_true(is_optional(optional_of(u, u.t_int)))
    assert_true(is_optional(union_of(u, [u.t_int, u.t_str, u.t_null])))
    assert_false(is_optional(union_of(u, [u.t_int, u.t_str])))
    assert_false(is_optional(u.t_int))
    # The question is about the type itself, not about what it contains.
    assert_false(is_optional(list_of(u, u.t_null)))
}

fn test_functions_classes_and_enums_are_callable() -> void {
    let u = Universe()
    assert_true(is_callable(fn_of(u, [u.t_int], u.t_str)))
    assert_true(is_callable(u.t_any))
    # The class object is its own constructor, and an enum is how a variant is
    # built; neither needs a dunder.
    assert_true(is_callable(named(u, TyKind.Class, "Box")))
    assert_true(is_callable(named(u, TyKind.Enum, "Colour")))
}

fn test_nothing_else_is_callable() -> void {
    let u = Universe()
    assert_false(is_callable(u.t_int))
    assert_false(is_callable(u.t_str))
    assert_false(is_callable(u.t_null))
    assert_false(is_callable(u.t_never))
    assert_false(is_callable(list_of(u, u.t_int)))
    assert_false(is_callable(generic_param(u, "T")))
    assert_false(is_callable(union_of(u, [fn_of(u, [], u.t_int), u.t_null])))
    assert_false(is_callable(null))
}

fn test_a_trait_is_callable_only_when_it_declares_the_dunder() -> void {
    let u = Universe()
    assert_true(is_callable(named(u, TyKind.Trait, "Fn", _decl("Fn", ["__call__"]))))
    assert_false(is_callable(named(u, TyKind.Trait, "Shape", _decl("Shape", ["area"]))))
    # A trait whose declaration is not visible has no members to find.
    assert_false(is_callable(named(u, TyKind.Trait, "Opaque")))
}

fn test_a_dunder_is_found_through_a_trait_and_a_superclass() -> void {
    let u = Universe()
    let inherited = _decl("Callable", ["__call__"])
    let sub = _decl("Sub", [])
    sub.superclass = inherited
    assert_true(is_callable(named(u, TyKind.Trait, "Sub", sub)))

    let implementer = _decl("Impl", [])
    implementer.traits.push(inherited)
    assert_true(is_callable(named(u, TyKind.Trait, "Impl", implementer)))
}

fn test_the_containers_iterate_as_the_c_says_they_do() -> void {
    let u = Universe()
    assert_true(is_iterable(u, list_of(u, u.t_int)) == u.t_int)
    assert_true(is_iterable(u, set_of(u, u.t_str)) == u.t_str)
    # A dict iterates its keys, not its pairs.
    assert_true(is_iterable(u, dict_of(u, u.t_str, u.t_int)) == u.t_str)
    # A str yields one-scalar strings; bytes is the one that yields ints.
    assert_true(is_iterable(u, u.t_str) == u.t_str)
    assert_true(is_iterable(u, u.t_bytes) == u.t_int)
    assert_true(is_iterable(u, u.t_range) == u.t_int)
    assert_true(is_iterable(u, u.t_any) == u.t_any)
}

fn test_what_cannot_be_looped_over() -> void {
    let u = Universe()
    assert_true(is_iterable(u, u.t_int) is null)
    assert_true(is_iterable(u, u.t_bool) is null)
    assert_true(is_iterable(u, u.t_null) is null)
    assert_true(is_iterable(u, u.t_never) is null)
    assert_true(is_iterable(u, fn_of(u, [], u.t_int)) is null)
    assert_true(is_iterable(u, generic_param(u, "T")) is null)
    assert_true(is_iterable(u, null) is null)
}

fn test_a_union_is_not_iterable_however_uniform_its_members_are() -> void {
    # The C's switch has no union case, so this falls to the default. A loop
    # over a union has to be narrowed first.
    let u = Universe()
    let both = union_of(u, [list_of(u, u.t_int), list_of(u, u.t_str)])
    assert_true(is_iterable(u, both) is null)
    assert_true(is_iterable(u, optional_of(u, list_of(u, u.t_int))) is null)
}

fn test_a_user_iterable_is_iterable_with_element_any() -> void {
    # The element type is real — it is the `T` of the `Iterator[T]` that `iter`
    # returns — but reading it means opening the trait instantiation, which
    # this layer cannot do, so both front ends widen to `any`.
    let u = Universe()
    assert_true(is_iterable(u, named(u, TyKind.Class, "Rows", _decl("Rows", ["iter"]))) == u.t_any)
    let dunder = _decl("Chars", ["__iter__"])
    assert_true(is_iterable(u, named(u, TyKind.Class, "Chars", dunder)) == u.t_any)
    let trait_ty = named(u, TyKind.Trait, "Seq", _decl("Seq", ["iter"]))
    assert_true(is_iterable(u, trait_ty) == u.t_any)
    let enum_ty = named(u, TyKind.Enum, "Suit", _decl("Suit", ["__iter__"]))
    assert_true(is_iterable(u, enum_ty) == u.t_any)
}

fn test_a_user_type_without_either_name_is_not_iterable() -> void {
    let u = Universe()
    let plain = named(u, TyKind.Class, "Point", _decl("Point", ["x", "y"]))
    assert_true(is_iterable(u, plain) is null)
    # No visible declaration at all is the same answer.
    assert_true(is_iterable(u, named(u, TyKind.Class, "Opaque")) is null)
}

fn test_the_member_walk_reaches_a_trait_and_a_superclass() -> void {
    let u = Universe()
    let provider = _decl("Iterable", ["iter"])
    let implementer = _decl("Bag", [])
    implementer.traits.push(provider)
    assert_true(is_iterable(u, named(u, TyKind.Class, "Bag", implementer)) == u.t_any)

    let sub = _decl("Sub", [])
    sub.superclass = provider
    assert_true(is_iterable(u, named(u, TyKind.Class, "Sub", sub)) == u.t_any)
}

fn test_the_member_walk_stops_at_max_decl_depth() -> void {
    # A superclass chain that closes on itself is a program the checker still
    # has to answer for, so the walk is capped rather than trusted. The head is
    # depth 0, so the deepest declaration the cap admits is at MAX_DECL_DEPTH.
    let u = Universe()
    let reachable = _chain("Near", MAX_DECL_DEPTH + 1, "iter")
    assert_true(is_iterable(u, named(u, TyKind.Class, "Near0", reachable)) == u.t_any)

    let beyond = _chain("Far", MAX_DECL_DEPTH + 2, "iter")
    assert_true(is_iterable(u, named(u, TyKind.Class, "Far0", beyond)) is null)
}

fn test_a_tuple_iterates_as_the_join_of_its_members() -> void {
    let u = Universe()
    assert_eq(render(is_iterable(u, tuple_of(u, [u.t_int, u.t_str]))), "int | str")
    # Duplicates collapse and a nested null becomes the optional sugar, both
    # because the join goes through union normalisation.
    assert_true(is_iterable(u, tuple_of(u, [u.t_int, u.t_int])) == u.t_int)
    assert_eq(render(is_iterable(u, tuple_of(u, [u.t_int, optional_of(u, u.t_int)]))), "int?")
    # An empty tuple joins nothing, and a join of nothing is `never`.
    assert_true(is_iterable(u, tuple_of(u, [])) == u.t_never)
}

fn test_a_join_past_eight_members_gives_up_and_answers_any() -> void {
    let u = Universe()
    let eight: list[Type] = [
        u.t_bool,
        u.t_int,
        u.t_float,
        u.t_str,
        u.t_bytes,
        u.t_range,
        list_of(u, u.t_int),
        set_of(u, u.t_int),
    ]
    assert_false(is_iterable(u, tuple_of(u, eight)) == u.t_any)

    let nine: list[Type] = [
        u.t_bool,
        u.t_int,
        u.t_float,
        u.t_str,
        u.t_bytes,
        u.t_range,
        list_of(u, u.t_int),
        set_of(u, u.t_int),
        dict_of(u, u.t_str, u.t_int),
    ]
    assert_true(is_iterable(u, tuple_of(u, nine)) == u.t_any)
}

fn test_a_union_already_past_the_cap_survives_the_join_intact() -> void {
    # The cap is on a join that *grew* past eight members, not on one that was
    # handed a wide union and gave it straight back. `_join_all` seeds every
    # fold with `never`, so if the join did not answer `never | X` as `X`
    # before consulting the cap, the seeding step alone would answer `any`.
    # `jaiTypeJoin` short-circuits `never` and identity ahead of the cap for
    # this reason, and the C's own checker agrees: `for x in (U, U)` on a
    # nine-member `U` binds `x` to `U`.
    let u = Universe()
    let nine: list[Type?] = [
        u.t_bool,
        u.t_int,
        u.t_float,
        u.t_str,
        u.t_bytes,
        list_of(u, u.t_int),
        dict_of(u, u.t_str, u.t_int),
        set_of(u, u.t_int),
        u.t_range,
    ]
    let wide = union_of(u, nine)
    assert_eq(
        render(wide),
        "bool | int | float | str | bytes | list[int] | dict[str, int] | set[int] | range"
    )
    assert_true(is_iterable(u, tuple_of(u, [wide])) == wide)
    assert_true(is_iterable(u, tuple_of(u, [wide, wide])) == wide)
    assert_true(is_indexable(u, tuple_of(u, [wide, wide])) == (u.t_int, wide))
    # Widening it by one genuinely new member is what the cap is for.
    assert_true(is_iterable(u, tuple_of(u, [wide, u.t_null])) == u.t_any)
}

fn test_never_and_any_answer_before_the_cap_does() -> void {
    # `jaiTypeJoin`'s opening lines, in the order it puts them: an operand back
    # unchanged for `never`, `any` for `any`.
    let u = Universe()
    assert_true(is_iterable(u, tuple_of(u, [u.t_never, u.t_int])) == u.t_int)
    assert_true(is_iterable(u, tuple_of(u, [u.t_int, u.t_never])) == u.t_int)
    assert_true(is_iterable(u, tuple_of(u, [u.t_never, u.t_never])) == u.t_never)
    assert_true(is_iterable(u, tuple_of(u, [u.t_any, u.t_int])) == u.t_any)
    assert_true(is_iterable(u, tuple_of(u, [u.t_int, u.t_any])) == u.t_any)
}

fn test_the_join_is_a_seam_and_only_a_tuple_reaches_it() -> void {
    # `relate.jai` owns the real `jaiTypeJoin` and is what the default here
    # resolves to; the parameter stays so a caller that has already computed a
    # better answer can pass it. Only the tuple arms consult it at all.
    let u = Universe()
    let pair = tuple_of(u, [u.t_int, u.t_str])
    assert_true(is_iterable(u, pair, _sentinel_join) == u.t_bool)
    assert_true(is_indexable(u, pair, _sentinel_join) == (u.t_int, u.t_bool))
    assert_true(is_iterable(u, list_of(u, u.t_int), _sentinel_join) == u.t_int)
    assert_true(is_indexable(u, u.t_str, _sentinel_join) == (u.t_int, u.t_str))
}

fn test_the_default_join_is_the_real_one_and_not_a_union_of_everything() -> void {
    # The seam's default has to be `relate.join`, not `jaiTypeJoin` minus its
    # subtype half. The two answer differently for exactly one shape of member
    # pair — one member a subtype of the other without being a member of it —
    # and a tuple is where that shape reaches a loop variable: the C binds
    # `for s in (a_circle, a_shape)` to `Shape`, and a placeholder join binds
    # it to `Circle | Shape`, which is not a `Shape` and loses every inherited
    # member the body can name.
    let u = Universe()
    let shape_decl = TypeDecl("Shape", Span.none())
    let circle_decl = TypeDecl("Circle", Span.none())
    circle_decl.superclass = shape_decl
    let shape = named(u, TyKind.Class, "Shape", shape_decl)
    let circle = named(u, TyKind.Class, "Circle", circle_decl)

    let pair = tuple_of(u, [circle, shape])
    assert_true(is_iterable(u, pair) == shape)
    assert_true(is_indexable(u, pair) == (u.t_int, shape))
    # In the other order too: the fold is not commutative by construction, and
    # `jaiTypeJoin` reaches the answer through a different one of its two
    # subtype tests each way round.
    let flipped = tuple_of(u, [shape, circle])
    assert_true(is_iterable(u, flipped) == shape)

    # The erased and dynamic pairs the same half decides, which a union of
    # members would also get wrong.
    let param = generic_param(u, "T")
    assert_true(is_iterable(u, tuple_of(u, [u.t_int, param])) == param)
    let maybe = optional_of(u, u.t_int)
    assert_true(is_iterable(u, tuple_of(u, [u.t_int, maybe])) == maybe)
    # And a sibling pair still keeps both, because neither contains the other.
    let square_decl = TypeDecl("Square", Span.none())
    square_decl.superclass = shape_decl
    let square = named(u, TyKind.Class, "Square", square_decl)
    assert_eq(render(is_iterable(u, tuple_of(u, [circle, square]))), "Circle | Square")
}

fn test_the_containers_index_as_the_c_says_they_do() -> void {
    let u = Universe()
    assert_true(is_indexable(u, list_of(u, u.t_str)) == (u.t_int, u.t_str))
    # A dict is the only container that wants something other than an int.
    assert_true(is_indexable(u, dict_of(u, u.t_str, u.t_int)) == (u.t_str, u.t_int))
    assert_true(is_indexable(u, u.t_str) == (u.t_int, u.t_str))
    assert_true(is_indexable(u, u.t_bytes) == (u.t_int, u.t_int))
    assert_true(is_indexable(u, u.t_range) == (u.t_int, u.t_int))
    assert_true(is_indexable(u, u.t_any) == (u.t_any, u.t_any))
}

fn test_a_set_is_iterable_and_not_indexable() -> void {
    let u = Universe()
    let numbers = set_of(u, u.t_int)
    assert_true(is_iterable(u, numbers) == u.t_int)
    assert_true(is_indexable(u, numbers) is null)
}

fn test_what_cannot_be_indexed() -> void {
    let u = Universe()
    assert_true(is_indexable(u, u.t_int) is null)
    assert_true(is_indexable(u, u.t_bool) is null)
    assert_true(is_indexable(u, u.t_null) is null)
    assert_true(is_indexable(u, u.t_never) is null)
    assert_true(is_indexable(u, fn_of(u, [], u.t_int)) is null)
    assert_true(is_indexable(u, generic_param(u, "T")) is null)
    assert_true(is_indexable(u, union_of(u, [list_of(u, u.t_int), u.t_str])) is null)
    assert_true(is_indexable(u, null) is null)
}

fn test_a_tuple_is_indexed_by_int_and_answers_the_join() -> void {
    # The weakest of the answers, and deliberate: a caller that can see a
    # constant subscript resolves it before ever asking.
    let u = Universe()
    let pair = tuple_of(u, [u.t_str, u.t_int])
    assert_true(is_indexable(u, pair) == (u.t_int, union_of(u, [u.t_str, u.t_int])))
}

fn test_a_user_type_is_indexable_only_through_getitem() -> void {
    let u = Universe()
    let subscriptable = _decl("Grid", ["__getitem__"])
    let grid = named(u, TyKind.Class, "Grid", subscriptable)
    # The dunder's signature is not consulted, so both halves are dynamic.
    assert_true(is_indexable(u, grid) == (u.t_any, u.t_any))

    let plain = named(u, TyKind.Class, "Point", _decl("Point", ["x"]))
    assert_true(is_indexable(u, plain) is null)
    assert_true(is_indexable(u, named(u, TyKind.Class, "Opaque")) is null)

    # `iter` is not `__getitem__`: iterating and subscripting are separate
    # questions with separate members behind them.
    let rows = named(u, TyKind.Class, "Rows", _decl("Rows", ["iter"]))
    assert_true(is_iterable(u, rows) == u.t_any)
    assert_true(is_indexable(u, rows) is null)
}

fn test_a_getitem_is_found_through_a_trait() -> void {
    let u = Universe()
    let provider = _decl("Index", ["__getitem__"])
    let implementer = _decl("Table", [])
    implementer.traits.push(provider)
    let table = named(u, TyKind.Trait, "Table", implementer)
    assert_true(is_indexable(u, table) == (u.t_any, u.t_any))
}

fn test_every_answer_is_an_interned_type() -> void {
    # Identity is the relation the whole checker leans on, so an answer built
    # here has to be the universe's type and not a structural twin of it.
    let u = Universe()
    assert_true(is_iterable(u, tuple_of(u, [u.t_int, u.t_str])) == union_of(u, [u.t_int, u.t_str]))
    assert_true(is_indexable(u, dict_of(u, u.t_str, u.t_int)) == (u.t_str, u.t_int))
    assert_true(is_iterable(u, list_of(u, list_of(u, u.t_int))) == list_of(u, u.t_int))
}

fn test_the_module_doc_example_is_true() -> void {
    # The header's fenced block sits above the imports, where `--doc` does not
    # reach it, so its three claims are asserted here instead.
    let u = Universe()
    assert_true(is_numeric(u.t_float))
    assert_true(is_optional(u.t_any))
    assert_false(is_callable(u.t_str))
}
