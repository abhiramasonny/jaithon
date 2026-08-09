#: `jaithon.compile.check.kinds` — the semantic type kinds and their ordinals.
#:
#: The ordinal is not a detail. Union members are sorted by it, the sort decides
#: the rendered union string, and the rendered string is what the sema diff
#: compares between the two front ends. These tests pin the order against
#: `src/sema/types.h` so a reordering fails here rather than as an unexplained
#: divergence on an unrelated file.
from std.test import assert_eq, assert_true
from jaithon.compile.check.kinds import FN_KWREST, FN_VARIADIC, JOIN_MAX_MEMBERS
from jaithon.compile.check.kinds import KINDS, MAX_RENDER_DEPTH, PRIMITIVE_SPELLING
from jaithon.compile.check.kinds import TyKind, ordinal

fn test_there_are_twenty_one_kinds() -> void {
    assert_eq(KINDS.len(), 21)
}

fn test_ordinals_match_the_c_enum_position_for_position() -> void {
    # src/sema/types.h:14-36, in declaration order. Written out rather than
    # derived, so that this test disagrees with KINDS when either moves.
    let expected = [
        (TyKind.Any, 0),
        (TyKind.Never, 1),
        (TyKind.Null, 2),
        (TyKind.Bool, 3),
        (TyKind.Int, 4),
        (TyKind.Float, 5),
        (TyKind.Str, 6),
        (TyKind.Bytes, 7),
        (TyKind.List, 8),
        (TyKind.Dict, 9),
        (TyKind.Set, 10),
        (TyKind.Tuple, 11),
        (TyKind.Range, 12),
        (TyKind.Fn, 13),
        (TyKind.Class, 14),
        (TyKind.Trait, 15),
        (TyKind.Enum, 16),
        (TyKind.Module, 17),
        (TyKind.Union, 18),
        (TyKind.GenericParam, 19),
        (TyKind.Alias, 20),
    ]
    for (kind, position) in expected {
        assert_eq(ordinal(kind), position)
    }
}

fn test_every_kind_has_exactly_one_ordinal() -> void {
    var seen: set[int] = set()
    for kind in KINDS {
        seen.add(ordinal(kind))
    }
    assert_eq(seen.len(), KINDS.len())
}

fn test_the_orders_that_the_measured_union_renderings_imply() -> void {
    # `float | bool | int` renders `bool | int | float`, so BOOL < INT < FLOAT.
    assert_true(ordinal(TyKind.Bool) < ordinal(TyKind.Int))
    assert_true(ordinal(TyKind.Int) < ordinal(TyKind.Float))
    # `list[int] | dict[str, int] | set[int]` renders unchanged, so the three
    # are already in ordinal order.
    assert_true(ordinal(TyKind.List) < ordinal(TyKind.Dict))
    assert_true(ordinal(TyKind.Dict) < ordinal(TyKind.Set))
    # Null sorts near the front but is forced last when rendered; the ordinal
    # is what makes that a special case rather than a consequence.
    assert_true(ordinal(TyKind.Null) < ordinal(TyKind.Int))
}

fn test_the_primitive_spellings() -> void {
    assert_eq(PRIMITIVE_SPELLING[TyKind.Any], "any")
    assert_eq(PRIMITIVE_SPELLING[TyKind.Int], "int")
    assert_eq(PRIMITIVE_SPELLING[TyKind.Float], "float")
    assert_eq(PRIMITIVE_SPELLING[TyKind.Str], "str")
    assert_eq(PRIMITIVE_SPELLING[TyKind.Bool], "bool")
    assert_eq(PRIMITIVE_SPELLING[TyKind.Bytes], "bytes")
    assert_eq(PRIMITIVE_SPELLING[TyKind.Range], "range")
    assert_eq(PRIMITIVE_SPELLING[TyKind.Null], "null")
    # A composite renders from its parts and so has no entry here.
    assert_true(TyKind.List not in PRIMITIVE_SPELLING)
    assert_true(TyKind.Class not in PRIMITIVE_SPELLING)
}

fn test_the_caps_match_the_c() -> void {
    assert_eq(FN_VARIADIC, 1)
    assert_eq(FN_KWREST, 2)
    assert_eq(JOIN_MAX_MEMBERS, 8)
    assert_eq(MAX_RENDER_DEPTH, 32)
}
