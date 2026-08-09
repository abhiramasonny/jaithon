#: `jaithon.compile.check.render` — the canonical text of a type.
#:
#: Every expected string here was measured by running
#: the C checker over a probe file and reading what it
#: actually emitted, not by reading `src/sema/types.c` and believing it. The
#: retired sema dump compared these strings across the two front ends, so each case
#: that disagrees is a false divergence on every program that mentions the
#: type.
#:
#: Types are built directly here rather than through the universe: rendering is
#: a pure function of its argument, and testing it without an intern table is
#: what lets it be verified before the rest of the port exists.
from std.test import assert_eq
from jaithon.compile.check.kinds import FN_KWREST, FN_VARIADIC, TyKind
from jaithon.compile.check.render import render
from jaithon.compile.check.ty import Type

fn _prim(kind: TyKind) -> Type {
    return Type(kind)
}

let ANY = _prim(TyKind.Any)
let NEVER = _prim(TyKind.Never)
let NULL = _prim(TyKind.Null)
let BOOL = _prim(TyKind.Bool)
let INT = _prim(TyKind.Int)
let FLOAT = _prim(TyKind.Float)
let STR = _prim(TyKind.Str)
let BYTES = _prim(TyKind.Bytes)
let RANGE = _prim(TyKind.Range)

fn _list(elem: Type) -> Type { return Type(TyKind.List, null, [elem]) }

fn _set(elem: Type) -> Type { return Type(TyKind.Set, null, [elem]) }

fn _dict(key: Type, value: Type) -> Type { return Type(TyKind.Dict, null, [key, value]) }

fn _tuple(members: list[Type]) -> Type { return Type(TyKind.Tuple, null, members) }

fn _union(members: list[Type]) -> Type { return Type(TyKind.Union, null, members) }

fn _fn(params: list[Type], returns: Type?, flags: int = 0) -> Type {
    return Type(TyKind.Fn, null, params, returns, flags)
}

fn _named(kind: TyKind, name: str?, args: list[Type] = []) -> Type {
    return Type(kind, name, args)
}

fn test_the_primitives() -> void {
    assert_eq(render(ANY), "any")
    assert_eq(render(NEVER), "never")
    assert_eq(render(NULL), "null")
    assert_eq(render(BOOL), "bool")
    assert_eq(render(INT), "int")
    assert_eq(render(FLOAT), "float")
    assert_eq(render(STR), "str")
    assert_eq(render(BYTES), "bytes")
    assert_eq(render(RANGE), "range")
}

fn test_a_missing_type_still_prints() -> void {
    assert_eq(render(null), "<unknown>")
}

fn test_the_containers() -> void {
    assert_eq(render(_list(INT)), "list[int]")
    assert_eq(render(_set(FLOAT)), "set[float]")
    assert_eq(render(_dict(STR, INT)), "dict[str, int]")
}

fn test_containers_nest() -> void {
    assert_eq(render(_list(_dict(STR, _list(INT)))), "list[dict[str, list[int]]]")
    assert_eq(render(_dict(STR, _tuple([INT, FLOAT]))), "dict[str, (int, float)]")
}

fn test_a_tuple_is_parenthesised_not_bracketed() -> void {
    # Written `tuple[int, str]`, rendered `(int, str)`. A one-member tuple has
    # no trailing comma and an empty one is `()`.
    assert_eq(render(_tuple([INT, STR])), "(int, str)")
    assert_eq(render(_tuple([INT])), "(int)")
    assert_eq(render(_tuple([])), "()")
}

fn test_the_functions() -> void {
    assert_eq(render(_fn([], NULL)), "fn() -> void")
    assert_eq(render(_fn([INT], STR)), "fn(int) -> str")
    assert_eq(render(_fn([INT, FLOAT], _list(STR))), "fn(int, float) -> list[str]")
}

fn test_a_function_with_no_written_return_is_any() -> void {
    # Null `ret` means none was written; a `Null` ret is what `-> void` means.
    assert_eq(render(_fn([INT], null)), "fn(int) -> any")
    assert_eq(render(_fn([INT], NULL)), "fn(int) -> void")
}

fn test_the_variadic_and_kwrest_markers() -> void {
    # Neither marker carries a trailing space.
    assert_eq(render(_fn([INT, _list(STR)], NULL, FN_VARIADIC)), "fn(int, ...list[str]) -> void")
    assert_eq(
        render(_fn([INT, _dict(STR, ANY)], NULL, FN_KWREST)),
        "fn(int, **dict[str, any]) -> void"
    )
    # With both, `...` sits one before the `**`, because the variadic index is
    # keyed on whether a kwrest parameter is present rather than on the flag.
    assert_eq(
        render(_fn([INT, _list(STR), _dict(STR, ANY)], NULL, FN_VARIADIC | FN_KWREST)),
        "fn(int, ...list[str], **dict[str, any]) -> void"
    )
}

fn test_the_named_types() -> void {
    assert_eq(render(_named(TyKind.Class, "Box")), "Box")
    assert_eq(render(_named(TyKind.Trait, "Show")), "Show")
    assert_eq(render(_named(TyKind.Enum, "Color")), "Color")
    assert_eq(render(_named(TyKind.GenericParam, "T")), "T")
}

fn test_a_generic_named_type_carries_its_arguments() -> void {
    assert_eq(render(_named(TyKind.Class, "Box", [INT])), "Box[int]")
    assert_eq(render(_named(TyKind.Class, "Pair", [INT, STR])), "Pair[int, str]")
}

fn test_an_unnamed_named_type() -> void {
    assert_eq(render(_named(TyKind.Class, null)), "<anonymous>")
}

fn test_a_module_is_named_rather_than_spelled() -> void {
    assert_eq(render(_named(TyKind.Module, "std.str")), "module std.str")
    assert_eq(render(_named(TyKind.Module, null)), "module <anonymous>")
}

fn test_a_two_member_union_with_null_is_the_optional_sugar() -> void {
    assert_eq(render(_union([NULL, INT])), "int?")
    assert_eq(render(_union([INT, NULL])), "int?")
    assert_eq(render(_union([NULL, _list(INT)])), "list[int]?")
    assert_eq(render(_union([NULL, _dict(STR, INT)])), "dict[str, int]?")
    assert_eq(render(_union([NULL, _named(TyKind.Class, "Box")])), "Box?")
}

fn test_the_sugar_is_refused_for_a_function_or_a_union() -> void {
    # `fn() -> int?` would read back as a function returning `int?`.
    assert_eq(render(_union([NULL, _fn([], INT)])), "fn() -> int | null")
    assert_eq(render(_union([NULL, _union([INT, STR])])), "int | str | null")
}

fn test_null_is_rendered_last_whatever_position_it_holds() -> void {
    # Members arrive in intern order, which sorts null near the front. It is
    # moved to the end when rendered.
    assert_eq(render(_union([NULL, INT, STR])), "int | str | null")
    assert_eq(render(_union([INT, NULL, STR])), "int | str | null")
    assert_eq(render(_union([INT, STR, NULL])), "int | str | null")
}

fn test_a_union_renders_its_members_in_the_order_it_holds_them() -> void {
    # render does not sort. The sort happens when the union is interned, and
    # these are the orders the C was measured to produce.
    assert_eq(render(_union([BOOL, INT, FLOAT])), "bool | int | float")
    assert_eq(render(_union([INT, FLOAT, STR])), "int | float | str")
    assert_eq(
        render(_union([_list(INT), _dict(STR, INT), _set(INT)])),
        "list[int] | dict[str, int] | set[int]"
    )
}

fn test_a_union_of_one_still_renders() -> void {
    assert_eq(render(_union([INT])), "int")
}

fn test_a_deeply_nested_type_is_truncated() -> void {
    # The top level is depth 0 and the guard is strictly greater than 32, so
    # the first `...` appears at the thirty-fourth level. Thirty-three nested
    # lists therefore still render in full.
    var deep = INT
    for _step in range(40) { deep = _list(deep) }
    let text = render(deep)
    assert_eq(text.contains("..."), true)
    assert_eq(text.starts_with("list[list["), true)
}
