#: Printing a type back as source, where `?` needs the parentheses it was
#: written with.
#:
#: `?` binds tighter than a function type's arrow and than a union's bar, so
#: `(fn(float) -> float)?` and `fn(float) -> float?` are two different types
#: spelt with the same characters in a different order. `TypeExpr.to_source`
#: dropped the parentheses, which made the first print as the second -- and
#: `fmt` rewrites a whole file through that printer, so formatting a file with
#: an optional function parameter did not merely reformat it, it changed what
#: the parameter meant and the checker then rejected the default `null`.
#:
#: Everything here is checked two ways: the FUNCTION still behaves as its
#: signature says, and `to_source` still round-trips. The first is what a user
#: sees, the second is what `fmt` depends on.
from std.test import assert_eq, assert_true
from jaithon.ast import TypeExpr

#: An optional function: null is allowed, and calling it returns a float.
fn optional_callback(g: (fn(float) -> float)? = null) -> float {
    if g is null { return -1.0 }
    return g(2.0)
}

#: A function returning an optional: the argument is required, its RESULT is
#: what may be null. Same characters, other meaning.
fn callback_returning_optional(g: fn(float) -> float?) -> float {
    return g(2.0) ?? -1.0
}

fn optional_union(v: (int | str)? = null) -> str {
    if v is null { return "none" }
    return "some"
}

fn test_an_optional_function_parameter_accepts_null_and_a_function() -> void {
    assert_eq(optional_callback(), -1.0, "omitted")
    assert_eq(optional_callback(g: fn(v: float) -> float { return v * 3.0 }), 6.0, "given")
}

fn test_a_function_returning_an_optional_is_the_other_type() -> void {
    assert_eq(callback_returning_optional(fn(v: float) -> float? { return null }), -1.0, "null")
    assert_eq(callback_returning_optional(fn(v: float) -> float? { return v + 1.0 }), 3.0, "value")
}

fn test_an_optional_union_accepts_null_and_a_member() -> void {
    assert_eq(optional_union(), "none", "omitted")
    assert_eq(optional_union(v: 3), "some", "given")
}

fn test_to_source_keeps_the_parentheses_an_optional_needs() -> void {
    let float_type = TypeExpr.named("float")
    let signature = TypeExpr.function([float_type], float_type)

    assert_eq(signature.to_source(), "fn(float) -> float", "the bare function type")
    assert_eq(
        TypeExpr.optional(signature).to_source(),
        "(fn(float) -> float)?",
        "an optional function keeps its parentheses"
    )
    assert_eq(
        TypeExpr.function([float_type], TypeExpr.optional(float_type)).to_source(),
        "fn(float) -> float?",
        "a function returning an optional needs none"
    )
    assert_eq(
        TypeExpr.optional(TypeExpr.named("int")).to_source(),
        "int?",
        "an ordinary optional is unchanged"
    )
}
