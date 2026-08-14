#: Type-system behaviour that must hold at run time (spec §2).
from std.test import assert_eq, assert_true, assert_false, assert_throws

fn test_int_and_float_are_distinct() -> void {
    assert_eq(type_of(1), "int")
    assert_eq(type_of(1.0), "float")
    assert_eq(type_of(1 / 1), "float")
    assert_eq(type_of(3 // 2), "int")
    assert_eq(type_of(1 + 1), "int")
    assert_eq(type_of(1 + 1.0), "float")
    assert_eq(1 + 1.0, 2.0)
}

fn test_numeric_equality_crosses_types() -> void {
    assert_true(1 == 1.0)
    assert_true(0 == 0.0)
    assert_false(1 == 1.5)
    assert_eq(hash(2), hash(2.0))
}

fn test_integer_overflow_raises() -> void {
    let big = 9_223_372_036_854_775_807
    assert_throws(OverflowError, || big + 1)
    assert_throws(OverflowError, || big * 2)
}

fn test_wrapping_operators_do_not_raise() -> void {
    let big = 9_223_372_036_854_775_807
    assert_eq(big +% 1, (-9_223_372_036_854_775_808))
}

fn test_floor_division_and_modulo_follow_the_divisor() -> void {
    assert_eq(7 // 2, 3)
    assert_eq(-7 // 2, -4)
    assert_eq(7 // -2, -4)
    assert_eq(7 % 3, 1)
    assert_eq(-7 % 3, 2)
    assert_eq(7 % -3, -2)
}

fn test_division_by_zero_raises() -> void {
    assert_throws(DivisionByZeroError, || 1 / 0)
    assert_throws(DivisionByZeroError, || 1 // 0)
    assert_throws(DivisionByZeroError, || 1 % 0)
}

fn test_power_is_right_associative() -> void {
    assert_eq(2 ** 3 ** 2, 512)
    assert_eq((2 ** 3) ** 2, 64)
}

fn test_unary_minus_binds_looser_than_power() -> void {
    assert_eq(-2 ** 2, -4)
}

fn test_optional_is_a_union_with_null() -> void {
    let a: int? = null
    let b: int? = 5
    assert_eq(a ?? -1, -1)
    assert_eq(b ?? -1, 5)
    assert_true(a == null)
    assert_false(b == null)
}

fn test_type_guard_on_any_boundary() -> void {
    fn dynamic() -> any { return "text" }
    assert_throws(TypeError, fn() {
        let n: int = dynamic()
        return n
    })
}

fn test_is_checks_identity_not_equality() -> void {
    let a = [1, 2]
    let b = [1, 2]
    assert_true(a == b)
    assert_false(a is b)
    assert_true(a is a)
}

fn test_isinstance_walks_the_hierarchy() -> void {
    let e = FileNotFoundError("nope")
    assert_true(isinstance(e, FileNotFoundError))
    assert_true(isinstance(e, IOError))
    assert_true(isinstance(e, Error))
    assert_false(isinstance(e, ValueError))
}

fn test_bitwise_is_int_only() -> void {
    assert_eq(0b1100 & 0b1010, 0b1000)
    assert_eq(0b1100 | 0b1010, 0b1110)
    assert_eq(0b1100 ^ 0b1010, 0b0110)
    assert_eq(~0, -1)
    assert_eq(1 << 62, 4611686018427387904)
    assert_eq(-8 >> 1, -4)
    let not_an_int: any = 1.0
    assert_throws(TypeError, || not_an_int & 1)
}
