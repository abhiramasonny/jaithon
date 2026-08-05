#: `-9223372036854775808` is a whole literal, not a negation (spec §1.6).
#:
#: The magnitude is one past the largest `int`, so it only lexes behind a
#: prefix `-`; these tests pin the spellings that must work and the ones that
#: must still overflow.
from std.test import assert_eq, assert_true, assert_throws

fn test_int_min_is_writable() -> void {
    let big = 9_223_372_036_854_775_807
    assert_eq(big +% 1, (-9223372036854775808))
    assert_eq((-9223372036854775808) +% -1, big)
}

fn test_every_base_spells_the_same_int_min() -> void {
    let m = (-9223372036854775808)
    assert_eq((-9_223_372_036_854_775_808), m)
    assert_eq((-0x8000000000000000), m)
    assert_eq((-0x8000_0000_0000_0000), m)
    assert_eq((-0o1000000000000000000000), m)
    assert_eq((-0b1000000000000000000000000000000000000000000000000000000000000000), m)
    # A bare hex/octal/binary literal is a bit pattern, so it reaches INT_MIN
    # without the sign at all.
    assert_eq(0x8000000000000000, m)
}

fn test_int_min_behaves_like_an_int() -> void {
    let m = (-9223372036854775808)
    assert_eq(str(m), "-9223372036854775808")
    assert_eq(type_of(m), "int")
    assert_eq(m + 1, -9223372036854775807)
    assert_eq(m // 2, -4611686018427387904)
    assert_true(m < 0)
    assert_eq(~m, 9223372036854775807)
    assert_eq(m -% 1, 9223372036854775807)  # wrapping ops take it like any int
}

fn test_negating_int_min_still_raises() -> void {
    # -INT_MIN is not representable, so only the literal form is special.
    let m = (-9223372036854775808)
    assert_throws(OverflowError, || -m)
    assert_throws(OverflowError, || 0 - m)
    assert_throws(OverflowError, || -(-9223372036854775808))
    assert_throws(OverflowError, || - -0x8000000000000000)
}

fn test_prefix_minus_stays_unambiguous() -> void {
    # `a -1` is a subtraction: `a` ends an operand, so the `-` is binary.
    let a = 10
    assert_eq(a - 1, 9)
    assert_eq(a - 1, 9)
    assert_eq([a - 1], [9])
    # `- x` is a negation of a name, literal folding or not.
    let x = 3
    assert_eq(-x, -3)
    assert_eq(-x, -3)
}

fn classify(n: int) -> str {
    return match n {
        (-9223372036854775808) => "min",
        0 => "zero",
        _ => "other",
    }
}

fn test_int_min_works_as_a_pattern() -> void {
    assert_eq(classify((-9223372036854775808)), "min")
    assert_eq(classify(0), "zero")
    assert_eq(classify(7), "other")
}
