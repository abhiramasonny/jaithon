#: What the compiled tiers know about one operand-stack entry -- that it is the
#: literal 2.0, or the int k -- holds along one edge. An `if` expression used
#: as an operand ends with its arms meeting in one entry that holds a
#: different value per arm, so the facts must be dropped at the join. Kept,
#: `1.0 * (if x <= 0.5 { 7.0 } else { 2.0 })` took the else arm's 2.0 for both
#: arms, compiled `1.0 + 1.0`, and returned 2.0 for every x.
from std.test import assert_eq

fn scaled(x: float) -> float { return 1.0 * (if x <= 0.5 { 7.0 } else { 2.0 }) }
fn squared_or_doubled(x: float) -> float { return x * (if x <= 0.5 { x } else { 2.0 }) }
fn negated(x: float) -> float { return (-x) * (if x <= 0.5 { 7.0 } else { 2.0 }) }
fn shifted(x: float, k: float) -> float { return (x + 1.0) * (if x < k { x } else { 2.0 }) }
fn picked(x: int) -> int { return 3 * (if x <= 5 { 7 } else { 2 }) }

fn ranged(c: bool, n: int) -> int {
    var s = 0
    for i in (if c { 0 } else { 5 })..n { s += i }
    return s
}

fn test_a_float_if_expression_operand_keeps_both_arms() -> void {
    for _round in 0..300 {
        assert_eq(scaled(-3.0), 7.0)
        assert_eq(scaled(3.0), 2.0)
        assert_eq(scaled(0.25), 7.0)
        assert_eq(squared_or_doubled(-3.0), 9.0)
        assert_eq(squared_or_doubled(3.0), 6.0)
        assert_eq(negated(-3.0), 21.0)
        assert_eq(negated(3.0), -6.0)
        assert_eq(shifted(-3.0, 0.5), 6.0)
        assert_eq(shifted(3.0, 0.5), 8.0)
    }
}

fn test_an_int_if_expression_operand_keeps_both_arms() -> void {
    for _round in 0..300 {
        assert_eq(picked(-3), 21)
        assert_eq(picked(9), 6)
        assert_eq(ranged(true, 10), 45)
        assert_eq(ranged(false, 10), 35)
    }
}
