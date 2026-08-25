#: Comparing two bools, in a loop hot enough to reach the compiled tier.
#:
#: The tier had no arm for it. Both operands are 0 or 1 in a register, so `cmp`
#: answers `==` and `!=` directly and the ordering conditions come out right as
#: well -- but the arm simply was not there, so a body containing one declined
#: WHOLE. `let a = ys[i - 1] > py` / `let b = ys[i] > py` / `if a != b` is how
#: a point-in-polygon crossing test is written, and jaicv's
#: `nearest_gap_squared` carried a comment saying it had to be spelt as two
#: nested `if`s, twice, to get compiled at all.
#:
#: These check the ANSWER: every pair of bools through every comparison, and
#: the crossing-count shape that found it, against the same count computed the
#: long way round. Run with and without JAITHON_NO_JIT.
# jit-compiles: count_crossings, count_matching_flags
from std.test import assert_eq, assert_true

let REPS = 60

fn count_crossings(ys: list[float], py: float, rounds: int) -> int {
    var crossings = 0
    var r = 0
    while r < rounds {
        var index = 1
        while index < ys.len() {
            let above = ys[index - 1] > py
            let next_above = ys[index] > py
            if above != next_above { crossings += 1 }
            index += 1
        }
        r += 1
    }
    return crossings
}

#: The same answer without ever comparing two bools, which is the shape the
#: refusal forced and the reference this is checked against.
fn count_crossings_nested(ys: list[float], py: float, rounds: int) -> int {
    var crossings = 0
    var r = 0
    while r < rounds {
        var index = 1
        while index < ys.len() {
            if ys[index - 1] > py {
                if ys[index] <= py { crossings += 1 }
            } else {
                if ys[index] > py { crossings += 1 }
            }
            index += 1
        }
        r += 1
    }
    return crossings
}

fn build(n: int) -> list[float] {
    var ys: list[float] = []
    var index = 0
    while index < n {
        ys.push(float(index % 17))
        index += 1
    }
    return ys
}

fn test_a_bool_comparison_counts_the_same_crossings() -> void {
    let ys = build(20_000)
    assert_eq(count_crossings(ys, 8.0, REPS), count_crossings_nested(ys, 8.0, REPS), "crossings")
    assert_eq(count_crossings(ys, -1.0, REPS), 0, "a level below everything")
    assert_eq(count_crossings(ys, 99.0, REPS), 0, "a level above everything")
}

#: All four bool pairs through all six comparisons, in a loop, so the compiled
#: arm answers each of them rather than one of them many times. `<` and `>` on
#: bools are legal here and order false below true, which is what 0 and 1 in a
#: register give for free -- checked so that stays true rather than being an
#: accident of the encoding.
fn count_matching_flags(rounds: int) -> list[int] {
    var eq = 0
    var ne = 0
    var lt = 0
    var le = 0
    var gt = 0
    var ge = 0
    var r = 0
    while r < rounds {
        var index = 0
        while index < 4 {
            let a = index // 2 == 1
            let b = index % 2 == 1
            if a == b { eq += 1 }
            if a != b { ne += 1 }
            if a < b { lt += 1 }
            if a <= b { le += 1 }
            if a > b { gt += 1 }
            if a >= b { ge += 1 }
            index += 1
        }
        r += 1
    }
    return [eq, ne, lt, le, gt, ge]
}

fn test_every_comparison_of_two_bools() -> void {
    #: Per round over the four pairs (F,F) (F,T) (T,F) (T,T):
    #: eq 2, ne 2, lt 1, le 3, gt 1, ge 3.
    let wanted = [2, 2, 1, 3, 1, 3]
    assert_eq(count_matching_flags(1), wanted, "one round, cold")
    let rounds = 40_000
    let counted = count_matching_flags(rounds)
    for index in 0..wanted.len() {
        assert_eq(counted[index], wanted[index] * rounds, f"comparison {index}, hot")
    }
}

#: The unfused form: the comparison's result is bound rather than branched on,
#: so it reaches the plain OP_EQ arm instead of the fused compare-and-branch.
#: Both needed the arm; only the fused one is exercised by the tests above.
fn count_unfused(rounds: int) -> int {
    var total = 0
    var r = 0
    while r < rounds {
        var index = 0
        while index < 4 {
            let a = index // 2 == 1
            let b = index % 2 == 1
            let same = a == b
            let different = a != b
            if same { total += 1 }
            if different { total += 2 }
            index += 1
        }
        r += 1
    }
    return total
}

fn test_an_unbranched_bool_comparison_answers_the_same() -> void {
    #: Two pairs are equal and two differ, so each round adds 2 * 1 + 2 * 2.
    assert_eq(count_unfused(40_000), 6 * 40_000, "the unfused form")
}

#: A float comparison whose result is BOUND rather than branched on reaches the
#: unfused arm, which had to release its float borrows before recording the NaN
#: guard exactly as the fused one already did. Without that the loop above
#: declined with "a float borrow reached OP_GT's guard" the moment the bool arm
#: let it get that far.
fn test_a_bound_float_comparison_compiles_too() -> void {
    let ys = build(20_000)
    var above = 0
    var r = 0
    while r < REPS {
        var index = 0
        while index < ys.len() {
            let over = ys[index] > 8.0
            if over { above += 1 }
            index += 1
        }
        r += 1
    }
    assert_true(above > 0, "some values are above the level")
    assert_eq(above % REPS, 0, "the same count every round")
}
