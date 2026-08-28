#: The unary operators, none of which had an arm.
#:
#: `-x`, `not x`, `~x` and unary `+x` all fell through to `default` and ENDED
#: THE WALK -- everything after one ran interpreted. Measured in one binary
#: through `JAITHON_JIT_NEGATE`:
#:
#:   -(i & 255)          48,000,320 -> 2,310,930    0.09s -> 0.01s   9x
#:   -x on floats        38,000,322 ->   253,666    0.10s -> 0.01s  10x
#:   not a               23,107,814 -> 1,450,119    0.08s -> 0.01s   8x
#:   ~(i & 255)          12,169,798 ->    52,249    0.06s -> 0.01s   6x
#:
#: The switched-off side of that A/B takes the same unarmed path the `default`
#: case used to, so these are ratios against no arm at all rather than against
#: a declined body -- see the note above `unarmedOpcode`.
#:
#: `not` is an xor with one, because SLOT_BOOL's contract is "0 or 1 in a
#: register" and the interpreter REQUIREs a bool. `~x` is an xor with -1. Unary
#: `+` on a number emits nothing at all -- the interpreter checks the type and
#: does nothing else -- and the whole point of the arm is that the walk does not
#: stop there.
#:
#: OP_NEG was the one that surfaced the class: it appeared only in
#: `inlinableBody`'s whitelist, so an ordinary body fell through to `default`.
#:
#: It hid because `emitUnarmedDeopt` is silent by design -- the point of it is
#: to interpret from an unsupported opcode ONWARD rather than decline the whole
#: body, so nothing is printed. What surfaced it was a different message
#: entirely: `std.math`'s `is_inf` is `x == INF or x == -INF`, whose `or`
#: records a forward branch to its own OP_RETURN before the walk reaches the
#: negation, and once the walk stops that branch has nowhere to land. The fixup
#: pass now names the opcode that ended the walk, which turned a baffling
#: "a branch to offset 25" into "OP_NEG at 23".
#:
#: `is_inf` compiles now, and with it `_require_finite` and so every guarded
#: function in std.math: the `math.sin` loop went from 53,464,406 interpreted
#: instructions to 30,051,073, and `docs/probes/p27_float_math.jai` from 0.43s
#: to 0.38s.
# jit-compiles: negate_ints, negate_floats, alternating, flip_bools, complement
from std.test import assert_eq, assert_true, assert_throws
import std.math

let CALLS = 100

fn negate_ints(rounds: int) -> int {
    var t = 0
    var i = 0
    while i < rounds {
        let a = -(i & 15)
        t += -a
        i += 1
    }
    return t
}

fn negate_floats(rounds: int) -> float {
    var t = 0.0
    var i = 0
    while i < rounds {
        let x = float(i & 15) + 1.5
        t += -x + -(x * 2.0)
        i += 1
    }
    return t
}

#: The shape that found it: an `or` whose short circuit branches forward past a
#: negation.
fn alternating(x: float, rounds: int) -> int {
    var seen = 0
    var i = 0
    while i < rounds {
        if x == 1.0e308 or x == -1.0e308 { seen += 1 }
        i += 1
    }
    return seen
}

fn flip_bools(rounds: int) -> int {
    var seen = 0
    var i = 0
    while i < rounds {
        let even = i & 1 == 0
        if not even { seen += 1 }
        i += 1
    }
    return seen
}

fn complement(rounds: int) -> int {
    var t = 0
    var i = 0
    while i < rounds {
        t += ~(i & 15) + 16
        i += 1
    }
    return t
}

pub fn test_not_is_an_xor_with_one() -> void {
    var seen = 0
    var call = 0
    while call < CALLS {
        seen += flip_bools(10)
        call += 1
    }
    assert_eq(seen, 5 * CALLS)
}

pub fn test_complement_is_an_xor_with_minus_one() -> void {
    var total = 0
    var call = 0
    while call < CALLS {
        #: ~x + 16 for x in 0..9 is 15 - x, summing to 105.
        total += complement(10)
        call += 1
    }
    assert_eq(total, 105 * CALLS)
}

pub fn test_unary_plus_keeps_the_value() -> void {
    assert_eq(+ -7, -7, "int")
    assert_true(+ -2.5 == -2.5, "float")
}

pub fn test_integer_negation() -> void {
    var total = 0
    var call = 0
    while call < CALLS {
        #: 0..9 negated twice is 45.
        total += negate_ints(10)
        call += 1
    }
    assert_eq(total, 45 * CALLS)
}

pub fn test_float_negation() -> void {
    var total = 0.0
    var call = 0
    while call < CALLS {
        total += negate_floats(10)
        call += 1
    }
    #: -(x) + -(2x) = -3x, and x runs 1.5 .. 10.5, so -3 * 60 = -180 a call.
    assert_true(total < -17999.0 and total > -18001.0, "-180 * CALLS")
}

pub fn test_a_branch_over_a_negation() -> void {
    var seen = 0
    var call = 0
    while call < CALLS {
        seen += alternating(2.5, 10)
        call += 1
    }
    assert_eq(seen, 0)
}

pub fn test_the_one_int_that_cannot_be_negated() -> void {
    #: INT64_MIN is its own negation, and the interpreter raises rather than
    #: wrapping. The compiled form guards on the overflow flag and resumes at
    #: the START of the instruction -- which is why the negated value goes to a
    #: scratch and reaches the operand's register only once the guard has
    #: passed. Writing it first made the interpreter negate an already-negated
    #: value and `-5` came back as `5`, visible only under deopt stress.
    assert_throws(OverflowError, fn() -> void {
        var i = 0
        var t = 0
        while i < 200 {
            t += -(-9223372036854775807 - 1)
            i += 1
        }
    }, "INT64_MIN")
}

pub fn test_the_float_edges() -> void {
    assert_true(math.is_nan(-math.NAN), "NaN stays NaN")
    assert_eq(str(-0.0), "-0.0", "negative zero")
    assert_true(-math.INF < 0.0, "-INF is negative")
    assert_true(math.is_inf(-math.INF), "-INF is still infinite")
}
