#: The wrapping operators `+% -% *%`, which had no arm at all.
#:
#: OP_ADD_WRAP/OP_SUB_WRAP/OP_MUL_WRAP fell through to `default`, so the walk
#: STOPPED at the first one and everything after it ran interpreted. It reached
#: the standard library: `_parse_int_text` in the lexer is
#: `value = value *% base +% _hex_value(character)` and reported
#: `walked only to OP_MUL_WRAP at 205` on a plain run of any program at all.
#:
#: The arms are the checked ones with the guard DELETED rather than relaxed.
#: `+%` is the interpreter's `(int64_t)((uint64_t)x + (uint64_t)y)`, which is
#: the low 64 bits `add` already writes, so there is nothing left to test and
#: no `b.vs` to take. `*%` drops OP_MUL's `smulh` with it: the high half only
#: ever existed to be compared against the low half's replicated sign, and
#: without a guard nobody reads it.
#:
#: What the edges are for. Every case below is a value where wrapping and
#: raising disagree, so an arm that quietly kept a guard, or that computed a
#: saturating or 32-bit answer, fails here rather than somewhere subtler:
#: `INT_MAX +% 1` is INT_MIN, `INT_MIN *% -1` is INT_MIN again (the one product
#: that is its own negation), and `INT_MAX *% INT_MAX` is 1.
#:
#: The last two tests are the refusals. `+%` is a 64-bit integer operator: a
#: float on either side is a TypeError and an object with `__add__` is a call,
#: and both have to reach the interpreter. The float one arrives 40,000
#: iterations in, so the body is already compiled and specialised to int when
#: it turns up -- which is the case that would be a wrong answer rather than a
#: decline if the arm emitted integer arithmetic on whatever the register held.
# jit-compiles: wrap_add, wrap_sub, wrap_mul, wrap_round_trip, wrap_double, wrap_negate, wrap_hash, wrap_in_a_try
from std.test import assert_eq, assert_throws
import std.math

let CALLS = 40
let ROUNDS = 12

#: Each helper recomputes the same operation ROUNDS times so the loop is hot,
#: and is called CALLS times so the whole-function tier gets it too. The value
#: returned is one application of the operator, not an accumulation, so the
#: assertion below is the wrapped answer itself.
fn wrap_add(a: int, b: int, rounds: int) -> int {
    var r = 0
    var i = 0
    while i < rounds {
        r = a +% b
        i += 1
    }
    return r
}

fn wrap_sub(a: int, b: int, rounds: int) -> int {
    var r = 0
    var i = 0
    while i < rounds {
        r = a -% b
        i += 1
    }
    return r
}

fn wrap_mul(a: int, b: int, rounds: int) -> int {
    var r = 0
    var i = 0
    while i < rounds {
        r = a *% b
        i += 1
    }
    return r
}

#: Identities that hold in two's complement for every pair, overflowing or not.
#: They are the part of the test that does not depend on a hand-computed
#: constant being right.
fn wrap_round_trip(a: int, b: int, rounds: int) -> int {
    var bad = 0
    var i = 0
    while i < rounds {
        if (a +% b) -% b != a { bad += 1 }
        if (a -% b) +% b != a { bad += 1 }
        if a *% b != b *% a { bad += 1 }
        i += 1
    }
    return bad
}

fn wrap_double(a: int, rounds: int) -> int {
    var bad = 0
    var i = 0
    while i < rounds {
        if a *% 2 != a +% a { bad += 1 }
        i += 1
    }
    return bad
}

fn wrap_negate(a: int, rounds: int) -> int {
    var bad = 0
    var i = 0
    while i < rounds {
        if a *% -1 != 0 -% a { bad += 1 }
        i += 1
    }
    return bad
}

#: The stdlib shape that surfaced this: a multiply and an add chained, with the
#: accumulator carried across iterations so the wrap is real rather than
#: recomputed. Same recurrence `_parse_int_text` runs.
fn wrap_hash(seed: int, rounds: int) -> int {
    var h = seed
    var i = 0
    while i < rounds {
        h = h *% 6364136223846793005 +% 1442695040888963407
        h = h -% i
        i += 1
    }
    return h
}

pub fn test_add_wraps_at_both_ends() -> void {
    var call = 0
    while call < CALLS {
        assert_eq(wrap_add(math.INT_MAX, 1, ROUNDS), math.INT_MIN,
                  "INT_MAX +% 1")
        assert_eq(wrap_add(math.INT_MIN, -1, ROUNDS), math.INT_MAX,
                  "INT_MIN +% -1")
        assert_eq(wrap_add(math.INT_MAX, math.INT_MAX, ROUNDS), -2,
                  "INT_MAX +% INT_MAX")
        assert_eq(wrap_add(math.INT_MIN, math.INT_MIN, ROUNDS), 0,
                  "INT_MIN +% INT_MIN")
        assert_eq(wrap_add(0, 0, ROUNDS), 0, "0 +% 0")
        assert_eq(wrap_add(-1, 1, ROUNDS), 0, "-1 +% 1")
        call += 1
    }
}

pub fn test_sub_wraps_at_both_ends() -> void {
    var call = 0
    while call < CALLS {
        assert_eq(wrap_sub(math.INT_MIN, 1, ROUNDS), math.INT_MAX,
                  "INT_MIN -% 1")
        assert_eq(wrap_sub(math.INT_MAX, -1, ROUNDS), math.INT_MIN,
                  "INT_MAX -% -1")
        assert_eq(wrap_sub(0, math.INT_MIN, ROUNDS), math.INT_MIN,
                  "0 -% INT_MIN")
        assert_eq(wrap_sub(math.INT_MAX, math.INT_MIN, ROUNDS), -1,
                  "INT_MAX -% INT_MIN")
        assert_eq(wrap_sub(math.INT_MIN, math.INT_MAX, ROUNDS), 1,
                  "INT_MIN -% INT_MAX")
        assert_eq(wrap_sub(0, 0, ROUNDS), 0, "0 -% 0")
        call += 1
    }
}

pub fn test_mul_wraps_and_keeps_the_low_half() -> void {
    var call = 0
    while call < CALLS {
        #: INT_MIN is its own negation: the one product `*` raises on and `*%`
        #: has no other answer for.
        assert_eq(wrap_mul(math.INT_MIN, -1, ROUNDS), math.INT_MIN,
                  "INT_MIN *% -1")
        assert_eq(wrap_mul(-1, math.INT_MIN, ROUNDS), math.INT_MIN,
                  "-1 *% INT_MIN")
        assert_eq(wrap_mul(math.INT_MAX, 2, ROUNDS), -2, "INT_MAX *% 2")
        assert_eq(wrap_mul(math.INT_MIN, 2, ROUNDS), 0, "INT_MIN *% 2")
        assert_eq(wrap_mul(math.INT_MAX, math.INT_MAX, ROUNDS), 1,
                  "INT_MAX *% INT_MAX")
        assert_eq(wrap_mul(math.INT_MIN, math.INT_MIN, ROUNDS), 0,
                  "INT_MIN *% INT_MIN")
        #: The smallest square that does not fit: 3037000500 ** 2 is one step
        #: past INT_MAX, so the low half is negative.
        assert_eq(wrap_mul(3037000500, 3037000500, ROUNDS),
                  -9223372036709301616, "3037000500 squared")
        assert_eq(wrap_mul(-1, -1, ROUNDS), 1, "-1 *% -1")
        assert_eq(wrap_mul(math.INT_MIN, 0, ROUNDS), 0, "INT_MIN *% 0")
        assert_eq(wrap_mul(1103515245, 12345, ROUNDS), 13622895699525,
                  "a product that fits is unchanged")
        call += 1
    }
}

pub fn test_the_wrapping_identities_hold() -> void {
    var bad = 0
    var call = 0
    while call < CALLS {
        bad += wrap_round_trip(math.INT_MAX, 1, ROUNDS)
        bad += wrap_round_trip(math.INT_MIN, 1, ROUNDS)
        bad += wrap_round_trip(math.INT_MIN, math.INT_MIN, ROUNDS)
        bad += wrap_round_trip(math.INT_MAX, math.INT_MIN, ROUNDS)
        bad += wrap_round_trip(-1, 0, ROUNDS)
        bad += wrap_round_trip(3037000500, 3037000500, ROUNDS)
        bad += wrap_double(math.INT_MAX, ROUNDS)
        bad += wrap_double(math.INT_MIN, ROUNDS)
        bad += wrap_double(-1, ROUNDS)
        bad += wrap_negate(math.INT_MIN, ROUNDS)
        bad += wrap_negate(math.INT_MAX, ROUNDS)
        bad += wrap_negate(0, ROUNDS)
        call += 1
    }
    assert_eq(bad, 0)
}

pub fn test_a_wrapping_recurrence() -> void {
    var call = 0
    while call < CALLS {
        assert_eq(wrap_hash(0, 1), 1442695040888963407, "one step from zero")
        assert_eq(wrap_hash(0, 16), -2566228959737388584, "sixteen steps")
        assert_eq(wrap_hash(math.INT_MIN, 16), 6657143077117387224,
                  "sixteen steps from INT_MIN")
        call += 1
    }
}

#: Inside a `try` the CHECKED arms compute into a scratch and only copy home
#: once the overflow guard has passed (see ovfDest): a guard there resumes at
#: the START of its instruction, so a result already sitting in its home would
#: be applied a second time by the interpreter. The wrapping arms deliberately
#: do NOT do that -- they take no guard, so nothing resumes at them and the
#: result may go straight to its register. This is the shape that catches it if
#: that reasoning is wrong; the same half of the differential fuzzer's corpus
#: catches it too, and the arm writing to a scratch and never copying home
#: fails both.
fn wrap_in_a_try(a: int, b: int, rounds: int) -> int {
    var acc = a
    var raised = 0
    var i = 0
    while i < rounds {
        try {
            acc = acc *% b +% i
            #: Checked, so it raises for `a = INT_MAX` and the catch runs --
            #: which is what puts a real guard in the same region as the wraps.
            acc = acc -% (a + 1)
        } catch _e {
            raised += 1
            acc = acc +% 1
        }
        i += 1
    }
    return acc +% raised
}

pub fn test_wrapping_inside_a_protected_region() -> void {
    var call = 0
    while call < CALLS {
        assert_eq(wrap_in_a_try(math.INT_MIN, -1, 8), -9223372036854775804,
                  "eight steps, no raise")
        assert_eq(wrap_in_a_try(math.INT_MAX, 3, ROUNDS), 9223372036854642953,
                  "twelve steps, a raise every one")
        assert_eq(wrap_in_a_try(0, 6364136223846793005, ROUNDS), 8547306004929263390,
                  "twelve steps from zero")
        call += 1
    }
}

#: `+%` never applies to a float, so the tier has to hand a float operand back
#: however warm the body is. `flip` is 40,000 iterations in, well past every
#: threshold, so the loop is compiled and specialised to int when the float
#: arrives.
fn late_float(rounds: int, flip: int) -> int {
    var total = 0
    var i = 0
    while i < rounds {
        var a: any = i
        if i == flip { a = 1.5 }
        total = total +% int(a +% 2)
        i += 1
    }
    return total
}

pub fn test_a_float_reaches_the_interpreter() -> void {
    assert_eq(late_float(50000, -1), 1250075000, "no float, plain sum")
    assert_throws(TypeError, fn() -> void {
        let _ = late_float(50000, 40000)
    }, "a float arriving after the body is compiled")
}

class Wrapped {
    pub var n: int
    pub fn init(self, n: int) -> void { self.n = n }
    pub fn __add__(self, other: Wrapped) -> Wrapped {
        return Wrapped(self.n + other.n)
    }
}

#: An object is an `__add__` call, not arithmetic: the interpreter looks the
#: dunder up for `+%` exactly as it does for `+`. The checker rejects `+%` on a
#: class outright, so the only way to put one in front of the opcode is through
#: `any` -- which is also the only way the compiled tier can meet one, since a
#: statically typed operand it would have refused never reaches the arm.
fn dunder_sum(rounds: int) -> int {
    var total = 0
    var i = 0
    while i < rounds {
        let a: any = Wrapped(i)
        let b: any = Wrapped(1)
        let sum = a +% b
        total += int(sum.n)
        i += 1
    }
    return total
}

pub fn test_an_object_still_reaches_its_dunder() -> void {
    assert_eq(dunder_sum(200), 20100)
}
