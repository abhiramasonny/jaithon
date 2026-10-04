#: A float operator that reads a local's d home instead of the copy taken when
#: the local was pushed.
#:
#: `sum += weight(i, j) * v[j]` pushes `sum` FIRST, then evaluates the call and
#: the subscript. The call is outside the FP whitelist, so the copy of `sum`
#: went out to an X register and the add brought it back with `fmov d,x` -- two
#: cross-register-file moves on the loop-carried chain. The compiled tier now
#: reads the local's own d register at the add, which is only right while
#: nothing has written ANY local since the push and no join has been crossed
#: (`fpOperandReread` in `src/vm/jit/jit_func.c`, JAITHON_JIT_FP_REREAD).
#:
#: So every shape here is one where a stale or a fresh read would differ, and
#: every expectation is the interpreter's: `JAITHON_NO_JIT=1` proves them and
#: the ordinary run proves the compiled code agrees. Each loop runs long enough
#: for the OSR tier to compile it, and each function is called past the
#: whole-function threshold.
from std.math import INF, NAN, is_nan
from std.test import assert_eq, assert_true

fn weight(i: int, j: int) -> float {
    let s = i + j
    return 1.0 / float(s * (s + 1) // 2 + i + 1)
}

fn spectral_row(v: list[float], i: int, n: int) -> float {
    var sum = 0.0
    for j in 0..n { sum += weight(i, j) * v[j] }
    return sum
}

fn spectral_row_long(v: list[float], i: int, n: int) -> float {
    var sum = 0.0
    var j = 0
    while j < n {
        sum = sum + weight(i, j) * v[j]
        j += 1
    }
    return sum
}

fn reference_row(v: list[float], i: int, n: int) -> float {
    var sum = 0.0
    for j in 0..n {
        let w = weight(i, j)
        let x = v[j]
        sum = sum + w * x
    }
    return sum
}

fn test_an_accumulator_read_before_a_call_matches_the_interpreter() -> void {
    var v: list[float] = []
    for k in 0..300 { v.push(1.0 + float(k) * 0.001) }
    for i in 0..200 {
        let want = reference_row(v, i, 300)
        assert_eq(spectral_row(v, i, 300), want, "for-range accumulator")
        assert_eq(spectral_row_long(v, i, 300), want, "while accumulator")
    }
}

fn sub_and_mul(v: list[float], n: int) -> float {
    var down = 1000.0
    var prod = 1.0
    for j in 0..n {
        down -= weight(j, 1) * v[j]
        prod *= 1.0 + weight(1, j) * v[j]
    }
    return down + prod
}

fn test_subtract_and_multiply_accumulators() -> void {
    var v: list[float] = []
    for k in 0..400 { v.push(float(k % 7) - 3.0) }
    var first = sub_and_mul(v, 400)
    for _round in 0..150 { assert_eq(sub_and_mul(v, 400), first, "stable") }
    var down = 1000.0
    var prod = 1.0
    for j in 0..400 {
        let a = weight(j, 1) * v[j]
        down = down - a
        let b = 1.0 + weight(1, j) * v[j]
        prod = prod * b
    }
    assert_eq(first, down + prod, "matches the spelled-out form")
}

#: The local is written between its push and the operator that consumes it:
#: the read has to see the value as it was at the push.
fn rebinds_in_between(v: list[float], n: int) -> float {
    var acc = 1.0
    var other = 0.0
    for j in 0..n {
        acc = acc + bump_then(v[j], j)
        other = acc
        acc = other * 0.5 + acc * 0.5
    }
    return acc + other
}

fn bump_then(x: float, j: int) -> float {
    if j % 3 == 0 { return x * 2.0 }
    return x
}

fn test_a_write_between_the_push_and_the_operator() -> void {
    var v: list[float] = []
    for k in 0..500 { v.push(float(k) * 0.25) }
    var want = 0.0
    var acc = 1.0
    var other = 0.0
    for j in 0..500 {
        let t = bump_then(v[j], j)
        acc = acc + t
        other = acc
        acc = other * 0.5 + acc * 0.5
    }
    want = acc + other
    for _round in 0..100 { assert_eq(rebinds_in_between(v, 500), want, "rebinds") }
}

#: A closure that writes the accumulator WHILE the right-hand side is being
#: evaluated. The interpreter read `total` before the call, so the call's write
#: is overwritten; a read of the local at the add would see it instead.
fn closure_writes_it(n: int) -> float {
    var total = 0.0
    let poke = fn(x: float) -> float {
        total = total + 1000.0
        return x
    }
    for j in 0..n { total += poke(float(j)) * 1.0 }
    return total
}

fn test_a_closure_writing_the_accumulator_mid_expression() -> void {
    var want = 0.0
    for j in 0..300 { want += float(j) }
    for _round in 0..100 {
        assert_eq(closure_writes_it(300), want, "the call's write is overwritten")
    }
}

#: A join between the push and the operator: one arm of the conditional
#: assigns through a call that returns a value computed from the local.
fn joins_in_between(v: list[float], n: int, flag: bool) -> float {
    var s = 0.5
    for j in 0..n {
        s = s + (if flag { v[j] * 2.0 } else { weight(j, j) }) * 1.0
    }
    return s
}

fn test_a_join_between_the_push_and_the_operator() -> void {
    var v: list[float] = []
    for k in 0..300 { v.push(float(k) * 0.5) }
    var a = 0.5
    var b = 0.5
    for j in 0..300 {
        a = a + v[j] * 2.0
        b = b + weight(j, j)
    }
    for _round in 0..100 {
        assert_eq(joins_in_between(v, 300, true), a, "taken arm")
        assert_eq(joins_in_between(v, 300, false), b, "other arm")
    }
}

#: The special values must come through exactly: -0.0 stays negative, NaN
#: stays NaN, infinities stay infinite.
fn signed_sum(v: list[float], n: int) -> float {
    var sum = -0.0
    for j in 0..n { sum += weight(0, 0) * v[j] }
    return sum
}

fn test_special_values_survive() -> void {
    var zeros: list[float] = []
    for _k in 0..200 { zeros.push(-0.0) }
    for _round in 0..100 {
        let z = signed_sum(zeros, 200)
        assert_eq(z, 0.0, "a sum of negative zeros is zero")
        assert_eq(str(z), "-0.0", "and keeps its sign")
    }
    var withNan: list[float] = []
    for k in 0..200 { withNan.push(if k == 150 { NAN } else { 1.0 }) }
    assert_true(is_nan(signed_sum(withNan, 200)), "NaN propagates")
    var withInf: list[float] = []
    for k in 0..200 { withInf.push(if k == 10 { INF } else { 1.0 }) }
    assert_eq(signed_sum(withInf, 200), INF, "inf propagates")
}

#: An assignment to the accumulator INSIDE the right-hand side, which a block
#: arm of an `if` allows. The interpreter pushed `s` before the arm ran, so the
#: arm's write is lost; a read of the local at the add would keep it.
fn writes_inside_the_expression(n: int, flag: bool) -> float {
    var s = 1.0
    for j in 0..n {
        s = s + (if flag {
            s = 100.0
            2.0
        } else {
            float(j)
        })
    }
    return s
}

fn test_a_write_inside_the_right_hand_side() -> void {
    var other = 1.0
    for j in 0..400 { other = other + float(j) }
    for _round in 0..100 {
        assert_eq(writes_inside_the_expression(400, true), 801.0, "the arm's write is lost")
        assert_eq(writes_inside_the_expression(400, false), other, "the other arm")
    }
}
