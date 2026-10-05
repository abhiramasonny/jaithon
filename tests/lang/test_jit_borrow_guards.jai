#: `sum += a[k] * b[k][j]` keeps `sum` in its d register across the subscripts
#: instead of copying it out first: the copy was an `fmov` on the loop-carried
#: chain. The subscripts guard, so a deopt in the middle of the expression has
#: to write `sum` out of the home it is borrowed from -- the bank register its
#: stack entry would have used was never written. Each case runs the compiled
#: body into such a guard and checks the interpreter carried on with the right
#: partial sum.
from std.test import assert_eq, assert_throws

fn floats(n: int, seed: int) -> list[float] {
    var out: list[float] = []
    var s = seed
    for _i in 0..n {
        s = (s * 1103515245 + 12345) % 2147483648
        out.push(float(s % 1000) / 1000.0)
    }
    return out
}

fn rows(n: int) -> list[list[float]] {
    var out: list[list[float]] = []
    for i in 0..n { out.push(floats(6, i + 1)) }
    return out
}

fn dot_column(a: list[float], b: list[list[float]], j: int) -> float {
    var sum = 0.5
    for k in 0..len(a) { sum += a[k] * b[k][j] }
    return sum
}

#: Row 37 swapped for a boxed list of the same values: its storage guard
#: fires with `sum` borrowed, and the answer must not move by a bit.
fn test_storage_guard_mid_sum() -> void {
    let a = floats(300, 9)
    let good = rows(300)
    let want = dot_column(a, good, 3)
    for _round in 0..200 { assert_eq(dot_column(a, good, 3), want) }
    var mixed: list[list[float]] = []
    for i in 0..300 { mixed.push(good[i]) }
    let r = good[37]
    mixed[37] = [r[0], r[1], r[2], r[3], r[4], r[5]]
    assert_eq(dot_column(a, mixed, 3), want)
    for _round in 0..50 { assert_eq(dot_column(a, good, 3), want) }
}

#: A short row: the bounds guard fires with `sum` borrowed and the
#: interpreter raises, as it would have without the tier.
fn test_bounds_guard_mid_sum() -> void {
    let a = floats(300, 4)
    let good = rows(300)
    for _round in 0..200 { let _s = dot_column(a, good, 5) }
    var short: list[list[float]] = []
    for i in 0..300 { short.push(good[i]) }
    short[120] = floats(3, 1)
    assert_throws(IndexError, || dot_column(a, short, 5))
}

#: The same shape as the inner loop of a nest, which the loop tier compiles
#: within the first few passes; from pass 150 one row holds ints, so every
#: later pass deoptimises at that row with `sum` borrowed. The reference binds
#: the product first, so its accumulator is never on the stack at a guard.
fn test_loop_tier_sum() -> void {
    let n = 2000
    let a = floats(n, 2)
    var b = []
    for i in 0..n { b.push(floats(2, i)) }
    var total = 0.0
    var ref_total = 0.0
    for rep in 0..200 {
        if rep == 150 { b[700] = [1, 2] }
        var sum = 0.25
        for k in 0..n {
            sum += a[k] * b[k][1]
        }
        total += sum
        var ref = 0.25
        for k in 0..n {
            let p = a[k] * b[k][1]
            ref = ref + p
        }
        ref_total += ref
    }
    assert_eq(total, ref_total)
}
