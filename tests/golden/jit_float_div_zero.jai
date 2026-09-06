#: Float `/` by zero must raise from COMPILED code, not just interpreted.
#:
#: The interpreter raises DivisionByZeroError when the divisor is 0.0
#: (floatArithmetic, vm_operator.c). The function tier's float-pair arm in
#: emitAddSubDiv emitted a bare `fdiv`, which yields inf/nan instead -- so the
#: answer depended on which tier ran the division.
#:
#: It shipped because every assert_throws(DivisionByZeroError, ...) in the
#: suite is INTEGER, and integer `/` is a hard whole-body refusal, so no test
#: had ever entered this arm. This one has to work to get there:
#:
#:   - both operands must be SLOT_FLOAT, or the arm is not the one under test
#:   - the divisor must be a RUNTIME zero, not a literal, or the walk could
#:     fold it
#:   - the body must run past JAI_JIT_THRESHOLD (64) before the zero arrives,
#:     or the raise happens interpreted and proves nothing
#:
#: -0.0 and NaN are checked on the same run: FCMP sets Z for -0.0 (which the
#: interpreter also throws on) and leaves it clear for NaN (which it divides).
import std.io
import std.math

fn ratio(a: float, b: float) -> float {
    return a / b
}

fn main() {
    # Warm the body well past the threshold with ordinary divisors, so the
    # division below is running compiled by the time the zero reaches it.
    var acc = 0.0
    var i = 0
    while i < 500 {
        acc = acc + ratio(1.0, 2.0)
        i = i + 1
    }
    print(f"warm {acc}")

    # A runtime zero, reached only after the body is compiled.
    var z = 0.0
    try {
        let bad = ratio(1.0, z)
        print(f"NO RAISE {bad}")
    } catch e: DivisionByZeroError {
        print("positive zero raised")
    }

    # -0.0 compares equal to 0.0, and the interpreter throws on it too.
    var nz = -0.0
    try {
        let bad = ratio(1.0, nz)
        print(f"NO RAISE {bad}")
    } catch e: DivisionByZeroError {
        print("negative zero raised")
    }

    # A NaN divisor is NOT zero -- FCMP leaves Z clear for it -- so both tiers
    # divide and yield NaN rather than raising. (math.NAN is INF - INF, so
    # constructing it does not itself divide by zero.)
    var nan = math.NAN
    print(f"nan divisor is nan: {math.is_nan(ratio(1.0, nan))}")

    # And ordinary division still works after the guard.
    print(f"still divides {ratio(7.0, 2.0)}")
}
