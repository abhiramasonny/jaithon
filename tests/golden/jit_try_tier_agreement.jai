#: A raise from inside a COMPILED `try`, in every shape the tier can meet it.
#:
#: This is the sharpest edge in the two-tier design and the code says so: a
#: raise from compiled code unwinds from a frame the interpreter never pushed
#: (function tier) or from the loop head rather than the faulting instruction
#: (OSR tier), so `vmThrow` can consult the wrong offset and skip the
#: function's OWN handler. `Emit::inProtected` exists for exactly this, and
#: `branchOnOverflow` / `branchOnDeoptInstStart` have separate paths for it.
#:
#: Nothing tested this end to end. The companion file jit_tier_agreement.jai
#: covers the arithmetic edges; this one covers who CATCHES them, which is a
#: different mechanism.
#:
#: Every body is warmed 500 times before the raising value arrives, so the raise
#: is taken in compiled code. Output must be byte-identical under
#: JAITHON_NO_JIT=1, and the golden runner also re-runs it at
#: JAITHON_JIT_THRESHOLD=1 and under DEOPT_STRESS.
#:
#: Shapes covered: overflow, index and division-by-zero raised inside a try;
#: nested try where the inner catch does not match; try beside a `defer`;
#: try/catch/finally where finally must still run; a raise from inside an
#: OSR-compiled `while` inside a try; an explicit `throw` from a hot body; and
#: a catch that re-throws to an outer handler.
    try { return a + b } catch _e: OverflowError { return -1 }
}
fn guarded_idx(xs: list[int], i: int) -> int {
    try { return xs[i] } catch _e: IndexError { return -2 }
}
fn guarded_div(a: int, b: int) -> float {
    try { return a / b } catch _e: DivisionByZeroError { return -3.0 }
}
fn nested(xs: list[int], i: int) -> int {
    try {
        try { return xs[i] } catch _e: ValueError { return -4 }
    } catch _e: IndexError { return -5 }
}
fn with_defer(xs: list[int], i: int) -> int {
    var seen = 0
    defer { seen = 1 }
    try { return xs[i] } catch _e: IndexError { return -6 }
}
fn finally_path(xs: list[int], i: int) -> int {
    var acc = 0
    try { acc = xs[i] } catch _e: IndexError { acc = -7 } finally { acc = acc + 1000 }
    return acc
}


fn loop_in_try(xs: list[int], upto: int) -> int {
    var acc = 0
    try {
        var k = 0
        while k < upto {
            acc = acc + xs[k]
            k = k + 1
        }
    } catch _e: IndexError { return -10 }
    return acc
}

fn throw_from_hot(n: int) -> int {
    try {
        if n > 100000 { throw ValueError("boom") }
        return n
    } catch _e: ValueError { return -11 }
}

fn reraise_hot(xs: list[int], i: int) -> int {
    try {
        try { return xs[i] } catch e: IndexError { throw e }
    } catch _e: IndexError { return -12 }
}

fn main() {
    let MAX = 9223372036854775807
    var xs: list[int] = []
    for k in 0..8 { xs.push(k * 10) }
    var w = 0
    var i = 0
    while i < 500 {
        w = w + guarded_ovf(1, 2) + guarded_idx(xs, 3) + nested(xs, 2)
        w = w + with_defer(xs, 1) + finally_path(xs, 1)
        i = i + 1
    }
    var fw = 0.0
    var j = 0
    while j < 500 {
        fw = fw + guarded_div(10, 2)
        j = j + 1
    }
    print(f"warm {w} {fw}")
    var z = 0
    print(f"ovf   {guarded_ovf(MAX, 1)}")
    print(f"idx   {guarded_idx(xs, 99)}")
    print(f"div   {guarded_div(1, z)}")
    print(f"nest  {nested(xs, 99)}")
    print(f"defer {with_defer(xs, 99)}")
    print(f"fin   {finally_path(xs, 99)}")
    var lw = 0
    var q = 0
    while q < 500 {
        lw = lw + loop_in_try(xs, 8) + throw_from_hot(1) + reraise_hot(xs, 2)
        q = q + 1
    }
    print(f"warm2 {lw}")
    print(f"loop  {loop_in_try(xs, 99)}")
    print(f"throw {throw_from_hot(200000)}")
    print(f"rerai {reraise_hot(xs, 99)}")
}
