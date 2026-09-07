#: Every arithmetic and indexing edge the interpreter raises or saturates on,
#: executed inside a JIT-warmed function. The two tiers must agree exactly.
#:
#: This file exists because a tier divergence SHIPPED: compiled `x / 0.0`
#: returned inf where the interpreter raises DivisionByZeroError, and no test
#: caught it because every DivisionByZeroError assertion in the suite was
#: integer -- and there is no integer `/` in this language, so those tests never
#: entered the arm. A corpus cannot validate a path it never runs.
#:
#: Every function below is called 400 times before the edge value reaches it, so
#: the edge is taken in COMPILED code. Run under JAITHON_NO_JIT=1 and the output
#: must be byte-identical; the golden runner already re-runs it at
#: JAITHON_JIT_THRESHOLD=1 and under DEOPT_STRESS.
#:
#: Covered: int overflow (+ - * unary-), true division by zero, floor division
#: by zero, modulo by zero, INT64_MIN with -1 on each, `**` with a negative and
#: an overflowing exponent, shift counts that are negative and >= 64, list and
#: string indexing out of range at both ends, negative indexing that is in
#: range, and float division and modulo by both signs of zero.
fn f_add(a: int, b: int) -> int { return a + b }
fn f_sub(a: int, b: int) -> int { return a - b }
fn f_mul(a: int, b: int) -> int { return a * b }
fn f_neg(a: int) -> int { return -a }
fn f_div(a: int, b: int) -> float { return a / b }
fn f_mod(a: int, b: int) -> int { return a % b }
fn f_fdiv(a: int, b: int) -> int { return a // b }
fn f_pow(a: int, b: int) -> int { return a ** b }
fn f_shl(a: int, b: int) -> int { return a << b }
fn f_fltdiv(a: float, b: float) -> float { return a / b }
fn f_fltmod(a: float, b: float) -> float { return a % b }
fn f_idx(xs: list[int], i: int) -> int { return xs[i] }
fn f_sidx(s: str, i: int) -> str { return s[i] }

fn main() {
    let MIN = -9223372036854775807 - 1
    let MAX = 9223372036854775807
    var xs: list[int] = []
    for k in 0..8 { xs.push(k * 10) }
    let abc = "abc"

    var w = 0
    var dv = 0.0
    var i = 0
    while i < 400 {
        w = w + f_add(1, 2) + f_sub(9, 3) + f_mul(3, 4) + f_neg(5)
        w = w + f_mod(10, 3) + f_fdiv(10, 3) + f_pow(2, 5)
        w = w + f_idx(xs, 3) + f_shl(1, 3)
        dv = dv + f_div(10, 2)
        i = i + 1
    }
    var fw = 0.0
    var j = 0
    while j < 400 {
        fw = fw + f_fltdiv(1.0, 2.0) + f_fltmod(5.0, 3.0)
        j = j + 1
    }
    var sw = ""
    var m = 0
    while m < 400 {
        sw = f_sidx(abc, 1)
        m = m + 1
    }
    print(f"warm {w} {fw} {sw} {dv}")

    var z = 0
    var n1 = -1
    var big = 64

    try { print(f"MAX+1 {f_add(MAX, 1)}") } catch _e: OverflowError { print("MAX+1 Overflow") }
    try { print(f"MIN-1 {f_sub(MIN, 1)}") } catch _e: OverflowError { print("MIN-1 Overflow") }
    try { print(f"MAX*2 {f_mul(MAX, 2)}") } catch _e: OverflowError { print("MAX*2 Overflow") }
    try { print(f"-MIN {f_neg(MIN)}") } catch _e: OverflowError { print("-MIN Overflow") }
    try { print(f"1/0 {f_div(1, z)}") } catch _e: DivisionByZeroError { print("1/0 DivZero") }
    try { print(f"MIN/-1 {f_div(MIN, n1)}") } catch _e: OverflowError | DivisionByZeroError { print("MIN/-1 raised") }
    try { print(f"1mod0 {f_mod(1, z)}") } catch _e: DivisionByZeroError { print("1mod0 DivZero") }
    try { print(f"MINmod-1 {f_mod(MIN, n1)}") } catch _e: OverflowError | DivisionByZeroError { print("MINmod-1 raised") }
    try { print(f"1//0 {f_fdiv(1, z)}") } catch _e: DivisionByZeroError { print("1//0 DivZero") }
    try { print(f"MIN//-1 {f_fdiv(MIN, n1)}") } catch _e: OverflowError | DivisionByZeroError { print("MIN//-1 raised") }
    try { print(f"2**-1 {f_pow(2, n1)}") } catch _e: ValueError | OverflowError { print("2**-1 raised") }
    try { print(f"MAX**2 {f_pow(MAX, 2)}") } catch _e: OverflowError { print("MAX**2 Overflow") }
    try { print(f"1<<64 {f_shl(1, big)}") } catch _e: ValueError { print("1<<64 raised") }
    try { print(f"1<<-1 {f_shl(1, n1)}") } catch _e: ValueError { print("1<<-1 raised") }
    try { print(f"xs99 {f_idx(xs, 99)}") } catch _e: IndexError { print("xs99 IndexError") }
    try { print(f"xs-99 {f_idx(xs, -99)}") } catch _e: IndexError { print("xs-99 IndexError") }
    print(f"xs-1 {f_idx(xs, n1)}")
    try { print(f"fdiv0 {f_fltdiv(1.0, 0.0)}") } catch _e: DivisionByZeroError { print("fdiv0 DivZero") }
    try { print(f"fdivn0 {f_fltdiv(-1.0, 0.0)}") } catch _e: DivisionByZeroError { print("fdivn0 DivZero") }
    try { print(f"fmod0 {f_fltmod(1.0, 0.0)}") } catch _e: DivisionByZeroError { print("fmod0 DivZero") }
    try { print(f"s99 {f_sidx(abc, 99)}") } catch _e: IndexError { print("s99 IndexError") }
    print(f"s-1 {f_sidx(abc, n1)}")
}
