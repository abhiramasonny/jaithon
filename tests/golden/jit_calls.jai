#: Calls the compiled tier makes without going out through the interpreter:
#: a direct branch to a compiled callee's entry, and a callee small enough to
#: be emitted where the call is. Both skip everything jaiJitEnterFunc checks,
#: so what is checked here is that each of those checks still happens.

# --- a guard inside an inlined body resumes at the CALL ----------------------
# `q` is straight-line arithmetic, so it is emitted in place. Its divide has a
# zero check, and when that fires the interpreter has to pick up at the call
# with the callee and its arguments still on its stack -- not part-way through
# a body it never entered.
fn q(a: int, b: int) -> int { return a // b }

fn divides(n: int, d: int) -> int {
    var total = 0
    for i in 0..n { total += q(i, d) }
    return total
}

print(divides(20000, 3))
var caught = 0
try { print(divides(4, 0)) } catch _e { caught = 1 }
print(caught)
print(divides(20000, 3))

# --- a callee that raises, reached by a direct branch ------------------------
# `bump` has a loop, so it is called rather than inlined. Its overflow is a
# real exception and must propagate rather than be retried.
fn bump(a: int) -> int {
    var r = a
    var k = 0
    while k < 1 {
        r = r + 1
        k += 1
    }
    return r
}

fn climb(n: int) -> int {
    var s = 0
    var i = 0
    while i < n {
        s = bump(s)
        i += 1
    }
    return s
}

print(climb(30000))
var overflowed = 0
try { print(bump(9223372036854775807)) } catch _e { overflowed = 1 }
print(overflowed)

# --- an inlined float body, whose result feeds a float chain ----------------
fn ev(i: int, j: int) -> float {
    let s = i + j
    return 1.0 / float(s * (s + 1) // 2 + i + 1)
}

fn total(n: int) -> float {
    var acc = 0.0
    for i in 0..n { for j in 0..n { acc += ev(i, j) } }
    return acc
}

print(f"{total(120):.9f}")
