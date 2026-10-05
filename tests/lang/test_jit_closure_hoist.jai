#: A closure called in a loop, inlined, with its guard and upvalues proved
#: once at the loop head (JAITHON_JIT_CLOSURE_HOIST).
#:
#: The head proves the local holds a closure over the function the body was
#: inlined from, and loads the scalar upvalues that body reads into registers.
#: Every way the head's facts could go stale between entries is driven here:
#: a different function behind the same local, the same function over an
#: upvalue of another kind, a cell the enclosing frame keeps writing, and an
#: inlined result stored straight into a local.
from std.test import assert_eq

fn adder(step: int) -> fn(int) -> int {
    return |x| x + step
}

fn scaler(k: float) -> fn(float) -> float {
    return |x| x * k
}

fn flipper(on: bool) -> fn(int) -> int {
    return |x| if on { x + 1 } else { x - 1 }
}

fn apply_n(f: fn(int) -> int, start: int, times: int) -> int {
    var acc = start
    var i = 0
    while i < times {
        acc = f(acc)
        i += 1
    }
    return acc
}

fn apply_f(f: fn(float) -> float, start: float, times: int) -> float {
    var acc = start
    for _ in 0..times {
        acc = f(acc)
    }
    return acc
}

#: The result is bound to a fresh local rather than stored over the argument.
fn bind_n(f: fn(int) -> int, times: int) -> int {
    var total = 0
    var i = 0
    while i < times {
        let y = f(i)
        total += y
        i += 1
    }
    return total
}

fn test_each_closure_gets_its_own_upvalue() -> void {
    var total = 0
    for k in 1..300 {
        total += apply_n(adder(k), 0, 500)
    }
    var want = 0
    for k in 1..300 { want += k * 500 }
    assert_eq(total, want)
}

fn test_another_function_behind_the_same_local() -> void {
    for _ in 0..200 { apply_n(adder(2), 0, 300) }
    assert_eq(apply_n(|x| x * 2, 1, 10), 1024)
    assert_eq(apply_n(adder(3), 0, 300), 900)
    assert_eq(apply_n(|x| x - 1, 0, 300), -300)
}

fn test_a_float_upvalue() -> void {
    var got = 0.0
    for _ in 0..200 { got = apply_f(scaler(1.5), 1.0, 4) }
    assert_eq(got, 5.0625)
    assert_eq(apply_f(scaler(0.5), 8.0, 3), 1.0)
}

fn test_a_bool_upvalue() -> void {
    var up = 0
    var down = 0
    for _ in 0..200 {
        up = apply_n(flipper(true), 0, 100)
        down = apply_n(flipper(false), 0, 100)
    }
    assert_eq(up, 100)
    assert_eq(down, -100)
}

fn test_a_bound_result() -> void {
    var got = 0
    for _ in 0..200 { got = bind_n(adder(1), 100) }
    assert_eq(got, 5050)
}

#: The cell is OPEN: `step` is this frame's own local, captured by reference,
#: and the loop writes it between calls. The loop must read the new value
#: every time, so nothing about the upvalue may be hoisted here.
fn test_the_enclosing_frame_writes_the_cell() -> void {
    var step = 1
    let f = |x| x + step
    var acc = 0
    var i = 0
    while i < 400000 {
        acc = f(acc)
        if i % 1000 == 999 { step += 1 }
        i += 1
    }
    var want = 0
    for j in 0..400000 { want += 1 + j // 1000 }
    assert_eq(acc, want)
}

#: The same, with the write made by a call in the loop: a loop that calls
#: anything keeps every read per-site.
fn test_a_call_writes_the_cell() -> void {
    var step = 1
    let f = |x| x + step
    let bump = fn() { step += 1 }
    var acc = 0
    for i in 0..400000 {
        acc = f(acc)
        if i % 1000 == 999 { bump() }
    }
    var want = 0
    for j in 0..400000 { want += 1 + j // 1000 }
    assert_eq(acc, want)
}

#: Overflow inside the inlined body must raise exactly as the call would.
fn test_overflow_inside_the_hoisted_body() -> void {
    for _ in 0..200 { apply_n(adder(1), 0, 100) }
    var raised = false
    try {
        let _unused = apply_n(adder(4611686018427387904), 0, 3)
    } catch _failure {
        raised = true
    }
    assert_eq(raised, true)
}

#: Called once, for long enough that the LOOP compiles (OSR) with the lambda
#: inlined. `x - 1` is the fused constant form, which writes the copy of `x`
#: it reads in place: an inline that only borrowed `x` there subtracted from a
#: register nothing had written.
fn count_with(f: fn(int) -> int, start: int, times: int) -> int {
    var acc = start
    var i = 0
    while i < times {
        acc = f(acc)
        i += 1
    }
    return acc
}

fn test_a_fused_constant_body_in_a_compiled_loop() -> void {
    assert_eq(count_with(|x| x - 1, 0, 1000000), -1000000)
}

#: The cell is open and this frame owns it, but the loop never writes it, so
#: its value is loaded once at the head; the write before the loop must be
#: the one the loop sees.
fn test_the_enclosing_frame_keeps_the_cell() -> void {
    var step = 1
    let f = |x| x + step
    step += 2
    var acc = 0
    var i = 0
    while i < 400000 {
        acc = f(acc)
        i += 1
    }
    assert_eq(acc, 1200000)
}

#: Bodies that compute on the parameter rather than pass it through: a fused
#: constant op reads the argument's register in place, `^` reads a borrowed
#: operand, and `%` puts a cold fix-up join inside the inlined body. The last
#: once made the function tier decline the caller -- the inlined body's own
#: offsets were checked against the caller's branch targets.
fn test_bodies_that_compute_on_the_parameter() -> void {
    let d = 3
    let f = |x| x + 1 + d
    let g = |x| (x + 1) % 65521 + d
    let h = |x| x ^ d
    let m = |x| (x * 3 + d) % 1000003
    var a = 0
    var b = 0
    var c = 5
    var q = 1
    var i = 0
    while i < 300001 {
        a = f(a)
        b = g(b)
        c = h(c)
        q = m(q)
        i += 1
    }
    assert_eq(a, 1200004)
    assert_eq(b, 20626)
    assert_eq(c, 6)
    assert_eq(q, 77203)
}
