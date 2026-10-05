#: A call through a function value, made from inside an inlined body.
#:
#: The straight-line inliner admitted `OP_CALL` whenever it took one argument,
#: on the stated belief that the only callee on the stack could be `float` or
#: `int` -- but a parameter holding a function value is a callee too, so
#: `fn forwards(f, x) { return f(x) }` was inlined with a real call inside it.
#: The closure-call arm then ran within the inline and named its registers from
#: the caller's bank rather than the inline's own: the guard read `fn` out of
#: whatever the caller kept there, and passed the frame pointer as the
#: argument. What that does depends on the stale register: where it held an
#: object the guard just failed every call, and where it held an int the load
#: segfaulted -- the two-accumulator loop below did so on every run, under
#: every configuration. Every record inside an inline also resumes at the
#: OUTER call, so a callee that deoptimised after writing would have run twice.
#:
#: Now the inliner only admits a call it has seen a builtin read for, and the
#: closure arm refuses to run inside an inline at all.
# jit-compiles: forwards, forwarder_loop, forwards_one, forwarder_while, forwarder_loop2
from std.test import assert_eq

let WARM = 300

var bumps = 0

fn forwards(f: fn(int) -> int, x: int) -> int {
    let r = f(x)
    return r + 0
}

fn forwarder_loop(f: fn(int) -> int, n: int) -> int {
    var t = 0
    for i in 0..n {
        t = t +% forwards(f, i)
    }
    return t
}

#: One callee only, so the inline's guard is never expected to fail.
fn forwards_one(f: fn(int) -> int, x: int) -> int {
    let r = f(x)
    return r + 0
}

fn forwarder_while(f: fn(int) -> int, n: int) -> int {
    var t = 0
    var i = 0
    while i < n {
        t = t +% forwards_one(f, i)
        i = i + 1
    }
    return t
}

#: A second accumulator moves the register the arm wrongly read.
fn forwarder_loop2(f: fn(int) -> int, n: int, m: int) -> int {
    var t = 0
    var u = 3
    for i in 0..n {
        u = u +% m
        t = t +% forwards_one(f, i) +% u
    }
    return t
}

fn adds_to_bumps(n: int) -> int {
    bumps = bumps +% n
    return n + 1
}

fn triples(n: int) -> int {
    let q = n * 3
    return q + 1
}

fn test_a_closure_call_inside_an_inline() -> void {
    bumps = 0
    var t = 0
    for k in 0..WARM {
        t = t +% forwards(adds_to_bumps, k)
        t = t +% forwards(triples, k)
    }
    for k in 0..WARM {
        t = t +% forwarder_loop(adds_to_bumps, 500)
        t = t +% forwarder_loop(triples, 500)
    }
    assert_eq(t, 150180000)
    assert_eq(bumps, 37469850)
}

fn test_a_monomorphic_forwarder_in_two_loops() -> void {
    bumps = 0
    var t = 0
    for k in 0..WARM {
        t = t +% forwards_one(adds_to_bumps, k)
    }
    for k in 0..WARM {
        t = t +% forwarder_while(adds_to_bumps, 50)
        t = t +% forwarder_loop2(adds_to_bumps, 50, k)
    }
    assert_eq(t, 58038900)
    assert_eq(bumps, 779850)
}
