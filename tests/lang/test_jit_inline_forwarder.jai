#: A call through a function value, made from inside an inlined body.
#:
#: The straight-line inliner admitted `OP_CALL` whenever it took one argument,
#: on the stated belief that the only callee on the stack could be `float` or
#: `int` -- but a parameter holding a function value is a callee too, so
#: `fn forwards(f, x) { return f(x) }` was inlined with a real call inside it.
#: The closure-call arm then ran within the inline and named its registers from
#: the caller's bank rather than the inline's own: the guard read `fn` out of
#: whatever the caller kept there. A loop calling `forwards` segfaulted on every
#: run once it had compiled, and under deopt stress even with a callee that
#: writes nothing. Every record inside an inline also resumes at the OUTER
#: call, so a callee that deoptimised after writing would have run twice.
#:
#: Now the inliner only admits a call it has seen a builtin read for, and the
#: closure arm refuses to run inside an inline at all.
# jit-compiles: forwards, forwarder_loop
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
