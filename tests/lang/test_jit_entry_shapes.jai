#: Calls from interpreted code into compiled bodies, across the shapes the
#: short entry path (JAITHON_JIT_LEAN_ENTRY, callClosure + jaiJitEnterFunc)
#: takes and the ones it hands to the old path: zero to four arguments on
#: the short path, five and more on the full one, a method whose receiver is
#: the first argument, a class initializer reached through the class, an
#: argument whose kind changes after the body compiled (a decline at entry),
#: and a body that deopts on entry. Every answer is the interpreter's, so
#: JAITHON_NO_JIT=1 proves the expectations and JAITHON_JIT_LEAN_ENTRY=0 the
#: old path; JAITHON_JIT_THRESHOLD=1 and JAITHON_JIT_DEOPT_STRESS=1 drive the
#: entry and resume arms on every call.
from std.test import assert_eq, assert_throws

fn zero() -> int { return 7 }

fn one(a: int) -> int { return a + 1 }

fn two(a: int, b: int) -> int { return a * 3 + b }

fn three(a: int, b: int, c: int) -> int { return a - b + c }

fn four(a: int, b: int, c: int, d: int) -> int { return a + b * c - d }

fn five(a: int, b: int, c: int, d: int, e: int) -> int {
    return a + b + c + d + e
}

fn half(x: float) -> float { return x / 2.0 }

fn either(x: any) -> int {
    if x == null { return 0 }
    return 1
}

class Acc {
    pub var total: int

    pub fn init(self, start: int) { self.total = start }

    pub fn add(self, a: int, b: int) -> int {
        self.total = self.total + a - b
        return self.total
    }

    pub fn get(self) -> int { return self.total }
}

fn test_every_arity_agrees() {
    var s = 0
    for i in 0..300 {
        s = s + zero() + one(i) + two(i, 1) + three(i, 2, 3) + four(i, 1, 2, 3)
        s = s + five(i, 1, 2, 3, 4)
    }
    assert_eq(s, 319650)
}

fn test_methods_and_initializers() {
    var s = 0
    for i in 0..300 {
        let acc = Acc(i)
        s = s + acc.add(i, 1) + acc.get()
    }
    assert_eq(s, 178800)
}

fn test_an_argument_that_changes_kind() {
    var s = 0.0
    for i in 0..300 { s = s + half(1.0) }
    assert_eq(s, 150.0)
    var n = 0
    for i in 0..300 {
        n = n + either(i)
        if i % 3 == 0 { n = n + either(null) }
    }
    assert_eq(n, 300)
}

fn test_deep_recursion_through_the_entry() {
    fn fib(n: int) -> int {
        if n < 2 { return n }
        return fib(n - 1) + fib(n - 2)
    }
    assert_eq(fib(20), 6765)
}

fn bump(a: int) -> int { return a + 1 }

fn add_up(xs: list, n: int) -> int {
    var t = 0
    for i in 0..n { t = t + xs[i] }
    return t
}

fn test_a_body_that_bails_after_entry() {
    var s = 0
    for i in 0..300 { s = s + bump(i) }
    assert_eq(s, 45150)
    assert_throws(OverflowError, fn() -> void { bump(9223372036854775807) },
                  "an add past the largest int, inside a compiled body")
    let ints = [1, 2, 3, 4]
    var t = 0
    for i in 0..300 { t = t + add_up(ints, 4) }
    assert_eq(t, 3000)
    #: Same list, same entry: the element read inside the body is what
    #: changes, so the body leaves through a deopt, not a decline at entry.
    let mixed: list = [1, 2, 3, 4]
    mixed[2] = "x"
    assert_throws(TypeError, fn() -> void { add_up(mixed, 4) },
                  "an element that is no longer an int")
    assert_eq(add_up(mixed, 2), 3)
}
