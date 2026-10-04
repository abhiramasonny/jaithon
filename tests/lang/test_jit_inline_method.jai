#: Method bodies the compiled tier emits where the call is (inlineMethodCall).
#:
#: A one-expression method -- a field read off `self`, a constant, a modulo --
#: used to cost a full direct call: prologue, stack check, root fill, verdict.
#: It is now walked into the caller with its receiver as slot 0, and every field
#: read becomes a guarded load against the receiver's pinned class. What has to
#: hold:
#:
#: - The field is read at the RECEIVER's layout. The classes below declare their
#:   fields in different orders, so a read at another class's slot index would
#:   return a plausible wrong number rather than crash.
#: - Each scalar kind reads right: an int, a float (whose bits must not be read
#:   as an int), and a bool (one byte wide).
#: - An inherited method reads a subclass instance's fields correctly.
#: - Overflow inside an inlined body raises, as the call would have.
#: - A polymorphic loop (`for op in ops`) inlines one body per class, and every
#:   class still runs its own.
#:
#: The answers are checked against the same arithmetic written inline, which
#: never goes through a method at all.
# jit-compiles: sum_mono, sum_poly
from std.test import assert_eq, assert_true, assert_throws

trait Step {
    fn step(self, x: int) -> int
}

#: `k` at slot 1, behind a str.
class AddK: Step {
    pub var name: str
    pub var k: int

    fn init(self, k: int) {
        self.name = "add"
        self.k = k
    }

    pub fn step(self, x: int) -> int { return (x + self.k) % 1009 }
}

#: `k` at slot 0, a float behind it.
class MulK: Step {
    pub var k: int
    pub var w: float

    fn init(self, k: int) {
        self.k = k
        self.w = 0.25
    }

    pub fn step(self, x: int) -> int { return x * self.k % 1009 }
}

#: Two fields read, one through OP_GET_FIELD_LOCAL and one through a copy.
class Mix: Step {
    pub var flag: bool
    pub var a: int
    pub var b: int

    fn init(self, a: int, b: int) {
        self.flag = true
        self.a = a
        self.b = b
    }

    pub fn step(self, x: int) -> int { return (x * self.a + self.b) % 1009 }
}

fn sum_mono(m: Mix, n: int) -> int {
    var acc = 1
    var i = 0
    while i < n {
        acc = (acc + m.step(i)) % 1000003
        i += 1
    }
    return acc
}

fn sum_mono_inline(a: int, b: int, n: int) -> int {
    var acc = 1
    var i = 0
    while i < n {
        acc = (acc + (i * a + b) % 1009) % 1000003
        i += 1
    }
    return acc
}

fn test_monomorphic_field_reads() {
    let m = Mix(7, 3)
    for _r in 0..80 {
        assert_eq(sum_mono(m, 2000), sum_mono_inline(7, 3, 2000))
    }
}

fn sum_poly(ops: list[Step], reps: int) -> int {
    var acc = 1
    for _r in 0..reps {
        for op in ops { acc = op.step(acc) }
    }
    return acc
}

fn sum_poly_inline(kinds: list[int], ks: list[int], reps: int) -> int {
    var acc = 1
    for _r in 0..reps {
        for j in 0..kinds.len() {
            let k = ks[j]
            if kinds[j] == 0 {
                acc = (acc + k) % 1009
            } elif kinds[j] == 1 {
                acc = acc * k % 1009
            } else {
                acc = (acc * k + (k + 1)) % 1009
            }
        }
    }
    return acc
}

fn test_polymorphic_ways_read_their_own_layout() {
    var ops: list[Step] = []
    var kinds: list[int] = []
    var ks: list[int] = []
    for i in 0..48 {
        let k = i % 11 + 2
        let which = (i * 7 + 3) % 3
        kinds.push(which)
        ks.push(k)
        if which == 0 {
            ops.push(AddK(k))
        } elif which == 1 {
            ops.push(MulK(k))
        } else {
            ops.push(Mix(k, k + 1))
        }
    }
    assert_eq(sum_poly(ops, 600), sum_poly_inline(kinds, ks, 600))
}

class Weights {
    pub var bias: float
    pub var scale: float
    pub var on: bool

    fn init(self, bias: float, scale: float) {
        self.bias = bias
        self.scale = scale
        self.on = true
    }

    pub fn apply(self, x: float) -> float { return x * self.scale + self.bias }

    pub fn enabled(self) -> bool { return self.on }
}

fn run_weights(w: Weights, n: int) -> float {
    var t = 0.0
    var hits = 0
    var i = 0
    while i < n {
        if w.enabled() { hits += 1 }
        t = t + w.apply(1.5)
        i += 1
    }
    return t + hits
}

fn test_float_and_bool_fields() {
    let w = Weights(0.5, 2.0)
    for _r in 0..80 {
        assert_eq(run_weights(w, 1000), 4500.0)
    }
    w.on = false
    assert_eq(run_weights(w, 1000), 3500.0)
}

class Base {
    pub var a: int

    fn init(self, a: int) { self.a = a }

    pub fn plus(self, x: int) -> int { return x + self.a }
}

class Derived extends Base {
    pub var c: int

    fn init(self, a: int, c: int) {
        self.a = a
        self.c = c
    }
}

fn run_plus(o: Base, n: int) -> int {
    var t = 0
    var i = 0
    while i < n {
        t = o.plus(t) % 100003
        i += 1
    }
    return t
}

fn test_inherited_method_on_a_subclass() {
    for _r in 0..80 {
        assert_eq(run_plus(Base(5), 1000), 5000)
        assert_eq(run_plus(Derived(9, 100), 1000), 9000)
    }
}

class Big {
    pub var k: int

    fn init(self, k: int) { self.k = k }

    pub fn times(self, x: int) -> int { return x * self.k }
}

fn run_times(b: Big, x: int, n: int) -> int {
    var t = 0
    var i = 0
    while i < n {
        t = b.times(x)
        i += 1
    }
    return t
}

fn test_overflow_in_an_inlined_body_raises() {
    let b = Big(1 << 40)
    for _r in 0..80 {
        assert_eq(run_times(b, 3, 100), 3 << 40)
    }
    assert_throws(OverflowError, || run_times(b, 1 << 30, 100))
    assert_true(true)
}
