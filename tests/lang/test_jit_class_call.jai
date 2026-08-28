#: `Klass.static_method(x)` inside a compiled body.
#:
#: It is NOT a field read plus a call, which is what an earlier attempt at
#: this assumed and why it measured nothing. The compiler emits
#:
#:     OP_GET_GLOBAL  "Box"
#:     OP_INT         1
#:     OP_INVOKE      "make"  1 arg
#:
#: an OP_INVOKE whose RECEIVER IS A CLASS, so an arm in OP_GET_FIELD never
#: sees it. The tier's own invoke arm dispatches on the receiver's kind and
#: took only instances, lists and objects; a class fell through and declined
#: the whole loop.
#:
#: It is really the same shape as `math.sqrt(x)`: a member of another
#: namespace, resolved at compile time, called with the receiver DROPPED --
#: the interpreter drops it too (resolveInvokeTarget leaves the class in
#: slot 0 and reports `isMethod == false`, so invokeCallable binds the first
#: ARGUMENT to the closure's first parameter).
#:
#: What this file holds down:
#:
#:   THE BINDING. `klass->statics.version` retires the baked callee. It is
#:   the counter that moves when `Box.make = other` overwrites the member --
#:   `keyVersion` does NOT, because overwriting an existing key never moves
#:   its entry, and a body pinned on keyVersion would call the old closure
#:   for ever. test_a_rebound_static_is_picked_up is that case: with the
#:   version guard removed it answers 400008 where 2000000 is correct.
#:
#:   WHAT IS NOT ADMITTED. An instance method IS reachable through the class
#:   -- `Box.twice(4)` resolves and runs -- but it gets the CLASS as `self`
#:   and raises the moment it touches a field. The arm declines it rather
#:   than compile a faster route to the same exception, and
#:   test_an_instance_method_through_the_class pins the exception itself.
#:
#: The answers here are the interpreter's, so `JAITHON_NO_JIT=1` proves the
#: expectations and the plain run proves the compiled code agrees. Under
#: `JAITHON_JIT_DEOPT_STRESS=1` every guard fires on every iteration, which
#: is what turns a wrong resume point into a wrong answer: the arguments are
#: still on the interpreter's stack at the version guard, and the call has
#: already happened at the result's tag guard, so a deopt that resumed at the
#: invoke rather than after it would call twice.
#:
#: JAITHON_JIT_CLASS_CALLS=0 turns the arm off; every assertion below holds
#: either way, because all of them are questions about the answer.
# jit-compiles: sum_box_doubles, sum_box_inverses, halves_until_zero
from std.test import assert_eq, assert_throws

fn double_it(v: int) -> int { return v + v }

fn times_ten(v: int) -> int { return v * 10 }

class Box {
    pub var v: int

    fn init(self, v: int) -> void { self.v = v }

    #: A `static fn` cannot be reassigned -- the checker rejects it with
    #: E0702 -- so the rebind test below needs a static VAR holding a
    #: function. Both compile to the same OP_INVOKE on the class, and both
    #: live in `klass->statics`, which is what the arm reads.
    pub static var make: fn(int) -> int = double_it

    pub static fn twice(v: int) -> int { return v + v }

    #: Reached as `Box.twice_inst(4)` by the test below, where `self` is the
    #: CLASS rather than an instance.
    pub fn twice_inst(self, k: int) -> int { return self.v + k }

    #: Returns an INSTANCE, so the result's kind is a class shape rather than
    #: a scalar and emitCallOutResult has an object type and a shapeId to
    #: check as well as a tag.
    pub static fn of(v: int) -> Box { return Box(v) }
}

class Ratio {
    #: Raises for v == 0. The throw arrives from inside the descriptor call
    #: and the interpreter owns the exception from there.
    pub static fn invert(v: int) -> int { return 100 // v }
}

#: The plain shape: one static call per iteration, its int result consumed by
#: the arithmetic around it. A wrong result kind shows up as a wrong sum
#: rather than as a crash, since the tag guard would send every iteration
#: back to the interpreter.
fn sum_box_doubles(n: int) -> int {
    var t = 0
    var i = 0
    while i < n {
        t += Box.make(i & 15) + Box.twice(i & 7)
        i += 1
    }
    return t
}

#: The same, through a static that hands back an instance the loop then reads
#: a field off.
fn sum_box_fields(n: int) -> int {
    var t = 0
    var i = 0
    while i < n {
        t += Box.of(i & 7).v
        i += 1
    }
    return t
}

fn sum_box_inverses(n: int) -> int {
    var t = 0
    var i = 1
    while i <= n {
        t += Ratio.invert(i)
        i += 1
    }
    return t
}

#: The bad value starts at `bad_at`, not at zero: a loop enters its compiled
#: form on a back edge, so a value that went wrong on iteration zero would
#: raise before the compiled body ever ran. `log` is what proves the
#: iterations BEFORE the bad one happened exactly once each -- a deopt that
#: resumed at the invoke rather than after it would push twice.
fn halves_until_zero(n: int, bad_at: int, log: list[int]) -> int {
    var t = 0
    var i = 0
    while i < n {
        var d = 4
        if i >= bad_at { d = 0 }
        log.push(i)
        t += Ratio.invert(d)
        i += 1
    }
    return t
}

#: Hotness is SIGPROF ticks on back edges, not iterations, so these are
#: durations. Nothing below asserts a compile happened -- every assertion is
#: correct interpreted -- but scripts/jit_compile_check.py fails if the
#: marked names stop reaching the compiled tier.
fn warm_doubles() -> void {
    for _r in 0..2000 { sum_box_doubles(64) }
}

fn warm_fields() -> void {
    for _r in 0..2000 { sum_box_fields(64) }
}

fn warm_inverses() -> void {
    for _r in 0..2000 { sum_box_inverses(64) }
}

#: `bad_at` past the end, so the warm-up never raises: what has to be hot is
#: the body the throw will later interrupt, and a loop that raises on every
#: pass never gets far enough round to be sampled.
fn warm_halves() -> void {
    for _r in 0..2000 {
        let log: list[int] = []
        halves_until_zero(64, 99, log)
    }
}

fn test_a_static_in_a_hot_loop_agrees_with_the_interpreter() -> void {
    warm_doubles()
    #: Asserted at three lengths so a body that ran the wrong number of terms
    #: is visible rather than merely wrong by a constant.
    assert_eq(sum_box_doubles(16), 240 + 112)
    assert_eq(sum_box_doubles(32), 480 + 224)
    assert_eq(sum_box_doubles(64), 960 + 448)
    assert_eq(sum_box_doubles(4), 0 + 2 + 4 + 6 + 0 + 2 + 4 + 6)
}

fn test_a_static_returning_an_instance() -> void {
    warm_fields()
    #: 0..7 repeated: the field read after the call is what needs the
    #: returned object's type and shape checked, not just its tag.
    assert_eq(sum_box_fields(8), 28)
    assert_eq(sum_box_fields(64), 28 * 8)
}

fn test_a_static_that_raises_leaves_the_loop_at_the_call() -> void {
    warm_inverses()
    warm_halves()
    let log: list[int] = []
    assert_throws(DivisionByZeroError, || halves_until_zero(64, 32, log))
    #: 33 entries: 0..31 ran whole, and the 32nd reached the call that threw.
    assert_eq(log.len(), 33)
    assert_eq(log[32], 32)
    #: Nothing is retried after the exception is caught.
    assert_eq(halves_until_zero(4, 99, log), 100)
    assert_eq(log.len(), 37)
}

fn test_a_rebound_static_is_picked_up() -> void {
    warm_doubles()
    assert_eq(sum_box_doubles(64), 960 + 448)

    Box.make = times_ten
    #: The compiled body baked the closure it resolved. Without the guard on
    #: `klass->statics.version` this still answers 960 + 448 -- for ever.
    assert_eq(sum_box_doubles(64), 4800 + 448)

    #: And back, through a body that has by now recompiled against the second
    #: binding, so the guard is exercised in both directions.
    for _r in 0..2000 { sum_box_doubles(64) }
    Box.make = double_it
    assert_eq(sum_box_doubles(64), 960 + 448)
}

#: An instance method named through the class. The checker allows it and the
#: interpreter runs it: `self` becomes the CLASS, so the first field read
#: raises AttributeError. The compiled tier must answer exactly that, which
#: it does by declining the site -- the arm reads `klass->statics` and never
#: `klass->methods`.
fn call_inst_through_class(n: int) -> int {
    var t = 0
    var i = 0
    while i < n {
        t += Box.twice_inst(4)
        i += 1
    }
    return t
}

fn test_an_instance_method_through_the_class() -> void {
    for _r in 0..200 {
        assert_throws(AttributeError, || call_inst_through_class(64))
    }
}

#: A private static is reachable only from inside its own class -- the
#: checker rejects `Hidden.step(i)` from anywhere else with E0701. The arm
#: declines a non-public member, because visibility is the interpreter's
#: `methodPermitted` and that reads the RUNNING FRAME's owner class, which a
#: once-and-for-all compile-time resolve cannot re-ask. This pins the answer,
#: not the decline.
class Hidden {
    static fn step(v: int) -> int { return v + 1 }

    pub static fn walk(n: int) -> int {
        var t = 0
        var i = 0
        while i < n {
            t += Hidden.step(i & 3)
            i += 1
        }
        return t
    }
}

fn test_a_private_static_still_answers() -> void {
    for _r in 0..2000 { Hidden.walk(64) }
    assert_eq(Hidden.walk(8), 1 + 2 + 3 + 4 + 1 + 2 + 3 + 4)
    assert_eq(Hidden.walk(64), (1 + 2 + 3 + 4) * 16)
}
