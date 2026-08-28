#: `math.sqrt(x)` inside a compiled body.
#:
#: A module member written in Jaithon is a CLOSURE, not a builtin, so the
#: OP_INVOKE arm used to refuse the site outright -- bare, with no reason
#: recorded, which is why a decline census could only ever say "OP_INVOKE".
#: It is not really a method call: it is a global call whose callee is
#: resolved through another module, and what was missing was the pinning.
#:
#: Two things are pinned, and this file exists to hold both of them down:
#:
#:   THE RECEIVER. `math` is read by ADDRESS behind a tag guard, so "some
#:   module" is all the type guard proves; the compiled body compares the
#:   receiver against the one ObjModule it resolved against.
#:
#:   THE BINDING. ObjModule::version is the counter for a memoised global
#:   value or a resolved callee, and `math.sqrt = f` moves it.
#:   `globals.keyVersion` -- the counter the by-address global load uses --
#:   does NOT move, because overwriting an existing key never moves its
#:   entry, so a body pinned on keyVersion would call the old closure for
#:   ever. test_a_rebound_module_member_is_picked_up is that case, and it
#:   fails with the version guard removed.
#:
#: The answers here are the interpreter's, so `JAITHON_NO_JIT=1` proves the
#: expectations and the plain run proves the compiled code agrees. Under
#: `JAITHON_JIT_DEOPT_STRESS=1` every guard fires on every iteration, which
#: is what turns a wrong resume point into a wrong answer: the receiver and
#: the arguments are still on the interpreter's stack at the guards, and the
#: call has already happened at the result's tag guard, so a deopt that
#: resumed at the invoke rather than after it would call twice.
#:
#: JAITHON_JIT_MODULE_CALLS=0 turns the arm off; every assertion below holds
#: either way, because all of them are questions about the answer.
# jit-compiles: sum_module_roots, sum_module_floors, floors_through
from std.test import assert_eq, assert_near, assert_throws
import std.math
import std.random

#: The plain shape: one module call per iteration, its float result consumed
#: by the arithmetic around it. A wrong result kind shows up here as a wrong
#: sum rather than as a crash, since the tag guard would send every iteration
#: back to the interpreter.
fn sum_module_roots(n: int) -> float {
    var t = 0.0
    var i = 0
    while i < n {
        let x = float(i % 16) + 1.0
        t = t + math.sqrt(x)
        i += 1
    }
    return t
}

#: `math.sqrt(-1.0)` raises. The throw arrives from inside the descriptor
#: call, and the interpreter owns the exception from there -- what this pins
#: is that the partial sum is abandoned rather than half-applied, and that
#: the iterations BEFORE the bad one happened exactly once each.
#:
#: The bad value starts at `bad_at`, not at zero: a loop enters its compiled
#: form on a back edge, so a value that went wrong on iteration zero would
#: raise before the compiled body ever ran.
fn roots_until_negative(n: int, bad_at: int, log: list[int]) -> float {
    var t = 0.0
    var i = 0
    while i < n {
        var x = 4.0
        if i >= bad_at { x = -1.0 }
        log.push(i)
        t = t + math.sqrt(x)
        i += 1
    }
    return t
}

#: Rebound between calls by the test below. Not `sqrt`: rebinding a member
#: the rest of the suite uses would outlive this file.
fn stand_in(x: float) -> float { return 5.0 }

fn sum_module_floors(n: int) -> float {
    var t = 0.0
    var i = 0
    while i < n {
        t = t + math.floor(2.7)
        i += 1
    }
    return t
}

#: The receiver arrives as a PARAMETER, so which module it is cannot be read
#: off the call site. The compiled body resolved `floor` against the module it
#: was given while warming up and compares the receiver against exactly that
#: one; handed another module it must not call the closure it baked. An import
#: is an immutable binding, so a parameter is the only way a receiver changes.
fn floors_through(m: any, n: int) -> float {
    var t = 0.0
    var i = 0
    while i < n {
        t = t + m.floor(2.7)
        i += 1
    }
    return t
}

#: Hotness is SIGPROF ticks on back edges, not iterations, so these are
#: durations. Nothing below asserts a compile happened -- every assertion is
#: correct interpreted -- but scripts/jit_compile_check.py fails if the
#: marked names stop reaching the compiled tier.
fn warm_roots() -> void {
    for _r in 0..2000 { sum_module_roots(64) }
}

fn warm_floors() -> void {
    for _r in 0..2000 { sum_module_floors(64) }
}

fn warm_through() -> void {
    for _r in 0..2000 { floors_through(math, 64) }
}

fn test_a_module_function_in_a_loop_agrees_with_the_interpreter() -> void {
    warm_roots()
    #: sqrt(1)..sqrt(16) is 1 + 1.414.. + ... ; asserted against itself at two
    #: lengths so a body that summed the wrong number of terms is visible.
    let one_pass = sum_module_roots(16)
    assert_near(sum_module_roots(32), one_pass * 2.0)
    assert_near(sum_module_roots(64), one_pass * 4.0)
    assert_near(sum_module_roots(4), 1.0 + math.sqrt(2.0) + math.sqrt(3.0) + 2.0)
}

fn test_a_module_function_that_raises_leaves_the_loop_at_the_call() -> void {
    warm_roots()
    let log: list[int] = []
    assert_throws(ValueError, || roots_until_negative(64, 32, log))
    #: 33 entries: 0..31 ran whole, and the 32nd reached the call that threw.
    assert_eq(log.len(), 33)
    assert_eq(log[32], 32)
    #: Nothing is retried after the exception is caught.
    assert_eq(roots_until_negative(4, 99, log), 8.0)
    assert_eq(log.len(), 37)
}

fn test_another_module_in_the_receiver_does_not_reach_the_baked_callee() -> void {
    warm_through()
    assert_near(floors_through(math, 64), 128.0)
    #: std.random has no exported `floor`. A body that ignored its receiver
    #: would answer 128.0 again instead of raising -- a wrong ANSWER, not a
    #: crash, which is why this is asserted rather than left to the guards.
    assert_throws(ImportError, || floors_through(random, 64))
    assert_near(floors_through(math, 64), 128.0)
}

#: A receiver for the bound-method case below. `add` is public so a module can
#: hold a bound copy of it.
class Adder {
    var base: float

    fn init(self, base: float) -> void { self.base = base }

    pub fn add(self, x: float) -> float { return self.base + x }
}

let adder = Adder(100.0)

#: A module member whose VALUE is a BOUND METHOD, which is what
#: `pub let handler = obj.method` puts in a module.
#:
#: jaiBuiltinMethod hands such a member back wrapped, and the invoke arm
#: unwraps it -- `onative = AS_BOUND(obound)->method` -- because the builtin
#: path below wants the bare native with the receiver already in the callee
#: slot. The module arm cannot use that unwrapped value: it DROPS the receiver,
#: so the bound `self` has nowhere left to come from. It called the unbound
#: closure with the first argument where `self` belongs and `self.base` raised
#: "'fn' object has no attribute 'base'" -- on code the interpreter answers.
#:
#: Rebinding is the only way to put one in a module from a single file; the
#: second warm-up is what makes the body recompile against the bound member,
#: which is the state that was wrong. Without the `!IS_BOUND` guard this fails
#: 8 runs out of 8.
fn test_a_module_member_that_is_a_bound_method() -> void {
    warm_floors()
    assert_near(sum_module_floors(64), 128.0)

    let original = math.floor
    math.floor = adder.add
    for _r in 0..20000 { sum_module_floors(64) }
    assert_near(sum_module_floors(4), 4.0 * 102.7)

    math.floor = original
    assert_near(sum_module_floors(64), 128.0)
}

fn test_a_rebound_module_member_is_picked_up() -> void {
    warm_floors()
    assert_near(sum_module_floors(64), 128.0)

    let original = math.floor
    math.floor = stand_in
    #: The compiled body baked the closure it resolved. Without the version
    #: guard this still answers 128.0 -- for ever.
    assert_near(sum_module_floors(64), 320.0)

    math.floor = original
    assert_near(sum_module_floors(64), 128.0)
}
