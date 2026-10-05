#: A compiled callee that writes and then deoptimises runs its writes ONCE.
#:
#: A compiled callee that meets a value it was not specialised for part-way
#: through answers its caller with verdict 4: "I stopped here, and here is my
#: record". Whatever it did before that guard has happened. The arm for a
#: closure held in a local or parameter -- `let r = f(x)` -- answered every
#: nonzero verdict by deoptimising at its own call, so the interpreter called
#: the callee again from the top and every write before the guard happened
#: twice: one `log.push(n)` became two. emitDirectCall never had this bug,
#: because it finishes a writing callee from its own record instead, and the
#: closure arm now does the same with the closure it read from its register.
#:
#: The same arm also failed to tell its own callers that it writes, so a body
#: holding `f(x)` advertised itself as having no effects. A compiled caller of
#: THAT body re-ran it from the top when it deoptimised after the call, and
#: the callee's writes happened twice one level further out.
#:
#: And finishing a callee from its record dropped any parameter the compiled
#: body never read: the record has no value for it, so the interpreted rest of
#: the body read null where the caller had passed 5.
#:
#: Every shape below warms its callee past the call threshold while `holder`
#: is exactly a Base, then swaps in a subclass so the field read AFTER the
#: write fails its class guard, and counts the writes.
# jit-compiles: through_a_param, through_a_param_field, through_a_param_global, through_a_param_b, through_a_param_up, loops_through_a_local, logs_then_reads, outer_reads_after, drive_outer, reads_late, drive_unread
from std.test import assert_eq

let WARM = 300

class Base {
    pub var k: int

    pub fn init(self, k: int) { self.k = k }
}

class Wide extends Base {
    pub var extra: int

    fn init(self, k: int) {
        super.init(k)
        self.extra = 7
    }
}

class Tally {
    pub var count: int

    pub fn init(self) { self.count = 0 }
}

var holder: Base = Base(1)
var log: list[int] = []
var tally: Tally = Tally()
var bumps = 0

fn logs_then_reads(n: int) -> int {
    log.push(n)
    return holder.k + n
}

fn counts_then_reads(n: int) -> int {
    tally.count += 1
    return holder.k + n
}

fn bumps_then_reads(n: int) -> int {
    bumps += 1
    return holder.k + n
}

fn halves_then_reads(n: int) -> float {
    log.push(n)
    return float(holder.k + n) / 2.0
}

fn through_a_param(f: fn(int) -> int, x: int) -> int {
    let r = f(x)
    return r + 0
}

fn through_a_param_f(f: fn(int) -> float, x: int) -> float {
    let r = f(x)
    return r + 0.0
}

#: One caller per callee: a compiled caller guards the ObjFunction it saw, so
#: sharing one would leave every later test calling through the interpreter.
fn through_a_param_field(f: fn(int) -> int, x: int) -> int {
    let r = f(x)
    return r + 0
}

fn through_a_param_global(f: fn(int) -> int, x: int) -> int {
    let r = f(x)
    return r + 0
}

fn reset() -> void {
    holder = Base(1)
    log = []
    tally = Tally()
    bumps = 0
}

fn test_a_closure_call_pushes_once() -> void {
    reset()
    var t = 0
    for i in 0..WARM {
        t += through_a_param(logs_then_reads, 0)
    }
    holder = Wide(2)
    t += through_a_param(logs_then_reads, 1)
    assert_eq(t, WARM + 3)
    assert_eq(len(log), WARM + 1)
}

fn test_a_closure_call_writes_a_field_once() -> void {
    reset()
    var t = 0
    for i in 0..WARM {
        t += through_a_param_field(counts_then_reads, 0)
    }
    holder = Wide(2)
    t += through_a_param_field(counts_then_reads, 1)
    assert_eq(t, WARM + 3)
    assert_eq(tally.count, WARM + 1)
}

fn test_a_closure_call_writes_a_global_once() -> void {
    reset()
    var t = 0
    for i in 0..WARM {
        t += through_a_param_global(bumps_then_reads, 0)
    }
    holder = Wide(2)
    t += through_a_param_global(bumps_then_reads, 1)
    assert_eq(t, WARM + 3)
    assert_eq(bumps, WARM + 1)
}

fn test_a_float_closure_call_pushes_once() -> void {
    reset()
    var t = 0.0
    for i in 0..WARM {
        t += through_a_param_f(halves_then_reads, 0)
    }
    holder = Wide(2)
    t += through_a_param_f(halves_then_reads, 1)
    assert_eq(t, float(WARM) / 2.0 + 1.5)
    assert_eq(len(log), WARM + 1)
}

fn above_then_reads(n: int) -> bool {
    log.push(n)
    return holder.k > n
}

fn through_a_param_b(f: fn(int) -> bool, x: int) -> int {
    let r = f(x)
    if r {
        return 1
    }
    return 0
}

fn test_a_bool_closure_call_pushes_once() -> void {
    reset()
    var t = 0
    for i in 0..WARM {
        t += through_a_param_b(above_then_reads, 0)
    }
    holder = Wide(2)
    t += through_a_param_b(above_then_reads, 1)
    assert_eq(t, WARM + 1)
    assert_eq(len(log), WARM + 1)
}

fn through_a_param_up(f: fn(int) -> int, x: int) -> int {
    let r = f(x)
    return r + 0
}

#: The callee reads an upvalue, so the call hands it the closure it was
#: given -- and finishing it must use that closure, not one baked in.
fn test_a_capturing_closure_call_pushes_once() -> void {
    reset()
    var base = 10
    let g = fn(n: int) -> int {
        log.push(n)
        return holder.k + n + base
    }
    var t = 0
    for i in 0..WARM {
        t += through_a_param_up(g, 0)
    }
    holder = Wide(2)
    t += through_a_param_up(g, 1)
    assert_eq(t, WARM * 11 + 13)
    assert_eq(len(log), WARM + 1)
}

#: The OSR loop tier makes the same call through the same arm. Long enough
#: that the sampler has compiled the loop well before the swap.
fn loops_through_a_local(n: int, swap_at: int) -> int {
    let f = logs_then_reads
    var t = 0
    for i in 0..n {
        if i == swap_at {
            holder = Wide(2)
        }
        t += f(0)
    }
    return t
}

fn test_an_osr_loop_closure_call_pushes_once() -> void {
    reset()
    let t = loops_through_a_local(100000, 99500)
    assert_eq(t, 99500 + 500 * 2)
    assert_eq(len(log), 100000)
}

#: The call returns normally; it is the caller that deoptimises, after it.
fn outer_reads_after(f: fn(int) -> int, x: int) -> int {
    let r = f(x)
    return r + holder.k
}

fn pushes(n: int) -> int {
    log.push(n)
    return n + 1
}

#: Takes the closure as a parameter, so the call to `outer_reads_after` passes
#: it in a register and can be a direct call.
fn drive_outer(n: int, f: fn(int) -> int) -> int {
    var t = 0
    for i in 0..n {
        t += outer_reads_after(f, 0)
    }
    return t
}

fn test_a_caller_of_a_closure_call_is_not_rerun() -> void {
    reset()
    var t = 0
    for i in 0..WARM {
        t += drive_outer(2, pushes)
    }
    holder = Wide(2)
    t += drive_outer(1, pushes)
    assert_eq(t, WARM * 4 + 3)
    assert_eq(len(log), WARM * 2 + 1)
}

#: `d` is null on every call that warms this, so the compiled body never reads
#: it -- and the walk stops at the lambda (no arm), so the read of `d` is
#: interpreted. Finishing the call there used to hand the interpreter a null
#: `d`. The caller is warmed with a real `d`, so it passes one in a register.
fn reads_late(n: int, d: int?) -> int {
    log.push(n)
    if n < 100000 {
        let g = |x| x + n
        if d is not null {
            return n + d
        }
        return n + 1
    }
    return n
}

fn drive_unread(k: int, d: int?) -> int {
    var t = 0
    for i in 0..k {
        t += reads_late(i, d)
    }
    return t
}

fn test_a_finished_callee_keeps_an_unread_argument() -> void {
    reset()
    var w = 0
    for i in 0..WARM {
        w += reads_late(i % 2, null)
    }
    assert_eq(w, WARM * 3 / 2)
    var t = 0
    for i in 0..WARM {
        t += drive_unread(2, 5)
    }
    assert_eq(t, WARM * 11)
    assert_eq(len(log), WARM * 3)
}
