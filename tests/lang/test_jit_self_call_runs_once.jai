#: A recursive body that finishes its own inner call in the interpreter runs
#: that call's writes ONCE.
#:
#: A compiled self-call answers an inner activation that deoptimised part-way
#: (verdict 4) by FINISHING it in the interpreter from its record. That runs
#: whatever the inner activation had left: a tail the walk never compiled (it
#: stops at a lambda), or a call into a callee whose own interpreted tail
#: writes. So the outer activation may have written by the time it continues
#: -- but the self-call never said so, and a body whose compiled instructions
#: write nothing advertised jitFuncNoWrite. A compiled caller answers such a
#: body's deopt by running it again from the top, so when the outer activation
#: then deoptimised AFTER its self-call, the inner call's writes happened
#: twice: one `log.push(1)` became two.
#:
#: Most shapes below warm the recursion past the call threshold while `holder`
#: is exactly a Base, then swap in a subclass so the field read after the
#: self-call fails its class guard, and count the writes. The last needs no
#: swap: the finished inner call answers null where the compiled self-call
#: took an int, and the outer activation deoptimises on that.
# jit-compiles: sc_leaf_push, sc_via_closure, sc_rec_direct, sc_via_direct, sc_tail_writer, sc_rec_tail, sc_via_tail, sc_rec_osr, sc_rec_bumps, sc_drive_bumps, sc_rec_tally, sc_drive_tally, sc_rec_null, sc_via_null
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

fn reset() -> void {
    holder = Base(1)
    log = []
    tally = Tally()
    bumps = 0
}

#: Writes nothing in its compiled part: the push sits after the lambda, where
#: the walk stops. The self-call finishes sc_leaf_push(0) in the interpreter,
#: which pushes.
fn sc_leaf_push(n: int) -> int {
    if n > 0 {
        let v = sc_leaf_push(n - 1)
        return v + holder.k
    }
    let g = |x| x + n
    log.push(1)
    return 0
}

fn sc_via_closure(f: fn(int) -> int, x: int) -> int {
    let q = f(x)
    return q + 0
}

fn test_a_closure_caller_runs_a_finished_self_call_once() -> void {
    reset()
    var t = 0
    for i in 0..WARM {
        t += sc_via_closure(sc_leaf_push, 1)
    }
    holder = Wide(2)
    t += sc_via_closure(sc_leaf_push, 1)
    assert_eq(t, WARM + 2)
    assert_eq(len(log), WARM + 1)
}

fn sc_rec_direct(n: int) -> int {
    if n > 0 {
        let v = sc_rec_direct(n - 1)
        return v + holder.k
    }
    let g = |x| x + n
    log.push(1)
    return 0
}

fn sc_via_direct(x: int) -> int {
    let q = sc_rec_direct(x)
    return q + 0
}

fn test_a_direct_caller_runs_a_finished_self_call_once() -> void {
    reset()
    var t = 0
    for i in 0..WARM {
        t += sc_via_direct(1)
    }
    holder = Wide(2)
    t += sc_via_direct(1)
    assert_eq(t, WARM + 2)
    assert_eq(len(log), WARM + 1)
}

#: Compiled only up to the lambda; the push after it is interpreted. Not
#: recursive, and it writes nothing it compiled.
fn sc_tail_writer(n: int) -> int {
    if n > 0 {
        return n
    }
    let g = |x| x + n
    log.push(1)
    return 0
}

#: Every instruction of this compiles and none of them writes: the push is in
#: sc_tail_writer's interpreted tail, which the self-call's finish runs.
fn sc_rec_tail(n: int) -> int {
    if n > 0 {
        let v = sc_rec_tail(n - 1)
        return v + holder.k
    }
    return sc_tail_writer(0) + 0
}

fn sc_via_tail(f: fn(int) -> int, x: int) -> int {
    let q = f(x)
    return q + 0
}

fn test_a_callee_tail_finished_by_a_self_call_runs_once() -> void {
    reset()
    var t = 0
    for i in 0..WARM {
        t += sc_via_tail(sc_rec_tail, 1)
    }
    holder = Wide(2)
    t += sc_via_tail(sc_rec_tail, 1)
    assert_eq(t, WARM + 2)
    assert_eq(len(log), WARM + 1)
}

fn sc_rec_osr(n: int) -> int {
    if n > 0 {
        let v = sc_rec_osr(n - 1)
        return v + holder.k
    }
    let g = |x| x + n
    log.push(1)
    return 0
}

#: The OSR loop tier makes the call through the closure arm. Long enough that
#: the sampler has compiled the loop well before the swap.
fn sc_loop_osr(n: int, swap_at: int) -> int {
    let f = sc_rec_osr
    var t = 0
    for i in 0..n {
        if i == swap_at {
            holder = Wide(2)
        }
        t += f(1)
    }
    return t
}

fn test_an_osr_loop_runs_a_finished_self_call_once() -> void {
    reset()
    let t = sc_loop_osr(100000, 99500)
    assert_eq(t, 99500 + 500 * 2)
    assert_eq(len(log), 100000)
}

fn sc_rec_bumps(n: int) -> int {
    if n > 0 {
        let v = sc_rec_bumps(n - 1)
        return v + holder.k
    }
    let g = |x| x + n
    bumps += 1
    return 0
}

fn sc_drive_bumps(x: int) -> int {
    let q = sc_rec_bumps(x)
    return q + 0
}

fn test_a_finished_self_call_writes_a_global_once() -> void {
    reset()
    var t = 0
    for i in 0..WARM {
        t += sc_drive_bumps(1)
    }
    holder = Wide(2)
    t += sc_drive_bumps(1)
    assert_eq(t, WARM + 2)
    assert_eq(bumps, WARM + 1)
}

fn sc_rec_tally(n: int) -> int {
    if n > 0 {
        let v = sc_rec_tally(n - 1)
        return v + holder.k
    }
    let g = |x| x + n
    tally.count += 1
    return 0
}

fn sc_drive_tally(x: int) -> int {
    let q = sc_rec_tally(x)
    return q + 0
}

fn test_a_finished_self_call_writes_a_field_once() -> void {
    reset()
    var t = 0
    for i in 0..WARM {
        t += sc_drive_tally(1)
    }
    holder = Wide(2)
    t += sc_drive_tally(1)
    assert_eq(t, WARM + 2)
    assert_eq(tally.count, WARM + 1)
}

#: No guard fails here. The compiled self-call takes an int; the finished inner
#: call answers null, so the outer activation deoptimises on the self-call's
#: own result record -- after the push has happened.
fn sc_rec_null(n: int, z: bool) -> int? {
    if n > 0 {
        let v = sc_rec_null(n - 1, z)
        return v
    }
    let g = |x| x + n
    log.push(1)
    if z {
        return null
    }
    return 7
}

fn sc_via_null(f: fn(int, bool) -> int?, x: int, z: bool) -> int {
    let q = f(x, z)
    if q is null {
        return 100
    }
    return q + 0
}

fn test_a_self_call_answered_with_another_kind_runs_once() -> void {
    reset()
    var t = 0
    for i in 0..WARM {
        t += sc_via_null(sc_rec_null, 1, false)
    }
    t += sc_via_null(sc_rec_null, 1, true)
    assert_eq(t, WARM * 7 + 100)
    assert_eq(len(log), WARM + 1)
}
