#: Module globals read and written in a loop that runs no Jaithon code, with
#: the globals table's keyVersion guard proved once at the loop head
#: (JAITHON_JIT_GLOBAL_GUARD_HOIST).
#:
#: An allocation of a simple-init class is the one call such a loop may hold:
#: it can collect, but it cannot add or remove a global. A loop that calls a
#: function keeps every guard where it was.
from std.test import assert_eq

class Pair {
    pub var x: int
    pub var y: int

    fn init(self, x: int, y: int) {
        self.x = x
        self.y = y
    }
}

var g_total = 0
var g_i = 0
while g_i < 200000 {
    let p = Pair(g_i % 10, g_i % 7)
    g_total = (g_total + p.x * p.y) % 1000003
    g_i += 1
}

var g_count = 0
var g_seen = 0

fn tally(n: int) -> int {
    g_count = 0
    var i = 0
    while i < n {
        g_count += i % 5
        i += 1
    }
    return g_count
}

fn note(v: int) -> void {
    g_seen += v
}

#: A call in the loop: the guard stays at every access.
fn tally_with_calls(n: int) -> int {
    g_seen = 0
    for i in 0..n {
        note(i % 3)
    }
    return g_seen
}

fn test_a_module_scope_loop_that_allocates() -> void {
    assert_eq(g_total, 699976)
    assert_eq(g_i, 200000)
}

fn test_a_function_loop_over_globals() -> void {
    var got = 0
    for _ in 0..100 { got = tally(3000) }
    assert_eq(got, 6000)
    assert_eq(tally(300000), 600000)
    assert_eq(g_count, 600000)
}

fn test_a_loop_that_calls_keeps_its_guards() -> void {
    var got = 0
    for _ in 0..100 { got = tally_with_calls(3000) }
    assert_eq(got, 3000)
    assert_eq(g_seen, 3000)
}

var g_any: any = 0

fn sum_any(n: int) -> any {
    g_any = g_any
    var i = 0
    while i < n {
        g_any = g_any + 1
        i += 1
    }
    return g_any
}

#: The global's kind changes between two entries to the same compiled loop:
#: the tag proved at the head must be proved again, not remembered.
fn test_a_global_that_changes_kind_between_entries() -> void {
    for _ in 0..50 {
        g_any = 0
        sum_any(2000)
    }
    g_any = 0
    assert_eq(sum_any(5000), 5000)
    g_any = 0.5
    assert_eq(sum_any(4), 4.5)
    g_any = 10
    assert_eq(sum_any(3000), 3010)
}

#: A module-scope loop that calls nothing: its globals live in registers for
#: the loop (JAITHON_JIT_GLOBAL_PROMOTE), written through to their entries.
var g_acc = 0
var g_k = 0
while g_k < 300000 {
    g_acc = (g_acc + g_k * 3) % 1000003
    g_k += 1
}

var g_big = 0

#: Overflows on the 64th step: the raise must leave the entry holding the
#: last value stored, which only writing through guarantees.
fn grow_big(n: int) -> int {
    var i = 0
    while i < n {
        g_big = g_big * 2 + 1
        i += 1
    }
    return g_big
}

fn test_a_promoted_module_loop() -> void {
    assert_eq(g_acc, 145003)
    assert_eq(g_k, 300000)
}

fn test_a_promoted_global_survives_a_raise() -> void {
    for _ in 0..100 {
        g_big = 0
        grow_big(40)
    }
    g_big = 0
    var raised = false
    try {
        let _unused = grow_big(100)
    } catch _failure {
        raised = true
    }
    assert_eq(raised, true)
    assert_eq(g_big, 9223372036854775807)
}
