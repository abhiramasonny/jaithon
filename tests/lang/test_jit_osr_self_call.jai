#: A compiled LOOP that calls the function it is in.
#:
#: The self-call arm branches to instruction 0 of the body being compiled. For
#: a whole-body form that is the function's own prologue; for a loop form it
#: is a loop prologue that expects the interpreter's frame, so the arm refused
#: there -- silently -- and `for kid in kids { total += size(kid) }` never ran
#: as a compiled loop. When the function has a whole-body form, the loop now
#: calls THAT, as it would any compiled global (JAITHON_JIT_OSR_SELF_GLOBAL).
#:
#: Checked: the totals of a deep and a wide recursion, and a callee that
#: throws from inside the compiled loop.
from std.test import assert_eq, assert_throws

class Box {
    pub var value: int
    pub var kids: list[Box]
    fn init(self, value: int) {
        self.value = value
        self.kids = []
    }
}

fn size(b: Box) -> int {
    var total = b.value
    for kid in b.kids { total += size(kid) }
    return total
}

fn checked(b: Box, limit: int) -> int {
    if b.value > limit { throw ValueError("too big") }
    var total = b.value
    for kid in b.kids { total += checked(kid, limit) }
    return total
}

fn wide(n: int) -> Box {
    var root = Box(1)
    for i in 0..n {
        var mid = Box(i)
        for j in 0..3 { mid.kids.push(Box(j)) }
        root.kids.push(mid)
    }
    return root
}

fn chain(n: int) -> Box {
    var top = Box(0)
    var at = top
    for i in 1..n {
        var next = Box(i)
        at.kids.push(next)
        at = next
    }
    return top
}

pub fn test_a_wide_tree() -> void {
    let root = wide(800)
    # 1 + sum(i) + 3 per mid of (0 + 1 + 2)
    let expect = 1 + 799 * 800 // 2 + 800 * 3
    var got = 0
    for _r in 0..5 { got = size(root) }
    assert_eq(got, expect, "wide")
}

pub fn test_a_deep_chain() -> void {
    let top = chain(200)
    var got = 0
    for _r in 0..50 { got = size(top) }
    assert_eq(got, 199 * 200 // 2, "deep")
}

pub fn test_a_throw_from_inside_the_loop() -> void {
    let root = wide(600)
    var fine = 0
    for _r in 0..5 { fine = checked(root, 100000) }
    assert_eq(fine, 1 + 599 * 600 // 2 + 600 * 3, "no throw")
    assert_throws(ValueError, || checked(root, 400))
}
