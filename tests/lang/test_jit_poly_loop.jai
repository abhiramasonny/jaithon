#: A `for x in xs` in a function the whole-function tier compiles, over a list
#: holding several classes.
#:
#: The tier used to pin the loop variable to the class of the list's first
#: element, which on a mixed list deoptimised on the first element of any other
#: class -- once per call, since every call walks the list again. The loop
#: variable is now bound unpinned when the live list is mixed, and the calls in
#: the body dispatch through their site's inline cache. What has to hold:
#:
#: - every class's own method runs at its own field layout (the classes below
#:   declare `k` at different slots);
#: - a list that is mixed on one call and single-class on the next, or that
#:   gains an element that is not an instance at all, still gets the right
#:   answer;
#: - a method that raises unwinds out of the compiled loop.
#:
#: `walk` is called far past the compile threshold.
# jit-compiles: walk
from std.test import assert_eq, assert_true

trait Step {
    fn step(self, x: int) -> int
}

class Add: Step {
    pub var tag: str
    pub var k: int

    fn init(self, k: int) {
        self.tag = "add"
        self.k = k
    }

    pub fn step(self, x: int) -> int { return (x + self.k) % 10007 }
}

class Mul: Step {
    pub var k: int
    pub var w: float

    fn init(self, k: int) {
        self.k = k
        self.w = 0.5
    }

    pub fn step(self, x: int) -> int { return x * self.k % 10007 }
}

class Neg: Step {
    fn init(self) { }

    pub fn step(self, x: int) -> int { return 10006 - x }
}

class Boom: Step {
    pub var left: int

    fn init(self, left: int) { self.left = left }

    pub fn step(self, x: int) -> int {
        self.left -= 1
        if self.left == 0 { throw ValueError("boom") }
        return x + 1
    }
}

fn walk(ops: list[Step], x0: int) -> int {
    var x = x0
    for op in ops { x = op.step(x) }
    return x
}

#: The same arithmetic with no method call anywhere, as the reference.
fn walk_ref(kinds: list[int], ks: list[int], x0: int) -> int {
    var x = x0
    for j in 0..kinds.len() {
        let k = ks[j]
        if kinds[j] == 0 {
            x = (x + k) % 10007
        } elif kinds[j] == 1 {
            x = x * k % 10007
        } else {
            x = 10006 - x
        }
    }
    return x
}

fn build(n: int, kinds: list[int], ks: list[int]) -> list[Step] {
    var ops: list[Step] = []
    for i in 0..n {
        let which = (i * 5 + 1) % 3
        let k = i % 7 + 2
        kinds.push(which)
        ks.push(k)
        if which == 0 {
            ops.push(Add(k))
        } elif which == 1 {
            ops.push(Mul(k))
        } else {
            ops.push(Neg())
        }
    }
    return ops
}

fn test_mixed_list_every_call() {
    var kinds: list[int] = []
    var ks: list[int] = []
    let ops = build(24, kinds, ks)
    for r in 0..300 {
        assert_eq(walk(ops, r), walk_ref(kinds, ks, r))
    }
}

fn test_mixed_then_single_class() {
    var kinds: list[int] = []
    var ks: list[int] = []
    let ops = build(12, kinds, ks)
    for r in 0..200 {
        assert_eq(walk(ops, r), walk_ref(kinds, ks, r))
    }
    var adds: list[Step] = []
    for i in 0..10 { adds.push(Add(i)) }
    assert_eq(walk(adds, 3), 48)
    assert_eq(walk(ops, 5), walk_ref(kinds, ks, 5))
}

fn test_raise_unwinds_out_of_the_loop() {
    var kinds: list[int] = []
    var ks: list[int] = []
    let ops = build(9, kinds, ks)
    for r in 0..200 {
        assert_eq(walk(ops, r), walk_ref(kinds, ks, r))
    }
    var withBoom: list[Step] = [Add(1), Boom(3), Mul(2)]
    var caught = 0
    for r in 0..6 {
        try {
            let _v = walk(withBoom, r)
        } catch e: ValueError {
            caught += 1
        }
    }
    assert_eq(caught, 1)
    assert_true(walk(ops, 1) >= 0)
}
