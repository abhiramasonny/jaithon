#: A method or function whose body ends `return C(...)`, inlined where it is
#: called (JAITHON_JIT_INLINE_CONSTRUCT).
#:
#: The construction allocates, and the allocator's slow path is a real call
#: that may collect, so the inlined body's values share the caller's
#: callee-saved bank and every object the caller holds must still be a root.
#: Each case runs long enough for its loop to compile and allocates enough to
#: collect many times on the way.
from std.test import assert_eq

class Vec2 {
    pub var x: float
    pub var y: float

    fn init(self, x: float, y: float) {
        self.x = x
        self.y = y
    }

    pub fn add(self, other: Vec2) -> Vec2 {
        return Vec2(self.x + other.x, self.y + other.y)
    }

    pub fn dot(self, other: Vec2) -> float {
        return self.x * other.x + self.y * other.y
    }
}

class Cell {
    pub var value: int
    pub var next: Cell?

    fn init(self, value: int, next: Cell?) {
        self.value = value
        self.next = next
    }

    #: The new object holds a reference to the receiver, which must survive
    #: any collection the allocation triggers.
    pub fn push(self, value: int) -> Cell {
        return Cell(value, self)
    }
}

#: Not a simple init: it computes a field, so a construction of it is never
#: inlined, and a body ending with one goes through the call.
class Scaled {
    pub var v: int

    fn init(self, v: int) {
        self.v = v * 3
    }

    pub fn bump(self, by: int) -> Scaled {
        return Scaled(self.v + by)
    }
}

fn make_pair(a: int, b: int) -> Cell {
    return Cell(a + b, null)
}

fn walk_sum(head: Cell?) -> int {
    var total = 0
    var at = head
    while at != null {
        total += at.value
        at = at.next
    }
    return total
}

fn accumulate(n: int) -> float {
    var acc = Vec2(0.0, 0.0)
    let step = Vec2(1.0, 2.0)
    var total = 0.0
    for _ in 0..n {
        acc = acc.add(step)
        total += acc.dot(step)
    }
    return total
}

fn build_chain(n: int) -> Cell {
    var head = Cell(0, null)
    for i in 1..n {
        head = head.push(i)
    }
    return head
}

fn test_a_method_that_constructs_in_a_compiled_loop() -> void {
    assert_eq(accumulate(200000), 100000500000.0)
}

fn test_the_receiver_survives_the_allocation() -> void {
    var got = 0
    for _ in 0..20 { got = walk_sum(build_chain(20000)) }
    assert_eq(got, 199990000)
}

fn test_a_function_that_constructs() -> void {
    var total = 0
    for i in 0..300000 {
        total += make_pair(i, 1).value
    }
    assert_eq(total, 45000150000)
}

fn test_an_init_that_computes_is_still_called() -> void {
    var s = Scaled(1)
    for _ in 0..30 {
        s = s.bump(1)
    }
    assert_eq(s.v, 926510094425919)
    var t = Scaled(1)
    t = t.bump(2)
    assert_eq(t.v, 15)
}

#: The field kinds change under a compiled caller: the inlined body's field
#: guards fail, and the call is made again from the start.
fn test_a_receiver_of_another_kind() -> void {
    for _ in 0..3 { accumulate(50000) }
    let a = Vec2(1.5, 2.5)
    let b = a.add(Vec2(0.5, 0.5))
    assert_eq(b.x, 2.0)
    assert_eq(b.y, 3.0)
}
