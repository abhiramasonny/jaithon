#: A loop that builds an instance into a local and only reads its fields keeps
#: the fields and never allocates the object (JAITHON_JIT_SINK, jit_sink.c).
#:
#: The object must still exist, with the right fields and one identity, the
#: moment anything else can see the local: after the loop, at a deopt, when an
#: exception leaves the loop. Each case runs long enough for its loop to
#: compile, and every one must give the answer the interpreter gives.
from std.test import assert_eq

class Point {
    pub var x: int
    pub var y: int

    fn init(self, x: int, y: int) {
        self.x = x
        self.y = y
    }

    pub fn dot(self, other: Point) -> int {
        return self.x * other.x + self.y * other.y
    }

    pub fn sum(self) -> int {
        return self.x + self.y
    }
}

class FPair {
    pub var a: float
    pub var b: float

    fn init(self, a: float, b: float) {
        self.a = a
        self.b = b
    }
}

#: Not a simple init, so never sunk.
class Doubled {
    pub var v: int

    fn init(self, v: int) {
        self.v = v * 2
    }
}

fn churn(n: int) -> int {
    var total = 0
    var i = 0
    while i < n {
        let a = Point(i % 100, i % 37)
        let b = Point(i % 53, i % 11)
        total = (total + a.dot(b)) % 1000000007
        i += 1
    }
    return total
}

fn churn_reference(n: int) -> int {
    var total = 0
    for i in 0..n {
        let ax = i % 100
        let ay = i % 37
        let bx = i % 53
        let by = i % 11
        total = (total + ax * bx + ay * by) % 1000000007
    }
    return total
}

fn test_sunk_pair_gives_the_interpreters_answer() -> void {
    assert_eq(churn(30000), churn_reference(30000))
}

#: The local is read after the loop: the object is built on the way out.
fn last_point(n: int) -> Point {
    var p = Point(0, 0)
    var total = 0
    var i = 0
    while i < n {
        p = Point(i, i * 2)
        total += p.x + p.sum()
        i += 1
    }
    assert_eq(total, (n - 1) * n // 2 * 4)
    return p
}

fn test_the_object_exists_after_the_loop() -> void {
    let p = last_point(20000)
    assert_eq(p.x, 19999)
    assert_eq(p.y, 39998)
    let q = p
    assert_eq(q is p, true)
    var keep: list[Point] = []
    keep.push(p)
    assert_eq(keep[0] is p, true)
}

#: A loop that never runs leaves the local as it was.
fn test_a_loop_that_never_runs_keeps_the_old_object() -> void {
    for _ in 0..5 { last_point(5000) }
    let p = last_point(1)
    assert_eq(p.x, 0)
    assert_eq(p.y, 0)
}

#: Floats, read straight from the homes.
fn float_pairs(n: int) -> float {
    var acc = 0.0
    var i = 0
    while i < n {
        let f = FPair(float(i) * 0.5, 1.25)
        acc += f.a * f.b
        i += 1
    }
    return acc
}

fn test_float_fields() -> void {
    var want = 0.0
    for i in 0..20000 { want += float(i) * 0.5 * 1.25 }
    assert_eq(float_pairs(20000), want)
}

#: Stored into a list: the object escapes, so the loop keeps allocating, and
#: every element is its own object.
fn test_an_escaping_object_is_not_sunk() -> void {
    var kept: list[Point] = []
    var i = 0
    while i < 3000 {
        let p = Point(i, i + 1)
        kept.push(p)
        i += 1
    }
    assert_eq(kept[2999].y, 3000)
    assert_eq(kept[5] is kept[6], false)
    assert_eq(kept[5].x, 5)
}

#: Compared by identity inside the loop: not sunk either.
fn test_identity_inside_the_loop() -> void {
    var same = 0
    var i = 0
    let fixed = Point(1, 1)
    while i < 20000 {
        let p = Point(i, i)
        if p is fixed { same += 1 }
        if p is p { same += 1 }
        i += 1
    }
    assert_eq(same, 20000)
}

#: The read is a field of a value that changes kind part way: the guard on
#: it fails with the sunk objects live, and the interpreter resumes with them.
fn mixed(xs: list[any], n: int) -> int {
    var total = 0
    var i = 0
    while i < n {
        let a = Point(i % 7, i % 5)
        let b = Point(i % 3, 2)
        let w = xs[i % 4]
        total += a.dot(b) + int(w)
        i += 1
    }
    return total
}

fn test_a_deopt_with_sunk_objects_live() -> void {
    var want = 0
    for i in 0..20000 {
        want += (i % 7) * (i % 3) + (i % 5) * 2 + i % 4
    }
    assert_eq(mixed([0, 1, 2, 3], 20000), want)
    want = 0
    for i in 0..20000 {
        want += (i % 7) * (i % 3) + (i % 5) * 2 + i % 4
    }
    assert_eq(mixed([0, 1, 2.0, 3], 20000), want)
}

#: An exception out of the loop, with the local read by the handler.
fn overflow_in_loop(n: int) -> int {
    var p = Point(0, 0)
    var big = 1
    var i = 0
    try {
        while i < n {
            p = Point(i, big)
            big = big + i * 1000000000000
            i += 1
        }
    } catch _failure {
        return p.x
    }
    return -1
}

fn test_an_exception_with_a_sunk_object_live() -> void {
    assert_eq(overflow_in_loop(200000), 4295)
}

#: An init that computes is called, not sunk.
fn test_a_computing_init_is_left_alone() -> void {
    var total = 0
    var i = 0
    while i < 20000 {
        let d = Doubled(i)
        total += d.v
        i += 1
    }
    assert_eq(total, 399980000)
}

#: Passed to a method that is not inlined: the compile falls back to
#: allocating, and the answer is the same.
class Big {
    pub var n: int

    fn init(self, n: int) {
        self.n = n
    }

    pub fn spread(self, p: Point) -> int {
        var t = 0
        for k in 0..3 { t += p.x * k + self.n }
        return t
    }
}

fn test_a_call_that_is_not_inlined() -> void {
    let big = Big(2)
    var total = 0
    var i = 0
    while i < 20000 {
        let p = Point(i % 10, 0)
        total += big.spread(p)
        i += 1
    }
    var want = 0
    for i in 0..20000 { want += (i % 10) * 3 + 6 }
    assert_eq(total, want)
}
