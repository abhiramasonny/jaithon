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
    for k in 0..20000 { want += (k % 10) * 3 + 6 }
    assert_eq(total, want)
}

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

    pub fn swapped(self) -> Vec2 {
        return Vec2(self.y, self.x)
    }
}

#: Each new object is built from the fields of the one it replaces, in the
#: other order: every argument must be read before any field is written.
fn swaps(n: int) -> Vec2 {
    var p = Vec2(1.0, 2.0)
    for _ in 0..n {
        p = p.swapped()
    }
    return p
}

fn swaps_direct(n: int) -> Vec2 {
    var p = Vec2(1.0, 2.0)
    var i = 0
    while i < n {
        p = Vec2(p.y, p.x)
        i += 1
    }
    return p
}

fn test_a_construction_from_the_object_it_replaces() -> void {
    assert_eq(swaps(20000).x, 1.0)
    assert_eq(swaps(20001).x, 2.0)
    assert_eq(swaps_direct(20000).y, 2.0)
    assert_eq(swaps_direct(20001).y, 1.0)
}

#: Loop-carried: the local is read before it is rebound, so the loop is
#: entered holding a real object, unpacked at entry, and every iteration's
#: object comes from an inlined `return Vec2(...)`.
fn accumulate(start: Vec2, n: int) -> Vec2 {
    var acc = start
    let step = Vec2(1.0, 2.0)
    var total = 0.0
    for _ in 0..n {
        acc = acc.add(step)
        total += acc.dot(step)
    }
    assert_eq(total, 5.0 * float(n) * float(n + 1) / 2.0 +
                     float(n) * (start.x + 2.0 * start.y))
    return acc
}

fn test_a_loop_carried_object() -> void {
    let start = Vec2(0.5, 0.25)
    let got = accumulate(start, 30000)
    assert_eq(got.x, 30000.5)
    assert_eq(got.y, 60000.25)
    assert_eq(got is start, false)
    assert_eq(start.x, 0.5)
}

#: Entered holding an object and left without rebinding it: the same object
#: comes back, not a copy.
fn maybe_rebind(start: Vec2, n: int, at: int) -> Vec2 {
    var acc = start
    let step = Vec2(1.0, 1.0)
    var total = 0.0
    var i = 0
    while i < n {
        total += acc.dot(step)
        if i == at { acc = acc.add(step) }
        i += 1
    }
    return acc
}

fn test_an_object_not_rebound_keeps_its_identity() -> void {
    let start = Vec2(2.0, 3.0)
    for _ in 0..4 { maybe_rebind(start, 20000, -1) }
    assert_eq(maybe_rebind(start, 20000, -1) is start, true)
    let moved = maybe_rebind(start, 20000, 5)
    assert_eq(moved is start, false)
    assert_eq(moved.x, 3.0)
}

#: Entered holding an object while the loop also calls out: the call could
#: reach that object, so the loop is compiled without sinking it.
var g_seen = 0.0
fn note(v: float) -> void {
    g_seen += v
}

fn test_a_call_out_beside_an_entered_object() -> void {
    var acc = Vec2(0.0, 0.0)
    let step = Vec2(1.0, 0.0)
    for _ in 0..20000 {
        acc = acc.add(step)
        note(acc.x)
    }
    assert_eq(acc.x, 20000.0)
    assert_eq(g_seen, 200010000.0)
}

#: Leaving by `break`, and skipping the rest by `continue`: both ways out of
#: an iteration see the object the iteration built.
fn first_big(n: int) -> Point {
    var p = Point(0, 0)
    var i = 0
    while i < n {
        p = Point(i, i * i)
        if p.y > 1000000 { break }
        i += 1
    }
    return p
}

fn odd_sum(n: int) -> int {
    var t = 0
    var i = 0
    while i < n {
        i += 1
        let p = Point(i, i % 2)
        if p.y == 0 { continue }
        t += p.x
    }
    return t
}

fn test_break_and_continue() -> void {
    let p = first_big(5000)
    assert_eq(p.x, 1001)
    assert_eq(p.y, 1002001)
    assert_eq(odd_sum(20000), 100000000)
}
