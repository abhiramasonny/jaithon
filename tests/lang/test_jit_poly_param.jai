#: A parameter that calls pass several classes through.
#:
#: The whole-function tier compiles on the call that crosses its threshold,
#: specialised to the class that call passed; a later call passing another
#: class is declined at entry and runs interpreted. When that keeps happening
#: the body is compiled again with the parameter unpinned -- its calls go
#: through the site's inline cache -- and the old form is kept if the new one
#: will not compile. What has to hold:
#:
#: - every class's own method runs (the classes lay their fields out
#:   differently, so a call reaching the wrong body reads the wrong number);
#: - a body that reads a FIELD off the parameter, which an unpinned receiver
#:   cannot do, keeps working through the form it already had;
#: - a recursive body passing the parameter to itself, and a method whose own
#:   receiver must stay pinned, agree with the interpreter.
#:
#: Each function is called well past the threshold and past the decline count
#: that triggers the recompile.
# jit-compiles: weigh, depth_sum
from std.test import assert_eq

trait Shape {
    fn area(self) -> int
}

class Sq: Shape {
    pub var tag: str
    pub var s: int

    fn init(self, s: int) {
        self.tag = "sq"
        self.s = s
    }

    pub fn area(self) -> int { return self.s * self.s }
}

class Rect: Shape {
    pub var w: int
    pub var h: int

    fn init(self, w: int, h: int) {
        self.w = w
        self.h = h
    }

    pub fn area(self) -> int { return self.w * self.h }
}

class Tri: Shape {
    pub var h: int
    pub var b: int

    fn init(self, b: int, h: int) {
        self.b = b
        self.h = h
    }

    pub fn area(self) -> int { return self.b * self.h // 2 }
}

fn weigh(s: Shape, k: int) -> int {
    return s.area() * k + 1
}

fn shapes(areas: list[int]) -> list[Shape] {
    var out: list[Shape] = []
    for i in 0..12 {
        let f = i % 5 + 1
        if i % 3 == 0 {
            out.push(Sq(f))
            areas.push(f * f)
        } elif i % 3 == 1 {
            out.push(Rect(f, 2))
            areas.push(f * 2)
        } else {
            out.push(Tri(f, 3))
            areas.push(f * 3 // 2)
        }
    }
    return out
}

fn test_every_class_through_one_parameter() {
    var areas: list[int] = []
    let xs = shapes(areas)
    for r in 0..200 {
        for j in 0..xs.len() {
            assert_eq(weigh(xs[j], r % 7), areas[j] * (r % 7) + 1)
        }
    }
}

#: Reads a field off the parameter: cannot be unpinned, keeps its form.
fn first_field(s: Sq, k: int) -> int {
    return s.s * k
}

fn test_a_field_read_keeps_the_pinned_form() {
    let a = Sq(3)
    for r in 0..300 {
        assert_eq(first_field(a, r), 3 * r)
    }
}

#: Recursion that hands the same parameter down.
fn depth_sum(s: Shape, n: int) -> int {
    if n == 0 { return 0 }
    return s.area() + depth_sum(s, n - 1)
}

fn test_recursion_through_an_unpinned_parameter() {
    var areas: list[int] = []
    let xs = shapes(areas)
    for _r in 0..100 {
        for j in 0..xs.len() {
            assert_eq(depth_sum(xs[j], 4), areas[j] * 4)
        }
    }
}

class Scale {
    pub var by: int

    fn init(self, by: int) { self.by = by }

    pub fn apply(self, s: Shape) -> int { return s.area() * self.by }
}

fn test_a_method_keeps_its_receiver_pinned() {
    let sc = Scale(3)
    var areas: list[int] = []
    let xs = shapes(areas)
    for _r in 0..200 {
        for j in 0..xs.len() {
            assert_eq(sc.apply(xs[j]), areas[j] * 3)
        }
    }
}

class Base {
    pub var a: int

    fn init(self, a: int) { self.a = a }
}

class Derived extends Base {
    pub var b: int

    fn init(self, a: int, b: int) {
        self.a = a
        self.b = b
    }
}

#: A field read off a parameter of two classes: the unpinned form cannot be
#: compiled, so the pinned one stays and the other class runs interpreted.
fn read_a(o: Base, k: int) -> int {
    return o.a * k
}

fn test_a_field_read_off_two_classes_keeps_the_old_form() {
    let p = Base(4)
    let q = Derived(5, 6)
    for r in 0..300 {
        assert_eq(read_a(p, r) + read_a(q, r), 9 * r)
    }
}
