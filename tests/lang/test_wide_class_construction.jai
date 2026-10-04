#: `ClassName(args)` compiled for a class whose simple init stores into a
#: field slot of 30 or more. The compiled constructor writes each field as one
#: pair store, whose signed offset reaches only 504 bytes past the instance --
#: slot 29. Slot 30 and up encoded as a NEGATIVE offset: the field stayed null
#: and the value landed up to 512 bytes before the instance, inside whatever
#: object sat there. Each test builds the class in a hot loop so the
#: constructor compiles, and keeps some instances alive so a stray store into
#: a neighbour shows up as a non-null field nobody assigned.
from std.test import assert_eq

class Wide {
    pub var f0: int?
    pub var f1: int?
    pub var f2: int?
    pub var f3: int?
    pub var f4: int?
    pub var f5: int?
    pub var f6: int?
    pub var f7: int?
    pub var f8: int?
    pub var f9: int?
    pub var f10: int?
    pub var f11: int?
    pub var f12: int?
    pub var f13: int?
    pub var f14: int?
    pub var f15: int?
    pub var f16: int?
    pub var f17: int?
    pub var f18: int?
    pub var f19: int?
    pub var f20: int?
    pub var f21: int?
    pub var f22: int?
    pub var f23: int?
    pub var f24: int?
    pub var f25: int?
    pub var f26: int?
    pub var f27: int?
    pub var f28: int?
    pub var f29: int?
    pub var f30: int?
    pub var f31: int?
    pub var f32: int?
    pub var f33: int?
    pub var f34: int?
    pub var f35: int?
    pub var f36: int?
    pub var f37: int?
    pub var f38: int
    pub var f39: int

    fn init(self, a: int, b: int) {
        self.f38 = a
        self.f39 = b
    }
}

#: The smallest class that reaches the bad slot: 31 fields, the last stored.
class Edge {
    pub var e0: int?
    pub var e1: int?
    pub var e2: int?
    pub var e3: int?
    pub var e4: int?
    pub var e5: int?
    pub var e6: int?
    pub var e7: int?
    pub var e8: int?
    pub var e9: int?
    pub var e10: int?
    pub var e11: int?
    pub var e12: int?
    pub var e13: int?
    pub var e14: int?
    pub var e15: int?
    pub var e16: int?
    pub var e17: int?
    pub var e18: int?
    pub var e19: int?
    pub var e20: int?
    pub var e21: int?
    pub var e22: int?
    pub var e23: int?
    pub var e24: int?
    pub var e25: int?
    pub var e26: int?
    pub var e27: int?
    pub var e28: int?
    pub var e29: int
    pub var e30: int

    fn init(self, a: int, b: int) {
        self.e29 = a
        self.e30 = b
    }
}

#: A 30-field base, and a subclass whose own fields land at slots 30 and 31.
class Base30 {
    pub var b0: int?
    pub var b1: int?
    pub var b2: int?
    pub var b3: int?
    pub var b4: int?
    pub var b5: int?
    pub var b6: int?
    pub var b7: int?
    pub var b8: int?
    pub var b9: int?
    pub var b10: int?
    pub var b11: int?
    pub var b12: int?
    pub var b13: int?
    pub var b14: int?
    pub var b15: int?
    pub var b16: int?
    pub var b17: int?
    pub var b18: int?
    pub var b19: int?
    pub var b20: int?
    pub var b21: int?
    pub var b22: int?
    pub var b23: int?
    pub var b24: int?
    pub var b25: int?
    pub var b26: int?
    pub var b27: int?
    pub var b28: int?
    pub var b29: int?
}

class Derived32 extends Base30 {
    pub var a: int
    pub var b: int

    fn init(self, a: int, b: int) {
        self.a = a
        self.b = b
    }
}

class Cell {
    pub var v: int?

    fn init(self, v: int?) {
        self.v = v
    }
}

fn build_wide(n: int) -> int {
    var total = 0
    var kept: list[Wide] = []
    var cells: list[Cell] = []
    var i = 0
    while i < n {
        let c = Cell(i)
        let w = Wide(i, i + 1)
        total += w.f38 + w.f39
        if i % 50 == 0 {
            kept.push(w)
            cells.push(c)
        }
        i += 1
    }
    for w in kept {
        assert_eq(w.f0, null)
        assert_eq(w.f20, null)
        assert_eq(w.f37, null)
        assert_eq(w.f39, w.f38 + 1)
    }
    var k = 0
    for c in cells {
        assert_eq(c.v, k * 50)
        k += 1
    }
    return total
}

fn build_edge(n: int) -> int {
    var total = 0
    var kept: list[Edge] = []
    var i = 0
    while i < n {
        let e = Edge(i, i * 2)
        total += e.e29 + e.e30
        if i % 50 == 0 { kept.push(e) }
        i += 1
    }
    for e in kept {
        assert_eq(e.e28, null)
        assert_eq(e.e30, e.e29 * 2)
    }
    return total
}

fn build_derived(n: int) -> int {
    var total = 0
    var kept: list[Derived32] = []
    var i = 0
    while i < n {
        let d = Derived32(i, i + 3)
        total += d.a + d.b
        if i % 50 == 0 { kept.push(d) }
        i += 1
    }
    for d in kept {
        assert_eq(d.b29, null)
        assert_eq(d.b, d.a + 3)
    }
    return total
}

fn test_a_forty_field_class_keeps_its_last_fields() -> void {
    for round in 0..4 {
        assert_eq(build_wide(3000), 3000 * 2999 + 3000)
    }
}

fn test_a_thirty_one_field_class_keeps_slot_thirty() -> void {
    for round in 0..4 {
        assert_eq(build_edge(3000), 3 * (3000 * 2999 / 2))
    }
}

fn test_a_subclass_past_a_thirty_field_base() -> void {
    for round in 0..4 {
        assert_eq(build_derived(3000), 3000 * 2999 + 3 * 3000)
    }
}
