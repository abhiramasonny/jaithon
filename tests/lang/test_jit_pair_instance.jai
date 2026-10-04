#: `for (name, node) in d.items()` where the values are instances.
#:
#: The pair arm bound every object component as a bare SLOT_OBJ. A loop
#: variable the frame already held as an instance -- which is what the OSR
#: head finds, and what a slot reused from an earlier instance-typed local
#: holds -- then clashed with it and the whole loop ran interpreted. It now
#: binds the sampled class (JAITHON_JIT_PAIR_INST), proving object type and
#: shape on every step, so a dict that later holds another class, a string, or
#: null in the value position deoptimises rather than being read as the first.
from std.test import assert_eq

let WARM = 300

class Leaf {
    pub var v: int

    fn init(self, v: int) { self.v = v }
}

class Other {
    pub var w: int
    pub var v: int

    fn init(self, w: int) {
        self.w = w
        self.v = w * 100
    }
}

fn sum_leaves(d: dict[str, Leaf]) -> int {
    var total = 0
    for (name, leaf) in d.items() { total += name.len() + leaf.v }
    return total
}

fn sum_any(d: dict[str, any]) -> int {
    var total = 0
    for (name, x) in d.items() { total += x.v + name.len() - name.len() }
    return total
}

fn recursive(d: dict[str, Leaf], depth: int) -> int {
    if depth == 0 { return 0 }
    var total = 0
    for (_name, leaf) in d.items() { total += leaf.v + recursive(d, depth - 1) }
    return total
}

pub fn test_instances_of_one_class() -> void {
    var d: dict[str, Leaf] = {}
    for i in 0..10 { d[f"k{i}"] = Leaf(i) }
    var got = 0
    for _w in 0..WARM { got = sum_leaves(d) }
    assert_eq(got, 20 + 45, "names and values")
}

pub fn test_another_class_after_warming() -> void {
    var d: dict[str, any] = {}
    for i in 0..10 { d[f"k{i}"] = Leaf(i) }
    var got = 0
    for _w in 0..WARM { got = sum_any(d) }
    assert_eq(got, 45, "all leaves")
    # Same field name, another class: the shape guard must send this one
    # back to the interpreter, which reads Other's own `v` slot.
    d["k3"] = Other(7)
    assert_eq(sum_any(d), 45 - 3 + 700, "a stranger")
}

pub fn test_recursion_inside_the_loop() -> void {
    var d: dict[str, Leaf] = {}
    for i in 1..4 { d[f"n{i}"] = Leaf(i) }
    var got = 0
    for _w in 0..50 { got = recursive(d, 3) }
    # level sums: 6 per level, each value recursing into 3 more
    assert_eq(got, 6 + 3 * (6 + 3 * 6), "nested")
}
