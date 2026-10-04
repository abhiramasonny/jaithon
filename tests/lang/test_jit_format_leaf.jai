#: f-strings formatted by compiled code through the leaf (JAITHON_JIT_FORMAT_LEAF).
#:
#: The leaf formats only parts that are str, int, float, bool or null, and
#: declines to the rooted path for anything else -- an instance, a list --
#: and whenever a collection is wanted. Each case here is
#: compared with the same f-string formatted in a body that has not compiled.
# jit-compiles: fmt_two, fmt_any
from std.test import assert_eq

let CALLS = 400

class Point {
    pub var x: int
    pub var y: int

    fn init(self, x: int, y: int) {
        self.x = x
        self.y = y
    }

    pub fn to_str(self) -> str { return f"P({self.x}, {self.y})" }
}

fn fmt_two(a: int, b: str) -> str { return f"{a}-{b}!" }

fn fmt_any(v: any) -> str { return f"<{v}>" }

pub fn test_primitive_parts() -> void {
    var total = 0
    for i in 0..CALLS { total += fmt_two(i, "w").len() }
    assert_eq(fmt_two(-42, "é"), "-42-é!", "negative and multi-byte")
    assert_eq(fmt_two(0, ""), "0-!", "an empty string part")
    for _i in 0..CALLS { fmt_any(1) }
    assert_eq(fmt_any(2.5), "<2.5>", "a float")
    assert_eq(fmt_any(true), "<true>", "a bool")
    assert_eq(fmt_any(null), "<null>", "null")
    assert_eq(fmt_any("s"), "<s>", "a string")
}

pub fn test_parts_the_leaf_declines() -> void {
    for _i in 0..CALLS { fmt_any(3) }
    let pt = Point(1, 2)
    assert_eq(fmt_any(pt), f"<{pt}>", "an instance")
    assert_eq(fmt_any([1, 2]), f"<{[1, 2]}>", "a list")
}

pub fn test_many_results_survive_collections() -> void {
    var keep: list[str] = []
    for i in 0..3000 { keep.push(fmt_two(i, "k")) }
    assert_eq(keep[2999], "2999-k!", "the last")
    assert_eq(keep[17], "17-k!", "an early one")
}
