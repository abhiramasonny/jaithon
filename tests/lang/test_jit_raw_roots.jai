#: Objects a compiled body holds in registers across a call that collects.
#:
#: A compiled call roots the body's live objects in its descriptor as bare
#: pointers, two to a store, and the collector finds them through the frame
#: chain (JAITHON_JIT_RAW_ROOTS). These hold an odd and an even number of
#: objects, a null one among them, across calls and a self-call that
#: force a collection, and then read every one back. A
#: root the fill missed is an object the collection freed, and its fields read
#: back as whatever the allocator put there next.
from std.test import assert_eq

class Box {
    pub var tag: int
    pub var label: str

    fn init(self, tag: int, label: str) {
        self.tag = tag
        self.label = label
    }
}

class Link {
    pub var value: int
    pub var next: Link?

    fn init(self, value: int, next: Link?) {
        self.value = value
        self.next = next
    }
}

# Kept out of line: a body that names `__prim__` itself compiles only up to
# that read, so the collect has to be reached through a call.
fn collect_now() -> void {
    __prim__.gc_collect()
}

# Enough garbage that the collector also runs on its own, inside compiled
# code, while the caller's roots are on the frame chain.
fn churn(n: int) -> int {
    var total = 0
    for i in 0..n * 40 {
        let b = Box(i, f"c{i}")
        total += b.tag % 40
    }
    if n % 7 == 3 { collect_now() }
    return total
}

# Five boxes and a null: an odd count of real roots plus a zero payload.
fn hold_five(seed: int) -> int {
    let a = Box(seed, f"a{seed}")
    let b = Box(seed + 1, f"b{seed}")
    let c = Box(seed + 2, f"c{seed}")
    let d = Box(seed + 3, f"d{seed}")
    let e = Box(seed + 4, f"e{seed}")
    var none: Box? = null
    if seed < 0 { none = a }
    let spent = churn(seed % 11)
    var sum = a.tag + b.tag + c.tag + d.tag + e.tag + spent
    if none != null { sum += 1000000 }
    if a.label != f"a{seed}" or e.label != f"e{seed}" { return -1 }
    if c.label != f"c{seed}" { return -2 }
    return sum
}

fn expected_five(seed: int) -> int {
    var spent = 0
    for i in 0..(seed % 11) * 40 { spent += i % 40 }
    return 5 * seed + 10 + spent
}

fn test_five_objects_survive_a_collecting_call() -> void {
    for seed in 0..3000 {
        assert_eq(hold_five(seed), expected_five(seed), "a held object was lost")
    }
}

# Eight: an even count, so every root goes down in a pair.
fn hold_eight(seed: int) -> int {
    let r0 = Box(seed, "r0")
    let r1 = Box(seed, "r1")
    let r2 = Box(seed, "r2")
    let r3 = Box(seed, "r3")
    let r4 = Box(seed, "r4")
    let r5 = Box(seed, "r5")
    let r6 = Box(seed, "r6")
    let r7 = Box(seed, "r7")
    if seed % 97 == 0 { collect_now() }
    let joined = r0.label + r1.label + r2.label + r3.label + r4.label +
                 r5.label + r6.label + r7.label
    return r0.tag + r7.tag + joined.len()
}

fn test_eight_objects_survive_a_native_collect() -> void {
    for seed in 0..3000 {
        assert_eq(hold_eight(seed), 2 * seed + 16, "a paired root was lost")
    }
}

# A fresh allocation held on the operand stack across an allocating
# self-call, the shape binary_trees builds its tree with.
fn build(depth: int, value: int) -> Link? {
    if depth == 0 {
        if value % 64 == 1 { collect_now() }
        return null
    }
    return Link(value, build(depth - 1, value * 2))
}

fn chain_sum(node: Link?) -> int {
    var total = 0
    var at = node
    while at != null {
        total += at.value
        at = at.next
    }
    return total
}

fn test_a_list_built_across_self_calls_survives() -> void {
    for start in 1..400 {
        let head = build(8, start)
        var want = 0
        var v = start
        for _ in 0..8 {
            want += v
            v *= 2
        }
        assert_eq(chain_sum(head), want, "a node built across a self-call was lost")
    }
}
