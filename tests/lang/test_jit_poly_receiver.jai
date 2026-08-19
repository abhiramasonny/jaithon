#: `for op in ops` over a list holding several implementations of one trait.
#:
#: The compiled tier resolves a method while it emits, which needs the
#: receiver's class. A loop variable bound from a list of unrelated classes has
#: no one class, and before this the tier gave the whole loop up for it: the
#: loop head reported "loop variable took two kinds" and the body ran
#: interpreted, at 2.4x the cost of the same loop over a single class.
#:
#: Two classes are not two KINDS -- an instance is an untagged pointer either
#: way -- so the slot widens to "an instance, class unknown" and the call site
#: dispatches by name against the shared megamorphic table. What has to hold:
#:
#: - Every class's own method runs, with its own field layout. `A` and `B`
#:   declare different fields at different slots, so a call that reached the
#:   wrong body would read the wrong words.
#: - The answers match the interpreter's, which is what `JAITHON_NO_JIT=1` on
#:   this file proves and the plain run proves the compiled form agrees with.
#: - A method that raises unwinds out of the compiled loop.
#: - A list that was homogeneous when the loop compiled and gains a second
#:   class afterwards deoptimises rather than calling the pinned method.
#:
#: The counts are large enough that the loops compile; `drive` is the body the
#: tier is expected to reach.
#:
#: `labels` is not marked: it returns a str and its body is a builtin invoke the
#: tier declines, so the trait-default case below is a correctness check rather
#: than a compiled one.
# jit-compiles: drive
from std.test import assert_eq, assert_true

trait Op {
    fn apply(self, x: int) -> int
}

#: Two int fields and a str, so `k` is not at slot 0 of every class here.
class Add: Op {
    pub var tag: str
    pub var k: int

    fn init(self, k: int) {
        self.tag = "add"
        self.k = k
    }

    pub fn apply(self, x: int) -> int { return x + self.k }
}

#: A float first, so a body reading `q` at Add's slot index would read a
#: double's bits as an integer.
class Mul: Op {
    pub var weight: float
    pub var q: int

    fn init(self, q: int) {
        self.weight = 0.5
        self.q = q
    }

    pub fn apply(self, x: int) -> int { return x * self.q }
}

#: No fields at all.
class Negate: Op {
    fn init(self) { }

    pub fn apply(self, x: int) -> int { return -x }
}

#: Allocates per call, so a collection can land inside the compiled loop.
class Trail: Op {
    pub var seen: list[str]

    fn init(self) { self.seen = [] }

    pub fn apply(self, x: int) -> int {
        self.seen.push(f"s{x % 7}")
        return x + self.seen.len() % 3
    }
}

class Fails: Op {
    pub var calls: int

    fn init(self) { self.calls = 0 }

    pub fn apply(self, x: int) -> int {
        self.calls += 1
        if self.calls % 5 == 0 { throw ValueError("every fifth call") }
        return x + 1
    }
}

#: The loop the tier is expected to compile with an unpinned receiver.
fn drive(ops: list[Op], reps: int) -> int {
    var acc = 1
    for _r in 0..reps {
        for op in ops { acc = (op.apply(acc) % 100003 + 100003) % 100003 }
    }
    return acc
}

fn four_kinds() -> list[Op] {
    var ops: list[Op] = []
    for i in 0..64 {
        if i % 4 == 0 {
            ops.push(Add(i + 1))
        } elif i % 4 == 1 {
            ops.push(Mul(i % 5 + 2))
        } elif i % 4 == 2 {
            ops.push(Negate())
        } else {
            ops.push(Trail())
        }
    }
    return ops
}

fn test_mixed_list_runs_every_class() {
    assert_eq(drive(four_kinds(), 400), 12715)
}

#: Two lists of the same four classes, driven through the same compiled body:
#: the second enters an already-compiled form whose entry guard must accept a
#: receiver of any of the four classes rather than pinning whichever came
#: first. Two fresh lists rather than one twice, because `Trail` keeps state
#: between calls and the same list driven twice is legitimately two answers.
fn test_mixed_list_reentered() {
    assert_eq(drive(four_kinds(), 200), drive(four_kinds(), 200))
}

#: Homogeneous when it compiles, mixed afterwards. The pinned form has to give
#: way rather than call `Add.apply` on a `Negate`.
fn test_list_gains_a_class_after_compiling() {
    var ops: list[Op] = []
    for i in 0..32 { ops.push(Add(i + 1)) }
    assert_eq(drive(ops, 400), 11195)
    ops.push(Negate())
    assert_eq(drive(ops, 400), 1)
}

fn test_raise_unwinds_out_of_a_compiled_loop() {
    var ops: list[Op] = []
    for i in 0..16 {
        if i % 2 == 0 {
            ops.push(Fails())
        } else {
            ops.push(Negate())
        }
    }
    var caught = 0
    for _r in 0..300 {
        try {
            let _ignored = drive(ops, 1)
        } catch e: ValueError {
            caught += 1
        }
    }
    assert_true(caught > 0)
}

#: A trait's own default body, which lives on the trait rather than in any
#: class's method table: the by-name path must find it where the megamorphic
#: table cannot answer.
trait Described {
    fn label(self) -> str { return "described" }
}

class Plain: Described {
    fn init(self) { }
}

class Named: Described {
    fn init(self) { }

    pub fn label(self) -> str { return "named" }
}

fn labels(items: list[Described], reps: int) -> int {
    var n = 0
    for _r in 0..reps {
        for item in items { n += item.label().len() }
    }
    return n
}

fn test_trait_default_reached_by_name() {
    var items: list[Described] = []
    for i in 0..32 {
        if i % 2 == 0 {
            items.push(Plain())
        } else {
            items.push(Named())
        }
    }
    assert_eq(labels(items, 400), 89600)
}
