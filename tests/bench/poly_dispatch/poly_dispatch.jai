#: Megamorphic dispatch — one call site, eight receiver classes.
#:
#: Holdout. `object_dispatch` calls two methods on one class, so its inline
#: cache is monomorphic and stays hot. Here the call site sees eight unrelated
#: implementations in an order the data picks, which is the case every inline
#: cache and every type-feedback guard is *worst* at. Nothing about the ops
#: matters; the loop body is one virtual call and one add.
from std.os import env_or

let LEVEL = env_or("BENCH_LEVEL", "hard")
let SCALE = if LEVEL == "easy" { 16 } elif LEVEL == "medium" { 4 } else { 1 }
let OPS = 512
let REPS = 8000 // SCALE
const M = 1000003

trait Op {
    fn apply(self, x: int) -> int
}

class AddK: Op {
    pub var k: int

    fn init(self, k: int) { self.k = k }

    pub fn apply(self, x: int) -> int { return (x + self.k) % M }
}

class MulK: Op {
    pub var k: int

    fn init(self, k: int) { self.k = k }

    pub fn apply(self, x: int) -> int { return x * self.k % M }
}

class SquareK: Op {
    pub var k: int

    fn init(self, k: int) { self.k = k }

    pub fn apply(self, x: int) -> int { return (x * x + self.k) % M }
}

class DoubleK: Op {
    pub var k: int

    fn init(self, k: int) { self.k = k }

    pub fn apply(self, x: int) -> int { return (x * 2 + self.k) % M }
}

class DivK: Op {
    pub var k: int

    fn init(self, k: int) { self.k = k }

    pub fn apply(self, x: int) -> int { return (x + x // self.k) % M }
}

class ModK: Op {
    pub var k: int

    fn init(self, k: int) { self.k = k }

    pub fn apply(self, x: int) -> int { return (x + x % self.k) % M }
}

class FlipK: Op {
    pub var k: int

    fn init(self, k: int) { self.k = k }

    pub fn apply(self, x: int) -> int { return (M - 1 - x + self.k) % M }
}

class MixK: Op {
    pub var k: int

    fn init(self, k: int) { self.k = k }

    pub fn apply(self, x: int) -> int { return (x + x // 3 + self.k) % M }
}

fn build(n: int) -> list[Op] {
    var ops: list[Op] = []
    var s = 7
    for _i in 0..n {
        s = (s * 1103515245 + 12345) % 2147483648
        let which = s // 65536 % 8
        let k = s % 97 + 2
        if which == 0 {
            ops.push(AddK(k))
        } elif which == 1 {
            ops.push(MulK(k))
        } elif which == 2 {
            ops.push(SquareK(k))
        } elif which == 3 {
            ops.push(DoubleK(k))
        } elif which == 4 {
            ops.push(DivK(k))
        } elif which == 5 {
            ops.push(ModK(k))
        } elif which == 6 {
            ops.push(FlipK(k))
        } else {
            ops.push(MixK(k))
        }
    }
    return ops
}

fn main() -> int {
    let ops = build(OPS)
    var acc = 1
    var check = 0
    var reps = max(REPS, 1)
    # A full pass over the 512 ops turns out to have a fixed point, so without
    # this every level printed the same answer and the output stopped being
    # evidence that the work happened. The perturbation costs nothing in the
    # inner loop, which is the thing being measured.
    for r in 0..reps {
        acc = (acc + r) % M
        for op in ops { acc = op.apply(acc) }
        check = (check + acc) % M
    }
    print(ops.len())
    print(acc)
    print(check)
    return 0
}
