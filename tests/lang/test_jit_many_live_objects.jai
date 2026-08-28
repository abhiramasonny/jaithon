#: A loop holding more live objects than the callee-saved register file has
#: registers.
#:
#: Two caps refused this, and both were the same mistake in different places: a
#: number chosen for one purpose reused for another that has nothing to do with
#: it.
#:
#: `JIT_MAX_SAVED` is 10 because x19..x28 is ten registers -- a hardware fact.
#: It was ALSO the length of the descriptor's `roots[]` array, which is an array
#: on the frame and has no such bound. What that array holds is mostly LOCALS,
#: and a local that earned no register lives in memory, so the count was never
#: tied to the ten at all. Now `JIT_MAX_ROOTS`, 24.
#:
#: `JAI_OSR_SHAPES` was 8 with a comment saying "a loop with more than eight
#: distinct instance locals does not exist in this tree". It does: jaicv's
#: `connected_components` has two such heads, each refused eighty times in one
#: `imgproc` run. Now 16, at 160 more bytes per ObjFunction.
#:
#: Twelve live instances across calls, two million rounds, measured in one
#: binary through `JAITHON_JIT_ROOT_LIMIT` and `JAITHON_JIT_SHAPE_LIMIT`, which
#: move the caps without changing the arrays:
#:
#:   roots 10, shapes  8   60,000,667   0.29 0.29 0.41 0.42
#:   roots 24, shapes  8          995   0.12 0.13 0.13 0.13   <- 2.2x
#:   roots 10, shapes 16   60,000,667   0.33 0.28 0.27 0.29   <- nothing
#:   roots 24, shapes 16      969,935   0.09 0.09 0.08 0.09   <- 3.2x
#:
#: So roots is the load-bearing one and shapes adds 1.4x on top of it, but only
#: once roots is raised -- neither is worth having alone.
#:
#: NO WORKLOAD IN THIS TREE MOVES. `connected_components`, the function whose
#: refusals found the shape cap, is 0.12s either way because it is GPU-bound;
#: the self-hosted compiler is 285 bodies and 0.70s either way. What the caps
#: buy is this shape -- a loop calling functions with a dozen live objects --
#: which ordinary application code has and this tree's benchmarks do not.
# jit-compiles: sum_cells
from std.test import assert_eq

let CALLS = 100

class Cell {
    pub var v: int

    fn init(self, v: int) -> void { self.v = v }
}

fn touch(a: Cell, b: Cell) -> int { return a.v + b.v }

fn sum_cells(rounds: int) -> int {
    let c1 = Cell(1)
    let c2 = Cell(2)
    let c3 = Cell(3)
    let c4 = Cell(4)
    let c5 = Cell(5)
    let c6 = Cell(6)
    let c7 = Cell(7)
    let c8 = Cell(8)
    let c9 = Cell(9)
    let ca = Cell(10)
    let cb = Cell(11)
    let cc = Cell(12)
    var t = 0
    var i = 0
    while i < rounds {
        t += touch(c1, c2) + touch(c3, c4) + touch(c5, c6)
        t += touch(c7, c8) + touch(c9, ca) + touch(cb, cc)
        i += 1
    }
    return t
}

pub fn test_twelve_live_objects_across_calls() -> void {
    var total = 0
    var call = 0
    while call < CALLS {
        #: 1+2 + 3+4 + ... + 11+12 = 78 per round.
        total += sum_cells(20)
        call += 1
    }
    assert_eq(total, 78 * 20 * CALLS)
}
