#: The tier's operand model, checked against the bytecode's own answer.
#:
#: The model is what the compiler believes is on the operand stack at each
#: offset. Roughly 140 hand-written arms maintain it, and nothing used to ask it
#: to agree with anything: a wrong depth still emits a SELF-CONSISTENT body,
#: because every register is derived from an entry's index and the indices all
#: shift together. What it corrupts is the deopt record -- the one part of the
#: model another component reads -- which then hands the interpreter operand
#: entries it has not got, and the instruction after the resume reads the wrong
#: one. Wrong answer, no crash, no diagnostic.
#:
#: Two ways in were found, both of them reachable only once an unarmed opcode
#: stopped declining the function that mentions it:
#:
#: 1. `inlineMethod` walks the callee's bytecode to decide whether it can be
#:    inlined, pushing and popping the model as it reads -- and every one of its
#:    two dozen refusals returns from the middle of that walk. Its caller does
#:    NOT decline when it declines; the OP_INVOKE arm falls through to the
#:    descriptor path with the model left as deep as the walk got. `crossing`
#:    below is the shape that found it in `lib/std/gui/path.jai`: two field
#:    reads and a subtraction are inlinable, the `let` that binds them is not,
#:    so the walk stops three instructions in having pushed two entries.
#:
#: 2. The walk past an unarmed opcode resumed at the next offset anything could
#:    branch to -- which may be reached only by a branch that was itself inside
#:    the skipped run, so nothing restated the model there and the stale one was
#:    carried on. `is_arith` below is that shape.
#:
#: A GOLDEN, not a `tests/lang` unit test: under `jaithon test` these bodies
#: never get hot enough to compile and the assertions pass with both fixes
#: reverted. `run_tests.sh` runs a golden through `jaithon run` at four
#: optimisation levels, plus `--gc-stress` and `JAITHON_JIT_DEOPT_STRESS=1`.
#: Every number below was taken from `JAITHON_NO_JIT=1`.
import std.io

class Edge {
    pub var y0: float
    pub var y1: float
    pub var x0: float
    pub var dir: int

    pub fn init(self, y0: float, y1: float, x0: float, dir: int) {
        self.y0 = y0
        self.y1 = y1
        self.x0 = x0
        self.dir = dir
    }

    #: Deliberately just past what `inlineMethod` accepts. The two field reads
    #: and the subtraction walk fine; the `let` that binds them does not, and
    #: the walk returns having already pushed. Shrinking this body to one
    #: expression would let it inline and the test would pin nothing.
    pub fn crossing(self, y: float) -> float {
        let span = self.y1 - self.y0
        if span == 0.0 { return self.x0 }
        return self.x0 + (y - self.y0) * span
    }
}

#: `hits.push((...))` is the unarmed opcode -- OP_BUILD_TUPLE -- and it sits
#: AFTER the invoke, so every deopt record the tier writes from there on is
#: taken against whatever the invoke left behind. With the model two entries
#: too deep the interpreter resumed holding the receiver and the argument a
#: second time, ran OP_BUILD_TUPLE over the wrong pair, and then invoked `push`
#: on a float: `AttributeError: 'float' object has no method 'push'`.
fn crossings(edges: list[Edge], y: float) -> any {
    var hits = []
    for edge in edges {
        if y >= edge.y0 and y < edge.y1 {
            hits.push((edge.crossing(y), edge.dir))
        }
    }
    return hits
}

#: A `match`, which is unarmed, followed by ordinary work. The only branch to
#: the code after the match is inside the run the walk skips, so there is
#: nothing to restate the model from -- the walk used to carry the match's own
#: two entries into everything below.
fn is_arith(op: int) -> int {
    match op {
        0 | 1 | 2 => { return 10 }
        3 | 4 => { return 20 }
        _ => { return 30 }
    }
}

#: Same again with the unarmed opcode before a LOOP rather than before straight
#: line code: the loop head is a branch target the walk has not reached, so it
#: is reachable only because the bytecode says the stack is empty there.
fn after_match(op: int, n: int) -> int {
    var seen = 0
    match op {
        0 => { seen = 1 }
        _ => { seen = 2 }
    }
    var total = seen
    for i in 0..n { total += i }
    return total
}

fn main() -> void {
    let edges = [Edge(0.0, 4.0, 1.0, 1), Edge(1.0, 3.0, 2.0, -1), Edge(2.0, 6.0, 4.0, 1)]

    #: Past JAI_JIT_THRESHOLD (64) so the FUNCTION tier compiles each of these
    #: -- deterministic, unlike the sampler-driven OSR tier.
    var checksum = 0.0
    var dirs = 0
    var count = 0
    for _round in 0..80 {
        let hits = crossings(edges, 2.5)
        count = count + hits.len()
        for hit in hits {
            let (x, d) = hit
            checksum = checksum + x
            dirs = dirs + d
        }
    }
    print(f"hits {count}")
    print(f"checksum {checksum}")
    print(f"dirs {dirs}")

    var arith = 0
    for round in 0..80 { arith = arith + is_arith(round % 6) }
    print(f"arith {arith}")

    var tail = 0
    for round in 0..80 { tail = tail + after_match(round % 2, 5) }
    print(f"tail {tail}")
}
