#: Where a compiled function's locals live when they do not all fit.
#:
#: The whole-function tier used to decide all-or-nothing: if the locals plus the
#: deepest expression exceeded the ten callee-saved registers, EVERY local moved
#: to the compiled frame as a full sixteen-byte Value. One local over budget
#: spilled all of them, so every read became a load and every write a tag store
#: and a payload store. The tier now ranks the slots by the instructions a
#: register would actually save them and hands out what the operand stack did
#: not want -- an X register for an integer or an object, one of d8..d15 for a
#: float, and a frame home for the rest.
#:
#: That puts one slot's value in one of three places, and this file exists
#: because getting any of them wrong is a wrong answer rather than a crash:
#:
#:   * more locals than there are registers, so the ranking has to choose and
#:     the ones it does not choose have to keep working;
#:   * float and instance locals in the same body, so an FP home and an X home
#:     have to coexist and a float has to come back bit-exact from either;
#:   * a local whose kind two paths disagree about -- the `for b in bodies`
#:     variable below -- which keeps its frame home BECAUSE its tag is a
#:     run-time fact, and is the one slot whose tag is still written and still
#:     checked;
#:   * a `Body?` local, whose tag is read off the payload rather than fixed, so
#:     the deopt stub has to rebuild it rather than copy it;
#:   * an allocation inside the body while floats are live in d8..d15, so a
#:     collection can happen with values in callee-saved FP registers;
#:   * every guard, whose stub has to write all three homes into the record the
#:     interpreter resumes from.
#:
#: Under `--gc-stress` the fifth of those collects on every allocation, and
#: under `JAITHON_JIT_DEOPT_STRESS=1` every guard fails, so the deopt stub's
#: locals loop runs at every guard rather than never.
#:
#: It also pins a wrong answer this work turned up, which is older than the
#: register plan and reproduces with no spilling and no floats at all: a guard
#: inside a range header. `OP_BUILD_RANGE` emits nothing -- the range is folded
#: into the `OP_GET_ITER` that must follow it -- so at that offset the model
#: holds two integers while the interpreter holds the range object. A record
#: taken there handed it two integers where it expected one range, and the
#: resumed interpreter reported `'int' object is not iterable`. Both loops below
#: are ranges, and the dynamic slot is what puts a guard in their headers: the
#: iterator's own descriptor has to root every object-shaped local first, and
#: reading a dynamic one checks its tag.
class Body {
    pub var x: float
    pub var y: float
    pub var z: float
    pub var w: float
    pub var tag: int

    fn init(self, x: float, y: float, z: float, w: float, tag: int) {
        self.x = x
        self.y = y
        self.z = z
        self.w = w
        self.tag = tag
    }
}

fn advance(bodies: list[Body], dt: float) -> float {
    let n = bodies.len()
    var acc = 0.0
    var seen: Body? = null
    for i in 0..n {
        let bi = bodies[i]
        for j in i + 1..n {
            let bj = bodies[j]
            let dx = bi.x - bj.x
            let dy = bi.y - bj.y
            let dz = bi.z - bj.z
            let d2 = dx * dx + dy * dy + dz * dz
            let mag = dt / (d2 * d2 ** 0.5)
            bi.w -= dx * bj.w * mag
            bj.w += dz * bi.w * mag
            acc += dy * mag
            if bj.tag > 0 { seen = bj }
        }
    }
    # A second loop over the same list, bound from an iterator rather than an
    # index: that is what makes this slot disagree with itself and stay in the
    # frame with a tag while everything around it takes a register.
    for b in bodies {
        b.x += dt * b.w
        b.y += dt * b.z
        acc += b.x
    }
    if seen != null { acc += float(seen.tag) }
    return acc
}

fn build(k: int) -> list[Body] {
    var out: list[Body] = []
    for i in 0..k {
        out.push(
            Body(
                float(i) + 1.0,
                float(i) * 0.5 + 2.0,
                float(i) * 0.25 + 3.0,
                float(i) + 0.125,
                i % 3
            )
        )
    }
    return out
}

fn main() -> int {
    # Fresh bodies every round, so the loop above allocates while the compiled
    # body is holding floats in d8..d15.
    var total = 0.0
    for _round in 0..400 {
        let bodies = build(5)
        total += advance(bodies, 0.01)
    }
    print(f"{total:.6f}")
    return 0
}
