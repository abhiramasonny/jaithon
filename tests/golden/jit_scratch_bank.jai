#: Which register bank a compiled body's operand stack lives in.
#:
#: The operand stack used to sit above the locals in x19..x28, so the two
#: competed for the same ten registers and a deep expression starved the
#: locals: a four-row stencil went seven entries deep, left nothing over, and
#: reloaded every row pointer and its index from the frame on every iteration.
#: A body that can prove nothing will clobber a caller-saved register while it
#: runs now puts its operand stack in x0..x8 instead, and the whole
#: callee-saved bank goes to the locals.
#:
#: "Nothing will clobber" is the entire correctness argument, so this file is
#: three shapes that each break it a different way -- getting any of them wrong
#: reads a value out of a register a helper overwrote, which is a wrong answer
#: and not a crash:
#:
#:   * no call at all, the case the bank is for, with an expression deep enough
#:     that the old plan had no registers left for the locals;
#:   * a real call in the middle of the expression, so entries pushed before it
#:     have to survive it -- this body must stay on the callee-saved bank;
#:   * a list push inside the expression, whose grow stub calls a helper and
#:     then branches BACK into the body. It is not an OP_CALL and the walk sees
#:     no call instruction, so it is the one that a bank decision keyed on
#:     "does this body contain a call" gets wrong.
#:
#: Under `JAITHON_JIT_DEOPT_STRESS=1` every guard fails, so the deopt stubs --
#: which read the operand stack out of whichever bank it is in -- run at every
#: guard rather than never, and under `--gc-stress` the third one collects on
#: every push while the expression around it is live.

#: Call-free, and seven operand-stack entries deep at `out[j] = 0.25 * (...)`.
fn stencil(n: int, steps: int) -> float {
    var a: list[float] = []
    var b: list[float] = []
    for i in 0..n {
        a.push(float(i % 7) * 0.5)
        b.push(0.0)
    }
    var s = 0
    while s < steps {
        for j in 1..n - 1 {
            b[j] = 0.25 * (a[j] + a[j - 1] + a[j + 1] + b[j])
        }
        for j in 1..n - 1 {
            a[j] = 0.25 * (b[j] + b[j - 1] + b[j + 1] + a[j])
        }
        s += 2
    }
    var total = 0.0
    for j in 0..n { total += a[j] }
    return total
}

#: Big enough not to be inlined, so the call is a real one.
fn weigh(v: float) -> float {
    var acc = v
    var k = 0
    while k < 3 {
        acc = acc * 0.5 + 0.125
        k += 1
    }
    return acc
}

#: The same expression with that call inside it. Three entries are already on
#: the operand stack when the call happens; every one of them has to still be
#: there afterwards.
fn stencil_call(n: int, steps: int) -> float {
    var a: list[float] = []
    var b: list[float] = []
    for i in 0..n {
        a.push(float(i % 5) + 0.125)
        b.push(0.0)
    }
    var s = 0
    while s < steps {
        for j in 1..n - 1 {
            b[j] = 0.25 * (a[j] + weigh(a[j - 1]) + a[j + 1] + b[j])
        }
        s += 1
    }
    var total = 0.0
    for j in 0..n { total += b[j] }
    return total
}

#: A push in the middle of the expression: the grow stub calls out and branches
#: back in, with the partial sum still live.
fn growing(n: int, rounds: int) -> float {
    var a: list[float] = []
    for i in 0..n { a.push(float(i % 5) + 0.25) }
    var total = 0.0
    for _r in 0..rounds {
        var out: list[float] = []
        for j in 2..n - 2 {
            out.push(0.25 * (a[j] + a[j - 1] + a[j + 1] + a[j - 2]))
            total += out[out.len() - 1] * 2.0
        }
    }
    return total
}

#: Short rows and many sweeps rather than the other way round. The tier is
#: sampler-driven, so a loop has to run for milliseconds to be offered at all --
#: at a tenth of these counts `growing` compiled in none of three runs and the
#: file had no teeth. But `--gc-stress` marks the whole live set on every
#: allocation, so widening the rows instead of repeating them cost half a minute
#: of every `make test` for the same number of iterations.
fn main() -> int {
    print(f"{stencil(3000, 1000):.6f}")
    print(f"{stencil_call(3000, 200):.6f}")
    print(f"{growing(3000, 40):.6f}")
    return 0
}
