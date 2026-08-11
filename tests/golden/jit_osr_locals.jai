#: Which of a loop's locals live in registers, and whether every home agrees.
#:
#: The OSR tier used to decide all-or-nothing: either every local of the
#: enclosing function fitted in registers or all of them stayed in the frame as
#: full sixteen-byte Values. No loop inside a real function ever passed that
#: test, so every hot loop in the language ran with every local in memory. The
#: tier now ranks the slots a loop actually names, gives the busiest ones
#: registers -- a float slot getting one of d8..d15 rather than an X register --
#: and leaves the rest in the frame.
#:
#: That makes one slot's value live in one of three places, and everything below
#: is here because getting any of them wrong is a wrong answer rather than a
#: crash:
#:
#:   * more locals than there are registers, so the ranking has to choose and
#:     the leftovers have to keep working;
#:   * float and int slots in the same loop, so a slot in the FP bank and a slot
#:     in an X register have to coexist;
#:   * list indexing, which reads its list and index out of wherever they landed;
#:   * an allocation inside a loop that is holding floats in d8..d15, so a
#:     collection can happen with live values in callee-saved FP registers;
#:   * a deopt, which has to write every register-resident local back before the
#:     interpreter reads the frame.
#:
#: Run under `--gc-stress` the fourth of those collects on every allocation, and
#: under `JAITHON_JIT_DEOPT_STRESS=1` every guard fails, so the write-back path
#: runs at every guard rather than never.
fn mandel(w: int, h: int, limit: int) -> int {
    var inside = 0
    for py in 0..h {
        let y0 = float(py) * 2.0 / float(h) - 1.0
        for px in 0..w {
            let x0 = float(px) * 3.0 / float(w) - 2.0
            var x = 0.0
            var y = 0.0
            var i = 0
            while i < limit {
                let x2 = x * x
                let y2 = y * y
                if x2 + y2 > 4.0 { break }
                let xy = x * y
                y = 2.0 * xy + y0
                x = x2 - y2 + x0
                i += 1
            }
            if i == limit { inside += 1 }
        }
    }
    return inside
}

#: Enough live float locals to exceed the bank, read back through list indexing.
fn dots(n: int) -> float {
    var a: list[float] = []
    var b: list[float] = []
    for i in 0..n {
        a.push(float(i) * 0.5)
        b.push(float(i) * 0.25 + 1.0)
    }
    var total = 0.0
    var scale = 1.0
    var count = 0
    var k = 0
    while k < n {
        let u = a[k]
        let v = b[k]
        let p = u * v
        let q = u + v
        total += p * scale - q
        scale = 1.0 + float(k % 3) * 0.125
        count += 1
        k += 1
    }
    return total + float(count)
}

#: An int accumulator and a float in the same loop, so the write-back at a guard
#: has to put each one back in the home its own kind says it has.
fn mixed(n: int) -> int {
    var xs: list[int] = []
    for i in 0..n { xs.push(i) }
    var acc = 0
    var f = 0.5
    var j = 0
    while j < n {
        acc += xs[j]
        f = f + 0.25
        j += 1
    }
    return acc + int(f)
}

#: Allocation inside a compiled loop that is holding floats in d8..d15.
fn allocating(n: int) -> float {
    var total = 0.0
    var a = 1.5
    var b = 2.5
    var c = 3.5
    var d = 4.5
    for i in 0..n {
        var row: list[float] = []
        row.push(a * b)
        row.push(c * d)
        a = a + 0.5
        b = b + 0.25
        c = c - 0.125
        d = d * 1.0009765625
        total += row[0] - row[1]
    }
    return total
}

fn main() -> int {
    print(mandel(200, 80, 60))
    print(f"{dots(40000):.6f}")
    print(mixed(40000))
    print(f"{allocating(4000):.6f}")
    return 0
}
