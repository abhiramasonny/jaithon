#: A chain of `if local <op> k { s = v }` links that all assign one int local
#: compiles to one `csel` per link, every value computed whether or not its
#: test holds. Each case here checks the converted body against a reference
#: written so it cannot convert (an early `return` per arm), over enough
#: inputs and calls that both the function tier and the loop tier compile it.
#: The values that may be computed speculatively are literals, int locals and
#: elements of an int list the loop head has proved in bounds; a list whose
#: storage is not the one compiled for deoptimises the whole chain to its
#: first test, before anything is written.
from std.io import Path, write_file
from std.os import argv, env, pid, run, temp_dir
from std.test import assert_eq, assert_true

fn rule_ref(n: int, cur: int) -> int {
    if n == 3 { return 1 }
    if n == 2 { return cur }
    return 0
}

fn noise(seed: int, count: int) -> list[int] {
    var out: list[int] = []
    var s = seed
    for _i in 0..count {
        s = (s * 1103515245 + 12345) % 2147483648
        out.push(s // 65536 % 9)
    }
    return out
}

#: life's rule exactly: the `elif` value is a list element, read before the
#: test that guards it.
fn apply_rule(ns: list[int], mid: list[int]) -> list[int] {
    var out: list[int] = []
    for c in 0..len(ns) {
        let n = ns[c]
        var alive = 0
        if n == 3 {
            alive = 1
        } elif n == 2 {
            alive = mid[c]
        }
        out.push(alive)
    }
    return out
}

fn test_life_rule_matches_branches() -> void {
    let ns = noise(7, 400)
    let mid = noise(11, 400).map(|v| v % 2)
    for _round in 0..120 {
        let got = apply_rule(ns, mid)
        for c in 0..400 { assert_eq(got[c], rule_ref(ns[c], mid[c])) }
    }
}

#: The same body handed a boxed list after it compiled against an unboxed
#: one: the storage guard sends the chain back to its first test.
fn test_boxed_list_takes_the_guard() -> void {
    let ns = noise(3, 64)
    let mid = noise(5, 64).map(|v| v % 2)
    for _round in 0..120 { let _w = apply_rule(ns, mid) }
    var boxed = []
    for v in mid { boxed.push(v) }
    boxed.push("not an int")
    let got = apply_rule(ns, boxed)
    for c in 0..64 { assert_eq(got[c], rule_ref(ns[c], mid[c])) }
}

fn pick_ref(a: int, b: int, c: int) -> int {
    if a < -2 { return b }
    if a >= 4 { return 7 }
    if a != 0 { return c }
    return -5
}

#: Three tests and an `else`, negative constants, and locals as values.
fn pick(a: int, b: int, c: int) -> int {
    var r = 0
    if a < -2 {
        r = b
    } elif a >= 4 {
        r = 7
    } elif a != 0 {
        r = c
    } else {
        r = -5
    }
    return r
}

fn test_else_and_locals() -> void {
    for round in 0..200 {
        for a in -6..8 {
            let b = round * 3 - a
            let c = a * a - round
            assert_eq(pick(a, b, c), pick_ref(a, b, c))
        }
    }
}

#: A one-link chain with no else keeps the old value when the test fails.
fn clamp_low(xs: list[int], floor: int) -> int {
    var total = 0
    for x in xs {
        var v = x
        if x < 2 { v = floor }
        total += v
    }
    return total
}

fn test_single_link_keeps_old_value() -> void {
    let xs = noise(17, 300)
    var want = 0
    for x in xs {
        if x < 2 { want += 100 } else { want += x }
    }
    for _round in 0..120 { assert_eq(clamp_low(xs, 100), want) }
}

#: The value read in the arm is out of range on the iterations whose test
#: fails, so it must not be read speculatively there; this stays branches.
fn guarded_read(xs: list[int], n: int) -> int {
    var total = 0
    for i in 0..n {
        var v = 0
        if i < 10 { v = xs[i] }
        total += v
    }
    return total
}

fn test_unproven_read_stays_guarded() -> void {
    var xs: list[int] = []
    for i in 0..10 { xs.push(i + 1) }
    for _round in 0..120 { assert_eq(guarded_read(xs, 40), 55) }
}

#: Two locals compared: the running maximum and minimum, and a clamp whose
#: bounds are locals too.
fn extremes(xs: list[int], lo: int, hi: int) -> list[int] {
    var best = xs[0]
    var least = xs[0]
    var clamped = 0
    for x in xs {
        if x > best { best = x }
        if x < least { least = x }
        var v = x
        if v < lo {
            v = lo
        } elif v > hi {
            v = hi
        }
        clamped += v
    }
    return [best, least, clamped]
}

fn test_two_local_tests() -> void {
    let xs = noise(23, 500)
    var best = xs[0]
    var least = xs[0]
    var clamped = 0
    for x in xs {
        best = max(best, x)
        least = min(least, x)
        clamped += min(max(x, 2), 6)
    }
    for _round in 0..120 {
        assert_eq(extremes(xs, 2, 6), [best, least, clamped])
    }
}

#: The test's own local is the one assigned: every test still reads its
#: value from before the chain.
fn step_down(x: int) -> int {
    var n = x
    if n > 5 {
        n = 5
    } elif n > 2 {
        n = 2
    }
    return n
}

fn test_assigned_local_is_tested() -> void {
    for _round in 0..200 {
        for x in -3..10 {
            let want = if x > 5 { 5 } elif x > 2 { 2 } else { x }
            assert_eq(step_down(x), want)
        }
    }
}

#: A running max keeps its branch: `if v > hi { hi = v }` with `hi` carried
#: round the loop is taken less and less often, so it predicts, and a csel
#: would put a compare and a select on the loop-carried chain (0.91x over a
#: million random ints). A clamp of a value loaded on the same trip is the
#: same shape with nothing carried, and still converts. Run in a child with
#: JAI_JIT_WHY so the decision itself can be read.
fn test_a_running_max_keeps_its_branch() -> void {
    let lines = [
        "fn scan(xs: list[int]) -> int {",
        "    var hi = -1",
        "    var lo = 4000",
        "    for i in 0..len(xs) {",
        "        let v = xs[i]",
        "        if v > hi { hi = v }",
        "        if lo > v { lo = v }",
        "    }",
        "    return hi * 10000 + lo",
        "}",
        "fn clamp(xs: list[int]) -> int {",
        "    var s = 0",
        "    for i in 0..len(xs) {",
        "        var v = xs[i]",
        "        if v < 1000 { v = 1000 } elif v > 3000 { v = 3000 }",
        "        s += v",
        "    }",
        "    return s",
        "}",
        "fn main() -> int {",
        "    var xs: list[int] = []",
        "    var q = 777",
        "    for _i in 0..2000 {",
        "        q = (q * 1103515245 + 12345) % 2147483648",
        "        xs.push(q // 65536 % 4000)",
        "    }",
        "    var total = 0",
        "    for _r in 0..300 { total += scan(xs) + clamp(xs) }",
        "    print(total)",
        "    return 0",
        "}",
    ]
    let src = "\n".join(lines) + "\n"
    let path = f"{temp_dir()}/jaithon_ifconv_{pid()}.jai"
    write_file(path, src)
    defer { Path(path).remove() }
    let ran = run(["sh", "-c", f"JAI_JIT_WHY=1 exec '{argv()[0]}' run --no-cache '{path}'"])
    assert_eq(ran.exit_code, 0)
    assert_eq(ran.stdout, "13187566800\n")
    var scanConverts = 0
    var clampConverts = 0
    for line in ran.stderr.split("\n") {
        if line.contains("if-converts") and line.contains(".scan ") { scanConverts += 1 }
        if line.contains("if-converts") and line.contains(".clamp ") { clampConverts += 1 }
    }
    assert_eq(scanConverts, 0)
    if env("JAITHON_NO_JIT") == null and env("JAITHON_JIT_IFCONV") == null {
        assert_true(clampConverts > 0, "the clamp did not convert")
    }
}
