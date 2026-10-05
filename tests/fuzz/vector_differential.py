#!/usr/bin/env python3
"""Differential fuzz for the loop vectoriser (src/vm/jit/jit_vector.c).

progen.py never writes `for j in a..b { xs[j + k] = <float expression> }`, so
the general fuzzer cannot reach the vectoriser at all. This generates exactly
that shape -- random float expressions over up to three lists at offsets
-2..2, float literals and a float local -- and calls each kernel on lists of
every length 0..40 full of NaN, infinities, -0.0 and subnormals, over windows
that sometimes wrap a negative index and sometimes run off the end (raising
IndexError part way through), sometimes with the counter and a based
subscript's base near opposite ends of int (so a step of the subscript raises
OverflowError), with the stored list sometimes passed in as one
of the lists it reads, and every few rounds with boxed lists instead of
unboxed ones. Every element is printed with f"{x}", which round-trips a
double's bits (all but a NaN's payload).

The interpreter is the oracle. `vector-off` (JAITHON_JIT_VECTOR=0) is the one
that says whether a disagreement is the vectoriser or something the scalar
tier already disagreed about.

    python3 tests/fuzz/vector_differential.py --count 40
"""
import argparse
import os
import random
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JAITHON = os.path.join(ROOT, "jaithon")

CONFIGS = [
    ("no-jit", {"JAITHON_NO_JIT": "1"}),
    ("baseline", {}),
    ("threshold-1", {"JAITHON_JIT_THRESHOLD": "1"}),
    ("tick-50", {"JAITHON_JIT_TICK_US": "50"}),
    ("deopt-stress", {"JAITHON_JIT_DEOPT_STRESS": "1"}),
    ("vector-off", {"JAITHON_JIT_VECTOR": "0"}),
]

LITERALS = ["0.25", "-0.0", "0.0", "1.5", "-3.0", "1e308", "1e-310", "2.0"]
LISTS = ["a", "b", "c"]
INT_MAX = 9223372036854775807


def subscript(name, k, based=False, rng=None):
    idx = "j" if k == 0 else (f"j + {k}" if k > 0 else f"j - {-k}")
    if based:
        # Every order of the three terms: the vectoriser folds them into one
        # offset, but the scalar loop adds them left to right, each step
        # overflow-checked, so `k + row + j` can raise where `row + j + k`
        # does not. The constant stays signed by its operator.
        kt = "" if k == 0 else (f" + {k}" if k > 0 else f" - {-k}")
        forms = [f"row + {idx}", f"{idx} + row", f"row{kt} + j",
                 f"j + row{kt}"]
        if k > 0:
            forms += [f"{k} + j + row", f"{k} + row + j"]
        idx = (rng.choice(forms) if rng is not None
               else forms[0] if k % 2 == 0 else forms[1])
    return f"{name}[{idx}]"


def expr(rng, depth, dst, s, sb, pb):
    """Reads of the stored list stay rare: at any offset but its own they are
    a recurrence the vectoriser refuses, so they would cost it its coverage.
    A fraction `pb` of the subscripts take the int parameter `row` as a
    base."""
    if depth == 0 or rng.random() < 0.3:
        r = rng.random()
        if r < 0.7:
            name = rng.choice([x for x in LISTS if x != dst])
            based = rng.random() < pb
            if rng.random() < 0.1:
                name = dst
                if rng.random() < 0.5:
                    k, based = s, sb
                else:
                    k = rng.randint(-2, 2)
            else:
                k = rng.randint(-2, 2)
            return subscript(name, k, based, rng)
        if r < 0.85:
            return "f"
        return rng.choice(LITERALS)
    op = rng.choice(["+", "-", "*"])
    return (f"({expr(rng, depth - 1, dst, s, sb, pb)} {op} "
            f"{expr(rng, depth - 1, dst, s, sb, pb)})")


def kernel(rng, i):
    dst = rng.choice(["a", "b"])
    s = rng.randint(-2, 2)
    # Two kernels in five base every subscript: only those reach the
    # vector run when the counter is near the end of int.
    pb = 1.0 if rng.random() < 0.4 else 0.33
    sb = rng.random() < pb
    typed = rng.random() < 0.5
    params = ("a: list[float], b: list[float], c: list[float], f: float"
              if typed else "a, b, c, f: float")
    text = (f"fn k{i}({params}, row: int, lo: int, hi: int) -> void {{\n"
            f"    for j in lo..hi {{ {subscript(dst, s, sb, rng)} = "
            f"{expr(rng, 3, dst, s, sb, pb)} }}\n"
            f"}}\n")
    return text, pb == 1.0


def program(rng, kernels, warm):
    src = ["from std.math import NAN, INF", "",
           "fn specials(n: int, seed: int) -> list[float] {",
           "    let pool = [1.5, -0.0, 0.0, NAN, INF, -INF, 1e-310, -2.5e-308,",
           "                1e308, -1e308, 0.1, 3.0, -7.25, 2.0, 1e-300, 4.9e-324]",
           "    var out: list[float] = []",
           "    for i in 0..n { out.push(pool[(i * 7 + seed) % pool.len()]) }",
           "    return out",
           "}", "",
           "fn boxed(n: int, seed: int) -> list[any] {",
           "    var out: list[any] = []",
           "    for i in 0..n {",
           "        if (i + seed) % 3 == 0 { out.push(i - 4) } else { out.push(0.5 * i - 1.0) }",
           "    }",
           "    return out",
           "}", "",
           "fn show(xs) -> str {",
           "    var t = \"\"",
           "    for x in xs { t = t + f\"{x},\" }",
           "    return t",
           "}", ""]
    all_based = []
    for i in range(kernels):
        text, based = kernel(rng, i)
        src.append(text)
        all_based.append(based)
    src.append("fn main() -> int {")
    src.append(f"    for round in 0..{warm} {{")
    src.append("        let loud = round == 0 or round % 37 == 5 or "
               f"round == {warm - 1}")
    for i in range(kernels):
        n = rng.randint(0, 40)
        lo = rng.choice([0, 1, 2, 2, 3])
        hi = max(lo, n + rng.choice([-3, -2, -2, -1, 0, 1]))
        alias = rng.choice(["none", "none", "ab", "ac", "bc"])
        f = rng.choice(["0.5", "-0.0", "NAN", "3.25", "INF"])
        row = rng.choice([0, 0, 1, 2, 3, -1, -2, -3, 5, 40])
        # Now and then the same indices with the counter near one end of int
        # and the base near the other: `j + row` is unchanged, but a step like
        # `j + 2` or `row + 1` on the way to it overflows.
        # Only a kernel whose every subscript is based can take the vector
        # run there; a plain `a[j]` is simply out of range.
        # Three placements: the base just under the top (`row + 2 + j`
        # overflows), the counter's end at the top (`j + 2 + row`), and its
        # start at the bottom (`j - 2 + row`). The shift moves the counter and
        # the base by opposite amounts, so each `j + row` stays the same index.
        if rng.random() < (0.7 if all_based[i] else 0.05):
            # Mostly a window every offset -2..2 stays inside, so the only
            # thing that can stop the vector run is the overflow itself.
            if n >= 6 and rng.random() < 0.8:
                row = rng.choice([0, 1, 2])
                lo = 2 - row
                hi = rng.randint(lo + 2, n - 2 - row)
            mode = rng.choice(["base-top", "end-top", "start-bottom"])
            if mode == "base-top":
                shift = row - (INT_MAX - rng.choice([0, 0, 1, 2, 20]))
            elif mode == "end-top":
                shift = INT_MAX - rng.choice([0, 0, 1, 20]) - hi
            else:
                e = max(0, lo + row) + rng.choice([0, 0, 1])
                shift = -INT_MAX + e - lo
            if (-INT_MAX <= lo + shift and hi + shift <= INT_MAX and
                    -INT_MAX <= row - shift <= INT_MAX):
                lo, hi, row = lo + shift, hi + shift, row - shift
        src.append(f"        var a{i} = specials({n}, round + {i})")
        src.append(f"        var b{i} = specials({n}, round + {i + 5})")
        src.append(f"        var c{i} = specials({n}, round + {i + 9})")
        if alias == "ab":
            src.append(f"        b{i} = a{i}")
        elif alias == "ac":
            src.append(f"        c{i} = a{i}")
        elif alias == "bc":
            src.append(f"        c{i} = b{i}")
        src.append(f"        var raised{i} = false")
        typed = f"fn k{i}(a: list[float]" in "".join(src)
        src.append("        try {")
        src.append(f"            k{i}(a{i}, b{i}, c{i}, {f}, {row}, {lo}, {hi})")
        src.append("        } catch _failure {")
        src.append(f"            raised{i} = true")
        src.append("        }")
        src.append(f"        if loud {{ print(f\"k{i} {{round}} {{raised{i}}} \" + "
                   f"show(a{i}) + \"|\" + show(b{i})) }}")
        if not typed:
            # Boxed storage now and then, on the same compiled kernel.
            src.append("        if round % 5 == 4 {")
            src.append(f"            var x{i} = boxed({n}, round)")
            src.append(f"            var y{i} = boxed({n}, round + 1)")
            src.append(f"            var r2{i} = false")
            src.append("            try {")
            src.append(f"                k{i}(x{i}, y{i}, x{i}, {f}, {row}, {lo}, {hi})")
            src.append("            } catch _failure {")
            src.append(f"                r2{i} = true")
            src.append("            }")
            src.append(f"            if loud {{ print(f\"k{i}b {{r2{i}}} \" + "
                       f"show(x{i}) + \"|\" + show(y{i})) }}")
            src.append("        }")
    src.append("    }")
    src.append("    return 0")
    src.append("}")
    return "\n".join(src) + "\n"


def run(path, env_extra):
    env = dict(os.environ)
    env.update(env_extra)
    env["JAITHON_NO_GPU"] = "1"
    p = subprocess.run([JAITHON, "run", "--no-cache", path], env=env,
                       capture_output=True, text=True, timeout=300)
    return p.returncode, p.stdout, p.stderr.strip()[-400:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--kernels", type=int, default=6)
    ap.add_argument("--warm", type=int, default=120)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    bad = 0
    tmp = tempfile.mkdtemp(prefix="jaivecfuzz")
    for case in range(args.count):
        src = program(rng, args.kernels, args.warm)
        path = os.path.join(tmp, f"case{case}.jai")
        with open(path, "w") as fh:
            fh.write(src)
        results = [(name,) + run(path, envx) for name, envx in CONFIGS]
        base = results[0]
        if base[1] != 0:
            print(f"ORACLE FAILED in {path}: {base[3]}")
            bad += 1
            continue
        for name, code, out, err in results[1:]:
            if (code, out) != (base[1], base[2]):
                bad += 1
                print(f"MISMATCH {name} vs no-jit in {path}")
                print(f"  {name} exit={code} err={err}")
                for x, y in zip(base[2].splitlines(), out.splitlines()):
                    if x != y:
                        print(f"  first differing line:\n    {x!r}\n    {y!r}")
                        break
    print(f"vector differential: {args.count} programs x {args.kernels} "
          f"kernels, {len(CONFIGS)} configurations, {bad} disagreement(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
