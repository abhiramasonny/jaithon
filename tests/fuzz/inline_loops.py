#!/usr/bin/env python3
"""Differential fuzzer for loop-bearing inlines (JAITHON_JIT_INLINE_LOOPS).

progen.py almost never writes the shape this mechanism takes -- a small
helper with a loop in it, called directly from a body hot enough for the
function tier -- so the main fuzzer cannot validate it. This one writes
nothing else: helpers with `while` and `for .. in a..b` loops, returns from
inside them at random points, parameters the helper reassigns, nested loops,
reads of int, float and mixed lists, indexes that run past the end (raised
and caught in the driver), and a list whose element kind changes half-way
through the run, so a guard fails after some iterations of an inlined loop
have already run. Drivers call the helpers inside expressions, twice in one
expression, and inside `try`.

Every program must print the same thing under every configuration:

    interp   JAITHON_NO_JIT=1                  the reference
    default  (nothing)
    off      JAITHON_JIT_INLINE_LOOPS=0        the same tier without the inline
    thresh   JAITHON_JIT_THRESHOLD=1
    deopt    JAITHON_JIT_DEOPT_STRESS=1        every guard fails: every call
                                               re-runs in the interpreter
    tick     JAITHON_JIT_TICK_US=50
    gc       --gc-stress=3 (with --gc)             a collection every third
                                               allocation around the inlines

    python3 tests/fuzz/inline_loops.py --count 200 --jobs 6
    python3 tests/fuzz/inline_loops.py --seed 17 --keep
"""
import argparse
import concurrent.futures
import os
import random
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JAITHON = os.path.join(ROOT, "jaithon")

CONFIGS = [
    ("interp", {"JAITHON_NO_JIT": "1"}),
    ("default", {}),
    ("off", {"JAITHON_JIT_INLINE_LOOPS": "0"}),
    ("thresh", {"JAITHON_JIT_THRESHOLD": "1"}),
    ("deopt", {"JAITHON_JIT_DEOPT_STRESS": "1"}),
    ("tick", {"JAITHON_JIT_TICK_US": "50"}),
]


class Gen:
    def __init__(self, seed):
        self.r = random.Random(seed)
        self.helpers = []   # (name, params, ret, src)

    def int_term(self, ints):
        r = self.r
        c = r.random()
        if c < 0.5:
            return r.choice(ints)
        if c < 0.75:
            return str(r.randint(-3, 9))
        a, b = r.choice(ints), r.choice(ints)
        return f"({a} {r.choice(['+%', '-%', '*%'])} {b})"

    def cond(self, ints):
        r = self.r
        return f"{self.int_term(ints)} {r.choice(['<', '<=', '>', '>=', '==', '!='])} {self.int_term(ints)}"

    def helper(self, idx):
        r = self.r
        name = f"h{idx}"
        kind = r.choice(["ints", "ints", "floats", "any", "plain"])
        params = ["n: int", "k: int"]
        if kind == "ints":
            params.insert(0, "xs: list[int]")
        elif kind == "floats":
            params.insert(0, "xs: list[float]")
        elif kind == "any":
            params.insert(0, "xs: list[any]")
        ret = "float" if kind == "floats" and r.random() < 0.6 else "int"
        ints = ["n", "k", "t", "i"]
        body = []
        body.append(f"    var t = {'0.0' if ret == 'float' else self.int_term(['n', 'k'])}")
        loop = r.choice(["while", "range", "nested"])
        if loop == "while":
            body.append("    var i = 0")
            body.append(f"    while i < n {{")
            ind = "        "
        elif loop == "range":
            body.append("    for i in 0..n {")
            ind = "        "
        else:
            body.append("    var i = 0")
            body.append("    while i < n {")
            body.append(f"        for j in 0..{r.randint(1, 4)} {{")
            ind = "            "
            ints = ints + ["j"]
        # The element read, if any.
        elem = None
        if kind in ("ints", "any", "floats"):
            off = r.choice(["i", "i", f"(i + {r.randint(0, 2)})"])
            elem = f"xs[{off}]"
            body.append(f"{ind}let e = {elem}")
        if ret == "float":
            src = "e" if elem else "float(i)"
            body.append(f"{ind}t = t * 0.5 + {src}")
            if r.random() < 0.6:
                body.append(f"{ind}if t > {r.randint(5, 60)}.0 {{ return t }}")
        else:
            if elem and kind == "floats":
                body.append(f"{ind}if e > {r.randint(0, 9)}.5 {{ t = t +% 1 }}")
            elif elem:
                # int(e): a mixed list's float element must not raise here.
                add = "int(e)" if kind == "any" else "e"
                body.append(f"{ind}if e {r.choice(['<', '>', '=='])} {self.int_term(['k', 'i'])} {{ t = t +% {add} }}")
            body.append(f"{ind}t = t +% {self.int_term(ints)}")
            if r.random() < 0.7:
                body.append(f"{ind}if {self.cond(ints)} {{ return t }}")
            if r.random() < 0.3:
                body.append(f"{ind}if t % {r.randint(2, 9)} == 0 {{ return {self.int_term(ints)} }}")
            if r.random() < 0.3:
                body.append(f"{ind}k = k -% 1")
        if loop == "nested":
            body.append("        }")
            body.append("        i += 1")
            body.append("    }")
        elif loop == "while":
            body.append("        i += 1")
            body.append("    }")
        else:
            body.append("    }")
        if ret == "float":
            body.append("    return t")
        else:
            body.append(f"    return t +% {r.choice(['n', 'k', '1'])}")
        src = f"fn {name}({', '.join(params)}) -> {ret} {{\n" + "\n".join(body) + "\n}"
        self.helpers.append((name, kind, ret, src))

    def method(self, idx):
        """A method whose loop reads the receiver's fields."""
        r = self.r
        name = f"m{idx}"
        ints = ["n", "k", "t", "i", "self.bias"]
        body = [f"    pub fn {name}(self, n: int, k: int) -> int {{",
                f"        var t = {self.int_term(['n', 'k'])}",
                "        var i = 0",
                "        while i < n {",
                f"            let e = self.ws[{r.choice(['i', 'i', '(i + 1)'])}]",
                f"            if e {r.choice(['<', '>', '=='])} {self.int_term(['k', 'i'])} {{ t = t +% e }}",
                f"            t = t +% {self.int_term(ints)}"]
        if r.random() < 0.7:
            body.append(f"            if {self.cond(ints)} {{ return t }}")
        body += ["            i += 1",
                 "        }",
                 "        return t +% self.bias",
                 "    }"]
        return name, "\n".join(body)

    def call(self, h, n_expr, k_expr):
        name, kind, ret, _ = h
        if kind == "method":
            return f"kk.{name}({n_expr}, {k_expr})"
        lst = {"ints": "ints", "floats": "fls", "any": "mix", "plain": None}[kind]
        args = ([lst] if lst else []) + [n_expr, k_expr]
        c = f"{name}({', '.join(args)})"
        return f"int({c})" if ret == "float" else c

    def program(self):
        r = self.r
        for i in range(r.randint(2, 4)):
            self.helper(i)
        lines = []
        for _, _, _, src in self.helpers:
            lines.append(src)
            lines.append("")
        methods = [self.method(i) for i in range(r.randint(1, 2))]
        lines.append("class Kk {")
        lines.append("    pub var ws: list[int]")
        lines.append("    pub var bias: int")
        lines.append("    fn init(self, ws: list[int], bias: int) {")
        lines.append("        self.ws = ws")
        lines.append("        self.bias = bias")
        lines.append("    }")
        for _, src in methods:
            lines.append(src)
        lines.append("}")
        lines.append("")
        for name, _ in methods:
            self.helpers.append((name, "method", "int", ""))
        lines.append("fn side(x: int) -> int { return x % 5 }")
        lines.append("")
        drivers = []
        for d in range(r.randint(1, 3)):
            name = f"drive{d}"
            hs = [r.choice(self.helpers) for _ in range(r.randint(1, 3))]
            body = ["    var acc = 0"]
            for h in hs:
                n_expr = r.choice(["r % 9", "r % 12", "4", "r % 3 + 6"])
                k_expr = r.choice(["r % 7", "r", "3", "r % 11 - 2"])
                c = self.call(h, n_expr, k_expr)
                if r.random() < 0.35:
                    body.append("    try {")
                    body.append(f"        acc = acc +% {c}")
                    body.append("    } catch e: IndexError {")
                    body.append("        acc = acc +% 1000")
                    body.append("    }")
                elif r.random() < 0.3:
                    body.append(f"    acc = acc +% ({c} *% 2 +% {c})")
                else:
                    body.append(f"    acc = acc +% {c}")
            body.append("    return acc +% side(r)")
            lines.append(f"fn {name}(ints: list[int], fls: list[float], mix: list[any], kk: Kk, r: int) -> int {{")
            lines.extend(body)
            lines.append("}")
            lines.append("")
            drivers.append(name)
        reps = r.randint(150, 400)
        flip = r.randint(50, reps - 20)
        lines.append("fn main() -> int {")
        lines.append("    var ints: list[int] = []")
        lines.append("    var fls: list[float] = []")
        lines.append("    var mix: list[any] = []")
        size = r.randint(8, 16)
        lines.append(f"    for q in 0..{size} {{")
        lines.append(f"        ints.push((q * {r.randint(3, 17)}) % {r.randint(5, 23)})")
        lines.append(f"        fls.push(float(q) * 0.25)")
        lines.append("        mix.push(q)")
        lines.append("    }")
        lines.append(f"    let kk = Kk(ints, {r.randint(-3, 9)})")
        lines.append("    var total = 0")
        lines.append(f"    for r in 0..{reps} {{")
        lines.append(f"        if r == {flip} {{ mix[{r.randint(0, size - 1)}] = 2.5 }}")
        for d in drivers:
            lines.append("        try {")
            lines.append(f"            total = total +% {d}(ints, fls, mix, kk, r)")
            lines.append("        } catch e: IndexError {")
            lines.append("            total = total +% 7")
            lines.append("        }")
        lines.append("    }")
        lines.append("    print(total)")
        lines.append("    return 0")
        lines.append("}")
        return "\n".join(lines) + "\n"


def run(path, env_extra, timeout):
    env = dict(os.environ)
    env.update(env_extra)
    flags = env.pop("_FLAGS", "").split()
    try:
        p = subprocess.run([JAITHON] + flags + ["run", path],
                           capture_output=True, text=True,
                           env=env, timeout=timeout, cwd=ROOT)
        err = p.stderr.strip().splitlines()
        last = err[-1] if (p.returncode != 0 and err) else ""
        return (p.returncode, p.stdout, last)
    except subprocess.TimeoutExpired:
        return ("timeout", "", "")


def check(seed, timeout, keep, tmpdir):
    src = Gen(seed).program()
    path = os.path.join(tmpdir, f"il_{seed}.jai")
    with open(path, "w") as f:
        f.write(src)
    results = [(name, run(path, env, timeout)) for name, env in CONFIGS]
    ref = results[0][1]
    bad = [(n, res) for n, res in results[1:] if res != ref]
    if not bad and not keep:
        os.unlink(path)
    return seed, ref, bad, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--gc", action="store_true")
    args = ap.parse_args()
    if args.gc:
        CONFIGS.append(("gc", {"_FLAGS": "--gc-stress=3"}))
    seeds = [args.seed] if args.seed is not None else \
        list(range(args.start, args.start + args.count))
    tmpdir = tempfile.mkdtemp(prefix="jai_inline_loops_")
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(check, s, args.timeout, args.keep, tmpdir) for s in seeds]
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            seed, ref, bad, path = fut.result()
            if bad:
                failed += 1
                print(f"seed {seed}: reference {ref!r}")
                for n, res in bad:
                    print(f"    {n}: {res!r}")
                print(f"    program: {path}")
            if i % 25 == 0:
                print(f"  {i}/{len(seeds)} checked, {failed} mismatched", flush=True)
    if failed:
        print(f"{failed} of {len(seeds)} programs disagreed.")
        sys.exit(1)
    print(f"no disagreements in {len(seeds)} programs.")


if __name__ == "__main__":
    main()
