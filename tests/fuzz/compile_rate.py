#!/usr/bin/env python3
"""Reports what fraction of generated probes actually reach the compiled tier.

A generator that emits programs the tier refuses is a generator that compares
the interpreter with itself, so the oracle in differential.py proves nothing.
`JAI_JIT_WHY=1` names every attempt and every refusal; this counts them.

"Compiled" is not the whole story, so this also counts how far the walk got:
the tier stops at the first opcode it does not model and interprets the rest,
and a probe reported as compiled can be compiled for three instructions.
Measured over 80 programs, 94 of 110 cuts land before 85% of the body, the
median at 68%, and 82 of those 94 are OP_MUL_WRAP or OP_ADD_WRAP -- opcodes
src/vm/jit has no handler for at all, and which progen.py chose deliberately
so an edge literal could not raise. That is a generator decision, so it is a
generator decision to revisit.

    python3 tests/fuzz/compile_rate.py --count 40
    python3 tests/fuzz/compile_rate.py --count 40 --warm 400 --reasons 20
"""

import argparse
import collections
import concurrent.futures
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import progen

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JAITHON = os.environ.get("JAITHON", os.path.join(ROOT, "jaithon"))

COMPILED = re.compile(r"^\[jit\] compiled (\S+)\s")
OSR = re.compile(r"^\[jit\] osr (\S+) at \d+:")
STOPPED = re.compile(r"^\[jit\] (\S+) stopped \(measuring\): (.*)$")
WALKED = re.compile(r"^\[jit\] (\S+) walked only to (\S+) at \d+ --")


def probe_names(source):
    return re.findall(r"^fn (probe\d+)\(", source, re.MULTILINE)


def bare(name):
    """`[jit]` labels a body `module.function`; the probe names are bare.

    jit_why.c's jitFnLabel qualifies every diagnostic it prints, because a bare
    name is ambiguous -- `check lib/std` has three hot functions called `init`.
    This script predates that and compared `__main__.probe1` against `probe1`,
    so every match failed and the whole report read 0.0% on a corpus that in
    fact compiles hundreds of bodies per program. Strip the module here rather
    than loosening the regexes, so an unqualified label still matches.
    """
    return name.rsplit(".", 1)[-1]


def one(seed, warm, timeout):
    source = progen.generate(seed, warm).render()
    names = probe_names(source)
    with tempfile.TemporaryDirectory(prefix=f"jitrate-{seed}-") as work:
        path = os.path.join(work, "case.jai")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(source)
        env = dict(os.environ)
        env["JAI_JIT_WHY"] = "1"
        env["JAITHON_PATH"] = os.path.join(ROOT, "lib")
        try:
            done = subprocess.run([JAITHON, "run", path], capture_output=True,
                                  env=env, timeout=timeout)
        except subprocess.TimeoutExpired:
            return seed, names, set(), set(), [], {}
        err = done.stderr.decode("utf-8", "replace")
    fn, osr, why, cut = set(), set(), [], {}
    for ln in err.splitlines():
        m = COMPILED.match(ln)
        if m and bare(m.group(1)) in names:
            fn.add(m.group(1))
        m = OSR.match(ln)
        if m and bare(m.group(1)) in names:
            osr.add(m.group(1))
        m = STOPPED.match(ln)
        if m and bare(m.group(1)) in names:
            why.append(m.group(2))
        m = WALKED.match(ln)
        if m and bare(m.group(1)) in names:
            why.append("walked only to " + m.group(2))
            # A compiled body is not a compiled BODY: the walk stops at the
            # first opcode this tier does not model and everything after it is
            # interpreted, so "compiled" alone overstates what the oracle
            # actually covers. Which opcode did the cutting is the actionable
            # part -- if it is one the generator chose, the generator can stop.
            cut.setdefault(m.group(1), m.group(2))
    return seed, names, fn, osr, why, cut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=40)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--warm", type=int, default=progen.WARM_DEFAULT)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--reasons", type=int, default=12)
    args = ap.parse_args()

    seeds = range(args.start, args.start + args.count)
    total = compiled_fn = compiled_any = 0
    progs_any = 0
    reasons = collections.Counter()
    cuts = collections.Counter()
    truncated = 0
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        for seed, names, fn, osr, why, cut in pool.map(
                lambda s: one(s, args.warm, args.timeout), seeds):
            total += len(names)
            compiled_fn += len(fn)
            compiled_any += len(fn | osr)
            progs_any += 1 if (fn | osr) else 0
            reasons.update(why)
            for name in fn | osr:
                if name in cut:
                    truncated += 1
                    cuts[cut[name]] += 1

    pct = lambda a, b: f"{a}/{b} = {100.0 * a / max(1, b):.1f}%"
    print(f"{args.count} programs, warm={args.warm}")
    print(f"  probes reaching ANY compiled tier   {pct(compiled_any, total)}")
    print(f"  probes reaching the function tier   {pct(compiled_fn, total)}")
    print(f"  programs with >=1 compiled probe    {pct(progs_any, args.count)}")
    print(f"  of those, walk cut short before the end     "
          f"{pct(truncated, compiled_any)}")
    print("\nwhat cut a compiled probe's walk short:")
    for op, n in cuts.most_common(6):
        print(f"  {n:5d}  {op}")
    print("\ntop refusals inside a probe:")
    for reason, n in reasons.most_common(args.reasons):
        print(f"  {n:5d}  {reason[:100]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
