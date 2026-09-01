#!/usr/bin/env python3
"""How much of what progen.py generates actually reaches the compiled tier.

The failure mode this exists to catch: a generator that emits programs the
tier REFUSES leaves differential.py comparing the interpreter with itself,
so every seed passes and nothing was tested. `JAI_JIT_WHY=1` names every
compile and every refusal, and this counts them.

A function counts as compiled when a run prints either

    [jit] compiled NAME arity=1 ...     the call-count whole-function tier
    [jit] osr NAME at N: M instructions the sampler-driven loop tier

and the refusal text is tallied for everything else, most common first --
which is the list to work down when the rate is low. Probes and container
helpers are counted separately because they fail for different reasons: a
probe declines on whatever its random body happens to contain, a container
helper on what the tier can do with the container it was handed.

    python3 tests/fuzz/compile_rate.py --count 80
    python3 tests/fuzz/compile_rate.py --count 80 --warm 400

Three findings came straight out of running this, each worth more than any
single grammar rule that was added, and all three are self-inflicted harness
problems rather than anything about the programs:

  - a container BUILT in the body that reads it has no live sample, so the
    dict index, the list element kind and the set length all decline. Passed
    in as a parameter the same work compiles: 1 of 72 against clean;
  - `sfold(str(x))` refuses the whole enclosing body on `OP_TYPE_GUARD: a
    str guard on a object`, and `sfold(f"{x}")` compiles. Same text;
  - a container helper that is generated but never CALLED is never hot, so
    it is never even considered -- 201 generated, 0 called, 0 compiled.

The self-hosted compiler and the test front end are loaded by every run and
are themselves hot, so their `[jit]` lines are filtered out by name.
"""

import argparse
import collections
import concurrent.futures
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import progen

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JAITHON = os.environ.get("JAITHON", os.path.join(ROOT, "jaithon"))

COMPILED = re.compile(r"^\[jit\] compiled (\S+)\s")
OSR = re.compile(r"^\[jit\] osr (\S+) at \d+: \d+ instructions")
STOPPED = re.compile(r"^\[jit\] (\S+) (?:stopped[^:]*|walked only to \S+ at \d+)"
                     r"(?:: (.*))?$")
# The container helpers progen names cd/cs/ct/cn/cm plus a counter. They carry
# the shape under test with the container arriving as a PARAMETER, which is the
# only way most of it reaches the whole-function tier, so they are counted
# separately rather than folded into the probe number.
HELPER = re.compile(r"^c[dstnm]\d+$")


def measure(seed, warm, timeout):
    """Run one generated program with JAI_JIT_WHY=1.

    Returns (probes present, probes compiled, helpers present, helpers
    compiled, [refusal texts]).
    """
    workdir = tempfile.mkdtemp(prefix=f"jitrate-{seed}-")
    empty = (set(), set(), set(), set())
    try:
        prog = progen.generate(seed, warm)
        source = prog.render()
        path = os.path.join(workdir, "case.jai")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(source)
        env = dict(os.environ)
        env["JAITHON_PATH"] = os.path.join(ROOT, "lib")
        env["JAI_JIT_WHY"] = "1"
        try:
            done = subprocess.run([JAITHON, "run", path], capture_output=True,
                                  env=env, timeout=timeout)
        except subprocess.TimeoutExpired:
            return empty + (["<timeout>"],)
        text = done.stderr.decode("utf-8", "replace")
        if done.returncode not in (0, 1):
            return empty + ([f"<exit {done.returncode}>"],)
        probes = {p.name for p in prog.probes}
        helpers = {m.group(1) for m in
                   re.finditer(r"^fn (c[dstnm]\d+)\(", source, re.M)}
        wanted = probes | helpers
        hit, why = set(), []
        for ln in text.splitlines():
            m = COMPILED.match(ln) or OSR.match(ln)
            if m and m.group(1) in wanted:
                hit.add(m.group(1))
                continue
            m = STOPPED.match(ln)
            if m and m.group(1) in wanted:
                why.append(m.group(2) or "walked only part of the body")
        return (probes, hit & probes, helpers, hit & helpers, why)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--warm", type=int, default=progen.WARM_DEFAULT)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    if not os.access(JAITHON, os.X_OK):
        print(f"error: {JAITHON} not built. Run 'make' first.", file=sys.stderr)
        return 2

    seeds = list(range(args.start, args.start + args.count))
    probes = compiled = helpers = hcompiled = 0
    programs = hot_programs = 0
    reasons = collections.Counter()
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        for pnames, phit, hnames, hhit, why in pool.map(
                lambda s: measure(s, args.warm, args.timeout), seeds):
            probes += len(pnames)
            compiled += len(phit)
            helpers += len(hnames)
            hcompiled += len(hhit)
            programs += 1
            if phit or hhit:
                hot_programs += 1
            reasons.update(w[:100] for w in why)

    def rate(a, b):
        return f"{a}/{b} ({100.0 * a / b:.1f}%)" if b else f"{a}/0"

    print(f"probes            {rate(compiled, probes)} reached a compiled tier")
    print(f"container helpers {rate(hcompiled, helpers)}")
    print(f"programs          {rate(hot_programs, programs)} had at least one")
    if reasons:
        print("\nwhy the rest declined:")
        for text, n in reasons.most_common(args.top):
            print(f"  {n:5d}  {text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
