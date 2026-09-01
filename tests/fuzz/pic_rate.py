#!/usr/bin/env python3
"""Reports what fraction of generated programs reach the polymorphic inline
cache -- `emitInvokePic1`, jit_call_pic.c, the most intricate arm in the tier.

It gets its own counter because it is the one arm a corpus can miss entirely
while every other number looks healthy. A census taken before the generator
knew the shape found it in 0 of 100 programs, in every configuration, while
compile_rate.py was reporting 42% of probes compiled: the arm needs a receiver
that is SLOT_INST with no class pinned, which happens at exactly one place in
the tier, and nothing progen.py emitted arrived there. See
src/vm/jit/README.md, "The polymorphic inline cache", for what that place is.

`JAI_JIT_WHY=1` prints one `[jit] pic N-way (of M recorded, state S)` line per
site the arm accepted; this counts programs that emitted at least one, under
each configuration the differential fuzzer runs. `interp` is not here -- with
no compiled tier there is nothing to count.

    python3 tests/fuzz/pic_rate.py --count 100
    python3 tests/fuzz/pic_rate.py --count 40 --start 500 --sites
"""

import argparse
import concurrent.futures
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import progen

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JAITHON = os.environ.get("JAITHON", os.path.join(ROOT, "jaithon"))

# The compiled half of differential.py's MODES, in its order.
MODES = [
    ("default", {}),
    ("tick", {"JAITHON_JIT_TICK_US": "50"}),
    ("deopt", {"JAITHON_JIT_DEOPT_STRESS": "1"}),
    ("split", {"JAITHON_JIT_SPLIT_STRESS": "1"}),
    ("thresh", {"JAITHON_JIT_THRESHOLD": "1"}),
]


def one(seed, warm, timeout):
    """Sites reached per configuration, or -1 where the program timed out."""
    source = progen.generate(seed, warm).render()
    work = tempfile.mkdtemp(prefix=f"jitpic-{seed}-")
    try:
        path = os.path.join(work, "case.jai")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(source)
        got = {}
        for name, extra in MODES:
            env = dict(os.environ)
            # A stray one of these in the ambient environment would apply to
            # every configuration and quietly cancel the experiment.
            for _, other in MODES:
                for key in other:
                    env.pop(key, None)
            env.pop("JAITHON_NO_JIT", None)
            env["JAI_JIT_WHY"] = "1"
            env["JAITHON_PATH"] = os.path.join(ROOT, "lib")
            env.update(extra)
            try:
                done = subprocess.run([JAITHON, "run", path],
                                      capture_output=True, env=env,
                                      timeout=timeout)
            except subprocess.TimeoutExpired:
                got[name] = -1
                continue
            err = done.stderr.decode("utf-8", "replace")
            got[name] = err.count("[jit] pic ")
        return seed, got
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--warm", type=int, default=progen.WARM_DEFAULT)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--sites", action="store_true",
                    help="also report total sites, not just programs")
    args = ap.parse_args()

    seeds = range(args.start, args.start + args.count)
    hit = {name: 0 for name, _ in MODES}
    sites = {name: 0 for name, _ in MODES}
    timeouts = {name: 0 for name, _ in MODES}
    any_hit = 0
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        for _, got in pool.map(
                lambda s: one(s, args.warm, args.timeout), seeds):
            for name, _ in MODES:
                if got[name] > 0:
                    hit[name] += 1
                    sites[name] += got[name]
                elif got[name] < 0:
                    timeouts[name] += 1
            if any(got[name] > 0 for name, _ in MODES):
                any_hit += 1

    n = args.count
    pct = lambda a: f"{a}/{n} = {100.0 * a / max(1, n):.0f}%"
    print(f"{n} programs, seeds {args.start}..{args.start + n - 1}, "
          f"warm={args.warm}")
    for name, _ in MODES:
        extra = f"   ({timeouts[name]} timed out)" if timeouts[name] else ""
        line = f"  {name:8s} {pct(hit[name]):>16s}"
        if args.sites:
            line += f"   {sites[name]:5d} sites"
        print(line + extra)
    print(f"  {'ANY':8s} {pct(any_hit):>16s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
