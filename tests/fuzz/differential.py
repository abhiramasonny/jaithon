#!/usr/bin/env python3
"""Runs generated programs under every configuration of the JIT and diffs them.

The oracle is the tier's own contract (src/vm/jit/jit.h): the compiled tier is
"an accelerator that may always decline", so declining is always allowed and
answering DIFFERENTLY never is. Every configuration below therefore has to
produce byte-identical stdout, the same exit status, and the same exception if
it raises, on every program progen.py invents. A disagreement is a miscompile;
there is no third explanation and no expected output to maintain.

The one deliberate loosening is traceback FRAMES, which legitimately differ: a
compiled frame is never pushed, so it cannot be listed. See `raised` below --
that is a difference in diagnostics, not in what was computed, and comparing it
would make a false positive of every program that raises.

The configurations, and what each is for:

    default   JAITHON_NO_JIT unset       the tier as it ships
    interp    JAITHON_NO_JIT=1           the reference; the tier never runs
    tick      JAITHON_JIT_TICK_US=50     the sampler at its ceiling, so the OSR
                                         loop tier enters far earlier and far
                                         more often than it normally would
    deopt     JAITHON_JIT_DEOPT_STRESS=1 every guard fails, so every deopt stub
                                         and every OSR exit is taken
    split     JAITHON_JIT_SPLIT_STRESS=1 the split operand bank in every body
                                         that can take one
    first     JAITHON_JIT_THRESHOLD=1    every body compiles on its FIRST call
                                         instead of its 64th

`tick` and `first` drive DIFFERENT TIERS and neither subsumes the other:
`first` lowers the entry counter, which is the whole-function tier, while
`tick` raises the sampler rate, which is the OSR loop tier. The nullable-local
miscompile fixed on 2026-09-01 was CORRECT under `first` and WRONG under
`tick`, which is what settled the question of running both.

`--gc` adds a sixth, `--gc-stress=N`, which is worth running occasionally but
is slow: a collection between almost every allocation.

A crash under one configuration and not another is the loudest possible hit and
is reported as such; exit 139 (or -11 from Python) is a segfault. A timeout
under one and not another counts too -- a compiled loop that fails to terminate
is as wrong as one that prints the wrong number.

    python3 tests/fuzz/differential.py                    # 200 programs
    python3 tests/fuzz/differential.py --count 2000
    python3 tests/fuzz/differential.py --seed 8123        # just that one
    python3 tests/fuzz/differential.py --shrink 8123      # reduce a known hit
    python3 tests/fuzz/differential.py --jobs 1 --keep

Exit status is non-zero if any program disagreed with itself.
"""

import argparse
import concurrent.futures
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import progen

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JAITHON = os.environ.get("JAITHON", os.path.join(ROOT, "jaithon"))

MODES = [
    ("default", {}, []),
    ("interp", {"JAITHON_NO_JIT": "1"}, []),
    ("tick", {"JAITHON_JIT_TICK_US": "50"}, []),
    ("deopt", {"JAITHON_JIT_DEOPT_STRESS": "1"}, []),
    ("split", {"JAITHON_JIT_SPLIT_STRESS": "1"}, []),
    ("first", {"JAITHON_JIT_THRESHOLD": "1"}, []),
]

GC_MODE = ("gc", {}, ["--gc-stress=64"])


def raised(err):
    """The exception line(s) of a traceback, with the frame list dropped.

    A compiled frame is never pushed, so it cannot appear in a traceback: the
    interpreter reports

        File ".../case.jai", line 182, in main
        File ".../case.jai", line 76, in probe0
        OverflowError: integer overflow in '+'

    and every configuration that compiled `probe0` reports the same exception
    with the `probe0` line missing. That is a real difference in what the two
    tiers can tell you, and it is worth knowing, but it is a difference in
    DIAGNOSTICS and not in what the program computed -- so comparing raw stderr
    makes every program that raises a false positive. Frame lines and the
    source they echo are indented; the exception itself is not, which is the
    whole rule. `--strict-traceback` turns this off for anyone who does want to
    chase the frame lists.
    """
    keep = []
    for ln in err.splitlines():
        if not ln or ln[0].isspace():
            continue
        if ln.startswith("Traceback (most recent call last)"):
            continue
        keep.append(ln)
    return "\n".join(keep)


class Result:
    __slots__ = ("out", "err", "code", "timedout", "strict")

    def __init__(self, out, err, code, timedout, strict):
        self.out = out
        self.err = err
        self.code = code
        self.timedout = timedout
        self.strict = strict

    def key(self):
        """What must match. stdout and exit status exactly; stderr by what
        was raised, unless --strict-traceback asked for the frames too."""
        if self.timedout:
            return "<timeout>"
        err = self.err if self.strict else raised(self.err)
        return f"exit={self.code}\n--stdout--\n{self.out}\n--raised--\n{err}"

    def summary(self):
        first = self.out.splitlines()[:1]
        if first:
            return first[0]
        line = raised(self.err).splitlines()[:1]
        return line[0] if line else "<no output>"

    def describe(self):
        if self.timedout:
            return "timed out"
        if self.code in (139, -11):
            return "SEGFAULT"
        return f"exit {self.code}"


def run(path, env_extra, flags, timeout, strict):
    env = dict(os.environ)
    env["JAITHON_PATH"] = os.path.join(ROOT, "lib")
    # A stray one of these in the ambient environment would silently apply to
    # every configuration and quietly cancel the whole experiment.
    for name, extra, _ in MODES:
        for key in extra:
            env.pop(key, None)
    env.update(env_extra)
    try:
        done = subprocess.run([JAITHON, "run"] + flags + [path],
                              capture_output=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return Result("", "", None, True, strict)
    return Result(done.stdout.decode("utf-8", "replace").strip(),
                  done.stderr.decode("utf-8", "replace").strip(),
                  done.returncode, False, strict)


def run_all(source, workdir, modes, timeout, strict=False):
    """Run one program under every mode. Returns {mode name: Result}."""
    path = os.path.join(workdir, "case.jai")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(source)
    return {name: run(path, extra, flags, timeout, strict)
            for name, extra, flags in modes}


def disagreeing(results):
    """The mode names that do not match the majority, or () if all agree.

    `interp` is the reference, so whatever it printed is right by definition
    and everything else is measured against it.
    """
    want = results["interp"].key()
    off = tuple(name for name, res in results.items() if res.key() != want)
    return off


def report(seed, results, off, out=sys.stdout):
    print(f"\nMISMATCH seed {seed}: "
          f"{', '.join(off)} disagree with interp", file=out)
    for name, res in results.items():
        mark = "!!" if name in off else "  "
        print(f"  {mark} {name:8s} {res.describe():10s} "
              f"{res.summary()[:110]}", file=out)
    ref, bad = results["interp"], results[off[0]]
    if ref.timedout or bad.timedout:
        return
    for what, a_text, b_text in (("stdout", ref.out, bad.out),
                                 ("raised", raised(ref.err), raised(bad.err))):
        if a_text == b_text:
            continue
        al, bl = a_text.splitlines(), b_text.splitlines()
        for i in range(max(len(al), len(bl))):
            a = al[i] if i < len(al) else "<missing>"
            b = bl[i] if i < len(bl) else "<missing>"
            if a != b:
                print(f"  {what} line {i + 1}:", file=out)
                print(f"    interp      {a[:150]}", file=out)
                print(f"    {off[0]:11s} {b[:150]}", file=out)
                break


def check_seed(seed, warm, modes, timeout, keep, strict=False):
    """Generate one program, run it every way, and say whether it agreed."""
    workdir = tempfile.mkdtemp(prefix=f"jitfuzz-{seed}-")
    try:
        source = progen.generate(seed, warm).render()
        results = run_all(source, workdir, modes, timeout, strict)
        off = disagreeing(results)
        if off and keep:
            shutil.copy(os.path.join(workdir, "case.jai"),
                        os.path.join(tempfile.gettempdir(),
                                     f"jitfuzz-hit-{seed}.jai"))
        return seed, results, off
    finally:
        if not keep:
            shutil.rmtree(workdir, ignore_errors=True)


def make_oracle(warm, modes, timeout, strict=False, reject_timeouts=False):
    """A callable the shrinker uses: source text in, disagreeing modes out.

    `reject_timeouts` makes a candidate where anything timed out count as "no
    disagreement", so the shrinker puts it back. A reduction can legitimately
    make a program much SLOWER -- delete the early `return` from a loop and the
    loop now runs every iteration -- and a program that is merely slow is
    indistinguishable here from one that hangs. Without this the shrinker
    adopts the slow candidate, every later candidate inherits it, and the run
    dissolves into 120-second timeouts. The cost is that a hit which is ONLY a
    hang cannot be reduced; it is reported at full size instead, which is the
    right way round.
    """
    def oracle(source):
        workdir = tempfile.mkdtemp(prefix="jitshrink-")
        try:
            results = run_all(source, workdir, modes, timeout, strict)
            if reject_timeouts and any(r.timedout for r in results.values()):
                return ()
            return disagreeing(results)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
    return oracle


def do_shrink(seed, warm, modes, timeout, strict=False, repeat=2):
    import shrink
    import time
    prog = progen.generate(seed, warm)
    # Time the original, then hold every candidate to a small multiple of it.
    # A reduction should not be slower than what it reduces, and the default
    # 120s is far too generous to notice when one is.
    started = time.time()
    check = make_oracle(warm, modes, timeout, strict)
    off = check(prog.render())
    budget = max(10, min(timeout, int((time.time() - started) * 4) + 5))
    oracle = make_oracle(warm, modes, budget, strict, reject_timeouts=True)
    if not off:
        print(f"seed {seed} does not reproduce -- nothing to shrink.")
        print("Note the tier is sampler-driven, so an OSR-only hit can be "
              "intermittent; try re-running, or --warm higher.")
        return 1
    print(f"seed {seed} reproduces: {', '.join(off)} disagree with interp")
    print("shrinking...")
    small, steps = shrink.reduce(prog, oracle, off, repeat)
    source = small.render()
    path = os.path.join(tempfile.gettempdir(), f"jitfuzz-min-{seed}.jai")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(source)
    lines = len(source.splitlines())
    print(f"\n{steps} oracle runs; {lines} lines left. Written to {path}")
    hits = shrink.confirm(small, oracle, off, 5)
    print(f"the reduced program reproduces {hits}/5 runs"
          + ("" if hits == 5 else "  <-- INTERMITTENT, see progen.py on OSR")
          + "\n")
    print(source)
    print("Reproduce with (!! marks the configurations that disagree):")
    for name, extra, flags in modes:
        parts = [f"{k}={v}" for k, v in extra.items()]
        parts += [JAITHON, "run"] + flags + [path]
        print(f"  {'!!' if name in off else '  '} {' '.join(parts)}")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=200,
                    help="how many programs to generate and diff")
    ap.add_argument("--start", type=int, default=0, help="first seed")
    ap.add_argument("--seed", type=int, default=None,
                    help="check exactly this one seed")
    ap.add_argument("--shrink", type=int, default=None, metavar="SEED",
                    help="reduce a known failing seed to a minimal program")
    ap.add_argument("--warm", type=int, default=progen.WARM_DEFAULT,
                    help="calls each probe gets; must exceed the tier's 64")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--gc", action="store_true",
                    help="add a --gc-stress configuration (slow)")
    ap.add_argument("--shrink-repeat", type=int, default=2,
                    help="consecutive confirmations before a reduction is "
                         "adopted; 1 is fast and unreliable, 3 is slow and "
                         "trustworthy on sampler-dependent hits")
    ap.add_argument("--strict-traceback", action="store_true",
                    help="also require traceback FRAMES to match; a compiled "
                         "frame is never pushed, so this reports every "
                         "program that raises")
    ap.add_argument("--keep", action="store_true",
                    help="keep the .jai of any program that disagreed")
    args = ap.parse_args()

    if not os.access(JAITHON, os.X_OK):
        print(f"error: {JAITHON} not built. Run 'make' first.", file=sys.stderr)
        return 2

    modes = list(MODES) + ([GC_MODE] if args.gc else [])

    if args.shrink is not None:
        return do_shrink(args.shrink, args.warm, modes, args.timeout,
                         args.strict_traceback, args.shrink_repeat)

    seeds = ([args.seed] if args.seed is not None
             else list(range(args.start, args.start + args.count)))
    print(f"{len(seeds)} programs x {len(modes)} configurations "
          f"({', '.join(n for n, _, _ in modes)}), warm={args.warm}, "
          f"jobs={args.jobs}")

    hits = []
    done = 0
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        futures = [pool.submit(check_seed, s, args.warm, modes, args.timeout,
                               args.keep, args.strict_traceback)
                   for s in seeds]
        for fut in concurrent.futures.as_completed(futures):
            try:
                seed, results, off = fut.result()
            except Exception:
                traceback.print_exc()
                continue
            done += 1
            if off:
                hits.append(seed)
                report(seed, results, off)
            if done % 25 == 0 or done == len(seeds):
                print(f"  {done}/{len(seeds)} checked, {len(hits)} mismatched",
                      flush=True)

    if not hits:
        print(f"\nno disagreements in {len(seeds)} programs.")
        return 0
    print(f"\n{len(hits)} of {len(seeds)} programs disagreed: "
          f"{sorted(hits)}")
    print("Reduce one with:")
    print(f"  python3 tests/fuzz/differential.py --shrink {sorted(hits)[0]}"
          f" --warm {args.warm}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
