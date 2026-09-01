#!/usr/bin/env python3
#: A/B two configurations of ONE binary, with the noise floor measured first.
#:
#: Every number this tier accepts comes from an A/B in a single build with the
#: two forms interleaved -- a switch is what makes that possible, and
#: src/vm/jit/README.md lists all fifty. What that discipline still left open is
#: the question this script exists to force: IS THE MACHINE QUIET ENOUGH TO SEE
#: THE EFFECT AT ALL?
#:
#: It usually is not. Six identical runs of `check --no-cache lib/std` spread
#: 3.36% and did so MONOTONICALLY -- 293M, 294M, 294M, 301M, 302M, 302M -- which
#: is drift, not noise, so averaging launders it rather than cancelling it. The
#: cause is the OSR tier: it arms on a SIGPROF tick, so which loops it catches
#: depends on the wall clock. --pin sets JAITHON_JIT_THRESHOLD=1 and
#: JAITHON_JIT_TICK_US=100000, which took the same six runs to 0.017%.
#:
#: Pinning measures a different regime -- nothing stays cold -- so it is the
#: right instrument for a change to the FUNCTION tier's decisions and the wrong
#: one for anything about warm-up or the OSR tier itself.
#:
#: AND THE TWO CAN BE IN TENSION, which is the trap worth knowing about.
#: JAITHON_JIT_NULLABLE_FB changes what a caller does when its callee has NO
#: compiled form yet, so it is read out of the interpreter's observation record.
#: Under --pin every callee compiles on its first call, so that record is barely
#: consulted -- the refusals it governs fall from 563 to 166 on lib/std -- and
#: the A/B comes back at EXACTLY 0.00% with a 0.000% floor. That is not "no
#: effect"; it is "this instrument cannot see this switch". A dead-flat result
#: under --pin is a reason to check that the switch is still reachable in the
#: pinned regime, not a verdict.
#:
#: And it does not survive a busy machine. With agents building in worktrees
#: (load ~3) the same pinned configuration read 426M in one measurement and 353M
#: twenty minutes later, and interleaved pairs disagreed in SIGN. That is why
#: the floor is measured every time rather than assumed.
#:
#:     scripts/dev/ab.py JAITHON_JIT_MAYBE_OBJ_STACK --pin
#:     scripts/dev/ab.py JAITHON_JIT_JOIN --off 0 --on 1 --pairs 5

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JAI = ROOT / "jaithon"
COUNT = re.compile(r"^vm: (\d+)", re.M)

PIN = {"JAITHON_JIT_THRESHOLD": "1", "JAITHON_JIT_TICK_US": "100000"}

#: A floor of exactly zero, which --pin does produce from a handful of runs, is
#: not evidence that a five-hundredth of a percent is real -- it is evidence
#: that the sample is small. Nothing below this is called an effect whatever the
#: floor says.
MIN_EFFECT = 0.05


def measure(workload, extra):
    env = dict(os.environ)
    env.update(extra)
    done = subprocess.run([str(JAI)] + workload, capture_output=True, env=env,
                          cwd=ROOT)
    text = (done.stdout + done.stderr).decode("utf-8", "replace")
    m = COUNT.search(text)
    if not m:
        print("error: no `vm: N instructions` line -- is --stats in the "
              "workload?", file=sys.stderr)
        sys.exit(2)
    return int(m.group(1))


def spread(values):
    return 100.0 * (max(values) - min(values)) / min(values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("switch", help="environment switch to A/B")
    ap.add_argument("--off", default="0")
    ap.add_argument("--on", default=None,
                    help="value for the ON side; omitted means unset (default)")
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--floor", type=int, default=6,
                    help="identical runs used to measure the noise floor")
    ap.add_argument("--pin", action="store_true",
                    help="pin both tiers: deterministic, but a different regime")
    ap.add_argument("--workload", default="check --no-cache --stats lib/std")
    args = ap.parse_args()

    if not JAI.exists():
        print("error: ./jaithon not built. Run 'make' first.", file=sys.stderr)
        return 2

    workload = args.workload.split()
    base = dict(PIN) if args.pin else {}
    on_env = dict(base)
    if args.on is not None:
        on_env[args.switch] = args.on
    else:
        on_env.pop(args.switch, None)
    off_env = dict(base, **{args.switch: args.off})

    # Warm whatever caches the workload builds, and discard it.
    measure(workload, on_env)

    floor = [measure(workload, on_env) for _ in range(args.floor)]
    fs = spread(floor)
    print(f"noise floor: {args.floor} identical runs, spread {fs:.3f}%")
    print("  " + "  ".join(str(v) for v in floor))
    if not args.pin and fs > 1.0:
        print("\n  The floor is wider than most changes worth making. Try "
              "--pin, or\n  measure when the machine is quiet -- and note the "
              "drift is monotonic,\n  so averaging over it does not help.")

    deltas = []
    print(f"\n{args.pairs} interleaved pairs:")
    for _ in range(args.pairs):
        a = measure(workload, off_env)
        b = measure(workload, on_env)
        d = 100.0 * (b - a) / a
        deltas.append(d)
        print(f"  off={a}  on={b}   {d:+.2f}%")

    lo, hi = min(deltas), max(deltas)
    print(f"\neffect: {lo:+.2f}% .. {hi:+.2f}%   (noise floor {fs:.3f}%)")

    bar = max(fs, MIN_EFFECT)
    if max(abs(lo), abs(hi)) < 1e-9:
        print("VERDICT: identical, to the instruction. Either the switch does "
              "nothing on this\n         workload, or -- check this FIRST -- "
              "this configuration cannot reach it.\n         Run with "
              "JAI_JIT_WHY=1 both ways and diff the refusal census.")
    elif abs(lo) <= bar and abs(hi) <= bar:
        print(f"VERDICT: no measurable effect -- both pairs are inside "
              f"{bar:.3f}% (the noise\n         floor, or the {MIN_EFFECT}% "
              f"below which nothing counts).")
    elif hi - lo > bar or lo * hi < 0:
        print("VERDICT: not separated from noise. The pairs disagree by more "
              "than the floor,\n         or they disagree in sign. This says "
              "nothing yet.")
    else:
        print(f"VERDICT: real, {'a regression' if lo > 0 else 'an improvement'}"
              f" (fewer interpreted instructions is better).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
