#!/usr/bin/env python3
#: Time a command in CPU cycles, not wall clock, so it can be believed on a
#: busy machine.
#:
#: macOS's `/usr/bin/time -l` reports `cycles elapsed` and `instructions
#: retired` for the child process alone. Neither counts another process's work,
#: so eight agents building in worktrees move them by a few percent where they
#: move wall clock by a factor of two or three (see ab.py's header for what wall
#: clock did under that load). Cycles still feel cache and memory-bandwidth
#: contention, so the floor is measured, not assumed: every A/B also runs the
#: SAME side twice and reports that spread next to the effect.
#:
#:     scripts/dev/cycles.py -- ./jaithon check --no-cache lib/jaithon
#:     scripts/dev/cycles.py -n 7 -- ./jaithon run tests/bench/alloc_churn/alloc_churn.jai
#:     scripts/dev/cycles.py --ab ./jaithon-base ./jaithon -- run x.jai
#:     scripts/dev/cycles.py --env JAITHON_FOO=0 JAITHON_FOO=1 -- ./jaithon run x.jai
#:
#: With --env the two settings share one binary and are interleaved A,B,A,B.
#:
#: With --ab the first word after `--` is replaced by each binary in turn; the
#: rest of the command is shared. Two builds from different sources have
#: different build fingerprints and `__jaicache__` is keyed on it, so
#: alternating them run by run makes EVERY sample recompile the stdlib (10x the
#: real runtime on json_parse, and a 1.86x change read as 1.16x). --ab therefore
#: runs BLOCKS -- A block, B block, A block -- and discards the first run of
#: each. Prefer an env switch in one binary whenever the change allows one.

import argparse
import os
import re
import statistics
import subprocess
import sys

FIELDS = {
    "cycles": re.compile(r"^\s*(\d+)\s+cycles elapsed", re.M),
    "instr": re.compile(r"^\s*(\d+)\s+instructions retired", re.M),
    "peak": re.compile(r"^\s*(\d+)\s+peak memory footprint", re.M),
}
REAL = re.compile(r"^\s*([\d.]+) real", re.M)


def run_once(cmd, env):
    full = dict(os.environ)
    full.update(env)
    p = subprocess.run(["/usr/bin/time", "-l"] + cmd, env=full,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                       text=True)
    err = p.stderr
    out = {}
    for k, rx in FIELDS.items():
        m = rx.search(err)
        out[k] = int(m.group(1)) if m else None
    m = REAL.search(err)
    out["real"] = float(m.group(1)) if m else None
    out["rc"] = p.returncode
    if out["cycles"] is None:
        sys.stderr.write(err[-2000:])
        raise SystemExit("cycles.py: /usr/bin/time -l printed no cycle count")
    return out


def summarise(rows, label):
    cyc = [r["cycles"] for r in rows]
    ins = [r["instr"] for r in rows]
    peak = [r["peak"] for r in rows if r["peak"]]
    rc = sorted({r["rc"] for r in rows})
    med = statistics.median(cyc)
    spread = (max(cyc) - min(cyc)) / med * 100 if med else 0.0
    print(f"{label:>10}: cycles median {med/1e6:10.2f}M  min {min(cyc)/1e6:10.2f}M"
          f"  spread {spread:5.2f}%  instr {statistics.median(ins)/1e6:10.2f}M"
          f"  peak {max(peak)/1e6 if peak else 0:7.1f}MB  rc {rc}")
    return med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=5, help="samples per side")
    ap.add_argument("--ab", nargs=2, metavar=("A", "B"),
                    help="two binaries; replaces the command's first word")
    ap.add_argument("--env", nargs=2, metavar=("A", "B"),
                    help="two KEY=VAL settings for one binary")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("no command after --")

    if not a.ab and not a.env:
        run_once(cmd, {})  # discard: cold cache
        rows = [run_once(cmd, {}) for _ in range(a.n)]
        summarise(rows, "run")
        return

    def side(which):
        if a.ab:
            return [a.ab[which]] + cmd[1:], {}
        k, _, v = a.env[which].partition("=")
        return cmd, {k: v}

    A, B, A2 = [], [], []
    if a.ab:
        for rows, which in ((A, 0), (B, 1), (A2, 0)):
            run_once(*side(which))  # discard: this block's cold cache
            rows.extend(run_once(*side(which)) for _ in range(a.n))
    else:
        run_once(*side(0))
        run_once(*side(1))
        for i in range(a.n):
            A.append(run_once(*side(0)))
            B.append(run_once(*side(1)))
            A2.append(run_once(*side(0)))
    ma = summarise(A, "A")
    mb = summarise(B, "B")
    ma2 = summarise(A2, "A again")
    floor = abs(ma2 - ma) / ma * 100
    eff = (mb - ma) / ma * 100
    print(f"B vs A: {eff:+.2f}% cycles (A vs A floor {floor:.2f}%)"
          f"  speedup {ma / mb:.3f}x")


if __name__ == "__main__":
    main()
