#!/usr/bin/env python3
"""What the JIT did to a program, and why -- in a page rather than a firehose.

`JAI_JIT_WHY=1` already reports every decision the tier makes, but it reports
them one line at a time, unsorted, tens of thousands of lines deep. Answering
"which function should I look at" from that meant a pipeline of grep, sed, sort
and uniq assembled by hand every time, and getting the pipeline subtly wrong is
how a census gets ranked by the wrong column.

The three rankings this prints are the ones that turned out to matter, and each
is here because ranking by the obvious alternative was measured and was wrong:

  * BY ATTEMPTS, not by unique sites. The tier retries a body as its model
    learns, so a refusal appearing eighty times is one it tried eighty times to
    get past. That is a hotness signal and it is free. `sort -u` throws it away,
    which is exactly what an earlier census did before ranking eighty stops at
    one loop below a single stop somewhere cold.

  * PARTIAL WALKS SEPARATELY from declines. They are different failures with
    different fixes. A declined body runs entirely interpreted. A partial body
    compiled a prefix and then deoptimises at one instruction every time it runs
    -- which is a WIN when that instruction is cold and a LOSS when it is hot,
    measured at 18% either way on the same change.

  * THE OPCODE, not just the reason string. Two unrelated causes used to share
    the line "OP_GET_LOCAL" and wanted different fixes.

Usage:
    scripts/jit_report.py run  tests/bench/fib_recursive/fib.jai
    scripts/jit_report.py check lib/jaithon/compile/parser.jai
    scripts/jit_report.py -- ./jaithon test tests/lang

Exit status is the program's own, so this can wrap a command in a script.
"""
from __future__ import annotations

import argparse
import collections
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BOLD = "\033[1m" if sys.stdout.isatty() else ""
DIM = "\033[2m" if sys.stdout.isatty() else ""
RESET = "\033[0m" if sys.stdout.isatty() else ""

#: `[jit] compiled <fn>  arity=...`
COMPILED = re.compile(r"^\[jit\] compiled (\S+)\s")
#: `[jit] osr <fn> at <n>: <n> instructions iter=<n>`
OSR_OK = re.compile(r"^\[jit\] osr (\S+) at (\d+): \d+ instructions")
#: `[jit] <fn> stopped: <reason>` and its `(measuring)` and `osr` variants.
STOPPED = re.compile(
    r"^\[jit\] (?:osr )?(\S+?)(?: at \d+)? stopped(?: \(measuring\))?: (.*)$"
)
#: `[jit] <fn> walked only to OP_X at <n> -- the rest ...`
PARTIAL = re.compile(
    r"^\[jit\] (?:osr )?(\S+?)(?: at \d+)? walked only to (OP_\w+) at (\d+)"
)
#: `[jit] <fn> declined at OP_X` -- the default case's own message.
DECLINED_AT = re.compile(r"^\[jit\] (\S+) declined at (OP_\w+)$")
#: `attrib <fn> <count> <pct>%` from --stats with JAI_JIT_ATTRIB=1.
ATTRIB = re.compile(r"^attrib (\S+)\s+(\d+)\s+([\d.]+)%$")
ATTRIB_TOTAL = re.compile(r"^attrib-total (\d+) of (\d+) \((\d+) unattributed\)$")

OPCODE = re.compile(r"^(OP_\w+)(?::\s*(.*))?$")

STATE = "(state, not an opcode)"


def fold(text: str) -> str:
    """Collapse the parts of a reason that name one site rather than one cause.

    A backticked name and a slot number make every site its own reason, which
    is right in the live output and wrong in a ranking -- eighty stops on the
    same cause would read as eighty separate one-off problems.
    """
    text = re.sub(r"`[^`]*`", "`X`", text)
    text = re.sub(r"\b(local|slot|offset) \d+", r"\1 N", text)
    return text


class Report:
    def __init__(self) -> None:
        self.compiled: set[str] = set()
        self.osr: set[str] = set()
        #: fn -> reason, last seen. A body is retried, and what matters is
        #: where it ended up, not every step it took to get there.
        self.stopped: dict[str, str] = {}
        self.partial: dict[str, tuple[str, str]] = {}
        self.attempts: collections.Counter = collections.Counter()
        self.opcode_attempts: collections.Counter = collections.Counter()
        self.partial_attempts: collections.Counter = collections.Counter()
        #: fn -> interpreted instructions executed in its frames. Exact when
        #: present: the VM asserts sum(attrib) == instructionCount.
        self.attrib: dict[str, int] = {}
        self.attrib_total = 0
        self.attrib_unattributed = 0
        self.lines = 0

    def feed(self, line: str) -> None:
        m = ATTRIB.match(line)
        if m:
            self.attrib[m.group(1)] = int(m.group(2))
            return
        m = ATTRIB_TOTAL.match(line)
        if m:
            self.attrib_total = int(m.group(2))
            self.attrib_unattributed = int(m.group(3))
            return
        if not line.startswith("[jit] "):
            return
        self.lines += 1

        m = COMPILED.match(line)
        if m:
            self.compiled.add(m.group(1))
            return
        m = OSR_OK.match(line)
        if m:
            self.osr.add(f"{m.group(1)} at {m.group(2)}")
            return
        m = PARTIAL.match(line)
        if m:
            fn, op, at = m.group(1), m.group(2), m.group(3)
            self.partial[fn] = (op, at)
            self.partial_attempts[op] += 1
            return
        m = DECLINED_AT.match(line)
        if m:
            fn, op = m.group(1), m.group(2)
            self.stopped[fn] = op
            self.attempts[op] += 1
            self.opcode_attempts[op] += 1
            return
        m = STOPPED.match(line)
        if m:
            fn, reason = m.group(1), fold(m.group(2))
            self.stopped[fn] = reason
            self.attempts[reason] += 1
            op = OPCODE.match(reason)
            #: A reason with no opcode prefix is a STATE refusal -- the tier
            #: knows the instruction and refuses on what it knows about the
            #: program instead ("callee's return kind not usable"). The
            #: distinction decides what kind of work would clear it: arming an
            #: opcode, or teaching the model something.
            self.opcode_attempts[op.group(1) if op else STATE] += 1
            return

    #: A body that stopped somewhere and ALSO reached the tier is a partial
    #: walk, not a decline; one that never reached it is a decline.
    def outcomes(self) -> tuple[set[str], set[str], set[str]]:
        reached = {name.split(" at ")[0] for name in self.osr} | self.compiled
        partial = set(self.partial)
        declined = {fn for fn in self.stopped if fn not in reached}
        return reached, partial, declined


def rule(width: int = 78) -> str:
    return "─" * width


def table(title: str, rows: list[tuple[int, str]], note: str, limit: int) -> None:
    if not rows:
        return
    print()
    print(f"{BOLD}{title}{RESET}")
    print(f"{DIM}{note}{RESET}")
    shown = rows[:limit]
    for count, label in shown:
        print(f"  {count:>6}  {label}")
    if len(rows) > limit:
        rest = sum(count for count, _ in rows[limit:])
        print(f"{DIM}  {rest:>6}  in {len(rows) - limit} further reasons"
              f" (--limit to see more){RESET}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarise what the JIT did to a program.")
    parser.add_argument("mode", nargs="?", default="run",
                        choices=("run", "check"),
                        help="how to invoke jaithon on FILE (default: run)")
    parser.add_argument("file", nargs="?", help="the program to report on")
    parser.add_argument("--limit", type=int, default=12,
                        help="rows per table (default 12)")
    parser.add_argument("--jaithon", default=str(ROOT / "jaithon"))

    #: Split on `--` before argparse sees it. REMAINDER after two optional
    #: positionals cannot do this: argparse tries the first word of the command
    #: as `mode` and rejects it.
    argv = sys.argv[1:]
    passthrough: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, passthrough = argv[:cut], argv[cut + 1:]
    args = parser.parse_args(argv)

    if passthrough:
        command = passthrough
    elif args.file:
        command = [args.jaithon, args.mode, "--stats"]
        if args.mode == "check":
            command.append("--no-cache")
        command.append(args.file)
    else:
        parser.error("give a FILE, or a command after --")

    env = os.environ.copy()
    env["JAI_JIT_WHY"] = "1"
    #: The attribution table is what turns "which refusal is printed most" into
    #: "where the interpreted work actually is", and those are different
    #: rankings. It costs a branch per instruction and only works alongside
    #: --stats, which is why both are set here and neither is on by default.
    env["JAI_JIT_ATTRIB"] = "1"
    done = subprocess.run(command, cwd=ROOT, env=env,
                          capture_output=True, text=True)

    report = Report()
    for line in done.stderr.splitlines():
        report.feed(line)
    #: --stats writes to stdout and JAI_JIT_WHY to stderr.
    for line in done.stdout.splitlines():
        report.feed(line)

    if report.lines == 0:
        print("no JIT activity: the program did not run anything hot enough to "
              "reach the tier, or it failed before it could.", file=sys.stderr)
        if done.returncode != 0:
            sys.stderr.write(done.stdout[-2000:])
            sys.stderr.write(done.stderr[-2000:])
        return done.returncode

    reached, partial, declined = report.outcomes()

    print(rule())
    print(f"{BOLD}jaithon JIT report{RESET}  {DIM}{' '.join(command)}{RESET}")
    print(rule())
    print(f"  {len(report.compiled):>6}  function bodies compiled")
    print(f"  {len(report.osr):>6}  loops compiled (OSR)")
    print(f"  {len(partial):>6}  bodies compiled only up to a point")
    print(f"  {len(declined):>6}  bodies not compiled at all")

    if report.attrib:
        print()
        print(f"{BOLD}where the interpreted work is{RESET}")
        print(f"{DIM}exact, not sampled -- the VM asserts these sum to the "
              f"instruction count\n({report.attrib_unattributed} unattributed). "
              f"This is the ranking that matters: a refusal\nprinted a hundred "
              f"times in code that runs once is worth nothing.{RESET}")
        ranked = sorted(report.attrib.items(), key=lambda kv: -kv[1])
        for fn, count in ranked[:args.limit]:
            share = (100.0 * count / report.attrib_total
                     if report.attrib_total else 0.0)
            if fn in report.partial:
                op, at = report.partial[fn]
                verdict = f"partial, stops at {op}@{at}"
            elif fn in reached:
                verdict = "compiled"
            elif fn in report.stopped:
                verdict = report.stopped[fn]
            else:
                verdict = "never considered"
            print(f"  {count:>12}  {share:>5.1f}%  {fn:<28} {verdict}")
        if len(ranked) > args.limit:
            rest = sum(c for _, c in ranked[args.limit:])
            print(f"{DIM}  {rest:>12}  {100.0 * rest / report.attrib_total:>5.1f}%"
                  f"  in {len(ranked) - args.limit} further functions{RESET}")

    table(
        "why a body was not compiled",
        [(n, r) for r, n in report.attempts.most_common()],
        "by ATTEMPTS, not by unique sites: the tier retries a body as its model "
        "learns,\nso a reason counted eighty times is one it tried eighty times "
        "to get past.",
        args.limit,
    )
    table(
        "...and which opcode that was",
        [(n, o) for o, n in report.opcode_attempts.most_common()],
        "the same attempts grouped by opcode, because one opcode can refuse for "
        "several\nunrelated reasons and they want different fixes. "
        f"{STATE} means the tier knows the\ninstruction and refuses on what it "
        "knows about the PROGRAM -- arming an opcode\nwould not help those; "
        "teaching the model something would.",
        args.limit,
    )
    table(
        "where a walk stopped early",
        [(n, o) for o, n in report.partial_attempts.most_common()],
        "these bodies DID compile, up to this opcode, and interpret from there "
        "on.\nThat is a win when the instruction is cold and a loss when it is "
        "hot.",
        args.limit,
    )

    if partial:
        print()
        print(f"{BOLD}the bodies that stop early{RESET}")
        for fn in sorted(partial)[:args.limit]:
            op, at = report.partial[fn]
            print(f"  {fn:<34} {op} at {at}")
        if len(partial) > args.limit:
            print(f"{DIM}  and {len(partial) - args.limit} more{RESET}")

    if not (report.compiled or report.osr or partial or declined):
        print()
        print("  nothing reached the tier. A function compiles after "
              f"{BOLD}JAI_JIT_THRESHOLD{RESET} entries and a")
        print("  loop only when the OSR sampler lands on it, so a short program "
              "can finish first.")

    print()
    print(f"{DIM}JAI_JIT_WHY=1 for the raw decisions this summarises.{RESET}")
    return done.returncode


if __name__ == "__main__":
    raise SystemExit(main())
