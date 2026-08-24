"""Run one benchmark suite against its peer library and print the comparison.

Shared by the suites whose rows are a single time each -- `jaicv` against
OpenCV and `jainum` against numpy. `jaitensor` keeps its own runner, because
its rows carry a FLOP count and print a GFLOPS column this one has no reading
for. The suite, the two files and the two column headings are arguments so
that adding a peer is a call rather than a copy of this file.

Each side is run several times and the median taken, because a single run of a
GPU workload measures the clock ramping as much as the kernel. The two sides
print the same records, so a name missing from either is an error rather than a
blank row -- a benchmark that quietly drops half its cases reads like a pass.

One row a process, and the two sides turn about. Running our whole side and
then the peer's is not a fair comparison on a laptop: by the time the later
rows come up, our side has held the GPU flat out for a minute, while the peer
runs on the CPU and reaches the same rows on a cool device. good-features read
1.32x when it was the last row of twenty-one and 0.75x when nine more rows were
added in front of it, with nothing about the code changed in between. Isolated,
it reads the same to within two per cent every time. The extra process starts
cost about a minute across the suite, which is the price of the numbers meaning
anything.
"""
from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
from pathlib import Path

BOLD = "\033[1m" if sys.stdout.isatty() else ""
DIM = "\033[2m" if sys.stdout.isatty() else ""
RESET = "\033[0m" if sys.stdout.isatty() else ""


class BenchError(RuntimeError):
    pass


def parse(output: str) -> dict[str, tuple[float, str]]:
    found: dict[str, tuple[float, str]] = {}
    for line in output.splitlines():
        if not line.startswith("jtb\t"):
            continue
        parts = line.split("\t")
        if len(parts) < 6:
            raise BenchError(f"malformed record: {line!r}")
        found[parts[1]] = (float(parts[2]), parts[5])
    if not found:
        raise BenchError("no records")
    return found


def one(command: list[str], cwd: Path, level: str) -> dict[str, tuple[float, str]]:
    env = os.environ.copy()
    env["BENCH_LEVEL"] = level
    env.setdefault("PYTHONWARNINGS", "ignore")
    done = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    if done.returncode != 0:
        detail = (done.stdout + done.stderr).strip() or "<no output>"
        raise BenchError(f"{' '.join(command)} failed: {detail}")
    return parse(done.stdout)


def listing(command: list[str], cwd: Path, level: str) -> list[str]:
    env = os.environ.copy()
    env["BENCH_LEVEL"] = level
    env.setdefault("PYTHONWARNINGS", "ignore")
    done = subprocess.run(command + ["list"], cwd=cwd, env=env, capture_output=True, text=True)
    if done.returncode != 0:
        detail = (done.stdout + done.stderr).strip() or "<no output>"
        raise BenchError(f"listing failed: {detail}")
    return [line.strip() for line in done.stdout.splitlines() if line.strip()]


def collect(mine: list[str], theirs: list[str], rows: list[str], runs: int,
            cwd: Path, level: str):
    """Time every row on both sides, alternating, one row a process."""
    ours: dict[str, list[float]] = {name: [] for name in rows}
    peer: dict[str, list[float]] = {name: [] for name in rows}
    checks: dict[str, str] = {}
    total = runs * len(rows)
    done_count = 0
    for index in range(runs):
        for name in rows:
            done_count += 1
            if sys.stderr.isatty():
                sys.stderr.write(f"\r\033[K  {name} ({done_count} of {total})")
                sys.stderr.flush()
            for side, command, into in ((True, mine, ours), (False, theirs, peer)):
                found = one(command + [name], cwd, level)
                if name not in found:
                    raise BenchError(f"{'jaicv' if side else 'opencv'} did not report {name}")
                seconds, check = found[name]
                into[name].append(seconds)
                if check != "ok":
                    checks[name] = check
    if sys.stderr.isatty():
        sys.stderr.write("\r\033[K")
    return (
        {name: statistics.median(values) for name, values in ours.items()},
        {name: statistics.median(values) for name, values in peer.items()},
        checks,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--jaithon", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--level", choices=("easy", "medium", "hard"), default="hard")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--suite", default="jaicv")
    parser.add_argument("--ours-file", default="imgproc.jai")
    parser.add_argument("--peer-file", default="imgproc.py")
    parser.add_argument("--ours-label", default="jaicv")
    parser.add_argument("--peer-label", default="opencv")
    parser.add_argument(
        "--footer",
        default="1920x1080 float32; opencv runs on the CPU, jaicv on the GPU.",
    )
    args = parser.parse_args()

    here = args.root / "tests" / "bench" / args.suite
    ours_command = [args.jaithon, "run", str(here / args.ours_file)]
    peer_command = [args.python, str(here / args.peer_file)]
    try:
        order = listing(ours_command, args.root, args.level)
        peer_rows = listing(peer_command, args.root, args.level)
        missing = [name for name in order if name not in peer_rows]
        if missing:
            print(
                f"error: {args.peer_label} has no row for {', '.join(missing)}",
                file=sys.stderr,
            )
            return 1
        mine, theirs, checks = collect(
            ours_command, peer_command, order, args.runs, args.root, args.level
        )
    except BenchError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    my_checks = checks
    their_checks: dict[str, str] = {}

    print(
        f"{BOLD}{'operation':<16}{args.ours_label:>10}{args.peer_label:>10}"
        f"{'ratio':>9}   result{RESET}"
    )
    print("─" * 54)
    total_mine = 0.0
    total_theirs = 0.0
    for name in order:
        ours = mine[name]
        peer = theirs[name]
        total_mine += ours
        total_theirs += peer
        note = my_checks.get(name) or their_checks.get(name) or "ok"
        print(
            f"{name:<16}{ours * 1000:>9.1f}ms{peer * 1000:>9.1f}ms"
            f"{peer / ours:>8.2f}x   {note}"
        )
    print("─" * 54)
    print(
        f"{BOLD}{'total':<16}{total_mine * 1000:>9.1f}ms{total_theirs * 1000:>9.1f}ms"
        f"{total_theirs / total_mine:>8.2f}x{RESET}"
    )
    print(f"{DIM}{args.footer}{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
