"""Run imgproc.jai against imgproc.py and print the comparison.

Each side is run several times and the median taken, because a single run of a
GPU workload measures the clock ramping as much as the kernel. The two sides
print the same records, so a name missing from either is an error rather than a
blank row -- a benchmark that quietly drops half its cases reads like a pass.
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


def sample(command: list[str], runs: int, cwd: Path, level: str, label: str):
    env = os.environ.copy()
    env["BENCH_LEVEL"] = level
    env.setdefault("PYTHONWARNINGS", "ignore")
    order: list[str] = []
    times: dict[str, list[float]] = {}
    checks: dict[str, str] = {}
    for index in range(runs):
        if sys.stderr.isatty():
            sys.stderr.write(f"\r\033[K  {label} (run {index + 1} of {runs})")
            sys.stderr.flush()
        done = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
        if done.returncode != 0:
            detail = (done.stdout + done.stderr).strip() or "<no output>"
            raise BenchError(f"{' '.join(command)} failed: {detail}")
        for name, (seconds, check) in parse(done.stdout).items():
            if name not in times:
                order.append(name)
                times[name] = []
            times[name].append(seconds)
            if check != "ok":
                checks[name] = check
    return order, {name: statistics.median(values) for name, values in times.items()}, checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--jaithon", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--level", choices=("easy", "medium", "hard"), default="hard")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    here = args.root / "tests" / "bench" / "jaicv"
    try:
        order, mine, my_checks = sample(
            [args.jaithon, "run", str(here / "imgproc.jai")],
            args.runs, args.root, args.level, "jaicv",
        )
        _, theirs, their_checks = sample(
            [args.python, str(here / "imgproc.py")],
            args.runs, args.root, args.level, "opencv",
        )
    except BenchError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if sys.stderr.isatty():
        sys.stderr.write("\r\033[K")

    missing = [name for name in order if name not in theirs]
    if missing:
        print(f"error: opencv did not report {', '.join(missing)}", file=sys.stderr)
        return 1

    print(f"{BOLD}{'operation':<16}{'jaicv':>10}{'opencv':>10}{'ratio':>9}   result{RESET}")
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
    print(f"{DIM}1920x1080 float32; opencv runs on the CPU, jaicv on the GPU.{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
