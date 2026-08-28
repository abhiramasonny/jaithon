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


# The route field is "<intent>/<what actually ran>". Only the second half is a
# fact; the first is what the benchmark asked for, and on this hardware the two
# differ often enough to matter. A route naming a 16-bit element ran at f16, and
# anything else at f32. No slash means the benchmark has one precision and the
# question does not arise -- do not guess from the intent label.
def precision_of(route: str) -> str:
    if "/" not in route:
        return ""
    tail = route.rsplit("/", 1)[-1]
    return "f16" if "16" in tail else "f32"



#: How loaded the machine is, and by what. A ratio taken on a contended machine
#: is not a measurement, and the failure is silent: the same two jaitensor rows
#: spread 3.9% and 2.4% when sampled inside one quiet process and 145% and 47%
#: with eight competing processes and nothing else changed. `mediaanalysisd` is
#: called out by name because it is the repeat offender on this hardware -- it
#: takes the GPU for photo analysis and has swung a suite from 43s to 780s.
LOAD_CEILING = float(os.environ.get("BENCH_LOAD_CEILING", "2.5"))


def machine_load() -> tuple[float, float]:
    """(load average over 1 minute, total %CPU held by mediaanalysisd)."""
    try:
        load = os.getloadavg()[0]
    except OSError:
        return (0.0, 0.0)
    hog = 0.0
    try:
        done = subprocess.run(["ps", "-Ao", "pcpu,comm"], capture_output=True,
                              text=True, timeout=10)
        for line in done.stdout.splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2 and "mediaanalysisd" in parts[1]:
                hog += float(parts[0])
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return (load, hog)


def preflight() -> str | None:
    """A refusal message when the machine is too busy to measure on, else None.

    Refusing is the point. The commonest way this suite produces a wrong number
    is not a bug in it -- it is being run while something else has the machine,
    and then the number is quoted. `BENCH_ANY_LOAD=1` is the escape hatch, the
    same shape `run_bench.sh` already uses to refuse a debug build.
    """
    if os.environ.get("BENCH_ANY_LOAD") == "1":
        return None
    load, hog = machine_load()
    if load <= LOAD_CEILING and hog < 40.0:
        return None
    detail = f"load average {load:.1f}"
    if hog >= 40.0:
        detail += f", mediaanalysisd at {hog:.0f}% CPU"
    return (f"the machine is too busy to measure on ({detail}; the ceiling is "
            f"{LOAD_CEILING:.1f}).\n"
            f"       Wait for it to settle, or set BENCH_ANY_LOAD=1 to run "
            f"anyway and have the table say so.")


def binary_id(root: Path, command: list[str]) -> str | None:
    """A hash of the binary a command runs, so a rebuild mid-run is visible.

    It happened four times in one night here, and every sample taken across the
    change is silently comparing two different programs.
    """
    for token in command:
        candidate = (root / token).resolve()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            try:
                with open(candidate, "rb") as handle:
                    import hashlib
                    return hashlib.sha256(handle.read()).hexdigest()[:12]
            except OSError:
                return None
    return None


def parse(output: str) -> dict[str, tuple[float, str, str]]:
    found: dict[str, tuple[float, str, str]] = {}
    for line in output.splitlines():
        if not line.startswith("jtb\t"):
            continue
        parts = line.split("\t")
        if len(parts) < 6:
            raise BenchError(f"malformed record: {line!r}")
        route = parts[6] if len(parts) > 6 else ""
        found[parts[1]] = (float(parts[2]), parts[5], route)
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
            cwd: Path, level: str, load_seen: list | None = None):
    """Time every row on both sides, alternating, one row a process."""
    ours: dict[str, list[float]] = {name: [] for name in rows}
    peer: dict[str, list[float]] = {name: [] for name in rows}
    checks: dict[str, str] = {}
    ours_route: dict[str, str] = {}
    peer_route: dict[str, str] = {}
    total = runs * len(rows)
    done_count = 0
    for index in range(runs):
        for name in rows:
            done_count += 1
            if sys.stderr.isatty():
                sys.stderr.write(f"\r\033[K  {name} ({done_count} of {total})")
                sys.stderr.flush()
            if load_seen is not None:
                load_seen.append(machine_load())
            for side, command, into in ((True, mine, ours), (False, theirs, peer)):
                found = one(command + [name], cwd, level)
                if name not in found:
                    raise BenchError(f"{'jaicv' if side else 'opencv'} did not report {name}")
                seconds, check, route = found[name]
                into[name].append(seconds)
                if check != "ok":
                    checks[name] = check
                (ours_route if side else peer_route)[name] = precision_of(route)
    if sys.stderr.isatty():
        sys.stderr.write("\r\033[K")
    #: A row whose two sides ran at different precisions is not a like-for-like
    #: ratio, and reading it as one has cost real time: jaitensor's tuner picks
    #: fp32 for most GEMM shapes while the peer's autocast is pinned to fp16, so
    #: three rows read as narrow losses when they were in fact 200x more
    #: accurate answers -- 1.4e-5 against 2.7e-3 relative to a float64
    #: reference. Nothing in the harness could see it, because nothing compared
    #: the two sides.
    for name in rows:
        mine_p, peer_p = ours_route.get(name, ""), peer_route.get(name, "")
        if mine_p and peer_p and mine_p != peer_p and name not in checks:
            checks[name] = f"{mine_p} vs {peer_p}"
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
        refusal = preflight()
        if refusal is not None:
            print(f"error: {refusal}", file=sys.stderr)
            return 1
        before_id = binary_id(args.root, ours_command)
        load_seen = [machine_load()]
        mine, theirs, checks = collect(
            ours_command, peer_command, order, args.runs, args.root, args.level,
            load_seen,
        )
        load_seen.append(machine_load())
        after_id = binary_id(args.root, ours_command)
        if before_id is not None and after_id is not None and before_id != after_id:
            print(f"error: the binary changed mid-run ({before_id} -> "
                  f"{after_id}); every sample straddling that is comparing two "
                  f"different programs. Re-run.", file=sys.stderr)
            return 1
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
    #: The table carries the conditions it was taken under, so it cannot be
    #: quoted out of them. A ratio without the load is not a measurement.
    loads = [entry[0] for entry in load_seen]
    hogs = [entry[1] for entry in load_seen]
    verdict = ("CONTENDED -- treat these numbers as indicative only"
               if max(loads) > LOAD_CEILING or max(hogs) >= 40.0 else "quiet")
    print(f"{DIM}load {min(loads):.1f}-{max(loads):.1f} during the run, "
          f"mediaanalysisd peak {max(hogs):.0f}% CPU; {verdict}{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
