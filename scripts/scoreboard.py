"""One view of every benchmark row that is still slower than its peer.

Three suites answer to three different peers -- the language benchmarks to
python3, C++ and Java, jaicv to OpenCV, jaitensor to PyTorch -- and each prints
its own table. The question anyone actually has is "what are we still losing
at", and answering it meant reading three tables and doing the filtering by
eye, which is how a row sits below parity for a month without anyone noticing.

Usage:
    python3 scripts/scoreboard.py lang.log cv.log tensor.log
    python3 scripts/scoreboard.py --all logs/*.log

Each argument is the raw stdout of a `make bench` run; the suite each one came
from is worked out from its contents rather than from its filename.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_delta import parse  # noqa: E402


def suite_of(text: str) -> str:
    if "opencv runs on the CPU" in text or "jaicv" in text.split("\n")[0:3][0:1]:
        return "jaicv"
    if "j GFLOPS" in text or "jaitensor" in text:
        return "jaitensor"
    return "language"


def main() -> int:
    paths = [Path(a) for a in sys.argv[1:] if a != "--all"]
    if not paths:
        print(__doc__.strip(), file=sys.stderr)
        return 2

    suites: dict[str, dict[str, dict[str, float]]] = {}
    for path in paths:
        if not path.is_file():
            print(f"skipping {path}: not a file", file=sys.stderr)
            continue
        text = path.read_text()
        rows = parse(text)
        if not rows:
            print(f"skipping {path}: no benchmark rows", file=sys.stderr)
            continue
        suites[suite_of(text)] = rows

    if not suites:
        print("nothing to report", file=sys.stderr)
        return 1

    below: list[tuple[str, str, float, float, float]] = []
    print(f"{'suite':<11}{'row':<26}{'ours':>10}{'peer':>10}{'ratio':>8}")
    print("-" * 65)
    for suite in sorted(suites):
        rows = suites[suite]
        losing = sorted(
            ((n, r) for n, r in rows.items() if n != "total" and r["ratio"] < 1.0),
            key=lambda item: item[1]["ratio"],
        )
        for name, row in losing:
            below.append((suite, name, row["ours"], row["peer"], row["ratio"]))
            print(
                f"{suite:<11}{name:<26}{row['ours']:>9.1f}m{row['peer']:>9.1f}m"
                f"{row['ratio']:>7.2f}x"
            )

    print("-" * 65)
    for suite in sorted(suites):
        rows = suites[suite]
        counted = [n for n in rows if n != "total"]
        losing = [n for n in counted if rows[n]["ratio"] < 1.0]
        total = rows.get("total")
        overall = f"{total['ratio']:.2f}x overall" if total else "no total reported"
        worst = min((rows[n]["ratio"] for n in counted), default=0.0)
        print(
            f"{suite:<11}{len(counted) - len(losing)} of {len(counted)} rows at or above "
            f"parity, worst {worst:.2f}x, {overall}"
        )

    if not below:
        print("\nnothing is below parity.")
        return 0
    print(f"\n{len(below)} rows below parity, worst first:")
    for suite, name, _ours, _peer, ratio in sorted(below, key=lambda item: item[4]):
        print(f"  {ratio:>5.2f}x  {suite}/{name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
