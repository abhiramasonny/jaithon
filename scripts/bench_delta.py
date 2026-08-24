"""Compare two benchmark runs and print what moved.

A night of agents changing kernels produces a lot of benchmark output, and the
interesting part of it is never the totals -- it is which single row went the
wrong way. Reading two tables side by side by eye misses that; this does not.

Usage:
    python3 scripts/bench_delta.py before.log after.log
    python3 scripts/bench_delta.py --save baseline.json after.log
    python3 scripts/bench_delta.py baseline.json after.log

Either argument may be a saved baseline (JSON) or the raw stdout of
`make bench`. Rows present in one run and not the other are reported rather
than dropped: a benchmark that quietly stops running a row reads like a pass.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# `name  1.23ms  4.56ms  3.71x  ok` with optional trailing route column, which
# is what both suites print. The units differ between rows (ms and s), so the
# suffix is captured and converted rather than assumed.
ROW = re.compile(
    r"^(?P<name>[A-Za-z][\w.-]*)\s+"
    r"(?P<ours>[\d.]+)(?P<ourUnit>ms|s|us)\s+"
    r"(?P<peer>[\d.]+)(?P<peerUnit>ms|s|us)\s+"
    r"(?:[\d.]+%\s+[\d.—-]+%?\s+)?"
    r"(?:[\d.]+\s+[\d.—-]+\s+[\d.—-]+\s+[\d.—-]+\s+)?"
    r"(?P<ratio>[\d.]+)x"
)

SCALE = {"us": 0.001, "ms": 1.0, "s": 1000.0}


def parse(text: str) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        found = ROW.match(line.strip())
        if not found:
            continue
        name = found.group("name")
        if name in ("operation", "benchmark", "total"):
            if name != "total":
                continue
        rows[name] = {
            "ours": float(found.group("ours")) * SCALE[found.group("ourUnit")],
            "peer": float(found.group("peer")) * SCALE[found.group("peerUnit")],
            "ratio": float(found.group("ratio")),
        }
    return rows


def load(path: Path) -> dict[str, dict[str, float]]:
    text = path.read_text()
    if path.suffix == ".json":
        return json.loads(text)
    rows = parse(text)
    if not rows:
        raise SystemExit(f"{path}: no benchmark rows found")
    return rows


def bar(change: float) -> str:
    """A change of ten per cent or more is worth an eye; less is noise here."""
    if change >= 0.10:
        return "+"
    if change <= -0.10:
        return "-"
    return " "


def main() -> int:
    argv = sys.argv[1:]
    save = None
    if argv and argv[0] == "--save":
        save = Path(argv[1])
        argv = argv[2:]
    if len(argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2

    before = load(Path(argv[0]))
    after = load(Path(argv[1]))
    if save is not None:
        save.write_text(json.dumps(after, indent=2, sort_keys=True) + "\n")

    names = [n for n in after if n != "total"]
    gone = [n for n in before if n not in after and n != "total"]
    fresh = [n for n in names if n not in before]

    print(f"{'row':<26}{'before':>9}{'after':>9}{'change':>9}   {'was':>7} {'now':>7}")
    print("-" * 72)
    moved = 0
    for name in sorted(names, key=lambda n: after[n]["ratio"]):
        now = after[name]
        if name not in before:
            print(f"{name:<26}{'—':>9}{now['ours']:>8.1f}m{'new':>9}   {'—':>7} {now['ratio']:>6.2f}x")
            continue
        was = before[name]
        change = (was["ours"] - now["ours"]) / was["ours"] if was["ours"] else 0.0
        mark = bar(change)
        if mark != " ":
            moved += 1
        print(
            f"{mark}{name:<25}{was['ours']:>8.1f}m{now['ours']:>8.1f}m"
            f"{change * 100:>8.0f}%   {was['ratio']:>6.2f}x {now['ratio']:>6.2f}x"
        )

    if gone:
        print(f"\nrows that stopped reporting: {', '.join(sorted(gone))}")
    if fresh:
        print(f"rows that appeared: {', '.join(sorted(fresh))}")

    losing = sorted((n for n in names if after[n]["ratio"] < 1.0), key=lambda n: after[n]["ratio"])
    print(f"\n{len(losing)} of {len(names)} rows still below parity")
    if losing:
        print("  " + ", ".join(f"{n} {after[n]['ratio']:.2f}x" for n in losing))
    if "total" in after and "total" in before:
        print(f"total: {before['total']['ratio']:.2f}x -> {after['total']['ratio']:.2f}x")
    print(f"{moved} rows moved by more than a tenth")
    return 0


if __name__ == "__main__":
    sys.exit(main())
