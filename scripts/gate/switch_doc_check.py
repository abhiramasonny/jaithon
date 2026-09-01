#!/usr/bin/env python3
#: Every environment switch src/vm/jit reads must appear in that directory's
#: README table, and every switch the table lists must still exist in the code.
#:
#: Why a gate and not a convention: a switch is how this tier is measured. A
#: change with no switch is a change with no number you can trust, because the
#: only measurement it accepts is an A/B in ONE binary with the two forms
#: interleaved. Fifty-odd of them accumulated before anything wrote them down,
#: and by then the only way to find one was to grep -- which meant new work
#: quietly re-derived switches that already existed, and retired ones stayed in
#: the docs forever.
#:
#: What this does NOT check: that the description is accurate, or that the
#: stated default matches the code. Both are prose. The check is presence in
#: both directions, which is the part that rots silently.

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "vm" / "jit"
README = SRC / "README.md"

GETENV = re.compile(r'getenv\("([A-Z][A-Z0-9_]*)"\)')
#: Only rows whose first cell is a backticked switch name, and only inside the
#: switch section -- the file is full of other tables whose first column is a
#: backticked SLOT_ kind or a JAI_JIT_ outcome, and a looser pattern demanded
#: getenv calls for all of them.
ROW = re.compile(r"^\|\s*`((?:JAITHON|JAI)_[A-Z0-9_]*)`\s*\|")
SECTION = "## Every switch this tier reads"

#: Switches the README documents that are read OUTSIDE src/vm/jit, because the
#: behaviour they change is one a reader of this tier needs to know about. Each
#: is verified to still exist where it says, so the exemption cannot go stale.
ELSEWHERE = {
    "JAITHON_MEGA_STRESS": "src/vm/vm_cache.c",
}


def switches_in_source():
    found = {}
    for c in sorted(SRC.glob("*.c")) + sorted(SRC.glob("*.h")):
        for name in GETENV.findall(c.read_text()):
            found.setdefault(name, []).append(c.name)
    for name, rel in ELSEWHERE.items():
        path = ROOT / rel
        if path.exists() and f'getenv("{name}")' in path.read_text():
            found.setdefault(name, []).append(rel)
    return found


def switches_in_readme():
    listed = {}
    inside = False
    for n, line in enumerate(README.read_text().splitlines(), 1):
        if line.startswith("## "):
            inside = line.strip() == SECTION
            continue
        if not inside:
            continue
        m = ROW.match(line)
        if m:
            listed.setdefault(m.group(1), n)
    return listed


def main():
    source = switches_in_source()
    listed = switches_in_readme()

    problems = []
    for name in sorted(source):
        if name not in listed:
            where = ", ".join(sorted(set(source[name])))
            problems.append(
                f"{name} is read in {where} but is not in {README.name}. "
                f"Add a row: what it does, and what its default is."
            )
    for name in sorted(listed):
        if name not in source:
            problems.append(
                f"{name} has a row in {README.name} at line {listed[name]} but "
                f"nothing in src/vm/jit reads it any more. Delete the row."
            )

    if problems:
        print(f"switch doc check FAILED ({len(problems)} problem(s)):")
        for p in problems:
            print(f"  {p}")
        return 1

    print(
        f"switch doc check ok: {len(source)} switches read in src/vm/jit, "
        f"all documented"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
