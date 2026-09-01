#!/usr/bin/env python3
#: The .jaic wire format is described twice, in two languages, and nothing made
#: the two descriptions agree.
#:
#: `src/vm/bytecode/serialize.h` holds the C reader/writer's constants; the
#: self-hosted front end holds its own copy in
#: `lib/jaithon/compile/jaic/format.jai`. Both read and write the same files.
#:
#: A disagreement here is not a crash. Bumping the version on one side only
#: means every cached image the other side wrote is silently accepted or
#: silently rejected -- and __jaicache__ is consulted before anything else, so
#: the symptom is a stale image being run, or a full recompile blamed on the
#: cache being "cold". The repo has already learned that a poisoned cache
#: outlives the fix that caused it.
#:
#: The check is pure text and one direction only by design: every `JAIC_*`
#: constant in the header must have a twin in the .jai file with the same
#: value. Extra constants on the .jai side are allowed -- COMPILER_VERSION has
#: no C counterpart -- because the header is the format's authority.

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "src" / "vm" / "bytecode" / "serialize.h"
JAI = ROOT / "lib" / "jaithon" / "compile" / "jaic" / "format.jai"

C_INT = re.compile(r"^#define\s+JAIC_([A-Z0-9_]+)\s+(\d+)\s*$", re.M)
C_STR = re.compile(r'^#define\s+JAIC_([A-Z0-9_]+)\s+"([^"]*)"\s*$', re.M)
J_INT = re.compile(r"^pub\s+const\s+([A-Z0-9_]+)\s*:\s*int\s*=\s*(\d+)\s*$", re.M)
J_BYTES = re.compile(
    r"^pub\s+let\s+([A-Z0-9_]+)\s*:\s*list\[int\]\s*=\s*\[([^\]]*)\]\s*$", re.M)


def main():
    if not HEADER.exists() or not JAI.exists():
        print(f"jaic wire check: missing {HEADER if not HEADER.exists() else JAI}")
        return 1

    h, j = HEADER.read_text(), JAI.read_text()
    c_ints = {k: int(v) for k, v in C_INT.findall(h)}
    c_strs = dict(C_STR.findall(h))
    j_ints = {k: int(v) for k, v in J_INT.findall(j)}
    j_bytes = {}
    for name, body in J_BYTES.findall(j):
        try:
            j_bytes[name] = [int(x.strip(), 0) for x in body.split(",") if x.strip()]
        except ValueError:
            pass

    problems = []
    for name, want in sorted(c_ints.items()):
        if name not in j_ints:
            problems.append(
                f"JAIC_{name} = {want} in {HEADER.name} has no twin in "
                f"{JAI.name}. Add `pub const {name}: int = {want}`.")
        elif j_ints[name] != want:
            problems.append(
                f"JAIC_{name} is {want} in {HEADER.name} but {j_ints[name]} in "
                f"{JAI.name}. The two sides read and write the same files; "
                f"bump both or neither.")

    for name, text in sorted(c_strs.items()):
        want = [ord(c) for c in text]
        if name not in j_bytes:
            problems.append(
                f'JAIC_{name} = "{text}" in {HEADER.name} has no twin in '
                f"{JAI.name}. Add `pub let {name}: list[int] = "
                f"[{', '.join(hex(b) for b in want)}]`.")
        elif j_bytes[name] != want:
            problems.append(
                f'JAIC_{name} is "{text}" ({want}) in {HEADER.name} but '
                f"{j_bytes[name]} in {JAI.name}.")

    if problems:
        print(f"jaic wire check FAILED ({len(problems)} problem(s)):")
        for p in problems:
            print(f"  {p}")
        return 1

    print(f"jaic wire check ok: {len(c_ints) + len(c_strs)} constants agree "
          f"across serialize.h and format.jai")
    return 0


if __name__ == "__main__":
    sys.exit(main())
