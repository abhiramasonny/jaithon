#!/usr/bin/env python3
#: Every module the seed carries must be able to import what it imports.
#:
#: boot/seed.manifest lists the modules compiled into the binary so the front
#: end can start before it can compile anything. The window is narrow and the
#: failure is total: a seeded module whose import is NOT itself seeded makes the
#: tree unable to compile at all, with
#:
#:     jaithon: internal error: no front end for `.../parse/state.jai`
#:
#: and, because `make reseed` needs a working compiler to build the next seed,
#: the tree cannot rebuild its way out -- boot/ has to be restored from the last
#: good commit and every __jaicache__ wiped by hand.
#:
#: That is not hypothetical. Splitting parser.jai into parse/ added eleven lines
#: here; a later merge overwrote the manifest with a copy from a tree that
#: predated the split, silently dropping all eleven. No git conflict, because
#: the file was copied rather than merged. This check is thirty lines and would
#: have caught it before the first reseed.
#:
#: `make seed-check` catches it too, but only by building a whole seed and
#: watching the compiler abort -- minutes, and a poisoned boot/ to clean up
#: afterwards. This is text, so it runs on every `make test`.

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "boot" / "seed.manifest"
LIB = ROOT / "lib"

#: `from a.b.c import x` and `import a.b.c`. The trailing part of a `from`
#: may be a module OR a name inside one, so a path that does not resolve to a
#: file is retried one segment shorter before it is reported.
FROM = re.compile(r"^\s*from\s+([A-Za-z_][\w.]*)\s+import\b", re.M)
IMPORT = re.compile(r"^\s*import\s+([A-Za-z_][\w.]*)", re.M)


def seeded_modules():
    out = []
    for line in MANIFEST.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def module_path(dotted):
    return LIB / (dotted.replace(".", "/") + ".jai")


def main():
    listed = seeded_modules()
    seeded = set(listed)
    problems = []

    for rel in listed:
        path = LIB / rel
        if not path.exists():
            problems.append(f"{rel} is in the manifest but does not exist.")
            continue
        text = path.read_text()
        for dotted in set(FROM.findall(text)) | set(IMPORT.findall(text)):
            target = module_path(dotted)
            if not target.exists():
                # `from a.b import c` where c is a name, not a module.
                parent = dotted.rsplit(".", 1)[0] if "." in dotted else None
                if parent is None:
                    continue
                target = module_path(parent)
                dotted = parent
                if not target.exists():
                    continue
            want = str(target.relative_to(LIB))
            if want not in seeded:
                problems.append(
                    f"{rel} imports {dotted}, which is NOT in the manifest. "
                    f"Add `{want}`, or the seed cannot bootstrap."
                )

    if problems:
        print(f"seed closure check FAILED ({len(problems)} problem(s)):")
        for p in sorted(set(problems)):
            print(f"  {p}")
        return 1

    print(f"seed closure ok: {len(listed)} seeded modules, every import seeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
