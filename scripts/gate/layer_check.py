#!/usr/bin/env python3
"""Every #include between src/ directories must be declared in src/layers.manifest.

The tree is layered -- common under vm under runtime under native under cli --
but nothing enforced it. The Makefile globs every .c into one binary, so any
file could include any other and the only thing standing between the tree and a
tangle was whoever happened to read the diff. rustc gets this for free: one
crate per phase, dependencies declared in Cargo.toml, and a layering violation
is a build error. This is that, for a C project.

It pins the graph rather than deriving it from a rule. Some edges here are
genuine cycles -- vm/, vm/object/ and vm/bytecode/ are siblings, not a stack,
because chunk.h needs value.h while object.c needs gc.h -- and a rule that
forbade them would be wrong. What the manifest is for is making a NEW edge a
deliberate act: adding one means writing it down, which is exactly the review
this had none of.

    scripts/gate/layer_check.py            check, the make target's mode
    scripts/gate/layer_check.py --write    rewrite the manifest from the tree
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
MANIFEST = ROOT / "src" / "layers.manifest"

# Longest first, so src/vm/jit wins over src/vm.
DIRS = [
    "common",
    "vm/bytecode",
    "vm/jit",
    "vm/object",
    "vm/trace",
    "vm",
    "runtime/builtins",
    "runtime/modules",
    "runtime",
    "native/apple",
    "native/posix",
    "native",
    "cli/commands",
    "cli",
]


def layer_of(rel):
    """Which layer a path under src/ belongs to, or None."""
    for d in DIRS:
        if rel.startswith(d + "/"):
            return d
    return None


def edges():
    """Every cross-layer include edge in the tree, as (from, to, example)."""
    found = {}
    for path in sorted(ROOT.joinpath("src").rglob("*")):
        if path.suffix not in (".c", ".h", ".m"):
            continue
        rel = path.relative_to(ROOT / "src").as_posix()
        src = layer_of(rel)
        if src is None:
            continue
        for m in re.finditer(r'#\s*include\s+"([^"]+)"', path.read_text(errors="ignore")):
            dst = layer_of(m.group(1))
            if dst is None or dst == src:
                continue
            found.setdefault((src, dst), f"src/{rel}: {m.group(1)}")
    return found


def read_manifest():
    if not MANIFEST.exists():
        return None
    declared = set()
    for line in MANIFEST.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        src, _, dst = line.partition("->")
        declared.add((src.strip(), dst.strip()))
    return declared


def write_manifest(found):
    lines = [
        "# Declared include edges between src/ directories.",
        "# Checked by scripts/gate/layer_check.py, which explains what this is for.",
        "# Regenerate with: scripts/gate/layer_check.py --write",
        "",
    ]
    for src, dst in sorted(found):
        lines.append(f"{src} -> {dst}")
    MANIFEST.write_text("\n".join(lines) + "\n")


def main():
    found = edges()
    if "--write" in sys.argv:
        write_manifest(found)
        print(f"wrote {MANIFEST.relative_to(ROOT)}: {len(found)} edges")
        return 0

    declared = read_manifest()
    if declared is None:
        print(f"FAIL: no {MANIFEST.relative_to(ROOT)}; run with --write", file=sys.stderr)
        return 1

    added = sorted(set(found) - declared)
    gone = sorted(declared - set(found))
    for src, dst in added:
        print(f"undeclared include edge: {src} -> {dst}\n    {found[(src, dst)]}", file=sys.stderr)
    for src, dst in gone:
        print(f"declared edge no longer exists: {src} -> {dst}", file=sys.stderr)
    if added or gone:
        print(
            f"\n{len(added)} new, {len(gone)} stale. If the change is intended, re-run with "
            f"--write and commit the manifest.",
            file=sys.stderr,
        )
        return 1

    print(f"layers ok, {len(found)} declared include edges")
    return 0


if __name__ == "__main__":
    sys.exit(main())
