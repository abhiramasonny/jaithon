#!/usr/bin/env python3
"""Every dependency between directories must be declared in src/layers.manifest.

Two trees, one rule. In src/ the edges are `#include "..."`; in lib/ they are
`import`/`from`. Neither had anything enforcing them: the Makefile globs every
.c into one binary, and the module loader finds .jai files on disk, so any file
could depend on any other and the only thing between the tree and a tangle was
whoever happened to read the diff. rustc gets this for free -- one crate per
phase, dependencies declared in Cargo.toml, a layering violation is a build
error. This is that, for a tree that has no such build system.

It pins the graph rather than deriving it from a rule, because some edges here
are genuine cycles and a rule forbidding them would be wrong: vm/, vm/object/
and vm/bytecode/ are siblings, not a stack -- chunk.h needs value.h while
object.c needs gc.h -- and jaithon/compile/ and its check/ and opt/ subtrees
import each other. What the manifest is for is making a NEW edge a deliberate
act. The one that matters most is the edge that is NOT here: nothing under
std/ imports jaithon/, so the standard library does not depend on the compiler.

    scripts/gate/layer_check.py            check, the make target's mode
    scripts/gate/layer_check.py --write    rewrite the manifest from the tree
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
MANIFEST = ROOT / "src" / "layers.manifest"

# Longest first, so src/vm/jit wins over src/vm.
SRC_DIRS = [
    "common",
    "vm/bytecode", "vm/jit", "vm/object", "vm/trace", "vm",
    "runtime/builtins", "runtime/modules", "runtime",
    "native/apple", "native/posix", "native",
    "cli/commands", "cli",
]

LIB_DIRS = [
    "jaithon/compile/check", "jaithon/compile/opt", "jaithon/compile",
    "jaithon/tool", "jaithon",
    "std/algo", "std/ds", "std/iter", "std/num", "std/gui", "std",
]


def layer_of(rel, dirs):
    for d in dirs:
        if rel.startswith(d + "/"):
            return d
    return None


def scan(tree, dirs, suffixes, pattern, to_path):
    """Cross-layer edges in one tree, as {(from, to): "an example site"}."""
    found = {}
    for path in sorted(ROOT.joinpath(tree).rglob("*")):
        if path.suffix not in suffixes or "__jaicache__" in path.parts:
            continue
        rel = path.relative_to(ROOT / tree).as_posix()
        src = layer_of(rel, dirs)
        if src is None:
            continue
        for m in re.finditer(pattern, path.read_text(errors="ignore"), re.M):
            target = to_path(m.group(1))
            dst = layer_of(target, dirs) or layer_of(target + "/", dirs)
            if dst is None or dst == src:
                continue
            found.setdefault((f"{tree}/{src}", f"{tree}/{dst}"),
                             f"{tree}/{rel}: {m.group(1)}")
    return found


def edges():
    found = scan("src", SRC_DIRS, (".c", ".h", ".m"),
                 r'#\s*include\s+"([^"]+)"', lambda s: s)
    found.update(scan("lib", LIB_DIRS, (".jai",),
                      r'^\s*(?:from|import)\s+([\w.]+)',
                      lambda s: s.replace(".", "/")))
    return found


def internal_leaks():
    """A *_internal.h included from outside the directory that owns it.

    The pair is the tier boundary: jit.h is what the rest of the VM may see,
    jit_internal.h is the shared state jit_*.c pass around. Seven directories
    use the convention and all seven keep it; this is what stops the eighth
    reader from being the one that quietly does not.

    A subdirectory of the owner counts as inside it: src/cli/commands/ is part
    of the CLI and reads src/cli/cli_internal.h.
    """
    owner = {p.name: p.parent.relative_to(ROOT).as_posix()
             for p in ROOT.joinpath("src").rglob("*_internal.h")}
    leaks = []
    for path in sorted(ROOT.joinpath("src").rglob("*")):
        if path.suffix not in (".c", ".h", ".m"):
            continue
        here = path.parent.relative_to(ROOT).as_posix()
        for m in re.finditer(r'#\s*include\s+"([^"]+)"', path.read_text(errors="ignore")):
            name = m.group(1).rsplit("/", 1)[-1]
            own = owner.get(name)
            if own is not None and here != own and not here.startswith(own + "/"):
                leaks.append((path.relative_to(ROOT).as_posix(), m.group(1), own))
    return leaks


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
        "# Declared dependency edges between directories: #include in src/,",
        "# import in lib/. Checked by scripts/gate/layer_check.py, which",
        "# explains what this is for.",
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

    leaks = internal_leaks()
    for where, inc, own in leaks:
        print(f"internal header crossing a directory: {where} includes {inc}\n"
              f"    only {own}/ may include it", file=sys.stderr)

    added = sorted(set(found) - declared)
    gone = sorted(declared - set(found))
    for src, dst in added:
        print(f"undeclared edge: {src} -> {dst}\n    {found[(src, dst)]}", file=sys.stderr)
    for src, dst in gone:
        print(f"declared edge no longer exists: {src} -> {dst}", file=sys.stderr)
    if added or gone or leaks:
        print(f"\n{len(added)} new, {len(gone)} stale, {len(leaks)} internal leaks. If the change is intended, re-run "
              f"with --write and commit the manifest.", file=sys.stderr)
        return 1

    print(f"layers ok, {len(found)} declared edges, no internal header crossings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
