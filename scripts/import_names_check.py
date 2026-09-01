#!/usr/bin/env python3
"""Fail when `from X import name` asks a module for a name it does not export.

Modules resolve at run time (spec §8). `OP_IMPORT_FROM` in src/vm/vm.c asks the
loaded module object for the name and raises `ImportError` when it is absent,
and nothing before that point looks: `_bind_imported` in
lib/jaithon/compile/check/checker.jai binds the names the signature loader
finds and is silent about the ones it does not. So a wrong name in an import
list is not a compile error, it is a failure on whatever line first runs the
import -- and a module whose bad import sits off the path the tests take ships
green. One did, for as long as it had existed:
`packages/jaiframe/src/jaiframe/io.jai`.

The surface checked against is the runtime's, out of `moduleMember` in
src/vm/vm.c. A module that writes neither `pub` nor `export {}` exposes every
top-level name it binds; one that writes either exposes exactly the union of
the two. A `from` import binds a global and exports nothing, so a re-export
facade *is* its `export {}` block, which is why that block is what this reads
rather than the imports above it. Each link is checked where it is written, so
a facade re-exporting a name its own source lacks is reported against the
facade rather than against everyone downstream.

Text only, so it costs nothing to run on every `make test`. A form it cannot
resolve is skipped and counted, never guessed at; `--verbose` names those.

    scripts/import_names_check.py            check, the make target's mode
    scripts/import_names_check.py --verbose  also list what was skipped
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TREES = ("lib", "packages")

#: `MODULE_EXT`, `PACKAGE_FILE` and `PROJECT_MANIFEST` in
#: lib/jaithon/compile/check/modsig.jai, mirroring src/runtime/modules/module_path.c.
MODULE_EXT = ".jai"
PACKAGE_FILE = "mod.jai"
PROJECT_MANIFEST = "jaithon.package.json"

MODULE_PATH = r"\.*[^\W\d]\w*(?:\.[^\W\d]\w*)*"
IMPORT = re.compile(r"^([ \t]*)import\s+(" + MODULE_PATH + r")\s*(?:\bas\s+(\w+))?\s*$")
FROM_IMPORT = re.compile(r"^([ \t]*)from\s+(" + MODULE_PATH + r")\s+import\s+(.+?)\s*$")
IMPORT_ITEM = re.compile(r"^([^\W\d]\w*)(?:\s+as\s+([^\W\d]\w*))?$")

DECL = re.compile(r"^(pub\s+)?(fn|class|trait|enum|type)\s+([^\W\d]\w*)")
BINDING = re.compile(r"^(pub\s+)?(?:let|const|var)\s+(.*)$")
EXPORT_OPEN = re.compile(r"^export\s*\{(.*)$")


def scrub(text):
    """The file's lines with comments and triple-quoted bodies blanked out.

    One left-to-right pass, because a `#` inside a Metal shader source is not a
    comment and a triple quote inside a doc comment does not open a string.
    """
    lines = []
    inside = False
    for raw in text.splitlines():
        kept = ""
        rest = raw
        while rest:
            if inside:
                cut = rest.find('"""')
                if cut < 0:
                    break
                rest = rest[cut + 3:]
                inside = False
                continue
            hash_at = rest.find("#")
            quote_at = rest.find('"""')
            if quote_at >= 0 and (hash_at < 0 or quote_at < hash_at):
                kept += rest[:quote_at]
                rest = rest[quote_at + 3:]
                inside = True
                continue
            if hash_at >= 0:
                kept += rest[:hash_at]
                break
            kept += rest
            break
        lines.append(kept)
    return lines


def split_module_name(dotted):
    """`splitModuleName`: leading dots, then the rest as a relative path."""
    dots = 0
    while dots < len(dotted) and dotted[dots] == ".":
        dots += 1
    if dots >= len(dotted):
        return None
    parts = dotted[dots:].split(".")
    for part in parts:
        if not part or re.search(r"\W", part):
            return None
    return dots, "/".join(parts)


def try_directory(directory, relative):
    """`_try_directory`: the module's own file wins over its package file."""
    own = directory / (relative + MODULE_EXT)
    if own.is_file():
        return own.resolve()
    package = directory / relative / PACKAGE_FILE
    if package.is_file():
        return package.resolve()
    return None


def resolve_module_path(dotted, from_dir, search):
    """`resolve_module_path` in modsig.jai, over real directories."""
    split = split_module_name(dotted)
    if split is None:
        return None
    dots, relative = split

    if dots > 0:
        base = from_dir
        for _ in range(dots - 1):
            if base.parent == base:
                return None
            base = base.parent
        return try_directory(base, relative)

    here = try_directory(from_dir, relative)
    if here is not None:
        return here
    for directory in search:
        found = try_directory(directory, relative)
        if found is not None:
            return found
    return None


def search_roots():
    """`lib/`, then every workspace package's import root: `_add_package_dirs`."""
    roots = [ROOT / "lib"]
    packages = ROOT / "packages"
    if packages.is_dir():
        for entry in sorted(packages.iterdir()):
            if (entry / PROJECT_MANIFEST).is_file() and (entry / "src").is_dir():
                roots.append(entry / "src")
    return roots


def binding_names(head):
    """The names one top-level `let`/`const`/`var` binds.

    A destructuring pattern binds every one of its names and the module exposes
    all of them, so `pub let (WIDTH, HEIGHT) = ...` is two names, not none.
    """
    head = head.split("=", 1)[0].strip()
    if head[:1] in ("(", "["):
        closer = ")" if head[0] == "(" else "]"
        end = head.find(closer)
        inner = head[1:end if end >= 0 else len(head)]
        return [found.group(1) for part in inner.split(",")
                for found in [re.match(r"\s*([^\W\d]\w*)", part)] if found]
    found = re.match(r"([^\W\d]\w*)", head)
    return [found.group(1)] if found else []


class Module:
    """One module's globals and export surface, and the imports it writes."""

    def __init__(self, path):
        self.path = path
        #: Every name bound at module scope: what `jaiModuleGet` can see.
        self.globals = set()
        #: The export table. Empty means "exposes everything" (`moduleMember`).
        self.exports = set()
        #: (line, dotted, [(name, alias)]), the names None for `import X`.
        self.imports = []
        self.skipped = []

    def surface(self):
        return self.exports if self.exports else self.globals


def read_module(path):
    module = Module(path)
    open_export = False

    for number, line in enumerate(scrub(path.read_text(errors="ignore")), 1):
        if open_export:
            head, closed, _rest = line.partition("}")
            module.exports.update(re.findall(r"[^\W\d]\w*", head))
            open_export = not closed
            continue

        stripped = line.strip()
        if not stripped:
            continue
        top = not line[:1].isspace()

        found = FROM_IMPORT.match(line)
        if found is not None:
            names = _import_items(module, number, found.group(3))
            module.imports.append((number, found.group(2), names))
            if top and names is not None:
                module.globals.update(alias or name for name, alias in names)
            continue

        found = IMPORT.match(line)
        if found is not None:
            module.imports.append((number, found.group(2), None))
            if top:
                module.globals.add(found.group(3) or found.group(2).rsplit(".", 1)[-1])
            continue

        if not top:
            continue

        found = EXPORT_OPEN.match(stripped)
        if found is not None:
            head, closed, _rest = found.group(1).partition("}")
            module.exports.update(re.findall(r"[^\W\d]\w*", head))
            open_export = not closed
            continue

        found = DECL.match(stripped)
        if found is not None:
            public = found.group(1) is not None
            #: A non-`pub` `type` binds no global at all: emit.jai writes an
            #: alias's `DefGlobal` only under `Visibility.Public`.
            if public or found.group(2) != "type":
                module.globals.add(found.group(3))
            if public:
                module.exports.add(found.group(3))
            continue

        found = BINDING.match(stripped)
        if found is not None:
            names = binding_names(found.group(2))
            module.globals.update(names)
            if found.group(1) is not None:
                module.exports.update(names)

    return module


def _import_items(module, number, rest):
    """The `name as alias` list of a `from` import, or None when unreadable."""
    if rest.strip() == "*":
        #: The emitter refuses `import *`, so there is nothing here to check
        #: and nothing to guess at either.
        module.skipped.append((number, "`import *`, which the emitter refuses"))
        return None
    items = []
    for piece in rest.split(","):
        found = IMPORT_ITEM.match(piece.strip())
        if found is None:
            module.skipped.append((number, f"unreadable import item `{piece.strip()}`"))
            return None
        items.append((found.group(1), found.group(2)))
    return items


def sources():
    for tree in TREES:
        for path in sorted(ROOT.joinpath(tree).rglob("*.jai")):
            if "__jaicache__" not in path.parts:
                yield path


def main():
    verbose = "--verbose" in sys.argv
    search = search_roots()
    loaded = {}

    def load(dotted, from_dir):
        resolved = resolve_module_path(dotted, from_dir, search)
        if resolved is None:
            return None
        if resolved not in loaded:
            loaded[resolved] = read_module(resolved)
        return loaded[resolved]

    missing = []
    hidden = []
    unresolved = []
    skipped = []
    files = 0
    checked = 0

    for path in sources():
        files += 1
        where = path.relative_to(ROOT).as_posix()
        module = loaded.setdefault(path.resolve(), read_module(path))
        for number, dotted, names in module.imports:
            target = load(dotted, path.parent)
            if target is None:
                unresolved.append((where, number, dotted))
                continue
            if names is None:
                continue
            surface = target.surface()
            told = target.path.relative_to(ROOT).as_posix()
            for name, _alias in names:
                checked += 1
                if name in surface:
                    continue
                if name in target.globals:
                    hidden.append((where, number, dotted, name, told))
                else:
                    missing.append((where, number, dotted, name, told))
        skipped.extend((where, number, why) for number, why in module.skipped)

    for where, number, dotted, name, told in missing:
        print(f"{where}:{number}: `{dotted}` has no `{name}`\n"
              f"    {told} declares no such top-level name", file=sys.stderr)
    for where, number, dotted, name, told in hidden:
        print(f"{where}:{number}: `{dotted}` does not export `{name}`\n"
              f"    {told} binds it but leaves it off its export surface", file=sys.stderr)
    for where, number, dotted in unresolved:
        print(f"{where}:{number}: no module resolves `{dotted}`", file=sys.stderr)

    if verbose:
        for where, number, why in skipped:
            print(f"skipped {where}:{number}: {why}")

    if missing or hidden or unresolved:
        print(f"\n{len(missing)} undefined, {len(hidden)} unexported, "
              f"{len(unresolved)} unresolvable. Each raises ImportError the "
              f"first time its line runs.", file=sys.stderr)
        return 1

    print(f"imports ok, {checked} imported names over {files} files all resolve"
          + (f", {len(skipped)} form(s) skipped" if skipped else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
