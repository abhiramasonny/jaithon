#!/usr/bin/env python3
"""A non-static function in src/ must be called from outside its own file.

Splitting a file is when linkage gets widened. Every helper that two halves of
the old file shared has to lose `static` and gain a declaration, and nothing
afterwards narrows it again -- so the tree ends up exporting symbols that only
one .c ever touches, and exporting some that nothing touches at all. Both are
invisible to the compiler: `static` is the only thing it can warn about, and a
function that is merely never called links fine.

The linker knows the answer, so ask it rather than the source. `nm` over the
object tree gives, per translation unit, the symbols it defines and the ones it
leaves undefined; a defined text symbol that no other object leaves undefined is
reached from nowhere else. That is what makes this trustworthy where grep is
not: taking a function's ADDRESS is an undefined reference too, so the JIT's
runtime thunks -- materialised with emitConst64 and never named at a call site
-- are correctly seen as reached, and so is anything sitting in a table of
function pointers.

Two verdicts, because they want different fixes:

    file-local   its own file calls it, nobody else does -- make it `static`
    dead         nothing calls it anywhere, including itself -- delete it

`dead` under-counts, because it is one round of a fixpoint and this runs one
round. jaiArenaAlloc's only callers are the five dead jaiArena* functions, and
the five chunk writers are reached only from each other; deleting a dead group
is what makes its private helpers dead in turn.

The tests/ harnesses link against these objects too (tests/vm/verify_chunk.c
and its seven siblings), so their text counts as a call site. Without them
jaiChunkPatchU16, jaiCodeArenaFree, jaiCrc32Table and the whole LTV1 line-table
group read as dead, which they are not.

It answers for the build in front of it, which is the one caveat worth knowing:
a function reached only from a branch this host preprocesses away -- the Linux
half of src/native/ -- has no caller here to find.

Findings are pinned in src/linkage.manifest so this starts green and a NEW one
fails, the same bargain scripts/gate/layer_check.py makes with the include
graph. Also checked, and free once the symbol table is loaded: a prototype in a
header that nothing defines.

    scripts/gate/dead_code_check.py --build build/release
    scripts/gate/dead_code_check.py --build build/release --write
    scripts/gate/dead_code_check.py --includes    (slow; see includes())
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
MANIFEST = ROOT / "src" / "linkage.manifest"

IDENT = re.compile(r"[A-Za-z_]\w*")
INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"')
KEYWORDS = {"if", "for", "while", "switch", "return", "sizeof", "defined",
            "do", "else", "case", "_Static_assert", "_Alignof"}


#: Compiler-generated and Objective-C runtime symbols. They are emitted by
#: clang, not written by anyone, so "nothing calls it" says nothing about them.
def generated(sym):
    return sym.startswith("___") or sym.startswith("_OBJC_") or sym == "_main"


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


SYMBOLS = {}


def symbols(build):
    """{defined: {sym: [obj]}, referenced: {sym}} for every object under build.

    One nm over all of them, not one per object: 116 separate calls take five
    seconds and the single call takes fifty milliseconds, which is the whole
    difference between this being a gate and being a chore.
    """
    if build in SYMBOLS:
        return SYMBOLS[build]
    objs = sorted(build.rglob("*.o"))
    if not objs:
        sys.exit(f"FAIL: no object files under {build}; build it first")
    out = subprocess.run(["nm", "-g"] + [str(o) for o in objs],
                         capture_output=True, text=True).stdout
    defined, referenced, kind = {}, set(), {}
    #: nm prints the path only when it was handed more than one file.
    here = objs[0].relative_to(build).as_posix()
    for line in out.splitlines():
        if line.endswith(":") and line[:-1].endswith(".o"):
            here = pathlib.Path(line[:-1]).relative_to(build).as_posix()
            continue
        parts = line.split()
        if len(parts) == 2 and parts[0] == "U":
            referenced.add(parts[1])
        elif len(parts) == 3:
            defined.setdefault(parts[2], []).append(here)
            kind[parts[2]] = parts[1]
    SYMBOLS[build] = (defined, referenced, kind)
    return SYMBOLS[build]


def source_of(objrel):
    """build/release/src/vm/table.o -> src/vm/table.c."""
    stem = objrel[:-2]
    for suffix in (".c", ".m", ".S"):
        path = ROOT / (stem + suffix)
        if path.exists():
            return path
    return None


def harness_words():
    """Identifiers named by the C test harnesses, which link these objects."""
    words = set()
    for path in sorted(ROOT.joinpath("tests").rglob("*.c")):
        words |= set(IDENT.findall(strip_comments(path.read_text(errors="ignore"))))
    return words


def unreferenced(build):
    """[(verdict, source, name)] for exports nothing outside their file reaches."""
    defined, referenced, kind = symbols(build)
    outside = harness_words()
    found = []
    for sym, where in defined.items():
        if kind[sym] != "T" or len(where) != 1 or generated(sym):
            continue
        name = sym[1:]
        if sym in referenced or name in outside:
            continue
        source = source_of(where[0])
        if source is None:
            continue
        text = strip_comments(source.read_text(errors="ignore"))
        word = re.compile(r"\b" + re.escape(name) + r"\b")
        mentions = sum(1 for line in text.splitlines() if word.search(line))
        verdict = "file-local" if mentions > 1 else "dead"
        found.append((verdict, source.relative_to(ROOT).as_posix(), name))
    return sorted(found, key=lambda f: (f[1], f[2]))


def blank_bodies(text):
    """Blank whatever sits between braces, keeping the braces themselves.

    `return jaiStringHash(AS_STRING(v));` inside a static inline in object.h
    ends in `);` like a declaration does, and reads as one. Keeping the braces
    is what then tells a definition from a declaration: only the declaration
    still ends in a semicolon.
    """
    out, depth = [], 0
    for char in text:
        if char == "{":
            out.append(char)
            depth += 1
        elif char == "}":
            depth = max(0, depth - 1)
            out.append(char)
        elif depth > 0:
            out.append("\n" if char == "\n" else " ")
        else:
            out.append(char)
    return "".join(out)


def undefined_prototypes(build):
    """Prototypes in src/*.h that nothing defines, anywhere.

    A prototype nothing CALLS is invisible to the linker -- the program links
    without ever asking for the symbol -- so only reading the headers finds it.
    A definition with no prototype needs no check: -Wmissing-prototypes is on.
    """
    defined, _, _ = symbols(build)
    have = {s[1:] for s in defined}
    proto = re.compile(
        r"(?:^|;|\}|\n)\s*((?:[A-Za-z_][\w \t\*]*?)\b(\w+)\s*"
        r"\([^;{()]*(?:\([^()]*\)[^;{()]*)*\)\s*;)", re.S)
    sources = {p: p.read_text(errors="ignore")
               for p in list(ROOT.joinpath("src").rglob("*.c"))
               + list(ROOT.joinpath("src").rglob("*.m"))}
    missing = []
    for header in sorted(ROOT.joinpath("src").rglob("*.h")):
        text = strip_comments(header.read_text(errors="ignore"))
        text = "\n".join("" if l.lstrip().startswith("#") else l
                         for l in text.splitlines())
        for match in proto.finditer(blank_bodies(text)):
            decl, name = match.group(1), match.group(2)
            if decl.lstrip().startswith(("typedef", "static")) or name in have:
                continue
            if name in KEYWORDS:
                continue
            #: A body behind a build flag nobody sets by default -- object.c's
            #: allocation census wants -DJAI_ALLOC_CENSUS -- is defined, just
            #: not in this build. Only a name with no body at all is a finding.
            if any(re.search(r"^[A-Za-z_][\w \t\*]*\b" + re.escape(name) + r"\s*\(",
                             body, re.M) for body in sources.values()):
                continue
            missing.append(f"{header.relative_to(ROOT).as_posix()}: {name}")
    return sorted(set(missing))


def header_provides(header, flags):
    """Names declared IN this header, from clang's own parse of it."""
    names = set(re.findall(r"^\s*#\s*define\s+(\w+)",
                           header.read_text(errors="ignore"), re.M))
    out = subprocess.run(["cc", "-std=c11"] + flags +
                         ["-Xclang", "-ast-dump=json", "-fsyntax-only", "-x", "c",
                          str(header)], capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout:
        return names, False
    try:
        tree = json.loads(out.stdout)
    except ValueError:
        return names, False
    current = None
    for decl in tree.get("inner", []):
        current = decl.get("loc", {}).get("file", current)
        if current != str(header):
            continue
        if decl.get("name"):
            names.add(decl["name"])
        for inner in decl.get("inner", []):
            if inner.get("kind") == "EnumConstantDecl" and inner.get("name"):
                names.add(inner["name"])
    return names, True


def includes(build):
    """#include lines whose header this file needs nothing from.

    Two questions, and both have to answer yes. Does the file NAME anything the
    header declares -- asked of clang's parse of the header, not of a guess
    about what a header called `gc.h` is for. And does the file still compile
    with the line deleted. Neither alone is worth reading. Measured on one
    tree, the compile question alone answered 150 and both together answered
    66: the 84 it drops are mostly a file's own header coming back through a
    second include, which is precisely the include you must keep.

    It costs one compile per candidate line -- a quarter of a minute on this
    tree, against half a second for the linkage check -- which is why it is
    opt-in and not in `make test`. It also only answers for the host: an
    include used inside a branch this platform preprocesses away cannot be
    seen to be used.
    """
    flags = ["-I" + str(ROOT), "-I" + str(ROOT / "src"), "-I" + str(build.parent),
             "-DJAI_DEBUG"]
    warn = ["-Wall", "-Wextra", "-Wshadow", "-Wstrict-prototypes",
            "-Wmissing-prototypes", "-Wpointer-arith", "-Wcast-align",
            "-Wwrite-strings", "-Wno-unused-parameter", "-Werror"]
    cache, found = {}, []
    for source in sorted(list(ROOT.joinpath("src").rglob("*.c"))
                         + list(ROOT.joinpath("src").rglob("*.m"))):
        lines = source.read_text(errors="ignore").splitlines(keepends=True)
        body = strip_comments("".join(l for l in lines if not INCLUDE.match(l)))
        used = set(IDENT.findall(body))
        for i, line in enumerate(lines):
            match = INCLUDE.match(line)
            if not match:
                continue
            header = next((p for p in (source.parent / match.group(1),
                                       ROOT / "src" / match.group(1),
                                       ROOT / match.group(1)) if p.exists()), None)
            if header is None:
                continue
            header = header.resolve()
            if header not in cache:
                cache[header] = header_provides(header, flags)
            names, parsed = cache[header]
            if not parsed or names & used:
                continue
            probe = source.with_name(source.stem + ".__probe" + source.suffix)
            probe.write_text("".join(lines[:i] + lines[i + 1:]))
            objc = ["-fobjc-arc"] if source.suffix == ".m" else []
            ok = subprocess.run(["cc", "-std=c11"] + warn + flags + objc +
                                ["-fno-common", "-O0", "-g0", "-c", str(probe),
                                 "-o", "/dev/null"],
                                capture_output=True).returncode == 0
            probe.unlink(missing_ok=True)
            if ok:
                found.append(f"{source.relative_to(ROOT).as_posix()}:{i + 1}: "
                             f"{match.group(1)}")
    return found


def read_manifest():
    """{(verdict, name)}. The path on each line is for a reader, not the check.

    Keying on the symbol rather than the file is what lets a split move a
    function between .c files without turning this gate red: the thing being
    pinned is which names are unreached, and that does not change when the
    file they sit in is renamed.
    """
    if not MANIFEST.exists():
        return None
    listed = set()
    for line in MANIFEST.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        listed.add((fields[0], fields[1]))
    return listed


def write_manifest(found):
    lines = [
        "# Non-static functions in src/ that nothing outside their own .c",
        "# reaches, and prototypes nothing defines. `file-local` wants",
        "# `static`; `dead` wants deleting. Checked by",
        "# scripts/gate/dead_code_check.py, which explains what this is for.",
        "# The path is for a reader; only the verdict and the name are checked.",
        "# Regenerate with: scripts/gate/dead_code_check.py --write",
        "",
    ]
    lines += [f"{verdict:11s}{name:28s}{where}" for verdict, where, name in found]
    MANIFEST.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="build/release")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--includes", action="store_true")
    args = ap.parse_args()
    build = pathlib.Path(args.build)
    if not build.is_absolute():
        build = ROOT / build

    if args.includes:
        found = includes(build)
        for item in found:
            print(item)
        print(f"\n{len(found)} includes whose header the file needs nothing from")
        return 0

    found = unreferenced(build) + [("undefined", item.split(": ")[0],
                                    item.split(": ")[1])
                                   for item in undefined_prototypes(build)]
    if args.write:
        write_manifest(found)
        dead = sum(1 for verdict, _, _ in found if verdict == "dead")
        undef = sum(1 for verdict, _, _ in found if verdict == "undefined")
        print(f"wrote {MANIFEST.relative_to(ROOT)}: {len(found) - undef} "
              f"unreferenced ({dead} dead), {undef} undefined")
        return 0

    listed = read_manifest()
    if listed is None:
        print(f"FAIL: no {MANIFEST.relative_to(ROOT)}; run with --write",
              file=sys.stderr)
        return 1

    where_of = {(verdict, name): where for verdict, where, name in found}
    current = set(where_of)
    added = sorted(current - listed)
    gone = sorted(listed - current)
    for verdict, name in added:
        where = where_of[(verdict, name)]
        if verdict == "undefined":
            print(f"declared and never defined: {where}: {name}", file=sys.stderr)
        else:
            print(f"nothing outside {where} reaches {name} ({verdict})",
                  file=sys.stderr)
    for verdict, name in gone:
        print(f"manifest entry no longer holds: {verdict} {name}", file=sys.stderr)
    if added or gone:
        print(f"\n{len(added)} new, {len(gone)} stale. If the change is intended, "
              f"re-run with --write and commit the manifest.", file=sys.stderr)
        return 1

    dead = sum(1 for verdict, _, _ in found if verdict == "dead")
    undef = sum(1 for verdict, _, _ in found if verdict == "undefined")
    print(f"linkage ok, {len(found) - undef} exports reach nobody else "
          f"({dead} dead), {undef} undefined prototypes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
