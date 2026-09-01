#!/usr/bin/env python3
"""The builtin method surface is written down twice, and both lists must agree.

`src/runtime/builtins/builtins.c` holds twelve `k*MethodNames[]` arrays -- one
per builtin receiver kind -- and the runtime holds twelve dispatch tables in
four incompatible C shapes scattered under `src/runtime/`. Nothing derives
either from the other. The names list is what `dir(x)` answers with; the
dispatch tables are what a real `x.name(...)` resolves through.

Both directions of disagreement are bugs, and they are different bugs:

  advertised but absent    `dir()` names a method the runtime will not bind.
                           This is the dangerous one: nothing rejects the call
                           earlier, so the program runs and raises
                           AttributeError at the user's run time. It reached 46
                           names once (see the k*MethodNames[] comments and
                           tests/stdlib/test_method_tables.jai). It is zero
                           today and this gate keeps it there.

  implemented, unadvertised  the runtime binds a method `dir()` never names.
                           Nothing crashes, but `dir()` is the only enumeration
                           of the builtin surface anywhere -- spec/LANGUAGE.md
                           names a dozen methods by way of example and stops --
                           so a name missing from it is a method no user can
                           find. Nineteen are in that state; they are listed in
                           UNADVERTISED below with what each one is, and this
                           gate fails on a twentieth.

The runtime, not a regex, is the oracle. Parsing the dispatch tables is how
this gets its CANDIDATE names -- four shapes, so a parse gap costs a finding --
but every name it reports is then asked of a live `jaithon`, through
`__prim__.obj_get_method`, which resolves exactly the way `x.name(...)` does
and raises AttributeError exactly when that call would. It answers presence
without calling anything, so `file.close` and `list.clear` are safe to probe.
An earlier audit that trusted the regex alone reported 102 names; the empirical
pass cut it to 46.

What this does NOT check, because there is nothing to check: `jaithon check`
never consults either table. `[1,2,3].totally_fake_method_xyz()` exits 0 and
`let k: str = [1,2].len()` exits 0, because a method call on a builtin receiver
types as `any`. The checker is not a third reader that could drift; it is a
reader that does not exist.

    scripts/gate/method_surface_check.py            check (make method-surface-check)
    scripts/gate/method_surface_check.py --list     print both tables and stop
"""

import os
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
BUILTINS = ROOT / "src" / "runtime" / "builtins" / "builtins.c"
BINARY = pathlib.Path(os.environ.get("JAITHON", ROOT / "jaithon"))

# One harmless instance per receiver kind, written as Jaithon source. Presence
# is decided by name and receiver kind, never by a receiver's contents, so an
# empty one of each is as good as a full one. `module` gets a generated file of
# its own rather than a stdlib module: a module resolves an exposed member
# before it reaches its own method table, so importing one with exports would
# let `std.math.name` answer for `module.name`.
RECEIVERS = {
    "int": "0",
    "float": "0.0",
    "str": '""',
    "bytes": '"".to_bytes()',
    "list": "[0]",
    "dict": "{0: 0}",
    "set": "{0}",
    "tuple": "(0, 0)",
    "range": "range(1)",
    "iter": "[0].iter()",
    "file": '__prim__.io_open("/dev/null", "r")',
    "module": "probe_empty_module",
}

# Implemented, callable, and named by no k*MethodNames[] array, so absent from
# dir(). Each is a method that works today and that no user can discover.
UNADVERTISED = {
    "bytes": [
        # The search and matching half of the bytes surface. `bytes` advertises
        # `contains` but not the five neighbours implemented beside it.
        "count", "ends_with", "find", "index", "repeat", "starts_with",
    ],
    "file": [
        # `is_eof` is the only way to ask a handle whether it is spent;
        # `path` is what the handle was opened on.
        "is_eof", "path",
    ],
    "int": [
        "bit_count",   # popcount, the sibling of the advertised bit_length
        "pow_mod",     # modular exponentiation, the 3-argument pow
    ],
    "list": [
        "for_each",    # map for effect
        "remove_at",   # remove by index, where the advertised remove is by value
        "window",      # sliding windows, the sibling of the advertised chunks
    ],
    "str": [
        "bytes",         # the UTF-8 units, as the advertised chars is scalars
        "code_points",   # the scalars as ints
        "parse_float",   # to_float that answers null instead of throwing
        "parse_int",     # to_int that answers null instead of throwing
        "rsplit",        # split from the right, beside the advertised split
        "to_bytes",      # the whole string as one bytes value
    ],
}

COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)
# The first string literal of a table row, which is the method name in all four
# shapes: JAI_METHOD("at", ...), JAI_METHOD_KW("all", ...), METHOD_ENTRY("abs",
# ...) and the bare {"len", strLen, 1, 1, NULL} that str and bytes use.
ROW = re.compile(r'[({]\s*"([A-Za-z_][A-Za-z0-9_]*)"')


def strip(text):
    return COMMENT.sub(" ", text)


def advertised():
    """{kind: [name]} out of the k*MethodNames[] arrays dir() answers from."""
    text = strip(BUILTINS.read_text())
    found = {}
    for m in re.finditer(r"k(\w+)MethodNames\[\]\s*=\s*\{(.*?)\};", text, re.S):
        found[m.group(1).lower()] = set(re.findall(r'"([^"]+)"', m.group(2)))
    return found


def candidates():
    """{kind: [name]} out of the dispatch tables, before the runtime confirms.

    Two of the twelve kinds have no table: `file` and `module` dispatch by a
    strcmp chain built from a file-local macro, so they are matched by that
    macro's name instead.
    """
    found = {}
    for path in sorted((ROOT / "src").rglob("*.c")):
        text = strip(path.read_text(errors="ignore"))
        for m in re.finditer(r"k(\w+)Methods\[\]\s*=\s*\{(.*?)\n\};", text, re.S):
            found.setdefault(m.group(1).lower(), set()).update(ROW.findall(m.group(2)))
        for macro, kind in (("FILE_METHOD", "file"), ("MODULE_METHOD", "module")):
            names = re.findall(macro + r'\("([^"]+)"', text)
            if names:
                found.setdefault(kind, set()).update(names)
    return found


def ask_runtime(wanted):
    """{(kind, name): bool}, answered by a live jaithon rather than a regex.

    `__prim__.obj_get_method` binds the method the way `x.name(...)` resolves
    it and raises AttributeError exactly when that call would, without calling
    anything -- which is what makes probing `file.close` and `list.clear` safe.
    """
    lines = [
        "import probe_empty_module",
        "",
        "fn present(receiver: any, name: str) -> bool {",
        "    try {",
        "        let _ = __prim__.obj_get_method(receiver, name)",
        "        return true",
        "    } catch _e: AttributeError {",
        "        return false",
        "    }",
        "}",
        "",
    ]
    for kind in sorted(wanted):
        names = ", ".join(f'"{n}"' for n in sorted(wanted[kind]))
        lines.append(f"let receiver_{kind} = {RECEIVERS[kind]}")
        lines.append(f"for name in [{names}] {{")
        lines.append(f'    print(f"{kind} {{name}} {{present(receiver_{kind}, name)}}")')
        lines.append("}")

    with tempfile.TemporaryDirectory() as work:
        here = pathlib.Path(work)
        (here / "probe_empty_module.jai").write_text("pub let _nothing = 0\n")
        probe = here / "probe.jai"
        probe.write_text("\n".join(lines) + "\n")
        run = subprocess.run([str(BINARY), "run", str(probe)],
                             capture_output=True, text=True)
    if run.returncode != 0:
        print(f"FAIL: the probe did not run\n{run.stdout}{run.stderr}", file=sys.stderr)
        return None

    answers = {}
    for line in run.stdout.splitlines():
        kind, name, verdict = line.split()
        answers[(kind, name)] = verdict == "true"
    return answers


def main():
    adv = advertised()
    cand = candidates()

    if "--list" in sys.argv:
        for kind in sorted(set(adv) | set(cand)):
            print(f"{kind}:")
            print(f"    advertised {sorted(adv.get(kind, ()))}")
            print(f"    dispatch   {sorted(cand.get(kind, ()))}")
        return 0

    if not BINARY.exists():
        print(f"FAIL: no {BINARY}; build it first, or point JAITHON at one",
              file=sys.stderr)
        return 1

    orphans = sorted((set(cand) | set(adv)) - set(RECEIVERS))
    for kind in orphans:
        print(f"receiver kind this gate cannot probe: {kind}\n"
              f"    add an instance of it to RECEIVERS in this script, or its "
              f"methods go unchecked", file=sys.stderr)

    # The baseline joins the probe set too, so a name listed there is asked of
    # the runtime rather than assumed absent when the table parse misses it.
    wanted = {k: adv.get(k, set()) | cand.get(k, set()) | set(UNADVERTISED.get(k, ()))
              for k in sorted(set(adv) | set(cand)) if k in RECEIVERS}
    answers = ask_runtime(wanted)
    if answers is None:
        return 1

    absent, hidden, stale = [], [], []
    for kind in sorted(wanted):
        for name in sorted(wanted[kind]):
            present = answers[(kind, name)]
            if name in adv.get(kind, ()) and not present:
                absent.append((kind, name))
            if present and name not in adv.get(kind, ()) \
                    and name not in UNADVERTISED.get(kind, ()):
                hidden.append((kind, name))
    for kind, names in sorted(UNADVERTISED.items()):
        for name in names:
            if not answers.get((kind, name), False):
                stale.append((kind, name))

    for kind, name in absent:
        print(f"advertised but absent: {kind}.{name}\n"
              f"    dir() names it, the runtime will not bind it, and "
              f"`jaithon check` does not reject the call -- so it raises "
              f"AttributeError at run time", file=sys.stderr)
    for kind, name in hidden:
        print(f"implemented but unadvertised: {kind}.{name}\n"
              f"    it works, and dir() does not name it, so nothing tells a "
              f"user it is there", file=sys.stderr)
    for kind, name in stale:
        print(f"stale baseline entry: {kind}.{name}\n"
              f"    UNADVERTISED still lists it and the runtime no longer "
              f"binds it; drop the line", file=sys.stderr)

    if absent or hidden or stale or orphans:
        print(f"\n{len(absent)} advertised-but-absent, {len(hidden)} newly "
              f"unadvertised, {len(stale)} stale baseline, {len(orphans)} "
              f"unprobed kinds. Add the name to k*MethodNames[] in "
              f"{BUILTINS.relative_to(ROOT)}, or drop it from the dispatch "
              f"table. If it is deliberately hidden, add it to UNADVERTISED "
              f"in this script with what it is.", file=sys.stderr)
        return 1

    baseline = sum(len(v) for v in UNADVERTISED.values())
    checked = sum(len(v) for v in wanted.values())
    print(f"method surface ok, {checked} names across {len(wanted)} receiver "
          f"kinds, 0 advertised-but-absent, {baseline} known unadvertised")
    return 0


if __name__ == "__main__":
    sys.exit(main())
