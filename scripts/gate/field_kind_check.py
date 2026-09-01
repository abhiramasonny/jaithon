#!/usr/bin/env python3
#: The FieldKind codes are written twice, in two languages, and the emitter's
#: copy is bare integer literals.
#:
#: `src/vm/object/object.h` defines the `FieldKind` enum -- the codes the VM
#: stores in a chunk and checks at run time. `_field_kind_code` in
#: `lib/jaithon/compile/emit/field_kind.jai` decides which code the emitter
#: WRITES, and it does so with `return 1`, `return 5`, `return 8`: nothing in
#: the source connects those numerals to the enum they are numerals of.
#:
#: Renumbering the enum, or inserting a member in the middle of it, changes what
#: every previously written .jaic means without changing a single line of the
#: emitter. The failure is silent and it is not a crash: a field declared `str`
#: would be recorded as, say, LIST, and the compiled tier -- which takes these
#: as a hint and guards on them -- would guard for the wrong thing. Codes also
#: travel in the packed OP_ELEM_KIND byte, so a shift corrupts container element
#: kinds too.
#:
#: Text only. The emitter's side is a ladder of `if n == "name" { return N }`,
#: which is the whole reason this is checkable without running anything.

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "src" / "vm" / "object" / "object.h"
JAI = ROOT / "lib" / "jaithon" / "compile" / "emit" / "field_kind.jai"

ENUM = re.compile(r"^\s*FIELD_KIND_([A-Z]+)\s*=\s*(\d+)\s*,?\s*$", re.M)
ARM = re.compile(r'if\s+n\s*==\s*"([a-z]+)"\s*\{\s*return\s+(\d+)\s*\}')

#: The emitter names a TYPE; the enum names a KIND. Only these are the same
#: thing. A type the emitter learns to name later must be added here rather than
#: assumed -- an unmapped name is reported, not skipped.
NAME_TO_KIND = {
    "int": "INT", "float": "FLOAT", "bool": "BOOL", "str": "STR",
    "list": "LIST", "dict": "DICT", "any": "ANY",
}


def main():
    if not HEADER.exists() or not JAI.exists():
        missing = HEADER if not HEADER.exists() else JAI
        print(f"field kind check: missing {missing}")
        return 1

    kinds = {k: int(v) for k, v in ENUM.findall(HEADER.read_text())}
    text = JAI.read_text()
    arms = ARM.findall(text)

    problems = []
    if not arms:
        problems.append(
            f"no `if n == \"...\" {{ return N }}` arms found in {JAI.name}. "
            f"If the ladder was rewritten, this gate must be rewritten with it.")

    seen = set()
    for name, code in arms:
        code = int(code)
        seen.add(name)
        kind = NAME_TO_KIND.get(name)
        if kind is None:
            problems.append(
                f'{JAI.name} maps the type "{name}" to code {code}, and this '
                f"gate has no FieldKind for it. Add it to NAME_TO_KIND, or the "
                f"code is unchecked.")
        elif kind not in kinds:
            problems.append(
                f"FIELD_KIND_{kind} is not in {HEADER.name} any more, but "
                f'{JAI.name} still emits code {code} for "{name}".')
        elif kinds[kind] != code:
            problems.append(
                f'{JAI.name} emits {code} for "{name}", but FIELD_KIND_{kind} '
                f"is {kinds[kind]} in {HEADER.name}. Every .jaic already "
                f"written carries the old number.")

    #: The fallback arm: a user type name is "some heap object", DECLARED.
    tail = re.search(r"return\s+(\d+)\s*\n\s*\}\s*\n\s*if\s+t\.kind\s*==\s*TypeKind\.Generic", text)
    if tail is None:
        problems.append(
            f"could not find the user-type fallback `return N` before the "
            f"Generic arm in {JAI.name}; the ladder's shape changed.")
    elif "DECLARED" not in kinds:
        problems.append(f"FIELD_KIND_DECLARED missing from {HEADER.name}.")
    elif int(tail.group(1)) != kinds["DECLARED"]:
        problems.append(
            f"{JAI.name} returns {tail.group(1)} for a user type, but "
            f"FIELD_KIND_DECLARED is {kinds['DECLARED']} in {HEADER.name}.")

    for name in sorted(set(NAME_TO_KIND) - seen):
        problems.append(
            f'this gate expects {JAI.name} to name the type "{name}", and it '
            f"no longer does. Drop it from NAME_TO_KIND if that is intended.")

    #: The same type is named in more than one ladder, so one renumbering
    #: produces the identical complaint several times over.
    problems = sorted(set(problems))
    if problems:
        print(f"field kind check FAILED ({len(problems)} problem(s)):")
        for p in problems:
            print(f"  {p}")
        return 1

    print(f"field kind check ok: {len(arms)} emitter arms agree with "
          f"{len(kinds)} FieldKind codes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
