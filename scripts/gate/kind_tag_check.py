#!/usr/bin/env python3
"""No NEW site in the JIT may build a Value tag out of a compile-time SlotKind.

A `Value` is a tag and a payload. The compiled tier keeps only payloads in
registers, so every time it hands a Value back to the interpreter -- a deopt
record, a call descriptor, a rooted spill, a return -- it has to invent the tag
again. The tier's SlotKind is the obvious thing to invent it from, and it is
the wrong thing, because:

    A TAG RECONSTRUCTED FROM A COMPILE-TIME KIND IS A LIE WHENEVER THE
    RUN-TIME VALUE DECIDES.

The payload decides null-ness. A `dynamic` slot's kind is a speculation, not a
fact. A register home carries no tag at all, so a local past the arity starts as
a bare zero. Two of the tier's confirmed miscompiles are that one sentence:

  * the deopt stub wrote an unconditional `VAL_OBJ` for object kinds, so a
    local that was still a bare zero became {VAL_OBJ, obj = NULL}. The next
    `IS_INSTANCE` -- which is `IS_OBJ(v) && AS_OBJ(v)->type == OBJ_INSTANCE`,
    and whose tag half passed -- dereferenced it. SIGSEGV under
    JAITHON_JIT_DEOPT_STRESS.
  * an OSR `dynamic` local was read as a bare payload with no tag check, and
    its operand-stack entry was then stamped with the walk's last-written kind,
    so `m ?? 100` answered 0.

`SLOT_MAYBE_INST` is the kind that names the hazard: the same register holds a
pointer or a zero, and only the payload says which. `emitTagFor` gets it right
-- a `subs`/`csel` off the payload -- and every ladder in the tier today either
routes that kind through `emitTagFor`, refuses it outright, or is only ever the
EXPECTED operand of a guard, where a wrong guess costs a deopt rather than a
wrong answer.

Telling those three apart is a judgement, not a pattern, so this script does
not try to make it. It does two things:

  1. Enumerates every SlotKind-to-tag mapping in `src/vm/jit/` and pins the set
     in tests/vm/jit_kind_tag.manifest with the exact kinds and tags each one
     maps. A NEW site, or an existing one whose kinds or tags moved, fails --
     and the failure message is the review checklist: decide which of the four
     justifications below applies, write it into the manifest, and say so in the
     commit. `--write` refreshes the mechanical columns and CARRIES FORWARD the
     justification you already wrote; a brand-new site is written
     `UNJUSTIFIED`, which is itself a failure. So regenerating cannot silence
     the gate -- only deciding can.

     Both-sided. A manifest line with no site left is a failure too, because a
     stale pin guards nothing.

  2. Fails outright, with no manifest exemption, on the one shape that is
     always the bug: a ladder that maps `SLOT_MAYBE_INST` itself to a constant
     tag. That is the first miscompile above written down, and it is the most
     natural way to reintroduce it -- one more `: kind == SLOT_MAYBE_INST ?
     VAL_OBJ` arm bolted onto a ladder that today routes the kind out to
     `emitTagFor` instead.

The four justifications, all of which exist in the tier today:

    guard    the tag is only ever the expected operand of a compare that
             deoptimises on mismatch (`emitCallOutResult`'s `wantTag`). A wrong
             expectation costs a deopt, never a wrong answer.
    refuses  the ladder has no fall-through: a kind it does not list is refused
             at compile time (`emitListStore`, whose sentinel is 0xffffffffu --
             which is exactly how SLOT_MAYBE_INST is kept out of it).
    payload  every kind whose tag the payload decides is routed out of the
             ladder first, to `emitTagFor`'s csel (both deopt stubs).
    sampled  the object arms carry a Value that was actually observed rather
             than a manufactured tag (`buildListExemplar`).

WHAT THIS DOES NOT COVER. The third miscompile in this family is not in this
shape at all: an empty `fn init(self) {}` compiled to a body whose caller wrote
`NULL_VAL` over `slotBase[0]`, because an initializer's `OP_RETURN_NULL` means
"return the receiver". That is a compile-time verdict about a WHOLE BODY, not a
kind-to-tag ladder, and nothing mechanical here would have seen it.

    python3 scripts/gate/kind_tag_check.py           # check
    python3 scripts/gate/kind_tag_check.py --write   # re-pin the manifest

Pure text, so it costs nothing to run on every `make test`.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
JIT = ROOT / "src/vm/jit"
MANIFEST = ROOT / "tests/vm/jit_kind_tag.manifest"

#: `VAL_OBJ` is the tag itself; `OBJ_VAL` is the C constructor that writes it.
#: Both count -- jitResultOut builds its Value with the constructors and is the
#: same kind of site as the deopt stub that builds one out of the tags.
TAG = re.compile(r"\bVAL_(?:INT|FLOAT|BOOL|NULL|OBJ)\b"
                 r"|\b(?:INT|FLOAT|BOOL|NULL|OBJ)_VAL\b")
KIND = re.compile(r"\bSLOT_[A-Z_]+\b")
#: A definition opens at column 0 in this tree, which is what makes the
#: enclosing function findable without parsing C. Naming the function rather
#: than a line is deliberate: this file gets reshaped constantly, and a pin on a
#: line number would be stale by the next commit.
FNDEF = re.compile(r"^[A-Za-z_].*[A-Za-z0-9_]\s*\(")
NAME = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")

#: The kind whose tag the PAYLOAD decides. A ladder may route it elsewhere or
#: refuse it; it may never map it to a constant.
NULLABLE_KIND = "SLOT_MAYBE_INST"

JUSTIFICATIONS = ("guard", "refuses", "payload", "sampled")

HEADER = """\
# Every place in src/vm/jit/ that maps a compile-time SlotKind onto a Value tag.
#
# A tag reconstructed from a compile-time kind is a lie whenever the RUN-TIME
# VALUE decides. The payload decides null-ness; a `dynamic` slot's kind is a
# speculation; a register home carries no tag at all. Two confirmed miscompiles
# were that one sentence -- a deopt stub that wrote VAL_OBJ over a bare zero and
# handed the interpreter {VAL_OBJ, obj = NULL}, and an OSR dynamic local read
# with no tag check at all.
#
# So every line here is a site somebody had to justify, and the `why` column
# records which justification it has:
#
#   guard    the tag is only the EXPECTED operand of a compare that deopts on
#            mismatch -- a wrong guess costs a deopt, never a wrong answer
#   refuses  no fall-through: a kind the ladder does not list is refused at
#            compile time
#   payload  every kind whose tag the payload decides is routed out of the
#            ladder first, to emitTagFor's csel
#   sampled  the object arms carry an observed Value, not a manufactured tag
#
# `--write` refreshes the kinds and tags and carries the `why` forward; a new
# site arrives as UNJUSTIFIED, which fails until a human replaces it with one of
# the four words above. A `#N` suffix on a target only separates two ladders in
# one function that assign to the same name, numbered in source order.
#
# Regenerate: python3 scripts/gate/kind_tag_check.py --write
# Enforced by that script, whose docstring says what it does and does not cover.
#
# function | target | why | kinds | tags
"""


def strip_comments(text):
    """Blank out comments and string literals, keeping every byte offset."""
    out = []
    i, n = 0, len(text)
    while i < n:
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(re.sub(r"[^\n]", " ", text[i:j]))
            i = j
        elif text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif text[i] == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            j = min(j + 1, n)
            out.append(" " * (j - i))
            i = j
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def function_at(lines, line_no):
    for i in range(line_no - 1, -1, -1):
        s = lines[i]
        if s and not s[0].isspace() and not s.startswith("#") and FNDEF.match(s):
            m = NAME.search(s)
            if m:
                return m.group(1)
    return "<file scope>"


def target_of(stmt):
    """What the ladder assigns to -- the half that tells two ladders in one
    function apart, since `compileBody` alone holds five of them."""
    head = stmt.split("=", 1)[0]
    if "return" in head and "?" not in head:
        return "return"
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_>\-\.\[\]]*", head)
    return words[-1] if words else "?"


def ladder_sites(path):
    """Conditional ladders: one statement naming two or more kinds and two or
    more tags. Statements are cut at `;{}`, so a switch body cannot merge into
    one and an array initialiser is cut per row."""
    code = strip_comments(path.read_text())
    lines = code.splitlines()
    sites = []
    start = 0
    for m in re.finditer(r"[;{}]", code):
        stmt, off, start = code[start:m.end()], start, m.end()
        kinds, tags = set(KIND.findall(stmt)), set(TAG.findall(stmt))
        if len(kinds) < 2 or len(tags) < 2:
            continue
        line_no = code[:off].count("\n") + 1
        sites.append({
            "function": function_at(lines, line_no),
            "target": target_of(stmt),
            "kinds": kinds,
            "tags": tags,
            "where": f"{path.relative_to(ROOT)}:{line_no}",
            "text": " ".join(stmt.split()),
        })
    return sites


def switch_sites(path):
    """`switch` on a SlotKind whose arms build Values. The ladder detector
    cannot see these: each arm is its own statement, naming one kind and one
    tag."""
    code = strip_comments(path.read_text())
    lines = code.splitlines()
    sites = []
    n = len(code)
    for m in re.finditer(r"\bswitch\s*\(", code):
        depth, i = 1, m.end()
        while i < n and depth:
            depth += (code[i] == "(") - (code[i] == ")")
            i += 1
        control = " ".join(code[m.end():i - 1].split())
        brace = code.find("{", i)
        if brace < 0:
            continue
        depth, j = 1, brace + 1
        while j < n and depth:
            depth += (code[j] == "{") - (code[j] == "}")
            j += 1
        body = code[brace:j]
        kinds = set(re.findall(r"\bcase\s+(SLOT_[A-Z_]+)\s*:", body))
        tags = set(TAG.findall(body))
        if len(kinds) < 2 or not tags:
            continue
        line_no = code[:m.start()].count("\n") + 1
        sites.append({
            "function": function_at(lines, line_no),
            "target": "switch " + re.sub(r"^\(\s*SlotKind\s*\)\s*", "", control),
            "kinds": kinds,
            "tags": tags,
            "where": f"{path.relative_to(ROOT)}:{line_no}",
            "text": " ".join(body.split()),
        })
    return sites


def scan():
    """Every site, with same-named ladders in one function separated by a `#N`
    suffix in source order. Two of them exist today: the function tier's deopt
    stub builds `tag` twice, once for the locals and once for the stack."""
    sites = []
    for path in sorted(JIT.glob("*.[ch]")):
        sites.extend(ladder_sites(path))
        sites.extend(switch_sites(path))
    counts = {}
    for site in sites:
        k = (site["function"], site["target"])
        counts[k] = counts.get(k, 0) + 1
    seen = {}
    for site in sites:
        k = (site["function"], site["target"])
        if counts[k] > 1:
            seen[k] = seen.get(k, 0) + 1
            site["target"] = f"{site['target']}#{seen[k]}"
    return sites


def key(site):
    return (site["function"], site["target"])


def columns(site):
    return (",".join(sorted(site["kinds"])), ",".join(sorted(site["tags"])))


def parse_manifest():
    """Returns {key: (why, kinds, tags)}, or a string naming what is wrong."""
    if not MANIFEST.exists():
        return f"missing {MANIFEST.relative_to(ROOT)} -- regenerate with --write"
    pinned = {}
    for raw in MANIFEST.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 5:
            return f"{MANIFEST.relative_to(ROOT)}: malformed line: {raw}"
        pinned[(parts[0], parts[1])] = tuple(parts[2:])
    return pinned


def write_manifest(sites, pinned):
    rows = []
    for site in sorted(sites, key=key):
        why = pinned.get(key(site), ("UNJUSTIFIED",))[0]
        kinds, tags = columns(site)
        rows.append(f"{site['function']} | {site['target']} | {why} | "
                    f"{kinds} | {tags}")
    MANIFEST.write_text(HEADER + "\n".join(rows) + "\n")
    return len(rows)


def main(argv):
    sites = scan()
    if not sites:
        print("kind/tag check FAILED: no sites found in src/vm/jit -- the "
              "detector is stale, not the tier", file=sys.stderr)
        return 1

    pinned = parse_manifest()
    fatal = None
    if isinstance(pinned, str):
        fatal, pinned = pinned, {}

    if "--write" in argv:
        count = write_manifest(sites, pinned)
        print(f"wrote {count} sites to {MANIFEST.relative_to(ROOT)}")
        return 0

    problems = [fatal] if fatal else []

    # Rule 2: never manifest-exempt. Mapping the payload-decided kind to a
    # constant tag is the miscompile, whatever else the site does.
    for site in sites:
        text = site["text"]
        if re.search(NULLABLE_KIND + r"\s*[?:]", text) or \
           re.search(r"case\s+" + NULLABLE_KIND + r"\s*:", text):
            problems.append(
                f"{site['where']}: {site['function']} maps {NULLABLE_KIND} to a "
                f"constant tag. That kind is a pointer OR a zero in one "
                f"register and only the payload says which -- route it to "
                f"emitTagFor's csel, or refuse it. No manifest line excuses "
                f"this one.")

    found = {key(s): s for s in sites}
    for site in sites:
        k = key(site)
        kinds, tags = columns(site)
        if k not in pinned:
            problems.append(
                f"{site['where']}: {k[0]} builds `{k[1]}` from a SlotKind and "
                f"is not pinned. A tag reconstructed from a compile-time kind "
                f"is a lie whenever the run-time value decides -- decide which "
                f"justification this has ({' / '.join(JUSTIFICATIONS)}), "
                f"then --write and replace UNJUSTIFIED with it.")
            continue
        why, was_kinds, was_tags = pinned[k]
        if (was_kinds, was_tags) != (kinds, tags):
            problems.append(
                f"{site['where']}: {k[0]}'s `{k[1]}` now maps {kinds} -> "
                f"{tags}, pinned as {was_kinds} -> {was_tags}. A kind or a tag "
                f"moved; re-check that the justification ({why}) still holds, "
                f"then --write.")
        elif why not in JUSTIFICATIONS:
            problems.append(
                f"{site['where']}: {k[0]}'s `{k[1]}` is pinned `{why}`. "
                f"--write cannot answer this one: say which of "
                f"{' / '.join(JUSTIFICATIONS)} is true of it, in the manifest "
                f"and in the commit.")

    for k in sorted(pinned):
        if k not in found:
            problems.append(
                f"{MANIFEST.relative_to(ROOT)} pins {k[0]}'s `{k[1]}`, which no "
                f"longer exists -- a stale pin guards nothing. --write.")

    if problems:
        print(f"kind/tag check FAILED ({len(problems)} problem(s)):",
              file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1

    print(f"kind/tag check ok: {len(sites)} SlotKind-to-tag sites in "
          f"src/vm/jit, all pinned and justified")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
