#!/usr/bin/env python3
"""The two branch-operand tables must agree.

Where an instruction keeps its jump displacement is written down twice:
`jaiOpBranchOperandAt` in src/vm/bytecode/verify.c, and `_build_branch_at` in
lib/jaithon/compile/opt/chunk.jai. Both are hand-maintained, and neither
consults the other.

A missing entry is not a missed optimisation. The optimiser REBUILDS every
chunk and re-resolves every branch, so a branching opcode it does not recognise
makes it lose whatever followed the loop -- the symptom is a chunk that simply
stops, with no diagnostic anywhere, and a program that runs off the end of its
own code. `OP_GET_ITER_ITEMS` shipped missing from both tables and cost an hour
to find from that symptom alone.

A wrong OFFSET is worse than a missing entry: the displacement is then read out
of the middle of some other operand, so the branch goes somewhere plausible.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
VERIFY_C = ROOT / "src/vm/bytecode/verify.c"
OPT_JAI = ROOT / "lib/jaithon/compile/opt/chunk.jai"
EMIT_JAI = ROOT / "lib/jaithon/compile/emit.jai"


def op_name_by_enum():
    """`Op.ForIterBind` -> `OP_FOR_ITER_BIND`, from emit.jai's own _OPS table."""
    text = EMIT_JAI.read_text()
    pairs = re.findall(r'OpSpec\(\s*Op\.(\w+)\s*,\s*"(OP_[A-Z0-9_]+)"', text)
    return {enum: name for enum, name in pairs}


def c_table():
    """OP_NAME -> offset, from the switch in verify.c."""
    text = VERIFY_C.read_text()
    body = text.split("int jaiOpBranchOperandAt(uint8_t op) {", 1)[1]
    body = body.split("\n}", 1)[0]
    table = {}
    pending = []
    for line in body.splitlines():
        line = line.strip()
        m = re.match(r"case\s+(OP_[A-Z0-9_]+)\s*:", line)
        if m:
            pending.append(m.group(1))
            continue
        m = re.match(r"return\s+(-?\d+)\s*;", line)
        if m and pending:
            for name in pending:
                table[name] = int(m.group(1))
            pending = []
    return table


def jai_table(names):
    """OP_NAME -> offset, from _build_branch_at in opt/chunk.jai."""
    text = OPT_JAI.read_text()
    body = text.split("fn _build_branch_at()", 1)[1].split("\nfn ", 1)[0]
    table = {}

    lead = re.search(r"let leading = \[(.*?)\]", body, re.S)
    if lead:
        for enum in re.findall(r"Op\.(\w+)", lead.group(1)):
            if enum in names:
                table[names[enum]] = 0

    for enum, off in re.findall(r"table\[opcode\(Op\.(\w+)\)\]\s*=\s*(-?\d+)", body):
        if enum in names:
            table[names[enum]] = int(off)

    for group, off in re.findall(
            r"for op in \[(.*?)\]\s*\{\s*table\[opcode\(op\)\]\s*=\s*(-?\d+)", body, re.S):
        for enum in re.findall(r"Op\.(\w+)", group):
            if enum in names:
                table[names[enum]] = int(off)
    return table


def main():
    names = op_name_by_enum()
    c = c_table()
    jai = jai_table(names)
    problems = []

    if not names:
        problems.append("emit.jai: _OPS parsed as empty -- regex is stale")
    if not c:
        problems.append("verify.c: jaiOpBranchOperandAt parsed as empty -- regex is stale")
    if not jai:
        problems.append("opt/chunk.jai: _build_branch_at parsed as empty -- regex is stale")

    for name in sorted(set(c) | set(jai)):
        in_c, in_jai = c.get(name), jai.get(name)
        if in_c is None:
            problems.append(
                f"{name}: branches per opt/chunk.jai (offset {in_jai}) but "
                f"verify.c does not list it -- the verifier cannot see the edge")
        elif in_jai is None:
            problems.append(
                f"{name}: branches per verify.c (offset {in_c}) but "
                f"opt/chunk.jai does not list it -- the optimiser's rebuild "
                f"will drop whatever follows it")
        elif in_c != in_jai:
            problems.append(
                f"{name}: verify.c says the displacement is at byte {in_c}, "
                f"opt/chunk.jai says {in_jai} -- one of them reads it out of "
                f"another operand")

    if problems:
        print(f"branch table check FAILED ({len(problems)} problem(s)):")
        for p in problems:
            print(f"  {p}")
        return 1

    print(f"branch table check ok: {len(c)} branching opcodes, both tables agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
