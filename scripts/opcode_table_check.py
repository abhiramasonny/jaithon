#!/usr/bin/env python3
"""Fail when the three opcode tables disagree.

`JAI_OPCODES(X)` in src/vm/bytecode/chunk.c is the wire format: its order is
the on-disk encoding. `_OPS` in lib/jaithon/compile/emit.jai is a
hand-transcribed copy of it, and spec/BYTECODE.md is normative documentation
for both. Nothing checked that the three agreed until this script, and the
survey that motivated it found four opcodes shipped and undocumented.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

C_ROW = re.compile(r"X\(\s*(OP_[A-Z0-9_]+)\s*,\s*([A-Z_]+|-?\d+)\s*,"
                   r"\s*([A-Z_]+|[-+]?\d+)\s*\)")
JAI_ROW = re.compile(r'OpSpec\(\s*Op\.\w+\s*,\s*"(OP_[A-Z0-9_]+)"\s*,'
                     r'\s*([A-Z_]+|-?\d+)\s*,\s*([A-Z_]+|[-+]?\d+)\s*\)')
SPEC_NAME = re.compile(r"`([A-Z][A-Z0-9_]*)`")

ALIASES = {
    "OPERANDS_VARIABLE": "-1",
    "SE_VAR": "var",
    "STACK_EFFECT_VARIABLE": "var",
}


def norm(token):
    if token in ALIASES:
        return ALIASES[token]
    return str(int(token))


def c_table():
    text = (ROOT / "src/vm/bytecode/chunk.c").read_text()
    body = text.split("#define JAI_OPCODES(X)", 1)[1].split("#define X_NAME", 1)[0]
    return [(n, norm(o), norm(e)) for n, o, e in C_ROW.findall(body)]


def jai_table():
    text = (ROOT / "lib/jaithon/compile/emit.jai").read_text()
    return [(n, norm(o), norm(e)) for n, o, e in JAI_ROW.findall(text)]


def spec_names():
    text = (ROOT / "spec/BYTECODE.md").read_text()
    return set(SPEC_NAME.findall(text))


def main():
    c, jai, spec = c_table(), jai_table(), spec_names()
    problems = []

    if not c:
        problems.append("chunk.c: JAI_OPCODES parsed as empty -- regex is stale")
    if not jai:
        problems.append("emit.jai: _OPS parsed as empty -- regex is stale")

    if len(c) != len(jai):
        problems.append(f"row count: chunk.c has {len(c)}, emit.jai has {len(jai)}")

    for i, (a, b) in enumerate(zip(c, jai)):
        if a != b:
            problems.append(f"row {i}: chunk.c {a} != emit.jai {b}")

    # spec/BYTECODE.md names opcodes without the OP_ prefix.
    for name, _, _ in c:
        if name[3:] not in spec:
            problems.append(f"spec/BYTECODE.md does not document {name}")

    if problems:
        print(f"opcode table check FAILED ({len(problems)} problem(s)):")
        for p in problems:
            print(f"  {p}")
        return 1

    print(f"opcode table check ok: {len(c)} opcodes, three tables agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
