#!/usr/bin/env python3
"""Every fused opcode must have an arm in the function JIT.

The function tier's opcode switch ends in `default: return false`, which
declines the WHOLE function -- not the instruction. So a fused opcode without an
arm does not merely run slower: it evicts every function that contains it from
the compiled tier, and a fused opcode is by construction emitted into exactly
the hot loops that most want compiling.

That has now happened twice to the same opcode. `OP_CMP_LOCAL_CONST_LT` shipped
without an arm and cost 14 distinct declines (docs/roadmap.md), and was still
missing one on 2026-08-12 -- invisible because no benchmark happened to contain
it. Fusion is a JIT ADMISSION gate, not a dispatch optimisation: at -O0 life's
`step` declines at OP_FOR_ITER and runs 151.2 ms; at -O2 the peephole produces
OP_FOR_ITER_BIND, it compiles, and it runs 21.8 ms. An unhandled fused opcode
turns that 7x the wrong way round.

So this is a build gate, not a lint. Adding a fused opcode without teaching the
JIT about it should fail the build, which is strictly stronger than letting it
fall back to something slower.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHUNK_C = ROOT / "src/vm/bytecode/chunk.c"
JIT_C = ROOT / "src/vm/jit/jit_func.c"

# A fused opcode is one the peephole SYNTHESISES -- it never comes from the
# parser, so it appears only when the optimiser put it there, and only in code
# hot enough for the optimiser to have bothered. Listed explicitly rather than
# inferred: "is this fused" is a judgement about intent, and a wrong guess here
# would either wave through a real gap or block a build for nothing.
FUSED = [
    "OP_ADD_INT_CONST",
    "OP_INC_LOCAL",
    "OP_CMP_LOCAL_CONST_LT",
    "OP_GET_LOCAL2",
    "OP_ADD_LOCALS",
    "OP_GET_FIELD_LOCAL",
    "OP_FOR_ITER_BIND",
    "OP_JUMP_IF_CMP_FALSE",
    "OP_JUMP_IF_CMP_LOCAL_K",
    "OP_MOD_INT_CONST",
    "OP_ADD_BIND",
    "OP_SUB_BIND",
    "OP_MUL_BIND",
]


def opcodes_in_chunk():
    body = CHUNK_C.read_text()
    body = body.split("#define JAI_OPCODES(X)", 1)[1].split("#define X_NAME", 1)[0]
    return set(re.findall(r"X\(\s*(OP_[A-Z0-9_]+)", body))


def opcodes_with_jit_arms():
    text = JIT_C.read_text()
    return set(re.findall(r"case\s+(OP_[A-Z0-9_]+)\s*:", text))


def main():
    declared = opcodes_in_chunk()
    armed = opcodes_with_jit_arms()
    problems = []

    if not declared:
        problems.append("chunk.c: JAI_OPCODES parsed as empty -- regex is stale")
    if not armed:
        problems.append("jit_func.c: no `case OP_...:` found -- regex is stale")

    # The list itself must stay true, or the gate silently checks nothing.
    for name in FUSED:
        if name not in declared:
            problems.append(
                f"{name} is in this script's FUSED list but not in chunk.c "
                f"-- the list is stale")

    for name in FUSED:
        if name in declared and name not in armed:
            problems.append(
                f"{name} is fused but has no `case {name}:` in "
                f"src/vm/jit/jit_func.c -- every function containing it will "
                f"decline WHOLE")

    if problems:
        print(f"jit fusion check FAILED ({len(problems)} problem(s)):")
        for p in problems:
            print(f"  {p}")
        return 1

    print(f"jit fusion check ok: {len(FUSED)} fused opcodes, all have JIT arms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
