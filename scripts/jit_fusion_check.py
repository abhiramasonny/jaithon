#!/usr/bin/env python3
"""No opcode may newly lack an arm in the function JIT.

What that costs CHANGED on 2026-08-13. The tier's opcode switch used to end in
`default: return false`, which declines the WHOLE function rather than the
instruction -- so an opcode without an arm did not run slower, it evicted every
function containing it from the compiled tier. It now deoptimises at its own
offset: the interpreter resumes exactly there and the code around it still
compiles. `OP_GET_EXC` sat on the baseline and every catch block holds one, so
NO function containing a `try` could be compiled at all; tests/bench/error_paths
ran its eight-million-iteration counted loop interpreted end to end because of a
handler its own opcode histogram says executes zero times.

So this is no longer a coverage cliff -- but it is still a ratchet, because the
deopt is not free. It is an unconditional exit from compiled code, so an unarmed
opcode on a HOT path costs the whole rest of that body, and on a loop's
straight-line path it costs one compiled entry and one deopt per iteration.
Some still decline outright: inside an INLINED body (half an inline cannot be
taken back), and where no deopt site can be built. Adding an opcode is therefore
still a JIT ADMISSION decision before it is anything else, and the cost of
getting it wrong has been measured twice:

  * `OP_CMP_LOCAL_CONST_LT` shipped with no arm. 906ms against 26ms compiled on
    a loop of that shape -- 34.8x -- and it went unnoticed because no benchmark
    happened to contain it.
  * `OP_ELEM_KIND` shipped with no arm. `tests/bench/sort_merge`'s `merge`, whose
    body is pushes onto a `list[int]`, declined whole: 270ms -> 510ms. That is a
    2% suite regression from one missing arm.

Two rules, because the two kinds of opcode differ in how they arrive:

1. **Fused opcodes must ALWAYS have an arm.** The peephole synthesises them into
   exactly the hot loops that most want compiling, so one without an arm is
   worse than not fusing at all. No exceptions list.

2. **Every other opcode is a RATCHET** against tests/vm/jit_unarmed.baseline.
   67 of them have no arm today and that is not a bug to fix in one go -- but a
   NEW one is a decision, and this makes it a deliberate one: add the arm, or
   add the opcode to the baseline with a reason in the commit. The baseline's
   own header says what being on it costs now.

An opcode that GAINS an arm is never an error; the check says so and asks for
the baseline line to be dropped.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHUNK_C = ROOT / "src/vm/bytecode/chunk.c"
JIT_C = ROOT / "src/vm/jit/jit_func.c"
BASELINE = ROOT / "tests/vm/jit_unarmed.baseline"

# Opcodes the PEEPHOLE synthesises. Listed explicitly rather than inferred:
# "is this fused" is a judgement about intent, and a wrong guess would either
# wave through a real gap or block a build for nothing.
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
    "OP_FOR_ITER_PAIR",
]


def declared_opcodes():
    body = CHUNK_C.read_text()
    body = body.split("#define JAI_OPCODES(X)", 1)[1].split("#define X_NAME", 1)[0]
    return re.findall(r"X\(\s*(OP_[A-Z0-9_]+)", body)


def armed_opcodes():
    return set(re.findall(r"case\s+(OP_[A-Z0-9_]+)\s*:", JIT_C.read_text()))


def main():
    declared = declared_opcodes()
    armed = armed_opcodes()
    problems = []
    notes = []

    if not declared:
        problems.append("chunk.c: JAI_OPCODES parsed as empty -- regex is stale")
    if not armed:
        problems.append("jit_func.c: no `case OP_...:` found -- regex is stale")

    for name in FUSED:
        if name not in declared:
            problems.append(f"{name} is in this script's FUSED list but not in "
                            f"chunk.c -- the list is stale")
        elif name not in armed:
            problems.append(
                f"{name} is FUSED and has no `case {name}:` in "
                f"src/vm/jit/jit_func.c -- every function the peephole puts it "
                f"in will decline WHOLE. A fused opcode has no baseline "
                f"exemption; write the arm.")

    baseline = set()
    if BASELINE.exists():
        baseline = {l.strip() for l in BASELINE.read_text().splitlines()
                    if l.strip() and not l.startswith("#")}
    else:
        problems.append(f"missing {BASELINE.relative_to(ROOT)} -- regenerate it")

    unarmed = [o for o in declared if o not in armed]
    for name in unarmed:
        if name in FUSED:
            continue          # already reported above, with a sharper message
        if name not in baseline:
            problems.append(
                f"{name} has no arm in src/vm/jit/jit_func.c and is not in "
                f"{BASELINE.relative_to(ROOT)} -- every function containing it "
                f"will decline WHOLE. Write the arm, or add it to the baseline "
                f"and say why in the commit.")

    for name in sorted(baseline - set(unarmed)):
        if name in declared:
            notes.append(f"{name} now has an arm -- drop it from "
                         f"{BASELINE.relative_to(ROOT)}")
        else:
            notes.append(f"{name} is in the baseline but no longer exists -- "
                         f"drop it from {BASELINE.relative_to(ROOT)}")

    for n in notes:
        print(f"note: {n}")

    if problems:
        print(f"jit arm check FAILED ({len(problems)} problem(s)):")
        for p in problems:
            print(f"  {p}")
        return 1

    print(f"jit arm check ok: {len(FUSED)} fused opcodes all armed; "
          f"{len(armed & set(declared))}/{len(declared)} opcodes armed, "
          f"{len(unarmed)} known-unarmed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
