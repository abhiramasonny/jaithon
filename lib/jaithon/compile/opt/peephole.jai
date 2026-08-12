#: The peephole: pass 3 of `spec/BYTECODE.md` §8.
#:
#: The Jaithon half of `jaiOptPeephole` (`src/codegen/optimize.c:1198`).
#:
#: Fusion is deliberately not here: run first (as the C used to), it consumes
#: every §3.3 window before branch rewrites finish, hiding e.g. `JUMP_IF_FALSE
#: over a lone JUMP` from the rule that would delete the `JUMP`. Pass 3 finishes
#: the control flow and pass 6 fuses what is left.
#:
#: Constant *folding* isn't here either, and isn't a back-end job at all: the C
#: folds it in the type checker (`src/sema/check_fold.c`, spec §8 pass 1), on
#: the AST, before emission.
from jaithon.compile.emit import ConstTag, FuncProto, Op, opcode
from .chunk import Code, Instr, decode, fits_i16, first_live, kill, live_at_or_after
from .chunk import mark_unreachable, next_live, op_branch_is_edge, prev_live, read_u24
from .chunk import rebuild, set_instr, window_clean, window_of

#: Rounds the scan and `mark_unreachable` alternate for before giving up.
const SETTLE_ROUNDS: int = 8

#: Largest count `OP_POPN`'s `u8` operand can hold.
const MAX_POPN: int = 255

#: Rewrite every peephole window in `proto`; answer how many rewrites were made.
#:
#: Answers 0 and leaves the chunk untouched when the chunk cannot be decoded or
#: the result cannot be re-encoded.
pub fn peephole(proto: FuncProto) -> int {
    let code = decode(proto)
    if code == null { return 0 }
    let changes = peephole_on(code)
    if changes > 0 and not rebuild(code) { return 0 }
    return changes
}

#: The pass itself, on an already-decoded chunk.
#:
#: Decode/re-encode cost more than every rewrite here combined -- measured, one
#: round trip over the standard library is 317ms, and the optimiser used to do
#: about six of them per function. `opt/mod.jai` now decodes once and shares
#: the `Code`; this is the entry point a normal compile runs, and the wrapper
#: above is for callers holding only a proto.
pub fn peephole_on(code: Code) -> int {
    # Rewriting strands blocks, and removing a stranded block puts new
    # neighbours next to each other, so the two feed each other until neither
    # has anything left to do.
    var changes = 0
    for _round in 0..SETTLE_ROUNDS {
        let before = changes
        changes += _scan(code)
        changes += mark_unreachable(code)
        if changes == before { break }
    }
    return changes
}

fn _scan(code: Code) -> int {
    var changes = 0
    # Every rewrite but the retarget removes an instruction, and the retarget is
    # idempotent, so this terminates; the guard is insurance against a future
    # rule that is not.
    var guard = 8 * code.count() + 64
    var index = first_live(code)
    while index >= 0 and guard > 0 {
        guard -= 1
        if not _rewrite_at(code, index) {
            index = next_live(code, index)
            continue
        }
        changes += 1
        if not code.data[index].dead { continue }
        # Back up one: deleting an instruction can make its neighbours adjacent
        # and open a pattern that spans the hole.
        let resume = prev_live(code, index)
        index = if resume >= 0 { resume } else { next_live(code, index) }
    }
    return changes
}

fn _rewrite_at(code: Code, index: int) -> bool {
    if _rewrite_branch(code, index) { return true }
    if _rewrite_const_condition(code, index) { return true }
    if _rewrite_not_jump(code, index) { return true }
    return _rewrite_stack_ops(code, index)
}

# ---- control flow ----------------------------------------------------- #

# The three branch rewrites, in the order `rewriteBranchAt` tries them: a JUMP
# onto the instruction after it, a branch onto a JUMP, and a conditional whose
# only job is to skip a JUMP. At most one of them fires per call.
fn _rewrite_branch(code: Code, index: int) -> bool {
    let instruction = code.data[index]
    if instruction.target < 0 { return false }

    let follow = next_live(code, index)

    # JUMP to the next instruction is a no-op. Deleting it is safe even when it
    # is itself a target: every edge into it simply lands on the instruction it
    # would have jumped to.
    if instruction.op == opcode(Op.Jump) and
        follow >= 0 and
        live_at_or_after(code, instruction.target) == follow {
        kill(code, index)
        return true
    }

    # JUMP to a JUMP: go straight to the final target.
    if op_branch_is_edge(instruction.op) and instruction.op != opcode(Op.Loop) {
        let resolved = _chase_jump(code, instruction.target)
        if resolved < code.count() and
            resolved != live_at_or_after(code, instruction.target) and
            resolved != index and
            _retarget_fits(code, index, resolved) {
            instruction.target = resolved
            code.data[resolved].is_target = true
            return true
        }
    }

    # JUMP_IF_FALSE over a lone JUMP: invert the test and take the JUMP's
    # target. The head keeps its position, so an edge into it is unaffected.
    if _is_conditional_jump(instruction.op) and follow >= 0 {
        let over = code.data[follow]
        let after = next_live(code, follow)
        if over.op == opcode(Op.Jump) and
            over.target >= 0 and
            not over.is_target and
            after >= 0 and
            live_at_or_after(code, instruction.target) == after {
            # `count()` here means everything past the JUMP's target is deleted;
            # the C reads/writes one past its array's end in this case, and
            # refusing matches every input where that read is defined, since
            # `rebuild` rejects a branch off the end anyway.
            let landing = live_at_or_after(code, over.target)
            if landing < code.count() and _retarget_fits(code, index, landing) {
                instruction.op = _inverted_jump(instruction.op)
                instruction.target = landing
                code.data[landing].is_target = true
                kill(code, follow)
                return true
            }
        }
    }
    return false
}

# Follow a chain of unconditional JUMPs to its end. LOOP is never followed: it
# is the GC and interrupt safepoint (spec §10), and jumping over one would let a
# tight loop run uninterrupted. The step bound makes a jump cycle terminate;
# landing anywhere inside such a cycle is equivalent anyway.
fn _chase_jump(code: Code, target: int) -> int {
    var at = live_at_or_after(code, target)
    var steps = 0
    while steps < code.count() and at < code.count() {
        let instruction = code.data[at]
        if instruction.op != opcode(Op.Jump) or instruction.target < 0 { break }
        let landing = live_at_or_after(code, instruction.target)
        if landing == at { break }
        at = landing
        steps += 1
    }
    return at
}

# Bytes between a branch and its target in the current layout. Deletions only
# pull the two closer, so a value that fits now still fits after re-encoding;
# rewrites that would overflow i16 are rejected here rather than aborting the
# whole pass in `rebuild`.
fn _retarget_fits(code: Code, at: int, landing: int) -> bool {
    let here = code.data[at].new_offset + code.data[at].size()
    return fits_i16(code.data[landing].new_offset - here)
}

# ---- conditions ------------------------------------------------------- #

# A constant condition in front of a conditional jump. Both rewrites delete the
# push, whose effect the jump cancelled, and either delete the jump or demote it
# to an unconditional one, never the other way round, so the code only ever
# shrinks.
fn _rewrite_const_condition(code: Code, index: int) -> bool {
    let window = window_of(code, index, 2)
    if window.len() < 2 or not window_clean(code, window) { return false }

    let value = _pushed_bool(code, code.data[window[0]])
    if value == null { return false }
    let jump = code.data[window[1]]
    if jump.target < 0 { return false }

    # The KEEP forms leave the value on the stack, so only the never-taken case
    # can be simplified, and only the jump goes.
    if jump.op == opcode(Op.JumpIfFalseKeep) or jump.op == opcode(Op.JumpIfTrueKeep) {
        let keep_taken = if jump.op == opcode(Op.JumpIfFalseKeep) { not value } else { value }
        if keep_taken { return false }
        kill(code, window[1])
        return true
    }

    var taken = false
    if jump.op == opcode(Op.JumpIfFalse) {
        taken = not value
    } elif jump.op == opcode(Op.JumpIfTrue) {
        taken = value
    } else {
        return false
    }

    if not taken {
        kill(code, window[0])
        kill(code, window[1])
        return true
    }
    # Reuse the jump's slot: OP_JUMP is the same width as the conditional it
    # replaces, so nothing downstream moves.
    jump.op = opcode(Op.Jump)
    kill(code, window[0])
    return true
}

# NOT feeding a conditional jump: invert the jump instead. Both instructions
# change meaning for an incoming edge, so neither may be a branch target.
#
# Erasing the NOT looks free since the jump checks its condition is a bool too,
# but the two report the failure differently ("operand of 'not'" vs.
# "condition") and a program can catch and read that message. So the rewrite
# waits for a producer that is a bool by construction, where the check can
# never fire and which instruction would have run it stops mattering.
fn _rewrite_not_jump(code: Code, index: int) -> bool {
    let window = window_of(code, index, 2)
    if window.len() < 2 { return false }

    let head = code.data[window[0]]
    if head.op != opcode(Op.Not) or head.is_target { return false }
    let jump = code.data[window[1]]
    if jump.is_target { return false }
    if jump.op != opcode(Op.JumpIfFalse) and jump.op != opcode(Op.JumpIfTrue) { return false }

    let producer = prev_live(code, window[0])
    if producer < 0 or not _pushes_bool(code.data[producer].op) { return false }
    # An edge landing on the producer runs the NOT; one landing between it and
    # the NOT does not exist, since the NOT is not a target.
    jump.op = _inverted_jump(jump.op)
    kill(code, window[0])
    return true
}

# The boolean an instruction pushes, or null when it is not a constant push.
fn _pushed_bool(code: Code, instruction: Instr) -> bool? {
    if instruction.op == opcode(Op.True) { return true }
    if instruction.op == opcode(Op.False) { return false }
    if instruction.op != opcode(Op.Const) { return null }
    let index = read_u24(instruction.operands, 0)
    if index >= code.chunk.constants.len() { return null }
    let value = code.chunk.constants[index]
    if value.tag != ConstTag.Bool { return null }
    return bool(value.value)
}

# ---- stack traffic ---------------------------------------------------- #

# POP;POP;… -> POPN n, and DUP;POP -> nothing.
fn _rewrite_stack_ops(code: Code, index: int) -> bool {
    let head = code.data[index]

    if head.op == opcode(Op.Dup) {
        let window = window_of(code, index, 2)
        if window.len() == 2 and
            window_clean(code, window) and
            code.data[window[1]].op == opcode(Op.Pop) {
            kill(code, window[0])
            kill(code, window[1])
            return true
        }
        return false
    }

    if head.op != opcode(Op.Pop) and head.op != opcode(Op.PopN) { return false }

    var total = if head.op == opcode(Op.Pop) { 1 } else { head.operands[0] }
    var last = index
    var merged = 0
    var at = next_live(code, index)
    while at >= 0 {
        let instruction = code.data[at]
        if instruction.is_target { break }
        var amount = 0
        if instruction.op == opcode(Op.Pop) {
            amount = 1
        } elif instruction.op == opcode(Op.PopN) {
            amount = instruction.operands[0]
        } else {
            break
        }
        if total + amount > MAX_POPN { break }
        total += amount
        last = at
        merged += 1
        at = next_live(code, at)
    }
    if merged == 0 { return false }

    set_instr(head, opcode(Op.PopN), [total])
    var victim = next_live(code, index)
    while victim >= 0 and victim <= last {
        kill(code, victim)
        victim = next_live(code, victim)
    }
    return true
}

# ---- opcode tables ---------------------------------------------------- #

# The four conditional jumps and their opposites. One table serves both
# `isConditionalJump` and `invertedJump`: the ops that have an opposite are
# exactly the ops that branch on a condition.
let _INVERTED: dict[int, int] = _build_inverted()

# Instructions whose result is a bool whatever their operands were. Any of them
# satisfies OP_NOT's operand check on its own.
let _PUSHES_BOOL: dict[int, bool] = _build_pushes_bool()

fn _build_inverted() -> dict[int, int] {
    var table: dict[int, int] = {}
    table[opcode(Op.JumpIfFalse)] = opcode(Op.JumpIfTrue)
    table[opcode(Op.JumpIfTrue)] = opcode(Op.JumpIfFalse)
    table[opcode(Op.JumpIfFalseKeep)] = opcode(Op.JumpIfTrueKeep)
    table[opcode(Op.JumpIfTrueKeep)] = opcode(Op.JumpIfFalseKeep)
    return table
}

fn _build_pushes_bool() -> dict[int, bool] {
    var table: dict[int, bool] = {}
    let ops = [
        Op.True,
        Op.False,
        Op.Eq,
        Op.Ne,
        Op.Lt,
        Op.Le,
        Op.Gt,
        Op.Ge,
        Op.Is,
        Op.IsNot,
        Op.In,
        Op.NotIn,
        Op.Not,
        Op.IsInstance,
        Op.CmpLocalConstLt,
    ]
    for op in ops { table[opcode(op)] = true }
    return table
}

fn _is_conditional_jump(op: int) -> bool { return op in _INVERTED }

fn _inverted_jump(op: int) -> int { return _INVERTED.get(op, op) }

fn _pushes_bool(op: int) -> bool { return _PUSHES_BOOL.get(op, false) }
