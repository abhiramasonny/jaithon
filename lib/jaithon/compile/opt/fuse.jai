#: Superinstruction selection: pass 6 of `spec/BYTECODE.md` §8.
#:
#: The Jaithon half of `jaiOptSuperinstructions`
#: (`src/codegen/optimize.c:895`): a greedy longest match over the fused set of
#: §3.3, rewriting the head of every window in place and deleting the rest, so
#: an edge that lands on the window head still performs the whole sequence.
#:
#: Fusion is deliberately not part of the peephole. Running it there consumed
#: every §3.3 window before `peephole` had finished rewriting the control flow,
#: which both hid the branch rewrites and made §8's two passes one.
#:
#: The file is `fuse.jai` and not `super.jai` because `super` is a keyword and
#: cannot be a module path (E0804).
from jaithon.compile.emit import ConstTag, FuncProto, Op, opcode
from .chunk import Code, Instr, NO_INDEX, decode, first_live, fits_i16, fits_i8
from .chunk import kill, kill_all, next_live, pool_int_index, prev_live, read_i16
from .chunk import read_u24, rebuild, set_instr, slot_of, u16_bytes, u24_bytes
from .chunk import window_clean, window_of

#: Fuse every §3.3 window in `proto`; answer how many fusions were made.
#:
#: Answers 0 and leaves the chunk untouched when the chunk cannot be decoded or
#: the result cannot be re-encoded.
pub fn superinstructions(proto: FuncProto) -> int {
    let code = decode(proto)
    if code == null { return 0 }

    var changes = 0
    # Every fusion deletes at least one instruction, so the scan drains on its
    # own; the guard is insurance against a future rule that does not.
    var guard = 8 * code.count() + 64
    var index = first_live(code)
    while index >= 0 and guard > 0 {
        guard -= 1
        if not _fuse_at(code, index) {
            index = next_live(code, index)
            continue
        }
        changes += 1
        # The window head always survives a fusion, so staying at or before
        # `index` is what chains GET_LOCAL2 into ADD_LOCALS; backing up one lets
        # the closed hole open a window for the instruction in front of it.
        let back = prev_live(code, index)
        if back >= 0 { index = back }
    }

    if changes > 0 and not rebuild(code) { changes = 0 }
    return changes
}

# Greedy longest match at one instruction. The window is decoded once, four
# deep, and every rule below tests a prefix of it: `n` is how much of it exists
# and `window_clean`'s third argument is how much of it the rule rewrites.
fn _fuse_at(code: Code, index: int) -> bool {
    let w = window_of(code, index, 4)
    let n = w.len()
    let head = code.data[index]

    # GET_LOCAL a; <int k>; ADD; BIND a  ->  INC_LOCAL a, k.
    #
    # The store is OP_BIND, not SET_LOCAL; POP: the emitter uses BIND for every
    # store-and-discard, so the two rules this replaces matched nothing in the
    # whole tree while `i += 1`, the single hottest pair measured, went unfused.
    #
    # SUB is *not* folded in as INC_LOCAL a, -k, which is what the dead rule
    # used to say. It is only the same instruction for numbers: on anything else
    # both raise, but INC_LOCAL's slow path adds, so `y - 1` on a str reported
    # "unsupported operand types for '+'" from -O1 up and for '-' at -O0. A
    # fused form may be a fast path and nothing else. Folding `-=` back in needs
    # an operator byte on INC_LOCAL.
    if n >= 4 and head.op == opcode(Op.GetLocal) and window_clean(code, w, 4) {
        let arith = code.data[w[2]]
        let store = code.data[w[3]]
        if arith.op == opcode(Op.Add) and
            store.op == opcode(Op.Bind) and
            slot_of(store, 0) == slot_of(head, 0) {
            let value = _pushed_int(code, code.data[w[1]])
            if value != null and fits_i8(value) {
                _make_inc_local(head, slot_of(head, 0), value)
                kill_all(code, [w[1], w[2], w[3]])
                return true
            }
        }
    }

    # ADD_INT_CONST a, k; BIND a  ->  INC_LOCAL a, k. Catches the increment left
    # behind when the three-instruction form fused first.
    if n >= 2 and head.op == opcode(Op.AddIntConst) and window_clean(code, w, 2) {
        let store = code.data[w[1]]
        let delta = read_i16(head.operands, 2)
        if store.op == opcode(Op.Bind) and slot_of(store, 0) == slot_of(head, 0) and fits_i8(delta) {
            _make_inc_local(head, slot_of(head, 0), delta)
            kill(code, w[1])
            return true
        }
    }

    # A local compared against a constant, branched on:
    #
    #     GET_LOCAL a; <const k>; <cmp>; JUMP_IF_FALSE J
    #     GET_LOCAL a; <const k>; JUMP_IF_CMP_FALSE cmp, J
    #     CMP_LOCAL_CONST_LT a, k; JUMP_IF_FALSE J
    #
    # all collapse to JUMP_IF_CMP_LOCAL_K cmp, a, k, J. That is `while i < N` and
    # `if n < 2`, a loop guard and a recursion base case, in one dispatch instead
    # of three. The three windows are the same shape caught at three stages: the
    # raw one, the one a previous round already half-fused into
    # JUMP_IF_CMP_FALSE, and the one the CMP_LOCAL_CONST_LT rule below claimed
    # first. The last two exist because fusion runs to a fixed point, so a rule
    # that only matched the raw form would lose every window whose head was
    # reached in a later round.
    #
    # K is a pool index rather than an immediate on purpose: the immediate form
    # OP_CMP_LOCAL_CONST_LT already exists and cannot express `i < 5_000_000`,
    # which is exactly the guard worth fusing.
    if n >= 4 and
        head.op == opcode(Op.GetLocal) and
        window_clean(code, w, 4) and
        _is_comparison(code.data[w[2]].op) and
        code.data[w[3]].op == opcode(Op.JumpIfFalse) {
        let k = _const_index_of(code, code.data[w[1]])
        if k != NO_INDEX {
            _make_jump_if_cmp_local_k(
                head,
                code.data[w[2]].op,
                slot_of(head, 0),
                k,
                code.data[w[3]].target
            )
            kill_all(code, [w[1], w[2], w[3]])
            return true
        }
    }

    if n >= 3 and
        head.op == opcode(Op.GetLocal) and
        window_clean(code, w, 3) and
        code.data[w[2]].op == opcode(Op.JumpIfCmpFalse) {
        let k = _const_index_of(code, code.data[w[1]])
        if k != NO_INDEX {
            _make_jump_if_cmp_local_k(
                head,
                code.data[w[2]].operands[0],
                slot_of(head, 0),
                k,
                code.data[w[2]].target
            )
            kill_all(code, [w[1], w[2]])
            return true
        }
    }

    if n >= 2 and
        head.op == opcode(Op.CmpLocalConstLt) and
        window_clean(code, w, 2) and
        code.data[w[1]].op == opcode(Op.JumpIfFalse) {
        let k = pool_int_index(code, read_i16(head.operands, 2))
        if k != NO_INDEX {
            _make_jump_if_cmp_local_k(
                head,
                opcode(Op.Lt),
                slot_of(head, 0),
                k,
                code.data[w[1]].target
            )
            kill(code, w[1])
            return true
        }
    }

    # GET_LOCAL a; GET_FIELD k, c  ->  GET_FIELD_LOCAL a, k, c. `self.x`: the
    # most frequently executed pair in the benchmark suite by a wide margin, and
    # the most frequent one statically too.
    if n >= 2 and
        head.op == opcode(Op.GetLocal) and
        window_clean(code, w, 2) and
        code.data[w[1]].op == opcode(Op.GetField) {
        # The u16 slot, then GET_FIELD's own u24 name and u16 cache verbatim.
        set_instr(
            head,
            opcode(Op.GetFieldLocal),
            u16_bytes(slot_of(head, 0)) + code.data[w[1]].operands
        )
        kill(code, w[1])
        return true
    }

    # FOR_ITER J; BIND a  ->  FOR_ITER_BIND J, a. The loop variable goes straight
    # to its slot. The jump operand keeps byte index 0, so `rebuild` resolves the
    # branch with no special case.
    if n >= 2 and
        head.op == opcode(Op.ForIter) and
        window_clean(code, w, 2) and
        code.data[w[1]].op == opcode(Op.Bind) {
        let target = head.target
        set_instr(
            head,
            opcode(Op.ForIterBind),
            # The first u16 is patched by `rebuild` from `target`.
            u16_bytes(0) + u16_bytes(slot_of(code.data[w[1]], 0))
        )
        head.target = target
        kill(code, w[1])
        return true
    }

    # <cmp>; JUMP_IF_FALSE J  ->  JUMP_IF_CMP_FALSE cmp, J. Only the popping form
    # fuses: the *_KEEP jumps leave the bool behind for `and`/`or`.
    if n >= 2 and
        _is_comparison(head.op) and
        window_clean(code, w, 2) and
        code.data[w[1]].op == opcode(Op.JumpIfFalse) {
        let target = code.data[w[1]].target
        set_instr(head, opcode(Op.JumpIfCmpFalse), [head.op] + u16_bytes(0))
        head.target = target
        kill(code, w[1])
        return true
    }

    if n >= 3 and head.op == opcode(Op.GetLocal) and window_clean(code, w, 3) {
        let second = code.data[w[1]]
        let third = code.data[w[2]].op

        # GET_LOCAL a; GET_LOCAL b; ADD  ->  ADD_LOCALS a, b
        if second.op == opcode(Op.GetLocal) and third == opcode(Op.Add) {
            _make_slot_pair(head, opcode(Op.AddLocals), slot_of(head, 0), slot_of(second, 0))
            kill_all(code, [w[1], w[2]])
            return true
        }

        let value = _pushed_int(code, second)
        if value != null and fits_i16(value) {
            # GET_LOCAL a; <int k>; ADD  ->  ADD_INT_CONST a, k
            if third == opcode(Op.Add) {
                _make_slot_imm16(head, opcode(Op.AddIntConst), slot_of(head, 0), value)
                kill_all(code, [w[1], w[2]])
                return true
            }
            # GET_LOCAL a; <int k>; LT  ->  CMP_LOCAL_CONST_LT a, k
            if third == opcode(Op.Lt) {
                _make_slot_imm16(head, opcode(Op.CmpLocalConstLt), slot_of(head, 0), value)
                kill_all(code, [w[1], w[2]])
                return true
            }
        }
    }

    # <int k>; MOD  ->  MOD_INT_CONST k, wherever the pair appears: the divisor
    # is an immediate and no slot is involved, so this fires after a GET_LOCAL2
    # as readily as after a GET_LOCAL. `loop_sum` computes `i % 7` every
    # iteration, one of its eight dispatches.
    #
    # k == 0 is left alone: the fused form treats the divisor as known-good, and
    # division by zero has to keep reporting itself from OP_MOD.
    if n >= 2 and window_clean(code, w, 2) and code.data[w[1]].op == opcode(Op.Mod) {
        let divisor = _pushed_int(code, head)
        if divisor != null and divisor != 0 and fits_i16(divisor) {
            _make_imm16(head, opcode(Op.ModIntConst), divisor)
            kill(code, w[1])
            return true
        }
    }

    # ADD; BIND a  ->  ADD_BIND a. The pair's net stack effect is unchanged, so
    # neither operand's origin matters. It removes the push-then-pop between
    # them, which every accumulating `x += ...` performs.
    if n >= 2 and window_clean(code, w, 2) and code.data[w[1]].op == opcode(Op.Bind) {
        let fused = _bind_fused_form(head.op)
        if fused != null {
            _make_imm16(head, fused, slot_of(code.data[w[1]], 0))
            kill(code, w[1])
            return true
        }
    }

    # GET_LOCAL2 a, b; ADD  ->  ADD_LOCALS a, b. The operand pair is already the
    # right shape, so only the opcode byte changes.
    if n >= 2 and
        head.op == opcode(Op.GetLocal2) and
        window_clean(code, w, 2) and
        code.data[w[1]].op == opcode(Op.Add) {
        head.op = opcode(Op.AddLocals)
        kill(code, w[1])
        return true
    }

    # GET_LOCAL a; GET_LOCAL b  ->  GET_LOCAL2 a, b, and GET_LOCAL a; DUP, which
    # pushes the same slot twice and is therefore GET_LOCAL2 a, a. The DUP form
    # is how a compound field assignment (`b.vx -= …`) starts, which makes it 6%
    # of every instruction nbody executes.
    if n >= 2 and head.op == opcode(Op.GetLocal) and window_clean(code, w, 2) {
        let second = code.data[w[1]]
        if second.op == opcode(Op.GetLocal) or second.op == opcode(Op.Dup) {
            let a = slot_of(head, 0)
            let b = if second.op == opcode(Op.Dup) { a } else { slot_of(second, 0) }
            _make_slot_pair(head, opcode(Op.GetLocal2), a, b)
            kill(code, w[1])
            return true
        }
    }
    return false
}

let _COMPARISONS: dict[int, bool] = _build_comparisons()

fn _build_comparisons() -> dict[int, bool] {
    var table: dict[int, bool] = {}
    for op in [Op.Eq, Op.Ne, Op.Lt, Op.Le, Op.Gt, Op.Ge] { table[opcode(op)] = true }
    return table
}

# True when `op` leaves exactly one bool on the stack for every input, which is
# what lets OP_JUMP_IF_CMP_FALSE swallow the jump's own bool check.
fn _is_comparison(op: int) -> bool {
    return _COMPARISONS.get(op, false)
}

# The integer an instruction pushes, for the fused forms that fold a constant
# into their operand. Both the immediate and the pool encodings qualify; null
# means the instruction pushes something else, or nothing.
fn _pushed_int(code: Code, instruction: Instr) -> int? {
    if instruction.op == opcode(Op.Int) { return read_i16(instruction.operands, 0) }
    if instruction.op != opcode(Op.Const) { return null }
    let index = read_u24(instruction.operands, 0)
    if index >= code.chunk.constants.len() { return null }
    let value = code.chunk.constants[index]
    if value.tag != ConstTag.Int { return null }
    return value.value
}

# The pool index of the constant `instruction` pushes, for the fused forms that
# name a constant by index rather than folding it into an immediate. `OP_INT`
# carries its value inline and so may have to be interned first, which is the
# one thing in this pass that can grow the constant pool.
fn _const_index_of(code: Code, instruction: Instr) -> int {
    if instruction.op == opcode(Op.Const) {
        let index = read_u24(instruction.operands, 0)
        if index >= code.chunk.constants.len() { return NO_INDEX }
        return index
    }
    if instruction.op == opcode(Op.Int) {
        return pool_int_index(code, read_i16(instruction.operands, 0))
    }
    return NO_INDEX
}

# u16 slot, i16 immediate: the shape of ADD_INT_CONST and CMP_LOCAL_CONST_LT.
# A bare i16 immediate: the shape of MOD_INT_CONST.
# The fused `<binop>; BIND a` form of `op`, or null when there is none.
#
# Every opcode answered here carries a slot, so every one of them also has to
# be listed in the coalescer's slot-operand table (`_build_slot_operands` in
# chunk.jai). Leaving one out renumbers the readers of a slot and not the fused
# writer, which is a miscompile three steps removed from its cause.
fn _bind_fused_form(op: int) -> int? {
    if op == opcode(Op.Add) { return opcode(Op.AddBind) }
    if op == opcode(Op.Sub) { return opcode(Op.SubBind) }
    if op == opcode(Op.Mul) { return opcode(Op.MulBind) }
    return null
}

fn _make_imm16(instruction: Instr, op: int, immediate: int) -> void {
    set_instr(instruction, op, u16_bytes(immediate))
}

fn _make_slot_imm16(instruction: Instr, op: int, slot: int, immediate: int) -> void {
    set_instr(instruction, op, u16_bytes(slot) + u16_bytes(immediate))
}

# u16, u16: the shape of GET_LOCAL2 and ADD_LOCALS.
fn _make_slot_pair(instruction: Instr, op: int, a: int, b: int) -> void {
    set_instr(instruction, op, u16_bytes(a) + u16_bytes(b))
}

# u16 slot, i8 delta.
fn _make_inc_local(instruction: Instr, slot: int, delta: int) -> void {
    set_instr(instruction, opcode(Op.IncLocal), u16_bytes(slot) + [delta & 0xFF])
}

# The branch operand is left zero and `target` carried on the `Instr`, which is
# what `rebuild` resolves branches from; the encoder needs no special case
# because `branch_operand_at` already knows where the i16 sits.
fn _make_jump_if_cmp_local_k(
    instruction: Instr,
    comparison: int,
    slot: int,
    k: int,
    target: int
) -> void {
    set_instr(
        instruction,
        opcode(Op.JumpIfCmpLocalK),
        [comparison] + u16_bytes(slot) + u24_bytes(k) + u16_bytes(0)
    )
    instruction.target = target
}
