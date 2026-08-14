#: Superinstruction selection: pass 6 of `spec/BYTECODE.md` §8.
#:
#: The Jaithon half of `jaiOptSuperinstructions` (`src/codegen/optimize.c:895`):
#: a greedy longest match over the fused set of §3.3, rewriting the head of
#: every window in place and deleting the rest, so an edge landing on the head
#: still performs the whole sequence.
#:
#: Deliberately not part of the peephole: run there, it used to consume every
#: §3.3 window before branch rewrites finished, hiding them and merging §8's
#: two passes into one.
#:
#: Named `fuse.jai`, not `super.jai`: `super` is a keyword and can't be a
#: module path (E0804).
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
    let changes = superinstructions_on(code)
    if changes > 0 and not rebuild(code) { return 0 }
    return changes
}

#: The pass itself, on an already-decoded chunk. See `peephole_on` for why the
#: decode is the caller's job.
pub fn superinstructions_on(code: Code) -> int {
    var changes = 0
    var guard = 8 * code.count() + 64
    var index = first_live(code)
    while index >= 0 and guard > 0 {
        guard -= 1
        if not _fuse_at(code, index) {
            index = next_live(code, index)
            continue
        }
        changes += 1
        let back = prev_live(code, index)
        if back >= 0 { index = back }
    }
    return changes
}

fn _fuse_at(code: Code, index: int) -> bool {
    let w = window_of(code, index, 4)
    let n = w.len()
    let head = code.data[index]

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

    if n >= 2 and head.op == opcode(Op.AddIntConst) and window_clean(code, w, 2) {
        let store = code.data[w[1]]
        let delta = read_i16(head.operands, 2)
        if store.op == opcode(Op.Bind) and slot_of(store, 0) == slot_of(head, 0) and fits_i8(delta) {
            _make_inc_local(head, slot_of(head, 0), delta)
            kill(code, w[1])
            return true
        }
    }

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

    if n >= 2 and
        head.op == opcode(Op.GetLocal) and
        window_clean(code, w, 2) and
        code.data[w[1]].op == opcode(Op.GetField) {
        set_instr(
            head,
            opcode(Op.GetFieldLocal),
            u16_bytes(slot_of(head, 0)) + code.data[w[1]].operands
        )
        kill(code, w[1])
        return true
    }

    if n >= 4 and
        head.op == opcode(Op.ForIter) and
        window_clean(code, w, 4) and
        code.data[w[1]].op == opcode(Op.Unpack) and
        code.data[w[1]].operands[0] == 2 and
        code.data[w[1]].operands[1] == 255 and
        code.data[w[2]].op == opcode(Op.Bind) and
        code.data[w[3]].op == opcode(Op.Bind) {
        let target = head.target
        set_instr(
            head,
            opcode(Op.ForIterPair),
            u16_bytes(0) +
                u16_bytes(slot_of(code.data[w[2]], 0)) +
                u16_bytes(slot_of(code.data[w[3]], 0))
        )
        head.target = target
        kill_all(code, [w[1], w[2], w[3]])
        return true
    }

    if n >= 2 and
        head.op == opcode(Op.ForIter) and
        window_clean(code, w, 2) and
        code.data[w[1]].op == opcode(Op.Bind) {
        let target = head.target
        set_instr(
            head,
            opcode(Op.ForIterBind),
            u16_bytes(0) + u16_bytes(slot_of(code.data[w[1]], 0))
        )
        head.target = target
        kill(code, w[1])
        return true
    }

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

        if second.op == opcode(Op.GetLocal) and third == opcode(Op.Add) {
            _make_slot_pair(head, opcode(Op.AddLocals), slot_of(head, 0), slot_of(second, 0))
            kill_all(code, [w[1], w[2]])
            return true
        }

        let value = _pushed_int(code, second)
        if value != null and fits_i16(value) {
            if third == opcode(Op.Add) {
                _make_slot_imm16(head, opcode(Op.AddIntConst), slot_of(head, 0), value)
                kill_all(code, [w[1], w[2]])
                return true
            }
            if third == opcode(Op.Lt) {
                _make_slot_imm16(head, opcode(Op.CmpLocalConstLt), slot_of(head, 0), value)
                kill_all(code, [w[1], w[2]])
                return true
            }
        }
    }

    if n >= 2 and window_clean(code, w, 2) and code.data[w[1]].op == opcode(Op.Mod) {
        let divisor = _pushed_int(code, head)
        if divisor != null and divisor != 0 and fits_i16(divisor) {
            _make_imm16(head, opcode(Op.ModIntConst), divisor)
            kill(code, w[1])
            return true
        }
    }

    if n >= 2 and window_clean(code, w, 2) and code.data[w[1]].op == opcode(Op.Bind) {
        let fused = _bind_fused_form(head.op)
        if fused != null {
            _make_imm16(head, fused, slot_of(code.data[w[1]], 0))
            kill(code, w[1])
            return true
        }
    }

    if n >= 2 and
        head.op == opcode(Op.GetLocal2) and
        window_clean(code, w, 2) and
        code.data[w[1]].op == opcode(Op.Add) {
        head.op = opcode(Op.AddLocals)
        kill(code, w[1])
        return true
    }

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

fn _is_comparison(op: int) -> bool {
    return _COMPARISONS.get(op, false)
}

fn _pushed_int(code: Code, instruction: Instr) -> int? {
    if instruction.op == opcode(Op.Int) { return read_i16(instruction.operands, 0) }
    if instruction.op != opcode(Op.Const) { return null }
    let index = read_u24(instruction.operands, 0)
    if index >= code.chunk.constants.len() { return null }
    let value = code.chunk.constants[index]
    if value.tag != ConstTag.Int { return null }
    return value.value
}

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

fn _make_slot_pair(instruction: Instr, op: int, a: int, b: int) -> void {
    set_instr(instruction, op, u16_bytes(a) + u16_bytes(b))
}

fn _make_inc_local(instruction: Instr, slot: int, delta: int) -> void {
    set_instr(instruction, opcode(Op.IncLocal), u16_bytes(slot) + [delta & 0xFF])
}

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
