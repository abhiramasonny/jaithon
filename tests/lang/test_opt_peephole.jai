#: `jaithon.compile.opt.peephole`, pass 3 of `spec/BYTECODE.md` §8.
#:
#: The chunks here are written by hand rather than compiled from source. A pass
#: is only correct if the *bytes* it produces are the bytes the C optimiser
#: produces, and a test that first has to find a source program exercising a
#: window would be testing the emitter's lowering choices instead of the
#: rewrite. Each case therefore states an instruction sequence, runs the pass
#: over it and asserts the re-encoded chunk byte for byte, jump operands
#: included: code that is different but still works is exactly the failure
#: `make bootstrap` catches and program output does not.
#:
#: The pass alternates its scan with `mark_unreachable`, so a rewrite that
#: strands a block settles into more than one change; the counts asserted here
#: are the settled totals.
from std.test import assert_eq, assert_true
from jaithon.compile.emit import FuncProto, Op, opcode
from jaithon.compile.opt.chunk import compact, decode, kill
from jaithon.compile.opt.peephole import peephole

fn _one(op: Op) -> list[int] {
    return [opcode(op)]
}

fn _jump(op: Op, delta: int) -> list[int] {
    return [opcode(op), delta & 0xFF, delta >> 8 & 0xFF]
}

fn _one_u8(op: Op, value: int) -> list[int] {
    return [opcode(op), value & 0xFF]
}

fn _one_u16(op: Op, value: int) -> list[int] {
    return [opcode(op), value & 0xFF, value >> 8 & 0xFF]
}

fn _proto(code: list[int]) -> FuncProto {
    let proto = FuncProto("body", 0, 0)
    proto.max_slots = 2
    proto.chunk.code = code
    return proto
}

fn test_a_jump_to_the_next_instruction_is_deleted() -> void {
    let proto = _proto(_jump(Op.Jump, 0) + _one(Op.ReturnNull))
    assert_eq(peephole(proto), 1)
    assert_eq(proto.chunk.code, _one(Op.ReturnNull))
}

fn test_a_branch_to_a_jump_takes_the_final_target() -> void {
    let code = _one(Op.Null) +
        _jump(Op.JumpIfTrue, 4) +
        _one(Op.Null) +
        _jump(Op.JumpIfFalse, 3) +
        _jump(Op.Jump, 4) +
        _one(Op.Nop) +
        _jump(Op.Loop, -7) +
        _one(Op.ReturnNull)
    let proto = _proto(code)
    assert_eq(peephole(proto), 1)

    let expected = _one(Op.Null) +
        _jump(Op.JumpIfTrue, 11) +
        _one(Op.Null) +
        _jump(Op.JumpIfFalse, 3) +
        _jump(Op.Jump, 4) +
        _one(Op.Nop) +
        _jump(Op.Loop, -7) +
        _one(Op.ReturnNull)
    assert_eq(proto.chunk.code, expected)
}

fn test_a_conditional_over_a_lone_jump_is_inverted() -> void {
    let code = _one(Op.Null) +
        _jump(Op.JumpIfFalse, 3) +
        _jump(Op.Jump, 1) +
        _one(Op.Nop) +
        _one(Op.ReturnNull)
    let proto = _proto(code)
    assert_eq(peephole(proto), 1)

    let expected = _one(Op.Null) + _jump(Op.JumpIfTrue, 1) + _one(Op.Nop) + _one(Op.ReturnNull)
    assert_eq(proto.chunk.code, expected)
}

fn test_a_never_taken_constant_condition_loses_both_instructions() -> void {
    let code = _one(Op.True) + _jump(Op.JumpIfFalse, 1) + _one(Op.Nop) + _one(Op.ReturnNull)
    let proto = _proto(code)
    assert_eq(peephole(proto), 1)
    assert_eq(proto.chunk.code, _one(Op.Nop) + _one(Op.ReturnNull))
}

fn test_an_always_taken_constant_condition_becomes_a_jump() -> void {
    let code = _one(Op.True) + _jump(Op.JumpIfTrue, 1) + _one(Op.Nop) + _one(Op.ReturnNull)
    let proto = _proto(code)
    assert_eq(peephole(proto), 3)
    assert_eq(proto.chunk.code, _one(Op.ReturnNull))
}

fn test_a_keep_form_only_loses_the_jump() -> void {
    let code = _one(Op.False) + _jump(Op.JumpIfTrueKeep, 1) + _one(Op.Pop) + _one(Op.ReturnNull)
    let proto = _proto(code)
    assert_eq(peephole(proto), 2)
    assert_eq(proto.chunk.code, _one(Op.ReturnNull))
}

fn test_a_taken_keep_form_is_left_alone() -> void {
    let code = _one(Op.True) + _jump(Op.JumpIfTrueKeep, 1) + _one(Op.Pop) + _one(Op.ReturnNull)
    let expected = _one(Op.True) + _jump(Op.JumpIfTrueKeep, 1) + _one(Op.Pop) + _one(Op.ReturnNull)
    let proto = _proto(code)
    assert_eq(peephole(proto), 0)
    assert_eq(proto.chunk.code, expected)
}

fn test_not_in_front_of_a_jump_inverts_the_jump() -> void {
    let code = _one(Op.True) +
        _one(Op.True) +
        _one(Op.Eq) +
        _one(Op.Not) +
        _jump(Op.JumpIfFalse, 1) +
        _one(Op.Nop) +
        _one(Op.ReturnNull)
    let proto = _proto(code)
    assert_eq(peephole(proto), 1)

    let expected = _one(Op.True) +
        _one(Op.True) +
        _one(Op.Eq) +
        _jump(Op.JumpIfTrue, 1) +
        _one(Op.Nop) +
        _one(Op.ReturnNull)
    assert_eq(proto.chunk.code, expected)
}

fn test_not_is_kept_when_its_operand_is_not_a_bool() -> void {
    let code = _one(Op.Null) +
        _one(Op.Not) +
        _jump(Op.JumpIfFalse, 1) +
        _one(Op.Nop) +
        _one(Op.ReturnNull)
    let expected = _one(Op.Null) +
        _one(Op.Not) +
        _jump(Op.JumpIfFalse, 1) +
        _one(Op.Nop) +
        _one(Op.ReturnNull)
    let proto = _proto(code)
    assert_eq(peephole(proto), 0)
    assert_eq(proto.chunk.code, expected)
}

fn test_a_run_of_pops_becomes_one_popn() -> void {
    let code = _one(Op.Null) + _one(Op.Null) + _one(Op.Pop) + _one(Op.Pop) + _one(Op.ReturnNull)
    let proto = _proto(code)
    assert_eq(peephole(proto), 2)
    assert_eq(proto.chunk.code, _one(Op.ReturnNull))
}

fn test_a_popn_absorbs_the_pop_after_it() -> void {
    let code = _one_u8(Op.PopN, 254) + _one(Op.Pop) + _one(Op.ReturnNull)
    let proto = _proto(code)
    assert_eq(peephole(proto), 1)
    assert_eq(proto.chunk.code, _one_u8(Op.PopN, 255) + _one(Op.ReturnNull))
}

fn test_a_merge_that_would_overflow_the_u8_count_is_refused() -> void {
    let code = _one_u8(Op.PopN, 255) + _one(Op.Pop) + _one(Op.ReturnNull)
    let expected = _one_u8(Op.PopN, 255) + _one(Op.Pop) + _one(Op.ReturnNull)
    let proto = _proto(code)
    assert_eq(peephole(proto), 0)
    assert_eq(proto.chunk.code, expected)
}

fn test_a_branch_target_stops_a_pop_run_from_merging() -> void {
    let code = _one(Op.Null) +
        _jump(Op.JumpIfFalse, 1) +
        _one(Op.Pop) +
        _one(Op.Pop) +
        _one(Op.ReturnNull)
    let expected = _one(Op.Null) +
        _jump(Op.JumpIfFalse, 1) +
        _one(Op.Pop) +
        _one(Op.Pop) +
        _one(Op.ReturnNull)
    let proto = _proto(code)
    assert_eq(peephole(proto), 0)
    assert_eq(proto.chunk.code, expected)
}

fn test_dup_followed_by_pop_disappears() -> void {
    let code = _one(Op.Null) + _one(Op.Dup) + _one(Op.Pop) + _one(Op.ReturnNull)
    let proto = _proto(code)
    assert_eq(peephole(proto), 1)
    assert_eq(proto.chunk.code, _one(Op.Null) + _one(Op.ReturnNull))
}

fn test_set_local_followed_by_pop_becomes_bind() -> void {
    let code = _one(Op.Null) + _one_u16(Op.SetLocal, 1) + _one(Op.Pop) + _one(Op.ReturnNull)
    let proto = _proto(code)
    assert_eq(peephole(proto), 1)
    assert_eq(proto.chunk.code, _one(Op.Null) + _one_u16(Op.Bind, 1) + _one(Op.ReturnNull))
}

fn test_dup_followed_by_bind_becomes_set_local() -> void {
    let code = _one(Op.Null) + _one(Op.Dup) + _one_u16(Op.Bind, 1) + _one(Op.ReturnNull)
    let proto = _proto(code)
    assert_eq(peephole(proto), 1)
    assert_eq(proto.chunk.code, _one(Op.Null) + _one_u16(Op.SetLocal, 1) + _one(Op.ReturnNull))
}

fn test_compaction_moves_a_deleted_target_marker_to_its_survivor() -> void {
    let proto = _proto(_jump(Op.Jump, 0) + _one(Op.Nop) + _one(Op.ReturnNull))
    let code = decode(proto)
    if code == null {
        assert_true(false)
        return
    }
    kill(code, 1)
    compact(code)
    assert_eq(code.data[0].target, 1)
    assert_true(code.data[1].is_target)
}

fn test_an_unused_pure_push_is_removed_with_its_pop() -> void {
    let code = _one_u16(Op.Int, 42) + _one(Op.Pop) + _one(Op.ReturnNull)
    let proto = _proto(code)
    assert_eq(peephole(proto), 1)
    assert_eq(proto.chunk.code, _one(Op.ReturnNull))
}

fn test_a_loop_is_never_chased_through_or_retargeted() -> void {
    let code = _one(Op.Null) +
        _jump(Op.JumpIfTrue, 3) +
        _jump(Op.Jump, 3) +
        _jump(Op.Loop, -6) +
        _one(Op.ReturnNull)
    let expected = _one(Op.Null) +
        _jump(Op.JumpIfTrue, 3) +
        _jump(Op.Jump, 3) +
        _jump(Op.Loop, -6) +
        _one(Op.ReturnNull)
    let proto = _proto(code)
    assert_eq(peephole(proto), 0)
    assert_eq(proto.chunk.code, expected)
}

fn test_an_empty_chunk_is_left_alone() -> void {
    let proto = _proto([])
    assert_eq(peephole(proto), 0)
    assert_eq(proto.chunk.code, [])
}
