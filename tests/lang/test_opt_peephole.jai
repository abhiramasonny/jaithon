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
from std.test import assert_eq
from jaithon.compile.emit import FuncProto, Op, opcode
from jaithon.compile.opt.peephole import peephole

# One instruction with no operands.
fn _one(op: Op) -> list[int] {
    return [opcode(op)]
}

# A branch: the opcode and the i16 distance from the end of the instruction to
# its target, which is how every jump operand in this bytecode is measured.
fn _jump(op: Op, delta: int) -> list[int] {
    return [opcode(op), delta & 0xFF, delta >> 8 & 0xFF]
}

# One instruction with a single u8 operand.
fn _one_u8(op: Op, value: int) -> list[int] {
    return [opcode(op), value & 0xFF]
}

# A function record holding exactly `code`. There are no exception regions and
# no default thunks, so the only thing under test is the rewrite.
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
    # NULL / JUMP_IF_TRUE -> Y / NULL / JUMP_IF_FALSE -> W / Y: JUMP -> Z /
    # W: NOP / LOOP back to Y / Z: RETURN. The second conditional keeps W alive
    # and the loop keeps Y alive, so nothing here is deleted and the retarget of
    # the first conditional is the only change the pass can make.
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

    # The first conditional now names Z, eleven bytes past the end of its own
    # operand, and every other byte is untouched.
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
    # NULL / JUMP_IF_FALSE past the JUMP / JUMP -> Z / NOP / Z: RETURN becomes
    # NULL / JUMP_IF_TRUE -> Z / NOP / Z: RETURN.
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
    # The push goes and the conditional is demoted to OP_JUMP, which strands the
    # NOP. The alternation with `mark_unreachable` then removes the NOP, and the
    # jump that is left over now lands on the instruction after it, which the
    # first rule deletes: three changes for one folded condition.
    let code = _one(Op.True) + _jump(Op.JumpIfTrue, 1) + _one(Op.Nop) + _one(Op.ReturnNull)
    let proto = _proto(code)
    assert_eq(peephole(proto), 3)
    assert_eq(proto.chunk.code, _one(Op.ReturnNull))
}

fn test_a_keep_form_only_loses_the_jump() -> void {
    # The value stays on the stack, so a never-taken JUMP_IF_TRUE_KEEP may lose
    # its jump and nothing else.
    let code = _one(Op.False) + _jump(Op.JumpIfTrueKeep, 1) + _one(Op.Pop) + _one(Op.ReturnNull)
    let proto = _proto(code)
    assert_eq(peephole(proto), 1)
    assert_eq(proto.chunk.code, _one(Op.False) + _one(Op.Pop) + _one(Op.ReturnNull))
}

fn test_a_taken_keep_form_is_left_alone() -> void {
    # Deleting the push would take the value the jump was told to keep with it.
    let code = _one(Op.True) + _jump(Op.JumpIfTrueKeep, 1) + _one(Op.Pop) + _one(Op.ReturnNull)
    let expected = _one(Op.True) + _jump(Op.JumpIfTrueKeep, 1) + _one(Op.Pop) + _one(Op.ReturnNull)
    let proto = _proto(code)
    assert_eq(peephole(proto), 0)
    assert_eq(proto.chunk.code, expected)
}

fn test_not_in_front_of_a_jump_inverts_the_jump() -> void {
    # OP_EQ is a bool by construction, so erasing the NOT cannot change which
    # TypeError a bad operand would have raised.
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
    # OP_NULL pushes a value the NOT is the one to reject. Deleting it would
    # move the TypeError onto the jump and change the message a program can
    # catch and read.
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
    assert_eq(peephole(proto), 1)

    let expected = _one(Op.Null) + _one(Op.Null) + _one_u8(Op.PopN, 2) + _one(Op.ReturnNull)
    assert_eq(proto.chunk.code, expected)
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
    # The second POP is jumped to, so an edge arriving there must still pop
    # exactly one value; folding it into the run in front would pop two.
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

fn test_a_loop_is_never_chased_through_or_retargeted() -> void {
    # OP_LOOP is the GC and interrupt safepoint (spec §10). A branch that
    # skipped one would let a tight loop run uninterrupted, so a chain is
    # followed through OP_JUMP only and a LOOP is never itself retargeted.
    #
    # NULL / JUMP_IF_TRUE -> the LOOP / J: JUMP -> Z / LOOP back to J /
    # Z: RETURN. Chasing through the LOOP would move the conditional to Z, and
    # retargeting the LOOP would move it past J to Z; neither may happen.
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
