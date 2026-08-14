#: `jaithon.compile.opt.fuse`, pass 6 of `spec/BYTECODE.md` §8.
#:
#: The chunks here are written by hand rather than compiled from source. A pass
#: is only correct if the *bytes* it produces are the bytes the C optimiser
#: produces, and a test that first has to find a source program exercising a
#: window would be testing the emitter's lowering choices instead of the fusion.
#: Each case therefore states an instruction sequence, runs the pass over it and
#: asserts the re-encoded chunk byte for byte, jump operands and pool indices
#: included: code that is different but still works is exactly the failure
#: `make bootstrap` catches and program output does not.
#:
#: The change count is asserted alongside the bytes because it is what
#: `jaithon.compile.opt.optimize` breaks its fixpoint loop on. A pass that
#: produced the right chunk and the wrong count would change how many rounds
#: every other pass gets.
from std.test import assert_eq, assert_true
from jaithon.compile.emit import Const, ConstTag, FuncProto, Op, opcode
from jaithon.compile.opt.fuse import superinstructions

fn _one(op: Op) -> list[int] {
    return [opcode(op)]
}

fn _u16(value: int) -> list[int] {
    return [value & 0xFF, value >> 8 & 0xFF]
}

fn _u24(value: int) -> list[int] {
    return [value & 0xFF, value >> 8 & 0xFF, value >> 16 & 0xFF]
}

fn _op16(op: Op, value: int) -> list[int] {
    return [opcode(op)] + _u16(value)
}

fn _proto(code: list[int]) -> FuncProto {
    let proto = FuncProto("body", 0, 0)
    proto.max_slots = 4
    proto.chunk.code = code
    return proto
}

fn _proto_with_ints(code: list[int], values: list[int]) -> FuncProto {
    let proto = _proto(code)
    for value in values { proto.chunk.add_constant(Const(ConstTag.Int, value)) }
    return proto
}

fn test_two_local_reads_become_one_instruction() -> void {
    let proto = _proto(_op16(Op.GetLocal, 1) + _op16(Op.GetLocal, 2) + _one(Op.ReturnNull))
    assert_eq(superinstructions(proto), 1)
    assert_eq(proto.chunk.code, [opcode(Op.GetLocal2)] + _u16(1) + _u16(2) + _one(Op.ReturnNull))
}

fn test_a_local_read_then_dup_is_the_same_slot_twice() -> void {
    let proto = _proto(_op16(Op.GetLocal, 3) + _one(Op.Dup) + _one(Op.ReturnNull))
    assert_eq(superinstructions(proto), 1)
    assert_eq(proto.chunk.code, [opcode(Op.GetLocal2)] + _u16(3) + _u16(3) + _one(Op.ReturnNull))
}

fn test_two_locals_and_an_add_go_straight_to_add_locals() -> void {
    let proto = _proto(
        _op16(Op.GetLocal, 1) +
            _op16(Op.GetLocal, 2) +
            _one(Op.Add) +
            _one(Op.ReturnNull)
    )
    assert_eq(superinstructions(proto), 1)
    assert_eq(proto.chunk.code, [opcode(Op.AddLocals)] + _u16(1) + _u16(2) + _one(Op.ReturnNull))
}

fn test_a_dup_form_chains_into_add_locals() -> void {
    let proto = _proto(_op16(Op.GetLocal, 2) + _one(Op.Dup) + _one(Op.Add) + _one(Op.ReturnNull))
    assert_eq(superinstructions(proto), 2)
    assert_eq(proto.chunk.code, [opcode(Op.AddLocals)] + _u16(2) + _u16(2) + _one(Op.ReturnNull))
}

fn test_a_local_plus_an_immediate_folds_into_the_operand() -> void {
    let proto = _proto(
        _op16(Op.GetLocal, 1) + _op16(Op.Int, 7) + _one(Op.Add) + _one(Op.ReturnNull)
    )
    assert_eq(superinstructions(proto), 1)
    assert_eq(proto.chunk.code, [opcode(Op.AddIntConst)] + _u16(1) + _u16(7) + _one(Op.ReturnNull))
}

fn test_a_pooled_int_folds_the_same_way() -> void {
    let proto = _proto_with_ints(
        _op16(Op.GetLocal, 1) +
            [opcode(Op.Const)] +
            _u24(0) +
            _one(Op.Add) +
            _one(Op.ReturnNull),
        [30000]
    )
    assert_eq(superinstructions(proto), 1)
    assert_eq(
        proto.chunk.code,
        [opcode(Op.AddIntConst)] + _u16(1) + _u16(30000) + _one(Op.ReturnNull)
    )
}

fn test_a_constant_too_wide_for_the_immediate_is_left_alone() -> void {
    let code = _op16(Op.GetLocal, 1) +
        [opcode(Op.Const)] +
        _u24(0) +
        _one(Op.Add) +
        _one(Op.ReturnNull)
    let proto = _proto_with_ints(code, [40000])
    assert_eq(superinstructions(proto), 0)
    assert_eq(proto.chunk.code, code)
}

fn test_a_float_constant_is_not_an_integer_push() -> void {
    let code = _op16(Op.GetLocal, 1) +
        [opcode(Op.Const)] +
        _u24(0) +
        _one(Op.Add) +
        _one(Op.ReturnNull)
    let proto = _proto(code)
    proto.chunk.add_constant(Const(ConstTag.Float, 1.0))
    assert_eq(superinstructions(proto), 0)
    assert_eq(proto.chunk.code, code)
}

fn test_less_than_a_constant_becomes_one_comparison() -> void {
    let proto = _proto(_op16(Op.GetLocal, 1) + _op16(Op.Int, 7) + _one(Op.Lt) + _one(Op.ReturnNull))
    assert_eq(superinstructions(proto), 1)
    assert_eq(
        proto.chunk.code,
        [opcode(Op.CmpLocalConstLt)] + _u16(1) + _u16(7) + _one(Op.ReturnNull)
    )
}

fn test_only_less_than_has_an_immediate_form() -> void {
    let proto = _proto(_op16(Op.GetLocal, 1) + _op16(Op.Int, 7) + _one(Op.Gt) + _one(Op.ReturnNull))
    assert_eq(superinstructions(proto), 0)
}

fn test_an_increment_writes_its_slot_in_place() -> void {
    let proto = _proto(
        _op16(Op.GetLocal, 1) +
            _op16(Op.Int, 1) +
            _one(Op.Add) +
            _op16(Op.Bind, 1) +
            _one(Op.ReturnNull)
    )
    assert_eq(superinstructions(proto), 1)
    assert_eq(proto.chunk.code, [opcode(Op.IncLocal)] + _u16(1) + [1] + _one(Op.ReturnNull))
}

fn test_a_negative_delta_is_encoded_as_an_i8() -> void {
    let proto = _proto_with_ints(
        _op16(Op.GetLocal, 1) +
            [opcode(Op.Const)] +
            _u24(0) +
            _one(Op.Add) +
            _op16(Op.Bind, 1) +
            _one(Op.ReturnNull),
        [-5]
    )
    assert_eq(superinstructions(proto), 1)
    assert_eq(proto.chunk.code, [opcode(Op.IncLocal)] + _u16(1) + [0xFB] + _one(Op.ReturnNull))
}

fn test_a_store_to_another_slot_stops_at_add_int_const() -> void {
    let proto = _proto(
        _op16(Op.GetLocal, 1) +
            _op16(Op.Int, 1) +
            _one(Op.Add) +
            _op16(Op.Bind, 2) +
            _one(Op.ReturnNull)
    )
    assert_eq(superinstructions(proto), 1)
    assert_eq(
        proto.chunk.code,
        [opcode(Op.AddIntConst)] + _u16(1) + _u16(1) + _op16(Op.Bind, 2) + _one(Op.ReturnNull)
    )
}

fn test_a_delta_wider_than_i8_stops_at_add_int_const() -> void {
    let proto = _proto(
        _op16(Op.GetLocal, 1) +
            _op16(Op.Int, 200) +
            _one(Op.Add) +
            _op16(Op.Bind, 1) +
            _one(Op.ReturnNull)
    )
    assert_eq(superinstructions(proto), 1)
    assert_eq(
        proto.chunk.code,
        [opcode(Op.AddIntConst)] + _u16(1) + _u16(200) + _op16(Op.Bind, 1) + _one(Op.ReturnNull)
    )
}

fn test_a_local_minus_an_immediate_folds_without_changing_the_operator() -> void {
    let proto = _proto(
        _op16(Op.GetLocal, 1) + _op16(Op.Int, 1) + _one(Op.Sub) + _one(Op.ReturnNull)
    )
    assert_eq(superinstructions(proto), 1)
    assert_eq(proto.chunk.code, [opcode(Op.SubIntConst)] + _u16(1) + _u16(1) + _one(Op.ReturnNull))
}

fn test_a_pooled_subtrahend_folds_the_same_way() -> void {
    let proto = _proto_with_ints(
        _op16(Op.GetLocal, 2) +
            [opcode(Op.Const)] +
            _u24(0) +
            _one(Op.Sub) +
            _one(Op.ReturnNull),
        [-30000]
    )
    assert_eq(superinstructions(proto), 1)
    assert_eq(
        proto.chunk.code,
        [opcode(Op.SubIntConst)] + _u16(2) + _u16(-30000) + _one(Op.ReturnNull)
    )
}

fn test_a_field_read_off_a_local_keeps_its_name_and_cache() -> void {
    let proto = _proto(
        _op16(Op.GetLocal, 1) +
            [opcode(Op.GetField)] +
            _u24(6) +
            _u16(2) +
            _one(Op.ReturnNull)
    )
    assert_eq(superinstructions(proto), 1)
    assert_eq(
        proto.chunk.code,
        [opcode(Op.GetFieldLocal)] + _u16(1) + _u24(6) + _u16(2) + _one(Op.ReturnNull)
    )
}

fn test_for_iter_binds_straight_into_its_slot() -> void {
    let proto = _proto(
        _op16(Op.ForIter, 4) + _op16(Op.Bind, 2) + _one(Op.Nop) + _one(Op.ReturnNull)
    )
    assert_eq(superinstructions(proto), 1)
    assert_eq(
        proto.chunk.code,
        [opcode(Op.ForIterBind)] + _u16(1) + _u16(2) + _one(Op.Nop) + _one(Op.ReturnNull)
    )
}

fn test_a_comparison_folds_into_the_branch_that_reads_it() -> void {
    let proto = _proto(_one(Op.Lt) + _op16(Op.JumpIfFalse, 1) + _one(Op.Nop) + _one(Op.ReturnNull))
    assert_eq(superinstructions(proto), 1)
    assert_eq(
        proto.chunk.code,
        [opcode(Op.JumpIfCmpFalse), opcode(Op.Lt)] + _u16(1) + _one(Op.Nop) + _one(Op.ReturnNull)
    )
}

fn test_the_keep_forms_never_fuse_a_comparison() -> void {
    let code = _one(Op.Lt) + _op16(Op.JumpIfFalseKeep, 1) + _one(Op.Nop) + _one(Op.ReturnNull)
    let proto = _proto(code)
    assert_eq(superinstructions(proto), 0)
    assert_eq(proto.chunk.code, code)
}

fn test_a_guard_against_a_pooled_constant_is_one_dispatch() -> void {
    let proto = _proto_with_ints(
        _op16(Op.GetLocal, 1) +
            [opcode(Op.Const)] +
            _u24(0) +
            _one(Op.Lt) +
            _op16(Op.JumpIfFalse, 1) +
            _one(Op.Nop) +
            _one(Op.ReturnNull),
        [5000000]
    )
    assert_eq(superinstructions(proto), 1)
    assert_eq(
        proto.chunk.code,
        [opcode(Op.JumpIfCmpLocalK), opcode(Op.Lt)] +
            _u16(1) +
            _u24(0) +
            _u16(1) +
            _one(Op.Nop) +
            _one(Op.ReturnNull)
    )
    assert_eq(proto.chunk.constants.len(), 1)
}

fn test_an_immediate_guard_interns_its_constant() -> void {
    let proto = _proto(
        _op16(Op.GetLocal, 1) +
            _op16(Op.Int, 7) +
            _one(Op.Ge) +
            _op16(Op.JumpIfFalse, 1) +
            _one(Op.Nop) +
            _one(Op.ReturnNull)
    )
    assert_eq(superinstructions(proto), 1)
    assert_eq(
        proto.chunk.code,
        [opcode(Op.JumpIfCmpLocalK), opcode(Op.Ge)] +
            _u16(1) +
            _u24(0) +
            _u16(1) +
            _one(Op.Nop) +
            _one(Op.ReturnNull)
    )
    assert_eq(proto.chunk.constants.len(), 1)
    assert_true(proto.chunk.constants[0].tag == ConstTag.Int)
    assert_eq(proto.chunk.constants[0].value, 7)
}

fn test_a_half_fused_guard_is_caught_in_a_later_round() -> void {
    let proto = _proto_with_ints(
        _op16(Op.GetLocal, 1) +
            [opcode(Op.Const)] +
            _u24(0) +
            [opcode(Op.JumpIfCmpFalse), opcode(Op.Ne)] +
            _u16(1) +
            _one(Op.Nop) +
            _one(Op.ReturnNull),
        [9]
    )
    assert_eq(superinstructions(proto), 1)
    assert_eq(
        proto.chunk.code,
        [opcode(Op.JumpIfCmpLocalK), opcode(Op.Ne)] +
            _u16(1) +
            _u24(0) +
            _u16(1) +
            _one(Op.Nop) +
            _one(Op.ReturnNull)
    )
}

fn test_the_immediate_comparison_is_reclaimed_by_the_guard() -> void {
    let proto = _proto(
        [opcode(Op.CmpLocalConstLt)] +
            _u16(1) +
            _u16(7) +
            _op16(Op.JumpIfFalse, 1) +
            _one(Op.Nop) +
            _one(Op.ReturnNull)
    )
    assert_eq(superinstructions(proto), 1)
    assert_eq(
        proto.chunk.code,
        [opcode(Op.JumpIfCmpLocalK), opcode(Op.Lt)] +
            _u16(1) +
            _u24(0) +
            _u16(1) +
            _one(Op.Nop) +
            _one(Op.ReturnNull)
    )
    assert_eq(proto.chunk.constants[0].value, 7)
}

fn test_a_whole_guard_settles_from_the_raw_window_in_one_step() -> void {
    let proto = _proto(
        _op16(Op.GetLocal, 1) +
            _op16(Op.Int, 7) +
            _one(Op.Lt) +
            _op16(Op.JumpIfFalse, 1) +
            _one(Op.Nop) +
            _one(Op.ReturnNull)
    )
    assert_eq(superinstructions(proto), 1)
    assert_eq(
        proto.chunk.code,
        [opcode(Op.JumpIfCmpLocalK), opcode(Op.Lt)] +
            _u16(1) +
            _u24(0) +
            _u16(1) +
            _one(Op.Nop) +
            _one(Op.ReturnNull)
    )
}

fn test_a_branch_into_a_window_blocks_the_fusion() -> void {
    let code = _op16(Op.Jump, 3) +
        _op16(Op.GetLocal, 1) +
        _op16(Op.GetLocal, 2) +
        _one(Op.ReturnNull)
    let proto = _proto(code)
    assert_eq(superinstructions(proto), 0)
    assert_eq(proto.chunk.code, code)
}

fn test_a_branch_onto_a_window_head_still_fuses() -> void {
    let proto = _proto(
        _op16(Op.Jump, 0) +
            _op16(Op.GetLocal, 1) +
            _op16(Op.GetLocal, 2) +
            _one(Op.ReturnNull)
    )
    assert_eq(superinstructions(proto), 1)
    assert_eq(
        proto.chunk.code,
        _op16(Op.Jump, 0) + [opcode(Op.GetLocal2)] + _u16(1) + _u16(2) + _one(Op.ReturnNull)
    )
}

fn test_an_empty_chunk_is_left_alone() -> void {
    let proto = _proto([])
    assert_eq(superinstructions(proto), 0)
    assert_eq(proto.chunk.code, [])
}
