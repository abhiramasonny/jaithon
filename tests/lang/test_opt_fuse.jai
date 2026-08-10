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

# One instruction with no operands.
fn _one(op: Op) -> list[int] {
    return [opcode(op)]
}

# A little-endian u16, which is also how an i16 branch distance is stored.
fn _u16(value: int) -> list[int] {
    return [value & 0xFF, value >> 8 & 0xFF]
}

# A little-endian u24, the width of every constant index.
fn _u24(value: int) -> list[int] {
    return [value & 0xFF, value >> 8 & 0xFF, value >> 16 & 0xFF]
}

# An instruction whose whole operand run is one u16: a slot, or a branch
# measured from the end of the instruction to its target.
fn _op16(op: Op, value: int) -> list[int] {
    return [opcode(op)] + _u16(value)
}

# A function record holding exactly `code`. There are no exception regions and
# no default thunks, so the only thing under test is the fusion.
fn _proto(code: list[int]) -> FuncProto {
    let proto = FuncProto("body", 0, 0)
    proto.max_slots = 4
    proto.chunk.code = code
    return proto
}

# The same, with `values` interned first so that a rule reading OP_CONST has
# something to read. The index of each is its position in `values`.
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
    # How a compound field assignment starts: the receiver is loaded once and
    # duplicated, which is GET_LOCAL2 a, a.
    let proto = _proto(_op16(Op.GetLocal, 3) + _one(Op.Dup) + _one(Op.ReturnNull))
    assert_eq(superinstructions(proto), 1)
    assert_eq(proto.chunk.code, [opcode(Op.GetLocal2)] + _u16(3) + _u16(3) + _one(Op.ReturnNull))
}

fn test_two_locals_and_an_add_go_straight_to_add_locals() -> void {
    # The three-instruction rule claims this window before the GET_LOCAL2 rule
    # sees it, so it settles in one change and not two.
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
    # GET_LOCAL; DUP; ADD matches no three-instruction rule, so it fuses to
    # GET_LOCAL2 first. This is the case the scan backs up for: the second
    # fusion is only visible once the first has closed the hole.
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
    # Both encodings of an integer push qualify, so a value too wide for OP_INT
    # is still folded as long as it fits the i16 the fused form carries.
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
    # There is no CMP_LOCAL_CONST_GT, and an immediate push is neither a local
    # read nor a DUP, so nothing here matches at all.
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
    # INC_LOCAL folds its delta into one signed byte, so 200 cannot ride in it;
    # the wider fused form still applies and the store stays where it was.
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

fn test_subtraction_is_never_folded_into_an_increment() -> void {
    # INC_LOCAL's slow path adds, so folding `-=` into it would report
    # "unsupported operand types for '+'" for `y - 1` on a str. A fused form may
    # be a fast path and nothing else.
    #
    # `SUB; BIND` does fuse, into SUB_BIND, whose slow path subtracts. What must
    # not happen is the *increment*, which is why this pins the opcode rather
    # than merely counting fusions.
    let code = _op16(Op.GetLocal, 1) +
        _op16(Op.Int, 1) +
        _one(Op.Sub) +
        _op16(Op.Bind, 1) +
        _one(Op.ReturnNull)
    let proto = _proto(code)
    superinstructions(proto)
    assert_true(not proto.chunk.code.contains(opcode(Op.IncLocal)))
    assert_true(not proto.chunk.code.contains(opcode(Op.AddIntConst)))
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
    # FOR_ITER +4 skips BIND and NOP on exhaustion. The fused form is one byte
    # longer than FOR_ITER but swallows the three-byte BIND, so the surviving
    # distance is two bytes shorter.
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
    # `and` and `or` need the bool the comparison pushed, and the fused form
    # does not leave one behind.
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
    # Nothing was interned: the constant was already named by index.
    assert_eq(proto.chunk.constants.len(), 1)
}

fn test_an_immediate_guard_interns_its_constant() -> void {
    # K is a pool index and OP_INT carries its value inline, so the fusion has
    # to put it in the pool first. The pool is empty here, so it lands at 0.
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
    # What the previous round left behind when the comparison fused into the
    # branch before the local and the constant were reached.
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
    # CMP_LOCAL_CONST_LT claimed the window in an earlier round; the branch in
    # front of it takes it back, interning the immediate on the way.
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
    # The raw four-instruction window is matched before the immediate rule sees
    # its first three, so `if i < 7` never passes through CMP_LOCAL_CONST_LT.
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
    # The JUMP lands on the second GET_LOCAL, so fusing the pair would make that
    # edge push two locals instead of one.
    let code = _op16(Op.Jump, 3) +
        _op16(Op.GetLocal, 1) +
        _op16(Op.GetLocal, 2) +
        _one(Op.ReturnNull)
    let proto = _proto(code)
    assert_eq(superinstructions(proto), 0)
    assert_eq(proto.chunk.code, code)
}

fn test_a_branch_onto_a_window_head_still_fuses() -> void {
    # An edge landing on the first instruction of a window still performs the
    # whole window, so it fuses; the jump is re-resolved against the new layout.
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
