#: `jaithon.compile.opt.hoist`, pass 7 of `spec/BYTECODE.md` §8.
#:
#: The chunks are written by hand rather than compiled from source, for the
#: reason `test_opt_peephole.jai` gives: what this pass has to get right is the
#: bytes, and a case that first has to find a source program whose lowering
#: happens to contain the shape under test is really testing the emitter. Every
#: expectation below is a full byte sequence, jump operands included, because
#: a pre-header spliced in front of the wrong instruction still runs and still
#: prints the right answer.
#:
#: The three numbers that are not code are asserted too: the slot the load goes
#: into, `max_slots`, and `cache_count`, since a hoisted read takes a fresh
#: inline cache and orphans the one the in-loop read was using.
#:
#: The guards get a case each. Most of the pass is guards: a loop that anything
#: in it might re-enter the module table through, or whose header can be reached
#: without running the pre-header, must be left exactly as it was, and a test
#: that only proves the rewrite happens would not notice one of them going
#: missing.
from std.test import assert_eq, assert_true
from jaithon.compile.emit import CATCH_ALL, ExceptionEntry, FuncProto, Op, opcode
from jaithon.compile.opt.chunk import decode
from jaithon.compile.opt.hoist import hoist_globals

# One instruction with no operands.
fn _one(op: Op) -> list[int] {
    return [opcode(op)]
}

# A branch: the opcode and the i16 distance from the end of the instruction to
# its target, which is how every jump operand in this bytecode is measured.
fn _jump(op: Op, delta: int) -> list[int] {
    return [opcode(op), delta & 0xFF, delta >> 8 & 0xFF]
}

# An instruction carrying one u16 frame slot: `GET_LOCAL` and `SET_LOCAL`.
fn _local(op: Op, slot: int) -> list[int] {
    return [opcode(op), slot & 0xFF, slot >> 8 & 0xFF]
}

# `OP_GET_GLOBAL`: a u24 pool index naming the global, then its u16 cache slot.
fn _get_global(name: int, cache: int) -> list[int] {
    return [
        opcode(Op.GetGlobal),
        name & 0xFF,
        name >> 8 & 0xFF,
        name >> 16 & 0xFF,
        cache & 0xFF,
        cache >> 8 & 0xFF,
    ]
}

# A function record holding exactly `code`. Nothing indexes the constant pool
# here: the pass reads the u24 of a `GET_GLOBAL` as an identity and never looks
# the name up, so the pool can stay empty.
fn _proto(code: list[int], slots: int = 2, caches: int = 1, arity: int = 0) -> FuncProto {
    let proto = FuncProto("body", arity, 0)
    proto.max_slots = slots
    proto.chunk.code = code
    proto.chunk.cache_count = caches
    return proto
}

# The shape every positive case below is built on: a header that tests a local,
# a body, and a backward branch. `body` is laid out at offset 3 + `lead`.
#
#   GET_LOCAL 1 / JUMP_IF_FALSE -> exit / <body> / LOOP -> header / RETURN_NULL
fn _loop(body: list[int], lead: list[int] = []) -> list[int] {
    let head = lead.len()
    let back = head + 6 + body.len()
    return lead +
        _local(Op.GetLocal, 1) +
        _jump(Op.JumpIfFalse, body.len() + 3) +
        body +
        _jump(Op.Loop, -(back + 3 - head)) +
        _one(Op.ReturnNull)
}

fn test_one_global_moves_to_a_pre_header() -> void {
    # The read is the only thing in the body, so the rewritten loop is a
    # GET_LOCAL of the slot the pre-header just filled.
    let proto = _proto(_loop(_get_global(7, 0) + _one(Op.Pop)))
    assert_eq(hoist_globals(proto), 1)

    # Slot 2: slot 1 is the highest the code names and slot 0 is the callee.
    # Cache 1: the load takes a fresh one and cache 0 is orphaned with the read
    # that used it, which is what the C pass does.
    let expected = _get_global(7, 1) +
        _local(Op.SetLocal, 2) +
        _one(Op.Pop) +
        _local(Op.GetLocal, 1) +
        _jump(Op.JumpIfFalse, 7) +
        _local(Op.GetLocal, 2) +
        _one(Op.Pop) +
        _jump(Op.Loop, -13) +
        _one(Op.ReturnNull)
    assert_eq(proto.chunk.code, expected)
    assert_eq(proto.max_slots, 3)
    assert_eq(proto.chunk.cache_count, 2)
}

fn test_the_backward_branch_still_lands_on_the_header() -> void {
    # Stated on its own because it is the one thing a pre-header can break and
    # still produce a program that runs: an edge that landed on the loop must
    # not now land on the load, or the global is re-read every iteration and
    # the pass has bought nothing.
    let proto = _proto(_loop(_get_global(7, 0) + _one(Op.Pop)))
    assert_eq(hoist_globals(proto), 1)

    # The LOOP ends at offset 23 and names offset 10, which is the GET_LOCAL
    # the header became, not the GET_GLOBAL at offset 0.
    let back = proto.chunk.code.len() - 4
    assert_eq(proto.chunk.code[back:], _jump(Op.Loop, -13) + _one(Op.ReturnNull))
}

fn test_two_globals_take_a_slot_each_and_a_repeat_shares_one() -> void {
    # Read order is slot order, and the second read of `7` is rewritten to the
    # slot the first one claimed rather than taking one of its own.
    let body = _get_global(7, 0) +
        _one(Op.Pop) +
        _get_global(9, 1) +
        _one(Op.Pop) +
        _get_global(7, 2) +
        _one(Op.Pop)
    let proto = _proto(_loop(body), 2, 3)
    assert_eq(hoist_globals(proto), 2)

    let expected = _get_global(7, 3) +
        _local(Op.SetLocal, 2) +
        _one(Op.Pop) +
        _get_global(9, 4) +
        _local(Op.SetLocal, 3) +
        _one(Op.Pop) +
        _local(Op.GetLocal, 1) +
        _jump(Op.JumpIfFalse, 15) +
        _local(Op.GetLocal, 2) +
        _one(Op.Pop) +
        _local(Op.GetLocal, 3) +
        _one(Op.Pop) +
        _local(Op.GetLocal, 2) +
        _one(Op.Pop) +
        _jump(Op.Loop, -21) +
        _one(Op.ReturnNull)
    assert_eq(proto.chunk.code, expected)
    assert_eq(proto.max_slots, 4)
    assert_eq(proto.chunk.cache_count, 5)
}

fn test_a_loop_takes_eight_globals_a_round() -> void {
    # `OPT_MAX_HOIST_PER_LOOP` is 8, so a loop reading nine distinct globals
    # needs a second round for the ninth. The pass answers the total, and the
    # ninth pre-header lands after the eight the first round spliced in,
    # because both went immediately in front of the same header.
    var body: list[int] = []
    for name in 0..9 { body = body + _get_global(100 + name, name) + _one(Op.Pop) }
    let proto = _proto(_loop(body), 2, 9)
    assert_eq(hoist_globals(proto), 9)

    var loads: list[int] = []
    var reads: list[int] = []
    for which in 0..9 {
        loads = loads +
            _get_global(100 + which, 9 + which) +
            _local(Op.SetLocal, 2 + which) +
            _one(Op.Pop)
        reads = reads + _local(Op.GetLocal, 2 + which) + _one(Op.Pop)
    }
    let expected = loads +
        _local(Op.GetLocal, 1) +
        _jump(Op.JumpIfFalse, 39) +
        reads +
        _jump(Op.Loop, -45) +
        _one(Op.ReturnNull)
    assert_eq(proto.chunk.code, expected)
    assert_eq(proto.max_slots, 11)
    assert_eq(proto.chunk.cache_count, 18)
}

fn test_two_sibling_loops_are_both_taken_in_one_round() -> void {
    # Disjoint loops do not overlap, so the second is still a candidate once
    # the first has been planned. The NOP is the instruction the first loop
    # exits onto, which is what lets the second header be fallen into.
    let first = _local(Op.GetLocal, 1) +
        _jump(Op.JumpIfFalse, 10) +
        _get_global(7, 0) +
        _one(Op.Pop) +
        _jump(Op.Loop, -16)
    let second = _local(Op.GetLocal, 1) +
        _jump(Op.JumpIfFalse, 10) +
        _get_global(9, 1) +
        _one(Op.Pop) +
        _jump(Op.Loop, -16)
    let proto = _proto(first + _one(Op.Nop) + second + _one(Op.ReturnNull), 2, 2)
    assert_eq(hoist_globals(proto), 2)

    let expected = _get_global(7, 2) +
        _local(Op.SetLocal, 2) +
        _one(Op.Pop) +
        _local(Op.GetLocal, 1) +
        _jump(Op.JumpIfFalse, 7) +
        _local(Op.GetLocal, 2) +
        _one(Op.Pop) +
        _jump(Op.Loop, -13) +
        _one(Op.Nop) +
        _get_global(9, 3) +
        _local(Op.SetLocal, 3) +
        _one(Op.Pop) +
        _local(Op.GetLocal, 1) +
        _jump(Op.JumpIfFalse, 7) +
        _local(Op.GetLocal, 3) +
        _one(Op.Pop) +
        _jump(Op.Loop, -13) +
        _one(Op.ReturnNull)
    assert_eq(proto.chunk.code, expected)
    assert_eq(proto.max_slots, 4)
    assert_eq(proto.chunk.cache_count, 4)
}

fn test_a_nested_loop_is_hoisted_into_the_outer_pre_header() -> void {
    # The outer loop is picked first because it is the wider one, and its body
    # contains the inner loop, so both reads leave through one pre-header. The
    # inner loop then overlaps a recorded plan and is not considered again.
    let inner = _local(Op.GetLocal, 1) +
        _jump(Op.JumpIfFalse, 10) +
        _get_global(9, 1) +
        _one(Op.Pop) +
        _jump(Op.Loop, -16)
    let proto = _proto(_loop(_get_global(7, 0) + _one(Op.Pop) + inner), 2, 2)
    assert_eq(hoist_globals(proto), 2)

    let expected = _get_global(7, 2) +
        _local(Op.SetLocal, 2) +
        _one(Op.Pop) +
        _get_global(9, 3) +
        _local(Op.SetLocal, 3) +
        _one(Op.Pop) +
        _local(Op.GetLocal, 1) +
        _jump(Op.JumpIfFalse, 20) +
        _local(Op.GetLocal, 2) +
        _one(Op.Pop) +
        _local(Op.GetLocal, 1) +
        _jump(Op.JumpIfFalse, 7) +
        _local(Op.GetLocal, 3) +
        _one(Op.Pop) +
        _jump(Op.Loop, -13) +
        _jump(Op.Loop, -26) +
        _one(Op.ReturnNull)
    assert_eq(proto.chunk.code, expected)
    assert_eq(proto.max_slots, 4)
    assert_eq(proto.chunk.cache_count, 4)
}

fn test_a_call_in_the_loop_blocks_the_whole_loop() -> void {
    # A call can rebind the global, so nothing in the loop is invariant. The
    # loop is skipped entirely rather than up to the call.
    let body = _get_global(7, 0) + [opcode(Op.Call), 0] + _one(Op.Pop)
    _assert_untouched(_proto(_loop(body)))
}

fn test_a_field_read_in_the_loop_blocks_the_whole_loop() -> void {
    # `OP_GET_FIELD` runs a property getter, which is user code and can reach
    # the module table exactly as a call can.
    let body = _get_global(7, 0) +
        _get_global(8, 1) +
        [opcode(Op.GetField), 0, 0, 0, 0, 0] +
        _one(Op.Pop)
    _assert_untouched(_proto(_loop(body), 2, 2))
}

fn test_a_global_written_in_the_loop_is_not_invariant() -> void {
    let body = _get_global(7, 0) + [opcode(Op.SetGlobal), 7, 0, 0, 1, 0] + _one(Op.Pop)
    _assert_untouched(_proto(_loop(body), 2, 2))
}

fn test_a_header_branched_to_from_outside_is_not_hoisted() -> void {
    # The forward branch enters the loop without running the pre-header, so
    # the slot would hold whatever the frame left in it.
    let body = _get_global(7, 0) + _one(Op.Pop)
    _assert_untouched(_proto(_loop(body, _jump(Op.Jump, 0))))
}

fn test_a_header_that_cannot_be_fallen_into_is_not_hoisted() -> void {
    # The instruction before the header is a JUMP, so control only ever
    # arrives at the header along the backward branch and a pre-header in
    # front of it would never run.
    let body = _get_global(7, 0) + _one(Op.Pop)
    _assert_untouched(_proto(_loop(body, _jump(Op.Jump, 16))))
}

fn test_a_header_that_starts_a_guarded_region_is_not_hoisted() -> void {
    # An exception edge is an entry the pre-header does not sit on. The same
    # chunk without the region is the case below, so this is the region and
    # nothing else.
    let proto = _proto(_loop(_get_global(7, 0) + _one(Op.Pop), _one(Op.Nop)), 2, 1)
    proto.chunk.exceptions.push(ExceptionEntry(1, 14, 17, CATCH_ALL))
    _assert_untouched(proto)
}

fn test_a_header_below_the_top_of_the_code_is_still_hoisted() -> void {
    # The control for the case above: the same loop, one instruction further
    # down, with nothing registering its header as a landing site.
    let proto = _proto(_loop(_get_global(7, 0) + _one(Op.Pop), _one(Op.Nop)), 2, 1)
    assert_eq(hoist_globals(proto), 1)

    let expected = _one(Op.Nop) +
        _get_global(7, 1) +
        _local(Op.SetLocal, 2) +
        _one(Op.Pop) +
        _local(Op.GetLocal, 1) +
        _jump(Op.JumpIfFalse, 7) +
        _local(Op.GetLocal, 2) +
        _one(Op.Pop) +
        _jump(Op.Loop, -13) +
        _one(Op.ReturnNull)
    assert_eq(proto.chunk.code, expected)
    assert_eq(proto.max_slots, 3)
}

fn test_a_hoisted_slot_never_lands_on_a_parameter() -> void {
    # Nothing in this loop names a slot, so the watermark comes from the
    # calling convention instead: the callee slot plus two parameters.
    let proto = _proto(_loop(_get_global(7, 0) + _one(Op.Pop)), 4, 1, 2)
    assert_eq(hoist_globals(proto), 1)

    let expected = _get_global(7, 1) +
        _local(Op.SetLocal, 4) +
        _one(Op.Pop) +
        _local(Op.GetLocal, 1) +
        _jump(Op.JumpIfFalse, 7) +
        _local(Op.GetLocal, 4) +
        _one(Op.Pop) +
        _jump(Op.Loop, -13) +
        _one(Op.ReturnNull)
    assert_eq(proto.chunk.code, expected)
    assert_eq(proto.max_slots, 5)
}

fn test_a_loop_with_no_global_read_is_left_alone() -> void {
    _assert_untouched(_proto(_loop(_local(Op.GetLocal, 1) + _one(Op.Pop))))
}

fn test_straight_line_code_is_left_alone() -> void {
    _assert_untouched(_proto(_get_global(7, 0) + _one(Op.Pop) + _one(Op.ReturnNull)))
}

# A pass that reports no changes has to leave every one of these exactly as it
# found it: the bytes, the frame size and the cache count, since a plan that was
# abandoned after taking a cache slot would show up nowhere else.
#
# The decode is asserted first. Every one of these cases is a refusal, and a
# chunk this substrate cannot read is also a refusal, so without it a case whose
# hand-written jump operand had drifted off an instruction boundary would pass
# while testing nothing at all.
fn _assert_untouched(proto: FuncProto) -> void {
    assert_true(decode(proto) != null, "the case is not a decodable chunk")
    let code = proto.chunk.code[:]
    let slots = proto.max_slots
    let caches = proto.chunk.cache_count
    assert_eq(hoist_globals(proto), 0)
    assert_eq(proto.chunk.code, code)
    assert_eq(proto.max_slots, slots)
    assert_eq(proto.chunk.cache_count, caches)
}
