#: Local slot coalescing, pass 4 of `spec/BYTECODE.md` §8.
#:
#: These assert on *bytecode*, not on what a program prints: a pass that
#: renumbers slots differently from `jaiOptCoalesceSlots` and still computes the
#: right answer is a bootstrap failure, and only the emitted operands say so.
#:
#: Every case compiles at `--O0`, which runs dead-code elimination and nothing
#: else, and then calls `coalesce_slots` by hand. That is deliberate: at `--O2`
#: the peephole and the superinstruction passes have already rewritten the
#: window, so a test written against that stream would be asserting on their
#: behaviour as much as on this one's.
from jaithon.compile import compile_source
from jaithon.compile.emit import ConstTag, FuncProto, op_name
from jaithon.compile.opt.chunk import decode, slot_of, slot_operands
from jaithon.compile.opt.coalesce import coalesce_slots
from std.test import assert_eq, assert_ne

#: What one case looks like once `coalesce_slots` has run.
class Outcome {
    #: Slots the pass moved, which is its return value.
    pub let moved: int
    #: Frame window before and after.
    pub let slots_before: int
    pub let slots_after: int
    #: Chunk length before and after. Coalescing rewrites operands in place, so
    #: these must always agree; a difference means a pass rebuilt the chunk into
    #: something else.
    pub let bytes_before: int
    pub let bytes_after: int
    #: `"OPCODE slot"` for every slot operand of every live instruction, in code
    #: order, before and after.
    pub let before: list[str]
    pub let after: list[str]

    #: Record one result.
    pub fn init(
        self,
        moved: int,
        slots_before: int,
        slots_after: int,
        bytes_before: int,
        bytes_after: int,
        before: list[str],
        after: list[str]
    ) {
        self.moved = moved
        self.slots_before = slots_before
        self.slots_after = slots_after
        self.bytes_before = bytes_before
        self.bytes_after = bytes_after
        self.before = before
        self.after = after
    }
}

fn test_a_dead_local_gives_up_its_number() -> void {
    let result = coalesce(
        "fn f(n: int) -> int {\n" +
            "    let a = n + 1\n" +
            "    let b = a + 1\n" +
            "    let c = b + 1\n" +
            "    return c\n" +
            "}\n"
    )
    assert_eq(result.moved, 2)
    assert_eq(result.slots_before, 7)
    assert_eq(result.slots_after, 5)
    assert_eq(result.bytes_after, result.bytes_before)
    assert_eq(
        result.before,
        [
            "OP_GET_LOCAL 1",
            "OP_BIND 2",
            "OP_GET_LOCAL 2",
            "OP_BIND 3",
            "OP_GET_LOCAL 3",
            "OP_BIND 4",
            "OP_GET_LOCAL 4",
        ]
    )
    assert_eq(
        result.after,
        [
            "OP_GET_LOCAL 1",
            "OP_BIND 2",
            "OP_GET_LOCAL 2",
            "OP_BIND 2",
            "OP_GET_LOCAL 2",
            "OP_BIND 2",
            "OP_GET_LOCAL 2",
        ]
    )
}

fn test_one_number_hosts_several_tenants_in_turn() -> void {
    let result = coalesce(
        "fn f(n: int) -> int {\n" +
            "    let keep = n + 1\n" +
            "    let a = n + 2\n" +
            "    let ua = a + keep\n" +
            "    let b = ua + 3\n" +
            "    let ub = b + keep\n" +
            "    return ub + keep\n" +
            "}\n"
    )
    assert_eq(result.moved, 3)
    assert_eq(result.slots_before, 9)
    assert_eq(result.slots_after, 6)
    assert_eq(result.bytes_after, result.bytes_before)
    assert_eq(
        result.after,
        [
            "OP_GET_LOCAL 1",
            "OP_BIND 2",
            "OP_GET_LOCAL 1",
            "OP_BIND 3",
            "OP_GET_LOCAL 3",
            "OP_GET_LOCAL 2",
            "OP_BIND 3",
            "OP_GET_LOCAL 3",
            "OP_BIND 3",
            "OP_GET_LOCAL 3",
            "OP_GET_LOCAL 2",
            "OP_BIND 3",
            "OP_GET_LOCAL 3",
            "OP_GET_LOCAL 2",
        ]
    )
}

fn test_a_back_edge_keeps_two_loop_locals_apart() -> void {
    let result = coalesce(
        "fn f(xs: list[int]) -> int {\n" +
            "    let seed = xs.len()\n" +
            "    let doubled = seed * 2\n" +
            "    var total = doubled\n" +
            "    for x in xs {\n" +
            "        let step = x + 1\n" +
            "        total = total + step\n" +
            "    }\n" +
            "    return total\n" +
            "}\n"
    )
    assert_eq(result.moved, 4)
    assert_eq(result.slots_before, 10)
    assert_eq(result.slots_after, 8)
    assert_eq(result.bytes_after, result.bytes_before)
    assert_eq(
        result.after,
        [
            "OP_GET_LOCAL 1",
            "OP_BIND 2",
            "OP_GET_LOCAL 2",
            "OP_BIND 2",
            "OP_GET_LOCAL 2",
            "OP_BIND 2",
            "OP_GET_LOCAL 1",
            "OP_BIND 3",
            "OP_GET_LOCAL 3",
            "OP_BIND 4",
            "OP_GET_LOCAL 2",
            "OP_GET_LOCAL 4",
            "OP_BIND 2",
            "OP_GET_LOCAL 2",
        ]
    )
}

fn test_a_captured_slot_is_pinned() -> void {
    let captured = coalesce(
        "fn f(n: int) -> fn() -> int {\n" +
            "    var held = n + 1\n" +
            "    let a = held + 2\n" +
            "    let b = a + 3\n" +
            "    return fn() -> int { return held + b }\n" +
            "}\n"
    )
    assert_eq(captured.moved, 0)
    assert_eq(captured.slots_after, captured.slots_before)
    assert_eq(captured.after, captured.before)
    assert_eq(
        captured.after,
        [
            "OP_GET_LOCAL 1",
            "OP_BIND 2",
            "OP_GET_LOCAL 2",
            "OP_BIND 3",
            "OP_GET_LOCAL 3",
            "OP_BIND 4",
        ]
    )

    let free = coalesce(
        "fn f(n: int) -> int {\n" +
            "    var held = n + 1\n" +
            "    let a = held + 2\n" +
            "    let b = a + 3\n" +
            "    return b\n" +
            "}\n"
    )
    assert_eq(free.moved, 2)
    assert_eq(free.slots_before, 7)
    assert_eq(free.slots_after, 5)
    assert_ne(free.after, free.before)
}

fn test_a_defer_frame_is_left_exactly_as_it_was() -> void {
    let result = coalesce(
        "fn f(n: int) -> int {\n" +
            "    let a = n + 1\n" +
            "    let b = a + 1\n" +
            "    let c = b + 1\n" +
            "    defer { print(c) }\n" +
            "    return c\n" +
            "}\n"
    )
    assert_eq(result.moved, 0)
    assert_eq(result.slots_after, result.slots_before)
    assert_eq(result.bytes_after, result.bytes_before)
    assert_eq(result.after, result.before)
    assert_eq(
        result.after,
        [
            "OP_GET_LOCAL 1",
            "OP_BIND 2",
            "OP_GET_LOCAL 2",
            "OP_BIND 3",
            "OP_GET_LOCAL 3",
            "OP_BIND 4",
            "OP_GET_LOCAL 4",
        ]
    )
}

fn test_a_slot_live_into_a_guarded_region_is_pinned() -> void {
    let result = coalesce(
        "fn f(n: int) -> int {\n" +
            "    let a = n + 1\n" +
            "    let b = a + 1\n" +
            "    let c = b + 1\n" +
            "    try {\n" +
            "        return c\n" +
            "    } catch _raised {\n" +
            "        return 0\n" +
            "    }\n" +
            "}\n"
    )
    assert_eq(result.moved, 1)
    assert_eq(result.slots_before, 8)
    assert_eq(result.slots_after, 8)
    assert_eq(result.bytes_after, result.bytes_before)
    assert_eq(
        result.after,
        [
            "OP_GET_LOCAL 1",
            "OP_BIND 2",
            "OP_GET_LOCAL 2",
            "OP_BIND 2",
            "OP_GET_LOCAL 2",
            "OP_BIND 4",
            "OP_GET_LOCAL 4",
            "OP_BIND 5",
        ]
    )
}

fn test_parameters_are_never_renumbered() -> void {
    let result = coalesce(
        "fn f(p: int, q: int, r: int) -> int {\n" +
            "    let a = p + q\n" +
            "    let b = a + r\n" +
            "    let c = b + 1\n" +
            "    return c\n" +
            "}\n"
    )
    assert_eq(result.moved, 2)
    assert_eq(result.slots_before, 9)
    assert_eq(result.slots_after, 7)
    assert_eq(
        result.after,
        [
            "OP_GET_LOCAL 1",
            "OP_GET_LOCAL 2",
            "OP_BIND 4",
            "OP_GET_LOCAL 4",
            "OP_GET_LOCAL 3",
            "OP_BIND 4",
            "OP_GET_LOCAL 4",
            "OP_BIND 4",
            "OP_GET_LOCAL 4",
        ]
    )
}

fn test_a_frame_with_nothing_above_the_parameters_is_untouched() -> void {
    let result = coalesce("fn f(n: int) -> int {\n    return n + 1\n}\n")
    assert_eq(result.moved, 0)
    assert_eq(result.slots_after, result.slots_before)
    assert_eq(result.bytes_after, result.bytes_before)
    assert_eq(result.after, ["OP_GET_LOCAL 1"])
}

# --- helpers ----------------------------------------------------------------
fn coalesce(source: str) -> Outcome {
    let built = compile_source(source, "<test_opt_coalesce>", false, 0, 0)
    assert_eq(built.ok(), true, built.report())
    let body = built.body
    assert_ne(body, null)

    let proto = _first_function(body ?? FuncProto(null, 0, 0))
    let before = _slot_trace(proto)
    let slots_before = proto.max_slots
    let bytes_before = proto.chunk.code.len()
    let moved = coalesce_slots(proto)
    return Outcome(
        moved,
        slots_before,
        proto.max_slots,
        bytes_before,
        proto.chunk.code.len(),
        before,
        _slot_trace(proto)
    )
}

fn _first_function(body: FuncProto) -> FuncProto {
    for value in body.chunk.constants {
        if value.tag == ConstTag.Func { return value.value }
    }
    return body
}

fn _slot_trace(proto: FuncProto) -> list[str] {
    var out: list[str] = []
    let code = decode(proto)
    if code == null { return out }
    for instruction in code.data {
        if instruction.dead { continue }
        for at in slot_operands(instruction.op) {
            out.push(f"{op_name(instruction.op)} {slot_of(instruction, at)}")
        }
    }
    return out
}
