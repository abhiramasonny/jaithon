#: The bytecode-rewriting substrate every optimisation pass sits on.
#:
#: This is the Jaithon half of the instruction list at the top of
#: `src/codegen/optimize.c`, and it exists for the reason that file gives: a
#: peephole that rewrites bytes in place cannot fix up the jumps that cross the
#: edit, and that is precisely how these passes normally corrupt a chunk. Every
#: pass therefore works on a *decoded instruction list*, never on raw bytes,
#: and always in the same four steps:
#:
#: 1. `decode` turns the chunk into `Instr` records and resolves every branch
#:    operand to the *index* of the instruction it lands on. A byte offset that
#:    is not an instruction boundary aborts the decode, which answers null, and
#:    the pass then reports zero changes.
#: 2. Everything branched to, every exception-region edge and every
#:    default-thunk entry point comes back flagged `is_target`. That flag is
#:    what stops a window from being rewritten across an incoming edge.
#: 3. The pass transforms the list: `set_instr` rewrites an instruction in
#:    place, `kill` deletes one, `splice_before` inserts a run.
#: 4. `rebuild` lays the survivors out to a fixpoint, recomputes every branch
#:    operand from the new layout, and rebuilds the line table, the exception
#:    table and the default-thunk offsets against it.
#:
#: Step 4 is what keeps tracebacks and `try` regions pointing at the code they
#: described before the pass ran. When anything in it cannot be represented —
#: a branch that no longer fits in `i16` — `rebuild` answers false, the pass is
#: abandoned and the chunk is left exactly as it was. A missed optimisation,
#: never a wrong one.
#:
#: Branch targets are held as instruction *indices* for the whole life of a
#: pass, so deleting an instruction needs no jump fixup at all: `rebuild`
#: resolves each edge through `live_at_or_after`, which walks past whatever
#: died. Only insertion renumbers the list, and `splice_before` is the one
#: primitive that does it — it remaps every target, every exception edge and
#: every default offset in one pass over the new numbering.
from jaithon.compile.emit import Chunk, Const, ConstTag, ExceptionEntry, FN_KWREST
from jaithon.compile.emit import FN_VARIADIC, FuncProto, LineEntry, MAX_CONSTANTS, Op
from jaithon.compile.emit import op_count, op_operand_bytes, opcode

#: "No instruction here": an unresolved branch target, a code offset that is
#: not an instruction boundary, a slot that is never referenced.
pub const NO_INDEX: int = -1

#: Bounds of the `i16` every branch operand is encoded in (`spec/BYTECODE.md` §2).
pub const I16_MIN: int = -32768
#: Largest value an `i16` branch operand can hold.
pub const I16_MAX: int = 32767

#: Bounds of the `i8` `OP_INC_LOCAL` folds its delta into.
pub const I8_MIN: int = -128
#: Largest value an `OP_INC_LOCAL` delta can hold.
pub const I8_MAX: int = 127

#: Most local slots one frame can hold; `JAI_MAX_LOCALS` in `src/common/common.h`.
pub const MAX_LOCALS: int = 65535

#: Rounds `rebuild` gives the layout to settle before it gives up.
const LAYOUT_ROUNDS: int = 8

#: One decoded instruction.
#:
#: `operands` holds the operand bytes only, so `size()` is `1 + operands.len()`
#: and the opcode byte is never in it. `OP_CLOSURE` needs no special case here
#: the way it does in C: a `list[int]` is as long as it needs to be, so the
#: variable-length operand run is carried like any other.
pub class Instr {
    #: The opcode byte, as `opcode(Op.…)` encodes it.
    pub var op: int
    #: True once the instruction has been deleted. Deleted instructions stay in
    #: the list so that every recorded index keeps meaning what it meant.
    pub var dead: bool
    #: True when something branches here, or when this is an exception-region
    #: edge or a default-thunk entry point. A window may never be rewritten
    #: across one.
    pub var is_target: bool
    #: Byte index of the `i16` branch operand inside `operands`, or `NO_INDEX`.
    pub var branch_at: int
    #: Instruction index this branches to, or `NO_INDEX` when it branches
    #: nowhere. This is the only representation of a branch target a pass ever
    #: sees; the bytes at `branch_at` are stale until `rebuild` runs.
    pub var target: int
    #: The operand bytes, without the opcode.
    pub var operands: list[int]
    #: Where the instruction started in the chunk `decode` read.
    pub var orig_offset: int
    #: Where it will start in the chunk `rebuild` writes.
    pub var new_offset: int
    #: Start of the source span, carried to the new line table.
    pub var span_start: int
    #: One past the end of the source span.
    pub var span_end: int

    #: Record one instruction.
    #:
    #: `decode` passes the offset it read the instruction at; an instruction
    #: built for `splice_before` has no offset of its own and passes 0, and
    #: carries the span of whatever it is being inserted in front of so that the
    #: rebuilt line table still attributes it to real source.
    pub fn init(
        self,
        op: int,
        operands: list[int],
        orig_offset: int = 0,
        span_start: int = 0,
        span_end: int = 0
    ) {
        self.op = op
        self.dead = false
        self.is_target = false
        self.branch_at = branch_operand_at(op)
        self.target = NO_INDEX
        self.operands = operands
        self.orig_offset = orig_offset
        self.new_offset = orig_offset
        self.span_start = span_start
        self.span_end = span_end
    }

    #: Bytes this instruction occupies, opcode included.
    pub fn size(self) -> int { return 1 + self.operands.len() }
}

#: A function's code as a list of instructions, plus the side tables that name
#: code offsets, recorded as instruction indices so they survive deletion.
pub class Code {
    #: The function being rewritten. `max_slots`, `arity` and `flags` are read
    #: from here, and a pass that widens the frame writes `max_slots` back.
    pub let proto: FuncProto
    #: The chunk being rewritten: the constant pool and the cache count are
    #: reached through it, and `rebuild` writes the new code into it.
    pub let chunk: Chunk
    #: The instructions, live and dead, in code order.
    pub var data: list[Instr]
    #: First instruction of each guarded region, one per `chunk.exceptions` row.
    pub var exception_start: list[int]
    #: One past the last instruction of each guarded region. May be `count()`.
    pub var exception_end: list[int]
    #: First instruction of each region's handler.
    pub var exception_handler: list[int]
    #: Entry point of each default-value thunk, one per `proto.default_offsets`.
    pub var defaults: list[int]

    #: Wrap a decoded list. Callers use `decode`, not this.
    pub fn init(self, proto: FuncProto) {
        self.proto = proto
        self.chunk = proto.chunk
        self.data = []
        self.exception_start = []
        self.exception_end = []
        self.exception_handler = []
        self.defaults = []
    }

    #: Number of instructions, live and dead. One past the last index, which is
    #: what "the end of the code" reads as everywhere in this module.
    pub fn count(self) -> int { return self.data.len() }

    #: Number of guarded regions.
    pub fn exception_count(self) -> int { return self.exception_start.len() }

    #: Number of default-value thunks.
    pub fn default_count(self) -> int { return self.defaults.len() }
}

#: True when `value` fits the `i16` a branch operand is encoded in.
pub fn fits_i16(value: int) -> bool { return value >= I16_MIN and value <= I16_MAX }

#: True when `value` fits the `i8` `OP_INC_LOCAL` folds its delta into.
pub fn fits_i8(value: int) -> bool { return value >= I8_MIN and value <= I8_MAX }

#: Read the little-endian `u16` at `at`.
pub fn read_u16(bytes: list[int], at: int) -> int {
    return bytes[at] | bytes[at + 1] << 8
}

#: Read the little-endian `i16` at `at`.
pub fn read_i16(bytes: list[int], at: int) -> int {
    let raw = read_u16(bytes, at)
    return if raw > I16_MAX { raw - 65536 } else { raw }
}

#: Read the little-endian `u24` at `at`, the width of every constant index.
pub fn read_u24(bytes: list[int], at: int) -> int {
    return bytes[at] | bytes[at + 1] << 8 | bytes[at + 2] << 16
}

#: The two bytes of `value` as a little-endian `u16`.
pub fn u16_bytes(value: int) -> list[int] {
    return [value & 0xFF, value >> 8 & 0xFF]
}

#: The three bytes of `value` as a little-endian `u24`.
pub fn u24_bytes(value: int) -> list[int] {
    return [value & 0xFF, value >> 8 & 0xFF, value >> 16 & 0xFF]
}

#: Overwrite the little-endian `u16` at `at`.
pub fn write_u16_at(bytes: list[int], at: int, value: int) -> void {
    bytes[at] = value & 0xFF
    bytes[at + 1] = value >> 8 & 0xFF
}

#: Overwrite the little-endian `u24` at `at`.
pub fn write_u24_at(bytes: list[int], at: int, value: int) -> void {
    bytes[at] = value & 0xFF
    bytes[at + 1] = value >> 8 & 0xFF
    bytes[at + 2] = value >> 16 & 0xFF
}

#: Byte index of the `i16` branch operand inside `op`'s operand run, or
#: `NO_INDEX` when the instruction carries no code address.
#:
#: `branchOperandAt` in `src/codegen/optimize.c`; keep the two in step with
#: `src/vm/chunk.h`.
pub fn branch_operand_at(op: int) -> int {
    return _BRANCH_AT.get(op, NO_INDEX)
}

#: True when the address `op` carries is an edge control actually takes.
#:
#: `PUSH_HANDLER` and `PUSH_FINALLY` only *register* an address; control does
#: not transfer there at that point, so no stack state flows along the edge.
pub fn op_branch_is_edge(op: int) -> bool {
    if branch_operand_at(op) < 0 { return false }
    return op != opcode(Op.PushHandler) and op != opcode(Op.PushFinally)
}

#: True when control can reach the instruction after `op`.
#:
#: `OP_TAIL_CALL` reuses the frame and normally ends it, but the emitter is
#: free to follow it with a `RETURN`; keeping the edge is the conservative
#: choice and costs one instruction.
pub fn op_falls_through(op: int) -> bool {
    return not _TERMINATORS.get(op, false)
}

#: Operand bytes of the `OP_CLOSURE` at `offset`, or `NO_INDEX` when the chunk
#: is malformed.
#:
#: The run is a `u24` constant index plus one `(u8 how, u16 index)` descriptor
#: per upvalue of the function it names, and that count lives in the referenced
#: record — which is why `op_operand_bytes` answers -1 for this one opcode and
#: every decoder has to special-case it.
pub fn closure_operand_bytes(chunk: Chunk, offset: int) -> int {
    if offset + 4 > chunk.code.len() { return NO_INDEX }
    let index = read_u24(chunk.code, offset + 1)
    if index < 0 or index >= chunk.constants.len() { return NO_INDEX }
    let value = chunk.constants[index]
    if value.tag != ConstTag.Func { return NO_INDEX }
    return 3 + 3 * value.value.upvalue_count
}

#: Number of upvalue descriptors an `OP_CLOSURE` carries.
pub fn closure_descriptor_count(instruction: Instr) -> int {
    return (instruction.operands.len() - 3) // 3
}

#: The `how` byte of one `OP_CLOSURE` upvalue descriptor.
#:
#: Bit 0 is set when the source is a local of the enclosing frame rather than
#: one of its upvalues; bit 1 is the by-value flag. A by-value capture of an
#: *upvalue* therefore reads as 2, and its operand is an upvalue index and not
#: a slot.
pub fn closure_descriptor_flags(instruction: Instr, which: int) -> int {
    return instruction.operands[3 + which * 3]
}

#: The `u16` operand of one `OP_CLOSURE` upvalue descriptor.
pub fn closure_descriptor_slot(instruction: Instr, which: int) -> int {
    return read_u16(instruction.operands, 3 + which * 3 + 1)
}

#: Byte offsets of the `u16` local-slot operands of `op`, in operand-run
#: coordinates. Empty when `op` names no slot.
#:
#: `slotOperands` in `src/codegen/optimize.c`. `OP_CLOSURE` is not here: its
#: slots live in a variable-length descriptor run, reached through
#: `closure_descriptor_slot`.
pub fn slot_operands(op: int) -> list[int] {
    return _SLOT_OPERANDS.get(op, [])
}

#: The `u16` slot operand of `instruction` at byte offset `at`.
pub fn slot_of(instruction: Instr, at: int) -> int {
    return read_u16(instruction.operands, at)
}

#: Overwrite the `u16` slot operand of `instruction` at byte offset `at`.
pub fn set_slot(instruction: Instr, at: int, slot: int) -> void {
    write_u16_at(instruction.operands, at, slot)
}

#: Slots the caller owns: the callee/self slot plus every parameter.
#:
#: They are filled in by the call sequence, so their numbering is part of the
#: calling convention and can never move.
pub fn reserved_slots(proto: FuncProto) -> int {
    var params = proto.arity  # already counts the defaulted parameters
    if proto.flags & FN_VARIADIC != 0 { params += 1 }
    if proto.flags & FN_KWREST != 0 { params += 1 }
    if proto.param_names.len() > params { params = proto.param_names.len() }

    var reserved = params + 1
    if reserved > proto.max_slots { reserved = proto.max_slots }
    return reserved
}

#: Highest frame slot any live instruction names, or -1 when none does.
#:
#: The `OP_CLOSURE` test here is `flags != 0`, not `flags & 1`, which reads a
#: by-value capture of an upvalue as if its operand were a slot. That is what
#: `highestSlotUsed` in `src/codegen/optimize.c` does and the difference is
#: deliberate: this answer only ever moves a watermark upwards, so reading one
#: index too many costs a slot and can never alias one.
pub fn highest_slot_used(code: Code) -> int {
    var highest = -1
    for instruction in code.data {
        if instruction.dead { continue }
        if instruction.op == opcode(Op.Closure) {
            for which in 0..closure_descriptor_count(instruction) {
                if closure_descriptor_flags(instruction, which) == 0 { continue }
                let slot = closure_descriptor_slot(instruction, which)
                if slot > highest { highest = slot }
            }
            continue
        }
        for at in slot_operands(instruction.op) {
            let slot = slot_of(instruction, at)
            if slot > highest { highest = slot }
        }
    }
    return highest
}

#: True when `proto` is a `defer` thunk.
#:
#: `jaiFunctionIsDeferThunk` in `src/vm/object.c` recognises one by name alone,
#: and the emitter names every thunk it builds `defer` for exactly that reason.
pub fn is_defer_thunk(proto: FuncProto) -> bool {
    return proto.name == "defer"
}

#: Decode `proto`'s chunk, or null when it cannot be decoded.
#:
#: Null means "this pass does nothing": a chunk that is not decodable is the
#: emitter's problem, and reporting it here would turn a malformed chunk into a
#: crash inside an optimisation instead of a diagnosis from the verifier.
pub fn decode(proto: FuncProto) -> Code? {
    let chunk = proto.chunk
    if chunk.code.len() == 0 { return null }

    let code = Code(proto)
    # Instruction index for a code offset; NO_INDEX where the offset is not an
    # instruction boundary. chunk.code.len() maps to count(), one past the end.
    var offset_index: list[int] = [NO_INDEX for _slot in 0..chunk.code.len() + 1]

    var line = 0
    var span_start = 0
    var span_end = 0
    var offset = 0
    while offset < chunk.code.len() {
        let op = chunk.code[offset]
        if op >= op_count() { return null }

        var operand_count = op_operand_bytes(op)
        if op == opcode(Op.Closure) { operand_count = closure_operand_bytes(chunk, offset) }
        if operand_count < 0 or offset + 1 + operand_count > chunk.code.len() { return null }

        # The line table is sorted by offset and covers everything from a row
        # up to the next one, so walking it alongside the decode gives the same
        # answer as jaiChunkSpanAt's binary search for a fraction of the work.
        while line < chunk.lines.len() and chunk.lines[line].offset <= offset {
            span_start = chunk.lines[line].span_start
            span_end = chunk.lines[line].span_end
            line += 1
        }

        let instruction = Instr(
            op,
            chunk.code[offset + 1:offset + 1 + operand_count],
            offset,
            span_start,
            span_end
        )
        offset_index[offset] = code.data.len()
        code.data.push(instruction)
        offset += 1 + operand_count
    }
    offset_index[chunk.code.len()] = code.data.len()

    for instruction in code.data {
        if instruction.branch_at < 0 { continue }
        let relative = read_i16(instruction.operands, instruction.branch_at)
        let landing = instruction.orig_offset + instruction.size() + relative
        let index = _index_of_offset(offset_index, landing)
        if index == NO_INDEX or index >= code.count() { return null }
        instruction.target = index
        code.data[index].is_target = true
    }

    # Exception regions. Marking the start and end instructions as targets is
    # what stops a peephole window from straddling a region edge: a fused
    # instruction that is half inside a protected region cannot be described by
    # (start, end) at all.
    for entry in chunk.exceptions {
        let start = _index_of_offset(offset_index, entry.start)
        let end = _index_of_offset(offset_index, entry.end)
        let handler = _index_of_offset(offset_index, entry.handler)
        if start == NO_INDEX or end == NO_INDEX or handler == NO_INDEX { return null }
        if handler >= code.count() or start > end { return null }
        code.exception_start.push(start)
        code.exception_end.push(end)
        code.exception_handler.push(handler)
        if start < code.count() { code.data[start].is_target = true }
        if end < code.count() { code.data[end].is_target = true }
        code.data[handler].is_target = true
    }

    for entry in proto.default_offsets {
        let index = _index_of_offset(offset_index, entry)
        if index == NO_INDEX or index >= code.count() { return null }
        code.defaults.push(index)
        code.data[index].is_target = true
    }
    return code
}

#: The next live instruction after `index`, or `NO_INDEX`.
pub fn next_live(code: Code, index: int) -> int {
    for candidate in index + 1..code.count() {
        if not code.data[candidate].dead { return candidate }
    }
    return NO_INDEX
}

#: The last live instruction before `index`, or `NO_INDEX`.
pub fn prev_live(code: Code, index: int) -> int {
    var candidate = index - 1
    while candidate >= 0 {
        if not code.data[candidate].dead { return candidate }
        candidate -= 1
    }
    return NO_INDEX
}

#: The first live instruction, or `NO_INDEX` when everything is dead.
pub fn first_live(code: Code) -> int {
    for candidate in 0..code.count() {
        if not code.data[candidate].dead { return candidate }
    }
    return NO_INDEX
}

#: The instruction an index lands on once deleted instructions are gone.
#:
#: Answers `count()` for "the end of the code", which is a legal exception
#: region end and nothing else.
pub fn live_at_or_after(code: Code, index: int) -> int {
    if index < 0 { return code.count() }
    for candidate in index..code.count() {
        if not code.data[candidate].dead { return candidate }
    }
    return code.count()
}

#: Up to `want` consecutive live instruction indices starting at `index`.
#:
#: Shorter than `want` when the code runs out, which is the caller's cue that
#: the window it wanted does not exist.
pub fn window_of(code: Code, index: int, want: int) -> list[int] {
    var window: list[int] = []
    var at = index
    while at >= 0 and window.len() < want {
        window.push(at)
        at = next_live(code, at)
    }
    return window
}

#: True when the first `want` instructions of the window may be rewritten as a
#: unit; `want` defaults to the whole of it.
#:
#: Nothing *inside* the tested prefix may be branched to: an edge landing on the
#: first instruction still executes the whole window, so fusing or deleting it
#: as a unit is invisible to that edge. An edge landing further in is not.
#:
#: The prefix length is a parameter because the fusion pass decodes one window
#: four deep and offers a *prefix* of it to each of its eleven rules; testing
#: all four when the rule only rewrites two would refuse a legal fusion. That is
#: `windowClean(c, idx, n)` in `src/codegen/optimize.c:347`, whose `n` is the
#: same number.
pub fn window_clean(code: Code, window: list[int], want: int = NO_INDEX) -> bool {
    let last = if want < 0 or want > window.len() { window.len() } else { want }
    for position in 1..last {
        if code.data[window[position]].is_target { return false }
    }
    return true
}

#: Rewrite `instruction` in place as `op` with `operands`.
#:
#: The instruction keeps its position, its span and — when the new opcode
#: carries a branch operand at all — its `target`, so an edge into it still
#: works. Deleting the rest of a window with `kill` and rewriting its head with
#: this is how every fusion is expressed.
pub fn set_instr(instruction: Instr, op: int, operands: list[int]) -> void {
    instruction.op = op
    instruction.operands = operands
    instruction.branch_at = branch_operand_at(op)
    if instruction.branch_at < 0 { instruction.target = NO_INDEX }
}

#: Delete the instruction at `index`.
#:
#: It stays in the list so that every index recorded anywhere keeps meaning
#: what it meant; `rebuild` drops it and resolves every edge that pointed at it
#: to the first survivor at or after it.
pub fn kill(code: Code, index: int) -> void {
    code.data[index].dead = true
}

#: Delete every instruction named in `indices`.
pub fn kill_all(code: Code, indices: list[int]) -> void {
    for index in indices { kill(code, index) }
}

#: A constant-pool index holding the int `value`, appending it when the pool
#: does not already carry it. `NO_INDEX` when the pool is full.
#:
#: Ints are immediate values, not heap objects, so this cannot hand the
#: collector a half-built object the way appending a string would; the only
#: failure mode is a pool that has run out of `u24` indices, and the read-back
#: catches that rather than letting index 0 be used as if it were the constant.
pub fn pool_int_index(code: Code, value: int) -> int {
    if code.chunk.constants.len() >= MAX_CONSTANTS - 1 { return NO_INDEX }
    let index = code.chunk.add_constant(Const(ConstTag.Int, value))
    if index < 0 or index >= code.chunk.constants.len() { return NO_INDEX }
    let got = code.chunk.constants[index]
    if got.tag != ConstTag.Int or got.value != value { return NO_INDEX }
    return index
}

#: Insert instruction runs into the list and remap every recorded index.
#:
#: `at[k]` is the instruction the run `runs[k]` goes immediately *before*, and
#: the instruction that was at `at[k]` keeps every edge that pointed at it — a
#: backward branch into a loop header still lands on the header and not on the
#: pre-header just spliced in front of it. `at` must be ascending and free of
#: duplicates, and both lists must be the same length.
#:
#: Every branch target, exception-region edge and default-thunk offset is
#: remapped onto the new numbering, which is the whole reason insertion is one
#: primitive and not something a pass does by hand. Byte offsets are
#: meaningless afterwards; `rebuild` works from indices only.
#:
#: Answers false and changes nothing when the plan is empty or malformed.
pub fn splice_before(code: Code, at: list[int], runs: list[list[Instr]]) -> bool {
    if at.len() == 0 or at.len() != runs.len() { return false }

    var inserted = 0
    var previous = NO_INDEX
    for (position, index) in at.enumerate() {
        if index <= previous or index < 0 or index > code.count() { return false }
        previous = index
        inserted += runs[position].len()
    }
    if inserted == 0 { return false }

    var out: list[Instr] = []
    var map: list[int] = [NO_INDEX for _slot in 0..code.count() + 1]
    var pending = 0

    for index in 0..code.count() {
        while pending < at.len() and at[pending] == index {
            for instruction in runs[pending] { out.push(instruction) }
            pending += 1
        }
        # The original instruction maps past its own prefix, which is what
        # keeps an incoming edge landing on it and not on the inserted run.
        map[index] = out.len()
        out.push(code.data[index])
    }
    while pending < at.len() and at[pending] == code.count() {
        for instruction in runs[pending] { out.push(instruction) }
        pending += 1
    }
    map[code.count()] = out.len()

    for instruction in out {
        if instruction.target >= 0 and instruction.target <= code.count() {
            instruction.target = map[instruction.target]
        }
    }
    for index in 0..code.exception_count() {
        code.exception_start[index] = map[code.exception_start[index]]
        code.exception_end[index] = map[code.exception_end[index]]
        code.exception_handler[index] = map[code.exception_handler[index]]
    }
    for index in 0..code.default_count() {
        code.defaults[index] = map[code.defaults[index]]
    }
    code.data = out
    return true
}

#: Mark everything unreachable from an entry point dead; answer how many that
#: removed.
#:
#: Entry points are instruction 0 and every default-value thunk, which the call
#: sequence enters directly. An exception handler is *not* an unconditional
#: entry point: it runs only when something inside its protected region throws,
#: so it is reachable exactly when some instruction of that region is. Seeding
#: it regardless resurrects the handler of a `try` the front end has just
#: proved unreachable, and with it — through the backward branch at the end of
#: an enclosing loop body — the loop's `OP_FOR_ITER`, whose iterator is
#: produced by the setup code that stayed dead.
#:
#: Shared substrate rather than part of `opt.dead`: the peephole alternates
#: rewriting with this until neither has anything left to do.
pub fn mark_unreachable(code: Code) -> int {
    if code.count() == 0 { return 0 }

    var reached: list[bool] = [false for _slot in 0..code.count()]
    var work: list[int] = []

    _seed(code, reached, work, 0)
    for entry in code.defaults { _seed(code, reached, work, entry) }

    # Reaching a handler can make another region's body reachable, so the drain
    # and the region test alternate until a round adds nothing.
    var settled = false
    while not settled {
        while work.len() > 0 {
            let index = work.pop()
            let instruction = code.data[index]
            # Every address an instruction carries counts, including the one
            # PUSH_HANDLER and PUSH_FINALLY register: control does not go there
            # now, but it can later, and only if this instruction runs.
            if instruction.target >= 0 { _seed(code, reached, work, instruction.target) }
            if op_falls_through(instruction.op) {
                let following = next_live(code, index)
                if following >= 0 { _seed(code, reached, work, following) }
            }
        }

        settled = true
        for region in 0..code.exception_count() {
            let handler = live_at_or_after(code, code.exception_handler[region])
            if handler >= code.count() or reached[handler] { continue }
            var body_lives = false
            for index in code.exception_start[region]..code.exception_end[region] {
                if index < code.count() and reached[index] { body_lives = true }
            }
            if not body_lives { continue }
            _seed(code, reached, work, handler)
            settled = false
        }
    }

    var removed = 0
    for index in 0..code.count() {
        if not code.data[index].dead and not reached[index] {
            code.data[index].dead = true
            removed += 1
        }
    }
    return removed
}

#: Re-encode the list into the chunk and rebuild every table that names a code
#: offset. False without touching the chunk when the result cannot be encoded,
#: in which case the caller must report zero changes.
#: Drop deleted instructions and renumber everything that named one.
#:
#: Deleted instructions stay in the list so that recorded indices keep meaning
#: what they meant, which is right within a pass and expensive between them:
#: `next_live`, `prev_live` and `live_at_or_after` all walk over the holes, so a
#: pipeline that never compacts turns a long dead run into a linear scan per
#: lookup. That used to be hidden because every pass ended in a `rebuild` and
#: began with a `decode`, and the round trip compacted as a side effect. With
#: one decode shared across the passes the compaction has to be asked for, and
#: this is the cheap half of what the round trip was doing -- no encoding, no
#: line table, no exception rewrite, just the list and the indices into it.
#:
#: A target that pointed at a deleted instruction lands on the next live one,
#: which is what `live_at_or_after` already answers for a stale target.
pub fn compact(code: Code) -> void {
    var dead_count = 0
    for instruction in code.data {
        if instruction.dead { dead_count += 1 }
    }
    if dead_count == 0 { return }

    # Compacting is two allocations and a copy of the whole list, which is the
    # same order of work as the decode it is standing in for -- so doing it for
    # a single hole costs more than the holes do. It pays once a run is long
    # enough that the live-walkers spend real time stepping over it, and the
    # amortisation argument is the usual one: only rewrite when a constant
    # fraction of the list is garbage.
    if dead_count * 8 < code.count() { return }

    let total = code.count()
    var mapping: list[int] = [0 for _slot in 0..total + 1]
    var live = 0
    for (index, instruction) in code.data.enumerate() {
        mapping[index] = live
        if not instruction.dead { live += 1 }
    }
    mapping[total] = live

    var kept: list[Instr] = []
    for instruction in code.data {
        if instruction.dead { continue }
        if instruction.target >= 0 and instruction.target <= total {
            instruction.target = mapping[instruction.target]
        }
        kept.push(instruction)
    }

    # `_retarget_fits` measures a branch against `new_offset`, so the layout is
    # refreshed here exactly as `rebuild` would lay it out. Leaving it stale
    # would make the peephole's reach test answer from a layout that no longer
    # exists, and its decisions are what the emitted code is made of.
    var offset = 0
    for instruction in kept {
        instruction.new_offset = offset
        offset += instruction.size()
    }

    code.data = kept
    for (row, start) in code.exception_start.enumerate() {
        if start >= 0 and start <= total { code.exception_start[row] = mapping[start] }
    }
    for (row, end) in code.exception_end.enumerate() {
        if end >= 0 and end <= total { code.exception_end[row] = mapping[end] }
    }
    for (row, handler) in code.exception_handler.enumerate() {
        if handler >= 0 and handler <= total { code.exception_handler[row] = mapping[handler] }
    }
    for (row, entry) in code.defaults.enumerate() {
        if entry >= 0 and entry <= total { code.defaults[row] = mapping[entry] }
    }
}

pub fn rebuild(code: Code) -> bool {
    var live_count = 0
    for instruction in code.data {
        if not instruction.dead { live_count += 1 }
    }
    if live_count == 0 { return false }

    # Lay the instructions out to a fixpoint. Every branch operand in this
    # bytecode is a fixed-width i16, so the second round always agrees with the
    # first; the loop is here so that a narrower encoding added later cannot
    # silently produce stale offsets.
    var total = 0
    var settled = false
    var round = 0
    while round < LAYOUT_ROUNDS and not settled {
        var offset = 0
        settled = true
        for instruction in code.data {
            if instruction.new_offset != offset {
                instruction.new_offset = offset
                settled = false
            }
            if not instruction.dead { offset += instruction.size() }
        }
        total = offset
        round += 1
    }
    if not settled { return false }

    # Resolve branches before allocating anything, so an unrepresentable jump
    # leaves the chunk untouched.
    #
    # Failing here is a *capacity* verdict, not a diagnosis: hoisting splices
    # instructions into a pre-header, and in a function whose jumps already sit
    # near the i16 limit that is enough to make one not fit. Reporting it would
    # make a program that compiles at -O0 fail at -O2, which is the one thing
    # the pass list may never do. The caller reports zero changes, the chunk
    # stays exactly as it was, and a chunk that was malformed before the pass
    # is still caught by the verifier afterwards.
    for instruction in code.data {
        if instruction.dead or instruction.branch_at < 0 { continue }
        let landing = live_at_or_after(code, instruction.target)
        if landing >= code.count() { return false }
        let delta = code.data[landing].new_offset - (instruction.new_offset + instruction.size())
        if not fits_i16(delta) { return false }
    }

    var bytes: list[int] = [0 for _slot in 0..total]
    var lines: list[LineEntry] = []

    for instruction in code.data {
        if instruction.dead { continue }
        bytes[instruction.new_offset] = instruction.op
        for (position, byte) in instruction.operands.enumerate() {
            bytes[instruction.new_offset + 1 + position] = byte
        }
        if instruction.branch_at >= 0 {
            let landing = live_at_or_after(code, instruction.target)
            let delta = code.data[landing].new_offset -
                (instruction.new_offset + instruction.size())
            write_u16_at(bytes, instruction.new_offset + 1 + instruction.branch_at, delta & 0xFFFF)
        }

        # One line entry per run of instructions sharing a span, matching what
        # recordSpan in src/vm/chunk.c would have produced during emission.
        let rows = lines.len()
        if rows == 0 or
            lines[rows - 1].span_start != instruction.span_start or
            lines[rows - 1].span_end != instruction.span_end {
            lines.push(
                LineEntry(instruction.new_offset, instruction.span_start, instruction.span_end)
            )
        }
    }

    # Exception table: a region whose body vanished is dropped, otherwise both
    # edges move to the first surviving instruction at or after them.
    var kept: list[ExceptionEntry] = []
    var kept_start: list[int] = []
    var kept_end: list[int] = []
    var kept_handler: list[int] = []
    for region in 0..code.exception_count() {
        let handler = live_at_or_after(code, code.exception_handler[region])
        if handler >= code.count() { continue }
        let start = _offset_of_index(
            code,
            live_at_or_after(code, code.exception_start[region]),
            total
        )
        let end = _offset_of_index(code, live_at_or_after(code, code.exception_end[region]), total)
        if start >= end { continue }
        kept.push(
            ExceptionEntry(
                start,
                end,
                _offset_of_index(code, handler, total),
                code.chunk.exceptions[region].type_const
            )
        )
        kept_start.push(code.exception_start[region])
        kept_end.push(code.exception_end[region])
        kept_handler.push(code.exception_handler[region])
    }
    code.chunk.exceptions = kept
    # The recorded indices have to shrink with the table they describe, or a
    # later pass in the same decode reads a region that is no longer there.
    code.exception_start = kept_start
    code.exception_end = kept_end
    code.exception_handler = kept_handler

    for index in 0..code.default_count() {
        code.proto.default_offsets[index] = _offset_of_index(
            code,
            live_at_or_after(code, code.defaults[index]),
            total
        )
    }

    code.chunk.code = bytes
    code.chunk.lines = lines
    return true
}

let _BRANCH_AT: dict[int, int] = _build_branch_at()
let _TERMINATORS: dict[int, bool] = _build_terminators()
let _SLOT_OPERANDS: dict[int, list[int]] = _build_slot_operands()

fn _build_branch_at() -> dict[int, int] {
    var table: dict[int, int] = {}
    let leading = [
        Op.Jump,
        Op.JumpIfFalse,
        Op.JumpIfTrue,
        Op.JumpIfFalseKeep,
        Op.JumpIfTrueKeep,
        Op.JumpIfNull,
        Op.Loop,
        Op.ForIter,
        Op.ForIterBind,
        Op.PushFinally,
        Op.PushHandler,
    ]
    for op in leading { table[opcode(op)] = 0 }
    table[opcode(Op.JumpIfCmpFalse)] = 1
    # After the u8 comparison, the u16 slot and the u24 constant index.
    table[opcode(Op.JumpIfCmpLocalK)] = 6
    table[opcode(Op.MatchSeq)] = 2
    for op in [Op.MatchConst, Op.MatchType, Op.MatchFields] { table[opcode(op)] = 3 }
    table[opcode(Op.MatchRange)] = 7
    return table
}

fn _build_terminators() -> dict[int, bool] {
    var table: dict[int, bool] = {}
    let ops = [Op.Jump, Op.Loop, Op.Return, Op.ReturnNull, Op.Throw, Op.Reraise, Op.Halt]
    for op in ops { table[opcode(op)] = true }
    return table
}

fn _build_slot_operands() -> dict[int, list[int]] {
    var table: dict[int, list[int]] = {}
    let leading = [
        Op.GetLocal,
        Op.SetLocal,
        Op.Bind,
        Op.CloseUpvalue,
        Op.AddIntConst,
        Op.IncLocal,
        Op.CmpLocalConstLt,
        Op.GetFieldLocal,
        # Without this the coalescer renumbers every *reader* of a slot and
        # leaves the fused writer pointing at the old one: `line` was stored to
        # slot 4 while every read of it had moved to slot 3, which still held an
        # int. A fused opcode that carries a slot has to be listed here.
        Op.AddBind,
        Op.SubBind,
        Op.MulBind,
    ]
    for op in leading { table[opcode(op)] = [0] }
    table[opcode(Op.ForIterBind)] = [2]      # the slot follows the i16 jump
    table[opcode(Op.JumpIfCmpLocalK)] = [1]  # the slot follows the u8 comparison
    for op in [Op.GetLocal2, Op.AddLocals] { table[opcode(op)] = [0, 2] }
    return table
}

fn _index_of_offset(offset_index: list[int], offset: int) -> int {
    if offset < 0 or offset >= offset_index.len() { return NO_INDEX }
    return offset_index[offset]
}

fn _offset_of_index(code: Code, index: int, end_offset: int) -> int {
    if index >= code.count() { return end_offset }
    return code.data[index].new_offset
}

fn _seed(code: Code, reached: list[bool], work: list[int], index: int) -> void {
    let at = live_at_or_after(code, index)
    if at < code.count() and not reached[at] {
        reached[at] = true
        work.push(at)
    }
}
