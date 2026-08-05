#: Loop-invariant `GET_GLOBAL` hoisting: pass 7 of `spec/BYTECODE.md` §8.
#:
#: The Jaithon half of `jaiOptHoistGlobals` (`src/codegen/optimize.c:1769`):
#: outermost loops first, one pre-header load per distinct global, and the
#: in-loop reads rewritten to `GET_LOCAL`.
#:
#: A loop is only a candidate when nothing in it can reach the module table —
#: no call, no global write, no import, and nothing that can run user code
#: behind a field, an index, an iterator or a format. If any of them appears the
#: global is not provably invariant and the loop is skipped entirely.
from jaithon.compile.emit import FuncProto, Op, opcode
from .chunk import Code, Instr, MAX_LOCALS, NO_INDEX, decode, highest_slot_used
from .chunk import live_at_or_after, op_branch_is_edge, op_falls_through, prev_live
from .chunk import read_u24, rebuild, reserved_slots, set_instr, splice_before
from .chunk import u16_bytes, u24_bytes

#: Rounds `hoist_globals` gives the function before it stops looking.
#:
#: A round takes one loop per nesting level, because the loops it takes have to
#: be disjoint; the round after it sees the bodies the first one left alone.
const HOIST_ROUNDS: int = 4

#: Distinct globals one loop may hoist, `OPT_MAX_HOIST_PER_LOOP` in the C.
#:
#: Every one of them costs a frame slot for the whole call, so a loop that reads
#: twenty globals is not worth twenty slots.
const MAX_HOIST_PER_LOOP: int = 8

# One loop and the globals its pre-header will load. `names` holds constant-pool
# indices and the three lists are parallel: `names[k]` is loaded into `slots[k]`
# through inline cache `caches[k]`.
#
# A plan with nothing in it is still recorded, which is what stops the loop it
# names from being picked again.
class HoistPlan {
    pub let header: int
    pub let tail: int
    pub var names: list[int]
    pub var slots: list[int]
    pub var caches: list[int]

    pub fn init(self, header: int, tail: int) {
        self.header = header
        self.tail = tail
        self.names = []
        self.slots = []
        self.caches = []
    }

    pub fn count(self) -> int { return self.names.len() }
}

#: Hoist every loop-invariant global read in `proto`; answer how many were
#: hoisted, with `proto.max_slots` already widened on success.
#:
#: Answers 0 and leaves the chunk untouched when the chunk cannot be decoded or
#: the result cannot be re-encoded.
pub fn hoist_globals(proto: FuncProto) -> int {
    var total = 0
    for _round in 0..HOIST_ROUNDS {
        let code = decode(proto)
        if code == null { break }

        # `_hoist_round` widens the frame itself, so a rebuild that cannot
        # represent the result has to put the old size back or the function
        # keeps slots nothing in it names.
        let slots_before = proto.max_slots
        var hoisted = _hoist_round(code, proto)
        if hoisted > 0 and not rebuild(code) {
            proto.max_slots = slots_before
            hoisted = 0
        }
        if hoisted == 0 { break }
        total += hoisted
    }
    return total
}

# One round: pick a set of non-overlapping loops, rewrite their in-loop reads
# and splice in the pre-header loads. Answers the number of globals hoisted,
# with proto.max_slots already widened on success.
fn _hoist_round(code: Code, proto: FuncProto) -> int {
    var plans: list[HoistPlan] = []
    var hoisted = 0

    var next_slot = highest_slot_used(code) + 1
    if next_slot < reserved_slots(proto) { next_slot = reserved_slots(proto) }
    if next_slot < proto.max_slots { next_slot = proto.max_slots }

    loop {
        let plan = _pick_loop(code, plans)
        if plan == null { break }

        _fill_plan(code, plan, next_slot)
        if plan.count() > 0 {
            _rewrite_reads(code, plan)
            next_slot += plan.count()
            hoisted += plan.count()
        }

        # Recorded even when nothing was hoisted, so the loop is not
        # reconsidered and any nested loop is left for the next round.
        plans.push(plan)
    }

    if hoisted == 0 { return 0 }
    if not _apply(code, plans) { return 0 }

    # One slot per hoisted global, but never below the watermark the plans
    # actually numbered from: `next_slot` already skips past a frame whose
    # highest named slot sits above `max_slots`.
    var want = proto.max_slots + hoisted
    if next_slot > want { want = next_slot }
    proto.max_slots = if want > MAX_LOCALS { MAX_LOCALS } else { want }
    return hoisted
}

# The widest loop that overlaps nothing already planned, or null when there is
# none left.
#
# Outermost first: hoisting an outer loop removes the reads that an inner one
# would otherwise hoist to a worse place, and the two cannot both be taken in
# one round because the pre-header of the inner one would sit inside the outer
# body and run every iteration.
fn _pick_loop(code: Code, plans: list[HoistPlan]) -> HoistPlan? {
    var best_tail = NO_INDEX
    var best_header = 0
    var best_span = -1

    for (index, instruction) in code.data.enumerate() {
        if instruction.dead or instruction.target < 0 { continue }
        if not op_branch_is_edge(instruction.op) { continue }

        # A loop is a backward edge and nothing else: no dominator tree is
        # needed to recognise the shape the emitter actually produces.
        let header = live_at_or_after(code, instruction.target)
        if header >= index { continue }

        var overlaps = false
        for plan in plans {
            if not (index < plan.header or header > plan.tail) { overlaps = true }
        }
        if overlaps or index - header <= best_span { continue }

        best_span = index - header
        best_header = header
        best_tail = index
    }

    if best_tail < 0 { return null }
    return HoistPlan(best_header, best_tail)
}

# Fill `plan` with the globals its loop reads, numbering their slots from
# `next_slot`. Leaves the plan empty when the loop is not a candidate at all.
fn _fill_plan(code: Code, plan: HoistPlan, next_slot: int) -> void {
    var usable = _header_is_private(code, plan.header, plan.tail)
    for index in plan.header..plan.tail + 1 {
        if not usable { break }
        let instruction = code.data[index]
        if not instruction.dead and _blocks_hoist(instruction.op) { usable = false }
    }
    if not usable { return }

    for index in plan.header..plan.tail + 1 {
        let instruction = code.data[index]
        if instruction.dead or instruction.op != opcode(Op.GetGlobal) { continue }

        # The pool index names the global; two reads of one name share a slot
        # and a cache, which is the whole point of the pass.
        let name = read_u24(instruction.operands, 0)
        if name in plan.names or plan.count() >= MAX_HOIST_PER_LOOP { continue }
        if next_slot + plan.count() >= MAX_LOCALS { break }

        let which = plan.count()
        plan.names.push(name)
        plan.slots.push(next_slot + which)
        plan.caches.push(code.chunk.add_cache())
    }
}

# Rewrite every planned read in the loop body as a read of the slot the
# pre-header will have filled.
fn _rewrite_reads(code: Code, plan: HoistPlan) -> void {
    for index in plan.header..plan.tail + 1 {
        let instruction = code.data[index]
        if instruction.dead or instruction.op != opcode(Op.GetGlobal) { continue }
        let name = read_u24(instruction.operands, 0)
        for (which, planned) in plan.names.enumerate() {
            if planned != name { continue }
            set_instr(instruction, opcode(Op.GetLocal), u16_bytes(plan.slots[which]))
            break
        }
    }
}

# Splice the pre-header of every non-empty plan into the list. False when
# nothing was inserted, which the caller reports as zero changes.
fn _apply(code: Code, plans: list[HoistPlan]) -> bool {
    var at: list[int] = []
    var runs: list[list[Instr]] = []

    # `splice_before` wants ascending insertion points and the plans were picked
    # widest first, so walk the code rather than the plan list. The planned
    # ranges are disjoint, so at most one plan starts at any index.
    for index in 0..code.count() {
        for plan in plans {
            if plan.header != index or plan.count() == 0 { continue }
            at.push(index)
            runs.push(_preheader(code.data[index], plan))
        }
    }
    if at.len() == 0 { return false }
    return splice_before(code, at, runs)
}

# The load sequence a plan's loop is entered through: read the global, park it
# in its slot, drop the copy the read left on the stack.
#
# The instructions carry the header's span so that the rebuilt line table still
# attributes them to the loop they belong to rather than to offset zero.
fn _preheader(header: Instr, plan: HoistPlan) -> list[Instr] {
    var run: list[Instr] = []
    for which in 0..plan.count() {
        run.push(
            Instr(
                opcode(Op.GetGlobal),
                u24_bytes(plan.names[which]) + u16_bytes(plan.caches[which]),
                0,
                header.span_start,
                header.span_end
            )
        )
        run.push(
            Instr(
                opcode(Op.SetLocal),
                u16_bytes(plan.slots[which]),
                0,
                header.span_start,
                header.span_end
            )
        )
        run.push(Instr(opcode(Op.Pop), [], 0, header.span_start, header.span_end))
    }
    return run
}

# True when the loop header may only be entered by falling into it from outside
# or by branching to it from inside the loop.
#
# Any other way in skips the pre-header, and the slot then holds whatever the
# frame left there. An exception edge counts, and so does a `PUSH_HANDLER` that
# merely registers the header as a landing site: control arrives there without
# having run the load.
fn _header_is_private(code: Code, header: int, tail: int) -> bool {
    for (index, instruction) in code.data.enumerate() {
        if instruction.dead or instruction.target != header { continue }
        if instruction.op == opcode(Op.PushHandler) or instruction.op == opcode(Op.PushFinally) {
            return false
        }
        if index < header or index > tail { return false }
    }
    for index in 0..code.exception_count() {
        if code.exception_start[index] == header or
            code.exception_end[index] == header or
            code.exception_handler[index] == header {
            return false
        }
    }
    for entry in code.defaults {
        if entry == header { return false }
    }

    # Falling in from the instruction before is the one entry the pre-header is
    # guaranteed to sit on. Instruction 0 has no predecessor at all, so it is
    # only ever entered at the top and needs no test.
    if header > 0 {
        let before = prev_live(code, header)
        if before < 0 or not op_falls_through(code.data[before].op) { return false }
    }
    return true
}

# True when `op` can reach the module table, directly or by transferring control
# into code that might.
fn _blocks_hoist(op: int) -> bool {
    return _BLOCKS_HOIST.get(op, false)
}

let _BLOCKS_HOIST: dict[int, bool] = _build_blocks_hoist()

# The call family, the global writers and the importers, plus everything that
# can run user code behind an operator. `opBlocksHoist` in
# `src/codegen/optimize.c`; a global read is only invariant because none of
# these appears between it and the top of the loop.
fn _build_blocks_hoist() -> dict[int, bool] {
    var table: dict[int, bool] = {}
    let ops = [
        Op.Call,
        Op.CallKw,
        Op.CallSpread,
        Op.Invoke,
        Op.SuperInvoke,
        Op.TailCall,
        Op.New,
        Op.EnumNew,
        Op.Closure,
        Op.Import,
        Op.ImportFrom,
        Op.SetGlobal,
        Op.DefGlobal,
        Op.Export,
        Op.PushDefer,
        Op.GetField,       # a property getter is user code
        Op.GetFieldLocal,
        Op.SetField,
        Op.GetIndex,
        Op.SetIndex,
        Op.GetSlice,
        Op.SetSlice,
        Op.GetIter,
        Op.ForIter,
        Op.ForIterBind,    # a user __next__ is user code
        Op.Format,         # a user __str__, or a rebound `str`, is too
        Op.In,
        Op.NotIn,
        Op.GetSuper,
        Op.Class,
        Op.Method,
        Op.FieldDef,
        Op.ImplTrait,
        Op.Inherit,
        Op.Throw,
        Op.Reraise,
        Op.PushHandler,
    ]
    for op in ops { table[opcode(op)] = true }
    return table
}
