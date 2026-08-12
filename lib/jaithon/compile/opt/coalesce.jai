#: Local slot coalescing: pass 4 of `spec/BYTECODE.md` §8.
#:
#: The Jaithon half of `jaiOptCoalesceSlots` (`src/codegen/optimize.c:1273`): a
#: linear scan over the live range of every non-reserved slot, giving each the
#: lowest number that is free where it starts, and shrinking the frame window
#: by exactly the slots that disappeared.
#:
#: Anything the scan cannot see through is left alone rather than guessed at: a
#: captured slot is aliased by an open upvalue, a `defer` closure reads slots in
#: an order that has no relation to this scan's linear one, and a handler may be
#: entered from the middle of a protected region with definitions skipped.
from jaithon.compile.emit import FuncProto, Op, opcode
from .chunk import Code, NO_INDEX, closure_descriptor_count, closure_descriptor_flags
from .chunk import closure_descriptor_slot, decode, live_at_or_after, op_branch_is_edge
from .chunk import rebuild, reserved_slots, set_slot, slot_of, slot_operands

#: Rounds the back-edge widening gives the live ranges to settle.
#:
#: One round per level of loop nesting is enough, since an inner loop's ranges
#: are final before the enclosing back edge is read in the next round; 8 is the
#: same bound `jaiOptCoalesceSlots` uses and stops a malformed chunk with a
#: cycle of edges from spinning.
const WIDEN_ROUNDS: int = 8

#: Where one frame slot is first and last referenced, and whether it may move.
#:
#: `first` and `last` are instruction indices, `NO_INDEX` when the slot is never
#: named by a slot operand. A pinned slot keeps its number *and* its position:
#: nothing may be renumbered onto it either.
class SlotRange {
    #: Index of the first instruction naming this slot, or `NO_INDEX`.
    pub var first: int
    #: Index of the last instruction naming this slot, or `NO_INDEX`.
    pub var last: int
    #: True when the slot's number is part of something outside this scan.
    pub var pinned: bool

    #: An unreferenced range, pinned or not.
    pub fn init(self, pinned: bool) {
        self.first = NO_INDEX
        self.last = NO_INDEX
        self.pinned = pinned
    }
}

#: Renumber the frame slots of `proto`; answer how many slots were moved.
#:
#: Answers 0 and leaves the chunk untouched when the chunk cannot be decoded,
#: the result cannot be re-encoded, or anything in the frame is pinned in a way
#: the scan cannot reason about.
pub fn coalesce_slots(proto: FuncProto) -> int {
    let code = decode(proto)
    if code == null { return 0 }
    let changed = coalesce_slots_on(code)
    if changed > 0 and not rebuild(code) { return 0 }
    return changed
}

#: The pass on an already-decoded chunk. See `peephole_on` for why the decode
#: is the caller's job.
#:
#: `max_slots` is narrowed here without waiting for a rebuild to confirm it,
#: so a caller whose final rebuild fails has to put the old value back; the
#: driver in `opt/mod.jai` saves it for exactly that.
pub fn coalesce_slots_on(code: Code) -> int {
    let proto = code.proto
    let slot_count = proto.max_slots
    let reserved = reserved_slots(proto)
    if slot_count <= reserved { return 0 }

    var ranges = [SlotRange(slot < reserved) for slot in 0..slot_count]

    # The first point at which exception flow can bypass a definition.
    var hazard_start = code.count()
    for region in 0..code.exception_count() {
        if code.exception_start[region] < hazard_start {
            hazard_start = code.exception_start[region]
        }
        if code.exception_handler[region] < hazard_start {
            hazard_start = code.exception_handler[region]
        }
    }

    if not _scan_ranges(code, ranges, slot_count) { return 0 }

    # Left alone once exception flow can divert control: a handler may be
    # entered mid-region, and this linear scan can't see which defs were skipped.
    for slot in reserved..slot_count {
        if ranges[slot].last >= hazard_start { ranges[slot].pinned = true }
    }

    _widen_across_back_edges(code, ranges, reserved, slot_count)

    var target = _assign(code, ranges, reserved, slot_count)
    var changed = 0
    for slot in 0..slot_count {
        if target[slot] != slot { changed += 1 }
    }
    if changed == 0 { return 0 }

    # The highest number in use before and after, counting only the slots this
    # scan actually saw. A slot named by nothing but an `OP_CLOSURE` descriptor
    # is in neither figure, exactly as in the C.
    var old_top = reserved
    var new_top = reserved
    for slot in 0..slot_count {
        if ranges[slot].first < 0 { continue }
        if slot + 1 > old_top { old_top = slot + 1 }
        if target[slot] + 1 > new_top { new_top = target[slot] + 1 }
    }

    for instruction in code.data {
        if instruction.dead { continue }
        for at in slot_operands(instruction.op) {
            set_slot(instruction, at, target[slot_of(instruction, at)])
        }
    }

    if new_top < old_top {
        # Shrink the window by exactly the slots that disappeared rather than
        # setting it to `new_top`: whatever headroom the emitter left above the
        # locals for temporaries has to survive.
        var shrunk = proto.max_slots - (old_top - new_top)
        if shrunk < new_top { shrunk = new_top }
        proto.max_slots = shrunk
    }
    return changed
}

# Record where every slot is referenced. False means the frame holds something
# this scan cannot reason about, and the whole pass is then abandoned.
fn _scan_ranges(code: Code, ranges: list[SlotRange], slot_count: int) -> bool {
    for (index, instruction) in code.data.enumerate() {
        if instruction.dead { continue }

        # A deferred closure runs at frame exit, reading slots in an order that
        # has no relation to this scan's linear one.
        if instruction.op == opcode(Op.PushDefer) { return false }

        # A captured slot is aliased by an open upvalue that is closed by stack
        # position, so both its number and its position must hold.
        if instruction.op == opcode(Op.Closure) {
            for which in 0..closure_descriptor_count(instruction) {
                # Bit 0 of the descriptor, not the whole byte: bit 1 is the
                # by-value flag, so a by-value capture of an *upvalue* reads as
                # 2 and its operand is an upvalue index, not a slot.
                if closure_descriptor_flags(instruction, which) & 1 == 0 { continue }
                let slot = closure_descriptor_slot(instruction, which)
                if slot >= slot_count { return false }
                ranges[slot].pinned = true
            }
            continue
        }

        for at in slot_operands(instruction.op) {
            let slot = slot_of(instruction, at)
            if slot >= slot_count { return false }
            if ranges[slot].first < 0 { ranges[slot].first = index }
            ranges[slot].last = index
        }

        # `OP_CLOSE_UPVALUE`'s operand names a position, not just a variable --
        # it closes this slot and everything above it, so renumbering would move
        # the boundary. The slots it covers are already pinned by their
        # `OP_CLOSURE`, so pinning the boundary alone keeps the relation intact.
        if instruction.op == opcode(Op.CloseUpvalue) {
            ranges[slot_of(instruction, 0)].pinned = true
        }
    }
    return true
}

# A backward edge makes every slot referenced inside the loop live across the
# whole loop; without this, a slot last read at the top of a loop body would
# look dead to a slot first written at the bottom of it.
fn _widen_across_back_edges(
    code: Code,
    ranges: list[SlotRange],
    reserved: int,
    slot_count: int
) -> void {
    for _round in 0..WIDEN_ROUNDS {
        var grew = false
        for (index, instruction) in code.data.enumerate() {
            if instruction.dead or instruction.target < 0 { continue }
            if not op_branch_is_edge(instruction.op) { continue }
            let top = live_at_or_after(code, instruction.target)
            if top >= index { continue }

            for slot in reserved..slot_count {
                if ranges[slot].first < 0 or ranges[slot].pinned { continue }
                if ranges[slot].first > index or ranges[slot].last < top { continue }
                if ranges[slot].first > top {
                    ranges[slot].first = top
                    grew = true
                }
                if ranges[slot].last < index {
                    ranges[slot].last = index
                    grew = true
                }
            }
        }
        if not grew { break }
    }
}

# Linear scan: walk the candidate slots in order of first reference and give
# each one the lowest number that is free where it starts. Answers the new
# number of every slot, which is the slot itself for everything left alone.
#
# `live_to[number]` tracks the furthest point any of that number's tenants
# reaches, not just the current one: tracking a single "current occupant" hands
# out a number that is still live as soon as a second tenant arrives. That was
# a real miscompile; see the `jaiOptCoalesceSlots` entry in ROADMAP.md.
#
# It combines the candidate still sitting in its own number (until it moves)
# and every candidate already assigned there; `count()` -- past the last
# instruction index -- means "never free".
fn _assign(code: Code, ranges: list[SlotRange], reserved: int, slot_count: int) -> list[int] {
    let never_free = code.count()
    var target = [slot for slot in 0..slot_count]
    var placed = [false for _slot in 0..slot_count]
    var live_to = [NO_INDEX for _slot in 0..slot_count]

    for slot in 0..slot_count {
        # A pinned number is never reusable, even unreferenced here: a closure
        # capture may carry no GET/SET here, and handing its number to another
        # candidate would alias the open upvalue.
        if ranges[slot].pinned {
            live_to[slot] = never_free
        } elif ranges[slot].first >= 0 {
            live_to[slot] = ranges[slot].last
        }
    }

    # One candidate is placed per round and every candidate is a slot, so the
    # slot count bounds the rounds; the C's unbounded loop ends on the same
    # condition, the round that finds nothing left to place.
    for _attempt in 0..slot_count {
        # Pick the unplaced candidate with the earliest first reference.
        var pick = NO_INDEX
        for slot in reserved..slot_count {
            if ranges[slot].first < 0 or ranges[slot].pinned or placed[slot] { continue }
            if pick < 0 or ranges[slot].first < ranges[pick].first { pick = slot }
        }
        if pick < 0 { break }
        placed[pick] = true

        # Only numbers below `pick` are worth trying: candidates are placed in
        # first-reference order, so a number above it is either still held by an
        # unplaced candidate that starts later and overlaps, or unused.
        var chosen = pick
        for number in reserved..pick {
            if live_to[number] >= ranges[pick].first { continue }
            chosen = number
            break
        }

        target[pick] = chosen
        if chosen != pick {
            # `pick` gives up its own number; whoever comes next may have it.
            live_to[pick] = NO_INDEX
            if ranges[pick].last > live_to[chosen] { live_to[chosen] = ranges[pick].last }
        }
    }
    return target
}
