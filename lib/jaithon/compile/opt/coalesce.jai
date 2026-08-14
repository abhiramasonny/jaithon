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
        var shrunk = proto.max_slots - (old_top - new_top)
        if shrunk < new_top { shrunk = new_top }
        proto.max_slots = shrunk
    }
    return changed
}

fn _scan_ranges(code: Code, ranges: list[SlotRange], slot_count: int) -> bool {
    for (index, instruction) in code.data.enumerate() {
        if instruction.dead { continue }

        if instruction.op == opcode(Op.PushDefer) { return false }

        if instruction.op == opcode(Op.Closure) {
            for which in 0..closure_descriptor_count(instruction) {
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

        if instruction.op == opcode(Op.CloseUpvalue) {
            ranges[slot_of(instruction, 0)].pinned = true
        }
    }
    return true
}

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

fn _assign(code: Code, ranges: list[SlotRange], reserved: int, slot_count: int) -> list[int] {
    let never_free = code.count()
    var target = [slot for slot in 0..slot_count]
    var placed = [false for _slot in 0..slot_count]
    var live_to = [NO_INDEX for _slot in 0..slot_count]

    for slot in 0..slot_count {
        if ranges[slot].pinned {
            live_to[slot] = never_free
        } elif ranges[slot].first >= 0 {
            live_to[slot] = ranges[slot].last
        }
    }

    for _attempt in 0..slot_count {
        var pick = NO_INDEX
        for slot in reserved..slot_count {
            if ranges[slot].first < 0 or ranges[slot].pinned or placed[slot] { continue }
            if pick < 0 or ranges[slot].first < ranges[pick].first { pick = slot }
        }
        if pick < 0 { break }
        placed[pick] = true

        var chosen = pick
        for number in reserved..pick {
            if live_to[number] >= ranges[pick].first { continue }
            chosen = number
            break
        }

        target[pick] = chosen
        if chosen != pick {
            live_to[pick] = NO_INDEX
            if ranges[pick].last > live_to[chosen] { live_to[chosen] = ranges[pick].last }
        }
    }
    return target
}
