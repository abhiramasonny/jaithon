#: The grid constants, the settle budget, and the compiled-kernel cache.
#:
#: Everything the other submodules need before they can dispatch: the sizes
#: that mirror the ones in `shader`, the three lanes a column binds, and the
#: small passes — a tally, a fill, a mask compressed to positions — that more
#: than one of them wants.
import std.gpu as gpu
from std.gpu import Buffer, Kernel
from ..column import Column, Dtype
from ..column import cached_kernel
from .shader import SOURCE

pub const TALLY_GROUP = 256
pub const TALLY_GROUPS = 256
pub const PAIR_GROUP = 256

#: How many times the insert or the probe walk may be sent round again before
#: the loop is called broken.
#:
#: A round is only ever spent on a pair of labels whose 32-bit hashes agreed
#: and whose values did not. A million labels produce on the order of a hundred
#: such pairs, and each of them clears in the round after it is found, so two
#: rounds is the realistic ceiling. Sixty-four is a tripwire, not a budget.
pub const SETTLE_ROUNDS = 64

var _kernels: dict[str, Kernel] = {}

pub fn kernel_for(entry: str) -> Kernel {
    let existing = _kernels.get(entry, null)
    if existing is not null { return existing }
    let compiled = Kernel.compile(SOURCE, entry)
    _kernels[entry] = compiled
    return compiled
}

pub fn require_device(what: str) -> void {
    if not gpu.is_available() {
        throw RuntimeError(f"jaiframe needs a GPU to {what}")
    }
}

pub fn flag(value: bool) -> int { return if value { 1 } else { 0 } }

#: Two's-complement bits of `value` as a non-negative int, because a kernel
#: scalar is a `uint32_t` and a negative one is refused before it reaches Metal.
pub fn u32(value: int) -> int { return value & 0xFFFFFFFF }

#: The three lanes a kernel binds for a column.
#:
#: A column that has no second lane and no validity lane binds its value lane
#: in their place and tells the kernel to ignore it with a flag, which is what
#: `Column` does internally and the only way to pass a lane that may not exist.
pub fn value_lane(column: Column) -> Buffer { return column.values }

pub fn wide_lane(column: Column) -> Buffer { return column.wide ?? column.values }

pub fn valid_lane(column: Column) -> Buffer { return column.valid ?? column.values }

#: Slots for a table over `rows` labels: a power of two, at least twice as many
#: as there are rows.
#:
#: Half full is what keeps the clusters in a linear-probed table short enough
#: that the in-kernel walk is a handful of reads. It also guarantees the walk
#: terminates, because an empty slot always exists.
pub fn table_capacity(rows: int) -> int {
    var capacity = 16
    while capacity < rows * 2 { capacity *= 2 }
    return capacity
}

#: How many entries of a 0/1 lane are set.
pub fn tally(flags: Buffer, count: int) -> int {
    if count == 0 { return 0 }
    let partials = Buffer(TALLY_GROUPS)
    kernel_for("idx_tally").dispatch_async(
        [flags, partials],
        TALLY_GROUPS * TALLY_GROUP,
        [count],
        group_size: TALLY_GROUP
    )
    var total = 0.0
    for part in partials.download() { total += part }
    partials.free()
    return int(total)
}

pub fn fill_u32(lane: Buffer, count: int, value: int) -> void {
    if count == 0 { return }
    kernel_for("idx_fill_u32").dispatch_async([lane], count, [count, u32(value)])
}

#: A `Bool` column over a freshly written flag lane.
pub fn mask_column(length: int) -> Column { return Column.allocate(Dtype.Bool, length) }

#: The rows a mask keeps, as a positions lane and its length.
#:
#: Built by compressing the identity permutation, so the whole thing is the
#: scan and the scatter `Column.compress` already dispatches — nothing walks
#: the mask on the host.
pub fn selected_positions(mask: Column) -> (Buffer, int) {
    let ramp = Column.allocate(Dtype.Float64, mask.len())
    if mask.len() > 0 {
        cached_kernel("col_iota").dispatch_async([ramp.values], mask.len(), [mask.len()])
    }
    let kept = ramp.compress(mask)
    ramp.free()
    return (kept.values, kept.length)
}

#: True where an indexer found a label, or where it did not when `found` is
#: false.
pub fn hit_mask(positions: Buffer, count: int, found: bool) -> Column {
    let mask = mask_column(count)
    if count > 0 {
        kernel_for("idx_hit_mask").dispatch_async(
            [positions, mask.values],
            count,
            [count, flag(found)]
        )
    }
    return mask
}
