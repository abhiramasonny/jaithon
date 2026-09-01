#: The open-addressed table rows hash into, and the tallies over it.
from std.gpu import Buffer

from ..column import prefix_sum
from .encode import KeyEncoding
from .lanes import REDUCE_GROUP, REDUCE_GROUPS, flag, kernel_for

#: The group each row falls in, the first row of each group, and how many
#: groups there are.
#:
#: The shared middle of `factorize`, `value_counts`, `mode` and `duplicated`.
#: The caller owns both lanes.
pub fn group_rows(keys: KeyEncoding, dropna: bool) -> (Buffer, Buffer, int) {
    let length = keys.length
    let codes = Buffer(length)
    if length == 0 { return (codes, Buffer(0), 0) }
    let hashes = keys.hash()
    defer { hashes.free() }
    let capacity = next_capacity(length)
    let slots = Buffer(capacity)
    slots.fill_zero()
    defer { slots.free() }
    let slot_of_row = Buffer(length)
    defer { slot_of_row.free() }
    kernel_for("key_table_insert").dispatch_async(
        [keys.packed, slots, hashes, slot_of_row],
        length,
        [length, capacity - 1, keys.lane_count()]
    )
    kernel_for("key_table_claim").dispatch_async([slots, slot_of_row], length, [length])

    let absent = keys.missing ?? keys.packed
    let drop = dropna and keys.missing is not null
    let heads_of_row = Buffer(length)
    defer { heads_of_row.free() }
    let flags = Buffer(length)
    defer { flags.free() }
    kernel_for("key_table_head").dispatch_async(
        [slots, slot_of_row, absent, heads_of_row, flags],
        length,
        [length, flag(drop)]
    )
    let scanned = prefix_sum(flags, length)
    let offsets = scanned[0]
    let group_count = scanned[1]
    defer { offsets.free() }
    kernel_for("key_group_codes").dispatch_async(
        [offsets, heads_of_row, absent, codes],
        length,
        [length, flag(drop)]
    )
    let heads = Buffer(group_count)
    if group_count > 0 {
        kernel_for("key_positions_where").dispatch_async([flags, offsets, heads], length, [length])
    }
    return (codes, heads, group_count)
}

#: A table at least 1.5x the rows, rounded up to a power of two so the probe
#: wraps with a mask instead of a remainder.
fn next_capacity(length: int) -> int {
    let wanted = length + length // 2 + 8
    var capacity = 16
    while capacity < wanted { capacity = capacity << 1 }
    return capacity
}

#: How many rows each group holds, as device uint bits.
pub fn tally_groups(codes: Buffer, length: int, group_count: int) -> Buffer {
    let tally = Buffer(if group_count > 0 { group_count } else { 1 })
    tally.fill_zero()
    if length == 0 or group_count == 0 { return tally }
    kernel_for("key_group_tally").dispatch_async([codes, tally], length, [length])
    return tally
}

#: How many rows survive a `dropna`.
pub fn kept_rows(keys: KeyEncoding, dropna: bool) -> int {
    let absent = keys.missing
    if not dropna or absent is null or keys.length == 0 { return keys.length }
    return keys.length - int(reduce_lane(absent ?? keys.packed, null, keys.length, 0) + 0.5)
}

#: Sum, smallest or largest over a lane, finished on the host over 256
#: partials rather than in a second dispatch — one download of 256 floats
#: against a kernel launch and a device wait.
pub fn reduce_lane(values: Buffer, valid: Buffer?, count: int, mode: int) -> float {
    if count == 0 { return 0.0 }
    let partials = Buffer(REDUCE_GROUPS)
    kernel_for("key_reduce").dispatch_async(
        [values, valid ?? values, partials],
        REDUCE_GROUPS * REDUCE_GROUP,
        [count, flag(valid is not null), mode, REDUCE_GROUPS * REDUCE_GROUP],
        group_size: REDUCE_GROUP
    )
    let read = partials.download()
    partials.free()
    var best = read[0]
    for at in 1..read.len() {
        if mode == 0 { best += read[at] }
        if mode == 1 and read[at] < best { best = read[at] }
        if mode == 2 and read[at] > best { best = read[at] }
    }
    return best
}

#: The rows a flag lane marks, packed, and how many there are.
pub fn positions_where(flags: Buffer, length: int) -> (Buffer, int) {
    let scanned = prefix_sum(flags, length)
    let offsets = scanned[0]
    let kept = scanned[1]
    defer { offsets.free() }
    let out = Buffer(kept)
    if length > 0 and kept > 0 {
        kernel_for("key_positions_where").dispatch_async([flags, offsets, out], length, [length])
    }
    return (out, kept)
}
