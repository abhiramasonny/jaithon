#: A selection by position, and how one resolves to a lane of rows.
from std.gpu import Buffer

from ..column import Column, Dtype
from ..column import iota, positions_buffer, prefix_sum
from .lanes import flag, kernel_for, u32, valid_lane, value_lane

#: Which of the five shapes a selection is.
#:
#: One enum for both `Pick` and `Labels`, because the shapes are the same
#: whether the subscript names positions or labels — and because `iloc` and
#: `loc` then differ only in how a shape resolves, not in what shapes exist.
pub enum SelectKind {
    #: A single row.
    One,
    #: A list of rows, in the order given, repeats allowed.
    Many,
    #: A half-open range with a step, as Python's slice is.
    Span,
    #: A `Bool` column, one flag per row.
    Mask,
    #: Every row, in order.
    Every,
}

#: A selection by position: one row, a list, a slice with a step, a boolean
#: mask, or everything.
#:
#: Negative positions count from the end, as pandas' do. A `Pick` borrows the
#: mask it was built from and never frees it.
pub class Pick {
    #: Which shape this is.
    pub let kind: SelectKind
    #: The row, for `One`.
    pub let one: int
    #: The rows, for `Many`.
    pub let many: list[int]
    #: First row of a `Span`; `null` is the near end for the step's direction.
    pub let start: int?
    #: One past the last row of a `Span`; `null` is the far end.
    pub let stop: int?
    #: Distance between the rows of a `Span`, negative to walk backwards.
    pub let step: int
    #: The flags, for `Mask`.
    pub let flags: Column?

    #: An empty shell. The named constructors are the ones to reach for.
    pub fn init(self) {
        self.kind = SelectKind.Every
        self.one = 0
        self.many = []
        self.start = null
        self.stop = null
        self.step = 1
        self.flags = null
    }

    #: One row, which `iloc` still returns as a one-row frame. `iat` is the
    #: call that returns the cell.
    pub static fn one_at(position: int) -> Pick {
        let pick = Pick()
        pick.kind = SelectKind.One
        pick.one = position
        return pick
    }

    #: The listed rows, in that order. Repeats are kept, which is what makes
    #: this the primitive an upsampling reindex is built on.
    pub static fn of(positions: list[int]) -> Pick {
        let pick = Pick()
        pick.kind = SelectKind.Many
        pick.many = [row for row in positions]
        return pick
    }

    #: A slice, with Python's bounds: `stop` is excluded, a null end runs to
    #: the edge, and a negative step walks backwards.
    pub static fn span(start: int? = null, stop: int? = null, step: int = 1) -> Pick {
        if step == 0 { throw ValueError("a slice of step 0 never advances") }
        let pick = Pick()
        pick.kind = SelectKind.Span
        pick.start = start
        pick.stop = stop
        pick.step = step
        return pick
    }

    #: The rows a `Bool` column marks. A null flag drops its row, the same way
    #: `Column.compress` treats one.
    pub static fn masked(flags: Column) -> Pick {
        let pick = Pick()
        pick.kind = SelectKind.Mask
        pick.flags = flags
        return pick
    }

    #: Every row, in order — the `:` of a two-axis subscript.
    pub static fn every() -> Pick {
        return Pick()
    }
}

pub fn resolve_at(position: int, length: int) -> int {
    let at = if position < 0 { length + position } else { position }
    if at < 0 or at >= length {
        throw IndexError(f"row {position} is outside a frame of {length}")
    }
    return at
}

pub fn clamp_at(raw: int, length: int, low: int, high: int) -> int {
    var at = if raw < 0 { length + raw } else { raw }
    if at < low { at = low }
    if at > high { at = high }
    return at
}

#: The first row of a slice and how many rows it has, with Python's bounds.
fn span_bounds(start: int?, stop: int?, step: int, length: int) -> (int, int) {
    if step == 0 { throw ValueError("a slice of step 0 never advances") }
    if step > 0 {
        let first = clamp_at(start ?? 0, length, 0, length)
        let limit = clamp_at(stop ?? length, length, 0, length)
        let span = limit - first
        return (first, if span <= 0 { 0 } else { (span + step - 1) // step })
    }
    let first = clamp_at(start ?? length - 1, length, -1, length - 1)
    let limit = if stop is null { -1 } else { clamp_at(stop ?? 0, length, -1, length - 1) }
    let span = first - limit
    let back = -step
    return (first, if span <= 0 { 0 } else { (span + back - 1) // back })
}

#: The positions a strided slice names, written straight onto the device.
pub fn span_positions(start: int?, stop: int?, step: int, length: int) -> (Buffer, int) {
    let bounds = span_bounds(start, stop, step, length)
    let first = bounds[0]
    let count = bounds[1]
    let lane = Buffer(count)
    if count == 0 { return (lane, 0) }
    if count % 4 == 0 {
        kernel_for("acc_span_positions4").dispatch_async(
            [lane],
            count // 4,
            [count // 4, u32(first), u32(step)]
        )
        return (lane, count)
    }
    kernel_for("acc_span_positions").dispatch_async([lane], count, [count, u32(first), u32(step)])
    return (lane, count)
}

#: A `Bool` column as a lane of 0/1 flags, with a null flag counting as false.
pub fn keep_flags(mask: Column, length: int) -> Buffer {
    if mask.dtype != Dtype.Bool {
        throw TypeError(f"a selection mask must be bool, not {mask.dtype.name()}")
    }
    if mask.len() != length {
        throw ValueError(f"a mask of {mask.len()} cannot select from {length} rows")
    }
    let flags = Buffer(length)
    if length == 0 { return flags }
    let has_valid = flag(mask.has_validity())
    if length % 4 == 0 {
        kernel_for("acc_keep_flags4").dispatch_async(
            [value_lane(mask), valid_lane(mask), flags],
            length // 4,
            [length // 4, has_valid]
        )
        return flags
    }
    kernel_for("acc_keep_flags").dispatch_async(
        [value_lane(mask), valid_lane(mask), flags],
        length,
        [length, has_valid]
    )
    return flags
}

#: The positions a flag lane marks, and how many there are.
#:
#: One scan and one scatter for the whole frame. Every masked selection here
#: goes through this rather than through `Column.compress` per column, which
#: would flag and scan the same rows once per column.
pub fn compact_flags(flags: Buffer, length: int) -> (Buffer, int) {
    let scanned = prefix_sum(flags, length)
    let offsets = scanned[0]
    let kept = scanned[1]
    let lane = Buffer(kept)
    if kept > 0 {
        kernel_for("acc_take_positions").dispatch_async([flags, offsets, lane], length, [length])
    }
    offsets.free()
    return (lane, kept)
}

pub fn mask_positions(mask: Column, length: int) -> (Buffer, int) {
    let flags = keep_flags(mask, length)
    let found = compact_flags(flags, length)
    flags.free()
    return found
}

#: The positions a `Pick` names, as a device lane the caller owns.
pub fn pick_positions(pick: Pick, length: int) -> (Buffer, int) {
    if pick.kind == SelectKind.Every { return (iota(length), length) }
    if pick.kind == SelectKind.One { return (positions_buffer([resolve_at(pick.one, length)]), 1) }
    if pick.kind == SelectKind.Many {
        let rows = pick.many.map(|row| resolve_at(row, length))
        return (positions_buffer(rows), rows.len())
    }
    if pick.kind == SelectKind.Span {
        return span_positions(pick.start, pick.stop, pick.step, length)
    }
    let flags = pick.flags
    if flags is null { throw ValueError("a masked selection carries no mask") }
    return mask_positions(flags, length)
}

#: The positions a `Pick` names, as a host list.
#:
#: For the column axis only, where the count is the frame's width and a
#: hundred columns is a wide frame. The row axis never comes through here.
pub fn pick_host(pick: Pick, length: int) -> list[int] {
    if pick.kind == SelectKind.Every { return [at for at in 0..length] }
    if pick.kind == SelectKind.One { return [resolve_at(pick.one, length)] }
    if pick.kind == SelectKind.Many { return pick.many.map(|at| resolve_at(at, length)) }
    if pick.kind == SelectKind.Span {
        let bounds = span_bounds(pick.start, pick.stop, pick.step, length)
        return [bounds[0] + step_of * pick.step for step_of in 0..bounds[1]]
    }
    let flags = pick.flags
    if flags is null { throw ValueError("a masked selection carries no mask") }
    if flags.dtype != Dtype.Bool {
        throw TypeError(f"a selection mask must be bool, not {flags.dtype.name()}")
    }
    if flags.len() != length {
        throw ValueError(f"a mask of {flags.len()} cannot select from {length} columns")
    }
    var out: list[int] = []
    for (at, kept) in flags.to_bools().enumerate() {
        if kept { out.push(at) }
    }
    return out
}
