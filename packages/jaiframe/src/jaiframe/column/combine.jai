#: Columns stacked end to end, and the dictionary tables that reconciles.
from std.gpu import Buffer
from .lanes import MAX_ROWS, cached_kernel, copy_lane
from .store import Column
from .types import Dtype

#: Every part's dictionary rebuilt onto one shared table.
#:
#: Codes are only meaningful against the table they were assigned from, so two
#: text columns cannot be compared, hashed, joined or concatenated until they
#: agree on one. Levels come out in first-appearance order, scanning the parts
#: in order, which leaves the first part's codes unchanged in the common case.
#:
#: Returns fresh columns the caller owns; the inputs are untouched. The table
#: is a host `list[str]`, so this costs a host loop over the distinct values —
#: not over the rows, which a kernel remaps.
pub fn unify_levels(parts: list[Column]) -> list[Column] {
    if parts.len() == 0 { return [] }
    let dtype = parts[0].dtype
    if not dtype.is_dictionary() {
        throw TypeError(f"{dtype.name()} has no dictionary to unify")
    }
    var levels: list[str] = []
    var intern: dict[str, int] = {}
    for part in parts {
        if part.dtype != dtype {
            throw TypeError(f"cannot unify {dtype.name()} with {part.dtype.name()}")
        }
        for name in part.levels ?? [] {
            if intern.get(name, -1) >= 0 { continue }
            intern[name] = levels.len()
            levels.push(name)
        }
    }
    var out: list[Column] = []
    for part in parts { out.push(remap_levels(part, levels, intern)) }
    return out
}

#: The parts stacked end to end into one new column.
#:
#: Every part must already carry the same dtype. Promoting an `Int64` beside a
#: `Float64` is a decision with a precision cost, so it belongs to the caller
#: and to `cast`, not to a concatenation that would make it quietly. Dictionary
#: parts are unified first, so the result's codes are on one table.
pub fn concat_columns(parts: list[Column]) -> Column {
    if parts.len() == 0 {
        throw ValueError("a concatenation needs at least one column to take a dtype from")
    }
    let dtype = parts[0].dtype
    var total = 0
    var any_missing = false
    for part in parts {
        if part.dtype != dtype {
            throw TypeError(f"cannot concatenate {part.dtype.name()} onto {dtype.name()}")
        }
        total += part.length
        if part.has_validity() { any_missing = true }
    }
    if total > MAX_ROWS {
        throw ValueError(f"{total} rows exceeds the {MAX_ROWS} a float32 position lane addresses")
    }
    if not dtype.is_dictionary() { return stack_lanes(parts, dtype, total, any_missing) }
    let unified = unify_levels(parts)
    let out = stack_lanes(unified, dtype, total, any_missing)
    for part in unified { part.free() }
    return out
}

fn remap_levels(part: Column, levels: list[str], intern: dict[str, int]) -> Column {
    let mapped = Column.allocate(part.dtype, part.length, with_validity: part.has_validity())
    mapped.set_levels([name for name in levels])
    let mask = part.valid
    let out_mask = mapped.valid
    if mask is not null and out_mask is not null { copy_lane(mask, out_mask, part.length) }
    if part.length == 0 { return mapped }
    let table = part.levels ?? []
    if table.len() == 0 {
        copy_lane(part.values, mapped.values, part.length)
        return mapped
    }
    var mapping: list[float] = []
    for name in table { mapping.push(float(intern.get(name, 0))) }
    let lane = Buffer(mapping.len())
    lane.upload(mapping)
    cached_kernel("col_remap").dispatch_async(
        [part.values, mapped.values, lane],
        part.length,
        [part.length, table.len()]
    )
    lane.free()
    return mapped
}

fn stack_lanes(parts: list[Column], dtype: Dtype, total: int, any_missing: bool) -> Column {
    let out = Column.allocate(dtype, total, with_validity: any_missing)
    let out_mask = out.valid
    #: Ones first, then the parts that carry a mask write over their stretch. A
    #: part with no mask has no missing row, so its stretch is already right.
    if out_mask is not null and total > 0 { out_mask.fill(1.0) }
    if dtype.is_dictionary() and parts.len() > 0 {
        out.set_levels([name for name in parts[0].levels ?? []])
    }
    var at = 0
    for part in parts {
        if part.length == 0 { continue }
        copy_lane(part.values, out.values, part.length, 0, at)
        let high = part.wide
        let out_high = out.wide
        if high is not null and out_high is not null {
            copy_lane(high, out_high, part.length, 0, at)
        }
        let mask = part.valid
        if mask is not null and out_mask is not null {
            copy_lane(mask, out_mask, part.length, 0, at)
        }
        at += part.length
    }
    return out
}
