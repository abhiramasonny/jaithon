#: Labels on the host side: naming a level, inferring what it holds, and
#: putting two indexes' text levels onto one dictionary.
#:
#: Nothing here decides where a label sits. It is the plumbing every plan
#: needs first — `recoded_like` above all, which is what makes a code from one
#: index comparable with a code from another, and `borrowed_levels`, which is
#: why reading an index does not copy it.
from std.gpu import Buffer
from ..column import Column, Dtype, Index, Value
from ..column import cached_kernel, f32_bits, wide_pack
from .kernels import flag, kernel_for, mask_column, u32
from .kernels import valid_lane, value_lane, wide_lane

pub fn names_of(index: Index) -> list[str?] { return [name for name in index.names] }

pub fn default_level_name(at: int, total: int) -> str {
    return if total == 1 { "index" } else { f"level_{at}" }
}

#: A level named by its position, negative from the end, or by its name.
pub fn resolve_level(index: Index, level: any) -> int {
    let total = index.nlevels()
    if level is int {
        let position: int = level
        let at = if position < 0 { total + position } else { position }
        if at < 0 or at >= total {
            throw IndexError(f"level {position} is outside an index of {total} levels")
        }
        return at
    }
    if level is str {
        for (at, name) in index.names.enumerate() {
            if name == level { return at }
        }
        throw KeyError(f"no index level named {level}")
    }
    throw TypeError("an index level is named by its position or by its name")
}

#: How one level of an index stores its labels. A range index stores none, and
#: the ramp it stands for is `Int64`.
pub fn level_dtype(index: Index, level: int) -> Dtype {
    if index.is_range() { return Dtype.Int64 }
    return index.levels[level].dtype
}

#: The name two indexes agree on for a level, `null` where they do not.
#:
#: pandas drops a name a set operation cannot justify keeping, and a
#: disagreement is exactly that case.
pub fn shared_names(left: Index, right: Index) -> list[str?] {
    var names: list[str?] = []
    for (at, name) in left.names.enumerate() {
        names.push(if at < right.names.len() and name == right.names[at] { name } else { null })
    }
    return names
}

#: The dtype a list of cells wants: the first one that is not missing, and
#: `Float64` when every one of them is.
pub fn inferred_dtype(cells: list[Value]) -> Dtype {
    for cell in cells {
        if not cell.is_na() { return cell.dtype }
    }
    return Dtype.Float64
}

#: Host cells as a column of `dtype`.
#:
#: The boundary where labels a program wrote down become a device column. The
#: validity list is left null when nothing is missing, so the column does not
#: carry a lane of ones for a kernel to read.
pub fn column_of(cells: list[Value], dtype: Dtype) -> Column {
    var missing = false
    for cell in cells {
        if cell.is_na() { missing = true }
    }
    if dtype.is_dictionary() {
        var text: list[str?] = []
        for cell in cells { text.push(if cell.is_na() { null } else { cell.to_str() }) }
        return Column.from_strs(text, dtype)
    }
    var nulls: list[bool]? = null
    if missing { nulls = cells.map(|cell| cell.is_na()) }
    if dtype.is_wide() {
        return Column.from_ints(cells.map(|cell| if cell.is_na() {
            0
        } else {
            cell.to_int()
        }), nulls, dtype)
    }
    if dtype == Dtype.Bool {
        return Column.from_bools(cells.map(|cell| not cell.is_na() and cell.to_bool()), nulls)
    }
    return Column.from_floats(cells.map(|cell| if cell.is_na() {
        0.0
    } else {
        cell.to_float()
    }), nulls)
}

#: A one-row index carrying one label, for the single-label lookups.
pub fn index_of_key(key: list[Value], dtypes: list[Dtype]) -> Index {
    if key.len() != dtypes.len() {
        throw ValueError(f"a label of {key.len()} levels cannot index {dtypes.len()}")
    }
    return index_of_key_columns(key.map(|cell| [cell]), dtypes)
}

#: An index over one host list of cells per level.
pub fn index_of_key_columns(cells: list[list[Value]], dtypes: list[Dtype]) -> Index {
    var levels: list[Column] = []
    for (at, column_cells) in cells.enumerate() {
        levels.push(column_of(column_cells, dtypes[at]))
    }
    return Index.of(levels)
}

pub fn spell_key(key: list[Value]) -> str {
    if key.len() == 1 { return key[0].to_str() }
    var parts: list[str] = []
    for cell in key { parts.push(cell.to_str()) }
    let joined = parts.join(", ")
    return f"({joined})"
}

#: `source` recoded onto `model`'s dictionary, so that the two columns' codes
#: mean the same string.
#:
#: The model's own codes are left alone — its table is the prefix of the merged
#: one — which is what lets a lookup hash an index once and reuse that hash
#: against any number of targets. A string the model has never seen is given a
#: code past the model's table, so it compares equal to nothing, which is
#: exactly the miss it should be.
#:
#: Returns `source` itself when there is nothing to do, so the caller has to
#: compare before freeing.
pub fn recoded_like(source: Column, model: Column) -> Column {
    if not model.dtype.is_dictionary() {
        if source.dtype != model.dtype {
            throw TypeError(
                f"cannot look a {source.dtype.name()} label up in a {model.dtype.name()} index"
            )
        }
        return source
    }
    if not source.dtype.is_dictionary() {
        throw TypeError(
            f"cannot look a {source.dtype.name()} label up in a {model.dtype.name()} index"
        )
    }
    var names: list[str] = [name for name in model.levels ?? []]
    var intern: dict[str, int] = {}
    for (code, name) in names.enumerate() { intern[name] = code }
    var novel = false
    for name in source.levels ?? [] {
        if intern.get(name, -1) >= 0 { continue }
        intern[name] = names.len()
        names.push(name)
        novel = true
    }
    if not novel and source.dtype == model.dtype and names.len() == source.level_count() {
        var identical_table = true
        for (code, name) in (source.levels ?? []).enumerate() {
            if names[code] != name { identical_table = false }
        }
        if identical_table { return source }
    }
    let out = Column.allocate(model.dtype, source.length, with_validity: source.has_validity())
    out.set_levels(names)
    let mask = source.valid
    let out_mask = out.valid
    if mask is not null and out_mask is not null and source.length > 0 {
        cached_kernel("col_copy_lane").dispatch_async(
            [mask, out_mask],
            source.length,
            [source.length, 0, 0]
        )
    }
    if source.length == 0 { return out }
    let table = source.levels ?? []
    if table.len() == 0 {
        cached_kernel("col_copy_lane").dispatch_async(
            [source.values, out.values],
            source.length,
            [source.length, 0, 0]
        )
        return out
    }
    var mapping: list[float] = []
    for name in table { mapping.push(float(intern.get(name, 0))) }
    let lane = Buffer(mapping.len())
    lane.upload(mapping)
    cached_kernel("col_remap").dispatch_async(
        [source.values, out.values, lane],
        source.length,
        [source.length, table.len()]
    )
    lane.free()
    return out
}

#: An index's levels as columns to read, and whether the caller has to free
#: them.
#:
#: A stored index lends its own columns; only a range index has to build the
#: ramp it stands for. Copying instead would put a full pass over the labels in
#: front of every lookup, alignment and comparison, which for an index that is
#: only ever read is a pass for nothing.
pub fn borrowed_levels(index: Index) -> (list[Column], bool) {
    if index.is_range() { return (index.materialize(), true) }
    return ([level for level in index.levels], false)
}

pub fn release_levels(levels: list[Column], owned: list[bool]) -> void {
    for (at, level) in levels.enumerate() {
        if owned[at] { level.free() }
    }
}

pub fn release_all(levels: list[Column], owned: bool) -> void {
    if not owned { return }
    for level in levels { level.free() }
}

#: `target`'s levels as columns comparable with `model`'s, with a flag per
#: level saying whether it was built here or borrowed from the index.
pub fn aligned_levels(model: list[Column], target: Index) -> (list[Column], list[bool]) {
    let borrowed = borrowed_levels(target)
    let raw = borrowed[0]
    let built = borrowed[1]
    if raw.len() != model.len() {
        release_all(raw, built)
        throw ValueError(f"an index of {raw.len()} levels cannot be looked up in {model.len()}")
    }
    var probes: list[Column] = []
    var owned: list[bool] = []
    for (at, level) in raw.enumerate() {
        var recoded = level
        try {
            recoded = recoded_like(level, model[at])
        } catch error: Error {
            release_levels(probes, owned)
            release_all(raw, built)
            throw error
        }
        if recoded != level {
            if built { level.free() }
            probes.push(recoded)
            owned.push(true)
        } else {
            probes.push(level)
            owned.push(built)
        }
    }
    return (probes, owned)
}

#: A `Bool` column, true where the column's label is `key`.
pub fn matches_scalar(column: Column, key: Value) -> Column {
    let mask = mask_column(column.len())
    if column.len() == 0 { return mask }
    if key.is_na() {
        if not column.has_validity() {
            mask.values.fill_zero()
            return mask
        }
        dispatch_match(column, mask, 0, 0, true)
        return mask
    }
    if column.dtype.is_dictionary() {
        let code = column.level_code(key.to_str())
        if code is null {
            #: A string the level has never held cannot label a row, and saying
            #: so costs a dictionary lookup rather than a pass over the column.
            mask.values.fill_zero()
            return mask
        }
        dispatch_match(column, mask, f32_bits(float(code ?? 0)), 0, false)
        return mask
    }
    if column.dtype.is_wide() {
        let split = wide_pack(key.to_int())
        dispatch_match(column, mask, f32_bits(split[0]), f32_bits(split[1]), false)
        return mask
    }
    dispatch_match(column, mask, f32_bits(key.to_float()), 0, false)
    return mask
}

pub fn dispatch_match(column: Column, mask: Column, low: int, high: int, want_null: bool) -> void {
    kernel_for("idx_match_scalar").dispatch_async(
        [value_lane(column), wide_lane(column), valid_lane(column), mask.values],
        column.len(),
        [
            column.len(),
            flag(column.dtype.is_wide()),
            flag(column.has_validity()),
            u32(low),
            u32(high),
            flag(want_null),
        ]
    )
}
