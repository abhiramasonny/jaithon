#: Building a multi-index, rearranging the levels of one, and the two calls
#: that move labels between the index and the columns.
#:
#: `set_index` and `reset_index` are that pair. Everything else here either
#: constructs an index from host data or drops, reorders, renames or reads
#: back a level the index already has.
from std.gpu import Buffer
from ..column import Column, DataFrame, Index, Value
from ..column import MAX_ROWS, iota
from .kernels import kernel_for, require_device, selected_positions
from .labels import column_of, default_level_name, inferred_dtype
from .labels import matches_scalar, resolve_level

#: A multi-index over one column per level.
#:
#: The columns are copied, so the caller keeps whatever it passed in. pandas
#: spells this `MultiIndex.from_arrays`; `Index.of`, in column.jai, is the
#: adopting form for a caller that just built the columns.
pub fn from_arrays(arrays: list[Column], names: list[str?]? = null) -> Index {
    if arrays.len() == 0 { throw ValueError("an index needs at least one level") }
    return Index.of(arrays.map(|level| level.copy()), names)
}

#: A multi-index over one label tuple per row.
#:
#: Host data, so the cost is the boxed cell per level per row that the caller
#: already paid to write them down. Each level takes its dtype from its first
#: cell that is not missing.
pub fn from_tuples(tuples: list[list[Value]], names: list[str?]? = null) -> Index {
    if tuples.len() == 0 { throw ValueError("an index needs at least one label") }
    let width = tuples[0].len()
    if width == 0 { throw ValueError("an index label needs at least one level") }
    var levels: list[Column] = []
    for at in 0..width {
        var cells: list[Value] = []
        for row in tuples {
            if row.len() != width {
                for level in levels { level.free() }
                throw ValueError(f"a label of {row.len()} levels among labels of {width}")
            }
            cells.push(row[at])
        }
        levels.push(column_of(cells, inferred_dtype(cells)))
    }
    return Index.of(levels, names)
}

#: Every combination of the levels, the last varying fastest.
#:
#: The digits are worked out on the device — one division per level, walked
#: from the fastest-varying end so the remainder carries down and no level
#: needs a modulo. Each level then gathers itself through its own digits, so a
#: product of a thousand by a thousand is two dispatches and never a host loop
#: over the million rows it makes.
pub fn from_product(levels: list[Column], names: list[str?]? = null) -> Index {
    if levels.len() == 0 { throw ValueError("a product needs at least one level") }
    require_device("build a product index")
    var total = 1
    var sizes: list[float] = []
    for level in levels {
        if level.len() == 0 { total = 0 }
        sizes.push(float(level.len()))
    }
    if total != 0 {
        for level in levels { total *= level.len() }
    }
    if total > MAX_ROWS {
        throw ValueError(f"a product of {total} rows exceeds the {MAX_ROWS} a lane addresses")
    }
    if total == 0 {
        let nothing = iota(0)
        let empty = levels.map(|level| level.take(nothing, 0))
        nothing.free()
        return Index.of(empty, names)
    }
    let extents = Buffer(sizes.len())
    extents.upload(sizes)
    let digits = Buffer(levels.len() * total)
    kernel_for("idx_product_digits").dispatch_async([extents, digits], total, [total, levels.len()])
    var built: list[Column] = []
    for (at, level) in levels.enumerate() {
        let window = digits.view(at * total, total)
        built.push(level.take(window, total))
        window.free()
    }
    extents.free()
    digits.free()
    return Index.of(built, names)
}

#: The index with one level taken out.
pub fn droplevel(index: Index, level: any = 0) -> Index {
    let at = resolve_level(index, level)
    if index.nlevels() < 2 {
        throw ValueError("an index cannot drop its only level")
    }
    var levels: list[Column] = []
    var names: list[str?] = []
    for (position, column) in index.levels.enumerate() {
        if position == at { continue }
        levels.push(column.copy())
        names.push(index.names[position])
    }
    return Index.of(levels, names)
}

#: The index with two levels exchanged. The last two by default, as pandas
#: does, because that is the pair a `stack` and an `unstack` move.
pub fn swaplevel(index: Index, first: any = -2, second: any = -1) -> Index {
    let a = resolve_level(index, first)
    let b = resolve_level(index, second)
    var order: list[int] = []
    for position in 0..index.nlevels() {
        order.push(if position == a { b } elif position == b { a } else { position })
    }
    return reorder_levels(index, order)
}

#: The index with its levels in the order `order` names.
pub fn reorder_levels(index: Index, order: list[int]) -> Index {
    if index.is_range() {
        throw ValueError("a range index has one level, so there is nothing to reorder")
    }
    if order.len() != index.nlevels() {
        throw ValueError(f"{order.len()} positions cannot reorder {index.nlevels()} levels")
    }
    var levels: list[Column] = []
    var names: list[str?] = []
    var seen: dict[int, bool] = {}
    for position in order {
        let at = if position < 0 { index.nlevels() + position } else { position }
        if at < 0 or at >= index.nlevels() {
            for level in levels { level.free() }
            throw IndexError(f"level {position} is outside an index of {index.nlevels()}")
        }
        if seen.get(at, false) {
            for level in levels { level.free() }
            throw ValueError(f"level {at} appears twice in the new order")
        }
        seen[at] = true
        levels.push(index.levels[at].copy())
        names.push(index.names[at])
    }
    return Index.of(levels, names)
}

#: One level's labels as a column, freshly allocated.
#:
#: A range index materialises into the `Int64` ramp it stands for, which is the
#: one place the implied labels become storage.
pub fn get_level_values(index: Index, level: any = 0) -> Column {
    let at = resolve_level(index, level)
    if index.is_range() {
        let built = index.materialize()
        return built[0]
    }
    return index.levels[at].copy()
}

#: The index with its level names replaced.
pub fn set_names(index: Index, names: list[str?]) -> Index {
    if names.len() != index.nlevels() {
        throw ValueError(f"{names.len()} names cannot name {index.nlevels()} levels")
    }
    if index.is_range() {
        return Index.range(index.len(), start: index.start, step: index.step, name: names[0])
    }
    return Index.of(index.levels.map(|level| level.copy()), [name for name in names])
}

#: The frame with its index levels renamed.
#:
#: pandas takes a scalar for a single-level index; here a one-element list says
#: the same thing without a second spelling of the argument.
pub fn rename_axis(frame: DataFrame, names: list[str?]) -> DataFrame {
    return frame._replace(index: set_names(frame.index, names))
}

#: The rows whose label on one level is `key`, with that level taken out.
#:
#: The comparison against `key` is one dispatch — a scalar resolved once on the
#: host and compared as an integer, which for a text level means resolving it
#: to a dictionary code and, when the level has never seen it, answering with
#: no rows and no dispatch at all.
#:
#: A single-level index keeps its level whatever `drop_level` says: there would
#: be nothing left to label the rows with.
pub fn xs(frame: DataFrame, key: Value, level: any = 0, drop_level: bool = true) -> DataFrame {
    let at = resolve_level(frame.index, level)
    let labels = get_level_values(frame.index, at)
    let mask = matches_scalar(labels, key)
    labels.free()
    let picked = selected_positions(mask)
    mask.free()
    let rows = picked[1]
    var columns: list[Column] = []
    for column in frame.columns { columns.push(column.take(picked[0], rows)) }
    let taken = frame.index.take(picked[0], rows)
    picked[0].free()
    if not drop_level or taken.nlevels() < 2 {
        return DataFrame([name for name in frame.names], columns, index: taken)
    }
    let narrowed = droplevel(taken, at)
    taken.free()
    return DataFrame([name for name in frame.names], columns, index: narrowed)
}

#: The frame relabelled by some of its own columns.
#:
#: `drop` leaves those columns out of the result, which is what pandas does by
#: default; `append` puts them under the index the frame already had rather
#: than replacing it. The index takes copies, so the result and the original
#: can both be freed.
pub fn set_index(
    frame: DataFrame,
    keys: list[str],
    drop: bool = true,
    append: bool = false
) -> DataFrame {
    if keys.len() == 0 { throw ValueError("set_index needs at least one column") }
    var levels: list[Column] = []
    var names: list[str?] = []
    if append {
        for level in frame.index.materialize() { levels.push(level) }
        for name in frame.index.names { names.push(name) }
    }
    var moved: dict[str, bool] = {}
    for key in keys {
        if moved.get(key, false) {
            for level in levels { level.free() }
            throw ValueError(f"column {key} is named twice")
        }
        moved[key] = true
        levels.push(frame.column(key).copy())
        names.push(key)
    }
    var kept: list[str] = []
    var columns: list[Column] = []
    for (at, name) in frame.names.enumerate() {
        if drop and moved.get(name, false) { continue }
        kept.push(name)
        columns.push(frame.columns[at])
    }
    return DataFrame(kept, columns, index: Index.of(levels, names))
}

#: The frame's index levels moved back into columns.
#:
#: `level` names which of them to move, all of them when it is null. `drop`
#: throws them away instead of inserting them. The levels land at the front, in
#: index order, which is where pandas puts them. A level with no name becomes
#: `index` on a single-level index and `level_0`, `level_1` ... on a
#: multi-index, again as pandas does.
pub fn reset_index(frame: DataFrame, level: list[int]? = null, drop: bool = false) -> DataFrame {
    let total = frame.index.nlevels()
    var moving: list[int] = []
    if level is null {
        for at in 0..total { moving.push(at) }
    } else {
        for entry in level ?? [] { moving.push(resolve_level(frame.index, entry)) }
    }
    var taken: dict[int, bool] = {}
    for at in moving { taken[at] = true }
    var names: list[str] = []
    var columns: list[Column] = []
    if not drop {
        let materialized = frame.index.materialize()
        for at in moving {
            names.push(frame.index.names[at] ?? default_level_name(at, total))
            columns.push(materialized[at])
        }
        for (at, level_column) in materialized.enumerate() {
            if not taken.get(at, false) { level_column.free() }
        }
    }
    for (at, name) in frame.names.enumerate() {
        names.push(name)
        columns.push(frame.columns[at])
    }
    var remaining: list[Column] = []
    var remaining_names: list[str?] = []
    if not frame.index.is_range() {
        for (at, level_column) in frame.index.levels.enumerate() {
            if taken.get(at, false) { continue }
            remaining.push(level_column.copy())
            remaining_names.push(frame.index.names[at])
        }
    }
    let rebuilt = if remaining.len() == 0 {
        Index.range(frame.len())
    } else {
        Index.of(remaining, remaining_names)
    }
    return DataFrame(names, columns, index: rebuilt)
}
