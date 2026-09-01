#: A selection by label, and how one resolves to a lane of rows.
from std.gpu import Buffer
from std.math import NAN

from ..column import Column, Dtype, Index, Value
from ..column import NA
from ..column import f32_bits, iota, positions_buffer, wide_pack
from ..index import IndexLookup
from .lanes import flag, kernel_for, valid_lane, value_lane, wide_lane
from .pick import SelectKind, compact_flags, mask_positions, span_positions

#: A selection by label, in the same five shapes `Pick` has.
#:
#: A label spans every level of the index, so each entry is a list of `Value`
#: one per level; `one` and `of` are the single-level spellings. A label that
#: matches several rows selects all of them, as pandas' `loc` does.
pub class Labels {
    #: Which shape this is.
    pub let kind: SelectKind
    #: The labels, one list of level values apiece, for `One` and `Many`.
    pub let keys: list[list[Value]]
    #: First label of a `Span`; `null` is the start of the index.
    pub let start: Value?
    #: Last label of a `Span` — included, as pandas' label slices are.
    pub let stop: Value?
    #: The flags, for `Mask`.
    pub let flags: Column?

    #: An empty shell. The named constructors are the ones to reach for.
    pub fn init(self) {
        self.kind = SelectKind.Every
        self.keys = []
        self.start = null
        self.stop = null
        self.flags = null
    }

    #: One label on a single-level index.
    pub static fn one_of(label: Value) -> Labels {
        return Labels.tuples([[label]])
    }

    #: Several labels on a single-level index, in the order given.
    pub static fn of(labels: list[Value]) -> Labels {
        return Labels.tuples(labels.map(|label| [label]))
    }

    #: Labels on a multi-index: one list of level values per label.
    pub static fn tuples(labels: list[list[Value]]) -> Labels {
        let picked = Labels()
        picked.kind = if labels.len() == 1 { SelectKind.One } else { SelectKind.Many }
        picked.keys = labels.map(|key| [part for part in key])
        return picked
    }

    #: A label range, both ends included.
    #:
    #: pandas closes the right end of a label slice and leaves a positional one
    #: open, on the grounds that a label the caller wrote down is one they
    #: meant. The ends are located by their first and last occurrence, so a
    #: repeated label still selects a contiguous stretch.
    pub static fn span(start: Value? = null, stop: Value? = null) -> Labels {
        let picked = Labels()
        picked.kind = SelectKind.Span
        picked.start = start
        picked.stop = stop
        return picked
    }

    #: The rows a `Bool` column marks — the same selection `Pick.masked` makes,
    #: spelled for a `loc` call site.
    pub static fn masked(flags: Column) -> Labels {
        let picked = Labels()
        picked.kind = SelectKind.Mask
        picked.flags = flags
        return picked
    }

    #: Every row, in index order.
    pub static fn every() -> Labels {
        return Labels()
    }
}

const MATCH_VALUE = 0
const MATCH_MISSING = 1
const MATCH_NONE = 2

#: How to look for `label` in a level, and the bits to look for.
#:
#: A text label that the level's dictionary has never seen answers
#: `MATCH_NONE`, and the caller skips the dispatch entirely: the codes cannot
#: hold a value the table does not name. That short-circuit is the point of
#: resolving the scalar on the host.
fn label_match_bits(level: Column, label: Value) -> (int, int, int) {
    if label.is_na() { return (0, 0, MATCH_MISSING) }
    if level.dtype.is_dictionary() {
        let code = level.level_code(label.to_str())
        if code is null { return (0, 0, MATCH_NONE) }
        return (f32_bits(float(code ?? 0)), 0, MATCH_VALUE)
    }
    if level.dtype.is_wide() {
        let split = wide_pack(label.to_int())
        return (f32_bits(split[0]), f32_bits(split[1]), MATCH_VALUE)
    }
    return (f32_bits(label.to_float()), 0, MATCH_VALUE)
}

#: A flag lane holding 1.0 for the rows whose label is `key`.
#:
#: One dispatch per level over a lane that starts full and is only ever
#: cleared, so a two-level label costs two passes and no intermediate.
fn stored_label_flags(index: Index, key: list[Value]) -> Buffer {
    if key.len() != index.nlevels() {
        throw KeyError(f"a label of {key.len()} parts cannot address {index.nlevels()} levels")
    }
    let length = index.len()
    let flags = Buffer(length)
    if length == 0 { return flags }
    flags.fill(1.0)
    for (at, part) in key.enumerate() {
        let level = index.levels[at]
        let found = label_match_bits(level, part)
        if found[2] == MATCH_NONE {
            flags.fill_zero()
            return flags
        }
        kernel_for("acc_label_match").dispatch_async(
            [value_lane(level), wide_lane(level), valid_lane(level), flags],
            length,
            [
                length,
                flag(level.dtype.is_wide()),
                flag(level.has_validity()),
                found[0],
                found[1],
                flag(found[2] == MATCH_MISSING),
            ]
        )
    }
    return flags
}

#: Where a label sits on a range index, or `null` when it is not on the ramp.
#:
#: Arithmetic, not a lookup: the labels of a range index are not stored, and
#: materialising the ramp to search it would cost a whole column to answer a
#: question one division answers.
fn range_label_position(index: Index, label: Value) -> int? {
    if label.is_na() { return null }
    if label.dtype.is_dictionary() {
        throw KeyError(f"a range index labels rows with integers, not {label.to_str()}")
    }
    let wanted = label.to_int()
    let at = (wanted - index.start) // index.step
    if index.start + at * index.step != wanted { return null }
    if at < 0 or at >= index.len() { return null }
    return at
}

pub fn key_positions(index: Index, key: list[Value]) -> (Buffer, int) {
    if index.is_range() {
        if key.len() != 1 {
            throw KeyError(f"a label of {key.len()} parts cannot address a range index")
        }
        let at = range_label_position(index, key[0])
        if at is null { return (Buffer(0), 0) }
        return (positions_buffer([at ?? 0]), 1)
    }
    let flags = stored_label_flags(index, key)
    let found = compact_flags(flags, index.len())
    flags.free()
    return found
}

pub fn label_text(key: list[Value]) -> str {
    if key.len() == 1 { return key[0].to_str() }
    return "(" + ", ".join(key.map(|part| part.to_str())) + ")"
}

#: The lanes stacked end to end into one, which the caller owns.
fn stack_positions(parts: list[Buffer], counts: list[int]) -> (Buffer, int) {
    var total = 0
    for count in counts { total += count }
    let lane = Buffer(total)
    var at = 0
    for (which, part) in parts.enumerate() {
        if counts[which] == 0 { continue }
        kernel_for("acc_copy_span").dispatch_async([part, lane], counts[which], [counts[which], at])
        at += counts[which]
    }
    return (lane, total)
}

fn label_span_positions(index: Index, start: Value?, stop: Value?) -> (Buffer, int) {
    let length = index.len()
    let first = if start is null { 0 } else { edge_position(index, start ?? NA, true) }
    let last = if stop is null { length - 1 } else { edge_position(index, stop ?? NA, false) }
    if first > last { return (Buffer(0), 0) }
    return span_positions(first, last + 1, 1, length)
}

#: The first or last row a slice endpoint names.
#:
#: Two one-element downloads rather than a download of the matches, because a
#: label that repeats a million times still has only two interesting rows.
fn edge_position(index: Index, label: Value, want_first: bool) -> int {
    if index.is_range() {
        let at = range_label_position(index, label)
        if at is null { throw KeyError(f"no row labelled {label.to_str()}") }
        return at ?? 0
    }
    let found = key_positions(index, [label])
    let lane = found[0]
    defer { lane.free() }
    if found[1] == 0 { throw KeyError(f"no row labelled {label.to_str()}") }
    let slot = if want_first { 0 } else { found[1] - 1 }
    return int(lane.download(slot, 1)[0])
}

#: The dtype of a level, `Int64` for the implied labels of a range index.
fn level_dtype_at(index: Index, level: int) -> Dtype {
    if index.is_range() { return Dtype.Int64 }
    return index.levels[level].dtype
}

#: One level's worth of asked-for labels as a column of the level's own dtype.
#:
#: Built at the level's dtype rather than at the `Value`'s so that `locate`
#: probes without a cast; it reconciles the two dictionaries itself, so a text
#: key needs no level table of its own here.
fn label_column(dtype: Dtype, cells: list[Value]) -> Column {
    let nulls = cells.map(|cell| cell.is_na())
    if dtype.is_dictionary() {
        return Column.from_strs(cells.map(|cell| if cell.is_na() {
            null
        } else {
            cell.to_str()
        }), dtype)
    }
    if dtype.is_wide() {
        return Column.from_ints(cells.map(|cell| if cell.is_na() {
            0
        } else {
            cell.to_int()
        }), nulls, dtype)
    }
    if dtype == Dtype.Bool {
        return Column.from_bools(cells.map(|cell| if cell.is_na() {
            false
        } else {
            cell.to_bool()
        }), nulls)
    }
    return Column.from_floats(cells.map(|cell| if cell.is_na() {
        NAN
    } else {
        cell.to_float()
    }), nulls)
}

#: The asked-for labels as an index to probe `index` with.
fn key_index(index: Index, keys: list[list[Value]]) -> Index {
    let levels = index.nlevels()
    for key in keys {
        if key.len() != levels {
            throw KeyError(f"a label of {key.len()} parts cannot address {levels} levels")
        }
    }
    var columns: list[Column] = []
    for at in 0..levels {
        try {
            columns.push(label_column(level_dtype_at(index, at), keys.map(|key| key[at])))
        } catch error: Error {
            for column in columns { column.free() }
            throw error
        }
    }
    return Index.of(columns)
}

#: The rows the listed labels sit at, through one hash table rather than one
#: scan apiece.
#:
#: `IndexLookup` costs one pass over the index to build and then answers every
#: label in a single probe, so asking for `k` labels out of `n` rows is
#: `n + k` rather than the `k * n` a flag scan per label would be. The lookup
#: reports one row per label, which is the whole answer only while no label
#: repeats — `label_positions` checks that before coming here.
#:
#: The misses come back on the host because the caller wrote the labels down
#: on the host to begin with, and because naming the first absent one in the
#: error needs the `Value` anyway. That download is `k` long, not `n`.
fn probe_positions(lookup: IndexLookup, index: Index, keys: list[list[Value]]) -> (Buffer, int) {
    let target = key_index(index, keys)
    defer { target.free() }
    let count = keys.len()
    let lane = lookup.locate(target)
    if count == 0 { return (lane, 0) }
    for (at, found) in lane.download(0, count).enumerate() {
        if found < 0.0 {
            lane.free()
            throw KeyError(f"no row labelled {label_text(keys[at])}")
        }
    }
    return (lane, count)
}

#: The rows the listed labels sit at when the index repeats a label.
#:
#: One flag scan per label, stacked. A repeat has to expand to every row it
#: names — that is what separates `loc` from a lookup — and a table that
#: reports one representative row cannot say how many rows a label covers.
fn scanned_positions(index: Index, keys: list[list[Value]]) -> (Buffer, int) {
    var parts: list[Buffer] = []
    var counts: list[int] = []
    for key in keys {
        let found = key_positions(index, key)
        if found[1] == 0 {
            for part in parts { part.free() }
            found[0].free()
            throw KeyError(f"no row labelled {label_text(key)}")
        }
        parts.push(found[0])
        counts.push(found[1])
    }
    let stacked = stack_positions(parts, counts)
    for part in parts { part.free() }
    return stacked
}

#: The positions a `Labels` names, as a device lane the caller owns.
pub fn label_positions(index: Index, labels: Labels) -> (Buffer, int) {
    if labels.kind == SelectKind.Every { return (iota(index.len()), index.len()) }
    if labels.kind == SelectKind.Mask {
        let flags = labels.flags
        if flags is null { throw ValueError("a masked selection carries no mask") }
        return mask_positions(flags, index.len())
    }
    if labels.kind == SelectKind.Span {
        return label_span_positions(index, labels.start, labels.stop)
    }
    let lookup = IndexLookup.of(index)
    defer { lookup.free() }
    if not lookup.is_unique() { return scanned_positions(index, labels.keys) }
    return probe_positions(lookup, index, labels.keys)
}
