#: `merge`, the index `join` written in terms of it, and the forward fill.
#:
#: Everything here is a pairing turned back into a frame: which columns key,
#: which of them collapse to one, what the suffixes rename, and which side's
#: labels the result carries.
from std.gpu import Buffer
from ..column import Column, DataFrame, Index, prefix_sum
from .kernels import kernel_for, require_device
from .pairing import Pairing, indicator_column, merged_key_column, names_of
from .pairing import pair_rows, sort_pairing, unified_pair
from .types import How, Validate

#: How many positions in `lane` are named by more than one row.
fn duplicate_positions(lane: Buffer, rows: int, extent: int) -> int {
    if rows == 0 or extent == 0 { return 0 }
    let counts = Buffer(extent)
    counts.fill_zero()
    kernel_for("jn_count_pairs").dispatch_async([lane, counts], rows, [rows])
    let flags = Buffer(extent)
    kernel_for("jn_over_one").dispatch_async([counts, flags], extent, [extent])
    counts.free()
    let scanned = prefix_sum(flags, extent)
    flags.free()
    scanned[0].free()
    return scanned[1]
}

#: Refuse a pairing that breaks the promise `rule` made.
#:
#: A right row paired with two left rows is a duplicate key on the *left*, and
#: the other way round — so the multiplicity the build side's chains produced
#: is read straight off the finished pairing rather than counted again.
fn check_validate(pairing: Pairing, n_left: int, n_right: int, rule: Validate) -> void {
    if rule.left_unique() and duplicate_positions(pairing.right, pairing.rows, n_right) > 0 {
        throw ValueError(f"the left keys repeat, so this merge is not {rule.name()}")
    }
    if rule.right_unique() and duplicate_positions(pairing.left, pairing.rows, n_left) > 0 {
        throw ValueError(f"the right keys repeat, so this merge is not {rule.name()}")
    }
}

#: Rows of `left` paired with rows of `right` on their keys.
#:
#: pandas' `merge`, with pandas' defaults. `on` names key columns present on
#: both sides and puts one merged key column in the result; `left_on` and
#: `right_on` name them separately and keep both; `left_index` and
#: `right_index` use the row labels instead. Naming none of them falls back to
#: the columns the two frames share, which is what pandas does.
#:
#: The result carries a fresh range index unless both sides joined on their
#: index, where the merged labels are kept.
pub fn merge(
    left: DataFrame,
    right: DataFrame,
    how: How = How.Inner,
    on: any? = null,
    left_on: any? = null,
    right_on: any? = null,
    left_index: bool = false,
    right_index: bool = false,
    suffixes: tuple[str, str] = ("_x", "_y"),
    indicator: bool = false,
    indicator_name: str = "_merge",
    validate: Validate? = null,
    sort: bool = false
) -> DataFrame {
    require_device("merge two frames")
    var owned: list[Column] = []
    defer {
        for column in owned { column.free() }
    }

    var left_names = names_of(on, "on")
    var right_names = left_names
    if how == How.Cross {
        let keyed = on is not null or left_on is not null or right_on is not null
        if keyed or left_index or right_index {
            throw ValueError("a cross join takes no keys")
        }
        left_names = null
        right_names = null
    } elif on is not null {
        if left_on is not null or right_on is not null or left_index or right_index {
            throw ValueError("on cannot be combined with left_on, right_on or an index key")
        }
    } else {
        left_names = if left_index { null } else { names_of(left_on, "left_on") }
        right_names = if right_index { null } else { names_of(right_on, "right_on") }
        if left_names is null and not left_index and right_names is null and not right_index {
            var shared: list[str] = []
            for name in left.names {
                if right.has(name) { shared.push(name) }
            }
            if shared.len() == 0 {
                throw ValueError("the two frames share no column to merge on")
            }
            left_names = shared
            right_names = shared
        }
        if left_names is null and not left_index {
            throw ValueError("left_on or left_index has to say what the left side keys on")
        }
        if right_names is null and not right_index {
            throw ValueError("right_on or right_index has to say what the right side keys on")
        }
    }

    var left_keys: list[Column] = []
    var right_keys: list[Column] = []
    if how != How.Cross {
        if left_index {
            for level in left.index.materialize() {
                owned.push(level)
                left_keys.push(level)
            }
        } else {
            for name in left_names ?? [] { left_keys.push(left.column(name)) }
        }
        if right_index {
            for level in right.index.materialize() {
                owned.push(level)
                right_keys.push(level)
            }
        } else {
            for name in right_names ?? [] { right_keys.push(right.column(name)) }
        }
        if left_keys.len() != right_keys.len() {
            throw ValueError(
                f"{left_keys.len()} left keys cannot pair with {right_keys.len()} right ones"
            )
        }
        var unified_left: list[Column] = []
        var unified_right: list[Column] = []
        for (at, key) in left_keys.enumerate() {
            let pair = unified_pair(key, right_keys[at], owned)
            unified_left.push(pair[0])
            unified_right.push(pair[1])
        }
        left_keys = unified_left
        right_keys = unified_right
    }

    let pairing = pair_rows(left_keys, right_keys, left.len(), right.len(), how)
    defer { pairing.free() }
    let rule = validate
    if rule is not null { check_validate(pairing, left.len(), right.len(), rule) }

    var merged_keys: list[Column] = []
    for (at, key) in left_keys.enumerate() {
        merged_keys.push(merged_key_column(key, right_keys[at], pairing))
    }
    if sort { merged_keys = sort_pairing(pairing, merged_keys) }

    #: A key named the same on both sides collapses to one column, as pandas
    #: does; keys named differently stay two columns and are carried by the
    #: ordinary column loop below.
    var merged_at: dict[str, int] = {}
    let named_left = left_names
    let named_right = right_names
    if named_left is not null and named_right is not null {
        for (at, name) in named_left.enumerate() {
            if name == named_right[at] { merged_at[name] = at }
        }
    }

    var clashing: dict[str, int] = {}
    for name in right.names {
        if left.has(name) and merged_at.get(name, -1) < 0 { clashing[name] = 1 }
    }
    if clashing.len() > 0 and suffixes[0].len() == 0 and suffixes[1].len() == 0 {
        throw ValueError("the two frames share a column name and neither suffix would rename it")
    }

    let rows = pairing.rows
    let left_missing = how.keeps_right()
    let right_missing = how.keeps_left()
    var out_names: list[str] = []
    var out_columns: list[Column] = []
    for (at, name) in left.names.enumerate() {
        let merged = merged_at.get(name, -1)
        if merged >= 0 {
            out_names.push(name)
            out_columns.push(merged_keys[merged].copy())
            continue
        }
        out_names.push(if clashing.get(name, 0) > 0 { f"{name}{suffixes[0]}" } else { name })
        out_columns.push(left.columns[at].take(pairing.left, rows, allow_missing: left_missing))
    }
    for (at, name) in right.names.enumerate() {
        if merged_at.get(name, -1) >= 0 { continue }
        out_names.push(if clashing.get(name, 0) > 0 { f"{name}{suffixes[1]}" } else { name })
        out_columns.push(right.columns[at].take(pairing.right, rows, allow_missing: right_missing))
    }
    if indicator {
        out_names.push(indicator_name)
        out_columns.push(indicator_column(pairing))
    }

    #: pandas passes an index on when one was used as a key: both sides keyed
    #: on their index gives the merged labels, and one side keyed on its index
    #: gives the other side's labels, which are the ones the pairing left
    #: intact. Two frames keyed on columns get a fresh range index, because
    #: neither set of labels survived.
    var index = Index.range(rows)
    if left_index and right_index and merged_keys.len() > 0 {
        index = Index.of(merged_keys.map(|level| level.copy()), [name for name in left.index.names])
    } elif left_index {
        index = right.index.take(pairing.right, rows, allow_missing: right_missing)
    } elif right_index {
        index = left.index.take(pairing.left, rows, allow_missing: left_missing)
    }
    for key in merged_keys { key.free() }
    return DataFrame(out_names, out_columns, index: index)
}

#: `left` joined to `right` on the row labels, pandas' `DataFrame.join`.
#:
#: With no `on` both sides key on their index; with `on` the named left columns
#: key against the right's index. Either way the left frame's labels are the
#: ones the result carries, which is the whole point of an index join.
pub fn join(
    left: DataFrame,
    right: DataFrame,
    on: any? = null,
    how: How = How.Left,
    lsuffix: str = "",
    rsuffix: str = ""
) -> DataFrame {
    return merge(
        left,
        right,
        how: how,
        left_on: on,
        left_index: on is null,
        right_index: true,
        suffixes: (lsuffix, rsuffix)
    )
}

#: Every missing row filled from the last present row before it.
#:
#: The scan over the validity lane already counts the present rows before each
#: row, so compacting their positions and reading that count back out of the
#: compacted lane is a forward fill without a serial walk and without a scan
#: kernel of its own.
pub fn fill_forward(column: Column) -> Column {
    let mask = column.valid
    if mask is null or column.length == 0 { return column.copy() }
    let scanned = prefix_sum(mask, column.length)
    let compact = Buffer(scanned[1])
    kernel_for("jn_compact_positions").dispatch_async(
        [mask, scanned[0], compact],
        column.length,
        [column.length]
    )
    let lane = Buffer(column.length)
    kernel_for("jn_fill_positions").dispatch_async(
        [mask, scanned[0], compact, lane],
        column.length,
        [column.length]
    )
    scanned[0].free()
    compact.free()
    let out = column.take(lane, column.length, allow_missing: true)
    lane.free()
    return out
}
