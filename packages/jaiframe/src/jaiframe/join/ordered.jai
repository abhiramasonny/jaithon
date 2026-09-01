#: The two merges that bisect a sorted key lane instead of hashing it.
#:
#: `merge_asof` pairs each left row with the nearest right row in a direction,
#: and `merge_ordered` keeps both sides' keys in order and fills forward. A
#: bucket answers "which rows hold this key", which is not what either is
#: asking, so both sort the key lane and search it.
from std.gpu import Buffer
from ..column import Column, DataFrame, Dtype, Index, MAX_ROWS, Value
from ..column import concat_columns, f32_bits, iota, prefix_sum, wide_pack
from ..keys import factorize
from .kernels import flag, kernel_for, require_device
from .merge import fill_forward
from .pairing import Pairing, bit_width, key_order, merged_key_column, names_of
from .pairing import radix_order, sort_pairing
from .types import AsofDirection, How

#: The direction as `jn_asof_search` numbers them, which is not something a
#: caller should have to know about.
fn direction_code(direction: AsofDirection) -> int {
    return match direction {
        AsofDirection.Backward => 0,
        AsofDirection.Forward => 1,
        AsofDirection.Nearest => 2,
    }
}

#: Whether a key lane never steps backwards and never goes missing.
#:
#: One dispatch and one scan, which is what makes it cheap enough to check on
#: every ordered merge rather than take the caller's word for it.
fn is_ascending(column: Column) -> bool {
    if column.length <= 1 { return true }
    let flags = Buffer(column.length)
    kernel_for("jn_key_breaks").dispatch_async(
        [column.values, column.wide ?? column.values, column.valid ?? column.values, flags],
        column.length,
        [column.length, flag(column.dtype.is_wide()), flag(column.valid is not null)]
    )
    let scanned = prefix_sum(flags, column.length)
    flags.free()
    scanned[0].free()
    return scanned[1] == 0
}

#: Refuse a key the two ordered merges cannot bisect.
fn require_searchable(column: Column, side: str, name: str) -> void {
    if column.dtype.is_dictionary() {
        throw TypeError(
            f"an ordered merge bisects the key lane, and {name} is {column.dtype.name()}, whose codes are in first-appearance order rather than the text's — merge(sort: true) is the call that joins text keys"
        )
    }
    if column.null_count() > 0 {
        throw ValueError(
            f"the {side} key {name} has missing rows, which have no place in a key order"
        )
    }
}

#: A side's key in ascending order, with the permutation that put it there.
#:
#: Already ascending is the case these merges are written for, and it costs the
#: monotonicity scan and an `iota`. Anything else pays one radix sort, which is
#: still less than refusing and making the caller reorder the whole frame.
fn ascending_side(column: Column) -> (Column, Buffer) {
    if is_ascending(column) { return (column.copy(), iota(column.length)) }
    let perm = key_order([column], column.length)
    return (column.take(perm, column.length), perm)
}

#: A tolerance as the scalars `jn_asof_search` compares in: the two halves of a
#: wide key's limit, or the float bits of everything else's.
#:
#: A wide limit stays integral all the way to the device because a `Datetime`
#: gap is milliseconds and passing it as a float would stop being exact after
#: four and a half hours.
fn tolerance_scalars(column: Column, limit: any) -> (int, int, int) {
    if column.dtype.is_wide() {
        let span = whole_units(limit, column)
        if span < 0 { throw ValueError("a tolerance cannot be negative") }
        let halves = wide_pack(span)
        return (int(halves[0]), int(halves[1]), 0)
    }
    let span = real_units(limit)
    if span < 0.0 { throw ValueError("a tolerance cannot be negative") }
    return (0, 0, f32_bits(span))
}

fn whole_units(limit: any, column: Column) -> int {
    if limit is int { return limit }
    if limit is Value { return limit.to_int() }
    throw TypeError(
        f"a {column.dtype.name()} tolerance is counted in whole units of the column, so it has to be an int"
    )
}

fn real_units(limit: any) -> float {
    if limit is float { return limit }
    if limit is int { return float(limit) }
    if limit is Value { return limit.to_float() }
    throw TypeError("a tolerance has to be a number")
}

#: The spans an asof search with no `by` uses: the whole right side for every
#: left row, and the identity permutation that leaves it where it is.
fn whole_side_spans(n_left: int, n_right: int) -> (Buffer, Buffer, Buffer) {
    let begin = Buffer(n_left)
    let end = Buffer(n_left)
    begin.fill(0.0)
    end.fill(float(n_right))
    return (iota(n_right), begin, end)
}

#: Where each left row may search: the stretch of the group-ordered right side
#: carrying its `by` group, and the ordering that made those stretches.
#:
#: The two sides are factorized together, over one stacked column per `by`
#: level, so a code means the same thing on both — which two independent hashes
#: would only manage probabilistically, and a wrong group is a wrong number
#: rather than a slow one. Dense codes then index the bounds table directly, so
#: there is no probe to write, and the sort that gathers the right side into
#: group runs is `bit_width(groups)` radix passes rather than the sixty-four a
#: pair of hash lanes would have cost.
#:
#: That sort is stable and the right side is already in key order, so each
#: group's rows stay in key order inside its run and the bisection is searching
#: sorted data.
fn group_spans(left: DataFrame, right: DataFrame, names: list[str]) -> (Buffer, Buffer, Buffer) {
    let n_left = left.len()
    let n_right = right.len()
    var stacked: list[Column] = []
    for name in names {
        let mine = left.column(name)
        let theirs = right.column(name)
        if mine.dtype != theirs.dtype {
            throw TypeError(
                f"by column {name} is {mine.dtype.name()} on the left and {theirs.dtype.name()} on the right"
            )
        }
        stacked.push(concat_columns([mine, theirs]))
    }
    let grouped = factorize(stacked)
    for column in stacked { column.free() }
    for column in grouped[1] { column.free() }
    let codes = grouped[0]
    let groups = grouped[2]

    let left_codes = Buffer(n_left)
    kernel_for("jn_code_lane").dispatch_async([codes, left_codes], n_left, [n_left, 0])
    let right_codes = Buffer(n_right)
    kernel_for("jn_code_lane").dispatch_async([codes, right_codes], n_right, [n_right, n_left])
    codes.free()

    let base = iota(n_right)
    let width = if groups > 1 { bit_width(groups - 1) } else { 0 }
    let sorted = radix_order(base, right_codes, n_right, width)
    base.free()
    right_codes.free()

    let starts = Buffer(groups)
    let ends = Buffer(groups)
    starts.fill_zero()
    ends.fill_zero()
    kernel_for("jn_bucket_bounds").dispatch_async([sorted[1], starts, ends], n_right, [n_right])
    sorted[1].free()

    let begin = Buffer(n_left)
    let end = Buffer(n_left)
    kernel_for("jn_group_span").dispatch_async(
        [left_codes, starts, ends, begin, end],
        n_left,
        [n_left]
    )
    left_codes.free()
    starts.free()
    ends.free()
    return (sorted[0], begin, end)
}

#: Every left row paired with the nearest right row, pandas' `merge_asof`.
#:
#: An asof merge is a left join on an inequality, which no hash table can
#: answer: two keys being close is not two keys being equal, and a bucket says
#: nothing at all about a neighbour. Both sides being sorted on the key — what
#: "asof" means, and what pandas requires — makes it one bisection per left row
#: over the right key lane instead, `log2(n)` reads where a nested loop would
#: have walked the whole side.
#:
#: `by` restricts each search to the rows sharing a group. `tolerance` drops a
#: match further than that from the left key, counted in whole units of a wide
#: column — milliseconds for a `Datetime` — and inclusive at the limit, as
#: pandas' is.
#:
#: The result carries the left frame's rows and labels unchanged, because every
#: left row appears exactly once whether it matched or not.
pub fn merge_asof(
    left: DataFrame,
    right: DataFrame,
    on: str,
    by: any? = null,
    direction: AsofDirection = AsofDirection.Backward,
    tolerance: any? = null,
    suffixes: tuple[str, str] = ("_x", "_y")
) -> DataFrame {
    require_device("merge two frames as of a key")
    let left_key = left.column(on)
    let right_key = right.column(on)
    if left_key.dtype != right_key.dtype {
        throw TypeError(
            f"{on} is {left_key.dtype.name()} on the left and {right_key.dtype.name()} on the right"
        )
    }
    require_searchable(left_key, "left", on)
    require_searchable(right_key, "right", on)
    if not is_ascending(left_key) {
        throw ValueError(f"merge_asof needs the left frame sorted on {on}")
    }
    if not is_ascending(right_key) {
        throw ValueError(f"merge_asof needs the right frame sorted on {on}")
    }

    let n_left = left.len()
    let n_right = right.len()
    let by_names = names_of(by, "by") ?? []
    let spans = if by_names.len() == 0 {
        whole_side_spans(n_left, n_right)
    } else {
        group_spans(left, right, by_names)
    }
    let perm = spans[0]
    let begin = spans[1]
    let end = spans[2]
    defer {
        perm.free()
        begin.free()
        end.free()
    }

    let sorted_key = right_key.take(perm, n_right)
    defer { sorted_key.free() }
    var tol_lo = 0
    var tol_hi = 0
    var tol_bits = 0
    let limit = tolerance
    if limit is not null {
        let packed = tolerance_scalars(left_key, limit)
        tol_lo = packed[0]
        tol_hi = packed[1]
        tol_bits = packed[2]
    }
    let found = Buffer(n_left)
    kernel_for("jn_asof_search").dispatch_async(
        [
            left_key.values,
            left_key.wide ?? left_key.values,
            sorted_key.values,
            sorted_key.wide ?? sorted_key.values,
            begin,
            end,
            found,
        ],
        n_left,
        [
            n_left,
            flag(left_key.dtype.is_wide()),
            direction_code(direction),
            flag(limit is not null),
            tol_lo,
            tol_hi,
            tol_bits,
        ]
    )
    let right_lane = Buffer(n_left)
    kernel_for("jn_map_positions").dispatch_async([found, perm, right_lane], n_left, [n_left])
    found.free()
    defer { right_lane.free() }

    #: The right side's copy of the key and of every `by` column is dropped:
    #: they agree with the left's by construction, and pandas drops them too.
    var skipped: dict[str, int] = {}
    skipped[on] = 1
    for name in by_names { skipped[name] = 1 }
    var clashing: dict[str, int] = {}
    for name in right.names {
        if skipped.get(name, 0) > 0 { continue }
        if left.has(name) { clashing[name] = 1 }
    }
    if clashing.len() > 0 and suffixes[0].len() == 0 and suffixes[1].len() == 0 {
        throw ValueError("the two frames share a column name and neither suffix would rename it")
    }

    var out_names: list[str] = []
    var out_columns: list[Column] = []
    for (at, name) in left.names.enumerate() {
        out_names.push(if clashing.get(name, 0) > 0 { f"{name}{suffixes[0]}" } else { name })
        out_columns.push(left.columns[at].copy())
    }
    for (at, name) in right.names.enumerate() {
        if skipped.get(name, 0) > 0 { continue }
        out_names.push(if clashing.get(name, 0) > 0 { f"{name}{suffixes[1]}" } else { name })
        out_columns.push(right.columns[at].take(right_lane, n_left, allow_missing: true))
    }
    return DataFrame(out_names, out_columns, index: left.index.copy())
}

#: The one column name an ordered merge keys on, from a pandas-shaped argument.
fn single_key(spec: any?, what: str) -> str? {
    let names = names_of(spec, what)
    if names is null { return null }
    if names.len() != 1 {
        throw ValueError(
            f"an ordered merge bisects one key lane, so {what} names one column, not {names.len()}"
        )
    }
    return names[0]
}

#: The key column each side of an ordered merge uses.
fn ordered_keys(
    left: DataFrame,
    right: DataFrame,
    on: any?,
    left_on: any?,
    right_on: any?
) -> (str, str) {
    let both = single_key(on, "on")
    if both is not null {
        if left_on is not null or right_on is not null {
            throw ValueError("on cannot be combined with left_on or right_on")
        }
        return (both, both)
    }
    let mine = single_key(left_on, "left_on")
    let theirs = single_key(right_on, "right_on")
    if mine is not null and theirs is not null { return (mine, theirs) }
    if mine is not null or theirs is not null {
        throw ValueError("left_on and right_on name a key each, or neither of them does")
    }
    var shared: list[str] = []
    for name in left.names {
        if right.has(name) { shared.push(name) }
    }
    if shared.len() != 1 {
        throw ValueError(
            f"an ordered merge keys on one column and the two frames share {shared.len()}"
        )
    }
    return (shared[0], shared[0])
}

#: A taken column as the fill method asks for it. Consumes its argument.
fn carried(column: Column, fill: bool) -> Column {
    if not fill { return column }
    let filled = fill_forward(column)
    column.free()
    return filled
}

#: The rows of two ordered frames interleaved on their key, pandas'
#: `merge_ordered`.
#:
#: The same shape as an outer `merge` with `sort: true`, but reached by
#: bisection rather than by a hash: both sides are put in key order once and
#: each left row's matches are the half-open stretch of the right side holding
#: its key, which two binary searches find. That is `log2(n)` reads a row
#: against a bucket read, and it is worth taking because the sides are usually
#: already sorted — an ordered merge is a time series call — so the ordering
#: costs a scan that finds nothing to do.
#:
#: `fill_method: "ffill"` carries the last present value down every result
#: column, which is what makes an outer ordered merge of two irregular series
#: into one regular one.
pub fn merge_ordered(
    left: DataFrame,
    right: DataFrame,
    on: any? = null,
    left_on: any? = null,
    right_on: any? = null,
    fill_method: str? = null,
    suffixes: tuple[str, str] = ("_x", "_y"),
    how: How = How.Outer
) -> DataFrame {
    require_device("merge two frames in key order")
    if how == How.Cross {
        throw ValueError("an ordered merge pairs rows on a key, so it cannot be a cross join")
    }
    let method = fill_method
    if method is not null and method != "ffill" and method != "pad" {
        throw ValueError(f"an ordered merge fills forward, not {method}")
    }
    let fill = method is not null
    let keyed = ordered_keys(left, right, on, left_on, right_on)
    let left_name = keyed[0]
    let right_name = keyed[1]
    let left_key = left.column(left_name)
    let right_key = right.column(right_name)
    if left_key.dtype != right_key.dtype {
        throw TypeError(
            f"{left_name} is {left_key.dtype.name()} and {right_name} is {right_key.dtype.name()}"
        )
    }
    require_searchable(left_key, "left", left_name)
    require_searchable(right_key, "right", right_name)

    let n_left = left.len()
    let n_right = right.len()
    let left_side = ascending_side(left_key)
    let right_side = ascending_side(right_key)
    defer {
        left_side[0].free()
        left_side[1].free()
        right_side[0].free()
        right_side[1].free()
    }
    let sorted_left = left_side[0]
    let sorted_right = right_side[0]
    let wide = flag(left_key.dtype.is_wide())

    let lower = Buffer(n_left)
    let upper = Buffer(n_left)
    kernel_for("jn_equal_range").dispatch_async(
        [
            sorted_left.values,
            sorted_left.wide ?? sorted_left.values,
            sorted_right.values,
            sorted_right.wide ?? sorted_right.values,
            lower,
            upper,
        ],
        n_left,
        [n_left, n_right, wide]
    )
    let sizes = Buffer(n_left)
    kernel_for("jn_span_sizes").dispatch_async(
        [lower, upper, sizes],
        n_left,
        [n_left, flag(how.keeps_left())]
    )
    let emitted = prefix_sum(sizes, n_left)
    sizes.free()
    let block = emitted[1]

    #: A right row is unmatched exactly when its own stretch of the left side
    #: is empty, which is the same two searches run the other way round.
    var extra = 0
    var absent: Buffer? = null
    var absent_at: Buffer? = null
    if how.keeps_right() and n_right > 0 {
        let back_lower = Buffer(n_right)
        let back_upper = Buffer(n_right)
        kernel_for("jn_equal_range").dispatch_async(
            [
                sorted_right.values,
                sorted_right.wide ?? sorted_right.values,
                sorted_left.values,
                sorted_left.wide ?? sorted_left.values,
                back_lower,
                back_upper,
            ],
            n_right,
            [n_right, n_left, wide]
        )
        let spans = Buffer(n_right)
        kernel_for("jn_span_sizes").dispatch_async(
            [back_lower, back_upper, spans],
            n_right,
            [n_right, 0]
        )
        back_lower.free()
        back_upper.free()
        let holes = Buffer(n_right)
        kernel_for("jn_not_flags").dispatch_async([spans, holes], n_right, [n_right])
        spans.free()
        let scanned = prefix_sum(holes, n_right)
        absent = holes
        absent_at = scanned[0]
        extra = scanned[1]
    }

    let rows = block + extra
    if rows > MAX_ROWS {
        throw ValueError(
            f"this merge produces {rows} rows, past the {MAX_ROWS} a position lane addresses"
        )
    }
    let left_sorted_lane = Buffer(rows)
    let right_sorted_lane = Buffer(rows)
    kernel_for("jn_span_pairs").dispatch_async(
        [lower, upper, emitted[0], left_sorted_lane, right_sorted_lane],
        n_left,
        [n_left, flag(how.keeps_left())]
    )
    lower.free()
    upper.free()
    emitted[0].free()
    let holes = absent
    let holes_at = absent_at
    if holes is not null and holes_at is not null {
        kernel_for("jn_append_unmatched").dispatch_async(
            [holes, holes_at, left_sorted_lane, right_sorted_lane],
            n_right,
            [n_right, block]
        )
        holes.free()
        holes_at.free()
    }

    #: Both lanes hold positions into the sorted sides, so both are put back
    #: into frame coordinates by the same pass — which is why the pair kernels
    #: above do not take a mapping of their own.
    let left_lane = Buffer(rows)
    kernel_for("jn_map_positions").dispatch_async(
        [left_sorted_lane, left_side[1], left_lane],
        rows,
        [rows]
    )
    let right_lane = Buffer(rows)
    kernel_for("jn_map_positions").dispatch_async(
        [right_sorted_lane, right_side[1], right_lane],
        rows,
        [rows]
    )
    left_sorted_lane.free()
    right_sorted_lane.free()
    let pairing = Pairing(left_lane, right_lane, rows)
    defer { pairing.free() }

    let ordered = sort_pairing(pairing, [merged_key_column(left_key, right_key, pairing)])
    let merged_key = ordered[0]
    defer { merged_key.free() }

    let shared_name = left_name == right_name
    var clashing: dict[str, int] = {}
    for name in right.names {
        if shared_name and name == right_name { continue }
        if left.has(name) { clashing[name] = 1 }
    }
    if clashing.len() > 0 and suffixes[0].len() == 0 and suffixes[1].len() == 0 {
        throw ValueError("the two frames share a column name and neither suffix would rename it")
    }

    var out_names: list[str] = []
    var out_columns: list[Column] = []
    for (at, name) in left.names.enumerate() {
        if shared_name and name == left_name {
            out_names.push(name)
            out_columns.push(merged_key.copy())
            continue
        }
        out_names.push(if clashing.get(name, 0) > 0 { f"{name}{suffixes[0]}" } else { name })
        out_columns.push(
            carried(left.columns[at].take(pairing.left, rows, allow_missing: true), fill)
        )
    }
    for (at, name) in right.names.enumerate() {
        if shared_name and name == right_name { continue }
        out_names.push(if clashing.get(name, 0) > 0 { f"{name}{suffixes[1]}" } else { name })
        out_columns.push(
            carried(right.columns[at].take(pairing.right, rows, allow_missing: true), fill)
        )
    }
    return DataFrame(out_names, out_columns, index: Index.range(rows))
}
