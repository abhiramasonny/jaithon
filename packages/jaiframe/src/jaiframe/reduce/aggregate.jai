#: The named reductions, over a column, a series or a frame.
#:
#: One `Agg` and one summary become one cell here: what an aggregation means
#: over a dtype, what dtype its answer is stored as, and the two axes a frame
#: can be reduced along. The label-valued reductions — `idxmin`, `idxmax` and
#: the two edges — are the same passes read for where rather than what.
import std.math
from std.math import NAN, is_nan
from ..column import Agg, Column, DataFrame, Dtype, Index, Series, Value
from ..column import NA
from ..column import cached_kernel, f32_bits, positions_buffer
from .kernels import AGG_ALL, AGG_ANY, AGG_ARGMAX, AGG_ARGMIN, AGG_COUNT, AGG_FIRST
from .kernels import AGG_LAST, AGG_MAX, AGG_MEAN, AGG_MEDIAN, AGG_MIN, AGG_PROD
from .kernels import AGG_QUANTILE, AGG_SIZE, AGG_STD, AGG_SUM, AGG_VAR
from .kernels import flag, kernel_for
from .stats import ColumnStats, batched_stats, borrow_lanes, column_quantile, column_stats
from .stats import extreme_row, pack_frame, release_lanes, select_order_stats, zeros

#: Whether an aggregation means anything over a dtype, with the reason when it
#: does not.
#:
#: The rules are pandas': a count counts anything, an extreme or an edge picks
#: one of the inputs and so works over text, and arithmetic needs numbers. A
#: timestamp is the awkward one — its middle is a timestamp and its spread is a
#: duration, and durations are the timeseries module's business, not this one's.
fn check_agg(dtype: Dtype, agg: Agg) -> void {
    if agg == Agg.Nunique {
        throw ValueError("nunique counts distinct values through a hash table, which is keys.jai")
    }
    if agg == Agg.Count or agg == Agg.Size or agg == Agg.First or agg == Agg.Last { return }
    if dtype.is_dictionary() {
        if agg == Agg.Min or agg == Agg.Max {
            throw TypeError(f"a {dtype.name()} extreme orders the level table, which keys.jai owns")
        }
        throw TypeError(f"{agg.name()} is not defined over {dtype.name()}")
    }
    if dtype == Dtype.Datetime {
        let extreme = agg == Agg.Min or agg == Agg.Max
        let middle = agg == Agg.Mean or agg == Agg.Median or agg == Agg.Quantile
        if extreme or middle { return }
        throw TypeError(f"{agg.name()} of a timestamp is a duration, which datetime64[ms] is not")
    }
}

fn agg_accepts(dtype: Dtype, agg: Agg) -> bool {
    if agg == Agg.Nunique { return false }
    if agg == Agg.Count or agg == Agg.Size or agg == Agg.First or agg == Agg.Last { return true }
    if dtype.is_dictionary() { return false }
    if dtype == Dtype.Datetime {
        let extreme = agg == Agg.Min or agg == Agg.Max
        let middle = agg == Agg.Mean or agg == Agg.Median or agg == Agg.Quantile
        return extreme or middle
    }
    return true
}

#: What dtype a reduction's answer is stored as.
#:
#: Counts are exact integers and say so. An extreme or an edge is one of the
#: inputs and keeps their dtype. Everything else is a `Float64`, even over an
#: `Int64` column where pandas would hand back an int64 — the sums are
#: accumulated in float32 lanes and past 2^24 they are not that integer, and a
#: dtype that admits it is better than one that is quietly wrong.
fn result_dtype(dtype: Dtype, agg: Agg) -> Dtype {
    if agg.counts_rows() { return Dtype.Int64 }
    if agg == Agg.Any or agg == Agg.All { return Dtype.Bool }
    if agg.picks_a_row() { return dtype }
    if dtype == Dtype.Datetime { return Dtype.Datetime }
    return Dtype.Float64
}

fn number_cell(dtype: Dtype, value: float) -> Value {
    if is_nan(value) { return Value.null_of(dtype) }
    if dtype == Dtype.Datetime { return Value.of_datetime(int(math.round(value))) }
    if dtype == Dtype.Int64 { return Value.of_int(int(math.round(value))) }
    if dtype == Dtype.Bool { return Value.of_bool(value != 0.0) }
    return Value.of_float(value)
}

#: The cell at a row, or NA when the reduction found no row at all.
#:
#: Goes back to the column rather than using the float the reduction carried,
#: because for an `Int64` or a `Datetime` column that float came off the
#: flattened lane and is only exact to 2^24. One extra short download, and only
#: for the aggregations that pick a row.
fn cell_at(column: Column, row: int) -> Value {
    if row < 0 { return Value.null_of(column.dtype) }
    return column.get(row)
}

fn empty_result(dtype: Dtype, agg: Agg) -> Value {
    if agg == Agg.Count or agg == Agg.Size { return Value.of_int(0) }
    if agg == Agg.Sum { return Value.of_float(0.0) }
    if agg == Agg.Prod { return Value.of_float(1.0) }
    if agg == Agg.Any { return Value.of_bool(false) }
    if agg == Agg.All { return Value.of_bool(true) }
    return Value.null_of(result_dtype(dtype, agg))
}

#: A reduction finished off a summary that has already been computed.
#:
#: Split out from `reduce_column` so a frame can summarise every column in one
#: round trip and then finish them one at a time without going back.
fn finish_reduction(
    column: Column,
    stats: ColumnStats,
    agg: Agg,
    skipna: bool,
    ddof: int,
    q: float,
    quantile_value: float
) -> Value {
    if agg == Agg.Count { return Value.of_int(stats.present) }
    if agg == Agg.Size { return Value.of_int(stats.length) }
    let target = result_dtype(column.dtype, agg)
    if not skipna and stats.missing() > 0 { return Value.null_of(target) }
    if agg == Agg.Min { return cell_at(column, extreme_row(column, stats, false)) }
    if agg == Agg.Max { return cell_at(column, extreme_row(column, stats, true)) }
    if agg == Agg.First { return cell_at(column, stats.first_present) }
    if agg == Agg.Last { return cell_at(column, stats.last_present) }
    if agg == Agg.Any { return Value.of_bool(stats.truthy > 0) }
    if agg == Agg.All { return Value.of_bool(stats.truthy == stats.present) }
    if agg == Agg.Sum { return Value.of_float(stats.total) }
    if agg == Agg.Prod { return Value.of_float(stats.product) }
    if agg == Agg.Median or agg == Agg.Quantile {
        return number_cell(target, quantile_value)
    }
    if stats.present == 0 { return Value.null_of(target) }
    if agg == Agg.Mean { return number_cell(target, stats.mean()) }
    let spread = stats.variance(ddof)
    if agg == Agg.Var { return number_cell(target, spread) }
    if is_nan(spread) { return Value.null_of(target) }
    return number_cell(target, math.sqrt(spread))
}

#: The `agg` of one column, as one cell.
#:
#: This is the call groupby, resample and pivot_table name too, so that
#: `Agg.Mean` means the same arithmetic wherever it is written. `ddof` is spent
#: by `Std` and `Var`; `q` is read only by `Quantile`.
pub fn reduce_column(
    column: Column,
    agg: Agg,
    skipna: bool = true,
    ddof: int = 1,
    q: float = 0.5
) -> Value {
    check_agg(column.dtype, agg)
    if agg == Agg.Size { return Value.of_int(column.len()) }
    if column.len() == 0 { return empty_result(column.dtype, agg) }
    let stats = column_stats(column)
    var picked = NAN
    if agg == Agg.Median or agg == Agg.Quantile {
        picked = column_quantile(column, stats.present, if agg == Agg.Median { 0.5 } else { q })
    }
    return finish_reduction(column, stats, agg, skipna, ddof, q, picked)
}

#: A column of cells the host already holds, in the dtype they share.
pub fn cells_column(cells: list[Value], dtype: Dtype) -> Column {
    if dtype.is_dictionary() {
        var labels: list[str?] = []
        for cell in cells { labels.push(if cell.is_na() { null } else { cell.to_str() }) }
        return Column.from_strs(labels, dtype)
    }
    var gaps: list[bool] = []
    for cell in cells { gaps.push(cell.is_na()) }
    if dtype.is_wide() {
        var numbers: list[int] = []
        for cell in cells { numbers.push(if cell.is_na() { 0 } else { cell.to_int() }) }
        return Column.from_ints(numbers, gaps, dtype)
    }
    if dtype == Dtype.Bool {
        var truths: list[bool] = []
        for cell in cells { truths.push(not cell.is_na() and cell.to_bool()) }
        return Column.from_bools(truths, gaps)
    }
    var numbers: list[float] = []
    for cell in cells { numbers.push(cell.to_float()) }
    return Column.from_floats(numbers)
}

#: The one dtype a per-column result can be stored in.
#:
#: `Float64` absorbs a mix of numbers. Text does not mix with anything, because
#: the alternative is pandas' object column and everything downstream then has
#: to ask what is really in it.
fn common_dtype(dtypes: list[Dtype], agg: Agg) -> Dtype {
    if dtypes.len() == 0 { return Dtype.Float64 }
    var chosen = result_dtype(dtypes[0], agg)
    for dtype in dtypes {
        let here = result_dtype(dtype, agg)
        if here == chosen { continue }
        if chosen.is_dictionary() or here.is_dictionary() {
            throw TypeError(
                f"{agg.name()} over both {chosen.name()} and {here.name()} has no one dtype"
            )
        }
        chosen = Dtype.Float64
    }
    return chosen
}

#: A series whose labels are column names, which is what a per-column
#: reduction of a frame is indexed by.
pub fn named_series(names: list[str], values: Column) -> Series {
    var labels: list[str?] = []
    for name in names { labels.push(name) }
    var titles: list[str?] = [null]
    return Series(values, index: Index.of([Column.from_strs(labels)], titles))
}

#: Which columns an aggregation is defined over, in frame order.
fn accepted_columns(frame: DataFrame, agg: Agg) -> list[int] {
    var chosen: list[int] = []
    for (at, column) in frame.columns.enumerate() {
        if agg_accepts(column.dtype, agg) { chosen.push(at) }
    }
    return chosen
}

#: Every column reduced, in one round trip for the summaries and one more for
#: the quantiles when the aggregation is one.
fn frame_axis0(frame: DataFrame, agg: Agg, skipna: bool, ddof: int, q: float) -> Series {
    if agg == Agg.Nunique {
        throw ValueError("nunique counts distinct values through a hash table, which is keys.jai")
    }
    let chosen = accepted_columns(frame, agg)
    var names: list[str] = []
    var columns: list[Column] = []
    for at in chosen {
        names.push(frame.names[at])
        columns.push(frame.columns[at])
    }
    let rows = frame.len()
    var cells: list[Value] = []
    if agg == Agg.Size {
        for _at in chosen { cells.push(Value.of_int(rows)) }
        return named_series(names, cells_column(cells, Dtype.Int64))
    }
    let borrowed = borrow_lanes(columns)
    let summaries = batched_stats(borrowed[0], borrowed[1], rows, zeros(columns.len()))
    release_lanes(borrowed[0], borrowed[2])
    var picks: list[float] = []
    if agg == Agg.Median or agg == Agg.Quantile {
        picks = frame_quantiles(columns, summaries, rows, if agg == Agg.Median { 0.5 } else { q })
    } else {
        picks = zeros(columns.len())
    }
    for (at, column) in columns.enumerate() {
        if rows == 0 {
            cells.push(empty_result(column.dtype, agg))
            continue
        }
        cells.push(finish_reduction(column, summaries[at], agg, skipna, ddof, q, picks[at]))
    }
    return named_series(names, cells_column(cells, common_dtype(columns.map(|c| c.dtype), agg)))
}

#: One quantile of every column, batched into a single radix search.
fn frame_quantiles(
    columns: list[Column],
    summaries: list[ColumnStats],
    rows: int,
    q: float
) -> list[float] {
    if columns.len() == 0 or rows == 0 { return zeros(columns.len()) }
    if q < 0.0 or q > 1.0 { throw ValueError(f"a quantile is between 0 and 1, not {q}") }
    let packed = pack_frame(columns, rows, zeros(columns.len()))
    var lanes: list[int] = []
    var ranks: list[int] = []
    var fractions: list[float] = []
    for (at, summary) in summaries.enumerate() {
        let present = max(summary.present, 1)
        let position = q * float(present - 1)
        let lower = int(math.floor(position))
        fractions.push(position - float(lower))
        lanes.push(at)
        ranks.push(lower)
        lanes.push(at)
        ranks.push(if lower + 1 < present { lower + 1 } else { lower })
    }
    let found = select_order_stats(packed[0], packed[1], rows, lanes, ranks)
    packed[0].free()
    packed[1].free()
    var out: list[float] = []
    for (at, summary) in summaries.enumerate() {
        if summary.present == 0 {
            out.push(NAN)
            continue
        }
        let low = found[at * 2]
        let high = found[at * 2 + 1]
        out.push(low + fractions[at] * (high - low))
    }
    return out
}

pub fn agg_code(agg: Agg) -> int {
    return match agg {
        Agg.Sum => AGG_SUM,
        Agg.Mean => AGG_MEAN,
        Agg.Min => AGG_MIN,
        Agg.Max => AGG_MAX,
        Agg.Count => AGG_COUNT,
        Agg.Size => AGG_SIZE,
        Agg.Prod => AGG_PROD,
        Agg.Std => AGG_STD,
        Agg.Var => AGG_VAR,
        Agg.Median => AGG_MEDIAN,
        Agg.Quantile => AGG_QUANTILE,
        Agg.First => AGG_FIRST,
        Agg.Last => AGG_LAST,
        Agg.Any => AGG_ANY,
        Agg.All => AGG_ALL,
        _ => throw ValueError(f"{agg.name()} has no across-the-row form here"),
    }
}

#: The numeric columns of a frame, which is what a per-row pass reads.
pub fn numeric_columns(frame: DataFrame) -> (list[str], list[Column]) {
    var names: list[str] = []
    var columns: list[Column] = []
    for (at, column) in frame.columns.enumerate() {
        if not column.dtype.is_numeric() { continue }
        names.push(frame.names[at])
        columns.push(column)
    }
    return (names, columns)
}

#: One value per row, reduced across the frame's numeric columns.
pub fn frame_axis1(frame: DataFrame, code: int, skipna: bool, ddof: int, q: float) -> Column {
    let picked = numeric_columns(frame)
    let columns = picked[1]
    let rows = frame.len()
    let out = Column.allocate(Dtype.Float64, rows, with_validity: true)
    if rows == 0 { return out }
    if columns.len() == 0 {
        out.values.fill(NAN)
        (out.valid ?? out.values).fill_zero()
        return out
    }
    let packed = pack_frame(columns, rows, zeros(columns.len()))
    kernel_for("red_axis").dispatch_async(
        [packed[0], packed[1], out.values, out.valid ?? out.values],
        rows,
        [rows, columns.len(), code, ddof, f32_bits(q), flag(skipna)]
    )
    packed[0].free()
    packed[1].free()
    return out
}

#: A float lane re-read as an exact `Int64` column, for the counts.
fn as_int_column(source: Column) -> Column {
    let out = Column.allocate(Dtype.Int64, source.len())
    if source.len() == 0 { return out }
    kernel_for("red_split_wide").dispatch_async(
        [source.values, out.values, out.wide ?? out.values],
        source.len(),
        [source.len()]
    )
    return out
}

fn dispatch_reduce(target: any, agg: Agg, axis: int, skipna: bool, ddof: int, q: float) -> any {
    if target is Column { return reduce_column(target, agg, skipna, ddof, q) }
    if target is Series { return reduce_column(target.values, agg, skipna, ddof, q) }
    if target is not DataFrame {
        throw TypeError("a reduction takes a Column, a Series or a DataFrame")
    }
    if axis == 0 { return frame_axis0(target, agg, skipna, ddof, q) }
    if axis != 1 { throw ValueError(f"axis {axis} is neither 0 nor 1") }
    let raw = frame_axis1(target, agg_code(agg), skipna, ddof, q)
    if not agg.counts_rows() { return Series(raw, index: target.index.copy()) }
    let counted = as_int_column(raw)
    raw.free()
    return Series(counted, index: target.index.copy())
}

#: The sum of the present values. 0 over a column that has none, as pandas is.
pub fn sum(target: any, axis: int = 0, skipna: bool = true) -> any {
    return dispatch_reduce(target, Agg.Sum, axis, skipna, 1, 0.5)
}

#: The arithmetic mean of the present values.
pub fn mean(target: any, axis: int = 0, skipna: bool = true) -> any {
    return dispatch_reduce(target, Agg.Mean, axis, skipna, 1, 0.5)
}

#: The smallest present value. Named for `min` being a builtin.
pub fn minimum(target: any, axis: int = 0, skipna: bool = true) -> any {
    return dispatch_reduce(target, Agg.Min, axis, skipna, 1, 0.5)
}

#: The largest present value. Named for `max` being a builtin.
pub fn maximum(target: any, axis: int = 0, skipna: bool = true) -> any {
    return dispatch_reduce(target, Agg.Max, axis, skipna, 1, 0.5)
}

#: The product of the present values. 1 over a column that has none.
pub fn product(target: any, axis: int = 0, skipna: bool = true) -> any {
    return dispatch_reduce(target, Agg.Prod, axis, skipna, 1, 0.5)
}

#: How many rows hold a value. Never missing, and never affected by `skipna` —
#: counting the present ones is what it is for.
pub fn count(target: any, axis: int = 0) -> any {
    return dispatch_reduce(target, Agg.Count, axis, true, 1, 0.5)
}

#: The sample standard deviation, one degree of freedom by default.
pub fn std(target: any, axis: int = 0, skipna: bool = true, ddof: int = 1) -> any {
    return dispatch_reduce(target, Agg.Std, axis, skipna, ddof, 0.5)
}

#: The sample variance. Spelled out because `var` is a keyword here.
pub fn variance(target: any, axis: int = 0, skipna: bool = true, ddof: int = 1) -> any {
    return dispatch_reduce(target, Agg.Var, axis, skipna, ddof, 0.5)
}

#: The middle present value, interpolated between the two middles when the
#: count is even.
pub fn median(target: any, axis: int = 0, skipna: bool = true) -> any {
    return dispatch_reduce(target, Agg.Median, axis, skipna, 1, 0.5)
}

#: The value `q` of the way through the present ones, interpolated linearly
#: between its neighbours the way pandas does by default.
pub fn quantile(target: any, q: float = 0.5, axis: int = 0, skipna: bool = true) -> any {
    return dispatch_reduce(target, Agg.Quantile, axis, skipna, 1, q)
}

#: Whether any present value is truthy. Named for `any` being a type.
pub fn any_(target: any, axis: int = 0, skipna: bool = true) -> any {
    return dispatch_reduce(target, Agg.Any, axis, skipna, 1, 0.5)
}

#: Whether every present value is truthy. True over none, as pandas is.
pub fn all_(target: any, axis: int = 0, skipna: bool = true) -> any {
    return dispatch_reduce(target, Agg.All, axis, skipna, 1, 0.5)
}

#: The labels at a list of rows, `MISSING_POSITION` giving a null.
#:
#: A multi-index gives its first level, which is what a single answer per
#: column can carry.
fn label_column(index: Index, rows: list[int]) -> Column {
    let levels = index.materialize()
    let lane = positions_buffer(rows)
    let out = levels[0].take(lane, rows.len(), allow_missing: true)
    lane.free()
    for level in levels { level.free() }
    return out
}

fn label_of(index: Index, row: int) -> Value {
    if row < 0 { return NA }
    return index.label_at(row)[0]
}

#: The label of the row holding the extreme, per column or per row.
fn dispatch_arg(target: any, want_high: bool, axis: int, skipna: bool) -> any {
    if target is Column {
        let stats = column_stats(target)
        if not skipna and stats.missing() > 0 { return NA }
        let row = extreme_row(target, stats, want_high)
        return if row < 0 { NA } else { Value.of_int(row) }
    }
    if target is Series {
        let stats = column_stats(target.values)
        if not skipna and stats.missing() > 0 { return NA }
        return label_of(target.index, extreme_row(target.values, stats, want_high))
    }
    if target is not DataFrame {
        throw TypeError("an extreme's label needs a Column, a Series or a DataFrame")
    }
    if axis == 1 {
        return Series(argmax_names(target, want_high, skipna), index: target.index.copy())
    }
    if axis != 0 { throw ValueError(f"axis {axis} is neither 0 nor 1") }
    let picked = numeric_columns(target)
    let columns = picked[1]
    let borrowed = borrow_lanes(columns)
    let summaries = batched_stats(borrowed[0], borrowed[1], target.len(), zeros(columns.len()))
    release_lanes(borrowed[0], borrowed[2])
    var rows: list[int] = []
    for (at, summary) in summaries.enumerate() {
        if not skipna and summary.missing() > 0 {
            rows.push(-1)
            continue
        }
        rows.push(extreme_row(columns[at], summary, want_high))
    }
    return named_series(picked[0], label_column(target.index, rows))
}

#: The name of the column holding each row's extreme, as a text column whose
#: level table is the frame's own column names.
fn argmax_names(frame: DataFrame, want_high: bool, skipna: bool) -> Column {
    let picked = numeric_columns(frame)
    let rows = frame.len()
    let code = if want_high { AGG_ARGMAX } else { AGG_ARGMIN }
    let raw = frame_axis1(frame, code, skipna, 1, 0.5)
    let out = Column.allocate(Dtype.Str, rows, with_validity: true)
    out.set_levels(picked[0])
    if rows > 0 {
        cached_kernel("col_copy_lane").dispatch_async([raw.values, out.values], rows, [rows, 0, 0])
        cached_kernel("col_copy_lane").dispatch_async(
            [raw.valid ?? raw.values, out.valid ?? out.values],
            rows,
            [rows, 0, 0]
        )
    }
    raw.free()
    return out
}

#: The label of the row holding the smallest value.
pub fn idxmin(target: any, axis: int = 0, skipna: bool = true) -> any {
    return dispatch_arg(target, false, axis, skipna)
}

#: The label of the row holding the largest value.
pub fn idxmax(target: any, axis: int = 0, skipna: bool = true) -> any {
    return dispatch_arg(target, true, axis, skipna)
}

fn frame_edge(frame: DataFrame, want_last: bool) -> int {
    let borrowed = borrow_lanes(frame.columns)
    let summaries = batched_stats(borrowed[0], borrowed[1], frame.len(), zeros(frame.width()))
    release_lanes(borrowed[0], borrowed[2])
    var edge = -1
    for summary in summaries {
        let here = if want_last { summary.last_present } else { summary.first_present }
        if here < 0 { continue }
        if edge < 0 {
            edge = here
        } elif want_last {
            edge = max(edge, here)
        } else {
            edge = min(edge, here)
        }
    }
    return edge
}

#: The label of the first row that holds a value, NA when none does.
#:
#: Over a frame this is the first row where any column holds one, which is
#: pandas' meaning and is the smallest of the per-column answers — so it comes
#: out of the same single pass that summarises the columns.
pub fn first_valid_index(target: any) -> Value {
    if target is Column {
        let row = column_stats(target).first_present
        return if row < 0 { NA } else { Value.of_int(row) }
    }
    if target is Series { return label_of(target.index, column_stats(target.values).first_present) }
    if target is not DataFrame {
        throw TypeError("an edge needs a Column, a Series or a DataFrame")
    }
    return label_of(target.index, frame_edge(target, false))
}

#: The label of the last row that holds a value, NA when none does.
pub fn last_valid_index(target: any) -> Value {
    if target is Column {
        let row = column_stats(target).last_present
        return if row < 0 { NA } else { Value.of_int(row) }
    }
    if target is Series { return label_of(target.index, column_stats(target.values).last_present) }
    if target is not DataFrame {
        throw TypeError("an edge needs a Column, a Series or a DataFrame")
    }
    return label_of(target.index, frame_edge(target, true))
}
