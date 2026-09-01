#: The statistics that read more of a summary than one number.
#:
#: The shape statistics — the standard error, the skew and the kurtosis — need
#: a second pass centred on the mean; `describe` needs the summary and a batch
#: of quantiles together; and a correlation needs the co-moments of a pair.
#: All three are arithmetic on what `stats` collected, so they live together.
import std.math
from std.gpu import Buffer
from std.math import NAN, is_nan
from ..column import Column, DataFrame, Dtype, Index, Series, Value
from .kernels import AGG_KURT, AGG_SEM, AGG_SKEW, PCHUNKS, PGROUP, STATS, kernel_for
from .aggregate import cells_column, frame_axis1, named_series, numeric_columns
from .stats import ColumnStats, batched_stats, borrow_lanes, column_stats, pack_frame
from .stats import release_lanes, select_order_stats, zeros

#: The three statistics that are arithmetic on a summary but have no `Agg`
#: spelling, because group-by does not name them.
const EXTRA_SEM = 0
const EXTRA_SKEW = 1
const EXTRA_KURT = 2

#: A shape statistic off a first pass and a pass centred on its mean.
#:
#: pandas' bias-corrected forms, which are scipy's with `bias=False`. Skew
#: needs three present rows and kurtosis four, because below that the
#: correction divides by zero — pandas answers NaN there and so does this.
fn extra_from(summary: ColumnStats, shaped: ColumnStats?, kind: int, ddof: int) -> Value {
    let rows = summary.present
    if kind == EXTRA_SEM {
        let spread = summary.variance(ddof)
        if rows == 0 or is_nan(spread) { return Value.null_of(Dtype.Float64) }
        return Value.of_float(math.sqrt(spread / float(rows)))
    }
    if shaped is null { return Value.null_of(Dtype.Float64) }
    let needed = if kind == EXTRA_SKEW { 3 } else { 4 }
    if rows < needed { return Value.null_of(Dtype.Float64) }
    let spread = shaped.sum_squares / float(rows - 1)
    if not spread > 0.0 { return Value.null_of(Dtype.Float64) }
    if kind == EXTRA_SKEW {
        let deviation = math.sqrt(spread)
        let scaled = shaped.sum_cubes / (deviation * deviation * deviation)
        return Value.of_float(float(rows) / float((rows - 1) * (rows - 2)) * scaled)
    }
    let lead = float(rows * (rows + 1)) / float((rows - 1) * (rows - 2) * (rows - 3))
    let tail = 3.0 * float((rows - 1) * (rows - 1)) / float((rows - 2) * (rows - 3))
    return Value.of_float(lead * (shaped.sum_quads / (spread * spread)) - tail)
}

fn column_extra(column: Column, kind: int, skipna: bool, ddof: int) -> Value {
    if not column.dtype.is_numeric() {
        throw TypeError(f"a shape statistic needs numbers, not {column.dtype.name()}")
    }
    if column.len() == 0 { return Value.null_of(Dtype.Float64) }
    let first = column_stats(column)
    if not skipna and first.missing() > 0 { return Value.null_of(Dtype.Float64) }
    if kind == EXTRA_SEM { return extra_from(first, null, kind, ddof) }
    return extra_from(first, column_stats(column, center: first.mean()), kind, ddof)
}

fn frame_extra(frame: DataFrame, kind: int, skipna: bool, ddof: int) -> Series {
    let picked = numeric_columns(frame)
    let columns = picked[1]
    let rows = frame.len()
    let borrowed = borrow_lanes(columns)
    let first = batched_stats(borrowed[0], borrowed[1], rows, zeros(columns.len()))
    var shaped: list[ColumnStats] = []
    if kind != EXTRA_SEM {
        shaped = batched_stats(borrowed[0], borrowed[1], rows, first.map(|one| one.mean()))
    }
    release_lanes(borrowed[0], borrowed[2])
    var cells: list[Value] = []
    for (at, summary) in first.enumerate() {
        if not skipna and summary.missing() > 0 {
            cells.push(Value.null_of(Dtype.Float64))
            continue
        }
        let second = if kind == EXTRA_SEM { null } else { shaped[at] }
        cells.push(extra_from(summary, second, kind, ddof))
    }
    return named_series(picked[0], cells_column(cells, Dtype.Float64))
}

fn dispatch_extra(target: any, kind: int, axis: int, skipna: bool, ddof: int) -> any {
    if target is Column { return column_extra(target, kind, skipna, ddof) }
    if target is Series { return column_extra(target.values, kind, skipna, ddof) }
    if target is not DataFrame {
        throw TypeError("a shape statistic takes a Column, a Series or a DataFrame")
    }
    if axis == 0 { return frame_extra(target, kind, skipna, ddof) }
    if axis != 1 { throw ValueError(f"axis {axis} is neither 0 nor 1") }
    let code = if kind == EXTRA_SEM {
        AGG_SEM
    } elif kind == EXTRA_SKEW {
        AGG_SKEW
    } else {
        AGG_KURT
    }
    return Series(frame_axis1(target, code, skipna, ddof, 0.5), index: target.index.copy())
}

#: The standard error of the mean: the deviation over the root of the count.
pub fn sem(target: any, axis: int = 0, skipna: bool = true, ddof: int = 1) -> any {
    return dispatch_extra(target, EXTRA_SEM, axis, skipna, ddof)
}

#: The bias-corrected third standardised moment. NaN under three present rows.
pub fn skew(target: any, axis: int = 0, skipna: bool = true) -> any {
    return dispatch_extra(target, EXTRA_SKEW, axis, skipna, 1)
}

#: The bias-corrected excess kurtosis, Fisher's — zero for a normal sample.
#: NaN under four present rows.
pub fn kurtosis(target: any, axis: int = 0, skipna: bool = true) -> any {
    return dispatch_extra(target, EXTRA_KURT, axis, skipna, 1)
}

fn percent_label(q: float) -> str {
    let scaled = q * 100.0
    if scaled == math.round(scaled) { return f"{int(math.round(scaled))}%" }
    return f"{scaled}%"
}

#: The count, the middle, the spread and the shape of every numeric column.
#:
#: The rows are pandas': count, mean, std, min, the percentiles, max. The
#: summaries come out of one round trip for the whole frame, and the
#: percentiles out of one batched radix search for every column and every
#: percentile together — so a frame of twenty columns described at three
#: percentiles is two round trips, not a hundred and twenty.
pub fn describe(frame: DataFrame, percentiles: list[float]? = null) -> DataFrame {
    let wanted = percentiles ?? [0.25, 0.5, 0.75]
    for q in wanted {
        if q < 0.0 or q > 1.0 { throw ValueError(f"a percentile is between 0 and 1, not {q}") }
    }
    let picked = numeric_columns(frame)
    let columns = picked[1]
    let rows = frame.len()
    let borrowed = borrow_lanes(columns)
    let summaries = batched_stats(borrowed[0], borrowed[1], rows, zeros(columns.len()))
    release_lanes(borrowed[0], borrowed[2])
    var found: list[float] = []
    var fractions: list[float] = []
    if columns.len() > 0 and rows > 0 {
        let packed = pack_frame(columns, rows, zeros(columns.len()))
        var lanes: list[int] = []
        var ranks: list[int] = []
        for (at, summary) in summaries.enumerate() {
            for q in wanted {
                let present = max(summary.present, 1)
                let position = q * float(present - 1)
                let lower = int(math.floor(position))
                fractions.push(position - float(lower))
                lanes.push(at)
                ranks.push(lower)
                lanes.push(at)
                ranks.push(if lower + 1 < present { lower + 1 } else { lower })
            }
        }
        found = select_order_stats(packed[0], packed[1], rows, lanes, ranks)
        packed[0].free()
        packed[1].free()
    }
    var stats: list[str?] = ["count", "mean", "std", "min"]
    for q in wanted { stats.push(percent_label(q)) }
    stats.push("max")
    var out: list[Column] = []
    for (at, summary) in summaries.enumerate() {
        var numbers: list[float] = []
        numbers.push(float(summary.present))
        numbers.push(summary.mean())
        let spread = summary.variance(1)
        numbers.push(if is_nan(spread) { NAN } else { math.sqrt(spread) })
        numbers.push(if summary.present == 0 { NAN } else { summary.lowest })
        for (which, _q) in wanted.enumerate() {
            let slot = (at * wanted.len() + which) * 2
            if summary.present == 0 or slot + 1 >= found.len() {
                numbers.push(NAN)
                continue
            }
            numbers.push(
                found[slot] + fractions[at * wanted.len() + which] * (found[slot + 1] - found[slot])
            )
        }
        numbers.push(if summary.present == 0 { NAN } else { summary.highest })
        out.push(Column.from_floats(numbers))
    }
    var titles: list[str?] = [null]
    return DataFrame(picked[0], out, index: Index.of([Column.from_strs(stats)], titles))
}

#: The co-moments of every named pair, in one pass over a packed matrix.
#:
#: The matrix is centred on each column's own mean first. A covariance from raw
#: sums subtracts two numbers of the same size and float32 has little left
#: underneath; shifting a column by a constant changes no covariance, so the
#: shortcut formula stays exact in algebra and becomes usable in arithmetic.
#: The centres come from the same batched summary the caller would have wanted
#: anyway.
fn pairwise_moments(columns: list[Column], rows: int, pairs: list[int]) -> list[float] {
    let lanes = pairs.len() // 2
    if lanes == 0 or rows == 0 or columns.len() == 0 { return [] }
    let borrowed = borrow_lanes(columns)
    let summaries = batched_stats(borrowed[0], borrowed[1], rows, zeros(columns.len()))
    release_lanes(borrowed[0], borrowed[2])
    var centers: list[float] = []
    for summary in summaries {
        let middle = summary.mean()
        centers.push(if is_nan(middle) { 0.0 } else { middle })
    }
    let packed = pack_frame(columns, rows, centers)
    let table = Buffer(pairs.len())
    table.upload(pairs.map(|at| float(at)))
    let partials = Buffer(lanes * PCHUNKS * STATS)
    kernel_for("red_pairwise").dispatch_async(
        [packed[0], packed[1], table, partials],
        lanes * PCHUNKS * PGROUP,
        [rows, PCHUNKS],
        group_size: PGROUP
    )
    let folded = Buffer(lanes * STATS)
    kernel_for("red_join_pairwise").dispatch_async(
        [partials, folded],
        lanes * STATS,
        [lanes, PCHUNKS]
    )
    let out = folded.download()
    packed[0].free()
    packed[1].free()
    table.free()
    partials.free()
    folded.free()
    return out
}

fn check_correlation_method(method: str) -> void {
    if method == "pearson" { return }
    if method == "spearman" {
        throw ValueError(
            "spearman is pearson over ranks; rank the frame with keys.jai and correlate that"
        )
    }
    if method == "kendall" {
        throw ValueError("kendall counts concordant pairs, which is quadratic and is not here")
    }
    throw ValueError(f"unknown correlation method {method}")
}

#: The correlation or covariance one folded pair carries.
fn pair_value(
    moments: list[float],
    slot: int,
    min_periods: int,
    ddof: int,
    correlate: bool
) -> float {
    let base = slot * STATS
    let rows = moments[base]
    if rows < float(min_periods) or rows <= 0.0 { return NAN }
    let cxy = moments[base + 5] - moments[base + 1] * moments[base + 2] / rows
    let cxx = moments[base + 3] - moments[base + 1] * moments[base + 1] / rows
    let cyy = moments[base + 4] - moments[base + 2] * moments[base + 2] / rows
    if correlate {
        if cxx <= 0.0 or cyy <= 0.0 { return NAN }
        return cxy / math.sqrt(cxx * cyy)
    }
    let degrees = rows - float(ddof)
    if degrees <= 0.0 { return NAN }
    return cxy / degrees
}

fn square_matrix(frame: DataFrame, correlate: bool, min_periods: int, ddof: int) -> DataFrame {
    let picked = numeric_columns(frame)
    let columns = picked[1]
    let width = columns.len()
    var pairs: list[int] = []
    var slots: dict[int, int] = {}
    for left in 0..width {
        for right in left..width {
            slots[left * width + right] = pairs.len() // 2
            pairs.push(left)
            pairs.push(right)
        }
    }
    let moments = pairwise_moments(columns, frame.len(), pairs)
    var out: list[Column] = []
    for right in 0..width {
        var numbers: list[float] = []
        for left in 0..width {
            let low = min(left, right)
            let high = max(left, right)
            let slot = slots.get(low * width + high, -1)
            if slot < 0 or (slot + 1) * STATS > moments.len() {
                numbers.push(NAN)
                continue
            }
            numbers.push(pair_value(moments, slot, min_periods, ddof, correlate))
        }
        out.push(Column.from_floats(numbers))
    }
    var labels: list[str?] = []
    for name in picked[0] { labels.push(name) }
    var titles: list[str?] = [null]
    return DataFrame(picked[0], out, index: Index.of([Column.from_strs(labels)], titles))
}

#: The correlation of every pair of numeric columns.
#:
#: Pairwise-complete, as pandas is: a row counts for a pair only when both of
#: its columns hold a value, so two entries of the matrix can rest on different
#: numbers of rows. Only `pearson` is here — `spearman` is `pearson` of the
#: ranks, and ranking is keys.jai's.
pub fn corr(frame: DataFrame, method: str = "pearson", min_periods: int = 1) -> DataFrame {
    check_correlation_method(method)
    return square_matrix(frame, true, min_periods, 1)
}

#: The covariance of every pair of numeric columns, `ddof` degrees spent.
pub fn cov(frame: DataFrame, min_periods: int = 1, ddof: int = 1) -> DataFrame {
    return square_matrix(frame, false, min_periods, ddof)
}

#: The correlation of each of this frame's columns with its counterpart.
#:
#: Against a `Series`, every column is correlated with that one series.
#: Against a `DataFrame`, columns are matched by name; `drop` says whether a
#: name that only one side has is left out or reported as missing.
pub fn corrwith(
    frame: DataFrame,
    other: any,
    method: str = "pearson",
    drop: bool = false,
    min_periods: int = 1
) -> Series {
    check_correlation_method(method)
    let picked = numeric_columns(frame)
    var names: list[str] = []
    var columns: list[Column] = []
    var pairs: list[int] = []
    if other is Series {
        if other.len() != frame.len() {
            throw ValueError(f"a series of {other.len()} rows cannot align with {frame.len()}")
        }
        for column in picked[1] { columns.push(column) }
        columns.push(other.values)
        for (at, name) in picked[0].enumerate() {
            names.push(name)
            pairs.push(at)
            pairs.push(picked[1].len())
        }
    } elif other is DataFrame {
        if other.len() != frame.len() {
            throw ValueError(f"a frame of {other.len()} rows cannot align with {frame.len()}")
        }
        for column in picked[1] { columns.push(column) }
        var missing: list[str] = []
        for (at, name) in picked[0].enumerate() {
            if not other.has(name) {
                if not drop { missing.push(name) }
                continue
            }
            names.push(name)
            pairs.push(at)
            pairs.push(columns.len())
            columns.push(other.column(name))
        }
        for name in missing { names.push(name) }
    } else {
        throw TypeError("corrwith needs a Series or a DataFrame to correlate against")
    }
    let moments = pairwise_moments(columns, frame.len(), pairs)
    var numbers: list[float] = []
    for at in 0..names.len() {
        if (at + 1) * STATS > moments.len() {
            numbers.push(NAN)
            continue
        }
        numbers.push(pair_value(moments, at, min_periods, 1, true))
    }
    return named_series(names, Column.from_floats(numbers))
}
